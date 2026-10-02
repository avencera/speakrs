//! ResNet34 multi-mask speaker embedding on the GPU
//!
//! Computes exactly what `wespeaker-multimask-tail.onnx` and its `-b32` variant
//! compute: fbank features and per-speaker masks in, one embedding per (chunk,
//! speaker) out
//!
//! - cuDNN runs the 36 trunk convolutions. Batch norm is already folded into the
//!   exported weights; one plan per distinct convolution shape is shared by every
//!   layer of that shape, and its algorithm comes from cuDNN's heuristic ranking
//!   (see [`ConvPlanner`]), so the choice never depends on timing noise. All plans
//!   share one workspace
//! - cuDNN's fused convolution, bias, residual and ReLU call
//!   (`cudnnConvolutionBiasActivationForward`) covers every 3x3 convolution with
//!   its epilogue; on the RTX 5070 Ti it cost 0.05 to 0.3 ms more than the bare
//!   convolution at batch 32, where a separate epilogue kernel cost about 1 ms
//! - cuda-oxide kernels transpose the fbank into the stem layout, add the 1x1
//!   shortcut biases, pool the masked statistics and fill the embedding bias
//! - cuBLAS runs the 5120 -> 256 embedding layer
//!
//! [`CudaMath`] selects FP32 or TF32 for every cuDNN and cuBLAS call. Device buffers
//! live in an [`EmbeddingBatch`], allocated once per batch class and reused, which
//! can also hold a CUDA graph of the whole forward pass

mod kernels;
mod trunk;

use std::sync::Arc;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, CudaView, CudaViewMut};
use tracing::debug;

use self::kernels::{ChannelBias, EmbeddingKernels, PoolShape};
use self::trunk::{ConvLayer, STEM_SLOT, Trunk};
use super::dnn::{ConvPlan, ConvPlanner, Residual};
use super::error::{check_len, element_count};
use super::fbank::{FBANK_FRAMES, FBANK_MEL_BINS};
use super::{CudaError, CudaMath, CudaRuntime, DeviceTensor, PtxTier, SafetensorsFile, Sgemm};

/// Frames of one speaker mask, at the segmentation model's frame rate
pub const MASK_FRAMES: usize = 589;

/// Speaker masks per fbank chunk
pub const SPEAKERS_PER_CHUNK: usize = 3;

/// Size of one speaker embedding
pub const EMBEDDING_DIM: usize = 256;

/// Weight and bias of the embedding layer in the exported weights
const HEAD_WEIGHT: &str = "resnet.seg_1.weight";
const HEAD_BIAS: &str = "resnet.seg_1.bias";

/// A point in the forward pass whose activation [`EmbeddingBatch::forward_with_taps`]
/// exposes, for comparing layers against reference intermediates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTap {
    /// Stem convolution after bias and ReLU, `[b, 32, 80, 998]`
    Stem,
    /// First convolution of basic block `block` (0-based over all 16 blocks) after
    /// bias and ReLU
    Hidden {
        /// Block index
        block: usize,
    },
    /// 1x1 shortcut convolution of basic block `block` after its bias
    Shortcut {
        /// Block index
        block: usize,
    },
    /// Output of basic block `block`: second convolution, bias, residual add, ReLU
    Block {
        /// Block index
        block: usize,
    },
    /// Pooled statistics `[b * 3, 5120]`, means then standard deviations
    Pooled,
    /// Embeddings `[b * 3, 256]`
    Output,
}

/// Callback of [`EmbeddingBatch::forward_with_taps`]: receives each tapped
/// activation as a device view that is valid only during the call
pub type EmbeddingTapFn<'a> =
    dyn FnMut(EmbeddingTap, &CudaView<'_, f32>) -> Result<(), CudaError> + 'a;

/// The WeSpeaker ResNet34 multi-mask embedding model on one CUDA runtime
///
/// Holds the weights and kernels; [`Self::batch`] creates the per-batch-class
/// state that runs it. Cloning is cheap and shares the weights
#[derive(Debug, Clone)]
pub struct ResNetEmbedding(Arc<Model>);

/// Weights, kernels and precision shared by a model and its batches
#[derive(Debug)]
struct Model {
    trunk: Trunk,
    head_weight: DeviceTensor,
    head_bias: DeviceTensor,
    kernels: EmbeddingKernels,
    math: CudaMath,
}

impl ResNetEmbedding {
    /// Uploads the trunk and embedding weights and loads the kernels
    ///
    /// `weights` holds the initializers of `wespeaker-multimask-tail.onnx` under their
    /// ONNX names, with batch norm folded into each `<conv>.weight` and
    /// `<conv>.weight_bias`. `math` applies to every convolution and to the
    /// embedding GEMM
    pub fn load(
        runtime: &CudaRuntime,
        weights: &SafetensorsFile,
        math: CudaMath,
    ) -> Result<Self, CudaError> {
        let trunk = Trunk::load(runtime, weights, FBANK_MEL_BINS, FBANK_FRAMES)?;
        let pooled = 2 * pool_columns(&trunk);
        let head_weight = weights.upload(runtime, HEAD_WEIGHT, &[EMBEDDING_DIM, pooled])?;
        let head_bias = weights.upload(runtime, HEAD_BIAS, &[EMBEDDING_DIM])?;
        let kernels = EmbeddingKernels::load(runtime)?;

        Ok(Self(Arc::new(Model {
            trunk,
            head_weight,
            head_bias,
            kernels,
            math,
        })))
    }

    /// The PTX tier the embedding kernels were loaded from
    pub fn kernel_tier(&self) -> PtxTier {
        self.0.kernels.tier()
    }

    /// Allocates the device buffers and plans the convolutions for `chunks` fbank
    /// chunks per forward pass
    ///
    /// Do this once per batch class and reuse the batch for every forward pass of
    /// that size. The batch keeps a handle to this model and must run on
    /// `runtime`, whose stream owns its buffers
    pub fn batch(&self, runtime: &CudaRuntime, chunks: usize) -> Result<EmbeddingBatch, CudaError> {
        let model = &self.0;
        let stream = runtime.stream();
        let rows = chunks * SPEAKERS_PER_CHUNK;
        let columns = pool_columns(&model.trunk);

        let lens = model.trunk.buffer_lens();
        let planner = ConvPlanner::new(runtime)?;
        let plans = model
            .trunk
            .shapes()
            .iter()
            .map(|shape| planner.plan(shape.conv(chunks, model.math)))
            .collect::<Result<Vec<_>, _>>()?;
        let workspace_bytes = plans
            .iter()
            .map(ConvPlan::workspace_bytes)
            .max()
            .unwrap_or(0);
        debug!(
            chunks,
            math = ?model.math,
            plans = plans.len(),
            ?lens,
            workspace_bytes,
            "Planned CUDA embedding batch"
        );

        let stem_len = element_count(
            "embedding stem input",
            &[chunks, FBANK_MEL_BINS, FBANK_FRAMES],
        )?;
        Ok(EmbeddingBatch {
            model: Arc::clone(model),
            chunks,
            fbank: DeviceTensor::zeros(stream, &[chunks, FBANK_FRAMES, FBANK_MEL_BINS])?,
            masks: DeviceTensor::zeros(stream, &[rows, MASK_FRAMES])?,
            stem_input: stream.alloc_zeros(stem_len)?,
            // at least one element, because CUDA cannot allocate zero bytes
            trunk: [
                stream.alloc_zeros((chunks * lens.trunk[0]).max(1))?,
                stream.alloc_zeros((chunks * lens.trunk[1]).max(1))?,
            ],
            hidden: stream.alloc_zeros((chunks * lens.hidden).max(1))?,
            shortcut: stream.alloc_zeros((chunks * lens.shortcut).max(1))?,
            pooled: stream.alloc_zeros(rows * 2 * columns)?,
            output: DeviceTensor::zeros(stream, &[rows, EMBEDDING_DIM])?,
            plans,
            workspace: stream.alloc_zeros(workspace_bytes.max(1))?,
            graph: None,
        })
    }
}

/// Device buffers, convolution plans and an optional CUDA graph for one batch class
/// of a [`ResNetEmbedding`]
///
/// Write the fbank `[chunks, 998, 80]` and masks `[chunks * 3, 589]` into
/// [`Self::fbank_mut`] and [`Self::masks_mut`], run [`Self::forward`], then read
/// [`Self::output`] `[chunks * 3, 256]`. Masks are chunk-major, then
/// speaker-major. Run a batch only on the runtime that created it
#[derive(Debug)]
pub struct EmbeddingBatch {
    /// the model whose weights, kernels and precision this batch runs
    model: Arc<Model>,
    chunks: usize,
    fbank: DeviceTensor,
    masks: DeviceTensor,
    stem_input: CudaSlice<f32>,
    /// block inputs and outputs; see [`trunk::BasicBlock`] for which holds what
    trunk: [CudaSlice<f32>; 2],
    /// the activation between a block's two convolutions
    hidden: CudaSlice<f32>,
    /// the shortcut convolution's output, the residual of a downsampling block
    shortcut: CudaSlice<f32>,
    pooled: CudaSlice<f32>,
    output: DeviceTensor,
    plans: Vec<ConvPlan>,
    /// cuDNN workspace shared by every plan, sized for the largest
    workspace: CudaSlice<u8>,
    graph: Option<ForwardGraph>,
}

/// A captured forward pass; cudarc's graph type has no `Debug`
struct ForwardGraph(CudaGraph);

impl std::fmt::Debug for ForwardGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ForwardGraph")
    }
}

impl EmbeddingBatch {
    /// Runs the forward pass on the batch's fbank and masks into its output
    ///
    /// Replays the batch's CUDA graph when [`Self::capture_graph`] recorded one.
    /// Work is queued on the runtime's stream; read the output after it, for
    /// example with [`EmbeddingBatch::download_output`]
    pub fn forward(&mut self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        if let Some(ForwardGraph(graph)) = &self.graph {
            graph.launch()?;
            return Ok(());
        }

        self.run(runtime, &mut |_, _| Ok(()))
    }

    #[cfg(test)]
    /// Runs the forward pass eagerly and calls `tap` with every intermediate
    /// activation listed in [`EmbeddingTap`], in execution order
    ///
    /// The view is only valid during the call; download it there if needed. This
    /// never replays a captured graph
    pub fn forward_with_taps(
        &mut self,
        runtime: &CudaRuntime,
        tap: &mut EmbeddingTapFn<'_>,
    ) -> Result<(), CudaError> {
        self.run(runtime, tap)
    }

    /// Records the batch's forward pass as a CUDA graph, which [`Self::forward`]
    /// then replays instead of issuing each launch
    ///
    /// The graph bakes in the batch's buffer addresses and its model's weights,
    /// both of which the batch keeps alive. Runs one eager pass first so cuDNN and
    /// cuBLAS finish their lazy setup outside the capture
    pub fn capture_graph(&mut self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        self.graph = None;
        self.run(runtime, &mut |_, _| Ok(()))?;
        runtime.synchronize()?;

        let context = runtime.context();
        let tracking = context.is_event_tracking();
        // cudarc records and waits on per-buffer events for cross-stream safety;
        // inside a capture those would become graph nodes or waits on events from
        // outside the capture, and every buffer here lives on this one stream
        // SAFETY: the runtime issues all work on its single stream, so stream order
        // alone keeps these buffers consistent while tracking is off
        unsafe { context.disable_event_tracking() };

        let stream = runtime.stream();
        let captured = stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(CudaError::from)
            .and_then(|()| {
                let run = self.run(runtime, &mut |_, _| Ok(()));
                // always end the capture, even after a failed launch, so the stream
                // leaves capture mode
                let graph = stream.end_capture(CUgraphInstantiate_flags(0));
                run?;
                Ok(graph?)
            });

        if tracking {
            // SAFETY: restores the tracking state from before the capture
            unsafe { context.enable_event_tracking() };
        }

        self.graph = captured?.map(ForwardGraph);
        debug!(
            chunks = self.chunks,
            captured = self.graph.is_some(),
            "Captured CUDA embedding graph"
        );
        Ok(())
    }

    #[cfg(test)]
    /// Copies host fbank and masks into the batch, runs the forward pass and
    /// downloads the embeddings
    ///
    /// `fbank` is `[chunks, 998, 80]` and `masks` is `[chunks * 3, 589]`; the result
    /// is `[chunks * 3, 256]`, all row-major
    pub fn embed(
        &mut self,
        runtime: &CudaRuntime,
        fbank: &[f32],
        masks: &[f32],
    ) -> Result<Vec<f32>, CudaError> {
        let stream = runtime.stream();
        self.fbank.copy_from_host(stream, fbank)?;
        self.masks.copy_from_host(stream, masks)?;
        self.forward(runtime)?;
        self.download_output(runtime)
    }

    fn run(
        &mut self,
        runtime: &CudaRuntime,
        tap: &mut EmbeddingTapFn<'_>,
    ) -> Result<(), CudaError> {
        let EmbeddingBatch {
            model,
            chunks,
            fbank,
            masks,
            stem_input,
            trunk,
            hidden,
            shortcut,
            pooled,
            output,
            plans,
            workspace,
            ..
        } = self;
        let model = &**model;
        let chunks = *chunks;
        let math = model.math;
        let mut convs = Convs {
            runtime,
            kernels: &model.kernels,
            plans,
            workspace,
            chunks,
            math,
        };

        let mut stem_input = stem_input.as_view_mut();
        model.kernels.fbank_transpose(
            runtime,
            &fbank.data().as_view(),
            FBANK_FRAMES,
            FBANK_MEL_BINS,
            &mut stem_input,
        )?;

        let stem = &model.trunk.stem;
        let stem_len = chunks * stem.output_len();
        // the trunk buffer the stem does not write stands in as its unused
        // residual operand
        let (scratch_buffer, stem_buffer) = read_write(trunk, 1 - STEM_SLOT);
        let mut stem_out = stem_buffer.slice_mut(..stem_len);
        let scratch = scratch_buffer.slice(..stem_len);
        convs.conv_bias_relu(
            stem,
            &stem_input.as_view(),
            Residual::None { scratch: &scratch },
            &mut stem_out,
        )?;
        tap(EmbeddingTap::Stem, &stem_out.as_view())?;

        for (index, block) in model.trunk.blocks.iter().enumerate() {
            let input_len = element_count(
                "embedding block input",
                &block.conv1.conv(chunks, math).input_shape(),
            )?;
            let output_len = chunks * block.conv2.output_len();
            let (input_buffer, output_buffer) = read_write(trunk, block.input_slot);
            let input = input_buffer.slice(..input_len);
            let mut block_out = output_buffer.slice_mut(..output_len);

            // the block output buffer is free until the second convolution, so it
            // stands in as the first convolution's unused residual operand
            let mut hidden_out = hidden.slice_mut(..output_len);
            let scratch = block_out.as_view();
            convs.conv_bias_relu(
                &block.conv1,
                &input,
                Residual::None { scratch: &scratch },
                &mut hidden_out,
            )?;
            tap(EmbeddingTap::Hidden { block: index }, &hidden_out.as_view())?;

            // only read for a block with a shortcut, where the buffer holds the whole
            // output; the bound keeps the view valid for the other blocks
            let mut shortcut_out = shortcut.slice_mut(..output_len.min(shortcut.len()));
            let residual = match &block.shortcut {
                Some(layer) => {
                    convs.conv(layer, &input, &mut shortcut_out)?;
                    convs.bias(layer, &mut shortcut_out)?;
                    tap(
                        EmbeddingTap::Shortcut { block: index },
                        &shortcut_out.as_view(),
                    )?;
                    shortcut_out.as_view()
                }
                None => input.slice(..output_len),
            };
            convs.conv_bias_relu(
                &block.conv2,
                &hidden_out.as_view(),
                Residual::Add(&residual),
                &mut block_out,
            )?;
            tap(EmbeddingTap::Block { block: index }, &block_out.as_view())?;
        }

        let columns = pool_columns(&model.trunk);
        let [_, _, frames] = model.trunk.output_shape();
        let pool = PoolShape {
            chunks,
            speakers: SPEAKERS_PER_CHUNK,
            columns,
            frames,
            mask_frames: MASK_FRAMES,
        };
        let features = trunk[model.trunk.output_slot()].slice(..chunks * columns * frames);
        let mut pooled = pooled.as_view_mut();
        model.kernels.mask_pool(
            runtime,
            &features,
            &masks.data().as_view(),
            pool,
            &mut pooled,
        )?;
        tap(EmbeddingTap::Pooled, &pooled.as_view())?;

        let rows = chunks * SPEAKERS_PER_CHUNK;
        let mut embeddings = output.data_mut().as_view_mut();
        model.kernels.broadcast_rows(
            runtime,
            &model.head_bias.data().as_view(),
            &mut embeddings,
        )?;
        let gemm = Sgemm {
            b_transposed: true,
            beta: 1.0,
            math,
            ..Sgemm::new(rows, EMBEDDING_DIM, 2 * columns)
        };
        runtime.sgemm(gemm, &pooled, model.head_weight.data(), &mut embeddings)?;
        tap(EmbeddingTap::Output, &embeddings.as_view())?;

        Ok(())
    }

    #[cfg(test)]
    /// Embeddings per forward pass, `chunks * 3`
    pub fn rows(&self) -> usize {
        self.chunks * SPEAKERS_PER_CHUNK
    }

    #[cfg(test)]
    /// Distinct convolution shapes, one cuDNN plan each
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    #[cfg(test)]
    /// Whether [`Self::forward`] replays a captured CUDA graph
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// The fbank input `[chunks, 998, 80]`, for the fbank stage to write into
    pub fn fbank_mut(&mut self) -> &mut DeviceTensor {
        &mut self.fbank
    }

    /// The speaker masks `[chunks * 3, 589]`
    pub fn masks_mut(&mut self) -> &mut DeviceTensor {
        &mut self.masks
    }

    /// Copies the embeddings to the host; this waits for the runtime's stream
    pub fn download_output(&self, runtime: &CudaRuntime) -> Result<Vec<f32>, CudaError> {
        self.output.download(runtime.stream())
    }

    #[cfg(test)]
    /// Bytes of the shared cuDNN workspace
    pub fn workspace_bytes(&self) -> usize {
        self.workspace.len()
    }

    #[cfg(test)]
    /// Bytes held by this batch's activation, input and output buffers, excluding
    /// the cuDNN workspace
    pub fn buffer_bytes(&self) -> usize {
        let floats = self.fbank.len()
            + self.masks.len()
            + self.stem_input.len()
            + self.trunk.iter().map(CudaSlice::len).sum::<usize>()
            + self.hidden.len()
            + self.shortcut.len()
            + self.pooled.len()
            + self.output.len();
        floats * size_of::<f32>()
    }
}

/// Convolution plus epilogue launches for one forward pass
struct Convs<'a> {
    runtime: &'a CudaRuntime,
    kernels: &'a EmbeddingKernels,
    plans: &'a [ConvPlan],
    workspace: &'a mut CudaSlice<u8>,
    chunks: usize,
    math: CudaMath,
}

impl<'a> Convs<'a> {
    /// The plan for this layer's shape
    fn plan(&self, layer: &ConvLayer) -> &'a ConvPlan {
        // `run` checked that the batch has one plan per trunk shape, and every
        // layer's slot indexes those shapes
        let plan = &self.plans[layer.plan_slot()];
        debug_assert_eq!(*plan.spec(), layer.conv(self.chunks, self.math));
        plan
    }

    /// `y = conv(x, layer.weight)`
    fn conv(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let plan = self.plan(layer);
        plan.forward(
            &mut self.workspace.as_view_mut(),
            x,
            &layer.weight().data().as_view(),
            y,
        )
    }

    /// `y = relu(conv(x, layer.weight) + residual + layer.bias)` in one cuDNN call
    fn conv_bias_relu(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        residual: Residual<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let plan = self.plan(layer);
        plan.forward_bias_relu(
            &mut self.workspace.as_view_mut(),
            x,
            &layer.weight().data().as_view(),
            &layer.bias().data().as_view(),
            residual,
            y,
        )
    }

    /// `y = y + layer.bias` in place, for the shortcut convolutions
    fn bias(&self, layer: &ConvLayer, y: &mut CudaViewMut<'_, f32>) -> Result<(), CudaError> {
        check_len(
            "embedding conv output",
            self.chunks * layer.output_len(),
            y.len(),
        )?;
        let [h, w] = layer.output();
        let layout = ChannelBias {
            channels: layer.out_channels(),
            plane: h * w,
        };
        self.kernels
            .bias(self.runtime, &layer.bias().data().as_view(), layout, y)
    }
}

/// The trunk buffer `read` and, mutably, the other one
fn read_write(
    buffers: &mut [CudaSlice<f32>; 2],
    read: usize,
) -> (&CudaSlice<f32>, &mut CudaSlice<f32>) {
    let [first, second] = buffers;
    match read {
        0 => (first, second),
        _ => (second, first),
    }
}

/// Channels times frequency bins of the trunk output: the statistics pooling
/// flattens those into one column per (channel, bin)
fn pool_columns(trunk: &Trunk) -> usize {
    let [channels, bins, _] = trunk.output_shape();
    channels * bins
}
