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
//! live in an [`EmbeddingBatch`], which can also hold a CUDA graph of the whole
//! forward pass. Serial class plans share one fixed-address activation allocation

mod batch_class;
pub(crate) use batch_class::EmbeddingBatchClass;
mod dispatch;
mod kernels;
#[cfg(test)]
pub(super) use kernels::REQUIRED_KERNELS;
mod trunk;

use std::sync::{Arc, Mutex};

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, CudaView, CudaViewMut};
use tracing::debug;

use self::dispatch::Plan;
use self::kernels::{ChannelBias, EmbeddingKernels, PoolShape};
use self::trunk::{ConvLayer, STEM_SLOT, Trunk};
use super::candidate::{DenseSite, DenseSpec};
use super::dense::DensePlan;
use super::error::{check_len, element_count};
use super::fbank::{FBANK_FRAMES, FBANK_MEL_BINS};
use super::geometry::Residual;
use super::implementation::{AreaTarget, BoundaryId, LibraryNeed, Selected, plan_selection};
use super::{CudaError, CudaMath, CudaRuntime, DeviceTensor, PtxTier, SafetensorsFile, Sgemm};
use super::{CudaLibrary, KernelModule};

/// Frames of one speaker mask, at the segmentation model's frame rate
pub const MASK_FRAMES: usize = 589;

/// Speaker masks per fbank chunk
pub const SPEAKERS_PER_CHUNK: usize = 3;

/// Size of one speaker embedding
pub const EMBEDDING_DIM: usize = 256;

/// Weight and bias of the embedding layer in the exported weights
const HEAD_WEIGHT: &str = "resnet.seg_1.weight";
const HEAD_BIAS: &str = "resnet.seg_1.bias";
/// The embedding head GEMM
const HEAD: BoundaryId = BoundaryId::named("resnet.seg_1");

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
/// Holds the weights and kernels; class plans create the per-batch state that
/// runs it. Cloning is cheap and shares the weights
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
        let target = AreaTarget::for_area(runtime, KernelModule::Resnet)?;
        let mut needs = Vec::new();
        // retain the established load checks; intermediate classes are resolved on first use
        for class in [EmbeddingBatchClass::One, EmbeddingBatchClass::ThirtyTwo] {
            let batch = class.chunks();
            for (layer, _) in trunk.layers() {
                if matches!(
                    plan_selection(
                        runtime,
                        layer.boundary(),
                        batch,
                        math,
                        #[cfg(all(test, feature = "_cuda-libraries"))]
                        None,
                    )?,
                    Selected::Library
                ) {
                    needs.push(LibraryNeed::new(
                        layer.boundary(),
                        batch,
                        math,
                        target,
                        CudaLibrary::Cudnn,
                    ));
                }
            }
            if matches!(
                plan_selection(
                    runtime,
                    HEAD,
                    EmbeddingHead::chunks_per_pass(batch),
                    math,
                    #[cfg(all(test, feature = "_cuda-libraries"))]
                    None
                )?,
                Selected::Library
            ) {
                needs.push(LibraryNeed::new(
                    HEAD,
                    EmbeddingHead::chunks_per_pass(batch),
                    math,
                    AreaTarget::for_area(runtime, KernelModule::Embedding)?,
                    CudaLibrary::Cublas,
                ));
            }
        }
        if super::driver_only() {
            for need in &needs {
                need.prepare(runtime)?;
            }
        }
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

    /// Allocate fixed-address trunk storage for serial batches on this runtime
    pub(crate) fn activations(
        &self,
        runtime: &CudaRuntime,
        chunks: usize,
    ) -> Result<SharedEmbeddingActivations, CudaError> {
        let lens = self.0.trunk.buffer_lens();
        let stream = runtime.stream();
        Ok(SharedEmbeddingActivations(Arc::new(Mutex::new(
            EmbeddingActivations {
                trunk: [
                    stream.alloc_zeros((chunks * lens.trunk[0]).max(1))?,
                    stream.alloc_zeros((chunks * lens.trunk[1]).max(1))?,
                ],
                hidden: stream.alloc_zeros((chunks * lens.hidden).max(1))?,
                shortcut: stream.alloc_zeros((chunks * lens.shortcut).max(1))?,
            },
        ))))
    }

    /// Plan one class using storage kept alive by every graph that refers to it
    pub(crate) fn batch_with_activations(
        &self,
        runtime: &CudaRuntime,
        chunks: usize,
        activations: SharedEmbeddingActivations,
    ) -> Result<EmbeddingBatch, CudaError> {
        let model = &self.0;
        let stream = runtime.stream();
        let rows = chunks * SPEAKERS_PER_CHUNK;
        let columns = pool_columns(&model.trunk);

        let lens = model.trunk.buffer_lens();
        {
            let buffers = activations.lock()?;
            for (expected, actual) in [
                (chunks * lens.trunk[0], buffers.trunk[0].len()),
                (chunks * lens.trunk[1], buffers.trunk[1].len()),
                (chunks * lens.hidden, buffers.hidden.len()),
                (chunks * lens.shortcut, buffers.shortcut.len()),
            ] {
                if actual < expected {
                    return Err(CudaError::BufferLength {
                        context: "embedding shared activation capacity",
                        expected,
                        actual,
                    });
                }
            }
        }

        let plans = dispatch::plan_layers(runtime, &model.trunk, chunks, model.math)?;
        let workspace_bytes = plans
            .iter()
            .map(|(_, plan)| plan.workspace_bytes())
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
            head: EmbeddingHead::new(runtime, model, chunks)?,
            fbank: DeviceTensor::zeros(stream, &[chunks, FBANK_FRAMES, FBANK_MEL_BINS])?,
            masks: DeviceTensor::zeros(stream, &[rows, MASK_FRAMES])?,
            stem_input: stream.alloc_zeros(stem_len)?,
            activations,
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
    head: EmbeddingHead,
    fbank: DeviceTensor,
    masks: DeviceTensor,
    stem_input: CudaSlice<f32>,
    activations: SharedEmbeddingActivations,
    pooled: CudaSlice<f32>,
    output: DeviceTensor,
    plans: Vec<(String, Plan)>,
    /// cuDNN workspace shared by every plan, sized for the largest
    workspace: CudaSlice<u8>,
    graph: Option<ForwardGraph>,
}

/// Fixed-address activations shared only by serial class plans on one runtime
///
/// Each batch retains this owner, so dropping a cache or a different class cannot
/// free captured pointers. The lock prevents overlapping mutable device views;
/// all eager and graph launches use the same stream and its execution order
#[derive(Debug, Clone)]
pub(crate) struct SharedEmbeddingActivations(Arc<Mutex<EmbeddingActivations>>);

#[derive(Debug)]
struct EmbeddingActivations {
    trunk: [CudaSlice<f32>; 2],
    hidden: CudaSlice<f32>,
    shortcut: CudaSlice<f32>,
}

impl SharedEmbeddingActivations {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, EmbeddingActivations>, CudaError> {
        self.0.lock().map_err(|_| CudaError::Unsupported {
            context: "embedding activations",
            reason: "a previous activation launch panicked".to_owned(),
        })
    }
}

/// Projection plans compose the compiled head classes without padding the trunk
#[derive(Debug)]
struct EmbeddingHead {
    plan: DensePlan,
    chunks_per_pass: usize,
    columns: usize,
}

impl EmbeddingHead {
    const fn chunks_per_pass(chunks: usize) -> usize {
        if chunks == 32 { 32 } else { 1 }
    }

    fn measurement_batch(chunks: usize) -> Option<usize> {
        (chunks == Self::chunks_per_pass(chunks)).then_some(chunks)
    }

    fn new(runtime: &CudaRuntime, model: &Model, chunks: usize) -> Result<Self, CudaError> {
        // the head kernels bake in 1 or 32 chunks; intermediate trunks reuse the
        // single-chunk head so its reduction order and precision stay unchanged
        let chunks_per_pass = Self::chunks_per_pass(chunks);
        let spec =
            DenseSpec::new(DenseSite::Embedding, chunks_per_pass, model.math).map_err(|error| {
                CudaError::Unsupported {
                    context: "embedding projection",
                    reason: error.to_string(),
                }
            })?;
        Ok(Self {
            plan: DensePlan::new(
                runtime,
                spec,
                SPEAKERS_PER_CHUNK,
                model.head_weight.data(),
                model.head_bias.data(),
            )?,
            chunks_per_pass,
            columns: 2 * pool_columns(&model.trunk),
        })
    }

    fn enqueue(
        &self,
        runtime: &CudaRuntime,
        input: &CudaView<'_, f32>,
        output: &mut cudarc::driver::CudaViewMut<'_, f32>,
        mut library: impl FnMut(
            usize,
            &CudaView<'_, f32>,
            &mut cudarc::driver::CudaViewMut<'_, f32>,
        ) -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        let rows = self.chunks_per_pass * SPEAKERS_PER_CHUNK;
        let input_step = rows * self.columns;
        let output_step = rows * EMBEDDING_DIM;
        for (pass, start) in (0..input.len()).step_by(input_step).enumerate() {
            let input = input.slice(start..start + input_step);
            let output_start = pass * output_step;
            let mut output = output.slice_mut(output_start..output_start + output_step);
            self.plan.enqueue(runtime, &input, &mut output, |output| {
                library(self.chunks_per_pass, &input, output)
            })?;
        }
        Ok(())
    }
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
            let _activations = self.activations.lock()?;
            graph.launch()?;
            return Ok(());
        }

        self.run(runtime, &mut |_, _| Ok(()))
    }

    /// Records the batch's forward pass as a CUDA graph, which [`Self::forward`]
    /// then replays instead of issuing each launch
    ///
    /// The graph bakes in the batch's buffer addresses and its model's weights,
    /// both of which the batch keeps alive. Library builds run one eager pass first
    /// so cuDNN and cuBLAS finish their lazy setup outside the capture. Driver-only
    /// plans are prepared by batch construction and do not need this extra pass
    pub fn capture_graph(&mut self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        self.graph = None;
        if !super::driver_only() {
            self.run(runtime, &mut |_, _| Ok(()))?;
        }
        // finish weight packing and prior buffer work before capture drops their events
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

    fn run(
        &mut self,
        runtime: &CudaRuntime,
        tap: &mut EmbeddingTapFn<'_>,
    ) -> Result<(), CudaError> {
        let EmbeddingBatch {
            model,
            chunks,
            head,
            fbank,
            masks,
            stem_input,
            activations,
            pooled,
            output,
            plans,
            workspace,
            ..
        } = self;
        let mut activations = activations.lock()?;
        let EmbeddingActivations {
            trunk,
            hidden,
            shortcut,
        } = &mut *activations;
        let model = &**model;
        #[cfg(not(feature = "_cuda-libraries"))]
        let _ = workspace;
        let chunks = *chunks;
        let math = model.math;
        let mut convs = Convs {
            runtime,
            kernels: &model.kernels,
            plans,
            #[cfg(feature = "_cuda-libraries")]
            workspace,
            chunks,
            math: model.math,
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
        #[cfg(feature = "_cuda-libraries")]
        let scratch = scratch_buffer.slice(..stem_len);
        #[cfg(not(feature = "_cuda-libraries"))]
        let _ = scratch_buffer;
        convs.conv_bias_relu(
            stem,
            &stem_input.as_view(),
            Residual::None {
                #[cfg(feature = "_cuda-libraries")]
                scratch: &scratch,
            },
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
            #[cfg(feature = "_cuda-libraries")]
            let scratch = block_out.as_view();
            convs.conv_bias_relu(
                &block.conv1,
                &input,
                Residual::None {
                    #[cfg(feature = "_cuda-libraries")]
                    scratch: &scratch,
                },
                &mut hidden_out,
            )?;
            tap(EmbeddingTap::Hidden { block: index }, &hidden_out.as_view())?;

            // only read for a block with a shortcut, where the buffer holds the whole
            // output; the bound keeps the view valid for the other blocks
            let mut shortcut_out = shortcut.slice_mut(..output_len.min(shortcut.len()));
            let residual = match &block.shortcut {
                Some(layer) => {
                    convs.shortcut(layer, &input, &mut shortcut_out)?;
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

        let mut embeddings = output.data_mut().as_view_mut();

        let mut enqueue_head = || {
            head.enqueue(
                runtime,
                &pooled.as_view(),
                &mut embeddings,
                |head_chunks, input, output| {
                    let rows = head_chunks * SPEAKERS_PER_CHUNK;
                    let gemm = Sgemm {
                        b_transposed: true,
                        beta: 1.0,
                        math,
                        ..Sgemm::new(rows, EMBEDDING_DIM, 2 * columns)
                    };

                    model.kernels.broadcast_rows(
                        runtime,
                        &model.head_bias.data().as_view(),
                        output,
                    )?;
                    runtime.sgemm(gemm, input, model.head_weight.data(), output)?;
                    Ok(())
                },
            )
        };
        match EmbeddingHead::measurement_batch(chunks) {
            Some(batch) => runtime.record_boundary(HEAD, batch, math, enqueue_head)?,
            None => enqueue_head()?,
        }
        tap(EmbeddingTap::Output, &embeddings.as_view())?;

        Ok(())
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
}

/// Convolution plus epilogue launches for one forward pass
struct Convs<'a> {
    math: CudaMath,
    runtime: &'a CudaRuntime,
    kernels: &'a EmbeddingKernels,
    plans: &'a [(String, Plan)],
    #[cfg(feature = "_cuda-libraries")]
    workspace: &'a mut CudaSlice<u8>,
    chunks: usize,
}

impl<'a> Convs<'a> {
    /// The owner selected when this batch was constructed
    fn plan(&self, layer: &ConvLayer) -> Result<&'a Plan, CudaError> {
        self.plans
            .iter()
            .find(|(name, _)| name == layer.name())
            .map(|(_, plan)| plan)
            .ok_or_else(|| CudaError::Unsupported {
                context: "embedding plan",
                reason: format!("no plan for {}", layer.name()),
            })
    }

    /// `y = conv(x, layer.weight)`
    fn conv(
        &mut self,
        layer: &ConvLayer,
        x: &CudaView<'_, f32>,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        #[cfg(feature = "_cuda-libraries")]
        {
            let plan = self.plan(layer)?.library()?;
            plan.forward(
                &mut self.workspace.as_view_mut(),
                x,
                &layer.weight().data().as_view(),
                y,
            )
        }
        #[cfg(not(feature = "_cuda-libraries"))]
        {
            let _ = (layer, x, y);
            Err(CudaError::Unsupported {
                context: "embedding shortcut",
                reason: "no qualified driver implementation".to_owned(),
            })
        }
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

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests;
