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
//!
//! TF32 mode may run same-channel trunk layers on FP16 tiles, which saturate operands
//! above [`FP16_OPERAND_LIMIT`]. Layers whose weights exceed it never select them. An
//! activation that exceeds it sets a word that follows the embeddings in the output
//! buffer, so the download that already ends every batch reads it; the batch is then
//! recomputed with plans selected without FP16 tiles

mod batch_class;
pub(crate) use batch_class::EmbeddingBatchClass;
mod dispatch;
mod kernels;
#[cfg(test)]
pub(super) use kernels::REQUIRED_KERNELS;
mod storage;
mod trunk;

use std::sync::{Arc, Mutex, Once};

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, CudaView, CudaViewMut};
use tracing::{debug, warn};

use self::dispatch::Plan;
use self::kernels::{ChannelBias, EmbeddingKernels, PoolShape};
use self::storage::{ActivationStorage, Captured};
use self::trunk::{BasicBlock, ConvLayer, STEM_SLOT, Trunk};
use super::candidate::{DenseSite, DenseSpec, FP16_OPERAND_LIMIT, Fp16Policy, HalfIo};
use super::dense::DensePlan;
use super::error::{check_len, element_count};
use super::fbank::{FBANK_FRAMES, FBANK_MEL_BINS};
use super::geometry::Residual;
use super::implementation::{
    AreaTarget, BoundaryId, LibraryNeed, Selected, plan_selection, plan_selection_with,
};
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

/// Words after the embeddings in a batch's output buffer: the out-of-range word, which
/// FP16 trunk tiles set nonzero when an activation saturates
const RANGE_WORDS: usize = 1;

/// The first batch recomputed without FP16 tiles in this process logs a warning
static FP16_RANGE_WARNING: Once = Once::new();

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
                    plan_selection_with(
                        runtime,
                        layer.boundary(),
                        batch,
                        math,
                        layer.fp16(),
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

    /// Allocate one window, or all classes for a multi-window session
    pub(crate) fn activations(
        &self,
        runtime: &CudaRuntime,
        chunks: usize,
    ) -> Result<SharedEmbeddingActivations, CudaError> {
        let storage = ActivationStorage::allocate(chunks, |capacity| {
            self.allocate_activations(runtime, capacity)
        })?;
        Ok(SharedEmbeddingActivations(Arc::new(Mutex::new(storage))))
    }

    /// Grow shared storage only when a larger class first needs it
    pub(crate) fn grow_activations(
        &self,
        runtime: &CudaRuntime,
        activations: &SharedEmbeddingActivations,
        chunks: usize,
    ) -> Result<(), CudaError> {
        activations.lock()?.grow(chunks, |capacity| {
            // captured work may still use old pointers after the host-side lock is released
            runtime.synchronize()?;
            self.allocate_activations(runtime, capacity)
        })
    }

    fn allocate_activations(
        &self,
        runtime: &CudaRuntime,
        chunks: usize,
    ) -> Result<EmbeddingActivations, CudaError> {
        let lens = self.0.trunk.buffer_lens();
        let stream = runtime.stream();
        Ok(EmbeddingActivations {
            trunk: [
                stream.alloc_zeros((chunks * lens.trunk[0]).max(1))?,
                stream.alloc_zeros((chunks * lens.trunk[1]).max(1))?,
            ],
            hidden: stream.alloc_zeros((chunks * lens.hidden).max(1))?,
            shortcut: stream.alloc_zeros((chunks * lens.shortcut).max(1))?,
        })
    }

    /// Plan one class using storage whose generation guards its captured pointers
    pub(crate) fn batch_with_activations(
        &self,
        runtime: &CudaRuntime,
        chunks: usize,
        activations: SharedEmbeddingActivations,
    ) -> Result<EmbeddingBatch, CudaError> {
        self.batch_with_policy(runtime, chunks, activations, Fp16Policy::Allowed)
    }

    /// Plan a class with the activation owner and an explicit FP16 policy
    fn batch_with_policy(
        &self,
        runtime: &CudaRuntime,
        chunks: usize,
        activations: SharedEmbeddingActivations,
        fp16: Fp16Policy,
    ) -> Result<EmbeddingBatch, CudaError> {
        let model = &self.0;
        let stream = runtime.stream();
        let rows = chunks * SPEAKERS_PER_CHUNK;
        let columns = pool_columns(&model.trunk);

        let lens = model.trunk.buffer_lens();
        {
            let storage = activations.lock()?;
            let buffers = storage.buffers();
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

        let plans = dispatch::plan_layers(runtime, &model.trunk, chunks, model.math, fp16)?;
        let workspace_bytes = dispatch::workspace_bytes(&plans);
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
            output: stream.alloc_zeros(rows * EMBEDDING_DIM + RANGE_WORDS)?,
            plans,
            workspace: stream.alloc_zeros(workspace_bytes.max(1))?,
            fallback: None,
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
    /// embeddings `[chunks * 3, 256]`, then the out-of-range word of `RANGE_WORDS`
    output: CudaSlice<f32>,
    plans: Vec<(String, Plan)>,
    /// cuDNN workspace shared by every plan, sized for the largest
    workspace: CudaSlice<u8>,
    /// plans without FP16 tiles, built when an activation first saturates one
    fallback: Option<Box<Fallback>>,
    graph: Option<ForwardGraph>,
}

/// The plans a batch recomputes with after an FP16 tile saturated an activation
#[derive(Debug)]
struct Fallback {
    plans: Vec<(String, Plan)>,
    workspace: CudaSlice<u8>,
}

impl Fallback {
    /// Selects every trunk layer with FP16 tiles excluded; runs outside graph capture
    fn new(runtime: &CudaRuntime, model: &Model, chunks: usize) -> Result<Self, CudaError> {
        let plans = dispatch::plan_layers(
            runtime,
            &model.trunk,
            chunks,
            model.math,
            Fp16Policy::Excluded,
        )?;
        let workspace_bytes = dispatch::workspace_bytes(&plans);
        debug!(
            chunks,
            workspace_bytes, "Planned CUDA embedding batch without FP16 tiles"
        );
        Ok(Self {
            plans,
            workspace: runtime.stream().alloc_zeros(workspace_bytes.max(1))?,
        })
    }
}

/// Which plans a forward pass runs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanSet {
    /// The plans selected when the batch was constructed
    Selected,
    /// The plans without FP16 tiles
    Fallback,
}

/// The form of the trunk activations between residual blocks' convolutions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivationForm {
    /// FP32 everywhere, which the [`EmbeddingTap`]s report
    #[cfg(all(test, feature = "_cuda-libraries"))]
    Fp32,
    /// Half where the plans on both sides have [`HalfIo`] launches: a block's hidden
    /// activation when both its convolutions do, and its output when its second
    /// convolution and both of the next block's do, unless that block has a shortcut
    Half,
}

/// Whether a residual block's hidden activation and output are half tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockForm {
    hidden: bool,
    output: bool,
}

impl BlockForm {
    /// The forms `form` gives `blocks` under `plans`
    ///
    /// A block's hidden activation is half when both its convolutions have [`HalfIo`]
    /// launches. Its output is half when its second convolution and both of the next
    /// block's have them, since the next block stages the output and adds it back as
    /// its residual, unless the next block has a shortcut, which reads it as FP32
    fn plan(blocks: &[BasicBlock], plans: &[(String, Plan)], form: ActivationForm) -> Vec<Self> {
        let half_io = |layer: &ConvLayer| {
            form == ActivationForm::Half
                && plans.iter().any(|(name, plan)| {
                    name == layer.name()
                        && matches!(plan, Plan::Wideconv(plan) if plan.has_half_io())
                })
        };
        let hidden: Vec<bool> = blocks
            .iter()
            .map(|block| half_io(&block.conv1) && half_io(&block.conv2))
            .collect();

        blocks
            .iter()
            .enumerate()
            .map(|(index, block)| Self {
                hidden: hidden[index],
                output: blocks.get(index + 1).is_some_and(|next| {
                    hidden[index + 1] && next.shortcut.is_none() && half_io(&block.conv2)
                }),
            })
            .collect()
    }
}

/// Growing activation storage shared only by serial class plans on one runtime
///
/// A successful growth changes the generation; each batch must recapture a graph
/// before it can replay with the new pointers. The lock covers generation checks,
/// capture and launch, preventing storage replacement between these steps
#[derive(Debug, Clone)]
pub(crate) struct SharedEmbeddingActivations(Arc<Mutex<ActivationStorage<EmbeddingActivations>>>);

#[derive(Debug)]
struct EmbeddingActivations {
    trunk: [CudaSlice<f32>; 2],
    hidden: CudaSlice<f32>,
    shortcut: CudaSlice<f32>,
}

impl SharedEmbeddingActivations {
    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, ActivationStorage<EmbeddingActivations>>, CudaError> {
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
struct ForwardGraph(Captured<CudaGraph>);

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
        let activations = self.activations.clone();
        let mut storage = activations.lock()?;
        if self
            .graph
            .as_ref()
            .is_some_and(|ForwardGraph(graph)| graph.current(&storage).is_none())
        {
            self.capture_graph_in_storage(runtime, &mut storage)?;
            debug!(
                target: "speakrs::inference::cuda::embedding::storage",
                chunks = self.chunks,
                captured = self.graph.is_some(),
                "Recaptured CUDA embedding graph for new storage generation"
            );
        }

        if let Some(ForwardGraph(graph)) = &self.graph {
            // the lock keeps this generation current through launch
            graph
                .current(&storage)
                .expect("captured current storage")
                .launch()?;
            return Ok(());
        }

        self.run_with_activations(
            runtime,
            PlanSet::Selected,
            ActivationForm::Half,
            &mut |_, _| Ok(()),
            storage.buffers_mut(),
        )
    }

    /// Records the batch's forward pass as a CUDA graph, which [`Self::forward`]
    /// then replays instead of issuing each launch
    ///
    /// The graph records the current storage generation and model weights. When
    /// storage grows, forward must recapture before replay. Batches with a Library
    /// plan run one eager pass to finish lazy setup outside capture. Kernel-only
    /// batches are prepared by construction and need no extra pass
    pub fn capture_graph(&mut self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        let activations = self.activations.clone();
        let mut storage = activations.lock()?;
        self.capture_graph_in_storage(runtime, &mut storage)
    }

    fn capture_graph_in_storage(
        &mut self,
        runtime: &CudaRuntime,
        storage: &mut ActivationStorage<EmbeddingActivations>,
    ) -> Result<(), CudaError> {
        self.graph = None;
        if self.head.plan.requires_warmup()
            || self.plans.iter().any(|(_, plan)| plan.requires_warmup())
        {
            self.run_with_activations(
                runtime,
                PlanSet::Selected,
                ActivationForm::Half,
                &mut |_, _| Ok(()),
                storage.buffers_mut(),
            )?;
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
                let run = self.run_with_activations(
                    runtime,
                    PlanSet::Selected,
                    ActivationForm::Half,
                    &mut |_, _| Ok(()),
                    storage.buffers_mut(),
                );
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

        self.graph = captured?.map(|graph| ForwardGraph(Captured::new(graph, storage)));
        debug!(
            chunks = self.chunks,
            captured = self.graph.is_some(),
            "Captured CUDA embedding graph"
        );
        Ok(())
    }

    fn run_with_activations(
        &mut self,
        runtime: &CudaRuntime,
        set: PlanSet,
        form: ActivationForm,
        tap: &mut EmbeddingTapFn<'_>,
        activations: &mut EmbeddingActivations,
    ) -> Result<(), CudaError> {
        let EmbeddingBatch {
            model,
            chunks,
            head,
            fbank,
            masks,
            stem_input,
            pooled,
            output,
            plans,
            workspace,
            fallback,
            ..
        } = self;
        let (plans, workspace) = match (set, fallback) {
            (PlanSet::Selected, _) => (&*plans, workspace),
            (PlanSet::Fallback, Some(fallback)) => {
                let Fallback { plans, workspace } = &mut **fallback;
                (&*plans, workspace)
            }
            (PlanSet::Fallback, None) => {
                return Err(CudaError::Unsupported {
                    context: "embedding fallback",
                    reason: "the plans without FP16 tiles were not built".to_owned(),
                });
            }
        };

        let EmbeddingActivations {
            trunk,
            hidden,
            shortcut,
        } = activations;
        let model = &**model;
        #[cfg(not(feature = "_cuda-libraries"))]
        let _ = workspace;
        let chunks = *chunks;
        let math = model.math;
        let embeddings_len = chunks * SPEAKERS_PER_CHUNK * EMBEDDING_DIM;
        let (mut embeddings, range) = output.split_at_mut(embeddings_len);
        let mut convs = Convs {
            runtime,
            kernels: &model.kernels,
            plans,
            #[cfg(feature = "_cuda-libraries")]
            workspace,
            range,
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
            HalfIo::FP32,
        )?;
        tap(EmbeddingTap::Stem, &stem_out.as_view())?;

        let blocks = &model.trunk.blocks;
        let forms = BlockForm::plan(blocks, plans, form);
        // whether the current block's input, the previous block's output, is half
        let mut input_half = false;
        for (index, block) in blocks.iter().enumerate() {
            let input_len = element_count(
                "embedding block input",
                &block.conv1.conv(chunks, math).input_shape(),
            )?;
            let output_len = chunks * block.conv2.output_len();

            let BlockForm {
                hidden: half,
                output: output_half,
            } = forms[index];
            // a half tensor holds two values per word
            let half_len = |half: bool, len: usize| if half { len / 2 } else { len };
            let (input_buffer, output_buffer) = read_write(trunk, block.input_slot);
            let input = input_buffer.slice(..half_len(input_half, input_len));
            let hidden_len = half_len(half, output_len);

            // the block output buffer is free until the second convolution, so it
            // stands in as the first convolution's unused residual operand; a library
            // plan sizes that operand as the FP32 output even when the block's output
            // is half
            let mut hidden_out = hidden.slice_mut(..hidden_len);
            #[cfg(feature = "_cuda-libraries")]
            let scratch = output_buffer.slice(..output_len);
            convs.conv_bias_relu(
                &block.conv1,
                &input,
                Residual::None {
                    #[cfg(feature = "_cuda-libraries")]
                    scratch: &scratch,
                },
                &mut hidden_out,
                HalfIo {
                    input: input_half,
                    residual: false,
                    output: half,
                },
            )?;
            if !half {
                tap(EmbeddingTap::Hidden { block: index }, &hidden_out.as_view())?;
            }

            let mut block_out = output_buffer.slice_mut(..half_len(output_half, output_len));

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
                None => input.slice(..half_len(input_half, output_len)),
            };
            convs.conv_bias_relu(
                &block.conv2,
                &hidden_out.as_view(),
                Residual::Add(&residual),
                &mut block_out,
                HalfIo {
                    input: half,
                    // without a shortcut the residual is the block input
                    residual: input_half,
                    output: output_half,
                },
            )?;
            if !output_half {
                tap(EmbeddingTap::Block { block: index }, &block_out.as_view())?;
            }
            input_half = output_half;
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
    ///
    /// The same copy reads the out-of-range word. When an FP16 tile saturated an
    /// activation, this recomputes the batch with plans selected without FP16 tiles,
    /// built on first use, and returns that output instead
    pub fn download_output(&mut self, runtime: &CudaRuntime) -> Result<Vec<f32>, CudaError> {
        let (embeddings, saturated) = self.read_output(runtime)?;
        if !saturated {
            return Ok(embeddings);
        }

        FP16_RANGE_WARNING.call_once(|| {
            warn!(
                limit = FP16_OPERAND_LIMIT,
                "CUDA embedding activation exceeded the FP16 operand range; recomputing without FP16 tiles"
            );
        });
        debug!(
            chunks = self.chunks,
            "Recomputing CUDA embedding batch without FP16 tiles"
        );
        if self.fallback.is_none() {
            let fallback = Fallback::new(runtime, &self.model, self.chunks)?;
            self.fallback = Some(Box::new(fallback));
        }
        // the next batch, which may replay the graph, must start from a clear word
        let embeddings_len = self.output.len() - RANGE_WORDS;
        runtime
            .stream()
            .memset_zeros(&mut self.output.slice_mut(embeddings_len..))?;
        {
            let activations = self.activations.clone();
            let mut storage = activations.lock()?;
            self.run_with_activations(
                runtime,
                PlanSet::Fallback,
                ActivationForm::Half,
                &mut |_, _| Ok(()),
                storage.buffers_mut(),
            )?;
        }

        let (embeddings, saturated) = self.read_output(runtime)?;
        if saturated {
            return Err(CudaError::Unsupported {
                context: "embedding fallback",
                reason: "a plan without FP16 tiles set the out-of-range word".to_owned(),
            });
        }
        Ok(embeddings)
    }

    /// The embeddings and whether the out-of-range word is set, in one copy
    fn read_output(&self, runtime: &CudaRuntime) -> Result<(Vec<f32>, bool), CudaError> {
        let mut embeddings = runtime.stream().clone_dtoh(&self.output)?;
        // cuMemcpyDtoHAsync into pageable memory has no completion guarantee
        runtime.synchronize()?;
        let range = embeddings.split_off(self.output.len() - RANGE_WORDS);
        Ok((embeddings, range.iter().any(|word| word.to_bits() != 0)))
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
    /// the out-of-range word FP16 tiles set
    range: CudaViewMut<'a, f32>,
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
mod range_test_support;
#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests;
