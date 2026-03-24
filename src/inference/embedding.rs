use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3, ArrayView2, s};
#[cfg(feature = "coreml")]
use objc2_core_ml::MLComputeUnits;
use ort::memory::Allocator;
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions, Session};
use ort::value::{Tensor, TensorRef};

#[cfg(feature = "coreml")]
use crate::inference::coreml::{
    CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel, coreml_model_path,
};
use crate::inference::{ExecutionMode, with_execution_mode};

const PRIMARY_BATCH_SIZE: usize = 64;
const MULTI_MASK_BATCH_SIZE: usize = 32;
const FBANK_BATCH_SIZE: usize = 32;
const CHUNK_SPEAKER_BATCH_SIZE: usize = 3;
const NUM_SPEAKERS: usize = 3;
const FBANK_FRAMES: usize = 998;
const FBANK_FEATURES: usize = 80;
const MASK_FRAMES: usize = 589;

pub struct MaskedEmbeddingInput<'a> {
    pub audio: &'a [f32],
    pub mask: &'a [f32],
    pub clean_mask: Option<&'a [f32]>,
}

pub(crate) struct SplitTailInput<'a> {
    pub fbank: &'a Array2<f32>,
    pub weights: &'a [f32],
}

/// Chunk embedding model: runs ResNet once on full-audio fbank, gathers per-window features
#[cfg(feature = "coreml")]
pub(crate) struct ChunkEmbeddingSession {
    model: SharedCoreMlModel,
    pub num_windows: usize,
    pub fbank_frames: usize,
    pub num_masks: usize,
    cached_fbank_shape: CachedInputShape,
    cached_masks_shape: CachedInputShape,
}

#[cfg(feature = "coreml")]
#[derive(Clone)]
struct ChunkSessionSpec {
    coreml_path: PathBuf,
    num_windows: usize,
    fbank_frames: usize,
    num_masks: usize,
}

#[cfg(feature = "coreml")]
pub(crate) struct ChunkSessionRefs<'a> {
    pub num_windows: usize,
    pub fbank_frames: usize,
    pub num_masks: usize,
    pub cached_fbank_shape: &'a CachedInputShape,
    pub cached_masks_shape: &'a CachedInputShape,
    pub model: &'a SharedCoreMlModel,
}

pub struct EmbeddingModel {
    model_path: String,
    mode: ExecutionMode,
    session: Session,
    primary_batched_session: Option<Session>,
    split_fbank_session: Option<Session>,
    split_fbank_batched_session: Option<Session>,
    split_tail_session: Option<Session>,
    split_tail_batched_session: Option<Session>,
    split_primary_tail_batched_session: Option<Session>,
    #[cfg(feature = "coreml")]
    native_tail_session: Option<CoreMlModel>,
    #[cfg(feature = "coreml")]
    native_tail_batched_session: Option<CoreMlModel>,
    #[cfg(feature = "coreml")]
    native_tail_primary_batched_session: Option<CoreMlModel>,
    #[cfg(feature = "coreml")]
    native_fbank_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_fbank_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_fbank_30s_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    cached_fbank_30s_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    native_multi_mask_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_chunk_compute_units: MLComputeUnits,
    #[cfg(feature = "coreml")]
    native_chunk_specs: Vec<ChunkSessionSpec>,
    #[cfg(feature = "coreml")]
    native_chunk_sessions: Vec<ChunkEmbeddingSession>,
    #[cfg(feature = "coreml")]
    native_chunk_session_ane: Option<ChunkEmbeddingSession>,
    multi_mask_session: Option<Session>,
    multi_mask_batched_session: Option<Session>,
    #[cfg(feature = "coreml")]
    cached_tail_fbank_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_tail_weights_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_fbank_single_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_fbank_batch_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_multi_mask_fbank_shape: CachedInputShape,
    #[cfg(feature = "coreml")]
    cached_multi_mask_masks_shape: CachedInputShape,
    primary_batch_run_options: Option<RunOptions<HasSelectedOutputs>>,
    multi_mask_fbank_buffer: Array3<f32>,
    multi_mask_masks_buffer: Array2<f32>,
    waveform_buffer: Array3<f32>,
    weights_buffer: Array2<f32>,
    primary_batch_waveform_buffer: Array3<f32>,
    primary_batch_weights_buffer: Array2<f32>,
    split_waveform_buffer: Array3<f32>,
    split_fbank_batch_buffer: Array3<f32>,
    split_feature_batch_buffer: Array3<f32>,
    split_weights_batch_buffer: Array2<f32>,
    split_primary_feature_batch_buffer: Array3<f32>,
    split_primary_weights_batch_buffer: Array2<f32>,
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    min_num_samples: usize,
}

impl EmbeddingModel {
    /// Load the WeSpeaker embedding model
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        Self::with_mode(model_path, ExecutionMode::Cpu)
    }

    /// Load the WeSpeaker embedding model with the requested execution mode
    pub fn with_mode(model_path: &str, mode: ExecutionMode) -> Result<Self, ort::Error> {
        Self::with_mode_and_config(model_path, mode, &crate::pipeline::RuntimeConfig::default())
    }

    /// Load the WeSpeaker embedding model with the requested execution mode and runtime config
    pub fn with_mode_and_config(
        model_path: &str,
        mode: ExecutionMode,
        config: &crate::pipeline::RuntimeConfig,
    ) -> Result<Self, ort::Error> {
        let metadata_path = Path::new(model_path)
            .with_extension("min_num_samples.txt")
            .to_string_lossy()
            .into_owned();
        let split_fbank_path = split_fbank_model_path(model_path);
        let split_fbank_batched_path = split_fbank_batched_model_path(model_path);
        let split_tail_path = split_tail_model_path(model_path, 1);
        let split_tail_batched_path = split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path = split_tail_model_path(model_path, PRIMARY_BATCH_SIZE);
        let has_multi_mask = multi_mask_model_path(model_path, 1).is_some_and(|p| p.exists());
        #[cfg(feature = "coreml")]
        let native_chunk_compute_units = config.chunk_emb_compute_units.to_ml_compute_units();
        #[cfg(not(feature = "coreml"))]
        let _ = config;
        // split-backend: CPU fbank + GPU tail/multi-mask
        let use_split_backend =
            split_fbank_path.exists() && (split_tail_path.exists() || has_multi_mask);

        macro_rules! timed {
            ($expr:expr) => {{
                let start = std::time::Instant::now();
                let value = $expr;
                (value, start.elapsed())
            }};
        }

        let (session, session_elapsed) = timed!(Self::build_session(
            model_path,
            Self::single_execution_mode(mode)
        )?);
        let (primary_batched_session, primary_batched_elapsed) = timed!(
            batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_batched_session(path.to_str().unwrap(), mode))
                .transpose()?
        );
        let (split_fbank_session, split_fbank_elapsed) = timed!(
            use_split_backend
                .then(|| {
                    Self::build_fbank_session(
                        split_fbank_path.to_str().unwrap(),
                        ExecutionMode::Cpu,
                    )
                })
                .transpose()?
        );
        let (split_fbank_batched_session, split_fbank_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_fbank_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_fbank_session(path.to_str().unwrap(), ExecutionMode::Cpu))
                .transpose()?
        );
        let (split_tail_session, split_tail_elapsed) = timed!(
            use_split_backend
                .then(|| Self::build_session(split_tail_path.to_str().unwrap(), mode))
                .transpose()?
        );
        let (split_tail_batched_session, split_tail_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(path.to_str().unwrap(), mode))
                .transpose()?
        );
        let (split_primary_tail_batched_session, split_primary_tail_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_primary_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(path.to_str().unwrap(), mode))
                .transpose()?
        );
        #[cfg(feature = "coreml")]
        let (native_tail_session, native_tail_elapsed) = (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_tail_batched_session, native_tail_batched_elapsed) =
            timed!(Option::<CoreMlModel>::None);
        #[cfg(feature = "coreml")]
        let (native_tail_primary_batched_session, native_tail_primary_batched_elapsed) =
            (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_fbank_session, native_fbank_elapsed) = (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_fbank_batched_session, native_fbank_batched_elapsed) =
            timed!(Option::<SharedCoreMlModel>::None);
        #[cfg(feature = "coreml")]
        let (native_fbank_30s_session, native_fbank_30s_elapsed) =
            (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_multi_mask_session, native_multi_mask_elapsed) =
            (None, std::time::Duration::ZERO);
        #[cfg(feature = "coreml")]
        let (native_chunk_specs, native_chunk_specs_elapsed) =
            timed!(Self::chunk_session_specs(model_path, mode));
        #[cfg(feature = "coreml")]
        let (native_chunk_sessions, native_chunk_sessions_elapsed) =
            (Vec::new(), std::time::Duration::ZERO);
        let (multi_mask_session, multi_mask_elapsed) = timed!(
            multi_mask_model_path(model_path, 1)
                .filter(|p| p.exists())
                .map(|p| Self::build_session(p.to_str().unwrap(), mode))
                .transpose()?
        );
        let (multi_mask_batched_session, multi_mask_batched_elapsed) = timed!(
            multi_mask_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|p| p.exists())
                .map(|p| Self::build_session(p.to_str().unwrap(), mode))
                .transpose()?
        );

        #[cfg(feature = "coreml")]
        {
            let total_ms = (session_elapsed
                + primary_batched_elapsed
                + split_fbank_elapsed
                + split_fbank_batched_elapsed
                + split_tail_elapsed
                + split_tail_batched_elapsed
                + split_primary_tail_batched_elapsed
                + native_tail_elapsed
                + native_tail_batched_elapsed
                + native_tail_primary_batched_elapsed
                + native_fbank_elapsed
                + native_fbank_batched_elapsed
                + native_fbank_30s_elapsed
                + native_multi_mask_elapsed
                + native_chunk_specs_elapsed
                + native_chunk_sessions_elapsed
                + multi_mask_elapsed
                + multi_mask_batched_elapsed)
                .as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_b64_ms = primary_batched_elapsed.as_millis(),
                split_fbank_ms = split_fbank_elapsed.as_millis(),
                split_fbank_b64_ms = split_fbank_batched_elapsed.as_millis(),
                split_tail_ms = split_tail_elapsed.as_millis(),
                split_tail_b32_ms = split_tail_batched_elapsed.as_millis(),
                split_tail_b64_ms = split_primary_tail_batched_elapsed.as_millis(),
                native_tail_ms = native_tail_elapsed.as_millis(),
                native_tail_b32_ms = native_tail_batched_elapsed.as_millis(),
                native_tail_b64_ms = native_tail_primary_batched_elapsed.as_millis(),
                native_fbank_ms = native_fbank_elapsed.as_millis(),
                native_fbank_b64_ms = native_fbank_batched_elapsed.as_millis(),
                native_fbank_30s_ms = native_fbank_30s_elapsed.as_millis(),
                native_multi_mask_ms = native_multi_mask_elapsed.as_millis(),
                native_chunk_spec_ms = native_chunk_specs_elapsed.as_millis(),
                native_chunk_ms = native_chunk_sessions_elapsed.as_millis(),
                ort_multi_mask_ms = multi_mask_elapsed.as_millis(),
                ort_multi_mask_b64_ms = multi_mask_batched_elapsed.as_millis(),
                total_ms,
                "Embedding model init",
            );
        }
        #[cfg(not(feature = "coreml"))]
        {
            let total_ms = (session_elapsed
                + primary_batched_elapsed
                + split_fbank_elapsed
                + split_fbank_batched_elapsed
                + split_tail_elapsed
                + split_tail_batched_elapsed
                + split_primary_tail_batched_elapsed
                + multi_mask_elapsed
                + multi_mask_batched_elapsed)
                .as_millis();
            tracing::trace!(
                ort_single_ms = session_elapsed.as_millis(),
                ort_b64_ms = primary_batched_elapsed.as_millis(),
                split_fbank_ms = split_fbank_elapsed.as_millis(),
                split_fbank_b64_ms = split_fbank_batched_elapsed.as_millis(),
                split_tail_ms = split_tail_elapsed.as_millis(),
                split_tail_b32_ms = split_tail_batched_elapsed.as_millis(),
                split_tail_b64_ms = split_primary_tail_batched_elapsed.as_millis(),
                ort_multi_mask_ms = multi_mask_elapsed.as_millis(),
                ort_multi_mask_b64_ms = multi_mask_batched_elapsed.as_millis(),
                total_ms,
                "Embedding model init",
            );
        }

        Ok(Self {
            model_path: model_path.to_owned(),
            mode,
            session,
            primary_batched_session,
            split_fbank_session,
            split_fbank_batched_session,
            split_tail_session,
            split_tail_batched_session,
            split_primary_tail_batched_session,
            #[cfg(feature = "coreml")]
            native_tail_session,
            #[cfg(feature = "coreml")]
            native_tail_batched_session,
            #[cfg(feature = "coreml")]
            native_tail_primary_batched_session,
            #[cfg(feature = "coreml")]
            native_fbank_session,
            #[cfg(feature = "coreml")]
            native_fbank_batched_session,
            #[cfg(feature = "coreml")]
            native_fbank_30s_session,
            #[cfg(feature = "coreml")]
            cached_fbank_30s_shape: CachedInputShape::new("waveform", &[1, 1, 480_000]),
            #[cfg(feature = "coreml")]
            native_multi_mask_session,
            #[cfg(feature = "coreml")]
            native_chunk_compute_units,
            #[cfg(feature = "coreml")]
            native_chunk_specs,
            #[cfg(feature = "coreml")]
            native_chunk_sessions,
            #[cfg(feature = "coreml")]
            native_chunk_session_ane: None,
            multi_mask_session,
            multi_mask_batched_session,
            #[cfg(feature = "coreml")]
            cached_tail_fbank_shape: CachedInputShape::new(
                "fbank",
                &[PRIMARY_BATCH_SIZE, FBANK_FRAMES, FBANK_FEATURES],
            ),
            #[cfg(feature = "coreml")]
            cached_tail_weights_shape: CachedInputShape::new(
                "weights",
                &[PRIMARY_BATCH_SIZE, MASK_FRAMES],
            ),
            #[cfg(feature = "coreml")]
            cached_fbank_single_shape: CachedInputShape::new("waveform", &[1, 1, 160_000]),
            #[cfg(feature = "coreml")]
            cached_fbank_batch_shape: CachedInputShape::new(
                "waveform",
                &[FBANK_BATCH_SIZE, 1, 160_000],
            ),
            #[cfg(feature = "coreml")]
            cached_multi_mask_fbank_shape: CachedInputShape::new(
                "fbank",
                &[MULTI_MASK_BATCH_SIZE, FBANK_FRAMES, FBANK_FEATURES],
            ),
            #[cfg(feature = "coreml")]
            cached_multi_mask_masks_shape: CachedInputShape::new(
                "masks",
                &[MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS, MASK_FRAMES],
            ),
            multi_mask_fbank_buffer: Array3::zeros((
                MULTI_MASK_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            multi_mask_masks_buffer: Array2::zeros((
                MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS,
                MASK_FRAMES,
            )),
            waveform_buffer: Array3::zeros((1, 1, 160_000)),
            weights_buffer: Array2::zeros((1, 589)),
            primary_batch_run_options: batched_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|_| {
                    let mut opts = RunOptions::new().unwrap().with_outputs(
                        OutputSelector::default().preallocate(
                            "output",
                            Tensor::<f32>::new(&Allocator::default(), [PRIMARY_BATCH_SIZE, 256])
                                .unwrap(),
                        ),
                    );
                    // skip device sync between batched calls for async CUDA execution
                    let _ = opts.disable_device_sync();
                    opts
                }),
            primary_batch_waveform_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, 160_000)),
            primary_batch_weights_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
            split_waveform_buffer: Array3::zeros((1, 1, 160_000)),
            split_fbank_batch_buffer: Array3::zeros((FBANK_BATCH_SIZE, 1, 160_000)),
            split_feature_batch_buffer: Array3::zeros((
                CHUNK_SPEAKER_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            split_weights_batch_buffer: Array2::zeros((CHUNK_SPEAKER_BATCH_SIZE, 589)),
            split_primary_feature_batch_buffer: Array3::zeros((
                PRIMARY_BATCH_SIZE,
                FBANK_FRAMES,
                FBANK_FEATURES,
            )),
            split_primary_weights_batch_buffer: Array2::zeros((PRIMARY_BATCH_SIZE, 589)),
            sample_rate: 16_000,
            window_samples: 160_000,
            mask_frames: 589,
            min_num_samples: read_min_num_samples(&metadata_path).unwrap_or(400),
        })
    }

    fn build_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        Self::build_session_with_graph(model_path, mode, false)
    }

    fn build_session_with_graph(
        model_path: &str,
        mode: ExecutionMode,
        cuda_graph: bool,
    ) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder =
            if cuda_graph && matches!(mode, ExecutionMode::Cuda | ExecutionMode::CudaFast) {
                Self::with_cuda_graph_mode(builder)?
            } else {
                with_execution_mode(builder, mode)?
            };
        builder.commit_from_file(model_path)
    }

    #[cfg(feature = "cuda")]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        use ort::ep;
        Ok(builder.with_execution_providers([ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
            .with_conv_max_workspace(true)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
            .with_prefer_nhwc(true)
            .with_cuda_graph(true)
            .build()
            .error_on_failure()])?)
    }

    #[cfg(not(feature = "cuda"))]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        with_execution_mode(builder, ExecutionMode::Cpu)
    }

    fn build_fbank_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(1);
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    fn single_execution_mode(mode: ExecutionMode) -> ExecutionMode {
        match mode {
            // keep single embeddings on the CPU path; native CoreML handles the tail
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => ExecutionMode::Cpu,
            _ => mode,
        }
    }

    fn build_batched_session(model_path: &str, mode: ExecutionMode) -> Result<Session, ort::Error> {
        // CUDA graphs don't work with fused model (has CPU-only fbank ops)
        Self::build_session(model_path, Self::single_execution_mode(mode))
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    pub fn min_num_samples(&self) -> usize {
        self.min_num_samples
    }

    #[cfg(feature = "coreml")]
    fn ensure_native_fbank_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.native_fbank_session.is_none() {
            let start = std::time::Instant::now();
            self.native_fbank_session = Self::load_native_fbank(&self.model_path, self.mode, 1);
            if self.native_fbank_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank 10s"
                );
            }
        }
        self.native_fbank_session.as_ref()
    }

    #[cfg(feature = "coreml")]
    fn ensure_native_fbank_batched_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.native_fbank_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.native_fbank_batched_session =
                Self::load_native_fbank(&self.model_path, self.mode, PRIMARY_BATCH_SIZE);
            if self.native_fbank_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank b64"
                );
            }
        }
        self.native_fbank_batched_session.as_ref()
    }

    #[cfg(feature = "coreml")]
    fn ensure_native_fbank_30s_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.native_fbank_30s_session.is_none() {
            let start = std::time::Instant::now();
            self.native_fbank_30s_session =
                Self::load_native_fbank_30s(&self.model_path, self.mode);
            if self.native_fbank_30s_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native fbank 30s"
                );
            }
        }
        self.native_fbank_30s_session.as_ref()
    }

    /// Get the shared fbank 30s model + shape for use from bridge thread
    #[cfg(feature = "coreml")]
    pub(crate) fn fbank_30s_refs(&mut self) -> Option<(&SharedCoreMlModel, &CachedInputShape)> {
        let _ = self.ensure_native_fbank_30s_loaded();
        let session = self.native_fbank_30s_session.as_ref()?;
        Some((session, &self.cached_fbank_30s_shape))
    }

    /// Get 10s fbank model ref for tiling on prep thread
    #[cfg(feature = "coreml")]
    pub(crate) fn fbank_10s_ref(&mut self) -> Option<&SharedCoreMlModel> {
        let _ = self.ensure_native_fbank_loaded();
        self.native_fbank_session.as_ref()
    }

    #[cfg(feature = "coreml")]
    fn ensure_native_multi_mask_loaded(&mut self) -> Option<&SharedCoreMlModel> {
        if self.native_multi_mask_session.is_none() {
            let start = std::time::Instant::now();
            self.native_multi_mask_session =
                Self::load_native_multi_mask(&self.model_path, self.mode);
            if self.native_multi_mask_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native multi mask"
                );
            }
        }
        self.native_multi_mask_session.as_ref()
    }

    #[cfg(feature = "coreml")]
    fn ensure_native_tail_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.native_tail_session.is_none() {
            let start = std::time::Instant::now();
            self.native_tail_session = Self::load_native_tail(&self.model_path, self.mode, 1);
            if self.native_tail_session.is_some() {
                tracing::trace!(ms = start.elapsed().as_millis(), "Lazy loaded native tail");
            }
        }
        self.native_tail_session.as_mut()
    }

    #[cfg(feature = "coreml")]
    fn ensure_native_tail_batched_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.native_tail_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.native_tail_batched_session =
                Self::load_native_tail(&self.model_path, self.mode, CHUNK_SPEAKER_BATCH_SIZE);
            if self.native_tail_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native tail b32"
                );
            }
        }
        self.native_tail_batched_session.as_mut()
    }

    #[cfg(feature = "coreml")]
    fn ensure_native_tail_primary_batched_loaded(&mut self) -> Option<&mut CoreMlModel> {
        if self.native_tail_primary_batched_session.is_none() {
            let start = std::time::Instant::now();
            self.native_tail_primary_batched_session =
                Self::load_native_tail(&self.model_path, self.mode, PRIMARY_BATCH_SIZE);
            if self.native_tail_primary_batched_session.is_some() {
                tracing::trace!(
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded native tail b64"
                );
            }
        }
        self.native_tail_primary_batched_session.as_mut()
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn chunk_window_capacity(&self) -> Option<usize> {
        self.native_chunk_specs.last().map(|spec| spec.num_windows)
    }

    #[cfg(feature = "coreml")]
    fn ensure_chunk_session_loaded(&mut self, num_windows: usize) -> bool {
        let Some(spec) = self
            .native_chunk_specs
            .iter()
            .find(|spec| spec.num_windows >= num_windows)
            .cloned()
        else {
            return false;
        };

        if self
            .native_chunk_sessions
            .iter()
            .any(|session| session.num_windows == spec.num_windows)
        {
            return true;
        }

        let start = std::time::Instant::now();
        match Self::load_chunk_session(&spec, self.native_chunk_compute_units) {
            Ok(session) => {
                tracing::trace!(
                    num_windows = spec.num_windows,
                    ms = start.elapsed().as_millis(),
                    "Lazy loaded chunk embedding",
                );
                self.native_chunk_sessions.push(session);
                self.native_chunk_sessions
                    .sort_by_key(|session| session.num_windows);
                true
            }
            Err(err) => {
                tracing::warn!(
                    num_windows = spec.num_windows,
                    "Failed to lazy load chunk embedding: {err}",
                );
                false
            }
        }
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn ensure_chunk_session_loaded_pub(&mut self, num_windows: usize) {
        let _ = self.ensure_chunk_session_loaded(num_windows);
    }

    /// Get all chunk session metadata + model refs for pipelined embedding
    #[cfg(feature = "coreml")]
    pub(crate) fn chunk_session_refs(&self) -> Vec<ChunkSessionRefs<'_>> {
        self.native_chunk_sessions
            .iter()
            .map(|s| ChunkSessionRefs {
                num_windows: s.num_windows,
                fbank_frames: s.fbank_frames,
                num_masks: s.num_masks,
                cached_fbank_shape: &s.cached_fbank_shape,
                cached_masks_shape: &s.cached_masks_shape,
                model: &s.model,
            })
            .collect()
    }

    /// Load the largest chunk session with CPUAndNeuralEngine compute units for ANE predict worker
    #[cfg(feature = "coreml")]
    pub(crate) fn ensure_chunk_session_ane_loaded(&mut self) {
        if self.native_chunk_session_ane.is_some() {
            return;
        }
        let Some(spec) = self.native_chunk_specs.last().cloned() else {
            return;
        };
        let start = std::time::Instant::now();
        match Self::load_chunk_session(&spec, MLComputeUnits::CPUAndNeuralEngine) {
            Ok(session) => {
                tracing::trace!(
                    num_windows = spec.num_windows,
                    ms = start.elapsed().as_millis(),
                    "Loaded ANE chunk embedding session",
                );
                self.native_chunk_session_ane = Some(session);
            }
            Err(err) => {
                tracing::warn!(
                    num_windows = spec.num_windows,
                    "Failed to load ANE chunk embedding: {err}",
                );
            }
        }
    }

    /// Get the ANE chunk session refs for pipelined embedding
    #[cfg(feature = "coreml")]
    pub(crate) fn chunk_session_ane_ref(&self) -> Option<ChunkSessionRefs<'_>> {
        self.native_chunk_session_ane
            .as_ref()
            .map(|s| ChunkSessionRefs {
                num_windows: s.num_windows,
                fbank_frames: s.fbank_frames,
                num_masks: s.num_masks,
                cached_fbank_shape: &s.cached_fbank_shape,
                cached_masks_shape: &s.cached_masks_shape,
                model: &s.model,
            })
    }

    pub fn primary_batch_size(&self) -> usize {
        if self.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            1
        }
    }

    pub fn best_batch_len(&self, pending_len: usize) -> usize {
        if pending_len >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            pending_len.min(1)
        }
    }

    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.session =
            Self::build_session(&self.model_path, Self::single_execution_mode(self.mode))?;
        self.primary_batched_session = batched_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_batched_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        let split_fbank_path = split_fbank_model_path(&self.model_path);
        let split_tail_path = split_tail_model_path(&self.model_path, 1);
        let split_tail_batched_path =
            split_tail_model_path(&self.model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path =
            split_tail_model_path(&self.model_path, PRIMARY_BATCH_SIZE);
        let has_multi_mask = multi_mask_model_path(&self.model_path, 1).is_some_and(|p| p.exists());
        let use_split_backend =
            (matches!(self.mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast)
                && split_fbank_path.exists()
                && split_tail_path.exists())
                || (has_multi_mask && split_fbank_path.exists());
        let split_fbank_batched_path = split_fbank_batched_model_path(&self.model_path);
        self.split_fbank_session = use_split_backend
            .then(|| {
                Self::build_fbank_session(split_fbank_path.to_str().unwrap(), ExecutionMode::Cpu)
            })
            .transpose()?;
        self.split_fbank_batched_session = use_split_backend
            .then_some(split_fbank_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_fbank_session(path.to_str().unwrap(), ExecutionMode::Cpu))
            .transpose()?;
        self.split_tail_session = use_split_backend
            .then(|| Self::build_session(split_tail_path.to_str().unwrap(), self.mode))
            .transpose()?;
        self.split_tail_batched_session = use_split_backend
            .then_some(split_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        self.split_primary_tail_batched_session = use_split_backend
            .then_some(split_primary_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(path.to_str().unwrap(), self.mode))
            .transpose()?;
        #[cfg(feature = "coreml")]
        {
            // keep existing compute units on reload
            self.native_tail_session = None;
            self.native_tail_batched_session = None;
            self.native_tail_primary_batched_session = None;
            self.native_fbank_session = None;
            self.native_fbank_batched_session = None;
            self.native_fbank_30s_session = None;
            self.native_multi_mask_session = None;
            self.native_chunk_specs = Self::chunk_session_specs(&self.model_path, self.mode);
            self.native_chunk_sessions.clear();
            self.native_chunk_session_ane = None;
        }
        self.multi_mask_session = multi_mask_model_path(&self.model_path, 1)
            .filter(|p| p.exists())
            .map(|p| Self::build_session(p.to_str().unwrap(), self.mode))
            .transpose()?;
        self.multi_mask_batched_session =
            multi_mask_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
                .filter(|p| p.exists())
                .map(|p| Self::build_session(p.to_str().unwrap(), self.mode))
                .transpose()?;
        Ok(())
    }

    pub fn prefers_chunk_embedding_path(&self) -> bool {
        let ort_split = self.split_fbank_session.is_some() && self.split_tail_session.is_some();
        #[cfg(feature = "coreml")]
        let ort_split = ort_split || Self::has_native_tail_model(&self.model_path, self.mode, 1);
        ort_split
    }

    pub(crate) fn split_primary_batch_size(&self) -> usize {
        if self.split_primary_tail_batched_session.is_some() {
            return PRIMARY_BATCH_SIZE;
        }
        #[cfg(feature = "coreml")]
        if Self::has_native_tail_model(&self.model_path, self.mode, PRIMARY_BATCH_SIZE) {
            return PRIMARY_BATCH_SIZE;
        }
        0
    }

    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let weights = vec![1.0; self.mask_frames];
        self.embed_single(audio, &weights)
    }

    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, ort::Error> {
        let used_mask = select_mask(mask, clean_mask, audio.len(), self.min_num_samples);
        self.embed_single(audio, used_mask)
    }

    pub fn embed_batch(
        &mut self,
        inputs: &[MaskedEmbeddingInput<'_>],
    ) -> Result<Array2<f32>, ort::Error> {
        if let Some(sess) = self
            .primary_batched_session
            .as_mut()
            .filter(|_| inputs.len() == PRIMARY_BATCH_SIZE)
        {
            for (batch_idx, input) in inputs.iter().enumerate() {
                let used_mask = select_mask(
                    input.mask,
                    input.clean_mask,
                    input.audio.len(),
                    self.min_num_samples,
                );
                Self::prepare_waveform(
                    batch_idx,
                    input.audio,
                    self.window_samples,
                    &mut self.primary_batch_waveform_buffer.view_mut(),
                );
                Self::prepare_weights(
                    batch_idx,
                    used_mask,
                    self.mask_frames,
                    &mut self.primary_batch_weights_buffer.view_mut(),
                );
            }

            let waveform_tensor =
                TensorRef::from_array_view(self.primary_batch_waveform_buffer.view())?;
            let weights_tensor =
                TensorRef::from_array_view(self.primary_batch_weights_buffer.view())?;
            let ort_inputs =
                ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor];
            let outputs = if let Some(opts) = &self.primary_batch_run_options {
                sess.run_with_options(ort_inputs, opts)?
            } else {
                sess.run(ort_inputs)?
            };
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            let n = inputs.len();
            let mut result = Array2::<f32>::zeros((n, 256));
            result
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(&data[..n * 256]);
            return Ok(result);
        }

        let mut stacked = Array2::<f32>::zeros((inputs.len(), 256));
        for (idx, input) in inputs.iter().enumerate() {
            let embedding = self.embed_masked(input.audio, input.mask, input.clean_mask)?;
            stacked.row_mut(idx).assign(&embedding);
        }
        Ok(stacked)
    }

    pub fn embed_chunk_speakers(
        &mut self,
        audio: &[f32],
        segmentations: ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
    ) -> Result<Array2<f32>, ort::Error> {
        let speaker_count = segmentations.ncols();
        let mut embeddings = Array2::<f32>::zeros((speaker_count, 256));
        if !self.prefers_chunk_embedding_path() {
            for speaker_idx in 0..speaker_count {
                let mask = segmentations.column(speaker_idx).to_owned();
                let clean_mask = clean_masks.column(speaker_idx).to_owned();
                let embedding = self.embed_masked(
                    audio,
                    mask.as_slice().unwrap(),
                    Some(clean_mask.as_slice().unwrap()),
                )?;
                embeddings.row_mut(speaker_idx).assign(&embedding);
            }
            return Ok(embeddings);
        }

        let fbank = self.compute_chunk_fbank(audio)?;
        let has_batched_tail = self.split_tail_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has_batched_tail = has_batched_tail
            || Self::has_native_tail_model(&self.model_path, self.mode, CHUNK_SPEAKER_BATCH_SIZE);
        if speaker_count == CHUNK_SPEAKER_BATCH_SIZE && has_batched_tail {
            return self.embed_tail_batch(&fbank, &segmentations, clean_masks, audio.len());
        }

        for speaker_idx in 0..speaker_count {
            let mask = segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let used_mask = select_mask(
                mask.as_slice().unwrap(),
                Some(clean_mask.as_slice().unwrap()),
                audio.len(),
                self.min_num_samples,
            );
            let embedding = self.embed_tail_single(&fbank, used_mask)?;
            embeddings.row_mut(speaker_idx).assign(&embedding);
        }

        Ok(embeddings)
    }

    fn embed_single(&mut self, audio: &[f32], weights: &[f32]) -> Result<Array1<f32>, ort::Error> {
        let copy_len = audio.len().min(self.window_samples);
        self.waveform_buffer
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        if copy_len < self.window_samples {
            self.waveform_buffer
                .slice_mut(s![0, 0, copy_len..])
                .fill(0.0);
        }
        self.prepare_single_weights(weights);

        let waveform_tensor = TensorRef::from_array_view(self.waveform_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.weights_buffer.view())?;
        let outputs = self
            .session
            .run(ort::inputs!["waveform" => waveform_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(data.to_vec()))
    }

    pub fn compute_chunk_fbank(&mut self, audio: &[f32]) -> Result<Array2<f32>, ort::Error> {
        let copy_len = audio.len().min(self.window_samples);
        self.split_waveform_buffer
            .slice_mut(s![0, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        if copy_len < self.window_samples {
            self.split_waveform_buffer
                .slice_mut(s![0, 0, copy_len..])
                .fill(0.0);
        }

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_fbank_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_fbank_session.as_ref() {
            let input_data = self.split_waveform_buffer.as_slice().unwrap();
            let (data, out_shape) = native
                .predict_cached(&[(&self.cached_fbank_single_shape, input_data)])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            let frames = out_shape[1];
            let features = out_shape[2];
            return Ok(Array2::from_shape_vec((frames, features), data).unwrap());
        }

        let waveform_tensor = TensorRef::from_array_view(self.split_waveform_buffer.view())?;
        let outputs = self
            .split_fbank_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["waveform" => waveform_tensor])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let frames = shape[1] as usize;
        let features = shape[2] as usize;
        Ok(Array2::from_shape_vec((frames, features), data.to_vec()).unwrap())
    }

    pub fn compute_chunk_fbanks_batch(
        &mut self,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        let has_batched = self.split_fbank_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has_batched = has_batched
            || Self::has_native_fbank_model(&self.model_path, self.mode, PRIMARY_BATCH_SIZE);
        if !has_batched {
            tracing::debug!(
                count = audios.len(),
                "fbank: no batched session, falling back to per-window"
            );
            return audios
                .iter()
                .map(|audio| self.compute_chunk_fbank(audio))
                .collect();
        }
        let mut results = Vec::with_capacity(audios.len());
        for batch_start in (0..audios.len()).step_by(FBANK_BATCH_SIZE) {
            let batch_end = (batch_start + FBANK_BATCH_SIZE).min(audios.len());
            let batch = &audios[batch_start..batch_end];

            if batch.len() == FBANK_BATCH_SIZE {
                for (idx, audio) in batch.iter().enumerate() {
                    let copy_len = audio.len().min(self.window_samples);
                    self.split_fbank_batch_buffer
                        .slice_mut(s![idx, 0, ..copy_len])
                        .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
                    if copy_len < self.window_samples {
                        self.split_fbank_batch_buffer
                            .slice_mut(s![idx, 0, copy_len..])
                            .fill(0.0);
                    }
                }

                #[cfg(feature = "coreml")]
                {
                    let _ = self.ensure_native_fbank_batched_loaded();
                }
                #[cfg(feature = "coreml")]
                if let Some(native) = self.native_fbank_batched_session.as_ref() {
                    let input_data = self.split_fbank_batch_buffer.as_slice().unwrap();
                    let (data, out_shape) = native
                        .predict_cached(&[(&self.cached_fbank_batch_shape, input_data)])
                        .map_err(|e| ort::Error::new(e.to_string()))?;
                    let frames = out_shape[1];
                    let features = out_shape[2];
                    let stride = frames * features;
                    for idx in 0..FBANK_BATCH_SIZE {
                        let start = idx * stride;
                        results.push(
                            Array2::from_shape_vec(
                                (frames, features),
                                data[start..start + stride].to_vec(),
                            )
                            .unwrap(),
                        );
                    }
                    continue;
                }

                let waveform_tensor =
                    TensorRef::from_array_view(self.split_fbank_batch_buffer.view())?;
                let outputs = self
                    .split_fbank_batched_session
                    .as_mut()
                    .unwrap()
                    .run(ort::inputs!["waveform" => waveform_tensor])?;
                let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
                let frames = shape[1] as usize;
                let features = shape[2] as usize;
                let flat = data.to_vec();
                let stride = frames * features;
                for idx in 0..PRIMARY_BATCH_SIZE {
                    let start = idx * stride;
                    results.push(
                        Array2::from_shape_vec(
                            (frames, features),
                            flat[start..start + stride].to_vec(),
                        )
                        .unwrap(),
                    );
                }
            } else if batch.len() > 1 {
                // zero-pad partial batch to full batch size for batched inference
                let actual_count = batch.len();
                for (idx, audio) in batch.iter().enumerate() {
                    let copy_len = audio.len().min(self.window_samples);
                    self.split_fbank_batch_buffer
                        .slice_mut(s![idx, 0, ..copy_len])
                        .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
                    if copy_len < self.window_samples {
                        self.split_fbank_batch_buffer
                            .slice_mut(s![idx, 0, copy_len..])
                            .fill(0.0);
                    }
                }
                // zero unused rows
                for idx in actual_count..FBANK_BATCH_SIZE {
                    self.split_fbank_batch_buffer
                        .slice_mut(s![idx, 0, ..])
                        .fill(0.0);
                }

                #[cfg(feature = "coreml")]
                {
                    let _ = self.ensure_native_fbank_batched_loaded();
                }
                #[cfg(feature = "coreml")]
                if let Some(native) = self.native_fbank_batched_session.as_ref() {
                    let input_data = self.split_fbank_batch_buffer.as_slice().unwrap();
                    let (data, out_shape) = native
                        .predict_cached(&[(&self.cached_fbank_batch_shape, input_data)])
                        .map_err(|e| ort::Error::new(e.to_string()))?;
                    let frames = out_shape[1];
                    let features = out_shape[2];
                    let stride = frames * features;
                    for idx in 0..actual_count {
                        let start = idx * stride;
                        results.push(
                            Array2::from_shape_vec(
                                (frames, features),
                                data[start..start + stride].to_vec(),
                            )
                            .unwrap(),
                        );
                    }
                    continue;
                }

                // ORT fallback for partial batch
                for audio in batch {
                    results.push(self.compute_chunk_fbank(audio)?);
                }
            } else {
                for audio in batch {
                    results.push(self.compute_chunk_fbank(audio)?);
                }
            }
        }

        Ok(results)
    }

    pub fn has_batched_fbank(&self) -> bool {
        let has = self.split_fbank_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has =
            has || Self::has_native_fbank_model(&self.model_path, self.mode, PRIMARY_BATCH_SIZE);
        has
    }

    pub fn prefers_multi_mask_path(&self) -> bool {
        let has = self.multi_mask_session.is_some();
        #[cfg(feature = "coreml")]
        let has = has || Self::has_native_multi_mask_model(&self.model_path, self.mode);
        has
    }

    pub fn multi_mask_batch_size(&self) -> usize {
        let has_batched = self.multi_mask_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has_batched =
            has_batched || Self::has_native_multi_mask_model(&self.model_path, self.mode);
        if has_batched {
            MULTI_MASK_BATCH_SIZE
        } else if self.multi_mask_session.is_some() {
            1
        } else {
            0
        }
    }

    /// Run multi-mask inference: B fbanks + B*3 masks -> B*3 embeddings (sliced to actual count)
    pub(crate) fn embed_multi_mask_batch(
        &mut self,
        fbanks: &[&Array2<f32>],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, ort::Error> {
        let num_fbanks = fbanks.len();
        let num_masks = masks.len();
        debug_assert_eq!(num_masks, num_fbanks * NUM_SPEAKERS);
        debug_assert!(num_fbanks <= MULTI_MASK_BATCH_SIZE);

        let fbank_row_stride = FBANK_FRAMES * FBANK_FEATURES;
        for (idx, fbank) in fbanks.iter().enumerate() {
            self.multi_mask_fbank_buffer
                .slice_mut(s![idx, ..fbank.nrows(), ..fbank.ncols()])
                .assign(fbank);
        }

        for (idx, mask) in masks.iter().enumerate() {
            Self::prepare_weights(
                idx,
                mask,
                self.mask_frames,
                &mut self.multi_mask_masks_buffer.view_mut(),
            );
        }
        // zero unused fbank rows
        if num_fbanks < MULTI_MASK_BATCH_SIZE {
            let start = num_fbanks * fbank_row_stride;
            let buf = self.multi_mask_fbank_buffer.as_slice_mut().unwrap();
            buf[start..].fill(0.0);
        }
        // zero unused mask rows
        if num_masks < MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS {
            self.multi_mask_masks_buffer
                .slice_mut(s![num_masks.., ..])
                .fill(0.0);
        }

        let full_mask_batch = MULTI_MASK_BATCH_SIZE * NUM_SPEAKERS;

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_multi_mask_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_multi_mask_session.as_ref() {
            let fbank_data = self.multi_mask_fbank_buffer.as_slice().unwrap();
            let masks_data = self.multi_mask_masks_buffer.as_slice().unwrap();
            let (data, _) = native
                .predict_cached(&[
                    (&self.cached_multi_mask_fbank_shape, fbank_data),
                    (&self.cached_multi_mask_masks_shape, masks_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            let batch = Array2::from_shape_vec((full_mask_batch, 256), data).unwrap();
            return Ok(batch.slice(s![0..num_masks, ..]).to_owned());
        }

        let use_batched =
            num_fbanks == MULTI_MASK_BATCH_SIZE && self.multi_mask_batched_session.is_some();

        if use_batched {
            let fbank_tensor = TensorRef::from_array_view(self.multi_mask_fbank_buffer.view())?;
            let masks_tensor = TensorRef::from_array_view(self.multi_mask_masks_buffer.view())?;
            let outputs = self
                .multi_mask_batched_session
                .as_mut()
                .unwrap()
                .run(ort::inputs!["fbank" => fbank_tensor, "masks" => masks_tensor])?;
            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            let batch = Array2::from_shape_vec((full_mask_batch, 256), data.to_vec()).unwrap();
            Ok(batch.slice(s![0..num_masks, ..]).to_owned())
        } else {
            let mut all_embeddings = Array2::<f32>::zeros((num_masks, 256));
            for fbank_idx in 0..num_fbanks {
                let fbank_slice =
                    self.multi_mask_fbank_buffer
                        .slice(s![fbank_idx..fbank_idx + 1, .., ..]);
                let mask_start = fbank_idx * NUM_SPEAKERS;
                let mask_end = mask_start + NUM_SPEAKERS;
                let masks_slice = self
                    .multi_mask_masks_buffer
                    .slice(s![mask_start..mask_end, ..]);
                let fbank_tensor = TensorRef::from_array_view(fbank_slice.view())?;
                let masks_tensor = TensorRef::from_array_view(masks_slice.view())?;
                let outputs = self
                    .multi_mask_session
                    .as_mut()
                    .unwrap()
                    .run(ort::inputs!["fbank" => fbank_tensor, "masks" => masks_tensor])?;
                let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
                for (local_idx, row_idx) in (mask_start..mask_end).enumerate() {
                    let start = local_idx * 256;
                    all_embeddings
                        .row_mut(row_idx)
                        .assign(&ndarray::ArrayView1::from(&data[start..start + 256]));
                }
            }
            Ok(all_embeddings)
        }
    }

    #[cfg(all(test, feature = "coreml"))]
    pub(crate) fn select_chunk_mask<'a>(
        &self,
        mask: &'a [f32],
        clean_mask: Option<&'a [f32]>,
        num_samples: usize,
    ) -> &'a [f32] {
        select_mask(mask, clean_mask, num_samples, self.min_num_samples)
    }

    pub(crate) fn embed_tail_batch_inputs(
        &mut self,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, ort::Error> {
        debug_assert!(inputs.len() <= PRIMARY_BATCH_SIZE);

        let row_stride = FBANK_FRAMES * FBANK_FEATURES;
        for (batch_idx, input) in inputs.iter().enumerate() {
            debug_assert_eq!(input.fbank.ncols(), FBANK_FEATURES);

            // reuse previous copy if consecutive items share the same fbank
            if batch_idx > 0 && std::ptr::eq(input.fbank, inputs[batch_idx - 1].fbank) {
                let buf = self
                    .split_primary_feature_batch_buffer
                    .as_slice_mut()
                    .unwrap();
                let prev_start = (batch_idx - 1) * row_stride;
                buf.copy_within(prev_start..prev_start + row_stride, batch_idx * row_stride);
            } else {
                self.split_primary_feature_batch_buffer
                    .slice_mut(s![batch_idx, ..input.fbank.nrows(), ..input.fbank.ncols()])
                    .assign(input.fbank);
            }

            Self::prepare_weights(
                batch_idx,
                input.weights,
                self.mask_frames,
                &mut self.split_primary_weights_batch_buffer.view_mut(),
            );
        }
        // zero unused weight rows so weighted pooling produces sentinel embeddings
        if inputs.len() < PRIMARY_BATCH_SIZE {
            self.split_primary_weights_batch_buffer
                .slice_mut(s![inputs.len().., ..])
                .fill(0.0);
        }

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_tail_primary_batched_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_tail_primary_batched_session.as_mut() {
            let fbank_data = self.split_primary_feature_batch_buffer.as_slice().unwrap();
            let weights_data = self.split_primary_weights_batch_buffer.as_slice().unwrap();
            let (data, _) = native
                .predict_cached(&[
                    (&self.cached_tail_fbank_shape, fbank_data),
                    (&self.cached_tail_weights_shape, weights_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            let batch = Array2::from_shape_vec((PRIMARY_BATCH_SIZE, 256), data).unwrap();
            return Ok(batch.slice(s![0..inputs.len(), ..]).to_owned());
        }

        let fbank_tensor =
            TensorRef::from_array_view(self.split_primary_feature_batch_buffer.view())?;
        let weights_tensor =
            TensorRef::from_array_view(self.split_primary_weights_batch_buffer.view())?;
        let outputs = self
            .split_primary_tail_batched_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let batch = Array2::from_shape_vec((PRIMARY_BATCH_SIZE, 256), data.to_vec()).unwrap();
        Ok(batch.slice(s![0..inputs.len(), ..]).to_owned())
    }

    fn embed_tail_single(
        &mut self,
        fbank: &Array2<f32>,
        weights: &[f32],
    ) -> Result<Array1<f32>, ort::Error> {
        self.split_feature_batch_buffer
            .slice_mut(s![0, ..fbank.nrows(), ..fbank.ncols()])
            .assign(fbank);
        Self::prepare_weights(
            0,
            weights,
            self.mask_frames,
            &mut self.split_weights_batch_buffer.view_mut(),
        );

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_tail_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_tail_session.as_mut() {
            let feature_slice = self.split_feature_batch_buffer.slice(s![0..1, .., ..]);
            let weight_slice = self.split_weights_batch_buffer.slice(s![0..1, ..]);
            let fbank_data = feature_slice.as_slice().unwrap();
            let weights_data = weight_slice.as_slice().unwrap();
            let (data, _) = native
                .predict(&[
                    ("fbank", &[1, FBANK_FRAMES, FBANK_FEATURES], fbank_data),
                    ("weights", &[1, self.mask_frames], weights_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            return Ok(Array1::from_vec(data));
        }

        let feature_slice = self.split_feature_batch_buffer.slice(s![0..1, .., ..]);
        let weight_slice = self.split_weights_batch_buffer.slice(s![0..1, ..]);
        let fbank_tensor = TensorRef::from_array_view(feature_slice.view())?;
        let weights_tensor = TensorRef::from_array_view(weight_slice.view())?;
        let outputs = self
            .split_tail_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(data.to_vec()))
    }

    fn embed_tail_batch(
        &mut self,
        fbank: &Array2<f32>,
        segmentations: &ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
        num_samples: usize,
    ) -> Result<Array2<f32>, ort::Error> {
        // copy fbank once to slot 0, then replicate via copy_within
        self.split_feature_batch_buffer
            .slice_mut(s![0, ..fbank.nrows(), ..fbank.ncols()])
            .assign(fbank);
        let row_stride = FBANK_FRAMES * FBANK_FEATURES;
        let fbank_elems = fbank.nrows() * fbank.ncols();
        let buf = self.split_feature_batch_buffer.as_slice_mut().unwrap();
        for speaker_idx in 1..segmentations.ncols() {
            buf.copy_within(0..fbank_elems, speaker_idx * row_stride);
        }

        for speaker_idx in 0..segmentations.ncols() {
            let mask_col = segmentations.column(speaker_idx);
            let clean_col = clean_masks.column(speaker_idx);
            let use_clean = should_use_clean_mask(
                &clean_col,
                mask_col.len(),
                num_samples,
                self.min_num_samples,
            );
            let weights: Vec<f32> = if use_clean {
                clean_col.iter().copied().collect()
            } else {
                mask_col.iter().copied().collect()
            };
            Self::prepare_weights(
                speaker_idx,
                &weights,
                self.mask_frames,
                &mut self.split_weights_batch_buffer.view_mut(),
            );
        }

        #[cfg(feature = "coreml")]
        {
            let _ = self.ensure_native_tail_batched_loaded();
        }
        #[cfg(feature = "coreml")]
        if let Some(native) = self.native_tail_batched_session.as_mut() {
            let fbank_data = self.split_feature_batch_buffer.as_slice().unwrap();
            let weights_data = self.split_weights_batch_buffer.as_slice().unwrap();
            let batch = CHUNK_SPEAKER_BATCH_SIZE;
            let (data, _) = native
                .predict(&[
                    ("fbank", &[batch, FBANK_FRAMES, FBANK_FEATURES], fbank_data),
                    ("weights", &[batch, self.mask_frames], weights_data),
                ])
                .map_err(|e| ort::Error::new(e.to_string()))?;
            return Ok(Array2::from_shape_vec((segmentations.ncols(), 256), data).unwrap());
        }

        let fbank_tensor = TensorRef::from_array_view(self.split_feature_batch_buffer.view())?;
        let weights_tensor = TensorRef::from_array_view(self.split_weights_batch_buffer.view())?;
        let outputs = self
            .split_tail_batched_session
            .as_mut()
            .unwrap()
            .run(ort::inputs!["fbank" => fbank_tensor, "weights" => weights_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array2::from_shape_vec((segmentations.ncols(), 256), data.to_vec()).unwrap())
    }

    fn prepare_waveform(
        batch_idx: usize,
        audio: &[f32],
        window_samples: usize,
        waveform_buffer: &mut ndarray::ArrayViewMut3<f32>,
    ) {
        let copy_len = audio.len().min(window_samples);
        waveform_buffer
            .slice_mut(s![batch_idx, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        if copy_len < window_samples {
            waveform_buffer
                .slice_mut(s![batch_idx, 0, copy_len..])
                .fill(0.0);
        }
    }

    fn prepare_weights(
        batch_idx: usize,
        weights: &[f32],
        mask_frames: usize,
        weights_buffer: &mut ndarray::ArrayViewMut2<f32>,
    ) {
        if weights.len() == mask_frames {
            weights_buffer
                .row_mut(batch_idx)
                .assign(&ndarray::ArrayView1::from(weights));
            return;
        }

        let copy_len = weights.len().min(mask_frames);
        weights_buffer
            .slice_mut(s![batch_idx, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&weights[..copy_len]));
    }

    fn prepare_single_weights(&mut self, weights: &[f32]) {
        if weights.len() == self.mask_frames {
            self.weights_buffer
                .row_mut(0)
                .assign(&ndarray::ArrayView1::from(weights));
            return;
        }

        let copy_len = weights.len().min(self.mask_frames);
        self.weights_buffer
            .slice_mut(s![0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&weights[..copy_len]));
    }

    #[cfg(feature = "coreml")]
    fn load_native_tail(
        model_path: &str,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Option<CoreMlModel> {
        let (resolve_path, compute_units) = match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => (
                coreml_model_path as fn(&str) -> std::path::PathBuf,
                CoreMlModel::default_compute_units(),
            ),
            _ => return None,
        };
        let tail_onnx = split_tail_model_path(model_path, batch_size);
        let coreml_path = resolve_path(tail_onnx.to_str().unwrap());
        if !coreml_path.exists() {
            if batch_size == 1 {
                tracing::warn!(
                    path = %coreml_path.display(),
                    "Native CoreML tail model not found, falling back to ORT CPU",
                );
            }
            return None;
        }
        match CoreMlModel::load(&coreml_path, compute_units, "output", GpuPrecision::Low) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!(batch_size, "Failed to load native CoreML tail: {e}");
                None
            }
        }
    }

    #[cfg(feature = "coreml")]
    fn has_native_tail_model(model_path: &str, mode: ExecutionMode, batch_size: usize) -> bool {
        let resolve_path = match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {
                coreml_model_path as fn(&str) -> std::path::PathBuf
            }
            _ => return false,
        };
        let tail_onnx = split_tail_model_path(model_path, batch_size);
        resolve_path(tail_onnx.to_str().unwrap()).exists()
    }

    #[cfg(feature = "coreml")]
    fn load_native_fbank(
        model_path: &str,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let fbank_onnx = if batch_size == 1 {
            split_fbank_model_path(model_path)
        } else {
            split_fbank_batched_model_path(model_path)
        };
        // fbank DFT matmul needs FP32 for accuracy -- always use FP32 CPU+GPU
        let coreml_path = coreml_model_path(fbank_onnx.to_str().unwrap());
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            CoreMlModel::default_compute_units(),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!(batch_size, "Failed to load native CoreML fbank: {e}");
                None
            }
        }
    }

    #[cfg(feature = "coreml")]
    fn has_native_fbank_model(model_path: &str, mode: ExecutionMode, batch_size: usize) -> bool {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return false;
        }
        let fbank_onnx = if batch_size == 1 {
            split_fbank_model_path(model_path)
        } else {
            split_fbank_batched_model_path(model_path)
        };
        coreml_model_path(fbank_onnx.to_str().unwrap()).exists()
    }

    /// Load a 30s fbank model (480000 samples → ~2998 frames)
    #[cfg(feature = "coreml")]
    fn load_native_fbank_30s(model_path: &str, mode: ExecutionMode) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let coreml_path = Path::new(model_path).with_file_name("wespeaker-fbank-30s.mlmodelc");
        if !coreml_path.exists() {
            return None;
        }
        // fbank on ANE frees CPU prep threads for mask extraction
        match SharedCoreMlModel::load(
            &coreml_path,
            objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => {
                tracing::info!("Loaded 30s fbank model (CPUAndNeuralEngine)");
                Some(model)
            }
            Err(e) => {
                tracing::warn!("Failed to load 30s fbank model: {e}");
                None
            }
        }
    }

    /// Compute fbank for up to 30s of audio in ONE call (no tiling)
    #[cfg(feature = "coreml")]
    pub fn compute_chunk_fbank_30s(
        &mut self,
        audio: &[f32],
    ) -> Option<Result<Array2<f32>, ort::Error>> {
        // only use if audio fits in 30s (480000 samples)
        if audio.len() > 480_000 {
            return None;
        }
        let _ = self.ensure_native_fbank_30s_loaded();
        let native = self.native_fbank_30s_session.as_ref()?;
        let mut buffer = vec![0.0f32; 480_000];
        buffer[..audio.len()].copy_from_slice(audio);
        let result = native
            .predict_cached(&[(&self.cached_fbank_30s_shape, &buffer)])
            .map_err(|e| ort::Error::new(e.to_string()));
        Some(result.map(|(data, out_shape)| {
            let frames = out_shape[1];
            let features = out_shape[2];
            Array2::from_shape_vec((frames, features), data).unwrap()
        }))
    }

    #[cfg(feature = "coreml")]
    fn load_native_multi_mask(model_path: &str, mode: ExecutionMode) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        // use the b32 compiled model (supports both b1 and b32 via EnumeratedShapes)
        let onnx_path = Path::new(model_path).with_file_name("wespeaker-multimask-tail-b32.onnx");
        // W8A16 embedding disabled pending DER validation
        let coreml_path = coreml_model_path(onnx_path.to_str().unwrap());
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            CoreMlModel::default_compute_units(),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!("Failed to load native CoreML multi-mask: {e}");
                None
            }
        }
    }

    #[cfg(feature = "coreml")]
    fn has_native_multi_mask_model(model_path: &str, mode: ExecutionMode) -> bool {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return false;
        }
        let onnx_path = Path::new(model_path).with_file_name("wespeaker-multimask-tail-b32.onnx");
        coreml_model_path(onnx_path.to_str().unwrap()).exists()
    }

    #[cfg(feature = "coreml")]
    fn chunk_session_config(mode: ExecutionMode) -> &'static [(usize, usize, usize, usize)] {
        match mode {
            ExecutionMode::CoreMlFast => &[
                (25, 11, 3000, 33),
                (25, 16, 4000, 48),
                (25, 21, 5000, 63),
                (25, 26, 6000, 78),
                (25, 36, 8000, 108),
                (25, 46, 10000, 138),
                (25, 56, 12000, 168),
            ],
            _ => &[
                (12, 22, 3016, 66),
                (12, 37, 4456, 111),
                (12, 53, 5992, 159),
                (12, 84, 8968, 252),
                (12, 116, 12040, 348),
            ],
        }
    }

    #[cfg(feature = "coreml")]
    fn chunk_session_specs(model_path: &str, mode: ExecutionMode) -> Vec<ChunkSessionSpec> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return Vec::new();
        }

        Self::chunk_session_config(mode)
            .iter()
            .filter_map(|&(step_resnet, num_windows, fbank_frames, num_masks)| {
                let stem = format!("wespeaker-chunk-emb-s{step_resnet}-w{num_windows}");
                let w8a16_path =
                    Path::new(model_path).with_file_name(format!("{stem}-w8a16.mlmodelc"));
                let fp32_path = Path::new(model_path).with_file_name(format!("{stem}.mlmodelc"));

                let coreml_path = if fp32_path.exists() {
                    fp32_path
                } else if w8a16_path.exists() {
                    w8a16_path
                } else {
                    return None;
                };

                Some(ChunkSessionSpec {
                    coreml_path,
                    num_windows,
                    fbank_frames,
                    num_masks,
                })
            })
            .collect()
    }

    #[cfg(feature = "coreml")]
    fn load_chunk_session(
        spec: &ChunkSessionSpec,
        compute_units: MLComputeUnits,
    ) -> Result<ChunkEmbeddingSession, crate::inference::coreml::CoreMlError> {
        let model = SharedCoreMlModel::load(
            &spec.coreml_path,
            compute_units,
            "output",
            GpuPrecision::Low,
        )?;
        Ok(ChunkEmbeddingSession {
            model,
            num_windows: spec.num_windows,
            fbank_frames: spec.fbank_frames,
            num_masks: spec.num_masks,
            cached_fbank_shape: CachedInputShape::new(
                "fbank",
                &[1, spec.fbank_frames, FBANK_FEATURES],
            ),
            cached_masks_shape: CachedInputShape::new("masks", &[spec.num_masks, MASK_FRAMES]),
        })
    }

    /// Find the best chunk session for a given number of windows
    #[cfg(feature = "coreml")]
    pub(crate) fn chunk_session_for_windows(
        &mut self,
        num_windows: usize,
    ) -> Option<&ChunkEmbeddingSession> {
        if !self.ensure_chunk_session_loaded(num_windows) {
            return None;
        }
        self.native_chunk_sessions
            .iter()
            .find(|s| s.num_windows >= num_windows)
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn embed_chunk_session(
        session: &ChunkEmbeddingSession,
        full_fbank: &[f32],
        masks: &[f32],
    ) -> Result<Array2<f32>, ort::Error> {
        let (data, _) = session
            .model
            .predict_cached(&[
                (&session.cached_fbank_shape, full_fbank),
                (&session.cached_masks_shape, masks),
            ])
            .map_err(|e| ort::Error::new(e.to_string()))?;
        let num_masks = session.num_masks;
        Ok(Array2::from_shape_vec((num_masks, 256), data).unwrap())
    }
}

fn batched_model_path(model_path: &str, batch_size: usize) -> Option<std::path::PathBuf> {
    let path = Path::new(model_path);
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}

fn split_fbank_model_path(model_path: &str) -> std::path::PathBuf {
    let path = Path::new(model_path);
    path.with_file_name("wespeaker-fbank.onnx")
}

fn split_fbank_batched_model_path(model_path: &str) -> std::path::PathBuf {
    let path = Path::new(model_path);
    path.with_file_name("wespeaker-fbank-b32.onnx")
}

fn split_tail_model_path(model_path: &str, batch_size: usize) -> std::path::PathBuf {
    let path = Path::new(model_path);
    if batch_size == 1 {
        path.with_file_name("wespeaker-voxceleb-resnet34-tail.onnx")
    } else {
        path.with_file_name(format!(
            "wespeaker-voxceleb-resnet34-tail-b{batch_size}.onnx"
        ))
    }
}

#[allow(dead_code)]
fn multi_mask_model_path(model_path: &str, batch_size: usize) -> Option<std::path::PathBuf> {
    let path = Path::new(model_path);
    if batch_size == 1 {
        Some(path.with_file_name("wespeaker-multimask-tail.onnx"))
    } else {
        Some(path.with_file_name(format!("wespeaker-multimask-tail-b{batch_size}.onnx")))
    }
}

fn read_min_num_samples(path: &str) -> Option<usize> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn select_mask<'a>(
    mask: &'a [f32],
    clean_mask: Option<&'a [f32]>,
    num_samples: usize,
    min_num_samples: usize,
) -> &'a [f32] {
    let Some(clean_mask) = clean_mask else {
        return mask;
    };

    if clean_mask.len() != mask.len() || num_samples == 0 {
        return mask;
    }

    let min_mask_frames = (mask.len() * min_num_samples).div_ceil(num_samples) as f32;
    let clean_weight: f32 = clean_mask.iter().copied().sum();
    if clean_weight > min_mask_frames {
        clean_mask
    } else {
        mask
    }
}

/// Decide whether clean mask has enough weight, working directly on column views
pub(crate) fn should_use_clean_mask(
    clean_col: &ndarray::ArrayView1<f32>,
    mask_len: usize,
    num_samples: usize,
    min_num_samples: usize,
) -> bool {
    if num_samples == 0 {
        return false;
    }
    let min_mask_frames = (mask_len * min_num_samples).div_ceil(num_samples) as f32;
    let clean_weight: f32 = clean_col.iter().copied().sum();
    clean_weight > min_mask_frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_mask_prefers_clean_mask_when_it_is_long_enough() {
        let mask = [1.0, 1.0, 1.0, 0.0];
        let clean = [1.0, 1.0, 1.0, 0.0];

        let selected = select_mask(&mask, Some(&clean), 16_000, 6_000);

        assert_eq!(selected, clean);
    }

    #[test]
    fn select_mask_falls_back_to_full_mask_when_clean_mask_is_too_short() {
        let mask = [1.0, 1.0, 1.0, 0.0];
        let clean = [1.0, 0.0, 0.0, 0.0];

        let selected = select_mask(&mask, Some(&clean), 16_000, 6_000);

        assert_eq!(selected, mask);
    }
}
