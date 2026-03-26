use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "coreml")]
use std::sync::Arc;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError, ensure_ort_ready};
use ndarray::{Array2, Array3, s};
#[cfg(feature = "coreml")]
use objc2_core_ml::MLComputeUnits;
use ort::session::{HasSelectedOutputs, RunOptions, Session};

mod batch;
mod fbank;
#[cfg(feature = "coreml")]
mod native;
mod paths;
mod run;
mod session;
mod tail;
mod tensor;

#[cfg(feature = "coreml")]
use paths::fp32_coreml_path;
use paths::{
    batched_model_path, multi_mask_model_path, read_min_num_samples, select_mask,
    split_fbank_batched_model_path, split_fbank_model_path, split_tail_model_path,
};
use tensor::{
    array1_slice, array2_from_shape_vec, array2_slice_mut, array3_slice_mut,
    preallocated_run_options,
};

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
    model: Arc<SharedCoreMlModel>,
    pub num_windows: usize,
    pub fbank_frames: usize,
    pub num_masks: usize,
    cached_fbank_shape: Arc<CachedInputShape>,
    cached_masks_shape: Arc<CachedInputShape>,
}

#[cfg(feature = "coreml")]
#[derive(Clone)]
struct ChunkSessionSpec {
    coreml_path: PathBuf,
    num_windows: usize,
    fbank_frames: usize,
    num_masks: usize,
}

/// All resources needed for chunk embedding, returned by `prepare_chunk_resources`
#[cfg(feature = "coreml")]
pub(crate) struct ChunkResourceBundle {
    pub sessions: Vec<ChunkSessionInfo>,
    pub fbank_30s: Option<Arc<SharedCoreMlModel>>,
    pub fbank_10s: Option<Arc<SharedCoreMlModel>>,
}

#[cfg(feature = "coreml")]
pub(crate) struct ChunkSessionInfo {
    pub model: Arc<SharedCoreMlModel>,
    pub cached_fbank_shape: Arc<CachedInputShape>,
    pub cached_masks_shape: Arc<CachedInputShape>,
    pub num_windows: usize,
    pub fbank_frames: usize,
    pub num_masks: usize,
}

/// WeSpeaker speaker embedding model with split-backend and chunk embedding support
pub struct EmbeddingModel {
    model_path: PathBuf,
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
    native_fbank_session: Option<Arc<SharedCoreMlModel>>,
    #[cfg(feature = "coreml")]
    native_fbank_batched_session: Option<SharedCoreMlModel>,
    #[cfg(feature = "coreml")]
    native_fbank_30s_session: Option<Arc<SharedCoreMlModel>>,
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
    fn split_backend_available(model_path: &Path) -> bool {
        let split_fbank_path = split_fbank_model_path(model_path);
        let split_tail_path = split_tail_model_path(model_path, 1);
        let has_multi_mask = multi_mask_model_path(model_path, 1).is_some_and(|path| path.exists());

        split_fbank_path.exists() && (split_tail_path.exists() || has_multi_mask)
    }

    /// Load the WeSpeaker embedding model
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, ModelLoadError> {
        Self::with_mode(model_path, ExecutionMode::Cpu)
    }

    /// Load the WeSpeaker embedding model with the requested execution mode
    pub fn with_mode(
        model_path: impl AsRef<Path>,
        mode: ExecutionMode,
    ) -> Result<Self, ModelLoadError> {
        Self::with_mode_and_config(model_path, mode, &crate::pipeline::RuntimeConfig::default())
    }

    /// Load the WeSpeaker embedding model with the requested execution mode and runtime config
    pub fn with_mode_and_config(
        model_path: impl AsRef<Path>,
        mode: ExecutionMode,
        config: &crate::pipeline::RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        mode.validate()?;
        ensure_ort_ready()?;

        let model_path = model_path.as_ref();
        let metadata_path = model_path.with_extension("min_num_samples.txt");
        let split_fbank_path = split_fbank_model_path(model_path);
        let split_fbank_batched_path = split_fbank_batched_model_path(model_path);
        let split_tail_path = split_tail_model_path(model_path, 1);
        let split_tail_batched_path = split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path = split_tail_model_path(model_path, PRIMARY_BATCH_SIZE);
        #[cfg(feature = "coreml")]
        let native_chunk_compute_units = config.chunk_emb_compute_units.to_ml_compute_units();
        #[cfg(not(feature = "coreml"))]
        let _ = config;
        // split-backend: CPU fbank + GPU tail/multi-mask
        let use_split_backend = Self::split_backend_available(model_path);

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
                .map(|path| Self::build_batched_session(&path, mode))
                .transpose()?
        );
        let (split_fbank_session, split_fbank_elapsed) = timed!(
            use_split_backend
                .then(|| { Self::build_fbank_session(&split_fbank_path, ExecutionMode::Cpu,) })
                .transpose()?
        );
        let (split_fbank_batched_session, split_fbank_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_fbank_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_fbank_session(&path, ExecutionMode::Cpu))
                .transpose()?
        );
        let (split_tail_session, split_tail_elapsed) = timed!(
            use_split_backend
                .then(|| Self::build_session(&split_tail_path, mode))
                .transpose()?
        );
        let (split_tail_batched_session, split_tail_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(&path, mode))
                .transpose()?
        );
        let (split_primary_tail_batched_session, split_primary_tail_batched_elapsed) = timed!(
            use_split_backend
                .then_some(split_primary_tail_batched_path)
                .filter(|path| path.exists())
                .map(|path| Self::build_session(&path, mode))
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
                .map(|p| Self::build_session(&p, mode))
                .transpose()?
        );
        let (multi_mask_batched_session, multi_mask_batched_elapsed) = timed!(
            multi_mask_model_path(model_path, PRIMARY_BATCH_SIZE)
                .filter(|p| p.exists())
                .map(|p| Self::build_session(&p, mode))
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
            model_path: model_path.to_path_buf(),
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
                    let mut opts = preallocated_run_options(
                        PRIMARY_BATCH_SIZE,
                        256,
                        "primary batched embedding output",
                    )?;
                    // skip device sync between batched calls for async CUDA execution
                    let _ = opts.disable_device_sync();
                    Ok::<RunOptions<HasSelectedOutputs>, ort::Error>(opts)
                })
                .transpose()?,
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

    /// Audio sample rate in Hz (16000)
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Minimum audio samples required for a valid embedding
    pub fn min_num_samples(&self) -> usize {
        self.min_num_samples
    }

    /// Maximum batch size for the primary (fused) embedding session
    pub fn primary_batch_size(&self) -> usize {
        if self.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            1
        }
    }

    /// Choose the best batch length given the number of pending embeddings
    pub fn best_batch_len(&self, pending_len: usize) -> usize {
        if pending_len >= PRIMARY_BATCH_SIZE && self.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            pending_len.min(1)
        }
    }

    /// Reload all ORT sessions from disk, resetting internal state
    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.session =
            Self::build_session(&self.model_path, Self::single_execution_mode(self.mode))?;
        self.primary_batched_session = batched_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| Self::build_batched_session(&path, self.mode))
            .transpose()?;
        let split_fbank_path = split_fbank_model_path(&self.model_path);
        let split_tail_path = split_tail_model_path(&self.model_path, 1);
        let split_tail_batched_path =
            split_tail_model_path(&self.model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path =
            split_tail_model_path(&self.model_path, PRIMARY_BATCH_SIZE);
        let use_split_backend = Self::split_backend_available(&self.model_path);
        let split_fbank_batched_path = split_fbank_batched_model_path(&self.model_path);
        self.split_fbank_session = use_split_backend
            .then(|| Self::build_fbank_session(&split_fbank_path, ExecutionMode::Cpu))
            .transpose()?;
        self.split_fbank_batched_session = use_split_backend
            .then_some(split_fbank_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_fbank_session(&path, ExecutionMode::Cpu))
            .transpose()?;
        self.split_tail_session = use_split_backend
            .then(|| Self::build_session(&split_tail_path, self.mode))
            .transpose()?;
        self.split_tail_batched_session = use_split_backend
            .then_some(split_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(&path, self.mode))
            .transpose()?;
        self.split_primary_tail_batched_session = use_split_backend
            .then_some(split_primary_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(&path, self.mode))
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
        }
        self.multi_mask_session = multi_mask_model_path(&self.model_path, 1)
            .filter(|p| p.exists())
            .map(|p| Self::build_session(&p, self.mode))
            .transpose()?;
        self.multi_mask_batched_session =
            multi_mask_model_path(&self.model_path, PRIMARY_BATCH_SIZE)
                .filter(|p| p.exists())
                .map(|p| Self::build_session(&p, self.mode))
                .transpose()?;
        Ok(())
    }

    /// Whether split fbank+tail models are available for chunk embedding
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

    /// Whether a batched fbank session is available for parallel chunk processing
    pub fn has_batched_fbank(&self) -> bool {
        let has = self.split_fbank_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has =
            has || Self::has_native_fbank_model(&self.model_path, self.mode, PRIMARY_BATCH_SIZE);
        has
    }

    /// Whether the multi-mask embedding model is available
    pub fn prefers_multi_mask_path(&self) -> bool {
        let has = self.multi_mask_session.is_some();
        #[cfg(feature = "coreml")]
        let has = has || Self::has_native_multi_mask_model(&self.model_path, self.mode);
        has
    }

    /// Maximum batch size for multi-mask embedding, or 0 if unavailable
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

    #[cfg(all(test, feature = "coreml"))]
    pub(crate) fn select_chunk_mask<'a>(
        &self,
        mask: &'a [f32],
        clean_mask: Option<&'a [f32]>,
        num_samples: usize,
    ) -> &'a [f32] {
        select_mask(mask, clean_mask, num_samples, self.min_num_samples)
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
