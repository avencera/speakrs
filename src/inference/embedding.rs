use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{CachedInputShape, CoreMlModel, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError, SharedSession};
use ndarray::{Array2, Array3, s};
#[cfg(feature = "coreml")]
use objc2_core_ml::MLComputeUnits;
use ort::session::{HasSelectedOutputs, RunOptions};

mod batch;
#[cfg(feature = "coreml")]
mod chunk;
mod fbank;
mod load;
#[cfg(feature = "coreml")]
mod native;
mod paths;
mod plan;
mod run;
mod session;
mod tail;
mod tensor;

#[cfg(feature = "coreml")]
use chunk::ChunkSessionSpec;
#[cfg(feature = "coreml")]
pub(crate) use chunk::{ChunkEmbeddingSession, ChunkResourceBundle, ChunkSessionInfo};
#[cfg(feature = "coreml")]
use paths::fp32_coreml_path;
pub(crate) use paths::read_min_num_samples;
use paths::{
    batched_model_path, multi_mask_model_path, select_mask, split_fbank_batched_model_path,
    split_fbank_model_path, split_tail_model_path,
};
use plan::EmbeddingExecutionPlan;
#[cfg(feature = "coreml")]
use plan::LazySession;
#[cfg(feature = "coreml")]
use tensor::fbank_hw_from_shape;
use tensor::{
    array1_slice, array2_from_shape_vec, array3_slice_mut, embedding_batch_from_ort,
    embedding_vector_from_ort, fbank_hw_from_i64, first_output, preallocated_run_options,
};
#[cfg(feature = "coreml")]
use tensor::{embedding_batch_from_coreml, embedding_vector_from_coreml};

const PRIMARY_BATCH_SIZE: usize = 64;
pub(crate) const EMBEDDING_WIDTH: usize = 256;
const MULTI_MASK_BATCH_SIZE: usize = 32;
const FBANK_BATCH_SIZE: usize = 32;
const CHUNK_SPEAKER_BATCH_SIZE: usize = 3;
const NUM_SPEAKERS: usize = 3;
pub(crate) const FBANK_FRAMES: usize = 998;
/// Hop between consecutive fbank frames, in samples (10ms at 16kHz)
#[cfg(feature = "coreml")]
pub(crate) const FBANK_HOP_SAMPLES: usize = 160;
pub(crate) const FBANK_FEATURES: usize = 80;
const MASK_FRAMES: usize = 589;

/// One masked audio window for [`EmbeddingModel::embed_batch`]
pub struct MaskedEmbeddingInput<'a> {
    /// 16 kHz mono samples (padded/truncated to the model window internally)
    pub audio: &'a [f32],
    /// Frame-level speaker weights over the segmentation mask grid
    pub mask: &'a [f32],
    /// Optional overlap-cleaned mask, preferred when it keeps enough weight
    pub clean_mask: Option<&'a [f32]>,
}

pub(crate) struct SplitTailInput<'a> {
    pub fbank: &'a Array2<f32>,
    pub weights: &'a [f32],
}

#[derive(Clone)]
struct EmbeddingMeta {
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    mode: ExecutionMode,
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    min_num_samples: usize,
}

struct OrtEmbeddingState {
    session: SharedSession,
    primary_batched_session: Option<SharedSession>,
    split_fbank_session: Option<SharedSession>,
    split_fbank_pool: SharedFbankPool,
    split_fbank_batched_session: Option<SharedSession>,
    split_tail_session: Option<SharedSession>,
    split_tail_batched_session: Option<SharedSession>,
    split_primary_tail_batched_session: Option<SharedSession>,
    multi_mask_session: Option<SharedSession>,
    multi_mask_batched_session: Option<SharedSession>,
    // per-handle state carries a preallocated output tensor
    // do not share it across concurrent runs
    primary_batch_run_options: Option<RunOptions<HasSelectedOutputs>>,
}

#[derive(Clone)]
struct SharedFbankPool(Arc<FbankPool>);

struct FbankPool {
    sessions: Vec<SharedSession>,
    execution: std::sync::Mutex<()>,
}

impl SharedFbankPool {
    fn new(sessions: Vec<SharedSession>) -> Self {
        Self(Arc::new(FbankPool {
            sessions,
            execution: std::sync::Mutex::new(()),
        }))
    }

    fn len(&self) -> usize {
        self.0.sessions.len()
    }

    fn run(
        &self,
        audios: &[&[f32]],
        window_samples: usize,
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        let _execution = self
            .0
            .execution
            .lock()
            .map_err(|_| ort::Error::new("shared filterbank pool lock was poisoned"))?;

        fbank::compute_fbanks_with_pool(&self.0.sessions, audios, window_samples)
    }
}

impl OrtEmbeddingState {
    fn fresh_primary_run_options(
        has_primary_batched: bool,
    ) -> Result<Option<RunOptions<HasSelectedOutputs>>, ort::Error> {
        has_primary_batched
            .then(|| {
                let mut opts = preallocated_run_options(
                    PRIMARY_BATCH_SIZE,
                    256,
                    "primary batched embedding output",
                )?;
                let _ = opts.disable_device_sync();
                Ok::<RunOptions<HasSelectedOutputs>, ort::Error>(opts)
            })
            .transpose()
    }

    #[cfg(not(feature = "coreml"))]
    fn clone_shared(&self) -> Result<Self, ort::Error> {
        Ok(Self {
            session: self.session.clone(),
            primary_batched_session: self.primary_batched_session.clone(),
            split_fbank_session: self.split_fbank_session.clone(),
            split_fbank_pool: self.split_fbank_pool.clone(),
            split_fbank_batched_session: self.split_fbank_batched_session.clone(),
            split_tail_session: self.split_tail_session.clone(),
            split_tail_batched_session: self.split_tail_batched_session.clone(),
            split_primary_tail_batched_session: self.split_primary_tail_batched_session.clone(),
            multi_mask_session: self.multi_mask_session.clone(),
            multi_mask_batched_session: self.multi_mask_batched_session.clone(),
            primary_batch_run_options: Self::fresh_primary_run_options(
                self.primary_batched_session.is_some(),
            )?,
        })
    }
}

#[cfg(feature = "coreml")]
struct CoreMlEmbeddingState {
    native_tail_session: LazySession<PathBuf, CoreMlModel>,
    native_tail_batched_session: LazySession<PathBuf, CoreMlModel>,
    native_tail_primary_batched_session: LazySession<PathBuf, CoreMlModel>,
    native_fbank_session: LazySession<PathBuf, Arc<SharedCoreMlModel>>,
    native_fbank_batched_session: LazySession<PathBuf, SharedCoreMlModel>,
    native_fbank_30s_session: LazySession<PathBuf, Arc<SharedCoreMlModel>>,
    cached_fbank_30s_shape: CachedInputShape,
    native_multi_mask_session: LazySession<PathBuf, SharedCoreMlModel>,
    native_embedding_compute_units: MLComputeUnits,
    native_chunk_sessions: Vec<LazySession<ChunkSessionSpec, ChunkEmbeddingSession>>,
    cached_tail_fbank_shape: CachedInputShape,
    cached_tail_weights_shape: CachedInputShape,
    cached_fbank_single_shape: CachedInputShape,
    cached_fbank_batch_shape: CachedInputShape,
    cached_multi_mask_fbank_shape: CachedInputShape,
    cached_multi_mask_masks_shape: CachedInputShape,
}

struct EmbeddingBuffers {
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
}

impl EmbeddingBuffers {
    fn fresh() -> Self {
        Self {
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
        }
    }
}

/// WeSpeaker speaker embedding model with split-backend and chunk embedding support
pub struct EmbeddingModel {
    meta: EmbeddingMeta,
    plan: EmbeddingExecutionPlan,
    ort: OrtEmbeddingState,
    #[cfg(feature = "coreml")]
    coreml: CoreMlEmbeddingState,
    buffers: EmbeddingBuffers,
}

impl EmbeddingModel {
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

    /// Create a handle that shares ORT sessions and owns new scratch buffers
    ///
    /// Session weights and arenas are shared. Staging buffers and preallocated
    /// output state remain private to the new handle.
    #[cfg(not(feature = "coreml"))]
    pub(crate) fn clone_shared(&self) -> Result<Self, ort::Error> {
        Ok(Self {
            meta: self.meta.clone(),
            plan: self.plan.clone(),
            ort: self.ort.clone_shared()?,
            buffers: EmbeddingBuffers::fresh(),
        })
    }

    /// Audio sample rate in Hz (16000)
    pub fn sample_rate(&self) -> usize {
        self.meta.sample_rate
    }

    /// Minimum audio samples required for a valid embedding
    pub fn min_num_samples(&self) -> usize {
        self.meta.min_num_samples
    }

    /// Maximum batch size for the primary (fused) embedding session
    pub(crate) fn primary_batch_size(&self) -> usize {
        if self.ort.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            1
        }
    }

    /// Choose the best batch length given the number of pending embeddings
    pub(crate) fn best_batch_len(&self, pending_len: usize) -> usize {
        if pending_len >= PRIMARY_BATCH_SIZE && self.ort.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            pending_len.min(1)
        }
    }

    /// Whether split fbank+tail models are available for chunk embedding
    pub(crate) fn prefers_chunk_embedding_path(&self) -> bool {
        self.plan.prefers_chunk_embedding_path()
    }

    pub(crate) fn split_primary_batch_size(&self) -> usize {
        self.plan.split_primary_batch_size()
    }

    /// Whether a batched fbank session is available for parallel chunk processing
    pub(crate) fn has_batched_fbank(&self) -> bool {
        self.plan.has_batched_fbank()
    }

    /// Whether the multi-mask embedding model is available
    pub(crate) fn prefers_multi_mask_path(&self) -> bool {
        self.plan.prefers_multi_mask_path()
    }

    /// Maximum batch size for multi-mask embedding, or 0 if unavailable
    pub(crate) fn multi_mask_batch_size(&self) -> usize {
        self.plan.multi_mask_batch_size()
    }

    pub(crate) fn has_batched_tail(&self) -> bool {
        self.plan.has_batched_tail()
    }

    #[cfg(all(test, feature = "coreml"))]
    pub(crate) fn select_chunk_mask<'a>(
        &self,
        mask: &'a [f32],
        clean_mask: Option<&'a [f32]>,
        num_samples: usize,
    ) -> &'a [f32] {
        select_mask(mask, clean_mask, num_samples, self.meta.min_num_samples)
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
        let mut row = weights_buffer.row_mut(batch_idx);
        if weights.len() == mask_frames {
            row.assign(&ndarray::ArrayView1::from(weights));
            return;
        }

        let copy_len = weights.len().min(mask_frames);
        row.fill(0.0);
        row.slice_mut(s![..copy_len])
            .assign(&ndarray::ArrayView1::from(&weights[..copy_len]));
    }

    fn prepare_single_weights(&mut self, weights: &[f32]) {
        Self::prepare_weights(
            0,
            weights,
            self.meta.mask_frames,
            &mut self.buffers.weights_buffer.view_mut(),
        );
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
    use ndarray::array;

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

    #[test]
    fn prepare_weights_clears_tail_when_mask_is_shorter_than_buffer() {
        let mut buffer = ndarray::Array2::from_elem((2, 4), 9.0);

        EmbeddingModel::prepare_weights(0, &[1.0, 2.0], 4, &mut buffer.view_mut());
        EmbeddingModel::prepare_weights(1, &[3.0, 4.0, 5.0, 6.0, 7.0], 4, &mut buffer.view_mut());

        assert_eq!(buffer, array![[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 6.0]]);
    }
}
