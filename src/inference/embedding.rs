use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "coreml")]
use std::sync::Arc;

#[cfg(feature = "coreml")]
use crate::inference::coreml::{CachedInputShape, CoreMlModel, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError};
use ndarray::{Array2, Array3, s};
#[cfg(feature = "coreml")]
use objc2_core_ml::MLComputeUnits;
use ort::session::{HasSelectedOutputs, RunOptions, Session};

mod batch;
#[cfg(feature = "coreml")]
mod chunk;
mod fbank;
mod load;
#[cfg(feature = "coreml")]
mod native;
mod paths;
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

struct EmbeddingMeta {
    model_path: PathBuf,
    mode: ExecutionMode,
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    min_num_samples: usize,
}

struct OrtEmbeddingState {
    session: Session,
    primary_batched_session: Option<Session>,
    split_fbank_session: Option<Session>,
    split_fbank_batched_session: Option<Session>,
    split_tail_session: Option<Session>,
    split_tail_batched_session: Option<Session>,
    split_primary_tail_batched_session: Option<Session>,
    multi_mask_session: Option<Session>,
    multi_mask_batched_session: Option<Session>,
    primary_batch_run_options: Option<RunOptions<HasSelectedOutputs>>,
}

#[cfg(feature = "coreml")]
struct CoreMlEmbeddingState {
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

/// WeSpeaker speaker embedding model with split-backend and chunk embedding support
pub struct EmbeddingModel {
    meta: EmbeddingMeta,
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

    /// Audio sample rate in Hz (16000)
    pub fn sample_rate(&self) -> usize {
        self.meta.sample_rate
    }

    /// Minimum audio samples required for a valid embedding
    pub fn min_num_samples(&self) -> usize {
        self.meta.min_num_samples
    }

    /// Maximum batch size for the primary (fused) embedding session
    pub fn primary_batch_size(&self) -> usize {
        if self.ort.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            1
        }
    }

    /// Choose the best batch length given the number of pending embeddings
    pub fn best_batch_len(&self, pending_len: usize) -> usize {
        if pending_len >= PRIMARY_BATCH_SIZE && self.ort.primary_batched_session.is_some() {
            PRIMARY_BATCH_SIZE
        } else {
            pending_len.min(1)
        }
    }

    /// Reload all ORT sessions from disk, resetting internal state
    pub fn reset_session(&mut self) -> Result<(), ort::Error> {
        self.ort.session = Self::build_session(
            &self.meta.model_path,
            Self::single_execution_mode(self.meta.mode),
        )?;
        self.ort.primary_batched_session =
            batched_model_path(&self.meta.model_path, PRIMARY_BATCH_SIZE)
                .filter(|path| path.exists())
                .map(|path| Self::build_batched_session(&path, self.meta.mode))
                .transpose()?;
        let split_fbank_path = split_fbank_model_path(&self.meta.model_path);
        let split_tail_path = split_tail_model_path(&self.meta.model_path, 1);
        let split_tail_batched_path =
            split_tail_model_path(&self.meta.model_path, CHUNK_SPEAKER_BATCH_SIZE);
        let split_primary_tail_batched_path =
            split_tail_model_path(&self.meta.model_path, PRIMARY_BATCH_SIZE);
        let use_split_backend = Self::split_backend_available(&self.meta.model_path);
        let split_fbank_batched_path = split_fbank_batched_model_path(&self.meta.model_path);
        self.ort.split_fbank_session = use_split_backend
            .then(|| Self::build_fbank_session(&split_fbank_path, ExecutionMode::Cpu))
            .transpose()?;
        self.ort.split_fbank_batched_session = use_split_backend
            .then_some(split_fbank_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_fbank_session(&path, ExecutionMode::Cpu))
            .transpose()?;
        self.ort.split_tail_session = use_split_backend
            .then(|| Self::build_session(&split_tail_path, self.meta.mode))
            .transpose()?;
        self.ort.split_tail_batched_session = use_split_backend
            .then_some(split_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(&path, self.meta.mode))
            .transpose()?;
        self.ort.split_primary_tail_batched_session = use_split_backend
            .then_some(split_primary_tail_batched_path)
            .filter(|path| path.exists())
            .map(|path| Self::build_session(&path, self.meta.mode))
            .transpose()?;
        #[cfg(feature = "coreml")]
        {
            // keep existing compute units on reload
            self.coreml.native_tail_session = None;
            self.coreml.native_tail_batched_session = None;
            self.coreml.native_tail_primary_batched_session = None;
            self.coreml.native_fbank_session = None;
            self.coreml.native_fbank_batched_session = None;
            self.coreml.native_fbank_30s_session = None;
            self.coreml.native_multi_mask_session = None;
            self.coreml.native_chunk_specs =
                Self::chunk_session_specs(&self.meta.model_path, self.meta.mode);
            self.coreml.native_chunk_sessions.clear();
        }
        self.ort.multi_mask_session = multi_mask_model_path(&self.meta.model_path, 1)
            .filter(|p| p.exists())
            .map(|p| Self::build_session(&p, self.meta.mode))
            .transpose()?;
        self.ort.multi_mask_batched_session =
            multi_mask_model_path(&self.meta.model_path, PRIMARY_BATCH_SIZE)
                .filter(|p| p.exists())
                .map(|p| Self::build_session(&p, self.meta.mode))
                .transpose()?;
        Ok(())
    }

    /// Whether split fbank+tail models are available for chunk embedding
    pub fn prefers_chunk_embedding_path(&self) -> bool {
        let ort_split =
            self.ort.split_fbank_session.is_some() && self.ort.split_tail_session.is_some();
        #[cfg(feature = "coreml")]
        let ort_split =
            ort_split || Self::has_native_tail_model(&self.meta.model_path, self.meta.mode, 1);
        ort_split
    }

    pub(crate) fn split_primary_batch_size(&self) -> usize {
        if self.ort.split_primary_tail_batched_session.is_some() {
            return PRIMARY_BATCH_SIZE;
        }
        #[cfg(feature = "coreml")]
        if Self::has_native_tail_model(&self.meta.model_path, self.meta.mode, PRIMARY_BATCH_SIZE) {
            return PRIMARY_BATCH_SIZE;
        }
        0
    }

    /// Whether a batched fbank session is available for parallel chunk processing
    pub fn has_batched_fbank(&self) -> bool {
        let has = self.ort.split_fbank_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has = has
            || Self::has_native_fbank_model(
                &self.meta.model_path,
                self.meta.mode,
                PRIMARY_BATCH_SIZE,
            );
        has
    }

    /// Whether the multi-mask embedding model is available
    pub fn prefers_multi_mask_path(&self) -> bool {
        let has = self.ort.multi_mask_session.is_some();
        #[cfg(feature = "coreml")]
        let has = has || Self::has_native_multi_mask_model(&self.meta.model_path, self.meta.mode);
        has
    }

    /// Maximum batch size for multi-mask embedding, or 0 if unavailable
    pub fn multi_mask_batch_size(&self) -> usize {
        let has_batched = self.ort.multi_mask_batched_session.is_some();
        #[cfg(feature = "coreml")]
        let has_batched =
            has_batched || Self::has_native_multi_mask_model(&self.meta.model_path, self.meta.mode);
        if has_batched {
            MULTI_MASK_BATCH_SIZE
        } else if self.ort.multi_mask_session.is_some() {
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
        if weights.len() == self.meta.mask_frames {
            self.buffers
                .weights_buffer
                .row_mut(0)
                .assign(&ndarray::ArrayView1::from(weights));
            return;
        }

        let copy_len = weights.len().min(self.meta.mask_frames);
        self.buffers
            .weights_buffer
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
