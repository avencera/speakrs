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
mod contract;
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
pub use contract::{
    EmbeddingArtifactMetadata, EmbeddingFrontend, EmbeddingGeometryError, EmbeddingInputError,
    EmbeddingInputGeometry, EmbeddingMaskInterpolation, EmbeddingMetadataError, EmbeddingPooling,
    EmbeddingPrecision, EmbeddingRuntimeCapabilities, EmbeddingRuntimeCapability,
    EmbeddingRuntimeProfile, Sha256Digest, VerifiedEmbeddingArtifact,
};
pub(crate) use contract::{
    LEGACY_POOLING_FRAMES, PrimaryTensorShape, geometry_from_primary_shapes,
};
pub(crate) use contract::{clean_mask_threshold, validate_audio_length, validate_mask_length};
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
    array1_slice, array2_from_shape_vec, array3_slice_mut, embedding_batch_from_ort_with_width,
    fbank_hw_from_i64, first_output, preallocated_run_options,
};
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
#[cfg(feature = "coreml")]
pub(crate) const MASK_FRAMES: usize = contract::LEGACY_MASK_FRAMES;

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
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    mode: ExecutionMode,
    geometry: EmbeddingInputGeometry,
    pooling_frames: usize,
    min_num_samples: usize,
    artifact: Option<VerifiedEmbeddingArtifact>,
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

/// WeSpeaker speaker embedding model with split-backend and chunk embedding support
pub struct EmbeddingModel {
    meta: EmbeddingMeta,
    capabilities: EmbeddingRuntimeCapabilities,
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

    /// Return the audio sample rate required by this model
    pub fn sample_rate(&self) -> usize {
        self.meta.geometry.sample_rate()
    }

    /// Return the fixed waveform, mask, and embedding dimensions of this model
    pub fn input_geometry(&self) -> EmbeddingInputGeometry {
        self.meta.geometry
    }

    /// Return the runtime paths admitted by this model's primary geometry
    pub fn capabilities(&self) -> EmbeddingRuntimeCapabilities {
        self.capabilities
    }

    /// Return the exact waveform window length in samples
    pub fn window_samples(&self) -> usize {
        self.meta.geometry.window_samples()
    }

    /// Return the exact segmentation mask length in frames
    pub fn mask_frames(&self) -> usize {
        self.meta.geometry.mask_frames()
    }

    /// Return the embedding vector width
    pub fn embedding_width(&self) -> usize {
        self.meta.geometry.embedding_width()
    }

    /// Return the internal ResNet and statistics-pooling frame target
    pub fn pooling_frames(&self) -> usize {
        self.meta.pooling_frames
    }

    /// Minimum audio samples required for a valid embedding
    pub fn min_num_samples(&self) -> usize {
        self.meta.min_num_samples
    }

    /// Return the verified fixed-shape embedding artifact identity, if present
    pub fn artifact_metadata(&self) -> Option<&VerifiedEmbeddingArtifact> {
        self.meta.artifact.as_ref()
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

    fn mask_selection_window_samples(&self, valid_audio_samples: usize) -> usize {
        mask_selection_window_samples(
            self.capabilities,
            valid_audio_samples,
            self.meta.geometry.window_samples(),
        )
    }

    #[cfg(all(test, feature = "coreml"))]
    pub(crate) fn select_chunk_mask<'a>(
        &self,
        mask: &'a [f32],
        clean_mask: Option<&'a [f32]>,
        num_samples: usize,
    ) -> &'a [f32] {
        select_mask(
            mask,
            clean_mask,
            self.mask_selection_window_samples(num_samples),
            self.meta.min_num_samples,
        )
    }

    fn prepare_waveform(
        batch_idx: usize,
        audio: &[f32],
        window_samples: usize,
        waveform_buffer: &mut ndarray::ArrayViewMut3<f32>,
    ) -> Result<(), ort::Error> {
        validate_audio_length(window_samples, audio.len())?;
        let copy_len = audio.len();
        waveform_buffer.slice_mut(s![batch_idx, 0, ..]).fill(0.0);
        waveform_buffer
            .slice_mut(s![batch_idx, 0, ..copy_len])
            .assign(&ndarray::ArrayView1::from(&audio[..copy_len]));
        Ok(())
    }

    fn prepare_weights(
        batch_idx: usize,
        weights: &[f32],
        mask_frames: usize,
        weights_buffer: &mut ndarray::ArrayViewMut2<f32>,
    ) -> Result<(), ort::Error> {
        validate_mask_length(mask_frames, weights.len())?;
        let mut row = weights_buffer.row_mut(batch_idx);
        row.assign(&ndarray::ArrayView1::from(weights));
        Ok(())
    }

    fn prepare_single_weights(&mut self, weights: &[f32]) -> Result<(), ort::Error> {
        Self::prepare_weights(
            0,
            weights,
            self.meta.geometry.mask_frames(),
            &mut self.buffers.weights_buffer.view_mut(),
        )
    }

    /// Validate audio and mask lengths before an embedding call
    pub fn validate_input(
        &self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<(), EmbeddingInputError> {
        validate_audio_length(self.meta.geometry.window_samples(), audio.len())?;
        validate_mask_length(self.meta.geometry.mask_frames(), mask.len())?;
        if let Some(clean_mask) = clean_mask
            && clean_mask.len() != self.meta.geometry.mask_frames()
        {
            return Err(EmbeddingInputError::CleanMaskLengthMismatch {
                expected: self.meta.geometry.mask_frames(),
                actual: clean_mask.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn validate_audio_input(&self, audio: &[f32]) -> Result<(), ort::Error> {
        validate_audio_length(self.meta.geometry.window_samples(), audio.len())?;
        Ok(())
    }
}

fn mask_selection_window_samples(
    capabilities: EmbeddingRuntimeCapabilities,
    valid_audio_samples: usize,
    declared_window_samples: usize,
) -> usize {
    if capabilities.supports_legacy_optimized() {
        valid_audio_samples
    } else {
        declared_window_samples
    }
}

/// Decide whether clean mask has enough weight, working directly on column views
pub(crate) fn should_use_clean_mask(
    clean_col: &ndarray::ArrayView1<f32>,
    mask_len: usize,
    window_samples: usize,
    min_num_samples: usize,
) -> bool {
    let Some(min_mask_frames) = clean_mask_threshold(mask_len, window_samples, min_num_samples)
    else {
        return false;
    };
    let clean_weight: f32 = clean_col.iter().copied().sum();
    clean_weight > min_mask_frames as f32
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
    fn legacy_short_chunk_uses_actual_audio_for_clean_mask_threshold() {
        let mask = vec![0.0; 399];
        let mut clean = vec![0.0; 399];
        clean[..3].fill(1.0);
        let legacy = EmbeddingRuntimeCapabilities::for_geometry(
            EmbeddingInputGeometry::new(16_000, 160_000, 589, 256).unwrap(),
        );
        let imported = EmbeddingRuntimeCapabilities::for_geometry(
            EmbeddingInputGeometry::new(16_000, 128_000, 399, 256).unwrap(),
        );

        let legacy_window = mask_selection_window_samples(legacy, 8_000, 160_000);
        let imported_window = mask_selection_window_samples(imported, 8_000, 128_000);
        assert_eq!(legacy_window, 8_000);
        assert_eq!(imported_window, 128_000);
        assert_eq!(
            select_mask(&mask, Some(&clean), legacy_window, 100),
            mask.as_slice()
        );
        assert_eq!(
            select_mask(&mask, Some(&clean), imported_window, 100),
            clean.as_slice()
        );
    }

    #[test]
    fn select_mask_falls_back_to_full_mask_when_clean_mask_is_too_short() {
        let mask = [1.0, 1.0, 1.0, 0.0];
        let clean = [1.0, 0.0, 0.0, 0.0];

        let selected = select_mask(&mask, Some(&clean), 16_000, 6_000);

        assert_eq!(selected, mask);
    }

    #[test]
    fn clean_mask_threshold_is_strictly_greater_than_the_fixed_window_threshold() {
        let exactly_at_threshold = array![1.0, 1.0, 0.0, 0.0];
        let above_threshold = array![1.0, 1.0, 1.0, 0.0];

        assert!(!should_use_clean_mask(
            &exactly_at_threshold.view(),
            4,
            16_000,
            6_000,
        ));
        assert!(should_use_clean_mask(
            &above_threshold.view(),
            4,
            16_000,
            6_000,
        ));
    }

    #[test]
    fn prepare_waveform_zero_pads_short_audio_and_rejects_long_audio() {
        let mut buffer = ndarray::Array3::from_elem((1, 1, 4), 9.0);

        assert!(
            EmbeddingModel::prepare_waveform(0, &[1.0, 2.0], 4, &mut buffer.view_mut(),).is_ok()
        );
        assert_eq!(buffer, array![[[1.0, 2.0, 0.0, 0.0]]]);

        assert!(
            EmbeddingModel::prepare_waveform(
                0,
                &[1.0, 2.0, 3.0, 4.0, 5.0],
                4,
                &mut buffer.view_mut(),
            )
            .is_err()
        );
    }

    #[test]
    fn prepare_weights_rejects_masks_that_do_not_match_the_buffer() {
        let mut buffer = ndarray::Array2::from_elem((2, 4), 9.0);

        assert!(
            EmbeddingModel::prepare_weights(0, &[1.0, 2.0, 3.0, 4.0], 4, &mut buffer.view_mut(),)
                .is_ok()
        );

        assert!(
            EmbeddingModel::prepare_weights(0, &[1.0, 2.0], 4, &mut buffer.view_mut()).is_err()
        );
        assert!(
            EmbeddingModel::prepare_weights(
                1,
                &[3.0, 4.0, 5.0, 6.0, 7.0],
                4,
                &mut buffer.view_mut()
            )
            .is_err()
        );

        assert_eq!(buffer, array![[1.0, 2.0, 3.0, 4.0], [9.0, 9.0, 9.0, 9.0]]);
    }
}
