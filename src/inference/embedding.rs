use std::path::Path;

use ndarray::{Array1, Array2, ArrayView2};

use crate::inference::{ExecutionMode, InferenceError, ModelLoadError};

mod buffers;
#[cfg(feature = "coreml")]
mod chunk;
mod load;
#[cfg(feature = "coreml")]
mod native;
#[cfg(feature = "_ort")]
mod onnx;
mod paths;
mod tensor;

#[cfg(feature = "coreml")]
use chunk::ChunkSessionSpec;
#[cfg(feature = "coreml")]
pub(crate) use chunk::{ChunkEmbeddingSession, ChunkResourceBundle, ChunkSessionInfo};
#[cfg(feature = "coreml")]
use native::CoreMlEmbedding;
#[cfg(feature = "_ort")]
use onnx::OrtEmbedding;
pub(crate) use paths::read_min_num_samples;
use paths::select_mask;
use tensor::array1_slice;

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

/// Window geometry shared by every embedding backend
#[derive(Debug, Clone, Copy)]
struct EmbeddingMeta {
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    min_num_samples: usize,
}

/// WeSpeaker speaker embedding model with split-backend and chunk embedding support
pub struct EmbeddingModel {
    meta: EmbeddingMeta,
    backend: EmbeddingBackend,
}

/// Sessions for the one runtime chosen from the execution mode at load time
///
/// A CoreML-mode model holds no ONNX Runtime sessions even when an ORT feature is also
/// enabled, and an ORT-mode model holds no CoreML handles
///
/// Both variants are boxed because the backends carry large inline staging state
enum EmbeddingBackend {
    #[cfg(feature = "_ort")]
    Ort(Box<OrtEmbedding>),
    #[cfg(feature = "coreml")]
    CoreMl(Box<CoreMlEmbedding>),
}

/// Run the same expression against whichever backend the model loaded
macro_rules! with_backend {
    ($backend:expr, $name:ident => $body:expr) => {
        match $backend {
            #[cfg(feature = "_ort")]
            EmbeddingBackend::Ort($name) => $body,
            #[cfg(feature = "coreml")]
            EmbeddingBackend::CoreMl($name) => $body,
        }
    };
}

impl EmbeddingModel {
    /// Load the WeSpeaker embedding model on the CPU
    ///
    /// Requires the `cpu` feature
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, ModelLoadError> {
        Self::with_mode(model_path, ExecutionMode::Cpu)
    }

    /// Load the WeSpeaker embedding model with the requested execution mode
    ///
    /// `model_path` names the base `wespeaker-voxceleb-resnet34.onnx` file. CoreML modes load
    /// the compiled bundles next to it and do not read any ONNX file
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
    #[cfg(all(feature = "_ort", not(feature = "coreml")))]
    pub(crate) fn clone_shared(&self) -> Result<Self, InferenceError> {
        let EmbeddingBackend::Ort(backend) = &self.backend;

        Ok(Self {
            meta: self.meta,
            backend: EmbeddingBackend::Ort(Box::new(backend.clone_shared()?)),
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
        with_backend!(&self.backend, backend => backend.primary_batch_size())
    }

    /// Choose the best batch length given the number of pending embeddings
    pub(crate) fn best_batch_len(&self, pending_len: usize) -> usize {
        if pending_len >= PRIMARY_BATCH_SIZE && self.primary_batch_size() == PRIMARY_BATCH_SIZE {
            PRIMARY_BATCH_SIZE
        } else {
            pending_len.min(1)
        }
    }

    /// Whether split fbank+tail models are available for chunk embedding
    pub(crate) fn prefers_chunk_embedding_path(&self) -> bool {
        with_backend!(&self.backend, backend => backend.prefers_chunk_embedding_path())
    }

    pub(crate) fn split_primary_batch_size(&self) -> usize {
        with_backend!(&self.backend, backend => backend.split_primary_batch_size())
    }

    /// Whether the multi-mask embedding model is available
    pub(crate) fn prefers_multi_mask_path(&self) -> bool {
        with_backend!(&self.backend, backend => backend.prefers_multi_mask_path())
    }

    /// Maximum batch size for multi-mask embedding, or 0 if unavailable
    pub(crate) fn multi_mask_batch_size(&self) -> usize {
        with_backend!(&self.backend, backend => backend.multi_mask_batch_size())
    }

    pub(crate) fn has_batched_tail(&self) -> bool {
        with_backend!(&self.backend, backend => backend.has_batched_tail())
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

    /// Extract a speaker embedding from raw audio with a uniform mask
    pub fn embed(&mut self, audio: &[f32]) -> Result<Array1<f32>, InferenceError> {
        let weights = vec![1.0; self.meta.mask_frames];
        self.embed_single(audio, &weights)
    }

    /// Extract a speaker embedding weighted by a segmentation mask
    pub fn embed_masked(
        &mut self,
        audio: &[f32],
        mask: &[f32],
        clean_mask: Option<&[f32]>,
    ) -> Result<Array1<f32>, InferenceError> {
        let used_mask = select_mask(mask, clean_mask, audio.len(), self.meta.min_num_samples);
        self.embed_single(audio, used_mask)
    }

    fn embed_single(
        &mut self,
        audio: &[f32],
        weights: &[f32],
    ) -> Result<Array1<f32>, InferenceError> {
        with_backend!(&mut self.backend, backend => backend.embed_single(&self.meta, audio, weights))
    }

    /// Extract speaker embeddings for a batch of masked audio windows
    pub fn embed_batch(
        &mut self,
        inputs: &[MaskedEmbeddingInput<'_>],
    ) -> Result<Array2<f32>, InferenceError> {
        let primary_batch = match &mut self.backend {
            #[cfg(feature = "_ort")]
            EmbeddingBackend::Ort(backend) => {
                backend.try_embed_primary_batch(&self.meta, inputs)?
            }
            #[cfg(feature = "coreml")]
            EmbeddingBackend::CoreMl(_) => None,
        };
        if let Some(batch) = primary_batch {
            return Ok(batch);
        }

        let mut stacked = Array2::<f32>::zeros((inputs.len(), EMBEDDING_WIDTH));
        for (idx, input) in inputs.iter().enumerate() {
            let embedding = self.embed_masked(input.audio, input.mask, input.clean_mask)?;
            stacked.row_mut(idx).assign(&embedding);
        }
        Ok(stacked)
    }

    pub(crate) fn embed_multi_mask_batch(
        &mut self,
        fbanks: &[&Array2<f32>],
        masks: &[&[f32]],
    ) -> Result<Array2<f32>, InferenceError> {
        with_backend!(
            &mut self.backend,
            backend => backend.embed_multi_mask_batch(&self.meta, fbanks, masks)
        )
    }

    pub(crate) fn embed_tail_batch_inputs(
        &mut self,
        inputs: &[SplitTailInput<'_>],
    ) -> Result<Array2<f32>, InferenceError> {
        with_backend!(
            &mut self.backend,
            backend => backend.embed_tail_batch_inputs(&self.meta, inputs)
        )
    }

    /// Extract per-speaker embeddings for one audio chunk using segmentation masks
    pub fn embed_chunk_speakers(
        &mut self,
        audio: &[f32],
        segmentations: ArrayView2<'_, f32>,
        clean_masks: &Array2<f32>,
    ) -> Result<Array2<f32>, InferenceError> {
        let speaker_count = segmentations.ncols();
        let mut embeddings = Array2::<f32>::zeros((speaker_count, EMBEDDING_WIDTH));
        if !self.prefers_chunk_embedding_path() {
            for speaker_idx in 0..speaker_count {
                let mask = segmentations.column(speaker_idx).to_owned();
                let clean_mask = clean_masks.column(speaker_idx).to_owned();
                let embedding = self.embed_masked(
                    audio,
                    array1_slice(&mask, "chunk speaker mask")?,
                    Some(array1_slice(&clean_mask, "chunk speaker clean mask")?),
                )?;
                embeddings.row_mut(speaker_idx).assign(&embedding);
            }
            return Ok(embeddings);
        }

        let fbank = self.compute_chunk_fbank(audio)?;
        if speaker_count == CHUNK_SPEAKER_BATCH_SIZE && self.has_batched_tail() {
            return with_backend!(
                &mut self.backend,
                backend => backend.embed_tail_batch(
                    &self.meta,
                    &fbank,
                    &segmentations,
                    clean_masks,
                    audio.len(),
                )
            );
        }

        for speaker_idx in 0..speaker_count {
            let mask = segmentations.column(speaker_idx).to_owned();
            let clean_mask = clean_masks.column(speaker_idx).to_owned();
            let used_mask = select_mask(
                array1_slice(&mask, "chunk tail mask")?,
                Some(array1_slice(&clean_mask, "chunk tail clean mask")?),
                audio.len(),
                self.meta.min_num_samples,
            );
            let embedding = with_backend!(
                &mut self.backend,
                backend => backend.embed_tail_single(&self.meta, &fbank, used_mask)
            )?;
            embeddings.row_mut(speaker_idx).assign(&embedding);
        }

        Ok(embeddings)
    }

    /// Compute fbank features for a single audio chunk via the split fbank model
    pub(crate) fn compute_chunk_fbank(
        &mut self,
        audio: &[f32],
    ) -> Result<Array2<f32>, InferenceError> {
        with_backend!(&mut self.backend, backend => backend.compute_chunk_fbank(&self.meta, audio))
    }

    /// Compute fbank features for multiple audio chunks in a single batched call
    pub fn compute_chunk_fbanks_batch(
        &mut self,
        audios: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, InferenceError> {
        with_backend!(
            &mut self.backend,
            backend => backend.compute_chunk_fbanks_batch(&self.meta, audios)
        )
    }
}

#[cfg(feature = "coreml")]
impl EmbeddingModel {
    fn coreml_backend(&mut self) -> Option<&mut CoreMlEmbedding> {
        match &mut self.backend {
            EmbeddingBackend::CoreMl(backend) => Some(backend),
            #[cfg(feature = "_ort")]
            EmbeddingBackend::Ort(_) => None,
        }
    }

    pub(crate) fn prepare_chunk_resources(
        &mut self,
    ) -> Result<Option<ChunkResourceBundle>, InferenceError> {
        match self.coreml_backend() {
            Some(backend) => backend.prepare_chunk_resources(),
            None => Ok(None),
        }
    }

    pub(crate) fn chunk_window_capacity(&self) -> Option<usize> {
        match &self.backend {
            EmbeddingBackend::CoreMl(backend) => backend.chunk_window_capacity(),
            #[cfg(feature = "_ort")]
            EmbeddingBackend::Ort(_) => None,
        }
    }

    /// Compute fbank for up to 30s of audio in one call
    pub fn compute_chunk_fbank_30s(
        &mut self,
        audio: &[f32],
    ) -> Result<Option<Array2<f32>>, InferenceError> {
        match self.coreml_backend() {
            Some(backend) => backend.compute_chunk_fbank_30s(audio),
            None => Ok(None),
        }
    }

    pub(crate) fn chunk_session_for_windows(
        &mut self,
        num_windows: usize,
    ) -> Result<Option<&ChunkEmbeddingSession>, InferenceError> {
        match self.coreml_backend() {
            Some(backend) => backend.chunk_session_for_windows(num_windows),
            None => Ok(None),
        }
    }

    pub(crate) fn embed_chunk_session(
        session: &ChunkEmbeddingSession,
        full_fbank: &[f32],
        masks: &[f32],
    ) -> Result<Array2<f32>, InferenceError> {
        CoreMlEmbedding::embed_chunk_session(session, full_fbank, masks)
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
