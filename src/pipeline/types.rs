use std::ops::Deref;

use ndarray::{Array2, Array3, s};
use tracing::info;

use crate::inference::embedding::{EmbeddingModel, MaskedEmbeddingInput, SplitTailInput};
use crate::powerset::PowersetMapping;
use crate::reconstruct::Reconstructor;
use crate::segment::to_segments;

use super::config::*;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("{0}")]
    Ort(String),
    #[error("{0}")]
    Plda(#[from] crate::clustering::plda::PldaError),
    #[cfg(feature = "online")]
    #[error("{0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    #[error("{0}")]
    Other(String),
}

impl From<ort::Error> for PipelineError {
    fn from(e: ort::Error) -> Self {
        Self::Ort(e.to_string())
    }
}

impl From<crate::inference::segmentation::SegmentationError> for PipelineError {
    fn from(e: crate::inference::segmentation::SegmentationError) -> Self {
        Self::Other(e.to_string())
    }
}

pub(crate) struct SpeakerEmbedding {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub embedding: Vec<f32>,
}

pub(super) struct PendingEmbedding<'a> {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub audio: &'a [f32],
    pub mask: Vec<f32>,
    pub clean_mask: Vec<f32>,
}

pub(super) struct PendingSplitEmbedding {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub fbank_idx: usize,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DecodedSegmentations(pub Array3<f32>);

impl Deref for DecodedSegmentations {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct ChunkEmbeddings(pub Array3<f32>);

impl Deref for ChunkEmbeddings {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct SpeakerCountTrack(pub Vec<usize>);

impl Deref for SpeakerCountTrack {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct ChunkSpeakerClusters(pub Array2<i32>);

impl Deref for ChunkSpeakerClusters {
    type Target = Array2<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct DiscreteDiarization(pub Array2<f32>);

impl Deref for DiscreteDiarization {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DiscreteDiarization {
    pub fn to_segments(
        &self,
        frame_step_seconds: f64,
        frame_duration_seconds: f64,
    ) -> Vec<crate::segment::Segment> {
        to_segments(&self.0, frame_step_seconds, frame_duration_seconds)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FrameActivations(pub(crate) Array2<f32>);

impl Deref for FrameActivations {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(super) struct RawSegmentationWindows(pub Vec<Array2<f32>>);

impl RawSegmentationWindows {
    pub(super) fn decode(self, powerset: &PowersetMapping) -> DecodedSegmentations {
        let num_windows = self.0.len();
        if num_windows == 0 {
            return DecodedSegmentations(Array3::zeros((0, 0, 0)));
        }

        let mut windows = self.0.into_iter();
        let first = powerset.hard_decode(&windows.next().unwrap());
        let mut stacked = Array3::<f32>::zeros((num_windows, first.nrows(), first.ncols()));
        stacked.slice_mut(s![0, .., ..]).assign(&first);

        for (window_idx, window) in windows.enumerate() {
            let decoded = powerset.hard_decode(&window);
            stacked
                .slice_mut(s![window_idx + 1, .., ..])
                .assign(&decoded);
        }

        DecodedSegmentations(stacked)
    }
}

pub(super) struct ChunkLayout {
    pub step_seconds: f64,
    pub step_samples: usize,
    pub window_samples: usize,
    pub start_frames: Vec<usize>,
    pub output_frames: usize,
}

impl ChunkLayout {
    pub fn new(
        step_seconds: f64,
        step_samples: usize,
        window_samples: usize,
        num_chunks: usize,
    ) -> Self {
        Self {
            step_seconds,
            step_samples,
            window_samples,
            start_frames: chunk_start_frames(num_chunks, step_seconds),
            output_frames: total_output_frames(num_chunks, step_seconds),
        }
    }

    pub fn without_frame_extent(
        step_seconds: f64,
        step_samples: usize,
        window_samples: usize,
    ) -> Self {
        Self::new(step_seconds, step_samples, window_samples, 0)
    }

    pub fn with_num_chunks(mut self, num_chunks: usize) -> Self {
        self.start_frames = chunk_start_frames(num_chunks, self.step_seconds);
        self.output_frames = total_output_frames(num_chunks, self.step_seconds);
        self
    }

    pub fn chunk_audio<'a>(&self, audio: &'a [f32], chunk_idx: usize) -> &'a [f32] {
        chunk_audio_raw(audio, self.step_samples, self.window_samples, chunk_idx)
    }
}

pub struct InferenceArtifacts {
    pub(super) layout: ChunkLayout,
    pub(super) segmentations: DecodedSegmentations,
    pub(super) embeddings: ChunkEmbeddings,
}

pub struct DiarizationResult {
    pub segmentations: DecodedSegmentations,
    pub embeddings: ChunkEmbeddings,
    pub speaker_count: SpeakerCountTrack,
    pub hard_clusters: ChunkSpeakerClusters,
    pub discrete_diarization: DiscreteDiarization,
    pub rttm: String,
}

impl DiarizationResult {
    pub fn rttm(&self, file_id: &str) -> String {
        if file_id == "file1" {
            return self.rttm.clone();
        }

        self.rttm
            .replace("SPEAKER file1 1", &format!("SPEAKER {file_id} 1"))
    }
}

pub(super) enum InferencePath {
    Sequential,
    Concurrent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum EmbeddingPath {
    Masked,
    Split,
    MultiMask,
}

// --- DecodedSegmentations methods ---

impl DecodedSegmentations {
    pub(super) fn nchunks(&self) -> usize {
        self.0.shape()[0]
    }

    pub(super) fn num_speakers(&self) -> usize {
        if self.0.ndim() < 3 {
            return 0;
        }
        self.0.shape()[2]
    }

    pub(super) fn speaker_count(&self, layout: &ChunkLayout) -> SpeakerCountTrack {
        let reconstructor = Reconstructor::new(self, &layout.start_frames, 0);
        reconstructor.speaker_count(layout.output_frames)
    }

    pub(super) fn extract_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embedding_path: EmbeddingPath,
    ) -> Result<ChunkEmbeddings, PipelineError> {
        let num_chunks = self.0.shape()[0];
        let num_speakers = self.0.shape()[2];
        let mut embeddings = Array3::<f32>::from_elem((num_chunks, num_speakers, 256), f32::NAN);

        match embedding_path {
            EmbeddingPath::MultiMask => {
                self.extract_multi_mask_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Split => {
                self.extract_split_embeddings(audio, emb_model, layout, &mut embeddings)?
            }
            EmbeddingPath::Masked => {
                if emb_model.prefers_chunk_embedding_path() {
                    for chunk_idx in 0..num_chunks {
                        let chunk_audio = layout.chunk_audio(audio, chunk_idx);
                        let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
                        let clean_masks = super::clean_masks(&chunk_segmentations);
                        let chunk_embeddings = emb_model.embed_chunk_speakers(
                            chunk_audio,
                            chunk_segmentations,
                            &clean_masks,
                        )?;
                        embeddings
                            .slice_mut(s![chunk_idx, .., ..])
                            .assign(&chunk_embeddings);
                    }
                } else {
                    self.extract_masked_embeddings(audio, emb_model, layout, &mut embeddings)?;
                }
            }
        }

        Ok(ChunkEmbeddings(embeddings))
    }

    fn extract_masked_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let mut pending = Vec::with_capacity(emb_model.primary_batch_size());

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = super::clean_masks(&chunk_segmentations);

            for speaker_idx in 0..self.0.shape()[2] {
                let mask = chunk_segmentations.column(speaker_idx);
                let activity: f32 = mask.iter().sum();
                if activity < MIN_SPEAKER_ACTIVITY {
                    continue;
                }

                pending.push(PendingEmbedding {
                    chunk_idx,
                    speaker_idx,
                    audio: chunk_audio,
                    mask: mask.to_vec(),
                    clean_mask: clean_masks.column(speaker_idx).to_vec(),
                });
                if pending.len() == emb_model.primary_batch_size() {
                    flush_embedding_batch(emb_model, &pending, embeddings)?;
                    pending.clear();
                }
            }
        }

        while !pending.is_empty() {
            let batch_len = emb_model.best_batch_len(pending.len());
            flush_embedding_batch(emb_model, &pending[..batch_len], embeddings)?;
            pending.drain(..batch_len);
        }

        Ok(())
    }

    fn extract_split_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.split_primary_batch_size();
        let num_speakers = self.0.shape()[2];
        let min_num_samples = emb_model.min_num_samples();

        let mut pending: Vec<PendingSplitEmbedding> = Vec::with_capacity(batch_size);
        let mut fbanks: Vec<Array2<f32>> = Vec::new();
        let mut tail_batches = 0usize;
        let mut active_items = 0usize;

        for chunk_idx in 0..self.0.shape()[0] {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbanks.push(fbank);
            let mut current_fbank_idx = fbanks.len() - 1;

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks = super::clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = super::select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    continue;
                };
                active_items += 1;
                pending.push(PendingSplitEmbedding {
                    chunk_idx,
                    speaker_idx,
                    fbank_idx: current_fbank_idx,
                    weights,
                });
                if pending.len() == batch_size {
                    flush_split_embedding_batch(emb_model, &pending, &fbanks, embeddings)?;
                    tail_batches += 1;
                    pending.clear();

                    if speaker_idx + 1 < num_speakers {
                        let kept_fbank = fbanks.swap_remove(current_fbank_idx);
                        fbanks.clear();
                        fbanks.push(kept_fbank);
                        current_fbank_idx = 0;
                    } else {
                        fbanks.clear();
                    }
                }
            }
        }

        if !pending.is_empty() {
            flush_split_embedding_batch(emb_model, &pending, &fbanks, embeddings)?;
            tail_batches += 1;
        }

        info!(
            batches = tail_batches,
            active_items,
            total_items = self.0.shape()[0] * num_speakers,
            "Split embeddings complete (fbank+tail streaming)"
        );

        Ok(())
    }

    fn extract_multi_mask_embeddings(
        &self,
        audio: &[f32],
        emb_model: &mut EmbeddingModel,
        layout: &ChunkLayout,
        embeddings: &mut Array3<f32>,
    ) -> Result<(), PipelineError> {
        let batch_size = emb_model.multi_mask_batch_size();
        let num_speakers = self.0.shape()[2];
        let num_chunks = self.0.shape()[0];
        let min_num_samples = emb_model.min_num_samples();

        let mut fbank_buffer: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        let mut masks_buffer: Vec<Vec<f32>> = Vec::with_capacity(batch_size * num_speakers);
        let mut chunk_indices: Vec<usize> = Vec::with_capacity(batch_size);
        let mut batches = 0usize;

        for chunk_idx in 0..num_chunks {
            let chunk_audio = layout.chunk_audio(audio, chunk_idx);
            let fbank = emb_model.compute_chunk_fbank(chunk_audio)?;
            fbank_buffer.push(fbank);
            chunk_indices.push(chunk_idx);

            let chunk_segmentations = self.0.slice(s![chunk_idx, .., ..]);
            let clean_masks_arr = super::clean_masks(&chunk_segmentations);

            for speaker_idx in 0..num_speakers {
                let Some(weights) = super::select_speaker_weights(
                    &chunk_segmentations,
                    &clean_masks_arr,
                    speaker_idx,
                    chunk_audio.len(),
                    min_num_samples,
                ) else {
                    masks_buffer.push(vec![0.0; 589]);
                    continue;
                };
                masks_buffer.push(weights);
            }

            if fbank_buffer.len() == batch_size {
                flush_multi_mask_batch(
                    emb_model,
                    &fbank_buffer,
                    &masks_buffer,
                    &chunk_indices,
                    num_speakers,
                    embeddings,
                )?;
                batches += 1;
                fbank_buffer.clear();
                masks_buffer.clear();
                chunk_indices.clear();
            }
        }

        if !fbank_buffer.is_empty() {
            flush_multi_mask_batch(
                emb_model,
                &fbank_buffer,
                &masks_buffer,
                &chunk_indices,
                num_speakers,
                embeddings,
            )?;
            batches += 1;
        }

        info!(
            batches,
            total_chunks = num_chunks,
            "Multi-mask embeddings complete"
        );

        Ok(())
    }
}

// --- Flush helpers ---

pub(super) fn flush_embedding_batch(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingEmbedding<'_>],
    embeddings: &mut Array3<f32>,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| MaskedEmbeddingInput {
            audio: item.audio,
            mask: &item.mask,
            clean_mask: Some(&item.clean_mask),
        })
        .collect();
    let batch_embeddings = emb_model.embed_batch(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        embeddings
            .slice_mut(s![item.chunk_idx, item.speaker_idx, ..])
            .assign(&batch_embeddings.row(batch_idx));
    }

    Ok(())
}

fn flush_split_embedding_batch(
    emb_model: &mut EmbeddingModel,
    pending: &[PendingSplitEmbedding],
    fbanks: &[Array2<f32>],
    embeddings: &mut Array3<f32>,
) -> Result<(), PipelineError> {
    let batch_inputs: Vec<_> = pending
        .iter()
        .map(|item| SplitTailInput {
            fbank: &fbanks[item.fbank_idx],
            weights: &item.weights,
        })
        .collect();
    let batch_embeddings = emb_model.embed_tail_batch_inputs(&batch_inputs)?;

    for (batch_idx, item) in pending.iter().enumerate() {
        embeddings
            .slice_mut(s![item.chunk_idx, item.speaker_idx, ..])
            .assign(&batch_embeddings.row(batch_idx));
    }

    Ok(())
}

fn flush_multi_mask_batch(
    emb_model: &mut EmbeddingModel,
    fbanks: &[Array2<f32>],
    masks: &[Vec<f32>],
    chunk_indices: &[usize],
    num_speakers: usize,
    embeddings: &mut Array3<f32>,
) -> Result<(), PipelineError> {
    let fbank_refs: Vec<_> = fbanks.iter().collect();
    let mask_refs: Vec<_> = masks.iter().map(|m| m.as_slice()).collect();
    let batch_embeddings = emb_model.embed_multi_mask_batch(&fbank_refs, &mask_refs)?;

    for (fbank_idx, &chunk_idx) in chunk_indices.iter().enumerate() {
        for speaker_idx in 0..num_speakers {
            let mask_idx = fbank_idx * num_speakers + speaker_idx;
            let is_active = masks[mask_idx].iter().any(|&v| v > 0.0);
            if !is_active {
                continue;
            }
            embeddings
                .slice_mut(s![chunk_idx, speaker_idx, ..])
                .assign(&batch_embeddings.row(mask_idx));
        }
    }

    Ok(())
}

// --- Audio helpers ---

pub(crate) fn chunk_audio_raw(
    audio: &[f32],
    step_samples: usize,
    window_samples: usize,
    chunk_idx: usize,
) -> &[f32] {
    let start = chunk_idx * step_samples;
    let end = (start + window_samples).min(audio.len());
    if start < audio.len() {
        &audio[start..end]
    } else {
        &[]
    }
}

pub(super) fn chunk_start_frames(num_chunks: usize, step_seconds: f64) -> Vec<usize> {
    (0..num_chunks)
        .map(|chunk_idx| {
            closest_frame(chunk_idx as f64 * step_seconds + 0.5 * FRAME_DURATION_SECONDS)
        })
        .collect()
}

pub(super) fn total_output_frames(num_chunks: usize, step_seconds: f64) -> usize {
    if num_chunks == 0 {
        return 0;
    }

    closest_frame(
        SEGMENTATION_WINDOW_SECONDS
            + (num_chunks - 1) as f64 * step_seconds
            + 0.5 * FRAME_DURATION_SECONDS,
    ) + 1
}

fn closest_frame(timestamp: f64) -> usize {
    ((timestamp - 0.5 * FRAME_DURATION_SECONDS) / FRAME_STEP_SECONDS).round() as usize
}
