use ndarray::{Array2, Array3, s};

use super::gpu::EmbeddedChunk;
use super::{
    ChunkEmbeddings, ChunkLayout, DecodedSegmentations, InferenceArtifacts, PipelineError,
    invariant_error,
};

pub(super) fn batch_embeddings(
    num_masks: usize,
    data: Vec<f32>,
    context: &str,
) -> Result<Array2<f32>, PipelineError> {
    Array2::from_shape_vec((num_masks, 256), data).map_err(|error| {
        invariant_error(format!(
            "{context} produced invalid embedding shape: {error}"
        ))
    })
}

pub(super) fn build_chunk_artifacts(
    step_seconds: f64,
    step_samples: usize,
    window_samples: usize,
    summary: super::EmbeddingSummary,
) -> Option<InferenceArtifacts> {
    if summary.num_chunks == 0 {
        return None;
    }
    Some(InferenceArtifacts {
        layout: ChunkLayout::new(
            step_seconds,
            step_samples,
            window_samples,
            summary.num_chunks,
        ),
        segmentations: DecodedSegmentations(summary.segmentations),
        embeddings: ChunkEmbeddings(summary.embeddings),
    })
}

pub(super) struct FileCollector {
    seg_array: Array3<f32>,
    emb_array: Array3<f32>,
    max_slot_used: usize,
    chunks_received: usize,
    expected_chunks: usize,
}

impl FileCollector {
    pub(super) fn new(
        max_slots: usize,
        num_frames: usize,
        num_speakers: usize,
        expected_chunks: usize,
    ) -> Self {
        Self {
            seg_array: Array3::zeros((max_slots, num_frames, num_speakers)),
            emb_array: Array3::from_elem((max_slots, num_speakers, 256), f32::NAN),
            max_slot_used: 0,
            chunks_received: 0,
            expected_chunks,
        }
    }

    pub(super) fn add(
        &mut self,
        local_start: usize,
        chunk_win_capacity: usize,
        num_speakers: usize,
        embedded: EmbeddedChunk,
    ) -> Result<(), PipelineError> {
        let batch_emb =
            batch_embeddings(embedded.num_masks, embedded.data, "batch chunk embedding")?;

        for &(local, speaker_idx) in &embedded.active {
            let slot = local_start * chunk_win_capacity + local;
            if slot < self.emb_array.shape()[0] {
                let mask_idx = local * num_speakers + speaker_idx;
                self.emb_array
                    .slice_mut(s![slot, speaker_idx, ..])
                    .assign(&batch_emb.row(mask_idx));
            }
        }

        for (local, decoded) in embedded.decoded_chunk.into_iter().enumerate() {
            let slot = local_start * chunk_win_capacity + local;
            if slot < self.seg_array.shape()[0] {
                self.seg_array.slice_mut(s![slot, .., ..]).assign(&decoded);
                self.max_slot_used = self.max_slot_used.max(slot + 1);
            }
        }

        self.chunks_received += 1;
        Ok(())
    }

    pub(super) fn is_complete(&self) -> bool {
        self.chunks_received >= self.expected_chunks
    }

    pub(super) fn into_artifacts(
        self,
        step_seconds: f64,
        step_samples: usize,
        window_samples: usize,
    ) -> Option<InferenceArtifacts> {
        if self.max_slot_used == 0 {
            return None;
        }
        let n = self.max_slot_used;
        Some(InferenceArtifacts {
            layout: ChunkLayout::new(step_seconds, step_samples, window_samples, n),
            segmentations: DecodedSegmentations(self.seg_array.slice_move(s![..n, .., ..])),
            embeddings: ChunkEmbeddings(self.emb_array.slice_move(s![..n, .., ..])),
        })
    }
}
