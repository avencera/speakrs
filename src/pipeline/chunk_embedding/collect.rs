use std::collections::BTreeSet;

use ndarray::{Array2, Array3, s};

use super::gpu::EmbeddedChunk;
use super::{
    ChunkEmbeddings, ChunkLayout, DecodedSegmentations, InferenceArtifacts, PipelineError,
    invariant_error,
};
#[cfg(feature = "_metrics")]
use crate::pipeline::types::InternalInferenceStageTimings;

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
    #[cfg(feature = "_metrics")] stage_timings: InternalInferenceStageTimings,
) -> Option<InferenceArtifacts> {
    if summary.num_chunks == 0 {
        return None;
    }
    InferenceArtifacts::try_new(
        ChunkLayout::new(
            step_seconds,
            step_samples,
            window_samples,
            summary.num_chunks,
        ),
        DecodedSegmentations(summary.segmentations),
        ChunkEmbeddings(summary.embeddings),
        #[cfg(feature = "_metrics")]
        Some(stage_timings),
    )
    .ok()
}

pub(super) struct FileCollector {
    seg_array: Array3<f32>,
    emb_array: Array3<f32>,
    max_slot_used: usize,
    expected_chunks: usize,
    received_groups: BTreeSet<usize>,
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
            expected_chunks,
            received_groups: BTreeSet::new(),
        }
    }

    pub(super) fn add(
        &mut self,
        chunk_win_capacity: usize,
        num_speakers: usize,
        embedded: EmbeddedChunk,
    ) -> Result<(), PipelineError> {
        let group = embedded.window_start / chunk_win_capacity;
        if !self.received_groups.insert(embedded.window_start) {
            return Err(invariant_error(format!(
                "duplicate chunk group {}",
                embedded.window_start
            )));
        }
        if group >= self.expected_chunks {
            return Err(invariant_error(format!(
                "chunk group {} is outside expected range {}",
                embedded.window_start, self.expected_chunks
            )));
        }
        let batch_emb =
            batch_embeddings(embedded.num_masks, embedded.data, "batch chunk embedding")?;

        for &(local, speaker_idx) in &embedded.active {
            let slot = embedded.window_start + local;
            if slot >= self.emb_array.shape()[0] {
                return Err(invariant_error(format!(
                    "chunk slot {slot} is outside collector capacity {}",
                    self.emb_array.shape()[0]
                )));
            }
            let mask_idx = local * num_speakers + speaker_idx;
            self.emb_array
                .slice_mut(s![slot, speaker_idx, ..])
                .assign(&batch_emb.row(mask_idx));
        }

        for (local, decoded) in embedded.decoded.into_iter().enumerate() {
            let slot = embedded.window_start + local;
            if slot >= self.seg_array.shape()[0] {
                return Err(invariant_error(format!(
                    "chunk slot {slot} is outside collector capacity {}",
                    self.seg_array.shape()[0]
                )));
            }
            self.seg_array.slice_mut(s![slot, .., ..]).assign(&decoded);
            self.max_slot_used = self.max_slot_used.max(slot + 1);
        }

        Ok(())
    }

    pub(super) fn is_complete(&self) -> bool {
        self.received_groups.len() >= self.expected_chunks
    }

    pub(super) fn into_artifacts(
        self,
        step_seconds: f64,
        step_samples: usize,
        window_samples: usize,
    ) -> Option<InferenceArtifacts> {
        if self.max_slot_used == 0 || !self.is_complete() {
            return None;
        }
        let n = self.max_slot_used;
        InferenceArtifacts::try_new(
            ChunkLayout::new(step_seconds, step_samples, window_samples, n),
            DecodedSegmentations(self.seg_array.slice_move(s![..n, .., ..])),
            ChunkEmbeddings(self.emb_array.slice_move(s![..n, .., ..])),
            #[cfg(feature = "_metrics")]
            None,
        )
        .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::embedding::EMBEDDING_WIDTH;
    use ndarray::Array2;

    fn silent_group(
        window_start: usize,
        windows: usize,
        frames: usize,
        speakers: usize,
    ) -> EmbeddedChunk {
        EmbeddedChunk {
            file_index: 0,
            window_start,
            decoded: vec![Array2::zeros((frames, speakers)); windows],
            data: vec![0.0; windows * speakers * EMBEDDING_WIDTH],
            active: Vec::new(),
            num_masks: windows * speakers,
            predict_us: 0,
        }
    }

    #[test]
    fn collector_rejects_duplicate_groups() {
        let mut collector = FileCollector::new(4, 2, 3, 1);
        collector.add(2, 3, silent_group(0, 2, 2, 3)).unwrap();
        assert!(collector.add(2, 3, silent_group(0, 2, 2, 3)).is_err());
    }

    #[test]
    fn collector_rejects_out_of_range_groups() {
        let mut collector = FileCollector::new(4, 2, 3, 1);
        assert!(collector.add(2, 3, silent_group(2, 2, 2, 3)).is_err());
    }

    #[test]
    fn collector_is_complete_only_after_expected_groups() {
        let mut collector = FileCollector::new(4, 2, 3, 2);
        collector.add(2, 3, silent_group(0, 2, 2, 3)).unwrap();
        assert!(!collector.is_complete());
        collector.add(2, 3, silent_group(2, 2, 2, 3)).unwrap();
        assert!(collector.is_complete());
        assert!(collector.into_artifacts(1.0, 16_000, 160_000).is_some());
    }
}
