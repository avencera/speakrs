use std::collections::BTreeSet;

use ndarray::{Array2, Array3, s};

use super::gpu::EmbeddedChunk;
use super::{
    ChunkEmbeddings, ChunkLayout, DecodedSegmentations, InferenceArtifacts, PipelineError,
    invariant_error,
};
use crate::inference::embedding::EMBEDDING_WIDTH;
#[cfg(feature = "_metrics")]
use crate::pipeline::types::InternalInferenceStageTimings;

pub(super) fn batch_embeddings(
    num_masks: usize,
    data: Vec<f32>,
    context: &str,
) -> Result<Array2<f32>, PipelineError> {
    let expected_values = num_masks.checked_mul(EMBEDDING_WIDTH).ok_or_else(|| {
        invariant_error(format!(
            "{context} embedding geometry exceeded addressable memory"
        ))
    })?;
    if data.len() != expected_values {
        return Err(invariant_error(format!(
            "{context} expected {expected_values} embedding values for {num_masks} masks, got {}",
            data.len()
        )));
    }

    Array2::from_shape_vec((num_masks, EMBEDDING_WIDTH), data).map_err(|error| {
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
) -> Result<InferenceArtifacts, PipelineError> {
    if summary.num_chunks == 0 {
        return Err(invariant_error(
            "chunk execution completed without any inference windows",
        ));
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
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CollectionPlan {
    file_index: usize,
    total_windows: usize,
    group_capacity: usize,
    group_count: usize,
    final_group_windows: usize,
    num_speakers: usize,
}

impl CollectionPlan {
    pub(super) fn new(
        file_index: usize,
        total_windows: usize,
        group_capacity: usize,
        num_speakers: usize,
    ) -> Result<Self, PipelineError> {
        if total_windows == 0 {
            return Err(invariant_error(
                "chunk collection requires at least one expected window",
            ));
        }
        if group_capacity == 0 {
            return Err(invariant_error(
                "chunk collection group capacity must be greater than zero",
            ));
        }
        if num_speakers == 0 {
            return Err(invariant_error(
                "chunk collection speaker count must be greater than zero",
            ));
        }

        let complete_groups = total_windows / group_capacity;
        let remainder = total_windows % group_capacity;
        let group_count = complete_groups + usize::from(remainder != 0);
        let final_group_windows = if remainder == 0 {
            group_capacity
        } else {
            remainder
        };

        Ok(Self {
            file_index,
            total_windows,
            group_capacity,
            group_count,
            final_group_windows,
            num_speakers,
        })
    }

    fn group_for_start(self, window_start: usize) -> Result<usize, PipelineError> {
        if !window_start.is_multiple_of(self.group_capacity) {
            return Err(invariant_error(format!(
                "chunk group start {window_start} is not aligned to capacity {}",
                self.group_capacity
            )));
        }

        let group = window_start / self.group_capacity;
        if group >= self.group_count {
            return Err(invariant_error(format!(
                "chunk group start {window_start} is outside {} expected groups",
                self.group_count
            )));
        }

        Ok(group)
    }

    fn windows_in_group(self, group: usize) -> usize {
        if group + 1 == self.group_count {
            self.final_group_windows
        } else {
            self.group_capacity
        }
    }

    pub(super) const fn group_capacity(self) -> usize {
        self.group_capacity
    }

    pub(super) const fn group_count(self) -> usize {
        self.group_count
    }
}

pub(super) struct CollectedChunks {
    pub(super) segmentations: Array3<f32>,
    pub(super) embeddings: Array3<f32>,
    pub(super) num_windows: usize,
}

pub(super) struct FileCollector {
    plan: CollectionPlan,
    seg_array: Option<Array3<f32>>,
    emb_array: Option<Array3<f32>>,
    num_frames: Option<usize>,
    received_groups: Vec<bool>,
}

impl FileCollector {
    pub(super) fn new(plan: CollectionPlan) -> Self {
        Self {
            plan,
            seg_array: None,
            emb_array: None,
            num_frames: None,
            received_groups: vec![false; plan.group_count],
        }
    }

    pub(super) fn add(&mut self, embedded: EmbeddedChunk) -> Result<(), PipelineError> {
        let group = self.validate_identity(&embedded)?;
        let num_frames = self.validate_segmentations(&embedded)?;
        self.validate_embedding_geometry(&embedded)?;
        self.validate_active_indices(&embedded)?;
        let batch_emb =
            batch_embeddings(embedded.num_masks, embedded.data, "chunk embedding payload")?;

        let seg_array = self.seg_array.get_or_insert_with(|| {
            Array3::zeros((self.plan.total_windows, num_frames, self.plan.num_speakers))
        });
        let emb_array = self.emb_array.get_or_insert_with(|| {
            Array3::from_elem(
                (
                    self.plan.total_windows,
                    self.plan.num_speakers,
                    EMBEDDING_WIDTH,
                ),
                f32::NAN,
            )
        });

        for &(local, speaker_idx) in &embedded.active {
            let slot = embedded.window_start + local;
            let mask_idx = local * self.plan.num_speakers + speaker_idx;
            emb_array
                .slice_mut(s![slot, speaker_idx, ..])
                .assign(&batch_emb.row(mask_idx));
        }

        for (local, decoded) in embedded.decoded.into_iter().enumerate() {
            let slot = embedded.window_start + local;
            seg_array.slice_mut(s![slot, .., ..]).assign(&decoded);
        }

        self.num_frames = Some(num_frames);
        self.received_groups[group] = true;
        Ok(())
    }

    fn validate_identity(&self, embedded: &EmbeddedChunk) -> Result<usize, PipelineError> {
        if embedded.file_index != self.plan.file_index {
            return Err(invariant_error(format!(
                "chunk payload file {} does not match collector file {}",
                embedded.file_index, self.plan.file_index
            )));
        }

        let group = self.plan.group_for_start(embedded.window_start)?;
        if self.received_groups[group] {
            return Err(invariant_error(format!(
                "duplicate chunk group {group} for file {}",
                self.plan.file_index
            )));
        }

        let expected_windows = self.plan.windows_in_group(group);
        if embedded.decoded.len() != expected_windows {
            return Err(invariant_error(format!(
                "chunk group {group} for file {} expected {expected_windows} decoded windows, got {}",
                self.plan.file_index,
                embedded.decoded.len()
            )));
        }

        Ok(group)
    }

    fn validate_segmentations(&self, embedded: &EmbeddedChunk) -> Result<usize, PipelineError> {
        let Some(first) = embedded.decoded.first() else {
            return Err(invariant_error("chunk payload has no decoded windows"));
        };
        let (num_frames, num_speakers) = first.dim();
        if num_frames == 0 {
            return Err(invariant_error(
                "chunk payload segmentation frame count must be greater than zero",
            ));
        }
        if num_speakers != self.plan.num_speakers {
            return Err(invariant_error(format!(
                "chunk payload has {num_speakers} speakers, expected {}",
                self.plan.num_speakers
            )));
        }
        if let Some(expected_frames) = self.num_frames
            && num_frames != expected_frames
        {
            return Err(invariant_error(format!(
                "chunk payload has {num_frames} frames, expected {expected_frames}"
            )));
        }
        if let Some((window, decoded)) = embedded
            .decoded
            .iter()
            .enumerate()
            .find(|(_, decoded)| decoded.dim() != (num_frames, self.plan.num_speakers))
        {
            return Err(invariant_error(format!(
                "decoded window {window} has shape {:?}, expected ({num_frames}, {})",
                decoded.dim(),
                self.plan.num_speakers
            )));
        }

        Ok(num_frames)
    }

    fn validate_embedding_geometry(&self, embedded: &EmbeddedChunk) -> Result<(), PipelineError> {
        let required_masks = embedded
            .decoded
            .len()
            .checked_mul(self.plan.num_speakers)
            .ok_or_else(|| invariant_error("chunk payload mask count overflowed"))?;
        if embedded.num_masks < required_masks {
            return Err(invariant_error(format!(
                "chunk payload provides {} masks, but {required_masks} are required",
                embedded.num_masks
            )));
        }

        let expected_values = embedded
            .num_masks
            .checked_mul(EMBEDDING_WIDTH)
            .ok_or_else(|| invariant_error("chunk payload embedding size overflowed"))?;
        if embedded.data.len() != expected_values {
            return Err(invariant_error(format!(
                "chunk payload expected {expected_values} embedding values, got {}",
                embedded.data.len()
            )));
        }

        Ok(())
    }

    fn validate_active_indices(&self, embedded: &EmbeddedChunk) -> Result<(), PipelineError> {
        let mut active = BTreeSet::new();
        for &(local, speaker_idx) in &embedded.active {
            if local >= embedded.decoded.len() {
                return Err(invariant_error(format!(
                    "active window index {local} is outside payload length {}",
                    embedded.decoded.len()
                )));
            }
            if speaker_idx >= self.plan.num_speakers {
                return Err(invariant_error(format!(
                    "active speaker index {speaker_idx} is outside speaker count {}",
                    self.plan.num_speakers
                )));
            }
            if !active.insert((local, speaker_idx)) {
                return Err(invariant_error(format!(
                    "duplicate active index ({local}, {speaker_idx})"
                )));
            }

            let mask_idx = local
                .checked_mul(self.plan.num_speakers)
                .and_then(|offset| offset.checked_add(speaker_idx))
                .ok_or_else(|| invariant_error("active mask index overflowed"))?;
            if mask_idx >= embedded.num_masks {
                return Err(invariant_error(format!(
                    "active mask index {mask_idx} is outside mask count {}",
                    embedded.num_masks
                )));
            }
        }

        Ok(())
    }

    pub(super) fn is_complete(&self) -> bool {
        self.received_groups.iter().all(|received| *received)
    }

    pub(super) fn finish(self) -> Result<CollectedChunks, PipelineError> {
        if !self.is_complete() {
            let missing: Vec<_> = self
                .received_groups
                .iter()
                .enumerate()
                .filter_map(|(group, received)| (!received).then_some(group))
                .collect();
            return Err(invariant_error(format!(
                "file {} is missing chunk groups {missing:?}",
                self.plan.file_index
            )));
        }

        let segmentations = self.seg_array.ok_or_else(|| {
            invariant_error(format!(
                "file {} completed without segmentations",
                self.plan.file_index
            ))
        })?;
        let embeddings = self.emb_array.ok_or_else(|| {
            invariant_error(format!(
                "file {} completed without embeddings",
                self.plan.file_index
            ))
        })?;

        Ok(CollectedChunks {
            segmentations,
            embeddings,
            num_windows: self.plan.total_windows,
        })
    }

    pub(super) fn into_artifacts(
        self,
        step_seconds: f64,
        step_samples: usize,
        window_samples: usize,
    ) -> Result<InferenceArtifacts, PipelineError> {
        let collected = self.finish()?;
        InferenceArtifacts::try_new(
            ChunkLayout::new(
                step_seconds,
                step_samples,
                window_samples,
                collected.num_windows,
            ),
            DecodedSegmentations(collected.segmentations),
            ChunkEmbeddings(collected.embeddings),
            #[cfg(feature = "_metrics")]
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn silent_group(
        file_index: usize,
        window_start: usize,
        windows: usize,
        frames: usize,
        speakers: usize,
    ) -> EmbeddedChunk {
        EmbeddedChunk {
            file_index,
            window_start,
            decoded: vec![Array2::zeros((frames, speakers)); windows],
            data: vec![0.0; windows * speakers * EMBEDDING_WIDTH],
            active: Vec::new(),
            num_masks: windows * speakers,
            predict_us: 0,
        }
    }

    fn collector(total_windows: usize, capacity: usize) -> FileCollector {
        FileCollector::new(CollectionPlan::new(0, total_windows, capacity, 3).unwrap())
    }

    #[test]
    fn collector_rejects_duplicate_groups() {
        let mut collector = collector(4, 2);
        collector.add(silent_group(0, 0, 2, 2, 3)).unwrap();
        assert!(collector.add(silent_group(0, 0, 2, 2, 3)).is_err());
    }

    #[test]
    fn collector_rejects_overlapping_misaligned_groups() {
        let mut collector = collector(4, 2);
        collector.add(silent_group(0, 0, 2, 2, 3)).unwrap();
        assert!(collector.add(silent_group(0, 1, 2, 2, 3)).is_err());
        assert!(!collector.is_complete());
    }

    #[test]
    fn collector_accepts_exact_out_of_order_groups() {
        let mut collector = collector(4, 2);
        collector.add(silent_group(0, 2, 2, 2, 3)).unwrap();
        collector.add(silent_group(0, 0, 2, 2, 3)).unwrap();
        assert!(collector.is_complete());
        assert!(collector.finish().is_ok());
    }

    #[test]
    fn collector_requires_exact_final_group_length() {
        let mut collector = collector(3, 2);
        collector.add(silent_group(0, 0, 2, 2, 3)).unwrap();
        assert!(collector.add(silent_group(0, 2, 2, 2, 3)).is_err());
        collector.add(silent_group(0, 2, 1, 2, 3)).unwrap();
        assert!(collector.finish().is_ok());
    }

    #[test]
    fn collector_reports_missing_groups_at_finish() {
        let mut collector = collector(4, 2);
        collector.add(silent_group(0, 0, 2, 2, 3)).unwrap();
        assert!(collector.finish().is_err());
    }

    #[test]
    fn collector_rejects_wrong_file_and_payload_shapes() {
        let mut collector = collector(2, 2);
        assert!(collector.add(silent_group(1, 0, 2, 2, 3)).is_err());
        assert!(collector.add(silent_group(0, 0, 2, 2, 2)).is_err());

        let mut payload = silent_group(0, 0, 2, 2, 3);
        payload.decoded[1] = Array2::zeros((1, 3));
        assert!(collector.add(payload).is_err());

        let mut payload = silent_group(0, 0, 2, 2, 3);
        payload.data.pop();
        assert!(collector.add(payload).is_err());
    }

    #[test]
    fn collector_rejects_empty_payload_and_changed_frame_shape() {
        let mut collector = collector(3, 2);
        let mut empty = silent_group(0, 0, 2, 2, 3);
        empty.decoded.clear();
        assert!(collector.add(empty).is_err());

        collector.add(silent_group(0, 0, 2, 2, 3)).unwrap();
        assert!(collector.add(silent_group(0, 2, 1, 3, 3)).is_err());
    }

    #[test]
    fn collector_rejects_invalid_active_indices_without_consuming_group() {
        let mut collector = collector(2, 2);
        let mut payload = silent_group(0, 0, 2, 2, 3);
        payload.active.push((2, 0));
        assert!(collector.add(payload).is_err());

        collector.add(silent_group(0, 0, 2, 2, 3)).unwrap();
        assert!(collector.finish().is_ok());
    }

    #[test]
    fn collection_plan_rejects_empty_or_invalid_geometry() {
        assert!(CollectionPlan::new(0, 0, 2, 3).is_err());
        assert!(CollectionPlan::new(0, 2, 0, 3).is_err());
        assert!(CollectionPlan::new(0, 2, 2, 0).is_err());
    }

    #[cfg(not(feature = "_metrics"))]
    #[test]
    fn artifact_construction_propagates_invalid_shapes() {
        let summary = super::super::EmbeddingSummary {
            segmentations: Array3::zeros((1, 2, 3)),
            embeddings: Array3::zeros((2, 3, EMBEDDING_WIDTH)),
            num_chunks: 1,
            gpu_predict_us: 0,
            prep_fbank_us: 0,
            prep_mask_us: 0,
        };

        assert!(build_chunk_artifacts(1.0, 16_000, 160_000, summary).is_err());
    }

    #[cfg(feature = "_metrics")]
    #[test]
    fn artifact_construction_propagates_invalid_shapes() {
        let summary = super::super::EmbeddingSummary {
            segmentations: Array3::zeros((1, 2, 3)),
            embeddings: Array3::zeros((2, 3, EMBEDDING_WIDTH)),
            num_chunks: 1,
            gpu_predict_us: 0,
            prep_fbank_us: 0,
            prep_mask_us: 0,
        };
        let timings = InternalInferenceStageTimings {
            segmentation_seconds: 0.0,
            embedding_seconds: 0.0,
            prediction_seconds: 0.0,
            filterbank_preparation_seconds: 0.0,
            mask_preparation_seconds: 0.0,
            total_seconds: 0.0,
            chunk_count: 1,
            pipelined: false,
        };

        assert!(build_chunk_artifacts(1.0, 16_000, 160_000, summary, timings).is_err());
    }
}
