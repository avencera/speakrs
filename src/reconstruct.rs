use ndarray::{Array2, s};

use crate::pipeline::{
    ChunkSpeakerClusters, DecodedSegmentations, DiscreteDiarization, FrameActivations,
    SpeakerCountTrack,
};

pub struct Reconstructor<'a> {
    segmentations: &'a DecodedSegmentations,
    hard_clusters: Option<&'a ChunkSpeakerClusters>,
    start_frames: &'a [usize],
    warmup_frames: usize,
}

impl<'a> Reconstructor<'a> {
    pub fn new(
        segmentations: &'a DecodedSegmentations,
        start_frames: &'a [usize],
        warmup_frames: usize,
    ) -> Self {
        Self {
            segmentations,
            hard_clusters: None,
            start_frames,
            warmup_frames,
        }
    }

    pub fn with_clusters(
        segmentations: &'a DecodedSegmentations,
        hard_clusters: &'a ChunkSpeakerClusters,
        start_frames: &'a [usize],
        warmup_frames: usize,
    ) -> Self {
        Self {
            segmentations,
            hard_clusters: Some(hard_clusters),
            start_frames,
            warmup_frames,
        }
    }

    pub fn speaker_count(&self, output_frames: usize) -> SpeakerCountTrack {
        let num_chunks = self.segmentations.shape()[0];
        if num_chunks == 0 {
            return SpeakerCountTrack(Vec::new());
        }

        let num_frames = self.segmentations.shape()[1];
        let warmup_end = num_frames.saturating_sub(self.warmup_frames);
        let mut numerator = vec![0.0f32; output_frames];
        let mut denominator = vec![0.0f32; output_frames];

        for (chunk_idx, &start_frame) in self.start_frames.iter().enumerate().take(num_chunks) {
            for frame_idx in self.warmup_frames..warmup_end {
                let out_frame = start_frame + frame_idx;
                if out_frame >= output_frames {
                    continue;
                }

                numerator[out_frame] += self
                    .segmentations
                    .slice(s![chunk_idx, frame_idx, ..])
                    .iter()
                    .sum::<f32>();
                denominator[out_frame] += 1.0;
            }
        }

        SpeakerCountTrack(
            numerator
                .into_iter()
                .zip(denominator)
                .map(|(sum, weight)| {
                    if weight == 0.0 {
                        0
                    } else {
                        round_ties_even(sum / weight).max(0.0) as usize
                    }
                })
                .collect(),
        )
    }

    pub(crate) fn frame_activations(&self, speaker_count: &SpeakerCountTrack) -> FrameActivations {
        let Some(hard_clusters) = self.hard_clusters else {
            return FrameActivations(Array2::zeros((speaker_count.len(), 0)));
        };
        let num_chunks = self.segmentations.shape()[0];
        let num_frames = self.segmentations.shape()[1];
        let num_clusters = hard_clusters
            .iter()
            .copied()
            .filter(|cluster| *cluster >= 0)
            .max()
            .map_or(0, |cluster| cluster as usize + 1);
        let warmup_end = num_frames.saturating_sub(self.warmup_frames);
        let mut activations = Array2::<f32>::zeros((speaker_count.len(), num_clusters));

        for (chunk_idx, &start_frame) in self.start_frames.iter().enumerate().take(num_chunks) {
            let chunk_labels = hard_clusters.row(chunk_idx);
            let chunk_segmentations = self.segmentations.slice(s![chunk_idx, .., ..]);
            let local_cluster_mapping = build_cluster_mapping(&chunk_labels, num_clusters);

            for (cluster_idx, local_indices) in local_cluster_mapping.iter().enumerate() {
                if local_indices.is_empty() {
                    continue;
                }

                for frame_idx in self.warmup_frames..warmup_end {
                    let out_frame = start_frame + frame_idx;
                    if out_frame >= speaker_count.len() {
                        continue;
                    }

                    let mut score = 0.0f32;
                    for &local_idx in local_indices {
                        score = score.max(chunk_segmentations[[frame_idx, local_idx]]);
                    }
                    activations[[out_frame, cluster_idx]] += score;
                }
            }
        }

        let max_speakers_per_frame = speaker_count.iter().copied().max().unwrap_or(0);
        if activations.ncols() < max_speakers_per_frame {
            let mut padded = Array2::<f32>::zeros((activations.nrows(), max_speakers_per_frame));
            padded
                .slice_mut(s![.., ..activations.ncols()])
                .assign(&activations);
            activations = padded;
        }

        FrameActivations(activations)
    }

    pub fn reconstruct(&self, speaker_count: &SpeakerCountTrack) -> DiscreteDiarization {
        let activations = self.frame_activations(speaker_count);
        let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
        for (frame_idx, &count) in speaker_count.iter().enumerate() {
            for speaker_idx in top_k_indices(&activations, frame_idx, count) {
                discrete[[frame_idx, speaker_idx]] = 1.0;
            }
        }
        DiscreteDiarization(discrete)
    }

    pub fn reconstruct_smoothed(
        &self,
        speaker_count: &SpeakerCountTrack,
        epsilon: f32,
    ) -> DiscreteDiarization {
        let activations = self.frame_activations(speaker_count);
        let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
        let mut previous_speakers: Vec<usize> = Vec::new();

        for (frame_idx, &count) in speaker_count.iter().enumerate() {
            let current_speakers =
                top_k_indices_smoothed(&activations, frame_idx, count, &previous_speakers, epsilon);
            for &speaker_idx in &current_speakers {
                discrete[[frame_idx, speaker_idx]] = 1.0;
            }
            previous_speakers = current_speakers;
        }

        DiscreteDiarization(discrete)
    }
}

/// Zero out all but the highest-scoring speaker in each frame, making activations exclusive
pub fn make_exclusive(activations: &mut Array2<f32>) {
    for mut row in activations.rows_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val == 0.0 {
            continue;
        }

        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        for (column_idx, value) in row.iter_mut().enumerate() {
            if column_idx != argmax {
                *value = 0.0;
            }
        }
    }
}

fn build_cluster_mapping(
    chunk_labels: &ndarray::ArrayView1<i32>,
    num_clusters: usize,
) -> Vec<Vec<usize>> {
    let mut mapping = vec![Vec::new(); num_clusters];
    for (local_idx, &label) in chunk_labels.iter().enumerate() {
        if label >= 0 {
            mapping[label as usize].push(local_idx);
        }
    }
    mapping
}

fn top_k_indices(matrix: &Array2<f32>, frame_idx: usize, k: usize) -> Vec<usize> {
    let num_columns = matrix.ncols();
    if k >= num_columns {
        return (0..num_columns).collect();
    }

    let mut indexed: Vec<(usize, f32)> = (0..num_columns)
        .map(|column_idx| (column_idx, matrix[[frame_idx, column_idx]]))
        .collect();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));

    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

fn top_k_indices_smoothed(
    matrix: &Array2<f32>,
    frame_idx: usize,
    k: usize,
    previous_speakers: &[usize],
    epsilon: f32,
) -> Vec<usize> {
    let num_columns = matrix.ncols();
    if k >= num_columns {
        return (0..num_columns).collect();
    }

    let mut indexed: Vec<(usize, f32)> = (0..num_columns)
        .map(|column_idx| (column_idx, matrix[[frame_idx, column_idx]]))
        .collect();

    indexed.sort_by(|left, right| {
        let score_diff = right.1 - left.1;
        if score_diff.abs() < epsilon {
            let left_was_active = previous_speakers.contains(&left.0);
            let right_was_active = previous_speakers.contains(&right.0);
            right_was_active.cmp(&left_was_active)
        } else {
            right.1.total_cmp(&left.1)
        }
    });

    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

fn round_ties_even(value: f32) -> f32 {
    let lower = value.floor();
    let fraction = value - lower;
    let epsilon = 1e-6;

    if fraction < 0.5 - epsilon {
        return lower;
    }

    if fraction > 0.5 + epsilon {
        return value.ceil();
    }

    if lower as i64 % 2 == 0 {
        lower
    } else {
        lower + 1.0
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;
    use crate::pipeline::{ChunkSpeakerClusters, DecodedSegmentations};

    #[test]
    fn speaker_count_rounds_overlap_added_sum() {
        let segmentations = DecodedSegmentations(array![
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ]);
        let reconstructor = Reconstructor::new(&segmentations, &[0, 1], 0);

        let count = reconstructor.speaker_count(4);

        assert_eq!(&*count, &[1, 1, 1, 1]);
    }

    #[test]
    fn reconstruct_selects_top_k_per_frame() {
        let segmentations =
            DecodedSegmentations(array![[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.2, 0.8]]]);
        let hard_clusters = ChunkSpeakerClusters(array![[0, 1], [0, 1]]);
        let reconstructor =
            Reconstructor::with_clusters(&segmentations, &hard_clusters, &[0, 1], 0);
        let speaker_count = SpeakerCountTrack(vec![1, 1, 1]);

        let result = reconstructor.reconstruct(&speaker_count);

        let expected: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        assert_eq!(&*result, &expected);
    }
}
