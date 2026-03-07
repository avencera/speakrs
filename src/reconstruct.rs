use ndarray::{Array2, Array3, Axis, s};

use crate::aggregate::{AggregateOptions, aggregate_at};

pub fn speaker_count(
    binarized_segmentations: &Array3<f32>,
    start_frames: &[usize],
    warmup_frames: usize,
    output_frames: usize,
) -> Vec<usize> {
    let chunk_count = binarized_segmentations
        .sum_axis(Axis(2))
        .insert_axis(Axis(2));
    let aggregated = aggregate_at(
        &chunk_count,
        start_frames,
        AggregateOptions {
            warmup_left: warmup_frames,
            warmup_right: warmup_frames,
            output_frames: Some(output_frames),
            ..Default::default()
        },
    );

    aggregated
        .column(0)
        .iter()
        .map(|value| round_ties_even(*value).max(0.0) as usize)
        .collect()
}

pub fn reconstruct(
    segmentations: &Array3<f32>,
    hard_clusters: &Array2<i32>,
    speaker_count: &[usize],
    start_frames: &[usize],
    warmup_frames: usize,
) -> Array2<f32> {
    let num_chunks = segmentations.shape()[0];
    let num_frames = segmentations.shape()[1];
    let num_clusters = hard_clusters
        .iter()
        .copied()
        .filter(|cluster| *cluster >= 0)
        .max()
        .map_or(0, |cluster| cluster as usize + 1);
    let mut clustered_segmentations = Array3::<f32>::zeros((num_chunks, num_frames, num_clusters));

    for chunk_idx in 0..num_chunks {
        let chunk_labels = hard_clusters.row(chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);

        for cluster_idx in 0..num_clusters {
            let local_indices: Vec<usize> = chunk_labels
                .iter()
                .enumerate()
                .filter_map(|(local_idx, &label)| {
                    (label == cluster_idx as i32).then_some(local_idx)
                })
                .collect();

            if local_indices.is_empty() {
                continue;
            }

            for frame_idx in 0..num_frames {
                let mut score = 0.0f32;
                for &local_idx in &local_indices {
                    score = score.max(chunk_segmentations[[frame_idx, local_idx]]);
                }
                clustered_segmentations[[chunk_idx, frame_idx, cluster_idx]] = score;
            }
        }
    }

    let mut activations = aggregate_at(
        &clustered_segmentations,
        start_frames,
        AggregateOptions {
            skip_average: true,
            warmup_left: warmup_frames,
            warmup_right: warmup_frames,
            output_frames: Some(speaker_count.len()),
            ..Default::default()
        },
    );

    let max_speakers_per_frame = speaker_count.iter().copied().max().unwrap_or(0);
    if activations.ncols() < max_speakers_per_frame {
        let mut padded = Array2::<f32>::zeros((activations.nrows(), max_speakers_per_frame));
        padded
            .slice_mut(s![.., ..activations.ncols()])
            .assign(&activations);
        activations = padded;
    }

    let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
    for (frame_idx, &count) in speaker_count.iter().enumerate() {
        for speaker_idx in top_k_indices(&activations, frame_idx, count) {
            discrete[[frame_idx, speaker_idx]] = 1.0;
        }
    }

    discrete
}

pub fn make_exclusive(activations: &mut Array2<f32>) {
    for mut row in activations.rows_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val == 0.0 {
            continue;
        }

        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        for (idx, value) in row.iter_mut().enumerate() {
            if idx != argmax {
                *value = 0.0;
            }
        }
    }
}

fn top_k_indices(matrix: &Array2<f32>, frame_idx: usize, k: usize) -> Vec<usize> {
    let ncols = matrix.ncols();
    if k >= ncols {
        return (0..ncols).collect();
    }

    let mut indexed: Vec<(usize, f32)> = (0..ncols)
        .map(|col_idx| (col_idx, matrix[[frame_idx, col_idx]]))
        .collect();
    indexed.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap());

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
    use super::*;
    use ndarray::array;

    #[test]
    fn speaker_count_rounds_overlap_added_sum() {
        let segmentations = array![
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
        ];

        let count = speaker_count(&segmentations, &[0, 1], 0, 4);

        assert_eq!(count, vec![1, 2, 1, 1]);
    }

    #[test]
    fn round_ties_even_matches_numpy_rint_behavior() {
        assert_eq!(round_ties_even(0.5), 0.0);
        assert_eq!(round_ties_even(1.5), 2.0);
        assert_eq!(round_ties_even(2.5), 2.0);
        assert_eq!(round_ties_even(3.5), 4.0);
    }

    #[test]
    fn reconstruct_maps_chunk_labels_to_global_clusters() {
        let segmentations = array![[[0.9, 0.1], [0.8, 0.2]], [[0.3, 0.7], [0.2, 0.8]],];
        let hard_clusters = array![[1, 0], [0, -2]];
        let count = vec![1, 1, 1];

        let result = reconstruct(&segmentations, &hard_clusters, &count, &[0, 1], 0);

        assert_eq!(result.ncols(), 2);
        assert_eq!(result[[0, 1]], 1.0);
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[1, 1]], 1.0);
        assert_eq!(result[[2, 0]], 1.0);
    }
}
