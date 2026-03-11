use ndarray::{Array2, Array3, s};

pub fn speaker_count(
    binarized_segmentations: &Array3<f32>,
    start_frames: &[usize],
    warmup_frames: usize,
    output_frames: usize,
) -> Vec<usize> {
    let num_chunks = binarized_segmentations.shape()[0];
    if num_chunks == 0 {
        return Vec::new();
    }

    let num_frames = binarized_segmentations.shape()[1];
    let warmup_end = num_frames.saturating_sub(warmup_frames);
    let mut numerator = vec![0.0f32; output_frames];
    let mut denominator = vec![0.0f32; output_frames];

    for (chunk_idx, &start_frame) in start_frames.iter().enumerate().take(num_chunks) {
        for frame_idx in warmup_frames..warmup_end {
            let out_frame = start_frame + frame_idx;
            if out_frame >= output_frames {
                continue;
            }

            numerator[out_frame] += binarized_segmentations
                .slice(s![chunk_idx, frame_idx, ..])
                .iter()
                .sum::<f32>();
            denominator[out_frame] += 1.0;
        }
    }

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
        .collect()
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

/// Accumulate per-cluster activation scores from chunk segmentations,
/// padding to at least max_speakers_per_frame columns
fn accumulate_activations(
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
    let warmup_end = num_frames.saturating_sub(warmup_frames);
    let mut activations = Array2::<f32>::zeros((speaker_count.len(), num_clusters));

    for (chunk_idx, &start_frame) in start_frames.iter().enumerate().take(num_chunks) {
        let chunk_labels = hard_clusters.row(chunk_idx);
        let chunk_segmentations = segmentations.slice(s![chunk_idx, .., ..]);
        let local_by_cluster = build_cluster_mapping(&chunk_labels, num_clusters);

        for (cluster_idx, local_indices) in local_by_cluster.iter().enumerate() {
            if local_indices.is_empty() {
                continue;
            }

            for frame_idx in warmup_frames..warmup_end {
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

    activations
}

pub fn reconstruct(
    segmentations: &Array3<f32>,
    hard_clusters: &Array2<i32>,
    speaker_count: &[usize],
    start_frames: &[usize],
    warmup_frames: usize,
) -> Array2<f32> {
    let activations = accumulate_activations(
        segmentations,
        hard_clusters,
        speaker_count,
        start_frames,
        warmup_frames,
    );

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

        for (col_idx, value) in row.iter_mut().enumerate() {
            if col_idx != argmax {
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

/// Top-K with temporal smoothing: when scores are within epsilon,
/// prefer speakers that were active in the previous frame to reduce flicker
fn top_k_indices_smoothed(
    matrix: &Array2<f32>,
    frame_idx: usize,
    k: usize,
    prev_speakers: &[usize],
    epsilon: f32,
) -> Vec<usize> {
    let ncols = matrix.ncols();
    if k >= ncols {
        return (0..ncols).collect();
    }

    let mut indexed: Vec<(usize, f32)> = (0..ncols)
        .map(|col_idx| (col_idx, matrix[[frame_idx, col_idx]]))
        .collect();

    // sort by score descending, tie-break by previous-frame presence
    indexed.sort_by(|lhs, rhs| {
        let score_diff = rhs.1 - lhs.1;
        if score_diff.abs() < epsilon {
            let lhs_prev = prev_speakers.contains(&lhs.0);
            let rhs_prev = prev_speakers.contains(&rhs.0);
            rhs_prev.cmp(&lhs_prev)
        } else {
            rhs.1.partial_cmp(&lhs.1).unwrap()
        }
    });

    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

/// Reconstruct with temporal smoothing to reduce single-frame speaker flickers
pub fn reconstruct_smoothed(
    segmentations: &Array3<f32>,
    hard_clusters: &Array2<i32>,
    speaker_count: &[usize],
    start_frames: &[usize],
    warmup_frames: usize,
    epsilon: f32,
) -> Array2<f32> {
    let activations = accumulate_activations(
        segmentations,
        hard_clusters,
        speaker_count,
        start_frames,
        warmup_frames,
    );

    let mut discrete = Array2::<f32>::zeros(activations.raw_dim());
    let mut prev_speakers: Vec<usize> = Vec::new();

    for (frame_idx, &count) in speaker_count.iter().enumerate() {
        let current =
            top_k_indices_smoothed(&activations, frame_idx, count, &prev_speakers, epsilon);
        for &speaker_idx in &current {
            discrete[[frame_idx, speaker_idx]] = 1.0;
        }
        prev_speakers = current;
    }

    discrete
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
