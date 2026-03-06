use ndarray::Array2;

/// Count the number of active speakers per frame from a binarized activation matrix
pub fn count_speakers(binarized: &Array2<f32>) -> Vec<usize> {
    binarized
        .rows()
        .into_iter()
        .map(|row| row.sum() as usize)
        .collect()
}

/// Reconstruct global speaker activations from local segmentation scores and cluster labels
///
/// Maps local speaker scores to global speaker indices via cluster_labels, using max-pooling
/// when multiple local speakers map to the same global cluster. Optionally applies per-frame
/// top-K selection based on speaker_count.
pub fn reconstruct(
    scores: &Array2<f32>,
    cluster_labels: &[usize],
    num_global_speakers: usize,
    speaker_count: Option<&[usize]>,
) -> Array2<f32> {
    let num_frames = scores.nrows();
    let mut output = Array2::<f32>::zeros((num_frames, num_global_speakers));

    // max-pool local speaker scores into global speaker slots
    for (local_speaker, &cluster_id) in cluster_labels.iter().enumerate() {
        for frame in 0..num_frames {
            let score = scores[[frame, local_speaker]];
            let current = &mut output[[frame, cluster_id]];
            *current = current.max(score);
        }
    }

    // apply per-frame top-K selection if speaker counts are provided
    if let Some(counts) = speaker_count {
        for (frame, &k) in counts.iter().enumerate() {
            top_k_zero(&mut output, frame, k);
        }
    }

    output
}

/// For each frame, keep only the speaker with the highest activation, zero out others
pub fn make_exclusive(activations: &mut Array2<f32>) {
    for mut row in activations.rows_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_val == 0.0 {
            continue;
        }

        let argmax = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        for (i, val) in row.iter_mut().enumerate() {
            if i != argmax {
                *val = 0.0;
            }
        }
    }
}

/// Keep only the top-K values in a row, zeroing out the rest
fn top_k_zero(matrix: &mut Array2<f32>, frame: usize, k: usize) {
    let ncols = matrix.ncols();
    if k >= ncols {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = (0..ncols).map(|i| (i, matrix[[frame, i]])).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(col, _) in &indexed[k..] {
        matrix[[frame, col]] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn passthrough_single_speaker() {
        let scores = array![[0.8], [0.3], [0.9]];
        let labels = &[0usize];
        let result = reconstruct(&scores, labels, 1, None);
        assert_eq!(result, scores);
    }

    #[test]
    fn merge_two_local_to_one_global() {
        let scores = array![[0.3, 0.7], [0.9, 0.2], [0.5, 0.5]];
        let labels = &[0usize, 0];
        let result = reconstruct(&scores, labels, 1, None);

        let expected = array![[0.7], [0.9], [0.5]];
        assert_eq!(result, expected);
    }

    #[test]
    fn top_k_limiting() {
        let scores = array![[0.8, 0.6, 0.4], [0.3, 0.9, 0.7], [0.5, 0.2, 0.1]];
        let labels = &[0usize, 1, 2];
        let counts = vec![1, 2, 1];
        let result = reconstruct(&scores, labels, 3, Some(&counts));

        // frame 0: keep top-1 → only speaker 0 (0.8)
        assert_abs_diff_eq!(result[[0, 0]], 0.8);
        assert_abs_diff_eq!(result[[0, 1]], 0.0);
        assert_abs_diff_eq!(result[[0, 2]], 0.0);

        // frame 1: keep top-2 → speakers 1 (0.9) and 2 (0.7)
        assert_abs_diff_eq!(result[[1, 0]], 0.0);
        assert_abs_diff_eq!(result[[1, 1]], 0.9);
        assert_abs_diff_eq!(result[[1, 2]], 0.7);

        // frame 2: keep top-1 → only speaker 0 (0.5)
        assert_abs_diff_eq!(result[[2, 0]], 0.5);
        assert_abs_diff_eq!(result[[2, 1]], 0.0);
        assert_abs_diff_eq!(result[[2, 2]], 0.0);
    }

    #[test]
    fn make_exclusive_keeps_argmax() {
        let mut activations = array![[0.3, 0.8, 0.1], [0.5, 0.2, 0.9], [0.0, 0.0, 0.0]];
        make_exclusive(&mut activations);

        assert_abs_diff_eq!(activations[[0, 0]], 0.0);
        assert_abs_diff_eq!(activations[[0, 1]], 0.8);
        assert_abs_diff_eq!(activations[[0, 2]], 0.0);

        assert_abs_diff_eq!(activations[[1, 0]], 0.0);
        assert_abs_diff_eq!(activations[[1, 1]], 0.0);
        assert_abs_diff_eq!(activations[[1, 2]], 0.9);

        // all zeros left as-is
        assert_abs_diff_eq!(activations[[2, 0]], 0.0);
        assert_abs_diff_eq!(activations[[2, 1]], 0.0);
        assert_abs_diff_eq!(activations[[2, 2]], 0.0);
    }

    #[test]
    fn count_speakers_known_input() {
        let binarized = array![
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];
        assert_eq!(count_speakers(&binarized), vec![2, 3, 0, 1]);
    }

    #[test]
    fn empty_input() {
        let scores = Array2::<f32>::zeros((0, 0));
        let result = reconstruct(&scores, &[], 0, None);
        assert_eq!(result.shape(), &[0, 0]);
    }

    #[test]
    fn single_frame() {
        let scores = array![[0.4, 0.7]];
        let labels = &[1usize, 0];
        let result = reconstruct(&scores, labels, 2, None);

        // local 0 → global 1, local 1 → global 0
        assert_abs_diff_eq!(result[[0, 0]], 0.7);
        assert_abs_diff_eq!(result[[0, 1]], 0.4);
    }
}
