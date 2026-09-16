use kodama::{Method, Step, linkage};
use ndarray::{Array2, ArrayView2};

use crate::utils::l2_normalize_rows;

/// Invalid agglomerative clustering configuration
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum AhcConfigError {
    /// Merge threshold was negative or non-finite
    #[error("AHC threshold must be finite and non-negative, got {0}")]
    InvalidThreshold(f32),
}

/// Agglomerative hierarchical clustering settings for speaker embeddings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AhcConfig {
    threshold: f32,
}

impl Default for AhcConfig {
    fn default() -> Self {
        Self { threshold: 0.6 }
    }
}

impl AhcConfig {
    /// Create a checked AHC configuration
    pub fn new(threshold: f32) -> Result<Self, AhcConfigError> {
        if threshold.is_finite() && threshold >= 0.0 {
            Ok(Self { threshold })
        } else {
            Err(AhcConfigError::InvalidThreshold(threshold))
        }
    }

    /// Maximum dendrogram merge distance to keep observations in one flat cluster
    pub const fn threshold(self) -> f32 {
        self.threshold
    }
}

pub fn cluster(embeddings: &ArrayView2<f32>, config: AhcConfig) -> Vec<usize> {
    let observations = embeddings.nrows();
    if observations == 0 {
        return Vec::new();
    }
    if observations == 1 {
        return vec![0];
    }

    let normalized = l2_normalize_rows(embeddings);
    let distance_start = std::time::Instant::now();
    let mut condensed = condensed_euclidean(&normalized);
    let distance_ms = distance_start.elapsed().as_millis();
    let linkage_start = std::time::Instant::now();
    let dendrogram = linkage(&mut condensed, observations, Method::Centroid);
    let linkage_ms = linkage_start.elapsed().as_millis();
    let flat_start = std::time::Instant::now();
    let labels = flat_clusters(observations, dendrogram.steps(), config.threshold());
    tracing::debug!(
        observations,
        distance_ms,
        linkage_ms,
        flat_ms = flat_start.elapsed().as_millis(),
        "AHC stage timing"
    );

    labels
}

fn condensed_euclidean(embeddings: &Array2<f32>) -> Vec<f32> {
    condensed_euclidean_with_workers(embeddings, pdist_worker_count())
}

fn condensed_euclidean_with_workers(embeddings: &Array2<f32>, workers: usize) -> Vec<f32> {
    let observations = embeddings.nrows();
    if observations < 2 {
        return Vec::new();
    }

    let mut condensed = vec![0.0; observations * (observations - 1) / 2];
    let squared_norms: Vec<f32> = embeddings
        .rows()
        .into_iter()
        .map(|row| row.dot(&row))
        .collect();

    const BLOCK_SIZE: usize = 1024;
    let row_offset = |row: usize| row * (observations - 1) - row * row.saturating_sub(1) / 2;

    // each block owns a contiguous output slice, so workers need no write lock
    let mut blocks = Vec::new();
    {
        let mut remaining = condensed.as_mut_slice();
        let mut consumed = 0;
        let mut block_start = 0;
        while block_start < observations - 1 {
            let block_end = (block_start + BLOCK_SIZE).min(observations - 1);
            let end_offset = row_offset(block_end);
            let (block, tail) = remaining.split_at_mut(end_offset - consumed);
            blocks.push((block_start, block_end, block));
            remaining = tail;
            consumed = end_offset;
            block_start = block_end;
        }
    }

    // a bounded queue limits the number of live Gram matrices
    blocks.reverse();
    let workers = workers.max(1).min(blocks.len());
    let blocks = std::sync::Mutex::new(blocks);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let next = blocks.lock().expect("pdist queue poisoned").pop();
                    let Some((block_start, block_end, output)) = next else {
                        break;
                    };

                    let left = embeddings.slice(ndarray::s![block_start..block_end, ..]);
                    let right = embeddings.slice(ndarray::s![block_start.., ..]);
                    let gram = left.dot(&right.t());
                    let mut output_index = 0;

                    for (local_row, row) in (block_start..block_end).enumerate() {
                        for col in row + 1..observations {
                            let dot = gram[[local_row, col - block_start]];
                            let gram_distance = squared_norms[row] + squared_norms[col] - 2.0 * dot;
                            // avoid cancellation changing distinct close vectors into duplicates
                            let squared_distance = if gram_distance <= 1e-6 {
                                embeddings
                                    .row(row)
                                    .iter()
                                    .zip(embeddings.row(col))
                                    .map(|(left, right)| {
                                        let delta = left - right;
                                        delta * delta
                                    })
                                    .sum()
                            } else {
                                gram_distance
                            };
                            output[output_index] = squared_distance.sqrt();
                            output_index += 1;
                        }
                    }
                }
            });
        }
    });

    condensed
}

fn pdist_worker_count() -> usize {
    std::env::var("SPEAKRS_AHC_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|workers| *workers > 0)
        .unwrap_or_else(|| {
            let available = std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1);
            let matrix_workers = std::env::var("MATMUL_NUM_THREADS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(available)
                .clamp(1, 4);

            available.div_ceil(matrix_workers).min(8)
        })
}

fn flat_clusters(observations: usize, steps: &[Step<f32>], threshold: f32) -> Vec<usize> {
    if observations == 0 {
        return Vec::new();
    }
    if observations == 1 {
        return vec![0];
    }

    let total_nodes = observations + steps.len();
    let mut children = Vec::with_capacity(steps.len());
    let mut heights = vec![f32::INFINITY; total_nodes];

    for (step_idx, step) in steps.iter().enumerate() {
        let node_idx = observations + step_idx;
        children.push((step.cluster1, step.cluster2));
        heights[node_idx] = step.dissimilarity;
    }

    let root = total_nodes - 1;
    let mut labels = vec![usize::MAX; observations];
    let mut next_label = 0usize;
    assign_flat_labels(
        root,
        observations,
        threshold,
        &children,
        &heights,
        &mut labels,
        &mut next_label,
    );
    labels
}

fn assign_flat_labels(
    node_idx: usize,
    observations: usize,
    threshold: f32,
    children: &[(usize, usize)],
    heights: &[f32],
    labels: &mut [usize],
    next_label: &mut usize,
) {
    if node_idx < observations {
        labels[node_idx] = *next_label;
        *next_label += 1;
        return;
    }

    if heights[node_idx] <= threshold {
        label_subtree(node_idx, observations, children, labels, *next_label);
        *next_label += 1;
        return;
    }

    let (left, right) = child_pair(children, observations, node_idx);
    assign_flat_labels(
        left,
        observations,
        threshold,
        children,
        heights,
        labels,
        next_label,
    );
    assign_flat_labels(
        right,
        observations,
        threshold,
        children,
        heights,
        labels,
        next_label,
    );
}

fn label_subtree(
    node_idx: usize,
    observations: usize,
    children: &[(usize, usize)],
    labels: &mut [usize],
    label: usize,
) {
    if node_idx < observations {
        labels[node_idx] = label;
        return;
    }

    let (left, right) = child_pair(children, observations, node_idx);
    label_subtree(left, observations, children, labels, label);
    label_subtree(right, observations, children, labels, label);
}

fn child_pair(children: &[(usize, usize)], observations: usize, node_idx: usize) -> (usize, usize) {
    debug_assert!(
        node_idx >= observations,
        "child_pair should only be called for merge nodes"
    );
    children[node_idx - observations]
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, array};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::PathBuf;

    use super::*;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    fn condensed_euclidean_reference(embeddings: &Array2<f32>) -> Vec<f32> {
        let observations = embeddings.nrows();
        let mut condensed = Vec::with_capacity(observations * observations.saturating_sub(1) / 2);

        for row in 0..observations.saturating_sub(1) {
            for col in row + 1..observations {
                let squared_distance = embeddings
                    .row(row)
                    .iter()
                    .zip(embeddings.row(col))
                    .map(|(left, right)| {
                        let delta = left - right;
                        delta * delta
                    })
                    .sum::<f32>();
                condensed.push(squared_distance.sqrt());
            }
        }

        condensed
    }

    #[test]
    fn blocked_distances_match_scalar_reference() {
        let rows = 1_030;
        let cols = 16;
        let data = (0..rows * cols)
            .map(|index| ((index * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let embeddings =
            l2_normalize_rows(&Array2::from_shape_vec((rows, cols), data).unwrap().view());
        let expected = condensed_euclidean_reference(&embeddings);

        for workers in [1, 2, 8] {
            let actual = condensed_euclidean_with_workers(&embeddings, workers);
            assert_eq!(actual.len(), expected.len());

            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "distance {index} differs with {workers} workers: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn blocked_distances_preserve_distinct_close_vectors() {
        let embeddings = array![[1.0, 0.0], [1.0, 1e-5]];

        let distances = condensed_euclidean_with_workers(&embeddings, 1);

        assert_eq!(distances, vec![1e-5]);
    }

    #[test]
    fn zero_threshold_keeps_close_vectors_separate() {
        let embeddings = array![[1.0, 0.0], [1.0, 1e-5]];

        let labels = cluster(&embeddings.view(), AhcConfig::new(0.0).unwrap());

        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn blocked_distances_are_identical_across_worker_counts() {
        let embeddings = Array2::from_shape_fn((1_030, 8), |(row, col)| {
            ((row * 17 + col * 31) % 97) as f32 / 97.0
        });
        let expected = condensed_euclidean_with_workers(&embeddings, 1);

        for workers in [2, 8] {
            let actual = condensed_euclidean_with_workers(&embeddings, workers);
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                "worker count {workers} changed the distance output"
            );
        }
    }

    #[test]
    fn default_worker_count_is_bounded() {
        let workers = pdist_worker_count();
        assert!(workers >= 1);
        if std::env::var_os("SPEAKRS_AHC_THREADS").is_none() {
            assert!(workers <= 8);
        }
    }

    #[test]
    fn separates_two_clusters() {
        let embeddings = array![[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0], [-0.95, -0.05],];

        let labels = cluster(&embeddings.view(), AhcConfig::new(0.6).unwrap());

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn flat_clusters_follow_scipy_leader_order() {
        let steps = vec![
            Step::new(2, 3, 0.1, 2),
            Step::new(0, 1, 1.0, 2),
            Step::new(4, 5, 10.45, 4),
        ];

        let labels = flat_clusters(4, &steps, 1.0);

        assert_eq!(labels, vec![1, 1, 0, 0]);
    }

    #[test]
    fn cluster_matches_scipy_label_order_on_toy_example() {
        let embeddings = array![[1.0, 0.0], [0.9, 0.3], [0.0, 1.0], [0.05, 1.0],];

        let labels = cluster(&embeddings.view(), AhcConfig::new(0.6).unwrap());

        assert_eq!(labels, vec![1, 1, 0, 0]);
    }

    #[test]
    fn cluster_matches_python_fixture() {
        let embeddings: Array2<f32> =
            Array2::read_npy(File::open(fixture_path("pipeline_train_embeddings.npy")).unwrap())
                .unwrap();
        let expected: Array1<i64> =
            Array1::read_npy(File::open(fixture_path("pipeline_ahc_clusters.npy")).unwrap())
                .unwrap();

        let labels = cluster(&embeddings.view(), AhcConfig::default());

        assert_eq!(labels.len(), expected.len());
        for (lhs, rhs) in labels.iter().zip(expected.iter()) {
            assert_eq!(*lhs as i64, *rhs);
        }
    }

    #[test]
    fn new_rejects_invalid_threshold() {
        for threshold in [f32::NAN, f32::INFINITY, -0.1] {
            assert!(AhcConfig::new(threshold).is_err());
        }
        assert_eq!(AhcConfig::new(0.0).unwrap().threshold(), 0.0);
    }
}
