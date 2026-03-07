use kodama::{Method, Step, linkage};
use ndarray::{Array2, ArrayView2};

use crate::utils::l2_normalize_rows;

#[derive(Debug, Clone, Copy)]
pub struct AhcConfig {
    pub threshold: f32,
}

impl Default for AhcConfig {
    fn default() -> Self {
        Self { threshold: 0.6 }
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
    let mut condensed = condensed_euclidean(&normalized);
    let dendrogram = linkage(&mut condensed, observations, Method::Centroid);
    flat_clusters(observations, dendrogram.steps(), config.threshold)
}

fn condensed_euclidean(embeddings: &Array2<f32>) -> Vec<f32> {
    let observations = embeddings.nrows();
    let mut condensed = Vec::with_capacity(observations * (observations - 1) / 2);
    for row in 0..observations.saturating_sub(1) {
        for col in row + 1..observations {
            let lhs = embeddings.row(row);
            let rhs = embeddings.row(col);
            let distance = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(left, right)| {
                    let delta = left - right;
                    delta * delta
                })
                .sum::<f32>()
                .sqrt();
            condensed.push(distance);
        }
    }
    condensed
}

fn flat_clusters(observations: usize, steps: &[Step<f32>], threshold: f32) -> Vec<usize> {
    if observations == 0 {
        return Vec::new();
    }
    if observations == 1 {
        return vec![0];
    }

    let total_nodes = observations + steps.len();
    let mut children = vec![None; total_nodes];
    let mut heights = vec![f32::INFINITY; total_nodes];

    for (step_idx, step) in steps.iter().enumerate() {
        let node_idx = observations + step_idx;
        children[node_idx] = Some((step.cluster1, step.cluster2));
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
    children: &[Option<(usize, usize)>],
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

    let (left, right) = children[node_idx].unwrap();
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
    children: &[Option<(usize, usize)>],
    labels: &mut [usize],
    label: usize,
) {
    if node_idx < observations {
        labels[node_idx] = label;
        return;
    }

    let (left, right) = children[node_idx].unwrap();
    label_subtree(left, observations, children, labels, label);
    label_subtree(right, observations, children, labels, label);
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

    #[test]
    fn separates_two_clusters() {
        let embeddings = array![[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0], [-0.95, -0.05],];

        let labels = cluster(&embeddings.view(), AhcConfig { threshold: 0.6 });

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

        let labels = cluster(&embeddings.view(), AhcConfig { threshold: 0.6 });

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
}
