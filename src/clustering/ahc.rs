use std::collections::HashMap;

use kodama::{Method, linkage};
use ndarray::{Array2, ArrayView2};

use crate::utils::l2_normalize;

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
    dense_relabel(flat_clusters(
        observations,
        dendrogram.steps(),
        config.threshold,
    ))
}

fn l2_normalize_rows(embeddings: &ArrayView2<f32>) -> Array2<f32> {
    let mut normalized = embeddings.to_owned();
    for mut row in normalized.rows_mut() {
        let normalized_row = l2_normalize(&row.view());
        row.assign(&normalized_row);
    }
    normalized
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

fn flat_clusters(observations: usize, steps: &[kodama::Step<f32>], threshold: f32) -> Vec<usize> {
    let mut parents: Vec<usize> = (0..(observations * 2).saturating_sub(1)).collect();
    let mut next_cluster = observations;

    for step in steps {
        if step.dissimilarity > threshold {
            break;
        }

        let left = find(&mut parents, step.cluster1);
        let right = find(&mut parents, step.cluster2);
        parents[left] = next_cluster;
        parents[right] = next_cluster;
        next_cluster += 1;
    }

    (0..observations)
        .map(|idx| find(&mut parents, idx))
        .collect()
}

fn find(parents: &mut [usize], idx: usize) -> usize {
    let parent = parents[idx];
    if parent == idx {
        return idx;
    }

    let root = find(parents, parent);
    parents[idx] = root;
    root
}

fn dense_relabel(labels: Vec<usize>) -> Vec<usize> {
    let mut mapping = HashMap::new();
    let mut next = 0usize;

    labels
        .into_iter()
        .map(|label| {
            *mapping.entry(label).or_insert_with(|| {
                let value = next;
                next += 1;
                value
            })
        })
        .collect()
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
    fn dense_relabel_compacts_ids() {
        let labels = dense_relabel(vec![7, 7, 12, 18, 12]);
        assert_eq!(labels, vec![0, 0, 1, 2, 1]);
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
