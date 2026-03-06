use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView2, Axis, s};

use crate::clustering::ClusterResult;
use crate::utils::logsumexp;

#[derive(Debug, Clone)]
pub struct VbxConfig {
    pub fa: f32,
    pub fb: f32,
    pub epsilon: f64,
    pub max_iters: usize,
    pub min_occupancy: f32,
}

impl Default for VbxConfig {
    fn default() -> Self {
        Self {
            fa: 0.01,
            fb: 0.04,
            epsilon: 1e-4,
            max_iters: 20,
            min_occupancy: 0.0,
        }
    }
}

pub fn cluster(
    features: &ArrayView2<f32>,
    num_speakers: usize,
    init_labels: Option<&[usize]>,
    config: &VbxConfig,
) -> ClusterResult {
    let (n_samples, dim) = (features.nrows(), features.ncols());

    let mut models = initialize_models(features, num_speakers, init_labels, dim);
    let mut prev_elbo = f64::NEG_INFINITY;

    for _iter in 0..config.max_iters {
        let log_gamma = compute_log_gamma(features, &models, config);
        let gamma = log_gamma.mapv(f32::exp);

        // M-step: update models, pruning low-occupancy speakers
        let mut new_models = Vec::new();
        for k in 0..models.nrows() {
            let n_k: f32 = gamma.column(k).sum();
            if n_k < config.min_occupancy {
                continue;
            }

            let mut f_k = Array1::zeros(dim);
            for t in 0..n_samples {
                f_k += &(features.row(t).to_owned() * gamma[[t, k]]);
            }

            new_models.push(f_k * config.fa / (1.0 + config.fa * config.fb * n_k));
        }

        if new_models.is_empty() {
            return ClusterResult {
                labels: vec![0; n_samples],
                num_clusters: 1,
                centroids: vec![features.mean_axis(Axis(0)).unwrap()],
            };
        }

        models = Array2::from_shape_vec(
            (new_models.len(), dim),
            new_models.iter().flat_map(|m| m.iter().copied()).collect(),
        )
        .unwrap();

        let elbo = compute_elbo(features, &models, &gamma, config);

        if prev_elbo.is_finite() && ((elbo - prev_elbo) / elbo.abs()).abs() < config.epsilon {
            break;
        }
        prev_elbo = elbo;
    }

    // final hard assignments
    let final_log_gamma = compute_log_gamma(features, &models, config);
    let raw_labels: Vec<usize> = (0..n_samples)
        .map(|t| {
            final_log_gamma
                .row(t)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();

    // renumber contiguously
    let mut label_map = HashMap::new();
    let mut next_label = 0usize;
    let labels: Vec<usize> = raw_labels
        .into_iter()
        .map(|l| {
            *label_map.entry(l).or_insert_with(|| {
                let v = next_label;
                next_label += 1;
                v
            })
        })
        .collect();

    let num_clusters = label_map.len();
    let centroids = (0..num_clusters)
        .map(|c| {
            let mut sum = Array1::zeros(dim);
            let mut count = 0.0f32;
            for (t, &l) in labels.iter().enumerate() {
                if l == c {
                    sum += &features.row(t).to_owned();
                    count += 1.0;
                }
            }
            if count > 0.0 { sum / count } else { sum }
        })
        .collect();

    ClusterResult {
        labels,
        num_clusters,
        centroids,
    }
}

fn compute_log_gamma(
    features: &ArrayView2<f32>,
    models: &Array2<f32>,
    config: &VbxConfig,
) -> Array2<f32> {
    let n_samples = features.nrows();
    let n_models = models.nrows();
    let mut log_gamma = Array2::zeros((n_samples, n_models));

    for t in 0..n_samples {
        for k in 0..n_models {
            let dot: f32 = features.row(t).dot(&models.row(k));
            let norm_sq: f32 = models.row(k).dot(&models.row(k));
            log_gamma[[t, k]] = config.fa * dot - 0.5 * config.fa * config.fb * norm_sq;
        }

        let row = log_gamma.row(t).to_owned();
        let lse = logsumexp(&row.view());
        for k in 0..n_models {
            log_gamma[[t, k]] -= lse;
        }
    }

    log_gamma
}

fn initialize_models(
    features: &ArrayView2<f32>,
    num_speakers: usize,
    init_labels: Option<&[usize]>,
    dim: usize,
) -> Array2<f32> {
    match init_labels {
        Some(labels) => {
            let mut models = Array2::zeros((num_speakers, dim));
            let mut counts = vec![0.0f32; num_speakers];

            for (t, &label) in labels.iter().enumerate() {
                if label < num_speakers {
                    for d in 0..dim {
                        models[[label, d]] += features[[t, d]];
                    }
                    counts[label] += 1.0;
                }
            }

            for k in 0..num_speakers {
                if counts[k] > 0.0 {
                    for d in 0..dim {
                        models[[k, d]] /= counts[k];
                    }
                }
            }
            models
        }
        None => {
            let n = num_speakers.min(features.nrows());
            features.slice(s![..n, ..]).to_owned()
        }
    }
}

fn compute_elbo(
    features: &ArrayView2<f32>,
    models: &Array2<f32>,
    gamma: &Array2<f32>,
    config: &VbxConfig,
) -> f64 {
    let mut elbo = 0.0f64;

    for t in 0..features.nrows() {
        for k in 0..models.nrows() {
            let g = gamma[[t, k]] as f64;
            if g > 1e-10 {
                let dot: f32 = features.row(t).dot(&models.row(k));
                let norm_sq: f32 = models.row(k).dot(&models.row(k));
                let log_like = (config.fa * dot - 0.5 * config.fa * config.fb * norm_sq) as f64;
                elbo += g * (log_like - g.ln());
            }
        }
    }

    elbo
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_well_separated_clusters() {
        let features = array![
            [10.0, 0.0],
            [10.1, 0.1],
            [9.9, -0.1],
            [-10.0, 0.0],
            [-10.1, 0.1],
            [-9.9, -0.1],
        ];

        let config = VbxConfig::default();
        let result = cluster(&features.view(), 2, None, &config);

        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[3], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn single_speaker_input() {
        let features = array![[1.0, 0.0], [1.1, 0.1], [0.9, -0.1]];

        let config = VbxConfig::default();
        let result = cluster(&features.view(), 1, None, &config);

        assert!(result.labels.iter().all(|&l| l == 0));
        assert_eq!(result.num_clusters, 1);
    }

    #[test]
    fn elbo_convergence() {
        let features = array![[5.0, 0.0], [5.1, 0.1], [-5.0, 0.0], [-5.1, 0.1],];

        let config = VbxConfig {
            max_iters: 50,
            ..Default::default()
        };
        let result = cluster(&features.view(), 2, None, &config);
        assert_eq!(result.num_clusters, 2);
    }

    #[test]
    fn with_init_labels() {
        let features = array![[10.0, 0.0], [10.1, 0.1], [-10.0, 0.0], [-10.1, 0.1],];

        let init_labels = vec![0, 0, 1, 1];
        let config = VbxConfig::default();
        let result = cluster(&features.view(), 2, Some(&init_labels), &config);

        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn convergence_within_max_iters() {
        let features = array![[20.0, 0.0], [20.1, 0.1], [0.0, 20.0], [0.1, 20.1],];

        // use init labels to guarantee proper starting point
        let init_labels = vec![0, 0, 1, 1];
        let config = VbxConfig {
            max_iters: 100,
            epsilon: 1e-6,
            ..Default::default()
        };
        let result = cluster(&features.view(), 2, Some(&init_labels), &config);
        assert_eq!(result.num_clusters, 2);
    }
}
