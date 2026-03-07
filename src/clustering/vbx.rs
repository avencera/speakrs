use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::utils::logsumexp;

#[derive(Debug, Clone, Copy)]
pub struct VbxConfig {
    pub fa: f32,
    pub fb: f32,
    pub max_iters: usize,
    pub epsilon: f64,
    pub init_smoothing: f32,
}

impl Default for VbxConfig {
    fn default() -> Self {
        Self {
            fa: 0.07,
            fb: 0.8,
            max_iters: 20,
            epsilon: 1e-4,
            init_smoothing: 7.0,
        }
    }
}

/// VBx clustering matching pyannote's implementation exactly
///
/// Takes PLDA-transformed features and per-dimension eigenvalues (Phi),
/// plus AHC-initialized gamma responsibilities
pub fn vbx(
    features: &ArrayView2<f32>,
    phi: &ArrayView1<f32>,
    gamma_init: &Array2<f32>,
    config: &VbxConfig,
) -> (Array2<f32>, Array1<f32>) {
    let (n_samples, dim) = features.dim();
    let n_speakers = gamma_init.ncols();
    let fa = config.fa;
    let fb = config.fb;
    let fa_over_fb = fa / fb;

    let mut gamma = gamma_init.clone();
    let mut pi = Array1::from_elem(n_speakers, 1.0 / n_speakers as f32);

    // precompute per-frame constant: G = -0.5 * (sum(X^2, axis=1) + D*ln(2*pi))
    let g: Array1<f32> = features
        .rows()
        .into_iter()
        .map(|row| -0.5 * (row.dot(&row) + dim as f32 * (2.0 * std::f32::consts::PI).ln()))
        .collect();

    // V = sqrt(Phi)
    let v = phi.mapv(f32::sqrt);

    // rho = X * V (element-wise broadcast)
    let mut rho = features.to_owned();
    for mut row in rho.rows_mut() {
        row *= &v;
    }

    let mut prev_elbo = f64::NEG_INFINITY;
    let mut scratch = Array1::<f32>::zeros(n_speakers);

    for iter in 0..config.max_iters {
        // M-step: compute speaker models
        // invL[k,d] = 1.0 / (1 + Fa/Fb * N_k * Phi[d])
        // alpha[k,d] = Fa/Fb * invL[k,d] * sum_t(gamma[t,k] * rho[t,d])
        let n_k: Array1<f32> = gamma.sum_axis(Axis(0));

        let mut inv_l = Array2::zeros((n_speakers, dim));
        let mut alpha = Array2::zeros((n_speakers, dim));

        for k in 0..n_speakers {
            for d in 0..dim {
                inv_l[[k, d]] = 1.0 / (1.0 + fa_over_fb * n_k[k] * phi[d]);
            }

            // gamma.T @ rho for speaker k
            let mut f_k = Array1::zeros(dim);
            for t in 0..n_samples {
                f_k.scaled_add(gamma[[t, k]], &rho.row(t));
            }

            for d in 0..dim {
                alpha[[k, d]] = fa_over_fb * inv_l[[k, d]] * f_k[d];
            }
        }

        // E-step
        // log_p_[t,k] = Fa * (rho[t] . alpha[k] - 0.5 * (invL[k] + alpha[k]^2) . Phi + G[t])
        let mut log_p = Array2::zeros((n_samples, n_speakers));
        for t in 0..n_samples {
            for k in 0..n_speakers {
                let rho_dot_alpha: f32 = rho.row(t).dot(&alpha.row(k));
                let penalty: f32 = (0..dim)
                    .map(|d| (inv_l[[k, d]] + alpha[[k, d]] * alpha[[k, d]]) * phi[d])
                    .sum();
                log_p[[t, k]] = fa * (rho_dot_alpha - 0.5 * penalty + g[t]);
            }
        }

        // GMM-style update with pi priors
        let eps = 1e-8f32;
        let lpi: Array1<f32> = pi.mapv(|p| (p + eps).ln());

        // log_p_x[t] = logsumexp(log_p[t] + lpi)
        let mut log_p_x = Array1::zeros(n_samples);
        for t in 0..n_samples {
            scratch.assign(&log_p.row(t));
            scratch += &lpi;
            log_p_x[t] = logsumexp(&scratch.view());
        }

        // gamma[t,k] = exp(log_p[t,k] + lpi[k] - log_p_x[t])
        for t in 0..n_samples {
            for k in 0..n_speakers {
                gamma[[t, k]] = (log_p[[t, k]] + lpi[k] - log_p_x[t]).exp();
            }
        }

        // update pi
        pi = gamma.sum_axis(Axis(0));
        let pi_sum = pi.sum();
        pi /= pi_sum;

        // ELBO = sum(log_p_x) + Fb * 0.5 * sum(ln(invL) - invL - alpha^2 + 1)
        let log_px_sum: f64 = log_p_x.iter().map(|&x| x as f64).sum();
        let reg: f64 = inv_l
            .iter()
            .zip(alpha.iter())
            .map(|(&il, &a)| (il.ln() - il - a * a + 1.0) as f64)
            .sum();
        let elbo = log_px_sum + fb as f64 * 0.5 * reg;

        if iter > 0 && elbo - prev_elbo < config.epsilon {
            break;
        }
        prev_elbo = elbo;
    }

    (gamma, pi)
}

pub fn cluster_vbx(
    ahc_labels: &[usize],
    features: &ArrayView2<f32>,
    phi: &ArrayView1<f32>,
    config: &VbxConfig,
) -> (Array2<f32>, Array1<f32>) {
    let gamma_init = build_gamma_init(ahc_labels, config.init_smoothing);
    vbx(features, phi, &gamma_init, config)
}

fn build_gamma_init(labels: &[usize], smoothing: f32) -> Array2<f32> {
    let num_samples = labels.len();
    let num_speakers = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut gamma = Array2::<f32>::zeros((num_samples, num_speakers));

    for (row, &label) in labels.iter().enumerate() {
        gamma[[row, label]] = 1.0;
    }

    if smoothing < 0.0 {
        return gamma;
    }

    for mut row in gamma.rows_mut() {
        row *= smoothing;
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|v| (v - max).exp());
        let denom = row.sum();
        row /= denom;
    }

    gamma
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn two_clusters_with_vbx() {
        // two well-separated clusters in 2D
        let features = array![
            [10.0, 0.0],
            [10.1, 0.1],
            [9.9, -0.1],
            [-10.0, 0.0],
            [-10.1, 0.1],
            [-9.9, -0.1],
        ];

        let phi = array![1.0, 1.0];

        // AHC-like init: first 3 → speaker 0, last 3 → speaker 1
        let mut gamma_init = Array2::zeros((6, 2));
        for t in 0..3 {
            gamma_init[[t, 0]] = 0.999;
            gamma_init[[t, 1]] = 0.001;
        }
        for t in 3..6 {
            gamma_init[[t, 0]] = 0.001;
            gamma_init[[t, 1]] = 0.999;
        }

        let (gamma, _pi) = vbx(
            &features.view(),
            &phi.view(),
            &gamma_init,
            &VbxConfig::default(),
        );

        // check hard assignments
        let labels: Vec<usize> = gamma
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn gamma_init_is_smoothed_one_hot() {
        let gamma = build_gamma_init(&[0, 0, 1], 7.0);
        assert_eq!(gamma.dim(), (3, 2));
        assert!(gamma[[0, 0]] > gamma[[0, 1]]);
        assert!(gamma[[2, 1]] > gamma[[2, 0]]);
    }

    #[test]
    fn cluster_vbx_matches_python_fixture() {
        let ahc_labels: Array1<i64> =
            Array1::read_npy(File::open(fixture_path("pipeline_ahc_clusters.npy")).unwrap())
                .unwrap();
        let features: Array2<f64> =
            Array2::read_npy(File::open(fixture_path("pipeline_plda_features.npy")).unwrap())
                .unwrap();
        let phi: Array1<f64> =
            Array1::read_npy(File::open(fixture_path("pipeline_plda_phi.npy")).unwrap()).unwrap();
        let expected_gamma: Array2<f64> =
            Array2::read_npy(File::open(fixture_path("pipeline_vbx_gamma.npy")).unwrap()).unwrap();
        let expected_pi: Array1<f64> =
            Array1::read_npy(File::open(fixture_path("pipeline_vbx_pi.npy")).unwrap()).unwrap();

        let ahc_labels: Vec<usize> = ahc_labels.iter().map(|value| *value as usize).collect();
        let features = features.mapv(|value| value as f32);
        let phi = phi.mapv(|value| value as f32);
        let (gamma, pi) = cluster_vbx(
            &ahc_labels,
            &features.view(),
            &phi.view(),
            &VbxConfig::default(),
        );

        for (lhs, rhs) in gamma.iter().zip(expected_gamma.iter()) {
            assert_abs_diff_eq!(*lhs, *rhs as f32, epsilon = 1e-4);
        }

        for (lhs, rhs) in pi.iter().zip(expected_pi.iter()) {
            assert_abs_diff_eq!(*lhs, *rhs as f32, epsilon = 1e-5);
        }
    }
}
