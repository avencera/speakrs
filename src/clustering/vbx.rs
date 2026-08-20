use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::utils::logsumexp_f64;

/// Variational Bayes HMM clustering settings.
#[derive(Debug, Clone, Copy)]
pub struct VbxConfig {
    /// Speaker model scale used by the VBx update equations.
    pub fa: f64,
    /// Speaker regularization scale used by the VBx update equations.
    pub fb: f64,
    /// Maximum number of VBx iterations.
    pub max_iters: usize,
    /// Minimum ELBO improvement required to continue iterating.
    pub epsilon: f64,
    /// Smoothing factor applied to AHC labels before VBx refinement.
    pub init_smoothing: f64,
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

/// VBx clustering matching pyannote's algorithm
///
/// Takes PLDA-transformed features and per-dimension eigenvalues (Phi),
/// plus AHC-initialized gamma responsibilities. All computation is done
/// in f64 to match pyannote's numpy default precision
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

    // promote all working arrays to f64 to match pyannote precision
    let features_f64 = features.mapv(|v| v as f64);
    let phi_f64: Array1<f64> = phi.mapv(|v| v as f64);

    let mut gamma = gamma_init.mapv(|v| v as f64);
    let mut pi = Array1::from_elem(n_speakers, 1.0 / n_speakers as f64);

    // precompute per-frame constant: G = -0.5 * (sum(X^2, axis=1) + D*ln(2*pi))
    let frame_constants: Array1<f64> = features_f64
        .rows()
        .into_iter()
        .map(|row| -0.5 * (row.dot(&row) + dim as f64 * (2.0 * std::f64::consts::PI).ln()))
        .collect();

    // v = sqrt(phi)
    let phi_sqrt = phi_f64.mapv(f64::sqrt);

    // rho = X * V (element-wise broadcast)
    let mut rho = features_f64;
    for mut row in rho.rows_mut() {
        row *= &phi_sqrt;
    }

    let mut prev_elbo = f64::NEG_INFINITY;
    let mut scratch = Array1::<f64>::zeros(n_speakers);

    for iter in 0..config.max_iters {
        // m-step: compute speaker models
        // invL[k,d] = 1.0 / (1 + Fa/Fb * N_k * Phi[d])
        // alpha[k,d] = Fa/Fb * invL[k,d] * sum_t(gamma[t,k] * rho[t,d])
        // diar-native patch: vectorized — the original per-element loops cost
        // O(N*K*D) scalar work per iteration (305 s at N=21k, K=1.9k, D=128).
        let n_k: Array1<f64> = gamma.sum_axis(Axis(0));

        let mut inv_l = Array2::<f64>::zeros((n_speakers, dim));
        for (speaker_idx, mut row) in inv_l.rows_mut().into_iter().enumerate() {
            let scale = fa_over_fb * n_k[speaker_idx];
            row.assign(&phi_f64.mapv(|p| 1.0 / (1.0 + scale * p)));
        }

        // f = gamma.T @ rho  (K x D), alpha = Fa/Fb * invL ⊙ f
        let f = gamma.t().dot(&rho);
        let mut alpha = &inv_l * &f;
        alpha.mapv_inplace(|v| v * fa_over_fb);

        // e-step
        // log_p_[t,k] = Fa * (rho[t] . alpha[k] - 0.5 * (invL[k] + alpha[k]^2) . Phi + G[t])
        // penalty depends only on k — compute once per iteration, not per sample.
        let penalty: Array1<f64> = (0..n_speakers)
            .map(|speaker_idx| {
                inv_l
                    .row(speaker_idx)
                    .iter()
                    .zip(alpha.row(speaker_idx).iter())
                    .zip(phi_f64.iter())
                    .map(|((&il, &a), &p)| (il + a * a) * p)
                    .sum()
            })
            .collect();

        let mut log_p = rho.dot(&alpha.t()); // N x K
        for (sample_idx, mut row) in log_p.rows_mut().into_iter().enumerate() {
            let g = frame_constants[sample_idx];
            row.zip_mut_with(&penalty, |value, &pen| {
                *value = fa * (*value - 0.5 * pen + g);
            });
        }

        // GMM-style update with pi priors (single fused pass per row)
        let lpi: Array1<f64> = pi.mapv(|p| (p + 1e-8).ln());

        let mut log_p_x = Array1::<f64>::zeros(n_samples);
        for ((log_p_row, mut gamma_row), log_p_x_slot) in log_p
            .rows()
            .into_iter()
            .zip(gamma.rows_mut())
            .zip(log_p_x.iter_mut())
        {
            scratch.assign(&log_p_row);
            scratch += &lpi;
            let lse = logsumexp_f64(&scratch.view());
            *log_p_x_slot = lse;
            gamma_row.zip_mut_with(&scratch, |g, &s| *g = (s - lse).exp());
        }

        // update pi
        pi = gamma.sum_axis(Axis(0));
        let pi_sum = pi.sum();
        pi /= pi_sum;

        // elbo = sum(log_p_x) + Fb * 0.5 * sum(ln(invL) - invL - alpha^2 + 1)
        let log_px_sum: f64 = log_p_x.sum();
        let reg: f64 = inv_l
            .iter()
            .zip(alpha.iter())
            .map(|(&il, &a)| il.ln() - il - a * a + 1.0)
            .sum();
        let elbo = log_px_sum + fb * 0.5 * reg;

        if iter > 0 && elbo - prev_elbo < config.epsilon {
            break;
        }
        prev_elbo = elbo;
    }

    // convert back to f32 for downstream consumption
    let gamma_f32 = gamma.mapv(|v| v as f32);
    let pi_f32 = pi.mapv(|v| v as f32);
    (gamma_f32, pi_f32)
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

fn build_gamma_init(labels: &[usize], smoothing: f64) -> Array2<f32> {
    let num_samples = labels.len();
    let num_speakers = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut gamma = Array2::<f32>::zeros((num_samples, num_speakers));

    for (row, &label) in labels.iter().enumerate() {
        gamma[[row, label]] = 1.0;
    }

    if smoothing < 0.0 {
        return gamma;
    }

    let smoothing_f32 = smoothing as f32;
    for mut row in gamma.rows_mut() {
        row *= smoothing_f32;
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
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
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
