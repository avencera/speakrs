use std::num::NonZeroUsize;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::utils::logsumexp_f64;

/// How AHC labels initialize Gaussian VBx responsibilities.
///
/// Negative smoothing maps to [`Hard`], zero to [`Uniform`], and a positive
/// finite value to [`Smoothed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponsibilityInitialization {
    /// One-hot responsibilities from AHC labels
    Hard,
    /// Uniform responsibilities after converting the one-hot rows
    Uniform,
    /// Softmax-smoothed responsibilities with the given positive scale
    Smoothed(f64),
}

impl ResponsibilityInitialization {
    /// Map the historical smoothing number onto a checked initialization
    pub fn from_smoothing(value: f64) -> Result<Self, VbxConfigError> {
        if !value.is_finite() {
            return Err(VbxConfigError::NonFiniteSmoothing(value));
        }
        if value < 0.0 {
            Ok(Self::Hard)
        } else if value == 0.0 {
            Ok(Self::Uniform)
        } else {
            Ok(Self::Smoothed(value))
        }
    }
}

/// Invalid Gaussian VBx configuration
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum VbxConfigError {
    /// FA or FB was not finite and greater than zero
    #[error("VBx scale must be finite and greater than zero, got {0}")]
    InvalidScale(f64),
    /// Iteration count was zero
    #[error("VBx max_iters must be greater than zero")]
    ZeroIterations,
    /// Convergence epsilon was negative or non-finite
    #[error("VBx epsilon must be finite and non-negative, got {0}")]
    InvalidEpsilon(f64),
    /// Historical smoothing number was non-finite
    #[error("VBx smoothing must be finite, got {0}")]
    NonFiniteSmoothing(f64),
    /// Smoothed initialization scale was non-finite or not greater than zero
    #[error("VBx smoothed initialization scale must be finite and greater than zero, got {0}")]
    InvalidSmoothedScale(f64),
}

/// Variational Bayes HMM clustering settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VbxConfig {
    fa: f64,
    fb: f64,
    max_iters: NonZeroUsize,
    epsilon: f64,
    initialization: ResponsibilityInitialization,
}

impl Default for VbxConfig {
    fn default() -> Self {
        Self {
            fa: 0.07,
            fb: 0.8,
            max_iters: NonZeroUsize::new(20).expect("20 is non-zero"),
            epsilon: 1e-4,
            initialization: ResponsibilityInitialization::Smoothed(7.0),
        }
    }
}

impl VbxConfig {
    /// Create a checked Gaussian VBx configuration
    pub fn new(
        fa: f64,
        fb: f64,
        max_iters: usize,
        epsilon: f64,
        initialization: ResponsibilityInitialization,
    ) -> Result<Self, VbxConfigError> {
        if !(fa.is_finite() && fa > 0.0) {
            return Err(VbxConfigError::InvalidScale(fa));
        }
        if !(fb.is_finite() && fb > 0.0) {
            return Err(VbxConfigError::InvalidScale(fb));
        }
        let max_iters = NonZeroUsize::new(max_iters).ok_or(VbxConfigError::ZeroIterations)?;
        if !(epsilon.is_finite() && epsilon >= 0.0) {
            return Err(VbxConfigError::InvalidEpsilon(epsilon));
        }
        if let ResponsibilityInitialization::Smoothed(scale) = initialization
            && !(scale.is_finite() && scale > 0.0)
        {
            return Err(VbxConfigError::InvalidSmoothedScale(scale));
        }
        Ok(Self {
            fa,
            fb,
            max_iters,
            epsilon,
            initialization,
        })
    }

    /// Speaker model scale
    pub const fn fa(self) -> f64 {
        self.fa
    }

    /// Speaker regularization scale
    pub const fn fb(self) -> f64 {
        self.fb
    }

    /// Maximum VBx iterations
    pub const fn max_iters(self) -> usize {
        self.max_iters.get()
    }

    /// ELBO improvement threshold
    pub const fn epsilon(self) -> f64 {
        self.epsilon
    }

    /// Responsibility initialization
    pub const fn initialization(self) -> ResponsibilityInitialization {
        self.initialization
    }

    /// Replace the iteration limit
    pub fn with_max_iters(self, max_iters: usize) -> Result<Self, VbxConfigError> {
        Self::new(
            self.fa,
            self.fb,
            max_iters,
            self.epsilon,
            self.initialization,
        )
    }

    /// Replace the FB scale
    pub fn with_fb(self, fb: f64) -> Result<Self, VbxConfigError> {
        Self::new(
            self.fa,
            fb,
            self.max_iters.get(),
            self.epsilon,
            self.initialization,
        )
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
    let fa = config.fa();
    let fb = config.fb();
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

    for iter in 0..config.max_iters() {
        // m-step: compute speaker models
        // invL[k,d] = 1.0 / (1 + Fa/Fb * N_k * Phi[d])
        // alpha[k,d] = Fa/Fb * invL[k,d] * sum_t(gamma[t,k] * rho[t,d])
        let n_k: Array1<f64> = gamma.sum_axis(Axis(0));

        let mut inv_l = Array2::zeros((n_speakers, dim));
        let mut alpha = Array2::zeros((n_speakers, dim));

        for speaker_idx in 0..n_speakers {
            for dim_idx in 0..dim {
                inv_l[[speaker_idx, dim_idx]] =
                    1.0 / (1.0 + fa_over_fb * n_k[speaker_idx] * phi_f64[dim_idx]);
            }

            // gamma.T @ rho for this speaker
            let mut f_k = Array1::<f64>::zeros(dim);
            for sample_idx in 0..n_samples {
                f_k.scaled_add(gamma[[sample_idx, speaker_idx]], &rho.row(sample_idx));
            }

            for dim_idx in 0..dim {
                alpha[[speaker_idx, dim_idx]] =
                    fa_over_fb * inv_l[[speaker_idx, dim_idx]] * f_k[dim_idx];
            }
        }

        // e-step
        // log_p_[t,k] = Fa * (rho[t] . alpha[k] - 0.5 * (invL[k] + alpha[k]^2) . Phi + G[t])
        let mut log_p = Array2::<f64>::zeros((n_samples, n_speakers));
        for sample_idx in 0..n_samples {
            for speaker_idx in 0..n_speakers {
                let rho_dot_alpha: f64 = rho.row(sample_idx).dot(&alpha.row(speaker_idx));
                let penalty: f64 = (0..dim)
                    .map(|dim_idx| {
                        (inv_l[[speaker_idx, dim_idx]]
                            + alpha[[speaker_idx, dim_idx]] * alpha[[speaker_idx, dim_idx]])
                            * phi_f64[dim_idx]
                    })
                    .sum();
                log_p[[sample_idx, speaker_idx]] =
                    fa * (rho_dot_alpha - 0.5 * penalty + frame_constants[sample_idx]);
            }
        }

        // GMM-style update with pi priors
        let lpi: Array1<f64> = pi.mapv(|p| (p + 1e-8).ln());

        // log_p_x[sample_idx] = logsumexp(log_p[sample_idx] + lpi)
        let mut log_p_x = Array1::<f64>::zeros(n_samples);
        for sample_idx in 0..n_samples {
            scratch.assign(&log_p.row(sample_idx));
            scratch += &lpi;
            log_p_x[sample_idx] = logsumexp_f64(&scratch.view());
        }

        // gamma[sample_idx,speaker_idx] = exp(log_p[sample_idx,speaker_idx] + lpi[speaker_idx] - log_p_x[sample_idx])
        for sample_idx in 0..n_samples {
            for speaker_idx in 0..n_speakers {
                gamma[[sample_idx, speaker_idx]] =
                    (log_p[[sample_idx, speaker_idx]] + lpi[speaker_idx] - log_p_x[sample_idx])
                        .exp();
            }
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

        if iter > 0 && elbo - prev_elbo < config.epsilon() {
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
    let gamma_init = build_gamma_init(ahc_labels, config.initialization());
    vbx(features, phi, &gamma_init, config)
}

fn build_gamma_init(labels: &[usize], initialization: ResponsibilityInitialization) -> Array2<f32> {
    let num_samples = labels.len();
    let num_speakers = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut gamma = Array2::<f32>::zeros((num_samples, num_speakers));

    for (row, &label) in labels.iter().enumerate() {
        gamma[[row, label]] = 1.0;
    }

    let smoothing = match initialization {
        ResponsibilityInitialization::Hard => return gamma,
        ResponsibilityInitialization::Uniform => 0.0,
        ResponsibilityInitialization::Smoothed(scale) => scale as f32,
    };

    for mut row in gamma.rows_mut() {
        row *= smoothing;
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|v| (v - max).exp());
        let denom = row.sum();
        if denom > 0.0 {
            row /= denom;
        } else {
            let uniform = 1.0 / num_speakers as f32;
            row.fill(uniform);
        }
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
        let gamma = build_gamma_init(&[0, 0, 1], ResponsibilityInitialization::Smoothed(7.0));
        assert_eq!(gamma.dim(), (3, 2));
        assert!(gamma[[0, 0]] > gamma[[0, 1]]);
        assert!(gamma[[2, 1]] > gamma[[2, 0]]);
    }

    #[test]
    fn gamma_init_is_hard_one_hot() {
        let gamma = build_gamma_init(&[0, 0, 1], ResponsibilityInitialization::Hard);
        assert_eq!(gamma, array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn gamma_init_is_uniform() {
        let gamma = build_gamma_init(&[0, 0, 1], ResponsibilityInitialization::Uniform);
        assert_eq!(gamma, array![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]);
    }

    #[test]
    fn from_smoothing_maps_negative_zero_and_positive() {
        assert_eq!(
            ResponsibilityInitialization::from_smoothing(-1.0).unwrap(),
            ResponsibilityInitialization::Hard
        );
        assert_eq!(
            ResponsibilityInitialization::from_smoothing(0.0).unwrap(),
            ResponsibilityInitialization::Uniform
        );
        assert_eq!(
            ResponsibilityInitialization::from_smoothing(7.0).unwrap(),
            ResponsibilityInitialization::Smoothed(7.0)
        );
        assert!(ResponsibilityInitialization::from_smoothing(f64::NAN).is_err());
    }

    #[test]
    fn vbx_config_rejects_invalid_values() {
        assert!(VbxConfig::new(0.0, 0.8, 20, 1e-4, ResponsibilityInitialization::Hard).is_err());
        assert!(VbxConfig::new(0.07, -1.0, 20, 1e-4, ResponsibilityInitialization::Hard).is_err());
        assert!(VbxConfig::new(0.07, 0.8, 0, 1e-4, ResponsibilityInitialization::Hard).is_err());
        assert!(VbxConfig::new(0.07, 0.8, 20, -1.0, ResponsibilityInitialization::Hard).is_err());
        for scale in [0.0, -1.0, f64::INFINITY] {
            let error = VbxConfig::new(
                0.07,
                0.8,
                20,
                1e-4,
                ResponsibilityInitialization::Smoothed(scale),
            )
            .unwrap_err();
            let VbxConfigError::InvalidSmoothedScale(rejected) = error else {
                panic!("expected invalid smoothed scale, got {error}");
            };
            assert_eq!(rejected, scale);
            assert!(error.to_string().contains("finite and greater than zero"));
        }
        assert_eq!(VbxConfig::default().max_iters(), 20);
        assert_eq!(
            VbxConfig::default().with_max_iters(3).unwrap().max_iters(),
            3
        );
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
