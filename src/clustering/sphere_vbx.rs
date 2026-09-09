use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::utils::logsumexp_f64;

/// Positive responsibility smoothing used to initialize SphereVBx-PF
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ResponsibilitySmoothing(f64);

impl ResponsibilitySmoothing {
    /// Create a positive, finite smoothing scale
    pub fn new(scale: f64) -> Result<Self, ResponsibilitySmoothingError> {
        if scale.is_finite() && scale > 0.0 {
            Ok(Self(scale))
        } else {
            Err(ResponsibilitySmoothingError(scale))
        }
    }

    /// Return the smoothing scale
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Error returned for an invalid responsibility smoothing scale
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("responsibility smoothing must be finite and greater than zero, got {0}")]
pub struct ResponsibilitySmoothingError(f64);

/// Positive convergence tolerance for SphereVBx-PF responsibilities
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SphereVbxResponsibilityTolerance(f64);

impl SphereVbxResponsibilityTolerance {
    /// Create a positive, finite responsibility tolerance
    pub fn new(tolerance: f64) -> Result<Self, SphereVbxPfConfigError> {
        if tolerance.is_finite() && tolerance > 0.0 {
            Ok(Self(tolerance))
        } else {
            Err(SphereVbxPfConfigError::InvalidResponsibilityTolerance(
                tolerance,
            ))
        }
    }

    /// Return the responsibility tolerance
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Responsibility initialization for SphereVBx-PF
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SphereVbxInitialization {
    /// Use hard one-hot AHC responsibilities
    Hard,
    /// Apply row-wise softmax to scaled one-hot AHC responsibilities
    Smoothed(ResponsibilitySmoothing),
}

/// Feature space used for AHC initialization before SphereVBx-PF
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SphereVbxAhcInitialization {
    /// Cluster the original length-normalized embeddings by cosine distance
    #[default]
    Cosine,
    /// Cluster the PLDA-transformed embeddings by cosine distance
    PldaTransformed,
}

/// Valid parameter-free SphereVBx configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereVbxPfConfig {
    fa: f64,
    fb: f64,
    max_iters: std::num::NonZeroUsize,
    responsibility_tolerance: SphereVbxResponsibilityTolerance,
    initialization: SphereVbxInitialization,
    ahc_initialization: SphereVbxAhcInitialization,
}

impl SphereVbxPfConfig {
    /// Create a validated parameter-free SphereVBx configuration
    pub fn new(
        fa: f64,
        fb: f64,
        max_iters: usize,
        responsibility_tolerance: f64,
        initialization: SphereVbxInitialization,
        ahc_initialization: SphereVbxAhcInitialization,
    ) -> Result<Self, SphereVbxPfConfigError> {
        if !fa.is_finite() || fa <= 0.0 {
            return Err(SphereVbxPfConfigError::InvalidFa(fa));
        }
        if !fb.is_finite() || fb <= 0.0 {
            return Err(SphereVbxPfConfigError::InvalidFb(fb));
        }
        let Some(max_iters) = std::num::NonZeroUsize::new(max_iters) else {
            return Err(SphereVbxPfConfigError::ZeroIterations);
        };
        let responsibility_tolerance =
            SphereVbxResponsibilityTolerance::new(responsibility_tolerance)?;

        Ok(Self {
            fa,
            fb,
            max_iters,
            responsibility_tolerance,
            initialization,
            ahc_initialization,
        })
    }

    /// Return the sufficient-statistics scale
    pub const fn fa(self) -> f64 {
        self.fa
    }

    /// Return the speaker regularization scale
    pub const fn fb(self) -> f64 {
        self.fb
    }

    /// Return the maximum number of update iterations
    pub const fn max_iters(self) -> usize {
        self.max_iters.get()
    }

    /// Return the responsibility convergence tolerance
    pub const fn responsibility_tolerance(self) -> SphereVbxResponsibilityTolerance {
        self.responsibility_tolerance
    }

    /// Return the responsibility initialization policy
    pub const fn initialization(self) -> SphereVbxInitialization {
        self.initialization
    }

    /// Return the AHC initialization feature space
    pub const fn ahc_initialization(self) -> SphereVbxAhcInitialization {
        self.ahc_initialization
    }
}

/// Error returned for an invalid SphereVBx-PF configuration
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SphereVbxPfConfigError {
    /// `FA` is not positive and finite
    #[error("SphereVBx-PF FA must be finite and greater than zero, got {0}")]
    InvalidFa(f64),
    /// `FB` is not positive and finite
    #[error("SphereVBx-PF FB must be finite and greater than zero, got {0}")]
    InvalidFb(f64),
    /// No update iteration was requested
    #[error("SphereVBx-PF max_iters must be greater than zero")]
    ZeroIterations,
    /// The convergence tolerance is not positive and finite
    #[error("SphereVBx-PF responsibility tolerance must be finite and greater than zero, got {0}")]
    InvalidResponsibilityTolerance(f64),
}

/// Error returned when SphereVBx-PF receives invalid numerical input
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SphereVbxError {
    /// Feature and responsibility rows do not agree
    #[error(
        "SphereVBx-PF received {feature_rows} feature rows and {responsibility_rows} responsibility rows"
    )]
    RowCountMismatch {
        /// Number of feature rows
        feature_rows: usize,
        /// Number of responsibility rows
        responsibility_rows: usize,
    },
    /// No feature dimension or speaker column was supplied
    #[error("SphereVBx-PF requires at least two feature dimensions and one speaker")]
    EmptyDimension,
    /// A feature row cannot be placed on the unit sphere
    #[error("SphereVBx-PF feature row {row} is non-finite or has zero norm")]
    InvalidFeatureRow {
        /// Zero-based invalid row index
        row: usize,
    },
    /// Initial responsibilities are not finite, non-negative row distributions
    #[error("SphereVBx-PF responsibility row {row} is not a finite non-negative distribution")]
    InvalidResponsibilityRow {
        /// Zero-based invalid row index
        row: usize,
    },
}

/// Run parameter-free spherical variational Bayes clustering
pub fn sphere_vbx_pf(
    features: &ArrayView2<f32>,
    gamma_init: &Array2<f32>,
    config: &SphereVbxPfConfig,
) -> Result<(Array2<f32>, Array1<f32>), SphereVbxError> {
    let (sample_count, dimension) = features.dim();
    let speaker_count = gamma_init.ncols();
    if gamma_init.nrows() != sample_count {
        return Err(SphereVbxError::RowCountMismatch {
            feature_rows: sample_count,
            responsibility_rows: gamma_init.nrows(),
        });
    }
    if dimension < 2 || speaker_count == 0 {
        return Err(SphereVbxError::EmptyDimension);
    }

    let features = normalized_features(features)?;
    let mut gamma = normalized_responsibilities(gamma_init)?;
    let mut pi = gamma.sum_axis(Axis(0));
    normalize_nonzero(&mut pi);
    let mut log_scores = Array2::<f64>::zeros((sample_count, speaker_count));
    let mut scratch = Array1::<f64>::zeros(speaker_count);

    for _ in 0..config.max_iters() {
        let expected_directions = expected_speaker_directions(&features, &gamma, *config);

        for sample_idx in 0..sample_count {
            for speaker_idx in 0..speaker_count {
                log_scores[[sample_idx, speaker_idx]] = config.fa()
                    * features
                        .row(sample_idx)
                        .dot(&expected_directions.row(speaker_idx));
            }
        }

        let previous_gamma = gamma.clone();
        for sample_idx in 0..sample_count {
            for speaker_idx in 0..speaker_count {
                scratch[speaker_idx] = if pi[speaker_idx] > 0.0 {
                    log_scores[[sample_idx, speaker_idx]] + pi[speaker_idx].ln()
                } else {
                    f64::NEG_INFINITY
                };
            }
            let denominator = logsumexp_f64(&scratch.view());
            for speaker_idx in 0..speaker_count {
                gamma[[sample_idx, speaker_idx]] = (scratch[speaker_idx] - denominator).exp();
            }
        }

        pi = gamma.sum_axis(Axis(0));
        normalize_nonzero(&mut pi);
        let maximum_change = gamma
            .iter()
            .zip(previous_gamma.iter())
            .map(|(current, previous)| (current - previous).abs())
            .fold(0.0, f64::max);
        if maximum_change <= config.responsibility_tolerance().get() {
            break;
        }
    }

    Ok((
        gamma.mapv(|value| value as f32),
        pi.mapv(|value| value as f32),
    ))
}

/// Run SphereVBx-PF from AHC labels
pub fn cluster_sphere_vbx_pf(
    ahc_labels: &[usize],
    features: &ArrayView2<f32>,
    config: &SphereVbxPfConfig,
) -> Result<(Array2<f32>, Array1<f32>), SphereVbxError> {
    let gamma_init = build_gamma_init(ahc_labels, config.initialization());
    sphere_vbx_pf(features, &gamma_init, config)
}

fn normalized_features(features: &ArrayView2<f32>) -> Result<Array2<f64>, SphereVbxError> {
    let mut normalized = Array2::<f64>::zeros(features.dim());
    for (row_idx, row) in features.rows().into_iter().enumerate() {
        let norm = row
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(SphereVbxError::InvalidFeatureRow { row: row_idx });
        }
        for (column_idx, value) in row.iter().enumerate() {
            normalized[[row_idx, column_idx]] = f64::from(*value) / norm;
        }
    }
    Ok(normalized)
}

fn normalized_responsibilities(gamma_init: &Array2<f32>) -> Result<Array2<f64>, SphereVbxError> {
    let mut gamma = gamma_init.mapv(f64::from);
    for (row_idx, mut row) in gamma.rows_mut().into_iter().enumerate() {
        if row.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(SphereVbxError::InvalidResponsibilityRow { row: row_idx });
        }
        let sum = row.sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(SphereVbxError::InvalidResponsibilityRow { row: row_idx });
        }
        row /= sum;
    }
    Ok(gamma)
}

fn normalize_nonzero(values: &mut Array1<f64>) {
    let sum = values.sum();
    if sum > 0.0 {
        *values /= sum;
    }
}

fn expected_speaker_directions(
    features: &Array2<f64>,
    gamma: &Array2<f64>,
    config: SphereVbxPfConfig,
) -> Array2<f64> {
    let speaker_count = gamma.ncols();
    let dimension = features.ncols();
    let mut expected = Array2::<f64>::zeros((speaker_count, dimension));
    let scale = config.fa() / config.fb();

    for speaker_idx in 0..speaker_count {
        let mut natural_parameter = Array1::<f64>::zeros(dimension);
        for sample_idx in 0..features.nrows() {
            natural_parameter.scaled_add(
                scale * gamma[[sample_idx, speaker_idx]],
                &features.row(sample_idx),
            );
        }
        let kappa = natural_parameter.dot(&natural_parameter).sqrt();
        if kappa == 0.0 {
            continue;
        }
        let mean_length = vmf_mean_length_ratio(dimension, kappa);
        expected
            .row_mut(speaker_idx)
            .assign(&natural_parameter.mapv(|value| value * mean_length / kappa));
    }
    expected
}

fn build_gamma_init(labels: &[usize], initialization: SphereVbxInitialization) -> Array2<f32> {
    let speaker_count = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut gamma = Array2::<f32>::zeros((labels.len(), speaker_count));
    for (row_idx, label) in labels.iter().copied().enumerate() {
        gamma[[row_idx, label]] = 1.0;
    }

    let SphereVbxInitialization::Smoothed(smoothing) = initialization else {
        return gamma;
    };
    let scale = smoothing.get() as f32;
    for mut row in gamma.rows_mut() {
        row *= scale;
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|value| (value - maximum).exp());
        let denominator = row.sum();
        row /= denominator;
    }
    gamma
}

fn vmf_mean_length_ratio(dimension: usize, kappa: f64) -> f64 {
    if kappa == 0.0 {
        return 0.0;
    }
    let nu = dimension as f64 / 2.0 - 1.0;
    if kappa > 100_000.0 {
        let first = (nu + 0.5) / kappa;
        let second = (4.0 * nu * nu - 1.0) / (8.0 * kappa * kappa);
        return (1.0 - first + second).clamp(0.0, 1.0);
    }

    // backward Perron continued fraction for I_(nu+1)(kappa) / I_nu(kappa)
    let depth = (8.0 * kappa.sqrt()).ceil().max(32.0) as usize;
    let squared = kappa * kappa;
    let mut denominator = 2.0 * (nu + depth as f64);
    for offset in (1..depth).rev() {
        denominator = 2.0 * (nu + offset as f64) + squared / denominator;
    }
    (kappa / denominator).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::PathBuf;

    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};
    use ndarray_npy::ReadNpyExt;

    use super::*;

    fn config(initialization: SphereVbxInitialization) -> SphereVbxPfConfig {
        SphereVbxPfConfig::new(
            12.0,
            0.3,
            10,
            1e-8,
            initialization,
            SphereVbxAhcInitialization::Cosine,
        )
        .unwrap()
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn bessel_ratio_matches_small_kappa_limit() {
        let cases = [
            (0.001, 0.000_003_906_249_999_940_858),
            (0.1, 0.000_390_624_940_857_424_6),
        ];
        for (kappa, expected) in cases {
            assert_abs_diff_eq!(vmf_mean_length_ratio(256, kappa), expected, epsilon = 1e-14);
        }
    }

    #[test]
    fn bessel_ratio_matches_finite_scipy_reference_values() {
        let cases = [
            (1.0, 0.003_906_190_859_183_730_6),
            (10.0, 0.039_003_534_458_179_96),
            (100.0, 0.344_539_325_158_259_75),
            (1_000.0, 0.880_540_065_591_269_4),
            (10_000.0, 0.987_330_648_562_838_9),
            (100_000.0, 0.998_725_806_445_239_5),
        ];
        for (kappa, expected) in cases {
            assert_abs_diff_eq!(vmf_mean_length_ratio(256, kappa), expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn hard_initialization_uses_normalized_cluster_mean() {
        let features = array![[1.0_f32, 0.0], [0.8, 0.6], [-1.0, 0.0], [-0.8, -0.6]];
        let gamma = build_gamma_init(&[0, 0, 1, 1], SphereVbxInitialization::Hard);
        let normalized = normalized_features(&features.view()).unwrap();
        let expected = expected_speaker_directions(
            &normalized,
            &gamma.mapv(f64::from),
            config(SphereVbxInitialization::Hard),
        );

        assert!(expected[[0, 0]] > 0.0);
        assert!(expected[[0, 1]] > 0.0);
        assert_abs_diff_eq!(expected[[0, 0]], -expected[[1, 0]], epsilon = 1e-12);
        assert_abs_diff_eq!(expected[[0, 1]], -expected[[1, 1]], epsilon = 1e-12);
    }

    #[test]
    fn empty_speaker_has_zero_expected_direction_and_prior() {
        let features = array![[1.0_f32, 0.0], [0.8, 0.6]];
        let gamma = array![[1.0_f32, 0.0], [1.0, 0.0]];
        let normalized = normalized_features(&features.view()).unwrap();
        let expected = expected_speaker_directions(
            &normalized,
            &gamma.mapv(f64::from),
            config(SphereVbxInitialization::Hard),
        );
        assert_eq!(expected.row(1).sum(), 0.0);

        let (_, pi) = sphere_vbx_pf(
            &features.view(),
            &gamma,
            &config(SphereVbxInitialization::Hard),
        )
        .unwrap();
        assert_eq!(pi[1], 0.0);
    }

    #[test]
    fn invalid_config_values_are_rejected() {
        assert!(
            SphereVbxPfConfig::new(
                0.0,
                0.3,
                10,
                1e-8,
                SphereVbxInitialization::Hard,
                SphereVbxAhcInitialization::Cosine,
            )
            .is_err()
        );
        assert!(
            SphereVbxPfConfig::new(
                12.0,
                f64::NAN,
                10,
                1e-8,
                SphereVbxInitialization::Hard,
                SphereVbxAhcInitialization::Cosine,
            )
            .is_err()
        );
        assert!(
            SphereVbxPfConfig::new(
                12.0,
                0.3,
                0,
                1e-8,
                SphereVbxInitialization::Hard,
                SphereVbxAhcInitialization::Cosine,
            )
            .is_err()
        );
    }

    #[test]
    fn sphere_vbx_matches_scipy_equations_fixture() {
        let features: Array2<f64> =
            Array2::read_npy(File::open(fixture_path("sphere_vbx_pf_features.npy")).unwrap())
                .unwrap();
        let gamma_init: Array2<f64> =
            Array2::read_npy(File::open(fixture_path("sphere_vbx_pf_initial_gamma.npy")).unwrap())
                .unwrap();
        let expected_gamma: Array2<f64> =
            Array2::read_npy(File::open(fixture_path("sphere_vbx_pf_gamma.npy")).unwrap()).unwrap();
        let expected_pi: Array1<f64> =
            Array1::read_npy(File::open(fixture_path("sphere_vbx_pf_pi.npy")).unwrap()).unwrap();
        let features = features.mapv(|value| value as f32);
        let gamma_init = gamma_init.mapv(|value| value as f32);
        let fixture_config = SphereVbxPfConfig::new(
            1.5,
            6.0,
            4,
            1e-12,
            SphereVbxInitialization::Hard,
            SphereVbxAhcInitialization::Cosine,
        )
        .unwrap();

        let (gamma, pi) = sphere_vbx_pf(&features.view(), &gamma_init, &fixture_config).unwrap();

        assert_eq!(gamma.dim(), expected_gamma.dim());
        assert_eq!(pi.len(), expected_pi.len());
        for (actual, expected) in gamma.iter().zip(expected_gamma.iter()) {
            assert_abs_diff_eq!(f64::from(*actual), *expected, epsilon = 5e-8);
        }
        for (actual, expected) in pi.iter().zip(expected_pi.iter()) {
            assert_abs_diff_eq!(f64::from(*actual), *expected, epsilon = 5e-8);
        }

        assert!(
            gamma
                .iter()
                .all(|value| (0.05..0.95).contains(&f64::from(*value)))
        );
    }
}
