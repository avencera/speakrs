use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use ndarray_npy::read_npy;

use crate::linalg::{Eigh, Inverse, LinalgError, UPLO};
use crate::utils::l2_normalize_rows_f64;

/// PLDA transform computed entirely in f64 to match pyannote's numpy precision
/// Parameters are stored as f64 internally, and the transform method returns f32
/// for downstream consumption
#[derive(Debug, Clone)]
pub struct PldaTransform {
    mean1: Array1<f64>,
    mean2: Array1<f64>,
    lda: Array2<f64>,
    mu: Array1<f64>,
    transform: Array2<f64>,
    phi: Array1<f64>,
}

/// Projected embeddings paired with the matching PLDA eigenvalues
#[derive(Debug, Clone)]
pub struct PldaFeatures {
    features: Array2<f32>,
    phi: Array1<f32>,
}

impl PldaFeatures {
    /// Projected feature matrix
    pub fn features(&self) -> ArrayView2<'_, f32> {
        self.features.view()
    }

    /// Eigenvalues aligned with the projected columns
    pub fn phi(&self) -> ArrayView1<'_, f32> {
        self.phi.view()
    }
}

struct PldaParameters {
    mean1: Array1<f64>,
    mean2: Array1<f64>,
    lda: Array2<f64>,
    mu: Array1<f64>,
    raw_transform: Array2<f64>,
    psi: Array1<f64>,
}

impl PldaTransform {
    pub fn from_dir(models_dir: &Path) -> Result<Self, PldaError> {
        Self::from_parameters(PldaParameters {
            mean1: read_array1_f64(models_dir.join("plda_mean1.npy"))?,
            mean2: read_array1_f64(models_dir.join("plda_mean2.npy"))?,
            lda: read_array2_f64(models_dir.join("plda_lda.npy"))?,
            mu: read_array1_f64(models_dir.join("plda_mu.npy"))?,
            raw_transform: read_array2_f64(models_dir.join("plda_tr.npy"))?,
            psi: read_array1_f64(models_dir.join("plda_psi.npy"))?,
        })
    }

    fn from_parameters(parameters: PldaParameters) -> Result<Self, PldaError> {
        let PldaParameters {
            mean1,
            mean2,
            lda,
            mu,
            raw_transform,
            psi,
        } = parameters;
        validate_plda_parameters(&mean1, &mean2, &lda, &mu, &raw_transform, &psi)?;

        let precision_matrix = raw_transform.t().dot(&raw_transform).inv()?;

        let mut tr_over_psi = raw_transform.t().to_owned();
        for (mut column, &psi_value) in tr_over_psi.columns_mut().into_iter().zip(psi.iter()) {
            column /= psi_value;
        }
        let between_class_covariance = tr_over_psi.dot(&raw_transform).inv()?;

        let (eigenvalues, (eigenvectors, _)) =
            (between_class_covariance, precision_matrix).eigh(UPLO::Lower)?;
        if eigenvalues.len() != lda.ncols() || eigenvectors.nrows() != lda.ncols() {
            return Err(PldaError::Shape(
                "eigensystem does not match LDA output dimension".to_owned(),
            ));
        }

        let dim = lda.ncols();
        let mut phi = Array1::<f64>::zeros(dim);
        let mut transform = Array2::<f64>::zeros((dim, dim));
        for dim_idx in 0..dim {
            let src = eigenvalues.len() - 1 - dim_idx;
            phi[dim_idx] = eigenvalues[src];
            transform.row_mut(dim_idx).assign(&eigenvectors.column(src));
        }

        Ok(Self {
            mean1,
            mean2,
            lda,
            mu,
            transform,
            phi,
        })
    }

    /// Project embeddings and return features paired with matching eigenvalues
    pub fn project(&self, embeddings: &ArrayView2<f32>, lda_dim: usize) -> PldaFeatures {
        let embeddings_f64 = embeddings.mapv(|v| v as f64);
        let xvec = self.xvec_transform(&embeddings_f64.view());
        let lda_dim = lda_dim.min(self.phi.len());
        let result = self.plda_transform(&xvec.view(), lda_dim);
        PldaFeatures {
            features: result.mapv(|value| value as f32),
            phi: self.phi.slice(s![..lda_dim]).mapv(|value| value as f32),
        }
    }

    fn xvec_transform(&self, embeddings: &ArrayView2<f64>) -> Array2<f64> {
        let centered = embeddings - &self.mean1;
        let normalized = l2_normalize_rows_f64(&centered.view());
        let scaled = normalized * (self.lda.nrows() as f64).sqrt();
        let projected = scaled.dot(&self.lda);
        let centered_projected = projected - &self.mean2;
        l2_normalize_rows_f64(&centered_projected.view()) * (self.lda.ncols() as f64).sqrt()
    }

    fn plda_transform(&self, embeddings: &ArrayView2<f64>, lda_dim: usize) -> Array2<f64> {
        let lda_dim = lda_dim.min(self.transform.nrows());
        let centered = embeddings - &self.mu;
        centered.dot(&self.transform.slice(s![..lda_dim, ..]).t())
    }
}

fn validate_plda_parameters(
    mean1: &Array1<f64>,
    mean2: &Array1<f64>,
    lda: &Array2<f64>,
    mu: &Array1<f64>,
    raw_transform: &Array2<f64>,
    psi: &Array1<f64>,
) -> Result<(), PldaError> {
    if lda.nrows() == 0 || lda.ncols() == 0 {
        return Err(PldaError::Shape("LDA matrix must be non-empty".to_owned()));
    }
    if mean1.len() != lda.nrows() {
        return Err(PldaError::Shape(format!(
            "mean1 length {} does not match LDA rows {}",
            mean1.len(),
            lda.nrows()
        )));
    }
    if mean2.len() != lda.ncols() {
        return Err(PldaError::Shape(format!(
            "mean2 length {} does not match LDA columns {}",
            mean2.len(),
            lda.ncols()
        )));
    }
    if mu.len() != lda.ncols() {
        return Err(PldaError::Shape(format!(
            "mu length {} does not match LDA columns {}",
            mu.len(),
            lda.ncols()
        )));
    }
    if raw_transform.ncols() != psi.len() {
        return Err(PldaError::Shape(format!(
            "transform columns {} do not match psi length {}",
            raw_transform.ncols(),
            psi.len()
        )));
    }
    if psi.iter().any(|value| *value == 0.0) {
        return Err(PldaError::InvalidPsi);
    }
    let finite = mean1
        .iter()
        .chain(mean2.iter())
        .chain(mu.iter())
        .chain(psi.iter())
        .all(|value| value.is_finite())
        && lda.iter().all(|value| value.is_finite())
        && raw_transform.iter().all(|value| value.is_finite());
    if !finite {
        return Err(PldaError::NonFinite);
    }
    Ok(())
}

fn read_array1_f64(path: impl AsRef<Path>) -> Result<Array1<f64>, PldaError> {
    let path = path.as_ref();
    match read_npy(path) {
        Ok(values) => Ok(values),
        Err(ndarray_npy::ReadNpyError::WrongDescriptor(_)) => {
            let values: Array1<f32> = read_npy(path)?;
            Ok(values.mapv(|value| value as f64))
        }
        Err(err) => Err(PldaError::Io(err)),
    }
}

fn read_array2_f64(path: impl AsRef<Path>) -> Result<Array2<f64>, PldaError> {
    let path = path.as_ref();
    match read_npy(path) {
        Ok(values) => Ok(values),
        Err(ndarray_npy::ReadNpyError::WrongDescriptor(_)) => {
            let values: Array2<f32> = read_npy(path)?;
            Ok(values.mapv(|value| value as f64))
        }
        Err(err) => Err(PldaError::Io(err)),
    }
}

/// PLDA load or parameter error
#[derive(Debug, thiserror::Error)]
pub enum PldaError {
    /// Array file could not be read
    #[error(transparent)]
    Io(#[from] ndarray_npy::ReadNpyError),
    /// Linear algebra failed while building the transform
    #[error(transparent)]
    Linalg(#[from] LinalgError),
    /// A psi eigenvalue was zero
    #[error("plda psi contained zeros")]
    InvalidPsi,
    /// Model arrays had incompatible shapes
    #[error("plda model shape is invalid: {0}")]
    Shape(String),
    /// A model array contained a non-finite value
    #[error("plda model contained a non-finite value")]
    NonFinite,
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;

    use super::*;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn transform_from_models_has_expected_shapes() {
        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let plda = PldaTransform::from_dir(&models_dir).unwrap();
        let sample = Array2::<f32>::zeros((2, 256));

        let projected = plda.project(&sample.view(), 128);

        assert_eq!(projected.phi().len(), 128);
        assert_eq!(projected.features().dim(), (2, 128));
        assert!(projected.features().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn batch_matches_single() {
        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let plda = PldaTransform::from_dir(&models_dir).unwrap();
        let sample = Array2::<f32>::ones((2, 256));

        let projected = plda.project(&sample.view(), 128);

        for row_idx in 0..sample.nrows() {
            let row = sample.row(row_idx).to_owned().insert_axis(ndarray::Axis(0));
            let single = plda.project(&row.view(), 128);
            for (lhs, rhs) in single
                .features()
                .row(0)
                .iter()
                .zip(projected.features().row(row_idx).iter())
            {
                assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn transform_matches_python_fixture() {
        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let plda = PldaTransform::from_dir(&models_dir).unwrap();
        let train_embeddings: Array2<f32> =
            Array2::read_npy(File::open(fixture_path("pipeline_train_embeddings.npy")).unwrap())
                .unwrap();
        let expected_phi: Array1<f64> =
            Array1::read_npy(File::open(fixture_path("pipeline_plda_phi.npy")).unwrap()).unwrap();
        let expected_features: Array2<f64> =
            Array2::read_npy(File::open(fixture_path("pipeline_plda_features.npy")).unwrap())
                .unwrap();

        let projected = plda.project(&train_embeddings.view(), 128);

        for (lhs, rhs) in projected.phi().iter().zip(expected_phi.iter()) {
            assert_abs_diff_eq!(*lhs, *rhs as f32, epsilon = 1e-4);
        }

        let features = projected.features().to_owned();
        for column_idx in 0..features.ncols() {
            let actual = features.column(column_idx);
            let expected = expected_features.column(column_idx);
            let sign = if actual
                .iter()
                .zip(expected.iter())
                .map(|(lhs, rhs)| *lhs as f64 * *rhs)
                .sum::<f64>()
                < 0.0
            {
                -1.0f32
            } else {
                1.0f32
            };

            for (lhs, rhs) in actual.iter().zip(expected.iter()) {
                assert_abs_diff_eq!(*lhs * sign, *rhs as f32, epsilon = 5e-4);
            }
        }
    }

    #[test]
    fn from_parameters_rejects_shape_mismatch_and_non_finite() {
        let lda = Array2::<f64>::eye(2);
        let mean1 = Array1::zeros(2);
        let mean2 = Array1::zeros(2);
        let mu = Array1::zeros(2);
        let raw_transform = Array2::<f64>::eye(2);
        let psi = Array1::from_elem(2, 1.0);

        let short_psi = PldaTransform::from_parameters(PldaParameters {
            mean1: mean1.clone(),
            mean2: mean2.clone(),
            lda: lda.clone(),
            mu: mu.clone(),
            raw_transform: raw_transform.clone(),
            psi: Array1::from_elem(1, 1.0),
        });
        assert!(matches!(short_psi, Err(PldaError::Shape(_))));

        let zero_psi = PldaTransform::from_parameters(PldaParameters {
            mean1: mean1.clone(),
            mean2: mean2.clone(),
            lda: lda.clone(),
            mu: mu.clone(),
            raw_transform: raw_transform.clone(),
            psi: Array1::zeros(2),
        });
        assert!(matches!(zero_psi, Err(PldaError::InvalidPsi)));

        let mut nan_mean = mean1.clone();
        nan_mean[0] = f64::NAN;
        let non_finite = PldaTransform::from_parameters(PldaParameters {
            mean1: nan_mean,
            mean2,
            lda,
            mu,
            raw_transform,
            psi,
        });
        assert!(matches!(non_finite, Err(PldaError::NonFinite)));
    }
}
