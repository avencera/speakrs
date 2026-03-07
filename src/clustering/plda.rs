use std::fmt::{Display, Formatter};
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use ndarray_npy::read_npy;

use crate::utils::l2_normalize_rows;

#[derive(Debug, Clone)]
pub struct PldaTransform {
    mean1: Array1<f32>,
    mean2: Array1<f32>,
    lda: Array2<f32>,
    mu: Array1<f32>,
    transform: Array2<f32>,
    phi: Array1<f32>,
}

impl PldaTransform {
    pub fn from_dir(models_dir: &Path) -> Result<Self, PldaError> {
        let mean1 = read_array1_f32(models_dir.join("plda_mean1.npy"))?;
        let mean2 = read_array1_f32(models_dir.join("plda_mean2.npy"))?;
        let lda = read_array2_f32(models_dir.join("plda_lda.npy"))?;
        let mu = read_array1_f32(models_dir.join("plda_mu.npy"))?;
        let tr = read_array2_f32(models_dir.join("plda_tr.npy"))?;
        let psi = read_array1_f32(models_dir.join("plda_psi.npy"))?;

        let w = tr.t().dot(&tr).inv()?;

        let mut tr_over_psi = tr.t().to_owned();
        for (mut column, &psi_value) in tr_over_psi.columns_mut().into_iter().zip(psi.iter()) {
            if psi_value == 0.0 {
                return Err(PldaError::InvalidPsi);
            }
            column /= psi_value;
        }
        let b = tr_over_psi.dot(&tr).inv()?;

        let (eigenvalues, (eigenvectors, _)) = (b, w).eigh(UPLO::Lower)?;

        let dim = lda.ncols();
        let mut phi = Array1::<f32>::zeros(dim);
        let mut transform = Array2::<f32>::zeros((dim, dim));
        for idx in 0..dim {
            let src = eigenvalues.len() - 1 - idx;
            phi[idx] = eigenvalues[src];
            transform.row_mut(idx).assign(&eigenvectors.column(src));
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

    pub fn phi(&self) -> &Array1<f32> {
        &self.phi
    }

    pub fn transform(&self, embeddings: &ArrayView2<f32>, lda_dim: usize) -> Array2<f32> {
        let xvec = self.xvec_transform(embeddings);
        self.plda_transform(&xvec.view(), lda_dim)
    }

    pub fn transform_one(&self, embedding: &ArrayView1<f32>, lda_dim: usize) -> Array1<f32> {
        let batch = embedding.to_owned().insert_axis(Axis(0));
        self.transform(&batch.view(), lda_dim).row(0).to_owned()
    }

    fn xvec_transform(&self, embeddings: &ArrayView2<f32>) -> Array2<f32> {
        let centered = embeddings - &self.mean1;
        let normalized = l2_normalize_rows(&centered.view());
        let scaled = normalized * (self.lda.nrows() as f32).sqrt();
        let projected = scaled.dot(&self.lda);
        let centered_projected = projected - &self.mean2;
        l2_normalize_rows(&centered_projected.view()) * (self.lda.ncols() as f32).sqrt()
    }

    fn plda_transform(&self, embeddings: &ArrayView2<f32>, lda_dim: usize) -> Array2<f32> {
        let lda_dim = lda_dim.min(self.transform.nrows());
        let centered = embeddings - &self.mu;
        centered.dot(&self.transform.slice(s![..lda_dim, ..]).t())
    }
}

fn read_array1_f32(path: impl AsRef<Path>) -> Result<Array1<f32>, PldaError> {
    let path = path.as_ref();
    match read_npy(path) {
        Ok(values) => Ok(values),
        Err(ndarray_npy::ReadNpyError::WrongDescriptor(_)) => {
            let values: Array1<f64> = read_npy(path)?;
            Ok(values.mapv(|value| value as f32))
        }
        Err(err) => Err(PldaError::Io(err)),
    }
}

fn read_array2_f32(path: impl AsRef<Path>) -> Result<Array2<f32>, PldaError> {
    let path = path.as_ref();
    match read_npy(path) {
        Ok(values) => Ok(values),
        Err(ndarray_npy::ReadNpyError::WrongDescriptor(_)) => {
            let values: Array2<f64> = read_npy(path)?;
            Ok(values.mapv(|value| value as f32))
        }
        Err(err) => Err(PldaError::Io(err)),
    }
}

#[derive(Debug)]
pub enum PldaError {
    Io(ndarray_npy::ReadNpyError),
    Linalg(ndarray_linalg::error::LinalgError),
    InvalidPsi,
}

impl Display for PldaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Linalg(err) => write!(f, "{err}"),
            Self::InvalidPsi => write!(f, "plda psi contained zeros"),
        }
    }
}

impl std::error::Error for PldaError {}

impl From<ndarray_npy::ReadNpyError> for PldaError {
    fn from(value: ndarray_npy::ReadNpyError) -> Self {
        Self::Io(value)
    }
}

impl From<ndarray_linalg::error::LinalgError> for PldaError {
    fn from(value: ndarray_linalg::error::LinalgError) -> Self {
        Self::Linalg(value)
    }
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

        let transformed = plda.transform(&sample.view(), 128);

        assert_eq!(plda.phi().len(), 128);
        assert_eq!(transformed.dim(), (2, 128));
        assert!(transformed.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn batch_matches_single() {
        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let plda = PldaTransform::from_dir(&models_dir).unwrap();
        let sample = Array2::<f32>::ones((2, 256));

        let transformed = plda.transform(&sample.view(), 128);

        for row_idx in 0..sample.nrows() {
            let single = plda.transform_one(&sample.row(row_idx), 128);
            for (lhs, rhs) in single.iter().zip(transformed.row(row_idx).iter()) {
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

        let transformed = plda.transform(&train_embeddings.view(), 128);

        for (lhs, rhs) in plda.phi().iter().zip(expected_phi.iter()) {
            assert_abs_diff_eq!(*lhs, *rhs as f32, epsilon = 1e-4);
        }

        for column_idx in 0..transformed.ncols() {
            let actual = transformed.column(column_idx);
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
}
