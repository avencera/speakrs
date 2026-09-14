use std::fs;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use ndarray_npy::read_npy;
use sha2::{Digest, Sha256};

use crate::imported_segmentation::{IdentityReference, Sha256Digest};
use crate::linalg::{Eigh, Inverse, LinalgError, UPLO};
use crate::utils::l2_normalize_rows_f64;

/// Stable identity for the fixed PLDA artifact accepted by the imported path
pub const PLDA_ARTIFACT_ID: &str = "plda";
/// Revision of the fixed PLDA artifact accepted by the imported path
pub const PLDA_ARTIFACT_REVISION: &str = "b2-fixed";

/// Identity receipt owned by a loaded PLDA artifact
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PldaArtifactReceipt {
    sha256: Sha256Digest,
}

impl PldaArtifactReceipt {
    /// Return the fixed PLDA artifact identity
    pub const fn id(&self) -> &'static str {
        PLDA_ARTIFACT_ID
    }

    /// Return the fixed PLDA artifact revision
    pub const fn revision(&self) -> &'static str {
        PLDA_ARTIFACT_REVISION
    }

    /// Return the digest associated with the loaded PLDA parameters
    pub fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }

    fn verify(&self, expected: &IdentityReference) -> Result<(), PldaError> {
        verify_identity("plda.id", expected.id.as_str(), self.id())?;
        verify_identity("plda.revision", expected.revision.as_str(), self.revision())?;
        verify_identity(
            "plda.sha256",
            expected.sha256.as_str(),
            self.sha256.as_str(),
        )
    }
}

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
    receipt: PldaArtifactReceipt,
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
    /// Load PLDA parameters from a directory without binding them to an imported manifest
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

    /// Load PLDA parameters and verify their identity against an imported manifest reference
    pub fn from_imported_artifact(
        models_dir: &Path,
        expected: &IdentityReference,
    ) -> Result<Self, PldaError> {
        let mut transform = Self::from_dir(models_dir)?;
        transform.receipt.sha256 = digest_tree(models_dir)?;
        transform.receipt.verify(expected)?;
        Ok(transform)
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
        for column_idx in 0..raw_transform.nrows() {
            let mut column = tr_over_psi.column_mut(column_idx);
            column /= psi[column_idx];
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

        let receipt = PldaArtifactReceipt {
            sha256: digest_parameters(&mean1, &mean2, &lda, &mu, &raw_transform, &psi),
        };

        Ok(Self {
            mean1,
            mean2,
            lda,
            mu,
            transform,
            phi,
            receipt,
        })
    }

    /// Return the identity of the loaded PLDA artifact
    pub fn receipt(&self) -> &PldaArtifactReceipt {
        &self.receipt
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

fn digest_parameters(
    mean1: &Array1<f64>,
    mean2: &Array1<f64>,
    lda: &Array2<f64>,
    mu: &Array1<f64>,
    raw_transform: &Array2<f64>,
    psi: &Array1<f64>,
) -> Sha256Digest {
    let mut digest = Sha256::new();
    for (name, values) in [
        ("mean1", mean1.as_slice().unwrap_or(&[])),
        ("mean2", mean2.as_slice().unwrap_or(&[])),
        ("lda", lda.as_slice().unwrap_or(&[])),
        ("mu", mu.as_slice().unwrap_or(&[])),
        ("transform", raw_transform.as_slice().unwrap_or(&[])),
        ("psi", psi.as_slice().unwrap_or(&[])),
    ] {
        update_length_prefixed(&mut digest, name.as_bytes());
        digest.update((values.len() as u64).to_le_bytes());
        for value in values {
            digest.update(value.to_le_bytes());
        }
    }
    finalize_digest(digest)
}

fn digest_tree(root: &Path) -> Result<Sha256Digest, PldaError> {
    let metadata = fs::symlink_metadata(root).map_err(|error| PldaError::ArtifactIo {
        path: root.to_owned(),
        message: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PldaError::ArtifactIo {
            path: root.to_owned(),
            message: "must be a real directory".to_owned(),
        });
    }
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    for (relative, path) in files {
        let bytes = fs::read(&path).map_err(|error| PldaError::ArtifactIo {
            path: path.clone(),
            message: error.to_string(),
        })?;
        update_length_prefixed(&mut digest, relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(finalize_digest(digest))
}

fn collect_files(
    root: &Path,
    current: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), PldaError> {
    let entries = fs::read_dir(current).map_err(|error| PldaError::ArtifactIo {
        path: current.to_owned(),
        message: error.to_string(),
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| PldaError::ArtifactIo {
            path: current.to_owned(),
            message: error.to_string(),
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| PldaError::ArtifactIo {
            path: path.clone(),
            message: error.to_string(),
        })?;
        if metadata.file_type().is_symlink() {
            return Err(PldaError::ArtifactIo {
                path,
                message: "symlinks are not allowed".to_owned(),
            });
        }
        if metadata.is_dir() {
            collect_files(root, &path, files)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| PldaError::ArtifactIo {
                    path: path.clone(),
                    message: "file escaped artifact root".to_owned(),
                })?
                .to_string_lossy()
                .replace('\\', "/");
            files.push((relative, path));
        } else {
            return Err(PldaError::ArtifactIo {
                path,
                message: "unsupported artifact member".to_owned(),
            });
        }
    }
    Ok(())
}

fn update_length_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn finalize_digest(digest: Sha256) -> Sha256Digest {
    Sha256Digest::new(format!("{:x}", digest.finalize()))
        .expect("SHA-256 formatting always produces a canonical digest")
}

fn verify_identity(field: &'static str, expected: &str, actual: &str) -> Result<(), PldaError> {
    if expected == actual {
        return Ok(());
    }
    Err(PldaError::ArtifactIdentityMismatch {
        field,
        expected: expected.to_owned(),
        actual: actual.to_owned(),
    })
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
    if raw_transform.nrows() == 0 || raw_transform.ncols() == 0 {
        return Err(PldaError::Shape(
            "PLDA transform must be non-empty".to_owned(),
        ));
    }
    if raw_transform.nrows() < raw_transform.ncols() {
        return Err(PldaError::Shape(format!(
            "transform rows {} are fewer than columns {}; TᵀT cannot be invertible",
            raw_transform.nrows(),
            raw_transform.ncols()
        )));
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
    if raw_transform.ncols() != lda.ncols() {
        return Err(PldaError::Shape(format!(
            "transform columns {} do not match LDA columns {}",
            raw_transform.ncols(),
            lda.ncols()
        )));
    }
    if raw_transform.nrows() != psi.len() {
        return Err(PldaError::Shape(format!(
            "transform rows {} do not match psi length {}",
            raw_transform.nrows(),
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
    /// The PLDA artifact inventory could not be inspected or read
    #[error("plda artifact `{path}` is invalid or unreadable: {message}")]
    ArtifactIo {
        /// Path that failed inspection
        path: PathBuf,
        /// Inspection or read failure
        message: String,
    },
    /// The loaded artifact did not match the imported identity reference
    #[error("plda artifact identity mismatch for {field}: expected {expected}, got {actual}")]
    ArtifactIdentityMismatch {
        /// Identity field that differed
        field: &'static str,
        /// Identity required by the imported manifest
        expected: String,
        /// Identity measured from the loaded artifact
        actual: String,
    },
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
    use ndarray::array;
    use ndarray_npy::ReadNpyExt;
    use std::fs::{self, File};

    use super::*;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name)
    }

    fn parameters_for(
        lda: Array2<f64>,
        raw_transform: Array2<f64>,
        psi: Array1<f64>,
    ) -> PldaParameters {
        let input_dim = lda.nrows();
        let output_dim = lda.ncols();
        PldaParameters {
            mean1: Array1::zeros(input_dim),
            mean2: Array1::zeros(output_dim),
            lda,
            mu: Array1::zeros(output_dim),
            raw_transform,
            psi,
        }
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
    fn loaded_models_have_a_stable_receipt() {
        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let first = PldaTransform::from_dir(&models_dir).unwrap();
        let second = PldaTransform::from_dir(&models_dir).unwrap();

        assert_eq!(first.receipt(), second.receipt());
        assert_eq!(first.receipt().id(), PLDA_ARTIFACT_ID);
        assert_eq!(first.receipt().revision(), PLDA_ARTIFACT_REVISION);
    }

    #[test]
    fn imported_loader_binds_the_complete_tree_digest() {
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let directory = tempfile::tempdir().unwrap();
        for name in [
            "plda_mean1.npy",
            "plda_mean2.npy",
            "plda_lda.npy",
            "plda_mu.npy",
            "plda_tr.npy",
            "plda_psi.npy",
        ] {
            fs::copy(source_dir.join(name), directory.path().join(name)).unwrap();
        }
        let models_dir = directory.path();
        let expected = IdentityReference {
            id: PLDA_ARTIFACT_ID.to_owned(),
            revision: PLDA_ARTIFACT_REVISION.to_owned(),
            sha256: digest_tree(models_dir).unwrap(),
        };

        let plda = PldaTransform::from_imported_artifact(models_dir, &expected).unwrap();

        assert_eq!(plda.receipt().sha256(), &expected.sha256);
    }

    #[test]
    fn tree_digest_matches_cross_language_vector() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("nested")).unwrap();
        fs::write(directory.path().join("a.txt"), b"alpha").unwrap();
        fs::write(directory.path().join("nested/b.bin"), [0, 1, 2, 255]).unwrap();

        // this vector matches the Python bridge tree digest contract
        assert_eq!(
            digest_tree(directory.path()).unwrap().as_str(),
            "48c74f4785da606a9f2147aa1d62640e8d2f9008b8101281983344624ca64e18"
        );
    }

    #[cfg(unix)]
    #[test]
    fn permissive_loader_accepts_symlinked_snapshot_files() {
        let directory = tempfile::tempdir().unwrap();
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let snapshot_dir = directory.path().join("snapshot");
        fs::create_dir(&snapshot_dir).unwrap();
        for name in [
            "plda_mean1.npy",
            "plda_mean2.npy",
            "plda_lda.npy",
            "plda_mu.npy",
            "plda_tr.npy",
            "plda_psi.npy",
        ] {
            std::os::unix::fs::symlink(source_dir.join(name), snapshot_dir.join(name)).unwrap();
        }

        assert!(PldaTransform::from_dir(&snapshot_dir).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn strict_imported_loader_rejects_a_symlinked_model_directory() {
        let directory = tempfile::tempdir().unwrap();
        let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let link = directory.path().join("models");
        std::os::unix::fs::symlink(models_dir, &link).unwrap();
        let expected = IdentityReference {
            id: PLDA_ARTIFACT_ID.to_owned(),
            revision: PLDA_ARTIFACT_REVISION.to_owned(),
            sha256: Sha256Digest::digest(b"expected"),
        };

        assert!(matches!(
            PldaTransform::from_imported_artifact(&link, &expected),
            Err(PldaError::ArtifactIo { message, .. }) if message == "must be a real directory"
        ));
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
    fn tall_transform_requires_one_psi_value_per_transform_row() {
        let result = PldaTransform::from_parameters(parameters_for(
            Array2::eye(2),
            array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            array![2.0, 3.0],
        ));

        assert!(matches!(result, Err(PldaError::Shape(_))));
    }

    #[test]
    fn tall_full_rank_transform_is_valid_when_psi_matches_rows() {
        let result = PldaTransform::from_parameters(parameters_for(
            Array2::eye(2),
            array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            array![2.0, 3.0, 5.0],
        ));

        assert!(result.is_ok());
    }

    #[test]
    fn rejects_wide_and_empty_transforms() {
        let wide = PldaTransform::from_parameters(parameters_for(
            Array2::eye(3),
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            array![2.0, 3.0],
        ));
        assert!(matches!(wide, Err(PldaError::Shape(_))));

        let empty = PldaTransform::from_parameters(parameters_for(
            Array2::eye(2),
            Array2::zeros((0, 2)),
            Array1::zeros(0),
        ));
        assert!(matches!(empty, Err(PldaError::Shape(_))));
    }

    #[test]
    fn rejects_transform_with_lda_incompatible_output_width() {
        let result = PldaTransform::from_parameters(parameters_for(
            Array2::zeros((2, 3)),
            Array2::eye(2),
            array![2.0, 3.0],
        ));

        assert!(matches!(result, Err(PldaError::Shape(_))));
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
