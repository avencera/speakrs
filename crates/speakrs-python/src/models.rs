//! Model preparation, manifests, and the download cache

use std::path::PathBuf;

use pyo3::prelude::*;
use speakrs_sdk::{
    CacheEntry, CancelToken, ModelManifest, ModelManifestEntry, ModelManifestError,
    ModelPrepareError, ModelStore, PrepareModelsOptions, PreparedModels, ProgressEvent, SdkError,
    SdkErrorCategory, mode_name,
};

use crate::pipeline::ProgressSink;
use crate::{category_err, mode_from, py_opt_str, sdk_err};

fn prepare_sdk_err(error: ModelPrepareError) -> SdkError {
    match error {
        ModelPrepareError::Manifest(error) => {
            SdkError::new(SdkErrorCategory::ModelManifest, error.to_string())
        }
        ModelPrepareError::Cancelled => SdkError::cancelled(),
        other => SdkError::new(SdkErrorCategory::ModelPrepare, other.to_string()),
    }
}

fn prepare_err(error: ModelPrepareError) -> PyErr {
    sdk_err(prepare_sdk_err(error))
}

fn manifest_err(error: ModelManifestError) -> PyErr {
    category_err(SdkErrorCategory::ModelManifest, error)
}

/// Size and SHA-256 digest of one model file
#[pyclass(name = "ModelManifestEntry", module = "speakrs", frozen)]
#[derive(Clone)]
pub(crate) struct PyModelManifestEntry(pub(crate) ModelManifestEntry);

#[pymethods]
impl PyModelManifestEntry {
    /// Path relative to the model directory
    #[getter]
    fn path(&self) -> &str {
        &self.0.path
    }

    /// File size in bytes
    #[getter]
    fn size_bytes(&self) -> u64 {
        self.0.size_bytes
    }

    /// Lowercase hex SHA-256 digest
    #[getter]
    fn sha256(&self) -> &str {
        &self.0.sha256
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelManifestEntry(path='{}', size_bytes={})",
            self.0.path, self.0.size_bytes
        )
    }
}

/// Sizes and digests of the files one mode loads
///
/// Save it with `to_json()` and pass it to `prepare(manifest=...)` later to check
/// that a model directory has not changed.
#[pyclass(name = "ModelManifest", module = "speakrs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyModelManifest(pub(crate) ModelManifest);

#[pymethods]
impl PyModelManifest {
    /// Parse a manifest written by `to_json()`
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        ModelManifest::from_json(json)
            .map(Self)
            .map_err(manifest_err)
    }

    /// Serialize as JSON
    fn to_json(&self) -> String {
        self.0.to_json()
    }

    /// Repository the files came from
    #[getter]
    fn repository(&self) -> &str {
        &self.0.repository
    }

    /// Repository revision the files came from
    #[getter]
    fn revision(&self) -> &str {
        &self.0.revision
    }

    /// One entry per file, sorted by path
    #[getter]
    fn files(&self) -> Vec<PyModelManifestEntry> {
        self.0
            .files
            .iter()
            .cloned()
            .map(PyModelManifestEntry)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelManifest(repository='{}', revision='{}', files={})",
            self.0.repository,
            self.0.revision,
            self.0.files.len()
        )
    }
}

/// A model directory that holds every file one mode loads
#[pyclass(name = "PreparedModels", module = "speakrs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyPreparedModels(pub(crate) PreparedModels);

#[pymethods]
impl PyPreparedModels {
    /// Directory the pipeline loads from
    #[getter]
    fn model_dir(&self) -> PathBuf {
        self.0.model_dir.clone()
    }

    /// Mode the directory was prepared for
    #[getter]
    fn mode(&self) -> &'static str {
        mode_name(self.0.mode)
    }

    /// Model revision, or `None` for an unverified local directory
    #[getter]
    fn revision(&self) -> Option<&str> {
        self.0.revision.as_deref()
    }

    /// Manifest the files were verified against, if one was given
    #[getter]
    fn manifest(&self) -> Option<PyModelManifest> {
        self.0.manifest.clone().map(PyModelManifest)
    }

    fn __repr__(&self) -> String {
        format!(
            "PreparedModels(model_dir='{}', mode='{}', revision={})",
            self.0.model_dir.display(),
            mode_name(self.0.mode),
            py_opt_str(self.0.revision.as_deref())
        )
    }
}

/// One cached revision of the model repository
#[pyclass(name = "CacheEntry", module = "speakrs", frozen)]
#[derive(Clone)]
pub(crate) struct PyCacheEntry(pub(crate) CacheEntry);

#[pymethods]
impl PyCacheEntry {
    /// Hugging Face repository id
    #[getter]
    fn repository(&self) -> &str {
        &self.0.repository
    }

    /// Revision (commit hash)
    #[getter]
    fn revision(&self) -> &str {
        &self.0.revision
    }

    /// Snapshot directory
    #[getter]
    fn path(&self) -> PathBuf {
        self.0.path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheEntry(revision='{}', path='{}')",
            self.0.revision,
            self.0.path.display()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (mode, cache_dir=None, model_dir=None, manifest=None, progress=None))]
pub(crate) fn _prepare(
    py: Python<'_>,
    mode: &str,
    cache_dir: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    manifest: Option<PyModelManifest>,
    progress: Option<Py<PyAny>>,
) -> PyResult<PyPreparedModels> {
    let mut options = PrepareModelsOptions::new(mode_from(mode)?);
    options.model_dir = model_dir;
    options.manifest = manifest.map(|manifest| manifest.0);
    let store = cache_dir.map_or_else(ModelStore::default, ModelStore::new);
    // a raising callback or Ctrl+C cancels before the next file downloads
    let token = CancelToken::new();
    options.cancel_token = Some(token.clone());
    let sink = ProgressSink::new(progress, token);
    let result = py.detach(|| {
        let on_download =
            |file: &str| sink.send(ProgressEvent::DownloadingModel { file: file.into() });
        store
            .prepare(options, &on_download)
            .map_err(prepare_sdk_err)
    });
    sink.finish(result).map(PyPreparedModels)
}

/// Hash every file `mode` loads from `model_dir`
#[pyfunction]
#[pyo3(signature = (model_dir, mode="CPU"))]
pub(crate) fn generate_manifest(
    py: Python<'_>,
    model_dir: PathBuf,
    mode: &str,
) -> PyResult<PyModelManifest> {
    let mode = mode_from(mode)?;
    py.detach(|| ModelManifest::generate(&model_dir, mode))
        .map(PyModelManifest)
        .map_err(manifest_err)
}

/// Files `mode` loads, relative to the model directory
#[pyfunction]
#[pyo3(signature = (mode="CPU"))]
pub(crate) fn required_model_files(mode: &str) -> PyResult<Vec<String>> {
    Ok(speakrs_sdk::required_model_files(mode_from(mode)?))
}

/// The model cache: `$SPEAKRS_CACHE_DIR`, else `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`
#[pyfunction]
pub(crate) fn default_cache_dir() -> PathBuf {
    ModelStore::default_cache_dir()
}

/// Cached revisions of the model repository
#[pyfunction]
#[pyo3(signature = (cache_dir=None))]
pub(crate) fn list_cache(cache_dir: Option<PathBuf>) -> PyResult<Vec<PyCacheEntry>> {
    cache_dir
        .map_or_else(ModelStore::default, ModelStore::new)
        .list_cache()
        .map(|entries| entries.into_iter().map(PyCacheEntry).collect())
        .map_err(prepare_err)
}

/// Delete one cached revision and the files no other revision shares
///
/// Returns `False` when the revision was not cached.
#[pyfunction]
#[pyo3(signature = (revision, cache_dir=None))]
pub(crate) fn cleanup_revision(revision: &str, cache_dir: Option<PathBuf>) -> PyResult<bool> {
    cache_dir
        .map_or_else(ModelStore::default, ModelStore::new)
        .cleanup_revision(revision)
        .map_err(prepare_err)
}
