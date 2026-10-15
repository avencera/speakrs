//! The `speakrs._native` extension module
//!
//! The Python package in `python/speakrs` wraps the pipeline and queue classes and
//! re-exports everything else from here unchanged

use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use speakrs_sdk::speakrs::ExecutionMode;
use speakrs_sdk::{SdkError, SdkErrorCategory, mode_name, parse_mode};

pyo3::import_exception!(speakrs._errors, SpeakrsError);

mod config;
mod models;
mod pipeline;
mod results;

use config::{
    PyActivityCleanup, PyAhcConfig, PyClusteringConfig, PyPipelineConfig, PyRuntimeConfig,
    PyVbxConfig,
};
use models::{
    _prepare, PyCacheEntry, PyModelManifest, PyModelManifestEntry, PyPreparedModels,
    cleanup_revision, default_cache_dir, generate_manifest, list_cache, required_model_files,
};
use pipeline::{_build_pipeline, PyCancelToken, PyPipeline, PyQueue};
use results::{PyDiarizationResult, PyProgressEvent, PyQueueResult, PySegment, PyTimingStats};

pub(crate) fn sdk_err(error: SdkError) -> PyErr {
    SpeakrsError::new_err((error.category.code(), error.code, error.message))
}

pub(crate) fn category_err(category: SdkErrorCategory, message: impl ToString) -> PyErr {
    sdk_err(SdkError::new(category, message.to_string()))
}

pub(crate) fn config_err(error: impl ToString) -> PyErr {
    category_err(SdkErrorCategory::Configuration, error)
}

pub(crate) fn mode_from(name: &str) -> PyResult<ExecutionMode> {
    parse_mode(name).map_err(sdk_err)
}

// a panic while holding one of these locks leaves plain data behind, so it stays usable
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Execution modes compiled into this build, such as `["CPU", "CoreML", "CoreMLFast"]`
///
/// A CUDA mode can still fail at load time without an NVIDIA driver or a
/// supported GPU.
#[pyfunction]
fn available_modes() -> Vec<&'static str> {
    speakrs_sdk::available_modes()
        .into_iter()
        .map(mode_name)
        .collect()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "DEFAULT_MODEL_REPOSITORY",
        speakrs_sdk::DEFAULT_MODEL_REPOSITORY,
    )?;
    m.add(
        "DEFAULT_MODEL_REVISION",
        speakrs_sdk::DEFAULT_MODEL_REVISION,
    )?;
    m.add_class::<PyCancelToken>()?;
    m.add_class::<PyActivityCleanup>()?;
    m.add_class::<PyAhcConfig>()?;
    m.add_class::<PyVbxConfig>()?;
    m.add_class::<PyClusteringConfig>()?;
    m.add_class::<PyPipelineConfig>()?;
    m.add_class::<PyRuntimeConfig>()?;
    m.add_class::<PySegment>()?;
    m.add_class::<PyTimingStats>()?;
    m.add_class::<PyDiarizationResult>()?;
    m.add_class::<PyQueueResult>()?;
    m.add_class::<PyProgressEvent>()?;
    m.add_class::<PyModelManifestEntry>()?;
    m.add_class::<PyModelManifest>()?;
    m.add_class::<PyPreparedModels>()?;
    m.add_class::<PyCacheEntry>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyQueue>()?;
    m.add_function(wrap_pyfunction!(_build_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(generate_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(required_model_files, m)?)?;
    m.add_function(wrap_pyfunction!(default_cache_dir, m)?)?;
    m.add_function(wrap_pyfunction!(list_cache, m)?)?;
    m.add_function(wrap_pyfunction!(cleanup_revision, m)?)?;
    m.add_function(wrap_pyfunction!(available_modes, m)?)?;
    Ok(())
}

pub(crate) fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

pub(crate) fn py_opt_str(value: Option<&str>) -> String {
    value.map_or_else(|| "None".to_string(), |value| format!("'{value}'"))
}
