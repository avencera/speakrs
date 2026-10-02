use std::path::PathBuf;

#[cfg(feature = "coreml")]
use super::CoreMlError;
use super::ExecutionMode;
use super::TensorShapeError;
#[cfg(feature = "_ort")]
use super::ort_runtime::OrtRuntimeError;

/// Errors that can occur while running segmentation or embedding inference
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InferenceError {
    /// ONNX Runtime returned an error
    #[cfg(feature = "_ort")]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(
            feature = "cpu",
            feature = "cuda",
            feature = "migraphx",
            feature = "load-dynamic"
        )))
    )]
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Native CoreML returned an error
    #[cfg(feature = "coreml")]
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    #[error(transparent)]
    CoreMl(#[from] CoreMlError),
    /// A model input or output did not match its tensor shape contract
    #[error(transparent)]
    Shape(#[from] TensorShapeError),
    /// An output array could not be built from the model's flat output
    #[error("{context}: invalid output shape: {source}")]
    OutputArray {
        /// Which output decode step failed
        context: &'static str,
        /// The ndarray shape error
        #[source]
        source: ndarray::ShapeError,
    },
    /// A model returned no output tensor
    #[error("{context}: missing output tensor")]
    MissingOutput {
        /// Which output decode step failed
        context: &'static str,
    },
    /// A batch has more useful rows than the loaded model accepts
    #[error("{context}: useful rows {rows} exceed model capacity {capacity}")]
    BatchTooLarge {
        /// Which batch step failed
        context: &'static str,
        /// Useful rows requested by the caller
        rows: usize,
        /// Maximum rows the model accepts
        capacity: usize,
    },
    /// The multi-mask mask count does not match its filterbank rows
    #[error(
        "multi-mask batch: expected {expected} mask rows for {fbanks} fbank rows, got {actual}"
    )]
    MaskCountMismatch {
        /// Filterbank rows in the batch
        fbanks: usize,
        /// Mask rows required for those filterbank rows
        expected: usize,
        /// Mask rows provided by the caller
        actual: usize,
    },
    /// The loaded backend has no model for the requested operation
    #[error("{model} is not available for this model inventory and execution mode")]
    ModelUnavailable {
        /// The model that the operation needs
        model: &'static str,
    },
    /// A scratch buffer was not contiguous in memory
    #[error("{context}: buffer was not contiguous")]
    NonContiguousBuffer {
        /// Which buffer was not contiguous
        context: &'static str,
    },
    /// A lock around a shared session was poisoned by a panic on another thread
    #[error("{resource} lock was poisoned")]
    LockPoisoned {
        /// The resource guarded by the lock
        resource: &'static str,
    },
    /// An inference worker thread panicked
    #[error("{worker} worker panicked")]
    WorkerPanic {
        /// The worker that panicked
        worker: &'static str,
    },
}

/// Errors that can occur while loading a model or initializing its inference runtime
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ModelLoadError {
    /// Hugging Face Hub could not resolve a required model asset
    #[cfg(feature = "online")]
    #[error(transparent)]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    /// Requested execution mode is not supported by this build
    #[error(transparent)]
    UnsupportedExecutionMode(#[from] ExecutionModeError),
    /// ONNX Runtime could not be prepared for this process
    #[cfg(feature = "_ort")]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(
            feature = "cpu",
            feature = "cuda",
            feature = "migraphx",
            feature = "load-dynamic"
        )))
    )]
    #[error(transparent)]
    Runtime(#[from] OrtRuntimeError),
    /// ONNX Runtime returned an error after initialization completed
    #[cfg(feature = "_ort")]
    #[cfg_attr(
        docsrs,
        doc(cfg(any(
            feature = "cpu",
            feature = "cuda",
            feature = "migraphx",
            feature = "load-dynamic"
        )))
    )]
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// A required native model asset is missing for the selected execution mode
    #[error("{mode} requires native asset `{path}`")]
    MissingNativeAsset {
        /// The execution mode that requires the asset
        mode: ExecutionMode,
        /// The missing compiled CoreML bundle path
        path: PathBuf,
    },
    /// A required native model asset exists but failed to load
    #[error("{mode} failed to load native asset `{path}`: {message}")]
    NativeAssetLoad {
        /// The execution mode that requires the asset
        mode: ExecutionMode,
        /// The compiled CoreML bundle path that failed to load
        path: PathBuf,
        /// The backend load error
        message: String,
    },
    /// A typed model or execution configuration is invalid
    #[error("invalid model configuration: {message}")]
    InvalidConfiguration {
        /// Boundary validation error
        message: String,
    },
}

/// Errors from requesting an execution mode that is not supported in the current build
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
#[error("{mode} requires the `{feature}` Cargo feature")]
pub struct ExecutionModeError {
    pub(super) mode: ExecutionMode,
    pub(super) feature: &'static str,
}
