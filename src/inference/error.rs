use std::path::PathBuf;

#[cfg(feature = "coreml")]
use super::CoreMlError;
#[cfg(feature = "cpu")]
use super::CpuError;
#[cfg(feature = "_cuda")]
use super::CudaError;
use super::ExecutionMode;
use super::TensorShapeError;
#[cfg(feature = "_ort")]
use super::ort_runtime::OrtRuntimeError;

/// Errors that can occur while running segmentation or embedding inference
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InferenceError {
    /// The native CPU backend returned an error
    #[cfg(feature = "cpu")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cpu")))]
    #[error(transparent)]
    Cpu(#[from] CpuError),
    /// ONNX Runtime returned an error
    #[cfg(feature = "_ort")]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "migraphx", feature = "load-dynamic"))))]
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Native CoreML returned an error
    #[cfg(feature = "coreml")]
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    #[error(transparent)]
    CoreMl(#[from] CoreMlError),
    /// The native CUDA backend returned an error
    #[cfg(feature = "_cuda")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_cuda")))]
    #[error(transparent)]
    Cuda(#[from] CudaError),
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
    /// The native CPU backend failed while loading a model
    #[cfg(feature = "cpu")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cpu")))]
    #[error(transparent)]
    Cpu(#[from] CpuError),
    /// Hugging Face Hub could not resolve a required model asset
    #[cfg(feature = "online")]
    #[error(transparent)]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    /// Requested execution mode is not supported by this build
    #[error(transparent)]
    UnsupportedExecutionMode(#[from] ExecutionModeError),
    /// ONNX Runtime could not be prepared for this process
    #[cfg(feature = "_ort")]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "migraphx", feature = "load-dynamic"))))]
    #[error(transparent)]
    Runtime(#[from] OrtRuntimeError),
    /// ONNX Runtime returned an error after initialization completed
    #[cfg(feature = "_ort")]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "migraphx", feature = "load-dynamic"))))]
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// The native CUDA backend failed while loading a model
    #[cfg(feature = "_cuda")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_cuda")))]
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// A required native model asset is missing for the selected execution mode
    #[error("{mode} requires native asset `{path}`")]
    MissingNativeAsset {
        /// The execution mode that requires the asset
        mode: ExecutionMode,
        /// The missing native weights or compiled model bundle path
        path: PathBuf,
    },
    /// The safetensors weights that the CUDA modes load are missing
    #[cfg(feature = "_cuda")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_cuda")))]
    #[error(
        "{mode} requires the CUDA weights `{path}`; export them with `scripts/cuda/export_weights.py --runtime-assets <dir>`"
    )]
    MissingCudaWeights {
        /// The execution mode that requires the weights
        mode: ExecutionMode,
        /// The missing weights file
        path: PathBuf,
    },
    /// The CUDA model assets could not be downloaded from Hugging Face
    #[cfg(all(feature = "_cuda", feature = "online"))]
    #[cfg_attr(docsrs, doc(cfg(all(feature = "_cuda", feature = "online"))))]
    #[error(
        "could not download the CUDA model assets for {mode} from Hugging Face: {source}; export them with `scripts/cuda/export_weights.py --runtime-assets <dir>` and load that directory with `from_dir`"
    )]
    CudaAssetsUnavailable {
        /// The execution mode whose assets were requested
        mode: ExecutionMode,
        /// The Hugging Face Hub error
        #[source]
        source: hf_hub::api::sync::ApiError,
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
