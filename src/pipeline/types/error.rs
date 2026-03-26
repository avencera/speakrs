use crate::inference::{ExecutionModeError, ModelLoadError};

/// Errors that can occur during the diarization pipeline
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// Model construction or ONNX Runtime initialization error
    #[error(transparent)]
    ModelLoad(#[from] ModelLoadError),
    /// ONNX Runtime error
    #[error(transparent)]
    Ort(#[from] ort::Error),
    /// Requested execution mode is not supported by this build
    #[error(transparent)]
    UnsupportedExecutionMode(#[from] ExecutionModeError),
    /// Segmentation inference error
    #[error(transparent)]
    Segmentation(#[from] crate::inference::segmentation::SegmentationError),
    /// PLDA scoring/training error
    #[error(transparent)]
    Plda(#[from] crate::clustering::plda::PldaError),
    /// Hugging Face Hub download error
    #[cfg(feature = "online")]
    #[error(transparent)]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    /// Queue setup or execution error
    #[error(transparent)]
    Queue(#[from] super::super::queued::QueueError),
    /// Internal pipeline invariant was violated
    #[error("{0}")]
    Invariant(String),
    /// Background worker panicked
    #[error("{worker} thread panicked")]
    WorkerPanic {
        /// Worker or thread name
        worker: String,
    },
    /// Backend-specific execution failed with additional context
    #[error("{context}: {message}")]
    Backend {
        /// Which backend step failed
        context: &'static str,
        /// Backend error message
        message: String,
    },
    /// Catch-all for other pipeline errors
    #[error("{0}")]
    Other(String),
}
