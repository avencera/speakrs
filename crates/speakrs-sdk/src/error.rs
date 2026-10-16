//! Errors with stable category and code strings that bindings expose as-is

use thiserror::Error;

/// Stable error category exposed to binding users
///
/// The string codes from [`Self::code`] are part of the binding API, so existing
/// codes must not change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdkErrorCategory {
    /// An invalid argument or configuration value
    Configuration,
    /// The execution mode is not compiled into this build
    UnsupportedRuntime,
    /// Model download or cache handling failed
    ModelPrepare,
    /// A model file is missing or does not match its manifest
    ModelManifest,
    /// The models could not be loaded for the selected mode
    ModelLoad,
    /// An audio file could not be opened or decoded
    AudioDecode,
    /// Decoded audio could not be resampled to 16 kHz
    AudioResample,
    /// Inference, clustering, or reconstruction failed
    Pipeline,
    /// The background queue rejected or lost a request
    Queue,
    /// The caller cancelled the operation
    Cancelled,
    /// A bug or broken invariant in the bindings
    Internal,
}

impl SdkErrorCategory {
    /// Stable string code for this category
    pub const fn code(self) -> &'static str {
        match self {
            Self::Configuration => "configuration",
            Self::UnsupportedRuntime => "unsupported_runtime",
            Self::ModelPrepare => "model_prepare",
            Self::ModelManifest => "model_manifest",
            Self::ModelLoad => "model_load",
            Self::AudioDecode => "audio_decode",
            Self::AudioResample => "audio_resample",
            Self::Pipeline => "pipeline",
            Self::Queue => "queue",
            Self::Cancelled => "cancelled",
            Self::Internal => "internal",
        }
    }
}

/// An error with a stable category, a finer stable code, and a readable message
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{message}")]
pub struct SdkError {
    /// Broad category that callers can match on
    pub category: SdkErrorCategory,
    /// Finer stable code, equal to the category code unless a variant needs its own
    pub code: &'static str,
    /// Human-readable description
    pub message: String,
}

impl SdkError {
    /// Create an error whose code is the category code
    pub fn new(category: SdkErrorCategory, message: impl Into<String>) -> Self {
        Self::with_code(category, category.code(), message)
    }

    /// Create an error with a finer code than its category
    pub fn with_code(
        category: SdkErrorCategory,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            category,
            code,
            message: message.into(),
        }
    }

    /// The error returned when a [`crate::CancelToken`] stops an operation
    pub fn cancelled() -> Self {
        Self::new(SdkErrorCategory::Cancelled, "operation was cancelled")
    }

    /// An invalid argument or configuration value
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::new(SdkErrorCategory::Configuration, message)
    }

    pub(crate) fn from_pipeline_error(error: speakrs::pipeline::PipelineError) -> Self {
        use speakrs::pipeline::PipelineError;

        let category = match &error {
            PipelineError::UnsupportedExecutionMode(_) => SdkErrorCategory::UnsupportedRuntime,
            PipelineError::ModelLoad(_) => SdkErrorCategory::ModelLoad,
            _ => SdkErrorCategory::Pipeline,
        };
        Self::new(category, error.to_string())
    }

    /// Map an error from building a pipeline, where any failure other than an
    /// unsupported mode means the models could not be loaded, including PLDA files
    pub(crate) fn from_load_error(error: speakrs::pipeline::PipelineError) -> Self {
        match error {
            speakrs::pipeline::PipelineError::UnsupportedExecutionMode(_) => {
                Self::from_pipeline_error(error)
            }
            other => Self::new(SdkErrorCategory::ModelLoad, other.to_string()),
        }
    }

    pub(crate) fn from_audio_error(error: crate::audio::AudioDecodeError) -> Self {
        let category = match &error {
            crate::audio::AudioDecodeError::Resample(_) => SdkErrorCategory::AudioResample,
            _ => SdkErrorCategory::AudioDecode,
        };
        Self::new(category, error.to_string())
    }
}
