//! Binding-friendly facade over speakrs
//!
//! Language bindings call this crate instead of `speakrs` directly. It adds what a
//! binding needs on top of the core pipeline: decoding audio files to 16 kHz mono,
//! downloading and verifying models, batch decoding that overlaps inference,
//! cancellation, progress events, and errors with stable category codes

#![warn(missing_docs)]

pub mod audio;
mod batch;
pub mod cancel;
pub mod error;
pub mod models;
pub mod pipeline;
pub mod queue;
pub mod runtime;

/// The core crate, re-exported so bindings use the same version
pub use speakrs;

pub use audio::{AudioDecodeError, DecodedAudio};
pub use cancel::CancelToken;
pub use error::{SdkError, SdkErrorCategory};
pub use models::{
    CacheEntry, DEFAULT_MODEL_REPOSITORY, DEFAULT_MODEL_REVISION, ModelManifest,
    ModelManifestEntry, ModelManifestError, ModelPrepareError, ModelStore, PrepareModelsOptions,
    PreparedModels, required_model_files,
};
pub use pipeline::{
    Diarization, ProgressCallback, ProgressEvent, RunOptions, SdkPipeline, TimingStats,
};
pub use queue::{QueueItem, SdkQueue};
pub use runtime::{RuntimeOptions, available_modes, mode_name, parse_mode};
