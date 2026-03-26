#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Speaker diarization in Rust
//!
//! `speakrs` implements the full pyannote `community-1` speaker diarization pipeline:
//! segmentation, powerset decode, aggregation, binarization, embedding, PLDA, and VBx
//! clustering. There is no Python dependency. Inference runs on ONNX Runtime or native
//! CoreML, and all post-processing stays in Rust.
//!
//! # Usage
//!
//! ```toml
//! # Apple Silicon (CoreML)
//! speakrs = { version = "0.2", features = ["coreml"] }
//!
//! # NVIDIA GPU
//! speakrs = { version = "0.2", features = ["cuda"] }
//!
//! # CPU only (default)
//! speakrs = "0.2"
//! ```
//!
//! # Quick start
//!
//! ```no_run
//! use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
//!
//! let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::Cpu)?;
//! let audio: Vec<f32> = vec![]; // 16 kHz mono f32 samples
//! let result = pipeline.run(&audio)?;
//! print!("{}", result.rttm("my-file"));
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! # Execution modes
//!
//! | Mode | Backend | Step | Platform |
//! |------|---------|------|----------|
//! | [`ExecutionMode::Cpu`] | ORT CPU | 1s | Any |
//! | [`ExecutionMode::CoreMl`] | Native CoreML | ~1s | macOS / iOS |
//! | [`ExecutionMode::CoreMlFast`] | Native CoreML | 2s | macOS / iOS |
//! | [`ExecutionMode::Cuda`] | ORT CUDA | ~1s | Linux / Windows |
//! | [`ExecutionMode::CudaFast`] | ORT CUDA | 2s | Linux / Windows |
//!
//! # Features
//!
//! - **`online`** (default) — automatic model download from HuggingFace via [`ModelManager`]
//! - **`coreml`** — native CoreML backend for Apple Silicon GPU/ANE acceleration
//! - **`cuda`** — NVIDIA CUDA backend via ONNX Runtime
//! - **`load-dynamic`** — load the CUDA runtime library at startup instead of static linking
//!
//! # Build requirements
//!
//! This crate links OpenBLAS statically (via `ndarray-linalg`), which requires a C compiler.
//! On most systems this is already available. The ONNX Runtime dependency (`ort` 2.0.0-rc.12)
//! is pre-release.

pub(crate) mod binarize;
pub(crate) mod clustering;
/// Segmentation and embedding model wrappers
pub mod inference;
/// Diarization error rate (DER) evaluation utilities
#[cfg(feature = "_metrics")]
pub mod metrics;
/// Model paths and automatic download from HuggingFace
pub mod models;
/// High-level diarization pipeline and result types
pub mod pipeline;
pub(crate) mod powerset;
pub(crate) mod reconstruct;
/// Speaker segments, merging, and RTTM output
pub mod segment;
pub(crate) mod utils;

// crate-root re-exports for the common path
pub use inference::ExecutionMode;
pub use models::ModelBundle;
#[cfg(feature = "online")]
pub use models::ModelManager;
pub use pipeline::{
    BatchInput, DiarizationPipeline, DiarizationResult, OwnedDiarizationPipeline, PipelineBuilder,
    PipelineConfig, PipelineError, QueueError, QueueReceiver, QueueReceiverIter, QueueSender,
    QueuedDiarizationJobId, QueuedDiarizationRequest, QueuedDiarizationResult, RuntimeConfig,
};
pub use reconstruct::make_exclusive;
pub use segment::Segment;

#[cfg(feature = "_metrics")]
pub use powerset::PowersetMapping;
