#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Fast Rust speaker diarization.
//!
//! `speakrs` runs the full pyannote `community-1` style pipeline in Rust:
//! segmentation, powerset decode, overlap-add aggregation, binarization,
//! embedding, PLDA, and VBx clustering. There is no Python dependency.
//! Inference runs on ONNX Runtime or native CoreML, and the rest stays in Rust.
//!
//! This crate is for people who want pyannote-level diarization without
//! shipping a Python stack. On VoxConverse dev, `speakrs` CoreML gets 7.1% DER
//! at 529x realtime versus pyannote's 7.2% at 24x. Full results live in
//! [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks).
//!
//! # Usage
//!
//! ```toml
//! # Apple Silicon (CoreML)
//! speakrs = { version = "0.4", features = ["coreml"] }
//!
//! # NVIDIA GPU
//! speakrs = { version = "0.4", features = ["cuda"] }
//!
//! # CPU only (default)
//! speakrs = "0.4"
//!
//! # System OpenBLAS
//! speakrs = { version = "0.4", default-features = false, features = ["online", "openblas-system"] }
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//!     let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
//!
//!     let audio: Vec<f32> = load_your_mono_16khz_audio_here();
//!     let result = pipeline.run(&audio)?;
//!
//!     print!("{}", result.rttm("my-audio"));
//!     Ok(())
//! }
//! # fn load_your_mono_16khz_audio_here() -> Vec<f32> { unimplemented!() }
//! ```
//!
//! ## Speaker turns
//!
//! ```no_run
//! # use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
//! use speakrs::pipeline::{FRAME_DURATION_SECONDS, FRAME_STEP_SECONDS};
//!
//! # let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
//! # let audio: Vec<f32> = vec![];
//! let result = pipeline.run(&audio)?;
//!
//! for segment in result
//!     .discrete_diarization
//!     .to_segments(FRAME_STEP_SECONDS, FRAME_DURATION_SECONDS)
//! {
//!     println!("{:.3} - {:.3}  {}", segment.start, segment.end, segment.speaker);
//! }
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! ## Background queue
//!
//! [`QueueSender`] and [`QueueReceiver`] run a background worker that can batch
//! work across files. Push audio from any thread and drain results as they
//! finish:
//!
//! ```no_run
//! use speakrs::{ExecutionMode, OwnedDiarizationPipeline, QueuedDiarizationRequest};
//!
//! # fn receive_files() -> Vec<(String, Vec<f32>)> { vec![] }
//! let pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
//! let (tx, rx) = pipeline.into_queued()?;
//!
//! std::thread::spawn(move || {
//!     for (file_id, audio) in receive_files() {
//!         tx.push(QueuedDiarizationRequest::new(file_id, audio)).unwrap();
//!     }
//! });
//!
//! for result in rx {
//!     let result = result?;
//!     print!("{}", result.result?.rttm(&result.file_id));
//! }
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! ## Local models
//!
//! For offline or airgapped use, load models from a local directory:
//!
//! ```no_run
//! use std::path::Path;
//! use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
//!
//! # let audio: Vec<f32> = vec![];
//! let mut pipeline = OwnedDiarizationPipeline::from_dir(
//!     Path::new("/path/to/models"),
//!     ExecutionMode::Cpu,
//! )?;
//! let result = pipeline.run(&audio)?;
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! # Choosing a mode
//!
//! | Mode | Backend | Step | When to use it |
//! |------|---------|------|----------------|
//! | `cpu` | ONNX Runtime CPU | 1s | Reference path, widest compatibility |
//! | `coreml` | Native CoreML | 1s | Apple Silicon, best accuracy |
//! | `coreml-fast` | Native CoreML | 2s | Apple Silicon, throughput-first |
//! | `cuda` | ONNX Runtime CUDA | 1s | NVIDIA GPU, best accuracy |
//! | `cuda-fast` | ONNX Runtime CUDA | 2s | NVIDIA GPU, throughput-first |
//!
//! The `*-fast` modes use a 2 second step instead of 1 second. The tradeoff is
//! simple: more throughput, slightly less precision at speaker boundaries.
//! On orderly turn-taking audio the gap is usually small, and on some datasets
//! the fast modes win anyway. If you want the safest default, start with
//! `coreml` or `cuda`.
//!
//! # Benchmarks
//!
//! VoxConverse dev, collar=0ms:
//!
//! | Platform | Implementation | DER | Time | RTFx |
//! |----------|----------------|-----|------|------|
//! | Apple M4 Pro | `speakrs` `coreml` | **7.1%** | 138s | 529x |
//! | Apple M4 Pro | `speakrs` `coreml-fast` | 7.4% | 169s | 434x |
//! | Apple M4 Pro | pyannote community-1 (MPS) | 7.2% | 2999s | 24x |
//! | RTX 4090 | `speakrs` `cuda` | **7.0%** | 1236s | 59x |
//! | RTX 4090 | `speakrs` `cuda-fast` | 7.4% | 604s | **121x** |
//! | RTX 4090 | pyannote community-1 (CUDA) | 7.2% | 2312s | 32x |
//!
//! On VoxConverse test, both `coreml` and `cuda` match pyannote at 11.1% DER
//! while staying much faster. See
//! [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks) for
//! the full tables across all datasets.
//!
//! CoreML and ONNX Runtime can differ slightly even in FP32 because the runtime
//! graphs are not identical and floating-point reduction order changes rounding.
//!
//! # Why not pyannote-rs?
//!
//! [pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) is the closest
//! Rust-only comparison point, but it is solving a different problem.
//!
//! | | `speakrs` | `pyannote-rs` |
//! |-|-----------|---------------|
//! | Pipeline | Full pyannote `community-1` style pipeline | Simpler window-level pipeline |
//! | Aggregation | Overlap-add plus binarization | No overlap-add or binarization |
//! | Clustering | PLDA + VBx | Cosine threshold |
//! | Goal | Match pyannote behavior on CPU/CUDA | Lightweight Rust diarization |
//!
//! On the VoxConverse dev subset where `pyannote-rs` emits output, `speakrs`
//! CoreML scores 11.5% DER versus 80.2% for `pyannote-rs`. In that same run,
//! `pyannote-rs` returned no segments on most files. If you want something close
//! to pyannote without Python, this is what `speakrs` is for.
//!
//! # Models
//!
//! With the default `online` feature, models download automatically on first use
//! from [avencera/speakrs-models](https://huggingface.co/avencera/speakrs-models).
//! Set `SPEAKRS_MODELS_DIR` if you want to force a local bundle instead.
//!
//! # Features and build notes
//!
//! Common features:
//!
//! - `online` (default): automatic model download via [`ModelManager`]
//! - `coreml`: native CoreML backend for Apple Silicon
//! - `cuda`: NVIDIA CUDA backend via ONNX Runtime
//! - `load-dynamic`: load the CUDA runtime at startup instead of static linking
//!
//! BLAS backends matter if you disable default features:
//!
//! - `x86_64` defaults to statically linked Intel MKL
//! - non-`x86_64` defaults to statically linked OpenBLAS and needs a C toolchain
//! - advanced opt-ins are `intel-mkl`, `openblas-static`, and `openblas-system`
//!
//! ```toml
//! speakrs = { version = "0.4", default-features = false, features = ["online", "intel-mkl"] }
//! speakrs = { version = "0.4", default-features = false, features = ["online", "openblas-system"] }
//! ```
//!
//! The ONNX Runtime dependency (`ort` 2.0.0-rc.12) is still pre-release.
//!
//! # Public API
//!
//! Start here:
//!
//! - [`OwnedDiarizationPipeline`]: the usual entry point
//! - [`QueueSender`] and [`QueueReceiver`]: background worker interface
//! - [`DiarizationResult`]: frame-level activations, segments, clusters, embeddings, RTTM
//! - [`PipelineConfig`] and [`RuntimeConfig`]: tuning knobs
//! - [`ModelManager`]: automatic model download when `online` is enabled
//! - [`Segment`]: a single speaker turn

pub(crate) mod binarize;
pub(crate) mod clustering;
/// Segmentation and embedding model wrappers
pub mod inference;
pub(crate) mod linalg;
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
pub use segment::Segment;

#[cfg(feature = "_metrics")]
pub use powerset::PowersetMapping;
