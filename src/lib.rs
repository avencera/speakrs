#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! `speakrs` implements the full pyannote `community-1` style diarization
//! pipeline in Rust: segmentation, powerset decode, overlap-add aggregation,
//! binarization, embedding, PLDA, and VBx clustering.
//!
//! There is no Python runtime in the library path. Inference runs on ONNX
//! Runtime or native CoreML, and the rest of the pipeline stays in Rust.
//!
//! # Usage
//!
//! No inference backend is enabled by default. Pick the one for your platform:
//!
//! ```toml
//! # macOS (CoreML)
//! speakrs = { version = "0.6", features = ["coreml"] }
//!
//! # NVIDIA GPU
//! speakrs = { version = "0.6", features = ["cuda"] }
//!
//! # CPU
//! speakrs = { version = "0.6", features = ["cpu"] }
//!
//! # AMD GPU
//! speakrs = { version = "0.6", features = ["migraphx"] }
//! ```
//!
//! The `coreml` feature runs on native CoreML only, so a macOS build with just `coreml`
//! does not compile, link, or download ONNX Runtime.
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
//!
//! # let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
//! # let audio: Vec<f32> = vec![];
//! let result = pipeline.run(&audio)?;
//!
//! for segment in result.discrete_diarization.to_segments() {
//!     println!("{:.3} - {:.3}  {}", segment.start, segment.end, segment.speaker);
//! }
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! ## Background queue
//!
//! [`QueueSender`] and [`QueueReceiver`] run a background worker. Use
//! [`QueueSender::try_push`] to submit audio without blocking. A full queue
//! returns the request in [`QueueError::Full`] so the sender can retry it:
//!
//! ```no_run
//! use std::time::Duration;
//!
//! use speakrs::{ExecutionMode, OwnedDiarizationPipeline, QueueError, QueuedDiarizationRequest};
//!
//! # fn receive_files() -> Vec<(String, Vec<f32>)> { vec![] }
//! let pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
//! let (tx, rx) = pipeline.into_queued()?;
//!
//! let sender = std::thread::spawn(move || -> Result<(), QueueError> {
//!     for (file_id, audio) in receive_files() {
//!         let mut request = QueuedDiarizationRequest::new(file_id, audio);
//!         loop {
//!             match tx.try_push(request) {
//!                 Err(QueueError::Full(rejected)) => {
//!                     request = rejected;
//!                     std::thread::sleep(Duration::from_millis(10));
//!                 }
//!                 Ok(_) => break,
//!                 Err(error) => return Err(error),
//!             }
//!         }
//!     }
//!     Ok(())
//! });
//!
//! for result in rx {
//!     let result = result?;
//!     print!("{}", result.result?.rttm(&result.file_id));
//! }
//! sender.join().expect("sender thread panicked")?;
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! ## Local models
//!
//! For offline or airgapped setups, load models from a local directory:
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
//! | Mode | Backend | Step | Use it for |
//! |------|---------|------|------------|
//! | `cpu` | ONNX Runtime CPU | 1s | CPU runs and widest compatibility |
//! | `coreml` | Native CoreML | 1s | macOS with CoreML acceleration |
//! | `coreml-fast` | Native CoreML | 2s | macOS with CoreML acceleration and higher throughput |
//! | `cuda` | ONNX Runtime CUDA | 1s | NVIDIA GPU |
//! | `cuda-fast` | ONNX Runtime CUDA | 2s | NVIDIA GPU for higher throughput |
//! | `migraphx` | ONNX Runtime MIGraphX | 1s | AMD GPU |
//!
//! Each mode needs its Cargo feature: `cpu`, `coreml` for both CoreML modes, `cuda` for both
//! CUDA modes, or `migraphx`. Requesting a mode whose feature is off returns
//! `ModelLoadError::UnsupportedExecutionMode`.
//!
//! The `*-fast` modes move the segmentation window every 2 seconds instead of
//! every 1 second. That gives the pipeline fewer windows to score, so it can be much faster, but speaker changes
//! may land a little farther from the exact word or pause where they happened.
//!
//! Use the 1 second modes when you care about exactly when each speaker starts and stops,
//! short clips, interviews with quick back-and-forth, or audio you plan to subtitle or edit. The 2 second modes
//! are usually worth trying for long recordings where speed matters more than exact speaker-change times, such as
//! meetings, lectures, podcasts, or bulk archives.
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
//! On VoxConverse test, `coreml` matches pyannote at 11.1% DER and runs at
//! 631x realtime versus pyannote's 23x. `cuda` matches pyannote at 11.1% DER
//! and runs at 50x realtime versus pyannote's 18x. See
//! [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks) for
//! the full tables across all datasets.
//!
//! CoreML and ONNX Runtime can differ slightly even in FP32 because the runtime
//! graphs are not identical and floating-point reduction order changes rounding.
//!
//! # Why not pyannote-rs?
//!
//! Despite its name, [pyannote-rs](https://github.com/thewh1teagle/pyannote-rs)
//! is not a Rust port of the full pyannote diarization pipeline. It provides
//! useful building blocks for a much simpler, lightweight diarizer.
//!
//! In the end-to-end pattern shown by its examples, `pyannote-rs`:
//!
//! 1. runs the pyannote segmentation model on non-overlapping 10-second windows;
//! 2. reduces every frame to speech or non-speech, emitting one segment for each
//!    uninterrupted region of speech;
//! 3. computes one speaker embedding for that entire segment; and
//! 4. assigns the segment online by comparing its embedding with previously seen
//!    speakers using a fixed cosine-similarity threshold.
//!
//! That is closer to voice activity detection followed by online speaker
//! matching than to pyannote `community-1`.
//!
//! | | `speakrs` | `pyannote-rs` |
//! |-|-----------|---------------|
//! | Segmentation output | Preserves separate local speaker tracks and overlapping speech | Collapses all non-silence classes into one speech track |
//! | Windows | Scores every 1s or 2s and reconciles overlapping 10s windows | Scores independent, non-overlapping 10s windows |
//! | Embeddings | One masked embedding per local speaker and window | One embedding per contiguous speech segment |
//! | Clustering | Recording-wide AHC initialization, PLDA, and VBx refinement | Immediate cosine-threshold match against stored embeddings |
//! | Final timeline | Reconstructs speaker count and identities from all windows, then filters short activity | Emits a segment only after a speech-to-silence transition |
//!
//! These differences explain the accuracy gap. A speech region can contain
//! several turns with no silence between them. `pyannote-rs` treats that region
//! as one segment, mixes all voices into one embedding, and gives it one speaker
//! label. It also discards the segmentation model's separate local speaker
//! tracks, so it cannot represent overlapping speakers. Its online assignments
//! are not reconsidered using evidence from the rest of the recording.
//!
//! `speakrs` keeps the per-speaker frame activity, extracts speaker-conditioned
//! embeddings, combines evidence from overlapping windows, and clusters all
//! usable embeddings together. PLDA makes the embeddings more discriminative
//! for speaker identity, while VBx uses the recording's temporal and global
//! evidence to refine speaker assignments. The final reconstruction can
//! therefore preserve rapid speaker changes and overlapping speech instead of
//! reducing a whole speech region to one voice.
//!
//! The benchmark reflects that architectural difference. With `pyannote-rs`
//! v0.3 on VoxConverse dev, it returned no RTTM segments for 183 of 216 files.
//! On the remaining 33 files where it produced at least five segments
//! (186 minutes, collar=0ms), the result was:
//!
//! | | DER | Missed speech | False alarm | Speaker confusion |
//! |-|-----|---------------|-------------|-------------------|
//! | `speakrs` CoreML | **11.5%** | 3.8% | 3.6% | 4.1% |
//! | `pyannote-rs` | 80.2% | 34.9% | 7.4% | 37.9% |
//!
//! Lower DER is better. The 80.2% figure is the subset-only comparison; it does
//! not count the 183 files for which `pyannote-rs` produced no output.
//!
//! # Models
//!
//! With the default `online` feature, models download on first use from
//! [avencera/speakrs-models](https://huggingface.co/avencera/speakrs-models).
//! Set `SPEAKRS_MODELS_DIR` if you want to force a local bundle instead.
//!
//! # Features and build notes
//!
//! Enable at least one inference backend; the build fails with a clear error otherwise:
//!
//! - `coreml`: native CoreML backend on macOS, without ONNX Runtime
//! - `cpu`: CPU backend via ONNX Runtime
//! - `cuda`: NVIDIA CUDA backend via ONNX Runtime
//! - `migraphx`: AMD GPU backend via ONNX Runtime MIGraphX
//!
//! Other features:
//!
//! - `online` (default): model download via [`ModelManager`]
//! - `load-dynamic`: load the ONNX Runtime library at startup instead of static linking; use it
//!   with `cpu`, `cuda`, or `migraphx`
//!
//! The ONNX Runtime dependency behind `cpu`, `cuda`, and `migraphx` (`ort` 2.0.0-rc.13) is still
//! pre-release.
//!
//! # Public API
//!
//! Start here:
//!
//! - [`OwnedDiarizationPipeline`]: pipeline entry point
//! - [`QueueSender`] and [`QueueReceiver`]: background worker interface
//! - [`QueueConfig`]: in-process queue capacity
//! - [`DiarizationResult`]: frame-level activations, segments, clusters, embeddings, RTTM
//! - [`PipelineConfig`] and [`RuntimeConfig`]: tuning knobs
//! - [`ModelManager`]: model download when `online` is enabled
//! - [`Segment`]: a single speaker turn

#[cfg(all(feature = "coreml", not(target_os = "macos")))]
compile_error!("the `coreml` feature is only supported on macOS");

#[cfg(not(feature = "_backend"))]
compile_error!(
    "speakrs needs an inference backend; enable at least one of these Cargo features:\n\
     - macOS (Apple Silicon): `coreml`\n\
     - NVIDIA GPU: `cuda`\n\
     - AMD GPU: `migraphx`\n\
     - CPU (ONNX Runtime): `cpu`\n\
     for example: speakrs = { version = \"0.6\", features = [\"coreml\"] }"
);

// a build without a backend reports only the `compile_error!` above: every crate item is gated
// on `_backend`, which each backend feature enables, so no follow-on type errors appear
#[cfg(feature = "_backend")]
pub(crate) mod binarize;
#[cfg(feature = "_backend")]
pub(crate) mod clustering;
/// Segmentation and embedding model wrappers
#[cfg(feature = "_backend")]
pub mod inference;
#[cfg(feature = "_backend")]
pub(crate) mod linalg;
/// Diarization error rate (DER) evaluation utilities
#[cfg(feature = "_metrics")]
#[cfg(feature = "_backend")]
pub mod metrics;
/// Model paths and HuggingFace download support
#[cfg(feature = "_backend")]
pub mod models;
/// High-level diarization pipeline and result types
#[cfg(feature = "_backend")]
pub mod pipeline;
#[cfg(feature = "_backend")]
pub(crate) mod powerset;
#[cfg(feature = "_backend")]
pub(crate) mod reconstruct;
/// Speaker segments, merging, and RTTM output
#[cfg(feature = "_backend")]
pub mod segment;
#[cfg(feature = "_backend")]
pub(crate) mod utils;

// crate-root re-exports for the main import path
#[cfg(feature = "_backend")]
pub use inference::{CoreMlComputeUnits, ExecutionMode};
#[cfg(feature = "_backend")]
pub use models::ModelBundle;
#[cfg(feature = "online")]
#[cfg_attr(docsrs, doc(cfg(feature = "online")))]
#[cfg(feature = "_backend")]
pub use models::ModelManager;
#[cfg(feature = "_backend")]
pub use pipeline::{
    ActivityCleanup, AhcConfig, AhcConfigError, BatchInput, ClusteringBackend, ClusteringConfig,
    ClusteringConfigError, DiarizationPipeline, DiarizationResult, FbankSessionPool,
    FbankSessionPoolSize, FbankSessionPoolSizeError, OrtThreadCount, OrtThreadCountError,
    OwnedDiarizationPipeline, PipelineBuilder, PipelineConfig, PipelineError, QueueConfig,
    QueueError, QueueReceiver, QueueReceiverIter, QueueSender, QueuedDiarizationJobId,
    QueuedDiarizationRequest, QueuedDiarizationResult, ReconstructError,
    ResponsibilityInitialization, RuntimeConfig, VbxConfig, VbxConfigError,
};
#[cfg(feature = "_backend")]
pub use segment::Segment;

#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[cfg(feature = "_backend")]
pub use powerset::{PowersetDecodeError, PowersetMapping};
