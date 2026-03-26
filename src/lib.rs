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
//! use speakrs::pipeline::{FRAME_STEP_SECONDS, FRAME_DURATION_SECONDS};
//!
//! # let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
//! # let audio: Vec<f32> = vec![];
//! let result = pipeline.run(&audio)?;
//!
//! for segment in result.discrete_diarization.to_segments(FRAME_STEP_SECONDS, FRAME_DURATION_SECONDS) {
//!     println!("{:.3} - {:.3}  {}", segment.start, segment.end, segment.speaker);
//! }
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! ## Background queue
//!
//! For processing many files, [`QueueSender`] and [`QueueReceiver`] run a background
//! worker that auto-batches requests for cross-file optimizations. Push files from
//! any thread, receive results as they complete:
//!
//! ```no_run
//! use speakrs::{ExecutionMode, OwnedDiarizationPipeline, QueuedDiarizationRequest};
//!
//! # fn receive_files() -> Vec<(String, Vec<f32>)> { vec![] }
//! let pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
//! let (tx, rx) = pipeline.into_queued()?;
//!
//! // producer: push files as they arrive from another thread
//! std::thread::spawn(move || {
//!     for (file_id, audio) in receive_files() {
//!         tx.push(QueuedDiarizationRequest::new(file_id, audio)).unwrap();
//!     }
//!     // tx drops when the producer is done, ending the rx loop
//! });
//!
//! // results arrive while files are still being pushed,
//! // the loop exits once tx is dropped and all queued work is done
//! for result in rx {
//!     let result = result?;
//!     print!("{}", result.result?.rttm(&result.file_id));
//! }
//! # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
//! ```
//!
//! ## Local models
//!
//! For offline or airgapped usage, load models from a local directory:
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
//! # Pipeline
//!
//! ```text
//! Audio (16kHz f32)
//!   │
//!   ├─ Segmentation ──────→ raw 7-class logits per 10s window
//!   │   (ONNX or CoreML)
//!   │
//!   ├─ Powerset Decode ───→ 3-speaker soft/hard activations
//!   │
//!   ├─ Overlap-Add ───────→ Hamming-windowed aggregation across windows
//!   │
//!   ├─ Binarize ──────────→ hysteresis thresholding + min-duration filtering
//!   │
//!   ├─ Embedding ─────────→ 256-dim WeSpeaker vectors per segment
//!   │   (ONNX or CoreML)
//!   │
//!   ├─ PLDA Transform ────→ 128-dim whitened features
//!   │
//!   ├─ VBx Clustering ────→ Bayesian HMM speaker assignments
//!   │
//!   ├─ Reconstruct ───────→ map clusters back to frame-level activations (temporal smoothing)
//!   │
//!   └─ Segments ──────────→ RTTM output
//! ```
//!
//! # macOS / iOS (CoreML)
//!
//! Requires the `coreml` Cargo feature. Uses Apple's CoreML framework for
//! GPU/ANE-accelerated inference.
//!
//! ## Execution modes
//!
//! | Mode | Backend | Step | Precision | Use case |
//! |------|---------|------|-----------|----------|
//! | `coreml` | Native CoreML | 1s | FP32 | Best accuracy (529x realtime) |
//! | `coreml-fast` | Native CoreML | 2s | FP32 | Best speed (912x realtime) |
//!
//! `coreml-fast` uses a wider step (2s instead of 1s) to get about 1.5x more speed.
//! That follows the same throughput-first tradeoff
//! [SpeakerKit](https://github.com/argmaxinc/WhisperKit) uses on Apple hardware. It
//! matches `coreml` on most clips, but on some inputs the coarser step loses temporal
//! resolution at speaker boundaries.
//!
//! ## Benchmarks
//!
//! All benchmarks on Apple M4 Pro, macOS 26.3, evaluated on VoxConverse dev
//! (216 files, 1217.8 min, collar=0ms):
//!
//! | Mode | DER | Time | RTFx |
//! |------|-----|------|------|
//! | `coreml` | **7.1%** | 138s | 529x |
//! | `coreml-fast` | 7.4% | 169s | 434x |
//! | pyannote community-1 (MPS) | 7.2% | 2999s | 24x |
//! | SpeakerKit | 7.8% | 234s | 312x |
//!
//! On VoxConverse test (232 files, 2612.2 min), `coreml` matches pyannote at 11.1%
//! DER while running at 631x realtime vs pyannote's 23x.
//!
//! CoreML results may differ slightly from ONNX CPU, the two runtimes apply different
//! graph optimizations (operator fusion, reduction order) that change floating-point
//! rounding, even on CPU in FP32. Apple's own measurements show ~96 dB SNR between
//! CoreML FP32 and source frameworks
//! ([typed execution docs](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html)).
//! See [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks) for
//! full results across multiple datasets.
//!
//! ## Choosing a mode
//!
//! The accuracy gap between `coreml` and `coreml-fast` depends on the type of audio.
//! On meeting recordings with orderly turn-taking (AMI), CoreML Fast matches CoreML
//! within 0.4% DER. On broadcast content with frequent speaker changes (VoxConverse),
//! the gap is ~0.3%. On some datasets like Earnings-21, CoreML Fast actually beats
//! CoreML on DER.
//!
//! `coreml-fast` never misses speech or hallucinates extra speech. The only extra
//! errors are misattributing speech to the wrong speaker near turn boundaries, because
//! the 2s step gives fewer data points to pinpoint where one speaker stops and another
//! starts.
//!
//! See [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks) for
//! full results across all datasets.
//!
//! # CPU & CUDA (Linux, Windows, macOS)
//!
//! Works on any platform with ONNX Runtime. No special Cargo features needed for CPU.
//! Enable the `cuda` feature for NVIDIA GPU acceleration.
//!
//! ## Execution modes
//!
//! | Mode | Backend | Step | Precision | Use case |
//! |------|---------|------|-----------|----------|
//! | `cpu` | ORT CPU | 1s | FP32 | Reference |
//! | `cuda` | ORT CUDA | 1s | FP32 | Best accuracy |
//! | `cuda-fast` | ORT CUDA | 2s | FP32 | Best speed |
//!
//! ## Benchmarks
//!
//! NVIDIA RTX 4090, AMD EPYC 7B13, evaluated on VoxConverse dev
//! (216 files, 1217.8 min, collar=0ms):
//!
//! | Mode | DER | Time | RTFx |
//! |------|-----|------|------|
//! | `cuda` | **7.0%** | 1236s | 59x |
//! | `cuda-fast` | 7.4% | 604s | **121x** |
//! | pyannote community-1 (CUDA) | 7.2% | 2312s | 32x |
//!
//! On VoxConverse test (232 files, 2612.2 min, L40S), `cuda` matches pyannote at
//! 11.1% DER at 50x realtime vs pyannote's 18x.
//!
//! # Models
//!
//! Models download automatically on first use from
//! [avencera/speakrs-models](https://huggingface.co/avencera/speakrs-models) on
//! HuggingFace. If you want a custom model directory, set `SPEAKRS_MODELS_DIR`.
//!
//! # Public API
//!
//! | Module / Type | Description |
//! |---------------|-------------|
//! | [`OwnedDiarizationPipeline`] | Main entry point, owns models and runs diarization |
//! | [`QueueSender`] / [`QueueReceiver`] | Background worker split into push and drain halves |
//! | [`DiarizationPipeline`] | Borrowed pipeline for manual model lifetime control |
//! | [`DiarizationResult`] | All outputs: segments, embeddings, clusters, RTTM |
//! | [`ExecutionMode`] | CPU, CoreML, CoreMLFast, CUDA, CUDAFast |
//! | [`PipelineConfig`] / [`RuntimeConfig`] | Tunable pipeline and hardware parameters |
//! | [`Segment`] | A single speaker turn with start/end times |
//! | [`ModelManager`] | Automatic model download from HuggingFace |
//! | [`inference`] | Segmentation and embedding model wrappers |
//! | [`segment`] | Segment conversion, merging, RTTM formatting |
//!
//! # Why not pyannote-rs?
//!
//! [pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) is another Rust
//! diarization crate, but it uses a simpler pipeline instead of the full pyannote
//! algorithm:
//!
//! | | speakrs | pyannote-rs |
//! |-|---------|-------------|
//! | Segmentation | Powerset decode → 3-speaker activations | Raw argmax on logits (binary speech/non-speech) |
//! | Aggregation | Hamming-windowed overlap-add | None (per-window only) |
//! | Binarization | Hysteresis + min-duration filtering | None |
//! | Embedding model | WeSpeaker ResNet34 (same as pyannote) | WeSpeaker CAM++ (only ONNX model they ship) |
//! | Clustering | PLDA + VBx (Bayesian HMM) | Cosine similarity with fixed 0.5 threshold |
//! | Speaker count | VBx EM estimation | Capped by max_speakers parameter |
//! | pyannote parity | Bit-exact on CPU/CUDA | No, different algorithm and different embedding model |
//!
//! On the VoxConverse dev set, using the 33 files where pyannote-rs produces output
//! (186 min, collar=0ms):
//!
//! | | DER | Missed | False Alarm | Confusion |
//! |-|-----|--------|-------------|-----------|
//! | speakrs CoreML | 11.5% | 3.8% | 3.6% | 4.1% |
//! | pyannote-rs | 80.2% | 34.9% | 7.4% | 37.9% |
//!
//! pyannote-rs produces 0 segments on 183 out of 216 VoxConverse files. Its segments
//! only close on speech to silence transitions, so continuous speech without silence
//! gaps yields no output. The 33 files above are the subset where it produces at least
//! 5 segments.
//!
//! Note: pyannote-rs's README says it uses `wespeaker-voxceleb-resnet34-LM`, but their
//! [build instructions](https://github.com/thewh1teagle/pyannote-rs/blob/main/BUILDING.md)
//! and [GitHub release](https://github.com/thewh1teagle/pyannote-rs/releases/tag/v0.1.0)
//! only ship `wespeaker_en_voxceleb_CAM++.onnx`. There is no ONNX export of
//! ResNet34-LM. The
//! [HuggingFace repo](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
//! only contains `pytorch_model.bin`. The benchmark here uses pyannote-rs exactly as
//! documented in their setup instructions.
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
pub use segment::Segment;

#[cfg(feature = "_metrics")]
pub use powerset::PowersetMapping;
