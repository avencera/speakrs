#![warn(missing_docs)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! `speakrs` implements the full pyannote `community-1` style diarization
//! pipeline in Rust: segmentation, powerset decode, overlap-add aggregation,
//! binarization, embedding, PLDA, and VBx clustering.
//!
//! There is no Python runtime in the library path. Inference runs on native CPU, native CUDA
//! (NVIDIA), native CoreML (macOS), or ONNX Runtime (AMD), and the rest of the
//! pipeline stays in Rust.
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
//! The `cpu`, `coreml` and `cuda` features run native backends, so a build with only those
//! features does not compile, link, or download ONNX Runtime.
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
//! | `cpu` | Native Rust CPU | 1s | CPU inference without ONNX Runtime |
//! | `coreml` | Native CoreML | 1s | macOS with CoreML acceleration |
//! | `coreml-fast` | Native CoreML | 2s | macOS with CoreML acceleration and higher throughput |
//! | `cuda` | Native CUDA | 1s | NVIDIA GPU |
//! | `cuda-fast` | Native CUDA | 2s | NVIDIA GPU for higher throughput |
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
//! On VoxConverse dev on an RTX 4090, speakrs `cuda` is about 56 times faster than
//! an earlier pyannote CUDA run, with 7.0% versus 7.2% DER. `cuda-fast` is about 105
//! times faster, with 7.4% DER.
//! On an RTX 4090, an hour of audio takes about 2.0 seconds with `cuda` and
//! 1.1 seconds with `cuda-fast`. On an Apple M4 Pro with CoreML it takes about 7 seconds.
//!
//! VoxConverse dev, collar=0ms:
//!
//! | Platform | Implementation | DER | Time | RTFx |
//! |----------|----------------|-----|------|------|
//! | RTX 4090 | `speakrs` `cuda` | **7.0%** | 40.9s | 1786x |
//! | RTX 4090 | `speakrs` `cuda-fast` | 7.4% | 22.0s | **3325x** |
//! | RTX 4090 | pyannote community-1 (CUDA) | 7.2% | 2301.3s | 32x |
//! | Apple M4 Pro | `speakrs` `coreml` | **7.1%** | 138s | **529x** |
//! | Apple M4 Pro | `speakrs` `coreml-fast` | 7.4% | 169s | 434x |
//! | Apple M4 Pro | pyannote community-1 (MPS) | 7.2% | 2999s | 24x |
//! | Apple M4 Pro | SpeakerKit (March 2026) | 7.8% | 234s | 312x |
//!
//! SpeakerKit was measured on the same M4 Pro in March 2026 with the version available then, and it has shipped releases since.
//!
//! The two speakrs RTX 4090 rows ran in the same container, with 16 CPU cores (the
//! cloud host doesn't report the CPU model), 8 GiB of RAM and NVIDIA driver
//! 580.126.18. Their times are the median of 3 runs. The pyannote row ran earlier in
//! a container with the same GPU, CPU count, RAM and driver, so the speedups compare
//! separate runs. The speakrs CUDA timings use the batch API,
//! [`OwnedDiarizationPipeline::run_batch_stream`], with WAV files decoded on worker
//! threads ahead of it; it clusters each finished file while the GPU runs the next.
//!
//! ## All datasets on an RTX 4090
//!
//! | Dataset | Audio | `cuda` DER | `cuda` RTFx | `cuda-fast` DER | `cuda-fast` RTFx |
//! |---|---:|---:|---:|---:|---:|
//! | VoxConverse dev | 20.3 h | 7.0% | 1786x | 7.4% | 3325x |
//! | VoxConverse test | 43.5 h | 11.1% | 1921x | 11.2% | 3722x |
//! | AMI IHM | 18.7 h | 17.0% | 1745x | 17.4% | 3507x |
//! | AMI SDM | 18.7 h | 19.7% | 1717x | 20.6% | 3286x |
//! | AISHELL-4 | 12.7 h | 11.1% | 1997x | 11.4% | 4258x |
//! | Earnings-21 | 39.3 h | 9.7% | 1747x | 9.2% | 3459x |
//! | ICSI | 71.7 h | 33.3% | 1980x | 33.7% | 4231x |
//! | AVA-AVD | 4.4 h | 45.4% | 1999x | 48.9% | 4286x |
//!
//! That's about 229 hours of audio in under 8 minutes with `cuda`, or under 4
//! minutes with `cuda-fast`. The VoxConverse dev row comes from the runs above.
//! On VoxConverse test, `cuda` and pyannote CUDA both score 11.1% DER; pyannote
//! ran at 25x in an earlier container, about 78 times slower than `cuda`. On AMI
//! IHM and Earnings-21, `cuda` matches the earlier pyannote runs at 17.0% and 9.7% DER,
//! where pyannote ran at 15x and 18x. On macOS, `coreml` runs these datasets at
//! 450x to 644x, with DER within 1.6 points of pyannote's. See
//! [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks) for
//! every table, the hardware used and the timing method.
//!
//! CoreML, CUDA and ONNX Runtime can differ slightly even in FP32, because
//! floating-point reduction order changes rounding.
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
//! Native CPU inference loads `segmentation-3.0.safetensors` and
//! `wespeaker-multimask-tail.safetensors`, plus the PLDA files and
//! `wespeaker-voxceleb-resnet34.min_num_samples.txt`. The online model manager downloads
//! these files for CPU mode, not ONNX graphs.
//!
//! The CPU model constructors accept the canonical `segmentation-3.0.onnx` and
//! `wespeaker-voxceleb-resnet34.onnx` paths as family selectors. Those files do not
//! need to exist; the native weights must be in the same directory. The known native
//! safetensors paths are also accepted. Arbitrary ONNX models are not supported by
//! native CPU inference. MIGraphX continues to load ONNX graphs.
//!
//! # Features and build notes
//!
//! Enable the backend feature for your platform. The build fails with a clear error
//! if none is enabled.
//!
//! | Platform | Feature | Notes |
//! |---|---|---|
//! | macOS | `coreml` | Native CoreML, no ONNX Runtime |
//! | Any CPU | `cpu` | Native Rust, no ONNX Runtime |
//! | Linux with an NVIDIA GPU | `cuda`, or a GPU-specific feature below | Native CUDA, no ONNX Runtime or CUDA toolkit |
//! | Linux with an AMD GPU | `migraphx` | ONNX Runtime MIGraphX; add `load-dynamic` |
//!
//! The default `online` feature downloads models with `ModelManager`.
//! `load-dynamic` loads ONNX Runtime at run time for `migraphx` or the external ONNX
//! session helper. The native CPU backend doesn't need it. The ONNX Runtime
//! dependency (`ort` 2.0.0-rc.13) is still a pre-release.
//!
//! ## NVIDIA GPUs
//!
//! The CUDA backend runs speakrs's own GPU kernels on Linux. It works on any NVIDIA
//! GPU from the RTX 20 series and T4 onward, and you don't need the CUDA toolkit to
//! build or run it.
//!
//! **Not sure which GPU you'll run on? Use `cuda`.** It includes kernels for every
//! supported GPU generation. It can also fall back to cuDNN 9 and cuBLAS 12 for a
//! layer without a speakrs kernel, and loads them only then. On the GPUs listed
//! below every layer has a kernel, so the libraries aren't loaded.
//!
//! ```toml
//! speakrs = { version = "0.6", features = ["cuda"] }
//! ```
//!
//! **Know your GPU? Use its feature for fewer dependencies and a smaller install.**
//! A GPU-specific build needs only the NVIDIA driver and never loads cuDNN or
//! cuBLAS, so you can leave about 2 GB of libraries out of every machine or
//! container image (cuDNN 9 is about 1.2 GB and cuBLAS 12 about 0.8 GB
//! as NVIDIA ships them).
//! On the GPUs we tested, these builds run within about 1% of `cuda`.
//!
//! | Your GPU | Feature |
//! |---|---|
//! | RTX 20 series, T4 | `cuda-rtx20` |
//! | RTX 30 series, A10, other Ampere | `cuda-rtx30` |
//! | RTX 40 series, L4, L40S, other Ada | `cuda-rtx40` |
//! | A100 | `cuda-a100` |
//! | H100 | `cuda-sm90` |
//! | RTX 50 series | `cuda-rtx50` |
//!
//! ```toml
//! speakrs = { version = "0.6", features = ["cuda-rtx40"] }
//! ```
//!
//! ### Tested NVIDIA GPUs
//!
//! Real-time factor (RTFx) is audio length divided by processing time, so 1453x
//! means an hour of audio takes about 2.5 seconds. These runs used a 10-file
//! VoxConverse subset, and their times include process start-up and model loading:
//!
//! | GPU | `cuda` | GPU-specific build |
//! |---|---:|---:|
//! | T4 | 294x | 294x |
//! | A10 | 432x | 436x |
//! | L4 | 310x | 312x |
//! | A100 40 GB | 1296x | 1296x |
//! | RTX 4090 | 1453x | 1438x |
//!
//! Every run had 16 CPU cores, and each value is the median of 3 runs. The RTX 4060
//! Ti and RTX 5060 Ti also have tuned kernel profiles. In an earlier version, on the
//! same subset, speakrs's own kernels were 1.4x to 2.0x faster than cuDNN and cuBLAS
//! on the RTX 4090, L4, A10 and H100. See [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks)
//! for full results across all datasets.
//!
//! ### Getting the most out of other GPUs
//!
//! GPUs without a built-in profile use good defaults. To get the best speed on
//! yours, run the tuner once. It measures the kernels on your GPU and saves the
//! fastest choices, which later runs load automatically. Build the tuner with the
//! same CUDA feature as your application, because a tune file only applies to the
//! build that wrote it:
//!
//! ```sh
//! cargo build --release --no-default-features --features cuda-rtx40 --bin speakrs
//! ./target/release/speakrs cuda tune --models-dir /path/to/models
//! ```
//!
//! The [CUDA backend details](https://github.com/avencera/speakrs/blob/master/docs/cuda.md)
//! cover kernel selection, built-in profiles, tuning options and debugging settings.
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
//! - `ModelManager`: model download when `online` is enabled
//! - [`Segment`]: a single speaker turn

#[cfg(all(
    feature = "_cuda",
    not(any(
        feature = "cuda-sm75",
        feature = "cuda-sm80",
        feature = "cuda-sm90",
        feature = "cuda-sm120"
    ))
))]
compile_error!(
    "CUDA requires a GPU target; enable `cuda`, `cuda-sm75`, `cuda-sm80`, `cuda-sm90`, or `cuda-sm120`"
);

#[cfg(all(feature = "coreml", not(target_os = "macos")))]
compile_error!("the `coreml` feature is only supported on macOS");

#[cfg(not(feature = "_backend"))]
compile_error!(
    "speakrs needs an inference backend; enable at least one of these Cargo features:\n\
     - macOS (Apple Silicon): `coreml`\n\
     - NVIDIA GPU: `cuda`, or a `cuda-sm75`/`cuda-sm80`/`cuda-sm90`/`cuda-sm120` target (native, no ONNX Runtime)\n\
     - AMD GPU: `migraphx`\n\
     - CPU (native Rust): `cpu`\n\
     for example: speakrs = { version = \"0.6\", features = [\"coreml\"] }"
);

#[cfg(all(
    feature = "load-dynamic",
    not(any(feature = "cpu", feature = "migraphx"))
))]
compile_error!(
    "the `load-dynamic` feature loads ONNX Runtime at run time and only applies to the `cpu` and \
     `migraphx` backends; the `cuda` and `coreml` backends do not use ONNX Runtime"
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
    ActivityCleanup, AhcConfig, AhcConfigError, BatchInput, BatchOutput, BatchStreamError,
    ClusteringBackend, ClusteringConfig, ClusteringConfigError, DiarizationPipeline,
    DiarizationResult, FbankSessionPool, FbankSessionPoolSize, FbankSessionPoolSizeError,
    OrtThreadCount, OrtThreadCountError, OwnedBatchInput, OwnedDiarizationPipeline,
    PipelineBuilder, PipelineConfig, PipelineError, QueueConfig, QueueError, QueueReceiver,
    QueueReceiverIter, QueueSender, QueuedDiarizationJobId, QueuedDiarizationRequest,
    QueuedDiarizationResult, ReconstructError, ResponsibilityInitialization, RuntimeConfig,
    VbxConfig, VbxConfigError,
};
#[cfg(feature = "_backend")]
pub use segment::Segment;

#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[cfg(feature = "_backend")]
pub use powerset::{PowersetDecodeError, PowersetMapping};

#[cfg(test)]
mod test_support;
