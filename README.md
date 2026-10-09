# speakrs

Fast Rust speaker diarization (who spoke when) with pyannote-level accuracy.

On VoxConverse dev, `speakrs` CoreML gets **7.1% DER at 529x realtime** versus pyannote's 7.2% at 24x. Full results are in [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks).

If you want a small end-to-end app using it, see [avencera/smrze](https://github.com/avencera/smrze).

## Overview

<!-- cargo-rdme start -->

`speakrs` implements the full pyannote `community-1` style diarization
pipeline in Rust: segmentation, powerset decode, overlap-add aggregation,
binarization, embedding, PLDA, and VBx clustering.

There is no Python runtime in the library path. Inference runs on native CUDA
(NVIDIA), native CoreML (macOS), or ONNX Runtime (CPU, AMD), and the rest of the
pipeline stays in Rust.

## Usage

No inference backend is enabled by default. Pick the one for your platform:

```toml
# macOS (CoreML)
speakrs = { version = "0.6", features = ["coreml"] }

# NVIDIA GPU
speakrs = { version = "0.6", features = ["cuda"] }

# CPU
speakrs = { version = "0.6", features = ["cpu"] }

# AMD GPU
speakrs = { version = "0.6", features = ["migraphx"] }
```

The `coreml` and `cuda` features run native backends, so a build with only those
features does not compile, link, or download ONNX Runtime.

### Quick start

```rust
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;

    let audio: Vec<f32> = load_your_mono_16khz_audio_here();
    let result = pipeline.run(&audio)?;

    print!("{}", result.rttm("my-audio"));
    Ok(())
}
```

### Speaker turns

```rust

let result = pipeline.run(&audio)?;

for segment in result.discrete_diarization.to_segments() {
    println!("{:.3} - {:.3}  {}", segment.start, segment.end, segment.speaker);
}
```

### Background queue

[`QueueSender`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueSender.html) and [`QueueReceiver`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueReceiver.html) run a background worker. Use
[`QueueSender::try_push`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueSender.html#method.try_push) to submit audio without blocking. A full queue
returns the request in [`QueueError::Full`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/enum.QueueError.html#variant.Full) so the sender can retry it:

```rust
use std::time::Duration;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline, QueueError, QueuedDiarizationRequest};

let pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
let (tx, rx) = pipeline.into_queued()?;

let sender = std::thread::spawn(move || -> Result<(), QueueError> {
    for (file_id, audio) in receive_files() {
        let mut request = QueuedDiarizationRequest::new(file_id, audio);
        loop {
            match tx.try_push(request) {
                Err(QueueError::Full(rejected)) => {
                    request = rejected;
                    std::thread::sleep(Duration::from_millis(10));
                }
                Ok(_) => break,
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
});

for result in rx {
    let result = result?;
    print!("{}", result.result?.rttm(&result.file_id));
}
sender.join().expect("sender thread panicked")?;
```

### Local models

For offline or airgapped setups, load models from a local directory:

```rust
use std::path::Path;
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

let mut pipeline = OwnedDiarizationPipeline::from_dir(
    Path::new("/path/to/models"),
    ExecutionMode::Cpu,
)?;
let result = pipeline.run(&audio)?;
```

## Choosing a mode

| Mode | Backend | Step | Use it for |
|------|---------|------|------------|
| `cpu` | ONNX Runtime CPU | 1s | CPU runs and widest compatibility |
| `coreml` | Native CoreML | 1s | macOS with CoreML acceleration |
| `coreml-fast` | Native CoreML | 2s | macOS with CoreML acceleration and higher throughput |
| `cuda` | Native CUDA | 1s | NVIDIA GPU |
| `cuda-fast` | Native CUDA | 2s | NVIDIA GPU for higher throughput |
| `migraphx` | ONNX Runtime MIGraphX | 1s | AMD GPU |

Each mode needs its Cargo feature: `cpu`, `coreml` for both CoreML modes, `cuda` for both
CUDA modes, or `migraphx`. Requesting a mode whose feature is off returns
`ModelLoadError::UnsupportedExecutionMode`.

The `*-fast` modes move the segmentation window every 2 seconds instead of
every 1 second. That gives the pipeline fewer windows to score, so it can be much faster, but speaker changes
may land a little farther from the exact word or pause where they happened.

Use the 1 second modes when you care about exactly when each speaker starts and stops,
short clips, interviews with quick back-and-forth, or audio you plan to subtitle or edit. The 2 second modes
are usually worth trying for long recordings where speed matters more than exact speaker-change times, such as
meetings, lectures, podcasts, or bulk archives.

## Benchmarks

VoxConverse dev, collar=0ms:

| Platform | Implementation | DER | Time | RTFx |
|----------|----------------|-----|------|------|
| Apple M4 Pro | `speakrs` `coreml` | **7.1%** | 138s | 529x |
| Apple M4 Pro | `speakrs` `coreml-fast` | 7.4% | 169s | 434x |
| Apple M4 Pro | pyannote community-1 (MPS) | 7.2% | 2999s | 24x |
| RTX 4090 | `speakrs` `cuda` | **7.0%** | 1236s | 59x |
| RTX 4090 | `speakrs` `cuda-fast` | 7.4% | 604s | **121x** |
| RTX 4090 | pyannote community-1 (CUDA) | 7.2% | 2312s | 32x |

On VoxConverse test, `coreml` matches pyannote at 11.1% DER and runs at
631x realtime versus pyannote's 23x. `cuda` matches pyannote at 11.1% DER
and runs at 50x realtime versus pyannote's 18x. See
[benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks) for
the full tables across all datasets.

The RTX 4090 rows were measured with the ONNX Runtime CUDA backend that the native
CUDA backend replaced. CoreML, CUDA and ONNX Runtime can differ slightly even in FP32,
because floating-point reduction order changes rounding.

## Why not pyannote-rs?

Despite its name, [pyannote-rs](https://github.com/thewh1teagle/pyannote-rs)
is not a Rust port of the full pyannote diarization pipeline. It provides
useful building blocks for a much simpler, lightweight diarizer.

In the end-to-end pattern shown by its examples, `pyannote-rs`:

1. runs the pyannote segmentation model on non-overlapping 10-second windows;
2. reduces every frame to speech or non-speech, emitting one segment for each
   uninterrupted region of speech;
3. computes one speaker embedding for that entire segment; and
4. assigns the segment online by comparing its embedding with previously seen
   speakers using a fixed cosine-similarity threshold.

That is closer to voice activity detection followed by online speaker
matching than to pyannote `community-1`.

| | `speakrs` | `pyannote-rs` |
|-|-----------|---------------|
| Segmentation output | Preserves separate local speaker tracks and overlapping speech | Collapses all non-silence classes into one speech track |
| Windows | Scores every 1s or 2s and reconciles overlapping 10s windows | Scores independent, non-overlapping 10s windows |
| Embeddings | One masked embedding per local speaker and window | One embedding per contiguous speech segment |
| Clustering | Recording-wide AHC initialization, PLDA, and VBx refinement | Immediate cosine-threshold match against stored embeddings |
| Final timeline | Reconstructs speaker count and identities from all windows, then filters short activity | Emits a segment only after a speech-to-silence transition |

These differences explain the accuracy gap. A speech region can contain
several turns with no silence between them. `pyannote-rs` treats that region
as one segment, mixes all voices into one embedding, and gives it one speaker
label. It also discards the segmentation model's separate local speaker
tracks, so it cannot represent overlapping speakers. Its online assignments
are not reconsidered using evidence from the rest of the recording.

`speakrs` keeps the per-speaker frame activity, extracts speaker-conditioned
embeddings, combines evidence from overlapping windows, and clusters all
usable embeddings together. PLDA makes the embeddings more discriminative
for speaker identity, while VBx uses the recording's temporal and global
evidence to refine speaker assignments. The final reconstruction can
therefore preserve rapid speaker changes and overlapping speech instead of
reducing a whole speech region to one voice.

The benchmark reflects that architectural difference. With `pyannote-rs`
v0.3 on VoxConverse dev, it returned no RTTM segments for 183 of 216 files.
On the remaining 33 files where it produced at least five segments
(186 minutes, collar=0ms), the result was:

| | DER | Missed speech | False alarm | Speaker confusion |
|-|-----|---------------|-------------|-------------------|
| `speakrs` CoreML | **11.5%** | 3.8% | 3.6% | 4.1% |
| `pyannote-rs` | 80.2% | 34.9% | 7.4% | 37.9% |

Lower DER is better. The 80.2% figure is the subset-only comparison; it does
not count the 183 files for which `pyannote-rs` produced no output.

## Models

With the default `online` feature, models download on first use from
[avencera/speakrs-models](https://huggingface.co/avencera/speakrs-models).
Set `SPEAKRS_MODELS_DIR` if you want to force a local bundle instead.

## Features and build notes

Enable at least one inference backend; the build fails with a clear error otherwise:

- `coreml`: native CoreML backend on macOS, without ONNX Runtime
- `cpu`: CPU backend via ONNX Runtime
- `cuda`: Linux-only native NVIDIA backend with speakrs kernels on every GPU,
  without ONNX Runtime. cuDNN and cuBLAS load only for a fallback or an explicit choice
- `migraphx`: AMD GPU backend via ONNX Runtime MIGraphX

Other features:

- `online` (default): model download via [`ModelManager`](https://docs.rs/speakrs/latest/speakrs/models/struct.ModelManager.html)
- `load-dynamic`: load the ONNX Runtime library at startup instead of static linking; use it
  with `cpu` or `migraphx`
- `cuda-sm75`, `cuda-sm80`, `cuda-sm90`, `cuda-sm120`: driver-only targets for
  Turing, Ampere/Ada, Hopper, and consumer Blackwell. Each embeds every area's best
  shipped kernel variant. These features do not include cuDNN or cuBLAS
- `cuda-rtx20`: RTX 20 (Turing); `cuda-rtx30`: RTX 30 (Ampere);
  `cuda-rtx40`: RTX 40 (Ada); `cuda-a100`: A100 (Ampere);
  `cuda-rtx50`: RTX 50 (consumer Blackwell)

CUDA is not in `default`, because the backend runs only on Linux. For a driver-only
build, use `speakrs = { default-features = false, features = ["cuda-rtx50"] }`.
Features are additive: adding `cuda` also adds cuDNN and cuBLAS. A target-only model
load returns a typed error with the boundary, batch and math if a kernel is missing.

The CUDA backend builds without a CUDA toolkit. It needs an NVIDIA driver and a
Turing (compute capability 7.5) or newer GPU. The `cuda` feature also needs cuDNN 9
and cuBLAS when a selected plan uses them, and NVRTC for `PersistDynamic` LSTM.
CUDA modes use `segmentation-3.0.safetensors` and
`wespeaker-multimask-tail.safetensors`. `RuntimeConfig` selects precision and graphs.
`SPEAKRS_CUDA_PTX_TIER=sm75` limits the kernel tier. The `cuda` build uses
speakrs kernels on every GPU, including devices with no measured recipe. It embeds
all target kernels and keeps cuDNN and cuBLAS for fallback. Selection uses this order:
**matching user tune file > built-in kernel recipe > device-class or portable kernel default > Library fallback**.
Library is used only when a kernel refuses a layer because of a capability limit,
its weight contract, or an unimplemented geometry, or when no kernel covers the
layer's arithmetic mode. Invalid geometry, CUDA errors, and artifact load errors
are returned to the caller. A tune file can explicitly select Library.

Built-in recipes contain kernel settings only. The RTX 4060 Ti (cc 8.9, 34 SMs)
and RTX 5060 Ti (cc 12.0, 36 SMs) recipes use exact device names. With FP32
segmentation and TF32 embedding, they use the driver-only kernel settings at
embedding batches 1, 4, 8, 16, and 32. The 4060 Ti uses FP16 same-channel trunk
kernels for C32/C64 at every batch and for C128/C256 from batch 8. Its FP32
SincNet recipe also applies outside this whole-pipeline precision mode.

The Tesla T4 recipe (cc 7.5, 40 SMs, exact device name) uses kernel settings
at all pipeline batch classes, including FP16 same-channel trunk kernels in
TF32 embedding mode. It does not select Library for small per-layer timing wins.
The A100 PCIe-40GB and SXM4-40GB recipes (cc 8.0, 108 SMs, exact device names)
use FP32 segmentation and TF32 embedding. Other devices use their kernel defaults.
Measured speed evidence is reported but is not required to select a kernel.
Target-only builds remain driver-only and return typed errors instead of Library
fallbacks.
`SPEAKRS_CUDA_FORCE_LIBRARY=1` makes a `cuda` build use Library at each replaceable
boundary, even when a tune file exists. The model-load log gives the source for
each boundary: tune file, recipe, default, or library.

### Tested NVIDIA GPUs

These end-to-end results use a 10-file VoxConverse subset (about 1.9 hours of
audio), FP32 segmentation and TF32 embedding. RTFx is audio duration divided by
wall time, so higher is faster. Each figure is the median of three alternating
rounds. The RTX 4090 host had 16 CPU cores and the others had 2. These are
measured results, not guarantees for other hosts.

| GPU | `cuda` RTFx | Driver-only RTFx |
|---|---:|---:|
| T4 | 197× | 198× |
| A100 40 GB | 752× | 733× |
| L4 | 285× | 286× |
| A10 | 404× | 405× |
| RTX 4090 | 965× | 1002× |

The RTX 4060 Ti and RTX 5060 Ti also have measured kernel recipes. On the same
subset, driver-only builds were 1.6× faster than cuDNN and cuBLAS on the RTX 4090
and L4, 2.0× on the A10, and 1.4× on the H100 PCIe. The H100 has no tuned
recipe: it is faster than the libraries end to end, but some individual layers
are slower.

Other Turing or newer GPUs use device-class defaults. `speakrs cuda tune`
measures and saves the fastest kernels for the current device; add
`--include-library` to also compare with cuDNN and cuBLAS.

The libraries still win the layers below, each by 2–8 µs. This doesn't change
end-to-end speed.

| GPU | Layer | Batch | speakrs | cuDNN/cuBLAS |
|---|---|---:|---:|---:|
| T4 | `layer4.0.shortcut` | 4 | 149 µs | 141 µs |
| T4 | `linear0` / `linear1` (FP32) | 1 | 18 / 13 µs | 15 / 11 µs |
| A100 | `layer4.0.conv1` | 4 | 59 µs | 57 µs |
| L4, A10, RTX 4090 | `seg_1` | 32 | 31 / 42 / 15 µs | 28 / 37 / 13 µs |

### CUDA tuning

Build the opt-in command on Linux:

```sh
cargo build --release --no-default-features --features cuda --bin speakrs
./target/release/speakrs cuda tune --models-dir /path/to/models --dry-run
./target/release/speakrs cuda tune --models-dir /path/to/models
```

The command uses real model weights and shapes. It measures approved kernel
configurations at the pipeline batch sizes. Use `--include-library` to also time
Library in a `cuda` build. Per-layer times do not include library load or handle
initialization costs. It uses
warm-up passes, alternating candidate order, and median CUDA-event times. The
summary table shows each candidate and marks the selected choice. Tuning does not
change accuracy rules or precision. The defaults are FP32 segmentation, FP32
filterbank, and TF32 embedding. Use `--segmentation-math fp32|tf32` and
`--embedding-math fp32|tf32` to match another pipeline configuration. Use
`--device N` to select a device. Run tuning when other GPU work is stopped.

A target-only build, such as `--features cuda-rtx40`, measures kernels only.
Only fixed pins with reviewed end-to-end accuracy are eligible. The tuner
lists FP16 and non-FP16 pins as separate choices and measures each approved
choice. FP16 is eligible only in TF32 mode, including on supported devices
without an FP16 startup recipe. Recipes remain startup defaults, not limits
on the tuning choices. Timing and edited JSON cannot approve other algorithms.
FP32 mode accepts direct FP32 algorithms. TF32 mode also accepts direct TF32
algorithms, staged one-product Winograd, C64 FFMA Winograd, and FP16 trunk
algorithms. A target-only tuner returns an error when a complete model boundary
has no approved choice. A hybrid build can use Library for such a boundary.
The tuner measures embedding trunk batches 1, 4, 8, 16, and 32. Segmentation
and embedding-head choices use batches 1 and 32. Filterbank uses batches 1
through 32. The exact device and build key rejects stale artifact bytes.
Arbitrary configuration pins cannot be selected by editing JSON. Tuning does
not approve new kernels or change the configured precision.

The file is saved under `$XDG_CONFIG_HOME/speakrs/`, or
`$HOME/.config/speakrs/`, with a per-device name. Use `--output PATH` to change
that location, then set `SPEAKRS_CUDA_TUNE_FILE=PATH` when loading a model.
The same environment variable can set the tuning output path. `--dry-run`
measures and prints the table without writing or changing a file.

A file is used only when the device name, compute capability, SM count, NVIDIA
driver release, cuDNN and cuBLAS versions (or driver-only mode), speakrs version,
embedded artifact digest, and accuracy-policy version all match. A bad key or
row rejects the complete file; selection then uses recipes and defaults.
Missing rows use the normal fallback. There is no automatic tuning at startup.
The library API is `speakrs::inference::cuda::{tune_cuda, CudaTuneOptions}`.

The ONNX Runtime dependency behind `cpu` and `migraphx` (`ort` 2.0.0-rc.13) is still
pre-release.

## Public API

Start here:

- [`OwnedDiarizationPipeline`](https://docs.rs/speakrs/latest/speakrs/pipeline/struct.OwnedDiarizationPipeline.html): pipeline entry point
- [`QueueSender`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueSender.html) and [`QueueReceiver`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueReceiver.html): background worker interface
- [`QueueConfig`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueConfig.html): in-process queue capacity
- [`DiarizationResult`](https://docs.rs/speakrs/latest/speakrs/pipeline/types/data/struct.DiarizationResult.html): frame-level activations, segments, clusters, embeddings, RTTM
- [`PipelineConfig`](https://docs.rs/speakrs/latest/speakrs/pipeline/config/struct.PipelineConfig.html) and [`RuntimeConfig`](https://docs.rs/speakrs/latest/speakrs/pipeline/config/struct.RuntimeConfig.html): tuning knobs
- [`ModelManager`](https://docs.rs/speakrs/latest/speakrs/models/struct.ModelManager.html): model download when `online` is enabled
- [`Segment`](https://docs.rs/speakrs/latest/speakrs/segment/struct.Segment.html): a single speaker turn

<!-- cargo-rdme end -->

## [Contributing](CONTRIBUTING.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, model downloads, fixture generation, and the standard check commands used in this repo.

## References

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Python reference implementation
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - VBx + PLDA pipeline
- [SpeakerKit](https://github.com/argmaxinc/WhisperKit) - Swift reference (same VBx architecture)

On a Linux GPU host, run `scripts/cuda/prove-driver-only.sh MODELS_DIR SHORT_WAV
cuda-rtx50` to check the binary links and library opens during model load and a
short diarization. Exit 0 means the full run passed. Exit 3 means a kernel is
missing, with no cuDNN or cuBLAS link or open.
