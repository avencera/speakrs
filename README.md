# speakrs

Fast Rust speaker diarization (who spoke when) with pyannote-level accuracy.

On VoxConverse dev, `speakrs` CoreML gets **7.1% DER at 529x realtime** versus pyannote's 7.2% at 24x. Full results are in [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks).

If you want a small end-to-end app using it, see [avencera/smrze](https://github.com/avencera/smrze).

## Overview

<!-- cargo-rdme start -->

`speakrs` implements the full pyannote `community-1` style diarization
pipeline in Rust: segmentation, powerset decode, overlap-add aggregation,
binarization, embedding, PLDA, and VBx clustering.

There is no Python runtime in the library path. Inference runs on native CPU, native CUDA
(NVIDIA), native CoreML (macOS), or ONNX Runtime (AMD), and the rest of the
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

The `cpu`, `coreml` and `cuda` features run native backends, so a build with only those
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
| `cpu` | Native Rust CPU | 1s | CPU inference without ONNX Runtime |
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
| Apple M4 Pro | `speakrs` `coreml` | **7.1%** | 138s | **529x** |
| Apple M4 Pro | `speakrs` `coreml-fast` | 7.4% | 169s | 434x |
| Apple M4 Pro | pyannote community-1 (MPS) | 7.2% | 2999s | 24x |
| RTX 4090 | `speakrs` `cuda` | **7.0%** | 75s | 978x |
| RTX 4090 | `speakrs` `cuda-fast` | 7.4% | 45s | **1627x** |
| RTX 4090 (earlier run) | pyannote community-1 (CUDA) | 7.2% | 2312s | 32x |

The speakrs CUDA rows use the native CUDA backend on an RTX 4090 with 16 CPU
cores (the cloud host doesn't report its CPU model). The pyannote CUDA row is
from an earlier RTX 4090 run with an AMD EPYC 7B13 CPU.

On VoxConverse test, `coreml` matches pyannote at 11.1% DER and runs at
631x realtime versus pyannote's 23x. Native `cuda` matches the earlier pyannote
result at 11.1% DER and runs at 962x realtime on RTX 4090. That pyannote
CUDA run used an L40S with an AMD EPYC 9354 CPU and reached 18x realtime. See
[benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks) for
the full tables across all datasets and the timing method.

CoreML, CUDA and ONNX Runtime can differ slightly even in FP32, because
floating-point reduction order changes rounding.

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

Native CPU inference loads `segmentation-3.0.safetensors` and
`wespeaker-multimask-tail.safetensors`, plus the PLDA files and
`wespeaker-voxceleb-resnet34.min_num_samples.txt`. The online model manager downloads
these files for CPU mode, not ONNX graphs.

The CPU model constructors accept the canonical `segmentation-3.0.onnx` and
`wespeaker-voxceleb-resnet34.onnx` paths as family selectors. Those files do not
need to exist; the native weights must be in the same directory. The known native
safetensors paths are also accepted. Arbitrary ONNX models are not supported by
native CPU inference. MIGraphX continues to load ONNX graphs.

## Features and build notes

Enable the backend feature for your platform. The build fails with a clear error
if none is enabled.

| Platform | Feature | Notes |
|---|---|---|
| macOS | `coreml` | Native CoreML, no ONNX Runtime |
| Any CPU | `cpu` | Native Rust, no ONNX Runtime |
| Linux with an NVIDIA GPU | `cuda`, or a GPU-specific feature below | Native CUDA, no ONNX Runtime or CUDA toolkit |
| Linux with an AMD GPU | `migraphx` | ONNX Runtime MIGraphX; add `load-dynamic` |

The default `online` feature downloads models with `ModelManager`.
`load-dynamic` loads ONNX Runtime at run time for `migraphx` or the external ONNX
session helper. The native CPU backend doesn't need it. The ONNX Runtime
dependency (`ort` 2.0.0-rc.13) is still a pre-release.

### NVIDIA GPUs

The CUDA backend runs speakrs's own GPU kernels on Linux. It works on any NVIDIA
GPU from the RTX 20 series and T4 onward, and you don't need the CUDA toolkit to
build or run it.

**Not sure which GPU you'll run on? Use `cuda`.** It includes kernels for every
supported GPU generation, and keeps cuDNN 9 and cuBLAS 12 as a fallback, so
those libraries need to be installed (about 2 GB).

```toml
speakrs = { version = "0.6", features = ["cuda"] }
```

**Know your GPU? Use its feature for fewer dependencies and a smaller install.**
A GPU-specific build needs only the NVIDIA driver: no cuDNN, no cuBLAS and no
CUDA toolkit. That saves about 2 GB of libraries on every machine or container
image (cuDNN 9 is about 1.2 GB and cuBLAS 12 about 0.8 GB as NVIDIA ships them).
On the GPUs we tested, these builds run within a few percent of `cuda`, and
often faster.

| Your GPU | Feature |
|---|---|
| RTX 20 series, T4 | `cuda-rtx20` |
| RTX 30 series, A10, other Ampere | `cuda-rtx30` |
| RTX 40 series, L4, L40S, other Ada | `cuda-rtx40` |
| A100 | `cuda-a100` |
| H100 | `cuda-sm90` |
| RTX 50 series | `cuda-rtx50` |

```toml
speakrs = { version = "0.6", features = ["cuda-rtx40"] }
```

#### Tested NVIDIA GPUs

Real-time factor (RTFx) is audio length divided by processing time, so 965x
means an hour of audio takes about 4 seconds. These runs used a 10-file
VoxConverse subset:

| GPU | `cuda` | GPU-specific build |
|---|---:|---:|
| T4 | 197x | 198x |
| A10 | 404x | 405x |
| L4 | 285x | 286x |
| A100 40 GB | 752x | 733x |
| RTX 4090 | 965x | 1002x |

The RTX 4090 runs had 16 CPU cores and the others had 2. The RTX 4060 Ti and
RTX 5060 Ti also have tuned kernel profiles. On the same subset, speakrs's own
kernels were 1.4x to 2.0x faster than cuDNN and cuBLAS on the RTX 4090, L4, A10
and H100. See [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks)
for full results across all datasets.

#### Getting the most out of other GPUs

GPUs without a built-in profile use good defaults. To get the best speed on
yours, run the tuner once. It measures the kernels on your GPU and saves the
fastest choices, which later runs load automatically:

```sh
cargo build --release --no-default-features --features cuda --bin speakrs
./target/release/speakrs cuda tune --models-dir /path/to/models
```

The [CUDA backend details](https://github.com/avencera/speakrs/blob/master/docs/cuda.md)
cover kernel selection, built-in profiles, tuning options and debugging settings.

## Public API

Start here:

- [`OwnedDiarizationPipeline`](https://docs.rs/speakrs/latest/speakrs/pipeline/struct.OwnedDiarizationPipeline.html): pipeline entry point
- [`QueueSender`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueSender.html) and [`QueueReceiver`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueReceiver.html): background worker interface
- [`QueueConfig`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueConfig.html): in-process queue capacity
- [`DiarizationResult`](https://docs.rs/speakrs/latest/speakrs/pipeline/types/data/struct.DiarizationResult.html): frame-level activations, segments, clusters, embeddings, RTTM
- [`PipelineConfig`](https://docs.rs/speakrs/latest/speakrs/pipeline/config/struct.PipelineConfig.html) and [`RuntimeConfig`](https://docs.rs/speakrs/latest/speakrs/pipeline/config/struct.RuntimeConfig.html): tuning knobs
- `ModelManager`: model download when `online` is enabled
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
