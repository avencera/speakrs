# speakrs

Fast Rust speaker diarization (who spoke when) with pyannote-level accuracy.

On VoxConverse dev, `speakrs` CoreML gets **7.1% DER at 529x realtime** versus pyannote's 7.2% at 24x. Full results are in [benchmarks/](https://github.com/avencera/speakrs/tree/master/benchmarks).

If you want a small end-to-end app using it, see [avencera/smrze](https://github.com/avencera/smrze).

## Overview

<!-- cargo-rdme start -->

`speakrs` implements the full pyannote `community-1` style diarization
pipeline in Rust: segmentation, powerset decode, overlap-add aggregation,
binarization, embedding, PLDA, and VBx clustering.

There is no Python runtime in the library path. Inference runs on ONNX
Runtime or native CoreML, and the rest of the pipeline stays in Rust.

## Usage

```toml
# macOS (CoreML)
speakrs = { version = "0.5", features = ["coreml"] }

# NVIDIA GPU
speakrs = { version = "0.5", features = ["cuda"] }

# CPU only
speakrs = "0.5"

# System OpenBLAS
speakrs = { version = "0.5", default-features = false, features = ["online", "openblas-system"] }

# AMD GPU
speakrs = { version = "0.5", features = ["migraphx"] }
```

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

[`QueueSender`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueSender.html) and [`QueueReceiver`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueReceiver.html) run a background worker. Push audio
from any thread and read results as they finish:

```rust
use speakrs::{ExecutionMode, OwnedDiarizationPipeline, QueuedDiarizationRequest};

let pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl)?;
let (tx, rx) = pipeline.into_queued()?;

std::thread::spawn(move || {
    for (file_id, audio) in receive_files() {
        tx.push(QueuedDiarizationRequest::new(file_id, audio)).unwrap();
    }
});

for result in rx {
    let result = result?;
    print!("{}", result.result?.rttm(&result.file_id));
}
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
| `cuda` | ONNX Runtime CUDA | 1s | NVIDIA GPU |
| `cuda-fast` | ONNX Runtime CUDA | 2s | NVIDIA GPU for higher throughput |
| `migraphx` | ONNX Runtime MIGraphX | 1s | AMD GPU |

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

CoreML and ONNX Runtime can differ slightly even in FP32 because the runtime
graphs are not identical and floating-point reduction order changes rounding.

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

Common features:

- `online` (default): model download via [`ModelManager`](https://docs.rs/speakrs/latest/speakrs/models/struct.ModelManager.html)
- `coreml`: native CoreML backend on macOS
- `cuda`: NVIDIA CUDA backend via ONNX Runtime
- `migraphx`: AMD GPU backend via ONNX Runtime MIGraphX
- `load-dynamic`: load the ONNX Runtime library at startup instead of static linking

BLAS backends matter if you disable default features:

- `x86_64` defaults to statically linked Intel MKL
- non-`x86_64` defaults to statically linked OpenBLAS and needs a C toolchain
- no-default builds must enable exactly one of `intel-mkl`, `openblas-static`, or `openblas-system`

```toml
speakrs = { version = "0.5", default-features = false, features = ["online", "intel-mkl"] }
speakrs = { version = "0.5", default-features = false, features = ["online", "openblas-system"] }
```

The ONNX Runtime dependency (`ort` 2.0.0-rc.12) is still pre-release.

## Public API

Start here:

- [`OwnedDiarizationPipeline`](https://docs.rs/speakrs/latest/speakrs/pipeline/struct.OwnedDiarizationPipeline.html): pipeline entry point
- [`QueueSender`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueSender.html) and [`QueueReceiver`](https://docs.rs/speakrs/latest/speakrs/pipeline/queued/struct.QueueReceiver.html): background worker interface
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
