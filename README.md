# speakrs

Speaker diarization in Rust. Runs **79–230x realtime** on Apple Silicon, matching pyannote accuracy at the high end and trading some accuracy for speed at the low end. On Apple Silicon you can choose `CoreML` mode for max accuracy, or `CoreMlFast` for 2x speed improvement trading off some accuracy for some workloads.

`speakrs` implements the full pyannote `community-1` pipeline in Rust: segmentation, powerset decode, aggregation, binarization, embedding, PLDA, and VBx clustering, plus temporal smoothing during reconstruction. There is no Python dependency. Inference runs on ONNX Runtime or native CoreML, and all post-processing stays in Rust.

On the full VoxConverse dev set (216 files), speakrs CoreML achieves **7.0% DER vs pyannote's 7.2%**, slightly better accuracy at 4x the speed on Apple Silicon. On the test set (232 files) both match at 11.1% DER. speakrs CoreML Fast is the fastest implementation tested across all datasets, beating FluidAudio on both speed and accuracy. See [benchmarks/](benchmarks/) for full results.

## Table of Contents

- [Pipeline](#pipeline)
- [macOS / iOS (CoreML)](#macos--ios-coreml)
- [CPU & CUDA (Linux, Windows, macOS)](#cpu--cuda-linux-windows-macos)
- [Usage](#usage)
- [Models](#models)
- [Modules](#modules)
- [Why Not pyannote-rs?](#why-not-pyannote-rs)
- [Contributing](#contributing)
- [References](#references)

## Pipeline

```
Audio (16kHz f32)
  │
  ├─ Segmentation ──────→ raw 7-class logits per 10s window
  │   (ONNX or CoreML)
  │
  ├─ Powerset Decode ───→ 3-speaker soft/hard activations
  │
  ├─ Overlap-Add ───────→ Hamming-windowed aggregation across windows
  │
  ├─ Binarize ──────────→ hysteresis thresholding + min-duration filtering
  │
  ├─ Embedding ─────────→ 256-dim WeSpeaker vectors per segment
  │   (ONNX or CoreML)
  │
  ├─ PLDA Transform ────→ 128-dim whitened features
  │
  ├─ VBx Clustering ────→ Bayesian HMM speaker assignments
  │
  ├─ Reconstruct ───────→ map clusters back to frame-level activations (temporal smoothing)
  │
  └─ Segments ──────────→ RTTM output
```

## macOS / iOS (CoreML)

Requires the `coreml` Cargo feature. Uses Apple's CoreML framework for GPU/ANE-accelerated inference.

### Execution Modes

| Mode          | Backend       | Step | Precision | Use case                     |
| ------------- | ------------- | ---- | --------- | ---------------------------- |
| `coreml`      | Native CoreML | 1s   | FP32      | Best accuracy (94x realtime) |
| `coreml-fast` | Native CoreML | 2s   | FP32      | Best speed (178x realtime)   |

`coreml-fast` uses a wider step (2s instead of 1s) to get about 2x more speed. That follows the same throughput-first tradeoff [FluidAudio](https://github.com/FluidInference/FluidAudio) uses on Apple hardware. It matches `coreml` on most clips, but on some inputs the coarser step loses temporal resolution at speaker boundaries.

### Benchmarks

All benchmarks on Apple M4 Pro, macOS 26.3, evaluated on VoxConverse dev (216 files, 1217.8 min, collar=0ms):

| Mode | DER | Time | RTFx |
|------|-----|------|------|
| `coreml` | **7.0%** | 779s | 94x |
| `coreml-fast` | 7.8% | 410s | **178x** |
| pyannote community-1 (MPS) | 7.2% | 2999s | 24x |
| FluidAudio | 22.3% | 496s | 147x |

On VoxConverse test (232 files, 2612.2 min), both `coreml` and pyannote score 11.1% DER, with `coreml` at 97x realtime vs pyannote's 23x. FluidAudio scores 32.6% DER on the test set.

CoreML may differ slightly from CPU due to GPU floating-point non-determinism. See [benchmarks/](benchmarks/) for full results across multiple datasets.

### Choosing a mode

The accuracy gap between `coreml` and `coreml-fast` depends on the type of audio. On meeting recordings with orderly turn-taking (AMI), CoreML Fast matches CoreML within 0.2% DER. The 2x speed boost comes at essentially no accuracy cost. On broadcast content with frequent speaker changes (VoxConverse), the gap widens to ~0.8%. On earnings calls with many Q&A participants, expect ~1% difference.

`coreml-fast` never misses speech or hallucinates extra speech. The only extra errors are misattributing speech to the wrong speaker near turn boundaries, because the 2s step gives fewer data points to pinpoint where one speaker stops and another starts.

See [benchmarks/](benchmarks/) for full results across all datasets.

## CPU & CUDA (Linux, Windows, macOS)

Works on any platform with ONNX Runtime. No special Cargo features needed for CPU. Enable the `cuda` feature for NVIDIA GPU acceleration.

### Execution Modes

| Mode   | Backend  | Step | Precision | Use case   |
| ------ | -------- | ---- | --------- | ---------- |
| `cpu`  | ORT CPU  | 1s   | FP32      | Reference  |
| `cuda` | ORT CUDA | 1s   | FP32      | NVIDIA GPU |

### Benchmarks

Coming soon.

## Usage

### Library

```rust
use speakrs::models::Mode;
use speakrs::pipeline::OwnedDiarizationPipeline;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Mode::CoreMl, Mode::CoreMlFast, Mode::Cpu, or Mode::Cuda
    let mut pipeline = OwnedDiarizationPipeline::from_pretrained(Mode::CoreMl)?;

    let audio: Vec<f32> = load_your_mono_16khz_audio_here();
    let result = pipeline.run(&audio)?;

    print!("{}", result.rttm("my-audio"));
    Ok(())
}

fn load_your_mono_16khz_audio_here() -> Vec<f32> {
    unimplemented!()
}
```

### CLI

```bash
# CoreML (best accuracy, requires coreml feature)
cargo run --release -p xtask --features coreml --bin diarize -- --mode coreml audio.wav

# CoreML Fast (fastest on Apple Silicon)
cargo run --release -p xtask --features coreml --bin diarize -- --mode coreml-fast audio.wav

# CPU
cargo run --release -p xtask --bin diarize -- --mode cpu audio.wav

# CUDA (NVIDIA GPU)
cargo run --release -p xtask --features cuda --bin diarize -- --mode cuda audio.wav

# Compare with pyannote
just compare audio.wav
```

The result also gives you access to intermediate data:

- `result.segmentations`
- `result.embeddings`
- `result.speaker_count`
- `result.hard_clusters`
- `result.discrete_diarization`

See [examples/README.md](examples/README.md) for runnable end-to-end examples, including speaker turn iteration, airtime reporting, and transcript speaker assignment.

## Models

Models download automatically on first use from [avencera/speakrs-models](https://huggingface.co/avencera/speakrs-models) on HuggingFace. If you want a custom model directory, set `SPEAKRS_MODELS_DIR`.

For development, `just download-models` exports the ONNX models and converts them to CoreML. That command requires Python through `uv`.

## Modules

| Module                    | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| `inference::segmentation` | Sliding window segmentation (ONNX or CoreML)                  |
| `inference::embedding`    | WeSpeaker embedding with fbank feature extraction             |
| `inference::coreml`       | Native CoreML wrapper with cached allocation                  |
| `powerset`                | 7-class → 3-speaker powerset decoding                         |
| `aggregate`               | Hamming-windowed overlap-add with warmup trimming             |
| `binarize`                | Hysteresis binarization + min-duration + padding              |
| `clustering::plda`        | PLDA whitening/dimensionality reduction (256→128)             |
| `clustering::vbx`         | VBx Bayesian HMM EM clustering                                |
| `reconstruct`             | Cluster-to-frame mapping, top-K selection, temporal smoothing |
| `segment`                 | Time segments, merging, RTTM formatting                       |
| `utils`                   | Cosine similarity, L2 norm, logsumexp, centroids              |

## Why Not pyannote-rs?

[pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) is another Rust diarization crate, but it uses a simpler pipeline instead of the full pyannote algorithm:

|                 | speakrs                                 | pyannote-rs                                           |
| --------------- | --------------------------------------- | ----------------------------------------------------- |
| Segmentation    | Powerset decode → 3-speaker activations | Raw argmax on logits (binary speech/non-speech)       |
| Aggregation     | Hamming-windowed overlap-add            | None (per-window only)                                |
| Binarization    | Hysteresis + min-duration filtering     | None                                                  |
| Embedding model | WeSpeaker ResNet34 (same as pyannote)   | WeSpeaker CAM++ (only ONNX model they ship)           |
| Clustering      | PLDA + VBx (Bayesian HMM)               | Cosine similarity with fixed 0.5 threshold            |
| Speaker count   | VBx EM estimation                       | Capped by max_speakers parameter                      |
| pyannote parity | Bit-exact on CPU/CUDA                   | No, different algorithm and different embedding model |

On the VoxConverse dev set, using the 33 files where pyannote-rs produces output (186 min, collar=0ms):

|                | DER   | Missed | False Alarm | Confusion |
| -------------- | ----- | ------ | ----------- | --------- |
| speakrs CoreML | 11.5% | 3.8%   | 3.6%        | 4.1%      |
| pyannote-rs    | 80.2% | 34.9%  | 7.4%        | 37.9%     |

pyannote-rs produces 0 segments on 183 out of 216 VoxConverse files. Its segments only close on speech to silence transitions, so continuous speech without silence gaps yields no output. The 33 files above are the subset where it produces at least 5 segments.

Note: pyannote-rs's README says it uses `wespeaker-voxceleb-resnet34-LM`, but their [build instructions](https://github.com/thewh1teagle/pyannote-rs/blob/main/BUILDING.md) and [GitHub release](https://github.com/thewh1teagle/pyannote-rs/releases/tag/v0.1.0) only ship `wespeaker_en_voxceleb_CAM++.onnx`. There is no ONNX export of ResNet34-LM. The [HuggingFace repo](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) only contains `pytorch_model.bin`. The benchmark here uses pyannote-rs exactly as documented in their setup instructions.

## [Contributing](CONTRIBUTING.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, model downloads, fixture generation, and the standard check commands used in this repo.

## References

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Python reference implementation
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - VBx + PLDA pipeline
- [FluidAudio](https://github.com/FluidInference/FluidAudio) - Swift reference (same VBx architecture)
