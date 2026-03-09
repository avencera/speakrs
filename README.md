# speakrs

Speaker diarization in Rust. Put audio in, get RTTM segments out. Runs up to **63x realtime** on Apple Silicon and is **bit-exact with pyannote** on CPU/CUDA.

`speakrs` implements the full pyannote community-1 pipeline in Rust: segmentation, powerset decode, aggregation, binarization, embedding, PLDA, and VBx clustering. There is no Python dependency. Inference runs on ONNX Runtime or native CoreML, and all post-processing stays in Rust.

The outputs have been checked against `pyannote.audio`. CPU and CUDA modes produce **identical RTTM** on all tested files. CoreML runs the same algorithm, but GPU floating-point non-determinism can change a small number of results. On the VoxConverse dev set, 9 out of 10 files matched exactly.

## Table of Contents

- [Pipeline](#pipeline)
- [macOS / iOS (CoreML)](#macos--ios-coreml)
- [CPU & CUDA (Linux, Windows, macOS)](#cpu--cuda-linux-windows-macos)
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
  ├─ Reconstruct ───────→ map clusters back to frame-level activations
  │
  └─ Segments ──────────→ RTTM output
```

## macOS / iOS (CoreML)

Requires the `native-coreml` Cargo feature. Uses Apple's CoreML framework for GPU/ANE-accelerated inference.

### Execution Modes

| Mode | Backend | Step | Precision | Use case |
|------|---------|------|-----------|----------|
| `coreml` | Native CoreML | 1s | FP32 | Best accuracy, GPU-accelerated |
| `coreml-lite` | Native CoreML | 2s | FP16+ANE | Best speed (63x realtime) |

`coreml-lite` uses a wider step (2s instead of 1s) and FP16 ANE inference to get about 2x more speed. That follows the same throughput-first tradeoff [FluidAudio](https://github.com/FluidInference/FluidAudio) uses on Apple hardware. It matches `coreml` on most clips, but on some inputs the coarser step drops a few segments. The 10.5-minute benchmark below shows one example.

### Benchmarks

All benchmarks were run on an Apple M4 Pro with macOS 26.3.

#### 7-min clip (424.8s)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `coreml-lite` (FP16) | 6 | 79 | 100% | 6.7s | **63x** |
| `coreml` (FP32) | 6 | 79 | 100% | 11.5s | 37x |
| pyannote MPS | 6 | 81 | reference | 28.8s | 15x |

#### 10.5-min clip (635.7s, 3 speakers)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `coreml-lite` (FP16) | 3 | 247 | 99.7% | 5.4s | **118x** |
| `coreml` (FP32) | 3 | 250 | 100% | 16.3s | 39x |
| pyannote MPS | 3 | 250 | reference | 36.9s | 17x |

#### 45-min clip (2700.0s)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `coreml-lite` (FP16) | 2 | 722 | 100% | 42.3s | **64x** |
| `coreml` (FP32) | 2 | 722 | 100% | 72.1s | 37x |
| pyannote MPS | 2 | 720 | reference | 145.3s | 19x |

Coverage is measured as mutual speech overlap with pyannote. Small segment count differences, such as 79 vs 81, come from f32 accumulation order at frame boundaries. No speech is lost or added in those cases. In the 10.5-minute clip, `coreml-lite` drops 3 segments and reaches 99.7% coverage because of the wider 2-second step.

### Accuracy (DER)

Evaluated on VoxConverse dev set (10 files, collar=0ms):

| Mode | DER | Notes |
|------|-----|-------|
| `coreml` (FP32) | 14.9% | +0.9% from GPU floating-point non-determinism |
| pyannote MPS | 14.0% | Reference |

The small `coreml` gap comes from GPU execution producing slightly different embedding vectors because accumulation order and fused multiply-add behavior are not deterministic. That is expected GPU behavior, not a correctness bug. CoreML and pyannote's MPS also use different software stacks on the same Apple GPU, so they drift from CPU in different ways. On 9 out of 10 files, CoreML matches CPU exactly. On one file (`gwtwd`), the embedding difference is enough to move a borderline speaker cluster below the VBx threshold.

### Library Usage

```rust
use speakrs::models::Mode;
use speakrs::pipeline::OwnedDiarizationPipeline;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

### CLI Usage

```bash
# Native CoreML (fastest)
cargo run --release --features native-coreml --bin diarize -- --mode coreml-lite audio.wav

# Native CoreML (accuracy)
cargo run --release --features native-coreml --bin diarize -- --mode coreml audio.wav

# Compare with pyannote
just compare audio.wav
```

## CPU & CUDA (Linux, Windows, macOS)

Works on any platform with ONNX Runtime. No special Cargo features needed for CPU. Enable the `cuda` feature for NVIDIA GPU acceleration.

### Execution Modes

| Mode | Backend | Step | Precision | Use case |
|------|---------|------|-----------|----------|
| `cpu` | ORT CPU | 1s | FP32 | Reference, bit-exact with pyannote CPU |
| `cuda` | ORT CUDA | 1s | FP32 | NVIDIA GPU, bit-exact with pyannote CPU |

Additional Cargo features are available for `directml` (Windows) and `tensorrt` (NVIDIA TensorRT).

### Benchmarks

<!-- TODO: add CPU benchmark numbers -->
<!-- TODO: add CUDA benchmark numbers -->

Benchmarks coming soon.

### Accuracy (DER)

Evaluated on VoxConverse dev set (10 files, collar=0ms):

| Mode | DER | Notes |
|------|-----|-------|
| `cpu` (ONNX) | 14.0% | Identical RTTM to pyannote CPU on all 10 files |
| `cuda` (ONNX) | 14.0% | Identical RTTM to pyannote CPU on all 10 files |
| pyannote CPU | 14.0% | Reference |

The `cpu` and `cuda` modes produce **bit-exact output** with pyannote's CPU backend. The RTTM is identical on every test file.

### Library Usage

```rust
use speakrs::models::Mode;
use speakrs::pipeline::OwnedDiarizationPipeline;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut pipeline = OwnedDiarizationPipeline::from_pretrained(Mode::Cpu)?;

    let audio: Vec<f32> = load_your_mono_16khz_audio_here();
    let result = pipeline.run(&audio)?;

    print!("{}", result.rttm("my-audio"));
    Ok(())
}

fn load_your_mono_16khz_audio_here() -> Vec<f32> {
    unimplemented!()
}
```

### CLI Usage

```bash
# CPU reference
cargo run --release --bin diarize -- --mode cpu audio.wav

# CUDA (NVIDIA GPU)
cargo run --release --features cuda --bin diarize -- --mode cuda audio.wav
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

| Module | Description |
|--------|-------------|
| `inference::segmentation` | Sliding window segmentation (ONNX or CoreML) |
| `inference::embedding` | WeSpeaker embedding with fbank feature extraction |
| `inference::coreml` | Native CoreML wrapper with cached allocation |
| `powerset` | 7-class → 3-speaker powerset decoding |
| `aggregate` | Hamming-windowed overlap-add with warmup trimming |
| `binarize` | Hysteresis binarization + min-duration + padding |
| `clustering::plda` | PLDA whitening/dimensionality reduction (256→128) |
| `clustering::vbx` | VBx Bayesian HMM EM clustering |
| `reconstruct` | Cluster-to-frame mapping, top-K selection |
| `segment` | Time segments, merging, RTTM formatting |
| `utils` | Cosine similarity, L2 norm, logsumexp, centroids |

## Why Not pyannote-rs?

[pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) is another Rust diarization crate, but it uses a simpler pipeline instead of the full pyannote algorithm:

| | speakrs | pyannote-rs |
|---|---------|-------------|
| Segmentation | Powerset decode → 3-speaker activations | Raw argmax on logits (binary speech/non-speech) |
| Aggregation | Hamming-windowed overlap-add | None (per-window only) |
| Binarization | Hysteresis + min-duration filtering | None |
| Embedding model | WeSpeaker ResNet34 (same as pyannote) | WeSpeaker CAM++ (only ONNX model they ship) |
| Clustering | PLDA + VBx (Bayesian HMM) | Cosine similarity with fixed 0.5 threshold |
| Speaker count | VBx EM estimation | Capped by max_speakers parameter |
| pyannote parity | Bit-exact on CPU/CUDA | No, different algorithm and different embedding model |

On the VoxConverse dev set, using the 33 files where pyannote-rs produces output (186 minutes total, collar=0ms):

| | DER | Missed | False Alarm | Confusion |
|---|-----|--------|-------------|-----------|
| speakrs CoreML | 11.5% | 3.8% | 3.6% | 4.1% |
| pyannote-rs | 80.2% | 34.9% | 7.4% | 37.9% |

pyannote-rs produces 0 segments on 183 out of 216 VoxConverse files. Its segments only close on speech to silence transitions, so continuous speech without silence gaps yields no output. The 33 files above are the subset where it produces at least 5 segments.

Note: pyannote-rs's README says it uses `wespeaker-voxceleb-resnet34-LM`, but their [build instructions](https://github.com/thewh1teagle/pyannote-rs/blob/main/BUILDING.md) and [GitHub release](https://github.com/thewh1teagle/pyannote-rs/releases/tag/v0.1.0) only ship `wespeaker_en_voxceleb_CAM++.onnx`. There is no ONNX export of ResNet34-LM. The [HuggingFace repo](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) only contains `pytorch_model.bin`. The benchmark here uses pyannote-rs exactly as documented in their setup instructions.

## [Contributing](CONTRIBUTING.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, model downloads, fixture generation, and the standard check commands used in this repo.

## References

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Python reference implementation
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - VBx + PLDA pipeline
- [FluidAudio](https://github.com/FluidInference/FluidAudio) - Swift reference (same VBx architecture)
