# speakrs

Speaker diarization in Rust. Audio in, RTTM segments out. **63x realtime** on Apple Silicon, **bit-exact with pyannote** on CPU/CUDA.

Implements the full pyannote community-1 pipeline (segmentation, powerset decode, aggregation, binarization, embedding, PLDA, VBx clustering) with no Python dependency. Inference runs on ONNX Runtime or native CoreML, all post-processing is pure Rust.

Numerically verified against `pyannote.audio` — CPU and CUDA modes produce **identical RTTM** on all tested files. CoreML GPU mode runs the same algorithm but GPU floating-point non-determinism means a small number of files may differ (9/10 matched exactly on VoxConverse dev set).

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

## Execution Modes

| Mode | Backend | Step | Precision | Use case |
|------|---------|------|-----------|----------|
| `cpu` | ORT CPU | 1s | FP32 | Reference, bit-exact with pyannote CPU |
| `coreml` | Native CoreML | 1s | FP32 | GPU-accelerated, same algorithm |
| `coreml-lite` | Native CoreML | 2s | FP16+ANE | Speed (63x realtime) |
| `cuda` | ORT CUDA | 1s | FP32 | NVIDIA GPU, bit-exact with pyannote CPU |

`coreml-lite` trades a wider step (2s vs 1s) and FP16 ANE inference for ~2x speed. On most clips it matches `coreml` exactly, but on some inputs the coarser step drops a few segments (see 10.5-min benchmark below).

## Benchmarks

All benchmarks on Apple M4 Pro, macOS 26.3.

### 7-min clip (424.8s)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `coreml-lite` (FP16) | 6 | 79 | 100% | 6.7s | **63x** |
| `coreml` (FP32) | 6 | 79 | 100% | 11.5s | 37x |
| pyannote MPS | 6 | 81 | reference | 28.8s | 15x |

### 10.5-min clip (635.7s, 3 speakers)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `coreml-lite` (FP16) | 3 | 247 | 99.7% | 5.4s | **118x** |
| `coreml` (FP32) | 3 | 250 | 100% | 16.3s | 39x |
| pyannote MPS | 3 | 250 | reference | 36.9s | 17x |

### 45-min clip (2700.0s)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `coreml-lite` (FP16) | 2 | 722 | 100% | 42.3s | **64x** |
| `coreml` (FP32) | 2 | 722 | 100% | 72.1s | 37x |
| pyannote MPS | 2 | 720 | reference | 145.3s | 19x |

Coverage is measured as mutual speech overlap with pyannote. Minor segment count differences (e.g. 79 vs 81) are due to f32 accumulation order at frame boundaries, no speech is lost or added. The 10.5-min clip shows `coreml-lite` dropping 3 segments (99.7% coverage) due to the coarser 2s step.

## Accuracy (DER)

Evaluated on VoxConverse dev set (10 files, collar=0ms):

| Mode | DER | Notes |
|------|-----|-------|
| `cpu` (ONNX) | 14.0% | Identical RTTM to pyannote CPU on all 10 files |
| `coreml` (FP32) | 14.9% | +0.9% from GPU floating-point non-determinism |
| pyannote CPU | 14.0% | Reference |
| pyannote MPS | 14.0% | Reference |

The `cpu` mode produces **bit-exact output** with pyannote's CPU backend — identical RTTM on every test file. The `coreml` gap comes from GPU execution producing slightly different embedding vectors due to non-deterministic accumulation order and fused multiply-add instructions, inherent to GPU computation (not a correctness issue). Both CoreML and pyannote's MPS run on the same Apple GPU but use different software stacks (CoreML vs Metal Performance Shaders), so they diverge from CPU in different ways. On 9/10 files CoreML matches CPU exactly; one file (gwtwd) sees embedding differences that push a marginal speaker cluster below the VBx threshold.

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

## Quick Start

### Models

Models download automatically on first use from [avencera/speakrs-models](https://huggingface.co/avencera/speakrs-models) on HuggingFace. To use a custom model directory, set `SPEAKRS_MODELS_DIR`.

For development, `just download-models` exports the ONNX models and converts to CoreML (requires Python via `uv`).

### Library Usage

`speakrs` expects mono 16kHz audio as `&[f32]` samples and returns a `DiarizationResult`:

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

The result also exposes intermediate data:
- `result.segmentations`
- `result.embeddings`
- `result.speaker_count`
- `result.hard_clusters`
- `result.discrete_diarization`

See [examples/README.md](examples/README.md) for runnable end-to-end examples, including speaker turn iteration, airtime reporting, and transcript speaker assignment.

## CLI Usage

```bash
# Native CoreML (fastest)
cargo run --release --features native-coreml --bin diarize -- --mode coreml-lite audio.wav

# Native CoreML (accuracy)
cargo run --release --features native-coreml --bin diarize -- --mode coreml audio.wav

# CPU reference
cargo run --release --bin diarize -- --mode cpu audio.wav

# Compare with pyannote
just compare audio.wav
```

## [Contributing](CONTRIBUTING.md)

## References

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Python reference implementation
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - VBx + PLDA pipeline
- [FluidAudio](https://github.com/FluidInference/FluidAudio) - Swift reference (same VBx architecture)
