# speakrs

Speaker diarization in Rust. Audio in, RTTM segments out. **63x realtime** on Apple Silicon, **100% pyannote coverage**.

Implements the full pyannote community-1 pipeline (segmentation, powerset decode, aggregation, binarization, embedding, PLDA, VBx clustering) with no Python dependency. Inference runs on ONNX Runtime or native CoreML, all post-processing is pure Rust.

Numerically verified against `pyannote.audio` CPU. Golden test fixtures are generated from the Python reference and matched exactly.

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
| `exact` | ORT CPU | 1s | FP32 | Reference, bit-exact with pyannote CPU |
| `coreml` | Native CoreML | 1s | FP32 | Accuracy (100% pyannote coverage) |
| `mini-coreml` | Native CoreML | 2s | FP16+ANE | Speed (63x realtime) |
| `cuda` | ORT CUDA | 1s | FP32 | NVIDIA GPU |
| `pyannote-cpu` | Python sidecar | 1s | FP32 | Upstream comparison |
| `pyannote-mps` | Python sidecar | 1s | varies | Upstream comparison (Apple GPU) |
| `pyannote-cuda` | Python sidecar | 1s | varies | Upstream comparison (NVIDIA) |

Both `coreml` and `mini-coreml` produce identical speaker counts and 100% mutual speech coverage with pyannote. `mini-coreml` trades a wider step (2s vs 1s) and FP16 ANE inference for ~2x speed.

## Benchmarks

All benchmarks on Apple M4 Pro, macOS 26.3.

### 7-min clip (424.8s)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `mini-coreml` (FP16) | 6 | 79 | 100% | 6.7s | **63x** |
| `coreml` (FP32) | 6 | 79 | 100% | 11.5s | 37x |
| pyannote MPS | 6 | 81 | reference | 28.8s | 15x |

### 45-min clip (2700.0s)

| Mode | Speakers | Segments | Coverage | Time | RTFx |
|------|----------|----------|----------|------|------|
| `mini-coreml` (FP16) | 2 | 722 | 100% | 42.3s | **64x** |
| `coreml` (FP32) | 2 | 722 | 100% | 72.1s | 37x |
| pyannote MPS | 2 | 720 | reference | 145.3s | 19x |

Coverage is measured as mutual speech overlap between speakrs and pyannote output. Minor segment count differences (79 vs 81) are due to f32 accumulation order at frame boundaries. No speech is lost or added.

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

### Model Setup

Download ONNX models and PLDA artifacts:

```bash
just download-models
```

Requires `HF_TOKEN` in the environment (or `huggingface-cli login`) and accepted access terms for:
- `pyannote/segmentation-3.0`
- `pyannote/wespeaker-voxceleb-resnet34-LM`

### Library Usage

`speakrs` expects mono 16kHz audio as `&[f32]` samples and returns a `DiarizationResult`:

```rust
use std::path::Path;

use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::DiarizationPipeline;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let models_dir = Path::new("fixtures/models");

    let mut segmentation = SegmentationModel::new(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        DiarizationPipeline::default_segmentation_step(),
    )?;
    let mut embedding = EmbeddingModel::new(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
    )?;

    let mut pipeline = DiarizationPipeline::new(
        &mut segmentation,
        &mut embedding,
        models_dir,
    )?;

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
cargo run --release --features native-coreml --bin diarize -- --mode mini-coreml audio.wav

# Native CoreML (accuracy)
cargo run --release --features native-coreml --bin diarize -- --mode coreml audio.wav

# CPU reference
cargo run --release --bin diarize -- --mode exact audio.wav

# Compare with pyannote
just compare audio.wav
```

## [Contributing](CONTRIBUTING.md)

## References

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Python reference implementation
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - VBx + PLDA pipeline
- [FluidAudio](https://github.com/FluidInference/FluidAudio) - Swift reference (same VBx architecture)
