# speakrs

Speaker diarization pipeline in Rust. Takes audio in, produces RTTM segments out.

Implements the full pyannote community-1 pipeline: audio → ONNX inference → powerset decode → overlap-add aggregation → hysteresis binarization → speaker embedding → PLDA transform → VBx clustering → segment reconstruction → RTTM output.

No pyannote-rs dependency — uses `ort` directly for ONNX inference to get raw logits, then does all post-processing natively in Rust.

Output is numerically verified against pyannote.audio — golden test fixtures are generated from the Python reference and matched exactly.

## Pipeline

```
Audio (16kHz f32)
  │
  ├─ Segmentation (ONNX) ─→ raw 7-class logits per 10s window
  │
  ├─ Powerset Decode ─────→ 3-speaker soft/hard activations
  │
  ├─ Overlap-Add ─────────→ Hamming-windowed aggregation across windows
  │
  ├─ Binarize ────────────→ hysteresis thresholding + min-duration filtering
  │
  ├─ Embedding (ONNX) ───→ 256-dim WeSpeaker vectors per segment
  │
  ├─ PLDA Transform ─────→ 128-dim whitened features
  │
  ├─ VBx Clustering ─────→ Bayesian HMM speaker assignments
  │
  ├─ Reconstruct ─────────→ map clusters back to frame-level activations
  │
  └─ Segments ────────────→ RTTM output
```

## Modules

| Module | Description |
|--------|-------------|
| `inference::segmentation` | ONNX segmentation model with sliding window |
| `inference::embedding` | WeSpeaker ONNX model + fbank feature extraction |
| `powerset` | 7-class → 3-speaker powerset decoding (hard/soft) |
| `aggregate` | Hamming-windowed overlap-add with warmup trimming |
| `binarize` | Hysteresis binarization + min-duration + padding |
| `clustering::plda` | PLDA whitening/dimensionality reduction (256→128) |
| `clustering::vbx` | VBx Bayesian HMM EM clustering |
| `reconstruct` | Cluster-to-frame mapping, top-K selection, exclusive mode |
| `segment` | Time segments, merging, RTTM formatting |
| `utils` | Cosine similarity, L2 norm, logsumexp, centroids |

## Dependencies

- **ort** — ONNX Runtime for segmentation + embedding model inference
- **kaldi-native-fbank** — pure Rust fbank feature extraction
- **ndarray** — array operations throughout

## Development

Set up the Python tool environment once:

```sh
uv sync --group dev
```

Download ONNX models and PLDA parameters (requires a [HuggingFace token](https://huggingface.co/settings/tokens) with access to the gated repos):

```sh
# accept terms at:
#   https://huggingface.co/pyannote/segmentation-3.0
#   https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
HF_TOKEN=your_token just download-models
```

Models are saved to `fixtures/models/` (gitignored).

Regenerate golden test fixtures from Python (requires `HF_TOKEN`):

```sh
just generate-fixtures
```

```sh
just check    # fmt + clippy + test
just test     # run tests (e2e tests require ONNX models)
just fmt      # cargo fmt + Python formatting
just clippy   # cargo clippy -- -D warnings + ty check
```

All Python scripts in the repo are fully typed and type checked with `ty`.

## References

- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — Python reference implementation
- [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) — VBx + PLDA pipeline
- [FluidAudio](https://github.com/FluidInference/FluidAudio) — Swift reference (same VBx architecture)
