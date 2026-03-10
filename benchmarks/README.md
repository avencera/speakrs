# Benchmarks

DER (Diarization Error Rate) evaluation on standard datasets, comparing speakrs against pyannote and other implementations

All benchmarks run on Apple M4 Pro, macOS 26.3, collar=0ms

## Implementations

| Name | Description |
|------|-------------|
| pyannote MPS | pyannote.audio on Apple GPU (MPS) — reference |
| speakrs CoreML | speakrs with native CoreML, 1s step, FP32   |
| speakrs CoreML Fast | speakrs with native CoreML, 2s step, FP32 |
| FluidAudio | [FluidAudio](https://github.com/FluidInference/FluidAudio) Swift implementation |
| pyannote-rs | [pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) Rust implementation |

## Results

- [VoxConverse Dev](voxconverse-dev.txt) — 39 files, 53 min
- [VoxConverse Test](voxconverse-test.txt) — 39 files, 74 min

## Reproduce

Requires models (`just export-models`) and VoxConverse dataset (auto-downloaded on first run)

```bash
# full dev set, all implementations
cargo xtask benchmark der --dataset voxconverse-dev --max-files 39 --max-minutes 60

# full test set, all implementations
cargo xtask benchmark der --dataset voxconverse-test --max-files 39 --max-minutes 80

# specific implementations only
cargo xtask benchmark der --max-files 39 --max-minutes 60 --impl pyannote --impl coreml

# list available implementations
cargo xtask benchmark der --impl list
```

Raw timestamped results from dev runs are in `_benchmarks/` (gitignored)
