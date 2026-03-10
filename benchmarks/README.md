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

### VoxConverse Dev (39 files, 53 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time |
|---|---|---|---|---|---|
| pyannote MPS | 6.4% | 2.4% | 2.0% | 2.1% | 128.4s |
| speakrs CoreML | 6.4% | 2.4% | 2.0% | 2.0% | 66.8s |
| speakrs CoreML Fast | 9.0% | 2.3% | 2.0% | 4.7% | 40.2s |
| FluidAudio | 16.1% | 5.3% | 1.8% | 8.9% | 29.3s |
| pyannote-rs | 92.4% | 84.1% | 1.0% | 7.3% | 16.9s |

### VoxConverse Test (39 files, 74 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time |
|---|---|---|---|---|---|
| pyannote MPS | 11.2% | 5.0% | 2.0% | 4.2% | 177.1s |
| speakrs CoreML | 11.1% | 5.0% | 2.0% | 4.1% | 66.1s |
| speakrs CoreML Fast | 13.9% | 5.1% | 2.0% | 6.8% | 37.9s |
| FluidAudio | 26.6% | 9.1% | 1.8% | 15.7% | 35.0s |
| pyannote-rs | 89.3% | 80.3% | 0.7% | 8.3% | 21.7s |

### AMI IHM (3 files, 47 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time |
|---|---|---|---|---|---|
| pyannote MPS | 16.0% | 8.6% | 4.2% | 3.2% | 145.1s |
| speakrs CoreML | 16.0% | 8.6% | 4.2% | 3.2% | 46.2s |
| speakrs CoreML Fast | 16.2% | 8.6% | 4.3% | 3.3% | 23.1s |
| FluidAudio | 40.1% | 14.3% | 3.1% | 22.6% | 19.3s |

### Raw Data

- [VoxConverse Dev](voxconverse-dev.txt)
- [VoxConverse Test](voxconverse-test.txt)
- [AMI IHM](ami-ihm.txt)

## Reproduce

Requires models (`just export-models`) and VoxConverse dataset (auto-downloaded on first run)

```bash
# full dev set, all implementations
cargo xtask benchmark der --dataset voxconverse-dev --max-files 39 --max-minutes 60

# full test set, all implementations
cargo xtask benchmark der --dataset voxconverse-test --max-files 39 --max-minutes 80

# AMI IHM
cargo xtask benchmark der --dataset ami-ihm --max-files 3 --max-minutes 50

# specific implementations only
cargo xtask benchmark der --max-files 39 --max-minutes 60 --impl pyannote --impl coreml

# list available implementations
cargo xtask benchmark der --impl list
```

Raw timestamped results from dev runs are in `_benchmarks/` (gitignored)
