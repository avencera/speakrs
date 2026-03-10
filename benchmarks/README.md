# Benchmarks

DER (Diarization Error Rate) evaluation on standard datasets, comparing speakrs against pyannote and other implementations.

On VoxConverse, speakrs CoreML matches or beats pyannote community-1 (MPS) on accuracy while running ~3x faster on Apple Silicon.

All benchmarks run on Apple M4 Pro, macOS 26.3, collar=0ms.

## Implementations

| Name | Description |
|------|-------------|
| pyannote community-1 (MPS) | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) on Apple GPU (MPS) — reference |
| speakrs CoreML | speakrs with native CoreML, 1s step, FP32   |
| speakrs CoreML Fast | speakrs with native CoreML, 2s step, FP32 |
| FluidAudio | [FluidAudio](https://github.com/FluidInference/FluidAudio) Swift implementation |

## Results

### VoxConverse Dev (216 files, 1217.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 7.2% | 2.3% | 2.3% | 2.6% | 2998.8s | 24x |
| **speakrs CoreML** | **7.0%** | 2.3% | 2.3% | 2.4% | 1009.1s | 72x |
| speakrs CoreML Fast | 7.8% | 2.3% | 2.3% | 3.3% | 524.0s | 139x |
| FluidAudio | 22.3% | 5.3% | 1.6% | 15.5% | 500.9s | 146x |

### VoxConverse Test (232 files, 2612.2 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **11.1%** | 3.4% | 4.1% | 3.7% | 6705.3s | 23x |
| **speakrs CoreML** | **11.1%** | 3.4% | 4.1% | 3.7% | 2133.5s | 73x |
| speakrs CoreML Fast | 11.9% | 3.3% | 4.1% | 4.5% | 1093.2s | 143x |
| FluidAudio | 32.6% | 5.4% | 3.4% | 23.8% | 1044.2s | 150x |

### AMI IHM (3 files, 47 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **16.0%** | 8.6% | 4.2% | 3.2% | 145.1s | 20x |
| **speakrs CoreML** | **16.0%** | 8.6% | 4.2% | 3.2% | 46.2s | 61x |
| speakrs CoreML Fast | 16.2% | 8.6% | 4.3% | 3.3% | 23.1s | 123x |
| FluidAudio | 40.1% | 14.3% | 3.1% | 22.6% | 19.3s | 147x |

### Other implementations

[pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) was tested on a 39-file subset and scored 89–92% DER across VoxConverse Dev and Test. See [Why Not pyannote-rs?](../README.md#why-not-pyannote-rs) for details.

### Raw Data

- [VoxConverse Dev](voxconverse-dev.txt)
- [VoxConverse Test](voxconverse-test.txt)
- [AMI IHM](ami-ihm.txt)

## Reproduce

Requires models (`just export-models`) and datasets (auto-downloaded on first run).

```bash
# full VoxConverse dev set
cargo xtask benchmark der --dataset voxconverse-dev --impl pyannote --impl coreml --impl coreml-fast --impl fluidaudio

# full VoxConverse test set
cargo xtask benchmark der --dataset voxconverse-test --impl pyannote --impl coreml --impl coreml-fast --impl fluidaudio

# AMI IHM
cargo xtask benchmark der --dataset ami-ihm --impl pyannote --impl coreml --impl coreml-fast --impl fluidaudio

# specific implementations only
cargo xtask benchmark der --dataset voxconverse-dev --impl pyannote --impl coreml

# list available implementations
cargo xtask benchmark der --impl list
```
