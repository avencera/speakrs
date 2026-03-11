# Benchmarks

DER (Diarization Error Rate) evaluation on standard datasets, comparing speakrs against pyannote and other implementations.

On VoxConverse, speakrs CoreML matches or beats pyannote community-1 (MPS) on accuracy while running ~4x faster on Apple Silicon. speakrs CoreML Fast is the fastest implementation tested across all datasets, beating FluidAudio on both speed and accuracy.

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
| **speakrs CoreML** | **7.0%** | 2.3% | 2.3% | 2.4% | 779.4s | 94x |
| speakrs CoreML Fast | 7.8% | 2.3% | 2.3% | 3.3% | 410.3s | **178x** |
| FluidAudio | 22.3% | 5.3% | 1.6% | 15.5% | 495.8s | 147x |

### VoxConverse Test (232 files, 2612.2 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **11.1%** | 3.4% | 4.1% | 3.7% | 6705.3s | 23x |
| **speakrs CoreML** | **11.1%** | 3.4% | 4.1% | 3.7% | 1610.8s | 97x |
| speakrs CoreML Fast | 11.9% | 3.3% | 4.1% | 4.5% | 826.3s | **190x** |
| FluidAudio | 32.6% | 5.4% | 3.4% | 23.8% | 1044.4s | 150x |

### AMI IHM (34 files, 1123.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 17.0% | 8.1% | 4.3% | 4.5% | 3326.2s | 20x |
| **speakrs CoreML** | **16.9%** | 8.1% | 4.3% | 4.5% | 855.4s | 79x |
| speakrs CoreML Fast | 17.0% | 8.2% | 4.3% | 4.6% | 425.0s | **159x** |
| FluidAudio | 59.4% | 16.7% | 3.1% | 39.6% | 464.2s | 145x |

### Earnings-21 (44 files, 2355.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **9.7%** | 2.6% | 2.4% | 4.7% | 8009.6s | 18x |
| **speakrs CoreML** | **9.7%** | 2.6% | 2.4% | 4.7% | 1238.8s | 114x |
| speakrs CoreML Fast | 10.9% | 2.5% | 2.5% | 5.9% | 615.1s | **230x** |
| FluidAudio | 43.1% | 4.4% | 2.2% | 36.5% | 1040.2s | 136x |

### Other implementations

[pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) was tested on a 39-file subset and scored 89–92% DER across VoxConverse Dev and Test. See [Why Not pyannote-rs?](../README.md#why-not-pyannote-rs) for details.

### Raw Data

- [VoxConverse Dev](voxconverse-dev.txt)
- [VoxConverse Test](voxconverse-test.txt)
- [AMI IHM](ami-ihm.txt)
- [Earnings-21](earnings-21.txt)

## Batch Sizes

Default pyannote batch size is 32 for both segmentation and embedding (all devices).

Override via environment variables:

```bash
PYANNOTE_SEGMENTATION_BATCH_SIZE=64 PYANNOTE_EMBEDDING_BATCH_SIZE=64 cargo xtask benchmark der ...
```

All results above use batch size 32.

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
