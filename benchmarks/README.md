# Benchmarks

DER (Diarization Error Rate) evaluation on standard datasets, comparing speakrs against pyannote and other implementations. All results use collar=0ms and pyannote batch size 32.

## macOS (Apple Silicon)

Hardware: Apple M4 Pro, macOS 26.3

| Name | Description |
|------|-------------|
| pyannote community-1 (MPS) | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) on Apple GPU (MPS) |
| speakrs CoreML | speakrs with native CoreML, 1s step, FP32 |
| speakrs CoreML Fast | speakrs with native CoreML, 2s step, FP32 |
| SpeakerKit | [SpeakerKit](https://github.com/FluidInference/SpeakerKit) Swift implementation |

### VoxConverse Dev (216 files, 1217.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 7.2% | 2.3% | 2.3% | 2.6% | 2998.8s | 24x |
| **speakrs CoreML** | **7.1%** | 2.3% | 2.3% | 2.6% | 138.2s | 529x |
| speakrs CoreML Fast | 7.4% | 2.3% | 2.3% | 2.8% | 168.5s | 434x |
| SpeakerKit | 7.8% | 2.3% | 2.8% | 2.7% | 234.1s | 312x |

### VoxConverse Test (232 files, 2612.2 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 11.1% | 3.4% | 4.1% | 3.7% | 6705.3s | 23x |
| **speakrs CoreML** | **11.1%** | 3.4% | 4.1% | 3.6% | 248.5s | 631x |
| speakrs CoreML Fast | 11.2% | 3.1% | 4.2% | 3.9% | 181.1s | **865x** |
| SpeakerKit | 11.2% | 3.3% | 4.6% | 3.3% | 211.3s | 742x |

### AMI IHM (34 files, 1123.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 17.0% | 8.1% | 4.3% | 4.5% | 3326.2s | 20x |
| **speakrs CoreML** | **17.2%** | 8.1% | 4.3% | 4.8% | 101.3s | 666x |
| speakrs CoreML Fast | 17.6% | 7.8% | 4.7% | 5.1% | 73.9s | **912x** |
| SpeakerKit | 18.0% | 8.5% | 5.2% | 4.3% | 82.8s | 814x |

### Earnings-21 (44 files, 2355.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 9.7% | 2.6% | 2.4% | 4.7% | 8009.6s | 18x |
| speakrs CoreML | 10.6% | 2.6% | 2.4% | 5.6% | 219.6s | 644x |
| **speakrs CoreML Fast** | **8.9%** | 3.0% | 1.6% | 4.2% | 158.7s | **890x** |
| SpeakerKit | 9.4% | 2.8% | 2.0% | 4.6% | 174.3s | 811x |

### AISHELL-4 (20 files, 763.5 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 11.8% | 3.9% | 3.9% | 4.0% | 2268.4s | 20x |
| **speakrs CoreML** | **11.1%** | 3.9% | 3.9% | 3.3% | 72.7s | 630x |
| speakrs CoreML Fast | 12.1% | 4.4% | 3.8% | 3.9% | 51.5s | **890x** |
| SpeakerKit | 11.7% | 4.3% | 4.1% | 3.3% | 56.5s | 810x |

### ICSI (75 files, 4301.2 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **33.4%** | 19.5% | 9.5% | 4.4% | 14584.5s | 18x |
| **speakrs CoreML** | **33.4%** | 19.5% | 9.5% | 4.4% | 422.6s | 611x |
| speakrs CoreML Fast | 34.0% | 19.2% | 9.9% | 4.9% | 283.8s | **909x** |
| SpeakerKit | 33.8% | 19.4% | 10.1% | 4.3% | 327.8s | 787x |

### AMI SDM (34 files, 1123.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **19.3%** | 9.6% | 4.3% | 5.3% | 3347.4s | 20x |
| speakrs CoreML | 19.7% | 9.6% | 4.4% | 5.8% | 120.2s | 561x |
| speakrs CoreML Fast | 20.8% | 9.4% | 4.6% | 6.9% | 80.7s | **835x** |
| SpeakerKit | 19.9% | 9.8% | 5.3% | 4.9% | 86.6s | 778x |

### AVA-AVD (54 files, 266.1 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **45.1%** | 16.1% | 10.8% | 18.2% | 650.0s | 25x |
| speakrs CoreML | 46.7% | 16.0% | 11.0% | 19.7% | 30.0s | 532x |
| speakrs CoreML Fast | 50.7% | 16.4% | 11.0% | 23.3% | 24.6s | **650x** |
| SpeakerKit | 48.3% | 15.5% | 12.9% | 19.8% | 24.6s | 650x |

## Linux (CUDA)

| Name | Description |
|------|-------------|
| pyannote community-1 (CUDA) | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) on NVIDIA GPU |
| speakrs CUDA | speakrs with ONNX Runtime CUDA EP, 1s step |
| speakrs CUDA Fast | speakrs with ONNX Runtime CUDA EP, 2s step |

### VoxConverse Dev (216 files, 1217.8 min)

Hardware: NVIDIA RTX 4090, AMD EPYC 7B13

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **7.0%** | 2.3% | 2.3% | 2.4% | 1236.3s | 59x |
| speakrs CUDA Fast | 7.4% | 2.3% | 2.2% | 2.8% | 604.4s | **121x** |
| pyannote CUDA | 7.2% | 2.3% | 2.3% | 2.6% | 2312.1s | 32x |

### VoxConverse Test (232 files, 2612.2 min)

Hardware: NVIDIA L40S, AMD EPYC 9354

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **11.1%** | 3.4% | 4.1% | 3.7% | 3109.7s | 50x |
| speakrs CUDA Fast | 11.2% | 3.3% | 4.1% | 3.8% | 1566.2s | **100x** |
| pyannote CUDA | 11.1% | 3.4% | 4.1% | 3.7% | 8677.9s | 18x |

### AMI IHM (34 files, 1123.8 min)

Hardware: NVIDIA L40S, AMD EPYC 9354

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **17.0%** | 8.1% | 4.3% | 4.5% | 1375.1s | 49x |
| speakrs CUDA Fast | 17.4% | 8.2% | 4.3% | 4.9% | 675.3s | **100x** |
| pyannote CUDA | 17.0% | 8.1% | 4.3% | 4.5% | 4388.1s | 15x |

### Earnings-21 (44 files, 2355.8 min)

Hardware: NVIDIA RTX 4090, AMD EPYC 7B13

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| speakrs CUDA | 9.7% | 2.6% | 2.4% | 4.7% | 2692.1s | 53x |
| **speakrs CUDA Fast** | **9.2%** | 2.5% | 2.4% | 4.2% | 1284.1s | **110x** |
| pyannote CUDA | 9.7% | 2.6% | 2.4% | 4.7% | 8036.8s | 18x |

## Other implementations

[pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) was tested on a 39-file subset and scored 89-92% DER across VoxConverse Dev and Test. See [Why Not pyannote-rs?](../README.md#why-not-pyannote-rs) for details.

## Raw Data

- [VoxConverse Dev](voxconverse-dev.txt)
- [VoxConverse Test](voxconverse-test.txt)
- [AMI IHM](ami-ihm.txt)
- [AMI SDM](ami-sdm.txt)
- [Earnings-21](earnings-21.txt)
- [AISHELL-4](aishell-4.txt)
- [ICSI](icsi.txt)
- [AVA-AVD](ava-avd.txt)

## Reproduce

Requires models (`just export-models`) and datasets (auto-downloaded on first run).

```bash
# macOS (CoreML)
cargo xtask benchmark der --dataset voxconverse-dev --impls pmps,scm,scmf,sk

# Linux (CUDA) -- via dstack
cargo xtask dstack bp my-bench --dataset voxconverse-dev,ami-ihm --impls sg,sgf,pg

# list available implementations and datasets
cargo xtask benchmark der --impls list
cargo xtask benchmark der --dataset list
```
