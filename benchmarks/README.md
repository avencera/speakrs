# Benchmarks

DER (Diarization Error Rate) results on standard datasets. These tables compare speakrs against pyannote and a few other implementations. All results use collar=0ms and pyannote batch size 32.

## macOS (Apple Silicon)

Hardware: Apple M4 Pro, macOS 26.3

| Name | Description |
|------|-------------|
| pyannote community-1 (MPS) | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) on Apple GPU (MPS) |
| speakrs CoreML | speakrs with native CoreML, 1s step, FP32 |
| speakrs CoreML Fast | speakrs with native CoreML, 2s step, FP32 |
| SpeakerKit | Argmax's [SpeakerKit](https://github.com/argmaxinc/argmax-oss-swift) Swift implementation, pulled through `argmaxinc/WhisperKit` 0.12 or later by `scripts/speakerkit-bench/Package.swift`; measured March 2026 |

### VoxConverse Dev (216 files, 1217.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| pyannote community-1 (MPS) | 7.2% | 2.3% | 2.3% | 2.6% | 2998.8s | 24x |
| **speakrs CoreML** | **7.1%** | 2.3% | 2.3% | 2.6% | 138.2s | **529x** |
| speakrs CoreML Fast | 7.4% | 2.3% | 2.3% | 2.8% | 168.5s | 434x |
| SpeakerKit | 7.8% | 2.3% | 2.8% | 2.7% | 234.1s | 312x |

### VoxConverse Test (232 files, 2612.2 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **11.1%** | 3.4% | 4.1% | 3.7% | 6705.3s | 23x |
| **speakrs CoreML** | **11.1%** | 3.4% | 4.1% | 3.6% | 248.5s | 631x |
| speakrs CoreML Fast | 11.2% | 3.1% | 4.2% | 3.9% | 181.1s | **865x** |
| SpeakerKit | 11.2% | 3.3% | 4.6% | 3.3% | 211.3s | 742x |

### AMI IHM (34 files, 1123.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **pyannote community-1 (MPS)** | **17.0%** | 8.1% | 4.3% | 4.5% | 3326.2s | 20x |
| **speakrs CoreML** | **17.0%** | 8.1% | 4.3% | 4.6% | 149.8s | 450x |
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
| SpeakerKit | 48.3% | 15.5% | 12.9% | 19.8% | 24.6s | **650x** |

## Linux (CUDA)

The speakrs CUDA rows use NVIDIA RTX 4090 containers with 16 CPU cores and
8 GiB of RAM. The cloud host doesn't report its CPU model. The VoxConverse
pyannote rows ran earlier in a container with the same GPU model, CPU count and
RAM, so those speedups compare separate runs.

Audio and models were copied to the container's local disk before each dataset
run. Time and RTFx use the `speakrs-bm` timer: WAV loading, inference,
clustering and RTTM output are included; downloads, data copy and DER scoring
are excluded. The current speakrs rows run through the library batch API
(`run_batch_stream`), with WAV files decoded on worker threads ahead of it; it
clusters each finished file while the GPU runs the next. Native speakrs timing
excludes model and pipeline construction.
Pyannote timing includes the complete Python subprocess, including pipeline
construction. Pyannote uses batch size 32.

Only the AMI IHM and Earnings-21 pyannote CUDA rows are from earlier runs on
the hardware listed for those tables. Their times are not a same-hardware
comparison.

| Name | Description |
|------|-------------|
| pyannote CUDA | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) on the NVIDIA GPU listed in its row |
| speakrs CUDA | Native CUDA with speakrs kernels, 1s step |
| speakrs CUDA Fast | Native CUDA with speakrs kernels, 2s step |

### VoxConverse Dev (216 files, 1217.8 min)

Hardware: NVIDIA RTX 4090, 16 CPU cores (model not reported), 8 GiB RAM;
driver 580.126.18, nvidia-smi CUDA 13.0.
The speakrs rows are the median of 3 runs in one container. pyannote CUDA ran
earlier in a container with the same configuration.

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **7.0%** | 2.3% | 2.3% | 2.4% | 40.9s | 1786x |
| speakrs CUDA Fast | 7.4% | 2.3% | 2.3% | 2.8% | 22.0s | **3325x** |
| pyannote CUDA | 7.2% | 2.3% | 2.3% | 2.6% | 2301.3s | 32x |

### VoxConverse Test (232 files, 2612.2 min)

Hardware: NVIDIA RTX 4090, 16 CPU cores (model not reported), 8 GiB RAM.
The speakrs rows are one run each, with driver 595.99.02 (nvidia-smi CUDA 13.2).
pyannote CUDA ran earlier in a container with driver 580.126.18 (nvidia-smi
CUDA 13.0) and otherwise the same configuration.

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **11.1%** | 3.4% | 4.1% | 3.7% | 81.6s | 1921x |
| speakrs CUDA Fast | 11.2% | 3.3% | 4.1% | 3.8% | 42.1s | **3722x** |
| **pyannote CUDA** | **11.1%** | 3.4% | 4.1% | 3.7% | 6341.3s | 25x |

### AMI IHM (34 files, 1123.8 min)

Earlier pyannote hardware: NVIDIA L40S, AMD EPYC 9354.

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **17.0%** | 8.1% | 4.3% | 4.5% | 34.5s | 1955x |
| speakrs CUDA Fast | 17.4% | 8.2% | 4.3% | 4.9% | 17.8s | **3796x** |
| **pyannote CUDA (L40S, earlier run)** | **17.0%** | 8.1% | 4.3% | 4.5% | 4388.1s | 15x |

### AMI SDM (34 files, 1123.8 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **19.7%** | 9.6% | 4.3% | 5.7% | 36.0s | 1874x |
| speakrs CUDA Fast | 20.6% | 9.6% | 4.3% | 6.6% | 18.4s | **3665x** |

### AISHELL-4 (20 files, 763.5 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **11.1%** | 3.9% | 3.9% | 3.3% | 23.9s | 1916x |
| speakrs CUDA Fast | 11.4% | 3.9% | 3.9% | 3.6% | 12.4s | **3706x** |

### Earnings-21 (44 files, 2355.8 min)

Earlier pyannote hardware: NVIDIA RTX 4090, AMD EPYC 7B13.

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| speakrs CUDA | 9.7% | 2.6% | 2.4% | 4.7% | 75.8s | 1864x |
| **speakrs CUDA Fast** | **9.2%** | 2.5% | 2.5% | 4.2% | 39.5s | **3578x** |
| pyannote CUDA (RTX 4090, earlier run) | 9.7% | 2.6% | 2.4% | 4.7% | 8036.8s | 18x |

### ICSI (75 files, 4301.2 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **33.3%** | 19.5% | 9.5% | 4.3% | 142.2s | 1814x |
| speakrs CUDA Fast | 33.7% | 19.4% | 9.6% | 4.7% | 71.1s | **3628x** |

### AVA-AVD (54 files, 266.1 min)

| Implementation | DER | Missed | False Alarm | Confusion | Time | RTFx |
|---|---|---|---|---|---|---|
| **speakrs CUDA** | **45.4%** | 16.1% | 10.8% | 18.6% | 7.7s | 2083x |
| speakrs CUDA Fast | 48.9% | 15.9% | 11.3% | 21.8% | 4.1s | **3899x** |

## Other implementations

[pyannote-rs](https://github.com/thewh1teagle/pyannote-rs) was tested on a 39-file subset and scored 89-92% DER across VoxConverse Dev and Test. See [Why not pyannote-rs?](../README.md#why-not-pyannote-rs) for details.

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
cargo xtask benchmark run --dataset voxconverse-dev --impls pmps,scm,scmf,sk

# Linux (CUDA) -- via dstack
cargo xtask dstack bp my-bench --dataset voxconverse-dev,ami-ihm --impls sg,sgf,pg

# list available implementations and datasets
cargo xtask benchmark run --impls list
cargo xtask benchmark run --dataset list
```

To recalculate a completed run from its recorded reference and hypothesis
RTTM files, run `cargo xtask benchmark score <run-directory>`. This does not
run inference. It atomically replaces `<run-directory>/score.json` on each
repeat and keeps the recorded results and RTTM files unchanged. Stored scoring
uses collar `0ms` and includes overlap; other scoring options are rejected.

Stored benchmark records use schema version 3. The schema version 2 flat
`results.json` writer format remains readable through explicit conversion to a
version 3 record because the typed record shape is not wire-compatible with
the old flat shape. New benchmark runs and score reports use version 3.
