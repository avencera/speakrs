# CUDA backend details

This page covers how the native CUDA backend picks kernels, how to tune it, and
the settings for debugging. For choosing a build, see
[Features and build notes](../README.md#features-and-build-notes) in the README.

## Requirements

The CUDA backend runs on Linux. It needs an NVIDIA driver and a Turing
(compute capability 7.5) or newer GPU, and it builds without a CUDA toolkit.

- **GPU-specific builds** (`cuda-rtx20`, `cuda-rtx30`, `cuda-rtx40`,
  `cuda-a100`, `cuda-rtx50`, and the tier features `cuda-sm75`, `cuda-sm80`,
  `cuda-sm90`, `cuda-sm120`) load only the driver.
- **The `cuda` build** also uses cuDNN 9 and cuBLAS when a selected plan uses
  them, and NVRTC for the `PersistDynamic` LSTM mode.

The family features are aliases for the kernel tiers: `cuda-rtx20` is
`cuda-sm75`, `cuda-rtx30`, `cuda-rtx40` and `cuda-a100` are `cuda-sm80`, and
`cuda-rtx50` is `cuda-sm120`. Each tier embeds every area's best shipped kernel
variant for that tier.

Features are additive: adding `cuda` also adds cuDNN and cuBLAS. In a
GPU-specific build, a model load returns a typed error naming the layer, batch
size and math mode if a kernel is missing, instead of falling back to a library.

CUDA modes use `segmentation-3.0.safetensors` and
`wespeaker-multimask-tail.safetensors`. `RuntimeConfig` selects precision and
CUDA graphs.

## Kernel selection

The `cuda` build uses speakrs kernels on every GPU, including devices with no
measured recipe. It embeds all kernel tiers and keeps cuDNN and cuBLAS as a
fallback. Each layer is chosen in this order:

**matching user tune file > built-in kernel recipe > device-class or portable kernel default > library fallback**

The library is used only when a kernel refuses a layer because of a capability
limit, its weight contract or an unimplemented geometry, or when no kernel
covers the layer's arithmetic mode. Invalid geometry, CUDA errors and artifact
load errors are returned to the caller. A tune file can explicitly select the
library.

### Built-in recipes

Built-in recipes contain kernel settings only, and match on the exact device
name and SM count.

- **RTX 4060 Ti (cc 8.9, 34 SMs) and RTX 5060 Ti (cc 12.0, 36 SMs):** with FP32
  segmentation and TF32 embedding, these use the driver-only kernel settings at
  embedding batches 1, 4, 8, 16 and 32.
  - The 4060 Ti uses FP16 same-channel trunk kernels for C32/C64 at every batch,
    and for C128/C256 from batch 8.
  - Its FP32 SincNet recipe also applies outside this whole-pipeline precision
    mode.
- **Tesla T4 (cc 7.5, 40 SMs):** uses kernel settings at all pipeline batch
  classes, including FP16 same-channel trunk kernels in TF32 embedding mode. It
  doesn't select the library for small per-layer timing wins.
- **A100 PCIe-40GB and SXM4-40GB (cc 8.0, 108 SMs):** use FP32 segmentation and
  TF32 embedding.
- **RTX 4090:** has a measured recipe.

Other devices use their kernel defaults. Measured speed evidence is reported,
but isn't required to select a kernel. GPU-specific builds stay driver-only and
return typed errors instead of library fallbacks.

### Environment variables

- `SPEAKRS_CUDA_PTX_TIER=sm75` limits the kernel tier.
- `SPEAKRS_CUDA_FORCE_LIBRARY=1` makes a `cuda` build use the library at each
  replaceable layer, even when a tune file exists.
- `SPEAKRS_CUDA_TUNE_FILE=PATH` loads a tune file from a custom path.

The model-load log gives the source for each layer: tune file, recipe, default,
or library.

## Tested GPUs: layers where the libraries still win

On tested GPUs, cuDNN and cuBLAS still win the layers below, each by 2–8 µs.
This doesn't change end-to-end speed.

| GPU | Layer | Batch | speakrs | cuDNN/cuBLAS |
|---|---|---:|---:|---:|
| T4 | `layer4.0.shortcut` | 4 | 149 µs | 141 µs |
| T4 | `linear0` / `linear1` (FP32) | 1 | 18 / 13 µs | 15 / 11 µs |
| A100 | `layer4.0.conv1` | 4 | 59 µs | 57 µs |
| L4, A10, RTX 4090 | `seg_1` | 32 | 31 / 42 / 15 µs | 28 / 37 / 13 µs |

The H100 has no tuned recipe. It's faster than the libraries end to end, but
some individual layers are slower.

## Tuning

Build the opt-in command on Linux:

```sh
cargo build --release --no-default-features --features cuda --bin speakrs
./target/release/speakrs cuda tune --models-dir /path/to/models --dry-run
./target/release/speakrs cuda tune --models-dir /path/to/models
```

### What it measures

The command uses real model weights and shapes, and measures the approved kernel
configurations at the pipeline batch sizes.

- **Library timing:** use `--include-library` to also time the library in a
  `cuda` build. Per-layer times don't include library load or handle
  initialization costs.
- **Method:** warm-up passes, alternating candidate order, and median
  CUDA-event times.
- **Output:** the summary table shows each candidate and marks the selected
  choice.
- **Batch sizes:** the embedding trunk is measured at batches 1, 4, 8, 16 and
  32. Segmentation and embedding-head choices use batches 1 and 32, and the
  filterbank uses batches 1 through 32.

Run tuning when other GPU work is stopped.

### Accuracy rules

Tuning doesn't change accuracy rules or precision.

- **Precision options:** the defaults are FP32 segmentation, FP32 filterbank and
  TF32 embedding. Use `--segmentation-math fp32|tf32` and
  `--embedding-math fp32|tf32` to match another pipeline configuration. Use
  `--device N` to select a device.
- **Eligible kernels:** only fixed pins with reviewed end-to-end accuracy are
  eligible. Timing and edited JSON can't approve other algorithms or select
  arbitrary configuration pins.
- **FP32 mode** accepts direct FP32 algorithms.
- **TF32 mode** also accepts direct TF32 algorithms, staged one-product Winograd,
  C64 FFMA Winograd and FP16 trunk algorithms.
- **FP16:** the tuner lists FP16 and non-FP16 pins as separate choices and
  measures each approved choice. FP16 is eligible only in TF32 mode, including
  on supported devices without an FP16 startup recipe. Recipes remain startup
  defaults, not limits on the tuning choices.
- **GPU-specific builds:** a build such as `--features cuda-rtx40` measures
  kernels only. It returns an error when a complete model layer has no approved
  choice. A `cuda` build can use the library for such a layer only with
  `--include-library`.

### Tune files

The file is saved under `$XDG_CONFIG_HOME/speakrs/`, or
`$HOME/.config/speakrs/`, with a per-device name.

- **Paths:** use `--output PATH` to change that location, then set
  `SPEAKRS_CUDA_TUNE_FILE=PATH` when loading a model. The same variable can set
  the tuning output path.
- **Dry runs:** `--dry-run` measures and prints the table without writing or
  changing a file.
- **Matching:** a file is used only when all of these match:
  - device name, compute capability and SM count
  - NVIDIA driver release
  - cuDNN and cuBLAS versions, or driver-only mode
  - speakrs version
  - embedded artifact digest
  - accuracy-policy version
- **Rejection:** a bad key or row rejects the complete file, and selection then
  uses recipes and defaults. Missing rows use the normal fallback.

There is no automatic tuning at startup. The library API is
`speakrs::inference::cuda::{tune_cuda, CudaTuneOptions}`.
