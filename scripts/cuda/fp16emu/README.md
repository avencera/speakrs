# Private FP16 operand experiment

This test is off by default. It changes operand rounding in the existing TF32
trunk kernels. It does not use FP16 MMA instructions. It has no public config
option.

Build the host with the normal embedded artifacts. On the CUDA build box, source
`/workspace/env.sh`, then build separate test artifacts:

```sh
bash scripts/cuda/fp16emu/build.sh /workspace/amb-fp16emu/alternate
```

The script restores the normal embedded PTX, cubins, and manifests when it ends.
Keep the artifact directory separate from `src/inference/cuda/ptx`.

Run variant A with these environment variables:

```sh
export SPEAKRS_FP16_EMU=operands
export SPEAKRS_FP16_EMU_DIR=/workspace/amb-fp16emu/alternate
```

For variant B, set `SPEAKRS_FP16_EMU=stores`. It also rounds the hidden output of
each basic-block conv1. Only conv2 reads those values. The buffers remain FP32;
this test checks storage rounding, not storage size or speed.

Both variants use this conversion for each operand:

```text
FP32 value -> multiply by 1024 -> FP16 round-to-nearest-even
           -> FP32 -> multiply by 1/1024 -> existing TF32 MMA
```

FP32 accumulation, bias, residuals, pooling, head, and segmentation do not change.
The embedding math must be TF32. Alternate artifacts are accepted only for the
sm80 trunk tier. The normal build must not set `SPEAKRS_FP16_EMU_BUILD`.

To collect ranges with the normal TF32 pipeline, unset `SPEAKRS_FP16_EMU` and set
`SPEAKRS_FP16_RANGE_PTX` to the separate `probe.ptx`. This disables embedding graph
capture. Each `FP16_RANGE` log row has these fields:

```text
layer kind max_abs small overflow zero nonfinite total scaled_small min_nonzero
```

`small` counts nonzero values below 2^-14. `scaled_small` counts nonzero values
below 2^-24 before scaling. `overflow` counts values above 65504. The input scan
also checks valid Winograd tiles for same-channel C128/C256 layers. Counts are
per logical tensor, not per repeated MMA load. GPU runs must hold
`flock /workspace/gpu-bench.lock`.
