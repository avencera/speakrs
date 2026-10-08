#!/usr/bin/env bash
# build private alternate PTX and keep the normal embedded artifacts unchanged
set -euo pipefail
out=${1:?usage: build.sh <artifact-directory>}
root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$root"
mkdir -p "$out"
out=$(cd "$out" && pwd)
backup=$(mktemp -d)
cp -a src/inference/cuda/ptx "$backup/ptx"
restore() {
    cp -a "$backup/ptx/." src/inference/cuda/ptx/
    rm -rf "$backup"
}
trap restore EXIT
nvcc --ptx -arch=compute_80 -O3 --fmad=false scripts/cuda/fp16emu/probe.cu -o "$out/probe.ptx"
SPEAKRS_FP16_EMU_BUILD=1 cargo xtask cuda-kernels build resnet wideconv
for area in resnet wideconv; do
    source_ptx=src/inference/cuda/ptx/$area.sm80.ptx
    rg -q 'cvt.rn.f16.f32' "$source_ptx"
    cp "$source_ptx" "$out/$area.sm80.ptx"
    printf '\n// SPEAKRS_FP16_EMU_SCALE=1024\n' >> "$out/$area.sm80.ptx"
done
