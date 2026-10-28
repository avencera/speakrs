#!/usr/bin/env bash
# Prepare a rented Ubuntu 24.04 NVIDIA box for the native CUDA backend work
#
# Run from a synced speakrs tree: `bash scripts/cuda/setup-gpu-box.sh`. cudakit's
# `scripts/setup-gpu-box.sh` installs the kernel toolchain with speakrs's pins and
# probes the speakrs kernel crate; this script then adds speakrs's variables and
# models. The cudakit script comes from `CUDAKIT_DIR` when set, otherwise from the
# cudakit release speakrs depends on
#
# Beyond cudakit's layout it produces:
#   /workspace/models      public CUDA-mode ONNX models, PLDA and metadata files
#   /workspace/ref         reference tensors and weights (filled by the references task)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# the commit cudakit 0.1.0 was published from; the crate package omits its scripts
CUDAKIT_REV="bcb03304b8f0325b2a0e91bd91f5de0ddd0acf4d"
WORKSPACE="/workspace"

log() { printf '\n==> %s\n' "$*"; }

# --- toolchain --------------------------------------------------------------------
if [[ -n "${CUDAKIT_DIR:-}" ]]; then
    CUDAKIT_SETUP="$CUDAKIT_DIR/scripts/setup-gpu-box.sh"
else
    CUDAKIT_SETUP="$(mktemp)"
    trap 'rm -f "$CUDAKIT_SETUP"' EXIT
    curl -fsSL --retry 3 -o "$CUDAKIT_SETUP" \
        "https://raw.githubusercontent.com/avencera/cudakit/${CUDAKIT_REV}/scripts/setup-gpu-box.sh"
fi

# speakrs pins its own cuda-oxide; keep the rev in sync with
# crates/speakrs-cuda-kernels/Cargo.toml
CUDA_OXIDE_REV="918bbde123671153b29a89542f1781f1c7494c14" \
    CUDA_OXIDE_NIGHTLY="nightly-2026-08-28" \
    bash "$CUDAKIT_SETUP" \
    --probe "$REPO_ROOT/crates/speakrs-cuda-kernels" --probe-features probe

# --- speakrs layout ---------------------------------------------------------------
mkdir -p "$WORKSPACE/models" "$WORKSPACE/ref"
if ! grep -q SPEAKRS_CUDA_MODELS "$WORKSPACE/env.sh"; then
    cat >>"$WORKSPACE/env.sh" <<EOF
# added by speakrs scripts/cuda/setup-gpu-box.sh
export SPEAKRS_CUDA_MODELS=${WORKSPACE}/models
# the CUDA modes' safetensors weights, exported on the Mac with
# \`scripts/cuda/export_weights.py --runtime-assets\` and synced here; tests/cuda_pipeline.rs
# reads them
export SPEAKRS_CUDA_ASSETS=${WORKSPACE}/models-native
EOF
fi

# --- models -----------------------------------------------------------------------
HF_REVISION="$(sed -n 's/^pub const MODEL_REVISION: &str = "\([0-9a-f]\{40\}\)";$/\1/p' "$REPO_ROOT/src/models.rs")"
if [[ -z "$HF_REVISION" ]]; then
    echo "could not read MODEL_REVISION from $REPO_ROOT/src/models.rs" >&2
    exit 1
fi

# mirrors `catalog_assets(ExecutionMode::Cuda)` in src/models.rs: PLDA and metadata,
# then segmentation, embedding, fbank and multi-mask tail including the batched variants
MODEL_FILES=(
    plda_lda.npy plda_tr.npy plda_mu.npy plda_psi.npy plda_mean1.npy plda_mean2.npy
    wespeaker-voxceleb-resnet34.min_num_samples.txt
    segmentation-3.0.onnx segmentation-3.0-b32.onnx
    wespeaker-voxceleb-resnet34.onnx wespeaker-voxceleb-resnet34.onnx.data
    wespeaker-voxceleb-resnet34-b64.onnx
    wespeaker-fbank.onnx wespeaker-fbank-b32.onnx
    wespeaker-multimask-tail.onnx wespeaker-multimask-tail-b32.onnx
)

log "models at revision ${HF_REVISION}"
echo "$HF_REVISION" >"$WORKSPACE/models/HF_REVISION"
for file in "${MODEL_FILES[@]}"; do
    target="$WORKSPACE/models/$file"
    if [[ -s "$target" ]]; then
        continue
    fi

    # anonymous download; never pass a token from this box
    curl -fsSL --retry 3 -o "$target.part" \
        "https://huggingface.co/avencera/speakrs-models/resolve/${HF_REVISION}/${file}"
    mv "$target.part" "$target"
done

echo "hf_revision=${HF_REVISION}" | tee -a "$WORKSPACE/PINS"
