#!/usr/bin/env bash
# prove link and loader isolation on a Linux GPU host; exit 3 means a port is missing
set -euo pipefail
models=${1:?usage: prove-driver-only.sh MODELS_DIR SHORT_WAV [TARGET_FEATURE]}
wav=${2:?short WAV file required}
feature=${3:-cuda-rtx50}
case "$feature" in
    cuda-sm75|cuda-sm80|cuda-sm90|cuda-sm120|cuda-rtx20|cuda-rtx30|cuda-rtx40|cuda-a100|cuda-rtx50) ;;
    *) echo "expected one driver-only target feature" >&2; exit 2 ;;
esac
for tool in cargo ldd nm strace rg flock; do
    command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 2; }
done
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"
evidence=${SPEAKRS_DRIVER_PROOF_OUTPUT:-$repo/_scratch/no-cudnn/driver-only/routing/loader-proof}
mkdir -p "$evidence"
evidence=$(mktemp -d "$evidence/run-$feature.XXXXXX")
echo "Proof evidence: $evidence"
# a separate target directory prevents another feature build from replacing this binary
mkdir -p "$repo/target/driver-only-proof"
export CARGO_TARGET_DIR
CARGO_TARGET_DIR=$(mktemp -d "$repo/target/driver-only-proof/build-$feature.XXXXXX")
unset SPEAKRS_CUDA_PTX_TIER SPEAKRS_CUDA_FORCE_LIBRARY
cargo build --locked -p xtask --bin xtask --no-default-features --features "$feature"
binary="$CARGO_TARGET_DIR/debug/xtask"
ldd "$binary" | tee "$evidence/ldd.txt"
nm -D --undefined-only "$binary" > "$evidence/nm.txt"
if rg -i 'lib(cudnn|cublas|cublasLt)|\b(cudnn|cublas)[A-Z_]' "$evidence/ldd.txt" "$evidence/nm.txt"; then
    echo "FAIL: a numerical library link or undefined symbol is present" >&2
    exit 1
fi
set +e
RUST_LOG=speakrs=info flock "${SPEAKRS_GPU_LOCK:-/workspace/gpu-bench.lock}" strace -f -e trace=openat -o "$evidence/openat.txt" \
    "$binary" diarize --mode cuda --models-dir "$models" "$wav" \
    > "$evidence/run.txt" 2>&1
status=$?
set -e
cat "$evidence/run.txt"
if rg -i 'lib(cudnn|cublas|cublasLt)[^"/ ]*\.so' "$evidence/openat.txt"; then
    echo "FAIL: the process tried to open a numerical library" >&2
    exit 1
fi
if [[ "$status" -eq 0 ]]; then
    echo "PASS: model load and diarization completed with no cuDNN or cuBLAS link or open"
    exit 0
fi
if rg 'driver-only CUDA missing kernel: .+ b[0-9]+ (Fp32|Tf32)' "$evidence/run.txt"; then
    echo "INCOMPLETE: typed missing-kernel error; no numerical library link or open"
    exit 3
fi
echo "FAIL: model load failed for a reason other than a typed missing kernel (exit $status)" >&2
exit 1
