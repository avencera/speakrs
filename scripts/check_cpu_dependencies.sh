#!/usr/bin/env bash
set -euo pipefail

# CPU and CoreML builds must not compile ONNX Runtime or the CUDA host stack
check_backend_dependencies() {
    local package="$1"
    local features="$2"
    local target="${3:-}"
    local dependencies
    dependencies=$(cargo tree --locked -p "$package" --no-default-features \
        --features "$features" ${target:+--target "$target"} --edges normal,build --prefix none)

    local found
    found=$(printf '%s\n' "$dependencies" | awk '
        $1 == "ort" || $1 == "ort-sys" || $1 == "cudarc" || $1 == "cudakit" { print $1 }
    ' | sort -u | tr '\n' ' ')
    if [[ -n "$found" ]]; then
        printf '%s features "%s" include %s\n' "$package" "$features" "$found" >&2
        return 1
    fi

    printf '%s features "%s" have no ONNX Runtime or CUDA dependencies\n' "$package" "$features"
}

check_backend_dependencies speakrs cpu
check_backend_dependencies speakrs 'online cpu'
# resolved for macOS, where the CoreML backend builds
check_backend_dependencies speakrs coreml aarch64-apple-darwin
check_backend_dependencies speakrs 'online coreml' aarch64-apple-darwin
check_backend_dependencies xtask cpu
