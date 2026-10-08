#!/usr/bin/env bash
set -euo pipefail

check_cpu_dependencies() {
    local package="$1"
    local features="$2"
    local dependencies
    dependencies=$(cargo tree --locked -p "$package" --no-default-features \
        --features "$features" --edges normal,build --prefix none)

    if printf '%s\n' "$dependencies" | awk '
        $1 == "ort" || $1 == "ort-sys" { found = 1 }
        END { exit !found }
    '; then
        printf '%s features "%s" include ONNX Runtime dependencies\n' "$package" "$features" >&2
        return 1
    fi

    printf '%s features "%s" have no ONNX Runtime dependencies\n' "$package" "$features"
}

check_cpu_dependencies speakrs cpu
check_cpu_dependencies speakrs 'online cpu'
check_cpu_dependencies xtask cpu
