#!/usr/bin/env bash
set -euo pipefail

# Linux C compiler for the macOS cross-target clippy checks in `just clippy`
#
# cc-rs passes `--target=x86_64-unknown-linux-gnu`, a Rust triple zig can't parse, so drop it
# and give zig its own triple instead. A script also keeps cc-rs's compiler detection working,
# which runs the configured compiler without the `cc -target` arguments of an inline command
args=()
for arg in "$@"; do
    [[ $arg == --target=* ]] || args+=("$arg")
done

exec zig cc -target x86_64-linux-gnu "${args[@]}"
