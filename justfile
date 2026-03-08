fmt:
    cargo fmt
    uv run --group dev ruff format scripts fixtures

clippy:
    cargo clippy -- -D warnings
    uv run --group dev ty check --python .venv scripts fixtures

test *args:
    cargo test {{args}}

check: fmt clippy test

# Download ONNX models and PLDA params, then build native CoreML bundles on macOS
download-models:
    #!/usr/bin/env bash
    set -euo pipefail
    uv run scripts/download_models.py fixtures/models
    if [[ "$(uname -s)" == "Darwin" ]]; then
        uv run --project scripts/native_coreml python scripts/native_coreml/convert_coreml.py --output-dir fixtures/models
    fi

download-models-coreml:
    uv run --project scripts/native_coreml python scripts/native_coreml/convert_coreml.py --output-dir fixtures/models

compare-models-coreml:
    uv run --project scripts/native_coreml python scripts/native_coreml/compare_coreml.py --output-dir fixtures/models

# Run fixtures/generate.py to regenerate test fixtures
generate-fixtures:
    uv run fixtures/generate.py

# Prepare local or remote audio and run both pipelines side-by-side
compare source python_device="cpu" rust_mode="exact":
    #!/usr/bin/env bash
    set -euo pipefail
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    wav="$tmp/audio.wav"
    source="{{source}}"
    echo "=== Preparing audio ==="
    if [[ "$source" =~ ^https?:// ]]; then
        if [[ "$source" =~ ^https?://(www\\.)?(youtube\\.com|youtu\\.be)/ ]]; then
            yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
                -o "$tmp/audio.%(ext)s" "$source"
        else
            curl --fail --location --silent --show-error "$source" -o "$tmp/input"
            ffmpeg -y -i "$tmp/input" -ar 16000 -ac 1 "$wav" >/dev/null 2>&1
        fi
    else
        if [[ ! -f "$source" ]]; then
            echo "Input does not exist: $source" >&2
            exit 1
        fi
        ffmpeg -y -i "$source" -ar 16000 -ac 1 "$wav" >/dev/null 2>&1
    fi
    rust_features=()
    if [[ "{{rust_mode}}" == "coreml" ]]; then
        if [[ "$(uname -s)" == "Darwin" ]]; then
            rust_features=(--features coreml)
        elif [[ "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]] || [[ "${OS:-}" == "Windows_NT" ]]; then
            rust_features=(--features directml)
        fi
    elif [[ "{{rust_mode}}" == "native-coreml" ]]; then
        rust_features=(--features native-coreml)
    elif [[ "{{rust_mode}}" == "cuda" ]]; then
        rust_features=(--features cuda)
    fi
    echo ""
    echo "=== speakrs (Rust) ==="
    cargo run --release "${rust_features[@]}" --bin diarize -- --mode "{{rust_mode}}" "$wav" | tee "$tmp/rust.rttm"
    echo ""
    echo "=== pyannote (Python) ==="
    uv run scripts/diarize_pyannote.py --device "{{python_device}}" "$wav" | tee "$tmp/python.rttm"
    echo ""
    echo "=== Comparison ==="
    cargo run --release --bin compare_rttm -- "$tmp/rust.rttm" "$tmp/python.rttm"

# Benchmark Rust and pyannote diarization on the same prepared WAV
benchmark source python_device="auto" runs="1" warmups="1" rust_mode="exact":
    #!/usr/bin/env bash
    set -euo pipefail
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    wav="$tmp/audio.wav"
    source="{{source}}"
    echo "=== Preparing audio ==="
    if [[ "$source" =~ ^https?:// ]]; then
        if [[ "$source" =~ ^https?://(www\\.)?(youtube\\.com|youtu\\.be)/ ]]; then
            yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
                -o "$tmp/audio.%(ext)s" "$source"
        else
            curl --fail --location --silent --show-error "$source" -o "$tmp/input"
            ffmpeg -y -i "$tmp/input" -ar 16000 -ac 1 "$wav" >/dev/null 2>&1
        fi
    else
        if [[ ! -f "$source" ]]; then
            echo "Input does not exist: $source" >&2
            exit 1
        fi
        ffmpeg -y -i "$source" -ar 16000 -ac 1 "$wav" >/dev/null 2>&1
    fi
    rust_features=()
    if [[ "{{rust_mode}}" == "coreml" ]]; then
        if [[ "$(uname -s)" == "Darwin" ]]; then
            rust_features=(--features coreml)
        elif [[ "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]] || [[ "${OS:-}" == "Windows_NT" ]]; then
            rust_features=(--features directml)
        fi
    elif [[ "{{rust_mode}}" == "native-coreml" ]]; then
        rust_features=(--features native-coreml)
    elif [[ "{{rust_mode}}" == "cuda" ]]; then
        rust_features=(--features cuda)
    fi
    echo ""
    echo "=== Building Rust binary ==="
    cargo build --release "${rust_features[@]}" --bin diarize
    echo ""
    echo "=== Benchmark ==="
    uv run scripts/benchmark_diarization.py "$wav" \
        --rust-binary target/release/diarize \
        --rust-mode "{{rust_mode}}" \
        --python-script scripts/diarize_pyannote.py \
        --python-device "{{python_device}}" \
        --runs "{{runs}}" \
        --warmups "{{warmups}}"

# Compare speakrs and FluidAudio against Python pyannote CPU on the same prepared WAV
compare-apple-accuracy source rust_mode="pyannote-mps":
    #!/usr/bin/env bash
    set -euo pipefail
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    wav="$tmp/audio.wav"
    source="{{source}}"
    fluidaudio_path="${FLUIDAUDIO_PATH:-$HOME/.cache/cmd/repos/github.com/FluidInference/FluidAudio}"
    echo "=== Preparing audio ==="
    if [[ "$source" =~ ^https?:// ]]; then
        if [[ "$source" =~ ^https?://(www\\.)?(youtube\\.com|youtu\\.be)/ ]]; then
            yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
                -o "$tmp/audio.%(ext)s" "$source"
        else
            curl --fail --location --silent --show-error "$source" -o "$tmp/input"
            ffmpeg -y -i "$tmp/input" -ar 16000 -ac 1 "$wav" >/dev/null 2>&1
        fi
    else
        if [[ ! -f "$source" ]]; then
            echo "Input does not exist: $source" >&2
            exit 1
        fi
        ffmpeg -y -i "$source" -ar 16000 -ac 1 "$wav" >/dev/null 2>&1
    fi
    rust_features=()
    case "{{rust_mode}}" in
        coreml)
            if [[ "$(uname -s)" == "Darwin" ]]; then
                rust_features=(--features coreml)
            elif [[ "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]] || [[ "${OS:-}" == "Windows_NT" ]]; then
                rust_features=(--features directml)
            fi
            ;;
        native-coreml)
            rust_features=(--features native-coreml)
            ;;
        cuda)
            rust_features=(--features cuda)
            ;;
    esac
    echo ""
    echo "=== speakrs (Rust) ==="
    cargo run --release "${rust_features[@]}" --bin diarize -- --mode "{{rust_mode}}" "$wav" | tee "$tmp/rust.rttm"
    echo ""
    echo "=== pyannote (Python CPU) ==="
    uv run scripts/diarize_pyannote.py --device cpu --output "$tmp/python_cpu.rttm" "$wav"
    cat "$tmp/python_cpu.rttm"
    echo ""
    echo "=== FluidAudio ==="
    swift run --package-path "$fluidaudio_path" fluidaudiocli process "$wav" --mode offline --output "$tmp/fluidaudio.json"
    uv run scripts/fluidaudio_json_to_rttm.py "$tmp/fluidaudio.json" --output "$tmp/fluidaudio.rttm"
    cat "$tmp/fluidaudio.rttm"
    echo ""
    echo "=== speakrs vs pyannote CPU ==="
    cargo run --release --bin compare_rttm -- "$tmp/rust.rttm" "$tmp/python_cpu.rttm"
    echo ""
    echo "=== FluidAudio vs pyannote CPU ==="
    cargo run --release --bin compare_rttm -- "$tmp/fluidaudio.rttm" "$tmp/python_cpu.rttm"
