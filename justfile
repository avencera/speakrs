platform_features := if os() == "macos" { "--features coreml" } else if os() == "windows" { "--features directml" } else { "" }

fmt:
    cargo fmt
    uv run --group dev ruff format scripts fixtures

clippy:
    cargo clippy -- -D warnings
    uv run --group dev ty check --python .venv scripts fixtures

test *args:
    cargo test {{args}}

check: fmt clippy test

# Download ONNX models and PLDA params (requires HF_TOKEN or huggingface-cli login)
download-models:
    uv run scripts/download_models.py fixtures/models

# Run fixtures/generate.py to regenerate test fixtures
generate-fixtures:
    uv run fixtures/generate.py

# Prepare local or remote audio and run both pipelines side-by-side
compare source:
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
    echo ""
    echo "=== speakrs (Rust) ==="
    cargo run --release {{platform_features}} --bin diarize -- "$wav" | tee "$tmp/rust.rttm"
    echo ""
    echo "=== pyannote (Python) ==="
    uv run scripts/diarize_pyannote.py "$wav" | tee "$tmp/python.rttm"
    echo ""
    echo "=== Comparison ==="
    cargo run --release --bin compare_rttm -- "$tmp/rust.rttm" "$tmp/python.rttm"
