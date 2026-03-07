fmt:
    cargo fmt

clippy:
    cargo clippy -- -D warnings

test *args:
    cargo test {{args}}

check: fmt clippy test

# Download ONNX models and PLDA params (requires HF_TOKEN or huggingface-cli login)
download-models:
    uv run scripts/download_models.py fixtures/models

# Run fixtures/generate.py to regenerate test fixtures
generate-fixtures:
    uv run fixtures/generate.py

# Download YouTube audio and run both pipelines side-by-side
compare url:
    #!/usr/bin/env bash
    set -euo pipefail
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    echo "=== Downloading audio ==="
    yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
        -o "$tmp/audio.%(ext)s" "{{url}}"
    wav="$tmp/audio.wav"
    echo ""
    echo "=== speakrs (Rust) ==="
    cargo run --release --bin diarize -- "$wav"
    echo ""
    echo "=== pyannote (Python) ==="
    uv run scripts/diarize_pyannote.py "$wav"
