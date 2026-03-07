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
