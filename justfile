fmt:
    cargo fmt --all
    uv run --group dev ruff format scripts fixtures

clippy:
    cargo clippy --all --all-targets --workspace -- -D warnings
    uv run --group dev ty check --python .venv --exclude 'scripts/pyannote_rs_bench/target' scripts fixtures

test *args:
    cargo test --workspace {{args}}

check: fmt clippy test

# Models
deploy-models:
    cargo xtask models deploy

export-models:
    cargo xtask models export

export-models-coreml:
    cargo xtask models export-coreml

compare-models-coreml:
    cargo xtask models compare-coreml

# Fixtures
generate-fixtures:
    cargo xtask fixtures generate

# Compare
compare source python_device="cpu" rust_mode="cpu":
    cargo xtask compare run {{source}} --python-device {{python_device}} --rust-mode {{rust_mode}}

compare-apple-accuracy source rust_mode="pyannote-mps":
    cargo xtask compare accuracy {{source}} --rust-mode {{rust_mode}}

# Benchmark
benchmark source python_device="auto" runs="1" warmups="1" rust_mode="cpu":
    cargo xtask benchmark run {{source}} --python-device {{python_device}} --runs {{runs}} --warmups {{warmups}} --rust-mode {{rust_mode}}

benchmark-compare source runs="1" warmups="1":
    cargo xtask benchmark compare {{source}} --runs {{runs}} --warmups {{warmups}}

benchmark-der max_files="10" max_minutes="30" *args="":
    cargo xtask benchmark der --max-files {{max_files}} --max-minutes {{max_minutes}} {{args}}
