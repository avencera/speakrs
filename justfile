fmt:
    cargo fmt --all
    uv run --group dev ruff format scripts fixtures

clippy:
    cargo clippy --all --all-targets --all-features --workspace -- -D warnings
    uv run --group dev ty check --python .venv --exclude 'scripts/pyannote_rs_bench/target' --exclude 'scripts/extract_hf_dataset.py' scripts fixtures

test *args:
    cargo test --workspace {{args}}

check: fmt clippy test

# passthrough to cargo xtask
x *args:
    cargo xtask {{args}}

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

# Benchmark (local)
bench-run source python_device="auto" runs="1" warmups="1" rust_mode="cpu":
    cargo xtask bench run {{source}} --python-device {{python_device}} --runs {{runs}} --warmups {{warmups}} --rust-mode {{rust_mode}}

bench-compare source runs="1" warmups="1":
    cargo xtask bench compare {{source}} --runs {{runs}} --warmups {{warmups}}

bench-der max_files="10" max_minutes="30" *args="":
    cargo xtask bench der --max-files {{max_files}} --max-minutes {{max_minutes}} {{args}}

# GPU image
gpu-image:
    #!/usr/bin/env bash
    set -euo pipefail
    TAG=$(git rev-parse --short HEAD)
    IMAGE="ghcr.io/avencera/speakrs-gpu:${TAG}"
    nsc build -f Dockerfile.gpu --platform linux/amd64 -t "$IMAGE" --push .
    mkdir -p _local
    echo "$TAG" > _local/gpu-image-tag
    echo "Built and pushed: $IMAGE"

gpu-base-image:
    nsc build -f docker/base.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-gpu-base:latest --push .

gpu-runtime-image:
    nsc build -f docker/runtime.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-gpu-runtime:latest --push .

gpu-models-image:
    nsc build -f docker/models.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-models:latest --push .

gpu-datasets-image:
    nsc build -f docker/datasets.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-datasets:latest --push .
