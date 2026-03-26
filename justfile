fmt:
    cargo fmt --all
    uv run --group dev ruff format scripts fixtures

clippy:
    cargo clippy --all --all-targets --all-features --workspace -- -D warnings

python-lint:
    uv run --group dev ty check --python .venv --exclude 'scripts/pyannote_rs_bench/target' --exclude 'scripts/extract_hf_dataset.py' --exclude 'scripts/speakerkit-bench/Packages' --exclude 'scripts/native_coreml' --exclude 'scripts/pyannote-bench' --exclude 'scripts/convert_fp16.py' scripts fixtures
    uv run --group dev ty check --project scripts/native_coreml --python scripts/native_coreml/.venv scripts/native_coreml
    uv run --group dev ty check --project scripts/pyannote-bench --python scripts/pyannote-bench/.venv scripts/pyannote-bench

lint: clippy python-lint

test *args:
    cargo test --workspace {{args}}

check: fmt lint test

# Bump version: just bump major|minor|patch
bump level:
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[] | select(.name=="speakrs") | .version')
    IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
    case "{{level}}" in
        major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
        minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
        patch) PATCH=$((PATCH + 1)) ;;
        *) echo "Usage: just bump major|minor|patch"; exit 1 ;;
    esac
    NEW="${MAJOR}.${MINOR}.${PATCH}"
    sed -i '' "s/^version = \"${CURRENT}\"/version = \"${NEW}\"/" Cargo.toml
    cargo generate-lockfile --quiet
    echo "Bumped ${CURRENT} → ${NEW}"

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

# GPU image: build via nsc to GHCR, then copy to Docker Hub via skopeo
gpu-image suffix="":
    #!/usr/bin/env bash
    set -euo pipefail
    TAG=$(git rev-parse --short HEAD)
    SUFFIX="{{suffix}}"
    if [ -n "$SUFFIX" ]; then
        TAG="${TAG}-${SUFFIX}"
    fi
    GHCR="ghcr.io/avencera/speakrs-gpu:${TAG}"
    DOCKERHUB="docker.io/avencera/speakrs-gpu:${TAG}"
    nsc build -f Dockerfile.gpu --platform linux/amd64 -t "$GHCR" --push .
    echo "Copying to Docker Hub..."
    skopeo copy "docker://${GHCR}" "docker://${DOCKERHUB}"
    mkdir -p _local
    echo "$TAG" > _local/gpu-image-tag
    sed -i '' "s|image:.*speakrs-gpu:[a-zA-Z0-9._-]*|image: avencera/speakrs-gpu:${TAG}|g" .dstack/*.yml
    echo "Built and pushed: $DOCKERHUB (updated .dstack/*.yml)"

gpu-base-image:
    nsc build -f docker/base.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-gpu-base:latest --push .

gpu-runtime-image:
    nsc build -f docker/runtime.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-gpu-runtime:latest --push .

gpu-models-image:
    nsc build -f docker/models.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-models:latest --push .

gpu-datasets-image:
    nsc build -f docker/datasets.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-datasets:latest --push .
