# the coreml feature builds only on macOS, where clippy checks it next to the cpu backend
backend_features := if os() == "macos" { "coreml cpu" } else { "cpu" }

fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo fmt --all
    uv run --group dev ruff format scripts fixtures
    # the README section between the cargo-rdme markers is generated from the crate docs in
    # src/lib.rs, so edit the docs there; CI fails when the two drift apart
    if command -v cargo-rdme >/dev/null; then
        cargo rdme
    else
        echo "cargo-rdme not found: README not regenerated (see CONTRIBUTING.md: cargo install cargo-rdme --version 2.2.2 --locked)" >&2
    fi

clippy:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo clippy --all --all-targets --workspace --features "{{backend_features}} cuda load-dynamic _metrics" -- -D warnings
    # the CUDA-only build has no ONNX Runtime either
    cargo clippy -p speakrs --all-targets --no-default-features --features "online cuda" -- -D warnings
    for features in cuda-sm75 cuda-sm80 cuda-sm90 cuda-sm120 cuda-rtx20 cuda-rtx30 cuda-rtx40 cuda-a100 cuda-rtx50; do
        cargo clippy -p speakrs --all-targets --no-default-features --features "$features" -- -D warnings
    done
    # the benchmark binary must compile its runners without the numerical libraries,
    # and every check builds without CPU feature unification from the workspace
    for features in cuda cuda-rtx40 cuda-rtx50; do
        cargo clippy -p xtask --all-targets --no-default-features --features "$features" -- -D warnings
    done
    cargo clippy -p speakrs --all-targets --features "cpu load-dynamic" -- -D warnings
    if [[ "$(uname)" == "Darwin" ]]; then
        # the CoreML-only build has no ONNX Runtime, so check it for dead code separately
        cargo clippy -p speakrs --all-targets --no-default-features --features "online coreml" -- -D warnings
        # Linux-only CUDA tests never compile for a macOS host, so lint them for the Linux target
        # zig cross-compiles the C dependencies; `online` is left out because it needs Linux OpenSSL
        if command -v zig >/dev/null; then
            for features in cuda cuda-sm75 cuda-rtx50; do
                CC_x86_64_unknown_linux_gnu="zig cc -target x86_64-linux-gnu" AR_x86_64_unknown_linux_gnu="zig ar" \
                    cargo clippy -p speakrs --all-targets --no-default-features --features "$features" \
                    --target x86_64-unknown-linux-gnu -- -D warnings
            done
        else
            echo "zig not found: skipping the Linux-target CUDA clippy checks" >&2
        fi
    fi

python-lint:
    uv run --group dev ty check --python .venv --exclude 'scripts/pyannote_rs_bench/target' --exclude 'scripts/extract_hf_dataset.py' --exclude 'scripts/speakerkit-bench/Packages' --exclude 'scripts/native_coreml' --exclude 'scripts/pyannote-bench' --exclude 'scripts/convert_fp16.py' scripts fixtures
    uv run --group dev ty check --project scripts/native_coreml --python scripts/native_coreml/.venv scripts/native_coreml
    uv run --group dev ty check --project scripts/pyannote-bench --python scripts/pyannote-bench/.venv scripts/pyannote-bench

lint: clippy python-lint

test *args:
    cargo test --workspace {{args}}
    cargo test -p speakrs --features "cpu load-dynamic" {{args}}
    cargo test -p speakrs --no-default-features --features "online cuda" {{args}}
    cargo test -p speakrs --no-default-features --features "online cuda-rtx50" {{args}}

check-cpu-dependencies:
    bash scripts/check_cpu_dependencies.sh

test-gpuq-workload:
    tests/gpuq-workload.sh

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

# Informal RTTM timeline comparison
compare-rttm a b:
    cargo xtask compare rttm {{a}} {{b}}

# Benchmark (local)
bench-der max_files="10" max_minutes="30" *args="":
    cargo xtask benchmark run --max-files {{max_files}} --max-minutes {{max_minutes}} {{args}}

bench-score run_dir:
    cargo xtask benchmark score {{run_dir}}

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

# CUDA 12.4 linux/amd64 image for the gpuq canary
gpuq-canary-image:
    #!/usr/bin/env bash
    set -euo pipefail
    TAG=$(git rev-parse --short HEAD)
    IMAGE="ghcr.io/avencera/speakrs-gpuq-canary:${TAG}"
    nsc build -f docker/gpuq-canary.Dockerfile --platform linux/amd64 -t "$IMAGE" --push .
    DIGEST=$(skopeo inspect "docker://${IMAGE}" | jq -r .Digest)
    echo "Built and pushed: ${IMAGE}"
    echo "Set gpuq.toml image to ghcr.io/avencera/speakrs-gpuq-canary@${DIGEST}"

gpu-base-image:
    nsc build -f docker/base.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-gpu-base:latest --push .

gpu-runtime-image:
    nsc build -f docker/runtime.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-gpu-runtime:latest --push .

gpu-models-image:
    nsc build -f docker/models.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-models:latest --push .

gpu-datasets-image:
    nsc build -f docker/datasets.Dockerfile --platform linux/amd64 -t ghcr.io/avencera/speakrs-datasets:latest --push .
