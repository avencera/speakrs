#!/usr/bin/env bash
# Release speakrs VERSION from a clean master checkout: tag it, wait for the
# package workflow, upload its wheels and sdist to TestPyPI and check them, then
# publish the crates, upload to PyPI, and create the GitHub release
#
#   TESTPYPI_TOKEN=... PYPI_TOKEN=... scripts/release.sh 0.7.0
#
# Each step skips work that is already done, so a failed run can be rerun
set -euo pipefail

usage() {
    echo "usage: TESTPYPI_TOKEN=... PYPI_TOKEN=... $0 VERSION" >&2
    exit 2
}

[ "$#" -eq 1 ] || usage
version="$1"
tag="v$version"
: "${TESTPYPI_TOKEN:?set TESTPYPI_TOKEN}"
: "${PYPI_TOKEN:?set PYPI_TOKEN}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

log() {
    printf '\n==> %s\n' "$*"
}

fail() {
    echo "error: $*" >&2
    exit 1
}

py() {
    uv run --no-project --quiet --python 3.12 python -I "$@"
}

check_checkout() {
    log "Checking the checkout"
    for tool in git gh cargo uv; do
        command -v "$tool" >/dev/null || fail "$tool is not installed"
    done

    [ -z "$(git status --porcelain)" ] || fail "the working tree has changes"
    [ "$(git branch --show-current)" = master ] || fail "not on master"
    git fetch --quiet origin master --tags
    [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/master)" ] ||
        fail "HEAD is not origin/master; run git pull --ff-only"

    local manifest_version
    manifest_version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)"
    [ "$manifest_version" = "$version" ] ||
        fail "Cargo.toml has version $manifest_version, not $version"
    grep -q "^## \[$version\] - " CHANGELOG.md || fail "CHANGELOG.md has no $version section"
}

push_tag() {
    local head
    head="$(git rev-parse HEAD)"

    # a rerun reuses the tag, but only when it already points at this commit
    if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
        [ "$(git rev-parse "$tag^{commit}")" = "$head" ] || fail "$tag exists on another commit"
        log "Tag $tag already exists at HEAD"
    else
        log "Tagging $tag"
        git tag -a "$tag" -m "speakrs $version"
    fi

    git push --quiet origin "refs/tags/$tag"
}

find_run() {
    local head run_id
    head="$(git rev-parse HEAD)"

    # the tag's run can take a few seconds to show up in the API
    for _ in $(seq 1 30); do
        run_id="$(gh run list --workflow package.yml --event push --commit "$head" \
            --json databaseId,headBranch \
            -q "[.[] | select(.headBranch == \"$tag\")][0].databaseId // empty")"
        if [ -n "$run_id" ]; then
            echo "$run_id"
            return
        fi
        sleep 10
    done

    fail "no package workflow run for $tag after 5 minutes"
}

download_files() {
    local run_id="$1"

    log "Waiting for package workflow run $run_id"
    gh run watch "$run_id" --exit-status

    log "Downloading wheels and sdist"
    gh run download "$run_id" --pattern 'python-*' --dir "$work/artifacts"
    mkdir "$work/dist"
    find "$work/artifacts" -type f \( -name '*.whl' -o -name '*.tar.gz' \) \
        -exec mv {} "$work/dist/" \;
    ls -la "$work/dist"

    [ "$(find "$work/dist" -type f | wc -l | tr -d ' ')" -eq 3 ] ||
        fail "expected two wheels and an sdist"
    for file in "$work/dist"/*; do
        case "$(basename "$file")" in
            "speakrs-$version"-*.whl | "speakrs-$version.tar.gz") ;;
            *) fail "$file does not match $tag" ;;
        esac
    done

    uvx --quiet twine check --strict "$work/dist"/*
}

# compares the index's SHA-256 digests with the local files, so the index serves
# exactly the files the workflow tested on each platform
check_index_files() {
    local json_url="$1"

    py - "$json_url" "$work/dist" <<'EOF'
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

url, dist = sys.argv[1], pathlib.Path(sys.argv[2])
local = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dist.iterdir()}

# a new release can take a few minutes to appear in the JSON API
for attempt in range(20):
    try:
        with urllib.request.urlopen(url) as response:
            remote = {f["filename"]: f["digests"]["sha256"] for f in json.load(response)["urls"]}
        if remote == local:
            print(f"{url} serves the {len(local)} uploaded files")
            sys.exit(0)
    except OSError:
        remote = None
    time.sleep(15)

sys.exit(f"{url} files do not match: expected {sorted(local)}, got {remote}")
EOF
}

# installs the published wheel, never the sdist, into a fresh environment and diarizes
check_install() {
    local index="$1"
    local rttm

    for _ in $(seq 1 10); do
        if rttm="$(uvx --no-cache --python 3.12 --no-build-package speakrs \
            --default-index "$index" "speakrs@$version" \
            diarize --mode cpu fixtures/test_short.wav)"; then
            printf '%s\n' "$rttm"
            grep -q '^SPEAKER ' <<<"$rttm" || fail "diarize printed no SPEAKER lines"
            return
        fi
        sleep 30
    done

    fail "could not install speakrs $version from $index"
}

upload() {
    local name="$1" publish_url="$2" check_url="$3" token="$4"

    log "Uploading to $name"
    # --check-url skips files the index already has, so reruns don't fail on them
    UV_PUBLISH_TOKEN="$token" uv publish --publish-url "$publish_url" \
        --check-url "$check_url" "$work/dist"/*
}

publish_crates() {
    local crate
    for crate in speakrs-ptx-sm75 speakrs-ptx-sm80 speakrs-ptx-sm120 speakrs; do
        if curl -fsS -A speakrs-release \
            "https://crates.io/api/v1/crates/$crate/$version" >/dev/null 2>&1; then
            log "$crate $version is already on crates.io"
            continue
        fi

        log "Publishing $crate $version to crates.io"
        # speakrs has no default backend, so verification needs one enabled
        if [ "$crate" = speakrs ]; then
            cargo publish --locked -p "$crate" --features cpu
        else
            cargo publish --locked -p "$crate"
        fi
    done
}

create_release() {
    if gh release view "$tag" >/dev/null 2>&1; then
        log "GitHub release $tag already exists"
        return
    fi

    log "Creating GitHub release $tag"
    awk -v heading="## [$version]" '
        index($0, heading) == 1 { found = 1; next }
        /^## \[/ { found = 0 }
        found
    ' CHANGELOG.md >"$work/notes.md"
    gh release create "$tag" --title "$tag" --notes-file "$work/notes.md" --verify-tag
}

check_checkout
push_tag
run_id="$(find_run)"
download_files "$run_id"

upload TestPyPI https://test.pypi.org/legacy/ https://test.pypi.org/simple/ "$TESTPYPI_TOKEN"
check_index_files "https://test.pypi.org/pypi/speakrs/$version/json"
log "Installing from TestPyPI"
check_install https://test.pypi.org/simple/

printf '\nTestPyPI checks passed. Publish speakrs %s to crates.io and PyPI? [y/N] ' "$version"
read -r answer
[ "$answer" = y ] || fail "stopped before publishing"

publish_crates
upload PyPI https://upload.pypi.org/legacy/ https://pypi.org/simple/ "$PYPI_TOKEN"
check_index_files "https://pypi.org/pypi/speakrs/$version/json"
log "Installing from PyPI"
check_install https://pypi.org/simple/
create_release

log "Released speakrs $version"
