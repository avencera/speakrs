#!/bin/sh
# Install the speakrs command, the CUDA tuner, from GitHub Releases.
#
# One-liner (Linux x86_64):
#   curl -LSfs https://github.com/avencera/speakrs/releases/latest/download/install.sh | sh
#
# Asset names match .github/workflows/release.yml:
#   speakrs-<tag>-x86_64-unknown-linux-gnu.tar.gz
#
# Modified from avencera/sam3rs, avencera/homebased and avencera/rustywind, which
# started from https://github.com/japaric/trust.

DEFAULT_GIT="avencera/speakrs"
BIN="speakrs"

help() {
    cat <<EOF
Install a binary release of the speakrs CUDA tuner from GitHub.

Usage:
    install.sh [options]

    curl -LSfs https://github.com/${DEFAULT_GIT}/releases/latest/download/install.sh | sh

Options:
    -h, --help      Display this message
    --git SLUG      GitHub repository (default ${DEFAULT_GIT})
    --tag TAG       Release tag (default latest)
    --to LOCATION   Install directory (default ~/.local/bin)
EOF
}

say() {
    echo "install.sh: $1" >&2
}

err() {
    if [ -n "${td:-}" ]; then
        rm -rf "$td"
    fi

    say "ERROR $1"
    exit 1
}

need() {
    if ! command -v "$1" > /dev/null 2>&1; then
        err "need $1 (command not found)"
    fi
}

need_value() {
    case ${2:-} in
        "" | --*)
            err "$1 requires a value"
            ;;
    esac
}

# expands a literal leading ~, as in --to=~/bin; the tildes are escaped because an
# unescaped one in a case pattern expands to $HOME and matches every path under it
expand_dest() {
    dest=$1
    case "$dest" in
        \~)
            dest=$HOME
            ;;
        \~/*)
            dest=$HOME/${dest#\~/}
            ;;
    esac
    printf '%s\n' "$dest"
}

# the binary tunes the CUDA backend, which runs on Linux only, and opens libcuda with
# dlopen, which a static musl build can't do, so the release ships glibc on x86_64
target_triple() {
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64 | Linux/amd64)
            printf 'x86_64-unknown-linux-gnu\n'
            ;;
        *)
            return 1
            ;;
    esac
}

# sha256sum is missing on older macOS, where shasum prints the same "<hash>  <file>" line
sha256() {
    if command -v sha256sum > /dev/null 2>&1; then
        sha256sum "$1"
    else
        shasum -a 256 "$1"
    fi
}

# true when the dynamic loader can find the library: LD_LIBRARY_PATH first, then the
# ldconfig cache, the same order speakrs's dlopen uses
has_library() {
    old_ifs=$IFS
    IFS=:
    for dir in ${LD_LIBRARY_PATH:-}; do
        if [ -n "$dir" ] && [ -e "$dir/$1" ]; then
            IFS=$old_ifs
            return 0
        fi
    done
    IFS=$old_ifs

    for ldconfig in ldconfig /sbin/ldconfig /usr/sbin/ldconfig; do
        if command -v "$ldconfig" > /dev/null 2>&1; then
            "$ldconfig" -p 2>/dev/null | grep -q "$1 " && return 0
            return 1
        fi
    done
    return 1
}

git=$DEFAULT_GIT
tag=""
dest=""

while test $# -gt 0; do
    case $1 in
        --git)
            need_value "$1" "${2:-}"
            git=$2
            shift
            ;;
        --help | -h)
            help
            exit 0
            ;;
        --tag)
            need_value "$1" "${2:-}"
            tag=$2
            shift
            ;;
        --to)
            need_value "$1" "${2:-}"
            dest=$2
            shift
            ;;
        *)
            err "unknown option: $1"
            ;;
    esac
    shift
done

need curl
need tar
need mkdir
need mktemp
need install
need rm
need uname
need awk
if ! command -v sha256sum > /dev/null 2>&1 && ! command -v shasum > /dev/null 2>&1; then
    err "need sha256sum or shasum (command not found)"
fi

if [ -z "${HOME:-}" ]; then
    err "HOME is not set"
fi

target=$(target_triple) \
    || err "no release binary for $(uname -s)/$(uname -m); the release covers Linux x86_64 only, since the binary tunes the CUDA backend. Elsewhere, build with: cargo install speakrs --no-default-features --features cuda"

if [ -z "$dest" ]; then
    dest=$HOME/.local/bin
fi
dest=$(expand_dest "$dest")

url="https://github.com/$git"
say "GitHub repository: $url"

if [ -z "$tag" ]; then
    latest_url=$(curl -fsSL -o /dev/null -w '%{url_effective}' "$url/releases/latest") \
        || err "failed to resolve the latest release of $git"
    case "$latest_url" in
        */releases/tag/*) ;;
        *)
            err "no GitHub release found for $git"
            ;;
    esac
    tag=${latest_url##*/}
    say "Tag: latest ($tag)"
else
    say "Tag: $tag"
fi
say "Target: $target"

base="$BIN-$tag-$target"
archive="$base.tar.gz"
download="$url/releases/download/$tag/$archive"
say "Downloading: $download"

td=$(mktemp -d 2>/dev/null || mktemp -d -t speakrs-install)
curl -fsSL "$download" -o "$td/$archive" \
    || err "failed to download $download"
curl -fsSL "$url/releases/download/$tag/$base.sha256" -o "$td/$base.sha256" \
    || err "failed to download $base.sha256"

# sha256sum output, one "<hash>  <file>" line per uploaded asset
expected=$(awk -v f="$archive" '$2 == f || $2 == "*" f { print $1 }' "$td/$base.sha256")
actual=$(sha256 "$td/$archive" | cut -d ' ' -f 1)
if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
    err "checksum mismatch for $archive: expected ${expected:-nothing}, got $actual"
fi

tar -C "$td" -xzf "$td/$archive" || err "failed to extract $archive"
if [ ! -f "$td/$BIN" ] || [ ! -x "$td/$BIN" ]; then
    err "$archive does not contain an executable $BIN"
fi

say "Installing to: $dest"
mkdir -p "$dest" || err "failed to create $dest"
# unlink first so a running binary can be replaced
rm -f "$dest/$BIN"
install -m 0755 "$td/$BIN" "$dest/$BIN" || err "failed to install $BIN to $dest"

rm -rf "$td"
td=""

say "installed $BIN $tag to $dest/$BIN"

case ":$PATH:" in
    *:"$dest":*) ;;
    *)
        say "warning: $dest is not on PATH"
        ;;
esac

# the binary loads the driver at run time, so a missing one only fails when tuning starts;
# cuDNN 9 and cuBLAS are optional, since the tuner times them only with --include-library
if ! has_library libcuda.so.1; then
    say "warning: libcuda.so.1 not found; speakrs needs an NVIDIA driver"
fi

say "run: $BIN cuda tune --models-dir /path/to/models"
