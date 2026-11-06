#!/bin/sh
# Install the speakrs command from GitHub Releases.
#
# One-liner (Linux x86_64 or arm64, or Apple silicon on macOS):
#   curl -LSfs https://github.com/avencera/speakrs/releases/latest/download/install.sh | sh
#
# Asset names match .github/workflows/release.yml:
#   speakrs-<tag>-x86_64-unknown-linux-gnu.tar.gz
#   speakrs-<tag>-aarch64-unknown-linux-gnu.tar.gz
#   speakrs-<tag>-aarch64-apple-darwin.tar.gz
#
# Modified from avencera/sam3rs, avencera/homebased and avencera/rustywind, which
# started from https://github.com/japaric/trust.

DEFAULT_GIT="avencera/speakrs"
BIN="speakrs"

help() {
    cat <<EOF
Install a binary release of the speakrs command from GitHub.

Usage:
    install.sh [options]

    curl -LSfs https://github.com/${DEFAULT_GIT}/releases/latest/download/install.sh | sh

Options:
    -h, --help      Display this message
    --git SLUG      GitHub repository (default ${DEFAULT_GIT})
    --tag TAG       Release tag (default latest)
    --to LOCATION   Install directory (default ~/.local/bin)
    --force         Replace a speakrs there that this script didn't install,
                    such as the Python package's command
EOF
}

say() {
    echo "install.sh: $1" >&2
}

err() {
    if [ -n "${td:-}" ]; then
        rm -rf "$td"
    fi
    if [ -n "${staged:-}" ]; then
        rm -f "$staged"
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

# the x86_64 binary opens libcuda with dlopen, which a static musl build can't do, so
# Linux ships glibc; Intel Macs and Windows have no release binary
target_triple() {
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64 | Linux/amd64)
            printf 'x86_64-unknown-linux-gnu\n'
            ;;
        Linux/aarch64 | Linux/arm64)
            printf 'aarch64-unknown-linux-gnu\n'
            ;;
        Darwin/arm64)
            printf 'aarch64-apple-darwin\n'
            ;;
        Darwin/x86_64)
            # a shell under Rosetta reports x86_64 on Apple silicon
            if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then
                printf 'aarch64-apple-darwin\n'
            else
                return 1
            fi
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
force=false

while test $# -gt 0; do
    case $1 in
        --git)
            need_value "$1" "${2:-}"
            git=$2
            shift
            ;;
        --force)
            force=true
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
need mv
need rm
need dd
need uname
need awk
if ! command -v sha256sum > /dev/null 2>&1 && ! command -v shasum > /dev/null 2>&1; then
    err "need sha256sum or shasum (command not found)"
fi

if [ -z "${HOME:-}" ]; then
    err "HOME is not set"
fi

target=$(target_triple) \
    || err "no release binary for $(uname -s)/$(uname -m); the release covers Linux x86_64 and arm64, and Apple silicon. Elsewhere, build with: cargo install --git https://github.com/$DEFAULT_GIT speakrs-cli"

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

td=$(mktemp -d 2>/dev/null || mktemp -d -t speakrs-install) \
    || err "failed to create a temporary directory"
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

# `uv tool install speakrs` and pip put the Python package's command at the same path,
# as a symlink or a script, and replacing it would break the user's Python install
if [ -L "$dest/$BIN" ] || [ "$(dd if="$dest/$BIN" bs=2 count=1 2>/dev/null)" = "#!" ]; then
    if [ "$force" != true ]; then
        err "$dest/$BIN is another speakrs install, such as the Python package's command; pass --to DIR to install elsewhere, or --force to replace it"
    fi
fi

say "Installing to: $dest"
mkdir -p "$dest" || err "failed to create $dest"
# the copy goes beside the target and then renames over it, so a failed copy keeps the
# old command, and a running binary can be replaced
staged="$dest/.$BIN.install.$$"
install -m 0755 "$td/$BIN" "$staged" || err "failed to install $BIN to $dest"
mv -f "$staged" "$dest/$BIN" || err "failed to install $BIN to $dest"
staged=""

rm -rf "$td"
td=""

say "installed $BIN $tag to $dest/$BIN"

run=$BIN
case ":$PATH:" in
    *:"$dest":*) ;;
    *)
        say "warning: $dest is not on PATH"
        run=$dest/$BIN
        ;;
esac

# the CUDA modes load the driver at run time, so only the x86_64 build looks for it, and
# a missing one fails only when a CUDA mode runs
if [ "$target" = x86_64-unknown-linux-gnu ] && ! has_library libcuda.so.1; then
    say "libcuda.so.1 not found; the cuda and cuda-fast modes need an NVIDIA driver, cpu works without one"
fi

say "run: $run diarize meeting.wav > meeting.rttm"
