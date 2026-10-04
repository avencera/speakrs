"""Keep large owner-locked qualification assets in an outside-tree cache."""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from lock import ROOT, SCOPE, LockError, scope


def cache_directory(root: Path = ROOT) -> Path:
    """Return an outside-tree cache; qualification assets must never enter git."""
    default = (
        Path.home() / "Library/Caches"
        if sys.platform == "darwin"
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    ) / "speakrs-cuda-qualify"
    cache = Path(os.environ.get("SPEAKRS_QUALIFY_CACHE", default)).resolve()
    if cache == root.resolve() or root.resolve() in cache.parents:
        raise LockError("SPEAKRS_QUALIFY_CACHE must be outside the repository")
    return cache


def asset_path(digest: str, root: Path = ROOT) -> Path:
    """Map only a SHA-256 digest to the cache, never an unchecked relative path."""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise LockError("invalid external asset digest")
    return cache_directory(root) / "assets" / digest


def resolve(name: str, root: Path = ROOT) -> Path:
    """Require the exact owner-locked bytes before using a cached asset."""
    snapshot = json.loads((root / "scripts/cuda/qualify/LOCK").read_text())
    digest = scope(root).get("external_assets", {}).get(name)
    if digest is None or snapshot["files"].get(name) != digest:
        raise LockError(f"external asset is not owner-locked: {name}")
    path = asset_path(digest, root)
    instruction = f"python3 scripts/cuda/qualify/assets.py --import-file /path/to/owner-file --name {name}"
    if path.is_symlink() or not path.is_file():
        raise LockError(
            f"missing qualification asset {path}; SHA-256 {digest}; import with: {instruction}"
        )
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != digest:
        raise LockError(
            f"qualification asset hash mismatch: {path}; expected {digest}; import with: {instruction}"
        )
    return path


def retain(name: str, contents: bytes, root: Path = ROOT) -> Path:
    """Store regenerated evidence and update scope before the owner makes a new lock."""
    digest = hashlib.sha256(contents).hexdigest()
    path = asset_path(digest, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    definition = scope(root)
    definition.setdefault("external_assets", {})[name] = digest
    (root / SCOPE).write_text(json.dumps(definition, indent=2) + "\n")
    return path


def main() -> None:
    """Import an owner file without changing its scope or lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-file", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    digest = scope().get("external_assets", {}).get(args.name)
    if digest is None:
        raise LockError(f"unknown external asset: {args.name}")
    contents = args.import_file.read_bytes()
    if hashlib.sha256(contents).hexdigest() != digest:
        raise LockError(f"import does not match owner SHA-256 {digest}")
    path = asset_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    print(resolve(args.name))


if __name__ == "__main__":
    main()
