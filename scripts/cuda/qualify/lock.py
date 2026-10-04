"""Hash the harness inventory. LOCK is excluded to avoid a self-reference.

The inventory is defined by the locked SCOPE.json, which the xtask entry point reads
too: the harness directories, every Rust file the qualification binary compiles except
candidate code, the Library-owned PTX, the manifests and a few single files.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "scripts/cuda/qualify/LOCK"
SCOPE = "scripts/cuda/qualify/SCOPE.json"


class LockError(ValueError):
    """The harness does not match the owner's recorded snapshot."""


def scope(root: Path = ROOT) -> dict:
    """Read the locked inventory definition; it is itself part of the inventory."""
    path = root / SCOPE
    if path.is_symlink() or not path.is_file():
        raise LockError("harness has no regular SCOPE.json")
    data = json.loads(path.read_text())
    if data.get("schema") != 1:
        raise LockError("unsupported harness scope schema")
    return data


def excluded(relative: Path, rules: list[str]) -> bool:
    """A path is excluded when it is a listed entry or lies under one; a rule ending
    in `*` excludes every path that starts with the rest of it."""
    text = relative.as_posix()
    return any(
        text.startswith(rule[:-1])
        if rule.endswith("*")
        else text == rule or text.startswith(rule + "/")
        for rule in rules
    )


def inventory(root: Path = ROOT) -> dict[str, str]:
    """Include new files and reject symlinks; exclude interpreter cache only."""
    definition = scope(root)
    paths = {root / name for name in definition["files"]}
    for directory in definition["directories"]:
        base = root / directory["path"]
        if base.is_symlink() or not base.is_dir():
            raise LockError(f"missing regular harness directory: {base}")
        for path in base.rglob("*"):
            if excluded(path.relative_to(base), directory["exclude"]):
                continue
            if path.is_symlink():
                raise LockError(f"symlink in harness: {path}")
            if path == root / "scripts/cuda/qualify/LOCK":
                continue
            if path.suffix == ".pyc" and path.parent.name == "__pycache__":
                continue
            if path.is_file():
                paths.add(path)
    result = {}
    for path in sorted(paths):
        if any(
            parent.is_symlink()
            for parent in path.parents
            if parent != root and root in parent.parents
        ):
            raise LockError(f"symlink in harness parent: {path}")
        if path.is_symlink() or not path.is_file():
            raise LockError(f"missing regular harness file: {path}")
        result[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for name, digest in definition.get("external_assets", {}).items():
        if name in result or (root / name).exists():
            raise LockError(f"external asset must stay outside the tree: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise LockError(f"invalid external asset digest: {name}")
        result[name] = digest
    return result


def canonical(files: dict[str, str]) -> bytes:
    """Use one stable digest format on the Mac and Linux box."""
    return (
        json.dumps({"schema": 1, "files": files}, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def verify(root: Path = ROOT, expected: str | None = None) -> str:
    """Check inventory and bytes, then optionally check the outside-tree digest."""
    path = root / "scripts/cuda/qualify/LOCK"
    if not path.is_file() or path.is_symlink():
        raise LockError("harness has no regular LOCK file")
    snapshot = path.read_bytes()
    current = canonical(inventory(root))
    if snapshot != current:
        raise LockError("harness files differ from LOCK; qualification refused")
    digest = hashlib.sha256(snapshot).hexdigest()
    if expected is not None and digest != expected:
        raise LockError("LOCK differs from the owner's outside-tree digest")
    return digest


def formatter_stable(root: Path = ROOT) -> None:
    """Refuse to lock bytes that `cargo fmt` or `ruff format` would rewrite.

    Routine formatting of the tree must never change a locked file.
    """
    files = inventory(root)
    python = sorted(name for name in files if name.endswith(".py"))
    ruff = ["ruff"] if shutil.which("ruff") else ["uv", "run", "--group", "dev", "ruff"]
    checks = [
        ["cargo", "fmt", "--all", "--check"],
        [*ruff, "format", "--check", *python],
    ]
    for argv in checks:
        process = subprocess.run(
            argv, cwd=root, capture_output=True, text=True, check=False
        )
        if process.returncode:
            raise LockError(
                f"locked files are not formatter-stable ({' '.join(argv[:4])}): "
                f"{(process.stdout + process.stderr)[-2000:]}"
            )


def main() -> None:
    """Write a formatter-stable snapshot, or check one without changing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--expected")
    args = parser.parse_args()
    if not args.check:
        formatter_stable()
        LOCK.write_bytes(canonical(inventory()))
    print(verify(expected=args.expected))


if __name__ == "__main__":
    main()
