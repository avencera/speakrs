"""Freeze the unchanged Library source, separate from later candidate edits."""

import gzip
import io
import tarfile

from lock import ROOT
from assets import retain


def freeze() -> None:
    """Write a deterministic source archive before the owner records the lock."""
    paths = {ROOT / "Cargo.toml", ROOT / "Cargo.lock"}
    for directory in ("src", "fixtures", "tests", "xtask", "crates", "examples"):
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            # baseline records bind to this archive; including them would create a cycle
            if relative.is_relative_to("tests/cuda_qualify/baselines"):
                continue
            # the harness's own Python and scan fixtures never build the control, so
            # editing them must not invalidate the control or its baseline
            if relative.is_relative_to("tests/cuda_qualify") and (
                path.suffix == ".py"
                or relative.is_relative_to("tests/cuda_qualify/scan_fixtures")
            ):
                continue
            if any(
                part in ("target", "models", "datasets", "__pycache__", ".venv")
                for part in relative.parts
            ):
                continue
            if path == ROOT / "tests/cuda_qualify/control.tar.gz":
                continue
            if path.is_symlink():
                raise ValueError(f"control source symlink refused: {relative}")
            if path.is_file():
                paths.add(path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted(paths):
            info = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
            data = path.read_bytes()
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    output = retain(
        "tests/cuda_qualify/control.tar.gz", gzip.compress(buffer.getvalue(), mtime=0)
    )
    print(output)


if __name__ == "__main__":
    freeze()
