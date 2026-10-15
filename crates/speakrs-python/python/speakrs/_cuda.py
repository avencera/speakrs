"""Make pip-installed cuDNN and cuBLAS visible to the CUDA backend.

The native CUDA backend opens ``libcudnn.so.9`` and ``libcublas.so.12`` by
soname only when a layer falls back to them. The ``nvidia-cudnn-cu12`` and
``nvidia-cublas-cu12`` wheels install those libraries inside site-packages,
where the dynamic loader does not look, so they are loaded here first: glibc
then resolves the backend's soname lookups to the already-loaded copies.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

_CUDA_MODES = {"cuda", "cudafast"}
# dependencies before dependents, so each load finds what it links against
_LIBRARIES = (
    ("cublas", ("libcublasLt.so.12", "libcublas.so.12")),
    (
        "cudnn",
        (
            "libcudnn_graph.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_heuristic.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_adv.so.9",
            "libcudnn.so.9",
        ),
    ),
)
_prepared = False
# a second thread must wait for the first to finish loading, not skip ahead
_lock = threading.Lock()


def prepare_libraries(mode: str) -> None:
    """Load the pip CUDA libraries once, before the first CUDA pipeline."""
    global _prepared
    normalized = mode.lower().replace("-", "").replace("_", "")
    if normalized not in _CUDA_MODES or not sys.platform.startswith("linux"):
        return
    with _lock:
        if not _prepared:
            _load()
            _prepared = True


def _load() -> None:
    try:
        import nvidia  # namespace package shared by the nvidia-* wheels
    except ImportError:
        return

    for root in getattr(nvidia, "__path__", []):
        for package, names in _LIBRARIES:
            lib_dir = Path(root) / package / "lib"
            for name in names:
                path = lib_dir / name
                if path.is_file():
                    try:
                        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        # a broken install falls back to the system search path
                        pass
