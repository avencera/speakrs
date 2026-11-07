"""Make pip-installed cuDNN and cuBLAS visible to the CUDA backend.

The native CUDA backend opens cuDNN 9 and cuBLAS 12 by name only when a layer
falls back to them: ``libcudnn.so.9`` and ``libcublas.so.12`` on Linux,
``cudnn64_9.dll`` and ``cublas64_12.dll`` on Windows. The ``nvidia-cudnn-cu12``
and ``nvidia-cublas-cu12`` wheels install those libraries inside site-packages,
where neither loader looks, so they are loaded here first. On Linux glibc then
resolves the backend's soname lookups to the already-loaded copies; Windows
reuses a loaded DLL with the requested name, from whatever directory it came
from.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

_CUDA_MODES = {"cuda", "cudafast"}
_WINDOWS = sys.platform == "win32"
# dependencies before dependents, so each load finds what it links against. Each
# entry names the package directory inside the wheels' ``nvidia`` namespace package
# and the libraries in its ``bin`` (Windows) or ``lib`` (Linux) directory
_LIBRARIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        ("cublas", ("cublasLt64_12.dll", "cublas64_12.dll")),
        (
            "cudnn",
            (
                "cudnn_graph64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_heuristic64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll",
                "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll",
                "cudnn64_9.dll",
            ),
        ),
    )
    if _WINDOWS
    else (
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
)
_LIBRARY_DIR = "bin" if _WINDOWS else "lib"
_prepared = False
# a second thread must wait for the first to finish loading, not skip ahead
_lock = threading.Lock()


def prepare_libraries(mode: str) -> None:
    """Load the pip CUDA libraries once, before the first CUDA pipeline."""
    global _prepared
    normalized = mode.lower().replace("-", "").replace("_", "")
    # macOS has no CUDA backend
    if normalized not in _CUDA_MODES or not (_WINDOWS or sys.platform.startswith("linux")):
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
            lib_dir = Path(root) / package / _LIBRARY_DIR
            for name in names:
                path = lib_dir / name
                if path.is_file():
                    try:
                        # Windows ignores the mode; an absolute path there also
                        # searches the DLL's own directory for its imports
                        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        # a broken install falls back to the system search path
                        pass
