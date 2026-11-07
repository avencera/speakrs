"""Host-only checks of the pip CUDA preloading against the real wheel layouts.

No GPU or nvidia wheel is needed: a fake ``nvidia`` namespace package holds empty
files laid out like the wheels, and ``ctypes.CDLL`` records each load instead of
opening it. ``_cuda`` is reloaded under each platform, since it picks its library
names when imported.
"""

from __future__ import annotations

import ctypes
import importlib
import sys
import types
from pathlib import Path

import pytest

import speakrs._cuda

# files under each wheel's ``nvidia`` directory, listed from nvidia-cublas-cu12
# 12.9.2.10 and nvidia-cudnn-cu12 9.28.1.4
_WHEEL_FILES = {
    "win32": [
        "cublas/bin/cublas64_12.dll",
        "cublas/bin/cublasLt64_12.dll",
        "cublas/bin/nvblas64_12.dll",
        "cudnn/bin/cudnn64_9.dll",
        "cudnn/bin/cudnn_adv64_9.dll",
        "cudnn/bin/cudnn_cnn64_9.dll",
        "cudnn/bin/cudnn_engines_precompiled64_9.dll",
        "cudnn/bin/cudnn_engines_runtime_compiled64_9.dll",
        "cudnn/bin/cudnn_engines_tensor_ir64_9.dll",
        "cudnn/bin/cudnn_ext64_9.dll",
        "cudnn/bin/cudnn_graph64_9.dll",
        "cudnn/bin/cudnn_heuristic64_9.dll",
        "cudnn/bin/cudnn_ops64_9.dll",
    ],
    "linux": [
        "cublas/lib/libcublas.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libnvblas.so.12",
        "cudnn/lib/libcudnn.so.9",
        "cudnn/lib/libcudnn_adv.so.9",
        "cudnn/lib/libcudnn_cnn.so.9",
        "cudnn/lib/libcudnn_engines_precompiled.so.9",
        "cudnn/lib/libcudnn_engines_runtime_compiled.so.9",
        "cudnn/lib/libcudnn_engines_tensor_ir.so.9",
        "cudnn/lib/libcudnn_ext.so.9",
        "cudnn/lib/libcudnn_graph.so.9",
        "cudnn/lib/libcudnn_heuristic.so.9",
        "cudnn/lib/libcudnn_ops.so.9",
    ],
}

# the names the native backend opens, and the libraries each must follow
_OPENED = {
    "win32": ("cublas64_12.dll", "cudnn64_9.dll"),
    "linux": ("libcublas.so.12", "libcudnn.so.9"),
}
_BEFORE = {
    "win32": [
        ("cublasLt64_12.dll", "cublas64_12.dll"),
        ("cublasLt64_12.dll", "cudnn64_9.dll"),
        ("cudnn_graph64_9.dll", "cudnn64_9.dll"),
        ("cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"),
    ],
    "linux": [
        ("libcublasLt.so.12", "libcublas.so.12"),
        ("libcublasLt.so.12", "libcudnn.so.9"),
        ("libcudnn_graph.so.9", "libcudnn.so.9"),
        ("libcudnn_ops.so.9", "libcudnn_cnn.so.9"),
    ],
}


@pytest.fixture(autouse=True)
def _restore_module():
    yield
    # later tests and the package itself see the module for the real platform
    importlib.reload(speakrs._cuda)


def _preload(monkeypatch, tmp_path: Path, platform: str, failing: str | None = None) -> list[Path]:
    root = tmp_path / "nvidia"
    for name in _WHEEL_FILES[platform]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()

    nvidia = types.ModuleType("nvidia")
    nvidia.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)

    loaded: list[Path] = []

    def record(path: str, mode: int = 0) -> None:
        loaded.append(Path(path))
        if Path(path).name == failing:
            raise OSError("broken copy")

    monkeypatch.setattr(ctypes, "CDLL", record)
    monkeypatch.setattr(sys, "platform", platform)
    cuda = importlib.reload(speakrs._cuda)

    cuda.prepare_libraries("cpu")
    assert loaded == [], "a CPU pipeline loads no CUDA libraries"
    cuda.prepare_libraries("CUDA-fast")
    cuda.prepare_libraries("cuda")
    return loaded


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_preloads_the_wheel_libraries_in_dependency_order(monkeypatch, tmp_path, platform):
    loaded = _preload(monkeypatch, tmp_path, platform)
    names = [path.name for path in loaded]

    assert len(names) == len(set(names)), "a second prepare loads nothing again"
    available = {Path(name).name for name in _WHEEL_FILES[platform]}
    assert set(names) <= available
    for name in _OPENED[platform]:
        assert name in names, f"{name} is opened by the backend but never preloaded"
    for first, then in _BEFORE[platform]:
        assert names.index(first) < names.index(then), f"{first} must load before {then}"
    # an absolute path makes Windows search the DLL's own directory for its imports
    assert all(path.is_absolute() and path.is_file() for path in loaded)


def test_a_broken_library_leaves_the_rest_to_load(monkeypatch, tmp_path):
    loaded = _preload(monkeypatch, tmp_path, "win32", failing="cublasLt64_12.dll")
    names = [path.name for path in loaded]

    assert "cublasLt64_12.dll" in names
    assert "cudnn64_9.dll" in names


def test_macos_loads_nothing(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: pytest.fail("loaded a library"))
    monkeypatch.setattr(sys, "platform", "darwin")
    importlib.reload(speakrs._cuda).prepare_libraries("cuda")
