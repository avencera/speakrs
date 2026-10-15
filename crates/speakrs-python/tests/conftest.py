"""Shared fixtures: model directories and the Rust reference RTTM."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import speakrs

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "fixtures"
TEST_WAV = FIXTURES / "test.wav"
SHORT_WAV = FIXTURES / "test_short.wav"


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono WAV as float32, scaled like the Rust examples."""
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16_000, 2)
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


@pytest.fixture(scope="session")
def cpu_models() -> speakrs.PreparedModels:
    """The CPU fixture models, or a download into the default cache if they are absent."""
    local = FIXTURES / "models"
    if all((local / name).is_file() for name in speakrs.required_model_files("cpu")):
        return speakrs.prepare("cpu", model_dir=local)
    return speakrs.prepare("cpu")


@pytest.fixture(scope="session")
def cpu_pipeline(cpu_models: speakrs.PreparedModels) -> speakrs.Pipeline:
    return speakrs.Pipeline.from_prepared(cpu_models)


def _reference_features() -> str:
    # the same backends as the wheel, so one cargo build serves every mode
    if sys.platform == "darwin":
        return "coreml"
    if sys.platform.startswith("linux"):
        return "cuda"
    return "cpu"


def rust_reference(mode: str, model_dir: Path, wav: Path, file_id: str) -> tuple[str, str]:
    """RTTM and exclusive RTTM from the core Rust pipeline for the same input."""
    command = os.environ.get("SPEAKRS_REFERENCE_RTTM")
    if command:
        argv = [command]
    else:
        if shutil.which("cargo") is None:
            pytest.skip("cargo is needed to compute the Rust reference RTTM")
        argv = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(REPO_ROOT / "Cargo.toml"),
            "-p",
            "speakrs-sdk",
            "--example",
            "reference_rttm",
            "--features",
            _reference_features(),
            "--",
        ]
    output = subprocess.run(
        [*argv, mode, str(model_dir), str(wav), file_id],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rttm, exclusive = output.split("# exclusive\n")
    return rttm, exclusive
