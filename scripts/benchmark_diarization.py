# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Benchmark Rust and pyannote diarization on the same WAV input."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchResult:
    name: str
    seconds: list[float]

    @property
    def mean_seconds(self) -> float:
        return statistics.mean(self.seconds)

    @property
    def min_seconds(self) -> float:
        return min(self.seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--rust-binary", required=True)
    parser.add_argument("--python-script", required=True)
    parser.add_argument(
        "--python-device",
        choices=("auto", "cpu", "mps"),
        default="auto",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    return parser.parse_args()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def run_once(command: list[str]) -> float:
    with tempfile.NamedTemporaryFile(suffix=".rttm") as output:
        started = time.perf_counter()
        subprocess.run(
            command,
            check=True,
            stdout=output,
            stderr=subprocess.DEVNULL,
        )
        return time.perf_counter() - started


def benchmark(name: str, command: list[str], runs: int, warmups: int) -> BenchResult:
    for _ in range(warmups):
        run_once(command)

    seconds = [run_once(command) for _ in range(runs)]
    return BenchResult(name=name, seconds=seconds)


def format_speed(audio_seconds: float, runtime_seconds: float) -> str:
    return f"{audio_seconds / runtime_seconds:.2f}x"


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav_path)
    audio_seconds = wav_duration_seconds(wav_path)

    rust_result = benchmark(
        "Rust",
        [args.rust_binary, str(wav_path)],
        runs=args.runs,
        warmups=args.warmups,
    )
    python_result = benchmark(
        "Python",
        [
            "uv",
            "run",
            args.python_script,
            "--device",
            args.python_device,
            str(wav_path),
        ],
        runs=args.runs,
        warmups=args.warmups,
    )

    print(f"Audio duration: {audio_seconds / 60.0:.2f} minutes ({audio_seconds:.1f}s)")
    print(f"Runs: {args.runs} measured, {args.warmups} warmup")
    print(f"Python device: {args.python_device}")
    print()
    print("Name\tmean_s\tmin_s\tspeed")
    for result in (rust_result, python_result):
        print(
            f"{result.name}\t"
            f"{result.mean_seconds:.3f}\t"
            f"{result.min_seconds:.3f}\t"
            f"{format_speed(audio_seconds, result.mean_seconds)}"
        )


if __name__ == "__main__":
    main()
