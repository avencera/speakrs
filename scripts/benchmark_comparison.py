# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Benchmark multiple diarization implementations and compare results.

Runs speakrs CoreML, speakrs MiniCoreML, pyannote MPS, and pyannote-rs
on the same WAV input. Outputs a comparison table with total time,
speaker count, segment count, and parity % vs pyannote MPS.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    name: str
    seconds: float
    rttm: str
    speakers: int = 0
    segments: int = 0

    def __post_init__(self) -> None:
        lines = [l for l in self.rttm.strip().splitlines() if l.startswith("SPEAKER")]
        self.segments = len(lines)
        speaker_set: set[str] = set()
        for line in lines:
            parts = line.split()
            if len(parts) >= 8:
                speaker_set.add(parts[7])
        self.speakers = len(speaker_set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark diarization implementations"
    )
    parser.add_argument("wav_path", help="Path to 16kHz mono WAV file")
    parser.add_argument(
        "--speakrs-binary", required=True, help="Path to speakrs diarize binary"
    )
    parser.add_argument(
        "--pyannote-rs-binary", required=True, help="Path to pyannote-rs binary"
    )
    parser.add_argument(
        "--pyannote-rs-seg-model",
        required=True,
        help="Segmentation ONNX for pyannote-rs",
    )
    parser.add_argument(
        "--pyannote-rs-emb-model", required=True, help="Embedding ONNX for pyannote-rs"
    )
    parser.add_argument(
        "--python-script", required=True, help="Path to diarize_pyannote.py"
    )
    parser.add_argument(
        "--fluidaudio-path",
        default="",
        help="Path to FluidAudio package (omit to skip)",
    )
    parser.add_argument(
        "--fluidaudio-rttm-script",
        default="",
        help="Path to fluidaudio_json_to_rttm.py",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def run_capture(command: list[str]) -> tuple[float, str]:
    """Run a command, return (elapsed_seconds, stdout_text)."""
    started = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        return elapsed, ""
    return elapsed, result.stdout


def run_fluidaudio(
    fluidaudio_path: str,
    rttm_script: str,
    wav_path: str,
) -> tuple[float, str]:
    """Run FluidAudio and convert JSON output to RTTM."""
    with (
        tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_f,
        tempfile.NamedTemporaryFile(suffix=".rttm", delete=False) as rttm_f,
    ):
        json_path = json_f.name
        rttm_path = rttm_f.name

    started = time.perf_counter()
    result = subprocess.run(
        [
            "swift",
            "run",
            "-c",
            "release",
            "--package-path",
            fluidaudio_path,
            "fluidaudiocli",
            "process",
            wav_path,
            "--mode",
            "offline",
            "--output",
            json_path,
        ],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        return elapsed, ""

    # convert JSON → RTTM
    subprocess.run(
        ["uv", "run", rttm_script, json_path, "--output", rttm_path],
        capture_output=True,
        text=True,
        check=True,
    )
    rttm = Path(rttm_path).read_text()

    Path(json_path).unlink(missing_ok=True)
    Path(rttm_path).unlink(missing_ok=True)
    return elapsed, rttm


def parse_rttm_intervals(rttm: str) -> list[tuple[float, float]]:
    """Parse RTTM into list of (start, end) intervals."""
    intervals = []
    for line in rttm.strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "SPEAKER":
            start = float(parts[3])
            dur = float(parts[4])
            intervals.append((start, start + dur))
    return intervals


def timeline_overlap_pct(ref_rttm: str, test_rttm: str) -> float | None:
    """Compute what % of ref's speech timeline is covered by test."""
    ref_intervals = parse_rttm_intervals(ref_rttm)
    test_intervals = parse_rttm_intervals(test_rttm)

    if not ref_intervals or not test_intervals:
        return None

    # merge test intervals
    test_intervals.sort()
    merged = [test_intervals[0]]
    for start, end in test_intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    ref_total = sum(e - s for s, e in ref_intervals)
    if ref_total <= 0:
        return None

    covered = 0.0
    for rs, re in ref_intervals:
        for ms, me in merged:
            if ms >= re:
                break
            if me <= rs:
                continue
            covered += min(re, me) - max(rs, ms)

    return covered / ref_total * 100.0


def main() -> None:
    args = parse_args()
    wav_path = args.wav_path
    audio_seconds = wav_duration_seconds(wav_path)

    implementations: list[tuple[str, list[str]]] = [
        (
            "pyannote MPS",
            ["uv", "run", args.python_script, "--device", "mps", wav_path],
        ),
        (
            "speakrs CoreML",
            [args.speakrs_binary, "--mode", "coreml", wav_path],
        ),
        (
            "speakrs MiniCoreML",
            [args.speakrs_binary, "--mode", "mini-coreml", wav_path],
        ),
        (
            "pyannote-rs",
            [
                args.pyannote_rs_binary,
                wav_path,
                args.pyannote_rs_seg_model,
                args.pyannote_rs_emb_model,
            ],
        ),
    ]

    has_fluidaudio = bool(args.fluidaudio_path and args.fluidaudio_rttm_script)

    results: list[RunResult] = []

    for name, command in implementations:
        for _ in range(args.warmups):
            run_capture(command)

        best_time = float("inf")
        best_rttm = ""
        for _ in range(args.runs):
            elapsed, rttm = run_capture(command)
            if elapsed < best_time:
                best_time = elapsed
                best_rttm = rttm

        results.append(RunResult(name=name, seconds=best_time, rttm=best_rttm))
        print(f"  {name}: {best_time:.2f}s", flush=True)

    if has_fluidaudio:
        for _ in range(args.warmups):
            run_fluidaudio(args.fluidaudio_path, args.fluidaudio_rttm_script, wav_path)

        best_time = float("inf")
        best_rttm = ""
        for _ in range(args.runs):
            elapsed, rttm = run_fluidaudio(
                args.fluidaudio_path, args.fluidaudio_rttm_script, wav_path
            )
            if elapsed < best_time:
                best_time = elapsed
                best_rttm = rttm

        results.append(RunResult(name="FluidAudio", seconds=best_time, rttm=best_rttm))
        print(f"  FluidAudio: {best_time:.2f}s", flush=True)

    # pyannote MPS is the reference
    ref_result = results[0]

    minutes = int(audio_seconds // 60)
    secs = audio_seconds % 60
    print()
    print(
        f"Audio: {minutes}:{secs:04.1f} ({audio_seconds:.1f}s)  |  Warmups: {args.warmups}  |  Runs: {args.runs}"
    )
    print()

    # table header
    name_w = 22
    print(
        f"{'Implementation':<{name_w}} {'Time':>8} {'Speakers':>9} {'Segments':>9} {'Parity %':>10}"
    )
    print("─" * (name_w + 8 + 9 + 9 + 10 + 4))

    for result in results:
        time_str = f"{result.seconds:.2f}s"
        speakers_str = str(result.speakers) if result.segments > 0 else "—"
        segments_str = str(result.segments) if result.segments > 0 else "0"

        if result is ref_result:
            parity_str = "(reference)"
        elif result.segments == 0:
            parity_str = "N/A"
        else:
            parity = timeline_overlap_pct(ref_result.rttm, result.rttm)
            if parity is not None:
                parity_str = f"{parity:.1f}%"
            else:
                parity_str = "N/A"

        print(
            f"{result.name:<{name_w}} {time_str:>8} {speakers_str:>9} {segments_str:>9} {parity_str:>10}"
        )

    # note about pyannote-rs limitation
    pyannote_rs_result = next((r for r in results if r.name == "pyannote-rs"), None)
    if pyannote_rs_result and pyannote_rs_result.segments == 0:
        print()
        print("Note: pyannote-rs returned 0 segments. It only emits segments when")
        print(
            "speech→silence transitions occur; continuous speech files produce no output."
        )


if __name__ == "__main__":
    main()
