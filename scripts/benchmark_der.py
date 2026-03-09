# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyannote.metrics",
#     "pyannote.core",
#     "pyannote.database",
#     "typing_extensions",
# ]
# ///
"""DER benchmark on VoxConverse dev set.

Runs multiple diarization implementations on VoxConverse audio files
and scores each with standard Diarization Error Rate (DER).

Usage:
    benchmark_der.py <voxconverse_dir> --speakrs-binary <path> [--max-files N] [--max-minutes N]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DER benchmark on VoxConverse")
    parser.add_argument(
        "voxconverse_dir",
        help="Path to VoxConverse directory with wav/ and rttm/ subdirs",
    )
    parser.add_argument(
        "--speakrs-binary", required=True, help="Path to speakrs diarize binary"
    )
    parser.add_argument(
        "--pyannote-rs-binary", default="", help="Path to pyannote-rs binary"
    )
    parser.add_argument(
        "--pyannote-rs-seg-model", default="", help="Segmentation ONNX for pyannote-rs"
    )
    parser.add_argument(
        "--pyannote-rs-emb-model", default="", help="Embedding ONNX for pyannote-rs"
    )
    parser.add_argument(
        "--python-script", default="", help="Path to diarize_pyannote.py"
    )
    parser.add_argument(
        "--fluidaudio-path", default="", help="Path to FluidAudio package"
    )
    parser.add_argument(
        "--fluidaudio-rttm-script",
        default="",
        help="Path to fluidaudio_json_to_rttm.py",
    )
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument(
        "--collar",
        type=float,
        default=0.0,
        help="Collar in seconds for DER (default: 0.0)",
    )
    return parser.parse_args()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def run_capture(command: list[str], timeout: int = 600) -> tuple[float, str]:
    """Run a command, return (elapsed_seconds, stdout_text)."""
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return time.perf_counter() - started, ""
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
        Path(json_path).unlink(missing_ok=True)
        Path(rttm_path).unlink(missing_ok=True)
        return elapsed, ""

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


def rttm_to_annotation(rttm_text: str, uri: str = "audio") -> Annotation:
    """Parse RTTM text into a pyannote Annotation."""
    annotation = Annotation(uri=uri)
    for line in rttm_text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            annotation[Segment(start, start + dur)] = speaker
    return annotation


def load_ref_rttm(rttm_path: Path) -> Annotation:
    """Load a reference RTTM file using pyannote."""
    rttms = load_rttm(str(rttm_path))
    # load_rttm returns dict of uri -> Annotation, take the first
    return next(iter(rttms.values()))


def discover_files(
    voxconverse_dir: Path, max_files: int, max_minutes: float
) -> list[tuple[Path, Path]]:
    """Find paired wav+rttm files, sorted by duration, within budget."""
    wav_dir = voxconverse_dir / "wav"
    rttm_dir = voxconverse_dir / "rttm"

    if not wav_dir.exists() or not rttm_dir.exists():
        # try flat layout
        wav_dir = voxconverse_dir
        rttm_dir = voxconverse_dir

    pairs: list[tuple[Path, Path, float]] = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        rttm_path = rttm_dir / f"{wav_path.stem}.rttm"
        if rttm_path.exists():
            try:
                dur = wav_duration_seconds(wav_path)
                pairs.append((wav_path, rttm_path, dur))
            except Exception:
                continue

    # sort by duration (shortest first) to fill budget efficiently
    pairs.sort(key=lambda x: x[2])

    selected: list[tuple[Path, Path]] = []
    total_minutes = 0.0
    for wav_path, rttm_path, dur in pairs:
        if len(selected) >= max_files:
            break
        if total_minutes + dur / 60.0 > max_minutes and selected:
            break
        selected.append((wav_path, rttm_path))
        total_minutes += dur / 60.0

    return selected


def main() -> None:
    args = parse_args()
    voxconverse_dir = Path(args.voxconverse_dir)

    files = discover_files(voxconverse_dir, args.max_files, args.max_minutes)
    if not files:
        print(f"No paired wav+rttm files found in {voxconverse_dir}")
        return

    total_audio_seconds = sum(wav_duration_seconds(wav) for wav, _ in files)
    total_audio_minutes = total_audio_seconds / 60.0

    print(f"Found {len(files)} files, {total_audio_minutes:.1f} min total audio")
    print()

    # build list of implementations
    implementations: list[tuple[str, str]] = []  # (name, type)
    if args.python_script:
        implementations.append(("pyannote MPS", "pyannote"))
    implementations.append(("speakrs CoreML", "speakrs-coreml"))
    implementations.append(("speakrs CoreML Lite", "speakrs-minicoreml"))
    has_fluidaudio = bool(args.fluidaudio_path and args.fluidaudio_rttm_script)
    if has_fluidaudio:
        implementations.append(("FluidAudio", "fluidaudio"))
    has_pyannote_rs = bool(
        args.pyannote_rs_binary
        and args.pyannote_rs_seg_model
        and args.pyannote_rs_emb_model
    )
    if has_pyannote_rs:
        implementations.append(("pyannote-rs", "pyannote-rs"))

    # score each implementation
    results: dict[str, dict] = {}

    for impl_name, impl_type in implementations:
        print(f"Running {impl_name}...")
        metric = DiarizationErrorRate(collar=args.collar)
        total_time = 0.0
        file_count = 0

        for wav_path, rttm_path in files:
            ref = load_ref_rttm(rttm_path)

            if impl_type == "pyannote":
                with tempfile.NamedTemporaryFile(suffix=".rttm", delete=False) as f:
                    out_path = f.name
                elapsed, _ = run_capture(
                    [
                        "uv",
                        "run",
                        args.python_script,
                        "--device",
                        "mps",
                        "--output",
                        out_path,
                        str(wav_path),
                    ]
                )
                try:
                    hyp_text = Path(out_path).read_text()
                except FileNotFoundError:
                    hyp_text = ""
                Path(out_path).unlink(missing_ok=True)
            elif impl_type == "speakrs-coreml":
                elapsed, hyp_text = run_capture(
                    [
                        args.speakrs_binary,
                        "--mode",
                        "coreml",
                        str(wav_path),
                    ]
                )
            elif impl_type == "speakrs-minicoreml":
                elapsed, hyp_text = run_capture(
                    [
                        args.speakrs_binary,
                        "--mode",
                        "coreml-lite",
                        str(wav_path),
                    ]
                )
            elif impl_type == "fluidaudio":
                elapsed, hyp_text = run_fluidaudio(
                    args.fluidaudio_path,
                    args.fluidaudio_rttm_script,
                    str(wav_path),
                )
            elif impl_type == "pyannote-rs":
                elapsed, hyp_text = run_capture(
                    [
                        args.pyannote_rs_binary,
                        str(wav_path),
                        args.pyannote_rs_seg_model,
                        args.pyannote_rs_emb_model,
                    ]
                )
            else:
                continue

            total_time += elapsed

            if hyp_text.strip():
                hyp = rttm_to_annotation(hyp_text, uri=ref.uri)
                metric(ref, hyp)
                file_count += 1
            else:
                file_count += 1

            print(f"  {wav_path.stem}: {elapsed:.1f}s", flush=True)

        # extract DER and component rates
        # results_ is list of (uri, detail_dict) tuples
        der_value = abs(metric)
        details = [d for _, d in metric.results_] if hasattr(metric, "results_") else []
        if der_value is not None and details:
            der_pct = der_value * 100
            ref_total = sum(d["total"] for d in details)
            if ref_total > 0:
                miss_pct = sum(d["missed detection"] for d in details) / ref_total * 100
                fa_pct = sum(d["false alarm"] for d in details) / ref_total * 100
                conf_pct = sum(d["confusion"] for d in details) / ref_total * 100
            else:
                miss_pct = fa_pct = conf_pct = 0.0
        else:
            der_pct = miss_pct = fa_pct = conf_pct = None

        results[impl_name] = {
            "der": der_pct,
            "missed": miss_pct,
            "false_alarm": fa_pct,
            "confusion": conf_pct,
            "time": total_time,
            "files": file_count,
        }
        print(
            f"  → DER: {der_pct:.1f}%, Time: {total_time:.1f}s"
            if der_pct is not None
            else f"  → N/A, Time: {total_time:.1f}s"
        )
        print()

    # print summary table
    print()
    print(
        f"VoxConverse DER ({len(files)} files, {total_audio_minutes:.1f} min total, collar={args.collar:.0f}ms)"
    )
    print()

    name_w = 22
    header = f"{'Implementation':<{name_w}} {'DER%':>8} {'Missed%':>10} {'FalseAlarm%':>13} {'Confusion%':>12} {'Time':>8}"
    print(header)
    print("─" * len(header))

    for impl_name, _ in implementations:
        r = results[impl_name]
        if r["der"] is not None:
            der_str = f"{r['der']:.1f}%"
            miss_str = f"{r['missed']:.1f}%"
            fa_str = f"{r['false_alarm']:.1f}%"
            conf_str = f"{r['confusion']:.1f}%"
        else:
            der_str = "N/A"
            miss_str = fa_str = conf_str = "—"
        time_str = f"{r['time']:.1f}s"
        print(
            f"{impl_name:<{name_w}} {der_str:>8} {miss_str:>10} {fa_str:>13} {conf_str:>12} {time_str:>8}"
        )


if __name__ == "__main__":
    main()
