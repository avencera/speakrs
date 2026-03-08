"""Convert a FluidAudio offline JSON result into RTTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-id", default="file1")
    return parser.parse_args()


def load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("expected 'segments' to be a list")
    return segments


def to_rttm_line(file_id: str, segment: dict[str, Any]) -> str:
    start = float(segment["startTimeSeconds"])
    end = float(segment["endTimeSeconds"])
    speaker = str(segment["speakerId"])
    duration = max(0.0, end - start)
    return (
        f"SPEAKER {file_id} 1 {start:.6f} {duration:.6f} <NA> <NA> {speaker} <NA> <NA>"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output)
    segments = load_segments(input_path)
    lines = [to_rttm_line(args.file_id, segment) for segment in segments]
    output = "\n".join(lines)
    if output:
        output += "\n"
    output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
