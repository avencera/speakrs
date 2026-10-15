"""The ``speakrs`` command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import (
    Pipeline,
    ProgressEvent,
    ProgressKind,
    SpeakrsError,
    __version__,
    available_modes,
    prepare,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    parser = argparse.ArgumentParser(
        prog="speakrs", description="Speaker diarization with RTTM output."
    )
    parser.add_argument("--version", action="version", version=f"speakrs {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    diarize = commands.add_parser(
        "diarize",
        help="diarize audio files and print RTTM",
        description="Diarize audio files and print RTTM to stdout. Several files run as one overlapped batch.",
    )
    diarize.add_argument("audio", type=Path, nargs="+", help="audio files (WAV, FLAC, MP3, Ogg, M4A)")
    diarize.add_argument(
        "--mode",
        default="cpu",
        help=f"execution mode; this build supports {', '.join(available_modes())} (default: cpu)",
    )
    diarize.add_argument("--model-dir", type=Path, help="load models from this directory instead of the cache")
    diarize.add_argument("--cache-dir", type=Path, help="model download cache (default: the Hugging Face hub cache)")
    diarize.add_argument("--file-id", help="RTTM file id for a single input (default: file name without extension)")
    diarize.add_argument(
        "--exclusive",
        action="store_true",
        help="print exclusive segments, with at most one speaker at a time",
    )
    args = parser.parse_args(argv)

    if args.file_id is not None and len(args.audio) != 1:
        parser.error("--file-id needs exactly one audio file")

    try:
        prepared = prepare(
            args.mode,
            cache_dir=args.cache_dir,
            model_dir=args.model_dir,
            progress=_report_download,
        )
        pipeline = Pipeline.from_prepared(prepared)
        if len(args.audio) == 1:
            results = [pipeline.diarize_file(args.audio[0], file_id=args.file_id)]
        else:
            results = pipeline.diarize_files(args.audio)
    except SpeakrsError as error:
        print(f"error: {error.message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    for result in results:
        sys.stdout.write(result.rttm(exclusive=args.exclusive))
    return 0


def _report_download(event: ProgressEvent) -> None:
    if event.kind == ProgressKind.DownloadingModel:
        print(f"downloading {event.name}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
