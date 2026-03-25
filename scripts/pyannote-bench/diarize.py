"""Run pyannote community-1 diarization on WAV files and print RTTM."""

import argparse
import os
import sys
import time
import wave
from datetime import datetime
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_files", nargs="+")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default=os.environ.get("PYANNOTE_DEVICE", "auto"),
        help="Execution device for pyannote",
    )
    parser.add_argument(
        "--output",
        help="Write RTTM output to this file instead of stdout",
    )
    parser.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if name == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def configure_torch(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        precision = "high"
        torch.set_float32_matmul_precision(precision)

    if device.type == "cpu":
        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(cpu_count)
        interop_threads = max(1, cpu_count // 2)
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass
        return

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def diarize_file(pipeline: Any, wav_path: str, file_id: str) -> str:
    with torch.inference_mode():
        result: Any = pipeline({"audio": wav_path})

    # handle both old Annotation and new DiarizeOutput APIs
    annotation = result
    if hasattr(result, "speaker_diarization"):
        annotation = result.speaker_diarization
    elif not hasattr(result, "itertracks"):
        for attr in ("annotation", "diarization", "output"):
            if hasattr(result, attr):
                annotation = getattr(result, attr)
                break

    lines = []
    for seg, _, speaker in annotation.itertracks(yield_label=True):
        lines.append(
            f"SPEAKER {file_id} 1 {seg.start:.6f} {seg.duration:.6f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    output = "\n".join(lines)
    if output:
        output += "\n"
    return output


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = int(seconds // 60)
    secs = int(round(seconds % 60))
    return f"{mins}m {secs:02d}s"


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        # try cached token
        for p in [
            os.path.expanduser("~/.cache/huggingface/token"),
            os.path.expanduser("~/.huggingface/token"),
        ]:
            if os.path.isfile(p):
                token = open(p).read().strip()
                if token:
                    break

    if not token:
        print("No HF_TOKEN found", file=sys.stderr)
        sys.exit(1)

    from pyannote.audio import Pipeline

    device = resolve_device(args.device)
    configure_torch(device)

    # CUDA works better with batch 32, MPS with 16
    default_batch = 32 if device.type == "cuda" else 16
    seg_batch = args.segmentation_batch_size or int(
        os.environ.get("PYANNOTE_SEGMENTATION_BATCH_SIZE", str(default_batch))
    )
    emb_batch = args.embedding_batch_size or int(
        os.environ.get("PYANNOTE_EMBEDDING_BATCH_SIZE", str(default_batch))
    )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    assert pipeline is not None
    pipeline.to(device)
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = seg_batch
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = emb_batch

    all_output = ""
    total = len(args.wav_files)
    cumulative = 0.0

    for i, wav_path in enumerate(args.wav_files):
        file_id = os.path.splitext(os.path.basename(wav_path))[0]
        with wave.open(wav_path, "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        t0 = time.monotonic()
        all_output += diarize_file(pipeline, wav_path, file_id)
        elapsed = time.monotonic() - t0
        cumulative += elapsed
        rtfx = duration / elapsed if elapsed > 0 else 0

        if total > 1:
            avg = cumulative / (i + 1)
            remaining = (total - i - 1) * avg
            eta = _format_eta(remaining)
            print(
                f"  [{i + 1}/{total}] {file_id}: {elapsed:.1f}s {rtfx:.1f}x RTFx (ETA {eta}) [{datetime.now():%H:%M:%S}]",
                file=sys.stderr,
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(all_output)
    else:
        sys.stdout.write(all_output)


if __name__ == "__main__":
    main()
