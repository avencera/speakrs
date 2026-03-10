# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyannote.audio>=3.3",
#     "torch>=2.6",
#     "torchaudio>=2.6",
#     "soundfile>=0.12",
#     "tqdm>=4.66",
# ]
# ///
"""Run pyannote community-1 diarization on WAV files and print RTTM."""

import argparse
import os
import sys
import time
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
        default=int(os.environ.get("PYANNOTE_SEGMENTATION_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.environ.get("PYANNOTE_EMBEDDING_BATCH_SIZE", "16")),
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
        precision = "highest" if device.type == "cuda" else "high"
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
        torch.backends.cudnn.benchmark = False
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
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    assert pipeline is not None
    pipeline.to(device)
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = args.segmentation_batch_size
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = args.embedding_batch_size

    all_output = ""
    multiple = len(args.wav_files) > 1

    if multiple:
        from tqdm import tqdm

        pbar = tqdm(
            args.wav_files, file=sys.stderr, bar_format="{l_bar}{bar:20}{r_bar}"
        )
    else:
        pbar = args.wav_files

    for wav_path in pbar:
        file_id = os.path.splitext(os.path.basename(wav_path))[0]
        t0 = time.monotonic()
        all_output += diarize_file(pipeline, wav_path, file_id)
        elapsed = time.monotonic() - t0
        if multiple:
            pbar.set_postfix_str(f"{file_id}: {elapsed:.1f}s")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(all_output)
    else:
        sys.stdout.write(all_output)


if __name__ == "__main__":
    main()
