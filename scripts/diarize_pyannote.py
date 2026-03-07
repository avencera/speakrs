# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyannote.audio>=3.3",
#     "torch>=2.6",
# ]
# ///
"""Run pyannote community-1 diarization on a WAV file and print RTTM."""

import os
import sys

import torch


def main():
    if len(sys.argv) != 2:
        print("Usage: diarize_pyannote.py <path/to/audio.wav>", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]
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

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    pipeline.to(torch.device("cpu"))

    result = pipeline({"audio": wav_path})

    for seg, _, speaker in result.itertracks(yield_label=True):
        print(
            f"SPEAKER file1 1 {seg.start:.6f} {seg.duration:.6f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )


if __name__ == "__main__":
    main()
