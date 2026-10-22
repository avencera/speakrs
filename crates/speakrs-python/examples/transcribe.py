# /// script
# requires-python = ">=3.11"
# dependencies = ["speakrs[cuda]", "faster-whisper", "av<19"]
# ///
# speakrs[cuda] supplies whisper's GPU libraries on Linux and installs nothing
# on macOS; av<19 because faster-whisper 1.2.1 passes an argument PyAV 19 removed
"""Write a speaker-labeled transcript of a recording.

speakrs finds who spoke when, faster-whisper finds what was said, and each
transcribed word goes to the speaker talking at that moment.

    uv run transcribe.py meeting.mp3
    uv run transcribe.py meeting.mp3 --mode cuda --whisper-model turbo
"""

import argparse
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import ctranslate2
import speakrs
from faster_whisper import WhisperModel

T = TypeVar("T")

# a lone word this close to a neighboring turn is timestamp jitter; a real
# interjection like "Yeah?" has a pause on both sides
JITTER_GAP = 0.6


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    words: list[str]


def default_mode() -> str:
    modes = speakrs.available_modes()
    if "CoreML" in modes:
        return "CoreML"

    if "CUDA" in modes and ctranslate2.get_cuda_device_count() > 0:
        return "CUDA"

    return "CPU"


def timed(label: str, step: Callable[..., T], *args: object) -> T:
    started = time.perf_counter()
    result = step(*args)
    print(f"{label}: {time.perf_counter() - started:.1f}s", file=sys.stderr)
    return result


def diarize(pipeline: speakrs.Pipeline, path: str) -> list[speakrs.Segment]:
    # exclusive segments give each moment to one speaker, so overlapping speech never splits a word
    return pipeline.diarize_file(path).exclusive_segments


def transcribe(path: str, model_name: str) -> list[tuple[float, float, str]]:
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    segments, _info = model.transcribe(path, word_timestamps=True, vad_filter=True)
    return [
        (word.start, word.end, word.word)
        for segment in segments
        for word in segment.words or []
    ]


def assign_speakers(
    words: list[tuple[float, float, str]], segments: list[speakrs.Segment]
) -> list[Turn]:
    turns: list[Turn] = []
    index = 0

    for start, end, text in words:
        # both lists are sorted by time, so the search only moves forward
        while index < len(segments) and segments[index].end <= start:
            index += 1

        speaker = best_speaker(segments, index, start, end)
        if turns and turns[-1].speaker == speaker:
            turns[-1].words.append(text)
            turns[-1].end = end
        elif is_jitter(turns, speaker, start):
            # word timestamps drift at speaker changes, so a lone word squeezed
            # between two turns by the same speaker belongs to that speaker
            lone = turns.pop()
            turns[-1].words += [*lone.words, text]
            turns[-1].end = end
        else:
            turns.append(Turn(speaker, start, end, [text]))

    return turns


def is_jitter(turns: list[Turn], speaker: str, start: float) -> bool:
    if len(turns) < 2 or len(turns[-1].words) > 1 or turns[-2].speaker != speaker:
        return False

    lone, before = turns[-1], turns[-2]
    return min(lone.start - before.end, start - lone.end) < JITTER_GAP


def best_speaker(
    segments: list[speakrs.Segment], index: int, start: float, end: float
) -> str:
    best, best_overlap = None, 0.0
    for segment in segments[index:]:
        if segment.start >= end:
            break

        overlap = min(segment.end, end) - max(segment.start, start)
        if overlap > best_overlap:
            best, best_overlap = segment.speaker, overlap

    if best is not None:
        return best

    # a word in a pause between speakers goes to whoever is closest in time
    nearby = segments[max(index - 1, 0) : index + 1]
    if not nearby:
        return "UNKNOWN"

    return min(nearby, key=lambda s: max(s.start - end, start - s.end)).speaker


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", help="WAV, FLAC, MP3, Ogg Vorbis, or M4A")
    parser.add_argument(
        "--mode", default=default_mode(), help="speakrs mode (default: %(default)s)"
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="faster-whisper model (default: %(default)s)",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    # load speakrs before whisper starts: in CUDA mode it preloads the cuDNN and
    # cuBLAS that the speakrs[cuda] extra installs, and whisper's GPU backend
    # cannot find those pip libraries on its own
    pipeline = speakrs.Pipeline.from_pretrained(args.mode)

    # speakrs releases the GIL while it runs, so diarization and transcription overlap
    with ThreadPoolExecutor(max_workers=2) as pool:
        diarization = pool.submit(
            timed, f"speakrs ({args.mode})", diarize, pipeline, args.audio
        )
        transcription = pool.submit(
            timed,
            f"whisper ({args.whisper_model})",
            transcribe,
            args.audio,
            args.whisper_model,
        )
        segments, words = diarization.result(), transcription.result()

    # number speakers in the order they first talk
    names: dict[str, str] = {}
    for turn in assign_speakers(words, segments):
        name = names.setdefault(turn.speaker, f"Speaker {len(names) + 1}")
        print(f"[{timestamp(turn.start)}] {name}: {''.join(turn.words).strip()}\n")

    print(
        f"{len(names)} speakers, {time.perf_counter() - started:.1f}s", file=sys.stderr
    )


if __name__ == "__main__":
    main()
