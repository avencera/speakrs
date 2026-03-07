# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compare two RTTM files and report differences."""

import sys
from collections import defaultdict


def parse_rttm(path):
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append((start, dur, speaker))
    return segments


def total_speech(segments):
    return sum(dur for _, dur, _ in segments)


def speaker_stats(segments):
    by_speaker = defaultdict(float)
    for _, dur, spk in segments:
        by_speaker[spk] += dur
    return dict(by_speaker)


def timeline_overlap(segs_a, segs_b):
    """Compute how much of segs_a's speech time is covered by segs_b (any speaker)"""
    # build sorted intervals for b
    intervals_b = sorted([(s, s + d) for s, d, _ in segs_b])
    if not intervals_b:
        return 0.0

    # merge overlapping intervals in b
    merged = [intervals_b[0]]
    for start, end in intervals_b[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    covered = 0.0
    for s, d, _ in segs_a:
        seg_start, seg_end = s, s + d
        for b_start, b_end in merged:
            if b_start >= seg_end:
                break
            if b_end <= seg_start:
                continue
            overlap_start = max(seg_start, b_start)
            overlap_end = min(seg_end, b_end)
            covered += max(0, overlap_end - overlap_start)
    return covered


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_rttm.py <rust.rttm> <python.rttm>", file=sys.stderr)
        sys.exit(1)

    rust_segs = parse_rttm(sys.argv[1])
    py_segs = parse_rttm(sys.argv[2])

    rust_total = total_speech(rust_segs)
    py_total = total_speech(py_segs)
    rust_speakers = speaker_stats(rust_segs)
    py_speakers = speaker_stats(py_segs)

    print(f"{'':30s} {'Rust':>10s} {'Python':>10s}")
    print(f"{'─' * 52}")
    print(f"{'Segments':30s} {len(rust_segs):10d} {len(py_segs):10d}")
    print(f"{'Speakers':30s} {len(rust_speakers):10d} {len(py_speakers):10d}")
    print(f"{'Total speech (s)':30s} {rust_total:10.1f} {py_total:10.1f}")
    print()

    print("Per-speaker duration (s):")
    print(f"  Rust:   {rust_speakers}")
    print(f"  Python: {py_speakers}")
    print()

    # speech overlap analysis
    if rust_total > 0 and py_total > 0:
        rust_covered = timeline_overlap(rust_segs, py_segs)
        py_covered = timeline_overlap(py_segs, rust_segs)
        print(f"{'Rust speech covered by Python':30s} {rust_covered / rust_total * 100:9.1f}%")
        print(f"{'Python speech covered by Rust':30s} {py_covered / py_total * 100:9.1f}%")

        # time range
        all_segs = rust_segs + py_segs
        audio_end = max(s + d for s, d, _ in all_segs)
        print(f"{'Audio span (s)':30s} {audio_end:10.1f}")


if __name__ == "__main__":
    main()
