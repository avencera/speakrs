# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow", "soundfile"]
# ///
"""Extract wav + rttm from a HuggingFace parquet dataset.

Usage: uv run scripts/extract_hf_dataset.py <parquet_dir> <wav_dir> <rttm_dir> [--split test]

Parquet schema (argmaxinc convention):
  audio: struct<bytes: binary, path: string>
  timestamps_start: list<double>
  timestamps_end: list<double>
  speakers: list<string>
"""

import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def extract(parquet_dir: Path, wav_dir: Path, rttm_dir: Path, split: str = "test"):
    wav_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not files:
        # fall back to any parquet files
        files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {parquet_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting {len(files)} parquet file(s) from {parquet_dir}")
    count = 0

    for pf in files:
        table = pq.read_table(pf)
        for i in range(len(table)):
            row = table.slice(i, 1).to_pydict()

            audio_bytes = row["audio"][0]["bytes"]
            audio_path = row["audio"][0].get("path", "")
            starts = row["timestamps_start"][0]
            ends = row["timestamps_end"][0]
            speakers = row["speakers"][0]

            # derive file id from audio path or index
            if audio_path:
                file_id = Path(audio_path).stem
            else:
                file_id = f"{pf.stem}_{i:04d}"

            wav_path = wav_dir / f"{file_id}.wav"
            if not wav_path.exists():
                # decode audio bytes and write as 16kHz mono wav
                with io.BytesIO(audio_bytes) as buf:
                    data, sr = sf.read(buf)

                # convert to mono if needed
                if data.ndim > 1:
                    data = data.mean(axis=1)

                # resample to 16kHz if needed
                if sr != 16000:
                    import numpy as np

                    ratio = 16000 / sr
                    n_samples = int(len(data) * ratio)
                    indices = np.arange(n_samples) / ratio
                    indices_floor = indices.astype(int)
                    indices_floor = np.clip(indices_floor, 0, len(data) - 1)
                    data = data[indices_floor]
                    sr = 16000

                sf.write(str(wav_path), data, sr)

            # write rttm
            rttm_path = rttm_dir / f"{file_id}.rttm"
            if not rttm_path.exists():
                lines = []
                for start, end, spk in zip(starts, ends, speakers):
                    dur = end - start
                    if dur > 0:
                        lines.append(
                            f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
                        )
                rttm_path.write_text("\n".join(lines) + "\n")

            count += 1

    print(f"Extracted {count} files")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print(f"Usage: {sys.argv[0]} <parquet_dir> <wav_dir> <rttm_dir> [--split test]")
        sys.exit(1)

    parquet_dir = Path(args[0])
    wav_dir = Path(args[1])
    rttm_dir = Path(args[2])
    split = "test"
    if "--split" in args:
        idx = args.index("--split")
        split = args[idx + 1]

    extract(parquet_dir, wav_dir, rttm_dir, split)
