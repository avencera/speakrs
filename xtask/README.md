# xtask

Development CLI for speakrs. Two binaries:

- **`xtask`** -- local dev tasks (benchmarks, model management, comparisons)
- **`speakrs-bm`** -- GPU benchmark runner for dstack containers (requires `cuda` feature)

## Commands

| Command | Description |
|---------|-------------|
| `models` | Export ONNX models, CoreML conversion, deploy to HF |
| `fixtures` | Regenerate test fixtures via Python |
| `compare rttm` | Informal RTTM timeline comparison |
| `benchmark run` | Measure implementations and write schema version 3 results |
| `benchmark score` | Score a stored schema version 3 run |
| `dstack` | Remote GPU benchmarks via dstack |
| `dataset` | Download/upload benchmark datasets |
| `diarize` | Run speaker diarization on WAV files |
| `profile-ort-embedding` | Profile ORT embedding inference strategies |
| `profile-stages` | Profile pipeline stages |

## Local benchmarks

```bash
# DER evaluation on a dataset
cargo xtask benchmark run --dataset voxconverse-dev --impls cpu,pyannote-cpu

# single-file benchmark
cargo xtask benchmark run --file path/to/audio.wav --rttm path/to/ref.rttm --impls scm,sk

# score a stored schema version 3 run (recalculates stored RTTM hypotheses)
cargo xtask benchmark score _benchmarks/20240101-010203
```

Benchmark records use schema version 3. The previous schema version 2 flat
`results.json` writer format remains readable through an explicit conversion to
the typed version 3 record. Version 3 is intentional because the typed record
shape is not wire-compatible with the old flat shape. New benchmark runs,
reads, and score reports use schema version 3.

`benchmark score` reads the reference and hypothesis RTTM locations recorded in
the run. It does not run inference and it supports a suite root with one
recorded child per dataset. It writes `score.json` at the run root. Repeating
the command atomically replaces `score.json` and leaves `results.json`,
hypotheses, and references unchanged. Stored-run scoring
currently supports `collar_seconds=0` and `ignore_overlap=false`; records with
other scoring options are rejected instead of being scored with different
rules. Failed and skipped implementations remain visible in the score report.

### DER implementation aliases

`--impls` accepts comma-separated full names or aliases:

| Alias | Full name | Description |
|-------|-----------|-------------|
| `pmps` | `pyannote` | pyannote MPS |
| `pcpu` | `pyannote-cpu` | pyannote CPU |
| `pg` | `pyannote-cuda` | pyannote CUDA |
| `scm` | `coreml` | speakrs CoreML |
| `scmf` | `coreml-fast` | speakrs CoreML Fast |
| `sg` | `cuda` | speakrs CUDA |
| `scpu` | `cpu` | speakrs CPU |
| `fa` | `fluidaudio` | FluidAudio |
| `prs` | `pyannote-rs` | pyannote-rs |

## Remote GPU benchmarks (dstack)

### Prerequisites

Set these env vars before running dstack commands:

- `AWS_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `HF_TOKEN`

### Single dataset (sequential)

```bash
# run one dataset on one GPU
cargo xtask dstack bench my-run --dataset voxconverse-dev

# run ALL datasets sequentially on one GPU
cargo xtask dstack bench my-run --dataset all

# detach immediately (don't follow logs)
cargo xtask dstack bench my-run --dataset all -d
```

### Multiple datasets in parallel (one GPU each)

Use `bench-parallel` (alias `bp`) to start one dstack task per dataset, each on its own GPU:

```bash
# all 9 datasets, each on its own GPU
cargo xtask dstack bp my-run

# specific datasets
cargo xtask dstack bp my-run --dataset voxconverse-dev,ami-ihm,earnings21

# dataset aliases work too
cargo xtask dstack bp my-run --dataset vd,ai,e21

# reuse existing fleet
cargo xtask dstack bp my-run -R
```

Each task is named `{name}-{dataset}` (for example `my-run-voxconverse-dev`) and submitted with `--detach`. Results upload to `s3://speakrs/benchmarks/{name}-{dataset}/`.

The fleet supports up to 8 concurrent GPU nodes (`nodes: 0..8` in `fleet.yml`).

### Fleet management

```bash
cargo xtask dstack fleet              # provision reusable GPU pool (30min idle)
cargo xtask dstack ps                 # show all runs
cargo xtask dstack logs my-run        # stream logs
cargo xtask dstack attach my-run      # reattach (logs + port forwarding)
cargo xtask dstack stop my-run        # stop a run (alias: kill)
```

### Results

```bash
cargo xtask dstack download my-run    # fetch from S3 to _benchmarks/my-run/
cargo xtask dstack delete benchmarks/my-run  # delete from S3 (with confirmation)
```

## Datasets

| ID | Alias | Name |
|----|-------|------|
| `voxconverse-dev` | `vd` | VoxConverse Dev |
| `voxconverse-test` | `vt` | VoxConverse Test |
| `ami-ihm` | `ai` | AMI IHM |
| `ami-sdm` | `as` | AMI SDM |
| `aishell4` | `a4` | AISHELL-4 |
| `earnings21` | `e21` | Earnings-21 |
| `alimeeting` | `ali` | AliMeeting |
| `ava-avd` | `ava` | AVA-AVD |
| `icsi` | | ICSI |

## GPU implementations (speakrs-bm)

The `speakrs-bm` binary uses the same `--impls` syntax with its GPU subset:

| CLI ID | Alias | Description |
|--------|-------|-------------|
| `speakrs` | `sg` | speakrs CUDA (fused, 1s step) |
| `speakrs-fast` | `sgf` | speakrs CUDA Fast (fused, 2s step) |
| `pyannote` | `pg` | pyannote CUDA |

## WavLM bridge experiments

`wavlm-bridge` is the bounded B3 adapter for a verified Python WavLM
segmentation bundle. It never loads a segmentation model. The bridge validates
the bundle, canonical 16 kHz audio, reference, and UEM identities before it
loads the Speakrs embedding and PLDA assets.

```bash
cargo xtask wavlm-bridge validate \
  --bundle /path/to/bundle \
  --audio /path/to/canonical.wav \
  --output validation.json

cargo xtask wavlm-bridge run \
  --spec bridge-spec.json \
  --models-dir /path/to/speakrs-models \
  --mode cpu \
  --output-dir _benchmarks/wavlm-bridge/run-001 \
  --cache-dir _benchmarks/wavlm-bridge/cache \
  --recipe reference

cargo xtask wavlm-bridge report \
  --spec bridge-spec.json \
  --unchanged-speakrs unchanged.json \
  --frozen-python frozen-python.json \
  --hybrid _benchmarks/wavlm-bridge/run-001/systems/hybrid_wavlm_speakrs/reference.json \
  --output-dir _benchmarks/wavlm-bridge/report-001
```

`run` writes `run.json`, one strict hybrid system manifest per recipe, five
stage dependency receipts per recording, `speaker_tracks.json`, and the exact
RTTM returned by the imported library path. Output and cache directories are
immutable. A cache entry is reusable only when its completion marker, hashes,
receipt chain, stage keys, geometry, and recording/recipe identities all pass
validation. A stale or partial entry is an error.

The cache has a separate immutable embedding-stage namespace. A recipe that
changes only clustering or reconstruction reuses the exact decoded masks and
typed per-slot embedding snapshot, then runs downstream stages again. The run
record marks final-output reuse with `cache_reused` and embedding-stage reuse
with `embedding_cache_reused`.

The runtime identity must bind the B2 fixed embedding asset
`wespeaker-voxceleb-resnet34-fixed.onnx`, its adjacent
`wespeaker-voxceleb-resnet34-fixed.embedding.json` sidecar, and the PLDA
directory. The bridge admits only `cpu` and, in CUDA builds, `cuda`; legacy
fast and CoreML modes are not bridge modes.

`report` only joins typed system manifests. It requires the unchanged Speakrs,
frozen Python WavLM, and hybrid systems to use identical membership, audio,
reference, UEM, and pyannote.metrics 4.0.0 scorer identities (zero collar,
overlap included, automatic speaker count). It does not calculate DER, JER, or
any other score. Missing score documents remain `unavailable` and make the
report `incomplete`; no absent score is replaced or claimed. An available score
document must use score schema version 1, contain finite per-record components,
counts, diagnostics, runtime, and peak-memory evidence, and cover every source,
domain, parent, equal-domain, and pooled aggregate. Invalid or incomplete score
documents are rejected.
