# xtask

Development CLI for speakrs. Two binaries:

- **`xtask`** -- local dev tasks (benchmarks, model management, comparisons)
- **`speakrs-bm`** -- GPU benchmark runner for dstack containers (requires `cuda` feature)

## Commands

| Command | Description |
|---------|-------------|
| `models` | Export ONNX models, CoreML conversion, deploy to HF |
| `fixtures` | Regenerate test fixtures via Python |
| `compare` | Side-by-side diarization comparisons (run, rttm, accuracy) |
| `bench` | Local benchmarks (run, compare, der) |
| `dstack` | Remote GPU benchmarks via dstack |
| `dataset` | Download/upload benchmark datasets |
| `diarize` | Run speaker diarization on WAV files |
| `profile-ort-embedding` | Profile ORT embedding inference strategies |
| `profile-stages` | Profile pipeline stages |

## Local benchmarks

```bash
# single-file timing: speakrs vs pyannote
cargo xtask bench run path/to/file.wav

# multi-tool comparison on one file
cargo xtask bench compare path/to/file.wav

# DER evaluation on a dataset
cargo xtask bench der --dataset voxconverse-dev --impls speakrs,pyannote

# single-file benchmark
cargo xtask bench der --file path/to/audio.wav --rttm path/to/ref.rttm --impls scm,sk
```

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

Use `bench-parallel` (alias `bp`) to fan out one dstack task per dataset, each on its own GPU:

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

Each task is named `{name}-{dataset}` (e.g. `my-run-voxconverse-dev`) and submitted with `--detach`. Results upload to `s3://speakrs/benchmarks/{name}-{dataset}/`.

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
