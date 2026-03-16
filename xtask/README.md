# xtask

`xtask` holds the project automation and benchmarking commands for `speakrs`.

## Common commands

```bash
# list commands
cargo run -p xtask -- --help

# benchmark on a local machine
cargo run --release -p xtask -- benchmark run audio.wav --rust-mode cpu

# manage remote GPU instances
cargo run -p xtask -- gpu create my-gpu
cargo run -p xtask -- gpu setup my-gpu --branch master
cargo run -p xtask -- gpu ssh my-gpu
```

## DER benchmarks

```bash
# run DER evaluation
cargo xtask benchmark der --dataset voxconverse-dev --impls pmps,scm,scmf,fa

# list available implementations
cargo xtask benchmark der --impls list
```

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

The `speakrs-bm` binary (GPU benchmarks) uses the same `--impls` syntax with its own subset (`sg`, `pg`).

