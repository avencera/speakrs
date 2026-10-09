# Contributing

Set up the Python tool environment once:

```sh
uv sync --group dev
```

Export model assets and PLDA parameters (requires a [HuggingFace token](https://huggingface.co/settings/tokens) with access to the gated repos):

```sh
# accept terms at:
#   https://huggingface.co/pyannote/segmentation-3.0
#   https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
HF_TOKEN=your_token just export-models
```

Models are saved to `fixtures/models/` (gitignored).

CPU and CUDA inference use `segmentation-3.0.safetensors` and
`wespeaker-multimask-tail.safetensors`, plus PLDA parameters and
`wespeaker-voxceleb-resnet34.min_num_samples.txt`. The export also produces ONNX
graphs for MIGraphX and reference checks. Native CPU inference does not use them.

To use another model directory for integration tests, set
`SPEAKRS_MODEL_FIXTURE_DIR`. Missing or invalid model assets fail the tests.

Regenerate golden test fixtures from Python (requires `HF_TOKEN`):

```sh
just generate-fixtures
```

```sh
just check    # fmt + lint + test
just test     # run tests (e2e tests require model assets)
just check-cpu-dependencies  # keep ORT out of CPU dependency trees
just fmt      # cargo fmt, Python formatting, and README regeneration when cargo-rdme is installed
just lint     # cargo clippy + Python ty checks
just clippy   # cargo clippy -- -D warnings
just python-lint  # ty check across the root and Python subprojects
```

The README section between the `cargo-rdme` markers is generated from the `src/lib.rs` doc comments with [cargo-rdme](https://github.com/orium/cargo-rdme). Edit the crate docs, not that part of the README. CI fails if the README and the crate docs differ. To regenerate it locally, install cargo-rdme and its intralink toolchain once:

```sh
cargo install cargo-rdme --version 2.2.2 --locked
cargo rdme install-rust-toolchain-for-intralinks
```

After that, `just fmt` regenerates the README, or you can run `cargo rdme` directly.

Sync the Python environments before running `just python-lint`:

```sh
uv sync --group dev
uv sync --project scripts/native_coreml
uv sync --project scripts/pyannote-bench
```

Python type checks run with `ty` in each script's intended environment.
