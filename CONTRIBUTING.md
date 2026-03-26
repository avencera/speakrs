# Contributing

Set up the Python tool environment once:

```sh
uv sync --group dev
```

Download ONNX models and PLDA parameters (requires a [HuggingFace token](https://huggingface.co/settings/tokens) with access to the gated repos):

```sh
# accept terms at:
#   https://huggingface.co/pyannote/segmentation-3.0
#   https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
HF_TOKEN=your_token just export-models
```

Models are saved to `fixtures/models/` (gitignored).

Regenerate golden test fixtures from Python (requires `HF_TOKEN`):

```sh
just generate-fixtures
```

```sh
just check    # fmt + lint + test
just test     # run tests (e2e tests require ONNX models)
just fmt      # cargo fmt + Python formatting
just lint     # cargo clippy + Python ty checks
just clippy   # cargo clippy -- -D warnings
just python-lint  # ty check across the root and Python subprojects
```

The README is generated from `src/lib.rs` doc comments via [cargo-rdme](https://github.com/orium/cargo-rdme). After editing crate docs, regenerate:

```sh
cargo rdme
```

Sync the Python environments before running `just python-lint`:

```sh
uv sync --group dev
uv sync --project scripts/native_coreml
uv sync --project scripts/pyannote-bench
```

Python type checks run with `ty` in each script's intended environment.
