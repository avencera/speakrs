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
just check    # fmt + clippy + test
just test     # run tests (e2e tests require ONNX models)
just fmt      # cargo fmt + Python formatting
just clippy   # cargo clippy -- -D warnings + ty check
```

All Python scripts in the repo are fully typed and type checked with `ty`.
