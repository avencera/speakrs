# Native weights and CPU references

## Runtime assets

The CPU and CUDA modes load two safetensors files instead of ONNX models. Export them, with the
PLDA and embedding metadata files they share with the other modes, into one directory:

```sh
uv run --no-project scripts/cuda/export_weights.py --runtime-assets <dir>
```

| File | Source model | Contents |
|---|---|---|
| `segmentation-3.0.safetensors` | `segmentation-3.0.onnx` | every FP32 initializer under its ONNX name: SincNet band edges, convolutions, normalizations, the 4 LSTM layers (`onnx::LSTM_*`, B/W/R per layer in ONNX gate order) and the 3 linear layers (`onnx::MatMul_*`, `[in, out]`) |
| `wespeaker-multimask-tail.safetensors` | `wespeaker-multimask-tail.onnx` | the `resnet.*` initializers: each convolution's `weight` and `weight_bias`, with batch normalization already folded in by `scripts/export_models.py`, and the `resnet.seg_1` embedding layer |
| `plda_*.npy`, `wespeaker-voxceleb-resnet34.min_num_samples.txt` | fixtures or Hugging Face | shared with every mode |

The ONNX sources come from `fixtures/models` or `--output` (default
`~/Library/Caches/speakrs-cuda-ref/models`), and are otherwise downloaded anonymously at
the `HF_REVISION` in `src/models.rs`. The export refuses a model that still has
`BatchNormalization` nodes. Load the directory with `OwnedDiarizationPipeline::from_dir`
or `PipelineBuilder::from_dir`; `ModelManager` requests the same two file names from
Hugging Face. `cargo xtask models export` writes them into `fixtures/models`, and
`cargo xtask models deploy` uploads them with the other models.

## Reference tensors

Run these commands from the repository root on the Mac:

```sh
uv run --no-project scripts/cuda/export_weights.py
uv run --no-project scripts/cuda/make_reference.py --self-test
uv run --no-project scripts/cuda/make_reference.py
uv run --no-project scripts/cuda/make_reference.py --verify
uv run --no-project scripts/cuda/export_weights.py
```

The last export adds observed shapes from the reference indices to the manifests.
The scripts read `HF_REVISION` from `src/models.rs`. They copy fixture models or
download missing public models without authentication. `--output` changes the
default directory: `~/Library/Caches/speakrs-cuda-ref`.

Each of the six model variants has a directory with FP32 initializer weights, a
graph manifest, and reference files with JSON indices. Original initializer names
are the weight keys. Integer constants also have native values in the manifest;
do not use their FP32 copies as ONNX indices. Constant-node values are in the
manifest and reference tensors. The manifests include nested graphs.

Reference keys use `input/<ONNX name>` and `tensor/<ONNX name>`. Final outputs have
the `output` role in the index. Intermediate integers and booleans keep their
native dtype. The index gives the node, graph scope, shape, dtype, byte count,
source rows, output ranges, and SHA-256. Optimizations are off during node capture.
Each final output is also checked against the unchanged, optimized ORT CPU graph.
`--verify` uses a fresh run and checks complete safetensors bytes and JSON data.
The temporary repeat file is deleted after the check.

## Input cases

- `test_first_b1`: the first full window of `fixtures/test.wav`
- `test_last_partial_b1`: the last partial window of `fixtures/test.wav`
- `test_short_partial_b1`: the zero-padded `fixtures/test_short.wav` window
- `test_and_short_b32`: all 18 test windows and the short window, then the first
  13 test windows again, to give 32 useful rows

The full batch contains both fixtures to limit storage. This is a reference batch,
not a change to pipeline routing. The Rust ORT path uses single calls for partial
batches. Each waveform has 160,000 FP32 samples; windows advance 16,000 samples.
Source lengths exclude zero padding when masks are selected. Tail masks have
shape `[B*3,589]`, with chunk order first and speaker order second. Hard powerset
decode uses the last class on an equal maximum. Activity below 10 gives a zero
mask. A clean mask removes overlap and is used only when its sum is greater than
`ceil(589*min_num_samples/audio_len)`. Fbank inputs come from the unchanged ORT graph.

The single segmentation graph has an `If`. The index lists the branch that ran.
Every output from that branch is captured in a separate CPU graph under
`branch/<If node>/<branch>/<ONNX name>`. The unused branch has no runtime values.
Its nodes and attributes remain in the manifest. Unknown static ranks have exact
runtime shapes in the reference index and observed shapes in the final manifest.

## Native implementation details

The ResNet batch normalization is already folded into Conv weights and biases.
The exporter can also add Conv/BatchNormalization folds without changing original
weights. Segmentation uses input-dependent InstanceNormalization, which cannot be
folded. Its first filters are built from learned cutoff parameters with Sin/Cos.
LSTM weights use ONNX gate order `[i,o,f,c]` and separate input/recurrent biases.

Fbank uses a 512-point one-sided DFT and replicate-edge preemphasis. These steps
need more than cuDNN Conv or cuBLAS GEMM.

Multi-mask pooling repeats trunk features through Reshape/Unsqueeze/Expand,
then uses Resize (nearest, asymmetric, floor) from 589 to 125 frames. Its core is
ReduceSum, Greater, Where, Mul, ReduceSum, Div, Unsqueeze, Sub, Pow, Pow,
ReduceSum, Div, Sub, Mul, ReduceSum, Div, Clip, Sqrt, Concat, LessOrEqual,
Expand, Tile, Where, Gemm. The batch-32 graph also uses Concat to build zero stats.
The variance denominator is `safe_sum - sum(weights²)/safe_sum`. The pinned graph
has no added epsilon at this step. Clip floors variance at `1e-10`. Zero masks
give zero means and `1e-5` standard deviations before the embedding head.

Allow about 15 GiB for one complete set and another 13 GiB during byte checking.
The batch-32 tail alone has about 12 GiB of tensor data. The writer streams buffers
and validates the file with the safetensors reader.

`python-lint` already checks all of `scripts`, including this directory. Use:

```sh
uv run --group dev ruff format scripts/cuda
uv run --group dev ty check --python .venv scripts/cuda
```

## Qualified custom kernels

The native CUDA backend uses the shipped sm75 cuda-oxide kernels only for qualified
(layer, batch, math) combinations. Other combinations use the Library path. The
qualified batch classes are 1, 7, 32, 33, and 64.

| Boundary | FP32 | TF32 |
| --- | --- | --- |
| ResNet stage 1 C32 and stage 2 strided convolution | All qualified batches | All qualified batches |
| ResNet stage 2 C64 convolutions | All qualified batches | 7, 32, 33, 64 |
| Four-layer bidirectional LSTM stack | All qualified batches | Library |
| SincNet convolution, absolute value, and pool | All qualified batches | Library |

Segmentation defaults to FP32. Embedding defaults to TF32. CUDA graphs are enabled,
and the Library LSTM default is `CudaLstmAlgorithm::PersistStaticSmallH`. The custom
LSTM still uses cuBLAS for input projections. cuDNN and cuBLAS remain required.

The shipped PTX targets Turing and newer. Qualification ran its sm75 image on an
RTX 5070 Ti (sm120); performance on a real Turing GPU is not measured. See
[the qualification guide](qualify/README.md) for the gates, cache setup, and lock
checks.
