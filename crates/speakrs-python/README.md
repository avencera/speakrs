# speakrs

speakrs does speaker diarization with pyannote-level accuracy. Inference runs in Rust
on the CPU, on Apple CoreML, or on NVIDIA CUDA, with no PyTorch or ONNX Runtime.

```sh
pip install speakrs
```

| Platform | Modes |
| --- | --- |
| macOS arm64 | `CPU`, `CoreML`, `CoreMLFast` |
| Linux x86_64 (glibc 2.28+) | `CPU`, `CUDA`, `CUDAFast` |

The package needs Python 3.11 or newer. One wheel covers every Python version.

## Use

```python
import speakrs

pipeline = speakrs.Pipeline.from_pretrained("cpu")  # downloads the models once
result = pipeline.diarize_file("meeting.wav")

for segment in result.segments:
    print(f"{segment.start:7.2f} {segment.end:7.2f} {segment.speaker}")

print(result.rttm())
```

To diarize many files, use the batch method. It decodes files ahead of
inference and overlaps clustering with the next file:

```python
results = pipeline.diarize_files(["a.wav", "b.flac", "c.mp3"])
```

Audio that is already in memory has to be 16 kHz mono float32. A NumPy array is
read in place, without a copy:

```python
result = pipeline.diarize_samples(samples, file_id="call-42")
```

The command line prints RTTM:

```sh
speakrs diarize meeting.wav --mode coreml
```

See [API.md](https://github.com/avencera/speakrs/blob/master/crates/speakrs-python/API.md)
for the full API: the queue, async methods, configuration, progress events,
cancellation, and error categories.

## CUDA

The Linux wheel includes the CUDA backend. It needs an NVIDIA driver and a
Turing (RTX 20, T4) or newer GPU, but no CUDA toolkit. On a machine without a
GPU, the wheel still imports and runs on the CPU.

speakrs runs its own kernels for every layer. cuDNN 9 and cuBLAS are only a
fallback for a layer that has no kernel for your GPU. If you want that
fallback, install it with:

```sh
pip install "speakrs[cuda]"
```

## Models

`prepare()` and `from_pretrained()` download the models from the
[`avencera/speakrs-models`](https://huggingface.co/avencera/speakrs-models)
Hugging Face repository. They go into the Hugging Face hub cache, which is set
by `SPEAKRS_CACHE_DIR` or `HF_HOME`.

To skip the download, pass `model_dir=` to use a local directory instead.
