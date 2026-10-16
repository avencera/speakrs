# speakrs Python API

Every public function and class has a docstring, and the package ships type
stubs (`py.typed`, `_native.pyi`). This page is an overview.

## Lifecycle

```python
import speakrs

prepared = speakrs.prepare("cpu")                    # download or locate models
pipeline = speakrs.Pipeline.from_prepared(prepared)  # load them
result = pipeline.diarize_file("meeting.wav")
print(result.rttm())
```

`Pipeline.from_pretrained(mode)` does both steps at once.

## Execution modes

`ExecutionMode` is a `str` enum. Every API that takes a mode also accepts its
value or a CLI spelling: case, `-`, and `_` are ignored, so `"cuda-fast"`
names `CUDAFast`.

| Mode | Backend | Built into |
| --- | --- | --- |
| `CPU` | native Rust | every wheel |
| `CoreML`, `CoreMLFast` | Apple CoreML | macOS arm64 wheel |
| `CUDA`, `CUDAFast` | native NVIDIA backend, Turing or newer | Linux x86_64 wheel |

`available_modes()` lists the modes compiled into the installed wheel. A CUDA
mode can still fail at load time without an NVIDIA driver; that raises
`SpeakrsError` with category `model_load`.

## Models and cache

- `prepare(mode="CPU", *, cache_dir=None, model_dir=None, manifest=None, progress=None) -> PreparedModels`
- `generate_manifest(model_dir, mode="CPU") -> ModelManifest`, with `to_json()` and `ModelManifest.from_json()`
- `required_model_files(mode="CPU") -> list[str]`
- `default_cache_dir() -> Path`: `$SPEAKRS_CACHE_DIR`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`
- `list_cache(cache_dir=None) -> list[CacheEntry]`
- `cleanup_revision(revision, cache_dir=None) -> bool`
- `DEFAULT_MODEL_REPOSITORY`, `DEFAULT_MODEL_REVISION`

The cache uses the Hugging Face hub layout, so it is shared with the Rust
crate's `ModelManager`. `HF_ENDPOINT` selects a mirror.

## Pipeline

- `Pipeline.from_prepared(prepared, mode=None, *, pipeline_config=None, runtime_config=None)`
- `Pipeline.from_pretrained(mode="CPU", *, cache_dir=None, pipeline_config=None, runtime_config=None, progress=None)`
- `pipeline.mode`, `pipeline.pipeline_config`
- `diarize_samples(samples, *, file_id="file1", pipeline_config=None, cancel_token=None, progress=None)`
- `diarize_file(path, *, file_id=None, ...)`; `file_id` defaults to the file name without its extension
- `diarize_files(paths, *, file_ids=None, ...)`: the batch entry point
- `diarize_samples_async`, `diarize_file_async`, `diarize_files_async`: cancelling the await cancels the call and returns once the worker has stopped, so a sample buffer can be reused afterwards
- `into_queue(*, pipeline_config=None, capacity=None) -> Queue`

`samples` is 16 kHz mono audio. A 1-D float32 buffer, such as a NumPy array,
is read in place without a copy; a sequence of floats is copied. Any other
buffer dtype raises `TypeError`, and more than one dimension raises
`ValueError`.

The buffer is read while the GIL is released, so it must not change until the
call returns: writing it from another thread during the call is undefined
behavior. Pass `samples.copy()` if another thread may write the array.
Copying 8.7 minutes of audio costs about 4 ms on a Linux server, 2-3% of a
CUDA-fast run on an H100, which is why the binding does not copy for you.

Every call releases the GIL while it runs. Calls on one pipeline run one at a
time, so build a pipeline per thread to run several in parallel.

`diarize_files` overlaps the work of a batch: up to four threads decode files
ahead of inference, and one file's clustering runs during the next file's
inference. Results keep input order. The first failure raises and discards the
batch. A `CancelToken` or Ctrl+C stops the batch at the next file boundary.
On CoreML, the batch decodes every file before inference starts, so memory
grows with the whole batch (about 230 MB per hour of audio), and cancellation
only applies while files are decoding.

File decoding supports WAV, FLAC, MP3, Ogg Vorbis, and AAC in MP4/M4A. Each
file is decoded into memory whole.

## Queue

- `queue.push_samples(file_id, samples) -> int`, `queue.push_file(file_id, path) -> int`
- `queue.recv()`, `queue.recv_async()`, `queue.try_recv()`; cancelling a `recv_async()` wait never consumes a result
- `queue.close()`, `for result in queue: ...`, `with pipeline.into_queue() as queue: ...`

Pushes never block. A full queue raises `SpeakrsError` with code `queue_full`.
After `close()`, receiving drains the remaining results and then raises with
code `queue_closed`, which also ends iteration. Ctrl+C interrupts a waiting
`recv()`.

## Results

`DiarizationResult` has these fields and methods:

- `segments` and `exclusive_segments`, both `list[Segment]`. Exclusive segments
  have at most one speaker at a time; each overlap goes to the speaker with the
  highest activation.
- `speakers`, `file_id`, `duration`, `mode`, `model_revision`
- `timing`, a `TimingStats`, or `None` for batch and queue results
- `rttm(file_id=None, *, exclusive=False)`

`Segment` has `start`, `end`, `speaker`, and `duration`.

`QueueResult` has `job_id`, `file_id`, `ok`, `result`, `error`, and `unwrap()`.

## Configuration

These are frozen value objects; omitted arguments keep the defaults.

- `PipelineConfig(activity=None, clustering=None, merge_gap=None, reconstruct_method=None, smoothing_epsilon=None)`
  and `PipelineConfig.for_mode(mode)`
- `ActivityCleanup(min_active_frames=0, max_inactive_gap_frames=0, pad_before_frames=0, pad_after_frames=0)`
- `ClusteringConfig(ahc=None, vbx=None, speaker_keep_threshold=None)`
- `AhcConfig(threshold=None)`
- `VbxConfig(fa=None, fb=None, max_iters=None, epsilon=None, init_smoothing=None)`
- `RuntimeConfig(*, coreml_compute_units=None, cuda_segmentation_math=None, cuda_embedding_math=None, cuda_graphs=None)`

Out-of-range values raise `SpeakrsError` with category `configuration`. Setting
a `RuntimeConfig` option whose backend the wheel lacks raises it with category
`unsupported_runtime`.

## Progress and cancellation

A `progress` callback receives `ProgressEvent` objects with these fields:

- `kind`, one of the `ProgressKind` values: `downloading_model`,
  `decoding_audio`, `resampling_audio`, `running_pipeline`, `batch_file`, or
  `completed`
- `name`
- `index`

The callback runs with the GIL held while the call holds its pipeline, so it
must not call a pipeline method; doing so raises `SpeakrsError`. An exception
the callback raises cancels the call, which then re-raises it.

`CancelToken().cancel()` or Ctrl+C stops a call at the next stage or file
boundary. A stage that has already started finishes first.
A call that fails or is interrupted never cancels the token passed in, and
`token.child()` gives a token that one call can cancel without affecting the
others sharing the parent.

## Errors

Every failure inside speakrs raises `SpeakrsError`, which has these fields:

- `category`, one of the stable values below
- `code`, a finer stable value
- `message`

| Category | Meaning |
| --- | --- |
| `configuration` | invalid argument or config value, or unknown mode |
| `unsupported_runtime` | the wheel lacks the mode or backend option |
| `model_prepare` | download or cache failure |
| `model_manifest` | a model file is missing or does not match the manifest |
| `model_load` | the models or the GPU runtime could not be loaded |
| `audio_decode` | the file could not be opened or decoded |
| `audio_resample` | the audio could not be resampled to 16 kHz |
| `pipeline` | inference, clustering, or reconstruction failed |
| `queue` | queue failure; codes `queue_full` and `queue_closed` |
| `cancelled` | a `CancelToken` stopped the call; Ctrl+C raises `KeyboardInterrupt` instead |
| `internal` | a bug in the bindings |

Wrong argument types raise `TypeError`.

## CLI

```sh
speakrs diarize meeting.wav --mode cpu
speakrs diarize a.wav b.flac c.mp3 --mode cuda --exclusive > out.rttm
```

`--model-dir`, `--cache-dir`, and `--file-id` work as they do in the Python
API. Errors print `error: <message>` to stderr and exit with status 1.
