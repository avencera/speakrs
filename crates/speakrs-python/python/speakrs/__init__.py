"""Fast speaker diarization with pyannote-level accuracy.

Prepare the models once, build a pipeline, and diarize files or 16 kHz mono
samples::

    import speakrs

    pipeline = speakrs.Pipeline.from_pretrained("cpu")
    result = pipeline.diarize_file("meeting.wav")
    print(result.rttm())

Inference runs in Rust with the GIL released, so other Python threads keep
running while a call is in progress.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from importlib import metadata
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Sequence, TypeVar, Union

from . import _cuda, _native
from ._errors import SpeakrsError
from ._native import (
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_REVISION,
    ActivityCleanup,
    AhcConfig,
    CacheEntry,
    CancelToken,
    ClusteringConfig,
    DiarizationResult,
    ModelManifest,
    ModelManifestEntry,
    PipelineConfig,
    PreparedModels,
    ProgressEvent,
    QueueResult,
    RuntimeConfig,
    Segment,
    TimingStats,
    VbxConfig,
    available_modes,
    cleanup_revision,
    default_cache_dir,
    generate_manifest,
    list_cache,
    required_model_files,
)

if TYPE_CHECKING:
    from typing import Any, Protocol

    from typing_extensions import Buffer

    class _ArrayInterface(Protocol):
        # NumPy only declares __buffer__ for Python 3.12+, so match its array
        # interface too
        @property
        def __array_interface__(self) -> dict[str, Any]: ...

    Samples = Union[Buffer, _ArrayInterface, Sequence[float]]

try:
    __version__ = metadata.version("speakrs")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

_T = TypeVar("_T")

# Queue.recv_async backs off between these, in seconds; a job takes far longer
# than the longest wait, so polling adds little latency
_RECV_ASYNC_POLL_MIN = 0.001
_RECV_ASYNC_POLL_MAX = 0.05


class ExecutionMode(str, Enum):
    """Inference backend and precision.

    Any API that takes a mode also accepts its value or a CLI spelling such as
    ``"cuda-fast"``; case, ``-``, and ``_`` are ignored.
    """

    CPU = "CPU"
    """Native Rust CPU inference, available in every build."""
    CoreML = "CoreML"
    """Apple CoreML with FP32 models (macOS builds)."""
    CoreMLFast = "CoreMLFast"
    """Apple CoreML with W8A16 segmentation and a 2 s step (macOS builds)."""
    CUDA = "CUDA"
    """Native NVIDIA backend (Linux x86_64 builds, Turing or newer GPU)."""
    CUDAFast = "CUDAFast"
    """Native NVIDIA backend with a 2 s step (Linux x86_64 builds)."""


class ProgressKind(str, Enum):
    """Values of :attr:`ProgressEvent.kind`."""

    DownloadingModel = "downloading_model"
    """A model file is about to be downloaded; ``name`` is the file."""
    DecodingAudio = "decoding_audio"
    """An audio file is being decoded."""
    ResamplingAudio = "resampling_audio"
    """Decoded audio is being resampled to 16 kHz."""
    RunningPipeline = "running_pipeline"
    """Inference and clustering are running."""
    BatchFile = "batch_file"
    """A batch started inference on a file; ``name`` is its file id and
    ``index`` its position."""
    Completed = "completed"
    """The operation finished."""


Mode = Union[ExecutionMode, str]
"""An :class:`ExecutionMode` or its name."""

ProgressCallback = Callable[[ProgressEvent], object]
"""Called with the GIL held while the operation runs.

It must not call a pipeline method. An exception it raises cancels the
operation, which then re-raises it.
"""

StrPath = Union[str, "PathLike[str]"]


def _mode(mode: Mode) -> str:
    return mode.value if isinstance(mode, ExecutionMode) else str(mode)


def prepare(
    mode: Mode = ExecutionMode.CPU,
    *,
    cache_dir: StrPath | None = None,
    model_dir: StrPath | None = None,
    manifest: ModelManifest | None = None,
    progress: ProgressCallback | None = None,
) -> PreparedModels:
    """Return a model directory with every file ``mode`` loads.

    Without ``model_dir``, missing files are downloaded from Hugging Face into
    ``cache_dir`` (default :func:`default_cache_dir`), which uses the hub cache
    layout. With ``model_dir``, the files must already be there. A ``manifest``
    from :func:`generate_manifest` also checks every file's size and SHA-256.

    Raises :class:`SpeakrsError` with category ``model_prepare`` when a download
    fails, or ``model_manifest`` when a file is missing or changed.
    """
    return _native._prepare(
        _mode(mode),
        Path(cache_dir) if cache_dir is not None else None,
        Path(model_dir) if model_dir is not None else None,
        manifest,
        progress,
    )


class Pipeline:
    """A loaded diarization pipeline for one execution mode.

    Calls on one pipeline run one at a time; build one pipeline per thread to run
    in parallel. Every call releases the GIL while it runs.
    """

    def __init__(self, native: _native._Pipeline) -> None:
        """Use :meth:`from_prepared` or :meth:`from_pretrained` instead."""
        self._native = native

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedModels,
        mode: Mode | None = None,
        *,
        pipeline_config: PipelineConfig | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> Pipeline:
        """Load the models in ``prepared``.

        ``mode`` defaults to the mode the models were prepared for.
        ``pipeline_config`` becomes the default for every call; without it,
        :meth:`PipelineConfig.for_mode` applies. Raises :class:`SpeakrsError`
        with category ``unsupported_runtime`` when this build lacks the mode, or
        ``model_load`` when the models or the GPU runtime cannot be loaded.
        """
        mode_name = _mode(mode) if mode is not None else prepared.mode
        _cuda.prepare_libraries(mode_name)
        return cls(
            _native._build_pipeline(prepared, mode_name, pipeline_config, runtime_config)
        )

    @classmethod
    def from_pretrained(
        cls,
        mode: Mode = ExecutionMode.CPU,
        *,
        cache_dir: StrPath | None = None,
        pipeline_config: PipelineConfig | None = None,
        runtime_config: RuntimeConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> Pipeline:
        """Download the models for ``mode`` if needed, then load them.

        Same as :func:`prepare` followed by :meth:`from_prepared`.
        """
        prepared = prepare(mode, cache_dir=cache_dir, progress=progress)
        return cls.from_prepared(
            prepared,
            mode,
            pipeline_config=pipeline_config,
            runtime_config=runtime_config,
        )

    @property
    def mode(self) -> ExecutionMode:
        """Mode the models were loaded for."""
        return ExecutionMode(self._native.mode)

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Config used when a call does not pass one."""
        return self._native.pipeline_config

    def diarize_samples(
        self,
        samples: Samples,
        *,
        file_id: str = "file1",
        pipeline_config: PipelineConfig | None = None,
        cancel_token: CancelToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> DiarizationResult:
        """Diarize 16 kHz mono audio.

        ``samples`` is a 1-D float32 buffer, such as a NumPy array, which is read
        in place without copying, or a sequence of floats, which is copied.
        Other buffer dtypes raise ``TypeError``. The buffer is read while the GIL
        is released, so it must not change until the call returns; writing it
        from another thread meanwhile is undefined behavior.
        """
        return self._native.diarize_samples(
            samples, file_id, pipeline_config, cancel_token, progress
        )

    def diarize_file(
        self,
        path: StrPath,
        *,
        file_id: str | None = None,
        pipeline_config: PipelineConfig | None = None,
        cancel_token: CancelToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> DiarizationResult:
        """Decode an audio file, resample it to 16 kHz mono, and diarize it.

        WAV, FLAC, MP3, Ogg Vorbis, and AAC in MP4/M4A are supported. The whole
        file is decoded into memory. ``file_id`` defaults to the file name
        without its extension.
        """
        path = Path(path)
        return self._native.diarize_file(
            path,
            file_id if file_id is not None else path.stem,
            pipeline_config,
            cancel_token,
            progress,
        )

    def diarize_files(
        self,
        paths: Sequence[StrPath],
        *,
        file_ids: Sequence[str] | None = None,
        pipeline_config: PipelineConfig | None = None,
        cancel_token: CancelToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[DiarizationResult]:
        """Diarize many audio files faster than calling :meth:`diarize_file` in a loop.

        Up to four threads decode files ahead of the one in inference, and one
        file's clustering overlaps the next file's inference. Results keep input
        order. The first failure raises and discards the batch. Cancellation and
        Ctrl+C take effect at the next file boundary. ``file_ids`` default to the
        file names without extensions.

        On CoreML, every file is decoded before inference starts, so memory grows
        with the whole batch.
        """
        paths = [Path(path) for path in paths]
        if file_ids is None:
            file_ids = [path.stem for path in paths]
        elif len(file_ids) != len(paths):
            raise ValueError(
                f"got {len(file_ids)} file_ids for {len(paths)} paths"
            )
        return self._native.diarize_files(
            list(zip(file_ids, paths)), pipeline_config, cancel_token, progress
        )

    async def diarize_samples_async(
        self, samples: Samples, **kwargs: object
    ) -> DiarizationResult:
        """:meth:`diarize_samples` on a worker thread.

        Cancelling the await cancels the call and returns only once the worker
        has stopped reading ``samples``, so the buffer can be reused afterwards.
        The worker stops at the next stage boundary, so that can take a while.
        """
        return await _run_on_worker(self.diarize_samples, samples, **kwargs)

    async def diarize_file_async(
        self, path: StrPath, **kwargs: object
    ) -> DiarizationResult:
        """:meth:`diarize_file` on a worker thread; cancelling the await cancels the call."""
        return await _run_on_worker(self.diarize_file, path, **kwargs)

    async def diarize_files_async(
        self, paths: Sequence[StrPath], **kwargs: object
    ) -> list[DiarizationResult]:
        """:meth:`diarize_files` on a worker thread; cancelling the await cancels the call."""
        return await _run_on_worker(self.diarize_files, paths, **kwargs)

    def into_queue(
        self,
        *,
        pipeline_config: PipelineConfig | None = None,
        capacity: int | None = None,
    ) -> Queue:
        """Move the pipeline onto a background worker fed by a bounded queue.

        This pipeline cannot be used afterwards. ``capacity`` (default 64) bounds
        the jobs waiting to run.
        """
        return Queue(self._native.into_queue(pipeline_config, capacity))



async def _run_on_worker(call: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    """Run a pipeline call on a worker thread, cancelling it with the await.

    A cancelled await waits for the worker before re-raising: the worker may be
    reading the caller's sample buffer, which the caller is free to reuse once the
    cancellation reaches it. The call runs on a child of any token the caller
    passed, so cancelling the await never cancels that token.
    """
    caller_token = kwargs.pop("cancel_token", None)
    if caller_token is not None and not isinstance(caller_token, CancelToken):
        raise TypeError("cancel_token must be a CancelToken")
    token = caller_token.child() if caller_token is not None else CancelToken()
    worker = asyncio.ensure_future(
        asyncio.to_thread(call, *args, cancel_token=token, **kwargs)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        token.cancel()
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                # a repeated cancel still has to wait for the buffer to be released
                pass
        if not worker.cancelled():
            worker.exception()  # retrieved, so asyncio does not log it as unhandled
        raise

class Queue:
    """A pipeline running on a background thread.

    Push jobs from any thread and receive results in completion order. Pushes
    never block: a full queue raises :class:`SpeakrsError` with code
    ``queue_full``. After :meth:`close`, receiving drains the remaining results
    and iteration stops. Using the queue as a context manager closes it on exit.
    """

    def __init__(self, native: _native._Queue) -> None:
        """Use :meth:`Pipeline.into_queue` instead."""
        self._native = native

    def push_samples(self, file_id: str, samples: Samples) -> int:
        """Submit 16 kHz mono samples and return the job id; the samples are copied."""
        return self._native.push_samples(file_id, samples)

    def push_file(self, file_id: str, path: StrPath) -> int:
        """Decode an audio file on this thread, submit it, and return the job id."""
        return self._native.push_file(file_id, Path(path))

    def close(self) -> None:
        """Stop accepting jobs; already submitted jobs still finish."""
        self._native.close()

    def recv(self) -> QueueResult:
        """Wait for the next finished job.

        Raises :class:`SpeakrsError` with code ``queue_closed`` once the queue is
        closed and drained. Ctrl+C interrupts the wait.
        """
        return self._native.recv()

    async def recv_async(self) -> QueueResult:
        """Wait for the next finished job without blocking the event loop.

        Cancelling the wait, for example with :func:`asyncio.wait_for`, never
        consumes a result. Raises like :meth:`recv` once the queue is closed and
        drained.
        """
        # poll on the event loop instead of awaiting a blocking recv on a worker
        # thread: a cancelled await would leave that thread waiting, and it would
        # take and drop the next result
        delay = _RECV_ASYNC_POLL_MIN
        while (result := self.try_recv()) is None:
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECV_ASYNC_POLL_MAX)
        return result

    def try_recv(self) -> QueueResult | None:
        """Return a finished job if one is ready, else ``None``; never blocks."""
        return self._native.try_recv()

    def __iter__(self) -> Iterator[QueueResult]:
        """Yield results until the queue is closed and drained."""
        while True:
            try:
                yield self.recv()
            except SpeakrsError as error:
                if error.code == "queue_closed":
                    return
                raise

    def __enter__(self) -> Queue:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_MODEL_REPOSITORY",
    "DEFAULT_MODEL_REVISION",
    "ActivityCleanup",
    "AhcConfig",
    "CacheEntry",
    "CancelToken",
    "ClusteringConfig",
    "DiarizationResult",
    "ExecutionMode",
    "Mode",
    "ModelManifest",
    "ModelManifestEntry",
    "Pipeline",
    "PipelineConfig",
    "PreparedModels",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressKind",
    "Queue",
    "QueueResult",
    "RuntimeConfig",
    "Segment",
    "SpeakrsError",
    "TimingStats",
    "VbxConfig",
    "available_modes",
    "cleanup_revision",
    "default_cache_dir",
    "generate_manifest",
    "list_cache",
    "prepare",
    "required_model_files",
]
