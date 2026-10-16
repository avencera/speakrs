"""Type stubs for the compiled ``speakrs._native`` extension."""

from os import PathLike
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, final

from typing_extensions import Buffer

class _ArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> dict[str, Any]: ...

_Samples = Buffer | _ArrayInterface | Sequence[float]
_Progress = Callable[[ProgressEvent], object] | None
_StrPath = str | PathLike[str]

DEFAULT_MODEL_REPOSITORY: str
"""Hugging Face repository that hosts the models."""
DEFAULT_MODEL_REVISION: str
"""Repository revision this release loads."""

@final
class CancelToken:
    """Cancels a running diarization from any thread.

    The check runs between stages and between batch files; a stage that has
    already started finishes first.
    """

    def __init__(self) -> None: ...
    def cancel(self) -> None:
        """Request cancellation."""
    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested."""
    def child(self) -> CancelToken:
        """A token cancelled along with this one, whose own cancellation leaves this one untouched."""

@final
class ActivityCleanup:
    """Frame-count cleanup applied to each speaker's activity after clustering.

    Frames are 16.875 ms apart. All zeros, the default for full-precision modes,
    leaves activity unchanged.
    """

    def __init__(
        self,
        min_active_frames: int = 0,
        max_inactive_gap_frames: int = 0,
        pad_before_frames: int = 0,
        pad_after_frames: int = 0,
    ) -> None: ...
    @property
    def min_active_frames(self) -> int:
        """Shortest active run kept, in frames."""
    @property
    def max_inactive_gap_frames(self) -> int:
        """Longest inactive gap filled between two active runs, in frames."""
    @property
    def pad_before_frames(self) -> int:
        """Frames added before each active run."""
    @property
    def pad_after_frames(self) -> int:
        """Frames added after each active run."""

@final
class AhcConfig:
    """Agglomerative clustering that seeds VBx."""

    def __init__(self, threshold: float | None = None) -> None:
        """Raises SpeakrsError (``configuration``) for a negative or non-finite threshold."""
    @property
    def threshold(self) -> float:
        """Distance threshold for merging clusters."""

@final
class VbxConfig:
    """Variational Bayes (VBx) speaker clustering."""

    def __init__(
        self,
        fa: float | None = None,
        fb: float | None = None,
        max_iters: int | None = None,
        epsilon: float | None = None,
        init_smoothing: float | None = None,
    ) -> None:
        """Omitted values keep the defaults.

        ``init_smoothing`` below zero starts from hard AHC assignments, zero from
        uniform responsibilities, and above zero from softmax-smoothed ones.
        Raises SpeakrsError (``configuration``) for out-of-range values.
        """
    @property
    def fa(self) -> float:
        """Speaker-model scale."""
    @property
    def fb(self) -> float:
        """Between-speaker scale."""
    @property
    def max_iters(self) -> int:
        """Maximum iterations."""
    @property
    def epsilon(self) -> float:
        """Convergence threshold on the evidence lower bound."""
    @property
    def init_smoothing(self) -> float:
        """Responsibility initialization: -1.0 hard, 0.0 uniform, > 0 smoothed."""

@final
class ClusteringConfig:
    """AHC, VBx, and the speaker-keep threshold."""

    def __init__(
        self,
        ahc: AhcConfig | None = None,
        vbx: VbxConfig | None = None,
        speaker_keep_threshold: float | None = None,
    ) -> None:
        """Raises SpeakrsError (``configuration``) for a negative or non-finite threshold."""
    @property
    def ahc(self) -> AhcConfig:
        """Agglomerative clustering settings."""
    @property
    def vbx(self) -> VbxConfig:
        """VBx settings."""
    @property
    def speaker_keep_threshold(self) -> float:
        """Minimum speaker activity weight that keeps a speaker."""

@final
class PipelineConfig:
    """Post-inference settings: activity cleanup, clustering, segment merging, and reconstruction."""

    def __init__(
        self,
        activity: ActivityCleanup | None = None,
        clustering: ClusteringConfig | None = None,
        merge_gap: float | None = None,
        reconstruct_method: str | None = None,
        smoothing_epsilon: float | None = None,
    ) -> None:
        """Omitted values keep the full-precision defaults.

        ``reconstruct_method`` is ``"smoothed"`` (default) or ``"standard"``;
        ``smoothing_epsilon`` only applies to ``"smoothed"``.
        """
    @staticmethod
    def for_mode(mode: str) -> PipelineConfig:
        """The defaults a pipeline uses for ``mode``."""
    @property
    def activity(self) -> ActivityCleanup:
        """Activity cleanup after clustering."""
    @property
    def clustering(self) -> ClusteringConfig:
        """Clustering settings."""
    @property
    def merge_gap(self) -> float:
        """Largest gap in seconds between two segments of one speaker that is merged."""
    @property
    def reconstruct_method(self) -> str:
        """``"smoothed"`` or ``"standard"``."""
    @property
    def smoothing_epsilon(self) -> float | None:
        """Score difference below which smoothing keeps the previous speaker."""

@final
class RuntimeConfig:
    """Backend-specific runtime settings.

    Setting an option for a backend this build lacks raises SpeakrsError
    (``unsupported_runtime``); reading one returns ``None``.
    """

    def __init__(
        self,
        *,
        coreml_compute_units: str | None = None,
        cuda_segmentation_math: str | None = None,
        cuda_embedding_math: str | None = None,
        cuda_graphs: bool | None = None,
    ) -> None: ...
    @property
    def coreml_compute_units(self) -> str | None:
        """``"all"``, ``"cpu_and_neural_engine"``, or ``"cpu_only"``."""
    @property
    def cuda_segmentation_math(self) -> str | None:
        """``"fp32"`` (default) or ``"tf32"``."""
    @property
    def cuda_embedding_math(self) -> str | None:
        """``"tf32"`` (default) or ``"fp32"``."""
    @property
    def cuda_graphs(self) -> bool | None:
        """Whether CUDA modes replay captured CUDA graphs."""

@final
class Segment:
    """One speaker turn."""

    def __init__(self, start: float, end: float, speaker: str) -> None: ...
    @property
    def start(self) -> float:
        """Start time in seconds."""
    @property
    def end(self) -> float:
        """End time in seconds."""
    @property
    def speaker(self) -> str:
        """Speaker label such as ``SPEAKER_00``."""
    @property
    def duration(self) -> float:
        """``end - start`` in seconds."""

@final
class TimingStats:
    """Wall-clock time per stage of a single-file call, in milliseconds."""

    @property
    def audio_decode_ms(self) -> float:
        """Decoding the audio file; zero for in-memory samples."""
    @property
    def audio_resample_ms(self) -> float:
        """Resampling to 16 kHz; zero when the audio was already 16 kHz."""
    @property
    def pipeline_ms(self) -> float:
        """Inference, clustering, and reconstruction."""
    @property
    def total_ms(self) -> float:
        """The whole call."""

@final
class DiarizationResult:
    """Speaker turns for one input."""

    @property
    def file_id(self) -> str:
        """Identifier used in RTTM output."""
    @property
    def duration(self) -> float:
        """Input duration in seconds."""
    @property
    def segments(self) -> list[Segment]:
        """Speaker turns in time order; overlapping speech gives overlapping segments."""
    @property
    def exclusive_segments(self) -> list[Segment]:
        """Speaker turns with at most one speaker at a time."""
    @property
    def speakers(self) -> list[str]:
        """Sorted, distinct speaker labels."""
    @property
    def mode(self) -> str:
        """Execution mode that produced the result, such as ``"CPU"``."""
    @property
    def model_revision(self) -> str | None:
        """Model revision, or ``None`` for an unverified local model directory."""
    @property
    def timing(self) -> TimingStats | None:
        """Stage timings for single-file calls; ``None`` for batch and queue results."""
    def rttm(self, file_id: str | None = None, *, exclusive: bool = False) -> str:
        """Render RTTM, one ``SPEAKER`` line per segment."""
    def __len__(self) -> int: ...

@final
class QueueResult:
    """A finished queue job."""

    @property
    def job_id(self) -> int:
        """Id returned by the push that submitted the job."""
    @property
    def file_id(self) -> str:
        """File id given at push time."""
    @property
    def ok(self) -> bool:
        """Whether the job succeeded."""
    @property
    def result(self) -> DiarizationResult | None:
        """The result, or ``None`` if the job failed."""
    @property
    def error(self) -> Exception | None:
        """The SpeakrsError the job failed with, or ``None``."""
    def unwrap(self) -> DiarizationResult:
        """Return the result, or raise the job's SpeakrsError."""

@final
class ProgressEvent:
    """A progress report passed to a ``progress`` callback."""

    @property
    def kind(self) -> str:
        """One of the ``ProgressKind`` values."""
    @property
    def name(self) -> str | None:
        """Model file for ``downloading_model``, file id for ``batch_file``."""
    @property
    def index(self) -> int | None:
        """Zero-based input position for ``batch_file``."""

@final
class ModelManifestEntry:
    """Size and SHA-256 digest of one model file."""

    @property
    def path(self) -> str:
        """Path relative to the model directory."""
    @property
    def size_bytes(self) -> int:
        """File size in bytes."""
    @property
    def sha256(self) -> str:
        """Lowercase hex SHA-256 digest."""

@final
class ModelManifest:
    """Sizes and digests of the files one mode loads."""

    @staticmethod
    def from_json(json: str) -> ModelManifest:
        """Parse a manifest written by ``to_json()``."""
    def to_json(self) -> str:
        """Serialize as JSON."""
    @property
    def repository(self) -> str:
        """Repository the files came from."""
    @property
    def revision(self) -> str:
        """Repository revision the files came from."""
    @property
    def files(self) -> list[ModelManifestEntry]:
        """One entry per file, sorted by path."""

@final
class PreparedModels:
    """A model directory that holds every file one mode loads."""

    @property
    def model_dir(self) -> Path:
        """Directory the pipeline loads from."""
    @property
    def mode(self) -> str:
        """Mode the directory was prepared for."""
    @property
    def revision(self) -> str | None:
        """Model revision, or ``None`` for an unverified local directory."""
    @property
    def manifest(self) -> ModelManifest | None:
        """Manifest the files were verified against, if one was given."""

@final
class CacheEntry:
    """One cached revision of the model repository."""

    @property
    def repository(self) -> str:
        """Hugging Face repository id."""
    @property
    def revision(self) -> str:
        """Revision (commit hash)."""
    @property
    def path(self) -> Path:
        """Snapshot directory."""

@final
class _Pipeline:
    @property
    def mode(self) -> str: ...
    @property
    def pipeline_config(self) -> PipelineConfig: ...
    def diarize_samples(
        self,
        samples: _Samples,
        file_id: str,
        pipeline_config: PipelineConfig | None = None,
        cancel_token: CancelToken | None = None,
        progress: _Progress = None,
    ) -> DiarizationResult: ...
    def diarize_file(
        self,
        path: _StrPath,
        file_id: str,
        pipeline_config: PipelineConfig | None = None,
        cancel_token: CancelToken | None = None,
        progress: _Progress = None,
    ) -> DiarizationResult: ...
    def diarize_files(
        self,
        files: Sequence[tuple[str, _StrPath]],
        pipeline_config: PipelineConfig | None = None,
        cancel_token: CancelToken | None = None,
        progress: _Progress = None,
    ) -> list[DiarizationResult]: ...
    def into_queue(
        self, pipeline_config: PipelineConfig | None = None, capacity: int | None = None
    ) -> _Queue: ...

@final
class _Queue:
    def push_samples(self, file_id: str, samples: _Samples) -> int: ...
    def push_file(self, file_id: str, path: _StrPath) -> int: ...
    def close(self) -> None: ...
    def recv(self) -> QueueResult: ...
    def try_recv(self) -> QueueResult | None: ...

def _build_pipeline(
    prepared: PreparedModels,
    mode: str,
    pipeline_config: PipelineConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> _Pipeline: ...
def _prepare(
    mode: str,
    cache_dir: _StrPath | None = None,
    model_dir: _StrPath | None = None,
    manifest: ModelManifest | None = None,
    progress: _Progress = None,
) -> PreparedModels: ...
def generate_manifest(model_dir: _StrPath, mode: str = "CPU") -> ModelManifest:
    """Hash every file ``mode`` loads from ``model_dir``."""

def required_model_files(mode: str = "CPU") -> list[str]:
    """Files ``mode`` loads, relative to the model directory."""

def default_cache_dir() -> Path:
    """``$SPEAKRS_CACHE_DIR``, else ``$HF_HOME/hub``, else ``~/.cache/huggingface/hub``."""

def list_cache(cache_dir: _StrPath | None = None) -> list[CacheEntry]:
    """Cached revisions of the model repository."""

def cleanup_revision(revision: str, cache_dir: _StrPath | None = None) -> bool:
    """Delete one cached revision and the files no other revision shares."""

def available_modes() -> list[str]:
    """Execution modes compiled into this build."""
