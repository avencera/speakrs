# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyannote.audio>=3.3",
#     "torch>=2.6",
#     "numpy",
#     "onnx",
#     "onnxscript",
#     "onnxruntime",
# ]
# ///
"""Download and export ONNX models + PLDA params for speakrs.

Args: models_dir (path to fixtures/models/)
Env: HF_TOKEN (HuggingFace token with model access)

Requires accepting terms at:
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
"""

import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import get_mel_banks

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


FIXED_EMBEDDING_FILENAME = "wespeaker-voxceleb-resnet34-fixed.onnx"
FIXED_SAMPLE_RATE = 16_000
FIXED_WINDOW_SAMPLES = 128_000
FIXED_MASK_FRAMES = 399
FIXED_EMBEDDING_WIDTH = 256
FIXED_RESNET_FRAMES = 100

# this covers the float32 reduction and FFT round-off seen when comparing the
# frozen source model with the exported wrapper on CPU
PARITY_ATOL = 3e-5
PARITY_RTOL = 1e-5


@dataclass(frozen=True)
class EmbeddingSidecar:
    """Strict metadata written beside the fixed-shape embedding wrapper."""

    schema_version: int
    onnx_sha256: str
    sample_rate: int
    window_samples: int
    mask_frames: int
    embedding_width: int
    frontend: str
    pooling: str
    interpolation: str
    resnet_frames: int
    precision: str
    min_num_samples: int

    def json_bytes(self) -> bytes:
        """Serialize the sidecar with only fields admitted by Rust."""

        return (
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")


class FixedEmbeddingExportError(RuntimeError):
    """Raised when a fixed-shape embedding asset cannot be qualified."""


def embedding_sidecar_path(model_path: str | os.PathLike[str]) -> Path:
    """Return the strict sidecar path adjacent to an ONNX model."""

    return Path(model_path).with_suffix(".embedding.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_embedding_sidecar(
    model_path: str | os.PathLike[str], min_num_samples: int
) -> EmbeddingSidecar:
    """Build strict metadata from the validated wrapper and source threshold."""

    threshold = _validate_min_num_samples(min_num_samples)
    return EmbeddingSidecar(
        schema_version=1,
        onnx_sha256=_sha256_file(Path(model_path)),
        sample_rate=FIXED_SAMPLE_RATE,
        window_samples=FIXED_WINDOW_SAMPLES,
        mask_frames=FIXED_MASK_FRAMES,
        embedding_width=FIXED_EMBEDDING_WIDTH,
        frontend="wespeaker_fbank_v1",
        pooling="masked_stats_pool_v1",
        interpolation="nearest",
        resnet_frames=FIXED_RESNET_FRAMES,
        precision="float32",
        min_num_samples=threshold,
    )


def _validate_min_num_samples(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FixedEmbeddingExportError(
            "the admitted WeSpeaker source metadata must provide a positive integer "
            "min_num_samples"
        )
    return value


def admitted_min_num_samples(pipeline: Any) -> int:
    """Read the positive threshold from the already admitted pipeline metadata."""

    embedding = getattr(pipeline, "_embedding", None)
    if embedding is None or not hasattr(embedding, "min_num_samples"):
        raise FixedEmbeddingExportError(
            "the admitted WeSpeaker source metadata is missing min_num_samples"
        )
    return _validate_min_num_samples(getattr(embedding, "min_num_samples"))


def fbank_frame_count(num_samples: int) -> int:
    """Return snip-edges WeSpeaker fbank frames for a sample count."""

    if num_samples < 400:
        return 0
    return (num_samples - 400) // 160 + 1


def resnet_frame_count(fbank_frames: int) -> int:
    """Return the current ResNet34 time frames for an fbank frame count."""

    frames = fbank_frames
    for _ in range(3):
        frames = (frames + 1) // 2
    return frames


def resize_mask_weights(weights: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Resize mask weights with the PyTorch nearest-neighbor policy."""

    return F.interpolate(
        weights.unsqueeze(1), size=target_frames, mode="nearest"
    ).squeeze(1)


class FbankWrapper(nn.Module):
    """Current WeSpeaker frontend expressed with exportable PyTorch operators."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        del model
        self.scale = float(1 << 15)
        self.preemph = 0.97

        window = torch.hamming_window(400, periodic=False, alpha=0.54, beta=0.46)
        mel, _ = get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)

        self.register_buffer("window", window)
        self.register_buffer("mel", F.pad(mel, (0, 1), value=0.0).T.contiguous())
        self.register_buffer("eps", torch.tensor(torch.finfo(torch.float32).eps))

    def compute_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        window = cast(torch.Tensor, self.window)
        mel_filters = cast(torch.Tensor, self.mel)
        eps = cast(torch.Tensor, self.eps)

        frames = waveforms[:, 0, :] * self.scale
        frames = frames.unfold(1, 400, 160)
        frames = frames - frames.mean(dim=2, keepdim=True)

        previous = F.pad(frames, (1, 0), mode="replicate")[..., :-1]
        frames = frames - self.preemph * previous
        frames = frames * window.view(1, 1, -1)
        frames = F.pad(frames, (0, 112))

        spectrum = torch.fft.rfft(frames, dim=2).abs().pow(2.0)
        mel = torch.matmul(spectrum, mel_filters.to(dtype=spectrum.dtype))
        mel = torch.clamp_min(mel, eps.to(device=mel.device, dtype=mel.dtype)).log()
        return mel - mel.mean(dim=1, keepdim=True)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.compute_fbank(waveforms)


NUM_SPEAKERS = 3


class EmbeddingTailWrapper(nn.Module):
    """Existing legacy embedding tail with its established zero-mask behavior."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.resnet = model.resnet

    def pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.unsqueeze(1)
        num_frames = sequences.size(-1)
        if weights.size(-1) != num_frames:
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        weight_sum = weights.sum(dim=2)
        safe_sum = torch.where(
            weight_sum > 0.0, weight_sum, torch.ones_like(weight_sum)
        )
        mean = torch.sum(sequences * weights, dim=2) / safe_sum
        dx2 = torch.square(sequences - mean.unsqueeze(2))
        weight_sq_sum = torch.square(weights).sum(dim=2)
        denom = safe_sum - weight_sq_sum / safe_sum + 1e-8
        var = torch.sum(dx2 * weights, dim=2) / denom
        std = torch.sqrt(torch.clamp_min(var, 1e-10))

        stats = torch.cat([mean, std], dim=-1)
        zero_stats = torch.cat(
            [torch.zeros_like(mean), torch.full_like(std, 1e-5)], dim=-1
        )
        zero_mask = (weight_sum <= 0.0).repeat(1, stats.size(1))
        return torch.where(zero_mask, zero_stats, stats)

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        frames = self.resnet.forward_frames(fbank)
        frames = frames.reshape(
            frames.size(0), frames.size(1) * frames.size(2), frames.size(3)
        )
        stats = self.pool(frames, weights)
        embed_a = self.resnet.seg_1(stats)
        if self.resnet.two_emb_layer:
            out = F.relu(embed_a)
            out = self.resnet.seg_bn_1(out)
            return self.resnet.seg_2(out)

        return embed_a


class MultiMaskTailWrapper(nn.Module):
    """fbanks [B, 998, 80] + masks [B*3, 589] -> embeddings [B*3, 256]"""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.resnet = model.resnet
        self.num_speakers = NUM_SPEAKERS

    def pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.unsqueeze(1)
        num_frames = sequences.size(-1)
        if weights.size(-1) != num_frames:
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        weight_sum = weights.sum(dim=2)
        safe_sum = torch.where(
            weight_sum > 0.0, weight_sum, torch.ones_like(weight_sum)
        )
        mean = torch.sum(sequences * weights, dim=2) / safe_sum
        dx2 = torch.square(sequences - mean.unsqueeze(2))
        weight_sq_sum = torch.square(weights).sum(dim=2)
        denom = safe_sum - weight_sq_sum / safe_sum + 1e-8
        var = torch.sum(dx2 * weights, dim=2) / denom
        std = torch.sqrt(torch.clamp_min(var, 1e-10))

        stats = torch.cat([mean, std], dim=-1)
        zero_stats = torch.cat(
            [torch.zeros_like(mean), torch.full_like(std, 1e-5)], dim=-1
        )
        zero_mask = (weight_sum <= 0.0).repeat(1, stats.size(1))
        return torch.where(zero_mask, zero_stats, stats)

    def forward(self, fbank: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        frames = self.resnet.forward_frames(fbank)
        batch_size = frames.size(0)
        channels = frames.size(1) * frames.size(2)
        time_frames = frames.size(3)
        frames = frames.reshape(batch_size, channels, time_frames)
        frames = torch.repeat_interleave(frames, self.num_speakers, dim=0)
        stats = self.pool(frames, masks)
        embed_a = self.resnet.seg_1(stats)
        if self.resnet.two_emb_layer:
            out = F.relu(embed_a)
            out = self.resnet.seg_bn_1(out)
            return self.resnet.seg_2(out)
        return embed_a


class FixedEmbeddingTailWrapper(nn.Module):
    """Per-speaker tail preserving the source WeSpeaker StatsPool equations."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.resnet = model.resnet

    def pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.unsqueeze(1)
        num_frames = sequences.size(-1)
        if weights.size(-1) != num_frames:
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        # keep the source StatsPool v1 epsilon placement and unbiased denominator
        weight_sum = weights.sum(dim=2) + 1e-8
        mean = torch.sum(sequences * weights, dim=2) / weight_sum
        dx2 = torch.square(sequences - mean.unsqueeze(2))
        weight_sq_sum = torch.square(weights).sum(dim=2)
        denom = weight_sum - weight_sq_sum / weight_sum + 1e-8
        # keep the source epsilon while preventing ONNX cancellation for one-frame masks
        variance = torch.sum(dx2 * weights, dim=2) / torch.clamp_min(denom, 1e-8)
        standard_deviation = torch.sqrt(torch.clamp_min(variance, 1e-10))
        return torch.cat([mean, standard_deviation], dim=1)

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        frames = self.resnet.forward_frames(fbank)
        frames = frames.reshape(
            frames.size(0), frames.size(1) * frames.size(2), frames.size(3)
        )
        stats = self.pool(frames, weights)
        embed_a = self.resnet.seg_1(stats)
        if self.resnet.two_emb_layer:
            out = F.relu(embed_a)
            out = self.resnet.seg_bn_1(out)
            return self.resnet.seg_2(out)

        return embed_a


class ExactEmbeddingWrapper(nn.Module):
    """Compose an exported frontend and tail without changing legacy shapes."""

    def __init__(self, fbank_model: nn.Module, tail_model: nn.Module) -> None:
        super().__init__()
        self.fbank_model = fbank_model
        self.tail_model = tail_model

    def forward(self, waveforms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.tail_model(self.fbank_model(waveforms), weights)


class FixedEmbeddingWrapper(ExactEmbeddingWrapper):
    """Compose the fixed-shape frontend and source-compatible per-speaker tail."""

    def __init__(self, source_model: Any) -> None:
        super().__init__(
            FbankWrapper(source_model), FixedEmbeddingTailWrapper(source_model)
        )


def fixed_parity_masks(min_num_samples: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """Return deterministic masks covering the fixed embedding policy cases."""

    threshold = math.ceil(
        FIXED_MASK_FRAMES
        * _validate_min_num_samples(min_num_samples)
        / FIXED_WINDOW_SAMPLES
    )
    sparse_count = min(FIXED_MASK_FRAMES, max(1, threshold + 3))
    fallback_count = min(FIXED_MASK_FRAMES, max(1, threshold))
    clean_count = min(FIXED_MASK_FRAMES, max(threshold + 1, sparse_count))

    all_one = torch.ones((1, FIXED_MASK_FRAMES), dtype=torch.float32)
    sparse = torch.zeros_like(all_one)
    sparse[0, :sparse_count] = 1.0

    overlap_clean = torch.zeros_like(all_one)
    clean_start = (FIXED_MASK_FRAMES - clean_count) // 2
    overlap_clean[0, clean_start : clean_start + clean_count] = 1.0

    overlap_fallback = torch.zeros_like(all_one)
    overlap_fallback[0, :fallback_count] = 1.0

    return (
        ("all_one", all_one),
        ("sparse", sparse),
        ("overlap_clean_selected", overlap_clean),
        ("overlap_fallback_selected", overlap_fallback),
    )


def fixed_parity_waveform() -> torch.Tensor:
    """Return the deterministic waveform used by every fixed-wrapper probe."""

    return torch.linspace(
        -0.75,
        0.75,
        FIXED_WINDOW_SAMPLES,
        dtype=torch.float32,
    ).reshape(1, 1, FIXED_WINDOW_SAMPLES)


def compare_fixed_wrapper_parity(
    source_model: nn.Module,
    wrapper: nn.Module,
    min_num_samples: int,
) -> None:
    """Compare fixed wrapper and source weights on deterministic masks and audio."""

    threshold = _validate_min_num_samples(min_num_samples)
    waveform = fixed_parity_waveform()
    source_model.eval()
    wrapper.eval()

    with torch.inference_mode():
        for mask_name, weights in fixed_parity_masks(threshold):
            expected = source_model(waveform, weights=weights)
            actual = wrapper(waveform, weights)
            if expected.shape != (1, FIXED_EMBEDDING_WIDTH):
                raise FixedEmbeddingExportError(
                    f"source embedding has shape {tuple(expected.shape)}, expected "
                    f"(1, {FIXED_EMBEDDING_WIDTH}) for {mask_name} parity"
                )
            if not torch.allclose(
                actual,
                expected,
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
            ):
                max_diff = (actual - expected).abs().max().item()
                raise FixedEmbeddingExportError(
                    f"fixed embedding parity failed for {mask_name}: "
                    f"max_abs_diff={max_diff:.6g}, atol={PARITY_ATOL}, rtol={PARITY_RTOL}"
                )


def compare_fixed_onnx_parity(
    model_path: Path,
    source_model: nn.Module,
    min_num_samples: int,
) -> None:
    """Compare staged ONNX Runtime vectors with the admitted PyTorch source."""

    threshold = _validate_min_num_samples(min_num_samples)
    try:
        onnxruntime: Any = importlib.import_module("onnxruntime")
        session: Any = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise FixedEmbeddingExportError(
            f"could not create ONNX Runtime session for {model_path}: {error}"
        ) from error

    input_names = {value.name for value in session.get_inputs()}
    output_names = {value.name for value in session.get_outputs()}
    if input_names != {"waveform", "weights"} or output_names != {"output"}:
        raise FixedEmbeddingExportError(
            "staged fixed wrapper has unexpected ONNX Runtime IO names: "
            f"inputs={sorted(input_names)!r}, outputs={sorted(output_names)!r}"
        )

    waveform = fixed_parity_waveform()
    waveform_array = waveform.numpy()
    source_model.eval()
    with torch.inference_mode():
        for mask_name, weights in fixed_parity_masks(threshold):
            expected_tensor = source_model(waveform, weights=weights)
            expected = expected_tensor.detach().cpu().numpy()
            if expected.shape != (1, FIXED_EMBEDDING_WIDTH):
                raise FixedEmbeddingExportError(
                    f"source embedding shape is {expected.shape}, expected "
                    f"(1, {FIXED_EMBEDDING_WIDTH}) for {mask_name}"
                )
            output_values = session.run(
                ["output"],
                {"waveform": waveform_array, "weights": weights.numpy()},
            )
            if len(output_values) != 1:
                raise FixedEmbeddingExportError(
                    f"ONNX Runtime returned {len(output_values)} outputs for {mask_name}"
                )
            actual = np.asarray(output_values[0])
            if actual.dtype != np.float32:
                raise FixedEmbeddingExportError(
                    f"ONNX Runtime output is {actual.dtype}, expected float32"
                )
            if actual.shape != (1, FIXED_EMBEDDING_WIDTH):
                raise FixedEmbeddingExportError(
                    f"ONNX Runtime output shape is {actual.shape}, expected "
                    f"(1, {FIXED_EMBEDDING_WIDTH}) for {mask_name}"
                )
            if not np.isfinite(actual).all():
                raise FixedEmbeddingExportError(
                    f"ONNX Runtime output is not finite for {mask_name}"
                )
            if not np.allclose(
                actual,
                expected,
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
            ):
                max_diff = np.max(np.abs(actual - expected))
                raise FixedEmbeddingExportError(
                    f"ONNX Runtime parity failed for {mask_name}: "
                    f"max_abs_diff={max_diff:.6g}, atol={PARITY_ATOL}, rtol={PARITY_RTOL}"
                )


def _validate_fixed_wrapper_geometry(source_model: Any, wrapper: nn.Module) -> None:
    waveform = torch.zeros((1, 1, FIXED_WINDOW_SAMPLES), dtype=torch.float32)
    weights = torch.ones((1, FIXED_MASK_FRAMES), dtype=torch.float32)
    with torch.inference_mode():
        fbank = cast(FbankWrapper, wrapper.fbank_model)(waveform)
        frames = source_model.resnet.forward_frames(fbank)
        output = wrapper(waveform, weights)

    expected_fbank_frames = fbank_frame_count(FIXED_WINDOW_SAMPLES)
    if fbank.shape != (1, expected_fbank_frames, 80):
        raise FixedEmbeddingExportError(
            f"fixed frontend produced shape {tuple(fbank.shape)}, expected "
            f"(1, {expected_fbank_frames}, 80)"
        )
    expected_resnet_frames = resnet_frame_count(expected_fbank_frames)
    if (
        frames.shape[-1] != expected_resnet_frames
        or expected_resnet_frames != FIXED_RESNET_FRAMES
    ):
        raise FixedEmbeddingExportError(
            f"fixed ResNet produced {frames.shape[-1]} time frames, expected "
            f"{FIXED_RESNET_FRAMES}"
        )
    if output.shape != (1, FIXED_EMBEDDING_WIDTH):
        raise FixedEmbeddingExportError(
            f"fixed wrapper produced shape {tuple(output.shape)}, expected "
            f"(1, {FIXED_EMBEDDING_WIDTH})"
        )


def _validate_exported_onnx(path: Path) -> None:
    """Check the staged model has strict static primary shapes before publish."""

    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def tensor_shape(value: Any) -> tuple[int, ...]:
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise FixedEmbeddingExportError(
                f"staged fixed wrapper tensor `{value.name}` is not float32"
            )
        return tuple(d.dim_value for d in tensor_type.shape.dim)

    inputs = {value.name: tensor_shape(value) for value in model.graph.input}
    outputs = {value.name: tensor_shape(value) for value in model.graph.output}
    if set(inputs) != {"waveform", "weights"}:
        raise FixedEmbeddingExportError(
            f"staged fixed wrapper inputs are {sorted(inputs)!r}"
        )
    if inputs.get("waveform") != (1, 1, FIXED_WINDOW_SAMPLES):
        raise FixedEmbeddingExportError(
            f"staged fixed wrapper waveform shape is {inputs.get('waveform')!r}"
        )
    if inputs.get("weights") != (1, FIXED_MASK_FRAMES):
        raise FixedEmbeddingExportError(
            f"staged fixed wrapper weights shape is {inputs.get('weights')!r}"
        )
    if set(outputs) != {"output"} or next(iter(outputs.values()), ()) != (
        1,
        FIXED_EMBEDDING_WIDTH,
    ):
        raise FixedEmbeddingExportError(
            f"staged fixed wrapper output shapes are {outputs!r}"
        )


def _validate_published_sidecar(
    model_path: Path, sidecar_path: Path, min_num_samples: int
) -> None:
    if not model_path.is_file():
        raise FixedEmbeddingExportError(
            f"published fixed wrapper is missing: {model_path}"
        )
    if not sidecar_path.is_file():
        raise FixedEmbeddingExportError(
            f"published fixed wrapper sidecar is missing: {sidecar_path}"
        )

    expected = build_embedding_sidecar(model_path, min_num_samples).json_bytes()
    actual = sidecar_path.read_bytes()
    if actual != expected:
        raise FixedEmbeddingExportError(
            "published fixed wrapper sidecar does not match the model hash or schema"
        )


def _publish_fixed_embedding_pair(
    staging_model: Path,
    staging_sidecar: Path,
    output_path: Path,
    sidecar_path: Path,
    min_num_samples: int,
) -> None:
    # publish the sidecar first so the model replacement is the hash-admission commit point
    os.replace(staging_sidecar, sidecar_path)
    os.replace(staging_model, output_path)
    _validate_published_sidecar(output_path, sidecar_path, min_num_samples)


def export_fixed_embedding_model(
    source_model: nn.Module,
    models_dir: str | os.PathLike[str],
    min_num_samples: int,
) -> Path:
    """Qualify and publish the fixed per-speaker WeSpeaker embedding wrapper."""

    threshold = _validate_min_num_samples(min_num_samples)
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / FIXED_EMBEDDING_FILENAME
    sidecar_path = embedding_sidecar_path(output_path)
    wrapper = FixedEmbeddingWrapper(source_model).eval()

    _validate_fixed_wrapper_geometry(source_model, wrapper)
    compare_fixed_wrapper_parity(source_model, wrapper, threshold)

    with tempfile.TemporaryDirectory(
        dir=output_dir,
        prefix=f".{output_path.stem}.staging-",
    ) as staging_name:
        staging_dir = Path(staging_name)
        staging_model = staging_dir / output_path.name
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (
                    torch.zeros((1, 1, FIXED_WINDOW_SAMPLES), dtype=torch.float32),
                    torch.ones((1, FIXED_MASK_FRAMES), dtype=torch.float32),
                ),
                staging_model,
                input_names=["waveform", "weights"],
                output_names=["output"],
                opset_version=18,
                dynamo=True,
                external_data=False,
            )
        _validate_exported_onnx(staging_model)
        compare_fixed_onnx_parity(staging_model, source_model, threshold)

        staging_sidecar = staging_dir / sidecar_path.name
        staging_sidecar.write_bytes(
            build_embedding_sidecar(staging_model, threshold).json_bytes()
        )
        _validate_published_sidecar(staging_model, staging_sidecar, threshold)

        _publish_fixed_embedding_pair(
            staging_model,
            staging_sidecar,
            output_path,
            sidecar_path,
            threshold,
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"  {output_path.name} ({size_mb:.1f} MB)")
    print(f"  {sidecar_path.name}")
    return output_path


def export_fixed_embedding(pipeline: Any, models_dir: str | os.PathLike[str]) -> Path:
    """Export a fixed wrapper from the pipeline's admitted WeSpeaker model."""

    threshold = admitted_min_num_samples(pipeline)
    source_model = getattr(getattr(pipeline, "_embedding", None), "model_", None)
    if source_model is None:
        raise FixedEmbeddingExportError(
            "the admitted WeSpeaker source metadata is missing model_"
        )
    return export_fixed_embedding_model(source_model, models_dir, threshold)


def main() -> None:
    models_dir = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    os.makedirs(models_dir, exist_ok=True)

    print("Loading community-1 pipeline...")
    from pyannote.audio import Pipeline

    from_pretrained: Any = Pipeline.from_pretrained
    if token:
        try:
            pipeline = from_pretrained(
                "pyannote/speaker-diarization-community-1", token=token
            )
        except TypeError:
            try:
                legacy_kwargs: dict[str, Any] = {"use_auth_token": token}
                pipeline = from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    **legacy_kwargs,
                )
            except TypeError:
                pipeline = from_pretrained("pyannote/speaker-diarization-community-1")
    else:
        pipeline = from_pretrained("pyannote/speaker-diarization-community-1")
    assert pipeline is not None
    pipeline.to(torch.device("cpu"))

    export_segmentation(pipeline, models_dir)
    export_embedding(pipeline, models_dir)
    export_plda(models_dir)

    print("Done!")


def export_segmentation(pipeline: Any, models_dir: str) -> None:
    print("Exporting segmentation model...")
    seg_model = pipeline._segmentation.model
    seg_model.eval()

    dummy = torch.randn(1, 1, 160000)
    with torch.no_grad():
        torch.onnx.export(
            seg_model,
            (dummy,),
            os.path.join(models_dir, "segmentation-3.0.onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {2: "samples"}, "output": {1: "frames"}},
            opset_version=14,
            dynamo=False,
        )
        torch.onnx.export(
            seg_model,
            (torch.randn(32, 1, 160000),),
            os.path.join(models_dir, "segmentation-3.0-b32.onnx"),
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            dynamo=False,
        )
        torch.onnx.export(
            seg_model,
            (torch.randn(64, 1, 160000),),
            os.path.join(models_dir, "segmentation-3.0-b64.onnx"),
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            dynamo=False,
        )

    sz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0.onnx")) / 1e6
    print(f"  segmentation-3.0.onnx ({sz:.1f} MB)")
    bsz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0-b32.onnx")) / 1e6
    print(f"  segmentation-3.0-b32.onnx ({bsz:.1f} MB)")
    b64sz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0-b64.onnx")) / 1e6
    print(f"  segmentation-3.0-b64.onnx ({b64sz:.1f} MB)")


def export_embedding(pipeline: Any, models_dir: str) -> None:
    """Export the exact WeSpeaker embedding path for batch-1 and batch-32 inference"""
    print("Exporting embedding model...")
    min_num_samples = admitted_min_num_samples(pipeline)

    emb_model = pipeline._embedding.model_
    emb_model.eval()
    fbank_wrapper = FbankWrapper(emb_model)
    fbank_wrapper.eval()
    tail_wrapper = EmbeddingTailWrapper(emb_model)
    tail_wrapper.eval()
    multi_mask_wrapper = MultiMaskTailWrapper(emb_model)
    multi_mask_wrapper.eval()

    dummy_waveform = torch.randn(1, 1, 160000)
    dummy_weights = torch.ones(1, 589)
    dummy_fbank = fbank_wrapper(dummy_waveform)

    # verify multi-mask parity with existing tail wrapper
    with torch.no_grad():
        tail_output = tail_wrapper(dummy_fbank, dummy_weights)
        multi_masks = dummy_weights.repeat(NUM_SPEAKERS, 1)
        multi_output = multi_mask_wrapper(dummy_fbank, multi_masks)
        max_diff = (tail_output - multi_output[0:1]).abs().max().item()
        assert max_diff < 1e-6, f"multi-mask parity check failed: max diff = {max_diff}"
        print(f"  multi-mask parity check passed (max diff = {max_diff:.2e})")

    with torch.no_grad():
        torch.onnx.export(
            fbank_wrapper,
            (dummy_waveform,),
            os.path.join(models_dir, "wespeaker-fbank.onnx"),
            input_names=["waveform"],
            output_names=["fbank"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
        torch.onnx.export(
            fbank_wrapper,
            (torch.randn(32, 1, 160000),),
            os.path.join(models_dir, "wespeaker-fbank-b32.onnx"),
            input_names=["waveform"],
            output_names=["fbank"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
    fbank_sz = os.path.getsize(os.path.join(models_dir, "wespeaker-fbank.onnx")) / 1e6
    print(f"  wespeaker-fbank.onnx ({fbank_sz:.1f} MB)")
    fbank_b32_sz = (
        os.path.getsize(os.path.join(models_dir, "wespeaker-fbank-b32.onnx")) / 1e6
    )
    print(f"  wespeaker-fbank-b32.onnx ({fbank_b32_sz:.1f} MB)")

    export_embedding_model(
        fbank_wrapper,
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34.onnx",
        batch_size=1,
    )
    export_embedding_model(
        fbank_wrapper,
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-b32.onnx",
        batch_size=32,
    )
    export_embedding_model(
        fbank_wrapper,
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-b64.onnx",
        batch_size=64,
    )
    export_embedding_tail_model(
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-tail.onnx",
        dummy_fbank,
        dummy_weights,
    )
    export_embedding_tail_model(
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-tail-b3.onnx",
        dummy_fbank.repeat(3, 1, 1),
        dummy_weights.repeat(3, 1),
    )
    export_embedding_tail_model(
        tail_wrapper,
        models_dir,
        "wespeaker-voxceleb-resnet34-tail-b32.onnx",
        dummy_fbank.repeat(32, 1, 1),
        dummy_weights.repeat(32, 1),
    )

    # export multi-mask models
    export_multi_mask_model(
        multi_mask_wrapper,
        models_dir,
        "wespeaker-multimask-tail.onnx",
        dummy_fbank,
        dummy_weights.repeat(NUM_SPEAKERS, 1),
    )
    export_multi_mask_model(
        multi_mask_wrapper,
        models_dir,
        "wespeaker-multimask-tail-b32.onnx",
        dummy_fbank.repeat(32, 1, 1),
        dummy_weights.repeat(32 * NUM_SPEAKERS, 1),
    )

    with open(
        os.path.join(models_dir, "wespeaker-voxceleb-resnet34.min_num_samples.txt"), "w"
    ) as f:
        f.write(f"{min_num_samples}\n")

    export_fixed_embedding_model(emb_model, models_dir, min_num_samples)


def export_embedding_model(
    fbank_wrapper: nn.Module,
    tail_wrapper: nn.Module,
    models_dir: str,
    filename: str,
    batch_size: int,
) -> None:
    dummy_waveform = torch.randn(batch_size, 1, 160000)
    dummy_weights = torch.ones(batch_size, 589)
    fbank_wrapper(dummy_waveform)

    class ExactEmbeddingWrapper(nn.Module):
        def __init__(self, fbank_model: nn.Module, tail_model: nn.Module) -> None:
            super().__init__()
            self.fbank_model = fbank_model
            self.tail_model = tail_model

        def forward(self, waveforms: torch.Tensor, weights: torch.Tensor) -> Any:
            fbank = self.fbank_model(waveforms)
            return self.tail_model(fbank, weights)

    wrapper = ExactEmbeddingWrapper(fbank_wrapper, tail_wrapper)
    wrapper.eval()
    output_path = os.path.join(models_dir, filename)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_waveform, dummy_weights),
            output_path,
            input_names=["waveform", "weights"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )

    sz = os.path.getsize(output_path) / 1e6
    print(f"  {filename} ({sz:.1f} MB)")


def export_embedding_tail_model(
    wrapper: nn.Module,
    models_dir: str,
    filename: str,
    dummy_fbank: torch.Tensor,
    dummy_weights: torch.Tensor,
) -> None:
    output_path = os.path.join(models_dir, filename)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_fbank, dummy_weights),
            output_path,
            input_names=["fbank", "weights"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )

    sz = os.path.getsize(output_path) / 1e6
    print(f"  {filename} ({sz:.1f} MB)")


def export_multi_mask_model(
    wrapper: nn.Module,
    models_dir: str,
    filename: str,
    dummy_fbank: torch.Tensor,
    dummy_masks: torch.Tensor,
) -> None:
    output_path = os.path.join(models_dir, filename)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_fbank, dummy_masks),
            output_path,
            input_names=["fbank", "masks"],
            output_names=["output"],
            opset_version=18,
            dynamo=True,
            external_data=False,
        )

    sz = os.path.getsize(output_path) / 1e6
    print(f"  {filename} ({sz:.1f} MB)")


def export_plda(models_dir: str) -> None:
    """Extract PLDA params from the cached pipeline blobs"""
    print("Extracting PLDA params...")
    blobs_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--pyannote--speaker-diarization-community-1/blobs"
    )

    if not os.path.isdir(blobs_dir):
        print("  Pipeline cache not found, skipping PLDA extraction")
        return

    for blob in sorted(os.listdir(blobs_dir)):
        blob_path = os.path.join(blobs_dir, blob)
        try:
            with open(blob_path, "rb") as f:
                magic = f.read(2)
                f.seek(0)
                if magic == b"PK":
                    data = np.load(f, allow_pickle=True)
                    if hasattr(data, "files"):
                        for name in data.files:
                            arr = data[name]
                            out = os.path.join(models_dir, f"plda_{name}.npy")
                            np.save(out, arr)
                            print(f"  plda_{name}.npy: shape={arr.shape}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
