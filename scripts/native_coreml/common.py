from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import get_mel_banks

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

PIPELINE_ID = "pyannote/speaker-diarization-community-1"
SEGMENTATION_SAMPLES = 160_000
SEGMENTATION_FRAMES = 589
FBANK_FRAMES = 998
FBANK_FEATURES = 80
TAIL_BATCH_SIZES = (1, 3, 32)
SEGMENTATION_BATCH_SIZES = tuple(range(1, 33))
SEGMENTATION_STEM = "segmentation-3.0"
SEGMENTATION_BATCHED_STEM = "segmentation-3.0-b32"
TAIL_STEM = "wespeaker-voxceleb-resnet34-tail"
TAIL_B3_STEM = "wespeaker-voxceleb-resnet34-tail-b3"
TAIL_B32_STEM = "wespeaker-voxceleb-resnet34-tail-b32"
PACKAGES_DIRNAME = "coreml-packages"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def load_pipeline() -> Any:
    from pyannote.audio import Pipeline

    from_pretrained: Any = Pipeline.from_pretrained
    token = hf_token()
    if token:
        try:
            pipeline = from_pretrained(PIPELINE_ID, token=token)
        except TypeError:
            try:
                legacy_kwargs: dict[str, Any] = {"use_auth_token": token}
                pipeline = from_pretrained(PIPELINE_ID, **legacy_kwargs)
            except TypeError:
                pipeline = from_pretrained(PIPELINE_ID)
    else:
        pipeline = from_pretrained(PIPELINE_ID)
    assert pipeline is not None
    pipeline.to(torch.device("cpu"))
    return pipeline


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def coreml_packages_dir(output_dir: Path) -> Path:
    return output_dir / PACKAGES_DIRNAME


def segmentation_package_path(output_dir: Path) -> Path:
    return coreml_packages_dir(output_dir) / f"{SEGMENTATION_STEM}.mlpackage"


def tail_package_path(output_dir: Path) -> Path:
    return coreml_packages_dir(output_dir) / f"{TAIL_STEM}.mlpackage"


def patch_sincnet_encoder_for_tracing(model: nn.Module) -> None:
    if not hasattr(model, "sincnet"):
        return

    conv_layers = getattr(model.sincnet, "conv1d", None)
    if conv_layers is None or len(conv_layers) == 0:
        return

    encoder = conv_layers[0]
    if encoder is None or not hasattr(encoder, "get_filters"):
        return

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        filters = self.get_filters()
        waveform = self.filterbank.pre_analysis(waveform)
        spectrum = F.conv1d(waveform, filters, stride=self.stride, padding=self.padding)
        return self.filterbank.post_analysis(spectrum)

    encoder.forward = forward.__get__(encoder, encoder.__class__)


class FbankWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = float(1 << 15)
        self.preemph = 0.97

        window = torch.hamming_window(400, periodic=False, alpha=0.54, beta=0.46)
        mel, _ = get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)

        self.register_buffer("window", window)
        self.register_buffer("mel", F.pad(mel, (0, 1), value=0.0).T.contiguous())
        self.register_buffer("eps", torch.tensor(torch.finfo(torch.float32).eps))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
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


class EmbeddingTailWrapper(nn.Module):
    def __init__(self, model: Any) -> None:
        super().__init__()
        self.resnet = model.resnet
        with torch.no_grad():
            dummy_fbank = torch.zeros(1, FBANK_FRAMES, FBANK_FEATURES)
            frames = self.resnet.forward_frames(dummy_fbank)
        self.target_frames = frames.size(-1)

    def pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.unsqueeze(1)
        weights = F.interpolate(weights, size=self.target_frames, mode="nearest")

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

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> Any:
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


def build_tail_wrapper(pipeline: Any) -> EmbeddingTailWrapper:
    tail_wrapper = EmbeddingTailWrapper(pipeline._embedding.model_)
    tail_wrapper.eval()
    return tail_wrapper


def build_fbank_wrapper() -> FbankWrapper:
    wrapper = FbankWrapper()
    wrapper.eval()
    return wrapper


def save_model_artifacts(
    mlmodel: Any,
    package_path: Path,
    compiled_paths: list[Path],
) -> None:
    remove_path(package_path)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(package_path))

    compiled_dir = Path(mlmodel.get_compiled_model_path())
    for compiled_path in compiled_paths:
        remove_path(compiled_path)
        compiled_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(compiled_dir, compiled_path)
