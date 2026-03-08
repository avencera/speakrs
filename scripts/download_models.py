# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyannote.audio>=3.3",
#     "torch>=2.6",
#     "numpy",
#     "onnx",
#     "onnxscript",
# ]
# ///
"""Download and export ONNX models + PLDA params for speakrs.

Args: models_dir (path to fixtures/models/)
Env: HF_TOKEN (HuggingFace token with model access)

Requires accepting terms at:
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
"""

import os
import sys
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import get_mel_banks

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


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

    sz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0.onnx")) / 1e6
    print(f"  segmentation-3.0.onnx ({sz:.1f} MB)")
    bsz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0-b32.onnx")) / 1e6
    print(f"  segmentation-3.0-b32.onnx ({bsz:.1f} MB)")


def export_embedding(pipeline: Any, models_dir: str) -> None:
    """Export the exact WeSpeaker embedding path for batch-1 and batch-32 inference"""
    print("Exporting embedding model...")

    class FbankWrapper(nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
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

    class EmbeddingTailWrapper(nn.Module):
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

    emb_model = pipeline._embedding.model_
    emb_model.eval()
    fbank_wrapper = FbankWrapper(emb_model)
    fbank_wrapper.eval()
    tail_wrapper = EmbeddingTailWrapper(emb_model)
    tail_wrapper.eval()

    dummy_waveform = torch.randn(1, 1, 160000)
    dummy_weights = torch.ones(1, 589)
    dummy_fbank = fbank_wrapper(dummy_waveform)

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
    with open(
        os.path.join(models_dir, "wespeaker-voxceleb-resnet34.min_num_samples.txt"), "w"
    ) as f:
        f.write(f"{pipeline._embedding.min_num_samples}\n")


def export_embedding_model(
    fbank_wrapper: nn.Module,
    tail_wrapper: nn.Module,
    models_dir: str,
    filename: str,
    batch_size: int,
) -> None:
    dummy_waveform = torch.randn(batch_size, 1, 160000)
    dummy_weights = torch.ones(batch_size, 589)
    dummy_fbank = fbank_wrapper(dummy_waveform)

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
