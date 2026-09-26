# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch>=2.6", "torchaudio", "onnx", "onnxscript"]
# ///
"""Generate fixtures for the host fbank frontend and the fixed-model graph split.

The fbank fixtures come from the exported `FbankWrapper`, so the Rust frontend is
checked against the exact PyTorch equations that the fixed embedding model uses.
The toy model wraps the same frontend and a small tail in `FixedEmbeddingWrapper`
and is exported the same way as the real fixed model, so it carries the same
exporter name scopes that the graph split relies on
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_models import FbankWrapper, FixedEmbeddingWrapper  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
TOY_WINDOW_SAMPLES = 8_000
TOY_MASK_FRAMES = 24
TOY_EMBEDDING_WIDTH = 4


class ToyResNet(nn.Module):
    """Smallest tail with the source ResNet interface used by the fixed wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)
        self.seg_1 = nn.Linear(2 * 40 * 2, TOY_EMBEDDING_WIDTH)
        self.two_emb_layer = False

    def forward_frames(self, fbank: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(fbank.permute(0, 2, 1).unsqueeze(1)))


class ToySource(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = ToyResNet()


def waveform(samples: int) -> torch.Tensor:
    """Deterministic speech-like input: two tones, a sweep, and seeded noise."""

    generator = torch.Generator().manual_seed(20260925)
    time = torch.arange(samples, dtype=torch.float64) / 16_000
    signal = (
        0.3 * torch.sin(2 * torch.pi * 220 * time)
        + 0.1 * torch.sin(2 * torch.pi * 1_375 * time)
        + 0.05 * torch.sin(2 * torch.pi * (100 + 2_000 * time) * time)
        + 0.02 * torch.randn(samples, generator=generator, dtype=torch.float64)
    )
    return signal.to(torch.float32).view(1, 1, samples)


def main() -> None:
    torch.manual_seed(20260925)
    frontend = FbankWrapper(None).eval()
    audio = waveform(TOY_WINDOW_SAMPLES)
    with torch.no_grad():
        fbank = frontend(audio)[0]

    np.save(FIXTURES / "wespeaker_fbank_input.npy", audio.view(-1).numpy())
    np.save(FIXTURES / "wespeaker_fbank_window.npy", frontend.window.numpy())
    np.save(FIXTURES / "wespeaker_fbank_mel.npy", frontend.mel.numpy())
    np.save(FIXTURES / "wespeaker_fbank_expected.npy", fbank.numpy())

    wrapper = FixedEmbeddingWrapper(ToySource()).eval()
    torch.onnx.export(
        wrapper,
        (
            torch.zeros((1, 1, TOY_WINDOW_SAMPLES), dtype=torch.float32),
            torch.ones((1, TOY_MASK_FRAMES), dtype=torch.float32),
        ),
        FIXTURES / "fixed_embedding_toy.onnx",
        input_names=["waveform", "weights"],
        output_names=["output"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )


if __name__ == "__main__":
    main()
