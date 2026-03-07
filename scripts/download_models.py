"""Download and export ONNX models + PLDA params for speakrs.

Args: models_dir (path to fixtures/models/)
Env: HF_TOKEN (HuggingFace token with model access)

Requires accepting terms at:
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn


def main():
    models_dir = sys.argv[1]
    token = os.environ["HF_TOKEN"]
    os.makedirs(models_dir, exist_ok=True)

    print("Loading community-1 pipeline...")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    pipeline.to(torch.device("cpu"))

    export_segmentation(pipeline, models_dir)
    export_embedding(pipeline, models_dir)
    export_plda(models_dir)

    print("Done!")


def export_segmentation(pipeline, models_dir):
    print("Exporting segmentation model...")
    seg_model = pipeline._segmentation.model
    seg_model.eval()

    dummy = torch.randn(1, 1, 160000)
    with torch.no_grad():
        torch.onnx.export(
            seg_model,
            dummy,
            os.path.join(models_dir, "segmentation-3.0.onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {2: "samples"}, "output": {1: "frames"}},
            opset_version=14,
            dynamo=False,
        )

    sz = os.path.getsize(os.path.join(models_dir, "segmentation-3.0.onnx")) / 1e6
    print(f"  segmentation-3.0.onnx ({sz:.1f} MB)")


def export_embedding(pipeline, models_dir):
    """Export the ResNet backbone only -- fbank is computed in Rust"""
    print("Exporting embedding model...")

    class ResNetWrapper(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet

        def forward(self, fbank):
            return self.resnet(fbank)[1]

    emb_model = pipeline._embedding.model_
    emb_model.eval()
    wrapper = ResNetWrapper(emb_model.resnet)
    wrapper.eval()

    dummy_fbank = torch.randn(1, 200, 80)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_fbank,
            os.path.join(models_dir, "wespeaker-voxceleb-resnet34.onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {1: "frames"}},
            opset_version=14,
            dynamo=False,
        )

    sz = os.path.getsize(
        os.path.join(models_dir, "wespeaker-voxceleb-resnet34.onnx")
    ) / 1e6
    print(f"  wespeaker-voxceleb-resnet34.onnx ({sz:.1f} MB)")


def export_plda(models_dir):
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
