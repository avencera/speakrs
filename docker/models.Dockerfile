# Downloads speakrs ONNX models from HuggingFace
#
# Rebuild: just gpu-models-image
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG HF_TOKEN=""

RUN if [ -n "$HF_TOKEN" ]; then export HF_TOKEN="$HF_TOKEN"; fi && \
    uv tool run --from huggingface-hub hf download avencera/speakrs-models \
        --local-dir /opt/models \
        --include "segmentation-3.0.onnx" \
        --include "wespeaker-voxceleb-resnet34.onnx" \
        --include "wespeaker-voxceleb-resnet34.onnx.data" \
        --include "plda_*.npy" \
        --include "wespeaker-voxceleb-resnet34.min_num_samples.txt"
