#!/bin/bash
set -e

# source S3 credentials written by `cargo xtask gpu setup`
if [ -f /root/.env ]; then
    source /root/.env
fi

# RunPod mounts a volume at /workspace, hiding anything baked into the image
# at that path. This entrypoint seeds baked assets from /opt/ into the volume
# on first boot (idempotent -- skips if already present)

if [ -d /opt/models ] && [ ! -f /workspace/models/segmentation-3.0.onnx ]; then
    echo "Seeding baked models into /workspace/models/..."
    mkdir -p /workspace/models
    cp -rn /opt/models/* /workspace/models/ 2>/dev/null || true
fi

if [ -d /opt/datasets ] && [ ! -d /workspace/datasets/voxconverse-dev/wav ]; then
    echo "Seeding baked datasets into /workspace/datasets/..."
    mkdir -p /workspace/datasets
    cp -rn /opt/datasets/* /workspace/datasets/ 2>/dev/null || true
fi

if [ -f /opt/scripts/diarize_pyannote.py ] && [ ! -f /workspace/scripts/diarize_pyannote.py ]; then
    echo "Seeding pyannote script..."
    mkdir -p /workspace/scripts
    cp /opt/scripts/diarize_pyannote.py /workspace/scripts/
fi

exec "$@"
