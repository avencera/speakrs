#!/bin/bash
set -e

# ORT CUDA provider looks for shared libs in CWD (/workspace/)
for lib in libonnxruntime_providers_shared.so libonnxruntime_providers_cuda.so; do
    [ -f /usr/local/lib/$lib ] && ln -sf /usr/local/lib/$lib /workspace/$lib
done

if [ -d /opt/pyannote-bench ] && [ ! -d /workspace/pyannote-bench ]; then
    cp -r /opt/pyannote-bench /workspace/pyannote-bench
fi

touch /tmp/.container-ready

exec "$@"
