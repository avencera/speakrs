# CUDA runtime + FFmpeg + tools for running speakrs-bm
#
# Rebuild: just gpu-runtime-image
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libpython3.12-dev \
    ca-certificates curl xz-utils git unzip \
    openssh-server tmux \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_linux_amd64.deb -o /tmp/s5cmd.deb \
    && dpkg -i /tmp/s5cmd.deb \
    && rm /tmp/s5cmd.deb

# FFmpeg 8 shared libs for torchcodec (pyannote.audio >=4.0)
RUN curl -fSL -o /tmp/ffmpeg8.tar.xz \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-linux64-lgpl-shared-8.0.tar.xz" && \
    tar xJf /tmp/ffmpeg8.tar.xz --strip-components=1 -C /usr/local && \
    rm /tmp/ffmpeg8.tar.xz && \
    ldconfig

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
