# CUDA + Rust + dev packages for building speakrs-bm
#
# Rebuild: just gpu-base-image
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config cmake \
    ca-certificates curl git python3 \
    libssl-dev libclang-dev libopenblas-dev \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV CARGO_HOME="/root/.cargo" \
    PATH="/root/.cargo/bin:${PATH}" \
    RUSTUP_HOME="/root/.rustup"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
