# syntax=docker/dockerfile:1.7

# linux/amd64 manifests for the CUDA 12.4.1 images
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libclang-dev \
        libopenblas-dev \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.89.0 --profile minimal

ENV CARGO_HOME=/root/.cargo \
    PATH=/root/.cargo/bin:${PATH} \
    RUSTUP_HOME=/root/.rustup

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
# Cargo.toml declares [[example]] targets, and cargo checks their files exist
COPY examples/ examples/
COPY xtask/Cargo.toml xtask/Cargo.toml
COPY xtask/src/ xtask/src/
COPY xtask/build.rs xtask/build.rs

RUN --mount=type=cache,target=/root/.cargo/registry,id=speakrs-gpuq-canary-registry \
    --mount=type=cache,target=/root/.cargo/git,id=speakrs-gpuq-canary-git \
    --mount=type=cache,target=/build/target,id=speakrs-gpuq-canary-target \
    cargo build --locked --release -p xtask --no-default-features --features cuda --bin speakrs-bm \
    && cp target/release/speakrs-bm /tmp/speakrs-bm

# the cudnn-runtime image provides the cuBLAS, cuDNN 9 and NVRTC libraries that the native
# CUDA backend loads at run time
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    RUST_BACKTRACE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        jq \
        git \
        unzip \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /tmp/speakrs-bm /usr/local/bin/speakrs-bm
COPY docker/gpuq-workload.sh /usr/local/bin/speakrs-gpuq-workload

RUN chmod 0755 /usr/local/bin/speakrs-bm /usr/local/bin/speakrs-gpuq-workload \
    && mkdir -p /workspace

WORKDIR /workspace
CMD ["/usr/local/bin/speakrs-gpuq-workload", "--dataset", "voxconverse-dev", "--impls", "speakrs", "--max-files", "1"]
