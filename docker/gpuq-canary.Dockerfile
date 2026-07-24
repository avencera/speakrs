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
        | sh -s -- -y --default-toolchain 1.88.0 --profile minimal

ENV CARGO_HOME=/root/.cargo \
    PATH=/root/.cargo/bin:${PATH} \
    RUSTUP_HOME=/root/.rustup

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY xtask/Cargo.toml xtask/Cargo.toml
COPY xtask/src/ xtask/src/
COPY xtask/build.rs xtask/build.rs

RUN --mount=type=cache,target=/root/.cargo/registry,id=speakrs-gpuq-canary-registry \
    --mount=type=cache,target=/root/.cargo/git,id=speakrs-gpuq-canary-git \
    --mount=type=cache,target=/build/target,id=speakrs-gpuq-canary-target \
    cargo build --locked --release -p xtask --features cuda --bin speakrs-bm \
    && cp target/release/speakrs-bm /tmp/speakrs-bm

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d AS runtime

ARG ORT_VERSION=1.24.2
ARG S5CMD_VERSION=2.3.0
ARG S5CMD_SHA256=81d02a17a13797dc5949adb99734ad4217d005638a7827f36d435945527b2e69

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    RUST_BACKTRACE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        jq \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL \
        "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_linux_amd64.deb" \
        -o /tmp/s5cmd.deb \
    && echo "${S5CMD_SHA256}  /tmp/s5cmd.deb" | sha256sum -c - \
    && dpkg -i /tmp/s5cmd.deb \
    && rm /tmp/s5cmd.deb \
    && curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
        -o /tmp/ort.tgz \
    && mkdir -p /usr/local/lib \
    && tar xzf /tmp/ort.tgz --strip-components=2 -C /usr/local/lib \
        --wildcards "*/lib/*.so*" \
    && rm /tmp/ort.tgz \
    && ldconfig

COPY --from=builder /tmp/speakrs-bm /usr/local/bin/speakrs-bm
COPY docker/gpuq-workload.sh /usr/local/bin/speakrs-gpuq-workload

RUN chmod 0755 /usr/local/bin/speakrs-bm /usr/local/bin/speakrs-gpuq-workload \
    && mkdir -p /workspace

WORKDIR /workspace
CMD ["/usr/local/bin/speakrs-gpuq-workload", "--dataset", "voxconverse-dev", "--impls", "speakrs", "--max-files", "1"]
