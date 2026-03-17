# Downloads VoxConverse Dev dataset for benchmarking
#
# Rebuild: just gpu-datasets-image
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/datasets/voxconverse-dev/rttm && \
    curl --fail -L -o /tmp/dev.zip \
      https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip && \
    unzip -q /tmp/dev.zip -d /opt/datasets/voxconverse-dev && \
    mv /opt/datasets/voxconverse-dev/audio /opt/datasets/voxconverse-dev/wav && \
    rm /tmp/dev.zip && \
    git clone --depth 1 https://github.com/joonson/voxconverse /tmp/voxconverse-clone && \
    cp /tmp/voxconverse-clone/dev/*.rttm /opt/datasets/voxconverse-dev/rttm/ && \
    rm -rf /tmp/voxconverse-clone
