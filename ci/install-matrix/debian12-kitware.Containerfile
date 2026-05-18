# Debian 12 with Kitware's cmake (>= 3.26 needed for nvcc 12.9 support).

FROM docker.io/debian:12
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates pkg-config gnupg wget \
    && rm -rf /var/lib/apt/lists/*

# Kitware's apt repo for newer cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    echo 'deb https://apt.kitware.com/ubuntu/ jammy main' > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*

# NVIDIA apt repo for CUDA 12.9
RUN curl -sSLO https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-9 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:/root/.cargo/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal

ENV CUDA_ARCHITECTURES=75
