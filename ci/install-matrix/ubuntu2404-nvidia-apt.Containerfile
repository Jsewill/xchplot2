# Ubuntu 24.04 + rustup cargo + NVIDIA's official apt repo CUDA toolkit
# (not the stock Ubuntu nvidia-cuda-toolkit). NVIDIA's repo installs to
# /usr/local/cuda-12.X/ with a /usr/local/cuda symlink. This is what the
# project's README recommends, so it should work — verifying.

FROM docker.io/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates pkg-config gnupg \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA's apt repo
RUN curl -sSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-9 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:/usr/local/cuda/bin:${PATH}"

ENV CUDA_ARCHITECTURES=75

RUN { echo "=== libcudart locations ==="; \
      find /usr -name 'libcudart*' 2>/dev/null | head -20; \
      echo "=== /usr/local/cuda symlink? ==="; \
      ls -la /usr/local/ | grep -i cuda; \
      echo "=== nvcc ==="; \
      nvcc --version | tail -3; \
    } > /root/env-probe.txt 2>&1 ; cat /root/env-probe.txt
