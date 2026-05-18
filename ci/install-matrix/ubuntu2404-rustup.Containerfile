# Like Containerfile.ubuntu2404 but uses rustup (newer cargo) instead of
# apt's cargo. This is what the original user must have had — their log
# got past build.rs and nvcc to the link stage, which apt cargo 1.75
# never reaches because it dies on the v4 Cargo.lock.

FROM docker.io/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates pkg-config \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Pin arch since there's no GPU to probe.
ENV CUDA_ARCHITECTURES=75
