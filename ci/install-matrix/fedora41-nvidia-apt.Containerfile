# Fedora 41 + rustup + NVIDIA dnf repo cuda-toolkit-12-9.
# Fedora ships cmake 3.30+ stock. Stock nvcc isn't packaged; NVIDIA's
# dnf repo gives us cuda-toolkit-12-9.

FROM docker.io/fedora:41

RUN dnf install -y --setopt=install_weak_deps=False \
    @development-tools cmake git curl ca-certificates \
    pkgconf-pkg-config gcc-c++ \
    && dnf clean all

RUN dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo && \
    dnf install -y --setopt=install_weak_deps=False cuda-toolkit-12-9 && \
    dnf clean all

ENV PATH="/usr/local/cuda/bin:/root/.cargo/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal

ENV CUDA_ARCHITECTURES=75
