# Rocky 9 (RHEL-compatible) + rustup + NVIDIA dnf repo cuda-toolkit-12-9.
# Rocky 9 ships cmake 3.26 (clears 3.24); NVIDIA dnf repo for the toolkit.

FROM docker.io/rockylinux:9

RUN dnf install -y --setopt=install_weak_deps=False --allowerasing \
    cmake git curl ca-certificates gcc-c++ make pkgconf-pkg-config dnf-plugins-core \
    && dnf clean all

RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y --setopt=install_weak_deps=False cuda-toolkit-12-9 && \
    dnf clean all

ENV PATH="/usr/local/cuda/bin:/root/.cargo/bin:${PATH}"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal

ENV CUDA_ARCHITECTURES=75
