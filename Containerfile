# syntax=docker/dockerfile:1
#
# Containerfile for xchplot2 — podman-first (works with docker too).
# Supports NVIDIA (default), AMD ROCm, and Intel oneAPI via build args.
#
# ── NVIDIA (default; CUB sort) ───────────────────────────────────────────────
#   podman build -t xchplot2:cuda .
#   podman run --rm --device nvidia.com/gpu=all -v $PWD/plots:/out \
#       xchplot2:cuda plot -k 28 -n 10 -f <farmer-pk> -c <pool-contract> -o /out
#   (Requires nvidia-container-toolkit + CDI on the host.)
#
# ── AMD ROCm (hand-rolled SYCL radix; XCHPLOT2_BUILD_CUDA=OFF) ───────────────
#   podman build -t xchplot2:rocm \
#       --build-arg BASE_DEVEL=docker.io/rocm/dev-ubuntu-24.04:latest \
#       --build-arg BASE_RUNTIME=docker.io/rocm/dev-ubuntu-24.04:latest \
#       --build-arg ACPP_TARGETS=hip:gfx1100 \
#       --build-arg XCHPLOT2_BUILD_CUDA=OFF \
#       --build-arg INSTALL_CUDA_HEADERS=1 \
#       .
#   podman run --rm --device /dev/kfd --device /dev/dri --group-add video \
#       -v $PWD/plots:/out xchplot2:rocm plot -k 28 -n 10 ... -o /out
#   (Adjust ACPP_TARGETS for your card: rocminfo | grep gfx.)
#
# ── Intel oneAPI (experimental, untested) ────────────────────────────────────
#   podman build -t xchplot2:intel \
#       --build-arg BASE_DEVEL=docker.io/intel/oneapi-basekit:latest \
#       --build-arg BASE_RUNTIME=docker.io/intel/oneapi-runtime:latest \
#       --build-arg ACPP_TARGETS=generic \
#       --build-arg XCHPLOT2_BUILD_CUDA=OFF \
#       --build-arg INSTALL_CUDA_HEADERS=1 \
#       .
#
# First build pulls + builds AdaptiveCpp from source — expect 10-30 min.
# Subsequent rebuilds reuse the cached AdaptiveCpp layer.

# BASE_RUNTIME defaults to the devel image because AdaptiveCpp's SSCP
# (LLVM "generic" target) JIT-assembles PTX at runtime via ptxas, which
# only ships in the CUDA *devel* image. The slim runtime image lacks it
# and produces "Code object construction failed". Override with a slim
# image only if you've switched ACPP_TARGETS to AOT (e.g. cuda:sm_89).
ARG BASE_DEVEL=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG BASE_RUNTIME=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG ACPP_REF=v25.10.0
ARG ACPP_TARGETS=
ARG XCHPLOT2_BUILD_CUDA=ON
ARG INSTALL_CUDA_HEADERS=0
ARG CUDA_ARCH=89
# LLVM/clang root used to build AdaptiveCpp. Default = Ubuntu's llvm-18.
# AMD/ROCm overrides this to /opt/rocm/llvm so the LLVM version matches
# ROCm's bitcode libraries (ocml.bc / ockl.bc), avoiding "Unknown
# attribute kind (102)" bitcode-version errors when targeting HIP.
# LLVM_CMAKE_DIR is the dir containing LLVMConfig.cmake (Ubuntu and
# ROCm lay these out differently — Ubuntu: $LLVM_ROOT/cmake, ROCm:
# $LLVM_ROOT/lib/cmake/llvm).
ARG LLVM_ROOT=/usr/lib/llvm-18
ARG LLVM_CMAKE_DIR=/usr/lib/llvm-18/cmake

# ─── builder ────────────────────────────────────────────────────────────────
FROM ${BASE_DEVEL} AS builder

ARG ACPP_REF
ARG ACPP_TARGETS
ARG XCHPLOT2_BUILD_CUDA
ARG INSTALL_CUDA_HEADERS
ARG CUDA_ARCH
ARG LLVM_ROOT
ARG LLVM_CMAKE_DIR

ENV DEBIAN_FRONTEND=noninteractive

# Common toolchain. AdaptiveCpp 25.10 wants LLVM ≥ 16 + clang + libclang;
# Ubuntu 24.04 ships llvm-18. Boost.Context, libnuma, libomp are AdaptiveCpp
# runtime deps. INSTALL_CUDA_HEADERS=1 pulls the CUDA Toolkit *headers* on
# non-NVIDIA bases — required because AdaptiveCpp's libkernel/half.hpp
# transitively includes cuda_fp16.h on every build path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake git ninja-build build-essential python3 pkg-config \
        curl ca-certificates \
        llvm-18 llvm-18-dev clang-18 libclang-18-dev libclang-cpp18-dev lld-18 \
        libboost-context-dev libnuma-dev libomp-18-dev \
 && if [ "${INSTALL_CUDA_HEADERS}" = "1" ]; then \
        apt-get install -y --no-install-recommends nvidia-cuda-toolkit-headers \
            || apt-get install -y --no-install-recommends nvidia-cuda-toolkit; \
    fi \
 && rm -rf /var/lib/apt/lists/*

# Rust toolchain (for keygen-rs and the `cargo install` entry point).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH=/root/.cargo/bin:${PATH}

# AdaptiveCpp from source, pinned. Installs to /opt/adaptivecpp.
RUN git clone --depth 1 --branch ${ACPP_REF} \
        https://github.com/AdaptiveCpp/AdaptiveCpp.git /tmp/acpp-src \
 && cmake -S /tmp/acpp-src -B /tmp/acpp-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp \
        -DCMAKE_C_COMPILER=${LLVM_ROOT}/bin/clang \
        -DCMAKE_CXX_COMPILER=${LLVM_ROOT}/bin/clang++ \
        -DLLVM_DIR=${LLVM_CMAKE_DIR} \
        -DACPP_LLD_PATH=${LLVM_ROOT}/bin/ld.lld \
 && cmake --build /tmp/acpp-build --parallel \
 && cmake --install /tmp/acpp-build \
 && rm -rf /tmp/acpp-src /tmp/acpp-build

ENV CMAKE_PREFIX_PATH=/opt/adaptivecpp:${CMAKE_PREFIX_PATH}
ENV PATH=/opt/adaptivecpp/bin:${PATH}

WORKDIR /xchplot2
COPY . .

# Build xchplot2. CUDA_ARCHITECTURES + ACPP_TARGETS + XCHPLOT2_BUILD_CUDA
# get picked up by build.rs; the latter switches the CMake source set
# between the CUB-using TUs (.cu files via nvcc) and the SYCL-only path.
RUN CUDA_ARCHITECTURES=${CUDA_ARCH} \
    ACPP_TARGETS=${ACPP_TARGETS} \
    XCHPLOT2_BUILD_CUDA=${XCHPLOT2_BUILD_CUDA} \
    cargo install --path . --root /usr/local --locked

# Also build the parity tests via plain CMake so they're available
# inside the container for first-port validation on new GPUs (especially
# AMD/Intel). Reuses the static libs cargo install just built.
RUN cmake -S . -B build-tests -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DACPP_TARGETS=${ACPP_TARGETS} \
        -DXCHPLOT2_BUILD_CUDA=${XCHPLOT2_BUILD_CUDA} \
 && cmake --build build-tests --parallel --target sycl_sort_parity \
                                          sycl_bucket_offsets_parity \
                                          sycl_g_x_parity \
                                          plot_file_parity \
 && install -m 0755 build-tests/tools/parity/sycl_sort_parity            /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/sycl_bucket_offsets_parity  /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/sycl_g_x_parity             /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/plot_file_parity            /usr/local/bin/ \
 && rm -rf build-tests target

# ─── runtime ────────────────────────────────────────────────────────────────
FROM ${BASE_RUNTIME}

ENV DEBIAN_FRONTEND=noninteractive

# AdaptiveCpp's runtime backend loaders dlopen libLLVM (for SSCP runtime
# specialization), libnuma (OMP backend), libomp, and Boost.Context.
# SSCP also shells out to LLVM's `opt` and `llc` binaries at runtime to
# generate PTX from the SSCP bitcode — install the full llvm-18 package
# (binaries + lib), not just libllvm18.
RUN apt-get update && apt-get install -y --no-install-recommends \
        llvm-18 lld-18 libnuma1 libomp5-18 libboost-context1.83.0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/xchplot2                    /usr/local/bin/xchplot2
COPY --from=builder /usr/local/bin/sycl_sort_parity            /usr/local/bin/sycl_sort_parity
COPY --from=builder /usr/local/bin/sycl_bucket_offsets_parity  /usr/local/bin/sycl_bucket_offsets_parity
COPY --from=builder /usr/local/bin/sycl_g_x_parity             /usr/local/bin/sycl_g_x_parity
COPY --from=builder /usr/local/bin/plot_file_parity            /usr/local/bin/plot_file_parity
COPY --from=builder /opt/adaptivecpp                           /opt/adaptivecpp

ENV LD_LIBRARY_PATH=/opt/adaptivecpp/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/adaptivecpp/bin:${PATH}

ENTRYPOINT ["/usr/local/bin/xchplot2"]
CMD ["--help"]
