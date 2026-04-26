# syntax=docker/dockerfile:1
#
# Containerfile for xchplot2 (cuda-only branch) — podman-first (works
# with docker too). NVIDIA only — this branch has no AMD/Intel paths.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   podman build -t xchplot2:cuda-only .
#   podman run --rm --device nvidia.com/gpu=all -v $PWD/plots:/out \
#       xchplot2:cuda-only plot -k 28 -n 10 -f <farmer-pk> -c <pool-contract> -o /out
#   (Requires nvidia-container-toolkit + CDI on the host.)
#
# ── Pascal / Volta (sm_61–sm_72): pin to CUDA 12.x ───────────────────────────
# The default base image is CUDA 13.x, which only supports sm_75+ (Turing
# and newer). Pascal (sm_61) and Volta (sm_70) builds need a 12.x base —
# pass it explicitly:
#
#   podman build -t xchplot2:cuda-only \
#       --build-arg BASE_DEVEL=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \
#       --build-arg BASE_RUNTIME=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04 \
#       --build-arg CUDA_ARCH=61 \
#       .
#
# scripts/build-container.sh handles this automatically by probing
# nvidia-smi and pinning the 12.9 base when CUDA_ARCH < 75. It also
# enumerates ALL GPUs and builds a fat binary for heterogeneous rigs
# (e.g. 1070 + 3060 → CUDA_ARCH="61;86").

# BASE_RUNTIME defaults to the devel image because the runtime image
# needs ptxas for any compute_NN PTX entries that get JIT-loaded when
# the runtime asks for an arch that wasn't AOT-compiled. The slim
# runtime image lacks ptxas. Override with a slim image only if
# CUDA_ARCH covers every device you intend to run on.
ARG BASE_DEVEL=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG BASE_RUNTIME=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG CUDA_ARCH=89

# ─── builder ────────────────────────────────────────────────────────────────
FROM ${BASE_DEVEL} AS builder

ARG CUDA_ARCH

ENV DEBIAN_FRONTEND=noninteractive

# cuda-only's dep list is intentionally short: no AdaptiveCpp / SYCL /
# LLVM / lld plumbing — just cmake, a C++20 compiler, and nvcc (from
# the base image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake git ninja-build build-essential python3 pkg-config \
        curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Rust toolchain (for keygen-rs and the cargo install entry point).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH=/root/.cargo/bin:${PATH}

WORKDIR /xchplot2
COPY . .

# Build xchplot2 CLI. CUDA_ARCHITECTURES gets picked up by build.rs
# (accepts CMake's "61;86" multi-arch list syntax for fat binaries).
RUN CUDA_ARCHITECTURES=${CUDA_ARCH} \
    cargo install --path . --root /usr/local --locked

# Also build the parity tests via plain CMake so they're available
# inside the runtime image for first-port validation on new GPUs.
# Reuses no state from cargo install — compiles its own copy of the
# static libs into build-tests/, then strips them after install.
RUN cmake -S . -B build-tests -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
 && cmake --build build-tests --parallel --target \
        aes_parity aes_bs_parity aes_bs_bench aes_tezcan_bench \
        xs_parity xs_bench t1_parity t2_parity t3_parity t1_debug \
        plot_file_parity \
 && install -m 0755 build-tests/tools/parity/aes_parity        /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/aes_bs_parity     /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/aes_bs_bench      /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/aes_tezcan_bench  /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/xs_parity         /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/xs_bench          /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/t1_parity         /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/t1_debug          /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/t2_parity         /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/t3_parity         /usr/local/bin/ \
 && install -m 0755 build-tests/tools/parity/plot_file_parity  /usr/local/bin/ \
 && rm -rf build-tests target

# ─── runtime ────────────────────────────────────────────────────────────────
FROM ${BASE_RUNTIME} AS runtime

COPY --from=builder /usr/local/bin/xchplot2           /usr/local/bin/xchplot2
COPY --from=builder /usr/local/bin/aes_parity         /usr/local/bin/aes_parity
COPY --from=builder /usr/local/bin/aes_bs_parity      /usr/local/bin/aes_bs_parity
COPY --from=builder /usr/local/bin/aes_bs_bench       /usr/local/bin/aes_bs_bench
COPY --from=builder /usr/local/bin/aes_tezcan_bench   /usr/local/bin/aes_tezcan_bench
COPY --from=builder /usr/local/bin/xs_parity          /usr/local/bin/xs_parity
COPY --from=builder /usr/local/bin/xs_bench           /usr/local/bin/xs_bench
COPY --from=builder /usr/local/bin/t1_parity          /usr/local/bin/t1_parity
COPY --from=builder /usr/local/bin/t1_debug           /usr/local/bin/t1_debug
COPY --from=builder /usr/local/bin/t2_parity          /usr/local/bin/t2_parity
COPY --from=builder /usr/local/bin/t3_parity          /usr/local/bin/t3_parity
COPY --from=builder /usr/local/bin/plot_file_parity   /usr/local/bin/plot_file_parity

ENTRYPOINT ["/usr/local/bin/xchplot2"]
CMD ["--help"]
