#!/usr/bin/env bash
#
# build-container.sh — auto-detect the host's NVIDIA GPU(s) and run
# `podman compose build cuda` with the right env vars.
#
# Container builds can't probe the GPU themselves (no device access),
# so this script does it from the host before invoking compose.
#
# cuda-only branch is NVIDIA-only; use the main branch for AMD/Intel.
#
# Usage:
#   ./scripts/build-container.sh                 # auto-detect
#   ./scripts/build-container.sh --no-cache      # force clean rebuild
#   ./scripts/build-container.sh --engine docker # use docker compose instead
#
# Override knobs (set in env before invoking):
#   CUDA_ARCH=61        # force a specific arch (e.g. cross-targeting)
#   CUDA_ARCH=61;86     # explicit fat-binary list
#   BASE_DEVEL=...      # override the auto-picked toolkit base image
#   BASE_RUNTIME=...    # override the runtime base image

set -euo pipefail

ENGINE=podman
declare -a EXTRA_BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)  ENGINE="$2"; shift 2 ;;
        # Force a clean rebuild (ignore podman/docker layer cache). Useful
        # after a host upgrade (new nvcc / new CUDA toolkit / etc.) where
        # the cached layers reference stale toolchain versions.
        --no-cache) EXTRA_BUILD_ARGS+=("--no-cache"); shift 1 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Detect GPU(s) and toolkit pinning ───────────────────────────────────────
# Enumerate ALL GPUs and build a fat binary (CMake's "61;86" list
# syntax) so heterogeneous rigs (e.g. 1070 + 3060) get native sm_NN
# codegen for each card, not just whichever one nvidia-smi happened to
# list first. Single-card hosts produce a single-arch list ("89").
# Skip the probe entirely if CUDA_ARCH is pre-set in the env so
# cross-targeting an absent GPU still works.
if [[ -z "${CUDA_ARCH:-}" ]] && command -v nvidia-smi >/dev/null; then
    # sed first (strip the dot), then sort -un (numeric dedup).
    # Without numeric sort, 1070+5090 would emit "120;61" because
    # sort -u defaults to lexicographic.
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
            | sed 's/\.//' | sort -un)
    if [[ -n "$caps" ]]; then
        # Split assignment from export so a non-zero exit from the
        # subshell pipeline propagates instead of being masked by
        # `export`'s own success (shellcheck SC2155).
        CUDA_ARCH=$(echo "$caps" | paste -sd';')
        export CUDA_ARCH
    fi
fi
: "${CUDA_ARCH:=89}"
export CUDA_ARCH

# Min arch drives the toolkit choice: a 1070+3060 mix needs a
# toolchain that targets sm_61, not just sm_86. Works for single-arch
# CUDA_ARCH=89 (min=89) and for user-set lists like "61;86" (min=61).
min_arch=$(echo "$CUDA_ARCH" | tr ';' '\n' | sort -n | head -1)

# CUDA 13.0 dropped codegen for sm_50/52/53/60/61/62/70/72 entirely
# — its nvcc fails the CMake TryCompile probe with "Unsupported gpu
# architecture 'compute_61'" on Pascal, "compute_70" on Volta, etc.
# Pin builds with ANY pre-Turing card to the last 12.x dev image,
# which still covers sm_50 (Maxwell) through sm_120 (Blackwell), so
# a mixed 1070+3060 (or 1070+5090) rig gets one toolchain that
# handles every arch in the list. Honour an explicit BASE_DEVEL /
# BASE_RUNTIME override from the env so users can pin to a different
# toolkit if they need to.
if (( min_arch < 75 )) && [[ -z "${BASE_DEVEL:-}" ]]; then
    export BASE_DEVEL="docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04"
    export BASE_RUNTIME="${BASE_RUNTIME:-$BASE_DEVEL}"
    echo "[build-container] sm_${min_arch} (pre-Turing) detected → pinning CUDA 12.9 base (CUDA 13.x dropped sub-Turing codegen)"
fi
echo "[build-container] vendor=nvidia service=cuda CUDA_ARCH=$CUDA_ARCH"

# ── Invoke compose ──────────────────────────────────────────────────────────
case "$ENGINE" in
    podman) COMPOSE=(podman compose) ;;
    docker) COMPOSE=(docker compose) ;;
    *) echo "unknown --engine: $ENGINE (expected podman|docker)" >&2; exit 1 ;;
esac

set -x
"${COMPOSE[@]}" build "${EXTRA_BUILD_ARGS[@]}" cuda
