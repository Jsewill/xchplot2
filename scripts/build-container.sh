#!/usr/bin/env bash
#
# build-container.sh — auto-detect GPU vendor on the host and run the
# matching `podman compose build <service>` with the right env vars.
#
# Container builds can't probe the GPU themselves (no device access),
# so this script does it from the host before invoking compose.
#
# Usage:
#   ./scripts/build-container.sh                 # auto-detect
#   ./scripts/build-container.sh --gpu nvidia    # force NVIDIA
#   ./scripts/build-container.sh --gpu amd       # force AMD
#   ./scripts/build-container.sh --gpu intel     # force Intel
#   ./scripts/build-container.sh --engine docker # use docker compose instead

set -euo pipefail

ENGINE=podman
GPU=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU="$2";    shift 2 ;;
        --engine)  ENGINE="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Detect vendor ───────────────────────────────────────────────────────────
# Capture output first so `set -o pipefail` doesn't bite us — rocminfo and
# some nvidia-smi configurations exit non-zero even when they print useful
# information, and the pipefail bash setting then makes the entire pipeline
# return non-zero regardless of grep's match status.
if [[ -z "$GPU" ]]; then
    nvidia_out=""
    rocm_out=""
    if command -v nvidia-smi >/dev/null; then
        nvidia_out=$(nvidia-smi -L 2>/dev/null || true)
    fi
    if command -v rocminfo >/dev/null; then
        rocm_out=$(rocminfo 2>/dev/null || true)
    fi

    if [[ "$nvidia_out" == *GPU* ]]; then
        GPU=nvidia
    elif [[ "$rocm_out" == *gfx* ]]; then
        GPU=amd
    else
        echo "[build-container] No GPU detected via nvidia-smi or rocminfo." >&2
        echo "[build-container]" >&2
        echo "[build-container] Either:" >&2
        echo "[build-container]   1. Install the discovery tool for your vendor:" >&2
        echo "[build-container]        Arch:    sudo pacman -S nvidia-utils    (NVIDIA)" >&2
        echo "[build-container]                 sudo pacman -S rocminfo        (AMD)" >&2
        echo "[build-container]        Ubuntu:  sudo apt install nvidia-utils-XXX  (NVIDIA)" >&2
        echo "[build-container]                 sudo apt install rocminfo          (AMD)" >&2
        echo "[build-container]        (or run scripts/install-deps.sh which does this)" >&2
        echo "[build-container]   2. Force a service explicitly:" >&2
        echo "[build-container]        $0 --gpu nvidia | amd | intel" >&2
        exit 1
    fi
fi

# ── Map vendor → compose service + env ──────────────────────────────────────
case "$GPU" in
    nvidia)
        SERVICE=cuda
        # Enumerate ALL GPUs and build a fat binary (CMake's "61;86"
        # list syntax) so heterogeneous rigs (e.g. 1070 + 3060) get
        # native sm_NN codegen for each card, not just whichever one
        # nvidia-smi happened to list first. Single-card hosts produce
        # a single-arch list ("89") — same end result as the prior
        # head -1 path. Skip the probe entirely if the user pre-set
        # CUDA_ARCH (single arch or "61;86" list) so cross-targeting
        # an absent GPU still works.
        if [[ -z "${CUDA_ARCH:-}" ]] && command -v nvidia-smi >/dev/null; then
            # sed first (strip the dot), then sort -un (numeric dedup).
            # Without the numeric sort, 1070+5090 would emit "120;61"
            # because sort -u defaults to lexicographic.
            caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
                    | sed 's/\.//' | sort -un)
            if [[ -n "$caps" ]]; then
                export CUDA_ARCH=$(echo "$caps" | paste -sd';')
            fi
        fi
        : "${CUDA_ARCH:=89}"
        export CUDA_ARCH
        # Min arch drives the toolkit choice: a 1070+3060 mix needs a
        # toolchain that targets sm_61, not just sm_86. Works for
        # single-arch CUDA_ARCH=89 (min=89) and for user-set lists
        # like "61;86" (min=61).
        min_arch=$(echo "$CUDA_ARCH" | tr ';' '\n' | sort -n | head -1)
        # CUDA 13.0 dropped codegen for sm_50/52/53/60/61/62/70/72 entirely
        # — its nvcc fails the CMake TryCompile probe with "Unsupported gpu
        # architecture 'compute_61'" on Pascal, "compute_70" on Volta, etc.
        # Pin builds with ANY pre-Turing card to the last 12.x dev image,
        # which still covers sm_50 (Maxwell) through sm_120 (Blackwell), so
        # a mixed 1070+3060 (or 1070+5090) rig gets one toolchain that
        # handles every arch in the list. Honour an explicit BASE_DEVEL /
        # BASE_RUNTIME override from the env so users can pin to a
        # different toolkit if they need to.
        if (( min_arch < 75 )) && [[ -z "${BASE_DEVEL:-}" ]]; then
            export BASE_DEVEL="docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04"
            export BASE_RUNTIME="${BASE_RUNTIME:-$BASE_DEVEL}"
            echo "[build-container] sm_${min_arch} (pre-Turing) detected → pinning CUDA 12.9 base (CUDA 13.x dropped sub-Turing codegen)"
        fi
        echo "[build-container] vendor=nvidia service=$SERVICE CUDA_ARCH=$CUDA_ARCH"
        ;;
    amd)
        SERVICE=rocm
        # Reuse the rocminfo output captured during vendor detection (or
        # capture it now if --gpu amd was forced and rocm_out is empty).
        # Avoid `rocminfo | awk '...; exit'` because awk's early exit
        # SIGPIPEs rocminfo, and pipefail + set -e then kills the script.
        if [[ -z "${rocm_out:-}" ]] && command -v rocminfo >/dev/null; then
            rocm_out=$(rocminfo 2>/dev/null || true)
        fi
        # Honour an explicit ACPP_GFX from the env first (lets the user
        # cross-target a different GPU than the host one), else autodetect.
        if [[ -z "${ACPP_GFX:-}" ]]; then
            if [[ -n "${rocm_out:-}" && "$rocm_out" =~ (gfx[0-9a-f]+) ]]; then
                detected_gfx="${BASH_REMATCH[1]}"
                # RDNA1 workaround: gfx1010/1011/1012 aren't direct
                # AdaptiveCpp HIP targets. Community-tested (Radeon Pro
                # W5700) that gfx1013 is ISA-close enough to run on
                # gfx1010 silicon. Not parity-validated.
                case "$detected_gfx" in
                    gfx1010|gfx1011|gfx1012)
                        echo "[build-container] RDNA1 $detected_gfx detected — " \
                             "using gfx1013 spoof (community workaround, not " \
                             "parity-validated; verify plots with \`xchplot2 " \
                             "verify\` before farming)" >&2
                        export ACPP_GFX=gfx1013
                        ;;
                    *)
                        export ACPP_GFX="$detected_gfx"
                        ;;
                esac
            fi
        fi
        if [[ -z "${ACPP_GFX:-}" ]]; then
            # No silent fallback: a wrong gfx target produces an image that
            # builds clean and runs without errors, but the AOT amdgcn ISA
            # is for the wrong arch and the SYCL kernels execute as silent
            # no-ops at runtime (sort returns input unchanged, AES match
            # finds zero results, plot output diverges from reference).
            # Fail loud here instead.
            echo "[build-container] ERROR: couldn't detect AMD gfx target." >&2
            echo "[build-container] Either install rocminfo so the host probe finds it," >&2
            echo "[build-container] or set ACPP_GFX explicitly to your card's arch:" >&2
            echo "[build-container]   ACPP_GFX=gfx1030  $0  --gpu amd  # RX 6800 / 6800 XT / 6900 XT" >&2
            echo "[build-container]   ACPP_GFX=gfx1031  $0  --gpu amd  # RX 6700 XT / 6700 / 6800M" >&2
            echo "[build-container]   ACPP_GFX=gfx1100  $0  --gpu amd  # RX 7900 XTX / XT" >&2
            echo "[build-container] (run \"rocminfo | grep gfx\" if available)" >&2
            exit 1
        fi
        echo "[build-container] vendor=amd service=$SERVICE ACPP_GFX=$ACPP_GFX"
        ;;
    intel)
        SERVICE=intel
        echo "[build-container] vendor=intel service=$SERVICE (experimental, untested)"
        ;;
    *)
        echo "unknown --gpu value: $GPU (expected nvidia|amd|intel)" >&2
        exit 1
        ;;
esac

# podman-compose (and docker compose to varying degrees) evaluates
# ${VAR:?msg} interpolations across ALL services at YAML-parse time,
# even when only one service is being built. The rocm service's
# `${ACPP_GFX:?set ACPP_GFX to your GPU arch ...}` will then abort the
# parse during a `build cuda` or `build intel` invocation if ACPP_GFX
# isn't set in the env. Plant a dummy value so the parse succeeds for
# non-rocm builds; the rocm service is never actually instantiated.
if [[ "$SERVICE" != "rocm" ]]; then
    : "${ACPP_GFX:=unused-non-rocm-build}"
    export ACPP_GFX
fi

# ── Invoke compose ──────────────────────────────────────────────────────────
case "$ENGINE" in
    podman) COMPOSE=(podman compose) ;;
    docker) COMPOSE=(docker compose) ;;
    *) echo "unknown --engine: $ENGINE (expected podman|docker)" >&2; exit 1 ;;
esac

set -x
"${COMPOSE[@]}" build "$SERVICE"
