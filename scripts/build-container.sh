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
        # Pick the first GPU's compute_cap (e.g. "8.9" → "89") for sm_NN.
        if command -v nvidia-smi >/dev/null; then
            cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [[ -n "$cap" ]]; then
                export CUDA_ARCH=${cap//./}
            fi
        fi
        echo "[build-container] vendor=nvidia service=$SERVICE CUDA_ARCH=${CUDA_ARCH:-89}"
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
                export ACPP_GFX="${BASH_REMATCH[1]}"
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

# ── Invoke compose ──────────────────────────────────────────────────────────
case "$ENGINE" in
    podman) COMPOSE=(podman compose) ;;
    docker) COMPOSE=(docker compose) ;;
    *) echo "unknown --engine: $ENGINE (expected podman|docker)" >&2; exit 1 ;;
esac

set -x
"${COMPOSE[@]}" build "$SERVICE"
