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
if [[ -z "$GPU" ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        GPU=nvidia
    elif command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q gfx; then
        GPU=amd
    else
        echo "[build-container] No GPU detected via nvidia-smi or rocminfo." >&2
        echo "[build-container] Use --gpu nvidia|amd|intel to force a service." >&2
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
        if command -v rocminfo >/dev/null; then
            gfx=$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]+gfx[0-9a-f]+/ {print $2; exit}')
            if [[ -n "$gfx" ]]; then
                export ACPP_GFX="$gfx"
            fi
        fi
        if [[ -z "${ACPP_GFX:-}" ]]; then
            echo "[build-container] couldn't detect gfx target; falling back to gfx1100." >&2
            echo "[build-container] override with ACPP_GFX=gfx1031 (Navi 22) etc." >&2
            export ACPP_GFX=gfx1100
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
