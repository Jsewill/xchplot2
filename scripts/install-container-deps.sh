#!/usr/bin/env bash
#
# install-container-deps.sh — bootstrap the host packages required to
# build & run xchplot2's CUDA container image via
# scripts/build-container.sh.
#
# Native build deps (CUDA Toolkit, Rust) all live INSIDE the container
# image — the host does not need them. This script only installs:
#   1. A container engine + compose plugin: `podman` + `podman-compose`
#      (default), or `docker` + the `docker compose` v2 plugin via
#      `--engine docker`.
#   2. `nvidia-utils` (Arch) / `nvidia-utils-$DRV_MAJOR` (apt) so
#      build-container.sh's `nvidia-smi --query-gpu=compute_cap` probe
#      can pick the right CUDA_ARCH. Falls back to sm_89 if absent.
#   3. `nvidia-container-toolkit` + a CDI spec at /etc/cdi/nvidia.yaml
#      (podman) or the docker runtime hook (docker), so compose.yaml's
#      `devices: - nvidia.com/gpu=all` actually passes a GPU through.
#
# cuda-only branch is NVIDIA-only; the main branch's variant of this
# script handles AMD / Intel / CPU as well.
#
# Note: cuda-only's compose.yaml uses CDI shorthand (podman-native); on
# `--engine docker` the `compose run` step won't see the GPU even after
# this script's `nvidia-ctk runtime configure`. Bare `docker run --gpus
# all` (post-configure) works. Use podman for the compose path.
#
# Usage:
#   scripts/install-container-deps.sh                  # podman + NVIDIA
#   scripts/install-container-deps.sh --engine docker  # docker instead of podman
#   scripts/install-container-deps.sh --no-nvidia-repo # skip adding NVIDIA's apt/dnf repo
#
# Supported distros: Arch family, Ubuntu/Debian, Fedora/RHEL.

set -euo pipefail

ENGINE=podman
ADD_NVIDIA_REPO=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)          ENGINE="$2"; shift 2 ;;
        --no-nvidia-repo)  ADD_NVIDIA_REPO=0; shift ;;
        -h|--help)         sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$ENGINE" in
    podman|docker) ;;
    *) echo "[install-container-deps] unknown --engine: $ENGINE (expected podman|docker)" >&2; exit 1 ;;
esac

# ── Detect distro ───────────────────────────────────────────────────────────
if [[ ! -f /etc/os-release ]]; then
    echo "[install-container-deps] Cannot detect distro: /etc/os-release missing" >&2
    exit 1
fi
# shellcheck source=/dev/null
. /etc/os-release
DISTRO=$ID
DISTRO_LIKE=${ID_LIKE:-}

echo "[install-container-deps] distro=$DISTRO, engine=$ENGINE"

# ── Per-distro packages ─────────────────────────────────────────────────────
install_arch() {
    local pkgs=()
    case "$ENGINE" in
        podman) pkgs+=(podman podman-compose) ;;
        docker) pkgs+=(docker docker-compose docker-buildx) ;;
    esac
    # nvidia-utils provides nvidia-smi (build-container.sh's CUDA_ARCH probe).
    # nvidia-container-toolkit provides nvidia-ctk + the CDI / runtime hook
    # libraries for GPU pass-through.
    pkgs+=(nvidia-utils nvidia-container-toolkit)
    sudo pacman -S --needed --noconfirm "${pkgs[@]}"
}

install_apt() {
    sudo apt-get update

    local pkgs=()
    case "$ENGINE" in
        # podman-compose lags upstream on LTS but covers what
        # build-container.sh exercises (build/run, no fancy flags).
        podman) pkgs+=(podman podman-compose) ;;
        # docker.io = Ubuntu's stock dockerd. The compose v2 plugin name
        # varies (24.04 universe: docker-compose-v2; via Docker's
        # official repo: docker-compose-plugin). Resolved below.
        docker) pkgs+=(docker.io docker-buildx) ;;
    esac

    # nvidia-utils-XXX is suffixed with the loaded driver branch. If a
    # driver is loaded, pin the matching utils branch via
    # /proc/driver/nvidia/version. If no driver is loaded, skip — the
    # toolkit still works without nvidia-smi, it just means
    # build-container.sh can't autodetect CUDA_ARCH (defaults to sm_89).
    local drv_major=""
    if [[ -r /proc/driver/nvidia/version ]]; then
        drv_major=$(grep -oE '[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null \
                    | head -1 | cut -d. -f1)
    fi
    if [[ -n "$drv_major" ]]; then
        pkgs+=("nvidia-utils-$drv_major")
    else
        echo "[install-container-deps] No loaded NVIDIA driver detected via" >&2
        echo "[install-container-deps] /proc/driver/nvidia/version. Skipping" >&2
        echo "[install-container-deps] nvidia-utils-* — install your driver" >&2
        echo "[install-container-deps] first, or pass CUDA_ARCH=NN through the" >&2
        echo "[install-container-deps] env to build-container.sh manually." >&2
    fi

    sudo apt-get install -y --no-install-recommends "${pkgs[@]}"

    if [[ "$ENGINE" == docker ]]; then
        local compose_pkg=""
        for cand in docker-compose-v2 docker-compose-plugin; do
            if apt-cache show "$cand" >/dev/null 2>&1; then
                compose_pkg="$cand"; break
            fi
        done
        if [[ -z "$compose_pkg" ]]; then
            echo "[install-container-deps] No compose v2 package available in apt." >&2
            echo "[install-container-deps] Add Docker's official repo for docker-compose-plugin:" >&2
            echo "[install-container-deps]   https://docs.docker.com/engine/install/ubuntu/" >&2
            echo "[install-container-deps] Or use --engine podman (default; tested with compose.yaml)." >&2
            exit 1
        fi
        sudo apt-get install -y --no-install-recommends "$compose_pkg"
    fi

    # nvidia-container-toolkit isn't in stock Ubuntu/Debian repos. Pull
    # it from NVIDIA's official apt repo (the path NVIDIA's own docs use).
    if [[ $ADD_NVIDIA_REPO -eq 1 ]] \
        && [[ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]]; then
        echo "[install-container-deps] Adding NVIDIA's container-toolkit apt repo to /etc/apt/sources.list.d/."
        sudo install -m 0755 -d /usr/share/keyrings
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --batch --yes --dearmor \
                -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update
    fi
    sudo apt-get install -y --no-install-recommends nvidia-container-toolkit
}

install_dnf() {
    local pkgs=()
    case "$ENGINE" in
        podman)
            # Fedora's first-class engine — both packages are in stock
            # repos (podman is default container tool on Fedora 36+).
            pkgs+=(podman podman-compose)
            ;;
        docker)
            # docker isn't in Fedora/RHEL stock repos; user has to add
            # docker-ce.repo per Docker's docs first. Bail rather than
            # silently fail.
            if ! sudo dnf list --installed docker-ce >/dev/null 2>&1 \
                && ! sudo dnf list --installed docker        >/dev/null 2>&1; then
                echo "[install-container-deps] Docker is not in Fedora/RHEL stock repos." >&2
                echo "[install-container-deps] Add docker-ce.repo per Docker's docs first," >&2
                echo "[install-container-deps] then re-run this script. Or use --engine podman" >&2
                echo "[install-container-deps] (default; Fedora's first-class engine)." >&2
                exit 1
            fi
            pkgs+=(docker-compose-plugin docker-buildx-plugin)
            ;;
    esac

    # Hint only — Fedora's nvidia driver lives in RPMFusion and
    # auto-enabling third-party repos behind the user's back is rude.
    if ! command -v nvidia-smi >/dev/null; then
        echo "[install-container-deps] WARNING: nvidia-smi not on PATH." >&2
        echo "[install-container-deps] Enable RPMFusion + install akmod-nvidia (or" >&2
        echo "[install-container-deps] akmod-nvidia-open) for the host driver, or" >&2
        echo "[install-container-deps] pass CUDA_ARCH=NN through the env to" >&2
        echo "[install-container-deps] build-container.sh manually." >&2
    fi

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        sudo dnf install -y "${pkgs[@]}"
    fi

    if [[ $ADD_NVIDIA_REPO -eq 1 ]] \
        && [[ ! -f /etc/yum.repos.d/nvidia-container-toolkit.repo ]]; then
        echo "[install-container-deps] Adding NVIDIA's container-toolkit dnf repo to /etc/yum.repos.d/."
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
            | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
    fi
    sudo dnf install -y nvidia-container-toolkit
}

# ── Distro-agnostic post-install ────────────────────────────────────────────
configure_nvidia_runtime() {
    if ! command -v nvidia-ctk >/dev/null; then
        echo "[install-container-deps] WARNING: nvidia-ctk not on PATH — skipping CDI / runtime setup." >&2
        return
    fi
    case "$ENGINE" in
        podman)
            # CDI spec at /etc/cdi/nvidia.yaml lets compose.yaml's
            # `devices: - nvidia.com/gpu=all` resolve to real GPUs.
            # Re-run after driver upgrades — the spec hard-codes
            # device file paths.
            sudo install -m 0755 -d /etc/cdi
            sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
            echo "[install-container-deps] Generated CDI spec at /etc/cdi/nvidia.yaml."
            ;;
        docker)
            # Writes /etc/docker/daemon.json's `runtimes.nvidia` entry +
            # restarts dockerd so the change takes effect.
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker || true
            echo "[install-container-deps] Configured docker NVIDIA runtime + restarted dockerd."
            ;;
    esac
}

enable_docker_service() {
    [[ "$ENGINE" == docker ]] || return 0
    command -v systemctl >/dev/null || return 0
    sudo systemctl enable --now docker.service || true
}

# ── Distro dispatch ─────────────────────────────────────────────────────────
case "$DISTRO" in
    arch|cachyos|manjaro|endeavouros)            install_arch ;;
    ubuntu|debian|pop|linuxmint)                 install_apt  ;;
    fedora|rhel|centos|rocky|almalinux)          install_dnf  ;;
    *)
        case "$DISTRO_LIKE" in
            *arch*)            install_arch ;;
            *debian*)          install_apt  ;;
            *rhel*|*fedora*)   install_dnf  ;;
            *)
                echo "[install-container-deps] Unknown distro '$DISTRO'. Install equivalents of:"
                if [[ "$ENGINE" == podman ]]; then
                    echo "  podman + podman-compose"
                else
                    echo "  docker + docker-compose-v2 (or docker-compose-plugin) + docker-buildx"
                fi
                echo "  nvidia-container-toolkit (from NVIDIA's repo: https://nvidia.github.io/libnvidia-container/)"
                exit 1
                ;;
        esac
        ;;
esac

enable_docker_service
configure_nvidia_runtime

# ── Final notes ─────────────────────────────────────────────────────────────
echo
echo "[install-container-deps] Done."
echo "  Build the image:"
echo "    ./scripts/build-container.sh --engine $ENGINE"
echo "  After future NVIDIA driver upgrades, re-run this script (or"
echo "  re-run nvidia-ctk cdi generate / nvidia-ctk runtime configure"
echo "  manually) so the CDI spec / docker runtime hook stays current."
