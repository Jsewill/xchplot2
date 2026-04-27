#!/usr/bin/env bash
#
# install-container-deps.sh — bootstrap the host packages required to
# build & run xchplot2's container images via scripts/build-container.sh.
#
# Native build deps (CUDA Toolkit, ROCm SDK, LLVM 18+, AdaptiveCpp,
# Boost.Context, libnuma, libomp, Rust) all live INSIDE the container
# image — the host does not need any of them. This script only
# installs:
#   1. A container engine + compose plugin: `podman` + `podman-compose`
#      (default), or `docker` + the `docker compose` v2 plugin via
#      `--engine docker`.
#   2. The GPU discovery tool used by build-container.sh's autodetect
#      (`nvidia-smi` for NVIDIA, `rocminfo` for AMD). build-container.sh
#      *errors* on AMD if ACPP_GFX can't be resolved, so rocminfo isn't
#      strictly optional unless you pass ACPP_GFX through the env.
#   3. The GPU container runtime: `nvidia-container-toolkit` + a CDI
#      spec at /etc/cdi/nvidia.yaml (podman) or the docker runtime hook
#      (docker) for NVIDIA. AMD and Intel only need /dev/kfd | /dev/dri
#      access via the `video` and `render` groups; this script adds
#      the invoking user to both.
#
# For NATIVE host builds (no container) use scripts/install-deps.sh
# instead — that path needs the full CUDA / ROCm / LLVM / AdaptiveCpp
# stack on the host and takes 30-45 min on a first run.
#
# Usage:
#   scripts/install-container-deps.sh                  # auto-detect distro + GPU
#   scripts/install-container-deps.sh --gpu nvidia
#   scripts/install-container-deps.sh --gpu amd
#   scripts/install-container-deps.sh --gpu intel
#   scripts/install-container-deps.sh --gpu cpu        # engine only, no GPU runtime
#   scripts/install-container-deps.sh --engine docker  # docker instead of podman
#   scripts/install-container-deps.sh --no-nvidia-repo # skip adding NVIDIA's apt/dnf repo
#   scripts/install-container-deps.sh --dry-run        # print the plan, change nothing
#
# Supported distros: Arch family, Ubuntu/Debian, Fedora/RHEL.

set -euo pipefail

ENGINE=podman
GPU=""
ADD_NVIDIA_REPO=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)             GPU="$2";    shift 2 ;;
        --engine)          ENGINE="$2"; shift 2 ;;
        --no-nvidia-repo)  ADD_NVIDIA_REPO=0; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$ENGINE" in
    podman|docker) ;;
    *) echo "[install-container-deps] unknown --engine: $ENGINE (expected podman|docker)" >&2; exit 1 ;;
esac

# ── Helpers ─────────────────────────────────────────────────────────────────
# In dry-run mode every mutating call is replaced with a `+ sudo …` stub;
# probes (`command -v`, `[[ -f ]]`, etc.) still run as normal because they
# don't change host state and the planning logic depends on them. The `+ `
# prefix mirrors `set -x`'s syntax so dry-run output reads as an executable
# trace.
sudo_or_dry() {
    if (( DRY_RUN )); then
        printf '+ sudo %s\n' "$*"
    else
        sudo "$@"
    fi
}

apt_update_or_dry() {
    if (( DRY_RUN )); then
        printf '+ sudo apt-get update\n'
    else
        sudo apt-get update
    fi
}

# Curl-piped-to-(sudo tee | sudo gpg --dearmor) write. Records "+ write
# DEST (from URL)" in dry-run mode. `mode=dearmor` covers the apt
# gpgkey path; default mode is plain tee.
write_url_or_dry() {
    local url="$1" dest="$2" mode="${3:-cat}"
    if (( DRY_RUN )); then
        case "$mode" in
            dearmor) printf '+ write %s (gpg --dearmor from %s)\n' "$dest" "$url" ;;
            *)       printf '+ write %s (from %s)\n' "$dest" "$url" ;;
        esac
        return
    fi
    case "$mode" in
        dearmor)
            curl -fsSL "$url" \
                | sudo gpg --batch --yes --dearmor -o "$dest"
            ;;
        *)
            curl -fsSL "$url" | sudo tee "$dest" >/dev/null
            ;;
    esac
}

# ── Detect distro ───────────────────────────────────────────────────────────
if [[ ! -f /etc/os-release ]]; then
    echo "[install-container-deps] Cannot detect distro: /etc/os-release missing" >&2
    exit 1
fi
# shellcheck source=/dev/null
. /etc/os-release
DISTRO=$ID
DISTRO_LIKE=${ID_LIKE:-}

# ── Detect GPU vendor ───────────────────────────────────────────────────────
# Two-tier strategy mirroring install-deps.sh: tool-based first (authoritative
# when the driver is loaded), PCI vendor-ID fallback (works pre-driver). The
# driver tools cannot be a hard prerequisite because installing them is one
# of the things this script is supposed to do.
detect_gpu_via_pci() {
    local found="" entry name vendor
    for entry in /sys/class/drm/card*; do
        name=$(basename "$entry")
        # Skip connector entries like card0-DP-1; only the bare cardN
        # nodes carry a `device/vendor` attribute we can read.
        [[ "$name" =~ ^card[0-9]+$ ]] || continue
        [[ -r "$entry/device/vendor" ]] || continue
        vendor=$(cat "$entry/device/vendor" 2>/dev/null)
        case "$vendor" in
            0x10de) found="nvidia"; break ;;            # highest precedence
            0x1002) found="amd" ;;                      # overrides intel
            0x8086) [[ -z "$found" ]] && found="intel" ;; # only if nothing else
        esac
    done
    echo "$found"
}

# Skip autodetect under --dry-run — CI containers have no GPU, and tests
# always pass --gpu explicitly. Avoids "could not auto-detect" exit on
# headless runners.
if [[ -z "$GPU" ]] && (( ! DRY_RUN )); then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        GPU=nvidia
        echo "[install-container-deps] Detected NVIDIA GPU (nvidia-smi)."
    elif command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q gfx; then
        GPU=amd
        echo "[install-container-deps] Detected AMD GPU (rocminfo)."
    else
        GPU=$(detect_gpu_via_pci)
        if [[ -n "$GPU" ]]; then
            echo "[install-container-deps] Detected $GPU GPU via /sys/class/drm (PCI vendor ID); driver tools not yet installed."
        fi
    fi
fi

if [[ -z "$GPU" ]]; then
    if (( DRY_RUN )); then
        echo "[install-container-deps] --dry-run requires --gpu to be set explicitly" >&2
    else
        echo "[install-container-deps] Could not auto-detect a GPU. Pass" >&2
        echo "[install-container-deps]   --gpu nvidia | amd | intel | cpu" >&2
        echo "[install-container-deps] explicitly. Use --gpu cpu for a GPU-less host" >&2
        echo "[install-container-deps] (CPU-only image; slow plotting, see README)." >&2
    fi
    exit 1
fi

case "$GPU" in
    nvidia|amd|intel|cpu) ;;
    *) echo "[install-container-deps] unknown --gpu: $GPU (expected nvidia|amd|intel|cpu)" >&2; exit 1 ;;
esac

echo "[install-container-deps] distro=$DISTRO, gpu=$GPU, engine=$ENGINE"

# ── Per-distro packages ─────────────────────────────────────────────────────
install_arch() {
    local pkgs=()
    case "$ENGINE" in
        podman) pkgs+=(podman podman-compose) ;;
        docker) pkgs+=(docker docker-compose docker-buildx) ;;
    esac
    case "$GPU" in
        # nvidia-utils provides nvidia-smi (used by build-container.sh's
        # CUDA_ARCH probe). nvidia-container-toolkit provides nvidia-ctk +
        # the CDI / runtime hook libraries for GPU pass-through.
        nvidia) pkgs+=(nvidia-utils nvidia-container-toolkit) ;;
        # rocminfo: build-container.sh fails fast on AMD if ACPP_GFX can't
        # be resolved from rocminfo (compose.yaml's ACPP_TARGETS default
        # is a deliberately invalid placeholder so wrong-arch builds fail
        # loudly instead of silently producing no-op kernels).
        # No ROCm SDK on the host — that lives inside the container.
        amd)    pkgs+=(rocminfo) ;;
    esac
    sudo_or_dry pacman -S --needed --noconfirm "${pkgs[@]}"
}

install_apt() {
    apt_update_or_dry

    local pkgs=()
    case "$ENGINE" in
        # podman-compose lags upstream on LTS but covers what
        # build-container.sh exercises (build/run, no fancy flags).
        podman) pkgs+=(podman podman-compose) ;;
        # docker.io = Ubuntu's stock dockerd. The compose v2 plugin name
        # varies (24.04: docker-compose-v2 in universe; via Docker's
        # official repo: docker-compose-plugin). Resolved below.
        docker) pkgs+=(docker.io docker-buildx) ;;
    esac
    case "$GPU" in
        nvidia)
            # nvidia-utils-XXX is suffixed with the loaded driver branch.
            # If a driver is already loaded, pin the matching utils branch
            # via /proc/driver/nvidia/version. If no driver is loaded, skip
            # — nvidia-container-toolkit still works without nvidia-smi,
            # it just means build-container.sh can't autodetect CUDA_ARCH.
            local drv_major=""
            if (( DRY_RUN )); then
                # Use a placeholder so dry-run output stays deterministic
                # regardless of whether the runner has a driver loaded.
                drv_major="<DRV_MAJOR>"
            elif [[ -r /proc/driver/nvidia/version ]]; then
                drv_major=$(grep -oE '[0-9]+\.[0-9]+' /proc/driver/nvidia/version 2>/dev/null \
                            | head -1 | cut -d. -f1)
            fi
            if [[ -n "$drv_major" ]]; then
                pkgs+=("nvidia-utils-$drv_major")
            else
                echo "[install-container-deps] No loaded NVIDIA driver detected via" >&2
                echo "[install-container-deps] /proc/driver/nvidia/version. Skipping" >&2
                echo "[install-container-deps] nvidia-utils-* — install your driver" >&2
                echo "[install-container-deps] first, or pass --gpu nvidia + CUDA_ARCH" >&2
                echo "[install-container-deps] manually to build-container.sh." >&2
            fi
            ;;
        amd) pkgs+=(rocminfo) ;;
    esac
    sudo_or_dry apt-get install -y --no-install-recommends "${pkgs[@]}"

    # Docker compose v2 plugin: the package name varies by source.
    # `docker-compose-v2` ships in 24.04+ universe; `docker-compose-plugin`
    # ships in Docker's official deb repo. Both install the same binary at
    # /usr/libexec/docker/cli-plugins/docker-compose. build-container.sh
    # uses the v2 `docker compose <subcmd>` syntax, so we MUST install one
    # of these two — the legacy v1 `docker-compose` (Python) won't work.
    if [[ "$ENGINE" == docker ]]; then
        local compose_pkg="docker-compose-v2"
        if (( ! DRY_RUN )); then
            compose_pkg=""
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
        fi
        sudo_or_dry apt-get install -y --no-install-recommends "$compose_pkg"
    fi

    # nvidia-container-toolkit isn't in stock Ubuntu/Debian repos. Pull it
    # from NVIDIA's official apt repo (the path NVIDIA's own docs use).
    if [[ "$GPU" == nvidia ]]; then
        if [[ $ADD_NVIDIA_REPO -eq 1 ]] \
            && { (( DRY_RUN )) || [[ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]]; }; then
            echo "[install-container-deps] Adding NVIDIA's container-toolkit apt repo to /etc/apt/sources.list.d/."
            sudo_or_dry install -m 0755 -d /usr/share/keyrings
            write_url_or_dry \
                https://nvidia.github.io/libnvidia-container/gpgkey \
                /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
                dearmor
            # The repo file gets a sed transform to inject signed-by= ;
            # in dry-run we record the URL → dest, which is the bit
            # users actually care about.
            if (( DRY_RUN )); then
                write_url_or_dry \
                    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                    /etc/apt/sources.list.d/nvidia-container-toolkit.list
            else
                curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
            fi
            apt_update_or_dry
        fi
        sudo_or_dry apt-get install -y --no-install-recommends nvidia-container-toolkit
    fi
}

install_dnf() {
    local pkgs=()
    case "$ENGINE" in
        podman)
            # Fedora's first-class engine — both packages are in the stock
            # repos (podman is the default container tool on Fedora 36+).
            pkgs+=(podman podman-compose)
            ;;
        docker)
            # docker isn't in Fedora/RHEL stock repos; the user has to add
            # docker-ce.repo per Docker's docs first. Bail rather than
            # silently fail mid-install. Skip the precondition check in
            # dry-run so the planning output stays useful even in CI
            # containers that haven't added the repo.
            if (( ! DRY_RUN )); then
                if ! sudo dnf list --installed docker-ce >/dev/null 2>&1 \
                    && ! sudo dnf list --installed docker        >/dev/null 2>&1; then
                    echo "[install-container-deps] Docker is not in Fedora/RHEL stock repos." >&2
                    echo "[install-container-deps] Add docker-ce.repo per Docker's docs first," >&2
                    echo "[install-container-deps] then re-run this script. Or use --engine podman" >&2
                    echo "[install-container-deps] (default; Fedora's first-class engine)." >&2
                    exit 1
                fi
            fi
            pkgs+=(docker-compose-plugin docker-buildx-plugin)
            ;;
    esac
    case "$GPU" in
        nvidia)
            # Hint only — Fedora's nvidia driver lives in RPMFusion and
            # auto-enabling third-party repos behind the user's back is
            # rude. nvidia-container-toolkit (added below) comes from
            # NVIDIA's own repo, which is already a precedent set by
            # NVIDIA's docs.
            if (( ! DRY_RUN )) && ! command -v nvidia-smi >/dev/null; then
                echo "[install-container-deps] WARNING: nvidia-smi not on PATH." >&2
                echo "[install-container-deps] Enable RPMFusion + install akmod-nvidia (or" >&2
                echo "[install-container-deps] akmod-nvidia-open) for the host driver, or" >&2
                echo "[install-container-deps] pass --gpu nvidia + CUDA_ARCH manually." >&2
            fi
            ;;
        amd) pkgs+=(rocminfo) ;;
    esac
    if [[ ${#pkgs[@]} -gt 0 ]]; then
        sudo_or_dry dnf install -y "${pkgs[@]}"
    fi

    if [[ "$GPU" == nvidia ]]; then
        if [[ $ADD_NVIDIA_REPO -eq 1 ]] \
            && { (( DRY_RUN )) || [[ ! -f /etc/yum.repos.d/nvidia-container-toolkit.repo ]]; }; then
            echo "[install-container-deps] Adding NVIDIA's container-toolkit dnf repo to /etc/yum.repos.d/."
            write_url_or_dry \
                https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
                /etc/yum.repos.d/nvidia-container-toolkit.repo
        fi
        sudo_or_dry dnf install -y nvidia-container-toolkit
    fi
}

# ── Distro-agnostic post-install (NVIDIA only) ──────────────────────────────
configure_nvidia_runtime() {
    if (( ! DRY_RUN )) && ! command -v nvidia-ctk >/dev/null; then
        echo "[install-container-deps] WARNING: nvidia-ctk not on PATH — skipping CDI / runtime setup." >&2
        return
    fi
    case "$ENGINE" in
        podman)
            # CDI spec at /etc/cdi/nvidia.yaml lets `--device nvidia.com/gpu=all`
            # (and the `deploy.resources.reservations.devices` shorthand in
            # compose.yaml's cuda service) resolve to real GPUs. Re-run after
            # driver upgrades — the spec hard-codes device file paths.
            sudo_or_dry install -m 0755 -d /etc/cdi
            sudo_or_dry nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
            echo "[install-container-deps] Generated CDI spec at /etc/cdi/nvidia.yaml."
            # nvidia-ctk's "discoverer" enumerates every NVIDIA-related path
            # the driver could expose — Vulkan ICDs, X11 configs, the
            # fabric-manager / MPS / IMEX sockets, etc. — and prints WARN
            # lines for ones it can't find. On any non-server, headless
            # GPU host most of these won't be present; the spec gracefully
            # omits them. Tell the user up front so the WARN volume on the
            # next line doesn't look like a failure.
            echo "[install-container-deps] (WARNings about libnvidia-vulkan-producer / X11 configs /"
            echo "[install-container-deps]  fabric-manager / MPS / IMEX from nvidia-ctk are expected on"
            echo "[install-container-deps]  non-server hosts — those are optional features the spec"
            echo "[install-container-deps]  gracefully omits when not present.)"
            ;;
        docker)
            # Writes /etc/docker/daemon.json's `runtimes.nvidia` entry +
            # restarts dockerd so the change takes effect.
            sudo_or_dry nvidia-ctk runtime configure --runtime=docker
            if (( DRY_RUN )); then
                printf '+ sudo systemctl restart docker\n'
            else
                sudo systemctl restart docker || true
            fi
            echo "[install-container-deps] Configured docker NVIDIA runtime + restarted dockerd."
            ;;
    esac
}

# ── Distro-agnostic post-install (AMD / Intel) ──────────────────────────────
# /dev/kfd (AMD) and /dev/dri (AMD + Intel) are group-owned by `video` (and
# `render` on newer udev/systemd setups). Add the invoking user to both so
# rootless containers can pass the device through. Effective on next login.
add_user_to_video_render_groups() {
    local target_user
    if (( DRY_RUN )); then
        # Stable placeholder so the fixture doesn't depend on $USER.
        target_user="<USER>"
    else
        target_user="${SUDO_USER:-${USER:-}}"
        if [[ -z "$target_user" || "$target_user" == root ]]; then
            echo "[install-container-deps] Skipping group membership (no non-root user detected)."
            return
        fi
    fi
    for grp in video render; do
        if (( ! DRY_RUN )); then
            getent group "$grp" >/dev/null 2>&1 || continue
            if id -nG "$target_user" | tr ' ' '\n' | grep -qx "$grp"; then
                continue
            fi
        fi
        sudo_or_dry usermod -aG "$grp" "$target_user"
        echo "[install-container-deps] Added $target_user to group $grp (re-login to apply)."
    done
}

# ── Enable docker daemon when applicable ────────────────────────────────────
enable_docker_service() {
    [[ "$ENGINE" == docker ]] || return 0
    if (( ! DRY_RUN )); then
        command -v systemctl >/dev/null || return 0
    fi
    if (( DRY_RUN )); then
        printf '+ sudo systemctl enable --now docker.service\n'
    else
        sudo systemctl enable --now docker.service || true
    fi
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
                case "$GPU" in
                    nvidia) echo "  nvidia-container-toolkit (from NVIDIA's repo: https://nvidia.github.io/libnvidia-container/)" ;;
                    amd)    echo "  rocminfo (only used by build-container.sh's ACPP_GFX autodetect)" ;;
                esac
                exit 1
                ;;
        esac
        ;;
esac

enable_docker_service

case "$GPU" in
    nvidia)        configure_nvidia_runtime ;;
    amd|intel)     add_user_to_video_render_groups ;;
    cpu)           : ;;
esac

# ── Final notes ─────────────────────────────────────────────────────────────
echo
echo "[install-container-deps] Done."
echo "  Build the image:"
echo "    ./scripts/build-container.sh --engine $ENGINE${GPU:+ --gpu $GPU}"
case "$GPU" in
    amd|intel)
        echo "  If this run added you to the video / render groups, log out"
        echo "  and back in before running plots — group changes only take"
        echo "  effect for fresh login sessions."
        ;;
    nvidia)
        echo "  After future NVIDIA driver upgrades, re-run this script (or"
        echo "  re-run nvidia-ctk cdi generate / nvidia-ctk runtime configure"
        echo "  manually) so the CDI spec / docker runtime hook stays current."
        ;;
esac
