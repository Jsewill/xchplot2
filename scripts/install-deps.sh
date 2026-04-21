#!/usr/bin/env bash
#
# install-deps.sh — bootstrap xchplot2's native build dependencies.
#
# Installs CUDA Toolkit (or CUDA *headers*-only on AMD systems), LLVM 18+,
# AdaptiveCpp 25.10, and a Rust toolchain via rustup. After this completes,
# you can build with either:
#   cargo install --git https://github.com/Jsewill/xchplot2
#   # or:
#   cmake -B build -S . && cmake --build build -j
#
# Usage:
#   scripts/install-deps.sh                # auto-detect distro + GPU
#   scripts/install-deps.sh --no-acpp      # skip AdaptiveCpp build (use FetchContent)
#   scripts/install-deps.sh --gpu amd      # force AMD path (CUDA headers only)
#   scripts/install-deps.sh --gpu nvidia   # force NVIDIA path (full CUDA Toolkit)
#
# Supported distros: Arch family, Ubuntu/Debian, Fedora/RHEL.
# For anything else, install the equivalents listed at the bottom and
# build AdaptiveCpp from source manually.

set -euo pipefail

ACPP_REF=${ACPP_REF:-v25.10.0}
ACPP_PREFIX=${ACPP_PREFIX:-/opt/adaptivecpp}
SKIP_ACPP=0
GPU=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-acpp)   SKIP_ACPP=1; shift ;;
        --gpu)       GPU="$2"; shift 2 ;;
        -h|--help)   sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Detect distro ───────────────────────────────────────────────────────────
if [[ ! -f /etc/os-release ]]; then
    echo "Cannot detect distro: /etc/os-release missing" >&2
    exit 1
fi
. /etc/os-release
DISTRO=$ID
DISTRO_LIKE=${ID_LIKE:-}

# ── Detect GPU vendor (NVIDIA vs AMD) ───────────────────────────────────────
if [[ -z "$GPU" ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        GPU=nvidia
    elif command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q gfx; then
        GPU=amd
    else
        echo "[install-deps] No GPU detected. Defaulting to nvidia (full CUDA install)."
        echo "[install-deps] Override with --gpu amd if this is an AMD-only host."
        GPU=nvidia
    fi
fi
echo "[install-deps] distro=$DISTRO, gpu=$GPU, acpp=${ACPP_REF}, prefix=${ACPP_PREFIX}"

# ── Per-distro packages ─────────────────────────────────────────────────────
install_arch() {
    local pkgs=(cmake git base-devel python ninja
                llvm clang lld
                boost numactl curl)
    case "$GPU" in
        nvidia) pkgs+=(cuda) ;;
        # rocminfo: needed by build-container.sh + scripts/install-deps.sh
        # autodetection (rocm-hip-sdk doesn't pull it transitively).
        amd)    pkgs+=(rocm-hip-sdk rocm-device-libs rocminfo cuda) ;;  # cuda for headers
    esac
    sudo pacman -S --needed --noconfirm "${pkgs[@]}"
}

install_apt() {
    local pkgs=(cmake git ninja-build build-essential python3 pkg-config
                llvm-18 llvm-18-dev clang-18 libclang-18-dev libclang-cpp18-dev
                libboost-context-dev libnuma-dev libomp-18-dev curl ca-certificates)
    case "$GPU" in
        nvidia) pkgs+=(nvidia-cuda-toolkit) ;;
        amd)    pkgs+=(rocm-hip-sdk rocm-libs rocminfo nvidia-cuda-toolkit-headers)
                # rocminfo is the discovery tool build-container.sh probes;
                # not pulled in transitively by rocm-hip-sdk.
                # nvidia-cuda-toolkit-headers may not exist on all releases;
                # fall back to the full toolkit (headers only used).
                ;;
    esac
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${pkgs[@]}" || {
        if [[ "$GPU" == "amd" ]]; then
            echo "[install-deps] retrying with full nvidia-cuda-toolkit (headers only used)"
            sudo apt-get install -y --no-install-recommends nvidia-cuda-toolkit
        else
            exit 1
        fi
    }
}

install_dnf() {
    local pkgs=(cmake git ninja-build gcc-c++ python3 pkg-config
                llvm llvm-devel clang clang-devel
                boost-devel numactl-devel libomp-devel curl)
    case "$GPU" in
        nvidia) pkgs+=(cuda-toolkit) ;;
        amd)    pkgs+=(rocm-hip-devel rocminfo cuda-toolkit) ;;  # cuda for headers
    esac
    sudo dnf install -y "${pkgs[@]}"
}

case "$DISTRO" in
    arch|cachyos|manjaro|endeavouros)            install_arch ;;
    ubuntu|debian|pop|linuxmint)                 install_apt  ;;
    fedora|rhel|centos|rocky|almalinux)          install_dnf  ;;
    *)
        case "$DISTRO_LIKE" in
            *arch*)   install_arch ;;
            *debian*) install_apt  ;;
            *rhel*|*fedora*) install_dnf ;;
            *)
                echo "[install-deps] Unknown distro '$DISTRO'. Install equivalents of:"
                echo "  CMake ≥ 3.24, Ninja, LLVM 18+, clang 18+, libclang dev,"
                echo "  Boost.Context, libnuma, libomp, Python 3, git,"
                if [[ "$GPU" == "nvidia" ]]; then
                    echo "  CUDA Toolkit 12+ (with nvcc)"
                else
                    echo "  ROCm 6+ HIP SDK + CUDA Toolkit *headers* (no driver needed)"
                fi
                echo "Then re-run with --no-acpp to skip pkg install and only build AdaptiveCpp."
                exit 1
                ;;
        esac
        ;;
esac

# ── Rust toolchain via rustup ───────────────────────────────────────────────
if ! command -v cargo >/dev/null; then
    echo "[install-deps] Installing Rust toolchain via rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal
    export PATH=$HOME/.cargo/bin:$PATH
fi

# ── AdaptiveCpp ─────────────────────────────────────────────────────────────
if [[ $SKIP_ACPP -eq 1 ]]; then
    echo "[install-deps] Skipping AdaptiveCpp build per --no-acpp."
    echo "[install-deps] CMakeLists will FetchContent it automatically (slow first build)."
    exit 0
fi

if [[ -d "$ACPP_PREFIX" ]] && [[ -f "$ACPP_PREFIX/lib/cmake/AdaptiveCpp/AdaptiveCppConfig.cmake" ]]; then
    echo "[install-deps] AdaptiveCpp already installed at $ACPP_PREFIX. Skipping."
    exit 0
fi

ACPP_BUILD_DIR=$(mktemp -d -t xchplot2-acpp-XXXXXX)
trap "rm -rf $ACPP_BUILD_DIR" EXIT

echo "[install-deps] Building AdaptiveCpp $ACPP_REF in $ACPP_BUILD_DIR"
git clone --depth 1 --branch "$ACPP_REF" \
    https://github.com/AdaptiveCpp/AdaptiveCpp.git "$ACPP_BUILD_DIR/src"

cmake -S "$ACPP_BUILD_DIR/src" -B "$ACPP_BUILD_DIR/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ACPP_PREFIX"
cmake --build "$ACPP_BUILD_DIR/build" --parallel
sudo cmake --install "$ACPP_BUILD_DIR/build"

echo
echo "[install-deps] Done."
echo "  AdaptiveCpp: $ACPP_PREFIX"
echo "  Build xchplot2:"
echo "    export CMAKE_PREFIX_PATH=$ACPP_PREFIX:\$CMAKE_PREFIX_PATH"
echo "    cargo install --path .                  # or:"
echo "    cmake -B build -S . && cmake --build build -j"
