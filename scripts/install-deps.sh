#!/usr/bin/env bash
#
# install-deps.sh — bootstrap xchplot2's native build dependencies.
#
# Installs CUDA Toolkit on NVIDIA, ROCm HIP SDK on AMD, LLVM 18+,
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
# shellcheck source=/dev/null
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
        # No CUDA pkg on the AMD path — CudaHalfShim.hpp guards the CUDA
        # headers via __has_include, and pulling CUDA alongside HIP causes
        # uchar1/char1 typedef redefinitions.
        amd)    pkgs+=(rocm-hip-sdk rocm-device-libs rocminfo) ;;
    esac
    sudo pacman -S --needed --noconfirm "${pkgs[@]}"
}

install_apt() {
    local pkgs=(cmake git ninja-build build-essential python3 pkg-config
                llvm-18 llvm-18-dev clang-18 lld-18 libclang-18-dev libclang-cpp18-dev
                libboost-context-dev libnuma-dev libomp-18-dev curl ca-certificates)
    case "$GPU" in
        nvidia) pkgs+=(nvidia-cuda-toolkit) ;;
        amd)    pkgs+=(rocm-hip-sdk rocm-libs rocminfo)
                # rocminfo is the discovery tool build-container.sh probes;
                # not pulled in transitively by rocm-hip-sdk.
                # No nvidia-cuda-toolkit-headers on the AMD path —
                # CudaHalfShim.hpp guards the CUDA headers via
                # __has_include, and pulling CUDA alongside HIP causes
                # uchar1/char1 typedef redefinitions.
                ;;
    esac
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${pkgs[@]}"
}

install_dnf() {
    local pkgs=(cmake git ninja-build gcc-c++ python3 pkg-config
                llvm llvm-devel clang clang-devel lld
                boost-devel numactl-devel libomp-devel curl)
    case "$GPU" in
        nvidia) pkgs+=(cuda-toolkit) ;;
        # No cuda-toolkit on the AMD path — CudaHalfShim.hpp guards the
        # CUDA headers via __has_include, and pulling CUDA alongside HIP
        # causes uchar1/char1 typedef redefinitions.
        amd)    pkgs+=(rocm-hip-devel rocminfo) ;;
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
                    echo "  ROCm 6+ HIP SDK (rocm-hip-sdk / rocm-hip-devel)"
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
trap 'rm -rf "$ACPP_BUILD_DIR"' EXIT

# ── Find a compatible LLVM ──────────────────────────────────────────────────
# AdaptiveCpp 25.10 only supports LLVM 16-20. On rolling distros (Arch,
# Fedora rawhide) the system LLVM is often 21+, which AdaptiveCpp rejects
# with "LLVM versions greater than 20 are not yet tested/supported". Probe
# the conventional install prefixes for the newest usable LLVM and pin
# AdaptiveCpp to it explicitly. Fail fast with a distro-specific install
# hint rather than letting AdaptiveCpp's CMake fail mid-configure.
LLVM_ROOT=""
for cand in \
    /usr/lib/llvm-20 /usr/lib/llvm-19 /usr/lib/llvm-18 \
    /usr/lib/llvm-17 /usr/lib/llvm-16 \
    /usr/lib/llvm20  /usr/lib/llvm19  /usr/lib/llvm18 \
    /usr/lib64/llvm20 /usr/lib64/llvm19 /usr/lib64/llvm18 \
    /opt/llvm20 /opt/llvm-20 /opt/llvm19 /opt/llvm-19 \
    /opt/llvm18 /opt/llvm-18; do
    if [[ -x "$cand/bin/clang" ]] && [[ -x "$cand/bin/ld.lld" ]]; then
        ver=$("$cand/bin/clang" --version 2>/dev/null \
              | head -1 | grep -oE 'version [0-9]+' | grep -oE '[0-9]+')
        if [[ -n "$ver" ]] && (( ver >= 16 && ver <= 20 )); then
            LLVM_ROOT="$cand"
            break
        fi
    fi
done

if [[ -z "$LLVM_ROOT" ]]; then
    echo "[install-deps] No compatible LLVM (16-20) with ld.lld found." >&2
    echo "[install-deps] AdaptiveCpp $ACPP_REF only supports LLVM 16-20." >&2
    echo "[install-deps] Install one and re-run, or use the container path:" >&2
    case "$DISTRO" in
        arch|cachyos|manjaro|endeavouros)
            echo "  yay -S llvm18-bin lld18-bin   # or paru -S, or any AUR helper" >&2 ;;
        ubuntu|debian|pop|linuxmint)
            echo "  sudo apt install llvm-18 llvm-18-dev clang-18 lld-18 libomp-18-dev" >&2 ;;
        fedora|rhel|centos|rocky|almalinux)
            echo "  sudo dnf install llvm18 llvm18-devel clang18 lld18-devel" >&2 ;;
        *)
            echo "  install LLVM 16-20 + clang + ld.lld for your distro" >&2 ;;
    esac
    echo "  ./scripts/build-container.sh   # container has LLVM 18 pinned" >&2
    exit 1
fi
echo "[install-deps] Using LLVM at $LLVM_ROOT for AdaptiveCpp build."

# ── ROCm device libs path (AMD only) ────────────────────────────────────────
# AdaptiveCpp's HIP backend needs ockl.bc / ocml.bc to compile kernels for
# amdgcn. The bitcode location moved between ROCm versions; probe the
# common spots. CMake will warn if the path's missing on AMD; without a
# match here, the build fails with "ROCm device library path not found".
ACPP_ROCM_FLAGS=()
if [[ "$GPU" == "amd" ]]; then
    for d in \
        /opt/rocm/amdgcn/bitcode \
        /opt/rocm/lib/llvm-amdgpu/amdgcn/bitcode \
        /opt/rocm/share/amdgcn/bitcode; do
        if [[ -f "$d/ockl.bc" ]]; then
            ACPP_ROCM_FLAGS+=(-DROCM_DEVICE_LIBS_PATH="$d")
            echo "[install-deps] ROCm device libs: $d"
            break
        fi
    done
fi

echo "[install-deps] Building AdaptiveCpp $ACPP_REF in $ACPP_BUILD_DIR"
git clone --depth 1 --branch "$ACPP_REF" \
    https://github.com/AdaptiveCpp/AdaptiveCpp.git "$ACPP_BUILD_DIR/src"

# AMD-only builds don't need AdaptiveCpp's CUDA backend. Skip the
# `find_package(CUDA)` probe that AdaptiveCpp's CMakeLists runs at
# line ~122: on hosts where a CUDA headers subset is installed (distro
# `cuda` package, JetPack fragments, /usr/lib from some wrappers), the
# probe finds a partial install and AdaptiveCpp's own `FindCUDA.cmake`
# emits `CUDAToolkit_LIBRARY_ROOT /usr/lib does not point to the
# correct directory, try setting it manually`. The warning is cosmetic
# (AdaptiveCpp continues without CUDA), but it looks like an error to
# users skimming the install log.
ACPP_CUDA_DISABLE=()
if [[ "$GPU" == "amd" ]]; then
    ACPP_CUDA_DISABLE+=(-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE)
fi

cmake -S "$ACPP_BUILD_DIR/src" -B "$ACPP_BUILD_DIR/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ACPP_PREFIX" \
    -DCMAKE_C_COMPILER="$LLVM_ROOT/bin/clang" \
    -DCMAKE_CXX_COMPILER="$LLVM_ROOT/bin/clang++" \
    -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" \
    -DACPP_LLD_PATH="$LLVM_ROOT/bin/ld.lld" \
    "${ACPP_CUDA_DISABLE[@]}" \
    "${ACPP_ROCM_FLAGS[@]}"
cmake --build "$ACPP_BUILD_DIR/build" --parallel
sudo cmake --install "$ACPP_BUILD_DIR/build"

echo
echo "[install-deps] Done."
echo "  AdaptiveCpp: $ACPP_PREFIX"
echo "  Build xchplot2:"
echo "    export CMAKE_PREFIX_PATH=$ACPP_PREFIX:\$CMAKE_PREFIX_PATH"
echo "    cargo install --path .                  # or:"
echo "    cmake -B build -S . && cmake --build build -j"
