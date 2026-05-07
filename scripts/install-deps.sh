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
#   scripts/install-deps.sh --rebuild-acpp # wipe + rebuild AdaptiveCpp even if
#                                          # $ACPP_PREFIX already has an install
#                                          # (use after a driver / toolchain
#                                          # change, or to re-pin $ACPP_REF).
#
# Supported distros: Arch family, Ubuntu/Debian, Fedora/RHEL.
# For anything else, install the equivalents listed at the bottom and
# build AdaptiveCpp from source manually.

set -euo pipefail

ACPP_REF=${ACPP_REF:-v25.10.0}
ACPP_PREFIX=${ACPP_PREFIX:-/opt/adaptivecpp}
SKIP_ACPP=0
REBUILD_ACPP=0
GPU=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-acpp)      SKIP_ACPP=1; shift ;;
        --rebuild-acpp) REBUILD_ACPP=1; shift ;;
        --gpu)          GPU="$2"; shift 2 ;;
        -h|--help)      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
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

# ── Detect GPU vendor (NVIDIA / AMD / Intel) ────────────────────────────────
# Two-tier detection so a fresh OS install (no driver tools yet) still works:
#   1. Tool-based (nvidia-smi / rocminfo) — authoritative when available,
#      because it confirms the driver+runtime is functional, not just that
#      a card is plugged in.
#   2. PCI vendor ID via /sys/class/drm — works pre-driver. The whole point
#      of running install-deps.sh is to install the driver/toolkit, so we
#      can't require the driver tools as a prerequisite for detection.
#
# Precedence (when multiple GPUs are present): NVIDIA > AMD > Intel.
# Matches the build.rs vendor-precedence logic.
detect_gpu_via_pci() {
    local found="" entry name vendor
    for entry in /sys/class/drm/card*; do
        name=$(basename "$entry")
        # Skip connector entries like card0-DP-1 — only the bare cardN
        # nodes have a `device/vendor` attribute we care about.
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

if [[ -z "$GPU" ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        GPU=nvidia
        echo "[install-deps] Detected NVIDIA GPU (nvidia-smi)."
    elif command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q gfx; then
        GPU=amd
        echo "[install-deps] Detected AMD GPU (rocminfo)."
    else
        GPU=$(detect_gpu_via_pci)
        if [[ -n "$GPU" ]]; then
            echo "[install-deps] Detected $GPU GPU via /sys/class/drm (PCI vendor ID); driver tools not yet installed."
        fi
    fi
fi

if [[ -z "$GPU" ]]; then
    echo "[install-deps] Could not auto-detect a GPU (no nvidia-smi / rocminfo," >&2
    echo "[install-deps] no usable PCI device under /sys/class/drm)." >&2
    echo "[install-deps] Pass --gpu nvidia or --gpu amd explicitly to override." >&2
    echo "[install-deps] Headless / CI builds: --gpu nvidia installs the LLVM" >&2
    echo "[install-deps] toolchain + CUDA Toolkit headers used by the SYCL path." >&2
    exit 1
fi

if [[ "$GPU" == "intel" ]]; then
    echo "[install-deps] Intel GPU detected, but install-deps.sh has no Intel-" >&2
    echo "[install-deps] specific package path yet. Options:" >&2
    echo "[install-deps]   --gpu nvidia     install LLVM + CUDA headers (the SYCL" >&2
    echo "[install-deps]                    path JITs onto Intel via AdaptiveCpp's" >&2
    echo "[install-deps]                    generic SSCP target at runtime)" >&2
    echo "[install-deps]   ./scripts/build-container.sh   container with Intel oneAPI" >&2
    exit 1
fi
echo "[install-deps] distro=$DISTRO, gpu=$GPU, acpp=${ACPP_REF}, prefix=${ACPP_PREFIX}"

# ── Per-distro packages ─────────────────────────────────────────────────────
install_arch() {
    # `openmp` is clang's libomp runtime — required by AdaptiveCpp's
    # OpenMP backend find_package check, even on the NVIDIA path.
    local pkgs=(cmake git base-devel python ninja
                llvm clang lld
                boost numactl curl
                openmp)
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

    # On rolling Arch, `llvm`/`clang` are often >20 — above AdaptiveCpp's
    # 16-20 cap. Pull the official side-by-side `llvm20`/`clang20`/`lld20`
    # from `extra`; they install under /usr/lib/llvm20, the first path
    # the LLVM probe further down checks.
    local sys_llvm_major
    sys_llvm_major=$(pacman -Q llvm 2>/dev/null \
                     | awk '{print $2}' | grep -oE '^[0-9]+')
    if [[ -n "$sys_llvm_major" ]] && (( sys_llvm_major > 20 )); then
        echo "[install-deps] System llvm is $sys_llvm_major (> AdaptiveCpp's 20 cap)."
        echo "[install-deps] Installing side-by-side llvm20/clang20/lld20 from extra."
        sudo pacman -S --needed --noconfirm llvm20 llvm20-libs clang20 lld20
    fi
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
    if [[ $REBUILD_ACPP -eq 1 ]]; then
        echo "[install-deps] --rebuild-acpp: wiping existing $ACPP_PREFIX and rebuilding."
        # $ACPP_PREFIX usually lives under /opt or another root-owned tree, so
        # the wipe needs sudo. The build + install step further down already
        # uses sudo for cmake --install; reusing it here is consistent.
        sudo rm -rf "$ACPP_PREFIX"
    else
        echo "[install-deps] AdaptiveCpp already installed at $ACPP_PREFIX. Skipping."
        echo "[install-deps] Pass --rebuild-acpp to wipe + rebuild (e.g. after a driver change)."
        exit 0
    fi
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

# GPU-specific knobs for AdaptiveCpp's `find_package(CUDA)` probe.
#
# The probe drives WITH_CUDA_BACKEND (CMakeLists.txt:237 —
# `set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL ...)`), so its outcome
# decides whether `librt-backend-cuda.so` ships in the install. We
# deliberately don't force WITH_CUDA_BACKEND=ON: lines 224-228 turn that
# into `SEND_ERROR` when CUDA can't be found, which is worse than the
# silent off we'd get from an honest auto-detect miss. Just feed the
# probe a toolkit root when we know one — auto-detect handles the rest.
#
# Arch: `cuda` installs to /opt/cuda, which isn't on the default PATH
# and isn't a location FindCUDA scans on its own — so without a hint
# the backend silently turns off and `acpp-info` lists only OpenMP/OCL,
# even with two NVIDIA cards in the box. Override the default with
# ACPP_CUDA_TOOLKIT_ROOT=... if your toolkit lives elsewhere.
#
# AMD: same probe, opposite problem — on hosts where a CUDA *headers*
# subset is installed (distro cuda, JetPack fragments, /usr/lib from
# some wrappers), AdaptiveCpp's FindCUDA emits
# `CUDAToolkit_LIBRARY_ROOT /usr/lib does not point to the correct
# directory, try setting it manually`. AdaptiveCpp continues fine, but
# the warning looks like an error in the log. Disable the probe.
ACPP_CUDA_FLAGS=()
case "$GPU" in
    nvidia)
        ACPP_CUDA_TOOLKIT_ROOT=${ACPP_CUDA_TOOLKIT_ROOT:-/opt/cuda}
        if [[ -d "$ACPP_CUDA_TOOLKIT_ROOT" ]]; then
            ACPP_CUDA_FLAGS+=(
                -DCUDA_TOOLKIT_ROOT_DIR="$ACPP_CUDA_TOOLKIT_ROOT"
                -DCUDAToolkit_ROOT="$ACPP_CUDA_TOOLKIT_ROOT"
            )
        fi
        ;;
    amd)
        ACPP_CUDA_FLAGS+=(-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE)
        ;;
esac

cmake -S "$ACPP_BUILD_DIR/src" -B "$ACPP_BUILD_DIR/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ACPP_PREFIX" \
    -DCMAKE_C_COMPILER="$LLVM_ROOT/bin/clang" \
    -DCMAKE_CXX_COMPILER="$LLVM_ROOT/bin/clang++" \
    -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" \
    -DACPP_LLD_PATH="$LLVM_ROOT/bin/ld.lld" \
    "${ACPP_CUDA_FLAGS[@]}" \
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

# Tell users about the nvcc/host-compiler dance when the system gcc is
# newer than what the installed nvcc supports. nvcc 12.8's default ccbin
# (/usr/bin/cc → system gcc) chokes on gcc 15+'s libstdc++ <type_traits>.
# Two ccbins work around it: (a) the side-by-side gcc-14 we install when
# pinning cuda 12.8 from the Arch archive (cleanest — nvcc's canonical
# Linux host), or (b) the clang we picked for AdaptiveCpp's build (works
# for the simple compiler-id test, but on some cuda 12.x point releases
# crt/math_functions.hpp's rsqrt definition trips clang's exception-spec
# check). Print whichever is available so non-Arch hosts (where g++-14
# isn't auto-installed) still get a usable hint.
if [[ "$GPU" == "nvidia" ]] && command -v gcc >/dev/null; then
    sys_gcc_major=$(gcc -dumpversion 2>/dev/null | grep -oE '^[0-9]+')
    if [[ -n "$sys_gcc_major" ]] && (( sys_gcc_major > 14 )); then
        echo
        echo "[install-deps] Note: system gcc is $sys_gcc_major; nvcc 12.8's default host"
        echo "[install-deps] compiler can't parse its libstdc++ headers. Use one of these"
        echo "[install-deps] as nvcc's ccbin via CMAKE_CUDA_HOST_COMPILER:"
        if command -v g++-14 >/dev/null; then
            echo "    -DCMAKE_CUDA_HOST_COMPILER=$(command -v g++-14)   # preferred — nvcc's canonical Linux host"
        fi
        if [[ -n "${LLVM_ROOT:-}" ]] && [[ -x "$LLVM_ROOT/bin/clang++" ]]; then
            echo "    -DCMAKE_CUDA_HOST_COMPILER=$LLVM_ROOT/bin/clang++   # works on most cuda 12.x point releases"
        fi
    fi
fi
