#!/usr/bin/env bash
#
# install-deps.sh — bootstrap xchplot2's native build dependencies.
#
# Installs CUDA Toolkit on NVIDIA, ROCm HIP SDK on AMD, the newest LLVM
# AdaptiveCpp supports (16-20 — installed side-by-side when the system one is
# newer), AdaptiveCpp 25.10, and a Rust toolchain via rustup. After this
# completes,
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
# AdaptiveCpp 25.10's CMake hard-errors outside this range ("LLVM versions
# greater than 20 are not yet tested/supported"). Single source of truth for
# both the probe and the packages we install — bump together with $ACPP_REF.
LLVM_MIN=16
LLVM_MAX=20
# Normalized package family (arch / debian / fedora), set by the distro
# dispatch below so the LLVM helpers don't re-derive it from $DISTRO.
PKG_FAMILY=""

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
            0x1002) found="amd" ;;                      # outranks intel
            0x8086) [[ -z "$found" ]] && found="intel" ;; # only if nothing else
        esac
    done
    echo "$found"
}

# Whether ANY Intel GPU is present, independent of which vendor won the
# primary $GPU slot above. That slot picks one toolkit (CUDA or ROCm), but a
# host can have GPUs from two vendors — an AMD APU alongside a discrete Arc is
# an ordinary desktop, and the precedence above hands the slot to AMD. Level
# Zero is not part of either toolkit, so without a separate signal AdaptiveCpp
# gets built with no Level Zero backend and the Intel card is invisible at
# runtime no matter how the plotter is invoked.
intel_gpu_present() {
    local entry name vendor
    for entry in /sys/class/drm/card*; do
        name=$(basename "$entry")
        [[ "$name" =~ ^card[0-9]+$ ]] || continue
        [[ -r "$entry/device/vendor" ]] || continue
        vendor=$(cat "$entry/device/vendor" 2>/dev/null)
        [[ "$vendor" == "0x8086" ]] && return 0
    done
    return 1
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

# Level Zero is installed whenever an Intel GPU is present, not only when it
# won the $GPU slot. AdaptiveCpp enables the backends it can find at ITS
# configure time, so the loader + headers have to be on disk before the build
# below or the Intel card can never be driven — and on a multi-vendor host the
# slot belongs to AMD or NVIDIA, so keying off it alone silently skips this.
WANT_INTEL=""
if [[ "$GPU" == "intel" ]] || intel_gpu_present; then
    WANT_INTEL=1
    if [[ "$GPU" != "intel" ]]; then
        echo "[install-deps] Intel GPU also present — adding the Level Zero stack"
        echo "[install-deps] so AdaptiveCpp builds its Level Zero backend too."
    fi
fi
echo "[install-deps] distro=$DISTRO, gpu=$GPU, acpp=${ACPP_REF}, prefix=${ACPP_PREFIX}"

# ── LLVM discovery + install ────────────────────────────────────────────────
# AdaptiveCpp needs an LLVM in [$LLVM_MIN, $LLVM_MAX] with clang, ld.lld AND
# the CMake package files. Rolling distros ship newer (Fedora 43 = LLVM 21,
# Arch = 22), so the usable one is often a side-by-side compat package rather
# than the system default. All three families package those; picking and
# installing the newest is our job, not the user's.

# Echo the CMake package dir for an LLVM prefix, or return 1.
# llvm-config knows the real answer: Fedora's compat packages report
# /usr/lib64/llvm20/lib64/cmake/llvm — a lib64 INSIDE the prefix — so
# assuming $root/lib/cmake/llvm silently points at a nonexistent dir.
llvm_cmake_dir() {
    local root=$1 d
    d=$("$root/bin/llvm-config" --cmakedir 2>/dev/null || true)
    if [[ -n "$d" ]] && [[ -d "$d" ]]; then
        printf '%s\n' "$d"; return 0
    fi
    for d in "$root/lib/cmake/llvm" "$root/lib64/cmake/llvm"; do
        [[ -d "$d" ]] && { printf '%s\n' "$d"; return 0; }
    done
    return 1
}

# Echo an LLVM prefix's major version if it is complete enough to build
# AdaptiveCpp against, else return 1. Requiring the CMake dir here (not just
# the binaries) rejects a clang+lld install whose -devel half is missing,
# which would otherwise be picked and then fail mid-configure.
llvm_prefix_version() {
    local root=$1 ver
    [[ -x "$root/bin/clang" ]] && [[ -x "$root/bin/ld.lld" ]] || return 1
    ver=$("$root/bin/clang" --version 2>/dev/null \
          | head -1 | grep -oE 'version [0-9]+' | grep -oE '[0-9]+')
    [[ -n "$ver" ]] || return 1
    (( ver >= LLVM_MIN && ver <= LLVM_MAX )) || return 1
    llvm_cmake_dir "$root" >/dev/null || return 1
    printf '%s\n' "$ver"
}

# Echo the prefix of the NEWEST compatible LLVM on the box, or return 1.
# Scores every candidate rather than taking the first hit off an ordered
# list, so the answer is "latest compatible" no matter which prefixes exist.
# /usr and /usr/local are included because on a distro whose system LLVM is
# already in range (Fedora 42 = 20, Ubuntu 24.04 = 18) the unversioned
# install is the right answer and installing a compat package is pure waste.
find_llvm_root() {
    local best_ver=0 best_root="" cand ver v
    local -a cands=()
    # Generated from the supported range rather than spelled out, so bumping
    # $LLVM_MAX for a newer AdaptiveCpp doesn't leave the probe behind.
    for v in $(seq "$LLVM_MAX" -1 "$LLVM_MIN"); do
        cands+=("/usr/lib/llvm-$v"   "/usr/lib/llvm$v"
                "/usr/lib64/llvm-$v" "/usr/lib64/llvm$v"
                "/opt/llvm-$v"       "/opt/llvm$v")
    done
    # Unversioned prefixes last: a distro whose default LLVM is in range is a
    # perfectly good answer, but on a tie an explicit versioned prefix is the
    # more stable one — the default moves at the next upgrade.
    cands+=(/usr /usr/local)
    for cand in "${cands[@]}"; do
        ver=$(llvm_prefix_version "$cand") || continue
        if (( ver > best_ver )); then
            best_ver=$ver
            best_root=$cand
        fi
    done
    [[ -n "$best_root" ]] || return 1
    printf '%s\n' "$best_root"
}

# The package set for a given LLVM major version, or for the distro's
# unversioned default when $1 is the literal `system`. Echoed space-separated
# for the caller to splat. Every name is derived from the version — there are
# no hardcoded LLVM versions anywhere in this script.
llvm_pkgs_for() {
    local v=$1
    case "$PKG_FAMILY" in
        arch)
            if [[ "$v" == system ]]; then
                echo "llvm clang lld"
            else
                echo "llvm$v llvm$v-libs clang$v lld$v"
            fi ;;
        debian)
            if [[ "$v" == system ]]; then
                echo "llvm llvm-dev clang lld libclang-dev libomp-dev"
            else
                echo "llvm-$v llvm-$v-dev clang-$v lld-$v libclang-$v-dev" \
                     "libclang-cpp$v-dev libclang-rt-$v-dev libomp-$v-dev"
            fi ;;
        fedora)
            if [[ "$v" == system ]]; then
                echo "llvm llvm-devel clang clang-devel lld libomp-devel"
            else
                # libomp$v-devel is only a weak dep of clang$v; name it so the
                # OpenMP backend still builds under --setopt=install_weak_deps=0.
                echo "llvm$v llvm$v-devel clang$v clang$v-devel lld$v libomp$v-devel"
            fi ;;
    esac
}

# Can a versioned LLVM $1 be installed from the enabled repos?
llvm_version_installable() {
    local v=$1
    case "$PKG_FAMILY" in
        arch)   pacman -Si "llvm$v" &>/dev/null ;;
        # policy, not `apt-cache show`: a package can be known to apt yet have
        # "Candidate: (none)" when no enabled suite carries it, and installing
        # that fails.
        debian) [[ "$(apt-cache policy "llvm-$v-dev" 2>/dev/null \
                      | awk '/Candidate:/{print $2}')" == [0-9]* ]] ;;
        # plain `list`, not `list --available`: --available hides a package
        # that is already installed, which would make us skip a version the
        # box has half of and silently fall back to an older one.
        fedora) dnf -q list "llvm$v-devel" &>/dev/null ;;
        *)      return 1 ;;
    esac
}

# Major version of the distro's UNVERSIONED llvm, as the repos would install
# it. Asking the repo rather than the installed binary lets us decide whether
# the platform default is usable BEFORE pulling ~1 GB of it down.
llvm_system_major() {
    local ver=""
    case "$PKG_FAMILY" in
        arch)   ver=$(pacman -Si llvm 2>/dev/null | awk -F': *' '/^Version/{print $2; exit}') ;;
        debian) ver=$(apt-cache policy llvm-dev 2>/dev/null | awk '/Candidate:/{print $2}') ;;
        fedora) ver=$(dnf -q info llvm 2>/dev/null | awk -F': *' '/^Version/{print $2; exit}') ;;
    esac
    ver=${ver#*:}                      # drop a Debian epoch ("1:18.0-59")
    ver=$(printf '%s' "$ver" | grep -oE '^[0-9]+') || return 1
    [[ -n "$ver" ]] || return 1
    printf '%s\n' "$ver"
}

# The newest LLVM this platform can provide that AdaptiveCpp supports.
# Prefers an explicitly versioned set; falls back to the unversioned default
# when that is itself in range, because a distro shipping a supported LLVM as
# its default (Fedora 42 ships 20) generally has no compat package to install
# instead. Echoes a major version or the literal `system`.
pick_llvm_target() {
    local v sys
    for v in $(seq "$LLVM_MAX" -1 "$LLVM_MIN"); do
        if llvm_version_installable "$v"; then
            printf '%s\n' "$v"
            return 0
        fi
    done
    if sys=$(llvm_system_major) && (( sys >= LLVM_MIN && sys <= LLVM_MAX )); then
        printf 'system\n'
        return 0
    fi
    return 1
}

# Install that choice. Separate from the per-distro base package sets so the
# platform's default LLVM is never dragged in just to be rejected: on a
# rolling release it is past the cap, and installing it costs ~1 GB for
# nothing.
install_llvm_target() {
    local target pkgs
    target=$(pick_llvm_target) || return 1
    pkgs=$(llvm_pkgs_for "$target")
    if [[ "$target" == system ]]; then
        echo "[install-deps] LLVM: distro default is in range (${LLVM_MIN}-${LLVM_MAX}); installing $pkgs"
    else
        echo "[install-deps] LLVM: $target is the newest this platform offers that AdaptiveCpp ${ACPP_REF} supports."
    fi
    # Word-splitting $pkgs is intentional — it is a package list.
    # shellcheck disable=SC2086
    case "$PKG_FAMILY" in
        arch)   sudo pacman -S --needed --noconfirm $pkgs ;;
        debian) sudo apt-get install -y --no-install-recommends $pkgs ;;
        fedora) sudo dnf install -y $pkgs ;;
        *)      return 1 ;;
    esac
}

# ── Per-distro packages ─────────────────────────────────────────────────────
install_arch() {
    # `openmp` is clang's libomp runtime — required by AdaptiveCpp's
    # OpenMP backend find_package check, even on the NVIDIA path.
    # No llvm/clang/lld here — install_llvm_target picks the version and
    # installs it, so we don't drag in a system LLVM that is past the cap.
    local pkgs=(cmake git base-devel python ninja
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
    # Appended, not an arm of the case: see WANT_INTEL.
    [[ -n "$WANT_INTEL" ]] && pkgs+=(level-zero-headers level-zero-loader
                                     intel-compute-runtime)
    sudo pacman -S --needed --noconfirm "${pkgs[@]}"

    # The side-by-side compat LLVM that rolling Arch needs (system llvm is
    # 22) is handled generically after the dispatch — every family has the
    # same problem, so it is no longer special-cased here.
}

install_apt() {
    # LLVM (including libclang-rt-*-dev, the compiler-rt builtins
    # AdaptiveCpp's HIP/ROCm backend link needs) comes from
    # install_llvm_target. This list used to hardcode 18, which left newer
    # releases on an old toolchain even where they package 19 or 20.
    local pkgs=(cmake git ninja-build build-essential python3 pkg-config
                libboost-context-dev libnuma-dev curl ca-certificates)
    case "$GPU" in
        nvidia)
            # Detect a pre-existing /usr/local/cuda-X.Y install (RunPod /
            # NGC / NVIDIA-supplied container images all ship CUDA at
            # /usr/local/cuda-X.Y from the official `cuda-toolkit-X-Y`
            # package or NVIDIA's runfile installer). Ubuntu's apt
            # `nvidia-cuda-toolkit` is a separate (often older) packaging
            # that drops nvcc at /usr/bin/nvcc; installing it on top of
            # an already-present /usr/local/cuda-X.Y shadows the newer
            # toolkit and triggers nvcc-vs-host-compiler incompatibilities
            # (e.g. CUDA 12.0's cudafe++ choking on glibc's _Float32).
            # Skip the apt install when a newer /usr/local/cuda-X.Y is
            # already on the box; the existing install plus the
            # /etc/profile.d/cuda.sh below covers our needs.
            local existing_cuda=""
            for d in /usr/local/cuda /usr/local/cuda-*; do
                if [[ -x "$d/bin/nvcc" ]]; then
                    existing_cuda="$d"
                    break
                fi
            done
            if [[ -n "$existing_cuda" ]]; then
                echo "[install-deps] Found existing CUDA at $existing_cuda — skipping apt nvidia-cuda-toolkit"
            else
                pkgs+=(nvidia-cuda-toolkit)
            fi
            ;;
        amd)    pkgs+=(rocm-hip-sdk rocm-libs rocminfo)
                # rocminfo is the discovery tool build-container.sh probes;
                # not pulled in transitively by rocm-hip-sdk.
                # No nvidia-cuda-toolkit-headers on the AMD path —
                # CudaHalfShim.hpp guards the CUDA headers via
                # __has_include, and pulling CUDA alongside HIP causes
                # uchar1/char1 typedef redefinitions.
                ;;
    esac
    # Appended, not an arm of the case: see WANT_INTEL.
    [[ -n "$WANT_INTEL" ]] && pkgs+=(libze-dev libze1 libze-intel-gpu1
                                     intel-opencl-icd)
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${pkgs[@]}"
}

install_dnf() {
    # No llvm/clang/lld/libomp here — install_llvm_target picks the version.
    local pkgs=(cmake git ninja-build gcc-c++ python3 pkg-config
                boost-devel numactl-devel curl)
    case "$GPU" in
        nvidia) pkgs+=(cuda-toolkit) ;;
        # No cuda-toolkit on the AMD path — CudaHalfShim.hpp guards the
        # CUDA headers via __has_include, and pulling CUDA alongside HIP
        # causes uchar1/char1 typedef redefinitions.
        amd)    pkgs+=(rocm-hip-devel rocminfo) ;;
    esac
    # Appended, not an arm of the case: see WANT_INTEL.
    [[ -n "$WANT_INTEL" ]] && pkgs+=(oneapi-level-zero oneapi-level-zero-devel
                                     intel-compute-runtime)
    sudo dnf install -y "${pkgs[@]}"
}

case "$DISTRO" in
    arch|cachyos|manjaro|endeavouros)            PKG_FAMILY=arch;   install_arch ;;
    ubuntu|debian|pop|linuxmint)                 PKG_FAMILY=debian; install_apt  ;;
    fedora|rhel|centos|rocky|almalinux)          PKG_FAMILY=fedora; install_dnf  ;;
    *)
        case "$DISTRO_LIKE" in
            *arch*)   PKG_FAMILY=arch;   install_arch ;;
            *debian*) PKG_FAMILY=debian; install_apt  ;;
            *rhel*|*fedora*) PKG_FAMILY=fedora; install_dnf ;;
            *)
                echo "[install-deps] Unknown distro '$DISTRO'. Install equivalents of:"
                echo "  CMake ≥ 3.24, Ninja, LLVM ${LLVM_MIN}-${LLVM_MAX} + clang + ld.lld, libclang dev,"
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

# ── Ensure an AdaptiveCpp-compatible LLVM ───────────────────────────────────
# Runs in the package phase, not lazily at the AdaptiveCpp build below, so
# every privileged install happens in one burst instead of prompting for sudo
# again ten minutes later. This is the common case rather than an edge one:
# the base package sets above pull each distro's *system* LLVM, which on a
# rolling release is past AdaptiveCpp's cap (Fedora 43 ships 21, Arch 22).
if llvm_have=$(find_llvm_root); then
    echo "[install-deps] Compatible LLVM already present: $llvm_have (LLVM $(llvm_prefix_version "$llvm_have"))"
else
    # A failure here is not fatal yet: the probe before the AdaptiveCpp
    # build re-checks and prints the full manual-install guidance.
    install_llvm_target \
        || echo "[install-deps] Automatic LLVM install failed; see the diagnosis below." >&2
fi

# ── Put nvcc on PATH ────────────────────────────────────────────────────────
# Distro packages disagree on where nvcc lands and whether it is on PATH:
#   Arch     /opt/cuda/bin       — reaches PATH only via /etc/profile.d/cuda.sh,
#                                  which calls append_path(), a helper defined in
#                                  /etc/profile. That makes it a *login*-shell
#                                  hook: the shell running this script never
#                                  sees it.
#   deb/NGC  /usr/local/cuda/bin — not on PATH by default at all.
# Either way the shell the user is sitting in cannot see nvcc when we finish, so
# the `cargo install --path .` we print below would fail its nvcc preflight on a
# box that just successfully installed the toolkit. Export it for the rest of
# this script, and make it stick for future shells.
CUDA_ROOT=""
if [[ "$GPU" == "nvidia" ]]; then
    for cand in "${ACPP_CUDA_TOOLKIT_ROOT:-}" /opt/cuda /usr/local/cuda /usr/local/cuda-*; do
        [[ -n "$cand" ]] || continue
        if [[ -x "$cand/bin/nvcc" ]]; then
            CUDA_ROOT="$cand"
            break
        fi
    done
    if [[ -n "$CUDA_ROOT" ]]; then
        export CUDA_PATH="$CUDA_ROOT"
        export PATH="$CUDA_ROOT/bin:$PATH"
        # `|| true`: set -o pipefail would abort the script if a broken nvcc
        # exits nonzero here, and a missing version string is not fatal.
        nvcc_release=$("$CUDA_ROOT/bin/nvcc" --version 2>/dev/null \
                       | sed -n 's/.*release \([0-9.]*\).*/\1/p' || true)
        echo "[install-deps] nvcc ${nvcc_release:-?} at $CUDA_ROOT/bin/nvcc"
        # Only write our own snippet when the distro didn't ship one — Arch's
        # cuda package already provides /etc/profile.d/cuda.sh (and sets
        # NVCC_CCBIN there); clobbering it would be rude and lossy.
        if [[ ! -f /etc/profile.d/cuda.sh ]] && [[ ! -f /etc/profile.d/xchplot2-cuda.sh ]]; then
            echo "[install-deps] Writing /etc/profile.d/xchplot2-cuda.sh (PATH front-load $CUDA_ROOT/bin)"
            # Single quotes on $PATH are intentional: the *file* must contain a
            # literal $PATH that expands per-shell, not this script's PATH baked
            # in at install time. shellcheck flags that as SC2016.
            # shellcheck disable=SC2016
            printf 'export CUDA_PATH=%s\nexport PATH=%s/bin:$PATH\n' "$CUDA_ROOT" "$CUDA_ROOT" \
                | sudo tee /etc/profile.d/xchplot2-cuda.sh >/dev/null
            sudo chmod +x /etc/profile.d/xchplot2-cuda.sh
        fi
    else
        echo "[install-deps] WARNING: packages installed but no nvcc under /opt/cuda" >&2
        echo "[install-deps] or /usr/local/cuda* — the CUDA build will not work." >&2
    fi
fi

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
# The package phase above should already have installed one. Re-probe here
# because --no-acpp reruns, $ACPP_PREFIX rebuilds and manual installs all
# reach this point without having gone through it, and because it is the
# honest check: what matters is what is on disk now, not what we tried.
if ! LLVM_ROOT=$(find_llvm_root); then
    echo "[install-deps] No compatible LLVM (${LLVM_MIN}-${LLVM_MAX}) with ld.lld and" >&2
    echo "[install-deps] CMake package files found, and none could be installed" >&2
    echo "[install-deps] automatically. AdaptiveCpp $ACPP_REF rejects anything newer." >&2
    echo "[install-deps] Install one by hand and re-run, or use the container path:" >&2
    # Package names come from llvm_pkgs_for, the same function the automatic
    # path uses — so this hint can never drift from what we'd have installed.
    case "$PKG_FAMILY" in
        arch)   _llvm_hint_cmd="sudo pacman -S" ;;
        debian) _llvm_hint_cmd="sudo apt install" ;;
        fedora) _llvm_hint_cmd="sudo dnf install" ;;
        *)      _llvm_hint_cmd="" ;;
    esac
    if [[ -n "$_llvm_hint_cmd" ]]; then
        echo "  $_llvm_hint_cmd $(llvm_pkgs_for "$LLVM_MAX")" >&2
        echo "  # or the highest version ≤ ${LLVM_MAX} your release actually carries" >&2
    else
        echo "  install LLVM ${LLVM_MIN}-${LLVM_MAX} + clang + ld.lld for your distro" >&2
    fi
    echo "  ./scripts/build-container.sh   # container has a supported LLVM pinned" >&2
    exit 1
fi
# Ask llvm-config rather than assuming $LLVM_ROOT/lib/cmake/llvm: Fedora's
# compat packages answer /usr/lib64/llvm20/lib64/cmake/llvm.
LLVM_CMAKE_DIR=$(llvm_cmake_dir "$LLVM_ROOT")
echo "[install-deps] Using LLVM $(llvm_prefix_version "$LLVM_ROOT") at $LLVM_ROOT for AdaptiveCpp build."

# AdaptiveCpp hunts for clang's resource dir only under
# ${LLVM_PREFIX_DIR}/{include,lib,lib64}/clang/<major>/include and SEND_ERRORs
# ("CLANG_INCLUDE_PATH does not exist") when every hint misses. Fedora's
# compat clang answers /usr/lib/clang/20 — outside the LLVM prefix entirely —
# so it always misses there. Ask clang: -print-resource-dir names the
# directory holding include/, exactly what AdaptiveCpp computes as
# FOUND_CLANG_INCLUDE_PATH/.. on the happy path.
ACPP_CLANG_FLAGS=()
CLANG_RES=$("$LLVM_ROOT/bin/clang" -print-resource-dir 2>/dev/null || true)
# -print-resource-dir answers relative to bin/ with ../.. hops.
[[ -n "$CLANG_RES" ]] && CLANG_RES=$(readlink -f "$CLANG_RES" 2>/dev/null || echo "$CLANG_RES")
if [[ -n "$CLANG_RES" ]] && [[ -f "$CLANG_RES/include/__clang_cuda_runtime_wrapper.h" ]]; then
    ACPP_CLANG_FLAGS+=(-DCLANG_INCLUDE_PATH="$CLANG_RES")
    echo "[install-deps] clang resource dir: $CLANG_RES"
fi
# Pin Clang_DIR alongside LLVM_DIR: leaving it unset lets AdaptiveCpp's
# find_package(Clang) resolve to the system clang's config and re-introduce
# the very version skew LLVM_DIR just removed.
for _cl in "$LLVM_ROOT/lib64/cmake/clang" "$LLVM_ROOT/lib/cmake/clang"; do
    if [[ -f "$_cl/ClangConfig.cmake" ]]; then
        ACPP_CLANG_FLAGS+=(-DClang_DIR="$_cl")
        break
    fi
done

# ── ROCm device libs path (AMD only) ────────────────────────────────────────
# AdaptiveCpp's HIP backend needs ockl.bc / ocml.bc to compile kernels for
# amdgcn; without a match the build dies with "ROCm device library path not
# found".
#
# This used to probe only /opt/rocm, AMD's own installer layout. Distros that
# package ROCm themselves put it somewhere else entirely — Fedora's
# rocm-device-libs ships
# /usr/lib64/rocm/llvm/lib/clang/<rocm-llvm-major>/lib/amdgcn/bitcode and has
# no /opt/rocm at all — so every distro-ROCm host failed here. Glob the known
# layouts and take the highest version when several are installed.
# Level Zero has to be switched on by hand. WITH_CUDA_BACKEND and
# WITH_OPENCL_BACKEND get CACHE defaults derived from whether their toolkit was
# found, so installing the packages is enough for those; WITH_LEVEL_ZERO_BACKEND
# has no default declaration anywhere in AdaptiveCpp 25.10, so it evaluates
# false unless passed on the command line. Installing the loader alone therefore
# produced a build with only librt-backend-omp.so, and an Intel GPU stayed
# invisible with the packages sitting right there on disk. The backend links
# -lze_loader directly, so the loader package is all it needs besides the flag.
ACPP_INTEL_FLAGS=()
if [[ -n "$WANT_INTEL" ]]; then
    ACPP_INTEL_FLAGS+=(-DWITH_LEVEL_ZERO_BACKEND=ON)
    echo "[install-deps] Enabling AdaptiveCpp's Level Zero backend (Intel GPU)."
fi

ACPP_ROCM_FLAGS=()
if [[ "$GPU" == "amd" ]]; then
    rocm_bc=""
    while IFS= read -r d; do
        rocm_bc="$d"
    done < <(
        for d in \
            /opt/rocm*/amdgcn/bitcode \
            /opt/rocm*/lib/llvm-amdgpu/amdgcn/bitcode \
            /opt/rocm*/share/amdgcn/bitcode \
            /usr/lib64/rocm/llvm/lib/clang/*/lib/amdgcn/bitcode \
            /usr/lib/rocm/llvm/lib/clang/*/lib/amdgcn/bitcode \
            /usr/lib64/amdgcn/bitcode \
            /usr/lib/amdgcn/bitcode \
            /usr/share/amdgcn/bitcode; do
            [[ -f "$d/ockl.bc" ]] && printf '%s\n' "$d"
        done | sort -V   # -V so clang/9 sorts below clang/22, unlike plain sort
    )
    if [[ -n "$rocm_bc" ]]; then
        ACPP_ROCM_FLAGS+=(-DROCM_DEVICE_LIBS_PATH="$rocm_bc")
        echo "[install-deps] ROCm device libs: $rocm_bc"
    else
        echo "[install-deps] WARNING: no amdgcn bitcode (ockl.bc) found — AdaptiveCpp's" >&2
        echo "[install-deps] HIP backend will fail to configure. Install your distro's" >&2
        echo "[install-deps] rocm-device-libs package, or pass the directory yourself:" >&2
        echo "[install-deps]   ROCM_DEVICE_LIBS_PATH=/path/to/amdgcn/bitcode" >&2
    fi
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
        # Probe candidate CUDA toolkit roots in order. Without an
        # explicit hint here, AdaptiveCpp's cmake silently builds
        # without the CUDA backend on hosts where CUDA isn't at the
        # standard cmake-default path (e.g. Ubuntu RunPod / NGC images
        # where CUDA lives under /usr/local/cuda-X.Y rather than
        # /opt/cuda). Caller can still override via ACPP_CUDA_TOOLKIT_
        # ROOT for non-standard installs.
        # The PATH step above already resolved a toolkit root — reuse it so
        # AdaptiveCpp and the xchplot2 build agree on which nvcc they target.
        if [[ -z "${ACPP_CUDA_TOOLKIT_ROOT:-}" ]] && [[ -n "${CUDA_ROOT:-}" ]]; then
            ACPP_CUDA_TOOLKIT_ROOT="$CUDA_ROOT"
        fi
        if [[ -z "${ACPP_CUDA_TOOLKIT_ROOT:-}" ]]; then
            for cand in /opt/cuda /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-13 /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
                if [[ -x "$cand/bin/nvcc" ]]; then
                    ACPP_CUDA_TOOLKIT_ROOT="$cand"
                    break
                fi
            done
        fi
        if [[ -n "${ACPP_CUDA_TOOLKIT_ROOT:-}" ]] && [[ -d "$ACPP_CUDA_TOOLKIT_ROOT" ]]; then
            echo "[install-deps] AdaptiveCpp CUDA backend → $ACPP_CUDA_TOOLKIT_ROOT"
            ACPP_CUDA_FLAGS+=(
                -DCUDA_TOOLKIT_ROOT_DIR="$ACPP_CUDA_TOOLKIT_ROOT"
                -DCUDAToolkit_ROOT="$ACPP_CUDA_TOOLKIT_ROOT"
            )
        else
            echo "[install-deps] WARNING: no CUDA toolkit found at /opt/cuda or /usr/local/cuda* — AdaptiveCpp will build without the CUDA backend"
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
    -DLLVM_DIR="$LLVM_CMAKE_DIR" \
    -DACPP_LLD_PATH="$LLVM_ROOT/bin/ld.lld" \
    "${ACPP_CLANG_FLAGS[@]}" \
    "${ACPP_CUDA_FLAGS[@]}" \
    "${ACPP_ROCM_FLAGS[@]}" \
    "${ACPP_INTEL_FLAGS[@]}"
cmake --build "$ACPP_BUILD_DIR/build" --parallel
sudo cmake --install "$ACPP_BUILD_DIR/build"

# AdaptiveCpp's SSCP path doesn't emit SPIR-V itself — it shells out to a
# version-matched llvm-spirv that it builds as an ExternalProject (branch
# llvm_release_<major>0, so our pinned LLVM decides which) and is supposed to
# install via an install(CODE) hook. The translator compiles, but the hook does
# not place it: `cmake --install` reports success and nothing warns, leaving
# lib/hipSYCL/ext/llvm-spirv absent. Its location is baked into the runtime at
# build time (HIPSYCL_RELATIVE_LLVMSPIRV_PATH) and consulted with no PATH
# fallback and no env override, so the gap only surfaces much later, on the
# first kernel submitted to an Intel device:
#   LLVMToSpirv: llvm-spirv invocation failed with exit code -1
#   ze_queue: Code object construction failed
# and then as kernels that complete without writing anything. Drive the install
# target explicitly from the top-level build dir, where it resolves.
if [[ -n "$WANT_INTEL" ]]; then
    LLVM_SPIRV_BIN="$ACPP_PREFIX/lib/hipSYCL/ext/llvm-spirv/bin/llvm-spirv"
    sudo cmake --build "$ACPP_BUILD_DIR/build" --target InstallLLVMSpirvTranslator || true
    if [[ -x "$LLVM_SPIRV_BIN" ]]; then
        echo "[install-deps] llvm-spirv installed for the SPIR-V JIT: $LLVM_SPIRV_BIN"
    else
        echo "[install-deps] WARNING: $LLVM_SPIRV_BIN is missing." >&2
        echo "[install-deps] Intel devices will enumerate but every kernel will fail" >&2
        echo "[install-deps] with \"llvm-spirv invocation failed\". Build it by hand:" >&2
        echo "[install-deps]   cmake --build $ACPP_BUILD_DIR/build --target InstallLLVMSpirvTranslator" >&2
    fi
fi

echo
echo "[install-deps] Done."
echo "  AdaptiveCpp: $ACPP_PREFIX"
echo "  Build xchplot2:"
echo "    export CMAKE_PREFIX_PATH=$ACPP_PREFIX:\$CMAKE_PREFIX_PATH"
if [[ -n "${CUDA_ROOT:-}" ]]; then
    echo "    export PATH=$CUDA_ROOT/bin:\$PATH        # nvcc (new login shells get this on their own)"
fi
echo "    cargo install --path .                  # or:"
echo "    cmake -B build -S . && cmake --build build -j"

# Warn about the nvcc/host-compiler mismatch ONLY when it actually bites.
# nvcc pins a maximum supported GCC in crt/host_config.h and refuses to
# compile against anything newer — but the ceiling moves every point release
# (CUDA 12.8 → gcc 14, 13.x → gcc 15+), so a hardcoded "your gcc is too new"
# note is a false alarm as often as not. Probe instead: compile a trivial .cu
# with nvcc's default ccbin and only speak up if it genuinely fails. Note the
# probe deliberately runs with whatever NVCC_CCBIN this shell has, which is
# exactly what the real build will see.
if [[ "$GPU" == "nvidia" ]] && [[ -n "${CUDA_ROOT:-}" ]]; then
    # Cleaned up inline, NOT via `trap ... EXIT` — an EXIT trap here would
    # silently replace the one guarding $ACPP_BUILD_DIR above and leak a
    # multi-GB AdaptiveCpp build tree.
    probe_dir=$(mktemp -d -t xchplot2-ccbin-XXXXXX)
    printf '#include <type_traits>\n__global__ void k(){}\nint main(){return 0;}\n' \
        > "$probe_dir/probe.cu"
    if ! "$CUDA_ROOT/bin/nvcc" -c "$probe_dir/probe.cu" -o "$probe_dir/probe.o" \
         >/dev/null 2>&1; then
        sys_gcc_major=$(gcc -dumpversion 2>/dev/null | grep -oE '^[0-9]+' || true)
        echo
        echo "[install-deps] Note: nvcc ${nvcc_release:-?} rejects its default host compiler"
        echo "[install-deps] (system gcc ${sys_gcc_major:-?} is newer than this toolkit supports)."
        echo "[install-deps] Pass an older ccbin — first of these that works:"
        for ccbin in /usr/bin/g++-15 /usr/bin/g++-14 /usr/bin/g++-13 \
                     "${LLVM_ROOT:-/nonexistent}/bin/clang++"; do
            [[ -x "$ccbin" ]] || continue
            if "$CUDA_ROOT/bin/nvcc" -ccbin "$ccbin" -c "$probe_dir/probe.cu" \
               -o "$probe_dir/probe.o" >/dev/null 2>&1; then
                echo "    export NVCC_CCBIN=$ccbin                 # picked up by cargo + cmake"
                echo "    # or, for a direct cmake build:  -DCMAKE_CUDA_HOST_COMPILER=$ccbin"
                break
            fi
        done
    fi
    rm -rf "$probe_dir"
fi
