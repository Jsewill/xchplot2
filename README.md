# xchplot2

GPU plotter for Chia v2 proofs of space (CHIP-48). Produces farmable
`.plot2` files byte-identical to the
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

> **Status — work in progress.** Plots are byte-identical to the
> pos2-chip CPU reference and deterministic across runs; performance,
> AMD/Intel support, and the install/CI story are still evolving. Use
> [`cuda-only`](https://github.com/Jsewill/xchplot2/tree/cuda-only) for
> the most-tested path.

> **Branches:** `main` — SYCL/AdaptiveCpp port, runs on NVIDIA +
> AMD + Intel (CUB fast path preserved on NVIDIA).
> [`cuda-only`](https://github.com/Jsewill/xchplot2/tree/cuda-only) —
> original pure-CUDA path, pick it if you only target NVIDIA. See
> [Performance](#performance) for the tradeoff.

## Quick start

```bash
# Install — needs CUDA Toolkit 12+ (or AdaptiveCpp for AMD/Intel),
# CMake ≥ 3.24, a C++20 compiler, and Rust. See Build for alternatives.
cargo install --git https://github.com/Jsewill/xchplot2

# Plot — 10 × k=28 files, keys derived internally from your BLS pair.
xchplot2 plot -k 28 -n 10 \
    -f <farmer-pk-hex> \
    -c <pool-contract-xch1-or-txch1> \
    -o /mnt/plots

# Multi-GPU — one worker per device, round-robin partition.
xchplot2 plot ... --devices all
```

See [Hardware compatibility](#hardware-compatibility) for GPU / VRAM
/ OS requirements, [Build](#build) for container / native / CMake
paths, and [Use](#use) for every flag.
**Windows users**: this `cargo install` line works under WSL2; for
native Windows or a non-WSL setup, jump to [Windows](#windows).

## Hardware compatibility

- **GPU:**
  - **NVIDIA**, compute capability ≥ 5.0 (Maxwell / GTX 750-class
    and newer) via the CUDA fast path. Builds auto-detect the
    installed GPU's `compute_cap` via `nvidia-smi`; override with
    `$CUDA_ARCHITECTURES` for fat or cross-target builds (see
    [Build](#build)). Pre-sm_53 cards lack native FP16 ALUs, but
    `cuda_fp16.h` falls back to fp32 emulation for the half-precision
    intrinsics — kernels work correctly with the emulation cost.
    On dual-vendor hosts (e.g. AMD primary + secondary NVIDIA),
    `build.rs` also routes around CUDA 13.x + sm < 75 (the toolkit
    dropped Maxwell-Volta codegen) so an old NVIDIA card next to a
    working AMD GPU no longer derails the build.
  - **AMD ROCm** via the SYCL / AdaptiveCpp path. Validated on RDNA2
    (`gfx1031`, RX 6700 XT, 12 GB) — bit-exact parity with the CUDA
    backend across the sort / bucket-offsets / g_x kernels, and
    farmable plots end-to-end. ROCm 6.2 required (newer ROCm versions
    have LLVM packaging breakage — see [`compose.yaml`](compose.yaml)
    rocm-service comments). Build picks `ACPP_TARGETS=hip:gfxXXXX`
    from `rocminfo` automatically for RDNA2+. Other gfx targets
    (`gfx1030` / `gfx1100`) build cleanly but are untested on real
    hardware. **RDNA1 cards (`gfx1010`/`gfx1011`/`gfx1012`, e.g.
    Radeon Pro W5700, RX 5700 / 5700 XT)** default to
    `ACPP_TARGETS=generic` (SSCP JIT) — a previous community
    workaround AOT-spoofed them as `gfx1013`, but that has been
    observed to silently produce no-op kernel stubs on at least one
    W5700 + ROCm 6 + AdaptiveCpp 25.10 setup. Generic SSCP works
    end-to-end through k=24 parity tests. Two opt-in escape hatches
    preserved: `XCHPLOT2_FORCE_GFX_SPOOF=1` to restore the legacy
    AOT spoof, `XCHPLOT2_NO_GFX_SPOOF=1` to AOT-target the actual
    ISA natively (build will fail clearly if AdaptiveCpp doesn't
    accept it).
  - **Intel oneAPI** is wired up but untested.
  - **CPU** (no GPU) via AdaptiveCpp's OpenMP backend. Opt-in with
    `--cpu` (or `--devices cpu`) — never the default. Plotting is
    1-2 orders of magnitude slower than a real GPU; intended for
    headless CI, GPU-less dev machines, or as an extra worker
    alongside GPUs (`--cpu --devices all` runs every visible GPU
    plus a CPU worker on the same batch). Build the container with
    `scripts/build-container.sh --gpu cpu` for the standalone CPU
    image (`xchplot2:cpu`, ~400 MB; no CUDA / ROCm in the image).
- **VRAM:** four tiers, picked automatically based on free device
  VRAM at k=28. All four produce byte-identical plots.
  - **Pool** (~11 GB device + ~4 GB pinned host): fastest steady-state,
    used on 12 GB+ cards.
  - **Plain streaming** (~7.3 GB peak + 128 MB margin): per-plot
    allocations, no pinned-host parks, single-pass T2 match. ~400 ms/
    plot faster than compact. Used on 10-11 GB cards that can't fit
    the pool but have headroom above compact.
  - **Compact streaming** (~5.2 GB peak + 128 MB margin): full
    park/rehydrate + N=2 T2 match tiling. Used on 6-8 GB cards where
    plain won't fit. 6 GB cards (RTX 2060, RX 6600) are on the edge;
    8 GB cards (3070, 2070 Super) comfortably fit.
  - **Minimal streaming** (~3.76 GB peak + 128 MB margin): six layered
    cuts on top of compact — N=8 T2 match staging, tiled gathers in
    T1/T2 sort, sliced T1 match (per section_l), sliced T3 match
    (T2 inputs parked on host, slice H2D'd per section pair),
    per-tile CUB outputs in T1/T2/T3 sort with USM-host merges, and
    tiled Xs gen+sort+pack with host-pinned accumulation. Bottleneck
    moves from compact's T1 sort (5200 MB) to T3 match (3754 MB).
    Targets 5 GiB+ cards (RTX 2060, RX 6600 XT, RX 7600) comfortably;
    4 GiB cards (GTX 1050 Ti, RTX 3050 4GB, MX450) are an edge case
    since real 4 GiB hardware reports ~3.5 GiB free post-CUDA-context.
    Trade-off: ~6 extra cap-sized PCIe round-trips per plot. k=28
    wall on sm_89: ~34 s/plot vs ~13 s for compact. Detailed
    breakdown in [VRAM](#vram).

  With [`--devices`](#multi-gpu---devices), each worker picks its own
  tier from its own GPU's free VRAM — heterogeneous rigs (e.g. one
  12 GB + one 8 GB card) plot concurrently with each device on its
  matching tier.
- **PCIe:** Gen4 x16 or wider recommended. A physically narrower slot
  (e.g. Gen4 x4) adds ~240 ms per plot to the final fragment D2H
  copy; check `cat /sys/bus/pci/devices/*/current_link_width`
  under load if throughput looks off.
- **Host RAM:** ≥ 16 GB recommended; `batch` mode pins ~4 GB of host
  memory for D2H double-buffering (pool or streaming).
- **CUDA Toolkit:** 12+ required for the NVIDIA build path (tested on
  13.x). Skipped automatically on AMD/Intel builds where `nvcc` isn't
  available — `build.rs` runs `nvcc --version` and flips
  `XCHPLOT2_BUILD_CUDA=OFF` when missing. The toolkit-vs-arch matrix:
  - `sm_50` – `sm_72` (Maxwell / Pascal / Volta): need CUDA **12.9**
    (last toolkit with codegen for these arches — 13.x dropped them
    entirely). `build.rs` catches the 13.x + old-arch pairing in a
    preflight and points at the fix path.
  - `sm_75` – `sm_90` (Turing / Ampere / Hopper): 12.x or 13.x both
    work.
  - `sm_120` (RTX 50-series Blackwell): need 12.8+; earlier toolkits
    lack Blackwell codegen.
- **OS:** Linux (tested on modern glibc distributions) is the supported
  path. Windows users route through either the `cuda-only` branch
  natively (NVIDIA + MSVC + CUDA) or WSL2 (any vendor WSL2 supports)
  — see [Windows](#windows) below. macOS is not supported (no CUDA,
  no modern SYCL runtime).

## Build

### Which path should I use?

- **"I just want to plot, Linux host"** → **container (path 1)**. Smallest
  host install (just `podman` + `podman-compose` + the GPU passthrough
  bits — `scripts/install-container-deps.sh` installs all of it). All
  toolchain lives inside the image. Auto-detects your GPU and pins the
  right CUDA / ROCm base.
- **"NVIDIA only, native binary, no SYCL/AdaptiveCpp"** → **`cuda-only`
  branch (path 2)**. Three host packages — `cmake` + `build-essential`
  + the CUDA Toolkit. No LLVM/lld/AdaptiveCpp install. Smaller dep
  surface than main; same end result for NVIDIA users.
- **"Full build — AMD / Intel / CPU support, parity tests on the host"**
  → **`install-deps.sh` (path 3)**. Auto-installs cmake, lld, LLVM 18,
  AdaptiveCpp from source. ~30-45 min first-time setup.

Three ways to get the dependencies in place, easiest first:

### 1. Container (`podman compose` or `docker compose`)

Easiest path — `scripts/build-container.sh` does host-side GPU
probing and feeds the right env vars to `compose build`. If you're
starting from a fresh host, `scripts/install-container-deps.sh`
installs the engine + GPU passthrough bits first (podman + GPU probe
+ `nvidia-container-toolkit` / video-render groups, as appropriate;
no native CUDA / ROCm / LLVM / AdaptiveCpp on the host):

```bash
./scripts/install-container-deps.sh    # one-time: engine + GPU passthrough
./scripts/build-container.sh           # auto: nvidia-smi → cuda, rocminfo → rocm
podman compose run --rm cuda plot -k 28 -n 10 -f <farmer-pk> -c <pool-contract> -o /out
```

**The script handles a handful of host-side decisions that bare
`podman compose build` can't:**

- **Vendor pick** (cuda / rocm / intel / cpu) from nvidia-smi /
  rocminfo, or `--gpu cpu` to force CPU.
- **Multi-GPU fat binary** (e.g. `CUDA_ARCH="61;86"` on a
  1070+3060 rig) — compose alone defaults to a single arch.
- **Pascal/Volta auto-pin** to `nvidia/cuda:12.9.1-devel-ubuntu24.04`
  when min arch < 75. CUDA 13 dropped sub-Turing codegen, so a Pascal
  user without this pin hits a build-time `Unsupported gpu
  architecture 'compute_61'` error inside the container.
- **AMD `ACPP_GFX` extract** from rocminfo + the RDNA1 (gfx1010 →
  gfx1013) workaround for Radeon Pro W5700.
- **`--no-cache`** pass-through to force a clean rebuild after a
  toolchain bump.

You CAN run `podman compose build` directly — it just means setting
those env vars yourself. The compose YAML's defaults are conservative
(CUDA 13.0, sm_89, no AMD target without `ACPP_GFX`), so plain
`podman compose build cuda` only "just works" on Turing-or-newer
NVIDIA hosts. Anything else needs the script or the equivalent
manual env:

[`compose.yaml`](compose.yaml) defines four vendor-specific services
sharing one [`Containerfile`](Containerfile); the script just runs
`compose build` against whichever matches your hardware. Override
manually if you prefer:

```bash
# NVIDIA (default sm_89; override via $CUDA_ARCH=120 etc.)
podman compose build cuda

# AMD ROCm — set $ACPP_GFX from `rocminfo | grep gfx`.
ACPP_GFX=gfx1031 podman compose build rocm    # Navi 22
ACPP_GFX=gfx1100 podman compose build rocm    # Navi 31 (default)

# Intel oneAPI (experimental, untested).
podman compose build intel

# CPU-only (no GPU; AdaptiveCpp OpenMP backend; ~400 MB image).
# Plotting is 1-2 orders of magnitude slower than GPU — see CPU bullet
# under Hardware compatibility for the use case.
podman compose build cpu
```

Plot files land in `./plots/` on the host. The container also bundles
the parity tests (`sycl_sort_parity`, `sycl_g_x_parity`, etc.) under
`/usr/local/bin/` for quick first-port validation on a new GPU:

```bash
podman compose run --rm --entrypoint /usr/local/bin/sycl_sort_parity rocm
```

First build is ~15-30 min (AdaptiveCpp + LLVM 18 compile from source);
subsequent rebuilds reuse the cached layers. GPU performance inside
the container is identical to native — kernels run on real hardware
via the engine's GPU pass-through:

- **NVIDIA**: requires `nvidia-container-toolkit` on the host. For
  Docker users, also run once after install:
  ```bash
  sudo apt install nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
  Podman 5.x with CDI works without the runtime-configure step.
- **AMD**: `/dev/kfd` + `/dev/dri` device files. The compose `rocm`
  service handles this automatically; for bare `podman/docker run`
  pass `--device /dev/kfd --device /dev/dri --group-add video`.

#### AMD container — sudo, `--privileged`, and `ACPP_GFX`

AMD GPUs need three pieces of friction handled correctly. None are
optional on most hosts, and getting any one wrong tends to fail
silently or in confusing ways:

1. **`ACPP_GFX` must be set** to your GPU's gfx target. The kernels
   are AOT-compiled for a specific amdgcn ISA at build time. If the
   wrong arch is baked in, HIP loads the fatbinary without complaint
   but the kernels execute as silent no-ops at runtime — sort returns
   input unchanged, AES match finds zero matches, plots look valid
   but contain non-canonical proofs that won't qualify against real
   challenges. `compose.yaml` defaults `ACPP_GFX` to a placeholder
   string that AdaptiveCpp's HIP backend rejects loudly at build
   time, so an unset value fails fast with the placeholder visible
   in the error rather than silently using a default like `gfx1100`.
   Common values (`rocminfo | grep gfx` to confirm yours):

   - `gfx1030` — RDNA2 Navi 21 (RX 6800 / 6800 XT / 6900 XT)
   - `gfx1031` — RDNA2 Navi 22 (RX 6700 XT / 6700 / 6800M)
   - `gfx1100` — RDNA3 Navi 31 (RX 7900 XTX / XT)
   - `gfx1101` — RDNA3 Navi 32 (RX 7800 XT / 7700 XT)

2. **Rootful `--privileged` for runs.** Rootless podman's default
   seccomp filter + capability set blocks some of the KFD ioctls
   `libhsa-runtime64` needs during DMA setup. Without them you get
   a segfault deep inside the HSA runtime on the very first
   host→device copy, even though `rocminfo` works fine. Builds don't
   need GPU access and can stay rootless if you prefer.

3. **`sudo` strips environment variables by default**, including
   the `ACPP_GFX` you set in your shell. So a bare
   `sudo podman compose build rocm` loses it. Either invoke the
   build script (it sets the var inside the sudo'd shell where
   compose can see it) or pass the var through explicitly.

The recommended invocation pair, in order of how short each one is:

```bash
# Build (autodetects ACPP_GFX from rocminfo — works under sudo too):
sudo ./scripts/build-container.sh

# Run a single test plot at k=22:
sudo podman run --rm --privileged \
    --device /dev/kfd --device /dev/dri \
    -v $PWD/plots:/out xchplot2:rocm \
    test 22 <plot_id_hex> 2 0 0 -G -o /out

# Run real plotting:
sudo podman run --rm --privileged \
    --device /dev/kfd --device /dev/dri \
    -v $PWD/plots:/out xchplot2:rocm \
    plot -k 28 -n 10 -f <farmer-pk> -c <pool-contract> -o /out
```

If `sudo` doesn't carry `/opt/rocm/bin` on your distro and the build
script can't find `rocminfo`, fall back to one of:

```bash
sudo -E ./scripts/build-container.sh                       # preserve your shell PATH
sudo ACPP_GFX=gfx1031 ./scripts/build-container.sh         # explicit, no rocminfo needed
```

Or skip the script entirely:

```bash
sudo ACPP_GFX=gfx1031 podman compose build rocm
```

For convenience, drop a wrapper at `~/.local/bin/xchplot2-amd`:

```bash
#!/bin/bash
exec sudo podman run --rm --privileged \
    --device /dev/kfd --device /dev/dri \
    -v "$PWD/plots:/out" xchplot2:rocm "$@"
```

Then `xchplot2-amd plot -k 28 -n 10 -f ... -c ... -o /out` just works.

### 2. Native install via `scripts/install-deps.sh`

```bash
./scripts/install-deps.sh        # auto-detects distro + GPU vendor
```

Installs the toolchain via the system package manager (Arch, Ubuntu /
Debian, Fedora) plus AdaptiveCpp from source into `/opt/adaptivecpp`.
GPU vendor is auto-detected: `nvidia-smi` / `rocminfo` first,
`/sys/class/drm` PCI IDs as fallback (so fresh installs without driver
tools still work). On a no-GPU host (CI / build box) the script
errors out — pass `--gpu nvidia` to install the toolchain anyway.
`--gpu amd` forces the AMD path on dual-vendor hosts. Intel detection
currently errors with a hint pointing at `--gpu nvidia` (the SYCL
toolchain JITs onto Intel via AdaptiveCpp's generic SSCP target) or
the container. Pass `--no-acpp` to skip the AdaptiveCpp build and
let CMake fall back to FetchContent.

### 3. Manual / FetchContent fallback

If you'd rather install dependencies yourself, the toolchain is:

| Dep | Notes |
|---|---|
| **AdaptiveCpp 25.10+** | SYCL implementation. CMake auto-fetches it via FetchContent if `find_package(AdaptiveCpp)` fails — first build adds ~15-30 min. Disable with `-DXCHPLOT2_FETCH_ADAPTIVECPP=OFF` if you want a hard error. |
| **CUDA Toolkit 12+** (headers) | Required on **every** build path because AdaptiveCpp's `half.hpp` includes `cuda_fp16.h`. `nvcc` itself only runs when `XCHPLOT2_BUILD_CUDA=ON`. Default is vendor-aware — `ON` for NVIDIA GPUs, `OFF` for AMD / Intel GPUs (even if `nvcc` is installed), falling through to `nvcc`-presence only when no GPU is probed (CI / container). Override with the env var. |
| **LLVM / Clang ≥ 18** | `clang`, `lld` (AdaptiveCpp's CMake requires `ld.lld`), plus the libclang dev packages. `install-deps.sh` installs all of them; manual installs need to add `lld-18` (apt) / `lld` (dnf, pacman) explicitly. |
| **C++20 compiler** | clang ≥ 18 or gcc ≥ 13. |
| **CMake ≥ 3.24**, **Ninja**, **Python 3** | build tools. |
| **Boost.Context, libnuma, libomp** | AdaptiveCpp runtime deps. |
| **Rust toolchain** (stable) | for `keygen-rs` and `cargo install`. |

`pos2-chip` and `FSE` are auto-fetched at CMake configure time
(`FetchContent`); override `-DPOS2_CHIP_DIR=/abs/path` for a local
checkout.

For non-NVIDIA targets, the build also probes:
- **ROCm 6+** (`rocminfo`): if found, sets `ACPP_TARGETS=hip:gfxXXXX`.
- **Intel oneAPI** (Level Zero / compute-runtime): manual `ACPP_TARGETS`.

### `cargo install`

```bash
cargo install --git https://github.com/Jsewill/xchplot2
```

`build.rs` auto-detects the local GPU's compute capability by querying
`nvidia-smi --query-gpu=compute_cap` and builds for only that
architecture. That keeps the binary small and the build fast when the
install and the target GPU are the same machine.

If auto-detection fails (no `nvidia-smi` in `PATH`, or
`nvidia-smi` can't see a GPU — common when building inside a container
or on a headless build host that lacks the CUDA driver), the build
falls back to `sm_89`. Note that arch-detect picks *which CUDA arch* —
*whether* CUDA TUs build at all is a separate vendor-aware decision
(see `XCHPLOT2_BUILD_CUDA` in [Environment variables](#environment-variables)).

If you need to target a GPU that isn't the one doing the build — or if
you want a single "fat build" binary that covers multiple
architectures — override with `$CUDA_ARCHITECTURES`:

```bash
# Fat build for Ada (4090) and Blackwell (5090):
CUDA_ARCHITECTURES="89;120" cargo install --git https://github.com/Jsewill/xchplot2

# Single target (e.g. Turing 2080 Ti):
CUDA_ARCHITECTURES=75 cargo install --git https://github.com/Jsewill/xchplot2
```

Common values: `61` GTX 10-series, `70` Volta, `75` Turing, `80` A100,
`86` RTX 30-series, `89` RTX 40-series, `90` H100, `120` RTX 50-series.

### CMake (also builds the parity tests)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`pos2-chip` is auto-fetched via `FetchContent`; override with
`-DPOS2_CHIP_DIR=/abs/path/to/pos2-chip` to point at a local checkout.

Outputs:

- `build/tools/xchplot2/xchplot2`
- `build/tools/parity/{aes,xs,t1,t2,t3}_parity` — bit-exact CPU/GPU tests

### Windows

Two supported paths — native `main` doesn't work because AdaptiveCpp
has hard Linux-isms (libnuma, pthreads, LLVM SSCP) that fall apart on
Windows. Jump to the relevant subsection below:

- [Native Windows build (`cuda-only` branch)](#native-windows-build-cuda-only-branch) — recommended NVIDIA path.
- [Native Windows build — SYCL path (adventurous)](#native-windows-build--sycl-path-adventurous) — AMD/Intel/cross-vendor, untested.

**NVIDIA only** → use the
[`cuda-only`](https://github.com/Jsewill/xchplot2/tree/cuda-only)
branch. Pure MSVC + CUDA Toolkit + Rust, no SYCL runtime involved.
See that branch's README for the VS 2022 / Windows SDK / `LIB`
troubleshooting (the `LNK1181: kernel32.lib` and friends).

**AMD or Intel, or if you just want the `main` code path** → run
under **WSL2**. WSL2 is a full Linux environment, so every install
option in this README works there unchanged — `cargo install`,
`scripts/install-deps.sh`, or the container (section 1 above).
Enable WSL2 once with `wsl --install` in an elevated PowerShell.
GPU access in WSL2:

- **NVIDIA**: install the latest "NVIDIA GPU Driver for Windows",
  nothing else — CUDA shows up inside WSL2 automatically.
- **AMD**: ROCm 6.1+ supports a limited card list on WSL2 (RX 7900
  XTX, Radeon Pro W7900, specific Instincts). Follow AMD's "Install
  ROCm on WSL" guide.
- **Intel**: oneAPI on WSL2 via the Intel Linux graphics driver.

Once the GPU is visible from a WSL2 shell (`nvidia-smi`, `rocminfo`,
or `sycl-ls`), proceed with the native Linux instructions above.

#### Native Windows build (cuda-only branch)

Full walkthrough for the NVIDIA native path, repeated here so you
don't have to flip between READMEs. Prerequisites:

- Windows 10 21H2+ or Windows 11, x64
- [Visual Studio 2022](https://visualstudio.microsoft.com/) Community
  with the **"Desktop development with C++"** workload. That workload
  bundles MSVC + the Windows SDK; the SDK is non-optional because it
  ships `kernel32.lib` / `user32.lib` / etc. that `link.exe`
  consumes. If you've trimmed the installer to "C++ build tools"
  only, open **Visual Studio Installer → Modify → Individual
  components** and tick the latest **Windows 11 SDK** before
  retrying.
- [CUDA Toolkit 12.0+](https://developer.nvidia.com/cuda-downloads) —
  install **after** Visual Studio so the CUDA installer wires up the
  MSBuild integration. 12.8+ required for RTX 50-series (Blackwell,
  `sm_120`).
- [Rust](https://www.rust-lang.org/tools/install) using the MSVC
  toolchain (`rustup default stable-x86_64-pc-windows-msvc`).
- [CMake 3.24+](https://cmake.org/download/) and [Git for
  Windows](https://gitforwindows.org/).

Launch the **x64 Native Tools Command Prompt for VS 2022** from the
Start menu — there are several similarly-named prompts (x86 /
x86_64 / 2019 / 2022); the one that matters is the x64 for 2022.
That prompt is the one that sets `LIB`, `INCLUDE`, and `PATH` so
`cl.exe`, `link.exe`, `nvcc`, and `cmake` all see each other plus
the Windows SDK. A plain `cmd` / PowerShell / Windows Terminal tab
does **not** do this — running `cargo install` from one of those
produces `LNK1181: cannot open input file 'kernel32.lib'` at the
first link step.

Quick sanity check in the prompt:

```cmd
where link.exe
echo %LIB%
```

`%LIB%` should include a `...\Windows Kits\10\Lib\...\um\x64`
entry. If it doesn't, you're in the wrong prompt or the Windows SDK
component isn't installed.

Build:

```cmd
set CUDA_ARCHITECTURES=89
cargo install --git https://github.com/Jsewill/xchplot2 --branch cuda-only
```

Or for a local checkout you can iterate on:

```cmd
git clone -b cuda-only https://github.com/Jsewill/xchplot2
cd xchplot2
set CUDA_ARCHITECTURES=89
cargo install --path .
```

Set `CUDA_ARCHITECTURES` to match your card (see the list above).
PowerShell users: use `$env:CUDA_ARCHITECTURES = "89"` instead of
`set`. The CMake path (`cmake -B build -S . && cmake --build build`)
also works inside the same Native Tools prompt if you prefer that
over `cargo install`.

#### Native Windows build — SYCL path (adventurous)

**Strongly recommend WSL2 first** (see the top of this section).
This subsection exists because the path is in principle buildable
on native Windows; in practice it's days of build-system tinkering
without hardware the maintainers can iterate on. Not validated by
us. File an issue with your findings.

What you're signing up for: AdaptiveCpp, built from source on
Windows, pointed at either **AMD HIP SDK for Windows** (for AMD) or
the **CUDA Toolkit** (for NVIDIA through SYCL, if you want the
`main` branch's cross-vendor code path on NVIDIA instead of
`cuda-only`'s CUB one). xchplot2's CMake then finds that install
via `find_package(AdaptiveCpp)` and builds normally. AdaptiveCpp's
FetchContent fallback is **not** viable on native Windows — its own
CMakeLists assumes Linux-isms (libnuma, pthreads) that fall apart.
Pre-install is mandatory.

Prerequisites (on top of the cuda-only prereqs above — MSVC,
Windows SDK, Rust, CMake, Git):

- **LLVM 16–20** with Clang + LLD + the CMake development package
  (`LLVMConfig.cmake` / `ClangConfig.cmake`). Version coverage of
  Windows binary installers is patchy for these components; a
  self-built LLVM is usually the path of least resistance. See
  [AdaptiveCpp's Windows install guide](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md)
  for the currently-recommended source.
- **AMD HIP SDK for Windows** (for the AMD target) from AMD's
  [HIP SDK download page](https://www.amd.com/en/developer/rocm-hub/hip-sdk.html).
  AMD officially flags it as preview: limited card list, different
  device-library layout vs Linux ROCm, runtime coverage varies per
  GPU.
- **CUDA Toolkit 12+** (for the NVIDIA-via-SYCL target). Same
  installer as the `cuda-only` path above.

Rough build sequence from a clean **x64 Native Tools Command Prompt
for VS 2022** (paths are indicative — match your installs):

```cmd
:: 1. Build AdaptiveCpp
git clone --branch v25.10.0 https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
cmake -B build -S . -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX=C:\opt\adaptivecpp ^
    -DLLVM_DIR=C:\path\to\llvm\lib\cmake\llvm ^
    -DWITH_CUDA_BACKEND=OFF ^
    -DWITH_HIP_BACKEND=ON ^
    -DROCM_PATH="C:\Program Files\AMD\ROCm\6.1"
cmake --build build --parallel
cmake --install build

:: 2. Build xchplot2 main against the install
cd \path\to\xchplot2
:: CMAKE_PREFIX_PATH only needed if you installed AdaptiveCpp to a
:: non-default Windows path. The build's auto-discovery only covers
:: Linux's /opt/adaptivecpp — Windows users tell CMake explicitly.
set CMAKE_PREFIX_PATH=C:\opt\adaptivecpp
set ACPP_TARGETS=hip:gfx1101
set XCHPLOT2_BUILD_CUDA=OFF
cargo install --path .
```

Flip `WITH_HIP_BACKEND` ↔ `WITH_CUDA_BACKEND` and set
`ACPP_TARGETS=cuda:sm_XX` for the NVIDIA-through-SYCL variant.

Failure modes you should expect to triage:

- **Missing LLVM CMake modules** — source-built LLVM with
  `LLVM_INSTALL_UTILS=ON` and the clang / clang-tools-extra
  projects enabled is the reliable recipe.
- **Generic SSCP compiler disabled** (`DEFAULT_TARGETS` warning
  during AdaptiveCpp configure) — harmless if you set
  `ACPP_TARGETS=hip:gfxXXXX` explicitly at xchplot2's configure.
- **`ROCM_PATH` mismatch** — AMD's Windows installer versions the
  directory (`C:\Program Files\AMD\ROCm\6.1\`); match it exactly.
- **Clean build, runtime kernel failures** — the HIP SDK for
  Windows preview doesn't cover every GPU the Linux ROCm path
  does. Run `scripts/test-multi-gpu.sh` / `xchplot2 test 22 ...`
  with a k=22 plot first and `xchplot2 verify` the result before
  committing a large batch.

Seriously, try WSL2 first.

## Use

### Standalone (farmable plots)

```bash
xchplot2 plot -k 28 -n 10 \
    -f <farmer-pk> \
    -c <pool-contract-address> \
    -o <output-dir>
```

Pool variants: `-p <pool-pk>` or `--pool-ph <pool-ph>`. Other common
flags: `-s <strength>`, `-T` testnet, `-S <seed>` for reproducible runs,
`-v` verbose. Full help: `xchplot2 -h`.

For long batches, `--skip-existing` skips plots whose output file is
already a complete `.plot2` (magic bytes + non-trivial size), and
`--continue-on-error` logs per-plot failures and keeps going instead of
aborting the whole run. Both flags work for `plot` and `batch` modes.

Plots are written to `<name>.plot2.partial` and atomically renamed on
completion, so a crash / `SIGINT` / `ENOSPC` mid-write never leaves a
malformed plot at the destination. A first `Ctrl-C` asks the plotter to
finish the plot in flight and stop; a second hard-kills.

#### Grouping plots: `-i <plot-index>` and `-g <meta-group>`

Both are v2 PoS fields and default to 0.
`<plot-index>` (u16) is the within-group identifier; `plot -n N`
uses it as the base and increments per plot (so `-i 0 -n 1000`
produces plots with `plot_index` 0..999).
`<meta-group>` (u8) is a challenge-isolation boundary — plots with
different meta_group values are guaranteed never to pass the same
challenge.

The PoS2 spec defines a grouped-plot file layout (multiple plots
interleaved into one container per storage device, for harvester
seek amortization), but the on-disk format is not yet defined
upstream in `pos2-chip` / `chia-rs`. xchplot2 currently produces one
`.plot2` file per plot — this is in lieu of those upstream
decisions. When the grouped layout lands, the auto-incrementing
`<plot-index>` above is the per-plot within-group identifier it
will expect.

#### Multi-device: `--devices` and `--cpu`

`xchplot2 devices` prints id, name, backend, VRAM, compute-unit count,
and which sort path each device will use (CUB on cuda-backend devices
when this build links CUB, SortSycl otherwise) — the printed `[N]`
index is the value `--devices N` accepts:

```
$ xchplot2 devices
Visible devices (2 GPU + 1 CPU):
  [0]   NVIDIA GeForce RTX 4090          backend=cuda       vram=24076 MB  CUs=128   sort:CUB
  [1]   AMD Radeon Pro W5700             backend=hip        vram= 8176 MB  CUs=36    sort:SYCL
  [cpu] Host CPU plotter                 backend=omp        threads=32             sort:SYCL  (1-2 orders slower than GPU)

Use `--devices N` (id) for a specific GPU, `--devices cpu`
for the host CPU, `--devices all` for one worker per GPU,
or any comma combination (e.g. `all,cpu`).
```

Both `plot` and `batch` accept `--devices <SPEC>` to fan plots out
across multiple devices — one worker thread per device, each with its
own buffer pool and writer channel. Plots are partitioned round-robin,
so a batch of 10 plots on 2 GPUs sends plots 0/2/4/6/8 to the first
GPU and 1/3/5/7/9 to the second.

```bash
# Every visible GPU — enumerated at runtime.
xchplot2 plot --k 28 --num 10 -f <farmer-pk> -c <pool-contract> \
    --out /mnt/plots --devices all

# Only these specific GPU ids (sorted, deduplicated).
xchplot2 plot ... --devices 0,2,3

# Explicit single id (same as omitting the flag on a single-GPU host).
xchplot2 plot ... --devices 0

# CPU-only: AdaptiveCpp OpenMP backend (slow). Use the `cpu` token in
# --devices, or the standalone --cpu flag (equivalent on its own).
xchplot2 plot ... --devices cpu
xchplot2 plot ... --cpu

# Heterogeneous: every GPU PLUS a CPU worker on the same batch.
# --cpu is orthogonal to --devices and appends a CPU worker.
xchplot2 plot ... --devices all --cpu
xchplot2 plot ... --devices 0,1,cpu     # same effect, written as a list
```

CPU plotting is **1-2 orders of magnitude slower than GPU** — meant for
GPU-less hosts, headless CI, or as an extra background worker. Don't
expect GPU-grade throughput from a CPU worker on a heterogeneous batch.

Omitted flag = single device via the default SYCL / CUDA selector —
identical to pre-multi-GPU behavior, zero regression risk.

**Caveats for v1:**

- Static round-robin partition. If your GPUs differ in speed the
  batch finishes only as fast as the slowest worker's slice; use
  `--devices` to pick matched cards when that matters.
- Each worker gets its own ~4 GB pinned host pool, so host RAM scales
  linearly. A 4-GPU rig pins ~16 GB — size accordingly.
- The workers share `stderr` (line-buffered, atomic per-`fprintf`) so
  log lines from different GPUs may interleave. Fine for progress,
  not for parsing.

Smoke test: `scripts/test-multi-gpu.sh` exercises argument parsing
(works on any host, even single-GPU) and, when 2+ GPUs are visible,
runs a live k=22 plot across `--devices 0,1`.

### Lower-level subcommands

```bash
xchplot2 test          <k> <plot-id-hex> [strength] ...    # single plot, raw inputs
xchplot2 batch         <manifest.tsv> [-v] [--skip-existing] [--continue-on-error]
                                             [--devices <SPEC>]
xchplot2 verify        <file.plot2> [--trials N]           # run N random challenges
xchplot2 parity-check  [--dir PATH]                        # CPU↔GPU regression screen
```

`verify` opens a `.plot2` through pos2-chip's CPU prover and runs N
(default 100) random challenges. Zero proofs across a reasonable sample
strongly indicates a corrupt plot; the command exits non-zero in that
case. Intended as a quick sanity check before farming a newly built
batch — not a replacement for `chia plots check`.

`parity-check` execs every `*_parity` binary in `--dir` (default
`./build/tools/parity`) and summarizes PASS/FAIL with per-test wall
time. Use after a refactor or driver update to confirm CPU↔GPU
agreement is still bit-exact across `aes` / `xs` / `t1` / `t2` / `t3` /
`plot_file`. Requires `cmake --build` to have produced the parity
binaries first.

## Troubleshooting

- **Listing visible GPUs**: `xchplot2 devices` prints id, name, backend,
  VRAM, compute-unit count, and which sort path each device will use
  (CUB on cuda-backend devices when this build links CUB; SortSycl
  otherwise). Use the printed `[N]` index with `--devices N` for
  `plot` / `batch`.

- **Hybrid hosts (NVIDIA + AMD/Intel on the same box)**: a single
  binary handles all visible GPUs. `xchplot2 plot --devices all`
  spawns a worker per GPU; each worker picks the right sort backend
  at queue construction (CUB on NVIDIA, hand-rolled SYCL radix on
  AMD/Intel) via the runtime dispatcher in `SortDispatch.cpp`. No
  rebuild required to add a second-vendor card.

- **`[AdaptiveCpp Warning] [backend_loader] Could not load library:
  /opt/adaptivecpp/lib/hipSYCL/librt-backend-cuda.so (libcudart.so.11.0:
  cannot open shared object file)`**: cosmetic only — AdaptiveCpp
  built with CUDA backend support but no CUDA runtime to load. Happens
  when AdaptiveCpp was installed out-of-band rather than via
  `scripts/install-deps.sh --gpu amd` (which sets
  `-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE`). To suppress without a
  rebuild: `export ACPP_VISIBILITY_MASK=hip;omp` so AdaptiveCpp skips
  the CUDA backend probe entirely.

- **`T1 match produced 0 entries`** on RDNA1 (`gfx1010` / `gfx1011` /
  `gfx1012`, including the Radeon Pro W5700 / RX 5700 XT). The
  community `gfx1013` AOT-spoof default was observed to silently
  compile no-op kernel stubs on at least one W5700 + ROCm 6 +
  AdaptiveCpp 25.10 host. Default flipped to `ACPP_TARGETS=generic`
  (SSCP JIT) in recent main; `cargo install --force` past commit
  `d939ee8` restores correct behavior. To restore the old spoof,
  `XCHPLOT2_FORCE_GFX_SPOOF=1 cargo install ...`. The startup self-
  test in `SyclBackend::queue()` catches the no-op-kernel case at
  queue construction with a clear exception, so this surfaces
  immediately rather than as empty pipeline output minutes in.

- **`CUB ... invalid argument`** mid-pipeline, or
  **`sycl_backend::queue: device id 0 out of range (found 0 usable
  GPU device(s))`** with `--devices N` while the default selector
  finds a GPU: pre-`762fde2` symptoms of CUB-only sort being
  dispatched against an AMD/Intel device (or being filtered out of
  the device list). The runtime sort dispatcher fixes both — `git
  pull && cargo install --path . --force` to upgrade.

- **Deep-pipeline diagnostics**: set `POS2GPU_T1_DEBUG=1` for verbose
  per-stage dumps (Xs gen / sort intermediates, T1 match input/output
  samples, AES T-table sanity). Useful when the symptom isn't on the
  list above and you want to localize where the data goes wrong.

## Environment variables

| Variable                      | Effect                                                                  |
|-------------------------------|-------------------------------------------------------------------------|
| `XCHPLOT2_BUILD_CUDA=ON\|OFF` | Override the build-time CUB / nvcc-TU switch. Default is vendor-aware (NVIDIA → ON; AMD / Intel → OFF; no GPU → `nvcc`-presence). Force `OFF` on dual-toolchain hosts (CUDA + ROCm) where you want the SYCL-only build. |
| `XCHPLOT2_STREAMING=1`        | Force the low-VRAM streaming pipeline even when the pool would fit.     |
| `XCHPLOT2_STREAMING_TIER=plain\|compact\|minimal` | Override the streaming-tier auto-pick (plain = ~7.3 GB peak, no parks; compact = ~5.2 GB peak, full parks + N=2 T2 match tiling; minimal = ~3.76 GB peak with full host-pinned slicing of T1/T3 match + tiled CUB outputs in all sort phases + tiled Xs gen/sort/pack — targets 5 GiB+ cards). Equivalent CLI flag: `--tier`. |
| `POS2GPU_MAX_VRAM_MB=N`       | Cap the pool/streaming VRAM query to N MB (exercise streaming fallback).|
| `POS2GPU_STREAMING_STATS=1`   | Log every streaming-path `malloc_device` / `free`.                      |
| `POS2GPU_POOL_DEBUG=1`        | Log pool allocation sizes at construction.                              |
| `POS2GPU_PHASE_TIMING=1`      | Per-phase wall-time breakdown (Xs / sort / T1 / T2 / T3) on stderr.     |
| `ACPP_GFX=gfxXXXX`            | AMD only — required at **build** time; sets AOT target for amdgcn ISA. |
| `ACPP_TARGETS=...`            | Override AdaptiveCpp target selection (defaults: NVIDIA `generic`, AMD `hip:$ACPP_GFX`). |
| `CUDA_ARCHITECTURES=sm_XX`    | Override the CUDA arch autodetected from `nvidia-smi`.                  |
| `CUDA_PATH=/path/to/cuda`     | Override the CUDA Toolkit root for linking (default: `/opt/cuda`, `/usr/local/cuda`). Useful on JetPack / non-standard installs. |
| `CUDA_HOME=/path/to/cuda`     | Fallback for `CUDA_PATH` — same effect.                                 |
| `POS2_CHIP_DIR=/path`         | Build-time: point at a local pos2-chip checkout instead of FetchContent.|
| `XCHPLOT2_TEST_GPU_COUNT=N`   | Override `scripts/test-multi-gpu.sh`'s auto-detected GPU count (forces run / skip without consulting `nvidia-smi`). |

## Testing farming on a testnet

v2 (CHIP-48) farming in stock chia-blockchain is presently unfinished
upstream — services aren't wired into the farmer group, a message
handler's signature doesn't match its decorator, `ProofOfSpace.
challenge` is computed from the wrong input, and the dependency pin
on `chia_rs` excludes the 0.42 release where `compute_plot_id_v2`
lives. `contrib/testnet-farming.patch` is a minimal self-contained
fix-up that gets a private testnet running end-to-end:

```bash
git clone https://github.com/Chia-Network/chia-blockchain
cd chia-blockchain
git checkout 39f8bec88   # 2.7.0 Checkpoint Merge
git apply /path/to/xchplot2/contrib/testnet-farming.patch
```

The patch's header comment describes each hunk. None of the changes
are xchplot2-specific — they're the farmer / harvester / daemon
pieces any v2 plot needs for farming, regardless of who produced it.

## Architecture

```
src/gpu/                 GPU kernels — AES, Xs, T1, T2, T3.
                           CUDA path: .cu files via nvcc + CUB sort.
                           SYCL path: matching .cpp files via
                             AdaptiveCpp + hand-rolled LSD radix.
src/host/
├── GpuPipeline          Xs → T1 → T2 → T3 device orchestration;
│                          pool + streaming (low-VRAM) variants
├── GpuBufferPool        persistent device + 2× pinned host pool
├── BatchPlotter         producer / consumer batch driver
└── PlotFileWriterParallel  sole TU touching pos2-chip headers
tools/xchplot2/          CLI: plot / test / batch
tools/parity/            CPU↔GPU bit-exactness tests
keygen-rs/               Rust staticlib: plot_id_v2, BLS HD, bech32m
```

## VRAM

PoS2 plots are k=28 by spec. Four code paths, dispatched automatically
based on available VRAM at batch start:

- **Pool path (~11 GB device + ~4 GB pinned host; 12 GB+ cards
  reliably).** The persistent buffer pool is sized worst-case and
  reused across plots in `batch` mode for amortised allocator cost and
  double-buffered D2H. Xs sort's keys_a slot aliases d_storage tail
  (idle during Xs gen+sort), trimming pair_b's worst case from
  `max(cap·12, 4·N·u32 + cub)` to `max(cap·12, 3·N·u32 + cub)` —
  saves ~1 GiB at k=28. Targets: RTX 4090 / 5090, A6000, H100,
  RTX 4080 (16 GB), and 12 GB cards like RTX 3060 / RX 6700 XT.
- **Plain streaming (~7.3 GB peak + 128 MB margin; ≥ 7.42 GiB free at
  k=28).** Allocates per-phase and frees between phases, but keeps
  large intermediates (`d_t1_meta`, `d_t1_keys_merged`, `d_t2_meta`,
  `d_t2_xbits`, `d_t2_keys_merged`) alive across their idle windows
  instead of parking them on pinned host. T2 match runs as a single
  full-cap pass (N=1). Used on 10-11 GB cards that can't fit the pool
  but have headroom above the compact floor. ~400 ms/plot faster than
  compact at k=28 because there are no park/rehydrate PCIe round-trips.
- **Compact streaming (~5.2 GB peak + 128 MB margin; ≥ 5.33 GiB free
  at k=28).** All three match phases (T1/T2/T3) are tiled N=2 across
  disjoint bucket ranges with half-cap device staging and
  D2H-to-pinned-host between passes. T1 + T2 sorts are tiled (N=2 and
  N=4) with merge trees, and `d_t1_meta`, `d_t2_meta`, and the
  `*_keys_merged` buffers are parked on pinned host across their
  sort phases and JIT-H2D'd only for the next consumer. Xs is inlined
  as gen → sort → pack with separate-allocation scratch so keys_a +
  vals_a can be freed right after CUB sort. Peak at k=28 is
  **5200 MB** (measured on sm_89); per-phase live maxes:

  | Phase     | Peak (MB) |
  |-----------|----------:|
  | Xs        | 4128 |
  | T1 match  | 5168 |
  | T1 sort   | 5200 |
  | T2 match  | 5200 |
  | T2 sort   | 5200 |
  | T3 match  | 5200 |
  | T3 sort   | 4228 |

  A BatchPlotter preflight rejects cards reporting less than
  `streaming_peak_bytes(k) + 128 MB` free before any queue work, so
  mid-pipeline OOM is impossible on supported configurations.
  Practical targets: 6 GB cards on the edge (card-dependent; RTX 2060
  typically has ~5.5 GiB free which has ~170 MB slack over the
  5328 MB requirement), 8 GB cards comfortable, 10 GB and up ample.
  Log the full alloc trace with `POS2GPU_STREAMING_STATS=1`.
- **Minimal streaming (~3.76 GB peak + 128 MB margin; ≥ 3.80 GiB free
  at k=28).** Layered cuts on top of compact:
  - **N=8 T2 match staging.** cap/8 ≈ 570 MB vs compact's cap/2
    ≈ 2280 MB — saves ~1.5 GB on the T2-match peak.
  - **Tiled gathers in T1 sort + T2 sort meta + T2 sort xbits.**
    Each gather output produced in N=4 tiles, D2H'd to host pinned
    (reusing the existing parking buffers) one tile at a time, then
    rebuilt on device after the cap-sized inputs are freed. Drops
    each gather peak from 5200 MB → ~3640 MB.
  - **Sliced T1 match.** N passes (one per section_l) emit to a
    cap/N device staging pair, D2H per pass to host pinned. d_xs
    (2048 MB at k=28) no longer co-resides with full-cap d_t1_meta +
    d_t1_mi → T1-match peak drops from 5168 MB → 3023 MB.
  - **Sliced T3 match.** d_t2_meta_sorted parked on host across
    T3 match; per pass H2Ds the (section_l, section_r) row slices
    onto a small device buffer pair. d_t2_xbits_sorted +
    d_t2_keys_merged remain full-cap on device for binary-search /
    target reads. T3-match peak: 5200 MB → 3754 MB.
  - **Per-tile CUB outputs in T1/T2/T3 sort sub-phases.** T1 and T2
    sort use cap/2 / cap/4 device output buffers respectively, D2H
    per tile to USM-host accumulators, with the existing 2-way merge
    kernel reading USM-host inputs. T2 additionally parks AB / CD
    intermediates to host between tree steps so the final merge
    sees only its own outputs. T3 sort uses cap/2 tile + host-side
    `std::inplace_merge`. CUB sub-phase peaks: 4170-4228 MB →
    3155-3640 MB.
  - **Tiled Xs gen+sort+pack.** N=2 position halves through cap/2
    ping-pong buffers + USM-host accumulator + 2-way merge, then
    pack runs in cap/2 halves with D2H per tile to a host-pinned
    `XsCandidateGpu` accumulator (final d_xs rehydrated H2D).
    Xs phase peak: 4128 MB → 3072 MB.

  Bottleneck after all six cuts is the T3 match phase at 3754 MB.
  Targets 5 GiB+ cards comfortably (RTX 2060, RX 6600 XT, RX 7600
  with ~1.7+ GiB headroom). 4 GiB cards (GTX 1050 Ti / 1650, RTX 3050
  4GB, MX450) are an edge case — real 4 GiB physical hardware
  reports ~3.5 GiB free post-CUDA-context, just under the 3.80 GiB
  required floor. Trade-off: ~6 extra cap-sized PCIe round-trips per
  plot push k=28 wall on sm_89 from ~13 s/plot (compact) to ~34
  s/plot (minimal). There is no smaller tier — a forced minimal on a
  card below the floor throws rather than falling further.

At pool construction `xchplot2` queries `cudaMemGetInfo` on the
CUDA-only build, or `global_mem_size` (device total) on the SYCL
path — SYCL has no portable free-memory query, so the check
effectively approximates "free == total" and lets the actual
`malloc_device` failure trigger the fallback. If the pool doesn't
fit, the streaming-tier dispatch picks the largest tier that fits
with the 128 MB margin: plain if free ≥ 7.42 GiB, else compact if
free ≥ 5.33 GiB, else minimal. `XCHPLOT2_STREAMING=1` forces
streaming even when the pool would fit; `--tier
plain|compact|minimal` (or `XCHPLOT2_STREAMING_TIER`) overrides the
auto-pick. Forced plain or compact below their floor warns and
proceeds (caller's risk); forced minimal below its floor throws
because there is no smaller tier to fall back to.

Plot output is bit-identical across all four paths — streaming
reorganises memory, not algorithms. Verified at k=22 with md5sum
across pool / plain / compact / minimal.

## Performance

k=28, strength=2, RTX 4090 (sm_89), PCIe Gen4 x16. Steady-state per-plot
wall from `xchplot2 batch` (10-plot manifest, mean):

| Build | Per plot | Notes |
|---|---|---|
| pos2-chip CPU baseline | ~50 s | reference |
| `cuda-only` branch | **2.15 s** | original CUDA-only path |
| `main`, `XCHPLOT2_BUILD_CUDA=ON` (CUB sort) | 2.41 s | NVIDIA fast path on the SYCL/AdaptiveCpp port |
| `main`, `XCHPLOT2_BUILD_CUDA=OFF` (hand-rolled SYCL radix) | 3.79 s | cross-vendor fallback (AMD/Intel) on AdaptiveCpp |
| plain streaming tier (10-11 GB cards) | ~5.7 s | no parks, single-pass T2 match; ~400 ms/plot faster than compact |
| compact streaming tier (6-8 GB cards) | ~7.3 s | full parks + N=2 T2 match |
| minimal streaming tier (4 GiB cards) | TBD | full parks + N=8 T2 match; smallest peak (~3.7 GB) |
| `main` on RX 6700 XT (gfx1031 / ROCm 6.2 / AdaptiveCpp HIP) | **9.97 s** | AMD batch steady-state at k=28; T-table AES near-optimal on RDNA2 via this compiler stack |

The `main`/CUB row is +12% over `cuda-only` from extra AdaptiveCpp
scheduling overhead. The SYCL row is +57% over CUB on the same NVIDIA
hardware; ~88% of GPU compute is identical between the two paths
(`nsys` per-kernel breakdown), and the gap is dominated by host-side
runtime overhead in AdaptiveCpp's DAG manager rather than kernel
performance. AMD and Intel runtimes are untested; expect roughly the
SYCL-row latency adjusted for relative GPU throughput.

Numbers above are single-GPU. With `--devices 0,1,...` the batch is
partitioned round-robin across N worker threads (one per device), so
wall-clock throughput is bounded by the slowest device's slice —
≈ linear scaling on matched cards, less if cards differ in speed.
Live multi-GPU plots were confirmed end-to-end on NVIDIA; per-device
numbers will vary with PCIe bandwidth sharing on the host root
complex.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).

## Like this? Send a coin my way!

If you appreciate this, and want to give back, feel free.

xch1d80tfje65xy97fpxg7kl89wugnd6svlv5uag2qays0um5ay5sn0qz8vph8
