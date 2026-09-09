# Installing xchplot2 (`main`)

[README](README.md) · [Reference](REFERENCE.md)

Run checkout-based commands from the repository directory:

```bash
git clone https://github.com/Jsewill/xchplot2
cd xchplot2
```

This branch requires SYCL/AdaptiveCpp, including on NVIDIA. For a native
CUDA build, use the [`cuda-only` installation guide](https://github.com/Jsewill/xchplot2/blob/cuda-only/INSTALL.md).

| Path | Use it for |
|---|---|
| [Container](#container) | Toolchains inside the image; GPU driver and container engine on the host |
| [Native install](#native-install) | System dependencies and AdaptiveCpp installed by the existing script |
| [Manual dependencies](#manual-dependencies) | An existing toolchain or a development setup |

## Container

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
- **AMD `ACPP_GFX` detection** from rocminfo, including the legacy RDNA1
  spoof. RDNA1 users should follow [AMD target selection](#amd-target-selection).
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
ACPP_GFX=gfx1100 podman compose build rocm    # Navi 31

# Intel oneAPI (experimental, untested).
podman compose build intel

# CPU-only (no GPU; AdaptiveCpp OpenMP backend; ~400 MB image).
# CPU plotting is opt-in; see REFERENCE.md for worker controls.
podman compose build cpu
```

Plot files land in `./plots/` on the host. The container also bundles
the parity tests (`sycl_sort_parity`, `sycl_g_x_parity`, etc.) under
`/usr/local/bin/` for quick first-port validation on a new GPU:

```bash
podman compose run --rm --entrypoint /usr/local/bin/sycl_sort_parity rocm
```

The first build also compiles AdaptiveCpp against the image's LLVM;
subsequent rebuilds reuse the cached layers. The container uses the host GPU and driver through GPU pass-through:

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

### AMD container permissions

The ROCm container requires the correct GPU target and access to KFD/DRI.
These notes describe the setup and fallback used on the tested ROCm stack:

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

2. **Permissions for runs.** The compose service relaxes seccomp and adds
   `SYS_ADMIN`. The rootful `--privileged` examples below are a fallback
   for hosts that still fail. Rootless podman's default
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
    -v "$PWD/plots:/out" xchplot2:rocm \
    test 22 <plot_id_hex> 2 0 0 -G -o /out

# Run real plotting:
sudo podman run --rm --privileged \
    --device /dev/kfd --device /dev/dri \
    -v "$PWD/plots:/out" xchplot2:rocm \
    plot -k 28 -n 10 -f <farmer-pk> -c <pool-contract> -o /out
```

If `sudo` doesn't carry `/opt/rocm/bin` on your distro and the build
script can't find `rocminfo`, set the target explicitly:

```bash
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

## Native install

```bash
./scripts/install-deps.sh        # auto-detects distro + GPU vendor
source "$HOME/.cargo/env"        # if the installer just installed rustup
cargo install --path . --locked
```

Installs the toolchain via the system package manager (Arch, Ubuntu /
Debian, Fedora) plus AdaptiveCpp from source into `/opt/adaptivecpp`.
GPU vendor is auto-detected: `nvidia-smi` / `rocminfo` first,
`/sys/class/drm` PCI IDs as fallback (so fresh installs without driver
tools still work). On a no-GPU host (CI / build box) the script
errors out — pass `--gpu nvidia`, `--gpu amd`, or `--gpu intel` to select
the toolchain explicitly. Intel installs Level Zero and its compute
runtime, then enables AdaptiveCpp's Level Zero backend and SPIR-V
translator. Debian 13 does not package the Intel compute runtime;
use the Intel container or a supported native distro instead.
Pass `--no-acpp` to install only system packages. CMake's FetchContent
fallback then builds and installs AdaptiveCpp to `~/.local` (or
`ACPP_PREFIX`), so its runtime survives Cargo's temporary build directory.
Cargo uses these system dependencies; it does not run
the system package manager itself.
On rolling distros, CMake probes nvcc and selects an installed compatible
host compiler if the default is rejected. Explicit `CUDAHOSTCXX`,
`NVCC_CCBIN`, and `CMAKE_CUDA_HOST_COMPILER` settings take precedence.

The [native install CI matrix](CONTRIBUTING.md#install-ci) exercises
fresh Ubuntu, Debian, Fedora, Arch, and Ubuntu WSL2 installs through
both Cargo and CMake. NVIDIA installs on Ubuntu/Debian and Fedora
use NVIDIA's toolkit repository when needed; WSL uses its toolkit-only
repository. Existing CUDA installations are retained on apt systems.

## Manual dependencies

If you'd rather install dependencies yourself, the toolchain is:

| Dep | Notes |
|---|---|
| **AdaptiveCpp 25.10+** | SYCL implementation. CMake auto-fetches and installs it to `~/.local` (or `ACPP_PREFIX`) if `find_package(AdaptiveCpp)` fails — first build adds ~15-30 min. Disable with `-DXCHPLOT2_FETCH_ADAPTIVECPP=OFF` if you want a hard error. |
| **CUDA Toolkit 12+** | NVIDIA only. `nvcc` runs when `XCHPLOT2_BUILD_CUDA=ON`; AMD and Intel builds do not require CUDA headers. The installer uses the current toolkit; pre-Turing GPUs need a compatible CUDA 12.x installation because CUDA 13 dropped their code generation. |
| **ROCm HIP headers, runtime, device libraries** | AMD only: apt `hipcc`, Fedora `rocm-hip-devel`, Arch `hip-runtime-amd` + `rocm-device-libs`; also install `rocminfo` for GPU detection. |
| **Level Zero headers, loader, Intel compute runtime** | Intel only. AdaptiveCpp also needs `WITH_LEVEL_ZERO_BACKEND=ON` and its LLVM SPIR-V translator installed; `install-deps.sh` handles both. |
| **LLVM / Clang 16–20** | AdaptiveCpp 25.10's supported LLVM range, including `lld`, libclang development files, and compiler-rt. The installer selects the newest complete compatible version available, alongside a newer system LLVM when necessary. |
| **C++20 compiler** | clang ≥ 18 or gcc ≥ 13. |
| **CMake ≥ 3.24**, **Ninja**, **Python 3** | build tools. |
| **Boost.Context, libnuma, libomp** | AdaptiveCpp runtime deps. |
| **Rust toolchain** (stable) | for `keygen-rs` and `cargo install`. |

`pos2-chip` and `FSE` are auto-fetched at CMake configure time
(`FetchContent`); override `-DPOS2_CHIP_DIR=/abs/path` for a local
checkout.

For non-NVIDIA targets, the build also probes:
- **ROCm** (`rocminfo`): selects `hip:gfxXXXX`, except RDNA1 defaults to generic SSCP in Cargo. See [AMD target selection](#amd-target-selection).
- **Intel** (Level Zero / compute-runtime): defaults to `ACPP_TARGETS=generic`.

### Toolkit and architecture selection

| GPU architecture | CUDA Toolkit |
|---|---|
| Maxwell, Pascal, Volta (`sm_50`–`sm_72`) | 12.x; the container script selects 12.9 |
| Turing through Hopper (`sm_75`–`sm_90`) | 12 or 13 |
| Blackwell (`sm_120`) | 12.8 or newer for native code generation |

NVIDIA documents the removal of pre-Turing code generation in the
[CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
and the new Blackwell targets in the
[CUDA 12.8 features](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html).

On an older toolkit, Cargo can fall back to PTX for its highest supported
architecture and let the driver JIT it. Use a matching toolkit for native
code generation.

## Cargo install

Toolchain prerequisites for the NVIDIA build:

- **CUDA Toolkit 12.0+** — 12.0 is the floor (the `_v2` runtime ABI we
  link, plus C++20 CUDA dialect, both require 12.0).
- **CMake ≥ 3.26** for nvcc 12.5+ (Debian 12's stock 3.25 doesn't know
  the dialect flags; install Kitware's repo).
- **rustc ≥ 1.85** (rustup `stable`). Distro-packaged Rust (Ubuntu
  24.04 apt cargo is 1.75) is too old for the `edition2024` feature
  required by `chia-client` 0.42.

### NVIDIA dependency sources

| Distro                | CUDA source                         | CMake source            | Rust source       |
|-----------------------|-------------------------------------|-------------------------|-------------------|
| Ubuntu 24.04          | NVIDIA apt `cuda-toolkit-12-9`      | apt `cmake` (3.28)      | rustup `stable`   |
| Ubuntu 22.04          | NVIDIA apt `cuda-toolkit-12-9`      | Kitware apt `cmake`     | rustup `stable`   |
| Debian 12 (Bookworm)  | NVIDIA apt `cuda-toolkit-12-9`      | Kitware apt `cmake`     | rustup `stable`   |
| Fedora             | NVIDIA dnf `cuda-toolkit-12-9`      | dnf `cmake` (3.30)      | rustup `stable`   |
| Rocky / Alma 9        | NVIDIA dnf `cuda-toolkit-12-9`      | dnf `cmake` (3.26)      | rustup `stable`   |
| Arch / CachyOS        | pacman `cuda`                | pacman `cmake`          | pacman `rust` or rustup |

Combinations that **don't** work on a stock install:
- **Ubuntu 24.04 + apt CUDA 12.0**: the old toolkit can fail against
  current glibc headers. Use NVIDIA's repository, as `install-deps.sh` does.
- **Ubuntu 22.04 + apt CUDA**: ships CUDA 11.5 — nvcc too old for the
  C++20 dialect, and `libcudart` predates the `_v2` ABI. Use NVIDIA's
  apt repo instead.
- **Debian 12 + stock CMake**: 3.25 doesn't know how to drive nvcc
  12.5+. Use Kitware's CMake apt repo.
- **Ubuntu 22.04/24.04 + apt cargo**: 1.75 can't parse `edition2024`.
  Install rustup.
- **WSL**: works the same as native — install toolkit + rustup inside
  the WSL distro. WSL's `/usr/lib/wsl/lib` only provides the driver
  (`libcuda.so`), not the runtime.

```bash
# rustup, if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

cargo install --git https://github.com/Jsewill/xchplot2 --locked
```

The application links the CUDA runtime statically. AdaptiveCpp still needs
its installed runtime libraries and a compatible GPU driver.

`build.rs` auto-detects the local GPU's compute capability by querying
`nvidia-smi --query-gpu=compute_cap` and builds for only that
architecture. That keeps the binary small and the build fast when the
install and the target GPU are the same machine.

If auto-detection fails (no `nvidia-smi` in `PATH`, or
`nvidia-smi` can't see a GPU — common when building inside a container
or on a headless build host that lacks the CUDA driver), the build
falls back to `sm_89`. Note that arch-detect picks *which CUDA arch* —
*whether* CUDA TUs build at all is a separate vendor-aware decision
(see `XCHPLOT2_BUILD_CUDA` in [Environment variables](REFERENCE.md#environment-variables)).

If you need to target a GPU that isn't the one doing the build — or if
you want a single "fat build" binary that covers multiple
architectures — override with `$CUDA_ARCHITECTURES`:

```bash
# Fat build for Ada (4090) and Blackwell (5090):
CUDA_ARCHITECTURES="89;120" cargo install --git https://github.com/Jsewill/xchplot2 --locked

# Single target (e.g. Turing 2080 Ti):
CUDA_ARCHITECTURES=75 cargo install --git https://github.com/Jsewill/xchplot2 --locked
```

Common values: `61` GTX 10-series, `70` Volta, `75` Turing, `80` A100,
`86` RTX 30-series, `89` RTX 40-series, `90` H100, `120` RTX 50-series.

## CMake

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

`pos2-chip` is auto-fetched via `FetchContent`; override with
`-DPOS2_CHIP_DIR=/abs/path/to/pos2-chip` to point at a local checkout.

Outputs:

- `build/tools/xchplot2/xchplot2`
- `build/tools/parity/{aes,xs,t1,t2,t3}_parity` — bit-exact CPU/GPU tests

## AMD target selection

The selection path matters on RDNA1 (`gfx1010`, `gfx1011`, `gfx1012`,
including the W5700 and RX 5700 series):

| Build path | Default AMD target |
|---|---|
| Cargo / `build.rs` | Detects the GPU with `rocminfo`; RDNA1 selects `ACPP_TARGETS=generic`, other detected targets select `hip:gfxXXXX`. |
| `scripts/build-container.sh` | Detects `ACPP_GFX`; RDNA1 is still changed to the legacy `gfx1013` AOT spoof. An explicit `ACPP_GFX` is used unchanged. |
| `podman compose build rocm` | AOT only: requires `ACPP_GFX`, then passes `hip:$ACPP_GFX`. A host `ACPP_TARGETS` value does not override this compose argument. |
| Direct CMake | Set `-DACPP_TARGETS=generic` or the intended `hip:gfxXXXX` target explicitly. |

For RDNA1, use the [native installation](#native-install) and Cargo's generic
SSCP path. The legacy spoof produced no-op kernels on a reported W5700 with
ROCm 6 and AdaptiveCpp 25.10; generic SSCP passed that host's checks through
k=24. This is separate from the current RX 6700 XT k=28 benchmark set.

Cargo preserves two explicit overrides for already validated stacks:
`XCHPLOT2_FORCE_GFX_SPOOF=1` selects the legacy `gfx1013` spoof;
`XCHPLOT2_NO_GFX_SPOOF=1` selects the GPU's actual AOT target, which the
toolchain may reject. An explicit `ACPP_TARGETS` takes precedence.

## Windows

Use WSL2 for `main`. Install the vendor's Windows driver and follow its WSL
GPU setup, then install the Linux toolkit and build dependencies inside the
WSL distro. WSL's injected driver library is not the CUDA Toolkit. The
[install matrix](CONTRIBUTING.md#install-ci) checks WSL builds, but its hosted
runners do not validate GPU execution.

For native Windows on NVIDIA, use the experimental
[`cuda-only` Windows recipe](https://github.com/Jsewill/xchplot2/blob/cuda-only/INSTALL.md#windows).
Native Windows plotting is outside the current hardware test set.

Native Windows SYCL is not supported by the current `main` build. Its
AdaptiveCpp setup and host code require Linux/POSIX facilities; the earlier
unvalidated source-build outline was not a tested installation path.
