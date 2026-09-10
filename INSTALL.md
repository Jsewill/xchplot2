# Installing xchplot2 (`cuda-only`)

[README](README.md) · [Reference](REFERENCE.md)

Run checkout-based commands from the repository directory:

```bash
git clone --branch cuda-only https://github.com/Jsewill/xchplot2
cd xchplot2
```

## Binary archives

Download the Linux or Windows x86-64 CUDA archive and its `.sha256` file from
[GitHub Releases](https://github.com/Jsewill/xchplot2/releases).
On Linux, verify it with `sha256sum -c ARCHIVE.tar.gz.sha256`, replacing `ARCHIVE`
with the downloaded filename without `.tar.gz`. Extract it, then run
`./bin/xchplot2 devices` from the extracted directory. Add that `bin`
directory to `PATH` to use `xchplot2` elsewhere.

The Linux archive requires glibc 2.35+, `libstdc++.so.6` with `GLIBCXX_3.4.30`
(available on updated Ubuntu 22.04), and an x86-64 CPU with AES, SSSE3,
and SSE4.1. It includes native GPU code for
Maxwell through Blackwell and the CUDA runtime; no development toolkit is
needed. Use a compatible NVIDIA driver; 575.57.08+ is recommended for the
pinned CUDA 12.9.1 build. Compiled architecture coverage does not imply that
every card has been tested. Check the release notes for hardware qualification.

On Windows, compare `(Get-FileHash ARCHIVE.zip -Algorithm SHA256).Hash`
with the first field in `ARCHIVE.zip.sha256`, then extract the ZIP and run
`.\bin\xchplot2.exe devices`. See [Windows](#windows) for its requirements.

Each archive includes `BUILDINFO.txt` with source and toolchain versions,
and `licenses/` with dependency notices. Releases without an archive require
a source build using the instructions below.

## Requirements

Requires CUDA Toolkit **12.0+** (12.0 is the floor — `cudaGetDeviceProperties_v2`,
the v2 ABI we link, and CUDA C++20 dialect all need 12.0; the latest benchmark used 13.3.73), **C++20** host compiler, **CMake ≥ 3.26** (3.26+ knows
how to drive nvcc 12.5+; lower works for older nvcc), and a Rust
toolchain from rustup `stable`. The release builds use **Rust 1.98.1**;
older distro-packaged Rust may not support the locked dependencies.

### Historical dependency sources

These package combinations were recorded in the earlier README. They are not
a current verified install matrix; use a toolkit and CMake version compatible
with your GPU and distro. The latest native CUDA benchmark used Ubuntu
24.04.4, CUDA 13.3.73, and NVIDIA driver 610.57.04.

| Distro              | CUDA source                                    | CMake source            | Rust source   |
|---------------------|------------------------------------------------|-------------------------|---------------|
| Ubuntu 24.04        | apt `nvidia-cuda-toolkit` (12.0)               | apt `cmake` (3.28)      | rustup `stable` |
| Ubuntu 24.04        | NVIDIA apt repo `cuda-toolkit-12-9`            | apt `cmake` (3.28)      | rustup `stable` |
| Ubuntu 22.04        | NVIDIA apt repo `cuda-toolkit-12-9`            | Kitware apt `cmake`     | rustup `stable` |
| Debian 12 (Bookworm)| NVIDIA apt repo `cuda-toolkit-12-9`            | Kitware apt `cmake`     | rustup `stable` |
| Fedora 41           | NVIDIA dnf repo `cuda-toolkit-12-9`            | dnf `cmake` (3.30)      | rustup `stable` |
| Rocky / Alma / RHEL 9 | NVIDIA dnf repo `cuda-toolkit-12-9`          | dnf `cmake` (3.26)      | rustup `stable` |
| Arch / CachyOS      | pacman `cuda`                           | pacman `cmake`          | pacman `rust` or rustup |

Combinations that **don't** work on a stock install:
- **Ubuntu 22.04 + apt CUDA**: ships CUDA 11.5 — nvcc too old for the
  C++20 dialect we use, and the v1-ABI `libcudart` lacks
  `cudaGetDeviceProperties_v2`. Use NVIDIA's apt repo instead.
- **Debian 12 + apt CUDA + apt CMake**: stock CMake 3.25 doesn't know
  how to drive nvcc 12.5+. Use Kitware's CMake apt repo.
- **Ubuntu 22.04/24.04 + apt cargo**: distro-packaged Rust (1.75) can't
  parse `edition2024` used by the locked dependency tree.
  Install rustup instead.
- **WSL**: works the same as native — the only WSL-specific bits are
  the `libcuda.so` injection at `/usr/lib/wsl/lib` (driver, not
  runtime). Install the toolkit + rustup inside the WSL distro.

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

## Cargo install

```bash
# rustup, if not already installed (apt/dnf cargo is too old)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

cargo install --git https://github.com/Jsewill/xchplot2 --branch cuda-only --locked
```

The CUDA runtime is statically linked into the binary. Runtime use still
requires a compatible NVIDIA driver.

`build.rs` auto-detects the local GPU's compute capability by querying
`nvidia-smi --query-gpu=compute_cap` and builds for only that
architecture. That keeps the binary small and the build fast when the
install and the target GPU are the same machine.

If auto-detection fails (no `nvidia-smi` in `PATH`, or
`nvidia-smi` can't see a GPU — common when building inside a container
or on a headless build host that lacks the CUDA driver), the build
falls back to `sm_89` on x86_64 or `sm_87` on aarch64. The aarch64 build
searches JetPack and SBSA CUDA library layouts; those platforms are outside
the current hardware benchmark set.

If you need to target a GPU that isn't the one doing the build — or if
you want a single "fat build" binary that covers multiple
architectures — override with `$CUDA_ARCHITECTURES`:

```bash
# Fat build for Ada (4090) and Blackwell (5090):
CUDA_ARCHITECTURES="89;120" cargo install --git https://github.com/Jsewill/xchplot2 --branch cuda-only --locked

# Single target (e.g. Turing 2080 Ti):
CUDA_ARCHITECTURES=75 cargo install --git https://github.com/Jsewill/xchplot2 --branch cuda-only --locked
```

Common values: `52` GTX 9-series (Maxwell, needs a CUDA 12.x toolkit),
`61` GTX 10-series, `70` Volta, `75` Turing, `80` A100, `86` RTX 30-
series, `89` RTX 40-series, `90` H100, `120` RTX 50-series.

## CMake

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

`pos2-chip` is auto-fetched via `FetchContent`; override with
`-DPOS2_CHIP_DIR=/abs/path/to/pos2-chip` to point at a local checkout.

Shared-memory AES round (hot kernels): `-DXCHPLOT2_AES_ROUND=auto`
(default) selects Tezcan 16x-replica on sm_89 when a kernel family has
opted in, and the 4-table path elsewhere. Force either path with
`ttable4` or `tezcan16` (the latter is measured on Ada; results for
sm_75/86/90/120 are welcome).

Outputs:

- `build/tools/xchplot2/xchplot2`
- `build/tools/parity/{aes,xs,t1,t2,t3}_parity` — bit-exact CPU/GPU tests

## Container

The CUDA Toolkit + Rust toolchain live inside the image — the host
only needs an engine plus `nvidia-container-toolkit` for GPU
pass-through. `scripts/install-container-deps.sh` installs both, then
`scripts/build-container.sh` probes `nvidia-smi` for the right
`CUDA_ARCH` and runs `compose build`:

```bash
./scripts/install-container-deps.sh    # one-time: podman + nvidia-container-toolkit + CDI
./scripts/build-container.sh           # auto-pins CUDA 12.9 base on pre-Turing rigs
podman compose run --rm cuda plot -k 28 -n 10 \
    -f <farmer-pk> -c <pool-contract> -o /out
```

Plot files land in `./plots/` on the host. `compose.yaml` uses CDI
shorthand (`devices: - nvidia.com/gpu=all`) so the runtime path is
podman-first; bare `docker run --gpus all` still works after
`install-container-deps.sh --engine docker`, but the `docker compose
run` step won't see the GPU.

## Windows

Native Windows uses the standalone CMake executable. The release ZIP targets
Windows 10 22H2, Windows 11, and Windows Server 2022/2025 on x64. It links
CUDA and the Microsoft C/C++ runtime statically. Runtime use requires a
compatible NVIDIA driver (576.57+ recommended for CUDA 12.9.1) and an
AES/SSSE3/SSE4.1-capable CPU, without installing a development toolkit.

Use NTFS or ReFS for plots, recovery manifests, and spill files. These files
contain private keys; creation requires filesystem support for access control
lists. Select spill storage with `--temp-dir`. The default configuration is
`%APPDATA%\xchplot2\config.toml`. Ctrl-C or Ctrl-Break drains the current
plots; a second signal aborts. Resume with the automatically saved manifest.
Consult the release notes for tested hardware and drivers. Hosted Windows
checks exercise the CPU path; GPU qualification requires Windows hardware.
WSL2 continues to use the Linux instructions.

To build from source, install:

- [Visual Studio 2022](https://visualstudio.microsoft.com/) C++ build tools
  with the **Desktop development with C++** workload and Windows SDK.
- [CUDA Toolkit 12.9.1](https://developer.nvidia.com/cuda-12-9-1-download-archive).
- [Rust](https://www.rust-lang.org/tools/install) with the
  `stable-x86_64-pc-windows-msvc` toolchain.
- [CMake 3.26+](https://cmake.org/download/), Ninja, and
  [Git for Windows](https://gitforwindows.org/).

In the **x64 Native Tools Command Prompt for VS 2022**, run:

```cmd
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel 2
.\build\tools\xchplot2\xchplot2.exe --help
ctest --test-dir build --output-on-failure
```

Change `89` to your GPU architecture from the table above. The full CTest
suite needs a GPU. Use the [release recipe](CONTRIBUTING.md#binary-releases)
for the supported archive build and its host-only checks. Windows Cargo
installation is not supported by this release path.
