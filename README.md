# xchplot2

GPU plotter for Chia v2 proofs of space (CHIP-48). Produces farmable
`.plot2` files byte-identical to the
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

## Quick start

```bash
# Install — needs CUDA Toolkit 12+, CMake ≥ 3.24, a C++20 compiler,
# and Rust. NVIDIA only.
cargo install --git https://github.com/Jsewill/xchplot2 --branch cuda-only

# Plot — 10 × k=28 files, keys derived internally from your BLS pair.
xchplot2 plot -k 28 -n 10 \
    -f <farmer-pk-hex> \
    -c <pool-contract-xch1-or-txch1> \
    -o /mnt/plots

# Multi-GPU — one worker per device, round-robin partition.
xchplot2 plot ... --devices all
```

See [Hardware compatibility](#hardware-compatibility) for GPU / VRAM /
OS requirements, [Build](#build) for alternative install paths, and
[Use](#use) for every flag. **Windows users**: that `cargo install`
line works as-is from an x64 Native Tools Command Prompt for VS 2022
— see [Windows (experimental)](#windows-experimental) for the
prereqs (Windows SDK, `LIB` setup, LNK1181 troubleshooting).

## Hardware compatibility

- **GPU:** NVIDIA, compute capability ≥ 6.1 (Pascal / GTX 10-series
  and newer). Builds auto-detect the installed GPU's `compute_cap`
  via `nvidia-smi`; override with `$CUDA_ARCHITECTURES` for fat or
  cross-target builds (see [Build](#build)).
- **VRAM:** 4 GiB minimum at k=28. Cards with < 15 GB free use the
  streaming pipeline (three sub-tiers — plain ~7.4 GiB, compact
  ~5.3 GiB, minimal ~3.8 GiB — auto-picked by free VRAM); 16 GB+
  cards use the persistent buffer pool for faster steady-state. All
  paths produce byte-identical plots. Detailed breakdown in
  [VRAM](#vram).

  With [`--devices`](#multi-gpu---devices), each worker picks its own
  pool-vs-streaming path from its own GPU's free VRAM — heterogeneous
  rigs (e.g. one 16 GB + one 8 GB card) plot concurrently with each
  device on its matching path.
- **PCIe:** Gen4 x16 or wider recommended. A physically narrower slot
  (e.g. Gen4 x4) adds ~240 ms per plot to the final fragment D2H
  copy; check `cat /sys/bus/pci/devices/*/current_link_width`
  under load if throughput looks off.
- **Host RAM:** ≥ 16 GB recommended; `batch` mode pins ~4 GB of host
  memory for D2H double-buffering (pool or streaming).
- **CUDA Toolkit:** 12+ required to build (tested on 13.x). Runtime
  users on RTX 50-series (Blackwell, `sm_120`) need a driver bundle
  that ships Toolkit 12.8+; earlier toolkits lack Blackwell codegen.
- **CPU architecture:** `x86_64` is the tested path. `aarch64` is also
  supported for NVIDIA ARM platforms — Jetson Orin (`sm_87`), IGX
  Orin, and Grace Hopper / GH200 (`sm_90`, SBSA). `build.rs` picks
  `sm_87` as the aarch64 fallback arch when `nvidia-smi` isn't
  available, and searches the JetPack (`targets/aarch64-linux/lib`)
  and SBSA (`targets/sbsa-linux/lib`) CUDA library layouts. Apple
  Silicon is not supported (no CUDA on macOS).
- **OS:** Linux (tested on modern glibc distributions) is the supported
  path. Windows builds are possible via MSVC + CUDA — see
  [Windows (experimental)](#windows-experimental) below. macOS is not
  supported (no CUDA).

## Build

Requires CUDA Toolkit 12+ (tested on 13.x), C++20 host compiler, CMake
≥ 3.24, and a Rust toolchain (for `keygen-rs`).

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
falls back to `sm_89`.

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

### Container (`podman compose` or `docker compose`)

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

### Windows (experimental)

This branch is CUDA-only, so a Windows build needs nothing beyond the
standard NVIDIA toolchain — no SYCL runtime required. Only one POSIX
site in the code (`Cancel.cpp`) and it's already `#if defined(__unix__)`
-guarded. This path is **untested** — please file an issue with your
results.

Prerequisites:

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
  toolchain (`rustup default stable-x86_64-pc-windows-msvc`)
- [CMake 3.24+](https://cmake.org/download/) and [Git for
  Windows](https://gitforwindows.org/)

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
also works inside the same Native Tools prompt if you prefer that over
`cargo install`.

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

#### Multi-GPU: `--devices`

Both `plot` and `batch` accept `--devices <SPEC>` to fan plots out
across multiple NVIDIA GPUs — one worker thread per device, each
bound via `cudaSetDevice` and carrying its own buffer pool + writer
channel. Plots are partitioned round-robin, so a batch of 10 plots on
2 GPUs sends plots 0/2/4/6/8 to the first GPU and 1/3/5/7/9 to the
second.

```bash
# Every visible CUDA device — enumerated at runtime.
xchplot2 plot --k 28 --num 10 -f <farmer-pk> -c <pool-contract> \
    --out /mnt/plots --devices all

# Only these specific device ids (sorted, deduplicated).
xchplot2 plot ... --devices 0,2,3

# Explicit single id (same as omitting the flag on a single-GPU host).
xchplot2 plot ... --devices 0
```

Omitted flag = single device on the CUDA-default device — identical
to pre-multi-GPU behavior, zero regression risk.

**Caveats for v1:**

- Static round-robin partition. If your GPUs differ in speed the
  batch finishes only as fast as the slowest worker's slice; use
  `--devices` to pick matched cards when that matters.
- Each worker gets its own ~4 GB pinned host pool (pool path) or
  ~6 GB pinned scratch (compact streaming), so host RAM scales
  linearly. A 4-GPU rig pins ~16-24 GB — size accordingly.
- The workers share `stderr` (line-buffered, atomic per-`fprintf`) so
  log lines from different GPUs may interleave.

Smoke test: `scripts/test-multi-gpu.sh` exercises argument parsing
(works on any host, even single-GPU) and, when 2+ GPUs are visible,
runs a live k=22 plot across `--devices 0,1`.

### Lower-level subcommands

```bash
xchplot2 test          <k> <plot-id-hex> [strength] ...   # single plot, raw inputs
xchplot2 batch         <manifest.tsv> [-v] [--devices <SPEC>]
xchplot2 parity-check  [--dir PATH]                       # CPU↔GPU regression screen
```

## Environment variables

| Variable                      | Effect                                                                  |
|-------------------------------|-------------------------------------------------------------------------|
| `XCHPLOT2_STREAMING=1`        | Force the low-VRAM streaming pipeline even when the pool would fit.     |
| `XCHPLOT2_STREAMING_TIER=plain\|compact\|minimal` | Override the streaming-tier auto-pick. Equivalent CLI flag: `--tier`. |
| `POS2GPU_MAX_VRAM_MB=N`       | Cap the VRAM query to N MB — exercises the streaming fallback.          |
| `POS2GPU_STREAMING_STATS=1`   | Log every streaming-path `cudaMalloc` / `cudaFree`.                     |
| `POS2GPU_POOL_DEBUG=1`        | Log pool allocation sizes at construction.                              |
| `POS2GPU_PHASE_TIMING=1`      | Per-phase wall-time breakdown (Xs / sort / T1 / T2 / T3) on stderr.     |
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
src/gpu/                 CUDA kernels — AES, Xs, T1, T2, T3
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
based on available VRAM:

- **Pool path (~15 GB, 16 GB+ cards).** The persistent buffer pool is
  sized worst-case and reused across plots in `batch` mode for
  amortised allocator cost and double-buffered D2H. Targets for
  steady-state: RTX 4080 / 4090 / 5080 / 5090, A6000, etc.
- **Plain streaming (~7.4 GiB floor).** Allocates per-phase and frees
  between phases; no pinned-host parks, single-pass T2 match. Used
  on 10-11 GB cards that can't fit the pool but have headroom above
  compact. ~400 ms/plot faster than compact.
- **Compact streaming (~5.3 GiB floor).** Park/rehydrate of the large
  intermediates on pinned host across their idle windows + N=2 T2
  match staging (cap/2 ≈ 2280 MB at k=28). T1/T2 sorts are tiled
  (N=2 and N=4) with merge trees. Targets 6-8 GiB cards.
- **Minimal streaming (~3.8 GiB floor).** Compact's parks plus N=8
  T2 match staging (cap/8 ≈ 570 MB at k=28). Targets 4 GiB cards
  (GTX 1050 Ti / 1650, RTX 3050 4GB, MX450) at the cost of extra
  PCIe round-trips during T2 match. Floor is estimated; please
  report actual fit on real 4 GiB hardware. There is no smaller
  tier — a forced minimal on a card below the floor throws.

`xchplot2` queries `cudaMemGetInfo` at pool construction; if the
pool doesn't fit, the streaming-tier dispatch picks the largest
streaming tier that fits with a 128 MB margin. Force streaming on
any card with `XCHPLOT2_STREAMING=1`. `--tier
plain|compact|minimal|auto` (or `XCHPLOT2_STREAMING_TIER`) overrides
the auto-pick — useful for testing or to step down from a tight
margin (e.g. an 8 GiB card OOMing mid-plot can `--tier compact`).

Plot output is bit-identical across all paths — streaming
reorganises memory, not algorithms.

## Performance

k=28, strength=2, RTX 4090 (sm_89), PCIe Gen4 x16:

| Mode | Per plot |
|---|---|
| pos2-chip CPU baseline | ~50 s |
| `xchplot2 batch` steady-state wall (pool path) | **2.15 s** |
| `xchplot2 batch` steady-state wall (streaming path, ≤8 GB cards) | ~3.7 s |
| Producer GPU time, steady-state | 1.96 s |
| Device-kernel floor (single-plot nsys) | 1.91 s |

Numbers above are single-GPU. With `--devices 0,1,...` the batch is
partitioned round-robin across N worker threads (one per device), so
wall-clock throughput is bounded by the slowest device's slice —
≈ linear scaling on matched cards, less if cards differ. Live
multi-GPU plots were confirmed end-to-end on NVIDIA.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).
