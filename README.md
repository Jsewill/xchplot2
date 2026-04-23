# xchplot2

GPU plotter for Chia v2 proofs of space (CHIP-48). Produces farmable
`.plot2` files byte-identical to the
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

> **Status — work in progress.** The plotter produces correct,
> spec-compliant `.plot2` output: per-phase parity tests verify
> byte-identical agreement with pos2-chip's CPU reference at every
> stage, the CUB and SYCL backends produce bit-identical files, and
> determinism holds across runs. The project is still actively under
> development — performance, cross-vendor support (AMD / Intel), and
> the install / CI story are evolving. Expect rough edges; use the
> [`cuda-only`](https://github.com/Jsewill/xchplot2/tree/cuda-only)
> branch if you want the most-tested code path.

> **Branches:** `main` carries the SYCL/AdaptiveCpp port that lets the
> plotter run on AMD and Intel GPUs (with an opt-out CUB sort path
> preserved for NVIDIA). The original CUDA-only implementation, which
> is ~1.5× faster on NVIDIA than the SYCL fallback at k=28, lives on
> the [`cuda-only`](https://github.com/Jsewill/xchplot2/tree/cuda-only)
> branch — use it if you only ever target NVIDIA and want the last
> bit of throughput.

## Hardware compatibility

- **GPU:**
  - **NVIDIA**, compute capability ≥ 6.1 (Pascal / GTX 10-series and
    newer) via the CUDA fast path. Builds auto-detect the installed
    GPU's `compute_cap` via `nvidia-smi`; override with
    `$CUDA_ARCHITECTURES` for fat or cross-target builds (see
    [Build](#build)).
  - **AMD ROCm** via the SYCL / AdaptiveCpp path. Validated on RDNA2
    (`gfx1031`, RX 6700 XT, 12 GB) — bit-exact parity with the CUDA
    backend across the sort / bucket-offsets / g_x kernels, and
    farmable plots end-to-end. ROCm 6.2 required (newer ROCm versions
    have LLVM packaging breakage — see [`compose.yaml`](compose.yaml)
    rocm-service comments). Build picks `ACPP_TARGETS=hip:gfxXXXX`
    from `rocminfo` automatically. Other gfx targets (`gfx1030` /
    `gfx1100`) build cleanly but are untested on real hardware.
  - **Intel oneAPI** is wired up but untested.
- **VRAM:** 8 GB minimum. Cards with less than ~11 GB free
  transparently use the streaming pipeline; 12 GB+ cards reliably use
  the persistent buffer pool for faster steady-state. Both paths
  produce byte-identical plots. Detailed breakdown in [VRAM](#vram).
- **PCIe:** Gen4 x16 or wider recommended. A physically narrower slot
  (e.g. Gen4 x4) adds ~240 ms per plot to the final fragment D2H
  copy; check `cat /sys/bus/pci/devices/*/current_link_width`
  under load if throughput looks off.
- **Host RAM:** ≥ 16 GB recommended; `batch` mode pins ~4 GB of host
  memory for D2H double-buffering (pool or streaming).
- **CUDA Toolkit:** 12+ required for the NVIDIA build path (tested on
  13.x). Skipped automatically on AMD/Intel builds where `nvcc` isn't
  available — `build.rs` runs `nvcc --version` and flips
  `XCHPLOT2_BUILD_CUDA=OFF` when missing. Runtime users on RTX
  50-series (Blackwell, `sm_120`) need a driver bundle that ships
  Toolkit 12.8+; earlier toolkits lack Blackwell codegen.
- **OS:** Linux (tested on modern glibc distributions). Windows and
  macOS are not currently tested.

## Build

Three ways to get the dependencies in place, easiest first:

### 1. Container (`podman compose` or `docker compose`)

Easiest path — let the wrapper detect your GPU and pick the right
compose service automatically:

```bash
./scripts/build-container.sh    # auto: nvidia-smi → cuda, rocminfo → rocm
podman compose run --rm cuda plot -k 28 -n 10 -f <farmer-pk> -c <pool-contract> -o /out
```

[`compose.yaml`](compose.yaml) defines three vendor-specific services
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
```

Plot files land in `./plots/` on the host. The container also bundles
the parity tests (`sycl_sort_parity`, `sycl_g_x_parity`, etc.) under
`/usr/local/bin/` for quick first-port validation on a new GPU:

```bash
podman compose run --rm --entrypoint /usr/local/bin/sycl_sort_parity rocm
```

First build is ~15-30 min (AdaptiveCpp + LLVM 18 compile from source);
subsequent rebuilds reuse the cached layers. GPU performance inside
the container is identical to native (devices pass through via CDI on
NVIDIA, `/dev/kfd`+`/dev/dri` on AMD; kernels run on real hardware).

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
   challenges. `compose.yaml` enforces this — an unset `ACPP_GFX`
   errors out at compose-parse time. Common values
   (`rocminfo | grep gfx` to confirm yours):

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
Pass `--gpu amd` to force the AMD path (CUDA Toolkit headers only,
plus ROCm). Pass `--no-acpp` to skip the AdaptiveCpp build and let
CMake fall back to FetchContent.

### 3. Manual / FetchContent fallback

If you'd rather install dependencies yourself, the toolchain is:

| Dep | Notes |
|---|---|
| **AdaptiveCpp 25.10+** | SYCL implementation. CMake auto-fetches it via FetchContent if `find_package(AdaptiveCpp)` fails — first build adds ~15-30 min. Disable with `-DXCHPLOT2_FETCH_ADAPTIVECPP=OFF` if you want a hard error. |
| **CUDA Toolkit 12+** (headers) | Required on **every** build path because AdaptiveCpp's `half.hpp` includes `cuda_fp16.h`. `nvcc` itself only runs when `XCHPLOT2_BUILD_CUDA=ON` (default; pass `OFF` for AMD/Intel). |
| **LLVM / Clang ≥ 18** | clang + libclang dev packages. |
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

### Lower-level subcommands

```bash
xchplot2 test   <k> <plot-id-hex> [strength] ...    # single plot, raw inputs
xchplot2 batch  <manifest.tsv> [-v] [--skip-existing] [--continue-on-error]
xchplot2 verify <file.plot2> [--trials N]           # run N random challenges
```

`verify` opens a `.plot2` through pos2-chip's CPU prover and runs N
(default 100) random challenges. Zero proofs across a reasonable sample
strongly indicates a corrupt plot; the command exits non-zero in that
case. Intended as a quick sanity check before farming a newly built
batch — not a replacement for `chia plots check`.

## Environment variables

| Variable                      | Effect                                                                  |
|-------------------------------|-------------------------------------------------------------------------|
| `XCHPLOT2_STREAMING=1`        | Force the low-VRAM streaming pipeline even when the pool would fit.     |
| `POS2GPU_MAX_VRAM_MB=N`       | Cap the pool/streaming VRAM query to N MB (exercise streaming fallback).|
| `POS2GPU_STREAMING_STATS=1`   | Log every streaming-path `malloc_device` / `free`.                      |
| `POS2GPU_POOL_DEBUG=1`        | Log pool allocation sizes at construction.                              |
| `POS2GPU_PHASE_TIMING=1`      | Per-phase wall-time breakdown (Xs / sort / T1 / T2 / T3) on stderr.     |
| `ACPP_GFX=gfxXXXX`            | AMD only — required at **build** time; sets AOT target for amdgcn ISA. |
| `ACPP_TARGETS=...`            | Override AdaptiveCpp target selection (defaults: NVIDIA `generic`, AMD `hip:$ACPP_GFX`). |
| `CUDA_ARCHITECTURES=sm_XX`    | Override the CUDA arch autodetected from `nvidia-smi`.                  |
| `POS2_CHIP_DIR=/path`         | Build-time: point at a local pos2-chip checkout instead of FetchContent.|

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

PoS2 plots are k=28 by spec. Two code paths, dispatched automatically
based on available VRAM:

- **Pool path (~11 GB device + ~4 GB pinned host; 12 GB+ cards
  reliably).** The persistent buffer pool is sized worst-case and
  reused across plots in `batch` mode for amortised allocator cost and
  double-buffered D2H. Xs sort's keys_a slot aliases d_storage tail
  (idle during Xs gen+sort), trimming pair_b's worst case from
  `max(cap·12, 4·N·u32 + cub)` to `max(cap·12, 3·N·u32 + cub)` —
  saves ~1 GiB at k=28. Targets: RTX 4090 / 5090, A6000, H100,
  RTX 4080 (16 GB), and 12 GB cards like RTX 3060 / RX 6700 XT.
- **Streaming path (~7.3 GB peak; 8 GB cards with ~500 MB driver /
  compositor headroom).** Allocates per-phase and frees between
  phases; T1/T2 sorts are tiled (N=2 and N=4 respectively) and the
  merge-with-gather is split into three passes so the live set stays
  under 8 GB. Peak at k=28 is **7288 MB** (measured on both sm_89 +
  CUB and gfx1031 + SortSycl — same algebra: T1 sorted 3.12 GB + T2
  match output 4.16 GB, with sort scratch in the tens of MB). Targets
  8 GB cards (GTX 1070 class and up). Slower per plot (~3.7 s vs
  ~2.4 s at k=28 on a 4090) because it pays per-phase
  `malloc_device`/`free` instead of amortising. Log the full alloc
  trace with `POS2GPU_STREAMING_STATS=1`.

At pool construction `xchplot2` queries `cudaMemGetInfo` on the
CUDA-only build, or `global_mem_size` (device total) on the SYCL
path — SYCL has no portable free-memory query, so the check
effectively approximates "free == total" and lets the actual
`malloc_device` failure trigger the fallback. Either way, if the
pool doesn't fit it transparently falls back to the streaming
pipeline with no flag needed. Force streaming on any card with
`XCHPLOT2_STREAMING=1`, useful for testing or for users who want
the smaller peak regardless.

Plot output is bit-identical between the two paths — the streaming
code reorganises memory, not algorithms.

## Performance

k=28, strength=2, RTX 4090 (sm_89), PCIe Gen4 x16. Steady-state per-plot
wall from `xchplot2 batch` (10-plot manifest, mean):

| Build | Per plot | Notes |
|---|---|---|
| pos2-chip CPU baseline | ~50 s | reference |
| `cuda-only` branch | **2.15 s** | original CUDA-only path |
| `main`, `XCHPLOT2_BUILD_CUDA=ON` (CUB sort) | 2.41 s | NVIDIA fast path on the SYCL/AdaptiveCpp port |
| `main`, `XCHPLOT2_BUILD_CUDA=OFF` (hand-rolled SYCL radix) | 3.79 s | cross-vendor fallback (AMD/Intel) on AdaptiveCpp |
| streaming path, ≤8 GB cards | ~3.7 s | pool path is preferred when VRAM allows |
| `main` on RX 6700 XT (gfx1031 / ROCm 6.2 / AdaptiveCpp HIP) | **9.97 s** | AMD batch steady-state at k=28; T-table AES near-optimal on RDNA2 via this compiler stack |

The `main`/CUB row is +12% over `cuda-only` from extra AdaptiveCpp
scheduling overhead. The SYCL row is +57% over CUB on the same NVIDIA
hardware; ~88% of GPU compute is identical between the two paths
(`nsys` per-kernel breakdown), and the gap is dominated by host-side
runtime overhead in AdaptiveCpp's DAG manager rather than kernel
performance. AMD and Intel runtimes are untested; expect roughly the
SYCL-row latency adjusted for relative GPU throughput.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).

## Like this? Send a coin my way!

If you appreciate this, and want to give back, feel free.

xch1d80tfje65xy97fpxg7kl89wugnd6svlv5uag2qays0um5ay5sn0qz8vph8
