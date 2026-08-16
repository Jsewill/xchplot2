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

# Multi-GPU — one worker per GPU, pulling from a shared queue.
# (`--devices all` adds the CPU too; `--devices gpu` sticks to GPUs.)
xchplot2 plot ... --devices gpu
```

See [Hardware compatibility](#hardware-compatibility) for GPU / VRAM /
OS requirements, [Build](#build) for alternative install paths, and
[Use](#use) for every flag. **Windows users**: that `cargo install`
line works as-is from an x64 Native Tools Command Prompt for VS 2022
— see [Windows (experimental)](#windows-experimental) for the
prereqs (Windows SDK, `LIB` setup, LNK1181 troubleshooting).

## Hardware compatibility

- **GPU:** NVIDIA, compute capability ≥ 5.0 (Maxwell / GTX 750-class
  and newer). Builds auto-detect the installed GPU's `compute_cap`
  via `nvidia-smi`; override with `$CUDA_ARCHITECTURES` for fat or
  cross-target builds (see [Build](#build)). Pre-sm_53 cards lack
  native FP16 ALUs, but `cuda_fp16.h` falls back to fp32 emulation
  for the half-precision intrinsics — any kernel paths touching
  FP16 still work correctly, with the emulation cost. The AES + match
  kernels at the heart of plotting are integer-only and see no FP16
  penalty.
- **VRAM:** ~1.3 GiB minimum at k=28. The pooled path needs 11612 MB
  free (11484 MB of buffers + a 128 MB margin); it takes another
  2080 MB to also enable the D2H/Xs overlap, which is a speed-up and
  not a requirement. Cards below the pool's floor use the streaming
  pipeline (four sub-tiers, auto-picked by free VRAM). Each streaming
  tier needs its working set plus a 256 MB margin:

  | tier | working set (k=28) | free VRAM needed |
  |------|--------------------|------------------|
  | plain | 7290 MB | 7546 MB |
  | compact | 5200 MB | 5456 MB |
  | minimal | 3640 MB | 3896 MB |
  | tiny | 1064 MB | 1320 MB |

  16 GB+ cards use the persistent buffer pool for faster steady-state.
  All paths produce byte-identical plots. Detailed breakdown in
  [VRAM](#vram).

  With [`--devices`](#multi-gpu---devices), each worker picks its own
  pool-vs-streaming path from its own GPU's free VRAM — heterogeneous
  rigs (e.g. one 16 GB + one 8 GB card) plot concurrently with each
  device on its matching path. The `<id>:<tier>` suffix on `--devices`
  (see [Per-GPU streaming tier](#per-gpu-streaming-tier)) overrides
  the auto-pick per GPU, useful when a card is also serving the
  desktop and needs more headroom than the picker would leave.
- **PCIe:** Gen4 x16 or wider recommended. A physically narrower slot
  (e.g. Gen4 x4) adds ~240 ms per plot to the final fragment D2H
  copy; check `cat /sys/bus/pci/devices/*/current_link_width`
  under load if throughput looks off.
- **Host RAM:** ≥ 16 GB recommended; `batch` mode pins ~4 GB of host
  memory for D2H double-buffering (pool or streaming). The streaming
  tiers need considerably more, and *inversely* to VRAM — `tiny` is the
  hungriest, not the leanest. A host short of RAM for its tier routes
  cold tables to disk and plots anyway; see [Host RAM and
  disk-offload](#host-ram-and-disk-offload).
- **CUDA Toolkit:** 12+ required to build (tested on 13.x). The
  toolkit-vs-arch matrix:
  - `sm_50` – `sm_72` (Maxwell / Pascal / Volta): need CUDA **12.9**
    (last toolkit with codegen for these arches — 13.x dropped them
    entirely). `build.rs` catches the 13.x + old-arch pairing in a
    preflight and points at the fix path.
  - `sm_75` – `sm_90` (Turing / Ampere / Hopper): 12.x or 13.x both
    work.
  - `sm_120` (RTX 50-series Blackwell): need 12.8+; earlier toolkits
    lack Blackwell codegen.
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

Requires CUDA Toolkit **12.0+** (12.0 is the floor — `cudaGetDeviceProperties_v2`,
the v2 ABI we link, and CUDA C++20 dialect all need 12.0; 12.9 is the
newest tested), **C++20** host compiler, **CMake ≥ 3.26** (3.26+ knows
how to drive nvcc 12.5+; lower works for older nvcc), and a Rust
toolchain new enough to parse `edition2024` (**rustc ≥ 1.85**, i.e.
rustup `stable`; most distro-packaged Rust is too old).

### Verified install matrix

| Distro              | CUDA source                                    | CMake source            | Rust source   |
|---------------------|------------------------------------------------|-------------------------|---------------|
| Ubuntu 24.04        | apt `nvidia-cuda-toolkit` (12.0)               | apt `cmake` (3.28)      | rustup `stable` |
| Ubuntu 24.04        | NVIDIA apt repo `cuda-toolkit-12-9`            | apt `cmake` (3.28)      | rustup `stable` |
| Ubuntu 22.04        | NVIDIA apt repo `cuda-toolkit-12-9`            | Kitware apt `cmake`     | rustup `stable` |
| Debian 12 (Bookworm)| NVIDIA apt repo `cuda-toolkit-12-9`            | Kitware apt `cmake`     | rustup `stable` |
| Fedora 41           | NVIDIA dnf repo `cuda-toolkit-12-9`            | dnf `cmake` (3.30)      | rustup `stable` |
| Rocky / Alma / RHEL 9 | NVIDIA dnf repo `cuda-toolkit-12-9`          | dnf `cmake` (3.26)      | rustup `stable` |
| Arch / CachyOS      | pacman `cuda` (12.x)                           | pacman `cmake`          | pacman `rust` or rustup |

Combinations that **don't** work on a stock install:
- **Ubuntu 22.04 + apt CUDA**: ships CUDA 11.5 — nvcc too old for the
  C++20 dialect we use, and the v1-ABI `libcudart` lacks
  `cudaGetDeviceProperties_v2`. Use NVIDIA's apt repo instead.
- **Debian 12 + apt CUDA + apt CMake**: stock CMake 3.25 doesn't know
  how to drive nvcc 12.5+. Use Kitware's CMake apt repo.
- **Ubuntu 22.04/24.04 + apt cargo**: distro-packaged Rust (1.75) can't
  parse `edition2024` required by the `chia-client` 0.42 dep tree.
  Install rustup instead.
- **WSL**: works the same as native — the only WSL-specific bits are
  the `libcuda.so` injection at `/usr/lib/wsl/lib` (driver, not
  runtime). Install the toolkit + rustup inside the WSL distro.

### `cargo install`

```bash
# rustup, if not already installed (apt/dnf cargo is too old)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

cargo install --git https://github.com/Jsewill/xchplot2
```

The CUDA runtime is statically linked into the binary, so users don't
need any `libcudart.so` version pinning at runtime, and there's no
class of "wrong libcudart on linker path" install failures regardless
of how mixed the user's previous CUDA installs are. The binary is ~1 MB
larger (3.8 MB vs 2.8 MB) for that property.

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

Common values: `52` GTX 9-series (Maxwell, needs CUDA 12.9 toolkit),
`61` GTX 10-series, `70` Volta, `75` Turing, `80` A100, `86` RTX 30-
series, `89` RTX 40-series, `90` H100, `120` RTX 50-series.

### CMake (also builds the parity tests)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
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

On a host that is short of RAM for the tier its GPU lands on, add
`--temp-dir <path>` to choose where the automatic disk-offload writes
(`--max-host-ram` to bound it explicitly, `--no-auto-spill` to turn it
off) — see [Host RAM and disk-offload](#host-ram-and-disk-offload).

#### Per-worker rates and the batch size that lands them together

Any multi-worker run — `bench`, `plot`, or `batch` — reports what each
worker actually did, because on a shared queue the batch average is
nobody's rate. On a TTY the progress display carries a line per worker as
it goes (the bars are each worker's share of the batch so far, so on a
mixed rig they are *supposed* to come out lopsided); redirected to a log
it stays one line, and the same breakdown prints once at the end.

```
[batch] progress: plot 5/6 done (83.3%, 0.37 s/plot avg, 1.6e-06 TiB/s, batch ETA ~0s)
[batch]   gpu0     34 plots     8.79 s/plot  ##########
[batch]   cpu0#0    5 plots    63.10 s/plot  #---------

[batch] per-worker:
[batch]   gpu0   34 plots — 8.79 s/plot
[batch]   cpu0#0  5 plots — 63.10 s/plot — then idle 4.2 s waiting on a peer
[batch]   optimal batch: multiples of 8 — gpu0 7, cpu0#0 1 — every worker lands within 1.47 s (2.3% idle)
[batch]     for a near-exact landing use multiples of 49 — gpu0 43, cpu0#0 6 — 0.01% idle
```

The **optimal batch** line answers "how many plots should I ask for so
nobody waits". The queue splits a batch in proportion to the workers'
rates without being told them, but it cannot split a *plot*: beside a
63 s/plot CPU, a 8.79 s/plot GPU is worth 7.17 of it, so the CPU's fair
share of 10 plots is 1.22 — and it must be handed 1 or 2. Either way
somebody waits. At some sizes the fair shares land on whole plots and
nobody does; those are the sizes worth plotting in, and any multiple of
one works.

Two are offered because they trade off. The first is the smallest size
that wastes little. The second is the smallest that lands near-exactly —
worth knowing because the first's error is a *fraction*, so doubling the
batch doubles the seconds lost (8 plots idles 1.47 s; 16 idles 2.94 s),
while the near-exact size stays tight however many you run.

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

`xchplot2 devices` prints id, name, VRAM, SM count, and compute
capability for every visible CUDA device, plus the host CPU plotter
row. Use the printed `[gpu0]` / `[cpu0]` index with `--devices`:

```
$ xchplot2 devices
Visible devices (1 GPU + 1 CPU node):
  [0]    NVIDIA GeForce RTX 4090          vram=24076 MB  SMs=128  CC=8.9
  [cpu0] Host CPU plotter                 threads=32                         (1-2 orders slower than GPU)
```

A multi-socket host prints one `[cpuN]` row per NUMA node, and each is
separately selectable.

Both `plot` and `batch` accept `--devices <SPEC>` to fan plots out
across multiple NVIDIA GPUs — one worker thread per device, each
bound via `cudaSetDevice` and carrying its own buffer pool + writer
channel. Plots are not partitioned up front: the workers race for the
next one off a shared queue, so each takes work in proportion to its own
speed. A GPU twice as fast as its neighbour simply finishes about twice
as many.

```bash
# Every visible CUDA device — enumerated at runtime. No CPU worker.
xchplot2 plot --k 28 --num 10 -f <farmer-pk> -c <pool-contract> \
    --out /mnt/plots --devices gpu

# Every CUDA device PLUS a CPU worker on the same batch.
xchplot2 plot ... --devices all

# Only these specific device ids (sorted, deduplicated).
xchplot2 plot ... --devices 0,2,3

# Explicit single id (same as omitting the flag on a single-GPU host).
xchplot2 plot ... --devices 0

# CPU only, or specific GPUs + CPU as a list.
xchplot2 plot ... --devices cpu
xchplot2 plot ... --devices 0,1,cpu
xchplot2 plot ... --devices gpu0,gpu1,cpu   # same thing, spelled out

# One NUMA node's worth of CPU only (multi-socket hosts).
xchplot2 plot ... --devices cpu1
```

##### CPU plotting is opt-in: `--cpu` / `--cpu-workers`

`--devices` names **devices**; `--cpu-workers` says how many plots run on each
of them. So `cpu` means every CPU node the way `gpu` means every GPU, `cpu0`
names one node the way `gpu0` names one card, and repeats dedup — `cpu,cpu` is
just `cpu`, exactly as `0,0` is `0`.

Once the CPU is in, its plots run alongside whatever GPUs are selected, niced
below them. The CPU plotter goes through pos2-chip's `Plotter` (no CUDA calls)
and is memory-latency-bound, so concurrent plots interleave each other's stalls
instead of queueing for a core — a handful is free throughput **when the CPU
plots alone**.

Measured, aggregate steady-state, 32-thread 5950X, same binary:

| workers | k=26 | k=28 |
|---|---|---|
| `N=1` | 13.57 s/plot | 52.28 s/plot |
| `N=2` | 10.59 s/plot (+28%) | 43.85 s/plot (+19%) |
| `N=4` | 9.63 s/plot (+41%) | 41.69 s/plot (+25%) |

Beside a GPU those are the wrong curve: they are `--devices cpu` figures, so
they see what extra workers ADD but not what they COST the GPU worker's
host-side FSE consumer. On an RTX 4090 at k=28 the knee of 4 ran a 55-plot
batch **2.39x slower than the GPU alone** (the GPU's own rate fell 2.56 → 4.23
s/plot); one worker was a wash. So `auto` picks **1 per node** beside a GPU. A
slower card flips this — its consumer has ten times as long for the same work
— so a slow-GPU host may want more, which `--cpu-workers N` sets, or
`XCHPLOT2_CPU_ADAPTIVE=1` picks automatically from the GPU's measured rate.

Either way, a slow worker no longer drags out the tail of a run. As the queue
drains, each worker stands down rather than start a plot the faster workers would
finish the whole remainder before — so a CPU worker beside a fast 4090 stops
pulling with ~16-20 plots left (fewer as the card slows, since each GPU plot
takes longer), and does not run at all on a batch too short to help. The same
guard covers a **slow GPU beside a fast one**: it is worker-general, not a CPU
special case. It can only ever decline work a worker would finish last on, so it
never lengthens a run, and a uniform fleet (nobody strictly faster) is left
untouched. `XCHPLOT2_TAIL_GUARD=0` turns it off.

```bash
# Default: no CPU. Zero-config is the GPU alone.
xchplot2 plot ...

# Opt in:
xchplot2 plot ... --cpu                # GPU(s) + CPU, without naming --devices
xchplot2 plot ... --devices all        # every GPU + every CPU node
xchplot2 plot ... --devices cpu        # CPU only

# Tune the count (naming one also opts the CPU in):
xchplot2 plot ... --cpu-workers 2      # exactly 2 per node
xchplot2 plot ... --cpu-workers max    # as many as fit, capped at core count
xchplot2 plot ... --cpu-workers auto   # the knee (1 beside a GPU) — the default once selected
xchplot2 plot ... --devices all --cpu-workers 0   # ...never mind, no CPU
```

On a **multi-socket host** each CPU worker is pinned to one NUMA node, so its
working set is allocated and read node-locally rather than across the
interconnect — which matters precisely because the plotter is
memory-latency-bound. `--cpu-workers N` is therefore *per node*: on the
single-socket hosts these numbers were tuned on, per-node and per-host are the
same thing.

Note that pinning buys **locality, not a thread cap**: pos2-chip sizes its
fan-out from `hardware_concurrency()`, which reports the whole host whatever the
affinity mask says, so a node-pinned worker still oversubscribes its node. The
~4 knee below was measured on a single socket, where a node *is* the machine; on
a multi-socket box expect the real knee to be lower and measure it
(`--cpu-workers N`) rather than trusting the default.

**Checking that the pin took.** A `cpuN#M` label says which node a worker was
*asked* for, not where it ended up, and a successful pin is otherwise silent. `-v`
prints each worker's mask read back out of the kernel:

```
[cpu1#0] pinned to NUMA node 1 (cpus 32-63)
```

Thread count is not the signal — the fan-out ignores the mask (above), so a
correctly pinned worker still shows the whole host's worth of threads. If you want
to confirm the mask reached pos2-chip's threads rather than just the worker that
set it, every task should report its node's cpus:

```bash
grep -h Cpus_allowed_list /proc/$(pgrep -n xchplot2)/task/*/status | sort | uniq -c
```

One bucket on a multi-socket box means nothing is pinned; one bucket per node in
use means it is. To measure what the pin is *worth*, `XCHPLOT2_CPU_NO_PIN=1` keeps
the same worker layout and skips only the pin — `--devices cpu0` is not that A/B,
since it also halves the cores available.

The auto count is the **knee of the throughput curve** (~4), then trimmed to what
host RAM holds — not "as many as fit". More than the knee only oversubscribes:
each worker already fans out to every core, so the gain plateaus fast (at k=28
the 3rd and 4th together buy ~5%), and at small k a pure RAM cap would spawn
hundreds of workers (k=22 permits ~500) for no benefit. `XCHPLOT2_CPU_AUTO_WORKERS`
overrides the knee.

**Each worker needs its own full copy of the plotter's working set** — 12.1 GiB
at k=28, 3.1 GiB at k=26 — because no streaming tier applies to the CPU path.
So `N` is an ask, not a promise: it is capped at what host RAM actually holds,
and you are told when it caps you:

```
[batch] cpu: asked for 6 CPU workers but only 4 fit: each needs 12.1 GiB at
        k=28, host has 61.4 GiB available and 2 GPU workers reserve 10.5 GiB of it
```

The gate is **re-asked at every plot boundary**, not just at startup. A batch runs
for hours, and the box it runs on is usually a machine you also use: if a compile
or a browser takes 40 GB an hour in, the count decided at t=0 is now a lie, and the
OOM killer is what notices — taking the GPU workers' in-flight plots down with the
CPU's. A plot boundary is the one place the question can be asked honestly, because
between plots a CPU worker holds nothing.

It cannot *shrink* a running plot — the actuator is a whole 52-second plot and the
OOM killer acts in milliseconds — so this is admission control, not a live
controller. Workers wait for room, and say so:

```
[batch:cpu#1] waiting for memory: its next plot needs 12.1 GiB, the host has
              3.2 GiB to spare (2 peers plotting — each one finishing frees 12.1 GiB)
[batch:cpu#1] resumed — memory came back
```

If the box stays too small for a full grace period (`XCHPLOT2_CPU_WAIT_SECS`,
default 300 s) with no peer of ours holding anything it could give back, the CPU
worker retires — quietly, if there are GPU workers still draining the queue.

| env var | effect |
|---|---|
| `XCHPLOT2_CPU_RESERVE_MB` | host RAM to leave alone — for the machine you also work on |
| `XCHPLOT2_CPU_WAIT_SECS` | how long a worker waits for memory before retiring (default 300) |
| `XCHPLOT2_CPU_WORKERS_UNGATED=1` | skip the gate entirely (and risk the OOM killer taking the whole batch) |
| `XCHPLOT2_CPU_NICE` | how far to de-prioritise CPU workers below GPU workers (default 10; `0` disables) |

The CPU workers are niced below the GPU workers when both are present, because
pos2-chip fans out to every core and would otherwise starve the GPU workers'
compression threads — costing more GPU throughput than the CPU adds.

##### Per-GPU streaming tier

Any GPU selector in `--devices` accepts a `:tier` suffix to pin the
streaming tier for that device. Tier ∈ `plain|compact|minimal|tiny|auto`.
Useful when GPUs differ in VRAM, or when one card is also serving
the desktop and you want to leave it more headroom:

```bash
# All GPUs auto-pick from free VRAM, except GPU 2 which uses tiny.
xchplot2 plot ... --devices gpu,2:tiny

# All GPUs + CPU worker; GPU 2 = tiny.
xchplot2 plot ... --devices all,2:tiny

# All GPUs pinned to tiny, except GPU 2 which uses plain.
xchplot2 plot ... --devices gpu:tiny,2:plain

# All GPUs pinned to tiny, except GPU 2 which auto-picks (the `:auto`
# sentinel explicitly re-enables auto-pick for a single GPU).
xchplot2 plot ... --devices gpu:tiny,2:auto

# All-explicit form (still works).
xchplot2 plot ... --devices 0:tiny,1:minimal,2:plain
```

Precedence (highest wins):
1. Per-GPU `<id>:<tier>` token
2. `gpu:<tier>` / `all:<tier>` shorthand
3. Global `--tier <name>` / `XCHPLOT2_STREAMING_TIER`
4. Auto-pick from free VRAM

`cpu:<tier>` is rejected (the CPU worker doesn't use streaming
tiers). Duplicate IDs with conflicting tiers (`0:tiny,0:plain`) and
unknown tier names are also rejected at parse time.

Omitted flag = single device on the CUDA-default device — identical
to pre-multi-GPU behavior, zero regression risk.

**Caveats for v1:**

- Mismatched devices still leave a tail. The shared queue keeps every
  worker fed, so speed differences cost nothing until it runs dry —
  but the last plots are held by specific workers at their own rates,
  and a fast card can idle while a slow one finishes. `bench` prints
  the batch sizes that land everyone together; see "Per-worker rates".
- Each worker gets its own ~4 GB pinned host pool (pool path) or
  ~6 GB pinned scratch (compact streaming), so host RAM scales
  linearly. A 4-GPU rig pins ~16-24 GB — size accordingly.
- The workers share `stderr` (line-buffered, atomic per-`fprintf`) so
  log lines from different GPUs may interleave.

Smoke test: `scripts/test-multi-gpu.sh` exercises argument parsing
(works on any host, even single-GPU) and, when 2+ GPUs are visible,
runs a live k=22 plot across `--devices 0,1`.

### Benchmark throughput

`bench` measures how fast your hardware plots by writing synthetic
unfarmable `.plot2` files (random plot_ids, no keys), then reports
steady-state throughput in TiB/s, TiB/hour, TiB/day, and TiB/month
(30-day basis):

```bash
# Quick smoke (k=18 finishes in seconds on most GPUs)
xchplot2 bench -k 18 -n 3 -o /tmp

# Full measurement at k=28 (default: 1 warmup + 10 measured plots/worker)
xchplot2 bench -k 28 -o /scratch

# Also run a tmpfs pass to isolate compute from disk I/O
xchplot2 bench -k 28 -o /scratch --compute-only
```

Bench deletes the files it creates unless `--keep` is set. Pass
`--target-size TiB` to estimate time-to-fill a specific capacity instead
of the output directory's free space.

### Lower-level subcommands

```bash
xchplot2 test          <k> <plot-id-hex> [strength] ...   # single plot, raw inputs
xchplot2 batch         <manifest.tsv> [-v] [--devices <SPEC>]
xchplot2 bench         [-k K] [-n N] [-o DIR] [--devices <SPEC>] [--compute-only]
xchplot2 parity-check  [--dir PATH]                       # CPU↔GPU regression screen
```

## Environment variables

| Variable                      | Effect                                                                  |
|-------------------------------|-------------------------------------------------------------------------|
| `XCHPLOT2_STREAMING=1`        | Force the low-VRAM streaming pipeline even when the pool would fit.     |
| `XCHPLOT2_STREAMING_TIER=plain\|compact\|minimal\|tiny` | Override the streaming-tier auto-pick (k=28 working sets / free-VRAM floors: plain 7290/7546 MB, compact 5200/5456, minimal 3640/3896, tiny 1064/1320). Equivalent CLI flag: `--tier`. Either form forces the streaming pipeline even on cards big enough to fit the pool, so `--tier tiny` works on a 4090 too. |
| `XCHPLOT2_MAX_HOST_RAM=8G\|min` | Cap the streaming path's unswappable host peak by routing its cold tables to the temp dir. Equivalent CLI flag: `--max-host-ram`, which wins if both are set. See [Host RAM and disk-offload](#host-ram-and-disk-offload). |
| `XCHPLOT2_TEMP_DIR=/path`     | Where routed tables live. Equivalent CLI flag: `--temp-dir`. Must be real disk — a RAM-backed dir is refused, since spilling there consumes the RAM the budget exists to cap. |
| `XCHPLOT2_NO_AUTO_SPILL=1`    | Refuse to plot when the tier does not fit host RAM, instead of routing tables automatically. Equivalent CLI flag: `--no-auto-spill`. |
| `XCHPLOT2_ALLOW_RAM_TEMP_DIR=1` | Downgrade the RAM-backed-temp-dir refusal to a warning, for the rare disk-backed `/tmp`. The reported host-RAM budget does not account for what the spill then writes into RAM. |
| `XCHPLOT2_DRAIN_SLOTS=N`      | Pin the D2H drain slot count (1..3) instead of letting the host-RAM policy choose. Fewer slots cost producer/consumer overlap across plots. |
| `POS2GPU_MAX_VRAM_MB=N`       | Cap the VRAM query to N MB — exercises the streaming fallback. Only caps what the *picker* sees; real allocation still succeeds on a big card, so it cannot validate that a tier fits. To rehearse a smaller card for real, hold the VRAM with a ballast process. |
| `POS2GPU_VRAM_MARGIN_MB=N`    | Free VRAM the pool's gate leaves unclaimed. Default 128 MB. This is headroom against *other tenants* on the card, not an allowance for our own unmodelled allocations (the pooled path has none) — raise it if the GPU also drives a desktop, leave it alone on a headless rig. |
| `POS2GPU_STREAMING_STATS=1`   | Log every streaming-path allocation, plus the CUDA memory pool's physical high-water and the plot's VRAM budget. The pool reserves more than it hands out — size a tier from the physical number, not the logical one. |
| `POS2GPU_ASSERT_VRAM=1`       | Fail a plot if a streaming tier's working set outgrows the peak its floor is derived from, or if the pooled path exceeds the buffers it declared. Armed by `bench`. |
| `POS2GPU_POOL_CACHE_MB=N`     | Override how much the CUDA memory pool may keep cached. Default: whatever the card has spare beyond the tier's working set — generous on a big card (cross-plot reuse), near zero on a card at its floor. Only for measurement. |
| `POS2GPU_POOL_DEBUG=1`        | Log pool allocation sizes at construction.                              |
| `POS2GPU_PHASE_TIMING=1`      | Per-phase wall-time breakdown (Xs / sort / T1 / T2 / T3) on stderr.     |
| `POS2GPU_NO_ASYNC_ALLOC=1`    | Disable stream-ordered `cudaMallocAsync` pooling (audit kill switch).   |
| `POS2GPU_TINY_OVERLAP=1`      | Opt in to tiny-tier T1-match double-buffer (off by default; regresses on Ada). |
| `POS2GPU_NO_TINY_OVERLAP=1`   | Force single-stream tiny T1 match even if `POS2GPU_TINY_OVERLAP=1`.     |
| `POS2GPU_NO_D2H_OVERLAP=1`    | Disable pool-path final-fragment D2H overlap with next plot's Xs.       |
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

- **Pool path (11612 MB floor; 12 GB+ cards).** The persistent buffer
  pool is sized worst-case and reused across plots in `batch` mode for
  amortised allocator cost and double-buffered D2H. 11484 MB of buffers
  + a 128 MB margin. Above 13692 MB free it also takes a dedicated 2080 MB
  fragment buffer to overlap the final D2H with the next plot's Xs phase —
  a speed-up bought with spare VRAM, never a requirement, so a card at the
  floor simply aliases it. Targets for steady-state: RTX 4080 / 4090 /
  5080 / 5090, A6000, etc.

  The floor was 11996 MB until the gate's margin was corrected from 512 MB
  to 128 MB. That 512 double-counted the ~390 MB CUDA context, which
  `cudaMemGetInfo` has already deducted from the free figure it reports —
  and it cost exactly the 12 GB cards this path was documented to serve.
  Measured: the pooled path allocates nothing at runtime, so those buffers
  are its whole device footprint, and the driver puts it 18 MB over them.
Every streaming floor below is that tier's measured working set at k=28
plus a 256 MB margin. The margin covers the 128 MB the streaming
allocator holds back for the CUDA context's growth, plus the allocator's
own granularity surplus (~30-70 MB).

A note on how these are derived, because it is easy to get wrong: the
per-plot VRAM trace (`POS2GPU_STREAMING_STATS=1`) counts the bytes the
pipeline *asked for*. The CUDA memory pool **reserves more than it hands
out** — it once sat on 3.2 GB past the working set on plain and compact —
and that reservation is what has to fit in the card. The trace now prints
the pool's physical high-water and the budget next to the logical peak.
Re-derive a floor from those, never from the logical peak alone.
`bench` fails loudly (`POS2GPU_ASSERT_VRAM`) if a tier outgrows the peak
its floor was derived from.

- **Plain streaming (7546 MB floor; 7290 MB working set).** Allocates
  per-phase and frees between phases; no pinned-host parks, single-pass
  T2 match. Used on 8-11 GB cards that can't fit the pool but have
  headroom above compact. ~400 ms/plot faster than compact.
- **Compact streaming (5456 MB floor; 5200 MB working set).** Park/rehydrate of the large
  intermediates on pinned host across their idle windows + N=2 T2
  match staging (cap/2 ≈ 2280 MB at k=28). T1/T2 sorts are tiled
  (N=2 and N=4) with merge trees. Targets 6-8 GiB cards.
- **Minimal streaming (3896 MB floor; 3640 MB working set).** Compact's parks plus
  six layered cuts that bring every phase below the 4 GiB cliff:
  (1) N=8 T2 match staging (cap/8 ≈ 570 MB at k=28); (2) N=4
  T1/T2 sort gather tiling — the merged-key + permuted-meta
  gather output is D2H'd per tile to pinned host; (3) T3 match
  section-pair input slicing — d_t2_meta_sorted is parked on
  pinned host across T3 match, with the section_l + section_r
  row slices H2D'd per pass to a cap/2 device buffer (xbits +
  keys stay full-cap for binary-search reads); (4) N=4 T1 match
  slicing — each section_l pass writes to cap/4 device staging,
  D2H to pinned host; (5) CUB sub-phase tiling in T1/T2/T3 sort
  — replaces the four cap-sized uint32/uint64 sort I/O buffers
  with cap/N per-tile staging + host pinned accumulators, with
  the multi-way merge done on the CPU; and (6) Xs gen+sort+pack
  tiling — generate the full (keys, vals) once, then sort in
  cap/N tiles to host pinned accumulators (carved out of
  scratch.h_meta), CPU-merge, and pack into d_xs via two strided
  cudaMemcpy2DAsync H2D copies (no separate device-side pack
  buffer pair).
  Measured overall peak at k=28 strength=2 on RTX 4090 (compact
  → minimal): 5200 → 3640 MB; per-phase peaks: Xs 2570, T1 sort
  3640, T2 sort 3640, T3 match 3640, T3 sort 3640. Targets 4 GiB
  cards (GTX 1050 Ti / 1650, RTX 3050 4GB, MX450) and fits
  comfortably on 5 GiB+ cards with ~2 GiB headroom. Trade-off:
  ~6 extra cap-sized PCIe round-trips per plot + ~6 sec/plot of
  host-CPU merge work — k=28 wall on sm_89: ~31 s/plot vs ~12 s
  for compact (~2.6×). 4 GiB cards remain an edge case since
  real 4 GiB hardware reports ~3.5 GiB free post-CUDA-context;
  please report actual fit.
- **Tiny streaming (1320 MB floor; 1064 MB working set).** Full Phase 1.4 + 1.5 + 1.6
  algorithm port, byte-for-byte peak parity with the SYCL Tiny
  tier. On top of Minimal's six cuts, adds: per-section-pair T1
  match tile (Xs data parks on pinned host h_xs_pinned; T1 reads
  via per-(L,R) section H2D), per-(section_l, match_key_r)
  bucket-pair sub-section for T1/T2/T3 match (per-pass tile is L
  section + one R bucket instead of full L+R), streaming-partition
  T1 sort + streaming-partition T2 sort with global_idx tiebreak +
  tile-and-merge T3 sort (eliminates the cap-sized d_t1_meta,
  d_t2_mi, and d_t3 on device by partitioning to per-bucket arenas
  + per-bucket CUB sort), host-side T2/T3 prepare offsets (binary
  search on already-sorted h_keys_merged, skipping the cap-sized
  GPU prepare-keys H2D), d_t3_stage + d_frags_out → host-pinned
  aliases (T3 match writes via UVA-mapped host pinned; T3 sort
  lands sorted fragments directly in pinned_dst), and Xs gen+sort
  per-tile generation via launch_xs_gen_range (eliminates the
  cap × 2 × u32 full-cap gen output that the non-range path
  requires). Targets sub-2 GiB NVIDIA cards (Quadro P620 2 GB,
  GTX 1050 2 GB, older laptop dGPUs).
  Measured at k=28 strength=2 on RTX 4090: **1064 MB plot peak —
  byte-identical to SYCL Tiny's measured 1064 MB**. Per-phase
  peaks: Xs 1030, T1 match 1040, T1 sort 1056, T2 match 1040,
  T2 sort 1064 (floor), T3 match 1024, T3 sort 1047. All phases
  ≤ 1064 MB. Trade-off: ~17 s/plot extra wall vs minimal
  (per-bucket sequential gen+sort+pack+merge) — k=28 wall on
  sm_89 ~50 s/plot. Byte-identical to other tiers at
  k=22/24/26/28 (validated). There is no smaller tier — a forced
  tiny on a card below the floor throws.

`xchplot2` queries `cudaMemGetInfo` at pool construction; if the
pool doesn't fit, the streaming-tier dispatch picks the largest
streaming tier that fits with a 256 MB margin. Force streaming on
any card with `XCHPLOT2_STREAMING=1`. `--tier
plain|compact|minimal|tiny|auto` (or `XCHPLOT2_STREAMING_TIER`)
overrides the auto-pick — useful for testing or to step down from
a tight margin (e.g. an 8 GiB card OOMing mid-plot can
`--tier compact`).

The two margins are different numbers on purpose. The pool's gate leaves
128 MB (`POS2GPU_VRAM_MARGIN_MB`) — pure headroom against other tenants on
the card, since the pooled path allocates nothing beyond the buffers it
sizes up front. A streaming tier leaves 256 MB, because it runs under the
budgeted allocator, which holds 128 MB back from its own budget: a floor of
`working set + 256` yields a working budget of `working set + 128`, and that
128 is what absorbs allocation granularity. Cut the streaming margin to 128
and the budget collapses to exactly the working set, with nothing left to
round up into.

Plot output is bit-identical across all paths — streaming
reorganises memory, not algorithms.

## Host RAM and disk-offload

The streaming tiers buy VRAM **with** host RAM, so the ladder runs
backwards from what you would expect: `plain` needs the least host RAM
and `tiny` the most. A small card therefore fails in a way that has
nothing to do with the card — the tier fits the GPU, and then the box
runs out of RAM.

xchplot2 models each tier's host peak and checks it before allocating
anything. When the tier does not fit, it **spills the cold tables to a
temp dir and plots anyway** — the alternative is refusing outright, so
this is on by default. It cannot slow down a run that already works:
nothing spills unless the run would otherwise have been rejected.

```bash
# Nothing to do — this is the default. A host that is short on RAM
# spills, announces it, and plots.
xchplot2 plot -k 28 -n 10 -f <farmer-pk> -c <pool-address>

# Put the spill somewhere specific — real disk, and fast (see below).
xchplot2 plot ... --temp-dir /mnt/nvme/xchplot2-spill

# Cap the unswappable host peak yourself. Accepts 8G / 8GiB / 8192M /
# raw bytes, or `min` for "route everything this tier can".
xchplot2 plot ... --max-host-ram 8G
xchplot2 plot ... --max-host-ram min

# Prefer a clear refusal over a slower plot.
xchplot2 plot ... --no-auto-spill
```

All three work on `plot`, `batch`, and `bench`. `XCHPLOT2_MAX_HOST_RAM`
and `XCHPLOT2_NO_AUTO_SPILL=1` are env equivalents; the flag wins.

What gets routed, in order, largest first — and only as far as the
budget requires:

1. `h_meta`, the 8-byte cap-sized table. It is **three** lifetime-disjoint
   roles (the T1 meta park, the T2 meta park, and the T3 pairing
   accumulator), each given its own buffer, so it crosses the temp dir
   three times per direction per plot rather than once;
2. `h_t2_xbits`, the 4-byte table — written during T2 match, read back
   once before the T2 sort gather;
3. the D2H drain slots, 3 → 1, **last**. Routing a table costs disk I/O
   per plot; a drain slot costs producer/consumer overlap across plots,
   which is the more expensive of the two in a batch.

Measured at k=28 compact on an RTX 4090, one plot per rung — every rung
produces a byte-identical plot (`sha256[:20] = 9e020867acd59d31164d`):

| routed | host peak |
|---|---:|
| nothing | 11.29 GiB |
| `h_meta` | 9.32 GiB |
| + `h_t2_xbits` | 8.30 GiB |
| + drain 3 → 1 (`--max-host-ram min`) | **4.24 GiB** |

That is **2.66× less host RAM for the same plot**. The wall cost at the
bottom rung, same card, temp dir on NVMe: **9.71 → 13.94 s/plot, ~44%**.
4.24 GiB is the floor of this mechanism at k=28 — below it the remaining
buffers are ones a GPU kernel writes directly through a device-visible
pointer, which cannot live in a file.

**Tier support is not uniform**, because what a tier does to a table
decides whether the table can leave RAM at all:

| tier | how tables are routed | k=28 peak, `min` |
|---|---|---:|
| `plain` | drain slots only — it parks nothing | — |
| `compact` | both tables through the spill engine, to disk | 12.44 → 5.33 GiB |
| `minimal` | both tables as file-backed **mappings** | 13.46 → 6.35 GiB |
| `tiny` | drain slots only — no table can leave RAM | 14.47 → 10.41 GiB |

`minimal` CPU-touches both tables, so they cannot go through the engine
— but a `MAP_SHARED` mapping serves CPU indexing and `cudaMemcpyAsync`
equally well. Those bytes leave the *unswappable* class (the kernel can
write them back and evict them under pressure) but stay resident while
there is no pressure, so they are reported as `reclaimable` (3.05 GiB at
k=28) and carry no modelled temp-dir traffic. `tiny` hands the same
buffers to kernels as USM-host, which a mapping cannot be, so only the
drain-slot lever is available to it — which is the tier that needs the
help most, and the honest answer is that this mechanism does not reach
it.

Notes:

- **The temp dir must be real disk.** `/tmp` is tmpfs on most systemd
  distributions, i.e. RAM — spilling there consumes the memory the budget
  exists to bound, and it defeats the mapping route just as thoroughly
  (a `MAP_SHARED` mapping over a tmpfs file is anonymous memory with
  extra steps). xchplot2 refuses a RAM-backed temp dir: an explicit
  `--max-host-ram` throws, and the automatic rescue stands down and says
  so in the out-of-RAM message rather than trading one confusing error
  for another. Override with `XCHPLOT2_ALLOW_RAM_TEMP_DIR=1` for the rare
  disk-backed `/tmp`. `--temp-dir` is also checked up front — exists,
  writable, and usable — so a mistyped path fails before the batch starts
  instead of minutes in.
- **Budget ~7.1 GiB of free space** for compact at k=28: three `h_meta`
  files of 2.03 GiB plus one `h_t2_xbits` of 1.02 GiB. Minimal needs
  ~3.0 GiB — it maps each table once rather than once per role. This is
  checked against the temp dir before the batch starts, so an undersized
  dir is a refusal rather than an ENOSPC part-way through a table; note
  the check is per *worker*, and each GPU in a multi-GPU run spills into
  its own files, so size the dir for the number of cards you are
  plotting with.
- **The temp dir sees more traffic than "one write, one read"**, because
  `h_meta` is three roles. At k=28 compact with everything routed, each
  plot moves **14.0 GiB (7.0 written, 7.0 read)**. Every run reports its
  own figure on the `[spill] this plot:` line, so you never have to trust
  this number for your own configuration. The **writes** column is what
  sizes a drive's endurance: at 100 plots/day that is ~0.68 TiB/day,
  which consumes a 600 TB TBW rating in a little over two years. Point
  `--temp-dir` at something you are willing to wear out.
- Files are unlinked at creation, so a crash cannot leave them behind,
  and each is `fallocate`d as it is created — a disk that fills anyway
  fails at once with its size rather than part-way through a table.

## Performance

k=28, strength=2, RTX 4090 (sm_89), PCIe Gen4 x16:

| Mode | Per plot |
|---|---|
| pos2-chip CPU baseline | ~50 s |
| `xchplot2 batch` steady-state wall (pool path) | **2.15 s** |
| `xchplot2 batch` steady-state wall (streaming path, ≤8 GB cards) | ~3.7 s |
| Producer GPU time, steady-state | 1.96 s |
| Device-kernel floor (single-plot nsys) | 1.91 s |

Numbers above are single-GPU. With `--devices 0,1,...` N worker threads
(one per device) race for plots off a shared queue, so each device takes
work at its own rate and throughput is the SUM of their rates — ≈ linear
scaling on matched cards, and mismatched cards still each contribute
fully rather than being held to the slowest. Live multi-GPU plots were
confirmed end-to-end on NVIDIA.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).
