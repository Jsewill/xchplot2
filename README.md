# xchplot2

CPU and GPU plotter for Chia v2 proofs of space (CHIP-48). Produces ordinary
PoS2 1.0 `.gplot` files using the pinned
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

This is the **`main` branch**, using SYCL/AdaptiveCpp with a CUB fast path
on NVIDIA. The [`cuda-only` branch](https://github.com/Jsewill/xchplot2/tree/cuda-only)
provides the native CUDA implementation for NVIDIA.

This build pins [pos2-chip 1.0.0](https://github.com/Chia-Network/pos2-chip/releases/tag/1.0.0).
The new Feistel cipher and group format are incompatible with earlier plots;
existing plots must be replotted and old job manifests cannot be resumed.
Farmer plot-group integration has merged upstream; release and activation
support must still be confirmed in the
[compatibility notes](contrib/pos2-pr118/README.md).
Only k28 plots are intended for farming; smaller supported sizes are for testing.

Linux and experimental native Windows builds include
[grouped plotting, drive filling and growth](REFERENCE.md#grouped-plotting-and-drive-filling).
Use `plot --group-members N` for several members per file, or `fill DIR...` to
plan and fill drives. `plan DIR...` previews the operation and can optionally
save a layout for later execution. Add `fill --replot` to
[replace eligible old plots](REFERENCE.md#replacing-an-existing-farm), using
configured farm directories and optional exclusions. Use the intended harvester's profile when plotting
on another machine. Repeat `fill DIR` to continue through any member, or add
options to change future work while keeping existing plots and reservations.
`pause`, `drain`, `resume` and `retire` control a job or selected destinations;
`status DIR --watch` follows saved progress.
`--discover-drives DIR` selects supported drives mounted beneath a directory
at any depth. See [drive filling](REFERENCE.md#grouped-plotting-and-drive-filling)
for recovery and policy changes. [Shared compute](REFERENCE.md#shared-compute)
lets enrolled machines contribute CPU/GPU capacity to multiple plotting hosts.
These features are available in `main` source builds; the v0.14.1 release
archives predate them. Follow the [installation guide](INSTALL.md) to build
from source.

[Quick start](#quick-start) · [Hardware](#hardware-compatibility) ·
[Build](#build) · [Commands](#use) · [Performance](#performance) ·
[Documentation](#documentation)

## Quick start

For prebuilt archives, follow [binary installation](INSTALL.md#binary-archives).

Install the [build dependencies](INSTALL.md#native-install) first.
For containers or Windows, follow [INSTALL.md](INSTALL.md).

```bash
cargo install --git https://github.com/Jsewill/xchplot2 --locked
xchplot2 devices

# Replace the key, contract address, and output directory with your values.
xchplot2 plot -k 28 -n 10 \
    -f <farmer-pk-hex> \
    -c <pool-contract-xch1-or-txch1> \
    -o /mnt/plots
```

Without shared compute, this command uses one GPU. Add `--devices gpu` to
use all GPUs, or `--devices all` to include CPU workers. See
[device selection](REFERENCE.md#devices-and-cpu-workers) for more options.

Keep the generated `xchplot2-job-*.tsv` manifest private: it contains plot
keys and lets you [resume an interrupted job](REFERENCE.md#plotting-and-recovery).

Each completed output path is printed to stdout. Check one output with:

```bash
xchplot2 verify /mnt/plots/NAME.gplot --full --trials 100
```

`verify --full` samples challenges and validates the resulting full proofs;
it does not check every byte. See [validation](CONTRIBUTING.md#building-and-running-tests)
for CPU-reference comparisons.

## Hardware compatibility

| Resource | Support and requirements |
|---|---|
| NVIDIA | Maxwell or newer via CUDA/CUB. Pre-Turing GPUs require a CUDA 12.x build. |
| AMD | AdaptiveCpp HIP with a compatible ROCm runtime and driver. RDNA1 needs the [target-selection guidance](INSTALL.md#amd-target-selection). |
| Intel | AdaptiveCpp Level Zero; see the [required settings and limitations](REFERENCE.md#troubleshooting). Use CPU plotting on integrated Intel GPUs for production. |
| VRAM | Tiny's base k=28 floor is 1,228 MiB free after context creation, including the default 128 MiB buffer. Backend sort scratch can raise it. |
| Host RAM | Depends on tier and worker count. Lower VRAM tiers generally use more host RAM; see [memory requirements](REFERENCE.md#memory-requirements). |
| CPU plotting | Opt in with `--devices cpu`, `--devices all`, or `--cpu`; uses pos2-chip's CPU plotter. |
| OS | Linux; WSL2 requires support from the GPU vendor. [Native Windows builds](INSTALL.md#windows) are experimental. macOS is unsupported. |

See the [installation guide](INSTALL.md) for drivers, runtime versions and
platform requirements. Memory needs depend on the backend, tier and workload;
use the [memory guide](REFERENCE.md#memory-requirements) when sizing a system.

## Build

See [INSTALL.md](INSTALL.md) for dependencies, containers, Cargo, CMake,
architecture selection, and Windows/WSL2. CMake also builds the parity and
host test binaries.

## Use

See [configuration and argument files](REFERENCE.md#configuration-and-argument-files)
for reusable options. `xchplot2 --help` prints command syntax.

### Storage planning and filling

| Command | Purpose and guide |
|---|---|
| [xchplot2 plan](REFERENCE.md#grouped-plotting-and-drive-filling) | Preview a farming layout; optionally save it for execution |
| [xchplot2 fill](REFERENCE.md#grouped-plotting-and-drive-filling) | Populate destination drives and resume interrupted work |

To replace an existing farm, use the separate
[inventory and migration workflow](REFERENCE.md#replacing-an-existing-farm).
Ordinary `fill` uses free space and does not remove existing plots. `fill --replot`
prepares and saves its migration before removing selected old plots. It permits
bounded early reclamation when needed; use `--reclaim-before-bytes 0` to require
replacement-first retirement. Complete farming resource bounds are required.
New reviews use bounded structural checks on replacement groups; optional
`sampled` or `strict` verification adds full proofs. Those later checks cannot
recover old plots already deleted under the allowance.

### Shared compute

| Command | Purpose and guide |
|---|---|
| [xchplot2 compute](REFERENCE.md#shared-compute) | Set up sharing, offer capacity, or run a dedicated broker |

Run `xchplot2 compute setup` on the first machine, then create an invitation for
each helper. On a helper, `xchplot2 compute setup INVITATION` enrolls it and starts
its persistent worker. Plotting jobs discover enrolled helpers automatically.

### Plotting jobs

| Command | Purpose and guide |
|---|---|
| [xchplot2 plot](REFERENCE.md#plotting-and-recovery) | Create plots from farmer and pool keys |
| [xchplot2 batch](REFERENCE.md#batch-manifests) | Run or resume a saved plot manifest |
| [xchplot2 grow](REFERENCE.md#growing-an-existing-group) | Extend or recover an existing group file |

See [plotting and recovery](REFERENCE.md#plotting-and-recovery) for saved jobs
and resuming interrupted work.

### Checks and performance

| Command | Purpose and guide |
|---|---|
| [xchplot2 devices](REFERENCE.md#devices-and-cpu-workers) | List GPUs and CPU NUMA nodes |
| [xchplot2 bench](REFERENCE.md#benchmarking) | Measure throughput and estimate time to fill storage |
| [xchplot2 verify](REFERENCE.md#verification) | Check an existing plot, including full proofs with `--full` |

### Testing

| Command | Purpose and guide |
|---|---|
| [xchplot2 test](REFERENCE.md#single-test-plot) | Build a test plot from a group ID and memo |
| [xchplot2 parity-check](REFERENCE.md#parity-checks) | Run the built parity and host tests |

### Shell setup

| Command | Purpose and guide |
|---|---|
| [xchplot2 completions](REFERENCE.md#shell-completions) | Generate Bash, zsh, or fish completions |

## Performance

Measure the current build on your hardware and destination disk:

```bash
xchplot2 bench -k 28 -n 10 --warmup 2 --devices gpu0 -o /scratch
```

Use `--devices cpu` to compare CPU plotting. Throughput depends on the tier,
available memory, compression and storage; sharing a GPU or disk can change
the result. See the [benchmarking guide](REFERENCE.md#benchmarking) for options.

[Recorded benchmarks](BENCHMARKS.md) use the older, pre-1.0 cipher and raw
format. They are historical comparisons, not current `.gplot` performance
estimates.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](INSTALL.md) | Dependencies, containers, Cargo/CMake, Windows and WSL2 |
| [Command reference](REFERENCE.md) | All commands, configuration, devices, memory, environment variables, troubleshooting |
| [Benchmark results](BENCHMARKS.md) | Dated measurements, methodology, and memory use |
| [Contributing](CONTRIBUTING.md) | Architecture, local tests, CI, and contribution conventions |
| [Compatibility and reference tools](contrib/pos2-pr118/README.md) | Upstream compatibility, independent checks, and the experimental reader extension |
| [Security](SECURITY.md) | Private key and manifest handling; vulnerability reporting |

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built with SI.

## Like this? Send a coin my way!

If you appreciate this, and want to give back, feel free.

xch1d80tfje65xy97fpxg7kl89wugnd6svlv5uag2qays0um5ay5sn0qz8vph8
