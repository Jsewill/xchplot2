# xchplot2

GPU plotter for Chia v2 proofs of space (CHIP-48). Produces `.plot2` files
byte-identical to the pinned
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

This is the **`main` branch**, using SYCL/AdaptiveCpp with a CUB fast path
on NVIDIA. The [`cuda-only` branch](https://github.com/Jsewill/xchplot2/tree/cuda-only)
provides the native CUDA implementation for NVIDIA.

This is a work in progress. Future changes to the plot format, including
grouping, may require replotting.

[Quick start](#quick-start) · [Hardware](#hardware-compatibility) ·
[Build](#build) · [Commands](#use) · [Performance](#performance) ·
[Documentation](#documentation)

## Quick start

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

Each completed output path is printed to stdout. Check one output with:

```bash
xchplot2 verify /mnt/plots/NAME.plot2 --full --trials 100
```

`verify --full` samples challenges and validates the resulting full proofs;
it does not check every byte. See [validation](CONTRIBUTING.md#building-and-running-tests)
for CPU-reference comparisons.

## Hardware compatibility

| Resource | Requirement or tested scope |
|---|---|
| NVIDIA | Maxwell or newer via CUDA/CUB. Pre-Turing GPUs require a CUDA 12.x build. RTX 4090 is in the current hardware benchmark set. |
| AMD | AdaptiveCpp HIP; RX 6700 XT (`gfx1031`) is in the current hardware benchmark set. RDNA1 needs the [installation-path guidance](INSTALL.md#amd-target-selection). |
| Intel | Arc B580 with AdaptiveCpp Level Zero; see the [runtime workaround](REFERENCE.md#troubleshooting). |
| VRAM | Tiny's base k=28 floor is 1,228 MiB free after context creation, including the default 128 MiB buffer. Backend sort scratch can raise it. |
| Host RAM | Depends on tier and worker count. Lower VRAM tiers generally use more host RAM; see [memory requirements](REFERENCE.md#memory-requirements). |
| CPU plotting | Opt in with `--devices cpu`, `--devices all`, or `--cpu`; uses pos2-chip's CPU plotter. |
| OS | Linux is tested. WSL2 requires support from the GPU vendor. Native Windows SYCL and macOS are unsupported by this build. |

The [benchmark report](BENCHMARKS.md) records tested hardware and toolchains.
Build checks and GPU hardware checks are described separately in
[CONTRIBUTING.md](CONTRIBUTING.md). Physical low-capacity cards are outside
the latest benchmark set; a software VRAM cap does not certify another card.

## Build

See [INSTALL.md](INSTALL.md) for dependencies, containers, Cargo, CMake,
architecture selection, and Windows/WSL2. CMake also builds the parity and
host test binaries.

## Use

| Command | Purpose and guide |
|---|---|
| [xchplot2 plot](REFERENCE.md#plotting-and-recovery) | Create plots from farmer and pool keys |
| [xchplot2 batch](REFERENCE.md#batch-manifests) | Run or resume a saved plot manifest |
| [xchplot2 bench](REFERENCE.md#benchmarking) | Measure throughput and estimate time to fill storage |
| [xchplot2 devices](REFERENCE.md#devices-and-cpu-workers) | List GPUs and CPU NUMA nodes |
| [xchplot2 verify](REFERENCE.md#verification) | Check an existing plot, including full proofs with `--full` |
| [xchplot2 test](REFERENCE.md#single-test-plot) | Build a test plot from a raw plot ID and memo |
| [xchplot2 parity-check](REFERENCE.md#parity-checks) | Run the built parity and host tests |
| [xchplot2 completions](REFERENCE.md#shell-completions) | Generate Bash, zsh, or fish completions |

See [configuration and argument files](REFERENCE.md#configuration-and-argument-files)
for reusable options. `xchplot2 --help` prints command syntax.

Ordinary plotting uses one GPU. Select all GPUs with `--devices gpu`; add
CPU workers with `--cpu`, or select both with `--devices all`. Each GPU
chooses a tier from its own free VRAM. See the
[device reference](REFERENCE.md#devices-and-cpu-workers).

Before starting, `plot` saves an `xchplot2-job-*.tsv` manifest in the output
directory. It contains private plot keys; keep it private and retain it to
recover the job. Repeat the original `plot` command with `--resume`, or
resume directly from the saved manifest:

```bash
xchplot2 batch /path/to/job.tsv --resume
```

Resume validates existing files before skipping them. See
[plotting and recovery](REFERENCE.md#plotting-and-recovery) for identity
matching, manifest selection, progress, stdout, and exit status.

If host RAM is short, use `--temp-dir` to select a real disk for automatic
spill, or `--max-host-ram` to set a budget. Available reductions differ by
tier and branch; see [host RAM and disk-offload](REFERENCE.md#host-ram-and-disk-offload).

## Performance

Measured September 9, 2026, at k=28, strength=2, using real file writes,
FSE compression, and durability barriers. Times are mean completion intervals
over ten measured plots after two warmups, using one GPU per host. Each
non-auto tier was forced.

| Tier (`main`) | RTX 4090, CUDA/CUB | RX 6700 XT, AdaptiveCpp HIP |
|---|---:|---:|
| Auto (pool) | 2.50 s | 9.63 s |
| Plain | 3.91 s; 2.66 s repeat | 9.76 s |
| Compact | 3.84 s | 10.51 s |
| Minimal | 16.40 s | 22.19 s |
| Tiny | 27.74 s | 29.77 s |
| Pinned | 27.42 s | 29.73 s |

The two RTX 4090 Plain timings differ for an undetermined reason.

Arc B580 / Level Zero: **13.75 s/plot** with Auto, **14.17 s/plot** with Plain;
see the [Intel configuration](BENCHMARKS.md#intel-arc-b580).

The native `cuda-only` auto path measured **2.20 s/plot** on the same RTX
4090, with its optional D2H/Xs overlap enabled. These runs do not isolate the
cause of the difference between the native and SYCL runtimes.

The [benchmarks](BENCHMARKS.md) include configurations, variability, and
memory use. Forced tiers can use additional match scratch on these roomy GPUs;
their timings do not predict performance on a card restricted to a tier's
minimum VRAM.

Multi-GPU throughput also depends on shared PCIe bandwidth, CPU compression,
and storage. This benchmark set uses one GPU per host.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](INSTALL.md) | Dependencies, containers, Cargo/CMake, Windows and WSL2 |
| [Command reference](REFERENCE.md) | All commands, configuration, devices, memory, environment variables, troubleshooting |
| [Benchmark results](BENCHMARKS.md) | Dated measurements, methodology, and memory use |
| [Contributing](CONTRIBUTING.md) | Architecture, local tests, CI, and contribution conventions |
| [Security](SECURITY.md) | Private key and manifest handling; vulnerability reporting |

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).

## Like this? Send a coin my way!

If you appreciate this, and want to give back, feel free.

xch1d80tfje65xy97fpxg7kl89wugnd6svlv5uag2qays0um5ay5sn0qz8vph8
