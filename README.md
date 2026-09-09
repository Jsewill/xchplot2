# xchplot2

GPU plotter for Chia v2 proofs of space (CHIP-48). Produces `.plot2` files
byte-identical to the pinned
[pos2-chip](https://github.com/Chia-Network/pos2-chip) CPU reference.

This is the **`cuda-only` branch**, using native CUDA for NVIDIA GPUs.
The [`main` branch](https://github.com/Jsewill/xchplot2) uses
SYCL/AdaptiveCpp for NVIDIA, AMD, and Intel.

This is a work in progress. Current output and the proposed grouped format
are described in the [compatibility and migration notes](contrib/pos2-pr118/README.md).

[Quick start](#quick-start) · [Hardware](#hardware-compatibility) ·
[Build](#build) · [Use](#use) · [Performance](#performance)

## Quick start

Install the [build dependencies](INSTALL.md#requirements) first.
For containers or Windows, follow [INSTALL.md](INSTALL.md).

```bash
cargo install --git https://github.com/Jsewill/xchplot2 --locked --branch cuda-only
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
| NVIDIA | Maxwell or newer. Pre-Turing GPUs require a CUDA 12.x build. RTX 4090 is in the current hardware benchmark set. |
| VRAM | Tiny needs 1,320 MiB free after context creation at k=28, including native CUDA's default 256 MiB buffer. |
| Host RAM | Depends on tier and worker count. Lower VRAM tiers generally use more host RAM; see [memory requirements](REFERENCE.md#memory-requirements). |
| CPU plotting | Opt in with `--devices cpu`, `--devices all`, or `--cpu`; uses pos2-chip's CPU plotter. |
| OS | Linux is tested. WSL2 uses the Linux build; native Windows remains experimental. macOS is unsupported. |

The [benchmark report](BENCHMARKS.md) records tested hardware and toolchains.
Build checks and GPU hardware checks are described separately in
[CONTRIBUTING.md](CONTRIBUTING.md). Physical low-capacity cards are outside
the latest benchmark set; a software VRAM cap does not certify another card.

## Build

See [INSTALL.md](INSTALL.md) for dependencies, containers, Cargo, CMake,
architecture selection, and Windows/WSL2. CMake also builds the parity and
host test binaries.

## Use

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

Measured September 9, 2026, at k=28, strength=2 on an RTX 4090 with a
Ryzen 9 5950X. Times are mean completion intervals and standard deviations
over ten measured plots after two warmups, including FSE compression,
real writes, and durability barriers. Each non-auto tier was forced.

| Tier (`cuda-only`) | Seconds/plot, mean ± σ | Driver peak, MiB | Host peak RSS, GiB |
|---|---:|---:|---:|
| Auto (pool) | 2.20 ± 0.03 | 13,582 | 7.19 |
| Plain | 2.89 ± 0.05 | 7,372 | 7.21 |
| Compact | 4.59 ± 0.05 | 5,302 | 11.31 |
| Minimal | 19.53 ± 0.23 | 3,926 | 13.30 |
| Tiny | 32.65 ± 0.29 | 1,116 | 14.36 |

All five configurations passed, with one output per run checked over
100 full-proof challenges. Native CUDA has no separate Pinned tier.
The auto path enabled D2H/Xs overlap. Driver peaks are deltas from initial
free VRAM and may include desktop activity; they are not minimum capacity
requirements. The local desktop test used the existing
`POS2GPU_VRAM_MARGIN_MB=512` override; production defaults are unchanged.

The [benchmark report](BENCHMARKS.md) includes the exact
method, source revisions, SYCL/AMD comparison, profiling, and correctness
checks. Multi-GPU throughput also depends on shared PCIe bandwidth,
CPU compression, and storage; these measurements use one GPU.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](INSTALL.md) | Dependencies, containers, Cargo/CMake, Windows and WSL2 |
| [Reference](REFERENCE.md) | Devices, tiers, memory, environment variables, troubleshooting |
| [Benchmarks](BENCHMARKS.md) | Dated measurements, methodology, memory use, and validation |
| [Contributing](CONTRIBUTING.md) | Architecture, local tests, CI, and contribution conventions |
| [Security](SECURITY.md) | Private key and manifest handling; vulnerability reporting |

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
attributions. Built collaboratively with
[Claude](https://claude.ai/code).
