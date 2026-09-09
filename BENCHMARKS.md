# Plotter benchmarks — September 9, 2026

Measured throughput and memory use. Command options are in the
[benchmark reference](REFERENCE.md#benchmarking).

## Sources and hardware

- Current NVIDIA/AMD results: `main` `b00ff1b`, `cuda-only` `3545835`.
- Paired comparison: `main` `1079b9e` → `c4f0af6`.
- All builds: Release, pos2-chip `b0da7aa`.

The SYCL NVIDIA build enables CUB (`XCHPLOT2_BUILD_CUDA=ON`), targets
`ACPP_TARGETS=generic`, and builds CUDA code for architecture 89. The AMD
build uses `XCHPLOT2_BUILD_CUDA=OFF` and `ACPP_TARGETS=hip:gfx1031`.

| | NVIDIA host | AMD host |
|---|---|---|
| GPU | RTX 4090, 24 GiB, sm_89, PCIe 4 x16 | RX 6700 XT, 12 GiB, gfx1031 |
| CPU / RAM | Ryzen 9 5950X, 32 threads, 125 GiB | Ryzen 7 5800X, 16 threads, 31 GiB |
| Runtime | NVIDIA 610.57.04; CUDA 13.3.73 | Fedora 44; ROCm 7.1.1 |
| Compiler | AdaptiveCpp 25.10.0, LLVM 20.1.2; nvcc 13.3.73 for CUB/native CUDA | AdaptiveCpp 25.10.0, LLVM 20.1.8 |
| Build environment | Ubuntu 24.04.4 container using the host GPU/driver | Native Fedora 44 |
| Plot destination | Btrfs, Lexar NM790 NVMe | Btrfs, Kingston SV300 SATA SSD |

## Method

Current results use k=28, strength=2, ten measured plots after two warmups,
and real file writes including FSE compression and fsync. Typical output is
0.921 GiB/plot. Timings are completion intervals; σ is their sample spread.
Runs were isolated from other plotter jobs and builds.

Host peak is Linux `ru_maxrss`. Driver VRAM peaks are deltas from initial free
memory and can include desktop activity; they are not admission floors.
The VRAM margin was 512 MiB on NVIDIA and 128 MiB on AMD/Intel. Roomy SYCL
cards also received optional match scratch: 1,170 MiB for Minimal and
780 MiB for Plain/Compact. Lower-capacity cards may have different throughput.

## Current throughput and memory

NVIDIA and AMD, September 9, 2026:

| GPU / build | Tier | Seconds/plot, mean ± σ | Driver peak, MiB | Host peak RSS, GiB |
|---|---|---:|---:|---:|
| RTX 4090 / SYCL-CUB | auto | 2.50 ± 0.10 | 11,268 | 7.30 |
| RTX 4090 / SYCL-CUB | plain, initial | 3.91 ± 0.30 | 8,088 | 7.30 |
| RTX 4090 / SYCL-CUB | plain, repeat | 2.66 ± 0.10 | 8,057 | 7.31 |
| RTX 4090 / SYCL-CUB | compact | 3.84 ± 0.04 | 6,012 | 14.38 |
| RTX 4090 / SYCL-CUB | minimal | 16.40 ± 0.18 | 5,090 | 17.46 |
| RTX 4090 / SYCL-CUB | tiny | 27.74 ± 0.55 | 1,275 | 19.53 |
| RTX 4090 / SYCL-CUB | pinned | 27.42 ± 0.35 | 1,309 | 19.49 |
| RX 6700 XT / SYCL-HIP | auto | 9.63 ± 0.76 | 11,232 | 7.30 |
| RX 6700 XT / SYCL-HIP | plain | 9.76 ± 0.32 | 8,088 | 7.30 |
| RX 6700 XT / SYCL-HIP | compact | 10.51 ± 0.63 | 5,998 | 14.41 |
| RX 6700 XT / SYCL-HIP | minimal | 22.19 ± 0.70 | 5,076 | 17.49 |
| RX 6700 XT / SYCL-HIP | tiny | 29.77 ± 0.34 | 1,082 | 19.58 |
| RX 6700 XT / SYCL-HIP | pinned | 29.73 ± 0.37 | 1,082 | 19.58 |
| RTX 4090 / native CUDA | auto, overlap enabled | 2.20 ± 0.03 | 13,582 | 7.19 |
| RTX 4090 / native CUDA | plain | 2.89 ± 0.05 | 7,372 | 7.21 |
| RTX 4090 / native CUDA | compact | 4.59 ± 0.05 | 5,302 | 11.31 |
| RTX 4090 / native CUDA | minimal | 19.53 ± 0.23 | 3,926 | 13.30 |
| RTX 4090 / native CUDA | tiny | 32.65 ± 0.29 | 1,116 | 14.36 |

Both SYCL-CUB Plain timings are retained; the difference was not explained.
Native CUDA has no Pinned tier. Its auto run includes optional D2H/Xs
overlap, which retains another 2,080 MiB of device fragments.

## Before and after

Pinned scratch reuse and CPU Xs merge/pack: six measured plots after two
warmups on NVIDIA; four after one warmup on AMD, at k=28, strength=2.

| GPU / tier | Before, s/plot | After, s/plot | Time reduction | Host RSS before → after, GiB |
|---|---:|---:|---:|---:|
| RTX 4090 / Minimal | 20.26 ± 0.22 | 16.37 ± 0.05 | 19.2% | 18.48 → 17.45 |
| RTX 4090 / Tiny | 29.06 ± 0.71 | 27.08 ± 0.45 | 6.8% | 21.45 → 19.53 |
| RX 6700 XT / Minimal | 26.11 ± 0.25 | 22.03 ± 0.09 | 15.6% | 18.50 → 17.48 |
| RX 6700 XT / Tiny | 31.83 ± 0.82 | 29.80 ± 0.34 | 6.4% | 21.47 → 19.56 |

## Profile

Nsight Systems 2026.1.3, RTX 4090 Minimal, `1079b9e` → `c4f0af6`.
Totals cover four k=28 plots (one warmup), including setup and teardown;
API/kernel times overlap and are not additive wall time.

| Metric | Baseline | Candidate |
|---|---:|---:|
| `cudaMallocHost` calls / API time | 57 / 26.55 s | 29 / 12.89 s |
| `cudaFreeHost` calls / API time | 57 / 8.74 s | 29 / 4.34 s |
| GPU merge kernel instances / time | 20 / 23.40 s | 16 / 19.11 s |
| Explicit D2H copies | 128.83 GB | 120.26 GB |

## Intel Arc B580

September 9, 2026; `main` `e523ac1` (Auto), `d9c4f80` (Plain), using the
k=28 method above.
Arc B580 12 GiB, Ryzen 5 7500X3D, 14.7 GiB usable RAM, WD SN810 NVMe/Btrfs.
Native Fedora 44, compute-runtime 26.22.38646.6, AdaptiveCpp 25.10.0/LLVM 20.1.8;
`ACPP_TARGETS=generic`, `XCHPLOT2_BUILD_CUDA=OFF`. Runtime:
`ACPP_VISIBILITY_MASK=ze NEOReadDebugKeys=1 EnableDirectSubmission=0`.

| Tier | Seconds/plot, mean ± σ | TiB/day | Driver peak, MiB | Host peak RSS, GiB |
|---|---:|---:|---:|---:|
| Auto | 13.75 ± 0.48 | 5.65 | 11,233 | 1.17 |
| Plain | 14.17 ± 0.57 | 5.48 | 8,139 | 1.14 |

Intel driver-owned host allocations are not fully counted in RSS; this
figure is not a host-RAM requirement.

## Earlier CPU and spill measurements

Historical values from `main` `20f6b67` and `cuda-only` `db2eee7`.
Measurement dates and build revisions were not recorded. These predate the
current pinned scratch reuse and are not current minimum-RAM guarantees.

### CPU worker concurrency

Ryzen 9 5950X, 32 threads, CPU-only aggregate steady-state throughput:

| Workers | k=28 aggregate, s/plot | k=28 per worker, s/plot | k=26 aggregate, s/plot |
|---|---:|---:|---:|
| 1 | 52.28 | 52.3 | 13.57 |
| 2 | 43.85 | 87.7 | 10.59 |
| 4 | 41.69 | 166.8 | 9.63 |

A separate mixed CPU/GPU observation on an RTX 4090 reported GPU completion
intervals of 2.56 → 4.23 s/plot with four CPU workers and a 55-plot batch
taking 2.39 times as long as GPU-only. One CPU worker was approximately a
wash. These results motivated the separate CPU-only and GPU-plus-CPU
defaults; they do not establish the best count for other hosts.

### SYCL spill

Earlier k=28 Tiny batch, three plots. The original record did not name the
GPU or measurement date:

| Spilled storage | Peak RSS, GiB |
|---|---:|
| None | 21.5 |
| T1 metadata and T3 | 17.4 |
| Also T2 metadata and T2 X-bits | 14.3 |
| Also reduce drain slots from three to one | 9.3 |

The single-plot observation was 19.5 → 8.4 GiB. These precede the pinned
scratch reuse measured above; use the current table for unspilled RSS.
An earlier Compact comparison reported 6.4 → 8.7 s/plot with spill, means
of three plots. The historical fully routed I/O counts at k=28 were:

| Tier | Total I/O, GiB/plot | Writes, GiB/plot |
|---|---:|---:|
| Compact | 6.0 | 3.0 |
| Minimal | 8.0 | 4.0 |
| Tiny | 29.0 | 12.0 |

### Native CUDA spill

Earlier k=28 Compact measurements on an RTX 4090, one plot per reduction:

| Routed storage | Peak RSS, GiB |
|---|---:|
| None | 11.29 |
| Metadata | 9.32 |
| Also T2 X-bits | 8.30 |
| Also reduce drain slots from three to one | 4.24 |

With NVMe temporary storage, the earlier timing was 9.6 → 12.6 s/plot,
means of three plots. Fully routed Compact I/O was 14.0 GiB/plot, including
7.0 GiB of writes. The earlier `--max-host-ram min` table was:

| Tier | Modeled unswappable peak before → after, GiB | Measured RSS, GiB |
|---|---:|---:|
| Plain | Not recorded | 7.20 |
| Compact | 12.44 → 5.33 | 4.24 |
| Minimal | 13.46 → 6.35 | 9.19 |
| Tiny | 14.47 → 10.41 | 10.28 |

Minimal's file-backed mappings can remain resident, so RSS and the
unswappable-memory model measure different quantities. Tiny's GPU-visible
host tables cannot be mapped to disk; only its drain slots can be reduced.

## Earlier multi-GPU measurements

The earlier `main` README recorded 10-plot k=28 batches on two RTX 4000 Ada
GPUs connected over PCIe, without NVLink:

| Strategy | Seconds/plot |
|---|---:|
| Independent plots through the work queue | 3.75 |
| One sharded plot using peer transport | 9.02 |
| One sharded plot using explicit host bounce | About 14 |

The source build and measurement date were not recorded with these values.
