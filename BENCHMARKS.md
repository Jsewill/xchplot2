# Plotter benchmarks — September 9, 2026

The full tier retest covers every supported GPU tier on both hosts and both
branches on NVIDIA. The earlier paired comparison measures a SYCL change that
reduces pinned allocation churn and merges/packs Minimal's host-resident Xs
runs on the CPU.
Ordinary `plot` jobs also persist their identities before work starts and
report output paths only after publication or a successful resume check.

## Sources and hardware

- Full tier retest: `main` `b00ff1b` (local SHA-256 checkout `0a8d3cc`)
  and `cuda-only` `3545835`.
- Earlier paired comparison and profile: baseline `main` v0.11.0 (`1079b9e`;
  local SHA-256 checkout `ff164ba`), candidate `c4f0af6` (local `78ab809`).
- Native CUDA's GPU code is unchanged from `2107292`, including its existing
  CPU merge and scratch reuse.
- All builds use pos2-chip `b0da7aa` and Release optimizations.

The SYCL NVIDIA build enables CUB (`XCHPLOT2_BUILD_CUDA=ON`), targets
`ACPP_TARGETS=generic`, and builds CUDA code for architecture 89. The AMD
build uses `XCHPLOT2_BUILD_CUDA=OFF` and `ACPP_TARGETS=hip:gfx1031`.

| | NVIDIA host | AMD host |
|---|---|---|
| GPU | RTX 4090, 24 GiB, sm_89, PCIe 4 x16 | RX 6700 XT, 12 GiB, gfx1031 |
| CPU / RAM | Ryzen 9 5950X, 32 threads, 125 GiB | Ryzen 7 5800X, 16 threads, 31 GiB |
| Runtime | NVIDIA 610.57.04; CUDA 13.3.73 | Fedora 44; ROCm 7.1.1 |
| Compiler | AdaptiveCpp 25.10.0, LLVM 20.1.2; nvcc 13.3.73 for CUB/native CUDA | AdaptiveCpp 25.10.0, LLVM 20.1.8 |
| Build environment | Ubuntu 24.04.4 test container using the host GPU/driver | Native Fedora 44 |
| Plot destination | Btrfs, Lexar NM790 NVMe | Btrfs, Kingston SV300 SATA SSD |

## Method

The paired and full tier throughput runs use `bench -k 28 -s 2 --devices 0` and write real synthetic
`.plot2` files. FSE compression, output writes, fsync, and the existing overlap
between GPU production and CPU writing are included. The reported seconds
are completion intervals after warmup, not single-plot latency from a cold start.
Typical output size is 0.921 GiB. No CPU builds or other plotter tests ran
concurrently with a benchmark on the same host.

The full tier retest uses ten measured plots after two warmups on both hosts
(12 actual plots per run). `σ` is the per-plot interval spread reported by
`bench`; host peak is Linux `ru_maxrss`, measured separately from verification.
Each run retained its outputs with `--keep`, then checked one output using
`verify --full --trials 100 --config /dev/null` before removing the temporary
plots. Runs were sequential on each host; the two hosts ran concurrently.
For example, with `binary` pointing at either build and `plots` at a disk directory:

```bash
POS2GPU_VRAM_MARGIN_MB=512 "$binary" bench -k 28 -s 2 -n 10 --warmup 2 \
    --devices 0 --tier minimal --out "$plots" --keep --config /dev/null
```

On AMD, use the default 128 MiB margin. The NVIDIA machine also drives a
desktop: during the earlier paired comparison, a browser's GPU allocation
grew by 239 MiB and tripped the 128 MiB watchdog on both baseline and candidate
runs. Those failed runs are excluded below. A repeat measured the same
5,450 MiB process peak (including its 388 MiB initial context) for both
versions. Subsequent local runs use the existing 512 MiB margin override;
production defaults and peak models are unchanged.

Driver peaks are deltas from the run's initial free-memory reading and can
include other desktop activity. They are not required free-VRAM floors.
Forced SYCL tiers on these roomy cards also receive optional two-phase match
scratch (Minimal: 1,170 MiB; Plain/Compact: 780 MiB). Floor-limited cards
have different throughput. Intel and physical low-VRAM cards are outside
this test set.

## Current throughput and memory

Full tier retest, September 9, 2026, 17:23–18:01 UTC, including the Plain repeat:

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

SYCL-CUB Plain was repeated after the full matrix because its initial timing
was slower than the earlier measurement. Both runs passed; the cause of the
difference was not established, so both measurements are retained.

Native CUDA has no separate Pinned tier. The native-vs-SYCL comparison does
not isolate the cause of their runtime differences: their memory footprints
differ, and the native auto run enables optional D2H/Xs overlap, which retains
another 2,080 MiB of device fragments.

## Before and after

These are the earlier paired runs: six measured plots after two warmups on
NVIDIA, four after one warmup on AMD. Their original after values are retained
so the comparison uses the same sample counts and method for both versions.

| GPU / tier | Before, s/plot | After, s/plot | Time reduction | Host RSS before → after, GiB |
|---|---:|---:|---:|---:|
| RTX 4090 / Minimal | 20.26 ± 0.22 | 16.37 ± 0.05 | 19.2% | 18.48 → 17.45 |
| RTX 4090 / Tiny | 29.06 ± 0.71 | 27.08 ± 0.45 | 6.8% | 21.45 → 19.53 |
| RX 6700 XT / Minimal | 26.11 ± 0.25 | 22.03 ± 0.09 | 15.6% | 18.50 → 17.48 |
| RX 6700 XT / Tiny | 31.83 ± 0.82 | 29.80 ± 0.34 | 6.4% | 21.47 → 19.56 |

The existing `h_t1_mi` and `h_t2_mi` cache slots now also hold Xs and sliced
T1/T2 sort inputs after each previous consumer has finished. Minimal packs
Xs into the existing metadata buffer before T1 writes it. Tiny keeps its
separate Xs input because T1 reads it while producing metadata. No new cache
slots or runtime tuning options were added; standalone calls still allocate
and free their own buffers.

## Profile

Nsight Systems 2026.1.3 traced the ordinary Minimal batch path with
`--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none`. Each profile contains
one warmup plus three measured plots. `POS2GPU_PHASE_TIMING` and `--profile`
were unset, preserving the normal overlap. The following totals cover all
four plots, including setup and teardown:

| Metric | Baseline | Candidate |
|---|---:|---:|
| `cudaMallocHost` calls / API time | 57 / 26.55 s | 29 / 12.89 s |
| `cudaFreeHost` calls / API time | 57 / 8.74 s | 29 / 4.34 s |
| GPU merge kernel instances / time | 20 / 23.40 s | 16 / 19.11 s |
| Explicit D2H copies | 128.83 GB | 120.26 GB |

This removes seven pinned allocation/free pairs per Minimal plot and about
2 GiB of explicit D2H traffic per plot. CPU merge/pack replaces the Xs GPU
merge's binary searches over host memory. T1/T2 host-input GPU merges remain.

The baseline's device-to-device copies took only 57 ms across four plots
(about 14 ms/plot), so this change does not redesign CUB's selected-buffer
API. These copy statistics exclude memory accessed directly by kernels.
API and kernel totals overlap across threads and are not additive wall time;
CPU FSE time is not separately attributed by this trace.

## Correctness and recovery checks

All 17 supported GPU configurations on these two hosts fit and passed the
full tier retest, as did the Plain repeat: 216 plots created, including 180
measured plots. One output per run passed 100 random full-proof challenges,
for 1,800 challenges and 1,789 validated full proofs across 18 runs.

The earlier correctness checks on both GPU hosts passed CPU-reference byte
comparisons at k=22 and k=28 for Plain, Minimal, Tiny, and Pinned, including
disk-spill variants of the last three: 14 plot comparisons per host. Each GPU
output also passed 100 full-proof challenges. The CPU-reference SHA-256 values were identical on
both hosts. These comparisons used strength 2, plot index/meta group 0,
mainnet parameters, a plot ID of 32 `ab` bytes, and a memo of 112 zero bytes:

```text
k22  3db63113c3af151a548471ab8cd7497e1d8f44eb26377b717e772976d0806f2d
k28  63d0ea779b8adf69501384bde872e54b926fbfbc92ff34d88b671a5ecc8f8385
```

The existing `sycl_tier_parity` check also passed all nine tier/spill cases
at k=22 on both hosts. `scripts/test/vram-tiers.sh` then completed three
k=28 plots at each streaming floor and rejected a budget 1 MiB below it:

| Tier | NVIDIA budget / measured peak, MiB | AMD budget / measured peak, MiB |
|---|---:|---:|
| Minimal | 4,412 / 3,921 | 4,028 / 3,904 |
| Tiny | 1,612 / 1,105 | 1,228 / 1,080 |
| Pinned | 1,662 / 1,123 | 1,278 / 1,080 |

NVIDIA budgets include the 512 MiB desktop margin described above; AMD uses
128 MiB. These are software-limited tests on the named GPUs, not measurements
on physical small-capacity cards.

Real CLI tests passed on SYCL/CUDA, SYCL/HIP, and native CUDA: interrupt an
unseeded six-plot job after two outputs, resume with the same identities,
preserve completed files, force an output-path write error, and confirm stdout
contains only published or validated paths with a nonzero failure status.
A resumed BLS-derived plot passed 100 full-proof challenges. Parallel CPU
workers also passed resume and failure-status checks using pool-public-key
memos; the GPU tests used pool puzzle hashes.

The expanded `cli_host_test` covers manifest quoting, private permissions,
concurrent publication, refusal to replace another job, ambiguous/mismatched
recovery, seeded recovery, and completed output reporting through cancellation
and errors. It passes in all three builds. Native CUDA's GPU pipeline was
unchanged; all five k=28 configurations above and its CLI checks passed.
Multi-GPU execution, Intel, and Windows were not retested. All hardware work
was invoked directly; neither machine was registered or used as a CI runner.

## Earlier CPU and spill measurements

The following results were retained from the README snapshots on September
9, 2026 (`main` `20f6b67`, `cuda-only` `db2eee7`). Their measurement dates
and source build revisions were not recorded with the tables. They are
historical observations, not reruns of the current full tier matrix or
current minimum-RAM guarantees. No new spill timings were taken for this
documentation cleanup.

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
Multi-GPU plotting was not rerun in the September 9 full tier matrix.
