# Plotter benchmarks — September 9, 2026

The measured change reduces pinned allocation churn in the SYCL streaming
pipeline and merges/packs Minimal's host-resident Xs runs on the CPU.
Ordinary `plot` jobs also persist their identities before work starts and
report output paths only after publication or a successful resume check.

## Sources and hardware

- Baseline: `main` v0.11.0 (`1079b9e`; local SHA-256 checkout `ff164ba`).
- Candidate: `main` `c4f0af6` (local SHA-256 checkout `78ab809`).
- Native CUDA: `cuda-only` `becab6c`, with GPU code unchanged from `2107292`.
  Its existing CPU merge and scratch reuse are unchanged.
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

All throughput runs use `bench -k 28 -s 2 --devices 0` and write real synthetic
`.plot2` files. FSE compression, output writes, fsync, and the existing overlap
between GPU production and CPU writing are included. The reported seconds
are completion intervals after warmup, not single-plot latency from a cold start.
Typical output size is 0.921 GiB. No CPU builds or other plotter tests ran
concurrently with a benchmark on the same host.

NVIDIA uses six measured plots after two warmups (eight actual plots per run).
AMD uses four measured plots after one warmup (five actual plots). `σ` is the
per-plot interval spread reported by `bench`; host peak is Linux `ru_maxrss`.
For example, with `binary` pointing at either build and `plots` at a disk directory:

```bash
POS2GPU_VRAM_MARGIN_MB=512 "$binary" bench -k 28 -s 2 -n 6 --warmup 2 \
    --devices 0 --tier minimal --out "$plots" --config /dev/null
```

On AMD, use `-n 4 --warmup 1` and the default 128 MiB margin. The NVIDIA
machine also drives a desktop: a browser's GPU allocation grew by 239 MiB
and tripped the 128 MiB watchdog on both baseline and candidate runs. Those
failed runs are excluded below. A repeat measured the same 5,450 MiB process
peak (including its 388 MiB initial context) for both versions. Subsequent
local runs use the existing 512 MiB margin override; production defaults and
peak models are unchanged.

Driver peaks are deltas from the run's initial free-memory reading and can
include other desktop activity. They are not required free-VRAM floors.
Forced tiers on these roomy cards also receive optional two-phase match
scratch (Minimal: 1,170 MiB; Plain/Compact: 780 MiB). Floor-limited cards
have different throughput. Intel and physical low-VRAM cards are outside
this test set.

## Current throughput and memory

| GPU / build | Tier | Seconds/plot, mean ± σ | Driver peak, MiB | Host peak RSS, GiB |
|---|---|---:|---:|---:|
| RTX 4090 / SYCL-CUB | auto | 2.55 ± 0.19 | 11,282 | 7.30 |
| RTX 4090 / SYCL-CUB | plain | 2.64 ± 0.04 | 8,087 | 7.31 |
| RTX 4090 / SYCL-CUB | compact | 3.85 ± 0.02 | 6,013 | 14.39 |
| RTX 4090 / SYCL-CUB | minimal | 16.37 ± 0.05 | 5,130 | 17.45 |
| RTX 4090 / SYCL-CUB | tiny | 27.08 ± 0.45 | 1,149 | 19.53 |
| RX 6700 XT / SYCL-HIP | auto | 9.84 ± 0.66 | 11,232 | 7.30 |
| RX 6700 XT / SYCL-HIP | plain | 9.60 ± 0.71 | 8,088 | 7.30 |
| RX 6700 XT / SYCL-HIP | compact | 10.53 ± 0.58 | 5,998 | 14.40 |
| RX 6700 XT / SYCL-HIP | minimal | 22.03 ± 0.09 | 5,076 | 17.48 |
| RX 6700 XT / SYCL-HIP | tiny | 29.80 ± 0.34 | 1,082 | 19.56 |
| RTX 4090 / native CUDA | auto, overlap enabled | 2.21 ± 0.02 | 13,584 | 7.20 |

Auto and Plain are close enough that these small samples do not establish a
consistent ordering. The native-vs-SYCL comparison does not isolate the cause
of their runtime difference; the native auto run also enables its optional
D2H/Xs overlap, which retains another 2,080 MiB of device fragments.

## Before and after

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

Both GPU hosts passed CPU-reference byte comparisons at k=22 and k=28 for
Plain, Minimal, Tiny, and Pinned, including disk-spill variants of the last
three: 14 plot comparisons per host. Each GPU output also passed 100
full-proof challenges. The CPU-reference SHA-256 values were identical on
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
unchanged; its k=28 auto benchmark and the CLI checks above were rerun.
Multi-GPU execution, Intel, and Windows were not retested. All hardware work
was invoked directly; neither machine was registered or used as a CI runner.
