# Benchmarks

The September GPU measurements use the pre-1.0 cipher and raw format. For
current `.gplot` measurements, see [grouped plots on HDDs](#grouped-plots-on-hdds).
Use [`xchplot2 bench`](REFERENCE.md#benchmarking) to measure your system.

Single-GPU results measured September 9, 2026, at **k=28, strength=2**.
Each run measured ten plots after two warmups, including FSE compression,
file writes, and fsync. Times are completion intervals, reported as mean ±
standard deviation (σ). Output size was approximately 0.921 GiB/plot.

## GPU results

| GPU / build | Tier | Time, s/plot (mean ± σ) | Driver VRAM, MiB | Host RSS, GiB |
|---|---|---:|---:|---:|
| RTX 4090 / SYCL-CUB | Auto | 2.50 ± 0.10 | 11,268 | 7.30 |
| RTX 4090 / SYCL-CUB | Plain (run 1) | 3.91 ± 0.30 | 8,088 | 7.30 |
| RTX 4090 / SYCL-CUB | Plain (run 2) | 2.66 ± 0.10 | 8,057 | 7.31 |
| RTX 4090 / SYCL-CUB | Compact | 3.84 ± 0.04 | 6,012 | 14.38 |
| RTX 4090 / SYCL-CUB | Minimal | 16.40 ± 0.18 | 5,090 | 17.46 |
| RTX 4090 / SYCL-CUB | Tiny | 27.74 ± 0.55 | 1,275 | 19.53 |
| RTX 4090 / SYCL-CUB | Pinned | 27.42 ± 0.35 | 1,309 | 19.49 |
| RX 6700 XT / SYCL-HIP | Auto | 9.63 ± 0.76 | 11,232 | 7.30 |
| RX 6700 XT / SYCL-HIP | Plain | 9.76 ± 0.32 | 8,088 | 7.30 |
| RX 6700 XT / SYCL-HIP | Compact | 10.51 ± 0.63 | 5,998 | 14.41 |
| RX 6700 XT / SYCL-HIP | Minimal | 22.19 ± 0.70 | 5,076 | 17.49 |
| RX 6700 XT / SYCL-HIP | Tiny | 29.77 ± 0.34 | 1,082 | 19.58 |
| RX 6700 XT / SYCL-HIP | Pinned | 29.73 ± 0.37 | 1,082 | 19.58 |
| Arc B580 / SYCL-Level Zero | Auto | 13.75 ± 0.48 | 11,233 | 1.17 |
| Arc B580 / SYCL-Level Zero | Plain | 14.17 ± 0.57 | 8,139 | 1.14 |
| RTX 4090 / native CUDA | Auto | 2.20 ± 0.03 | 13,582 | 7.19 |
| RTX 4090 / native CUDA | Plain | 2.89 ± 0.05 | 7,372 | 7.21 |
| RTX 4090 / native CUDA | Compact | 4.59 ± 0.05 | 5,302 | 11.31 |
| RTX 4090 / native CUDA | Minimal | 19.53 ± 0.23 | 3,926 | 13.30 |
| RTX 4090 / native CUDA | Tiny | 32.65 ± 0.29 | 1,116 | 14.36 |

RTX 4090 / SYCL-CUB Plain varied between runs; both results are shown.
Unlisted configurations were not benchmarked. Native CUDA has no Pinned tier.

## Hardware and builds

All builds use Release optimizations and pos2-chip `b0da7aa`.
SYCL builds use AdaptiveCpp 25.10.0; native CUDA uses nvcc.

| Configuration | NVIDIA | AMD | Intel |
|---|---|---|---|
| GPU | RTX 4090, 24 GiB, PCIe 4 x16 | RX 6700 XT, 12 GiB | Arc B580, 12 GiB |
| CPU / RAM | Ryzen 9 5950X, 32 threads, 125 GiB | Ryzen 7 5800X, 16 threads, 31 GiB | Ryzen 5 7500X3D, 12 threads, 14.7 GiB |
| OS | Ubuntu 24.04.4 container | Native Fedora 44 | Native Fedora 44 |
| Driver / runtime | NVIDIA 610.57.04, CUDA 13.3.73 | ROCm 7.1.1 | compute-runtime 26.22.38646.6 |
| Compiler | LLVM 20.1.2; nvcc 13.3.73 | LLVM 20.1.8 | LLVM 20.1.8 |
| SYCL build | `ACPP_TARGETS=generic`, CUB on, CUDA arch 89 | `ACPP_TARGETS=hip:gfx1031`, CUB off | `ACPP_TARGETS=generic`, CUB off |
| Storage | Btrfs, Lexar NM790 NVMe | Btrfs, Kingston SV300 SATA SSD | Btrfs, WD SN810 NVMe |
| Source revision | `main` `b00ff1b`; `cuda-only` `3545835` | `main` `b00ff1b` | `main` `e523ac1` (Auto), `d9c4f80` (Plain) |
| VRAM margin | 512 MiB | 128 MiB | 128 MiB |

Intel commands use the environment prefix
`ACPP_VISIBILITY_MASK=ze NEOReadDebugKeys=1 EnableDirectSubmission=0`.

## Interpreting the results

- Driver VRAM is the peak increase in driver-reported use during a run and
  can include desktop activity. These figures are not minimum VRAM requirements.
- Host RSS is the process peak (`ru_maxrss`). Intel driver-owned host memory
  is not fully included, so Intel's RSS must not be used as a RAM requirement.
- Forced SYCL tiers used optional match scratch: 780 MiB for Plain/Compact,
  1,170 MiB for Minimal. Native CUDA Auto used D2H/Xs overlap (+2,080 MiB VRAM).
  These differences prevent a direct comparison of runtime overhead.
- Results use one GPU with no concurrent builds or plotting jobs. Tighter
  VRAM budgets, disk spilling, and shared CPU or storage resources can change
  throughput. See [memory requirements](REFERENCE.md#memory-requirements).

## Windows RX 6700 XT report

User-reported September 13, 2026, using the updated 0.12.0 Windows SYCL
archive published from [`1fa46e9`](https://github.com/Jsewill/xchplot2/commit/1fa46e9691129d6d5b0bcabf066adad5a02311c0).
`xchplot2.exe bench --devices gpu` completed all 11 plots on an RX 6700 XT
(12 GiB, HIP backend), at k=28 and strength=2: one excluded warmup and ten
measured plots, with automatic tier selection.

| Completion interval, s/plot (mean ± σ) | Interval range, s | Peak driver VRAM, MiB | End-to-end throughput, TiB/hour |
|---:|---:|---:|---:|
| 11.73 ± 0.69 | 10.67–12.98 | 11,228 | 0.276 |

The report did not specify the driver version, CPU model, storage, or Windows
version. Full-proof and CPU byte-parity results were not provided. Treat this
as a completed benchmark report; hardware qualification still needs those
correctness checks.

## Grouped plots on HDDs

On November 17, 2026, a Linux storage host with a Ryzen 7 5800X used one
RTX 4090 worker over 1 Gb Ethernet to build k=28, strength=2 groups on
18 TB Toshiba MG09SCA18TA HDDs with Btrfs. Plotting used `--workers 8` on
the storage host. The SYCL Release build included the grouped-compute changes
based on `e0173cbad19a` and pos2-chip `ff89d3e`.

With a 4 GiB plotting RAM budget, a three-member group on one HDD took
153 s with the original external construction path. The updated receiver
and codec policy took 99 s in each of two runs with the same inputs.
These are whole-job times sampled once per second, with no separate warmup.

Across four matching HDDs, the plotter wrote one six-member group per drive
under an 8 GiB RAM budget. With destination scratch enabled, the original
plotter took 16.78 minutes and the updated plotter took 8.87 minutes. Both
reported 53.42 GB of construction writes. With destination scratch disabled,
the updated plotter took 11.59 minutes and reported 86.04 GB of writes. Each
configuration had one uninterrupted timed run with the same plot identities.

### Group size, build time and reader latency

Each size had one fresh build with the same group identity, HDD and 8 GiB
plotting RAM budget. Destination scratch was enabled. All three file sizes
rounded to 988.2 MB per member in decimal units.

| Members | Full build, min | Build, s/member | First quality lookup, ms | Repeat lookup, ms |
|---:|---:|---:|---:|---:|
| 71 | 47.91 | 40.49 | 98.27 | 42.25 |
| 72 | 48.45 | 40.37 | 99.18 | 42.76 |
| 73 | 49.17 | 40.41 | 98.53 | 43.31 |

Reader times are means over three passes of the same 32 deterministic
challenges per file. Each size had a separate calibration before the timed
passes. File order rotated on each pass. The first lookup includes opening
a new reader. The repeat uses the same reader and challenge. Before each pass
the test requested file-cache eviction. Linux controlled which pages stayed
cached. The per-pass mean first-lookup times overlapped across group sizes.
The 71-member file had the fastest mean repeat lookup.

Growing an already published group on one HDD took 13.66 minutes from 71 to
72 members and 13.98 minutes from 72 to 73. Each run used an 8 GiB plotting
RAM budget and included rewriting the group. Preparing independent copies
for the comparison took place outside those timings.

For a layout of 10,000 members on each of four HDDs, a target of 73 used
137 files per drive. A target of 71 used 141. The calibrated stateless
reader profile gave 73 the lowest modeled recurring work of the three,
1.67% below 71. The model used four harvester cores, four lookup workers,
the default storage service estimates and an eight-second response deadline.
With automatic sizing it selected groups of up to 1,667 members on four
drives and up to 385 on all 16 test drives at the same per-drive workload.
The corresponding modeled response bounds were 7.08 s and 7.78 s. The
November 17 builds and reader measurements covered 71 through 73 members.
See [farming layout and reader calibration](REFERENCE.md#farming-layout).

### Matched 72-member builds

On November 18, the same storage host built the same 72-member group twice
on one HDD. Each build used one remote RTX 4090 over 1 Gb Ethernet, eight
owner threads and an 8 GiB owner RAM budget. Both used destination scratch.

| Build | Elapsed, min | Sizing, s | Logical reads, GB | Logical writes, GB |
|---|---:|---:|---:|---:|
| `faf608e` | 48.75 | 471.48 | 194.1 | 193.3 |
| Contribution summary reuse and measured compute batches | 40.19 | 33.00 | 113.9 | 182.4 |

The updated build finished 17.6% sooner in this comparison. The owner reused
chunk summaries it had validated during receipt and received up to three
members per compute task. The baseline received one member per task. Both
builds completed 12 construction batches. Read and write totals are bytes
counted by the plotter, in decimal GB. Over the same intervals, Linux
block-device counters recorded 133.4 GB and 113.9 GB of reads on the test
volume, respectively. Both outputs were 71,156,950,278 bytes and had the
same SHA-256 digest.

### Two HDDs sharing one worker

Two independently seeded 72-member groups each had a dedicated HDD. Their
serial control builds took 39.38 and 38.66 minutes. Building both together
took 64.84 minutes, a 16.9% reduction from the 78.04-minute serial total.
Both schedules shared the same single-slot RTX 4090 and 1 Gb link. The
parallel job shared the same eight owner threads and 8 GiB across both drives.
The two HDD writers worked concurrently for 12.16 minutes. Each serial output
had the same SHA-256 digest as its parallel counterpart.

Across all 144 members in the parallel job, the worker spent 16.2 minutes
generating and encoding and 28.6 minutes sending. After the first output
frame of each task, the owner spent 30.7 minutes waiting for frames. It spent
7.7 minutes processing received frames, including validation and scratch
writes. The two endpoints' times overlap. The serial jobs received up to
three members in a compute task and the parallel job received up to two.

### Growing a published group

Two private copies of the same 72-member plot were grown on the same HDD.
The first copy was grown to 73, then to 74. For the second copy,
`grow --max-members 74` admitted both new members in one batch. Copying the
starting files took place outside the timings.

| Growth to 74 members | Rewrites | Elapsed, min | Logical reads, GB | Logical writes, GB |
|---|---:|---:|---:|---:|
| Two successive additions | 2 | 26.55 | 286.6 | 145.9 |
| Both additions together | 1 | 12.86 | 142.3 | 73.3 |

A fresh 74-member build took 40.28 minutes. All three 74-member outputs were
73,132,555,982 bytes and had the same SHA-256 digest.

### A 385-member group

With the same group identity and resource limits, the updated plotter built a
385-member file on another matching HDD. The file was 380,396,188,656 bytes
and took 238.14 minutes to publish. Construction took 37.11 seconds per
member, compared with 33.49 for the updated 72-member build. Final packing
took 3,444 seconds. The owner completed all final encoding locally.

On the same host, Chia's reference plot manager discovered both native
xchplot2 `.gplot` files during an offline refresh with test keys. It read
the correct member counts and reconstructed the group IDs from the memos.
Chia's solver and verifier validated a full proof from member index 71 in
the smaller file and index 382 in the larger file. This check used
`chia_rs` 0.53.0 and [chia-blockchain at `a156155`](https://github.com/Chia-Network/chia-blockchain/tree/a15615586432d30f9c50dd743d56931596287646).

The installed Gigahorse 2.7.4.giga37 `chia.bin plots check` scanned a
private plot list. It skipped the native xchplot2 `.gplot` filenames. With
the same file contents under `.plot2`, the checker reported “Not a plotfile”.
Under `.plot`, it reported “invalid plot header magic”. Both sizes reached
the same result before any proof lookup.

### Quality lookups on one HDD

The pos2-chip `ff89d3e` grouped reader replayed 256 distinct challenges on
the 72- and 385-member files on the same HDD. Both files used the same group
identity. Challenges came from public signage points captured at the site,
using the plot group ID and Chia's challenge derivation. Every challenge
triggered a quality lookup.
The fresh-reader measurement includes opening the file. The retained reader
was opened and warmed once before the timed lookups.

In the loaded runs, three other readers queried a separate 72-member file
on the same HDD using independently seeded challenges. The workload shared
four logical CPUs and a 4 GiB address-space limit. Each pass began with a
file-specific cache-eviction request. Linux controlled caching during the
pass.

| Reader | Competing readers | 72 members, p50 / p95 / p99 (ms) | 385 members, p50 / p95 / p99 (ms) |
|---|---:|---:|---:|
| Fresh | 0 | 94 / 105 / 110 | 278 / 296 / 304 |
| Fresh | 3 | 109 / 136 / 144 | 307 / 348 / 368 |
| Retained | 0 | 64 / 75 / 79 | 249 / 269 / 277 |
| Retained | 3 | 76 / 105 / 118 | 275 / 310 / 328 |

Percentiles use nearest rank. Under concurrent reads, the slowest observed
fresh lookup was 182 ms at 72 members and 418 ms at 385.

Chia's reference harvester reader at `a156155` also replayed all 256
challenges with its Python prover object retained. With the HDD to itself,
its p99 was 112 ms at 72 members and 302 ms at 385. With three competing
reference readers, p99 was 142 ms and 350 ms, respectively. These processes
shared four logical CPUs and each had a 1 GiB address-space limit. Both
implementations returned the same total qualities in every run at each size.

A separate planner calibration sampled 128 challenges at each size. For
10,000 members on each of 16 HDDs, automatic sizing selected 385 members
per full group. The model used a stateless reader, four harvester cores,
four lookup workers, the default HDD service estimates and an eight-second
response deadline.

| Members per full group | Files per HDD | Modeled work per signage point, s | Modeled response bound, s |
|---:|---:|---:|---:|
| 72 | 139 | 6.01 | 2.77 |
| 385 | 26 | 3.16 | 7.78 |

The 385-member layout reduced modeled recurring work by 47.5% and left
0.22 seconds of modeled response margin. The lookup table measures one
file at a time under the stated HDD load.
