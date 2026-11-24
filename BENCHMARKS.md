# Benchmarks

The September GPU measurements use the pre-1.0 cipher and raw format.
Current `.gplot` measurements cover [native CPU plotting](#native-cpu-plotting),
[native Windows grouped plotting](#native-windows-grouped-plotting)
and [grouped plots on HDDs](#grouped-plots-on-hdds). See also the
[CPU plotting memory experiment](#cpu-plotting-memory-experiment).
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

The September report recorded benchmark completion. The November measurements
below include CPU byte comparisons and full-proof verification.

## Native Windows grouped plotting

On November 22–23, 2026, a Radeon RX 6700 XT (12 GiB) and a Ryzen 9 5950X
with 32 GiB of RAM built k=28, strength=2 plot groups.
The native Windows 11 build used the HIP backend and AMD driver
32.0.21045.5002. Construction used RAM, with final output on an NTFS NVMe
SSD and destination scratch disabled.

| Generator | CPU worker budget | Construction | Seconds per durable member |
|---|---:|---|---:|
| RX 6700 XT | Automatic | Automatic | 15.40 |
| RX 6700 XT | 4 | Automatic | 13.86 |
| RX 6700 XT | 4 | Incremental | 13.89 |
| Ryzen 9 5950X | 4 | Incremental | 78.22 |

Each row above is a two-member run of `xchplot2 batch` with compact member
storage, before the Windows write and GPU handoff changes.
Times include generation, assembly and durable publication of the complete group.
The first Radeon run with a fresh runtime cache took 15.83 seconds per member
with four workers and incremental construction. The table uses the warmed GPU
cache. Every group matched all 1,991,815,377 bytes of the independent CPU
reference and passed a separate full-proof check.

Repeating automatic construction with phase timing enabled took 14.27, 13.84,
14.21 and 14.42 seconds per member at two, four, eight and sixteen workers,
respectively. The default worker budget took 15.45. GPU generation stayed near
11.1 seconds per member. The append for the pair took 4.53 seconds with four
workers and 7.79 seconds with the default budget.

With four workers and automatic construction, profiling measured about
11.1 seconds of GPU generation and host compaction per member, then
4.74 seconds to append the pair. Raw member storage shortened generation
but increased the pair's append time to 7.02 seconds and finished at
14.24 seconds per member. Forcing generation overlap on this 12 GiB card
retained the tiled output path and took 13.88 seconds per member.

### Coalesced writes and overlapping compaction

The updated Windows writer combines codec output from multiple workers in
the space already reserved for the output window. On this 12 GiB Radeon,
automatic generation overlap also lets the CPU compact one member while
the GPU generates the next. Each run published one group using automatic
construction with phase timing disabled. The warm runs reused each binary's
working directory and reversed the order of the binaries between pairs.

| GPU kernel cache | Members per group | CPU workers | Before, seconds per published member | After, seconds per published member |
|---|---:|---|---:|---:|
| Fresh | 2 | Automatic | 16.55, 17.15 | 15.00 |
| Fresh | 4 | Automatic | 15.58, 15.63 | 13.77 |
| Warm | 2 | Automatic | 15.00, 15.62 | 13.63, 13.65 |
| Warm | 4 | Automatic | 15.03, 15.10 | 13.17, 13.25 |
| Warm | 2 | 4 | 14.05, 14.74 | 13.69, 14.52 |

In each warm four-member run with automatic workers, Windows recorded
51,160 writes from the plotting process before the change and 3,396 after.
With two members and four fixed workers, the timing ranges overlapped.

In the profiled four-member runs the append took 12.38 seconds before and
7.83 seconds after. With four workers on the updated build, switching
generation overlap from off to automatic reduced
publication time from 14.61 to 13.64 seconds per member. Allocating the
pinned host buffer took 0.31 seconds and releasing it took 0.13 seconds for
the whole group.

All two-member and four-member outputs matched the independent CPU files
and passed full-proof verification. The four-member file was 3,971,617,699
bytes. The bundled Windows AdaptiveCpp runtime keeps compiled kernels in a
`.acpp` directory under the working directory. Reusing that directory across
plotting commands lets later runs reuse those kernels.

As a temporary compute worker for a Linux storage host, the same Radeon
processed one k=28 member in 15.22 and 15.36 seconds in two trials. Both
sent 1,157,902,871 bytes through an SSH forward and produced the same
verified file. Sending took 9.21 seconds in the first trial and 97.78
seconds in the second. Windows reported a negotiated Ethernet speed of
100 Mb/s after the second trial. Payload throughput during that send was
94.74 Mb/s.

On November 23, 2026, with Windows reporting a 1 Gb/s Ethernet link, the same singleton took
37.14 seconds through durable publication on the Linux owner. The worker
reported 22.88 seconds of processing and 9.16 seconds sending the result.
The owner received 1,157,902,871 bytes in 9.93 seconds, or 0.93 Gb/s of
plot data. The 1,002,771,036-byte file matched the CPU reference and
passed 79 full proofs. The preceding trial
at 100 Mb/s took 117.98 seconds through publication, including 97.67 seconds
of worker send time. These were single-member runs with a fresh Windows
worker and a waiting owner.

On the same Radeon, a separate process allocated 192 MiB of GPU memory
while a worker advertised a k=18 GPU profile. The test set the worker's
memory reserve close to the available headroom so this small allocation
would cross its admission threshold. The broker withdrew GPU availability
and held the owner's request in its queue. After the allocation was
released, the same worker instance became eligible and delivered the
member. The published file matched the reference and passed 84 full-proof
checks. Plot construction used RAM throughout.

## Native CPU plotting

### Batched CPU matching

Both versions plotted the same k=28, strength-2 members on four pinned
E-cores of a shared Core Ultra 7 270K Plus. Timings include compression,
writing and publication, from process launch through successful exit.

| Completed job | Original native generator | Batched CPU matching | Peak RAM with batched matching |
|---|---:|---:|---:|
| One member | 36.93 s/member | 28.52 s/member | 7.85 GiB |
| Two members in one group | 40.41 s/member | 33.60 s/member | 8.96 GiB |

The single-member times are medians of three runs per version. Individual
runs took 36.91–37.75 seconds before and 28.29–29.81 seconds after. Each
two-member group used automatic construction, a 12 GiB limit and one
destination. Whole-group times were 80.83 and 67.19 seconds. Both groups
matched the independent pos2-chip file byte for byte.

In a separate k=28 run, generation took 27.21 seconds before the writer.
Building and probing the match indexes, evaluating candidate pairs and
writing their output rows accounted for 19.62 seconds. Exporting and
sorting the final fragments took 1.49 seconds.

On a shared Ryzen 9 5950X running Windows, two k=28 runs per version took
61.25–62.05 seconds before and 47.85–47.93 seconds after. These times
include the SSH invocation. Both versions used four threads at below-normal
priority, published the same file and used zero scratch bytes.

### Initial native CPU measurements

On November 24, 2026, four pinned E-cores on a shared Core Ultra 7 270K Plus
published a k=28, strength-2 member through the ordinary batch writer.
Timing starts at process launch and ends at successful exit. The two builds
contain the same C++ member generator. Each row is one run.

| Build | Peak process RAM | Seconds per published member |
|---|---:|---:|
| SYCL/AdaptiveCpp | 7.85 GiB | 38.47 |
| CUDA-only | 7.84 GiB | 36.83 |

The native and reference generators produced the same SHA-256 across all
268,498,840 sorted fragments, including duplicates. Both 1,002,771,036-byte
files matched the independent reference byte for byte and passed full-proof
checks. A two-member k=28 group using automatic
construction, four threads and a 12 GiB RAM budget took 80.95 seconds through
process exit, or 40.48 seconds per member. Its file matched the independent
two-member reference and passed 145 full-proof checks.

The k=22 file also matched on Linux and Windows.
The Windows grouped CPU path published the same k=28 file using four threads
and RAM-only construction in 51.20 seconds on a different shared host.

On a Ryzen 7 5800X with an RTX 3070, the harvester had 17,886 plots loaded.
Twelve k=24, strength-2 members went to one separate test HDD with four
host cores available to each batch. Elapsed time includes publication.

| Devices | Whole job | Seconds per member | Members completed by CPU / GPU |
|---|---:|---:|---:|
| CPU | 38.51 s | 3.21 | 12 / 0 |
| GPU | 11.05, 11.12 s | 0.92–0.93 | 0 / 12 |
| CPU and GPU | 9.35, 9.31 s | 0.78 | 2 / 10 |

Every job published the same twelve files. With the previous cold-start
estimate the CPU stood down for all twelve members. On a shared RTX 4090
workstation, the updated scheduler left a two-member k=24 job to the GPU.
It finished in 0.63 seconds and both files matched.

## CPU plotting memory experiment

On Linux, each variant published the same k=28, strength-2 member using
four pinned E-cores on a shared Core Ultra 7 270K Plus. The three C++
variants used the same writer and publication path in a test harness.
Each row is one run through process exit.

| CPU generation path | Peak process RAM | Seconds per published member |
|---|---:|---:|
| Upstream final fragment vector copy (former standalone path) | 12.13 GiB | 43.43 |
| Move the fragment vector (former grouped and pooled path) | 10.13 GiB | 42.93 |
| Move the vector and release unused table pages before export (prototype) | 8.13 GiB | 42.94 |
| cpos v0.3.0, normal CPU plot and group writer | 7.28 GiB | 58.18 |

The cpos generator reused seven arrays of 32-bit words across plotting
stages. Its separate 16 GiB scratch reservation is virtual address space.
The RAM column measures peak resident memory for each process.

All three C++ files matched byte for byte. The cpos file differed only
in its memo. All four passed full-proof checks. At k=22, nine interleaved
runs per variant measured two independent timing improvements. Skipping
eager zeroing of the table arena reduced median time from 0.602 to
0.574 s/member. Reserving matched-output space in batches took
0.572 s/member. Their combined effect and their k=28 timings are unmeasured.

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

### Automatic fills on an active farm

On November 24, 2026, a Linux storage host ran k=28, strength=2 fills on
empty HDDs while its Chia harvester had 17,886 plots loaded. Two enrolled
GPU workers supplied compute. Each group contained three members. The owner
had a 32 GiB plotting RAM cap and chose incremental RAM construction in all
three automatic runs. These measurements preceded the change that lets an
independent writer try a complete encoded group when the pool has spare capacity.

| Destination layout | New members | First group durable, s | Last group durable, s | Seconds per new durable member | Eligible lookup p95, s (count) |
|---|---:|---:|---:|---:|---:|
| One group on one HDD | 3 | 23.50 | 23.50 | 7.83 | 10.24 (2) |
| Three successive groups on one HDD | 9 | 23.16 | 60.05 | 6.67 | 14.90 (6) |
| One group on each of four HDDs | 12 | 42.60 | 131.74 | 10.98 | 24.28 (13) |

Times run from the start of the command through publication. On one HDD,
workers encoded complete groups for direct publication. In the four-HDD run,
four destinations received packed members for assembly on the owner. Across
the 12 members, the workers recorded 109.61 seconds waiting to send frames.
Waits on different workers can overlap.

For the same four destinations and plot identities, forcing incremental
construction took 128.44 seconds with automatic worker output. With
`--compute-output encoded`, the workers returned four complete groups.
The first published at 21.96 seconds and the last at 73.36 seconds,
or **6.11 seconds per durable member**. The encoded run transferred 12.15 GB
and spent 0.93 cumulative seconds waiting to send frames. Every published
file matched its automatic counterpart and passed full-proof verification.
Each transfer policy had one timed run on this layout.

Forced external construction took 29.37 seconds per member on the single
group and 20.48 across four HDDs. The three-group external run on one HDD
ended early. One attempt received a zero-entry GPU result. A later attempt
published its first group before the broker rejected a delivery whose lease
had been fenced.

A separate two-minute farming-only window recorded a p95 of 17.75 seconds
across 12 eligible lookups. The machines were shared and the lookup counts
varied by run. A comparison of sustained lookup latency would require longer
matched farming windows.

### Fill startup at 1,024 directories

A managed fill on one local filesystem requested two single-member groups
in each of 1,024 empty directories. The command took 132 seconds to issue its
first grant, including durable job setup in each directory. The controller
spent 0.47 ms on the forecast for that grant. It accounted for all 2,048
members, including 1,024 in later groups, and reported unknown timings for
all 1,024 destinations. The test stopped after the first grant.

### External scratch bandwidth

A 4 GiB stream through the external run writer and reader measured the
following rates on the storage host. The scratch write includes physical
reservation, payload writes and durable flushes. Reads checked ordering after
clean-cache eviction. The direct-I/O columns use separate sequential tests
with 2 MiB requests and five seconds of work per HDD. Rates are decimal MB/s.

| Scratch HDDs | External write | External read | Direct write | Direct read |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 819 | 918 | 1,064–1,071 | 1,072–1,073 |
| 16 | 2,281 | 2,492 | 3,322–3,329 | 3,234–3,237 |

The scratch runs used a 4 GiB RAM grant, a 16 MiB I/O window and 32 MiB
extents. A longer 16 GiB stream with those settings sustained 807 MB/s
writing and 957 MB/s reading on four HDDs, and 2,395 MB/s writing and
2,797 MB/s reading on 16. For the 4 GiB stream the previous external writer
and reader delivered 259 and 269 MB/s with the run distributed across four
disks. Supplying those same directories to the previous plotter for a fresh
job selected one scratch disk.

A 24-member k=26, strength=2 plot used four scratch HDDs, a separate output
HDD, eight CPU workers and 4 GiB of RAM. With the same identity and two-member
batches, the previous plotter finished in 379.1 s and the updated plotter
finished in 345.6 and 347.6 s. Encoding fell from 25.9 to 11.1–11.5 s,
merging from 13.9 to 5.3–6.1 s, and final packing from 43.3 to 33.8–34.9 s.
CPU generation took 291.3 s before and 289.3–290.0 s after the change.
All three 5,530,724,485-byte files had the same SHA-256.

With two simultaneous eight-member k=24 groups, each destination used a
different pair of the four scratch HDDs. The previous planner assigned one
scratch HDD per destination. Both runs took about 48 s under the shared
eight-worker CPU budget. Each group matched its counterpart byte for byte.

In a separate 12-member k=24 plot, an independent sequential reader sustained
279 MB/s on one of the four scratch HDDs. That disk received 7.2% of the
physical scratch-drive writes, compared with 32.0% for the same plot with the
reader stopped. The two runs took 41.5 and 40.1 s respectively and produced
identical files.

#### Final packing on four scratch HDDs

Both builds used bandwidth-aware scratch placement. The previous planner
merged four runs before packing the eight that remained. The updated planner
packed all 11 directly. This avoided 1.89 GB each of scratch reads and writes
for a 24-member k=26 group, and 4.13 GB each for a 12-member k=28 group. Both
used strength 2, eight CPU workers, four scratch HDDs and a separate output
HDD. The modeled peak RAM and the SHA-256 of each finished plot matched across
builds.

The k=26 job generated its members on the CPU in two-member batches under a
4 GiB RAM limit. The k=28 job used one-member batches and an 8 GiB limit.
The same twelve raw inputs were generated once on an RTX 4090 in 26.30 s
and loaded from the storage host for each run. The k=28 construction time
excludes the measured time to load and decode those inputs. Each build ran
the k=26 job once. The k=28 ranges cover three runs of the previous build
and two of the updated build.

| Per-group measurement | Previous | Updated |
| --- | ---: | ---: |
| k=26, 24 members, entire CPU job | 347.1 s | 333.1 s |
| k=26, final packing | 34.1 s | 26.6 s |
| k=28, 12 members, construction | 126.1–127.5 s | 97.6–99.1 s |
| k=28, final packing | 71.4–72.2 s | 55.6–57.1 s |

At 2.2 s of production per member, 12 members add 26.4 s when generation
and construction run in sequence. On this layout that projects to 281–283
finished k=28 members per hour with the previous build and 344–348 with the
updated build. Final packing produced 208–214 MB/s of finished plot data.
Two isolated sequential writes on the output HDD measured 251 and 253 MB/s.
Keeping up with one member every 2.2 s at this plot size requires 450 MB/s of
aggregate output bandwidth.

A separate k=28 comparison used three dedicated scratch HDDs and shared the
output HDD as a fourth scratch disk, with a 3 GiB scratch allowance per disk.
Both runs packed all 11 inputs directly. Packing took 77.8 s with the previous
read-ahead schedule and 72.2 s when the packer retried deferred reads each
window, with one run per setting.

#### Packing onto independent output HDDs

Three k=28, strength-2 groups each had 11 members staged on four dedicated
scratch HDDs and a separate output HDD. The same public `batch --resume`
command finished the three groups under a shared 24 GiB RAM budget. Timing
starts at resume and ends after publication. Member generation, staging and
subsequent hash verification fall outside this interval.

| Shared codec workers | Previous time | Updated time | Previous finished MB/s | Updated finished MB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 160.1 s | 97.8 s | 204 | 334 |
| 8 | 56.0 s | 47.5 s | 583 | 687 |

With one codec worker, the previous planner wrote one destination at a time.
The updated planner wrote all three concurrently. With eight workers both
planners used all three writers. The update aligned the buffered output writes
and averaged 716 MB/s of physical scratch reads and 687 MB/s of physical
output writes. This finished one member every 1.44 s. A producer supplying
one of these members every 2.2 s needs 450 MB/s of finished output. The three
SHA-256 digests matched at both worker counts. Each configuration had one
timed run.

Two direct-I/O trials on the same disks paced scratch reads to the plotter's
measured read/write ratio. Each output wrote 2 GiB sequentially. Counting
finished bytes through the last device completion gave 743–744 MB/s across
three outputs. The eight-worker pack reached 92.4% of that rate.

With one output HDD, four scratch HDDs and eight workers, a separate
12-member k=28 group packed at 216 MB/s before the change and 241 MB/s
afterward. Both builds produced the same 11,876,141,735-byte file. A
direct-write prototype with an additional 16 MiB buffer per writer packed
this file in 48.4–48.8 s. The buffered implementation took 49.4 s using
the two output windows already covered by the RAM grant.

Using the same 15 HDDs, seven outputs shared eight scratch drives. One
output had two dedicated scratch drives and the other six had one each.
With 16 workers, a 16 MiB I/O window and 24 GiB of RAM, the previous build
published seven six-member k=28 groups in 49.0 s (849 MB/s). The updated
build took 34.3 and 35.7 s (1,167–1,214 MB/s), or 0.82–0.85 s per member.
All seven files matched across builds and passed full-proof checks.

The two final direct-I/O trials yielded 1,572 MB/s at the same per-disk
traffic mix. Packing reached 74–77% of that sequential rate. Block-device
counters averaged 1,210–1,255 MB/s of scratch reads and 1,163–1,206 MB/s
of output writes across the two plotting runs. The HBA negotiated PCIe 3.0
x4, which carries up to 3.94 GB/s in each direction before transaction
overhead.

Before enabling read-ahead on individual scratch HDDs, the updated planner
and aligned writer reached 1,006 MB/s with a 16 MiB window and 906 MB/s
with a 64 MiB window on those seven groups. A separate six-member group
using one scratch HDD and one output HDD packed at 180 MB/s with synchronous
refills and 188 MB/s with the ordered read-ahead queue.

For a fresh, CPU-bound comparison, one worker generated and packed three
eight-member k=24 groups. The previous build took 216.5 s and the final
build took 216.2 s. All three files matched across builds. The planner keeps
its ordinary concurrency during unmeasured CPU generation and can open
independent HDDs for final packing.

#### Time to publish a group assembled in RAM

A single RTX 4090 worker sent three k=28 members over 10 GbE to a storage
host with eight plotting CPU workers and a 32 GiB RAM grant. With
`--construction auto`, `fill` kept all three members in RAM and wrote the
group directly to one HDD. Each build ran twice with the same owner and seed.

| Mean time per group | Previous | Updated |
| --- | ---: | ---: |
| Generate and receive three members | 16.0 s | 15.9 s |
| Plan the first batch | 4.60 s | 1.00 s |
| Encode and durably write the group | 14.4 s | 14.4 s |
| Launch to publication | 35.5 s | 31.9 s |

Planning now acquires a CPU slot per output window and can reuse the member
hash calculated during delivery. Mean time to publication fell 10.1%.
The final write phase delivered 208 MB/s against a 241 MB/s sequential
direct-write measurement on the same HDD. All four files had the same
SHA-256 and contained 2,983,531,299 bytes.

With two output HDDs sharing the same GPU and RAM grant, each drive received
one three-member group. The mean time until both groups were published fell
from 51.4 s to 46.0 s across two runs per build. Each drive's file matched
across builds, and every run used zero scratch bytes.

#### Streaming a fresh group to HDD

On November 19, 2026, the same RTX 4090 and 10 GbE connection ran the
three-member k=28 workload with one worker slot, comparing the streaming
build with `48f679f`. For one destination the worker encoded the complete
group in file order. The owner validated incoming chunks and started writing
them while the rest of the group was still in transit.

| Workload | Previous per member | Streaming per member | Previous elapsed | Streaming elapsed | Less time |
| --- | ---: | ---: | ---: | ---: | ---: |
| One HDD, 3 members | 10.64 s | 6.68 s | 31.93 s | 20.05 s | 37.2% |
| Two HDDs, 6 members | 7.67 s | 5.53 s | 46.01 s | 33.16 s | 27.9% |

Per-member time divides wall time from launch to the last publication by the
total members published across all drives. Times are means of two runs per
configuration with no untimed plotting warmup. The files matched the previous build's
SHA-256 on each destination and all runs used zero scratch. Full proof checks
passed for the single-drive output and both paired outputs.

The default checkpoint setting and explicit `--checkpoint-bytes auto`
published in 20.10 s and 20.01 s. Both selected a 4 GiB target after the
first measured interval and finished the 2,983,531,299-byte file with three
checkpoints. Fixed-target comparisons on the same drive averaged 21.61 s
at 256 MiB, 20.11 s at 1 GiB, 20.26 s at 2 GiB, and 19.88 s at 4 GiB
(two runs each). Four additional automatic runs on the same streaming path
ranged from 19.87 s to 20.31 s.

During the receive, validate, and write phase, the owner delivered 235.2 MB/s
of final plot data. An independent 2 GiB sequential direct write with
`fdatasync` measured 240.9 MB/s on that drive. One-second block and network
counters showed substantial disk writes by eight seconds from launch while
data was still arriving. Assuming 2.2 s to produce each member, producing
all three and then writing the file at the measured sequential rate would
take 18.98 s before scheduling and publication overhead.

With two drives the worker sent three packed members to the first destination
in one lease. That drive encoded and wrote its group locally while the worker
produced the second group in file order. Both HDDs wrote concurrently. Their
combined peak over a one-second sample was 528 MB/s in the first run and
508 MB/s in the second. Peak receive traffic on the 10 GbE interface was
3.92 Gb/s and 3.93 Gb/s, respectively.

#### More output HDDs with one GPU worker

On November 19, 2026, `fill` at `2e41083` ran on up to 16 independent
output HDDs with one RTX 4090 worker slot, 10 GbE, eight plotting CPU
workers and a 32 GiB plotting RAM budget. Each drive received one fresh
three-member k=28 group with automatic construction and checkpointing.
Each table entry is the mean of two runs without an untimed plotting warmup.
Seconds per member include launch through publication of the last group.

| Output HDDs | Members published per run | Mean seconds per published member |
| ---: | ---: | ---: |
| 1 | 3 | 6.68 |
| 2 | 6 | 5.53 |
| 3 | 9 | 5.33 |
| 4 | 12 | 5.48 |
| 8 | 24 | 5.61 |
| 16 | 48 | 16.36 |

Through eight HDDs the groups used zero scratch. At 16, automatic
construction selected external storage on the output HDDs. The worker
sent one contribution per task, compared with three members per task on
eight drives. To produce 47.73 GB of plots, the 16 disks recorded 108.3 GB
of writes and 49.56 GB of reads per run. The highest one-second samples
were 588 MB/s of aggregate disk writes and 1.92 Gb/s of inbound traffic.
The same 16 drives sustained 3,328 MB/s in the independent sequential
write test.

A fixed-work comparison placed eight groups on each of two HDDs. Across
two runs, those drives published 48 members at 6.15 s/member with zero
scratch. The default 16-drive layout took 2.66 times as long for the
same number of members.

With all 16 HDDs and `--destination-temp off`, one additional run took
6.61 s/member, 59.6% less time than the 16-drive default. Incremental
construction grew partial groups one member at a time. The drives recorded
47.8 GB of reads and 96.5 GB of writes. All 16 files matched the SHA-256
digests of the full-proof-verified outputs from the default run.

#### Rotating output and scratch roles during `fill`

On the same 16 HDDs, a planner based on `2e41083` used the single advertised
GPU worker slot when choosing the initial set of output groups. The workload,
32 GiB plotting RAM budget, and farming limits matched the preceding test.
Each filesystem had a 1 GiB free-space reserve. Elapsed time starts at
launch and ends at publication. Each build published one three-member
k=28 group on each drive. Live harvester lookup latency was unmeasured.

| Planner | Runs | Seconds to first group | Seconds per published member | Disk reads per run | Disk writes per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2e41083` | 2 | 98.9 | 16.36 | 49.56 GB | 108.3 GB |
| Worker-aware HDD admission | 2 | 38.0 | 5.23 | 0 GB | 48.10 GB |
| Final build with bounded codec reservations | 1 | 47.8 | 5.26 | 0 GB | 48.10 GB |

Times and byte counts are means where two runs are shown. The revised
planner wrote the groups directly to their destinations. All 16 files
matched the corresponding SHA-256 digests from the full-proof checks
in each run. Disk counters cover plotting through process exit. Hashing
and verification took place afterward.

In the final run the highest one-second samples were 536 MB/s combined
disk writes and 3.89 Gb/s inbound Ethernet. The independent array
write test reached 3,328 MB/s. Thirteen generation batches started
while a destination writer was active.

The 48 members occupied 47.73 GB. At a production time of 2.2 s/member,
the destinations would have to sustain about 452 MB/s of final writes.
The single-HDD direct-write reference was 241 MB/s. At that speed at
least two independent destination writers are needed to keep up, even
with all scratch reads and encoding overlapped. At the measured 208 MB/s
for encoding and durably writing on one HDD, two such writers would deliver
416 MB/s if each sustained that rate.

At 4 GiB, the same worker filled eight HDDs with 24 members using disk
scratch. The choice of destinations affected the first publication and
the total fill time differently.

| Eight-HDD layout | Runs | Seconds to first group | Seconds per published member |
| --- | ---: | ---: | ---: |
| `2e41083`, three scratch HDDs for seven groups and seven for one | 1 | 43.9 | 13.33 |
| One destination with seven scratch HDDs per group | 1 | 41.1 | 13.45 |
| Two destinations with three scratch HDDs each | 2 | 51.8 | 9.78 |

The two-destination runs took 9.81 and 9.76 s/member. Each started ten
generation batches during a destination write and published a group on
each of the eight HDDs. The other two layouts started every generation
batch while the writers were idle. Both final runs matched all eight
full-proof-checked reference files. During testing an abandoned encoding
request kept the only GPU slot reserved while a generation task waited.
The final broker expires unclaimed encoding requests at the window's deadline.

With four HDDs, a 4 GiB run put two groups on each destination for the
same 24 members. Each group used one scratch HDD. All four drives
published their first group before any published a second. The final
build took 14.34 s/member, compared with 9.78 s/member on eight drives.
All eight groups on the four drives passed full-proof verification.
A four-drive trial with single-member batches earlier in development
took 13.94 s/member. Each four-drive configuration ran once.

In the two eight-HDD runs, the single worker spent 6.85 s/member generating
and encoding compressed contributions, excluding time sending them. In
the final 32 GiB run it spent 1.43 s/member on that counter and transferred
three members per task. Worker time and storage time overlap.

#### Worker assembly and destination writer count

Four empty HDDs on the storage host each received a three-member k=28 group.
The owner had 32 GiB of plotting RAM and eight plotting threads. One RTX 4090
worker with eight CPU threads sent the same seeded plots over dedicated
10 GbE. All six runs used three-member batches and wrote the same four files.
Each row is a single run. Timing starts at process launch and ends after the
owner reports durable publication. The timestamps were sampled every 50 ms.
Worker phase timing was enabled for the fixed-format comparisons.

| Active destination writers | Worker output | First group (s) | Full fill (s/member across 12 members) |
| ---: | --- | ---: | ---: |
| 1 | Whole encoded group | 19.2 | 6.57 |
| 2 | Whole encoded group | 38.2 | 6.35 |
| 3 | Whole encoded group | 57.4 | 7.24 |
| 1 | Packed members | 27.7 | 9.54 |
| 2 | Packed members | 29.5 | 7.40 |
| 3 | Packed members | 47.0 | 6.34 |

Three writers helped when the owner assembled packed members. For whole-group
worker assembly, two writers gave the shortest fill and one writer published
the first group earliest. All outputs matched previously full-proof-verified
files by SHA-256, and a fresh full-proof check ran for each configuration.

After the first publication, a separate reference reader opened one of the
new groups and looked up a fresh challenge every 9.375 seconds. Across these
six fills, the 37 open-plus-lookup times had a median of 66.37 ms and a
maximum of 119.9 ms. All finished within the farming model's eight-second
deadline. The reader shared the host page cache with the writer. The live
harvester had zero plots during these tests.

On eight empty HDDs with a 4 GiB owner budget, both scratch formats used
the same two-writer limit, single-member batches, seed and 24 members.
The packed receiver wrote fragment tiles directly to temporary HDD space.
Disk traffic below sums the physical block counters on the eight drives
through process exit, including the concurrent reference reader. Transferred
bytes come from the owner's compute protocol counters. GB are decimal.

| Scratch input | Seconds per published member | GB received/member | GB read from HDDs/member | GB written to HDDs/member |
| --- | ---: | ---: | ---: | ---: |
| Worker-encoded contributions | 10.17 | 1.058 | 1.034 | 2.249 |
| Packed fragments | 10.67 | 1.158 | 2.191 | 3.411 |

The packed format increased transfer size and added a scratch write and
read before compression. Compressed contributions finished the fill 12
seconds earlier. All eight files matched across formats. The 10 GbE
interface averaged 0.871 Gb/s receiving encoded contributions and
0.908 Gb/s receiving packed fragments. Their highest five-second rates
were 1.77 and 1.94 Gb/s, respectively. At one new member every 2.2 s,
these payload sizes would supply an average of 3.85 or 4.21 Gb/s from
one GPU, before network overhead.

With one group per drive and both choices set to `auto`, the planner used
two active destinations and three scratch HDDs per group, with six scratch
HDDs in use at peak. All 24 members arrived as compressed contributions.
The first group was durable in 62.33 s and the last in 244.88 s, or
10.20 s/member.

On the same compressed-scratch workload, using eight codec lanes on the
worker reduced encoding wall time from 2.431 to 1.154 s/member. Member
generation kept its eight-thread grant in both runs and averaged 2.183
and 2.233 s/member, respectively. The time to
publish all 24 members changed from 10.23 to 10.17 s/member. Each setting
ran once. The faster codec freed worker time, with little change in the
time to finish this HDD workload.

With 16 empty HDDs, a 32 GiB owner budget and one three-member group per
drive, automatic construction published its first group in 32.29 s and
its last in 254.46 s, or 5.30 s/member. For a common endpoint across
versions, process exit after the final sync took 257.78 s (5.37 s/member)
with the previous build and 255.11 s (5.31 s/member) with the updated
build. Both runs used the same 48 members with GPU phase timing disabled.
The difference was 1.03% in one run per build.

The updated planner admitted up to four destinations, received 45 packed
members and one complete three-member group, and used zero HDD scratch.
Thirteen generation batches overlapped a destination write. Inbound
Ethernet averaged 1.81 Gb/s and peaked at 3.73 Gb/s over five seconds.
Aggregate disk writes averaged 189 MB/s and peaked at 549 MB/s over
one second. The independent 16-drive sequential test reached 3,328 MB/s.

A four-HDD fill with two groups per drive exercised writer selection across
successive groups. With the same 32 GiB budget, automatic construction
received 21 packed members and one complete three-member group. The first
group was durable at 45.6 s and the last at 154.97 s, or 6.46 s/member.
Across the three automatic fills on four, eight and 16 drives, all 56
reference-reader lookups completed within the eight-second deadline. The
longest open and lookup took 129.51 ms.

#### Plot construction alongside live v1 farming

Gigahorse 2.7.4.giga37 harvested 1,164 k32 C29 plots on six HDDs while
`xchplot2 fill` wrote seeded three-member k=28 groups to the empty HDDs.
Both sets of drives used the same SAS3008 HBA. One RTX 4090 supplied the
members over 10 GbE. GPU phase timing was off for the RAM runs and on
for the scratch runs, matching the empty-farm comparisons. Plotting time
starts at launch and ends when the final group is durable. Harvester times
cover challenges with at least one eligible plot whose lookup started
and finished within the measurement window. The idle window immediately
preceded the 16-HDD RAM fill and matched its duration through process exit.

| Plotting workload | Published members | Seconds/member | Eligible challenges | Median lookup (s) | 95th percentile (s) | Longest lookup (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Idle | — | — | 27 | 0.490 | 3.51 | 3.76 |
| 16 HDDs, 32 GiB RAM | 48 | 5.26 | 26 | 0.504 | 3.24 | 3.67 |
| Same RAM fill, 4 MiB socket receive buffer | 48 | 5.61 | 28 | 0.503 | 3.18 | 3.35 |
| 8 HDDs, 4 GiB RAM, one-member scratch batches | 24 | 9.89 | 25 | 0.532 | 0.914 | 3.24 |
| Same scratch fill, default batch size | 24 | 9.81 | 23 | 0.560 | 3.29 | 3.44 |

All 102 eligible challenges during plotting finished within the farming
model's eight-second deadline. Gigahorse logged six quality-fetch errors
during startup and zero in the timed windows. Each plotting row represents
one complete fill. The 95th percentile uses the nearest rank. Every
published group matched its full-proof-checked reference by SHA-256.
After plotting, Gigahorse's standalone reader verified 13 full proofs
from eight challenges against one of the mounted C29 plots.

The 16-HDD RAM fill took 5.30 s/member in the earlier empty-farm run.
During live farming it averaged 1.83 Gb/s inbound and 190 MB/s of writes
across the plotting HDDs. All six farm HDDs recorded physical reads.
A separate build requested a 4 MiB socket receive buffer for incoming
generation results. It published the first group in 29.89 s and the last
in 269.35 s. With the system default those times were 47.69 s and 252.52 s.
The GPU worker was identical. Automatic scheduling used 15 packed tasks
and one whole-group task with the default buffer, and 28 packed tasks
with the larger buffer. Across sampled transfers of at least 1 GB, Linux
TCP counters attributed 99.2% of sender busy time to the peer's receive-window
limit with the default and 98.2% with the larger buffer.

Both 4 GiB fills used two destinations and three scratch disks per group.
With default batching, one construction batch held two members. The
worker still delivered all 24 compressed contributions individually.
The first group was durable in 52.40 s with default batching and 59.82 s
with the one-member cap. Both runs included ten timed-out remote codec
calibration attempts. Generation completed on the GPU worker in both.

#### Packed worker results during live farming

The storage host received the same seeded three-member k=28 groups over
10 GbE while Gigahorse farmed the six C29 HDDs. The updated receiver can
validate one packed tile on the owner's CPU pool while reading the next
frame. The control used the preceding receiver. Both used one RTX 4090
worker slot on the plotting workstation. Gigahorse's recompute server ran
on the storage host's RTX 3070. Times end at durable publication of the
last group.

| Plotting workload | Members | Control (s/member) | Updated (s/member) | Control fill (s) | Updated fill (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 HDDs, 32 GiB, automatic writers and output | 48 | 5.35 | 4.92 | 256.56 | 236.37 |
| 4 HDDs, one writer, three scratch HDDs, one packed member per batch | 12 | 18.36 | 18.16 | 220.33 | 217.89 |
| 8 HDDs, 4 GiB, automatic writers and compressed scratch | 24 | 9.80 | 9.80 | 235.20 | 235.30 |

The 16-HDD runs assembled 45 packed members in RAM and received one
complete three-member group. The control used 24 generation tasks and the
updated build used 16. Owner processing of received data fell from 63.6 to
31.3 s across all tasks. Average Ethernet receive throughput rose from
1.80 to 1.95 Gb/s and the highest five-second rate rose from 3.71 to
4.73 Gb/s. Writes across the plotting drives averaged 187.1 and 202.9 MB/s
respectively. Before frame reuse was limited to packed output, the same
fill took 235.83 s (4.91 s/member). In that run Linux attributed 99.3%
of sender busy time on the sampled bulk transfers to the receiver-window
limit.

The four-HDD comparison held the task shape fixed. Each build sent the
same 12 packed members in 12 tasks. Owner processing fell from 13.1 to
5.6 s, while the whole fill shortened by 2.44 s.

The 4 GiB runs used two writers and six scratch drives. All 24 members
arrived as compressed contributions, which use a separate receive path.
The table compares runs with identical broker monitoring. Two earlier
development builds took 10.63 and 10.68 s/member. Each had a gap of about
36 s between two consecutive GPU member starts. The monitored control
and final builds both kept that interval below 14 s. Both took
9.80 s/member on the compressed-scratch path. All published groups
matched their full-proof-checked references by SHA-256.

Gigahorse completed all 175 eligible challenges observed across these
seven fills within the eight-second farming deadline. The longest took
3.57 s. During the final 16-HDD fill, one-second GPU samples averaged 5.5%
utilization on the farming RTX 3070 and 43.2% on the plotting RTX 4090.
Both GPUs reached 100% in individual samples. This measurement covers
1,164 v1 plots on six farm drives, with separate GPUs for the two workloads.

#### Compressed scratch with a 4 GiB owner

The storage host built three-member k=28 groups on the empty HDDs while
Gigahorse harvested 1,164 C29 plots on six separate HDDs sharing the HBA.
One RTX 4090 worker delivered compressed contributions over 10 GbE. The
owner had 4 GiB of plotting RAM, eight CPU workers and automatic scratch,
batch and destination selection.

The planner used two destinations at a time, with three scratch drives per
destination on eight HDDs and seven per destination on sixteen. Each row
compares one control fill and one updated fill under similar GPU load.
Timing starts at launch and ends when the last group is durable.

| Plotting HDDs | Members | Control (s/member) | Updated (s/member) | Reduction in elapsed time |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 24 | 10.77 | 8.77 | 18.6% |
| 16 | 48 | 10.52 | 8.18 | 22.3% |

At the same 24-member endpoint, the updated sixteen-drive fill took
8.36 s/member. Generation on eight drives averaged 3.26 s/member in
the control and 3.29 s/member in the updated build. An earlier control
under lighter GPU load generated a member in 2.14 s and published one
in 9.84 s on average.

On sixteen HDDs, the worker spent 30.2 s encoding contributions, down from
58.1 s. The storage host spent 72.6 s processing incoming contributions,
down from 189.0 s. The worker's generation phase averaged 3.22 s/member
in the updated build and 3.24 s/member in the control while other GPU
workloads were active. Optional remote encoding probes timed out on the
single-slot worker in both builds and final packing ran on the storage
host. The updated fill averaged 126 MB/s of physical reads and 275 MB/s
of physical writes across the plotting HDDs and 1.08 Gb/s received on
the NIC. Its best five-second intervals reached 341 MB/s read, 525 MB/s
written and 1.77 Gb/s received.

Every published group matched its full-proof-checked reference by SHA-256.
During the two updated fills, Gigahorse completed 64 challenges with
eligible plots. The longest lookup took 0.87 s. All six farm HDDs served
reads during both fills.

In the eight-HDD control, each published member accounted for 1.03 GB of
physical reads, 2.24 GB of physical writes and 0.994 GB of final plot.
Sustaining one member every 2.2 s at that traffic mix would require
470 MB/s of reads and 1,018 MB/s of writes, including 452 MB/s of final
output, plus 3.85 Gb/s of worker application payload. With two independent
destinations that is 226 MB/s of final writes per HDD. At the 241 MB/s
measured on one output HDD above, final writes alone take 4.13 s/member on
one drive, or 2.06 s/member if two can each sustain that rate. The independent
15-HDD test below sustained 2,127 MB/s read and 1,860 MB/s written
simultaneously. Those rates used eight read-only and seven write-only
spindles. The complete plotting workload also generates, validates and
checkpoints each group.

#### Scratch reservations and parallel checkpoint writes

A follow-up compared commit `596dd0a` with the updated plotter on the same
4 GiB storage host and single RTX 4090 worker. The updated plotter bounds
temporary contributions using the plot format and flushes checkpoint
summaries across independent scratch HDDs. The automatic planner selected
two destinations at a time for the three-member k=28 groups.

| Plotting HDDs | Members | Control (s/member) | Updated (s/member) | Control scratch (GiB/destination) | Updated scratch (GiB/destination) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 24 | 8.74 | 8.31 | 7.08 | 3.45 |
| 16 | 48 | 8.18 | 7.93 | 6.85 | 3.45 |

Time runs from launch through the last durable publication. Scratch is the
mean reported peak reservation per destination. Those peaks can occur at
different times. At the same 24-member endpoint, the updated sixteen-drive
fill averaged 8.19 s/member. The eight-drive run recorded 2,161 file syncs,
down from 4,681. A separate trial using larger GPU requests took 224 s
for 24 members, versus 210 s for the control.

Both sixteen-drive controls measured 8.18 s/member. In the paired runs,
the worker averaged 3.25 s/member generating fragments in the control and
3.23 s/member in the updated build.
Processing and sending together occupied its single slot for 6.25 s/member.
The plotting HDDs averaged 130 MB/s read and 283 MB/s written. The storage
NIC received 1.12 Gb/s, with a best five-second interval of 1.77 Gb/s.
Every updated group matched its full-proof-checked reference by SHA-256.
Across the two updated fills the harvester completed 61 challenges with
eligible plots on the six C29 farm drives. The longest lookup took 0.93 s.

#### Reading pooled contributions ahead of scratch writes

The storage host had a 4 GiB plotting budget and eight CPU workers. One
RTX 4090 worker sent compressed contributions over 10 GbE. Both builds
used the same worker binary, seed and three-member k=28 groups. The
automatic planner used two destinations at a time. Time runs from launch
through the last durable publication.

| Plotting HDDs | Members | Previous owner (s/member) | Read-ahead owner (s/member) | Less time |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 24 | 8.35 | 7.23 | 13.3% |
| 16 | 48 | 7.86 | 6.71 | 14.6% |

With the updated binary on both hosts, repeat fills took 7.25 s/member
on eight HDDs and 6.73 s/member on sixteen. Both sixteen-drive read-ahead
runs reached 24 published members at 7.17 s/member.

GPU generation in the sixteen-drive comparison averaged 3.10 s/member in
the control and 3.09 s/member with read-ahead. Worker processing and
sending together fell from 6.15 to 5.04 s/member. The owner
spent 1.35 s/member reading frames after first output. Overlapping those
reads with validation and scratch writes cut the time waiting for frames
from 1.32 to 0.104 s/member.

The updated sixteen-drive fill averaged 154 MB/s of physical reads and
333 MB/s of writes across the plotting HDDs. The NIC averaged 1.32 Gb/s
inbound and peaked at 1.88 Gb/s over five seconds. Every published group
matched its full-proof-checked reference by SHA-256. During the read-ahead
fills in the table, the harvester farmed 1,164 plots on six additional
HDDs. It completed 54 challenges with eligible plots. The longest took
0.821 s.

With 32 GiB of owner RAM, four HDDs and the same worker, four complete
groups streamed directly to their destinations with zero scratch. The
twelve members took 88.12 s with the existing writer and 88.76 s with
experimental read-ahead. Automatic read-ahead applies to staged contributions.

#### Pooled construction with a memory-limited GPU worker

The storage host and RTX 4090 worker each had a 4 GiB plotting RAM budget.
The worker had one slot, eight CPU threads and a 15 GiB GPU allowance.
Three-member k=28 groups used the same seed and 10 GbE link as the
read-ahead measurements above.

The previous worker requested 4.14 GiB of task RAM for the first member.
With 2.70 GiB available after service reservations, the broker kept that
request queued. Reusing the pinned GPU output lowered the initial task
reservation to 2.11 GiB. Subsequent singleton tasks requested 80.0 MiB
while the worker retained its pinned buffers.

Both planners used the updated worker. The updated planner allowed a second
destination to queue a compressed batch while the first flushed scratch.
Each planner ran once per drive count, in reverse order on sixteen drives.
Times run from launch through the last durable publication. Two HDDs served
as destinations at a time.

| Plotting HDDs | Peak scratch HDDs | Members | Previous planner (s/member) | Updated planner (s/member) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 6 | 24 | 7.18 | 6.93 |
| 16 | 14 | 48 | 6.65 | 6.67 |

Queueing cut the eight-drive time by 3.37%. The sixteen-drive pair showed
no end-to-end gain. At the same 24-member endpoint the updated sixteen-drive
fill averaged 7.14 s/member. Across all 48 members, GPU generation took
3.19 s/member. Worker processing and sending together took 5.03 s/member,
leaving 1.64 s/member outside the active worker. The plotting HDDs averaged
155 MB/s read and 335 MB/s written. The 10 GbE interface received 1.33 Gb/s.

Every output matched its full-proof-checked reference by SHA-256. During
the two updated fills, the harvester farmed 1,164 plots on six separate HDDs.
It completed 51 challenges with eligible plots. The longest took 0.6231 s.

#### Overlapping GPU generation and encoding with the full farm online

The harvester had 17,886 v1 plots loaded. Each fill wrote the same sixteen
three-member k=28 groups on separate empty HDDs. The storage host and the
single-slot RTX 4090 worker each had a 4 GiB plotting RAM budget. The
automatic planner used two destinations and up to fourteen scratch HDDs.
Times start at launch and end at durable publication.

| 48 published members per fill | Previous (s/member) | Updated (s/member) | Previous first group (s) | Updated first group (s) |
| --- | ---: | ---: | ---: | ---: |
| First comparison | 7.10 | 6.32 | 42.90 | 33.22 |
| Repeat | 6.93 | 6.28 | 44.05 | 33.87 |

The updated fills took 10.2% less time per member across the two comparisons.
Each used 16 GPU leases and 16 construction batches. The controls used 48
leases and 47 batches. The GPU generated the next member while the CPU
encoded the previous member from the pinned buffer. Scheduling a destination's
members together let its HDD start packing earlier. Every group matched its
reference by SHA-256. A fresh full-proof check of all 16 distinct groups
validated 777 proofs.

On the repeat the storage host averaged 164 MB/s of plotting-disk reads,
336 MB/s of writes and 1.41 Gb/s inbound on 10 GbE. Its highest five-second
receive rate was 3.80 Gb/s. The single worker spent 3.42 s/member servicing
leases. The remaining 2.85 s/member includes admission, storage work between
leases and the final publication. Gigahorse proof recomputation was active
on both the plotting RTX 4090 and the storage host's RTX 3070.

The harvester completed 64 challenges with eligible plots during the two
updated fills. The longest took 5.16 s. A separate 351 s farming-only
interval had 37 eligible challenges, with a maximum of 5.60 s. Across the
two previous-build fills, one of 74 eligible challenges exceeded eight
seconds, taking 9.05 s.

With a three-writer cap, the 4 GiB owner still admitted two active
destinations and published at 6.30 s/member. An eight-HDD fill with GPU
phase timing enabled published 24 members at 6.82 s/member. Its GPU
pipeline averaged 2.83 s/member, including 2.29 s in T1, T2 and T3 matching.

At an assumed 2.2 s/member, these same files would require 452 MB/s of
final writes and 3.85 Gb/s of compressed worker payload. At the measured
241 MB/s single-HDD sequential write rate, final writes alone take
4.13 s/member on one destination or an ideal 2.06 s/member across two.
Scratch I/O, encoding and publication also have to fit within that interval.

With four plotting HDDs, 32 GiB on the storage host and a 16 GiB worker,
automatic construction used zero scratch in both builds. The previous build
published twelve members at 6.13 s/member and the updated build at
5.69 s/member. The first group arrived at 33.53 s and 46.21 s respectively.

On local NVMe, each build also ran two four-member fills with GPU phase
timing and an 8 GiB plotting RAM budget. The wider GPU match index reduced
combined T2 and T3 matching from 1.97 to 1.40 s/member. Launch-to-exit
times averaged 5.33 and 5.27 s/member respectively, with overlapping run
times. All outputs matched the full-proof-checked reference files.

#### Sizing ordered GPU contributions on scratch HDDs

The storage host and its single-slot RTX 4090 worker each had a 4 GiB
plotting RAM budget. The harvester had 17,886 plots loaded on separate
HDDs. Each GPU lease delivered three members in plot order. The storage
host kept each member's summary as its last compressed chunk arrived.

| Plotting HDDs | Published k=28 members | Previous (s/member) | Updated (s/member) | Time saved | First group, previous → updated (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 24 | 6.72 | 6.49 | 3.4% | 35.31 → 36.06 |
| 16 | 48 | 6.30 | 5.70 | 9.6% | 37.23 → 35.78 |

In the first 16-HDD comparison, sizing reads fell from 33.04 GB to zero
and cumulative planning time fell from 96.97 s to 3.27 s. The plotting
disks reported 49.6 GB of physical reads in both fills. The storage host
saved the time it had spent reading and inspecting validated chunks again.
Final packing still read the contributions and wrote the finished plots.
The optimized files matched the controls by SHA-256 in both comparisons.
Fresh verification of all sixteen groups in the wider fill validated 727
full proofs.

A repeat published at 6.65 s/member. During startup the broker reported a
host-memory wait after the GPU worker retained its output buffer. The first
group took 67 s. Sizing still recorded zero scratch reads.

With the worker reporting available RAM and per-job requirements in the
same revision, the next 16-HDD fill published at 6.00 s/member. The
first group arrived in 36.23 s. Midway through that fill the worker
deferred generation when the host's available RAM fell below its standing
allocation. An eight-HDD fill using the revised worker with the previous
broker published at 6.34 s/member.

The final 16-HDD repeat published at 5.73 s/member. Its first group
arrived in 35.28 s. The worker served all 48 members in 16 leases and
spent 3.48 s/member processing and sending them. The 10 GbE link averaged
1.54 Gb/s inbound and peaked at 3.79 Gb/s over five-second intervals.
Final packing remained the longest storage-host phase.

Across these three fills, the harvester reported 77 completed challenges
with eligible plots. The longest took 6.62 s. It also logged one C3
decoding error while fetching qualities from an existing v1 plot.

#### Encoding k28 contributions

On a Ryzen 9 5950X, one CPU thread re-encoded 8,192 chunks from a
full-proof-checked k28 member 120 times per run. Preserving just the
partially written byte when appending each delta field reduced median
elapsed time from 0.651 s to 0.529 s (18.8%) across four runs per build.
Both builds reproduced the source bytes for every chunk.

On a Core Ultra 7 270K Plus, a separate single-thread test joined 256
synthetic k28 chunks with three members each. Copying the leading
contribution's validated low bits directly reduced median join time by
1.2% for three separately encoded members and 8.7% when all three had
been encoded together. Each build ran ten times, with 768 passes per run,
and reproduced the bytes from encoding the original fragments together.

#### Scheduling GPU workers on different network links

Each `fill` published 48 k=28 members in 16 three-member groups on
separate plotting HDDs. The storage host and each GPU worker had a
4 GiB plotting RAM budget. The RTX 4090 used 10 GbE. The RX 6700 XT
used 1 GbE with temporary SSH forwarding. The harvester had 17,886 v1
plots loaded on separate drives. Automatic construction used up to 14
scratch HDDs and admitted two active destinations. Times include
generation, transfer, scratch assembly and durable publication. The
updated executable ran first with the 4090 alone and then with both GPUs.
The repeat reversed that order.

| GPU workers | Previous (s/member) | Updated (s/member) | Repeat (s/member) | Updated mean (s/member) |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 | 6.01 | 5.89 | 5.62 | 5.76 |
| RTX 4090 and RX 6700 XT | 7.55 | 5.97 | 6.22 | 6.09 |

The updated two-worker fills took 19.3% less time per member on average
than the previous two-worker run.
The 4090 alone was faster in both comparisons.
With both GPUs, the previous scheduler issued 48 single-member leases.
In each updated two-GPU fill, the scheduler delivered 46 members in
16 pipelined leases and the other two in single-member leases. The
Radeon produced three members in the first fill and five in the repeat.
All 16 groups matched the controls byte for byte, and a fresh check of
every group in the first two-GPU fill validated 742 full proofs.

In the first two-GPU fill, the storage host's highest five-second receive rates
were 0.98 Gb/s on 1 GbE and 3.75 Gb/s on 10 GbE. The plotting HDDs averaged
173 MB/s of reads and 353 MB/s of writes, with a five-second peak of
973 MB/s combined. Final packing remained the longest measured storage phase.
Across the two updated two-GPU fills the harvester completed 63 challenges
with eligible plots. The longest took 6.07 s.

#### Delegating by measured completion time

Each fill started with no worker timing history and published the same 48
k=28 members on 16 HDDs. The storage host and each worker had a 4 GiB
plotting RAM budget. The RTX 4090 used 10 GbE and the RX 6700 XT used
1 GbE over SSH. The harvester had 17,886 v1 plots loaded on separate
HDDs. Times run from the start of the fill through durable publication of
the last group.

| Scheduler and workers | First fill (s/member) | Second fill (s/member) | Mean (s/member) |
| --- | ---: | ---: | ---: |
| Previous, both GPUs | 6.14 | 7.26 | 6.70 |
| Measured completion time, both GPUs | 5.87 | 4.96 | 5.41 |
| RTX 4090 alone | 4.69 | — | — |

The two-GPU mean fell by 19.2%. The previous scheduler sent five members
to the Radeon in one fill and eleven in the other. The updated scheduler
sent two to the Radeon and 46 to the 4090 in each fill. Generation and
delivery overlapped across workers for 16.53 s and 16.57 s, respectively.
The first updated fill logged broker decisions. The second used the
production build throughout. The 4090 alone finished fastest in this series.

In the production fill, the storage host's highest five-second receive
rates were 0.98 Gb/s on 1 GbE and 4.14 Gb/s on 10 GbE. Final packing was
the longest measured storage phase in both fills. Both fills reproduced
all 16 full-proof-checked groups byte for byte. Fresh checks of one group
from each fill verified 280 proofs from 100 challenges and 47 proofs from
16 challenges, respectively. Across the two fills the harvester completed
55 challenges with eligible plots. The longest took 6.23 s.

#### Concurrent GPU generation and independent HDD publication

Each fill published the same 48 k=28 members on 16 HDDs while the harvester
farmed 17,886 v1 plots on separate drives. The RTX 4090 used 10 GbE and
the RX 6700 XT used 1 GbE over SSH. Construction, batch size, worker
output and writer count were automatic. Seconds per member divide the time
from launch to durable publication of the last group by 48. Each comparison
used the same build for both GPU configurations. The 32 GiB repeat and the
4 GiB runs included the updated scratch reservation for the next writer.

| Plotting RAM (owner / each worker) | RTX 4090 (s/member) | RTX 4090 and RX 6700 XT (s/member) |
| --- | ---: | ---: |
| 32 / 10 GiB, both GPUs first | 4.83 | 4.09 |
| 32 / 10 GiB, 4090 first | 4.73 | 4.38 |
| 4 / 4 GiB, both GPUs first | 4.46 | 5.04 |

With `fill -M 32G` on the storage host and `compute worker --host-bytes 10G`
on each worker, neither GPU configuration needed HDD scratch. Adding the
Radeon reduced mean time per member by 11.4%, from 4.78 s to 4.23 s.
In both fills the 4090 delivered 40 members and the Radeon delivered
eight. Generation and delivery overlapped across workers for 156 s and
155 s. The 4090 alone published the first group sooner in both comparisons
(30.30 s and 29.10 s, versus 39.14 s and 39.74 s with both GPUs).

With 4 GiB on the owner and each worker, automatic construction admitted
two destinations and used up to 14 scratch HDDs. The Radeon delivered six
members. Two HDDs packed simultaneously for 24.6 s, and the last
group began receiving members while the preceding group was still packing.

Every output matched its full-proof-checked reference by SHA-256. A fresh
full-proof check sampled one group from each fill. Across these six fills,
the harvester completed 141 challenges with eligible plots. The longest
took 6.97 s.

#### Choosing a delivery route on the live farm

Each fill started with a fresh broker and published the same 48 k=28 members
as 16 three-member groups on separate HDDs. The RTX 4090 used 10 GbE and
the RX 6700 XT used 1 GbE over SSH. The harvester had 17,886 v1 plots
loaded on separate disks. Times cover launch through durable publication of
the last group. All 16 SHA-256 hashes matched full-proof-checked references.
A fresh full-proof check sampled one group per fill. One earlier control
was repeated after its log missed two publication events.

| Owner / worker RAM cap | RTX 4090 alone (s/member) | Both GPUs (s/member) | Both GPUs, repeat (s/member) |
| --- | ---: | ---: | ---: |
| 4 / 4 GiB, HDD scratch | 4.49 | 4.61 | — |
| 32 / 10 GiB, no HDD scratch | 4.83 | 3.88 | 3.85 |

These comparisons ran with the workstation tests stopped and automatic
construction, batching, output and writer count. With 32 GiB the two GPUs
worked concurrently. The Radeon delivered seven members in the first run
and six in the repeat. The two fills averaged 3.87 s/member, 19.8% less
time than the 4090-only control. In the first two-GPU fill, the 4090 spent
180.4 of 186.4 seconds generating or delivering members, and the 10 GbE
interface averaged 2.11 Gb/s received. The 4090 alone published the first
group in 28.35 s. The two-GPU runs took 41.20 s and 40.14 s to publish
their first groups.

Before the controller used measured packed delivery times to size
multi-worker batches, an otherwise comparable run took 4.79 s/member
on the 4090 and 4.57 s/member with both GPUs. In the dual-GPU run,
45 members arrived in singleton leases. A control that forced encoded output
and two HDD writers took 9.37 s/member, with six entire groups streamed
from the Radeon.

The 4 GiB comparison exercised the scratch contribution path before the
resident packed-batch change. Two destinations could receive or pack at
once, with up to 14 scratch HDDs. The 4090 delivered 46 members and the
Radeon delivered two. The 16 HDDs averaged 224 MB/s read and 457 MB/s
written, and the 10 GbE interface averaged 1.84 Gb/s received from launch
through process exit.

In a separate cold-start comparison on scratch, limiting the first request
per ready destination to a singleton reduced the two-GPU time from 5.10
to 4.60 s/member. In a later fill the broker sent two batches of two
members to the Radeon shortly before the 4090 became available. Using the next
larger measured independent batch to bound the wait reduced the two-GPU
time from 5.37 to 5.11 s/member on the repeat.

Available worker RAM also changed the result. During a 32 / 10 GiB
4090-only control with other CPU tests running, three singleton requests
waited for host memory. The fill took 6.44 s/member.

Across the five timed fills in the table, the harvester logged 114 completed
challenges with eligible plots. The longest lookup took 7.25 s.

#### Packing fixed-size groups from HDD scratch

Each fill published 48 k=28 members as 16 three-member groups. The storage
host and each worker had a 4 GiB plotting RAM cap. The RTX 4090 used 10 GbE
and the RX 6700 XT used 1 GbE over SSH. The harvester had 17,886 v1 plots
loaded on separate drives. Automatic construction admitted two destinations
and used up to 14 scratch HDDs.

With other workstation work paused, the time from launch to durable
publication of the last group gave the following results.

| GPU workers | Previous (s/member) | Updated (s/member) |
| --- | ---: | ---: |
| RTX 4090 | 4.49 | 4.43 |
| RTX 4090 and RX 6700 XT | 4.66 | 4.66 |

The two-GPU fills finished in 223.9 and 223.7 s respectively. Earlier
4090-only fills took 4.40 s/member with the previous build and 4.45
with the update. The solo timings overlap across the two comparisons.

The previous build wrote 3.22 GB of staging summaries per fill. Committing
the sealed runs directly in the packing checkpoint eliminated those writes.
Packing still read 49.6 GB per fill. On the updated two-GPU run with the
workstation available, the plotting disks averaged 221 MB/s read and
437 MB/s written. The 10 GbE interface averaged 1.82 Gb/s received.
These counters cover launch through process exit. Final packing on two
destinations overlapped for 17.8 s during that fill.

In a separate run with workstation memory pressure, available host RAM
on the 4090 worker dropped to 588 MiB. The Radeon delivered 18 of the
48 members, and durable publication took 7.43 s/member. With the
workstation available, the Radeon delivered two members.
All 16 groups matched full-proof-checked references by SHA-256 in every
fill. Across the four fills in the table the harvester completed 92
challenges with eligible plots. The longest lookup took 6.32 s.

A separate two-GPU trial admitted four destinations early under the same
4 GiB limits. It received 39 members as packed fragments and nine as
compressed scratch contributions. Publication took 10.35 s/member,
compared with 4.83 s/member in its two-destination control.

#### Choosing GPU contribution batches with more worker RAM

The storage host's plotting budget stayed at 4 GiB while each GPU worker received
10 GiB. Each fill started with a fresh broker and published the same
48 members on the 16 HDDs. More worker memory made joint encoding possible.
The previous owner then stopped requesting independently pipelined batches.

| Two-GPU delivery policy | Seconds per member | Members in independent batches | Members from RTX 4090 / RX 6700 XT |
| --- | ---: | ---: | ---: |
| Previous automatic choice | 6.53 | 0 | 40 / 8 |
| Prefer independent delivery whenever available | 4.54 | 43 | 46 / 2 |
| Choose by measured delivery time | 5.38 | 28 | 41 / 7 |
| Measured delivery time, fresh-broker repeat | 5.39 | 27 | 42 / 6 |

Choosing between the two delivery modes reduced mean time per member by
17.5% against the previous two-GPU run. The diagnostic policy that
preferred independent delivery throughout was faster on these links.
The 10 GbE interface averaged 1.40 Gb/s received in the first measured
run and 1.87 Gb/s with the diagnostic policy. In that first measured run
the plotting drives averaged 191 MB/s read and 378 MB/s written.

The 4090 alone took 5.62 s/member at the same 10 GiB worker cap. Its
leases alternated between two and four members across the two active
destinations. Adding the Radeon reduced mean time per member by 4.2%.
At 4 GiB the solo control received three members per lease. Returning
both workers to 4 GiB with measured selection gave 4.77 s/member.

All 16 files matched the full-proof-checked references in each fill, and
a fresh full-proof check sampled one group per fill. Across the two
automatic two-GPU runs at 10 GiB, the harvester logged 56 challenges with
eligible plots. One lookup in the first run took 8.08 s, above the
eight-second farming target. The longest lookup in the repeat took 4.62 s.

#### Keeping the faster GPU available under host memory pressure

Two fills used the same 48 k=28 members, 16 plotting HDDs, 4 GiB owner
budget and 10 GiB worker budgets. A 5 GiB Linux memory limit applied to
the RTX 4090 plotting process in both fills. The RX 6700 XT and the
harvester ran with their usual limits. Each fill started with a fresh
broker and used automatic batch selection. The owner and 4090 worker
changed builds between fills. The Radeon used the same build for both.

The previous build published the last group in 662.9 s (13.81 s/member).
After its first member, the 4090 kept its pinned host output and advertised
only one byte of task RAM. The Radeon delivered the remaining 47 members
over its 1 GbE link. Releasing the idle output when the next member would
otherwise be ineligible, together with the revised batch selection, reduced
publication time to 253.4 s (5.28 s/member), a 61.8% reduction under this limit.
The 4090 delivered 42 members and the Radeon six. The first group was
durable at 39.9 s, compared with 63.2 s in the control. The updated run
averaged 1.47 Gb/s received on the 10 GbE link, with 195 MB/s read and
386 MB/s written across the 16 plotting disks from launch through exit.

With the additional process limit removed, the same build on both GPUs
published in 5.45 s/member (43 members from the 4090 and five from the
Radeon). The 4090 alone took 4.47 s/member. In the two-GPU run, jointly
encoding and delivering three members on the Radeon took 55.7 s. The
owner also opened a third destination near the end. Its scratch and final
output shared one HDD, and it packed in 25.7 s with two codec workers.
In the 5 GiB run the same group used seven separate scratch HDDs and
four codec workers, and packed in 11.3 s. The 10 GbE interface averaged
1.45 Gb/s received in the ordinary two-GPU run.

Holding the two-GPU fill to two writers and encoded contributions gave
every destination seven scratch HDDs. Individual packs took 11.3–14.0 s.
The Radeon delivered seven members, including five in two jointly encoded
batches that occupied it for 92.8 s. The full fill took 261.5 s
(5.45 s/member), within 0.2 s of the automatic two-GPU run.

All 16 groups in each fill matched full-proof-checked references by
SHA-256. A fresh 100-challenge check on one group from the updated 5 GiB
run validated 287 full proofs. The harvester had 17,886 plots loaded
throughout. Its longest eligible lookup during the two automatic
two-GPU runs on the updated build was 7.43 s.

#### Smallest GPU delivery on the 1 GbE route

The smallest GPU generation request is one plot ID. Across six individual
k=28 requests in two scratch fills, David received and validated each
RX 6700 XT member in 18.47–20.10 s (mean 19.17 s). The Radeon worker
averaged 9.79 s outside its sends and 9.27 s sending the encoded output
over 1 GbE. These intervals include worker encoding and socket
backpressure. The owner still had to pack and publish the group on HDD.
Five further singletons averaged 1.058 GB of framed output each. At the
nominal 1 Gb/s line rate those bytes alone take 8.46 s. One request spent
44.45 s waiting for assignment and 18.57 s in service, taking 63.02 s
from the owner's request to validated delivery.

An isolated run on Jen timed three k=28 plot IDs with the same worker and
eight codec threads. Generation took 10.37 s for the first member and
9.02 s and 9.03 s for the next two. Encoding took 0.473–0.500 s per member.
With a local sink, the two warm requests averaged 9.77 s including setup
and output handling.

The Radeon delivered three independently encoded members in one request
in 37.25 s, or 12.42 s/member. The RTX 4090 alone averaged 3.03 s/member
during its 16 three-member deliveries over 10 GbE. Its full 48-member
fill on 16 HDDs took 199.35 s, including scratch I/O and publication.

With the updated scheduler and worker memory accounting, David published
the same 48 members as 16 three-member groups. The owner had a 4 GiB
plotting budget and each worker had 10 GiB. Each row is one fill.

| GPU workers | HDD writer setting | Members from RTX 4090 / RX 6700 XT | Last group published | Seconds per member |
| --- | --- | ---: | ---: | ---: |
| RTX 4090 | Automatic | 48 / 0 | 206.55 s | 4.30 |
| RTX 4090 and RX 6700 XT | Automatic | 45 / 3 | 226.51 s | 4.72 |
| RTX 4090 and RX 6700 XT | Up to three, encoded contributions | 47 / 1 | 213.90 s | 4.46 |

All three fills used two active destinations and up to 14 scratch HDDs.
In the automatic two-GPU fill, the Radeon delivered three singletons and
the two workers overlapped for 46.7 s. Two destinations packed
simultaneously for 29.9 s while neither GPU was delivering. The 4090
served 20 requests in the two-GPU fill and 17 when it ran alone.

Raising only the owner budget to 32 GiB let automatic construction publish
without HDD scratch. The 4090 alone used three active destinations and
finished in 216.36 s (4.51 s/member). Both GPUs used five destinations
and finished in 180.09 s (3.75 s/member), a 16.8% reduction in elapsed
time. The 4090 delivered 44 members and the Radeon four, with 76.0 s of
concurrent delivery. The first group reached disk at 28.8 s with the 4090
alone and 40.5 s with both GPUs.

Every group in these five fills matched its full-proof-checked reference
by SHA-256. A fresh proof check also sampled one group in each fill. The
harvester reported 17,886 plots throughout. Its longest eligible lookup
during these fills took 5.34 s.

#### Sizing out-of-order GPU contributions on HDDs

David published the same 48 k=28 members in 16 three-member groups with
a 4 GiB owner budget and 10 GiB per GPU worker. Automatic construction used
two active destinations and up to 14 scratch HDDs. Elapsed time ends when
the last group reaches disk.

| Build and workers | RTX 4090 / RX 6700 XT members | Last publication | Seconds per member | Scratch read for sizing |
| --- | ---: | ---: | ---: | ---: |
| Previous, RTX 4090 | 48 / 0 | 206.97 s | 4.31 | 0 |
| Updated, RTX 4090 | 48 / 0 | 203.71 s | 4.24 | 0 |
| Previous, both GPUs | 47 / 1 | 204.35 s | 4.26 | 0.96 GB |
| Updated, both GPUs | 47 / 1 | 213.39 s | 4.45 | 0 |

In the mixed fills, the 4090 took 153.39 s to deliver 19 requests on the
previous build and 161.47 s for 21 requests on the updated build. Packing
CPU time across the 16 groups fell from 218.30 to 214.26 s. Each row is one
fill. An intermediate build with batched packing still reread 7.15 GB to
size groups when the Radeon delivered five members. That fill took 240.78 s
(5.02 s/member).

All 16 groups in each fill matched their full-proof-checked references by
SHA-256, with a fresh full-proof sample from one group per fill. The live
harvester reported 17,886 plots. Its longest eligible lookup took 7.48 s
during the updated single-GPU fill and 5.34 s during the updated mixed fill.

To isolate construction from GPU and network timing, two scratch HDDs
received the same three synthetic k=28 member streams in reverse membership
order. A third HDD received the finished group. Each member repeated a
fixed encoded chunk with 64 fragments. Each build ran twice, reversing
the build order for the second pair. All four output hashes matched.
The previous build reread 2.06 GB to size each group. The updated build
read zero bytes for sizing. Mean time from sealing the input to the packing
checkpoint fell from 7.16 s to 82.9 ms. Mean time from reserving scratch
through publication fell from 27.79 to 20.69 s per group, or from 9.26 to
6.90 s/member.

At k=28, a group has 4,194,304 chunk lengths. On the plotting workstation,
computing the size of a three-member index took 4.44 ms, down from 9.34 ms
to encode it and measure the result. Both paths returned 3,971,787 bytes
for the same synthetic lengths. These are medians over 13 trials with
alternating execution order. Exact sizing in merged construction also
avoids a 32 MiB temporary length table at k=28.

#### Overlapping GPU deliveries with HDD packing

David filled 16 empty HDDs with the same 48 k=28 members, three per group,
using a 4 GiB owner budget and 10 GiB per GPU worker. The RTX 4090 sent
results over 10 GbE and the RX 6700 XT used its 1 GbE route over SSH.
Each fill started with a fresh broker. Times include publication of the
last group on disk. Construction, scratch selection and writer count were
automatic.

| Build and workers | RTX / Radeon members | Peak active destinations | First group | Last group | Seconds per member |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous, RTX 4090 | 48 / 0 | 2 | 21.76 s | 204.27 s | 4.26 |
| Previous, both GPUs | 47 / 1 | 2 | 33.61 s | 213.41 s | 4.45 |
| Intermediate, RTX 4090 | 48 / 0 | 2 | 21.47 s | 197.80 s | 4.12 |
| Intermediate, both GPUs | 46 / 2 | 4 | 28.09 s | 172.19 s | 3.59 |
| Updated, RTX 4090 | 48 / 0 | 5 | 22.26 s | 152.53 s | 3.18 |
| Updated, both GPUs | 41 / 7 | 6 | 33.03 s | 151.04 s | 3.15 |
| Updated, both GPUs, repeat | 43 / 5 | 5 | 23.02 s | 160.16 s | 3.34 |
| Updated, RTX 4090, repeat | 48 / 0 | 5 | 22.26 s | 169.83 s | 3.54 |

A separate 4090-only control with four HDD writers and encoded output
took 174.37 s (3.63 s/member). Much of the initial improvement came from
concurrent HDD construction. In the previous mixed fill both GPUs had no
generation or delivery in progress for 28.3 s while two HDD groups packed.
That interval fell to 7.56 s in the intermediate mixed fill.

On the Radeon, two singleton requests each returned 1.06 GB and took
18.8 and 20.3 s from assignment through delivery. A request for three
independently encoded members took 37.8 s, or 12.6 s/member on that worker.
One development run placed a three-member request on the Radeon about
half a second before the 4090 finished its current batch. The 4090 finished
all its work at 169.91 s, the Radeon delivered at 197.03 s and the last
group reached disk at 209.18 s.

The updated controller admits more scratch receivers while final HDD writes
are in flight. The broker uses the estimated time remaining on independent
destinations to decide when a slower worker can finish a whole batch.
Adding the Radeon reduced elapsed time by 0.97% in the first pair and
5.69% when the order was reversed. Across both pairs the averages were
3.36 s/member with the 4090 alone and 3.24 s/member with both GPUs.
A background reference build overlapped the opening 13 seconds of the
first two-GPU run. The reverse-order pair ran after that build finished.
All 16 group hashes matched the full-proof-checked references. A fresh
100-challenge sample validated 299 full proofs in the first one-GPU fill
and 292 in the first two-GPU fill.

#### An Intel Arc worker over Wi-Fi

An Arc B580 worker generated the same k=28 member for a local SSD and
for David. Its 2.4 GHz Wi-Fi link reported a transmit rate of 143.3 Mb/s.
The worker had one GPU slot, an 8 GiB host RAM cap and a 10 GiB GPU RAM
cap. Both owners published the same 1,002,434,213-byte file and matched
its SHA-256. Verification ran after publication.

| Storage owner | Worker processing | Worker sending | Launch to clean exit | Full proofs in 100 challenges |
| --- | ---: | ---: | ---: | ---: |
| Arc host, local SSD | 17.98 s | 1.27 s | 19.36 s | 110 |
| David, direct Wi-Fi delivery | 16.29 s | 78.65 s | 95.48 s | 117 |

The Wi-Fi transfer carried 1.058 GB of contribution data, averaging
107.56 Mb/s over the worker's send interval. Even at the reported
143.3 Mb/s radio rate, those bytes alone would take 59.04 s. During its
busiest 12-second sample the wireless interface transmitted 116.05 Mb/s.

Two independent storage owners on David submitted a four-member k=28 group
and then a single member while the RTX 4090 was busy. Outputs went to tmpfs
to isolate scheduling and transfer time. With both routes measured, the
previous broker sent the singleton to the idle Arc. It took 97.14 s from
submission to publication. The updated broker waited for the RTX to deliver
the first owner's group. The singleton took 27.38 s in the first trial and
26.93 s after deployment on David. Each output matched its CPU reference
and passed full-proof verification.

With an empty timing history, the deployed broker sampled the Arc and took
96.73 s to finish all five members (19.35 s/member). After calibration, it
finished all five in 27.45 s (5.49 s/member). The four-member group reached
publication in 19.14 s and 19.02 s, respectively.

A separate test on that broker filled 16 physical HDDs with one three-member
k=28 group per drive. Each run used 32 GiB of owner RAM, incremental
construction, zero scratch drives and the same 48 members. Times include
durable publication of the last group. The harvester had 17,886 plots loaded.

| Storage owner | Accepting GPUs | Time for 48 members | Seconds per durable member |
| --- | --- | ---: | ---: |
| Previous | RTX 4090 | 170.23 s | 3.55 |
| Previous | RTX 4090 and Arc B580 | 181.13, 180.61 s | 3.77, 3.76 |
| Batch during the Arc's first transfer | RTX 4090 | 169.77 s | 3.54 |
| Batch during the Arc's first transfer | RTX 4090 and Arc B580 | 175.26, 175.66 s | 3.65, 3.66 |
| Count confirmed worker time and compare the last batches | RTX 4090 and Arc B580 | 170.05, 170.34 s | 3.54, 3.55 |
| Same final owner | RTX 4090 | 171.33 s | 3.57 |

In each two-GPU fill in the table, the Arc delivered one member over Wi-Fi
and the RTX delivered 47 over 10 Gb Ethernet. The final owner kept the RTX
on three-member requests through the end of both fills. The Arc finished
its member at 110.45 and 111.20 s. Both runs had zero retries, matched all
16 reference hashes and passed a fresh full-proof sample. The owner's
10 Gb interface received 334 MB/s on average, with five-second peaks of
561 and 572 MB/s. The 16 HDDs together wrote 281 MB/s on average and
peaked at 616 and 631 MB/s over five seconds.

The two-GPU fills finished 0.99 to 1.28 seconds ahead of the RTX-only fill
on the final executable. Earlier RTX-only controls ranged from 169.77 to
170.23 seconds. Across the paired two-GPU runs, the owner changes cut the
mean time per durable member by 0.22 seconds (5.9%).

Two intermediate fills experienced an Intel compute-engine reset and its
aftermath. Both recovered, published the same reference files and took
219.36 and 189.20 s with two retries apiece. The Arc benchmark worker was
restarted while idle before the final two-GPU measurements.

David also filled the same 16 HDDs through a persistent broker with the Arc
and the two wired GPUs enrolled. With a 4 GiB owner budget, the original
controller ran all 48 members as singleton requests when the Arc was
available. The RTX 4090 produced 44 members, the Radeon two and the Arc two.
Only two destinations were active at once. Last publication took 349.13 s
(7.27 s/member). With the Arc drained, five destinations were active and
the two wired GPUs finished in 190.08 s (3.96 s/member). The Radeon
received a late two-member joint encoding job after its first two deliveries
had both been independently encoded batches.

The updated owner queried the batch width available to a compatible GPU
when several destinations had work. With all three workers enrolled, the
broker reported a common independent width of zero and an eligible width
of four. The following fills used the same executable on all four hosts.
Each started with a fresh broker timing history. Member counts are in RTX
4090 / RX 6700 XT / Arc B580 order. Times stop at the last durable HDD
publication.

| Owner RAM | Workers | Members by GPU | Peak active destinations | First group | Last group | Seconds per member |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 GiB | RTX only | 48 / 0 / 0 | 5 | 21.61 s | 158.29 s | 3.30 |
| 4 GiB | RTX and Radeon | 41 / 7 / 0 | 6 | 31.71 s | 148.71 s | 3.10 |
| 4 GiB | RTX, Radeon and Arc | 40 / 7 / 1 | 7 | 25.32 s | 160.17 s | 3.34 |
| 4 GiB | RTX, Radeon and Arc, repeat | 40 / 7 / 1 | 6 | 24.06 s | 156.54 s | 3.26 |
| 4 GiB | RTX and Radeon, repeat | 41 / 7 / 0 | 4 | 23.92 s | 147.81 s | 3.08 |
| 32 GiB | RTX only | 48 / 0 / 0 | 3 | 28.75 s | 209.00 s | 4.35 |
| 32 GiB | RTX and Radeon | 44 / 4 / 0 | 5 | 38.29 s | 164.39 s | 3.42 |

With 32 GiB, both fills assembled their groups in RAM. Adding the Radeon
reduced time to the last group by 44.61 s (21.34%). On the 4 GiB wired
fill the owner used HDD scratch. Its 10 GbE interface averaged 2.45 Gb/s
of received data and peaked at 5.05 Gb/s over five consecutive sample
intervals of about a second each.

The order of the two- and three-worker scratch fills was reversed for
the repeat. The wired pair averaged 3.09 s/member and all three averaged
3.30 s/member. In both automatic three-worker fills the wired GPUs returned
42 members in independent batches while the Arc delivered one member at
about 103 s. Fourteen of the 16 groups wrote contributions to a single
scratch spindle in each of those fills. Four groups did so in the first
wired fill.

Limiting the three-worker fill to four simultaneous writers with
`--compute-writers 4` increased the time to 228.63 s (4.76 s/member).
The owner received 12 packed members and 36 contributions, including 27
members produced in independent batches.

In all eight fills, the SHA-256 on each of the 16 destination drives
matched the reference group for that drive, and fresh proof checks passed.
David's harvester reported 17,886 farm plots throughout the 147 eligible
challenge samples recorded during these fills. The longest recorded lookup
took 7.24 s.

#### A Windows Radeon over 1 Gb Ethernet

David published 48 k=28, strength=2 members as 16 three-member groups on
separate 18 TB HDDs. The owner used 32 GiB for construction. The RTX 4090
used 10 Gb Ethernet. The Windows RX 6700 XT used its 1 Gb Ethernet link
through authenticated SSH forwards and kept plot data in RAM. The existing
mainnet harvester reported 17,886 plots on its separate farming HDDs.

| Owner and workers | RTX / Radeon members | Last durable group | Seconds per member | Harvester p95 |
|---|---:|---:|---:|---:|
| Previous owner, RTX only | 48 / 0 | 173.49 s | 3.61 | 18.62 s |
| Earlier payload writeback, RTX only | 48 / 0 | 170.14 s | 3.54 | 13.03 s |
| Previous owner, RTX and Radeon | 43 / 5 | 170.15 s | 3.54 | 14.79 s |
| Revised batching and writeback, RTX and Radeon | 45 / 3 | 158.09 s | 3.29 | 15.08 s |

Each row is one November 23, 2026 run. The Linux worker service was
already running, and each Radeon run enrolled a fresh Windows worker
from `69a33e4`. The candidate owners were built from `8cdcd21` with
the indicated changes. Each run produced the same 47,731,349,036 bytes. Every file matched
its independently proof-checked reference, and each run also checked fresh
full proofs. The revised owner received its first two-member RTX result
at 28.56 s. The previous owner received its first
multi-member RTX result at 34.82 s.

In a farming-only window before the plotting series, the
harvester recorded 17 lookups with a median of 9.23 s and p95 of 19.86 s.
Farming was already exceeding the eight-second target during that window.

In the RTX control the last group spent 3.56 s hashing its three members
and 15.97 s writing its payload, including 11.83 s inside sync calls. The
final file commit took 0.13 s. Renaming the file and syncing its directory
took 0.013 s. With earlier writeback as the only change, the last group's
payload took 15.60 s. Time inside its sync calls fell to 0.22 s, while
time in writes rose from 0.36 to 12.10 s. The full 48-member run gained
3.36 s. These timings use the owner's monotonic clock.

#### Time to complete a job

These matched runs used 32 GiB of owner RAM and three k=28
members per file. Time runs from launching `fill` to successful process
exit. Each pair wrote identical files. Reference hashing and full-proof
checks ran afterward.

| Job and change | Members | Previous job | Updated job | Previous s/member | Updated s/member |
|---|---:|---:|---:|---:|---:|
| One file, three GPUs available, measured route selection | 3 | 67.10 s | 19.04 s | 22.37 | 6.35 |
| Four files on one HDD, RTX only, packed transfers with next-file generation in RAM | 12 | 111.78 s | 86.33 s | 9.32 | 7.19 |
| Three files per HDD on two drives, RTX only, scheduling from measured checkpoint time | 18 | 87.25 s | 80.36 s | 4.85 | 4.46 |
| Four files per HDD on eight drives, RTX only, updated batching and writeback | 96 | 406.59 s | 387.77 s | 4.24 | 4.04 |

For the one-file job the previous scheduler sent all three members to
the Windows Radeon. After the broker had measured the RTX route,
the revised scheduler sent the entire three-member request to the RTX.
Both jobs published the same 2,983,316,893-byte file and passed fresh
full-proof checks. An earlier RTX-only control finished the same file
in 19.20 s.

In the single-HDD comparison, all nine members of the three subsequent
files arrived while the preceding file was being written. The eight-drive
jobs each published 95,459,665,643 bytes.

A separate four-file job measured the cost of hashing the next file's
packed members on that HDD. The RTX worker delivered the same 12 members
in each run. The owner used spare CPU time to hash members already in RAM
while the preceding file was being written. With the original and updated
owners run in opposite orders across two pairs, the original took 86.04
and 86.41 seconds (7.17 and 7.20 s/member). Overlapping the hashing took
79.42 and 75.64 seconds (6.62 and 6.30 s/member). All four runs published
the same 11,932,394,244 bytes and passed fresh full-proof checks.

#### Whole-job cost of changing the compute pool

A separate control used ordinary pool enrollment, 32 GiB of owner RAM,
eight owner CPU threads and three k=28 members per file. The RTX 4090
used 10 GbE, the Windows RX 6700 XT used 1 GbE, and the Arc B580 used
Wi-Fi. The harvester had 17,886 plots loaded on separate drives. Times
include process startup and exit. Each configuration produced the same
files as its matched control and passed full-proof verification.

| Work requested | RTX only | RTX and Radeon | RTX, Radeon and Arc |
|---|---:|---:|---:|
| One file on each of 16 HDDs, 48 members | 172.27 s (3.59 s/member) | 165.24–166.77 s (3.44–3.47 s/member) | 172.23–175.72 s (3.59–3.66 s/member) |
| Four files on each of eight HDDs, 96 members | 377.73 s (3.93 s/member) | — | 387.56 s (4.04 s/member) |

Each 48-member run with the wired pair credited five members to the Radeon.
Across the two three-worker runs the Radeon delivered six and seven members,
the Arc delivered zero, and the owner recorded 30 and 39 failed assignments.
The RTX returned every member individually in those runs. With the wired
pair it also returned batches of two or three. The Radeon transferred packed
payloads at 0.92–0.93 Gb/s while delivering them.

In a separate 48-member run a Radeon joining during plotting delivered
five members. The job finished in 194.12 s. When it disconnected during
an active transfer, the RTX finished all 48 in 276.66 s. The next RTX
assignment spent 115.20 s waiting for its grant and 3.51 s executing and
transferring the result. The files matched in both cases.

On one HDD, a 12-member control with all three GPUs available waited for
a packed member from the Arc and finished in 186.87 s (15.57 s/member).
The revised scheduler compared the cost of trying a new route with the
remaining work on that HDD. After measuring a whole-group delivery from
the RTX, it assigned all four groups to that worker. The job finished in
75.99 s (6.33 s/member). A repeat after the short-term route timings
expired took 76.14 s (6.35 s/member). All three runs produced the same
11,932,394,244 bytes. For the Arc's packed member, the owner measured
15.46 s for execution and 95.73 s for transfer over Wi-Fi. A separate
Windows Radeon test transferred a comparable 1.16 GB member over its
wired link in 9.94 s.

In a later 16-HDD comparison the broker could keep a measured RTX batch
intact while a different destination tried the Radeon. The wired pair
finished 48 members in 158.95 s (3.31 s/member). With the Arc also
enrolled, the same job took 158.51 s (3.30 s/member). Both jobs credited
43 members to the RTX and five to the Radeon, wrote the same 16 files
and passed fresh full-proof checks. The Arc reported a lost device
during the second job and delivered zero members.
In the wired run the Radeon averaged 0.92 Gb/s over its accepted
transfers. The RTX averaged 4.02 Gb/s on its 10 GbE link. The 16
destination disks together averaged 302 MB/s of writes and reached
573 MB/s over a five-second interval.

The machines remained in ordinary use. These fills used the generic
harvester profile while David's mainnet harvester continued farming.
Of 18 eligible lookups during the wired run, 16 exceeded the eight-second
target and p95 was 20.53 s. All 20 during the three-worker run exceeded
the target and p95 was 38.96 s. An earlier farming-only window had p95
19.86 s across 17 eligible lookups.

In a second block, with a fresh Windows worker registration, the
three-worker job ran first and finished in 160.67 s (3.35 s/member).
The wired pair took 162.24 s (3.38 s/member). Both jobs matched the
reference files and passed fresh proof checks. The Arc again delivered
zero members. David's farming GPU showed 99% median utilization across
ten samples in each job. Before this block, 27 of 34 eligible farm
lookups exceeded eight seconds and p95 was 15.22 s. During the two
jobs p95 was 16.75 s and 26.59 s, respectively. The plotter's share of
the delay in these separate windows is unknown.

On eight HDDs, the earlier scheduler finished a 96-member job in
397.28 s (4.14 s/member). With the changes to batch placement and retries,
the job took 347.11 s (3.62 s/member). Both runs delivered 83 members
from the RTX and 13 from the Radeon. Larger packed batches reduced the
number of successful generation tasks from 80 to 41. The earlier run
retried seven tasks and the later one retried zero. All 32 files matched
byte for byte and both runs passed fresh proof checks. Measured at the last
durable file, the two runs took 396.91 and 346.87 s (4.13 and
3.61 s/member).

In a 48-member comparison, the Windows worker disconnected during its
first active transfer, before any file had been published. The earlier
run finished in 262.84 s (5.48 s/member). Allowing the broker to reassign
the remaining work as soon as the owner reported the loss shortened the
job to 175.45 s (3.66 s/member). The disconnect occurred 6.45 s and
14.86 s after launch, respectively. Both runs retried once, and the RTX
delivered all 48 members. All 16 final files matched.

#### Choosing how the last group reaches its HDD

A 16-drive experiment changed the RTX 4090's output format for the final
three members of a 48-member job. In the first pair, the preceding 45
members followed the same assignments in the same order. The same three
other HDDs were still writing when the owner requested the final group.

| Delivery of the last group | Worker lease | Grant to last durable file | Whole job | Seconds per member |
|---|---:|---:|---:|---:|
| Packed members, assembled by the owner | 8.48 s | 29.32–29.38 s | 170.11 s | 3.54 |
| Encoded group, streamed into the destination | 17.96 s | 18.22–18.27 s | 159.19 s | 3.32 |

Streaming took longer on the worker. It let the owner write the file while
receiving it, shortening the time to publication. The other drives finished
about 16.6 seconds after the grant in both runs. In reverse order, streaming
took 162.03 seconds and packed delivery took 169.45 seconds. Two other HDDs
were writing at the streamed grant and three at the packed grant in that
pair. All four runs produced 16 identical files and freshly verified full
proofs from the final file. The first pair measures the effect of changing
the final delivery on this configuration. The owner had observed only
packed deliveries when it made that choice. During that pair, David's
farming GPU showed median utilization of 98.5% and 99%, respectively,
from samples taken every 15 seconds. The harvester took over eight
seconds for 16 of 18 eligible lookups in the packed run and 15 of 16
in the streamed run.

A separate RTX-only test gave each of 16 HDDs two three-member files.
The previous scheduler finished in 342.83 s (3.57 s/member). With the
final-batch ordering enabled it took 344.73 s (3.59 s/member). When three
destinations remained, two already had every member they needed. Only one
could take the final grant, and both jobs used 32 three-member packed
transfers. All 32 files matched and passed fresh full-proof checks.
Across the two runs the 16 disks averaged 279–280 MB/s of writes and the
10 GbE receive link peaked at 4.44–4.48 Gb/s over five seconds.

On one HDD, one remote RTX 4090 produced a single three-member group with
32 GiB of owner RAM. In alternating runs, packed delivery took 27.68 and
27.51 s through publication. Streaming the encoded group took 19.12 and
18.72 s. The worker instance, granted resources and output file matched
in all four runs. The owner retained its per-drive timings across a
broker restart after the first packed run. Automatic mode then selected
streaming and published in 18.88 s (6.29 s/member). The previous policy
also chose streaming, finishing in 19.12 s before the comparison and
18.80 s afterward. All seven jobs produced the same file and passed
fresh full-proof checks.

#### Mixed workers on shared machines

With delivery times recorded separately for each batch size, four automatic
k=28 runs each wrote the same 48 members to 16 HDDs using 32 GiB of owner
RAM. They shared a broker and followed one RTX-only calibration run. Times
below end when the last file was durable.

| Workers in run order | Members from RTX / Radeon / Arc | Last file | Seconds per member |
|---|---:|---:|---:|
| RTX and Radeon | 43 / 5 / 0 | 167.30 s | 3.49 |
| RTX, Radeon and Arc | 43 / 5 / 0 | 164.88 s | 3.43 |
| RTX and Radeon, repeated | 42 / 6 / 0 | 183.56 s | 3.82 |
| RTX alone | 48 / 0 / 0 | 172.75 s | 3.60 |

In both wired runs the last file received one Radeon member followed by two
separate RTX members. Each RTX delivery took about 3.7 s once assigned. In
the first run those requests also waited 11.90 and 8.82 s for a grant. In
the repeat they waited about 4 ms each. After the final delivery the owner
took another 19.81–21.57 s to publish the last file across the four runs.
Every output matched the reference. Fresh full-proof checks covered all
16 files in each of the last two runs and five files in each of the first
two, including every file to which the Radeon contributed.

David farmed 17,886 plots throughout. At 15-second sampling intervals,
median utilization of its farming GPU was 98.5–99%. During the two wired
runs the RTX host averaged 9.9% and 15.3% CPU use. No sample showed Luanti
on the Windows host. The fastest and slowest wired runs fell on opposite
sides of the RTX-only time. In the four minutes immediately after the
comparison the plotting broker was idle. David's farming GPU still
showed 99% median utilization and 24 of 26 eligible lookups exceeded
eight seconds, with p95 at 15.84 s. During the four plotting runs the
harvester's p95 ranged from 13.36 to 24.79 s.

In a separate two-worker comparison the updated owner ran first. It
published the 48 members in 157.69 s (3.29 s/member), followed by
162.79 s (3.39 s/member) with the previous owner. Both used the same
worker instances, 16 drives and initial owner-cache contents. The broker
retained its growing delivery history between runs. The updated owner
accepted 46 RTX members and two Radeon members. The previous owner
accepted 43 and five. Both GPUs held active work simultaneously.

At 34.23 s the updated owner declined a three-member Radeon grant for
a batch it had priced on the RTX. It checked the two remaining members
and selected single-member transfers. That file finished at 75.55 s.
In both runs the RTX delivered the last file's three members in one batch.
The 5.10 s whole-job difference cannot be attributed to regrouping that one file.
All 16 files matched. Both runs checked fresh full proofs on the first
and last files and every file containing a Radeon member. The updated
run averaged 303 MB/s of HDD writes and peaked at 590 MB/s over five
seconds. The 10 GbE link reached 4.46 Gb/s over five seconds, and Ramen
delivered payload at 0.92 Gb/s during transfer on its 1 Gb/s link.

#### Adding a 1 Gb/s Radeon worker to a 10 Gb/s RTX worker

The same plotting executable built 48 k=28 members on 16 HDDs, first
with the RTX 4090 alone and then with the RX 6700 XT available. Both
jobs used automatic construction, batching and output format, 32 GiB
of owner RAM and eight owner threads. They started from identical copies
of the owner's timing cache. Before the first job the broker measured two
three-member RTX deliveries. It retained those and the solo job's 16 RTX
deliveries when the Radeon joined with no prior delivery on that broker.

| Available GPUs | Members from RTX / Radeon | Last durable file | Seconds per durable member | Job exit |
|---|---:|---:|---:|---:|
| RTX 4090 | 48 / 0 | 178.09 s | 3.71 | 178.65 s |
| RTX 4090 and RX 6700 XT | 44 / 4 | 157.02 s | 3.27 | 157.55 s |

The Radeon delivered four single members while the RTX supplied 11 triples,
four pairs and three singles. Both workers ran concurrently and each job
retried zero tasks. With the Radeon, the last file was durable 21.06 s
earlier, a reduction of 0.44 s/member (11.8%) in this comparison.
In both runs the RTX delivered the final three members for the same HDD.
They arrived at 157.32 s in the solo run and 135.97 s in the paired run.
That HDD's file became durable about 21 s after delivery in each run.
Ramen delivered 0.925 Gb/s of payload while transferring. David's 10 GbE
receive link peaked at 4.43 Gb/s over five seconds. The 16 output disks
averaged 305 MB/s of writes and peaked at 599 MB/s over five seconds.

Both jobs produced the same 16 files, totaling 47,730,868,909 bytes.
Fresh full-proof checks covered the first and last files in each run and
every file containing a Radeon member. David farmed 17,886 plots throughout.
Its farming GPU had median utilization of 98% in the solo run and 99%
in the paired run. Eligible harvester lookups had a p95 of 11.33 s
across 20 samples in the solo run and 17.92 s across 18 in the paired
run. The short samples and the variation seen with farming alone limit
how much of the difference can be attributed to plotting.

#### Dedicated destinations with 4 GiB of construction RAM

One RTX 4090 supplied 48 members to the same 16 HDDs on November 23, 2026.
Each run had eight owner threads and produced 16 three-member files. The regular farming
planner chose the collections. With dedicated destinations the other
HDDs were available exclusively for scratch. With all destinations
eligible the planner could temporarily use idle destinations for scratch.
Job times include successful process exit.

| Destination HDDs | Scratch HDDs used at peak | Previous job | Revised job | Previous s/member | Revised s/member |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 392.31 s | 291.36 s | 8.17 | 6.07 |
| 2 | 14 | 272.61 s | 266.51 s | 5.68 | 5.55 |
| 16 | 14 | 162.09 s | 148.62 s | 3.38 | 3.10 |

On the single destination, 13 succeeding groups received their 39 members
while the preceding file packed. The first publication took 23.08 s,
compared with 23.29 s before the change. The last publication moved from
392.15 s to 291.24 s. The output HDD delivered 163.9 MB/s of final plot
bytes over that full interval. Aggregate writes across scratch and output
rose from 258 to 348 MB/s. Linux block counters put the destination's
busy time at 75.6%, up from 54.9%. All 16 final files matched the previous run.
With two destinations, three groups received nine members in total while
their predecessors packed. The two runs finished writing identical files in 272.53 s
and 266.39 s. Each of the 16 destinations received a single group, making
that row a control for the other scheduling changes and run-to-run variation.

Each layout was compared against its own reference because automatic
collection assignment can change the output when files move between drives.

For a byte-identical comparison across layouts, the storage controls fixed
collection 0 and used the previous owner. The harvester model allowed 16
eligible families, eight cores, eight lookup workers and 1 GiB of reader
memory. Plot construction still had 4 GiB. One destination took 393.60 s,
two took 271.21 s and 16 took 148.89 s. All three layouts produced the
same 47,731,349,036 bytes. The one- and two-drive modeled response bounds were 5.70 s and
3.55 s against the same 8 s deadline. The default four-core reader with
these 16 families on one HDD would have required 10.44 s.

#### Scheduling GPU deliveries and HDD packing

On November 22, 2026, David filled the same 16 rotational HDDs with
48 k=28 members, three per group. Each fill had eight owner CPU threads,
automatic construction and a fresh broker timing history. Worker services
were warm. The RTX 4090 used 10 GbE, the RX 6700 XT used its wired relay
and the Arc B580 used Wi-Fi. Times run through the last durable
publication on any drive. Hashing and proof verification ran afterward.

With 4 GiB of owner RAM, commit `fea81ca` took 154.95 s with the wired
pair and 167.92 s with all three. Adding the Arc put 12 of the 16 groups
on a single scratch spindle and gave 12 groups one thread for final
packing. With the wired pair only one group used a single scratch
spindle, and all 16 groups packed with two threads.

At `c287b4e` the owner reserved scratch bandwidth by the batch sizes
each route could deliver. Six groups used a single scratch spindle in
the three-worker fill. Ten still got one packing thread. The wired pair
took 165.37 s and all three took 175.08 s. The Arc delivered one member
at 110.10 s, including 93.50 s of transfer.

At `7eb35b`, packers could use owner CPU threads while remote receivers
waited for data. The wired pair took 146.55 and 150.38 s. The three-GPU
fills took 172.43 and 198.19 s. In the slower fill, the penultimate
group held the only remaining destination HDD as scratch. The owner
recorded 10.81 s waiting for a storage lease after the other writers
finished. The planner now checks for a spare spindle with enough space
and bandwidth to free that destination. On a single scratch HDD it also
starts writing back completed contribution segments during reception.

All four hosts used `eb6a445` for the following comparisons. The scratch
repeat reversed the order of the two- and three-GPU fills.

| Owner RAM and construction | Wired pair, s/member | Wired pair and Arc, s/member |
| --- | ---: | ---: |
| 4 GiB, HDD scratch | 3.07 | 3.38 |
| 4 GiB, HDD scratch, repeat | 3.24 | 3.69 |
| 32 GiB, RAM assembly | 3.61 | 5.62 |

The first matched 4 GiB runs show how a slow delivery can
affect other groups. Both runs granted the same three scratch spindles
to one group. With the wired pair it published at 38.17 s. With the
Arc delivering one of its members it published at 112.61 s. The wired
fill could reuse two of those spindles at 38.37 s. Two groups used a
single scratch disk in the wired fill and seven did with all three GPUs.
At 22.01 s in the three-GPU run, the existing grants left only one
scratch spindle free for a three-member RTX batch. The Arc group's
three scratch spindles remained reserved at that point.
The 4090 spent 2.24 s/member between its first frame and completed
delivery with the wired pair and 2.63 s/member with all three. These
intervals include backpressure from the receiver. Five RTX batches sent
identical bytes for the same destinations in both runs and went from
multiple scratch disks to one. They averaged an extra 1.26 s/member in
delivery. The owner receive callbacks took an extra 1.25 s/member.
On the repeat, five matching batches averaged an extra 1.48 s/member
in delivery and 1.42 s/member in those callbacks. Other work on the
drives also changed between fills.

Across the four scratch fills, the highest five-second rates were
4.75 Gb/s received on David's 10 GbE interface and 1,103 MB/s written
across the 16 HDDs. The independent sequential write test on those
drives reached 3,328 MB/s. Plotting mixed reads and writes and could
keep individual scratch spindles busy while other drives were idle.

With 32 GiB, the owner assembled the groups in RAM. In a wired control,
owner build `9ca8b56` accepted 45 packed members and asked the Radeon
to assemble and stream the last three. The RTX delivered its final members
at 144.22 s, the Radeon finished at 198.46 s and the group was
durable at 198.81 s (4.14 s/member). At `eb6a445` the owner used the
measured packed route for all 48 members and published in 173.12 s
(3.61 s/member), a 12.9% reduction in this comparison.

With the Arc enrolled, its second packed member was assigned at about
114.30 s. The owner reported delivery of the first at 114.75 s. The
second arrived at 245.79 s and the last group published at 269.68 s.
The two transfers used the same worker, route and batch size.

At `2b10c10` the broker used the elapsed time on a just-finished GPU
lease while its first owner receipt was pending. In the three-GPU RAM
rerun the Arc received one member, delivered it at 184.72 s and spent
169.63 s transferring it. All 48 members reached disk in 206.60 s,
a 23.4% reduction from the run with the repeated Arc assignment.

| Construction at `2b10c10` | Wired pair, last group | Wired pair, s/member | All three, last group | All three, s/member |
| --- | ---: | ---: | ---: | ---: |
| 4 GiB, HDD scratch | 156.57 s | 3.26 | 172.64 s | 3.60 |
| 32 GiB, RAM assembly | 176.14 s | 3.67 | 206.60 s | 4.30 |

The wired pair finished sooner in both configurations. The 4 GiB
measurements at `fea81ca` were 3.23 and 3.50 s/member respectively.
Run-to-run variation on scratch exceeded the difference between builds.
In each of the four final fills, all 16 group hashes matched their
full-proof-verified references and a fresh 16-challenge proof check passed
on the first group. David's harvester reported 17,886 plots across all
75 eligible challenge samples during those fills. Its longest recorded
lookup took 5.63 s.

#### Adding a worker to a running fill

On November 22, 2026, David published the same 48 k=28 members on 16
Btrfs HDDs, three per group, with eight owner threads and an 8 GiB
output limit per HDD. The RTX 4090 used 10 GbE and the Arc B580 used
Wi-Fi. Each fill started a new owner. The observer checked for each
published file after the owner's
completion message, sleeping 50 ms between scans. On the previous
build, successful process exit followed the last observation within
0.602 s. All completed fills matched the 16 full-proof-verified group
hashes and passed a fresh 16-challenge proof check on one output.

| Owner and broker | RAM, construction | Worker admission | RTX / Arc members | First group | Last group | s/member |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Previous | 4 GiB, scratch | Both, warm broker | 47 / 1 | 28.29 s | 218.64 s | 4.55 |
| Previous, repeat | 4 GiB, scratch | Both, warm broker | 48 / 0 | 25.23 s | 166.11 s | 3.46 |
| Previous | 4 GiB, scratch | Both, fresh broker | 47 / 1 | 30.25 s | 199.00 s | 4.15 |
| Candidate | 4 GiB, scratch | Both, fresh broker | 47 / 1 | 24.97 s | 180.58 s | 3.76 |
| Candidate | 4 GiB, scratch | Both, warm broker | 47 / 1 | 38.92 s | 171.59 s | 3.57 |
| Candidate | 4 GiB, scratch | RTX only | 48 / 0 | 22.27 s | 176.26 s | 3.67 |
| Candidate | 4 GiB, scratch | Arc joined during fill | 46 / 2 | 20.70 s | 281.83 s | 5.87 |
| Candidate | 32 GiB, automatic RAM | Both | 47 / 1 | 42.35 s | 205.46 s | 4.28 |
| Candidate | 32 GiB, forced RAM | Both | 47 / 1 | 38.34 s | 208.93 s | 4.35 |
| Candidate | 32 GiB, forced scratch | Both | 47 / 1 | 40.47 s | 189.17 s | 3.94 |
| Corrected broker | 4 GiB, scratch | RTX only | 48 / 0 | 22.41 s | 168.38 s | 3.51 |

The 52.53 s difference between the previous warm runs set the practical
comparison threshold for the 4 GiB two-worker case before the candidate
ran. The fresh-broker difference of 18.42 s fell inside it. Builds and
CPU tests ran on the RTX workstation. Its worker could use logical CPUs
0–23, including CPU 20 used by a pinned test. CPU use during individual
GPU deliveries was unmeasured.

In the live join, the broker listed the new Arc instance 5.567 s after
owner launch. Its first member took 115.8 s from request to validated
receipt. The broker assigned it another singleton at 158.26 s, just
before an RTX pair finished at 158.46 s. That Arc request took 128.9 s
including its broker wait. The last two group publications were 103.6 s
apart. The corrected broker uses the observed time for a larger joint
batch on the same route to estimate when the busy RTX can take the
singleton. A regression fixture reproduces this assignment. The corrected
live-join run assigned one member to the Arc and published six groups
before the trial was interrupted. Its final throughput and farming
outcome remain unverified.

At 32 GiB the automatic and forced RAM fills received 48 packed
members. Forced scratch received 48 encoded contributions and finished
16.29 s before automatic. Each target published once. At admission the
automatic owner had no comparable scratch read, merge or encoded-delivery
measurement for those targets. The three runs followed the table order.
The RTX-only comparison below repeats the two modes in reverse order.
The mixed-worker runs have one observation per mode. On the corrected
RTX-only control, 19,022 forecast evaluations used 4.146 s of wall time.
The last admission had a partial forecast. Historical replay of the four
earlier 48-member traces also lacked sufficient pre-grant evidence for
a complete through-publication forecast.

During the fresh-broker candidate fill, the highest five-second rates
were 4.627 Gb/s received on David's 10 GbE interface, 689.9 MB/s read and
913.8 MB/s written across the 16 benchmark volumes. Both SAS paths of
every tested disk and all 88 mounted farming JBODs used one SAS9300-8e
with a SAS3008 controller. It negotiated PCIe 3.0 ×4 and eight 12 Gb/s
SAS phys. The 10 GbE X540 negotiated PCIe 2.0 ×4 and shared an upstream
PCIe 4.0 ×4 bridge with the HBA. The disk counters cover the benchmark
volumes, and the NIC counters include other traffic. Controller payload
and link utilization were unavailable.

An external sampler read the live farm logs during ordinary plotting.
The farming-only control recorded 14 eligible completions, with a
maximum of 3.36384 s. Across the completed plotting runs the harvester
reported 17,886 plots and 233 eligible completions. Each of the 229
recorded full-node signage arrivals fell within one second of the inferred
start of its matched lookup. Three lookups finished after their fill's
last publication. One took 8.82234 s during the adverse join, exceeding the
8 s operating target. The farm log recorded zero timeouts, V1 proofs or
V2 qualities in these windows.

A separate local NVMe comparison pinned one CPU and published three
three-member k=22 groups per invocation. Run averages were 1.990, 2.049
and 2.017 s/member on the previous owner and 2.032, 2.031 and 2.056
s/member on the candidate. All group hashes matched and each reference
passed a 100-challenge full-proof check.
On the same local fixture in a pseudo-terminal, the final build with the
dashboard (2.040 s/member), phase tracing (2.067) and both (2.039) fell
within the two plain runs (2.164 and 2.037), with matching hashes.

#### Automatic RAM and forced scratch with one RTX 4090 and 16 HDDs

These controls used the owner and broker at `f2d5863`. The previous
owner in the live-join comparison was `2b10c10`, and the candidate and
corrected rows there used intermediate builds. Each control started a fresh
owner, cache and private broker for the same 48 members on David's 16
HDDs at 32 GiB. The same RTX 4090 was the sole registered worker. The
owner reported its first and last durable publication directly. Its
verbose diagnostics used a 112 × 32 pseudo-terminal, with JSON on a
separate stream. David's running farmer reported 17,886 k=32 plots,
1.7296 PiB of effective plot size and 0.77955 PiB of loaded files.

| Run in order | First durable (s) | Last durable (s) | s/member | Forecast evaluations / wall seconds | Eligible farm lookups / maximum (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Automatic, plain | 31.49 | 243.26 | 5.068 | 87,172 / 15.77 | 26 / 5.214 |
| Forced scratch, plain | 21.95 | 176.42 | 3.675 | 16,109 / 3.922 | 18 / 4.629 |
| Automatic, dashboard | 29.95 | 234.53 | 4.886 | 83,600 / 15.09 | 25 / 4.927 |
| Automatic, phase tracing | 29.19 | 230.88 | 4.810 | 82,744 / 15.31 | 24 / 6.700 |
| Automatic, both | 29.28 | 226.33 | 4.715 | 80,576 / 14.67 | 24 / 6.698 |
| Forced scratch, plain | 21.14 | 171.07 | 3.564 | 16,038 / 3.864 | 18 / 5.654 |
| Automatic, plain | 28.44 | 231.49 | 4.823 | 82,475 / 15.06 | 26 / 5.173 |

Automatic used RAM for all groups. Each traced admission lacked a complete
finish estimate for either construction mode. Each automatic fill received
48 packed members, and each forced fill received 48 encoded contributions.
Forced scratch finished 66.84 s earlier in the first pair and 60.42 s
earlier in the reverse pair. The dashboard-only time fell between the two
plain times. Tracing alone and with the dashboard finished sooner than both
plain controls. The RTX worker could use all 24 logical CPUs, and its
host's average CPU utilization ranged from 5.73% to 15.2% across the seven
fills. Each display configuration had one observation.

All seven fills matched the 16 full-proof-checked reference hashes and
passed a fresh 16-challenge proof check on one group. The external sampler
matched all 158 recorded full-node signage arrivals to a lookup start
within one second and counted 161 eligible completions. One lookup
completed after its fill's last durable publication. The farm recorded
zero lookups over 8 s and zero timeout lines. An 80 s farming-only window
before the controls had eight eligible lookups with a maximum of 3.917 s.

Two further automatic runs at `a90f048` published the same 48 members in
225.60 and 231.73 s (4.700 and 4.828 s/member). Forecasting took 2.001
and 2.039 s across 86,632 and 89,452 evaluations. Both returned 48 packed
members and peaked at three active destination engines and one in-flight
GPU generation request, as in the earlier plain controls.
At the same revision, forcing incremental RAM construction and encoded
output returned 15 whole groups from the RTX and three packed members for
the remaining group. The owner recorded zero local fallbacks, retries or
failed tasks. The first group became durable at 19.72 s and the last at
324.76 s (6.766 s/member). Up to nine requests were in flight to the
single GPU slot.

At `3c5ff1f`, the automatic RAM fills had two generation requests in
flight to the RTX and peaked at five and four active destination engines.
Their first groups were durable at 29.03 and 29.54 s. All 48 packed members
were durable at 180.16 and 185.35 s (3.753 and 3.861 s/member). Forecasting
took 1.440 and 1.487 s. Mean time per member fell 20.1% from `a90f048`.
The automatic runs still averaged 5.18% longer than the forced-scratch
controls. Both new runs matched all 16 reference hashes and passed
a fresh 16-challenge full-proof check. The farm logged 39 eligible
completions, with a maximum of 6.174 s. All 38 recorded signage arrivals
matched and their lookups completed before the last publication.

A separate ordinary fill selected the live Gigahorse 2.7.4.giga37 reader
with `--harvester local`. For its 30 configured lookup workers, the
per-job model required 1,040 MiB for 16 HDDs, 520 MiB for eight and
260 MiB for four. With the diagnostic from `41dd81e`, a fresh 16-HDD
fill reported the reader requirement and budget in JSON and exited 2
with all target directories empty. The four-HDD fill fit the existing
512 MiB budget and published 12 members in 70.96 s. All four hashes
matched and all four groups passed fresh full-proof checks. The final Farm
screen showed 17,886 loaded plots and three completions in 30 s, all with
eligible plots. Median response was 2,733.2 ms and p95 and maximum
were 3,942.3 ms. The external sampler recorded seven eligible completions
over the full run, matched all six recorded signage arrivals and counted
zero lookups over 8 s.

#### Shared compute over a dedicated Ethernet link

A GPU workstation and a storage host had separate 1 and 10 GbE links. With
MTU 1500, four concurrent encrypted SSH streams carried 9.04 Gb/s of payload
to the storage host and 9.08 Gb/s back. Each direction transferred 4 GiB.

On the 1 GbE route, a fixed-seed `fill` generated a three-member k=28
group on the remote GPU and published it on one HDD using four scratch
HDDs. Both builds produced the same 2,983,126,302-byte file. Elapsed time
includes generation, network transfer, assembly and publication.

| Enrolled compute transport | Elapsed | Seconds per member | Highest five-second NIC receive rate |
| --- | ---: | ---: | ---: |
| Previous | 83.4 s | 27.8 | 0.51 Gb/s |
| Updated | 57.0 s | 19.0 | 0.94 Gb/s |

With twelve scratch HDDs and three output HDDs, the updated build published
three three-member groups in 139.8 s, or 15.5 s per member. Its fastest
five-second interval reached 0.97 Gb/s on the 1 GbE link. All three
groups passed full-proof verification. The storage host received 9.98 GB.
At the same bytes per member, a producer finishing one member every 2.2 s
would require 4.03 Gb/s inbound and 452 MB/s of final plot writes, plus
scratch I/O.

On the same drives, a 5 GiB cap per output filesystem fit five members in
two groups. Allowing up to nine members per destination took 305.0 s and
generated 21 members to find and fill the available space. Repeating the
fixed-seed job with `--max-groups 2 --max-total-members 5` took 239.1 s
and generated 15 members. All six published files matched byte for byte.
The known member limit saved 7.28 GB of traffic received by the storage host.

The dedicated 10 GbE comparison kept the broker on the management network.
Plot data went directly from one RTX 4090 worker slot to the storage host.
Both builds used the same seed, three output HDDs, twelve scratch HDDs,
24 GiB of owner RAM and a 5 GiB limit per output. Timings start at `fill`
launch and end at publication and clean exit. File hashing, proof verification
and the subsequent resume check ran outside that interval.

| Capacity-limited fill, 15 published members | Previous planner | Updated planner |
| --- | ---: | ---: |
| Initial comparison | 221.1 s | 172.1 s |
| Repeat with the final scratch reservation policy | 187.2 s | 167.3 s |
| Seconds per published member on the repeat | 12.48 | 11.15 |
| Complete members received | 21 | 15 |
| Storage-host NIC receive bytes on the repeat | 24.53 GB | 19.42 GB |

The updated planner stopped six transfers after the validated chunks proved
the candidate could not fit. The repeat took 10.6% less wall time and
received 20.8% fewer bytes. Worker generation still started for each probe.
On the repeat, the highest five-second NIC receive rate was 1.77 Gb/s.
Across the fifteen drives, the highest five-second read and write rates
were 322 MB/s and 476 MB/s, respectively. For the 15 completed contributions,
the worker reported 6.76 s per member processing generation and encoding,
excluding sends.

With the five-member limit supplied in advance, the same 15 members took
130.9 s with the previous planner and 132.5 s with the final planner. Each
received 17.25 GB. Reserving larger compressed-input runs for every rollover
had taken 142.9 s in this control. With one three-member group per drive,
the nine-member job took 106.7 s with the previous planner and 101.1 s
with the final planner. Both received 9.98 GB. All six distinct output files
matched the baseline SHA-256 hashes and passed full-proof verification.
Resuming the capacity-limited fill generated zero additional members.

The previous transport verified the signed membership registry repeatedly
while receiving a frame. The update checks both authenticated identities
against one verified registry and reuses that result while the trusted
certificate and signed policy match. Two 512 MiB loopback transfers on
the GPU workstation, with enrolled peers and 1 MiB frames, delivered
1.1 Gb/s with the previous build and 18.9–22.1 Gb/s with the update.
On the storage host, the updated transport delivered 7.4–7.7 Gb/s on one
enrolled loopback connection and 13.9–14.7 Gb/s across two connections.

Independent direct-I/O tests used unnamed temporary files on the mounted
filesystems. The mixed workload ran for ten seconds. The all-read and
all-write workloads ran for twelve seconds. Rates include the last write flush.

| Load on the storage host | HDD reads (MB/s) | HDD writes (MB/s) | 10 GbE traffic during disk I/O |
| --- | ---: | ---: | --- |
| Eight readers and seven writers | 2,127 | 1,860 | Idle |
| Eight readers and seven writers with inbound SSH | 2,125 | 1,856 | 9.79 Gb/s received |
| Sixteen readers | 3,240 | — | Idle |
| Sixteen readers with inbound SSH | 3,240 | — | 9.80 Gb/s received |
| Sixteen writers | — | 3,328 | Idle |
| Sixteen writers with outbound SSH | — | 3,329 | 9.84 Gb/s sent |

Both SAS paths of every disk led to the same SAS3008 HBA. Its PCIe 3.0 x4
link carries up to 3.94 GB/s in each direction before transaction overhead.
The sixteen-drive runs reached 82.3% of that rate on reads and 84.5% on
writes. The X540 NIC negotiated PCIe 2.0 x4. Their common upstream
bridge negotiated PCIe 4.0 x4. NIC counters include protocol overhead.

An eight-second sweep on the same filesystems measured where adding disks
stopped increasing sustained bandwidth in one direction.

| Active HDDs | 8 | 12 | 14 | 16 |
| --- | ---: | ---: | ---: | ---: |
| Read MB/s | 2,255 | 3,215 | 3,236 | 3,237 |
| Write MB/s | 2,249 | 3,221 | 3,311 | 3,309 |

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
