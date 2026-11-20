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
