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
