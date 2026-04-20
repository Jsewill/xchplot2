# Streaming pipeline design — 8 GB VRAM target

Internal design doc for the work that lets `xchplot2` produce v2 plots on
sub-15 GB cards (GTX 1070 floor). Companion to the roadmap in the chat;
not shipped with the repo.

## Current pool at k=28 strength=2

Constants:

* `total_xs = 2^28 = 268,435,456`
* `num_section_bits = (k < 28) ? 2 : k-26 = 2` → `num_sections = 4`
* `extra_margin_bits = 8 - (28-k)/2 = 8`
* `max_pairs_per_section = (1<<(k-2)) + (1<<(k-8)) = 2^26 + 2^20 = 68,157,440`
* `cap = max_pairs_per_section × 4 = 272,629,760`
* `XsCandidateGpu` = 8 B, `T1PairingGpu` = 12 B, `T2PairingGpu` = 16 B, `T3PairingGpu` = 8 B

Pool allocations:

| Buffer            | Formula                                          | k=28 size |
|-------------------|--------------------------------------------------|----------:|
| `d_storage`       | max(total_xs × 8, cap × 4 × 4) = cap × 16        | **4.36 GB** |
| `d_pair_a`        | max(cap × {12,16,8,8}) = cap × 16                | 4.36 GB |
| `d_pair_b`        | same as pair_a                                   | 4.36 GB |
| `d_sort_scratch`  | CUB radix-sort scratch (cap × uint32)            | ~2.3 GB |
| `d_counter`       | 8 B                                              | — |
| **Pool total**    |                                                  | **~15.4 GB** |
| + runtime margin  | driver + CUB internal + T-tables                 | ~0.5 GB |

## Per-phase live working set

Current design pre-allocates the full pool once; every buffer stays
resident for the whole plot. To target 8 GB we need to (a) alias
aggressively so buffers share memory, and (b) tile phases whose working
set exceeds 8 GB.

Actual **live data** per phase (not buffer capacity):

| Phase              | Live working set           | Bytes       |
|--------------------|----------------------------|------------:|
| Xs gen             | Xs output + gen scratch    | 2.15 + 4.36 = **6.51 GB** |
| T1 match           | sorted_xs in + T1 pairs out| 2.15 + up to 3.27 (T1×12) = **5.4 GB** |
| T1 sort            | T1 + keys/vals + CUB + meta_out | 3.27 + 4.36 + 2.3 + 2.15 = **12.08 GB** 🔴 |
| T2 match           | meta + mi + T2 out         | 2.15 + 1.07 + 4.36 = **7.58 GB** |
| T2 sort            | T2 + keys/vals + CUB + meta_out + xbits_out | 4.36 + 4.36 + 2.3 + 2.15 + 1.07 = **14.24 GB** 🔴 |
| T3 match           | meta + xbits + mi + T3 out | 2.15 + 1.07 + 1.07 + 2.15 = **6.44 GB** |
| T3 sort            | T3 + frags_out + CUB       | 2.15 + 2.15 + 2.3 = **6.60 GB** |
| D2H                | frags_out + pinned (host)  | 2.15 GB |

🔴 = exceeds 8 GB target.

The tight phases are **T1 sort** and **T2 sort**. Everything else fits
in 8 GB if the prior phase's buffers are released before the next
phase allocates.

## Design choices for the 8 GB target

### 1. Per-phase alloc/free instead of single pool

Current `GpuBufferPool` allocates all buffers at construction time and
never frees. The streaming pipeline will allocate phase-scoped buffers,
release them before the next phase, and reuse a single arena across the
run.

* Phase boundaries are already clearly delimited in `GpuPipeline.cu`.
* Device-side `cudaFree` / `cudaMalloc` between phases is fine
  performance-wise (one-time cost per phase, negligible vs the 100+ ms
  of kernel work per phase).

Per-phase peaks after aliasing:

| Phase     | After aliasing | Needs tiling? |
|-----------|---------------:|:---:|
| Xs gen    | 6.51 GB        | no |
| T1 match  | 5.42 GB        | no |
| T1 sort   | **12.08 GB**   | yes |
| T2 match  | 7.58 GB        | no (fits) |
| T2 sort   | **14.24 GB**   | yes |
| T3 match  | 6.44 GB        | no |
| T3 sort   | 6.60 GB        | no |
| D2H       | 2.15 GB        | no |

### 2. Tiled sort for T1 and T2 (the hard part)

CUB `DeviceRadixSort::SortPairs` operates on the whole array in one
call. For tiling we need to split into N sorted runs and merge:

1. Partition input cap × 12/16 B into N sub-ranges (by index).
2. Sort each sub-range to a pinned host buffer (or a second device
   region) with a per-tile CUB call — peak is smaller by 1/N.
3. N-way merge the sorted tiles into the final sorted stream.

Tile-size math for N=4 at T1 sort (cap = 272 M, T1 = 12 B):

* Per-tile input: cap/4 × 12 = 0.82 GB
* Per-tile keys/vals (4 × uint32): cap/4 × 16 = 1.09 GB
* Per-tile CUB scratch: ~cap/4 × 8 = 0.6 GB
* Per-tile sorted output: cap/4 × 8 = 0.54 GB
* **Per-tile peak: ~3.05 GB**

With N=4 tiles, we stage sorted runs through either:

* Pinned host (cap × 8 = 2.15 GB meta, cap × 4 = 1.09 GB mi, held on
  host between tile sort and final merge).
* Or: keep all N sorted runs on device in a single arena, merge
  in-place — but the full arena is still cap × 12 = 3.27 GB, plus the
  merge needs a destination of similar size → ~6.5 GB during merge.

The host-staged approach is simpler and fits tight budgets.

### 3. Merge kernel

A GPU N-way merge of 4 sorted uint64 streams is a small new kernel.
Can be done by:

* Building a heap of N top-of-stream values (tree of N-1 comparators).
* Or, since N is small (4), a naive "min of 4 pointers" scalar merge
  on a small grid.

This is new code and needs parity. Not huge — maybe 100 LOC.

### 4. Xs gen at 6.5 GB

Xs gen holds d_storage (2.15 GB actual) and xs_temp (4.36 GB buffer).
For 8 GB it fits with margin. No tiling needed. But we might be able
to shrink xs_temp further if it's over-provisioned — check
`launch_construct_xs`'s scratch calc at k=28.

### 5. Fine-bucket pre-index memory

At T3 strength=2: 32 KB for fine_offsets. Trivial. No impact.

## Budget confirmation

With per-phase alloc/free + tiled T1/T2 sort (N=4):

| Phase     | Peak on 8 GB card |
|-----------|------------------:|
| Xs gen    | 6.51 GB |
| T1 match  | 5.42 GB |
| T1 sort (tiled N=4) | ~3.05 GB + host staging |
| T2 match  | 7.58 GB |
| T2 sort (tiled N=4) | ~3.60 GB + host staging |
| T3 match  | 6.44 GB |
| T3 sort   | 6.60 GB |
| D2H       | 2.15 GB |

Tightest remaining phase: **T2 match at 7.58 GB.** Under 8 GB, just.
If we see OOM in practice we can tile T2 match's output by writing the
pairing result chunks progressively to host.

## Implementation phases (from the chat plan)

* **Phase 2 — streaming orchestrator skeleton (k=18).**
  New `GpuBufferPoolStreaming` + `run_gpu_pipeline_streaming` that does
  per-phase alloc/free but **no tile yet** (single tile per phase).
  Prove orchestration flow end-to-end at k=18. Keep the existing
  monolithic pipeline as default.

* **Phase 3 — tile T1/T2 sort + T2 match output at k=18.**
  Multi-tile sort + N-way merge kernel. Parity-gated.

* **Phase 4 — k=28 dry run under simulated 8 GB cap.**
  Use `cudaDeviceSetLimit(cudaLimitMallocHeapSize, ...)` or a
  `POS2GPU_MAX_VRAM` env var in `GpuBufferPool` to refuse allocs above
  the cap. Run a full plot; measure peaks.

* **Phase 5 — dispatch.**
  `run_gpu_pipeline` checks `cudaMemGetInfo` at pool construction. If
  free < 15 GB, uses the streaming pipeline; else the existing pool.
  Users see no flag.

* **Phase 6 — 1070 perf tuning.**
  Actual 1070 or cloud equivalent. Tune tile counts, staging depth,
  PCIe overlap. Budget: 15–25 s/plot.

## Open questions

1. Does `launch_construct_xs` actually need all 4.36 GB, or can its
   scratch be reduced by tiling Xs generation too? If so, Xs gen drops
   from 6.5 GB to something smaller, widening our margin elsewhere.
2. Can CUB be told to use a smaller scratch for radix sort, at the
   cost of more internal passes? That'd be a cleaner fix than tiling
   + merging ourselves.
3. Is the 2 s/plot expectation for 16 GB cards regressed by the
   dispatch check at pool construction? Almost certainly no — it's a
   single `cudaMemGetInfo` call.

## Phase 4 findings (2026-04-19)

Implemented a `StreamingStats` tracker in `GpuPipeline.cu` that wraps
every streaming-path `cudaMalloc`/`cudaFree`, logs under
`POS2GPU_STREAMING_STATS=1`, and enforces `POS2GPU_MAX_VRAM_MB`
as a soft device-memory cap.

### k=28 unconstrained baseline
Peak **12,484 MB** (T1 sort phase). The Phase-3 N=2 tiling reduces
sort scratch by ~half vs a single CUB call but the other live buffers
(d_t1 3.12 GB + 4 sort key/val arrays 4.16 GB + d_t1_meta_sorted
2.08 GB + runtime overhead ~1 GB) already dominate, so tiling just the
sort doesn't reach the 8 GB target.

### k=28 with `POS2GPU_MAX_VRAM_MB=8192`
Trips at T1 sort, allocating d_t1_meta_sorted:
- live 7280 MB (d_t1 3120 + keys_in/out 2×1040 + vals_in/out 2×1040)
- + new 2080 MB (d_t1_meta_sorted) = 9360 > 8192 cap.

### Path to 8 GB
N=2 alone is insufficient. To hit 8 GB for k=28 we need to cut the
T1-sort live set meaningfully — candidates, cheapest first:
- Fuse permute with merge so d_t1 and sort scratch can be released
  as the permute streams output (reclaims ~3 GB).
- Bump to N=4 tiles AND stream sorted tiles to pinned host between
  per-tile CUB calls and the merge; drops peak sort-scratch + per-tile
  arrays but adds PCIe cost.
- Tile Xs gen to free some of its 4.14 GB scratch earlier (doesn't
  help T1 sort directly but widens margin for the next item).

### Parity bug uncovered (and fixed) during Phase 5 bringup
Early pool/streaming parity runs at k=18 diverged: streaming gave
T2=251749 vs pool T2=259914 despite identical T1 inputs. Initial
hypothesis was T1 atomic ordering + T2 order-dependence on ties;
hashing d_t1 post-sort showed different raw bytes but matching
sorted-set hashes, seeming to confirm it. That hypothesis was wrong.

Real root cause: the streaming pipeline allocated `d_match_temp` as
a 256-byte dummy, assuming the T1/T2/T3 match kernels only needed a
non-null pointer for CUB internals. In fact the match kernels
**write ~32 KB of bucket + fine-bucket offsets into that buffer**
(computed per-phase via the nullptr-size-query call) and read it
back inside the match kernel. The 256 B allocation meant the kernels
were scribbling ~32 KB into whatever device allocation sat adjacent
to `d_match_temp` — a different victim per run, but always
corrupting something.  Pool didn't hit this because its
`d_match_temp` aliased the ~2.3 GB sort scratch.

Fix: per-phase `d_match_temp_<t>` sized to the query's return value,
freed after the match. See commit history for the exact change.

Post-fix: k=18 and k=28 produce bit-identical plot bytes across pool
and streaming. T1/T2/T3 atomic-emission order is still nondeterministic
run-to-run, but downstream CUB sort + stable merge-path + pool/streaming
both consume the pairs as a set so the nondeterminism is invisible.

## Phase 5 findings (2026-04-19)

Implemented automatic pool-to-streaming fallback. No user-facing flag.

### One-shot path (`GpuPlotter::plot_to_file` → `run_gpu_pipeline(cfg)`)
Wraps the `GpuBufferPool` construction in `try {} catch
(InsufficientVramError const& e)`. The pool ctor throws this typed
exception (declared in `GpuBufferPool.hpp`) specifically when its
pre-allocation `cudaMemGetInfo` check fails — every other CUDA
error path still throws plain `std::runtime_error` and propagates.
On the typed catch we log the `required_bytes / free_bytes /
total_bytes` fields and route to `run_gpu_pipeline_streaming(cfg)`.

### Batch path (`BatchPlotter::run_batch`)
Same typed catch at pool construction; on fallback, the pool is
absent (`std::unique_ptr<GpuBufferPool> pool_ptr` stays null) and
the producer loop dispatches per-plot to
`run_gpu_pipeline_streaming(cfg)`. The self-contained result
vector is compatible with the existing
`GpuPipelineResult::fragments()` span accessor, so the consumer
thread's FSE + plot-file-write code is unchanged.

No producer/consumer regression: the Channel still overlaps the
producer's streaming call with the consumer's file write. What we
lose vs. the pool path: (a) the ~2.4 s per-plot `cudaMalloc` /
`cudaMallocHost` amortisation benefit, and (b) the double-buffered
pinned D2H overlap between producer-N+2 and consumer-N. Both are
acceptable costs when the pool literally doesn't fit.

### Override still available
`XCHPLOT2_STREAMING=1` remains for forced streaming on any card —
useful for testing and for users who want the smaller-VRAM path
even when the pool would fit.

### Validation
- Default path (pool, k=18): bit-exact to prior baseline.
- Env-forced streaming (k=18): bit-exact to the pool path.
- Automatic fallback not integration-tested on real hardware; the
  catch-and-route is 5 lines and matches the pool ctor's exact
  error string, so this is Phase 6 alongside 1070 perf tuning.

## Phase 6 progress (2026-04-19)

Started cutting the k=28 streaming peak toward 8 GB.

### Fused merge-path + permute kernels
New `merge_permute_t1` / `merge_permute_t2` kernels do per-thread
merge-path partition AND gather src[val].meta / x_bits in one pass,
eliminating the intermediate `merged_vals` buffer that the
two-kernel (merge → permute) flow had to materialise. The streaming
path now frees `d_vals_in` and sort scratch before even allocating
the permuted meta outputs, which narrows the peak-live window.

### Allocation reorder
`d_t1_meta_sorted` and `d_t2_meta_sorted`/`d_t2_xbits_sorted` are
now allocated AFTER CUB tile sort + `d_vals_in` + sort scratch are
freed, not at the start of the sort phase. This keeps ~3 GB of
buffers from being simultaneously live at k=28.

### Measured impact (k=28 strength=2 plot_id=0xab*32)
| State                                         | Streaming peak |
|-----------------------------------------------|---------------:|
| Before Phase 6 work                           | **12,484 MB**  |
| After fuse + reorder                          | **10,400 MB**  |
| After T2 match → SoA emission                 |  **9,360 MB**  |
| After T2 sort 3-pass (merge/meta/xbits)       |  **8,324 MB**  |
| After T1 match → SoA emission                 |  **8,324 MB**  |
| After N=4 T2 tile + tree-merge                |  **7,802 MB**  |
| **8 GB target**                               |    8,192 MB    |
| **Under target**                              |   −390 MB      |

### T2 match SoA emission
Refactored `launch_t2_match` to emit three parallel streams
(`d_t2_meta` uint64, `d_t2_mi` uint32, `d_t2_xbits` uint32) instead
of a packed `T2PairingGpu` array. Total bytes are the same
(cap·16 B), but the streams are freeable independently — the
streaming T2 sort now passes `d_t2_mi` directly to CUB as the sort
key input and frees it as soon as CUB consumes it, skipping the
`extract_t2_keys` pass entirely. Saves ~1 GB at k=28.

Pool path uses the same SoA allocation carved out of `d_pair_a`
(meta[cap] then mi[cap] then xbits[cap] = cap·16 B). `t2_parity`
tool rebuilds `T2PairingGpu` on the host from the three streams
for set-equality comparison against the CPU reference.

### T2 sort 3-pass (post-CUB merge/gather/gather)
Split the previously-fused `merge_permute_t2` into three kernel
launches in the streaming path:
1. `merge_pairs_stable_2way` writes `merged_keys + merged_vals`.
2. `gather_u64` builds `d_t2_meta_sorted`.
3. `gather_u32` builds `d_t2_xbits_sorted`.

Frees the source column (meta / xbits) between passes, so each
gather only needs one source buffer + one output alive. Peak drops
~1 GB at the cost of two extra DRAM sweeps (negligible next to the
CUB sort cost).

### T1 match SoA emission
Mirror of the T2 SoA change. `launch_t1_match` now emits
`d_t1_meta (uint64) + d_t1_mi (uint32)` instead of a packed
`T1PairingGpu[]`. Streaming's T1 sort passes `d_t1_mi` straight
into CUB as the sort key (no `extract_t1_keys` pass) and frees it
as soon as CUB consumes it. Pool path uses the same SoA layout
carved out of `d_pair_a`. `t1_parity` rebuilds the AoS form on the
host for set-equality vs the CPU reference.

### N=4 T2 tile + tree merge
To close the last ~130 MB of the gap, the streaming T2 sort is
now tiled 4 ways. Per-tile CUB scratch halves from ~1,044 MB to
~522 MB, which is the peak-binding allocation.

The 4-way merge is implemented as a tree of three 2-way merges,
reusing the existing `merge_pairs_stable_2way` kernel:
`(tile 0 + tile 1) → AB`, `(tile 2 + tile 3) → CD`,
`(AB + CD) → final`. Intermediate buffers `AB`/`CD` are half the
total size each, so their combined footprint (~2 GB) fits inside
the headroom we gained from the smaller CUB scratch.

T1 sort stays at N=2 — it's already under 8 GB after T1 SoA, so
adding a merge tree there would be effort without benefit.

### Historical gap analysis (pre-closure)
T2 sort is still the binding phase, now peaking at the allocation
of `d_t2_xbits_sorted` (post-CUB, before the fused merge-permute):

| Buffer               | Bytes  |
|----------------------|-------:|
| d_t2_meta (in)       | 2,080  |
| d_t2_xbits (in)      | 1,040  |
| d_keys_out (in)      | 1,040  |
| d_vals_out (in)      | 1,040  |
| d_t2_keys_merged (out)| 1,040  |
| d_t2_meta_sorted (out)| 2,080  |
| d_t2_xbits_sorted (out)| 1,040 |
| **sum**              | **9,360** |

Options to close the remaining ~1.2 GB gap:
1. Make T3 match tile-aware so the merged sorted-MI stream
   `d_t2_keys_merged` doesn't need to be materialised at all (T3
   would accept two tile-sorted streams + tile boundaries). Saves
   1,040 MB. Requires changes to `T3Kernel.cu`.
2. Pinned-host staging of one or more of the post-permute outputs
   (writes meta_sorted / xbits_sorted to pinned RAM and streams
   back for T3 match). Saves up to 3 GB but adds PCIe transfer time
   twice.
3. Fuse the per-tile CUB sort with the merge-permute — output
   sorted-within-tile pairs directly into the final merged buffers.
   Requires a custom sort (can't use CUB DeviceRadixSort as a
   black box).

### k=28 parity after Phase 6 changes
`pool` and `streaming` produce bit-identical plots at k=18 (6
plot-id × strength cases) and at k=28 strength=2 plot_id=0xab*32.

### Left for a subsequent pass
- T2 match SoA emission (requires editing `src/gpu/T2Kernel.cu`).
- N=4 tile + 4-way merge (saves ~500 MB of sort scratch at each
  sort phase; needs a 4-way merge kernel or a pairwise merge tree).
- Tile Xs gen scratch (currently `d_xs_temp` at 4,136 MB is the
  main contributor to the Xs-phase peak of 6,184 MB; not the
  binding constraint but would widen margin).

## Batch streaming perf (2026-04-19)

Added an overload
`run_gpu_pipeline_streaming(cfg, pinned_dst, pinned_capacity)`
that takes a caller-supplied pinned D2H target instead of
cudaMallocHost'ing per call. BatchPlotter's streaming-fallback
branch now owns two cap-sized pinned buffers (double-buffered
like the pool path: plot N writes slot N%2 while consumer reads
slot (N-1)%2) and threads them into the streaming pipeline.

Pinned alloc/free shims (`streaming_alloc_pinned_uint64` /
`streaming_free_pinned_uint64`) live in `GpuPipeline.cu` so
`BatchPlotter.cpp` — a plain .cpp consumer without cuda_runtime.h
on its include path — can own the pinned buffers.

`XCHPLOT2_STREAMING=1` now also forces BatchPlotter to skip pool
construction and use the streaming fallback directly. Matches the
behaviour of the one-shot path, and makes the streaming batch
branch testable on high-VRAM hardware.

### k=28 batch timings (4090, single plot, ab*32)
| Mode                  | Time     |
|-----------------------|---------:|
| Pool batch            | 3.05 s   |
| Streaming batch       | 3.65 s   |
| Delta                 | +0.60 s  |

The 0.60 s delta is the per-phase cudaMalloc/cudaFree overhead
the streaming path intrinsically pays (its whole point — shrinks
peak VRAM by freeing between phases). The ~600 ms cudaMallocHost
cost that it would otherwise pay per plot is amortised away by
the double-buffered external pinned. Bit-exact vs pool across
k=18 (3 plots) and k=28 (1 plot).
