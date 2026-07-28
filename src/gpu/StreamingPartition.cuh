// StreamingPartition.cuh — partition (u32 keys, u64 vals) pairs by
// top-N-bits of the key, with the values streamed tile-by-tile from
// host-pinned memory and the bucketed output landing in host-pinned
// arenas via zero-copy device-side writes.
//
// The point: T1 sort in StreamingPinned mode can't hold both
// d_t1_meta (full-cap u64 ~2 GB at k=28) AND a sort scratch on a
// 2-3 GB device. By streaming the meta tile-by-tile from
// host-pinned and writing partitioned output directly back to
// host-pinned bucket arenas, the device-side working set drops to
// (one tile + bucket counters + small misc), and the actual sort
// happens per-bucket on data that's already host-resident.
//
// Phase 1.3b of project_streaming_pinned_disk_spec. Consumed by
// Phase 1.3c (T1 sort path) + Phase 2 (StreamingDisk uses the same
// API with TempFile-backed h_vals_in / output arenas).
//
// What this is NOT:
//   - It's not a sort. Within each output bucket, entries are in
//     arbitrary order (atomic-claim). Caller runs
//     launch_sort_pairs_u32_u64 (Phase 1.3a) per bucket to finish.
//   - It doesn't free the input. The input keys + vals stay in
//     their original buffers; the output is fresh memory the
//     caller owns.
//   - It doesn't pick num_top_bits — caller passes it. Phase 1.3c
//     will use the same value the two-level sort picks.
//
// Cost model (k=28, ~250M entries, num_top_bits=8 → 256 buckets):
//   Histogram pass: one full-cap u32 scan + atomic increments on
//     a 256 × u32 hist array. ~20 ms on Gen4/RTX-class.
//   Partition pass: per-tile (~16 MB) H2D + partition kernel with
//     zero-copy writes to host-pinned bucket arenas. Random PCIe
//     writes dominate — ~3 GB of partition output × ~7 GB/s
//     realistic random PCIe = ~430 ms wall. Slow but acceptable
//     given the alternative (full-cap d_t1_meta on device) doesn't
//     fit on a 2-3 GB card.
//
// API contract:
//   d_keys_in:     count × u32 on device. Source keys (e.g. d_t1_mi).
//                  Not clobbered.
//   h_vals_in:     count × u64 host-pinned. Source values (e.g.
//                  h_t1_meta unsorted). Not clobbered.
//   h_part_keys:   count × u32 host-pinned. Bucketed output keys.
//   h_part_vals:   count × u64 host-pinned. Bucketed output vals.
//   h_bucket_starts: (num_buckets+1) × u32 host-pinned. Exclusive-
//                  scan offsets; bucket b occupies
//                  [h_bucket_starts[b], h_bucket_starts[b+1]).
//                  Tail entry equals count.
//   tile_count:    >=1. Source is split into this many roughly-equal
//                  tiles for the partition pass. 0 selects a
//                  reasonable default (~16 MB per tile of u64).
//
// d_scratch / scratch_bytes follow the same query-then-execute
// contract as launch_sort_pairs.

#pragma once

#include <cstdint>
#include <cstddef>

#include <sycl/sycl.hpp>

namespace pos2gpu {

// P1.5 host-RAM disk-offload (XCHPLOT2_SPILL_T1META): overlapped
// source-tile reader for the streaming-partition spill path. When a
// non-null SpillTileReader is passed, pass-2 does NOT read from
// h_vals_in (which is then null); instead each source-values tile is
// pread from disk by a background I/O worker into one of TWO ping-pong
// windows, double-buffered so tile t+1's pread overlaps tile t's
// partition kernel. The primitive drives it:
//   submit(ctx, slot, off_bytes, bytes)  — enqueue an async pread of
//       `bytes` at file offset `off_bytes` into window `slot` (0/1).
//   wait(ctx, slot)                      — block until window `slot`'s
//       most recent submit has completed.
// win[slot] is the pinned destination window; each tile must be
// <= win_entries u64 entries (caller sizes tile_count to guarantee it).
// Null reader keeps the in-memory h_vals_in path, byte-identical.
struct SpillTileReader {
    void*     ctx         = nullptr;
    uint64_t* win[2]      = {nullptr, nullptr};
    uint64_t  win_entries = 0;
    bool      overlap     = true;   // false => synchronous per-tile (measurement only)
    void (*submit)(void* ctx, int slot, uint64_t off_bytes, uint64_t bytes) = nullptr;
    void (*wait)(void* ctx, int slot)                                       = nullptr;
};

void launch_streaming_partition_u32_u64(
    void* d_scratch,
    size_t& scratch_bytes,
    uint32_t const* d_keys_in,
    uint64_t const* h_vals_in,
    uint32_t* h_part_keys,
    uint64_t* h_part_vals,
    uint32_t* h_bucket_starts,
    uint64_t count,
    int top_bit_offset,
    int num_top_bits,
    uint64_t tile_count,
    sycl::queue& q,
    // P1.5 host-RAM disk-offload: non-null routes pass-2's source-values
    // tiles through the double-buffered disk reader instead of h_vals_in
    // (which is then null). Null = in-memory path, byte-identical.
    SpillTileReader const* spill_reader = nullptr);

// Triple-val variant: same shape as launch_streaming_partition_u32_u64
// but also carries a u32 second-value array alongside the u64 first
// value. Used by Phase 1.5 (T2 sort streaming) where each entry has
// a u64 meta AND a u32 xbits that must stay paired through the sort.
//
// Why two outputs in ONE pass rather than two separate partition
// calls: each call's atomic-claim ordering is non-deterministic, so
// running launch_streaming_partition_u32_u64 twice (once with meta as
// val, once with xbits) would produce DIFFERENT slot orderings for
// duplicate keys → meta[i] and xbits[i] would belong to different
// original entries. Carrying both vals through a single partition
// preserves the meta-xbits pairing.
//
// API mirrors u32_u64 with an extra (h_vals2_in, h_part_vals2) pair,
// including the optional spill readers: pass a reader for either value
// stream to feed that stream's per-tile source from a TempFile instead of
// the host array (which is then null). Both null = in-memory path,
// byte-identical.
//
// The two readers share the engine's window pair, one window each — the
// u64 stream reads into window 0, the u32 stream into window 1 — so a tile
// of BOTH streams is resident at once. That costs the cross-tile prefetch
// the single-stream u32_u64 variant gets (which needs both windows for one
// stream), so this path is unoverlapped by construction. Deliberate: the
// simplest correct mechanism first, and whether overlap is worth another
// window pair is a question for measurement, not assumption.
// Both vals are written at the same atomic-claim slot, so the i-th
// output triple always corresponds to a single input position.
void launch_streaming_partition_u32_u64_u32(
    void* d_scratch,
    size_t& scratch_bytes,
    uint32_t const* d_keys_in,
    uint64_t const* h_vals_in,
    uint32_t const* h_vals2_in,
    uint32_t* h_part_keys,
    uint64_t* h_part_vals,
    uint32_t* h_part_vals2,
    uint32_t* h_bucket_starts,
    uint64_t count,
    int top_bit_offset,
    int num_top_bits,
    uint64_t tile_count,
    sycl::queue& q,
    // Non-null routes that value stream's per-tile source through the
    // double-buffered disk reader instead of its h_*_in array (which is
    // then null). Independent: either, both, or neither.
    SpillTileReader const* vals_reader  = nullptr,
    SpillTileReader const* vals2_reader = nullptr);

} // namespace pos2gpu
