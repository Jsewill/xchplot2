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
    sycl::queue& q);

} // namespace pos2gpu
