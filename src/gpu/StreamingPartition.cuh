// StreamingPartition.cuh — partition (u32 keys, u64 vals) pairs by
// top-N-bits of the key, with the values streamed tile-by-tile from
// host-pinned memory and the bucketed output landing in host-pinned
// arenas via UVA-mapped writes from the partition kernel.
//
// CUDA port of the SYCL StreamingPartition primitive (Phase 1.3b of
// the streaming-pinned-disk spec). Used by Tiny tier:
//   - T1 sort streaming partition (Phase 1.3c-ii)
//   - T2 sort streaming partition via the triple-val variant (Phase 1.5b)
//
// The point: Tiny tier can't hold both d_t1_meta (full-cap u64 ~2 GB
// at k=28) AND a sort scratch on a 2-3 GB device. By streaming the
// meta tile-by-tile from host-pinned and writing partitioned output
// directly back to host-pinned bucket arenas, the device-side working
// set drops to (one tile + bucket counters + small misc), and the
// actual sort happens per-bucket on data that's already host-resident.
//
// What this is NOT:
//   - It's not a sort. Within each output bucket, entries are in
//     arbitrary order (atomic-claim). Caller runs CUB SortPairs per
//     bucket to finish.
//   - It doesn't free the input. The input keys + vals stay in
//     their original buffers; the output is fresh memory the caller
//     owns.
//   - It doesn't pick num_top_bits — caller passes it.
//
// USM-host writes from CUDA kernels: cudaMallocHost-allocated host
// memory is device-accessible via UVA on all supported compute
// capabilities (Kepler+). The writes go across PCIe — random and
// uncoalesced. Slow but bounded; the alternative (full-cap d_t1_meta
// on device) doesn't fit on a 2-3 GB card at all.
//
// API contract (mirrors the SYCL header):
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
// contract as the existing CUB calls in cuda-only.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace pos2gpu {

cudaError_t launch_streaming_partition_u32_u64(
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
    cudaStream_t stream = nullptr);

// Triple-val variant: same shape as launch_streaming_partition_u32_u64
// but also carries a u32 second-value array alongside the u64 first
// value. Used by Tiny T2 sort (Phase 1.5b) where each entry has a u64
// meta AND a u32 xbits that must stay paired through the sort.
//
// Why two outputs in ONE pass rather than two separate partition
// calls: each call's atomic-claim ordering is non-deterministic, so
// running launch_streaming_partition_u32_u64 twice (once with meta as
// val, once with xbits) would produce DIFFERENT slot orderings for
// duplicate keys → meta[i] and xbits[i] would belong to different
// original entries. Carrying both vals through a single partition
// preserves the meta-xbits pairing.
//
// API mirrors u32_u64 with an extra (h_vals2_in, h_part_vals2) pair.
// Both vals are written at the same atomic-claim slot, so the i-th
// output triple always corresponds to a single input position.
cudaError_t launch_streaming_partition_u32_u64_u32(
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
    cudaStream_t stream = nullptr);

} // namespace pos2gpu
