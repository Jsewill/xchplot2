// TwoLevelSort.cuh — destination-tile-bucketed radix sort.
//
// Wraps the existing launch_sort_pairs_u32_u32 with a top-bits
// partition pass so the output is naturally segmented into
// 2^num_top_bits contiguous buckets. Each bucket can then be
// processed (gathered, scattered, D2H'd, consumed by the next
// pipeline phase) without cross-bucket random access.
//
// This is the gating primitive for the StreamingPinned and
// StreamingDisk tiers: it lets the T1/T2 sort produce output that
// can be streamed to host-pinned or disk in tile-sized chunks
// without holding the full sorted array on device. See the spec
// memory project_streaming_pinned_disk_spec.
//
// Output is byte-identical to launch_sort_pairs_u32_u32 on the
// same (count, begin_bit, end_bit) inputs (assuming unique keys —
// CUB radix sort is unstable within equal keys, and so is this
// wrapper). The d_bucket_starts output is a separate gain: callers
// can use it to drive per-bucket downstream work.
//
// Trade-offs vs single-level sort:
//   + output is segmented — enables per-bucket streaming downstream
//   + per-bucket sort scratch shrinks roughly proportional to 1 /
//     num_buckets vs single-level (per-call, not amortized — we
//     reuse one scratch arena across buckets)
//   − one extra histogram pass + one partition pass (~3× N
//     reads + 1× N atomic-write before per-bucket sort)
//   − 2^num_top_bits sort launches vs 1 (launch overhead dominates
//     for small buckets — pick num_top_bits so buckets stay > ~16K
//     entries)
//   − needs 2 × count × u32 of temp storage for the partition
//     intermediate (same magnitude as a single-level sort's scratch)
//
// Memory layout of d_temp_storage (caller's responsibility to
// allocate after a query call with d_temp_storage == nullptr):
//   - inner sort scratch (max-bucket-sized)
//   - d_keys_partition       (count × u32)
//   - d_vals_partition       (count × u32)
//   - d_cursors              ((1 << num_top_bits) × u32)
//
// Layout details are private to TwoLevelSortSycl.cpp; callers just
// pass the contiguous buffer back.

#pragma once

#include <cstdint>
#include <cstddef>

#include <sycl/sycl.hpp>

namespace pos2gpu {

// Two-level radix sort wrapping launch_sort_pairs_u32_u32.
//
// d_temp_storage / temp_bytes follow the same query-then-execute
// contract as launch_sort_pairs_u32_u32: pass NULL to query the
// required size, pass real storage to execute.
//
// d_bucket_starts must point to (1 << num_top_bits) + 1 u32s of
// device memory; on return holds exclusive-scan offsets such that
// bucket b occupies output positions [d_bucket_starts[b],
// d_bucket_starts[b+1]). The tail entry equals count.
//
// num_top_bits picks the bucket count. Must satisfy
//   begin_bit + num_top_bits <= end_bit
// so there's at least one bit left to sort within each bucket.
// Practical range: 4 (16 buckets) through 12 (4096 buckets).
// At k=28 (count ~2.5e8) num_top_bits=8 gives ~1M entries per
// bucket — sweet spot for amortizing launch overhead while
// keeping per-bucket peak low.
void launch_two_level_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    int num_top_bits,
    uint32_t* d_bucket_starts,
    sycl::queue& q);

} // namespace pos2gpu
