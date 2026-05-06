// SortDistributed.hpp — multi-shard radix sort wrapper.
//
// Each shard provides a sycl::queue + its local input/output buffers.
// After the call, each shard's output buffer holds the elements whose
// bucket (defined by the upper bits of the key) belongs to that shard,
// sorted within that range.
//
// Phase 2.0b scope: N == 1 fast-path that delegates to the single-shard
// sort in Sort.cuh. N > 1 throws a clear error pointing at the design
// doc. Phase 2.1+ will land the real distributed radix sort
// (per-shard local count → exchange via host pinned → per-shard local
// sort + scatter to bucket-owner → final sort within bucket).
//
// See docs/multi-gpu-single-plot-alt-bucket-partition.md for the
// design rationale.

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

namespace pos2gpu {

// Per-shard handles for the distributed sort. The caller owns the
// queue and the input/output buffers; the wrapper coordinates the
// cross-shard exchange (Phase 2.1+) without taking ownership.
//
// Same in/out ping-pong contract as the single-shard sort:
// keys_in / vals_in are scratch on input — they get clobbered. The
// result lands in keys_out / vals_out.
struct DistributedSortPairsShard {
    sycl::queue* queue;

    std::uint32_t* keys_in;
    std::uint32_t* vals_in;
    std::uint64_t  count;          // input element count for this shard

    std::uint32_t* keys_out;
    std::uint32_t* vals_out;
    std::uint64_t  out_capacity;   // max elements this shard's output can hold

    // Filled by the implementation post-sort:
    std::uint64_t  out_count;      // actual elements placed in this shard's output
};

struct DistributedSortKeysU64Shard {
    sycl::queue* queue;

    std::uint64_t* keys_in;
    std::uint64_t  count;

    std::uint64_t* keys_out;
    std::uint64_t  out_capacity;

    std::uint64_t  out_count;
};

// Sort (key, value) pairs by uint32 key over [begin_bit, end_bit) bits,
// distributed across the shards.
//
// Two-call sizing-then-sort contract mirrors CUB:
//   - First call with d_temp_storage == nullptr fills temp_bytes with
//     the required scratch size (per-shard allocation; caller owns).
//     Phase 2.0b: same as single-shard temp_bytes (N=1 fast-path).
//   - Second call with d_temp_storage != nullptr performs the sort.
//     Each shard's output bucket range is determined by the wrapper.
//
// On N=1, behaves identically to launch_sort_pairs_u32_u32 against the
// sole shard's data. On N>1, throws std::runtime_error in Phase 2.0b.
void launch_sort_pairs_u32_u32_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortPairsShard>& shards,
    int begin_bit, int end_bit);

// Distributed analogue of launch_sort_keys_u64.
void launch_sort_keys_u64_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortKeysU64Shard>& shards,
    int begin_bit, int end_bit);

} // namespace pos2gpu
