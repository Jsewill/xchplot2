// SortDistributed.cpp — implementation of the multi-shard sort wrapper.
//
// Phase 2.0b: N == 1 fast-path delegates to the single-shard sort in
// Sort.cuh. N > 1 throws std::runtime_error pointing at the design doc.

#include "gpu/SortDistributed.hpp"
#include "gpu/Sort.cuh"

#include <stdexcept>
#include <string>

namespace pos2gpu {

namespace {

[[noreturn]] void throw_n_gt_1(std::size_t shard_count, char const* which)
{
    throw std::runtime_error(std::string(
        "launch_sort_") + which + "_distributed: distributed sort across "
        + std::to_string(shard_count) +
        " shards is not yet implemented (Phase 2.0b ships only the N=1 "
        "fast path; the per-shard distributed radix lands in Phase 2.1). "
        "See docs/multi-gpu-single-plot-alt-bucket-partition.md for the "
        "planned design.");
}

} // namespace

void launch_sort_pairs_u32_u32_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortPairsShard>& shards,
    int begin_bit, int end_bit)
{
    if (shards.empty()) {
        throw std::runtime_error(
            "launch_sort_pairs_u32_u32_distributed: shards.empty() — "
            "no work to do, but no graceful no-op contract either; "
            "callers must pass at least one shard.");
    }

    if (shards.size() == 1) {
        // Fast path: delegate to single-shard sort. Output count equals
        // input count by definition (no scatter, no bucket split).
        DistributedSortPairsShard& s = shards[0];
        launch_sort_pairs_u32_u32(
            d_temp_storage, temp_bytes,
            s.keys_in, s.keys_out, s.vals_in, s.vals_out,
            s.count, begin_bit, end_bit, *s.queue);
        if (d_temp_storage != nullptr) {
            // Sort actually ran; record the output count.
            s.out_count = s.count;
        }
        return;
    }

    throw_n_gt_1(shards.size(), "pairs_u32_u32");
}

void launch_sort_keys_u64_distributed(
    void* d_temp_storage,
    std::size_t& temp_bytes,
    std::vector<DistributedSortKeysU64Shard>& shards,
    int begin_bit, int end_bit)
{
    if (shards.empty()) {
        throw std::runtime_error(
            "launch_sort_keys_u64_distributed: shards.empty() — "
            "callers must pass at least one shard.");
    }

    if (shards.size() == 1) {
        DistributedSortKeysU64Shard& s = shards[0];
        launch_sort_keys_u64(
            d_temp_storage, temp_bytes,
            s.keys_in, s.keys_out,
            s.count, begin_bit, end_bit, *s.queue);
        if (d_temp_storage != nullptr) {
            s.out_count = s.count;
        }
        return;
    }

    throw_n_gt_1(shards.size(), "keys_u64");
}

} // namespace pos2gpu
