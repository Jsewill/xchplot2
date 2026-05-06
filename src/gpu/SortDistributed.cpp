// SortDistributed.cpp — implementation of the multi-shard sort wrapper.
//
// Phase 2.0b: N == 1 fast-path delegates to the single-shard sort in
// Sort.cuh.
//
// Phase 2.1: N >= 2 — distributed radix sort via host-pinned bounce.
// Each shard's output owns a contiguous bucket range of the sorted
// keyspace. Algorithm:
//
//   1. D2H — every source shard memcpy's its inputs to its own
//      host-pinned staging region. Wait on every source queue.
//   2. Host-side scatter — for each (source, receiver) pair, walk
//      the source's inputs and append items whose bucket destination
//      is the receiver to the receiver's host-pinned recv buffer.
//      Iterating sources in shard-id ascending order preserves the
//      cross-shard concatenation order so a single stable sort over
//      [shard 0 input | shard 1 input | …] is reproduced byte-for-byte.
//   3. H2D — each receiver memcpy's its received host-pinned buffer to
//      its device input scratch.
//   4. Per-receiver local single-shard sort on the gathered bucket
//      range. Stable; preserves the source-ordering carried through
//      step 2.
//   5. The receiver's keys_out / vals_out hold the sorted bucket;
//      out_count is filled in.
//
// Performance: 1× full D2H + 1× full H2H copy (the scatter walk) +
// 1× full H2D + per-receiver local sort. PCIe-bound on multi-physical-
// GPU hosts; on single-physical-GPU hosts (the dev-box parity-test
// configuration) the H2D/D2H still runs but stays on one device.
// Phase 2.4 swaps the host-pinned bounce for direct device-to-device
// peer copies where the topology supports it.
//
// See docs/multi-gpu-single-plot-alt-bucket-partition.md.

#include "gpu/SortDistributed.hpp"
#include "gpu/Sort.cuh"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

namespace {

// Map a u32 key to a destination shard 0..N-1 such that the buckets
// are in sort order (items in bucket k all sort before items in
// bucket k+1). Splits the [0, 2^bits) value-space of the active sort
// bits into N equal ranges where bits = end_bit - begin_bit.
//
// For begin_bit=0, end_bit=32 the active range is the full key. For
// partial-range sorts (end_bit < 32 or begin_bit > 0) the bucket is
// computed against the active bits only — the inactive high/low bits
// don't affect sort order, so they don't affect bucket assignment
// either.
std::size_t bucket_of_u32(std::uint32_t key, std::size_t N,
                          int begin_bit, int end_bit)
{
    int const bits = end_bit - begin_bit;
    if (bits >= 32) {
        return (static_cast<std::uint64_t>(key) *
                static_cast<std::uint64_t>(N)) >> 32;
    }
    std::uint32_t const mask  = (1u << bits) - 1u;
    std::uint32_t const value = (key >> begin_bit) & mask;
    return (static_cast<std::uint64_t>(value) *
            static_cast<std::uint64_t>(N)) >> bits;
}

// u64 analogue. For active-bit ranges <= 32 we work in 32 bits; for
// > 32 we use the upper 32 bits to define buckets (equivalent to
// splitting [0, 2^bits) into N ranges, since the upper portion
// dominates sort order for pos2-chip's u64 sorts).
std::size_t bucket_of_u64(std::uint64_t key, std::size_t N,
                          int begin_bit, int end_bit)
{
    int const bits = end_bit - begin_bit;
    if (bits >= 64) {
        std::uint32_t const hi = static_cast<std::uint32_t>(key >> 32);
        return (static_cast<std::uint64_t>(hi) *
                static_cast<std::uint64_t>(N)) >> 32;
    }
    if (bits > 32) {
        // Active range straddles the 32-bit boundary; use the top
        // (bits - 32) bits relative to begin_bit as the discriminator.
        int const shift = begin_bit + (bits - 32);
        std::uint64_t const hi_mask = (1ull << (bits - 32)) - 1ull;
        std::uint64_t const value = (key >> shift) & hi_mask;
        return (value * static_cast<std::uint64_t>(N)) >> (bits - 32);
    }
    std::uint64_t const mask  = (1ull << bits) - 1ull;
    std::uint64_t const value = (key >> begin_bit) & mask;
    return (value * static_cast<std::uint64_t>(N)) >> bits;
}

// Allocate sycl::malloc_host on a sentinel queue. Any queue works
// (pinned host memory is process-wide, not queue-scoped) — pick the
// first shard's so we don't take an extra queue handle.
template <class T>
T* pinned_alloc(std::size_t n, sycl::queue& q)
{
    return sycl::malloc_host<T>(n, q);
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
            s.out_count = s.count;
        }
        return;
    }

    // Sizing-query call: Phase 2.1 allocates its own scratch
    // internally (host pinned + per-receiver device sort scratch),
    // so the external temp_bytes is 0 for N>=2. Future Phase 2.4 may
    // expose an opt-in "caller owns scratch" path for users who want
    // to amortize allocations across plots.
    if (d_temp_storage == nullptr) {
        temp_bytes = 0;
        return;
    }

    std::size_t const N = shards.size();

    // Step 1: D2H every source shard's inputs into per-source pinned
    // host buffers. Use shards[0]'s queue as the alloc context.
    sycl::queue& alloc_q = *shards[0].queue;
    std::vector<std::uint32_t*> host_keys(N, nullptr);
    std::vector<std::uint32_t*> host_vals(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        std::size_t const c = static_cast<std::size_t>(shards[s].count);
        host_keys[s] = pinned_alloc<std::uint32_t>(c == 0 ? 1 : c, alloc_q);
        host_vals[s] = pinned_alloc<std::uint32_t>(c == 0 ? 1 : c, alloc_q);
        if (c > 0) {
            shards[s].queue->memcpy(host_keys[s], shards[s].keys_in,
                                    c * sizeof(std::uint32_t));
            shards[s].queue->memcpy(host_vals[s], shards[s].vals_in,
                                    c * sizeof(std::uint32_t));
        }
    }
    for (std::size_t s = 0; s < N; ++s) shards[s].queue->wait();

    // Step 2: Host-side scatter into per-receiver pinned buffers.
    // First pass counts items per (source, receiver). Second pass
    // copies. Walking sources in ascending shard-id preserves the
    // cross-shard concatenation order that a single stable sort over
    // [shard 0 input | … | shard N-1 input] would see.
    std::vector<std::vector<std::size_t>> counts_src_dst(
        N, std::vector<std::size_t>(N, 0));
    for (std::size_t s = 0; s < N; ++s) {
        std::size_t const c = static_cast<std::size_t>(shards[s].count);
        for (std::size_t i = 0; i < c; ++i) {
            std::size_t const d = bucket_of_u32(host_keys[s][i], N,
                                                begin_bit, end_bit);
            ++counts_src_dst[s][d];
        }
    }

    std::vector<std::size_t> recv_count(N, 0);
    for (std::size_t s = 0; s < N; ++s)
        for (std::size_t d = 0; d < N; ++d)
            recv_count[d] += counts_src_dst[s][d];

    // Allocate per-receiver pinned recv buffers (worst case is the
    // total input, but recv_count[d] is exact — no over-alloc).
    std::vector<std::uint32_t*> recv_keys(N, nullptr);
    std::vector<std::uint32_t*> recv_vals(N, nullptr);
    for (std::size_t d = 0; d < N; ++d) {
        std::size_t const c = recv_count[d] == 0 ? 1 : recv_count[d];
        recv_keys[d] = pinned_alloc<std::uint32_t>(c, alloc_q);
        recv_vals[d] = pinned_alloc<std::uint32_t>(c, alloc_q);
    }

    // Scatter walk.
    std::vector<std::size_t> recv_pos(N, 0);
    for (std::size_t s = 0; s < N; ++s) {
        std::size_t const c = static_cast<std::size_t>(shards[s].count);
        for (std::size_t i = 0; i < c; ++i) {
            std::uint32_t const k = host_keys[s][i];
            std::uint32_t const v = host_vals[s][i];
            std::size_t const d = bucket_of_u32(k, N, begin_bit, end_bit);
            recv_keys[d][recv_pos[d]] = k;
            recv_vals[d][recv_pos[d]] = v;
            ++recv_pos[d];
        }
    }
    for (std::size_t d = 0; d < N; ++d) {
        // sanity (defensive): recv_pos[d] must equal recv_count[d]
        if (recv_pos[d] != recv_count[d]) {
            throw std::runtime_error(
                "launch_sort_pairs_u32_u32_distributed: scatter walk "
                "produced inconsistent counts (internal bug)");
        }
    }

    // Free source pinned buffers — no longer needed.
    for (std::size_t s = 0; s < N; ++s) {
        sycl::free(host_keys[s], alloc_q);
        sycl::free(host_vals[s], alloc_q);
    }

    // Step 3 & 4: H2D and per-receiver local sort. Each receiver's
    // keys_in / vals_in serve as the H2D landing zone (they're scratch
    // per the public ping-pong contract); keys_out / vals_out hold the
    // sorted result.
    for (std::size_t d = 0; d < N; ++d) {
        DistributedSortPairsShard& sd = shards[d];
        std::size_t const cd = recv_count[d];
        if (cd > sd.out_capacity) {
            throw std::runtime_error(std::string(
                "launch_sort_pairs_u32_u32_distributed: shard ")
                + std::to_string(d) + " received " + std::to_string(cd)
                + " elements but its out_capacity is only "
                + std::to_string(sd.out_capacity)
                + ". Sizing must allow each shard to receive up to "
                "the total input count in the worst case (skewed input "
                "distribution).");
        }
        sd.out_count = cd;
        if (cd == 0) continue;

        sd.queue->memcpy(sd.keys_in, recv_keys[d],
                         cd * sizeof(std::uint32_t));
        sd.queue->memcpy(sd.vals_in, recv_vals[d],
                         cd * sizeof(std::uint32_t));
        sd.queue->wait();

        // Per-receiver sort scratch sizing query, then alloc, then sort.
        std::size_t scratch_bytes = 0;
        launch_sort_pairs_u32_u32(
            nullptr, scratch_bytes, nullptr, nullptr, nullptr, nullptr,
            cd, begin_bit, end_bit, *sd.queue);
        void* scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, *sd.queue) : nullptr;

        launch_sort_pairs_u32_u32(
            scratch ? scratch : reinterpret_cast<void*>(std::uintptr_t{1}),
            scratch_bytes,
            sd.keys_in, sd.keys_out, sd.vals_in, sd.vals_out,
            cd, begin_bit, end_bit, *sd.queue);
        sd.queue->wait();

        if (scratch) sycl::free(scratch, *sd.queue);
    }

    // Free recv pinned buffers.
    for (std::size_t d = 0; d < N; ++d) {
        sycl::free(recv_keys[d], alloc_q);
        sycl::free(recv_vals[d], alloc_q);
    }
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

    if (d_temp_storage == nullptr) {
        temp_bytes = 0;
        return;
    }

    std::size_t const N = shards.size();
    sycl::queue& alloc_q = *shards[0].queue;

    std::vector<std::uint64_t*> host_keys(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        std::size_t const c = static_cast<std::size_t>(shards[s].count);
        host_keys[s] = pinned_alloc<std::uint64_t>(c == 0 ? 1 : c, alloc_q);
        if (c > 0) {
            shards[s].queue->memcpy(host_keys[s], shards[s].keys_in,
                                    c * sizeof(std::uint64_t));
        }
    }
    for (std::size_t s = 0; s < N; ++s) shards[s].queue->wait();

    std::vector<std::vector<std::size_t>> counts_src_dst(
        N, std::vector<std::size_t>(N, 0));
    for (std::size_t s = 0; s < N; ++s) {
        std::size_t const c = static_cast<std::size_t>(shards[s].count);
        for (std::size_t i = 0; i < c; ++i) {
            std::size_t const d = bucket_of_u64(host_keys[s][i], N,
                                                begin_bit, end_bit);
            ++counts_src_dst[s][d];
        }
    }
    std::vector<std::size_t> recv_count(N, 0);
    for (std::size_t s = 0; s < N; ++s)
        for (std::size_t d = 0; d < N; ++d)
            recv_count[d] += counts_src_dst[s][d];

    std::vector<std::uint64_t*> recv_keys(N, nullptr);
    for (std::size_t d = 0; d < N; ++d) {
        std::size_t const c = recv_count[d] == 0 ? 1 : recv_count[d];
        recv_keys[d] = pinned_alloc<std::uint64_t>(c, alloc_q);
    }

    std::vector<std::size_t> recv_pos(N, 0);
    for (std::size_t s = 0; s < N; ++s) {
        std::size_t const c = static_cast<std::size_t>(shards[s].count);
        for (std::size_t i = 0; i < c; ++i) {
            std::uint64_t const k = host_keys[s][i];
            std::size_t const d = bucket_of_u64(k, N, begin_bit, end_bit);
            recv_keys[d][recv_pos[d]] = k;
            ++recv_pos[d];
        }
    }
    for (std::size_t d = 0; d < N; ++d) {
        if (recv_pos[d] != recv_count[d]) {
            throw std::runtime_error(
                "launch_sort_keys_u64_distributed: scatter walk produced "
                "inconsistent counts (internal bug)");
        }
    }

    for (std::size_t s = 0; s < N; ++s) sycl::free(host_keys[s], alloc_q);

    for (std::size_t d = 0; d < N; ++d) {
        DistributedSortKeysU64Shard& sd = shards[d];
        std::size_t const cd = recv_count[d];
        if (cd > sd.out_capacity) {
            throw std::runtime_error(std::string(
                "launch_sort_keys_u64_distributed: shard ")
                + std::to_string(d) + " received " + std::to_string(cd)
                + " elements but its out_capacity is only "
                + std::to_string(sd.out_capacity));
        }
        sd.out_count = cd;
        if (cd == 0) continue;

        sd.queue->memcpy(sd.keys_in, recv_keys[d],
                         cd * sizeof(std::uint64_t));
        sd.queue->wait();

        std::size_t scratch_bytes = 0;
        launch_sort_keys_u64(
            nullptr, scratch_bytes, nullptr, nullptr,
            cd, begin_bit, end_bit, *sd.queue);
        void* scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, *sd.queue) : nullptr;

        launch_sort_keys_u64(
            scratch ? scratch : reinterpret_cast<void*>(std::uintptr_t{1}),
            scratch_bytes,
            sd.keys_in, sd.keys_out,
            cd, begin_bit, end_bit, *sd.queue);
        sd.queue->wait();

        if (scratch) sycl::free(scratch, *sd.queue);
    }

    for (std::size_t d = 0; d < N; ++d) sycl::free(recv_keys[d], alloc_q);
}

} // namespace pos2gpu
