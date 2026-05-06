// sycl_distributed_sort_parity — exercises the Phase 2.0b N=1 fast path
// of launch_sort_pairs_u32_u32_distributed / launch_sort_keys_u64_distributed
// and confirms it produces byte-identical output to the single-shard
// sort. Acts as a regression test so the fast-path delegation in
// SortDistributed.cpp doesn't drift from Sort.cuh's behavior as the
// distributed sort grows in Phase 2.1+.
//
// Doesn't exercise N>1 — that path throws today and needs multi-GPU
// hardware to validate, which lands in a separate parity test once
// Phase 2.1 implements it.

#include "gpu/Sort.cuh"
#include "gpu/SortDistributed.hpp"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool run_pairs(uint32_t seed, uint64_t count)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(seed);
    std::vector<uint32_t> h_keys(count), h_vals(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i] = static_cast<uint32_t>(i);
        h_vals[i] = static_cast<uint32_t>(i);
    }
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    // Reference: existing single-shard launch_sort_pairs_u32_u32.
    std::vector<uint32_t> ref_keys(count), ref_vals(count);
    {
        uint32_t* d_keys_in  = sycl::malloc_device<uint32_t>(count, q);
        uint32_t* d_keys_out = sycl::malloc_device<uint32_t>(count, q);
        uint32_t* d_vals_in  = sycl::malloc_device<uint32_t>(count, q);
        uint32_t* d_vals_out = sycl::malloc_device<uint32_t>(count, q);
        q.memcpy(d_keys_in, h_keys.data(), sizeof(uint32_t) * count);
        q.memcpy(d_vals_in, h_vals.data(), sizeof(uint32_t) * count).wait();

        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_pairs_u32_u32(
            nullptr, scratch_bytes, nullptr, nullptr, nullptr, nullptr,
            count, 0, 32, q);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;

        pos2gpu::launch_sort_pairs_u32_u32(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            count, 0, 32, q);
        q.wait();

        q.memcpy(ref_keys.data(), d_keys_out, sizeof(uint32_t) * count);
        q.memcpy(ref_vals.data(), d_vals_out, sizeof(uint32_t) * count).wait();

        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_keys_in,  q);
        sycl::free(d_keys_out, q);
        sycl::free(d_vals_in,  q);
        sycl::free(d_vals_out, q);
    }

    // Under-test: distributed sort with N=1.
    std::vector<uint32_t> dist_keys(count), dist_vals(count);
    uint64_t dist_out_count = 0;
    {
        uint32_t* d_keys_in  = sycl::malloc_device<uint32_t>(count, q);
        uint32_t* d_keys_out = sycl::malloc_device<uint32_t>(count, q);
        uint32_t* d_vals_in  = sycl::malloc_device<uint32_t>(count, q);
        uint32_t* d_vals_out = sycl::malloc_device<uint32_t>(count, q);
        q.memcpy(d_keys_in, h_keys.data(), sizeof(uint32_t) * count);
        q.memcpy(d_vals_in, h_vals.data(), sizeof(uint32_t) * count).wait();

        std::vector<pos2gpu::DistributedSortPairsShard> shards(1);
        shards[0].queue = &q;
        shards[0].keys_in = d_keys_in;
        shards[0].vals_in = d_vals_in;
        shards[0].count = count;
        shards[0].keys_out = d_keys_out;
        shards[0].vals_out = d_vals_out;
        shards[0].out_capacity = count;
        shards[0].out_count = 0;

        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_pairs_u32_u32_distributed(
            nullptr, scratch_bytes, shards, 0, 32);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;

        auto const t0 = std::chrono::steady_clock::now();
        pos2gpu::launch_sort_pairs_u32_u32_distributed(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes, shards, 0, 32);
        q.wait();
        auto const t1 = std::chrono::steady_clock::now();
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        (void)ms;

        dist_out_count = shards[0].out_count;

        q.memcpy(dist_keys.data(), d_keys_out, sizeof(uint32_t) * count);
        q.memcpy(dist_vals.data(), d_vals_out, sizeof(uint32_t) * count).wait();

        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_keys_in,  q);
        sycl::free(d_keys_out, q);
        sycl::free(d_vals_in,  q);
        sycl::free(d_vals_out, q);
    }

    bool const keys_ok = std::memcmp(ref_keys.data(), dist_keys.data(),
                                     sizeof(uint32_t) * count) == 0;
    bool const vals_ok = std::memcmp(ref_vals.data(), dist_vals.data(),
                                     sizeof(uint32_t) * count) == 0;
    bool const count_ok = (dist_out_count == count);
    bool const ok = keys_ok && vals_ok && count_ok;

    std::printf("%s pairs N=1 seed=%u count=%lu  [keys=%d vals=%d count=%d]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long)count,
                keys_ok ? 1 : 0, vals_ok ? 1 : 0, count_ok ? 1 : 0);
    return ok;
}

bool run_keys(uint32_t seed, uint64_t count)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> h_keys(count);
    for (uint64_t i = 0; i < count; ++i) h_keys[i] = i * 37u + 1u;
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    // Reference.
    std::vector<uint64_t> ref_keys(count);
    {
        uint64_t* d_in  = sycl::malloc_device<uint64_t>(count, q);
        uint64_t* d_out = sycl::malloc_device<uint64_t>(count, q);
        q.memcpy(d_in, h_keys.data(), sizeof(uint64_t) * count).wait();

        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_keys_u64(
            nullptr, scratch_bytes, nullptr, nullptr,
            count, 0, 64, q);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;

        pos2gpu::launch_sort_keys_u64(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes, d_in, d_out, count, 0, 64, q);
        q.wait();

        q.memcpy(ref_keys.data(), d_out, sizeof(uint64_t) * count).wait();
        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_in,  q);
        sycl::free(d_out, q);
    }

    // Distributed N=1.
    std::vector<uint64_t> dist_keys(count);
    uint64_t dist_out_count = 0;
    {
        uint64_t* d_in  = sycl::malloc_device<uint64_t>(count, q);
        uint64_t* d_out = sycl::malloc_device<uint64_t>(count, q);
        q.memcpy(d_in, h_keys.data(), sizeof(uint64_t) * count).wait();

        std::vector<pos2gpu::DistributedSortKeysU64Shard> shards(1);
        shards[0].queue = &q;
        shards[0].keys_in = d_in;
        shards[0].count = count;
        shards[0].keys_out = d_out;
        shards[0].out_capacity = count;
        shards[0].out_count = 0;

        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_keys_u64_distributed(
            nullptr, scratch_bytes, shards, 0, 64);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;

        pos2gpu::launch_sort_keys_u64_distributed(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes, shards, 0, 64);
        q.wait();

        dist_out_count = shards[0].out_count;
        q.memcpy(dist_keys.data(), d_out, sizeof(uint64_t) * count).wait();
        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_in,  q);
        sycl::free(d_out, q);
    }

    bool const keys_ok = std::memcmp(ref_keys.data(), dist_keys.data(),
                                     sizeof(uint64_t) * count) == 0;
    bool const count_ok = (dist_out_count == count);
    bool const ok = keys_ok && count_ok;

    std::printf("%s keys  N=1 seed=%u count=%lu  [match=%d count=%d]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long)count,
                keys_ok ? 1 : 0, count_ok ? 1 : 0);
    return ok;
}

// Phase 2.1 — pairs sort with N=2 virtual shards (both pointing at the
// same physical queue on the dev box). Splits the input arbitrarily
// across the two shards, then verifies the union of distributed
// outputs equals what a single-shard sort over the concatenated input
// would produce. Validates correctness of the host-pinned bounce
// algorithm on hardware accessible to a 1-GPU dev box; real multi-
// physical-GPU runs are downstream once a multi-GPU rig is reachable.
bool run_pairs_n2(uint32_t seed, uint64_t total)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(seed);
    std::vector<uint32_t> h_keys(total), h_vals(total);
    for (uint64_t i = 0; i < total; ++i) {
        h_keys[i] = static_cast<uint32_t>(i);
        h_vals[i] = static_cast<uint32_t>(i);
    }
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    // Reference: single-shard sort over the FULL concatenated input.
    std::vector<uint32_t> ref_keys(total), ref_vals(total);
    {
        uint32_t* d_keys_in  = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_keys_out = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_vals_in  = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_vals_out = sycl::malloc_device<uint32_t>(total, q);
        q.memcpy(d_keys_in, h_keys.data(), sizeof(uint32_t) * total);
        q.memcpy(d_vals_in, h_vals.data(), sizeof(uint32_t) * total).wait();

        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_pairs_u32_u32(
            nullptr, scratch_bytes, nullptr, nullptr, nullptr, nullptr,
            total, 0, 32, q);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;
        pos2gpu::launch_sort_pairs_u32_u32(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            total, 0, 32, q);
        q.wait();
        q.memcpy(ref_keys.data(), d_keys_out, sizeof(uint32_t) * total);
        q.memcpy(ref_vals.data(), d_vals_out, sizeof(uint32_t) * total).wait();
        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_keys_in,  q);
        sycl::free(d_keys_out, q);
        sycl::free(d_vals_in,  q);
        sycl::free(d_vals_out, q);
    }

    // Distributed N=2: split input into [0, total/2) on shard 0,
    // [total/2, total) on shard 1. Each shard's out_capacity is `total`
    // because in the worst skew case all items could land in one shard's
    // bucket.
    uint64_t const split = total / 2;
    uint64_t const c0 = split;
    uint64_t const c1 = total - split;

    std::vector<uint32_t> dist_combined_keys, dist_combined_vals;
    dist_combined_keys.reserve(total);
    dist_combined_vals.reserve(total);

    {
        uint32_t* d_keys_in_0  = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_keys_in_1  = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_keys_out_0 = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_keys_out_1 = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_vals_in_0  = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_vals_in_1  = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_vals_out_0 = sycl::malloc_device<uint32_t>(total, q);
        uint32_t* d_vals_out_1 = sycl::malloc_device<uint32_t>(total, q);

        q.memcpy(d_keys_in_0, h_keys.data(),      sizeof(uint32_t) * c0);
        q.memcpy(d_keys_in_1, h_keys.data() + c0, sizeof(uint32_t) * c1);
        q.memcpy(d_vals_in_0, h_vals.data(),      sizeof(uint32_t) * c0);
        q.memcpy(d_vals_in_1, h_vals.data() + c0, sizeof(uint32_t) * c1).wait();

        std::vector<pos2gpu::DistributedSortPairsShard> shards(2);
        shards[0].queue = &q;
        shards[0].keys_in = d_keys_in_0;
        shards[0].vals_in = d_vals_in_0;
        shards[0].count = c0;
        shards[0].keys_out = d_keys_out_0;
        shards[0].vals_out = d_vals_out_0;
        shards[0].out_capacity = total;
        shards[0].out_count = 0;

        shards[1].queue = &q;
        shards[1].keys_in = d_keys_in_1;
        shards[1].vals_in = d_vals_in_1;
        shards[1].count = c1;
        shards[1].keys_out = d_keys_out_1;
        shards[1].vals_out = d_vals_out_1;
        shards[1].out_capacity = total;
        shards[1].out_count = 0;

        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_pairs_u32_u32_distributed(
            nullptr, scratch_bytes, shards, 0, 32);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;
        pos2gpu::launch_sort_pairs_u32_u32_distributed(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes, shards, 0, 32);
        q.wait();

        // Pull each shard's bucket-range output back to host and
        // concatenate in shard order — this is the union expected to
        // match the single-shard reference.
        std::vector<uint32_t> sk0(shards[0].out_count), sk1(shards[1].out_count);
        std::vector<uint32_t> sv0(shards[0].out_count), sv1(shards[1].out_count);
        if (shards[0].out_count > 0) {
            q.memcpy(sk0.data(), d_keys_out_0,
                     sizeof(uint32_t) * shards[0].out_count);
            q.memcpy(sv0.data(), d_vals_out_0,
                     sizeof(uint32_t) * shards[0].out_count);
        }
        if (shards[1].out_count > 0) {
            q.memcpy(sk1.data(), d_keys_out_1,
                     sizeof(uint32_t) * shards[1].out_count);
            q.memcpy(sv1.data(), d_vals_out_1,
                     sizeof(uint32_t) * shards[1].out_count);
        }
        q.wait();
        dist_combined_keys.insert(dist_combined_keys.end(), sk0.begin(), sk0.end());
        dist_combined_keys.insert(dist_combined_keys.end(), sk1.begin(), sk1.end());
        dist_combined_vals.insert(dist_combined_vals.end(), sv0.begin(), sv0.end());
        dist_combined_vals.insert(dist_combined_vals.end(), sv1.begin(), sv1.end());

        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_keys_in_0,  q); sycl::free(d_keys_in_1,  q);
        sycl::free(d_keys_out_0, q); sycl::free(d_keys_out_1, q);
        sycl::free(d_vals_in_0,  q); sycl::free(d_vals_in_1,  q);
        sycl::free(d_vals_out_0, q); sycl::free(d_vals_out_1, q);
    }

    bool const total_ok = (dist_combined_keys.size() == ref_keys.size());
    bool const keys_ok = total_ok && std::memcmp(
        ref_keys.data(), dist_combined_keys.data(),
        sizeof(uint32_t) * total) == 0;
    bool const vals_ok = total_ok && std::memcmp(
        ref_vals.data(), dist_combined_vals.data(),
        sizeof(uint32_t) * total) == 0;
    bool const ok = total_ok && keys_ok && vals_ok;
    std::printf("%s pairs N=2 seed=%u total=%lu  [size=%d keys=%d vals=%d]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long)total,
                total_ok ? 1 : 0, keys_ok ? 1 : 0, vals_ok ? 1 : 0);
    return ok;
}

// Same shape but for u64 keys (no values).
bool run_keys_n2(uint32_t seed, uint64_t total)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> h_keys(total);
    for (uint64_t i = 0; i < total; ++i) h_keys[i] = i * 37u + 1u;
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    std::vector<uint64_t> ref_keys(total);
    {
        uint64_t* d_in  = sycl::malloc_device<uint64_t>(total, q);
        uint64_t* d_out = sycl::malloc_device<uint64_t>(total, q);
        q.memcpy(d_in, h_keys.data(), sizeof(uint64_t) * total).wait();
        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_keys_u64(
            nullptr, scratch_bytes, nullptr, nullptr, total, 0, 64, q);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;
        pos2gpu::launch_sort_keys_u64(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes, d_in, d_out, total, 0, 64, q);
        q.wait();
        q.memcpy(ref_keys.data(), d_out, sizeof(uint64_t) * total).wait();
        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_in,  q); sycl::free(d_out, q);
    }

    uint64_t const split = total / 2;
    uint64_t const c0 = split, c1 = total - split;

    std::vector<uint64_t> dist_combined;
    dist_combined.reserve(total);
    {
        uint64_t* d_in_0  = sycl::malloc_device<uint64_t>(total, q);
        uint64_t* d_in_1  = sycl::malloc_device<uint64_t>(total, q);
        uint64_t* d_out_0 = sycl::malloc_device<uint64_t>(total, q);
        uint64_t* d_out_1 = sycl::malloc_device<uint64_t>(total, q);
        q.memcpy(d_in_0, h_keys.data(),      sizeof(uint64_t) * c0);
        q.memcpy(d_in_1, h_keys.data() + c0, sizeof(uint64_t) * c1).wait();

        std::vector<pos2gpu::DistributedSortKeysU64Shard> shards(2);
        shards[0].queue = &q;
        shards[0].keys_in = d_in_0; shards[0].count = c0;
        shards[0].keys_out = d_out_0; shards[0].out_capacity = total;
        shards[0].out_count = 0;
        shards[1].queue = &q;
        shards[1].keys_in = d_in_1; shards[1].count = c1;
        shards[1].keys_out = d_out_1; shards[1].out_capacity = total;
        shards[1].out_count = 0;

        size_t scratch_bytes = 0;
        pos2gpu::launch_sort_keys_u64_distributed(
            nullptr, scratch_bytes, shards, 0, 64);
        void* d_scratch = scratch_bytes
            ? sycl::malloc_device(scratch_bytes, q) : nullptr;
        pos2gpu::launch_sort_keys_u64_distributed(
            d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
            scratch_bytes, shards, 0, 64);
        q.wait();

        std::vector<uint64_t> s0(shards[0].out_count), s1(shards[1].out_count);
        if (shards[0].out_count > 0)
            q.memcpy(s0.data(), d_out_0, sizeof(uint64_t) * shards[0].out_count);
        if (shards[1].out_count > 0)
            q.memcpy(s1.data(), d_out_1, sizeof(uint64_t) * shards[1].out_count);
        q.wait();
        dist_combined.insert(dist_combined.end(), s0.begin(), s0.end());
        dist_combined.insert(dist_combined.end(), s1.begin(), s1.end());

        if (d_scratch) sycl::free(d_scratch, q);
        sycl::free(d_in_0,  q); sycl::free(d_in_1,  q);
        sycl::free(d_out_0, q); sycl::free(d_out_1, q);
    }

    bool const total_ok = (dist_combined.size() == total);
    bool const keys_ok = total_ok && std::memcmp(
        ref_keys.data(), dist_combined.data(),
        sizeof(uint64_t) * total) == 0;
    bool const ok = total_ok && keys_ok;
    std::printf("%s keys  N=2 seed=%u total=%lu  [size=%d match=%d]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long)total,
                total_ok ? 1 : 0, keys_ok ? 1 : 0);
    return ok;
}

} // namespace

int main()
{
    bool all_ok = true;
    // N=1 fast-path regression — must continue to match single-shard.
    for (uint32_t seed : {7u, 31u}) {
        for (uint64_t count : {16ull, 16384ull, 262144ull, 1048576ull}) {
            all_ok = run_pairs(seed, count) && all_ok;
            all_ok = run_keys (seed, count) && all_ok;
        }
    }
    // N=2 distributed (2 virtual shards on the same physical device on
    // single-GPU dev boxes) — validates the host-pinned bounce
    // algorithm without needing real multi-GPU hardware. Real multi-
    // physical-GPU validation lands once a multi-GPU rig is reachable.
    for (uint32_t seed : {7u, 31u}) {
        for (uint64_t total : {16ull, 16384ull, 262144ull, 1048576ull}) {
            all_ok = run_pairs_n2(seed, total) && all_ok;
            all_ok = run_keys_n2 (seed, total) && all_ok;
        }
    }
    return all_ok ? 0 : 1;
}
