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

// Smoke-test that N>1 throws a clear error (Phase 2.0b contract).
bool run_n_gt_1_throws()
{
    auto& q = pos2gpu::sycl_backend::queue();
    std::vector<pos2gpu::DistributedSortPairsShard> shards(2);
    for (auto& s : shards) {
        s.queue = &q;
        s.keys_in = s.vals_in = s.keys_out = s.vals_out = nullptr;
        s.count = 0;
        s.out_capacity = 0;
        s.out_count = 0;
    }
    std::size_t temp_bytes = 0;
    try {
        pos2gpu::launch_sort_pairs_u32_u32_distributed(
            nullptr, temp_bytes, shards, 0, 32);
    } catch (std::exception const& e) {
        std::string msg = e.what();
        bool const ok = msg.find("not yet implemented") != std::string::npos
                     && msg.find("Phase 2.1") != std::string::npos;
        std::printf("%s pairs N=2 throws Phase-2.1-pending error\n",
                    ok ? "PASS" : "FAIL");
        return ok;
    }
    std::printf("FAIL pairs N=2 did not throw\n");
    return false;
}

} // namespace

int main()
{
    bool all_ok = true;
    for (uint32_t seed : {7u, 31u}) {
        for (uint64_t count : {16ull, 16384ull, 262144ull, 1048576ull}) {
            all_ok = run_pairs(seed, count) && all_ok;
            all_ok = run_keys (seed, count) && all_ok;
        }
    }
    all_ok = run_n_gt_1_throws() && all_ok;
    return all_ok ? 0 : 1;
}
