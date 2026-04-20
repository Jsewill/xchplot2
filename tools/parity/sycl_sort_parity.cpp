// sycl_sort_parity — exercises launch_sort_pairs_u32_u32 and
// launch_sort_keys_u64 on synthetic input and compares against a
// std::sort reference. Built always (independent of XCHPLOT2_BUILD_CUDA),
// so it validates whichever Sort backend is wired into pos2_gpu:
// CUB on the NVIDIA build, oneDPL on the SYCL/AdaptiveCpp build.
//
// Pass criterion: byte-identical sorted streams.

#include "gpu/Sort.cuh"
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

    // Use unique keys (shuffled 0..count-1) so stable and unstable sorts
    // produce byte-identical output — lets us test both CUB (stable) and
    // the hand-rolled SYCL radix (unstable within equal keys) the same way.
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> h_keys(count), h_vals(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i] = static_cast<uint32_t>(i);
        h_vals[i] = static_cast<uint32_t>(i);
    }
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    // Reference: std::sort over indices by key.
    std::vector<uint32_t> ref_keys = h_keys;
    std::vector<uint32_t> ref_vals = h_vals;
    {
        std::vector<uint32_t> idx(count);
        for (uint64_t i = 0; i < count; ++i) idx[i] = static_cast<uint32_t>(i);
        std::sort(idx.begin(), idx.end(),
            [&](uint32_t a, uint32_t b) { return h_keys[a] < h_keys[b]; });
        for (uint64_t i = 0; i < count; ++i) {
            ref_keys[i] = h_keys[idx[i]];
            ref_vals[i] = h_vals[idx[i]];
        }
    }

    uint32_t* d_keys_in  = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_keys_out = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vals_in  = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vals_out = sycl::malloc_device<uint32_t>(count, q);
    q.memcpy(d_keys_in, h_keys.data(), sizeof(uint32_t) * count);
    q.memcpy(d_vals_in, h_vals.data(), sizeof(uint32_t) * count).wait();

    size_t scratch_bytes = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, scratch_bytes,
        nullptr, nullptr, nullptr, nullptr,
        count, 0, 32, q);

    void* d_scratch = scratch_bytes ? sycl::malloc_device(scratch_bytes, q) : nullptr;

    auto const t0 = std::chrono::steady_clock::now();
    pos2gpu::launch_sort_pairs_u32_u32(
        d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),  // any non-null
        scratch_bytes,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        count, 0, 32, q);
    q.wait();
    auto const t1 = std::chrono::steady_clock::now();
    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<uint32_t> h_sorted_keys(count), h_sorted_vals(count);
    q.memcpy(h_sorted_keys.data(), d_keys_out, sizeof(uint32_t) * count);
    q.memcpy(h_sorted_vals.data(), d_vals_out, sizeof(uint32_t) * count).wait();

    if (d_scratch) sycl::free(d_scratch, q);
    sycl::free(d_keys_in,  q);
    sycl::free(d_keys_out, q);
    sycl::free(d_vals_in,  q);
    sycl::free(d_vals_out, q);

    bool const keys_ok = std::memcmp(ref_keys.data(), h_sorted_keys.data(),
                                     sizeof(uint32_t) * count) == 0;
    bool const vals_ok = std::memcmp(ref_vals.data(), h_sorted_vals.data(),
                                     sizeof(uint32_t) * count) == 0;
    bool const sorted = std::is_sorted(h_sorted_keys.begin(),
                                       h_sorted_keys.end());
    bool const ok = keys_ok && vals_ok;
    std::printf("%s  pairs  seed=%u count=%llu  [keys=%d vals=%d sorted=%d  %.2fms]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long long)count,
                keys_ok, vals_ok, sorted, ms);
    if (!ok) {
        uint64_t const show = std::min<uint64_t>(count, 16);
        std::printf("  got [0..%llu): ", (unsigned long long)show);
        for (uint64_t i = 0; i < show; ++i) std::printf("%u ", h_sorted_keys[i]);
        std::printf("\n  ref [0..%llu): ", (unsigned long long)show);
        for (uint64_t i = 0; i < show; ++i) std::printf("%u ", ref_keys[i]);
        std::printf("\n  got [N-%llu..N): ", (unsigned long long)show);
        for (uint64_t i = count - show; i < count; ++i) std::printf("%u ", h_sorted_keys[i]);
        std::printf("\n");
    }
    return ok;
}

bool run_keys(uint32_t seed, uint64_t count)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> h_keys(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i] = rng() & 0x0000FFFFFFFFFFFFull;  // ~48-bit keys
    }

    std::vector<uint64_t> ref = h_keys;
    std::sort(ref.begin(), ref.end());

    uint64_t* d_in  = sycl::malloc_device<uint64_t>(count, q);
    uint64_t* d_out = sycl::malloc_device<uint64_t>(count, q);
    q.memcpy(d_in, h_keys.data(), sizeof(uint64_t) * count).wait();

    size_t scratch_bytes = 0;
    pos2gpu::launch_sort_keys_u64(nullptr, scratch_bytes, nullptr, nullptr,
                                  count, 0, 48, q);
    void* d_scratch = scratch_bytes ? sycl::malloc_device(scratch_bytes, q) : nullptr;
    auto const t0 = std::chrono::steady_clock::now();
    pos2gpu::launch_sort_keys_u64(
        d_scratch ? d_scratch : reinterpret_cast<void*>(uintptr_t{1}),
        scratch_bytes,
        d_in, d_out,
        count, 0, 48, q);
    q.wait();
    auto const t1 = std::chrono::steady_clock::now();
    double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<uint64_t> h_sorted(count);
    q.memcpy(h_sorted.data(), d_out, sizeof(uint64_t) * count).wait();

    if (d_scratch) sycl::free(d_scratch, q);
    sycl::free(d_in, q);
    sycl::free(d_out, q);

    bool const ok = std::memcmp(ref.data(), h_sorted.data(),
                                sizeof(uint64_t) * count) == 0;
    bool const sorted = std::is_sorted(h_sorted.begin(), h_sorted.end());
    std::printf("%s  keys   seed=%u count=%llu  [match=%d sorted=%d  %.2fms]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long long)count,
                ok, sorted, ms);
    return ok;
}

} // namespace

int main()
{
    auto& q = pos2gpu::sycl_backend::queue();
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    bool all_pass = true;
    for (uint32_t seed : { 1u, 7u, 31u }) {
        for (uint64_t n : { 16ull, 1ull << 14, 1ull << 18, 1ull << 20 }) {
            if (!run_pairs(seed, n)) all_pass = false;
            if (!run_keys (seed, n)) all_pass = false;
        }
    }
    return all_pass ? 0 : 1;
}
