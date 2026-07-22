// sycl_sort_bench — wall-clock A/B for the SYCL radix sort in SortSycl.cpp.
// Times launch_sort_pairs_u32_u32 (u32 keys, the Xs/T1/T2 shape, bits [0,k))
// and launch_sort_keys_u64 (u64 keys, the T3 shape, bits [0,2k)) at a
// caller-chosen count, so the scan variants can be compared on any backend:
//
//   ./sycl_sort_bench [count] [iters]
//   XCHPLOT2_SCAN_SINGLE_WG=1 ./sycl_sort_bench <count>   # interim single-WG scan
//   XCHPLOT2_SCAN_SERIAL=1    ./sycl_sort_bench <count>   # serial control
//   XCHPLOT2_ACPP_SCAN=1      ./sycl_sort_bench <count>   # AdaptiveCpp lookback
//
// The parity tests top out at 2^20 (scan_n=16384); a real k=28 plot sorts at
// count~2^28 (scan_n~4.2M), where the tile-offset scan's parallelism actually
// matters. Reports the min wall time over `iters` timed runs after one warmup.
// Correctness is sycl_sort_parity's job — this only asserts is_sorted as a
// guard against a broken A/B build.

#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;

int bits_for(uint64_t count)
{
    int b = 0;
    while ((1ull << b) < count) ++b;
    return b < 1 ? 1 : b;
}

// Returns the min sort wall time (ms) over `iters` timed runs after one warmup.
double time_pairs(uint64_t count, int iters, int end_bit, bool& sorted_ok)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(12345);
    std::vector<uint32_t> h_keys(count), h_vals(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i] = static_cast<uint32_t>(i);
        h_vals[i] = static_cast<uint32_t>(i);
    }
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    uint32_t* d_ki = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_ko = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vi = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vo = sycl::malloc_device<uint32_t>(count, q);

    size_t sb = 0;
    pos2gpu::launch_sort_pairs_u32_u32(nullptr, sb, nullptr, nullptr, nullptr, nullptr,
                                       count, 0, end_bit, q);
    void* ds = sb ? sycl::malloc_device(sb, q) : nullptr;

    double best = 1e30;
    for (int it = -1; it < iters; ++it) {   // it == -1 is an untimed warmup
        // keys_in is clobbered as scratch by the radix ping-pong, so re-seed.
        q.memcpy(d_ki, h_keys.data(), sizeof(uint32_t) * count);
        q.memcpy(d_vi, h_vals.data(), sizeof(uint32_t) * count).wait();
        auto const t0 = clock_t_::now();
        pos2gpu::launch_sort_pairs_u32_u32(
            ds ? ds : reinterpret_cast<void*>(uintptr_t{1}), sb,
            d_ki, d_ko, d_vi, d_vo, count, 0, end_bit, q);
        q.wait();
        auto const t1 = clock_t_::now();
        if (it >= 0)
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::vector<uint32_t> hk(count);
    q.memcpy(hk.data(), d_ko, sizeof(uint32_t) * count).wait();
    sorted_ok = std::is_sorted(hk.begin(), hk.end());

    if (ds) sycl::free(ds, q);
    sycl::free(d_ki, q); sycl::free(d_ko, q);
    sycl::free(d_vi, q); sycl::free(d_vo, q);
    return best;
}

double time_keys(uint64_t count, int iters, int end_bit, bool& sorted_ok)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(6789);
    std::vector<uint64_t> h_keys(count);
    uint64_t const mask = (end_bit >= 64) ? ~0ull : ((1ull << end_bit) - 1);
    for (uint64_t i = 0; i < count; ++i) h_keys[i] = rng() & mask;

    uint64_t* d_in  = sycl::malloc_device<uint64_t>(count, q);
    uint64_t* d_out = sycl::malloc_device<uint64_t>(count, q);

    size_t sb = 0;
    pos2gpu::launch_sort_keys_u64(nullptr, sb, nullptr, nullptr, count, 0, end_bit, q);
    void* ds = sb ? sycl::malloc_device(sb, q) : nullptr;

    double best = 1e30;
    for (int it = -1; it < iters; ++it) {
        q.memcpy(d_in, h_keys.data(), sizeof(uint64_t) * count).wait();
        auto const t0 = clock_t_::now();
        pos2gpu::launch_sort_keys_u64(
            ds ? ds : reinterpret_cast<void*>(uintptr_t{1}), sb,
            d_in, d_out, count, 0, end_bit, q);
        q.wait();
        auto const t1 = clock_t_::now();
        if (it >= 0)
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::vector<uint64_t> hk(count);
    q.memcpy(hk.data(), d_out, sizeof(uint64_t) * count).wait();
    sorted_ok = std::is_sorted(hk.begin(), hk.end());

    if (ds) sycl::free(ds, q);
    sycl::free(d_in, q); sycl::free(d_out, q);
    return best;
}

char const* scan_variant()
{
    auto on = [](char const* n) { char const* e = std::getenv(n); return e && e[0] == '1'; };
    if (on("XCHPLOT2_SCAN_SERIAL"))    return "serial";
    if (on("XCHPLOT2_SCAN_SINGLE_WG")) return "single-wg";
    if (on("XCHPLOT2_ACPP_SCAN"))      return "acpp-lookback";
    return "parallel";
}

} // namespace

int main(int argc, char** argv)
{
    uint64_t const count = (argc > 1) ? std::strtoull(argv[1], nullptr, 0) : (1ull << 26);
    int const      iters = (argc > 2) ? std::atoi(argv[2]) : 3;
    int const      kbits = bits_for(count);
    int const      u64_end = std::min(2 * kbits, 64);

    auto& q = pos2gpu::sycl_backend::queue();
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("count=%llu (k~%d)  iters=%d  scan=%s\n",
                (unsigned long long)count, kbits, iters, scan_variant());

    bool ps_ok = false, ks_ok = false;
    double const p = time_pairs(count, iters, kbits,   ps_ok);
    double const k = time_keys (count, iters, u64_end, ks_ok);

    std::printf("pairs_u32 [0,%2d): %9.2f ms    keys_u64 [0,%2d): %9.2f ms    sum %9.2f ms\n",
                kbits, p, u64_end, k, p + k);
    if (!ps_ok || !ks_ok)
        std::printf("WARNING: is_sorted failed (pairs=%d keys=%d) — A/B result is not trustworthy\n",
                    ps_ok, ks_ok);
    return (ps_ok && ks_ok) ? 0 : 1;
}
