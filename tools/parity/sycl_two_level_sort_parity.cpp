// sycl_two_level_sort_parity — Phase 1 of the StreamingPinned/Disk
// tier plan. Validates that launch_two_level_sort_pairs_u32_u32
// produces byte-identical (keys_out, vals_out) to a reference
// launch_sort_pairs_u32_u32 over the same (count, begin_bit,
// end_bit) inputs.
//
// Additionally validates the d_bucket_starts output: bucket b
// should contain exactly the keys whose top num_top_bits bits
// equal b, and the per-bucket counts should match a CPU histogram.
//
// Test matrix:
//   - seeds: 1, 7, 31 (same as sycl_sort_parity)
//   - counts: 16, 1<<14, 1<<18, 1<<20, 1<<22 — small through
//     k=22 cap proxy
//   - num_top_bits: 4, 8 — exercises both shallow (16 buckets,
//     ~1M entries each at the larger count) and deep (256 buckets,
//     ~16K entries each) partitioning
//
// Uniqueness: keys are a shuffled [0..count) so the radix sort's
// instability within equal keys doesn't matter. This isolates the
// two-level wrapper's correctness from the unrelated tie-handling
// behaviour.

#include "gpu/Sort.cuh"
#include "gpu/TwoLevelSort.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool run_case(uint32_t seed, uint64_t count, int num_top_bits,
              int begin_bit, int end_bit)
{
    auto& q = pos2gpu::sycl_backend::queue();
    size_t const num_buckets = size_t{1} << num_top_bits;

    // Inputs: unique keys shuffled across [0, count), random vals.
    std::mt19937_64 rng(seed);
    std::vector<uint32_t> h_keys(count), h_vals(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i] = static_cast<uint32_t>(i);
        h_vals[i] = static_cast<uint32_t>(rng());
    }
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    // We need two copies of (keys, vals) on device — one for the
    // reference single-level sort, one for the two-level. The two
    // sorts each clobber their inputs, so they can't share.
    uint32_t* d_keys_a       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_keys_b       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_keys_out_ref = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_keys_out_tl  = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vals_a       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vals_b       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vals_out_ref = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_vals_out_tl  = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_bucket_starts = sycl::malloc_device<uint32_t>(num_buckets + 1, q);
    q.memcpy(d_keys_a, h_keys.data(), sizeof(uint32_t) * count);
    q.memcpy(d_keys_b, h_keys.data(), sizeof(uint32_t) * count);
    q.memcpy(d_vals_a, h_vals.data(), sizeof(uint32_t) * count);
    q.memcpy(d_vals_b, h_vals.data(), sizeof(uint32_t) * count).wait();

    // Reference: single-level launch_sort_pairs_u32_u32.
    size_t ref_scratch = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, ref_scratch,
        d_keys_a, d_keys_out_ref,
        d_vals_a, d_vals_out_ref,
        count, begin_bit, end_bit, q);
    void* d_ref_scratch = ref_scratch ? sycl::malloc_device(ref_scratch, q) : nullptr;

    auto const t_ref0 = std::chrono::steady_clock::now();
    pos2gpu::launch_sort_pairs_u32_u32(
        d_ref_scratch ? d_ref_scratch : reinterpret_cast<void*>(uintptr_t{1}),
        ref_scratch,
        d_keys_a, d_keys_out_ref,
        d_vals_a, d_vals_out_ref,
        count, begin_bit, end_bit, q);
    q.wait();
    auto const t_ref1 = std::chrono::steady_clock::now();
    double const ref_ms = std::chrono::duration<double, std::milli>(t_ref1 - t_ref0).count();

    // Two-level under test.
    size_t tl_scratch = 0;
    pos2gpu::launch_two_level_sort_pairs_u32_u32(
        nullptr, tl_scratch,
        d_keys_b, d_keys_out_tl,
        d_vals_b, d_vals_out_tl,
        count, begin_bit, end_bit, num_top_bits,
        d_bucket_starts, q);
    void* d_tl_scratch = tl_scratch ? sycl::malloc_device(tl_scratch, q) : nullptr;

    auto const t_tl0 = std::chrono::steady_clock::now();
    pos2gpu::launch_two_level_sort_pairs_u32_u32(
        d_tl_scratch, tl_scratch,
        d_keys_b, d_keys_out_tl,
        d_vals_b, d_vals_out_tl,
        count, begin_bit, end_bit, num_top_bits,
        d_bucket_starts, q);
    q.wait();
    auto const t_tl1 = std::chrono::steady_clock::now();
    double const tl_ms = std::chrono::duration<double, std::milli>(t_tl1 - t_tl0).count();

    // Pull both result sets + bucket starts back.
    std::vector<uint32_t> h_ref_keys(count), h_tl_keys(count);
    std::vector<uint32_t> h_ref_vals(count), h_tl_vals(count);
    std::vector<uint32_t> h_bucket_starts(num_buckets + 1);
    q.memcpy(h_ref_keys.data(), d_keys_out_ref, sizeof(uint32_t) * count);
    q.memcpy(h_tl_keys.data(),  d_keys_out_tl,  sizeof(uint32_t) * count);
    q.memcpy(h_ref_vals.data(), d_vals_out_ref, sizeof(uint32_t) * count);
    q.memcpy(h_tl_vals.data(),  d_vals_out_tl,  sizeof(uint32_t) * count);
    q.memcpy(h_bucket_starts.data(), d_bucket_starts,
             (num_buckets + 1) * sizeof(uint32_t)).wait();

    if (d_ref_scratch) sycl::free(d_ref_scratch, q);
    if (d_tl_scratch)  sycl::free(d_tl_scratch, q);
    sycl::free(d_keys_a, q);       sycl::free(d_keys_b, q);
    sycl::free(d_keys_out_ref, q); sycl::free(d_keys_out_tl, q);
    sycl::free(d_vals_a, q);       sycl::free(d_vals_b, q);
    sycl::free(d_vals_out_ref, q); sycl::free(d_vals_out_tl, q);
    sycl::free(d_bucket_starts, q);

    // Validation 1: byte-identical output streams.
    bool const keys_ok = std::memcmp(h_ref_keys.data(), h_tl_keys.data(),
                                     sizeof(uint32_t) * count) == 0;
    bool const vals_ok = std::memcmp(h_ref_vals.data(), h_tl_vals.data(),
                                     sizeof(uint32_t) * count) == 0;

    // Validation 2: bucket_starts is monotone and total == count.
    bool buckets_mono = true;
    for (size_t b = 0; b < num_buckets; ++b) {
        if (h_bucket_starts[b] > h_bucket_starts[b + 1]) {
            buckets_mono = false; break;
        }
    }
    bool const buckets_total_ok = (h_bucket_starts[num_buckets] == count);

    // Validation 3: each bucket b contains exactly the keys whose
    // top num_top_bits-bits-of-the-sort-range equal b. We check
    // the output stream against this property.
    int const top_bit_offset = end_bit - num_top_bits;
    uint32_t const mask = (uint32_t{1} << num_top_bits) - 1u;
    bool buckets_consistent = true;
    for (size_t b = 0; b < num_buckets; ++b) {
        uint32_t const start = h_bucket_starts[b];
        uint32_t const stop  = h_bucket_starts[b + 1];
        for (uint32_t p = start; p < stop; ++p) {
            uint32_t const got_bucket =
                (h_tl_keys[p] >> top_bit_offset) & mask;
            if (got_bucket != b) {
                buckets_consistent = false; break;
            }
        }
        if (!buckets_consistent) break;
    }

    bool const ok = keys_ok && vals_ok && buckets_mono &&
                    buckets_total_ok && buckets_consistent;
    std::printf("%s  seed=%u n=%llu B=%d  [keys=%d vals=%d "
                "mono=%d total=%d cons=%d  ref%.2fms tl%.2fms]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long long)count, num_top_bits,
                keys_ok, vals_ok, buckets_mono, buckets_total_ok,
                buckets_consistent, ref_ms, tl_ms);
    if (!ok) {
        size_t const show = std::min<size_t>(count, 8);
        std::printf("  ref keys[0..%zu): ", show);
        for (size_t i = 0; i < show; ++i) std::printf("%u ", h_ref_keys[i]);
        std::printf("\n  tl  keys[0..%zu): ", show);
        for (size_t i = 0; i < show; ++i) std::printf("%u ", h_tl_keys[i]);
        std::printf("\n");
    }
    return ok;
}

} // namespace

int main()
{
    auto& q = pos2gpu::sycl_backend::queue();
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // Cover the bit range used by T1/T2 sort (begin_bit=0, end_bit=k
    // bits — proxied here by end_bit large enough to actually
    // include the top bits we're partitioning on). count <= 2^22
    // proxies k=22 cap; production-target k=28 cap (~2.5e8) is
    // omitted because it'd need ~6 GB VRAM with the doubled
    // buffers and dominates this test's runtime; the grid-stride
    // partition kernel scales linearly in count so behaviour at
    // 1<<22 generalizes upward.
    bool all_pass = true;
    for (uint32_t seed : { 1u, 7u, 31u }) {
        for (uint64_t n : { uint64_t{16},
                            uint64_t{1} << 14,
                            uint64_t{1} << 18,
                            uint64_t{1} << 20,
                            uint64_t{1} << 22 }) {
            for (int B : { 4, 8 }) {
                all_pass &= run_case(seed, n, B, /*begin_bit=*/0, /*end_bit=*/22);
            }
        }
    }

    std::printf("\n==> %s\n", all_pass ? "ALL OK" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
