// sycl_sort_u32_u64_parity — Phase 1.3a of the StreamingPinned/Disk
// plan. Validates launch_sort_pairs_u32_u64 against a reference
// built by running launch_sort_pairs_u32_u32 with identity values
// then gather_u64-ing the u64 payload through the resulting
// permutation. Both paths produce the same final layout for unique
// keys; equality of the two confirms the new primitive's correctness
// across both the CUB-backed and the SYCL-fallback paths.

#include "gpu/PipelineKernels.cuh"
#include "gpu/Sort.cuh"
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

bool run_case(uint32_t seed, uint64_t count, int begin_bit, int end_bit)
{
    auto& q = pos2gpu::sycl_backend::queue();

    std::mt19937_64 rng(seed);
    std::vector<uint32_t> h_keys(count);
    std::vector<uint64_t> h_vals(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_keys[i] = static_cast<uint32_t>(i);
        h_vals[i] = rng();
    }
    std::shuffle(h_keys.begin(), h_keys.end(), rng);

    // Device buffers: separate copies for the reference and the
    // under-test path.
    uint32_t* d_keys_a       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_keys_b       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_keys_out_ref = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_keys_out_uut = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_idx_in       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_idx_out      = sycl::malloc_device<uint32_t>(count, q);
    uint64_t* d_vals_a       = sycl::malloc_device<uint64_t>(count, q);
    uint64_t* d_vals_b       = sycl::malloc_device<uint64_t>(count, q);
    uint64_t* d_vals_out_ref = sycl::malloc_device<uint64_t>(count, q);
    uint64_t* d_vals_out_uut = sycl::malloc_device<uint64_t>(count, q);
    q.memcpy(d_keys_a, h_keys.data(), sizeof(uint32_t) * count);
    q.memcpy(d_keys_b, h_keys.data(), sizeof(uint32_t) * count);
    q.memcpy(d_vals_a, h_vals.data(), sizeof(uint64_t) * count);
    q.memcpy(d_vals_b, h_vals.data(), sizeof(uint64_t) * count).wait();

    // Reference: u32_u32 sort with identity vals + gather_u64.
    pos2gpu::launch_init_u32_identity(d_idx_in, count, q);
    size_t ref_scratch = 0;
    pos2gpu::launch_sort_pairs_u32_u32(
        nullptr, ref_scratch,
        d_keys_a, d_keys_out_ref,
        d_idx_in, d_idx_out,
        count, begin_bit, end_bit, q);
    void* d_ref_scratch = ref_scratch ? sycl::malloc_device(ref_scratch, q) : nullptr;

    auto const t_ref0 = std::chrono::steady_clock::now();
    pos2gpu::launch_sort_pairs_u32_u32(
        d_ref_scratch ? d_ref_scratch : reinterpret_cast<void*>(uintptr_t{1}),
        ref_scratch,
        d_keys_a, d_keys_out_ref,
        d_idx_in, d_idx_out,
        count, begin_bit, end_bit, q);
    pos2gpu::launch_gather_u64(d_vals_a, d_idx_out, d_vals_out_ref, count, q);
    q.wait();
    auto const t_ref1 = std::chrono::steady_clock::now();
    double const ref_ms = std::chrono::duration<double, std::milli>(t_ref1 - t_ref0).count();

    // Under test: launch_sort_pairs_u32_u64 direct.
    size_t uut_scratch = 0;
    pos2gpu::launch_sort_pairs_u32_u64(
        nullptr, uut_scratch,
        d_keys_b, d_keys_out_uut,
        d_vals_b, d_vals_out_uut,
        count, begin_bit, end_bit, q);
    void* d_uut_scratch = uut_scratch ? sycl::malloc_device(uut_scratch, q) : nullptr;

    auto const t_uut0 = std::chrono::steady_clock::now();
    pos2gpu::launch_sort_pairs_u32_u64(
        d_uut_scratch ? d_uut_scratch : reinterpret_cast<void*>(uintptr_t{1}),
        uut_scratch,
        d_keys_b, d_keys_out_uut,
        d_vals_b, d_vals_out_uut,
        count, begin_bit, end_bit, q);
    q.wait();
    auto const t_uut1 = std::chrono::steady_clock::now();
    double const uut_ms = std::chrono::duration<double, std::milli>(t_uut1 - t_uut0).count();

    std::vector<uint32_t> h_ref_keys(count), h_uut_keys(count);
    std::vector<uint64_t> h_ref_vals(count), h_uut_vals(count);
    q.memcpy(h_ref_keys.data(), d_keys_out_ref, sizeof(uint32_t) * count);
    q.memcpy(h_uut_keys.data(), d_keys_out_uut, sizeof(uint32_t) * count);
    q.memcpy(h_ref_vals.data(), d_vals_out_ref, sizeof(uint64_t) * count);
    q.memcpy(h_uut_vals.data(), d_vals_out_uut, sizeof(uint64_t) * count).wait();

    if (d_ref_scratch) sycl::free(d_ref_scratch, q);
    if (d_uut_scratch) sycl::free(d_uut_scratch, q);
    sycl::free(d_keys_a, q);       sycl::free(d_keys_b, q);
    sycl::free(d_keys_out_ref, q); sycl::free(d_keys_out_uut, q);
    sycl::free(d_idx_in, q);       sycl::free(d_idx_out, q);
    sycl::free(d_vals_a, q);       sycl::free(d_vals_b, q);
    sycl::free(d_vals_out_ref, q); sycl::free(d_vals_out_uut, q);

    bool const keys_ok = std::memcmp(h_ref_keys.data(), h_uut_keys.data(),
                                     sizeof(uint32_t) * count) == 0;
    bool const vals_ok = std::memcmp(h_ref_vals.data(), h_uut_vals.data(),
                                     sizeof(uint64_t) * count) == 0;
    bool const ok = keys_ok && vals_ok;
    std::printf("%s  seed=%u n=%llu  [keys=%d vals=%d  ref%.2fms uut%.2fms]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long long)count,
                keys_ok, vals_ok, ref_ms, uut_ms);
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
        for (uint64_t n : { uint64_t{16},
                            uint64_t{1} << 14,
                            uint64_t{1} << 18,
                            uint64_t{1} << 20,
                            uint64_t{1} << 22 }) {
            all_pass &= run_case(seed, n, /*begin_bit=*/0, /*end_bit=*/22);
        }
    }

    std::printf("\n==> %s\n", all_pass ? "ALL OK" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
