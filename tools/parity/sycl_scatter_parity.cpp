// sycl_scatter_parity — Phase 0 of the StreamingPinned/StreamingDisk
// tier plan. Validates that the new scatter family (launch_scatter_u64,
// launch_scatter_u32, launch_permute_t2_scatter) produces byte-identical
// output to the existing gather family (launch_gather_u64,
// launch_gather_u32, launch_permute_t2) when driven by a paired
// (indices, inv_indices) permutation pair.
//
// Sizes cover the k=18 / 20 / 22 caps that the rest of the parity suite
// uses, plus a small fuzz size and a deliberately-large size that
// stresses the inverse-permutation pass at production-target volume.
//
// Pass criterion: for every (seed, count), both the per-element scatter
// outputs AND the fused permute_t2_scatter outputs match the gather
// reference byte-for-byte. No correctness tolerance — these are
// permutation kernels with no FP math.

#include "gpu/PipelineKernels.cuh"
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

// Build a random permutation in [0, count) — this stands in for the
// output of a radix sort's value-stream. The actual values it carries
// in the pipeline are dense, distinct, and have no special structure
// beyond being a permutation, so a shuffled identity is a faithful
// proxy.
std::vector<uint32_t> make_permutation(uint32_t seed, uint64_t count)
{
    std::vector<uint32_t> v(count);
    for (uint64_t i = 0; i < count; ++i) v[i] = static_cast<uint32_t>(i);
    std::mt19937_64 rng(seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}

bool run_case(uint32_t seed, uint64_t count)
{
    auto& q = pos2gpu::sycl_backend::queue();

    // Synthetic data: meta_u64 with bit patterns spanning all 64 bits,
    // xbits_u32 distinct from meta so a swap would surface.
    std::mt19937_64 rng(seed ^ 0xA5A5'5A5A'A5A5'5A5Aull);
    std::vector<uint64_t> h_src_u64(count);
    std::vector<uint32_t> h_src_u32(count);
    for (uint64_t i = 0; i < count; ++i) {
        h_src_u64[i] = rng();
        h_src_u32[i] = static_cast<uint32_t>(rng());
    }

    std::vector<uint32_t> h_indices = make_permutation(seed, count);

    // Device buffers.
    uint64_t* d_src_u64       = sycl::malloc_device<uint64_t>(count, q);
    uint32_t* d_src_u32       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_indices       = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_inv_indices   = sycl::malloc_device<uint32_t>(count, q);
    uint64_t* d_dst_u64_g     = sycl::malloc_device<uint64_t>(count, q);
    uint64_t* d_dst_u64_s     = sycl::malloc_device<uint64_t>(count, q);
    uint32_t* d_dst_u32_g     = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_dst_u32_s     = sycl::malloc_device<uint32_t>(count, q);
    uint64_t* d_dst_meta_g    = sycl::malloc_device<uint64_t>(count, q);
    uint64_t* d_dst_meta_s    = sycl::malloc_device<uint64_t>(count, q);
    uint32_t* d_dst_xbits_g   = sycl::malloc_device<uint32_t>(count, q);
    uint32_t* d_dst_xbits_s   = sycl::malloc_device<uint32_t>(count, q);

    q.memcpy(d_src_u64, h_src_u64.data(), sizeof(uint64_t) * count);
    q.memcpy(d_src_u32, h_src_u32.data(), sizeof(uint32_t) * count);
    q.memcpy(d_indices, h_indices.data(), sizeof(uint32_t) * count).wait();

    // Pre-compute inverse permutation. This is the new helper kernel —
    // its correctness is implicit in the byte-identical scatter result
    // below: a wrong inverse would corrupt every scatter output. The
    // CPU-side check on h_inv post-hoc is overkill; we run it anyway
    // because the cost is trivial and a separate check helps localize
    // failures.
    auto const t_inv0 = std::chrono::steady_clock::now();
    pos2gpu::launch_compute_inverse_u32(d_indices, d_inv_indices, count, q);
    q.wait();
    auto const t_inv1 = std::chrono::steady_clock::now();
    double const inv_ms = std::chrono::duration<double, std::milli>(t_inv1 - t_inv0).count();

    std::vector<uint32_t> h_inv(count);
    q.memcpy(h_inv.data(), d_inv_indices, sizeof(uint32_t) * count).wait();
    bool inv_ok = true;
    for (uint64_t p = 0; p < count; ++p) {
        if (h_inv[h_indices[p]] != static_cast<uint32_t>(p)) { inv_ok = false; break; }
    }

    // Gather reference (existing kernels).
    pos2gpu::launch_gather_u64(d_src_u64, d_indices, d_dst_u64_g, count, q);
    pos2gpu::launch_gather_u32(d_src_u32, d_indices, d_dst_u32_g, count, q);
    pos2gpu::launch_permute_t2(d_src_u64, d_src_u32, d_indices,
                               d_dst_meta_g, d_dst_xbits_g, count, q);
    q.wait();

    // Scatter under test.
    auto const t_s0 = std::chrono::steady_clock::now();
    pos2gpu::launch_scatter_u64(d_src_u64, d_inv_indices, d_dst_u64_s, count, q);
    pos2gpu::launch_scatter_u32(d_src_u32, d_inv_indices, d_dst_u32_s, count, q);
    pos2gpu::launch_permute_t2_scatter(d_src_u64, d_src_u32, d_inv_indices,
                                       d_dst_meta_s, d_dst_xbits_s, count, q);
    q.wait();
    auto const t_s1 = std::chrono::steady_clock::now();
    double const scatter_ms = std::chrono::duration<double, std::milli>(t_s1 - t_s0).count();

    // Pull both result sets back for byte-compare.
    std::vector<uint64_t> h_dst_u64_g(count), h_dst_u64_s(count);
    std::vector<uint32_t> h_dst_u32_g(count), h_dst_u32_s(count);
    std::vector<uint64_t> h_dst_meta_g(count), h_dst_meta_s(count);
    std::vector<uint32_t> h_dst_xbits_g(count), h_dst_xbits_s(count);
    q.memcpy(h_dst_u64_g.data(),   d_dst_u64_g,   sizeof(uint64_t) * count);
    q.memcpy(h_dst_u64_s.data(),   d_dst_u64_s,   sizeof(uint64_t) * count);
    q.memcpy(h_dst_u32_g.data(),   d_dst_u32_g,   sizeof(uint32_t) * count);
    q.memcpy(h_dst_u32_s.data(),   d_dst_u32_s,   sizeof(uint32_t) * count);
    q.memcpy(h_dst_meta_g.data(),  d_dst_meta_g,  sizeof(uint64_t) * count);
    q.memcpy(h_dst_meta_s.data(),  d_dst_meta_s,  sizeof(uint64_t) * count);
    q.memcpy(h_dst_xbits_g.data(), d_dst_xbits_g, sizeof(uint32_t) * count);
    q.memcpy(h_dst_xbits_s.data(), d_dst_xbits_s, sizeof(uint32_t) * count).wait();

    sycl::free(d_src_u64, q);     sycl::free(d_src_u32, q);
    sycl::free(d_indices, q);     sycl::free(d_inv_indices, q);
    sycl::free(d_dst_u64_g, q);   sycl::free(d_dst_u64_s, q);
    sycl::free(d_dst_u32_g, q);   sycl::free(d_dst_u32_s, q);
    sycl::free(d_dst_meta_g, q);  sycl::free(d_dst_meta_s, q);
    sycl::free(d_dst_xbits_g, q); sycl::free(d_dst_xbits_s, q);

    bool const u64_ok = std::memcmp(h_dst_u64_g.data(), h_dst_u64_s.data(),
                                    sizeof(uint64_t) * count) == 0;
    bool const u32_ok = std::memcmp(h_dst_u32_g.data(), h_dst_u32_s.data(),
                                    sizeof(uint32_t) * count) == 0;
    bool const meta_ok = std::memcmp(h_dst_meta_g.data(), h_dst_meta_s.data(),
                                     sizeof(uint64_t) * count) == 0;
    bool const xbits_ok = std::memcmp(h_dst_xbits_g.data(), h_dst_xbits_s.data(),
                                      sizeof(uint32_t) * count) == 0;

    bool const ok = inv_ok && u64_ok && u32_ok && meta_ok && xbits_ok;
    std::printf("%s  seed=%u count=%llu  [inv=%d u64=%d u32=%d meta=%d xbits=%d  inv%.2fms scatter%.2fms]\n",
                ok ? "PASS" : "FAIL", seed, (unsigned long long)count,
                inv_ok, u64_ok, u32_ok, meta_ok, xbits_ok,
                inv_ms, scatter_ms);

    if (!ok) {
        uint64_t const show = std::min<uint64_t>(count, 8);
        std::printf("  gather  u64[0..%llu): ", (unsigned long long)show);
        for (uint64_t i = 0; i < show; ++i) std::printf("%016llx ",
                                                (unsigned long long)h_dst_u64_g[i]);
        std::printf("\n  scatter u64[0..%llu): ", (unsigned long long)show);
        for (uint64_t i = 0; i < show; ++i) std::printf("%016llx ",
                                                (unsigned long long)h_dst_u64_s[i]);
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

    // Sizes:
    //   - 16: smoke (manually-inspectable shape)
    //   - 1<<18, 1<<20, 1<<22: k=18 / 20 / 22 cap proxies — matches the
    //     volume range exercised by sycl_phase_split_parity, so we
    //     overlap the validation surface with the existing harness.
    //   - 1<<25 (~33M): exercises the inverse-permutation pass at
    //     production-target volume. k=28 cap is ~250M but anchoring at
    //     1<<25 keeps the whole test under ~2 GB of VRAM, which fits the
    //     "build-and-run on every dev machine" expectation. The kernel
    //     is grid-stride so behaviour at 1<<25 generalizes to 1<<28.
    bool all_pass = true;
    for (uint32_t seed : { 1u, 7u, 31u }) {
        for (uint64_t n : { uint64_t{16},
                            uint64_t{1} << 18,
                            uint64_t{1} << 20,
                            uint64_t{1} << 22,
                            uint64_t{1} << 25 }) {
            all_pass &= run_case(seed, n);
        }
    }

    std::printf("\n==> %s\n", all_pass ? "ALL OK" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
