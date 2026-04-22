// XsKernelsSycl.cpp — SYCL implementation of Xs gen/pack kernels.
//
// Xs gen uses the sub_group-cooperative bit-sliced AES path
// (AesHashBsSycl.hpp). Each sub_group of 32 lanes computes 32 g_x
// hashes in parallel via bit-logic + native amdgcn ballot
// (__builtin_amdgcn_ballot_w32 behind bs_ballot), with no T-table
// LDS lookups.

#include "gpu/AesHashBsSycl.hpp"
#include "gpu/SyclBackend.hpp"
#include "gpu/XsKernels.cuh"

#include <sycl/sycl.hpp>

namespace pos2gpu {

void launch_xs_gen(
    AesHashKeys keys,
    uint32_t* keys_out,
    uint32_t* vals_out,
    uint64_t total,
    int k,
    uint32_t xor_const,
    sycl::queue& q)
{
    constexpr size_t threads = 256;
    size_t   const groups    = (total + threads - 1) / threads;

    // total = 2^k with k >= 18 is always a multiple of 256, so the
    // global range matches `total` exactly — no bounds check needed.
    // Every sub_group is fully in-range and can participate in bs32
    // cooperatively.

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=, keys_copy = keys](sycl::nd_item<1> it)
            [[sycl::reqd_sub_group_size(32)]]
        {
            auto sg = it.get_sub_group();
            uint64_t idx = it.get_global_id(0);
            uint32_t x   = static_cast<uint32_t>(idx);
            uint32_t mixed = x ^ xor_const;
            keys_out[idx] = pos2gpu::g_x_bs32(sg, keys_copy, mixed, k);
            vals_out[idx] = x;
        }).wait();
}

void launch_xs_pack(
    uint32_t const* keys_in,
    uint32_t const* vals_in,
    XsCandidateGpu* d_out,
    uint64_t total,
    sycl::queue& q)
{
    constexpr size_t threads = 256;
    size_t   const groups    = (total + threads - 1) / threads;

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            uint64_t idx = it.get_global_id(0);
            if (idx >= total) return;
            d_out[idx] = XsCandidateGpu{ keys_in[idx], vals_in[idx] };
        }).wait();
}

} // namespace pos2gpu
