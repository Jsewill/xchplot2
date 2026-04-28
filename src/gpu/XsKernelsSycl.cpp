// XsKernelsSycl.cpp — SYCL implementation of Xs gen/pack kernels.
// Same shape as the T1/T2/T3 SYCL impls; gen reuses the AES T-table USM
// buffer from SyclBackend.hpp, pack is a pure grid-stride lambda.

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
    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    size_t   const groups    = (total + threads - 1) / threads;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sT_local{
            sycl::range<1>{4 * 256}, h};

        h.parallel_for(
            sycl::nd_range<1>{ groups * threads, threads },
            [=, keys_copy = keys](sycl::nd_item<1> it) {
                // Cooperative load of AES T-tables into local memory.
                uint32_t* sT = &sT_local[0];
                size_t local_id = it.get_local_id(0);
                #pragma unroll 1
                for (size_t i = local_id; i < 4 * 256; i += threads) {
                    sT[i] = d_aes_tables[i];
                }
                it.barrier(sycl::access::fence_space::local_space);

                uint64_t idx = it.get_global_id(0);
                if (idx >= total) return;
                uint32_t x = static_cast<uint32_t>(idx);
                uint32_t mixed = x ^ xor_const;
                keys_out[idx] = pos2gpu::g_x_smem(keys_copy, mixed, k, sT);
                vals_out[idx] = x;
            });
    }).wait();
}

void launch_xs_gen_range(
    AesHashKeys keys,
    uint32_t* keys_out,
    uint32_t* vals_out,
    uint64_t pos_begin,
    uint64_t pos_end,
    int k,
    uint32_t xor_const,
    sycl::queue& q)
{
    if (pos_end <= pos_begin) return;
    uint64_t const range_n = pos_end - pos_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    size_t   const groups    = (range_n + threads - 1) / threads;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sT_local{
            sycl::range<1>{4 * 256}, h};

        h.parallel_for(
            sycl::nd_range<1>{ groups * threads, threads },
            [=, keys_copy = keys](sycl::nd_item<1> it) {
                uint32_t* sT = &sT_local[0];
                size_t local_id = it.get_local_id(0);
                #pragma unroll 1
                for (size_t i = local_id; i < 4 * 256; i += threads) {
                    sT[i] = d_aes_tables[i];
                }
                it.barrier(sycl::access::fence_space::local_space);

                uint64_t local_idx = it.get_global_id(0);
                if (local_idx >= range_n) return;
                uint32_t x = static_cast<uint32_t>(pos_begin + local_idx);
                uint32_t mixed = x ^ xor_const;
                keys_out[local_idx] = pos2gpu::g_x_smem(keys_copy, mixed, k, sT);
                vals_out[local_idx] = x;
            });
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

void launch_xs_pack_range(
    uint32_t const* keys_in,
    uint32_t const* vals_in,
    XsCandidateGpu* d_out,
    uint64_t count,
    sycl::queue& q)
{
    // Same body as launch_xs_pack — caller passes already-offset pointers
    // (keys_in, vals_in, d_out) and the slice count.
    if (count == 0) return;
    constexpr size_t threads = 256;
    size_t   const groups    = (count + threads - 1) / threads;

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            uint64_t idx = it.get_global_id(0);
            if (idx >= count) return;
            d_out[idx] = XsCandidateGpu{ keys_in[idx], vals_in[idx] };
        }).wait();
}

} // namespace pos2gpu
