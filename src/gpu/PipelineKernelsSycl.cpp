// PipelineKernelsSycl.cpp — SYCL implementation of the simple pipeline
// kernels. Mirrors PipelineKernelsCuda.cu; reuses the shared queue from
// SyclBackend.hpp. None of these touch AES so no T-table buffer is
// needed.

#include "gpu/PipelineKernels.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>

namespace pos2gpu {

namespace {

constexpr size_t kThreads = 256;

inline size_t global_for(uint64_t count)
{
    size_t groups = static_cast<size_t>((count + kThreads - 1) / kThreads);
    return groups * kThreads;
}

} // namespace

void launch_init_u32_identity(
    uint32_t* d_vals, uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t idx = it.get_global_id(0);
            if (idx >= count) return;
            d_vals[idx] = uint32_t(idx);
        }).wait();
}

void launch_gather_u64(
    uint64_t const* d_src, uint32_t const* d_indices,
    uint64_t* d_dst, uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t p = it.get_global_id(0);
            if (p >= count) return;
            d_dst[p] = d_src[d_indices[p]];
        }).wait();
}

void launch_gather_u32(
    uint32_t const* d_src, uint32_t const* d_indices,
    uint32_t* d_dst, uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t p = it.get_global_id(0);
            if (p >= count) return;
            d_dst[p] = d_src[d_indices[p]];
        }).wait();
}

void launch_permute_t2(
    uint64_t const* d_src_meta, uint32_t const* d_src_xbits,
    uint32_t const* d_indices,
    uint64_t* d_dst_meta, uint32_t* d_dst_xbits,
    uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t idx = it.get_global_id(0);
            if (idx >= count) return;
            uint32_t i = d_indices[idx];
            d_dst_meta[idx]  = d_src_meta[i];
            d_dst_xbits[idx] = d_src_xbits[i];
        }).wait();
}

// Scatter family — see PipelineKernels.cuh for the design rationale.
// Each kernel walks the source sequentially; the write target is
// d_dst[d_inv_indices[i]], which is random but VRAM-cache-friendly.
// No atomics needed: inv_indices is a permutation, so every dst slot
// is written exactly once.

void launch_compute_inverse_u32(
    uint32_t const* d_indices, uint32_t* d_inv,
    uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t p = it.get_global_id(0);
            if (p >= count) return;
            // indices[p] is the original position of the element that
            // landed at sorted position p; the inverse maps the other
            // way. Cast p down to u32 — inputs to this primitive are
            // bounded by k=32 cap (≪ 2^32).
            d_inv[d_indices[p]] = static_cast<uint32_t>(p);
        }).wait();
}

void launch_scatter_u64(
    uint64_t const* d_src, uint32_t const* d_inv_indices,
    uint64_t* d_dst, uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            d_dst[d_inv_indices[i]] = d_src[i];
        }).wait();
}

void launch_scatter_u32(
    uint32_t const* d_src, uint32_t const* d_inv_indices,
    uint32_t* d_dst, uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            d_dst[d_inv_indices[i]] = d_src[i];
        }).wait();
}

void launch_permute_t2_scatter(
    uint64_t const* d_src_meta, uint32_t const* d_src_xbits,
    uint32_t const* d_inv_indices,
    uint64_t* d_dst_meta, uint32_t* d_dst_xbits,
    uint64_t count, sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(count), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t i = it.get_global_id(0);
            if (i >= count) return;
            uint32_t const dst = d_inv_indices[i];
            d_dst_meta[dst]  = d_src_meta[i];
            d_dst_xbits[dst] = d_src_xbits[i];
        }).wait();
}

void launch_merge_pairs_stable_2way_u32_u32(
    uint32_t const* d_A_keys, uint32_t const* d_A_vals, uint64_t nA,
    uint32_t const* d_B_keys, uint32_t const* d_B_vals, uint64_t nB,
    uint32_t* d_out_keys, uint32_t* d_out_vals, uint64_t total,
    sycl::queue& q)
{
    q.parallel_for(
        sycl::nd_range<1>{ global_for(total), kThreads },
        [=](sycl::nd_item<1> it) {
            uint64_t p = it.get_global_id(0);
            if (p >= total) return;

            uint64_t lo = (p > nB) ? (p - nB) : 0;
            uint64_t hi = (p < nA) ? p : nA;
            while (lo < hi) {
                uint64_t i = lo + (hi - lo + 1) / 2;
                uint64_t j = p - i;
                uint32_t a_prev = d_A_keys[i - 1];
                uint32_t b_here = (j < nB) ? d_B_keys[j] : 0xFFFFFFFFu;
                if (a_prev > b_here) {
                    hi = i - 1;
                } else {
                    lo = i;
                }
            }
            uint64_t i = lo;
            uint64_t j = p - i;

            bool take_a;
            if (i >= nA)      take_a = false;
            else if (j >= nB) take_a = true;
            else              take_a = d_A_keys[i] <= d_B_keys[j];

            if (take_a) {
                d_out_keys[p] = d_A_keys[i];
                d_out_vals[p] = d_A_vals[i];
            } else {
                d_out_keys[p] = d_B_keys[j];
                d_out_vals[p] = d_B_vals[j];
            }
        }).wait();
}

} // namespace pos2gpu
