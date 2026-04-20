// PipelineKernels.cuh — backend-dispatched wrappers for the simple
// orchestration kernels in src/host/GpuPipeline.cu (init, gather,
// permute, merge). All five are pure grid-stride compute — no AES, no
// shared memory, no atomics — so the SYCL ports are mechanical.
//
// Selection at configure time via XCHPLOT2_BACKEND, same shape as
// T1Offsets / T2Offsets / T3Offsets.

#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <sycl/sycl.hpp>

namespace pos2gpu {

// vals[i] = i  for i in [0, count). Used to seed the index stream that
// the subsequent radix sort permutes.
void launch_init_u32_identity(
    uint32_t* d_vals,
    uint64_t count,
    sycl::queue& q);

// dst[p] = src[indices[p]]  for p in [0, count). Two width specialisations.
void launch_gather_u64(
    uint64_t const* d_src,
    uint32_t const* d_indices,
    uint64_t* d_dst,
    uint64_t count,
    sycl::queue& q);

void launch_gather_u32(
    uint32_t const* d_src,
    uint32_t const* d_indices,
    uint32_t* d_dst,
    uint64_t count,
    sycl::queue& q);

// dst_meta[idx]  = src_meta [indices[idx]]
// dst_xbits[idx] = src_xbits[indices[idx]]
// for idx in [0, count). T2's two-stream gather, fused.
void launch_permute_t2(
    uint64_t const* d_src_meta,
    uint32_t const* d_src_xbits,
    uint32_t const* d_indices,
    uint64_t* d_dst_meta,
    uint32_t* d_dst_xbits,
    uint64_t count,
    sycl::queue& q);

// Stable 2-way merge of two sorted (key, value) runs via per-thread
// merge-path binary search. A wins on ties (load-bearing for parity
// with the pool path's CUB radix sort). Only the (uint32, uint32)
// instantiation is currently used — both T1 and T2 streaming-merge
// paths sort uint32 keys (match_info) by uint32 indices.
void launch_merge_pairs_stable_2way_u32_u32(
    uint32_t const* d_A_keys, uint32_t const* d_A_vals, uint64_t nA,
    uint32_t const* d_B_keys, uint32_t const* d_B_vals, uint64_t nB,
    uint32_t* d_out_keys, uint32_t* d_out_vals,
    uint64_t total,
    sycl::queue& q);

} // namespace pos2gpu
