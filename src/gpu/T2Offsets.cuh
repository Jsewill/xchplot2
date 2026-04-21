// T2Offsets.cuh — backend-dispatched wrappers for T2's three kernels.
// Parallel to T1Offsets.cuh; selected at configure time via XCHPLOT2_BACKEND
// (T2OffsetsCuda.cu vs T2OffsetsSycl.cpp).
//
// T2's input stream is SoA (uint64 meta + uint32 match_info) rather than
// T1's AoS XsCandidateGpu, so the bucket/fine-offset wrappers take the
// match_info array directly. The match kernel emits three output streams
// (meta, match_info, x_bits) instead of T1's two.

#pragma once

#include "gpu/AesHashGpu.cuh"

#include <cstdint>

#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>

namespace pos2gpu {

void launch_t2_compute_bucket_offsets(
    uint32_t const* d_sorted_mi,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* d_offsets,
    sycl::queue& q);

void launch_t2_compute_fine_bucket_offsets(
    uint32_t const* d_sorted_mi,
    uint64_t const* d_bucket_offsets,
    int num_match_target_bits,
    int fine_bits,
    uint32_t num_buckets,
    uint64_t* d_fine_offsets,
    sycl::queue& q);

// Fused T2 match. table_id=2, no strength scaling on AES rounds. Emits
// (meta, match_info, x_bits) triples via an atomic cursor; x_bits packs
// the upper-half-k bits of meta_l and meta_r per Table2Constructor.
void launch_t2_match_all_buckets(
    AesHashKeys keys,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    sycl::queue& q);

} // namespace pos2gpu
