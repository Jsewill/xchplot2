// T1Offsets.cuh — backend-dispatched wrapper for compute_bucket_offsets.
//
// One-thread-per-bucket binary search that emits offsets[num_buckets+1]
// for T1's sorted XsCandidateGpu stream. Two implementations live in
// sibling TUs and are selected at configure time:
//
//   XCHPLOT2_BACKEND=cuda  →  T1OffsetsCuda.cu  (default; existing __global__)
//   XCHPLOT2_BACKEND=sycl  →  T1OffsetsSycl.cpp (AdaptiveCpp parallel_for)
//
// The CUDA stream parameter is honoured by both: the CUDA path launches
// directly on it; the SYCL path syncs the stream before its own launch
// and waits for the SYCL queue to complete before returning, so the
// caller can chain subsequent CUDA work on `stream` unchanged.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/XsCandidateGpu.hpp"

#include <cstdint>

// Forward-declare cudaStream_t instead of including <cuda_runtime.h>, so the
// SYCL backend implementation (compiled by acpp/clang in non-CUDA mode) can
// include this header without dragging in nvcc-only intrinsics from the
// transitive AesGpu.cuh chain. CUDA-side TUs include <cuda_runtime.h>
// themselves; the typedef redeclaration to the same type is permitted.
#include <cuda_fp16.h>
#include <sycl/sycl.hpp>

namespace pos2gpu {

void launch_compute_bucket_offsets(
    XsCandidateGpu const* d_sorted,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* d_offsets,
    sycl::queue& q);

// Per-fine-key offsets: for each (r_bucket, fine_key) in
// [0, num_buckets) × [0, 2^fine_bits), find the lowest index i in
// `sorted[bucket_offsets[r_bucket] .. bucket_offsets[r_bucket+1])` such
// that ((sorted[i].match_info & target_mask) >> shift) >= fine_key, where
// target_mask = (1<<num_match_target_bits)-1 and shift = num_match_target_bits
// - fine_bits. Sentinel: fine_offsets[total] = bucket_offsets[num_buckets].
void launch_compute_fine_bucket_offsets(
    XsCandidateGpu const* d_sorted,
    uint64_t const* d_bucket_offsets,
    int num_match_target_bits,
    int fine_bits,
    uint32_t num_buckets,
    uint64_t* d_fine_offsets,
    sycl::queue& q);

// Fused T1 match: for each (section_l, match_key_r) bucket, walk the L
// candidates against the matching R bucket with AES-derived target_l, and
// emit T1Pairings into out_meta[] / out_mi[] via an atomic cursor.
//
// Grid arrangement (CUDA): grid.y = num_buckets, grid.x slices L; the SYCL
// path uses an analogous 2D nd_range. l_count_max is the per-section L
// upper bound used to size grid.x without a host fence on the actual L
// count — excess threads early-exit on `l >= l_end`.
void launch_t1_match_all_buckets(
    AesHashKeys keys,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    int extra_rounds_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    sycl::queue& q);

} // namespace pos2gpu
