// T3Offsets.cuh — backend-dispatched wrapper for T3's match kernel.
//
// T3 reuses T2's bucket / fine-bucket offset wrappers (the input is the
// same uint32_t* sorted_mi stream and the algorithm is identical), so
// only the match kernel — which differs in the Feistel-encrypted output
// — is declared here.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/FeistelCipherGpu.cuh"
#include "gpu/T3Kernel.cuh"  // T3PairingGpu

#include <cstdint>

#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>

namespace pos2gpu {

// Fused T3 match. table_id=3, no strength scaling. For each surviving
// (l, r) pair, emits T3PairingGpu{ proof_fragment = feistel_encrypt(
// (xb_l << k) | xb_r) } via an atomic cursor.
//
// bucket_begin / bucket_end select which bucket-id range to process
// (inclusive / exclusive). Passing (0, num_buckets) preserves the
// original full-pass behavior. Smaller ranges let callers split T3
// match into temporally-separated passes so downstream memory does
// not need to hold the full T3 output at once — parallel to the T2
// match bucket-range plumbing in T2Offsets.cuh.
//
// Across all passes sharing the same d_out_pairings / d_out_count,
// results append via the atomic counter in the kernel.
void launch_t3_match_all_buckets(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
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
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q);

// Sliced variant: same algorithm as launch_t3_match_all_buckets but with
// d_sorted_meta accessed via two per-section slices instead of a full
// cap-sized device buffer. The kernel reads:
//   meta_l = d_meta_l_slice[l - section_l_row_start]
//   meta_r = d_meta_r_slice[r - section_r_row_start]
// Caller MUST ensure that all bucket ids in [bucket_begin, bucket_end)
// share the same section_l (i.e., the range is contained in
// [section_l*num_match_keys, (section_l+1)*num_match_keys)) so that
// every l read falls in section_l's row range and every r read falls in
// the (uniquely-determined) section_r's row range. d_sorted_xbits and
// d_sorted_mi remain full-cap on device (no slicing). Used by minimal
// tier to keep d_t2_meta_sorted parked on host pinned across T3 match;
// drops T3 match peak from ~5200 MB to ~3380 MB at k=28.
void launch_t3_match_section_pair(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint64_t section_r_row_start,
    uint32_t const* d_sorted_xbits,
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
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q);

// Fully-sliced variant (tiny tier). Same kernel logic as
// launch_t3_match_section_pair, but every per-row stream is read via a
// per-section slice instead of a full-cap device pointer:
//   meta_l   = d_meta_l_slice [l - section_l_row_start]
//   xb_l     = d_xbits_l_slice[l - section_l_row_start]
//   target_r = d_mi_r_slice   [r - section_r_row_start] & target_mask
//   meta_r   = d_meta_r_slice [r - section_r_row_start]
//   xb_r     = d_xbits_r_slice[r - section_r_row_start]
// Caller MUST guarantee (same as section_pair) that bucket range
// [bucket_begin, bucket_end) fits inside one section_l so every l read
// stays in section_l's range and every r read stays in section_r's
// range. Used by tiny tier to keep d_t2_xbits_sorted and d_t2_keys_merged
// parked on host pinned across T3 match — drops T3 match peak from
// ~3380 MB to ~1300 MB at k=28.
void launch_t3_match_section_pair_split(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint32_t const* d_xbits_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_xbits_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
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
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q);

} // namespace pos2gpu
