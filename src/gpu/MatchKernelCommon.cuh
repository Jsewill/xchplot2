// MatchKernelCommon.cuh — device helpers shared verbatim across the
// T1/T2/T3 match kernels and their section-pair / two-phase variants.
//
// These two snippets — the section_r rotation and the fine-bucket
// lower-bound search — were copy-pasted into every match kernel. They
// are pure, self-contained, and exactly the kind of fiddly logic where
// a silent copy-paste divergence would be a correctness bug with no
// obvious symptom. Extracting them keeps each kernel readable
// top-to-bottom (it just calls these instead of re-inlining the
// fiddly bits) while the dangerous logic lives in one parity-covered
// place. Deliberately NOT a policy-template: the per-table differences
// (AoS/SoA input, output shape, Feistel) stay as per-table kernel code
// so each kernel is still independently verifiable.
//
// Fully portable — POS2_DEVICE_INLINE compiles under both nvcc and
// acpp/clang, same as the AesHashGpu.cuh _smem family.

#pragma once

#include "gpu/PortableAttrs.hpp"

#include <cstdint>

namespace pos2gpu {

// The r-side section paired with section_l: rotate-left-by-1 within
// num_section_bits, +1, rotate back. Identical in every match kernel
// (T1/T2/T3, all_buckets + section_pair + two-phase variants).
POS2_DEVICE_INLINE uint32_t matching_section_r(
    uint32_t section_l, int num_section_bits)
{
    uint32_t mask = (1u << num_section_bits) - 1u;
    uint32_t rl   = ((section_l << 1) | (section_l >> (num_section_bits - 1))) & mask;
    uint32_t rl1  = (rl + 1) & mask;
    return ((rl1 >> 1) | (rl1 << (num_section_bits - 1))) & mask;
}

// Lower-bound search over a sorted match_info (SoA u32 stream) for the
// first index in [lo, hi) whose masked target is >= target_l. Used by
// the T2/T3 match kernels (all_buckets, section_pair, two-phase).
//
// The sliced section_pair_split variants read d_mi_r_slice[mid -
// section_r_row_start]; they call this with the base pointer adjusted
// by -section_r_row_start so the [mid] indexing still lands correctly.
// T1's match_info lives in an AoS struct (XsCandidateGpu.match_info)
// and keeps its own inline search — genuinely different access, single
// copy, not a copy-paste hazard.
POS2_DEVICE_INLINE uint64_t fine_bucket_lower_bound(
    uint32_t const* __restrict__ d_mi,
    uint64_t lo, uint64_t hi,
    uint32_t target_l, uint32_t target_mask)
{
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t target_mid = d_mi[mid] & target_mask;
        if (target_mid < target_l) lo = mid + 1;
        else                       hi = mid;
    }
    return lo;
}

} // namespace pos2gpu
