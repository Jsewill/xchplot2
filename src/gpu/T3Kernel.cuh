// T3Kernel.cuh — final-table GPU kernel.
//
// Inputs : sorted T2Pairing[] (sorted by match_info ascending — same way
//          pos2-chip sorts its T2 output via post_construct_span).
// Outputs: T3PairingGpu[] (proof_fragment only) in arbitrary order;
//          caller sorts for parity comparison.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/T2Kernel.cuh"

#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace pos2gpu {

// Mirrors pos2-chip/src/pos/ProofCore.hpp:66 T3Pairing (no RETAIN_X_VALUES_TO_T3 fields).
struct T3PairingGpu {
    uint64_t proof_fragment;
};
static_assert(sizeof(T3PairingGpu) == 8, "must match pos2-chip T3Pairing layout");

struct T3MatchParams {
    int k;
    int strength;
    int num_section_bits;     // = (k < 28) ? 2 : (k - 26)
    int num_match_key_bits;   // = strength
    int num_match_target_bits;// = k - section_bits - match_key_bits
};

T3MatchParams make_t3_params(int k, int strength);

// sorted_t2 input is SoA-split: d_sorted_meta[i] is T2Pairing.meta and
// d_sorted_xbits[i] is T2Pairing.x_bits after the T2 sort. match_info is
// carried in the parallel d_sorted_mi stream.
void launch_t3_match(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_sorted_meta,   // cap entries, uint64 meta
    uint32_t const* d_sorted_xbits,  // cap entries, uint32 x_bits
    uint32_t const* d_sorted_mi,     // parallel match_info stream, may be nullptr
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q);

// Two-step entry point for callers that want to run T3 match in multiple
// bucket-range passes (stage 4d — parallel to the T2 prepare/range split).
// Equivalent to calling launch_t3_match with (0, num_buckets) when the
// range covers the whole bucket space.
//
// launch_t3_match_prepare: computes bucket + fine-bucket offsets into
//   d_temp_storage (reusing T2's wrappers, which T3's input is
//   bit-identical to) and zeroes d_out_count. Same sizing protocol as
//   launch_t3_match (d_temp_storage==nullptr fills *temp_bytes).
//
// launch_t3_match_range: runs the match kernel for bucket range
//   [bucket_begin, bucket_end). Multiple calls sharing d_temp_storage /
//   d_out_pairings / d_out_count produce a concatenated output via
//   atomic append, byte-equivalent to a single full-range call after
//   the subsequent T3 sort.
void launch_t3_match_prepare(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    uint64_t* d_out_count,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q);

void launch_t3_match_range(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q);

// Sliced-meta variant of launch_t3_match_range (minimal tier). Caller
// must ensure that all bucket ids in [bucket_begin, bucket_end) share
// the same section_l so that l reads always fall within section_l's
// row range and r reads always fall within section_r's row range. The
// caller pre-computes the row starts for each section (from the
// d_offsets table sitting in d_temp_storage) and H2Ds the relevant
// section slices of d_sorted_meta into d_meta_l_slice / d_meta_r_slice.
// d_sorted_xbits and d_sorted_mi are still full-cap on device.
void launch_t3_match_section_pair_range(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint64_t section_r_row_start,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q);

} // namespace pos2gpu
