// T3Kernel.cuh — final-table GPU kernel.
//
// Inputs : sorted T2Pairing[] (sorted by match_info ascending — same way
//          pos2-chip sorts its T2 output via post_construct_span).
// Outputs: T3PairingGpu[] (proof_fragment only) in arbitrary order;
//          caller sorts for parity comparison.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/T2Kernel.cuh"

#include <cuda_runtime.h>
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
cudaError_t launch_t3_match(
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
    cudaStream_t stream = nullptr);

// Split entry points — compute bucket / fine offsets once, then run the
// match kernel over one or more disjoint bucket sub-ranges. The compact
// streaming tier uses these to emit T3 in halves into a half-cap device
// staging buffer, D2H each half to pinned host between passes, so the
// cap-sized T3 output never has to be alive on device in full.
//
// Across multiple launch_t3_match_range calls sharing the same
// d_out_pairings + d_out_count, the kernel's atomic counter accumulates
// — output is concatenated. If the caller wants non-overlapping output
// per pass (e.g. separate tile-cap staging reused across passes), it
// must reset d_out_count between passes itself.
cudaError_t launch_t3_match_prepare(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    uint64_t* d_out_count,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaStream_t stream = nullptr);

cudaError_t launch_t3_match_range(
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
    cudaStream_t stream = nullptr);

} // namespace pos2gpu
