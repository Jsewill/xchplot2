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

cudaError_t launch_t3_match(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    T2PairingGpu const* d_sorted_t2, // sorted by match_info ascending
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaStream_t stream = nullptr);

} // namespace pos2gpu
