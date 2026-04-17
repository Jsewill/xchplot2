// T1Kernel.cuh — GPU port of pos2-chip's Table1Constructor::construct.
//
// Inputs: sorted Xs_Candidate (from launch_construct_xs).
// Outputs: T1PairingGpu array, count returned via host. Output is NOT
// sorted — caller (or t1_parity) sorts before bit-exact comparison.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/XsKernel.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace pos2gpu {

// Mirrors pos2-chip/src/pos/ProofCore.hpp:39 T1Pairing
struct T1PairingGpu {
    uint32_t meta_lo;
    uint32_t meta_hi;
    uint32_t match_info;
};
static_assert(sizeof(T1PairingGpu) == 12, "must match pos2-chip T1Pairing layout");

struct T1MatchParams {
    int k;
    int strength;             // ≥ 2
    int num_section_bits;     // = (k < 28) ? 2 : (k - 26)
    int num_match_key_bits;   // = 2 for T1
    int num_match_target_bits;// = k - section_bits - match_key_bits
};

// Compute the parameters from k and strength, mirroring ProofParams.
T1MatchParams make_t1_params(int k, int strength);

// Run the full T1 phase.
//   d_sorted_xs        : output of launch_construct_xs (sorted by match_info)
//   total              : 1 << k
//   d_out_pairings     : caller-allocated, capacity entries
//   d_out_count        : single uint64_t, will hold actual emitted count
//   capacity           : max number of T1Pairings d_out_pairings can hold
//   d_temp_storage     : nullptr to query *temp_bytes; otherwise must be
//                        at least *temp_bytes large
cudaError_t launch_t1_match(
    uint8_t const* plot_id_bytes,
    T1MatchParams const& params,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t total,
    T1PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaStream_t stream = nullptr);

} // namespace pos2gpu
