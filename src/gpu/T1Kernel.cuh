// T1Kernel.cuh — GPU port of pos2-chip's Table1Constructor::construct.
//
// Inputs: sorted Xs_Candidate (from launch_construct_xs).
// Outputs: T1PairingGpu array, count returned via host. Output is NOT
// sorted — caller (or t1_parity) sorts before bit-exact comparison.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/XsKernel.cuh"

#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>
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
//   d_out_meta         : caller-allocated, capacity entries (uint64 meta).
//   d_out_mi           : caller-allocated, capacity entries (uint32 match_info).
//   d_out_count        : single uint64_t, will hold actual emitted count
//   capacity           : max number of T1Pairings the output arrays can hold
//   d_temp_storage     : nullptr to query *temp_bytes; otherwise must be
//                        at least *temp_bytes large
//
// Output is SoA (two parallel streams) rather than an AoS T1PairingGpu
// array so the streaming pipeline can feed d_out_mi straight into CUB
// as the sort-key input and free it as soon as CUB consumes it, without
// touching the meta stream. Saves ~1 GB at k=28 during the T1 sort
// phase. t1_parity and other consumers rebuild the AoS form locally if
// they need it.
void launch_t1_match(
    uint8_t const* plot_id_bytes,
    T1MatchParams const& params,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t total,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q);

} // namespace pos2gpu
