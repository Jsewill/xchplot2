// T2Kernel.cuh — GPU port of pos2-chip's Table2Constructor::construct.
//
// Inputs : sorted T1Pairing[] (from CPU Table1Constructor or future GPU
//          equivalent — sorted by match_info ascending).
// Outputs: T2PairingGpu[] in arbitrary order; caller sorts for parity.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/T1Kernel.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace pos2gpu {

// Mirrors pos2-chip/src/pos/ProofCore.hpp:57 T2Pairing (no RETAIN_X_VALUES_TO_T3 fields).
struct T2PairingGpu {
    uint64_t meta;       // pair.meta_result (2k-bit value packed into 64 bits)
    uint32_t match_info; // pair.r[0] & mask(k)
    uint32_t x_bits;     // combination of upper-k of meta_l / meta_r per Table2Constructor::handle_pair_into
};
static_assert(sizeof(T2PairingGpu) == 16, "must match pos2-chip T2Pairing layout");

// T2 uses the same (k, num_section_bits, num_match_key_bits=strength,
// num_match_target_bits) as T1 for table_id=2. We reuse T1MatchParams
// for now since the only difference would be num_match_key_bits, which
// for T1 is hard-coded to 2 and for T2 equals strength. Pass the correct
// values from the host.
struct T2MatchParams {
    int k;
    int strength;             // ≥ 2
    int num_section_bits;     // = (k < 28) ? 2 : (k - 26)
    int num_match_key_bits;   // = strength for T2
    int num_match_target_bits;// = k - section_bits - match_key_bits
};

T2MatchParams make_t2_params(int k, int strength);

cudaError_t launch_t2_match(
    uint8_t const* plot_id_bytes,
    T2MatchParams const& params,
    T1PairingGpu const* d_sorted_t1, // sorted by match_info ascending
    uint64_t t1_count,
    T2PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaStream_t stream = nullptr);

} // namespace pos2gpu
