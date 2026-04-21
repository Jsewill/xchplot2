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
    sycl::queue& q);

} // namespace pos2gpu
