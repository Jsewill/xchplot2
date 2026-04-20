// T2Kernel.cuh — GPU port of pos2-chip's Table2Constructor::construct.
//
// Inputs : sorted T1Pairing[] (from CPU Table1Constructor or future GPU
//          equivalent — sorted by match_info ascending).
// Outputs: T2PairingGpu[] in arbitrary order; caller sorts for parity.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/T1Kernel.cuh"

#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <sycl/sycl.hpp>
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

// The sorted T1 input is now split into two parallel arrays:
//   d_sorted_meta : uint64 meta per entry (was T1PairingGpu.meta_lo|meta_hi).
//   d_sorted_mi   : uint32 match_info per entry — the CUB SortPairs key output,
//                   reused so we don't rematerialise it from the struct.
// Dropping the 4-byte match_info from the permuted stream trims the sorted-T1
// footprint 12 B → 8 B per entry and removes wasted bandwidth on the match
// kernel's hot meta loads.
//
// Output is also SoA: three parallel streams instead of a packed
// T2PairingGpu array. This lets the streaming pipeline free the mi
// stream early (after it's consumed by the subsequent CUB sort as the
// key input) without touching the meta/xbits streams, shaving ~1 GB
// off the k=28 T2-sort peak. The matching-parity tool rebuilds
// T2PairingGpu locally when it needs the AoS form.
void launch_t2_match(
    uint8_t const* plot_id_bytes,
    T2MatchParams const& params,
    uint64_t const* d_sorted_meta,  // meta, sorted by match_info ascending
    uint32_t const* d_sorted_mi,    // parallel match_info stream
    uint64_t t1_count,
    uint64_t* d_out_meta,           // uint64 meta per emitted pair
    uint32_t* d_out_mi,             // uint32 match_info per emitted pair
    uint32_t* d_out_xbits,          // uint32 x_bits per emitted pair
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q);

} // namespace pos2gpu
