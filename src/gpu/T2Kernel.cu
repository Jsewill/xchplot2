// T2Kernel.cu — sort T1 pairings by (section, match_key, target) using
// CUB DeviceRadixSort, then run a sliding-window match kernel to emit
// T2Pairings (4-x sequences).
//
// CPU reference: pos2-chip/src/plot/TableConstructorGeneric.hpp
//   Table2Constructor::construct(t1_pairs, out, post_sort_tmp).
//
// Status: skeleton. The CUB sort call site is wired but the matching
// predicate kernel is a TODO until t1_parity passes (no point matching
// what we can't yet produce identically).

#include "gpu/AesHashGpu.cuh"
#include "gpu/T2Kernel.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace pos2gpu {

cudaError_t launch_t2(
    AesHashKeys const& /*keys*/,
    void const* /*d_t1_pairings*/,
    uint64_t /*t1_count*/,
    void* /*d_t2_pairings*/,
    uint64_t* /*d_t2_count*/,
    void* /*d_temp_storage*/,
    size_t /*temp_storage_bytes*/,
    cudaStream_t /*stream*/)
{
    // TODO:
    //   1. Pack each T1Pairing into a (key, value) pair where the key is
    //      (section << match_key_bits | match_key) and the value is the
    //      pairing's array index.
    //   2. cub::DeviceRadixSort::SortPairs on (key, value).
    //   3. Launch a per-thread match kernel: each thread inspects a window
    //      of N adjacent sorted entries and tests pairing(meta_l, meta_r)
    //      against the matching predicate (see AesHash::matching_target
    //      with table_id=2 in TableConstructorGeneric.hpp).
    //   4. cub::DeviceSelect::Flagged to compact matches into d_t2_pairings.
    //
    // For sizing d_temp_storage, the host should:
    //   size_t bytes = 0;
    //   cub::DeviceRadixSort::SortPairs(nullptr, bytes, ...);
    //   cudaMalloc(&d_temp_storage, bytes);
    return cudaErrorNotSupported;
}

// Helper to query the temp-storage requirement for the radix sort step.
// Once implemented, callers do:
//   size_t bytes = 0;
//   query_t2_temp_bytes(t1_count, &bytes);
//   cudaMalloc(&temp, bytes);
cudaError_t query_t2_temp_bytes(uint64_t /*t1_count*/, size_t* out_bytes)
{
    // Conservative initial estimate: 4 × t1_count × sizeof(uint64_t) for
    // (key, value) plus CUB scratch. Real value comes from CUB's
    // SortPairs(nullptr, bytes, ...) once the implementation is filled in.
    *out_bytes = 0;
    return cudaErrorNotSupported;
}

} // namespace pos2gpu
