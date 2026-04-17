// T3Kernel.cu — final phase: sort T2 pairings, run a second AES pass for
// chaining, drop stub bits, emit ProofFragments ready for CPU FSE
// compression.
//
// CPU reference: pos2-chip/src/plot/TableConstructorGeneric.hpp
//   Table3Constructor::construct(t2_pairs, out, post_sort_tmp).
//
// Status: skeleton. Same shape as T2 — sort+match+chain wired structurally,
// the predicate/bit-drop body is a TODO.

#include "gpu/AesHashGpu.cuh"
#include "gpu/T3Kernel.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace pos2gpu {

cudaError_t launch_t3(
    AesHashKeys const& /*keys*/,
    void const* /*d_t2_pairings*/,
    uint64_t /*t2_count*/,
    void* /*d_t3_results*/,    // T3Pairing[] with proof_fragment field
    uint64_t* /*d_t3_count*/,
    void* /*d_temp_storage*/,
    size_t /*temp_storage_bytes*/,
    int /*k*/,
    int /*minus_stub_bits*/,
    cudaStream_t /*stream*/)
{
    // TODO:
    //   1. cub::DeviceRadixSort::SortPairs T2 pairings by output key.
    //   2. Sliding-window match (table_id=3 uses strength match key bits;
    //      see ProofParams::get_num_match_key_bits).
    //   3. For each match: compute the chained AES (AesHash::chain or
    //      pairing depending on the path) over the 8-x sequence.
    //   4. Bit-drop: each fragment retains (k - PlotFile::MINUS_STUB_BITS)
    //      bits of stub data. See PlotFile::MINUS_STUB_BITS = 2.
    //   5. Write the resulting ProofFragment (uint64_t deltas + stub bits)
    //      into d_t3_results in *sorted* order — the CPU FSE compressor
    //      assumes ascending fragment values per chunk.
    return cudaErrorNotSupported;
}

cudaError_t query_t3_temp_bytes(uint64_t /*t2_count*/, size_t* out_bytes)
{
    *out_bytes = 0;
    return cudaErrorNotSupported;
}

} // namespace pos2gpu
