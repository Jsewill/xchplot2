// T1Kernel.cu — generate f1(x) = AesHash::g_x(x, k) for x in [0, 2^k) on
// GPU. This is the parallelisable foundation of T1; matching to form
// T1Pairings happens in a later kernel (sort + sliding window) which is
// stubbed below pending parity confirmation of g_x itself.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/T1Kernel.cuh"

#include <cuda_runtime.h>

namespace pos2gpu {

namespace {

__global__ void g_x_kernel(
    AesHashKeys keys,
    uint32_t* __restrict__ out,
    uint64_t total,
    int k,
    int rounds)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    out[idx] = g_x(keys, static_cast<uint32_t>(idx), k, rounds);
}

} // namespace

cudaError_t launch_g_x(
    uint8_t const* plot_id_bytes,
    int k,
    uint32_t* d_out,
    uint64_t total,
    int rounds,
    cudaStream_t stream)
{
    AesHashKeys keys = make_keys(plot_id_bytes);
    constexpr int kThreads = 256;
    uint64_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > std::numeric_limits<unsigned>::max()) {
        return cudaErrorInvalidValue;
    }
    g_x_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        keys, d_out, total, k, rounds);
    return cudaGetLastError();
}

// TODO: full T1 matching kernel.
//
// CPU reference: pos2-chip/src/plot/TableConstructorGeneric.hpp
//   Table1Constructor::construct(xs_candidates, out, post_sort_tmp).
// Pipeline:
//   1. Compute g_x for all x (this kernel above).
//   2. For each x build (section, match_key, target) using ProofParams::
//      extract_section_from_match_info / extract_match_key_from_match_info.
//   3. Radix-sort by (section, match_key, target) — use cub::DeviceRadixSort.
//   4. Sliding-window match adjacent entries with equal (section, match_key)
//      whose targets satisfy the matching predicate (see AesHash::
//      matching_target). Emit T1Pairing { meta = (xL << k) | xR, ... }.
//   5. Stream-compact via cub::DeviceSelect into output buffer.
//
// Until the predicate / packing semantics are mirrored end-to-end, the
// gpu_plotter binary calls Table1Constructor::construct on host with the
// CPU AesHash to fall back. Once t1_parity passes byte-for-byte, replace
// the host call with launch_t1_match() below.

cudaError_t launch_t1_match(
    AesHashKeys const& /*keys*/,
    uint32_t const* /*d_g_x_values*/,
    uint64_t /*total*/,
    void* /*d_out_pairings*/,
    uint64_t* /*d_out_count*/,
    cudaStream_t /*stream*/)
{
    // Not implemented yet; intentionally returns success so the harness can
    // wire up call sites and fall back to CPU until this is filled in.
    return cudaErrorNotSupported;
}

} // namespace pos2gpu
