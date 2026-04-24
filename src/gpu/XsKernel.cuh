// XsKernel.cuh — generate and sort the Xs_Candidate array, the input to
// Table1Constructor. Bit-exact with pos2-chip's XsConstructor::construct.
//
// Layout: Xs_Candidate is { uint32_t match_info; uint32_t x; } — see
// pos2-chip/src/plot/TableConstructorGeneric.hpp:496. We mirror that
// layout here so a host-side reinterpret_cast to the pos2-chip type is
// safe without an explicit conversion.

#pragma once

#include "gpu/AesHashGpu.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace pos2gpu {

struct XsCandidateGpu {
    uint32_t match_info;
    uint32_t x;
};
static_assert(sizeof(XsCandidateGpu) == 8, "must match pos2-chip Xs_Candidate layout");

// Generate Xs_Candidate[2^k], sorted by match_info (low k bits, stable).
// Caller must have called initialize_aes_tables() once before invocation.
//
//   plot_id_bytes  : 32 bytes
//   k              : even, 18..32
//   testnet        : matches ProofHashing::g semantics — XORs x with
//                    TESTNET_G_XOR_CONST (0xA3B1C4D7) before hashing
//   d_out          : device buffer of at least (1ULL << k) XsCandidateGpu
//   d_temp_storage : device scratch; pass nullptr first to query size
//   temp_bytes     : in/out — when d_temp_storage is null, set to required size
//   stream         : optional CUDA stream
//
// Returns cudaSuccess on launch success. The sort is asynchronous on the
// stream — synchronize before reading d_out on the host.
cudaError_t launch_construct_xs(
    uint8_t const* plot_id_bytes,
    int k,
    bool testnet,
    XsCandidateGpu* d_out,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaStream_t stream = nullptr);

// Optional callback fired between the gen kernel and the sort, useful for
// per-stage cudaEvent timing. Pass nullptr to skip.
cudaError_t launch_construct_xs_profiled(
    uint8_t const* plot_id_bytes,
    int k,
    bool testnet,
    XsCandidateGpu* d_out,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaEvent_t after_gen,    // nullable; recorded after gen kernel queued
    cudaEvent_t after_sort,   // nullable; recorded after sort queued
    cudaStream_t stream = nullptr);

// Sub-kernel launchers — exposed so the low-VRAM streaming path can
// allocate keys_a/vals_a/keys_b/vals_b as SEPARATE buffers and free
// the gen-side pair between the sort and the pack. launch_construct_xs
// bundles them into a single d_temp_storage blob which makes freeing
// the intermediate keys_a+vals_a impossible before pack; splitting
// drops the Xs phase peak at k=28 from ~6.2 GB (d_xs 2 GB + blob
// 4.1 GB) to ~4.1 GB (max of sort-time and pack-time live sets) at
// zero per-plot PCIe cost.
//
// Caller is responsible for allocating d_keys_out/d_vals_out (each
// total × sizeof(uint32_t)) and initialising AES tables via
// initialize_aes_tables() before calling launch_xs_gen.
cudaError_t launch_xs_gen(
    uint8_t const* plot_id_bytes,
    int k,
    bool testnet,
    uint32_t* d_keys_out,
    uint32_t* d_vals_out,
    cudaStream_t stream = nullptr);

cudaError_t launch_xs_pack(
    uint32_t const* d_keys_in,
    uint32_t const* d_vals_in,
    XsCandidateGpu* d_out,
    uint64_t total,
    cudaStream_t stream = nullptr);

} // namespace pos2gpu
