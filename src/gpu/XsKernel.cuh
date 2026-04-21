// XsKernel.cuh — generate and sort the Xs_Candidate array, the input to
// Table1Constructor. Bit-exact with pos2-chip's XsConstructor::construct.
//
// Layout: Xs_Candidate is { uint32_t match_info; uint32_t x; } — see
// pos2-chip/src/plot/TableConstructorGeneric.hpp:496. We mirror that
// layout here so a host-side reinterpret_cast to the pos2-chip type is
// safe without an explicit conversion.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/XsCandidateGpu.hpp"

#include <cuda_runtime.h>

#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace pos2gpu {

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
void launch_construct_xs(
    uint8_t const* plot_id_bytes,
    int k,
    bool testnet,
    XsCandidateGpu* d_out,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q);

// Optional callback fired between the gen kernel and the sort, useful for
// per-stage cudaEvent timing. Pass nullptr to skip.
void launch_construct_xs_profiled(
    uint8_t const* plot_id_bytes,
    int k,
    bool testnet,
    XsCandidateGpu* d_out,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaEvent_t after_gen,    // nullable; recorded after gen kernel queued
    cudaEvent_t after_sort,   // nullable; recorded after sort queued
    sycl::queue& q);

} // namespace pos2gpu
