#pragma once

#include "gpu/AesHashGpu.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace pos2gpu {

// Phase 1a: compute f1(x) = g_x(x, k) for x in [0, total). Output buffer
// must be at least `total` uint32_t. Async on `stream`.
cudaError_t launch_g_x(
    uint8_t const* plot_id_bytes,
    int k,
    uint32_t* d_out,
    uint64_t total,
    int rounds = kAesGRounds,
    cudaStream_t stream = nullptr);

// Phase 1b: form T1 pairings from the g_x values. Currently NOT
// implemented — see T1Kernel.cu for the design notes. Returns
// cudaErrorNotSupported until the matching kernel lands.
cudaError_t launch_t1_match(
    AesHashKeys const& keys,
    uint32_t const* d_g_x_values,
    uint64_t total,
    void* d_out_pairings,
    uint64_t* d_out_count,
    cudaStream_t stream = nullptr);

} // namespace pos2gpu
