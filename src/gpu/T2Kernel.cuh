#pragma once

#include "gpu/AesHashGpu.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace pos2gpu {

cudaError_t launch_t2(
    AesHashKeys const& keys,
    void const* d_t1_pairings,
    uint64_t t1_count,
    void* d_t2_pairings,
    uint64_t* d_t2_count,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    cudaStream_t stream = nullptr);

cudaError_t query_t2_temp_bytes(uint64_t t1_count, size_t* out_bytes);

} // namespace pos2gpu
