#pragma once

#include "gpu/AesHashGpu.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace pos2gpu {

cudaError_t launch_t3(
    AesHashKeys const& keys,
    void const* d_t2_pairings,
    uint64_t t2_count,
    void* d_t3_results,
    uint64_t* d_t3_count,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    int k,
    int minus_stub_bits,
    cudaStream_t stream = nullptr);

cudaError_t query_t3_temp_bytes(uint64_t t2_count, size_t* out_bytes);

} // namespace pos2gpu
