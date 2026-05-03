// CudaDeviceList.hpp — plain-types declaration for `xchplot2 devices`
// (and any other consumer that needs to enumerate CUDA devices without
// pulling cuda_runtime.h into its TU).
//
// cli.cpp is compiled by g++ and doesn't have CUDA's headers on its
// include path. The implementation in CudaDeviceList.cu is compiled by
// nvcc, which does, so the host code only sees this plain-types view.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pos2gpu {

struct CudaDeviceInfo {
    int           id;
    std::string   name;
    std::uint64_t vram_bytes;
    int           sm_count;       // multiProcessorCount
    int           cc_major;       // major compute capability
    int           cc_minor;       // minor compute capability
};

struct CudaDeviceQueryResult {
    std::vector<CudaDeviceInfo> devices;
    // Non-empty when the query failed (no driver, no GPU, libcuda
    // mismatch, etc.) — caller surfaces it as a single user-facing
    // error line. Empty `devices` with empty `error` means a healthy
    // driver reporting zero installed GPUs.
    std::string error;
};

CudaDeviceQueryResult list_cuda_devices();

} // namespace pos2gpu
