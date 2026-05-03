// CudaDeviceList.cu — implementation of list_cuda_devices().
// Compiled by nvcc so cuda_runtime.h is in scope; the host-facing
// header (CudaDeviceList.hpp) carries only plain types so cli.cpp
// (g++) doesn't need the CUDA include path.

#include "gpu/CudaDeviceList.hpp"

#include <cuda_runtime.h>

namespace pos2gpu {

CudaDeviceQueryResult list_cuda_devices()
{
    CudaDeviceQueryResult out;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        out.error = cudaGetErrorString(err);
        return out;
    }

    out.devices.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaError_t perr = cudaGetDeviceProperties(&prop, i);
        if (perr != cudaSuccess) {
            CudaDeviceInfo info{};
            info.id = i;
            info.name = std::string("(cudaGetDeviceProperties failed: ")
                      + cudaGetErrorString(perr) + ")";
            out.devices.push_back(std::move(info));
            continue;
        }
        CudaDeviceInfo info{};
        info.id         = i;
        info.name       = prop.name;
        info.vram_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
        info.sm_count   = prop.multiProcessorCount;
        info.cc_major   = prop.major;
        info.cc_minor   = prop.minor;
        out.devices.push_back(std::move(info));
    }
    return out;
}

} // namespace pos2gpu
