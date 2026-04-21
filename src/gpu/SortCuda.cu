// SortCuda.cu — CUB-backed implementation of the Sort.cuh wrappers.
// Compiled by nvcc; required when targeting NVIDIA. CUB's radix sort is
// state-of-the-art, so on NVIDIA we lean on it directly even from the
// SYCL host code by bridging the queue↔CUDA-stream boundary: drain the
// SYCL queue with q.wait(), run CUB on the default CUDA stream, then
// cudaStreamSynchronize(nullptr). Both backends share the same primary
// CUDA context (libcuda underneath both), so device pointers interop
// natively. Two host fences per sort call (~50µs each, well under
// 1ms/plot at the typical 3 sorts/plot rate).

// cuda_fp16.h must be included before sycl/sycl.hpp (pulled in via Sort.cuh)
// so AdaptiveCpp's half.hpp sees the __hdiv / __hlt / __hge intrinsics.
#include <cuda_fp16.h>

#include "gpu/Sort.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace pos2gpu {

namespace {

inline void cuda_check_or_throw(cudaError_t err, char const* what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUB ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

} // namespace

void launch_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
    if (d_temp_storage == nullptr) {
        // Sizing query — stream argument is unused.
        cuda_check_or_throw(cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes,
            keys_in, keys_out,
            vals_in, vals_out,
            count, begin_bit, end_bit, /*stream=*/nullptr),
            "SortPairs (sizing)");
        return;
    }

    // Drain the SYCL queue so any prior kernel writes to keys_in / vals_in
    // are visible before CUB runs.
    q.wait();

    cuda_check_or_throw(cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_bytes,
        keys_in, keys_out,
        vals_in, vals_out,
        count, begin_bit, end_bit, /*stream=*/nullptr),
        "SortPairs");

    // Wait for CUB to finish on the default stream so subsequent SYCL
    // submits see the sorted result.
    cuda_check_or_throw(cudaStreamSynchronize(nullptr),
        "cudaStreamSynchronize after SortPairs");
}

void launch_sort_keys_u64(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
    if (d_temp_storage == nullptr) {
        cuda_check_or_throw(cub::DeviceRadixSort::SortKeys(
            nullptr, temp_bytes,
            keys_in, keys_out,
            count, begin_bit, end_bit, /*stream=*/nullptr),
            "SortKeys (sizing)");
        return;
    }

    q.wait();

    cuda_check_or_throw(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_bytes,
        keys_in, keys_out,
        count, begin_bit, end_bit, /*stream=*/nullptr),
        "SortKeys");
    cuda_check_or_throw(cudaStreamSynchronize(nullptr),
        "cudaStreamSynchronize after SortKeys");
}

} // namespace pos2gpu
