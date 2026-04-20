// SortSycl.cpp — non-CUDA Sort.cuh wrapper stub.
//
// Compiled when XCHPLOT2_BUILD_CUDA=OFF. The CUB-backed implementation in
// SortCuda.cu requires nvcc and is the right choice on NVIDIA hardware;
// for AMD/Intel targets we'll land a real SYCL radix sort in a follow-up
// slice. Until then, this TU exists so the SYCL build links — calling
// either entry point throws.

#include "gpu/Sort.cuh"

#include <stdexcept>

namespace pos2gpu {

void launch_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t const* /*keys_in*/, uint32_t* /*keys_out*/,
    uint32_t const* /*vals_in*/, uint32_t* /*vals_out*/,
    uint64_t /*count*/,
    int /*begin_bit*/, int /*end_bit*/,
    sycl::queue& /*q*/)
{
    if (d_temp_storage == nullptr) {
        temp_bytes = 0;
        return;
    }
    throw std::runtime_error(
        "launch_sort_pairs_u32_u32: SYCL sort backend not yet implemented; "
        "build with XCHPLOT2_BUILD_CUDA=ON to use the CUB path");
}

void launch_sort_keys_u64(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t const* /*keys_in*/, uint64_t* /*keys_out*/,
    uint64_t /*count*/,
    int /*begin_bit*/, int /*end_bit*/,
    sycl::queue& /*q*/)
{
    if (d_temp_storage == nullptr) {
        temp_bytes = 0;
        return;
    }
    throw std::runtime_error(
        "launch_sort_keys_u64: SYCL sort backend not yet implemented; "
        "build with XCHPLOT2_BUILD_CUDA=ON to use the CUB path");
}

} // namespace pos2gpu
