// SortSyclCub.cpp — SYCL-typed entry points for the CUB-backed sort.
//
// Compiled by acpp (the AdaptiveCpp compiler), so <sycl/sycl.hpp>
// is in scope here. SortCuda.cu (compiled by nvcc) used to provide
// these directly with a `sycl::queue&` parameter, but that meant
// nvcc was reaching into AdaptiveCpp's libkernel headers — a path
// AdaptiveCpp doesn't intend to support. We now keep nvcc's view
// SYCL-free (see SortCubInternal.cuh) and bridge here:
//
//   q.wait()                             — drain the producing SYCL
//                                          queue so CUB sees the
//                                          right inputs.
//   cub_sort_*(...)                      — pure-CUDA CUB kernel +
//                                          internal cudaStreamSync.
//
// This file is only built when XCHPLOT2_BUILD_CUDA=ON. The dispatcher
// in SortDispatch.cpp routes here for CUDA-backend queues; non-CUDA
// queues (HIP / Level Zero / OpenMP host) flow to SortSycl.cpp's
// launch_sort_*_sycl variants instead. AMD-only / Intel-only / CPU
// builds skip this file entirely (BUILD_CUDA=OFF).

#include "gpu/Sort.cuh"
#include "gpu/SortCubInternal.cuh"

namespace pos2gpu {

void launch_sort_pairs_u32_u32_cub(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
    // The sizing-query path (d_temp_storage == nullptr) never touches
    // device memory — no need to fence the SYCL queue.
    if (d_temp_storage != nullptr) {
        q.wait();
    }
    cub_sort_pairs_u32_u32(d_temp_storage, temp_bytes,
        keys_in, keys_out, vals_in, vals_out,
        count, begin_bit, end_bit);
}

void launch_sort_keys_u64_cub(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
    if (d_temp_storage != nullptr) {
        q.wait();
    }
    cub_sort_keys_u64(d_temp_storage, temp_bytes,
        keys_in, keys_out, count, begin_bit, end_bit);
}

void launch_segmented_sort_pairs_u32_u32_cub(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t const* keys_in, uint32_t* keys_out,
    uint32_t const* vals_in, uint32_t* vals_out,
    uint64_t num_items,
    int num_segments,
    uint32_t const* d_begin_offsets,
    uint32_t const* d_end_offsets,
    int begin_bit, int end_bit,
    sycl::queue& q)
{
    if (d_temp_storage != nullptr) {
        q.wait();
    }
    cub_segmented_sort_pairs_u32_u32(d_temp_storage, temp_bytes,
        keys_in, keys_out, vals_in, vals_out,
        num_items, num_segments,
        d_begin_offsets, d_end_offsets,
        begin_bit, end_bit);
}

} // namespace pos2gpu
