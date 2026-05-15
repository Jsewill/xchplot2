// SortCuda.cu — CUB-backed implementation of the Sort.cuh wrappers.
// Compiled by nvcc; required when targeting NVIDIA. CUB's radix sort is
// state-of-the-art, so on NVIDIA we lean on it directly even from the
// SYCL host code by bridging the queue↔CUDA-stream boundary: drain the
// SYCL queue with q.wait(), run CUB on the default CUDA stream, then
// cudaStreamSynchronize(nullptr). Both backends share the same primary
// CUDA context (libcuda underneath both), so device pointers interop
// natively. Two host fences per sort call (~50µs each, well under
// 1ms/plot at the typical 3 sorts/plot rate).

// Pure-CUDA TU — never include <sycl/sycl.hpp> here, directly or
// transitively. AdaptiveCpp's libkernel reaches into nvcc's CUDA
// device pass via __acpp_backend_switch when the SYCL header is in
// scope, and that path was never intended to be used from
// nvcc-driver-compiled consumer TUs (per the AdaptiveCpp dev's
// guidance: stick to --acpp-targets=generic, or stay out of the
// SYCL header tree from non-acpp compilers). The SYCL-typed entry
// points live in SortSyclCub.cpp (compiled by acpp) and call into
// the cub_sort_* declarations below.
#include "gpu/SortCubInternal.cuh"

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

// CUB DoubleBuffer mode: caller passes both buffers as a ping-pong pair,
// CUB picks which one the result lands in (db.Current()), and CUB's own
// scratch shrinks to ~MB of histograms instead of ~2 GB of internal
// temp keys/vals buffers it would otherwise allocate. We then memcpy
// db.Current() to keys_out if needed so the public API contract holds.
//
// Caller (SortSyclCub.cpp) drains the producing SYCL queue with q.wait()
// before this is called. This function syncs the default CUDA stream
// internally before returning so the caller can hand keys_out / vals_out
// straight back to SYCL without another fence.
void cub_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit)
{
    if (d_temp_storage == nullptr) {
        cub::DoubleBuffer<uint32_t> d_keys(keys_in, keys_out);
        cub::DoubleBuffer<uint32_t> d_vals(vals_in, vals_out);
        cuda_check_or_throw(cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes,
            d_keys, d_vals,
            static_cast<int>(count), begin_bit, end_bit, /*stream=*/nullptr),
            "SortPairs (sizing)");
        return;
    }

    cub::DoubleBuffer<uint32_t> d_keys(keys_in, keys_out);
    cub::DoubleBuffer<uint32_t> d_vals(vals_in, vals_out);
    cuda_check_or_throw(cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_bytes,
        d_keys, d_vals,
        static_cast<int>(count), begin_bit, end_bit, /*stream=*/nullptr),
        "SortPairs");

    // CUB picks the output buffer; copy to keys_out/vals_out if it landed
    // in keys_in/vals_in instead.
    if (d_keys.Current() != keys_out) {
        cuda_check_or_throw(cudaMemcpyAsync(keys_out, d_keys.Current(),
            count * sizeof(uint32_t), cudaMemcpyDeviceToDevice, nullptr),
            "memcpy keys_out");
    }
    if (d_vals.Current() != vals_out) {
        cuda_check_or_throw(cudaMemcpyAsync(vals_out, d_vals.Current(),
            count * sizeof(uint32_t), cudaMemcpyDeviceToDevice, nullptr),
            "memcpy vals_out");
    }

    cuda_check_or_throw(cudaStreamSynchronize(nullptr),
        "cudaStreamSynchronize after SortPairs");
}

void cub_sort_keys_u64(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit)
{
    if (d_temp_storage == nullptr) {
        cub::DoubleBuffer<uint64_t> d_keys(keys_in, keys_out);
        cuda_check_or_throw(cub::DeviceRadixSort::SortKeys(
            nullptr, temp_bytes,
            d_keys,
            static_cast<int>(count), begin_bit, end_bit, /*stream=*/nullptr),
            "SortKeys (sizing)");
        return;
    }

    cub::DoubleBuffer<uint64_t> d_keys(keys_in, keys_out);
    cuda_check_or_throw(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_bytes,
        d_keys,
        static_cast<int>(count), begin_bit, end_bit, /*stream=*/nullptr),
        "SortKeys");

    if (d_keys.Current() != keys_out) {
        cuda_check_or_throw(cudaMemcpyAsync(keys_out, d_keys.Current(),
            count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, nullptr),
            "memcpy keys_out");
    }

    cuda_check_or_throw(cudaStreamSynchronize(nullptr),
        "cudaStreamSynchronize after SortKeys");
}

void cub_sort_pairs_u32_u64(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint64_t* vals_in, uint64_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit)
{
    if (d_temp_storage == nullptr) {
        cub::DoubleBuffer<uint32_t> d_keys(keys_in, keys_out);
        cub::DoubleBuffer<uint64_t> d_vals(vals_in, vals_out);
        cuda_check_or_throw(cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes,
            d_keys, d_vals,
            static_cast<int>(count), begin_bit, end_bit, /*stream=*/nullptr),
            "SortPairs u32_u64 (sizing)");
        return;
    }

    cub::DoubleBuffer<uint32_t> d_keys(keys_in, keys_out);
    cub::DoubleBuffer<uint64_t> d_vals(vals_in, vals_out);
    cuda_check_or_throw(cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_bytes,
        d_keys, d_vals,
        static_cast<int>(count), begin_bit, end_bit, /*stream=*/nullptr),
        "SortPairs u32_u64");

    if (d_keys.Current() != keys_out) {
        cuda_check_or_throw(cudaMemcpyAsync(keys_out, d_keys.Current(),
            count * sizeof(uint32_t), cudaMemcpyDeviceToDevice, nullptr),
            "memcpy keys_out u32_u64");
    }
    if (d_vals.Current() != vals_out) {
        cuda_check_or_throw(cudaMemcpyAsync(vals_out, d_vals.Current(),
            count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, nullptr),
            "memcpy vals_out u32_u64");
    }

    cuda_check_or_throw(cudaStreamSynchronize(nullptr),
        "cudaStreamSynchronize after SortPairs u32_u64");
}

// CUB segmented radix sort. Sorts `num_segments` independent
// segments of the (keys_in, vals_in) arrays in a single launch.
// d_begin_offsets[i] / d_end_offsets[i] are the inclusive/exclusive
// segment bounds; for contiguous segments the caller can pass
// d_begin_offsets and d_begin_offsets + 1 (since segment i ends
// where segment i+1 begins).
//
// We don't use the DoubleBuffer variant here: segmented sort with
// uint32_t offsets has narrower template instantiations and the
// non-DoubleBuffer signature lands the result directly in
// (keys_out, vals_out) without the post-hoc memcpy dance the
// regular sort needs.
void cub_segmented_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t const* keys_in, uint32_t* keys_out,
    uint32_t const* vals_in, uint32_t* vals_out,
    uint64_t num_items,
    int num_segments,
    uint32_t const* d_begin_offsets,
    uint32_t const* d_end_offsets,
    int begin_bit, int end_bit)
{
    if (d_temp_storage == nullptr) {
        cuda_check_or_throw(cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, temp_bytes,
            keys_in, keys_out,
            vals_in, vals_out,
            static_cast<int>(num_items),
            num_segments,
            d_begin_offsets, d_end_offsets,
            begin_bit, end_bit, /*stream=*/nullptr),
            "SegmentedSortPairs (sizing)");
        return;
    }

    cuda_check_or_throw(cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_bytes,
        keys_in, keys_out,
        vals_in, vals_out,
        static_cast<int>(num_items),
        num_segments,
        d_begin_offsets, d_end_offsets,
        begin_bit, end_bit, /*stream=*/nullptr),
        "SegmentedSortPairs");

    cuda_check_or_throw(cudaStreamSynchronize(nullptr),
        "cudaStreamSynchronize after SegmentedSortPairs");
}

} // namespace pos2gpu
