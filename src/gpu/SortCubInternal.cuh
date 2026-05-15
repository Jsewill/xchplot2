// SortCubInternal.cuh — pure-CUDA, SYCL-free declarations of the
// CUB-backed radix sort. This header is the only entry point that
// SortCuda.cu (compiled by nvcc) needs to see — it deliberately
// does NOT include <sycl/sycl.hpp> so the nvcc translation unit
// never reaches into AdaptiveCpp's libkernel headers.
//
// AdaptiveCpp's expected consumer pattern is "compile through acpp,
// or stay out of the SYCL header tree." Pulling <sycl/sycl.hpp>
// into a .cu file hits the legacy CUDA branch of half.hpp's
// __acpp_backend_switch and tries to reference __hadd / __hsub /
// etc. that aren't in scope without cuda_fp16.h. Keeping nvcc TUs
// SYCL-free removes that whole class of bug.
//
// The SYCL-typed public API stays in Sort.cuh; SortSyclCub.cpp
// (compiled by acpp) bridges by draining the SYCL queue, calling
// these CUB symbols, and the cudaStreamSynchronize at the end is
// already done inside the CUB body — see comments below.

#pragma once

#include <cstdint>
#include <cstddef>

namespace pos2gpu {

// Pure-CUDA CUB radix sort. Caller responsibilities:
//   - Inputs (keys_in / vals_in) must be ready on the device — the
//     SYCL adapter handles this by draining the producing queue
//     with q.wait() before calling.
//   - Output is on the default CUDA stream and is fully drained
//     before the function returns (we cudaStreamSynchronize(nullptr)
//     internally so the caller can immediately consume keys_out /
//     vals_out without further fences).
//
// Sizing-query mode: pass d_temp_storage = nullptr; *temp_bytes is
// filled with the required scratch size and the function returns
// immediately without doing any work or any sync.
//
// Same in/out ping-pong contract as the SYCL-typed public API in
// Sort.cuh: keys_in/vals_in are clobbered, the result lands in
// keys_out/vals_out (memcpy from the CUB-chosen buffer if needed).
void cub_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit);

void cub_sort_keys_u64(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit);

// CUB segmented radix sort of (key, value) pairs. Sorts each of
// num_segments contiguous ranges of the (keys_in, vals_in) arrays
// independently, in a single CUB launch — significantly cheaper
// than a sequential loop of cub_sort_pairs_u32_u32 calls when the
// segments are small (~1M entries) because per-launch overhead
// dominates a tiny sort.
//
// Segments are described by d_begin_offsets[num_segments] and
// d_end_offsets[num_segments]. For contiguous segments (which is
// the two-level-sort caller's case) pass d_begin_offsets and
// d_begin_offsets + 1; segment i then runs from offsets[i] to
// offsets[i+1].
//
// Inputs are const here (CUB does not clobber them in this mode,
// since the result lands in d_keys_out / d_vals_out without a
// ping-pong via the inputs). Same internal-sync contract as the
// non-segmented wrapper above.
void cub_segmented_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t const* keys_in, uint32_t* keys_out,
    uint32_t const* vals_in, uint32_t* vals_out,
    uint64_t num_items,
    int num_segments,
    uint32_t const* d_begin_offsets,
    uint32_t const* d_end_offsets,
    int begin_bit, int end_bit);

} // namespace pos2gpu
