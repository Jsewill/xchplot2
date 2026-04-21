// Sort.cuh — backend-dispatched radix sort wrappers.
//
// Two implementations:
//   SortCuda.cu — CUB-backed, compiled by nvcc. NVIDIA-only target. The
//                 wrapper takes sycl::queue& q and bridges by draining q
//                 with q.wait(), calling CUB on the default stream, then
//                 cudaStreamSynchronize(nullptr). CUB and the SYCL backend
//                 share the same primary CUDA context (libcuda underneath
//                 both), so device pointers interop natively. ~2 host
//                 fences per sort call (~50µs each, well under 1ms/plot).
//   SortSycl.cpp — TODO: oneDPL-backed for AMD/Intel targets. Slower than
//                  CUB on NVIDIA but the only path on non-NVIDIA hardware.
//
// CMake selects between them based on the target. For now (NVIDIA-only)
// SortCuda.cu is always built.
//
// API mirrors CUB's two-mode contract: pass d_temp_storage=nullptr to
// query the required temp_bytes; pass real storage to perform the sort.

#pragma once

#include <cstdint>
#include <cstddef>

#include <sycl/sycl.hpp>

namespace pos2gpu {

// Sort (key, value) pairs by uint32 key over [begin_bit, end_bit) bits.
// Stable. Used for T1 / T2 / Xs sorts (key=match_info, value=index or x).
//
// Both keys_in/vals_in AND keys_out/vals_out are writable: the SYCL
// implementation uses them as a ping-pong pair across radix passes to
// avoid allocating its own (8 × N bytes) alt buffers. Caller treats
// keys_in/vals_in as scratch on input — they get clobbered. The result
// always lands in keys_out/vals_out (the wrapper does a final memcpy
// internally if the pass count is odd). The CUB backend ignores the
// non-constness — it still treats keys_in/vals_in as read-only.
void launch_sort_pairs_u32_u32(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint32_t* keys_in, uint32_t* keys_out,
    uint32_t* vals_in, uint32_t* vals_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q);

// Sort uint64 keys over [begin_bit, end_bit) bits. Used for the final
// T3 fragment sort (sort by proof_fragment's low 2k bits).
// Same in/out ping-pong contract as launch_sort_pairs_u32_u32.
void launch_sort_keys_u64(
    void* d_temp_storage,
    size_t& temp_bytes,
    uint64_t* keys_in, uint64_t* keys_out,
    uint64_t count,
    int begin_bit, int end_bit,
    sycl::queue& q);

} // namespace pos2gpu
