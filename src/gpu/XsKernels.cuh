// XsKernels.cuh — backend-dispatched wrappers for the two non-sort phases
// of Xs construction. The orchestration (sizing query, sort, fold-into-AoS)
// lives in XsKernel.cpp and chains these via a sycl::queue.
//
// Phase 1: launch_xs_gen — fill (keys_out[x], vals_out[x]) = (g_x(x⊕xor_const), x)
//          for x in [0, total). Loads AES T-tables into local memory once
//          per workgroup, mirroring the CUDA gen_kernel pattern.
//
// Phase 3: launch_xs_pack — pack sorted (keys_in, vals_in) back into AoS
//          XsCandidateGpu[total]. Pure grid-stride; no AES.

#pragma once

#include "gpu/AesHashGpu.cuh"
#include "gpu/XsCandidateGpu.hpp"

#include <cstdint>

#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>

namespace pos2gpu {

void launch_xs_gen(
    AesHashKeys keys,
    uint32_t* keys_out,
    uint32_t* vals_out,
    uint64_t total,
    int k,
    uint32_t xor_const,
    sycl::queue& q);

// Position-range variant of launch_xs_gen. Generates Xs candidates for
// positions x ∈ [pos_begin, pos_end) and writes to keys_out[i] /
// vals_out[i] where i = x - pos_begin (relative indexing). keys_out /
// vals_out must be sized for at least (pos_end - pos_begin) elements.
// Used by minimal tier to tile the Xs gen + sort phase below the
// 4 GiB-cap peak.
void launch_xs_gen_range(
    AesHashKeys keys,
    uint32_t* keys_out,
    uint32_t* vals_out,
    uint64_t pos_begin,
    uint64_t pos_end,
    int k,
    uint32_t xor_const,
    sycl::queue& q);

void launch_xs_pack(
    uint32_t const* keys_in,
    uint32_t const* vals_in,
    XsCandidateGpu* d_out,
    uint64_t total,
    sycl::queue& q);

// Position-range variant of launch_xs_pack. Reads keys_in[i] / vals_in[i]
// for i ∈ [0, count) and writes XsCandidateGpu{keys_in[i], vals_in[i]}
// to d_out[i + dst_begin]. Lets the caller pack incrementally.
void launch_xs_pack_range(
    uint32_t const* keys_in,
    uint32_t const* vals_in,
    XsCandidateGpu* d_out,
    uint64_t count,
    sycl::queue& q);

} // namespace pos2gpu
