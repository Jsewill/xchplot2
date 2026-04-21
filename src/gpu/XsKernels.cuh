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

void launch_xs_pack(
    uint32_t const* keys_in,
    uint32_t const* vals_in,
    XsCandidateGpu* d_out,
    uint64_t total,
    sycl::queue& q);

} // namespace pos2gpu
