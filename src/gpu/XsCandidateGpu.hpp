// XsCandidateGpu.hpp — minimal header carrying just the Xs_Candidate POD.
//
// Split out from XsKernel.cuh so the type can be referenced from non-CUDA
// translation units (notably the SYCL backend implementations), which can't
// pull in the CUDA-laden XsKernel.cuh → AesHashGpu.cuh → AesGpu.cuh chain.
//
// Layout mirrors pos2-chip/src/plot/TableConstructorGeneric.hpp:496 so a
// host-side reinterpret_cast to the pos2-chip type is safe.

#pragma once

#include <cstdint>

namespace pos2gpu {

struct XsCandidateGpu {
    uint32_t match_info;
    uint32_t x;
};
static_assert(sizeof(XsCandidateGpu) == 8, "must match pos2-chip Xs_Candidate layout");

} // namespace pos2gpu
