// SyclBackend.hpp — shared SYCL infrastructure for the cross-backend
// kernel implementations in T*OffsetsSycl.cpp.
//
// Both helpers are header-only inline so multiple SYCL TUs (T1OffsetsSycl,
// T2OffsetsSycl, T3OffsetsSycl) share a single queue and a single AES
// T-table USM buffer per process — function-local statics inside inline
// functions have unique-instance semantics under ISO C++17+.
//
// This file is consumed only by the SYCL backend; CUDA TUs never include
// it. It depends on PortableAttrs.hpp solely for the AesTables namespace
// dependency through AesTables.inl, which has no CUDA-specific content.

#pragma once

#include "gpu/AesTables.inl"

// cuda_fp16.h must precede sycl/sycl.hpp when this header is consumed
// from an nvcc TU — AdaptiveCpp's libkernel/detail/half_representation.hpp
// references __half, which only exists once cuda_fp16 has been seen.
#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>

#include <vector>

namespace pos2gpu::sycl_backend {

// Persistent SYCL queue. gpu_selector_v ensures the CUDA-backed RTX 4090
// (or whichever GPU the AdaptiveCpp build was configured for) is picked
// over the AdaptiveCpp OpenMP host device that's also visible.
inline sycl::queue& queue()
{
    static sycl::queue q{ sycl::gpu_selector_v };
    return q;
}

// AES T-tables uploaded into a USM device buffer on first use, kept
// alive for the process lifetime — mirrors the CUDA path's __constant__
// T-tables, which are also never freed. Pointer layout matches what the
// _smem family expects: [T0|T1|T2|T3], 256 entries each.
inline uint32_t* aes_tables_device(sycl::queue& q)
{
    static uint32_t* d_tables = nullptr;
    if (d_tables) return d_tables;

    std::vector<uint32_t> sT_host(4 * 256);
    for (int i = 0; i < 256; ++i) {
        sT_host[0 * 256 + i] = pos2gpu::aes_tables::T0[i];
        sT_host[1 * 256 + i] = pos2gpu::aes_tables::T1[i];
        sT_host[2 * 256 + i] = pos2gpu::aes_tables::T2[i];
        sT_host[3 * 256 + i] = pos2gpu::aes_tables::T3[i];
    }
    d_tables = sycl::malloc_device<uint32_t>(4 * 256, q);
    q.memcpy(d_tables, sT_host.data(), sizeof(uint32_t) * 4 * 256).wait();
    return d_tables;
}

} // namespace pos2gpu::sycl_backend
