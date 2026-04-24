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

#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu::sycl_backend {

// Async-exception handler for the persistent queue. AdaptiveCpp's
// default policy for unhandled async errors is to call std::terminate()
// via its `throw_result` path, which is what caused the observed
// "Aborted (core dumped)" after a synchronous malloc_device failure
// threw a clean std::runtime_error — secondary async errors (e.g. a
// CUDA:2 from in-flight work on the now-starved context) hit the
// default handler and killed the process before the CLI could exit
// normally. Logging and swallowing here keeps the synchronous
// std::runtime_error as the primary signal.
inline void async_error_handler(sycl::exception_list exns) noexcept
{
    for (std::exception_ptr const& ep : exns) {
        try { std::rethrow_exception(ep); }
        catch (sycl::exception const& e) {
            std::fprintf(stderr, "[sycl async] %s\n", e.what());
        }
        catch (std::exception const& e) {
            std::fprintf(stderr, "[sycl async] %s\n", e.what());
        }
        catch (...) {
            std::fprintf(stderr, "[sycl async] (unknown exception type)\n");
        }
    }
}

// Per-thread target device id. A worker thread sets this once at startup
// via set_current_device_id() so that its subsequent queue() call returns
// a queue bound to the requested GPU. Value of -1 (the default) means
// "use the default gpu_selector_v" — which is the single-device path, the
// only path pre-multi-GPU and the zero-configuration user experience.
//
// Thread-local, not global: the multi-device fan-out in BatchPlotter runs
// N worker threads, each binding to a distinct GPU. The main thread stays
// at -1 and sees the default selector.
inline int& current_device_id_ref()
{
    thread_local int id = -1;
    return id;
}

inline void set_current_device_id(int id)
{
    current_device_id_ref() = id;
}

inline int current_device_id()
{
    return current_device_id_ref();
}

// Per-thread SYCL queue. Bound to the thread's current device id, or to
// gpu_selector_v when the id is -1 (default, single-device path). A
// unique_ptr wrapper lets us defer construction until the thread has had
// a chance to set its device id.
//
// gpu_selector_v ensures the CUDA-backed GPU (or whichever AdaptiveCpp
// was configured for) is picked over the OpenMP host device.
inline sycl::queue& queue()
{
    thread_local std::unique_ptr<sycl::queue> q;
    if (!q) {
        int const id = current_device_id();
        if (id < 0) {
            q = std::make_unique<sycl::queue>(sycl::gpu_selector_v,
                                              async_error_handler);
        } else {
            auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (id >= static_cast<int>(devices.size())) {
                throw std::runtime_error(
                    "sycl_backend::queue: device id " + std::to_string(id) +
                    " out of range (found " + std::to_string(devices.size()) +
                    " GPU device(s))");
            }
            q = std::make_unique<sycl::queue>(devices[id], async_error_handler);
        }
    }
    return *q;
}

// Return the number of SYCL GPU devices visible to the process. Used by
// BatchOptions::use_all_devices to expand "all" into an explicit list.
inline int get_gpu_device_count()
{
    return static_cast<int>(
        sycl::device::get_devices(sycl::info::device_type::gpu).size());
}

// AES T-tables uploaded into a USM device buffer on first use, kept
// alive for the thread's queue lifetime — mirrors the CUDA path's
// __constant__ T-tables. Thread-local because each worker thread's queue
// is on a different device; the table upload must happen once per device,
// not once per process.
//
// Pointer layout matches what the _smem family expects: [T0|T1|T2|T3],
// 256 entries each.
inline uint32_t* aes_tables_device(sycl::queue& q)
{
    thread_local uint32_t* d_tables = nullptr;
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
