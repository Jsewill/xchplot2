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
#include "gpu/DeviceIds.hpp"

// cuda_fp16.h must precede sycl/sycl.hpp when this header is consumed
// from an nvcc TU — AdaptiveCpp's libkernel/detail/half_representation.hpp
// references __half, which only exists once cuda_fp16 has been seen.
#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>

#include <algorithm>
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
// a queue bound to the requested device. Sentinel values:
//   kDefaultGpuId (-1)  : sycl::gpu_selector_v (single-device default,
//                         pre-multi-GPU zero-config path)
//   kCpuDeviceId  (-2)  : sycl::cpu_selector_v (latent — kept so a future
//                         SYCL-on-CPU benchmark path can compare against
//                         pos2-chip's hand-tuned CPU plotter; production
//                         --cpu / --devices cpu plotting bypasses this
//                         and dispatches directly to run_one_plot_cpu()
//                         in BatchPlotter, see CpuPlotter.cpp)
//   0..N-1              : explicit GPU index from
//                         sycl::device::get_devices(gpu)
//
// Thread-local, not global: the multi-device fan-out in BatchPlotter runs
// N worker threads, each binding to a distinct device. The main thread
// stays at kDefaultGpuId and sees the default selector.
inline int& current_device_id_ref()
{
    thread_local int id = kDefaultGpuId;
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

// Mixed-vendor SYCL host filter: when this build links the CUB sort path
// (XCHPLOT2_HAVE_CUB), drop any non-CUDA SYCL devices from the
// enumeration. Otherwise a host with NVIDIA + AMD (e.g. user passed
// `--gpus all` AND `--device /dev/kfd --device /dev/dri` to docker)
// returns 2+ "GPU devices" from the SYCL view, BatchPlotter's
// `--devices all` spawns a worker per device, and the CUB sort path
// errors out with `cudaErrorInvalidDevice` ("invalid device ordinal")
// when CUB is called against the AMD card. Skipping non-CUDA backends
// here keeps the enumeration aligned with what CUB can actually use.
//
// Intel L0 / OCL devices are likewise filtered; HIP-only builds (the
// rocm container) wouldn't define XCHPLOT2_HAVE_CUB and pass through.
inline std::vector<sycl::device> usable_gpu_devices()
{
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
#ifdef XCHPLOT2_HAVE_CUB
    devs.erase(std::remove_if(devs.begin(), devs.end(),
        [](sycl::device const& d) {
            return d.get_backend() != sycl::backend::cuda;
        }),
        devs.end());
#endif
    return devs;
}

// Per-thread SYCL queue. Bound to the thread's current device id (see
// the kDefaultGpuId / kCpuDeviceId sentinels above). A unique_ptr wrapper
// lets us defer construction until the thread has had a chance to set
// its device id.
//
// gpu_selector_v ensures the CUDA-backed GPU (or whichever AdaptiveCpp
// was configured for) is picked over the OpenMP host device. cpu_selector_v
// bypasses GPU enumeration entirely and lands on AdaptiveCpp's OMP backend
// (CPU build path, ACPP_TARGETS=omp).
inline sycl::queue& queue()
{
    thread_local std::unique_ptr<sycl::queue> q;
    if (!q) {
        int const id = current_device_id();
        if (id == kCpuDeviceId) {
            q = std::make_unique<sycl::queue>(sycl::cpu_selector_v,
                                              async_error_handler);
        } else if (id < 0) {
            q = std::make_unique<sycl::queue>(sycl::gpu_selector_v,
                                              async_error_handler);
        } else {
            auto devices = usable_gpu_devices();
            if (id >= static_cast<int>(devices.size())) {
                throw std::runtime_error(
                    "sycl_backend::queue: device id " + std::to_string(id) +
                    " out of range (found " + std::to_string(devices.size()) +
                    " usable GPU device(s))");
            }
            q = std::make_unique<sycl::queue>(devices[id], async_error_handler);
        }
    }
    return *q;
}

// Return the number of SYCL GPU devices visible to the process AND
// usable by this build. Used by BatchOptions::use_all_devices to expand
// "all" into an explicit list. See usable_gpu_devices() for the filter.
inline int get_gpu_device_count()
{
    return static_cast<int>(usable_gpu_devices().size());
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
