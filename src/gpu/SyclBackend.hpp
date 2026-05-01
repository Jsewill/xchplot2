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
//
// Runs a one-shot dispatch sanity check on first construction (see
// validate_kernel_dispatch below). If AdaptiveCpp's HIP / CUDA backend
// on this host produces a no-op kernel stub at JIT/AOT time, the throw
// surfaces here — at the first GPU work request — instead of much later
// as a confusing "T1 match produced 0 entries" / streaming-tier error.
// Set POS2GPU_SKIP_SELFTEST=1 to bypass; useful when you've already
// validated the device this session and want lower startup overhead
// across many short-lived processes.
inline void validate_kernel_dispatch(sycl::queue& q)
{
    if (char const* v = std::getenv("POS2GPU_SKIP_SELFTEST"); v && v[0] == '1') {
        return;
    }

    constexpr std::size_t   N        = 16;
    constexpr std::uint32_t kPattern = 0xDEADBEEFu;

    std::uint32_t* d = sycl::malloc_device<std::uint32_t>(N, q);
    if (!d) {
        throw std::runtime_error(
            "[selftest] sycl::malloc_device(16 * u32) returned null. "
            "The SYCL runtime can't allocate even tiny device buffers — "
            "device discovery probably failed (check rocminfo / nvidia-smi, "
            "ACPP_VISIBILITY_MASK).");
    }

    // Sentinel-fill: a "no kernel writes landed" outcome shows the
    // sentinel, not random uninitialised bytes that might happen to
    // match the expected pattern by coincidence.
    q.memset(d, 0xCD, N * sizeof(std::uint32_t)).wait();
    q.parallel_for(sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> it) {
        std::size_t idx = it.get_global_id(0);
        d[idx] = kPattern + static_cast<std::uint32_t>(idx);
    }).wait();

    std::uint32_t host[N] = {};
    q.memcpy(host, d, N * sizeof(std::uint32_t)).wait();
    sycl::free(d, q);

    int fails = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (host[i] != kPattern + static_cast<std::uint32_t>(i)) ++fails;
    }
    if (fails == 0) return;

    char head[64];
    std::snprintf(head, sizeof(head), "0x%08x (expected 0x%08x)",
                  host[0], kPattern);
    std::string msg =
        "[selftest] SYCL kernel writes are not landing on the device. "
        "A trivial parallel_for(16) writing a known pattern produced "
        "host[0]=";
    msg += head;
    msg += ".\n  ";
    if (host[0] == 0xCDCDCDCDu) {
        msg += "The pre-launch sentinel (0xCDCDCDCD) is intact, so the "
               "kernel completed without writing anything. ";
    } else {
        msg += "The sentinel was overwritten but with a wrong value — "
               "the kernel is dispatching but its output is corrupted. ";
    }
    msg += "Most likely AdaptiveCpp's HIP / CUDA backend on this host is "
           "producing a no-op or miscompiled kernel stub at JIT/AOT time. "
           "Diagnose with:\n"
           "  - ACPP_DEBUG_LEVEL=2 ./xchplot2 ...   (shows the JIT log)\n"
           "  - rocminfo / nvidia-smi              (confirm the actual ISA "
           "matches the AOT target — see cargo:warning lines from your "
           "last `cargo install`)\n"
           "  - try ACPP_TARGETS=generic           (forces SSCP JIT instead "
           "of an AOT spoof)\n"
           "Bypass the self-test with POS2GPU_SKIP_SELFTEST=1 if you've "
           "already validated this device this session.";
    throw std::runtime_error(msg);
}

inline sycl::queue& queue()
{
    thread_local std::unique_ptr<sycl::queue> q;
    if (!q) {
        int const id = current_device_id();
        if (id == kCpuDeviceId) {
            // AdaptiveCpp's OpenMP backend exposes its host device as
            // `info::device_type::host`, which SYCL 2020's
            // `cpu_selector_v` *can* reject (host-device is deprecated
            // in 2020). And a custom selector lambda does too on the
            // 25.10 headers. Bypass selectors and take the first device
            // visible under whatever ACPP_VISIBILITY_MASK is in effect —
            // when limited to omp, that's the OMP host device by
            // construction. When CPU + GPU are both visible, set the
            // mask to "omp" before invoking to disambiguate.
            auto devs = sycl::device::get_devices();
            if (devs.empty()) {
                throw std::runtime_error(
                    "sycl_backend::queue (CPU): no SYCL devices visible. "
                    "Set ACPP_VISIBILITY_MASK=omp to expose AdaptiveCpp's "
                    "OpenMP backend.");
            }
            q = std::make_unique<sycl::queue>(devs.front(),
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
        validate_kernel_dispatch(*q);
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
