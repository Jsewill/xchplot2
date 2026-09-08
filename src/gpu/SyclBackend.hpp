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
#include "gpu/AsyncErrorLog.hpp"
#include "gpu/DeviceIds.hpp"

// cuda_fp16.h must precede sycl/sycl.hpp when this header is consumed
// from an nvcc TU — AdaptiveCpp's libkernel/detail/half_representation.hpp
// references __half, which only exists once cuda_fp16 has been seen.
#include "gpu/CudaHalfShim.hpp"
#include <sycl/sycl.hpp>
#include "gpu/TwoPhaseScratch.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
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
// Async errors are also COUNTED, not just logged — see AsyncErrorLog.hpp for
// why. This is the only place that writes the log.
inline void async_error_handler(sycl::exception_list exns) noexcept
{
    for (std::exception_ptr const& ep : exns) {
        try { std::rethrow_exception(ep); }
        catch (sycl::exception const& e) {
            std::fprintf(stderr, "[sycl async] %s\n", e.what());
            record_async_error(e.what());
        }
        catch (std::exception const& e) {
            std::fprintf(stderr, "[sycl async] %s\n", e.what());
            record_async_error(e.what());
        }
        catch (...) {
            std::fprintf(stderr, "[sycl async] (unknown exception type)\n");
            record_async_error("(unknown exception type)");
        }
    }
}

// Per-thread target device id. A worker thread sets this once at startup
// via set_current_device_id() so that its subsequent queue() call returns
// a queue bound to the requested device. Sentinel values:
//   kDefaultGpuId (-1)  : sycl::gpu_selector_v (single-device default,
//                         pre-multi-GPU zero-config path)
//   kCpuDeviceId  (-2)  : AdaptiveCpp's OpenMP host device. NOT latent — it is
//                         live behind XCHPLOT2_SYCL_CPU_BENCH=1, which A/Bs our
//                         SYCL kernels on the CPU against pos2-chip's hand-tuned
//                         CPU plotter. Production --cpu / --devices cpu plotting
//                         bypasses this and dispatches straight to
//                         run_one_plot_cpu() (see CpuPlotter.cpp), because
//                         pos2-chip wins that A/B by 4.3x at k=22 on a 5950X
//                         (0.95 s/plot vs 4.10): our kernels are GPU kernels —
//                         written for tens of thousands of threads and coalesced
//                         access — and they do not transpose to 32 CPU threads.
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

// Every SYCL GPU device this process can see. Used by --devices N to
// translate the user's index into a sycl::device, and by --devices all
// to spawn a worker per device.
//
// Used to filter non-CUDA backends out when the CUB sort path was
// linked, on the theory that a worker landing on an AMD device with
// CUB-only sort would just die mid-pipeline. The runtime backend
// dispatch in SortDispatch.cpp made that filter unnecessary — a hybrid
// host (NVIDIA + AMD) can now run a worker per device, with each
// worker picking the right sort backend at queue construction time.
inline std::vector<sycl::device> usable_gpu_devices()
{
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    return devs;
}

// Minimum max_compute_units for a GPU to be eligible for AUTOMATIC dispatch
// (--devices gpu/all and the zero-config default). An integrated GPU -- an AMD
// APU's 1-CU Raphael iGPU, say -- sits far below any discrete card (an Arc A310
// is 6 Xe-cores, an RX 6400 is 12 CUs, the Arc B580 is 160). Auto-dispatching a
// plot to one crawls, and the tier picker would size buffers against the system
// RAM the iGPU presents as VRAM. Overridable via XCHPLOT2_MIN_GPU_CUS; an
// explicit --devices <index> bypasses the filter, so such a device stays
// targetable on purpose.
inline int min_auto_dispatch_cus()
{
    if (char const* e = std::getenv("XCHPLOT2_MIN_GPU_CUS")) {
        int const v = std::atoi(e);
        if (v >= 0) return v;
    }
    return 4;
}

inline bool is_auto_dispatchable(sycl::device const& d)
{
    auto const cus = d.get_info<sycl::info::device::max_compute_units>();
    return static_cast<int>(cus) >= min_auto_dispatch_cus();
}

// Indices into usable_gpu_devices() eligible for automatic dispatch -- the tiny
// integrated GPUs filtered out. Falls back to EVERY device if the filter would
// leave nothing, so a box whose only GPU is an iGPU can still plot on it.
// Callers that expand "gpu"/"all" or pick the default GPU use this; the explicit
// --devices <index> path indexes usable_gpu_devices() directly, so its indices
// stay stable and any device stays targetable.
inline std::vector<int> auto_dispatchable_indices()
{
    auto const devs = usable_gpu_devices();
    std::vector<int> out;
    for (int i = 0; i < static_cast<int>(devs.size()); ++i)
        if (is_auto_dispatchable(devs[i])) out.push_back(i);
    if (out.empty())
        for (int i = 0; i < static_cast<int>(devs.size()); ++i) out.push_back(i);
    return out;
}

// The single GPU an implicit (zero-config) run should use: the BIGGEST
// auto-dispatchable one by compute units, ties going to the lowest index.
//
// Deliberately not "the first dispatchable index". min_auto_dispatch_cus() is a
// threshold, and a threshold can be wrong for a device nobody has tested --
// report an iGPU that claims 4+ CUs and the filter waves it through, so "first"
// would hand every zero-config plot to an integrated GPU sitting at index 0
// ahead of a discrete card. Picking the largest cannot make that mistake: with
// no flags, the biggest visible GPU is what the user meant, and on a mixed box
// the discrete card wins by one to two orders of magnitude (an AMD Raphael iGPU
// is 1-2 CUs against an Arc B580's 160). The threshold still does its job for
// --devices gpu/all, where every device is a candidate rather than just one.
inline int default_dispatch_index()
{
    auto const devs = usable_gpu_devices();
    auto const idx  = auto_dispatchable_indices();
    if (idx.empty()) return 0;
    int      best    = idx.front();
    unsigned best_cu = 0;
    for (int i : idx) {
        auto const cu = devs[static_cast<std::size_t>(i)]
                            .get_info<sycl::info::device::max_compute_units>();
        if (cu > best_cu) { best_cu = cu; best = i; }
    }
    return best;
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
// Forward-declared type used purely as the SYCL kernel name for the
// selftest body in SyclBackend.cpp. AdaptiveCpp uses the type identity
// to key kernel-launcher registration; the class is never defined.
class selftest_dispatch_kernel;

// Body in SyclBackend.cpp — must NOT be inline in this header, because
// AdaptiveCpp's SSCP IR pass runs per-TU and an inline parallel_for in
// a header included by every SYCL TU produces a duplicate HCF entry per
// TU (same kernel name, different HCF object IDs). At runtime the SSCP
// dispatch then fell through kernel_launcher.hpp:119 with "No kernel
// launcher is present for requested backend".
void validate_kernel_dispatch(sycl::queue& q);

inline sycl::queue& queue()
{
    struct QueueOwner {
        std::unordered_map<int, std::unique_ptr<sycl::queue>> queues;
        ~QueueOwner() {
            for (auto const& [id, q] : queues) {
                if (!q) continue;
                try { release_twophase_scratch(*q); }
                catch (std::exception const& e) {
                    std::fprintf(stderr, "two-phase scratch cleanup failed: %s\n", e.what());
                }
            }
        }
    };
    thread_local QueueOwner owner;
    // A coordinator can bind several GPUs on this thread. Keep each queue
    // alive until thread exit so buffers never migrate to a different device.
    auto& q = owner.queues[current_device_id()];
    if (!q) {
        int const id = current_device_id();
        if (is_cpu_device(id)) {
            // AdaptiveCpp's OpenMP backend exposes its host device as
            // `info::device_type::host`, which SYCL 2020's `cpu_selector_v`
            // *can* reject (host-device is deprecated in 2020), and a custom
            // selector lambda does too on the 25.10 headers. So we bypass
            // selectors and pick out of get_devices() by hand.
            //
            // This used to take devs.front() and rely on the caller having set
            // ACPP_VISIBILITY_MASK=omp. Nothing in the tree ever set it. With
            // the CUDA backend live, devs.front() is the GPU — so asking for
            // the CPU device silently handed back the GPU, and the whole
            // pipeline ran there while every log line still said "[batch:cpu]".
            // XCHPLOT2_SYCL_CPU_BENCH=1 reported 8.05 s/plot at k=28 for a
            // "CPU" that was an RTX 4090. A benchmark that can hand you the
            // wrong device is worse than one that refuses to run, so:
            // *never* return a GPU here. Filter for a non-GPU device and throw
            // if there isn't one.
            //
            // Accept cpu OR host device_type (AdaptiveCpp has used both across
            // versions), and reject accelerators — an FPGA/other offload device
            // is no more "the CPU" than a GPU is.
            auto devs = sycl::device::get_devices();
            sycl::device const* host_dev = nullptr;
            for (auto const& d : devs) {
                if (!d.is_gpu() && !d.is_accelerator()) { host_dev = &d; break; }
            }
            if (!host_dev) {
                throw std::runtime_error(
                    "sycl_backend::queue (CPU): no CPU/host SYCL device visible"
                    " — this build sees " + std::to_string(devs.size()) +
                    " device(s), all GPU/accelerator. Refusing to fall back to"
                    " a GPU and report it as the CPU. Build AdaptiveCpp with"
                    " the OpenMP backend (ACPP_TARGETS must include omp), or"
                    " run with ACPP_VISIBILITY_MASK=omp.");
            }
            q = std::make_unique<sycl::queue>(*host_dev, async_error_handler);
        } else if (id < 0) {
            // Default GPU: prefer an auto-dispatchable card so a zero-config run
            // skips a tiny iGPU, but never reject outright -- an iGPU-only box
            // must still land somewhere. Higher score wins; among peers, more
            // compute units. (SYCL's gpu_selector_v gives no such guarantee.)
            auto const default_gpu_scorer = [](sycl::device const& d) -> int {
                if (!d.is_gpu()) return -1;
                int const cus = static_cast<int>(
                    d.get_info<sycl::info::device::max_compute_units>());
                return (is_auto_dispatchable(d) ? 1000000 : 1) + cus;
            };
            q = std::make_unique<sycl::queue>(default_gpu_scorer,
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
// alive for the process lifetime — mirrors the CUDA path's
// __constant__ T-tables. The cache is keyed by the queue's device, NOT
// per-thread: the multi-GPU shard pipeline drives several queues (one
// per device) from a single thread, so a plain thread_local pointer
// would hand shards 1..N-1 a pointer allocated on shard 0's device —
// an illegal cross-device access. Two threads racing on the same
// device at worst upload a duplicate 4 KiB table; each caller still
// gets a pointer valid for its own device.
//
// Pointer layout matches what the _smem family expects: [T0|T1|T2|T3],
// 256 entries each.
inline uint32_t* aes_tables_device(sycl::queue& q)
{
    thread_local std::vector<std::tuple<sycl::device, sycl::context, std::shared_ptr<uint32_t>>> cache;
    auto const dev = q.get_device();
    auto const context = q.get_context();
    for (auto const& [d, ctx, ptr] : cache) {
        if (d == dev && ctx == context) return ptr.get();
    }

    std::vector<uint32_t> sT_host(4 * 256);
    for (int i = 0; i < 256; ++i) {
        sT_host[0 * 256 + i] = pos2gpu::aes_tables::T0[i];
        sT_host[1 * 256 + i] = pos2gpu::aes_tables::T1[i];
        sT_host[2 * 256 + i] = pos2gpu::aes_tables::T2[i];
        sT_host[3 * 256 + i] = pos2gpu::aes_tables::T3[i];
    }
    std::shared_ptr<uint32_t> tables(sycl::malloc_device<uint32_t>(4 * 256, q),
        [context](uint32_t* p) {
            try { if (p) sycl::free(p, context); }
            catch (std::exception const& e) { record_async_error(e.what()); }
        });
    if (!tables) throw std::bad_alloc();
    q.memcpy(tables.get(), sT_host.data(), sizeof(uint32_t) * 4 * 256).wait();
    cache.emplace_back(dev, context, tables);
    return tables.get();
}


} // namespace pos2gpu::sycl_backend
