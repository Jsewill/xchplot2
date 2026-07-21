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
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
// Async errors are also COUNTED, not just logged. A backend that fails every
// kernel launch and one that runs every kernel but miscompiles it both end at
// the same symptom — a phase that produced zero entries — with completely
// different causes and completely different fixes. The count is what tells
// them apart, so the diagnostic can stop guessing. See validate_t1_count().
struct AsyncErrorLog {
    std::atomic<unsigned> count{0};
    std::mutex            mu;
    std::string           first;   // guarded by mu
};

inline AsyncErrorLog& async_errors()
{
    static AsyncErrorLog log;
    return log;
}

inline void async_error_handler(sycl::exception_list exns) noexcept
{
    auto record = [](char const* what) noexcept {
        auto& log = async_errors();
        log.count.fetch_add(1, std::memory_order_relaxed);
        try {
            std::lock_guard<std::mutex> lk(log.mu);
            if (log.first.empty()) log.first = what;
        } catch (...) {
            // Diagnostics must never be the thing that kills the run.
        }
    };
    for (std::exception_ptr const& ep : exns) {
        try { std::rethrow_exception(ep); }
        catch (sycl::exception const& e) {
            std::fprintf(stderr, "[sycl async] %s\n", e.what());
            record(e.what());
        }
        catch (std::exception const& e) {
            std::fprintf(stderr, "[sycl async] %s\n", e.what());
            record(e.what());
        }
        catch (...) {
            std::fprintf(stderr, "[sycl async] (unknown exception type)\n");
            record("(unknown exception type)");
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
    thread_local std::unique_ptr<sycl::queue> q;
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
    thread_local std::vector<std::pair<sycl::device, uint32_t*>> cache;
    sycl::device const dev = q.get_device();
    for (auto const& [d, ptr] : cache) {
        if (d == dev) return ptr;
    }

    std::vector<uint32_t> sT_host(4 * 256);
    for (int i = 0; i < 256; ++i) {
        sT_host[0 * 256 + i] = pos2gpu::aes_tables::T0[i];
        sT_host[1 * 256 + i] = pos2gpu::aes_tables::T1[i];
        sT_host[2 * 256 + i] = pos2gpu::aes_tables::T2[i];
        sT_host[3 * 256 + i] = pos2gpu::aes_tables::T3[i];
    }
    uint32_t* d_tables = sycl::malloc_device<uint32_t>(4 * 256, q);
    q.memcpy(d_tables, sT_host.data(), sizeof(uint32_t) * 4 * 256).wait();
    cache.emplace_back(dev, d_tables);
    return d_tables;
}

// ---- Two-phase match candidate-scratch budget --------------------------
//
// The T2/T3 two-phase match trades VRAM for speed: it stages {l, r}
// candidate index pairs in a device buffer before running the pairing AES
// over them. That buffer is sized from the plot's capacity, not from the
// tier's memory budget — in
//
//     cand_cap = (out_capacity / num_buckets_in_range) * M
//
// both terms scale with the tier's slicing factor N, so N cancels and the
// scratch lands at the same size in every tier. It was allocated with a
// raw sycl::malloc_device, so it never appeared in the streaming-stats
// trace that the tier peak models were calibrated against (see the anchor
// comments in GpuBufferPool.cpp, which say so outright). The models
// therefore under-predicted the true peak by ~3.1 GB at k=28, and the tier
// picker handed memory-constrained cards a tier that could not fit: a
// 7.6 GB Tesla P4 OOM'd on every tier except tiny, having been told
// minimal needed 3.67 GiB when it really needed 7.74.
//
// The budget makes the trade explicit. The pipeline declares how many
// bytes of candidate scratch this queue may hold — free VRAM minus the
// tier's modelled peak minus the safety margin — and the match falls back
// to the single-kernel path when the scratch will not fit. That path is
// correct for any input and allocates nothing, so the worst case is
// slower, never wrong and never OOM. A budget of 0 disables two-phase.
//
// Because the grant is (free - peak - margin), the arithmetic is safe by
// construction: peak + scratch + margin <= free, for any tier and any k.
// ONE scratch buffer, shared by the T2 and T3 two-phase match.
//
// T2 and T3 are never live at the same time — T2 match, T2 sort, then T3
// match, strictly in that order on one queue — but each used to keep its own
// per-queue cache, so both sat resident for the whole plot and the peak
// carried their SUM (2080 + 1040 = 3120 MiB at k=28 before right-sizing).
// Sharing one buffer, grown to whichever phase asks for more, makes the peak
// the MAX instead: it halves the scratch at zero cost, because the loser of
// the max simply reuses a buffer that is already big enough.
//
// Growing frees the old buffer before allocating the new one, so there is no
// moment where both are held. That is safe because run_t{2,3}_match_twophase
// ends with q.wait(): nothing in flight still references the old pointer.
struct TwoPhaseScratch {
    uint32_t* d_cand_l     = nullptr;
    uint32_t* d_cand_r     = nullptr;
    uint64_t* d_cand_count = nullptr;
    uint32_t* d_overflow   = nullptr;  // sticky flag: a bucket blew cand_cap
    uint64_t  cap          = 0;
    uint64_t  bytes        = 0;        // device bytes currently held
};

struct TwoPhaseState {
    uint64_t        limit = 0;   // bytes the pipeline permits on this queue
    TwoPhaseScratch scratch;
};

inline std::mutex& twophase_mutex()
{
    static std::mutex mu;
    return mu;
}

inline std::unordered_map<sycl::queue*, TwoPhaseState>& twophase_map()
{
    static std::unordered_map<sycl::queue*, TwoPhaseState> m;
    return m;
}

inline void set_twophase_budget(sycl::queue& q, uint64_t bytes)
{
    std::lock_guard<std::mutex> lk(twophase_mutex());
    twophase_map()[&q].limit = bytes;
}

inline void free_twophase_scratch(TwoPhaseScratch& s, sycl::queue& q)
{
    if (s.d_cand_l)     { sycl::free(s.d_cand_l, q);     s.d_cand_l     = nullptr; }
    if (s.d_cand_r)     { sycl::free(s.d_cand_r, q);     s.d_cand_r     = nullptr; }
    if (s.d_cand_count) { sycl::free(s.d_cand_count, q); s.d_cand_count = nullptr; }
    if (s.d_overflow)   { sycl::free(s.d_overflow, q);   s.d_overflow   = nullptr; }
    s.cap   = 0;
    s.bytes = 0;
}

// Returns nullptr when the scratch will not fit this queue's grant, or when
// the allocation fails outright. Neither is fatal: the caller rewinds the
// output counter and runs the single-kernel path, which allocates nothing and
// is correct for any input. The old code threw on a failed malloc, which is
// how a Tesla P4 with a perfectly valid plot configuration died with
// "T3 two-phase: candidate buffer alloc failed" instead of quietly taking the
// slower path.
inline TwoPhaseScratch* acquire_twophase_scratch(sycl::queue& q,
                                                 uint64_t cand_cap)
{
    std::lock_guard<std::mutex> lk(twophase_mutex());
    auto& st = twophase_map()[&q];
    auto& s  = st.scratch;
    if (s.cap >= cand_cap) return &s;   // already big enough — the shared win

    uint64_t const want = cand_cap * (sizeof(uint32_t) * 2)
                        + sizeof(uint64_t) + sizeof(uint32_t);
    if (want > st.limit) return nullptr;

    free_twophase_scratch(s, q);
    try {
        s.d_cand_l     = sycl::malloc_device<uint32_t>(cand_cap, q);
        s.d_cand_r     = sycl::malloc_device<uint32_t>(cand_cap, q);
        s.d_cand_count = sycl::malloc_device<uint64_t>(1, q);
        s.d_overflow   = sycl::malloc_device<uint32_t>(1, q);
    } catch (sycl::exception const&) {
        // Handled by the null check below — AdaptiveCpp returns nullptr on an
        // OOM, but throws for other allocation errors.
    }
    if (!s.d_cand_l || !s.d_cand_r || !s.d_cand_count || !s.d_overflow) {
        free_twophase_scratch(s, q);
        return nullptr;
    }
    s.cap   = cand_cap;
    s.bytes = want;
    return &s;
}

// Bytes the two-phase scratch currently holds on this queue. The bench VRAM
// watchdog uses this to separate the tier's modelled footprint from the
// optional scratch sitting on top of it — so it can tell "this tier is over
// budget" (a bug) apart from "we deliberately spent spare VRAM on the fast
// path" (working as intended).
inline uint64_t twophase_bytes_held(sycl::queue& q)
{
    std::lock_guard<std::mutex> lk(twophase_mutex());
    return twophase_map()[&q].scratch.bytes;
}

} // namespace pos2gpu::sycl_backend
