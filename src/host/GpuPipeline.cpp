// GpuPipeline.cu — orchestrates Xs → T1 → T2 → T3 on the device, with
// CUB radix sort between phases (each phase consumes sorted-by-match_info
// input). Final T3 output is sorted by proof_fragment (low 2k bits) to
// match pos2-chip Table3Constructor::post_construct_span.
//
// Two overloads live here:
//   run_gpu_pipeline(cfg)       — transient pool, one-shot.
//   run_gpu_pipeline(cfg, pool) — shared pool, batch-friendly. This is the
//                                 real implementation; the one-shot form
//                                 just wraps it in a temporary pool.

#include "host/GpuPipeline.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/PoolSizing.hpp"

#include "gpu/AesGpu.cuh"
#include "gpu/XsKernel.cuh"
#include "gpu/XsKernels.cuh"   // launch_xs_gen / launch_xs_pack (stage 4e)
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"
#include "gpu/PipelineKernels.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>


#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pos2gpu {

namespace {


// =====================================================================
// T1 sort: by match_info, low k bits, stable. Uses CUB SortPairs with
// (key=match_info, value=index) then permutes T1Pairings.
// =====================================================================
// T2 sort: same shape — sort indices by match_info.
// =====================================================================
// Streaming allocation tracker.
//
// Wraps cudaMalloc / cudaFree so we can: (a) account for live/peak VRAM
// used by the streaming pipeline, (b) honour a soft device-memory cap
// set via POS2GPU_MAX_VRAM_MB (throws before the underlying cudaMalloc
// when an alloc would push live past the cap), and (c) emit a per-alloc
// trace under POS2GPU_STREAMING_STATS=1 for manual audits.
//
// Pinned host allocations are NOT counted — the cap is specifically for
// device VRAM, and the pinned D2H staging buffer is host-resident.
// =====================================================================
struct StreamingStats {
    size_t cap  = 0;   // 0 = no cap
    size_t live = 0;
    size_t peak = 0;
    std::unordered_map<void*, size_t> sizes;
    bool        verbose = false;
    char const* phase   = "(init)";

    // Free any allocations still alive on destruction. If the streaming
    // pipeline throws partway (e.g. d_xs_temp OOM after d_xs already
    // succeeded), this dtor releases the still-live device buffers
    // instead of leaking them across batch iterations.
    ~StreamingStats() {
        if (sizes.empty()) return;
        auto& q = sycl_backend::queue();
        for (auto& [ptr, _bytes] : sizes) {
            if (ptr) sycl::free(ptr, q);
        }
        sizes.clear();
    }
};

inline void s_init_from_env(StreamingStats& s)
{
    if (char const* v = std::getenv("POS2GPU_MAX_VRAM_MB"); v && v[0]) {
        s.cap = size_t(std::strtoull(v, nullptr, 10)) * (1ULL << 20);
    }
    if (char const* v = std::getenv("POS2GPU_STREAMING_STATS"); v && v[0] == '1') {
        s.verbose = true;
    }
}

template <typename T>
inline void s_malloc(StreamingStats& s, T*& out, size_t bytes, char const* reason)
{
    if (s.cap && s.live + bytes > s.cap) {
        throw std::runtime_error(
            std::string("streaming VRAM cap: phase=") + s.phase +
            " alloc=" + reason +
            " live=" + std::to_string(s.live >> 20) +
            " + new="  + std::to_string(bytes  >> 20) +
            " would exceed cap=" + std::to_string(s.cap >> 20) + " MB");
    }
    void* p = sycl::malloc_device(bytes, sycl_backend::queue());
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_device(") + reason + "): null — phase=" +
            s.phase + " requested=" + std::to_string(bytes >> 20) +
            " MB live=" + std::to_string(s.live >> 20) +
            " MB. Card likely too small for this k via the streaming "
            "pipeline; try a smaller k or a card with more VRAM.");
    }
    out = static_cast<T*>(p);
    s.live += bytes;
    if (s.live > s.peak) s.peak = s.live;
    s.sizes[p] = bytes;
    if (s.verbose) {
        std::fprintf(stderr,
            "[stream %-8s] +%7.2f MB  %-20s  live=%8.2f  peak=%8.2f\n",
            s.phase, bytes / 1048576.0, reason,
            s.live / 1048576.0, s.peak / 1048576.0);
    }
}

template <typename T>
inline void s_free(StreamingStats& s, T*& ptr)
{
    if (!ptr) return;
    void* raw = static_cast<void*>(ptr);
    auto it = s.sizes.find(raw);
    if (it != s.sizes.end()) {
        s.live -= it->second;
        if (s.verbose) {
            std::fprintf(stderr,
                "[stream %-8s] -%7.2f MB  %-20s  live=%8.2f  peak=%8.2f\n",
                s.phase, it->second / 1048576.0, "(free)",
                s.live / 1048576.0, s.peak / 1048576.0);
        }
        s.sizes.erase(it);
    }
    sycl::free(raw, sycl_backend::queue());
    ptr = nullptr;
}

} // namespace

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool,
                                   int pinned_index)
{

    sycl::queue& q = sycl_backend::queue();
    if (cfg.k < 18 || cfg.k > 32 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 32]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }
    if (pool.k != cfg.k || pool.strength != cfg.strength
        || pool.testnet != cfg.testnet)
    {
        throw std::runtime_error(
            "GpuBufferPool was sized for different (k, strength, testnet)");
    }
    if (pinned_index < 0 || pinned_index >= GpuBufferPool::kNumPinnedBuffers) {
        throw std::runtime_error(
            "pinned_index must be in [0, GpuBufferPool::kNumPinnedBuffers)");
    }

    uint64_t const total_xs = pool.total_xs;
    uint64_t const cap      = pool.cap;

    constexpr int kThreads = 256;
    auto blocks = [&](uint64_t n) {
        return unsigned((n + kThreads - 1) / kThreads);
    };

    // ---- pool aliases ----
    // d_pair_a carries the "current phase match output": T1, then T2, then T3.
    // d_pair_b carries the "current phase sort output": sorted T1, sorted T2,
    // then final uint64_t fragments. Each subsequent phase's output overwrites
    // the previous (consumed) contents in the same slot.
    XsCandidateGpu* d_xs             = static_cast<XsCandidateGpu*>(pool.d_storage);
    // d_pair_a-derived aliases (d_t1_meta, d_t1_mi, d_t2_meta, d_t2_mi,
    // d_t2_xbits, d_t3) are NOT declared here. They're declared inside
    // the Xs phase block below, right after pool.ensure_pair_a()
    // performs the lazy malloc_device for d_pair_a. Deferring that
    // alloc until after Xs gen has been submitted to the queue lets
    // the ~400-500 ms CPU-side malloc_device overlap with Xs's
    // ~750 ms GPU execution — saves ~400-500 ms off first-plot wall;
    // batch plots 2+ hit ensure_pair_a's cached-pointer fast path
    // so the alloc cost is paid exactly once per pool.
    //
    // d_pair_b-derived aliases stay up here because d_pair_b is
    // eager-allocated by the pool ctor: Xs gen needs it as scratch
    // from the start of the pipeline.
    uint64_t*       d_t1_meta_sorted  = static_cast<uint64_t*>      (pool.d_pair_b);
    uint64_t*       d_t2_meta_sorted  = static_cast<uint64_t*>      (pool.d_pair_b);
    uint32_t*       d_t2_xbits_sorted = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(pool.d_pair_b) + pool.cap * sizeof(uint64_t));
    uint64_t*       d_frags_out       = static_cast<uint64_t*>      (pool.d_pair_b);

    uint64_t*       d_count        = pool.d_counter;
    // Xs phase needs ~3.22 GB scratch at k=28 in split-keys_a mode
    // (3 × total_xs × u32 + cub); d_pair_b is idle through the whole
    // Xs phase (not touched until T1 sort permute writes to it), so
    // we alias it rather than allocating separately.
    //
    // Split-keys_a: the Xs sort's keys_a (total_xs · u32 = 1 GiB at
    // k=28) lives in d_storage's tail — bytes [total_xs·8, storage_bytes)
    // which is idle during Xs gen+sort. The final pack phase writes
    // d_storage[0..total_xs·8) only, leaving keys_a's memory region
    // undisturbed (and its contents unread after the sort anyway, so
    // the overlap on T1/T2/T3-sort aliases in d_storage after pack is
    // a pure write-without-read of stale bytes). Saves ~1 GiB off the
    // pair_b xs-scratch region — see GpuBufferPool.cpp for sizing.
    void* const d_xs_split_keys_a = static_cast<uint8_t*>(pool.d_storage)
                                    + pool.total_xs * sizeof(XsCandidateGpu);
    void*           d_xs_temp      = pool.d_pair_b;
    void*           d_sort_scratch = pool.d_sort_scratch;
    // Lazy pinned-host alloc: skips ~600 ms × (kNumPinnedBuffers-1)
    // on single-plot runs (only slot 0 gets allocated). See
    // GpuBufferPool::ensure_pinned header comment for rationale.
    uint64_t*       h_pinned_t3    = pool.ensure_pinned(pinned_index);
    // T1/T2/T3 match kernels report 0 scratch bytes, but some CUDA paths
    // reject a nullptr d_temp_storage with cudaErrorInvalidArgument even
    // when bytes==0. Point them at d_sort_scratch (idle during match) to
    // give the kernel a valid non-null handle.
    void*           d_match_temp   = pool.d_sort_scratch;

    // Sort key/val arrays alias d_storage. Safe because Xs is fully consumed
    // by T1 match (stream-synchronised) before we enter T1 sort.
    //
    // Only three slots live here — keys_out, vals_in, vals_out. The
    // sort's keys_input is always the SoA match-info stream from
    // d_pair_a (d_t1_mi / d_t2_mi), so the fourth slot that would
    // have hosted "d_keys_in" is neither allocated nor used. See
    // GpuBufferPool.cpp for the matching storage_bytes shrink.
    auto     storage_u32 = static_cast<uint32_t*>(pool.d_storage);
    uint32_t* d_keys_out = storage_u32 + 0 * cap;
    uint32_t* d_vals_in  = storage_u32 + 1 * cap;
    uint32_t* d_vals_out = storage_u32 + 2 * cap;

    // ---- per-phase wall-time profiling ----
    // Enabled when either cfg.profile is set (xchplot2 -P / --profile) or
    // POS2GPU_PHASE_TIMING=1 is in the env. Each phase's wall is measured
    // around q.wait()s so launches actually drain to the device before the
    // next start sample — adds a sync point but gives an honest breakdown.
    // When disabled, begin/end/report are early-out and add ~zero cost.
    bool const phase_timing = cfg.profile || [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING");
        return v && v[0] == '1';
    }();
    using phase_clock = std::chrono::steady_clock;
    std::vector<std::pair<char const*, phase_clock::time_point>> phase_starts;
    std::vector<std::pair<char const*, double>>                  phase_records;
    auto begin_phase = [&](char const* label) -> int {
        if (!phase_timing) return -1;
        q.wait();
        phase_starts.emplace_back(label, phase_clock::now());
        return static_cast<int>(phase_starts.size() - 1);
    };
    auto end_phase = [&](int idx) {
        if (idx < 0) return;
        q.wait();
        auto const t1 = phase_clock::now();
        auto const& [name, t0] = phase_starts[idx];
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        phase_records.emplace_back(name, ms);
    };
    auto report_phases = [&]() {
        if (!phase_timing || phase_records.empty()) return;
        double total = 0.0;
        for (auto const& [_n, ms] : phase_records) total += ms;
        std::fprintf(stderr, "[phase-timing]");
        for (auto const& [name, ms] : phase_records) {
            std::fprintf(stderr, " %s=%.1fms(%.0f%%)",
                name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        }
        std::fprintf(stderr, " total=%.1fms\n", total);
    };

    // ---------- Phase Xs ----------
    size_t xs_temp_bytes = 0;
    launch_construct_xs(cfg.plot_id.data(), cfg.k, cfg.testnet,
                              nullptr, nullptr, &xs_temp_bytes, q,
                              d_xs_split_keys_a);
    int p_xs = begin_phase("Xs gen+sort");
    // Xs phase events stubbed in slice 17b — pass nullptr for the (no-op)
    // profiling event slots. The launch_construct_xs_profiled signature still
    // accepts cudaEvent_t for API compatibility but ignores the values.
    launch_construct_xs_profiled(cfg.plot_id.data(), cfg.k, cfg.testnet,
                                       d_xs, d_xs_temp, &xs_temp_bytes,
                                       nullptr, nullptr, q,
                                       d_xs_split_keys_a);
    // Overlap d_pair_a's lazy malloc_device (~400-500 ms for 4.36 GB at
    // k=28) with Xs gen's GPU execution. In production
    // (POS2GPU_PHASE_TIMING unset), launch_construct_xs_profiled returns
    // immediately with the kernel in-flight on the queue; this CPU-side
    // alloc then runs in parallel and its wall is hidden behind Xs's
    // ~750 ms GPU work. In phase_timing mode xs-timing's internal
    // q.waits serialise Xs first, then this alloc pays full wall — a
    // diagnostic-mode trade-off.
    void* const d_pair_a_raw = pool.ensure_pair_a();
    end_phase(p_xs);

    // d_pair_a-derived aliases, now that the lazy alloc has resolved.
    // Same layout as the old eager version — just computed from the
    // local d_pair_a_raw instead of pool.d_pair_a so there's no
    // confusion about when the pointer became valid.
    //
    // T1 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B) then mi[cap] (cap·4 B). Total cap·12 B, fits in d_pair_a's
    // cap·16 B budget.
    uint64_t*     d_t1_meta = static_cast<uint64_t*>(d_pair_a_raw);
    uint32_t*     d_t1_mi   = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * sizeof(uint64_t));
    // T2 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B), then mi[cap] (cap·4 B), then xbits[cap] (cap·4 B). Total
    // cap·16 B, matching d_pair_a's size.
    uint64_t*     d_t2_meta  = static_cast<uint64_t*>(d_pair_a_raw);
    uint32_t*     d_t2_mi    = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * sizeof(uint64_t));
    uint32_t*     d_t2_xbits = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * (sizeof(uint64_t) + sizeof(uint32_t)));
    T3PairingGpu* d_t3       = static_cast<T3PairingGpu*>(d_pair_a_raw);

    // ---------- Phase T1 ----------
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_count, cap,
                          nullptr, &t1_temp_bytes, q);
    q.memset(d_count, 0, sizeof(uint64_t));
    int p_t1 = begin_phase("T1 match");
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1_meta, d_t1_mi, d_count, cap,
                          d_match_temp, &t1_temp_bytes, q);
    end_phase(p_t1);

    // No explicit sync: the next cudaMemcpy (non-async, default stream)
    // implicitly drains prior stream work before the host reads t1_count.
    uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_count, sizeof(uint64_t)).wait();
    if (t1_count > cap) throw std::runtime_error("T1 overflow");


    // Sort T1 by match_info (low k bits). d_storage is now repurposed
    // as (keys_in, keys_out, vals_in, vals_out), Xs having been fully
    // consumed by T1 match above. T1 match emits match_info in a SoA
    // stream (d_t1_mi), so we feed that directly to CUB as the sort key
    // input rather than extracting from a packed struct.
    int p_t1_sort = begin_phase("T1 sort");
    {
        launch_init_u32_identity(d_vals_in, t1_count, q);
        size_t sort_bytes = pool.sort_scratch_bytes;
        launch_sort_pairs_u32_u32(
            d_sort_scratch, sort_bytes,
            d_t1_mi, d_keys_out, d_vals_in, d_vals_out,
            t1_count, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);

        launch_gather_u64(d_t1_meta, d_vals_out, d_t1_meta_sorted, t1_count, q);
    }
    end_phase(p_t1_sort);

    // ---------- Phase T2 ----------
    // Sorted T1 = (d_t1_meta_sorted: uint64 meta, d_keys_out: uint32 match_info).
    // No AoS struct anymore — saves 33 % of sorted-T1 bandwidth on both the
    // permute write and the match-kernel hot path.
    auto t2p = make_t2_params(cfg.k, cfg.strength);
    size_t t2_temp_bytes = 0;
    launch_t2_match(cfg.plot_id.data(), t2p, nullptr, nullptr, t1_count,
                          nullptr, nullptr, nullptr, d_count, cap,
                          nullptr, &t2_temp_bytes, q);
    q.memset(d_count, 0, sizeof(uint64_t));
    int p_t2 = begin_phase("T2 match");
    launch_t2_match(cfg.plot_id.data(), t2p, d_t1_meta_sorted, d_keys_out, t1_count,
                          d_t2_meta, d_t2_mi, d_t2_xbits, d_count, cap,
                          d_match_temp, &t2_temp_bytes, q);
    end_phase(p_t2);

    uint64_t t2_count = 0;
    q.memcpy(&t2_count, d_count, sizeof(uint64_t)).wait();
    if (t2_count > cap) throw std::runtime_error("T2 overflow");

    int p_t2_sort = begin_phase("T2 sort");
    {
        // T2 match emitted match_info as a SoA stream (d_t2_mi) — feed
        // it straight into CUB as the sort key input rather than
        // re-extracting from a packed struct. vals_in just needs a
        // 0..n-1 identity fill.
        launch_init_u32_identity(d_vals_in, t2_count, q);
        size_t sort_bytes = pool.sort_scratch_bytes;
        launch_sort_pairs_u32_u32(
            d_sort_scratch, sort_bytes,
            d_t2_mi, d_keys_out, d_vals_in, d_vals_out,
            t2_count, 0, cfg.k, q);

        launch_permute_t2(d_t2_meta, d_t2_xbits, d_vals_out,
                          d_t2_meta_sorted, d_t2_xbits_sorted, t2_count, q);
    }
    end_phase(p_t2_sort);

    // ---------- Phase T3 ----------
    // d_keys_out now holds the T2 sorted match_info (T1's was overwritten by
    // the T2 sort above) — pass as the slim stream for binary search in T3.
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          nullptr, t2_count,
                          d_t3, d_count, cap,
                          nullptr, &t3_temp_bytes, q);
    q.memset(d_count, 0, sizeof(uint64_t));
    int p_t3 = begin_phase("T3 match + Feistel");
    launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          d_keys_out, t2_count,
                          d_t3, d_count, cap,
                          d_match_temp, &t3_temp_bytes, q);
    end_phase(p_t3);

    uint64_t t3_count = 0;
    q.memcpy(&t3_count, d_count, sizeof(uint64_t)).wait();
    if (t3_count > cap) throw std::runtime_error("T3 overflow");

    // Sort T3 by proof_fragment (low 2k bits). T3PairingGpu is just a
    // uint64_t, so reinterpret the d_pair_a slot directly.
    uint64_t* d_frags_in = reinterpret_cast<uint64_t*>(d_t3);
    int p_t3_sort = begin_phase("T3 sort");
    {
        size_t sort_bytes = pool.sort_scratch_bytes;
        launch_sort_keys_u64(
            d_sort_scratch, sort_bytes,
            d_frags_in, d_frags_out,
            t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
    }
    end_phase(p_t3_sort);

    // ---------- D2H ----------
    int p_d2h = begin_phase("D2H copy T3 fragments (pinned)");
    GpuPipelineResult result;
    result.t1_count = t1_count;
    result.t2_count = t2_count;
    result.t3_count = t3_count;

    if (t3_count > 0) {
        q.memcpy(h_pinned_t3, d_frags_out, sizeof(uint64_t) * t3_count);
        q.wait();
    }
    end_phase(p_d2h);

    if (t3_count > 0) {
        // Borrow: caller (batch producer) promises to finish consuming this
        // pinned slot before reusing it for another plot.
        result.external_fragments_ptr   = h_pinned_t3;
        result.external_fragments_count = t3_count;
    }

    // Xs gen / sort per-phase timings stubbed in slice 17b — see profiling
    // notes above.

    // Release d_pair_a so it isn't held between plots in a batch run.
    // At ~5 ms/alloc on amdgcn (sycl::malloc_device effectively just
    // reserves virtual address space), the per-plot realloc cost is
    // below noise, but freeing 4.36 GB during the inter-plot gap means
    // the pool path is viable on cards with ~7-8 GiB free that would
    // otherwise hit InsufficientVramError and fall back to streaming.
    // The final q.wait() inside the D2H block above has already drained
    // T3 sort so the buffer is safe to free.
    pool.release_pair_a();

    report_phases();
    return result;
}

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg)
{
    // Explicit override for callers that want the streaming path without
    // having to rebuild anything. Handy for testing and for users who know
    // their hardware won't fit the pool.
    if (char const* env = std::getenv("XCHPLOT2_STREAMING");
        env && env[0] == '1')
    {
        return run_gpu_pipeline_streaming(cfg);
    }

    // Default: build a transient pool and run through it. Pays the full
    // per-call allocator overhead (~2.4 s for k=28) — batch callers should
    // construct a pool once and reuse it via the 3-arg overload.
    //
    // On insufficient device VRAM the pool ctor throws
    // InsufficientVramError; catch it specifically and fall back to
    // streaming so users on small-VRAM cards get a working plot with no
    // flags. Other CUDA errors propagate.
    try {
        GpuBufferPool pool(cfg.k, cfg.strength, cfg.testnet);
        GpuPipelineResult r = run_gpu_pipeline(cfg, pool, /*pinned_index=*/0);
        // Pool (and its pinned buffer) is about to be destroyed, so
        // materialise a self-contained copy before returning.
        if (r.external_fragments_ptr && r.external_fragments_count > 0) {
            r.t3_fragments_storage.resize(r.external_fragments_count);
            std::memcpy(r.t3_fragments_storage.data(),
                        r.external_fragments_ptr,
                        sizeof(uint64_t) * r.external_fragments_count);
        }
        r.external_fragments_ptr   = nullptr;
        r.external_fragments_count = 0;
        return r;
    } catch (InsufficientVramError const& e) {
        std::fprintf(stderr,
            "[xchplot2] pool needs %.2f GiB, only %.2f GiB free of "
            "%.2f GiB — falling back to streaming pipeline\n",
            e.required_bytes / double(1ULL << 30),
            e.free_bytes     / double(1ULL << 30),
            e.total_bytes    / double(1ULL << 30));
        return run_gpu_pipeline_streaming(cfg);
    }
}

// =====================================================================
// Streaming pipeline — per-phase cudaMalloc / cudaFree, no persistent pool.
//
// Only buffers required for the CURRENT and NEXT phase are resident at any
// point. Tiled sorts + SoA emission drive the peak down under 8 GB at
// k=28, so an 8 GB card can run this path.
//
// The implementation body below accepts an optional caller-provided
// pinned D2H buffer — used by BatchPlotter to amortise cudaMallocHost
// across plots and double-buffer the D2H with the FSE consumer.
//
// Exception safety: on throw mid-pipeline we currently leak the
// still-live device allocations. The CLI terminates on exception anyway,
// so the OS reclaims the context. If we later embed this in a long-lived
// process we can add RAII owners without changing the public surface.
// =====================================================================
namespace { // anon: shared impl, not part of the public API.

GpuPipelineResult run_gpu_pipeline_streaming_impl(
    GpuPipelineConfig const& cfg,
    uint64_t* pinned_dst,                       // nullable
    size_t    pinned_capacity,                  // count, not bytes; ignored if pinned_dst null
    StreamingPinnedScratch const& scratch);     // any field nullptr → per-plot malloc_host fallback

} // namespace

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg)
{

    sycl::queue& q = sycl_backend::queue();
    return run_gpu_pipeline_streaming_impl(cfg, /*pinned_dst=*/nullptr,
                                                /*pinned_capacity=*/0,
                                                StreamingPinnedScratch{});
}

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg,
                                             uint64_t* pinned_dst,
                                             size_t    pinned_capacity)
{
    if (!pinned_dst || pinned_capacity == 0) {
        throw std::runtime_error(
            "run_gpu_pipeline_streaming(cfg, pinned, cap): pinned buffer must be non-null");
    }
    return run_gpu_pipeline_streaming_impl(cfg, pinned_dst, pinned_capacity,
                                           StreamingPinnedScratch{});
}

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg,
                                             uint64_t* pinned_dst,
                                             size_t    pinned_capacity,
                                             StreamingPinnedScratch const& scratch)
{
    if (!pinned_dst || pinned_capacity == 0) {
        throw std::runtime_error(
            "run_gpu_pipeline_streaming(cfg, pinned, cap, scratch): pinned buffer must be non-null");
    }
    return run_gpu_pipeline_streaming_impl(cfg, pinned_dst, pinned_capacity, scratch);
}

namespace {

GpuPipelineResult run_gpu_pipeline_streaming_impl(
    GpuPipelineConfig const& cfg,
    uint64_t* pinned_dst,
    size_t    pinned_capacity,
    StreamingPinnedScratch const& scratch)
{

    sycl::queue& q = sycl_backend::queue();
    if (cfg.k < 18 || cfg.k > 32 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 32]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }

    int const num_section_bits = (cfg.k < 28) ? 2 : (cfg.k - 26);
    uint64_t const total_xs = 1ULL << cfg.k;
    uint64_t const cap =
        max_pairs_per_section(cfg.k, num_section_bits) *
        (1ULL << num_section_bits);

    constexpr int kThreads = 256;
    auto blocks = [&](uint64_t n) {
        return unsigned((n + kThreads - 1) / kThreads);
    };

    StreamingStats stats;
    s_init_from_env(stats);

    // ---- per-phase wall-time profiling ----
    // Identical shape to the pool path (run_gpu_pipeline above); the
    // [phase-timing] output format matches so POS2GPU_PHASE_TIMING=1 now
    // produces the same breakdown whether the pipeline runs pool or
    // falls back to streaming. On 12 GiB cards at k=28 (where pool
    // overflows and we always streams) this is the only way to see
    // which phase is eating the wall.
    bool const phase_timing = cfg.profile || [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING");
        return v && v[0] == '1';
    }();
    using phase_clock = std::chrono::steady_clock;
    std::vector<std::pair<char const*, phase_clock::time_point>> phase_starts;
    std::vector<std::pair<char const*, double>>                  phase_records;
    auto begin_phase = [&](char const* label) -> int {
        if (!phase_timing) return -1;
        q.wait();
        phase_starts.emplace_back(label, phase_clock::now());
        return static_cast<int>(phase_starts.size() - 1);
    };
    auto end_phase = [&](int idx) {
        if (idx < 0) return;
        q.wait();
        auto const t1 = phase_clock::now();
        auto const& [name, t0] = phase_starts[idx];
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        phase_records.emplace_back(name, ms);
    };
    auto report_phases = [&]() {
        if (!phase_timing || phase_records.empty()) return;
        double total = 0.0;
        for (auto const& [_n, ms] : phase_records) total += ms;
        std::fprintf(stderr, "[phase-timing]");
        for (auto const& [name, ms] : phase_records) {
            std::fprintf(stderr, " %s=%.1fms(%.0f%%)",
                name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        }
        std::fprintf(stderr, " total=%.1fms\n", total);
    };

    // --- pipeline-wide tiny allocations ---
    // d_counter: per-phase uint64 count output (reused).
    // The match kernels each need their own temp-storage buffer sized via
    // their size query; we allocate it per-phase rather than globally so
    // that the peak VRAM is the phase's alone.
    stats.phase = "init";
    uint64_t* d_counter = nullptr;
    s_malloc(stats, d_counter, sizeof(uint64_t), "d_counter");

    // ---------- Phase Xs (stage 4e: inlined gen+sort+pack) ----------
    // launch_construct_xs lumps keys_a/keys_b/vals_a/vals_b into a single
    // d_xs_temp blob (~4 GB at k=28). keys_a+vals_a are dead after the
    // CUB sort but can't be freed because they're interior slices of a
    // single allocation. Inline the three sub-kernels so we can:
    //   1. alloc cub_scratch + keys_a + vals_a
    //   2. gen fills keys_a, vals_a
    //   3. alloc keys_b + vals_b
    //   4. CUB sort keys_a/vals_a -> keys_b/vals_b; keys_a/vals_a now dead
    //   5. free cub_scratch + keys_a + vals_a       <- 2078 MB freed
    //   6. alloc d_xs
    //   7. pack keys_b/vals_b -> d_xs
    //   8. free keys_b + vals_b
    // Phase peak at k=28 drops from d_xs (2048) + d_xs_temp (4128) =
    // 6176 MB to max(sort 4126 MB, pack 4096 MB) = 4126 MB.
    stats.phase = "Xs";

    // Query CUB scratch size via the sort wrapper.
    size_t xs_cub_bytes = 0;
    launch_sort_pairs_u32_u32(
        nullptr, xs_cub_bytes,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        total_xs, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);

    void*     d_xs_cub_scratch = nullptr;
    uint32_t* d_xs_keys_a      = nullptr;
    uint32_t* d_xs_vals_a      = nullptr;
    s_malloc(stats, d_xs_cub_scratch, xs_cub_bytes,                     "d_xs_cub");
    s_malloc(stats, d_xs_keys_a,      total_xs * sizeof(uint32_t),      "d_xs_keys_a");
    s_malloc(stats, d_xs_vals_a,      total_xs * sizeof(uint32_t),      "d_xs_vals_a");

    AesHashKeys const xs_keys = make_keys(cfg.plot_id.data());
    uint32_t    const xs_xor_const = cfg.testnet ? 0xA3B1C4D7u : 0u;

    int p_xs = begin_phase("Xs gen+sort");
    launch_xs_gen(xs_keys, d_xs_keys_a, d_xs_vals_a, total_xs,
                  cfg.k, xs_xor_const, q);

    // keys_b + vals_b appear here — minimum Xs-phase live set between
    // gen and sort.
    uint32_t* d_xs_keys_b = nullptr;
    uint32_t* d_xs_vals_b = nullptr;
    s_malloc(stats, d_xs_keys_b, total_xs * sizeof(uint32_t), "d_xs_keys_b");
    s_malloc(stats, d_xs_vals_b, total_xs * sizeof(uint32_t), "d_xs_vals_b");

    launch_sort_pairs_u32_u32(
        d_xs_cub_scratch, xs_cub_bytes,
        d_xs_keys_a, d_xs_keys_b,
        d_xs_vals_a, d_xs_vals_b,
        total_xs, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
    end_phase(p_xs);

    // sort consumed keys_a + vals_a; free them and CUB scratch before
    // allocating d_xs so the pack phase peak stays under the sort peak.
    s_free(stats, d_xs_cub_scratch);
    s_free(stats, d_xs_keys_a);
    s_free(stats, d_xs_vals_a);

    XsCandidateGpu* d_xs = nullptr;
    s_malloc(stats, d_xs, total_xs * sizeof(XsCandidateGpu), "d_xs");

    int p_xs_pack = begin_phase("Xs pack");
    launch_xs_pack(d_xs_keys_b, d_xs_vals_b, d_xs, total_xs, q);
    end_phase(p_xs_pack);

    s_free(stats, d_xs_keys_b);
    s_free(stats, d_xs_vals_b);

    // ---------- Phase T1 match ----------
    stats.phase = "T1 match";
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_counter, cap,
                          nullptr, &t1_temp_bytes, q);
    // SoA output: meta (uint64) + mi (uint32). Same 12 B/pair as the old
    // AoS struct, but the two streams can be freed independently — we
    // drop d_t1_mi as soon as CUB consumes it in the T1 sort phase.
    uint64_t* d_t1_meta = nullptr;
    uint32_t* d_t1_mi   = nullptr;
    void*     d_t1_match_temp = nullptr;
    s_malloc(stats, d_t1_meta,        cap * sizeof(uint64_t), "d_t1_meta");
    s_malloc(stats, d_t1_mi,          cap * sizeof(uint32_t), "d_t1_mi");
    s_malloc(stats, d_t1_match_temp,  t1_temp_bytes,          "d_t1_match_temp");

    int p_t1 = begin_phase("T1 match");
    q.memset(d_counter, 0, sizeof(uint64_t));
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1_meta, d_t1_mi, d_counter, cap,
                          d_t1_match_temp, &t1_temp_bytes, q);
    end_phase(p_t1);

    uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_counter, sizeof(uint64_t)).wait();
    if (t1_count > cap) throw std::runtime_error("T1 overflow");

    s_free(stats, d_t1_match_temp);
    // Xs fully consumed.
    s_free(stats, d_xs);

    // Stage 4b (compact only): park d_t1_meta on pinned host across
    // the T1 sort phase. d_t1_meta is only needed again for
    // launch_gather_u64 at the end of T1 sort — holding it alive
    // through CUB setup was responsible for the 6256 MB overall
    // streaming peak (d_t1_meta 2080 + d_t1_mi 1040 + CUB working 3120
    // + scratch). JIT H2D before the gather below, free right after.
    // Mirror of stage 4a for T2.
    //
    // Stage 4f: use caller-provided scratch when present (amortised
    // across batch); fall back to per-plot malloc_host otherwise. Same
    // pattern applied to h_t1_keys_merged, h_t2_*, h_t3 below.
    //
    // Plain mode skips the park entirely: d_t1_meta stays live through
    // T1 sort. Costs ~2 GB peak but saves a PCIe round-trip.
    bool      const h_meta_owned = (!scratch.plain_mode && scratch.h_meta == nullptr);
    uint64_t* h_t1_meta = nullptr;
    if (!scratch.plain_mode) {
        h_t1_meta = h_meta_owned
            ? static_cast<uint64_t*>(sycl::malloc_host(cap * sizeof(uint64_t), q))
            : scratch.h_meta;
        if (!h_t1_meta) throw std::runtime_error("sycl::malloc_host(h_t1_meta) failed");
        q.memcpy(h_t1_meta, d_t1_meta, t1_count * sizeof(uint64_t)).wait();
        s_free(stats, d_t1_meta);
        d_t1_meta = nullptr;
    }

    // ---------- Phase T1 sort (tiled, N=2) ----------
    // Partition T1 into two halves by index, CUB-sort each with scratch
    // sized for the larger half, then stable 2-way merge the sorted runs
    // back into the extract-input slot (d_keys_in / d_vals_in) — that
    // slot is free because the CUB sort has already consumed it.
    //
    // N=2 is the minimal case that exercises the tile + merge path; a
    // larger N shrinks per-tile CUB scratch further but needs a multi-
    // way merge or a tree of pairwise merges. Phase 6 can bump N once
    // Phase 4's k=28 VRAM measurement shows how tight the budget is.
    uint64_t const t1_tile_n0  = t1_count / 2;
    uint64_t const t1_tile_n1  = t1_count - t1_tile_n0;
    uint64_t const t1_tile_max = (t1_tile_n0 > t1_tile_n1) ? t1_tile_n0 : t1_tile_n1;

    size_t t1_sort_bytes = 0;
    launch_sort_pairs_u32_u32(
        nullptr, t1_sort_bytes,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        t1_tile_max, 0, cfg.k, q);

    stats.phase = "T1 sort";
    // With T1 SoA emission, d_t1_mi IS the CUB key input. We only need
    // d_keys_out (CUB sort output), d_vals_in (identity) + d_vals_out
    // (sorted vals). d_t1_mi is freed as soon as CUB consumes it.
    uint32_t* d_keys_out     = nullptr;
    uint32_t* d_vals_in      = nullptr;
    uint32_t* d_vals_out     = nullptr;
    void*     d_sort_scratch = nullptr;
    s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
    s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
    s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
    s_malloc(stats, d_sort_scratch, t1_sort_bytes,          "d_sort_scratch(t1)");

    int p_t1_sort = begin_phase("T1 sort");
    launch_init_u32_identity(d_vals_in, t1_count, q);
    if (t1_tile_n0 > 0) {
        launch_sort_pairs_u32_u32(
            d_sort_scratch, t1_sort_bytes,
            d_t1_mi + 0, d_keys_out + 0,
            d_vals_in + 0, d_vals_out + 0,
            t1_tile_n0, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
    }
    if (t1_tile_n1 > 0) {
        launch_sort_pairs_u32_u32(
            d_sort_scratch, t1_sort_bytes,
            d_t1_mi + t1_tile_n0, d_keys_out + t1_tile_n0,
            d_vals_in + t1_tile_n0, d_vals_out + t1_tile_n0,
            t1_tile_n1, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
    }

    // Scratch + vals_in + d_t1_mi dead after CUB.
    s_free(stats, d_sort_scratch);
    s_free(stats, d_vals_in);
    s_free(stats, d_t1_mi);

    // 3-pass post-CUB (merge → gather meta) — same shape as T2 sort,
    // but T1 only has one gather stream (meta) so it's 2 passes here.
    uint32_t* d_t1_keys_merged  = nullptr;
    uint32_t* d_t1_merged_vals  = nullptr;
    s_malloc(stats, d_t1_keys_merged, cap * sizeof(uint32_t), "d_t1_keys_merged");
    s_malloc(stats, d_t1_merged_vals, cap * sizeof(uint32_t), "d_t1_merged_vals");

    launch_merge_pairs_stable_2way_u32_u32(
        d_keys_out + 0,          d_vals_out + 0,          t1_tile_n0,
        d_keys_out + t1_tile_n0, d_vals_out + t1_tile_n0, t1_tile_n1,
        d_t1_keys_merged, d_t1_merged_vals, t1_count, q);
    s_free(stats, d_keys_out);
    s_free(stats, d_vals_out);

    // Stage 4c (compact only): d_t1_keys_merged is not used by the
    // gather below (gather uses d_t1_merged_vals for indices); it is
    // only consumed by T2 match as the "d_sorted_mi" input. Park it on
    // pinned host across the gather peak so the 1040 MB doesn't coexist
    // with d_t1_merged_vals + d_t1_meta + d_t1_meta_sorted. H2D'd back
    // at T2 match entry.
    //
    // Plain mode keeps d_t1_keys_merged live across the gather peak.
    bool      const h_keys_owned = (!scratch.plain_mode && scratch.h_keys_merged == nullptr);
    uint32_t* h_t1_keys_merged = nullptr;
    if (!scratch.plain_mode) {
        h_t1_keys_merged = h_keys_owned
            ? static_cast<uint32_t*>(sycl::malloc_host(cap * sizeof(uint32_t), q))
            : scratch.h_keys_merged;
        if (!h_t1_keys_merged) throw std::runtime_error("sycl::malloc_host(h_t1_keys_merged) failed");
        q.memcpy(h_t1_keys_merged, d_t1_keys_merged, t1_count * sizeof(uint32_t)).wait();
        s_free(stats, d_t1_keys_merged);
        d_t1_keys_merged = nullptr;
    }

    // Stage 4b (compact only): JIT H2D d_t1_meta back onto the device
    // for the gather, then free it immediately. Peak during this window:
    //   d_t1_keys_merged (1040) + d_t1_merged_vals (1040)
    //   + d_t1_meta (2080 H2D) + d_t1_meta_sorted (2080 populated)
    //   = 6240 MB — same as T2 sort's gather peak, and no longer the
    // overall bottleneck on its own.
    //
    // Plain mode: d_t1_meta is already live (never parked).
    if (!scratch.plain_mode) {
        s_malloc(stats, d_t1_meta, cap * sizeof(uint64_t), "d_t1_meta");
        q.memcpy(d_t1_meta, h_t1_meta, t1_count * sizeof(uint64_t)).wait();
        if (h_meta_owned) sycl::free(h_t1_meta, q);
        h_t1_meta = nullptr;
    }

    uint64_t* d_t1_meta_sorted = nullptr;
    s_malloc(stats, d_t1_meta_sorted, cap * sizeof(uint64_t), "d_t1_meta_sorted");
    launch_gather_u64(d_t1_meta, d_t1_merged_vals, d_t1_meta_sorted, t1_count, q);
    end_phase(p_t1_sort);
    s_free(stats, d_t1_meta);
    s_free(stats, d_t1_merged_vals);

    // Stage 4c (compact only): H2D d_t1_keys_merged back now that T2
    // match (its consumer) is about to start. Pinned host freed after
    // H2D. Plain mode: d_t1_keys_merged is already live.
    if (!scratch.plain_mode) {
        s_malloc(stats, d_t1_keys_merged, cap * sizeof(uint32_t), "d_t1_keys_merged");
        q.memcpy(d_t1_keys_merged, h_t1_keys_merged, t1_count * sizeof(uint32_t)).wait();
        if (h_keys_owned) sycl::free(h_t1_keys_merged, q);
        h_t1_keys_merged = nullptr;
    }

    // ---------- Phase T2 match ----------
    // Plain mode: single-pass full-cap N=1 match. Device live set
    // during match is T1 sorted (3.07 GB at k=28) + full-cap T2 output
    // (4.16 GB) ≈ 7.23 GB. No PCIe round-trips.
    //
    // Compact mode (tiled N=2, D2H per pass): two bucket-range passes
    // through half-cap device staging + pinned host accumulators. Match
    // live set drops to T1 sorted + half-cap staging ≈ 5.15 GB, at the
    // cost of ~70 ms of PCIe per pass. This is stage 3 of C (see
    // docs/t2-match-tiling-plan.md). Pool path uses the single-shot
    // launch_t2_match — it has the VRAM and doesn't pay the staging
    // round-trip cost.
    //
    // Per-pass compact safety: we expect each half to produce ≤ cap/2
    // pairs because the match output is roughly uniform across bucket
    // ids. cap itself has a built-in safety margin (see
    // extra_margin_bits in PoolSizing), and typical actual utilisation
    // is well under 100 %. If a pass ever exceeds staging capacity we
    // throw rather than silently dropping pairs.
    stats.phase = "T2 match";
    auto t2p = make_t2_params(cfg.k, cfg.strength);

    // Shared outputs. In plain mode d_t2_meta / d_t2_xbits / d_t2_mi
    // all become live full-cap buffers here; the T2 sort / gather
    // sections below skip the JIT H2D re-hydrations. In compact mode
    // only d_t2_mi is live here (hydrated from the per-plot h_t2_mi),
    // and h_t2_meta / h_t2_xbits hold the concatenated outputs on
    // pinned host until JIT H2D at the gather site.
    uint64_t* d_t2_meta  = nullptr;
    uint32_t* d_t2_mi    = nullptr;
    uint32_t* d_t2_xbits = nullptr;
    uint64_t t2_count    = 0;
    uint64_t* h_t2_meta  = nullptr;
    uint32_t* h_t2_xbits = nullptr;
    bool      h_xbits_owned = false;

    if (scratch.plain_mode) {
        // Plain: one-shot launch_t2_match into full-cap device buffers.
        size_t t2_temp_bytes = 0;
        launch_t2_match(cfg.plot_id.data(), t2p, nullptr, nullptr, t1_count,
                        nullptr, nullptr, nullptr, d_counter, cap,
                        nullptr, &t2_temp_bytes, q);

        void* d_t2_match_temp = nullptr;
        s_malloc(stats, d_t2_meta,       cap * sizeof(uint64_t), "d_t2_meta");
        s_malloc(stats, d_t2_mi,         cap * sizeof(uint32_t), "d_t2_mi");
        s_malloc(stats, d_t2_xbits,      cap * sizeof(uint32_t), "d_t2_xbits");
        s_malloc(stats, d_t2_match_temp, t2_temp_bytes,          "d_t2_match_temp");

        q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        int p_t2 = begin_phase("T2 match");
        launch_t2_match(cfg.plot_id.data(), t2p,
                        d_t1_meta_sorted, d_t1_keys_merged, t1_count,
                        d_t2_meta, d_t2_mi, d_t2_xbits,
                        d_counter, cap,
                        d_t2_match_temp, &t2_temp_bytes, q);
        end_phase(p_t2);

        q.memcpy(&t2_count, d_counter, sizeof(uint64_t)).wait();
        if (t2_count > cap) throw std::runtime_error("T2 overflow");

        s_free(stats, d_t2_match_temp);
        s_free(stats, d_t1_meta_sorted);
        s_free(stats, d_t1_keys_merged);
    } else {
        // Compact: N-tile cap/N staging with pinned-host accumulators.
        // N = scratch.t2_tile_count: 2 = compact (~2.3 GB staging at
        // k=28); 8 = minimal (~570 MB) for 4 GiB cards. Must be a power
        // of 2 ≤ t2_num_buckets so even bucket distribution is exact.
        uint32_t const t2_num_buckets =
            (1u << t2p.num_section_bits) * (1u << t2p.num_match_key_bits);
        int const N = scratch.t2_tile_count;
        if (N < 2 || (N & (N - 1)) != 0) {
            throw std::runtime_error(
                "scratch.t2_tile_count must be a power of 2 ≥ 2 (got " +
                std::to_string(N) + ")");
        }
        if (static_cast<uint32_t>(N) > t2_num_buckets) {
            throw std::runtime_error(
                "scratch.t2_tile_count " + std::to_string(N) +
                " exceeds t2_num_buckets " + std::to_string(t2_num_buckets));
        }
        uint64_t const t2_tile_cap = (cap + uint64_t(N) - 1) / uint64_t(N);

        size_t t2_temp_bytes = 0;
        launch_t2_match_prepare(cfg.plot_id.data(), t2p, nullptr, t1_count,
                                d_counter, nullptr, &t2_temp_bytes, q);

        // Tile-cap device staging (reused across all N passes).
        uint64_t* d_t2_meta_stage  = nullptr;
        uint32_t* d_t2_mi_stage    = nullptr;
        uint32_t* d_t2_xbits_stage = nullptr;
        void*     d_t2_match_temp  = nullptr;
        s_malloc(stats, d_t2_meta_stage,  t2_tile_cap * sizeof(uint64_t), "d_t2_meta_stage");
        s_malloc(stats, d_t2_mi_stage,    t2_tile_cap * sizeof(uint32_t), "d_t2_mi_stage");
        s_malloc(stats, d_t2_xbits_stage, t2_tile_cap * sizeof(uint32_t), "d_t2_xbits_stage");
        s_malloc(stats, d_t2_match_temp,  t2_temp_bytes,                  "d_t2_match_temp");

        // Full-cap pinned host that will hold the concatenated T2 output.
        // Stage 4f: reuse the caller-provided scratch for h_meta / h_xbits
        // (amortised across batch). h_t2_mi is still allocated per-plot.
        auto alloc_pinned_or_throw = [&](size_t bytes, char const* what) {
            void* p = sycl::malloc_host(bytes, q);
            if (!p) throw std::runtime_error(std::string("sycl::malloc_host(")
                                             + what + ") failed");
            return p;
        };
        h_t2_meta  = h_meta_owned
            ? static_cast<uint64_t*>(alloc_pinned_or_throw(cap * sizeof(uint64_t), "h_t2_meta"))
            : scratch.h_meta;
        uint32_t* h_t2_mi = static_cast<uint32_t*>(
            alloc_pinned_or_throw(cap * sizeof(uint32_t), "h_t2_mi"));
        h_xbits_owned = (scratch.h_t2_xbits == nullptr);
        h_t2_xbits = h_xbits_owned
            ? static_cast<uint32_t*>(alloc_pinned_or_throw(cap * sizeof(uint32_t), "h_t2_xbits"))
            : scratch.h_t2_xbits;

        // Compute bucket + fine-bucket offsets once; both passes share
        // them. Also zeroes d_counter.
        launch_t2_match_prepare(cfg.plot_id.data(), t2p,
                                d_t1_keys_merged, t1_count,
                                d_counter, d_t2_match_temp, &t2_temp_bytes, q);

        auto run_pass_and_stage = [&](uint32_t bucket_begin, uint32_t bucket_end,
                                      uint64_t host_offset) -> uint64_t
        {
            launch_t2_match_range(cfg.plot_id.data(), t2p,
                                  d_t1_meta_sorted, d_t1_keys_merged, t1_count,
                                  d_t2_meta_stage, d_t2_mi_stage, d_t2_xbits_stage,
                                  d_counter, t2_tile_cap, d_t2_match_temp,
                                  bucket_begin, bucket_end, q);
            uint64_t pass_count = 0;
            q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
            if (pass_count > t2_tile_cap) {
                throw std::runtime_error(
                    "T2 match pass overflow: bucket range [" +
                    std::to_string(bucket_begin) + "," + std::to_string(bucket_end) +
                    ") produced " + std::to_string(pass_count) +
                    " pairs, staging holds " + std::to_string(t2_tile_cap) +
                    " (consider lower N or fall back to compact tier).");
            }
            q.memcpy(h_t2_meta  + host_offset, d_t2_meta_stage,  pass_count * sizeof(uint64_t));
            q.memcpy(h_t2_mi    + host_offset, d_t2_mi_stage,    pass_count * sizeof(uint32_t));
            q.memcpy(h_t2_xbits + host_offset, d_t2_xbits_stage, pass_count * sizeof(uint32_t));
            q.wait();
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();
            return pass_count;
        };

        int p_t2 = begin_phase("T2 match");
        // N evenly-spaced bucket ranges. host_offset accumulates so each
        // pass appends to the pinned host buffer behind the prior pass.
        t2_count = 0;
        for (int pass = 0; pass < N; ++pass) {
            uint32_t const bucket_begin =
                uint32_t(uint64_t(pass)     * t2_num_buckets / uint64_t(N));
            uint32_t const bucket_end =
                uint32_t(uint64_t(pass + 1) * t2_num_buckets / uint64_t(N));
            t2_count += run_pass_and_stage(bucket_begin, bucket_end,
                                           /*host_offset=*/t2_count);
        }
        end_phase(p_t2);

        if (t2_count > cap) throw std::runtime_error("T2 overflow");

        // Free device staging + T1 sorted + match temp before
        // re-allocating the full-cap d_t2_mi that T2 sort expects.
        s_free(stats, d_t2_match_temp);
        s_free(stats, d_t2_meta_stage);
        s_free(stats, d_t2_mi_stage);
        s_free(stats, d_t2_xbits_stage);
        s_free(stats, d_t1_meta_sorted);
        s_free(stats, d_t1_keys_merged);

        // Stage 4a: hydrate full-cap d_t2_mi from h_t2_mi. d_t2_meta
        // and d_t2_xbits are NOT hydrated yet — they stay on pinned
        // host until their gather calls at the end of T2 sort.
        s_malloc(stats, d_t2_mi, cap * sizeof(uint32_t), "d_t2_mi");
        q.memcpy(d_t2_mi, h_t2_mi, t2_count * sizeof(uint32_t));
        q.wait();
        sycl::free(h_t2_mi, q);
    }

    // ---------- Phase T2 sort (tiled, N=2) ----------
    // Mirror of T1 sort above — same tile-and-merge shape, but permute
    // writes a meta-xbits pair (T2 match output is 16 B, split SoA for
    // T3's L1-bound read pattern) instead of plain meta.
    // N=4 tiling halves the CUB scratch peak (~1044 MB → ~522 MB at
    // k=28), bringing the T2 CUB-alloc peak under 8 GB. Merge is done
    // as a tree of three 2-way merges: (0+1)→AB, (2+3)→CD, (AB+CD)→final.
    constexpr int kNumT2Tiles = 4;
    uint64_t t2_tile_n  [kNumT2Tiles];
    uint64_t t2_tile_off[kNumT2Tiles + 1];
    uint64_t const t2_base_tile = t2_count / kNumT2Tiles;
    uint64_t       t2_rem       = t2_count % kNumT2Tiles;
    t2_tile_off[0] = 0;
    for (int t = 0; t < kNumT2Tiles; ++t) {
        t2_tile_n[t]     = t2_base_tile + (t2_rem > 0 ? 1 : 0);
        if (t2_rem > 0) --t2_rem;
        t2_tile_off[t+1] = t2_tile_off[t] + t2_tile_n[t];
    }
    uint64_t t2_tile_max = 0;
    for (int t = 0; t < kNumT2Tiles; ++t)
        if (t2_tile_n[t] > t2_tile_max) t2_tile_max = t2_tile_n[t];

    size_t t2_sort_bytes = 0;
    launch_sort_pairs_u32_u32(
        nullptr, t2_sort_bytes,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        t2_tile_max, 0, cfg.k, q);

    stats.phase = "T2 sort";
    // CUB sort key input = d_t2_mi (emitted SoA by T2 match); no extract
    // needed, so d_keys_in only needs to hold the merged sorted-MI output
    // that downstream T3 match will consume. Allocate it AFTER the CUB
    // tile-sort has freed d_t2_mi to keep peak narrow.
    s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
    s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
    s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
    s_malloc(stats, d_sort_scratch, t2_sort_bytes,          "d_sort_scratch(t2)");

    int p_t2_sort = begin_phase("T2 sort");
    launch_init_u32_identity(d_vals_in, t2_count, q);
    for (int t = 0; t < kNumT2Tiles; ++t) {
        if (t2_tile_n[t] == 0) continue;
        uint64_t off = t2_tile_off[t];
        launch_sort_pairs_u32_u32(
            d_sort_scratch, t2_sort_bytes,
            d_t2_mi    + off, d_keys_out + off,
            d_vals_in  + off, d_vals_out + off,
            t2_tile_n[t], 0, cfg.k, q);
    }

    s_free(stats, d_sort_scratch);
    s_free(stats, d_vals_in);
    s_free(stats, d_t2_mi);

    // Tree-of-2-way-merges: (tile 0 + tile 1) → AB, (tile 2 + tile 3) → CD,
    // then (AB + CD) → final merged stream. AB and CD buffers hold half
    // of the total output each, so their combined footprint (2080 MB at
    // k=28) fits under the budget freed by shrinking the CUB scratch.
    uint64_t const ab_count = t2_tile_n[0] + t2_tile_n[1];
    uint64_t const cd_count = t2_tile_n[2] + t2_tile_n[3];
    uint32_t* d_AB_keys = nullptr;
    uint32_t* d_AB_vals = nullptr;
    uint32_t* d_CD_keys = nullptr;
    uint32_t* d_CD_vals = nullptr;
    s_malloc(stats, d_AB_keys, ab_count * sizeof(uint32_t), "d_t2_AB_keys");
    s_malloc(stats, d_AB_vals, ab_count * sizeof(uint32_t), "d_t2_AB_vals");
    s_malloc(stats, d_CD_keys, cd_count * sizeof(uint32_t), "d_t2_CD_keys");
    s_malloc(stats, d_CD_vals, cd_count * sizeof(uint32_t), "d_t2_CD_vals");

    if (ab_count > 0) {
        launch_merge_pairs_stable_2way_u32_u32(
            d_keys_out + t2_tile_off[0], d_vals_out + t2_tile_off[0], t2_tile_n[0],
            d_keys_out + t2_tile_off[1], d_vals_out + t2_tile_off[1], t2_tile_n[1],
            d_AB_keys, d_AB_vals, ab_count, q);
    }
    if (cd_count > 0) {
        launch_merge_pairs_stable_2way_u32_u32(
            d_keys_out + t2_tile_off[2], d_vals_out + t2_tile_off[2], t2_tile_n[2],
            d_keys_out + t2_tile_off[3], d_vals_out + t2_tile_off[3], t2_tile_n[3],
            d_CD_keys, d_CD_vals, cd_count, q);
    }

    // Per-tile CUB outputs are consumed; free before alloc'ing the
    // final merged buffers.
    s_free(stats, d_keys_out);
    s_free(stats, d_vals_out);

    uint32_t* d_t2_keys_merged = nullptr;   // merged sorted MI for T3.
    uint32_t* d_merged_vals    = nullptr;   // merged sorted src indices.
    s_malloc(stats, d_t2_keys_merged, cap * sizeof(uint32_t), "d_t2_keys_merged");
    s_malloc(stats, d_merged_vals,    cap * sizeof(uint32_t), "d_merged_vals");

    launch_merge_pairs_stable_2way_u32_u32(
        d_AB_keys, d_AB_vals, ab_count,
        d_CD_keys, d_CD_vals, cd_count,
        d_t2_keys_merged, d_merged_vals, t2_count, q);
    s_free(stats, d_AB_keys);
    s_free(stats, d_AB_vals);
    s_free(stats, d_CD_keys);
    s_free(stats, d_CD_vals);

    // Stage 4c (compact only): d_t2_keys_merged is not consumed by the
    // gather calls below (they use d_merged_vals for indices) — it's
    // only needed later by T3 match as the sorted-MI input. Park it on
    // pinned host across the gather peak so the 1040 MB doesn't coexist
    // with d_merged_vals + d_t2_meta + d_t2_meta_sorted. H2D'd back
    // before T3 match.
    //
    // Plain mode keeps d_t2_keys_merged live across the gather peak.
    uint32_t* h_t2_keys_merged = nullptr;
    if (!scratch.plain_mode) {
        h_t2_keys_merged = h_keys_owned  // reuse t1_keys flag: same scratch
            ? static_cast<uint32_t*>(sycl::malloc_host(cap * sizeof(uint32_t), q))
            : scratch.h_keys_merged;
        if (!h_t2_keys_merged) throw std::runtime_error("sycl::malloc_host(h_t2_keys_merged) failed");
        q.memcpy(h_t2_keys_merged, d_t2_keys_merged, t2_count * sizeof(uint32_t)).wait();
        s_free(stats, d_t2_keys_merged);
        d_t2_keys_merged = nullptr;
    }

    // Stage 4a (compact only): JIT H2D the gather source buffers.
    // d_t2_meta is alive only for the duration of its gather (2080 MB
    // at k=28), then freed before d_t2_xbits is H2D'd. With stage 4c
    // the gather peak drops to d_merged_vals (1040) + d_t2_meta (2080)
    // + d_t2_meta_sorted (2080) = 5200 MB (no more d_t2_keys_merged).
    //
    // Plain mode: d_t2_meta and d_t2_xbits are already live from T2
    // match (never parked). Gather reads them directly and frees after.
    if (!scratch.plain_mode) {
        s_malloc(stats, d_t2_meta, cap * sizeof(uint64_t), "d_t2_meta");
        q.memcpy(d_t2_meta, h_t2_meta, t2_count * sizeof(uint64_t));
        q.wait();
        if (h_meta_owned) sycl::free(h_t2_meta, q);
        h_t2_meta = nullptr;
    }

    uint64_t* d_t2_meta_sorted = nullptr;
    s_malloc(stats, d_t2_meta_sorted, cap * sizeof(uint64_t), "d_t2_meta_sorted");
    launch_gather_u64(d_t2_meta, d_merged_vals, d_t2_meta_sorted, t2_count, q);
    q.wait();
    s_free(stats, d_t2_meta);

    if (!scratch.plain_mode) {
        s_malloc(stats, d_t2_xbits, cap * sizeof(uint32_t), "d_t2_xbits");
        q.memcpy(d_t2_xbits, h_t2_xbits, t2_count * sizeof(uint32_t));
        q.wait();
        if (h_xbits_owned) sycl::free(h_t2_xbits, q);
        h_t2_xbits = nullptr;
    }

    uint32_t* d_t2_xbits_sorted = nullptr;
    s_malloc(stats, d_t2_xbits_sorted, cap * sizeof(uint32_t), "d_t2_xbits_sorted");
    launch_gather_u32(d_t2_xbits, d_merged_vals, d_t2_xbits_sorted, t2_count, q);
    end_phase(p_t2_sort);
    s_free(stats, d_t2_xbits);
    s_free(stats, d_merged_vals);

    // ---------- Phase T3 match ----------
    // Plain mode: one-shot launch_t3_match writing directly into
    // full-cap d_t3. No pinned-host staging, no round-trips — saves
    // the per-plot sycl::malloc_host(2 GB) (~500 ms on NVIDIA) plus
    // the two D2H halves + H2D re-hydration. Match live set:
    //   d_t2_keys_merged (1040) + d_t2_meta_sorted (2080)
    //   + d_t2_xbits_sorted (1040) + d_t3 (2080) + temp
    //   = ~6240 MB — fits under plain's 7290 MB T2-match floor.
    //
    // Compact mode (stage 4d.3, N=2 tiled): half-cap d_t3 staging +
    // D2H-to-pinned-host between passes, then full-cap d_t3 + H2D
    // before T3 sort. Keeps T3 match peak at 5200 MB.
    stats.phase = "T3 match";
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    launch_t3_match_prepare(cfg.plot_id.data(), t3p, nullptr, t2_count,
                            d_counter, nullptr, &t3_temp_bytes, q);

    // Stage 4c (compact only): H2D d_t2_keys_merged back from pinned
    // host now that we're about to enter T3 match (its consumer).
    // Pinned host freed after H2D. Plain mode: d_t2_keys_merged is
    // already live (never parked).
    if (!scratch.plain_mode) {
        s_malloc(stats, d_t2_keys_merged, cap * sizeof(uint32_t), "d_t2_keys_merged");
        q.memcpy(d_t2_keys_merged, h_t2_keys_merged, t2_count * sizeof(uint32_t)).wait();
        if (h_keys_owned) sycl::free(h_t2_keys_merged, q);
        h_t2_keys_merged = nullptr;
    }

    T3PairingGpu* d_t3    = nullptr;
    uint64_t      t3_count = 0;

    if (scratch.plain_mode) {
        // Plain: one-shot full-cap T3 match.
        void* d_t3_match_temp = nullptr;
        s_malloc(stats, d_t3,            cap * sizeof(T3PairingGpu), "d_t3");
        s_malloc(stats, d_t3_match_temp, t3_temp_bytes,              "d_t3_match_temp");

        q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        int p_t3 = begin_phase("T3 match + Feistel");
        launch_t3_match(cfg.plot_id.data(), t3p,
                        d_t2_meta_sorted, d_t2_xbits_sorted,
                        d_t2_keys_merged, t2_count,
                        d_t3, d_counter, cap,
                        d_t3_match_temp, &t3_temp_bytes, q);
        end_phase(p_t3);

        q.memcpy(&t3_count, d_counter, sizeof(uint64_t)).wait();
        if (t3_count > cap) throw std::runtime_error("T3 overflow");

        s_free(stats, d_t3_match_temp);
        s_free(stats, d_t2_meta_sorted);
        s_free(stats, d_t2_xbits_sorted);
        s_free(stats, d_t2_keys_merged);
    } else {
        // Compact: N=2 half-cap staging with pinned-host h_t3 accumulator.
        uint64_t const t3_half_cap = (cap + 1) / 2;

        T3PairingGpu* d_t3_stage    = nullptr;
        void*         d_t3_match_temp = nullptr;
        s_malloc(stats, d_t3_stage,      t3_half_cap * sizeof(T3PairingGpu), "d_t3_stage");
        s_malloc(stats, d_t3_match_temp, t3_temp_bytes,                     "d_t3_match_temp");

        // Full-cap pinned host that will hold the concatenated T3 output.
        // Stage 4f: reuse scratch.h_t3 when provided (amortised across
        // batch). T3PairingGpu is just a uint64 proof_fragment, so the
        // scratch buffer is declared as uint64_t* and reinterpret-cast.
        bool const h_t3_owned = (scratch.h_t3 == nullptr);
        T3PairingGpu* h_t3 = h_t3_owned
            ? static_cast<T3PairingGpu*>(sycl::malloc_host(cap * sizeof(T3PairingGpu), q))
            : reinterpret_cast<T3PairingGpu*>(scratch.h_t3);
        if (!h_t3) throw std::runtime_error("sycl::malloc_host(h_t3) failed");

        // Compute bucket + fine-bucket offsets once; both match passes
        // share them. Also zeroes d_counter.
        launch_t3_match_prepare(cfg.plot_id.data(), t3p,
                                d_t2_keys_merged, t2_count,
                                d_counter, d_t3_match_temp, &t3_temp_bytes, q);

        uint32_t const t3_num_buckets =
            (1u << t3p.num_section_bits) * (1u << t3p.num_match_key_bits);
        uint32_t const t3_bucket_mid = t3_num_buckets / 2;

        auto run_t3_pass = [&](uint32_t bucket_begin, uint32_t bucket_end,
                               uint64_t host_offset) -> uint64_t
        {
            launch_t3_match_range(cfg.plot_id.data(), t3p,
                                  d_t2_meta_sorted, d_t2_xbits_sorted,
                                  d_t2_keys_merged, t2_count,
                                  d_t3_stage, d_counter, t3_half_cap,
                                  d_t3_match_temp, bucket_begin, bucket_end, q);
            uint64_t pass_count = 0;
            q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
            if (pass_count > t3_half_cap) {
                throw std::runtime_error(
                    "T3 match pass overflow: bucket range [" +
                    std::to_string(bucket_begin) + "," + std::to_string(bucket_end) +
                    ") produced " + std::to_string(pass_count) +
                    " pairs, staging holds " + std::to_string(t3_half_cap) +
                    ". Lower N or widen staging.");
            }
            q.memcpy(h_t3 + host_offset, d_t3_stage,
                     pass_count * sizeof(T3PairingGpu)).wait();
            // Reset counter so the next pass writes at stage index 0.
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();
            return pass_count;
        };

        int p_t3 = begin_phase("T3 match + Feistel");
        uint64_t const t3_count1 = run_t3_pass(0,              t3_bucket_mid,   /*host_offset=*/0);
        uint64_t const t3_count2 = run_t3_pass(t3_bucket_mid,  t3_num_buckets,  /*host_offset=*/t3_count1);
        end_phase(p_t3);

        t3_count = t3_count1 + t3_count2;
        if (t3_count > cap) throw std::runtime_error("T3 overflow");

        // Free everything that was alive across T3 match: staging, temp,
        // sorted T2 inputs, keys_merged.
        s_free(stats, d_t3_match_temp);
        s_free(stats, d_t3_stage);
        s_free(stats, d_t2_meta_sorted);
        s_free(stats, d_t2_xbits_sorted);
        s_free(stats, d_t2_keys_merged);

        // Re-hydrate full-cap d_t3 on device for T3 sort.
        s_malloc(stats, d_t3, cap * sizeof(T3PairingGpu), "d_t3");
        q.memcpy(d_t3, h_t3, t3_count * sizeof(T3PairingGpu)).wait();
        if (h_t3_owned) sycl::free(h_t3, q);
    }

    // ---------- Phase T3 sort ----------
    size_t t3_sort_bytes = 0;
    launch_sort_keys_u64(
        nullptr, t3_sort_bytes,
        static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        cap, 0, 2 * cfg.k, q);

    stats.phase = "T3 sort";
    uint64_t* d_frags_in  = reinterpret_cast<uint64_t*>(d_t3);
    uint64_t* d_frags_out = nullptr;
    s_malloc(stats, d_frags_out,    cap * sizeof(uint64_t), "d_frags_out");
    s_malloc(stats, d_sort_scratch, t3_sort_bytes,          "d_sort_scratch(t3)");

    int p_t3_sort = begin_phase("T3 sort");
    launch_sort_keys_u64(
        d_sort_scratch, t3_sort_bytes,
        d_frags_in, d_frags_out,
        t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
    end_phase(p_t3_sort);

    s_free(stats, d_t3);
    s_free(stats, d_sort_scratch);

    // ---------- D2H ----------
    // Two destination modes:
    //   caller-supplied pinned_dst (batch): copy D2H into pinned_dst and
    //     return a BORROWING result (external_fragments_ptr). Consumer
    //     must finish reading pinned_dst before the caller reuses it.
    //   no pinned_dst (one-shot): alloc a temp pinned region sized to
    //     t3_count, D2H, copy to an OWNING vector, free the temp.
    stats.phase = "D2H";
    GpuPipelineResult result;
    result.t1_count = t1_count;
    result.t2_count = t2_count;
    result.t3_count = t3_count;

    int p_d2h = begin_phase("D2H copy T3 fragments (pinned)");
    if (t3_count > 0) {
        if (pinned_dst) {
            if (pinned_capacity < t3_count) {
                throw std::runtime_error(
                    "run_gpu_pipeline_streaming: pinned_capacity " +
                    std::to_string(pinned_capacity) +
                    " < t3_count " + std::to_string(t3_count));
            }
            q.memcpy(pinned_dst, d_frags_out, sizeof(uint64_t) * t3_count);
            q.wait();
            result.external_fragments_ptr   = pinned_dst;
            result.external_fragments_count = t3_count;
        } else {
            uint64_t* h_pinned = nullptr;
            h_pinned = static_cast<uint64_t*>(
                sycl::malloc_host(sizeof(uint64_t) * t3_count, sycl_backend::queue()));
            if (!h_pinned) throw std::runtime_error("sycl::malloc_host(h_pinned) failed");
            q.memcpy(h_pinned, d_frags_out, sizeof(uint64_t) * t3_count);
            q.wait();
            result.t3_fragments_storage.resize(t3_count);
            std::memcpy(result.t3_fragments_storage.data(), h_pinned,
                        sizeof(uint64_t) * t3_count);
            sycl::free(h_pinned, sycl_backend::queue());
        }
    }
    end_phase(p_d2h);

    s_free(stats, d_frags_out);
    s_free(stats, d_counter);

    if (stats.verbose) {
        std::fprintf(stderr,
            "[streaming] k=%d strength=%d  peak device VRAM = %.2f MB\n",
            cfg.k, cfg.strength, stats.peak / 1048576.0);
    }
    report_phases();
    return result;
}

} // namespace (anon — streaming impl)

uint64_t* streaming_alloc_pinned_uint64(size_t count)
{
    uint64_t* p = nullptr;
    p = static_cast<uint64_t*>(
        sycl::malloc_host(count * sizeof(uint64_t), sycl_backend::queue()));
    if (!p) return nullptr;
    return p;
}

uint32_t* streaming_alloc_pinned_uint32(size_t count)
{
    uint32_t* p = static_cast<uint32_t*>(
        sycl::malloc_host(count * sizeof(uint32_t), sycl_backend::queue()));
    return p;  // nullptr on failure
}

void streaming_free_pinned_uint32(uint32_t* ptr)
{
    if (ptr) sycl::free(ptr, sycl_backend::queue());
}

void streaming_free_pinned_uint64(uint64_t* ptr)
{
    if (ptr) sycl::free(ptr, sycl_backend::queue());
}

void bind_current_device(int device_id)
{
    sycl_backend::set_current_device_id(device_id);
}

int gpu_device_count()
{
    try {
        return sycl_backend::get_gpu_device_count();
    } catch (...) {
        return 0;
    }
}

} // namespace pos2gpu
