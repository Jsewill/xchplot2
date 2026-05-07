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


#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
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

// Format a byte count as both raw bytes and decimal MB. The previous
// `bytes >> 20` form (integer right-shift = truncating divide by 1 MiB)
// rounded any sub-MiB request down to "0 MB", which masked both the
// real allocation size and any genuine zero-byte sizing bug at the
// call site. Use this helper in every error path so a future
// `requested=0` is unambiguous (raw bytes settles it).
inline std::string s_fmt_bytes(size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "%zu bytes (%.2f MB)", bytes, bytes / 1048576.0);
    return std::string(buf);
}

template <typename T>
inline void s_malloc(StreamingStats& s, T*& out, size_t bytes, char const* reason)
{
    // Zero-byte requests come from sizing queries that returned 0,
    // which downstream callers honour as "skip this alloc" only by
    // accident (sycl::malloc_device(0) returns null on HIP). Surface
    // the actual upstream cause instead of triggering the misleading
    // "Card likely too small" path below.
    if (bytes == 0) {
        throw std::runtime_error(
            std::string("internal: s_malloc('") + reason + "') called with "
            "bytes=0 — an upstream sizing query returned 0 (count=0). On "
            "AMD/HIP this most often indicates a kernel correctness issue "
            "on an unvalidated device — either an AOT target outside the "
            "validated set (the gfx1013/RDNA1 community spoof is the known "
            "case) or AdaptiveCpp's generic SSCP JIT miscompiling a kernel "
            "for the actual gfx ISA. Run the parity tests on this device "
            "to localise: sycl_g_x_parity, sycl_sort_parity, "
            "sycl_bucket_offsets_parity, sycl_t1_parity.");
    }
    if (s.cap && s.live + bytes > s.cap) {
        throw std::runtime_error(
            std::string("streaming VRAM cap: phase=") + s.phase +
            " alloc=" + reason +
            " live=" + s_fmt_bytes(s.live) +
            " + new=" + s_fmt_bytes(bytes) +
            " would exceed cap=" + s_fmt_bytes(s.cap));
    }
    void* p = sycl::malloc_device(bytes, sycl_backend::queue());
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_device(") + reason + "): null — phase=" +
            s.phase + " requested=" + s_fmt_bytes(bytes) +
            " live=" + s_fmt_bytes(s.live) +
            ". Card likely too small for this k via the streaming "
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

// Sanity-check t1_count after T1 match. Healthy plots produce ~2^k
// entries; anything below total_xs/64 (= 2^(k-6)) — let alone literal
// zero — points at kernel correctness on the device, not a VRAM
// shortfall. Catching this here surfaces a clear diagnostic instead of
// letting downstream sort-scratch alloc fail with the misleading
// "Card likely too small" message. Two AMD/HIP cases produce 0 T1
// matches at k=28: the gfx1013/RDNA1 community spoof on a W5700, and
// AdaptiveCpp's generic SSCP JIT on the same RDNA1 silicon (the JIT
// path is theoretically more compatible than the AOT spoof but has
// been observed to miscompile the matcher). Only the OOM further down
// was visible before this check.
inline void validate_t1_count(uint64_t t1_count, int k)
{
    uint64_t const min_plausible = (1ULL << k) >> 6;
    if (t1_count >= min_plausible) return;

    throw std::runtime_error(
        "T1 match produced " + std::to_string(t1_count) + " entries "
        "(expected ~2^" + std::to_string(k) + " = " +
        std::to_string(1ULL << k) + " for k=" + std::to_string(k) +
        "). This indicates a kernel correctness issue on this device, "
        "not a VRAM shortfall. On AMD/HIP this most often means the "
        "AdaptiveCpp target produced wrong output for the actual gfx "
        "ISA — either the gfx1013/RDNA1 community AOT spoof or the "
        "generic SSCP JIT path on an unvalidated card. Build the "
        "parity tests via cmake and verify on this device: "
        "sycl_g_x_parity, sycl_sort_parity, sycl_bucket_offsets_parity, "
        "sycl_t1_parity. The first three exercise individual kernels at "
        "small N; sycl_t1_parity runs the full T1 matcher against the "
        "pos2-chip CPU reference and is the closest reproducer of the "
        "k=28 failure. README's 'Community-tested, not parity-validated' "
        "caveat applies.");
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
    validate_t1_count(t1_count, cfg.k);


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
    StreamingPinnedScratch scratch{};
    // Honor XCHPLOT2_STREAMING_TIER in the no-arg path so test mode and
    // standalone callers can exercise non-default tiers without going
    // through BatchPlotter. Mirrors BatchPlotter's tier selection.
    if (char const* tier_env = std::getenv("XCHPLOT2_STREAMING_TIER")) {
        std::string t = tier_env;
        if (t == "plain") {
            scratch.plain_mode = true;
        } else if (t == "compact") {
            // compact = default (no flags set). Explicitly leave both off.
        } else if (t == "minimal") {
            scratch.t2_tile_count     = 8;
            scratch.gather_tile_count = 4;
        } else if (t == "tiny") {
            scratch.t2_tile_count     = 8;
            scratch.gather_tile_count = 4;
            scratch.tiny_mode         = true;
        }
        // Unrecognized values fall through to default (compact).
    }
    return run_gpu_pipeline_streaming_impl(cfg, /*pinned_dst=*/nullptr,
                                                /*pinned_capacity=*/0,
                                                scratch);
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

    AesHashKeys const xs_keys = make_keys(cfg.plot_id.data());
    uint32_t    const xs_xor_const = cfg.testnet ? 0xA3B1C4D7u : 0u;

    XsCandidateGpu* d_xs = nullptr;
    uint32_t* d_xs_keys_b = nullptr;
    uint32_t* d_xs_vals_b = nullptr;

    bool const xs_sliced = !scratch.plain_mode && scratch.gather_tile_count > 1;

    if (!xs_sliced) {
        // Compact / plain — full-cap gen+sort+pack (4128 MB sort peak).
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

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            // Sentinel-fill keys_a / vals_a head/mid/tail with 0xCD.
            uint64_t const off_mid  = total_xs / 2;
            uint64_t const off_tail = (total_xs >= 16ULL) ? total_xs - 16ULL : 0ULL;
            q.memset(d_xs_keys_a,            0xCD, 64).wait();
            q.memset(d_xs_keys_a + off_mid,  0xCD, 64).wait();
            q.memset(d_xs_keys_a + off_tail, 0xCD, 64).wait();
            q.memset(d_xs_vals_a,            0xCD, 64).wait();
            q.memset(d_xs_vals_a + off_mid,  0xCD, 64).wait();
            q.memset(d_xs_vals_a + off_tail, 0xCD, 64).wait();

            // Trivial-kernel sanity: writes 0xDEADBEEF to keys_a[0..16]
            // with no LDS / no captured struct / no AES. If this
            // produces 0xCDCDCDCD post-launch, AdaptiveCpp's HIP
            // submission path is producing no-op stubs for ANY kernel
            // — the problem is below our level. If it produces
            // 0xDEADBEEF, simple kernels work and the issue is
            // specific to the cooperative-LDS / AES kernel pattern.
            {
                uint32_t* p = d_xs_keys_a;
                q.parallel_for(
                    sycl::nd_range<1>{256, 256},
                    [=](sycl::nd_item<1> it) {
                        size_t idx = it.get_global_id(0);
                        if (idx < 16) p[idx] = 0xDEADBEEFu;
                    }).wait();
                uint32_t check[16] = {};
                q.memcpy(check, d_xs_keys_a, 16 * sizeof(uint32_t)).wait();
                bool const ok = (check[0] == 0xDEADBEEFu);
                std::fprintf(stderr,
                    "[t1-debug] trivial kernel test: %s  (keys_a[0]=0x%08x)\n",
                    ok ? "PASS — simple kernels can write"
                       : "FAIL — kernel writes are not landing",
                    check[0]);
                // Restore sentinel since the trivial kernel overwrote
                // the head region.
                q.memset(d_xs_keys_a, 0xCD, 64).wait();
            }

            // Dump d_aes_tables[0..16]. Standard AES T0[0] = 0xC66363A5.
            // If we see 0xBE / 0xCD here, the T-table USM buffer was
            // never populated by aes_tables_device's q.memcpy — kernels
            // would then read garbage and produce nothing useful.
            {
                uint32_t* d_tables = sycl_backend::aes_tables_device(q);
                uint32_t aes_check[16] = {};
                q.memcpy(aes_check, d_tables, 16 * sizeof(uint32_t)).wait();
                std::fprintf(stderr,
                    "[t1-debug] d_aes_tables[0..16] (T0[a] = (2S[a],S[a],S[a],3S[a]) packed LE; T0[0] = 0xa56363c6):\n");
                for (int i = 0; i < 16; ++i) {
                    std::fprintf(stderr, "  [%2d] 0x%08x\n", i, aes_check[i]);
                }
            }
        }

        int p_xs = begin_phase("Xs gen+sort");
        launch_xs_gen(xs_keys, d_xs_keys_a, d_xs_vals_a, total_xs,
                      cfg.k, xs_xor_const, q);

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sn = (total_xs < 16ULL) ? total_xs : 16ULL;
            uint64_t const off_mid  = total_xs / 2;
            uint64_t const off_tail = (total_xs >= 16ULL) ? total_xs - 16ULL : 0ULL;
            uint32_t ka_h[16] = {}, va_h[16] = {};
            uint32_t ka_m[16] = {}, va_m[16] = {};
            uint32_t ka_t[16] = {}, va_t[16] = {};
            q.memcpy(ka_h, d_xs_keys_a,            sn * sizeof(uint32_t)).wait();
            q.memcpy(va_h, d_xs_vals_a,            sn * sizeof(uint32_t)).wait();
            q.memcpy(ka_m, d_xs_keys_a + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(va_m, d_xs_vals_a + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(ka_t, d_xs_keys_a + off_tail, sn * sizeof(uint32_t)).wait();
            q.memcpy(va_t, d_xs_vals_a + off_tail, sn * sizeof(uint32_t)).wait();
            std::fprintf(stderr,
                "[t1-debug] post-xs_gen   total_xs=%llu (head idx=0, mid idx=%llu, tail idx=%llu):\n",
                (unsigned long long)total_xs,
                (unsigned long long)off_mid, (unsigned long long)off_tail);
            for (uint64_t i = 0; i < sn; ++i) {
                std::fprintf(stderr,
                    "  H[%2llu] ka=0x%08x va=0x%08x  M[%2llu] ka=0x%08x va=0x%08x  T[%2llu] ka=0x%08x va=0x%08x\n",
                    (unsigned long long)i,            ka_h[i], va_h[i],
                    (unsigned long long)(off_mid + i),  ka_m[i], va_m[i],
                    (unsigned long long)(off_tail + i), ka_t[i], va_t[i]);
            }
        }

        s_malloc(stats, d_xs_keys_b, total_xs * sizeof(uint32_t), "d_xs_keys_b");
        s_malloc(stats, d_xs_vals_b, total_xs * sizeof(uint32_t), "d_xs_vals_b");

        launch_sort_pairs_u32_u32(
            d_xs_cub_scratch, xs_cub_bytes,
            d_xs_keys_a, d_xs_keys_b,
            d_xs_vals_a, d_xs_vals_b,
            total_xs, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
        end_phase(p_xs);

        s_free(stats, d_xs_cub_scratch);
        s_free(stats, d_xs_keys_a);
        s_free(stats, d_xs_vals_a);

        s_malloc(stats, d_xs, total_xs * sizeof(XsCandidateGpu), "d_xs");

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sn = (total_xs < 16ULL) ? total_xs : 16ULL;
            uint64_t const off_mid  = total_xs / 2;
            uint64_t const off_tail = (total_xs >= 16ULL) ? total_xs - 16ULL : 0ULL;
            uint32_t kb_h[16] = {}, vb_h[16] = {};
            uint32_t kb_m[16] = {}, vb_m[16] = {};
            uint32_t kb_t[16] = {}, vb_t[16] = {};
            q.memcpy(kb_h, d_xs_keys_b,            sn * sizeof(uint32_t)).wait();
            q.memcpy(vb_h, d_xs_vals_b,            sn * sizeof(uint32_t)).wait();
            q.memcpy(kb_m, d_xs_keys_b + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(vb_m, d_xs_vals_b + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(kb_t, d_xs_keys_b + off_tail, sn * sizeof(uint32_t)).wait();
            q.memcpy(vb_t, d_xs_vals_b + off_tail, sn * sizeof(uint32_t)).wait();
            std::fprintf(stderr,
                "[t1-debug] post-xs_sort  total_xs=%llu (head idx=0, mid idx=%llu, tail idx=%llu):\n",
                (unsigned long long)total_xs,
                (unsigned long long)off_mid, (unsigned long long)off_tail);
            for (uint64_t i = 0; i < sn; ++i) {
                std::fprintf(stderr,
                    "  H[%2llu] kb=0x%08x vb=0x%08x  M[%2llu] kb=0x%08x vb=0x%08x  T[%2llu] kb=0x%08x vb=0x%08x\n",
                    (unsigned long long)i,            kb_h[i], vb_h[i],
                    (unsigned long long)(off_mid + i),  kb_m[i], vb_m[i],
                    (unsigned long long)(off_tail + i), kb_t[i], vb_t[i]);
            }
        }

        int p_xs_pack = begin_phase("Xs pack");
        launch_xs_pack(d_xs_keys_b, d_xs_vals_b, d_xs, total_xs, q);
        end_phase(p_xs_pack);

        s_free(stats, d_xs_keys_b);
        s_free(stats, d_xs_vals_b);
    } else {
        // Sliced (minimal). Tile gen+sort in N=2 position halves into
        // cap/2 device buffers, D2H per tile to USM-host. Then merge
        // host-pinned tile outputs into device d_xs_keys_b + d_xs_vals_b
        // (full cap). Then pack in N=2 halves with D2H per tile to a
        // host-pinned XsCandidateGpu accumulator. Finally rehydrate
        // d_xs from host pinned. Drops sort peak from 4128 MB → 2056 MB
        // and pack peak from 4096 MB → 3072 MB at k=28.
        uint64_t const xs_tile_n0  = total_xs / 2;
        uint64_t const xs_tile_n1  = total_xs - xs_tile_n0;
        uint64_t const xs_tile_max = (xs_tile_n0 > xs_tile_n1) ? xs_tile_n0 : xs_tile_n1;

        size_t xs_cub_tile_bytes = 0;
        launch_sort_pairs_u32_u32(
            nullptr, xs_cub_tile_bytes,
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            xs_tile_max, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);

        void*     d_xs_cub_scratch  = nullptr;
        uint32_t* d_xs_keys_a_tile  = nullptr;
        uint32_t* d_xs_vals_a_tile  = nullptr;
        uint32_t* d_xs_keys_b_tile  = nullptr;
        uint32_t* d_xs_vals_b_tile  = nullptr;
        s_malloc(stats, d_xs_keys_a_tile, xs_tile_max * sizeof(uint32_t), "d_xs_keys_a_tile");
        s_malloc(stats, d_xs_vals_a_tile, xs_tile_max * sizeof(uint32_t), "d_xs_vals_a_tile");
        s_malloc(stats, d_xs_keys_b_tile, xs_tile_max * sizeof(uint32_t), "d_xs_keys_b_tile");
        s_malloc(stats, d_xs_vals_b_tile, xs_tile_max * sizeof(uint32_t), "d_xs_vals_b_tile");
        s_malloc(stats, d_xs_cub_scratch, xs_cub_tile_bytes,              "d_xs_cub");

        uint32_t* h_xs_keys = static_cast<uint32_t*>(
            sycl::malloc_host(total_xs * sizeof(uint32_t), q));
        if (!h_xs_keys) throw std::runtime_error("sycl::malloc_host(h_xs_keys) failed");
        uint32_t* h_xs_vals = static_cast<uint32_t*>(
            sycl::malloc_host(total_xs * sizeof(uint32_t), q));
        if (!h_xs_vals) throw std::runtime_error("sycl::malloc_host(h_xs_vals) failed");

        int p_xs = begin_phase("Xs gen+sort");
        auto run_tile = [&](uint64_t pos_begin, uint64_t pos_end, uint64_t out_offset) {
            uint64_t tile_n = pos_end - pos_begin;
            if (tile_n == 0) return;
            launch_xs_gen_range(
                xs_keys, d_xs_keys_a_tile, d_xs_vals_a_tile,
                pos_begin, pos_end, cfg.k, xs_xor_const, q);
            launch_sort_pairs_u32_u32(
                d_xs_cub_scratch, xs_cub_tile_bytes,
                d_xs_keys_a_tile, d_xs_keys_b_tile,
                d_xs_vals_a_tile, d_xs_vals_b_tile,
                tile_n, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
            q.memcpy(h_xs_keys + out_offset, d_xs_keys_b_tile,
                     tile_n * sizeof(uint32_t)).wait();
            q.memcpy(h_xs_vals + out_offset, d_xs_vals_b_tile,
                     tile_n * sizeof(uint32_t)).wait();
        };
        run_tile(0,           xs_tile_n0,  0);
        run_tile(xs_tile_n0,  total_xs,    xs_tile_n0);
        end_phase(p_xs);

        s_free(stats, d_xs_cub_scratch);
        s_free(stats, d_xs_vals_b_tile);
        s_free(stats, d_xs_keys_b_tile);
        s_free(stats, d_xs_vals_a_tile);
        s_free(stats, d_xs_keys_a_tile);

        // Full-cap merge outputs on device. Merge from USM-host inputs.
        s_malloc(stats, d_xs_keys_b, total_xs * sizeof(uint32_t), "d_xs_keys_b");
        s_malloc(stats, d_xs_vals_b, total_xs * sizeof(uint32_t), "d_xs_vals_b");
        launch_merge_pairs_stable_2way_u32_u32(
            h_xs_keys + 0,           h_xs_vals + 0,           xs_tile_n0,
            h_xs_keys + xs_tile_n0,  h_xs_vals + xs_tile_n0,  xs_tile_n1,
            d_xs_keys_b, d_xs_vals_b, total_xs, q);
        sycl::free(h_xs_keys, q);
        sycl::free(h_xs_vals, q);

        // Tiled pack. d_xs_pack_tile (cap/2 × XsCandidate = 1024 MB
        // at k=28) reuses across tiles; the packed output collects on
        // host pinned h_xs (cap × XsCandidate = 2048 MB host).
        uint64_t const pack_tile_n0  = total_xs / 2;
        uint64_t const pack_tile_n1  = total_xs - pack_tile_n0;
        uint64_t const pack_tile_max = (pack_tile_n0 > pack_tile_n1) ? pack_tile_n0 : pack_tile_n1;

        XsCandidateGpu* d_xs_pack_tile = nullptr;
        s_malloc(stats, d_xs_pack_tile, pack_tile_max * sizeof(XsCandidateGpu), "d_xs_pack_tile");

        XsCandidateGpu* h_xs = static_cast<XsCandidateGpu*>(
            sycl::malloc_host(total_xs * sizeof(XsCandidateGpu), q));
        if (!h_xs) throw std::runtime_error("sycl::malloc_host(h_xs) failed");

        int p_xs_pack = begin_phase("Xs pack");
        if (pack_tile_n0 > 0) {
            launch_xs_pack_range(d_xs_keys_b + 0, d_xs_vals_b + 0,
                                 d_xs_pack_tile, pack_tile_n0, q);
            q.memcpy(h_xs + 0, d_xs_pack_tile,
                     pack_tile_n0 * sizeof(XsCandidateGpu)).wait();
        }
        if (pack_tile_n1 > 0) {
            launch_xs_pack_range(d_xs_keys_b + pack_tile_n0,
                                 d_xs_vals_b + pack_tile_n0,
                                 d_xs_pack_tile, pack_tile_n1, q);
            q.memcpy(h_xs + pack_tile_n0, d_xs_pack_tile,
                     pack_tile_n1 * sizeof(XsCandidateGpu)).wait();
        }
        end_phase(p_xs_pack);

        s_free(stats, d_xs_pack_tile);
        s_free(stats, d_xs_keys_b);
        s_free(stats, d_xs_vals_b);
        d_xs_keys_b = nullptr;
        d_xs_vals_b = nullptr;

        // Re-hydrate full d_xs on device from host pinned.
        s_malloc(stats, d_xs, total_xs * sizeof(XsCandidateGpu), "d_xs");
        q.memcpy(d_xs, h_xs, total_xs * sizeof(XsCandidateGpu)).wait();
        sycl::free(h_xs, q);
    }

    // ---------- Phase T1 match ----------
    // SoA output: meta (uint64) + mi (uint32). Same 12 B/pair as the old
    // AoS struct, but the two streams can be freed independently — we
    // drop d_t1_mi as soon as CUB consumes it in the T1 sort phase.
    //
    // Minimal mode (gather_tile_count > 1) splits T1 match into N=
    // num_sections passes (one per section_l) with cap/N staging
    // outputs that are D2H'd to host pinned per pass — keeps d_xs +
    // d_t1_meta + d_t1_mi from being co-resident at full-cap. Drops
    // the T1 match peak from
    //   d_xs (2048) + d_t1_meta (2080) + d_t1_mi (1040) = 5168 MB
    // to
    //   d_xs (2048) + d_t1_meta_stage (cap/N × 8) +
    //   d_t1_mi_stage (cap/N × 4) = ~2870 MB at k=28 N=4.
    //
    // d_t1_meta + d_t1_mi (full cap) are then re-allocated on device
    // for T1 sort, with the data H2D'd from host pinned. d_t1_meta
    // stays parked on h_t1_meta across T1 sort exactly as in compact
    // mode (the existing park dance is skipped — data is already on
    // host).
    bool const t1_match_sliced = !scratch.plain_mode && scratch.gather_tile_count > 1;

    stats.phase = "T1 match";
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_counter, cap,
                          nullptr, &t1_temp_bytes, q);

    uint64_t* d_t1_meta = nullptr;
    uint32_t* d_t1_mi   = nullptr;
    void*     d_t1_match_temp = nullptr;

    // Lift h_t1_meta / h_t1_mi out of the T1 sort scope so the sliced
    // T1 match path can populate them directly. h_t1_mi is sliced-only
    // — it's freed in T1 sort once CUB has consumed the H2D'd copy.
    bool      const h_meta_owned = (!scratch.plain_mode && scratch.h_meta == nullptr);
    uint64_t* h_t1_meta = nullptr;
    bool      h_t1_mi_owned = false;
    uint32_t* h_t1_mi = nullptr;

    uint64_t t1_count = 0;

    if (!t1_match_sliced) {
        // Single-shot path (compact / plain): d_t1_meta + d_t1_mi
        // allocated full-cap on device.
        s_malloc(stats, d_t1_meta,        cap * sizeof(uint64_t), "d_t1_meta");
        s_malloc(stats, d_t1_mi,          cap * sizeof(uint32_t), "d_t1_mi");
        s_malloc(stats, d_t1_match_temp,  t1_temp_bytes,          "d_t1_match_temp");

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sample_n = (total_xs < 16ULL) ? total_xs : 16ULL;
            XsCandidateGpu sample[16] = {};
            q.memcpy(sample, d_xs, sample_n * sizeof(XsCandidateGpu)).wait();
            std::fprintf(stderr,
                "[t1-debug] plain pre-launch  k=%d total_xs=%llu cap=%llu  d_xs[0..%llu]:\n",
                cfg.k, (unsigned long long)total_xs,
                (unsigned long long)cap, (unsigned long long)sample_n);
            for (uint64_t i = 0; i < sample_n; ++i) {
                std::fprintf(stderr,
                    "  [%2llu] match_info=0x%08x x=0x%08x\n",
                    (unsigned long long)i, sample[i].match_info, sample[i].x);
            }
        }

        int p_t1 = begin_phase("T1 match");
        q.memset(d_counter, 0, sizeof(uint64_t));
        launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                              d_t1_meta, d_t1_mi, d_counter, cap,
                              d_t1_match_temp, &t1_temp_bytes, q);
        end_phase(p_t1);

        q.memcpy(&t1_count, d_counter, sizeof(uint64_t)).wait();
        if (t1_count > cap) throw std::runtime_error("T1 overflow");
        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            std::fprintf(stderr,
                "[t1-debug] plain post-launch t1_count=%llu\n",
                (unsigned long long)t1_count);
        }
        validate_t1_count(t1_count, cfg.k);

        s_free(stats, d_t1_match_temp);
        s_free(stats, d_xs);
    } else {
        // Sliced path (minimal): N=num_sections passes with cap/N
        // staging buffers. Output accumulates on host pinned, then
        // d_t1_mi + h_t1_meta receive their final populations after
        // d_xs is freed.
        uint32_t const t1_num_sections   = 1u << t1p.num_section_bits;
        uint32_t const t1_num_match_keys = 1u << t1p.num_match_key_bits;
        // 25% safety over the per-section average expected output.
        uint64_t const t1_section_cap =
            ((cap + t1_num_sections - 1) / t1_num_sections) * 5ULL / 4ULL;

        s_malloc(stats, d_t1_match_temp, t1_temp_bytes, "d_t1_match_temp");

        // Compute bucket + fine-bucket offsets once; passes share them.
        // Also zeros d_counter.
        launch_t1_match_prepare(cfg.plot_id.data(), t1p, d_xs, total_xs,
                                d_counter, d_t1_match_temp, &t1_temp_bytes, q);

        // Host pinned full-cap accumulators for meta + mi.
        h_t1_meta = h_meta_owned
            ? static_cast<uint64_t*>(sycl::malloc_host(cap * sizeof(uint64_t), q))
            : scratch.h_meta;
        if (!h_t1_meta) throw std::runtime_error("sycl::malloc_host(h_t1_meta) failed");
        h_t1_mi_owned = true;
        h_t1_mi = static_cast<uint32_t*>(sycl::malloc_host(cap * sizeof(uint32_t), q));
        if (!h_t1_mi) throw std::runtime_error("sycl::malloc_host(h_t1_mi) failed");

        // Per-pass staging device buffers (cap/N).
        uint64_t* d_t1_meta_stage = nullptr;
        uint32_t* d_t1_mi_stage   = nullptr;
        s_malloc(stats, d_t1_meta_stage, t1_section_cap * sizeof(uint64_t), "d_t1_meta_stage");
        s_malloc(stats, d_t1_mi_stage,   t1_section_cap * sizeof(uint32_t), "d_t1_mi_stage");

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sample_n = (total_xs < 16ULL) ? total_xs : 16ULL;
            XsCandidateGpu sample[16] = {};
            q.memcpy(sample, d_xs, sample_n * sizeof(XsCandidateGpu)).wait();
            std::fprintf(stderr,
                "[t1-debug] sliced pre-launch k=%d total_xs=%llu cap=%llu  d_xs[0..%llu]:\n",
                cfg.k, (unsigned long long)total_xs,
                (unsigned long long)cap, (unsigned long long)sample_n);
            for (uint64_t i = 0; i < sample_n; ++i) {
                std::fprintf(stderr,
                    "  [%2llu] match_info=0x%08x x=0x%08x\n",
                    (unsigned long long)i, sample[i].match_info, sample[i].x);
            }
        }

        int p_t1 = begin_phase("T1 match");
        uint64_t host_offset = 0;
        for (uint32_t section_l = 0; section_l < t1_num_sections; ++section_l) {
            uint32_t const bucket_begin = section_l * t1_num_match_keys;
            uint32_t const bucket_end   = (section_l + 1) * t1_num_match_keys;

            launch_t1_match_range(
                cfg.plot_id.data(), t1p, d_xs, total_xs,
                d_t1_meta_stage, d_t1_mi_stage, d_counter, t1_section_cap,
                d_t1_match_temp, bucket_begin, bucket_end, q);

            uint64_t pass_count = 0;
            q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
            if (pass_count > t1_section_cap) {
                throw std::runtime_error(
                    "T1 match (sliced) section_l=" + std::to_string(section_l) +
                    " produced " + std::to_string(pass_count) +
                    " pairs, staging holds " + std::to_string(t1_section_cap) +
                    ". Increase t1_section_cap safety factor.");
            }
            q.memcpy(h_t1_meta + host_offset, d_t1_meta_stage,
                     pass_count * sizeof(uint64_t)).wait();
            q.memcpy(h_t1_mi   + host_offset, d_t1_mi_stage,
                     pass_count * sizeof(uint32_t)).wait();
            host_offset += pass_count;
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        }
        end_phase(p_t1);

        t1_count = host_offset;
        if (t1_count > cap) throw std::runtime_error("T1 overflow");
        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            std::fprintf(stderr,
                "[t1-debug] sliced post-launch t1_count=%llu (sum across %u sections)\n",
                (unsigned long long)t1_count, t1_num_sections);
        }
        validate_t1_count(t1_count, cfg.k);

        s_free(stats, d_t1_meta_stage);
        s_free(stats, d_t1_mi_stage);
        s_free(stats, d_t1_match_temp);

        // Xs fully consumed.
        s_free(stats, d_xs);

        // Re-hydrate d_t1_mi full-cap on device for T1 sort (CUB
        // sort key input). h_t1_meta stays on host across T1 sort.
        s_malloc(stats, d_t1_mi, cap * sizeof(uint32_t), "d_t1_mi");
        q.memcpy(d_t1_mi, h_t1_mi, t1_count * sizeof(uint32_t)).wait();
        if (h_t1_mi_owned) sycl::free(h_t1_mi, q);
        h_t1_mi = nullptr;
        // d_t1_meta stays nullptr — h_t1_meta has the data; the
        // existing T1-sort park block will see d_t1_meta == nullptr
        // and skip the d_t1_meta → h_t1_meta memcpy.
    }

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
    //
    // Sliced mode: h_t1_meta was already populated by the T1 match
    // passes — d_t1_meta is nullptr and the park dance is skipped
    // here. h_meta_owned + h_t1_meta were declared above (lifted out
    // of the original T1-sort scope) so the rest of T1 sort sees the
    // same variables in both paths.
    if (!scratch.plain_mode && !t1_match_sliced) {
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
    //
    // Compact / plain: full-cap d_keys_out + d_vals_in + d_vals_out
    // (1040 MB each at k=28); plus d_t1_mi (1040, full-cap input) +
    // scratch ≈ 4176 MB peak.
    //
    // Minimal: per-tile cap/2 output buffers (520 each) instead of
    // full-cap + USM-host h_keys/h_vals to collect tile outputs +
    // launch_merge_pairs_stable_2way_u32_u32 reading USM-host inputs.
    // Drops T1 sort CUB peak to:
    //   d_t1_mi (1040) + 3 × cap/2 u32 (1560) + scratch ≈ 2616 MB.
    void* d_sort_scratch = nullptr;
    uint32_t* d_keys_out = nullptr;     // populated in compact path; minimal uses h_keys instead
    uint32_t* d_vals_in  = nullptr;     // T2 sort below also uses this; declared at wider scope
    uint32_t* d_vals_out = nullptr;     // populated in compact path; minimal uses h_vals instead
    uint32_t* h_keys     = nullptr;     // USM-host, sliced path only
    uint32_t* h_vals     = nullptr;     // USM-host, sliced path only

    int p_t1_sort = begin_phase("T1 sort");

    if (!t1_match_sliced) {
        // Compact / plain — existing full-cap path.
        s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
        s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
        s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
        s_malloc(stats, d_sort_scratch, t1_sort_bytes,          "d_sort_scratch(t1)");

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

        s_free(stats, d_sort_scratch);
        s_free(stats, d_vals_in);
        s_free(stats, d_t1_mi);
    } else {
        // Sliced — per-tile cap/2 output buffers, D2H to USM-host.
        uint32_t* d_keys_out_tile = nullptr;
        uint32_t* d_vals_in_tile  = nullptr;
        uint32_t* d_vals_out_tile = nullptr;
        s_malloc(stats, d_keys_out_tile, t1_tile_max * sizeof(uint32_t), "d_t1_keys_out_tile");
        s_malloc(stats, d_vals_in_tile,  t1_tile_max * sizeof(uint32_t), "d_t1_vals_in_tile");
        s_malloc(stats, d_vals_out_tile, t1_tile_max * sizeof(uint32_t), "d_t1_vals_out_tile");
        s_malloc(stats, d_sort_scratch,  t1_sort_bytes,                  "d_sort_scratch(t1)");

        h_keys = static_cast<uint32_t*>(sycl::malloc_host(cap * sizeof(uint32_t), q));
        if (!h_keys) throw std::runtime_error("sycl::malloc_host(h_keys t1) failed");
        h_vals = static_cast<uint32_t*>(sycl::malloc_host(cap * sizeof(uint32_t), q));
        if (!h_vals) throw std::runtime_error("sycl::malloc_host(h_vals t1) failed");

        auto run_tile = [&](uint64_t tile_off, uint64_t tile_n) {
            if (tile_n == 0) return;
            uint32_t const off32 = static_cast<uint32_t>(tile_off);
            uint32_t* d_vals_in_tile_local = d_vals_in_tile;
            q.parallel_for(
                sycl::range<1>{ static_cast<size_t>(tile_n) },
                [=](sycl::id<1> i) {
                    d_vals_in_tile_local[i] = off32 + uint32_t(i);
                }).wait();
            launch_sort_pairs_u32_u32(
                d_sort_scratch, t1_sort_bytes,
                d_t1_mi + tile_off, d_keys_out_tile,
                d_vals_in_tile,    d_vals_out_tile,
                tile_n, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
            q.memcpy(h_keys + tile_off, d_keys_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
            q.memcpy(h_vals + tile_off, d_vals_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
        };
        run_tile(0,            t1_tile_n0);
        run_tile(t1_tile_n0,   t1_tile_n1);

        s_free(stats, d_sort_scratch);
        s_free(stats, d_vals_out_tile);
        s_free(stats, d_vals_in_tile);
        s_free(stats, d_keys_out_tile);
        s_free(stats, d_t1_mi);
    }

    // 3-pass post-CUB (merge → gather meta) — same shape as T2 sort,
    // but T1 only has one gather stream (meta) so it's 2 passes here.
    uint32_t* d_t1_keys_merged  = nullptr;
    uint32_t* d_t1_merged_vals  = nullptr;
    s_malloc(stats, d_t1_keys_merged, cap * sizeof(uint32_t), "d_t1_keys_merged");
    s_malloc(stats, d_t1_merged_vals, cap * sizeof(uint32_t), "d_t1_merged_vals");

    if (!t1_match_sliced) {
        launch_merge_pairs_stable_2way_u32_u32(
            d_keys_out + 0,          d_vals_out + 0,          t1_tile_n0,
            d_keys_out + t1_tile_n0, d_vals_out + t1_tile_n0, t1_tile_n1,
            d_t1_keys_merged, d_t1_merged_vals, t1_count, q);
        s_free(stats, d_keys_out);
        s_free(stats, d_vals_out);
    } else {
        // Merge inputs are USM-host; the kernel reads via PCIe (sequential
        // 2-way merge → bandwidth-bound, ~3.27 GB at k=28 / ~25 GB/s ≈
        // 130 ms). Live device set during merge is just the two cap-sized
        // output buffers (d_t1_keys_merged + d_t1_merged_vals = 2080 MB).
        launch_merge_pairs_stable_2way_u32_u32(
            h_keys + 0,            h_vals + 0,            t1_tile_n0,
            h_keys + t1_tile_n0,   h_vals + t1_tile_n0,   t1_tile_n1,
            d_t1_keys_merged, d_t1_merged_vals, t1_count, q);
        sycl::free(h_keys, q); h_keys = nullptr;
        sycl::free(h_vals, q); h_vals = nullptr;
    }

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
    int const t1_gather_N = scratch.plain_mode ? 1 : scratch.gather_tile_count;
    if (!scratch.plain_mode) {
        s_malloc(stats, d_t1_meta, cap * sizeof(uint64_t), "d_t1_meta");
        q.memcpy(d_t1_meta, h_t1_meta, t1_count * sizeof(uint64_t)).wait();
        // With gather_tile_count > 1 we reuse h_t1_meta to stage the
        // sorted output (overwriting the unsorted data we just
        // rehydrated from); defer the free until after the H2D rebuild.
        if (t1_gather_N <= 1) {
            if (h_meta_owned) sycl::free(h_t1_meta, q);
            h_t1_meta = nullptr;
        }
    }

    uint64_t* d_t1_meta_sorted = nullptr;
    if (t1_gather_N <= 1) {
        s_malloc(stats, d_t1_meta_sorted, cap * sizeof(uint64_t), "d_t1_meta_sorted");
        launch_gather_u64(d_t1_meta, d_t1_merged_vals, d_t1_meta_sorted, t1_count, q);
        end_phase(p_t1_sort);
        s_free(stats, d_t1_meta);
        s_free(stats, d_t1_merged_vals);
    } else {
        // Tiled-output gather (minimal tier). Produce the sorted output
        // in N tiles, D2H each tile to h_t1_meta (overwriting the
        // unsorted data we just rehydrated from), then free the inputs
        // and rebuild the full d_t1_meta_sorted on device. Peak during
        // gather drops from
        //   d_t1_meta (2080) + d_t1_merged_vals (1040)
        //   + d_t1_meta_sorted (2080) = 5200 MB
        // to
        //   d_t1_meta (2080) + d_t1_merged_vals (1040)
        //   + d_tile (cap/N × u64 = 520 at N=4) = ~3640 MB.
        uint64_t const tile_max =
            (t1_count + uint64_t(t1_gather_N) - 1) / uint64_t(t1_gather_N);
        uint64_t* d_tile = nullptr;
        s_malloc(stats, d_tile, tile_max * sizeof(uint64_t), "d_t1_meta_sorted_tile");
        for (int n = 0; n < t1_gather_N; ++n) {
            uint64_t const tile_off = uint64_t(n) * tile_max;
            if (tile_off >= t1_count) break;
            uint64_t const tile_n = std::min(tile_max, t1_count - tile_off);
            launch_gather_u64(
                d_t1_meta, d_t1_merged_vals + tile_off,
                d_tile, tile_n, q);
            q.memcpy(h_t1_meta + tile_off, d_tile,
                     tile_n * sizeof(uint64_t)).wait();
        }
        s_free(stats, d_tile);
        s_free(stats, d_t1_meta);
        s_free(stats, d_t1_merged_vals);
        // Tiny tier: skip the full-cap d_t1_meta_sorted rehydration. The
        // sliced T2 match path (per-section meta_l/meta_r H2D) reads
        // section-sized slices from h_t1_meta directly. Saves 2080 MB of
        // device VRAM at k=28 across T2 match.
        if (!scratch.tiny_mode) {
            s_malloc(stats, d_t1_meta_sorted, cap * sizeof(uint64_t), "d_t1_meta_sorted");
            q.memcpy(d_t1_meta_sorted, h_t1_meta, t1_count * sizeof(uint64_t)).wait();
        }
        end_phase(p_t1_sort);
        // Tiny: keep h_t1_meta alive across T2 match for slicing. Free
        // happens inside the tiny T2 match block.
        if (!scratch.tiny_mode) {
            if (h_meta_owned) sycl::free(h_t1_meta, q);
            h_t1_meta = nullptr;
        }
    }

    // Stage 4c (compact only): H2D d_t1_keys_merged back now that T2
    // match (its consumer) is about to start. Pinned host freed after
    // H2D. Plain mode: d_t1_keys_merged is already live.
    //
    // Tiny tier: skip the rehydration. h_t1_keys_merged stays alive
    // across T2 match; the split kernel reads section_r's mi slice each
    // pass. The T2 prepare step needs mi on device for histogram counts;
    // the tiny T2 match block briefly rehydrates for prepare and frees.
    if (!scratch.plain_mode && !scratch.tiny_mode) {
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
        //
        // Tiny mode: d_t1_keys_merged is parked on host. The prepare
        // kernel needs the sorted mi stream on device for its histogram
        // counts. Briefly rehydrate, run prepare, then free.
        uint32_t* d_t1_keys_for_prepare = nullptr;
        if (scratch.tiny_mode) {
            s_malloc(stats, d_t1_keys_for_prepare, cap * sizeof(uint32_t), "d_t1_keys_merged_prep");
            q.memcpy(d_t1_keys_for_prepare, h_t1_keys_merged,
                     t1_count * sizeof(uint32_t)).wait();
        }
        launch_t2_match_prepare(cfg.plot_id.data(), t2p,
                                scratch.tiny_mode ? d_t1_keys_for_prepare
                                                   : d_t1_keys_merged,
                                t1_count,
                                d_counter, d_t2_match_temp, &t2_temp_bytes, q);
        if (scratch.tiny_mode) {
            s_free(stats, d_t1_keys_for_prepare);
            d_t1_keys_for_prepare = nullptr;
        }

        // Tiny mode: D2H the bucket-offsets table so we can compute each
        // section's row range host-side. Only the bucket-offsets prefix
        // is needed (fine-offsets stay on device for the kernel's binary
        // search).
        uint32_t const num_sections_t2   = 1u << t2p.num_section_bits;
        uint32_t const num_match_keys_t2 = 1u << t2p.num_match_key_bits;
        uint32_t const num_buckets_t2    = num_sections_t2 * num_match_keys_t2;
        std::vector<uint64_t> h_t2_bucket_offsets;
        if (scratch.tiny_mode) {
            h_t2_bucket_offsets.resize(num_buckets_t2 + 1);
            q.memcpy(h_t2_bucket_offsets.data(), d_t2_match_temp,
                     (num_buckets_t2 + 1) * sizeof(uint64_t)).wait();
        }

        auto compute_section_r_t2 = [&](uint32_t section_l) -> uint32_t {
            uint32_t const mask = num_sections_t2 - 1u;
            uint32_t const rl   = ((section_l << 1) |
                                   (section_l >> (t2p.num_section_bits - 1))) & mask;
            uint32_t const rl1  = (rl + 1u) & mask;
            return ((rl1 >> 1) |
                    (rl1 << (t2p.num_section_bits - 1))) & mask;
        };

        // Per-section state (tiny only): re-allocate slices when the
        // pass crosses into a new section_l. Slices stay on device for
        // all passes within a section.
        int32_t  cur_section_l = -1;
        uint64_t cur_section_l_row_start = 0;
        uint64_t cur_section_r_row_start = 0;
        uint64_t* d_t2_meta_l_slice = nullptr;
        uint64_t* d_t2_meta_r_slice = nullptr;
        uint32_t* d_t2_mi_r_slice   = nullptr;

        auto release_t2_slices = [&]() {
            if (d_t2_mi_r_slice)   { s_free(stats, d_t2_mi_r_slice);   d_t2_mi_r_slice   = nullptr; }
            if (d_t2_meta_r_slice) { s_free(stats, d_t2_meta_r_slice); d_t2_meta_r_slice = nullptr; }
            if (d_t2_meta_l_slice) { s_free(stats, d_t2_meta_l_slice); d_t2_meta_l_slice = nullptr; }
            cur_section_l = -1;
        };

        auto ensure_t2_slices = [&](uint32_t section_l) {
            if (static_cast<int32_t>(section_l) == cur_section_l) return;
            release_t2_slices();

            uint32_t const section_r = compute_section_r_t2(section_l);
            cur_section_l_row_start = h_t2_bucket_offsets[section_l * num_match_keys_t2];
            uint64_t section_l_row_end =
                h_t2_bucket_offsets[(section_l + 1) * num_match_keys_t2];
            uint64_t section_l_count = section_l_row_end - cur_section_l_row_start;
            cur_section_r_row_start = h_t2_bucket_offsets[section_r * num_match_keys_t2];
            uint64_t section_r_row_end =
                h_t2_bucket_offsets[(section_r + 1) * num_match_keys_t2];
            uint64_t section_r_count = section_r_row_end - cur_section_r_row_start;

            if (section_l_count > 0) {
                s_malloc(stats, d_t2_meta_l_slice, section_l_count * sizeof(uint64_t), "d_t2_meta_l_slice");
                q.memcpy(d_t2_meta_l_slice, h_t1_meta + cur_section_l_row_start,
                         section_l_count * sizeof(uint64_t)).wait();
            }
            if (section_r_count > 0) {
                s_malloc(stats, d_t2_meta_r_slice, section_r_count * sizeof(uint64_t), "d_t2_meta_r_slice");
                s_malloc(stats, d_t2_mi_r_slice,   section_r_count * sizeof(uint32_t), "d_t2_mi_r_slice");
                q.memcpy(d_t2_meta_r_slice, h_t1_meta + cur_section_r_row_start,
                         section_r_count * sizeof(uint64_t)).wait();
                q.memcpy(d_t2_mi_r_slice, h_t1_keys_merged + cur_section_r_row_start,
                         section_r_count * sizeof(uint32_t)).wait();
            }
            cur_section_l = static_cast<int32_t>(section_l);
        };

        auto run_pass_and_stage = [&](uint32_t bucket_begin, uint32_t bucket_end,
                                      uint64_t host_offset) -> uint64_t
        {
            if (scratch.tiny_mode) {
                // Tiny: every pass must fit entirely in one section_l,
                // so caller (the loop below) is responsible for chunking
                // at section boundaries.
                uint32_t const section_l = bucket_begin / num_match_keys_t2;
                if (bucket_end > 0 && (bucket_end - 1) / num_match_keys_t2 != section_l) {
                    throw std::runtime_error(
                        "tiny T2 match: pass [" + std::to_string(bucket_begin) +
                        "," + std::to_string(bucket_end) +
                        ") crosses section_l boundary (num_match_keys=" +
                        std::to_string(num_match_keys_t2) + ")");
                }
                ensure_t2_slices(section_l);
                if (d_t2_meta_l_slice && d_t2_meta_r_slice && d_t2_mi_r_slice) {
                    launch_t2_match_section_pair_split_range(
                        cfg.plot_id.data(), t2p,
                        d_t2_meta_l_slice, cur_section_l_row_start,
                        d_t2_meta_r_slice, d_t2_mi_r_slice, cur_section_r_row_start,
                        d_t2_meta_stage, d_t2_mi_stage, d_t2_xbits_stage,
                        d_counter, t2_tile_cap, d_t2_match_temp,
                        bucket_begin, bucket_end, q);
                }
                // Empty section pair → no pairings, fall through to drain.
            } else {
                launch_t2_match_range(cfg.plot_id.data(), t2p,
                                      d_t1_meta_sorted, d_t1_keys_merged, t1_count,
                                      d_t2_meta_stage, d_t2_mi_stage, d_t2_xbits_stage,
                                      d_counter, t2_tile_cap, d_t2_match_temp,
                                      bucket_begin, bucket_end, q);
            }
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
        t2_count = 0;
        if (scratch.tiny_mode) {
            // Section-aware iteration: for each section_l, run all
            // passes whose [bucket_begin, bucket_end) range falls in
            // that section's bucket range. With t2_tile_count=N and
            // num_match_keys_t2 per section, alignment requires
            // num_buckets/N to divide num_match_keys_t2 evenly OR
            // num_match_keys_t2 to divide num_buckets/N evenly. The
            // latter would put multiple sections in one pass (which
            // would cross slice boundaries) — guard against it.
            uint32_t const pass_size = num_buckets_t2 / uint32_t(N);
            if (pass_size > num_match_keys_t2) {
                throw std::runtime_error(
                    "tiny T2 match: pass spans multiple sections "
                    "(pass_size=" + std::to_string(pass_size) +
                    ", num_match_keys=" + std::to_string(num_match_keys_t2) +
                    "). Increase t2_tile_count.");
            }
            if (num_match_keys_t2 % pass_size != 0) {
                throw std::runtime_error(
                    "tiny T2 match: pass_size " + std::to_string(pass_size) +
                    " does not evenly divide num_match_keys " +
                    std::to_string(num_match_keys_t2) +
                    ". Use t2_tile_count = power-of-2 multiple of num_sections.");
            }
            for (int pass = 0; pass < N; ++pass) {
                uint32_t const bucket_begin =
                    uint32_t(uint64_t(pass)     * num_buckets_t2 / uint64_t(N));
                uint32_t const bucket_end =
                    uint32_t(uint64_t(pass + 1) * num_buckets_t2 / uint64_t(N));
                t2_count += run_pass_and_stage(bucket_begin, bucket_end,
                                               /*host_offset=*/t2_count);
            }
            release_t2_slices();
        } else {
            // N evenly-spaced bucket ranges. host_offset accumulates so
            // each pass appends to the pinned host buffer behind the
            // prior pass.
            for (int pass = 0; pass < N; ++pass) {
                uint32_t const bucket_begin =
                    uint32_t(uint64_t(pass)     * num_buckets_t2 / uint64_t(N));
                uint32_t const bucket_end =
                    uint32_t(uint64_t(pass + 1) * num_buckets_t2 / uint64_t(N));
                t2_count += run_pass_and_stage(bucket_begin, bucket_end,
                                               /*host_offset=*/t2_count);
            }
        }
        end_phase(p_t2);

        if (t2_count > cap) throw std::runtime_error("T2 overflow");

        // Free device staging + T1 sorted + match temp before
        // re-allocating the full-cap d_t2_mi that T2 sort expects.
        s_free(stats, d_t2_match_temp);
        s_free(stats, d_t2_meta_stage);
        s_free(stats, d_t2_mi_stage);
        s_free(stats, d_t2_xbits_stage);
        // Tiny: d_t1_meta_sorted and d_t1_keys_merged are null (parked
        // on host pinned). Free the host buffers; T2 sort below will
        // build its inputs from h_t2_meta/h_t2_mi/h_t2_xbits.
        if (scratch.tiny_mode) {
            if (h_meta_owned) sycl::free(h_t1_meta, q);
            h_t1_meta = nullptr;
            if (h_keys_owned) sycl::free(h_t1_keys_merged, q);
            h_t1_keys_merged = nullptr;
        } else {
            s_free(stats, d_t1_meta_sorted);
            s_free(stats, d_t1_keys_merged);
        }

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
    //
    // Compact / plain: full-cap d_keys_out + d_vals_in + d_vals_out
    // (~4168 MB peak with d_t2_mi during tile sort).
    //
    // Sliced (minimal): per-tile cap/N output buffers + USM-host
    // accumulators, then USM-host parking of AB / CD between merge
    // tree steps so the final merge sees only its own outputs +
    // USM-host inputs (live device ~2080 MB at k=28). Peaks under
    // 4 GiB at every step.

    uint64_t const ab_count = t2_tile_n[0] + t2_tile_n[1];
    uint64_t const cd_count = t2_tile_n[2] + t2_tile_n[3];

    int p_t2_sort = begin_phase("T2 sort");

    if (!t1_match_sliced) {
        // Compact / plain — existing full-cap CUB tile sort.
        s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
        s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
        s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
        s_malloc(stats, d_sort_scratch, t2_sort_bytes,          "d_sort_scratch(t2)");

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
    } else {
        // Sliced — per-tile cap/N output, D2H to USM-host h_keys/h_vals.
        uint32_t* d_keys_out_tile = nullptr;
        uint32_t* d_vals_in_tile  = nullptr;
        uint32_t* d_vals_out_tile = nullptr;
        s_malloc(stats, d_keys_out_tile, t2_tile_max * sizeof(uint32_t), "d_t2_keys_out_tile");
        s_malloc(stats, d_vals_in_tile,  t2_tile_max * sizeof(uint32_t), "d_t2_vals_in_tile");
        s_malloc(stats, d_vals_out_tile, t2_tile_max * sizeof(uint32_t), "d_t2_vals_out_tile");
        s_malloc(stats, d_sort_scratch,  t2_sort_bytes,                  "d_sort_scratch(t2)");

        h_keys = static_cast<uint32_t*>(sycl::malloc_host(cap * sizeof(uint32_t), q));
        if (!h_keys) throw std::runtime_error("sycl::malloc_host(h_keys t2) failed");
        h_vals = static_cast<uint32_t*>(sycl::malloc_host(cap * sizeof(uint32_t), q));
        if (!h_vals) throw std::runtime_error("sycl::malloc_host(h_vals t2) failed");

        for (int t = 0; t < kNumT2Tiles; ++t) {
            uint64_t const tile_n = t2_tile_n[t];
            if (tile_n == 0) continue;
            uint64_t const tile_off = t2_tile_off[t];
            uint32_t const off32    = static_cast<uint32_t>(tile_off);
            uint32_t* d_vals_in_tile_local = d_vals_in_tile;
            q.parallel_for(
                sycl::range<1>{ static_cast<size_t>(tile_n) },
                [=](sycl::id<1> i) {
                    d_vals_in_tile_local[i] = off32 + uint32_t(i);
                }).wait();
            launch_sort_pairs_u32_u32(
                d_sort_scratch, t2_sort_bytes,
                d_t2_mi + tile_off, d_keys_out_tile,
                d_vals_in_tile,    d_vals_out_tile,
                tile_n, 0, cfg.k, q);
            q.memcpy(h_keys + tile_off, d_keys_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
            q.memcpy(h_vals + tile_off, d_vals_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
        }

        s_free(stats, d_sort_scratch);
        s_free(stats, d_vals_out_tile);
        s_free(stats, d_vals_in_tile);
        s_free(stats, d_keys_out_tile);
        s_free(stats, d_t2_mi);
    }

    // Tree-of-2-way-merges: (tile 0 + tile 1) → AB, (tile 2 + tile 3) → CD,
    // then (AB + CD) → final merged stream.
    //
    // Compact: AB + CD live across the final merge → peak ~4160 MB.
    // Sliced: AB and CD parked to USM-host between tree steps so the
    // final merge sees only itself + USM-host inputs (~2080 MB peak).
    uint32_t* d_AB_keys = nullptr;
    uint32_t* d_AB_vals = nullptr;
    uint32_t* d_CD_keys = nullptr;
    uint32_t* d_CD_vals = nullptr;
    uint32_t* h_AB_keys = nullptr;
    uint32_t* h_AB_vals = nullptr;
    uint32_t* h_CD_keys = nullptr;
    uint32_t* h_CD_vals = nullptr;

    if (!t1_match_sliced) {
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

        s_free(stats, d_keys_out);
        s_free(stats, d_vals_out);
    } else {
        // AB merge: read USM-host slices, write device d_AB. Then D2H
        // to USM-host and free device.
        s_malloc(stats, d_AB_keys, ab_count * sizeof(uint32_t), "d_t2_AB_keys");
        s_malloc(stats, d_AB_vals, ab_count * sizeof(uint32_t), "d_t2_AB_vals");
        if (ab_count > 0) {
            launch_merge_pairs_stable_2way_u32_u32(
                h_keys + t2_tile_off[0], h_vals + t2_tile_off[0], t2_tile_n[0],
                h_keys + t2_tile_off[1], h_vals + t2_tile_off[1], t2_tile_n[1],
                d_AB_keys, d_AB_vals, ab_count, q);
        }
        h_AB_keys = static_cast<uint32_t*>(sycl::malloc_host(ab_count * sizeof(uint32_t), q));
        h_AB_vals = static_cast<uint32_t*>(sycl::malloc_host(ab_count * sizeof(uint32_t), q));
        if (!h_AB_keys || !h_AB_vals) throw std::runtime_error("sycl::malloc_host(h_AB) failed");
        if (ab_count > 0) {
            q.memcpy(h_AB_keys, d_AB_keys, ab_count * sizeof(uint32_t));
            q.memcpy(h_AB_vals, d_AB_vals, ab_count * sizeof(uint32_t)).wait();
        }
        s_free(stats, d_AB_vals);
        s_free(stats, d_AB_keys);

        // CD merge: same shape.
        s_malloc(stats, d_CD_keys, cd_count * sizeof(uint32_t), "d_t2_CD_keys");
        s_malloc(stats, d_CD_vals, cd_count * sizeof(uint32_t), "d_t2_CD_vals");
        if (cd_count > 0) {
            launch_merge_pairs_stable_2way_u32_u32(
                h_keys + t2_tile_off[2], h_vals + t2_tile_off[2], t2_tile_n[2],
                h_keys + t2_tile_off[3], h_vals + t2_tile_off[3], t2_tile_n[3],
                d_CD_keys, d_CD_vals, cd_count, q);
        }
        h_CD_keys = static_cast<uint32_t*>(sycl::malloc_host(cd_count * sizeof(uint32_t), q));
        h_CD_vals = static_cast<uint32_t*>(sycl::malloc_host(cd_count * sizeof(uint32_t), q));
        if (!h_CD_keys || !h_CD_vals) throw std::runtime_error("sycl::malloc_host(h_CD) failed");
        if (cd_count > 0) {
            q.memcpy(h_CD_keys, d_CD_keys, cd_count * sizeof(uint32_t));
            q.memcpy(h_CD_vals, d_CD_vals, cd_count * sizeof(uint32_t)).wait();
        }
        s_free(stats, d_CD_vals);
        s_free(stats, d_CD_keys);

        // h_keys + h_vals consumed by AB/CD merges — free.
        sycl::free(h_keys, q); h_keys = nullptr;
        sycl::free(h_vals, q); h_vals = nullptr;
    }

    uint32_t* d_t2_keys_merged = nullptr;   // merged sorted MI for T3.
    uint32_t* d_merged_vals    = nullptr;   // merged sorted src indices.
    s_malloc(stats, d_t2_keys_merged, cap * sizeof(uint32_t), "d_t2_keys_merged");
    s_malloc(stats, d_merged_vals,    cap * sizeof(uint32_t), "d_merged_vals");

    if (!t1_match_sliced) {
        launch_merge_pairs_stable_2way_u32_u32(
            d_AB_keys, d_AB_vals, ab_count,
            d_CD_keys, d_CD_vals, cd_count,
            d_t2_keys_merged, d_merged_vals, t2_count, q);
        s_free(stats, d_AB_keys);
        s_free(stats, d_AB_vals);
        s_free(stats, d_CD_keys);
        s_free(stats, d_CD_vals);
    } else {
        // Final merge from USM-host inputs into device outputs.
        launch_merge_pairs_stable_2way_u32_u32(
            h_AB_keys, h_AB_vals, ab_count,
            h_CD_keys, h_CD_vals, cd_count,
            d_t2_keys_merged, d_merged_vals, t2_count, q);
        sycl::free(h_AB_keys, q); h_AB_keys = nullptr;
        sycl::free(h_AB_vals, q); h_AB_vals = nullptr;
        sycl::free(h_CD_keys, q); h_CD_keys = nullptr;
        sycl::free(h_CD_vals, q); h_CD_vals = nullptr;
    }

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
    int const t2_gather_N = scratch.plain_mode ? 1 : scratch.gather_tile_count;
    uint64_t* d_t2_meta_sorted  = nullptr;
    uint32_t* d_t2_xbits_sorted = nullptr;

    if (t2_gather_N <= 1) {
        // Single-shot path (compact / plain).
        if (!scratch.plain_mode) {
            s_malloc(stats, d_t2_meta, cap * sizeof(uint64_t), "d_t2_meta");
            q.memcpy(d_t2_meta, h_t2_meta, t2_count * sizeof(uint64_t));
            q.wait();
            if (h_meta_owned) sycl::free(h_t2_meta, q);
            h_t2_meta = nullptr;
        }

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

        s_malloc(stats, d_t2_xbits_sorted, cap * sizeof(uint32_t), "d_t2_xbits_sorted");
        launch_gather_u32(d_t2_xbits, d_merged_vals, d_t2_xbits_sorted, t2_count, q);
        end_phase(p_t2_sort);
        s_free(stats, d_t2_xbits);
        s_free(stats, d_merged_vals);
    } else {
        // Tiled-output gather (minimal tier). Both gathers stage their
        // sorted outputs to host pinned (reusing h_t2_meta and
        // h_t2_xbits — same buffers that just held the parked unsorted
        // data) one tile at a time. Crucially, d_t2_meta_sorted is NOT
        // re-allocated on device until BOTH gathers and d_merged_vals
        // are done — otherwise the xbits gather peak (d_t2_meta_sorted
        // 2080 + d_merged_vals 1040 + d_t2_xbits 1040 + tile 260) would
        // still hit ~4420 MB. Deferring the rehydrate keeps the xbits
        // gather peak at d_merged_vals (1040) + d_t2_xbits (1040) +
        // tile (260 at N=4) = ~2340 MB. Final rehydrate peak:
        // d_t2_meta_sorted (2080) + d_t2_xbits_sorted (1040) = 3120 MB.
        uint64_t const tile_max =
            (t2_count + uint64_t(t2_gather_N) - 1) / uint64_t(t2_gather_N);

        // --- Meta gather (tiled output → h_t2_meta) ---
        s_malloc(stats, d_t2_meta, cap * sizeof(uint64_t), "d_t2_meta");
        q.memcpy(d_t2_meta, h_t2_meta, t2_count * sizeof(uint64_t)).wait();
        {
            uint64_t* d_meta_tile = nullptr;
            s_malloc(stats, d_meta_tile, tile_max * sizeof(uint64_t), "d_t2_meta_sorted_tile");
            for (int n = 0; n < t2_gather_N; ++n) {
                uint64_t const tile_off = uint64_t(n) * tile_max;
                if (tile_off >= t2_count) break;
                uint64_t const tile_n = std::min(tile_max, t2_count - tile_off);
                launch_gather_u64(
                    d_t2_meta, d_merged_vals + tile_off,
                    d_meta_tile, tile_n, q);
                q.memcpy(h_t2_meta + tile_off, d_meta_tile,
                         tile_n * sizeof(uint64_t)).wait();
            }
            s_free(stats, d_meta_tile);
        }
        s_free(stats, d_t2_meta);

        // --- Xbits gather (tiled output → h_t2_xbits) ---
        s_malloc(stats, d_t2_xbits, cap * sizeof(uint32_t), "d_t2_xbits");
        q.memcpy(d_t2_xbits, h_t2_xbits, t2_count * sizeof(uint32_t)).wait();
        {
            uint32_t* d_xbits_tile = nullptr;
            s_malloc(stats, d_xbits_tile, tile_max * sizeof(uint32_t), "d_t2_xbits_sorted_tile");
            for (int n = 0; n < t2_gather_N; ++n) {
                uint64_t const tile_off = uint64_t(n) * tile_max;
                if (tile_off >= t2_count) break;
                uint64_t const tile_n = std::min(tile_max, t2_count - tile_off);
                launch_gather_u32(
                    d_t2_xbits, d_merged_vals + tile_off,
                    d_xbits_tile, tile_n, q);
                q.memcpy(h_t2_xbits + tile_off, d_xbits_tile,
                         tile_n * sizeof(uint32_t)).wait();
            }
            s_free(stats, d_xbits_tile);
        }
        s_free(stats, d_t2_xbits);

        // d_merged_vals dead now that both gathers have produced their
        // sorted outputs on host.
        s_free(stats, d_merged_vals);

        // Rehydrate d_t2_xbits_sorted to device (1040 MB at k=28). The
        // T3 match kernel reads d_sorted_xbits[l] / d_sorted_xbits[r]
        // by index and the random-access pattern would be too slow via
        // PCIe with USM-host.
        //
        // Tiny tier: skip the rehydration. h_t2_xbits stays alive across
        // T3 match, and the per-section split kernel H2Ds the section-l
        // + section-r slices into small device buffers each pass. d_t2_
        // xbits_sorted remains nullptr in this path; the T3 match block
        // skips its s_free below.
        if (!scratch.tiny_mode) {
            s_malloc(stats, d_t2_xbits_sorted, cap * sizeof(uint32_t), "d_t2_xbits_sorted");
            q.memcpy(d_t2_xbits_sorted, h_t2_xbits, t2_count * sizeof(uint32_t)).wait();
            if (h_xbits_owned) sycl::free(h_t2_xbits, q);
            h_t2_xbits = nullptr;
        }

        // Site 4: do NOT rehydrate d_t2_meta_sorted to device. h_t2_meta
        // (now containing the sorted meta) stays alive across T3 match;
        // the sliced T3 match path H2Ds a section_l + section_r pair of
        // slices per pass, dropping T3 match peak from
        //   d_t2_meta_sorted (2080) + d_t2_xbits_sorted (1040) +
        //   d_t2_keys_merged (1040) + d_t3_stage (1040) = 5200 MB
        // to
        //   d_meta_l (cap/N_sections × u64 = 520) + d_meta_r (520) +
        //   d_t2_xbits_sorted (1040) + d_t2_keys_merged (1040) +
        //   d_t3_stage (cap/N_sections × u64 = 520) = ~3640 MB at k=28.
        // h_t2_meta is freed inside the T3 match block once all
        // section-pair passes complete.

        end_phase(p_t2_sort);
    }

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
    //
    // Tiny tier: skip the rehydration. h_t2_keys_merged stays alive
    // across T3 match; the split kernel reads section_r's mi slice
    // each pass (d_t2_keys_merged is only used as the binary-search
    // and r-stream input, both indexed within section_r's row range).
    if (!scratch.plain_mode && !scratch.tiny_mode) {
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
    } else if (scratch.gather_tile_count > 1) {
        // Minimal (sliced T3 match — site 4). d_t2_meta_sorted is NOT
        // on device in this path; the sorted meta is parked on
        // h_t2_meta (from the T2 sort tiled gather). For each section_l
        // we H2D the matching pair of sections (l + r) into small
        // device slices, run the kernel against those slices, D2H the
        // stage output to h_t3, then free the slices. Drops T3 match
        // peak from ~5200 MB (compact) to ~3665 MB at k=28.
        //
        // Tiny mode: also park d_t2_xbits_sorted and d_t2_keys_merged
        // on host pinned (they were never rehydrated above). Per
        // section, allocate xbits + mi slices alongside meta slices and
        // call the fully-sliced split kernel. Drops T3 match peak by
        // an additional ~2080 MB at k=28.
        uint32_t const num_sections   = 1u << t3p.num_section_bits;
        uint32_t const num_match_keys = 1u << t3p.num_match_key_bits;
        uint32_t const num_buckets_t3 = num_sections * num_match_keys;
        // Per-pass output capacity sized at cap/N × 1.25 (25% safety
        // margin over the expected uniform-distribution average).
        uint64_t const t3_section_cap =
            ((cap + num_sections - 1) / num_sections) * 5ULL / 4ULL;

        T3PairingGpu* d_t3_stage      = nullptr;
        void*         d_t3_match_temp = nullptr;
        s_malloc(stats, d_t3_stage,      t3_section_cap * sizeof(T3PairingGpu), "d_t3_stage");
        s_malloc(stats, d_t3_match_temp, t3_temp_bytes,                          "d_t3_match_temp");

        bool const h_t3_owned = (scratch.h_t3 == nullptr);
        T3PairingGpu* h_t3 = h_t3_owned
            ? static_cast<T3PairingGpu*>(sycl::malloc_host(cap * sizeof(T3PairingGpu), q))
            : reinterpret_cast<T3PairingGpu*>(scratch.h_t3);
        if (!h_t3) throw std::runtime_error("sycl::malloc_host(h_t3) failed");

        // Compute bucket + fine-bucket offsets in d_t3_match_temp; also
        // zero d_counter. Same call shape as compact path.
        //
        // Tiny mode: d_t2_keys_merged is parked. The prepare kernel
        // needs the sorted mi stream on device for its histogram
        // counts. Briefly rehydrate, run prepare, then free again. The
        // 1040 MB spike is bounded to this prepare phase; the subsequent
        // per-section loop reads only sliced mi.
        uint32_t* d_t2_keys_for_prepare = nullptr;
        if (scratch.tiny_mode) {
            s_malloc(stats, d_t2_keys_for_prepare, cap * sizeof(uint32_t), "d_t2_keys_merged_prep");
            q.memcpy(d_t2_keys_for_prepare, h_t2_keys_merged,
                     t2_count * sizeof(uint32_t)).wait();
        }
        launch_t3_match_prepare(cfg.plot_id.data(), t3p,
                                scratch.tiny_mode ? d_t2_keys_for_prepare
                                                   : d_t2_keys_merged,
                                t2_count,
                                d_counter, d_t3_match_temp, &t3_temp_bytes, q);
        if (scratch.tiny_mode) {
            s_free(stats, d_t2_keys_for_prepare);
            d_t2_keys_for_prepare = nullptr;
        }

        // D2H the bucket-offsets table (small: 17 × u64 at k=28
        // strength=2) so we can compute each section's global row range
        // host-side.
        std::vector<uint64_t> h_t3_offsets(num_buckets_t3 + 1);
        q.memcpy(h_t3_offsets.data(), d_t3_match_temp,
                 (num_buckets_t3 + 1) * sizeof(uint64_t)).wait();

        auto compute_section_r = [&](uint32_t section_l) -> uint32_t {
            // Mirror the kernel's section_l → section_r permutation.
            uint32_t const mask = num_sections - 1u;
            uint32_t const rl   = ((section_l << 1) |
                                   (section_l >> (t3p.num_section_bits - 1))) & mask;
            uint32_t const rl1  = (rl + 1u) & mask;
            return ((rl1 >> 1) |
                    (rl1 << (t3p.num_section_bits - 1))) & mask;
        };

        int p_t3 = begin_phase("T3 match + Feistel");
        uint64_t host_offset = 0;
        for (uint32_t section_l = 0; section_l < num_sections; ++section_l) {
            uint32_t const section_r = compute_section_r(section_l);
            uint64_t const section_l_row_start = h_t3_offsets[section_l * num_match_keys];
            uint64_t const section_l_row_end   = h_t3_offsets[(section_l + 1) * num_match_keys];
            uint64_t const section_l_count     = section_l_row_end - section_l_row_start;
            uint64_t const section_r_row_start = h_t3_offsets[section_r * num_match_keys];
            uint64_t const section_r_row_end   = h_t3_offsets[(section_r + 1) * num_match_keys];
            uint64_t const section_r_count     = section_r_row_end - section_r_row_start;

            // Skip empty sections — happens for tiny test plots where
            // a section has zero rows. The kernel would early-return
            // anyway but the slice malloc rejects bytes==0 since f1d3c67.
            if (section_l_count == 0) continue;

            uint64_t* d_meta_l_slice  = nullptr;
            uint64_t* d_meta_r_slice  = nullptr;
            uint32_t* d_xbits_l_slice = nullptr;
            uint32_t* d_xbits_r_slice = nullptr;
            uint32_t* d_mi_r_slice    = nullptr;
            s_malloc(stats, d_meta_l_slice, section_l_count * sizeof(uint64_t), "d_t3_meta_l_slice");
            if (section_r_count > 0) {
                s_malloc(stats, d_meta_r_slice, section_r_count * sizeof(uint64_t), "d_t3_meta_r_slice");
            }
            if (scratch.tiny_mode) {
                s_malloc(stats, d_xbits_l_slice, section_l_count * sizeof(uint32_t), "d_t3_xbits_l_slice");
                if (section_r_count > 0) {
                    s_malloc(stats, d_xbits_r_slice, section_r_count * sizeof(uint32_t), "d_t3_xbits_r_slice");
                    s_malloc(stats, d_mi_r_slice,    section_r_count * sizeof(uint32_t), "d_t3_mi_r_slice");
                }
            }

            q.memcpy(d_meta_l_slice, h_t2_meta + section_l_row_start,
                     section_l_count * sizeof(uint64_t)).wait();
            if (section_r_count > 0) {
                q.memcpy(d_meta_r_slice, h_t2_meta + section_r_row_start,
                         section_r_count * sizeof(uint64_t)).wait();
            }
            if (scratch.tiny_mode) {
                q.memcpy(d_xbits_l_slice, h_t2_xbits + section_l_row_start,
                         section_l_count * sizeof(uint32_t)).wait();
                if (section_r_count > 0) {
                    q.memcpy(d_xbits_r_slice, h_t2_xbits + section_r_row_start,
                             section_r_count * sizeof(uint32_t)).wait();
                    q.memcpy(d_mi_r_slice, h_t2_keys_merged + section_r_row_start,
                             section_r_count * sizeof(uint32_t)).wait();
                }
            }

            uint32_t const bucket_begin = section_l * num_match_keys;
            uint32_t const bucket_end   = (section_l + 1) * num_match_keys;
            if (scratch.tiny_mode) {
                if (section_r_count > 0) {
                    launch_t3_match_section_pair_split_range(
                        cfg.plot_id.data(), t3p,
                        d_meta_l_slice, d_xbits_l_slice, section_l_row_start,
                        d_meta_r_slice, d_xbits_r_slice, d_mi_r_slice, section_r_row_start,
                        d_t3_stage, d_counter, t3_section_cap,
                        d_t3_match_temp, bucket_begin, bucket_end, q);
                }
                // section_r_count == 0 → no pairings can form; skip kernel
                // (output count for this section is 0).
            } else {
                launch_t3_match_section_pair_range(
                    cfg.plot_id.data(), t3p,
                    d_meta_l_slice, section_l_row_start,
                    d_meta_r_slice, section_r_row_start,
                    d_t2_xbits_sorted, d_t2_keys_merged, t2_count,
                    d_t3_stage, d_counter, t3_section_cap,
                    d_t3_match_temp, bucket_begin, bucket_end, q);
            }

            uint64_t pass_count = 0;
            q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
            if (pass_count > t3_section_cap) {
                throw std::runtime_error(
                    "T3 match (sliced) section_l=" + std::to_string(section_l) +
                    " produced " + std::to_string(pass_count) +
                    " pairs, staging holds " + std::to_string(t3_section_cap) +
                    ". Lower N or widen t3_section_cap safety factor.");
            }
            q.memcpy(h_t3 + host_offset, d_t3_stage,
                     pass_count * sizeof(T3PairingGpu)).wait();
            host_offset += pass_count;
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();

            if (scratch.tiny_mode) {
                if (d_mi_r_slice)    s_free(stats, d_mi_r_slice);
                if (d_xbits_r_slice) s_free(stats, d_xbits_r_slice);
                s_free(stats, d_xbits_l_slice);
            }
            if (section_r_count > 0) s_free(stats, d_meta_r_slice);
            s_free(stats, d_meta_l_slice);
        }
        end_phase(p_t3);

        t3_count = host_offset;
        if (t3_count > cap) throw std::runtime_error("T3 overflow");

        // d_t2_meta_sorted is null in this path (never allocated) — skip
        // its s_free. Free everything else that was alive across T3 match.
        s_free(stats, d_t3_match_temp);
        s_free(stats, d_t3_stage);
        // Tiny: d_t2_xbits_sorted and d_t2_keys_merged are null (parked
        // on host pinned). Free the host buffers instead.
        if (scratch.tiny_mode) {
            if (h_xbits_owned) sycl::free(h_t2_xbits, q);
            h_t2_xbits = nullptr;
            if (h_keys_owned)  sycl::free(h_t2_keys_merged, q);
            h_t2_keys_merged = nullptr;
        } else {
            s_free(stats, d_t2_xbits_sorted);
            s_free(stats, d_t2_keys_merged);
        }

        // h_t2_meta was kept alive across T3 match for slicing; free now
        // that all section pairs have been H2D'd.
        if (h_meta_owned) sycl::free(h_t2_meta, q);
        h_t2_meta = nullptr;

        // Re-hydrate full-cap d_t3 on device for T3 sort.
        s_malloc(stats, d_t3, cap * sizeof(T3PairingGpu), "d_t3");
        q.memcpy(d_t3, h_t3, t3_count * sizeof(T3PairingGpu)).wait();
        if (h_t3_owned) sycl::free(h_t3, q);
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
    // Compact / plain: full-cap CUB sort_keys with separate keys_in
    // (= d_t3) and keys_out (= d_frags_out) buffers — peaks at
    // 2 × cap × u64 + scratch ≈ 4228 MB at k=28.
    //
    // Minimal: tile the sort in halves with a single cap/2 output
    // buffer, D2H each tile to host pinned, std::inplace_merge on
    // host, then H2D the merged result back into the full-cap
    // d_frags_out the D2H phase below expects. Drops T3 sort peak to
    // ~3152 MB at k=28 (d_t3 2080 + tile output 1040 + sort scratch
    // sized for cap/2 ≈ 32). Adds one cap-sized PCIe round-trip per
    // plot.
    stats.phase = "T3 sort";
    uint64_t* d_frags_in  = reinterpret_cast<uint64_t*>(d_t3);
    uint64_t* d_frags_out = nullptr;

    if (!t1_match_sliced) {
        size_t t3_sort_bytes = 0;
        launch_sort_keys_u64(
            nullptr, t3_sort_bytes,
            static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
            cap, 0, 2 * cfg.k, q);

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
    } else {
        // Tiled sort + host merge.
        uint64_t const tile_max = (cap + 1) / 2;
        uint64_t const tile_n0  = t3_count / 2;
        uint64_t const tile_n1  = t3_count - tile_n0;

        size_t t3_tile_sort_bytes = 0;
        launch_sort_keys_u64(
            nullptr, t3_tile_sort_bytes,
            static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
            tile_max, 0, 2 * cfg.k, q);

        uint64_t* d_frags_out_tile     = nullptr;
        void*     d_sort_scratch_tile  = nullptr;
        s_malloc(stats, d_frags_out_tile,    tile_max * sizeof(uint64_t), "d_frags_out_tile");
        s_malloc(stats, d_sort_scratch_tile, t3_tile_sort_bytes,          "d_sort_scratch(t3_tile)");

        uint64_t* h_frags = static_cast<uint64_t*>(
            sycl::malloc_host(cap * sizeof(uint64_t), q));
        if (!h_frags) throw std::runtime_error("sycl::malloc_host(h_frags) failed");

        int p_t3_sort = begin_phase("T3 sort");
        if (tile_n0 > 0) {
            launch_sort_keys_u64(
                d_sort_scratch_tile, t3_tile_sort_bytes,
                d_frags_in, d_frags_out_tile,
                tile_n0, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
            q.memcpy(h_frags, d_frags_out_tile,
                     tile_n0 * sizeof(uint64_t)).wait();
        }
        if (tile_n1 > 0) {
            launch_sort_keys_u64(
                d_sort_scratch_tile, t3_tile_sort_bytes,
                d_frags_in + tile_n0, d_frags_out_tile,
                tile_n1, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
            q.memcpy(h_frags + tile_n0, d_frags_out_tile,
                     tile_n1 * sizeof(uint64_t)).wait();
        }
        end_phase(p_t3_sort);

        s_free(stats, d_frags_out_tile);
        s_free(stats, d_sort_scratch_tile);
        s_free(stats, d_t3);

        // Stable in-place merge of [0, tile_n0) and [tile_n0, t3_count)
        // — both halves are individually sorted by launch_sort_keys_u64.
        std::inplace_merge(h_frags, h_frags + tile_n0, h_frags + t3_count);

        // Re-hydrate full-cap d_frags_out for the existing D2H phase.
        s_malloc(stats, d_frags_out, cap * sizeof(uint64_t), "d_frags_out");
        if (t3_count > 0) {
            q.memcpy(d_frags_out, h_frags, t3_count * sizeof(uint64_t)).wait();
        }
        sycl::free(h_frags, q);
    }

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
