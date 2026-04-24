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
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pos2gpu {

namespace {

// Variadic so the preprocessor does not split on template-argument commas
// (e.g. cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(...)).
#define CHECK(...) do {                                                  \
    cudaError_t err = (__VA_ARGS__);                                     \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(std::string("CUDA: ") +                 \
                                 cudaGetErrorString(err));               \
    }                                                                    \
} while (0)

// =====================================================================
// T1 sort: by match_info, low k bits, stable. Uses CUB SortPairs with
// (key=match_info, value=index) then permutes T1Pairings.
// =====================================================================

// Permute the T1 match output by sort indices, writing only the 8-byte
// meta (meta_hi << 32 | meta_lo). match_info already lives in the sort's
// key-output stream so we don't rematerialise it; the T2 match kernel
// consumes (sorted_meta, sorted_mi) directly.
__global__ void permute_t1(
    T1PairingGpu const* __restrict__ src,
    uint32_t const* __restrict__ indices,
    uint64_t* __restrict__ dst_meta,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    T1PairingGpu s = src[indices[idx]];
    dst_meta[idx] = (uint64_t(s.meta_hi) << 32) | uint64_t(s.meta_lo);
}

__global__ void extract_t1_keys(
    T1PairingGpu const* __restrict__ src,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ vals_out,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    keys_out[idx] = src[idx].match_info;
    vals_out[idx] = uint32_t(idx);
}

// =====================================================================
// T2 sort: same shape — sort indices by match_info.
// =====================================================================

// T3 match reads meta (8 B) and x_bits (4 B) from sorted_t2 but does not
// touch match_info (passed as the parallel sorted_mi stream). Splitting
// the sort output into meta[] and xbits[] arrays drops the per-access
// line footprint from 16 B to 12 B, cutting L1/TEX line fetches on an
// L1-throughput-bound kernel.
//
// Reads SoA input (src_meta/src_xbits) since T2 match emits SoA.
__global__ void permute_t2(
    uint64_t const* __restrict__ src_meta,
    uint32_t const* __restrict__ src_xbits,
    uint32_t const* __restrict__ indices,
    uint64_t* __restrict__ dst_meta,
    uint32_t* __restrict__ dst_xbits,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    uint32_t i = indices[idx];
    dst_meta[idx]  = src_meta[i];
    dst_xbits[idx] = src_xbits[i];
}

// Fills vals[i] = i — used in place of the old extract_t2_keys, now
// that T2 match emits match_info directly as a SoA stream (no need to
// pull it out of a struct on host).
__global__ void init_u32_identity(uint32_t* __restrict__ vals, uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    vals[idx] = uint32_t(idx);
}

// Gather-by-index helpers. Used to split the fused merge-permute into
// merge + per-column gather, letting the streaming path free the source
// column between gather passes and shrink the peak VRAM window.
__global__ void gather_u64(uint64_t const* __restrict__ src,
                           uint32_t const* __restrict__ indices,
                           uint64_t* __restrict__ dst, uint64_t count)
{
    uint64_t p = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (p >= count) return;
    dst[p] = src[indices[p]];
}

__global__ void gather_u32(uint32_t const* __restrict__ src,
                           uint32_t const* __restrict__ indices,
                           uint32_t* __restrict__ dst, uint64_t count)
{
    uint64_t p = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (p >= count) return;
    dst[p] = src[indices[p]];
}



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
    // succeeded), this dtor runs on unwind and releases the still-live
    // device buffers instead of leaking them across batch iterations.
    // Without this, an 8 GB card hitting OOM at k=28 leaked ~130 GB of
    // host-side pinned accounting per failed batch retry.
    ~StreamingStats() {
        if (sizes.empty()) return;
        for (auto& [ptr, _bytes] : sizes) {
            if (ptr) cudaFree(ptr);
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
    void* p = nullptr;
    cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc(") + reason + "): " + cudaGetErrorString(err) +
            " — phase=" + s.phase +
            " requested=" + std::to_string(bytes >> 20) +
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
    cudaFree(raw);
    ptr = nullptr;
}

// =====================================================================
// Stable 2-way merge of two sorted (key, value) runs — used by the
// streaming path to recombine per-tile CUB sort outputs into a single
// sorted stream. Stability (A wins on ties) is load-bearing: the pool
// path's single CUB radix sort is stable, and we want the merged
// streaming output to be bit-identical to it for parity testing.
//
// Algorithm: per-thread binary merge-path (Odeh/Green/Bader). Each output
// position p independently locates the path partition (i, j) with
// i + j = p such that A[i-1] <= B[j] and B[j-1] < A[i], then emits
// A[i] or B[j] — whichever is smaller, with A winning ties.
//
// Work is O(total × log total) — not linear. That is fine at k=18 (a few
// hundred microseconds) and bearable at k=28; a block-cooperative
// linear-work version is the natural Phase 6 upgrade if merge time
// becomes the bottleneck.
// =====================================================================
template <typename K, typename V>
__global__ void merge_pairs_stable_2way(
    K const* __restrict__ A_keys, V const* __restrict__ A_vals, uint64_t nA,
    K const* __restrict__ B_keys, V const* __restrict__ B_vals, uint64_t nB,
    K* __restrict__ out_keys, V* __restrict__ out_vals, uint64_t total)
{
    uint64_t p = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (p >= total) return;

    // i in [max(0, p-nB), min(p, nA)]. Upper-biased midpoint so the loop
    // converges to `lo = i` (not lo = i+1), letting us index A[i-1]
    // unconditionally inside the body.
    uint64_t lo = (p > nB) ? (p - nB) : 0;
    uint64_t hi = (p < nA) ? p : nA;
    while (lo < hi) {
        uint64_t i = lo + (hi - lo + 1) / 2;  // i in [lo+1, hi]
        uint64_t j = p - i;
        K a_prev = A_keys[i - 1];
        K b_here = (j < nB) ? B_keys[j] : K(~K(0));
        if (a_prev > b_here) {
            hi = i - 1;       // consumed too many from A
        } else {
            lo = i;
        }
    }
    uint64_t i = lo;
    uint64_t j = p - i;

    bool take_a;
    if (i >= nA)      take_a = false;
    else if (j >= nB) take_a = true;
    else              take_a = A_keys[i] <= B_keys[j];  // A wins ties → stable

    if (take_a) {
        out_keys[p] = A_keys[i];
        out_vals[p] = A_vals[i];
    } else {
        out_keys[p] = B_keys[j];
        out_vals[p] = B_vals[j];
    }
}

// =====================================================================
// Fused merge-path + permute kernels.
//
// The streaming pipeline does (tile-sort → merge → permute) in three
// passes. The merge pass only exists to materialise merged (keys, vals)
// arrays that the permute pass then consumes. Fusing merge with permute
// lets us skip materialising `merged_vals` entirely — each thread
// computes its merge-path winner, then gathers src[winner].meta
// directly and writes it to the permuted meta stream.
//
// The win is that `d_vals_in` (or equivalent) can be freed before the
// fused kernel runs, reclaiming ~1 GB at k=28. See
// docs/streaming-pipeline-design.md Phase 6 section for the budget.
//
// merged_keys is still written out (downstream match kernels want
// match_info as a separate slim stream for binary search) — that slot
// aliases the CUB extract-input buffer, which is dead by the time the
// fused kernel runs.
// =====================================================================
__global__ void merge_permute_t1(
    uint32_t const* __restrict__ A_keys, uint32_t const* __restrict__ A_vals, uint64_t nA,
    uint32_t const* __restrict__ B_keys, uint32_t const* __restrict__ B_vals, uint64_t nB,
    uint64_t const* __restrict__ src_meta,
    uint32_t* __restrict__ out_keys, uint64_t* __restrict__ out_meta, uint64_t total)
{
    uint64_t p = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (p >= total) return;

    uint64_t lo = (p > nB) ? (p - nB) : 0;
    uint64_t hi = (p < nA) ? p : nA;
    while (lo < hi) {
        uint64_t i = lo + (hi - lo + 1) / 2;
        uint64_t j = p - i;
        uint32_t a_prev = A_keys[i - 1];
        uint32_t b_here = (j < nB) ? B_keys[j] : 0xFFFFFFFFu;
        if (a_prev > b_here) hi = i - 1;
        else                 lo = i;
    }
    uint64_t i = lo;
    uint64_t j = p - i;

    bool take_a;
    if (i >= nA)      take_a = false;
    else if (j >= nB) take_a = true;
    else              take_a = A_keys[i] <= B_keys[j];

    uint32_t val; uint32_t key;
    if (take_a) { val = A_vals[i]; key = A_keys[i]; }
    else        { val = B_vals[j]; key = B_keys[j]; }

    out_keys[p] = key;
    out_meta[p] = src_meta[val];
}

__global__ void merge_permute_t2(
    uint32_t const* __restrict__ A_keys, uint32_t const* __restrict__ A_vals, uint64_t nA,
    uint32_t const* __restrict__ B_keys, uint32_t const* __restrict__ B_vals, uint64_t nB,
    uint64_t const* __restrict__ src_meta,
    uint32_t const* __restrict__ src_xbits,
    uint32_t* __restrict__ out_keys,
    uint64_t* __restrict__ out_meta, uint32_t* __restrict__ out_xbits,
    uint64_t total)
{
    uint64_t p = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (p >= total) return;

    uint64_t lo = (p > nB) ? (p - nB) : 0;
    uint64_t hi = (p < nA) ? p : nA;
    while (lo < hi) {
        uint64_t i = lo + (hi - lo + 1) / 2;
        uint64_t j = p - i;
        uint32_t a_prev = A_keys[i - 1];
        uint32_t b_here = (j < nB) ? B_keys[j] : 0xFFFFFFFFu;
        if (a_prev > b_here) hi = i - 1;
        else                 lo = i;
    }
    uint64_t i = lo;
    uint64_t j = p - i;

    bool take_a;
    if (i >= nA)      take_a = false;
    else if (j >= nB) take_a = true;
    else              take_a = A_keys[i] <= B_keys[j];

    uint32_t val; uint32_t key;
    if (take_a) { val = A_vals[i]; key = A_keys[i]; }
    else        { val = B_vals[j]; key = B_keys[j]; }

    out_keys[p]  = key;
    out_meta[p]  = src_meta[val];
    out_xbits[p] = src_xbits[val];
}

} // namespace

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool,
                                   int pinned_index)
{
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

    cudaStream_t stream = nullptr; // default stream

    // ---- pool aliases ----
    // d_pair_a carries the "current phase match output": T1, then T2, then T3.
    // d_pair_b carries the "current phase sort output": sorted T1, sorted T2,
    // then final uint64_t fragments. Each subsequent phase's output overwrites
    // the previous (consumed) contents in the same slot.
    XsCandidateGpu* d_xs             = static_cast<XsCandidateGpu*>(pool.d_storage);
    // T1 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B) then mi[cap] (cap·4 B). Total cap·12 B, fits in d_pair_a's
    // cap·16 B budget.
    uint64_t*       d_t1_meta = static_cast<uint64_t*>(pool.d_pair_a);
    uint32_t*       d_t1_mi   = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(pool.d_pair_a) + pool.cap * sizeof(uint64_t));
    // Sorted T1 is now just meta (8 B/entry) — match_info comes from sort keys.
    uint64_t*       d_t1_meta_sorted = static_cast<uint64_t*>      (pool.d_pair_b);
    // T2 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B), then mi[cap] (cap·4 B), then xbits[cap] (cap·4 B). Total
    // cap·16 B, matching d_pair_a's size.
    uint64_t*       d_t2_meta  = static_cast<uint64_t*>(pool.d_pair_a);
    uint32_t*       d_t2_mi    = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(pool.d_pair_a) + pool.cap * sizeof(uint64_t));
    uint32_t*       d_t2_xbits = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(pool.d_pair_a) + pool.cap * (sizeof(uint64_t) + sizeof(uint32_t)));
    // Sorted T2 is SoA-split across d_pair_b: meta[cap] then xbits[cap],
    // 12 B total per entry (fits in d_pair_b's 16 B/entry budget). T3
    // match reads both; frags_out later reuses d_pair_b from offset 0.
    uint64_t*       d_t2_meta_sorted  = static_cast<uint64_t*>      (pool.d_pair_b);
    uint32_t*       d_t2_xbits_sorted = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(pool.d_pair_b) + pool.cap * sizeof(uint64_t));
    T3PairingGpu*   d_t3             = static_cast<T3PairingGpu*>  (pool.d_pair_a);
    uint64_t*       d_frags_out      = static_cast<uint64_t*>      (pool.d_pair_b);

    uint64_t*       d_count        = pool.d_counter;
    // Xs phase needs ~4.34 GB scratch at k=28; d_pair_b is idle through
    // the whole Xs phase (not touched until T1 sort permute writes to it),
    // so we alias it rather than allocating separately.
    void*           d_xs_temp      = pool.d_pair_b;
    void*           d_sort_scratch = pool.d_sort_scratch;
    uint64_t*       h_pinned_t3    = pool.h_pinned_t3[pinned_index];
    // T1/T2/T3 match kernels report 0 scratch bytes, but some CUDA paths
    // reject a nullptr d_temp_storage with cudaErrorInvalidArgument even
    // when bytes==0. Point them at d_sort_scratch (idle during match) to
    // give the kernel a valid non-null handle.
    void*           d_match_temp   = pool.d_sort_scratch;

    // Sort key/val arrays alias d_storage. Safe because Xs is fully consumed
    // by T1 match (stream-synchronised) before we enter T1 sort.
    auto     storage_u32 = static_cast<uint32_t*>(pool.d_storage);
    uint32_t* d_keys_in  = storage_u32 + 0 * cap;
    uint32_t* d_keys_out = storage_u32 + 1 * cap;
    uint32_t* d_vals_in  = storage_u32 + 2 * cap;
    uint32_t* d_vals_out = storage_u32 + 3 * cap;

    // ---- profiling: cudaEvent helpers ----
    struct PhaseTimer {
        cudaEvent_t start, stop;
        std::string label;
    };
    std::vector<PhaseTimer> phases;
    auto begin_phase = [&](char const* label) -> int {
        if (!cfg.profile) return -1;
        PhaseTimer pt;
        pt.label = label;
        cudaEventCreate(&pt.start);
        cudaEventCreate(&pt.stop);
        cudaEventRecord(pt.start, stream);
        phases.push_back(pt);
        return int(phases.size()) - 1;
    };
    auto end_phase = [&](int idx) {
        if (!cfg.profile || idx < 0) return;
        cudaEventRecord(phases[idx].stop, stream);
    };
    auto report_phases = [&]() {
        if (!cfg.profile) return;
        cudaDeviceSynchronize();
        std::fprintf(stderr, "=== gpu_pipeline phase breakdown ===\n");
        float total_ms = 0;
        for (auto& pt : phases) {
            float ms = 0;
            cudaEventElapsedTime(&ms, pt.start, pt.stop);
            std::fprintf(stderr, "  %-30s %8.2f ms\n", pt.label.c_str(), ms);
            total_ms += ms;
            cudaEventDestroy(pt.start);
            cudaEventDestroy(pt.stop);
        }
        std::fprintf(stderr, "  %-30s %8.2f ms\n", "TOTAL device time:", total_ms);
    };

    // ---------- Phase Xs ----------
    size_t xs_temp_bytes = 0;
    CHECK(launch_construct_xs(cfg.plot_id.data(), cfg.k, cfg.testnet,
                              nullptr, nullptr, &xs_temp_bytes));
    cudaEvent_t e_xs_start = nullptr, e_xs_gen_done = nullptr, e_xs_sort_done = nullptr;
    if (cfg.profile) {
        cudaEventCreate(&e_xs_start);
        cudaEventCreate(&e_xs_gen_done);
        cudaEventCreate(&e_xs_sort_done);
        cudaEventRecord(e_xs_start, stream);
    }
    CHECK(launch_construct_xs_profiled(cfg.plot_id.data(), cfg.k, cfg.testnet,
                                       d_xs, d_xs_temp, &xs_temp_bytes,
                                       e_xs_gen_done, e_xs_sort_done, stream));

    // ---------- Phase T1 ----------
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    CHECK(launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_count, cap,
                          nullptr, &t1_temp_bytes));
    CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));
    int p_t1 = begin_phase("T1 match");
    CHECK(launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1_meta, d_t1_mi, d_count, cap,
                          d_match_temp, &t1_temp_bytes, stream));
    end_phase(p_t1);

    // No explicit sync: the next cudaMemcpy (non-async, default stream)
    // implicitly drains prior stream work before the host reads t1_count.
    uint64_t t1_count = 0;
    CHECK(cudaMemcpy(&t1_count, d_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t1_count > cap) throw std::runtime_error("T1 overflow");


    // Sort T1 by match_info (low k bits). d_storage is now repurposed
    // as (keys_in, keys_out, vals_in, vals_out), Xs having been fully
    // consumed by T1 match above. T1 match emits match_info in a SoA
    // stream (d_t1_mi), so we feed that directly to CUB as the sort key
    // input rather than extracting from a packed struct.
    int p_t1_sort = begin_phase("T1 sort");
    {
        init_u32_identity<<<blocks(t1_count), kThreads, 0, stream>>>(
            d_vals_in, t1_count);
        CHECK(cudaGetLastError());

        size_t sort_bytes = pool.sort_scratch_bytes;
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, sort_bytes,
            d_t1_mi, d_keys_out, d_vals_in, d_vals_out,
            t1_count, /*begin_bit=*/0, /*end_bit=*/cfg.k, stream));

        gather_u64<<<blocks(t1_count), kThreads, 0, stream>>>(
            d_t1_meta, d_vals_out, d_t1_meta_sorted, t1_count);
        CHECK(cudaGetLastError());
    }
    end_phase(p_t1_sort);

    // ---------- Phase T2 ----------
    // Sorted T1 = (d_t1_meta_sorted: uint64 meta, d_keys_out: uint32 match_info).
    // No AoS struct anymore — saves 33 % of sorted-T1 bandwidth on both the
    // permute write and the match-kernel hot path.
    auto t2p = make_t2_params(cfg.k, cfg.strength);
    size_t t2_temp_bytes = 0;
    CHECK(launch_t2_match(cfg.plot_id.data(), t2p, nullptr, nullptr, t1_count,
                          nullptr, nullptr, nullptr, d_count, cap,
                          nullptr, &t2_temp_bytes));
    CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));
    int p_t2 = begin_phase("T2 match");
    CHECK(launch_t2_match(cfg.plot_id.data(), t2p, d_t1_meta_sorted, d_keys_out, t1_count,
                          d_t2_meta, d_t2_mi, d_t2_xbits, d_count, cap,
                          d_match_temp, &t2_temp_bytes, stream));
    end_phase(p_t2);

    uint64_t t2_count = 0;
    CHECK(cudaMemcpy(&t2_count, d_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t2_count > cap) throw std::runtime_error("T2 overflow");

    int p_t2_sort = begin_phase("T2 sort");
    {
        // T2 match emitted match_info as a SoA stream (d_t2_mi) — feed
        // it straight into CUB as the sort key input rather than
        // re-extracting from a packed struct. vals_in just needs a
        // 0..n-1 identity fill.
        init_u32_identity<<<blocks(t2_count), kThreads, 0, stream>>>(
            d_vals_in, t2_count);
        CHECK(cudaGetLastError());

        size_t sort_bytes = pool.sort_scratch_bytes;
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, sort_bytes,
            d_t2_mi, d_keys_out, d_vals_in, d_vals_out,
            t2_count, 0, cfg.k, stream));

        permute_t2<<<blocks(t2_count), kThreads, 0, stream>>>(
            d_t2_meta, d_t2_xbits, d_vals_out,
            d_t2_meta_sorted, d_t2_xbits_sorted, t2_count);
        CHECK(cudaGetLastError());
    }
    end_phase(p_t2_sort);

    // ---------- Phase T3 ----------
    // d_keys_out now holds the T2 sorted match_info (T1's was overwritten by
    // the T2 sort above) — pass as the slim stream for binary search in T3.
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          nullptr, t2_count,
                          d_t3, d_count, cap,
                          nullptr, &t3_temp_bytes));
    CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));
    int p_t3 = begin_phase("T3 match + Feistel");
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          d_keys_out, t2_count,
                          d_t3, d_count, cap,
                          d_match_temp, &t3_temp_bytes, stream));
    end_phase(p_t3);

    uint64_t t3_count = 0;
    CHECK(cudaMemcpy(&t3_count, d_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t3_count > cap) throw std::runtime_error("T3 overflow");

    // Sort T3 by proof_fragment (low 2k bits). T3PairingGpu is just a
    // uint64_t, so reinterpret the d_pair_a slot directly.
    uint64_t* d_frags_in = reinterpret_cast<uint64_t*>(d_t3);
    int p_t3_sort = begin_phase("T3 sort");
    {
        size_t sort_bytes = pool.sort_scratch_bytes;
        CHECK(cub::DeviceRadixSort::SortKeys(
            d_sort_scratch, sort_bytes,
            d_frags_in, d_frags_out,
            t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, stream));
    }
    end_phase(p_t3_sort);

    // ---------- D2H ----------
    int p_d2h = begin_phase("D2H copy T3 fragments (pinned)");
    GpuPipelineResult result;
    result.t1_count = t1_count;
    result.t2_count = t2_count;
    result.t3_count = t3_count;

    if (t3_count > 0) {
        CHECK(cudaMemcpyAsync(h_pinned_t3, d_frags_out,
                              sizeof(uint64_t) * t3_count,
                              cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
    }
    end_phase(p_d2h);

    if (t3_count > 0) {
        // Borrow: caller (batch producer) promises to finish consuming this
        // pinned slot before reusing it for another plot.
        result.external_fragments_ptr   = h_pinned_t3;
        result.external_fragments_count = t3_count;
    }

    // Inject Xs gen / sort timings before reporting (avoids the double-event
    // ownership headache by handling them out-of-band here).
    if (cfg.profile) {
        cudaDeviceSynchronize();
        float gen_ms = 0, sort_ms = 0;
        cudaEventElapsedTime(&gen_ms,  e_xs_start,    e_xs_gen_done);
        cudaEventElapsedTime(&sort_ms, e_xs_gen_done, e_xs_sort_done);
        std::fprintf(stderr, "  %-30s %8.2f ms\n", "Xs gen (g_x)", gen_ms);
        std::fprintf(stderr, "  %-30s %8.2f ms\n", "Xs sort", sort_ms);
        cudaEventDestroy(e_xs_start);
        cudaEventDestroy(e_xs_gen_done);
        cudaEventDestroy(e_xs_sort_done);
    }

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
    StreamingPinnedScratch const& scratch);     // any nullptr field → plain streaming for that buffer

} // namespace

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg)
{
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

    cudaStream_t stream = nullptr;  // default stream

    StreamingStats stats;
    s_init_from_env(stats);

    // --- pipeline-wide tiny allocations ---
    // d_counter: per-phase uint64 count output (reused).
    // The match kernels each need their own temp-storage buffer sized via
    // their size query; we allocate it per-phase rather than globally so
    // that the peak VRAM is the phase's alone.
    stats.phase = "init";
    uint64_t* d_counter = nullptr;
    s_malloc(stats, d_counter, sizeof(uint64_t), "d_counter");

    // ---------- Phase Xs (inlined gen+sort+pack) ----------
    // launch_construct_xs bundles keys_a/b + vals_a/b inside a single
    // d_xs_temp blob (~4 GB at k=28), so keys_a + vals_a are kept alive
    // through pack even though they're dead after CUB sort. Inline the
    // three sub-kernels with separate s_malloc per buffer so we can:
    //
    //   1. alloc keys_a + vals_a
    //   2. launch_xs_gen -> keys_a, vals_a
    //   3. alloc keys_b + vals_b + cub_scratch
    //   4. cub::DeviceRadixSort::SortPairs: a -> b
    //   5. free keys_a + vals_a + cub_scratch    <- 2 GB freed
    //   6. alloc d_xs
    //   7. launch_xs_pack: keys_b, vals_b -> d_xs
    //   8. free keys_b + vals_b
    //
    // Phase peak at k=28 drops from d_xs (2048) + d_xs_temp (4096) =
    // 6144 MB to max(sort ~4080 MB, pack ~4080 MB) ≈ 4080 MB. Zero
    // extra PCIe cost (purely a lifetime rearrangement of on-device
    // buffers).
    stats.phase = "Xs";

    uint32_t* d_xs_keys_a = nullptr;
    uint32_t* d_xs_vals_a = nullptr;
    s_malloc(stats, d_xs_keys_a, total_xs * sizeof(uint32_t), "d_xs_keys_a");
    s_malloc(stats, d_xs_vals_a, total_xs * sizeof(uint32_t), "d_xs_vals_a");

    CHECK(launch_xs_gen(cfg.plot_id.data(), cfg.k, cfg.testnet,
                        d_xs_keys_a, d_xs_vals_a));

    // CUB DoubleBuffer mode: ping-pongs between keys_a/keys_b so the
    // internal scratch shrinks from ~2 GB at k=28 down to ~MB of
    // histograms (matches launch_construct_xs and the streaming
    // T1/T2/T3 sorts). Caller canonicalises to keys_b/vals_b via a
    // conditional D2D copy if CUB landed in the a side.
    uint32_t* d_xs_keys_b = nullptr;
    uint32_t* d_xs_vals_b = nullptr;
    s_malloc(stats, d_xs_keys_b, total_xs * sizeof(uint32_t), "d_xs_keys_b");
    s_malloc(stats, d_xs_vals_b, total_xs * sizeof(uint32_t), "d_xs_vals_b");

    size_t xs_cub_bytes = 0;
    {
        cub::DoubleBuffer<uint32_t> probe_keys(nullptr, nullptr);
        cub::DoubleBuffer<uint32_t> probe_vals(nullptr, nullptr);
        CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, xs_cub_bytes,
            probe_keys, probe_vals,
            total_xs, /*begin_bit=*/0, /*end_bit=*/cfg.k));
    }
    void* d_xs_cub_scratch = nullptr;
    s_malloc(stats, d_xs_cub_scratch, xs_cub_bytes, "d_xs_cub");

    {
        cub::DoubleBuffer<uint32_t> dk(d_xs_keys_a, d_xs_keys_b);
        cub::DoubleBuffer<uint32_t> dv(d_xs_vals_a, d_xs_vals_b);
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_xs_cub_scratch, xs_cub_bytes,
            dk, dv,
            total_xs, /*begin_bit=*/0, /*end_bit=*/cfg.k));
        // Canonicalise sorted result into keys_b/vals_b so downstream
        // free-before-pack treats keys_a/vals_a as the disposable pair.
        if (dk.Current() != d_xs_keys_b) {
            CHECK(cudaMemcpyAsync(d_xs_keys_b, dk.Current(),
                total_xs * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }
        if (dv.Current() != d_xs_vals_b) {
            CHECK(cudaMemcpyAsync(d_xs_vals_b, dv.Current(),
                total_xs * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }
    }

    // Sort consumed keys_a + vals_a; free them + CUB scratch before
    // allocating d_xs so the pack phase peak stays under the sort peak.
    s_free(stats, d_xs_cub_scratch);
    s_free(stats, d_xs_keys_a);
    s_free(stats, d_xs_vals_a);

    XsCandidateGpu* d_xs = nullptr;
    s_malloc(stats, d_xs, total_xs * sizeof(XsCandidateGpu), "d_xs");

    CHECK(launch_xs_pack(d_xs_keys_b, d_xs_vals_b, d_xs, total_xs));

    s_free(stats, d_xs_keys_b);
    s_free(stats, d_xs_vals_b);

    // ---------- Phase T1 match ----------
    stats.phase = "T1 match";
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    CHECK(launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_counter, cap,
                          nullptr, &t1_temp_bytes));
    // SoA output: meta (uint64) + mi (uint32). Same 12 B/pair as the old
    // AoS struct, but the two streams can be freed independently — we
    // drop d_t1_mi as soon as CUB consumes it in the T1 sort phase.
    uint64_t* d_t1_meta = nullptr;
    uint32_t* d_t1_mi   = nullptr;
    void*     d_t1_match_temp = nullptr;
    s_malloc(stats, d_t1_meta,        cap * sizeof(uint64_t), "d_t1_meta");
    s_malloc(stats, d_t1_mi,          cap * sizeof(uint32_t), "d_t1_mi");
    s_malloc(stats, d_t1_match_temp,  t1_temp_bytes,          "d_t1_match_temp");

    CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint64_t), stream));
    CHECK(launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1_meta, d_t1_mi, d_counter, cap,
                          d_t1_match_temp, &t1_temp_bytes, stream));

    uint64_t t1_count = 0;
    CHECK(cudaMemcpy(&t1_count, d_counter, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t1_count > cap) throw std::runtime_error("T1 overflow");

    s_free(stats, d_t1_match_temp);
    // Xs fully consumed.
    s_free(stats, d_xs);

    // Compact-streaming: park d_t1_meta on pinned host across T1 sort.
    // The sort only needs d_t1_mi as key; d_t1_meta is only consumed
    // by the final gather_u64 at the end of this phase. Parking drops
    // the T1 sort live set by ~2 GB at k=28. No-op when scratch.h_meta
    // is nullptr (plain streaming).
    if (scratch.h_meta) {
        CHECK(cudaMemcpyAsync(scratch.h_meta, d_t1_meta,
                              t1_count * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, stream));
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

    // CUB DoubleBuffer mode: caller's two buffers act as the radix
    // ping-pong, so CUB's internal scratch shrinks from ~2 GB at k=28
    // down to ~MB of histograms. Frees enough VRAM to fit on 8 GB cards.
    size_t t1_sort_bytes = 0;
    {
        cub::DoubleBuffer<uint32_t> probe_keys(nullptr, nullptr);
        cub::DoubleBuffer<uint32_t> probe_vals(nullptr, nullptr);
        CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, t1_sort_bytes,
            probe_keys, probe_vals,
            t1_tile_max, 0, cfg.k, stream));
    }

    stats.phase = "T1 sort";
    // With T1 SoA emission, d_t1_mi IS the CUB key input. With DoubleBuffer
    // mode CUB ping-pongs between d_t1_mi and d_keys_out, so the result
    // may land in either; we cudaMemcpyAsync to d_keys_out when it lands
    // in d_t1_mi to keep downstream code unchanged.
    uint32_t* d_keys_out     = nullptr;
    uint32_t* d_vals_in      = nullptr;
    uint32_t* d_vals_out     = nullptr;
    void*     d_sort_scratch = nullptr;
    s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
    s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
    s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
    s_malloc(stats, d_sort_scratch, t1_sort_bytes,          "d_sort_scratch(t1)");

    init_u32_identity<<<blocks(t1_count), kThreads, 0, stream>>>(
        d_vals_in, t1_count);
    CHECK(cudaGetLastError());

    auto sort_t1_tile = [&](uint64_t off, uint64_t n) {
        if (n == 0) return;
        cub::DoubleBuffer<uint32_t> dk(d_t1_mi + off, d_keys_out + off);
        cub::DoubleBuffer<uint32_t> dv(d_vals_in + off, d_vals_out + off);
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, t1_sort_bytes,
            dk, dv,
            n, /*begin_bit=*/0, /*end_bit=*/cfg.k, stream));
        if (dk.Current() != d_keys_out + off) {
            CHECK(cudaMemcpyAsync(d_keys_out + off, dk.Current(),
                n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
        }
        if (dv.Current() != d_vals_out + off) {
            CHECK(cudaMemcpyAsync(d_vals_out + off, dv.Current(),
                n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
        }
    };
    sort_t1_tile(0, t1_tile_n0);
    sort_t1_tile(t1_tile_n0, t1_tile_n1);

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

    merge_pairs_stable_2way<<<blocks(t1_count), kThreads, 0, stream>>>(
        d_keys_out + 0,          d_vals_out + 0,          t1_tile_n0,
        d_keys_out + t1_tile_n0, d_vals_out + t1_tile_n0, t1_tile_n1,
        d_t1_keys_merged, d_t1_merged_vals, t1_count);
    CHECK(cudaGetLastError());

    s_free(stats, d_keys_out);
    s_free(stats, d_vals_out);

    // Compact-streaming: park d_t1_keys_merged across the remainder of
    // T1 sort. It's only needed again at T2 match entry as the
    // d_sorted_mi input; freeing it here drops the gather-time and
    // T2-match-setup live set by 1 GB at k=28.
    if (scratch.h_keys_merged) {
        CHECK(cudaMemcpyAsync(scratch.h_keys_merged, d_t1_keys_merged,
                              t1_count * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, stream));
        s_free(stats, d_t1_keys_merged);
        d_t1_keys_merged = nullptr;
    }

    // Compact-streaming: JIT H2D d_t1_meta back before gather.
    if (scratch.h_meta) {
        s_malloc(stats, d_t1_meta, cap * sizeof(uint64_t), "d_t1_meta");
        CHECK(cudaMemcpyAsync(d_t1_meta, scratch.h_meta,
                              t1_count * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream));
    }

    uint64_t* d_t1_meta_sorted = nullptr;
    s_malloc(stats, d_t1_meta_sorted, cap * sizeof(uint64_t), "d_t1_meta_sorted");
    gather_u64<<<blocks(t1_count), kThreads, 0, stream>>>(
        d_t1_meta, d_t1_merged_vals, d_t1_meta_sorted, t1_count);
    CHECK(cudaGetLastError());

    s_free(stats, d_t1_meta);
    s_free(stats, d_t1_merged_vals);

    // ---------- Phase T2 match ----------
    stats.phase = "T2 match";
    auto t2p = make_t2_params(cfg.k, cfg.strength);
    size_t t2_temp_bytes = 0;
    CHECK(launch_t2_match(cfg.plot_id.data(), t2p, nullptr, nullptr, t1_count,
                          nullptr, nullptr, nullptr, d_counter, cap,
                          nullptr, &t2_temp_bytes));

    // Compact-streaming: H2D d_t1_keys_merged back from pinned host
    // (we parked it across T1 sort's gather peak).
    if (scratch.h_keys_merged) {
        s_malloc(stats, d_t1_keys_merged, cap * sizeof(uint32_t), "d_t1_keys_merged");
        CHECK(cudaMemcpyAsync(d_t1_keys_merged, scratch.h_keys_merged,
                              t1_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream));
    }

    // In compact mode the match output is accumulated into the pinned
    // host slots (scratch.h_meta / scratch.h_t2_xbits) across two
    // half-cap passes, so the full-cap d_t2_meta / d_t2_xbits are never
    // live on device — the gather-time JIT H2D blocks below allocate
    // them when needed. In plain mode the match allocates full-cap on
    // device directly (the gather path uses them without H2D).
    uint64_t* d_t2_meta  = nullptr;
    uint32_t* d_t2_mi    = nullptr;
    uint32_t* d_t2_xbits = nullptr;
    void*     d_t2_match_temp = nullptr;
    uint64_t  t2_count   = 0;
    bool const t2_compact_path = (scratch.h_meta != nullptr &&
                                  scratch.h_t2_xbits != nullptr);

    if (t2_compact_path) {
        // Stages 2+3: N=2 tiled T2 match into half-cap device staging,
        // D2H each pass into pinned host accumulators. Peak during
        // match drops from cap*(8+4+4)=4160 MB to half_cap*(8+4+4)=2080
        // MB at k=28, at the cost of one extra round-trip PCIe + a
        // per-plot pinned h_t2_mi allocation (freed before T2 sort).
        // num_buckets = (1<<section_bits) * (1<<match_key_bits). For T2,
        // num_match_key_bits = strength (≥ 2), so buckets = 16 at
        // strength=2 regardless of k. Splitting at the midpoint gives
        // two roughly equal-sized passes.
        uint32_t const t2_num_buckets = (1u << t2p.num_section_bits)
                                      * (1u << t2p.num_match_key_bits);
        uint32_t const t2_bucket_mid  = t2_num_buckets / 2;
        uint64_t const t2_half_cap    = (cap + 1) / 2;

        uint64_t* d_t2_meta_stage  = nullptr;
        uint32_t* d_t2_mi_stage    = nullptr;
        uint32_t* d_t2_xbits_stage = nullptr;
        s_malloc(stats, d_t2_meta_stage,  t2_half_cap * sizeof(uint64_t), "d_t2_meta_stage");
        s_malloc(stats, d_t2_mi_stage,    t2_half_cap * sizeof(uint32_t), "d_t2_mi_stage");
        s_malloc(stats, d_t2_xbits_stage, t2_half_cap * sizeof(uint32_t), "d_t2_xbits_stage");
        s_malloc(stats, d_t2_match_temp,  t2_temp_bytes,                  "d_t2_match_temp");

        // scratch.h_meta (cap * u64) doubles as the pinned accumulator
        // for T2 meta — its T1 meta park lifetime ended at T1 gather.
        // Same for scratch.h_t2_xbits (dedicated slot). h_t2_mi has no
        // scratch slot and is allocated per-plot; it's freed right
        // after hydrating into d_t2_mi before T2 sort.
        uint64_t* const h_t2_meta  = scratch.h_meta;
        uint32_t* const h_t2_xbits = scratch.h_t2_xbits;
        uint32_t* h_t2_mi = streaming_alloc_pinned_uint32(cap);
        if (!h_t2_mi) throw std::runtime_error("pinned alloc for h_t2_mi failed");

        CHECK(launch_t2_match_prepare(cfg.plot_id.data(), t2p,
                                      d_t1_keys_merged, t1_count,
                                      d_counter,
                                      d_t2_match_temp, &t2_temp_bytes,
                                      stream));

        auto run_pass = [&](uint32_t b_begin, uint32_t b_end,
                            uint64_t host_offset) -> uint64_t {
            CHECK(launch_t2_match_range(cfg.plot_id.data(), t2p,
                                        d_t1_meta_sorted, d_t1_keys_merged, t1_count,
                                        d_t2_meta_stage, d_t2_mi_stage, d_t2_xbits_stage,
                                        d_counter, t2_half_cap,
                                        d_t2_match_temp,
                                        b_begin, b_end, stream));
            uint64_t pass_count = 0;
            CHECK(cudaMemcpyAsync(&pass_count, d_counter, sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            if (pass_count > t2_half_cap) {
                throw std::runtime_error(
                    "T2 match pass overflow: bucket range [" +
                    std::to_string(b_begin) + "," + std::to_string(b_end) +
                    ") produced " + std::to_string(pass_count) +
                    " pairs, staging holds " + std::to_string(t2_half_cap));
            }
            CHECK(cudaMemcpyAsync(h_t2_meta + host_offset, d_t2_meta_stage,
                                  pass_count * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(h_t2_mi + host_offset, d_t2_mi_stage,
                                  pass_count * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(h_t2_xbits + host_offset, d_t2_xbits_stage,
                                  pass_count * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost, stream));
            // Reset counter so the next pass writes at index 0 of the
            // half-cap staging buffers.
            CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint64_t), stream));
            CHECK(cudaStreamSynchronize(stream));
            return pass_count;
        };

        uint64_t const count1 = run_pass(0,              t2_bucket_mid,   /*host_offset=*/0);
        uint64_t const count2 = run_pass(t2_bucket_mid,  t2_num_buckets,  /*host_offset=*/count1);
        t2_count = count1 + count2;
        if (t2_count > cap) throw std::runtime_error("T2 overflow");

        s_free(stats, d_t2_match_temp);
        s_free(stats, d_t2_meta_stage);
        s_free(stats, d_t2_mi_stage);
        s_free(stats, d_t2_xbits_stage);
        s_free(stats, d_t1_meta_sorted);
        s_free(stats, d_t1_keys_merged);

        // Hydrate full-cap d_t2_mi from h_t2_mi (T2 sort consumes it as
        // the CUB sort key input). h_t2_mi is freed immediately after —
        // its data now lives on-device and no further consumer needs it.
        s_malloc(stats, d_t2_mi, cap * sizeof(uint32_t), "d_t2_mi");
        CHECK(cudaMemcpyAsync(d_t2_mi, h_t2_mi,
                              t2_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream));
        CHECK(cudaStreamSynchronize(stream));
        streaming_free_pinned_uint32(h_t2_mi);
    } else {
        // Plain streaming: single-pass full-cap match. Unchanged behavior.
        s_malloc(stats, d_t2_meta,       cap * sizeof(uint64_t), "d_t2_meta");
        s_malloc(stats, d_t2_mi,         cap * sizeof(uint32_t), "d_t2_mi");
        s_malloc(stats, d_t2_xbits,      cap * sizeof(uint32_t), "d_t2_xbits");
        s_malloc(stats, d_t2_match_temp, t2_temp_bytes,          "d_t2_match_temp");

        CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint64_t), stream));
        CHECK(launch_t2_match(cfg.plot_id.data(), t2p,
                              d_t1_meta_sorted, d_t1_keys_merged, t1_count,
                              d_t2_meta, d_t2_mi, d_t2_xbits,
                              d_counter, cap,
                              d_t2_match_temp, &t2_temp_bytes, stream));

        CHECK(cudaMemcpy(&t2_count, d_counter, sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
        if (t2_count > cap) throw std::runtime_error("T2 overflow");

        s_free(stats, d_t2_match_temp);
        s_free(stats, d_t1_meta_sorted);
        s_free(stats, d_t1_keys_merged);
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
    {
        cub::DoubleBuffer<uint32_t> probe_keys(nullptr, nullptr);
        cub::DoubleBuffer<uint32_t> probe_vals(nullptr, nullptr);
        CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, t2_sort_bytes,
            probe_keys, probe_vals,
            t2_tile_max, 0, cfg.k, stream));
    }

    stats.phase = "T2 sort";
    // DoubleBuffer mode: ping-pong over d_t2_mi/d_keys_out (and
    // d_vals_in/d_vals_out). Internal scratch shrinks ~2 GB → ~MB.
    s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
    s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
    s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
    s_malloc(stats, d_sort_scratch, t2_sort_bytes,          "d_sort_scratch(t2)");

    init_u32_identity<<<blocks(t2_count), kThreads, 0, stream>>>(
        d_vals_in, t2_count);
    CHECK(cudaGetLastError());

    for (int t = 0; t < kNumT2Tiles; ++t) {
        if (t2_tile_n[t] == 0) continue;
        uint64_t off = t2_tile_off[t];
        cub::DoubleBuffer<uint32_t> dk(d_t2_mi + off, d_keys_out + off);
        cub::DoubleBuffer<uint32_t> dv(d_vals_in + off, d_vals_out + off);
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, t2_sort_bytes,
            dk, dv,
            t2_tile_n[t], 0, cfg.k, stream));
        if (dk.Current() != d_keys_out + off) {
            CHECK(cudaMemcpyAsync(d_keys_out + off, dk.Current(),
                t2_tile_n[t] * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
        }
        if (dv.Current() != d_vals_out + off) {
            CHECK(cudaMemcpyAsync(d_vals_out + off, dv.Current(),
                t2_tile_n[t] * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
        }
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
        merge_pairs_stable_2way<<<blocks(ab_count), kThreads, 0, stream>>>(
            d_keys_out + t2_tile_off[0], d_vals_out + t2_tile_off[0], t2_tile_n[0],
            d_keys_out + t2_tile_off[1], d_vals_out + t2_tile_off[1], t2_tile_n[1],
            d_AB_keys, d_AB_vals, ab_count);
        CHECK(cudaGetLastError());
    }
    if (cd_count > 0) {
        merge_pairs_stable_2way<<<blocks(cd_count), kThreads, 0, stream>>>(
            d_keys_out + t2_tile_off[2], d_vals_out + t2_tile_off[2], t2_tile_n[2],
            d_keys_out + t2_tile_off[3], d_vals_out + t2_tile_off[3], t2_tile_n[3],
            d_CD_keys, d_CD_vals, cd_count);
        CHECK(cudaGetLastError());
    }

    // Per-tile CUB outputs are consumed; free before alloc'ing the
    // final merged buffers.
    s_free(stats, d_keys_out);
    s_free(stats, d_vals_out);

    uint32_t* d_t2_keys_merged = nullptr;   // merged sorted MI for T3.
    uint32_t* d_merged_vals    = nullptr;   // merged sorted src indices.
    s_malloc(stats, d_t2_keys_merged, cap * sizeof(uint32_t), "d_t2_keys_merged");
    s_malloc(stats, d_merged_vals,    cap * sizeof(uint32_t), "d_merged_vals");

    merge_pairs_stable_2way<<<blocks(t2_count), kThreads, 0, stream>>>(
        d_AB_keys, d_AB_vals, ab_count,
        d_CD_keys, d_CD_vals, cd_count,
        d_t2_keys_merged, d_merged_vals, t2_count);
    CHECK(cudaGetLastError());

    s_free(stats, d_AB_keys);
    s_free(stats, d_AB_vals);
    s_free(stats, d_CD_keys);
    s_free(stats, d_CD_vals);

    // Compact-streaming: park d_t2_keys_merged across the gather peak.
    // It's not used by the gather (that reads d_merged_vals as indices)
    // and only needed again at T3 match entry. Frees 1 GB at k=28
    // during the gather phase.
    if (scratch.h_keys_merged) {
        CHECK(cudaMemcpyAsync(scratch.h_keys_merged, d_t2_keys_merged,
                              t2_count * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, stream));
        s_free(stats, d_t2_keys_merged);
        d_t2_keys_merged = nullptr;
    }

    // Compact-streaming: JIT H2D d_t2_meta back for gather_u64.
    if (scratch.h_meta) {
        s_malloc(stats, d_t2_meta, cap * sizeof(uint64_t), "d_t2_meta");
        CHECK(cudaMemcpyAsync(d_t2_meta, scratch.h_meta,
                              t2_count * sizeof(uint64_t),
                              cudaMemcpyHostToDevice, stream));
    }

    uint64_t* d_t2_meta_sorted = nullptr;
    s_malloc(stats, d_t2_meta_sorted, cap * sizeof(uint64_t), "d_t2_meta_sorted");
    gather_u64<<<blocks(t2_count), kThreads, 0, stream>>>(
        d_t2_meta, d_merged_vals, d_t2_meta_sorted, t2_count);
    CHECK(cudaGetLastError());
    s_free(stats, d_t2_meta);

    // Compact-streaming: JIT H2D d_t2_xbits back for gather_u32.
    if (scratch.h_t2_xbits) {
        s_malloc(stats, d_t2_xbits, cap * sizeof(uint32_t), "d_t2_xbits");
        CHECK(cudaMemcpyAsync(d_t2_xbits, scratch.h_t2_xbits,
                              t2_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream));
    }

    uint32_t* d_t2_xbits_sorted = nullptr;
    s_malloc(stats, d_t2_xbits_sorted, cap * sizeof(uint32_t), "d_t2_xbits_sorted");
    gather_u32<<<blocks(t2_count), kThreads, 0, stream>>>(
        d_t2_xbits, d_merged_vals, d_t2_xbits_sorted, t2_count);
    CHECK(cudaGetLastError());
    s_free(stats, d_t2_xbits);
    s_free(stats, d_merged_vals);

    // ---------- Phase T3 match ----------
    stats.phase = "T3 match";
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          nullptr, t2_count,
                          nullptr, d_counter, cap,
                          nullptr, &t3_temp_bytes));
    // Compact-streaming: H2D d_t2_keys_merged back for T3 match (its
    // consumer). Allocated here so d_t3 + d_t3_match_temp can pick the
    // freed region left by d_t2_meta during the meta gather above.
    if (scratch.h_keys_merged) {
        s_malloc(stats, d_t2_keys_merged, cap * sizeof(uint32_t), "d_t2_keys_merged");
        CHECK(cudaMemcpyAsync(d_t2_keys_merged, scratch.h_keys_merged,
                              t2_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, stream));
    }

    T3PairingGpu* d_t3 = nullptr;
    void*         d_t3_match_temp = nullptr;
    s_malloc(stats, d_t3,            cap * sizeof(T3PairingGpu), "d_t3");
    s_malloc(stats, d_t3_match_temp, t3_temp_bytes,              "d_t3_match_temp");

    CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint64_t), stream));
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          d_t2_keys_merged, t2_count,
                          d_t3, d_counter, cap,
                          d_t3_match_temp, &t3_temp_bytes, stream));

    uint64_t t3_count = 0;
    CHECK(cudaMemcpy(&t3_count, d_counter, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t3_count > cap) throw std::runtime_error("T3 overflow");

    s_free(stats, d_t3_match_temp);
    s_free(stats, d_t2_meta_sorted);
    s_free(stats, d_t2_xbits_sorted);
    s_free(stats, d_t2_keys_merged);

    // ---------- Phase T3 sort ----------
    size_t t3_sort_bytes = 0;
    {
        cub::DoubleBuffer<uint64_t> probe(nullptr, nullptr);
        CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr, t3_sort_bytes,
            probe,
            cap, 0, 2 * cfg.k, stream));
    }

    stats.phase = "T3 sort";
    uint64_t* d_frags_in  = reinterpret_cast<uint64_t*>(d_t3);
    uint64_t* d_frags_out = nullptr;
    s_malloc(stats, d_frags_out,    cap * sizeof(uint64_t), "d_frags_out");
    s_malloc(stats, d_sort_scratch, t3_sort_bytes,          "d_sort_scratch(t3)");

    {
        cub::DoubleBuffer<uint64_t> dk(d_frags_in, d_frags_out);
        CHECK(cub::DeviceRadixSort::SortKeys(
            d_sort_scratch, t3_sort_bytes,
            dk,
            t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, stream));
        if (dk.Current() != d_frags_out) {
            CHECK(cudaMemcpyAsync(d_frags_out, dk.Current(),
                t3_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
        }
    }

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

    if (t3_count > 0) {
        if (pinned_dst) {
            if (pinned_capacity < t3_count) {
                throw std::runtime_error(
                    "run_gpu_pipeline_streaming: pinned_capacity " +
                    std::to_string(pinned_capacity) +
                    " < t3_count " + std::to_string(t3_count));
            }
            CHECK(cudaMemcpyAsync(pinned_dst, d_frags_out,
                                  sizeof(uint64_t) * t3_count,
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            result.external_fragments_ptr   = pinned_dst;
            result.external_fragments_count = t3_count;
        } else {
            uint64_t* h_pinned = nullptr;
            CHECK(cudaMallocHost(&h_pinned, sizeof(uint64_t) * t3_count));
            CHECK(cudaMemcpyAsync(h_pinned, d_frags_out,
                                  sizeof(uint64_t) * t3_count,
                                  cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
            result.t3_fragments_storage.resize(t3_count);
            std::memcpy(result.t3_fragments_storage.data(), h_pinned,
                        sizeof(uint64_t) * t3_count);
            CHECK(cudaFreeHost(h_pinned));
        }
    }

    s_free(stats, d_frags_out);
    s_free(stats, d_counter);

    if (stats.verbose) {
        std::fprintf(stderr,
            "[streaming] k=%d strength=%d  peak device VRAM = %.2f MB\n",
            cfg.k, cfg.strength, stats.peak / 1048576.0);
    }
    return result;
}

} // namespace (anon — streaming impl)

uint32_t* streaming_alloc_pinned_uint32(size_t count)
{
    uint32_t* p = nullptr;
    if (cudaMallocHost(&p, count * sizeof(uint32_t)) != cudaSuccess) {
        return nullptr;
    }
    return p;
}

void streaming_free_pinned_uint32(uint32_t* ptr)
{
    if (ptr) cudaFreeHost(ptr);
}

size_t streaming_query_free_vram_bytes()
{
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    // Honour POS2GPU_MAX_VRAM_MB so the tier dispatch can be tested on
    // a high-VRAM card by capping the reported free memory, matching
    // how the streaming-path s_malloc tracker also caps.
    if (char const* v = std::getenv("POS2GPU_MAX_VRAM_MB"); v && v[0]) {
        size_t const cap = size_t(std::strtoull(v, nullptr, 10)) * (1ULL << 20);
        if (cap > 0 && cap < free_b) free_b = cap;
    }
    return free_b;
}

uint64_t* streaming_alloc_pinned_uint64(size_t count)
{
    uint64_t* p = nullptr;
    if (cudaMallocHost(&p, count * sizeof(uint64_t)) != cudaSuccess) return nullptr;
    return p;
}

void streaming_free_pinned_uint64(uint64_t* ptr)
{
    if (ptr) cudaFreeHost(ptr);
}

void bind_current_device(int device_id)
{
    if (device_id < 0) return;
    // cudaSetDevice binds the current HOST thread's CUDA context to the
    // given device. All subsequent cudaMalloc / kernel launches /
    // cudaMemcpyToSymbol calls from this thread route to that device.
    // Peer devices share no state by default — each worker effectively
    // gets its own __constant__ memory after re-running
    // initialize_aes_tables() on the new current device.
    cudaError_t const rc = cudaSetDevice(device_id);
    if (rc != cudaSuccess) {
        throw std::runtime_error(
            std::string("bind_current_device(") + std::to_string(device_id) +
            ") failed: " + cudaGetErrorString(rc));
    }
}

int gpu_device_count()
{
    int n = 0;
    cudaError_t const rc = cudaGetDeviceCount(&n);
    if (rc != cudaSuccess) return 0;
    return n;
}

} // namespace pos2gpu
