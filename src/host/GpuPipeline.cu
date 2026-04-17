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

#include "gpu/AesGpu.cuh"
#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

namespace {

#define CHECK(call) do {                                                 \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(std::string("CUDA: ") +                 \
                                 cudaGetErrorString(err));               \
    }                                                                    \
} while (0)

// =====================================================================
// T1 sort: by match_info, low k bits, stable. Uses CUB SortPairs with
// (key=match_info, value=index) then permutes T1Pairings.
// =====================================================================

__global__ void permute_t1(
    T1PairingGpu const* __restrict__ src,
    uint32_t const* __restrict__ indices,
    T1PairingGpu* __restrict__ dst,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = src[indices[idx]];
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

__global__ void permute_t2(
    T2PairingGpu const* __restrict__ src,
    uint32_t const* __restrict__ indices,
    T2PairingGpu* __restrict__ dst,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = src[indices[idx]];
}

__global__ void extract_t2_keys(
    T2PairingGpu const* __restrict__ src,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ vals_out,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    keys_out[idx] = src[idx].match_info;
    vals_out[idx] = uint32_t(idx);
}

} // namespace

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool)
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
    XsCandidateGpu* d_xs        = static_cast<XsCandidateGpu*>(pool.d_storage);
    T1PairingGpu*   d_t1        = static_cast<T1PairingGpu*>  (pool.d_pair_a);
    T1PairingGpu*   d_t1_sorted = static_cast<T1PairingGpu*>  (pool.d_pair_b);
    T2PairingGpu*   d_t2        = static_cast<T2PairingGpu*>  (pool.d_pair_a);
    T2PairingGpu*   d_t2_sorted = static_cast<T2PairingGpu*>  (pool.d_pair_b);
    T3PairingGpu*   d_t3        = static_cast<T3PairingGpu*>  (pool.d_pair_a);
    uint64_t*       d_frags_out = static_cast<uint64_t*>      (pool.d_pair_b);

    uint64_t*       d_count        = pool.d_counter;
    // Xs phase needs ~4.34 GB scratch at k=28; d_pair_b is idle through
    // the whole Xs phase (not touched until T1 sort permute writes to it),
    // so we alias it rather than allocating separately.
    void*           d_xs_temp      = pool.d_pair_b;
    void*           d_sort_scratch = pool.d_sort_scratch;
    uint64_t*       h_pinned_t3    = pool.h_pinned_t3;
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
                          d_t1, d_count, cap,
                          nullptr, &t1_temp_bytes));
    CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));
    int p_t1 = begin_phase("T1 match");
    CHECK(launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1, d_count, cap,
                          d_match_temp, &t1_temp_bytes, stream));
    end_phase(p_t1);
    CHECK(cudaStreamSynchronize(stream));

    uint64_t t1_count = 0;
    CHECK(cudaMemcpy(&t1_count, d_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t1_count > cap) throw std::runtime_error("T1 overflow");

    // Sort T1 by match_info (low k bits). d_storage is now repurposed
    // as (keys_in, keys_out, vals_in, vals_out), Xs having been fully
    // consumed by T1 match above.
    int p_t1_sort = begin_phase("T1 sort");
    {
        extract_t1_keys<<<blocks(t1_count), kThreads, 0, stream>>>(
            d_t1, d_keys_in, d_vals_in, t1_count);
        CHECK(cudaGetLastError());

        size_t sort_bytes = pool.sort_scratch_bytes;
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, sort_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            t1_count, /*begin_bit=*/0, /*end_bit=*/cfg.k, stream));

        permute_t1<<<blocks(t1_count), kThreads, 0, stream>>>(
            d_t1, d_vals_out, d_t1_sorted, t1_count);
        CHECK(cudaGetLastError());
        CHECK(cudaStreamSynchronize(stream));
    }
    end_phase(p_t1_sort);

    // ---------- Phase T2 ----------
    auto t2p = make_t2_params(cfg.k, cfg.strength);
    size_t t2_temp_bytes = 0;
    CHECK(launch_t2_match(cfg.plot_id.data(), t2p, d_t1_sorted, t1_count,
                          d_t2, d_count, cap,
                          nullptr, &t2_temp_bytes));
    CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));
    int p_t2 = begin_phase("T2 match");
    CHECK(launch_t2_match(cfg.plot_id.data(), t2p, d_t1_sorted, t1_count,
                          d_t2, d_count, cap,
                          d_match_temp, &t2_temp_bytes, stream));
    end_phase(p_t2);
    CHECK(cudaStreamSynchronize(stream));

    uint64_t t2_count = 0;
    CHECK(cudaMemcpy(&t2_count, d_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t2_count > cap) throw std::runtime_error("T2 overflow");

    int p_t2_sort = begin_phase("T2 sort");
    {
        extract_t2_keys<<<blocks(t2_count), kThreads, 0, stream>>>(
            d_t2, d_keys_in, d_vals_in, t2_count);
        CHECK(cudaGetLastError());

        size_t sort_bytes = pool.sort_scratch_bytes;
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, sort_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            t2_count, 0, cfg.k, stream));

        permute_t2<<<blocks(t2_count), kThreads, 0, stream>>>(
            d_t2, d_vals_out, d_t2_sorted, t2_count);
        CHECK(cudaGetLastError());
        CHECK(cudaStreamSynchronize(stream));
    }
    end_phase(p_t2_sort);

    // ---------- Phase T3 ----------
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p, d_t2_sorted, t2_count,
                          d_t3, d_count, cap,
                          nullptr, &t3_temp_bytes));
    CHECK(cudaMemsetAsync(d_count, 0, sizeof(uint64_t), stream));
    int p_t3 = begin_phase("T3 match + Feistel");
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p, d_t2_sorted, t2_count,
                          d_t3, d_count, cap,
                          d_match_temp, &t3_temp_bytes, stream));
    end_phase(p_t3);
    CHECK(cudaStreamSynchronize(stream));

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
        CHECK(cudaStreamSynchronize(stream));
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
        result.t3_fragments.resize(t3_count);
        std::memcpy(result.t3_fragments.data(), h_pinned_t3,
                    sizeof(uint64_t) * t3_count);
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
    // One-shot convenience path: build a transient pool and run through it.
    // Pays the full per-call allocator overhead (~2.4 s for k=28). Batch
    // callers should construct a pool once and reuse it via the overload.
    GpuBufferPool pool(cfg.k, cfg.strength, cfg.testnet);
    return run_gpu_pipeline(cfg, pool);
}

} // namespace pos2gpu
