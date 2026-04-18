// GpuPipeline.hpp — host-only API for running the full Xs → T1 → T2 → T3
// pipeline on the GPU. Returns the sorted ProofFragment stream that
// PlotFile::writeData expects.
//
// Two entry points:
//   run_gpu_pipeline(cfg)        — allocates all device buffers per call.
//                                  Simplest for one-shot plotting.
//   run_gpu_pipeline(cfg, pool)  — reuses caller-owned buffers. Use for
//                                  batch plotting to amortise the ~2.4 s
//                                  of cudaMalloc / cudaMallocHost overhead
//                                  across all plots.
//
// Implementation in src/host/GpuPipeline.cu (CUDA TU). This header is
// intentionally CUDA-free so plain .cpp consumers (GpuPlotter.cpp,
// xchplot2/main.cpp) can include it without dragging in nvcc.

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

struct GpuBufferPool;

struct GpuPipelineConfig {
    std::array<uint8_t, 32> plot_id{};
    int k = 28;
    int strength = 2;
    bool testnet = false;
    bool profile = false;   // print per-phase cudaEvent timing breakdown to stderr
};

// T3 fragment ownership depends on which overload produced this result.
//   run_gpu_pipeline(cfg)                        — t3_fragments_storage owns.
//   run_gpu_pipeline(cfg, pool, pinned_index)    — external_fragments_ptr
//       borrows pool.h_pinned_t3[pinned_index]; valid until producer reuses
//       that pinned slot for a subsequent plot.
// Consumers should prefer fragments() which hides the distinction.
struct GpuPipelineResult {
    std::vector<uint64_t> t3_fragments_storage;          // one-shot path
    uint64_t const*       external_fragments_ptr   = nullptr;  // pool path
    size_t                external_fragments_count = 0;

    std::span<uint64_t const> fragments() const noexcept
    {
        if (!t3_fragments_storage.empty()) {
            return {t3_fragments_storage.data(), t3_fragments_storage.size()};
        }
        return {external_fragments_ptr, external_fragments_count};
    }

    uint64_t t1_count = 0;
    uint64_t t2_count = 0;
    uint64_t t3_count = 0;
};

// One-shot path: allocates a transient pool, runs the pipeline, then copies
// the pinned T3 fragments into t3_fragments_storage so the result is
// self-contained after the pool is destroyed.
GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg);

// Batch path: runs the pipeline writing D2H into pool.h_pinned_t3[pinned_index]
// and returns a borrowing result. The consumer must process the fragments
// before the producer reuses the same pinned_index for a future plot.
//
// `pool` must have been sized with the same (k, strength, testnet) as cfg —
// otherwise throws.
GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool,
                                   int pinned_index);

} // namespace pos2gpu
