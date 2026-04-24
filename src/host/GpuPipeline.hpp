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
//
// If XCHPLOT2_STREAMING=1 is set in the environment, this routes through
// run_gpu_pipeline_streaming() instead — useful for exercising the low-VRAM
// path from unchanged call sites.
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

// Streaming path: per-phase cudaMalloc / cudaFree instead of a persistent
// pool. Targets GPUs where the full pool (~15 GB at k=28) will not fit.
//
// Two overloads:
//   run_gpu_pipeline_streaming(cfg)
//     Allocates an internal pinned staging buffer for the final D2H,
//     copies fragments into an owning std::vector, frees the pinned
//     buffer. Self-contained result. Simplest for one-shot callers.
//
//   run_gpu_pipeline_streaming(cfg, pinned_dst, pinned_capacity)
//     Caller supplies a pinned host buffer (size ≥ cap × sizeof(uint64_t))
//     that the pipeline uses as the D2H target. Result borrows into
//     pinned_dst via external_fragments_ptr; caller must not overwrite
//     pinned_dst while the consumer is still reading it. Use this from
//     BatchPlotter's streaming fallback to amortise the ~600 ms
//     cudaMallocHost cost across plots and double-buffer D2H with the
//     FSE consumer thread the same way the pool path does.
GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg);
GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg,
                                             uint64_t* pinned_dst,
                                             size_t    pinned_capacity);

// Caller-provided pinned-host scratch buffers for the streaming path.
// Allocate once per batch in BatchPlotter, reuse across all plots —
// avoids paying the ~300–600 ms sycl::malloc_host cost per plot per
// buffer on NVIDIA (measured as the dominant per-plot overhead in
// stages 4b-4e streaming runs). Lifetime analysis shows that phases
// using these buffers do not overlap, so two pairs can share a single
// allocation each:
//   h_meta        (cap × u64): T1 meta park → T2 meta park
//   h_keys_merged (cap × u32): T1 keys_merged park → T2 keys_merged park
//   h_t2_xbits    (cap × u32): T2 xbits park (distinct)
//   h_t3          (cap × T3PairingGpu = u64): T3 staging (distinct)
//
// Any field left nullptr makes the streaming pipeline allocate-on-
// demand for that buffer (one-shot `test` mode). A fully-populated
// StreamingPinnedScratch saves all 6 sycl::malloc_host calls per plot.
struct StreamingPinnedScratch {
    uint64_t* h_meta         = nullptr;
    uint32_t* h_keys_merged  = nullptr;
    uint32_t* h_t2_xbits     = nullptr;
    uint64_t* h_t3           = nullptr;  // reinterpreted as T3PairingGpu*

    // Plain mode: skip all parks and use single-pass T2 match. Higher
    // peak (~7.3 GB at k=28) than compact (~5.2 GB) but ~400 ms/plot
    // faster because there are no PCIe round-trips for T1 meta / T1
    // keys_merged / T2 meta / T2 xbits / T2 keys_merged parks. The
    // BatchPlotter picks this tier when free VRAM fits the plain peak
    // but not the pool (12-14 GB cards). When true, the h_* pointers
    // above are ignored — plain mode does not park anything.
    bool plain_mode          = false;
};

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg,
                                             uint64_t* pinned_dst,
                                             size_t    pinned_capacity,
                                             StreamingPinnedScratch const& scratch);

// Allocate / free host-pinned memory — thin wrappers around
// cudaMallocHost / cudaFreeHost, exposed so plain .cpp consumers (which
// do not have cuda_runtime.h on the include path) can own the pinned
// buffers the streaming overload expects. Returns nullptr on failure.
uint64_t* streaming_alloc_pinned_uint64(size_t count);
void      streaming_free_pinned_uint64(uint64_t* ptr);

uint32_t* streaming_alloc_pinned_uint32(size_t count);
void      streaming_free_pinned_uint32(uint32_t* ptr);

} // namespace pos2gpu
