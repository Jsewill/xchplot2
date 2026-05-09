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
class  HostPinnedPool;

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

    // Optional T2-sorted-meta buffer (distinct from h_meta). When the
    // caller provides h_meta as a single buffer, the streaming pipeline
    // historically reused it for both T1 sorted meta (input) and T2
    // sorted meta (output). In tiny mode, T2 match's per-pass loop
    // reads h_t1_meta WHILE D2H'ing T2 output to the same buffer; at
    // small k the cumulative D2H extent reaches the next section's
    // input range and corrupts the read. h_t2_meta lets the caller
    // provide a separate destination so T1 input and T2 output never
    // alias. nullptr falls back to a per-plot allocation in tiny mode
    // (so the bug is fixed for all callers, with the per-plot alloc
    // overhead avoidable by setting this field). Compact / plain
    // modes are unaffected — they don't read h_t1_meta during T2 match.
    uint64_t* h_t2_meta      = nullptr;

    // Plain mode: skip all parks and use single-pass T2 match. Higher
    // peak (~7.3 GB at k=28) than compact (~5.2 GB) but ~400 ms/plot
    // faster because there are no PCIe round-trips for T1 meta / T1
    // keys_merged / T2 meta / T2 xbits / T2 keys_merged parks. The
    // BatchPlotter picks this tier when free VRAM fits the plain peak
    // but not the pool (12-14 GB cards). When true, the h_* pointers
    // above are ignored — plain mode does not park anything.
    bool plain_mode          = false;

    // T2 match staging tile count (compact path only — ignored when
    // plain_mode is true). compact uses 2 (cap/2 staging, ~2.3 GB at
    // k=28); minimal sets it to 8 (cap/8 staging, ~570 MB) to fit 4
    // GiB cards at the cost of more PCIe round-trips during T2 match.
    // Must be a power of 2 in [2, t2_num_buckets] — at k=28 strength=2
    // that's [2, 16]. BatchPlotter's tier selection sets it.
    int t2_tile_count        = 2;

    // Sort-gather tile count (compact path only — ignored when
    // plain_mode is true). Each of T1-sort gather, T2-sort meta gather,
    // and T2-sort xbits gather peaks at ~5200 MB at k=28 because the
    // input meta + indices + output buffer are all cap-sized and live
    // simultaneously. With gather_tile_count = N > 1, the gather runs
    // in N tiles, D2H'ing each tile to a host pinned staging buffer
    // (reusing the parking scratch h_meta / h_t2_xbits) and
    // re-allocating the full sorted output afterward via H2D. Drops
    // each gather peak from 5200 to ~3640 MB at N=4 (peak = full input
    // 2080 + indices 1040 + tile output 520). Default 1 = no tiling
    // (compact / plain). Minimal tier sets it to 4. Adds ~3 PCIe round
    // trips of cap-sized data per plot.
    int gather_tile_count    = 1;

    // Tiny tier (host-park-everything-across-T3-match). Builds on
    // minimal: same N=4 sort-gather tiling and N=8 T2-match staging,
    // PLUS d_t2_xbits_sorted and d_t2_keys_merged stay parked on host
    // pinned memory across T3 match. T3 match runs per-section with
    // fully-sliced reads (meta + xbits + mi all sliced) via
    // launch_t3_match_section_pair_split_range. Drops T3 match peak
    // from minimal's ~3380 MB to ~1300 MB at k=28 — fits 2 GB cards.
    // Cost: two extra cap-sized PCIe round-trips (D2H xbits/keys after
    // sort, and per-section H2D slices into T3 match) on top of
    // minimal's already-elevated PCIe traffic. Requires plain_mode==
    // false and gather_tile_count > 1 (i.e., the minimal-path
    // prerequisites). BatchPlotter sets this when tier==Tiny.
    bool tiny_mode           = false;

    // Phase 2 (pipeline-parallel pooled VRAM) split fields. The
    // streaming pipeline normally runs Xs → T1 → T2 → T3 → write end-
    // to-end. To split the work across two GPUs, the first GPU runs
    // through T2 sort and parks the sorted T2 output on host pinned;
    // the second GPU starts at T3 match using those host buffers as
    // input.
    //
    // stop_after_t2_sort: when true, the streaming pipeline runs the
    //   first half (Xs / T1 / T2) and returns immediately after T2
    //   sort completes. The result struct's t2_count is the surviving
    //   count of T2 entries; t1_count is also populated. t3_count is
    //   0 and result.fragments() returns an empty span. The caller
    //   takes ownership of h_meta / h_t2_xbits / h_keys_merged (they
    //   hold the sorted T2 outputs). The h_t3 scratch is unused in
    //   this mode. Implies plain_mode==false (the first-half/second-
    //   half handoff requires the host-pinned park machinery).
    //
    // start_at_t3_match: when true, the streaming pipeline skips the
    //   Xs / T1 / T2 phases entirely and starts at T3 match. The
    //   caller MUST populate t2_count_in (count of valid entries in
    //   the T2 buffers) and provide h_meta / h_t2_xbits /
    //   h_keys_merged with that many sorted T2 entries. The caller
    //   retains ownership of those buffers; this entry point will not
    //   free them.
    //
    // The two flags are mutually exclusive — set exactly one for
    // Phase 2 splits, or leave both false for the existing
    // single-GPU full-pipeline behaviour.
    bool     stop_after_t2_sort = false;
    bool     start_at_t3_match  = false;
    uint64_t t2_count_in        = 0;
    uint64_t t1_count_in        = 0;

    // Optional host-pinned pool for amortising per-plot malloc_host
    // calls across a batch. When non-null, the streaming pipeline
    // routes its per-plot pinned-host allocations (currently h_t1_mi)
    // through pool->acquire(name, ...) instead of sycl::malloc_host.
    // The pool keeps the buffers alive across plots; the pipeline
    // does NOT free pool-owned buffers at function exit. nullptr
    // preserves the historical per-plot malloc_host + free behaviour.
    HostPinnedPool* pool = nullptr;
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

// Multi-GPU device binding. bind_current_device() sets a thread-local
// target device id that sycl_backend::queue() reads when lazily
// constructing the worker thread's queue. Must be called on the worker
// thread BEFORE any kernel launch on that thread — ideally as the very
// first statement of the worker lambda.
//
// device_id < 0 → use the default SYCL gpu_selector_v (single-device,
// pre-multi-GPU behavior). Calling with -1 from the main thread is a
// no-op and is always safe.
//
// gpu_device_count() returns the number of SYCL GPU devices the runtime
// can enumerate, or 0 on error. BatchPlotter uses it to expand
// `--devices all` into an explicit id list.
//
// Declared here (instead of in SyclBackend.hpp) so plain .cpp consumers
// like BatchPlotter.cpp can call them without pulling <sycl/sycl.hpp>
// onto their include path.
void bind_current_device(int device_id);
int  gpu_device_count();

} // namespace pos2gpu
