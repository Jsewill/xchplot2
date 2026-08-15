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

    // When the pool path overlaps D2H with the next plot, the producer
    // returns before the copy drains. Opaque cudaEvent_t (void* so this
    // header stays CUDA-free); consumer must call wait_pipeline_d2h()
    // before reading fragments(). Null when overlap is off / one-shot.
    void* d2h_done_event = nullptr;
};

// Block until a pipeline result's overlapped D2H has landed in the
// pinned slot. No-op when d2h_done_event is null.
void wait_pipeline_d2h(GpuPipelineResult const& result);

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

// Tiered streaming — "compact" variant for low-VRAM cards.
//
// The plain streaming path above has a peak of ~7290 MB at k=28 (T2
// sort + T2 match dominate). 6 GB cards and tight 8 GB cards can't fit
// that. Populating a StreamingPinnedScratch and passing it to the
// overload below triggers park-and-rehydrate of the big d_t1_meta /
// d_t2_meta / d_t2_xbits / *_keys_merged buffers across their idle
// windows — drops overall peak to ~5200 MB at k=28 at the cost of
// ~1-2 s/plot of PCIe round-trips.
//
// Any field left nullptr makes the pipeline keep that buffer on device
// (plain behavior for that buffer). Populating all fields gives the
// full compact path. BatchPlotter picks the tier based on free VRAM at
// batch start.
//
// Lifetime-disjoint sharing: h_meta (cap*u64) is reused for t1_meta
// then t2_meta; h_keys_merged (cap*u32) is reused for t1 then t2.
struct StreamingPinnedScratch {
    uint64_t* h_meta         = nullptr;  // parks t1_meta, then t2_meta, then t3 accumulator
    uint32_t* h_keys_merged  = nullptr;  // parks t1_keys_merged, then t2_keys_merged
    uint32_t* h_t2_xbits     = nullptr;  // parks t2_xbits
    // T2 match staging tile count. compact uses 2 (cap/2 staging, ~2.3 GB at
    // k=28); minimal sets it to 8 (cap/8 staging, ~570 MB) to fit 4 GiB cards
    // at the cost of more PCIe round-trips during T2 match. Must be a power
    // of 2 in [2, t2_num_buckets] — at k=28 strength=2 that's [2, 16].
    // BatchPlotter's tier selection (kMinimalFloorBytes constant) gates
    // when minimal vs compact gets picked.
    int t2_tile_count = 2;
    // T3 match staging tile count. After T2-match got tiled, T3 match
    // became the new overall pipeline peak (d_t2_meta_sorted +
    // d_t2_xbits_sorted + d_t2_keys_merged + d_t3 ≈ 6240 MiB at k=28).
    // Setting t3_tile_count >= 2 emits T3 into a cap/N device staging
    // buffer and accumulates into h_meta (reused as the T3 host
    // accumulator — its T2 meta park lifetime ended at the gather phase
    // above, so the buffer is dead by T3 match entry). Default 1 = no
    // tiling (original single-shot kernel call). Must be a power of 2
    // in [1, t3_num_buckets] — at k=28 strength=2 that's [1, 16]; the
    // staging path (>= 2) requires h_meta to be non-null.
    int t3_tile_count = 1;
    // T1 / T2 sort gather tile count. When >= 2 the sort phase emits
    // the merged-key + permuted-meta gather output in N tiles, D2H'ing
    // each tile to host pinned (h_meta / h_keys_merged) so the cap-sized
    // sorted_meta never has to be alive on device in full. Re-hydrated
    // before the next match phase via H2D from the host accumulator.
    // Drops T1-sort and T2-sort phase peaks from 5200 → ~3640 MB at
    // k=28. Default 1 = no tiling. Used by the minimal tier (set to 4
    // by BatchPlotter); requires h_meta + h_keys_merged populated.
    int gather_tile_count = 1;
    // T3 match input-slice count. When >= 2 d_t2_meta_sorted is parked
    // on h_meta across T3 match; each pass H2Ds the section_l +
    // section_r row slices onto cap/N device buffers. d_t2_xbits_sorted
    // and d_t2_keys_merged stay full-cap on device for binary-search /
    // target reads. Caller iterates section_l ∈ [0, num_sections) using
    // bucket_begin = section_l × num_match_keys, bucket_end =
    // (section_l+1) × num_match_keys. Must equal num_sections (= 4 at
    // k=28 strength=2) when active. Default 1 = no slicing. Requires
    // h_meta populated and orthogonal to t3_tile_count (input slicing
    // already gives one D2H accumulator pass per section, no separate
    // output staging needed when input slicing is on).
    int t3_input_slice_count = 1;
    // Tiny tier: most-aggressive streaming. Mirrors the SYCL Tiny tier
    // (k=28 measured peak 1064 MB) for sub-2GB NVIDIA cards. When set,
    // GpuPipeline.cu activates per-section-pair T1 match + per-bucket-pair
    // T1/T2/T3 match sub-section + streaming-partition per-bucket sorts +
    // CPU merge+pack Xs + host-pinned d_t3_stage + d_frags_out host
    // alias + host-side T2/T3 prepare offsets. See [project_cuda_only_tiny_port].
    // Initially scaffolding-only (Tier::Tiny picker entry exists, but
    // GpuPipeline still routes Tiny through the Minimal code path until
    // the per-Phase wiring lands).
    bool tiny_mode = false;
    // The chosen tier's logical peak device VRAM at this k — BatchPlotter's
    // per-tier peak constant, the same one its floor is derived from. The
    // streaming allocator uses it to decide how much the stream-ordered pool
    // may cache: everything the card has beyond this working set is spare, and
    // only that much may be hoarded. 0 = unknown, in which case the pool is
    // allowed to cache up to the whole budget (correct only on a card with room
    // to spare). See s_init_budget in GpuPipeline.cu.
    uint64_t expected_peak_bytes = 0;

    // ---- Host-RAM disk-offload (see host/SpillEngine.hpp) ----
    //
    // Which full-cap host tables to redirect to a temp dir instead of pinned
    // RAM. All false (the default) is no spill and is byte-identical to the
    // historical all-pinned path.
    //
    // Only h_t2_xbits so far, and only in Compact, which is a deliberate floor
    // rather than a stopping point:
    //
    //   - h_meta is NOT here because on this branch it is ONE allocation
    //     playing four roles in sequence — T1 meta, then T2 meta, then the T3
    //     accumulator, then Xs staging via a u32 reinterpret. main spills its
    //     equivalents as four INDEPENDENT tables with disjoint access phases;
    //     that shape does not exist here, and choosing between splitting the
    //     buffer (which raises the un-spilled peak) and spilling it whole
    //     (which needs a per-phase coverage epoch the engine does not have) is
    //     an open design question, not a transcription.
    //
    //   - Minimal and Tiny touch h_t2_xbits from the CPU: Minimal's tiled T1
    //     gather merges through it by direct indexing, and Tiny's T2 sort
    //     gather reads h_t2_xbits[orig_idx] at RANDOM positions. A SpillBuffer
    //     serves sequential device<->disk ranges; a random-access host read
    //     needs the mmap class instead. Compact touches it only through
    //     sequential cudaMemcpyAsync — one append per T2 match pass, one full
    //     read before T2 sort — which is precisely what the engine is for.
    //     run_gpu_pipeline_streaming THROWS rather than silently ignoring a
    //     spill request on a tier that would CPU-touch the table.
    struct SpillPlan {
        bool h_t2_xbits = false;   // ~cap*4 B, Compact only (DMA)

        bool any() const { return h_t2_xbits; }
    };
    SpillPlan spill;

    // Suppress the pipeline's per-plot [spill] chatter (BatchOptions::quiet).
    bool quiet = false;
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

// Returns an approximate free-VRAM count on the current CUDA device.
// Used by BatchPlotter to pick between plain and compact streaming.
size_t streaming_query_free_vram_bytes();

// Hand every block the stream-ordered allocator is caching back to the driver,
// on the CALLING thread's device (the pool is per-device and the enable flag is
// thread_local, so call this from the worker that did the allocating).
//
// Why this is public rather than an internal detail: the pool's release
// threshold only takes effect at the NEXT synchronization point, so a caller
// that finishes a batch and then reads free VRAM — bench's second pass, a
// supervisor sizing the next job — measures the card while the pool still holds
// the previous batch's blocks. Measured on a 4090 at k=28 with the card
// squeezed to ~9 GiB free: 9.06 GiB before, 6.97 GiB after, which is enough to
// drop the tier picker from plain to compact. Setting the release threshold to
// 0 does NOT fix it — the release still waits for a sync that has not happened
// yet. Only an explicit trim does.
//
// No-op when the async pool is off (POS2GPU_NO_ASYNC_ALLOC=1, or a device with
// no memory-pool support). Synchronizes the device, so call it at a batch
// boundary, never inside a plot.
void streaming_release_cached_vram();

// Raw cudaMemGetInfo for the given device ordinal. False when the query fails.
//
// Deliberately does NOT honour POS2GPU_MAX_VRAM_MB, unlike the query above: this
// is the driver's own accounting, used by bench's VRAM watchdog to check what a
// plot really consumed. Feeding it a synthetic cap would make the watchdog
// measure the pretend card instead of the real one — and the watchdog exists
// precisely to catch the cases where our model and the driver disagree.
bool streaming_device_memory_probe(int ordinal, size_t& free_b, size_t& total_b);

// Multi-GPU device binding. bind_current_device() calls cudaSetDevice
// on the calling thread, which routes all subsequent CUDA runtime
// calls (cudaMalloc, kernel launches, cudaMemcpyToSymbol, etc.) to the
// given device. Must be called on the worker thread BEFORE any kernel
// launch on that thread — ideally as the first statement of the
// worker lambda.
//
// device_id < 0 → leave the current device untouched (use whatever
// CUDA picked as the default — matches pre-multi-GPU behavior).
//
// gpu_device_count() returns the number of visible CUDA devices, or 0
// on error. BatchPlotter uses it to expand `--devices all` into an
// explicit id list.
//
// Declared here (instead of in a .cuh) so plain .cpp consumers like
// BatchPlotter.cpp can call them without including cuda_runtime.h.
void bind_current_device(int device_id);
int  gpu_device_count();

} // namespace pos2gpu
