// GpuBufferPool.hpp — owns all device and pinned host buffers needed by
// run_gpu_pipeline(), sized once at construction and reused across plots.
//
// Motivation: per-plot cudaMalloc / cudaMallocHost calls cost ~2.4 s in a
// k=28 batch run (dominated by cudaMallocHost on a 2 GB pinned region,
// ~600 ms). Amortising that across a batch of plots removes the gap
// between device time (~2.75 s) and producer wall time (~5.1 s).
//
// Memory layout with aliasing (k=28 worst-case sizes in parens):
//   d_storage      (~2-3 GB)  — Xs candidates during Xs phase, then
//                               3×uint32[cap] sort keys/vals during sorts
//                               (keys_out + vals ping-pong pair; key input
//                               ping-pongs against the match output's mi
//                               column via cub::DoubleBuffer)
//   d_pair_a       (~1.3 GB)  — T1/T2/T3 match output (reused across phases).
//                               Sized to the largest match-output: cap·16 B
//                               for T2 (meta+mi+xbits SoA). Does NOT alias the
//                               Xs phase scratch — that lives in d_pair_b.
//   d_pair_b       (~4.4 GB)  — *_sorted / frags_out (reused across phases),
//                               AND the Xs construction scratch. Sized to
//                               max(largest sorted-output, xs_temp_bytes);
//                               at k=28 xs_temp dominates.
//   d_sort_scratch (~MB)      — CUB DoubleBuffer mode shrinks scratch from
//                               ~2 GB to ~MB by ping-ponging caller buffers.
//   d_counter      (8 B)      — reused uint64_t count output
//   h_pinned_t3[N] (~2.2 GB ea) — rotating final-fragments DMA targets.
//                                 Producer writes plot K into slot K mod N
//                                 while consumer reads earlier plots from
//                                 the other slots; channel depth N-1 keeps
//                                 the producer from overwriting in-flight
//                                 reads. N defaults to 3 (see kNumPinnedBuffers).
//
// MEASURED at k=28 (bench's VramWatchdog, i.e. the driver's own accounting, not
// a sum of what we think we asked for): 11484 MiB of device buffers, and the
// driver puts the running pipeline 18 MiB over that — run_gpu_pipeline allocates
// nothing at runtime, so these buffers really are the whole footprint. Plus
// ~6.6 GB pinned *host* (3 rotating slots), which costs no VRAM.
//
// With the 128 MiB gate margin that is an 11612 MiB floor, so it does fit a
// Navi 22 / RTX 4080 12 GB — but only just, and only since the margin was
// corrected from 512 (which put the floor at 11996 and locked those cards out
// of the path this header claims they run). D2H/Xs overlap wants a further
// 2080 MiB on top; below that the pool aliases d_pair_b and runs anyway.
//
// Pre-split this peaked at ~12.7 GB device because pair_bytes was a single
// max(pairings, xs_temp) and applied to BOTH d_pair_a and d_pair_b,
// double-counting the Xs scratch.
//
// Note: T1/T2/T3 match kernels report temp_bytes = 0 (no scratch needed).
// Only the Xs phase wants ~4.4 GB of scratch, and we alias d_pair_b for that.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace pos2gpu {

// Typed exception for the "pool sizing exceeds available device VRAM"
// case. Callers that want to fall back to the streaming pipeline when
// the pool does not fit should catch this specifically rather than
// string-matching a generic std::runtime_error.
struct InsufficientVramError : std::runtime_error {
    using std::runtime_error::runtime_error;
    size_t required_bytes = 0;
    size_t free_bytes     = 0;
    size_t total_bytes    = 0;
};

// Free VRAM the pool gate leaves unclaimed, in bytes. 128 MiB, overridable with
// POS2GPU_VRAM_MARGIN_MB. Matches the SYCL tree's vram_safety_margin().
//
// This margin is NOT an accounting fudge for unmodelled allocations. The pooled
// path (run_gpu_pipeline) allocates nothing at runtime — every device byte it
// touches comes from the buffers sized in this ctor — so required_device_bytes
// is its complete device footprint, and the driver agrees to within 8 MiB.
// What the margin actually buys is tolerance for *other tenants* on the card
// taking VRAM after the gate has already waved us through.
//
// It was 512 MiB, which double-counted the CUDA context: cudaMemGetInfo reports
// free *after* context creation, so the ~390 MiB context is already excluded
// from free_b — the margin was reserving it a second time. That cost every card
// 384 MiB of capability for nothing, and at k=28 it is the difference between a
// 12 GiB card getting the pooled path and being pushed onto streaming.
//
// DO NOT unify this with BatchPlotter's kFloorMarginBytes (256 MiB), which
// looks like the same number and is not. The streaming tiers run under the
// budgeted allocator in GpuPipeline.cu, which holds kStreamSafetyBytes
// (128 MiB) back from its own budget; a streaming floor of peak+256 therefore
// yields a working budget of peak+128. Cut that to 128 and the budget collapses
// to exactly peak, with nothing left for allocation granularity, and the tier
// OOMs on its first rounded-up request. The pooled path has no such allocator
// and no such holdback, which is why it can run on the thinner margin.
size_t vram_safety_margin();

// Modelled HOST-RAM peak of each streaming tier, in bytes.
//
// Host RAM moves OPPOSITE to VRAM here: the lower the tier, the more full-cap
// tables it parks in pinned host memory, so the card that is forced down the
// ladder is the one whose box can least afford it. Reaching for --tier tiny to
// "use less memory" therefore makes host memory worse, not better. Without a
// model the picker weighs VRAM only, and a small card on a modest host walks
// straight into the OOM killer with no diagnostic.
//
// | tier    | B/entry | k=28 modelled | k=28 measured |
// |---------|--------:|--------------:|--------------:|
// | plain   |      24 |     7.36 GiB  |     7.210     |
// | compact |      44 |    12.44 GiB  |    12.288     |
// | minimal |      48 |    13.46 GiB  |    13.304     |
// | tiny    |      52 |    14.47 GiB  |    14.361     |
//
// Measured on this branch, k=28, n=3, peak RSS from VmHWM, 2026-08-02. Every
// reconstruction lands 0.11-0.16 GiB ABOVE its reading, which is the safe
// direction for a gate.
//
// These are NOT the SYCL branch's constants and must never be copied from it.
// That branch measures 24/52/68/80 for the same four tiers — its tiny needs
// 21.45 GiB where this one needs 14.36. Taking its numbers would over-state
// this branch's tiny by 7 GiB and refuse hosts that plot fine.
//
// CALIBRATE AT n>=3, NOT n=1. Tiny is the only tier whose peak grows with batch
// depth, because the producer runs ahead of the file writer and tiny also
// aliases the rotating D2H slots as device-visible working buffers. A one-plot
// calibration understates exactly one tier, and it is the tier whose users have
// the least RAM to spare.
//
// The fixed term is what does not scale with cap: the CUDA context, the binary,
// and the file writer's compressed-chunk heap. Keeping it separate matters at
// small k, where it dominates — a pure per-entry model would predict tens of MB
// at k=20 against an actual ~1.3 GiB, i.e. wrong in the unsafe direction.
size_t streaming_plain_host_bytes(int k);
size_t streaming_compact_host_bytes(int k);
size_t streaming_minimal_host_bytes(int k);
size_t streaming_tiny_host_bytes(int k);

// Host RAM to leave for the rest of the box when admitting a streaming tier.
// XCHPLOT2_HOST_RESERVE_MB overrides; default 512 MiB.
//
// Distinct from host_reserve_bytes() in BatchPlotter.cpp, which is the CPU
// workers' XCHPLOT2_CPU_RESERVE_MB and defaults to zero. Same shape, different
// question, different knob — do not unify them.
size_t streaming_host_reserve();

struct GpuBufferPool {
    // Allocates all buffers sized for (k, strength, testnet). Throws
    // InsufficientVramError when the sized pool will not fit in free
    // device VRAM; throws std::runtime_error on any other CUDA
    // allocation or API failure.
    GpuBufferPool(int k, int strength, bool testnet);
    ~GpuBufferPool();

    GpuBufferPool(GpuBufferPool const&) = delete;
    GpuBufferPool& operator=(GpuBufferPool const&) = delete;

    // Configuration this pool was sized for — callers must match.
    int  k = 0;
    int  strength = 0;
    bool testnet = false;

    // Derived sizes (for diagnostics / assertions).
    uint64_t total_xs           = 0;
    uint64_t cap                = 0;
    size_t   storage_bytes      = 0;
    size_t   pair_a_bytes       = 0; // max(T1/T2/T3 match-output footprints)
    size_t   pair_b_bytes       = 0; // max(*_sorted footprints, xs_temp_bytes)
    size_t   xs_temp_bytes      = 0; // scratch size the Xs phase asks for
    size_t   sort_scratch_bytes = 0;
    size_t   pinned_bytes       = 0; // per pinned buffer

    // Every device byte this pool holds. The pooled path allocates nothing at
    // runtime, so these two are its complete device footprint — which is what
    // makes them meaningful as a *declaration*: `bench`'s VRAM watchdog checks
    // the driver's observed peak against them, so an allocation someone adds to
    // run_gpu_pipeline later without accounting for it here gets caught instead
    // of silently OOMing whichever card was closest to the line.
    size_t   required_device_bytes = 0; // the five base buffers (what the gate checks)
    size_t   frags_dedicated_bytes = 0; // d_frags_dedicated, or 0 when overlap is off

    // Device buffers (void* because the same region serves multiple roles;
    // callers reinterpret_cast).
    void*     d_storage      = nullptr;
    void*     d_pair_a       = nullptr;
    void*     d_pair_b       = nullptr;
    void*     d_sort_scratch = nullptr;
    uint64_t* d_counter      = nullptr;

    // Number of rotating pinned slots for the final T3-fragment D2H.
    // Set to 3 so the channel can hold depth-2 of in-flight plots
    // without the producer ever overwriting a slot the consumer is
    // still reading — useful when consumer wall > producer wall
    // (slow disk / FSE-heavy strengths). 2 was enough for the
    // previously measured producer-slower-than-consumer case, but
    // 3 costs only ~2 GB of host pinned at k=28 and widens the
    // "safe" consumer/producer ratio.
    static constexpr int kNumPinnedBuffers = 3;
    uint64_t* h_pinned_t3[kNumPinnedBuffers] = {};

    // Optional dedicated fragment buffer for overlapping the final D2H
    // with the next plot's Xs phase (P3). Auto-enabled when free VRAM
    // covers pool + cap*8 + vram_safety_margin(); POS2GPU_NO_D2H_OVERLAP=1
    // forces the aliased (d_pair_b) path. When overlap_d2h is false,
    // d_frags_dedicated is null and behaviour matches the pre-P3 pool.
    // A card sitting exactly on the pool's floor gets the aliased path —
    // the overlap is a speed-up bought with spare VRAM, never a requirement.
    void*     d_frags_dedicated = nullptr;
    bool      overlap_d2h       = false;
    // Opaque cudaEvent_t / cudaStream_t handles (void* so this header
    // stays CUDA-free for .cpp consumers). Owned by the pool; valid
    // only when overlap_d2h is true.
    void*     copy_stream       = nullptr; // cudaStream_t
    void*     e_t3_sorted       = nullptr; // cudaEvent_t
    void*     e_evacuated       = nullptr; // cudaEvent_t
    void*     e_d2h_done        = nullptr; // cudaEvent_t
};

} // namespace pos2gpu
