// GpuBufferPool.hpp — owns all device and pinned host buffers needed by
// run_gpu_pipeline(), sized once at construction and reused across plots.
//
// Motivation: per-plot cudaMalloc / cudaMallocHost calls cost ~2.4 s in a
// k=28 batch run (dominated by cudaMallocHost on a 2 GB pinned region,
// ~600 ms). Amortising that across a batch of plots removes the gap
// between device time (~2.75 s) and producer wall time (~5.1 s).
//
// Memory layout with aliasing (k=28 worst-case sizes in parens):
//   d_storage      (~2-3 GB)  — Xs candidates during Xs phase,
//                               then 4×uint32[cap] sort keys/vals during sorts
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
// Total ~9 GB device + ~6.6 GB pinned host at k=28 — fits in 12 GB free VRAM
// on a Navi 22 / RTX 4080 12 GB. Pre-split this peaked at ~12.7 GB device
// because pair_bytes was a single max(pairings, xs_temp) and applied to BOTH
// d_pair_a and d_pair_b, double-counting the Xs scratch.
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
};

} // namespace pos2gpu
