// GpuBufferPool.hpp — owns all device and pinned host buffers needed by
// run_gpu_pipeline(), sized once at construction and reused across plots.
//
// Motivation: per-plot cudaMalloc / cudaMallocHost calls cost ~2.4 s in a
// k=28 batch run (dominated by cudaMallocHost on a 2 GB pinned region,
// ~600 ms). Amortising that across a batch of plots removes the gap
// between device time (~2.75 s) and producer wall time (~5.1 s).
//
// Memory layout with aliasing (k=28 worst-case sizes in parens):
//   d_storage      (4.36 GB)  — Xs candidates during Xs phase,
//                               then 4×uint32[cap] sort keys/vals during sorts
//   d_pair_a       (4.36 GB)  — T1/T2/T3 match output (reused across phases);
//                               also serves as Xs phase scratch before T1
//   d_pair_b       (4.36 GB)  — *_sorted / frags_out (reused across phases);
//                               also serves as Xs phase scratch before T1
//   d_sort_scratch (~2.3 GB)  — CUB radix-sort scratch (largest across phases)
//   d_counter      (8 B)      — reused uint64_t count output
//   h_pinned_t3[2] (2.18 GB ea) — double-buffered final fragments DMA target.
//                                 Producer writes plot N to buffer (N%2) while
//                                 consumer reads plot N-1 from the other slot.
//                                 With a depth-1 channel + producer being
//                                 slower than consumer, this is race-free.
//
// Total ~15 GB device + ~4.36 GB pinned host — fits in 17 GB free VRAM on a
// 24 GB 4090.
//
// Note: T1/T2/T3 match kernels report temp_bytes = 0 (no scratch needed).
// Only the Xs phase wants ~4.34 GB of scratch, so we alias d_pair_b for that.

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
    size_t   pair_bytes         = 0;
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

    // Pinned host buffers for final T3 fragment D2H. Double-buffered so the
    // consumer can read plot N directly from one slot while producer writes
    // plot N+1 into the other — no intermediate ~2 GB heap copy per plot.
    uint64_t* h_pinned_t3[2] = {nullptr, nullptr};
};

} // namespace pos2gpu
