// GpuBufferPool.hpp — owns all device and pinned host buffers needed by
// run_gpu_pipeline(), sized once at construction and reused across plots.
//
// Motivation: per-plot cudaMalloc / cudaMallocHost calls cost ~2.4 s in a
// k=28 batch run (dominated by cudaMallocHost on a 2 GB pinned region,
// ~600 ms). Amortising that across a batch of plots removes the gap
// between device time (~2.75 s) and producer wall time (~5.1 s).
//
// Memory layout with aliasing (k=28 worst-case sizes in parens):
//   d_storage      (~3.3 GB)  — Xs candidates during Xs phase (2.1 GB),
//                               then 3×uint32[cap] sort keys_out/vals_in/
//                               vals_out during sorts. The fourth
//                               (keys_in) slot the sort API would want
//                               is ALWAYS the SoA match-info stream
//                               from d_pair_a (d_t1_mi / d_t2_mi), so
//                               d_storage doesn't allocate for it —
//                               saves cap·4 B (~1.09 GiB at k=28) vs
//                               the old 4-slot layout.
//   d_pair_a       (~4.4 GB)  — T1/T2/T3 match output (reused across phases).
//                               Sized to the largest match-output: cap·16 B
//                               for T2 (meta+mi+xbits SoA). Does NOT alias the
//                               Xs phase scratch — that lives in d_pair_b.
//   d_pair_b       (~4.4 GB)  — *_sorted / frags_out (reused across phases),
//                               AND the Xs construction scratch. Sized to
//                               max(largest sorted-output, xs_temp_bytes);
//                               at k=28 xs_temp dominates.
//   d_sort_scratch (~MB)      — Radix sort scratch. After ping-pong refactor:
//                               CUB DoubleBuffer mode shrinks this from ~2 GB
//                               to ~MB; SortSycl already ping-pongs over the
//                               caller's keys_in/keys_out buffers.
//   d_counter      (8 B)      — reused uint64_t count output
//   h_pinned_t3[N] (~2.2 GB ea) — rotating final-fragments DMA targets.
//                                 Producer writes plot K into slot K mod N
//                                 while consumer reads earlier plots from
//                                 the other slots; channel depth N-1 keeps
//                                 the producer from overwriting in-flight
//                                 reads. N defaults to 3 (see kNumPinnedBuffers).
//
// Total ~12 GB device + ~6.6 GB pinned host at k=28 — fits (just) in the
// 11.98 GiB free VRAM of a Navi 22 (RX 6700 XT) after the d_storage
// slot-trim above. Pre-trim the total was ~13.1 GB and overshot this
// card's budget by ~0.7 GiB, forcing a fallback to the streaming
// pipeline which costs an extra ~5 s at k=28.
//
// Note: T1/T2/T3 match kernels report temp_bytes = 0 (no scratch needed).
// Only the Xs phase wants ~4.4 GB of scratch, and we alias d_pair_b for that.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
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
    //
    // Pinned slots are allocated LAZILY on first use via
    // ensure_pinned(idx). The ctor no longer pays ~1.8 s at k=28
    // for the 3 × 2.2 GB malloc_host calls; single-plot runs
    // (plot -n 1) only ever allocate slot 0, saving ~1.2 s of
    // ctor time. Batch runs (plot -n N, N ≥ 3) amortise the
    // allocation cost across the first three plots' D2H phases
    // instead of the ctor — identical total batch time.
    static constexpr int kNumPinnedBuffers = 3;
    uint64_t* h_pinned_t3[kNumPinnedBuffers] = {};

    // Returns pool.h_pinned_t3[idx], allocating the slot if it
    // hasn't been used yet. Thread-safe via a per-slot mutex
    // (concurrent callers with the same idx cooperate through
    // double-checked locking; different idx values proceed
    // independently). Throws std::runtime_error on host alloc
    // failure.
    uint64_t* ensure_pinned(int idx);

    // Returns pool.d_pair_a, allocating it on first use. Deferred
    // from ctor so run_gpu_pipeline can submit Xs gen *before*
    // paying this 4.36 GB malloc_device. Thread-safe via double-
    // checked locking on pair_a_mu_.
    //
    // Measured on RX 6700 XT / ROCm 6.2 / AdaptiveCpp HIP:
    // sycl::malloc_device of 4.36 GB takes ~5 ms (the driver
    // almost certainly just reserves virtual-address space and
    // defers physical commit to first write). Overlap benefit
    // vs eager alloc is therefore ~5 ms in practice, below noise.
    // The lazy pattern is kept because (a) it's a drop-in
    // replacement with zero regression, (b) it mirrors
    // ensure_pinned, and (c) it enables release_pair_a() below.
    void* ensure_pair_a();

    // Frees d_pair_a if it's allocated, so a subsequent
    // ensure_pair_a() will re-allocate. Called by the pool path
    // at the end of each plot in a batch to shrink the
    // inter-plot VRAM peak. With ~5 ms malloc on AMD, the
    // release-and-realloc cost is below noise per plot, while
    // the 4.36 GB VRAM freed during file-write / D2H-consume
    // phases lets the pool path fit cards with ~7-8 GiB free
    // that would otherwise hit the InsufficientVramError path
    // and fall back to streaming.
    //
    // Thread-safe via pair_a_mu_; lock-order is
    // (pair_a_mu_ → sycl::free) so release can run concurrently
    // with a future ensure_pair_a from a different thread
    // without deadlock. In practice run_batch is single-producer
    // so contention is zero.
    void release_pair_a();

private:
    std::mutex pinned_mu_[kNumPinnedBuffers];
    std::mutex pair_a_mu_;
};

} // namespace pos2gpu
