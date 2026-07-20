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

// Free + total device VRAM at call time. On SYCL backends without a
// portable free-memory query, free_bytes is approximated as
// total_bytes (AdaptiveCpp's global_mem_size = device total). Used as
// a preflight signal; sycl::malloc_device remains the source of
// truth. POS2GPU_MAX_VRAM_MB caps both fields when set.
struct DeviceMemInfo {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
};
DeviceMemInfo query_device_memory();

// Driver-level free/total VRAM for a device ordinal, WITHOUT touching the SYCL
// queue. query_device_memory() goes through sycl_backend::queue(), which is
// thread_local — calling it from a sampler thread would construct a second
// queue (and on some backends a second context) on that thread, which is both
// wasteful and self-defeating when the thing being measured is VRAM. This is
// the probe the VRAM watchdog polls with.
//
// Returns false when no driver-level query is available (non-CUDA build or
// host), leaving the caller to skip the measurement rather than report a
// fabricated one.
bool device_memory_probe(int device_ordinal,
                         size_t& free_bytes,
                         size_t& total_bytes);

// VRAM the picker holds back beyond a path's model peak, shared by the pool
// gate and the streaming tier picker.
//
// This is NOT an accounting number. Every path was measured (bench's watchdog,
// each ballasted to its own floor so the two-phase grant is zero) to overshoot
// its model by at most 28 MiB:
//
//   pool 10468 -> 10476 (+8)     plain   7290 -> 7304 (+14)
//   compact 5200 -> 5228 (+28)   minimal 3900 -> 3917 (+17)
//   tiny 1100 -> 1094 (-6)       pinned  1150 -> 1112 (-38)
//
// What the margin actually buys is tolerance for *other tenants* on the card —
// a compositor, a browser, a second plotter — taking VRAM after the pick. 128
// MiB leaves ~100 MiB for that. That is right for a headless plotting rig and
// thin for a GPU also driving a desktop: raise it with POS2GPU_VRAM_MARGIN_MB
// if you share the card. (Idle drift on a desktop RTX 4090 measured 31 MiB over
// 40 s, for calibration.)
//
// It was 512 MiB, which double-counted the CUDA context. free_bytes comes from
// cudaMemGetInfo, and query_device_memory constructs the SYCL queue *before*
// calling it, so the ~390 MiB context is already excluded from free — the margin
// was reserving it a second time. The watchdog's `peak` is baseline-relative
// (memory consumed after the pick), which is the quantity that has to fit inside
// free; sizing the margin against the *process* peak instead was the error. It
// cost every card 512 MiB of capability: an 8 GB card was denied `plain`, which
// consumes 7304 MiB and fits with ~290 MiB to spare, and demoted to compact.
size_t vram_safety_margin();

// Upper bound on streaming-pipeline peak device VRAM at given k.
//
// IMPORTANT for anyone re-anchoring these: the streaming-stats trace
// (POS2GPU_STREAMING_STATS=1) only sees allocations that go through s_malloc.
// It is NOT the true device peak. Raw sycl::malloc_device allocations — the
// T2/T3 two-phase candidate scratch was one, to the tune of 3128 MiB — are
// invisible to it, and calibrating an anchor against a blind instrument is
// exactly how the ladder came to under-predict every non-tiny tier by ~3 GB
// and OOM every 4-11 GB NVIDIA card. Validate against the driver's own
// accounting (`bench`, which runs the VramWatchdog), not the trace.
//
// Measured true peaks at k=28, two-phase disabled, INCLUDING the ~390 MiB
// context: tiny 1454, minimal 4274, compact 5590, plain 7680, pool 10860 MiB.
//
// streaming_peak_bytes: compact tier (anchored at 5200 MB at k=28).
//   Serves ~6 GiB cards and up.
// streaming_plain_peak_bytes: plain tier (anchored at 7290 MB at k=28,
//   pre-park pipeline — saves ~400 ms/plot over compact via fewer PCIe
//   round-trips, at the cost of the higher peak). Serves ~8 GiB and up.
// streaming_minimal_peak_bytes: minimal tier (anchored at 3900 MB at k=28).
//   Same parks as compact plus N=8 T2 match staging (cap/8 vs compact's
//   cap/2) at the cost of more PCIe round-trips during T2 match.
//
//   This used to claim it "targets 4 GiB cards". It does not, and never did:
//   the tracked peak is 3884 MB and the context adds ~390 MB, so the true
//   process peak is ~4274 MiB — more than a 4 GiB card can offer. A 4 GiB card
//   correctly lands on tiny. Minimal serves roughly 5 GiB and up.
// streaming_tiny_peak_bytes: tiny tier (anchored at 1100 MB at k=28 — the
//   header said 1500 long after the code moved to 1100). Layers further cuts
//   on top of minimal: tighter match staging, aggressive park-to-host of the
//   full Xs and intermediate streams, smaller per-tile sort scratch. Serves
//   ~2 GiB cards and up, and is the FLOOR of the auto-pick ladder.
// Dominant terms scale with 2^k, so other k extrapolate linearly.
size_t streaming_peak_bytes(int k);
size_t streaming_plain_peak_bytes(int k);
size_t streaming_minimal_peak_bytes(int k);
size_t streaming_tiny_peak_bytes(int k);
// streaming_pinned_peak_bytes: pinned tier — replaces the in-VRAM T1 sort
// gather with a streaming-partition + per-bucket-sort flow that keeps d_t1_meta
// host-resident.
//
// It is NOT a sub-Tiny tier and must not be used as one: measured at k=28 it
// peaks at 1128 MB against Tiny's 1118 MB — the same footprint, not smaller —
// so it cannot serve as the fallback below Tiny, and the auto ladder still
// floors at Tiny. What it IS, on that measurement, is a peer of Tiny that runs
// a few percent faster, reachable as `--tier pinned` / `<id>:pinned`.
//
// Its anchor said 2200 MB until 2026-07-12 — twice its real peak. That number
// was never measured at k=28: it was extrapolated from k=26, then re-derived as
// a ratio against Tiny's *old* anchor, which was itself ~3x Tiny's real peak.
// The bench watchdog cannot catch this class of error (it only fires when the
// true peak EXCEEDS the declaration); over-declaring just quietly costs the user
// card capability. Re-derive from the watchdog's true peak, never from a ratio.
size_t streaming_pinned_peak_bytes(int k);

// ---------------------------------------------------------------------------
// Host memory. The tier ladder above bounds VRAM; these bound the other side.
// ---------------------------------------------------------------------------

// Peak HOST memory a streaming tier needs at given k, including the pinned
// staging buffers, the CUDA/SYCL runtime, and the file writer's heap.
//
// Host RAM moves OPPOSITE to VRAM across the ladder, because parking tables on
// the host is what buys the VRAM back. Measured on an RTX 4090, k=28, peak RSS
// sampled from /proc/PID/status over `bench -n 1 --warmup 0`:
//
//   tier      host RSS    VRAM peak   s/plot
//   plain      7.36 GiB    7.89 GiB     7.46
//   compact   14.46 GiB    5.86 GiB     6.44
//   minimal   18.52 GiB    5.00 GiB    26.69
//   tiny      19.56 GiB    1.12 GiB    40.38
//
// So compact buys 2.03 GiB of VRAM for 7.10 GiB of host, and tiny buys 6.77
// for 12.20 — plus 5.4x the wall clock. Reaching for a lower --tier to "use
// less memory" makes host RAM worse, which is the single most confusing thing
// about the ladder and the reason these numbers are in the header.
//
// Modelled as cap(k) * bytes_per_entry + a fixed term, because every host
// buffer is a whole table: cap = 2^k + 2^(k-6), so one u64 array over it is
// 2.03 GiB at k=28 and that is the quantum the numbers move in. Occupancy is
// ~98.5% of cap, so there is no headroom to reclaim by sizing.
//
// These are ESTIMATES for planning — picking a tier, refusing early with a
// clear message. They are not the safety mechanism: an analytic model of which
// buffers a tier allocates drifts the moment a buffer is added, and this one
// already did, by 27%. host_pinned_reserve_check() below is what actually
// keeps the machine alive.
size_t streaming_plain_host_bytes(int k);
size_t streaming_compact_host_bytes(int k);
size_t streaming_minimal_host_bytes(int k);
size_t streaming_tiny_host_bytes(int k);

// Host RAM held back from pinned allocation, so the OS and other tenants keep
// a working set. Override with XCHPLOT2_HOST_RESERVE_MB.
size_t host_memory_reserve();

// Throws if committing `bytes` of pinned host memory would push the host below
// host_memory_reserve(). Inert when the host probe is unavailable.
//
// Called at every pinned-host allocation site rather than predicted once up
// front. Pinned pages are unswappable: the kernel cannot reclaim them under
// pressure, so it kills processes instead — and it does not necessarily kill
// ours. A 14 GiB box running k=28 lost its browser, dbus and session systemd
// before xchplot2 itself died, which is a failure mode no return value can
// describe. Checking per-allocation costs a /proc/meminfo read on a path that
// already does a multi-GiB malloc, and is immune to the model drift above:
// each call re-reads what is actually left, so buffers nobody remembered to
// count are accounted for by construction.
void host_pinned_reserve_check(size_t bytes, char const* what);

} // namespace pos2gpu
