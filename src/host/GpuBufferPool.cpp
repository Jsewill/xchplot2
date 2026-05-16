// GpuBufferPool.cu — queries per-phase scratch sizes once and allocates
// worst-case-sized persistent buffers. Slice 13 migrated the device and
// pinned-host allocations from the cudaMalloc / cudaMallocHost family to
// sycl::malloc_device / sycl::malloc_host on the shared SYCL queue;
// cudaMemGetInfo is left as-is because it's a context-level query that
// works regardless of which runtime is doing the allocations (SYCL +
// CUDA host code share the same primary CUDA context).

#include "host/GpuBufferPool.hpp"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"
#include "host/PoolSizing.hpp"

#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pos2gpu {

namespace {


// Allocate `bytes` of device memory on `q` and check for null. The cap-and-
// throw helpers in GpuPipeline.cu are streaming-pipeline specific; the pool
// just allocates worst-case sizes once at construction so a one-line wrap
// suffices.
// Format a byte count as "<N> bytes (<N.NN> MB)" for diagnostics. The
// raw byte count surfaces sub-MiB requests that would otherwise round
// to "0 MB"; the MB form keeps human readability for the > 1 MiB case.
inline std::string fmt_alloc_bytes(size_t bytes)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%zu bytes (%.2f MB)",
                  bytes, double(bytes) / (1024.0 * 1024.0));
    return std::string(buf);
}

// AdaptiveCpp's CUDA allocator throws sycl::exception on cudaMalloc
// failure (e.g. "cuda_allocator: cudaMalloc() failed (error code =
// CUDA:2)" for cudaErrorMemoryAllocation). Older / non-CUDA backends
// may instead return nullptr. Cover both paths with one diagnostic
// shape so callers see "sycl::malloc_device(d_pair_a, 4690 MB) failed:
// <underlying>" regardless of which branch fired. This also catches
// the throw synchronously so the async error handler doesn't log the
// same CUDA error a second time after caller cleanup.
inline void* sycl_alloc_device_or_throw(size_t bytes, sycl::queue& q,
                                        char const* what)
{
    void* p = nullptr;
    try {
        p = sycl::malloc_device(bytes, q);
    } catch (sycl::exception const& e) {
        throw std::runtime_error(
            std::string("sycl::malloc_device(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") failed: " + e.what() +
            ". Likely transient OOM — check `nvidia-smi` for other GPU "
            "consumers, or set POS2GPU_MAX_VRAM_MB lower if VRAM is "
            "shared with display/compositor.");
    }
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_device(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") returned null (out of device "
            "memory). Likely transient OOM — check `nvidia-smi` for "
            "other GPU consumers, or set POS2GPU_MAX_VRAM_MB lower if "
            "VRAM is shared with display/compositor.");
    }
    return p;
}

inline void* sycl_alloc_host_or_throw(size_t bytes, sycl::queue& q,
                                      char const* what)
{
    void* p = nullptr;
    try {
        p = sycl::malloc_host(bytes, q);
    } catch (sycl::exception const& e) {
        throw std::runtime_error(
            std::string("sycl::malloc_host(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") failed: " + e.what());
    }
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_host(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") returned null (out of host pinned memory)");
    }
    return p;
}

} // namespace

GpuBufferPool::GpuBufferPool(int k_, int strength_, bool testnet_)
    : k(k_), strength(strength_), testnet(testnet_)
{
    sycl::queue& q = sycl_backend::queue();

    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    total_xs = 1ULL << k;
    cap      = max_pairs_per_section(k, num_section_bits) * (1ULL << num_section_bits);

    // d_storage must hold EITHER total_xs XsCandidateGpu (8 B each) OR
    // THREE cap-sized uint32 key/val arrays during sort. Only three, not
    // four: the sort API signature takes a (keys_in, keys_out, vals_in,
    // vals_out) quad, but pool-path callers always pass the SoA match-info
    // stream (d_t1_mi / d_t2_mi, living in d_pair_a) as keys_in, so the
    // keys_in slot inside d_storage was never read. Dropping it saves
    // cap·4 B (~1.09 GiB at k=28) — enough to close the 0.71 GiB pool
    // shortfall on 12 GiB cards.
    storage_bytes = std::max(
        static_cast<size_t>(total_xs) * sizeof(XsCandidateGpu),
        static_cast<size_t>(cap) * 3 * sizeof(uint32_t));

    // d_pair_a holds the *match output* of the current phase: T1 SoA
    // (meta·8 B + mi·4 B = 12 B), T2 SoA (meta·8 B + mi·4 B + xbits·4 B =
    // 16 B), then T3 (T3PairingGpu, 8 B). Worst case is T2 at 16 B/entry.
    // It does NOT alias the Xs construction scratch — that's d_pair_b.
    pair_a_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(T1PairingGpu),
        static_cast<size_t>(cap) * sizeof(T2PairingGpu),
        static_cast<size_t>(cap) * sizeof(T3PairingGpu),
        static_cast<size_t>(cap) * sizeof(uint64_t),
    });

    // d_pair_b holds the *sort output* of the current phase (sorted T1
    // meta, sorted T2 meta+xbits, T3 frags) AND the Xs construction
    // scratch. Sized to the max of those.
    //
    // Split-keys_a optimisation: the pool places the Xs sort's keys_a
    // slot (total_xs·u32 = 1 GiB at k=28) in d_storage's tail — idle
    // during Xs gen+sort, and the final pack phase only writes
    // d_storage[0..total_xs·8), leaving the tail region undisturbed.
    // This drops xs_temp_bytes from ~4.36 GB (4·N·u32 + cub) to
    // ~3.22 GB (3·N·u32 + cub). At k=28 pair_b is then bounded by
    // cap·12 (sorted T2 meta+xbits = 3.27 GB) rather than xs scratch,
    // saving ~1.09 GB off the pool's peak VRAM requirement vs the
    // pre-split layout.
    uint8_t dummy_plot_id[32] = {};
    // Non-null sentinel tells launch_construct_xs to report the
    // split-layout size. The sentinel value is read only in sizing
    // mode (d_temp_storage == nullptr), where only its non-null-ness
    // matters.
    void* const xs_split_sentinel = reinterpret_cast<void*>(uintptr_t{1});
    launch_construct_xs(dummy_plot_id, k, testnet,
                                   nullptr, nullptr, &xs_temp_bytes, q,
                                   xs_split_sentinel);
    pair_b_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(uint64_t),                          // sorted T1 meta
        static_cast<size_t>(cap) * (sizeof(uint64_t) + sizeof(uint32_t)),     // sorted T2 meta+xbits
        static_cast<size_t>(cap) * sizeof(uint64_t),                          // T3 frags out
        xs_temp_bytes,                                                        // Xs aliased scratch (3·N·u32 + cub)
    });

    // Query CUB sort scratch sizes (largest across T1/T2/T3 sorts).
    size_t s_pairs = 0;
    launch_sort_pairs_u32_u32(
        nullptr, s_pairs,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        cap, 0, k, q);
    size_t s_keys = 0;
    launch_sort_keys_u64(
        nullptr, s_keys,
        static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        cap, 0, 2 * k, q);
    sort_scratch_bytes = std::max(s_pairs, s_keys);

    pinned_bytes = cap * sizeof(uint64_t);

    // Check VRAM before attempting allocation so we can give a useful
    // diagnostic instead of a generic allocation failure. The margin covers
    // GPU driver/context state, sort scratch, AES T-tables, and other small
    // runtime allocations.
    //
    // SYCL has no portable free-memory query, so slice 17c approximates
    // free_b == total_b. The actual sycl::malloc_device call will throw if
    // VRAM is exhausted; the diagnostic message is just less precise about
    // how much of the total is already consumed by other processes.
    {
        size_t const required_device =
            storage_bytes + pair_a_bytes + pair_b_bytes + sort_scratch_bytes + sizeof(uint64_t);
        // Margin covers per-context driver state + AES T-tables + the
        // tiny (sizeof(uint64_t)) d_counter alloc that's not counted in
        // sort_scratch. Originally 512 MB (slice 17c); trimmed to 256 MB
        // after measuring actual runtime overhead on gfx1031/ROCm 6.2
        // and sm_89/CUDA 13: both land under 150 MB of non-pool device
        // allocations, so a 256 MB margin leaves >100 MB headroom while
        // letting cards on the threshold (e.g. 12 GiB reporting ~11.8
        // GiB free at ctor time) now succeed into the pool path.
        size_t const margin = 256ULL * 1024 * 1024; // 256 MB
        size_t const total_b =
            q.get_device().get_info<sycl::info::device::global_mem_size>();
        size_t const free_b = total_b;  // approximation — see comment above
        if (free_b < required_device + margin) {
            auto to_gib = [](size_t b) { return b / double(1ULL << 30); };
            InsufficientVramError e(
                "GpuBufferPool: insufficient device VRAM for k=" +
                std::to_string(k) + " strength=" + std::to_string(strength) +
                "; need ~" + std::to_string(to_gib(required_device + margin)).substr(0, 5) +
                " GiB (pool " + std::to_string(to_gib(required_device)).substr(0, 5) +
                " GiB + ~0.25 GiB runtime), only " +
                std::to_string(to_gib(free_b)).substr(0, 5) +
                " GiB free of " + std::to_string(to_gib(total_b)).substr(0, 5) +
                " GiB total. Use a smaller k or a GPU with more VRAM.");
            e.required_bytes = required_device + margin;
            e.free_bytes     = free_b;
            e.total_bytes    = total_b;
            throw e;
        }
    }

    if (getenv("POS2GPU_POOL_DEBUG")) {
        size_t const total_b =
            q.get_device().get_info<sycl::info::device::global_mem_size>();
        std::fprintf(stderr,
            "[pool] k=%d strength=%d cap=%llu total_xs=%llu "
            "total=%.2fGB (free unavailable in SYCL build)\n",
            k, strength, (unsigned long long)cap, (unsigned long long)total_xs,
            total_b/1e9);
        std::fprintf(stderr,
            "[pool] sizes: storage=%.2fGB pair_a=%.2fGB pair_b=%.2fGB "
            "xs_temp(alias→pair_b)=%.2fGB sort_scratch=%.2fGB pinned=%.2fGB\n",
            storage_bytes/1e9, pair_a_bytes/1e9, pair_b_bytes/1e9,
            xs_temp_bytes/1e9, sort_scratch_bytes/1e9, pinned_bytes/1e9);
    }

    // Wrap allocations so a mid-sequence failure (e.g. d_pair_b OOM after
    // d_storage + d_pair_a have already succeeded) frees the pre-allocated
    // buffers instead of leaking ~10 GB of device VRAM and ~7 GB of host
    // pinned memory per failed pool ctor across a batch retry loop.
    auto cleanup_partial = [&]{
        if (d_storage)       { sycl::free(d_storage,      q); d_storage      = nullptr; }
        if (d_pair_a)        { sycl::free(d_pair_a,       q); d_pair_a       = nullptr; }
        if (d_pair_b)        { sycl::free(d_pair_b,       q); d_pair_b       = nullptr; }
        if (d_sort_scratch)  { sycl::free(d_sort_scratch, q); d_sort_scratch = nullptr; }
        if (d_counter)       { sycl::free(d_counter,      q); d_counter      = nullptr; }
        for (int i = 0; i < kNumPinnedBuffers; ++i) {
            if (h_pinned_t3[i]) { sycl::free(h_pinned_t3[i], q); h_pinned_t3[i] = nullptr; }
        }
    };
    try {
        d_storage      = sycl_alloc_device_or_throw(storage_bytes,      q, "d_storage");
        // d_pair_a is allocated lazily in ensure_pair_a(), called by
        // run_gpu_pipeline's pool path right after submitting Xs gen
        // — the malloc_device then overlaps with Xs GPU execution.
        // Saves ~400-500 ms on first-plot wall vs eager alloc; batch
        // plots 2+ are unaffected (fast-path pointer lookup).
        d_pair_b       = sycl_alloc_device_or_throw(pair_b_bytes,       q, "d_pair_b");
        d_sort_scratch = sycl_alloc_device_or_throw(sort_scratch_bytes, q, "d_sort_scratch");
        d_counter      = static_cast<uint64_t*>(
            sycl_alloc_device_or_throw(sizeof(uint64_t),                q, "d_counter"));
        // h_pinned_t3[] is allocated lazily in ensure_pinned(); see
        // the header comment for why. Single-plot runs only ever
        // touch slot 0 so the other two 2.2 GB malloc_host calls
        // aren't paid at all.
    } catch (...) {
        cleanup_partial();
        throw;
    }
}

void* GpuBufferPool::ensure_pair_a()
{
    if (d_pair_a) return d_pair_a;
    std::lock_guard<std::mutex> lk(pair_a_mu_);
    if (d_pair_a) return d_pair_a;
    sycl::queue& q = sycl_backend::queue();
    d_pair_a = sycl_alloc_device_or_throw(pair_a_bytes, q, "d_pair_a");
    return d_pair_a;
}

void GpuBufferPool::release_pair_a()
{
    std::lock_guard<std::mutex> lk(pair_a_mu_);
    if (!d_pair_a) return;
    sycl::free(d_pair_a, sycl_backend::queue());
    d_pair_a = nullptr;
}

uint64_t* GpuBufferPool::ensure_pinned(int idx)
{
    if (idx < 0 || idx >= kNumPinnedBuffers) {
        throw std::runtime_error("GpuBufferPool::ensure_pinned: idx out of range");
    }
    // Double-checked locking: fast path skips the mutex once the
    // slot's pointer is visible. Writes inside the mutex are
    // release-ordered w.r.t. the mutex release; the unlocked read
    // on the fast path is an acquire (relaxed access is fine here
    // because x86 and arm64 give us acquire ordering for aligned
    // pointer reads; if this ever needs to be portable to weaker
    // architectures, make h_pinned_t3 std::atomic<uint64_t*>[]).
    if (h_pinned_t3[idx]) return h_pinned_t3[idx];
    std::lock_guard<std::mutex> lk(pinned_mu_[idx]);
    if (h_pinned_t3[idx]) return h_pinned_t3[idx];
    sycl::queue& q = sycl_backend::queue();
    h_pinned_t3[idx] = static_cast<uint64_t*>(
        sycl_alloc_host_or_throw(pinned_bytes, q, "h_pinned_t3"));
    return h_pinned_t3[idx];
}

GpuBufferPool::~GpuBufferPool()
{
    sycl::queue& q = sycl_backend::queue();
    if (d_storage)       sycl::free(d_storage,      q);
    if (d_pair_a)        sycl::free(d_pair_a,       q);
    if (d_pair_b)        sycl::free(d_pair_b,       q);
    if (d_sort_scratch)  sycl::free(d_sort_scratch, q);
    if (d_counter)       sycl::free(d_counter,      q);
    for (int i = 0; i < kNumPinnedBuffers; ++i) {
        if (h_pinned_t3[i]) sycl::free(h_pinned_t3[i], q);
    }
}

DeviceMemInfo query_device_memory()
{
    sycl::queue& q = sycl_backend::queue();
    DeviceMemInfo info;
    info.total_bytes =
        q.get_device().get_info<sycl::info::device::global_mem_size>();
    // SYCL has no portable free-memory query; AdaptiveCpp's
    // global_mem_size returns the device total. On the CUDA backend
    // the underlying driver often subtracts active reservations
    // (framebuffer, compositor) before reporting, which gets us
    // closer to "free" in practice. Treat the result as an upper
    // bound; sycl::malloc_device is still the source of truth.
    info.free_bytes = info.total_bytes;

    if (char const* v = std::getenv("POS2GPU_MAX_VRAM_MB"); v && v[0]) {
        size_t const cap = size_t(std::strtoull(v, nullptr, 10)) * (1ULL << 20);
        info.free_bytes  = std::min(info.free_bytes,  cap);
        info.total_bytes = std::min(info.total_bytes, cap);
    }
    return info;
}

namespace {

// CUB's DeviceRadixSort temp_storage_bytes at k=28 with our key/val
// shape lands around 64-128 MB on sm_89; the streaming peak anchors
// below were measured with that overhead already live, so they
// implicitly budget for it. AdaptiveCpp's HIP backend routes the
// same `launch_sort_*` calls through a hand-rolled SYCL radix in
// SortSycl.cpp that uses ping-pong buffers sized to the input —
// multi-GiB at k=28, far exceeding what CUB's in-place radix needs.
// The streaming peak prediction has to add that excess so dispatch
// in BatchPlotter doesn't pick a tier whose "predicted peak" is
// several GiB short of the actual T1-sort live, the way an 8 GiB
// W5700 (gfx1010 → gfx1013 spoof) currently does.
//
// Baseline set at 256 MB at k=28 (a touch over CUB's typical scratch
// on sm_89 to keep headroom on NVIDIA cards near the threshold) and
// scaled 2× per +k step (linear in cap, matching how CUB's actual
// DeviceRadixSort scratch grows). The returned adjustment is
// `max(0, runtime_sort_scratch - baseline)`, so NVIDIA hosts whose
// runtime scratch is at or below the baseline see no change in
// predicted peak.
inline size_t streaming_sort_scratch_adjustment(int k)
{
    constexpr size_t cub_baseline_at_k28_bytes = 256ULL << 20;

    sycl::queue& q = sycl_backend::queue();
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    size_t const cap_for_k =
        max_pairs_per_section(k, num_section_bits) * (1ULL << num_section_bits);

    size_t s_pairs = 0;
    launch_sort_pairs_u32_u32(
        nullptr, s_pairs,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        cap_for_k, 0, k, q);
    size_t s_keys = 0;
    launch_sort_keys_u64(
        nullptr, s_keys,
        static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        cap_for_k, 0, 2 * k, q);
    size_t const actual = std::max(s_pairs, s_keys);

    int const dk = k - 28;
    size_t baseline = cub_baseline_at_k28_bytes;
    if (dk > 0)      baseline <<= dk;
    else if (dk < 0) baseline >>= -dk;

    return (actual > baseline) ? (actual - baseline) : 0;
}

} // namespace

size_t streaming_peak_bytes(int k)
{
    // Anchor: 5200 MB at k=28 (measured post-stage-4e on sm_89).
    // After the full T1/T2/T3 match/sort work (stages 1-4d) + Xs
    // gen+sort+pack inlining (4e), all match + sort phases cap out at
    // cap·sizeof(uint64_t) × ~2.5 aliases = ~5200 MB. Xs peak is 4128,
    // T3 sort 4228, all others ≤ 5200. Dominant terms scale with 2^k.
    constexpr size_t anchor_mb = 5200;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;       // floor for tiny test plots
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;  // cap halves per −1 in k → 2× smaller
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_plain_peak_bytes(int k)
{
    // Anchor: 7290 MB at k=28 (pre-stage-1-4 peak — d_t1_meta +
    // d_t1_keys_merged + d_t2_meta + d_t2_mi + d_t2_xbits all live
    // concurrently during T2 match, no parks). Plain tier skips all
    // park/rehydrate round-trips for ~400 ms/plot over compact at the
    // cost of this higher peak. Scales the same way as compact.
    constexpr size_t anchor_mb = 7290;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_minimal_peak_bytes(int k)
{
    // Anchor: 3760 MB at k=28 (measured 3754 MB on sm_89 + the
    // streaming-stats trace; rounded up for safety). Bottleneck is T3
    // match where d_t2_keys_merged + d_t2_xbits_sorted + meta-l/r
    // slices + d_t3_stage are co-resident.
    //
    // Minimal layers cumulative cuts on top of compact:
    //   1. N=8 T2 match staging (cap/8 ≈ 570 MB vs compact's cap/2).
    //   2. T1 sort gather, T2 sort meta+xbits gathers — tiled output,
    //      D2H per tile to host pinned, rebuild on device after free.
    //   3. T3 match — d_t2_meta_sorted parked on host pinned, sliced
    //      device buffers H2D'd per (section_l, section_r) pass.
    //   4. T1 match — sliced into N passes per section_l, output
    //      accumulated to host pinned.
    //   5. T1, T2, T3 sort CUB sub-phases — per-tile cap/N output
    //      buffers, USM-host accumulation, merges with USM-host inputs.
    //   6. Xs phase — gen+sort tiled in N=2 position halves with
    //      USM-host accumulators; pack tiled with D2H per tile.
    //
    // Cumulative effect at k=28: peak drops from 5200 MB (compact) →
    // 3754 MB (minimal). Trade-off: ~6 extra cap-sized PCIe round-
    // trips per plot (~2.5× wall on NVIDIA — 13 s/plot → 34 s/plot
    // at k=28). Same k-scaling as compact / plain.
    constexpr size_t anchor_mb = 3760;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_tiny_peak_bytes(int k)
{
    // Anchor: 1250 MB at k=28. Tiny absorbed the Phase 1.4 + 1.5
    // algorithms that were originally developed under the "Pinned"
    // tier name. After Phase 1.6 sub-section attacks (per-bucket-pair
    // T1/T2/T3 match + host-side T2/T3 prepare offsets), measured on
    // RTX 4090:
    //   k=22:  ~22 MB →  ~352 MB extrapolated at k=28
    //   k=24:   92 MB → ~1472 MB extrapolated at k=28
    //   k=26:  288 MB → ~1152 MB extrapolated at k=28
    // The k=24 → k=28 extrapolation is the conservative one; set
    // anchor to 1250 MB — ~8% safety margin above the k=26 → k=28
    // line. The current floor is T2 sort scratch (CUB tile_max-sized
    // workspace at 288 MB at k=26 / ~1152 MB at k=28); the match
    // phases are all at ~256 MB.
    //
    // What Tiny now does (all the host-park + streaming techniques):
    //   - Xs: CPU merge+pack to host h_xs, no device d_xs_keys_b/vals_b
    //     intermediate (Phase 1.4a+b)
    //   - T1 match: per-section-pair tile H2D from h_xs, no full-cap
    //     d_xs on device (Phase 1.4c)
    //   - T1 sort: streaming partition (top-bits bucket) + per-bucket
    //     u32_u64 sort, no full-cap d_t1_meta on device (Phase 1.3c-ii)
    //   - T2 sort: streaming partition with triple-val (key/meta/xbits
    //     paired through duplicate keys) + per-bucket sort, no
    //     full-cap d_t2_meta on device (Phase 1.5b)
    //   - T3 match: d_t3_stage allocated as USM-host so device peak
    //     drops by ~200 MB at k=26 / ~800 MB at k=28 (Phase 1.5c-a)
    //   - T3 sort: N=4 tile + multi-way host merge (vs N=2 before)
    //
    // Wall trade vs the original (pre-promotion) Tiny implementation:
    // approximately +16% at k=26 on RTX 4090. Acceptable on target
    // hardware (2-3 GB GPUs) which couldn't run the original Tiny at
    // all. Larger cards should use Plain/Compact/Minimal which are
    // unchanged.
    //
    // Going below ~1.1 GB at k=28 requires attacking the T2 sort
    // CUB scratch (the new floor — 288 MB at k=26 / ~1152 MB at
    // k=28). Options: (a) finer per-bucket sort with smaller cub
    // scratch, (b) host-side merge of pre-sorted partition tiles,
    // (c) Phase 2 Disk tier for spill.
    constexpr size_t anchor_mb = 1250;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

size_t streaming_pinned_peak_bytes(int k)
{
    // Anchor: 2900 MB at k=28. After Phase 1.3c-ii + 1.4a/b/c the
    // Pinned tier eliminates the T1-sort-gather floor (d_t1_meta
    // full-cap on device, ~2 GB at k=28) AND the Xs phase floor
    // (d_xs_keys_b + d_xs_vals_b + d_xs_pack_tile, ~3 GB at k=28,
    // plus the d_xs rehydrate ~2 GB). What remains is the T2 sort /
    // T3 match phase peak. Measured at k=26: Pinned 720 MB / Tiny
    // 792 MB → ~91% of Tiny. Extrapolated to k=28 (which scales
    // ~4× from k=26 for the dominant terms): Pinned ≈ 2900 MB.
    //
    // The spec's original 1500 MB-at-k=28 target is unreachable
    // without also streaming T2 sort and T3 match phases — those
    // are now the floor (project_streaming_pinned_disk_spec
    // memory documents the analysis). Phase 1.5+ would attack
    // those; not currently scoped.
    //
    // Auto-picker window for k=28: free VRAM in [2900, 3200) MB
    // picks Pinned instead of Tiny. Outside that window the picker
    // behaves as before (Tiny for >= 3200, throw below 2900 - margin).
    // Post-1.5c-a measurement: Pinned is ~68% of Tiny at k=22/24/26
    // on RTX 4090. Lower the anchor from 2900 to 2200 MB at k=28
    // (= 3200 * 0.68 ≈ 2176, round up to 2200 for safety margin).
    // This widens the auto-picker's Pinned-only window from
    // [2900, 3200) MB to [2200, 3200) MB.
    constexpr size_t anchor_mb = 2200;
    size_t const adj = streaming_sort_scratch_adjustment(k);
    if (k == 28) return (anchor_mb << 20) + adj;
    if (k <  18) return (size_t(16) << 20) + adj;
    if (k >  32) return (size_t(anchor_mb) << (20 + (32 - 28))) + adj;

    if (k < 28) {
        int const shift = 28 - k;
        return ((size_t(anchor_mb) << 20) >> shift) + adj;
    }
    int const shift = k - 28;
    return ((size_t(anchor_mb) << 20) << shift) + adj;
}

} // namespace pos2gpu
