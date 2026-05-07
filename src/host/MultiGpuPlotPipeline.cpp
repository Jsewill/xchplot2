// MultiGpuPlotPipeline.cpp — Phase 2.2 (Xs) + Phase 2.3a (T1) +
// Phase 2.3b (T2) + Phase 2.3c (T3) + Phase 2.3d (fragment) impl.
//
// ---------------------------------------------------------------------
// Architecture notes
//
// Replicate-full strategy. Every match phase (T1, T2, T3) currently
// host-bounces the FULL output of the previous phase to every shard,
// then runs match over the shard's assigned bucket subset. This is
// simple and correct but uses ~N × phase_output bytes of VRAM total
// (N = shard count). At k=28 with N=2 that's roughly 4 GB Xs +
// 3.2 GB T1 + 3.2 GB T2 of replicated input, on top of the per-shard
// match scratch. Tight on 12 GB cards once T2/T3 staging is also
// live.
//
// Two ways to reduce, both significant work and tracked here so the
// next pass at sharded perf knows where to look:
//   (a) Section-pair fetch — a shard processing buckets in section_l
//       only needs section_l's rows + matching_section(section_l)'s
//       rows, not the full input. At num_sections=4 that's a 2x
//       reduction in replicated data.
//   (b) Shared-context single-pointer reads — on backends that share
//       a primary context across devices (AdaptiveCpp's CUDA backend
//       on a single host fits this), every shard could read the same
//       device pointer instead of allocating its own copy. CUDA's
//       peer access handles the cross-device fetch transparently.
//       Trade-off: the source-device's VRAM bandwidth is shared by
//       all consumers.
//
// Distributed sort destination assignment stays uniform (the
// `bucket_of_u32` formula in SortDistributed.cpp). Phase 2.4a
// introduced weighted bucket assignment for the match phases but did
// NOT propagate to the sort destination — that's deliberate, because
// the host-bounce replication makes the match input the same on
// every shard regardless of which shard "received" it from the sort.
// Only the emit-side weighting matters for load balancing. If the
// replicate-full strategy ever changes (see (a)/(b) above), the sort
// destination would need to follow the weight too.
// ---------------------------------------------------------------------

#include "host/MultiGpuPlotPipeline.hpp"

#include "gpu/AesHashGpu.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SortDistributed.hpp"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"
#include "gpu/XsCandidateGpu.hpp"
#include "gpu/XsKernels.cuh"
#include "host/PoolSizing.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string>
#include <string_view>
#include <vector>

namespace pos2gpu {

namespace {

// Move-only RAII for sycl::malloc_device / malloc_host allocations.
// Holds (queue*, ptr); frees in dtor unless release() was called.
// Used to keep per-phase intermediate scratch from leaking when a
// downstream launch throws mid-loop.
template <class T>
struct SyclDevicePtr {
    sycl::queue* q = nullptr;
    T* p = nullptr;

    SyclDevicePtr() = default;
    SyclDevicePtr(sycl::queue* q_, T* p_) : q(q_), p(p_) {}
    SyclDevicePtr(SyclDevicePtr const&) = delete;
    SyclDevicePtr& operator=(SyclDevicePtr const&) = delete;
    SyclDevicePtr(SyclDevicePtr&& o) noexcept : q(o.q), p(o.p) { o.p = nullptr; }
    SyclDevicePtr& operator=(SyclDevicePtr&& o) noexcept {
        if (this != &o) { reset(); q = o.q; p = o.p; o.p = nullptr; }
        return *this;
    }
    ~SyclDevicePtr() { reset(); }

    void reset() noexcept {
        if (p && q) { sycl::free(p, *q); p = nullptr; }
    }
    T* release() noexcept { T* tmp = p; p = nullptr; return tmp; }
    T* get() const noexcept { return p; }
};

template <class T>
SyclDevicePtr<T> sycl_alloc_device_owned(std::size_t n, sycl::queue& q)
{
    return SyclDevicePtr<T>(&q, sycl::malloc_device<T>(n, q));
}

// Untyped variant — distributed sort scratch + match-prepare temp
// arrive as void* so we can't directly use SyclDevicePtr<void>
// (sycl::free on void* still needs a queue). Functionally identical.
struct SyclDeviceVoid {
    sycl::queue* q = nullptr;
    void* p = nullptr;

    SyclDeviceVoid() = default;
    SyclDeviceVoid(sycl::queue* q_, void* p_) : q(q_), p(p_) {}
    SyclDeviceVoid(SyclDeviceVoid const&) = delete;
    SyclDeviceVoid& operator=(SyclDeviceVoid const&) = delete;
    SyclDeviceVoid(SyclDeviceVoid&& o) noexcept : q(o.q), p(o.p) { o.p = nullptr; }
    SyclDeviceVoid& operator=(SyclDeviceVoid&& o) noexcept {
        if (this != &o) { reset(); q = o.q; p = o.p; o.p = nullptr; }
        return *this;
    }
    ~SyclDeviceVoid() { reset(); }

    void reset() noexcept {
        if (p && q) { sycl::free(p, *q); p = nullptr; }
    }
    void* get() const noexcept { return p; }
};

// Pinned host: same lifetime story but allocated via malloc_host.
template <class T>
struct SyclHostPtr {
    sycl::queue* q = nullptr;
    T* p = nullptr;

    SyclHostPtr() = default;
    SyclHostPtr(sycl::queue* q_, T* p_) : q(q_), p(p_) {}
    SyclHostPtr(SyclHostPtr const&) = delete;
    SyclHostPtr& operator=(SyclHostPtr const&) = delete;
    SyclHostPtr(SyclHostPtr&& o) noexcept : q(o.q), p(o.p) { o.p = nullptr; }
    SyclHostPtr& operator=(SyclHostPtr&& o) noexcept {
        if (this != &o) { reset(); q = o.q; p = o.p; o.p = nullptr; }
        return *this;
    }
    ~SyclHostPtr() { reset(); }

    void reset() noexcept {
        if (p && q) { sycl::free(p, *q); p = nullptr; }
    }
    T* get() const noexcept { return p; }
};

template <class T>
SyclHostPtr<T> sycl_alloc_host_owned(std::size_t n, sycl::queue& q)
{
    return SyclHostPtr<T>(&q, sycl::malloc_host<T>(n, q));
}

// Pool-or-malloc helper. When the shard has a buffer pool attached
// (run_batch_sharded sets one up per shard at batch start), the large
// per-phase allocations are routed through the pool so consecutive
// plots reuse the buffers — same VRAM footprint, no per-plot malloc
// cost. The returned SyclDevicePtr is non-owning when from the pool
// (q == nullptr makes reset()/dtor a no-op); the pool's destructor
// frees on its own schedule.
template <class T>
SyclDevicePtr<T> pool_or_alloc(
    ShardBufferPool* pool, std::string_view label,
    std::uint64_t n, sycl::queue& q)
{
    if (pool) {
        return SyclDevicePtr<T>(nullptr, pool->ensure<T>(label, n));
    }
    return sycl_alloc_device_owned<T>(n, q);
}

// Partition `num_buckets` across the shards proportional to their
// weights. Returns a vector of size N+1 where shard s owns the range
// [partition[s], partition[s+1]). Boundary rounding is to the nearest
// integer with a clamp to ensure each shard has at least one bucket
// (otherwise a near-zero weight would produce an empty range and the
// kernel launch would early-exit, leaving that shard idle but the
// distributed sort still expecting non-zero data — a wedge). The last
// boundary is always exactly num_buckets.
std::vector<std::uint32_t> compute_bucket_partition(
    std::vector<MultiGpuShardContext> const& shards,
    std::uint32_t num_buckets)
{
    std::size_t const N = shards.size();
    if (N == 0) {
        throw std::runtime_error(
            "compute_bucket_partition: shards.empty()");
    }
    if (num_buckets < N) {
        throw std::runtime_error(
            "compute_bucket_partition: num_buckets ("
            + std::to_string(num_buckets) + ") < shard count ("
            + std::to_string(N) + "). Cannot give every shard at "
            "least one bucket; either reduce shard count or use a "
            "(k, strength) producing more buckets.");
    }

    double total = 0.0;
    for (auto const& s : shards) {
        if (!(s.weight > 0.0)) {
            throw std::runtime_error(
                "compute_bucket_partition: every shard must have a "
                "positive weight (got " + std::to_string(s.weight)
                + ").");
        }
        total += s.weight;
    }

    std::vector<std::uint32_t> partition(N + 1, 0);
    double cum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        cum += shards[i].weight;
        std::uint32_t b = static_cast<std::uint32_t>(
            std::round(cum / total * static_cast<double>(num_buckets)));
        // Monotonic + at least one bucket per shard. The arithmetic
        // above is monotonic but rounding can collide — clamp.
        std::uint32_t const min_b = partition[i] + 1u;
        if (b < min_b) b = min_b;
        if (b > num_buckets) b = num_buckets;
        partition[i + 1] = b;
    }
    // Force the last boundary exactly even after the clamp dance so
    // every bucket is owned (no off-by-one in the last shard's range).
    partition[N] = num_buckets;
    if (partition[N] < partition[N - 1] + 1u) {
        // The clamp earlier should have prevented this, but if it
        // didn't (e.g. extreme weights with num_buckets == N), surface
        // a clear error rather than launching with an empty range.
        throw std::runtime_error(
            "compute_bucket_partition: unable to give shard "
            + std::to_string(N - 1) + " at least one bucket "
            "(num_buckets=" + std::to_string(num_buckets)
            + ", N=" + std::to_string(N) + ").");
    }
    return partition;
}

// u64 sibling: partition a `total` count of items (positions, not
// buckets) across the shards proportional to their weights. Used by
// the Xs gen step where each shard generates Xs for an interval
// [pos_begin, pos_end) of the 2^k position space — fast shards take
// proportionally larger intervals so Xs gen runtime tracks match
// runtime.
std::vector<std::uint64_t> compute_position_partition(
    std::vector<MultiGpuShardContext> const& shards,
    std::uint64_t total)
{
    std::size_t const N = shards.size();
    if (N == 0) {
        throw std::runtime_error(
            "compute_position_partition: shards.empty()");
    }

    double total_weight = 0.0;
    for (auto const& s : shards) {
        if (!(s.weight > 0.0)) {
            throw std::runtime_error(
                "compute_position_partition: every shard must have a "
                "positive weight (got " + std::to_string(s.weight) + ").");
        }
        total_weight += s.weight;
    }

    std::vector<std::uint64_t> partition(N + 1, 0);
    double cum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        cum += shards[i].weight;
        // Monotonic non-decreasing; not clamped to "at least one
        // position per shard" because the Xs gen kernel handles c==0
        // gracefully (no work; subsequent steps see shard_in_count=0).
        partition[i + 1] = static_cast<std::uint64_t>(
            std::round(cum / total_weight * static_cast<double>(total)));
        if (partition[i + 1] < partition[i]) partition[i + 1] = partition[i];
        if (partition[i + 1] > total)        partition[i + 1] = total;
    }
    partition[N] = total;
    return partition;
}

} // namespace

MultiGpuPlotPipeline::MultiGpuPlotPipeline(
        BatchEntry const& entry,
        BatchOptions const& opts,
        std::vector<MultiGpuShardContext> shards)
    : entry_(entry), opts_(opts), shards_(std::move(shards))
{
    if (shards_.empty()) {
        throw std::runtime_error(
            "MultiGpuPlotPipeline: no shards — caller must provide at "
            "least one MultiGpuShardContext.");
    }
    for (std::size_t i = 0; i < shards_.size(); ++i) {
        if (!shards_[i].queue) {
            throw std::runtime_error(
                std::string("MultiGpuPlotPipeline: shard ")
                + std::to_string(i) + " has null queue pointer.");
        }
    }
    xs_phase_d_xs_.assign(shards_.size(), nullptr);
    xs_phase_count_.assign(shards_.size(), 0);
    t1_phase_d_mi_.assign(shards_.size(), nullptr);
    t1_phase_d_meta_.assign(shards_.size(), nullptr);
    t1_phase_count_.assign(shards_.size(), 0);
    t2_phase_d_mi_.assign(shards_.size(), nullptr);
    t2_phase_d_meta_.assign(shards_.size(), nullptr);
    t2_phase_d_xbits_.assign(shards_.size(), nullptr);
    t2_phase_count_.assign(shards_.size(), 0);
    t3_phase_d_frags_.assign(shards_.size(), nullptr);
    t3_phase_count_.assign(shards_.size(), 0);
}

MultiGpuPlotPipeline::~MultiGpuPlotPipeline()
{
    free_phase_outputs();
}

void MultiGpuPlotPipeline::free_phase_outputs()
{
    for (std::size_t k = 0; k < shards_.size(); ++k) {
        if (xs_phase_d_xs_[k]) {
            sycl::free(xs_phase_d_xs_[k], *shards_[k].queue);
            xs_phase_d_xs_[k] = nullptr;
        }
        if (t1_phase_d_mi_[k]) {
            sycl::free(t1_phase_d_mi_[k], *shards_[k].queue);
            t1_phase_d_mi_[k] = nullptr;
        }
        if (t1_phase_d_meta_[k]) {
            sycl::free(t1_phase_d_meta_[k], *shards_[k].queue);
            t1_phase_d_meta_[k] = nullptr;
        }
        if (t2_phase_d_mi_[k]) {
            sycl::free(t2_phase_d_mi_[k], *shards_[k].queue);
            t2_phase_d_mi_[k] = nullptr;
        }
        if (t2_phase_d_meta_[k]) {
            sycl::free(t2_phase_d_meta_[k], *shards_[k].queue);
            t2_phase_d_meta_[k] = nullptr;
        }
        if (t2_phase_d_xbits_[k]) {
            sycl::free(t2_phase_d_xbits_[k], *shards_[k].queue);
            t2_phase_d_xbits_[k] = nullptr;
        }
        if (t3_phase_d_frags_[k]) {
            sycl::free(t3_phase_d_frags_[k], *shards_[k].queue);
            t3_phase_d_frags_[k] = nullptr;
        }
    }
    if (h_fragments_) {
        sycl::free(h_fragments_, *shards_[0].queue);
        h_fragments_     = nullptr;
        fragments_count_ = 0;
    }
}

void MultiGpuPlotPipeline::run()
{
    run_through(Phase::Fragment);
}

void MultiGpuPlotPipeline::run_through(Phase phase)
{
    // Phase-wall timing — POS2GPU_PHASE_TIMING=1 mirrors GpuPipeline's
    // single-GPU breakdown so users can compare sharded vs single
    // costs side by side. Each begin/end waits on every shard queue
    // so the sample brackets actual GPU work; when disabled, the
    // lambdas early-out and add ~zero cost. Phase names match the
    // single-GPU labels where they map cleanly (Xs / T1 / T2 / T3 /
    // fragment) so a user grepping `[phase-timing]` can correlate.
    //
    // POS2GPU_PHASE_TIMING_VERBOSE=1 also enables sub-phase tracing
    // via time_sub() inside run_xs / run_t1 / run_t2 / run_t3 /
    // run_fragment. Each parent phase emits a second
    // `[phase-timing][shard][verbose]` line afterwards listing its
    // sub-steps (replicate D2H/H2D, match, distributed sort, etc.).
    // Adds wait_all() calls inside the phase, which serialises the
    // parallel-submission pattern; only enable when measuring.
    bool const phase_timing = [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING");
        return v && v[0] == '1';
    }();
    bool const subphase_verbose = phase_timing && [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING_VERBOSE");
        return v && v[0] == '1';
    }();
    using phase_clock = std::chrono::steady_clock;
    std::vector<std::pair<char const*, double>> phase_records;
    std::vector<std::pair<char const*, double>> sub_records;
    if (subphase_verbose) subphase_records_ = &sub_records;
    auto wait_all = [&] {
        for (auto& shard : shards_) shard.queue->wait();
    };
    auto run_timed = [&](char const* label, auto&& fn) {
        if (!phase_timing) { fn(); return; }
        wait_all();
        auto const t0 = phase_clock::now();
        fn();
        wait_all();
        auto const t1 = phase_clock::now();
        phase_records.emplace_back(
            label,
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    };
    auto report_subphase = [&](char const* phase_name) {
        if (!subphase_verbose || sub_records.empty()) return;
        double total = 0.0;
        for (auto const& [_n, ms] : sub_records) total += ms;
        std::fprintf(stderr, "[phase-timing][shard][verbose] %s:", phase_name);
        for (auto const& [name, ms] : sub_records) {
            std::fprintf(stderr, " %s=%.1fms(%.0f%%)",
                name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        }
        std::fprintf(stderr, " sum=%.1fms\n", total);
        sub_records.clear();
    };
    auto report = [&] {
        if (!phase_timing || phase_records.empty()) return;
        double total = 0.0;
        for (auto const& [_n, ms] : phase_records) total += ms;
        std::fprintf(stderr, "[phase-timing][shard]");
        for (auto const& [name, ms] : phase_records) {
            std::fprintf(stderr, " %s=%.1fms(%.0f%%)",
                name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        }
        std::fprintf(stderr, " total=%.1fms\n", total);
    };

    run_timed("Xs",       [&] { run_xs_phase_impl(); });
    report_subphase("Xs");
    if (phase == Phase::Xs) { report(); subphase_records_ = nullptr; return; }
    run_timed("T1",       [&] { run_t1_phase(); });
    report_subphase("T1");
    if (phase == Phase::T1) { report(); subphase_records_ = nullptr; return; }
    run_timed("T2",       [&] { run_t2_phase(); });
    report_subphase("T2");
    if (phase == Phase::T2) { report(); subphase_records_ = nullptr; return; }
    run_timed("T3",       [&] { run_t3_phase(); });
    report_subphase("T3");
    if (phase == Phase::T3) { report(); subphase_records_ = nullptr; return; }
    run_timed("Fragment", [&] { run_fragment_phase(); });
    report_subphase("Fragment");
    report();
    subphase_records_ = nullptr;
}

void MultiGpuPlotPipeline::run_xs_phase_impl()
{
    std::size_t const N = shards_.size();
    int const k = entry_.k;
    std::uint64_t const total_xs    = std::uint64_t{1} << k;
    std::uint32_t const xs_xor_const = entry_.testnet ? 0xA3B1C4D7u : 0u;
    AesHashKeys const xs_keys = make_keys(entry_.plot_id.data());

    // Step 1 — per-shard Xs gen via launch_xs_gen_range. Shard k owns
    // the position range [k*total_xs/N, (k+1)*total_xs/N). Outputs
    // land in per-shard u32 keys/vals scratch on the shard's device.
    auto const t_xs_gen = sub_begin();
    std::vector<std::uint32_t*> d_keys_in (N, nullptr);
    std::vector<std::uint32_t*> d_vals_in (N, nullptr);
    std::vector<std::uint32_t*> d_keys_out(N, nullptr);
    std::vector<std::uint32_t*> d_vals_out(N, nullptr);
    std::vector<std::uint64_t>  shard_in_count(N, 0);

    // Per-shard receive capacity. g_x's value-range bucket function
    // produces a near-uniform distribution across the N receivers, so
    // each shard gets ~total_xs/N items in steady state. We size to
    // the per-shard share + 25 % slack, mirroring the T1/T2/T3 sort
    // capacity formulas. Distributed-sort's `cd > out_capacity` check
    // throws cleanly on the (very unlikely) skewed input that exceeds
    // the bound. Sizing to the full total_xs would multiply per-shard
    // VRAM by N for buffers that are essentially write-by-1/N — at
    // k=28, total_xs is 1 GiB / stream and there are 4 streams, so
    // the old worst-case sizing wasted ~1.5 GiB per shard.
    std::uint64_t const out_cap_share = (total_xs + N - 1) / N;
    std::uint64_t const out_cap       = out_cap_share + out_cap_share / 4 + 1024;

    auto const xs_partition = compute_position_partition(shards_, total_xs);

    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const pos_begin = xs_partition[s];
        std::uint64_t const pos_end   = xs_partition[s + 1];
        std::uint64_t const c         = pos_end - pos_begin;
        shard_in_count[s] = c;

        sycl::queue& q = *shards_[s].queue;
        d_keys_in [s] = sycl::malloc_device<std::uint32_t>(out_cap, q);
        d_vals_in [s] = sycl::malloc_device<std::uint32_t>(out_cap, q);
        d_keys_out[s] = sycl::malloc_device<std::uint32_t>(out_cap, q);
        d_vals_out[s] = sycl::malloc_device<std::uint32_t>(out_cap, q);

        if (c > 0) {
            launch_xs_gen_range(
                xs_keys, d_keys_in[s], d_vals_in[s],
                pos_begin, pos_end, k, xs_xor_const, q);
        }
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();
    sub_end("xs.gen", t_xs_gen);

    // Step 2 — distributed sort. Each shard becomes a
    // DistributedSortPairsShard. After the call, shard k's keys_out /
    // vals_out hold its bucket range of sorted (key, val) pairs;
    // shards[k].out_count is the actual count.
    auto const t_xs_sort = sub_begin();
    std::vector<DistributedSortPairsShard> sort_shards(N);
    for (std::size_t s = 0; s < N; ++s) {
        sort_shards[s].queue        = shards_[s].queue;
        sort_shards[s].keys_in      = d_keys_in[s];
        sort_shards[s].vals_in      = d_vals_in[s];
        sort_shards[s].count        = shard_in_count[s];
        sort_shards[s].keys_out     = d_keys_out[s];
        sort_shards[s].vals_out     = d_vals_out[s];
        sort_shards[s].out_capacity = out_cap;
        sort_shards[s].out_count    = 0;
        sort_shards[s].pool         = shards_[s].pool;
    }
    std::size_t scratch_bytes = 0;
    launch_sort_pairs_u32_u32_distributed(
        nullptr, scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k, transport());
    void* d_scratch = scratch_bytes
        ? sycl::malloc_device(scratch_bytes, *shards_[0].queue)
        : nullptr;
    launch_sort_pairs_u32_u32_distributed(
        d_scratch ? d_scratch : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k, transport());
    if (d_scratch) sycl::free(d_scratch, *shards_[0].queue);
    sub_end("xs.sort", t_xs_sort);

    // Step 3 — per-shard launch_xs_pack into a packed XsCandidateGpu
    // output sized for the shard's bucket range. The packed output is
    // what T1 match consumes in Phase 2.3.
    auto const t_xs_pack = sub_begin();
    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const c = sort_shards[s].out_count;
        xs_phase_count_[s] = c;
        sycl::queue& q = *shards_[s].queue;
        if (c == 0) {
            xs_phase_d_xs_[s] = nullptr;
        } else {
            xs_phase_d_xs_[s] = sycl::malloc_device<XsCandidateGpu>(c, q);
            launch_xs_pack(
                sort_shards[s].keys_out, sort_shards[s].vals_out,
                xs_phase_d_xs_[s], c, q);
        }
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();
    sub_end("xs.pack", t_xs_pack);

    // Free intermediate u32 keys/vals — only the packed XsCandidateGpu
    // output survives into Phase 2.3.
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        sycl::free(d_keys_in [s], q);
        sycl::free(d_vals_in [s], q);
        sycl::free(d_keys_out[s], q);
        sycl::free(d_vals_out[s], q);
    }

    // Sanity: total post-sort count must equal total_xs (no items
    // lost in the bucket scatter).
    std::uint64_t total_out = 0;
    for (std::size_t s = 0; s < N; ++s) total_out += xs_phase_count_[s];
    if (total_out != total_xs) {
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_xs_phase: post-sort count mismatch "
            "(expected " + std::to_string(total_xs) + ", got "
            + std::to_string(total_out) + ")");
    }
}

void MultiGpuPlotPipeline::run_t1_phase()
{
    // Phase 2.3a — sharded T1 match.
    //
    // Algorithm:
    //   1. Replicate sorted Xs onto every shard via host-pinned bounce.
    //      The Xs phase's bucket-partition by match_info means each shard
    //      already holds a contiguous bucket range; concatenating in
    //      shard-id order yields the single-GPU sorted-Xs layout
    //      byte-for-byte (Phase 2.2 already validated this).
    //   2. Each shard runs launch_t1_match_prepare (full-input bucket
    //      offsets) + launch_t1_match_range over its assigned bucket
    //      subset. The atomic-cursor append plumbing inside the kernel
    //      means each shard fills a local (mi, meta) stream containing
    //      exactly the matches whose bucket falls in its range.
    //   3. Distributed sort the per-shard (mi, meta) streams by mi using
    //      launch_sort_pairs_u32_u64_distributed. After the sort, shard k
    //      holds matches whose mi falls in [k * 2^k / N, (k+1) * 2^k / N),
    //      sorted within that range — the layout T2 will consume.
    //
    // Replication is the right tradeoff at this slice: T1 needs both
    // section_l data and section_r = matching_section(section_l) data,
    // and matching_section is a non-trivial permutation (rotate-left +1
    // rotate-right) that doesn't admit a section partition where every
    // match stays intra-shard. Replicating sorted Xs (k=28: 2 GB per
    // shard) is cheap on 12+ GiB cards; the multi-GPU memory savings
    // come in T2/T3, not Xs.

    std::size_t const N = shards_.size();
    int const k = entry_.k;
    std::uint64_t const total_xs = std::uint64_t{1} << k;
    auto const t1p = make_t1_params(k, entry_.strength);

    // Mirror derive_t1: num_buckets = num_sections * num_match_keys.
    std::uint32_t const num_buckets =
        (std::uint32_t{1} << t1p.num_section_bits) *
        (std::uint32_t{1} << t1p.num_match_key_bits);

    auto const t1_partition = compute_bucket_partition(shards_, num_buckets);

    // ---------- Step 1 — replicate sorted Xs across shards. ----------
    auto const t_t1_replicate = sub_begin();
    // Per-shard offsets in the concatenated layout. Computed once so
    // both the D2H pull and the receiver-side scatter can index without
    // re-walking the count array.
    std::vector<std::uint64_t> shard_off(N, 0);
    {
        std::uint64_t off = 0;
        for (std::size_t s = 0; s < N; ++s) {
            shard_off[s] = off;
            off += xs_phase_count_[s];
        }
        if (off != total_xs) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t1_phase: Xs phase outputs sum to "
                + std::to_string(off) + " entries but total_xs = "
                + std::to_string(total_xs)
                + ". run_xs_phase() must complete before run_t1_phase().");
        }
    }

    // Allocate destination buffers (one full-Xs replica per shard) up
    // front so we can stream into them in parallel below.
    std::vector<SyclDevicePtr<XsCandidateGpu>> d_full_xs(N);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_full_xs[s] = pool_or_alloc<XsCandidateGpu>(
            shards_[s].pool, "full_xs", total_xs, q);
    }

    sycl::queue& alloc_q = *shards_[0].queue;

    if (opts_.prefer_peer_copy) {
        // Peer-copy fast path: each receiver issues N independent D2D
        // memcpys directly from each source's per-shard d_xs buffer into
        // its own d_full_xs at the correct offset. AdaptiveCpp's CUDA
        // backend resolves cross-device pointers via cudaMemcpyPeerAsync;
        // single-context (e.g. all-shards-on-one-device parity tests)
        // collapses to a same-device copy. Skips the host bounce entirely.
        std::vector<sycl::event> evts;
        evts.reserve(N * N);
        for (std::size_t r = 0; r < N; ++r) {
            sycl::queue& q = *shards_[r].queue;
            for (std::size_t s = 0; s < N; ++s) {
                std::uint64_t const c = xs_phase_count_[s];
                if (c == 0) continue;
                evts.push_back(q.memcpy(
                    d_full_xs[r].get() + shard_off[s], xs_phase_d_xs_[s],
                    sizeof(XsCandidateGpu) * c));
            }
        }
        for (auto& e : evts) e.wait();
    } else {
        // Host-bounce path: parallel D2H into a single pinned-host
        // bounce buffer, parallel H2D fan-out. The pinned buffer is
        // pooled when shards_[0].pool is attached so consecutive plots
        // (and the next match phase) reuse the page-locked range.
        XsCandidateGpu* h_full_ptr = nullptr;
        SyclHostPtr<XsCandidateGpu> h_full_owned;
        if (shards_[0].pool) {
            h_full_ptr = shards_[0].pool->ensure_host<XsCandidateGpu>(
                "h_bounce_xs", total_xs);
        } else {
            h_full_owned = sycl_alloc_host_owned<XsCandidateGpu>(
                total_xs, alloc_q);
            h_full_ptr = h_full_owned.get();
        }

        std::vector<sycl::event> d2h(N);
        for (std::size_t s = 0; s < N; ++s) {
            std::uint64_t const c = xs_phase_count_[s];
            if (c == 0) continue;
            d2h[s] = shards_[s].queue->memcpy(
                h_full_ptr + shard_off[s], xs_phase_d_xs_[s],
                sizeof(XsCandidateGpu) * c);
        }
        for (std::size_t s = 0; s < N; ++s) {
            if (xs_phase_count_[s] > 0) d2h[s].wait();
        }

        std::vector<sycl::event> h2d(N);
        for (std::size_t s = 0; s < N; ++s) {
            sycl::queue& q = *shards_[s].queue;
            h2d[s] = q.memcpy(d_full_xs[s].get(), h_full_ptr,
                              sizeof(XsCandidateGpu) * total_xs);
        }
        for (auto& e : h2d) e.wait();
        // h_full_owned RAII-frees if pool was unavailable.
    }

    // The bucket-partitioned Xs phase outputs are no longer needed —
    // d_full_xs holds the same data on every shard. Free them now to
    // recover memory before T1 staging allocations.
    for (std::size_t s = 0; s < N; ++s) {
        if (xs_phase_d_xs_[s]) {
            sycl::free(xs_phase_d_xs_[s], *shards_[s].queue);
            xs_phase_d_xs_[s] = nullptr;
        }
    }

    sub_end("t1.replicate", t_t1_replicate);

    // ---------- Step 2 — per-shard T1 match over assigned buckets. ----
    auto const t_t1_match = sub_begin();
    // Capacity: max_pairs_per_section * num_sections — the standard
    // pos2-chip pool sizing for T1 output, which adds an extra-margin-
    // bits term over the naive 2^k count to absorb expected skew. Same
    // formula GpuBufferPool uses for the single-GPU T1 cap, giving each
    // shard the same per-bucket safety as the production single-GPU
    // path. Each shard sees the FULL input + emits only its bucket
    // Each shard processes its bucket-range subset of inputs and
    // produces ~match_phase_capacity/N matches (uniform distribution).
    // Sizing the unsorted buffers to the full t1_cap allocates N×
    // more than the shard needs and OOMs at k>=28 on 20 GB cards.
    // 25% slack covers any partition imbalance; the
    // shard_count[s] > t1_cap check below throws cleanly if a
    // pathological input ever exceeds it.
    std::uint64_t const t1_cap_full = match_phase_capacity(k, t1p.num_section_bits);
    std::uint64_t const t1_cap_share = (t1_cap_full + N - 1) / N;
    std::uint64_t const t1_cap = t1_cap_share + t1_cap_share / 4 + 1024;

    std::vector<SyclDevicePtr<std::uint64_t>> d_t1_meta_unsorted(N);
    std::vector<SyclDevicePtr<std::uint32_t>> d_t1_mi_unsorted  (N);
    std::vector<SyclDevicePtr<std::uint64_t>> d_t1_count        (N);
    std::vector<SyclDeviceVoid>               d_t1_temp         (N);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t1_meta_unsorted[s] = sycl_alloc_device_owned<std::uint64_t>(t1_cap, q);
        d_t1_mi_unsorted  [s] = sycl_alloc_device_owned<std::uint32_t>(t1_cap, q);
        d_t1_count        [s] = sycl_alloc_device_owned<std::uint64_t>(1, q);

        std::size_t tb = 0;
        launch_t1_match_prepare(entry_.plot_id.data(), t1p,
            d_full_xs[s].get(), total_xs,
            d_t1_count[s].get(), nullptr, &tb, q);
        d_t1_temp[s] = SyclDeviceVoid(&q, sycl::malloc_device(tb, q));
        launch_t1_match_prepare(entry_.plot_id.data(), t1p,
            d_full_xs[s].get(), total_xs,
            d_t1_count[s].get(), d_t1_temp[s].get(), &tb, q);

        std::uint32_t const bucket_begin = t1_partition[s];
        std::uint32_t const bucket_end   = t1_partition[s + 1];

        launch_t1_match_range(entry_.plot_id.data(), t1p,
            d_full_xs[s].get(), total_xs,
            d_t1_meta_unsorted[s].get(), d_t1_mi_unsorted[s].get(),
            d_t1_count[s].get(),
            t1_cap, d_t1_temp[s].get(),
            bucket_begin, bucket_end, q);
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();

    // Pull the N shard counts in one batch — submit all then drain.
    std::vector<std::uint64_t> shard_count(N, 0);
    {
        std::vector<sycl::event> cnt_evts(N);
        for (std::size_t s = 0; s < N; ++s) {
            cnt_evts[s] = shards_[s].queue->memcpy(
                &shard_count[s], d_t1_count[s].get(), sizeof(std::uint64_t));
        }
        for (auto& e : cnt_evts) e.wait();
    }
    for (std::size_t s = 0; s < N; ++s) {
        if (shard_count[s] > t1_cap) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t1_phase: shard "
                + std::to_string(s) + " T1 produced "
                + std::to_string(shard_count[s])
                + " entries, exceeds capacity " + std::to_string(t1_cap)
                + ". Bucket-skew worse than expected; raise t1_cap.");
        }
    }

    // Early-release the heavy intermediates before allocating the sort
    // outputs — d_full_xs is the largest (full Xs replica per shard).
    for (std::size_t s = 0; s < N; ++s) {
        d_t1_temp [s].reset();
        d_t1_count[s].reset();
        d_full_xs [s].reset();
    }

    sub_end("t1.match", t_t1_match);

    // ---------- Step 3 — distributed sort by mi. -------------------
    auto const t_t1_sort = sub_begin();
    // Each shard receives ~t1_total/N items after the value-range
    // redistribution; sizing each output buffer to t1_total
    // over-allocates by N× and OOMs at k>=28 on 20 GB cards. 25% slack
    // on the expected share is plenty for uniform mi (the
    // recv_count > out_capacity check in the sort still throws cleanly
    // on pathological inputs). See run_t3_phase below for the same
    // pattern.
    std::uint64_t t1_total = 0;
    for (auto c : shard_count) t1_total += c;

    std::uint64_t const t1_per_shard_share = (t1_total + N - 1) / N;
    std::uint64_t const sort_cap = t1_per_shard_share + t1_per_shard_share / 4 + 1024;
    std::vector<SyclDevicePtr<std::uint32_t>> d_t1_mi_sorted  (N);
    std::vector<SyclDevicePtr<std::uint64_t>> d_t1_meta_sorted(N);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t1_mi_sorted  [s] = sycl_alloc_device_owned<std::uint32_t>(sort_cap, q);
        d_t1_meta_sorted[s] = sycl_alloc_device_owned<std::uint64_t>(sort_cap, q);
    }

    std::vector<DistributedSortPairsU32U64Shard> sort_shards(N);
    for (std::size_t s = 0; s < N; ++s) {
        sort_shards[s].queue        = shards_[s].queue;
        sort_shards[s].keys_in      = d_t1_mi_unsorted[s].get();
        sort_shards[s].vals_in      = d_t1_meta_unsorted[s].get();
        sort_shards[s].count        = shard_count[s];
        sort_shards[s].keys_out     = d_t1_mi_sorted[s].get();
        sort_shards[s].vals_out     = d_t1_meta_sorted[s].get();
        sort_shards[s].out_capacity = sort_cap;
        sort_shards[s].out_count    = 0;
        sort_shards[s].pool         = shards_[s].pool;
    }

    std::size_t scratch_bytes = 0;
    launch_sort_pairs_u32_u64_distributed(
        nullptr, scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k, transport());
    SyclDeviceVoid d_scratch;
    if (scratch_bytes) {
        d_scratch = SyclDeviceVoid(shards_[0].queue,
            sycl::malloc_device(scratch_bytes, *shards_[0].queue));
    }
    launch_sort_pairs_u32_u64_distributed(
        d_scratch.get() ? d_scratch.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k, transport());
    d_scratch.reset();

    // Early-free the unsorted scratch — the distributed sort already
    // wrote the result into d_t1_*_sorted.
    for (std::size_t s = 0; s < N; ++s) {
        d_t1_mi_unsorted  [s].reset();
        d_t1_meta_unsorted[s].reset();
    }

    // Hand off ownership of the sorted buffers to the phase-output
    // member vectors (managed by free_phase_outputs from now on).
    for (std::size_t s = 0; s < N; ++s) {
        t1_phase_d_mi_  [s] = d_t1_mi_sorted  [s].release();
        t1_phase_d_meta_[s] = d_t1_meta_sorted[s].release();
        t1_phase_count_ [s] = sort_shards[s].out_count;
    }
    sub_end("t1.sort", t_t1_sort);

    std::uint64_t out_total = 0;
    for (auto c : t1_phase_count_) out_total += c;
    if (out_total != t1_total) {
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_t1_phase: post-sort count mismatch "
            "(expected " + std::to_string(t1_total) + ", got "
            + std::to_string(out_total) + ")");
    }

    // The replicated Xs buffer is dead for the remainder of the plot
    // (T2/T3 work from the per-shard sorted T1 output). Drop the pool
    // slot on every shard so its ~3 GB at k=28 doesn't sit alongside
    // T2/T3's working set. The next plot will re-allocate via ensure().
    for (std::size_t s = 0; s < N; ++s) {
        if (shards_[s].pool) {
            shards_[s].pool->clear_slot("full_xs");
            shards_[s].pool->clear_host_slot("h_bounce_xs");
        }
    }
}

void MultiGpuPlotPipeline::run_t2_phase()
{
    // Phase 2.3b — sharded T2 match.
    //
    // Same shape as Phase 2.3a, with the input being the T1-sorted
    // (mi, meta) streams produced by run_t1_phase and the output adding
    // an x_bits stream per match:
    //   1. Replicate the full T1-sorted (mi, meta) streams onto every
    //      shard via host-pinned bounce. T2's matching_section is the
    //      same rotate-+1-rotate permutation as T1, so cross-shard
    //      reads are unavoidable; replication is the right tradeoff.
    //   2. Per-shard launch_t2_match_prepare + launch_t2_match_range
    //      over the assigned bucket subset.
    //   3. Distributed sort the per-shard (mi, meta, xbits) streams by
    //      mi via launch_sort_pairs_u32_u64u32_distributed.
    //
    // num_match_key_bits = strength for T2 (T1 hardcoded to 2). Default
    // strength=2 keeps num_buckets at 16 — same as T1.

    std::size_t const N = shards_.size();
    int const k = entry_.k;
    auto const t2p = make_t2_params(k, entry_.strength);

    std::uint32_t const num_buckets =
        (std::uint32_t{1} << t2p.num_section_bits) *
        (std::uint32_t{1} << t2p.num_match_key_bits);

    auto const t2_partition = compute_bucket_partition(shards_, num_buckets);

    // ---------- Step 1 — replicate T1 sorted streams. ----------
    auto const t_t2_replicate = sub_begin();
    std::uint64_t t1_total = 0;
    for (auto c : t1_phase_count_) t1_total += c;

    std::vector<std::uint64_t> shard_off(N, 0);
    {
        std::uint64_t off = 0;
        for (std::size_t s = 0; s < N; ++s) {
            shard_off[s] = off;
            off += t1_phase_count_[s];
        }
        if (off != t1_total) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t2_phase: T1 outputs sum to "
                + std::to_string(off) + " entries but t1_total = "
                + std::to_string(t1_total));
        }
    }

    std::vector<SyclDevicePtr<std::uint32_t>> d_full_mi  (N);
    std::vector<SyclDevicePtr<std::uint64_t>> d_full_meta(N);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_full_mi  [s] = pool_or_alloc<std::uint32_t>(
            shards_[s].pool, "t2_full_mi",   t1_total, q);
        d_full_meta[s] = pool_or_alloc<std::uint64_t>(
            shards_[s].pool, "t2_full_meta", t1_total, q);
    }

    sycl::queue& alloc_q = *shards_[0].queue;

    if (opts_.prefer_peer_copy) {
        // Peer-copy: per receiver, fan-in N D2D memcpys per stream.
        std::vector<sycl::event> evts;
        evts.reserve(N * N * 2);
        for (std::size_t r = 0; r < N; ++r) {
            sycl::queue& q = *shards_[r].queue;
            for (std::size_t s = 0; s < N; ++s) {
                std::uint64_t const c = t1_phase_count_[s];
                if (c == 0) continue;
                evts.push_back(q.memcpy(
                    d_full_mi  [r].get() + shard_off[s], t1_phase_d_mi_[s],
                    c * sizeof(std::uint32_t)));
                evts.push_back(q.memcpy(
                    d_full_meta[r].get() + shard_off[s], t1_phase_d_meta_[s],
                    c * sizeof(std::uint64_t)));
            }
        }
        for (auto& e : evts) e.wait();
    } else {
        std::uint32_t* h_mi   = nullptr;
        std::uint64_t* h_meta = nullptr;
        SyclHostPtr<std::uint32_t> h_mi_owned;
        SyclHostPtr<std::uint64_t> h_meta_owned;
        if (shards_[0].pool) {
            h_mi   = shards_[0].pool->ensure_host<std::uint32_t>(
                "h_bounce_mi",   t1_total);
            h_meta = shards_[0].pool->ensure_host<std::uint64_t>(
                "h_bounce_meta", t1_total);
        } else {
            h_mi_owned   = sycl_alloc_host_owned<std::uint32_t>(t1_total, alloc_q);
            h_meta_owned = sycl_alloc_host_owned<std::uint64_t>(t1_total, alloc_q);
            h_mi   = h_mi_owned.get();
            h_meta = h_meta_owned.get();
        }

        // D2H — submit per-shard pulls concurrently, then wait.
        std::vector<sycl::event> d2h_mi(N);
        std::vector<sycl::event> d2h_meta(N);
        for (std::size_t s = 0; s < N; ++s) {
            std::uint64_t const c = t1_phase_count_[s];
            if (c == 0) continue;
            d2h_mi  [s] = shards_[s].queue->memcpy(
                h_mi   + shard_off[s], t1_phase_d_mi_[s],
                c * sizeof(std::uint32_t));
            d2h_meta[s] = shards_[s].queue->memcpy(
                h_meta + shard_off[s], t1_phase_d_meta_[s],
                c * sizeof(std::uint64_t));
        }
        for (std::size_t s = 0; s < N; ++s) {
            if (t1_phase_count_[s] == 0) continue;
            d2h_mi  [s].wait();
            d2h_meta[s].wait();
        }

        // H2D fan-out — submit per-shard, then wait.
        std::vector<sycl::event> h2d_mi(N);
        std::vector<sycl::event> h2d_meta(N);
        for (std::size_t s = 0; s < N; ++s) {
            sycl::queue& q = *shards_[s].queue;
            h2d_mi  [s] = q.memcpy(d_full_mi  [s].get(), h_mi,
                                   t1_total * sizeof(std::uint32_t));
            h2d_meta[s] = q.memcpy(d_full_meta[s].get(), h_meta,
                                   t1_total * sizeof(std::uint64_t));
        }
        for (auto& e : h2d_mi)   e.wait();
        for (auto& e : h2d_meta) e.wait();
    }

    // The bucket-partitioned T1 outputs are no longer needed.
    for (std::size_t s = 0; s < N; ++s) {
        if (t1_phase_d_mi_[s]) {
            sycl::free(t1_phase_d_mi_[s], *shards_[s].queue);
            t1_phase_d_mi_[s] = nullptr;
        }
        if (t1_phase_d_meta_[s]) {
            sycl::free(t1_phase_d_meta_[s], *shards_[s].queue);
            t1_phase_d_meta_[s] = nullptr;
        }
    }

    sub_end("t2.replicate", t_t2_replicate);

    // ---------- Step 2 — per-shard T2 match. ----------
    auto const t_t2_match = sub_begin();
    // Per-shard share of the full T2 capacity; see run_t1_phase for
    // the rationale. T2 has the largest per-item footprint of the
    // three phases (mi+meta+xbits = 16 B/item), so the saving from
    // shrinking these unsorted buffers is the biggest of any single
    // change.
    std::uint64_t const t2_cap_full = match_phase_capacity(k, t2p.num_section_bits);
    std::uint64_t const t2_cap_share = (t2_cap_full + N - 1) / N;
    std::uint64_t const t2_cap = t2_cap_share + t2_cap_share / 4 + 1024;

    std::vector<SyclDevicePtr<std::uint64_t>> d_t2_meta_unsorted (N);
    std::vector<SyclDevicePtr<std::uint32_t>> d_t2_mi_unsorted   (N);
    std::vector<SyclDevicePtr<std::uint32_t>> d_t2_xbits_unsorted(N);
    std::vector<SyclDevicePtr<std::uint64_t>> d_t2_count         (N);
    std::vector<SyclDeviceVoid>               d_t2_temp          (N);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t2_meta_unsorted [s] = sycl_alloc_device_owned<std::uint64_t>(t2_cap, q);
        d_t2_mi_unsorted   [s] = sycl_alloc_device_owned<std::uint32_t>(t2_cap, q);
        d_t2_xbits_unsorted[s] = sycl_alloc_device_owned<std::uint32_t>(t2_cap, q);
        d_t2_count         [s] = sycl_alloc_device_owned<std::uint64_t>(1, q);

        std::size_t tb = 0;
        launch_t2_match_prepare(entry_.plot_id.data(), t2p,
            d_full_mi[s].get(), t1_total,
            d_t2_count[s].get(), nullptr, &tb, q);
        d_t2_temp[s] = SyclDeviceVoid(&q, sycl::malloc_device(tb, q));
        launch_t2_match_prepare(entry_.plot_id.data(), t2p,
            d_full_mi[s].get(), t1_total,
            d_t2_count[s].get(), d_t2_temp[s].get(), &tb, q);

        std::uint32_t const bucket_begin = t2_partition[s];
        std::uint32_t const bucket_end   = t2_partition[s + 1];

        launch_t2_match_range(entry_.plot_id.data(), t2p,
            d_full_meta[s].get(), d_full_mi[s].get(), t1_total,
            d_t2_meta_unsorted[s].get(), d_t2_mi_unsorted[s].get(),
            d_t2_xbits_unsorted[s].get(), d_t2_count[s].get(),
            t2_cap, d_t2_temp[s].get(),
            bucket_begin, bucket_end, q);
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();

    std::vector<std::uint64_t> shard_count(N, 0);
    {
        std::vector<sycl::event> cnt_evts(N);
        for (std::size_t s = 0; s < N; ++s) {
            cnt_evts[s] = shards_[s].queue->memcpy(
                &shard_count[s], d_t2_count[s].get(), sizeof(std::uint64_t));
        }
        for (auto& e : cnt_evts) e.wait();
    }
    for (std::size_t s = 0; s < N; ++s) {
        if (shard_count[s] > t2_cap) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t2_phase: shard "
                + std::to_string(s) + " T2 produced "
                + std::to_string(shard_count[s])
                + " entries, exceeds capacity " + std::to_string(t2_cap));
        }
    }

    // Early-release the heavy intermediates (full T1 replicas + match
    // scratch + count) before allocating sort outputs.
    for (std::size_t s = 0; s < N; ++s) {
        d_t2_temp  [s].reset();
        d_t2_count [s].reset();
        d_full_mi  [s].reset();
        d_full_meta[s].reset();
    }

    sub_end("t2.match", t_t2_match);

    // ---------- Step 3 — distributed sort by mi. ----------
    auto const t_t2_sort = sub_begin();
    // Each shard receives ~t2_total/N items; t2_total over-allocates by
    // N×. T2 has the largest per-item footprint of the three sorts
    // (u32 + u64 + u32 = 16 bytes/item), so this is where the OOM
    // bites hardest at high k. See run_t1_phase / run_t3_phase for the
    // same pattern.
    std::uint64_t t2_total = 0;
    for (auto c : shard_count) t2_total += c;

    std::uint64_t const t2_per_shard_share = (t2_total + N - 1) / N;
    std::uint64_t const sort_cap = t2_per_shard_share + t2_per_shard_share / 4 + 1024;
    std::vector<SyclDevicePtr<std::uint32_t>> d_t2_mi_sorted   (N);
    std::vector<SyclDevicePtr<std::uint64_t>> d_t2_meta_sorted (N);
    std::vector<SyclDevicePtr<std::uint32_t>> d_t2_xbits_sorted(N);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t2_mi_sorted   [s] = sycl_alloc_device_owned<std::uint32_t>(sort_cap, q);
        d_t2_meta_sorted [s] = sycl_alloc_device_owned<std::uint64_t>(sort_cap, q);
        d_t2_xbits_sorted[s] = sycl_alloc_device_owned<std::uint32_t>(sort_cap, q);
    }

    std::vector<DistributedSortPairsU32U64U32Shard> sort_shards(N);
    for (std::size_t s = 0; s < N; ++s) {
        sort_shards[s].queue        = shards_[s].queue;
        sort_shards[s].keys_in      = d_t2_mi_unsorted[s].get();
        sort_shards[s].vals_a_in    = d_t2_meta_unsorted[s].get();
        sort_shards[s].vals_b_in    = d_t2_xbits_unsorted[s].get();
        sort_shards[s].count        = shard_count[s];
        sort_shards[s].keys_out     = d_t2_mi_sorted[s].get();
        sort_shards[s].vals_a_out   = d_t2_meta_sorted[s].get();
        sort_shards[s].vals_b_out   = d_t2_xbits_sorted[s].get();
        sort_shards[s].out_capacity = sort_cap;
        sort_shards[s].out_count    = 0;
        sort_shards[s].pool         = shards_[s].pool;
    }

    std::size_t scratch_bytes = 0;
    launch_sort_pairs_u32_u64u32_distributed(
        nullptr, scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k, transport());
    SyclDeviceVoid d_scratch;
    if (scratch_bytes) {
        d_scratch = SyclDeviceVoid(shards_[0].queue,
            sycl::malloc_device(scratch_bytes, *shards_[0].queue));
    }
    launch_sort_pairs_u32_u64u32_distributed(
        d_scratch.get() ? d_scratch.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k, transport());
    d_scratch.reset();

    for (std::size_t s = 0; s < N; ++s) {
        d_t2_mi_unsorted   [s].reset();
        d_t2_meta_unsorted [s].reset();
        d_t2_xbits_unsorted[s].reset();
    }

    for (std::size_t s = 0; s < N; ++s) {
        t2_phase_d_mi_   [s] = d_t2_mi_sorted   [s].release();
        t2_phase_d_meta_ [s] = d_t2_meta_sorted [s].release();
        t2_phase_d_xbits_[s] = d_t2_xbits_sorted[s].release();
        t2_phase_count_  [s] = sort_shards[s].out_count;
    }
    sub_end("t2.sort", t_t2_sort);

    // Drop the T2 input replication slots — T3 reads from
    // t2_phase_d_mi_/meta_/xbits_ (the per-shard sorted T2 output),
    // not from t2_full_*. Saves ~3 GB at k=28 during T3.
    for (std::size_t s = 0; s < N; ++s) {
        if (shards_[s].pool) {
            shards_[s].pool->clear_slot("t2_full_mi");
            shards_[s].pool->clear_slot("t2_full_meta");
        }
    }

    std::uint64_t out_total = 0;
    for (auto c : t2_phase_count_) out_total += c;
    if (out_total != t2_total) {
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_t2_phase: post-sort count mismatch "
            "(expected " + std::to_string(t2_total) + ", got "
            + std::to_string(out_total) + ")");
    }
}

void MultiGpuPlotPipeline::run_t3_phase()
{
    // Phase 2.3c — sharded T3 match.
    //
    // Same shape as 2.3a/2.3b on the T2 sorted output:
    //   1. Replicate the per-shard T2 (mi, meta, xbits) streams onto
    //      every shard via host-pinned bounce. T3 reuses T2's offset
    //      computation (same input layout), and matching_section is
    //      again the rotate-+1-rotate permutation, so cross-shard
    //      reads remain unavoidable.
    //   2. Per-shard launch_t3_match_prepare + launch_t3_match_range
    //      over the assigned bucket subset. Output is T3PairingGpu
    //      (a u64 proof_fragment).
    //   3. Distributed sort the per-shard fragment streams over the
    //      low 2*k bits via launch_sort_keys_u64_distributed (already
    //      shipped in Phase 2.1).

    std::size_t const N = shards_.size();
    int const k = entry_.k;
    auto const t3p = make_t3_params(k, entry_.strength);

    std::uint32_t const num_buckets =
        (std::uint32_t{1} << t3p.num_section_bits) *
        (std::uint32_t{1} << t3p.num_match_key_bits);

    auto const t3_partition = compute_bucket_partition(shards_, num_buckets);

    // ---------- Step 1 — replicate T2 sorted streams. ----------
    auto const t_t3_replicate = sub_begin();
    std::uint64_t t2_total = 0;
    for (auto c : t2_phase_count_) t2_total += c;

    std::vector<std::uint64_t> shard_off(N, 0);
    {
        std::uint64_t off = 0;
        for (std::size_t s = 0; s < N; ++s) {
            shard_off[s] = off;
            off += t2_phase_count_[s];
        }
        if (off != t2_total) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t3_phase: T2 outputs sum to "
                + std::to_string(off) + " entries but t2_total = "
                + std::to_string(t2_total));
        }
    }

    std::vector<SyclDevicePtr<std::uint32_t>> d_full_mi   (N);
    std::vector<SyclDevicePtr<std::uint64_t>> d_full_meta (N);
    std::vector<SyclDevicePtr<std::uint32_t>> d_full_xbits(N);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_full_mi   [s] = pool_or_alloc<std::uint32_t>(
            shards_[s].pool, "t3_full_mi",    t2_total, q);
        d_full_meta [s] = pool_or_alloc<std::uint64_t>(
            shards_[s].pool, "t3_full_meta",  t2_total, q);
        d_full_xbits[s] = pool_or_alloc<std::uint32_t>(
            shards_[s].pool, "t3_full_xbits", t2_total, q);
    }

    sycl::queue& alloc_q = *shards_[0].queue;

    if (opts_.prefer_peer_copy) {
        std::vector<sycl::event> evts;
        evts.reserve(N * N * 3);
        for (std::size_t r = 0; r < N; ++r) {
            sycl::queue& q = *shards_[r].queue;
            for (std::size_t s = 0; s < N; ++s) {
                std::uint64_t const c = t2_phase_count_[s];
                if (c == 0) continue;
                evts.push_back(q.memcpy(
                    d_full_mi   [r].get() + shard_off[s], t2_phase_d_mi_[s],
                    c * sizeof(std::uint32_t)));
                evts.push_back(q.memcpy(
                    d_full_meta [r].get() + shard_off[s], t2_phase_d_meta_[s],
                    c * sizeof(std::uint64_t)));
                evts.push_back(q.memcpy(
                    d_full_xbits[r].get() + shard_off[s], t2_phase_d_xbits_[s],
                    c * sizeof(std::uint32_t)));
            }
        }
        for (auto& e : evts) e.wait();
    } else {
        std::uint32_t* h_mi    = nullptr;
        std::uint64_t* h_meta  = nullptr;
        std::uint32_t* h_xbits = nullptr;
        SyclHostPtr<std::uint32_t> h_mi_owned;
        SyclHostPtr<std::uint64_t> h_meta_owned;
        SyclHostPtr<std::uint32_t> h_xbits_owned;
        if (shards_[0].pool) {
            h_mi    = shards_[0].pool->ensure_host<std::uint32_t>(
                "h_bounce_mi",    t2_total);
            h_meta  = shards_[0].pool->ensure_host<std::uint64_t>(
                "h_bounce_meta",  t2_total);
            h_xbits = shards_[0].pool->ensure_host<std::uint32_t>(
                "h_bounce_xbits", t2_total);
        } else {
            h_mi_owned    = sycl_alloc_host_owned<std::uint32_t>(t2_total, alloc_q);
            h_meta_owned  = sycl_alloc_host_owned<std::uint64_t>(t2_total, alloc_q);
            h_xbits_owned = sycl_alloc_host_owned<std::uint32_t>(t2_total, alloc_q);
            h_mi    = h_mi_owned.get();
            h_meta  = h_meta_owned.get();
            h_xbits = h_xbits_owned.get();
        }

        std::vector<sycl::event> d2h_mi(N);
        std::vector<sycl::event> d2h_meta(N);
        std::vector<sycl::event> d2h_xbits(N);
        for (std::size_t s = 0; s < N; ++s) {
            std::uint64_t const c = t2_phase_count_[s];
            if (c == 0) continue;
            d2h_mi   [s] = shards_[s].queue->memcpy(
                h_mi    + shard_off[s], t2_phase_d_mi_[s],
                c * sizeof(std::uint32_t));
            d2h_meta [s] = shards_[s].queue->memcpy(
                h_meta  + shard_off[s], t2_phase_d_meta_[s],
                c * sizeof(std::uint64_t));
            d2h_xbits[s] = shards_[s].queue->memcpy(
                h_xbits + shard_off[s], t2_phase_d_xbits_[s],
                c * sizeof(std::uint32_t));
        }
        for (std::size_t s = 0; s < N; ++s) {
            if (t2_phase_count_[s] == 0) continue;
            d2h_mi   [s].wait();
            d2h_meta [s].wait();
            d2h_xbits[s].wait();
        }

        std::vector<sycl::event> h2d_mi(N);
        std::vector<sycl::event> h2d_meta(N);
        std::vector<sycl::event> h2d_xbits(N);
        for (std::size_t s = 0; s < N; ++s) {
            sycl::queue& q = *shards_[s].queue;
            h2d_mi   [s] = q.memcpy(d_full_mi   [s].get(), h_mi,
                                    t2_total * sizeof(std::uint32_t));
            h2d_meta [s] = q.memcpy(d_full_meta [s].get(), h_meta,
                                    t2_total * sizeof(std::uint64_t));
            h2d_xbits[s] = q.memcpy(d_full_xbits[s].get(), h_xbits,
                                    t2_total * sizeof(std::uint32_t));
        }
        for (auto& e : h2d_mi)    e.wait();
        for (auto& e : h2d_meta)  e.wait();
        for (auto& e : h2d_xbits) e.wait();
    }

    // The bucket-partitioned T2 outputs are no longer needed.
    for (std::size_t s = 0; s < N; ++s) {
        if (t2_phase_d_mi_[s]) {
            sycl::free(t2_phase_d_mi_[s], *shards_[s].queue);
            t2_phase_d_mi_[s] = nullptr;
        }
        if (t2_phase_d_meta_[s]) {
            sycl::free(t2_phase_d_meta_[s], *shards_[s].queue);
            t2_phase_d_meta_[s] = nullptr;
        }
        if (t2_phase_d_xbits_[s]) {
            sycl::free(t2_phase_d_xbits_[s], *shards_[s].queue);
            t2_phase_d_xbits_[s] = nullptr;
        }
    }

    sub_end("t3.replicate", t_t3_replicate);

    // ---------- Step 2 — per-shard T3 match. ----------
    auto const t_t3_match = sub_begin();
    // Per-shard share of the full T3 capacity; see run_t1_phase.
    std::uint64_t const t3_cap_full = match_phase_capacity(k, t3p.num_section_bits);
    std::uint64_t const t3_cap_share = (t3_cap_full + N - 1) / N;
    std::uint64_t const t3_cap = t3_cap_share + t3_cap_share / 4 + 1024;

    std::vector<SyclDevicePtr<T3PairingGpu>>  d_t3_unsorted(N);
    std::vector<SyclDevicePtr<std::uint64_t>> d_t3_count   (N);
    std::vector<SyclDeviceVoid>               d_t3_temp    (N);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t3_unsorted[s] = sycl_alloc_device_owned<T3PairingGpu>(t3_cap, q);
        d_t3_count   [s] = sycl_alloc_device_owned<std::uint64_t>(1, q);

        std::size_t tb = 0;
        launch_t3_match_prepare(entry_.plot_id.data(), t3p,
            d_full_mi[s].get(), t2_total,
            d_t3_count[s].get(), nullptr, &tb, q);
        d_t3_temp[s] = SyclDeviceVoid(&q, sycl::malloc_device(tb, q));
        launch_t3_match_prepare(entry_.plot_id.data(), t3p,
            d_full_mi[s].get(), t2_total,
            d_t3_count[s].get(), d_t3_temp[s].get(), &tb, q);

        std::uint32_t const bucket_begin = t3_partition[s];
        std::uint32_t const bucket_end   = t3_partition[s + 1];

        launch_t3_match_range(entry_.plot_id.data(), t3p,
            d_full_meta[s].get(), d_full_xbits[s].get(), d_full_mi[s].get(),
            t2_total,
            d_t3_unsorted[s].get(), d_t3_count[s].get(),
            t3_cap, d_t3_temp[s].get(),
            bucket_begin, bucket_end, q);
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();

    std::vector<std::uint64_t> shard_count(N, 0);
    {
        std::vector<sycl::event> cnt_evts(N);
        for (std::size_t s = 0; s < N; ++s) {
            cnt_evts[s] = shards_[s].queue->memcpy(
                &shard_count[s], d_t3_count[s].get(), sizeof(std::uint64_t));
        }
        for (auto& e : cnt_evts) e.wait();
    }
    for (std::size_t s = 0; s < N; ++s) {
        if (shard_count[s] > t3_cap) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t3_phase: shard "
                + std::to_string(s) + " T3 produced "
                + std::to_string(shard_count[s])
                + " entries, exceeds capacity " + std::to_string(t3_cap));
        }
    }

    // Early-release the heavy intermediates.
    for (std::size_t s = 0; s < N; ++s) {
        d_t3_temp   [s].reset();
        d_t3_count  [s].reset();
        d_full_mi   [s].reset();
        d_full_meta [s].reset();
        d_full_xbits[s].reset();
    }

    sub_end("t3.match", t_t3_match);

    // ---------- Step 3 — distributed sort by proof_fragment. ----------
    auto const t_t3_sort = sub_begin();
    // T3PairingGpu is just a uint64_t; reinterpret in place. Sort over
    // the low 2*k bits to match GpuPipeline.cpp's launch_sort_keys_u64
    // call.
    std::uint64_t t3_total = 0;
    for (auto c : shard_count) t3_total += c;

    // out_capacity must hold this shard's bucket share after the
    // distributed sort redistributes by value range. With N shards and
    // a well-distributed bucket function (top log2(N) bits of
    // [begin_bit, end_bit)), each shard receives ~t3_total/N items —
    // proof_fragments are derived hashes, the standard deviation is
    // O(sqrt(t3_total/N)) and lands well under 1% by k=22. Sizing each
    // output buffer to t3_total (the previous setting) over-allocates
    // by N×: at k=28 with N=2 that's ~22 GB per shard, past the 20 GB
    // RTX 4000 Ada VRAM budget. 25% slack covers any bucket imbalance
    // we'd realistically see; the Peer / HostBounce path's
    // recv_count > out_capacity check still throws cleanly on
    // pathological inputs.
    std::uint64_t const per_shard_share = (t3_total + N - 1) / N;
    std::uint64_t const sort_cap = per_shard_share + per_shard_share / 4 + 1024;
    std::vector<SyclDevicePtr<std::uint64_t>> d_t3_frags_sorted(N);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t3_frags_sorted[s] = sycl_alloc_device_owned<std::uint64_t>(sort_cap, q);
    }

    std::vector<DistributedSortKeysU64Shard> sort_shards(N);
    for (std::size_t s = 0; s < N; ++s) {
        sort_shards[s].queue        = shards_[s].queue;
        sort_shards[s].keys_in      =
            reinterpret_cast<std::uint64_t*>(d_t3_unsorted[s].get());
        sort_shards[s].count        = shard_count[s];
        sort_shards[s].keys_out     = d_t3_frags_sorted[s].get();
        sort_shards[s].out_capacity = sort_cap;
        sort_shards[s].out_count    = 0;
        sort_shards[s].pool         = shards_[s].pool;
    }

    int const t3_end_bit = 2 * k;
    std::size_t scratch_bytes = 0;
    launch_sort_keys_u64_distributed(
        nullptr, scratch_bytes, sort_shards,
        /*begin_bit=*/0, /*end_bit=*/t3_end_bit, transport());
    SyclDeviceVoid d_scratch;
    if (scratch_bytes) {
        d_scratch = SyclDeviceVoid(shards_[0].queue,
            sycl::malloc_device(scratch_bytes, *shards_[0].queue));
    }
    launch_sort_keys_u64_distributed(
        d_scratch.get() ? d_scratch.get() : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards,
        /*begin_bit=*/0, /*end_bit=*/t3_end_bit, transport());
    d_scratch.reset();

    // Free unsorted T3 (sort already wrote into d_t3_frags_sorted).
    for (std::size_t s = 0; s < N; ++s) d_t3_unsorted[s].reset();

    for (std::size_t s = 0; s < N; ++s) {
        t3_phase_d_frags_[s] = d_t3_frags_sorted[s].release();
        t3_phase_count_  [s] = sort_shards[s].out_count;
    }
    sub_end("t3.sort", t_t3_sort);

    // Drop the T3 input replication slots — Fragment phase only needs
    // t3_phase_d_frags_ (the sorted output). Saves ~4 GB at k=28
    // during the fragment-phase D2H, and across plots until the next
    // plot's T3 phase re-grows them.
    for (std::size_t s = 0; s < N; ++s) {
        if (shards_[s].pool) {
            shards_[s].pool->clear_slot("t3_full_mi");
            shards_[s].pool->clear_slot("t3_full_meta");
            shards_[s].pool->clear_slot("t3_full_xbits");
        }
    }

    std::uint64_t out_total = 0;
    for (auto c : t3_phase_count_) out_total += c;
    if (out_total != t3_total) {
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_t3_phase: post-sort count mismatch "
            "(expected " + std::to_string(t3_total) + ", got "
            + std::to_string(out_total) + ")");
    }
}
void MultiGpuPlotPipeline::run_fragment_phase()
{
    // Phase 2.3d — D2H fan-in + concatenate.
    //
    // Each shard's t3_phase_d_frags_[s] holds proof_fragments whose low
    // 2*k bits fall in [s/N, (s+1)/N) of the 2k-bit value-space, sorted
    // within that range. Concatenating per-shard outputs in shard-id
    // order reproduces the single-GPU low-2k-bit-sorted layout —
    // byte-for-byte modulo the same-low-2k-bits tie order, which is
    // already non-deterministic between runs (the radix sort is stable
    // but the upstream T3 match emits via atomic cursor).
    //
    // Lands in a single pinned-host buffer h_fragments_ that survives
    // until run() returns; caller (BatchPlotter) feeds it straight to
    // write_plot_file_parallel, then the pipeline destructor frees it.

    std::size_t const N = shards_.size();
    sycl::queue& alloc_q = *shards_[0].queue;

    std::uint64_t total = 0;
    for (auto c : t3_phase_count_) total += c;

    if (h_fragments_) {
        sycl::free(h_fragments_, alloc_q);
        h_fragments_     = nullptr;
        fragments_count_ = 0;
    }
    if (total == 0) return;

    auto const t_frag_alloc = sub_begin();
    h_fragments_ = sycl::malloc_host<std::uint64_t>(total, alloc_q);
    sub_end("fragment.host_alloc", t_frag_alloc);

    // Submit all per-shard D2H copies concurrently into the right host
    // offset, then drain. Each shard's queue is on a different physical
    // GPU, so serializing per-shard with .wait() forces the transfers
    // onto a single PCIe lane at a time.
    auto const t_frag_d2h = sub_begin();
    std::vector<std::uint64_t> shard_off(N, 0);
    {
        std::uint64_t off = 0;
        for (std::size_t s = 0; s < N; ++s) {
            shard_off[s] = off;
            off += t3_phase_count_[s];
        }
    }
    std::vector<sycl::event> evts(N);
    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const c = t3_phase_count_[s];
        if (c == 0) continue;
        evts[s] = shards_[s].queue->memcpy(
            h_fragments_ + shard_off[s], t3_phase_d_frags_[s],
            c * sizeof(std::uint64_t));
    }
    for (std::size_t s = 0; s < N; ++s) {
        if (t3_phase_count_[s] > 0) evts[s].wait();
    }
    fragments_count_ = total;
    sub_end("fragment.d2h", t_frag_d2h);

    // Per-shard device fragments are no longer needed.
    for (std::size_t s = 0; s < N; ++s) {
        if (t3_phase_d_frags_[s]) {
            sycl::free(t3_phase_d_frags_[s], *shards_[s].queue);
            t3_phase_d_frags_[s] = nullptr;
        }
    }
}

} // namespace pos2gpu
