// MultiGpuPlotPipeline.cpp — Phase 2.2 (Xs) + Phase 2.3a (T1) +
// Phase 2.3b (T2) + Phase 2.3c (T3) + Phase 2.3d (fragment) impl.

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
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

namespace {

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
    run_xs_phase();
    run_t1_phase();
    run_t2_phase();
    run_t3_phase();
    run_fragment_phase();
}

void MultiGpuPlotPipeline::run_xs_phase()
{
    std::size_t const N = shards_.size();
    int const k = entry_.k;
    std::uint64_t const total_xs    = std::uint64_t{1} << k;
    std::uint32_t const xs_xor_const = entry_.testnet ? 0xA3B1C4D7u : 0u;
    AesHashKeys const xs_keys = make_keys(entry_.plot_id.data());

    // Step 1 — per-shard Xs gen via launch_xs_gen_range. Shard k owns
    // the position range [k*total_xs/N, (k+1)*total_xs/N). Outputs
    // land in per-shard u32 keys/vals scratch on the shard's device.
    std::vector<std::uint32_t*> d_keys_in (N, nullptr);
    std::vector<std::uint32_t*> d_vals_in (N, nullptr);
    std::vector<std::uint32_t*> d_keys_out(N, nullptr);
    std::vector<std::uint32_t*> d_vals_out(N, nullptr);
    std::vector<std::uint64_t>  shard_in_count(N, 0);

    // Worst-case per-shard receive count for the distributed sort:
    // total_xs (if all items happened to land in one bucket — highly
    // unlikely with g_x's near-uniform distribution, but the API
    // contract is "out_capacity ≥ max possible inflow").
    std::uint64_t const out_cap = total_xs;

    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const pos_begin = (s * total_xs) / N;
        std::uint64_t const pos_end   = ((s + 1) * total_xs) / N;
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

    // Step 2 — distributed sort. Each shard becomes a
    // DistributedSortPairsShard. After the call, shard k's keys_out /
    // vals_out hold its bucket range of sorted (key, val) pairs;
    // shards[k].out_count is the actual count.
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
    }
    std::size_t scratch_bytes = 0;
    launch_sort_pairs_u32_u32_distributed(
        nullptr, scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k);
    void* d_scratch = scratch_bytes
        ? sycl::malloc_device(scratch_bytes, *shards_[0].queue)
        : nullptr;
    launch_sort_pairs_u32_u32_distributed(
        d_scratch ? d_scratch : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k);
    if (d_scratch) sycl::free(d_scratch, *shards_[0].queue);

    // Step 3 — per-shard launch_xs_pack into a packed XsCandidateGpu
    // output sized for the shard's bucket range. The packed output is
    // what T1 match consumes in Phase 2.3.
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

void MultiGpuPlotPipeline::run_xs_then_t1_phase()
{
    run_xs_phase();
    run_t1_phase();
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
    sycl::queue& alloc_q = *shards_[0].queue;
    XsCandidateGpu* h_full = sycl::malloc_host<XsCandidateGpu>(total_xs, alloc_q);

    std::uint64_t off = 0;
    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const c = xs_phase_count_[s];
        if (c > 0) {
            shards_[s].queue->memcpy(
                h_full + off, xs_phase_d_xs_[s],
                sizeof(XsCandidateGpu) * c).wait();
        }
        off += c;
    }
    if (off != total_xs) {
        sycl::free(h_full, alloc_q);
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_t1_phase: Xs phase outputs sum to "
            + std::to_string(off) + " entries but total_xs = "
            + std::to_string(total_xs)
            + ". run_xs_phase() must complete before run_t1_phase().");
    }

    std::vector<XsCandidateGpu*> d_full_xs(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_full_xs[s] = sycl::malloc_device<XsCandidateGpu>(total_xs, q);
        q.memcpy(d_full_xs[s], h_full,
                 sizeof(XsCandidateGpu) * total_xs).wait();
    }
    sycl::free(h_full, alloc_q);

    // The bucket-partitioned Xs phase outputs are no longer needed —
    // d_full_xs holds the same data on every shard. Free them now to
    // recover memory before T1 staging allocations.
    for (std::size_t s = 0; s < N; ++s) {
        if (xs_phase_d_xs_[s]) {
            sycl::free(xs_phase_d_xs_[s], *shards_[s].queue);
            xs_phase_d_xs_[s] = nullptr;
        }
    }

    // ---------- Step 2 — per-shard T1 match over assigned buckets. ----
    // Capacity: max_pairs_per_section * num_sections — the standard
    // pos2-chip pool sizing for T1 output, which adds an extra-margin-
    // bits term over the naive 2^k count to absorb expected skew. Same
    // formula GpuBufferPool uses for the single-GPU T1 cap, giving each
    // shard the same per-bucket safety as the production single-GPU
    // path. Each shard sees the FULL input + emits only its bucket
    // subset, so per-shard count <= full t1 cap on the worst case.
    std::uint32_t const num_sections = std::uint32_t{1} << t1p.num_section_bits;
    std::uint64_t const t1_cap =
        static_cast<std::uint64_t>(
            max_pairs_per_section(k, t1p.num_section_bits)) * num_sections;

    std::vector<std::uint64_t*> d_t1_meta_unsorted(N, nullptr);
    std::vector<std::uint32_t*> d_t1_mi_unsorted  (N, nullptr);
    std::vector<std::uint64_t*> d_t1_count        (N, nullptr);
    std::vector<void*>          d_t1_temp         (N, nullptr);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t1_meta_unsorted[s] = sycl::malloc_device<std::uint64_t>(t1_cap, q);
        d_t1_mi_unsorted  [s] = sycl::malloc_device<std::uint32_t>(t1_cap, q);
        d_t1_count        [s] = sycl::malloc_device<std::uint64_t>(1, q);

        std::size_t tb = 0;
        launch_t1_match_prepare(entry_.plot_id.data(), t1p,
            d_full_xs[s], total_xs,
            d_t1_count[s], nullptr, &tb, q);
        d_t1_temp[s] = sycl::malloc_device(tb, q);
        launch_t1_match_prepare(entry_.plot_id.data(), t1p,
            d_full_xs[s], total_xs,
            d_t1_count[s], d_t1_temp[s], &tb, q);

        std::uint32_t const bucket_begin = t1_partition[s];
        std::uint32_t const bucket_end   = t1_partition[s + 1];

        launch_t1_match_range(entry_.plot_id.data(), t1p,
            d_full_xs[s], total_xs,
            d_t1_meta_unsorted[s], d_t1_mi_unsorted[s], d_t1_count[s],
            t1_cap, d_t1_temp[s],
            bucket_begin, bucket_end, q);
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();

    std::vector<std::uint64_t> shard_count(N, 0);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        q.memcpy(&shard_count[s], d_t1_count[s], sizeof(std::uint64_t)).wait();
        if (shard_count[s] > t1_cap) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t1_phase: shard "
                + std::to_string(s) + " T1 produced "
                + std::to_string(shard_count[s])
                + " entries, exceeds capacity " + std::to_string(t1_cap)
                + ". Bucket-skew worse than expected; raise t1_cap.");
        }
    }

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        sycl::free(d_t1_temp [s], q);
        sycl::free(d_t1_count[s], q);
        sycl::free(d_full_xs [s], q);
    }

    // ---------- Step 3 — distributed sort by mi. -------------------
    // Worst-case per-shard receive: the union total (skewed input case).
    std::uint64_t t1_total = 0;
    for (auto c : shard_count) t1_total += c;

    std::uint64_t const sort_cap = t1_total;
    std::vector<std::uint32_t*> d_t1_mi_sorted  (N, nullptr);
    std::vector<std::uint64_t*> d_t1_meta_sorted(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t1_mi_sorted  [s] = sycl::malloc_device<std::uint32_t>(sort_cap, q);
        d_t1_meta_sorted[s] = sycl::malloc_device<std::uint64_t>(sort_cap, q);
    }

    std::vector<DistributedSortPairsU32U64Shard> sort_shards(N);
    for (std::size_t s = 0; s < N; ++s) {
        sort_shards[s].queue        = shards_[s].queue;
        sort_shards[s].keys_in      = d_t1_mi_unsorted[s];
        sort_shards[s].vals_in      = d_t1_meta_unsorted[s];
        sort_shards[s].count        = shard_count[s];
        sort_shards[s].keys_out     = d_t1_mi_sorted[s];
        sort_shards[s].vals_out     = d_t1_meta_sorted[s];
        sort_shards[s].out_capacity = sort_cap;
        sort_shards[s].out_count    = 0;
    }

    std::size_t scratch_bytes = 0;
    launch_sort_pairs_u32_u64_distributed(
        nullptr, scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k);
    void* d_scratch = scratch_bytes
        ? sycl::malloc_device(scratch_bytes, *shards_[0].queue) : nullptr;
    launch_sort_pairs_u32_u64_distributed(
        d_scratch ? d_scratch : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k);
    if (d_scratch) sycl::free(d_scratch, *shards_[0].queue);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        sycl::free(d_t1_mi_unsorted  [s], q);
        sycl::free(d_t1_meta_unsorted[s], q);
    }

    for (std::size_t s = 0; s < N; ++s) {
        t1_phase_d_mi_  [s] = d_t1_mi_sorted  [s];
        t1_phase_d_meta_[s] = d_t1_meta_sorted[s];
        t1_phase_count_ [s] = sort_shards[s].out_count;
    }

    std::uint64_t out_total = 0;
    for (auto c : t1_phase_count_) out_total += c;
    if (out_total != t1_total) {
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_t1_phase: post-sort count mismatch "
            "(expected " + std::to_string(t1_total) + ", got "
            + std::to_string(out_total) + ")");
    }
}

void MultiGpuPlotPipeline::run_xs_then_t1_then_t2_phase()
{
    run_xs_phase();
    run_t1_phase();
    run_t2_phase();
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
    std::uint64_t t1_total = 0;
    for (auto c : t1_phase_count_) t1_total += c;

    sycl::queue& alloc_q = *shards_[0].queue;
    std::uint32_t* h_mi   = sycl::malloc_host<std::uint32_t>(t1_total, alloc_q);
    std::uint64_t* h_meta = sycl::malloc_host<std::uint64_t>(t1_total, alloc_q);

    std::uint64_t off = 0;
    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const c = t1_phase_count_[s];
        if (c > 0) {
            shards_[s].queue->memcpy(
                h_mi + off,   t1_phase_d_mi_[s],
                c * sizeof(std::uint32_t)).wait();
            shards_[s].queue->memcpy(
                h_meta + off, t1_phase_d_meta_[s],
                c * sizeof(std::uint64_t)).wait();
        }
        off += c;
    }
    if (off != t1_total) {
        sycl::free(h_mi,   alloc_q);
        sycl::free(h_meta, alloc_q);
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_t2_phase: T1 outputs sum to "
            + std::to_string(off) + " entries but t1_total = "
            + std::to_string(t1_total));
    }

    std::vector<std::uint32_t*> d_full_mi  (N, nullptr);
    std::vector<std::uint64_t*> d_full_meta(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_full_mi  [s] = sycl::malloc_device<std::uint32_t>(t1_total, q);
        d_full_meta[s] = sycl::malloc_device<std::uint64_t>(t1_total, q);
        q.memcpy(d_full_mi  [s], h_mi,
                 t1_total * sizeof(std::uint32_t)).wait();
        q.memcpy(d_full_meta[s], h_meta,
                 t1_total * sizeof(std::uint64_t)).wait();
    }
    sycl::free(h_mi,   alloc_q);
    sycl::free(h_meta, alloc_q);

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

    // ---------- Step 2 — per-shard T2 match. ----------
    // Same pos2-chip pool sizing as the single-GPU path: T2 emits at
    // most max_pairs_per_section * num_sections matches.
    std::uint32_t const num_sections_t2 =
        std::uint32_t{1} << t2p.num_section_bits;
    std::uint64_t const t2_cap =
        static_cast<std::uint64_t>(
            max_pairs_per_section(k, t2p.num_section_bits)) * num_sections_t2;

    std::vector<std::uint64_t*> d_t2_meta_unsorted (N, nullptr);
    std::vector<std::uint32_t*> d_t2_mi_unsorted   (N, nullptr);
    std::vector<std::uint32_t*> d_t2_xbits_unsorted(N, nullptr);
    std::vector<std::uint64_t*> d_t2_count         (N, nullptr);
    std::vector<void*>          d_t2_temp          (N, nullptr);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t2_meta_unsorted [s] = sycl::malloc_device<std::uint64_t>(t2_cap, q);
        d_t2_mi_unsorted   [s] = sycl::malloc_device<std::uint32_t>(t2_cap, q);
        d_t2_xbits_unsorted[s] = sycl::malloc_device<std::uint32_t>(t2_cap, q);
        d_t2_count         [s] = sycl::malloc_device<std::uint64_t>(1, q);

        std::size_t tb = 0;
        launch_t2_match_prepare(entry_.plot_id.data(), t2p,
            d_full_mi[s], t1_total,
            d_t2_count[s], nullptr, &tb, q);
        d_t2_temp[s] = sycl::malloc_device(tb, q);
        launch_t2_match_prepare(entry_.plot_id.data(), t2p,
            d_full_mi[s], t1_total,
            d_t2_count[s], d_t2_temp[s], &tb, q);

        std::uint32_t const bucket_begin = t2_partition[s];
        std::uint32_t const bucket_end   = t2_partition[s + 1];

        launch_t2_match_range(entry_.plot_id.data(), t2p,
            d_full_meta[s], d_full_mi[s], t1_total,
            d_t2_meta_unsorted[s], d_t2_mi_unsorted[s],
            d_t2_xbits_unsorted[s], d_t2_count[s],
            t2_cap, d_t2_temp[s],
            bucket_begin, bucket_end, q);
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();

    std::vector<std::uint64_t> shard_count(N, 0);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        q.memcpy(&shard_count[s], d_t2_count[s], sizeof(std::uint64_t)).wait();
        if (shard_count[s] > t2_cap) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t2_phase: shard "
                + std::to_string(s) + " T2 produced "
                + std::to_string(shard_count[s])
                + " entries, exceeds capacity " + std::to_string(t2_cap));
        }
    }

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        sycl::free(d_t2_temp  [s], q);
        sycl::free(d_t2_count [s], q);
        sycl::free(d_full_mi  [s], q);
        sycl::free(d_full_meta[s], q);
    }

    // ---------- Step 3 — distributed sort by mi. ----------
    std::uint64_t t2_total = 0;
    for (auto c : shard_count) t2_total += c;

    std::uint64_t const sort_cap = t2_total;
    std::vector<std::uint32_t*> d_t2_mi_sorted   (N, nullptr);
    std::vector<std::uint64_t*> d_t2_meta_sorted (N, nullptr);
    std::vector<std::uint32_t*> d_t2_xbits_sorted(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t2_mi_sorted   [s] = sycl::malloc_device<std::uint32_t>(sort_cap, q);
        d_t2_meta_sorted [s] = sycl::malloc_device<std::uint64_t>(sort_cap, q);
        d_t2_xbits_sorted[s] = sycl::malloc_device<std::uint32_t>(sort_cap, q);
    }

    std::vector<DistributedSortPairsU32U64U32Shard> sort_shards(N);
    for (std::size_t s = 0; s < N; ++s) {
        sort_shards[s].queue        = shards_[s].queue;
        sort_shards[s].keys_in      = d_t2_mi_unsorted[s];
        sort_shards[s].vals_a_in    = d_t2_meta_unsorted[s];
        sort_shards[s].vals_b_in    = d_t2_xbits_unsorted[s];
        sort_shards[s].count        = shard_count[s];
        sort_shards[s].keys_out     = d_t2_mi_sorted[s];
        sort_shards[s].vals_a_out   = d_t2_meta_sorted[s];
        sort_shards[s].vals_b_out   = d_t2_xbits_sorted[s];
        sort_shards[s].out_capacity = sort_cap;
        sort_shards[s].out_count    = 0;
    }

    std::size_t scratch_bytes = 0;
    launch_sort_pairs_u32_u64u32_distributed(
        nullptr, scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k);
    void* d_scratch = scratch_bytes
        ? sycl::malloc_device(scratch_bytes, *shards_[0].queue) : nullptr;
    launch_sort_pairs_u32_u64u32_distributed(
        d_scratch ? d_scratch : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards, /*begin_bit=*/0, /*end_bit=*/k);
    if (d_scratch) sycl::free(d_scratch, *shards_[0].queue);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        sycl::free(d_t2_mi_unsorted   [s], q);
        sycl::free(d_t2_meta_unsorted [s], q);
        sycl::free(d_t2_xbits_unsorted[s], q);
    }

    for (std::size_t s = 0; s < N; ++s) {
        t2_phase_d_mi_   [s] = d_t2_mi_sorted   [s];
        t2_phase_d_meta_ [s] = d_t2_meta_sorted [s];
        t2_phase_d_xbits_[s] = d_t2_xbits_sorted[s];
        t2_phase_count_  [s] = sort_shards[s].out_count;
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

void MultiGpuPlotPipeline::run_xs_then_t1_then_t2_then_t3_phase()
{
    run_xs_phase();
    run_t1_phase();
    run_t2_phase();
    run_t3_phase();
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
    std::uint64_t t2_total = 0;
    for (auto c : t2_phase_count_) t2_total += c;

    sycl::queue& alloc_q = *shards_[0].queue;
    std::uint32_t* h_mi    = sycl::malloc_host<std::uint32_t>(t2_total, alloc_q);
    std::uint64_t* h_meta  = sycl::malloc_host<std::uint64_t>(t2_total, alloc_q);
    std::uint32_t* h_xbits = sycl::malloc_host<std::uint32_t>(t2_total, alloc_q);

    std::uint64_t off = 0;
    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const c = t2_phase_count_[s];
        if (c > 0) {
            shards_[s].queue->memcpy(
                h_mi + off,    t2_phase_d_mi_[s],
                c * sizeof(std::uint32_t)).wait();
            shards_[s].queue->memcpy(
                h_meta + off,  t2_phase_d_meta_[s],
                c * sizeof(std::uint64_t)).wait();
            shards_[s].queue->memcpy(
                h_xbits + off, t2_phase_d_xbits_[s],
                c * sizeof(std::uint32_t)).wait();
        }
        off += c;
    }
    if (off != t2_total) {
        sycl::free(h_mi,    alloc_q);
        sycl::free(h_meta,  alloc_q);
        sycl::free(h_xbits, alloc_q);
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_t3_phase: T2 outputs sum to "
            + std::to_string(off) + " entries but t2_total = "
            + std::to_string(t2_total));
    }

    std::vector<std::uint32_t*> d_full_mi   (N, nullptr);
    std::vector<std::uint64_t*> d_full_meta (N, nullptr);
    std::vector<std::uint32_t*> d_full_xbits(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_full_mi   [s] = sycl::malloc_device<std::uint32_t>(t2_total, q);
        d_full_meta [s] = sycl::malloc_device<std::uint64_t>(t2_total, q);
        d_full_xbits[s] = sycl::malloc_device<std::uint32_t>(t2_total, q);
        q.memcpy(d_full_mi   [s], h_mi,
                 t2_total * sizeof(std::uint32_t)).wait();
        q.memcpy(d_full_meta [s], h_meta,
                 t2_total * sizeof(std::uint64_t)).wait();
        q.memcpy(d_full_xbits[s], h_xbits,
                 t2_total * sizeof(std::uint32_t)).wait();
    }
    sycl::free(h_mi,    alloc_q);
    sycl::free(h_meta,  alloc_q);
    sycl::free(h_xbits, alloc_q);

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

    // ---------- Step 2 — per-shard T3 match. ----------
    std::uint32_t const num_sections_t3 =
        std::uint32_t{1} << t3p.num_section_bits;
    std::uint64_t const t3_cap =
        static_cast<std::uint64_t>(
            max_pairs_per_section(k, t3p.num_section_bits)) * num_sections_t3;

    std::vector<T3PairingGpu*>  d_t3_unsorted(N, nullptr);
    std::vector<std::uint64_t*> d_t3_count   (N, nullptr);
    std::vector<void*>          d_t3_temp    (N, nullptr);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t3_unsorted[s] = sycl::malloc_device<T3PairingGpu>(t3_cap, q);
        d_t3_count   [s] = sycl::malloc_device<std::uint64_t>(1, q);

        std::size_t tb = 0;
        launch_t3_match_prepare(entry_.plot_id.data(), t3p,
            d_full_mi[s], t2_total,
            d_t3_count[s], nullptr, &tb, q);
        d_t3_temp[s] = sycl::malloc_device(tb, q);
        launch_t3_match_prepare(entry_.plot_id.data(), t3p,
            d_full_mi[s], t2_total,
            d_t3_count[s], d_t3_temp[s], &tb, q);

        std::uint32_t const bucket_begin = t3_partition[s];
        std::uint32_t const bucket_end   = t3_partition[s + 1];

        launch_t3_match_range(entry_.plot_id.data(), t3p,
            d_full_meta[s], d_full_xbits[s], d_full_mi[s], t2_total,
            d_t3_unsorted[s], d_t3_count[s],
            t3_cap, d_t3_temp[s],
            bucket_begin, bucket_end, q);
    }
    for (std::size_t s = 0; s < N; ++s) shards_[s].queue->wait();

    std::vector<std::uint64_t> shard_count(N, 0);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        q.memcpy(&shard_count[s], d_t3_count[s], sizeof(std::uint64_t)).wait();
        if (shard_count[s] > t3_cap) {
            throw std::runtime_error(
                "MultiGpuPlotPipeline::run_t3_phase: shard "
                + std::to_string(s) + " T3 produced "
                + std::to_string(shard_count[s])
                + " entries, exceeds capacity " + std::to_string(t3_cap));
        }
    }

    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        sycl::free(d_t3_temp   [s], q);
        sycl::free(d_t3_count  [s], q);
        sycl::free(d_full_mi   [s], q);
        sycl::free(d_full_meta [s], q);
        sycl::free(d_full_xbits[s], q);
    }

    // ---------- Step 3 — distributed sort by proof_fragment. ----------
    // T3PairingGpu is just a uint64_t; reinterpret in place. Sort over
    // the low 2*k bits to match GpuPipeline.cpp's launch_sort_keys_u64
    // call.
    std::uint64_t t3_total = 0;
    for (auto c : shard_count) t3_total += c;

    std::uint64_t const sort_cap = t3_total;
    std::vector<std::uint64_t*> d_t3_frags_sorted(N, nullptr);
    for (std::size_t s = 0; s < N; ++s) {
        sycl::queue& q = *shards_[s].queue;
        d_t3_frags_sorted[s] = sycl::malloc_device<std::uint64_t>(sort_cap, q);
    }

    std::vector<DistributedSortKeysU64Shard> sort_shards(N);
    for (std::size_t s = 0; s < N; ++s) {
        sort_shards[s].queue        = shards_[s].queue;
        sort_shards[s].keys_in      =
            reinterpret_cast<std::uint64_t*>(d_t3_unsorted[s]);
        sort_shards[s].count        = shard_count[s];
        sort_shards[s].keys_out     = d_t3_frags_sorted[s];
        sort_shards[s].out_capacity = sort_cap;
        sort_shards[s].out_count    = 0;
    }

    int const t3_end_bit = 2 * k;
    std::size_t scratch_bytes = 0;
    launch_sort_keys_u64_distributed(
        nullptr, scratch_bytes, sort_shards,
        /*begin_bit=*/0, /*end_bit=*/t3_end_bit);
    void* d_scratch = scratch_bytes
        ? sycl::malloc_device(scratch_bytes, *shards_[0].queue) : nullptr;
    launch_sort_keys_u64_distributed(
        d_scratch ? d_scratch : reinterpret_cast<void*>(std::uintptr_t{1}),
        scratch_bytes, sort_shards,
        /*begin_bit=*/0, /*end_bit=*/t3_end_bit);
    if (d_scratch) sycl::free(d_scratch, *shards_[0].queue);

    for (std::size_t s = 0; s < N; ++s) {
        sycl::free(d_t3_unsorted[s], *shards_[s].queue);
    }

    for (std::size_t s = 0; s < N; ++s) {
        t3_phase_d_frags_[s] = d_t3_frags_sorted[s];
        t3_phase_count_  [s] = sort_shards[s].out_count;
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

    h_fragments_ = sycl::malloc_host<std::uint64_t>(total, alloc_q);

    std::uint64_t off = 0;
    for (std::size_t s = 0; s < N; ++s) {
        std::uint64_t const c = t3_phase_count_[s];
        if (c > 0) {
            shards_[s].queue->memcpy(
                h_fragments_ + off, t3_phase_d_frags_[s],
                c * sizeof(std::uint64_t)).wait();
        }
        off += c;
    }
    if (off != total) {
        sycl::free(h_fragments_, alloc_q);
        h_fragments_     = nullptr;
        fragments_count_ = 0;
        throw std::runtime_error(
            "MultiGpuPlotPipeline::run_fragment_phase: per-shard counts "
            "sum to " + std::to_string(off) + " but expected "
            + std::to_string(total));
    }
    fragments_count_ = total;

    // Per-shard device fragments are no longer needed.
    for (std::size_t s = 0; s < N; ++s) {
        if (t3_phase_d_frags_[s]) {
            sycl::free(t3_phase_d_frags_[s], *shards_[s].queue);
            t3_phase_d_frags_[s] = nullptr;
        }
    }
}

} // namespace pos2gpu
