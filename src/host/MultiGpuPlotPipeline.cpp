// MultiGpuPlotPipeline.cpp — Phase 2.2 implementation: Xs phase only.

#include "host/MultiGpuPlotPipeline.hpp"

#include "gpu/AesHashGpu.cuh"
#include "gpu/Sort.cuh"
#include "gpu/SortDistributed.hpp"
#include "gpu/XsCandidateGpu.hpp"
#include "gpu/XsKernels.cuh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

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

void MultiGpuPlotPipeline::run_t1_phase()
{
    throw std::runtime_error(
        "MultiGpuPlotPipeline: T1 match is not yet implemented for the "
        "sharded path (Phase 2.3 in the plan). Phase 2.2 ships the Xs "
        "phase end-to-end (gen + sort + pack across shards) and the "
        "distributed sort primitive; the per-shard T1/T2/T3 matches "
        "with their cross-shard boundary fixups land next. See "
        "docs/multi-gpu-single-plot-alt-bucket-partition.md.");
}

void MultiGpuPlotPipeline::run_t2_phase() { /* unreachable past T1's throw */ }
void MultiGpuPlotPipeline::run_t3_phase() { /* unreachable */ }
void MultiGpuPlotPipeline::run_fragment_phase() { /* unreachable */ }

} // namespace pos2gpu
