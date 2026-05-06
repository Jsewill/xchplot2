// MultiGpuPlotPipeline.hpp — orchestrator for single-plot multi-GPU
// plotting (the --shard-plot path). Each shard owns its own
// sycl::queue (typically bound to a different physical GPU on a real
// multi-GPU host; on single-GPU dev boxes the parity tests use multiple
// shards pointing at the same physical queue) and a slice of the per-
// phase working set.
//
// Phase 2.2 scope: Xs gen + Xs sort + Xs pack across shards via
// launch_xs_gen_range + the distributed sort wrapper. T1/T2/T3
// matches and fragment serialization throw "not implemented in this
// phase" until Phase 2.3 lands them. See
// docs/multi-gpu-single-plot-alt-bucket-partition.md.

#pragma once

#include "host/BatchPlotter.hpp"
#include "host/MultiGpuShardBufferPool.hpp"
#include "gpu/SortDistributed.hpp"

#include <cstdint>
#include <span>
#include <vector>

#include <sycl/sycl.hpp>

namespace pos2gpu {

struct XsCandidateGpu;  // forward-declared; full def in gpu/XsCandidateGpu.hpp

// Pipeline phase identifier — used by run_through() and any caller
// that wants to reason about phase ordering. Strictly ordered: each
// phase requires the previous one's output.
enum class Phase {
    Xs,
    T1,
    T2,
    T3,
    Fragment,
};

// Per-shard runtime context. The caller (run_batch_sharded) constructs
// these — one per device id passed via --devices.
//
// `weight` is the relative throughput of this shard for load-balancing
// the per-phase match work across asymmetric rigs (e.g. a 4090 + a
// 3060 host). Each match phase's bucket range for shard s is
// proportional to weight[s] / sum(weights). Default 1.0 reproduces
// the uniform partition behaviour from before Phase 2.4a; pass
// non-uniform values to skew the work toward the faster cards.
struct MultiGpuShardContext {
    sycl::queue*     queue     = nullptr;
    int              device_id = -1;
    double           weight    = 1.0;
    // Optional buffer pool that persists across plots in a batch.
    // When non-null, the pipeline routes its largest allocations
    // (d_full_xs, full T1 / T2 replicas, sort-output staging) through
    // the pool so consecutive plots reuse the buffers. When null, the
    // pipeline falls back to per-plot malloc + free (matching the
    // pre-pool behaviour and the parity-test setup).
    ShardBufferPool* pool      = nullptr;
};

class MultiGpuPlotPipeline {
public:
    MultiGpuPlotPipeline(BatchEntry const& entry,
                         BatchOptions const& opts,
                         std::vector<MultiGpuShardContext> shards);
    ~MultiGpuPlotPipeline();

    // Runs the full plot pipeline through every phase. Phase 2.2
    // implements the Xs phase (gen + sort + pack); subsequent phases
    // throw a clear "Phase 2.3 not yet implemented" until they land.
    void run();

    // Run the pipeline up to and including the named phase. Each phase
    // requires the previous one's output, so passing Phase::T2 runs
    // Xs + T1 + T2. Public for parity tests that need to inspect the
    // intermediate per-shard state of a specific phase without driving
    // the rest of the chain. Production callers use run() (=
    // run_through(Phase::Fragment)).
    void run_through(Phase phase);

    // Convenience alias kept for the Xs-only parity test.
    void run_xs_phase() { run_through(Phase::Xs); }

    // After run() completes, returns a span over the concatenated host
    // buffer of sorted T3 proof_fragments — same shape as
    // GpuPipelineResult::fragments(), suitable to feed straight into
    // write_plot_file_parallel. Buffer lifetime is tied to the pipeline
    // object; freed by the destructor.
    std::span<std::uint64_t const> fragments() const noexcept
    {
        return {h_fragments_, fragments_count_};
    }
    std::uint64_t fragments_count() const noexcept { return fragments_count_; }

    // Per-shard Xs phase outputs. shard k's d_xs holds a packed
    // XsCandidateGpu array sized xs_phase_count(k). Concatenating in
    // shard-id order reproduces the single-GPU sorted XsCandidateGpu
    // array byte-for-byte.
    XsCandidateGpu* xs_phase_d_xs(std::size_t shard) const
    { return xs_phase_d_xs_[shard]; }
    std::uint64_t xs_phase_count(std::size_t shard) const
    { return xs_phase_count_[shard]; }

    // Per-shard T1 phase outputs (sorted by match_info). shard k holds
    // matches whose match_info falls in the [k/N, (k+1)/N) bucket of
    // the k-bit value-space. Concatenating in shard-id order reproduces
    // the single-GPU T1-sort output (same SoA layout as GpuPipeline:
    // d_keys_out / d_t1_meta_sorted).
    std::uint32_t* t1_phase_d_mi(std::size_t shard) const
    { return t1_phase_d_mi_[shard]; }
    std::uint64_t* t1_phase_d_meta(std::size_t shard) const
    { return t1_phase_d_meta_[shard]; }
    std::uint64_t  t1_phase_count(std::size_t shard) const
    { return t1_phase_count_[shard]; }

    // Per-shard T2 phase outputs (sorted by match_info). Same shard-
    // partitioning as the T1 outputs: shard k holds matches whose mi
    // falls in [k/N, (k+1)/N) of the k-bit value-space. Three SoA
    // streams mirror GpuPipeline.cpp's d_t2_meta_sorted /
    // d_t2_xbits_sorted / d_keys_out (the T2-sort match_info).
    std::uint32_t* t2_phase_d_mi(std::size_t shard) const
    { return t2_phase_d_mi_[shard]; }
    std::uint64_t* t2_phase_d_meta(std::size_t shard) const
    { return t2_phase_d_meta_[shard]; }
    std::uint32_t* t2_phase_d_xbits(std::size_t shard) const
    { return t2_phase_d_xbits_[shard]; }
    std::uint64_t  t2_phase_count(std::size_t shard) const
    { return t2_phase_count_[shard]; }

    // Per-shard T3 phase outputs (sorted by proof_fragment low 2k bits).
    // shard k holds fragments in [k * 2^(2k) / N, (k+1) * 2^(2k) / N).
    // Concatenating in shard-id order reproduces the single-GPU T3-sort
    // output (GpuPipeline.cpp's d_frags_out).
    std::uint64_t* t3_phase_d_frags(std::size_t shard) const
    { return t3_phase_d_frags_[shard]; }
    std::uint64_t  t3_phase_count(std::size_t shard) const
    { return t3_phase_count_[shard]; }

    std::size_t shard_count() const { return shards_.size(); }
    sycl::queue& shard_queue(std::size_t shard) const
    { return *shards_[shard].queue; }

private:
    BatchEntry                          entry_;
    BatchOptions                        opts_;
    std::vector<MultiGpuShardContext>   shards_;

    std::vector<XsCandidateGpu*>  xs_phase_d_xs_;
    std::vector<std::uint64_t>    xs_phase_count_;

    std::vector<std::uint32_t*>   t1_phase_d_mi_;
    std::vector<std::uint64_t*>   t1_phase_d_meta_;
    std::vector<std::uint64_t>    t1_phase_count_;

    std::vector<std::uint32_t*>   t2_phase_d_mi_;
    std::vector<std::uint64_t*>   t2_phase_d_meta_;
    std::vector<std::uint32_t*>   t2_phase_d_xbits_;
    std::vector<std::uint64_t>    t2_phase_count_;

    std::vector<std::uint64_t*>   t3_phase_d_frags_;
    std::vector<std::uint64_t>    t3_phase_count_;

    // Pinned-host concatenated fragment buffer produced by
    // run_fragment_phase. The pinning queue is shards_[0].queue (any
    // queue works; pinned host memory is process-wide).
    std::uint64_t* h_fragments_     = nullptr;
    std::uint64_t  fragments_count_ = 0;

    DistributedSortTransport transport() const noexcept
    {
        return opts_.prefer_peer_copy
            ? DistributedSortTransport::Peer
            : DistributedSortTransport::HostBounce;
    }

    void run_xs_phase_impl();
    void run_t1_phase();
    void run_t2_phase();
    void run_t3_phase();
    void run_fragment_phase();

    // Free all per-phase device allocations. Called from dtor on the
    // happy path and the unwind path so partial pipelines don't leak.
    void free_phase_outputs();
};

} // namespace pos2gpu
