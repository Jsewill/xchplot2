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

#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

namespace pos2gpu {

struct XsCandidateGpu;  // forward-declared; full def in gpu/XsCandidateGpu.hpp

// Per-shard runtime context. The caller (run_batch_sharded) constructs
// these — one per device id passed via --devices.
struct MultiGpuShardContext {
    sycl::queue* queue     = nullptr;
    int          device_id = -1;
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

    // Run only the Xs phase (gen + sort + pack across shards). Public
    // for parity tests that need to inspect intermediate state without
    // hitting the T1 phase's "not yet implemented" throw. Production
    // callers should use run() — the multi-step pipeline.
    void run_xs_phase();

    // Per-shard Xs phase outputs. shard k's d_xs holds a packed
    // XsCandidateGpu array sized xs_phase_count(k). Concatenating in
    // shard-id order reproduces the single-GPU sorted XsCandidateGpu
    // array byte-for-byte.
    XsCandidateGpu* xs_phase_d_xs(std::size_t shard) const
    { return xs_phase_d_xs_[shard]; }
    std::uint64_t xs_phase_count(std::size_t shard) const
    { return xs_phase_count_[shard]; }
    std::size_t shard_count() const { return shards_.size(); }
    sycl::queue& shard_queue(std::size_t shard) const
    { return *shards_[shard].queue; }

private:
    BatchEntry                          entry_;
    BatchOptions                        opts_;
    std::vector<MultiGpuShardContext>   shards_;

    std::vector<XsCandidateGpu*> xs_phase_d_xs_;
    std::vector<std::uint64_t>   xs_phase_count_;

    void run_t1_phase();
    void run_t2_phase();
    void run_t3_phase();
    void run_fragment_phase();

    // Free all per-phase device allocations. Called from dtor on the
    // happy path and the unwind path so partial pipelines don't leak.
    void free_phase_outputs();
};

} // namespace pos2gpu
