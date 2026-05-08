// MultiGpuPipelineParallel.hpp — Phase 2.1c orchestrator for plotting
// a single plot across two GPUs by splitting the streaming pipeline at
// the T2-sort boundary.
//
// GPU A (first device id) runs Xs / T1 / T2 phases. The sorted T2
// outputs are populated in host pinned scratch (h_meta / h_t2_meta /
// h_t2_xbits / h_keys_merged). GPU B (second device id) starts at T3
// match, reading those host buffers, runs through fragment + sort,
// and emits the final plot fragments.
//
// Supports the case where both ids point at the same physical device
// (single-GPU virtual 2-GPU test) so parity tests can validate the
// coordinator without real two-device hardware.

#pragma once

#include "host/GpuPipeline.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace pos2gpu {

struct PipelineParallelSplitResult {
    std::vector<std::uint64_t> fragments_storage;
    std::uint64_t t1_count = 0;
    std::uint64_t t2_count = 0;
    std::uint64_t t3_count = 0;

    std::span<std::uint64_t const> fragments() const noexcept
    {
        return {fragments_storage.data(), fragments_storage.size()};
    }
};

// Run a single plot across two GPUs by splitting at the T2-sort
// boundary. Requires tier=tiny on both halves (the boundary handoff
// uses the host-pinned T2 buffers that tiny mode populates).
//
// device_first runs Xs / T1 / T2; device_second runs T3 / Frag.
// Setting them equal runs both halves on the same device sequentially
// (validation path).
PipelineParallelSplitResult run_pipeline_parallel_split(
    GpuPipelineConfig const& cfg,
    int                      device_first,
    int                      device_second);

// Pipelined batch entry point. Runs a sequence of plots through the
// two-stage split with depth in-flight at the boundary — while plot
// N is on device_second (T3 + Frag), plot N+1 starts on device_first
// (Xs + T1 + T2). Steady-state throughput per plot ≈
// max(stage1_wall, stage2_wall) instead of (stage1+stage2).
//
// `depth` controls the number of pre-allocated boundary buffer sets.
// 2 is enough to overlap one plot per stage; higher depth helps when
// stage variance is high. Each set is cap-sized: ~6.2 GB pinned host
// at k=28 (4 buffers × 2 GB + 1 × 2 GB pinned_dst). depth × that is
// the host-pinned cost; default is 2.
//
// Returns one fragments vector per entry, in input order.
std::vector<PipelineParallelSplitResult> run_pipeline_parallel_batch(
    std::vector<GpuPipelineConfig> const& cfgs,
    int                                   device_first,
    int                                   device_second,
    int                                   depth = 2);

} // namespace pos2gpu
