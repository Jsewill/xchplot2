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

} // namespace pos2gpu
