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
#include <functional>
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

// Per-stage streaming tier. Only Tiny and Minimal are valid in the
// pipeline-parallel boundary handoff — the upper tiers (Plain /
// Compact) don't surface T2 sorted outputs to host pinned memory.
//
// Tiny is the smaller-VRAM default; Minimal trades device VRAM for
// (typically) lower wall-clock per stage. On a heterogeneous rig you
// can pick Minimal on the larger-VRAM card and Tiny on the smaller.
enum class PipelineStageTier {
    Tiny,
    Minimal,
};

// Run a single plot across N GPUs by splitting the streaming pipeline
// at one or more phase boundaries. Today only the T2-sort boundary is
// implemented, so device_ids.size() must be 2: device_ids[0] runs
// Xs / T1 / T2; device_ids[1] runs T3 / Frag. Setting both equal runs
// both halves on the same device sequentially (validation path).
//
// `tiers[i]` selects per-stage streaming tier; empty (default) means
// Tiny for every stage. When non-empty, tiers.size() must equal
// device_ids.size().
PipelineParallelSplitResult run_pipeline_parallel_split(
    GpuPipelineConfig const&              cfg,
    std::vector<int> const&               device_ids,
    std::vector<PipelineStageTier> const& tiers = {});

// Pipelined batch entry point. Runs a sequence of plots through the
// N-stage split with `depth` plots in-flight at each boundary — while
// plot N is on the last stage, plot N+1 starts on the first. Steady-
// state throughput per plot ≈ max(stage_i_wall) once depth ≥ N.
//
// `depth` controls the number of pre-allocated boundary buffer sets
// per boundary. 2 is enough to overlap one plot per stage; higher
// depth helps when stage variance is high. Each set is cap-sized
// (~6.2 GB pinned host at k=28 for the T2-sort boundary).
// depth × num_boundaries × per-set cost is the host-pinned total;
// default depth is 2.
//
// `device_ids[i]` runs stage i; today device_ids.size() must be 2.
// `tiers` is parallel to device_ids; empty defaults to Tiny per stage.
//
// Returns one fragments vector per entry, in input order.
// Optional plot-completion callback. Phase 2.1f: fires from the
// final-stage worker thread immediately after plot N's result is
// captured. Lets BatchPlotter hand the completed fragments to a
// writer thread without waiting for the whole batch to finish
// (overlaps FSE+disk with subsequent plots' GPU work). Runs on
// the worker thread — keep it short + thread-safe; typically just
// pushes (idx, result) onto a queue.
//
// When the callback is set the orchestrator MOVES result[cfg_idx]
// into the callback parameter (avoids a ~240 MB copy at k=28); the
// returned results vector has hollow fragments_storage for those
// slots. Callers using the callback should consume the result via
// the callback and not the return value.
using PipelineBatchPlotCallback =
    std::function<void(int cfg_idx, PipelineParallelSplitResult result)>;

std::vector<PipelineParallelSplitResult> run_pipeline_parallel_batch(
    std::vector<GpuPipelineConfig> const& cfgs,
    std::vector<int> const&               device_ids,
    int                                   depth = 2,
    std::vector<PipelineStageTier> const& tiers = {},
    PipelineBatchPlotCallback             on_plot_complete = {});

// Phase 2-C: device-VRAM-aware stage assignment.
//
// Stage 1 (Xs + T1 + T2 sort) is the heavier VRAM consumer in every
// tier — it has to hold the Xs candidate stream + CUB radix-sort
// scratch on device. Stage 2 (T3 match + frag) is meaningfully
// smaller, especially in the sliced minimal/tiny tiers.
//
// On a heterogeneous rig (e.g. 24 GB + 8 GB) the right policy is to
// put stage 1 on the larger card so the small card can auto-pick a
// smaller streaming tier (compact / minimal / tiny) without forcing
// the large card down. select_pipeline_devices() applies that policy.
//
// On a uniform rig the function is a no-op (input order preserved
// when VRAM ties).
struct PipelineDeviceAssignment {
    // N-stage canonical view (always populated). dev_ids[i] is the
    // device assigned to stage i; dev_vram_bytes[i] is its VRAM.
    std::vector<int>           dev_ids;
    std::vector<std::uint64_t> dev_vram_bytes;
    bool                       reordered = false;

    // 2-stage scalar view (only populated when dev_ids.size() == 2).
    // Pre-N-stage callers and existing parity tests read these; new
    // code should prefer dev_ids / dev_vram_bytes.
    int           dev_first              = 0;
    int           dev_second             = 0;
    std::uint64_t dev_first_vram_bytes   = 0;
    std::uint64_t dev_second_vram_bytes  = 0;
};

// Default form — looks up VRAM via
//   sycl_backend::usable_gpu_devices()[id].get_info<global_mem_size>()
// device_ids.size() must equal 2 until N-stage assignment lands.
PipelineDeviceAssignment select_pipeline_devices(
    std::vector<int> const& device_ids);

// Test seam — caller supplies the VRAM lookup. Used by parity tests
// to exercise the swap without real heterogeneous hardware.
PipelineDeviceAssignment select_pipeline_devices(
    std::vector<int> const&                  device_ids,
    std::function<std::uint64_t(int)> const& vram_for_device);

} // namespace pos2gpu
