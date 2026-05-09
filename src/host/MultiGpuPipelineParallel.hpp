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

// Run a single plot across two GPUs by splitting at the T2-sort
// boundary. tier_first / tier_second select per-stage streaming tier
// (default Tiny+Tiny matches pre-Phase-2-E behaviour).
//
// device_first runs Xs / T1 / T2; device_second runs T3 / Frag.
// Setting them equal runs both halves on the same device sequentially
// (validation path).
PipelineParallelSplitResult run_pipeline_parallel_split(
    GpuPipelineConfig const& cfg,
    int                      device_first,
    int                      device_second,
    PipelineStageTier        tier_first  = PipelineStageTier::Tiny,
    PipelineStageTier        tier_second = PipelineStageTier::Tiny);

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
// tier_first / tier_second default to Tiny+Tiny.
std::vector<PipelineParallelSplitResult> run_pipeline_parallel_batch(
    std::vector<GpuPipelineConfig> const& cfgs,
    int                                   device_first,
    int                                   device_second,
    int                                   depth = 2,
    PipelineStageTier                     tier_first  = PipelineStageTier::Tiny,
    PipelineStageTier                     tier_second = PipelineStageTier::Tiny);

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
    int           dev_first              = 0;
    int           dev_second             = 0;
    std::uint64_t dev_first_vram_bytes   = 0;
    std::uint64_t dev_second_vram_bytes  = 0;
    bool          reordered              = false;
};

// Default form — looks up VRAM via
//   sycl_backend::usable_gpu_devices()[id].get_info<global_mem_size>()
PipelineDeviceAssignment select_pipeline_devices(int dev_a, int dev_b);

// Test seam — caller supplies the VRAM lookup. Used by parity tests
// to exercise the swap without real heterogeneous hardware.
PipelineDeviceAssignment select_pipeline_devices(
    int dev_a, int dev_b,
    std::function<std::uint64_t(int)> const& vram_for_device);

} // namespace pos2gpu
