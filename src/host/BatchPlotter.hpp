// BatchPlotter.hpp — staggered multi-plot pipeline.
//
// One producer thread runs the GPU pipeline back-to-back; one consumer
// thread runs the (already-parallelised) FSE compression + plot file
// write. A bounded queue of depth 1 between them lets GPU compute for
// plot N+1 overlap CPU FSE for plot N.
//
// Steady-state per-plot wall time = max(GPU_compute, CPU_FSE) instead of
// (GPU_compute + CPU_FSE). For k=28 strength=2 on the current build that's
// roughly 3 s vs 7 s — about 2x throughput.

#pragma once

#include "host/GpuPlotter.hpp"
#include "host/MultiGpuPipelineParallel.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pos2gpu {

struct BatchEntry {
    int k = 28;
    int strength = 2;
    int plot_index = 0;
    int meta_group = 0;
    bool testnet = false;
    std::array<uint8_t, 32> plot_id{};
    std::vector<uint8_t> memo;
    std::string out_dir;
    std::string out_name;
};

struct BatchResult {
    size_t plots_written = 0;
    size_t plots_skipped = 0;  // present + skipped via BatchOptions::skip_existing
    size_t plots_failed  = 0;  // raised an exception under BatchOptions::continue_on_error
    double total_wall_seconds = 0.0;
};

// Options controlling batch behavior.
//   verbose           — per-plot progress on stderr
//   skip_existing     — if an output .plot2 already exists (and passes a
//                       lightweight magic/size check), skip the plot
//                       instead of overwriting it
//   continue_on_error — catch per-plot exceptions and log rather than
//                       aborting the batch; plots_failed in the result
//                       counts how many skipped this way
//   device_ids        — explicit list of GPU device ids to use. When empty
//                       and use_all_devices is false, run on a single
//                       device picked by the default SYCL gpu_selector_v
//                       (zero-configuration, pre-multi-GPU behavior).
//                       With multiple ids, the batch is partitioned
//                       across workers — one thread per device, each
//                       with its own GpuBufferPool and producer/consumer
//                       channel. Plots are assigned round-robin
//                       (entry i → worker i % N).
//   use_all_devices   — enumerate all SYCL GPU devices at runtime and
//                       use them. Overrides device_ids. Useful when the
//                       caller doesn't know the host's device count up
//                       front (e.g. `--devices all` on the CLI).
//   include_cpu       — append the CPU as a worker device alongside any
//                       GPUs already selected. Set by `--cpu` (orthogonal
//                       to --devices) or by passing `cpu` as a token in
//                       --devices. CPU is encoded as kCpuDeviceId (-2) in
//                       device_ids — see src/gpu/DeviceIds.hpp. Plotting
//                       on CPU is 1-2 orders of magnitude slower than on
//                       GPU; this is meant for headless CI / GPU-less
//                       hosts / heterogeneous device-list mixing.
//   streaming_tier    — optional manual override for the streaming
//                       pipeline tier (when the GPU pool doesn't fit).
//                       Accepted values: "plain" (~7.24 GB floor at k=28,
//                       ~10-15% faster), "compact" (~5.33 GB floor, fits
//                       on tight 8 GB cards), "minimal" (~3.76 GB floor,
//                       4 GB cards), "tiny" (~1.5 GB floor target, sub-
//                       4 GB cards — scaffolding only as of this commit;
//                       runs the minimal-tier path until tighter slicing
//                       lands). Empty string = auto (pick the largest
//                       tier that fits free VRAM). Equivalent to
//                       XCHPLOT2_STREAMING_TIER env var but settable via
//                       --tier on the CLI; the struct field takes
//                       precedence over the env var.
//   shard_plot        — opt in to single-plot multi-GPU mode. Default
//                       (false) keeps the existing work-queue dispatch:
//                       N workers, one plot each, round-robin. With
//                       shard_plot=true, the workers form a "team" and
//                       run plots one at a time, each owning a shard
//                       of every plot. Phase 1 scaffold lands the
//                       option but only supports N=1 (no-op fall-through
//                       to single-GPU); N > 1 throws a clear error
//                       until later phases implement the real sharding.
//                       See docs/multi-gpu-single-plot-*.md for the
//                       design.
//   shard_strategy    — partition strategy when shard_plot is on.
//                       Reserved for the multi-GPU work; "bucket" is
//                       the planned default (output-bucket partition
//                       with distributed radix sort). "section_l" is
//                       the alternative (input-section partition with
//                       gather-sort-scatter). Ignored when N=1.
// Phase 2.4 batch-strategy picker. Auto = pick at runtime based on
// device VRAM and k. WorkQueue = N independent plotters round-robin
// (default for equal-VRAM PCIe-only rigs where each card fits the
// pool). PipelinePlot = N-stage split (--pipeline-plot semantics).
// ShardPlot = single-plot multi-GPU (--shard-plot semantics).
enum class BatchStrategy {
    Auto,
    WorkQueue,
    PipelinePlot,
    ShardPlot,
};

struct BatchOptions {
    bool verbose           = false;
    bool skip_existing     = false;
    bool continue_on_error = false;
    std::vector<int> device_ids;
    bool use_all_devices   = false;
    bool include_cpu       = false;
    std::string streaming_tier;
    // Phase 2.4: explicit strategy (Auto = picker decides). Legacy
    // shard_plot / pipeline_plot bool fields below are still honoured
    // for backward compatibility and act as explicit overrides if
    // strategy == Auto.
    BatchStrategy strategy = BatchStrategy::Auto;
    bool shard_plot        = false;
    std::string shard_strategy = "bucket";
    // Pipeline-parallel mode (Phase 2.1d): split each plot at the T2-
    // sort boundary across exactly two devices. device_ids[0] runs
    // Xs / T1 / T2; device_ids[1] runs T3 / Frag. Plots are pipelined
    // (depth=2) so plot N's stage 2 overlaps plot N+1's stage 1. On
    // PCIe-only hosts the two stages contend for host bandwidth and
    // throughput is below work-queue; the value is correctness on
    // heterogeneous rigs and per-plot latency on NVLink-equipped
    // hosts. Mutually exclusive with shard_plot.
    bool pipeline_plot     = false;
    // Per-stage streaming tier for the pipeline-plot path. Empty
    // defaults to Tiny per stage. When non-empty, size must match
    // device_ids.size() (currently 2 or 3). Tiny is the safe default;
    // Minimal trades device VRAM for fewer PCIe round-trips per
    // stage. Heterogeneous-rig benchmarks can pick Minimal on the
    // larger card and Tiny on the smaller. Ignored when pipeline_plot
    // is false.
    std::vector<PipelineStageTier> pipeline_tiers;
    // When true and shard_plot is on, the distributed sorts route data
    // via direct device-to-device memcpy (Peer transport). On NVLink
    // hosts this stays on the fabric; on PCIe-only hosts the SYCL/CUDA
    // backend resolves D2D as an implicit single host bounce, which is
    // still ~one fewer copy than the explicit two-bounce HostBounce
    // path. Equivalent on a single-GPU dev box (peer-on-same-context =
    // ordinary device memcpy).
    //
    // Default flipped to true after k=28 measurements on 2× RTX 4000
    // Ada showed Peer at ~9.2 s/plot vs HostBounce at ~14.0 s/plot
    // (PCIe-only). Set to false (CLI: `--host-bounce`) on tight-VRAM
    // (<10 GB) cards at large k where the per-source staging cost
    // matters: Peer allocates source-side staging sized to the source
    // shard's full input count (~1.6 GB/shard for u32_u32 at k=28; up
    // to ~3.2 GB/shard for u32_u64+u32 in T2's sort).
    bool prefer_peer_copy  = true;
};

// Parse a manifest file in the format described in tools/xchplot2/main.cpp
// (tab-separated, one plot per line). Throws std::runtime_error on bad input.
std::vector<BatchEntry> parse_manifest(std::string const& path);

// Run the staggered pipeline. Producer/consumer share a queue of depth 1.
// The first plot pays the full GPU+FSE cost; subsequent plots overlap.
BatchResult run_batch(std::vector<BatchEntry> const& entries,
                      BatchOptions const& opts);

// Legacy bool-verbose shim kept for source-compat with older callsites.
inline BatchResult run_batch(std::vector<BatchEntry> const& entries,
                             bool verbose = false)
{
    BatchOptions opts;
    opts.verbose = verbose;
    return run_batch(entries, opts);
}

// Phase 2.4 auto-strategy picker. Heuristic:
//   N == 1                              → WorkQueue
//   smallest device VRAM < tiny peak    → PipelinePlot (work-queue won't fit)
//   else                                → WorkQueue
// shard_plot is never auto-selected (niche; remains explicit opt-in).
//
// `reason_out`, if non-null, gets a short human-readable string for
// `[strategy] auto-picked X because Y` verbose printout.
//
// `vram_for_device` is the injectable VRAM lookup (test seam). The
// default-form overload uses sycl_backend's usable_gpu_devices.
struct StrategyPickInputs {
    std::vector<int> device_ids;
    int              k;
};
BatchStrategy select_strategy(
    StrategyPickInputs const&                inputs,
    std::function<std::uint64_t(int)> const& vram_for_device,
    std::string*                             reason_out = nullptr);

BatchStrategy select_strategy(
    StrategyPickInputs const& inputs,
    std::string*              reason_out = nullptr);

} // namespace pos2gpu
