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
//                       on tight 8 GB cards). Empty string = auto (the
//                       pre-existing behavior: pick plain if it fits,
//                       else compact). Equivalent to XCHPLOT2_STREAMING_TIER
//                       env var but settable via --tier on the CLI; the
//                       struct field takes precedence over the env var.
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
struct BatchOptions {
    bool verbose           = false;
    bool skip_existing     = false;
    bool continue_on_error = false;
    std::vector<int> device_ids;
    bool use_all_devices   = false;
    bool include_cpu       = false;
    std::string streaming_tier;
    bool shard_plot        = false;
    std::string shard_strategy = "bucket";
    // Phase 2.4b: when true and shard_plot is on, the distributed sorts
    // route data via direct device-to-device memcpy (Peer transport)
    // instead of the default host-pinned bounce. Equivalent on a
    // single-GPU dev box (peer-on-same-context = ordinary device
    // memcpy); the win lands on real multi-GPU hosts where the SYCL
    // backend can route through NVLink/peer-PCIe.
    bool prefer_peer_copy  = false;
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

} // namespace pos2gpu
