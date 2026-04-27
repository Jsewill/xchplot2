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
    double total_wall_seconds = 0.0;
};

// Options controlling batch behavior.
//   verbose         — per-plot progress on stderr
//   device_ids      — explicit list of CUDA device ids to use. When
//                     empty and use_all_devices is false, run on the
//                     currently-bound CUDA device (pre-multi-GPU
//                     behavior: device 0 by default).
//                     With multiple ids the batch is partitioned across
//                     workers — one thread per device, each with its
//                     own GpuBufferPool and producer/consumer channel.
//                     Plots are assigned round-robin (entry i → worker
//                     i % N).
//   use_all_devices — enumerate all visible CUDA devices at runtime and
//                     use them. Overrides device_ids.
//   include_cpu     — append a CPU worker alongside any GPUs already
//                     selected. Set by `--cpu` (orthogonal to --devices)
//                     or by passing `cpu` as a token in --devices. CPU
//                     is encoded as kCpuDeviceId (-2) in device_ids —
//                     see src/gpu/DeviceIds.hpp. Plotting on CPU goes
//                     through pos2-chip's Plotter directly (no CUDA
//                     calls); 1-2 orders of magnitude slower than GPU,
//                     useful for GPU-less hosts or as an extra worker.
struct BatchOptions {
    bool             verbose         = false;
    std::vector<int> device_ids;
    bool             use_all_devices = false;
    bool             include_cpu     = false;
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
