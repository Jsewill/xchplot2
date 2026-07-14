// CpuPlotter.hpp — single-plot CPU pipeline using pos2-chip's Plotter
// directly (no SYCL / no GPU code path involved).
//
// Format-compatible with the GPU output: same plot_id derivation, same
// .plot2 file layout, byte-identical proofs. pos2-chip is the upstream
// PoS2 reference implementation, already in our build tree via
// FetchContent (third_party/pos2-chip), so we link its CPU plotter
// directly rather than routing SYCL kernels through AdaptiveCpp's
// OpenMP backend.
//
// Single-threaded internally (the Plotter constructs T1/T2/T3 in
// sequence), and BatchPlotter spawns exactly one of it — include_cpu is a
// bool, so `--devices cpu,cpu` runs ONE worker, not two. (This comment
// used to promise one per token; nothing ever counted them.) Concurrent
// CPU plots would need N copies of this Plotter's working set in RAM.
//
// Throws std::runtime_error on plotting failure (caller decides
// whether to continue under continue_on_error).

#pragma once

#include <cstdint>

namespace pos2gpu {

struct BatchEntry;
struct BatchOptions;

// Returns on-disk .plot2 size in bytes (via file_size post-write).
std::uint64_t run_one_plot_cpu(BatchEntry const& entry, BatchOptions const& opts);

} // namespace pos2gpu
