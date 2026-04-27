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
// sequence). Multi-core utilization comes from BatchPlotter spawning
// one of these per `cpu` token in --devices, e.g. `--devices cpu,cpu`
// runs two concurrent plots on two cores.
//
// Throws std::runtime_error on plotting failure (caller decides
// whether to continue under continue_on_error).

#pragma once

namespace pos2gpu {

struct BatchEntry;
struct BatchOptions;

void run_one_plot_cpu(BatchEntry const& entry, BatchOptions const& opts);

} // namespace pos2gpu
