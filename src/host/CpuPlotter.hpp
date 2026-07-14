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
// NOT single-threaded. pos2-chip fans out to hardware_concurrency()
// internally — RadixSort.hpp:33 and TableConstructorGeneric.hpp:391 call it
// directly, and parallel_for_range defaults to it — with no cap, no pool, and
// a fresh set of threads per parallel region. Measured at k=28 on a 32-thread
// 5950X: 26–30 of 32 cores busy right through table construction. (This
// comment claimed "single-threaded internally" for two revisions. It is why
// the CPU worker looked free to co-schedule alongside a GPU, and it is not.)
//
// The tail is the exception: PlotFile::writeData FSE-compresses the plot in a
// plain serial for-loop (PlotFile.hpp:114), so a k=28 plot ends with ~10 s
// pinned to ONE core — ~19% of its wall. Our own write_plot_file_parallel
// already parallelises exactly that, through a shared pool, and already
// accepts pos2-chip's fragment layout (ProofFragment is uint64_t, and
// PlotData::t3_proof_fragments is a flat vector of them). See
// PlotFileWriterParallel.hpp.
//
// BatchPlotter spawns exactly one of these — include_cpu is a bool, so
// `--devices cpu,cpu` runs ONE worker, not two. Concurrent CPU plots each need
// their own copy of the Plotter's working set: 12.13 GiB peak RSS per plot at
// k=28, so N workers costs N × that.
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
