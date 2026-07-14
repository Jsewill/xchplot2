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
// BatchPlotter spawns as many of these as --cpu-workers asks for (`--devices
// cpu,cpu` runs two), and they share nothing: each concurrent plot needs its
// own copy of the Plotter's working set. Measured peak RSS per worker —
// VmHWM, one plot, CPU only:
//
//     k=22    223 568 kB      k=26  3 226 100 kB
//     k=24    927 036 kB      k=28 12 727 020 kB   (12.14 GiB)
//
// so N workers cost N × that, and resolve_batch_devices gates N against host
// RAM before any of them start.
//
// Worth it, but less than it used to be. Aggregate steady-state on a 32-thread
// 5950X, same binary, in-process:
//
//            k=26                        k=28
//   N=1   13.57 s/plot              52.28 s/plot
//   N=2   10.59  (+28%)             43.85  (+19%)
//   N=4    9.63  (+41%)             41.69  (+25%)
//
// The plotter is memory-latency-bound, so concurrent plots interleave each
// other's stalls rather than queueing for a core — that is the whole gain. It
// shrinks as k rises (a 12 GiB working set already saturates memory bandwidth
// on its own, so a second worker contends instead of filling idle time) and it
// shrinks per worker added: at k=28 the 3rd and 4th together buy 5%.
//
// An earlier measurement said +27% / +52% at k=28. That was taken BEFORE the
// serial FSE tail was parallelised (write_plot_file_parallel), and a serial
// tail is exactly the dead time a concurrent plot fills — so fixing it took
// back part of what N>1 was being paid for. The two fixes are not additive.
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
