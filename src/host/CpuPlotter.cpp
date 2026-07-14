// CpuPlotter.cpp — runs pos2-chip's CPU Plotter, then writes the plot through
// OUR parallel writer rather than pos2-chip's.
//
// Why not pos2-chip's PlotFile::writeData: it FSE-compresses the whole plot in
// a plain serial for-loop (pos2-chip plot/PlotFile.hpp:114 — one
// ChunkCompressor::compressProofFragments call per chunk, no threads anywhere).
// Sampling /proc/<pid>/stat through a k=28 plot on a 32-thread 5950X shows the
// process running 26-30 cores flat through table construction and then
// collapsing to EXACTLY ONE thread for the last ~10 s — 19% of the plot's wall,
// single-threaded, invisible in the s/plot number.
//
// write_plot_file_parallel compresses those same chunks across the shared
// WriterThreadPool. It is a documented drop-in for PlotFile::writeData and its
// output is byte-identical (see PlotFileWriterParallel.hpp) — the parity test
// in tools/parity covers it. Two things fall out for free:
//
//   * The CPU worker's FSE now queues on the SAME pool as every GPU worker's,
//     instead of running as an extra uncoordinated thread. The process-wide
//     compression thread count stays at hardware_concurrency() no matter how
//     many workers are plotting.
//   * The writer owns durability end-to-end — .partial + scope guard + fsync +
//     PARENT-DIRECTORY fsync + atomic rename. Writing straight to the final
//     name (which is what this TU used to do) leaves a truncated file with a
//     valid header behind on a hard kill / ENOSPC / crash, and --skip-existing
//     would then treat it as a complete plot.
//
// pos2-chip's plot/* and pos/* headers are deliberately NOT included here: the
// non-inline soft_aesenc / soft_aesdec in pos/aes/soft_aes.hpp cause
// multiple-definition link errors if more than one TU pulls them in, so
// PlotFileWriterParallel.cpp is the single TU that does. Going through
// run_cpu_plotter_to_fragments() keeps that invariant (and keeps the whole
// Table1/2/3Constructor + RadixSort template stack out of this TU's build).
//
// Throws std::runtime_error on plotting failure (caller decides whether to
// continue under continue_on_error).

#include "host/CpuPlotter.hpp"
#include "host/BatchPlotter.hpp"           // BatchEntry / BatchOptions
#include "host/PlotFileWriterParallel.hpp" // run_cpu_plotter_to_fragments + writer

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

std::uint64_t run_one_plot_cpu(BatchEntry const& entry, BatchOptions const& opts)
{
    // pos2-chip's PlotFile writes the memo behind a 1-byte length prefix, so
    // any size in [0, 255] is on-disk valid. keygen-rs emits two layouts:
    //   - pool-PH mode: 32-byte pool_ph + 48-byte farmer_pk + 32-byte master_sk
    //                   = 112 bytes
    //   - pool-PK mode: 48-byte pool_pk + 48-byte farmer_pk + 32-byte master_sk
    //                   = 128 bytes
    // BatchEntry.memo already holds the bytes in the on-disk layout, so it goes
    // through as a span. (A strict 112-byte check here once rejected pool-PK
    // plots produced via `xchplot2 plot -p ...`.)
    if (entry.memo.size() > 255) {
        throw std::runtime_error(
            "CpuPlotter: memo size " + std::to_string(entry.memo.size()) +
            " exceeds the 255-byte on-disk limit");
    }

    std::uint8_t const k        = static_cast<std::uint8_t>(entry.k);
    std::uint8_t const strength = static_cast<std::uint8_t>(entry.strength);
    std::uint8_t const testnet  = entry.testnet ? std::uint8_t{1} : std::uint8_t{0};

    // Table construction. Internally this fans out to hardware_concurrency()
    // threads (see CpuPlotter.hpp) — it is the parallel 81% of the plot.
    std::vector<std::uint64_t> const frags = run_cpu_plotter_to_fragments(
        entry.plot_id.data(), k, strength, testnet, opts.verbose);

    std::filesystem::path const out_path =
        std::filesystem::path(entry.out_dir) / entry.out_name;

    // ProofFragment is uint64_t (pos2-chip pos/ProofFragment.hpp) and PlotData
    // holds a flat vector of them, so the span is a view, not a copy.
    //
    // thread_count 0 = one task per pool worker. The tasks queue on the shared
    // WriterThreadPool; they do not spawn threads of their own.
    return write_plot_file_parallel(
        out_path.string(),
        std::span<std::uint64_t const>(frags.data(), frags.size()),
        entry.plot_id.data(),
        k,
        strength,
        testnet,
        static_cast<std::uint16_t>(entry.plot_index),
        static_cast<std::uint8_t>(entry.meta_group),
        std::span<std::uint8_t const>(entry.memo.data(), entry.memo.size()),
        /*thread_count=*/0);
}

} // namespace pos2gpu
