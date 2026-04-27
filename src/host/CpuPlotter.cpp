// CpuPlotter.cpp — wraps pos2-chip's Plotter + PlotFile::writeData.
//
// Isolated to one TU because pos2-chip's Plotter.hpp pulls in the full
// table-construction template stack (Table1/2/3Constructor + RadixSort
// + ChunkCompressor + ...). Including that header anywhere else in the
// build would balloon compile times for no benefit — only this TU
// actually invokes Plotter::run().

#include "host/CpuPlotter.hpp"
#include "host/BatchPlotter.hpp"  // for BatchEntry / BatchOptions

// pos2-chip headers — header-only, no separate compilation needed.
// pos2_chip_headers (PUBLIC dep of pos2_gpu_host) provides the
// include path + fse link.
#include "plot/Plotter.hpp"
#include "plot/PlotFile.hpp"
#include "pos/ProofParams.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

namespace pos2gpu {

void run_one_plot_cpu(BatchEntry const& entry, BatchOptions const& opts)
{
    // Build pos2-chip's ProofParams from BatchEntry's existing fields.
    // ProofParams is in the global namespace (pos2-chip doesn't wrap
    // its public types in a namespace).
    ::ProofParams params(entry.plot_id.data(),
                         static_cast<uint8_t>(entry.k),
                         static_cast<uint8_t>(entry.strength),
                         static_cast<uint8_t>(entry.testnet ? 1 : 0));

    ::Plotter::Options pl_opts;
    pl_opts.verbose = opts.verbose;

    ::Plotter plotter(params);
    ::PlotData plot = plotter.run(pl_opts);

    // pos2-chip's PlotFile::writeData accepts the memo as a span and
    // writes a 1-byte length prefix on disk, so any size in [0, 255]
    // is valid. keygen-rs emits two layouts:
    //   - pool-PH mode: 32-byte pool_ph + 48-byte farmer_pk + 32-byte
    //                   master_sk = 112 bytes
    //   - pool-PK mode: 48-byte pool_pk + 48-byte farmer_pk + 32-byte
    //                   master_sk = 128 bytes
    // BatchEntry.memo already holds the bytes in the on-disk layout, so
    // pass them through as a span. The previous strict 112-byte check
    // rejected pool-PK plots produced via `xchplot2 plot -p ...`.
    if (entry.memo.size() > 255) {
        throw std::runtime_error(
            "CpuPlotter: memo size " + std::to_string(entry.memo.size()) +
            " exceeds the 255-byte on-disk limit");
    }

    std::filesystem::path const out_path =
        std::filesystem::path(entry.out_dir) / entry.out_name;

    ::PlotFile::writeData(out_path.string(),
                          plot,
                          params,
                          static_cast<uint16_t>(entry.plot_index),
                          static_cast<uint8_t>(entry.meta_group),
                          std::span<uint8_t const>(entry.memo.data(),
                                                   entry.memo.size()));
}

} // namespace pos2gpu
