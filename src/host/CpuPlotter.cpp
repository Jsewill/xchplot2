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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
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

    // pos2-chip's PlotFile::writeData expects the memo as a fixed
    // 112-byte array (32-byte sk_hash + 48-byte farmer_pk + 32-byte
    // pool_ph). xchplot2's BatchEntry stores the memo as
    // std::vector<uint8_t> already in the same v2-format layout —
    // copy into the expected fixed-size array.
    constexpr size_t kMemoSize = 32 + 48 + 32;
    if (entry.memo.size() != kMemoSize) {
        throw std::runtime_error(
            "CpuPlotter: memo size mismatch (got " +
            std::to_string(entry.memo.size()) + " bytes, expected " +
            std::to_string(kMemoSize) + ")");
    }
    std::array<uint8_t, kMemoSize> memo_arr{};
    std::copy(entry.memo.begin(), entry.memo.end(), memo_arr.begin());

    std::filesystem::path const out_path =
        std::filesystem::path(entry.out_dir) / entry.out_name;

    ::PlotFile::writeData(out_path.string(),
                          plot,
                          params,
                          static_cast<uint16_t>(entry.plot_index),
                          static_cast<uint8_t>(entry.meta_group),
                          memo_arr);
}

} // namespace pos2gpu
