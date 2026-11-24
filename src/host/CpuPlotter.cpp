// Standalone and batch CPU members use the shared host generator and writer.
#include "host/CpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "host/Cancel.hpp"
#include "host/CpuPlotMemory.hpp"
#include "host/NumaTopology.hpp"
#include "host/PlotFileWriterParallel.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace pos2gpu {

std::uint64_t run_one_plot_cpu(BatchEntry const& entry, BatchOptions const& opts)
{
    validate_batch_entry(entry);
    if (entry.memo.size() > 255) {
        throw std::runtime_error(
            "CpuPlotter: memo size " + std::to_string(entry.memo.size()) +
            " exceeds the 255-byte on-disk limit");
    }

    if (entry.testnet) throw std::invalid_argument("PoS2 1.0 removes testnet-specific plots");
    const auto cpus = current_thread_cpus();
    const unsigned threads = cpus.empty() ? std::max(1u, std::thread::hardware_concurrency()) : unsigned(cpus.size());
    const cpu::Layout layout(entry.k, threads);
    if (opts.verbose)
        std::fprintf(stderr, "[batch:cpu] generating k=%d with %u CPU lanes\n", int(entry.k), layout.lanes);
    auto fragments = native_cpu_fragments(entry.plot_id.data(), uint8_t(entry.k), uint8_t(entry.strength),
        layout.lanes, layout.peak_bytes(), [] {
            if (cancel_requested()) throw std::runtime_error("CPU plotting cancelled");
        });
    const auto path = std::filesystem::path(entry.out_dir) / entry.out_name;
    return write_plot_file_parallel(path.string(), fragments, entry, layout.lanes);
}

} // namespace pos2gpu
