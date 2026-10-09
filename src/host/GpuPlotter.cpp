// GpuPlotter.cpp — orchestrates per-phase CPU/GPU strategy. For any
// phase not yet implemented on GPU, dispatches to the pos2-chip CPU
// reference (via run_cpu_plotter_to_fragments in PlotFileWriterParallel.cpp).
// We then call write_plot_file_parallel for the FSE compression and
// serialization, regardless of which path produced the fragments.
//
// Deliberately includes NO pos2-chip headers — those all live behind
// PlotFileWriterParallel.{hpp,cpp} so that one .cpp is the sole TU
// pulling soft_aes.hpp into the link.

#include "host/GpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PlotFileWriterParallel.hpp"

#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace pos2gpu {

namespace {

void warn_if_gpu_requested_but_unimplemented(GpuPlotOptions const& o)
{
    auto warn = [&](char const* phase) {
        std::cerr << "[xchplot2] WARNING: " << phase
                  << " GPU path not implemented yet — falling back to CPU.\n";
    };
    if (o.t1 == PhaseStrategy::Gpu) warn("T1");
    if (o.t2 == PhaseStrategy::Gpu) warn("T2");
    if (o.t3 == PhaseStrategy::Gpu) warn("T3");
}

} // namespace

std::string plot_to_file(GpuPlotOptions const& opts, std::string const& output_dir)
{
    if (opts.k < 18 || opts.k > 28 || (opts.k & 1) != 0) {
        throw std::runtime_error("k must be even and in [18, 28]");
    }
    if (opts.strength < 2 || opts.strength > 63) {
        throw std::runtime_error("strength must be in [2, 63]");
    }
    if (opts.testnet)
        throw std::invalid_argument("PoS2 1.0 removes testnet-specific plots; omit --testnet");
    if (!opts.raw && opts.plot_index != 0)
        throw std::invalid_argument("single-plot groups must start at plot index 0; use --raw for group members");

    // Build output filename. Caller may override via opts.out_name (used
    // by integrations). Default includes the group identity and parameters.
    std::string filename;
    if (!opts.out_name.empty()) {
        filename = opts.out_name;
    } else {
        std::string group_id_hex;
        group_id_hex.reserve(64);
        static char const hex[] = "0123456789abcdef";
        for (uint8_t b : opts.group_id) {
            group_id_hex += hex[b >> 4];
            group_id_hex += hex[b & 0xF];
        }
        filename = "plot_" + std::to_string(opts.k)
                 + "_" + std::to_string(opts.strength)
                 + "_" + std::to_string(opts.plot_index)
                 + "_" + std::to_string(opts.meta_group)
                 + "_" + group_id_hex
                 + (opts.raw ? ".plot2" : ".gplot");
    }

    std::filesystem::create_directories(output_dir);
    auto full_path = std::filesystem::path(output_dir) / filename;

    // Memo: caller-supplied bytes if present (real farmable plots), else
    // a 112-byte stub (test plots only — harvester will reject).
    std::vector<uint8_t> memo_bytes;
    if (!opts.memo.empty()) {
        memo_bytes = opts.memo;
    } else {
        memo_bytes.assign(32 + 48 + 32, 0);
    }
    if (memo_bytes.size() > 255) {
        throw std::runtime_error("memo too long (max 255 bytes; PlotFile uses uint8 length)");
    }
    BatchEntry entry;
    entry.k = opts.k;
    entry.strength = opts.strength;
    entry.plot_index = opts.plot_index;
    entry.meta_group = opts.meta_group;
    entry.raw = opts.raw;
    entry.group_id = opts.group_id;
    entry.plot_id = opts.plot_id;
    entry.memo = std::move(memo_bytes);
    entry.out_dir = output_dir;
    entry.out_name = filename;
    validate_batch_entry(entry);

    bool const all_gpu = (opts.t1 == PhaseStrategy::Gpu)
                      && (opts.t2 == PhaseStrategy::Gpu)
                      && (opts.t3 == PhaseStrategy::Gpu);

    if (!all_gpu) {
        warn_if_gpu_requested_but_unimplemented(opts);
    }

    // Either path produces a vector of ProofFragments we pass to the writer
    // as a span. GPU path owns via GpuPipelineResult::t3_fragments_storage;
    // CPU path owns via cpu_fragments below.
    GpuPipelineResult pr;
    std::vector<uint64_t> cpu_fragments;
    std::span<uint64_t const> fragments;
    if (all_gpu) {
        // Run the full pipeline on GPU; hand the sorted T3 fragments to
        // the CPU PlotFile writer for FSE compression and serialization.
        GpuPipelineConfig cfg;
        cfg.plot_id  = opts.plot_id;
        cfg.k        = opts.k;
        cfg.strength = opts.strength;
        cfg.testnet  = opts.testnet;
        cfg.profile  = opts.profile;
        pr = run_gpu_pipeline(cfg);
        fragments = pr.fragments();
        if (opts.verbose) {
            std::cerr << "[xchplot2] T1=" << pr.t1_count
                      << " T2=" << pr.t2_count
                      << " T3=" << pr.t3_count
                      << " (all on GPU)\n";
        }
    } else {
        cpu_fragments = run_cpu_plotter_to_fragments(
            opts.plot_id.data(),
            static_cast<uint8_t>(opts.k),
            static_cast<uint8_t>(opts.strength),
            opts.testnet ? uint8_t{1} : uint8_t{0},
            opts.verbose);
        fragments = std::span<uint64_t const>(cpu_fragments.data(),
                                              cpu_fragments.size());
    }

    write_plot_file_parallel(full_path.string(), fragments, entry);

    return full_path.string();
}

} // namespace pos2gpu
