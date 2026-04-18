// GpuPlotter.cpp — orchestrates per-phase CPU/GPU strategy. For any
// phase not yet implemented on GPU, dispatches to the pos2-chip CPU
// reference. The CPU plotter writes the final plot file via PlotFile,
// so we always end on the CPU regardless.

#include "host/GpuPlotter.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PlotFileWriterParallel.hpp"

// pos2-chip headers (relative include surface set in CMake).
#include "plot/Plotter.hpp"
#include "plot/PlotFile.hpp"
#include "pos/ProofParams.hpp"

namespace pos2gpu {
// Host-callable, defined in src/gpu/AesGpu.cu. We forward-declare here so
// this TU doesn't pull in <cuda_runtime.h>.
void initialize_aes_tables();
}

#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>

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
    if (opts.k < 18 || opts.k > 32 || (opts.k & 1) != 0) {
        throw std::runtime_error("k must be even and in [18, 32]");
    }
    if (opts.strength < 2 || opts.strength > 63) {
        throw std::runtime_error("strength must be in [2, 63]");
    }

    initialize_aes_tables();

    bool const all_gpu = (opts.t1 == PhaseStrategy::Gpu)
                      && (opts.t2 == PhaseStrategy::Gpu)
                      && (opts.t3 == PhaseStrategy::Gpu);

    if (!all_gpu) {
        warn_if_gpu_requested_but_unimplemented(opts);
    }

    ProofParams params(opts.plot_id.data(),
                       static_cast<uint8_t>(opts.k),
                       static_cast<uint8_t>(opts.strength),
                       opts.testnet ? uint8_t{1} : uint8_t{0});

    // Either path produces a vector of ProofFragments we pass to the writer
    // as a span. GPU path owns via GpuPipelineResult::t3_fragments_storage;
    // CPU path owns via PlotData::t3_proof_fragments.
    PlotData plot;
    GpuPipelineResult pr;
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
        Plotter::Options plotter_opts{};
        plotter_opts.validate = false;
        plotter_opts.verbose = opts.verbose;
        Plotter plotter(params);
        plot = plotter.run(plotter_opts);
        fragments = std::span<uint64_t const>(plot.t3_proof_fragments.data(),
                                              plot.t3_proof_fragments.size());
    }

    // Build output filename. Caller may override via opts.out_name (used
    // by chia plots create --gpu to match its naming convention). Default
    // is the legacy xchplot2 test scheme.
    std::string filename;
    if (!opts.out_name.empty()) {
        filename = opts.out_name;
    } else {
        std::string plot_id_hex;
        plot_id_hex.reserve(64);
        static char const hex[] = "0123456789abcdef";
        for (uint8_t b : opts.plot_id) {
            plot_id_hex += hex[b >> 4];
            plot_id_hex += hex[b & 0xF];
        }
        filename = "plot_" + std::to_string(opts.k)
                 + "_" + std::to_string(opts.strength)
                 + "_" + std::to_string(opts.plot_index)
                 + "_" + std::to_string(opts.meta_group)
                 + (opts.testnet ? "_testnet" : "")
                 + "_" + plot_id_hex
                 + ".plot2";
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
    write_plot_file_parallel(full_path.string(),
                             fragments,
                             params,
                             static_cast<uint16_t>(opts.plot_index),
                             static_cast<uint8_t>(opts.meta_group),
                             std::span<uint8_t const>(memo_bytes.data(), memo_bytes.size()));

    return full_path.string();
}

} // namespace pos2gpu
