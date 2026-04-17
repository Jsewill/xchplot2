// GpuPlotter.cpp — orchestrates per-phase CPU/GPU strategy. For any
// phase not yet implemented on GPU, dispatches to the pos2-chip CPU
// reference. The CPU plotter writes the final plot file via PlotFile,
// so we always end on the CPU regardless.

#include "host/GpuPlotter.hpp"

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
        std::cerr << "[gpu_plotter] WARNING: " << phase
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
    warn_if_gpu_requested_but_unimplemented(opts);

    ProofParams params(opts.plot_id.data(),
                       static_cast<uint8_t>(opts.k),
                       static_cast<uint8_t>(opts.strength),
                       opts.testnet ? uint8_t{1} : uint8_t{0});

    Plotter::Options plotter_opts{};
    plotter_opts.validate = false;
    plotter_opts.verbose = opts.verbose;

    Plotter plotter(params);
    PlotData plot = plotter.run(plotter_opts);

    // Build output filename matching pos2-chip's plotter_main.cpp scheme.
    std::string plot_id_hex;
    plot_id_hex.reserve(64);
    static char const hex[] = "0123456789abcdef";
    for (uint8_t b : opts.plot_id) {
        plot_id_hex += hex[b >> 4];
        plot_id_hex += hex[b & 0xF];
    }
    std::string filename = "plot_" + std::to_string(opts.k)
                         + "_" + std::to_string(opts.strength)
                         + "_" + std::to_string(opts.plot_index)
                         + "_" + std::to_string(opts.meta_group)
                         + (opts.testnet ? "_testnet" : "")
                         + "_" + plot_id_hex
                         + ".plot2";

    std::filesystem::create_directories(output_dir);
    auto full_path = std::filesystem::path(output_dir) / filename;

    std::array<uint8_t, 32 + 48 + 32> stub_memo{}; // gpu_plotter is a test tool;
                                                   // real keys come via chia plots create
    PlotFile::writeData(full_path.string(),
                        plot,
                        plotter.getProofParams(),
                        static_cast<uint16_t>(opts.plot_index),
                        static_cast<uint8_t>(opts.meta_group),
                        std::span<uint8_t const>(stub_memo.data(), stub_memo.size()));

    return full_path.string();
}

} // namespace pos2gpu
