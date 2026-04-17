// GpuPlotter.hpp — host-side orchestration. Wraps the GPU kernels (where
// implemented) and falls back to pos2-chip's CPU plotter for phases that
// haven't been ported yet. Per-phase strategy flag lets the user mix CPU
// and GPU during bring-up.

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace pos2gpu {

enum class PhaseStrategy : uint8_t {
    Cpu, // use pos2-chip's reference implementation
    Gpu, // use GPU kernels in this repo
};

struct GpuPlotOptions {
    int k = 28;
    int strength = 2;
    int plot_index = 0;
    int meta_group = 0;
    bool testnet = false;
    bool verbose = false;
    std::array<uint8_t, 32> plot_id{};

    PhaseStrategy t1 = PhaseStrategy::Cpu;
    PhaseStrategy t2 = PhaseStrategy::Cpu;
    PhaseStrategy t3 = PhaseStrategy::Cpu;
};

// Run the full pipeline and write a .plot2 file. Returns the absolute
// output path. Throws std::runtime_error on failure.
std::string plot_to_file(
    GpuPlotOptions const& opts,
    std::string const& output_dir);

} // namespace pos2gpu
