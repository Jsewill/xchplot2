// GpuPipeline.hpp — host-only API for running the full Xs → T1 → T2 → T3
// pipeline on the GPU. Returns the sorted ProofFragment stream that
// PlotFile::writeData expects.
//
// Implementation in src/host/GpuPipeline.cu (CUDA TU). This header is
// intentionally CUDA-free so plain .cpp consumers (GpuPlotter.cpp,
// gpu_plotter/main.cpp) can include it without dragging in nvcc.

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

struct GpuPipelineConfig {
    std::array<uint8_t, 32> plot_id{};
    int k = 28;
    int strength = 2;
    bool testnet = false;
};

struct GpuPipelineResult {
    std::vector<uint64_t> t3_fragments; // sorted by proof_fragment, low 2k bits
    // Counts per phase (for diagnostics)
    uint64_t t1_count = 0;
    uint64_t t2_count = 0;
    uint64_t t3_count = 0;
};

// Runs the full GPU plotter pipeline. Throws std::runtime_error on CUDA
// failure. Caller must have called pos2gpu::initialize_aes_tables() once.
GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg);

} // namespace pos2gpu
