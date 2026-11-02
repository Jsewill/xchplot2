// Compiled with the ORIGINAL upstream include root, outside the worker overlay.
#include "plot/Plotter.hpp"
#include <array>
#include <vector>

std::vector<uint64_t> worker_original_reference(uint8_t k, uint8_t strength,
    std::array<uint8_t, 32> const& id)
{
    Plotter::Options options{};
    options.validate = false; options.verbose = false;
    return Plotter(PlotProofParams::create_raw(PlotId(id), k, strength)).run(options).t3_proof_fragments;
}
