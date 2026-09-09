// Test-only consumer of PR #118. Inputs are fixtures made by check.py.
#undef NDEBUG
#include "plot/PlotFile.hpp"
#include "plot/Plotter.hpp"
#include "pos/ProofValidator.hpp"
#include "prove/Prover.hpp"
#include "solve/Solver.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
try {
    if (argc != 3) {
        std::cerr << "Usage: pos2_pr118_check FIXTURE.raw EXPECTED_PLOT_ID\n";
        return 1;
    }
    PlotFile raw(argv[1]);
    auto const header = raw.getHeader();
    auto const& group_params = header.params;
    auto const params = group_params.get_plot_params_for_index(header.index);
    assert(params.get_k() == 18);
    // Expected ID comes from Python hashlib, independently of upstream's hash.
    assert(params.get_plot_id().to_string() == argv[2]);

    auto const plot = ChunkedProofFragments::convertToPlotData(raw.readAllChunkedData().data);
    Plotter::Options options{};
    options.validate = false;
    options.verbose = false;
    auto const reference = Plotter(params).run(options);
    assert(!plot.t3_proof_fragments.empty());
    assert(plot.t3_proof_fragments == reference.t3_proof_fragments);

    // ponytail: upstream's writer only emits single-plot test groups; use the
    // production grouping tool when its API and format are approved.
    std::string const grouped = std::string(argv[1]) + ".gplot";
    std::array<uint8_t, 4> const memo{1, 2, 3, 4};
    PlotGroupFile::writeData(grouped, plot, group_params,
        PlotGroupFile::PROOFS_PER_CHUNK_BITS, memo);
    GroupProver prover(grouped);
    auto& group = prover.getPlotGroup();
    auto const& info = group.getInfo();
    assert(info.group_size == 1 && info.group_id == group_params.get_plot_group_id());
    assert(info.k == params.get_k() && info.strength == params.get_strength());
    assert(info.meta_group == group_params.get_meta_group());
    assert(group.readMemo() == std::vector<uint8_t>(memo.begin(), memo.end()));
    auto const roundtrip = group.readProofsInRange({0, (uint64_t{1} << (2 * info.k)) - 1});
    // PR #118 deliberately removes duplicate fragments during group encoding.
    // The raw CPU comparison above remains exact, including multiplicities.
    auto unique_fragments = plot.t3_proof_fragments;
    unique_fragments.erase(std::unique(unique_fragments.begin(), unique_fragments.end()),
        unique_fragments.end());
    assert(roundtrip.size() == 1 && roundtrip[0] == unique_fragments);

    Solver solver(params);
    ProofFragmentCodec codec(params);
    ProofValidator validator(group_params, header.index);
    size_t validated = 0;
    for (uint8_t trial = 0; trial < 32; ++trial) {
        std::array<uint8_t, 32> challenge{};
        challenge.back() = trial;
        for (auto const& qualities : prover.prove(challenge, header.index)) {
            assert(qualities.plot_index == header.index);
            for (auto const& quality : qualities.quality_chains) {
                std::array<uint32_t, TOTAL_XS_IN_PROOF / 2> x_bits{};
                size_t offset = 0;
                for (auto fragment : quality.chain_links)
                    for (auto x : codec.get_x_bits_from_proof_fragment(fragment))
                        x_bits.at(offset++) = x;
                assert(offset == x_bits.size());
                bool valid = false;
                for (auto const& proof : solver.solve(x_bits)) {
                    auto const chain = validator.validate_full_proof(proof, challenge);
                    if (chain && *chain == quality.chain_links) { valid = true; break; }
                }
                assert(valid);
                ++validated;
            }
        }
    }
    assert(validated > 0);
    std::cout << "PASS index=" << header.index
              << " meta_group=" << group_params.get_meta_group()
              << " strength=" << int(params.get_strength())
              << ": " << plot.t3_proof_fragments.size() << " matching fragments, "
              << validated << " full group proofs\n";
    return 0;
} catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
