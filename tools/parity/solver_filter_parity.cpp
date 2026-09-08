#undef NDEBUG
#include "solve/Solver.hpp"
#include <cassert>

int main()
{
    std::array<uint8_t, 32> id{};
    ProofParams params(id.data(), 18, 2, 0);
    Solver solver(params);
    ProofCore core(params);
    std::vector<uint32_t> bits((1u << 18) / 32, ~uint32_t{0});
    std::vector<uint32_t> xs, hashes;

    // Every candidate passes, deliberately exceeding the old per-thread
    // estimate. Growing the buffers must preserve both values and ordering.
    solver.filterX2Candidates(bits, 1, xs, hashes);
    assert(xs.size() == (1u << 18) && hashes.size() == xs.size());
    for (uint32_t x = 0; x < xs.size(); ++x) {
        assert(xs[x] == x);
        assert(hashes[x] == core.hashing.g(x));
    }

    std::fill(bits.begin(), bits.end(), 0);
    solver.filterX2Candidates(bits, 1, xs, hashes);
    assert(xs.empty() && hashes.empty());
}
