#undef NDEBUG
#include "solve/Solver.hpp"
#include <cassert>
#include <iostream>
#include <random>

int main(int argc, char** argv)
{
    unsigned const k = argc > 1 ? unsigned(std::stoul(argv[1])) : 18;
    assert(k == 18 || k == 22 || k == 28);
    std::array<uint8_t, 32> id{};
    auto const params = PlotProofParams::create_raw(PlotId(id), uint8_t(k), 2);
    Solver solver(params);
    ProofCore core(params);
    std::vector<uint32_t> bits((1u << k) / 32, k == 18 ? ~uint32_t{0} : 0);
    if (k != 18) for (uint32_t x : {0u, 1u, 65535u, (1u << k) - 1})
    { auto const hash = core.hashing.g(x); bits[hash >> 5] |= uint32_t(1) << (hash & 31); }
    std::vector<uint32_t> xs, hashes;

    // Every candidate passes, deliberately exceeding the old per-thread
    // estimate. Growing the buffers must preserve both values and ordering.
    solver.filterX2Candidates(bits, 1, xs, hashes);
    assert((k != 18 || xs.size() == (1u << k)) && hashes.size() == xs.size());
    assert(!xs.empty() && std::is_sorted(xs.begin(), xs.end()));
    for (size_t i = 0; i < xs.size(); ++i) {
        assert((k != 18 || xs[i] == i) && hashes[i] == core.hashing.g(xs[i]));
    }
    for (unsigned threads : {1u, 2u}) {
        Solver bounded(params, threads);
        std::vector<uint32_t> bounded_xs, bounded_hashes;
        bounded.filterX2Candidates(bits, 1, bounded_xs, bounded_hashes);
        assert(bounded_xs == xs && bounded_hashes == hashes);
    }

    std::fill(bits.begin(), bits.end(), 0);
    solver.filterX2Candidates(bits, 1, xs, hashes);
    assert(xs.empty() && hashes.empty());

    std::mt19937 random(0);
    std::vector<uint32_t> keys(65537), values(keys.size());
    std::vector<std::pair<uint32_t, uint32_t>> expected;
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = random() & ((1u << k) - 1); values[i] = uint32_t(i);
        expected.emplace_back(keys[i], values[i]);
    }
    std::stable_sort(expected.begin(), expected.end(), [](auto const& a, auto const& b) { return a.first < b.first; });
    auto sorted = keys; std::sort(sorted.begin(), sorted.end());
    for (unsigned threads : {std::thread::hardware_concurrency(), 1u, 2u}) {
        ParallelRadixSort radix(threads);
        auto actual = keys; std::vector<uint32_t> buffer(keys.size());
        radix.sort(actual, buffer); assert(actual == sorted);
        actual = keys; auto actual_values = values; std::vector<uint32_t> value_buffer(keys.size());
        radix.sortByKey(actual, actual_values, buffer, value_buffer, int(k));
        for (size_t i = 0; i < actual.size(); ++i) assert(std::pair(actual[i], actual_values[i]) == expected[i]);
    }
    std::cout << "solver filter and radix default/capped parity passed at k=" << k << '\n';
}
