// Bounded production proof sampling, independent of the reference solver.
#pragma once
#include "host/PlotFileWriterParallel.hpp"
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace pos2gpu::verification_detail
{
template<class Random, class Lookup, class Solve>
VerifyResult sample(BoundedVerifyBudget const& budget, Random& random, Lookup&& lookup,
    Solve&& solve, std::function<void()> const& check_cancelled)
{
    if (!budget.max_challenges || !budget.full_proofs || budget.full_proofs > budget.max_challenges)
        throw std::invalid_argument("invalid bounded plot verification challenge or solve budget");
    auto check = [&] { if (check_cancelled) check_cancelled(); };
    auto add = [](size_t left, size_t right)
    {
        if (right > std::numeric_limits<size_t>::max() - left)
            throw std::runtime_error("bounded plot verification sample count overflow");
        return left + right;
    };
    VerifyResult result;
    for (; result.trials < budget.max_challenges;)
    {
        check();
        std::array<uint8_t, 32> challenge{};
        for (size_t word = 0; word < challenge.size(); word += sizeof(uint64_t))
        {
            auto const value = std::uniform_int_distribution<uint64_t>{}(random);
            std::memcpy(challenge.data() + word, &value, sizeof(value));
        }
        auto&& qualities = lookup(challenge);
        check(); ++result.trials;
        size_t count = 0;
        for (auto const& member : qualities) count = add(count, member.quality_chains.size());
        result.proofs_found = add(result.proofs_found, count);
        if (!count) continue;
        ++result.challenges_with_proof;
        // The reader already materializes its bounded-by-worker result. Select
        // an index over that result without copying or retaining chain lists.
        size_t selected = std::uniform_int_distribution<size_t>(0, count - 1)(random);
        for (auto const& member : qualities)
        {
            if (selected >= member.quality_chains.size()) { selected -= member.quality_chains.size(); continue; }
            check(); solve(challenge, member, member.quality_chains[selected]);
            ++result.full_proofs_validated; break;
        }
        check();
        if (result.full_proofs_validated == budget.full_proofs) return result;
    }
    throw std::runtime_error("bounded plot verification exhausted its challenge budget without the required positive full proofs");
}
} // namespace pos2gpu::verification_detail
