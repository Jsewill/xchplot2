#include "host/BoundedProofSampling.hpp"
#include <iostream>
#include <set>

namespace
{
void check(bool value, char const* message)
{ if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function&& operation, char const* message)
{
    bool caught = false; try { operation(); } catch (std::exception const&) { caught = true; }
    check(caught, message);
}
struct Member { size_t plot_index; std::vector<size_t> quality_chains; };
}
int main() try
{
    using namespace pos2gpu;
    using verification_detail::sample;
    std::mt19937_64 random(0);
    size_t lookups = 0, solves = 0;
    auto empty = [&](auto const&) { ++lookups; return std::vector<Member>{}; };
    auto solve = [&](auto const&, auto const&, auto const&) { ++solves; };
    for (auto budget : {BoundedVerifyBudget{0, 1}, {1, 0}, {1, 2}})
        rejects([&] { sample(budget, random, empty, solve, {}); }, "invalid budget reached the reader");
    check(!lookups && !solves, "invalid budget performed work");
    rejects([&] { sample({7, 1}, random, empty, solve, {}); }, "empty challenge sequence succeeded");
    check(lookups == 7 && !solves, "empty sample escaped its challenge or solver bound");

    std::vector<Member> group;
    for (size_t member = 0; member < UINT16_MAX; ++member)
        group.push_back({member, {member, member, member, member}});
    lookups = solves = 0;
    auto large = [&](auto const&) -> auto const& { ++lookups; return group; };
    auto result = sample({100, 1}, random, large, solve, {});
    check(lookups == 1 && solves == 1 && result.trials == 1 && result.full_proofs_validated == 1,
        "large group triggered all-member/all-chain reconstruction");
    check(result.proofs_found == group.size() * group.front().quality_chains.size(), "returned quality count was truncated");

    // A zero-sized first member must neither become an empty success nor bias
    // selection toward the first available member/chain on every restart.
    group = {{0, {}}, {1, {10}}, {2, {20, 21, 22}}};
    std::set<size_t> selected;
    auto record = [&](auto const&, auto const& member, auto const& chain)
    { check(member.plot_index != 0, "empty member reached solver"); selected.insert(chain); ++solves; };
    lookups = solves = 0;
    result = sample({100, 100}, random, large, record, {});
    check(solves == 100 && lookups == 100 && selected == std::set<size_t>{10, 20, 21, 22},
        "sampling failed to reach returned members/chains or exceeded the requested reconstruction target");
    check(result.full_proofs_validated == 100, "positive sample target was not met");

    lookups = solves = 0;
    auto sparse = [&](auto const&) { ++lookups; return lookups < 3 ? std::vector<Member>{} : group; };
    result = sample({3, 1}, random, sparse, solve, {});
    check(result.trials == 3 && solves == 1, "sparse positives did not respect independent challenge/solve budgets");
    lookups = solves = 0;
    auto invalid = [&](auto const&, auto const&, auto const&)
    { ++solves; throw std::runtime_error("invalid reconstructed proof"); };
    rejects([&] { sample({100, 1}, random, large, invalid, {}); }, "invalid proof was ignored");
    check(lookups == 1 && solves == 1, "invalid proof triggered retries beyond the solve budget");

    lookups = solves = 0;
    rejects([&] { sample({100, 1}, random, large, solve, [] { throw std::runtime_error("cancelled"); }); },
        "cancelled work succeeded");
    check(!lookups && !solves, "pre-cancellation reached reader or solver");
    auto cancel_after_lookup = [&] { if (lookups) throw std::runtime_error("deadline"); };
    rejects([&] { sample({100, 1}, random, large, solve, cancel_after_lookup); }, "deadline after lookup was ignored");
    check(lookups == 1 && !solves, "deadline after lookup reached solver");
    auto cancel_after_solve = [&] { if (solves) throw std::runtime_error("cancelled after solve"); };
    rejects([&] { sample({100, 1}, random, large, solve, cancel_after_solve); }, "cancelled proof became durable success");
    result = sample({100, 1}, random, large, solve, {});
    check(result.full_proofs_validated == 1 && result.trials == 1, "restart reused in-memory failed sampling progress");
    std::cout << "bounded proof sampling budgets, positive evidence, large groups, corruption and cancellation passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
