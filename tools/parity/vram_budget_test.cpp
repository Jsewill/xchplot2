#undef NDEBUG
#include "host/VramBudget.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <iostream>

int main()
{
    using namespace pos2gpu;
    constexpr std::uint64_t MiB = 1ULL << 20, GiB = 1ULL << 30;
    constexpr std::uint64_t buffer = 256 * MiB;
    std::array tiers{StreamingTier::Plain, StreamingTier::Compact,
                     StreamingTier::Minimal, StreamingTier::Tiny, StreamingTier::Pinned};
    for (int k = 18; k <= 32; k += 2) {
        for (auto tier : tiers) {
            auto const peak = streaming_base_peak_bytes(k, tier);
            assert(vram_fits(peak + buffer, peak, buffer));
            assert(!vram_fits(peak + buffer - 1, peak, buffer));
            assert(vram_scratch_budget(peak + buffer, peak, buffer) == 0);
            assert(vram_scratch_budget(peak + buffer - 1, peak, buffer) == 0);
            assert(vram_scratch_budget(peak + buffer + 812, peak, buffer) == 812);
        }
    }
    // Target card capacities after a representative 390 MiB context. The
    // picker must include the buffer exactly once and stop at Tiny.
    for (auto [gib, expected] : std::array{
             std::pair{2, StreamingTier::Tiny}, std::pair{4, StreamingTier::Tiny},
             std::pair{6, StreamingTier::Compact}, std::pair{8, StreamingTier::Plain}}) {
        auto const free = gib * GiB - 390 * MiB;
        for (auto tier : tiers) {
            if (vram_fits(free, streaming_base_peak_bytes(28, tier), buffer)) {
                assert(tier == expected);
                break;
            }
        }
    }
    assert(!vram_fits(GiB - 390 * MiB, streaming_base_peak_bytes(28, StreamingTier::Tiny), buffer));
    assert(vram_fits(12 * GiB - 390 * MiB, 10468 * MiB, buffer));
    assert(!vram_fits(std::numeric_limits<std::uint64_t>::max(),
                      std::numeric_limits<std::uint64_t>::max(), 1));
    assert(vram_mib_bytes("256", "buffer") == buffer);
    for (auto value : {"", "0", "-1", "+1", " 1", "128junk", "18446744073709551615"}) {
        bool threw = false;
        try { vram_mib_bytes(value, "buffer"); }
        catch (std::invalid_argument const&) { threw = true; }
        assert(threw);
    }
    for (int k : {-1, 0, 17, 19, 33}) {
        bool threw = false;
        try { streaming_base_peak_bytes(k, StreamingTier::Tiny); }
        catch (std::invalid_argument const&) { threw = true; }
        assert(threw);
    }
    std::cout << "VRAM budgets: exact boundaries, target capacities, and overflow passed\n";
}
