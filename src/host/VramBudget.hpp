#pragma once

#include "PoolSizing.hpp"

#include <cstdint>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pos2gpu {

enum class StreamingTier { Plain, Compact, Minimal, Tiny, Pinned };

inline std::size_t vram_mib_bytes(char const* value, char const* name)
{
    std::string_view const text(value);
    std::size_t mib = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), mib);
    if (error != std::errc{} || end != text.data() + text.size() || !mib ||
        mib > (std::numeric_limits<std::size_t>::max() >> 20))
        throw std::invalid_argument(std::string(name) + " must be a positive MiB count that fits in memory size");
    return mib << 20;
}

// Device-allocation floors measured at k=28 with optional match scratch off.
// Backend sort scratch and the configurable safety buffer are added separately.
inline std::uint64_t streaming_base_peak_bytes(int k, StreamingTier tier)
{
    if (k < 18 || k > 32 || (k & 1))
        throw std::invalid_argument("k must be even in [18, 32]");
    std::uint64_t mib = 0;
    switch (tier) {
        case StreamingTier::Plain:   mib = 7290; break;
        case StreamingTier::Compact: mib = 5200; break;
        case StreamingTier::Minimal: mib = 3900; break;
        case StreamingTier::Tiny:    mib = 1100; break;
        case StreamingTier::Pinned:  mib = 1150; break;
    }
    // Capacity includes the section overflow allowance, which does not scale
    // as 2^k. Tiny and Pinned also retain a fixed 24 MiB partition tile.
    std::uint64_t const fixed_mib =
        tier == StreamingTier::Tiny || tier == StreamingTier::Pinned ? 24 : 0;
    int const section_bits = k < 28 ? 2 : k - 26;
    auto const cap = match_phase_capacity(k, section_bits);
    auto const reference_mib = match_phase_capacity(28, 2) >> 20;
    return ((mib - fixed_mib) * cap + reference_mib - 1) / reference_mib
        + (fixed_mib << 20);
}

inline bool vram_fits(std::uint64_t free, std::uint64_t peak,
                     std::uint64_t buffer)
{
    return buffer <= free && peak <= free - buffer;
}

inline std::uint64_t vram_scratch_budget(std::uint64_t free,
                                        std::uint64_t peak,
                                        std::uint64_t buffer)
{
    return vram_fits(free, peak, buffer) ? free - buffer - peak : 0;
}

} // namespace pos2gpu
