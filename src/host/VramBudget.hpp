#pragma once

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
// Native CUDA calibration; the configurable safety buffer is added separately.
inline std::uint64_t streaming_base_peak_bytes(int k, StreamingTier tier)
{
    if (k < 18 || k > 32 || (k & 1))
        throw std::invalid_argument("k must be even in [18, 32]");
    std::uint64_t mib = 0;
    switch (tier) {
        case StreamingTier::Plain:   mib = 7290; break;
        case StreamingTier::Compact: mib = 5200; break;
        case StreamingTier::Minimal: mib = 3640; break;
        case StreamingTier::Tiny:    mib = 1064; break;
        case StreamingTier::Pinned:  mib = 1150; break;
    }
    auto const bytes = mib << 20;
    return k < 28 ? bytes >> (28 - k) : bytes << (k - 28);
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
