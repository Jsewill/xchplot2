// Internal compact transport contract shared by the GPU drain and receiver.
#pragma once

#include <cstdint>
#include <span>

namespace group_grow
{
struct PackedMemberValidation
{
    static constexpr uint32_t out_of_range = 1;
    static constexpr uint32_t unsorted = 2;
    static constexpr uint32_t wide_delta = 4;
    static constexpr uint32_t known_flags = out_of_range | unsorted | wide_delta;

    // Coverage is reported by completed device validation work, including the
    // predecessor of each tile. Never infer success from a zeroed flag alone.
    uint64_t checked_count = 0;
    uint32_t flags = 0;
};

// The first callback contains the complete boundary prelude and validation
// summary, with no payload. Later callbacks contain consecutive packed words.
// All spans borrow the transfer window only until the callback returns.
struct PackedFragmentTile
{
    std::span<uint32_t const> boundaries;
    std::span<uint64_t const> words;
    PackedMemberValidation validation;
};
} // namespace group_grow
