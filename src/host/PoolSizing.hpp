// PoolSizing.hpp — inline helpers shared by the buffer pool, the
// pipeline orchestrator, and the match-kernel wrappers. Kept here so a
// single formula change updates every consumer.

#pragma once

#include <cstddef>
#include <cstdint>

namespace pos2gpu {

// Maximum L-side rows that can fall into any single (section, match_key)
// bucket at the given (k, section_bits). Used to size the persistent
// pool AND as the safe over-launch upper bound for the match kernels'
// `blocks_x` dimension. Over-launched threads early-exit on the
// `l >= l_end` guard at the top of the match body, so slight
// over-launch is free on the GPU.
//
// Formula mirrors pos2-chip's TableConstructorGeneric.hpp:23.
inline std::size_t max_pairs_per_section(int k, int num_section_bits) noexcept
{
    int const extra_margin_bits = 8 - ((28 - k) / 2);
    return (1ULL << (k - num_section_bits)) + (1ULL << (k - extra_margin_bits));
}

} // namespace pos2gpu
