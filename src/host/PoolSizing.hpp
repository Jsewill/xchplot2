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

// T1 / T2 / T3 match-output upper bound: max_pairs_per_section * num_sections.
// Centralises the cap formula that the per-shard match phases use to size
// d_t*_meta / d_t*_mi / d_t*_xbits scratch.
inline std::uint64_t match_phase_capacity(int k, int num_section_bits) noexcept
{
    return static_cast<std::uint64_t>(
        max_pairs_per_section(k, num_section_bits))
        * (std::uint64_t{1} << num_section_bits);
}

// T2 and T3 share one per-bucket candidate allocation. Keep admission and
// kernel wrappers on the same sizing rule; overflow still falls back to the
// allocation-free match path. Callers supply admitted match capacities.
inline std::uint64_t twophase_candidate_capacity(std::uint64_t output_capacity,
                                               std::uint64_t buckets) noexcept
{
    return (output_capacity / (buckets ? buckets : 1)) * 6 + 4096;
}

inline std::uint64_t twophase_scratch_bytes(std::uint64_t output_capacity,
                                          std::uint64_t buckets) noexcept
{
    return twophase_candidate_capacity(output_capacity, buckets) * (2 * sizeof(std::uint32_t)) +
        sizeof(std::uint64_t) + sizeof(std::uint32_t);
}

} // namespace pos2gpu
