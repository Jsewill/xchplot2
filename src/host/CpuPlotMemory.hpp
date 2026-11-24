// CPU member-generation allocations. The same layout is used for admission
// and execution on both GPU branches and by standalone compute workers.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pos2gpu::cpu {

inline uint32_t member_bound(unsigned k)
{
    if (k < 18 || k > 28 || k % 2) throw std::invalid_argument("unsupported CPU plot k");
    const uint32_t n = uint32_t{1} << k;
    // Same table cardinality envelope as the plot format's reference producer.
    return n + (uint32_t{4} << (k - (8 - (28 - k) / 2)));
}

struct Layout {
    unsigned k, lanes, bucket_bits;
    uint32_t n, bound, buckets, block, query_rows, right_tile, heads;

    Layout(unsigned size, unsigned threads) : k(size), bound(member_bound(k))
    {
        if (!threads) throw std::invalid_argument("CPU plotting requires a thread grant");
        n = uint32_t{1} << k;
        bucket_bits = std::clamp(k - 14u, 4u, 12u);
        buckets = uint32_t{1} << bucket_bits;
        // Bound partly filled bucket tails and keep useful work per lane.
        lanes = std::min(threads, n / (buckets * 512));
        block = std::min(512u, std::bit_floor(n / (32 * buckets * lanes)));
        query_rows = bound / 4 + (buckets / 4) * lanes * block;
        const uint32_t per_bucket = n / buckets;
        right_tile = std::min(2 * per_bucket, n / (32 * lanes));
        heads = per_bucket;
    }

    uint32_t blocks(uint32_t rows, uint32_t bins) const
    {
        return (rows + block - 1) / block + bins * lanes;
    }

    uint64_t table_bytes(uint32_t rows, uint32_t bins, unsigned stride) const
    {
        return uint64_t(blocks(rows, bins)) * (uint64_t(block) * stride + sizeof(uint32_t))
             + uint64_t(lanes) * bins * 3 * sizeof(uint32_t);
    }

    uint64_t peak_bytes(unsigned strength = 2) const
    {
        if (strength < 2 || strength > std::min(17u, k - 3)) throw std::invalid_argument("unsupported CPU match strength");
        // Xs/T1/T2 carry 8/12/16 bytes. T3 emits an 8-byte fragment.
        // All supported sizes have four sections. T1 has four match keys,
        // and T2/T3 use the requested strength.
        const auto join = [&](unsigned input, unsigned output, unsigned key_bits) {
            const uint32_t bins = std::max(1u, (buckets / 4) >> key_bits);
            const uint32_t targets = (n / 4) >> key_bits;
            const auto tables = table_bytes(bound, buckets, input) + table_bytes(bound, buckets, output);
            if (bins == 1) return tables + uint64_t(targets) * (2 * input + sizeof(uint32_t));
            return tables + table_bytes(query_rows, bins, input)
                + uint64_t(std::min(lanes, bins)) * (uint64_t(right_tile) * input + uint64_t(heads) * sizeof(uint32_t));
        };
        // The left block directory contains at most one span per block.
        // Each final fragment can coexist with its bucket and its sort scratch.
        const uint64_t directories = uint64_t(blocks(bound, buckets)) * 32;
        const uint64_t sort = table_bytes(bound, buckets, 8) + uint64_t(bound) * 16;
        return std::max({join(8, 12, 2), join(12, 16, strength), join(16, 8, strength), sort}) + directories
            + uint64_t(lanes) * 65536 + (uint64_t{2} << 20);
    }
};

// A worker can receive fewer permits than it offered. Block sizes change at
// powers of two, so the largest allocation need not use the most lanes.
inline uint64_t member_peak(unsigned k, unsigned threads = 0, unsigned strength = 2)
{
    const auto max_lanes = Layout(k, threads ? threads : std::numeric_limits<unsigned>::max()).lanes;
    uint64_t peak = 0;
    for (unsigned n = 1; n <= max_lanes; ++n)
        peak = std::max(peak, Layout(k, n).peak_bytes(strength));
    return peak;
}

// The worker also needs its thread runtime and writer/codec bookkeeping.
inline uint64_t worker_peak(unsigned k, unsigned threads = 0, unsigned strength = 2)
{
    return member_peak(k, threads, strength) + (uint64_t{128} << 20);
}

} // namespace pos2gpu::cpu
