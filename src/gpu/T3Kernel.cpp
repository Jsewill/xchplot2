// T3Kernel.cu — port of pos2-chip Table3Constructor.
//
// Differences from T2:
//   - Input is T2Pairing { meta(64), match_info(32), x_bits(32) }.
//   - matching_target uses table_id=3 and meta=T2Pairing.meta (no extra rounds).
//   - pairing_t3 only consumes test_result; no match_info / meta extraction
//     from the AES output. AES rounds = AES_PAIRING_ROUNDS (16), no strength
//     bonus.
//   - Emit T3Pairing { proof_fragment = FeistelCipher.encrypt(all_x_bits) }
//     where all_x_bits = (l.x_bits << k) | r.x_bits.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/FeistelCipherGpu.cuh"
#include "gpu/T2Offsets.cuh"
#include "gpu/T3Kernel.cuh"
#include "gpu/T3Offsets.cuh"
#include "host/PoolSizing.hpp"

#include <climits>
#include <cstdint>

namespace pos2gpu {

// The CUDA __constant__ FeistelKey + its setup have moved to
// T3OffsetsCuda.cu, scoped to the wrapper that uses them. The SYCL
// path captures FeistelKey by value in the lambda instead.

T3MatchParams make_t3_params(int k, int strength)
{
    T3MatchParams p{};
    p.k                     = k;
    p.strength              = strength;
    p.num_section_bits      = (k < 28) ? 2 : (k - 26);
    p.num_match_key_bits    = strength;
    p.num_match_target_bits = k - p.num_section_bits - p.num_match_key_bits;
    return p;
}

// T3's three kernels (compute_bucket_offsets, compute_fine_bucket_offsets,
// match_all_buckets) have moved to the cross-backend path. The two offset
// kernels are bit-identical to T2's and reuse T2Offsets.cuh's wrappers; the
// match kernel — Feistel-encrypted output — has its own wrapper in
// T3Offsets.cuh. The previously-unused matching_section helper went with
// them.


void launch_t3_match(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q)
{
    if (!plot_id_bytes || !temp_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 32) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");

    uint32_t num_sections    = 1u << params.num_section_bits;
    uint32_t num_match_keys  = 1u << params.num_match_key_bits;
    uint32_t num_buckets     = num_sections * num_match_keys;

    // Fine-bucket pre-index: 2^FINE_BITS slots per bucket shrinks the
    // match-kernel bsearch window by the same factor. Requires at least
    // FINE_BITS+1 bits of target range; num_match_target_bits is
    // k - section_bits - match_key_bits = 14..30 across the supported
    // (k, strength) matrix, so 8 fine bits always leaves ≥6 for bsearch.
    constexpr int FINE_BITS = 8;
    uint64_t const fine_count    = 1ull << FINE_BITS;
    uint64_t const fine_entries  = uint64_t(num_buckets) * fine_count + 1;

    size_t const bucket_bytes = sizeof(uint64_t) * (num_buckets + 1);
    size_t const fine_bytes   = sizeof(uint64_t) * fine_entries;
    size_t const needed       = bucket_bytes + fine_bytes;

    if (d_temp_storage == nullptr) {
        *temp_bytes = needed;

        return;
    }
    if (*temp_bytes < needed)        throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_sorted_meta || !d_sorted_xbits || !d_sorted_mi
        || !d_out_pairings || !d_out_count) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.num_match_target_bits <= FINE_BITS) {
        // Fall-back would be needed here; not expected for supported
        // (k, strength) combinations, so fail loudly if we ever trip it.
        throw std::invalid_argument("invalid argument to launch wrapper");
    }

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = d_offsets + (num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);
    FeistelKey  fk   = make_feistel_key(plot_id_bytes, params.k, /*rounds=*/4);

    // Bucket + fine-bucket offsets — reuse T2's wrappers (algorithm and
    // input layout are identical between T2 and T3).
    launch_t2_compute_bucket_offsets(
        d_sorted_mi, t2_count,
        params.num_match_target_bits,
        num_buckets, d_offsets, q);
    launch_t2_compute_fine_bucket_offsets(
        d_sorted_mi, d_offsets,
        params.num_match_target_bits, FINE_BITS,
        num_buckets, d_fine_offsets, q);
    q.memset(d_out_count, 0, sizeof(uint64_t)).wait();

    // See T1Kernel.cu for rationale: static per-section cap as over-
    // launch upper bound, excess threads early-exit on `l >= l_end`.
    uint64_t l_count_max =
        static_cast<uint64_t>(max_pairs_per_section(params.k, params.num_section_bits));

    uint32_t target_mask = (params.num_match_target_bits >= 32)
                            ? 0xFFFFFFFFu
                            : ((1u << params.num_match_target_bits) - 1u);
    int num_test_bits = params.num_match_key_bits;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    // Match — backend-dispatched via T3Offsets.cuh. Full bucket range
    // (0, num_buckets) preserves current single-pass behavior. Callers
    // wanting to split T3 match across temporally-separated passes
    // (see stage 4d in docs/t2-match-tiling-plan.md; same shape as T2)
    // should invoke launch_t3_match_all_buckets directly with a
    // sub-range.
    launch_t3_match_all_buckets(
        keys, fk,
        d_sorted_meta, d_sorted_xbits, d_sorted_mi,
        d_offsets, d_fine_offsets,
        num_match_keys, num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, FINE_BITS,
        target_mask, num_test_bits,
        d_out_pairings, d_out_count,
        capacity, l_count_max,
        /*bucket_begin=*/0, /*bucket_end=*/num_buckets,
        q);
}

} // namespace pos2gpu
