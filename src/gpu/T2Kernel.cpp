// T2Kernel.cu — port of pos2-chip Table2Constructor.
//
// Differences from T1 (see T1Kernel.cu):
//   - Input is T1Pairing (12 bytes, has 64-bit meta accessor), not Xs_Candidate.
//   - matching_target uses table_id=2 and meta=T1Pairing.meta() (64-bit).
//     ProofHashing::matching_target sets extra_rounds_bits=0 for table_id != 1.
//   - pairing_t2 calls AesHash::pairing without extra_rounds_bits (always 0).
//   - num_match_key_bits = strength (not hard-coded 2 like T1).
//   - Output T2Pairing has the AES pair.meta_result (64-bit) + x_bits derived
//     from upper-k bits of meta_l/meta_r.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T2Offsets.cuh"
#include "host/PoolSizing.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace pos2gpu {

T2MatchParams make_t2_params(int k, int strength)
{
    T2MatchParams p{};
    p.k                     = k;
    p.strength              = strength;
    p.num_section_bits      = (k < 28) ? 2 : (k - 26);
    p.num_match_key_bits    = strength; // T2 uses strength match_key bits
    p.num_match_target_bits = k - p.num_section_bits - p.num_match_key_bits;
    return p;
}

// T2's three kernels — compute_bucket_offsets, compute_fine_bucket_offsets,
// match_all_buckets — have moved to T2Offsets.cuh / T2OffsetsCuda.cu /
// T2OffsetsSycl.cpp on the cross-backend path. The previously-unused
// matching_section helper went with them.

void launch_t2_match(
    uint8_t const* plot_id_bytes,
    T2MatchParams const& params,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_mi,
    uint64_t t1_count,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
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

    // Fine-bucket pre-index; see T3Kernel.cu for the scheme.
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
    if (!d_sorted_meta || !d_sorted_mi ||
        !d_out_meta || !d_out_mi || !d_out_xbits || !d_out_count)
    {
        throw std::invalid_argument("invalid argument to launch wrapper");
    }
    if (params.num_match_target_bits <= FINE_BITS) throw std::invalid_argument("invalid argument to launch wrapper");

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = d_offsets + (num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);

    // Bucket + fine-bucket offsets — backend-dispatched via T2Offsets.cuh.
    launch_t2_compute_bucket_offsets(
        d_sorted_mi, t1_count,
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
    int num_info_bits = params.k;
    int half_k        = params.k / 2;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    // Match — backend-dispatched via T2Offsets.cuh.
    launch_t2_match_all_buckets(
        keys, d_sorted_meta, d_sorted_mi,
        d_offsets, d_fine_offsets,
        num_match_keys, num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, FINE_BITS,
        target_mask, num_test_bits, num_info_bits, half_k,
        d_out_meta, d_out_mi, d_out_xbits, d_out_count,
        capacity, l_count_max, q);
}

} // namespace pos2gpu
