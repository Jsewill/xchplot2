// T1Kernel.cu — port of pos2-chip Table1Constructor.
//
// Algorithm (mirrors pos2-chip/src/plot/TableConstructorGeneric.hpp):
//
//   For each section_l in {0,1,2,3} (order doesn't affect the *set* of
//     T1Pairings produced; CPU iterates 3,0,2,1 but the post-construct
//     sort by match_info collapses ordering):
//     section_r = matching_section(section_l)
//     For each match_key_r in [0, num_match_keys):
//       L = sorted_xs[section_l..section_l+1)            (entire section)
//       R = sorted_xs in (section_r, match_key_r) bucket
//       For each L candidate (one thread):
//         target_l = matching_target(1, match_key_r, x_l) & target_mask
//         binary-search R for first entry with match_target == target_l
//         walk forward while still equal; for each:
//           pairing_t1(x_l, x_r); if test_result == 0, emit T1Pairing
//             { meta = (x_l << k) | x_r, match_info = pair.r[0] mask k }

#include "host/PoolSizing.hpp"

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T1Offsets.cuh"

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace pos2gpu {

T1MatchParams make_t1_params(int k, int strength)
{
    T1MatchParams p{};
    p.k                     = k;
    p.strength              = strength;
    p.num_section_bits      = (k < 28) ? 2 : (k - 26);
    p.num_match_key_bits    = 2; // table_id == 1
    p.num_match_target_bits = k - p.num_section_bits - p.num_match_key_bits;
    return p;
}

// All T1 kernels (compute_bucket_offsets, compute_fine_bucket_offsets,
// match_all_buckets) and the previously-unused matching_section helper
// have moved to T1Offsets.cuh / T1OffsetsSycl.cpp on the cross-backend path.

void launch_t1_match(
    uint8_t const* plot_id_bytes,
    T1MatchParams const& params,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t total,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
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

    // temp layout: offsets[num_buckets + 1] uint64 || fine_offsets[num_buckets * 2^FINE_BITS + 1]
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
    if (!d_sorted_xs || !d_out_meta || !d_out_mi || !d_out_count)
        throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.num_match_target_bits <= FINE_BITS) throw std::invalid_argument("invalid argument to launch wrapper");

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = d_offsets + (num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);

    // 1) Bucket offsets — backend-dispatched (CUDA or SYCL) via T1Offsets.cuh.
    launch_compute_bucket_offsets(
        d_sorted_xs, total,
        params.num_match_target_bits,
        num_buckets,
        d_offsets, q);
    // 1b) Fine-bucket offsets — backend-dispatched via T1Offsets.cuh.
    launch_compute_fine_bucket_offsets(
        d_sorted_xs, d_offsets,
        params.num_match_target_bits, FINE_BITS,
        num_buckets, d_fine_offsets, q);
    // Reset out_count to 0.
    q.memset(d_out_count, 0, sizeof(uint64_t)).wait();

    // Use the static per-section capacity as the over-launch upper
    // bound for blocks_x. Avoids a D2H copy + stream sync that the
    // actual-max computation would need; excess threads early-exit on
    // `l >= l_end` inside match_all_buckets. Saves ~50–150 µs of host
    // fence per plot (× 3 phases) and unblocks stream-level overlap.
    uint64_t l_count_max =
        static_cast<uint64_t>(max_pairs_per_section(params.k, params.num_section_bits));

    uint32_t target_mask = (params.num_match_target_bits >= 32)
                            ? 0xFFFFFFFFu
                            : ((1u << params.num_match_target_bits) - 1u);
    int extra_rounds_bits = params.strength - 2;
    int num_test_bits     = params.num_match_key_bits;
    int num_info_bits     = params.k;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    // Match — backend-dispatched (CUDA or SYCL) via T1Offsets.cuh.
    launch_t1_match_all_buckets(
        keys, d_sorted_xs, d_offsets, d_fine_offsets,
        num_match_keys, num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, FINE_BITS,
        extra_rounds_bits, target_mask,
        num_test_bits, num_info_bits,
        d_out_meta, d_out_mi, d_out_count,
        capacity, l_count_max, q);
}

} // namespace pos2gpu
