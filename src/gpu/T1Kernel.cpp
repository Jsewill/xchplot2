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

namespace {

constexpr int kT1FineBits = 8;

struct T1Derived {
    uint32_t num_sections;
    uint32_t num_match_keys;
    uint32_t num_buckets;
    uint64_t fine_entries;
    size_t   bucket_bytes;
    size_t   fine_bytes;
    size_t   temp_needed;
    uint32_t target_mask;
    uint64_t l_count_max;
};

T1Derived derive_t1(T1MatchParams const& params)
{
    T1Derived d{};
    d.num_sections    = 1u << params.num_section_bits;
    d.num_match_keys  = 1u << params.num_match_key_bits;
    d.num_buckets     = d.num_sections * d.num_match_keys;
    uint64_t const fine_count = 1ull << kT1FineBits;
    d.fine_entries    = uint64_t(d.num_buckets) * fine_count + 1;
    d.bucket_bytes    = sizeof(uint64_t) * (d.num_buckets + 1);
    d.fine_bytes      = sizeof(uint64_t) * d.fine_entries;
    d.temp_needed     = d.bucket_bytes + d.fine_bytes;
    d.target_mask     = (params.num_match_target_bits >= 32)
                          ? 0xFFFFFFFFu
                          : ((1u << params.num_match_target_bits) - 1u);
    d.l_count_max =
        static_cast<uint64_t>(max_pairs_per_section(params.k, params.num_section_bits));
    return d;
}

} // namespace

void launch_t1_match_prepare(
    uint8_t const* plot_id_bytes,
    T1MatchParams const& params,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t total,
    uint64_t* d_out_count,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q)
{
    if (!plot_id_bytes || !temp_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 32) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");

    T1Derived const d = derive_t1(params);

    if (d_temp_storage == nullptr) {
        *temp_bytes = d.temp_needed;
        return;
    }
    if (*temp_bytes < d.temp_needed) throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_sorted_xs || !d_out_count)  throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.num_match_target_bits <= kT1FineBits) throw std::invalid_argument("invalid argument to launch wrapper");

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = d_offsets + (d.num_buckets + 1);

    launch_compute_bucket_offsets(
        d_sorted_xs, total,
        params.num_match_target_bits,
        d.num_buckets, d_offsets, q);
    launch_compute_fine_bucket_offsets(
        d_sorted_xs, d_offsets,
        params.num_match_target_bits, kT1FineBits,
        d.num_buckets, d_fine_offsets, q);
    q.memset(d_out_count, 0, sizeof(uint64_t)).wait();
}

void launch_t1_match_range(
    uint8_t const* plot_id_bytes,
    T1MatchParams const& params,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t total,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)total;
    if (!plot_id_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 32) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_temp_storage)                throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_sorted_xs || !d_out_meta || !d_out_mi || !d_out_count)
        throw std::invalid_argument("invalid argument to launch wrapper");

    T1Derived const d = derive_t1(params);
    if (bucket_end > d.num_buckets) throw std::invalid_argument("invalid argument to launch wrapper");
    if (bucket_end <= bucket_begin) return;

    constexpr int kThreads = 256;
    uint64_t const blocks_x_u64 = (d.l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    auto const* d_offsets      = reinterpret_cast<uint64_t const*>(d_temp_storage);
    auto const* d_fine_offsets = d_offsets + (d.num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);

    int const extra_rounds_bits = params.strength - 2;
    int const num_test_bits     = params.num_match_key_bits;
    int const num_info_bits     = params.k;

    launch_t1_match_all_buckets(
        keys, d_sorted_xs,
        const_cast<uint64_t const*>(d_offsets),
        const_cast<uint64_t const*>(d_fine_offsets),
        d.num_match_keys, d.num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, kT1FineBits,
        extra_rounds_bits, d.target_mask,
        num_test_bits, num_info_bits,
        d_out_meta, d_out_mi, d_out_count,
        capacity, d.l_count_max,
        bucket_begin, bucket_end, q);
}

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
    // Single-shot wrapper: prepare + one full-range match. Preserves
    // the original API for pool path, test mode, and parity tests.
    launch_t1_match_prepare(
        plot_id_bytes, params, d_sorted_xs, total,
        d_out_count, d_temp_storage, temp_bytes, q);
    if (d_temp_storage == nullptr) return;  // size-query path

    T1Derived const d = derive_t1(params);
    launch_t1_match_range(
        plot_id_bytes, params, d_sorted_xs, total,
        d_out_meta, d_out_mi, d_out_count,
        capacity, d_temp_storage,
        /*bucket_begin=*/0, /*bucket_end=*/d.num_buckets, q);
}

} // namespace pos2gpu
