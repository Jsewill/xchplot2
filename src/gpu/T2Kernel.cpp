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

namespace {

// Fine-bucket pre-index; see T3Kernel.cu for the scheme.
constexpr int kT2FineBits = 8;

// Shared parameter derivation so launch_t2_match, launch_t2_match_prepare,
// and launch_t2_match_range all agree on bucket counts, offset layout,
// and temp_storage sizing.
struct T2Derived {
    uint32_t num_sections;
    uint32_t num_match_keys;
    uint32_t num_buckets;
    uint64_t fine_entries;
    size_t   bucket_bytes;
    size_t   fine_bytes;
    size_t   temp_needed;
    uint32_t target_mask;
    int      num_test_bits;
    int      num_info_bits;
    int      half_k;
    uint64_t l_count_max;
};

T2Derived derive_t2(T2MatchParams const& params)
{
    T2Derived d{};
    d.num_sections    = 1u << params.num_section_bits;
    d.num_match_keys  = 1u << params.num_match_key_bits;
    d.num_buckets     = d.num_sections * d.num_match_keys;
    uint64_t const fine_count = 1ull << kT2FineBits;
    d.fine_entries    = uint64_t(d.num_buckets) * fine_count + 1;
    d.bucket_bytes    = sizeof(uint64_t) * (d.num_buckets + 1);
    d.fine_bytes      = sizeof(uint64_t) * d.fine_entries;
    d.temp_needed     = d.bucket_bytes + d.fine_bytes;
    d.target_mask     = (params.num_match_target_bits >= 32)
                          ? 0xFFFFFFFFu
                          : ((1u << params.num_match_target_bits) - 1u);
    d.num_test_bits   = params.num_match_key_bits;
    d.num_info_bits   = params.k;
    d.half_k          = params.k / 2;
    d.l_count_max =
        static_cast<uint64_t>(max_pairs_per_section(params.k, params.num_section_bits));
    return d;
}

} // namespace

void launch_t2_match_prepare(
    uint8_t const* plot_id_bytes,
    T2MatchParams const& params,
    uint32_t const* d_sorted_mi,
    uint64_t t1_count,
    uint64_t* d_out_count,
    void* d_temp_storage,
    size_t* temp_bytes,
    sycl::queue& q)
{
    if (!plot_id_bytes || !temp_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 32) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");

    T2Derived const d = derive_t2(params);

    if (d_temp_storage == nullptr) {
        *temp_bytes = d.temp_needed;
        return;
    }
    if (*temp_bytes < d.temp_needed) throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_sorted_mi || !d_out_count) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.num_match_target_bits <= kT2FineBits) throw std::invalid_argument("invalid argument to launch wrapper");

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = d_offsets + (d.num_buckets + 1);

    launch_t2_compute_bucket_offsets(
        d_sorted_mi, t1_count,
        params.num_match_target_bits,
        d.num_buckets, d_offsets, q);
    launch_t2_compute_fine_bucket_offsets(
        d_sorted_mi, d_offsets,
        params.num_match_target_bits, kT2FineBits,
        d.num_buckets, d_fine_offsets, q);
    q.memset(d_out_count, 0, sizeof(uint64_t)).wait();
}

void launch_t2_match_range(
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
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)t1_count;
    if (!plot_id_bytes) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.k < 18 || params.k > 32) throw std::invalid_argument("invalid argument to launch wrapper");
    if (params.strength < 2)            throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_temp_storage)                throw std::invalid_argument("invalid argument to launch wrapper");
    if (!d_sorted_meta || !d_sorted_mi ||
        !d_out_meta || !d_out_mi || !d_out_xbits || !d_out_count)
    {
        throw std::invalid_argument("invalid argument to launch wrapper");
    }

    T2Derived const d = derive_t2(params);

    if (bucket_end > d.num_buckets) throw std::invalid_argument("invalid argument to launch wrapper");
    if (bucket_end <= bucket_begin) return;  // empty range is a no-op

    constexpr int kThreads = 256;
    uint64_t const blocks_x_u64 = (d.l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) throw std::invalid_argument("invalid argument to launch wrapper");

    auto const* d_offsets      = reinterpret_cast<uint64_t const*>(d_temp_storage);
    auto const* d_fine_offsets = d_offsets + (d.num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);

    launch_t2_match_all_buckets(
        keys, d_sorted_meta, d_sorted_mi,
        // launch_t2_match_all_buckets takes mutable pointers to the
        // offset arrays (historical — they're treated as const inside
        // the kernel). Cast away const at the ABI boundary only.
        const_cast<uint64_t*>(d_offsets),
        const_cast<uint64_t*>(d_fine_offsets),
        d.num_match_keys, d.num_buckets,
        params.k, params.num_section_bits,
        params.num_match_target_bits, kT2FineBits,
        d.target_mask, d.num_test_bits, d.num_info_bits, d.half_k,
        d_out_meta, d_out_mi, d_out_xbits, d_out_count,
        capacity, d.l_count_max,
        bucket_begin, bucket_end,
        q);
}

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
    // Single-shot wrapper: prepare + one full-range match. Preserves the
    // original API for test-mode, the pool path, and parity-test callers.
    launch_t2_match_prepare(
        plot_id_bytes, params, d_sorted_mi, t1_count,
        d_out_count, d_temp_storage, temp_bytes, q);
    if (d_temp_storage == nullptr) return;  // size-query path

    T2Derived const d = derive_t2(params);
    launch_t2_match_range(
        plot_id_bytes, params,
        d_sorted_meta, d_sorted_mi, t1_count,
        d_out_meta, d_out_mi, d_out_xbits, d_out_count,
        capacity, d_temp_storage,
        /*bucket_begin=*/0, /*bucket_end=*/d.num_buckets, q);
}

} // namespace pos2gpu
