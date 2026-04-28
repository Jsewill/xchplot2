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
#include "gpu/T3Kernel.cuh"
#include "host/PoolSizing.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace pos2gpu {

// FeistelKey is 40 bytes (32-byte plot_id + 2 ints). Passed by value as
// a kernel arg, the compiler spilled it to local memory (STACK:40), so
// `fk.plot_id[i]` accesses inside feistel_encrypt became scattered LMEM
// LDGs — brutal for an L1-bound kernel. Stashing it in __constant__
// memory makes those loads broadcast-cached across the warp instead.
__constant__ FeistelKey g_t3_fk;

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

namespace {

__host__ __device__ inline uint32_t matching_section(uint32_t section, int num_section_bits)
{
    uint32_t num_sections = 1u << num_section_bits;
    uint32_t mask = num_sections - 1u;
    uint32_t rotated_left = ((section << 1) | (section >> (num_section_bits - 1))) & mask;
    uint32_t rotated_left_plus_1 = (rotated_left + 1) & mask;
    uint32_t section_new = ((rotated_left_plus_1 >> 1)
                          | (rotated_left_plus_1 << (num_section_bits - 1))) & mask;
    return section_new;
}

// One thread per bucket; last thread writes the sentinel.
__global__ void compute_bucket_offsets(
    uint32_t const* __restrict__ sorted_mi,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* __restrict__ offsets)
{
    uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b > num_buckets) return;
    if (b == num_buckets) {
        offsets[num_buckets] = total;
        return;
    }

    uint32_t bucket_shift = static_cast<uint32_t>(num_match_target_bits);
    uint64_t lo = 0, hi = total;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t bucket_mid = sorted_mi[mid] >> bucket_shift;
        if (bucket_mid < b) lo = mid + 1;
        else                hi = mid;
    }
    offsets[b] = lo;
}

// Compute fine-grained bucket offsets: one offset per (r_bucket,
// top-FINE_BITS-of-target) pair. Lets the match kernel replace a
// ~24-iteration bsearch on sorted_mi with a 2-LDG lookup + an ~16-
// iteration bsearch in a 256× narrower window. Each thread writes
// one fine_offsets entry via an in-range bsearch over sorted_mi
// restricted to its parent bucket.
__global__ void compute_fine_bucket_offsets(
    uint32_t const* __restrict__ sorted_mi,
    uint64_t const* __restrict__ bucket_offsets,
    int num_match_target_bits,
    int fine_bits,
    uint32_t num_buckets,
    uint64_t* __restrict__ fine_offsets)
{
    uint32_t const fine_count = 1u << fine_bits;
    uint32_t const total      = num_buckets * fine_count;
    uint32_t const tid        = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    uint32_t const r_bucket = tid / fine_count;
    uint32_t const fine_key = tid % fine_count;

    uint64_t const r_start = bucket_offsets[r_bucket];
    uint64_t const r_end   = bucket_offsets[r_bucket + 1];

    uint32_t const target_mask = (num_match_target_bits >= 32)
                                  ? 0xFFFFFFFFu
                                  : ((1u << num_match_target_bits) - 1u);
    uint32_t const shift       = static_cast<uint32_t>(num_match_target_bits - fine_bits);

    uint64_t lo = r_start, hi = r_end;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t t   = (sorted_mi[mid] & target_mask) >> shift;
        if (t < fine_key) lo = mid + 1;
        else              hi = mid;
    }
    fine_offsets[tid] = lo;

    // Last thread writes the sentinel (overall end = sorted_mi length).
    if (tid == total - 1) {
        fine_offsets[total] = bucket_offsets[num_buckets];
    }
}

// Biases for section-pair input slicing (cut #3). When the streaming
// minimal tier slices d_sorted_meta to just the section_l + section_r
// rows for each pass, the kernel still receives global l/r indices
// (l_start/l_end and r/lo/hi all derive from the full d_offsets +
// d_fine_offsets, which are built once across the full pipeline). The
// biases shift `l` and `r` to the corresponding offset inside the
// sliced buffer:
//
//   meta_l = sorted_meta[ l + meta_l_index_bias ]
//   meta_r = sorted_meta[ r + meta_r_index_bias ]
//
// Full-cap callers pass biases = 0 so indexing is unchanged. Sliced
// callers pass:
//   meta_l_index_bias = section_l_slice_start - section_l_row_start
//   meta_r_index_bias = section_r_slice_start - section_r_row_start
// which are negative when the slice is concatenated tightly. The
// kernel uses int64_t arithmetic to handle the negative case.
__global__ __launch_bounds__(256, 4) void match_all_buckets(
    AesHashKeys keys,
    uint64_t const* __restrict__ sorted_meta,
    uint32_t const* __restrict__ sorted_xbits,
    uint32_t const* __restrict__ sorted_mi,
    uint64_t const* __restrict__ d_offsets,
    uint64_t const* __restrict__ d_fine_offsets,
    uint32_t num_match_keys,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* __restrict__ out,
    unsigned long long* __restrict__ out_count,
    uint64_t out_capacity,
    uint32_t bucket_begin,
    int64_t meta_l_index_bias,
    int64_t meta_r_index_bias)
{
    __shared__ uint32_t sT[4 * 256];
    load_aes_tables_smem(sT);
    __syncthreads();

    uint32_t bucket_id   = bucket_begin + blockIdx.y;
    uint32_t section_l   = bucket_id / num_match_keys;
    uint32_t match_key_r = bucket_id % num_match_keys;

    uint32_t section_r;
    {
        uint32_t mask = (1u << num_section_bits) - 1u;
        uint32_t rl   = ((section_l << 1) | (section_l >> (num_section_bits - 1))) & mask;
        uint32_t rl1  = (rl + 1) & mask;
        section_r = ((rl1 >> 1) | (rl1 << (num_section_bits - 1))) & mask;
    }

    uint64_t l_start = d_offsets[section_l * num_match_keys];
    uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
    uint32_t r_bucket = section_r * num_match_keys + match_key_r;

    uint64_t l = l_start + blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (l >= l_end) return;

    uint64_t const meta_l_idx = static_cast<uint64_t>(
        static_cast<int64_t>(l) + meta_l_index_bias);
    uint64_t meta_l = sorted_meta[meta_l_idx];
    uint32_t xb_l   = sorted_xbits[l];

    uint32_t target_l = matching_target_smem(keys, 3u, match_key_r, meta_l, sT, 0)
                      & target_mask;

    // Fine-bucket pre-index: narrows the bsearch range by 2^fine_bits
    // using a precomputed offset table indexed by (r_bucket, top
    // fine_bits of target_l). Two cached LDGs replace the outer d_offsets
    // r_start/r_end and shrink the bsearch window 256× at fine_bits=8.
    uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
    uint32_t fine_key   = target_l >> fine_shift;
    uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
    uint64_t lo         = d_fine_offsets[fine_idx];
    uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
    uint64_t hi         = fine_hi;

    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t target_mid = sorted_mi[mid] & target_mask;
        if (target_mid < target_l) lo = mid + 1;
        else                       hi = mid;
    }

    uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                : ((1u << num_test_bits) - 1u);

    for (uint64_t r = lo; r < fine_hi; ++r) {
        uint32_t target_r = sorted_mi[r] & target_mask;
        if (target_r != target_l) break;

        uint64_t const meta_r_idx = static_cast<uint64_t>(
            static_cast<int64_t>(r) + meta_r_index_bias);
        uint64_t meta_r = sorted_meta[meta_r_idx];
        uint32_t xb_r   = sorted_xbits[r];

        Result128 res = pairing_smem(keys, meta_l, meta_r, sT, 0);
        uint32_t test_result = res.r[3] & test_mask;
        if (test_result != 0) continue;

        uint64_t all_x_bits = (uint64_t(xb_l) << k) | uint64_t(xb_r);
        uint64_t fragment   = feistel_encrypt(g_t3_fk, all_x_bits);

        unsigned long long out_idx = atomicAdd(out_count, 1ULL);
        if (out_idx >= out_capacity) return;

        T3PairingGpu p;
        p.proof_fragment = fragment;
        out[out_idx] = p;
    }
}

} // namespace

namespace {

// Fine-bucket pre-index: 2^kT3FineBits slots per bucket shrinks the
// match-kernel bsearch window by the same factor. Requires at least
// kT3FineBits+1 bits of target range; num_match_target_bits is
// k - section_bits - match_key_bits = 14..30 across the supported
// (k, strength) matrix, so 8 fine bits always leaves ≥6 for bsearch.
constexpr int kT3FineBits = 8;

struct T3Derived {
    uint32_t num_sections;
    uint32_t num_match_keys;
    uint32_t num_buckets;
    uint64_t fine_entries;
    size_t   bucket_bytes;
    size_t   fine_bytes;
    size_t   temp_needed;
    uint32_t target_mask;
    int      num_test_bits;
    uint64_t l_count_max;
};

T3Derived derive_t3(T3MatchParams const& params)
{
    T3Derived d{};
    d.num_sections    = 1u << params.num_section_bits;
    d.num_match_keys  = 1u << params.num_match_key_bits;
    d.num_buckets     = d.num_sections * d.num_match_keys;
    uint64_t const fine_count = 1ull << kT3FineBits;
    d.fine_entries    = uint64_t(d.num_buckets) * fine_count + 1;
    d.bucket_bytes    = sizeof(uint64_t) * (d.num_buckets + 1);
    d.fine_bytes      = sizeof(uint64_t) * d.fine_entries;
    d.temp_needed     = d.bucket_bytes + d.fine_bytes;
    d.target_mask     = (params.num_match_target_bits >= 32)
                          ? 0xFFFFFFFFu
                          : ((1u << params.num_match_target_bits) - 1u);
    d.num_test_bits   = params.num_match_key_bits;
    // See T1Kernel.cu for rationale: static per-section cap as over-
    // launch upper bound, excess threads early-exit on `l >= l_end`.
    d.l_count_max     = static_cast<uint64_t>(
        max_pairs_per_section(params.k, params.num_section_bits));
    return d;
}

} // namespace

cudaError_t launch_t3_match_prepare(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint32_t const* d_sorted_mi,
    uint64_t t2_count,
    uint64_t* d_out_count,
    void* d_temp_storage,
    size_t* temp_bytes,
    cudaStream_t stream)
{
    if (!plot_id_bytes || !temp_bytes) return cudaErrorInvalidValue;
    if (params.k < 18 || params.k > 32) return cudaErrorInvalidValue;
    if (params.strength < 2)            return cudaErrorInvalidValue;

    T3Derived const d = derive_t3(params);

    if (d_temp_storage == nullptr) {
        *temp_bytes = d.temp_needed;
        return cudaSuccess;
    }
    if (*temp_bytes < d.temp_needed)  return cudaErrorInvalidValue;
    if (!d_sorted_mi || !d_out_count) return cudaErrorInvalidValue;
    if (params.num_match_target_bits <= kT3FineBits) {
        // Fall-back would be needed here; not expected for supported
        // (k, strength) combinations, so fail loudly if we ever trip it.
        return cudaErrorInvalidValue;
    }

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = d_offsets + (d.num_buckets + 1);

    // Upload Feistel key once per prepare. The staging caller may invoke
    // launch_t3_match_range repeatedly without re-preparing, so keep the
    // upload in prepare rather than range to avoid redundant H2D.
    FeistelKey fk = make_feistel_key(plot_id_bytes, params.k, /*rounds=*/4);
    cudaError_t fk_err = cudaMemcpyToSymbolAsync(
        g_t3_fk, &fk, sizeof(fk), 0, cudaMemcpyHostToDevice, stream);
    if (fk_err != cudaSuccess) return fk_err;

    {
        constexpr int kOffThreads = 256;
        unsigned off_blocks = static_cast<unsigned>(
            (d.num_buckets + 1 + kOffThreads - 1) / kOffThreads);
        compute_bucket_offsets<<<off_blocks, kOffThreads, 0, stream>>>(
            d_sorted_mi, t2_count,
            params.num_match_target_bits,
            d.num_buckets,
            d_offsets);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // One thread per (r_bucket, fine_key). At T3 k=28 strength=2:
    // 16 × 256 = 4096 threads = 16 blocks × 256.
    uint32_t fine_threads_total = d.num_buckets * uint32_t(1ull << kT3FineBits);
    unsigned fine_blocks = (fine_threads_total + 255) / 256;
    compute_fine_bucket_offsets<<<fine_blocks, 256, 0, stream>>>(
        d_sorted_mi, d_offsets,
        params.num_match_target_bits, kT3FineBits,
        d.num_buckets, d_fine_offsets);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    return cudaMemsetAsync(d_out_count, 0, sizeof(uint64_t), stream);
}

cudaError_t launch_t3_match_range(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t /*t2_count*/,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    cudaStream_t stream)
{
    if (!plot_id_bytes || !d_temp_storage)  return cudaErrorInvalidValue;
    if (params.k < 18 || params.k > 32)     return cudaErrorInvalidValue;
    if (params.strength < 2)                return cudaErrorInvalidValue;
    if (!d_sorted_meta || !d_sorted_xbits || !d_sorted_mi
        || !d_out_pairings || !d_out_count) return cudaErrorInvalidValue;

    T3Derived const d = derive_t3(params);
    if (bucket_end > d.num_buckets) return cudaErrorInvalidValue;
    if (bucket_end <= bucket_begin) return cudaSuccess;

    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (d.l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) return cudaErrorInvalidValue;
    dim3 grid(static_cast<unsigned>(blocks_x_u64), num_buckets_in_range, 1);

    auto const* d_offsets      = reinterpret_cast<uint64_t const*>(d_temp_storage);
    auto const* d_fine_offsets = d_offsets + (d.num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);

    match_all_buckets<<<grid, kThreads, 0, stream>>>(
        keys, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
        d_offsets, d_fine_offsets,
        d.num_match_keys,
        params.k, params.num_section_bits,
        params.num_match_target_bits, kT3FineBits,
        d.target_mask, d.num_test_bits,
        d_out_pairings,
        reinterpret_cast<unsigned long long*>(d_out_count),
        capacity,
        bucket_begin,
        /*meta_l_index_bias=*/0, /*meta_r_index_bias=*/0);
    return cudaGetLastError();
}

// Cut #3: section-pair input-slicing variant. d_sorted_meta points to
// a buffer holding the section_l row + section_r row (concatenated)
// rather than the full-cap sorted meta. d_sorted_xbits and d_sorted_mi
// stay full-cap (cheaper to keep on device than to slice). The biases
// remap kernel-internal global indices into the slice's positions.
//
// Caller computes: section_l = bucket_begin / num_match_keys (must
// equal (bucket_end-1) / num_match_keys, i.e. one section_l per call),
// section_r = matching_section(section_l), and looks up the row
// boundaries from the SAME d_offsets the prepare phase built. The
// slice contains [section_l_row | section_r_row] tightly packed at
// indices 0 and section_l_row_count.
cudaError_t launch_t3_match_section_pair_range(
    uint8_t const* plot_id_bytes,
    T3MatchParams const& params,
    uint64_t const* d_sorted_meta_slice,    // section_l + section_r rows, packed
    uint32_t const* d_sorted_xbits,         // full cap
    uint32_t const* d_sorted_mi,            // full cap
    uint64_t /*t2_count*/,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t capacity,
    void const* d_temp_storage,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    int64_t meta_l_index_bias,
    int64_t meta_r_index_bias,
    cudaStream_t stream)
{
    if (!plot_id_bytes || !d_temp_storage)  return cudaErrorInvalidValue;
    if (params.k < 18 || params.k > 32)     return cudaErrorInvalidValue;
    if (params.strength < 2)                return cudaErrorInvalidValue;
    if (!d_sorted_meta_slice || !d_sorted_xbits || !d_sorted_mi
        || !d_out_pairings || !d_out_count) return cudaErrorInvalidValue;

    T3Derived const d = derive_t3(params);
    if (bucket_end > d.num_buckets) return cudaErrorInvalidValue;
    if (bucket_end <= bucket_begin) return cudaSuccess;

    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (d.l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) return cudaErrorInvalidValue;
    dim3 grid(static_cast<unsigned>(blocks_x_u64), num_buckets_in_range, 1);

    auto const* d_offsets      = reinterpret_cast<uint64_t const*>(d_temp_storage);
    auto const* d_fine_offsets = d_offsets + (d.num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);

    match_all_buckets<<<grid, kThreads, 0, stream>>>(
        keys, d_sorted_meta_slice, d_sorted_xbits, d_sorted_mi,
        d_offsets, d_fine_offsets,
        d.num_match_keys,
        params.k, params.num_section_bits,
        params.num_match_target_bits, kT3FineBits,
        d.target_mask, d.num_test_bits,
        d_out_pairings,
        reinterpret_cast<unsigned long long*>(d_out_count),
        capacity,
        bucket_begin,
        meta_l_index_bias, meta_r_index_bias);
    return cudaGetLastError();
}

cudaError_t launch_t3_match(
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
    cudaStream_t stream)
{
    // Single-shot wrapper: prepare + one full-range match. Preserves
    // the original API for the pool path and parity tests.
    cudaError_t err = launch_t3_match_prepare(
        plot_id_bytes, params, d_sorted_mi, t2_count,
        d_out_count, d_temp_storage, temp_bytes, stream);
    if (err != cudaSuccess) return err;
    if (d_temp_storage == nullptr) return cudaSuccess;  // size-query path

    T3Derived const d = derive_t3(params);
    return launch_t3_match_range(
        plot_id_bytes, params,
        d_sorted_meta, d_sorted_xbits, d_sorted_mi, t2_count,
        d_out_pairings, d_out_count,
        capacity, d_temp_storage,
        /*bucket_begin=*/0, /*bucket_end=*/d.num_buckets, stream);
}

uint32_t matching_section_host(uint32_t section_l, int num_section_bits)
{
    // Same math as the anonymous matching_section helper above, but
    // exposed for host-side computation (streaming pipeline needs to
    // know section_r before launching the per-section_l T3 pass).
    uint32_t const num_sections = 1u << num_section_bits;
    uint32_t const mask         = num_sections - 1u;
    uint32_t const rl  = ((section_l << 1) | (section_l >> (num_section_bits - 1))) & mask;
    uint32_t const rl1 = (rl + 1) & mask;
    return ((rl1 >> 1) | (rl1 << (num_section_bits - 1))) & mask;
}

} // namespace pos2gpu
