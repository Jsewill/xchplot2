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

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <vector>

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

__global__ void compute_bucket_offsets(
    uint32_t const* __restrict__ sorted_mi,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* __restrict__ offsets)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t bucket_shift = static_cast<uint32_t>(num_match_target_bits);

    uint64_t pos = 0;
    for (uint32_t b = 0; b < num_buckets; ++b) {
        uint64_t lo = pos, hi = total;
        while (lo < hi) {
            uint64_t mid = lo + ((hi - lo) >> 1);
            uint32_t bucket_mid = sorted_mi[mid] >> bucket_shift;
            if (bucket_mid < b) lo = mid + 1;
            else                hi = mid;
        }
        offsets[b] = lo;
        pos = lo;
    }
    offsets[num_buckets] = total;
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
    uint64_t out_capacity)
{
    __shared__ uint32_t sT[4 * 256];
    load_aes_tables_smem(sT);
    __syncthreads();

    uint32_t bucket_id   = blockIdx.y;
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

    uint64_t meta_l = sorted_meta[l];
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

        uint64_t meta_r = sorted_meta[r];
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
    if (!plot_id_bytes || !temp_bytes) return cudaErrorInvalidValue;
    if (params.k < 18 || params.k > 32) return cudaErrorInvalidValue;
    if (params.strength < 2)            return cudaErrorInvalidValue;

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
        return cudaSuccess;
    }
    if (*temp_bytes < needed)        return cudaErrorInvalidValue;
    if (!d_sorted_meta || !d_sorted_xbits || !d_sorted_mi
        || !d_out_pairings || !d_out_count) return cudaErrorInvalidValue;
    if (params.num_match_target_bits <= FINE_BITS) {
        // Fall-back would be needed here; not expected for supported
        // (k, strength) combinations, so fail loudly if we ever trip it.
        return cudaErrorInvalidValue;
    }

    auto* d_offsets      = reinterpret_cast<uint64_t*>(d_temp_storage);
    auto* d_fine_offsets = d_offsets + (num_buckets + 1);

    AesHashKeys keys = make_keys(plot_id_bytes);
    FeistelKey  fk   = make_feistel_key(plot_id_bytes, params.k, /*rounds=*/4);
    cudaError_t fk_err = cudaMemcpyToSymbolAsync(g_t3_fk, &fk, sizeof(fk),
                                                 0, cudaMemcpyHostToDevice, stream);
    if (fk_err != cudaSuccess) return fk_err;

    compute_bucket_offsets<<<1, 1, 0, stream>>>(
        d_sorted_mi, t2_count,
        params.num_match_target_bits,
        num_buckets,
        d_offsets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // One thread per (r_bucket, fine_key). At T3 k=28 strength=2:
    // 16 × 256 = 4096 threads = 16 blocks × 256.
    uint32_t fine_threads_total = num_buckets * uint32_t(fine_count);
    unsigned fine_blocks = (fine_threads_total + 255) / 256;
    compute_fine_bucket_offsets<<<fine_blocks, 256, 0, stream>>>(
        d_sorted_mi, d_offsets,
        params.num_match_target_bits, FINE_BITS,
        num_buckets, d_fine_offsets);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(d_out_count, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    std::vector<uint64_t> h_offsets(num_buckets + 1);
    err = cudaMemcpyAsync(h_offsets.data(), d_offsets,
                          sizeof(uint64_t) * (num_buckets + 1),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    uint64_t l_count_max = 0;
    for (uint32_t s = 0; s < num_sections; ++s) {
        uint64_t l_count = h_offsets[(s + 1) * num_match_keys]
                         - h_offsets[s * num_match_keys];
        if (l_count > l_count_max) l_count_max = l_count;
    }

    uint32_t target_mask = (params.num_match_target_bits >= 32)
                            ? 0xFFFFFFFFu
                            : ((1u << params.num_match_target_bits) - 1u);
    int num_test_bits = params.num_match_key_bits;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) return cudaErrorInvalidValue;
    dim3 grid(static_cast<unsigned>(blocks_x_u64), num_buckets, 1);

    match_all_buckets<<<grid, kThreads, 0, stream>>>(
        keys, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
        d_offsets, d_fine_offsets,
        num_match_keys,
        params.k, params.num_section_bits,
        params.num_match_target_bits, FINE_BITS,
        target_mask, num_test_bits,
        d_out_pairings,
        reinterpret_cast<unsigned long long*>(d_out_count),
        capacity);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

} // namespace pos2gpu
