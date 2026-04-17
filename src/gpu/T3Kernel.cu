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
    T2PairingGpu const* __restrict__ sorted,
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
            uint32_t bucket_mid = sorted[mid].match_info >> bucket_shift;
            if (bucket_mid < b) lo = mid + 1;
            else                hi = mid;
        }
        offsets[b] = lo;
        pos = lo;
    }
    offsets[num_buckets] = total;
}

__global__ void match_one_bucket(
    AesHashKeys keys,
    FeistelKey fk,
    T2PairingGpu const* __restrict__ sorted_t2,
    uint64_t l_start, uint64_t l_end,
    uint64_t r_start, uint64_t r_end,
    uint32_t match_key_r,
    int k,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* __restrict__ out,
    unsigned long long* __restrict__ out_count,
    uint64_t out_capacity)
{
    uint64_t l = l_start + blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (l >= l_end) return;

    uint64_t meta_l = sorted_t2[l].meta;
    uint32_t xb_l   = sorted_t2[l].x_bits;

    uint32_t target_l = matching_target(keys, 3u, match_key_r, meta_l, 0)
                      & target_mask;

    uint64_t lo = r_start, hi = r_end;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t target_mid = sorted_t2[mid].match_info & target_mask;
        if (target_mid < target_l) lo = mid + 1;
        else                       hi = mid;
    }

    uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                : ((1u << num_test_bits) - 1u);

    for (uint64_t r = lo; r < r_end; ++r) {
        uint32_t target_r = sorted_t2[r].match_info & target_mask;
        if (target_r != target_l) break;

        uint64_t meta_r = sorted_t2[r].meta;
        uint32_t xb_r   = sorted_t2[r].x_bits;

        // pairing_t3: AES pairing with extra_rounds_bits=0; only test_result used.
        Result128 res = pairing(keys, meta_l, meta_r, 0);
        uint32_t test_result = res.r[3] & test_mask;
        if (test_result != 0) continue;

        // proof_fragment = FeistelCipher.encrypt((xb_l << k) | xb_r)
        uint64_t all_x_bits = (uint64_t(xb_l) << k) | uint64_t(xb_r);
        uint64_t fragment   = feistel_encrypt(fk, all_x_bits);

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
    T2PairingGpu const* d_sorted_t2,
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

    size_t needed = sizeof(uint64_t) * (num_buckets + 1);

    if (d_temp_storage == nullptr) {
        *temp_bytes = needed;
        return cudaSuccess;
    }
    if (*temp_bytes < needed)        return cudaErrorInvalidValue;
    if (!d_sorted_t2 || !d_out_pairings || !d_out_count) return cudaErrorInvalidValue;

    auto* d_offsets = reinterpret_cast<uint64_t*>(d_temp_storage);

    AesHashKeys keys = make_keys(plot_id_bytes);
    FeistelKey  fk   = make_feistel_key(plot_id_bytes, params.k, /*rounds=*/4);

    compute_bucket_offsets<<<1, 1, 0, stream>>>(
        d_sorted_t2, t2_count,
        params.num_match_target_bits,
        num_buckets,
        d_offsets);
    cudaError_t err = cudaGetLastError();
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

    uint32_t target_mask = (params.num_match_target_bits >= 32)
                            ? 0xFFFFFFFFu
                            : ((1u << params.num_match_target_bits) - 1u);
    int num_test_bits = params.num_match_key_bits; // = strength

    constexpr int kThreads = 256;

    for (uint32_t section_l = 0; section_l < num_sections; ++section_l) {
        uint32_t section_r = matching_section(section_l, params.num_section_bits);
        uint64_t l_start = h_offsets[section_l * num_match_keys];
        uint64_t l_end   = h_offsets[(section_l + 1) * num_match_keys];
        if (l_end <= l_start) continue;
        uint64_t l_count = l_end - l_start;

        uint64_t blocks_u64 = (l_count + kThreads - 1) / kThreads;
        if (blocks_u64 > UINT_MAX) return cudaErrorInvalidValue;
        unsigned blocks = static_cast<unsigned>(blocks_u64);

        for (uint32_t match_key_r = 0; match_key_r < num_match_keys; ++match_key_r) {
            uint64_t r_start = h_offsets[section_r * num_match_keys + match_key_r];
            uint64_t r_end   = h_offsets[section_r * num_match_keys + match_key_r + 1];
            if (r_end <= r_start) continue;

            match_one_bucket<<<blocks, kThreads, 0, stream>>>(
                keys, fk, d_sorted_t2,
                l_start, l_end,
                r_start, r_end,
                match_key_r,
                params.k,
                target_mask,
                num_test_bits,
                d_out_pairings,
                reinterpret_cast<unsigned long long*>(d_out_count),
                capacity);
            err = cudaGetLastError();
            if (err != cudaSuccess) return err;
        }
    }
    return cudaSuccess;
}

} // namespace pos2gpu
