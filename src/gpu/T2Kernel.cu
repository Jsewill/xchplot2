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

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <vector>

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
    T1PairingGpu const* __restrict__ sorted,
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

__global__ void match_all_buckets(
    AesHashKeys keys,
    T1PairingGpu const* __restrict__ sorted_t1,
    uint64_t const* __restrict__ d_offsets,
    uint32_t num_match_keys,
    int k,
    int num_section_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    int half_k,
    T2PairingGpu* __restrict__ out,
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
    uint64_t r_start = d_offsets[section_r * num_match_keys + match_key_r];
    uint64_t r_end   = d_offsets[section_r * num_match_keys + match_key_r + 1];

    uint64_t l = l_start + blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (l >= l_end) return;

    uint64_t meta_l = (uint64_t(sorted_t1[l].meta_hi) << 32)
                    | uint64_t(sorted_t1[l].meta_lo);

    uint32_t target_l = matching_target_smem(keys, 2u, match_key_r, meta_l, sT, 0)
                      & target_mask;

    uint64_t lo = r_start, hi = r_end;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t target_mid = sorted_t1[mid].match_info & target_mask;
        if (target_mid < target_l) lo = mid + 1;
        else                       hi = mid;
    }

    uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                : ((1u << num_test_bits) - 1u);
    uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                     : ((1u << num_match_info_bits) - 1u);
    int meta_bits = 2 * k;

    for (uint64_t r = lo; r < r_end; ++r) {
        uint32_t target_r = sorted_t1[r].match_info & target_mask;
        if (target_r != target_l) break;

        uint64_t meta_r = (uint64_t(sorted_t1[r].meta_hi) << 32)
                        | uint64_t(sorted_t1[r].meta_lo);

        Result128 res = pairing_smem(keys, meta_l, meta_r, sT, 0);

        uint32_t test_result = res.r[3] & test_mask;
        if (test_result != 0) continue;

        uint32_t match_info_result = res.r[0] & info_mask;
        uint64_t meta_result_full = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
        uint64_t meta_result = (meta_bits == 64)
                               ? meta_result_full
                               : (meta_result_full & ((1ULL << meta_bits) - 1ULL));

        uint32_t x_bits_l = static_cast<uint32_t>((meta_l >> k) >> half_k);
        uint32_t x_bits_r = static_cast<uint32_t>((meta_r >> k) >> half_k);
        uint32_t x_bits   = (x_bits_l << half_k) | x_bits_r;

        unsigned long long out_idx = atomicAdd(out_count, 1ULL);
        if (out_idx >= out_capacity) return;

        T2PairingGpu p;
        p.meta       = meta_result;
        p.match_info = match_info_result;
        p.x_bits     = x_bits;
        out[out_idx] = p;
    }
}

} // namespace

cudaError_t launch_t2_match(
    uint8_t const* plot_id_bytes,
    T2MatchParams const& params,
    T1PairingGpu const* d_sorted_t1,
    uint64_t t1_count,
    T2PairingGpu* d_out_pairings,
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
    if (!d_sorted_t1 || !d_out_pairings || !d_out_count) return cudaErrorInvalidValue;

    auto* d_offsets = reinterpret_cast<uint64_t*>(d_temp_storage);

    AesHashKeys keys = make_keys(plot_id_bytes);

    compute_bucket_offsets<<<1, 1, 0, stream>>>(
        d_sorted_t1, t1_count,
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
    int num_info_bits = params.k;
    int half_k        = params.k / 2;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) return cudaErrorInvalidValue;
    dim3 grid(static_cast<unsigned>(blocks_x_u64), num_buckets, 1);

    match_all_buckets<<<grid, kThreads, 0, stream>>>(
        keys, d_sorted_t1, d_offsets,
        num_match_keys,
        params.k, params.num_section_bits,
        target_mask, num_test_bits, num_info_bits, half_k,
        d_out_pairings,
        reinterpret_cast<unsigned long long*>(d_out_count),
        capacity);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

} // namespace pos2gpu
