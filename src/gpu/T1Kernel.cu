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

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/T1Kernel.cuh"

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <vector>

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

namespace {

// Mirrors pos2-chip/src/pos/ProofCore.hpp:198 matching_section.
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
    XsCandidateGpu const* __restrict__ sorted,
    uint64_t total,
    int num_match_target_bits, // bucket id = match_info >> num_match_target_bits
    uint32_t num_buckets,      // num_sections * num_match_keys
    uint64_t* __restrict__ offsets) // offsets[num_buckets + 1]
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

// Fused match kernel: handles all (section_l, match_key_r) buckets in a
// single launch. blockIdx.y identifies the bucket, blockIdx.x slices L.
// Loads AES T-tables into shared memory once per block.
__global__ void match_all_buckets(
    AesHashKeys keys,
    XsCandidateGpu const* __restrict__ sorted_xs,
    uint64_t const* __restrict__ d_offsets, // [num_buckets+1]
    uint32_t num_match_keys,
    int k,
    int num_section_bits,
    int extra_rounds_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    T1PairingGpu* __restrict__ out,
    unsigned long long* __restrict__ out_count,
    uint64_t out_capacity)
{
    __shared__ uint32_t sT[4 * 256];
    load_aes_tables_smem(sT);
    __syncthreads();

    uint32_t bucket_id   = blockIdx.y;            // 0..num_buckets
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

    uint32_t x_l = sorted_xs[l].x;

    uint32_t target_l = matching_target_smem(keys, 1u, match_key_r, uint64_t(x_l), sT, 0)
                      & target_mask;

    uint64_t lo = r_start, hi = r_end;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        uint32_t target_mid = sorted_xs[mid].match_info & target_mask;
        if (target_mid < target_l) lo = mid + 1;
        else                       hi = mid;
    }

    uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                : ((1u << num_test_bits) - 1u);
    uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                     : ((1u << num_match_info_bits) - 1u);

    for (uint64_t r = lo; r < r_end; ++r) {
        uint32_t target_r = sorted_xs[r].match_info & target_mask;
        if (target_r != target_l) break;

        uint32_t x_r = sorted_xs[r].x;
        Result128 res = pairing_smem(keys, uint64_t(x_l), uint64_t(x_r), sT, extra_rounds_bits);

        uint32_t test_result = res.r[3] & test_mask;
        if (test_result != 0) continue;

        uint32_t match_info_result = res.r[0] & info_mask;

        unsigned long long out_idx = atomicAdd(out_count, 1ULL);
        if (out_idx >= out_capacity) return;

        uint64_t meta = (uint64_t(x_l) << k) | uint64_t(x_r);
        T1PairingGpu p;
        p.meta_lo    = uint32_t(meta);
        p.meta_hi    = uint32_t(meta >> 32);
        p.match_info = match_info_result;
        out[out_idx] = p;
    }
}

} // namespace

cudaError_t launch_t1_match(
    uint8_t const* plot_id_bytes,
    T1MatchParams const& params,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t total,
    T1PairingGpu* d_out_pairings,
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

    // temp layout: offsets[num_buckets + 1] uint64
    size_t needed = sizeof(uint64_t) * (num_buckets + 1);

    if (d_temp_storage == nullptr) {
        *temp_bytes = needed;
        return cudaSuccess;
    }
    if (*temp_bytes < needed)        return cudaErrorInvalidValue;
    if (!d_sorted_xs || !d_out_pairings || !d_out_count) return cudaErrorInvalidValue;

    auto* d_offsets = reinterpret_cast<uint64_t*>(d_temp_storage);

    AesHashKeys keys = make_keys(plot_id_bytes);

    // 1) Bucket offsets.
    compute_bucket_offsets<<<1, 1, 0, stream>>>(
        d_sorted_xs, total,
        params.num_match_target_bits,
        num_buckets,
        d_offsets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Reset out_count to 0.
    err = cudaMemsetAsync(d_out_count, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    // 2) Compute max L-count across sections (small H2D copy only for sizing).
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
    int extra_rounds_bits = params.strength - 2;
    int num_test_bits     = params.num_match_key_bits;
    int num_info_bits     = params.k;

    constexpr int kThreads = 256;
    uint64_t blocks_x_u64 = (l_count_max + kThreads - 1) / kThreads;
    if (blocks_x_u64 > UINT_MAX) return cudaErrorInvalidValue;
    dim3 grid(static_cast<unsigned>(blocks_x_u64), num_buckets, 1);

    match_all_buckets<<<grid, kThreads, 0, stream>>>(
        keys, d_sorted_xs, d_offsets,
        num_match_keys,
        params.k, params.num_section_bits,
        extra_rounds_bits, target_mask,
        num_test_bits, num_info_bits,
        d_out_pairings,
        reinterpret_cast<unsigned long long*>(d_out_count),
        capacity);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

} // namespace pos2gpu
