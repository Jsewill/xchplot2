// T3OffsetsSycl.cpp — SYCL implementation of T3's match kernel. Mirrors
// the CUDA path; FeistelKey (40 B) is captured by value in the parallel_for
// lambda instead of going through CUDA constant memory. AdaptiveCpp's
// SSCP backend handles the capture via the kernel-arg mechanism, which is
// fine at this size — if local-memory spills ever bite, switch to a USM
// upload analogous to the CUDA cudaMemcpyToSymbolAsync path.

#include "gpu/SyclBackend.hpp"
#include "gpu/T3Offsets.cuh"

#include <sycl/sycl.hpp>

namespace pos2gpu {

void launch_t3_match_all_buckets(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    uint32_t target_mask,
    int num_test_bits,
    T3PairingGpu* d_out_pairings,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    sycl::queue& q)
{
    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads  = 256;
    // Coarsening factor: see T1OffsetsSycl.cpp for rationale.
    constexpr int    kCoarsen = 2;
    uint64_t blocks_x_u64 =
        (l_count_max + threads * kCoarsen - 1) / (threads * kCoarsen);
    size_t   const blocks_x  = static_cast<size_t>(blocks_x_u64);

    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> sT_local{
            sycl::range<1>{4 * 256}, h};

        h.parallel_for(
            sycl::nd_range<2>{
                sycl::range<2>{ static_cast<size_t>(num_buckets),
                                blocks_x * threads },
                sycl::range<2>{ 1, threads }
            },
            [=, keys_copy = keys, fk_copy = fk](sycl::nd_item<2> it) {
                uint32_t* sT = &sT_local[0];
                size_t local_id = it.get_local_id(1);
                #pragma unroll 1
                for (size_t i = local_id; i < 4 * 256; i += threads) {
                    sT[i] = d_aes_tables[i];
                }
                it.barrier(sycl::access::fence_space::local_space);

                uint32_t bucket_id   = static_cast<uint32_t>(it.get_group(0));
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

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);

                uint64_t const l_group_base = l_start
                    + it.get_group(1) * uint64_t(threads * kCoarsen);
                #pragma unroll
                for (int c = 0; c < kCoarsen; ++c) {
                    uint64_t l = l_group_base + uint64_t(c) * threads + local_id;
                    if (l >= l_end) break;

                    uint64_t meta_l = d_sorted_meta[l];
                    uint32_t xb_l   = d_sorted_xbits[l];

                    uint32_t target_l = pos2gpu::matching_target_smem(
                                            keys_copy, 3u, match_key_r, meta_l, sT, 0)
                                      & target_mask;

                    uint32_t fine_key = target_l >> fine_shift;
                    uint64_t fine_idx = (uint64_t(r_bucket) << fine_bits) | fine_key;
                    uint64_t lo       = d_fine_offsets[fine_idx];
                    uint64_t fine_hi  = d_fine_offsets[fine_idx + 1];
                    uint64_t hi       = fine_hi;

                    while (lo < hi) {
                        uint64_t mid = lo + ((hi - lo) >> 1);
                        uint32_t target_mid = d_sorted_mi[mid] & target_mask;
                        if (target_mid < target_l) lo = mid + 1;
                        else                       hi = mid;
                    }

                    for (uint64_t r = lo; r < fine_hi; ++r) {
                        uint32_t target_r = d_sorted_mi[r] & target_mask;
                        if (target_r != target_l) break;

                        uint64_t meta_r = d_sorted_meta[r];
                        uint32_t xb_r   = d_sorted_xbits[r];

                        pos2gpu::Result128 res = pos2gpu::pairing_smem(
                            keys_copy, meta_l, meta_r, sT, 0);
                        uint32_t test_result = res.r[3] & test_mask;
                        if (test_result != 0) continue;

                        uint64_t all_x_bits = (uint64_t(xb_l) << k) | uint64_t(xb_r);
                        uint64_t fragment   = pos2gpu::feistel_encrypt(fk_copy, all_x_bits);

                        sycl::atomic_ref<unsigned long long,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>
                            out_count_atomic{ *d_out_count_ull };
                        unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                        if (out_idx >= out_capacity) return;

                        T3PairingGpu p;
                        p.proof_fragment = fragment;
                        d_out_pairings[out_idx] = p;
                    }
                }
            });
    }).wait();
}

} // namespace pos2gpu
