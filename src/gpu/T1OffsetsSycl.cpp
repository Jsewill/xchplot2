// T1OffsetsSycl.cpp — SYCL/AdaptiveCpp implementation of
// launch_compute_bucket_offsets, selected when XCHPLOT2_BACKEND=sycl.
//
// Same algorithm and output layout as T1OffsetsCuda.cu. The SYCL queue
// uses AdaptiveCpp's CUDA backend (gpu_selector picks the RTX 4090 in
// our test bench), which uses libcuda directly and shares the primary
// CUDA context with the rest of the pipeline — so raw CUDA device
// pointers from cudaMalloc are valid USM device pointers in the SYCL
// kernel without any copy or remap.
//
// Synchronisation: the function syncs `stream` before launching SYCL
// (so prior CUDA writes to d_sorted are visible) and waits for the
// SYCL queue after (so subsequent CUDA reads of d_offsets see the
// SYCL writes). Two extra host syncs vs. the pure-CUDA path; not
// perf-relevant for slice 2.

#include "gpu/AesHashBsSycl.hpp"
#include "gpu/SyclBackend.hpp"
#include "gpu/T1Offsets.cuh"

#include <sycl/sycl.hpp>

namespace pos2gpu {


void launch_compute_bucket_offsets(
    XsCandidateGpu const* d_sorted,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* d_offsets,
    sycl::queue& q)
{
    constexpr size_t threads   = 256;
    size_t   const out_count   = static_cast<size_t>(num_buckets) + 1;
    size_t   const groups      = (out_count + threads - 1) / threads;

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            uint32_t b = static_cast<uint32_t>(it.get_global_id(0));
            if (b > num_buckets) return;
            if (b == num_buckets) { d_offsets[num_buckets] = total; return; }

            uint32_t bucket_shift = static_cast<uint32_t>(num_match_target_bits);
            uint64_t lo = 0, hi = total;
            while (lo < hi) {
                uint64_t mid = lo + ((hi - lo) >> 1);
                uint32_t v   = d_sorted[mid].match_info >> bucket_shift;
                if (v < b) lo = mid + 1;
                else       hi = mid;
            }
            d_offsets[b] = lo;
        }).wait();
}

void launch_compute_fine_bucket_offsets(
    XsCandidateGpu const* d_sorted,
    uint64_t const* d_bucket_offsets,
    int num_match_target_bits,
    int fine_bits,
    uint32_t num_buckets,
    uint64_t* d_fine_offsets,
    sycl::queue& q)
{
    constexpr size_t threads      = 256;
    uint32_t const   fine_count   = 1u << fine_bits;
    uint32_t const   total        = num_buckets * fine_count;
    size_t   const   groups       = (total + threads - 1) / threads;

    q.parallel_for(
        sycl::nd_range<1>{ groups * threads, threads },
        [=](sycl::nd_item<1> it) {
            uint32_t tid = static_cast<uint32_t>(it.get_global_id(0));
            if (tid >= total) return;

            uint32_t r_bucket = tid / fine_count;
            uint32_t fine_key = tid % fine_count;

            uint64_t r_start = d_bucket_offsets[r_bucket];
            uint64_t r_end   = d_bucket_offsets[r_bucket + 1];

            uint32_t target_mask = (num_match_target_bits >= 32)
                                    ? 0xFFFFFFFFu
                                    : ((1u << num_match_target_bits) - 1u);
            uint32_t shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);

            uint64_t lo = r_start, hi = r_end;
            while (lo < hi) {
                uint64_t mid = lo + ((hi - lo) >> 1);
                uint32_t t   = (d_sorted[mid].match_info & target_mask) >> shift;
                if (t < fine_key) lo = mid + 1;
                else              hi = mid;
            }
            d_fine_offsets[tid] = lo;

            if (tid == total - 1) {
                d_fine_offsets[total] = d_bucket_offsets[num_buckets];
            }
        }).wait();
}

void launch_t1_match_all_buckets(
    AesHashKeys keys,
    XsCandidateGpu const* d_sorted_xs,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
    uint32_t num_buckets,
    int k,
    int num_section_bits,
    int num_match_target_bits,
    int fine_bits,
    int extra_rounds_bits,
    uint32_t target_mask,
    int num_test_bits,
    int num_match_info_bits,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    sycl::queue& q)
{
    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    constexpr size_t threads = 256;
    uint64_t blocks_x_u64    = (l_count_max + threads - 1) / threads;
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
            [=, keys_copy = keys](sycl::nd_item<2> it)
                [[sycl::reqd_sub_group_size(32)]]
            {
                // T-tables are still loaded because the inner pairing loop
                // is T-table-based (variable trip count per lane). Only the
                // outer matching_target has been lifted to the sub_group
                // bitsliced path — that call is sub_group-uniform so all 32
                // lanes can cooperate on 32 matching_target hashes at once.
                uint32_t* sT = &sT_local[0];
                size_t local_id = it.get_local_id(1);
                #pragma unroll 1
                for (size_t i = local_id; i < 4 * 256; i += threads) {
                    sT[i] = d_aes_tables[i];
                }
                it.barrier(sycl::access::fence_space::local_space);

                auto sg = it.get_sub_group();

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

                uint64_t l = l_start
                           + it.get_group(1) * uint64_t(threads)
                           + local_id;
                bool in_range = (l < l_end);

                // All 32 lanes participate in the bitsliced matching_target;
                // out-of-range lanes feed dummy x_l. Result is discarded
                // below via the `if (!in_range) return;` early-exit.
                uint32_t x_l = in_range ? d_sorted_xs[l].x : 0u;

                uint32_t target_l = pos2gpu::matching_target_bs32(
                                        sg, keys_copy, 1u, match_key_r, uint64_t(x_l),
                                        extra_rounds_bits)
                                  & target_mask;

                if (!in_range) return;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t lo         = d_fine_offsets[fine_idx];
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t hi         = fine_hi;

                while (lo < hi) {
                    uint64_t mid = lo + ((hi - lo) >> 1);
                    uint32_t target_mid = d_sorted_xs[mid].match_info & target_mask;
                    if (target_mid < target_l) lo = mid + 1;
                    else                       hi = mid;
                }

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint32_t target_r = d_sorted_xs[r].match_info & target_mask;
                    if (target_r != target_l) break;

                    uint32_t x_r = d_sorted_xs[r].x;
                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, uint64_t(x_l), uint64_t(x_r), sT, extra_rounds_bits);

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) continue;

                    uint32_t match_info_result = res.r[0] & info_mask;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    uint64_t meta = (uint64_t(x_l) << k) | uint64_t(x_r);
                    d_out_meta[out_idx] = meta;
                    d_out_mi  [out_idx] = match_info_result;
                }
            });
    }).wait();
}

} // namespace pos2gpu
