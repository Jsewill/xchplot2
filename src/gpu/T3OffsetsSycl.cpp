// T3OffsetsSycl.cpp — SYCL implementation of T3's match kernel. Mirrors
// the CUDA path; FeistelKey (40 B) is captured by value in the parallel_for
// lambda instead of going through CUDA constant memory. AdaptiveCpp's
// SSCP backend handles the capture via the kernel-arg mechanism, which is
// fine at this size — if local-memory spills ever bite, switch to a USM
// upload analogous to the CUDA cudaMemcpyToSymbolAsync path.

#include "gpu/MatchKernelCommon.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/T3Offsets.cuh"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace pos2gpu {

namespace {

// ---- Phase 2 prototype: two-phase T3 match ----------------------------
//
// ncu on the single-kernel launch_t3_match_all_buckets showed ~61%
// Speed-of-Light, latency-bound, with only ~12.5 of 32 warp lanes active:
// each thread runs one l-entry through a *variable-length* inner r-loop,
// and the expensive pairing AES (32 rounds) + Feistel sit inside that
// divergent loop. This splits the work in two:
//
//   Phase A — enumerate: one thread per l-entry. Does matching_target
//     AES + binary search + the r-loop, but only EMITS {l, r} candidate
//     index pairs (8 B each) to a scratch buffer. No pairing/Feistel.
//     Still divergent in the r-loop, but the loop body is now a cheap
//     atomic store, not 32 AES rounds.
//   Phase B — process: one thread per candidate pair. Re-gathers
//     meta/xbits, runs the pairing AES + test + Feistel + output write.
//     Fully convergent — every thread does exactly one pairing.
//
// Gated behind POS2GPU_T3_TWOPHASE=1. Prototype scope: tiled one bucket
// at a time so the candidate scratch stays small (~1 GB at k=28) and the
// path works regardless of card size; the scratch is malloc'd here, not
// pool-integrated. Output order still races on the atomic, which is fine
// — T3 output is sorted by proof_fragment afterward.
void run_t3_match_twophase(
    AesHashKeys const& keys,
    FeistelKey const& fk,
    uint64_t const* d_sorted_meta,
    uint32_t const* d_sorted_xbits,
    uint32_t const* d_sorted_mi,
    uint64_t const* d_offsets,
    uint64_t const* d_fine_offsets,
    uint32_t num_match_keys,
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
    uint32_t bucket_begin,
    uint32_t bucket_end,
    uint32_t* d_aes_tables,
    sycl::queue& q)
{
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;
    constexpr size_t threads = 256;
    size_t const blocks_x = static_cast<size_t>(
        (l_count_max + threads - 1) / threads);

    bool const dbg = [] {
        char const* v = std::getenv("POS2GPU_T3_TWOPHASE_DEBUG");
        return v && v[0] == '1';
    }();

    // Per-bucket candidate scratch: {u32 l, u32 r} per pair. The T3 test
    // filter passes ~1/4, so total candidates ≈ 4× outputs. Sized at 8×
    // the average per-bucket output count → ~2× headroom over the
    // expected per-bucket candidate count, absorbing inter-bucket
    // variance. Allocated once, reused across all buckets in the range.
    uint64_t const cand_cap =
        (out_capacity / (num_buckets_in_range ? num_buckets_in_range : 1)) * 8
        + 4096;
    uint32_t* d_cand_l     = sycl::malloc_device<uint32_t>(cand_cap, q);
    uint32_t* d_cand_r     = sycl::malloc_device<uint32_t>(cand_cap, q);
    uint64_t* d_cand_count = sycl::malloc_device<uint64_t>(1, q);
    if (!d_cand_l || !d_cand_r || !d_cand_count) {
        if (d_cand_l)     sycl::free(d_cand_l, q);
        if (d_cand_r)     sycl::free(d_cand_r, q);
        if (d_cand_count) sycl::free(d_cand_count, q);
        throw std::runtime_error("T3 two-phase: candidate buffer alloc failed");
    }
    auto* d_cand_count_ull =
        reinterpret_cast<unsigned long long*>(d_cand_count);
    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    // Process one bucket per iteration: Phase A enumerates that bucket's
    // candidate pairs, Phase B drains them into the shared output.
    for (uint32_t b = bucket_begin; b < bucket_end; ++b) {
        q.memset(d_cand_count, 0, sizeof(uint64_t)).wait();

        // ---- Phase A: enumerate candidate (l, r) index pairs ----
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> sT_local{
                sycl::range<1>{4 * 256}, h};
            h.parallel_for(
                sycl::nd_range<1>{ blocks_x * threads, threads },
                [=, keys_copy = keys](sycl::nd_item<1> it) {
                    uint32_t* sT = &sT_local[0];
                    size_t local_id = it.get_local_id(0);
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    uint32_t bucket_id   = b;
                    uint32_t section_l   = bucket_id / num_match_keys;
                    uint32_t match_key_r = bucket_id % num_match_keys;

                    uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                    uint64_t l_start = d_offsets[section_l * num_match_keys];
                    uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                    uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                    uint64_t l = l_start + it.get_global_id(0);
                    if (l >= l_end) return;

                    uint64_t meta_l = d_sorted_meta[l];
                    uint32_t target_l = pos2gpu::matching_target_smem(
                                            keys_copy, 3u, match_key_r, meta_l, sT, 0)
                                      & target_mask;

                    uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                    uint32_t fine_key   = target_l >> fine_shift;
                    uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                    uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                    uint64_t lo         = pos2gpu::fine_bucket_lower_bound(
                        d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                        target_l, target_mask);

                    for (uint64_t r = lo; r < fine_hi; ++r) {
                        uint32_t target_r = d_sorted_mi[r] & target_mask;
                        if (target_r != target_l) break;

                        sycl::atomic_ref<unsigned long long,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>
                            cand_atomic{ *d_cand_count_ull };
                        unsigned long long ci = cand_atomic.fetch_add(1ULL);
                        if (ci >= cand_cap) return;
                        d_cand_l[ci] = static_cast<uint32_t>(l);
                        d_cand_r[ci] = static_cast<uint32_t>(r);
                    }
                });
        }).wait();

        uint64_t cand_n = 0;
        q.memcpy(&cand_n, d_cand_count, sizeof(uint64_t)).wait();
        if (dbg) {
            std::fprintf(stderr,
                "[t3-twophase] bucket %u: %llu candidates (cap %llu)%s\n",
                b, static_cast<unsigned long long>(cand_n),
                static_cast<unsigned long long>(cand_cap),
                cand_n > cand_cap ? "  *** OVERFLOW ***" : "");
        }
        if (cand_n > cand_cap) cand_n = cand_cap;
        if (cand_n == 0) continue;

        // ---- Phase B: process this bucket's candidates convergently ----
        size_t const b_blocks = static_cast<size_t>(
            (cand_n + threads - 1) / threads);
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> sT_local{
                sycl::range<1>{4 * 256}, h};
            h.parallel_for(
                sycl::nd_range<1>{ b_blocks * threads, threads },
                [=, keys_copy = keys, fk_copy = fk](sycl::nd_item<1> it) {
                    uint32_t* sT = &sT_local[0];
                    size_t local_id = it.get_local_id(0);
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    uint64_t idx = it.get_global_id(0);
                    if (idx >= cand_n) return;

                    uint32_t l = d_cand_l[idx];
                    uint32_t r = d_cand_r[idx];

                    uint64_t meta_l = d_sorted_meta[l];
                    uint32_t xb_l   = d_sorted_xbits[l];
                    uint64_t meta_r = d_sorted_meta[r];
                    uint32_t xb_r   = d_sorted_xbits[r];

                    uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                                : ((1u << num_test_bits) - 1u);

                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, meta_l, meta_r, sT, 0);
                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) return;

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
                });
        }).wait();
    }

    sycl::free(d_cand_l, q);
    sycl::free(d_cand_r, q);
    sycl::free(d_cand_count, q);
}

bool t3_twophase_enabled()
{
    static bool const enabled = [] {
        char const* v = std::getenv("POS2GPU_T3_TWOPHASE");
        return v && v[0] == '1';
    }();
    return enabled;
}

} // namespace

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
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)num_buckets;  // only the [begin, end) sub-range is iterated
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    // Phase 2 prototype: env-gated two-phase split (see run_t3_match_twophase).
    if (t3_twophase_enabled()) {
        run_t3_match_twophase(
            keys, fk, d_sorted_meta, d_sorted_xbits, d_sorted_mi,
            d_offsets, d_fine_offsets, num_match_keys,
            k, num_section_bits, num_match_target_bits, fine_bits,
            target_mask, num_test_bits,
            d_out_pairings, d_out_count, out_capacity, l_count_max,
            bucket_begin, bucket_end, d_aes_tables, q);
        return;
    }

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
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
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

                uint32_t bucket_id   = bucket_begin + static_cast<uint32_t>(it.get_group(0));
                uint32_t section_l   = bucket_id / num_match_keys;
                uint32_t match_key_r = bucket_id % num_match_keys;

                uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                uint64_t l_start = d_offsets[section_l * num_match_keys];
                uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                uint64_t l = l_start
                           + it.get_group(1) * uint64_t(threads)
                           + local_id;
                if (l >= l_end) return;

                uint64_t meta_l = d_sorted_meta[l];
                uint32_t xb_l   = d_sorted_xbits[l];

                uint32_t target_l = pos2gpu::matching_target_smem(
                                        keys_copy, 3u, match_key_r, meta_l, sT, 0)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t lo         = pos2gpu::fine_bucket_lower_bound(
                    d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                    target_l, target_mask);

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);

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
            });
    }).wait();
}

void launch_t3_match_section_pair(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint64_t section_r_row_start,
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
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)num_buckets;
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

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
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
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

                uint32_t bucket_id   = bucket_begin + static_cast<uint32_t>(it.get_group(0));
                uint32_t section_l   = bucket_id / num_match_keys;
                uint32_t match_key_r = bucket_id % num_match_keys;

                uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                uint64_t l_start = d_offsets[section_l * num_match_keys];
                uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                uint64_t l = l_start
                           + it.get_group(1) * uint64_t(threads)
                           + local_id;
                if (l >= l_end) return;

                // Sliced read: caller guarantees l ∈ [section_l_row_start, ...).
                uint64_t meta_l = d_meta_l_slice[l - section_l_row_start];
                uint32_t xb_l   = d_sorted_xbits[l];

                uint32_t target_l = pos2gpu::matching_target_smem(
                                        keys_copy, 3u, match_key_r, meta_l, sT, 0)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t lo         = pos2gpu::fine_bucket_lower_bound(
                    d_sorted_mi, d_fine_offsets[fine_idx], fine_hi,
                    target_l, target_mask);

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint32_t target_r = d_sorted_mi[r] & target_mask;
                    if (target_r != target_l) break;

                    // Sliced read: caller guarantees r ∈ [section_r_row_start, ...).
                    uint64_t meta_r = d_meta_r_slice[r - section_r_row_start];
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
            });
    }).wait();
}

void launch_t3_match_section_pair_split(
    AesHashKeys keys,
    FeistelKey fk,
    uint64_t const* d_meta_l_slice,
    uint32_t const* d_xbits_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
    uint32_t const* d_xbits_r_slice,
    uint32_t const* d_mi_r_slice,
    uint64_t section_r_row_start,
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
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)num_buckets;
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

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
                sycl::range<2>{ static_cast<size_t>(num_buckets_in_range),
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

                uint32_t bucket_id   = bucket_begin + static_cast<uint32_t>(it.get_group(0));
                uint32_t section_l   = bucket_id / num_match_keys;
                uint32_t match_key_r = bucket_id % num_match_keys;

                uint32_t section_r = pos2gpu::matching_section_r(section_l, num_section_bits);

                uint64_t l_start = d_offsets[section_l * num_match_keys];
                uint64_t l_end   = d_offsets[(section_l + 1) * num_match_keys];
                uint32_t r_bucket = section_r * num_match_keys + match_key_r;

                uint64_t l = l_start
                           + it.get_group(1) * uint64_t(threads)
                           + local_id;
                if (l >= l_end) return;

                // Sliced reads (l-side): caller guarantees l ∈ [section_l_row_start, section_l_row_end).
                uint64_t l_off  = l - section_l_row_start;
                uint64_t meta_l = d_meta_l_slice[l_off];
                uint32_t xb_l   = d_xbits_l_slice[l_off];

                uint32_t target_l = pos2gpu::matching_target_smem(
                                        keys_copy, 3u, match_key_r, meta_l, sT, 0)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t lo         = d_fine_offsets[fine_idx];
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t hi         = fine_hi;

                while (lo < hi) {
                    uint64_t mid = lo + ((hi - lo) >> 1);
                    // Sliced read (r-side): all r in [lo, fine_hi) live in section_r's row range.
                    uint32_t target_mid = d_mi_r_slice[mid - section_r_row_start] & target_mask;
                    if (target_mid < target_l) lo = mid + 1;
                    else                       hi = mid;
                }

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint64_t r_off    = r - section_r_row_start;
                    uint32_t target_r = d_mi_r_slice[r_off] & target_mask;
                    if (target_r != target_l) break;

                    uint64_t meta_r = d_meta_r_slice[r_off];
                    uint32_t xb_r   = d_xbits_r_slice[r_off];

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
            });
    }).wait();
}

} // namespace pos2gpu
