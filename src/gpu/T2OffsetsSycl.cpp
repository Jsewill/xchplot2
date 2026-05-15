// T2OffsetsSycl.cpp — SYCL implementation of T2's three backend-dispatched
// kernels. Pattern mirrors T1OffsetsSycl.cpp; reuses the shared SYCL
// queue + AES-table USM buffer from SyclBackend.hpp.

#include "gpu/MatchKernelCommon.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/T2Offsets.cuh"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace pos2gpu {

void launch_t2_compute_bucket_offsets(
    uint32_t const* d_sorted_mi,
    uint64_t total,
    int num_match_target_bits,
    uint32_t num_buckets,
    uint64_t* d_offsets,
    sycl::queue& q)
{
    constexpr size_t threads = 256;
    size_t   const out_count = static_cast<size_t>(num_buckets) + 1;
    size_t   const groups    = (out_count + threads - 1) / threads;

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
                uint32_t v   = d_sorted_mi[mid] >> bucket_shift;
                if (v < b) lo = mid + 1;
                else       hi = mid;
            }
            d_offsets[b] = lo;
        }).wait();
}

void launch_t2_compute_fine_bucket_offsets(
    uint32_t const* d_sorted_mi,
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
                uint32_t t   = (d_sorted_mi[mid] & target_mask) >> shift;
                if (t < fine_key) lo = mid + 1;
                else              hi = mid;
            }
            d_fine_offsets[tid] = lo;

            if (tid == total - 1) {
                d_fine_offsets[total] = d_bucket_offsets[num_buckets];
            }
        }).wait();
}

namespace {

// Per-queue cached candidate scratch for run_t2_match_twophase. See
// the equivalent in T3OffsetsSycl.cpp for the full rationale; T2 uses
// its own cache (16× per-bucket-output sizing vs T3's 8×, so they
// grow independently).
struct T2CandScratch {
    uint32_t* d_cand_l     = nullptr;
    uint32_t* d_cand_r     = nullptr;
    uint64_t* d_cand_count = nullptr;
    uint64_t  cap          = 0;
};

T2CandScratch& acquire_t2_cand_scratch(sycl::queue& q, uint64_t cand_cap)
{
    static std::mutex mu;
    static std::unordered_map<sycl::queue*, T2CandScratch> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto& s = cache[&q];
    if (s.cap < cand_cap) {
        if (s.d_cand_l)     sycl::free(s.d_cand_l, q);
        if (s.d_cand_r)     sycl::free(s.d_cand_r, q);
        if (s.d_cand_count) sycl::free(s.d_cand_count, q);
        s.d_cand_l     = sycl::malloc_device<uint32_t>(cand_cap, q);
        s.d_cand_r     = sycl::malloc_device<uint32_t>(cand_cap, q);
        s.d_cand_count = sycl::malloc_device<uint64_t>(1, q);
        if (!s.d_cand_l || !s.d_cand_r || !s.d_cand_count) {
            if (s.d_cand_l)     { sycl::free(s.d_cand_l, q);     s.d_cand_l     = nullptr; }
            if (s.d_cand_r)     { sycl::free(s.d_cand_r, q);     s.d_cand_r     = nullptr; }
            if (s.d_cand_count) { sycl::free(s.d_cand_count, q); s.d_cand_count = nullptr; }
            s.cap = 0;
            throw std::runtime_error("T2 two-phase: candidate buffer alloc failed");
        }
        s.cap = cand_cap;
    }
    return s;
}

// ---- Phase 2 prototype: two-phase T2 match ----------------------------
//
// Mirrors run_t3_match_twophase (T3OffsetsSycl.cpp). The single-kernel
// launch_t2_match_all_buckets has the same divergent shape as T3 — one
// thread per l-entry with a variable-length inner r-loop and the 32-round
// pairing AES inside it. This splits it:
//   Phase A — enumerate {u32 l, u32 r} candidate index pairs
//     (matching_target AES + binary search + r-loop). No pairing.
//   Phase B — process the flat pair list convergently: one thread per
//     candidate does exactly one pairing AES + test + the T2 output
//     (meta / match_info / x_bits).
// Tiled one bucket at a time so the candidate scratch stays bounded.
// Gated behind POS2GPU_T2_TWOPHASE=1; default path unchanged when unset.
void run_t2_match_twophase(
    AesHashKeys const& keys,
    uint64_t const* d_sorted_meta,
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
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
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
        char const* v = std::getenv("POS2GPU_T2_TWOPHASE_DEBUG");
        return v && v[0] == '1';
    }();

    // Per-bucket candidate scratch: {u32 l, u32 r}. Sized at 16× the
    // average per-bucket output count — generous headroom over the
    // expected candidate:output ratio (~4× for the T3 test filter; T2's
    // is measured via the debug print). Allocated once, reused per bucket
    // — and now also reused across calls on the same queue via the
    // per-queue cache (acquire_t2_cand_scratch above).
    uint64_t const cand_cap =
        (out_capacity / (num_buckets_in_range ? num_buckets_in_range : 1)) * 16
        + 4096;
    auto& scratch = acquire_t2_cand_scratch(q, cand_cap);
    uint32_t* d_cand_l     = scratch.d_cand_l;
    uint32_t* d_cand_r     = scratch.d_cand_r;
    uint64_t* d_cand_count = scratch.d_cand_count;
    auto* d_cand_count_ull =
        reinterpret_cast<unsigned long long*>(d_cand_count);
    auto* d_out_count_ull =
        reinterpret_cast<unsigned long long*>(d_out_count);

    // Production v2: per-bucket ops chain through SYCL event deps —
    // memset → Phase A → Phase B → next bucket — with zero host sync per
    // bucket. See run_t3_match_twophase for the full rationale; this
    // mirrors that design.
    size_t const b_blocks_max = static_cast<size_t>(
        (cand_cap + threads - 1) / threads);

    sycl::event prev_event;
    bool have_prev = false;

    for (uint32_t b = bucket_begin; b < bucket_end; ++b) {
        auto e_memset = q.submit([&](sycl::handler& h) {
            if (have_prev) h.depends_on(prev_event);
            h.memset(d_cand_count, 0, sizeof(uint64_t));
        });

        // ---- Phase A: enumerate candidate (l, r) index pairs ----
        auto e_phase_a = q.submit([&](sycl::handler& h) {
            h.depends_on(e_memset);
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
                                            keys_copy, 2u, match_key_r, meta_l, sT, 0)
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
        });

        // ---- Phase B: process this bucket's candidates convergently ----
        // cand_cap-sized grid; threads read the actual cand_n from the
        // device counter (Phase A's writes are visible via e_phase_a).
        auto e_phase_b = q.submit([&](sycl::handler& h) {
            h.depends_on(e_phase_a);
            sycl::local_accessor<uint32_t, 1> sT_local{
                sycl::range<1>{4 * 256}, h};
            h.parallel_for(
                sycl::nd_range<1>{ b_blocks_max * threads, threads },
                [=, keys_copy = keys](sycl::nd_item<1> it) {
                    uint32_t* sT = &sT_local[0];
                    size_t local_id = it.get_local_id(0);
                    #pragma unroll 1
                    for (size_t i = local_id; i < 4 * 256; i += threads) {
                        sT[i] = d_aes_tables[i];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    uint64_t idx = it.get_global_id(0);
                    uint64_t cand_n = *d_cand_count;
                    if (cand_n > cand_cap) cand_n = cand_cap;
                    if (idx >= cand_n) return;

                    uint32_t l = d_cand_l[idx];
                    uint32_t r = d_cand_r[idx];

                    uint64_t meta_l = d_sorted_meta[l];
                    uint64_t meta_r = d_sorted_meta[r];

                    uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                                : ((1u << num_test_bits) - 1u);
                    uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                     : ((1u << num_match_info_bits) - 1u);
                    int meta_bits = 2 * k;

                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, meta_l, meta_r, sT, 0);

                    uint32_t test_result = res.r[3] & test_mask;
                    if (test_result != 0) return;

                    uint32_t match_info_result = res.r[0] & info_mask;
                    uint64_t meta_result_full = uint64_t(res.r[1]) | (uint64_t(res.r[2]) << 32);
                    uint64_t meta_result = (meta_bits == 64)
                                            ? meta_result_full
                                            : (meta_result_full & ((1ULL << meta_bits) - 1ULL));

                    uint32_t x_bits_l = static_cast<uint32_t>((meta_l >> k) >> half_k);
                    uint32_t x_bits_r = static_cast<uint32_t>((meta_r >> k) >> half_k);
                    uint32_t x_bits   = (x_bits_l << half_k) | x_bits_r;

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                });
        });

        // Optional debug print — adds one host sync per bucket.
        if (dbg) {
            uint64_t cand_n_dbg = 0;
            q.submit([&](sycl::handler& h) {
                h.depends_on(e_phase_a);
                h.memcpy(&cand_n_dbg, d_cand_count, sizeof(uint64_t));
            }).wait();
            std::fprintf(stderr,
                "[t2-twophase] bucket %u: %llu candidates (cap %llu)%s\n",
                b, static_cast<unsigned long long>(cand_n_dbg),
                static_cast<unsigned long long>(cand_cap),
                cand_n_dbg > cand_cap ? "  *** OVERFLOW ***" : "");
        }

        prev_event = e_phase_b;
        have_prev  = true;
    }

    // Single drain — the caller expects synchronous completion. After
    // the wait the cached scratch is fully drained for the next call;
    // no free — the per-queue cache owns the buffers.
    q.wait();
}

bool t2_twophase_enabled()
{
    static bool const enabled = [] {
        char const* v = std::getenv("POS2GPU_T2_TWOPHASE");
        return v && v[0] == '1';
    }();
    return enabled;
}

} // namespace

void launch_t2_match_all_buckets(
    AesHashKeys keys,
    uint64_t const* d_sorted_meta,
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
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
    uint64_t* d_out_count,
    uint64_t out_capacity,
    uint64_t l_count_max,
    uint32_t bucket_begin,
    uint32_t bucket_end,
    sycl::queue& q)
{
    (void)num_buckets; // only the [begin, end) sub-range is iterated
    if (bucket_end <= bucket_begin) return;
    uint32_t const num_buckets_in_range = bucket_end - bucket_begin;

    uint32_t* d_aes_tables = sycl_backend::aes_tables_device(q);

    // Phase 2 prototype: env-gated two-phase split (see run_t2_match_twophase).
    if (t2_twophase_enabled()) {
        run_t2_match_twophase(
            keys, d_sorted_meta, d_sorted_mi, d_offsets, d_fine_offsets,
            num_match_keys, k, num_section_bits, num_match_target_bits,
            fine_bits, target_mask, num_test_bits, num_match_info_bits, half_k,
            d_out_meta, d_out_mi, d_out_xbits, d_out_count, out_capacity,
            l_count_max, bucket_begin, bucket_end, d_aes_tables, q);
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
            [=, keys_copy = keys](sycl::nd_item<2> it) {
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

                uint32_t target_l = pos2gpu::matching_target_smem(
                                        keys_copy, 2u, match_key_r, meta_l, sT, 0)
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
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);
                int meta_bits = 2 * k;

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint32_t target_r = d_sorted_mi[r] & target_mask;
                    if (target_r != target_l) break;

                    uint64_t meta_r = d_sorted_meta[r];

                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, meta_l, meta_r, sT, 0);

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

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                }
            });
    }).wait();
}

void launch_t2_match_section_pair_split(
    AesHashKeys keys,
    uint64_t const* d_meta_l_slice,
    uint64_t section_l_row_start,
    uint64_t const* d_meta_r_slice,
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
    int num_match_info_bits,
    int half_k,
    uint64_t* d_out_meta,
    uint32_t* d_out_mi,
    uint32_t* d_out_xbits,
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
            [=, keys_copy = keys](sycl::nd_item<2> it) {
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

                // Sliced read (l-side): l ∈ [section_l_row_start, section_l_row_end).
                uint64_t meta_l = d_meta_l_slice[l - section_l_row_start];

                uint32_t target_l = pos2gpu::matching_target_smem(
                                        keys_copy, 2u, match_key_r, meta_l, sT, 0)
                                  & target_mask;

                uint32_t fine_shift = static_cast<uint32_t>(num_match_target_bits - fine_bits);
                uint32_t fine_key   = target_l >> fine_shift;
                uint64_t fine_idx   = (uint64_t(r_bucket) << fine_bits) | fine_key;
                uint64_t lo         = d_fine_offsets[fine_idx];
                uint64_t fine_hi    = d_fine_offsets[fine_idx + 1];
                uint64_t hi         = fine_hi;

                while (lo < hi) {
                    uint64_t mid = lo + ((hi - lo) >> 1);
                    // Sliced read (r-side): r ∈ [section_r_row_start, section_r_row_end).
                    uint32_t target_mid = d_mi_r_slice[mid - section_r_row_start] & target_mask;
                    if (target_mid < target_l) lo = mid + 1;
                    else                       hi = mid;
                }

                uint32_t test_mask = (num_test_bits >= 32) ? 0xFFFFFFFFu
                                                            : ((1u << num_test_bits) - 1u);
                uint32_t info_mask = (num_match_info_bits >= 32) ? 0xFFFFFFFFu
                                                                 : ((1u << num_match_info_bits) - 1u);
                int meta_bits = 2 * k;

                for (uint64_t r = lo; r < fine_hi; ++r) {
                    uint64_t r_off    = r - section_r_row_start;
                    uint32_t target_r = d_mi_r_slice[r_off] & target_mask;
                    if (target_r != target_l) break;

                    uint64_t meta_r = d_meta_r_slice[r_off];

                    pos2gpu::Result128 res = pos2gpu::pairing_smem(
                        keys_copy, meta_l, meta_r, sT, 0);

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

                    sycl::atomic_ref<unsigned long long,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        out_count_atomic{ *d_out_count_ull };
                    unsigned long long out_idx = out_count_atomic.fetch_add(1ULL);
                    if (out_idx >= out_capacity) return;

                    d_out_meta [out_idx] = meta_result;
                    d_out_mi   [out_idx] = match_info_result;
                    d_out_xbits[out_idx] = x_bits;
                }
            });
    }).wait();
}

} // namespace pos2gpu
