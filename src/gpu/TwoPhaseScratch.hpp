#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace pos2gpu::sycl_backend {

// ---- Two-phase match candidate-scratch budget --------------------------
//
// The T2/T3 two-phase match trades VRAM for speed: it stages {l, r}
// candidate index pairs in a device buffer before running the pairing AES
// over them. That buffer is sized from the plot's capacity, not from the
// tier's memory budget — in
//
//     cand_cap = (out_capacity / num_buckets_in_range) * M
//
// both terms scale with the tier's slicing factor N, so N cancels and the
// scratch lands at the same size in every tier. It was allocated with a
// raw sycl::malloc_device, so it never appeared in the streaming-stats
// trace that the tier peak models were calibrated against (see the anchor
// comments in GpuBufferPool.cpp, which say so outright). The models
// therefore under-predicted the true peak by ~3.1 GB at k=28, and the tier
// picker handed memory-constrained cards a tier that could not fit: a
// 7.6 GB Tesla P4 OOM'd on every tier except tiny, having been told
// minimal needed 3.67 GiB when it really needed 7.74.
//
// The budget makes the trade explicit. The pipeline declares how many
// bytes of candidate scratch this queue may hold — free VRAM minus the
// tier's modelled peak minus the safety margin — and the match falls back
// to the single-kernel path when the scratch will not fit. That path is
// correct for any input and allocates nothing, so the worst case is
// slower, never wrong and never OOM. A budget of 0 disables two-phase.
//
// Because the grant is (free - peak - margin), the arithmetic is safe by
// construction: peak + scratch + margin <= free, for any tier and any k.
// ONE scratch buffer, shared by the T2 and T3 two-phase match.
//
// T2 and T3 are never live at the same time — T2 match, T2 sort, then T3
// match, strictly in that order on one queue — but each used to keep its own
// per-queue cache, so both sat resident for the whole plot and the peak
// carried their SUM (2080 + 1040 = 3120 MiB at k=28 before right-sizing).
// Sharing one buffer, grown to whichever phase asks for more, makes the peak
// the MAX instead: it halves the scratch at zero cost, because the loser of
// the max simply reuses a buffer that is already big enough.
//
// Growing frees the old buffer before allocating the new one, so there is no
// moment where both are held. That is safe because run_t{2,3}_match_twophase
// ends with q.wait(): nothing in flight still references the old pointer.
struct TwoPhaseScratch {
    uint32_t* d_cand_l     = nullptr;
    uint32_t* d_cand_r     = nullptr;
    uint64_t* d_cand_count = nullptr;
    uint32_t* d_overflow   = nullptr;  // sticky flag: a bucket blew cand_cap
    uint64_t  cap          = 0;
    uint64_t  bytes        = 0;        // device bytes currently held
};

struct TwoPhaseState {
    uint64_t        limit = 0;   // bytes the pipeline permits on this queue
    TwoPhaseScratch scratch;
};

inline std::mutex& twophase_mutex()
{
    static std::mutex mu;
    return mu;
}

inline std::unordered_map<sycl::queue*, TwoPhaseState>& twophase_map()
{
    static std::unordered_map<sycl::queue*, TwoPhaseState> m;
    return m;
}

inline void free_twophase_scratch(TwoPhaseScratch& s, sycl::queue& q)
{
    if (s.d_cand_l)     { sycl::free(s.d_cand_l, q);     s.d_cand_l     = nullptr; }
    if (s.d_cand_r)     { sycl::free(s.d_cand_r, q);     s.d_cand_r     = nullptr; }
    if (s.d_cand_count) { sycl::free(s.d_cand_count, q); s.d_cand_count = nullptr; }
    if (s.d_overflow)   { sycl::free(s.d_overflow, q);   s.d_overflow   = nullptr; }
    s.cap   = 0;
    s.bytes = 0;
}

// Lowering a grant releases oversized cached buffers before the next plot.
inline void set_twophase_budget(sycl::queue& q, uint64_t bytes)
{
    std::lock_guard<std::mutex> lk(twophase_mutex());
    auto& st = twophase_map()[&q];
    if (st.scratch.bytes > bytes) {
        q.wait();
        free_twophase_scratch(st.scratch, q);
    }
    st.limit = bytes;
}

inline void release_twophase_scratch(sycl::queue& q)
{
    std::lock_guard<std::mutex> lk(twophase_mutex());
    auto const it = twophase_map().find(&q);
    if (it == twophase_map().end()) return;
    q.wait();
    free_twophase_scratch(it->second.scratch, q);
    twophase_map().erase(it);
}

// Returns nullptr when the scratch will not fit this queue's grant, or when
// the allocation fails outright. Neither is fatal: the caller rewinds the
// output counter and runs the single-kernel path, which allocates nothing and
// is correct for any input. The old code threw on a failed malloc, which is
// how a Tesla P4 with a perfectly valid plot configuration died with
// "T3 two-phase: candidate buffer alloc failed" instead of quietly taking the
// slower path.
inline TwoPhaseScratch* acquire_twophase_scratch(sycl::queue& q,
                                                 uint64_t cand_cap)
{
    std::lock_guard<std::mutex> lk(twophase_mutex());
    auto& st = twophase_map()[&q];
    auto& s  = st.scratch;
    constexpr uint64_t fixed = sizeof(uint64_t) + sizeof(uint32_t);
    constexpr uint64_t per_candidate = sizeof(uint32_t) * 2;
    if (cand_cap == 0 || st.limit < fixed ||
        cand_cap > (st.limit - fixed) / per_candidate) return nullptr;
    uint64_t const want = cand_cap * per_candidate + fixed;
    if (s.cap >= cand_cap && s.bytes <= st.limit) return &s;

    free_twophase_scratch(s, q);
    try {
        s.d_cand_l     = sycl::malloc_device<uint32_t>(cand_cap, q);
        s.d_cand_r     = sycl::malloc_device<uint32_t>(cand_cap, q);
        s.d_cand_count = sycl::malloc_device<uint64_t>(1, q);
        s.d_overflow   = sycl::malloc_device<uint32_t>(1, q);
    } catch (sycl::exception const&) {
        // Handled by the null check below — AdaptiveCpp returns nullptr on an
        // OOM, but throws for other allocation errors.
    }
    if (!s.d_cand_l || !s.d_cand_r || !s.d_cand_count || !s.d_overflow) {
        free_twophase_scratch(s, q);
        return nullptr;
    }
    s.cap   = cand_cap;
    s.bytes = want;
    return &s;
}

// Bytes the two-phase scratch currently holds on this queue. The bench VRAM
// watchdog uses this to separate the tier's modelled footprint from the
// optional scratch sitting on top of it — so it can tell "this tier is over
// budget" (a bug) apart from "we deliberately spent spare VRAM on the fast
// path" (working as intended).
inline uint64_t twophase_bytes_held(sycl::queue& q)
{
    std::lock_guard<std::mutex> lk(twophase_mutex());
    auto const it = twophase_map().find(&q);
    return it == twophase_map().end() ? 0 : it->second.scratch.bytes;
}


} // namespace pos2gpu::sycl_backend
