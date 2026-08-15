// HostGuard — redzone canaries around pinned-host allocations.
//
// WHY THIS EXISTS
// ---------------
// HISTORICAL NOTE FIRST, because it changes how to read the rest: this file
// was written while the tiny tier's intermittent corrupt plot (78% of the
// expected bytes, right manifest, wrong hash, ~8% of k=28 runs) was still
// unattributed, and an out-of-bounds device write was the leading theory.
// That is NO LONGER an open question. The cause was the T1 sub-section
// match staging its tile as two USM copies and waiting only the second on
// an out-of-order queue; losing the R half made a whole pass match exactly
// zero pairs, and the compounding 15/16 -> (15/16)^2 -> (15/16)^4 through
// T1/T2/T3 is precisely the 78%. POS2GPU_T1_DROP_R reproduced it
// byte-for-byte, which is proof rather than inference. Fixed, with an
// always-on zero-yield guard behind it.
//
// So nothing below is chasing a live mystery. What survives that fix is a
// narrower and still-good reason to keep the canaries:
//
//     an out-of-bounds write is DETERMINISTIC.
//     whether it lands somewhere that changes the plot is LAYOUT-DEPENDENT.
//
// Every streaming kernel writes to host memory through a cursor
// (`part_vals[pos] = v`, fragment drains, sort scatter), and a cursor that
// runs one entry long lands in whatever allocation happens to sit next in
// the address space. Under spill, `h_meta` and `h_t3` are never allocated
// at all, so every later buffer moves — same overrun, different victim.
// That means a soak samples the SECOND question and can pass ~50 times
// while the first has been true all along. Canaries answer the first one
// directly, on every run, in the buffer that was actually overrun rather
// than in the innocent one downstream of it.
//
// A clean run is therefore a real result, not a null one — which is why the
// guard is deliberately cheap enough to leave on for a long soak, and why
// its own arithmetic is unit-tested (see USAGE): a canary that has quietly
// stopped firing is indistinguishable from a healthy tree.
//
// USAGE
//   XCHPLOT2_HOST_GUARD=1    enable, 1 MiB redzone either side
//   XCHPLOT2_HOST_GUARD=<N>  enable, N MiB either side
// Disabled (the default) every entry point below compiles to a branch on
// one bool and the allocation is byte-for-byte what it was before.
//
// Header-only and device-free so a unit test can exercise the arithmetic
// without a GPU (see tools/parity/host_guard_test.cpp) — the failure mode
// that matters is a guard that silently never fires, and no end-to-end
// plot run can distinguish that from a clean one.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace pos2gpu {

class HostGuard {
public:
    static HostGuard& instance()
    {
        static HostGuard g;
        return g;
    }

    bool   enabled()   const { return pad_ != 0; }
    // Bytes to add to EACH side of a request. 0 when disabled, so callers
    // can add `2 * pad_bytes()` unconditionally.
    size_t pad_bytes() const { return pad_; }

    // Take the raw base of an allocation of (bytes + 2*pad_bytes()), paint
    // both redzones, and return the pointer the caller should use. `what`
    // must outlive the allocation (it is always a string literal here).
    void* arm(void* base, size_t bytes, char const* what)
    {
        if (!enabled() || !base) return base;
        std::lock_guard<std::mutex> lk(mtx_);
        uint64_t const id   = ++next_id_;
        auto*    const b    = static_cast<uint8_t*>(base);
        void*    const user = b + pad_;
        paint(b, id, 0);            // head redzone
        paint(b + pad_ + bytes, id, 1);   // tail redzone
        live_.emplace(user, Entry{base, bytes, what, id});
        return user;
    }

    // Verify both redzones of `user` and return the raw base to free.
    // Reports (does not throw — this runs on free paths, including a
    // destructor unwinding another error) and keeps going.
    void* disarm(void* user, char const* where)
    {
        if (!enabled() || !user) return user;
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = live_.find(user);
        if (it == live_.end()) return user;   // not ours; pass through
        check_entry(it->second, where);
        void* const base = it->second.base;
        live_.erase(it);
        return base;
    }

    // Verify every live allocation. Call at phase boundaries: it names the
    // buffer that was overrun AND the phase it happened in, which is the
    // difference between a bug report and a mystery.
    void check_all(char const* where)
    {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto const& [user, e] : live_) {
            (void)user;
            check_entry(e, where);
        }
    }

    uint64_t damage_count() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return damaged_;
    }

    // Number of live guarded allocations (tests).
    size_t live_count() const
    {
        std::lock_guard<std::mutex> lk(mtx_);
        return live_.size();
    }

private:
    struct Entry {
        void*       base  = nullptr;
        size_t      bytes = 0;
        char const* what  = "";
        uint64_t    id    = 0;
    };

    HostGuard()
    {
        char const* v = std::getenv("XCHPLOT2_HOST_GUARD");
        if (!v || !v[0] || v[0] == '0') return;
        unsigned long mib = std::strtoul(v, nullptr, 10);
        if (mib == 0) mib = 1;              // "XCHPLOT2_HOST_GUARD=1" / "=on"
        if (mib > 64) mib = 64;
        pad_ = size_t(mib) << 20;           // page-multiple: keeps USM alignment
    }

    // Distinct 64-bit word per (allocation, side, offset). A neighbouring
    // buffer's redzone, a shifted copy of this one, and a run of zeros are
    // all detected — a single constant would miss the first two.
    static uint64_t word_for(uint64_t id, int side, size_t index)
    {
        uint64_t x = id * 0x9E3779B97F4A7C15ull
                   + uint64_t(side) * 0xD6E8FEB86659FD93ull
                   + index * 0xBF58476D1CE4E5B9ull;
        x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27; x *= 0x94D049BB133111EBull;
        x ^= x >> 31;
        return x | 1ull;                    // never 0: zeroed memory is damage
    }

    void paint(void* p, uint64_t id, int side) const
    {
        auto* w = static_cast<uint64_t*>(p);
        size_t const n = pad_ / sizeof(uint64_t);
        for (size_t i = 0; i < n; ++i) w[i] = word_for(id, side, i);
    }

    // Scan one redzone. Returns the number of damaged words and, via
    // `first`, the index of the first — the overrun DISTANCE, which is
    // usually the most diagnostic number in the whole report.
    size_t scan(void const* p, uint64_t id, int side, size_t& first) const
    {
        auto const* w = static_cast<uint64_t const*>(p);
        size_t const n = pad_ / sizeof(uint64_t);
        size_t bad = 0;
        first = 0;
        for (size_t i = 0; i < n; ++i) {
            if (w[i] == word_for(id, side, i)) continue;
            if (bad == 0) first = i;
            ++bad;
        }
        return bad;
    }

    void check_entry(Entry const& e, char const* where)
    {
        auto* const b = static_cast<uint8_t*>(e.base);
        for (int side = 0; side < 2; ++side) {
            uint8_t const* zone = side == 0 ? b : b + pad_ + e.bytes;
            size_t first = 0;
            size_t const bad = scan(zone, e.id, side, first);
            if (!bad) continue;
            ++damaged_;
            // Signed offset from the buffer: negative = underrun.
            long long const delta = side == 0
                ? -(long long)((pad_ - first * sizeof(uint64_t)))
                :  (long long)(e.bytes + first * sizeof(uint64_t));
            std::fprintf(stderr,
                "[host-guard] CORRUPTION at %s: buffer '%s' (%zu bytes) "
                "%s redzone damaged — %zu of %zu words, first at byte "
                "offset %lld relative to the buffer%s. Something wrote "
                "outside this allocation; the plot from this run is not "
                "trustworthy.\n",
                where, e.what, e.bytes,
                side == 0 ? "HEAD (underrun)" : "TAIL (overrun)",
                bad, pad_ / sizeof(uint64_t), delta,
                side == 0 ? " (negative = before the start)" : "");
            // Repaint so one overrun does not re-report at every later
            // checkpoint and drown the one that comes after it.
            paint(const_cast<uint8_t*>(zone), e.id, side);
        }
    }

    mutable std::mutex mtx_;
    std::unordered_map<void*, Entry> live_;
    size_t   pad_      = 0;
    uint64_t next_id_  = 0;
    uint64_t damaged_  = 0;
};

// Convenience wrappers so call sites stay one line.
inline bool   host_guard_on()   { return HostGuard::instance().enabled(); }
inline size_t host_guard_pad()  { return HostGuard::instance().pad_bytes(); }
inline void*  host_guard_arm(void* base, size_t bytes, char const* what)
{
    return HostGuard::instance().arm(base, bytes, what);
}
inline void*  host_guard_disarm(void* user, char const* where)
{
    return HostGuard::instance().disarm(user, where);
}
inline void   host_guard_check(char const* where)
{
    HostGuard::instance().check_all(where);
}

}  // namespace pos2gpu
