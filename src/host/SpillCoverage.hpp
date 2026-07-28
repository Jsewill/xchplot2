// SpillCoverage.hpp — which byte ranges of a spill file have been written.
//
// The host-RAM disk-offload spills tables to a TempFile (mkstemp + unlink),
// which is SPARSE: a pread of a range that was never pwritten succeeds and
// returns ZEROS. Downstream that does not look like an error — it looks like
// the plot lost data. The observed signature is a short, unusually
// well-compressed .plot2 with unchanged T1/T2/T3 match counts, produced
// intermittently, which no amount of end-to-end hashing can attribute.
//
// So every read asserts its range was written first. The cost is a handful of
// std::map operations per plot: writes arrive per-bucket or per-pass and
// collapse to a single interval as the table fills.
//
// Header-only and free of SYCL/GPU deps on purpose, so spill_coverage_test can
// exercise the interval arithmetic anywhere, with no device present. That
// arithmetic is the part worth testing: too loose and the guard never fires
// (protecting nothing while passing every parity run), too tight and it
// rejects the legitimate in-place rewrite the bucket sorts perform.

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>

namespace pos2gpu {

class SpillCoverage {
public:
    // Record [off, off+len) as written, merging into any interval it overlaps
    // or abuts. Overlapping rewrites (the per-bucket sorts overwrite ranges
    // the match phase already wrote) are expected and must not fragment.
    void note_written(std::uint64_t off, std::uint64_t len)
    {
        if (len == 0) return;
        std::uint64_t end = off + len;
        auto it = written_.upper_bound(off);
        if (it != written_.begin()) {
            auto prev = std::prev(it);
            if (prev->second >= off) {
                off = prev->first;
                end = std::max(end, prev->second);
                written_.erase(prev);
            }
        }
        it = written_.lower_bound(off);
        while (it != written_.end() && it->first <= end) {
            end = std::max(end, it->second);
            it  = written_.erase(it);
        }
        written_.emplace(off, end);
    }

    // True when [off, off+len) lies entirely inside one written interval.
    [[nodiscard]] bool covered(std::uint64_t off, std::uint64_t len) const
    {
        if (len == 0) return true;
        std::uint64_t const end = off + len;
        auto it = written_.upper_bound(off);
        if (it != written_.begin()) --it;
        return it != written_.end() && it->first <= off && it->second >= end;
    }

    // How far contiguous coverage reaches from `off` — for error messages.
    [[nodiscard]] std::uint64_t covered_to(std::uint64_t off) const
    {
        auto it = written_.upper_bound(off);
        if (it != written_.begin()) --it;
        if (it != written_.end() && it->first <= off) return it->second;
        return off;
    }

    [[nodiscard]] std::size_t interval_count() const { return written_.size(); }

private:
    std::map<std::uint64_t, std::uint64_t> written_;   // offset -> end, half-open
};

}  // namespace pos2gpu
