// spill_coverage_test — SpillCoverage's interval arithmetic.
//
// This guard is what stands between a spill-path bug and a silently short
// .plot2: spill TempFiles are sparse, so a read of a never-written range
// returns zeros instead of failing. Two ways to get that wrong, and a green
// end-to-end parity run catches NEITHER:
//   - too loose  -> the guard never fires and protects nothing;
//   - too tight  -> it rejects the in-place rewrite the per-bucket sorts do,
//                   turning good plots into hard errors.
// So the negative cases below matter at least as much as the positive ones.
//
// Pure arithmetic, no SYCL/GPU — builds and runs on any host.

#include "host/SpillCoverage.hpp"

#include <cstdio>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

}  // namespace

int main()
{
    using pos2gpu::SpillCoverage;

    {   // The T1-match / T3-match pattern: sequential appends.
        SpillCoverage c;
        for (std::uint64_t i = 0; i < 10; ++i) c.note_written(i * 100, 100);
        check(c.interval_count() == 1, "contiguous appends merge to one interval");
        check(c.covered(0, 1000),      "the whole appended range is covered");
        check(!c.covered(0, 1001),     "one byte past the end is not covered");
        check(!c.covered(1000, 8),     "a read just past the end is not covered");
    }

    {   // A hole — the case that silently returns zeros on a real file.
        SpillCoverage c;
        c.note_written(0, 100);
        c.note_written(200, 100);          // gap at [100, 200)
        check(c.interval_count() == 2,     "disjoint writes stay separate");
        check(c.covered(0, 100),           "block before the hole is covered");
        check(c.covered(200, 100),         "block after the hole is covered");
        check(!c.covered(100, 100),        "the hole itself is not covered");
        check(!c.covered(0, 300),          "a span across the hole is not covered");
        check(!c.covered(50, 100),         "a read straddling into the hole is not covered");
        check(c.covered_to(0) == 100,      "covered_to reports the gap edge");
    }

    {   // Out-of-order writes that later bridge.
        SpillCoverage c;
        c.note_written(500, 100);
        c.note_written(0, 100);
        check(!c.covered(0, 600),      "not covered while the middle is missing");
        c.note_written(100, 400);      // bridges both sides
        check(c.interval_count() == 1, "the bridging write merges everything");
        check(c.covered(0, 600),       "bridged range is covered");
    }

    {   // The per-bucket sorts rewrite ranges the match phase already wrote.
        // That must stay one interval, not fragment into false gaps.
        SpillCoverage c;
        c.note_written(0, 1000);
        for (std::uint64_t b = 0; b < 10; ++b) c.note_written(b * 100, 100);
        check(c.interval_count() == 1, "in-place rewrite keeps one interval");
        check(c.covered(0, 1000),      "still covered after the rewrite");
    }

    {   // Abutting (end == next offset) must merge, or every append fragments.
        SpillCoverage c;
        c.note_written(0, 100);
        c.note_written(100, 100);
        check(c.interval_count() == 1, "abutting writes merge");
        check(c.covered(0, 200),       "abutted range is covered");
    }

    {   // Degenerate inputs.
        SpillCoverage c;
        check(!c.covered(0, 8),        "reading before any write is not covered");
        check(c.covered(0, 0),         "a zero-length read is trivially covered");
        c.note_written(0, 0);
        check(c.interval_count() == 0, "a zero-length write records nothing");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall good\n", failures);
    return failures ? 1 : 0;
}
