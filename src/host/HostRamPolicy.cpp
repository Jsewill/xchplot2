#include "host/HostRamPolicy.hpp"

#include <algorithm>

namespace pos2gpu {
namespace {

// Full-table passes each spilled table makes over the temp dir in ONE plot,
// counted from the call sites in GpuPipeline.cpp. "Write once, read once" is
// wrong for everything except h_t3: the T1 and T2 sorts read the table as
// their partition source and then write the SORTED result back over it, which
// is two extra passes nobody was counting.
//
// h_t1_meta (Tiny only)                                        2 W + 3 R = 5
//   W  T1 match, append per sub-section pass
//   R  T1 sort, streaming-partition source tiles
//   W  T1 sort, per-bucket sorted result, in place
//   R  T2 match, L section (each section loaded once, cached)
//   R  T2 match, R bucket (each bucket visited exactly once)
//
// h_t2_meta (Tiny only)                                        2 W + 3 R = 5
//   Same five, one table later: T2 match / T2 sort / T3 match.
//
// h_t2_xbits, tiled gather (Minimal, Tiny)                     3 W + 3 R = 6
//   W  T2 match append; R partition source; W per-bucket sorted
//   R  T2-sort gather rehydrate; W gather's sorted output tiles
//   R  release to d_t2_xbits_sorted (Minimal)
//   Tiny replaces that last read with TWO — T3 match's L and R — so it is 7;
//   see kXbitsPassesTiny.
//
// h_t2_xbits, single-shot gather (Compact)                     2 W + 2 R = 4
//   W append; R partition source; W per-bucket sorted; R release.
//
// h_t3 (Compact, Minimal, Tiny)                                1 W + 1 R = 2
//   W  T3 match, append per pass
//   R  T3 sort, whole table back (tiled in Tiny, same total bytes)
//
// These are the ONLY numbers here that can go stale silently, because they
// mirror control flow in another translation unit rather than deriving from
// it. host_spill_policy_test pins them; the SpillEngine's measured counters
// are the cross-check that says whether they are still true.
struct Passes { int writes; int reads; };

constexpr Passes kT1MetaPasses     {2, 3};
constexpr Passes kT2MetaPasses     {2, 3};
constexpr Passes kXbitsPassesTiny  {3, 4};
constexpr Passes kXbitsPassesTiled {3, 3};
constexpr Passes kXbitsPassesFlat  {2, 2};
constexpr Passes kT3Passes         {1, 1};

}  // namespace

HostRamSpillPlan plan_host_ram_spill(HostRamSpillInputs const& in)
{
    HostRamSpillPlan out;

    uint64_t const table8 = uint64_t(8) * in.cap_entries;  // an 8-B/entry table
    uint64_t const table4 = uint64_t(4) * in.cap_entries;  // a 4-B/entry table
    uint64_t const B      = in.budget;                     // 0 == "min"

    uint64_t est            = in.host_required;
    uint64_t routable_total = 0;

    // Route LARGEST-FIRST (every 8-B table before the 4-B one) and stop as
    // soon as the budget is met, so a run only pays for the I/O it needs.
    // "min" (B == 0) never stops.
    // `passes` is per-table temp-dir traffic, accumulated only for the tables
    // this plan actually routes — a table left pinned costs no disk I/O. The
    // mmap class is excluded (h_frags passes {0,0}): the kernel writes those
    // pages back on its own schedule, so a fixed per-plot figure would be a
    // guess dressed as a measurement.
    auto consider = [&](uint64_t table, bool available, bool& bit,
                        Passes passes, bool mmap_class = false) {
        if (!available) return;
        routable_total += table;
        if (B == 0 || est > B) {
            bit = true;
            est -= std::min(table, est);
            out.spilled_bytes += table;
            if (mmap_class) out.reclaimable += table;
            out.traffic_written += uint64_t(passes.writes) * table;
            out.traffic_read    += uint64_t(passes.reads)  * table;
        }
    };

    Passes const xbits_passes =
        in.tier_tiny          ? kXbitsPassesTiny
      : in.tier_tiled_gather  ? kXbitsPassesTiled
                              : kXbitsPassesFlat;

    // Which tables are routable per tier, and why the rest are not, is
    // documented at the call site in BatchPlotter::run().
    consider(table8, in.tier_tiny,                        out.tables.h_t1_meta,
             kT1MetaPasses);
    consider(table8, in.tier_streams,                     out.tables.h_t3,
             kT3Passes);
    consider(table8, in.tier_tiny,                        out.tables.h_t2_meta,
             kT2MetaPasses);
    consider(table8, in.tier_streams && !in.tier_tiny,    out.tables.h_frags,
             Passes{0, 0}, /*mmap_class=*/true);
    consider(table4, in.tier_streams,                     out.tables.h_t2_xbits,
             xbits_passes);

    // Drain slots are the LAST resort, deliberately. Spilling a table costs
    // disk I/O per plot; giving up a drain slot costs producer/consumer
    // overlap, which is the more expensive of the two once a batch is deeper
    // than one plot. Measured at k=28 compact, n=3: the full spill set is
    // 1.29x baseline wall, a 1-slot drain is 2.91x. At n=1 the slots buy
    // nothing (there is no next plot to overlap with) and cutting them is
    // free — but the policy cannot see the batch depth here, so it just takes
    // the cheaper lever first and stops as soon as the budget is met. "min"
    // (B == 0) still drives to one slot.
    out.pinned_slots = in.pinned_slots;
    if (!in.forced_slots) {
        while ((B == 0 || est > B) && out.pinned_slots > 1) {
            --out.pinned_slots;
            est -= std::min<uint64_t>(table8, est);
        }
    } else {
        est -= std::min<uint64_t>(
            uint64_t(in.baseline_slots - out.pinned_slots) * table8, est);
    }
    out.drain_freed =
        uint64_t(in.baseline_slots - out.pinned_slots) * table8;

    // The reachable floor spills every routable table AND keeps a single
    // drain slot — that is the most this policy can do. Independent of the
    // budget, so it is meaningful even when the budget is unreachable.
    uint64_t const reachable =
        routable_total + uint64_t(in.baseline_slots - 1) * table8;
    out.floor_bytes =
        in.host_required - std::min(reachable, in.host_required);

    out.resident     = est;
    out.meets_budget = !(B != 0 && est > B);
    return out;
}

} // namespace pos2gpu
