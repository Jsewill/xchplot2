#include "host/HostRamPolicy.hpp"

#include <algorithm>

namespace pos2gpu {

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
    auto consider = [&](uint64_t table, bool available, bool& bit,
                        bool mmap_class = false) {
        if (!available) return;
        routable_total += table;
        if (B == 0 || est > B) {
            bit = true;
            est -= std::min(table, est);
            out.spilled_bytes += table;
            if (mmap_class) out.reclaimable += table;
        }
    };

    // Which tables are routable per tier, and why the rest are not, is
    // documented at the call site in BatchPlotter::run().
    consider(table8, in.tier_tiny,                        out.tables.h_t1_meta);
    consider(table8, in.tier_streams,                     out.tables.h_t3);
    consider(table8, in.tier_tiny,                        out.tables.h_t2_meta);
    consider(table8, in.tier_streams && !in.tier_tiny,    out.tables.h_frags,
             /*mmap_class=*/true);
    consider(table4, in.tier_streams,                     out.tables.h_t2_xbits);

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
