// HostRamPolicy.cpp — see the header for why this is a separate, pure TU.

#include "host/HostRamPolicy.hpp"

#include <algorithm>

namespace pos2gpu {
namespace {

// Per-plot temp-dir passes over a routed table, by direction.
struct Passes { unsigned writes; unsigned reads; };

// h_meta is routed as THREE independent roles in Compact — the T1 meta park,
// the T2 meta park, and the T3 pairing accumulator — each of which is written
// once and read back once. So the table's cap-sized extent crosses the temp dir
// three times in each direction per plot, not once. Getting this wrong
// understates drive endurance, which is the number that decides whether this
// feature is safe to leave on for months.
constexpr Passes kMetaPasses  {3, 3};

// h_t2_xbits is appended during T2 match and read back once before the T2 sort
// gather. Compact's gather is single-shot, so there is no re-read.
constexpr Passes kXbitsPasses {1, 1};

// Slots given up, floored at zero. A kept count ABOVE the baseline would wrap
// on the cast to uint64_t, saturate the `est` subtraction, and report a
// modelled peak of ZERO for a plan that frees nothing — the budget arithmetic
// failing in the one direction that looks like success.
constexpr uint64_t slots_freed(int baseline, int kept)
{
    return baseline > kept ? uint64_t(baseline - kept) : uint64_t(0);
}

}  // namespace

HostRamSpillPlan plan_host_ram_spill(HostRamSpillInputs const& in)
{
    HostRamSpillPlan out;

    uint64_t const table8 = uint64_t(8) * in.cap_entries;  // h_meta
    uint64_t const table4 = uint64_t(4) * in.cap_entries;  // h_t2_xbits
    uint64_t const B      = in.budget;                     // 0 == "min"

    uint64_t est            = in.host_required;
    uint64_t routable_total = 0;

    // Route LARGEST-FIRST (the 8-B table before the 4-B one) and stop as soon
    // as the budget is met, so a run only pays for the I/O it actually needs.
    // "min" (B == 0) never stops.
    //
    // `passes` is per-table temp-dir traffic, accumulated only for the tables
    // this plan actually routes — a table left pinned costs no disk I/O.
    auto consider = [&](uint64_t table, bool available, bool& bit,
                        Passes passes) {
        if (!available) return;
        routable_total += table;
        if (B == 0 || est > B) {
            bit = true;
            est -= std::min(table, est);
            out.spilled_bytes  += table;
            out.traffic_written += uint64_t(passes.writes) * table;
            out.traffic_read    += uint64_t(passes.reads)  * table;
        }
    };

    consider(table8, in.tier_compact, out.tables.h_meta,     kMetaPasses);
    consider(table4, in.tier_compact, out.tables.h_t2_xbits, kXbitsPasses);

    // Drain slots are the LAST resort, deliberately. Spilling a table costs
    // disk I/O per plot; giving up a drain slot costs producer/consumer
    // overlap, which is the more expensive of the two once a batch is deeper
    // than one plot. At n=1 the slots buy nothing (there is no next plot to
    // overlap with) and cutting them is free — but the policy cannot see the
    // batch depth from here, so it takes the cheaper lever first and stops as
    // soon as the budget is met. "min" (B == 0) still drives to one slot.
    out.pinned_slots = in.pinned_slots;
    if (!in.forced_slots) {
        while ((B == 0 || est > B) && out.pinned_slots > 1) {
            --out.pinned_slots;
            est -= std::min<uint64_t>(table8, est);
        }
    } else {
        est -= std::min<uint64_t>(
            slots_freed(in.baseline_slots, out.pinned_slots) * table8, est);
    }
    out.drain_freed =
        slots_freed(in.baseline_slots, out.pinned_slots) * table8;

    // The reachable floor spills every routable table AND keeps a single drain
    // slot — the most this policy can do. Independent of the budget, so it is
    // meaningful even when the budget is unreachable.
    uint64_t const reachable =
        routable_total + slots_freed(in.baseline_slots, 1) * table8;
    out.floor_bytes =
        in.host_required - std::min(reachable, in.host_required);

    out.resident     = est;
    out.meets_budget = !(B != 0 && est > B);
    return out;
}

} // namespace pos2gpu
