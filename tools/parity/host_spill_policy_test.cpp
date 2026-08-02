// host_spill_policy_test — the host-RAM disk-offload budget policy.
//
// plan_host_ram_spill() decides which pinned tables go to disk and how many
// D2H drain slots survive. Get it wrong in one direction and a host that could
// have plotted is refused; get it wrong in the other and the process is
// admitted, allocates past what the box has, and is OOM-killed mid-plot — with
// a partial .plot2 on disk and no diagnostic pointing here.
//
// Reaching these branches for real needs a GPU, a k=28-sized host AND a
// specific free-RAM reading, which is why the policy shipped untested: on the
// 63 GiB dev box every path below is unreachable. XCHPLOT2_HOST_FREE_MB can
// fake the shortage for an end-to-end run, but that is one sample of one
// branch per run. This covers all of them, in milliseconds, on any machine.
//
// Pure arithmetic over integers — no SYCL, no CUDA, no device, no filesystem.

#include "host/HostRamPolicy.hpp"

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

void check_eq(uint64_t got, uint64_t want, char const* what)
{
    bool const ok = (got == want);
    std::printf("%s %s (got %llu, want %llu)\n", ok ? "PASS" : "FAIL", what,
                static_cast<unsigned long long>(got),
                static_cast<unsigned long long>(want));
    if (!ok) ++failures;
}

// The real k=28 shape, so the numbers below are the ones that ship.
// cap = 2^28 + 2^22; the tier constants are GpuBufferPool's B/entry.
constexpr uint64_t kCap      = (uint64_t{1} << 28) + (uint64_t{1} << 22);
constexpr uint64_t kTable8   = 8 * kCap;
constexpr uint64_t kTable4   = 4 * kCap;
constexpr uint64_t kFixed    = uint64_t(1300) << 20;
constexpr uint64_t kTinyNeed = kCap * 80 + kFixed;   // 21.58 GiB
constexpr uint64_t kMinNeed  = kCap * 48 + kFixed;   // 13.46 GiB
constexpr uint64_t kPlainNeed = kCap * 24 + kFixed;   //  7.39 GiB
constexpr uint64_t kGiB      = uint64_t{1} << 30;

// Baseline drain slots (GpuBufferPool::kNumPinnedBuffers). Hard-coded rather
// than included so this test stays free of the GPU headers; if the constant
// moves, the "slots untouched" cases below start failing, which is the
// intended signal.
constexpr int kSlots = 3;

pos2gpu::HostRamSpillInputs tiny_at(uint64_t budget)
{
    pos2gpu::HostRamSpillInputs in;
    in.host_required  = kTinyNeed;
    in.cap_entries    = kCap;
    in.budget         = budget;
    in.tier_tiny      = true;
    in.tier_streams   = true;
    in.pinned_slots   = kSlots;
    in.baseline_slots = kSlots;
    return in;
}

pos2gpu::HostRamSpillInputs minimal_at(uint64_t budget)
{
    pos2gpu::HostRamSpillInputs in = tiny_at(budget);
    in.host_required = kMinNeed;
    in.tier_tiny     = false;
    return in;
}

pos2gpu::HostRamSpillInputs plain_at(uint64_t budget)
{
    pos2gpu::HostRamSpillInputs in = tiny_at(budget);
    in.host_required = kPlainNeed;
    in.tier_tiny     = false;
    in.tier_streams  = false;
    return in;
}

// ---- tier gating -----------------------------------------------------------
// Which tables are routable is a property of the TIER, never of the budget.
// Routing a table the tier cannot service is the corruption case: the pipeline
// would read a buffer nobody wrote, or strand an alias.

void test_tier_gating()
{
    // Plain parks nothing on the host, so there is nothing to route. Even
    // "min" must leave it alone — spilling a table Plain never allocates
    // would leave the pipeline reading an empty TempFile.
    auto const p = pos2gpu::plan_host_ram_spill(plain_at(0));
    check(!p.tables.any(), "plain: nothing routable even at min");
    check_eq(p.spilled_bytes, 0, "plain: nothing spilled");
    check_eq(p.floor_bytes, kPlainNeed - 2 * kTable8,
             "plain: floor is drain slots only");

    // Compact/Minimal: h_t3 + h_frags + h_t2_xbits. NOT h_t1_meta, and NOT
    // h_t2_meta — there it ALIASES h_meta, so routing it strands the alias.
    auto const m = pos2gpu::plan_host_ram_spill(minimal_at(0));
    check(!m.tables.h_t1_meta, "minimal: h_t1_meta stays (tiny-only buffer)");
    check(!m.tables.h_t2_meta, "minimal: h_t2_meta stays (aliases h_meta)");
    check(m.tables.h_t3 && m.tables.h_frags && m.tables.h_t2_xbits,
          "minimal: h_t3 + h_frags + h_t2_xbits routed");

    // Tiny: h_t1_meta + h_t3 + h_t2_meta + h_t2_xbits. NOT h_frags — tiny
    // aliases the rotating D2H slots as device-visible working memory, so the
    // fragment buffer cannot become an mmap.
    auto const t = pos2gpu::plan_host_ram_spill(tiny_at(0));
    check(t.tables.h_t1_meta && t.tables.h_t3 && t.tables.h_t2_meta
              && t.tables.h_t2_xbits,
          "tiny: all four DMA tables routed");
    check(!t.tables.h_frags, "tiny: h_frags stays device-visible");
}

// ---- no-op case ------------------------------------------------------------

void test_fits_without_spilling()
{
    // A budget above the requirement must cost nothing: no I/O, no lost
    // overlap. This is the case an over-eager policy would silently make
    // slower for every user with enough RAM.
    auto const r = pos2gpu::plan_host_ram_spill(tiny_at(kTinyNeed + kGiB));
    check(!r.tables.any(), "fits: nothing routed");
    check_eq(r.pinned_slots, kSlots, "fits: drain slots untouched");
    check_eq(r.resident, kTinyNeed, "fits: peak unchanged");
    check(r.meets_budget, "fits: budget met");
}

// ---- largest-first, and stop as soon as the budget is met ------------------

void test_largest_first_and_minimal_work()
{
    // One 8-B table short. Exactly one table should move, and it must be an
    // 8-B one (h_t1_meta, first in the order) — routing the 4-B h_t2_xbits
    // instead would be two spills' worth of I/O for the same relief.
    auto const r = pos2gpu::plan_host_ram_spill(tiny_at(kTinyNeed - 1));
    check(r.tables.h_t1_meta, "one-short: h_t1_meta goes first");
    check(!r.tables.h_t3 && !r.tables.h_t2_meta && !r.tables.h_t2_xbits,
          "one-short: no other table routed");
    check_eq(r.spilled_bytes, kTable8, "one-short: exactly one 8-B table");
    check_eq(r.resident, kTinyNeed - kTable8, "one-short: peak down by one table");
    check_eq(r.pinned_slots, kSlots, "one-short: drain slots NOT touched");

    // Two tables' worth short: two tables, still no drain cut.
    auto const r2 =
        pos2gpu::plan_host_ram_spill(tiny_at(kTinyNeed - kTable8 - 1));
    check(r2.tables.h_t1_meta && r2.tables.h_t3, "two-short: two 8-B tables");
    check(!r2.tables.h_t2_meta, "two-short: stopped at two");
    check_eq(r2.pinned_slots, kSlots, "two-short: drain slots NOT touched");
}

// ---- drain slots are the last resort, and are walked one at a time ---------

void test_drain_slots_last()
{
    // Past every routable table, the walk starts. A budget one byte under the
    // all-tables peak must give up ONE slot, not all of them: spilling costs
    // I/O per plot, but a drain slot costs producer/consumer overlap, which is
    // worse once the batch is deeper than one plot.
    uint64_t const all_tables = kTinyNeed - 3 * kTable8 - kTable4;
    auto const r = pos2gpu::plan_host_ram_spill(tiny_at(all_tables - 1));
    check_eq(r.pinned_slots, kSlots - 1, "drain: exactly one slot given up");
    check_eq(r.drain_freed, kTable8, "drain: one slot == one 8-B table");
    check_eq(r.resident, all_tables - kTable8, "drain: peak down one more table");

    // Two slots' worth under: down to one slot, the floor.
    auto const r2 =
        pos2gpu::plan_host_ram_spill(tiny_at(all_tables - kTable8 - 1));
    check_eq(r2.pinned_slots, 1, "drain: walks to one slot");
    check_eq(r2.drain_freed, 2 * kTable8, "drain: two slots freed");
    check(r2.meets_budget, "drain: floor budget is met");
}

// ---- "min" ----------------------------------------------------------------

void test_min_takes_everything()
{
    // budget == 0 means "min": route everything and drive the drain to one
    // slot regardless of headroom. It must NOT be read as "a budget of zero
    // bytes, which is unreachable" — that would turn --max-host-ram min into
    // a hard error instead of the most aggressive setting.
    auto const r = pos2gpu::plan_host_ram_spill(tiny_at(0));
    check_eq(r.spilled_bytes, 3 * kTable8 + kTable4, "min: every table routed");
    check_eq(r.pinned_slots, 1, "min: drain cut to one slot");
    check(r.meets_budget, "min: never reports an unmet budget");
    check_eq(r.resident, r.floor_bytes, "min: lands exactly on the floor");
}

// ---- the mmap class is not the unswappable class --------------------------

void test_mmap_class_accounting()
{
    // h_frags spills to a MAP_SHARED file. Those bytes leave the unswappable
    // class (they are written back and evicted under pressure instead of
    // getting the process killed) but stay RESIDENT while there is no
    // pressure. `resident` is the budgeted class; RSS is resident +
    // reclaimable. Report only the first and a user watching top sees a
    // number the tool said would not happen.
    auto const m = pos2gpu::plan_host_ram_spill(minimal_at(0));
    check_eq(m.reclaimable, kTable8, "mmap: h_frags counted as reclaimable");
    check_eq(m.spilled_bytes, 2 * kTable8 + kTable4,
             "mmap: h_frags still counted as spilled I/O");

    // Tiny routes no mmap-class table at all.
    auto const t = pos2gpu::plan_host_ram_spill(tiny_at(0));
    check_eq(t.reclaimable, 0, "mmap: tiny has no reclaimable class");
}

// ---- an unreachable budget reports the floor, and does not lie ------------

void test_unreachable_budget()
{
    // 1 GiB is below anything tiny can reach at k=28. The policy must say so
    // rather than silently returning a plan that does not meet the budget:
    // the caller turns meets_budget==false into either a hard error (explicit
    // --max-host-ram) or a stand-down (automatic spill).
    auto const r = pos2gpu::plan_host_ram_spill(tiny_at(kGiB));
    check(!r.meets_budget, "unreachable: budget reported unmet");
    check(r.resident > kGiB, "unreachable: peak really is over budget");

    // Everything was tried before giving up.
    check_eq(r.spilled_bytes, 3 * kTable8 + kTable4, "unreachable: all routed");
    check_eq(r.pinned_slots, 1, "unreachable: drain walked to the floor");

    // The floor is what the error message quotes. 11177820160 B = 10.41 GiB
    // at k=28 tiny — this exact figure is what a user sees, so pin it.
    check_eq(r.floor_bytes, 11177820160ull, "unreachable: k=28 tiny floor");
    check_eq(r.resident, r.floor_bytes, "unreachable: peak IS the floor");
}

void test_floor_is_budget_independent()
{
    // The floor describes the tier, not the request. If it moved with the
    // budget, the "still needs ~N GiB" in the error would change depending on
    // what the user asked for, which is exactly when they are least able to
    // tell a real limit from a bad guess.
    uint64_t const f_min   = pos2gpu::plan_host_ram_spill(tiny_at(0)).floor_bytes;
    uint64_t const f_low   = pos2gpu::plan_host_ram_spill(tiny_at(kGiB)).floor_bytes;
    uint64_t const f_high  =
        pos2gpu::plan_host_ram_spill(tiny_at(kTinyNeed + kGiB)).floor_bytes;
    check(f_min == f_low && f_low == f_high, "floor: same for any budget");
    check_eq(f_min, 11177820160ull, "floor: k=28 tiny");
    check_eq(pos2gpu::plan_host_ram_spill(minimal_at(0)).floor_bytes,
             4634705920ull, "floor: k=28 minimal");
}

// ---- a forced drain-slot count outranks the policy ------------------------

void test_forced_slots()
{
    // XCHPLOT2_DRAIN_SLOTS pins the count. The policy must not walk it —
    // neither down (the user's number is explicit) nor, by arithmetic
    // accident, into a credit for slots that were never given up.
    pos2gpu::HostRamSpillInputs in = tiny_at(0);
    in.pinned_slots = 2;
    in.forced_slots = true;
    auto const r = pos2gpu::plan_host_ram_spill(in);
    check_eq(r.pinned_slots, 2, "forced: count left alone at min");
    check_eq(r.drain_freed, kTable8, "forced: credited the one slot below baseline");

    // Forced at the baseline: no credit at all.
    pos2gpu::HostRamSpillInputs in3 = tiny_at(0);
    in3.pinned_slots = kSlots;
    in3.forced_slots = true;
    auto const r3 = pos2gpu::plan_host_ram_spill(in3);
    check_eq(r3.drain_freed, 0, "forced: baseline earns no credit");
    check_eq(r3.resident, kTinyNeed - 3 * kTable8 - kTable4,
             "forced: tables only, no drain relief");
}

// ---- saturation ------------------------------------------------------------

void test_no_underflow()
{
    // Small k: the fixed term dominates and the routable total can exceed the
    // requirement. Both subtractions saturate, so neither `resident` nor
    // `floor_bytes` may wrap into a colossal number — which would read as
    // "this needs 16 exabytes" in an error message, or as "peak is fine"
    // depending on which one wrapped.
    pos2gpu::HostRamSpillInputs in = tiny_at(0);
    in.host_required = kTable8;          // less than the routable total
    auto const r = pos2gpu::plan_host_ram_spill(in);
    check_eq(r.resident, 0, "saturate: peak floors at zero, not wrap");
    check_eq(r.floor_bytes, 0, "saturate: floor floors at zero, not wrap");
    check(r.meets_budget, "saturate: min still reports met");

    // Same shape with a real (non-min) budget: still no wrap, still met.
    pos2gpu::HostRamSpillInputs in2 = tiny_at(kGiB);
    in2.host_required = kTable8;
    auto const r2 = pos2gpu::plan_host_ram_spill(in2);
    check(r2.resident <= kTable8, "saturate: bounded peak with a real budget");
    check(r2.meets_budget, "saturate: budget met once everything is routed");
}

// ---- the ladder this feature shipped for ----------------------------------

void test_k28_tiny_ladder()
{
    // The rungs the release notes quote, in bytes, for k=28 tiny. Measured
    // 21.453 GiB unspilled against 21.58 modelled; the model is the contract
    // the policy enforces, so it is what is pinned here.
    check_eq(kTinyNeed, 23173529600ull, "ladder: tiny k=28 requirement");

    // Rung 1 — the three 8-B DMA tables.
    auto const r1 = pos2gpu::plan_host_ram_spill(
        tiny_at(kTinyNeed - 3 * kTable8));
    check_eq(r1.resident, kTinyNeed - 3 * kTable8, "ladder: 3x 8-B tables");
    check_eq(r1.pinned_slots, kSlots, "ladder: rung 1 keeps every slot");

    // Rung 2 — plus the 4-B table.
    auto const r2 = pos2gpu::plan_host_ram_spill(
        tiny_at(kTinyNeed - 3 * kTable8 - kTable4));
    check_eq(r2.spilled_bytes, 3 * kTable8 + kTable4, "ladder: all four tables");
    check_eq(r2.pinned_slots, kSlots, "ladder: rung 2 keeps every slot");

    // Rung 3 — the drain walk, ending on the floor.
    auto const r3 = pos2gpu::plan_host_ram_spill(tiny_at(0));
    check_eq(r3.resident, 11177820160ull, "ladder: floor is 10.41 GiB");
}

} // namespace

int main()
{
    test_tier_gating();
    test_fits_without_spilling();
    test_largest_first_and_minimal_work();
    test_drain_slots_last();
    test_min_takes_everything();
    test_mmap_class_accounting();
    test_unreachable_budget();
    test_floor_is_budget_independent();
    test_forced_slots();
    test_no_underflow();
    test_k28_tiny_ladder();

    if (failures) {
        std::printf("\n%d FAILURE(S)\n", failures);
        return 1;
    }
    std::printf("\nAll host-RAM spill policy checks passed.\n");
    return 0;
}
