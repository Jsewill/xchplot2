// host_spill_policy_test — the host-RAM disk-offload budget policy.
//
// This decides which full-cap host tables get redirected to a temp dir and how
// many D2H drain slots survive. Get it wrong in one direction and a host that
// could have plotted refuses; get it wrong in the other and a plan routes a
// table on a tier that CPU-touches it, which the pipeline then throws on —
// or worse, would have silently produced a plot missing a table's worth of
// data if it did not.
//
// Reaching these branches for real needs a GPU, a k=28-sized host AND a
// specific free-RAM reading, so on any normal dev box most of them are simply
// unreachable by running the plotter. Here they are integers.
//
// Pure arithmetic — no CUDA, no device, no filesystem, no environment. Take any
// change to the policy through this file, not through a plot run.

#include "host/HostRamPolicy.hpp"

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    std::printf("%-64s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

void check_eq(std::uint64_t got, std::uint64_t want, char const* what)
{
    bool const ok = (got == want);
    std::printf("%-64s %s", what, ok ? "PASS" : "FAIL");
    if (!ok) std::printf("  (got %llu want %llu)",
                         (unsigned long long)got, (unsigned long long)want);
    std::printf("\n");
    if (!ok) ++failures;
}

// The real k=28 shape, so the numbers below are the ones that ship.
// cap = (2^(k - nsb) + 2^(k - emb)) * 2^nsb with nsb = 2, emb = 8.
constexpr std::uint64_t kCap    = ((1ull << 26) + (1ull << 20)) * 4ull;
constexpr std::uint64_t kTable8 = 8ull * kCap;   // h_meta        ~2.031 GiB
constexpr std::uint64_t kTable4 = 4ull * kCap;   // h_t2_xbits    ~1.016 GiB

// Measured on THIS branch, not copied from the SYCL one — its constants are
// 24/44/48/52 against SYCL's 24/52/68/80, and taking the SYCL numbers would
// refuse hosts that plot here perfectly well.
constexpr std::uint64_t kCompactNeed = 12ull * 1024 * 1024 * 1024;

// Baseline drain slots (GpuBufferPool::kNumPinnedBuffers). Hard-coded rather
// than included so this stays free of the GPU headers.
constexpr int kBaselineSlots = 3;

pos2gpu::HostRamSpillInputs base(std::uint64_t budget, bool compact = true)
{
    pos2gpu::HostRamSpillInputs in;
    in.host_required   = kCompactNeed;
    in.cap_entries     = kCap;
    in.budget          = budget;
    in.tier_compact    = compact;
    in.pinned_slots    = kBaselineSlots;
    in.baseline_slots  = kBaselineSlots;
    return in;
}

}  // namespace

int main()
{
    using pos2gpu::plan_host_ram_spill;

    // ---- tier gating: routability is a property of the TIER, never the
    // budget. Routing a table the tier cannot service is the corruption case.
    {
        auto const p = plan_host_ram_spill(base(0, /*compact=*/false));
        check(!p.tables.any(), "non-compact: nothing routable even at min");
        check_eq(p.spilled_bytes, 0, "non-compact: nothing spilled");
        // Drain slots are still available on any tier.
        check_eq(p.pinned_slots, 1, "non-compact: drain still cut to 1 at min");

        auto const c = plan_host_ram_spill(base(0));
        check(c.tables.h_meta && c.tables.h_t2_xbits,
              "compact: both tables route at min");
    }

    // ---- a budget above the requirement must cost NOTHING ----
    {
        auto const r = plan_host_ram_spill(base(kCompactNeed + (1ull << 30)));
        check(!r.tables.any(), "fits: nothing routed");
        check_eq(r.pinned_slots, kBaselineSlots, "fits: all drain slots kept");
        check_eq(r.resident, kCompactNeed, "fits: peak unchanged");
        check_eq(r.traffic_written, 0, "fits: no disk traffic");
        check(r.meets_budget, "fits: budget met");
    }

    // ---- largest-first, and stop as soon as the budget is met ----
    {
        // Just under the requirement: the 8-B table alone should close it, and
        // the 4-B one must NOT be routed.
        auto const p = plan_host_ram_spill(base(kCompactNeed - 1));
        check(p.tables.h_meta, "largest-first: h_meta routed first");
        check(!p.tables.h_t2_xbits,
              "stop-as-soon-as-met: h_t2_xbits left pinned");
        check_eq(p.resident, kCompactNeed - kTable8,
                 "largest-first: peak dropped by exactly one 8-B table");
        check_eq(p.pinned_slots, kBaselineSlots,
                 "drain-last: no slot given up while tables suffice");
    }

    // ---- drain slots are the LAST resort ----
    {
        // Force past both tables so the walk must reach the slots.
        std::uint64_t const target = kCompactNeed - kTable8 - kTable4 - 1;
        auto const p = plan_host_ram_spill(base(target));
        check(p.tables.h_meta && p.tables.h_t2_xbits,
              "drain-last: both tables routed before any slot is cut");
        check(p.pinned_slots < kBaselineSlots, "drain-last: a slot was cut");
        check(p.meets_budget, "drain-last: budget met");
    }

    // ---- "min" drives everything to the floor ----
    {
        auto const p = plan_host_ram_spill(base(0));
        check_eq(p.pinned_slots, 1, "min: drain driven to a single slot");
        check_eq(p.resident, p.floor_bytes, "min: peak equals the floor");
        check_eq(p.drain_freed, 2ull * kTable8, "min: two slots' worth freed");
        check_eq(p.floor_bytes,
                 kCompactNeed - kTable8 - kTable4 - 2ull * kTable8,
                 "min: floor is every table plus two slots");
    }

    // ---- an UNREACHABLE budget must not lie ----
    {
        auto const p = plan_host_ram_spill(base(1));   // 1 byte: impossible
        check(!p.meets_budget, "unreachable: reported as not met");
        check(p.tables.any(), "unreachable: still routes everything it can");
        check_eq(p.pinned_slots, 1, "unreachable: still cuts to one slot");
        check_eq(p.resident, p.floor_bytes,
                 "unreachable: peak is the floor, not zero");
    }

    // ---- forced slots outrank the policy, in BOTH directions ----
    {
        auto in = base(0);
        in.pinned_slots = 3;
        in.forced_slots = true;
        auto const p = plan_host_ram_spill(in);
        check_eq(p.pinned_slots, 3, "forced: 3 slots kept even at min");
        check_eq(p.drain_freed, 0, "forced: no drain credit when none given up");

        // The wrap case: a forced count ABOVE the baseline must not turn into a
        // giant credit and report a peak of zero — the arithmetic failing in
        // the one direction that looks like success.
        auto in2 = base(0);
        in2.pinned_slots   = 5;
        in2.baseline_slots = 3;
        in2.forced_slots   = true;
        auto const q = plan_host_ram_spill(in2);
        check_eq(q.drain_freed, 0, "forced above baseline: credit floored at 0");
        check(q.resident > 0, "forced above baseline: peak did not wrap to 0");
    }

    // ---- traffic model: h_meta crosses the temp dir THREE times each way ----
    {
        auto const p = plan_host_ram_spill(base(0));
        check_eq(p.traffic_written, 3ull * kTable8 + 1ull * kTable4,
                 "traffic: h_meta written 3x (one per role), xbits 1x");
        check_eq(p.traffic_read, 3ull * kTable8 + 1ull * kTable4,
                 "traffic: h_meta read 3x, xbits 1x");
        check_eq(p.spilled_bytes, kTable8 + kTable4,
                 "spilled_bytes counts each table's extent once, not per pass");
    }

    // ---- small k must not saturate into nonsense ----
    {
        auto in = base(0);
        in.cap_entries   = 1024;              // tiny tables
        in.host_required = 64ull << 20;       // 64 MiB
        auto const p = plan_host_ram_spill(in);
        check(p.resident <= in.host_required,
              "small k: peak never exceeds the requirement");
        check_eq(p.pinned_slots, 1, "small k: still drives the drain to 1");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall good\n", failures);
    return failures ? 1 : 0;
}
