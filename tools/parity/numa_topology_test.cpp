// numa_topology_test — the cpulist grammar behind CPU-node pinning.
//
// This is tested rather than eyeballed because its failure mode is silent: a
// misparsed list yields a plausible mask, and a plausible-but-wrong mask pins a
// worker to the wrong cores instead of erroring. The forms below are what
// /sys/devices/system/node/ actually emits — single ("0"), range ("0-31"), and
// the comma-mixed lists that multi-socket and hyperthread-interleaved hosts
// produce, which is precisely the shape no single-socket dev box can generate.

#include "host/NumaTopology.hpp"

#include <cstdio>
#include <vector>

namespace {

bool check(bool cond, char const* what)
{
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    return cond;
}

bool eq(std::vector<int> const& got, std::vector<int> const& want)
{
    return got == want;
}

}  // namespace

int main()
{
    using pos2gpu::parse_cpu_list;
    bool all_ok = true;

    // The two forms this dev box emits — verified against its real sysfs.
    all_ok = check(eq(parse_cpu_list("0"), {0}), "cpulist: single id") && all_ok;
    all_ok = check(parse_cpu_list("0-31").size() == 32, "cpulist: 0-31 is 32 cpus")
             && all_ok;
    all_ok = check(eq(parse_cpu_list("0-3"), {0, 1, 2, 3}), "cpulist: range expands")
             && all_ok;

    // A two-socket host with hyperthread interleaving: node 0 owns the low half
    // of each thread group. This is the layout the feature exists for.
    all_ok = check(eq(parse_cpu_list("0-15,32-47"),
                      {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                       32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47}),
                   "cpulist: dual-socket interleaved range list") && all_ok;
    all_ok = check(eq(parse_cpu_list("1,3,5"), {1, 3, 5}), "cpulist: comma singles")
             && all_ok;
    all_ok = check(eq(parse_cpu_list("4-4"), {4}), "cpulist: degenerate range")
             && all_ok;

    // Ascending and deduplicated, whatever order sysfs used.
    all_ok = check(eq(parse_cpu_list("5,1,1,3"), {1, 3, 5}),
                   "cpulist: sorted and deduplicated") && all_ok;
    all_ok = check(eq(parse_cpu_list("2-4,3-5"), {2, 3, 4, 5}),
                   "cpulist: overlapping ranges merge") && all_ok;

    // Absent sysfs reads as empty, which must mean "no cpus", never "cpu 0".
    // pin_thread_to_cpus treats empty as "do not pin" — a mask of exactly CPU 0
    // would instead confine the whole plotter to one core.
    all_ok = check(parse_cpu_list("").empty(), "cpulist: empty stays empty") && all_ok;
    all_ok = check(parse_cpu_list("\n").empty(), "cpulist: newline-only stays empty")
             && all_ok;
    all_ok = check(parse_cpu_list("garbage").empty(), "cpulist: unparsable yields nothing")
             && all_ok;
    all_ok = check(eq(parse_cpu_list("0,garbage,2"), {0, 2}),
                   "cpulist: a bad token drops itself, not its neighbours") && all_ok;
    all_ok = check(parse_cpu_list("9-1").empty(), "cpulist: inverted range rejected")
             && all_ok;

    // Never empty: callers do not branch on "this host has no NUMA", they pin to
    // the only node and find it is the whole machine.
    auto const nodes = pos2gpu::host_numa_nodes();
    all_ok = check(!nodes.empty(), "topology: always at least one node") && all_ok;
    std::printf("       (this host: %zu node%s", nodes.size(),
                nodes.size() == 1 ? "" : "s");
    for (auto const& n : nodes) std::printf(", node%d has %zu cpus", n.node_id, n.cpus.size());
    std::printf(")\n");

    // Pinning to nothing must decline rather than build an empty mask and hand
    // it to the kernel.
    all_ok = check(!pos2gpu::pin_thread_to_cpus({}), "pin: empty cpu list declines")
             && all_ok;

    // ScopedThreadAffinity must hand the thread back unchanged. Checked by
    // actually narrowing this thread to one CPU inside a scope and confirming
    // the mask widens again on the way out — the property run_batch's
    // single-worker fast path depends on, since it pins the CALLER's thread.
    {
        auto const before = pos2gpu::current_thread_cpus();
        if (before.size() < 2) {
            std::printf("SKIP scoped-affinity round trip (need >= 2 usable cpus)\n");
        } else {
            {
                pos2gpu::ScopedThreadAffinity guard;
                all_ok = check(pos2gpu::pin_thread_to_cpus({before[0]}),
                               "scoped: narrowed to one cpu inside the scope")
                         && all_ok;
                all_ok = check(pos2gpu::current_thread_cpus().size() == 1,
                               "scoped: the narrowing actually took effect")
                         && all_ok;
            }
            all_ok = check(pos2gpu::current_thread_cpus() == before,
                           "scoped: mask restored exactly on scope exit") && all_ok;
        }
    }

    return all_ok ? 0 : 1;
}
