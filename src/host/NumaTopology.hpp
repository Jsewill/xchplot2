// NumaTopology.hpp — host NUMA node enumeration and per-node thread affinity.
//
// Why this exists: the CPU plotter is memory-latency-bound (that is the whole
// premise of running several concurrent plots — they interleave each other's
// stalls). On a multi-socket host, a worker whose threads are spread across
// every socket pays a remote-memory penalty on the exact access pattern the
// design leans on. Confining each worker to one node, so its working set is
// allocated and read node-locally, is the fix.
//
// What this canNOT do, and it matters: pos2-chip sizes its fan-out from
// std::thread::hardware_concurrency(), which reports the WHOLE HOST regardless
// of the affinity mask (measured on glibc 2.43 — see nice_current_thread). So a
// worker pinned to one node of an N-socket box still spawns N nodes' worth of
// threads onto that node's cores. Pinning buys locality, not a thread cap, and
// the residual oversubscription is why the per-node worker knee has to be
// re-measured rather than inherited from the single-socket number (~4).
// Capping the fan-out needs pos2-chip to take a thread count, which it does not
// expose at the Plotter level, and it is a FetchContent dependency (gitignored,
// pinned) — so it is not ours to patch in-tree.
//
// Deliberately free of GPU / SYCL headers, and the parsing is split out from
// the syscalls, so the cpulist grammar is unit-testable on synthetic input
// without a multi-socket host to run on (tools/parity/numa_topology_test.cpp).

#pragma once

#include <string>
#include <vector>

namespace pos2gpu {

struct NumaNode {
    int              node_id = 0;
    std::vector<int> cpus;  // logical CPU ids, ascending
};

// Parse a Linux cpulist ("0-15,32", "3", "") into ascending CPU ids.
//
// Split out from the sysfs read purely so it can be tested: this grammar is the
// part that can silently produce a plausible-but-wrong mask, and a wrong mask
// pins a worker to the wrong cores rather than failing.
std::vector<int> parse_cpu_list(std::string const& text);

// Online NUMA nodes and their CPUs, read from /sys/devices/system/node/.
//
// Never returns empty: a host with no NUMA sysfs at all (a kernel built without
// CONFIG_NUMA) yields ONE node with an EMPTY cpu list, meaning "the machine,
// don't pin". Callers therefore do not special-case "no NUMA".
//
// Note that a single-socket host with CONFIG_NUMA — the common case, and every
// rig this was tuned on — is NOT that case: it reports one node that lists all
// its cpus (verified: `online` = "0", `node0/cpulist` = "0-31"). An empty cpu
// list is rarer than "single-node" and the two must not be conflated.
std::vector<NumaNode> host_numa_nodes();

// Confine the CALLING thread to `cpus` — and, because Linux copies the affinity
// mask into children at clone(), everything that thread goes on to spawn. Same
// inheritance trick nice_current_thread() relies on, so it must be called on
// the worker thread before it enters pos2-chip's plotter.
//
// This sets CPU affinity only, and that is deliberate: Linux's default memory
// policy is first-touch, so pages land on the node of the thread that first
// writes them. Pinning BEFORE the working set is allocated therefore places it
// node-locally without an explicit mbind()/set_mempolicy() — which is the whole
// point, and also why the call has to come this early rather than merely before
// the hot loop. (Corollary: the RAM gate in BatchPlotter is host-wide, while the
// memory a pinned worker actually consumes comes from ITS node. Workers are
// handed out round-robin so the two track each other on a symmetric box; on an
// asymmetric one, or one where another tenant has filled a node, a node could
// be oversubscribed while the host-wide figure still looks fine.)
//
// Returns false if the mask could not be set (empty list, sched_setaffinity
// refused, or a platform with no affinity call); callers should carry on
// unpinned rather than fail the plot, since an unpinned worker is slow, not
// wrong.
bool pin_thread_to_cpus(std::vector<int> const& cpus);

// The CPUs the calling thread is currently allowed to run on, ascending. Empty
// if it cannot be read (or on a platform with no affinity call), which callers
// must read as "unknown", never as "none".
std::vector<int> current_thread_cpus();

// Puts the calling thread's affinity mask back the way it was, on scope exit.
//
// Needed because run_batch's single-worker fast path runs the slice on the
// CALLER's thread, not a worker of its own — so a lone CPU worker pinned to a
// node would leave the main thread, and every pool it later spawns (writers,
// FSE), confined to that node for the life of the process. Bench makes that
// concrete: it drives run_batch twice, and the second pass would inherit the
// first's mask.
//
// Restoring does not un-pin the plotter's fan-out, which is the point — those
// threads inherited the mask at clone() and keep it until they exit, which they
// do inside the slice. This only hands back the thread we borrowed.
//
// nice_current_thread has the same hazard and cannot do this: an unprivileged
// process may raise its nice value but never lower it. Affinity is symmetric,
// so the borrow can simply be returned.
class ScopedThreadAffinity {
public:
    ScopedThreadAffinity();   // snapshots the current mask
    ~ScopedThreadAffinity();  // restores it, if the snapshot worked

    ScopedThreadAffinity(ScopedThreadAffinity const&) = delete;
    ScopedThreadAffinity& operator=(ScopedThreadAffinity const&) = delete;

private:
    std::vector<int> saved_;
    bool             valid_ = false;
};

}  // namespace pos2gpu
