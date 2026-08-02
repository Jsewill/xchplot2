// HostRamPolicy.hpp — the host-RAM disk-offload budget policy, as a pure
// function over integers.
//
// This decides which pinned host tables get redirected to a TempFile and how
// many D2H drain slots survive, so that a streaming tier's modelled host peak
// fits a budget. It used to be written inline in BatchPlotter::run(), where it
// was reachable only by owning a GPU, a k=28-sized host, and the right free-RAM
// reading — which meant the arithmetic that decides whether this process gets
// OOM-killed had no test at all. It is pulled out here so it has one.
//
// Deliberately takes the tier's modelled peak and the cap as PARAMETERS rather
// than calling streaming_*_host_bytes() itself: that keeps the policy free of
// SYCL, CUDA and any device probe, so host_spill_policy_test can exercise every
// branch on any machine. BatchPlotter supplies the real numbers.
//
// The budget being enforced is the UNSWAPPABLE class — pinned plus anonymous —
// because that is the class that gets a process killed. Bytes routed to a
// MAP_SHARED file leave that class (they are written back and evicted under
// pressure instead) but stay resident while there is no pressure, so they are
// tracked separately in `reclaimable` and must be reported as RSS on top of
// `resident`. Reporting only `resident` reads as a lie to anyone watching top.

#pragma once

#include "host/GpuPipeline.hpp"   // StreamingPinnedScratch::SpillPlan

#include <cstdint>

namespace pos2gpu {

struct HostRamSpillInputs {
    // Modelled unswappable host peak of the chosen tier, in bytes
    // (streaming_*_host_bytes(k) in the real caller).
    uint64_t host_required = 0;

    // Entries in a full-cap table. One 8-B/entry table is 8·cap, the
    // 4-B/entry one is 4·cap; those are the only two quanta the policy
    // moves, and a D2H drain slot is also an 8-B table.
    uint64_t cap_entries = 0;

    // Target for the unswappable peak. 0 means "min" — spill everything
    // routable and drive the drain to one slot, regardless of the total.
    uint64_t budget = 0;

    // Which tables are routable is a property of the tier, not of the
    // budget. See the table in BatchPlotter for why each one is or is not
    // safe to route.
    bool tier_tiny    = false;   // Tiny or Pinned
    bool tier_streams = false;   // anything but Plain

    // D2H drain slots to start from, and whether the user pinned that
    // count themselves (XCHPLOT2_DRAIN_SLOTS). A forced count is honoured
    // even when it leaves the budget unmet — the user's explicit number
    // outranks the policy's preference.
    int  pinned_slots = 0;
    bool forced_slots = false;

    // Slot count `host_required` was modelled at (GpuBufferPool::
    // kNumPinnedBuffers in the real caller). Every slot below this frees one
    // 8-B table. `pinned_slots` must not exceed it — the env parser clamps
    // to [1, baseline] precisely so this cannot go negative and wrap the
    // unsigned credit into a peak of zero.
    int baseline_slots = 0;
};

struct HostRamSpillPlan {
    StreamingPinnedScratch::SpillPlan tables;

    int pinned_slots = 0;        // slots left after the walk

    uint64_t resident      = 0;  // modelled unswappable peak under this plan
    uint64_t reclaimable   = 0;  // of which file-backed mmap (add for RSS)
    uint64_t spilled_bytes = 0;  // total routed to disk, for the I/O estimate
    uint64_t drain_freed   = 0;  // bytes given up by cutting drain slots

    // Lowest peak this policy can reach for these inputs: every routable
    // table spilled AND the drain cut to one slot. Independent of `budget`,
    // so it is the number to quote when a budget is unreachable.
    uint64_t floor_bytes = 0;

    // False when `resident` is still over a non-"min" budget. The caller
    // decides what that means: an explicit --max-host-ram throws, an
    // automatic spill stands down and lets the host-RAM guard speak.
    bool meets_budget = true;
};

// Pure: same inputs, same plan, no environment, no allocation, no I/O.
HostRamSpillPlan plan_host_ram_spill(HostRamSpillInputs const& in);

} // namespace pos2gpu
