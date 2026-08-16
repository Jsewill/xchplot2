// HostRamPolicy.hpp — which host tables to spill, and how many D2H drain slots
// to keep, for a given host-RAM budget.
//
// Pure arithmetic over integers: no CUDA, no device probe, no filesystem, no
// environment. That is deliberate and is the whole reason this is a separate
// translation unit — the logic used to be inline in BatchPlotter::run(), where
// reaching any branch needed a GPU, a k=28-sized host, AND a specific free-RAM
// reading, so on a large dev box most paths were simply unreachable by any
// test. host_spill_policy_test exercises every one of them in milliseconds.
//
// Take any future change to this policy through that test, not through a plot
// run.
//
// TWO RESIDENCY CLASSES, because "how much RAM does this use" has two answers.
// `resident` is the UNSWAPPABLE peak — pinned and anonymous pages, the ones
// that get a process OOM-killed rather than paged out. Pages backed by a
// MAP_SHARED file leave that class (they are written back and evicted under
// pressure) but still count in RSS while there is no pressure, so they are
// tracked separately in `reclaimable`. Reporting only `resident` would read as
// a lie to anyone watching top.

#pragma once

#include "host/GpuPipeline.hpp"   // StreamingPinnedScratch::SpillPlan

#include <cstdint>

namespace pos2gpu {

struct HostRamSpillInputs {
    // Modelled unswappable host peak of the chosen tier, in bytes
    // (streaming_*_host_bytes(k) in the real caller).
    uint64_t host_required = 0;

    // Entries in a full-cap table. h_meta is 8 B/entry and h_t2_xbits is
    // 4 B/entry; those are the only two quanta this policy moves, and a D2H
    // drain slot is also an 8-B table.
    uint64_t cap_entries = 0;

    // Target for the unswappable peak. 0 means "min" — spill everything
    // routable and drive the drain to one slot, regardless of the total.
    uint64_t budget = 0;

    // Which tables are routable is a property of the TIER, not of the budget.
    //
    // Only Compact can route anything today, and that is a real constraint
    // rather than an unfinished edge: Minimal reuses h_meta for Xs staging and
    // merges through h_t2_xbits by direct host indexing, and Tiny hands h_meta
    // to the streaming partition as USM-host and reads h_t2_xbits[orig_idx] at
    // random positions. A SpillBuffer serves SEQUENTIAL device<->disk ranges;
    // those sites need the mmap class instead. The pipeline throws rather than
    // silently ignoring a spill request on those tiers, so a plan that sets a
    // table here for a tier that cannot take it is a bug, not a no-op.
    bool tier_compact = false;

    // Minimal takes the SAME two tables by a different mechanism. It touches
    // them from the CPU — the tiled T1 gather merges through h_t2_xbits by
    // direct indexing and h_meta doubles as Xs staging — which a sequential
    // device<->disk SpillBuffer cannot serve. But it never hands either to a
    // kernel as a dereferenceable pointer: every device access is a plain
    // cudaMemcpyAsync, which works on pageable memory. So Minimal gets the
    // MMAP class instead: a MAP_SHARED TempFile mapping, where CPU indexing
    // works unchanged and the pages are RECLAIMABLE under pressure rather than
    // pinned.
    //
    // Tiny gets neither, and that is not an oversight. It hands h_meta to the
    // streaming partition as USM-HOST — the GPU dereferences it in place — and
    // a file mapping is not device-accessible. Same wall the SYCL branch hit.
    bool tier_minimal = false;

    // D2H drain slots to start from, and whether the user pinned that count
    // themselves (XCHPLOT2_DRAIN_SLOTS). A forced count is honoured even when
    // it leaves the budget unmet — the user's explicit number outranks the
    // policy's preference.
    int  pinned_slots = 0;
    bool forced_slots = false;

    // Slot count `host_required` was modelled at (GpuBufferPool::
    // kNumPinnedBuffers in the real caller). Every slot below this frees one
    // 8-B table. `pinned_slots` must not exceed it — the env parser clamps to
    // [1, baseline] precisely so this cannot go negative and wrap the unsigned
    // credit into a peak of zero.
    int baseline_slots = 0;
};

struct HostRamSpillPlan {
    // Tables the PIPELINE routes through a SpillBuffer (Compact only). These
    // must stay clear for any other tier: the pipeline throws on a spill
    // request it cannot service, deliberately, rather than silently returning a
    // plot missing a table.
    StreamingPinnedScratch::SpillPlan tables;

    // Tables the CALLER should allocate as a MAP_SHARED TempFile mapping
    // instead of pinned host memory (Minimal only). The pipeline never learns
    // about these — a mapping is just a host pointer, so every CPU index and
    // every cudaMemcpyAsync in that path works unchanged.
    bool mmap_h_meta     = false;
    bool mmap_h_t2_xbits = false;

    int pinned_slots = 0;        // slots left after the walk

    uint64_t resident      = 0;  // modelled unswappable peak under this plan
    uint64_t reclaimable   = 0;  // of which file-backed mmap (add for RSS)
    uint64_t spilled_bytes = 0;  // total routed to disk, for the I/O estimate
    uint64_t drain_freed   = 0;  // bytes given up by cutting drain slots

    // Bytes the temp dir must actually HOLD — which is NOT spilled_bytes.
    // spilled_bytes counts each table's extent ONCE, because it answers "how
    // much host RAM did routing this table give back". The disk answers a
    // different question: h_meta is routed as THREE roles, each its own
    // SpillBuffer with its own file, so it occupies 3x its extent on disk
    // while occupying 1x of the RAM saving. Sizing a free-space check from
    // spilled_bytes would under-book by 2x table8 — 4.06 GiB at k=28 — and
    // wave through a temp dir that cannot hold the spill.
    //
    // Deliberately NOT discounted for lifetime overlap. The roles are
    // lifetime-disjoint and each SpillBuffer is destroyed when its role ends,
    // so the true concurrent high-water is lower; but every file is reserved
    // with fallocate at construction, and a spill that fits only because of
    // lifetime overlap is one scheduling change away from ENOSPC mid-plot.
    uint64_t disk_extent = 0;

    // Estimated temp-dir traffic for ONE plot, split by direction. A spilled
    // table is NOT simply written once and read once. h_meta is written and
    // read once per ROLE and it has three of them in Compact (T1 meta park, T2
    // meta park, T3 accumulator), so it makes three of each. h_t2_xbits is
    // written once during T2 match and read once before the T2 sort gather.
    //
    // `traffic_written` is the number that sizes a drive's endurance; the sum
    // of the two is what sizes its throughput.
    //
    // Modelled, not measured. The SpillEngine counts the real bytes and reports
    // them at the end of each plot; if the two disagree, one of them is wrong
    // and that is worth knowing.
    uint64_t traffic_written = 0;
    uint64_t traffic_read    = 0;

    // Lowest peak this policy can reach for these inputs: every routable table
    // spilled AND the drain cut to one slot. Independent of `budget`, so it is
    // the number to quote when a budget is unreachable.
    uint64_t floor_bytes = 0;

    // False when `resident` is still over a non-"min" budget. The caller
    // decides what that means: an explicit --max-host-ram throws, an automatic
    // spill stands down and lets the host-RAM guard speak.
    bool meets_budget = true;
};

// Pure: same inputs, same plan, no environment, no allocation, no I/O.
HostRamSpillPlan plan_host_ram_spill(HostRamSpillInputs const& in);

} // namespace pos2gpu
