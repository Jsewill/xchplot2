// BenchStats.hpp — throughput statistics for `xchplot2 bench`.
//
// Deliberately free of every GPU / SYCL / filesystem header: the multi-worker
// window arithmetic below is the part of the bench that is easy to get subtly
// wrong and impossible to eyeball from a log, so it has to be unit-testable on
// synthetic timelines (see tools/parity/bench_stats_test.cpp).
//
// Why multi-worker throughput is not just "wall / plots":
//
//   * Workers pull from a shared queue (run_batch's work-queue), so they do
//     NOT finish an equal number of plots — a GPU 3x faster than a CPU takes
//     roughly 3x the plots. "n plots per worker" never happens, and reporting
//     it as though it did is what sent this file's predecessor wrong.
//
//   * The queue is finite, so it runs dry. From the moment the first worker
//     has nothing left to pull, the rig is draining: that worker idles while
//     the rest finish. Wall spent draining is not steady state — the
//     survivors are running under less contention than a full queue gives
//     them — so averaging it in understates the rig.
//
//   * Each worker has its own cold-start plot. Dropping the first `warmup`
//     completions *globally* (which a merged, sorted timeline invites you to
//     do) drops the fast worker's first two plots and keeps the slow worker's
//     cold one.
//
// So: measure each worker over its own window — from its own last warmup
// completion through its last completion that still lands inside the
// all-workers-busy period — and sum the per-worker rates. Summing rates,
// rather than averaging inter-completion gaps on the merged timeline, is what
// makes the answer "what will this rig sustain with a full queue", which is
// the question a plotter operator is actually asking.
//
// For a single worker every rule above degenerates to the obvious thing and
// the result is identical to the pre-existing arithmetic, so single-GPU
// numbers stay comparable with previously published ones.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace pos2gpu {

// One worker's timeline. Offsets are seconds from the RUN's epoch, shared by
// every worker in the batch (run_batch hands the same steady_clock origin to
// each slice). Offsets taken from a per-worker epoch cannot be compared across
// workers and must never be passed here — that skew is invisible in the output
// and silently corrupts every cross-worker window below.
struct WorkerTimeline {
    int device_id = -1;  // kDefaultGpuId / kCpuDeviceId / 0..N-1
    // When this worker began its first plot — i.e. after its device init and
    // pool construction. Only consulted with --warmup 0; otherwise a warmup
    // completion is the epoch and init is excluded by construction.
    double work_start_seconds = 0.0;
    std::vector<double> completion_seconds;  // ascending, one per finished plot
};

struct WorkerStats {
    int device_id = -1;
    std::size_t plots_total = 0;     // everything this worker finished
    std::size_t warmup_dropped = 0;  // excluded as this worker's own cold plots
    std::size_t past_window = 0;     // finished after a peer ran dry (excluded)
    std::size_t plots_measured = 0;
    double s_per_plot = 0.0;  // this worker's own steady-state
    double interval_min = 0.0;
    double interval_max = 0.0;
    double interval_stddev = 0.0;
    // Wall between this worker running dry and the last plot finishing anywhere.
    // Non-zero means it sat idle while a slower peer drained the queue.
    double idle_tail_seconds = 0.0;
    bool measured = false;  // false → contributed nothing to the aggregate
};

struct BenchStats {
    std::vector<WorkerStats> workers;
    double s_per_plot = 0.0;  // aggregate: 1 / sum of per-worker rates
    double plots_per_second = 0.0;
    // The all-workers-busy, all-workers-warm window the rates were taken over.
    double window_begin = 0.0;
    double window_end = 0.0;
    std::size_t plots_measured = 0;
    // > 0 → at least one worker finished too few plots to measure, so it
    // contributed nothing and s_per_plot is a LOWER BOUND on the rig.
    std::size_t workers_unmeasured = 0;
    bool valid = false;
};

inline BenchStats compute_bench_stats(
    std::vector<WorkerTimeline> const& workers,
    std::size_t warmup_per_worker)
{
    BenchStats out;
    if (workers.empty()) return out;

    // The instant the first worker ran out of queue. Past it, at least one
    // worker is idle, so anything finishing later was produced by a rig under
    // less contention than a full queue imposes. It bounds every worker's
    // measurement window.
    //
    // A worker that finished nothing never "ran dry" — it was still on its
    // first plot when the run ended — so it cannot bound the window (and is
    // unmeasurable regardless).
    double window_end = 0.0;
    double run_end = 0.0;
    bool have_end = false;
    for (auto const& w : workers) {
        if (w.completion_seconds.empty()) continue;
        double const last = w.completion_seconds.back();
        if (!have_end || last < window_end) window_end = last;
        if (!have_end || last > run_end) run_end = last;
        have_end = true;
    }
    if (!have_end) return out;  // nothing finished anywhere

    double rate_sum = 0.0;
    double window_begin = 0.0;
    bool have_begin = false;

    for (auto const& w : workers) {
        WorkerStats s;
        s.device_id = w.device_id;
        s.plots_total = w.completion_seconds.size();
        s.warmup_dropped = std::min(warmup_per_worker, s.plots_total);
        s.idle_tail_seconds =
            s.plots_total ? run_end - w.completion_seconds.back() : 0.0;

        // This worker's epoch: its own last warmup completion. Anchoring on a
        // completion — rather than on the first measured plot's start — is what
        // keeps that plot's own production time inside the window. Without it
        // we'd divide N plots by N-1 intervals of wall and inflate the rate by
        // N/(N-1).
        double const epoch = s.warmup_dropped > 0
            ? w.completion_seconds[s.warmup_dropped - 1]
            : w.work_start_seconds;

        std::vector<double> kept;
        for (std::size_t i = s.warmup_dropped; i < s.plots_total; ++i) {
            double const t = w.completion_seconds[i];
            if (t <= window_end) kept.push_back(t);
            else ++s.past_window;
        }

        if (kept.empty() || kept.back() <= epoch) {
            ++out.workers_unmeasured;
            out.workers.push_back(s);
            continue;
        }

        s.plots_measured = kept.size();
        double const window = kept.back() - epoch;
        s.s_per_plot = window / static_cast<double>(kept.size());
        rate_sum += static_cast<double>(kept.size()) / window;

        // Intervals are completion-to-completion within THIS worker, so they
        // are its real per-plot times. The gap between successive completions
        // on the *merged* timeline is a bimodal interleaving artifact and says
        // nothing about per-plot variance — which is why it is not reported.
        std::vector<double> iv;
        iv.reserve(kept.size());
        double prev = epoch;
        for (double t : kept) {
            iv.push_back(t - prev);
            prev = t;
        }
        s.interval_min = iv.front();
        s.interval_max = iv.front();
        for (double v : iv) {
            s.interval_min = std::min(s.interval_min, v);
            s.interval_max = std::max(s.interval_max, v);
        }
        double var = 0.0;
        for (double v : iv) {
            double const d = v - s.s_per_plot;
            var += d * d;
        }
        s.interval_stddev = iv.size() > 1
            ? std::sqrt(var / static_cast<double>(iv.size() - 1))
            : 0.0;

        if (!have_begin || epoch > window_begin) {
            window_begin = epoch;
            have_begin = true;
        }
        s.measured = true;
        out.plots_measured += kept.size();
        out.workers.push_back(s);
    }

    if (rate_sum <= 0.0) return out;  // no worker produced a usable window

    out.plots_per_second = rate_sum;
    out.s_per_plot = 1.0 / rate_sum;
    out.window_begin = window_begin;
    out.window_end = window_end;
    out.valid = true;
    return out;
}

// ---------------------------------------------------------------------------
// Live ETA for the batch progress line.
//
// `remaining x batch_mean` is not an ETA for a work-queue. It is right only
// while the queue is deep enough to keep every worker fed; once it drains, the
// last plots are held by SPECIFIC workers running at their OWN rates, and the
// batch mean stops describing any of them. A real 2-worker run with one plot
// left on a 63 s/plot CPU announced "batch ETA ~6s" — the batch mean, which the
// 7 s/plot GPU had earned — and then took 33 more seconds.
//
// Model: worker i retires a plot every s_i seconds, so its j-th future
// completion lands at base_i + j*s_i. The first in_flight_i of those are already
// committed — those entries are off the queue and no peer can take them. The
// `unclaimed` entries still ON the queue go to whichever worker offers the
// earliest free slot, which is exactly what a work-queue does. The batch ends at
// the latest completion anyone is left holding.
//
// For a single worker this is just base + remaining*s, so single-device ETAs
// keep their old meaning.
struct WorkerLive {
    double      s_per_plot = 0.0;  // observed; <= 0 means "no estimate yet"
    double      last_done  = 0.0;  // run-epoch seconds of its latest completion
    std::size_t in_flight  = 0;    // pulled off the queue, not yet retired
};

struct EtaEstimate {
    double seconds = 0.0;
    // A worker holding a plot but with no completed plot yet has no rate, so it
    // cannot be priced — and inventing one for it (the batch mean, say) is how
    // you get a 63 s/plot CPU billed at the GPU's 7 s. It is dropped from the
    // model instead, which makes `seconds` a floor: the batch cannot finish
    // before that worker's plot does, and that plot could take any amount of
    // time. Callers must present the number as "at least", not "about".
    bool lower_bound = false;
};

// Time from `now` (run-epoch seconds) until the last worker retires the last
// plot. Zero when there is nothing left to do.
inline EtaEstimate estimate_eta_seconds(std::vector<WorkerLive> const& live,
                                        std::size_t unclaimed,
                                        double now)
{
    EtaEstimate out;

    struct Train { double base; double s; std::size_t committed; };
    std::vector<Train> ws;
    ws.reserve(live.size());
    for (auto const& l : live) {
        if (!(l.s_per_plot > 0.0)) {
            // Unmodellable. If it is holding work, whatever we return is a floor.
            if (l.in_flight > 0) out.lower_bound = true;
            continue;
        }
        // Phase of this worker's completion train. A worker that is overdue is
        // mid-plot, not instantly done, so hold its next completion at `now`
        // rather than letting a stale last_done predict completions in the past.
        ws.push_back(Train{std::max(l.last_done, now - l.s_per_plot),
                           l.s_per_plot, l.in_flight});
    }
    if (ws.empty()) return out;

    // Committed work: the in-flight plots land s apart and cannot be reassigned.
    double finish = now;
    for (auto const& w : ws) {
        if (w.committed > 0) {
            finish = std::max(
                finish, w.base + static_cast<double>(w.committed) * w.s);
        }
    }
    if (unclaimed == 0) {
        out.seconds = std::max(0.0, finish - now);
        return out;
    }

    // Queued work: handing each entry to the earliest-free worker means the LAST
    // queued entry lands at the `unclaimed`-th smallest free slot across all
    // workers, where worker i's free slots are base_i + (committed_i + j)*s_i for
    // j >= 1. Binary-search for that instant rather than simulating the handout
    // one entry at a time: a manifest can hold 100k entries and this runs once
    // per completion, so the simulation would be quadratic in the batch size.
    auto free_slots_by = [&](double t) -> std::size_t {
        std::size_t n = 0;
        for (auto const& w : ws) {
            double const issued = std::floor((t - w.base) / w.s);
            double const avail = issued - static_cast<double>(w.committed);
            if (avail <= 0.0) continue;
            if (avail >= static_cast<double>(unclaimed)) return unclaimed;
            n += static_cast<std::size_t>(avail);
            if (n >= unclaimed) return unclaimed;
        }
        return n;
    };

    // Upper bound: whichever single worker could absorb every queued entry
    // soonest on its own necessarily has them all done by then.
    double lo = now;
    double hi = std::numeric_limits<double>::max();
    for (auto const& w : ws) {
        hi = std::min(hi, w.base + static_cast<double>(w.committed + unclaimed) * w.s);
    }
    for (int it = 0; it < 100 && (hi - lo) > 1e-6; ++it) {
        double const mid = lo + 0.5 * (hi - lo);
        if (free_slots_by(mid) >= unclaimed) hi = mid;
        else                                 lo = mid;
    }
    out.seconds = std::max(0.0, std::max(finish, hi) - now);
    return out;
}

// ---------------------------------------------------------------------------
// Batch sizes that leave nobody waiting.
//
// The work-queue hands the next plot to whichever worker frees up first, so it
// already splits a batch in proportion to the workers' rates without being told
// them. What it cannot do is split a PLOT. Beside a 63 s/plot CPU, a 8.79
// s/plot GPU is worth 7.17 of it, so the CPU's fair share of a 10-plot batch is
// 1.22 plots — and the queue has to hand it 1 or 2. At 1 the GPU covers the
// other 9 and runs 7 s past the CPU; at 2 the CPU grinds a second plot while
// the GPU sits idle for 56 s. Either way somebody waits, and waiting is rig you
// paid for and did not use.
//
// At some batch sizes the fair shares land on (near-)whole plots and nobody
// waits: on that rig 49 plots split 43/6 and both workers finish within 0.03 s
// of each other. Those sizes are what this finds. They are a property of the
// RATES, not of the batch, so the answer holds for any multiple — the split
// scales with it and the idle stays the same fraction of the run.
//
// Two answers, because they trade off:
//   * `workable` — the smallest size that wastes little. Easy to act on.
//   * `exact`    — the smallest size where the landing is essentially perfect.
//     Always >= workable, and can be an awkward number, but its multiples stay
//     tight where a workable size's error compounds in absolute terms (that
//     8-plot split idles 1.47 s; 16 idles 2.94 s; 24 idles 4.41 s).
//
// The search simulates the scheduler rather than rounding the fair shares:
// list-scheduling identical jobs onto uniform machines is precisely what
// fetch_add-on-a-shared-queue does, so modelling it directly costs nothing and
// removes the question of whether the rounding agrees with it. And because
// greedy assignment is incremental — job j's destination does not depend on how
// many jobs come after it — ONE pass out to the cap yields the split for every
// size along the way as its own prefix, instead of a fresh simulation per size.
//
// Every worker gets at least one plot by construction: they all start free, so
// the first W jobs go one apiece. Sizes below W are therefore never considered
// — a size where a device plots nothing is not a size where every device lands
// together, it is a size where you should have used fewer devices.

// A worker's share of a batch sized by pick_batch_sizing().
struct WorkerSplit {
    int         device_id = -1;
    // Index into the caller's own s_per_plot / device_ids arrays. Unrated
    // workers are dropped from the split, so this is NOT the position in
    // `split` — and it is what a caller must use to name the worker:
    // worker_labels() numbers repeats by their position in the FULL device
    // list ("cpu#0", "cpu#1"), so an unrated CPU in the middle would shift
    // every later ordinal if the labels were rebuilt from the split alone.
    std::size_t worker_index = 0;
    std::size_t plots = 0;
    double      finish_seconds = 0.0;  // its last plot lands here, batch-relative
};

struct BatchSizing {
    std::size_t plots = 0;  // the batch size this describes ("N")
    std::vector<WorkerSplit> split;
    // Wall between the first worker going idle and the last one finishing. This
    // is the waste: every worker but one is done and holding still for it.
    double      spread_seconds = 0.0;
    double      makespan_seconds = 0.0;
    double      idle_fraction = 0.0;  // spread / makespan
    bool        valid = false;
};

struct OptimalBatch {
    BatchSizing workable;
    BatchSizing exact;
    // The two searches landed on the same size — the smallest size that wastes
    // little already wastes ~nothing. Callers should print one answer, not two.
    bool        coincide = false;
    std::size_t workers_modelled = 0;
    // Workers with no rate yet. They are left out of the model entirely rather
    // than priced at a peer's rate, so a size chosen here does not account for
    // them — see the same reasoning on EtaEstimate::lower_bound.
    std::size_t workers_unrated = 0;
    bool        valid = false;
};

// Idle you would not bother restructuring a batch to recover.
inline constexpr double kWorkableIdleFraction = 0.05;
// Idle indistinguishable from run-to-run noise.
inline constexpr double kExactIdleFraction = 0.001;
// Far past any batch a person would size by hand, and the walk out to it is
// O(cap * workers) total — the incremental-prefix property above means raising
// it is nearly free.
inline constexpr std::size_t kMaxBatchSearch = 4096;

// s_per_plot / device_ids are parallel; a worker with s_per_plot <= 0 has no
// rate and is dropped. Returns valid=false when fewer than 2 workers have one,
// since a lone worker's plots always land in a row and there is nothing to
// balance.
inline OptimalBatch pick_batch_sizing(std::vector<double> const& s_per_plot,
                                      std::vector<int> const& device_ids,
                                      std::size_t max_n = kMaxBatchSearch)
{
    OptimalBatch out;

    std::vector<double>      s;
    std::vector<int>         ids;
    std::vector<std::size_t> origin;  // back-reference into the caller's arrays
    for (std::size_t i = 0; i < s_per_plot.size(); ++i) {
        if (s_per_plot[i] > 0.0) {
            s.push_back(s_per_plot[i]);
            ids.push_back(i < device_ids.size() ? device_ids[i] : -1);
            origin.push_back(i);
        } else {
            ++out.workers_unrated;
        }
    }
    out.workers_modelled = s.size();
    if (s.size() < 2) return out;
    if (max_n < s.size()) return out;

    // free[i] is when worker i next comes free — and, since it starts free and
    // plots back to back, also when its last plot so far landed. So the finish
    // times fall out of the simulation with no second pass.
    std::vector<double>      free_at(s.size(), 0.0);
    std::vector<std::size_t> counts(s.size(), 0);

    auto snapshot = [&](std::size_t n) {
        BatchSizing b;
        b.plots = n;
        b.split.reserve(s.size());
        double lo = free_at[0], hi = free_at[0];
        for (std::size_t i = 0; i < s.size(); ++i) {
            b.split.push_back(
                WorkerSplit{ids[i], origin[i], counts[i], free_at[i]});
            lo = std::min(lo, free_at[i]);
            hi = std::max(hi, free_at[i]);
        }
        b.spread_seconds = hi - lo;
        b.makespan_seconds = hi;
        b.idle_fraction = hi > 0.0 ? (hi - lo) / hi : 0.0;
        b.valid = true;
        return b;
    };

    for (std::size_t n = 1; n <= max_n; ++n) {
        // Next plot goes to whoever is free soonest. Ties break to the lowest
        // index, which is what puts one plot on each worker before any worker
        // takes a second — see the note on sizes below W above.
        std::size_t pick = 0;
        for (std::size_t i = 1; i < s.size(); ++i) {
            if (free_at[i] < free_at[pick]) pick = i;
        }
        free_at[pick] += s[pick];
        ++counts[pick];

        if (n < s.size()) continue;  // somebody is still on zero plots

        double lo = free_at[0], hi = free_at[0];
        for (double f : free_at) { lo = std::min(lo, f); hi = std::max(hi, f); }
        double const idle = hi > 0.0 ? (hi - lo) / hi : 0.0;

        if (!out.workable.valid && idle <= kWorkableIdleFraction) {
            out.workable = snapshot(n);
        }
        if (idle <= kExactIdleFraction) {
            out.exact = snapshot(n);
            // Both searches want the SMALLEST qualifying size, and any size
            // this tight is trivially workable too — so if the looser search is
            // still empty here, this size is its answer as well.
            if (!out.workable.valid) out.workable = out.exact;
            break;
        }
    }

    out.valid = out.workable.valid;
    out.coincide = out.exact.valid && out.workable.plots == out.exact.plots;
    return out;
}

}  // namespace pos2gpu
