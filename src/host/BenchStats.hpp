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

#include <cmath>
#include <cstddef>
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

}  // namespace pos2gpu
