// bench_stats_test — unit tests for compute_bench_stats().
//
// The multi-worker bench arithmetic cannot be checked by eye from a log, so it
// is checked here against synthetic timelines with known-correct answers.
//
// The headline case is a reconstruction of the run that motivated the rewrite:
// an RTX-class GPU (~49 s/plot) sharing a 22-plot work-queue with a CPU worker
// (~120 s/plot). The old code merged both workers' completions into one sorted
// list, dropped the first `warmup * workers` entries, and averaged the gaps.
// On this timeline that drops the GPU's first TWO plots, keeps the CPU's
// cold-start plot, and averages in the drain tail where the GPU is idle —
// reporting 37.83 s/plot for a rig that actually sustains 34.79.

#include "host/BenchStats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool check(bool cond, char const* what)
{
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    return cond;
}

bool near(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) <= tol;
}

// The pre-rewrite algorithm, kept here as the thing we are asserting we no
// longer do: merge every worker's completions, sort, drop the first
// `warmup * workers`, mean of the inter-completion gaps.
double legacy_s_per_plot(std::vector<pos2gpu::WorkerTimeline> const& workers,
                         std::size_t warmup_total)
{
    std::vector<double> times;
    for (auto const& w : workers) {
        times.insert(times.end(), w.completion_seconds.begin(),
                     w.completion_seconds.end());
    }
    std::sort(times.begin(), times.end());
    if (times.size() <= warmup_total) return 0.0;
    double const epoch = warmup_total > 0 ? times[warmup_total - 1] : 0.0;
    times.erase(times.begin(),
                times.begin() + static_cast<std::ptrdiff_t>(warmup_total));
    return (times.back() - epoch) / static_cast<double>(times.size());
}

// A worker that runs `count` plots back-to-back: a cold first plot of
// `warm_s`, then `steady_s` apiece, starting after `init_s` of device setup.
pos2gpu::WorkerTimeline make_worker(int dev, double init_s, double warm_s,
                                    double steady_s, std::size_t count)
{
    pos2gpu::WorkerTimeline w;
    w.device_id = dev;
    w.work_start_seconds = init_s;
    double t = init_s + warm_s;
    for (std::size_t i = 0; i < count; ++i) {
        w.completion_seconds.push_back(t);
        t += steady_s;
    }
    return w;
}

}  // namespace

int main()
{
    bool all_ok = true;

    // ---------------------------------------------------------------------
    // 1. Single worker: must be bit-identical to the legacy arithmetic, so
    //    previously published single-GPU numbers stay comparable.
    // ---------------------------------------------------------------------
    {
        // 5 s init, 60 s cold plot (completes at 65), then 10 plots at 49 s.
        auto const gpu = make_worker(0, 5.0, 60.0, 49.0, 11);
        std::vector<pos2gpu::WorkerTimeline> const workers{gpu};
        auto const st = pos2gpu::compute_bench_stats(workers, 1);

        all_ok = check(st.valid, "single: valid") && all_ok;
        all_ok = check(st.workers.size() == 1, "single: one worker") && all_ok;
        all_ok = check(st.workers[0].plots_total == 11,
                       "single: 11 plots finished") && all_ok;
        all_ok = check(st.workers[0].plots_measured == 10,
                       "single: 10 measured (1 warmup dropped)") && all_ok;
        all_ok = check(st.workers[0].past_window == 0,
                       "single: nothing past the window") && all_ok;
        all_ok = check(near(st.s_per_plot, 49.0),
                       "single: 49.00 s/plot") && all_ok;
        all_ok = check(near(st.workers[0].interval_stddev, 0.0),
                       "single: zero variance on a uniform timeline") && all_ok;
        all_ok = check(near(st.workers[0].idle_tail_seconds, 0.0),
                       "single: no idle tail") && all_ok;
        all_ok = check(near(st.s_per_plot, legacy_s_per_plot(workers, 1)),
                       "single: matches legacy arithmetic exactly") && all_ok;
    }

    // ---------------------------------------------------------------------
    // 2. The motivating run: GPU (49 s/plot) + CPU (120 s/plot), 22-plot queue.
    //    The work-queue splits it 15/7 by speed, not 11/11.
    // ---------------------------------------------------------------------
    {
        // GPU: init 5, cold 60 → completes at 65, then 14 more at 49 → last 751.
        auto const gpu = make_worker(0, 5.0, 60.0, 49.0, 15);
        // CPU: init 0.5, cold 150 → completes at 150.5, then 6 more at 120
        //      → last at 870.5, i.e. 119.5 s after the GPU ran dry.
        auto const cpu = make_worker(-2, 0.5, 150.0, 120.0, 7);
        std::vector<pos2gpu::WorkerTimeline> const workers{gpu, cpu};
        auto const st = pos2gpu::compute_bench_stats(workers, 1);

        all_ok = check(st.valid, "mixed: valid") && all_ok;
        all_ok = check(st.workers_unmeasured == 0,
                       "mixed: both workers measurable") && all_ok;

        // The GPU is the first to run dry (751 < 870.5), so it bounds the
        // window and keeps every post-warmup plot it finished.
        all_ok = check(near(st.window_end, 751.0),
                       "mixed: window ends when the GPU runs dry") && all_ok;
        all_ok = check(near(st.window_begin, 150.5),
                       "mixed: window opens at the CPU's warmup completion")
                 && all_ok;

        auto const& g = st.workers[0];
        all_ok = check(g.plots_total == 15 && g.plots_measured == 14,
                       "mixed: gpu 15 finished / 14 measured") && all_ok;
        all_ok = check(near(g.s_per_plot, 49.0),
                       "mixed: gpu measures its true 49.00 s/plot") && all_ok;
        all_ok = check(near(g.idle_tail_seconds, 119.5),
                       "mixed: gpu idle tail of 119.5 s is surfaced") && all_ok;

        auto const& c = st.workers[1];
        all_ok = check(c.plots_total == 7, "mixed: cpu finished 7") && all_ok;
        all_ok = check(c.plots_measured == 5 && c.past_window == 1,
                       "mixed: cpu 5 measured, 1 dropped past the window")
                 && all_ok;
        all_ok = check(near(c.s_per_plot, 120.0),
                       "mixed: cpu measures its true 120.00 s/plot") && all_ok;

        // Aggregate = sum of rates = 14/686 + 5/600 = 0.0287415 plots/s.
        double const expect = 1.0 / (14.0 / 686.0 + 5.0 / 600.0);
        all_ok = check(near(st.s_per_plot, expect, 1e-9),
                       "mixed: aggregate is the sum of per-worker rates")
                 && all_ok;
        all_ok = check(near(st.s_per_plot, 34.7928, 1e-4),
                       "mixed: aggregate 34.79 s/plot") && all_ok;

        // And the legacy arithmetic gets it wrong, in the pessimistic
        // direction, on exactly this timeline. This is the regression.
        double const legacy = legacy_s_per_plot(workers, 2);
        all_ok = check(near(legacy, 37.825, 1e-3),
                       "mixed: legacy reports 37.83 s/plot") && all_ok;
        all_ok = check(legacy > st.s_per_plot + 3.0,
                       "mixed: legacy understates the rig by >3 s/plot")
                 && all_ok;
    }

    // ---------------------------------------------------------------------
    // 3. A worker too slow to finish a measurable plot must not be silently
    //    folded into the aggregate — it makes the result a lower bound.
    // ---------------------------------------------------------------------
    {
        auto const gpu = make_worker(0, 5.0, 60.0, 49.0, 15);   // last at 751
        auto const cpu = make_worker(-2, 0.5, 900.0, 900.0, 1); // one cold plot
        std::vector<pos2gpu::WorkerTimeline> const workers{gpu, cpu};
        auto const st = pos2gpu::compute_bench_stats(workers, 1);

        all_ok = check(st.valid, "starved: still valid") && all_ok;
        all_ok = check(st.workers_unmeasured == 1,
                       "starved: cpu flagged unmeasured") && all_ok;
        all_ok = check(!st.workers[1].measured,
                       "starved: cpu contributes no rate") && all_ok;
        all_ok = check(near(st.s_per_plot, 49.0),
                       "starved: aggregate falls back to the GPU alone")
                 && all_ok;
    }

    // ---------------------------------------------------------------------
    // 4. --warmup 0: the epoch is when the worker started plotting, so device
    //    init is still excluded from the window (it is not per-plot cost).
    // ---------------------------------------------------------------------
    {
        pos2gpu::WorkerTimeline w;
        w.device_id = 0;
        w.work_start_seconds = 5.0;
        w.completion_seconds = {54.0, 103.0};  // 49 s apiece from t=5
        auto const st = pos2gpu::compute_bench_stats({w}, 0);

        all_ok = check(st.valid, "warmup0: valid") && all_ok;
        all_ok = check(st.workers[0].plots_measured == 2,
                       "warmup0: both plots measured") && all_ok;
        all_ok = check(near(st.s_per_plot, 49.0),
                       "warmup0: 49.00 s/plot from the work-start epoch")
                 && all_ok;
    }

    // ---------------------------------------------------------------------
    // 5. Degenerate inputs must not crash or fabricate a rate.
    // ---------------------------------------------------------------------
    {
        auto const empty = pos2gpu::compute_bench_stats({}, 1);
        all_ok = check(!empty.valid, "degenerate: no workers → invalid")
                 && all_ok;

        // A worker that pulled nothing (queue drained before it started) must
        // not bound the window at t=0 and wipe out its peer's measurement.
        pos2gpu::WorkerTimeline idle;
        idle.device_id = -2;
        auto const gpu = make_worker(0, 5.0, 60.0, 49.0, 11);
        auto const st = pos2gpu::compute_bench_stats({gpu, idle}, 1);
        all_ok = check(st.valid, "degenerate: idle peer → still valid")
                 && all_ok;
        all_ok = check(near(st.s_per_plot, 49.0),
                       "degenerate: idle peer does not bound the window")
                 && all_ok;
        all_ok = check(st.workers_unmeasured == 1,
                       "degenerate: idle peer flagged unmeasured") && all_ok;

        // Fewer plots than the warmup asks for → nothing measurable.
        auto const one = make_worker(0, 5.0, 60.0, 49.0, 1);
        auto const st2 = pos2gpu::compute_bench_stats({one}, 1);
        all_ok = check(!st2.valid, "degenerate: warmup eats every plot → invalid")
                 && all_ok;
    }

    return all_ok ? 0 : 1;
}
