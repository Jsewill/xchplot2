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

// Dependency-free (it is <string> and two constants), so it does not compromise
// the no-GPU-headers rule BenchStats.hpp keeps for testability.
#include "gpu/DeviceIds.hpp"

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

// The queue handout estimate_eta_seconds() is standing in for: give each queued
// entry to whichever worker's next free slot is earliest, one entry at a time,
// exactly as the work-queue does. Correct but O(queue x workers), and the real
// path runs it on every completion of a manifest that can hold 100k entries — so
// the shipped version binary-searches for the same instant. Asserting the two
// agree is what licenses that substitution.
double greedy_eta(std::vector<pos2gpu::WorkerLive> const& live,
                  std::size_t unclaimed, double now)
{
    struct Train { double base; double s; std::size_t n; };
    std::vector<Train> ws;
    for (auto const& l : live) {
        if (!(l.s_per_plot > 0.0)) continue;
        ws.push_back(Train{std::max(l.last_done, now - l.s_per_plot),
                           l.s_per_plot, l.in_flight});
    }
    if (ws.empty()) return 0.0;

    double finish = now;
    for (auto const& w : ws) {
        if (w.n > 0) finish = std::max(finish, w.base + double(w.n) * w.s);
    }
    for (std::size_t u = 0; u < unclaimed; ++u) {
        std::size_t best = 0;
        double best_t = ws[0].base + double(ws[0].n + 1) * ws[0].s;
        for (std::size_t i = 1; i < ws.size(); ++i) {
            double const t = ws[i].base + double(ws[i].n + 1) * ws[i].s;
            if (t < best_t) { best_t = t; best = i; }
        }
        ++ws[best].n;
        finish = std::max(finish, best_t);
    }
    return std::max(0.0, finish - now);
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

    // ---------------------------------------------------------------------
    // --warmup 0: the epoch is work_start_seconds, so per-batch SETUP (device
    // bind, pool construction, the tier probe) must sit outside the window.
    //
    // This is the case that had no test, and its absence let a real bug ship:
    // run_batch_sharded and run_batch_pipeline_plot never assigned
    // work_start_seconds, so it stayed 0.0 and the whole multi-GPU setup got
    // amortised across every plot — silently understating those rigs by a few
    // percent, and only under --warmup 0. Pin the invariant here so a path that
    // forgets to publish its work start fails loudly instead of quietly.
    // ---------------------------------------------------------------------
    {
        // Same 10 plots at 49 s, but one rig spent 5 s in setup and the other 90.
        auto const quick = make_worker(0, 5.0, 49.0, 49.0, 10);
        auto const slow_setup = make_worker(0, 90.0, 49.0, 49.0, 10);
        auto const a = pos2gpu::compute_bench_stats({quick}, 0);
        auto const b = pos2gpu::compute_bench_stats({slow_setup}, 0);

        all_ok = check(near(a.s_per_plot, 49.0),
                       "warmup 0: epoch is work_start, so setup is excluded")
                 && all_ok;
        all_ok = check(near(a.s_per_plot, b.s_per_plot),
                       "warmup 0: 85 s more setup must not change s/plot") && all_ok;

        // And the failure it is guarding against: a path that leaves
        // work_start_seconds at 0 charges its setup to the plots.
        auto forgot = slow_setup;
        forgot.work_start_seconds = 0.0;
        auto const c = pos2gpu::compute_bench_stats({forgot}, 0);
        all_ok = check(c.s_per_plot > 57.0,
                       "warmup 0: unset work_start inflates s/plot (the bug)")
                 && all_ok;
    }

    // ---------------------------------------------------------------------
    // RateWindow: the run report asks a different question than the bench, and
    // asking the bench's question of a finished run destroys it.
    //
    // The timeline is a real reported run: `plot -n 4 --devices cpu`, four CPU
    // workers on a 32-core box, one plot each, k=28. It is the worst case for
    // FullQueue by construction — a batch of one round. A batch ENDS by
    // draining, so the earliest worker's last completion always bounds the
    // window, and here that is every plot anyone has. Three of four workers
    // then report no rate, the run's own sizing advice cannot be computed at
    // all, and the aggregate misses reality by 4x.
    //
    // The reasons matter as much as the rates: those three workers each
    // finished exactly as many plots as the one that WAS measured, so a
    // printer that assumes "unmeasured == too few plots" tells their operator
    // to raise -n against a problem they do not have.
    // ---------------------------------------------------------------------
    {
        auto one_plot = [](int dev, double at) {
            pos2gpu::WorkerTimeline w;
            w.device_id = dev;
            w.work_start_seconds = 0.0;  // CPU workers have no device init
            w.completion_seconds = {at};
            return w;
        };
        std::vector<pos2gpu::WorkerTimeline> const run{
            one_plot(pos2gpu::cpu_device_id(0), 165.59),
            one_plot(pos2gpu::cpu_device_id(0), 165.33),  // earliest → bounds it
            one_plot(pos2gpu::cpu_device_id(0), 166.10),  // last → the real end
            one_plot(pos2gpu::cpu_device_id(0), 165.69),
        };

        // What the run report used to do, and must never do again.
        auto const clipped = pos2gpu::compute_bench_stats(run, 0);
        all_ok = check(clipped.workers_unmeasured == 3,
                       "window: FullQueue discards 3 of 4 one-plot workers")
                 && all_ok;
        all_ok = check(near(clipped.s_per_plot, 165.33),
                       "window: FullQueue aggregate is ~4x reality (the bug)")
                 && all_ok;
        for (std::size_t i : {std::size_t{0}, std::size_t{2}, std::size_t{3}}) {
            all_ok = check(clipped.workers[i].why ==
                               pos2gpu::Unmeasured::AllPastWindow,
                           "window: excluded for landing late, NOT for plot count")
                     && all_ok;
            all_ok = check(clipped.workers[i].plots_total == 1 &&
                               clipped.workers[i].past_window == 1,
                           "window: it finished as many plots as the one kept")
                     && all_ok;
        }

        // What it does now.
        auto const whole =
            pos2gpu::compute_bench_stats(run, 0, pos2gpu::RateWindow::WholeRun);
        all_ok = check(whole.valid && whole.workers_unmeasured == 0,
                       "window: WholeRun measures every worker") && all_ok;
        all_ok = check(near(whole.workers[0].s_per_plot, 165.59) &&
                           near(whole.workers[1].s_per_plot, 165.33) &&
                           near(whole.workers[2].s_per_plot, 166.10) &&
                           near(whole.workers[3].s_per_plot, 165.69),
                       "window: WholeRun rates are each worker's own plot")
                 && all_ok;
        // The batch really wrote 4 plots in 166.097 s = 41.52 s/plot. The
        // aggregate is a sum of per-worker rates, not that division, so it is
        // not required to equal it — but it must not disagree materially, and
        // agreeing to 0.3% is what makes the block readable next to the
        // summary line directly above it.
        all_ok = check(std::fabs(whole.s_per_plot - 41.524) < 0.15,
                       "window: WholeRun aggregate agrees with wall/plots")
                 && all_ok;
        // Idle tail is measured against the last completion anywhere, so it is
        // unaffected by the window and stays true in both modes.
        all_ok = check(near(whole.workers[1].idle_tail_seconds, 0.77) &&
                           near(whole.workers[2].idle_tail_seconds, 0.0),
                       "window: idle tail is unchanged by the window mode")
                 && all_ok;

        // Every rated worker means the run can finally size its next batch.
        std::vector<double> rates;
        std::vector<int> ids;
        for (auto const& w : whole.workers) {
            rates.push_back(w.s_per_plot);
            ids.push_back(w.device_id);
        }
        auto const sized = pos2gpu::pick_batch_sizing(rates, ids);
        all_ok = check(sized.valid && sized.workers_modelled == 4 &&
                           sized.workers_unrated == 0,
                       "window: WholeRun rates make the run sizable") && all_ok;
    }

    // ---------------------------------------------------------------------
    // The other unmeasured reasons, so the printers can name each one.
    // ---------------------------------------------------------------------
    {
        // Finished nothing: still on its first plot when the batch ended.
        pos2gpu::WorkerTimeline idle;
        idle.device_id = pos2gpu::cpu_device_id(0);
        auto const busy = make_worker(0, 0.0, 49.0, 49.0, 4);
        auto const st = pos2gpu::compute_bench_stats(
            {busy, idle}, 0, pos2gpu::RateWindow::WholeRun);
        all_ok = check(st.workers[1].why == pos2gpu::Unmeasured::NoPlots,
                       "reason: a worker that finished nothing says so") && all_ok;

        // Everything it finished was warmup (test 3's starved CPU, named).
        auto const gpu = make_worker(0, 5.0, 60.0, 49.0, 15);
        auto const cold = make_worker(pos2gpu::cpu_device_id(0), 0.5, 900.0,
                                      900.0, 1);
        auto const st2 = pos2gpu::compute_bench_stats({gpu, cold}, 1);
        all_ok = check(st2.workers[1].why == pos2gpu::Unmeasured::AllWarmup,
                       "reason: a worker with only warmup plots says so") && all_ok;
    }

    // ---- estimate_eta_seconds -------------------------------------------
    {
        using pos2gpu::WorkerLive;
        using pos2gpu::estimate_eta_seconds;

        // The case that motivated the fix. At the instant the GPU retired the
        // last queued plot (t=325.3), the queue was empty, the GPU was idle, and
        // the CPU was 9.8 s into a plot it had started at 315.5. The old ETA was
        // remaining x batch-mean = 1 x (325.3/51) = 6.4 s. It took 32.6 more.
        {
            std::vector<WorkerLive> live = {
                WorkerLive{6.93, 325.3, 0},   // gpu: done, nothing in flight
                WorkerLive{63.70, 315.5, 1},  // cpu: one plot in flight
            };
            double const eta = estimate_eta_seconds(live, 0, 325.3).seconds;
            all_ok = check(near(eta, 53.9, 0.05),
                           "eta: drain tail prices the in-flight CPU plot") && all_ok;
            // What the old arithmetic said, kept as the thing we no longer do.
            double const legacy = (325.3 / 51.0) * 1.0;
            all_ok = check(near(legacy, 6.378, 0.01) && eta > 8.0 * legacy,
                           "eta: old batch-mean understated the tail by >8x") && all_ok;
            // Truth was 32.6 s (the CPU sped up once the GPU stopped competing
            // for cores). Erring long beats claiming the batch is 6 s from done.
            all_ok = check(eta > 32.6, "eta: errs long, not short, on the tail")
                     && all_ok;
        }

        // Nothing in flight and nothing queued → the batch is over.
        {
            std::vector<WorkerLive> live = {WorkerLive{7.0, 100.0, 0}};
            all_ok = check(near(estimate_eta_seconds(live, 0, 100.0).seconds, 0.0),
                           "eta: idle rig with an empty queue → 0") && all_ok;
        }

        // One worker: base + remaining*s, i.e. exactly the old meaning.
        {
            std::vector<WorkerLive> live = {WorkerLive{7.0, 100.0, 1}};
            all_ok = check(near(estimate_eta_seconds(live, 9, 100.0).seconds, 70.0, 1e-3),
                           "eta: single worker degenerates to remaining * s")
                     && all_ok;
        }

        // Deep queue: converges on the aggregate rate (the sum of the per-worker
        // rates), which is the regime the old batch-mean got right.
        {
            std::vector<WorkerLive> live = {
                WorkerLive{7.0, 100.0, 1},
                WorkerLive{63.0, 95.0, 1},
            };
            double const eta = estimate_eta_seconds(live, 200, 100.0).seconds;
            double const aggregate = 200.0 / (1.0 / 7.0 + 1.0 / 63.0);
            all_ok = check(near(eta, aggregate, 0.05 * aggregate),
                           "eta: deep queue converges on the summed rate") && all_ok;
        }

        // A worker that has finished nothing has no rate, so it cannot be
        // modelled — and must not be credited with capacity it has not shown.
        {
            std::vector<WorkerLive> live = {
                WorkerLive{7.0, 100.0, 1},
                WorkerLive{0.0, 0.0, 1},  // no completion yet
            };
            auto const est = estimate_eta_seconds(live, 9, 100.0);
            double const eta = est.seconds;
            all_ok = check(near(eta, 70.0, 1e-3),
                           "eta: rate-less worker excluded, not invented") && all_ok;
            // ...but it is holding a plot, and the batch cannot end before that
            // plot does. Dropping it silently is the same optimism this whole
            // estimator exists to kill, so the number has to be flagged a floor.
            all_ok = check(est.lower_bound,
                           "eta: busy rate-less worker makes the ETA a floor")
                     && all_ok;
            auto const idle_est = estimate_eta_seconds(
                {WorkerLive{7.0, 100.0, 1}, WorkerLive{0.0, 0.0, 0}}, 9, 100.0);
            all_ok = check(!idle_est.lower_bound,
                           "eta: an idle rate-less worker holds nothing → not a floor")
                     && all_ok;
        }

        // The binary search must agree with the handout it is standing in for.
        // This is the assertion that makes the closed form safe: if they ever
        // disagree, the fast path is wrong.
        {
            bool agree = true;
            double worst = 0.0;
            for (int a = 1; a <= 9; ++a) {
                for (int b = 1; b <= 9; ++b) {
                    for (std::size_t f1 = 0; f1 <= 2; ++f1) {
                        for (std::size_t f2 = 0; f2 <= 2; ++f2) {
                            for (std::size_t q : {std::size_t(0), std::size_t(1),
                                                  std::size_t(3), std::size_t(17),
                                                  std::size_t(64)}) {
                                std::vector<WorkerLive> live = {
                                    WorkerLive{double(a) * 1.7, 100.0 - a, f1},
                                    WorkerLive{double(b) * 9.3, 100.0 - b, f2},
                                    WorkerLive{double(a + b), 100.0, f1 + f2},
                                };
                                double const fast =
                                    estimate_eta_seconds(live, q, 100.0).seconds;
                                double const slow = greedy_eta(live, q, 100.0);
                                worst = std::max(worst, std::fabs(fast - slow));
                                if (!near(fast, slow, 1e-4)) agree = false;
                            }
                        }
                    }
                }
            }
            std::printf("       (worst |binary-search - greedy| = %.2e s)\n", worst);
            all_ok = check(agree,
                           "eta: closed form == brute-force queue handout (3645 cases)")
                     && all_ok;
        }
    }

    // ---------------------------------------------------------------------
    // pick_batch_sizing(): batch sizes where every worker lands together.
    //
    // The headline case is the rig this was written for: a 8.79 s/plot GPU
    // beside a 63 s/plot CPU. Hand-worked so the expectations are independent
    // of the implementation --
    //
    //   The CPU is worth 8.79/63 of the GPU, so its fair share of a batch is
    //   8.79/(8.79+63) = 0.12244 of it. Multiply out:
    //     8 plots  -> 0.9795 CPU, 7.0205 GPU. The queue hands it 7/1: the GPU
    //                 lands at 7*8.79 = 61.53, the CPU at 63.0. Idle 1.47 s,
    //                 2.33% of the run -- inside the 5% workable bar.
    //    49 plots  -> 5.9995 CPU, 43.0005 GPU. 43/6: GPU 43*8.79 = 377.97,
    //                 CPU 6*63 = 378.0. Idle 0.03 s, 0.008% -- inside the
    //                 0.1% exact bar.
    // ---------------------------------------------------------------------
    {
        using pos2gpu::pick_batch_sizing;
        auto const opt = pick_batch_sizing({8.79, 63.0}, {0, pos2gpu::kCpuDeviceId});

        all_ok = check(opt.valid, "sizing: valid") && all_ok;
        all_ok = check(opt.workers_modelled == 2 && opt.workers_unrated == 0,
                       "sizing: both workers modelled") && all_ok;

        all_ok = check(opt.workable.plots == 8, "sizing: workable is 8 plots")
                 && all_ok;
        all_ok = check(opt.workable.split[0].plots == 7
                           && opt.workable.split[1].plots == 1,
                       "sizing: 8 splits gpu 7 / cpu 1") && all_ok;
        all_ok = check(near(opt.workable.split[0].finish_seconds, 61.53, 1e-9),
                       "sizing: 8 -> gpu lands at 61.53 s") && all_ok;
        all_ok = check(near(opt.workable.split[1].finish_seconds, 63.0, 1e-9),
                       "sizing: 8 -> cpu lands at 63.00 s") && all_ok;
        all_ok = check(near(opt.workable.spread_seconds, 1.47, 1e-9),
                       "sizing: 8 -> 1.47 s idle") && all_ok;
        all_ok = check(near(opt.workable.idle_fraction, 1.47 / 63.0, 1e-9),
                       "sizing: 8 -> 2.33% idle") && all_ok;

        all_ok = check(opt.exact.plots == 49, "sizing: exact is 49 plots") && all_ok;
        all_ok = check(opt.exact.split[0].plots == 43
                           && opt.exact.split[1].plots == 6,
                       "sizing: 49 splits gpu 43 / cpu 6") && all_ok;
        all_ok = check(opt.exact.spread_seconds < 0.05,
                       "sizing: 49 -> under 0.05 s idle") && all_ok;
        all_ok = check(!opt.coincide, "sizing: 8 != 49, so both are worth printing")
                 && all_ok;
        std::printf("       (workable %zu plots, %.3f s idle; exact %zu plots, "
                    "%.3f s idle)\n",
                    opt.workable.plots, opt.workable.spread_seconds,
                    opt.exact.plots, opt.exact.spread_seconds);
    }

    // Matched workers need no balancing act: one plot each already lands them
    // together, so the smallest answer is the worker count and it is exact.
    {
        auto const opt = pos2gpu::pick_batch_sizing({9.0, 9.0, 9.0}, {0, 1, 2});
        all_ok = check(opt.valid && opt.workable.plots == 3,
                       "sizing: 3 matched GPUs -> 3 plots") && all_ok;
        all_ok = check(near(opt.workable.spread_seconds, 0.0),
                       "sizing: matched GPUs land together exactly") && all_ok;
        all_ok = check(opt.coincide,
                       "sizing: matched GPUs -> workable == exact, print once")
                 && all_ok;
    }

    // A worker with no rate yet cannot be sized around. It must be excluded and
    // COUNTED -- pricing it at a peer's rate is the same mistake EtaEstimate
    // refuses to make, and here it would silently recommend a batch size that
    // starves or stalls on it.
    {
        auto const opt = pos2gpu::pick_batch_sizing({8.79, 0.0, 63.0},
                                                    {0, 1, pos2gpu::kCpuDeviceId});
        all_ok = check(opt.workers_modelled == 2 && opt.workers_unrated == 1,
                       "sizing: unrated worker excluded and counted") && all_ok;
        all_ok = check(opt.valid && opt.workable.plots == 8,
                       "sizing: unrated worker does not perturb the answer")
                 && all_ok;
        all_ok = check(opt.workable.split[1].device_id == pos2gpu::kCpuDeviceId,
                       "sizing: split carries device ids past the dropped worker")
                 && all_ok;
        // The dropped worker sat at index 1, so the CPU is split[1] but worker 2.
        // Naming it from its position in `split` would call it "gpu1".
        all_ok = check(opt.workable.split[0].worker_index == 0
                           && opt.workable.split[1].worker_index == 2,
                       "sizing: worker_index points past the dropped worker")
                 && all_ok;
    }

    // worker_index must survive a rig whose unrated worker is a REPEAT of a
    // rated one -- the case where naming from `split` alone silently renames
    // every later worker. Two CPUs where the first has no rate yet: the rated
    // one is worker 2 ("cpu#1" to worker_labels), not worker 1 ("cpu#0").
    {
        auto const opt = pos2gpu::pick_batch_sizing(
            {8.79, 0.0, 63.0},
            {0, pos2gpu::kCpuDeviceId, pos2gpu::kCpuDeviceId});
        all_ok = check(opt.valid && opt.workers_modelled == 2,
                       "sizing: repeat-device rig modelled") && all_ok;
        all_ok = check(opt.workable.split[1].worker_index == 2,
                       "sizing: rated cpu keeps its ordinal past an unrated twin")
                 && all_ok;
    }

    // Nothing to balance with fewer than two rated workers.
    {
        all_ok = check(!pos2gpu::pick_batch_sizing({8.79}, {0}).valid,
                       "sizing: lone worker -> no answer") && all_ok;
        all_ok = check(!pos2gpu::pick_batch_sizing({}, {}).valid,
                       "sizing: no workers -> no answer") && all_ok;
        all_ok = check(!pos2gpu::pick_batch_sizing({0.0, 0.0}, {0, 1}).valid,
                       "sizing: no rates -> no answer") && all_ok;
    }

    // ---------------------------------------------------------------------
    // The shipped search walks the queue ONCE to the cap and reads each batch
    // size off as a prefix of that walk, which is only legal because greedy
    // assignment is incremental: where job j goes cannot depend on how many
    // jobs follow it. Assert that against the obvious-but-quadratic version --
    // simulate each size from an empty queue -- exactly as the ETA test above
    // licenses its binary search against a brute-force handout.
    // ---------------------------------------------------------------------
    {
        auto from_scratch = [](std::vector<double> const& s, std::size_t n) {
            std::vector<double> free_at(s.size(), 0.0);
            for (std::size_t j = 0; j < n; ++j) {
                std::size_t pick = 0;
                for (std::size_t i = 1; i < s.size(); ++i) {
                    if (free_at[i] < free_at[pick]) pick = i;
                }
                free_at[pick] += s[pick];
            }
            return free_at;
        };

        bool   agree = true;
        double worst = 0.0;
        std::size_t cases = 0;
        // Rates spanning the real spread: matched cards, mild asymmetry
        // (4090 + 3060), and the ~7x GPU-vs-CPU gap that motivates the feature.
        for (double a : {6.6, 8.79, 9.0, 14.0}) {
            for (double b : {6.6, 9.0, 21.5, 63.0, 120.0}) {
                for (double c : {0.0, 10.0, 63.7}) {
                    std::vector<double> rates = {a, b};
                    std::vector<int>    ids   = {0, 1};
                    if (c > 0.0) { rates.push_back(c); ids.push_back(2); }

                    auto const opt = pos2gpu::pick_batch_sizing(rates, ids);
                    if (!opt.valid) continue;
                    for (auto const* b2 : {&opt.workable, &opt.exact}) {
                        if (!b2->valid) continue;
                        ++cases;
                        auto const ref = from_scratch(rates, b2->plots);
                        for (std::size_t i = 0; i < b2->split.size(); ++i) {
                            double const d =
                                std::fabs(b2->split[i].finish_seconds - ref[i]);
                            worst = std::max(worst, d);
                            if (d > 1e-9) agree = false;
                        }
                    }
                }
            }
        }
        std::printf("       (worst |prefix-walk - from-scratch| = %.2e s over "
                    "%zu sizings)\n", worst, cases);
        all_ok = check(agree,
                       "sizing: one-pass prefix == per-size simulation from empty")
                 && all_ok;
    }

    return all_ok ? 0 : 1;
}
