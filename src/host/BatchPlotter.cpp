// BatchPlotter.cu — implementation of staggered multi-plot pipeline.

#include "host/BatchPlotter.hpp"
#include "host/Cancel.hpp"
#include "host/CpuPlotter.hpp"  // run_one_plot_cpu — pos2-chip CPU pipeline
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#include "host/HostPinnedPool.hpp"
#include "host/HostRamPolicy.hpp"  // plan_host_ram_spill — the spill budget policy
#include "host/MultiGpuPlotPipeline.hpp"        // --shard-plot path (Phase 2.2+)
#include "host/MultiGpuPipelineParallel.hpp"   // --pipeline-plot path (Phase 2.1d)
#include "host/MultiGpuShardBufferPool.hpp"  // batch-amortised buffer reuse
#include "host/PlotFileWriterParallel.hpp"
#include "host/TempFile.hpp"  // resolve_dir / dir_is_ram_backed — spill temp-dir guard
#include "gpu/DeviceIds.hpp"  // kCpuDeviceId for the --cpu device-list mixin
#include "host/NumaTopology.hpp"  // CPU-node enumeration + per-worker pinning
#include "gpu/SyclBackend.hpp"  // sycl_backend::queue, set_current_device_id

// Deliberately no pos2-chip includes here — see PlotFileWriterParallel.cpp.

#ifdef __linux__
#include <sys/resource.h>  // setpriority / PRIO_PROCESS — see nice_current_thread
#include <cerrno>
#include <cstring>         // std::strerror
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#include <unistd.h>  // isatty — in-place progress line only on a TTY

namespace pos2gpu {

void initialize_aes_tables(); // forward decl from AesGpu.cu

namespace {

bool parse_hex(std::string const& s, std::vector<uint8_t>& out)
{
    if (s.size() % 2) return false;
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    out.clear();
    out.reserve(s.size() / 2);
    for (size_t i = 0; i < s.size(); i += 2) {
        int hi = val(s[i]), lo = val(s[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back(uint8_t((hi << 4) | lo));
    }
    return true;
}

bool parse_hex_array32(std::string const& s, std::array<uint8_t, 32>& out)
{
    std::vector<uint8_t> tmp;
    if (!parse_hex(s, tmp) || tmp.size() != 32) return false;
    std::copy(tmp.begin(), tmp.end(), out.begin());
    return true;
}

} // namespace

std::vector<BatchEntry> parse_manifest(std::string const& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open manifest: " + path);

    std::vector<BatchEntry> out;
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        BatchEntry e;
        std::string testnet_s, plot_id_s, memo_s;
        if (!(is >> e.k >> e.strength >> e.plot_index >> e.meta_group
                 >> testnet_s >> plot_id_s >> memo_s >> e.out_dir >> e.out_name)) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) +
                                     ": expected 9 whitespace-separated fields "
                                     "(k strength plot_index meta_group testnet "
                                     "plot_id_hex memo_hex out_dir out_name)");
        }
        e.testnet = (testnet_s == "1" || testnet_s == "true" || testnet_s == "True");
        if (!parse_hex_array32(plot_id_s, e.plot_id)) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) +
                                     ": plot_id must be 64 hex chars");
        }
        if (!parse_hex(memo_s, e.memo) || e.memo.size() > 255) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) +
                                     ": memo invalid hex or > 255 bytes");
        }
        out.push_back(std::move(e));
    }
    return out;
}

namespace {

struct WorkItem {
    BatchEntry        entry;
    GpuPipelineResult result;
    size_t            index = 0;
};

// Rough per-plot upper-bound estimate for the disk preflight. The actual
// compressed .plot2 is smaller (FSE over proof-fragment stubs); this
// uncompressed ceiling is deliberately pessimistic so we only WARN when
// the disk is genuinely too small, not for boundary cases.
//
// Formula: 2^k fragments × (proof_fragment_bits) / 8, where
// proof_fragment_bits ≈ k + (k - MINUS_STUB_BITS) + overhead, ≈ 2k bytes*bits.
uint64_t approx_plot_bytes_upper_bound(int k)
{
    if (k <= 0 || k > 32) return 0;
    uint64_t const fragments = uint64_t(1) << k;
    uint64_t const bits_per  = uint64_t(2 * k);  // k stub + k-2 xbits, rounded up
    return (fragments * bits_per) / 8;
}

// Check `.plot2` is present at path AND looks like a valid plot file
// (magic bytes "pos2" + nonzero size). Used for --skip-existing so we
// don't silently skip a zero-byte or crash-truncated leftover.
bool looks_like_complete_plot(std::filesystem::path const& path)
{
    std::error_code ec;
    auto const sz = std::filesystem::file_size(path, ec);
    if (ec || sz < 64) return false;  // header alone is >64 B

    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[4]{};
    in.read(magic, 4);
    return in.good() && magic[0] == 'p' && magic[1] == 'o'
                     && magic[2] == 's' && magic[3] == '2';
}

// Print a warning if the available free space on each unique output
// directory looks insufficient for the plots targeted there. Purely
// advisory — the atomic .partial write handles actual ENOSPC cleanly.
void preflight_disk_space(std::vector<BatchEntry> const& entries,
                          BatchOptions const& opts)
{
    if (entries.empty()) return;

    std::map<std::string, std::pair<size_t, uint64_t>> per_dir;  // dir -> (count, bytes)
    for (auto const& e : entries) {
        uint64_t const est = approx_plot_bytes_upper_bound(e.k);
        auto& slot = per_dir[e.out_dir.empty() ? std::string(".") : e.out_dir];
        slot.first  += 1;
        slot.second += est;
    }

    constexpr double GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
    for (auto const& [dir, tally] : per_dir) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);  // space() needs it to exist
        auto const info = std::filesystem::space(dir, ec);
        if (ec) {
            if (opts.verbose) {
                std::fprintf(stderr,
                    "[batch] preflight: cannot stat free space on %s (%s) — "
                    "skipping check\n", dir.c_str(), ec.message().c_str());
            }
            continue;
        }
        double const need_gb = tally.second * GB;
        double const free_gb = info.available * GB;
        if (info.available < tally.second) {
            std::fprintf(stderr,
                "[batch] WARNING: %s has %.1f GB free but %zu plot(s) may need "
                "up to ~%.1f GB (uncompressed upper bound). The batch will "
                "still run; .partial writes are atomic so mid-plot ENOSPC is "
                "recoverable, but consider freeing space or reducing count.\n",
                dir.c_str(), free_gb, tally.first, need_gb);
        } else if (opts.verbose) {
            std::fprintf(stderr,
                "[batch] preflight: %s has %.1f GB free, %zu plot(s) need "
                "up to ~%.1f GB\n",
                dir.c_str(), free_gb, tally.first, need_gb);
        }
    }
}

constexpr double kTibBytes = 1024.0 * 1024.0 * 1024.0 * 1024.0;

// Live, lock-free progress for the ETA. A work-queue's remaining plots are held
// by SPECIFIC workers running at their OWN rates, so no batch-wide mean can
// price the drain — the estimator needs each worker's rate and its own in-flight
// count. See estimate_eta_seconds() in BenchStats.hpp for the model.
//
// Every strategy publishes here, single-worker ones into a one-slot table, so
// emit_progress_line has exactly one shape to reason about.
// This process's resident set. Pairs with the free-memory probe — see
// CpuMemoryGate, which needs the SUM of the two.
std::uint64_t self_rss_bytes()
{
#if defined(__linux__)
    std::FILE* fp = std::fopen("/proc/self/statm", "re");
    if (!fp) return 0;
    unsigned long long total_pages = 0;
    unsigned long long rss_pages   = 0;
    int const got = std::fscanf(fp, "%llu %llu", &total_pages, &rss_pages);
    std::fclose(fp);
    if (got != 2) return 0;
    long const page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0) return 0;
    return static_cast<std::uint64_t>(rss_pages) *
           static_cast<std::uint64_t>(page);
#else
    return 0;  // gate degrades to the start-of-batch answer — see budget_now()
#endif
}

std::uint64_t host_free_bytes_now()
{
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    if (!device_memory_probe(kCpuDeviceId, free_b, total_b)) return 0;
    return static_cast<std::uint64_t>(free_b);
}

// ---------------------------------------------------------------------------
// Admission control for CPU workers.
//
// The start-of-batch gate answers "how many workers can this host fund?" exactly
// once, before any of them exist. That is the right question at t=0 and the wrong
// one an hour in: a batch runs for hours, and the box it runs on is usually
// somebody's actual computer. If a compile, a browser, or a second plotter takes
// 40 GB while we are running, the count decided at t=0 is now a lie — and the OOM
// killer is what notices, taking the GPU workers' in-flight plots down with the
// CPU's.
//
// So ask again at every plot boundary. That is the one place it can be asked
// honestly: between plots a CPU worker holds NOTHING (pos2-chip's Plotter is
// constructed per plot, inside run_one_plot_cpu), so a worker about to start one
// is asking "is there room for 12.1 GiB?" while holding none of it.
//
// What does NOT work is to simply probe MemAvailable and compare:
//
//   * The herd. N workers finishing at the same moment all probe, all see the
//     same free memory, and all conclude there is room for one more — because
//     none of them has allocated yet. A probe can see an allocation; it cannot
//     see a decision.
//
//   * The self-measurement. Our own workers' 12 GiB apiece IS most of what
//     MemAvailable is missing, so a controller that re-reads it mid-run reads its
//     own footprint as external pressure and shrinks the pool it just created.
//
// Both vanish once the two quantities are measured separately:
//
//   OUR usage      a reservation counter, incremented on ADMISSION. Exact, and it
//                  counts decisions rather than allocations, so no herd can form.
//
//   EXTERNAL usage (MemAvailable + our own RSS) is invariant under our own
//                  allocations: touching a page drops MemAvailable and raises
//                  VmRSS by the same amount, so the sum does not move. The drop in
//                  that SUM since batch start is therefore what the REST of the
//                  box has taken — blind to what we are doing, which is exactly
//                  what a controller of our own size needs it to be.
//
// (The GPU workers' host memory is handled by the same invariance: their pinned
// pools raise our RSS and lower MemAvailable together, so they never look like
// external pressure — and their share was already subtracted from the budget.)
class CpuMemoryGate {
public:
    enum class Verdict { Admitted, Denied };

    CpuMemoryGate(std::uint64_t per_worker, std::uint64_t budget_at_start)
        : per_worker_(per_worker)
        , budget_at_start_(budget_at_start)
        // Both halves probed at the same instant, or their sum means nothing.
        , baseline_sum_(static_cast<std::int64_t>(host_free_bytes_now()) +
                        static_cast<std::int64_t>(self_rss_bytes()))
    {}

    // Waits until this worker's next plot can be funded.
    //
    // `still_wanted` is what keeps a waiter from outliving the work: a worker
    // blocked in here cannot see the queue drain or a cancel arrive, so on a
    // GPU+CPU rig the GPUs would finish the batch and then hang forever joining a
    // CPU worker that is still waiting for memory to make a plot nobody needs.
    //
    // Two ways to be woken, and they are not the same:
    //
    //   a peer finishing   — notifies, and is GUARANTEED to free per_worker bytes.
    //   the box relenting  — sends no notification at all. So poll for it, on a
    //                        1 s tick. This is the case the whole gate exists for,
    //                        and an earlier version of this function got it badly
    //                        wrong: it treated "no peer of ours holds memory" as
    //                        "nothing can ever free memory" and killed the batch
    //                        on the spot. The rest of the box is not ours to
    //                        account for — it takes memory without asking and
    //                        gives it back without telling us.
    //
    // Give up only after the box has been too small for a plot CONTINUOUSLY for
    // the grace period (XCHPLOT2_CPU_WAIT_SECS, default 5 min) with nobody of ours
    // holding anything — that is a box that has genuinely shrunk, not a compile
    // that will end.
    Verdict acquire(char const* who, bool quiet,
                    std::function<bool()> const& still_wanted)
    {
        using clock = std::chrono::steady_clock;
        std::unique_lock<std::mutex> lk(m_);
        bool                          announced = false;
        std::optional<clock::time_point> starved_since;

        for (;;) {
            std::int64_t const budget = budget_now_locked();
            if (static_cast<std::int64_t>(committed_ + per_worker_) <= budget) {
                committed_ += per_worker_;
                ++holders_;
                if (announced && !quiet) {
                    std::fprintf(stderr, "%s resumed — memory came back\n", who);
                }
                return Verdict::Admitted;
            }

            // Nothing left to plot (or we are being cancelled)? Then stop waiting
            // for the memory to plot it with.
            if (still_wanted && !still_wanted()) return Verdict::Denied;

            if (!announced && !quiet) {
                announced = true;
                std::int64_t const spare =
                    budget - static_cast<std::int64_t>(committed_);
                std::fprintf(stderr,
                    "%s waiting for memory: its next plot needs %.1f GiB, the host "
                    "has %.1f GiB to spare",
                    who, gib(per_worker_),
                    spare > 0 ? gib(static_cast<std::uint64_t>(spare)) : 0.0);
                if (holders_ > 0) {
                    std::fprintf(stderr,
                        " (%d peer%s plotting — each one finishing frees %.1f GiB)\n",
                        holders_, holders_ == 1 ? "" : "s", gib(per_worker_));
                } else {
                    std::fprintf(stderr,
                        " and no peer of ours is holding any. Waiting up to %lld s "
                        "for the rest of the box to give it back.\n",
                        static_cast<long long>(grace_.count()));
                }
            }

            auto const now = clock::now();
            if (holders_ == 0) {
                if (!starved_since) {
                    starved_since = now;
                } else if (now - *starved_since > grace_) {
                    return Verdict::Denied;   // the box really has shrunk
                }
            } else {
                starved_since.reset();  // a peer is running; it WILL free memory
            }

            cv_.wait_for(lk, std::chrono::seconds(1));
        }
    }

    void release()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            committed_ = committed_ > per_worker_ ? committed_ - per_worker_ : 0;
            if (holders_ > 0) --holders_;
        }
        cv_.notify_all();  // a waiter can now re-ask
    }

    std::uint64_t per_worker() const { return per_worker_; }

    // Only meaningful for the error message on a Denied verdict.
    double budget_gib_now()
    {
        std::lock_guard<std::mutex> lk(m_);
        std::int64_t const b = budget_now_locked();
        return b > 0 ? gib(static_cast<std::uint64_t>(b)) : 0.0;
    }

private:
    static double gib(std::uint64_t b)
    {
        return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0);
    }

    std::int64_t budget_now_locked() const
    {
        std::uint64_t const free_now = host_free_bytes_now();
        std::uint64_t const rss_now  = self_rss_bytes();
        if (free_now == 0 || rss_now == 0) {
            // No probe (not Linux, or /proc unreadable). Fall back to the answer
            // the start-of-batch gate already gave — never to "unlimited".
            return static_cast<std::int64_t>(budget_at_start_);
        }
        std::int64_t const sum_now = static_cast<std::int64_t>(free_now) +
                                     static_cast<std::int64_t>(rss_now);
        std::int64_t const external = baseline_sum_ - sum_now;  // >0: box got busier
        return static_cast<std::int64_t>(budget_at_start_) - external;
    }

    // How long the box may stay too small for a single plot, with nobody of ours
    // holding memory, before we accept that it is not going to relent.
    static std::chrono::seconds grace_seconds()
    {
        if (char const* v = std::getenv("XCHPLOT2_CPU_WAIT_SECS"); v && v[0]) {
            long const s = std::atol(v);
            if (s >= 0) return std::chrono::seconds(s);
        }
        return std::chrono::seconds(300);
    }

    std::uint64_t const        per_worker_;
    std::uint64_t const        budget_at_start_;
    std::int64_t const         baseline_sum_;
    std::chrono::seconds const grace_ = grace_seconds();
    std::uint64_t              committed_ = 0;  // bytes promised to admitted workers
    int                        holders_   = 0;  // CPU workers currently plotting
    mutable std::mutex         m_;
    std::condition_variable    cv_;
};

// Releases on every path out of the loop body — including the throw.
class CpuMemoryLease {
public:
    CpuMemoryLease() = default;
    explicit CpuMemoryLease(CpuMemoryGate* g) : gate_(g) {}
    CpuMemoryLease(CpuMemoryLease&& o) noexcept : gate_(o.gate_) { o.gate_ = nullptr; }
    CpuMemoryLease& operator=(CpuMemoryLease&& o) noexcept
    {
        if (this != &o) { reset(); gate_ = o.gate_; o.gate_ = nullptr; }
        return *this;
    }
    CpuMemoryLease(CpuMemoryLease const&)            = delete;
    CpuMemoryLease& operator=(CpuMemoryLease const&) = delete;
    ~CpuMemoryLease() { reset(); }

    void reset()
    {
        if (gate_) { gate_->release(); gate_ = nullptr; }
    }

private:
    CpuMemoryGate* gate_ = nullptr;
};

// Enabled by XCHPLOT2_CPU_ADAPTIVE=1: beside a GPU, spawn the RAM-capped knee of
// CPU workers and let CpuRateGovernor throttle how many actually RUN by the GPU's
// measured plot rate, instead of hardcoding 1. Opt-in while it proves out; the
// default path is unchanged, so it cannot regress the shipped behaviour.
inline bool cpu_adaptive_enabled()
{
    char const* v = std::getenv("XCHPLOT2_CPU_ADAPTIVE");
    return v && v[0] == '1';
}

// The tail guard (default ON; XCHPLOT2_TAIL_GUARD=0 disables). A worker about to
// pull the next job retires instead when the workers strictly faster than it can
// drain everything left before it would finish one more plot — so a slow worker
// (a CPU beside a GPU, or a slow card beside a fast one) never grabs a tail job a
// faster peer would clear sooner, and never starts at all on a batch too short to
// help. Worker-general, and one-directional: it can only ever DECLINE work a
// worker would finish last on, so it cannot lengthen a run, and a uniform fleet
// (nobody strictly faster) is left untouched. Unlike XCHPLOT2_CPU_ADAPTIVE, which
// governs how MANY CPU workers run beside a GPU, this governs whether ANY worker
// should take the next job as the queue drains — a different question, so a
// different switch.
inline bool tail_guard_enabled()
{
    char const* v = std::getenv("XCHPLOT2_TAIL_GUARD");
    return !(v && v[0] == '0');
}

// XCHPLOT2_DRAIN_SLOTS=N pins the rotating pinned D2H drain to N slots,
// overriding both the default (kNumPinnedBuffers) and the budget policy's
// demand-driven reduction. Returns 0 when unset or out of [1,
// kNumPinnedBuffers]. It exists so the RAM-vs-overlap trade can be
// A/B-measured, and so someone under a budget on a host that can afford the
// slots can buy the overlap back. See the B1 note in run_batch_slice.
inline int drain_slots_env()
{
    char const* v = std::getenv("XCHPLOT2_DRAIN_SLOTS");
    if (!v || !v[0]) return 0;
    int const parsed = std::atoi(v);
    if (parsed < 1 || parsed > GpuBufferPool::kNumPinnedBuffers) return 0;
    return parsed;
}

// Seconds/plot the tail guard assumes for a worker that has not measured its own
// rate yet, seeded by device class so a known-slow CPU worker can stand down on a
// batch too short to help before it has finished even one plot. Replaced by the
// worker's measured rate after two completions. The CPU prior is overridable (for
// other CPUs / other k); the GPU prior only has to sit well below it, so any card
// reads as "faster" than the CPU pre-measurement.
inline double cpu_plot_seconds_prior()
{
    if (char const* v = std::getenv("XCHPLOT2_CPU_PLOT_SECONDS"); v && v[0]) {
        double const d = std::atof(v);
        if (d > 0.0) return d;
    }
    return 45.0;  // measured 40-50 s on a desktop core group at k=28
}
inline constexpr double kGpuPlotSecondsPrior = 3.0;

// Defined below, next to cpu_worker_auto_count; CpuRateGovernor needs it here.
int cpu_target_for_gpu_rate(double s_gpu_seconds);

struct LiveWorker {
    std::atomic<std::size_t> written{0};
    std::atomic<std::size_t> in_flight{0};  // pulled off the queue, not retired
    std::atomic<double>      work_start{0.0};
    std::atomic<double>      first_done{0.0};
    std::atomic<double>      last_done{0.0};

    // What to call this worker in a log line — "gpu0", "cpu#1". Written by
    // run_batch before any worker thread exists, read-only thereafter (the
    // vector is sized once and never resized, so no reallocation can race a
    // reader). Empty on the single-worker fast path, which keeps its historical
    // device-derived prefix.
    std::string label;
};

// A worker's measured seconds-per-plot from its own completions, anchored on its
// first (cold) plot the way the ETA and the governor do — 0 until it has two, so
// a single cold plot cannot fake a rate. The tail guard falls back to the seeded
// prior while this is 0.
inline double worker_measured_s_per_plot(LiveWorker const& w)
{
    std::size_t const n = w.written.load(std::memory_order_relaxed);
    if (n < 2) return 0.0;
    double const first = w.first_done.load(std::memory_order_relaxed);
    double const last  = w.last_done.load(std::memory_order_relaxed);
    if (last <= first) return 0.0;
    return (last - first) / static_cast<double>(n - 1);
}

// Adaptive admission for CPU workers, complementing CpuMemoryGate.
//
// CpuMemoryGate answers "does the host have RAM for one more plot?" — a question
// about the box. This answers "does the GPU's measured speed justify one more CPU
// worker?" — a question about the peer. cpu_worker_auto_count picks 1 beside a GPU
// because at t=0 it cannot know the GPU's speed; here, a few plots in, we can.
//
// The pattern mirrors the memory gate: spawn the RAM-capped knee of CPU workers,
// then keep only cpu_target_for_gpu_rate(measured) of them ACTIVE. A surplus
// worker parks in wait_active() holding no RAM and (having never claimed a plot)
// no cores, ready to step in if the GPU turns out slow. Two deliberate choices
// keep the bimodal rate noise (the very thing that defeated the crossover sweep)
// from thrashing workers:
//
//   * MEASURE FIRST. Until the GPUs have a few WARM plots (rate anchored on each
//     one's first completion, cold plot excluded — as the ETA does), hold at the
//     floor of 1. A cold first plot must not fake a slow reading and over-spawn on
//     a fast card.
//   * MONOTONIC UP. The active count only ever rises — we raise it once we learn
//     the GPU is slow, and never kill an already-running worker on a later wobble.
//
// Bound to BatchProgress::workers (constructed first). No notification path: a
// parked worker re-checks on a 1 s tick, which is immaterial against ~seconds per
// plot and needs no producer to remember to wake it.
class CpuRateGovernor {
public:
    // gpu_slots / cpu_ordinal index into the same `workers` vector: gpu_slots are
    // the GPU workers whose rate we read; cpu_ordinal[i] is worker i's 0-based
    // rank among CPU workers, or -1 if it is not one.
    CpuRateGovernor(std::vector<LiveWorker> const& workers,
                    std::vector<std::size_t>       gpu_slots,
                    std::vector<int>               cpu_ordinal)
        : workers_(workers)
        , gpu_slots_(std::move(gpu_slots))
        , cpu_ordinal_(std::move(cpu_ordinal))
    {}

    // Blocks until worker `worker_id`'s CPU ordinal is within the active count the
    // GPU's rate currently allows. Returns false (→ the worker retires) when the
    // work is gone, so a parked worker can never outlive the batch and hang join.
    bool wait_active(int worker_id, std::function<bool()> const& still_wanted)
    {
        int const ord = (worker_id >= 0 &&
                         worker_id < static_cast<int>(cpu_ordinal_.size()))
                            ? cpu_ordinal_[static_cast<std::size_t>(worker_id)]
                            : -1;
        if (ord < 0) return true;  // not a CPU worker — nothing to govern

        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            if (still_wanted && !still_wanted()) return false;
            if (ord < target_locked()) return true;
            cv_.wait_for(lk, std::chrono::seconds(1));  // re-check as rate settles
        }
    }

private:
    // How many CPU workers the GPUs' current rate justifies. Only ever grows.
    int target_locked()
    {
        std::size_t gpu_done = 0;
        double      sum_s    = 0.0;
        int         rated    = 0;
        for (std::size_t slot : gpu_slots_) {
            LiveWorker const& w = workers_[slot];
            std::size_t const n = w.written.load(std::memory_order_relaxed);
            gpu_done += n;
            if (n >= 2) {
                double const first = w.first_done.load(std::memory_order_relaxed);
                double const last  = w.last_done.load(std::memory_order_relaxed);
                if (last > first) {
                    sum_s += (last - first) / static_cast<double>(n - 1);
                    ++rated;
                }
            }
        }
        // Still measuring: hold at whatever we have committed (the floor at start).
        if (gpu_done < kMeasurePlots || rated == 0) return committed_target_;

        int const want = cpu_target_for_gpu_rate(sum_s / rated);  // mean per-GPU
        if (want > committed_target_) committed_target_ = want;   // monotonic up
        return committed_target_;
    }

    // Warm GPU plots (across all GPUs) required before the rate is trusted. The
    // first is cold and excluded from the rate, so this leaves ~kMeasurePlots-1
    // warm intervals to average — enough to shrug off a single slow plot.
    static constexpr std::size_t kMeasurePlots = 6;

    std::vector<LiveWorker> const& workers_;
    std::vector<std::size_t> const gpu_slots_;
    std::vector<int> const         cpu_ordinal_;
    std::mutex                     m_;
    std::condition_variable        cv_;
    int                            committed_target_ = 1;  // opt-in floor
};

struct BatchProgress {
    std::vector<LiveWorker>    workers;
    // Retired = written + skipped. It doubles as the ticket counter for the
    // display: each emitting thread prints the value its own fetch_add returned,
    // so two threads can never race to the same total and print it twice. (That
    // is what summing the per-worker counters at print time would allow.)
    //
    // It does NOT by itself stop the line ticking backwards — minting ordered
    // tickets says nothing about the order threads then reach the terminal in.
    // painted_ticket below is what enforces that.
    std::atomic<std::size_t>   retired{0};
    std::atomic<std::size_t>   written{0};
    std::atomic<std::size_t>   skipped{0};
    std::atomic<std::uint64_t> bytes{0};
    std::size_t                total = 0;

    // Lines the multi-worker progress block last painted, so the next paint
    // knows how far back up to move. Per-batch rather than a file-scope static
    // because a bench run calls run_batch twice in one process (e2e, then
    // --compute-only) and the second pass must not try to scroll back over the
    // first's block. Zero = nothing painted yet, so don't move at all.
    //
    // mutable because emitting progress does not change the BATCH — every other
    // field here is read-only to the emitter, and widening its reference to
    // non-const to carry a cursor position would lose that guarantee.
    mutable std::atomic<int>   painted_lines{0};

    // Highest ticket already on screen. Tickets are unique and ordered, but the
    // fetch_add that mints them does NOT order the printing that follows: the
    // thread holding N can reach the terminal before the thread holding N-1.
    // The comment above once claimed the ticket alone stopped the line ticking
    // backwards; it stops two threads printing the SAME total, which is not the
    // same thing. Whoever loses that race must not paint — with one line it
    // showed a stale count, and with the multi-line block it skips the cursor-up
    // (its `painted_lines` was already consumed) and leaves a second, stale
    // block below the final one.
    mutable std::atomic<std::size_t> painted_ticket{0};

    // Is any worker in this batch a GPU?
    //
    // The CPU worker nices itself down so it stops starving the GPU workers,
    // and that is a one-way door for an unprivileged process — so it must fire
    // only when there is somebody to yield TO. `workers.size() > 1` used to be
    // exactly that test, on the reasoning that there was at most one CPU worker
    // and therefore any peer was a GPU. --cpu-workers makes that false: two CPU
    // workers and no GPU is now a legal batch, and nicing all of them equally
    // yields to nobody while permanently out-ranking them under the writer pool.
    bool gpu_peer_present = false;

    // Admission control for the CPU workers, re-asked at every plot boundary.
    // Null when the batch has no CPU worker. See CpuMemoryGate.
    std::unique_ptr<CpuMemoryGate> cpu_gate;

    // Adaptive CPU-worker throttle, keyed on the GPU's measured rate. Null unless
    // XCHPLOT2_CPU_ADAPTIVE=1 on a GPU+CPU batch. See CpuRateGovernor.
    std::unique_ptr<CpuRateGovernor> cpu_rate_gov;

    // Per-worker seconds/plot prior for the tail guard, used until each worker has
    // measured its own rate. Seeded by device class in run_batch (CPU slow, GPU
    // fast); 0 in an unseeded slot, which makes the guard fall back to measured-
    // only for it. Sized with the worker count so the guard can index it freely.
    std::vector<double> worker_prior_s;

    BatchProgress(std::size_t worker_count, std::size_t total_entries)
        : workers(worker_count), total(total_entries),
          worker_prior_s(worker_count, 0.0) {}
};

// Worker-general tail guard — see tail_guard_enabled(). Returns true (→ this
// worker should retire from the queue rather than pull) when the workers strictly
// faster than it can drain the `remaining` unclaimed plots before it would finish
// one more of its own. `why`, if given, gets a one-line reason for the log.
//
// The rule: faster workers supply rate_faster plots/sec, so they clear `remaining`
// in remaining / rate_faster s; retire once that is <= this worker's own per-plot
// time, i.e. remaining <= t_self * rate_faster. Nobody strictly faster — a uniform
// fleet, or the fastest worker itself — gives rate_faster 0 and never retires. The
// 10% margin keeps near-equal peers from counting each other, so rate noise cannot
// churn a balanced fleet. Rates are read lock-free: a torn read only misprices one
// iteration of a heuristic re-asked at every pull.
bool tail_guard_should_retire(BatchProgress const& live, int worker_id,
                              std::size_t remaining, std::string* why)
{
    if (remaining == 0) return false;  // queue already dry — the pull loop breaks
    if (worker_id < 0 ||
        static_cast<std::size_t>(worker_id) >= live.workers.size()) return false;
    if (live.worker_prior_s.size() != live.workers.size()) return false;
    std::size_t const self = static_cast<std::size_t>(worker_id);

    auto s_per_plot = [&](std::size_t i) -> double {
        double const m = worker_measured_s_per_plot(live.workers[i]);
        return m > 0.0 ? m : live.worker_prior_s[i];  // measured, else seeded prior
    };
    double const t_self = s_per_plot(self);
    if (t_self <= 0.0) return false;  // no estimate for this worker — do not guard

    double rate_faster = 0.0;  // plots/sec of workers meaningfully faster than self
    int    n_faster    = 0;
    std::string faster_label;  // the sole faster worker's label, when there is one
    for (std::size_t i = 0; i < live.workers.size(); ++i) {
        if (i == self) continue;
        double const t_i = s_per_plot(i);
        if (t_i > 0.0 && t_i < 0.9 * t_self) {
            rate_faster += 1.0 / t_i;
            if (++n_faster == 1) faster_label = live.workers[i].label;
        }
    }
    if (rate_faster <= 0.0) return false;  // nobody faster can cover the tail

    if (static_cast<double>(remaining) > t_self * rate_faster) return false;

    if (why) {
        // Name the faster worker when there is one; agree the verb with the count.
        std::string who;
        char const* verb;
        if (n_faster == 1) {
            who  = faster_label.empty() ? "the faster worker" : faster_label;
            verb = "clears";
        } else {
            who  = "the " + std::to_string(n_faster) + " faster workers";
            verb = "clear";
        }
        char buf[224];
        std::snprintf(buf, sizeof(buf),
            "standing down — %s %s the remaining %zu plot%s faster than one more "
            "would here (~%.0f s)",
            who.c_str(), verb, remaining, remaining == 1 ? "" : "s", t_self);
        *why = buf;
    }
    return true;
}

std::string format_duration_hms(double seconds)
{
    if (seconds < 0.0) seconds = 0.0;
    int const total_s = static_cast<int>(seconds);
    int const h = total_s / 3600;
    int const m = (total_s % 3600) / 60;
    int const s = total_s % 60;
    char buf[32];
    if (h > 0) {
        std::snprintf(buf, sizeof(buf), "%dh%02dm%02ds", h, m, s);
    } else if (m > 0) {
        std::snprintf(buf, sizeof(buf), "%dm%02ds", m, s);
    } else {
        std::snprintf(buf, sizeof(buf), "%ds", s);
    }
    return buf;
}

// Width of the per-worker share bar in the multi-worker progress block. Small
// on purpose: it shares an 80-column line with the label, count, and rate.
inline constexpr std::size_t kProgressBarCells = 10;

// done_now is the emitting thread's own `retired` ticket, not a re-read of the
// counter — see BatchProgress.
void emit_progress_line(std::string const& log_prefix,
                        BatchOptions const& opts,
                        BatchProgress const& live,
                        std::size_t done_now,
                        double elapsed_s)
{
    if (!opts.progress || done_now == 0 || elapsed_s <= 0.0) return;

    // Claim the screen for this ticket, or drop out. Only strictly-increasing
    // tickets paint, so a thread overtaken between its fetch_add and here says
    // nothing rather than repainting the past. The final ticket is the batch's
    // highest by construction, so it can never be the one dropped.
    {
        std::size_t seen = live.painted_ticket.load(std::memory_order_relaxed);
        do {
            if (done_now <= seen) return;
        } while (!live.painted_ticket.compare_exchange_weak(
                     seen, done_now, std::memory_order_relaxed));
    }

    std::size_t const total   = live.total;
    std::size_t const written = live.written.load(std::memory_order_relaxed);
    std::size_t const skipped = live.skipped.load(std::memory_order_relaxed);

    // Per-worker view for the ETA. Each worker's rate is taken from its OWN
    // completions, anchored on its first — exactly as the bench's steady-state
    // does — so its cold plot stops dragging the estimate from the second plot
    // onward. With a single completion there is nothing to measure between, so
    // fall back to that plot's full cold duration: runs long, for one plot.
    std::vector<pos2gpu::WorkerLive> model;
    model.reserve(live.workers.size());
    std::size_t in_flight_total = 0;
    for (auto const& w : live.workers) {
        std::size_t const n  = w.written.load(std::memory_order_relaxed);
        std::size_t const nf = w.in_flight.load(std::memory_order_relaxed);
        in_flight_total += nf;

        pos2gpu::WorkerLive m;
        m.in_flight = nf;
        m.last_done = w.last_done.load(std::memory_order_relaxed);
        if (n >= 2) {
            double const first = w.first_done.load(std::memory_order_relaxed);
            m.s_per_plot = (m.last_done - first) / static_cast<double>(n - 1);
        } else if (n == 1) {
            m.s_per_plot =
                m.last_done - w.work_start.load(std::memory_order_relaxed);
        }
        model.push_back(m);
    }

    // Read the retired counter rather than recomputing written + skipped: a
    // FAILED entry is retired too — off the queue, never to be plotted — and
    // counting it as still-queued would have the ETA wait on it.
    std::size_t const retired = live.retired.load(std::memory_order_relaxed);
    std::size_t const unclaimed =
        total > retired + in_flight_total ? total - retired - in_flight_total : 0;

    // Entries that get skipped on arrival cost nothing, so the queue's tail is
    // not all work. Discount it by the skip rate this run has actually seen,
    // rather than promising to plot entries that are already on disk.
    double const write_frac =
        retired > 0 ? static_cast<double>(written) / static_cast<double>(retired)
                    : 1.0;
    std::size_t const queued_writes = static_cast<std::size_t>(
        std::llround(static_cast<double>(unclaimed) * write_frac));

    double const avg =
        written > 0 ? elapsed_s / static_cast<double>(written) : 0.0;
    pos2gpu::EtaEstimate const eta =
        pos2gpu::estimate_eta_seconds(model, queued_writes, elapsed_s);
    double const rate_tib_s =
        static_cast<double>(live.bytes.load(std::memory_order_relaxed))
        / elapsed_s / kTibBytes;

    // On a TTY, rewrite one line in place ("\r" + clear-to-EOL); keep
    // one-line-per-plot when redirected to a file/pipe or when verbose
    // logging would interleave and garble the in-place line.
    static bool const stderr_tty = ::isatty(::fileno(stderr)) != 0;
    bool const in_place = stderr_tty && !opts.verbose;

    // Only surfaces on a resume (--skip-existing). Without it the line counts
    // skipped entries as done in "N/M" while excluding them from every rate
    // beside it, and there is nothing on screen to explain the gap.
    char skip_note[40] = {0};
    if (skipped > 0) {
        std::snprintf(skip_note, sizeof(skip_note), "%zu skipped, ", skipped);
    }

    // "batch ETA", not "fully plotted in": this is the time left to finish the
    // plots in THIS run, whereas `bench` closes with a "fully plotted in ~3d
    // 20h" line meaning the time to fill the destination disk. The same phrase
    // for both had a 95%-done progress line reading "fully plotted in ~35s".
    //
    // ">=" when a worker is holding a plot we have no rate for: the batch cannot
    // end before that plot does and we cannot say when that is, so the number is
    // a floor. Saying "~" there is how a slow CPU's first plot gets quietly
    // written out of the estimate.
    //
    // "%.3g", not "%.6f", for TiB/s: at k=18 the true rate is 5.6e-7 TiB/s, which
    // %.6f rounds to a printed "0.000000" while the TiB/hour field beside it
    // (already %.3g) shows three real digits — the same line contradicting itself.
    char head[512];
    std::snprintf(head, sizeof(head),
        "%s progress: plot %zu/%zu done "
        "(%.1f%%, %s%.2f s/plot avg, %.3g TiB/s, batch ETA %s%s)",
        log_prefix.c_str(),
        done_now, total,
        100.0 * double(done_now) / double(total),
        skip_note,
        avg, rate_tib_s,
        eta.lower_bound ? ">=" : "~",
        format_duration_hms(eta.seconds).c_str());

    // One worker, or nowhere to rewind to (a pipe, or verbose logging that would
    // land between the block and its next repaint): the historical single line.
    // The per-worker rates are not lost in that case — print_run_summary prints
    // them once the batch ends, which is what a redirected run is read for.
    if (!in_place || live.workers.size() < 2) {
        std::fprintf(stderr, "%s%s%s",
                     in_place ? "\r\033[K" : "", head,
                     (!in_place || done_now >= total) ? "\n" : "");
        if (in_place) std::fflush(stderr);
        return;
    }

    // Multi-worker on a TTY: the aggregate header, then a line per worker.
    //
    // The header keeps the aggregate prefix and the aggregate average, and the
    // per-worker numbers live on their own labelled lines — never side by side.
    // That is the whole point of the split: a real run printed "[batch:cpu] 6.88
    // s/plot avg" for a CPU actually running at 63.7 (see progress_prefix), and
    // a per-worker label next to a batch-wide number is how that reads.
    //
    // Rates come from `model` — the same ones the ETA is priced from, so the
    // block and the ETA beside it cannot disagree.
    //
    // The whole block goes out as ONE write. N workers means N+1 lines, and the
    // cursor-up that precedes them only lands correctly if nothing interleaves
    // between it and the repaint; separate fprintf calls per line would let a
    // peer's completion land mid-block and scroll the rest into the wrong rows.
    std::size_t max_written = 1;
    for (auto const& w : live.workers) {
        max_written = std::max(max_written, w.written.load(std::memory_order_relaxed));
    }

    int const lines = 1 + static_cast<int>(live.workers.size());
    // Leave the finished block on screen: zero means the next paint (a late
    // failure, say) starts a fresh one below rather than scribbling over it.
    int const prev = live.painted_lines.exchange(
        done_now >= total ? 0 : lines, std::memory_order_relaxed);

    std::string out;
    out.reserve(static_cast<std::size_t>(lines) * 96);
    if (prev > 0) out += "\033[" + std::to_string(prev) + "A";
    out += "\r\033[K";
    out += head;
    out += "\n";

    for (std::size_t i = 0; i < live.workers.size(); ++i) {
        auto const& w = live.workers[i];
        std::size_t const n = w.written.load(std::memory_order_relaxed);

        // Share of the batch so far, against the busiest worker. This is the
        // work-queue's split made visible: on a mixed rig the bars are supposed
        // to come out lopsided, because that is the queue paying each worker by
        // what it can actually do.
        char bar[kProgressBarCells + 1];
        std::size_t const filled = max_written > 0
            ? (n * kProgressBarCells + max_written / 2) / max_written : 0;
        for (std::size_t c = 0; c < kProgressBarCells; ++c) {
            bar[c] = c < filled ? '#' : '-';
        }
        bar[kProgressBarCells] = '\0';

        char rate[24];
        if (model[i].s_per_plot > 0.0) {
            std::snprintf(rate, sizeof(rate), "%7.2f s/plot", model[i].s_per_plot);
        } else {
            // No completion yet, so no rate. The ETA refuses to invent one for
            // this worker (EtaEstimate::lower_bound); neither does the display.
            // Six spaces + the dash to occupy the same seven columns "%7.2f"
            // would — it is three BYTES, so a width field here would pad it by
            // display width and break the column.
            std::snprintf(rate, sizeof(rate), "      — s/plot");
        }

        char line[256];
        std::snprintf(line, sizeof(line), "%s   %-6s %4zu plot%s  %s  %s",
                      log_prefix.c_str(),
                      w.label.empty() ? "?" : w.label.c_str(),
                      n, n == 1 ? " " : "s", rate, bar);
        out += "\r\033[K";
        out += line;
        out += "\n";
    }

    std::fwrite(out.data(), 1, out.size(), stderr);
    std::fflush(stderr);
}

// Drop the CALLING thread's scheduling priority — and, with it, every thread
// that thread goes on to spawn.
//
// Why this exists: pos2-chip's Plotter fans out to hardware_concurrency()
// threads with no cap and no pool (RadixSort.hpp, TableConstructorGeneric.hpp),
// and our WriterThreadPool is also sized at hardware_concurrency(). A GPU+CPU
// batch therefore puts ~2x core_count runnable threads on the box, and the CPU
// plotter is the one that runs hot continuously. CFS splits the machine roughly
// evenly during every FSE burst, the GPU worker's consumer takes twice as long,
// its depth-1 channel backs up, and the GPU — which retires a plot 6-8x faster
// than the CPU does — stalls waiting on it. On the tester's 12900k/RTX3080 that
// cost 62% of the CPU worker's whole contribution: adding the CPU should have
// taken 6.7 -> 5.74 s/plot, and only reached 6.3.
//
// How it works: Linux nice is a PER-THREAD attribute (POSIX says per-process;
// NPTL disagrees), and clone() copies it into children. So nicing this one
// worker thread also nices the entire fan-out pos2-chip spawns inside
// run_one_plot_cpu(). We cannot cap that fan-out without patching pos2-chip —
// but we can make all of it yield.
//
// Why priority rather than a thread cap: the GPU workers' demand for the host is
// BURSTY (FSE fires once per plot completion, then nothing). A statically
// reserved core would sit idle between bursts. Priority is work-conserving — the
// CPU plotter still gets the whole machine whenever no GPU worker wants it, and
// steps aside the instant one does. (Affinity is not an option either: measured,
// glibc 2.43's hardware_concurrency() ignores the affinity mask, so taskset
// confines the threads without reducing how many spawn.)
//
// ONE-WAY DOOR: an unprivileged process may RAISE its nice value but not lower
// it again — that needs CAP_SYS_NICE or RLIMIT_NICE headroom. So this may only
// be called on a thread we are content to leave niced for its entire life, and
// never on a thread that will later construct something shared. That is exactly
// why run_batch calls warm_writer_pool() from the main thread first: the CPU
// worker now writes through the shared WriterThreadPool, and if it won the race
// to construct it, all 32 compression threads would be born niced and every GPU
// worker's FSE would inherit the penalty — permanently, and silently.
//
// Windows: SetThreadPriority affects only the calling thread and is NOT
// inherited by threads it creates, so the fan-out would stay at normal priority
// and nothing would change. Left unimplemented rather than faked. The CPU worker
// still contends there; the fix on Windows is a pos2-chip thread cap.
// How many nice levels to drop the CPU worker BELOW its peers. A delta, not an
// absolute — see nice_current_thread. 0 disables the whole mechanism.
int cpu_worker_nice_delta()
{
    if (char const* v = std::getenv("XCHPLOT2_CPU_NICE"); v && v[0]) {
        int const n = std::atoi(v);
        if (n >= 0 && n <= 39) return n;  // 39 = the full -20..19 span
    }
    return 10;  // ~9x less CFS weight than its peers under contention
}

void nice_current_thread(int nice_delta, char const* who)
{
    if (nice_delta <= 0) return;
#if defined(__linux__)
    // PRIO_PROCESS with who=0 acts on the CALLING THREAD on Linux — the kernel
    // resolves who==0 to `current`, which is the task, i.e. the thread — despite
    // the POSIX wording that says process.
    //
    // setpriority() takes an ABSOLUTE nice value, not a delta, so read where we
    // actually are and add. The process baseline is NOT reliably 0: this dev box
    // runs ananicy-cpp, which renices by rule, and every thread starts at nice
    // -4. Writing the delta straight through as an absolute would have dropped
    // the worker 14 levels instead of 10 — a ~28x CFS weight ratio against its
    // peers rather than the ~9x the default is tuned for. Same code, two very
    // different machines.
    errno = 0;
    int const cur = ::getpriority(PRIO_PROCESS, 0);
    if (cur == -1 && errno != 0) {  // -1 is also a legal nice value, hence errno
        std::fprintf(stderr,
            "%s warning: could not read scheduling priority (%s); leaving the "
            "CPU worker at normal priority.\n", who, std::strerror(errno));
        return;
    }
    int const target = std::min(19, cur + nice_delta);  // 19 = the nice ceiling
    errno = 0;
    if (::setpriority(PRIO_PROCESS, 0, target) != 0 && errno != 0) {
        std::fprintf(stderr,
            "%s warning: could not lower scheduling priority to nice %d (%s). "
            "The CPU worker will compete with the GPU workers for the host.\n",
            who, target, std::strerror(errno));
    }
#else
    (void)nice_delta;
    (void)who;
#endif
}

// Where this worker's per-plot cost starts. Device bind, pool construction and
// the tier probe sit before it and are per-batch setup, not per-plot cost.
void live_work_start(BatchProgress& live, int slot, double at_s)
{
    live.workers[static_cast<std::size_t>(slot)].work_start.store(
        at_s, std::memory_order_relaxed);
}

// A worker pulled an entry off the queue. It now owes a retirement — a plot or a
// skip — and the ETA prices it as work in flight until then. Getting this wrong
// in either direction is what makes a drain ETA lie, so it is paired strictly
// with live_skip / record_plot_completion below.
void live_claim(BatchProgress& live, int slot)
{
    live.workers[static_cast<std::size_t>(slot)].in_flight.fetch_add(
        1, std::memory_order_relaxed);
}

// Take one plot out of flight, SATURATING AT ZERO.
//
// Not defensive padding — it is load-bearing. run_batch_sharded and
// run_batch_pipeline_plot deliberately never live_claim(): each publishes as a
// single worker, and a one-train ETA prices an in-flight plot and a queued one
// identically, so tracking in-flight there buys nothing. That reasoning is
// sound. What it missed is that they still RETIRE — and an unmatched fetch_sub
// on an unsigned counter does not land on -1, it lands on 2^64-1.
//
// It did exactly that. in_flight wrapped to SIZE_MAX, `retired + in_flight`
// wrapped back to 0, so `unclaimed` pinned to the full batch size and never
// moved, and estimate_eta_seconds priced ~1.8e19 committed plots. --pipeline-plot
// printed "batch ETA ~14m08s" on a three-second run — and still said 14m08s at
// 100% done. Neither strategy runs on a single-GPU box, which is why it survived.
//
// Saturating here rather than forcing a claim on both paths keeps the invariant
// that matters (a retire never invents work) true for any future caller that
// legitimately does not model in-flight, instead of relying on every one of them
// to remember.
void live_retire_in_flight(LiveWorker& w)
{
    std::size_t cur = w.in_flight.load(std::memory_order_relaxed);
    while (cur > 0 &&
           !w.in_flight.compare_exchange_weak(cur, cur - 1,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
        // cur is reloaded by compare_exchange_weak on failure.
    }
}

// ...retired without plotting: the entry was already on disk. Costs no time, so
// it must leave in_flight (or the ETA prices a plot that will never run) but
// must not touch `written` (or it dilutes every rate beside it).
std::size_t live_skip(BatchProgress& live, int slot)
{
    live_retire_in_flight(live.workers[static_cast<std::size_t>(slot)]);
    live.skipped.fetch_add(1, std::memory_order_relaxed);
    return live.retired.fetch_add(1, std::memory_order_relaxed) + 1;
}

// ...retired by failing. It is off the queue and will never be plotted, so it
// must leave in_flight (or the ETA waits forever on a plot nobody is making) and
// must not land in `written` (or it credits the rate with a plot that produced
// nothing). Exactly one of live_skip / live_fail / record_plot_completion pairs
// with each live_claim — on the paths that claim at all; see
// live_retire_in_flight for the two that do not.
std::size_t live_fail(BatchProgress& live, int slot)
{
    live_retire_in_flight(live.workers[static_cast<std::size_t>(slot)]);
    return live.retired.fetch_add(1, std::memory_order_relaxed) + 1;
}

// ...retired by producing a plot. Returns this thread's display ticket.
std::size_t record_plot_completion(BatchResult& res,
                                   std::uint64_t plot_bytes,
                                   double completion_offset_s,
                                   BatchProgress& live,
                                   int slot)
{
    res.bytes_written += plot_bytes;
    res.completion_seconds.push_back(completion_offset_s);

    LiveWorker& w = live.workers[static_cast<std::size_t>(slot)];
    std::size_t const n = w.written.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1) w.first_done.store(completion_offset_s, std::memory_order_relaxed);
    w.last_done.store(completion_offset_s, std::memory_order_relaxed);
    live_retire_in_flight(w);

    live.bytes.fetch_add(plot_bytes, std::memory_order_relaxed);
    live.written.fetch_add(1, std::memory_order_relaxed);
    return live.retired.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Bounded SPSC queue + end-of-stream signal.
//
// Depth = num_pinned_slots - 1 (floored at 1) so the producer never
// overtakes the consumer by more than (N - 1) plots, where N is the run's
// slot count — kNumPinnedBuffers normally, 1 under a --max-host-ram budget
// (see resolve_drain_slots). The pinned slot the producer writes is
// slot (i % N); with depth (N-1) the consumer is guaranteed to have popped
// plot (i - N) before the producer overwrites its slot. At N = 1 the depth
// floor makes that guarantee vacuous, which is fine: the SlotGate below,
// not the depth, is what actually makes reuse safe.
class Channel {
public:
    explicit Channel(std::size_t capacity) : capacity_(capacity) {}

    // Returns false when the channel was closed while waiting — the
    // item was NOT enqueued and will never reach the consumer. Callers
    // must check the result: silently dropping a finished plot here
    // would under-count failures, and a consumer that dies with the
    // queue full would otherwise leave the producer blocked forever.
    [[nodiscard]] bool push(WorkItem item) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_full_.wait(lock, [&]{ return q_.size() < capacity_ || closed_; });
        if (closed_) return false;
        q_.push(std::move(item));
        cv_not_empty_.notify_one();
        return true;
    }
    // Returns false when the channel is closed AND empty.
    bool pop(WorkItem& out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_empty_.wait(lock, [&]{ return !q_.empty() || closed_; });
        if (!q_.empty()) {
            out = std::move(q_.front());
            q_.pop();
            cv_not_full_.notify_one();
            return true;
        }
        return false;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }
private:
    std::mutex                mu_;
    std::condition_variable   cv_not_empty_, cv_not_full_;
    std::queue<WorkItem>      q_;
    std::size_t               capacity_;
    bool                      closed_ = false;
};

// Consumption acknowledgment for the rotating pinned slots.
//
// The Channel's depth alone is NOT enough to make slot reuse safe: a
// depth of (num_pinned_slots - 1) only guarantees the consumer has
// POPPED plot (i - N) before the producer starts plot i — the consumer
// may still be reading that slot's fragments inside
// write_plot_file_parallel (FSE compression + disk write borrow the
// pinned memory via external_fragments_ptr). Whenever the file write
// is slower than one GPU pass (slow disk, NFS), the producer's D2H
// for plot i would land on top of the in-flight read and silently
// corrupt the written plot.
//
// The consumer therefore signals here after it has FINISHED with each
// item (write complete or failed — either way the slot is no longer
// read), and the producer waits until the previous occupant of its
// target slot has been signalled before running the GPU pipeline into
// that slot.
class SlotGate {
public:
    // Consumer: item fully processed; its pinned slot is reusable.
    void signal_consumed() {
        { std::lock_guard<std::mutex> lock(mu_); ++consumed_; }
        cv_.notify_all();
    }
    // Consumer failed: wake any waiting producer so it can observe the
    // failure instead of blocking forever on an ack that never comes.
    void abort() {
        { std::lock_guard<std::mutex> lock(mu_); aborted_ = true; }
        cv_.notify_all();
    }
    // Producer: block until at least `need` items have been consumed
    // (or the consumer aborted). Returns false on abort.
    bool wait_consumed(std::size_t need) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]{ return consumed_ >= need || aborted_; });
        return !aborted_;
    }
private:
    std::mutex              mu_;
    std::condition_variable cv_;
    std::size_t             consumed_ = 0;
    bool                    aborted_  = false;
};

} // namespace

namespace {

// Per-worker pipeline. Extracted from run_batch so the multi-device
// fan-out can spawn N of these concurrently — one thread per device,
// each with its own pool / channel / consumer. The outer run_batch
// validates homogeneity and runs the disk-space preflight once; this
// helper assumes both have already been done on `entries`.
//
// device_id sentinels (see src/gpu/DeviceIds.hpp):
//   kDefaultGpuId (-1) → keep the default SYCL gpu_selector_v
//                        (single-device default; zero-config users
//                        see unchanged behavior).
//   kCpuDeviceId  (-2) → CPU worker via sycl::cpu_selector_v
//                        (--cpu / --devices cpu; AdaptiveCpp OMP
//                        backend, much slower than GPU).
//   0..N-1            → explicit GPU index from get_devices(gpu).
// worker_id  < 0 → single-device path; currently unused beyond
//                  documenting intent but reserved for a future per-
//                  worker log prefix (see fprintf calls below — one
//                  line per call means ordering is already atomic
//                  per-line, so interleaving across workers is
//                  acceptable for v1 without prefix disambiguation).
// shared_idx (default null) lets multiple workers race for the next plot
// out of a single shared `entries` list. When set, every worker calls
// shared_idx->fetch_add(1) and exits when the result >= entries.size() —
// dynamic load balancing, so a fast GPU worker keeps pulling plots while
// a slow CPU worker handles only what it can finish in the same wall.
// When null (single-device path), the worker iterates 0..entries.size()-1
// in order — original behaviour.
namespace {

// Polls the driver's own free-VRAM counter for the life of a batch slice and
// records the low-water mark, giving us the peak device memory the process
// ACTUALLY held — as opposed to what the s_malloc trace believes it held.
//
// This exists because the tier peak models are calibrated against that trace,
// and the trace cannot see a raw sycl::malloc_device by construction. A
// 3128 MiB T2/T3 candidate scratch therefore shipped completely invisible to
// every model in the ladder: the picker told a 7.6 GB Tesla P4 that minimal
// needed 3.67 GiB when it really needed 7.74, and the card OOM'd on every tier
// but tiny. No amount of care with s_malloc would have caught that. Watching
// the number the driver reports is the only check that cannot be fooled by an
// allocation someone forgot to account for.
//
// Caveat: on a GPU shared with another process, that process allocating
// mid-run inflates our measured peak. The check is therefore fatal only under
// POS2GPU_ASSERT_VRAM=1 (which `bench` sets, being a controlled measurement)
// and merely loud elsewhere.
class VramWatchdog {
public:
    explicit VramWatchdog(int ordinal) : ordinal_(ordinal)
    {
        size_t f = 0;
        size_t t = 0;
        // physical_space=true: this measures a DELTA from an idle baseline,
        // and the allocatable clamp saturates at the top, so a clamped baseline
        // under-reports the peak by the physical-minus-allocatable gap (83 MiB
        // on an Arc B580). Under-reporting is the unsafe direction for a check
        // whose job is catching a tier that uses more than it declared.
        if (!device_memory_probe(ordinal_, f, t, /*physical_space=*/true))
            return;  // unsupported → inert
        baseline_free_ = f;
        min_free_.store(f, std::memory_order_relaxed);
        started_ = true;
        th_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed)) {
                size_t f = 0;
                size_t t = 0;
                if (device_memory_probe(ordinal_, f, t,
                                        /*physical_space=*/true)) {
                    size_t cur = min_free_.load(std::memory_order_relaxed);
                    while (f < cur &&
                           !min_free_.compare_exchange_weak(
                               cur, f, std::memory_order_relaxed)) {
                        // cur is reloaded by compare_exchange_weak
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    ~VramWatchdog() { stop(); }

    VramWatchdog(VramWatchdog const&)            = delete;
    VramWatchdog& operator=(VramWatchdog const&) = delete;

    void stop()
    {
        if (!started_) return;
        stop_.store(true, std::memory_order_relaxed);
        if (th_.joinable()) th_.join();
        started_ = false;
    }

    bool     available() const { return baseline_free_ > 0; }
    uint64_t baseline()  const { return baseline_free_; }
    uint64_t peak() const
    {
        size_t const lo = min_free_.load(std::memory_order_relaxed);
        return (baseline_free_ > lo) ? (baseline_free_ - lo) : 0;
    }

private:
    int                 ordinal_;
    std::atomic<bool>   stop_{false};
    std::atomic<size_t> min_free_{0};
    size_t              baseline_free_ = 0;
    bool                started_       = false;
    std::thread         th_;
};

} // namespace

// run_epoch (default null) is the steady_clock origin every worker in a
// multi-device run must share. Completion offsets from a per-worker origin
// cannot be compared across workers — a GPU slice takes its origin after pool
// construction and pinned-host allocation, seconds behind a CPU slice's — so
// merging them yields a timeline that never happened. Null → take a local
// origin (single-worker paths, where nothing is compared across workers).
BatchResult run_batch_slice(std::vector<BatchEntry> const& entries,
                            BatchOptions const& opts,
                            int                 device_id,
                            int                 worker_id,
                            BatchProgress&      live,
                            std::atomic<std::size_t>* shared_idx = nullptr,
                            std::chrono::steady_clock::time_point const*
                                run_epoch = nullptr)
{

    // Per-worker log prefix. Multi-GPU runs interleave stderr from N
    // workers, so prefix every line with which device it came from to
    // keep the log readable. Single-default-device path keeps the
    // historical "[batch]" prefix unchanged for zero log-diff churn
    // on the common case.
    //
    // run_batch fills in a label per worker (see worker_labels) because the
    // device id alone stopped being unique once N CPU workers could share one:
    // four of them all printing "[batch:cpu]" is a log you cannot read. Fall
    // back to the device-derived form when there is no label — that is the
    // single-worker fast path, where the id IS unique.
    std::string const log_prefix = [&]() -> std::string {
        auto const& lbl = live.workers[static_cast<std::size_t>(worker_id)].label;
        if (!lbl.empty()) return "[batch:" + lbl + "]";
        return is_cpu_device(device_id) ? std::string("[batch:cpu]") :
               (device_id <  0)            ? std::string("[batch]")     :
               ("[batch:gpu" + std::to_string(device_id) + "]");
    }();

    // "vram" is a lie on the CPU device: its "device memory" IS host RAM, and
    // the free figure behind it now comes from /proc/meminfo MemAvailable
    // rather than a GPU (it used to come from GPU 0 — see device_memory_probe).
    // The tier machinery below is otherwise identical for both, so only the
    // noun changes.
    char const* const mem_label =
        is_cpu_device(device_id) ? "ram" : "vram";

    // ...but the progress line is the one thing here that is NOT this worker's.
    // Its counters are the whole batch's, so "s/plot avg" is the average across
    // every worker. Wearing a per-worker prefix, it read as that worker's rate: a
    // real 2-worker run signed off with "[batch:cpu] ... 6.88 s/plot avg" while
    // that CPU was plodding at 63.7 s/plot and the GPU was carrying the batch.
    // Aggregate numbers get an aggregate prefix; per-worker rates live in bench's
    // per-worker block. Single-worker keeps its prefix — there the batch IS the
    // worker, and the two agree.
    std::string const progress_prefix =
        live.workers.size() > 1 ? std::string("[batch]") : log_prefix;

    // CPU worker: bypass the GPU pool / streaming path entirely. pos2-chip's
    // Plotter manages all internal state itself, so each plot is a
    // synchronous run_one_plot_cpu() call.
    //
    // It is NOT single-threaded (this comment said so for two revisions; see
    // CpuPlotter.hpp for the measurements). pos2-chip fans out to
    // hardware_concurrency() with no cap and no pool, so on a GPU+CPU batch it
    // and our WriterThreadPool — also sized at hardware_concurrency() — put
    // ~2x core_count runnable threads on the box, and the CPU plotter is the
    // one that runs hot continuously. CFS then splits the machine during every
    // FSE burst, the GPU's consumer slows, its depth-1 channel backs up, and
    // the GPU stalls. On a 12900k/RTX3080 that cost 62% of the CPU worker's
    // theoretical contribution: adding it should have taken 6.7 -> 5.74 s/plot
    // and only reached 6.3. The fix is to de-prioritise this worker (nice is a
    // per-thread attribute on Linux and clone() carries it into the threads
    // pos2-chip spawns), NOT to cap its thread count: capping is static, and
    // the GPU's demand for the host is bursty, so a reserved core sits idle
    // between bursts. Affinity is not a substitute — measured, glibc 2.43's
    // hardware_concurrency() ignores the affinity mask, so taskset confines
    // the threads without reducing how many get spawned.
    //
    // There can be N of these now (--cpu-workers / repeated `cpu` tokens). Each
    // is an ordinary work-queue worker pulling off the same shared counter, and
    // they share nothing but the queue and the writer pool — so nothing here
    // changes for N > 1 except that the RAM cost is N x 12.14 GiB at k=28, which
    // resolve_batch_devices gates against the host before we ever get here.
    //
    // XCHPLOT2_SYCL_CPU_BENCH=1 routes --cpu through the SYCL pipeline on
    // AdaptiveCpp's CPU backend instead of pos2-chip — exposed as an env
    // var purely for benchmarking the two CPU paths against each other,
    // not as a supported plotting mode (pos2-chip is faster + leaner).
    bool const sycl_cpu_bench = [] {
        char const* v = std::getenv("XCHPLOT2_SYCL_CPU_BENCH");
        return v && v[0] == '1';
    }();
    if (is_cpu_device(device_id) && !sycl_cpu_bench) {
        BatchResult res;
        res.pipeline = "cpu";  // no tier to pick; see WorkerTimeline::pipeline
        if (entries.empty()) return res;

        // Confine this worker to its own NUMA node — and, because Linux copies
        // the affinity mask at clone(), the entire fan-out pos2-chip spawns
        // inside run_one_plot_cpu() too. Same inheritance trick the nice below
        // rides on, and it must happen here for the same reason: before the
        // threads that need to inherit it exist.
        //
        // The point is memory locality, not core budgeting. This plotter is
        // memory-latency-bound, so a worker reading a working set that lives on
        // another socket pays the remote hop on its hottest path. It does NOT
        // reduce the fan-out — hardware_concurrency() reports the whole host
        // whatever the mask says (NumaTopology.hpp) — so a node-pinned worker
        // still oversubscribes its node, and the per-node knee has to be
        // measured rather than assumed.
        //
        // Gated on nodes.size() > 1, so a single-node host never pins: there is
        // nowhere else for the memory to be, and narrowing a mask to the only
        // node it could already use buys nothing.
        //
        // The guard outlives the pin for the whole slice and hands the thread's
        // mask back on the way out — load-bearing because the single-worker fast
        // path runs this on the CALLER's thread. See ScopedThreadAffinity.
        //
        // XCHPLOT2_CPU_NO_PIN=1 keeps the node-per-worker layout and the labels
        // but skips the pin, which is the only way to A/B what locality is worth
        // on a given box: the alternative (`--devices cpu0`) also halves the
        // cores, so it measures two changes at once. It is a diagnostic, and it
        // makes the cpuN labels describe an intent the run no longer honours —
        // so it announces itself rather than skewing a benchmark silently.
        char const* no_pin = std::getenv("XCHPLOT2_CPU_NO_PIN");
        bool const  pin_disabled = no_pin && no_pin[0] == '1';

        ScopedThreadAffinity affinity_restore;
        auto const nodes = host_numa_nodes();
        // Both branches require a multi-node host: on a single-node box there is
        // no pin to make and none to skip, so NO_PIN must stay silent there
        // rather than announce a no-op as though it had changed something.
        //
        // `quiet` gates only the message, never the dispatch — folding it into
        // this condition would let `--quiet` fall through to the pin and make
        // NO_PIN silently do nothing, which is the one way a diagnostic can lie.
        if (nodes.size() > 1 && pin_disabled) {
            if (!opts.quiet) {
                std::fprintf(stderr,
                    "%s XCHPLOT2_CPU_NO_PIN=1 — NUMA pin skipped; this worker "
                    "may run on any node regardless of its label\n",
                    log_prefix.c_str());
            }
        } else if (nodes.size() > 1) {
            int const want = cpu_numa_node(device_id);
            for (auto const& n : nodes) {
                if (n.node_id != want) continue;
                if (!pin_thread_to_cpus(n.cpus)) {
                    if (!opts.quiet) {
                        std::fprintf(stderr,
                            "%s could not pin to NUMA node %d — running unpinned "
                            "(slower on a multi-socket host, but not wrong)\n",
                            log_prefix.c_str(), want);
                    }
                } else if (opts.verbose) {
                    // Read the mask BACK rather than echoing what we asked for.
                    // The label already says which node this worker was meant to
                    // get; only the kernel can say which cpus it may actually
                    // use, and a silent success is indistinguishable from a
                    // no-op. Thread count cannot stand in for this: pos2-chip
                    // sizes its fan-out from hardware_concurrency(), which
                    // ignores the mask, so a pinned worker still spawns the
                    // whole host's worth of threads onto this node's cpus.
                    auto const got = current_thread_cpus();
                    std::fprintf(stderr, "%s pinned to NUMA node %d (cpus %s)\n",
                                 log_prefix.c_str(), want,
                                 got.empty() ? "unreadable"
                                             : format_cpu_list(got).c_str());
                }
                break;
            }
        }

        // Yield to the GPU workers. Only when there ARE any:
        //
        //  * With the CPU as the sole worker, this slice runs on the MAIN thread
        //    (run_batch's single-worker fast path calls run_batch_slice inline),
        //    and nicing is irreversible for an unprivileged process — we would
        //    permanently de-prioritise the whole process to yield to nobody.
        //
        //  * With N CPU workers and no GPU, nicing all of them equally yields to
        //    nobody either — it just parks every plotter below the writer pool's
        //    threads, which are the one thing that must NOT outrank them.
        //
        // So the test is "is there a GPU peer", not "is there a peer".
        if (live.gpu_peer_present) {
            nice_current_thread(cpu_worker_nice_delta(), log_prefix.c_str());
        }

        auto const t_start = run_epoch ? *run_epoch
                                       : std::chrono::steady_clock::now();
        res.work_start_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        live_work_start(live, worker_id, res.work_start_seconds);
        std::size_t local_idx = 0;
        while (true) {
            // Fund the next plot BEFORE claiming it.
            //
            // The order matters: a worker that gave up AFTER pulling an index off
            // the shared counter would strand that plot — the queue is a
            // fetch_add, there is nowhere to put one back — and the batch would
            // quietly finish short. Ask for the memory first; if the answer is no,
            // this worker simply never claims the work, and its peers take it.
            //
            // Declared inside the loop so every exit path — break, continue,
            // throw, or a normal iteration — hands the reservation back.
            // Stop waiting the moment the work is gone. A worker parked in a gate
            // cannot see the queue drain or a cancel land, and on a GPU+CPU rig
            // that is a hang: the GPUs finish the batch and run_batch joins a CPU
            // worker still waiting to fund (or be admitted for) a plot nobody
            // needs any more. Shared by both admission gates below.
            auto const still_wanted = [&]() -> bool {
                if (cancel_requested()) return false;
                if (shared_idx &&
                    shared_idx->load(std::memory_order_relaxed) >=
                        entries.size()) {
                    return false;
                }
                return true;
            };

            // Adaptive admission, ahead of the memory gate: park this worker until
            // the GPU's measured rate justifies its ordinal (XCHPLOT2_CPU_ADAPTIVE).
            // A no-op when the governor is unset (the default) or the worker is
            // already within target. Parking here holds no memory, so it never
            // deadlocks the memory gate.
            if (live.cpu_rate_gov &&
                !live.cpu_rate_gov->wait_active(worker_id, still_wanted)) {
                break;  // the batch is over, or cancelled — this worker retires
            }

            // Tail guard (default on): stop pulling once the faster workers will
            // clear the queue before this CPU worker finishes one more plot — the
            // 40-50 s CPU tail a fast GPU would otherwise wait on, and the whole of
            // a batch too short to help. Ahead of the memory gate so a stood-down
            // worker never sat holding RAM.
            if (tail_guard_enabled() && shared_idx) {
                std::size_t const claimed =
                    shared_idx->load(std::memory_order_relaxed);
                std::size_t const remaining =
                    claimed < entries.size() ? entries.size() - claimed : 0;
                std::string why;
                if (tail_guard_should_retire(live, worker_id, remaining, &why)) {
                    if (!opts.quiet)
                        std::fprintf(stderr, "%s %s\n",
                                     log_prefix.c_str(), why.c_str());
                    break;
                }
            }

            CpuMemoryLease lease;
            if (live.cpu_gate) {
                if (live.cpu_gate->acquire(log_prefix.c_str(), opts.quiet,
                                           still_wanted) ==
                    CpuMemoryGate::Verdict::Admitted) {
                    lease = CpuMemoryLease(live.cpu_gate.get());
                } else if (!still_wanted()) {
                    break;  // the batch is over, or cancelled — nothing to fund
                } else {
                    // Denied: the box has been too small for a single plot,
                    // continuously, for the whole grace period, with no peer of
                    // ours holding anything it could give back.
                    if (live.gpu_peer_present) {
                        // The GPUs are fine and still have a queue to drain. Leave
                        // quietly rather than taking the whole batch down.
                        std::fprintf(stderr,
                            "%s retiring: the host no longer has room for a %.1f "
                            "GiB plot (%.1f GiB free to us). The GPU workers "
                            "continue.\n",
                            log_prefix.c_str(),
                            static_cast<double>(live.cpu_gate->per_worker()) /
                                (1024.0 * 1024.0 * 1024.0),
                            live.cpu_gate->budget_gib_now());
                        break;
                    }
                    // Nothing else can make progress. Say so instead of spinning.
                    throw std::runtime_error(
                        "CPU worker: the host no longer has room for a plot at "
                        "this k — something else on this machine took the memory "
                        "since the batch started, and there is no GPU worker to "
                        "fall back on. Lower --cpu-workers, lower -k, or free "
                        "memory.");
                }
            }

            std::size_t const i = shared_idx
                ? shared_idx->fetch_add(1, std::memory_order_relaxed)
                : local_idx++;
            if (i >= entries.size()) break;
            live_claim(live, worker_id);
            if (opts.skip_existing) {
                auto out_path = std::filesystem::path(entries[i].out_dir)
                                / entries[i].out_name;
                if (looks_like_complete_plot(out_path)) {
                    if (opts.verbose) {
                        std::fprintf(stderr,
                            "%s skipping plot %zu: %s (already exists)\n",
                            log_prefix.c_str(), i, out_path.string().c_str());
                    }
                    ++res.plots_skipped;
                    std::size_t const done_now = live_skip(live, worker_id);
                    if (opts.progress) {
                        emit_progress_line(
                            progress_prefix, opts, live, done_now,
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count());
                    }
                    continue;
                }
            }
            try {
                std::uint64_t const plot_bytes =
                    run_one_plot_cpu(entries[i], opts);
                ++res.plots_written;
                double const completion_offset = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                std::size_t const done_now = record_plot_completion(
                    res, plot_bytes, completion_offset, live, worker_id);
                if (opts.verbose) {
                    std::fprintf(stderr,
                        "%s plot %zu done: %s\n",
                        log_prefix.c_str(),
                        i, entries[i].out_name.c_str());
                }
                if (opts.progress) {
                    emit_progress_line(
                        progress_prefix, opts, live, done_now,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_start).count());
                }
            } catch (std::exception const& ex) {
                std::fprintf(stderr,
                    "%s plot %zu FAILED: %s\n",
                    log_prefix.c_str(), i, ex.what());
                ++res.plots_failed;
                live_fail(live, worker_id);
                if (!opts.continue_on_error) {
                    res.total_wall_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start).count();
                    return res;
                }
            }
            if (cancel_requested()) break;
        }
        res.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return res;
    }

    if (device_id >= 0 || is_cpu_device(device_id)) bind_current_device(device_id);
    initialize_aes_tables();

    bool const verbose = opts.verbose;

    BatchResult res;
    if (entries.empty()) return res;

    // Pool shape from the first entry. Homogeneity (all entries share
    // k/strength/testnet) was checked by the outer run_batch.
    int  pool_k        = entries[0].k;
    int  pool_strength = entries[0].strength;
    bool pool_testnet  = entries[0].testnet;

    // Allocate the pool once; destructor frees at function exit. This is
    // the whole point of the batch path — eliminate the per-plot ~2.4 s
    // allocator cost (dominated by cudaMallocHost(2 GB)).
    //
    // On insufficient device VRAM (small card), the pool ctor throws
    // InsufficientVramError. Fall back to the streaming pipeline per
    // plot — slower (no buffer amortisation across plots, no
    // producer/consumer overlap between GPU D2H and consumer I/O on
    // pinned double-buffered pool slots), but it fits inside the card's
    // VRAM and is still overlapped via the Channel between the producer
    // thread's streaming call and the consumer thread's FSE compression
    // + plot-file write.
    // Device bytes the T2/T3 two-phase match may spend on its candidate
    // scratch, granted below once we know which path (pool or streaming tier)
    // we are on. See SyclBackend.hpp for what this is guarding against.
    uint64_t twophase_budget_bytes = 0;

    // Modelled device footprint of whichever path we end up on — the tier's
    // peak for streaming, the pool's full sizing for the pool path. The VRAM
    // watchdog checks the real peak against this, plus the scratch we granted,
    // plus the margin.
    uint64_t declared_base_bytes = 0;

    // Start sampling BEFORE anything is allocated, so the baseline is genuinely
    // "free VRAM before we touched the card" and the peak covers the pool
    // allocation itself.
    VramWatchdog vram(device_id);

    // Free VRAM before we allocate anything. Both grants below are computed
    // against this, never against free-after-allocation — see the note in the
    // pool branch about d_pair_a being allocated lazily.
    DeviceMemInfo const mem_before_pool = query_device_memory();

    std::unique_ptr<GpuBufferPool> pool_ptr;
    // Streaming-fallback pinned buffers — double-buffered the same way the
    // pool does, so producer's D2H of plot N+1 can run concurrently with
    // the consumer reading plot N. cudaMallocHost is ~600 ms, so doing it
    // once instead of per plot is a significant win on long batches.
    uint64_t* stream_pinned[GpuBufferPool::kNumPinnedBuffers] = {};
    size_t    stream_pinned_cap = 0;
    // B1 (host-RAM disk-offload): how many of those rotating slots this run
    // actually uses. The slots buy ONE thing — overlapping plot K's D2H with
    // the file writer draining K-1/K-2. They are not a correctness
    // requirement: slot_gate (below) already blocks reuse until the consumer
    // has finished with a slot, so a single slot is correct, merely less
    // overlapped. Each slot given up hands back cap * 8 B of pinned host
    // memory (2.03 GiB at k=28).
    //
    // The win lands even for a one-plot run, unlike the pool path's lazy
    // ensure_pinned(): the streaming branch allocates every slot UP FRONT
    // (see the loop below), which is why compact measured the same host peak
    // at n=1 and n=3.
    //
    // Finalised by the budget policy below, which cuts slots only as far as
    // the budget actually demands — see the "last resort" note there.
    int num_pinned_slots = GpuBufferPool::kNumPinnedBuffers;
    int const forced_drain_slots = drain_slots_env();      // 0 when unset
    if (forced_drain_slots) num_pinned_slots = forced_drain_slots;
    // Stage 4f: amortised streaming-path pinned-host scratch. Populated
    // in the streaming-fallback branch below; nullptr fields when the
    // pool path is active (pool_ptr != null).
    StreamingPinnedScratch stream_scratch{};
    // The pipeline's [spill] lines are per plot; -q silences them. The budget
    // line below still prints (it is one per slice and names every routed
    // table), so -q loses no information about what the spill is doing.
    stream_scratch.quiet = opts.quiet;
    // Phase 2-26: per-batch host-pinned pool for the per-plot allocs
    // that stream_scratch fields don't already amortise (h_t1_mi,
    // h_t2_mi, and h_keys_merged when stream_scratch.h_keys_merged is
    // null). Wired into stream_scratch.pool below in the streaming-
    // fallback branch only — the GpuBufferPool path doesn't hit the
    // streaming code at all.
    HostPinnedPool stream_pool;

    // Force-streaming override (matches the one-shot run_gpu_pipeline
    // dispatch). Useful for testing the streaming path on a high-VRAM
    // card and for users who want the smaller peak even when the pool
    // would fit. Also triggered by an explicit `--tier <non-auto>` or
    // `XCHPLOT2_STREAMING_TIER` selection — without this the tier
    // selector below is unreachable on cards where the pool fits, so
    // `--tier tiny` on a 24 GB card was silently ignored.
    char const* tier_env_pre = std::getenv("XCHPLOT2_STREAMING_TIER");

    // Resolve the effective tier for THIS worker. Precedence (highest
    // wins):
    //   1. --devices `<id>:<tier>` override for this device_id
    //   2. --devices `gpu:<tier>` / `all:<tier>` shorthand
    //   3. global --tier / XCHPLOT2_STREAMING_TIER
    //   4. (empty) → auto-pick from free VRAM
    // The `auto` value is a sentinel that explicitly opts back into
    // auto-pick — used to override a higher-level default for one GPU.
    std::string effective_tier;
    if (auto it = opts.per_device_tier.find(device_id);
        it != opts.per_device_tier.end())
    {
        effective_tier = (it->second == "auto") ? std::string() : it->second;
    } else if (!opts.all_gpus_tier.empty()) {
        effective_tier = opts.all_gpus_tier;
    } else if (!opts.streaming_tier.empty()) {
        effective_tier = opts.streaming_tier;
    } else if (tier_env_pre && tier_env_pre[0] != '\0') {
        effective_tier = tier_env_pre;
    }

    bool const tier_forced = !effective_tier.empty();
    bool const force_streaming = [tier_forced] {
        char const* v = std::getenv("XCHPLOT2_STREAMING");
        if (v && v[0] == '1') return true;
        return tier_forced;
    }();

    try {
        if (force_streaming) {
            throw InsufficientVramError(
                tier_forced ? "--tier override forced streaming"
                            : "XCHPLOT2_STREAMING=1 forced");
        }
        pool_ptr = std::make_unique<GpuBufferPool>(
            pool_k, pool_strength, pool_testnet);

        // The pool's full device footprint — the same sum its own VRAM gate
        // checks. Note this is NOT what is resident right now: d_pair_a
        // (4.36 GB at k=28) is allocated lazily on first use, so querying free
        // VRAM straight after the constructor overstates the headroom by more
        // than 4 GB. Grant the two-phase scratch against the pool's full
        // requirement, exactly as the streaming path grants against the tier's
        // modelled peak.
        declared_base_bytes = pool_ptr->storage_bytes
                            + pool_ptr->pair_a_bytes
                            + pool_ptr->pair_b_bytes
                            + pool_ptr->sort_scratch_bytes;
        twophase_budget_bytes =
            (mem_before_pool.free_bytes > declared_base_bytes + vram_safety_margin())
                ? mem_before_pool.free_bytes - declared_base_bytes - vram_safety_margin()
                : 0;
        // The pool fit — record it, same reason as the streaming tier below.
        // Pool-vs-streaming is the LARGER of the two differences a two-pass
        // caller can accidentally straddle, so it has to be distinguishable
        // from a tier name, not folded into one.
        res.pipeline = "pool";
    } catch (InsufficientVramError const& e) {
        if (opts.quiet) {
            // info-level: which pipeline was picked and why
        } else if (tier_forced) {
            std::fprintf(stderr, "%s --tier override — using "
                                 "streaming pipeline per plot\n",
                                 log_prefix.c_str());
        } else if (force_streaming) {
            std::fprintf(stderr, "%s XCHPLOT2_STREAMING=1 — using "
                                 "streaming pipeline per plot\n",
                                 log_prefix.c_str());
        } else if (e.from_allocation) {
            // The gate said yes and the driver said no. Worth its own line:
            // the free-VRAM figure the gate trusted was wrong, which on a
            // backend without a real free-memory query (Level Zero before the
            // Sysman probe) is expected, and elsewhere means another process
            // took VRAM in between. Quoting a "free" number here would be
            // quoting the very figure that just proved untrue.
            std::fprintf(stderr,
                "%s pool allocation of %.2f GiB failed despite the gate "
                "allowing it — using streaming pipeline per plot. (%s)\n",
                log_prefix.c_str(),
                e.required_bytes / double(1ULL << 30), e.what());
        } else {
            std::fprintf(stderr,
                "%s pool needs %.2f GiB, only %.2f GiB free — using "
                "streaming pipeline per plot\n",
                log_prefix.c_str(),
                e.required_bytes / double(1ULL << 30),
                e.free_bytes     / double(1ULL << 30));
        }
        // Streaming tier dispatch — increasing PCIe pressure for decreasing
        // peak VRAM. Peaks below are the s_malloc-tracked anchors; add ~390 MB
        // of CUDA context for the true process peak:
        //   plain   (~7290 MB at k=28): no parks, single-pass T2 match.
        //                               Fastest, ~400 ms/plot over compact.
        //                               Serves ~8 GiB cards and up.
        //   compact (~5200 MB at k=28): all parks + N=2 T2 match staging.
        //                               Serves ~6 GiB cards and up.
        //   minimal (~3900 MB at k=28): compact's parks + N=8 T2 match
        //                               staging. Serves ~5 GiB cards (NOT
        //                               4 GiB — the true peak with the context
        //                               is ~4274 MiB; a 4 GiB card lands on
        //                               tiny, correctly).
        //   tiny    (~1100 MB at k=28): the FLOOR. Serves ~2 GiB and up.
        //
        // Auto-pick takes the largest tier that fits with the margin, and
        // floors at tiny. It used to fall through to `pinned` below tiny —
        // which could never work, because pinned's anchor (2200 MB) is TWICE
        // tiny's (1100 MB). The streaming-partition work pinned was scaffolded
        // for landed inside tiny_mode instead (Phase 1.5/1.6 took tiny from
        // 2.1 GB to 1064 MB) and pinned's anchor was never revisited, so the
        // "smaller tier below tiny" was in fact the largest of the two. A card
        // under tiny's floor got handed a tier needing 2712 MiB when it had
        // less than 1612, and the very next check threw. Tiny is the floor;
        // throw there, with an honest message. `--tier pinned` survives as a
        // manual label only.
        //
        // opts.streaming_tier (--tier CLI flag) > XCHPLOT2_STREAMING_TIER env
        // var > auto. A forced tier below its floor warns but proceeds
        // (caller's risk); auto-picked tiny below its floor throws, because
        // there is nothing smaller to fall back to.
        {
            auto const mem            = query_device_memory();
            size_t const plain_peak   = streaming_plain_peak_bytes(pool_k);
            size_t const compact_peak = streaming_peak_bytes(pool_k);
            size_t const minimal_peak = streaming_minimal_peak_bytes(pool_k);
            size_t const tiny_peak    = streaming_tiny_peak_bytes(pool_k);
            size_t const pinned_peak  = streaming_pinned_peak_bytes(pool_k);
            size_t const margin       = vram_safety_margin();
            auto to_gib = [](size_t b) { return b / double(1ULL << 30); };

            // Use effective_tier resolved at the top of run_batch_slice
            // (per-device override > gpu:tier shorthand > global --tier
            // > env), falling back to auto-pick by free VRAM when no
            // override is in effect.
            std::string const& tier_pref = effective_tier;

            enum class Tier { Plain, Compact, Minimal, Tiny, Pinned };
            Tier tier;
            if (tier_pref == "plain") {
                tier = Tier::Plain;
            } else if (tier_pref == "compact") {
                tier = Tier::Compact;
            } else if (tier_pref == "minimal") {
                tier = Tier::Minimal;
            } else if (tier_pref == "tiny") {
                tier = Tier::Tiny;
            } else if (tier_pref == "pinned") {
                tier = Tier::Pinned;
            } else {
                // Auto: pick the largest tier that fits with margin, flooring
                // at Tiny. Pinned is deliberately NOT in this ladder — see the
                // block comment above: its anchor is 2x Tiny's, so using it as
                // the sub-Tiny fallback guaranteed a throw.
                tier = (mem.free_bytes >= plain_peak   + margin) ? Tier::Plain   :
                       (mem.free_bytes >= compact_peak + margin) ? Tier::Compact :
                       (mem.free_bytes >= minimal_peak + margin) ? Tier::Minimal :
                                                                   Tier::Tiny;
            }

            auto tier_name = [](Tier t) -> char const* {
                return t == Tier::Plain   ? "plain"
                     : t == Tier::Compact ? "compact"
                     : t == Tier::Minimal ? "minimal"
                     : t == Tier::Tiny    ? "tiny"
                     :                      "pinned";
            };
            size_t const required =
                tier == Tier::Plain   ? plain_peak   :
                tier == Tier::Compact ? compact_peak :
                tier == Tier::Minimal ? minimal_peak :
                tier == Tier::Tiny    ? tiny_peak    :
                                        pinned_peak;

            // Open-ended fallback: if even the smallest tier (tiny) won't fit,
            // throw. A tier the caller FORCED below its floor warns and
            // proceeds at their risk — including a forced tiny, which is why
            // the throw is gated on the tier having been auto-picked.
            bool const auto_picked = tier_pref.empty();
            if (tier == Tier::Tiny && auto_picked
                && mem.free_bytes < required + margin) {
                InsufficientVramError se(
                    log_prefix + " streaming pipeline needs ~" +
                    std::to_string(to_gib(required + margin)).substr(0, 5) +
                    " GiB peak for k=" + std::to_string(pool_k) +
                    " (tiny tier, the smallest available), device reports " +
                    std::to_string(to_gib(mem.free_bytes)).substr(0, 5) +
                    " GiB free of " +
                    std::to_string(to_gib(mem.total_bytes)).substr(0, 5) +
                    " GiB total. Use a smaller k or a larger GPU "
                    "(or --cpu for pos2-chip CPU plotting).");
                se.required_bytes = required + margin;
                se.free_bytes     = mem.free_bytes;
                se.total_bytes    = mem.total_bytes;
                throw se;
            }
            if (!(tier == Tier::Tiny && auto_picked)
                && mem.free_bytes < required + margin) {
                std::fprintf(stderr,
                    "%s streaming tier: %s forced (%.2f GiB free < %.2f GiB "
                    "%s floor) — proceeding, may OOM mid-plot\n",
                    log_prefix.c_str(),
                    tier_name(tier),
                    to_gib(mem.free_bytes),
                    to_gib(required + margin),
                    tier_name(tier));
            }

            // Host RAM. Everything above sized the DEVICE peak; this is the
            // other half of the budget, and it moves the OPPOSITE way — see
            // the measured table in GpuBufferPool.hpp. A card small enough to
            // need Tiny usually sits in a box that can least afford Tiny's
            // 19.6 GiB of host staging, so weighing only VRAM is how a 14 GiB
            // machine ends up inviting the OOM killer instead of being told no.
            //
            // These are estimates. host_pinned_reserve_check() at each
            // allocation is the mechanism that actually keeps the host alive;
            // this exists to fail before the setup cost rather than during it.
            auto const host_need = [pool_k](Tier t) -> size_t {
                switch (t) {
                    case Tier::Plain:   return streaming_plain_host_bytes(pool_k);
                    case Tier::Compact: return streaming_compact_host_bytes(pool_k);
                    case Tier::Minimal: return streaming_minimal_host_bytes(pool_k);
                    // Pinned shares tiny_mode's host-park behaviour, so it
                    // carries tiny's host cost too.
                    case Tier::Tiny:
                    case Tier::Pinned:  break;
                }
                return streaming_tiny_host_bytes(pool_k);
            };
            size_t const host_required = host_need(tier);
            size_t const host_reserve  = host_memory_reserve();

            size_t host_free = 0, host_total = 0;
            bool const have_host =
                device_memory_probe(kCpuDeviceId, host_free, host_total);

            // A forced tier that costs MORE host RAM than the auto-pick is
            // worth saying out loud: --tier tiny is the obvious reach for "use
            // less memory", and it is the single worst choice for host RAM.
            if (!auto_picked && have_host) {
                Tier const auto_tier =
                    (mem.free_bytes >= plain_peak   + margin) ? Tier::Plain   :
                    (mem.free_bytes >= compact_peak + margin) ? Tier::Compact :
                    (mem.free_bytes >= minimal_peak + margin) ? Tier::Minimal :
                                                                Tier::Tiny;
                size_t const auto_host = host_need(auto_tier);
                if (tier != auto_tier && host_required > auto_host) {
                    std::fprintf(stderr,
                        "%s --tier %s needs ~%.2f GiB of HOST RAM; %s also fits "
                        "this GPU and needs ~%.2f GiB. Lower tiers buy VRAM WITH "
                        "host memory — they do not reduce memory overall.\n",
                        log_prefix.c_str(),
                        tier_name(tier), to_gib(host_required),
                        tier_name(auto_tier), to_gib(auto_host));
                }
            }

            // Host-RAM disk-offload budget policy (the arithmetic is
            // HostRamPolicy.hpp). Turn the guard's "refuse"
            // (below) into "spill enough, then proceed": redirect the large
            // cold pinned tables to a TempFile-backed home, LARGEST-FIRST,
            // until the modelled resident host peak fits under the budget.
            // The chosen set is threaded to the streaming pipeline via
            // stream_scratch.spill; the matching scratch fields are left null
            // in the allocation block further down so the pipeline OWNS and
            // spills them.
            //
            // Two ways in. EXPLICIT: --max-host-ram names the budget. AUTO:
            // nobody named one, but the tier does not fit host RAM and the
            // guard below would throw — so adopt the budget the host actually
            // has and spill into it. AUTO fires only where the alternative is
            // refusing to plot at all, so it can never slow down a run that
            // works today; that asymmetry is what makes it safe to have on by
            // default. --no-auto-spill / XCHPLOT2_NO_AUTO_SPILL=1 opts out,
            // for anyone who would rather be told no than plot slowly.
            bool        spill_auto  = false;
            bool        do_spill    = opts.has_max_host_ram;
            uint64_t    budget      = opts.max_host_ram;   // 0 == "min"
            bool        auto_blocked_ram_dir = false;
            bool        auto_blocked_bad_dir = false;
            std::string auto_dir_problem;
            std::string auto_ram_dir;
            uint64_t    auto_floor_bytes = 0;
            bool        auto_unreachable = false;

            if (!do_spill && opts.auto_host_ram_spill && have_host
                && host_free < host_required + host_reserve) {
                budget     = (host_free > host_reserve)
                                 ? uint64_t(host_free) - uint64_t(host_reserve)
                                 : 0;   // nothing to spare -> "min"
                do_spill   = true;
                spill_auto = true;
            }

            // Guard (fail-fast, before any allocation or plotting): refuse
            // to "spill" onto a RAM-backed temp dir. On most systemd/Arch
            // boxes /tmp is tmpfs — i.e. RAM — so writing the spill tables
            // there consumes the very RAM this budget is meant to cap and
            // invites the OOM killer, defeating the feature. The spill
            // TempFiles resolve their dir from XCHPLOT2_TEMP_DIR / TMPDIR /
            // /tmp (--temp-dir feeds XCHPLOT2_TEMP_DIR), so probe that same
            // resolved dir. XCHPLOT2_ALLOW_RAM_TEMP_DIR (1/true/yes/on)
            // downgrades the refusal to a warning for the rare disk-backed
            // /tmp.
            //
            // An EXPLICIT budget throws here: the user asked for a spill and
            // needs to know it would have been a lie. AUTO must not — the user
            // asked for a plot, not a spill, so a tmpfs temp dir simply means
            // this rescue is unavailable. Record why and let the host-RAM
            // guard below deliver the verdict with that clause attached, which
            // tells them both facts at once instead of trading one confusing
            // error for another.
            if (do_spill) {
                std::string const spill_dir = TempFile::resolve_dir("");
                if (TempFile::dir_is_ram_backed(spill_dir)) {
                    char const* ov =
                        std::getenv("XCHPLOT2_ALLOW_RAM_TEMP_DIR");
                    std::string const ovs = ov ? ov : "";
                    bool const allow = (ovs == "1" || ovs == "true" ||
                                        ovs == "yes" || ovs == "on");
                    if (allow) {
                        std::fprintf(stderr,
                            "%s WARNING: host-RAM spill temp dir '%s' "
                            "is on a RAM-backed filesystem (tmpfs); "
                            "proceeding anyway because "
                            "XCHPLOT2_ALLOW_RAM_TEMP_DIR is set.\n",
                            log_prefix.c_str(), spill_dir.c_str());
                    } else if (spill_auto) {
                        do_spill              = false;
                        auto_blocked_ram_dir  = true;
                        auto_ram_dir          = spill_dir;
                    } else {
                        throw std::runtime_error(
                            "--max-host-ram is set but the spill temp dir '" +
                            spill_dir + "' is on a RAM-backed filesystem "
                            "(tmpfs); spilling there consumes RAM and "
                            "defeats the budget. Point --temp-dir (or "
                            "XCHPLOT2_TEMP_DIR) at real disk.");
                    }
                }
            }

            // Second half of the same guard: the dir must actually be
            // usable. dir_is_ram_backed() above returns false when it cannot
            // probe at all — a mistyped path, an unmounted drive — so a bad
            // --temp-dir passes that check and then dies deep in the pipeline
            // with a raw mkstemp errno, minutes into a batch. Since the tmpfs
            // message tells users to reach for --temp-dir, mistyping it is
            // the expected next failure and belongs here, before any work.
            if (do_spill) {
                std::string const spill_dir = TempFile::resolve_dir("");
                std::string const problem   = TempFile::dir_problem(spill_dir);
                if (!problem.empty()) {
                    if (spill_auto) {
                        do_spill             = false;
                        auto_blocked_bad_dir = true;
                        auto_dir_problem     = problem;
                        auto_ram_dir         = spill_dir;
                    } else {
                        throw std::runtime_error(
                            "--max-host-ram is set but no spill file can be "
                            "created in the temp dir '" + spill_dir +
                            "': " + problem +
                            ". Point --temp-dir (or XCHPLOT2_TEMP_DIR) at an "
                            "existing, writable directory on real disk.");
                    }
                }
            }

            if (do_spill) {

                // cap entries, from the public estimators: each is
                // cap·bpe + fixed, so the difference over the bpe delta
                // (52-24) is cap exactly.
                uint64_t const cap_entries =
                    (streaming_compact_host_bytes(pool_k)
                       - streaming_plain_host_bytes(pool_k)) / 28;
                uint64_t const B = budget;   // 0 == "min"

                // Routable tables for this tier, LARGEST-FIRST (all 8-B
                // tables before the 4-B one). Only those whose spill path
                // is implemented AND safe for the tier are listed (see
                // GpuPipeline.cpp):
                //   h_t1_meta  (8 B, Tiny only,            DMA/SpillBuffer)
                //   h_t3       (8 B, Compact/Minimal/Tiny, DMA/SpillBuffer)
                //   h_t2_meta  (8 B, Tiny only,            DMA/SpillBuffer)
                //   h_frags    (8 B, Compact/Minimal only, mmap/pageable)
                //   h_t2_xbits (4 B, Compact/Minimal/Tiny, DMA/SpillBuffer)
                //
                // Deliberately NOT routable (device KERNELS read/write them
                // through USM-host pointers, so they must stay device-
                // accessible pinned memory — a disk file or mmap would
                // corrupt or crash the kernel access):
                //   h_keys_merged — the streaming-partition kernel writes it.
                //   h_t2_meta in Compact/Minimal — it ALIASES h_meta there, so
                //                   routing it would strand the alias. Tiny
                //                   gives it its own buffer, which is what
                //                   makes it routable in that tier (A2).
                //
                // A2 lifted the Tiny restriction on h_t2_meta / h_t2_xbits:
                // the T2-sort partition used to read both USM-host, and now
                // pulls each through its own SpillTileReader.
                bool const tier_tiny    = (tier == Tier::Tiny || tier == Tier::Pinned);
                bool const tier_streams = (tier != Tier::Plain);

                // The arithmetic itself lives in HostRamPolicy.cpp, as a pure
                // function, so host_spill_policy_test can reach every branch
                // without a GPU or a k=28-sized host. Everything below is
                // reporting and the two ways this can fail.
                //
                // `resident` tracks the UNSWAPPABLE class only — pinned plus
                // anonymous. That is the budget the knob enforces, because it
                // is the class that can get the process OOM-killed.
                //
                // The mmap class does NOT belong in it. h_frags spills to a
                // MAP_SHARED file: its dirty pages are written back and
                // evicted under memory pressure instead of killing anything,
                // so routing it genuinely removes those bytes from the
                // dangerous class. But they stay RESIDENT while there is no
                // pressure, so they still show up in RSS — which is the number
                // the user actually watches. Track them separately and report
                // both, or a user who measures 7.26 GiB against a modelled
                // "5.33 GiB" concludes the tool lied to them.
                HostRamSpillInputs pin;
                pin.host_required  = host_required;
                pin.cap_entries    = cap_entries;
                pin.budget         = B;
                pin.tier_tiny      = tier_tiny;
                pin.tier_streams   = tier_streams;
                // Minimal joins Tiny in gathering T2 sort in tiles, which
                // costs h_t2_xbits two extra passes over the temp dir that
                // Compact does not pay. Traffic estimate only.
                pin.tier_tiled_gather = (tier == Tier::Minimal) || tier_tiny;
                pin.pinned_slots   = num_pinned_slots;
                pin.forced_slots   = (forced_drain_slots != 0);
                pin.baseline_slots = GpuBufferPool::kNumPinnedBuffers;

                HostRamSpillPlan const sp = plan_host_ram_spill(pin);

                num_pinned_slots                         = sp.pinned_slots;
                StreamingPinnedScratch::SpillPlan const& plan = sp.tables;
                uint64_t const est           = sp.resident;
                uint64_t const reclaimable   = sp.reclaimable;
                uint64_t const drain_freed   = sp.drain_freed;
                uint64_t const floor_bytes   = sp.floor_bytes;

                if (!sp.meets_budget) {
                    // AUTO cannot reach the host's own free RAM even with
                    // everything routed. Do not throw here — the user never
                    // asked for a spill, so the honest error is the host-RAM
                    // one below, carrying the floor we could have reached.
                    if (spill_auto) {
                        do_spill         = false;
                        auto_unreachable = true;
                        auto_floor_bytes = floor_bytes;
                    } else {
                        throw std::runtime_error(
                            log_prefix + " host-RAM budget " +
                            std::to_string(to_gib(B)).substr(0, 5) +
                            " GiB is unreachable for tier " + tier_name(tier) +
                            " at k=" + std::to_string(pool_k) +
                            ": the lowest floor with every routable table "
                            "spilled is ~" +
                            std::to_string(to_gib(floor_bytes)).substr(0, 5) +
                            " GiB. Raise --max-host-ram, choose a higher "
                            "--tier, or route more buffers.");
                    }
                }

                if (do_spill) {
                    stream_scratch.spill = plan;
                    char budget_label[32];
                    if (B == 0) std::snprintf(budget_label, sizeof(budget_label), "min");
                    else        std::snprintf(budget_label, sizeof(budget_label),
                                              "%.2f GiB", to_gib(B));
                    std::string spilled;
                    if (plan.h_t1_meta)  spilled += " h_t1_meta";
                    if (plan.h_t3)       spilled += " h_t3";
                    if (plan.h_t2_meta)  spilled += " h_t2_meta";
                    if (plan.h_frags)    spilled += " h_frags";
                    if (plan.h_t2_xbits) spilled += " h_t2_xbits";
                    if (spilled.empty()) spilled = " (none)";
                    // Only mention RSS when the two figures actually differ —
                    // otherwise the extra clause is noise on every line.
                    char rss_note[128] = "";
                    if (reclaimable) {
                        std::snprintf(rss_note, sizeof(rss_note),
                            " (~%.2f GiB RSS; %.2f GiB of that is file-backed and "
                            "reclaimable under pressure)",
                            to_gib(est + reclaimable), to_gib(reclaimable));
                    }
                    std::fprintf(stderr,
                        "%s host-RAM budget %s (tier %s, k=%d): D2H drain %d->%d "
                        "slot%s (-%.2f GiB), spilling%s -> modelled unswappable "
                        "host peak ~%.2f GiB%s (routable floor ~%.2f GiB); "
                        "~%.1f GiB/plot of temp-dir traffic, ~%.1f GiB of it "
                        "writes\n",
                        log_prefix.c_str(), budget_label, tier_name(tier), pool_k,
                        GpuBufferPool::kNumPinnedBuffers, num_pinned_slots,
                        num_pinned_slots == 1 ? "" : "s", to_gib(drain_freed),
                        spilled.c_str(),
                        to_gib(est), rss_note, to_gib(floor_bytes),
                        to_gib(sp.traffic_written + sp.traffic_read),
                        to_gib(sp.traffic_written));

                    // AUTO was not asked for, so it must announce itself. A
                    // user who never passed a flag still needs to know why
                    // this run touches the disk and got slower.
                    if (spill_auto) {
                        std::fprintf(stderr,
                            "%s tier %s needs ~%.2f GiB of host RAM but only "
                            "~%.2f GiB is available: automatically spilling the "
                            "cold tables to '%s' to plot anyway. Expect roughly "
                            "10-30%% slower plots and ~%.1f GiB of temp-dir "
                            "traffic per plot (~%.1f GiB of it WRITES — size "
                            "the drive's endurance from that). Pass "
                            "--max-host-ram to control this, --temp-dir to "
                            "move it, or --no-auto-spill to be refused "
                            "instead.\n",
                            log_prefix.c_str(), tier_name(tier),
                            to_gib(host_required), to_gib(budget),
                            TempFile::resolve_dir("").c_str(),
                            to_gib(sp.traffic_written + sp.traffic_read),
                            to_gib(sp.traffic_written));
                    }
                }
            }

            // The host-RAM guard. Reached only when no spill is in play —
            // either none was needed, or the rescue above was unavailable.
            if (!do_spill
                && have_host && host_free < host_required + host_reserve) {
                std::string why;
                if (auto_blocked_ram_dir) {
                    why = " Automatic disk-offload could have run this, but the "
                          "temp dir '" + auto_ram_dir + "' is on a RAM-backed "
                          "filesystem (tmpfs), where spilling would consume the "
                          "very RAM that is short; point --temp-dir (or "
                          "XCHPLOT2_TEMP_DIR) at real disk to enable it.";
                } else if (auto_blocked_bad_dir) {
                    why = " Automatic disk-offload could have run this, but no "
                          "spill file can be created in the temp dir '" +
                          auto_ram_dir + "': " + auto_dir_problem +
                          ". Point --temp-dir (or XCHPLOT2_TEMP_DIR) at an "
                          "existing, writable directory on real disk to "
                          "enable it.";
                } else if (auto_unreachable) {
                    why = " Automatic disk-offload cannot close this gap: with "
                          "every routable table spilled and the D2H drain cut to "
                          "one slot, tier " + std::string(tier_name(tier)) +
                          " still needs ~" +
                          std::to_string(to_gib(auto_floor_bytes)).substr(0, 5) +
                          " GiB.";
                } else if (!opts.auto_host_ram_spill) {
                    why = " Automatic disk-offload could have run this; it is "
                          "off because --no-auto-spill / XCHPLOT2_NO_AUTO_SPILL "
                          "is set.";
                }
                throw std::runtime_error(
                    log_prefix + " tier " + tier_name(tier) + " needs ~" +
                    std::to_string(to_gib(host_required)).substr(0, 5) +
                    " GiB of HOST RAM at k=" + std::to_string(pool_k) +
                    " plus a " +
                    std::to_string(to_gib(host_reserve)).substr(0, 5) +
                    " GiB reserve; host reports " +
                    std::to_string(to_gib(host_free)).substr(0, 5) +
                    " GiB available of " +
                    std::to_string(to_gib(host_total)).substr(0, 5) +
                    " GiB total. This is host memory, not VRAM — and a LOWER "
                    "--tier costs MORE of it, not less: plain needs the least "
                    "(~" +
                    std::to_string(to_gib(streaming_plain_host_bytes(pool_k))).substr(0, 5) +
                    " GiB), tiny the most (~" +
                    std::to_string(to_gib(streaming_tiny_host_bytes(pool_k))).substr(0, 5) +
                    " GiB). The requirement is fixed for PoS2's k=28." + why +
                    " Otherwise close what else is holding RAM, or plot on a "
                    "host with more. (XCHPLOT2_HOST_RESERVE_MB tunes the "
                    "reserve.)");
            }

            // Two-phase match candidate scratch (see SyclBackend.hpp): grant
            // what is left after the tier's modelled peak and the margin.
            // Because the grant is free - peak - margin, the invariant
            // peak + scratch + margin <= free holds for any tier and any k —
            // the scratch can never be the thing that pushes a tier past the
            // VRAM budget it was picked for. When the remainder is too small
            // the match falls back to the single-kernel path: correct for any
            // input, allocates nothing, and merely slower.
            twophase_budget_bytes = (mem.free_bytes > required + margin)
                ? mem.free_bytes - required - margin
                : 0;
            stream_scratch.twophase_budget_bytes = twophase_budget_bytes;
            declared_base_bytes = required;

            stream_scratch.plain_mode  = (tier == Tier::Plain);
            // Pinned inherits Tiny's host-park behaviour as the
            // baseline; the streaming-partition algorithm change
            // ships in Phase 1.3c-ii and is then gated by an
            // additional pinned_mode flag on the scratch struct.
            stream_scratch.tiny_mode   = (tier == Tier::Tiny || tier == Tier::Pinned);
            stream_scratch.pinned_mode = (tier == Tier::Pinned);
            // Tiny / Pinned share minimal's tile counts (gather_tile_count > 1
            // is the trigger for the host-pinned T2 sort path that
            // tiny extends across T3 match).
            if (tier == Tier::Minimal || tier == Tier::Tiny || tier == Tier::Pinned) {
                stream_scratch.t2_tile_count     = 8;
                stream_scratch.gather_tile_count = 4;
            }

            // Record what was actually picked, so a two-pass caller can tell
            // whether its two passes are comparable. See WorkerTimeline.
            res.pipeline = tier_name(tier);

            if (!opts.quiet) {
                std::fprintf(stderr,
                    "%s streaming tier: %s "
                    "(%.2f GiB free, %.2f GiB peak, %.2f GiB plain floor)\n",
                    log_prefix.c_str(),
                    tier_name(tier),
                    to_gib(mem.free_bytes),
                    to_gib(required),
                    to_gib(plain_peak + margin));
            }
        }
        // Size the pinned buffers using the same cap formula as the pool.
        int const num_section_bits = (pool_k < 28) ? 2 : (pool_k - 26);
        int const extra_margin_bits = 8 - ((28 - pool_k) / 2);
        uint64_t const per_section =
            (1ULL << (pool_k - num_section_bits)) +
            (1ULL << (pool_k - extra_margin_bits));
        uint64_t const cap = per_section * (1ULL << num_section_bits);
        stream_pinned_cap = size_t(cap);

        // The host-RAM pre-flight used to live here, modelling which buffers
        // each tier allocates. It undercounted by 27% — it missed h_t1_mi,
        // h_t2_mi and others — which is the failure mode any such model has.
        // It now sits with the tier pick above (measured per-tier estimates),
        // and host_pinned_reserve_check() guards each allocation for real.

        // Only the slots this run will actually rotate through (B1: one under
        // a host-RAM budget). The rest stay null; every free path below is
        // null-guarded and the array is zero-initialised.
        bool any_fail = false;
        for (int s = 0; s < num_pinned_slots; ++s) {
            stream_pinned[s] = streaming_alloc_pinned_uint64(stream_pinned_cap);
            if (!stream_pinned[s]) { any_fail = true; break; }
        }
        if (any_fail) {
            for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
            }
            throw std::runtime_error(
                log_prefix + " streaming-fallback: pinned D2H buffer allocation failed");
        }

        // Stage 4f (compact tier only): amortise streaming-path
        // pinned-host scratch across all plots in the batch. Lifetime
        // analysis (see StreamingPinnedScratch doc) lets four shared
        // buffers cover all six internal park/staging roles. At k=28:
        // h_meta 2080 MB + h_keys_merged 1040 MB + h_t2_xbits 1040 MB
        // + h_t3 2080 MB = ~6.24 GB of pinned host, paid ONCE for the
        // whole batch.
        //
        // Plain tier does not park anything, so these pinned-host
        // scratch buffers are not needed.
        // Wire the per-batch pool so per-plot allocs not covered by
        // stream_scratch fields (h_t1_mi, h_t2_mi, h_keys_merged when
        // not pre-allocated below) amortise across the batch instead
        // of round-tripping malloc_host on every plot.
        stream_scratch.pool = &stream_pool;
        if (!stream_scratch.plain_mode) {
            // Host-RAM disk-offload: for a table the budget policy
            // (stream_scratch.spill, set above) or the legacy
            // XCHPLOT2_SPILL_T1META flag selected for spill, do NOT
            // pre-allocate its shared pinned buffer. Leaving it null
            // makes the streaming pipeline OWN the table and redirect it to
            // a TempFile via the shared SpillEngine instead of a full
            // pinned alloc.
            //   - h_meta backs h_t1_meta; spillable in tiny only (T2 meta
            //     uses the separate h_t2_meta below, so nulling is safe).
            //   - h_t3 is spillable in compact / minimal / tiny.
            bool const spill_t1meta = stream_scratch.tiny_mode &&
                (stream_scratch.spill.h_t1_meta ||
                 [] { char const* v = std::getenv("XCHPLOT2_SPILL_T1META");
                      return v && v[0] == '1'; }());
            stream_scratch.spill.h_t1_meta = spill_t1meta;  // reconcile with legacy env
            bool const spill_t3 = stream_scratch.spill.h_t3;
            // h_t2_xbits routes in every streaming tier since A2 taught the
            // T2-sort partition to pull its source streams off disk; h_t2_meta
            // routes in Tiny only (elsewhere it aliases h_meta).
            bool const spill_t2xbits = stream_scratch.spill.h_t2_xbits;
            bool const spill_t2meta  = stream_scratch.tiny_mode &&
                                       stream_scratch.spill.h_t2_meta;
            stream_scratch.spill.h_t2_meta = spill_t2meta;
            if (!spill_t1meta) {
                stream_scratch.h_meta    = streaming_alloc_pinned_uint64(stream_pinned_cap);
            }
            stream_scratch.h_keys_merged = streaming_alloc_pinned_uint32(stream_pinned_cap);
            if (!spill_t2xbits) {
                stream_scratch.h_t2_xbits = streaming_alloc_pinned_uint32(stream_pinned_cap);
            }
            if (!spill_t3) {
                stream_scratch.h_t3      = streaming_alloc_pinned_uint64(stream_pinned_cap);
            }
            // Tiny tier needs a separate h_t2_meta to avoid the
            // h_t1_meta/h_t2_meta buffer-reuse race in T2 match's
            // per-pass loop. Compact / minimal modes don't trip the
            // race (they read d_t1_meta_sorted on device, not h_t1_meta
            // on host) so leave h_t2_meta null and the streaming
            // pipeline reuses h_meta as before.
            if (stream_scratch.tiny_mode && !spill_t2meta) {
                stream_scratch.h_t2_meta = streaming_alloc_pinned_uint64(stream_pinned_cap);
            }
            if ((!spill_t1meta && !stream_scratch.h_meta) || !stream_scratch.h_keys_merged ||
                (!spill_t2xbits && !stream_scratch.h_t2_xbits) ||
                (!spill_t3 && !stream_scratch.h_t3) ||
                (stream_scratch.tiny_mode && !spill_t2meta && !stream_scratch.h_t2_meta))
            {
                if (stream_scratch.h_meta)        streaming_free_pinned_uint64(stream_scratch.h_meta);
                if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
                if (stream_scratch.h_t2_xbits)    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
                if (stream_scratch.h_t3)          streaming_free_pinned_uint64(stream_scratch.h_t3);
                if (stream_scratch.h_t2_meta)     streaming_free_pinned_uint64(stream_scratch.h_t2_meta);
                for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                    if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
                }
                throw std::runtime_error(
                    log_prefix + " streaming-fallback: pinned-host scratch allocation failed");
            }
        }
    }
    // RAII release of the streaming-fallback pinned buffers. These
    // MUST be freed on every exit path, not just the straight-line
    // return: the producer rethrows through `catch (...)` and the
    // consumer error path rethrows after join — and in multi-device
    // work-queue mode the peer workers keep running, so ~6-12 GB of
    // leaked pinned host memory would starve their allocations long
    // before process exit.
    struct StreamBuffersGuard {
        uint64_t** pinned;
        StreamingPinnedScratch* scratch;
        ~StreamBuffersGuard() {
            for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                if (pinned[s]) streaming_free_pinned_uint64(pinned[s]);
                pinned[s] = nullptr;
            }
            if (scratch->h_meta)        streaming_free_pinned_uint64(scratch->h_meta);
            if (scratch->h_keys_merged) streaming_free_pinned_uint32(scratch->h_keys_merged);
            if (scratch->h_t2_xbits)    streaming_free_pinned_uint32(scratch->h_t2_xbits);
            if (scratch->h_t2_meta)     streaming_free_pinned_uint64(scratch->h_t2_meta);
            if (scratch->h_t3)          streaming_free_pinned_uint64(scratch->h_t3);
            scratch->h_meta = nullptr;
            scratch->h_keys_merged = nullptr;
            scratch->h_t2_xbits = nullptr;
            scratch->h_t2_meta = nullptr;
            scratch->h_t3 = nullptr;
        }
    } stream_buffers_guard{stream_pinned, &stream_scratch};

    if (verbose && pool_ptr) {
        double gb = 1.0 / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr,
            "%s pool: storage=%.2f GB pair_a=%.2f GB pair_b=%.2f GB "
            "sort_scratch=%.2f GB pinned=2x%.2f GB "
            "(Xs scratch aliased in pair_b)\n",
            log_prefix.c_str(),
            pool_ptr->storage_bytes * gb,
            pool_ptr->pair_a_bytes  * gb,
            pool_ptr->pair_b_bytes  * gb,
            pool_ptr->sort_scratch_bytes * gb,
            pool_ptr->pinned_bytes       * gb);
    }

    // Depth = num_pinned_slots - 1. See Channel's comment block above.
    // Floored at 1: with a single slot (B1) the arithmetic depth is 0, and a
    // zero-capacity Channel can never accept a push — the producer would
    // block forever on the first plot. Depth 1 is safe because slot_gate,
    // not the channel depth, is what actually serialises slot reuse.
    Channel chan(static_cast<std::size_t>(
        std::max(1, num_pinned_slots - 1)));
    // Slot-reuse acknowledgment — see SlotGate's comment block. The
    // channel depth bounds queue growth; the gate is what actually
    // makes pinned-slot reuse safe when the consumer is slower than
    // the producer.
    SlotGate slot_gate;
    std::atomic<bool>     consumer_failed{false};
    std::atomic<size_t>   plots_done{0};
    std::exception_ptr    consumer_err;

    // Everything above — device bind, pool construction, pinned-host
    // allocation, the tier probe — is per-batch setup, not per-plot cost. With
    // a run epoch it stays in the timeline (so this worker's offsets are
    // comparable with its peers'), and work_start_seconds below records where
    // it ended so the bench can exclude it.
    auto t_start = run_epoch ? *run_epoch : std::chrono::steady_clock::now();
    res.work_start_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    live_work_start(live, worker_id, res.work_start_seconds);

    std::atomic<size_t> plots_failed_consumer{0};

    // Consumer: takes finished GpuPipelineResults and writes plot files.
    // Under continue_on_error, per-plot exceptions (e.g. ENOSPC for a
    // specific plot) are logged and the loop continues rather than
    // tearing down the batch. The .partial + rename in
    // write_plot_file_parallel guarantees failed writes leave nothing
    // behind at the destination.
    std::thread consumer([&] {
        try {
            WorkItem item;
            while (chan.pop(item)) {
                auto full_path = std::filesystem::path(item.entry.out_dir) / item.entry.out_name;
                try {
                    std::filesystem::create_directories(item.entry.out_dir);

                    std::vector<uint8_t> memo_bytes = item.entry.memo;
                    if (memo_bytes.empty()) memo_bytes.assign(32 + 48 + 32, 0);

                    // Fragments are borrowed from the pool's pinned slot; the
                    // producer is synchronised via the depth-1 channel so that
                    // slot won't be reused until we're done here.
                    std::uint64_t const plot_bytes = write_plot_file_parallel(
                        full_path.string(),
                        item.result.fragments(),
                        item.entry.plot_id.data(),
                        static_cast<uint8_t>(item.entry.k),
                        static_cast<uint8_t>(item.entry.strength),
                        item.entry.testnet ? uint8_t{1} : uint8_t{0},
                        static_cast<uint16_t>(item.entry.plot_index),
                        static_cast<uint8_t>(item.entry.meta_group),
                        std::span<uint8_t const>(memo_bytes.data(), memo_bytes.size()));

                    ++plots_done;
                    double const completion_offset = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start).count();
                    std::size_t const done_now = record_plot_completion(
                        res, plot_bytes, completion_offset, live, worker_id);
                    if (verbose) {
                        std::fprintf(stderr, "%s consumer wrote plot %zu: %s\n",
                                     log_prefix.c_str(),
                                     item.index, full_path.string().c_str());
                    }
                    if (opts.progress) {
                        emit_progress_line(
                            progress_prefix, opts, live, done_now,
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count());
                    }
                } catch (std::exception const& e) {
                    if (!opts.continue_on_error) throw;
                    ++plots_failed_consumer;
                    live_fail(live, worker_id);
                    std::fprintf(stderr,
                        "%s plot %zu FAILED (write %s): %s — continuing\n",
                        log_prefix.c_str(),
                        item.index, full_path.string().c_str(), e.what());
                }
                // Success or logged failure: either way we're done
                // reading this item's pinned slot — let the producer
                // reuse it.
                slot_gate.signal_consumed();
            }
        } catch (...) {
            consumer_err = std::current_exception();
            consumer_failed = true;
            // Unblock a producer waiting on a slot ack that will
            // never come.
            slot_gate.abort();
            // ENOSPC from the writer, or any other consumer-side I/O
            // failure, means peer workers will hit the same problem.
            // Cooperative-cancel them so they stop pulling new plots
            // off the queue instead of writing more partials that the
            // RAII guard then has to clean up.
            request_cancel();
        }
    });

    size_t producer_failed = 0;

    // Producer (this thread): drives the GPU pipeline, hands off to consumer.
    // local_count rotates this worker's own pinned-buffer slots (channel
    // depth = kNumPinnedBuffers); it must NOT use the global plot index
    // when shared_idx is in play, because peer workers also hold slots in
    // their own pools.
    try {
        std::size_t local_idx = 0;
        std::size_t local_count = 0;
        while (true) {
            if (consumer_failed) break;

            // Tail guard (default on): a slower card stops pulling once the faster
            // cards can drain what's left before it finishes one more plot, so it
            // never grabs a tail job a faster peer would clear sooner. A no-op for
            // the fastest worker (nobody strictly faster) and for single-worker
            // runs (no shared queue). See tail_guard_should_retire.
            if (tail_guard_enabled() && shared_idx) {
                std::size_t const claimed =
                    shared_idx->load(std::memory_order_relaxed);
                std::size_t const remaining =
                    claimed < entries.size() ? entries.size() - claimed : 0;
                std::string why;
                if (tail_guard_should_retire(live, worker_id, remaining, &why)) {
                    if (!opts.quiet)
                        std::fprintf(stderr, "%s %s\n",
                                     log_prefix.c_str(), why.c_str());
                    break;
                }
            }

            std::size_t const i = shared_idx
                ? shared_idx->fetch_add(1, std::memory_order_relaxed)
                : local_idx++;
            if (i >= entries.size()) break;

            if (cancel_requested()) {
                std::fprintf(stderr,
                    "%s cancel received — stopping before plot %zu\n",
                    log_prefix.c_str(), i);
                break;
            }

            // Claim after the cancel check: an entry we bail out before touching
            // was never in flight, and leaving a phantom there would have the ETA
            // wait on a plot nobody is making.
            live_claim(live, worker_id);

            if (opts.skip_existing) {
                auto out_path = std::filesystem::path(entries[i].out_dir)
                                / entries[i].out_name;
                if (looks_like_complete_plot(out_path)) {
                    if (verbose) {
                        std::fprintf(stderr,
                            "%s skipping plot %zu: %s (already exists)\n",
                            log_prefix.c_str(),
                            i, out_path.string().c_str());
                    }
                    ++res.plots_skipped;
                    std::size_t const done_now = live_skip(live, worker_id);
                    if (opts.progress) {
                        emit_progress_line(
                            progress_prefix, opts, live, done_now,
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count());
                    }
                    continue;
                }
            }

            auto t_plot = std::chrono::steady_clock::now();

            GpuPipelineConfig cfg;
            cfg.plot_id  = entries[i].plot_id;
            cfg.k        = entries[i].k;
            cfg.strength = entries[i].strength;
            cfg.testnet  = entries[i].testnet;
            cfg.profile  = false;
            // Pool path reads the two-phase grant from cfg; the streaming path
            // reads it from stream_scratch. Both are set above.
            cfg.twophase_budget_bytes = twophase_budget_bytes;

            WorkItem item;
            item.entry  = entries[i];
            item.index  = i;
            int const slot = static_cast<int>(
                local_count % std::size_t(num_pinned_slots));
            // Slot-reuse gate: the previous occupant of this slot was
            // push number (local_count - num_pinned_slots); wait until
            // the consumer has fully finished it (not merely popped it)
            // before the pipeline's D2H writes into the slot.
            if (local_count >= std::size_t(num_pinned_slots)) {
                std::size_t const need =
                    local_count - std::size_t(num_pinned_slots) + 1;
                if (!slot_gate.wait_consumed(need)) break;  // consumer died
            }
            try {
                if (pool_ptr) {
                    // Pool path: rotate pinned slot per plot. The channel
                    // bounds queue depth; the slot_gate wait above is what
                    // guarantees the consumer is done reading this slot.
                    item.result = run_gpu_pipeline(cfg, *pool_ptr, slot);
                } else {
                    // Streaming path with externally-owned pinned: same
                    // rotation + gate invariant.
                    item.result = run_gpu_pipeline_streaming(
                        cfg, stream_pinned[slot], stream_pinned_cap,
                        stream_scratch);
                }
            } catch (std::exception const& e) {
                if (!opts.continue_on_error) throw;
                ++producer_failed;
                live_fail(live, worker_id);
                std::fprintf(stderr,
                    "%s plot %zu FAILED (GPU): %s — continuing\n",
                    log_prefix.c_str(), i, e.what());
                continue;
            }

            if (verbose) {
                auto ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t_plot).count();
                std::fprintf(stderr,
                    "%s producer finished GPU for plot %zu in %.2f ms "
                    "(T1=%lu T2=%lu T3=%lu)\n",
                    log_prefix.c_str(), i, ms,
                    (unsigned long)item.result.t1_count,
                    (unsigned long)item.result.t2_count,
                    (unsigned long)item.result.t3_count);
            }

            if (!chan.push(std::move(item))) {
                // Channel closed under us (consumer died): this plot's
                // GPU work completed but will never be written. Count
                // it as a failure so the batch summary stays honest.
                ++producer_failed;
                live_fail(live, worker_id);
                break;
            }
            ++local_count;
        }
    } catch (...) {
        chan.close();
        consumer.join();
        throw;
    }

    chan.close();
    consumer.join();

    if (consumer_failed && consumer_err) std::rethrow_exception(consumer_err);

    // Pinned buffers + streaming scratch are freed by
    // stream_buffers_guard on every exit path (including the rethrows
    // above and the producer's catch).

    // VRAM watchdog: compare the peak the driver actually saw against what we
    // told the tier picker we would use. The peak models cannot see raw
    // sycl::malloc_device allocations, so this is the only check that catches
    // an unaccounted one — which is exactly the bug that made every 4-11 GB
    // NVIDIA card OOM on every tier but tiny.
    vram.stop();
    if (vram.available() && declared_base_bytes > 0) {
        uint64_t const held     = twophase_bytes_held();
        uint64_t const declared = declared_base_bytes + held + vram_safety_margin();
        uint64_t const peak     = vram.peak();
        auto to_mib = [](uint64_t b) { return b / double(1ULL << 20); };

        if (!opts.quiet) {
            std::fprintf(stderr,
                "%s %s: peak %.0f MiB of %.0f free "
                "(model %.0f + two-phase %.0f + margin %.0f = %.0f declared)\n",
                log_prefix.c_str(), mem_label,
                to_mib(peak), to_mib(vram.baseline()),
                to_mib(declared_base_bytes), to_mib(held),
                to_mib(vram_safety_margin()), to_mib(declared));
        }
        // The watchdog above only catches a model that under-declares — the OOM
        // direction. Over-declaring is silent and still wrong: the picker then
        // refuses the tier to cards that could run it. Pinned sat at an anchor
        // of 2200 MB against a true peak of 1128 for exactly this reason (the
        // number was a ratio against another anchor, never a measurement), and
        // nothing complained while it demanded 2.7 GB for a tier that fits in
        // 1.7 GB. Warn, never fail: an over-declared model costs throughput, not
        // correctness, and a legitimately idle-heavy phase could trip a strict
        // bound.
        if (peak > 0 && declared_base_bytes > peak * 3 / 2) {
            std::fprintf(stderr,
                "%s %s: WARNING — model declares %.0f MiB but the path only "
                "peaked at %.0f MiB. An over-declared model denies this path to "
                "cards that can run it. Re-derive the peak from this measured "
                "number, not from a ratio against another tier's anchor.\n",
                log_prefix.c_str(), mem_label,
                to_mib(declared_base_bytes), to_mib(peak));
        }

        // A breach of `declared` by less than the safety margin is runtime overhead
        // the peak model cannot see — backend scratch, module loads, allocation
        // rounding — sitting just past the margin, not an unaccounted
        // sycl::malloc_device. It is not worth a word on a normal run nor a thrown
        // bench. Only flag a breach past a SECOND margin's worth: that is the scale
        // of a real under-declaration, the kind that OOMs a card sized from the
        // model (the ~GB tier-accounting bug, not tens of MiB). The picker still
        // gates on `declared`; this slack is only the complaint's noise floor. bench
        // sets POS2GPU_ASSERT_VRAM, so this doubles as the regression test — with the
        // slack it fires on real drift, not jitter.
        uint64_t const slack = vram_safety_margin();
        if (peak > declared + slack) {
            std::fprintf(stderr,
                "%s %s: ERROR — peak %.0f MiB exceeds the %.0f MiB declared for this "
                "path by %.0f MiB, past the safety margin. A device allocation is "
                "unaccounted in the peak model; a card sized from it will OOM. See "
                "the two-phase budget notes in SyclBackend.hpp.\n",
                log_prefix.c_str(), mem_label, to_mib(peak), to_mib(declared),
                to_mib(peak - declared));
            if (char const* v = std::getenv("POS2GPU_ASSERT_VRAM");
                v && v[0] == '1')
            {
                throw std::runtime_error(
                    "VRAM assertion failed: peak " +
                    std::to_string(uint64_t(to_mib(peak))) + " MiB > declared " +
                    std::to_string(uint64_t(to_mib(declared))) + " MiB + margin");
            }
        }
    }

    res.plots_written = plots_done.load();
    res.plots_failed  = producer_failed + plots_failed_consumer.load();
    res.total_wall_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count();
    return res;
}

// Single-plot multi-GPU dispatch. Each plot in `entries` runs through
// the sharded pipeline using all of `device_ids` cooperatively.
// Phase 2.2 implements the Xs phase end-to-end; Phase 2.3+ wires
// up T1/T2/T3 matches and fragment serialize. The pipeline class
// throws a clear "Phase 2.3 not yet implemented" message when run()
// reaches the T1 match step until that phase lands.
//
// Caller invariants (enforced in run_batch before this is called):
//   - opts.shard_plot is true
//   - device_ids.size() > 1
//   - entries is non-empty
BatchResult run_batch_sharded(std::vector<BatchEntry> const& entries,
                              BatchOptions const& opts,
                              std::vector<int> const& device_ids)
{
    BatchResult res;
    auto const t_start = std::chrono::steady_clock::now();

    // Per-shard SYCL queues — one sycl::queue per target device.
    //
    // Earlier this loop went through the sycl_backend::queue() factory
    // which is a thread_local unique_ptr. From a single thread (which
    // is how the pipeline drives all shards today), the first call
    // wins: every shard pointer aliased the same queue, all bound to
    // whichever device the first call landed on. On the dev box that
    // looked fine because there was only one GPU; on real multi-GPU
    // hosts every shard's work piled onto device 0, OOMing the T2
    // phase at k>=28 even though the second card was idle.
    //
    // Construct the queues directly here, one owned by this function
    // (storage keeps them alive across pipeline.run() calls). Each
    // queue runs the SyclBackend selftest before pipeline use so a
    // miscompiled kernel on any of the shards surfaces here, not deep
    // in the streaming pipeline.
    auto const& gpu_devs = sycl_backend::usable_gpu_devices();
    std::vector<std::unique_ptr<sycl::queue>> shard_queue_storage;
    shard_queue_storage.reserve(device_ids.size());
    std::vector<sycl::queue*> shard_queues;
    shard_queues.reserve(device_ids.size());
    for (int dev_id : device_ids) {
        if (is_cpu_device(dev_id)) {
            throw std::runtime_error(
                "run_batch_sharded: --cpu in --shard-plot is not "
                "supported (CPU device can't host the per-shard SYCL "
                "queues used by MultiGpuPlotPipeline).");
        }
        if (dev_id < 0 || dev_id >= static_cast<int>(gpu_devs.size())) {
            throw std::runtime_error(
                "run_batch_sharded: device id "
                + std::to_string(dev_id) + " out of range (found "
                + std::to_string(gpu_devs.size())
                + " usable GPU device(s))");
        }
        auto q = std::make_unique<sycl::queue>(
            gpu_devs[static_cast<std::size_t>(dev_id)],
            sycl_backend::async_error_handler);
        sycl_backend::validate_kernel_dispatch(*q);
        shard_queues.push_back(q.get());
        shard_queue_storage.push_back(std::move(q));
    }
    // Per-shard buffer pools: persist across plots in this batch so
    // the largest replicated allocations (full Xs, full T1/T2/T3
    // streams) avoid the malloc/free round-trip per plot. Pools live
    // on the stack here; their dtor frees device memory at function
    // return.
    std::vector<ShardBufferPool> shard_pools(device_ids.size());
    for (std::size_t k = 0; k < device_ids.size(); ++k) {
        shard_pools[k].attach(shard_queues[k]);
    }

    std::vector<MultiGpuShardContext> shard_ctx;
    shard_ctx.reserve(device_ids.size());
    for (std::size_t k = 0; k < device_ids.size(); ++k) {
        MultiGpuShardContext c{};
        c.queue     = shard_queues[k];
        c.device_id = device_ids[k];
        c.pool      = &shard_pools[k];
        shard_ctx.push_back(c);
    }

    // Producer/consumer overlap between the GPU pipeline and the plot
    // file writer. The producer (this thread) drives MultiGpuPlotPipeline
    // through every phase including the fragment D2H, then hands the
    // pipeline + target path off to the consumer thread, which calls
    // write_plot_file_parallel. While the consumer is FSE-encoding +
    // writing plot N, the producer is already running the GPU pipeline
    // for plot N+1 on the same shards. The pipeline pointer is moved
    // into the WriteJob so the pinned-host fragments buffer (h_fragments_)
    // stays alive until write_plot_file_parallel returns; the pipeline's
    // destructor then frees it on the consumer thread, off the producer's
    // critical path.
    //
    // Channel depth=1: the queue holds at most one waiting job, plus
    // one in-flight at the consumer. Producer blocks on push when the
    // queue is full so we never accumulate finished plots in pinned
    // host memory. With sharded GPU compute ~10 s/plot at k=28 and
    // file write ~1 s/plot, the producer is the slow side and never
    // sees the back-pressure path in practice.
    struct WriteJob {
        std::unique_ptr<MultiGpuPlotPipeline> pipeline;
        std::filesystem::path                 full_path;
        BatchEntry                            entry;
        std::vector<std::uint8_t>             memo_bytes;
    };

    std::mutex                   q_mu;
    std::condition_variable      cv_not_empty;
    std::condition_variable      cv_not_full;
    std::queue<WriteJob>         q;
    bool                         producer_done   = false;
    std::atomic<bool>            consumer_failed{false};
    std::exception_ptr           consumer_err;
    std::atomic<std::size_t>     plots_written_consumer{0};
    std::atomic<std::size_t>     plots_failed_consumer{0};
    // The team plots one at a time, so it publishes as a single worker. No
    // live_claim(): with one worker an in-flight plot and a queued one land at
    // the same instant (base + k*s), so the claim would change nothing.
    BatchProgress                live(1, entries.size());

    std::thread consumer([&] {
        for (;;) {
            WriteJob job;
            {
                std::unique_lock<std::mutex> lock(q_mu);
                cv_not_empty.wait(lock, [&] {
                    return !q.empty() || producer_done;
                });
                if (q.empty()) return;  // drained + producer_done
                job = std::move(q.front());
                q.pop();
                cv_not_full.notify_one();
            }
            try {
                std::uint64_t const plot_bytes = write_plot_file_parallel(
                    job.full_path.string(),
                    job.pipeline->fragments(),
                    job.entry.plot_id.data(),
                    static_cast<std::uint8_t>(job.entry.k),
                    static_cast<std::uint8_t>(job.entry.strength),
                    job.entry.testnet ? std::uint8_t{1} : std::uint8_t{0},
                    static_cast<std::uint16_t>(job.entry.plot_index),
                    static_cast<std::uint8_t>(job.entry.meta_group),
                    std::span<std::uint8_t const>(
                        job.memo_bytes.data(), job.memo_bytes.size()));
                ++plots_written_consumer;
                double const completion_offset = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                std::size_t const done_now = record_plot_completion(
                    res, plot_bytes, completion_offset, live, 0);
                if (opts.progress) {
                    emit_progress_line(
                        "[shard-plot]", opts, live, done_now,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_start).count());
                }
            } catch (std::exception const& e) {
                live_fail(live, 0);
                std::fprintf(stderr,
                    "[shard-plot] FAILED writing '%s': %s\n",
                    job.entry.out_name.c_str(), e.what());
                ++plots_failed_consumer;
                if (!opts.continue_on_error) {
                    consumer_err = std::current_exception();
                    consumer_failed.store(true, std::memory_order_release);
                    // Wake a producer that may be blocked on cv_not_full
                    // so it can observe consumer_failed and exit cleanly.
                    cv_not_full.notify_all();
                    return;
                }
            }
            // job (incl. pipeline) destructs here; h_fragments_ is freed
            // off the producer's critical path.
        }
    });

    std::size_t plots_failed_producer = 0;
    bool        early_stop            = false;
    // Everything above — per-shard SYCL queues, shard pools, shard contexts — is
    // per-batch setup, not per-plot cost. Record where it ended: under --warmup 0
    // this is the bench's epoch, and leaving it at 0 amortised the whole
    // multi-GPU setup across every plot and quietly understated the rig.
    res.work_start_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    live_work_start(live, 0, res.work_start_seconds);

    for (BatchEntry const& entry : entries) {
        if (consumer_failed.load(std::memory_order_acquire)) {
            early_stop = true;
            break;
        }
        try {
            // Resolve target path before running so an out_dir failure
            // surfaces before the (~minutes) plot work.
            auto full_path = std::filesystem::path(entry.out_dir) / entry.out_name;
            std::filesystem::create_directories(entry.out_dir);

            auto pipeline = std::make_unique<MultiGpuPlotPipeline>(
                entry, opts, shard_ctx);
            pipeline->run();

            WriteJob job;
            job.pipeline   = std::move(pipeline);
            job.full_path  = std::move(full_path);
            job.entry      = entry;
            job.memo_bytes = entry.memo;
            if (job.memo_bytes.empty()) job.memo_bytes.assign(32 + 48 + 32, 0);

            {
                std::unique_lock<std::mutex> lock(q_mu);
                cv_not_full.wait(lock, [&] {
                    return q.size() < 1
                        || consumer_failed.load(std::memory_order_acquire);
                });
                if (consumer_failed.load(std::memory_order_acquire)) {
                    early_stop = true;
                    break;
                }
                q.push(std::move(job));
                cv_not_empty.notify_one();
            }
        } catch (std::exception const& e) {
            std::fprintf(stderr,
                "[shard-plot] FAILED for plot '%s': %s\n",
                entry.out_name.c_str(), e.what());
            ++plots_failed_producer;
            if (!opts.continue_on_error) {
                early_stop = true;
                break;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(q_mu);
        producer_done = true;
        cv_not_empty.notify_all();
    }
    consumer.join();

    if (consumer_failed.load(std::memory_order_acquire) && consumer_err) {
        std::rethrow_exception(consumer_err);
    }

    res.plots_written = plots_written_consumer.load();
    res.plots_failed  = plots_failed_producer + plots_failed_consumer.load();
    (void)early_stop;
    res.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return res;
}

BatchResult run_batch_pipeline_plot(std::vector<BatchEntry> const& entries,
                                    BatchOptions const& opts,
                                    std::vector<int> const& device_ids)
{
    BatchResult res{};
    if (device_ids.size() != 2 && device_ids.size() != 3) {
        throw std::runtime_error(
            "run_batch_pipeline_plot: --pipeline-plot requires 2 or 3 device "
            "ids (got " + std::to_string(device_ids.size()) + ")");
    }
    for (int id : device_ids) {
        if (id < 0) throw std::runtime_error(
            "run_batch_pipeline_plot: device ids must be non-negative");
    }
    if (!opts.pipeline_tiers.empty() &&
        opts.pipeline_tiers.size() != device_ids.size())
    {
        throw std::runtime_error(
            "run_batch_pipeline_plot: pipeline_tiers must be empty or match "
            "device_ids size (got " +
            std::to_string(opts.pipeline_tiers.size()) + " tiers vs " +
            std::to_string(device_ids.size()) + " devices)");
    }

    // VRAM-aware assignment: heaviest stage on largest-VRAM card.
    // Generalised for N=2 (T2-sort split) and N=3 (T1-sort + T2-sort
    // split). The heaviness order is encoded in select_pipeline_devices;
    // see stage_heaviness_order() in MultiGpuPipelineParallel.cpp.
    auto const assign      = select_pipeline_devices(device_ids);
    auto const& staged_devices    = assign.dev_ids;
    auto const& stage_vram_bytes  = assign.dev_vram_bytes;
    bool const reordered          = assign.reordered;

    if (opts.verbose) {
        std::fprintf(stderr, "[pipeline-plot] %zu plots:", entries.size());
        for (std::size_t s = 0; s < staged_devices.size(); ++s) {
            double const gb = static_cast<double>(stage_vram_bytes[s]) / 1.0e9;
            std::fprintf(stderr, " stage%zu on dev %d (%.1f GB)%s",
                         s + 1, staged_devices[s], gb,
                         (s + 1 == staged_devices.size()) ? "" : " →");
        }
        std::fprintf(stderr, "%s (depth=2)\n",
                     reordered ? " [reordered by VRAM]" : "");
    }

    // Phase 2.2g: per-stage tier auto-pick. When the user doesn't
    // pin tiers explicitly, default each stage to Minimal if its
    // device has the VRAM (with safety headroom), else Tiny. Big
    // cards get the faster tier — fewer PCIe round-trips — without
    // forcing the user to spell it out.
    int const k_for_tiers = entries[0].k;
    std::vector<PipelineStageTier> resolved_tiers = opts.pipeline_tiers;
    if (resolved_tiers.empty()) {
        std::uint64_t const minimal_peak =
            streaming_minimal_peak_bytes(k_for_tiers);
        // Leave ~25% headroom on top of the predicted peak so we
        // don't slam right up to the cap — accounts for pinned-host
        // allocations and any sort-scratch oscillation.
        std::uint64_t const minimal_threshold =
            minimal_peak + (minimal_peak / 4);
        resolved_tiers.reserve(staged_devices.size());
        for (std::size_t s = 0; s < staged_devices.size(); ++s) {
            bool const fits_minimal =
                stage_vram_bytes[s] >= minimal_threshold;
            resolved_tiers.push_back(fits_minimal
                ? PipelineStageTier::Minimal
                : PipelineStageTier::Tiny);
        }
        if (opts.verbose) {
            std::fprintf(stderr,
                "[pipeline-plot] auto-tier (k=%d, minimal_peak=%.1f GB +25%% headroom):",
                k_for_tiers,
                static_cast<double>(minimal_peak) / 1.0e9);
            for (std::size_t s = 0; s < resolved_tiers.size(); ++s) {
                char const* tname =
                    (resolved_tiers[s] == PipelineStageTier::Minimal)
                        ? "minimal" : "tiny";
                std::fprintf(stderr, " stage%zu=%s",
                             s + 1, tname);
            }
            std::fprintf(stderr, "\n");
        }
    }

    // Convert BatchEntry sequence to GpuPipelineConfig sequence.
    std::vector<GpuPipelineConfig> cfgs;
    cfgs.reserve(entries.size());
    for (auto const& e : entries) {
        GpuPipelineConfig cfg;
        cfg.k        = e.k;
        cfg.strength = e.strength;
        cfg.testnet  = e.testnet;
        cfg.plot_id  = e.plot_id;
        cfgs.push_back(cfg);
    }

    // Phase 2.2h: pipeline_depth from BatchOptions (default 2,
    // override via --pipeline-depth). Higher depth amortises fill/
    // drain over more in-flight plots; trade-off is host-pinned
    // memory per slot per boundary (~6 GB at k=28 for the T2-sort
    // boundary, less for the T1-sort boundary).
    int const depth = (opts.pipeline_depth > 0) ? opts.pipeline_depth : 2;

    // Phase 2.1f: concurrent plot writer. The orchestrator fires
    // on_plot_complete from the final-stage worker thread as soon as
    // each plot's fragments are ready. We push (idx, result) onto a
    // queue; a writer thread drains the queue and runs FSE+disk
    // write in parallel with subsequent plots' GPU pipelines.
    //
    // Steady-state effect: per-plot wall = max(GPU pipeline, FSE+disk)
    // instead of sum(GPU, FSE+disk). For k=28 the post-batch sequential
    // writes were ~8 s/plot; overlapping cuts the 10-plot total by a
    // similar amount.
    struct WriteJob {
        int                                idx;
        PipelineParallelSplitResult        result;
    };
    std::mutex              q_mtx;
    std::condition_variable q_cv;
    std::queue<WriteJob>    q;
    std::atomic<bool>       q_done{false};
    std::atomic<bool>       writer_abort{false};

    std::atomic<std::size_t> plots_written_ct{0};
    std::atomic<std::size_t> plots_failed_ct{0};
    // The stages plot one at a time, so the rig publishes as a single worker —
    // see the note in run_batch_sharded for why no live_claim() is needed.
    BatchProgress live(1, entries.size());
    auto const t_start = std::chrono::steady_clock::now();

    auto writer_fn = [&]() {
        while (true) {
            WriteJob job;
            {
                std::unique_lock<std::mutex> lk(q_mtx);
                q_cv.wait(lk, [&] {
                    return !q.empty() || q_done.load();
                });
                if (q.empty()) return;  // q_done && drained → exit
                job = std::move(q.front());
                q.pop();
            }
            // Drop pending work if a prior write failed and the
            // batch is in "stop on first failure" mode.
            if (writer_abort.load()) continue;

            auto const& entry = entries[static_cast<std::size_t>(job.idx)];
            try {
                auto full_path = std::filesystem::path(entry.out_dir)
                                 / entry.out_name;
                std::filesystem::create_directories(entry.out_dir);

                std::vector<uint8_t> memo_bytes = entry.memo;
                if (memo_bytes.empty()) memo_bytes.assign(32 + 48 + 32, 0);

                auto frags = job.result.fragments();
                std::uint64_t const plot_bytes = write_plot_file_parallel(
                    full_path.string(),
                    frags,
                    entry.plot_id.data(),
                    static_cast<uint8_t>(entry.k),
                    static_cast<uint8_t>(entry.strength),
                    entry.testnet ? uint8_t{1} : uint8_t{0},
                    static_cast<uint16_t>(entry.plot_index),
                    static_cast<uint8_t>(entry.meta_group),
                    std::span<uint8_t const>(memo_bytes.data(),
                                             memo_bytes.size()),
                    /*thread_count=*/0);
                ++plots_written_ct;
                double const completion_offset = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                std::size_t const done_now = record_plot_completion(
                    res, plot_bytes, completion_offset, live, 0);
                if (opts.progress) {
                    emit_progress_line(
                        "[pipeline-plot]", opts, live, done_now,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_start).count());
                }
                if (opts.verbose) {
                    std::fprintf(stderr,
                        "[pipeline-plot] wrote %s (%llu fragments)\n",
                        full_path.c_str(),
                        static_cast<unsigned long long>(frags.size()));
                }
            } catch (std::exception const& e) {
                std::fprintf(stderr,
                    "[pipeline-plot] FAILED for plot '%s': %s\n",
                    entry.out_name.c_str(), e.what());
                ++plots_failed_ct;
                live_fail(live, 0);
                if (!opts.continue_on_error) writer_abort.store(true);
            }
        }
    };

    std::thread writer(writer_fn);

    auto on_plot_complete =
        [&](int cfg_idx, PipelineParallelSplitResult result) {
            {
                std::lock_guard<std::mutex> lk(q_mtx);
                q.push(WriteJob{cfg_idx, std::move(result)});
            }
            q_cv.notify_one();
        };

    // The measurement epoch is the instant the rig can actually plot — after
    // every stage has bound its device and built its pool. Those are constructed
    // on the stage threads INSIDE run_pipeline_parallel_batch, so this used to
    // be stamped before the call and silently amortised the rig's whole setup
    // across every plot under --warmup 0. The orchestrator now fires on_ready
    // from the last stage to come up, which is what WorkerTimeline's contract
    // ("after its device init and pool construction") always said this was.
    auto on_ready = [&]() {
        res.work_start_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        live_work_start(live, 0, res.work_start_seconds);
    };

    std::vector<PipelineStageStats> stage_stats;

    try {
        run_pipeline_parallel_batch(
            cfgs, staged_devices, depth, resolved_tiers, on_plot_complete,
            on_ready, &stage_stats);
    } catch (...) {
        // Make sure the writer thread can exit even if the
        // orchestrator throws — otherwise we leak the std::thread.
        q_done.store(true);
        q_cv.notify_all();
        writer.join();
        throw;
    }

    q_done.store(true);
    q_cv.notify_all();
    writer.join();

    res.plots_written = plots_written_ct.load();
    res.plots_failed  = plots_failed_ct.load();
    res.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    // Name the bottleneck. An aggregate s/plot tells a --pipeline-plot user
    // nothing they can act on: their only lever is WHICH phase runs on WHICH
    // device, and a pipeline retires plots at the rate of its slowest stage. So
    // report where each stage's wall actually went. The bottleneck is the stage
    // that is busy nearly all the time and waiting on nobody; every other stage
    // is either backpressured by it (blocked, sitting above it) or starved by it
    // (waiting, sitting below it).
    if (!opts.quiet && !stage_stats.empty() && res.total_wall_seconds > 0.0) {
        double busiest = 0.0;
        for (auto const& s : stage_stats) busiest = std::max(busiest, s.busy_seconds);

        std::fprintf(stderr,
            "[pipeline-plot] stage walls over %.1fs (the rig runs at the speed "
            "of its slowest stage):\n", res.total_wall_seconds);
        for (auto const& s : stage_stats) {
            bool const is_bottleneck = (s.busy_seconds >= busiest - 1e-9);
            char const* note =
                is_bottleneck                              ? "  <-- BOTTLENECK" :
                (s.blocked_seconds > s.starved_seconds)    ? "  (backpressured — the stage below it is slower)" :
                (s.starved_seconds > 0.0)                  ? "  (starved — the stage above it cannot feed it)" :
                                                             "";
            std::fprintf(stderr,
                "[pipeline-plot]   stage %d (gpu%d): busy %.1fs (%.0f%%), "
                "starved %.1fs, blocked %.1fs, %zu plots%s\n",
                s.stage, s.device_id,
                s.busy_seconds,
                100.0 * s.busy_seconds / res.total_wall_seconds,
                s.starved_seconds, s.blocked_seconds, s.plots, note);
        }
        if (busiest > 0.0) {
            std::fprintf(stderr,
                "[pipeline-plot] move work off the bottleneck stage, or give it "
                "the faster device — nothing else changes the rig's rate.\n");
        }
    }
    return res;
}

} // namespace

// Phase 2.4 auto-strategy picker. Decides at runtime which multi-GPU
// strategy fits the rig + k. Pure function — testable via the
// injected `vram_for_device` lookup. Heuristic:
//   - N <= 1                                 → WorkQueue (only option)
//   - smallest dev's VRAM < tiny streaming   → PipelinePlot (work-queue
//                                              can't fit on smallest card;
//                                              pipeline gives it the
//                                              lightest stage)
//   - else                                   → WorkQueue (proven
//                                              throughput winner on
//                                              equal-VRAM PCIe rigs;
//                                              pool path on 24 GB cards
//                                              wins by ~40× over
//                                              pipeline-plot at k=28)
// shard-plot is never auto-selected — it's a niche opt-in with worse
// PCIe-only throughput than work-queue (see README).
BatchStrategy select_strategy(
    StrategyPickInputs const&                inputs,
    std::function<std::uint64_t(int)> const& vram_for_device,
    std::string*                             reason_out)
{
    std::size_t const N = inputs.device_ids.size();
    if (N <= 1) {
        if (reason_out) *reason_out =
            "N=" + std::to_string(N) +
            " device(s); work-queue is the only multi-GPU strategy";
        return BatchStrategy::WorkQueue;
    }

    std::uint64_t const tiny_peak  = streaming_tiny_peak_bytes(inputs.k);
    std::uint64_t       min_vram   = UINT64_MAX;
    int                 min_dev    = -1;
    for (int id : inputs.device_ids) {
        if (id < 0) continue;  // CPU worker — skip in VRAM heuristic
        std::uint64_t const v = vram_for_device(id);
        if (v < min_vram) { min_vram = v; min_dev = id; }
    }

    // If we couldn't sample any GPU (all-CPU device list): WorkQueue.
    if (min_dev < 0) {
        if (reason_out) *reason_out =
            "no GPU in device list; work-queue handles CPU";
        return BatchStrategy::WorkQueue;
    }

    if (min_vram < tiny_peak) {
        if (reason_out) {
            *reason_out =
                "smallest GPU (dev " + std::to_string(min_dev) + ", " +
                std::to_string(min_vram / (1ULL << 30)) + " GB) below tiny "
                "streaming peak (" +
                std::to_string(tiny_peak / (1ULL << 30)) +
                " GB) at k=" + std::to_string(inputs.k) +
                "; pipeline-plot puts the small card on the lightest stage";
        }
        return BatchStrategy::PipelinePlot;
    }

    if (reason_out) {
        *reason_out =
            "all " + std::to_string(N) + " GPU(s) fit at k=" +
            std::to_string(inputs.k) + " (smallest = " +
            std::to_string(min_vram / (1ULL << 30)) +
            " GB ≥ tiny peak); work-queue is the throughput winner";
    }
    return BatchStrategy::WorkQueue;
}

BatchStrategy select_strategy(StrategyPickInputs const& inputs,
                              std::string*              reason_out)
{
    auto vram = [](int id) -> std::uint64_t {
        auto const& devs = sycl_backend::usable_gpu_devices();
        if (id < 0 || static_cast<std::size_t>(id) >= devs.size()) return 0;
        return devs[id].get_info<sycl::info::device::global_mem_size>();
    };
    return select_strategy(inputs, vram, reason_out);
}

// ---------------------------------------------------------------------------
// The --cpu-workers RAM gate.
//
// N CPU workers means N concurrent pos2-chip Plotters, and they share nothing:
// no tier shrinks them (the CPU branch of run_batch_slice returns long before
// the tier machinery), no pool bounds them. So the only defence against asking
// for more than the host can hold is to not start them — and getting it wrong
// does not cost throughput, it costs the whole batch, because the OOM killer
// takes the GPU workers' in-flight plots down with the CPU's.
//
// Both curves below are measured, not modelled: VmHWM out of /proc, the
// kernel's own high-water mark, polled to process exit on a 32-thread 5950X +
// RTX 4090. tmp/ram_model.sh reproduces them.
// ---------------------------------------------------------------------------

namespace {

// Host RSS one GPU worker adds — its pinned pool, fragment buffers, FSE
// scratch, CUDA context and the binary itself. Not device VRAM: this is the
// host memory a GPU worker takes AWAY from the CPU workers.
//
//   k=22    345 668 kB      (model says 350 396 — +1.4%)
//   k=26  1 583 524 kB
//   k=28  5 529 348 kB      (5.27 GiB)
//
// Fits A·2^k + B to within 1.4% across all three, with A = 20.07 B/entry and
// B = 262 MB (context + binary + pool, none of which scale with k).
//
// Measured on a 24 GB card, which auto-picks the largest streaming tier and so
// the largest host pinned pool. A smaller card picks a smaller tier and needs
// less host memory than this — over-estimating the reserve costs at most one
// CPU worker, and that is the direction to err in.
std::uint64_t gpu_worker_host_peak_bytes(int k)
{
    constexpr double kBytesPerEntry = 20.07;
    constexpr double kFixedBytes    = 262.0 * 1024.0 * 1024.0;
    double const entries = static_cast<double>(std::uint64_t{1} << k);
    return static_cast<std::uint64_t>(kBytesPerEntry * entries + kFixedBytes);
}

// Free host RAM, probed ONCE per process.
//
// Cached deliberately. run_batch and the bench's own batch_worker_count() both
// resolve the device list, and they must agree on how many workers exist — the
// bench sizes its plot queue off that count. A live probe would not agree with
// itself: the CPU workers' own 12 GiB apiece is subtracted from MemAvailable
// the moment they start, so asking again mid-run answers a different question
// than the one the gate is for ("how many can I start?", not "how much is left
// now?").
std::uint64_t host_free_bytes_once()
{
    static std::uint64_t const cached = []() -> std::uint64_t {
        std::size_t free_b = 0;
        std::size_t total_b = 0;
        if (!device_memory_probe(kCpuDeviceId, free_b, total_b)) return 0;
        return static_cast<std::uint64_t>(free_b);
    }();
    return cached;
}

// Slack between "the kernel says this much is available" and "starting another
// 12 GiB allocation right now is a good idea". MemAvailable already excludes
// what the kernel wants to keep and already counts reclaimable page cache, so
// this is not a second guess at the same thing — it is headroom for everything
// on the box that is not us.
constexpr std::uint64_t kHostSlackBytes = 1ULL << 30;  // 1 GiB

// Host RAM the user wants left alone — XCHPLOT2_CPU_RESERVE_MB.
//
// The gate's job is to stop the CPU workers OOMing the box. But "the box has
// 73 GiB free" and "you may take 73 GiB" are different claims: plenty of people
// plot on the machine they also work on, and would rather keep 16 GB for the
// thing they are actually doing than discover, an hour in, that their editor got
// swapped out. This is how they say so, and it comes off the top of the budget
// for both the start-of-batch count and the live gate.
std::uint64_t host_reserve_bytes()
{
    if (char const* v = std::getenv("XCHPLOT2_CPU_RESERVE_MB"); v && v[0]) {
        long const mb = std::atol(v);
        if (mb > 0) return static_cast<std::uint64_t>(mb) << 20;
    }
    return 0;
}

// How many bytes the CPU workers may collectively hold, at batch start.
//
// One expression, two users: the start-of-batch gate divides it by the per-worker
// peak to pick N, and CpuMemoryGate carries it as the baseline it tracks external
// pressure against. If they disagreed about what "budget" means, the live gate
// would spend the batch either denying workers the static gate had already
// approved, or approving ones it had not.
std::uint64_t cpu_budget_bytes(int k, std::size_t gpu_count)
{
    std::uint64_t const free_now = host_free_bytes_once();
    std::uint64_t const reserve =
        kHostSlackBytes + host_reserve_bytes() +
        static_cast<std::uint64_t>(gpu_count) * gpu_worker_host_peak_bytes(k);
    return free_now > reserve ? free_now - reserve : 0;
}


// The auto-default CPU worker count: the knee of the throughput curve.
//
// pos2-chip's plotter is memory-latency-bound, so concurrent plots interleave
// each other's stalls — but each already fans out to every core, so the gain
// plateaus fast (k=28: N=2 +19%, N=4 +25%, flat after). Past the knee you only
// oversubscribe, and at small k the RAM cap is no help at all (k=22 would permit
// ~500 workers, each spawning 32 threads). So auto stops at the knee.
//
// Capped at half the cores so tiny hosts don't over-spawn; XCHPLOT2_CPU_AUTO_WORKERS
// overrides for tuning.
//
// `node_cores` is the core count of the NODE this count is for, not the host's.
// The knee is a property of one node's memory system — it is where extra plots
// stop hiding each other's stalls — so a 2-socket box wants the knee twice, not
// a knee sized by twice the cores.
//
// CAUTION: the ~4 was measured on a single-socket host, where a node IS the
// machine. On a multi-socket host a node-pinned worker still spawns a
// WHOLE-HOST-sized thread fan-out onto its node's cores (see NumaTopology.hpp),
// so the real knee there is probably lower, and this is a starting point for
// measurement rather than a number anyone has stood behind.
//
// `gpu_peer` collapses the knee to 1, because the curve above is the wrong one
// to be standing on beside a GPU. It was measured under `--devices cpu`, so it
// can only see what extra workers ADD (they interleave each other's memory
// stalls: +36% at N=4). What it cannot see is what they COST, which is the GPU
// worker's FSE consumer: that runs on the host, its work per plot is fixed by k
// while the GPU's compute time is not, and a fast card needs it to turn a plot
// around every couple of seconds. A CPU worker's whole-host thread fan-out
// starves exactly that, and the peer it starves is worth ~24 of it.
//
// Measured 2026-07-17, RTX 4090 + 5950X at k=28:
//
//   workers   GPU rate      aggregate    vs GPU alone
//   0         2.56 s/plot   2.56         —
//   1         2.68          2.57         a wash (-0.35%)
//   4 (knee)  4.23          4.23         2.39x SLOWER whole-run
//
// At the CPU-only knee of 4 the batch took 336.4 s against the 140.8 s the GPU
// would have needed alone, and the GPU's own rate fell by 65%. One worker is a
// wash here and still buys 74% of the CPU's own N=4 throughput, so it is the
// risk-adjusted pick: near-free on a card this fast, and most of the prize on a
// slow one. The two effects that make 4 wrong both ease as the GPU slows (a
// slower card's consumer has a lower duty cycle, and the CPU's share of the
// total rises), so a slow-GPU host may well want more than 1 — but nobody has
// measured one. Until somebody does, 1 is the number that cannot lose badly.
// XCHPLOT2_CPU_AUTO_WORKERS overrides for exactly that measurement.
int cpu_worker_auto_count(int node_cores, bool gpu_peer)
{
    if (char const* v = std::getenv("XCHPLOT2_CPU_AUTO_WORKERS"); v && v[0]) {
        int const n = std::atoi(v);
        if (n >= 1 && n <= 64) return n;
    }
    int const cores = node_cores > 0 ? node_cores : 8;
    int const knee  = std::min(4, std::max(1, cores / 2));
    // Beside a GPU the safe pick is 1 (see the table above) — UNLESS the adaptive
    // governor is on, in which case we spawn the knee and let CpuRateGovernor
    // throttle the ACTIVE count down from it by the GPU's measured rate.
    if (gpu_peer) return cpu_adaptive_enabled() ? knee : 1;
    return knee;
}

// Adaptive CPU-worker target from the GPU's MEASURED plot rate.
//
// cpu_worker_auto_count cannot know the GPU's speed — it sees only cores and RAM
// — so beside a GPU it plays safe and picks 1. But the right count is a steep
// function of how fast the GPU actually is, and that becomes knowable a few plots
// in. This maps a measured GPU s/plot to the worker count that maximised rig
// throughput in the 2026-07-17 k=28 bench-A/B sweeps (RTX 4090 + 5950X):
//
//   s_gpu (s/plot)   n*   evidence
//   2.56 (native)    ~0   1 was a wash, 4 ran 2.39x slower whole-run
//   6.86 (1005 MHz)  1    gains 13.3 / 11.2 / 10.7 — decreasing in n
//   9.45-9.50        4    two runs agree: native 510 MHz + bracket-validated 705
//
// The 6.9-9.5 gap is UNMEASURED: the throttled 4090's rate proved too bimodal to
// pin the crossover. The ramp across it is a monotone interpolation between the
// two solid endpoints — and as an adaptive input (re-evaluated against the live
// rate, not frozen at t=0) it is a forgiving place for a guess. The floor is 1,
// never 0: --cpu is opt-in, so resolving a request to zero would ignore it, and 1
// is a wash at worst on a card this fast. The ceiling is 4 (measured); RAM trims
// it further at the call site, so this need not know the host's memory.
int cpu_target_for_gpu_rate(double s_gpu_seconds)
{
    if (s_gpu_seconds <= 0.0) return 1;   // no rate yet — the safe opt-in floor
    if (s_gpu_seconds < 7.0)  return 1;   // fast GPU: 6.86 measured n*=1
    if (s_gpu_seconds < 8.0)  return 2;   // crossover ramp (interpolated)
    if (s_gpu_seconds < 9.0)  return 3;   // crossover ramp (interpolated)
    return 4;                             // slow GPU: 9.45-9.50 measured n*=4
}

// How many CPU workers actually fit, resolving the cpu_workers request against
// host RAM. `asked` is PER NODE; the return is the HOST-WIDE TOTAL across
// `nodes` of them, because RAM is a host resource and gating per node would let
// a 4-socket box authorise 4x what /proc/meminfo actually has.
//   kCpuWorkersAuto (-1): the knee per node, then trimmed to what RAM holds.
//   kCpuWorkersMax  (-2): as many as RAM holds, no knee cap.
//   0                   : none.
//   N > 0               : exactly N per node, still RAM-trimmed.
//
// Never throws: batch_worker_count() calls this from outside the CLI's try block
// (cli.cpp), so a throw here would terminate instead of printing.
int cpu_workers_that_fit(int asked, int k, std::size_t gpu_count,
                         std::vector<NumaNode> const& nodes,
                         std::string* note)
{
    if (asked == 0 || nodes.empty()) return 0;  // --cpu-workers 0, or no CPU selected

    bool const is_auto = (asked == kCpuWorkersAuto);
    bool const is_max  = (asked == kCpuWorkersMax);
    if (asked < 0 && !is_auto && !is_max) return 0;  // unknown sentinel: be safe

    unsigned const hc = std::thread::hardware_concurrency();
    int const host_cores = static_cast<int>(hc ? hc : 8);
    // A node only reports an empty cpu list when the kernel has no NUMA sysfs at
    // all (host_numa_nodes then synthesises one node meaning "the machine"). A
    // single-node CONFIG_NUMA host — the common case — DOES list its cpus, so
    // this fallback is for the sysfs-less host, not the single-socket one.
    auto node_cores = [&](NumaNode const& n) {
        return n.cpus.empty() ? host_cores : static_cast<int>(n.cpus.size());
    };

    // Only `auto` is GPU-aware. `max` and an exact N are the caller saying what
    // they want; this is the one setting that asked US to choose, so it is the
    // only one entitled to change its mind about the answer.
    bool const gpu_peer = gpu_count > 0;
    int knee = 0;
    for (auto const& n : nodes) knee += cpu_worker_auto_count(node_cores(n), gpu_peer);

    // Pre-RAM target: auto → the summed per-node knee; max → the HOST core count
    // (each worker already fans out to every core, so more than that is pure
    // oversubscription — at k=22 RAM alone would permit ~500, and that cap is
    // about the machine, so it is not multiplied by the node count); N → N on
    // each node. RAM trims all of them further.
    long const target = is_auto ? static_cast<long>(knee)
                      : is_max  ? static_cast<long>(host_cores)
                                : static_cast<long>(asked) *
                                      static_cast<long>(nodes.size());

    auto gib = [](std::uint64_t b) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f",
                      static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0));
        return std::string(buf);
    };

    if (char const* v = std::getenv("XCHPLOT2_CPU_WORKERS_UNGATED");
        v && v[0] == '1') {
        // "I know my box better than /proc does." No RAM check; honour the
        // resolved target as-is (auto → knee, max → core count, N → N).
        int const n = static_cast<int>(target);
        return n > 0 ? n : 0;
    }

    std::uint64_t const per_worker = cpu_worker_peak_bytes(k);
    std::uint64_t const free_now   = host_free_bytes_once();
    if (free_now == 0 || per_worker == 0) {
        // The probe failed (no /proc/meminfo? a kernel older than 3.14?). Do not
        // silently drop the CPU over a failure to READ memory — grant the knee
        // (auto/max) or the exact N, and say so.
        int const n = (is_auto || is_max) ? knee : static_cast<int>(target);
        if (note) {
            *note = "could not read host memory — running " + std::to_string(n) +
                    " CPU worker(s) ungated (each needs ~" +
                    std::to_string(per_worker >> 30) + " GiB at k=" +
                    std::to_string(k) + ")";
        }
        return n;
    }

    std::uint64_t const budget = cpu_budget_bytes(k, gpu_count);
    long const ram_fits = static_cast<long>(budget / per_worker);
    int const fits = static_cast<int>(std::max(0L, std::min(target, ram_fits)));

    if (note) {
        auto reserves = [&](std::string& s) {
            if (gpu_count > 0) {
                s += ", " + std::to_string(gpu_count) + " GPU worker" +
                     (gpu_count == 1 ? " reserves " : "s reserve ") +
                     gib(static_cast<std::uint64_t>(gpu_count) *
                         gpu_worker_host_peak_bytes(k)) + " GiB";
            }
            if (std::uint64_t const held = host_reserve_bytes(); held > 0) {
                s += ", " + gib(held) + " GiB kept back (XCHPLOT2_CPU_RESERVE_MB)";
            }
        };

        if ((is_auto || is_max) && fits > 0) {
            // A new opt-OUT default, so say what it picked and why, once.
            *note = (is_max ? "filling RAM: " : "auto: ") + std::to_string(fits) +
                    " CPU worker" + (fits == 1 ? "" : "s") + " (each " +
                    gib(per_worker) + " GiB at k=" + std::to_string(k) +
                    ", host has " + gib(free_now) + " GiB available";
            reserves(*note);
            *note += ")";
            if (is_auto && ram_fits < knee) {
                *note += " — RAM holds " + std::to_string(fits) + " of the ~" +
                         std::to_string(knee) + " that help here";
            }
            // Why this is not the ~4 the CPU-only tuning would predict. Said
            // here rather than left to the release notes because the number
            // visibly disagrees with the knee documented on --cpu-workers, and
            // a user who reads both without this deserves to think it is a bug.
            if (is_auto && gpu_peer && cpu_adaptive_enabled()) {
                *note += " — spawned beside a GPU for the adaptive governor to "
                         "throttle: only as many run as the GPU's measured plot "
                         "rate justifies (1 on a fast card, up to the knee on a "
                         "slow one)";
            } else if (is_auto && gpu_peer) {
                *note += " — held to 1 per node beside a GPU: the CPU-only knee "
                         "(~4) measures what extra workers add, not what they "
                         "cost the GPU's FSE consumer (4 of them ran 2.39x "
                         "slower than the GPU alone on a 4090)";
            }
            *note += is_auto && gpu_peer
                         ? ". --cpu-workers N to override, 0 to disable"
                         : ". --cpu-workers 0 to disable";
        // Auto used to stay silent here, because auto was the DEFAULT and
        // declining to add CPU workers nobody asked for is not news. Under the
        // opt-in it is: reaching auto at all means the CPU was requested, so
        // resolving to zero is a request going unhonoured and has to be said —
        // not least because the "nothing to plot on" throw quotes this note.
        } else if ((is_max || is_auto) && fits == 0) {
            *note = "no CPU worker fits: each needs " + gib(per_worker) +
                    " GiB at k=" + std::to_string(k) + ", host has " +
                    gib(free_now) + " GiB available";
            reserves(*note);
        // `target`, not `asked`: both sides must be host-wide totals. asked is
        // per node, so on a 2-node box `--cpu-workers 4` that trimmed to 5 would
        // compare 5 < 4, call it a success, and never mention losing 3 workers.
        } else if (!is_auto && !is_max && fits < target) {
            *note = "asked for " + std::to_string(target) + " CPU worker" +
                    (target == 1 ? "" : "s") +
                    (nodes.size() > 1 ? " (" + std::to_string(asked) + " on each of " +
                                            std::to_string(nodes.size()) + " nodes)"
                                      : "") +
                    " but " +
                    (fits == 0 ? std::string("none fit")
                               : "only " + std::to_string(fits) +
                                     (fits == 1 ? " fits" : " fit")) +
                    ": each needs " + gib(per_worker) + " GiB at k=" +
                    std::to_string(k) + ", host has " + gib(free_now) +
                    " GiB available";
            reserves(*note);
            *note += ". Set XCHPLOT2_CPU_WORKERS_UNGATED=1 to override (and risk "
                     "the OOM killer taking the whole batch)";
        }
        // Every (asked, fits) pair now says something; nothing falls through
        // silently. See the auto note above for why that changed.
    }
    return fits;
}

}  // namespace

std::uint64_t cpu_worker_peak_bytes(int k)
{
    // Peak RSS of a one-plot, CPU-only process:
    //
    //   k=22      223 568 kB     54.58 B per 2^k entry
    //   k=24      927 036 kB     56.58
    //   k=26    3 226 100 kB     49.23
    //   k=28   12 727 020 kB     48.55      (12.14 GiB)
    //
    // The k=28 figure reproduced 0.04% apart across two separate runs, and the
    // spread across four concurrent workers was 0.03% — this is a tight, stable
    // number, so there is no fudge factor on it. (The safety lives in the
    // reserve, which is the term that is genuinely uncertain.)
    //
    // It is ~2^k, but not a clean power of two: the coefficient drifts DOWN as
    // k rises, because table sizes track match counts rather than 2^k exactly.
    // So interpolate the coefficient between anchors instead of picking one and
    // pretending it holds everywhere.
    //
    // OUTSIDE the measured range, take the largest coefficient ever seen (56.58,
    // at k=24) rather than extending the trend. The trend is downward, which
    // means extrapolating it is exactly the way to under-estimate — and this is
    // the number that decides whether the box survives.
    struct Anchor { int k; double bytes_per_entry; };
    static constexpr Anchor kAnchors[] = {
        {22, 54.58}, {24, 56.58}, {26, 49.23}, {28, 48.55},
    };
    static constexpr std::size_t kN = sizeof(kAnchors) / sizeof(kAnchors[0]);
    static constexpr double kWorstSeen = 56.58;

    double coeff = kWorstSeen;
    if (k >= kAnchors[0].k && k <= kAnchors[kN - 1].k) {
        for (std::size_t i = 1; i < kN; ++i) {
            if (k <= kAnchors[i].k) {
                Anchor const& lo = kAnchors[i - 1];
                Anchor const& hi = kAnchors[i];
                double const t =
                    static_cast<double>(k - lo.k) / static_cast<double>(hi.k - lo.k);
                coeff = lo.bytes_per_entry +
                        t * (hi.bytes_per_entry - lo.bytes_per_entry);
                break;
            }
        }
    }
    if (k < 1 || k > 40) return 0;  // nonsense k — let the caller's guards speak
    double const entries = static_cast<double>(std::uint64_t{1} << k);
    return static_cast<std::uint64_t>(coeff * entries);
}

std::vector<std::string> worker_labels(std::vector<int> const& device_ids)
{
    // device_label() alone is ambiguous the moment a device repeats — and with
    // N CPU workers it does, so a 4-worker bench would print four lines all
    // called "cpu" and a log would interleave four "[batch:cpu]" prefixes with
    // no way to tell which worker stalled. Suffix repeats with #ordinal; leave
    // unique devices exactly as they were, so single-CPU logs do not churn.
    std::vector<std::string> labels;
    labels.reserve(device_ids.size());
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        std::string base = device_label(device_ids[i]);
        std::size_t const total = static_cast<std::size_t>(
            std::count(device_ids.begin(), device_ids.end(), device_ids[i]));
        if (total > 1) {
            std::size_t const ordinal = static_cast<std::size_t>(
                std::count(device_ids.begin(), device_ids.begin() + static_cast<long>(i),
                           device_ids[i]));
            base += "#" + std::to_string(ordinal);
        }
        labels.push_back(std::move(base));
    }
    return labels;
}

std::vector<int> resolve_batch_devices(BatchOptions const& opts,
                                       int                 k,
                                       std::string*        gate_note)
{
    std::vector<int> device_ids;
    if (opts.use_all_devices) {
        // "gpu"/"all" means every card fit for AUTOMATIC dispatch -- the tiny
        // integrated GPUs (a 1-CU iGPU) are filtered out here so a plot never
        // lands on one. The explicit --devices <index> branch below is verbatim
        // and bypasses this, so such a device stays deliberately targetable.
        device_ids = sycl_backend::auto_dispatchable_indices();
    } else if (!opts.device_ids.empty()) {
        device_ids = opts.device_ids;
    }

    // Zero-config (no --devices and not --devices all) historically means "the
    // one default GPU, on the single-worker fast path" — signalled by leaving the
    // list empty. `--cpu` can join CPU workers to that implicit selection, and
    // appending them to an empty list would run CPU-ONLY, silently dropping the
    // default GPU. So when CPU workers join an implicit selection, materialise
    // the default GPU (device 0) so it shares the work queue.
    //
    // `--devices cpu` ALSO arrives here with an empty device_ids — CPU nodes are
    // tracked separately from GPU ids — but it is an EXPLICIT CPU-only request,
    // not zero-config. devices_specified tells the two apart, so we do not bolt
    // a GPU onto a run that named CPU and only CPU.
    bool const gpu_implicit = device_ids.empty() && !opts.use_all_devices
                              && !opts.devices_specified;
    bool default_gpu_available = false;
    std::size_t gpu_count = device_ids.size();
    if (gpu_implicit) {
        default_gpu_available = (gpu_device_count() > 0);
        if (default_gpu_available) ++gpu_count;  // it costs host RAM once real
    }

    // Which CPU nodes are in. `cpu` / `all` take every node the host has, as
    // `gpu` takes every card; `cpu0` names one, as `gpu0` does. An unselected
    // CPU yields an empty list, and everything below is then a no-op.
    std::vector<NumaNode> cpu_nodes;
    if (opts.cpu_selected()) {
        auto const topo = host_numa_nodes();  // never empty; see NumaTopology.hpp
        if (opts.use_all_cpu_nodes || opts.cpu_node_ids.empty()) {
            cpu_nodes = topo;
        } else {
            for (int want : opts.cpu_node_ids) {
                auto const it = std::find_if(
                    topo.begin(), topo.end(),
                    [&](NumaNode const& n) { return n.node_id == want; });
                if (it != topo.end()) {
                    cpu_nodes.push_back(*it);
                } else if (gate_note) {
                    // Naming a node the host does not have is a typo worth
                    // saying out loud — silently plotting on fewer nodes than
                    // asked is the kind of thing nobody notices until the
                    // throughput number is unexplainable.
                    *gate_note = "no NUMA node " + std::to_string(want) +
                                 " on this host (it has " +
                                 std::to_string(topo.size()) + ") — ignoring cpu" +
                                 std::to_string(want);
                }
            }
        }
    }

    // Every CPU worker is one more CPU-node id in the list — the work-queue needs
    // no idea several of them share a socket, because what they share (cores,
    // memory bandwidth) is not something it arbitrates. Resolved per node
    // (auto/max/exact) and RAM-trimmed host-wide.
    int const cpu_count =
        cpu_workers_that_fit(opts.cpu_workers, k, gpu_count, cpu_nodes, gate_note);

    // Materialise the default GPU whenever CPU plotting is in play and one
    // exists — EVEN IF the count trimmed to zero. That way a run that wanted CPU
    // workers but couldn't fit them still lands on the GPU (with the gate note
    // explaining why), instead of being dropped to CPU-only or spuriously
    // refused. No CPU leaves the list empty, so the zero-config GPU fast path is
    // byte-for-byte unchanged.
    if (gpu_implicit && default_gpu_available) {
        // The default GPU is the first AUTO-DISPATCHABLE one, not blindly index
        // 0 -- index 0 may be a tiny iGPU that would otherwise get the plots.
        //
        // This used to fire ONLY when CPU workers joined an implicit selection;
        // plain zero-config left the list empty and rode kDefaultGpuId, where
        // AdaptiveCpp's own gpu_selector_v chose with NO compute-unit filter at
        // all. So the filter that exists to keep plots off a 1-CU iGPU did not
        // cover the single most common invocation -- `xchplot2 plot` with no
        // flags. Materialising the id here applies it uniformly, and gives the
        // run a concrete `[batch:gpuN]` prefix instead of an ambiguous
        // `[batch]` that hides which device was picked.
        //
        // Downstream is unaffected: one id still takes the single-worker fast
        // path at `device_ids.size() <= 1`.
        device_ids.push_back(sycl_backend::default_dispatch_index());
    }
    // Round-robin across the selected nodes rather than filling one at a time:
    // the RAM trim is host-wide, so when it cuts the total the survivors should
    // still be spread over the sockets instead of piling onto node 0 — which is
    // the arrangement the pinning exists to avoid.
    for (int i = 0; i < cpu_count && !cpu_nodes.empty(); ++i) {
        device_ids.push_back(
            cpu_device_id(cpu_nodes[static_cast<std::size_t>(i) % cpu_nodes.size()].node_id));
    }
    return device_ids;
}

// Drop selected GPUs that cannot actually run kernels, so one bad device does
// not take down a run the host's other devices could have finished.
//
// sycl_backend::queue() already answers "can this device work at all" — it
// constructs the queue and runs validate_kernel_dispatch on it — but it answers
// fatally, and the throw propagates out of whichever worker touched the device
// first. Observed on a host with an AMD APU and a discrete Intel card: the Arc
// failed to build a code object, and the run died instead of continuing on the
// other device. The same applies to a card whose kernels complete without
// writing, and to a driver that hangs and resets mid-probe.
//
// Each probe runs on its own thread: queue() and the AES-table cache are
// thread_local and keyed off the thread's device id, so probing inline would
// bind this thread to the last device probed and leave a queue behind on it.
//
// Only explicit ids are probed. kDefaultGpuId means "whatever the default
// selector picks", and there is nothing to fall back TO if it fails, so its
// failure should stay fatal and keep its original message.
std::vector<int> usable_batch_devices(std::vector<int> const&    device_ids,
                                      std::vector<std::string>* dropped)
{
    std::vector<int> keep;
    keep.reserve(device_ids.size());
    for (int id : device_ids) {
        if (id < 0) { keep.push_back(id); continue; }  // CPU nodes + default GPU
        std::string err;
        std::thread probe([&] {
            try {
                sycl_backend::set_current_device_id(id);
                (void)sycl_backend::queue();
            } catch (std::exception const& e) {
                err = e.what();
            } catch (...) {
                err = "unknown exception";
            }
        });
        probe.join();
        if (err.empty()) { keep.push_back(id); continue; }
        if (dropped) {
            // First line only — the selftest failure is a paragraph of advice.
            auto const nl = err.find('\n');
            dropped->push_back(device_label(id) + ": " +
                               (nl == std::string::npos ? err : err.substr(0, nl)));
        }
    }
    return keep;
}

BatchStrategy resolve_batch_strategy(BatchOptions const& opts,
                                     std::vector<int> const& device_ids,
                                     int k,
                                     std::string* reason_out)
{
    if (opts.strategy != BatchStrategy::Auto) {
        if (reason_out) *reason_out = "explicit --strategy override";
        return opts.strategy;
    }
    if (opts.shard_plot) {
        if (reason_out) *reason_out = "legacy --shard-plot opt-in";
        return BatchStrategy::ShardPlot;
    }
    if (opts.pipeline_plot) {
        if (reason_out) *reason_out = "legacy --pipeline-plot opt-in";
        return BatchStrategy::PipelinePlot;
    }
    StrategyPickInputs inputs{device_ids, k};
    return select_strategy(inputs, reason_out);
}

std::size_t batch_worker_count(BatchOptions const& opts, int k)
{
    auto const device_ids = resolve_batch_devices(opts, k);
    auto const strategy   = resolve_batch_strategy(opts, device_ids, k);
    if (strategy == BatchStrategy::ShardPlot && device_ids.size() > 1) {
        return 1;  // devices form one team; one plot in flight
    }
    if (strategy == BatchStrategy::PipelinePlot) return 1;
    if (device_ids.size() <= 1) return 1;
    return device_ids.size();
}

BatchResult run_batch(std::vector<BatchEntry> const& entries,
                      BatchOptions const& opts)
{
    if (entries.empty()) return BatchResult{};

    // Pin WHO builds the shared FSE pool: this thread, before any worker exists.
    //
    // The pool is a function-local static built on first use, and on Linux its
    // 32 workers inherit the nice value of the thread that constructs them. The
    // CPU worker is niced down (nice_current_thread) and — since the writer swap
    // — also writes through this pool, so if it got there first every GPU
    // worker's FSE would be born niced too. That is irreversible for an
    // unprivileged process. Constructing the pool here, on the un-niced caller,
    // makes the race unlosable rather than merely unlikely.
    //
    // Free: the pool would be constructed by the first write regardless.
    warm_writer_pool();

    // Homogeneity check (all entries must share k/strength/testnet) —
    // runs once on the full list before any per-worker dispatch so both
    // the single- and multi-device paths share the same error surface.
    int  const pool_k        = entries[0].k;
    int  const pool_strength = entries[0].strength;
    bool const pool_testnet  = entries[0].testnet;
    for (size_t i = 1; i < entries.size(); ++i) {
        if (entries[i].k != pool_k
            || entries[i].strength != pool_strength
            || entries[i].testnet  != pool_testnet)
        {
            throw std::runtime_error(
                "run_batch: all entries must share (k, strength, testnet)");
        }
    }

    preflight_disk_space(entries, opts);

    // Resolve the target device list (see resolve_batch_devices):
    //   use_all_devices  → enumerate at runtime, one worker per GPU
    //   device_ids       → use these explicit ids
    //   (neither)        → empty list → single-device default selector
    //   cpu_workers      → orthogonal: append that many kCpuDeviceId
    //                      entries so the CPU runs as N more workers.
    //                      Mixes with the above (--cpu alone → CPU only;
    //                      --cpu --devices all → all GPUs + CPU; etc.).
    //                      The count is capped at what host RAM holds.
    std::string gate_note;
    std::vector<int> device_ids =
        resolve_batch_devices(opts, pool_k, &gate_note);

    // Validate the selection before committing to it, but only when something
    // else could carry the run — with a single worker there is nothing to fall
    // back to, and the original failure is a better message than "dropped".
    if (device_ids.size() > 1) {
        std::vector<std::string> dropped;
        device_ids = usable_batch_devices(device_ids, &dropped);
        for (auto const& d : dropped) {
            if (!opts.quiet) {
                std::fprintf(stderr, "[batch] device dropped — %s\n", d.c_str());
            }
        }
        if (device_ids.empty() && !dropped.empty()) {
            std::string msg = "every selected device failed its dispatch check:";
            for (auto const& d : dropped) msg += "\n  - " + d;
            throw std::runtime_error(msg);
        }
    }
    if (!gate_note.empty() && !opts.quiet) {
        // Once per distinct message. bench drives two passes (warmup + measured)
        // through run_batch back to back, and the CPU auto/gate line is identical
        // across them — no reason to print it twice. (run_batch is a sequential
        // top-level entry point, so this un-synchronised static is safe.)
        static std::string last_cpu_note;
        if (gate_note != last_cpu_note) {
            last_cpu_note = gate_note;
            std::fprintf(stderr, "[batch] cpu: %s\n", gate_note.c_str());
        }
    }

    // The gate can take the count to zero. If the CPU was asked for and none
    // fit AND no GPU was selected, say so — the fast path below would otherwise
    // read the empty list as "no device selected", fall back to the default SYCL
    // selector, and either plot on an unrequested GPU or fail obscurely.
    //
    // This used to exclude `auto`, on the reasoning that auto resolving to zero
    // was not a failure but the tool declining to add CPU workers nobody had
    // asked for. That reasoning died with the opt-in: auto is now only reached
    // once something HAS asked for the CPU, so `--devices cpu` on a host too
    // tight to fit one is a request that cannot be honoured, not a default
    // quietly declining. Ask the SELECTION, not the count.
    //
    // A run that also named GPUs still falls through to them with the gate note
    // — device_ids is non-empty, so the throw does not fire.
    std::size_t const cpu_workers_placed = static_cast<std::size_t>(
        std::count_if(device_ids.begin(), device_ids.end(),
                      [](int id) { return is_cpu_device(id); }));
    if (opts.cpu_selected() && cpu_workers_placed == 0 && device_ids.empty()) {
        throw std::runtime_error(
            "run_batch: no CPU worker fits in host RAM and no GPU was selected"
            " — nothing to plot on. " + gate_note);
    }

    if (opts.use_all_devices &&
        std::none_of(device_ids.begin(), device_ids.end(),
                     [](int id) { return !is_cpu_device(id); })) {
        std::fprintf(stderr,
            "[batch] --devices all: runtime enumerated 0 GPUs — "
            "falling back to the default SYCL selector\n");
    }

    auto const t_start = std::chrono::steady_clock::now();

    // Phase 2.4: resolve the multi-GPU strategy. The user can set
    // opts.strategy explicitly, OR keep the legacy bool fields
    // (shard_plot / pipeline_plot) which are honoured when
    // strategy == Auto. When strategy == Auto and neither legacy bool
    // is set, the picker runs the heuristic.
    std::string resolved_reason;
    BatchStrategy const resolved_strategy =
        resolve_batch_strategy(opts, device_ids, pool_k, &resolved_reason);
    if (opts.verbose) {
        char const* name = "work-queue";
        switch (resolved_strategy) {
            case BatchStrategy::Auto:         name = "auto(?)"; break;
            case BatchStrategy::WorkQueue:    name = "work-queue"; break;
            case BatchStrategy::PipelinePlot: name = "pipeline-plot"; break;
            case BatchStrategy::ShardPlot:    name = "shard-plot"; break;
        }
        std::fprintf(stderr, "[strategy] picked %s — %s\n",
                     name, resolved_reason.c_str());
    }

    // Single-worker strategies below run one plot at a time, so their whole
    // timeline belongs to one worker: publish it as such. Nothing is compared
    // across workers when there is only one, so their own steady_clock origin
    // is fine as the epoch — only the work-queue fan-out needs a shared one.
    auto as_single_worker = [](BatchResult& r, int dev) {
        WorkerTimeline w;
        w.device_id            = dev;
        w.work_start_seconds   = r.work_start_seconds;
        w.completion_seconds   = r.completion_seconds;
        w.pipeline             = r.pipeline;
        r.workers.assign(1, std::move(w));
    };

    // Single-plot-multi-GPU dispatch (shard-plot strategy). Each plot
    // runs across all selected devices as a "team" instead of
    // distributing plots between independent workers. Niche on
    // PCIe-only — see README. N=1 falls through to single-GPU path.
    if (resolved_strategy == BatchStrategy::ShardPlot && device_ids.size() > 1) {
        BatchResult r = run_batch_sharded(entries, opts, device_ids);
        r.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        as_single_worker(r, device_ids[0]);  // the team plots as one worker
        return r;
    }

    if (resolved_strategy == BatchStrategy::PipelinePlot) {
        BatchResult r = run_batch_pipeline_plot(entries, opts, device_ids);
        r.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        as_single_worker(r, device_ids.empty() ? kDefaultGpuId : device_ids[0]);
        return r;
    }

    // Fast path: zero-config default or one explicit id. Runs on the
    // caller thread — identical control flow to pre-multi-GPU except
    // for the optional thread-local device bind at the top of the
    // slice.
    // The live memory gate, shared by every CPU worker. Re-asks at each plot
    // boundary what the start-of-batch gate could only ask once — see
    // CpuMemoryGate. Only built when there is a CPU worker to gate, and skipped
    // entirely when the user has taken the override: XCHPLOT2_CPU_WORKERS_UNGATED
    // means "I know my box better than /proc does", and it would be a strange
    // reading of that to keep second-guessing them every 50 seconds.
    auto make_cpu_gate = [&](std::size_t gpu_count)
        -> std::unique_ptr<CpuMemoryGate> {
        if (cpu_workers_placed == 0) return nullptr;
        if (char const* v = std::getenv("XCHPLOT2_CPU_WORKERS_UNGATED");
            v && v[0] == '1') {
            return nullptr;
        }
        return std::make_unique<CpuMemoryGate>(
            cpu_worker_peak_bytes(pool_k),
            cpu_budget_bytes(pool_k, gpu_count));
    };

    if (device_ids.size() <= 1) {
        int const dev = device_ids.empty() ? kDefaultGpuId : device_ids[0];
        BatchProgress live(1, entries.size());
        live.cpu_gate = make_cpu_gate(/*gpu_count=*/0);
        BatchResult r = run_batch_slice(entries, opts, dev, 0, live,
                                        nullptr, &t_start);
        r.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        as_single_worker(r, dev);
        return r;
    }

    // Multi-device: workers race to pull plots from a single shared
    // queue (atomic counter into `entries`) so a fast GPU keeps pulling
    // work while a slow CPU only handles what it can finish in the same
    // wall. Each worker still constructs its own GpuBufferPool /
    // producer-consumer channel / writer thread on its target device —
    // zero cross-worker shared state beyond `next_idx`, stderr, and
    // the filesystem.
    size_t const N = device_ids.size();
    auto const labels = worker_labels(device_ids);
    if (!opts.quiet) {
        std::string devs;
        for (size_t i = 0; i < N; ++i) {
            if (i) devs += ", ";
            devs += labels[i];
        }
        std::fprintf(stderr,
            "[batch] multi-device: %zu plots across %zu workers "
            "(work-queue: pulled by speed, not split evenly) — devices: %s\n",
            entries.size(), N, devs.c_str());
    }

    std::atomic<std::size_t> next_idx{0};
    // Shared progress state for --progress on multi-device runs. The emitted
    // "N/M done" line is aggregate across workers — not the per-worker slice it'd
    // otherwise reflect — and the ETA reads each worker's OWN rate and in-flight
    // count out of here, because a work-queue's last plots are held by specific
    // workers and no batch-wide mean can price that drain.
    BatchProgress live(N, entries.size());
    for (size_t i = 0; i < N; ++i) live.workers[i].label = labels[i];

    // Only a GPU peer is worth yielding to — see the nice call in the CPU branch
    // of run_batch_slice. Everything in this list that is not the CPU is a GPU
    // (the default-selector sentinel never reaches the multi-device path).
    live.gpu_peer_present =
        std::any_of(device_ids.begin(), device_ids.end(),
                    [](int id) { return !is_cpu_device(id); });

    // Seed the tail guard's per-worker priors by device class, so a known-slow CPU
    // worker can stand down on a batch too short to help before it has measured
    // its own rate. Each is replaced by that worker's measured rate after 2 plots.
    for (std::size_t i = 0; i < N && i < live.worker_prior_s.size(); ++i) {
        live.worker_prior_s[i] = is_cpu_device(device_ids[i])
                                     ? cpu_plot_seconds_prior()
                                     : kGpuPlotSecondsPrior;
    }

    live.cpu_gate = make_cpu_gate(N - cpu_workers_placed);  // the rest are GPUs

    // Adaptive CPU throttle: only when opted in, and only on a mixed GPU+CPU team
    // — it reads GPU rate to gate CPU workers, so it needs both present. The spawn
    // count stayed the knee (cpu_worker_auto_count under adaptive); the governor
    // decides how many of those workers actually run.
    if (cpu_adaptive_enabled() && live.gpu_peer_present) {
        std::vector<std::size_t> gpu_slots;
        std::vector<int>         cpu_ordinal(N, -1);
        int cpu_seen = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (is_cpu_device(device_ids[i])) cpu_ordinal[i] = cpu_seen++;
            else                              gpu_slots.push_back(i);
        }
        if (!gpu_slots.empty() && cpu_seen > 0) {
            live.cpu_rate_gov = std::make_unique<CpuRateGovernor>(
                live.workers, std::move(gpu_slots), std::move(cpu_ordinal));
        }
    }

    std::vector<BatchResult>         per_worker(N);
    std::vector<std::exception_ptr>  per_worker_exc(N);
    std::vector<std::thread>         workers;
    workers.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        workers.emplace_back([&, i]() {
            try {
                per_worker[i] = run_batch_slice(
                    entries, opts, device_ids[i],
                    static_cast<int>(i), live, &next_idx, &t_start);
            } catch (...) {
                per_worker_exc[i] = std::current_exception();
                // Tell peer workers to drain after their current plot
                // and stop pulling new ones. Without this, an ENOSPC
                // on one disk (or any other worker-side failure) keeps
                // peers plotting until the manifest is exhausted, only
                // to surface the failure at join time. Cooperative
                // cancel saves the wasted work + the partial cleanup.
                request_cancel();
            }
        });
    }
    for (auto& t : workers) t.join();

    // Propagate the first worker exception after every worker has
    // joined — prevents a fast failure from leaving peer workers still
    // running and printing to a half-torn-down pipeline.
    for (auto& ep : per_worker_exc) {
        if (ep) std::rethrow_exception(ep);
    }

    BatchResult agg;
    agg.workers.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        BatchResult const& r = per_worker[i];
        agg.plots_written += r.plots_written;
        agg.plots_skipped += r.plots_skipped;
        agg.plots_failed  += r.plots_failed;
        agg.bytes_written += r.bytes_written;
        agg.completion_seconds.insert(
            agg.completion_seconds.end(),
            r.completion_seconds.begin(), r.completion_seconds.end());

        // Keep each worker's timeline intact alongside the merged one. Who
        // finished what is not recoverable once the lists are merged, and a
        // work-queue gives no worker a predictable share — so any per-worker
        // question (its own steady-state, its warmup, whether it idled while a
        // peer drained the queue) can only be answered from here.
        WorkerTimeline w;
        w.device_id          = device_ids[i];
        w.work_start_seconds = r.work_start_seconds;
        w.completion_seconds = r.completion_seconds;  // already ascending
        w.pipeline           = r.pipeline;
        agg.workers.push_back(std::move(w));
    }
    std::sort(agg.completion_seconds.begin(), agg.completion_seconds.end());
    agg.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return agg;
}

} // namespace pos2gpu
