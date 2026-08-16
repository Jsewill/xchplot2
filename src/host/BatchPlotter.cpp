// BatchPlotter.cu — implementation of staggered multi-plot pipeline.

#include "host/BatchPlotter.hpp"
#include "host/Cancel.hpp"
#include "host/CpuPlotter.hpp"  // run_one_plot_cpu — pos2-chip CPU pipeline
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#include "host/HostRamPolicy.hpp"  // plan_host_ram_spill — the budget policy
#include "host/TempFile.hpp"       // --temp-dir plumbing
#include "host/PlotFileWriterParallel.hpp"
#include "gpu/DeviceIds.hpp"  // kCpuDeviceId for the --cpu device-list mixin
#include "host/NumaTopology.hpp"  // CPU-node enumeration + per-worker pinning

// Deliberately no pos2-chip includes here — see PlotFileWriterParallel.cpp.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>  // std::function — CpuMemoryGate::acquire's still_wanted
#include <map>
#include <memory>
#include <mutex>
#include <optional>    // std::optional — CpuMemoryGate's starvation clock
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h>  // isatty — in-place progress line only on a TTY

#ifdef __linux__
#include <sys/resource.h>  // setpriority / PRIO_PROCESS — see nice_current_thread
#include <cerrno>
#include <cstring>         // std::strerror
#endif
#ifdef _WIN32
#include <windows.h>  // GlobalMemoryStatusEx — see host_memory_probe
#endif

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

// Check `.plot2` is present at path AND looks like a valid plot file
// (magic bytes "pos2" + nonzero size). Used for skip_existing so we
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

// Rough per-plot upper-bound estimate for the disk preflight. The
// actual compressed .plot2 is smaller (FSE over proof-fragment stubs);
// this uncompressed ceiling is deliberately pessimistic so we only
// WARN when the disk is genuinely too small, not for boundary cases.
//
// Formula: 2^k fragments × proof_fragment_bits/8, where
// proof_fragment_bits ≈ k stub + (k - 2) xbits + overhead ≈ 2k bits.
uint64_t approx_plot_bytes_upper_bound(int k)
{
    if (k <= 0 || k > 32) return 0;
    uint64_t const fragments = uint64_t(1) << k;
    uint64_t const bits_per  = uint64_t(2 * k);
    return (fragments * bits_per) / 8;
}

// Group the batch by output directory, statvfs each, and WARN (don't
// throw) when free space is below the upper-bound need. Advisory only:
// the writer's ENOSPC handling is the real safety net; this just gives
// the user a chance to free space before a long run dies mid-write.
void preflight_disk_space(std::vector<BatchEntry> const& entries,
                          BatchOptions const& opts)
{
    if (entries.empty()) return;

    std::map<std::string, std::pair<size_t, uint64_t>> per_dir;
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
                "still run; consider freeing space or reducing count.\n",
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

// Host RAM. The CPU "device" IS the host, so anything sizing a CPU worker has to
// ask the host — putting the question to a GPU runtime is a category error.
//
// (On main this lives in GpuBufferPool.cpp, as a branch of a device_memory_probe()
// that folds every negative ordinal onto GPU 0 — which quietly swallowed
// kCpuDeviceId and answered "how much memory does the CPU have?" with the 4090's
// free VRAM. cuda-only has no such dispatcher: its CPU worker returns from
// run_batch_slice long before any pool or tier code, so the bug never existed
// here. The probe still has to exist, because the RAM gate below is its only
// caller and it is pure host code — hence it lives here, not in a .cu.)
//
// MemAvailable, not MemFree: it is the kernel's own estimate of what a fresh
// allocation can obtain without swapping, and it counts reclaimable page cache.
// MemFree on a box that has been plotting reads near zero (the cache is holding
// the plots just written), which would starve every CPU worker for no reason.
// This box: MemFree 13 GiB, MemAvailable 77 GiB.
// Testing knob: XCHPLOT2_HOST_FREE_MB makes the host look SMALLER than it is,
// so the streaming tiers' host-RAM gate can be exercised on a box that has
// plenty. It CLAMPS — it can only ever lower the figure, never raise it — so no
// setting of it can talk the plotter past a real shortage and into the OOM
// killer. Every consumer sees the reduced number, which is what makes a run
// under it a faithful simulation rather than just a way to steer a decision.
void apply_host_free_override(std::size_t& free_bytes)
{
    static std::size_t const cap = [] () -> std::size_t {
        if (char const* v = std::getenv("XCHPLOT2_HOST_FREE_MB"); v && v[0]) {
            std::size_t const mb = std::size_t(std::strtoull(v, nullptr, 10));
            if (mb > 0) return mb << 20;
        }
        return 0;   // 0 == no override
    }();
    if (cap && cap < free_bytes) free_bytes = cap;
}

bool host_memory_probe(std::size_t& free_bytes, std::size_t& total_bytes)
{
#if defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (!::GlobalMemoryStatusEx(&st)) return false;
    free_bytes  = static_cast<std::size_t>(st.ullAvailPhys);
    total_bytes = static_cast<std::size_t>(st.ullTotalPhys);
    apply_host_free_override(free_bytes);
    return true;
#else
    std::FILE* fp = std::fopen("/proc/meminfo", "re");
    if (!fp) return false;
    unsigned long long avail_kb = 0;
    unsigned long long total_kb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), fp)) {
        unsigned long long v = 0;
        if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1)     avail_kb = v;
        else if (std::sscanf(line, "MemTotal: %llu kB", &v) == 1)    total_kb = v;
    }
    std::fclose(fp);
    if (total_kb == 0) return false;
    // MemAvailable landed in Linux 3.14. Older kernels report only MemFree, which
    // understates badly — fall back to MemTotal and let the gate's own arithmetic
    // be the backstop, rather than silently refusing to start any CPU worker.
    free_bytes  = static_cast<std::size_t>(avail_kb ? avail_kb : total_kb) << 10;
    total_bytes = static_cast<std::size_t>(total_kb) << 10;
    apply_host_free_override(free_bytes);
    return true;
#endif
}

std::uint64_t host_free_bytes_now()
{
    std::size_t free_b  = 0;
    std::size_t total_b = 0;
    if (!host_memory_probe(free_b, total_b)) return 0;
    return static_cast<std::uint64_t>(free_b);
}

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
// buffers raise our RSS and lower MemAvailable together, so they never look like
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
        bool                             announced = false;
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

// Live, lock-free progress for the ETA. A work-queue's remaining plots are held
// by SPECIFIC workers running at their OWN rates, so no batch-wide mean can
// price the drain — the estimator needs each worker's rate and its own in-flight
// count. See estimate_eta_seconds() in BenchStats.hpp for the model.
//
// Every strategy publishes here, single-worker ones into a one-slot table, so
// emit_progress_line has exactly one shape to reason about.
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
// overriding the default (kNumPinnedBuffers). Returns 0 when unset or outside
// [1, kNumPinnedBuffers].
//
// The slots buy ONE thing: overlapping plot K's D2H with the file writer
// draining K-1/K-2. They are not a correctness requirement — slot_gate already
// blocks reuse until the consumer has finished with a slot, so a single slot is
// correct, merely less overlapped. On the streaming path each slot given up
// hands back cap * 8 B of pinned host memory (2.03 GiB at k=28), which is why
// this is the cheapest lever on host RAM that exists on this branch: it needs
// no spill engine, no temp dir, and no change to any kernel.
//
// It is an env knob rather than a flag because the RAM-vs-overlap trade wants
// A/B measurement on the host that cares, and because a host that CAN afford
// the slots should keep them.
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
    // The CPU worker nices itself down so it stops starving the GPU workers, and
    // that is a one-way door for an unprivileged process — so it must fire only
    // when there is somebody to yield TO. `workers.size() > 1` used to be exactly
    // that test, on the reasoning that there was at most one CPU worker and
    // therefore any peer was a GPU. --cpu-workers makes that false: two CPU
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
// its depth-1 channel backs up, and the GPU — which retires a plot 6-16x faster
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
// with live_skip / live_fail / record_plot_completion below.
void live_claim(BatchProgress& live, int slot)
{
    live.workers[static_cast<std::size_t>(slot)].in_flight.fetch_add(
        1, std::memory_order_relaxed);
}

// Take one plot out of flight, SATURATING AT ZERO.
//
// Every retire on this branch is currently paired with a live_claim, so a plain
// fetch_sub would be correct today. It is written this way because an UNPAIRED
// one does not land on -1: in_flight is unsigned, so it lands on 2^64-1, and the
// damage is silent and total. On main, where run_batch_sharded and
// run_batch_pipeline_plot are implemented, exactly that happened — both retire
// without claiming (each publishes as a single worker, and a one-train ETA
// prices an in-flight plot and a queued one identically, so tracking in-flight
// there buys nothing). in_flight wrapped to SIZE_MAX, `retired + in_flight`
// wrapped back to 0, `unclaimed` pinned to the full batch size and never moved,
// and the ETA priced ~1.8e19 committed plots: --pipeline-plot printed "batch ETA
// ~14m08s" on a three-second run, and still said 14m08s at 100% done.
//
// cuda-only cannot hit that yet — run_batch_sharded here is a scaffold that
// throws, and there is no pipeline-plot path at all. But that scaffold is going
// to be filled in, and whoever fills it inherits the trap: the natural way to
// write a single-worker strategy is to retire without claiming. Saturating makes
// "a retire never invents work" a property of the counter instead of a
// convention every future caller has to remember.
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
// nothing).
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
// Depth = num_pinned_slots - 1 so the producer never overtakes the
// consumer by more than (num_pinned - 1) plots. The pinned slot the
// producer writes is slot (i % num_pinned_slots); with depth-(N-1)
// the consumer is guaranteed to have popped plot (i - N) before the
// producer overwrites its slot.
//
// N is a RUNTIME value (XCHPLOT2_DRAIN_SLOTS, see drain_slots_env), not
// kNumPinnedBuffers, and at N = 1 the depth is floored at 1 — which makes the
// guarantee above vacuous. That is fine, and it is why SlotGate exists: the
// gate, not this depth, is what actually makes reuse safe.
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
// pinned memory via the fragments span). Whenever the file write is
// slower than one GPU pass (slow disk, NFS), the producer's D2H for
// plot i would land on top of the in-flight read and silently corrupt
// the written plot.
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

// Samples free VRAM from the driver and remembers the low-water mark, so a run
// can be checked against what it declared it would use.
//
// peak() is baseline-relative: free-at-start minus free-at-worst, i.e. the VRAM
// this process consumed *after* the tier was picked. That is deliberately the
// same quantity the picker compares against free VRAM, so peak() and the tier
// model are directly comparable. (The *process* peak would additionally include
// the ~390 MiB CUDA context, which cudaMemGetInfo has already deducted from the
// free figure the picker saw — conflating the two is exactly how the SYCL tree's
// safety margin came to double-count the context and got set 4x too high.)
//
// Why sample the driver at all when the pool sizes every buffer up front: because
// the sizing is a *model*, and a model only catches allocations someone
// remembered to model. The bug that made every 4-11 GB card OOM was a 3128 MiB
// buffer allocated outside the accounted path, invisible to every in-process
// counter and obvious to the driver.
//
// Caveat: on a shared GPU another process allocating mid-run inflates our
// measured peak. The check is therefore fatal only under POS2GPU_ASSERT_VRAM=1
// (which `bench` sets, being a controlled measurement) and merely loud elsewhere.
class VramWatchdog {
public:
    explicit VramWatchdog(int ordinal) : ordinal_(ordinal)
    {
        size_t f = 0;
        size_t t = 0;
        if (!streaming_device_memory_probe(ordinal_, f, t)) return;  // unsupported → inert
        baseline_free_ = f;
        min_free_.store(f, std::memory_order_relaxed);
        started_ = true;
        th_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed)) {
                size_t f = 0;
                size_t t = 0;
                if (streaming_device_memory_probe(ordinal_, f, t)) {
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

bool assert_vram_enabled()
{
    char const* v = std::getenv("POS2GPU_ASSERT_VRAM");
    return v && v[0] == '1';
}

} // namespace

namespace {

// Per-worker pipeline. Extracted from run_batch so the multi-device
// fan-out can spawn N of these concurrently — one thread per GPU, each
// with its own pool / channel / consumer. The outer run_batch validates
// homogeneity once; this helper assumes it has already been done on
// `entries`.
//
// device_id < 0  → keep the CUDA-default current device (single-device
//                  default; zero-config users see unchanged behavior).
// worker_id      → this worker's slot in `live`. Used to be passed in and
//                  immediately (void)-cast; the ETA needs each worker's own rate
//                  and in-flight count, so it is now load-bearing.
// shared_idx (default null) lets multiple workers race for the next plot
// out of a single shared `entries` list. When set, every worker calls
// shared_idx->fetch_add(1) and exits when the result >= entries.size() —
// dynamic load balancing, so a fast GPU worker keeps pulling plots while
// a slow CPU worker handles only what it can finish in the same wall.
// When null (single-device path), the worker iterates 0..entries.size()-1
// in order — original behaviour.
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

    // CPU worker: bypass GPU pool / streaming entirely. pos2-chip's
    // Plotter manages its own state, so each plot is a synchronous
    // run_one_plot_cpu() call — no CUDA, no GpuBufferPool.
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
    // There is exactly ONE of it: BatchOptions::include_cpu is a bool, so
    // repeating `cpu` in --devices changes nothing. (It used to claim
    // `--devices cpu,cpu,cpu,cpu` gave four workers. It never did — nothing
    // counted the tokens — so anyone who believed it benchmarked one worker and
    // attributed it to four. The CLI now warns on a repeated `cpu`.) N CPU
    // workers IS a real win — the plotter is memory-latency-bound, so concurrent
    // plots interleave each other's stalls rather than queueing for a core — but
    // each one needs its own copy of the working set (12.14 GiB at k=28), so it
    // needs an explicit flag and a RAM gate, not a bool.
    if (is_cpu_device(device_id)) {
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
        //  * With the CPU as the sole worker, this slice runs on the MAIN THREAD
        //    (run_batch's single-worker fast path calls run_batch_slice inline),
        //    and nicing is irreversible for an unprivileged process — we would
        //    permanently de-prioritise the whole process to yield to nobody.
        //
        //  * With N CPU workers and no GPU, nicing all of them equally yields to
        //    nobody either — it just parks every plotter below the writer pool's
        //    threads, which are the one thing that must NOT outrank them.
        //
        // So the test is "is there a GPU peer", not "is there a peer". It used to
        // be workers.size() > 1, which was the same thing only while a batch could
        // hold at most one CPU worker.
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
                        log_prefix.c_str(), i, entries[i].out_name.c_str());
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
                live_fail(live, worker_id);
                // cuda-only's BatchOptions doesn't have continue_on_error
                // — match the GPU path's behavior of returning early on
                // a per-plot failure (caller decides whether to retry).
                res.total_wall_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                return res;
            }
        }
        res.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return res;
    }

    if (device_id >= 0) bind_current_device(device_id);
    // Must happen AFTER bind_current_device so __constant__ uploads
    // land on this worker's device.
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
    std::unique_ptr<GpuBufferPool> pool_ptr;

    // Start sampling free VRAM before anything allocates, so the baseline is
    // "the card as the tier picker saw it" and peak() is what this run consumed
    // on top. Reported at the end of the run; see the VramWatchdog comment.
    VramWatchdog vram(device_id);

    // Streaming-fallback pinned buffers — double-buffered the same way the
    // pool does, so producer's D2H of plot N+1 can run concurrently with
    // the consumer reading plot N. cudaMallocHost is ~600 ms, so doing it
    // once instead of per plot is a significant win on long batches.
    uint64_t* stream_pinned[GpuBufferPool::kNumPinnedBuffers] = {};
    size_t    stream_pinned_cap = 0;
    // How many of those rotating slots this run actually uses. See
    // drain_slots_env() for why giving one up is safe and what it buys back.
    //
    // The win lands even for a ONE-PLOT run, unlike the pool path's lazy
    // ensure_pinned(): the streaming branch below allocates every slot up
    // front, so the full cost is paid at n=1.
    int num_pinned_slots = GpuBufferPool::kNumPinnedBuffers;
    int const forced_drain_slots = drain_slots_env();      // 0 when unset
    if (forced_drain_slots) num_pinned_slots = forced_drain_slots;
    // Minimal's half of the disk-offload: these two tables become MAP_SHARED
    // file mappings rather than pinned allocations. Set by the budget policy
    // below and consumed at the allocation site; the TempFiles are owned here
    // so the mappings outlive every plot in the slice.
    bool mmap_h_meta = false, mmap_h_t2_xbits = false;
    std::unique_ptr<TempFile> h_meta_map_file, h_t2_xbits_map_file;
    // Tiered streaming scratch. Populated only if the compact tier is
    // selected — see the VRAM dispatch at the end of the catch block.
    StreamingPinnedScratch stream_scratch{};
    bool stream_compact = false;

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
        } else {
            std::fprintf(stderr,
                "%s pool needs %.2f GiB, only %.2f GiB free — using "
                "streaming pipeline per plot\n",
                log_prefix.c_str(),
                e.required_bytes / double(1ULL << 30),
                e.free_bytes     / double(1ULL << 30));
        }
        // Size the pinned buffers using the same cap formula as the pool.
        int const num_section_bits = (pool_k < 28) ? 2 : (pool_k - 26);
        int const extra_margin_bits = 8 - ((28 - pool_k) / 2);
        uint64_t const per_section =
            (1ULL << (pool_k - num_section_bits)) +
            (1ULL << (pool_k - extra_margin_bits));
        uint64_t const cap = per_section * (1ULL << num_section_bits);
        stream_pinned_cap = size_t(cap);
        bool any_fail = false;
        // Only num_pinned_slots of them — the rest stay null, and every free
        // loop below is already null-guarded, so a partial set is safe.
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

        // Tiered dispatch: pick plain vs compact streaming based on
        // free device VRAM. The plain path's peak at k=28 is ~7290 MB;
        // compact drops to ~5200 MB by combining two techniques:
        //   (a) Park/rehydrate on pinned host across idle windows
        //       (d_t1_meta, d_t1_keys_merged, d_t2_meta, d_t2_xbits,
        //        d_t2_keys_merged).
        //   (b) N=2 T2 match tiling: emit T2 into half-cap device
        //       staging + pinned host accumulators, skipping the
        //       full-cap d_t2_meta/mi/xbits peak entirely. Saves
        //       ~2168 MB at k=28 where T2 match is the overall peak.
        // Compact pays ~1-2 s/plot of PCIe round-trips, so we only opt
        // into it when the card can't fit plain.
        //
        // Every floor below is (logical peak at k=28) + 128 MB, where 128 MB is
        // kStreamSafetyBytes in GpuPipeline.cu — the headroom the streaming
        // allocator holds back for the CUDA context's growth after this point.
        // A tier fits iff free >= its logical peak + that margin, so the two
        // constants must agree; the allocator budgets against the driver's real
        // reservation to keep it that way.
        //
        // The peaks are what StreamingStats::peak reports, which is only the
        // truth because the allocator is now budgeted. It did not used to be:
        // the stream-ordered pool reserved 3174 MB beyond the logical peak on
        // plain and 3184 MB on compact (it cannot recycle a cached block for a
        // differently-sized request, and these tiers churn 2080/1040 MB
        // buffers), which no tier floor accounted for and no VRAM trace could
        // see. Cards sized to these floors OOM'd with GBs reserved-but-idle.
        // Do not re-derive a floor from stats.peak without checking it against
        // the pool's reserved_high (POS2GPU_STREAMING_STATS=1 prints both).
        //
        // Each tier's logical peak is the single source of truth: the floor is
        // derived from it, and it is handed to the streaming allocator (as
        // cfg.expected_peak_bytes) to bound how much the CUDA memory pool may
        // cache. Change a peak here and both follow.
        //
        // Peaks measured on sm_89 at k=28 via StreamingStats::peak, each one
        // cross-checked against the pool's reserved_high — see the note above.
        // The margin has to cover two things at once, or a card sitting exactly
        // on a floor does not actually fit:
        //   - kStreamSafetyBytes (128 MB), which the streaming allocator holds
        //     back from its budget for the CUDA context's growth; and
        //   - the allocator's own granularity surplus (physical reservation
        //     over logical bytes), ~30-50 MB at k=28.
        // 128 MB covered only the first, which left compact with 7 MB of slack
        // at its floor and made it fail intermittently. 256 MB covers both with
        // ~78 MB to spare. POS2GPU_STREAMING_STATS=1 prints the physical
        // high-water and the budget, which is how to re-derive this.
        constexpr uint64_t kFloorMarginBytes  = 256ULL * 1024 * 1024;
        constexpr uint64_t kPlainPeakBytes    = 7290ULL * 1024 * 1024;
        constexpr uint64_t kCompactPeakBytes  = 5200ULL * 1024 * 1024;
        constexpr uint64_t kPlainFloorBytes   = kPlainPeakBytes   + kFloorMarginBytes;  // 7546
        constexpr uint64_t kCompactFloorBytes = kCompactPeakBytes + kFloorMarginBytes;  // 5456
        // Minimal tier: compact's pinned-host parking + N=8 T2 match
        // staging (cap/8 vs compact's cap/2). Saves ~1.5 GiB of T2-match
        // peak VRAM at the cost of 6 extra PCIe round-trips during T2
        // match. Targets 4 GiB cards (GTX 1050 Ti / 1650, RTX 3050 4GB,
        // MX450).
        //   minimal: peak 3640 + 256 = 3896 MB floor
        // (3640 is measured; an earlier comment estimated 3760, which left the
        // old 3768 floor an 8 MB margin it only survived by luck.)
        constexpr uint64_t kMinimalPeakBytes  = 3640ULL * 1024 * 1024;
        constexpr uint64_t kMinimalFloorBytes = kMinimalPeakBytes + kFloorMarginBytes;  // 3896
        // Tiny tier: full Phase 1.4 + 1.5 + 1.6 algorithm port,
        // capped off with the Xs gen+sort tiling, T3 sort streaming,
        // and host-pinned d_t3_stage that brought cuda-only Tiny to
        // BYTE-FOR-BYTE PEAK PARITY with the SYCL Tiny tier.
        // Measured at k=28 on RTX 4090: 1064 MB plot peak — EXACTLY
        // matches SYCL Tiny's measured 1064 MB on the same plot_id.
        // Per-phase peaks at k=28: Xs 1030, T1 match 1040, T1 sort
        // 1056, T2 match 1040, T2 sort 1064 (floor), T3 match 1024,
        // T3 sort 1047. All phases ≤ 1064 MB.
        //
        //   tiny: peak 1064 + 256 = 1320 MB floor
        // Was 1100 MB, a 36 MB margin — thinner than the 128 MB the streaming
        // allocator holds back, so Tiny could not in fact run on a card at its
        // own floor: it OOM'd in T2 sort asking for 24 MB. Auto-picker selects
        // Tiny from ~1.3 GB free up to Minimal's 3.9 GB floor. Targets sub-2
        // GiB NVIDIA cards (Quadro P620 2 GB, GTX 1050 2 GB, laptop dGPUs),
        // all of which clear 1320 MB.
        constexpr uint64_t kTinyPeakBytes     = 1064ULL * 1024 * 1024;
        constexpr uint64_t kTinyFloorBytes    = kTinyPeakBytes + kFloorMarginBytes;  // 1320
        size_t const free_bytes = streaming_query_free_vram_bytes();

        // Tier selection: use the effective_tier resolved at the top
        // of run_batch_slice (per-device override > gpu:tier shorthand
        // > global --tier > env), falling back to auto-pick by free
        // VRAM when no override is in effect. The manual overrides
        // bypass the auto-pick threshold but still bail out cleanly
        // if the chosen tier definitely won't fit (Tiny's floor is
        // the hard lower bound — there is no smaller tier; a forced
        // higher tier on a card below that tier's floor warns and
        // proceeds — caller asked).
        std::string const& tier_pref = effective_tier;

        enum class Tier { Plain, Compact, Minimal, Tiny };
        Tier tier;
        if (tier_pref == "plain") {
            tier = Tier::Plain;
        } else if (tier_pref == "compact") {
            tier = Tier::Compact;
        } else if (tier_pref == "minimal") {
            tier = Tier::Minimal;
        } else if (tier_pref == "tiny") {
            tier = Tier::Tiny;
        } else {
            // Auto: pick the largest tier that fits.
            tier = (free_bytes >= kPlainFloorBytes)   ? Tier::Plain   :
                   (free_bytes >= kCompactFloorBytes) ? Tier::Compact :
                   (free_bytes >= kMinimalFloorBytes) ? Tier::Minimal :
                                                        Tier::Tiny;
        }

        // HOST-RAM gate. Every check above weighs VRAM, and host RAM runs the
        // other way: the lower the tier, the more full-cap tables it parks in
        // pinned host memory. So the card small enough to need Tiny usually
        // sits in the box that can least afford Tiny's host staging, and
        // weighing VRAM alone is how a modest machine ends up inviting the OOM
        // killer instead of being told no. Fail here, before the setup cost,
        // rather than part-way through the first plot.
        auto const tier_label = [](Tier t) -> char const* {
            switch (t) {
                case Tier::Plain:   return "plain";
                case Tier::Compact: return "compact";
                case Tier::Minimal: return "minimal";
                case Tier::Tiny:    break;
            }
            return "tiny";
        };
        auto const host_need = [pool_k](Tier t) -> std::size_t {
            switch (t) {
                case Tier::Plain:   return streaming_plain_host_bytes(pool_k);
                case Tier::Compact: return streaming_compact_host_bytes(pool_k);
                case Tier::Minimal: return streaming_minimal_host_bytes(pool_k);
                case Tier::Tiny:    break;
            }
            return streaming_tiny_host_bytes(pool_k);
        };
        {
            std::size_t const host_required = host_need(tier);
            std::size_t const host_reserve  = streaming_host_reserve();
            std::size_t host_free = 0, host_total = 0;
            if (host_memory_probe(host_free, host_total)) {
                auto const gib = [](std::size_t b) {
                    return b / double(1ULL << 30);
                };
                // Worth saying out loud when a FORCED tier costs more host RAM
                // than the auto-pick would have: --tier tiny is the obvious
                // reach for "use less memory" and is the single worst choice
                // for host memory.
                bool const tier_forced =
                    (tier_pref == "plain" || tier_pref == "compact" ||
                     tier_pref == "minimal" || tier_pref == "tiny");
                if (tier_forced) {
                    Tier const auto_tier =
                        (free_bytes >= kPlainFloorBytes)   ? Tier::Plain   :
                        (free_bytes >= kCompactFloorBytes) ? Tier::Compact :
                        (free_bytes >= kMinimalFloorBytes) ? Tier::Minimal :
                                                             Tier::Tiny;
                    std::size_t const auto_host = host_need(auto_tier);
                    if (tier != auto_tier && host_required > auto_host) {
                        std::fprintf(stderr,
                            "%s --tier %s needs ~%.2f GiB of HOST RAM; %s also "
                            "fits this GPU and needs ~%.2f GiB. Lower tiers buy "
                            "VRAM WITH host memory — they do not reduce memory "
                            "overall.\n",
                            log_prefix.c_str(), tier_label(tier),
                            gib(host_required), tier_label(auto_tier),
                            gib(auto_host));
                    }
                }
                // ---- Host-RAM disk-offload: spill rather than refuse ----
                //
                // An explicit --max-host-ram always runs the policy. Otherwise
                // it engages ONLY when the tier does not fit, which is the
                // asymmetry that makes auto-spill safe on by default: it fires
                // exactly where the alternative is a hard error today, so it
                // cannot slow down a run that already works.
                //
                // Compact is the only tier that can route a table (Minimal and
                // Tiny CPU-touch both of them), so on the others this reduces
                // to the drain-slot lever and then stands down.
                bool const short_on_ram =
                    host_free < host_required + host_reserve;
                bool want_policy =
                    opts.max_host_ram_bytes > 0 ||
                    (short_on_ram && !opts.no_auto_spill);

                // A RAM-backed temp dir makes the whole budget a lie: on most
                // systemd distributions /tmp is tmpfs, so "spilling" there
                // moves the tables from one part of RAM to another and invites
                // the OOM killer the budget exists to avoid. It defeats BOTH
                // classes — a MAP_SHARED mapping over a tmpfs file is
                // anonymous memory with extra steps, so Minimal's mmap route
                // is no safer than Compact's engine route. Probe the same dir
                // the spill TempFiles will resolve (--temp-dir feeds
                // XCHPLOT2_TEMP_DIR, already setenv'd by run_batch).
                //
                // An EXPLICIT budget throws: the user asked for a spill and
                // needs to know it would not have been one. AUTO must not —
                // they asked for a plot, not a spill, so a tmpfs temp dir just
                // means this rescue is unavailable. Record why and let the
                // host-RAM guard below deliver the verdict with that clause
                // attached, so they get both facts at once.
                bool        ram_dir_blocked_auto = false;
                std::string ram_dir_path;
                if (want_policy) {
                    std::string const spill_dir = TempFile::resolve_dir("");
                    if (TempFile::dir_is_ram_backed(spill_dir)) {
                        char const* ov =
                            std::getenv("XCHPLOT2_ALLOW_RAM_TEMP_DIR");
                        std::string const ovs = ov ? ov : "";
                        bool const allow = (ovs == "1" || ovs == "true" ||
                                            ovs == "yes" || ovs == "on");
                        if (allow) {
                            std::fprintf(stderr,
                                "%s WARNING: spill temp dir '%s' is on a "
                                "RAM-backed filesystem (tmpfs); proceeding "
                                "anyway because XCHPLOT2_ALLOW_RAM_TEMP_DIR "
                                "is set. The host-RAM budget below does NOT "
                                "account for what the spill writes there.\n",
                                log_prefix.c_str(), spill_dir.c_str());
                        } else if (opts.max_host_ram_bytes > 0) {
                            throw std::runtime_error(
                                "--max-host-ram is set but the spill temp dir "
                                "'" + spill_dir + "' is on a RAM-backed "
                                "filesystem (tmpfs); spilling there consumes "
                                "the RAM the budget exists to cap. Point "
                                "--temp-dir at real disk, or set "
                                "XCHPLOT2_ALLOW_RAM_TEMP_DIR=1 if this path "
                                "really is disk-backed.");
                        } else {
                            want_policy          = false;
                            ram_dir_blocked_auto = true;
                            ram_dir_path         = spill_dir;
                        }
                    }
                }
                if (want_policy) {
                    // Sentinel translation. BatchOptions uses 0 for "not set"
                    // and 1 for "min"; the policy uses 0 for "min". Without
                    // this, --max-host-ram min reads as a 1-byte budget, the
                    // policy correctly reports it as unreachable, and the run
                    // fails having just built the exact plan the user asked
                    // for.
                    bool const budget_is_min = (opts.max_host_ram_bytes == 1);

                    HostRamSpillInputs pin;
                    pin.host_required  = host_required;
                    pin.cap_entries    = stream_pinned_cap;
                    pin.budget         = budget_is_min
                        ? 0 : opts.max_host_ram_bytes;
                    pin.tier_compact   = (tier == Tier::Compact);
                    pin.tier_minimal   = (tier == Tier::Minimal);
                    pin.pinned_slots   = num_pinned_slots;
                    pin.forced_slots   = (forced_drain_slots != 0);
                    pin.baseline_slots = GpuBufferPool::kNumPinnedBuffers;

                    // Auto adopts the host's own free RAM as the budget.
                    if (opts.max_host_ram_bytes == 0 && short_on_ram) {
                        pin.budget = (host_free > host_reserve)
                            ? std::uint64_t(host_free - host_reserve) : 0;
                    }

                    HostRamSpillPlan const sp = plan_host_ram_spill(pin);

                    // An EXPLICIT budget that cannot be met is the user's
                    // error and throws below with the floor quoted. An
                    // automatic one just stands down and lets the host-RAM
                    // guard speak, unchanged.
                    if (sp.meets_budget || opts.max_host_ram_bytes > 0) {
                        stream_scratch.spill = sp.tables;
                        num_pinned_slots     = sp.pinned_slots;
                        mmap_h_meta          = sp.mmap_h_meta;
                        mmap_h_t2_xbits      = sp.mmap_h_t2_xbits;
                        bool const any_mmap =
                            sp.mmap_h_meta || sp.mmap_h_t2_xbits;
                        if (sp.tables.any() || any_mmap || sp.drain_freed) {
                            std::fprintf(stderr,
                                "%s host-RAM budget: modelled peak %.2f -> "
                                "%.2f GiB (floor %.2f) — %s%s%s%s%s%s%d drain "
                                "slot%s; ~%.2f GiB temp-dir traffic per plot "
                                "(%.2f W / %.2f R), %.2f GiB reclaimable\n",
                                log_prefix.c_str(), gib(host_required),
                                gib(sp.resident), gib(sp.floor_bytes),
                                sp.tables.h_meta ? "h_meta " : "",
                                sp.tables.h_t2_xbits ? "h_t2_xbits " : "",
                                sp.tables.any() ? "-> disk, " : "",
                                sp.mmap_h_meta ? "h_meta " : "",
                                sp.mmap_h_t2_xbits ? "h_t2_xbits " : "",
                                any_mmap ? "-> mmap, " : "",
                                sp.pinned_slots,
                                sp.pinned_slots == 1 ? "" : "s",
                                gib(sp.traffic_written + sp.traffic_read),
                                gib(sp.traffic_written), gib(sp.traffic_read),
                                gib(sp.reclaimable));
                        }
                        if (!sp.meets_budget) {
                            char m2[512];
                            std::snprintf(m2, sizeof(m2),
                                "%s --max-host-ram %.2f GiB is below what this "
                                "tier can reach: the floor is %.2f GiB at k=%d "
                                "with every routable table on disk and a single "
                                "drain slot.",
                                log_prefix.c_str(),
                                gib(opts.max_host_ram_bytes),
                                gib(sp.floor_bytes), pool_k);
                            throw std::runtime_error(m2);
                        }
                        // The spill covers the shortfall; skip the refusal.
                        if (short_on_ram) goto host_ram_ok;
                    }
                }

                if (host_free < host_required + host_reserve) {
                    // The auto-spill would have rescued this run; say why it
                    // did not, or the user reads a plain out-of-RAM refusal on
                    // a build that advertises a disk-offload and concludes the
                    // feature is broken rather than misconfigured.
                    char ramdir[512] = {0};
                    if (ram_dir_blocked_auto) {
                        std::snprintf(ramdir, sizeof(ramdir),
                            " The automatic disk-offload could have covered "
                            "this, but stood down because the temp dir '%s' is "
                            "on a RAM-backed filesystem (tmpfs) — spilling "
                            "there would consume the RAM it is trying to save. "
                            "Point --temp-dir at real disk to enable it.",
                            ram_dir_path.c_str());
                    }
                    char msg[1600];
                    std::snprintf(msg, sizeof(msg),
                        "%s streaming tier %s needs ~%.2f GiB of HOST RAM at "
                        "k=%d plus a %.2f GiB reserve; host reports %.2f GiB "
                        "available of %.2f GiB total. This is host memory, not "
                        "VRAM — and a LOWER --tier costs MORE of it, not less: "
                        "plain needs the least (~%.2f GiB), tiny the most "
                        "(~%.2f GiB). Close what else is holding RAM, or plot "
                        "on a host with more. (XCHPLOT2_HOST_RESERVE_MB tunes "
                        "the reserve.)%s",
                        log_prefix.c_str(), tier_label(tier),
                        gib(host_required), pool_k, gib(host_reserve),
                        gib(host_free), gib(host_total),
                        gib(streaming_plain_host_bytes(pool_k)),
                        gib(streaming_tiny_host_bytes(pool_k)), ramdir);
                    throw std::runtime_error(msg);
                }
host_ram_ok:;
            }
        }

        // Hand the chosen tier's working set to the streaming allocator. It
        // caches only what the card has beyond this, so a big card keeps the
        // memory pool's cross-plot reuse while a card sized to its floor holds
        // the pool to the working set — which is what the floors assume.
        stream_scratch.expected_peak_bytes =
            (tier == Tier::Plain)   ? kPlainPeakBytes   :
            (tier == Tier::Compact) ? kCompactPeakBytes :
            (tier == Tier::Minimal) ? kMinimalPeakBytes :
                                      kTinyPeakBytes;

        // Forced-tier fit warnings. Forced tiers below their floor are
        // allowed (caller's risk) — except Tiny below its floor still
        // throws because there's no smaller tier to fall back to.
        if (tier == Tier::Plain && free_bytes < kPlainFloorBytes) {
            std::fprintf(stderr,
                "%s streaming tier: plain forced (%.2f GiB free < %.2f GiB "
                "plain floor) — proceeding, may OOM mid-plot\n",
                log_prefix.c_str(),
                free_bytes / double(1ULL << 30),
                kPlainFloorBytes / double(1ULL << 30));
        } else if (tier == Tier::Compact && free_bytes < kCompactFloorBytes) {
            std::fprintf(stderr,
                "%s streaming tier: compact forced (%.2f GiB free < %.2f GiB "
                "compact floor) — proceeding, may OOM mid-plot\n",
                log_prefix.c_str(),
                free_bytes / double(1ULL << 30),
                kCompactFloorBytes / double(1ULL << 30));
        } else if (tier == Tier::Minimal && free_bytes < kMinimalFloorBytes) {
            std::fprintf(stderr,
                "%s streaming tier: minimal forced (%.2f GiB free < %.2f GiB "
                "minimal floor) — proceeding, may OOM mid-plot\n",
                log_prefix.c_str(),
                free_bytes / double(1ULL << 30),
                kMinimalFloorBytes / double(1ULL << 30));
        }

        // Record what was actually picked, so a two-pass caller can tell
        // whether its two passes are comparable. See WorkerTimeline.
        res.pipeline = (tier == Tier::Plain)   ? "plain"
                     : (tier == Tier::Compact) ? "compact"
                     : (tier == Tier::Minimal) ? "minimal"
                     : (tier == Tier::Tiny)    ? "tiny"
                                               : "";

        if (tier == Tier::Plain) {
            // Plain: zero PCIe overhead, all parks skipped.
            if (!opts.quiet) {
                std::fprintf(stderr,
                    "%s streaming tier: plain (%.2f GiB free, %.2f GiB floor)\n",
                    log_prefix.c_str(),
                    free_bytes / double(1ULL << 30),
                    kPlainFloorBytes / double(1ULL << 30));
            }
        } else if (tier == Tier::Compact || tier == Tier::Minimal ||
                   tier == Tier::Tiny) {
            // Compact + Minimal share the same pinned-host scratch
            // (h_meta / h_keys_merged / h_t2_xbits, ~4.2 GB at k=28).
            // Both also set t3_tile_count = 2: T3 match emits into a
            // half-cap d_t3 staging buffer and accumulates into h_meta
            // (its T2-meta park lifetime ends at the meta gather above),
            // dropping the T3-match peak from 6240 MiB → 5200 MiB at
            // k=28 so compact gets back under its sub-6 GiB design
            // target. Minimal additionally sets t2_tile_count = 8 (vs
            // compact's default 2) so T2 match staging shrinks from
            // ~2.3 GB to ~570 MB.
            // Host-RAM disk-offload: when h_t2_xbits is routed to a temp dir
            // the pipeline services every access through its SpillBuffer, so
            // the cap-sized pinned buffer must NOT be allocated — allocating
            // it anyway would route correctly and save nothing, which is the
            // whole point of the feature. Compact only; the pipeline throws on
            // the tiers that would CPU-touch the table.
            bool const spill_t2_xbits =
                stream_scratch.spill.h_t2_xbits ||
                [] { char const* v = std::getenv("XCHPLOT2_SPILL_T2XBITS");
                     return v && v[0] == '1'; }();
            // h_meta is the big one — cap*8, 2.03 GiB at k=28 — and the
            // pipeline routes all three of its Compact roles (T1 meta park, T2
            // meta park, T3 accumulator) through one SpillBuffer each.
            bool const spill_h_meta =
                stream_scratch.spill.h_meta ||
                [] { char const* v = std::getenv("XCHPLOT2_SPILL_HMETA");
                     return v && v[0] == '1'; }();
            stream_scratch.spill.h_t2_xbits = spill_t2_xbits;
            stream_scratch.spill.h_meta     = spill_h_meta;
            stream_scratch.quiet            = opts.quiet;

            // Three ways a table can be backed now: pinned (the default),
            // absent because the pipeline routes it through a SpillBuffer
            // (Compact), or a MAP_SHARED file mapping (Minimal). The mapping is
            // just a host pointer, so nothing downstream changes — CPU indexing
            // and cudaMemcpyAsync both work on it.
            //
            // NB the saving from a mapping does NOT show in ru_maxrss on a box
            // with free RAM: the pages stay resident until something needs the
            // memory. That is the point — they are RECLAIMABLE, not absent —
            // but it means this arm has to be judged on a host under pressure,
            // not on a 125 GiB dev box.
            if (mmap_h_meta) {
                h_meta_map_file = std::make_unique<TempFile>();
                stream_scratch.h_meta = static_cast<std::uint64_t*>(
                    h_meta_map_file->map(stream_pinned_cap * sizeof(std::uint64_t)));
            } else if (!spill_h_meta) {
                stream_scratch.h_meta = streaming_alloc_pinned_uint64(stream_pinned_cap);
            }
            stream_scratch.h_keys_merged = streaming_alloc_pinned_uint32(stream_pinned_cap);
            if (mmap_h_t2_xbits) {
                h_t2_xbits_map_file = std::make_unique<TempFile>();
                stream_scratch.h_t2_xbits = static_cast<std::uint32_t*>(
                    h_t2_xbits_map_file->map(stream_pinned_cap * sizeof(std::uint32_t)));
            } else if (!spill_t2_xbits) {
                stream_scratch.h_t2_xbits = streaming_alloc_pinned_uint32(stream_pinned_cap);
            }
            if ((!spill_h_meta && !stream_scratch.h_meta) ||
                !stream_scratch.h_keys_merged ||
                (!spill_t2_xbits && !stream_scratch.h_t2_xbits))
            {
                if (stream_scratch.h_meta && !mmap_h_meta)
                    streaming_free_pinned_uint64(stream_scratch.h_meta);
                if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
                if (stream_scratch.h_t2_xbits && !mmap_h_t2_xbits)
                    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
                for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                    if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
                }
                throw std::runtime_error(
                    log_prefix + " streaming-fallback: compact/minimal pinned scratch alloc failed");
            }
            stream_compact = true;
            stream_scratch.t3_tile_count = 2;
            // Tiny tier: scaffolding only. Currently sets all Minimal
            // flags + tiny_mode (consumed by GpuPipeline.cu when the
            // per-Phase wiring lands — see [project_cuda_only_tiny_port]).
            // Until then, Tiny produces byte-identical output to Minimal
            // since the tiny_mode flag isn't yet acted on. Lets the
            // tier-picker scaffolding land + validate the workflow
            // before the algorithm changes.
            if (tier == Tier::Tiny) {
                stream_scratch.tiny_mode = true;
            }
            if (tier == Tier::Minimal || tier == Tier::Tiny) {
                stream_scratch.t2_tile_count = 8;
                // Cuts #1+#2: tile T1/T2 sort gathers through pinned host so
                // the cap-sized sorted_meta / sorted_xbits never co-reside
                // with the unsorted-meta + merged_vals on device. N=4 = one
                // tile per section_l at k=28 strength=2; tile size cap/4 ≈
                // 520 MB at k=28 — same envelope as the t2 stage tile.
                stream_scratch.gather_tile_count = 4;
                // Cut #3: T3 match section-pair input slicing. Equals
                // num_sections (= (1<<2) at k=28 strength=2 = 4); the T3
                // match phase iterates section_l ∈ [0, num_sections) and
                // H2Ds the section_l + section_r row slices per pass
                // instead of holding the cap-sized d_t2_meta_sorted on
                // device. Drops T3 match peak from ~5200 → ~3700 MB.
                int const num_section_bits = (pool_k < 28) ? 2 : (pool_k - 26);
                stream_scratch.t3_input_slice_count = 1 << num_section_bits;
                if (opts.quiet) {
                    // info-level tier note suppressed
                } else if (tier == Tier::Tiny) {
                    std::fprintf(stderr,
                        "%s streaming tier: tiny (%.2f GiB free, %.2f GiB floor; "
                        "per-bucket-pair T1/T2/T3 match + streaming-partition T1/T2/T3 sort "
                        "+ host-prepare T2/T3 offsets + d_t3_stage/d_frags_out host alias "
                        "+ Xs gen+sort tiling, ~17 s/plot extra PCIe vs minimal at k=28)\n",
                        log_prefix.c_str(),
                        free_bytes / double(1ULL << 30),
                        kTinyFloorBytes / double(1ULL << 30));
                } else {
                    std::fprintf(stderr,
                        "%s streaming tier: minimal (%.2f GiB free, %.2f GiB floor; "
                        "park/rehydrate + N=8 T2 + N=%d T1-match + T1/T2 sort gather + "
                        "N=%d T3 input slicing, expect ~5-15 s/plot extra PCIe)\n",
                        log_prefix.c_str(),
                        free_bytes / double(1ULL << 30),
                        kMinimalFloorBytes / double(1ULL << 30),
                        stream_scratch.t3_input_slice_count,
                        stream_scratch.t3_input_slice_count);
                }
            } else if (!opts.quiet) {
                std::fprintf(stderr,
                    "%s streaming tier: compact (%.2f GiB free < %.2f GiB plain floor; "
                    "park/rehydrate + N=2 T3 staging, expect ~1-2 s/plot extra PCIe)\n",
                    log_prefix.c_str(),
                    free_bytes / double(1ULL << 30),
                    kPlainFloorBytes / double(1ULL << 30));
            }
        } else {
            // Unreachable — the auto-pick branch above always picks one
            // of Plain/Compact/Minimal regardless of free_bytes (Minimal
            // is the open-ended fallback). Kept for switch-completeness.
            for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
            }
            throw std::runtime_error(log_prefix + " internal: unhandled streaming tier");
        }
        // Forced-tiny hard floor: there's no smaller tier to fall back
        // to, so a card below the tiny floor genuinely can't plot at
        // this k. Bail with a clear message.
        if (tier == Tier::Tiny && free_bytes < kTinyFloorBytes) {
            for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
            }
            if (stream_scratch.h_meta && !mmap_h_meta)
                    streaming_free_pinned_uint64(stream_scratch.h_meta);
            if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
            if (stream_scratch.h_t2_xbits && !mmap_h_t2_xbits)
                    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
            throw std::runtime_error(
                log_prefix + " card too small for k=" + std::to_string(pool_k) +
                " streaming at any tier: " +
                std::to_string(free_bytes / (1ULL << 20)) + " MB free < " +
                std::to_string(kTinyFloorBytes / (1ULL << 20)) +
                " MB tiny floor. Use a smaller k or a larger GPU "
                "(or --cpu for pos2-chip CPU plotting).");
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
            scratch->h_meta = nullptr;
            scratch->h_keys_merged = nullptr;
            scratch->h_t2_xbits = nullptr;
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
    // Floored at 1: with a single slot the arithmetic depth is 0, and a
    // zero-capacity Channel can never accept a push — the producer would block
    // forever on the first plot. Depth 1 is safe because slot_gate, not the
    // channel depth, is what actually serialises slot reuse.
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

    // Consumer: takes finished GpuPipelineResults and writes plot files.
    std::thread consumer([&] {
        try {
            WorkItem item;
            while (chan.pop(item)) {
                std::filesystem::create_directories(item.entry.out_dir);
                auto full_path = std::filesystem::path(item.entry.out_dir) / item.entry.out_name;

                std::vector<uint8_t> memo_bytes = item.entry.memo;
                if (memo_bytes.empty()) memo_bytes.assign(32 + 48 + 32, 0);

                // Fragments are borrowed from the pool's pinned slot;
                // wait for any overlapped D2H to land before reading,
                // then the SlotGate ack below lets the producer reuse
                // that slot once we're done.
                wait_pipeline_d2h(item.result);
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
                // Retires the claim the producer took when it pulled this entry,
                // and publishes this worker's rate for the ETA. The line it feeds
                // is aggregate across every worker, hence progress_prefix.
                std::size_t const done_now = record_plot_completion(
                    res, plot_bytes, completion_offset, live, worker_id);
                if (verbose) {
                    std::fprintf(stderr, "%s consumer wrote plot %zu: %s\n",
                                 log_prefix.c_str(), item.index, full_path.string().c_str());
                }
                if (opts.progress) {
                    emit_progress_line(
                        progress_prefix, opts, live, done_now,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_start).count());
                }
                // Done reading this item's pinned slot — let the
                // producer reuse it.
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

    // Producer (this thread): drives the GPU pipeline, hands off to consumer.
    // local_count rotates this worker's own pinned-buffer slots (channel
    // depth = num_pinned_slots); it must NOT use the global plot index
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
            // wait on a plot nobody is making. The consumer retires it.
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

            auto t_plot = std::chrono::steady_clock::now();

            GpuPipelineConfig cfg;
            cfg.plot_id  = entries[i].plot_id;
            cfg.k        = entries[i].k;
            cfg.strength = entries[i].strength;
            cfg.testnet  = entries[i].testnet;
            cfg.profile  = false;

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
                // GPU work completed but will never be written. The
                // consumer's exception is rethrown below.
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
    {
        (void)stream_compact;  // avoid unused-warning on plain-only builds
    }

    // VRAM watchdog: what the driver saw vs what we declared.
    //
    // The bound is only asserted on the POOLED path, and deliberately so. The
    // pool sizes every buffer up front and run_gpu_pipeline allocates nothing at
    // runtime, so its declaration is exact and any excess is a real bug. The
    // streaming path runs under the budgeted allocator in GpuPipeline.cu, which
    // is *supposed* to let the async pool cache freely on a card with headroom —
    // its peak legitimately exceeds the tier model there, and it enforces its own
    // bound (POS2GPU_ASSERT_VRAM + POS2GPU_STREAMING_STATS=1 for the breakdown).
    // Asserting a driver-level bound on it would fire on every roomy card.
    // -q suppresses the routine line, never the warning or the assert: a run
    // that quietly skipped its own VRAM check would be worse than useless.
    vram.stop();
    if (vram.available()) {
        auto to_mib = [](uint64_t b) { return b / double(1ULL << 20); };
        uint64_t const peak = vram.peak();

        if (pool_ptr) {
            uint64_t const declared = pool_ptr->required_device_bytes
                                    + pool_ptr->frags_dedicated_bytes
                                    + vram_safety_margin();
            if (!opts.quiet) {
                std::fprintf(stderr,
                    "%s vram: peak %.0f MiB of %.0f free "
                    "(pool %.0f + frags %.0f + margin %.0f = %.0f declared)\n",
                    log_prefix.c_str(), to_mib(peak), to_mib(vram.baseline()),
                    to_mib(pool_ptr->required_device_bytes),
                    to_mib(pool_ptr->frags_dedicated_bytes),
                    to_mib(vram_safety_margin()), to_mib(declared));
            }

            // A small excess over `declared` is CUDA-runtime overhead the pool
            // cannot pre-size — CUB scratch, module loads, cudaMalloc rounding —
            // sitting just past the safety margin. It is not a run_gpu_pipeline bug
            // and not worth a word on a normal run (nor a thrown bench). Only flag a
            // breach past a SECOND margin's worth: that is the scale of a real
            // under-declaration, the kind that OOMs a card sized from the model
            // (historically GBs, not tens of MiB). The picker still gates on
            // `declared`; this slack is only the noise floor for the complaint.
            // bench sets POS2GPU_ASSERT_VRAM (cli.cpp), so this check doubles as the
            // regression test — with the slack it fires on real drift, not runtime
            // jitter.
            uint64_t const slack = vram_safety_margin();
            if (peak > declared + slack) {
                std::fprintf(stderr,
                    "%s vram: %s — pooled path peaked at %.0f MiB, %.0f MiB over the "
                    "%.0f MiB declared (pool + frags + margin) — past the margin, so "
                    "the model is under-declaring and a card sized from it may OOM. "
                    "Look for a device allocation outside the pool.\n",
                    log_prefix.c_str(),
                    assert_vram_enabled() ? "FATAL" : "WARNING",
                    to_mib(peak), to_mib(peak - declared), to_mib(declared));
                if (assert_vram_enabled()) {
                    throw std::runtime_error(
                        "POS2GPU_ASSERT_VRAM: pooled path exceeded its declared VRAM "
                        "by more than the safety margin");
                }
            }
        } else if (!opts.quiet) {
            std::fprintf(stderr,
                "%s vram: peak %.0f MiB of %.0f free (streaming; the budgeted "
                "allocator caches spare VRAM by design — POS2GPU_STREAMING_STATS=1 "
                "for the per-phase breakdown)\n",
                log_prefix.c_str(), to_mib(peak), to_mib(vram.baseline()));
        }
    }

    res.plots_written = plots_done.load();
    res.total_wall_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count();

    // Give the stream-ordered allocator's cache back to the driver before this
    // slice reports done. Every streaming device allocation is already freed —
    // logically. Physically the pool keeps holding those pages, because its
    // release threshold only acts at the next synchronization point, and the
    // next thing a caller does may well be to READ free VRAM: bench's second
    // pass, or a supervisor sizing the next job. It then measures the card
    // minus our cache and sizes down accordingly. Measured on a 4090 at k=28
    // squeezed to 9 GiB free: pass 1 saw 9.06 GiB and picked plain, pass 2 saw
    // 6.97 GiB and picked compact — a tier apart, on an idle unchanged card.
    //
    // This runs on the slice's own thread, which bind_current_device() bound to
    // this slice's device (~line 1752), so a multi-device run trims each card
    // from the worker that filled it. Not in a scope guard on purpose: on the
    // throwing path the batch has failed and nobody is about to size anything
    // from its leftovers, and the guard would have to outlive the pool object
    // to be correct for the pooled path too.
    streaming_release_cached_vram();
    return res;
}

// Phase 1 scaffold: opt-in single-plot multi-GPU dispatch. The
// design proper lives in docs/multi-gpu-single-plot-*.md; this entry
// point reserves the API and errors clearly for N > 1 until the
// partition + multi-GPU sort machinery lands in Phase 2.
//
// Caller invariants (enforced in run_batch before this is called):
//   - opts.shard_plot is true
//   - device_ids.size() > 1
//   - entries is non-empty
BatchResult run_batch_sharded(std::vector<BatchEntry> const& entries,
                              BatchOptions const& opts,
                              std::vector<int> const& device_ids)
{
    (void)entries;
    (void)opts;
    char const* strategy = opts.shard_strategy.empty()
        ? "bucket"
        : opts.shard_strategy.c_str();
    throw std::runtime_error(std::string(
        "--shard-plot is currently a scaffold (Phase 1): the dispatch "
        "wiring is in place but the multi-GPU sharded pipeline "
        "(strategy='") + strategy + "', " +
        std::to_string(device_ids.size()) +
        " devices) hasn't been implemented yet. To run the existing "
        "multi-plot work-queue (one plot per device, round-robin), "
        "drop the --shard-plot flag — `--devices all` or `--devices "
        "0,1,...` keep working as before. "
        "See docs/multi-gpu-single-plot-*.md for the planned design.");
}

} // namespace

namespace {
// Defined with the rest of the RAM-gate helpers at the bottom of this file, next
// to resolve_batch_devices — the public API they exist to serve. run_batch needs
// it up here to build the live gate, and on this branch resolve_batch_devices
// happens to sit BELOW run_batch (on main it is above, so main needs no
// declaration). Same anonymous namespace either way, so this is the same entity.
std::uint64_t cpu_budget_bytes(int k, std::size_t gpu_count);
}  // namespace

BatchResult run_batch(std::vector<BatchEntry> const& entries,
                      BatchOptions const& opts)
{
    if (entries.empty()) return BatchResult{};

    // --temp-dir reaches TempFile through the environment rather than a
    // parameter: TempFile already resolves XCHPLOT2_TEMP_DIR first, it is
    // reached from call sites that have no BatchOptions in scope, and this runs
    // once on the entry thread before any worker exists. setenv copies the
    // string, so passing c_str() of a member is safe.
    //
    // Validated UP FRONT, not at first use. dir_is_ram_backed() returns false
    // when it cannot even statfs, so a mistyped path would sail through the
    // tmpfs guard and die on a raw mkstemp errno minutes into a batch — and the
    // tmpfs message is exactly what tells a user to reach for this flag.
    if (!opts.temp_dir.empty()) {
        ::setenv("XCHPLOT2_TEMP_DIR", opts.temp_dir.c_str(), /*overwrite=*/1);
        std::string const problem = TempFile::dir_problem(opts.temp_dir);
        if (!problem.empty()) {
            throw std::runtime_error("--temp-dir " + opts.temp_dir + ": " + problem);
        }
    }

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

    // Disk-space preflight (advisory, never throws). Runs once before
    // any worker dispatch so the user sees the warning up front instead
    // of N minutes into a doomed batch.
    preflight_disk_space(entries, opts);

    // Resolve the target device list (see resolve_batch_devices):
    //   use_all_devices  → enumerate at runtime, one worker per GPU
    //   device_ids       → use these explicit ids
    //   (neither)        → empty list → single-device, CUDA-default device
    //   cpu_workers      → orthogonal: append that many kCpuDeviceId
    //                      entries so the CPU runs as N more workers.
    //                      Mixes with the above (--cpu alone → CPU only;
    //                      --cpu --devices all → all GPUs + CPU; etc.).
    //                      The count is capped at what host RAM holds.
    std::string gate_note;
    std::vector<int> const device_ids =
        resolve_batch_devices(opts, pool_k, &gate_note);
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

    // The gate can take the count to zero. If the user EXPLICITLY asked for CPU
    // workers (a positive N, or `max`) and none fit AND no GPU was selected, say
    // so — the fast path below would otherwise read the empty list as "no device
    // selected", fall back to the CUDA-default device, and either plot on an
    // unrequested GPU or fail obscurely.
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
            "falling back to the CUDA-default device\n");
    }

    auto const t_start = std::chrono::steady_clock::now();

    // Single-worker strategies below run one plot at a time, so their whole
    // timeline belongs to one worker: publish it as such. Nothing is compared
    // across workers when there is only one, so their own steady_clock origin
    // is fine as the epoch — only the work-queue fan-out needs a shared one.
    auto as_single_worker = [](BatchResult& r, int dev) {
        WorkerTimeline w;
        w.device_id          = dev;
        w.work_start_seconds = r.work_start_seconds;
        w.completion_seconds = r.completion_seconds;
        w.pipeline           = r.pipeline;
        r.workers.assign(1, std::move(w));
    };

    // Single-plot-multi-GPU dispatch (opt in via --shard-plot). Each
    // plot runs across all selected devices as a "team" instead of
    // distributing plots between independent workers. Phase 1 ships
    // the surface area only — N=1 falls through to the existing
    // single-GPU path (a no-op equivalent), N > 1 throws a clear
    // error until Phase 2 lands the spatial / bucket partition.
    // See docs/multi-gpu-single-plot-*.md.
    if (opts.shard_plot && device_ids.size() > 1) {
        BatchResult r = run_batch_sharded(entries, opts, device_ids);
        r.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        as_single_worker(r, device_ids[0]);  // the team plots as one worker
        return r;
    }

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

    // Fast path: zero-config default or one explicit id. Runs on the
    // caller thread — identical control flow to pre-multi-GPU except
    // for the optional cudaSetDevice at the top of the slice.
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
    // (the CUDA-default sentinel never reaches the multi-device path).
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
// Both curves below are measured, not modelled: VmHWM out of /proc, the kernel's
// own high-water mark, polled to process exit on a 32-thread 5950X + RTX 4090.
// ---------------------------------------------------------------------------

namespace {

// Host RSS one GPU worker adds — its pinned buffers, fragment buffers, FSE
// scratch, CUDA context and the binary itself. Not device VRAM: this is the host
// memory a GPU worker takes AWAY from the CPU workers.
//
//   k=22    345 668 kB      (model says 350 396 — +1.4%)
//   k=26  1 583 524 kB
//   k=28  5 529 348 kB      (5.27 GiB)
//
// Fits A·2^k + B to within 1.4% across all three, with A = 20.07 B/entry and
// B = 262 MB (context + binary + buffers, none of which scale with k).
//
// Measured on a 24 GB card, which auto-picks the largest streaming tier and so
// the largest host-side footprint. A smaller card picks a smaller tier and needs
// less than this — over-estimating the reserve costs at most one CPU worker, and
// that is the direction to err in.
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
// itself: the CPU workers' own 12 GiB apiece is subtracted from MemAvailable the
// moment they start, so asking again mid-run answers a different question than
// the one the gate is for ("how many can I start?", not "how much is left now?").
std::uint64_t host_free_bytes_once()
{
    static std::uint64_t const cached = host_free_bytes_now();
    return cached;
}

// Slack between "the kernel says this much is available" and "starting another
// 12 GiB allocation right now is a good idea". MemAvailable already excludes what
// the kernel wants to keep and already counts reclaimable page cache, so this is
// not a second guess at the same thing — it is headroom for everything on the box
// that is not us.
constexpr std::uint64_t kHostSlackBytes = 1ULL << 30;  // 1 GiB

// Host RAM the user wants left alone — XCHPLOT2_CPU_RESERVE_MB.
//
// The gate's job is to stop the CPU workers OOMing the box. But "the box has 73
// GiB free" and "you may take 73 GiB" are different claims: plenty of people plot
// on the machine they also work on, and would rather keep 16 GB for the thing they
// are actually doing than discover, an hour in, that their editor got swapped out.
std::uint64_t host_reserve_bytes()
{
    if (char const* v = std::getenv("XCHPLOT2_CPU_RESERVE_MB"); v && v[0]) {
        long const mb = std::atol(v);
        if (mb > 0) return static_cast<std::uint64_t>(mb) << 20;
    }
    return 0;
}

// How many bytes the CPU workers may collectively hold, at batch start.
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
//   kCpuWorkersMax  (-2): as many as RAM holds, capped at the core count.
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
    // number, so there is no fudge factor on it. (The safety lives in the reserve,
    // which is the term that is genuinely uncertain.)
    //
    // It is ~2^k, but not a clean power of two: the coefficient drifts DOWN as k
    // rises, because table sizes track match counts rather than 2^k exactly. So
    // interpolate the coefficient between anchors instead of picking one and
    // pretending it holds everywhere.
    //
    // OUTSIDE the measured range, take the largest coefficient ever seen (56.58,
    // at k=24) rather than extending the trend. The trend is downward, which means
    // extrapolating it is exactly the way to under-estimate — and this is the
    // number that decides whether the box survives.
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
    // device_label() alone is ambiguous the moment a device repeats — and with N
    // CPU workers it does, so a 4-worker bench would print four lines all called
    // "cpu" and a log would interleave four "[batch:cpu]" prefixes with no way to
    // tell which worker stalled. Suffix repeats with #ordinal; leave unique
    // devices exactly as they were, so single-CPU logs do not churn.
    std::vector<std::string> labels;
    labels.reserve(device_ids.size());
    for (std::size_t i = 0; i < device_ids.size(); ++i) {
        std::string base = device_label(device_ids[i]);
        std::size_t const total = static_cast<std::size_t>(
            std::count(device_ids.begin(), device_ids.end(), device_ids[i]));
        if (total > 1) {
            std::size_t const ordinal = static_cast<std::size_t>(
                std::count(device_ids.begin(),
                           device_ids.begin() + static_cast<long>(i),
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
        int const n = gpu_device_count();
        if (n > 0) {
            device_ids.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) device_ids.push_back(i);
        }
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
    if (opts.cpu_selected() && gpu_implicit && default_gpu_available) {
        device_ids.push_back(0);
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

std::size_t batch_worker_count(BatchOptions const& opts, int k)
{
    auto const device_ids = resolve_batch_devices(opts, k);
    if (opts.shard_plot && device_ids.size() > 1) {
        return 1;  // devices form one team; one plot in flight
    }
    if (device_ids.size() <= 1) return 1;
    return device_ids.size();
}

} // namespace pos2gpu
