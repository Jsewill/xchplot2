// BatchPlotter.cu — implementation of staggered multi-plot pipeline.

#include "host/BatchPlotter.hpp"
#include "host/Cancel.hpp"
#include "host/CpuPlotter.hpp"  // run_one_plot_cpu — pos2-chip CPU pipeline
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PlotFileWriterParallel.hpp"
#include "gpu/DeviceIds.hpp"  // kCpuDeviceId for the --cpu device-list mixin

// Deliberately no pos2-chip includes here — see PlotFileWriterParallel.cpp.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
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

// Live, lock-free progress for the ETA. A work-queue's remaining plots are held
// by SPECIFIC workers running at their OWN rates, so no batch-wide mean can
// price the drain — the estimator needs each worker's rate and its own in-flight
// count. See estimate_eta_seconds() in BenchStats.hpp for the model.
//
// Every strategy publishes here, single-worker ones into a one-slot table, so
// emit_progress_line has exactly one shape to reason about.
struct LiveWorker {
    std::atomic<std::size_t> written{0};
    std::atomic<std::size_t> in_flight{0};  // pulled off the queue, not retired
    std::atomic<double>      work_start{0.0};
    std::atomic<double>      first_done{0.0};
    std::atomic<double>      last_done{0.0};
};

struct BatchProgress {
    std::vector<LiveWorker>    workers;
    // Retired = written + skipped. It doubles as the ticket counter for the
    // display: each emitting thread prints the value its own fetch_add returned,
    // so the in-place TTY line cannot tick backwards even though N workers write
    // to it. (Summing the per-worker counters instead would let two threads race
    // to the same total and print it out of order.)
    std::atomic<std::size_t>   retired{0};
    std::atomic<std::size_t>   written{0};
    std::atomic<std::size_t>   skipped{0};
    std::atomic<std::uint64_t> bytes{0};
    std::size_t                total = 0;

    BatchProgress(std::size_t worker_count, std::size_t total_entries)
        : workers(worker_count), total(total_entries) {}
};

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

// done_now is the emitting thread's own `retired` ticket, not a re-read of the
// counter — see BatchProgress.
void emit_progress_line(std::string const& log_prefix,
                        BatchOptions const& opts,
                        BatchProgress const& live,
                        std::size_t done_now,
                        double elapsed_s)
{
    if (!opts.progress || done_now == 0 || elapsed_s <= 0.0) return;

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
    std::fprintf(stderr,
        "%s%s progress: plot %zu/%zu done "
        "(%.1f%%, %s%.2f s/plot avg, %.3g TiB/s, batch ETA %s%s)%s",
        in_place ? "\r\033[K" : "",
        log_prefix.c_str(),
        done_now, total,
        100.0 * double(done_now) / double(total),
        skip_note,
        avg, rate_tib_s,
        eta.lower_bound ? ">=" : "~",
        format_duration_hms(eta.seconds).c_str(),
        (!in_place || done_now >= total) ? "\n" : "");
    if (in_place) std::fflush(stderr);
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
// Depth = kNumPinnedBuffers - 1 so the producer never overtakes the
// consumer by more than (num_pinned - 1) plots. The pinned slot the
// producer writes is slot (i % kNumPinnedBuffers); with depth-(N-1)
// the consumer is guaranteed to have popped plot (i - N) before the
// producer overwrites its slot.
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
// depth of (kNumPinnedBuffers - 1) only guarantees the consumer has
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
    std::string const log_prefix =
        (device_id == kCpuDeviceId) ? std::string("[batch:cpu]") :
        (device_id <  0)            ? std::string("[batch]")     :
        ("[batch:gpu" + std::to_string(device_id) + "]");

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
    if (device_id == kCpuDeviceId) {
        BatchResult res;
        if (entries.empty()) return res;

        // Yield to the GPU workers — but only when there ARE any. Nicing is
        // irreversible for an unprivileged process, and with the CPU as the sole
        // worker this slice runs ON THE MAIN THREAD (run_batch's single-worker
        // fast path calls run_batch_slice inline), so we would permanently
        // de-prioritise the whole process against everything else on the box in
        // order to yield to nobody.
        if (live.workers.size() > 1) {
            nice_current_thread(cpu_worker_nice_delta(), log_prefix.c_str());
        }

        auto const t_start = run_epoch ? *run_epoch
                                       : std::chrono::steady_clock::now();
        res.work_start_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        live_work_start(live, worker_id, res.work_start_seconds);
        std::size_t local_idx = 0;
        while (true) {
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
        for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
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
            stream_scratch.h_meta        = streaming_alloc_pinned_uint64(stream_pinned_cap);
            stream_scratch.h_keys_merged = streaming_alloc_pinned_uint32(stream_pinned_cap);
            stream_scratch.h_t2_xbits    = streaming_alloc_pinned_uint32(stream_pinned_cap);
            if (!stream_scratch.h_meta || !stream_scratch.h_keys_merged ||
                !stream_scratch.h_t2_xbits)
            {
                if (stream_scratch.h_meta)        streaming_free_pinned_uint64(stream_scratch.h_meta);
                if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
                if (stream_scratch.h_t2_xbits)    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
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
            if (stream_scratch.h_meta)        streaming_free_pinned_uint64(stream_scratch.h_meta);
            if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
            if (stream_scratch.h_t2_xbits)    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
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

    // Depth = kNumPinnedBuffers - 1. See Channel's comment block above.
    Channel chan(static_cast<std::size_t>(GpuBufferPool::kNumPinnedBuffers - 1));
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
    // depth = kNumPinnedBuffers); it must NOT use the global plot index
    // when shared_idx is in play, because peer workers also hold slots in
    // their own pools.
    try {
        std::size_t local_idx = 0;
        std::size_t local_count = 0;
        while (true) {
            if (consumer_failed) break;
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
                local_count % GpuBufferPool::kNumPinnedBuffers);
            // Slot-reuse gate: the previous occupant of this slot was
            // push number (local_count - kNumPinnedBuffers); wait until
            // the consumer has fully finished it (not merely popped it)
            // before the pipeline's D2H writes into the slot.
            if (local_count >= std::size_t(GpuBufferPool::kNumPinnedBuffers)) {
                std::size_t const need =
                    local_count - GpuBufferPool::kNumPinnedBuffers + 1;
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

            if (peak > declared) {
                std::fprintf(stderr,
                    "%s vram: %s — pooled path peaked at %.0f MiB but declared "
                    "%.0f MiB. Something in run_gpu_pipeline is allocating "
                    "outside the pool; the gate that decides whether this card "
                    "can hold the pool is now wrong by at least %.0f MiB.\n",
                    log_prefix.c_str(),
                    assert_vram_enabled() ? "FATAL" : "WARNING",
                    to_mib(peak), to_mib(declared), to_mib(peak - declared));
                if (assert_vram_enabled()) {
                    throw std::runtime_error(
                        "POS2GPU_ASSERT_VRAM: pooled path exceeded its declared VRAM");
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

    // Disk-space preflight (advisory, never throws). Runs once before
    // any worker dispatch so the user sees the warning up front instead
    // of N minutes into a doomed batch.
    preflight_disk_space(entries, opts);

    // Resolve the target device list (see resolve_batch_devices):
    //   use_all_devices  → enumerate at runtime, one worker per GPU
    //   device_ids       → use these explicit ids
    //   (neither)        → empty list → single-device, CUDA-default device
    //   include_cpu      → orthogonal: append a CPU worker (kCpuDeviceId)
    std::vector<int> const device_ids = resolve_batch_devices(opts);
    if (opts.use_all_devices &&
        std::none_of(device_ids.begin(), device_ids.end(),
                     [](int id) { return id != kCpuDeviceId; })) {
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

    // Fast path: zero-config default or one explicit id. Runs on the
    // caller thread — identical control flow to pre-multi-GPU except
    // for the optional cudaSetDevice at the top of the slice.
    if (device_ids.size() <= 1) {
        int const dev = device_ids.empty() ? kDefaultGpuId : device_ids[0];
        BatchProgress live(1, entries.size());
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
    if (!opts.quiet) {
        std::string devs;
        for (size_t i = 0; i < N; ++i) {
            if (i) devs += ", ";
            devs += device_label(device_ids[i]);
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
        agg.workers.push_back(std::move(w));
    }
    std::sort(agg.completion_seconds.begin(), agg.completion_seconds.end());
    agg.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return agg;
}

std::vector<int> resolve_batch_devices(BatchOptions const& opts)
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
    // include_cpu is orthogonal: append a CPU worker (kCpuDeviceId)
    // alongside whatever GPUs are already selected. Don't dedup —
    // caller can pass `cpu` multiple times for multi-core CPU — but
    // collapse the case where include_cpu was set both via --cpu
    // AND via a `cpu` token in --devices.
    if (opts.include_cpu &&
        std::find(device_ids.begin(), device_ids.end(), kCpuDeviceId)
            == device_ids.end()) {
        device_ids.push_back(kCpuDeviceId);
    }
    return device_ids;
}

std::size_t batch_worker_count(BatchOptions const& opts)
{
    auto const device_ids = resolve_batch_devices(opts);
    if (opts.shard_plot && device_ids.size() > 1) {
        return 1;  // devices form one team; one plot in flight
    }
    if (device_ids.size() <= 1) return 1;
    return device_ids.size();
}

} // namespace pos2gpu
