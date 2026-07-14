// BatchPlotter.cu — implementation of staggered multi-plot pipeline.

#include "host/BatchPlotter.hpp"
#include "host/Cancel.hpp"
#include "host/CpuPlotter.hpp"  // run_one_plot_cpu — pos2-chip CPU pipeline
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#include "host/HostPinnedPool.hpp"
#include "host/MultiGpuPlotPipeline.hpp"        // --shard-plot path (Phase 2.2+)
#include "host/MultiGpuPipelineParallel.hpp"   // --pipeline-plot path (Phase 2.1d)
#include "host/MultiGpuShardBufferPool.hpp"  // batch-amortised buffer reuse
#include "host/PlotFileWriterParallel.hpp"
#include "gpu/DeviceIds.hpp"  // kCpuDeviceId for the --cpu device-list mixin
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
#include <memory>
#include <mutex>
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
        if (!device_memory_probe(ordinal_, f, t)) return;  // unsupported → inert
        baseline_free_ = f;
        min_free_.store(f, std::memory_order_relaxed);
        started_ = true;
        th_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed)) {
                size_t f = 0;
                size_t t = 0;
                if (device_memory_probe(ordinal_, f, t)) {
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
    std::string const log_prefix =
        (device_id == kCpuDeviceId) ? std::string("[batch:cpu]") :
        (device_id <  0)            ? std::string("[batch]")     :
        ("[batch:gpu" + std::to_string(device_id) + "]");

    // "vram" is a lie on the CPU device: its "device memory" IS host RAM, and
    // the free figure behind it now comes from /proc/meminfo MemAvailable
    // rather than a GPU (it used to come from GPU 0 — see device_memory_probe).
    // The tier machinery below is otherwise identical for both, so only the
    // noun changes.
    char const* const mem_label =
        (device_id == kCpuDeviceId) ? "ram" : "vram";

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
    // There is exactly ONE of it: BatchOptions::include_cpu is a bool, so
    // repeating `cpu` in --devices changes nothing. (It used to claim
    // `--devices cpu,cpu,cpu,cpu` gave four workers. It never did — nothing
    // counted the tokens — so anyone who believed it benchmarked one worker and
    // attributed it to four. The CLI now warns on a repeated `cpu`.) N CPU
    // workers is a real win — measured +27% at N=2, +52% at N=4 on a 5950X at
    // k=28, because the plotter is memory-latency-bound and concurrent plots
    // interleave each other's stalls — but it costs 12.13 GiB of RAM per
    // worker, so it needs an explicit flag and a RAM gate, not a bool.
    //
    // XCHPLOT2_SYCL_CPU_BENCH=1 routes --cpu through the SYCL pipeline on
    // AdaptiveCpp's CPU backend instead of pos2-chip — exposed as an env
    // var purely for benchmarking the two CPU paths against each other,
    // not as a supported plotting mode (pos2-chip is faster + leaner).
    bool const sycl_cpu_bench = [] {
        char const* v = std::getenv("XCHPLOT2_SYCL_CPU_BENCH");
        return v && v[0] == '1';
    }();
    if (device_id == kCpuDeviceId && !sycl_cpu_bench) {
        BatchResult res;
        if (entries.empty()) return res;

        // Yield to the GPU workers. Only when there ARE any: with the CPU as the
        // sole worker there is nothing to yield to, this slice runs on the MAIN
        // thread (run_batch's single-worker fast path calls run_batch_slice
        // inline), and nicing is irreversible for an unprivileged process — so
        // we would be permanently de-prioritising the whole process to no end.
        //
        // live.workers.size() > 1 is exactly the right test: there is at most one
        // CPU worker, so any peer is a GPU.
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

    if (device_id >= 0 || device_id == kCpuDeviceId) bind_current_device(device_id);
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
    // Stage 4f: amortised streaming-path pinned-host scratch. Populated
    // in the streaming-fallback branch below; nullptr fields when the
    // pool path is active (pool_ptr != null).
    StreamingPinnedScratch stream_scratch{};
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
            stream_scratch.h_meta        = streaming_alloc_pinned_uint64(stream_pinned_cap);
            stream_scratch.h_keys_merged = streaming_alloc_pinned_uint32(stream_pinned_cap);
            stream_scratch.h_t2_xbits    = streaming_alloc_pinned_uint32(stream_pinned_cap);
            stream_scratch.h_t3          = streaming_alloc_pinned_uint64(stream_pinned_cap);
            // Tiny tier needs a separate h_t2_meta to avoid the
            // h_t1_meta/h_t2_meta buffer-reuse race in T2 match's
            // per-pass loop. Compact / minimal modes don't trip the
            // race (they read d_t1_meta_sorted on device, not h_t1_meta
            // on host) so leave h_t2_meta null and the streaming
            // pipeline reuses h_meta as before.
            if (stream_scratch.tiny_mode) {
                stream_scratch.h_t2_meta = streaming_alloc_pinned_uint64(stream_pinned_cap);
            }
            if (!stream_scratch.h_meta || !stream_scratch.h_keys_merged ||
                !stream_scratch.h_t2_xbits || !stream_scratch.h_t3 ||
                (stream_scratch.tiny_mode && !stream_scratch.h_t2_meta))
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

        if (peak > declared) {
            // Loud on every run, fatal when asserting. A path that exceeds its
            // declared footprint is not a benign overshoot: the picker hands
            // that tier to cards sized from the model, so whatever slipped
            // through here is an OOM on somebody's smaller GPU.
            std::fprintf(stderr,
                "%s %s: ERROR — peak %.0f MiB exceeds the %.0f MiB declared "
                "for this path by %.0f MiB. Some device allocation is not "
                "accounted for in the peak model; a card sized from that model "
                "will OOM. See the two-phase budget notes in SyclBackend.hpp.\n",
                log_prefix.c_str(), mem_label, to_mib(peak), to_mib(declared),
                to_mib(peak - declared));
            if (char const* v = std::getenv("POS2GPU_ASSERT_VRAM");
                v && v[0] == '1')
            {
                throw std::runtime_error(
                    "VRAM assertion failed: peak " +
                    std::to_string(uint64_t(to_mib(peak))) + " MiB > declared " +
                    std::to_string(uint64_t(to_mib(declared))) + " MiB");
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
        if (dev_id == kCpuDeviceId) {
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
    if (opts.include_cpu &&
        std::find(device_ids.begin(), device_ids.end(), kCpuDeviceId)
            == device_ids.end()) {
        device_ids.push_back(kCpuDeviceId);
    }
    return device_ids;
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
    auto const device_ids = resolve_batch_devices(opts);
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
    //   include_cpu      → orthogonal: also append kCpuDeviceId so the
    //                      CPU runs as one more worker. Mixes with the
    //                      above (--cpu alone → CPU only; --cpu --devices
    //                      all → all GPUs + CPU; etc.).
    std::vector<int> const device_ids = resolve_batch_devices(opts);
    if (opts.use_all_devices &&
        std::none_of(device_ids.begin(), device_ids.end(),
                     [](int id) { return id != kCpuDeviceId; })) {
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
        agg.workers.push_back(std::move(w));
    }
    std::sort(agg.completion_seconds.begin(), agg.completion_seconds.end());
    agg.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return agg;
}

} // namespace pos2gpu
