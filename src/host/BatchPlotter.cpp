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

    void push(WorkItem item) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_not_full_.wait(lock, [&]{ return q_.size() < capacity_ || closed_; });
        if (closed_) return;
        q_.push(std::move(item));
        cv_not_empty_.notify_one();
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
BatchResult run_batch_slice(std::vector<BatchEntry> const& entries,
                            BatchOptions const& opts,
                            int                 device_id,
                            int                 worker_id,
                            std::atomic<std::size_t>* shared_idx = nullptr)
{
    (void)worker_id;

    // CPU worker: bypass the GPU pool / streaming path entirely. pos2-chip's
    // Plotter manages all internal state itself, so each plot is a
    // synchronous run_one_plot_cpu() call. Single-threaded internally;
    // multi-core utilization comes from passing `cpu` multiple times in
    // --devices (e.g. --devices cpu,cpu,cpu,cpu on a 4-core host).
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
        auto const t_start = std::chrono::steady_clock::now();
        std::size_t local_idx = 0;
        while (true) {
            std::size_t const i = shared_idx
                ? shared_idx->fetch_add(1, std::memory_order_relaxed)
                : local_idx++;
            if (i >= entries.size()) break;
            if (opts.skip_existing) {
                auto out_path = std::filesystem::path(entries[i].out_dir)
                                / entries[i].out_name;
                if (looks_like_complete_plot(out_path)) {
                    if (opts.verbose) {
                        std::fprintf(stderr,
                            "[batch:cpu] skipping plot %zu: %s (already exists)\n",
                            i, out_path.string().c_str());
                    }
                    ++res.plots_skipped;
                    continue;
                }
            }
            try {
                run_one_plot_cpu(entries[i], opts);
                ++res.plots_written;
                if (opts.verbose) {
                    std::fprintf(stderr,
                        "[batch:cpu] plot %zu done: %s\n",
                        i, entries[i].out_name.c_str());
                }
            } catch (std::exception const& ex) {
                std::fprintf(stderr,
                    "[batch:cpu] plot %zu FAILED: %s\n", i, ex.what());
                ++res.plots_failed;
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

    // Force-streaming override (matches the one-shot run_gpu_pipeline
    // dispatch). Useful for testing the streaming path on a high-VRAM
    // card and for users who want the smaller peak even when the pool
    // would fit.
    bool const force_streaming = [] {
        char const* v = std::getenv("XCHPLOT2_STREAMING");
        return v && v[0] == '1';
    }();

    try {
        if (force_streaming) {
            throw InsufficientVramError("XCHPLOT2_STREAMING=1 forced");
        }
        pool_ptr = std::make_unique<GpuBufferPool>(
            pool_k, pool_strength, pool_testnet);
    } catch (InsufficientVramError const& e) {
        if (force_streaming) {
            std::fprintf(stderr, "[batch] XCHPLOT2_STREAMING=1 — using "
                                 "streaming pipeline per plot\n");
        } else {
            std::fprintf(stderr,
                "[batch] pool needs %.2f GiB, only %.2f GiB free — using "
                "streaming pipeline per plot\n",
                e.required_bytes / double(1ULL << 30),
                e.free_bytes     / double(1ULL << 30));
        }
        // Streaming tier dispatch — three tiers, increasing PCIe pressure
        // for decreasing peak VRAM:
        //   plain   (~7290 MB at k=28): no parks, single-pass T2 match.
        //                               Fastest, ~400 ms/plot over compact.
        //   compact (~5200 MB at k=28): all parks + N=2 T2 match staging.
        //                               Targets 6-8 GiB cards.
        //   minimal (~3700 MB at k=28): compact's parks + N=8 T2 match
        //                               staging. Targets 4 GiB cards at
        //                               the cost of extra PCIe round-trips
        //                               during T2 match.
        // Auto-pick takes the largest tier that fits with the margin.
        // 128 MB margin above measured CUDA-context + driver overhead
        // on headless cards.
        //
        // opts.streaming_tier (--tier CLI flag) > XCHPLOT2_STREAMING_TIER
        // env var > auto. Forced plain/compact below their floor warn but
        // proceed (caller's risk); forced minimal below its floor throws
        // because there is no smaller tier to fall back to.
        {
            auto const mem            = query_device_memory();
            size_t const plain_peak   = streaming_plain_peak_bytes(pool_k);
            size_t const compact_peak = streaming_peak_bytes(pool_k);
            size_t const minimal_peak = streaming_minimal_peak_bytes(pool_k);
            size_t const margin       = 128ULL << 20;
            auto to_gib = [](size_t b) { return b / double(1ULL << 30); };

            char const* tier_env = std::getenv("XCHPLOT2_STREAMING_TIER");
            std::string const tier_pref =
                !opts.streaming_tier.empty() ? opts.streaming_tier :
                (tier_env ? std::string(tier_env) : std::string());

            enum class Tier { Plain, Compact, Minimal };
            Tier tier;
            if (tier_pref == "plain") {
                tier = Tier::Plain;
            } else if (tier_pref == "compact") {
                tier = Tier::Compact;
            } else if (tier_pref == "minimal") {
                tier = Tier::Minimal;
            } else {
                // Auto: pick the largest tier that fits with margin.
                tier = (mem.free_bytes >= plain_peak   + margin) ? Tier::Plain   :
                       (mem.free_bytes >= compact_peak + margin) ? Tier::Compact :
                                                                   Tier::Minimal;
            }

            auto tier_name = [](Tier t) -> char const* {
                return t == Tier::Plain   ? "plain"
                     : t == Tier::Compact ? "compact"
                     :                      "minimal";
            };
            size_t const required =
                tier == Tier::Plain   ? plain_peak   :
                tier == Tier::Compact ? compact_peak :
                                        minimal_peak;

            // Minimal is the open-ended fallback — if even minimal won't
            // fit, throw. Forced higher tier below its floor warns and
            // proceeds (caller asked).
            if (tier == Tier::Minimal && mem.free_bytes < required + margin) {
                InsufficientVramError se(
                    "[batch] streaming pipeline needs ~" +
                    std::to_string(to_gib(required + margin)).substr(0, 5) +
                    " GiB peak for k=" + std::to_string(pool_k) +
                    " (minimal tier, the smallest available), device reports " +
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
            if (tier != Tier::Minimal && mem.free_bytes < required + margin) {
                std::fprintf(stderr,
                    "[batch] streaming tier: %s forced (%.2f GiB free < %.2f GiB "
                    "%s floor) — proceeding, may OOM mid-plot\n",
                    tier_name(tier),
                    to_gib(mem.free_bytes),
                    to_gib(required + margin),
                    tier_name(tier));
            }

            stream_scratch.plain_mode = (tier == Tier::Plain);
            if (tier == Tier::Minimal) {
                stream_scratch.t2_tile_count     = 8;
                stream_scratch.gather_tile_count = 4;
            }

            std::fprintf(stderr,
                "[batch] streaming tier: %s "
                "(%.2f GiB free, %.2f GiB peak, %.2f GiB plain floor)\n",
                tier_name(tier),
                to_gib(mem.free_bytes),
                to_gib(required),
                to_gib(plain_peak + margin));
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
                "[batch] streaming-fallback: pinned D2H buffer allocation failed");
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
        if (!stream_scratch.plain_mode) {
            stream_scratch.h_meta        = streaming_alloc_pinned_uint64(stream_pinned_cap);
            stream_scratch.h_keys_merged = streaming_alloc_pinned_uint32(stream_pinned_cap);
            stream_scratch.h_t2_xbits    = streaming_alloc_pinned_uint32(stream_pinned_cap);
            stream_scratch.h_t3          = streaming_alloc_pinned_uint64(stream_pinned_cap);
            if (!stream_scratch.h_meta || !stream_scratch.h_keys_merged ||
                !stream_scratch.h_t2_xbits || !stream_scratch.h_t3)
            {
                if (stream_scratch.h_meta)        streaming_free_pinned_uint64(stream_scratch.h_meta);
                if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
                if (stream_scratch.h_t2_xbits)    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
                if (stream_scratch.h_t3)          streaming_free_pinned_uint64(stream_scratch.h_t3);
                for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                    if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
                }
                throw std::runtime_error(
                    "[batch] streaming-fallback: pinned-host scratch allocation failed");
            }
        }
    }
    if (verbose && pool_ptr) {
        double gb = 1.0 / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr,
            "[batch] pool: storage=%.2f GB pair_a=%.2f GB pair_b=%.2f GB "
            "sort_scratch=%.2f GB pinned=2x%.2f GB "
            "(Xs scratch aliased in pair_b)\n",
            pool_ptr->storage_bytes * gb,
            pool_ptr->pair_a_bytes  * gb,
            pool_ptr->pair_b_bytes  * gb,
            pool_ptr->sort_scratch_bytes * gb,
            pool_ptr->pinned_bytes       * gb);
    }

    // Depth = kNumPinnedBuffers - 1. See Channel's comment block above.
    Channel chan(static_cast<std::size_t>(GpuBufferPool::kNumPinnedBuffers - 1));
    std::atomic<bool>     consumer_failed{false};
    std::atomic<size_t>   plots_done{0};
    std::exception_ptr    consumer_err;

    auto t_start = std::chrono::steady_clock::now();

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
                    write_plot_file_parallel(
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
                    if (verbose) {
                        std::fprintf(stderr, "[batch] consumer wrote plot %zu: %s\n",
                                     item.index, full_path.string().c_str());
                    }
                } catch (std::exception const& e) {
                    if (!opts.continue_on_error) throw;
                    ++plots_failed_consumer;
                    std::fprintf(stderr,
                        "[batch] plot %zu FAILED (write %s): %s — continuing\n",
                        item.index, full_path.string().c_str(), e.what());
                }
            }
        } catch (...) {
            consumer_err = std::current_exception();
            consumer_failed = true;
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
                    "[batch] cancel received — stopping before plot %zu\n", i);
                break;
            }

            if (opts.skip_existing) {
                auto out_path = std::filesystem::path(entries[i].out_dir)
                                / entries[i].out_name;
                if (looks_like_complete_plot(out_path)) {
                    if (verbose) {
                        std::fprintf(stderr,
                            "[batch] skipping plot %zu: %s (already exists)\n",
                            i, out_path.string().c_str());
                    }
                    ++res.plots_skipped;
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
            try {
                if (pool_ptr) {
                    // Pool path: rotate pinned slot per plot. The channel's
                    // (kNumPinnedBuffers - 1) depth holds the producer back
                    // before it overtakes the consumer's read of that slot.
                    item.result = run_gpu_pipeline(cfg, *pool_ptr, slot);
                } else {
                    // Streaming path with externally-owned pinned: same
                    // rotation + channel-depth invariant.
                    item.result = run_gpu_pipeline_streaming(
                        cfg, stream_pinned[slot], stream_pinned_cap,
                        stream_scratch);
                }
            } catch (std::exception const& e) {
                if (!opts.continue_on_error) throw;
                ++producer_failed;
                std::fprintf(stderr,
                    "[batch] plot %zu FAILED (GPU): %s — continuing\n",
                    i, e.what());
                continue;
            }

            if (verbose) {
                auto ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t_plot).count();
                std::fprintf(stderr,
                    "[batch] producer finished GPU for plot %zu in %.2f ms "
                    "(T1=%lu T2=%lu T3=%lu)\n",
                    i, ms,
                    (unsigned long)item.result.t1_count,
                    (unsigned long)item.result.t2_count,
                    (unsigned long)item.result.t3_count);
            }

            chan.push(std::move(item));
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

    for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
        streaming_free_pinned_uint64(stream_pinned[s]);
    }
    // Stage 4f: free the amortised streaming scratch (no-op if pool path
    // was used — all fields stay nullptr in that case).
    if (stream_scratch.h_meta)        streaming_free_pinned_uint64(stream_scratch.h_meta);
    if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
    if (stream_scratch.h_t2_xbits)    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
    if (stream_scratch.h_t3)          streaming_free_pinned_uint64(stream_scratch.h_t3);

    res.plots_written = plots_done.load();
    res.plots_failed  = producer_failed + plots_failed_consumer.load();
    res.total_wall_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count();
    return res;
}

} // namespace

BatchResult run_batch(std::vector<BatchEntry> const& entries,
                      BatchOptions const& opts)
{
    if (entries.empty()) return BatchResult{};

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

    // Resolve the target device list:
    //   use_all_devices  → enumerate at runtime, one worker per GPU
    //   device_ids       → use these explicit ids
    //   (neither)        → empty list → single-device default selector
    //   include_cpu      → orthogonal: also append kCpuDeviceId so the
    //                      CPU runs as one more worker. Mixes with the
    //                      above (--cpu alone → CPU only; --cpu --devices
    //                      all → all GPUs + CPU; etc.).
    std::vector<int> device_ids;
    if (opts.use_all_devices) {
        int const n = gpu_device_count();
        if (n <= 0) {
            std::fprintf(stderr,
                "[batch] --devices all: runtime enumerated 0 GPUs — "
                "falling back to the default SYCL selector\n");
        } else {
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

    auto const t_start = std::chrono::steady_clock::now();

    // Fast path: zero-config default or one explicit id. Runs on the
    // caller thread — identical control flow to pre-multi-GPU except
    // for the optional thread-local device bind at the top of the
    // slice.
    if (device_ids.size() <= 1) {
        int const dev = device_ids.empty() ? -1 : device_ids[0];
        BatchResult r = run_batch_slice(entries, opts, dev, -1);
        r.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
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
    std::fprintf(stderr,
        "[batch] multi-device: %zu plots across %zu workers (work-queue) — devices:",
        entries.size(), N);
    for (size_t i = 0; i < N; ++i) {
        std::fprintf(stderr, " %d", device_ids[i]);
    }
    std::fprintf(stderr, "\n");

    std::atomic<std::size_t> next_idx{0};
    std::vector<BatchResult>         per_worker(N);
    std::vector<std::exception_ptr>  per_worker_exc(N);
    std::vector<std::thread>         workers;
    workers.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        workers.emplace_back([&, i]() {
            try {
                per_worker[i] = run_batch_slice(
                    entries, opts, device_ids[i],
                    static_cast<int>(i), &next_idx);
            } catch (...) {
                per_worker_exc[i] = std::current_exception();
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
    for (auto const& r : per_worker) {
        agg.plots_written += r.plots_written;
        agg.plots_skipped += r.plots_skipped;
        agg.plots_failed  += r.plots_failed;
    }
    agg.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return agg;
}

} // namespace pos2gpu
