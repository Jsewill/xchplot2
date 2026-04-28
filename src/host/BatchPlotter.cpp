// BatchPlotter.cu — implementation of staggered multi-plot pipeline.

#include "host/BatchPlotter.hpp"
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
#include <memory>
#include <mutex>
#include <queue>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
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
// fan-out can spawn N of these concurrently — one thread per GPU, each
// with its own pool / channel / consumer. The outer run_batch validates
// homogeneity once; this helper assumes it has already been done on
// `entries`.
//
// device_id < 0  → keep the CUDA-default current device (single-device
//                  default; zero-config users see unchanged behavior).
// worker_id  < 0 → single-device path; reserved for future per-worker
//                  log prefix. v1 leaves stderr lines as-is since each
//                  fprintf is atomic per-line on POSIX and interleaving
//                  is tolerable.
BatchResult run_batch_slice(std::vector<BatchEntry> const& entries,
                            BatchOptions const& opts,
                            int                 device_id,
                            int                 worker_id)
{
    (void)worker_id;

    // CPU worker: bypass GPU pool / streaming entirely. pos2-chip's
    // Plotter manages its own state, so each plot is a synchronous
    // run_one_plot_cpu() call — no CUDA, no GpuBufferPool. Single-
    // threaded internally; multi-core utilization comes from passing
    // `cpu` multiple times in --devices (e.g. --devices cpu,cpu,cpu,cpu
    // on a 4-core host).
    if (device_id == kCpuDeviceId) {
        BatchResult res;
        if (entries.empty()) return res;
        auto const t_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < entries.size(); ++i) {
            try {
                run_one_plot_cpu(entries[i], opts);
                ++res.plots_written;
                if (opts.verbose) {
                    std::fprintf(stderr,
                        "[batch:cpu] plot %zu/%zu done: %s\n",
                        i + 1, entries.size(),
                        entries[i].out_name.c_str());
                }
            } catch (std::exception const& ex) {
                std::fprintf(stderr,
                    "[batch:cpu] plot %zu FAILED: %s\n", i, ex.what());
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
        // Thresholds (measured on sm_89):
        //   plain:   peak 7290 + margin 128 = 7418 MB floor
        //   compact: peak 5200 + margin 128 = 5328 MB floor
        constexpr uint64_t kPlainFloorBytes   = 7418ULL * 1024 * 1024;
        constexpr uint64_t kCompactFloorBytes = 5328ULL * 1024 * 1024;
        // Minimal tier: compact's pinned-host parking + N=8 T2 match
        // staging (cap/8 vs compact's cap/2). Saves ~1.5 GiB of T2-match
        // peak VRAM at the cost of 6 extra PCIe round-trips during T2
        // match. Targets 4 GiB cards (GTX 1050 Ti / 1650, RTX 3050 4GB,
        // MX450). Floor is estimated, not measured on real 4 GiB
        // hardware — please report actual fit on a 4 GiB card.
        constexpr uint64_t kMinimalFloorBytes = 3828ULL * 1024 * 1024;
        size_t const free_bytes = streaming_query_free_vram_bytes();

        // Tier selection precedence: opts.streaming_tier (--tier CLI flag)
        // > XCHPLOT2_STREAMING_TIER env var > auto-pick by free VRAM. The
        // manual overrides bypass the auto-pick threshold but still bail
        // out cleanly if the chosen tier definitely won't fit (minimal
        // floor is the hard lower bound; forced higher tier on a card
        // below that tier's floor warns + proceeds — caller asked).
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
            // Auto: pick the largest tier that fits.
            tier = (free_bytes >= kPlainFloorBytes)   ? Tier::Plain   :
                   (free_bytes >= kCompactFloorBytes) ? Tier::Compact :
                                                        Tier::Minimal;
        }

        // Forced-tier fit warnings. Forced tiers below their floor are
        // allowed (caller's risk) — except minimal below its floor still
        // throws because there's no smaller tier to fall back to.
        if (tier == Tier::Plain && free_bytes < kPlainFloorBytes) {
            std::fprintf(stderr,
                "[batch] streaming tier: plain forced (%.2f GiB free < %.2f GiB "
                "plain floor) — proceeding, may OOM mid-plot\n",
                free_bytes / double(1ULL << 30),
                kPlainFloorBytes / double(1ULL << 30));
        } else if (tier == Tier::Compact && free_bytes < kCompactFloorBytes) {
            std::fprintf(stderr,
                "[batch] streaming tier: compact forced (%.2f GiB free < %.2f GiB "
                "compact floor) — proceeding, may OOM mid-plot\n",
                free_bytes / double(1ULL << 30),
                kCompactFloorBytes / double(1ULL << 30));
        }

        if (tier == Tier::Plain) {
            // Plain: zero PCIe overhead, all parks skipped.
            std::fprintf(stderr,
                "[batch] streaming tier: plain (%.2f GiB free, %.2f GiB floor)\n",
                free_bytes / double(1ULL << 30),
                kPlainFloorBytes / double(1ULL << 30));
        } else if (tier == Tier::Compact || tier == Tier::Minimal) {
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
                    "[batch] streaming-fallback: compact/minimal pinned scratch alloc failed");
            }
            stream_compact = true;
            stream_scratch.t3_tile_count = 2;
            if (tier == Tier::Minimal) {
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
                std::fprintf(stderr,
                    "[batch] streaming tier: minimal (%.2f GiB free, %.2f GiB floor; "
                    "park/rehydrate + N=8 T2 + N=%d T1-match + T1/T2 sort gather + "
                    "N=%d T3 input slicing, expect ~5-15 s/plot extra PCIe)\n",
                    free_bytes / double(1ULL << 30),
                    kMinimalFloorBytes / double(1ULL << 30),
                    stream_scratch.t3_input_slice_count,
                    stream_scratch.t3_input_slice_count);
            } else {
                std::fprintf(stderr,
                    "[batch] streaming tier: compact (%.2f GiB free < %.2f GiB plain floor; "
                    "park/rehydrate + N=2 T3 staging, expect ~1-2 s/plot extra PCIe)\n",
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
            throw std::runtime_error("[batch] internal: unhandled streaming tier");
        }
        // Forced-minimal hard floor: there's no smaller tier to fall back
        // to, so a card below the minimal floor genuinely can't plot at
        // this k. Bail with a clear message.
        if (tier == Tier::Minimal && free_bytes < kMinimalFloorBytes) {
            for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
            }
            if (stream_scratch.h_meta)        streaming_free_pinned_uint64(stream_scratch.h_meta);
            if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
            if (stream_scratch.h_t2_xbits)    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
            throw std::runtime_error(
                "[batch] card too small for k=" + std::to_string(pool_k) +
                " streaming at any tier: " +
                std::to_string(free_bytes / (1ULL << 20)) + " MB free < " +
                std::to_string(kMinimalFloorBytes / (1ULL << 20)) +
                " MB minimal floor. Use a smaller k or a larger GPU "
                "(or --cpu for pos2-chip CPU plotting).");
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

    // Consumer: takes finished GpuPipelineResults and writes plot files.
    std::thread consumer([&] {
        try {
            WorkItem item;
            while (chan.pop(item)) {
                std::filesystem::create_directories(item.entry.out_dir);
                auto full_path = std::filesystem::path(item.entry.out_dir) / item.entry.out_name;

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
            }
        } catch (...) {
            consumer_err = std::current_exception();
            consumer_failed = true;
        }
    });

    // Producer (this thread): drives the GPU pipeline, hands off to consumer.
    try {
        for (size_t i = 0; i < entries.size(); ++i) {
            if (consumer_failed) break;

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
            int const slot = static_cast<int>(i % GpuBufferPool::kNumPinnedBuffers);
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
    // Compact-tier scratch (nullptr-safe; no-op if plain tier ran).
    if (stream_scratch.h_meta)        streaming_free_pinned_uint64(stream_scratch.h_meta);
    if (stream_scratch.h_keys_merged) streaming_free_pinned_uint32(stream_scratch.h_keys_merged);
    if (stream_scratch.h_t2_xbits)    streaming_free_pinned_uint32(stream_scratch.h_t2_xbits);
    {
        (void)stream_compact;  // avoid unused-warning on plain-only builds
    }

    res.plots_written = plots_done.load();
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

    // Resolve the target device list:
    //   use_all_devices  → enumerate at runtime, one worker per GPU
    //   device_ids       → use these explicit ids
    //   (neither)        → empty list → single-device, CUDA-default device
    std::vector<int> device_ids;
    if (opts.use_all_devices) {
        int const n = gpu_device_count();
        if (n <= 0) {
            std::fprintf(stderr,
                "[batch] --devices all: runtime enumerated 0 GPUs — "
                "falling back to the CUDA-default device\n");
        } else {
            device_ids.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) device_ids.push_back(i);
        }
    } else if (!opts.device_ids.empty()) {
        device_ids = opts.device_ids;
    }
    // include_cpu is orthogonal: append a CPU worker (kCpuDeviceId)
    // alongside whatever GPUs are already selected. Don't dedup —
    // caller can pass `cpu` multiple times for multi-core CPU
    // (e.g. --devices cpu,cpu,cpu,cpu on a 4-core host) — but
    // collapse the case where include_cpu was set both via --cpu
    // AND via a `cpu` token in --devices.
    if (opts.include_cpu &&
        std::find(device_ids.begin(), device_ids.end(), kCpuDeviceId)
            == device_ids.end()) {
        device_ids.push_back(kCpuDeviceId);
    }

    auto const t_start = std::chrono::steady_clock::now();

    // Fast path: zero-config default or one explicit id. Runs on the
    // caller thread — identical control flow to pre-multi-GPU except
    // for the optional cudaSetDevice at the top of the slice.
    if (device_ids.size() <= 1) {
        int const dev = device_ids.empty() ? -1 : device_ids[0];
        BatchResult r = run_batch_slice(entries, opts, dev, -1);
        r.total_wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        return r;
    }

    // Multi-device: round-robin-partition the entries and spawn one
    // worker thread per GPU. Each worker constructs its own
    // GpuBufferPool, producer/consumer channel, and writer thread on
    // its target device — zero cross-worker shared state beyond stderr
    // and the filesystem. Plot output names come from the manifest, so
    // distinct plots already land in distinct files.
    size_t const N = device_ids.size();
    std::vector<std::vector<BatchEntry>> buckets(N);
    for (size_t i = 0; i < entries.size(); ++i) {
        buckets[i % N].push_back(entries[i]);
    }

    std::fprintf(stderr,
        "[batch] multi-device: %zu plots across %zu workers — devices:",
        entries.size(), N);
    for (size_t i = 0; i < N; ++i) {
        std::fprintf(stderr, " %d", device_ids[i]);
    }
    std::fprintf(stderr, "\n");

    std::vector<BatchResult>         per_worker(N);
    std::vector<std::exception_ptr>  per_worker_exc(N);
    std::vector<std::thread>         workers;
    workers.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        workers.emplace_back([&, i]() {
            try {
                per_worker[i] = run_batch_slice(
                    buckets[i], opts, device_ids[i], static_cast<int>(i));
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
    }
    agg.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    return agg;
}

} // namespace pos2gpu
