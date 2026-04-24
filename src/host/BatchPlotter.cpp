// BatchPlotter.cu — implementation of staggered multi-plot pipeline.

#include "host/BatchPlotter.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PlotFileWriterParallel.hpp"

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

BatchResult run_batch(std::vector<BatchEntry> const& entries, bool verbose)
{
    initialize_aes_tables();

    BatchResult res;
    if (entries.empty()) return res;

    // All entries in a batch must share (k, strength, testnet) so one pool
    // fits all plots. Mixed-shape batches could be supported by splitting
    // into homogeneous sub-batches; not needed in practice.
    int  pool_k        = entries[0].k;
    int  pool_strength = entries[0].strength;
    bool pool_testnet  = entries[0].testnet;
    for (size_t i = 1; i < entries.size(); ++i) {
        if (entries[i].k != pool_k
            || entries[i].strength != pool_strength
            || entries[i].testnet  != pool_testnet)
        {
            throw std::runtime_error(
                "run_batch: all entries must share (k, strength, testnet)");
        }
    }

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
        // compact drops to ~5200 MB by parking big intermediates on
        // pinned host (T1/T2 meta, keys_merged, T2 xbits). Compact pays
        // ~1-2 s/plot of PCIe round-trips, so we only opt into it when
        // the card can't fit plain.
        //
        // Thresholds (measured on sm_89):
        //   plain:   peak 7290 + margin 128 = 7418 MB floor
        //   compact: peak 6260 + margin 128 = 6388 MB floor
        //
        // Compact is the parks-only variant (stage 4a/4b/4c ports from
        // main): parks d_t1_meta / d_t1_keys_merged / d_t2_meta /
        // d_t2_xbits / d_t2_keys_merged on pinned host across their
        // idle windows. Drops T1 sort + T2 sort peaks from ~7290 to
        // ~6260 MB. Getting below ~6 GB (main's 5200 MB target) would
        // additionally need the T2 match N=2 tiling (main's stages
        // 1/2/3) — a deeper kernel-level port that isn't in this
        // commit.
        constexpr uint64_t kPlainFloorBytes   = 7418ULL * 1024 * 1024;
        constexpr uint64_t kCompactFloorBytes = 6388ULL * 1024 * 1024;
        size_t const free_bytes = streaming_query_free_vram_bytes();

        if (free_bytes >= kPlainFloorBytes) {
            // Plain tier: zero PCIe overhead, all parks skipped.
            std::fprintf(stderr,
                "[batch] streaming tier: plain (%.2f GiB free ≥ %.2f GiB floor)\n",
                free_bytes / double(1ULL << 30),
                kPlainFloorBytes / double(1ULL << 30));
        } else if (free_bytes >= kCompactFloorBytes) {
            // Compact tier: allocate the shared scratch buffers once
            // per batch. At k=28 this is ~4.2 GB of pinned host:
            //   h_meta        cap·u64 = 2.04 GB (reused t1_meta → t2_meta)
            //   h_keys_merged cap·u32 = 1.02 GB (reused t1_keys → t2_keys)
            //   h_t2_xbits    cap·u32 = 1.02 GB
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
                    "[batch] streaming-fallback: compact-tier pinned scratch alloc failed");
            }
            stream_compact = true;
            std::fprintf(stderr,
                "[batch] streaming tier: compact (%.2f GiB free < %.2f GiB plain floor; "
                "using park/rehydrate, expect ~1-2 s/plot extra PCIe)\n",
                free_bytes / double(1ULL << 30),
                kPlainFloorBytes / double(1ULL << 30));
        } else {
            for (int s = 0; s < GpuBufferPool::kNumPinnedBuffers; ++s) {
                if (stream_pinned[s]) streaming_free_pinned_uint64(stream_pinned[s]);
            }
            throw std::runtime_error(
                "[batch] card too small for k=28 streaming at any tier: " +
                std::to_string(free_bytes / (1ULL << 20)) + " MB free < " +
                std::to_string(kCompactFloorBytes / (1ULL << 20)) +
                " MB compact floor. Use a smaller k or a larger GPU.");
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

} // namespace pos2gpu
