// BatchPlotter.cu — implementation of staggered multi-plot pipeline.

#include "host/BatchPlotter.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#include "host/PlotFileWriterParallel.hpp"

#include "plot/PlotData.hpp"
#include "pos/ProofParams.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
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

// Bounded SPSC queue of depth 1 plus end-of-stream signal.
class Channel {
public:
    void push(WorkItem item) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]{ return !item_.has_value() && !closed_; });
        if (closed_) return;
        item_ = std::move(item);
        cv_.notify_all();
    }
    // Returns false when channel is closed AND empty.
    bool pop(WorkItem& out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]{ return item_.has_value() || closed_; });
        if (item_.has_value()) {
            out = std::move(*item_);
            item_.reset();
            cv_.notify_all();
            return true;
        }
        return false;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        cv_.notify_all();
    }
private:
    std::mutex                mu_;
    std::condition_variable   cv_;
    std::optional<WorkItem>   item_;
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
    GpuBufferPool pool(pool_k, pool_strength, pool_testnet);
    if (verbose) {
        double gb = 1.0 / (1024.0 * 1024.0 * 1024.0);
        std::fprintf(stderr,
            "[batch] pool: storage=%.2f GB pair_a=%.2f GB pair_b=%.2f GB "
            "sort_scratch=%.2f GB pinned=%.2f GB (Xs scratch aliased in pair_b)\n",
            pool.storage_bytes * gb,
            pool.pair_bytes    * gb,
            pool.pair_bytes    * gb,
            pool.sort_scratch_bytes * gb,
            pool.pinned_bytes       * gb);
    }

    Channel chan;
    std::atomic<bool>     consumer_failed{false};
    std::atomic<size_t>   plots_done{0};
    std::exception_ptr    consumer_err;

    auto t_start = std::chrono::steady_clock::now();

    // Consumer: takes finished GpuPipelineResults and writes plot files.
    std::thread consumer([&] {
        try {
            WorkItem item;
            while (chan.pop(item)) {
                ProofParams params(item.entry.plot_id.data(),
                                   static_cast<uint8_t>(item.entry.k),
                                   static_cast<uint8_t>(item.entry.strength),
                                   item.entry.testnet ? uint8_t{1} : uint8_t{0});
                PlotData plot;
                plot.t3_proof_fragments = std::move(item.result.t3_fragments);

                std::filesystem::create_directories(item.entry.out_dir);
                auto full_path = std::filesystem::path(item.entry.out_dir) / item.entry.out_name;

                std::vector<uint8_t> memo_bytes = item.entry.memo;
                if (memo_bytes.empty()) memo_bytes.assign(32 + 48 + 32, 0);

                write_plot_file_parallel(
                    full_path.string(), plot, params,
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
            item.result = run_gpu_pipeline(cfg, pool);

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

    res.plots_written = plots_done.load();
    res.total_wall_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count();
    return res;
}

} // namespace pos2gpu
