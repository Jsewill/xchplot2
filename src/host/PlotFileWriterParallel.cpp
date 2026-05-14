// PlotFileWriterParallel.cpp — body of the parallel plot writer + CPU
// plotter wrapper.
//
// This is the SOLE TU in pos2-gpu that includes pos2-chip's plot/* and
// pos/ProofParams.hpp headers. That chain transitively pulls in
// pos/aes/soft_aes.hpp, which defines `soft_aesenc` / `soft_aesdec`
// without `inline`. If more than one TU saw those definitions, the
// final link would fail with multiple-definition errors. By keeping all
// pos2-chip-touching code here and exposing only a raw-byte / vector API
// to the rest of pos2-gpu, we sidestep the issue without patching
// pos2-chip.

#include "host/PlotFileWriterParallel.hpp"

#include "plot/ChunkCompressor.hpp"
#include "plot/PlotData.hpp"
#include "plot/PlotFile.hpp"
#include "plot/PlotIO.hpp"
#include "plot/Plotter.hpp"
#include "pos/ProofParams.hpp"
#include "pos/ProofValidator.hpp"
#include "prove/Prover.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace pos2gpu {

namespace {

// Process-global worker pool for plot-file FSE compression.
//
// Every write_plot_file_parallel() call routes its per-chunk tasks
// through this single pool. In a multi-GPU work-queue batch each GPU
// worker runs its own consumer thread, and each consumer used to call
// std::async with hardware_concurrency() tasks — so N concurrent
// writers spawned N × core_count OS threads, destructively
// oversubscribing the host. With the shared pool the total number of
// compression threads is fixed at hardware_concurrency() regardless of
// N; concurrent writers' tasks simply queue and drain through the same
// workers (work-conserving — a lone writer still gets every core).
//
// Re-entrancy is safe: the BatchPlotter consumer threads that call
// write_plot_file_parallel() are not pool workers, and a single
// write_plot_file_parallel() call's two parallel regions (chunkify,
// then compress) run sequentially, never nested.
class WriterThreadPool {
public:
    static WriterThreadPool& instance() {
        static WriterThreadPool pool;
        return pool;
    }

    std::size_t size() const noexcept { return workers_.size(); }

    std::future<void> submit(std::function<void()> fn) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
        std::future<void> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

private:
    WriterThreadPool() {
        unsigned n = std::thread::hardware_concurrency();
        if (n == 0) n = 4;
        workers_.reserve(n);
        for (unsigned i = 0; i < n; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~WriterThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop();
            }
            job();
        }
    }

    std::mutex                        mu_;
    std::condition_variable           cv_;
    std::queue<std::function<void()>> queue_;
    std::vector<std::thread>          workers_;
    bool                              stop_ = false;
};

// Inline equivalent of pos2-chip's
// ChunkedProofFragments::convertToChunkedProofFragments, but operating on
// a span so callers can point at pinned memory directly.
ChunkedProofFragments chunkify_proof_fragments_span(
    std::span<uint64_t const> t3_fragments, uint64_t range_per_chunk,
    unsigned thread_count)
{
    if (range_per_chunk == 0) {
        throw std::invalid_argument("range_per_chunk must be > 0");
    }
    ChunkedProofFragments chunked;
    if (t3_fragments.empty()) return chunked;

    uint64_t const max_value = t3_fragments.back();
    std::size_t const num_spans = static_cast<std::size_t>(max_value / range_per_chunk + 1);
    chunked.proof_fragments_chunks.resize(num_spans);

    // Step 1: find chunk boundaries in the already-sorted fragment array.
    // fragments are ascending, each chunk i covers [i*R, (i+1)*R) where R =
    // range_per_chunk, so a single O(N) sweep records the position where
    // each chunk starts. No per-fragment allocations.
    std::vector<std::size_t> boundaries(num_spans + 1);
    boundaries[0] = 0;
    std::size_t ci          = 0;
    uint64_t    chunk_end   = range_per_chunk;
    std::size_t const N     = t3_fragments.size();
    for (std::size_t i = 0; i < N; ++i) {
        while (t3_fragments[i] >= chunk_end) {
            boundaries[++ci] = i;
            chunk_end += range_per_chunk;
        }
    }
    for (std::size_t c = ci + 1; c <= num_spans; ++c) boundaries[c] = N;

    // Step 2: parallel copy, one task per contiguous range of chunks.
    // Tasks route through the shared WriterThreadPool — thread_count is
    // the split granularity, not a thread count.
    std::size_t const tasks_n       = std::min<std::size_t>(thread_count, num_spans);
    std::size_t const chunks_per_tk = (num_spans + tasks_n - 1) / tasks_n;

    auto& pool = WriterThreadPool::instance();
    std::vector<std::future<void>> tasks;
    tasks.reserve(tasks_n);
    for (std::size_t tstart = 0; tstart < num_spans; tstart += chunks_per_tk) {
        std::size_t const tend = std::min<std::size_t>(tstart + chunks_per_tk, num_spans);
        tasks.emplace_back(pool.submit(
            [&, tstart, tend]() {
                for (std::size_t c = tstart; c < tend; ++c) {
                    std::size_t const a = boundaries[c];
                    std::size_t const b = boundaries[c + 1];
                    chunked.proof_fragments_chunks[c].assign(
                        t3_fragments.begin() + a, t3_fragments.begin() + b);
                }
            }));
    }
    for (auto& f : tasks) f.get();

    return chunked;
}

} // namespace

size_t write_plot_file_parallel(
    std::string const& filename,
    std::span<uint64_t const> t3_fragments,
    uint8_t const* plot_id_32,
    uint8_t const k,
    uint8_t const strength,
    uint8_t const testnet,
    uint16_t const index,
    uint8_t const meta_group,
    std::span<uint8_t const> const memo,
    unsigned thread_count)
{
    ProofParams params(plot_id_32, k, strength, testnet);

    // thread_count is the task-split granularity, not a thread count:
    // every task routes through the shared WriterThreadPool, whose
    // worker count is fixed at hardware_concurrency(). 0 ⇒ split into
    // one task per pool worker. See WriterThreadPool above for why this
    // matters in a multi-GPU work-queue batch.
    if (thread_count == 0) {
        thread_count =
            static_cast<unsigned>(WriterThreadPool::instance().size());
    }

    // Build chunked representation (cheap; single pass over fragments)
    uint64_t const range_per_chunk = (1ULL << (params.get_k() + PlotFile::CHUNK_SPAN_RANGE_BITS));
    ChunkedProofFragments chunked
        = chunkify_proof_fragments_span(t3_fragments, range_per_chunk, thread_count);

    uint64_t const num_chunks = static_cast<uint64_t>(chunked.proof_fragments_chunks.size());
    int const stub_bits = params.get_k() - PlotFile::MINUS_STUB_BITS;

    // Parallel chunk compression. Static partitioning: tasks_n tasks,
    // each loops over a contiguous range of chunks, all routed through
    // the shared WriterThreadPool.
    std::vector<std::vector<uint8_t>> compressed(num_chunks);
    if (num_chunks > 0) {
        uint64_t const tasks_n       = std::min<uint64_t>(thread_count, num_chunks);
        uint64_t const chunks_per_tk = (num_chunks + tasks_n - 1) / tasks_n;
        auto& pool = WriterThreadPool::instance();
        std::vector<std::future<void>> tasks;
        tasks.reserve(tasks_n);
        for (uint64_t tstart = 0; tstart < num_chunks; tstart += chunks_per_tk) {
            uint64_t const tend = std::min<uint64_t>(tstart + chunks_per_tk, num_chunks);
            tasks.emplace_back(pool.submit(
                [&, tstart, tend]() {
                    for (uint64_t i = tstart; i < tend; ++i) {
                        uint64_t start_range = i * range_per_chunk;
                        compressed[i] = ChunkCompressor::compressProofFragments(
                            chunked.proof_fragments_chunks[i], start_range, stub_bits);
                    }
                }));
        }
        for (auto& f : tasks) f.get();
    }

    // Serial write phase — file I/O is sequential anyway. Write to
    // <filename>.partial and rename on success so SIGINT / crash / ENOSPC
    // never leaves a malformed .plot2 at the destination. The guard
    // unlinks the partial on early exit.
    std::string const partial = filename + ".partial";
    struct PartialGuard {
        std::string const& path;
        bool committed = false;
        ~PartialGuard() {
            if (!committed) {
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
        }
    } guard{partial};

    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to open " + filename);

    out.write("pos2", 4);
    uint8_t const ver = PlotFile::FORMAT_VERSION;
    out.write(reinterpret_cast<char const*>(&ver), 1);
    out.write(reinterpret_cast<char const*>(params.get_plot_id_bytes()), 32);

    uint8_t const k_byte = static_cast<uint8_t>(params.get_k());
    uint8_t const mkb    = static_cast<uint8_t>(params.get_match_key_bits());
    out.write(reinterpret_cast<char const*>(&k_byte), 1);
    out.write(reinterpret_cast<char const*>(&mkb),    1);
    out.write(reinterpret_cast<char const*>(&index), 2);
    out.write(reinterpret_cast<char const*>(&meta_group), 1);

    uint8_t const memo_size = static_cast<uint8_t>(memo.size());
    out.write(reinterpret_cast<char const*>(&memo_size), 1);
    out.write(reinterpret_cast<char const*>(memo.data()), memo.size());

    out.write(reinterpret_cast<char const*>(&num_chunks), sizeof(num_chunks));
    if (!out) throw std::runtime_error("Failed to write chunk count to " + filename);

    std::streampos offsets_start_pos = out.tellp();
    uint64_t zero = 0;
    for (uint64_t i = 0; i < num_chunks; ++i) {
        out.write(reinterpret_cast<char const*>(&zero), sizeof(zero));
    }
    if (!out) throw std::runtime_error("Failed to write chunk offset placeholders to " + filename);

    std::vector<uint64_t> offsets(num_chunks);
    for (uint64_t i = 0; i < num_chunks; ++i) {
        offsets[i] = static_cast<uint64_t>(out.tellp());
        writeVector(out, compressed[i]);
        if (!out) {
            throw std::runtime_error(
                "Failed to write chunk " + std::to_string(i) + " to " + filename);
        }
    }

    size_t bytes_written = static_cast<size_t>(out.tellp());

    out.seekp(offsets_start_pos);
    if (!out) throw std::runtime_error("Failed to seek to chunk offsets in " + filename);
    for (uint64_t i = 0; i < num_chunks; ++i) {
        out.write(reinterpret_cast<char const*>(&offsets[i]), sizeof(offsets[i]));
    }
    if (!out) throw std::runtime_error("Failed to write chunk offsets to " + filename);
    out.seekp(0, std::ios::end);

    // Close before rename so buffered writes are flushed and the destination
    // sees the final byte image.
    out.close();
    if (!out) throw std::runtime_error("Failed to close " + partial);

    std::error_code ec;
    std::filesystem::rename(partial, filename, ec);
    if (ec) {
        throw std::runtime_error(
            "Failed to rename " + partial + " -> " + filename + ": " + ec.message());
    }
    guard.committed = true;

    return bytes_written;
}

VerifyResult verify_plot_file(std::string const& filename, size_t n_trials)
{
    VerifyResult res;
    if (n_trials == 0) return res;

    Prover prover(filename);

    // Fresh entropy per call; the result only depends on the plot content,
    // not the specific challenges, beyond being a uniform sample.
    std::random_device rd;
    std::mt19937_64    gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    for (size_t i = 0; i < n_trials; ++i) {
        std::array<uint8_t, 32> challenge{};
        for (size_t j = 0; j < 32; j += 8) {
            uint64_t const v = dist(gen);
            std::memcpy(challenge.data() + j, &v, 8);
        }
        auto const chains = prover.prove(
            std::span<uint8_t const, 32>(challenge.data(), 32));
        res.trials++;
        res.proofs_found += chains.size();
        if (!chains.empty()) res.challenges_with_proof++;
    }
    return res;
}

std::vector<uint64_t> read_plot_file_fragments(std::string const& filename)
{
    PlotFile::PlotFileContents contents = PlotFile::readAllChunkedData(filename);
    std::vector<uint64_t> flat;
    size_t total = 0;
    for (auto const& chunk : contents.data.proof_fragments_chunks) total += chunk.size();
    flat.reserve(total);
    for (auto const& chunk : contents.data.proof_fragments_chunks) {
        flat.insert(flat.end(), chunk.begin(), chunk.end());
    }
    return flat;
}

std::vector<uint64_t> run_cpu_plotter_to_fragments(
    uint8_t const* plot_id_32,
    uint8_t k,
    uint8_t strength,
    uint8_t testnet,
    bool    verbose)
{
    ProofParams params(plot_id_32, k, strength, testnet);
    Plotter::Options opts{};
    opts.validate = false;
    opts.verbose  = verbose;
    Plotter plotter(params);
    PlotData plot = plotter.run(opts);
    return std::move(plot.t3_proof_fragments);
}

} // namespace pos2gpu
