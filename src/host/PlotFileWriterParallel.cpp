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
#include "host/BatchPlotter.hpp"


#include "plot/ChunkCompressor.hpp"
#include "plot/PlotData.hpp"
#include "plot/PlotFile.hpp"
#include "plot/PlotIO.hpp"
#include "plot/Plotter.hpp"
#include "pos/ProofParams.hpp"
#include "pos/ProofValidator.hpp"
#include "prove/Prover.hpp"
#include "solve/Solver.hpp"

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
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

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

// Wait for ALL futures before propagating any failure. Rethrowing on
// the first failed future (plain `for (f : tasks) f.get()`) would
// unwind the caller's frame while sibling tasks are still queued or
// running — and those tasks capture the caller's stack locals by
// reference, so a single std::bad_alloc in one task would turn into
// use-after-free writes in every other. Drain everything, then rethrow
// the first stored exception.
void wait_all_rethrow_first(std::vector<std::future<void>>& tasks)
{
    std::exception_ptr first;
    for (auto& f : tasks) {
        try { f.get(); }
        catch (...) { if (!first) first = std::current_exception(); }
    }
    if (first) std::rethrow_exception(first);
}

// Flush the directory entry after a rename so the new name survives a
// crash. Best-effort: some filesystems reject directory fsync, and the
// data itself was already fsynced.
void fsync_parent_dir_best_effort(std::string const& path)
{
#ifndef _WIN32
    auto dir = std::filesystem::path(path).parent_path();
    if (dir.empty()) dir = ".";
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#else
    (void)path;
#endif
}

// Chunk boundary table for the already-sorted fragment array. Chunk i
// covers value range [i*R, (i+1)*R) where R = range_per_chunk; a single
// O(N) sweep records where each chunk starts. The fragments themselves
// are NOT copied — ChunkCompressor::compressProofFragments takes a
// span, so each chunk is compressed straight out of the (often pinned)
// source buffer. The previous implementation materialised every chunk
// into its own std::vector first: a full extra pass over ~2 GB at k=28
// plus thousands of allocations, on the consumer thread that gates
// plot completion.
std::vector<std::size_t> chunk_boundaries_span(
    std::span<uint64_t const> t3_fragments, uint64_t range_per_chunk)
{
    if (range_per_chunk == 0) {
        throw std::invalid_argument("range_per_chunk must be > 0");
    }
    if (t3_fragments.empty()) return {};

    uint64_t const max_value = t3_fragments.back();
    std::size_t const num_spans = static_cast<std::size_t>(max_value / range_per_chunk + 1);

    std::vector<std::size_t> boundaries(num_spans + 1);
    boundaries[0] = 0;
    std::size_t ci          = 0;
    uint64_t    chunk_end   = range_per_chunk;
    std::size_t const N     = t3_fragments.size();
    for (std::size_t i = 0; i < N; ++i) {
        if (t3_fragments[i] > max_value || (i && t3_fragments[i] < t3_fragments[i - 1]))
            throw std::invalid_argument("proof fragments must be sorted");
        while (t3_fragments[i] >= chunk_end) {
            boundaries[++ci] = i;
            chunk_end += range_per_chunk;
        }
    }
    for (std::size_t c = ci + 1; c <= num_spans; ++c) boundaries[c] = N;
    return boundaries;
}

// Check structure before a reader can allocate from file-controlled lengths.
// This scans the small index and length prefixes, not every compressed payload.
bool valid_plot_structure(std::string const& filename, BatchEntry const* expected)
{
    std::error_code ec;
    uint64_t const size = std::filesystem::file_size(filename, ec);
    if (ec || size < 51) return false;
    std::ifstream in(filename, std::ios::binary);
    auto read = [&](void* data, size_t bytes) {
        in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
        return bool(in);
    };
    char magic[4];
    uint8_t version = 0, k = 0, strength = 0, group = 0, memo_size = 0;
    uint16_t index = 0;
    std::array<uint8_t, 32> id{};
    if (!read(magic, 4) || std::memcmp(magic, "pos2", 4) != 0 ||
        !read(&version, 1) || version != PlotFile::FORMAT_VERSION ||
        !read(id.data(), id.size()) || !read(&k, 1) || !read(&strength, 1) ||
        !read(&index, sizeof(index)) || !read(&group, 1) || !read(&memo_size, 1)) return false;
    if (k < 18 || k > 32 || (k & 1) || strength < 2 ||
        strength > k - (k < 28 ? 2 : k - 26) - 1) return false;
    std::vector<uint8_t> memo(memo_size);
    if (memo_size && !read(memo.data(), memo.size())) return false;
    if (expected && (id != expected->plot_id || k != expected->k ||
        strength != expected->strength || index != expected->plot_index ||
        group != expected->meta_group || memo != expected->memo)) return false;
    uint64_t count = 0;
    if (!read(&count, sizeof(count)) || count == 0 || count > (1ULL << (k - PlotFile::CHUNK_SPAN_RANGE_BITS)))
        return false;
    uint64_t const header_end = 51 + memo_size + count * sizeof(uint64_t);
    if (header_end > size) return false;
    std::vector<uint64_t> offsets(count);
    if (!read(offsets.data(), offsets.size() * sizeof(uint64_t))) return false;
    uint64_t end = header_end;
    for (auto offset : offsets) {
        if (offset != end || offset > size || size - offset < sizeof(uint64_t)) return false;
        in.seekg(static_cast<std::streamoff>(offset));
        uint64_t length = 0;
        if (!read(&length, sizeof(length)) || length > size - offset - sizeof(length)) return false;
        end = offset + sizeof(length) + length;
    }
    return end == size;
}

} // namespace

// Construct the pool on the CALLING thread. See the header for why the caller's
// identity matters: on Linux the pool's workers inherit the constructing
// thread's nice value, and BatchPlotter deliberately nices its CPU worker down.
// Whoever touches the pool first therefore decides the priority of every GPU
// worker's FSE, and there is no way to raise it back afterwards.
void warm_writer_pool()
{
    (void)WriterThreadPool::instance();
}

bool plot_file_matches(std::string const& filename, BatchEntry const& expected)
{
    return valid_plot_structure(filename, &expected);
}

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
    if (k < 18 || k > 32 || (k & 1)) throw std::invalid_argument("k must be even in [18, 32]");
    if (memo.size() > 255) throw std::invalid_argument("memo exceeds 255 bytes");
    if (k < 32 && !t3_fragments.empty() && (t3_fragments.back() >> (2 * k)))
        throw std::invalid_argument("proof fragment exceeds the plot's bit width");
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

    // Chunk boundary table (cheap; single pass over fragments). Chunks
    // are compressed directly from the source span — no per-chunk copy.
    uint64_t const range_per_chunk = (1ULL << (params.get_k() + PlotFile::CHUNK_SPAN_RANGE_BITS));
    std::vector<std::size_t> const boundaries =
        chunk_boundaries_span(t3_fragments, range_per_chunk);

    uint64_t const num_chunks =
        boundaries.empty() ? 0 : static_cast<uint64_t>(boundaries.size() - 1);
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
                            t3_fragments.subspan(boundaries[i],
                                                 boundaries[i + 1] - boundaries[i]),
                            start_range, stub_bits);
                    }
                }));
        }
        wait_all_rethrow_first(tasks);
    }

    // Exclusive temporary file in the destination directory. Keep its open
    // descriptor through the durability barrier; never reopen a pathname.
    std::vector<char> iobuf(size_t{4} << 20);
    std::string partial = filename + ".partial.XXXXXX";
#ifdef _WIN32
    if (_mktemp_s(partial.data(), partial.size() + 1) != 0)
        throw std::runtime_error("Failed to create temporary name for " + filename);
    int const fd = ::_open(partial.c_str(), _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
                          _S_IREAD | _S_IWRITE);
#else
    int const fd = ::mkstemp(partial.data());
#endif
    if (fd < 0) throw std::runtime_error("Failed to create temporary file for " + filename);
    struct PartialGuard {
        std::string const& path;
        std::FILE* out = nullptr;
        bool committed = false;
        ~PartialGuard() {
            if (out) std::fclose(out);
            if (!committed) {
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
        }
    } guard{partial};
#ifdef _WIN32
    guard.out = ::_fdopen(fd, "wb");
    if (!guard.out) ::_close(fd);
#else
    guard.out = ::fdopen(fd, "wb");
    if (!guard.out) ::close(fd);
#endif
    if (!guard.out) throw std::runtime_error("Failed to open stream for " + partial);
    if (std::setvbuf(guard.out, iobuf.data(), _IOFBF, iobuf.size()) != 0)
        throw std::runtime_error("Failed to buffer " + partial);
    size_t bytes_written = 0;
    auto write = [&](void const* data, size_t bytes) {
        if (bytes && std::fwrite(data, 1, bytes, guard.out) != bytes)
            throw std::runtime_error("Failed to write " + partial);
        bytes_written += bytes;
    };
    write("pos2", 4);
    uint8_t const ver = PlotFile::FORMAT_VERSION;
    write(&ver, 1);
    write(params.get_plot_id_bytes(), 32);
    uint8_t const k_byte = params.get_k();
    uint8_t const mkb = params.get_match_key_bits();
    write(&k_byte, 1);
    write(&mkb, 1);
    write(&index, sizeof(index));
    write(&meta_group, 1);
    uint8_t const memo_size = static_cast<uint8_t>(memo.size());
    write(&memo_size, 1);
    write(memo.data(), memo.size());
    write(&num_chunks, sizeof(num_chunks));

    // Compression already determined every chunk length, so write the final
    // offsets directly instead of seeking back to patch placeholders.
    uint64_t offset = bytes_written + num_chunks * sizeof(uint64_t);
    for (auto const& chunk : compressed) {
        write(&offset, sizeof(offset));
        offset += sizeof(uint64_t) + chunk.size();
    }
    for (auto const& chunk : compressed) {
        uint64_t const size = chunk.size();
        write(&size, sizeof(size));
        write(chunk.data(), chunk.size());
    }
    if (std::fflush(guard.out) != 0)
        throw std::runtime_error("Failed to flush " + partial);
#ifdef _WIN32
    if (::_commit(fd) != 0)
#else
    if (::fsync(fd) != 0)
#endif
        throw std::runtime_error("Failed to sync " + partial);
    if (std::fclose(std::exchange(guard.out, nullptr)) != 0)
        throw std::runtime_error("Failed to close " + partial);

    // Preserve the existing replace policy: concurrent successful writers may
    // replace the destination, but each publishes its own complete file.
    std::error_code ec;
    std::filesystem::rename(partial, filename, ec);
    if (ec) throw std::runtime_error("Failed to publish " + filename + ": " + ec.message());
    guard.committed = true;
    fsync_parent_dir_best_effort(filename);
    return bytes_written;
}


VerifyResult verify_plot_file(std::string const& filename, size_t n_trials, bool full)
{
    VerifyResult res;
    if (n_trials == 0) return res;

    if (!valid_plot_structure(filename, nullptr))
        throw std::runtime_error("Invalid or truncated plot header/chunk layout: " + filename);
    Prover prover(filename);
    std::unique_ptr<Solver> solver;
    if (full) solver = std::make_unique<Solver>(prover.getProofParams());

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
        if (solver) {
            ProofFragmentCodec codec(prover.getProofParams());
            ProofValidator validator(prover.getProofParams());
            for (auto const& chain : chains) {
                std::array<uint32_t, TOTAL_T1_PAIRS_IN_PROOF> x_bits{};
                size_t index = 0;
                for (auto fragment : chain.chain_links)
                    for (auto x : codec.get_x_bits_from_proof_fragment(fragment)) x_bits[index++] = x;
                auto proofs = solver->solve(x_bits);
                bool valid = false;
                for (auto const& proof : proofs) {
                    auto const links = validator.validate_full_proof(proof, challenge);
                    if (links && *links == chain.chain_links) { valid = true; break; }
                }
                if (!valid) throw std::runtime_error("Quality chain has no valid full proof");
                ++res.full_proofs_validated;
            }
        }
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
