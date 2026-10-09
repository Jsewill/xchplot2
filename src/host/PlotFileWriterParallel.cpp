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

#include "plot/ChunkCompression.hpp"
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
#include <cmath>
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
#include "host/WindowsFile.hpp"
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
bool valid_raw_structure(std::string const& filename, BatchEntry const* expected)
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
    if (k < 18 || k > 28 || (k & 1) || strength < 2 ||
        strength > k - (k < 28 ? 2 : k - 26) - 1) return false;
    std::vector<uint8_t> memo(memo_size);
    if (memo_size && !read(memo.data(), memo.size())) return false;
    if (expected && (!expected->raw || id != expected->group_id || k != expected->k ||
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

bool valid_group_structure(std::string const& filename, BatchEntry const* expected)
{
    std::error_code ec;
    uint64_t const size = std::filesystem::file_size(filename, ec);
    if (ec || size < sizeof(PlotGroupFile::Header)) return false;
    try {
        std::ifstream in(filename, std::ios::binary);
        in.exceptions(std::ios::failbit | std::ios::badbit);
        PlotGroupFile::Header header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (header.magic != PlotGroupFile::MAGIC ||
            header.version != PlotGroupFile::FORMAT_VERSION || header.group_size == 0)
            return false;
        PlotProofParams::validate_input_args(header.k, header.strength);
        std::vector<uint8_t> memo(header.memo_length);
        in.read(reinterpret_cast<char*>(memo.data()), memo.size());
        if (expected && (expected->raw || header.group_size != 1 ||
            header.group_id != PlotGroupId(expected->group_id) ||
            header.k != expected->k || header.strength != expected->strength ||
            header.meta_group != expected->meta_group || memo != expected->memo)) return false;
        uint64_t const begin = sizeof(header) + memo.size();
        uint64_t const chunks = PlotGroupFile::getChunkCountForK(header.k);
        if (header.chunk_index_offset < begin || header.chunk_index_offset >= size ||
            size - header.chunk_index_offset > chunks * 16) return false;
        uint64_t const index_size = size - header.chunk_index_offset;
        std::vector<uint64_t> packed((index_size + 7) / 8);
        in.seekg(header.chunk_index_offset);
        in.read(reinterpret_cast<char*>(packed.data()), index_size);
        BitReader reader(packed, index_size * 8);
        std::vector<uint64_t> sizes(chunks);
        gsz_decode(sizes, header.group_size, reader);
        uint64_t end = begin;
        for (auto bytes : sizes) {
            if (bytes == 0 || bytes > header.chunk_index_offset - end) return false;
            end += bytes;
        }
        return end == header.chunk_index_offset;
    } catch (std::exception const&) {
        return false;
    }
}

bool valid_plot_structure(std::string const& filename, BatchEntry const* expected)
{
    return valid_group_structure(filename, expected) || valid_raw_structure(filename, expected);
}

// Same table as upstream PlotGroupFile::createFSECTable and the existing
// contrib group assembler. A compression task owns its table and buffers.
auto group_compression_table()
{
    std::array<short, 256> norm{};
    std::array<double, 178> weights{};
    double total = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::exp(-double(i) / 256.0);
        total += weights[i];
    }
    int assigned = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        norm[i] = std::max(short(1), short(weights[i] / total * 2048 + 0.5));
        assigned += norm[i];
    }
    norm[0] += short(2048 - assigned);
    std::unique_ptr<FSE_CTable, decltype(&POS2_FSE_freeCTable)> table(
        POS2_FSE_createCTable(177, 11), POS2_FSE_freeCTable);
    if (!table || POS2_FSE_isError(POS2_FSE_buildCTable(table.get(), norm.data(), 177, 11)))
        throw std::runtime_error("cannot build group compression table");
    return table;
}

void compress_group_chunk(std::vector<uint8_t>& output, std::span<uint64_t const> fragments,
    uint64_t start, uint8_t k, FSE_CTable const* table,
    std::vector<uint8_t>& high, std::vector<uint8_t>& ans, BitWriter& bits)
{
    if (fragments.empty()) throw std::invalid_argument("group chunk has no proof fragments");
    high.clear();
    bits.clear();
    uint64_t const threshold = 178ull << (k - 8);
    uint64_t previous = start;
    bool first = true;
    for (auto fragment : fragments) {
        if (!first && fragment == previous) continue;
        first = false;
        auto const delta = fragment - previous;
        previous = fragment;
        auto const quotient = delta / threshold;
        auto const remainder = delta % threshold;
        if (quotient + 1 + k - 8 > 64)
            throw std::invalid_argument("unencodable group fragment delta");
        bits.append((((1ull << quotient) - 1) << (k - 8)) |
            (remainder & ((1ull << (k - 8)) - 1)), uint32_t(quotient + 1 + k - 8));
        high.push_back(uint8_t(remainder >> (k - 8)));
    }
    ans.resize(POS2_FSE_compressBound(high.size()));
    size_t const size = POS2_FSE_compress_usingCTable(
        ans.data(), ans.size(), high.data(), high.size(), table);
    if (size == 0 || POS2_FSE_isError(size))
        throw std::runtime_error("cannot compress group chunk");
    uint64_t leb = size;
    do {
        output.push_back(uint8_t(leb & 0x7f) | (leb > 0x7f ? 0x80 : 0));
        leb >>= 7;
    } while (leb);
    output.insert(output.end(), ans.begin(), ans.begin() + size);
    auto const bytes = bits.asBytes();
    output.insert(output.end(), bytes.begin(), bytes.end());
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

std::array<uint8_t, 32> plot_id_for_group(
    std::array<uint8_t, 32> const& group_id, uint16_t index, uint8_t meta_group)
{
    std::array<uint8_t, 32> id{};
    posCalculatePlotIdForIndex(group_id, id, index, meta_group);
    return id;
}

bool plot_file_matches(std::string const& filename, BatchEntry const& expected)
{
    try { validate_batch_entry(expected); }
    catch (std::invalid_argument const&) { return false; }
    return valid_plot_structure(filename, &expected);
}

size_t write_plot_file_parallel(
    std::string const& filename,
    std::span<uint64_t const> t3_fragments,
    BatchEntry const& entry,
    unsigned thread_count)
{
    validate_batch_entry(entry);
    uint8_t const k = static_cast<uint8_t>(entry.k);
    auto const& memo = entry.memo;
    if (t3_fragments.empty() || (t3_fragments.back() >> (2 * k)))
        throw std::invalid_argument("empty plot or proof fragment exceeds the plot's bit width");

    // Compression tasks share the existing process-wide worker pool. A task
    // coalesces its chunks into one buffer: grouped k28 has 2^22 chunks, so
    // storing a separate vector per chunk would add millions of allocations.
    if (thread_count == 0)
        thread_count = static_cast<unsigned>(WriterThreadPool::instance().size());
    uint64_t const range_per_chunk = 1ull << (k + (entry.raw
        ? PlotFile::CHUNK_SPAN_RANGE_BITS : PlotGroupFile::PROOFS_PER_CHUNK_BITS));
    auto const boundaries = chunk_boundaries_span(t3_fragments, range_per_chunk);
    uint64_t const num_chunks = boundaries.size() - 1;
    if (!entry.raw && num_chunks != PlotGroupFile::getChunkCountForK(k))
        throw std::invalid_argument("incomplete group fragment range");
    uint64_t const tasks_n = std::min<uint64_t>(thread_count, num_chunks);
    uint64_t const chunks_per_task = (num_chunks + tasks_n - 1) / tasks_n;
    std::vector<std::vector<uint8_t>> compressed(tasks_n);
    std::vector<uint64_t> chunk_sizes(num_chunks);
    {
        auto& pool = WriterThreadPool::instance();
        std::vector<std::future<void>> tasks;
        tasks.reserve(tasks_n);
        // Also drain already-submitted work if a later submission throws.
        struct Drain {
            std::vector<std::future<void>>& tasks;
            ~Drain() { for (auto& task : tasks) if (task.valid()) task.wait(); }
        } drain{tasks};
        for (uint64_t task = 0, begin = 0; begin < num_chunks;
             ++task, begin += chunks_per_task) {
            auto const end = std::min(begin + chunks_per_task, num_chunks);
            tasks.emplace_back(pool.submit([&, task, begin, end] {
                auto& output = compressed[task];
                output.reserve((boundaries[end] - boundaries[begin]) * (k + 2) / 8
                    + (end - begin) * 8);
                auto table = entry.raw ? decltype(group_compression_table())(
                    nullptr, POS2_FSE_freeCTable) : group_compression_table();
                std::vector<uint8_t> high, ans;
                BitWriter bits;
                for (uint64_t i = begin; i < end; ++i) {
                    auto const fragments = t3_fragments.subspan(boundaries[i],
                        boundaries[i + 1] - boundaries[i]);
                    auto const offset = output.size();
                    if (entry.raw) {
                        auto const chunk = ChunkCompressor::compressProofFragments(
                            fragments, i * range_per_chunk, k - PlotFile::MINUS_STUB_BITS);
                        uint64_t const size = chunk.size();
                        auto const* bytes = reinterpret_cast<uint8_t const*>(&size);
                        output.insert(output.end(), bytes, bytes + sizeof(size));
                        output.insert(output.end(), chunk.begin(), chunk.end());
                    } else {
                        compress_group_chunk(output, fragments, i * range_per_chunk,
                            k, table.get(), high, ans, bits);
                    }
                    chunk_sizes[i] = output.size() - offset;
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
    int const fd = create_private_temp(partial);
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
    uint8_t const strength = uint8_t(entry.strength);
    uint8_t const meta_group = uint8_t(entry.meta_group);
    uint8_t const memo_size = uint8_t(memo.size());
    BitWriter group_index;
    if (entry.raw) {
        write("pos2", 4);
        uint8_t const version = PlotFile::FORMAT_VERSION;
        write(&version, 1);
        write(entry.group_id.data(), entry.group_id.size());
        write(&k, 1);
        write(&strength, 1);
        uint16_t const index = uint16_t(entry.plot_index);
        write(&index, sizeof(index));
        write(&meta_group, 1);
        write(&memo_size, 1);
        write(memo.data(), memo.size());
        write(&num_chunks, sizeof(num_chunks));
        uint64_t offset = bytes_written + num_chunks * sizeof(uint64_t);
        for (auto size : chunk_sizes) {
            write(&offset, sizeof(offset));
            offset += size;
        }
    } else {
        uint64_t offset = sizeof(PlotGroupFile::Header) + memo.size();
        for (auto size : chunk_sizes) offset += size;
        PlotGroupFile::Header const header{PlotGroupFile::MAGIC,
            PlotGroupFile::FORMAT_VERSION, PlotGroupId(entry.group_id),
            k, strength, 1, meta_group, offset, memo_size};
        gsz_encode(chunk_sizes, 1, group_index);
        write(&header, sizeof(header));
        write(memo.data(), memo.size());
    }
    for (auto const& chunk : compressed) write(chunk.data(), chunk.size());
    if (!entry.raw) {
        auto const bytes = group_index.asBytes();
        write(bytes.data(), bytes.size());
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
#ifdef _WIN32
    // Concurrent Windows replacements can fail with ERROR_ACCESS_DENIED.
    // ponytail: one publication lock; use per-path locks if it limits throughput.
    static std::mutex publish_mutex;
    std::lock_guard<std::mutex> publish_lock(publish_mutex);
    if (!::MoveFileExW(std::filesystem::path(partial).c_str(),
                      std::filesystem::path(filename).c_str(),
                      MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
#else
    std::filesystem::rename(partial, filename, ec);
#endif
    if (ec) throw std::runtime_error("Failed to publish " + filename + ": " + ec.message());
    guard.committed = true;
    fsync_parent_dir_best_effort(filename);
    return bytes_written;
}


VerifyResult verify_plot_file(std::string const& filename, size_t n_trials, bool full)
{
    VerifyResult res;
    if (n_trials == 0) return res;
    bool const grouped = valid_group_structure(filename, nullptr);
    if (!grouped && !valid_raw_structure(filename, nullptr))
        throw std::runtime_error("Invalid, obsolete, or truncated plot header/chunk layout: " + filename);
    std::unique_ptr<GroupProver> group;
    std::unique_ptr<Prover> raw;
    if (grouped) group = std::make_unique<GroupProver>(filename);
    else raw = std::make_unique<Prover>(filename);
    auto const group_params = grouped
        ? PlotGroupParams(group->getPlotGroup().getInfo().group_id,
            group->getPlotGroup().getInfo().k, group->getPlotGroup().getInfo().strength,
            group->getPlotGroup().getInfo().meta_group)
        : raw->getGroupParams();

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < n_trials; ++i) {
        std::array<uint8_t, 32> challenge{};
        for (size_t j = 0; j < 32; j += 8) {
            uint64_t const v = dist(gen);
            std::memcpy(challenge.data() + j, &v, 8);
        }
        std::vector<PlotQualityChains> qualities;
        if (grouped) qualities = group->prove(challenge);
        else qualities.push_back({raw->prove(challenge), raw->getPlotProofParams().get_plot_index()});
        ++res.trials;
        size_t count = 0;
        for (auto const& member : qualities) {
            count += member.quality_chains.size();
            if (!full || member.quality_chains.empty()) continue;
            auto const params = group_params.get_plot_params_for_index(member.plot_index);
            Solver solver(params);
            ProofFragmentCodec codec(params);
            ProofValidator validator(group_params, member.plot_index);
            for (auto const& chain : member.quality_chains) {
                std::array<uint32_t, TOTAL_T1_PAIRS_IN_PROOF> x_bits{};
                size_t index = 0;
                for (auto fragment : chain.chain_links)
                    for (auto x : codec.get_x_bits_from_proof_fragment(fragment)) x_bits.at(index++) = x;
                bool valid = false;
                for (auto const& proof : solver.solve(x_bits)) {
                    auto const links = validator.validate_full_proof(proof, challenge);
                    if (links && *links == chain.chain_links) { valid = true; break; }
                }
                if (!valid) throw std::runtime_error("Quality chain has no valid full proof");
                ++res.full_proofs_validated;
            }
        }
        res.proofs_found += count;
        if (count) ++res.challenges_with_proof;
    }
    return res;
}

std::vector<uint64_t> read_plot_file_fragments(std::string const& filename)
{
    if (valid_group_structure(filename, nullptr)) {
        auto group = PlotGroupFile::open(filename);
        if (group.getInfo().group_size != 1)
            throw std::invalid_argument("flat fragment reads require a single-plot group");
        auto data = group.readProofsInRange({0, (1ull << (2 * group.getInfo().k)) - 1});
        return std::move(data.front());
    }
    if (!valid_raw_structure(filename, nullptr))
        throw std::runtime_error("Invalid raw plot layout: " + filename);
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
    if (testnet) throw std::invalid_argument("PoS2 1.0 removes testnet-specific plots");
    auto const params = PlotProofParams::create_raw(PlotId(plot_id_32), k, strength);
    Plotter::Options opts{};
    opts.validate = false;
    opts.verbose  = verbose;
    Plotter plotter(params);
    PlotData plot = plotter.run(opts);
    return std::move(plot.t3_proof_fragments);
}

} // namespace pos2gpu
