// Same-file construction of ordinary upstream v2 groups. The temporary redo
// workspace is part of the destination file and is removed before publication.
#pragma once
#include "host/GroupCodec.hpp"
#include "host/GroupFile.hpp"
#include "pos2_keygen.h"

#include <cerrno>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <optional>
#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/falloc.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <sys/vfs.h>
#endif

#ifndef GROW_IDENTITY
#define GROW_IDENTITY "unversioned-testing-build"
#endif

namespace group_grow
{
using group_tools::require;
using Header = PlotGroupFile::Header;

struct ConstructionInterrupted : std::runtime_error
{
    ConstructionInterrupted() : std::runtime_error("interrupted; repeat the command with --resume") {}
};

struct Config
{
    uint64_t capacity_bytes = uint64_t(std::numeric_limits<int64_t>::max());
    uint64_t window_bytes = 16ull * 1024 * 1024;
    uint64_t codec_bytes = 256ull * 1024 * 1024;
    uint64_t checkpoint_bytes = 256ull * 1024 * 1024;
    bool io_overlap = true;
    bool release_cache = false;
    unsigned workers = 1;
    std::shared_ptr<CodecScheduler> cpu_scheduler;
    // Optional computation service; storage and durable acceptance stay local.
    // Its retained buffers use a separate controller-admitted RAM partition.
    std::shared_ptr<RemoteEncoder> remote_encoder;
    unsigned merge_inputs = 2;
    uint64_t memory_limit_bytes = UINT64_MAX;
    // Runtime controller authority, never serialized into a checkpoint. Check
    // only before bounded work or after a recoverable transaction, and join
    // borrowed windows before allowing the exception to leave their owner.
    std::function<bool()> cancelled;
    void check_running() const
    { if (cancelled && cancelled()) throw ConstructionInterrupted{}; }
};
struct Identity
{
    Header header{};
    std::vector<uint8_t> memo;
    uint64_t bytes = 0;
    bool pending = false;
    uint16_t old_members = 0, target_members = 0;
    uint64_t pending_peak_bytes = 0;
};
// A stopped construction is reviewed through its existing loader, never a
// writable engine. This evidence binds that inspection; the checkpoint remains
// the authority for recovery and is not copied into another mutable manifest.
struct ConstructionStamp
{
    uint64_t device = 0, inode = 0, bytes = 0, allocated = 0;
    int64_t modified = 0, changed = 0;
    uint32_t modified_ns = 0, changed_ns = 0;
    bool operator==(ConstructionStamp const&) const = default;
};
struct ConstructionInspection
{
    Identity identity;
    ConstructionStamp target;
    std::array<uint8_t, 32> authority_digest{};
    uint64_t inspection_memory_bytes = 0, recovery_memory_bytes = 0;
    uint16_t committed_members = 0;
    bool checkpoint = false, can_append = false;
};
struct ConstructionInspectionLimits
{
    uint64_t memory_bytes = 0;
    std::chrono::steady_clock::time_point deadline{};
    std::function<bool()> cancelled;
    void check(uint64_t retained_bytes = 0) const
    {
        require(retained_bytes <= memory_bytes, "construction inspection exceeds its memory grant");
        require(!cancelled || !cancelled(), "construction inspection was cancelled");
        require(std::chrono::steady_clock::now() < deadline, "construction inspection exceeded its time limit");
    }
};
inline ConstructionStamp construction_stamp(int fd, std::string const& path, bool directory = false)
{
    file_io::Stat opened{}, named{};
    auto valid = [&](file_io::Stat const& value)
    { return directory ? file_io::is_directory(value.st_mode) : file_io::is_regular(value.st_mode) && value.st_nlink == 1; };
    require(!file_io::inspect_fd(fd, &opened) && valid(opened) &&
        opened.st_size >= 0 && opened.st_blocks >= 0 && !file_io::inspect_link(path.c_str(), &named) &&
        valid(named) && named.st_dev == opened.st_dev && named.st_ino == opened.st_ino,
        "construction target filename or links changed during inspection");
    require(uint64_t(opened.st_blocks) <= UINT64_MAX / 512, "construction allocation overflow");
#ifdef __APPLE__
    auto const modified = opened.st_mtimespec, changed = opened.st_ctimespec;
#else
    auto const modified = opened.st_mtim, changed = opened.st_ctim;
#endif
    return {uint64_t(opened.st_dev), uint64_t(opened.st_ino), uint64_t(opened.st_size), uint64_t(opened.st_blocks) * 512,
        modified.tv_sec, changed.tv_sec, uint32_t(modified.tv_nsec), uint32_t(changed.tv_nsec)};
}
struct Plan
{
    bool accepted = false;
    // Fresh planning stores proven upper bounds; apply replaces them with the
    // actual encoded sizes in place, without retaining another full table.
    bool fresh_single_pass = false;
    uint64_t final_bytes = 0, peak_file_bytes = 0, workspace_bytes = 0;
    uint64_t minimum_peak_file_bytes = 0;
    uint64_t largest_chunk_bytes = 0;
    uint16_t new_members = 0;
    std::string reason;
    Header header{};
    std::vector<uint64_t> sizes;
    std::array<uint8_t, 32> members_digest{};
};
struct Result
{
    uint64_t final_bytes = 0, peak_file_bytes = 0, workspace_bytes = 0;
    uint16_t members = 0;
};
struct AppendResult
{
    bool accepted = false, fresh_single_pass = false;
    uint64_t attempted_final_bytes = 0, attempted_peak_bytes = 0;
    std::string reason;
    Result result;
};
enum class WorkPhase : size_t { plan, encode, merge, pack, layout, count };
struct WindowBudgetError : std::runtime_error
{ WindowBudgetError() : std::runtime_error("chunk exceeds the recoverable source-window budget") {} };
struct PhaseStats
{
    uint64_t elapsed_ns = 0, read_ns = 0, write_ns = 0, sync_ns = 0, codec_ns = 0, decode_ns = 0;
    uint64_t read_bytes = 0, written_bytes = 0, read_calls = 0, write_calls = 0, syncs = 0;
};
struct Stats
{
    uint64_t encode_chunks = 0, sized_chunks = 0, write_calls = 0;
    uint64_t checkpoints = 0, syncs = 0, overlap_windows = 0;
    uint64_t read_calls = 0, read_bytes = 0, written_bytes = 0;
    uint64_t read_ns = 0, write_ns = 0, sync_ns = 0;
    uint64_t merge_passes = 0, reclaimed_bytes = 0, max_active_runs = 0;
    uint64_t codec_ns = 0, decode_ns = 0;
    uint64_t layout_moves = 0, layout_bytes = 0, layout_batches = 0, placed_blocks = 0;
    uint64_t contribution_workers = 0, packing_workers = 0, packing_overlap_windows = 0, merge_inputs = 0;
    std::array<PhaseStats, size_t(WorkPhase::count)> phases{};
};

// These measure time inside storage syscalls, including kernel waits. They do
// not claim physical device traffic: buffered reads may be served from cache.
inline uint64_t elapsed_ns(std::chrono::steady_clock::time_point begin)
{
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count());
}

// A phase observes final counters only after its codec tasks have drained.
// Phase times include storage waits; codec and decoder times are measured
// separately around their actual work.
class PhaseTimer
{
public:
    PhaseTimer(Stats& stats, WorkPhase phase) : stats_(stats), phase_(phase),
        before_{0, stats.read_ns, stats.write_ns, stats.sync_ns, stats.codec_ns, stats.decode_ns,
            stats.read_bytes, stats.written_bytes, stats.read_calls, stats.write_calls, stats.syncs} {}
    ~PhaseTimer() { finish(); }
    void finish()
    {
        if (finished_) return;
        auto& value = stats_.phases[size_t(phase_)];
        value.elapsed_ns += elapsed_ns(begin_);
        value.read_ns += stats_.read_ns - before_.read_ns;
        value.write_ns += stats_.write_ns - before_.write_ns;
        value.sync_ns += stats_.sync_ns - before_.sync_ns;
        value.codec_ns += stats_.codec_ns - before_.codec_ns;
        value.decode_ns += stats_.decode_ns - before_.decode_ns;
        value.read_bytes += stats_.read_bytes - before_.read_bytes;
        value.written_bytes += stats_.written_bytes - before_.written_bytes;
        value.read_calls += stats_.read_calls - before_.read_calls;
        value.write_calls += stats_.write_calls - before_.write_calls;
        value.syncs += stats_.syncs - before_.syncs;
        finished_ = true;
    }
private:
    Stats& stats_;
    WorkPhase phase_;
    PhaseStats before_;
    std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
    bool finished_ = false;
};

// Optional :N chooses the Nth occurrence of a durable boundary. Abrupt exits
// deliberately skip cleanup so the tests exercise the on-disk protocol.
inline void grow_failpoint(std::string const& name)
{
    char const* value = std::getenv("XCHPLOT2_GROW_FAILPOINT");
    if (!value) return;
    std::string setting(value);
    auto const colon = setting.find(':');
    if (setting.substr(0, colon) != name) return;
    static std::atomic<uint64_t> hits{0};
    auto const target = colon == setting.npos ? 1 : group_tools::number(setting.substr(colon + 1).c_str());
    if (hits.fetch_add(1, std::memory_order_relaxed) + 1 == target) std::_Exit(86);
}

class Engine
{
public:
    Engine(std::string path, Config config, bool resume = false,
           std::optional<Identity> initial = std::nullopt);
    ~Engine();
    Engine(Engine const&) = delete;
    Engine& operator=(Engine const&) = delete;

    Identity const& identity() const { return identity_; }
    // Conservative, fixed-memory admission probe. Ambiguous reads and private
    // markers keep the existing serial recovery envelope. An orphan tail can
    // still be discovered while decoding an otherwise ordinary index.
    static bool pending(std::string const& path);
    static ConstructionInspection inspect(std::string const& path, Config config, ConstructionInspectionLimits const& limits,
        std::optional<Identity> const& initial = {});
    static uint64_t inspection_memory_bytes(uint8_t k)
    { return uint64_t(PlotGroupFile::getChunkCountForK(k)) * 32 + 65536; }
    static uint64_t memory_bytes(uint8_t k, Config const& config, uint16_t batch_members = 1,
                                 uint16_t wide_members = 0, uint16_t compact_members = 0);
    uint64_t memory_bytes(uint16_t batch_members = 1, uint16_t wide_members = 0,
                          uint16_t compact_members = 0) const
    { return memory_bytes(identity_.header.k, config_, batch_members, wide_members, compact_members); }
    uint64_t memory_bytes(Members members) const;
    Plan plan(Members members, MemberBatchHash const* hash = nullptr);
    std::optional<Plan> plan_fresh(Members members, MemberBatchHash const* hash = nullptr);
    Result apply(Plan& plan, Members members);
    AppendResult append(Members members, MemberBatchHash const* hash = nullptr, bool reject_window = false);
    Result resume(Members members, MemberBatchHash const* hash = nullptr);
    void confirm_identity();
    Stats const& stats() const { return stats_; }
    unsigned encoding_workers() const { return active_workers_; }
    // The owner calls between operations, after joining any borrowed windows.
    // Workers share codec_bytes; task descriptors retain their own small reserve.
    void set_workers(unsigned workers)
    {
        auto cfg = config_; cfg.workers = workers;
        require(workers && memory_bytes(identity_.header.k, cfg) <= config_.memory_limit_bytes,
            "ordinary worker grant exceeds memory admission");
        if (workers == config_.workers) return;
        encoder_.reset(); config_.workers = active_workers_ = workers;
    }
    uint64_t largest_chunk() const
    { return old_sizes_.empty() ? 0 : *std::max_element(old_sizes_.begin(), old_sizes_.end()); }
    void retune(uint64_t window, uint64_t checkpoint)
    {
        require(!identity_.pending && window && checkpoint, "only a complete ordinary group can change its I/O budgets");
        encoder_.reset();
        config_.window_bytes = window;
        config_.checkpoint_bytes = checkpoint;
    }

private:
    struct ReadOnly {};
    Engine(std::string path, Config config, ReadOnly, ConstructionInspectionLimits const& limits, std::optional<Identity> const& initial, ConstructionStamp& target,
           std::array<uint8_t, 32>& authority);
    static constexpr uint8_t construction_version = 0xf0, finalizing_version = 0xf1;
    static constexpr size_t record_bytes = 1024;
    static constexpr size_t max_tasks = 65536;
    static constexpr size_t digest_words = 1024;
    struct Record;
    struct Window
    {
        uint64_t first = 0, count = 0, source_begin = 0;
        std::vector<uint8_t> source;
        std::vector<ChunkInput> tasks;
        std::vector<FragmentRegion> regions;
        std::vector<uint64_t> sizes;
        std::vector<EncodedBlock> blocks;
    };
    // One bounded producer, reused across windows. Joining on unwind keeps
    // borrowed member spans alive even when the writer or encoder fails.
    class WindowWorker
    {
    public:
        WindowWorker(Engine& engine, Members members) : engine_(engine), members_(members),
            thread_([this] { run(); }) {}
        ~WindowWorker()
        {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_one();
            thread_.join();
        }
        std::future<std::unique_ptr<Window>> encode(uint64_t first, uint64_t count, uint64_t source)
        {
            std::packaged_task<std::unique_ptr<Window>()> task([this, first, count, source]
            { return engine_.encode_owned(first, count, source, members_); });
            auto future = task.get_future();
            {
                std::lock_guard lock(mutex_);
                require(!task_.valid(), "window encoder queue is full");
                task_ = std::move(task);
            }
            ready_.notify_one();
            return future;
        }
    private:
        void run()
        {
            for (;;)
            {
                std::packaged_task<std::unique_ptr<Window>()> task;
                {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [this] { return stopping_ || task_.valid(); });
                    if (stopping_ && !task_.valid()) return;
                    task = std::move(task_);
                }
                task();
            }
        }
        Engine& engine_;
        Members members_;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::packaged_task<std::unique_ptr<Window>()> task_;
        bool stopping_ = false;
        std::thread thread_;
    };
    std::string path_;
    Config config_;
    ConstructionInspectionLimits const* inspection_limits_ = nullptr;
    Identity identity_;
    int fd_ = -1;
    uint64_t device_ = 0, inode_ = 0;
    uint64_t record_alignment_ = record_bytes;
    bool created_ = false, finalizing_ = false;
    bool initialization_orphan_ = false, confirmed_ = false;
    std::vector<uint64_t> old_sizes_;
    std::vector<FragmentBoundaries> member_boundaries_;
    std::unique_ptr<Record> record_;
    std::unique_ptr<ParallelEncoder> encoder_;
    bool cow_filesystem_ = false;
    bool overlap_ = false;
    unsigned active_workers_ = 0;
    mutable Stats stats_{};
    // Only immutable, uniquely identified compact members may reuse this digest.
    // Public apply/resume compare the complete identity sequence every time.
    mutable std::unique_ptr<uint64_t[]> digest_ids_;
    mutable size_t digest_count_ = 0;
    mutable std::array<uint8_t, 32> digest_{};

    void read_at(uint64_t offset, std::span<uint8_t> bytes) const;
    void write_at(uint64_t offset, std::span<uint8_t const> bytes) const;
    void write_blocks(uint64_t offset, std::span<EncodedBlock const> blocks,
                      uint64_t begin, uint64_t length) const;
    void sync() const;
    void release_clean(uint64_t begin, uint64_t end) const;
    uint64_t file_bytes() const;
    void check_path() const;
    void discover(bool allow_resume, std::optional<Identity> const& initial,
                  std::array<uint8_t, 32>* authority = nullptr);
    uint64_t preallocate(uint64_t end, uint64_t minimum);
    void checkpoint(Record& record);
    uint64_t record_offset(uint64_t storage_bytes, unsigned slot) const;
    void replay(Record& record);
    void rewrite(Plan const& plan, Members members, Record& record);
    Result finish(Plan const& plan);
    Result create(Plan& plan, Members members, Record& record);
    Result apply_prepared(Plan& plan, Members members);
    std::unique_ptr<Window> encode_owned(uint64_t first, uint64_t count, uint64_t source_begin, Members members);
    void load_index(Header const& header, uint64_t bytes, std::vector<uint64_t>& sizes) const;
    void scan_sizes(Header const& header, uint64_t first, uint64_t count, uint64_t begin,
                    uint64_t end, std::vector<uint64_t>& sizes) const;
    void prepare_members(Members members, uint16_t first);
    uint64_t window_codec_bytes() const { return config_.codec_bytes / (overlap_ ? 2 : 1); }
    void reset_encoding();
    void admit_chunk(uint64_t work);
    void prepare_encoding(uint64_t edge, bool reverse, Members members, uint64_t stop = UINT64_MAX);
    std::array<uint8_t, 32> digest_members(Members members, MemberBatchHash const* hash = nullptr) const;
    uint64_t window_count(uint64_t edge, bool reverse, Members members, uint64_t stop = UINT64_MAX);
    std::span<EncodedBlock const> encode_window(uint64_t first, uint64_t count, uint64_t source_begin,
                       std::span<uint64_t const> source_sizes, Members members,
                       std::vector<uint8_t>& source, std::vector<ChunkInput>& tasks,
                       std::vector<FragmentRegion>& regions,
                       std::span<uint64_t> sizes, bool size_only = false);
};

inline std::array<uint8_t, 32> grow_digest(std::span<uint8_t const> bytes, CodecScheduler* scheduler = nullptr)
{
    CodecScheduler::Lease compute(scheduler);
    struct sha256 context;
    std::array<uint8_t, 32> result{};
    sha256_init(&context);
    if (!bytes.empty()) sha256_update(&context, bytes.data(), bytes.size());
    sha256_sum(&context, result.data());
    return result;
}
inline uint64_t grow_add(uint64_t a, uint64_t b)
{
    require(b <= uint64_t(INT64_MAX) && a <= uint64_t(INT64_MAX) - b, "group file offset overflow");
    return a + b;
}
inline std::array<uint8_t, 32> construction_digest(ConstructionInspection const& inspected,
    std::array<uint8_t, 32> const& checkpoint, std::string_view protocol)
{
    struct sha256 context; sha256_init(&context);
    sha256_update(&context, reinterpret_cast<uint8_t const*>(protocol.data()), protocol.size());
    sha256_update(&context, checkpoint.data(), checkpoint.size());
    auto number = [&](uint64_t value)
    {
        std::array<uint8_t, 8> bytes{};
        for (auto& byte : bytes) { byte = uint8_t(value); value >>= 8; }
        sha256_update(&context, bytes.data(), bytes.size());
    };
    auto const& stamp = inspected.target;
    for (auto value : {stamp.device, stamp.inode, stamp.bytes, stamp.allocated,
        uint64_t(stamp.modified), uint64_t(stamp.changed), uint64_t(stamp.modified_ns), uint64_t(stamp.changed_ns)}) number(value);
    auto const& header = inspected.identity.header;
    sha256_update(&context, reinterpret_cast<uint8_t const*>(&header), sizeof(header));
    if (!inspected.identity.memo.empty()) sha256_update(&context, inspected.identity.memo.data(), inspected.identity.memo.size());
    number(inspected.checkpoint); number(inspected.committed_members); number(inspected.can_append);
    std::array<uint8_t, 32> result{}; sha256_sum(&context, result.data()); return result;
}

struct Engine::Record
{
    std::array<char, 16> magic{};
    uint64_t sequence = 0;
    uint32_t format = 1;
    uint8_t pending = 0, fresh = 0;
    Header original{}, target{};
    uint64_t original_bytes = 0, final_bytes = 0, storage_bytes = 0, record_alignment = 0;
    uint64_t redo_offset = 0, redo_capacity = 0, redo_bytes = 0, redo_chunks = 0;
    uint64_t remaining = 0, old_frontier = 0, new_frontier = 0;
    uint64_t largest_chunk = 0;
    std::array<uint8_t, 255> memo{};
    std::array<uint8_t, 32> members_digest{}, redo_digest{};
    std::array<char, 64> build{};
};

inline bool Engine::pending(std::string const& path)
{
    int fd = file_io::open_file(path.c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow);
    if (fd < 0) return true;
    struct Close { int fd; ~Close() { file_io::close_file(fd); } } close{fd};
    file_io::Stat info{}; file_io::Space allocation{}; long const page = file_io::page_size();
    if (file_io::inspect_fd(fd, &info) || !file_io::is_regular(info.st_mode) || info.st_size < file_io::Offset(sizeof(Header)) ||
        file_io::space_fd(fd, &allocation) || page <= 0 || info.st_blksize <= 0) return true;
    auto read = [&](uint64_t offset, std::span<uint8_t> bytes)
    {
        while (!bytes.empty())
        {
            auto n = file_io::read_at(fd, bytes.data(), bytes.size(), file_io::Offset(offset));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
        }
        return true;
    };
    Header header{};
    if (!read(0, {reinterpret_cast<uint8_t*>(&header), sizeof(header)}) ||
        header.magic != PlotGroupFile::MAGIC || header.version != PlotGroupFile::FORMAT_VERSION) return true;
    uint64_t const alignment = std::max({uint64_t(record_bytes), uint64_t(page), uint64_t(info.st_blksize),
        uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize)});
    if (uint64_t(info.st_size) < alignment + record_bytes) return false;
    uint64_t const last = (uint64_t(info.st_size) - record_bytes) / alignment * alignment;
    for (uint64_t offset : {last - alignment, last})
    {
        std::array<uint8_t, record_bytes> bytes{};
        if (!read(offset, bytes)) return true;
        auto const digest = grow_digest(std::span(bytes).first(record_bytes - 32));
        if (std::equal(digest.begin(), digest.end(), bytes.end() - 32) &&
            !std::memcmp(bytes.data(), "XCHPLOT2GROWv1\0\0\0", 16)) return true;
    }
    return false;
}

inline Engine::Engine(std::string path, Config config, bool allow_resume,
                      std::optional<Identity> initial)
    : path_(std::move(path)), config_(config), active_workers_(config.workers)
{
    require(config_.window_bytes && config_.window_bytes <= SIZE_MAX && config_.codec_bytes && config_.checkpoint_bytes &&
            config_.workers && config_.capacity_bytes <= uint64_t(INT64_MAX), "invalid growth budget");
    reset_encoding();
    if (initial && !allow_resume)
    {
        fd_ = file_io::open_file(path_.c_str(), file_io::create | file_io::exclusive | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600);
        created_ = fd_ >= 0;
    }
    else fd_ = file_io::open_file(path_.c_str(), file_io::read_write | file_io::close_on_exec | file_io::no_follow);
    if (fd_ < 0 && errno == ENOENT && initial)
    {
        fd_ = file_io::open_file(path_.c_str(), file_io::create | file_io::exclusive | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600);
        created_ = fd_ >= 0;
    }
    require(fd_ >= 0, "cannot open growth target: " + path_ + ": " + std::strerror(errno));
    try
    {
        require(file_io::lock_file(fd_, file_io::lock_exclusive | file_io::lock_nonblocking) == 0, "growth target is already in use");
        file_io::Stat info{};
        require(file_io::inspect_fd(fd_, &info) == 0 && file_io::is_regular(info.st_mode) && info.st_nlink == 1,
                "growth target must be a regular file with one link");
        device_ = info.st_dev;
        inode_ = info.st_ino;
        file_io::Space allocation{};
        long const page_size = file_io::page_size();
        require(file_io::space_fd(fd_, &allocation) == 0 && allocation.f_bsize && info.st_blksize > 0 && page_size > 0,
                "cannot determine recovery record alignment");
        record_alignment_ = std::max({uint64_t(record_bytes), uint64_t(info.st_blksize),
            uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize), uint64_t(page_size)});
#ifdef __linux__
        struct statfs filesystem{};
        require(::fstatfs(fd_, &filesystem) == 0, "cannot inspect target filesystem");
        cow_filesystem_ = unsigned(filesystem.f_type) == 0x9123683eu;
#endif
        discover(allow_resume, initial);
    }
    catch (...)
    {
        if (created_ && file_bytes() == 0) file_io::unlink_file(path_.c_str());
        file_io::close_file(fd_);
        fd_ = -1;
        throw;
    }
}
inline Engine::~Engine()
{
    if (fd_ >= 0)
    {
#ifdef __linux__
        // The controller has finished with this synced group. Release its
        // cached pages so completed files do not consume the next group's RAM.
        // Advice is best effort; retain cache while a group is still growing.
        if (stats_.syncs && !identity_.pending && !finalizing_)
            (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
#endif
        if (created_ && file_bytes() == 0)
        {
            file_io::Stat info{};
            if (file_io::inspect_link(path_.c_str(), &info) == 0 && uint64_t(info.st_dev) == device_ &&
                uint64_t(info.st_ino) == inode_) file_io::unlink_file(path_.c_str());
        }
        file_io::close_file(fd_);
    }
}
inline Engine::Engine(std::string path, Config config, ReadOnly, ConstructionInspectionLimits const& limits, std::optional<Identity> const& initial, ConstructionStamp& target,
                      std::array<uint8_t, 32>& authority)
    : path_(std::move(path)), config_(config), inspection_limits_(&limits)
{
    limits.check(65536);
    require(config_.window_bytes && config_.window_bytes <= SIZE_MAX && config_.codec_bytes && config_.checkpoint_bytes &&
        config_.workers && config_.capacity_bytes <= uint64_t(INT64_MAX), "invalid growth budget");
    require(config_.codec_bytes >= ChunkCodec::estimate_base(), "codec budget cannot hold one worker");
    fd_ = file_io::open_file(path_.c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow);
    require(fd_ >= 0, "cannot inspect growth target: " + path_);
    try
    {
        target = construction_stamp(fd_, path_); auto const& before = target;
        device_ = before.device; inode_ = before.inode;
        file_io::Stat info{}; file_io::Space allocation{}; long const page = file_io::page_size();
        require(!file_io::inspect_fd(fd_, &info) && !file_io::space_fd(fd_, &allocation) &&
            allocation.f_bsize && info.st_blksize > 0 && page > 0, "cannot inspect growth record alignment");
        record_alignment_ = std::max({uint64_t(record_bytes), uint64_t(info.st_blksize),
            uint64_t(allocation.f_bsize), uint64_t(allocation.f_frsize), uint64_t(page)});
        discover(true, initial, &authority);
        require(!record_ || record_->largest_chunk <= config_.window_bytes,
            "saved growth chunks exceed the current window grant; restore a compatible window before recovery");
        require(before.bytes <= config_.capacity_bytes && identity_.bytes <= config_.capacity_bytes,
            "saved growth workspace exceeds capacity");
        require(construction_stamp(fd_, path_) == before, "growth target changed during inspection");
    }
    catch (...) { file_io::close_file(fd_); fd_ = -1; throw; }
}
inline ConstructionInspection Engine::inspect(std::string const& path, Config config, ConstructionInspectionLimits const& limits,
                                              std::optional<Identity> const& initial)
{
    ConstructionInspection result; std::array<uint8_t, 32> authority{};
    Engine engine(path, config, ReadOnly{}, limits, initial, result.target, authority); result.identity = engine.identity_;
    result.checkpoint = bool(engine.record_); result.committed_members = engine.identity_.old_members;
    result.can_append = !engine.identity_.pending;
    result.inspection_memory_bytes = inspection_memory_bytes(result.identity.header.k);
    result.recovery_memory_bytes = memory_bytes(result.identity.header.k, config,
        uint16_t(result.identity.target_members - result.identity.old_members), 0,
        uint16_t(result.identity.target_members - result.identity.old_members));
    result.authority_digest = construction_digest(result, authority, "ordinary");
    require(construction_stamp(engine.fd_, path) == result.target, "growth target changed during inspection"); return result;
}
inline uint64_t Engine::file_bytes() const
{
    file_io::Stat info{};
    require(file_io::inspect_fd(fd_, &info) == 0 && info.st_size >= 0, "cannot inspect growth target size");
    return uint64_t(info.st_size);
}
inline void Engine::check_path() const
{
    file_io::Stat info{};
    require(file_io::inspect_link(path_.c_str(), &info) == 0 && file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
                uint64_t(info.st_dev) == device_ && uint64_t(info.st_ino) == inode_,
            "growth target filename or links changed while open");
}
inline void Engine::read_at(uint64_t offset, std::span<uint8_t> bytes) const
{
    config_.check_running();
    if (inspection_limits_) inspection_limits_->check();
    grow_add(offset, bytes.size());
    while (!bytes.empty())
    {
        if (inspection_limits_) inspection_limits_->check();
        auto const begin = std::chrono::steady_clock::now();
        auto const n = file_io::read_at(fd_, bytes.data(), std::min(bytes.size(), size_t(file_io::max_transfer)), file_io::Offset(offset));
        stats_.read_ns += elapsed_ns(begin); ++stats_.read_calls;
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "cannot read growth target: " + std::string(std::strerror(errno)));
        stats_.read_bytes += uint64_t(n);
        offset += uint64_t(n);
        bytes = bytes.subspan(size_t(n));
    }
}
inline void Engine::write_at(uint64_t offset, std::span<uint8_t const> bytes) const
{
    grow_add(offset, bytes.size());
    while (!bytes.empty())
    {
        auto const begin = std::chrono::steady_clock::now();
        auto const n = file_io::write_at(fd_, bytes.data(), std::min(bytes.size(), size_t(file_io::max_transfer)), file_io::Offset(offset));
        stats_.write_ns += elapsed_ns(begin);
        ++stats_.write_calls;
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "cannot write growth target; retain this file and resume after resolving: " +
                       std::string(std::strerror(errno)));
        stats_.written_bytes += uint64_t(n);
        offset += uint64_t(n);
        bytes = bytes.subspan(size_t(n));
    }
}
inline void Engine::write_blocks(uint64_t offset, std::span<EncodedBlock const> blocks,
                                 uint64_t begin, uint64_t length) const
{
    grow_add(offset, length);
    uint64_t const end = grow_add(begin, length);
    // POSIX guarantees at least 16 vectors. Borrow codec blocks without another
    // window-sized copy, and handle partial writes just as write_at does.
    std::array<file_io::IoVec, 16> vectors{};
    size_t count = 0, buffered = 0;
    uint64_t written = 0, block_begin = 0;
    auto flush = [&]
    {
        size_t first = 0;
        while (first < count)
        {
            auto const started = std::chrono::steady_clock::now();
            auto const n = file_io::writev_at(fd_, vectors.data() + first, int(count - first), file_io::Offset(offset));
            stats_.write_ns += elapsed_ns(started);
            ++stats_.write_calls;
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "cannot write growth target; retain this file and resume after resolving: " +
                           std::string(std::strerror(errno)));
            stats_.written_bytes += uint64_t(n);
            offset += uint64_t(n);
            written += uint64_t(n);
            size_t used = size_t(n);
            while (first < count && used >= vectors[first].iov_len)
            {
                used -= vectors[first].iov_len;
                ++first;
            }
            if (used)
            {
                vectors[first].iov_base = static_cast<uint8_t*>(vectors[first].iov_base) + used;
                vectors[first].iov_len -= used;
            }
        }
        count = buffered = 0;
    };
    for (auto const& block : blocks)
    {
        uint64_t const block_end = grow_add(block_begin, block.bytes.size());
        uint64_t const lo = std::max(begin, block_begin), hi = std::min(end, block_end);
        if (lo < hi)
        {
            auto bytes = block.bytes.subspan(size_t(lo - block_begin), size_t(hi - lo));
            while (!bytes.empty())
            {
                size_t const size = std::min(bytes.size(), size_t(file_io::max_transfer) - buffered);
                vectors[count++] = {const_cast<uint8_t*>(bytes.data()), size};
                buffered += size;
                bytes = bytes.subspan(size);
                if (count == vectors.size() || buffered == size_t(file_io::max_transfer)) flush();
            }
        }
        block_begin = block_end;
    }
    flush();
    require(written == length, "encoded output block size mismatch");
}
inline void Engine::sync() const
{
    int result;
    auto const begin = std::chrono::steady_clock::now();
    do { result = file_io::flush(fd_); } while (result && errno == EINTR);
    stats_.sync_ns += elapsed_ns(begin);
    require(result == 0, "cannot sync growth target; retain this file for resume");
    ++stats_.syncs;
}
inline void Engine::release_clean(uint64_t begin, uint64_t end) const
{
#ifdef __linux__
    if (!config_.release_cache || begin >= end) return;
    // Called only after data and its advanced checkpoint are durable. Keep
    // partial pages at both edges; they may still contain unread source bytes.
    uint64_t const remainder = begin % record_alignment_;
    if (remainder) begin += std::min(end - begin, record_alignment_ - remainder);
    end -= end % record_alignment_;
    if (begin < end) (void)::posix_fadvise(fd_, file_io::Offset(begin), file_io::Offset(end - begin), POSIX_FADV_DONTNEED);
#else
    (void)begin; (void)end;
#endif
}
inline uint64_t Engine::memory_bytes(uint8_t k, Config const& config, uint16_t batch_members,
                                     uint16_t wide_members, uint16_t compact_members)
{
    require(k >= 18 && k <= 28 && !(k & 1), "invalid group k for memory admission");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
    // Fixed chunk-size tables, bounded source windows, aggregate codec scratch,
    // the packed index and its replacement writer, task descriptors and spans.
    uint64_t const lanes = config.io_overlap ? 2 : 1;
    uint64_t result = grow_add(grow_add(config.window_bytes, config.io_overlap ? config.window_bytes : 0),
                               config.codec_bytes);
    result = grow_add(result, chunks * (sizeof(uint64_t) * 4));
    // Bounded logical-fragment decode buffer plus conservative SHA/FFI stack.
    result = grow_add(result, digest_words * sizeof(uint64_t) + 4096);
    result = grow_add(result, lanes * std::min<uint64_t>(chunks, max_tasks) *
        (sizeof(ChunkInput) + sizeof(uint64_t) * 4 + uint64_t(batch_members) * sizeof(FragmentRegion)));
    require(compact_members <= batch_members && wide_members <= batch_members - compact_members,
            "invalid member storage counts");
    result = grow_add(result, uint64_t(batch_members) * sizeof(FragmentBoundaries));
    result = grow_add(result, uint64_t(batch_members - compact_members) * (chunks + 1) * sizeof(uint32_t));
    result = grow_add(result, uint64_t(wide_members) * (chunks + 1) * (sizeof(size_t) - sizeof(uint32_t)));
    if (compact_members == batch_members)
        result = grow_add(result, uint64_t(batch_members) * sizeof(uint64_t));
    result = grow_add(result, ParallelEncoder::memory_bytes(config.workers, !config.cpu_scheduler));
    return grow_add(result, sizeof(Engine) + sizeof(Record) + sizeof(WindowWorker) + record_bytes * 2);
}

inline uint64_t Engine::memory_bytes(Members members) const
{
    require(members.size() <= UINT16_MAX, "invalid memory-admission membership");
    uint16_t compact = 0, wide = 0;
    for (size_t i = 0; i < members.size(); ++i)
    {
        auto const member = members[i];
        if (member.compact()) ++compact;
        else if (member.size() > UINT32_MAX) ++wide;
    }
    return memory_bytes(uint16_t(members.size()), wide, compact);
}

inline void Engine::load_index(Header const& header, uint64_t bytes, std::vector<uint64_t>& sizes) const
{
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(header.k);
    require(header.group_size && header.chunk_index_offset >= sizeof(Header) + header.memo_length &&
        header.chunk_index_offset < bytes, "invalid group index bounds");
    uint64_t const available = std::min(bytes - header.chunk_index_offset, chunks * 8);
    std::vector<uint8_t> packed(size_t(available), 0);
    read_at(header.chunk_index_offset, packed);
    sizes.resize(size_t(chunks));
    uint64_t position = 0, payload = sizeof(Header) + header.memo_length;
    auto bit = [&]()
    {
        require(position / 8 < packed.size(), "truncated group index");
        bool const value = (packed[size_t(position / 8)] >> (position % 8)) & 1;
        ++position;
        return value;
    };
    unsigned const rice = unsigned(gsz_rice_k_for_g(header.group_size));
    uint64_t const expected = gsz_expected_for_g(header.group_size);
    require(rice < 63, "invalid group index parameter");
    for (auto& size : sizes)
    {
        config_.check_running();
        if (inspection_limits_) inspection_limits_->check();
        bool const negative = bit();
        uint64_t remainder = 0;
        for (unsigned i = 0; i < rice; ++i) remainder |= uint64_t(bit()) << i;
        uint64_t quotient = 0;
        while (bit()) { ++quotient; require(quotient + rice + 2 <= 64, "invalid group index unary"); }
        uint64_t const magnitude = (quotient << rice) | remainder;
        require(!negative || magnitude < expected, "invalid group index delta");
        size = negative ? expected - magnitude : grow_add(expected, magnitude);
        check_gsz_size(size, header.group_size);
        payload = grow_add(payload, size);
        require(payload <= header.chunk_index_offset, "group index exceeds payload");
    }
    require(payload == header.chunk_index_offset, "group index does not cover payload");
    BitWriter canonical;
    gsz_encode(sizes, header.group_size, canonical);
    auto const index = canonical.asBytes();
    require(index.size() <= packed.size() && std::equal(index.begin(), index.end(), packed.begin()),
            "group index is not canonical");
}

inline void Engine::discover(bool allow_resume, std::optional<Identity> const& initial,
                             std::array<uint8_t, 32>* authority)
{
    uint64_t const bytes = file_bytes();
    static_assert(sizeof(Record) <= record_bytes - 32);
    auto valid_header = [&](Header const& header)
    {
        require(header.magic == PlotGroupFile::MAGIC && header.k >= 18 && header.k <= 28 &&
            !(header.k & 1) && header.strength >= 2 && header.strength <= 63,
            "invalid ordinary upstream group identity");
        if (inspection_limits_) inspection_limits_->check(inspection_memory_bytes(header.k));
        require(memory_bytes(header.k, config_, 0) <= config_.memory_limit_bytes,
                "group index/windows/codec exceed the memory budget");
    };
    if (bytes >= grow_add(record_alignment_, record_bytes))
    {
        for (unsigned slot = 0; slot < 2; ++slot)
        {
            std::array<uint8_t, record_bytes> packed{};
            read_at(record_offset(bytes, slot), packed);
            auto const checksum = grow_digest(std::span(packed).first(record_bytes - 32));
            if (!std::equal(checksum.begin(), checksum.end(), packed.end() - 32)) continue;
            Record candidate{};
            std::memcpy(&candidate, packed.data(), sizeof(candidate));
            if (std::string_view(candidate.magic.data(), candidate.magic.size()) !=
                std::string_view("XCHPLOT2GROWv1\0\0\0", 16)) continue;
            require(candidate.format == 1 && candidate.storage_bytes == bytes &&
                    candidate.record_alignment == record_alignment_,
                    "invalid growth checkpoint format or length");
            std::array<char, 64> expected_build{};
            auto const build = std::string_view(GROW_IDENTITY);
            std::copy_n(build.begin(), std::min(build.size(), expected_build.size()), expected_build.begin());
            require(candidate.build == expected_build,
                "growth checkpoint requires the same codec/build identity");
            if (!record_ || candidate.sequence > record_->sequence)
            {
                record_ = std::make_unique<Record>(candidate);
                if (authority) *authority = checksum;
            }
        }
    }
    if (record_)
    {
        require(allow_resume, "in-progress group is not farmable; use --resume");
        auto const& saved = *record_;
        valid_header(saved.original);
        valid_header(saved.target);
        uint64_t const start = sizeof(Header) + saved.original.memo_length;
        uint64_t const chunks = PlotGroupFile::getChunkCountForK(saved.target.k);
        uint16_t const pending_members = saved.remaining ?
            uint16_t(saved.target.group_size - saved.original.group_size) : 0;
        // Discovery has no incoming payload yet. Charge every pending task
        // descriptor and recovery buffer; exact storage admission follows in
        // prepare_members before allocating boundaries or replaying mutations.
        require(memory_bytes(saved.target.k, config_, pending_members, 0, pending_members) <=
                config_.memory_limit_bytes, "saved batch metadata exceeds memory budget");
        require(saved.original.version == PlotGroupFile::FORMAT_VERSION &&
            saved.target.version == PlotGroupFile::FORMAT_VERSION &&
            saved.target.group_size > saved.original.group_size &&
            saved.target.memo_length == saved.original.memo_length &&
            saved.original.group_id == saved.target.group_id && saved.original.k == saved.target.k &&
            saved.original.strength == saved.target.strength && saved.original.meta_group == saved.target.meta_group &&
            saved.pending <= 1 && saved.fresh <= 1 &&
            (!saved.fresh || (!saved.original.group_size && !saved.pending && !saved.redo_bytes && !saved.redo_chunks)) &&
            saved.final_bytes <= bytes && saved.redo_offset <= record_offset(bytes, 0) &&
            saved.redo_offset == saved.target.chunk_index_offset &&
            saved.redo_capacity == record_offset(bytes, 0) - saved.redo_offset &&
            saved.redo_bytes <= saved.redo_capacity &&
            saved.remaining <= chunks && saved.redo_chunks <= chunks - saved.remaining &&
            saved.original.chunk_index_offset >= start && saved.target.chunk_index_offset >= start &&
            saved.final_bytes > saved.target.chunk_index_offset &&
            (saved.original.group_size ? saved.original_bytes > saved.original.chunk_index_offset :
                saved.original_bytes == 0 && saved.original.chunk_index_offset == start) &&
            saved.original_bytes <= bytes && saved.old_frontier >= start && saved.new_frontier >= start &&
            saved.old_frontier <= saved.original.chunk_index_offset &&
            saved.new_frontier <= saved.target.chunk_index_offset && saved.new_frontier >= saved.old_frontier &&
            (!saved.pending || (saved.redo_chunks && saved.redo_bytes &&
                saved.redo_bytes <= saved.target.chunk_index_offset - saved.new_frontier)),
            "inconsistent growth checkpoint");
        identity_ = {saved.original,
            std::vector<uint8_t>(saved.memo.begin(), saved.memo.begin() + saved.original.memo_length),
            saved.original_bytes, true, saved.remaining ? saved.original.group_size : saved.target.group_size,
            saved.target.group_size, bytes};
        return;
    }
    if (bytes == 0)
    {
        require(initial.has_value(), "empty target needs its group identity");
        identity_ = *initial;
        valid_header(identity_.header);
        require(identity_.header.group_size == 0 && identity_.memo.size() <= 255 &&
            identity_.header.memo_length == identity_.memo.size(), "invalid new group identity");
        identity_.header.version = PlotGroupFile::FORMAT_VERSION;
        identity_.header.chunk_index_offset = sizeof(Header) + identity_.memo.size();
        identity_.bytes = 0;
        identity_.old_members = identity_.target_members = 0;
        old_sizes_.assign(size_t(PlotGroupFile::getChunkCountForK(identity_.header.k)), 0);
        return;
    }
    require(bytes >= sizeof(Header), "truncated group header");
    Header header{};
    read_at(0, {reinterpret_cast<uint8_t*>(&header), sizeof(header)});
    valid_header(header);
    require(header.version == PlotGroupFile::FORMAT_VERSION || header.version == finalizing_version ||
            (allow_resume && header.version == construction_version && !header.group_size),
            "unrecognized or damaged construction header; valid checkpoint required");
    require((header.group_size || header.version == construction_version) && bytes >= sizeof(Header) + header.memo_length,
            "invalid group membership or memo");
    std::vector<uint8_t> memo(header.memo_length);
    read_at(sizeof(Header), memo);
    if (header.version == construction_version && !header.group_size)
    {
        require(header.chunk_index_offset == sizeof(Header) + memo.size(), "invalid initial construction marker");
        header.version = PlotGroupFile::FORMAT_VERSION;
        identity_ = {header, std::move(memo), 0, true, 0, 0, bytes};
        initialization_orphan_ = true;
        old_sizes_.assign(size_t(PlotGroupFile::getChunkCountForK(header.k)), 0);
        return;
    }
    if (header.version == finalizing_version)
    {
        require(allow_resume, "payload-complete construction is not farmable; use --resume");
        header.version = PlotGroupFile::FORMAT_VERSION;
        identity_ = {header, std::move(memo), bytes, true, header.group_size, header.group_size, bytes};
        finalizing_ = true;
        return;
    }
    load_index(header, bytes, old_sizes_);
    BitWriter index;
    gsz_encode(old_sizes_, header.group_size, index);
    uint64_t const canonical_bytes = grow_add(header.chunk_index_offset, index.asBytes().size());
    require(canonical_bytes == bytes || allow_resume,
            "group has trailing bytes; --resume may discard an interrupted initialization tail");
    if (canonical_bytes != bytes)
    {
        // No target byte was overwritten before a valid checkpoint existed.
        // Verify the full original index above before removing an orphan tail.
        initialization_orphan_ = true;
    }
    identity_ = {header, std::move(memo), canonical_bytes, initialization_orphan_,
        header.group_size, header.group_size, initialization_orphan_ ? bytes : 0};
}

inline void Engine::confirm_identity()
{
    check_path();
    require(identity_.bytes <= config_.capacity_bytes &&
        (!identity_.pending || identity_.pending_peak_bytes <= config_.capacity_bytes),
        "existing construction exceeds supplied capacity");
    confirmed_ = true;
}

inline void Engine::prepare_members(Members members, uint16_t first)
{
    CodecScheduler::Lease compute(config_.cpu_scheduler.get(), 1, [&] { config_.check_running(); });
    require(!members.empty() && members.size() <= size_t(UINT16_MAX - first), "invalid incoming membership");
    uint8_t const k = identity_.header.k;
    uint64_t const end = 1ull << (k * 2);
    uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(k));
    bool immutable = true;
    for (size_t i = 0; i < members.size(); ++i) immutable = immutable && members[i].compact();
    if (!immutable || digest_count_ != members.size())
    {
        digest_ids_.reset();
        digest_count_ = 0;
    }
    // Release obsolete allocations before narrowing admission. Compact members
    // lend their own immutable boundary arrays; raw members retain the existing
    // sorted sweep and count-dependent wide-index fallback.
    if (member_boundaries_.size() != members.size())
        std::vector<FragmentBoundaries>().swap(member_boundaries_);
    for (size_t i = 0; i < member_boundaries_.size(); ++i)
    {
        auto const view = members[i];
        if (view.compact() || member_boundaries_[i].storage_bytes() >
                FragmentBoundaries::allocation_bytes(chunks, view.size()))
            member_boundaries_[i].clear();
    }
    require(memory_bytes(members) <= config_.memory_limit_bytes,
            "batch descriptors exceed the memory budget");
    if (member_boundaries_.empty()) member_boundaries_.resize(members.size());
    for (size_t member = 0; member < members.size(); ++member)
    {
        config_.check_running();
        auto const view = members[member];
        if (view.compact())
        {
            require(view.k() == k && !view.empty(), "invalid packed incoming member");
            member_boundaries_[member] = {};
            continue;
        }
        auto const fragments = view.raw();
        require(!fragments.empty() && fragments.back() < end,
                "empty, unsorted or out-of-range incoming member");
        auto& boundaries = member_boundaries_[member];
        boundaries.reset(chunks, fragments.size());
        boundaries.set(0, 0);
        size_t chunk = 0;
        uint64_t chunk_end = range;
        for (size_t i = 0; i < fragments.size(); ++i)
        {
            if (i % digest_words == 0) config_.check_running();
            require(fragments[i] < end && (!i || fragments[i] >= fragments[i - 1]),
                    "empty, unsorted or out-of-range incoming member");
            while (fragments[i] >= chunk_end)
            {
                boundaries.set(++chunk, i);
                chunk_end += range;
            }
        }
        for (size_t i = chunk + 1; i <= chunks; ++i) boundaries.set(i, fragments.size());
    }
}
inline std::array<uint8_t, 32> Engine::digest_members(Members members, MemberBatchHash const* hash) const
{
    CodecScheduler::Lease compute(config_.cpu_scheduler.get(), 1, [&] { config_.check_running(); });
    bool same = digest_count_ != 0 && digest_count_ == members.size();
    for (size_t i = 0; same && i < members.size(); ++i)
        same = members[i].compact() && members[i].immutable_id() == digest_ids_[i];
    if (same) return digest_;
    auto remember = [&](std::array<uint8_t, 32> const& digest)
    {
        bool immutable = !members.empty();
        for (size_t i = 0; i < members.size(); ++i)
            immutable = immutable && members[i].compact() && members[i].immutable_id();
        if (immutable)
        {
            if (digest_count_ != members.size())
            {
                digest_ids_.reset();
                digest_count_ = 0;
                digest_ids_.reset(new uint64_t[members.size()]);
            }
            for (size_t i = 0; i < members.size(); ++i) digest_ids_[i] = members[i].immutable_id();
            digest_ = digest;
            digest_count_ = members.size();
        }
        else
        {
            digest_ids_.reset();
            digest_count_ = 0;
        }
        return digest;
    };
    if (hash)
        if (auto digest = hash->digest_for(members)) return remember(*digest);
    // Keep the native uint64 size prefix and logical fragment bytes exactly.
    // Raw inputs hash in place; packed inputs reuse one bounded decode buffer.
    // The accelerated Rust shim owns the SHA state and consumes each segment
    // before asking for the next one.
    struct Reader
    {
        Members members;
        uint8_t k;
        size_t member = 0, chunk = 0, offset = 0;
        bool prefix = true;
        uint64_t count = 0;
        std::array<uint64_t, digest_words> decoded;
        std::exception_ptr error;
        Config const* config = nullptr;
    } reader{members, identity_.header.k};
    reader.config = &config_;
    auto next = [](void* context, pos2_sha256_segment* segment) noexcept -> int
    {
        auto& reader = *static_cast<Reader*>(context);
        try
        {
            reader.config->check_running();
            if (reader.member == reader.members.size()) return 0;
            auto const view = reader.members[reader.member];
            if (reader.prefix)
            {
                reader.count = view.size();
                *segment = {reinterpret_cast<uint8_t const*>(&reader.count), sizeof(reader.count)};
                reader.prefix = false;
                return 1;
            }
            if (!view.compact())
            {
                auto const raw = view.raw();
                require(raw.size() <= SIZE_MAX / sizeof(uint64_t), "member hash length overflow");
                auto const part = raw.subspan(reader.offset, std::min(raw.size() - reader.offset, reader.decoded.size()));
                *segment = {reinterpret_cast<uint8_t const*>(part.data()), part.size_bytes()};
                reader.offset += part.size();
                if (reader.offset == raw.size())
                { ++reader.member; reader.offset = 0; reader.prefix = true; }
                return 1;
            }
            size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(reader.k));
            size_t filled = 0;
            while (reader.chunk < chunks && filled < reader.decoded.size())
            {
                auto const region = view.region(reader.chunk);
                size_t const count = std::min(region.size() - reader.offset, reader.decoded.size() - filled);
                for (size_t i = 0; i < count; ++i) reader.decoded[filled + i] = region[reader.offset + i];
                reader.offset += count;
                filled += count;
                if (reader.offset == region.size()) { ++reader.chunk; reader.offset = 0; }
            }
            *segment = {reinterpret_cast<uint8_t const*>(reader.decoded.data()), filled * sizeof(uint64_t)};
            if (reader.chunk == chunks)
            {
                ++reader.member;
                reader.chunk = 0;
                reader.prefix = true;
            }
            return 1;
        }
        catch (...) { reader.error = std::current_exception(); return -1; }
    };
    std::array<uint8_t, 32> result{};
    int const status = pos2_sha256_stream(next, &reader, result.data());
    if (reader.error) std::rethrow_exception(reader.error);
    require(status == POS2_OK, "cannot hash incoming group members");
    return remember(result);
}

inline std::span<EncodedBlock const> Engine::encode_window(uint64_t first, uint64_t count, uint64_t source_begin,
    std::span<uint64_t const> source_sizes, Members members, std::vector<uint8_t>& source,
    std::vector<ChunkInput>& tasks, std::vector<FragmentRegion>& regions,
    std::span<uint64_t> sizes, bool size_only)
{
    uint64_t total = 0;
    for (auto size : source_sizes) total = grow_add(total, size);
    require(count <= max_tasks && count == sizes.size() && total <= config_.window_bytes,
            "group source window exceeds budget");
    source.resize(size_t(total));
    if (total) read_at(source_begin, source);
    tasks.clear();
    regions.clear();
    size_t const task_limit = size_t(std::min<uint64_t>(old_sizes_.size(), max_tasks));
    tasks.reserve(task_limit);
    regions.reserve(task_limit * members.size());
    uint64_t source_offset = 0;
    for (uint64_t i = 0; i < count; ++i)
    {
        uint64_t const chunk = first + i;
        auto const region_start = regions.size();
        for (size_t member = 0; member < members.size(); ++member)
        {
            auto const view = members[member];
            auto const& boundaries = member_boundaries_[member];
            regions.push_back(view.compact() ? view.region(chunk) : view.region(chunk,
                boundaries[size_t(chunk)], boundaries[size_t(chunk + 1)] - boundaries[size_t(chunk)]));
        }
        auto const size = source_sizes.empty() ? 0 : source_sizes[size_t(i)];
        tasks.push_back({chunk, std::span<uint8_t const>(source).subspan(size_t(source_offset), size_t(size)),
            identity_.old_members, std::span<FragmentRegion const>(regions).subspan(region_start, members.size())});
        source_offset += size;
    }
    if (!encoder_) encoder_ = std::make_unique<ParallelEncoder>(identity_.header.k, active_workers_, config_.codec_bytes, config_.cpu_scheduler);
    auto const begin = std::chrono::steady_clock::now();
    if (size_only)
    {
        encoder_->measure(tasks, sizes);
        stats_.codec_ns += elapsed_ns(begin);
        stats_.sized_chunks += count;
        return {};
    }
    auto const result = encoder_->encode(tasks, sizes, overlap_);
    stats_.codec_ns += elapsed_ns(begin);
    stats_.encode_chunks += count;
    return result;
}

inline void Engine::reset_encoding()
{
    encoder_.reset();
    overlap_ = config_.io_overlap && config_.codec_bytes / 2 >= ChunkCodec::estimate_base();
    active_workers_ = unsigned(std::min<uint64_t>(config_.workers,
        window_codec_bytes() / ChunkCodec::estimate_base()));
    require(active_workers_, "codec budget cannot hold one worker");
}

inline void Engine::prepare_encoding(uint64_t edge, bool reverse, Members members, uint64_t stop)
{
    // Admission may reduce workers or disable overlap. Discover that decision
    // before borrowing any encoder output, then keep it fixed during I/O.
    reset_encoding();
    if (reverse)
        while (edge) edge -= window_count(edge, true, members);
    else
        while (edge < std::min<uint64_t>(old_sizes_.size(), stop))
            edge += window_count(edge, false, members, stop);
}

inline void Engine::admit_chunk(uint64_t work)
{
    uint64_t budget = window_codec_bytes();
    if (overlap_ && work > budget && work <= config_.codec_bytes)
    {
        // Preserve admission for a chunk that fits the aggregate budget but
        // cannot coexist with a borrowed previous output.
        overlap_ = false;
        budget = config_.codec_bytes;
        encoder_.reset();
    }
    if (work > budget / active_workers_)
    {
        require(work <= budget, "one group chunk exceeds the codec window budget");
        active_workers_ = unsigned(budget / work);
        encoder_.reset();
    }
}

inline uint64_t Engine::window_count(uint64_t edge, bool reverse, Members members, uint64_t stop)
{
    config_.check_running();
    uint64_t const chunks = old_sizes_.size();
    uint64_t const limit = reverse ? edge : std::min(chunks, stop) - edge;
    uint64_t const baseline = ChunkCodec::estimate_base();
    std::vector<uint64_t> costs;
    costs.reserve(size_t(std::min<uint64_t>(chunks, max_tasks)));
    uint64_t source_bytes = 0, variable = 0;
    while (costs.size() < limit && costs.size() < max_tasks)
    {
        uint64_t const chunk = reverse ? edge - costs.size() - 1 : edge + costs.size();
        uint64_t incoming = 0;
        for (size_t member = 0; member < members.size(); ++member)
        {
            auto const view = members[member];
            auto const& boundaries = member_boundaries_[member];
            incoming = grow_add(incoming, view.compact()
                ? view.boundary(size_t(chunk + 1)) - view.boundary(size_t(chunk))
                : boundaries[size_t(chunk + 1)] - boundaries[size_t(chunk)]);
        }
        uint64_t const work = ChunkCodec::estimate_work(identity_.header.k, old_sizes_[size_t(chunk)], incoming);
        admit_chunk(work);
        if (costs.empty() && old_sizes_[size_t(chunk)] > config_.window_bytes) throw WindowBudgetError();
        uint64_t const available = window_codec_bytes() - uint64_t(active_workers_) * baseline;
        if (old_sizes_[size_t(chunk)] > config_.window_bytes - source_bytes ||
            work - baseline > available - variable) break;
        source_bytes += old_sizes_[size_t(chunk)];
        variable += work - baseline;
        costs.push_back(work - baseline);
    }
    require(!costs.empty(), "codec/window budget cannot admit a group chunk");
    if (reverse) std::reverse(costs.begin(), costs.end());
    size_t count = costs.size();
    while (true)
    {
        size_t const workers = std::min<size_t>(active_workers_, count);
        size_t const per_worker = count / workers + (count % workers != 0);
        bool fits = true;
        for (size_t begin = 0; begin < count; begin += per_worker)
        {
            uint64_t estimate = baseline;
            for (size_t i = begin; i < std::min(count, begin + per_worker); ++i)
                estimate += costs[(reverse ? costs.size() - count : 0) + i];
            fits = fits && estimate <= window_codec_bytes() / active_workers_;
        }
        if (fits) return count;
        count = std::max<size_t>(1, count / 2);
    }
}

inline Plan Engine::plan(Members members, MemberBatchHash const* hash)
{
    config_.check_running();
    PhaseTimer timing(stats_, WorkPhase::plan);
    require(confirmed_, "caller must confirm requested identity and budgets before planning");
    require(!identity_.pending, "pending construction requires resume with its saved batch");
    prepare_members(members, identity_.old_members);
    check_path();
    require(file_bytes() == identity_.bytes, "growth target size changed");
    reset_encoding();
    Plan result;
    result.new_members = uint16_t(identity_.old_members + members.size());
    result.header = identity_.header;
    result.header.group_size = result.new_members;
    result.members_digest = digest_members(members, hash);
    uint64_t const chunks = old_sizes_.size();
    result.sizes.resize(size_t(chunks));
    std::vector<uint8_t> source;
    source.reserve(size_t(config_.window_bytes));
    std::vector<ChunkInput> tasks;
    std::vector<FragmentRegion> regions;
    uint64_t old_offset = sizeof(Header) + identity_.memo.size(), new_offset = old_offset;
    for (uint64_t first = 0; first < chunks;)
    {
        config_.check_running();
        uint64_t const count = window_count(first, false, members);
        encode_window(first, count, old_offset,
            std::span<uint64_t const>(old_sizes_).subspan(size_t(first), size_t(count)), members,
            source, tasks, regions, std::span(result.sizes).subspan(size_t(first), size_t(count)), true);
        for (uint64_t i = first; i < first + count; ++i)
        {
            auto const size = result.sizes[size_t(i)];
            if (size > config_.window_bytes) throw WindowBudgetError();
            result.largest_chunk_bytes = std::max(result.largest_chunk_bytes, size);
            old_offset = grow_add(old_offset, old_sizes_[size_t(i)]);
            new_offset = grow_add(new_offset, size);
            require(new_offset >= old_offset, "unsafe backward rewrite: a new prefix is shorter");
        }
        first += count;
    }
    require(old_offset == identity_.header.chunk_index_offset, "old payload bounds changed");
    result.header.chunk_index_offset = new_offset;
    BitWriter index;
    gsz_encode(result.sizes, result.new_members, index);
    result.final_bytes = grow_add(new_offset, index.asBytes().size());
    uint64_t const needed = std::max(grow_add(new_offset, result.largest_chunk_bytes), identity_.bytes);
    uint64_t const first_record = grow_add(needed,
        needed % record_alignment_ ? record_alignment_ - needed % record_alignment_ : 0);
    uint64_t const journal_end = grow_add(grow_add(first_record, record_alignment_), record_bytes);
    result.peak_file_bytes = std::max(result.final_bytes, journal_end);
    result.minimum_peak_file_bytes = result.peak_file_bytes;
    result.workspace_bytes = result.peak_file_bytes - result.final_bytes;
    result.accepted = result.peak_file_bytes <= config_.capacity_bytes;
    if (!result.accepted) result.reason = result.final_bytes > config_.capacity_bytes ?
        "final output exceeds capacity" : "same-file recovery workspace exceeds capacity";
    // The exact minimum above decides admission. When spare space permits,
    // bound the redo workspace by one I/O window so overlapping transactions
    // can commit many chunks with the same durability barriers.
    uint64_t const journal_padding = grow_add(grow_add(record_alignment_, record_alignment_), record_bytes);
    if (result.accepted && journal_padding <= uint64_t(INT64_MAX) - new_offset &&
        config_.window_bytes <= uint64_t(INT64_MAX) - new_offset - journal_padding)
    {
        uint64_t const wanted = std::max(grow_add(new_offset, config_.window_bytes), identity_.bytes);
        uint64_t const aligned = grow_add(wanted, wanted % record_alignment_ ?
            record_alignment_ - wanted % record_alignment_ : 0);
        uint64_t const peak = std::max(result.final_bytes,
            grow_add(grow_add(aligned, record_alignment_), record_bytes));
        if (peak <= config_.capacity_bytes && peak > result.peak_file_bytes)
        {
            file_io::Space space{};
            require(file_io::space_fd(fd_, &space) == 0 && space.f_frsize,
                    "cannot inspect optional redo allocation headroom");
            uint64_t needed = peak - identity_.bytes;
            if (cow_filesystem_)
            {
                if (config_.window_bytes > (uint64_t(INT64_MAX) - needed) / 2) return result;
                needed += config_.window_bytes * 2;
            }
            uint64_t const blocks = needed / space.f_frsize + (needed % space.f_frsize != 0);
            if (blocks <= space.f_bavail)
            {
                result.peak_file_bytes = peak;
                result.workspace_bytes = peak - result.final_bytes;
            }
        }
    }
    return result;
}

inline std::optional<Plan> Engine::plan_fresh(Members members, MemberBatchHash const* hash)
{
    config_.check_running();
    require(confirmed_, "caller must confirm requested identity and budgets before planning");
    if (identity_.pending || identity_.old_members) return std::nullopt;
    PhaseTimer timing(stats_, WorkPhase::plan);
    prepare_members(members, 0);
    check_path();
    require(file_bytes() == 0 && identity_.bytes == 0, "fresh growth target size changed");
    reset_encoding();
    Plan result;
    result.fresh_single_pass = true;
    result.new_members = uint16_t(members.size());
    result.header = identity_.header;
    result.header.group_size = result.new_members;
    result.members_digest = digest_members(members, hash);
    uint64_t const chunks = old_sizes_.size();
    result.sizes.resize(size_t(chunks));
    std::vector<FragmentRegion> regions(members.size());
    uint64_t offset = sizeof(Header) + identity_.memo.size();
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        config_.check_running();
        uint64_t incoming = 0;
        for (size_t member = 0; member < members.size(); ++member)
        {
            auto const view = members[member];
            auto const& boundaries = member_boundaries_[member];
            regions[member] = view.compact() ? view.region(chunk) : view.region(chunk,
                boundaries[size_t(chunk)], boundaries[size_t(chunk + 1)] - boundaries[size_t(chunk)]);
            incoming = grow_add(incoming, regions[member].size());
        }
        admit_chunk(ChunkCodec::estimate_work(identity_.header.k, 0, incoming));
        uint64_t const bound = ChunkCodec::fresh_bound(identity_.header.k, chunk, regions, config_.cpu_scheduler.get());
        if (bound > config_.window_bytes) return std::nullopt;
        result.sizes[size_t(chunk)] = bound;
        result.largest_chunk_bytes = std::max(result.largest_chunk_bytes, bound);
        offset = grow_add(offset, bound);
    }
    result.header.chunk_index_offset = offset;
    // Every canonical size code is at most a machine word (check_gsz_size).
    result.final_bytes = grow_add(offset, chunks * sizeof(uint64_t));
    uint64_t const aligned = grow_add(result.final_bytes, result.final_bytes % record_alignment_ ?
        record_alignment_ - result.final_bytes % record_alignment_ : 0);
    result.peak_file_bytes = grow_add(grow_add(aligned, record_alignment_), record_bytes);
    result.minimum_peak_file_bytes = result.peak_file_bytes;
    result.workspace_bytes = result.peak_file_bytes - result.final_bytes;
    if (result.peak_file_bytes > config_.capacity_bytes) return std::nullopt;
    file_io::Space space{};
    require(file_io::space_fd(fd_, &space) == 0 && space.f_frsize, "cannot inspect fresh allocation headroom");
    uint64_t const needed = grow_add(result.peak_file_bytes,
        cow_filesystem_ ? grow_add(config_.checkpoint_bytes, grow_add(record_alignment_, record_alignment_)) : 0);
    uint64_t const blocks = needed / space.f_frsize + (needed % space.f_frsize != 0);
    if (blocks > space.f_bavail) return std::nullopt;
    result.accepted = true;
    return result;
}

inline uint64_t Engine::preallocate(uint64_t end, uint64_t minimum)
{
    check_path();
    uint64_t const original = file_bytes();
    require(minimum >= original && end >= minimum && end <= config_.capacity_bytes, "construction exceeds capacity");
#ifdef __linux__
    // Shared old extents can need an entire extra copy during overwrite. Refuse
    // them before touching file bytes; callers can explicitly unshare first.
    if (original)
    {
        alignas(struct fiemap) std::array<uint8_t, sizeof(struct fiemap) + sizeof(struct fiemap_extent) * 64> storage{};
        uint64_t position = 0;
        while (position < original)
        {
            storage.fill(0);
            auto* map = reinterpret_cast<struct fiemap*>(storage.data());
            map->fm_start = position;
            map->fm_length = original - position;
            map->fm_flags = FIEMAP_FLAG_SYNC;
            map->fm_extent_count = 64;
            int const status = ::ioctl(fd_, FS_IOC_FIEMAP, map);
            if (status != 0)
            {
                require(!cow_filesystem_ && (errno == EOPNOTSUPP || errno == ENOTTY || errno == ENOSYS),
                        "cannot inspect existing file sharing before growth");
                break;
            }
            if (!map->fm_mapped_extents) break;
            auto const* extents = reinterpret_cast<struct fiemap_extent*>(storage.data() + sizeof(struct fiemap));
            for (uint32_t i = 0; i < map->fm_mapped_extents; ++i)
                require(!(extents[i].fe_flags & FIEMAP_EXTENT_SHARED),
                        "shared/reflinked target extents require explicit unsharing before growth");
            auto const& last = extents[map->fm_mapped_extents - 1];
            uint64_t const next = grow_add(last.fe_logical, last.fe_length);
            require(next > position, "invalid filesystem extent map");
            position = next;
            if (last.fe_flags & FIEMAP_EXTENT_LAST) break;
        }
    }
#endif
#if defined(__linux__) || defined(_WIN32)
    if (end)
    {
        int status;
        do { status = file_io::allocate_file(fd_, file_io::keep_size, 0, file_io::Offset(end)); }
        while (status && errno == EINTR);
        if (status && errno == ENOSPC && minimum < end)
        {
            // An opportunistic larger journal must not reject an exact fit.
            // Discard only new allocations beyond the unchanged logical EOF,
            // then physically reserve the already-admitted minimum instead.
            do { status = file_io::allocate_file(fd_, file_io::punch_hole | file_io::keep_size,
                                      file_io::Offset(original), file_io::Offset(end - original)); }
            while (status && errno == EINTR);
            require(status == 0, "cannot release optional redo allocation; original bytes are intact");
            end = minimum;
            do { status = file_io::allocate_file(fd_, file_io::keep_size, 0, file_io::Offset(end)); }
            while (status && errno == EINTR);
        }
        require(status == 0, "physical allocation preflight failed without changing original bytes: " +
            std::string(std::strerror(errno)) + "; a filesystem supporting fallocate is required");
        sync();
    }
#else
    (void)end;
    throw std::runtime_error("same-file growth requires supported native physical allocation");
#endif
    grow_failpoint("allocated");
    return end;
}

inline uint64_t Engine::record_offset(uint64_t storage_bytes, unsigned slot) const
{
    require(slot < 2 && storage_bytes >= grow_add(record_alignment_, record_bytes),
            "invalid recovery record placement");
    uint64_t const last = (storage_bytes - record_bytes) / record_alignment_ * record_alignment_;
    require(last >= record_alignment_, "recovery records overlap file header");
    return slot ? last : last - record_alignment_;
}

inline void Engine::checkpoint(Record& record)
{
    require(record.sequence < UINT64_MAX, "growth checkpoint sequence overflow");
    check_path();
    ++record.sequence;
    std::array<uint8_t, record_bytes> packed{};
    std::memcpy(packed.data(), &record, sizeof(record));
    auto const checksum = grow_digest(std::span(packed).first(record_bytes - 32));
    std::copy(checksum.begin(), checksum.end(), packed.end() - 32);
    uint64_t const offset = record_offset(record.storage_bytes, unsigned(record.sequence % 2));
    // Exercise a torn checkpoint while leaving the other generation intact.
    write_at(offset, std::span<uint8_t const>(packed).first(record_bytes / 2));
    grow_failpoint("checkpoint-partial");
    write_at(offset + record_bytes / 2, std::span<uint8_t const>(packed).subspan(record_bytes / 2));
    sync();
    ++stats_.checkpoints;
}

inline Result Engine::apply(Plan& plan, Members members)
{
    require(confirmed_, "caller must confirm requested identity and budgets before mutation");
    require(!identity_.pending && plan.accepted && plan.peak_file_bytes <= config_.capacity_bytes,
            "growth plan is not accepted");
    prepare_members(members, identity_.old_members);
    require(plan.new_members == identity_.old_members + members.size() &&
        plan.members_digest == digest_members(members) && plan.sizes.size() == old_sizes_.size(),
        "incoming members changed after sizing");
    return apply_prepared(plan, members);
}

inline AppendResult Engine::append(Members members, MemberBatchHash const* hash, bool reject_window)
{
    std::optional<Plan> plan;
    try
    {
        plan = plan_fresh(members, hash);
        if (!plan) plan = this->plan(members, hash);
    }
    catch (WindowBudgetError const& error)
    {
        // Sizing has not created a durable transaction. Exceptions during
        // apply retain their ordinary pending-work recovery behavior.
        if (!reject_window) throw;
        return {false, false, 0, 0, error.what(), {}};
    }
    AppendResult result{plan->accepted, plan->fresh_single_pass,
        plan->final_bytes, plan->peak_file_bytes, plan->reason, {}};
    // No caller can change the input between these phases. Public plan/apply
    // retains its revalidation when the caller owns that interval.
    if (plan->accepted) result.result = apply_prepared(*plan, members);
    return result;
}

inline Result Engine::apply_prepared(Plan& plan, Members members)
{
    config_.check_running();
    PhaseTimer timing(stats_, WorkPhase::encode);
    auto saved = std::make_unique<Record>();
    std::memcpy(saved->magic.data(), "XCHPLOT2GROWv1\0\0\0", saved->magic.size());
    saved->original = identity_.header;
    saved->target = plan.header;
    saved->original_bytes = identity_.bytes;
    saved->final_bytes = plan.final_bytes;
    saved->storage_bytes = plan.peak_file_bytes;
    saved->record_alignment = record_alignment_;
    saved->redo_offset = plan.header.chunk_index_offset;
    saved->redo_capacity = record_offset(plan.peak_file_bytes, 0) - saved->redo_offset;
    saved->remaining = plan.sizes.size();
    saved->old_frontier = identity_.header.chunk_index_offset;
    saved->new_frontier = plan.header.chunk_index_offset;
    saved->largest_chunk = plan.largest_chunk_bytes;
    saved->fresh = plan.fresh_single_pass;
    if (saved->fresh)
    {
        require(identity_.old_members == 0, "single-pass creation requires an empty source");
        saved->new_frontier = saved->old_frontier;
    }
    std::copy(identity_.memo.begin(), identity_.memo.end(), saved->memo.begin());
    saved->members_digest = plan.members_digest;
    auto const build = std::string_view(GROW_IDENTITY);
    std::copy_n(build.begin(), std::min(build.size(), saved->build.size()), saved->build.begin());
    if (!identity_.old_members)
    {
        // Give interrupted initialization a durable identity before extending
        // the file or creating a journal. There is no old payload to preserve.
        Header marker = identity_.header;
        marker.version = construction_version;
        write_at(0, {reinterpret_cast<uint8_t const*>(&marker), sizeof(marker)});
        write_at(sizeof(Header), identity_.memo);
        sync();
        auto parent = file_io::native_path(path_).parent_path();
        if (parent.empty()) parent = ".";
        int const directory = file_io::open_file(parent.c_str(), file_io::read_only | file_io::close_on_exec | file_io::directory);
        require(directory >= 0, "cannot open construction parent directory");
        int const result = file_io::flush(directory);
        file_io::close_file(directory);
        require(result == 0, "cannot sync construction parent directory");
        grow_failpoint("bootstrap");
    }
    plan.peak_file_bytes = preallocate(plan.peak_file_bytes,
        plan.minimum_peak_file_bytes ? plan.minimum_peak_file_bytes : plan.peak_file_bytes);
    plan.workspace_bytes = plan.peak_file_bytes - plan.final_bytes;
    saved->storage_bytes = plan.peak_file_bytes;
    saved->redo_capacity = record_offset(plan.peak_file_bytes, 0) - saved->redo_offset;
    try
    {
        require(file_io::resize_file(fd_, file_io::Offset(plan.peak_file_bytes)) == 0, "cannot extend allocated construction file");
        checkpoint(*saved);
    }
    catch (...)
    {
        // A normal initialization failure has not changed a single old byte.
        // If killed here, explicit resume verifies the old index and trims tail.
        file_io::resize_file(fd_, file_io::Offset(identity_.bytes));
        sync();
        throw;
    }
    record_ = std::move(saved);
    identity_.pending = true;
    identity_.target_members = plan.new_members;
    identity_.pending_peak_bytes = plan.peak_file_bytes;
    grow_failpoint("initialized");
    if (identity_.old_members)
        write_at(offsetof(Header, version), std::span<uint8_t const>(&construction_version, 1));
    sync();
    grow_failpoint("marked");
    if (plan.fresh_single_pass) return create(plan, members, *record_);
    rewrite(plan, members, *record_);
    return finish(plan);
}

inline std::unique_ptr<Engine::Window> Engine::encode_owned(uint64_t first, uint64_t count,
                                                         uint64_t source_begin, Members members)
{
    auto window = std::make_unique<Window>();
    window->first = first;
    window->count = count;
    window->source_begin = source_begin;
    window->sizes.resize(size_t(count));
    // Reserve the complete configured window so vector growth cannot leave an
    // unaccounted replacement allocation live beside the prior source window.
    window->source.reserve(size_t(config_.window_bytes));
    auto const blocks = encode_window(first, count, source_begin,
        std::span<uint64_t const>(old_sizes_).subspan(size_t(first), size_t(count)), members,
        window->source, window->tasks, window->regions, window->sizes);
    window->blocks.assign(blocks.begin(), blocks.end());
    return window;
}

inline Result Engine::create(Plan& plan, Members members, Record& record)
{
    uint64_t const chunks = old_sizes_.size();
    uint64_t const payload_bound = record.target.chunk_index_offset;
    // Discover the minimum worker count before any output is borrowed. The
    // count never changes while a previous codec block is being written.
    prepare_encoding(chunks - record.remaining, false, members);
    std::future<std::unique_ptr<Window>> pending;
    auto worker = overlap_ ? std::make_unique<WindowWorker>(*this, members) : nullptr;
    auto next = record;
    uint64_t buffered = 0;
    while (next.remaining)
    {
        config_.check_running();
        uint64_t const first = chunks - next.remaining;
        auto window = pending.valid() ? pending.get() :
            encode_owned(first, window_count(first, false, members), 0, members);
        uint64_t const count = window->count;
        auto const sizes = std::span(plan.sizes).subspan(size_t(first), size_t(count));
        std::copy(window->sizes.begin(), window->sizes.end(), sizes.begin());
        if (overlap_ && first + count < chunks)
        {
            uint64_t const following = first + count;
            uint64_t const following_count = window_count(following, false, members);
            pending = worker->encode(following, following_count, 0);
            ++stats_.overlap_windows;
        }
        uint64_t offset = next.new_frontier;
        for (uint64_t size : sizes)
        {
            require(size <= config_.window_bytes && size <= payload_bound - offset,
                    "fresh encoding exceeded its admitted bound");
            offset += size;
        }
        uint64_t const length = offset - next.new_frontier, partial = length / 2;
        write_blocks(next.new_frontier, window->blocks, 0, partial);
        grow_failpoint("direct-target-partial");
        write_blocks(next.new_frontier + partial, window->blocks, partial, length - partial);
        buffered = grow_add(buffered, offset - next.new_frontier);
        next.new_frontier = offset;
        next.remaining -= count;
        if (buffered >= config_.checkpoint_bytes || !next.remaining)
        {
            sync();
            grow_failpoint("direct-target");
            checkpoint(next);
            release_clean(record.new_frontier, next.new_frontier);
            record = next;
            buffered = 0;
            grow_failpoint("direct-committed");
        }
    }
    plan.header.chunk_index_offset = record.new_frontier;
    {
        BitWriter index;
        gsz_encode(plan.sizes, plan.new_members, index);
        plan.final_bytes = grow_add(plan.header.chunk_index_offset, index.asBytes().size());
    }
    require(plan.final_bytes <= record.final_bytes, "fresh index exceeded its admitted bound");
    plan.workspace_bytes = plan.peak_file_bytes - plan.final_bytes;
    grow_failpoint("payload");
    return finish(plan);
}

inline void Engine::rewrite(Plan const& plan, Members members, Record& record)
{
    prepare_encoding(record.remaining, true, members);
    std::future<std::unique_ptr<Window>> pending;
    auto worker = overlap_ ? std::make_unique<WindowWorker>(*this, members) : nullptr;
    std::vector<uint64_t> old_prefix, new_prefix;
    size_t const task_limit = size_t(std::min<uint64_t>(old_sizes_.size(), max_tasks));
    old_prefix.reserve(task_limit + 1);
    new_prefix.reserve(task_limit + 1);
    auto durable = record;
    uint64_t buffered = 0;
    auto commit_direct = [&]
    {
        if (!buffered) return;
        sync();
        grow_failpoint("direct-target");
        checkpoint(record);
        release_clean(record.old_frontier, durable.old_frontier);
        release_clean(record.new_frontier, durable.new_frontier);
        durable = record;
        buffered = 0;
        grow_failpoint("direct-committed");
    };
    while (record.remaining)
    {
        config_.check_running();
        std::unique_ptr<Window> window;
        if (pending.valid()) window = pending.get();
        else
        {
            uint64_t const count = window_count(record.remaining, true, members);
            uint64_t const first = record.remaining - count;
            uint64_t bytes = 0;
            for (uint64_t i = first; i < record.remaining; ++i) bytes += old_sizes_[size_t(i)];
            window = encode_owned(first, count, record.old_frontier - bytes, members);
        }
        auto const count = window->count, first = window->first, source_begin = window->source_begin;
        auto const& actual = window->sizes;
        auto const& blocks = window->blocks;
        old_prefix.assign(size_t(count + 1), 0);
        new_prefix.assign(size_t(count + 1), 0);
        for (uint64_t i = 0; i < count; ++i)
        {
            require(actual[size_t(i)] == plan.sizes[size_t(first + i)], "nondeterministic group encoding");
            old_prefix[size_t(i + 1)] = grow_add(old_prefix[size_t(i)], old_sizes_[size_t(first + i)]);
            new_prefix[size_t(i + 1)] = grow_add(new_prefix[size_t(i)], actual[size_t(i)]);
        }
        uint64_t const destination_begin = record.new_frontier - new_prefix.back();
        require(destination_begin >= source_begin, "unsafe window prefix");
        if (overlap_ && first)
        {
            uint64_t const following_count = window_count(first, true, members);
            uint64_t const following = first - following_count;
            uint64_t bytes = 0;
            for (uint64_t i = following; i < first; ++i) bytes += old_sizes_[size_t(i)];
            uint64_t const following_source = source_begin - bytes;
            pending = worker->encode(following, following_count, following_source);
            ++stats_.overlap_windows;
        }
        auto transfer = [&](uint64_t destination, uint64_t begin, uint64_t length)
        {
            write_blocks(destination, blocks, begin, length);
        };
        auto digest = [&](uint64_t begin, uint64_t length)
        {
            struct sha256 context;
            std::array<uint8_t, 32> result{};
            sha256_init(&context);
            for (auto const& block : blocks)
            {
                uint64_t const block_begin = new_prefix[block.begin], block_end = new_prefix[block.end];
                uint64_t const lo = std::max(begin, block_begin), hi = std::min(begin + length, block_end);
                if (lo < hi) sha256_update(&context, block.bytes.data() + size_t(lo - block_begin), size_t(hi - lo));
            }
            sha256_sum(&context, result.data());
            return result;
        };
        auto check_cow_space = [&](uint64_t bytes, bool redo)
        {
            if (!cow_filesystem_) return;
            file_io::Space space{};
            require(file_io::space_fd(fd_, &space) == 0 && space.f_frsize,
                    "cannot inspect copy-on-write working space");
            uint64_t const needed = grow_add(redo ? grow_add(bytes, bytes) : bytes,
                                            grow_add(record_alignment_, record_alignment_));
            uint64_t const blocks_needed = needed / space.f_frsize + (needed % space.f_frsize != 0);
            require(space.f_bavail >= blocks_needed,
                "copy-on-write filesystem needs bounded overwrite headroom; free space and --resume this file");
        };
        if (destination_begin >= record.old_frontier && destination_begin < durable.old_frontier)
            commit_direct();
        if (destination_begin >= durable.old_frontier)
        {
            // The entire target window lies beyond every unread old byte. Its
            // old source remains available until the advanced checkpoint is
            // durable, so a crash can regenerate this window without redo.
            check_cow_space(new_prefix.back(), false);
            check_path();
            uint64_t const partial = new_prefix.back() / 2;
            transfer(destination_begin, 0, partial);
            grow_failpoint("direct-target-partial");
            transfer(destination_begin + partial, partial, new_prefix.back() - partial);
            auto next = record;
            next.pending = 0;
            next.redo_bytes = next.redo_chunks = 0;
            next.remaining = first;
            next.old_frontier = source_begin;
            next.new_frontier = destination_begin;
            record = next;
            buffered = grow_add(buffered, new_prefix.back());
            if (buffered >= config_.checkpoint_bytes || !record.remaining) commit_direct();
            continue;
        }
        commit_direct();
        for (uint64_t end = count; end;)
        {
            config_.check_running();
            uint64_t begin = end;
            while (begin && new_prefix[size_t(end)] - new_prefix[size_t(begin - 1)] <= record.redo_capacity) --begin;
            require(begin < end, "recovery workspace cannot hold one encoded chunk");
            uint64_t const bytes = new_prefix[size_t(end)] - new_prefix[size_t(begin)];
            check_cow_space(bytes, true);
            check_path();
            auto next = record;
            next.pending = 1;
            next.remaining = first + begin;
            next.old_frontier = source_begin + old_prefix[size_t(begin)];
            next.new_frontier = destination_begin + new_prefix[size_t(begin)];
            next.redo_bytes = bytes;
            next.redo_chunks = end - begin;
            next.redo_digest = digest(new_prefix[size_t(begin)], bytes);
            require(next.new_frontier >= next.old_frontier, "unsafe transaction prefix");
            transfer(record.redo_offset, new_prefix[size_t(begin)], bytes);
            grow_failpoint("redo-written");
            sync();
            grow_failpoint("redo");
            checkpoint(next);
            record = next;
            grow_failpoint("pending");
            uint64_t const partial = bytes / 2;
            transfer(record.new_frontier, new_prefix[size_t(begin)], partial);
            grow_failpoint("target-partial");
            transfer(record.new_frontier + partial, new_prefix[size_t(begin)] + partial, bytes - partial);
            sync();
            grow_failpoint("target");
            record.pending = 0;
            checkpoint(record);
            release_clean(record.old_frontier, durable.old_frontier);
            release_clean(record.new_frontier, grow_add(record.new_frontier, bytes));
            durable = record;
            grow_failpoint("committed");
            end = begin;
        }
    }
    commit_direct();
    require(record.old_frontier == sizeof(Header) + identity_.memo.size() &&
            record.new_frontier == record.old_frontier, "incomplete reverse rewrite");
    grow_failpoint("payload");
}

inline void Engine::replay(Record& record)
{
    config_.check_running();
    if (!record.pending) return;
    require(record.redo_bytes && record.new_frontier <= record.target.chunk_index_offset &&
        record.redo_bytes <= record.target.chunk_index_offset - record.new_frontier,
        "invalid pending redo bounds");
    std::vector<uint8_t> buffer(size_t(std::min(config_.window_bytes, record.redo_bytes)));
    struct sha256 context;
    std::array<uint8_t, 32> checksum{};
    sha256_init(&context);
    for (uint64_t position = 0; position < record.redo_bytes;)
    {
        config_.check_running();
        auto const bytes = std::min<uint64_t>(buffer.size(), record.redo_bytes - position);
        read_at(record.redo_offset + position, std::span(buffer).first(size_t(bytes)));
        sha256_update(&context, buffer.data(), size_t(bytes));
        position += bytes;
    }
    sha256_sum(&context, checksum.data());
    require(checksum == record.redo_digest, "pending redo checksum mismatch; preserve this file");
    for (uint64_t position = 0; position < record.redo_bytes;)
    {
        config_.check_running();
        auto const bytes = std::min<uint64_t>(buffer.size(), record.redo_bytes - position);
        read_at(record.redo_offset + position, std::span(buffer).first(size_t(bytes)));
        write_at(record.new_frontier + position, std::span<uint8_t const>(buffer).first(size_t(bytes)));
        position += bytes;
        grow_failpoint("replay-partial");
    }
    sync();
    grow_failpoint("replayed");
    record.pending = 0;
    checkpoint(record);
}

inline void Engine::scan_sizes(Header const& header, uint64_t first, uint64_t count,
    uint64_t begin, uint64_t end, std::vector<uint64_t>& sizes) const
{
    require(begin <= end && end <= file_bytes(), "invalid reconstruction bounds");
    ChunkCodec codec(header.k, config_.codec_bytes, config_.cpu_scheduler);
    std::vector<uint8_t> buffer(size_t(std::min(config_.window_bytes, end - begin)));
    uint64_t offset = begin, buffered_begin = begin, buffered_end = begin;
    for (uint64_t chunk = first; chunk < first + count; ++chunk)
    {
        config_.check_running();
        auto refill = [&]
        {
            buffered_begin = offset;
            uint64_t const bytes = std::min(config_.window_bytes, end - offset);
            require(bytes, "truncated chunk reconstruction");
            read_at(offset, std::span(buffer).first(size_t(bytes)));
            buffered_end = offset + bytes;
        };
        if (offset >= buffered_end) refill();
        size_t size = 0;
        try { size = codec.scan(chunk,
            std::span<uint8_t const>(buffer).subspan(size_t(offset - buffered_begin), size_t(buffered_end - offset)),
            header.group_size); }
        catch (...)
        {
            if (offset == buffered_begin || buffered_end == end) throw;
            refill();
            size = codec.scan(chunk, std::span<uint8_t const>(buffer).first(size_t(buffered_end - offset)),
                              header.group_size);
        }
        require(size && size <= end - offset, "reconstructed chunk exceeds payload");
        sizes[size_t(chunk)] = size;
        offset += size;
    }
    require(offset == end, "reconstructed chunks do not cover saved frontier");
}

inline Result Engine::resume(Members members, MemberBatchHash const* hash)
{
    config_.check_running();
    PhaseTimer timing(stats_, WorkPhase::encode);
    require(confirmed_, "caller must confirm requested identity and budgets before recovery");
    require(identity_.pending, "no saved construction requires resume");
    check_path();
    Plan plan;
    if (initialization_orphan_)
    {
        require(members.empty(), "orphan initialization recovery needs no incoming batch");
        uint64_t const peak = file_bytes();
        require(file_io::resize_file(fd_, file_io::Offset(identity_.bytes)) == 0, "cannot remove orphan initialization tail");
        sync();
        initialization_orphan_ = false;
        identity_.pending = false;
        identity_.pending_peak_bytes = 0;
        return {identity_.bytes, peak, peak - identity_.bytes, identity_.header.group_size};
    }
    if (finalizing_)
    {
        require(members.empty(), "payload-complete recovery needs no incoming batch");
        plan.header = identity_.header;
        plan.new_members = plan.header.group_size;
        plan.sizes.resize(size_t(PlotGroupFile::getChunkCountForK(plan.header.k)));
        encoder_.reset();
        scan_sizes(plan.header, 0, plan.sizes.size(), sizeof(Header) + identity_.memo.size(),
                   plan.header.chunk_index_offset, plan.sizes);
        BitWriter index;
        gsz_encode(plan.sizes, plan.new_members, index);
        plan.final_bytes = grow_add(plan.header.chunk_index_offset, index.asBytes().size());
        plan.peak_file_bytes = file_bytes();
        plan.workspace_bytes = plan.peak_file_bytes > plan.final_bytes ? plan.peak_file_bytes - plan.final_bytes : 0;
        require(plan.final_bytes <= file_bytes(), "payload-complete file is shorter than its canonical index");
        return finish(plan);
    }
    require(record_ && record_->storage_bytes <= config_.capacity_bytes,
            "saved construction exceeds the supplied capacity");
    auto& record = *record_;
    if (record.remaining || !members.empty())
    {
        prepare_members(members, record.original.group_size);
        require(record.target.group_size == record.original.group_size + members.size() &&
                digest_members(members, hash) == record.members_digest,
                "regenerated incoming batch does not match saved identity/digest");
    }
    encoder_.reset();
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(record.target.k);
    old_sizes_.assign(size_t(chunks), 0);
    plan.sizes.assign(size_t(chunks), 0);
    plan.header = record.target;
    plan.new_members = record.target.group_size;
    plan.final_bytes = record.final_bytes;
    plan.peak_file_bytes = record.storage_bytes;
    plan.workspace_bytes = record.storage_bytes - record.final_bytes;
    plan.members_digest = record.members_digest;
    plan.accepted = true;
    if (record.fresh)
    {
        plan.fresh_single_pass = true;
        uint64_t const completed = chunks - record.remaining;
        scan_sizes(record.target, 0, completed, sizeof(Header) + identity_.memo.size(),
                   record.new_frontier, plan.sizes);
        return create(plan, members, record);
    }
    if (record.remaining)
    {
        require(record.original.group_size || record.old_frontier == sizeof(Header) + identity_.memo.size(),
                "invalid empty-source frontier");
        if (record.original.group_size)
            scan_sizes(record.original, 0, record.remaining, sizeof(Header) + identity_.memo.size(),
                       record.old_frontier, old_sizes_);
    }
    if (record.pending)
        scan_sizes(record.target, record.remaining, record.redo_chunks, record.redo_offset,
                   record.redo_offset + record.redo_bytes, plan.sizes);
    uint64_t const suffix_first = record.remaining + (record.pending ? record.redo_chunks : 0);
    if (suffix_first < chunks)
        scan_sizes(record.target, suffix_first, chunks - suffix_first,
                   record.new_frontier + (record.pending ? record.redo_bytes : 0),
                   record.target.chunk_index_offset, plan.sizes);
    {
    std::vector<uint8_t> source;
    source.reserve(size_t(config_.window_bytes));
    std::vector<ChunkInput> tasks;
    std::vector<FragmentRegion> regions;
    uint64_t old_offset = sizeof(Header) + identity_.memo.size(), new_offset = old_offset;
    reset_encoding();
    for (uint64_t first = 0; first < record.remaining;)
    {
        uint64_t const count = window_count(first, false, members, record.remaining);
        encode_window(first, count, old_offset,
            std::span<uint64_t const>(old_sizes_).subspan(size_t(first), size_t(count)), members,
            source, tasks, regions, std::span(plan.sizes).subspan(size_t(first), size_t(count)));
        for (uint64_t i = first; i < first + count; ++i)
        {
            old_offset = grow_add(old_offset, old_sizes_[size_t(i)]);
            new_offset = grow_add(new_offset, plan.sizes[size_t(i)]);
            require(new_offset >= old_offset, "unsafe reconstructed prefix");
        }
        first += count;
    }
    require(old_offset == record.old_frontier && new_offset == record.new_frontier,
            "regenerated batch does not reproduce saved frontier");
    BitWriter index;
    gsz_encode(plan.sizes, plan.new_members, index);
    require(grow_add(plan.header.chunk_index_offset, index.asBytes().size()) == plan.final_bytes,
            "reconstructed index length differs from exact plan");
    }
    replay(record);
    rewrite(plan, members, record);
    return finish(plan);
}

inline Result Engine::finish(Plan const& plan)
{
    config_.check_running();
    check_path();
    Header marker = plan.header;
    marker.version = finalizing_version;
    auto const header_bytes = std::span<uint8_t const>(reinterpret_cast<uint8_t const*>(&marker), sizeof(marker));
    // Footer checkpoints remain authoritative until this complete marker has
    // reached storage. Afterwards the payload alone can reconstruct its index.
    write_at(0, header_bytes.first(sizeof(Header) / 2));
    grow_failpoint("final-header-partial");
    write_at(sizeof(Header) / 2, header_bytes.subspan(sizeof(Header) / 2));
    sync();
    grow_failpoint("finalizing");
    if (record_)
    {
        // An aligned footer can extend beyond the final index. Retire both
        // generations before reusing redo bytes, so recovery never selects an
        // older surviving checkpoint after the index erased its newer peer.
        std::array<uint8_t, 16> retired{};
        unsigned const latest = unsigned(record_->sequence % 2);
        write_at(record_offset(record_->storage_bytes, 1 - latest), retired);
        grow_failpoint("retire-partial");
        write_at(record_offset(record_->storage_bytes, latest), retired);
        grow_failpoint("retire-partial");
        sync();
        grow_failpoint("retired");
    }
    BitWriter index;
    gsz_encode(plan.sizes, plan.new_members, index);
    auto const bytes = index.asBytes();
    require(grow_add(plan.header.chunk_index_offset, bytes.size()) == plan.final_bytes,
            "final canonical index length mismatch");
    size_t const partial = bytes.size() / 2;
    write_at(plan.header.chunk_index_offset, bytes.first(partial));
    grow_failpoint("index-partial");
    write_at(plan.header.chunk_index_offset + partial, bytes.subspan(partial));
    sync();
    grow_failpoint("index");
    require(file_io::resize_file(fd_, file_io::Offset(plan.final_bytes)) == 0, "cannot trim same-file recovery workspace");
    sync();
    grow_failpoint("truncated");
    uint8_t const version = PlotGroupFile::FORMAT_VERSION;
    write_at(offsetof(Header, version), {&version, 1});
    sync();
    grow_failpoint("published");
    identity_.header = plan.header;
    identity_.bytes = plan.final_bytes;
    identity_.pending = false;
    identity_.old_members = identity_.target_members = plan.new_members;
    identity_.pending_peak_bytes = 0;
    old_sizes_ = plan.sizes;
    record_.reset();
    finalizing_ = false;
    config_.check_running();
    return {plan.final_bytes, plan.peak_file_bytes, plan.workspace_bytes, plan.new_members};
}
} // namespace group_grow
