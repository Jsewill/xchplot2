// Distributed contribution runs followed by direct sequential ordinary-v2
// packing. The packet codec is shared with same-file merge construction.
#pragma once
#include "host/GroupEngine.hpp"
#include "host/GroupStorage.hpp"
#include "host/GroupWriter.hpp"
#include "host/GroupPolicy.hpp"
#include <charconv>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <syncstream>
#ifdef __linux__
#include <sys/syscall.h>
#include <sys/xattr.h>
#endif

namespace group_grow
{
namespace external_detail
{
struct IOContext
{
    char const* role = "unclassified";
    std::string const* owner = nullptr;
    ExecutionIO* service = nullptr;
};
inline bool io_profile()
{
    static bool const enabled = [] { auto value = std::getenv("POS2GPU_PHASE_TIMING"); return value && value[0] == '1'; }();
    return enabled;
}
// Resolve file identity only for diagnostics. A missing/replaced descriptor or
// unavailable /proc must neither fail the operation nor change its errno.
class IOTrace
{
public:
    IOTrace(int fd, char const* operation, IOContext context, uint64_t offset = 0, uint64_t requested = 0) noexcept
        : operation_(operation), context_(context), offset_(offset), requested_(requested), enabled_(io_profile())
    {
        if (!enabled_) return;
        int const saved = errno;
        try
        {
            file_io::Stat value{};
            if (!file_io::inspect_fd(fd, &value)) { device_ = uint64_t(value.st_dev); inode_ = uint64_t(value.st_ino); }
#ifdef __linux__
#ifdef SYS_gettid
            auto const tid = ::syscall(SYS_gettid); if (tid > 0) tid_ = uint64_t(tid);
#endif
            std::array<char, PATH_MAX> path{};
            auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
            auto const size = ::readlink(descriptor.c_str(), path.data(), path.size());
            if (size > 0) path_.assign(path.data(), size_t(size));
#endif
        }
        catch (...) { enabled_ = false; }
        errno = saved;
    }
    void report(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end,
                int64_t result, int error) const noexcept
    {
        if (!enabled_) return;
        int const saved = errno;
        try
        {
            auto stamp = [](auto time)
            { return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count(); };
            std::osyncstream(std::cerr) << "[grow-io-timing] op=" << operation_ << " role=" << context_.role
                << " owner=" << std::quoted(context_.owner ? context_.owner->c_str() : "")
                << " path=" << std::quoted(path_) << " device=" << device_ << " inode=" << inode_
                << " thread=" << std::this_thread::get_id() << " tid=" << tid_ << " offset=" << offset_ << " requested_bytes=" << requested_
                << " result=" << result << " error=" << error << " begin_ns=" << stamp(begin) << " end_ns=" << stamp(end) << '\n';
        }
        catch (...) {}
        errno = saved;
    }
private:
    char const* operation_;
    IOContext context_;
    uint64_t offset_, requested_, device_ = 0, inode_ = 0, tid_ = 0;
    bool enabled_;
    std::string path_;
};
inline std::unique_lock<std::recursive_mutex> target_lock(std::shared_ptr<std::recursive_mutex> const& gate,
                                                        int fd, std::string const& owner)
{
    if (!gate) return {};
    std::unique_lock guard(*gate, std::defer_lock);
    if (!io_profile()) { guard.lock(); return guard; }
    IOTrace trace(fd, "target_lock", {"target_gate", &owner}); auto const begin = std::chrono::steady_clock::now();
    try { guard.lock(); }
    catch (std::system_error const& error)
    { trace.report(begin, std::chrono::steady_clock::now(), -1, error.code().value()); throw; }
    trace.report(begin, std::chrono::steady_clock::now(), 0, 0); return guard;
}
}
struct ScratchGrant
{
    std::string directory;
    uint64_t capacity_bytes = 0;
};
struct ExternalConfig
{
    std::vector<ScratchGrant> scratch;
    uint64_t segment_bytes = 64ull * 1024 * 1024;
    unsigned fan_in = 4;
    // Zero keeps the merge arity as the proactive reader-retention gate.
    unsigned packing_inputs = 0;
    // Runtime target-resource ownership, shared by adjacent group engines.
    // It is not part of the persisted construction settings.
    std::shared_ptr<std::recursive_mutex> target_gate;
};

class ExternalEngine
{
public:
    ExternalEngine(std::string path, Config config, ExternalConfig external,
                   bool resume = false, std::optional<Identity> initial = {});
    ~ExternalEngine();
    ExternalEngine(ExternalEngine const&) = delete;
    ExternalEngine& operator=(ExternalEngine const&) = delete;
    struct Inspection
    {
        ExternalConfig external;
        Identity identity;
        std::vector<ScratchGrant> scratch;
        std::vector<uint64_t> reserved_bytes, peak_reserved_bytes, allocated_bytes, occupied_bytes;
        uint64_t target_bytes = 0, target_allocated_bytes = 0, metadata_reserved_bytes = 0, metadata_allocated_bytes = 0, metadata_occupied_bytes = 0, peak_bytes = 0;
        uint16_t members = 0;
        bool finishing = false, complete = false;
    };
    static bool pending(std::string const& path);
    static Inspection inspect(std::string const& path, uint64_t memory_limit_bytes = UINT64_MAX);
    static Inspection inspect_retired(std::string const& path, std::string const& directory, uint64_t memory_limit_bytes = UINT64_MAX);
    static uint64_t memory_bytes(uint8_t k, Config const& config, ExternalConfig const& external,
                                 uint16_t batch_members = 1);
    enum class MemoryPhase { staging, packing, publication };
    static uint64_t phase_memory_bytes(uint8_t k, Config const& config, ExternalConfig const& external,
                                       MemoryPhase phase, uint16_t batch_members = 1);
    static uint64_t metadata_bytes(uint8_t k, ExternalConfig const& external);
    uint64_t required_memory(uint16_t batch_members = 1, unsigned workers = 0) const
    {
        auto cfg = config_; if (workers) cfg.workers = workers;
        return phase_memory_bytes(identity_.header.k, cfg, external_, memory_phase(), batch_members);
    }
    MemoryPhase memory_phase() const
    { return phase_ == Phase::stage ? MemoryPhase::staging : phase_ == Phase::pack ? MemoryPhase::packing : MemoryPhase::publication; }
    Identity const& identity() const { return identity_; }
    uint16_t committed_member_count() const { return members_; }
    bool can_append() const { return phase_ == Phase::stage; }
    Stats const& stats() const { return stats_; }
    using RunShape = ConstructionRun;
    void run_shapes(std::vector<RunShape>& result) const
    {
        result.clear(); result.reserve(runs_.size());
        for (auto const& run : runs_) result.push_back({run.count, run.bytes});
    }
    ExecutionIO scratch_service() const { return execution_scratch_; }
    ExecutionIO destination_service() const { return execution_destination_; }
    CpuTime contribution_cpu_time() const { auto result = contribution_cpu_; result.add(contribution_owner_cpu_); return result; }
    CpuTime packing_cpu_time() const { auto result = packing_cpu_; result.add(packing_owner_cpu_); return result; }
    CpuTime contribution_serial_cpu_time() const { auto result = contribution_driver_cpu_; result.add(contribution_owner_cpu_); return result; }
    CpuTime packing_serial_cpu_time() const { auto result = packing_driver_cpu_; result.add(packing_owner_cpu_); return result; }
    CpuTime packing_owner_cpu_time() const { return packing_owner_cpu_; }
    bool service_attributed() const { return roots_.size() == 1; }
    uint64_t packing_body_ns() const { return packing_body_ns_; }
    // Like append/finish, retuning belongs to the owner between operations.
    // Workers share codec storage; each task retains a small metadata reserve.
    // No encoder survives here.
    void set_workers(unsigned workers)
    {
        require(workers && required_memory(1, workers) <= config_.memory_limit_bytes, "external worker retune exceeds its admitted budget");
        config_.workers = workers;
    }
    // The controller opts in for automatic intervals. Keep the configured
    // fixed interval unchanged for later ordinary handoff or explicit resume.
    void tune_packing_checkpoints(uint64_t ceiling)
    {
        checkpoint_ceiling_ = std::min(ceiling, PackingCheckpointHistory::maximum_bytes);
        packing_checkpoint_bytes_ = checkpoint_ceiling_ ? std::min(config_.checkpoint_bytes, checkpoint_ceiling_) : config_.checkpoint_bytes;
    }
    uint64_t packing_checkpoint_bytes() const
    { return packing_checkpoint_bytes_ ? packing_checkpoint_bytes_ : config_.checkpoint_bytes; }
    uint64_t checkpoint_tuning_changes() const { return checkpoint_tuning_changes_; }
    uint64_t contribution_bytes() const { return contribution_bytes_; }
    struct BatchResources
    {
        uint16_t members = 0;
        uint64_t runs = 0, packets = 0, bytes = 0, source_bytes = 0, packet_bytes = 0;
        unsigned readers = 0;
    };
    BatchResources batch_resources() const;
    std::vector<uint64_t> scratch_reserved_bytes() const;
    std::vector<uint64_t> scratch_peak_bytes() const;
    void confirm_identity();
    class PreparedInput;
    // Controller-thread preparation is separate from CPU append. The engine
    // must be idle, and the returned owner must die before its engine. A null
    // result leaves the ordinary in-memory/exact-admission path available.
    static uint64_t input_memory_bytes(Config const& config, uint16_t members);
    // Logical complete preflight payload; reserve() additionally admits every
    // ownership header/alignment unit. Wide count bounds use ordinary input.
    static uint64_t input_scratch_bytes(uint8_t k, uint16_t members, uint64_t fragment_bound);
    std::unique_ptr<PreparedInput> prepare_input(uint16_t members, uint64_t fragment_bound);
    AppendResult append(PreparedInput& input, uint16_t prefix = 0);
    class ContributionInput;
    // Portable compressed member streams become ordinary owned runs only
    // after local validation, exact capacity admission and a durable append.
    static uint64_t contribution_input_memory_bytes(Config const& config, uint16_t members);
    static uint64_t contribution_input_scratch_bytes(uint8_t k, uint16_t members, uint64_t fragment_bound);
    std::unique_ptr<ContributionInput> prepare_contributions(uint16_t members, uint64_t fragment_bound);
    AppendResult append(ContributionInput& input, uint16_t prefix = 0);
    AppendResult append(Members members);
    // Exclusive engine ownership is required, as for append/finish. A queued
    // owner may run this boundary; its caller can lower the grant only after
    // joining it. The staging summary and checkpoint buffer are then gone.
    void start_packing();
    Result finish();
    bool release_completed();
    uint64_t remaining_metadata_bytes() const { return remaining_metadata_bytes_; }

private:
    enum class Phase { stage, pack, ready, done, empty };
    struct FileIdentity
    {
        uint64_t device = 0, inode = 0;
        bool operator==(FileIdentity const&) const = default;
    };
    struct Stamp
    {
        FileIdentity file;
        uint64_t bytes = 0;
        int64_t modified = 0, changed = 0;
        uint32_t modified_ns = 0, changed_ns = 0;
        bool operator==(Stamp const&) const = default;
    };
    struct Root
    {
        std::string path;
        FileIdentity directory, parent;
        uint64_t capacity = 0, alignment = 0, used = 0, peak = 0;
    };
    struct Segment
    {
        uint32_t root = 0, ordinal = 0;
        uint64_t capacity = 0, bytes = 0;
        Stamp stamp;
    };
    struct Run
    {
        uint64_t id = 0, bytes = 0, consumed = 0;
        uint16_t first = 0, count = 0;
        std::vector<Segment> segments;
    };
    struct Packet { uint16_t first, count; uint32_t bytes; };
    static_assert(sizeof(Packet) == 8);
    struct Summary { ChunkSummary codec; uint64_t bytes = 0; };
    static constexpr uint64_t segment_header_bytes = 64;
    static constexpr size_t max_runs = 64, max_roots = 1024;
    std::string path_, work_, token_;
    Config config_;
    ExternalConfig external_;
    Identity identity_;
    FileIdentity target_, work_identity_, parent_, checkpoint_identity_;
    Stamp completed_;
    int fd_ = -1;
    bool created_ = false, confirmed_ = false, initialized_ = false, released_ = false, bootstrap_recovered_ = false;
    bool input_cleanup_failed_ = false;
    Phase phase_ = Phase::stage;
    uint16_t members_ = 0;
    uint64_t next_id_ = 0, final_bound_ = 0, output_bytes_ = 0, packed_chunks_ = 0, contribution_bytes_ = 0;
    uint64_t checkpoint_ceiling_ = 0, packing_checkpoint_bytes_ = 0, checkpoint_tuning_changes_ = 0;
    std::vector<Root> roots_;
    std::vector<Run> runs_;
    Run summary_run_, pending_summary_;
    std::array<uint8_t, 32> summary_digest_{};
    uint64_t remaining_metadata_bytes_ = 0;
    uint64_t index_bytes_ = 0, packing_source_bytes_ = 0, packing_work_ = 0, packet_count_ = 0;
    uint32_t index_carry_bits_ = 0, index_carry_value_ = 0;
    FileIdentity index_identity_;
    std::string index_name_;
    Stats stats_;
    CpuTime contribution_cpu_, packing_cpu_;
    CpuTime contribution_owner_cpu_, packing_owner_cpu_, contribution_driver_cpu_, packing_driver_cpu_;
    ExecutionIO execution_scratch_, execution_destination_;
    uint64_t packing_body_ns_ = 0;
    struct Orphan { uint32_t root; std::string name; Stamp stamp; uint64_t reserved = 0; };
    std::vector<Orphan> orphans_;

    class Reader;
    class Writer;
    class InputReader;
    struct InputWindowExceeded : std::runtime_error
    { InputWindowExceeded() : std::runtime_error("prepared chunk exceeds its input window") {} };
    PreparedInput* prepared_ = nullptr;
    ContributionInput* contributions_ = nullptr;
    AppendResult append_source(Members members, PreparedInput* input, uint16_t count);
    void discard_input(Run& run, bool refresh = false);
    bool staging_slot();
    uint64_t prepared_segments() const;
    void release_prepared() noexcept;
    void release_contributions() noexcept;
    static uint64_t multiply(uint64_t a, uint64_t b)
    { require(!b || a <= UINT64_MAX / b, "external size overflow"); return a * b; }
    uint64_t codec_window_chunks(unsigned workers, uint64_t maximum_work, uint64_t limit, uint64_t codec_bytes) const
    {
        uint64_t const baseline = ChunkCodec::estimate_base(), budget = codec_bytes / workers;
        require(limit && maximum_work > baseline && maximum_work <= budget, "external codec window exceeds admission");
        // ParallelEncoder partitions contiguous chunks with ceil(count/workers)
        // tasks per worker. Bound that ceiling by each worker's codec budget.
        uint64_t const per_worker = (budget - baseline) / (maximum_work - baseline);
        return std::min(limit, multiply(workers, per_worker));
    }
    static uint64_t descriptor_limit(ExternalConfig const& external);
    static uint64_t checkpoint_buffer_bytes(ExternalConfig const& external);
    static uint64_t packing_index_bytes(Config const& config)
    { return std::clamp<uint64_t>(config.window_bytes, 65536, 1024 * 1024); }
    static Inspection inspect_at(std::string const& path, std::string const& work, uint64_t memory_limit_bytes, bool retired);
    static Inspection inspect_receipt(std::string const& path, std::string const& holder, uint64_t memory_limit_bytes);
    bool save_retired_receipt(std::string const& holder, FileIdentity holder_identity);
    void measure_remaining_metadata(std::string const& holder, FileIdentity holder_identity);
    static uint64_t aligned(uint64_t bytes, uint64_t alignment)
    { return multiply(bytes / alignment + (bytes % alignment != 0), alignment); }
    static FileIdentity file_identity(std::string const& path, bool directory);
    static FileIdentity file_identity(int fd);
    static Stamp stamp(std::string const& path);
    static std::string digest(std::string_view body);
    static std::string checkpoint_path(std::string const& work, std::string const& target, uint64_t limit);
    static void sync_directory(std::string const& path, external_detail::IOContext context = {"directory"});
    std::unique_lock<std::recursive_mutex> target_lock() const
    { return external_detail::target_lock(external_.target_gate, fd_, path_); }
    void check_paths() const;
    void sync_target();
    void read_target(uint64_t offset, std::span<uint8_t> bytes);
    void write_target(uint64_t offset, std::span<uint8_t const> bytes);
    void save(std::string const& label, std::span<Summary const> summaries = {});
    void load();
    void initialize();
    void validate_runs() const;
    void cleanup();
    std::string segment_path(Run const& run, Segment const& segment) const;
    std::array<uint8_t, segment_header_bytes> segment_header(Run const& run, Segment const& segment) const;
    static std::array<uint8_t, segment_header_bytes> segment_header(std::string const& token, Run const& run, Segment const& segment);
    static void read_summary(std::vector<Root> const& roots, std::string const& token, Run const& run,
                             std::array<uint8_t, 32> const& checksum, std::span<uint8_t> destination = {}, Stats* stats = nullptr,
                             external_detail::IOContext context = {}, CodecScheduler* scheduler = nullptr);
    void write_summary(std::span<Summary const> summaries);
    bool reserve(Run& run, uint64_t bytes, char const* role = "contribution");
    void discard(Run const& run);
    void incoming_regions(Members members, uint64_t chunk, std::vector<size_t>& cursors,
                          std::vector<FragmentRegion>& regions) const;
    void balance();
    bool merge_range(size_t first, size_t count);
    void pack();
};
}

namespace group_grow
{
namespace external_detail
{
inline bool independent_packing_io(StorageDevice const& target, std::span<StorageDevice const> sources)
{
    if (target.drives.empty()) return false;
    for (auto const& source : sources)
    {
        if (source.drives.empty()) return false;
        for (auto const& drive : target.drives)
            if (std::find(source.drives.begin(), source.drives.end(), drive) != source.drives.end()) return false;
    }
    return true;
}
struct Descriptor
{
    int fd = -1;
    explicit Descriptor(int value) : fd(value) { require(fd >= 0, "cannot open external construction file"); }
    ~Descriptor() { if (fd >= 0) file_io::close_file(fd); }
    Descriptor(Descriptor const&) = delete;
    Descriptor& operator=(Descriptor const&) = delete;
};
inline void reserve_file(int fd, uint64_t bytes)
{
#if defined(__linux__) || defined(_WIN32)
    int status;
    do { status = file_io::allocate_file(fd, 0, 0, file_io::Offset(bytes)); } while (status && errno == EINTR);
    require(!status, "cannot physically reserve external construction space: " + std::string(std::strerror(errno)));
#else
    (void)fd; (void)bytes;
    throw std::runtime_error("external construction requires supported native physical allocation");
#endif
}
inline void evict(int fd)
{
#if defined(__linux__) && defined(POSIX_FADV_DONTNEED)
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#else
    (void)fd;
#endif
}
inline void write(int fd, uint64_t offset, std::span<uint8_t const> bytes, Stats* stats = nullptr, IOContext context = {})
{
    while (!bytes.empty())
    {
        auto const requested = std::min(bytes.size(), size_t(file_io::max_transfer));
        IOTrace trace(fd, "pwrite", context, offset, requested);
        auto const begin = std::chrono::steady_clock::now();
        file_io::Count const n = file_io::write_at(fd, bytes.data(), requested, file_io::Offset(offset));
        int const error = n < 0 ? errno : 0; auto const end = std::chrono::steady_clock::now();
        uint64_t const elapsed = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        if (stats) { stats->write_ns += elapsed; ++stats->write_calls; }
        if (context.service) { context.service->write_ns += elapsed; ++context.service->write_calls; }
        trace.report(begin, end, n, error);
        if (n < 0 && error == EINTR) continue;
        require(n > 0, "external construction write failed: " + std::string(std::strerror(error)));
        if (stats) stats->written_bytes += uint64_t(n);
        if (context.service) context.service->written_bytes += uint64_t(n);
        offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
    }
}
inline void read(int fd, uint64_t offset, std::span<uint8_t> bytes, Stats* stats = nullptr, IOContext context = {})
{
    while (!bytes.empty())
    {
        auto const requested = std::min(bytes.size(), size_t(file_io::max_transfer));
        IOTrace trace(fd, "pread", context, offset, requested);
        auto const begin = std::chrono::steady_clock::now();
        file_io::Count const n = file_io::read_at(fd, bytes.data(), requested, file_io::Offset(offset));
        int const error = n < 0 ? errno : 0; auto const end = std::chrono::steady_clock::now();
        uint64_t const elapsed = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        if (stats) { stats->read_ns += elapsed; ++stats->read_calls; }
        if (context.service) { context.service->read_ns += elapsed; ++context.service->read_calls; }
        trace.report(begin, end, n, error);
        if (n < 0 && error == EINTR) continue;
        require(n > 0, "external construction read failed");
        if (stats) stats->read_bytes += uint64_t(n);
        if (context.service) context.service->read_bytes += uint64_t(n);
        offset += uint64_t(n); bytes = bytes.subspan(size_t(n));
    }
}
inline void sync(int fd, Stats* stats = nullptr, IOContext context = {})
{
    int result, error; uint64_t elapsed = 0;
    do
    {
        IOTrace trace(fd, "fsync", context); auto const begin = std::chrono::steady_clock::now();
        result = file_io::flush(fd); error = result ? errno : 0; auto const end = std::chrono::steady_clock::now();
        elapsed += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        trace.report(begin, end, result, error);
    } while (result && error == EINTR);
    if (stats) { stats->sync_ns += elapsed; ++stats->syncs; }
    if (context.service) { context.service->sync_ns += elapsed; ++context.service->syncs; }
    require(!result, "cannot sync external construction file");
}
class CheckpointBuffer
{
public:
    explicit CheckpointBuffer(uint64_t limit)
    {
        require(limit >= 65 && limit <= SIZE_MAX, "external checkpoint buffer exceeds addressable memory");
        limit_ = size_t(limit); bytes_.reset(new uint8_t[limit_]);
    }
    void write(char const* data, size_t size)
    {
        require(size <= limit_ - used_, "external checkpoint exceeds metadata reservation");
        if (size) std::memcpy(bytes_.get() + used_, data, size);
        used_ += size;
    }
    std::span<uint8_t const> record(CodecScheduler* scheduler = nullptr)
    {
        auto const checksum = group_tools::hex(grow_digest({bytes_.get() + 65, used_ - 65}, scheduler));
        std::copy(checksum.begin(), checksum.end(), bytes_.get()); bytes_[64] = '\n';
        return {bytes_.get(), used_};
    }
private:
    std::unique_ptr<uint8_t[]> bytes_;
    size_t limit_ = 0, used_ = 65;
};
// The record owns these bytes for the complete parse. A bounded get area
// keeps the existing istream failure and in_avail checks without copying it.
class CheckpointInput : private std::streambuf, public std::istream
{
public:
    explicit CheckpointInput(std::string_view body) : std::istream(this)
    {
        auto* begin = const_cast<char*>(body.data()); setg(begin, begin, begin + body.size());
        exceptions(std::ios::badbit | std::ios::failbit);
    }
};
template<class Output, class T> void put(Output& out, T const& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<char const*>(&value), sizeof(value));
}
template<class T> T get(std::istream& in)
{
    static_assert(std::is_trivially_copyable_v<T>);
    T value{}; in.read(reinterpret_cast<char*>(&value), sizeof(value)); return value;
}
template<class Output> void put_string(Output& out, std::string_view value)
{ put(out, uint64_t(value.size())); out.write(value.data(), value.size()); }
inline std::string get_string(std::istream& in, uint64_t limit = 4096)
{
    auto const bytes = get<uint64_t>(in); require(bytes <= limit, "external checkpoint string is too long");
    std::string value(size_t(bytes), '\0'); in.read(value.data(), value.size()); return value;
}
}

inline ExternalEngine::FileIdentity ExternalEngine::file_identity(std::string const& path, bool directory)
{
    file_io::Stat value{};
    require(!file_io::inspect_link(path.c_str(), &value) && (directory ? file_io::is_directory(value.st_mode) : (file_io::is_regular(value.st_mode) && value.st_nlink == 1)),
            "external path is missing, replaced or has the wrong type: " + path);
    return {uint64_t(value.st_dev), uint64_t(value.st_ino)};
}
inline ExternalEngine::FileIdentity ExternalEngine::file_identity(int fd)
{
    file_io::Stat value{};
    require(!file_io::inspect_fd(fd, &value) && file_io::is_regular(value.st_mode) && value.st_nlink == 1, "external file handle is not exclusively owned");
    return {uint64_t(value.st_dev), uint64_t(value.st_ino)};
}
inline ExternalEngine::Stamp ExternalEngine::stamp(std::string const& path)
{
    file_io::Stat value{};
    require(!file_io::inspect_link(path.c_str(), &value) && file_io::is_regular(value.st_mode) && value.st_nlink == 1,
            "external segment is missing, replaced or shared: " + path);
#ifdef __APPLE__
    auto const modified = value.st_mtimespec, changed = value.st_ctimespec;
#else
    auto const modified = value.st_mtim, changed = value.st_ctim;
#endif
    return {{uint64_t(value.st_dev), uint64_t(value.st_ino)}, uint64_t(value.st_size),
        modified.tv_sec, changed.tv_sec, uint32_t(modified.tv_nsec), uint32_t(changed.tv_nsec)};
}
inline std::string ExternalEngine::digest(std::string_view body)
{
    return group_tools::hex(grow_digest({reinterpret_cast<uint8_t const*>(body.data()), body.size()}));
}
inline void ExternalEngine::sync_directory(std::string const& path, external_detail::IOContext context)
{
    external_detail::Descriptor directory(file_io::open_file(path.c_str(), file_io::read_only | file_io::directory | file_io::no_follow | file_io::close_on_exec));
    external_detail::IOTrace trace(directory.fd, "fsync", context);
    bool const timing = context.service || external_detail::io_profile();
    auto const begin = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    int const result = file_io::flush(directory.fd); int const error = result ? errno : 0;
    auto const end = timing ? std::chrono::steady_clock::now() : begin;
    trace.report(begin, end, result, error);
    if (context.service)
    {
        context.service->sync_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        ++context.service->syncs;
    }
    require(!result, "cannot sync external directory: " + path);
}
inline uint64_t ExternalEngine::descriptor_limit(ExternalConfig const& external)
{
    require(external.segment_bytes > segment_header_bytes && external.scratch.size() <= max_roots, "invalid external scratch layout");
    uint64_t segments = multiply(max_runs, grow_add(external.scratch.size(), 2));
    for (auto const& grant : external.scratch) segments = grow_add(segments, grant.capacity_bytes / external.segment_bytes + 1);
    require(segments <= UINT32_MAX, "external segment metadata exceeds addressable limits"); return segments;
}
inline ExternalEngine::BatchResources ExternalEngine::batch_resources() const
{
    uint64_t bytes = 0;
    for (auto const& run : runs_) bytes = grow_add(bytes, run.bytes);
    return {members_, runs_.size(), packet_count_, bytes, packing_source_bytes_, sizeof(Packet),
        external_.packing_inputs ? external_.packing_inputs : external_.fan_in};
}
inline uint64_t ExternalEngine::metadata_bytes(uint8_t k, ExternalConfig const& external)
{
    // Target metadata contains the small manifest and the bounded index spool.
    // Immutable staging summaries are physically reserved in scratch instead.
    return multiply(2, grow_add(checkpoint_buffer_bytes(external),
        multiply(PlotGroupFile::getChunkCountForK(k), sizeof(uint64_t))));
}
inline uint64_t ExternalEngine::memory_bytes(uint8_t k, Config const& config,
                                           ExternalConfig const& external, uint16_t batch_members)
{
    return std::max({phase_memory_bytes(k, config, external, MemoryPhase::staging, batch_members),
        phase_memory_bytes(k, config, external, MemoryPhase::packing, batch_members),
        phase_memory_bytes(k, config, external, MemoryPhase::publication, batch_members)});
}
inline uint64_t ExternalEngine::checkpoint_buffer_bytes(ExternalConfig const& external)
{
    uint64_t paths = 0;
    for (auto const& grant : external.scratch) paths = grow_add(paths, multiply(grant.directory.size() + 256, 4));
    return grow_add(1024 * 1024, grow_add(paths, multiply(descriptor_limit(external), 256)));
}
inline uint64_t ExternalEngine::phase_memory_bytes(uint8_t k, Config const& config,
    ExternalConfig const& external, MemoryPhase phase, uint16_t batch_members)
{
    require(k >= 18 && k <= 28 && !(k & 1) && config.window_bytes && external.fan_in >= 2 &&
        external.fan_in <= max_runs && (!external.packing_inputs ||
        (external.packing_inputs >= external.fan_in && external.packing_inputs <= max_runs)),
        "invalid external memory admission");
    require(phase == MemoryPhase::staging || phase == MemoryPhase::packing || phase == MemoryPhase::publication,
        "invalid external memory phase");
    bool const staging = phase == MemoryPhase::staging;
    // The checkpoint sink allocates one bounded record. Recovery retains its
    // input record while rebuilding persistent metadata. Keep
    // the existing recovery/inspection admission of a record within one eighth
    // of the grant, including descriptor-vector capacity and parser temporaries.
    // Staging additionally owns one candidate summary. Exact sizing retains
    // one canonical size field; committed summaries remain in scratch.
    uint64_t const metadata = multiply(checkpoint_buffer_bytes(external), 8);
    if (phase == MemoryPhase::publication) return metadata;
    // Aggregate reader buffers, gathered payload, codec and canonical index
    // words coexist with bounded tasks. An overlapping target writer owns
    // another complete output window. Planning and encoding arrays are scoped.
    uint64_t const lanes = config.io_overlap ? 2 : 1;
    uint64_t const contribution_window = std::max(config.window_bytes,
        grow_add(sizeof(ChunkInput) + sizeof(uint64_t) + sizeof(EncodedBlock), multiply(batch_members, sizeof(FragmentRegion))));
    uint64_t const packing_window = std::max(config.window_bytes,
        uint64_t(sizeof(ChunkInput) + sizeof(uint64_t) + sizeof(EncodedBlock) + max_runs * sizeof(uint64_t)));
    uint64_t const planning_window = std::max(config.window_bytes,
        grow_add(sizeof(ChunkInput) + sizeof(ChunkSummary) + sizeof(EncodedBlock) + 3 * sizeof(uint64_t),
            multiply(batch_members, sizeof(FragmentRegion))));
    // BitWriter retains its geometrically grown words after each flush. One
    // encoded field can cross the threshold before that flush. Growth keeps
    // the old allocation alive beside its doubled replacement.
    uint64_t const index_memory = multiply(3, grow_add(packing_index_bytes(config), sizeof(uint64_t)));
    uint64_t const packing_tasks = grow_add(multiply(lanes, packing_window),
        grow_add(index_memory, PackingCheckpointHistory::memory_bytes() + max_roots * sizeof(bool)));
    uint64_t const tasks = staging ? std::max({planning_window, multiply(lanes, contribution_window), packing_tasks}) : packing_tasks;
    return grow_add(grow_add(config.codec_bytes, ParallelEncoder::memory_bytes(config.workers, !config.cpu_scheduler)), grow_add(grow_add(grow_add(multiply(config.window_bytes, 2 * lanes), tasks), std::max(config.window_bytes, uint64_t(max_runs * sizeof(Packet)))),
        grow_add(metadata,
        // Exact admission retains the candidate summary and a bounded writer
        // for one canonical size field, with payload and bit totals in scalars.
        grow_add(staging ? grow_add(multiply(PlotGroupFile::getChunkCountForK(k), sizeof(Summary)), sizeof(uint64_t)) : 0,
            grow_add(staging ? multiply(batch_members, sizeof(size_t) + sizeof(FragmentRegion)) : 0,
                multiply(multiply(lanes, UINT16_MAX), sizeof(EncodedChunk)))))));
}
inline bool ExternalEngine::pending(std::string const& path)
{
    file_io::Stat value{};
    if (!file_io::inspect_link((path + ".external").c_str(), &value))
    {
        require(file_io::is_directory(value.st_mode), "external checkpoint path is not an owned directory"); return true;
    }
    require(errno == ENOENT, "cannot inspect external checkpoint"); return false;
}
inline std::vector<uint64_t> ExternalEngine::scratch_reserved_bytes() const
{
    std::vector<uint64_t> result; result.reserve(roots_.size());
    for (auto const& root : roots_) result.push_back(root.used); return result;
}
inline std::vector<uint64_t> ExternalEngine::scratch_peak_bytes() const
{
    std::vector<uint64_t> result; result.reserve(roots_.size());
    for (auto const& root : roots_) result.push_back(root.peak); return result;
}
inline void ExternalEngine::check_paths() const
{
    require(!released_, "external construction was handed off to ordinary growth");
    require(file_identity(file_io::path_text(file_io::native_path(path_).parent_path()), true) == parent_, "external destination parent directory was replaced");
    require(file_identity(path_, false) == target_, "external destination was replaced");
    if (!initialized_) return;
    require(file_identity(work_, true) == work_identity_, "external checkpoint directory was replaced");
    for (auto const& root : roots_)
    {
        require(file_identity(file_io::path_text(file_io::native_path(root.path).parent_path()), true) == root.parent, "external scratch parent directory was replaced");
        require(file_identity(root.path, true) == root.directory, "external scratch directory was replaced");
    }
}
inline void ExternalEngine::sync_target()
{ auto target_guard = target_lock(); external_detail::sync(fd_, &stats_, {"final_output", &path_, &execution_destination_}); }
inline void ExternalEngine::read_target(uint64_t offset, std::span<uint8_t> bytes)
{ external_detail::read(fd_, offset, bytes, &stats_, {"final_output", &path_, &execution_destination_}); }
inline void ExternalEngine::write_target(uint64_t offset, std::span<uint8_t const> bytes)
{
    auto target_guard = target_lock();
    require(offset <= config_.capacity_bytes && bytes.size() <= config_.capacity_bytes - offset,
        "external destination exceeds admitted capacity");
    external_detail::write(fd_, offset, bytes, &stats_, {"final_output", &path_, &execution_destination_});
}
inline std::string ExternalEngine::segment_path(Run const& run, Segment const& segment) const
{
    return roots_.at(segment.root).path + "/run-" + std::to_string(run.id) + "-" +
        std::to_string(segment.ordinal);
}
inline std::array<uint8_t, ExternalEngine::segment_header_bytes>
ExternalEngine::segment_header(Run const& run, Segment const& segment) const
{ return segment_header(token_, run, segment); }
inline std::array<uint8_t, ExternalEngine::segment_header_bytes>
ExternalEngine::segment_header(std::string const& token, Run const& run, Segment const& segment)
{
    std::array<uint8_t, segment_header_bytes> result{};
    auto const hash = group_tools::hex(digest(token + ":" + std::to_string(run.id) + ":" +
        std::to_string(segment.ordinal)));
    std::copy(hash.begin(), hash.end(), result.begin()); return result;
}
inline void ExternalEngine::read_summary(std::vector<Root> const& roots, std::string const& token, Run const& run,
    std::array<uint8_t, 32> const& checksum, std::span<uint8_t> destination, Stats* stats, external_detail::IOContext io, CodecScheduler* scheduler)
{
    using namespace external_detail;
    require(destination.empty() || destination.size() == run.bytes, "external summary destination coverage changed");
    std::unique_ptr<uint8_t[]> buffer;
    if (destination.empty()) buffer.reset(new uint8_t[65536]);
    struct sha256 context; sha256_init(&context);
    uint64_t position = 0; uint32_t ordinal = 0;
    for (auto const& segment : run.segments)
    {
        require(segment.root < roots.size() && segment.ordinal == ordinal++ && segment.capacity > segment_header_bytes &&
            segment.bytes && segment.bytes <= segment.capacity - segment_header_bytes && segment.stamp.bytes == segment.capacity &&
            segment.stamp.file.inode && position <= run.bytes && segment.bytes <= run.bytes - position,
            "invalid external summary segment descriptor");
        auto const path = roots[segment.root].path + "/run-" + std::to_string(run.id) + "-" + std::to_string(segment.ordinal);
        require(stamp(path) == segment.stamp, "external summary segment changed");
        Descriptor file(file_io::open_file(path.c_str(), file_io::read_only | file_io::no_follow | file_io::close_on_exec));
        require(file_identity(file.fd) == segment.stamp.file, "external summary handle was replaced");
        std::array<uint8_t, segment_header_bytes> header{}; read(file.fd, 0, header, stats, io);
        require(header == segment_header(token, run, segment), "external summary ownership header changed");
        for (uint64_t offset = 0; offset < segment.bytes;)
        {
            size_t const bytes = size_t(std::min<uint64_t>(65536, segment.bytes - offset));
            auto const part = destination.empty() ? std::span(buffer.get(), bytes) : destination.subspan(size_t(position + offset), bytes);
            read(file.fd, segment_header_bytes + offset, part, stats, io);
            { CodecScheduler::Lease compute(scheduler); sha256_update(&context, part.data(), part.size()); }
            offset += bytes;
        }
        require(stamp(path) == segment.stamp, "external summary changed while reading"); position += segment.bytes;
    }
    std::array<uint8_t, 32> actual{}; sha256_sum(&context, actual.data());
    require(position == run.bytes && actual == checksum, "external summary checksum or coverage changed");
}
inline void ExternalEngine::write_summary(std::span<Summary const> summaries)
{
    using namespace external_detail;
    require(phase_ == Phase::stage && members_ && !pending_summary_.bytes, "invalid external pending summary generation");
    require(summaries.size() == PlotGroupFile::getChunkCountForK(identity_.header.k), "external replacement summary coverage changed");
    auto bytes = std::span<uint8_t const>(reinterpret_cast<uint8_t const*>(summaries.data()), summaries.size_bytes());
    summary_digest_ = grow_digest(bytes, config_.cpu_scheduler.get());
    for (auto& segment : pending_summary_.segments)
    {
        auto const path = segment_path(pending_summary_, segment);
        require(stamp(path) == segment.stamp, "external pending summary segment changed");
        Descriptor file(file_io::open_file(path.c_str(), file_io::read_write | file_io::no_follow | file_io::close_on_exec));
        require(file_identity(file.fd) == segment.stamp.file && segment.bytes <= bytes.size(),
            "external pending summary handle or coverage changed");
        for (uint64_t offset = 0; offset < segment.bytes;)
        {
            size_t const count = size_t(std::min(config_.window_bytes, segment.bytes - offset));
            write(file.fd, segment_header_bytes + offset, bytes.subspan(size_t(offset), count), &stats_, {"checkpoint_summary", &path_, &execution_scratch_}); offset += count;
        }
        external_detail::sync(file.fd, &stats_, {"checkpoint_summary", &path_, &execution_scratch_});
        require(stamp(path).file == segment.stamp.file, "external written summary path was replaced");
        segment.stamp = stamp(path); pending_summary_.bytes += segment.bytes; bytes = bytes.subspan(size_t(segment.bytes));
        if (config_.release_cache) external_detail::evict(file.fd);
    }
    require(bytes.empty(), "external summary exceeded its admitted reservation");
    for (auto const& root : roots_) sync_directory(root.path, {"directory", &path_, &execution_scratch_});
    grow_failpoint("external-summary-flushed");
}
inline void ExternalEngine::save(std::string const& label, std::span<Summary const> summaries)
{
    using namespace external_detail;
    check_paths();
    for (auto const& segment : summary_run_.segments)
        require(stamp(segment_path(summary_run_, segment)) == segment.stamp, "external current summary changed before checkpoint");
    // A replacement summary stays charged alongside the previous immutable
    // generation until its referencing manifest is durable on the target.
    if (!pending_summary_.segments.empty()) write_summary(summaries);
    auto const& summary = pending_summary_.segments.empty() ? summary_run_ : pending_summary_;
    CheckpointBuffer out(checkpoint_buffer_bytes(external_));
    put_string(out, "xchplot2-external-3"); put_string(out, GROW_IDENTITY); put_string(out, path_);
    put_string(out, token_); put(out, target_); put(out, work_identity_); put(out, parent_);
    put(out, identity_.header); put_string(out, {reinterpret_cast<char const*>(identity_.memo.data()), identity_.memo.size()});
    put(out, external_.segment_bytes); put(out, external_.fan_in); put(out, external_.packing_inputs);
    put(out, uint64_t(roots_.size()));
    for (auto const& root : roots_)
    {
        put_string(out, root.path); put(out, root.directory); put(out, root.parent); put(out, root.capacity); put(out, root.alignment); put(out, root.peak);
    }
    put(out, phase_); put(out, members_); put(out, next_id_); put(out, final_bound_);
    put(out, output_bytes_); put(out, packed_chunks_); put(out, stats_);
    put(out, uint64_t(runs_.size()));
    for (auto const& run : runs_)
    {
        put(out, run.id); put(out, run.bytes); put(out, run.consumed); put(out, run.first); put(out, run.count);
        put(out, uint64_t(run.segments.size()));
        for (auto const& segment : run.segments) put(out, segment);
    }
    uint64_t const chunks = phase_ == Phase::stage ? PlotGroupFile::getChunkCountForK(identity_.header.k) : 0;
    put(out, chunks);
    if (phase_ == Phase::stage && members_)
    {
        require(!summary.segments.empty() && summary.count == members_ && summary.bytes == multiply(chunks, sizeof(Summary)),
            "external summary has no durable scratch generation");
        put(out, summary.id); put(out, summary.bytes); put(out, uint64_t(summary.segments.size()));
        for (auto const& segment : summary.segments) put(out, segment);
        put(out, summary_digest_);
    }
    put(out, index_bytes_); put(out, index_carry_bits_); put(out, index_carry_value_); put(out, index_identity_); put_string(out, index_name_); put(out, completed_); put(out, packing_source_bytes_); put(out, packing_work_); put(out, packet_count_);
    auto const record = out.record(config_.cpu_scheduler.get());
    auto check_summary = [&]
    {
        if (phase_ == Phase::stage && members_) for (auto const& segment : summary.segments)
            require(stamp(segment_path(summary, segment)) == segment.stamp, "external referenced summary changed before checkpoint");
    };
    // Bulk scratch I/O finishes before taking the shared target-I/O resource.
    {
        auto target_guard = target_lock(); check_paths(); check_summary();
        uint64_t occupied = 0;
        for (auto const& entry : std::filesystem::directory_iterator(file_io::native_path(work_)))
        {
            file_io::Stat value{}; require(!file_io::inspect_link(entry.path().c_str(), &value) && file_io::is_regular(value.st_mode), "unrecognized external checkpoint contents");
            occupied = grow_add(occupied, std::max(uint64_t(value.st_size), multiply(uint64_t(value.st_blocks), 512)));
        }
        file_io::Space allocation{}; require(!file_io::space_path(work_.c_str(), &allocation), "cannot inspect external checkpoint allocation");
        uint64_t const alignment = allocation.f_frsize ? allocation.f_frsize : allocation.f_bsize;
        require(alignment && grow_add(occupied, aligned(record.size(), alignment)) <= metadata_bytes(identity_.header.k, external_),
            "external checkpoint directory exceeds its metadata reservation");
        std::string temporary = work_ + "/checkpoint.partial.XXXXXX";
        Descriptor handle(file_io::temp_file(temporary.data())); auto const temporary_identity = file_identity(handle.fd);
        try
        {
            write(handle.fd, 0, record, &stats_, {"checkpoint_manifest", &path_, &execution_destination_});
            external_detail::sync(handle.fd, &stats_, {"checkpoint_manifest", &path_, &execution_destination_});
            auto const temporary_stamp = stamp(temporary); require(temporary_stamp.file == temporary_identity, "external temporary checkpoint was replaced");
            grow_failpoint(label + "-before-checkpoint"); check_paths(); check_summary();
            require(stamp(temporary) == temporary_stamp, "external temporary checkpoint changed before publication");
            file_io::Stat previous{}; int const exists = file_io::inspect_link((work_ + "/checkpoint").c_str(), &previous);
            if (checkpoint_identity_.inode) require(!exists && file_io::is_regular(previous.st_mode) && previous.st_nlink == 1 &&
                FileIdentity{uint64_t(previous.st_dev), uint64_t(previous.st_ino)} == checkpoint_identity_, "external checkpoint was replaced");
            else require(exists && errno == ENOENT, "unrecognized file occupies external checkpoint path");
            require(!file_io::rename_file(temporary.c_str(), (work_ + "/checkpoint").c_str()), "cannot install external checkpoint");
            checkpoint_identity_ = temporary_identity; sync_directory(work_, {"directory", &path_, &execution_destination_}); ++stats_.checkpoints;
            grow_failpoint(label + "-after-checkpoint");
        }
        catch (...)
        {
            file_io::Stat named{};
            if (!file_io::inspect_link(temporary.c_str(), &named) && file_io::is_regular(named.st_mode) &&
                FileIdentity{uint64_t(named.st_dev), uint64_t(named.st_ino)} == temporary_identity) file_io::unlink_file(temporary.c_str());
            throw;
        }
    }
    if (!pending_summary_.segments.empty())
    {
        auto retired = std::move(summary_run_); summary_run_ = std::move(pending_summary_); pending_summary_ = {};
        if (!retired.segments.empty()) { discard(retired); grow_failpoint("external-summary-retired"); }
    }
    else if (phase_ != Phase::stage && !summary_run_.segments.empty())
    {
        auto retired = std::move(summary_run_); summary_run_ = {};
        discard(retired); grow_failpoint("external-summary-retired");
    }
}
inline void ExternalEngine::load()
{
    using namespace external_detail;
    require(file_identity(work_, true).inode != 0, "invalid external checkpoint directory");
    auto const checkpoint = checkpoint_path(work_, path_, config_.memory_limit_bytes / 8);
    bootstrap_recovered_ = checkpoint != work_ + "/checkpoint";
    auto const info = stamp(checkpoint);
    require(info.bytes <= checkpoint_buffer_bytes(external_) && info.bytes <= config_.memory_limit_bytes / 8,
        "external checkpoint exceeds memory budget");
    Descriptor handle(file_io::open_file(checkpoint.c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow));
    require(file_identity(handle.fd) == info.file, "external checkpoint handle was replaced");
    if (!bootstrap_recovered_) checkpoint_identity_ = info.file;
    std::string record(size_t(info.bytes), '\0');
    read(handle.fd, 0, {reinterpret_cast<uint8_t*>(record.data()), record.size()}, nullptr, {"checkpoint_manifest", &path_, &execution_destination_});
    require(record.size() >= 65 && record[64] == '\n' && std::string_view(record).substr(0, 64) == digest(std::string_view(record).substr(65)),
        "external checkpoint checksum mismatch");
    CheckpointInput in(std::string_view(record).substr(65));
    require(get_string(in) == "xchplot2-external-3" && get_string(in) == GROW_IDENTITY,
        "external checkpoint source/version mismatch");
    require(get_string(in) == path_, "external checkpoint destination mismatch");
    token_ = get_string(in); require(token_.size() == 64, "invalid external ownership token");
    require(get<FileIdentity>(in) == target_, "external destination inode changed");
    work_identity_ = get<FileIdentity>(in);
    require(get<FileIdentity>(in) == parent_, "external destination parent inode changed");
    require(file_identity(work_, true) == work_identity_, "external checkpoint directory inode changed");
    identity_.header = get<Header>(in);
    require(identity_.header.magic == PlotGroupFile::MAGIC && identity_.header.k >= 18 && identity_.header.k <= 28 && !(identity_.header.k & 1), "invalid external checkpoint header");
    auto memo = get_string(in, 255);
    identity_.memo.assign(memo.begin(), memo.end());
    require(get<uint64_t>(in) == external_.segment_bytes && get<unsigned>(in) == external_.fan_in &&
        get<unsigned>(in) == external_.packing_inputs, "external checkpoint segment/reader settings changed");
    auto const roots = get<uint64_t>(in); require(roots == external_.scratch.size(), "external scratch roots changed");
    roots_.resize(size_t(roots));
    for (size_t i = 0; i < roots_.size(); ++i)
    {
        auto& root = roots_[i]; root.path = get_string(in); root.directory = get<FileIdentity>(in); root.parent = get<FileIdentity>(in);
        root.capacity = get<uint64_t>(in); root.alignment = get<uint64_t>(in); root.peak = get<uint64_t>(in);
        require(root.capacity <= external_.scratch[i].capacity_bytes && root.alignment &&
            file_io::native_path(root.path).parent_path() == external_.scratch[i].directory,
            "external scratch grant or directory changed");
    }
    phase_ = get<Phase>(in); members_ = get<uint16_t>(in); next_id_ = get<uint64_t>(in);
    require(required_memory() <= config_.memory_limit_bytes, "saved external workspace exceeds memory budget");
    final_bound_ = get<uint64_t>(in); output_bytes_ = get<uint64_t>(in); packed_chunks_ = get<uint64_t>(in);
    stats_ = get<Stats>(in);
    require(unsigned(phase_) <= unsigned(Phase::empty) && next_id_ <= UINT32_MAX &&
        output_bytes_ <= config_.capacity_bytes && packed_chunks_ <= PlotGroupFile::getChunkCountForK(identity_.header.k),
        "invalid external checkpoint progress");
    auto count = get<uint64_t>(in); require(count <= max_runs, "too many external checkpoint runs");
    runs_.resize(size_t(count)); uint64_t segments = 0;
    for (auto& run : runs_)
    {
        run.id = get<uint64_t>(in); run.bytes = get<uint64_t>(in); run.consumed = get<uint64_t>(in); run.first = get<uint16_t>(in); run.count = get<uint16_t>(in);
        count = get<uint64_t>(in); segments = grow_add(segments, count);
        require(segments <= descriptor_limit(external_) && count && run.id < next_id_ && run.consumed <= run.bytes, "invalid external segment count/identity");
        require(count <= uint64_t(in.rdbuf()->in_avail()) / sizeof(Segment), "external segment descriptors exceed the checkpoint record");
        run.segments.resize(size_t(count));
        for (auto& segment : run.segments) segment = get<Segment>(in);
    }
    count = get<uint64_t>(in);
    require(count == (phase_ == Phase::stage ? PlotGroupFile::getChunkCountForK(identity_.header.k) : 0), "external summary coverage changed");
    if (phase_ == Phase::stage && members_)
    {
        summary_run_.id = get<uint64_t>(in); summary_run_.bytes = get<uint64_t>(in); summary_run_.count = members_;
        count = get<uint64_t>(in); segments = grow_add(segments, count);
        require(count && segments <= descriptor_limit(external_) && count <= uint64_t(in.rdbuf()->in_avail()) / sizeof(Segment) &&
            summary_run_.id < next_id_ && summary_run_.bytes == multiply(PlotGroupFile::getChunkCountForK(identity_.header.k), sizeof(Summary)) &&
            std::none_of(runs_.begin(), runs_.end(), [&](Run const& run) { return run.id == summary_run_.id; }),
            "invalid external summary generation");
        summary_run_.segments.resize(size_t(count));
        for (auto& segment : summary_run_.segments) segment = get<Segment>(in);
        summary_digest_ = get<std::array<uint8_t, 32>>(in);
    }
    index_bytes_ = get<uint64_t>(in); index_carry_bits_ = get<uint32_t>(in); index_carry_value_ = get<uint32_t>(in);
    index_identity_ = get<FileIdentity>(in); index_name_ = get_string(in, 64); completed_ = get<Stamp>(in);
    require(index_name_.empty() || (index_name_.starts_with("index.") && file_io::native_path(index_name_).filename() == index_name_), "invalid external index name");
    packing_source_bytes_ = get<uint64_t>(in); packing_work_ = get<uint64_t>(in); packet_count_ = get<uint64_t>(in);
    require(index_carry_bits_ < 8 && (index_carry_bits_ ? index_carry_value_ < (1u << index_carry_bits_) : !index_carry_value_), "invalid external partial index byte");
    if (phase_ == Phase::pack)
    {
        auto const index = stamp(work_ + "/" + index_name_);
        require(index.file == index_identity_ && index.bytes >= ((packed_chunks_ || index_bytes_) ? grow_add(segment_header_bytes, index_bytes_) : 0), "external index is missing, replaced or truncated");
    }
    require(in.rdbuf()->in_avail() == 0, "external checkpoint has trailing data");
    if (bootstrap_recovered_) require(phase_ == Phase::stage && !members_ && !next_id_ && !packed_chunks_ && runs_.empty() &&
        output_bytes_ == sizeof(Header) + identity_.memo.size(),
        "only an unacknowledged initial external checkpoint may be recovered from a temporary record");
    uint64_t const metadata = metadata_bytes(identity_.header.k, external_);
    require(metadata <= config_.capacity_bytes && final_bound_ <= config_.capacity_bytes - metadata,
        "saved external output/checkpoint reservation exceeds capacity");
    require(index_bytes_ <= multiply(PlotGroupFile::getChunkCountForK(identity_.header.k), 8), "external index cursor exceeds its bound");
    if (phase_ == Phase::empty) require(!members_ && runs_.empty() && !output_bytes_ && !packed_chunks_ && !index_identity_.inode, "invalid empty external checkpoint");
    if (phase_ == Phase::stage) require(!packed_chunks_ && !index_bytes_ && !index_identity_.inode &&
        output_bytes_ == sizeof(Header) + identity_.memo.size(), "invalid external staging progress");
    require(!members_ || (packet_count_ && packet_count_ <= members_ && packing_source_bytes_ && packing_source_bytes_ <= config_.window_bytes &&
        packing_work_ == ChunkCodec::estimate_work(identity_.header.k, packing_source_bytes_, 0) && packing_work_ <= config_.codec_bytes),
        "saved external packing workspace exceeds admission");
    initialized_ = true; check_paths(); validate_runs();
    if (!summary_run_.segments.empty())
    {
        read_summary(roots_, token_, summary_run_, summary_digest_, {}, &stats_, {"checkpoint_summary", &path_, &execution_scratch_}, config_.cpu_scheduler.get());
        for (auto const& segment : summary_run_.segments)
            roots_[segment.root].used = grow_add(roots_[segment.root].used, segment.capacity);
    }
    for (auto const& run : runs_) for (auto const& segment : run.segments)
    {
        file_io::Stat value{};
        if (!file_io::inspect_link(segment_path(run, segment).c_str(), &value))
            roots_[segment.root].used = grow_add(roots_[segment.root].used, segment.capacity);
        else require(errno == ENOENT, "cannot inspect external run allocation");
    }
    file_io::Stat target{}; require(!file_io::inspect_fd(fd_, &target) && uint64_t(target.st_size) >= output_bytes_, "external output is truncated");
    if (phase_ == Phase::ready || phase_ == Phase::done)
        require(uint64_t(target.st_size) == output_bytes_, "published external output length changed");
    if (phase_ == Phase::done) require(stamp(path_) == completed_, "completed external output changed");
    // All required inputs were checked before considering any uncommitted
    // scratch tail. Adopt only files carrying this namespace's ownership token.
    for (size_t i = 0; i < roots_.size(); ++i)
    {
        auto& root = roots_[i];
        for (auto const& entry : std::filesystem::directory_iterator(file_io::native_path(root.path)))
        {
            std::string const name = file_io::path_text(entry.path().filename());
            uint64_t id = 0; uint32_t ordinal = 0; auto const dash = name.find('-', 4);
            bool named = name.starts_with("run-") && dash != name.npos;
            if (named)
            {
                auto const a = std::from_chars(name.data() + 4, name.data() + dash, id);
                auto const b = std::from_chars(name.data() + dash + 1, name.data() + name.size(), ordinal);
                named = a.ec == std::errc{} && a.ptr == name.data() + dash && b.ec == std::errc{} && b.ptr == name.data() + name.size();
            }
            bool known = false;
            if (named) for (auto const& run : runs_)
                if (run.id == id && ordinal < run.segments.size())
                {
                    auto const& segment = run.segments[ordinal];
                    known = segment.root == i && segment_path(run, segment) == entry.path(); break;
                }
            if (named && id == summary_run_.id && ordinal < summary_run_.segments.size())
            {
                auto const& segment = summary_run_.segments[ordinal];
                known = segment.root == i && segment_path(summary_run_, segment) == entry.path();
            }
            if (known) continue;
            auto const value = stamp(file_io::path_text(entry.path())); file_io::Stat allocation{};
            require(!file_io::inspect_link(entry.path().c_str(), &allocation), "cannot inspect external orphan allocation");
            uint64_t const reserved = std::max(aligned(value.bytes, root.alignment), multiply(uint64_t(allocation.st_blocks), 512));
            root.used = grow_add(root.used, reserved);
            if (!named || value.bytes < segment_header_bytes) continue;
            Run run; run.id = id; Segment segment; segment.ordinal = ordinal;
            std::array<uint8_t, segment_header_bytes> header{};
            Descriptor file(file_io::open_file(entry.path().c_str(), file_io::read_only | file_io::no_follow | file_io::close_on_exec)); read(file.fd, 0, header, nullptr, {"scratch_ownership", &path_});
            if (header == segment_header(run, segment))
            {
                require(grow_add(segments, orphans_.size()) < descriptor_limit(external_), "external orphan metadata exceeds its memory reservation");
                orphans_.push_back({uint32_t(i), name, value, reserved});
            }
        }
        require(root.used <= root.capacity && root.peak <= root.capacity, "external scratch occupancy exceeds its saved grant");
        root.peak = std::max(root.peak, root.used);
    }
    for (auto const& entry : std::filesystem::directory_iterator(file_io::native_path(work_)))
    {
        if (!file_io::path_text(entry.path().filename()).starts_with("checkpoint.partial.")) continue;
        auto const saved = stamp(file_io::path_text(entry.path()));
        if (saved.bytes < 65 || saved.bytes > checkpoint_buffer_bytes(external_) || saved.bytes > config_.memory_limit_bytes / 8) continue;
        Descriptor file(file_io::open_file(entry.path().c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow));
        std::string partial(size_t(saved.bytes), '\0'); read(file.fd, 0, {reinterpret_cast<uint8_t*>(partial.data()), partial.size()}, nullptr, {"checkpoint_manifest", &path_, &execution_destination_});
        if (partial[64] != '\n' || std::string_view(partial).substr(0, 64) != digest(std::string_view(partial).substr(65))) continue;
        CheckpointInput candidate(std::string_view(partial).substr(65));
        if (get_string(candidate) == "xchplot2-external-3" && get_string(candidate) == GROW_IDENTITY &&
            get_string(candidate) == path_ && get_string(candidate) == token_ && get<FileIdentity>(candidate) == target_ &&
            get<FileIdentity>(candidate) == work_identity_ && get<FileIdentity>(candidate) == parent_)
        {
            require(grow_add(segments, orphans_.size()) < descriptor_limit(external_), "external orphan metadata exceeds its memory reservation");
            orphans_.push_back({uint32_t(roots_.size()), file_io::path_text(entry.path().filename()), saved});
        }
    }
}
inline void ExternalEngine::validate_runs() const
{
    uint32_t first = 0;
    for (auto const& run : runs_)
    {
        require(run.first == first && run.count && uint32_t(run.count) <= UINT16_MAX - first,
            "external run member coverage changed");
        first += run.count; uint64_t bytes = 0; uint32_t ordinal = 0;
        for (auto const& segment : run.segments)
        {
            require(segment.root < roots_.size() && segment.ordinal == ordinal++ && segment.capacity > segment_header_bytes &&
                segment.bytes && segment.bytes <= segment.capacity - segment_header_bytes && segment.stamp.bytes == segment.capacity &&
                segment.stamp.file.inode, "invalid external run segment descriptor");
            bytes = grow_add(bytes, segment.bytes);
            file_io::Stat value{};
            if (file_io::inspect_link(segment_path(run, segment).c_str(), &value))
                require(errno == ENOENT && bytes <= run.consumed, "required external run segment is missing");
            else require(stamp(segment_path(run, segment)) == segment.stamp, "external run segment changed");
        }
        require(bytes == run.bytes, "external run length changed");
    }
    require(first == members_ || (phase_ == Phase::done && runs_.empty()), "external committed member coverage changed");
}
inline ExternalEngine::ExternalEngine(std::string path, Config config, ExternalConfig external,
                                     bool resume, std::optional<Identity> initial)
    : path_(file_io::path_text(std::filesystem::absolute(file_io::native_path(path)).lexically_normal())), work_(path_ + ".external"),
      config_(config), external_(std::move(external))
{
    require(config_.window_bytes >= sizeof(Header) && config_.codec_bytes && config_.checkpoint_bytes && config_.workers &&
        config_.capacity_bytes <= uint64_t(INT64_MAX) && !external_.scratch.empty() && external_.scratch.size() <= max_roots &&
        external_.segment_bytes > segment_header_bytes && external_.segment_bytes <= uint64_t(INT64_MAX) &&
        external_.fan_in >= 2 && external_.fan_in <= max_runs && (!external_.packing_inputs ||
        (external_.packing_inputs >= external_.fan_in && external_.packing_inputs <= max_runs)),
        "invalid external construction budgets");
    std::vector<std::string> canonical_roots; std::vector<FileIdentity> root_identities;
    for (auto& grant : external_.scratch)
    {
        grant.directory = file_io::path_text(std::filesystem::canonical(file_io::native_path(grant.directory)));
        auto const root_identity = file_identity(grant.directory, true);
        require(root_identity.inode && grant.capacity_bytes <= uint64_t(INT64_MAX), "invalid external scratch grant");
        require(std::find(root_identities.begin(), root_identities.end(), root_identity) == root_identities.end(), "duplicate external scratch directory inode");
        root_identities.push_back(root_identity);
        require(std::find(canonical_roots.begin(), canonical_roots.end(), grant.directory) == canonical_roots.end(), "duplicate external scratch directory grant");
        canonical_roots.push_back(grant.directory); require_disk_scratch(grant.directory);
    }
    {
        auto target_guard = target_lock();
        fd_ = file_io::open_file(path_.c_str(), (resume ? 0 : file_io::create | file_io::exclusive) | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600);
    }
    require(fd_ >= 0, "cannot open external destination: " + path_); created_ = !resume;
    try
    {
        require(!file_io::lock_file(fd_, file_io::lock_exclusive | file_io::lock_nonblocking), "external destination is already in use");
        auto const target = stamp(path_); target_ = target.file;
        require(file_identity(fd_) == target_, "external destination handle does not match its path");
        parent_ = file_identity(file_io::path_text(file_io::native_path(path_).parent_path()), true);
        if (resume)
        {
            require(pending(path_), "external resume requires its checkpoint directory"); load();
        }
        else
        {
            require(initial && !target.bytes && !initial->header.group_size && initial->memo.size() == initial->header.memo_length,
                "fresh external construction needs an empty group identity");
            require(!pending(path_), "external checkpoint already exists"); identity_ = *initial;
            identity_.header.version = PlotGroupFile::FORMAT_VERSION;
            identity_.header.chunk_index_offset = sizeof(Header) + identity_.memo.size();
        }
        auto const& h = identity_.header;
        require(h.magic == PlotGroupFile::MAGIC && h.version == PlotGroupFile::FORMAT_VERSION && h.k >= 18 && h.k <= 28 &&
            !(h.k & 1) && h.strength >= 2 && h.strength <= h.k - 3 && identity_.memo.size() <= 255 &&
            h.memo_length == identity_.memo.size(), "invalid external group identity");
        require(required_memory() <= config_.memory_limit_bytes, "external workspace exceeds memory budget");
        identity_.pending = phase_ != Phase::done && members_;
        identity_.old_members = 0; identity_.target_members = members_;
        identity_.pending_peak_bytes = grow_add(final_bound_, metadata_bytes(h.k, external_));
        if (phase_ == Phase::done) { identity_.bytes = output_bytes_; identity_.header.group_size = members_; }
    }
    catch (...)
    {
        auto target_guard = target_lock();
        file_io::Stat opened{}, named{};
        if (created_ && !file_io::inspect_fd(fd_, &opened) && !opened.st_size && !file_io::inspect_link(path_.c_str(), &named) && file_io::is_regular(named.st_mode) &&
            opened.st_dev == named.st_dev && opened.st_ino == named.st_ino) file_io::unlink_file(path_.c_str());
        file_io::close_file(fd_); fd_ = -1; throw;
    }
}
inline ExternalEngine::~ExternalEngine()
{
    release_contributions();
    release_prepared();
    if (fd_ < 0) return;
    if (created_ && !initialized_)
    {
        auto target_guard = target_lock();
        file_io::Stat value{};
        if (!file_io::inspect_link(path_.c_str(), &value) && !value.st_size &&
            FileIdentity{uint64_t(value.st_dev), uint64_t(value.st_ino)} == target_) file_io::unlink_file(path_.c_str());
    }
    if (phase_ == Phase::done && stats_.syncs) external_detail::evict(fd_);
    file_io::close_file(fd_);
}
inline void ExternalEngine::initialize()
{
    if (initialized_) return;
    {
        auto target_guard = target_lock();
        require(!file_io::make_directory(work_.c_str(), 0700), "cannot exclusively create external checkpoint directory");
        work_identity_ = file_identity(work_, true);
    }
    std::string seed = path_ + ":" + std::to_string(target_.device) + ":" + std::to_string(target_.inode) + ":" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()); token_ = digest(seed);
    for (auto const& grant : external_.scratch)
    {
        std::string const directory = grant.directory + "/.xchplot2-external-" + token_;
        require(!file_io::make_directory(directory.c_str(), 0700), "cannot exclusively create external scratch directory");
        file_io::Space space{}; require(!file_io::space_path(directory.c_str(), &space) && (space.f_frsize || space.f_bsize), "cannot inspect external scratch alignment");
        uint64_t const alignment = space.f_frsize ? space.f_frsize : space.f_bsize;
        roots_.push_back({directory, file_identity(directory, true), file_identity(grant.directory, true), grant.capacity_bytes, alignment, 0});
        sync_directory(grant.directory, {"directory", &path_});
    }
    initialized_ = true;
    {
        auto target_guard = target_lock();
        Header header = identity_.header; header.version = 0xf5;
        write_target(0, {reinterpret_cast<uint8_t const*>(&header), sizeof(header)});
        write_target(sizeof(Header), identity_.memo); sync_target();
        output_bytes_ = sizeof(Header) + identity_.memo.size();
        sync_directory(file_io::path_text(file_io::native_path(path_).parent_path()), {"directory", &path_, &execution_destination_});
    }
    save("external-initialized");
}
inline void ExternalEngine::confirm_identity()
{
    require(!input_cleanup_failed_, "prepared input cleanup failed; retain files for resume");
    check_paths(); confirmed_ = true;
    if (!initialized_) initialize();
    if (bootstrap_recovered_) { save("external-bootstrap-recovered"); bootstrap_recovered_ = false; }
    for (auto const& orphan : orphans_)
    {
        auto target_guard = orphan.root < roots_.size() ? std::unique_lock<std::recursive_mutex>() : target_lock();
        auto const path = (orphan.root < roots_.size() ? roots_[orphan.root].path : work_) + "/" + orphan.name;
        require(stamp(path) == orphan.stamp, "external uncommitted segment changed before cleanup");
        require(!file_io::unlink_file(path.c_str()), "cannot discard uncommitted external segment");
        if (orphan.root < roots_.size())
        {
            auto& root = roots_[orphan.root]; root.used -= orphan.reserved; sync_directory(root.path, {"directory", &path_, &execution_scratch_});
        }
    }
    orphans_.clear();
    { auto target_guard = target_lock(); sync_directory(work_, {"directory", &path_, &execution_destination_}); }
}
inline bool ExternalEngine::reserve(Run& run, uint64_t bytes, char const* role)
{
    check_paths(); std::vector<Segment> plan; auto remaining = bytes;
    require(run.id < UINT32_MAX, "external run identities exhausted");
    uint64_t existing = grow_add(summary_run_.segments.size(), pending_summary_.segments.size());
    for (auto const& saved : runs_) existing = grow_add(existing, saved.segments.size());
    existing = grow_add(existing, prepared_segments());
    uint64_t const limit = descriptor_limit(external_);
    require(existing <= limit, "external live descriptors exceed metadata reservation");
    std::vector<uint64_t> used; for (auto const& root : roots_) used.push_back(root.used);
    size_t next_root = size_t(run.id % roots_.size());
    while (remaining)
    {
        bool found = false;
        for (size_t search = 0; search < roots_.size(); ++search)
        {
            size_t const index = (next_root + search) % roots_.size(); auto const& root = roots_[index];
            uint64_t const available = (root.capacity - used[index]) / root.alignment * root.alignment;
            if (available <= segment_header_bytes) continue;
            uint64_t const desired = std::min(external_.segment_bytes, aligned(grow_add(remaining, segment_header_bytes), root.alignment));
            uint64_t const capacity = std::min(available, desired / root.alignment * root.alignment);
            if (capacity <= segment_header_bytes) continue;
            if (plan.size() >= limit - existing) return false;
            uint64_t const n = std::min(remaining, capacity - segment_header_bytes);
            plan.push_back({uint32_t(index), uint32_t(plan.size()), capacity, n, {}});
            used[index] += capacity; remaining -= n; next_root = (index + 1) % roots_.size(); found = true; break;
        }
        if (!found) return false;
    }
    run.segments = std::move(plan);
    // A crash before a segment's ownership header reaches disk can leave an
    // unrecognized filename. Preserve it and choose a new exclusive identity.
    for (;;)
    {
        bool occupied = false;
        for (auto const& segment : run.segments)
        {
            file_io::Stat value{};
            if (!file_io::inspect_link(segment_path(run, segment).c_str(), &value)) { occupied = true; break; }
            require(errno == ENOENT, "cannot inspect planned external segment");
        }
        if (!occupied) break;
        require(run.id < UINT32_MAX - 1, "external run identities exhausted"); ++run.id;
    }
    size_t charged = 0;
    try
    {
        for (auto& segment : run.segments)
        {
            auto const path = segment_path(run, segment);
            external_detail::Descriptor file(file_io::open_file(path.c_str(), file_io::create | file_io::exclusive | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600));
            segment.stamp = stamp(path); // Preserve ownership even if the first write fails.
            auto const header = segment_header(run, segment); external_detail::write(file.fd, 0, header, &stats_, {role, &path_, &execution_scratch_});
            external_detail::reserve_file(file.fd, segment.capacity);
            external_detail::sync(file.fd, &stats_, {role, &path_, &execution_scratch_});
            auto const current = stamp(path); require(current.file == segment.stamp.file && file_identity(file.fd) == current.file,
                "external reservation path was replaced"); segment.stamp = current; roots_[segment.root].used += segment.capacity; ++charged;
            roots_[segment.root].peak = std::max(roots_[segment.root].peak, roots_[segment.root].used);
        }
        for (auto const& root : roots_) sync_directory(root.path, {"directory", &path_, &execution_scratch_});
    }
    catch (...)
    {
        for (size_t i = 0; i < run.segments.size(); ++i)
        {
            auto const& segment = run.segments[i];
            if (!segment.stamp.file.inode) continue;
            auto const path = segment_path(run, segment);
            require(stamp(path).file == segment.stamp.file, "external reservation path was replaced");
            require(!file_io::unlink_file(path.c_str()), "cannot remove failed external reservation");
            if (i < charged) roots_[segment.root].used -= segment.capacity;
        }
        run.segments.clear(); run.bytes = 0;
        for (auto const& root : roots_) sync_directory(root.path, {"directory", &path_, &execution_scratch_});
        throw;
    }
    return true;
}
inline void ExternalEngine::discard(Run const& run)
{
    check_paths(); uint64_t end = 0;
    for (auto const& segment : run.segments)
    {
        end += segment.bytes;
        std::string const path = segment_path(run, segment); file_io::Stat exists{};
        if (file_io::inspect_link(path.c_str(), &exists))
        { require(errno == ENOENT && end <= run.consumed, "required external segment vanished"); continue; }
        require(stamp(path) == segment.stamp, "external segment changed before reclamation");
        require(!file_io::unlink_file(path.c_str()), "cannot reclaim external segment");
        roots_[segment.root].used -= segment.capacity; stats_.reclaimed_bytes += segment.capacity;
        grow_failpoint("external-segment-unlinked");
    }
    for (auto const& root : roots_) sync_directory(root.path, {"directory", &path_, &execution_scratch_});
}
class ExternalEngine::Writer
{
public:
    Writer(ExternalEngine& engine, Run& run, char const* role = "contribution", bool gated_copy = false)
        : engine_(engine), run_(run), role_(role), gated_copy_(gated_copy)
    { buffer_.reserve(size_t(engine.config_.window_bytes)); }
    size_t room() const { return buffer_.capacity() - buffer_.size(); }
    void write_unflushed(std::span<uint8_t const> bytes)
    {
        require(bytes.size() <= room(), "external buffered write exceeded its compute slice");
        buffer_.insert(buffer_.end(), bytes.begin(), bytes.end());
    }
    template<class T> void value_unflushed(T const& value)
    { write_unflushed({reinterpret_cast<uint8_t const*>(&value), sizeof(value)}); }
    void write(std::span<uint8_t const> bytes)
    {
        while (!bytes.empty())
        {
            size_t const n = std::min(bytes.size(), room());
            if (gated_copy_)
            {
                CodecScheduler::Lease compute(engine_.config_.cpu_scheduler.get());
                write_unflushed(bytes.first(n));
            }
            else write_unflushed(bytes.first(n));
            bytes = bytes.subspan(n);
            if (!room()) flush();
        }
    }
    uint64_t buffered_bytes() const { return buffer_.size(); }
    void flush()
    { write_direct(buffer_); buffer_.clear(); }
    void write_direct(std::span<uint8_t const> bytes)
    {
        while (!bytes.empty())
        {
            require(segment_ < run_.segments.size(), "external run exceeded its admitted reservation");
            auto const& segment = run_.segments[segment_];
            auto const n = size_t(std::min<uint64_t>(bytes.size(), segment.bytes - position_));
            if (opened_ != segment_)
            {
                file_ = std::make_unique<external_detail::Descriptor>(file_io::open_file(engine_.segment_path(run_, segment).c_str(), file_io::read_write | file_io::no_follow | file_io::close_on_exec));
                require(engine_.stamp(engine_.segment_path(run_, segment)).file == segment.stamp.file &&
                    engine_.file_identity(file_->fd) == segment.stamp.file, "external output segment was replaced"); opened_ = segment_;
            }
            external_detail::write(file_->fd, segment_header_bytes + position_, bytes.first(n), &engine_.stats_, {role_, &engine_.path_, &engine_.execution_scratch_});
            position_ += n; run_.bytes += n; bytes = bytes.subspan(n);
            if (position_ == segment.bytes) { ++segment_; position_ = 0; }
        }
    }
    template<class T> void value(T const& value)
    { write({reinterpret_cast<uint8_t const*>(&value), sizeof(value)}); }
    void finish()
    {
        flush(); file_.reset();
        uint64_t remaining = run_.bytes; std::vector<Segment> kept;
        for (auto segment : run_.segments)
        {
            auto const path = engine_.segment_path(run_, segment);
            require(engine_.stamp(path).file == segment.stamp.file, "external written segment was replaced");
            if (!remaining)
            {
                require(!file_io::unlink_file(path.c_str()), "cannot release unused external reservation");
                engine_.roots_[segment.root].used -= segment.capacity; continue;
            }
            segment.bytes = std::min(remaining, segment.bytes); remaining -= segment.bytes;
            uint64_t const capacity = aligned(segment.bytes + segment_header_bytes, engine_.roots_[segment.root].alignment);
            external_detail::Descriptor file(file_io::open_file(path.c_str(), file_io::read_write | file_io::no_follow | file_io::close_on_exec));
            require(engine_.file_identity(file.fd) == segment.stamp.file, "external completed segment handle was replaced");
            require(!file_io::resize_file(file.fd, file_io::Offset(capacity)), "cannot trim external contribution"); external_detail::sync(file.fd, &engine_.stats_, {role_, &engine_.path_, &engine_.execution_scratch_});
            engine_.roots_[segment.root].used -= segment.capacity - capacity; segment.capacity = capacity;
            segment.stamp = engine_.stamp(path); kept.push_back(segment);
            if (engine_.config_.release_cache) external_detail::evict(file.fd);
        }
        require(!remaining, "external contribution length exceeds segments"); run_.segments = std::move(kept);
        for (auto const& root : engine_.roots_) sync_directory(root.path, {"directory", &engine_.path_, &engine_.execution_scratch_});
    }
private:
    std::vector<uint8_t> buffer_;
    ExternalEngine& engine_; Run& run_; char const* role_; bool gated_copy_; size_t segment_ = 0, opened_ = SIZE_MAX; uint64_t position_ = 0;
    std::unique_ptr<external_detail::Descriptor> file_;
};
class ExternalEngine::Reader
{
public:
    Reader(ExternalEngine& engine, Run const& run, size_t active = 0, uint64_t start = UINT64_MAX,
           char const* role = "contribution", uint64_t* activity = nullptr)
        : engine_(engine), run_(run), position_(start == UINT64_MAX ? run.consumed : start), read_position_(position_), role_(role), activity_(activity),
          buffer_(size_t(std::max<uint64_t>(sizeof(Packet), engine.config_.window_bytes / (active ? active : std::max<size_t>(1, engine.runs_.size())))))
    {
        uint64_t skip = position_;
        while (segment_ < run_.segments.size() && skip >= run_.segments[segment_].bytes)
        { skip -= run_.segments[segment_].bytes; ++segment_; }
        offset_ = skip;
    }
    uint64_t position() const { return position_; }
    uint64_t remaining() const { return run_.bytes - position_; }
    void checkpoint(uint64_t consumed)
    { if (file_ && file_end_ <= consumed) { if (engine_.config_.release_cache) external_detail::evict(file_->fd); file_.reset(); opened_ = SIZE_MAX; } }
    void read(std::span<uint8_t> bytes)
    {
        require(bytes.size() <= remaining(), "external packet is truncated");
        while (!bytes.empty())
        {
            if (buffer_begin_ == buffer_end_)
            {
                buffer_begin_ = 0; buffer_end_ = size_t(std::min<uint64_t>(buffer_.size(), run_.bytes - read_position_));
                require(buffer_end_, "external preread reached an unexpected end"); read_direct(std::span(buffer_).first(buffer_end_));
            }
            size_t const n = std::min(bytes.size(), buffer_end_ - buffer_begin_);
            std::memcpy(bytes.data(), buffer_.data() + buffer_begin_, n);
            bytes = bytes.subspan(n); buffer_begin_ += n; position_ += n;
        }
    }
    Packet packet(uint16_t first)
    {
        Packet result{}; read({reinterpret_cast<uint8_t*>(&result), sizeof(result)});
        require(result.first == first && result.count && result.count <= uint32_t(run_.first) + run_.count - first &&
            result.bytes > 1 && result.bytes <= remaining(), "invalid external packet member coverage or length"); return result;
    }
private:
    void read_direct(std::span<uint8_t> bytes)
    {
        while (!bytes.empty())
        {
            auto const& segment = run_.segments.at(segment_);
            size_t const n = size_t(std::min<uint64_t>(bytes.size(), segment.bytes - offset_));
            if (opened_ != segment_)
            {
                file_ = std::make_unique<external_detail::Descriptor>(file_io::open_file(engine_.segment_path(run_, segment).c_str(), file_io::read_only | file_io::no_follow | file_io::close_on_exec));
                require(engine_.stamp(engine_.segment_path(run_, segment)) == segment.stamp &&
                    engine_.file_identity(file_->fd) == segment.stamp.file, "external input segment changed");
                opened_ = segment_; file_end_ = read_position_ + segment.bytes - offset_;
            }
            external_detail::read(file_->fd, segment_header_bytes + offset_, bytes.first(n), &engine_.stats_, {role_, &engine_.path_, &engine_.execution_scratch_});
            if (activity_) *activity_ = grow_add(*activity_, n);
            bytes = bytes.subspan(n); offset_ += n; read_position_ += n;
            if (offset_ == segment.bytes) { offset_ = 0; ++segment_; }
        }
    }
    ExternalEngine& engine_; Run const& run_;
    uint64_t position_ = 0, read_position_ = 0, offset_ = 0, file_end_ = 0;
    char const* role_; uint64_t* activity_;
    size_t segment_ = 0, opened_ = SIZE_MAX, buffer_begin_ = 0, buffer_end_ = 0;
    std::vector<uint8_t> buffer_;
    std::unique_ptr<external_detail::Descriptor> file_;
};
// Independent network member streams may arrive in any member order. Each
// stream is written to one owned provisional run, in chunk order, and all
// filesystem/codec mutation remains serialized by this sink. No sender paths,
// ownership records, or summaries enter the local checkpoint.
class ExternalEngine::ContributionInput
{
public:
    ~ContributionInput() { if (engine_) engine_->release_contributions(); }
    ContributionInput(ContributionInput const&) = delete;
    ContributionInput& operator=(ContributionInput const&) = delete;
    uint16_t size() const { return uint16_t(members_.size()); }
    bool sealed() const { return sealed_; }
    uint64_t written_bytes() const
    {
        std::lock_guard lock(mutex_);
        uint64_t bytes = 0;
        for (auto const& member : members_)
            bytes = grow_add(bytes, member.run.bytes + (member.writer ? member.writer->buffered_bytes() : 0));
        return bytes;
    }
    uint64_t reserved_bytes() const
    {
        std::lock_guard lock(mutex_);
        uint64_t bytes = 0;
        for (auto const& member : members_) for (auto const& segment : member.run.segments)
            bytes = grow_add(bytes, segment.capacity);
        for (auto const& segment : summary_.segments) bytes = grow_add(bytes, segment.capacity);
        return bytes;
    }
    void append_chunk(uint16_t member, uint64_t chunk, std::span<uint8_t const> bytes, uint16_t count = 1)
    {
        std::lock_guard lock(mutex_);
        require(engine_ && !sealed_ && member < size() && count && count <= size() - member,
            "invalid contribution input member");
        claim(member, count);
        auto& entry = members_[member];
        require(!entry.failed && !entry.finished && chunk == entry.next && chunk < PlotGroupFile::getChunkCountForK(engine_->identity_.header.k),
            "contribution input chunks must arrive once in order");
        entry.failed = true;
        require(bytes.size() > 1 && bytes.size() <= UINT32_MAX && bytes.size() <= engine_->config_.window_bytes,
            "contribution input exceeds its receive window");
        auto const work = ChunkCodec::estimate_work(engine_->identity_.header.k, bytes.size(), 0);
        require(work <= engine_->config_.codec_bytes, "contribution validation exceeds codec budget");
        {
            ThreadCpuTimer timing(engine_->contribution_cpu_);
            validator_->prepare(work);
            auto const inspected = validator_->inspect(chunk, bytes, count);
            require(inspected.summary.symbols <= multiply(fragment_bound_, count) - entry.symbols,
                "contribution input exceeds its admitted fragment count");
            entry.symbols += inspected.summary.symbols;
        }
        if (!entry.writer) entry.writer = std::make_unique<Writer>(*engine_, entry.run, "contribution", true);
        Packet const packet{uint16_t(frontier_ + member), count, uint32_t(bytes.size())};
        entry.writer->value(packet); entry.writer->write(bytes);
        ++entry.next; entry.failed = false;
    }
    void finish_member(uint16_t member, uint16_t count = 1)
    {
        std::lock_guard lock(mutex_);
        require(engine_ && !sealed_ && member < size(), "invalid contribution member completion");
        auto& entry = members_[member];
        require(entry.owner == member && entry.count == count && !entry.failed && !entry.finished && entry.writer && entry.next == PlotGroupFile::getChunkCountForK(engine_->identity_.header.k),
            "contribution member has missing chunks");
        entry.failed = true; entry.writer->finish(); entry.writer.reset(); entry.finished = true; entry.failed = false;
    }
    // Only the controller of these members may restart them, after the prior
    // producer has stopped. Restore their original singleton reservations so
    // a refused batch can retry intact or split into new smaller tasks.
    void reset_member(uint16_t member, uint16_t count = 1)
    {
        std::lock_guard lock(mutex_);
        require(engine_ && !sealed_ && member < size() && count && count <= size() - member,
            "invalid contribution member reset");
        auto const& entry = members_[member];
        require(entry.owner == member && (!entry.count || entry.count == count),
            "contribution reset changes another batch");
        for (uint16_t i = member; i < member + count; ++i)
            require(entry.count ? members_[i].owner == member :
                members_[i].owner == i && !members_[i].count, "contribution reset overlaps another batch");
        for (uint16_t i = member; i < member + count; ++i)
        {
            auto& slot = members_[i]; slot.writer.reset();
            engine_->discard_input(slot.run, !slot.finished);
            slot = {}; slot.owner = i; slot.failed = true;
        }
        for (uint16_t i = member; i < member + count; ++i) reserve_group(i, 1);
    }
    void seal()
    {
        std::lock_guard lock(mutex_);
        require(engine_ && !sealed_ &&
            std::all_of(members_.begin(), members_.end(), [&](auto const& member)
            { auto const& owner = members_.at(member.owner); return owner.count && owner.finished && !owner.failed; }),
            "contribution input is incomplete");
        validator_.reset(); sealed_ = true; grow_failpoint("external-contribution-input-sealed");
    }
private:
    friend class ExternalEngine;
    struct MemberRecord
    {
        Run run;
        std::unique_ptr<Writer> writer;
        uint64_t next = 0, symbols = 0;
        uint16_t owner = 0, count = 0;
        bool finished = false, failed = false;
    };
    ContributionInput(ExternalEngine& engine, uint16_t count, uint64_t bound)
        : engine_(&engine), frontier_(engine.members_), fragment_bound_(bound), members_(count),
          validator_(std::make_unique<ChunkCodec>(engine.identity_.header.k, engine.config_.codec_bytes, engine.config_.cpu_scheduler))
    { for (uint16_t member = 0; member < count; ++member) members_[member].owner = member; }
    void reserve_group(uint16_t member, uint16_t count)
    {
        auto& entry = members_[member]; entry.failed = true;
        for (uint16_t i = member; i < member + count; ++i)
        {
            auto& slot = members_[i]; slot.writer.reset();
            engine_->discard_input(slot.run, !slot.finished);
            slot.next = 0; slot.symbols = 0; slot.finished = false;
        }
        entry.run.id = engine_->next_id_; entry.run.first = uint16_t(frontier_ + member); entry.run.count = count;
        uint64_t const bytes = contribution_input_scratch_bytes(engine_->identity_.header.k, count, fragment_bound_) -
            multiply(PlotGroupFile::getChunkCountForK(engine_->identity_.header.k), sizeof(Summary));
        require(engine_->reserve(entry.run, bytes), "cannot reserve restarted contribution input");
        engine_->next_id_ = entry.run.id + 1; entry.failed = false;
    }
    void claim(uint16_t member, uint16_t count)
    {
        auto& entry = members_[member];
        require(entry.owner == member && (!entry.count || entry.count == count),
            "contribution batch membership changed or overlaps another batch");
        if (entry.count) return;
        for (uint16_t i = member; i < member + count; ++i)
            require(members_[i].owner == i && !members_[i].count,
                "contribution batches overlap");
        entry.count = count;
        for (uint16_t i = member + 1; i < member + count; ++i) members_[i].owner = member;
        if (count > 1) reserve_group(member, count);
    }
    ExternalEngine* engine_;
    uint16_t frontier_;
    uint64_t fragment_bound_;
    std::vector<MemberRecord> members_;
    Run summary_;
    std::unique_ptr<ChunkCodec> validator_;
    mutable std::mutex mutex_;
    bool sealed_ = false;
};

// Private, uncommitted member-major transport. Its two streams contain narrow
// chunk boundaries and the same low-field words as GPU compact transfer. Only
// ordinary contribution runs and summaries enter the durable checkpoint; a
// crash regenerates this input after authenticated orphan cleanup.
class ExternalEngine::PreparedInput
{
public:
    ~PreparedInput() { if (engine_) engine_->release_prepared(); }
    PreparedInput(PreparedInput const&) = delete;
    PreparedInput& operator=(PreparedInput const&) = delete;
    uint16_t size() const { return uint16_t(members_.size()); }
    bool sealed() const { return sealed_; }
    uint64_t written_bytes() const { return sealed_ ? written_bytes_ : boundaries_.bytes + payload_.bytes; }
    uint64_t read_bytes() const { return read_bytes_; }
    uint64_t reserved_bytes() const
    {
        uint64_t bytes = 0;
        for (auto const* run : {&boundaries_, &payload_, &contribution_, &summary_})
            for (auto const& segment : run->segments) bytes = grow_add(bytes, segment.capacity);
        return bytes;
    }
    void begin_member(uint16_t member, uint64_t count)
    {
        require(engine_ && !sealed_ && !active_ && !failed_ && member == finished_ && member < size() && count &&
            count <= fragment_bound_ && count <= UINT32_MAX, "invalid prepared input member declaration");
        current_ = {}; current_.count = count; current_.word_offset = payload_.bytes + payload_writer_->buffered_bytes();
        written_ = words_ = chunk_ = previous_ = packed_word_ = 0; packed_bits_ = 0;
        packed_ = prelude_ = false; active_ = true;
    }
    void append_raw(std::span<uint64_t const> values)
    {
        require(engine_ && active_ && !sealed_ && !failed_ && !packed_ && values.size() <= current_.count - written_,
            "prepared raw input exceeds its declaration");
        failed_ = true;
        unsigned const width = engine_->identity_.header.k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        uint64_t const end = 1ull << (2 * engine_->identity_.header.k), mask = (1ull << width) - 1;
        uint64_t const maximum_delta = uint64_t(72 - engine_->identity_.header.k) * (178ull << (engine_->identity_.header.k - 8));
        while (!values.empty())
        {
            // A full buffer ends the compute slice. A boundary gap can span
            // buffers, so leave the current fragment unconsumed until all of
            // its preceding boundary entries have been emitted.
            if (boundary_writer_->room() < sizeof(uint32_t)) boundary_writer_->flush();
            if (payload_writer_->room() < sizeof(uint64_t)) payload_writer_->flush();
            CodecScheduler::Lease compute(engine_->config_.cpu_scheduler.get());
            while (!values.empty() && boundary_writer_->room() >= sizeof(uint32_t) &&
                   payload_writer_->room() >= sizeof(uint64_t))
            {
                uint64_t const value = values.front();
                require(value < end && (!written_ || value >= previous_), "unsorted/out-of-range prepared input");
                uint64_t const chunk = value >> width;
                if (written_ && chunk == (previous_ >> width) && value - previous_ >= maximum_delta) current_.validated = false;
                if (chunk_ <= chunk)
                { boundary_writer_->value_unflushed(uint32_t(written_)); ++chunk_; continue; }
                uint64_t const low = value & mask;
                packed_word_ |= low << packed_bits_;
                unsigned const remaining = 64 - packed_bits_;
                if (width >= remaining)
                {
                    payload_writer_->value_unflushed(packed_word_); ++words_;
                    packed_word_ = width > remaining ? low >> remaining : 0; packed_bits_ = width - remaining;
                }
                else packed_bits_ += width;
                previous_ = value; ++written_; values = values.subspan(1);
            }
        }
        failed_ = false;
    }
    void append_packed(PackedFragmentTile const& tile)
    {
        require(engine_ && active_ && !sealed_ && !failed_ && !written_, "prepared packed input follows raw fragments");
        failed_ = true;
        auto const chunks = PlotGroupFile::getChunkCountForK(engine_->identity_.header.k);
        if (!prelude_)
        {
            auto const validation = tile.validation;
            require(tile.words.empty() && tile.boundaries.size() == chunks + 1 &&
                validation.checked_count == current_.count && !(validation.flags & ~PackedMemberValidation::known_flags) &&
                !(validation.flags & (PackedMemberValidation::out_of_range | PackedMemberValidation::unsorted)) &&
                tile.boundaries.front() == 0 && tile.boundaries.back() == current_.count,
                "invalid prepared packed validation prelude");
            {
                CodecScheduler::Lease compute(engine_->config_.cpu_scheduler.get());
                uint32_t previous = 0;
                for (auto next : tile.boundaries)
                { require(next >= previous && next <= current_.count, "invalid prepared packed boundaries"); previous = next; }
            }
            boundary_writer_->write({reinterpret_cast<uint8_t const*>(tile.boundaries.data()), tile.boundaries.size_bytes()});
            current_.validated = !(validation.flags & PackedMemberValidation::wide_delta);
            packed_ = prelude_ = true; failed_ = false; return;
        }
        uint64_t const expected = word_count(current_.count);
        require(tile.boundaries.empty() && !tile.validation.checked_count && !tile.validation.flags &&
            tile.words.size() <= expected - words_, "invalid prepared packed payload tile");
        payload_writer_->write({reinterpret_cast<uint8_t const*>(tile.words.data()), tile.words.size_bytes()});
        words_ += tile.words.size(); if (!tile.words.empty()) packed_word_ = tile.words.back();
        failed_ = false;
    }
    void finish_member()
    {
        require(engine_ && active_ && !sealed_ && !failed_, "prepared input has no active member");
        failed_ = true;
        if (packed_)
        {
            require(prelude_ && words_ == word_count(current_.count), "incomplete prepared packed member");
            unsigned const tail = unsigned((current_.count *
                (engine_->identity_.header.k + PlotGroupFile::PROOFS_PER_CHUNK_BITS)) & 63);
            require(!tail || !(packed_word_ >> tail), "prepared packed tail is not zero padded");
        }
        else
        {
            require(written_ == current_.count, "incomplete prepared raw member");
            auto const chunks = PlotGroupFile::getChunkCountForK(engine_->identity_.header.k);
            while (chunk_ <= chunks)
            {
                if (boundary_writer_->room() < sizeof(uint32_t)) boundary_writer_->flush();
                CodecScheduler::Lease compute(engine_->config_.cpu_scheduler.get());
                while (chunk_ <= chunks && boundary_writer_->room() >= sizeof(uint32_t))
                { boundary_writer_->value_unflushed(uint32_t(written_)); ++chunk_; }
            }
            if (packed_bits_) { payload_writer_->value(packed_word_); ++words_; }
            require(words_ == word_count(current_.count), "prepared raw word count differs");
        }
        members_[finished_++] = current_; active_ = false; failed_ = false;
    }
    void seal_prefix()
    {
        require(engine_ && !sealed_ && !active_ && !failed_ && finished_, "prepared prefix is not complete");
        members_.resize(finished_); seal();
    }
    void seal()
    {
        require(engine_ && !sealed_ && !active_ && !failed_ && finished_ == size(), "incomplete prepared batch");
        boundary_writer_->finish(); payload_writer_->finish();
        written_bytes_ = boundaries_.bytes + payload_.bytes;
        boundary_writer_.reset(); payload_writer_.reset(); sealed_ = true;
        grow_failpoint("external-input-sealed");
    }
private:
    friend class ExternalEngine;
    friend class InputReader;
    struct MemberRecord { uint64_t count = 0, word_offset = 0; bool validated = true; };
    PreparedInput(ExternalEngine& engine, uint16_t members, uint64_t fragment_bound)
        : engine_(&engine), frontier_(engine.members_), fragment_bound_(fragment_bound), members_(members) {}
    uint64_t word_count(uint64_t count) const
    {
        uint64_t const bits = multiply(count, engine_->identity_.header.k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        return bits / 64 + (bits % 64 != 0);
    }
    ExternalEngine* engine_;
    uint16_t frontier_, finished_ = 0;
    uint64_t fragment_bound_, read_bytes_ = 0, written_bytes_ = 0;
    std::vector<MemberRecord> members_;
    Run boundaries_, payload_, contribution_, summary_;
    std::unique_ptr<Writer> boundary_writer_, payload_writer_;
    MemberRecord current_;
    uint64_t written_ = 0, words_ = 0, chunk_ = 0, previous_ = 0, packed_word_ = 0;
    unsigned packed_bits_ = 0;
    bool sealed_ = false, active_ = false, packed_ = false, prelude_ = false, failed_ = false, contribution_started_ = false;
};
inline uint64_t ExternalEngine::prepared_segments() const
{
    uint64_t count = 0;
    if (prepared_) for (auto const* run : {&prepared_->boundaries_, &prepared_->payload_, &prepared_->contribution_, &prepared_->summary_})
        count = grow_add(count, run->segments.size());
    if (contributions_)
    {
        for (auto const& member : contributions_->members_) count = grow_add(count, member.run.segments.size());
        count = grow_add(count, contributions_->summary_.segments.size());
    }
    return count;
}
inline void ExternalEngine::discard_input(Run& run, bool refresh)
{
    if (run.segments.empty()) return;
    if (!refresh) { discard(run); run = {}; return; }
    // Writers can fail before finish updates timestamps. Verify the exclusive
    // inode and ownership header before refreshing our own provisional stamp.
    for (auto& segment : run.segments)
    {
        auto const path = segment_path(run, segment); auto const current = stamp(path);
        require(current.file == segment.stamp.file, "prepared input segment was replaced");
        external_detail::Descriptor file(file_io::open_file(path.c_str(), file_io::read_only | file_io::no_follow | file_io::close_on_exec));
        require(file_identity(file.fd) == current.file, "prepared input handle was replaced");
        std::array<uint8_t, segment_header_bytes> header{};
        external_detail::read(file.fd, 0, header, &stats_, {"compact_member", &path_});
        require(header == segment_header(run, segment), "prepared input ownership changed"); segment.stamp = current;
    }
    discard(run); run = {};
}
inline void ExternalEngine::release_prepared() noexcept
{
    auto* input = std::exchange(prepared_, nullptr); if (!input) return;
    input->boundary_writer_.reset(); input->payload_writer_.reset();
    try
    {
        discard_input(input->boundaries_, !input->sealed_); discard_input(input->payload_, !input->sealed_);
        discard_input(input->contribution_, input->contribution_started_); discard_input(input->summary_);
    }
    catch (...) { confirmed_ = false; input_cleanup_failed_ = true; } // Retain charged files for resume.
    input->engine_ = nullptr;
}
inline void ExternalEngine::release_contributions() noexcept
{
    auto* input = std::exchange(contributions_, nullptr); if (!input) return;
    input->validator_.reset();
    for (auto& member : input->members_) member.writer.reset();
    try
    {
        for (auto& member : input->members_) discard_input(member.run, !member.finished);
        discard_input(input->summary_);
    }
    catch (...) { confirmed_ = false; input_cleanup_failed_ = true; }
    input->engine_ = nullptr;
}
inline uint64_t ExternalEngine::contribution_input_memory_bytes(Config const& config, uint16_t members)
{
    require(members && members <= max_runs, "invalid contribution input membership");
    // Codec scratch and the append's candidate summary use the existing
    // exclusive engine workspace. Concurrent producers additionally retain
    // one bounded writer each; their transport frames belong to the caller.
    return grow_add(sizeof(ContributionInput) + sizeof(ChunkCodec),
        multiply(members, grow_add(config.window_bytes, sizeof(ContributionInput::MemberRecord) + sizeof(Writer))));
}
inline uint64_t ExternalEngine::contribution_input_scratch_bytes(uint8_t k, uint16_t members, uint64_t fragment_bound)
{
    require(k >= 18 && k <= 28 && !(k & 1) && members && members <= max_runs && fragment_bound && fragment_bound <= UINT32_MAX,
        "invalid contribution input bound");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
    uint64_t const member = grow_add(grow_add(multiply(fragment_bound, 9), fragment_bound / 128),
        multiply(chunks, POS2_FSE_compressBound(0) + sizeof(Packet) + 11));
    return grow_add(multiply(members, member), multiply(chunks, sizeof(Summary)));
}
inline std::unique_ptr<ExternalEngine::ContributionInput> ExternalEngine::prepare_contributions(uint16_t members, uint64_t fragment_bound)
{
    require(confirmed_ && can_append() && !prepared_ && !contributions_ && members && members <= UINT16_MAX - members_,
        "contribution input requires an idle confirmed staging engine"); check_paths();
    if (members > max_runs || !fragment_bound || fragment_bound > UINT32_MAX) return {};
    require(grow_add(required_memory(members), contribution_input_memory_bytes(config_, members)) <= config_.memory_limit_bytes,
        "contribution input exceeds memory budget");
    // Reserve all future run slots before generation. Compaction remains the
    // existing adjacent-run operation; no receive stream can mutate its inputs.
    while (runs_.size() + members > max_runs)
    {
        size_t const count = external_.fan_in;
        if (runs_.size() < count) return {};
        size_t first = 0; uint64_t least = UINT64_MAX;
        for (size_t begin = 0; begin + count <= runs_.size(); ++begin)
        {
            uint64_t bytes = 0;
            for (size_t i = begin; i < begin + count; ++i) bytes = grow_add(bytes, runs_[i].bytes);
            if (bytes <= least) { least = bytes; first = begin; }
        }
        if (!merge_range(first, count)) return {};
    }
    auto input = std::unique_ptr<ContributionInput>(new ContributionInput(*this, members, fragment_bound));
    contributions_ = input.get();
    uint64_t const summary_bytes = multiply(PlotGroupFile::getChunkCountForK(identity_.header.k), sizeof(Summary));
    uint64_t const member_bytes = (contribution_input_scratch_bytes(identity_.header.k, members, fragment_bound) - summary_bytes) / members;
    for (uint16_t member = 0; member < members; ++member)
    {
        auto& run = input->members_[member].run;
        run.id = next_id_; run.first = uint16_t(members_ + member); run.count = 1;
        if (!reserve(run, member_bytes)) return {};
        next_id_ = run.id + 1;
    }
    input->summary_.id = next_id_; input->summary_.count = uint16_t(members_ + members);
    if (!reserve(input->summary_, summary_bytes, "checkpoint_summary")) return {};
    next_id_ = input->summary_.id + 1;
    return input;
}
inline uint64_t ExternalEngine::input_scratch_bytes(uint8_t k, uint16_t members, uint64_t fragment_bound)
{
    require(k >= 18 && k <= 28 && !(k & 1) && members && fragment_bound && fragment_bound <= UINT32_MAX,
        "invalid prepared input bound");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k), symbols = multiply(members, fragment_bound);
    uint64_t const bits = multiply(fragment_bound, k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    uint64_t const compact = multiply(members, grow_add(multiply(chunks + 1, sizeof(uint32_t)),
        multiply(bits / 64 + (bits % 64 != 0), sizeof(uint64_t))));
    // Every valid delta has at most eight low bytes and one ANS symbol.
    // FSE's per-chunk bound is n + (n >> 7) + 512; include prefix, packet and
    // rounding per chunk. This preflight may decline in favor of exact ordinary
    // append, but can never consume an unreserved generated member.
    uint64_t const contribution = grow_add(grow_add(multiply(symbols, 9), symbols / 128),
        multiply(chunks, grow_add(POS2_FSE_compressBound(0), sizeof(Packet) + 11)));
    return grow_add(compact, grow_add(contribution, multiply(chunks, sizeof(Summary))));
}
inline bool ExternalEngine::staging_slot()
{
    if (runs_.size() < max_runs) return true;
    size_t const count = external_.fan_in; size_t first = 0; uint64_t least = UINT64_MAX;
    for (size_t begin = 0; begin + count <= runs_.size(); ++begin)
    {
        uint64_t bytes = 0;
        for (size_t i = begin; i < begin + count; ++i) bytes = grow_add(bytes, runs_[i].bytes);
        if (bytes <= least) { least = bytes; first = begin; }
    }
    return merge_range(first, count);
}
inline std::unique_ptr<ExternalEngine::PreparedInput> ExternalEngine::prepare_input(uint16_t members, uint64_t fragment_bound)
{
    require(confirmed_ && can_append() && !prepared_ && !contributions_ && members && members <= UINT16_MAX - members_,
        "prepared input requires an idle confirmed staging engine"); check_paths();
    require(grow_add(required_memory(members), input_memory_bytes(config_, members)) <= config_.memory_limit_bytes,
        "prepared input exceeds memory budget");
    if (!fragment_bound || fragment_bound > UINT32_MAX || !staging_slot()) return {};
    auto input = std::unique_ptr<PreparedInput>(new PreparedInput(*this, members, fragment_bound)); prepared_ = input.get();
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(identity_.header.k);
    uint64_t const boundary_bytes = multiply(members, multiply(chunks + 1, sizeof(uint32_t)));
    uint64_t const payload_bytes = multiply(members, multiply(input->word_count(fragment_bound), sizeof(uint64_t)));
    uint64_t const total = input_scratch_bytes(identity_.header.k, members, fragment_bound);
    uint64_t const summary_bytes = multiply(chunks, sizeof(Summary));
    auto allocate = [&](Run& run, uint64_t bytes, char const* role)
    {
        run.id = next_id_; run.first = members_; run.count = members;
        if (!reserve(run, bytes, role)) return false;
        next_id_ = run.id + 1; return true;
    };
    if (!allocate(input->boundaries_, boundary_bytes, "compact_member") ||
        !allocate(input->payload_, payload_bytes, "compact_member") ||
        !allocate(input->contribution_, total - boundary_bytes - payload_bytes - summary_bytes, "contribution") ||
        !allocate(input->summary_, summary_bytes, "checkpoint_summary")) return {};
    input->boundary_writer_ = std::make_unique<Writer>(*this, input->boundaries_, "compact_member", true);
    input->payload_writer_ = std::make_unique<Writer>(*this, input->payload_, "compact_member", true);
    return input;
}
class ExternalEngine::InputReader
{
public:
    InputReader(ExternalEngine& engine, PreparedInput& input, uint16_t count)
        : engine_(engine), input_(input), count_(count)
    {
        cursors_.reserve(count);
        auto const bytes = multiply(PlotGroupFile::getChunkCountForK(engine.identity_.header.k) + 1, sizeof(uint32_t));
        for (uint16_t member = 0; member < count; ++member)
        {
            Cursor cursor;
            cursor.boundaries = std::make_unique<Reader>(engine, input.boundaries_, size_t(count) * 2, multiply(member, bytes), "compact_member", &input.read_bytes_);
            cursor.payload = std::make_unique<Reader>(engine, input.payload_, size_t(count) * 2, input.members_[member].word_offset, "compact_member", &input.read_bytes_);
            read(*cursor.boundaries, cursor.first); require(!cursor.first, "prepared boundaries do not start at zero");
            cursors_.push_back(std::move(cursor));
        }
    }
    uint64_t next_words()
    {
        if (lookahead_) return next_words_;
        unsigned const width = engine_.identity_.header.k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        next_words_ = 0;
        for (uint16_t member = 0; member < count_; ++member)
        {
            auto& cursor = cursors_[member]; read(*cursor.boundaries, cursor.last);
            require(cursor.last > cursor.first && cursor.last <= input_.members_[member].count,
                "empty or invalid prepared member chunk");
            next_words_ = grow_add(next_words_, (multiply(cursor.last, width) + 63) / 64 - multiply(cursor.first, width) / 64);
        }
        lookahead_ = true; return next_words_;
    }
    void regions(uint64_t chunk, std::vector<FragmentRegion>& regions, std::vector<uint64_t>& words)
    {
        require(chunk == next_++, "prepared chunks must be read in order");
        if (next_words() > words.capacity() - words.size()) throw InputWindowExceeded();
        lookahead_ = false;
        auto const k = engine_.identity_.header.k;
        unsigned const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        for (uint16_t member = 0; member < count_; ++member)
        {
            auto& cursor = cursors_[member]; uint32_t const last = cursor.last;
            auto const& record = input_.members_[member];
            require(last > cursor.first && last <= record.count, "empty or invalid prepared member chunk");
            uint64_t const first_bit = multiply(cursor.first, width), last_bit = multiply(last, width);
            size_t const count = size_t((last_bit + 63) / 64 - first_bit / 64), offset = words.size();
            if (count > words.capacity() - words.size()) throw InputWindowExceeded();
            unsigned const shift = unsigned(first_bit & 63);
            if (shift) words.push_back(cursor.carry);
            auto const begin = words.size(); words.resize(offset + count);
            auto const fresh = std::span(words).subspan(begin);
            cursor.payload->read({reinterpret_cast<uint8_t*>(fresh.data()), fresh.size_bytes()});
            if (last_bit & 63) cursor.carry = words.back();
            regions.push_back(FragmentRegion::packed(std::span<uint64_t const>(words).subspan(offset, count), k,
                chunk, shift, last - cursor.first, record.validated));
            cursor.first = last;
            if (chunk + 1 == PlotGroupFile::getChunkCountForK(k)) require(last == record.count, "prepared boundary endpoint differs");
        }
    }
private:
    friend class ExternalEngine;
    struct Cursor { std::unique_ptr<Reader> boundaries, payload; uint32_t first = 0, last = 0; uint64_t carry = 0; };
    template<class T> void read(Reader& reader, T& value)
    { reader.read({reinterpret_cast<uint8_t*>(&value), sizeof(value)}); }
    ExternalEngine& engine_; PreparedInput& input_; uint16_t count_; uint64_t next_ = 0, next_words_ = 0; bool lookahead_ = false;
    std::vector<Cursor> cursors_;
};
inline uint64_t ExternalEngine::input_memory_bytes(Config const& config, uint16_t members)
{
    // Preparation owns two writers. Append replaces them with aggregate reader
    // buffers and packed storage for both codec windows. The minimum packet
    // buffer can exceed the aggregate division for a large member count.
    uint64_t const per_member = sizeof(PreparedInput::MemberRecord) + sizeof(InputReader::Cursor) +
        2 * (sizeof(Reader) + sizeof(external_detail::Descriptor) + sizeof(Packet));
    return grow_add(multiply(config.window_bytes, 3), grow_add(multiply(members, per_member),
        sizeof(PreparedInput) + sizeof(InputReader) + 2 * sizeof(Writer)));
}
inline void ExternalEngine::incoming_regions(Members members, uint64_t chunk, std::vector<size_t>& cursors,
                                            std::vector<FragmentRegion>& regions) const
{
    uint64_t const end = (chunk + 1) << (identity_.header.k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t i = 0; i < members.size(); ++i)
    {
        auto const member = members[i];
        if (member.compact()) regions[i] = member.region(chunk);
        else
        {
            auto raw = member.raw(); auto const first = cursors[i]; auto next = first;
            while (next < raw.size() && raw[next] < end) ++next;
            regions[i] = member.region(chunk, first, next - first); cursors[i] = next;
        }
    }
}
inline AppendResult ExternalEngine::append(Members members)
{
    require(!prepared_ && !contributions_, "ordinary append cannot overlap a prepared input");
    require(!members.empty() && members.size() <= size_t(UINT16_MAX - members_), "invalid external batch membership");
    return append_source(members, nullptr, uint16_t(members.size()));
}
inline AppendResult ExternalEngine::append(PreparedInput& input, uint16_t prefix)
{
    require(prepared_ == &input && input.engine_ == this && input.sealed_ && input.frontier_ == members_,
        "prepared input owner/frontier differs");
    uint16_t const count = prefix ? prefix : input.size();
    require(count && count <= input.size(), "invalid prepared input prefix");
    try { return append_source({}, &input, count); }
    catch (InputWindowExceeded const&)
    {
        return {false, false, final_bound_, grow_add(final_bound_, metadata_bytes(identity_.header.k, external_)),
            "prepared final chunk exceeds input window budget", {}};
    }
}
inline AppendResult ExternalEngine::append(ContributionInput& input, uint16_t prefix)
{
    std::lock_guard lock(input.mutex_);
    require(contributions_ == &input && input.engine_ == this && input.sealed_ &&
        input.frontier_ == members_ && confirmed_ && can_append(), "contribution input owner/frontier differs");
    uint16_t const count = prefix ? prefix : input.size();
    require(count && count <= input.size() && count <= UINT16_MAX - members_, "invalid contribution input prefix");
    uint16_t records = 0;
    for (uint16_t member = 0; member < count; member += input.members_[member].count)
    {
        auto const& entry = input.members_[member];
        require(entry.owner == member && entry.count && entry.finished, "invalid sealed contribution batch");
        // A remote batch is one canonical run. Preserve its source until
        // acceptance; the caller can regenerate an exact fitting member via
        // the existing ordinary fallback instead of reserving a second run.
        if (entry.count > count - member)
            return {false, false, final_bound_, grow_add(final_bound_, metadata_bytes(identity_.header.k, external_)),
                "received prefix crosses a compute batch; regenerate its fitting members", {}};
        ++records;
    }
    require(grow_add(required_memory(count), contribution_input_memory_bytes(config_, input.size())) <= config_.memory_limit_bytes,
        "contribution append exceeds memory budget");
    check_paths(); ThreadCpuTimer cpu(contribution_owner_cpu_);
    PhaseTimer planning(stats_, WorkPhase::plan);
    uint16_t const after = uint16_t(members_ + count);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(identity_.header.k);
    std::vector<Summary> summaries(size_t(chunks), Summary{});
    if (members_) read_summary(roots_, token_, summary_run_, summary_digest_,
        {reinterpret_cast<uint8_t*>(summaries.data()), summaries.size() * sizeof(Summary)}, &stats_,
        {"checkpoint_summary", &path_, &execution_scratch_}, config_.cpu_scheduler.get());
    auto codec = std::make_unique<ChunkCodec>(identity_.header.k, config_.codec_bytes, config_.cpu_scheduler);
    std::vector<uint8_t> payload; payload.reserve(size_t(config_.window_bytes));
    std::vector<EncodedChunk> parts; parts.reserve(UINT16_MAX);
    std::vector<Run const*> sources;
    std::vector<std::unique_ptr<Reader>> readers;
    auto rewind = [&](bool all)
    {
        readers.clear(); sources.clear();
        if (all) for (auto const& run : runs_) sources.push_back(&run);
        for (uint16_t member = 0; member < count; member += input.members_[member].count)
            sources.push_back(&input.members_[member].run);
        for (auto const* run : sources) readers.push_back(std::make_unique<Reader>(*this, *run, sources.size()));
    };
    auto read_chunk = [&]()
    {
        payload.clear(); parts.clear();
        for (size_t source = 0; source < sources.size(); ++source)
        {
            auto const& run = *sources[source];
            for (uint32_t member = run.first; member < uint32_t(run.first) + run.count;)
            {
                auto const packet = readers[source]->packet(uint16_t(member));
                if (packet.bytes > config_.window_bytes - payload.size()) return false;
                require(parts.size() < parts.capacity(), "contribution append descriptor budget exceeded");
                size_t const offset = payload.size(); payload.resize(offset + packet.bytes);
                auto const bytes = std::span(payload).subspan(offset, packet.bytes); readers[source]->read(bytes);
                parts.push_back({bytes, packet.count}); member += packet.count;
            }
        }
        return true;
    };
    uint64_t final_bound = sizeof(Header) + identity_.memo.size();
    bool fits = true;
    rewind(false);
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        if (!read_chunk()) { fits = false; break; }
        auto& summary = summaries[size_t(chunk)];
        for (auto const& part : parts)
        {
            uint64_t const work = ChunkCodec::estimate_work(identity_.header.k, part.bytes.size(), 0);
            if (work > config_.codec_bytes) { fits = false; break; }
            codec->prepare(work);
            auto const inspected = codec->inspect(chunk, part.bytes, part.members);
            summary.codec = ChunkCodec::combine(identity_.header.k, summary.codec, inspected.summary, inspected.first, chunk);
            summary.bytes = grow_add(summary.bytes, part.bytes.size());
        }
        if (!fits) break;
        auto const sizes = ChunkCodec::size_bounds(summary.codec);
        final_bound = grow_add(final_bound, sizes.second);
        try { check_gsz_size(sizes.first, after); check_gsz_size(sizes.second, after); }
        catch (UnencodableChunkSize const&) { fits = false; }
        if (sizes.second > config_.window_bytes || summary.bytes > config_.window_bytes ||
            ChunkCodec::estimate_work(identity_.header.k, summary.bytes, 0) > config_.codec_bytes) fits = false;
    }
    final_bound = grow_add(final_bound, multiply(chunks, sizeof(uint64_t)));
    uint64_t peak = grow_add(final_bound, metadata_bytes(identity_.header.k, external_));
    if (!fits || peak > config_.capacity_bytes)
    {
        // Exact sizing can rescue a conservative bound, including the final
        // partial group. Inspect again while gathering the complete canonical
        // chunk so even a failed first scan leaves a complete candidate table.
        rewind(true); final_bound = sizeof(Header) + identity_.memo.size();
        uint64_t index_bits = 0; BitWriter field; fits = true;
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
        {
            if (!read_chunk()) { fits = false; break; }
            uint64_t const work = ChunkCodec::estimate_work(identity_.header.k, payload.size(), 0);
            if (work > config_.codec_bytes) { fits = false; break; }
            codec->prepare(work);
            uint64_t bytes = 0;
            try { bytes = codec->joined_size(chunk, parts); check_gsz_size(bytes, after); }
            catch (UnencodableChunkSize const&) { fits = false; break; }
            ++stats_.sized_chunks;
            if (bytes > config_.window_bytes) { fits = false; break; }
            summaries[size_t(chunk)] = {codec->summary(chunk, after), payload.size()};
            final_bound = grow_add(final_bound, bytes);
            field.clear(); gsz_encode(std::span<uint64_t const>(&bytes, 1), after, field);
            index_bits = grow_add(index_bits, field.bitCount());
        }
        final_bound = grow_add(final_bound, grow_add(index_bits, 7) / 8);
        peak = grow_add(final_bound, metadata_bytes(identity_.header.k, external_));
        if (!fits || peak > config_.capacity_bytes)
            return {false, false, final_bound, peak, "received contributions exceed exact capacity or codec/window budget", {}};
    }
    for (auto const& reader : readers) require(!reader->remaining(), "contribution input has trailing packets");
    readers.clear(); sources.clear(); codec.reset();
    std::vector<uint8_t>().swap(payload); std::vector<EncodedChunk>().swap(parts);
    require(pending_summary_.segments.empty() && runs_.size() + count <= max_runs,
        "contribution append lost its reserved run slots");
    planning.finish(); PhaseTimer encoding(stats_, WorkPhase::encode);
    pending_summary_ = std::move(input.summary_); input.summary_ = {}; pending_summary_.count = after;
    contribution_bytes_ = 0;
    for (uint16_t member = 0; member < count; member += input.members_[member].count)
    {
        auto& run = input.members_[member].run;
        contribution_bytes_ = grow_add(contribution_bytes_, run.bytes);
        runs_.push_back(std::move(run)); run = {};
    }
    members_ = after; final_bound_ = final_bound; packet_count_ = grow_add(packet_count_, records);
    for (auto const& summary : summaries) packing_source_bytes_ = std::max(packing_source_bytes_, summary.bytes);
    packing_work_ = ChunkCodec::estimate_work(identity_.header.k, packing_source_bytes_, 0);
    identity_.pending = true; identity_.target_members = members_; identity_.pending_peak_bytes = peak;
    save("external-run", summaries); std::vector<Summary>().swap(summaries);
    release_contributions(); encoding.finish(); balance();
    return {true, true, final_bound_, peak, {}, {final_bound_, peak, peak - final_bound_, members_}};
}
inline AppendResult ExternalEngine::append_source(Members members, PreparedInput* input, uint16_t count)
{
    ThreadCpuTimer cpu(contribution_owner_cpu_);
    require(confirmed_ && can_append(), "external append requires confirmed staging phase"); check_paths();
    require(count && count <= UINT16_MAX - members_, "invalid external batch membership");
    require(required_memory(uint16_t(count)) <= config_.memory_limit_bytes, "external batch exceeds memory budget");
    for (size_t i = 0; !input && i < count; ++i)
    {
        CodecScheduler::Lease compute(config_.cpu_scheduler.get());
        auto const member = members[i]; require(!member.empty() && (!member.compact() || member.k() == identity_.header.k), "invalid external member");
        if (!member.compact()) require(std::is_sorted(member.raw().begin(), member.raw().end()) &&
            member.raw().back() < (1ull << (identity_.header.k * 2)), "unsorted/out-of-range external member");
    }
    if (!staging_slot())
        return {false, false, final_bound_, grow_add(final_bound_, metadata_bytes(identity_.header.k, external_)),
            "external run limit needs more scratch; finish the committed prefix", {}};
    PhaseTimer planning(stats_, WorkPhase::plan);
    uint16_t const after = uint16_t(members_ + count);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(identity_.header.k);
    std::vector<Summary> summaries(size_t(chunks), Summary{});
    if (members_) read_summary(roots_, token_, summary_run_, summary_digest_,
        {reinterpret_cast<uint8_t*>(summaries.data()), summaries.size() * sizeof(Summary)}, &stats_, {"checkpoint_summary", &path_, &execution_scratch_}, config_.cpu_scheduler.get());
    std::vector<size_t> cursors(count, 0);
    std::vector<FragmentRegion> regions(count);
    std::unique_ptr<InputReader> input_reader;
    std::vector<uint64_t> input_words;
    if (input) input_words.reserve(size_t(config_.window_bytes / sizeof(uint64_t)));
    auto rewind = [&]
    {
        std::fill(cursors.begin(), cursors.end(), 0);
        if (input) { input_reader.reset(); input_reader = std::make_unique<InputReader>(*this, *input, count); }
    };
    auto read_regions = [&](uint64_t chunk, std::vector<FragmentRegion>& fresh, std::vector<uint64_t>& words)
    {
        if (input) { fresh.clear(); input_reader->regions(chunk, fresh, words); }
        else incoming_regions(members, chunk, cursors, fresh);
    };
    rewind();
    uint64_t maximum_input_words = 0;
    uint64_t contribution_bound = 0, final_bound = sizeof(Header) + identity_.memo.size();
    uint64_t maximum_incoming_work = ChunkCodec::estimate_base();
    bool fits = true; std::string reason;
    auto reduce = [&](uint64_t chunk, MemberRegions fresh_regions, ChunkSummary const& fresh)
    {
        uint64_t incoming_symbols = 0;
        for (size_t member = 0; member < fresh_regions.size(); ++member)
            incoming_symbols = grow_add(incoming_symbols, fresh_regions[member].size());
        maximum_incoming_work = std::max(maximum_incoming_work, ChunkCodec::estimate_work(identity_.header.k, 0, incoming_symbols));
        auto const bounded = std::min(ChunkCodec::fresh_bound(identity_.header.k, chunk, fresh_regions, config_.cpu_scheduler.get()), ChunkCodec::size_bounds(fresh).second);
        require(bounded <= UINT32_MAX, "external contribution packet exceeds its size field");
        contribution_bound = grow_add(contribution_bound, grow_add(sizeof(Packet), bounded));
        auto& summary = summaries[size_t(chunk)];
        summary.codec = ChunkCodec::combine(identity_.header.k, summary.codec, fresh, fresh_regions[0].front(), chunk);
        // Keep prior actual source bytes until encoding commits this chunk's
        // new contribution. Exact sizing may retry with a smaller reservation.
        uint64_t const source_bytes = grow_add(summary.bytes, bounded);
        auto const sizes = ChunkCodec::size_bounds(summary.codec);
        final_bound = grow_add(final_bound, sizes.second);
        try { check_gsz_size(sizes.first, after); check_gsz_size(sizes.second, after); }
        catch (UnencodableChunkSize const&) { fits = false; reason = "external canonical chunk index exceeds conservative admission"; }
        if (sizes.second > config_.window_bytes || source_bytes > config_.window_bytes || ChunkCodec::estimate_work(identity_.header.k, source_bytes, 0) > config_.codec_bytes ||
            ChunkCodec::estimate_work(identity_.header.k, 0, incoming_symbols) > config_.codec_bytes)
        { fits = false; reason = "external final chunk exceeds codec/window budget"; }
    };
    bool immutable = !input;
    for (size_t member = 0; immutable && member < count; ++member)
        immutable = immutable && members[member].compact() && members[member].immutable_id();
    uint64_t const summary_metadata = grow_add(sizeof(ChunkInput) + sizeof(ChunkSummary) + sizeof(EncodedBlock) +
        3 * sizeof(uint64_t), multiply(count, sizeof(FragmentRegion)));
    size_t const summary_limit = size_t(std::min({chunks, uint64_t(65536),
        std::max<uint64_t>(1, config_.window_bytes / summary_metadata)}));
    unsigned const summary_workers = unsigned(std::min({uint64_t(config_.workers), uint64_t(summary_limit),
        config_.codec_bytes / ChunkCodec::estimate_base()}));
    if ((immutable || input) && summary_workers > 1)
    {
        // Scan independent immutable chunks on the existing codec pool, then
        // reduce their bounded summaries in order. Release the planning pool
        // and descriptors before exact sizing or contribution encoding.
        ParallelEncoder encoder(identity_.header.k, summary_workers, config_.codec_bytes, config_.cpu_scheduler);
        std::vector<FragmentRegion> window_regions; window_regions.reserve(size_t(multiply(summary_limit, count)));
        std::vector<ChunkInput> tasks; tasks.reserve(summary_limit);
        std::vector<ChunkSummary> details; details.reserve(summary_limit);
        for (size_t first = 0; first < summaries.size();)
        {
            size_t const limit = std::min(summary_limit, summaries.size() - first);
            tasks.clear(); window_regions.clear(); input_words.clear();
            for (size_t i = 0; i < limit; ++i)
            {
                if (input && input_reader->next_words() > input_words.capacity() - input_words.size())
                {
                    if (tasks.empty()) throw InputWindowExceeded();
                    break;
                }
                size_t const begin = window_regions.size(), words = input_words.size();
                read_regions(first + i, regions, input_words);
                maximum_input_words = std::max<uint64_t>(maximum_input_words, input_words.size() - words);
                window_regions.insert(window_regions.end(), regions.begin(), regions.end());
                tasks.push_back({first + i, {}, 0,
                    std::span<FragmentRegion const>(window_regions).subspan(begin, count)});
            }
            details.resize(tasks.size()); encoder.summarize(tasks, details);
            if (tasks.size() > summary_workers) grow_failpoint("external-summary-window");
            for (size_t i = 0; i < tasks.size(); ++i) reduce(first + i, tasks[i].new_members, details[i]);
            first += tasks.size();
        }
        contribution_cpu_.add(encoder.cpu_time());
    }
    else for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        input_words.clear(); read_regions(chunk, regions, input_words);
        maximum_input_words = std::max<uint64_t>(maximum_input_words, input_words.size());
        reduce(chunk, regions, ChunkCodec::fresh_summary(identity_.header.k, chunk, regions, config_.cpu_scheduler.get()));
    }
    final_bound = grow_add(final_bound, multiply(chunks, 8));
    uint64_t peak = grow_add(final_bound, metadata_bytes(identity_.header.k, external_));
    bool contribution_sized = false;
    auto exact = [&](bool final)
    {
        contribution_sized = false; rewind();
        ChunkCodec codec(identity_.header.k, config_.codec_bytes, config_.cpu_scheduler);
        std::vector<std::unique_ptr<Reader>> readers;
        if (final) for (auto const& run : runs_) readers.push_back(std::make_unique<Reader>(*this, run));
        std::vector<uint8_t> payload; payload.reserve(size_t(config_.window_bytes));
        std::vector<EncodedChunk> contributions; contributions.reserve(UINT16_MAX);
        BitWriter field;
        uint64_t final_bytes = sizeof(Header) + identity_.memo.size(), index_bits = 0;
        contribution_bound = 0; bool buffers = true, encodable = true;
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
        {
            input_words.clear(); read_regions(chunk, regions, input_words);
            uint64_t symbols = 0; for (auto const& region : regions) symbols = grow_add(symbols, region.size());
            payload.clear(); contributions.clear();
            if (final) for (size_t i = 0; i < runs_.size(); ++i)
            {
                auto const& run = runs_[i];
                for (uint32_t member = run.first; member < uint32_t(run.first) + run.count;)
                {
                    auto const packet = readers[i]->packet(uint16_t(member));
                    require(packet.bytes <= config_.window_bytes - payload.size(), "external sizing source exceeds window budget");
                    size_t const first = payload.size(); payload.resize(first + packet.bytes);
                    auto const part = std::span(payload).subspan(first, packet.bytes); readers[i]->read(part);
                    contributions.push_back({part, packet.count}); member += packet.count;
                }
            }
            uint64_t const work = ChunkCodec::estimate_work(identity_.header.k, payload.size(), symbols);
            if (work > config_.codec_bytes) { buffers = false; break; }
            codec.prepare(work); auto const begin = std::chrono::steady_clock::now();
            uint64_t final_size = 0, bytes = 0;
            try
            {
                if (final)
                {
                    final_size = contributions.empty() ? codec.encoded_size(chunk, {}, 0, regions) :
                        codec.joined_size(chunk, contributions, regions);
                    check_gsz_size(final_size, after);
                    if (final_size > config_.window_bytes) buffers = false; ++stats_.sized_chunks;
                }
                bytes = codec.encoded_size(chunk, {}, 0, regions); ++stats_.sized_chunks;
            }
            catch (UnencodableChunkSize const&) { encodable = false; break; }
            stats_.codec_ns += elapsed_ns(begin);
            if (final)
            {
                final_bytes = grow_add(final_bytes, final_size);
                field.clear(); gsz_encode(std::span<uint64_t const>(&final_size, 1), after, field);
                index_bits = grow_add(index_bits, field.bitCount());
            }
            uint64_t const source_bytes = grow_add(summaries[size_t(chunk)].bytes, bytes);
            if (source_bytes > config_.window_bytes ||
                ChunkCodec::estimate_work(identity_.header.k, source_bytes, 0) > config_.codec_bytes)
                buffers = false;
            contribution_bound = grow_add(contribution_bound, grow_add(sizeof(Packet), bytes));
        }
        if (final && buffers && encodable)
        {
            final_bound = grow_add(final_bytes, grow_add(index_bits, 7) / 8);
            peak = grow_add(final_bound, metadata_bytes(identity_.header.k, external_));
        }
        contribution_sized = buffers && encodable; return contribution_sized;
    };
    if (!fits || peak > config_.capacity_bytes)
    {
        fits = exact(true) && peak <= config_.capacity_bytes;
        if (!fits) return {false, false, final_bound, peak,
            reason.empty() ? "external exact output and checkpoint reserve exceed capacity" : reason, {}};
    }
    require(pending_summary_.segments.empty(), "external summary generation is already pending");
    Run incoming_storage;
    Run& incoming = input ? input->contribution_ : incoming_storage;
    auto reserve_batch = [&]
    {
        if (input)
        {
            incoming.first = members_; incoming.count = count;
            return true; // Preparation reserved this run and its summary together.
        }
        incoming = {}; incoming.id = next_id_; incoming.first = members_; incoming.count = uint16_t(count);
        if (!reserve(incoming, contribution_bound)) return false;
        pending_summary_.id = incoming.id + 1; pending_summary_.count = after;
        // Include the contribution descriptors while admitting the replacement;
        // the old summary and all physical reservations remain charged.
        runs_.push_back(incoming); bool summary_reserved;
        try { summary_reserved = reserve(pending_summary_, multiply(chunks, sizeof(Summary)), "checkpoint_summary"); }
        catch (...)
        { runs_.pop_back(); pending_summary_ = {}; discard(incoming); throw; }
        runs_.pop_back();
        if (!summary_reserved) { pending_summary_ = {}; discard(incoming); return false; }
        next_id_ = pending_summary_.id + 1; return true;
    };
    if (!reserve_batch())
    {
        if ((!contribution_sized && !exact(false)) || !reserve_batch())
            return {false, false, final_bound, peak, "external scratch or run metadata reservation is full; finish the committed prefix", {}};
    }
    grow_failpoint("external-summary-reserved");
    planning.finish(); PhaseTimer encoding(stats_, WorkPhase::encode);
    {
        if (input) input->contribution_started_ = true;
        Writer output(*this, incoming);
        uint64_t const descriptors = sizeof(ChunkInput) + sizeof(uint64_t) + sizeof(EncodedBlock) + count * sizeof(FragmentRegion);
        uint64_t const limit = std::min({chunks, uint64_t(65536), std::max<uint64_t>(1, config_.window_bytes / descriptors),
            input ? std::max<uint64_t>(1, config_.window_bytes / multiply(maximum_input_words, sizeof(uint64_t))) : chunks});
        bool const overlap = config_.io_overlap && maximum_incoming_work <= config_.codec_bytes / 2;
        uint64_t const budget = overlap ? config_.codec_bytes / 2 : config_.codec_bytes;
        unsigned const workers = unsigned(std::max<uint64_t>(1, std::min({uint64_t(config_.workers),
            budget / maximum_incoming_work, limit})));
        uint64_t const window_chunks = codec_window_chunks(workers, maximum_incoming_work, limit, budget);
        ParallelEncoder encoder(identity_.header.k, workers, config_.codec_bytes, config_.cpu_scheduler);
        CpuTime driver_cpu;
        stats_.contribution_workers = std::max<uint64_t>(stats_.contribution_workers, workers);
        struct Window
        {
            uint64_t first = 0, next = 0, codec_ns = 0;
            std::vector<FragmentRegion> regions;
            std::vector<uint64_t> words;
            std::vector<ChunkInput> tasks;
            std::vector<uint64_t> lengths;
            std::vector<EncodedBlock> blocks;
            Window(size_t chunks, size_t members, size_t workers)
            { regions.reserve(chunks * members); tasks.reserve(chunks); lengths.reserve(chunks); blocks.reserve(workers); }
        };
        auto current = std::make_unique<Window>(size_t(window_chunks), count, workers);
        auto next = overlap ? std::make_unique<Window>(size_t(window_chunks), count, workers) : nullptr;
        rewind();
        if (input)
        {
            std::vector<uint64_t>().swap(input_words);
            current->words.reserve(size_t(config_.window_bytes / sizeof(uint64_t)));
            if (next) next->words.reserve(size_t(config_.window_bytes / sizeof(uint64_t)));
        }
        auto prepare = [&](Window& window, uint64_t first)
        {
            window.regions.clear(); window.words.clear(); window.tasks.clear(); window.first = first;
            uint64_t const window_count = std::min<uint64_t>(window_chunks, chunks - first);
            for (uint64_t i = 0; i < window_count; ++i)
            {
                read_regions(first + i, regions, window.words); size_t const offset = window.regions.size();
                window.regions.insert(window.regions.end(), regions.begin(), regions.end());
                window.tasks.push_back({first + i, {}, 0, std::span<FragmentRegion const>(window.regions).subspan(offset, count)});
            }
            window.lengths.resize(size_t(window_count)); window.next = first + window_count;
        };
        auto encode = [&](Window& window)
        {
            std::optional<ThreadCpuTimer> cpu;
            if (overlap) cpu.emplace(driver_cpu);
            auto const begin = std::chrono::steady_clock::now();
            auto const blocks = encoder.encode(window.tasks, window.lengths, overlap);
            window.blocks.assign(blocks.begin(), blocks.end()); window.codec_ns = elapsed_ns(begin);
        };
        // The producer is destroyed before the encoder, windows and borrowed
        // member spans, including when a provisional output write fails.
        auto worker = overlap ? std::make_unique<OrderedWriter>([] {}) : nullptr;
        std::future<void> pending;
        prepare(*current, 0);
        if (worker) pending = worker->submit([&] { encode(*current); });
        else encode(*current);
        if (next && current->next < chunks) prepare(*next, current->next);
        if (worker) pending.get();
        for (;;)
        {
            if (next && current->next < chunks)
            {
                pending = worker->submit([&, following = next.get()] { encode(*following); });
                ++stats_.overlap_windows;
            }
            stats_.codec_ns += current->codec_ns; stats_.encode_chunks += current->tasks.size();
            if (current->tasks.size() > workers) grow_failpoint("external-contribution-window");
            for (auto const& block : current->blocks)
            {
                size_t offset = 0;
                for (size_t i = block.begin; i < block.end; ++i)
                {
                    auto const bytes = current->lengths[i]; require(bytes <= UINT32_MAX && bytes <= block.bytes.size() - offset,
                        "external encoded block changed length");
                    Packet const packet{members_, uint16_t(count), uint32_t(bytes)};
                    output.value(packet); output.write(block.bytes.subspan(offset, size_t(bytes))); offset += size_t(bytes);
                    auto& summary = summaries[size_t(current->first) + i]; summary.bytes = grow_add(summary.bytes, bytes);
                }
                require(offset == block.bytes.size(), "external encoded block has trailing data");
            }
            if (current->next == chunks) break;
            if (next)
            {
                current.swap(next);
                if (current->next < chunks) prepare(*next, current->next);
                pending.get();
            }
            else { prepare(*current, current->next); encode(*current); }
        }
        output.finish(); grow_failpoint("external-contribution-flushed");
        contribution_driver_cpu_.add(driver_cpu); contribution_cpu_.add(driver_cpu); contribution_cpu_.add(encoder.cpu_time());
    }
    if (input)
    {
        input_reader.reset();
        pending_summary_ = std::move(input->summary_); input->summary_ = {}; pending_summary_.count = after;
    }
    contribution_bytes_ = incoming.bytes; runs_.push_back(std::move(incoming)); incoming = {}; members_ = after; final_bound_ = final_bound;
    ++packet_count_; for (auto const& summary : summaries) packing_source_bytes_ = std::max(packing_source_bytes_, summary.bytes);
    packing_work_ = ChunkCodec::estimate_work(identity_.header.k, packing_source_bytes_, 0);
    identity_.pending = true; identity_.target_members = members_; identity_.pending_peak_bytes = peak;
    save("external-run", summaries); std::vector<Summary>().swap(summaries);
    if (input) release_prepared();
    encoding.finish(); balance();
    return {true, true, final_bound_, peak, {}, {final_bound_, peak, peak - final_bound_, members_}};
}
inline bool ExternalEngine::merge_range(size_t first, size_t count)
{
    if (count < 2) return true;
    require(phase_ == Phase::stage && first <= runs_.size() && count <= runs_.size() - first, "invalid external premerge");
    PhaseTimer merging(stats_, WorkPhase::merge);
    Run merged; merged.id = next_id_; merged.first = runs_[first].first;
    uint64_t bytes = 0;
    for (size_t i = 0; i < count; ++i)
    { require(!runs_[first + i].consumed, "cannot premerge consumed external runs"); merged.count = uint16_t(merged.count + runs_[first + i].count); bytes = grow_add(bytes, runs_[first + i].bytes); }
    if (!reserve(merged, bytes)) return false;
    next_id_ = merged.id + 1; Writer output(*this, merged);
    std::vector<std::unique_ptr<Reader>> readers;
    for (size_t i = 0; i < count; ++i) readers.push_back(std::make_unique<Reader>(*this, runs_[first + i], count));
    stats_.max_active_runs = std::max<uint64_t>(stats_.max_active_runs, count);
    std::vector<uint8_t> copy(size_t(config_.window_bytes));
    for (uint64_t chunk = 0; chunk < PlotGroupFile::getChunkCountForK(identity_.header.k); ++chunk)
        for (size_t i = 0; i < count; ++i)
        {
            auto const& run = runs_[first + i];
            for (uint32_t member = run.first; member < uint32_t(run.first) + run.count;)
            {
                auto const packet = readers[i]->packet(uint16_t(member)); output.value(packet);
                for (uint64_t remaining = packet.bytes; remaining;)
                {
                    size_t const n = size_t(std::min<uint64_t>(remaining, copy.size())); readers[i]->read(std::span(copy).first(n));
                    output.write(std::span(copy).first(n)); remaining -= n;
                }
                member += packet.count;
            }
        }
    for (auto const& reader : readers) require(!reader->remaining(), "external premerge has trailing data");
    output.finish(); grow_failpoint("external-merge-flushed");
    std::vector<Run> retired(runs_.begin() + first, runs_.begin() + first + count);
    runs_.erase(runs_.begin() + first, runs_.begin() + first + count); runs_.insert(runs_.begin() + first, merged);
    ++stats_.merge_passes; save("external-merge");
    readers.clear(); for (auto const& run : retired) discard(run); return true;
}
inline void ExternalEngine::balance()
{
    // The final packer can retain more readers than one compaction consumes.
    // Merge equal adjacent tail ranges only beyond that admitted reader set.
    size_t const count = external_.fan_in;
    size_t const limit = external_.packing_inputs ? external_.packing_inputs : count;
    while (runs_.size() > limit)
    {
        size_t const first = runs_.size() - count; bool equal = true;
        for (size_t i = first + 1; i < runs_.size(); ++i) equal = equal && runs_[i].count == runs_[first].count;
        if (!equal || !merge_range(first, count)) break;
    }
}
inline void ExternalEngine::start_packing()
{
    require(!prepared_ && !contributions_, "external packing cannot overlap a prepared input");
    require(confirmed_, "external packing requires confirmed identity"); check_paths(); validate_runs();
    if (phase_ == Phase::stage && members_)
    {
        PhaseTimer packing(stats_, WorkPhase::pack);
        require(required_memory() <= config_.memory_limit_bytes, "external packing transition exceeds memory budget");
        {
            auto target_guard = target_lock();
            std::string name = work_ + "/index.XXXXXX";
            external_detail::Descriptor index(file_io::temp_file(name.data()));
            index_name_ = file_io::path_text(file_io::native_path(name).filename()); index_identity_ = stamp(name).file;
            require(file_identity(index.fd) == index_identity_, "external initial index path was replaced");
            external_detail::sync(index.fd, &stats_, {"index", &path_, &execution_destination_}); sync_directory(work_, {"directory", &path_, &execution_destination_});
        }
        // Persist ownership before any index write. An interrupted write can
        // then safely restore this inode; uncheckpointed files are preserved.
        phase_ = Phase::pack; save("external-pack-initialized");
    }
}
inline void ExternalEngine::pack()
{
    if (phase_ == Phase::stage) start_packing();
    else { check_paths(); validate_runs(); }
    PhaseTimer packing(stats_, WorkPhase::pack);
    struct BodyTime
    {
        uint64_t& total;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        ~BodyTime() { total += elapsed_ns(begin); }
    } body_time{packing_body_ns_};
    ThreadCpuTimer cpu(packing_owner_cpu_);
    require(phase_ == Phase::pack, "invalid external packing phase");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(identity_.header.k);
    std::string const index_path = work_ + "/" + index_name_;
    require(stamp(index_path).file == index_identity_, "external index was replaced");
    external_detail::Descriptor index(file_io::open_file(index_path.c_str(), file_io::read_write | file_io::close_on_exec | file_io::no_follow));
    require(file_identity(index.fd) == index_identity_, "external index handle was replaced");
    Run marker; marker.id = UINT64_MAX; Segment marker_segment; auto const expected = segment_header(marker, marker_segment);
    {
        auto target_guard = target_lock();
        external_detail::write(index.fd, 0, expected, &stats_, {"index", &path_, &execution_destination_});
        require(!file_io::resize_file(index.fd, file_io::Offset(segment_header_bytes + index_bytes_)), "cannot discard uncommitted external index tail");
    }
    BitWriter index_bits; if (index_carry_bits_) index_bits.append(index_carry_value_, index_carry_bits_);
    auto flush_index = [&](bool finish)
    {
        auto target_guard = target_lock();
        auto const bytes = index_bits.asBytes(); uint64_t const n = finish ? bytes.size() : index_bits.bitCount() / 8;
        uint32_t const carry = finish ? 0 : uint32_t(index_bits.bitCount() % 8);
        uint8_t const tail = carry ? bytes[size_t(n)] : 0;
        external_detail::write(index.fd, segment_header_bytes + index_bytes_, bytes.first(size_t(n)), &stats_, {"index", &path_, &execution_destination_});
        index_bytes_ += n; index_bits.clear(); if (carry) index_bits.append(tail, carry);
    };
    auto sync_index = [&]
    {
        auto target_guard = target_lock();
        external_detail::sync(index.fd, &stats_, {"index", &path_, &execution_destination_});
    };
    {
        auto target_guard = target_lock();
        require(!file_io::resize_file(fd_, file_io::Offset(output_bytes_)), "cannot discard uncommitted external output tail");
        external_detail::reserve_file(fd_, final_bound_);
    }
    std::vector<std::unique_ptr<Reader>> readers;
    for (auto const& run : runs_) readers.push_back(std::make_unique<Reader>(*this, run));
    stats_.max_active_runs = std::max<uint64_t>(stats_.max_active_runs, readers.size());
    stats_.merge_inputs = readers.size();
    uint64_t const task_bytes = sizeof(ChunkInput) + sizeof(uint64_t) + sizeof(EncodedBlock) + runs_.size() * sizeof(uint64_t);
    uint64_t const limit = std::min({chunks, uint64_t(65536), config_.window_bytes / packing_source_bytes_,
        uint64_t(UINT16_MAX) / packet_count_, std::max<uint64_t>(1, config_.window_bytes / task_bytes)});
    bool const overlap = config_.io_overlap && packing_work_ <= config_.codec_bytes / 2;
    uint64_t const budget = overlap ? config_.codec_bytes / 2 : config_.codec_bytes;
    unsigned const workers = unsigned(std::max<uint64_t>(1, std::min({uint64_t(config_.workers),
        budget / packing_work_, limit})));
    uint64_t const window_chunks = codec_window_chunks(workers, packing_work_, limit, budget);
    stats_.packing_workers = workers;
    ParallelEncoder encoder(identity_.header.k, workers, config_.codec_bytes, config_.cpu_scheduler, config_.remote_encoder);
    CpuTime driver_cpu, output_cpu;
    struct Window
    {
        uint64_t first = 0, next = 0, codec_ns = 0;
        std::vector<uint8_t> payload;
        std::vector<EncodedChunk> contributions;
        std::vector<ChunkInput> tasks;
        std::vector<uint64_t> lengths, frontiers;
        std::vector<EncodedBlock> blocks;
        Window(size_t bytes, size_t chunks, size_t runs, size_t workers)
        {
            payload.reserve(bytes); contributions.reserve(UINT16_MAX); tasks.reserve(chunks);
            lengths.reserve(chunks); frontiers.reserve(chunks * runs); blocks.reserve(workers);
        }
    };
    auto current = std::make_unique<Window>(size_t(config_.window_bytes), size_t(window_chunks), runs_.size(), workers);
    auto next = overlap ? std::make_unique<Window>(size_t(config_.window_bytes), size_t(window_chunks), runs_.size(), workers) : nullptr;
    std::vector<size_t> released_segments(runs_.size(), 0); std::vector<uint64_t> released_bytes(runs_.size(), 0);
    std::vector<uint8_t> output_window; output_window.reserve(size_t(config_.window_bytes));
    bool output_overlap = overlap;
    if (output_overlap)
    {
        auto const target = storage_device(file_io::native_path(path_).parent_path());
        for (auto const& root : roots_)
        {
            auto const source = storage_device(file_io::native_path(root.path));
            if (!external_detail::independent_packing_io(target, {&source, 1})) { output_overlap = false; break; }
        }
    }
    std::vector<uint8_t> writing_window;
    if (output_overlap) writing_window.reserve(size_t(config_.window_bytes));
    Stats output_stats;
    ExecutionIO output_service;
    // The consumer owns its buffer and counters until the future is joined.
    // Its destructor drains before either borrowed object can be destroyed.
    auto output_worker = output_overlap ? std::make_unique<OrderedWriter>([] {}) : nullptr;
    std::future<void> pending_output;
    auto drain_output = [&]
    {
        if (!pending_output.valid()) return;
        pending_output.get();
        stats_.written_bytes += output_stats.written_bytes; stats_.write_calls += output_stats.write_calls;
        stats_.write_ns += output_stats.write_ns;
        execution_destination_.write_ns += output_service.write_ns;
        execution_destination_.write_calls += output_service.write_calls;
        execution_destination_.written_bytes += output_service.written_bytes;
        output_stats = {}; output_service = {};
        writing_window.clear();
    };
    uint64_t window_offset = output_bytes_;
    auto flush_output = [&]
    {
        if (!output_window.empty())
        {
            if (output_worker)
            {
                drain_output(); output_window.swap(writing_window);
                require(window_offset <= config_.capacity_bytes && writing_window.size() <= config_.capacity_bytes - window_offset,
                    "external destination exceeds admitted capacity");
                pending_output = output_worker->submit([&, offset = window_offset]
                {
                    ThreadCpuTimer cpu(output_cpu);
                    auto target_guard = target_lock(); external_detail::write(fd_, offset, writing_window, &output_stats, {"final_output", &path_, &output_service});
                });
            }
            else { write_target(window_offset, output_window); output_window.clear(); }
        }
        window_offset = output_bytes_;
    };
    auto emit = [&](std::span<uint8_t const> bytes)
    {
        if (bytes.size() > output_window.capacity() - output_window.size()) flush_output();
        require(bytes.size() <= output_window.capacity(), "external canonical chunk exceeds output window");
        output_window.insert(output_window.end(), bytes.begin(), bytes.end()); output_bytes_ += bytes.size();
    };
    uint64_t committed = output_bytes_;
    PackingCheckpointHistory checkpoint_history;
    uint64_t interval_codec_ns = stats_.codec_ns;
    auto interval_begin = std::chrono::steady_clock::now();
    auto checkpoint = [&](std::span<uint64_t const> cursors)
    {
        auto const begin = std::chrono::steady_clock::now();
        uint64_t const bytes = output_bytes_ - committed;
        require(cursors.size() == runs_.size(), "external checkpoint frontier coverage changed");
        // A pending writer takes the target gate itself. Drain before locking
        // the checkpoint transaction, then release it before scratch cleanup.
        flush_output(); drain_output();
        {
            auto target_guard = target_lock();
            flush_index(false); sync_target(); sync_index(); grow_failpoint("external-pack-flushed");
            index_carry_bits_ = uint32_t(index_bits.bitCount());
            index_carry_value_ = index_carry_bits_ ? index_bits.asBytes()[0] : 0;
            for (size_t i = 0; i < runs_.size(); ++i) runs_[i].consumed = cursors[i];
            save("external-pack");
        }
        for (size_t i = 0; i < readers.size(); ++i) readers[i]->checkpoint(cursors[i]);
        // Reclaim each segment once. The durable emitted frontier can trail
        // both prereads and later chunks retained in this packing window.
        // Packing allocates no new scratch, so longer intervals retain only
        // segments already covered by the original grants.
        std::array<bool, max_roots> changed{};
        for (size_t i = 0; i < runs_.size(); ++i)
        {
            auto const& run = runs_[i];
            while (released_segments[i] < run.segments.size())
            {
                auto const& segment = run.segments[released_segments[i]];
                uint64_t const end = released_bytes[i] + segment.bytes; if (end > run.consumed) break;
                auto const path = segment_path(run, segment); file_io::Stat value{};
                if (!file_io::inspect_link(path.c_str(), &value))
                {
                    require(stamp(path) == segment.stamp, "consumed external segment changed before reclamation");
                    require(!file_io::unlink_file(path.c_str()), "cannot reclaim consumed external segment");
                    roots_[segment.root].used -= segment.capacity; stats_.reclaimed_bytes += segment.capacity;
                    changed[segment.root] = true;
                    grow_failpoint("external-segment-unlinked");
                }
                else require(errno == ENOENT, "cannot inspect consumed external segment");
                released_bytes[i] = end; ++released_segments[i];
            }
        }
        for (size_t i = 0; i < roots_.size(); ++i) if (changed[i]) sync_directory(roots_[i].path, {"directory", &path_, &execution_scratch_});
        if (checkpoint_ceiling_ && packed_chunks_ < chunks)
        {
            auto const checkpoint_ns = elapsed_ns(begin);
            checkpoint_history.add({bytes, elapsed_ns(interval_begin), checkpoint_ns, stats_.codec_ns - interval_codec_ns});
            auto const selected = checkpoint_history.select(packing_checkpoint_bytes(), checkpoint_ceiling_, final_bound_ - output_bytes_);
            if (selected != packing_checkpoint_bytes()) { packing_checkpoint_bytes_ = selected; ++checkpoint_tuning_changes_; }
        }
        committed = output_bytes_;
        interval_codec_ns = stats_.codec_ns;
        interval_begin = std::chrono::steady_clock::now();
    };
    auto prepare = [&](Window& window, uint64_t first)
    {
        auto& payload = window.payload; auto& contributions = window.contributions;
        auto& tasks = window.tasks; auto& frontiers = window.frontiers;
        payload.clear(); contributions.clear(); tasks.clear(); frontiers.clear();
        window.first = first;
        uint64_t const count = std::min<uint64_t>(window_chunks, chunks - first);
        for (uint64_t i = 0; i < count; ++i)
        {
            size_t const begin = contributions.size(); uint32_t covered = 0;
            for (size_t r = 0; r < runs_.size(); ++r)
            {
                auto const& run = runs_[r];
                for (uint32_t member = run.first; member < uint32_t(run.first) + run.count;)
                {
                    auto const packet = readers[r]->packet(uint16_t(member));
                    require(packet.bytes <= config_.window_bytes - payload.size() && contributions.size() < contributions.capacity(),
                        "external packing source/descriptor window exceeds admission");
                    size_t const offset = payload.size(); payload.resize(offset + packet.bytes);
                    auto const part = std::span(payload).subspan(offset, packet.bytes); readers[r]->read(part);
                    contributions.push_back({part, packet.count}); member += packet.count; covered += packet.count;
                }
                frontiers.push_back(readers[r]->position());
            }
            require(covered == members_, "external final member coverage changed");
            tasks.push_back({first + i, {}, 0, {}, std::span<EncodedChunk const>(contributions).subspan(begin)});
        }
        window.lengths.resize(size_t(count)); window.next = first + count;
    };
    auto encode = [&](Window& window)
    {
        std::optional<ThreadCpuTimer> cpu;
        if (overlap) cpu.emplace(driver_cpu);
        auto const begin = std::chrono::steady_clock::now();
        auto const blocks = encoder.encode(window.tasks, window.lengths, overlap);
        window.blocks.assign(blocks.begin(), blocks.end()); window.codec_ns = elapsed_ns(begin);
    };
    // The codec producer and independent target consumer borrow separate
    // windows. Readers, index and durable frontiers stay on the foreground.
    // Join both workers before their windows can die, including on failure.
    auto worker = overlap ? std::make_unique<OrderedWriter>([] {}) : nullptr;
    std::future<void> pending;
    if (packed_chunks_ < chunks)
    {
        prepare(*current, packed_chunks_);
        if (worker) pending = worker->submit([&] { encode(*current); });
        else encode(*current);
        if (next && current->next < chunks) prepare(*next, current->next);
        if (worker) pending.get();
    }
    while (packed_chunks_ < chunks)
    {
        if (next && current->next < chunks)
        {
            pending = worker->submit([&, following = next.get()] { encode(*following); });
            ++stats_.packing_overlap_windows;
        }
        stats_.codec_ns += current->codec_ns; stats_.encode_chunks += current->tasks.size();
        for (auto const& block : current->blocks)
        {
            size_t offset = 0;
            for (size_t i = block.begin; i < block.end; ++i)
            {
                uint64_t const bytes = current->lengths[i]; require(bytes <= block.bytes.size() - offset, "external packed block changed length");
                emit(block.bytes.subspan(offset, size_t(bytes))); offset += size_t(bytes);
                gsz_encode(std::span<uint64_t const>(&bytes, 1), members_, index_bits);
                if (index_bits.asBytes().size() >= packing_index_bytes(config_)) { drain_output(); flush_index(false); }
                packed_chunks_ = current->first + i + 1;
                if (output_bytes_ - committed >= packing_checkpoint_bytes() || packed_chunks_ == chunks)
                {
                    checkpoint(std::span<uint64_t const>(current->frontiers).subspan(i * runs_.size(), runs_.size()));
                    if (current->tasks.size() > workers && packed_chunks_ < current->next) grow_failpoint("external-pack-window-checkpoint");
                }
            }
            require(offset == block.bytes.size(), "external packed block has trailing bytes");
        }
        if (current->next == chunks) break;
        if (next)
        {
            current.swap(next);
            if (current->next < chunks) prepare(*next, current->next);
            pending.get();
        }
        else { prepare(*current, current->next); encode(*current); }
    }
    drain_output();
    for (auto const& reader : readers) require(!reader->remaining(), "external packed input has trailing data");
    Header header = identity_.header; header.group_size = members_; header.chunk_index_offset = output_bytes_;
    flush_index(true); sync_index();
    auto& payload = current->payload; payload.resize(size_t(config_.window_bytes));
    for (uint64_t position = 0; position < index_bytes_;)
    {
        size_t const n = size_t(std::min<uint64_t>(payload.size(), index_bytes_ - position));
        { auto target_guard = target_lock();
          external_detail::read(index.fd, segment_header_bytes + position, std::span(payload).first(n), &stats_, {"index", &path_, &execution_destination_}); }
        write_target(output_bytes_, std::span(payload).first(n)); output_bytes_ += n; position += n;
    }
    require(output_bytes_ <= final_bound_, "external ordinary output exceeded its admitted bound");
    auto target_guard = target_lock();
    require(!file_io::resize_file(fd_, file_io::Offset(output_bytes_)), "cannot trim external output reservation");
    sync_target(); grow_failpoint("external-index-flushed");
    identity_.header = header; phase_ = Phase::ready; save("external-ready");
    packing_driver_cpu_.add(driver_cpu); packing_driver_cpu_.add(output_cpu);
    packing_cpu_.add(driver_cpu); packing_cpu_.add(output_cpu); packing_cpu_.add(encoder.cpu_time());
}
inline void ExternalEngine::cleanup()
{
    check_paths();
    for (auto const& run : runs_) discard(run);
    if (index_identity_.inode)
    {
        auto target_guard = target_lock();
        std::string const path = work_ + "/" + index_name_; file_io::Stat exists{};
        if (!file_io::inspect_link(path.c_str(), &exists))
        {
            require(stamp(path).file == index_identity_, "completed external index was replaced");
            require(!file_io::unlink_file(path.c_str()), "cannot remove completed external index"); sync_directory(work_, {"directory", &path_, &execution_destination_});
        }
        else require(errno == ENOENT, "cannot inspect completed external index");
    }
}
inline Result ExternalEngine::finish()
{
    require(confirmed_, "external finish requires confirmed identity"); check_paths();
    if (!members_ && phase_ == Phase::stage)
    {
        phase_ = Phase::empty; output_bytes_ = final_bound_ = 0;
        identity_.pending = false; identity_.pending_peak_bytes = 0; save("external-empty");
    }
    if (phase_ == Phase::empty)
    {
        auto target_guard = target_lock();
        require(!members_ && runs_.empty() && !index_identity_.inode, "invalid empty external handoff");
        require(!file_io::resize_file(fd_, 0), "cannot restore empty external target"); sync_target();
        completed_ = stamp(path_); save("external-empty-ready");
    }
    if (phase_ == Phase::stage || phase_ == Phase::pack) pack();
    if (phase_ == Phase::ready)
    {
        auto target_guard = target_lock();
        // Payload and index are already synced, with a durable ready record.
        // A crash around this header patch never makes resume trim the file.
        Header header = identity_.header; header.version = PlotGroupFile::FORMAT_VERSION;
        write_target(0, {reinterpret_cast<uint8_t const*>(&header), sizeof(header)}); sync_target();
        grow_failpoint("external-header-flushed");
        identity_.header = header; identity_.bytes = output_bytes_; identity_.pending = false;
        completed_ = stamp(path_); phase_ = Phase::done; save("external-published");
        sync_directory(file_io::path_text(file_io::native_path(path_).parent_path()), {"directory", &path_, &execution_destination_});
    }
    cleanup();
    if (!runs_.empty())
    {
        runs_.clear(); save("external-retired");
    }
    return {output_bytes_, grow_add(final_bound_, metadata_bytes(identity_.header.k, external_)),
        metadata_bytes(identity_.header.k, external_), members_};
}
}

namespace group_grow
{
inline ExternalEngine::Inspection ExternalEngine::inspect_at(std::string const& path, std::string const& work, uint64_t memory_limit_bytes, bool retired)
{
    using namespace external_detail;
    std::string const target = file_io::path_text(std::filesystem::absolute(file_io::native_path(path)).lexically_normal());
    std::string const checkpoint = retired ? work + "/checkpoint" : checkpoint_path(work, target, memory_limit_bytes / 8);
    auto const value = stamp(checkpoint);
    require(value.bytes <= uint64_t(SIZE_MAX) && value.bytes <= memory_limit_bytes / 8, "external checkpoint inspection exceeds memory budget");
    Descriptor file(file_io::open_file(checkpoint.c_str(), file_io::read_only | file_io::close_on_exec | file_io::no_follow));
    std::string record(size_t(value.bytes), '\0'); read(file.fd, 0, {reinterpret_cast<uint8_t*>(record.data()), record.size()}, nullptr, {"checkpoint_manifest", &target});
    require(record.size() >= 65 && record[64] == '\n' && std::string_view(record).substr(0, 64) == digest(std::string_view(record).substr(65)),
        "external checkpoint checksum mismatch");
    CheckpointInput in(std::string_view(record).substr(65));
    require(get_string(in) == "xchplot2-external-3" && get_string(in) == GROW_IDENTITY && get_string(in) == target,
        "external checkpoint source/version/destination mismatch");
    auto const token = get_string(in);
    require(token.size() == 64 && (!retired || work == target + ".external.retired-" + token),
        "retired external directory name does not match its ownership record");
    require(get<FileIdentity>(in) == file_identity(target, false), "external destination was replaced");
    require(get<FileIdentity>(in) == file_identity(work, true), "external checkpoint directory was replaced");
    require(get<FileIdentity>(in) == file_identity(file_io::path_text(file_io::native_path(target).parent_path()), true), "external destination parent directory was replaced");
    auto const header = get<Header>(in);
    require(header.magic == PlotGroupFile::MAGIC && header.k >= 18 && header.k <= 28 && !(header.k & 1), "invalid external checkpoint header");
    auto const memo = get_string(in, 255);
    ExternalConfig external; external.segment_bytes = get<uint64_t>(in); external.fan_in = get<unsigned>(in);
    external.packing_inputs = get<unsigned>(in);
    require(external.fan_in >= 2 && external.fan_in <= max_runs && (!external.packing_inputs ||
        (external.packing_inputs >= external.fan_in && external.packing_inputs <= max_runs)),
        "invalid external checkpoint reader settings");
    auto const roots = get<uint64_t>(in); require(roots && roots <= max_roots, "invalid external scratch root count");
    std::vector<Root> saved_roots; saved_roots.reserve(size_t(roots));
    Inspection result; result.identity.header = header; result.identity.memo.assign(memo.begin(), memo.end());
    for (uint64_t i = 0; i < roots; ++i)
    {
        auto const root = get_string(in); auto const root_identity = get<FileIdentity>(in), root_parent = get<FileIdentity>(in);
        file_io::Stat root_exists{}; bool const exists = !file_io::inspect_link(root.c_str(), &root_exists);
        if (!exists) require(retired && errno == ENOENT, "external scratch directory is missing");
        else require(root_identity == file_identity(root, true) &&
            root_parent == file_identity(file_io::path_text(file_io::native_path(root).parent_path()), true), "external scratch namespace or parent was replaced");
        uint64_t const capacity = get<uint64_t>(in), alignment = get<uint64_t>(in), peak = get<uint64_t>(in);
        require(alignment && peak <= capacity, "invalid external scratch alignment/peak");
        saved_roots.push_back({root, root_identity, root_parent, capacity, alignment, 0, peak});
        ScratchGrant const grant{file_io::path_text(file_io::native_path(root).parent_path()), capacity};
        external.scratch.push_back(grant); result.scratch.push_back(grant);
        uint64_t bytes = 0, occupied = exists ? multiply(uint64_t(root_exists.st_blocks), 512) : 0;
        if (exists) for (auto const& entry : std::filesystem::directory_iterator(file_io::native_path(root)))
        {
            auto const saved = stamp(file_io::path_text(entry.path())); file_io::Stat allocation{};
            require(!file_io::inspect_link(entry.path().c_str(), &allocation), "cannot inspect external scratch allocation");
            uint64_t const allocated = multiply(uint64_t(allocation.st_blocks), 512);
            bytes = grow_add(bytes, std::max(aligned(saved.bytes, alignment), allocated)); occupied = grow_add(occupied, allocated);
        }
        require(bytes <= capacity, "external scratch occupancy exceeds saved grant"); result.reserved_bytes.push_back(bytes); result.peak_reserved_bytes.push_back(std::max(peak, bytes)); result.allocated_bytes.push_back(0); result.occupied_bytes.push_back(occupied);
    }
    auto const phase = get<Phase>(in); result.members = get<uint16_t>(in); auto const next_id = get<uint64_t>(in);
    auto const bound = get<uint64_t>(in), committed = get<uint64_t>(in);
    require(unsigned(phase) <= unsigned(Phase::empty), "invalid external checkpoint phase");
    if (retired) require(phase == Phase::done || phase == Phase::empty, "retired external record is not terminal");
    if (checkpoint != work + "/checkpoint") require(phase == Phase::stage && !result.members, "only an initial external checkpoint can be recovered from a temporary record");
    result.finishing = phase != Phase::stage; result.complete = retired || phase == Phase::done; result.external = external;
    result.identity.pending = !result.complete && result.members; result.identity.target_members = result.members;
    if (result.complete) { result.identity.bytes = committed; result.identity.header.group_size = result.members; }
    file_io::Stat target_info{}; require(!file_io::inspect_link(target.c_str(), &target_info) && file_io::is_regular(target_info.st_mode) && target_info.st_nlink == 1 &&
        (retired || uint64_t(target_info.st_size) >= committed), "external destination is truncated or shared");
    result.target_allocated_bytes = multiply(uint64_t(target_info.st_blocks), 512);
    result.target_bytes = std::max(uint64_t(target_info.st_size), result.target_allocated_bytes);
    auto const chunk = get<uint64_t>(in); (void)get<Stats>(in);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(header.k);
    require(chunk <= chunks && next_id <= UINT32_MAX, "invalid external checkpoint progress");
    auto const count = get<uint64_t>(in); require(count <= max_runs, "too many external checkpoint runs");
    uint32_t member = 0; uint64_t descriptors = 0;
    std::vector<uint64_t> run_ids; run_ids.reserve(size_t(count));
    for (uint64_t i = 0; i < count; ++i)
    {
        auto const id = get<uint64_t>(in), bytes = get<uint64_t>(in), consumed = get<uint64_t>(in);
        run_ids.push_back(id);
        auto const first = get<uint16_t>(in), members = get<uint16_t>(in);
        require(id < next_id && first == member && members && members <= UINT16_MAX - member && consumed <= bytes,
            "invalid external checkpoint run coverage");
        require(!retired || consumed == bytes, "retired external record still needs unread scratch"); member += members;
        auto const segments = get<uint64_t>(in);
        descriptors = grow_add(descriptors, segments);
        require(segments && descriptors <= descriptor_limit(external) && segments <= uint64_t(in.rdbuf()->in_avail()) / sizeof(Segment),
            "external segment descriptors exceed checkpoint record");
        uint64_t position = 0;
        for (uint64_t ordinal = 0; ordinal < segments; ++ordinal)
        {
            auto const segment = get<Segment>(in);
            require(segment.root < saved_roots.size() && segment.ordinal == ordinal && segment.capacity > segment_header_bytes &&
                segment.bytes && segment.bytes <= segment.capacity - segment_header_bytes && segment.stamp.bytes == segment.capacity,
                "invalid external segment descriptor");
            position = grow_add(position, segment.bytes);
            std::string const path = saved_roots[segment.root].path + "/run-" + std::to_string(id) + "-" + std::to_string(ordinal);
            file_io::Stat exists{};
            if (file_io::inspect_link(path.c_str(), &exists)) require(errno == ENOENT && position <= consumed, "required external segment is missing");
            else
            {
                require(stamp(path) == segment.stamp, "external segment changed");
                result.allocated_bytes[segment.root] = grow_add(result.allocated_bytes[segment.root],
                    std::min(segment.capacity, multiply(uint64_t(exists.st_blocks), 512)));
            }
        }
        require(position == bytes, "external run length changed");
    }
    require(member == result.members || (phase == Phase::done && !count), "external member coverage is incomplete");
    auto const summaries = get<uint64_t>(in);
    require(summaries == (phase == Phase::stage ? chunks : 0), "external summary table coverage changed");
    if (phase == Phase::stage && result.members)
    {
        Run summary; summary.id = get<uint64_t>(in); summary.bytes = get<uint64_t>(in); summary.count = result.members;
        auto const segments = get<uint64_t>(in); descriptors = grow_add(descriptors, segments);
        require(segments && descriptors <= descriptor_limit(external) && segments <= uint64_t(in.rdbuf()->in_avail()) / sizeof(Segment) &&
            summary.id < next_id && summary.bytes == multiply(chunks, sizeof(Summary)) &&
            std::find(run_ids.begin(), run_ids.end(), summary.id) == run_ids.end(), "invalid external summary generation");
        summary.segments.resize(size_t(segments));
        for (auto& segment : summary.segments) segment = get<Segment>(in);
        auto const checksum = get<std::array<uint8_t, 32>>(in);
        require(grow_add(multiply(value.bytes, 8), 65536) <= memory_limit_bytes,
            "external summary inspection exceeds memory budget");
        read_summary(saved_roots, token, summary, checksum, {}, nullptr, {"checkpoint_summary", &target});
        for (auto const& segment : summary.segments)
        {
            file_io::Stat allocation{};
            auto const path = saved_roots[segment.root].path + "/run-" + std::to_string(summary.id) + "-" + std::to_string(segment.ordinal);
            require(!file_io::inspect_link(path.c_str(), &allocation), "cannot inspect external summary allocation");
            result.allocated_bytes[segment.root] = grow_add(result.allocated_bytes[segment.root],
                std::min(segment.capacity, multiply(uint64_t(allocation.st_blocks), 512)));
        }
    }
    auto const index_bytes = get<uint64_t>(in); auto const carry_bits = get<uint32_t>(in), carry_value = get<uint32_t>(in);
    auto const index_identity = get<FileIdentity>(in); auto const index_name = get_string(in, 64); auto const completed = get<Stamp>(in);
    require(index_name.empty() || (index_name.starts_with("index.") && file_io::native_path(index_name).filename() == index_name), "invalid external index name");
    auto const packing_source = get<uint64_t>(in), packing_work = get<uint64_t>(in), packets = get<uint64_t>(in);
    require(!result.members || (packets && packets <= result.members && packing_source &&
        packing_work == ChunkCodec::estimate_work(header.k, packing_source, 0)), "invalid external packing admission record");
    require(index_bytes <= multiply(chunks, 8) && carry_bits < 8 && (carry_bits ? carry_value < (1u << carry_bits) : !carry_value),
        "invalid external index checkpoint");
    require(in.rdbuf()->in_avail() == 0, "external checkpoint has trailing data");
    if (phase == Phase::pack)
    {
        auto const index = stamp(work + "/" + index_name);
        require(index.file == index_identity && index.bytes >= ((chunk || index_bytes) ? grow_add(segment_header_bytes, index_bytes) : 0), "external index is missing, replaced or truncated");
    }
    if (!retired && (phase == Phase::ready || phase == Phase::done))
        require(chunk == chunks && uint64_t(target_info.st_size) == committed, "published external output changed length");
    if (!retired && phase == Phase::done) require(stamp(target) == completed, "completed external output changed");
    result.metadata_reserved_bytes = metadata_bytes(header.k, external);
    uint64_t actual_metadata = 0;
    for (auto const& entry : std::filesystem::directory_iterator(file_io::native_path(work)))
    {
        auto const saved = stamp(file_io::path_text(entry.path()));
        file_io::Stat allocation{}; require(!file_io::inspect_link(entry.path().c_str(), &allocation), "cannot inspect external metadata allocation");
        uint64_t const allocated = multiply(uint64_t(allocation.st_blocks), 512);
        actual_metadata = grow_add(actual_metadata, std::max(saved.bytes, allocated));
        result.metadata_occupied_bytes = grow_add(result.metadata_occupied_bytes, allocated);
        if (entry.path() == checkpoint || (entry.path().filename() == index_name && saved.file == index_identity))
            result.metadata_allocated_bytes = grow_add(result.metadata_allocated_bytes, allocated);
    }
    file_io::Stat work_allocation{}; require(!file_io::inspect_link(work.c_str(), &work_allocation), "cannot inspect external metadata directory allocation");
    result.metadata_occupied_bytes = grow_add(result.metadata_occupied_bytes, multiply(uint64_t(work_allocation.st_blocks), 512));
    result.metadata_reserved_bytes = retired ? result.metadata_occupied_bytes : std::max(result.metadata_reserved_bytes, actual_metadata);
    result.peak_bytes = grow_add(retired ? result.target_bytes : std::max(bound, result.target_bytes), result.metadata_reserved_bytes);
    return result;
}
}

namespace group_grow
{
inline bool ExternalEngine::release_completed()
{
    require(confirmed_ && (phase_ == Phase::done || phase_ == Phase::empty), "external handoff requires a durable ordinary or empty group");
    check_paths(); require(stamp(path_) == completed_, "completed external output changed before ordinary handoff");
    if (members_)
    {
        Header header{}; read_target(0, {reinterpret_cast<uint8_t*>(&header), sizeof(header)});
        require(!std::memcmp(&header, &identity_.header, sizeof(header)) && header.version == PlotGroupFile::FORMAT_VERSION &&
            header.group_size == members_, "completed external header changed before ordinary handoff");
    }
    else require(phase_ == Phase::empty && !completed_.bytes, "external empty target was not durably restored");
    cleanup();
    auto const checkpoint = stamp(work_ + "/checkpoint");
    require(checkpoint.file == checkpoint_identity_, "external checkpoint changed before handoff");
    bool clean = true;
    for (auto const& root : roots_) clean = clean && std::filesystem::is_empty(file_io::native_path(root.path));
    for (auto const& entry : std::filesystem::directory_iterator(file_io::native_path(work_)))
        clean = clean && entry.path().filename() == "checkpoint";
    std::string const holder = work_ + ".retired-" + token_;
    // The existing checkpoint directory supplies its own ownership proof.
    // A no-replace rename leaves either that complete directory at work_ or
    // the same inode at holder; no unauthenticated intermediate can remain.
    auto const holder_identity = work_identity_;
    {
        auto target_guard = target_lock();
        check_paths(); require(stamp(work_ + "/checkpoint") == checkpoint, "external checkpoint changed before retirement");
        grow_failpoint("external-handoff-before-retirement");
#if defined(__linux__) || defined(_WIN32)
        int const renamed = file_io::rename_file(work_, holder, false);
        int const error = renamed ? errno : 0;
        require(!renamed, "cannot atomically retire external checkpoint without replacing an occupied path: " + std::string(std::strerror(error)));
#else
        throw std::runtime_error("external checkpoint retirement requires native atomic no-replace rename");
#endif
        released_ = true;
        require(file_identity(holder, true) == holder_identity, "retired external checkpoint directory was replaced");
        sync_directory(holder, {"directory", &path_}); sync_directory(file_io::path_text(file_io::native_path(path_).parent_path()), {"directory", &path_, &execution_destination_});
        grow_failpoint("external-handoff-retired");
    }
    if (!clean) { measure_remaining_metadata(holder, holder_identity); return false; }
    for (auto const& root : roots_)
    {
        require(file_identity(file_io::path_text(file_io::native_path(root.path).parent_path()), true) == root.parent &&
            file_identity(root.path, true) == root.directory && std::filesystem::is_empty(file_io::native_path(root.path)),
            "external scratch namespace changed before final cleanup");
        require(!file_io::remove_directory(root.path.c_str()), "cannot remove completed external scratch namespace");
        sync_directory(file_io::path_text(file_io::native_path(root.path).parent_path()), {"directory", &path_});
    }
    auto target_guard = target_lock();
    require(file_identity(holder, true) == work_identity_ && stamp(holder + "/checkpoint") == checkpoint,
        "retired external checkpoint was replaced");
    // Keep the last ownership proof until an inode-resident receipt is
    // durable. Filesystems without user xattrs retain the compact record.
    if (!save_retired_receipt(holder, holder_identity)) { measure_remaining_metadata(holder, holder_identity); return true; }
    require(!file_io::unlink_file((holder + "/checkpoint").c_str()), "cannot remove retired external checkpoint"); sync_directory(holder, {"directory", &path_});
    grow_failpoint("external-handoff-checkpoint-removed");
    require(file_identity(holder, true) == holder_identity && !file_io::remove_directory(holder.c_str()), "cannot remove external handoff directory");
    sync_directory(file_io::path_text(file_io::native_path(path_).parent_path()), {"directory", &path_, &execution_destination_}); measure_remaining_metadata(holder, holder_identity); return true;
}
}

namespace group_grow
{
inline std::string ExternalEngine::checkpoint_path(std::string const& work, std::string const& target, uint64_t limit)
{
    using namespace external_detail;
    file_io::Stat exists{}; std::string const ordinary = work + "/checkpoint";
    if (!file_io::inspect_link(ordinary.c_str(), &exists)) return ordinary;
    require(errno == ENOENT, "cannot inspect external checkpoint");
    // The initial atomic-save record is complete and synced before its test
    // boundary. Recover that record read-only; confirmation installs it before
    // any cleanup. Later checkpoints are never inferred from temporary files.
    for (auto const& entry : std::filesystem::directory_iterator(file_io::native_path(work)))
    {
        if (!file_io::path_text(entry.path().filename()).starts_with("checkpoint.partial.")) continue;
        auto const saved = stamp(file_io::path_text(entry.path())); if (saved.bytes < 65 || saved.bytes > limit) continue;
        Descriptor file(file_io::open_file(entry.path().c_str(), file_io::read_only | file_io::no_follow | file_io::close_on_exec));
        std::string record(size_t(saved.bytes), '\0'); read(file.fd, 0, {reinterpret_cast<uint8_t*>(record.data()), record.size()}, nullptr, {"checkpoint_manifest", &target});
        if (record[64] != '\n' || std::string_view(record).substr(0, 64) != digest(std::string_view(record).substr(65))) continue;
        CheckpointInput in(std::string_view(record).substr(65));
        if (get_string(in) != "xchplot2-external-3" || get_string(in) != GROW_IDENTITY || get_string(in) != target) continue;
        (void)get_string(in);
        if (get<FileIdentity>(in) == file_identity(target, false) && get<FileIdentity>(in) == file_identity(work, true) &&
            get<FileIdentity>(in) == file_identity(file_io::path_text(file_io::native_path(target).parent_path()), true))
            return file_io::path_text(entry.path());
    }
    throw std::runtime_error("external construction has no durable initial checkpoint; no members were acknowledged");
}
}

namespace group_grow
{
inline ExternalEngine::Inspection ExternalEngine::inspect(std::string const& path, uint64_t memory_limit_bytes)
{
    std::string const target = file_io::path_text(std::filesystem::absolute(file_io::native_path(path)).lexically_normal());
    return inspect_at(target, target + ".external", memory_limit_bytes, false);
}
inline ExternalEngine::Inspection ExternalEngine::inspect_retired(std::string const& path, std::string const& directory,
                                                                 uint64_t memory_limit_bytes)
{
    std::string const target = file_io::path_text(std::filesystem::absolute(file_io::native_path(path)).lexically_normal());
    std::string const holder = file_io::path_text(std::filesystem::absolute(file_io::native_path(directory)).lexically_normal());
    require(file_io::native_path(holder).parent_path() == file_io::native_path(target).parent_path() &&
        file_identity(holder, true).inode, "retired external directory is outside the target parent");
    file_io::Stat checkpoint{};
    if (file_io::inspect_link((holder + "/checkpoint").c_str(), &checkpoint))
    {
        require(errno == ENOENT, "cannot inspect retired external checkpoint");
        return inspect_receipt(target, holder, memory_limit_bytes);
    }
    return inspect_at(target, holder, memory_limit_bytes, true);
}
}

namespace group_grow
{
inline bool ExternalEngine::save_retired_receipt(std::string const& holder, FileIdentity holder_identity)
{
    auto target_guard = target_lock();
#ifdef __linux__
    using namespace external_detail;
    std::ostringstream out(std::ios::binary); out.exceptions(std::ios::badbit | std::ios::failbit);
    put_string(out, "xchplot2-external-retired-1"); put_string(out, GROW_IDENTITY); put_string(out, path_); put_string(out, token_);
    put(out, holder_identity); put(out, target_); put(out, parent_); put(out, work_identity_);
    put(out, identity_.header); put_string(out, {reinterpret_cast<char const*>(identity_.memo.data()), identity_.memo.size()});
    put(out, members_); put(out, output_bytes_);
    std::string const body = out.str(), receipt = digest(body) + '\n' + body;
    Descriptor directory(file_io::open_file(holder.c_str(), file_io::read_only | file_io::directory | file_io::no_follow | file_io::close_on_exec));
    file_io::Stat value{}; require(!file_io::inspect_fd(directory.fd, &value) && file_io::is_directory(value.st_mode) &&
        FileIdentity{uint64_t(value.st_dev), uint64_t(value.st_ino)} == holder_identity && file_identity(holder, true) == holder_identity,
        "external retired holder was replaced before its ownership receipt");
    if (::fsetxattr(directory.fd, "user.xchplot2.external-retired", receipt.data(), receipt.size(), XATTR_CREATE))
    {
        require(errno == ENOTSUP || errno == EOPNOTSUPP || errno == ENOSYS || errno == EPERM || errno == EACCES ||
            errno == ENOSPC || errno == EEXIST || errno == E2BIG || errno == ERANGE, "cannot install external retired ownership receipt");
        return false;
    }
    external_detail::sync(directory.fd, nullptr, {"checkpoint_manifest", &path_, &execution_destination_}); return true;
#else
    (void)holder; (void)holder_identity; return false;
#endif
}
inline ExternalEngine::Inspection ExternalEngine::inspect_receipt(std::string const& path, std::string const& holder,
                                                                 uint64_t memory_limit_bytes)
{
#ifdef __linux__
    using namespace external_detail;
    Descriptor directory(file_io::open_file(holder.c_str(), file_io::read_only | file_io::directory | file_io::no_follow | file_io::close_on_exec));
    file_io::Count const size = ::fgetxattr(directory.fd, "user.xchplot2.external-retired", nullptr, 0);
    require(size >= 65 && uint64_t(size) <= memory_limit_bytes / 8, "retired external metadata has no bounded ownership receipt");
    std::string record(size_t(size), '\0');
    require(::fgetxattr(directory.fd, "user.xchplot2.external-retired", record.data(), record.size()) == size &&
        record[64] == '\n' && std::string_view(record).substr(0, 64) == digest(std::string_view(record).substr(65)), "retired external ownership receipt is corrupt");
    CheckpointInput in(std::string_view(record).substr(65));
    require(get_string(in) == "xchplot2-external-retired-1" && get_string(in) == GROW_IDENTITY && get_string(in) == path,
        "retired external receipt source/destination mismatch");
    auto const token = get_string(in); require(token.size() == 64 && holder == path + ".external.retired-" + token,
        "retired external receipt name mismatch");
    auto const owner = get<FileIdentity>(in), target = get<FileIdentity>(in), parent = get<FileIdentity>(in), checkpoint_directory = get<FileIdentity>(in);
    file_io::Stat value{}; require(!file_io::inspect_fd(directory.fd, &value) && file_io::is_directory(value.st_mode) &&
        FileIdentity{uint64_t(value.st_dev), uint64_t(value.st_ino)} == owner && file_identity(holder, true) == owner &&
        file_identity(path, false) == target && file_identity(file_io::path_text(file_io::native_path(path).parent_path()), true) == parent,
        "retired external receipt inode ownership mismatch");
    require(checkpoint_directory == owner, "retired receipt checkpoint directory differs from its holder");
    Inspection result; result.identity.header = get<Header>(in); auto const memo = get_string(in, 255);
    result.identity.memo.assign(memo.begin(), memo.end()); result.members = get<uint16_t>(in);
    result.identity.bytes = get<uint64_t>(in); result.identity.target_members = result.members;
    auto const& header = result.identity.header;
    require(header.magic == PlotGroupFile::MAGIC && header.version == PlotGroupFile::FORMAT_VERSION && header.k >= 18 && header.k <= 28 &&
        !(header.k & 1) && header.group_size == result.members && header.memo_length == memo.size() &&
        (result.members ? result.identity.bytes > header.chunk_index_offset : !result.identity.bytes) && in.rdbuf()->in_avail() == 0,
        "invalid terminal external ownership receipt");
    result.complete = result.finishing = true;
    result.metadata_occupied_bytes = multiply(uint64_t(value.st_blocks), 512);
    require(std::filesystem::is_empty(file_io::native_path(holder)), "unrecognized contents in retired external holder");
    auto const output = stamp(path); file_io::Stat allocation{}; require(!file_io::inspect_link(path.c_str(), &allocation), "cannot inspect retired target allocation");
    result.target_allocated_bytes = multiply(uint64_t(allocation.st_blocks), 512);
    result.target_bytes = std::max(output.bytes, result.target_allocated_bytes);
    result.metadata_reserved_bytes = result.metadata_occupied_bytes;
    result.peak_bytes = grow_add(result.target_bytes, result.metadata_reserved_bytes); return result;
#else
    (void)path; (void)holder; (void)memory_limit_bytes;
    throw std::runtime_error("retired external checkpoint is missing its ownership proof");
#endif
}
}

namespace group_grow
{
inline void ExternalEngine::measure_remaining_metadata(std::string const& holder, FileIdentity holder_identity)
{
    remaining_metadata_bytes_ = 0;
    auto count = [&](std::string const& path, FileIdentity expected, bool directory)
    {
        file_io::Stat value{};
        if (file_io::inspect_link(path.c_str(), &value)) { require(errno == ENOENT, "cannot inspect retained external metadata"); return; }
        require((directory ? file_io::is_directory(value.st_mode) : (file_io::is_regular(value.st_mode) && value.st_nlink == 1)) &&
            FileIdentity{uint64_t(value.st_dev), uint64_t(value.st_ino)} == expected, "retained external metadata was replaced");
        remaining_metadata_bytes_ = grow_add(remaining_metadata_bytes_, multiply(uint64_t(value.st_blocks), 512));
    };
    count(holder, holder_identity, true); count(holder + "/checkpoint", checkpoint_identity_, false);
}
}
