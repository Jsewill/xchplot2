#include "host/GroupExternal.hpp"
#include <iostream>
#include <sys/wait.h>
using namespace group_grow;
namespace fs = std::filesystem;
#ifdef GROW_TEST_WRAP_PWRITE
std::atomic<bool> fail_next_write{false};
std::atomic<bool> fail_next_codec{false};
std::atomic<bool> interrupt_next_write{false};
std::atomic<unsigned> writes_before_failure{0};
std::atomic<bool> reject_retired_receipt{false};
std::atomic<bool> reject_atomic_retirement{false};
std::atomic<unsigned> retirement_flags{0};
std::atomic<bool> slow_checkpoint_writes{false};
std::atomic<bool> observe_oversized_partial{false};
std::atomic<uint64_t> oversized_partial_reads{0};
std::atomic<bool> observe_index_writes{false};
std::atomic<uint64_t> largest_index_write{0};
std::atomic<int> index_write_error{0};
struct PreparedWriteBarrier
{
    std::mutex mutex; std::condition_variable ready;
    bool armed = false, blocked = false, released = false, timed_out = false;
} prepared_write_barrier;
struct SummaryWriteObserver
{
    std::mutex mutex;
    std::condition_variable ready;
    std::string namespace_name;
    uint64_t bytes = 0;
    bool armed = false;
} summary_write_observer;
struct SummaryReadObserver
{
    std::mutex mutex;
    std::string prefix;
    uint64_t bytes = 0, maximum = 0;
    bool armed = false, fail = false;
} summary_read_observer;
struct SegmentSyncFailure
{
    std::mutex mutex;
    std::string path;
    unsigned remaining = 0;
    bool armed = false, after_payload = false, payload = false;
} segment_sync_failure;
// Hold an actual codec call until foreground I/O observes it. The controller
// releases every barrier, including failed assertions, and each worker wait
// has a deadline so a broken serial/pipeline schedule reports rather than hangs.
struct PipelineBarrier
{
    std::mutex mutex;
    std::condition_variable ready;
    bool armed = false, enabled = false, released = false, blocked = false, observed = false, timed_out = false;
    bool read = false, fail = false;
    size_t calls = 0, hold = 0;
    uint64_t read_offset = 0, write_offset = 0;
} pipeline_barrier;
struct TargetPipelineBarrier
{
    std::mutex mutex;
    std::condition_variable ready;
    bool armed = false, enabled = false, released = false, codec_blocked = false, write_blocked = false;
    bool read_observed = false, timed_out = false, fail_read = false, fail_write = false;
    bool observe_stage = false, stage_checkpoint = false;
    size_t calls = 0;
    uint64_t read_offset = 0, stage_bytes = 0, stage_expected_bytes = 0;
    std::string stage_checkpoint_prefix;
    dev_t target_device = 0;
    ino_t target_inode = 0;
} target_pipeline_barrier;
struct SharedStorageBarrier
{
    std::mutex mutex;
    std::condition_variable ready;
    std::string path;
    bool armed = false, sync = false, blocked = false, released = false, timed_out = false;
} shared_storage_barrier;
void hold_shared_storage(int fd, bool sync)
{
    std::unique_lock lock(shared_storage_barrier.mutex); auto& state = shared_storage_barrier;
    if (!state.armed || state.blocked || state.sync != sync) return;
    std::array<char, PATH_MAX> path{}; auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
    auto const size = ::readlink(descriptor.c_str(), path.data(), path.size());
    if (size <= 0 || std::string_view(path.data(), size_t(size)) != state.path) return;
    state.blocked = true; state.ready.notify_all();
    if (!state.ready.wait_for(lock, std::chrono::seconds(10), [&] { return state.released; })) state.timed_out = true;
}
extern "C" size_t __real_POS2_FSE_compress_usingCTable(void*, size_t, void const*, size_t, FSE_CTable const*);
extern "C" size_t __wrap_POS2_FSE_compress_usingCTable(void* dst, size_t capacity, void const* src, size_t size, FSE_CTable const* table)
{
    if (fail_next_codec.exchange(false)) return 0;
    {
        std::unique_lock lock(target_pipeline_barrier.mutex); auto& state = target_pipeline_barrier;
        if (state.armed && state.enabled && state.calls++ == 2)
        {
            state.codec_blocked = true; state.ready.notify_all();
            if (!state.ready.wait_for(lock, std::chrono::seconds(10), [&] { return state.released; })) state.timed_out = true;
        }
    }
    {
        std::unique_lock lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
        if (state.armed && state.enabled && state.calls++ == state.hold)
        {
            state.blocked = true; state.ready.notify_all();
            if (!state.ready.wait_for(lock, std::chrono::seconds(10), [&] { return state.released; })) state.timed_out = true;
        }
    }
    return __real_POS2_FSE_compress_usingCTable(dst, capacity, src, size, table);
}
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* bytes, size_t count, off_t offset)
{
    {
        std::lock_guard lock(summary_read_observer.mutex); auto& state = summary_read_observer;
        if (state.armed && offset >= 64)
        {
            std::array<char, PATH_MAX> path{}; auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
            auto const n = ::readlink(descriptor.c_str(), path.data(), path.size());
            std::string_view const name(path.data(), n > 0 ? size_t(n) : 0);
            if (name.find(state.prefix) != std::string_view::npos)
            {
                state.bytes += count; state.maximum = std::max<uint64_t>(state.maximum, count);
                if (state.fail) { state.fail = false; errno = EIO; return -1; }
            }
        }
    }
    if (observe_oversized_partial.load())
    {
        std::array<char, PATH_MAX> path{}; auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
        auto const n = ::readlink(descriptor.c_str(), path.data(), path.size());
        if (n > 0 && std::string_view(path.data(), size_t(n)).ends_with("/checkpoint.partial.foreign")) ++oversized_partial_reads;
    }
    {
        std::unique_lock lock(target_pipeline_barrier.mutex); auto& state = target_pipeline_barrier;
        if (state.armed && state.enabled && uint64_t(offset) >= state.read_offset)
        {
            if (!state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.codec_blocked && state.write_blocked; })) state.timed_out = true;
            state.read_observed = true; state.ready.notify_all();
            if (state.fail_read) { errno = EIO; return -1; }
        }
    }
    {
        std::unique_lock lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
        if (state.armed && state.enabled && state.read && uint64_t(offset) >= state.read_offset)
        {
            if (!state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.blocked; })) state.timed_out = true;
            state.observed = true; state.ready.notify_all();
            if (state.fail) { errno = EIO; return -1; }
        }
    }
    return __real_pread(fd, bytes, count, offset);
}
extern "C" int __real_renameat2(int, char const*, int, char const*, unsigned);
extern "C" int __wrap_renameat2(int olddir, char const* oldpath, int newdir, char const* newpath, unsigned flags)
{
    retirement_flags = flags;
    if (reject_atomic_retirement.exchange(false)) { errno = ENOSYS; return -1; }
    return __real_renameat2(olddir, oldpath, newdir, newpath, flags);
}
extern "C" int __real_fsetxattr(int, char const*, void const*, size_t, int);
extern "C" int __wrap_fsetxattr(int fd, char const* name, void const* value, size_t size, int flags)
{
    if (reject_retired_receipt.exchange(false)) { errno = EOPNOTSUPP; return -1; }
    return __real_fsetxattr(fd, name, value, size, flags);
}
extern "C" ssize_t __real_pwrite(int, void const*, size_t, off_t);
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd)
{
    hold_shared_storage(fd, true);
    {
        std::lock_guard lock(segment_sync_failure.mutex); auto& state = segment_sync_failure;
        if (state.armed && (!state.after_payload || state.payload))
        {
            std::array<char, PATH_MAX> path{}; auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
            auto const n = ::readlink(descriptor.c_str(), path.data(), path.size());
            std::string_view const name(path.data(), n > 0 ? size_t(n) : 0);
            if (name.find(state.path) != std::string_view::npos)
            {
                if (state.remaining) --state.remaining;
                else { state.armed = false; errno = EIO; return -1; }
            }
        }
    }
    return __real_fsync(fd);
}
extern "C" ssize_t __wrap_pwrite(int fd, void const* bytes, size_t count, off_t offset)
{
    hold_shared_storage(fd, false);
    {
        std::unique_lock lock(prepared_write_barrier.mutex); auto& state = prepared_write_barrier;
        if (state.armed && offset >= 64)
        {
            state.armed = false; state.blocked = true; state.ready.notify_all();
            if (!state.ready.wait_for(lock, std::chrono::seconds(10), [&] { return state.released; })) state.timed_out = true;
        }
    }

    if (interrupt_next_write.exchange(false)) { errno = EINTR; return -1; }
    if (observe_index_writes.load() && offset >= 64)
    {
        std::array<char, PATH_MAX> path{}; auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
        auto const n = ::readlink(descriptor.c_str(), path.data(), path.size());
        if (n > 0 && std::string_view(path.data(), size_t(n)).find("/index.") != std::string_view::npos)
        {
            largest_index_write = std::max(largest_index_write.load(), uint64_t(count));
            if (count > 65536) if (int const error = index_write_error.exchange(0)) { errno = error; return -1; }
        }
    }
    {
        std::lock_guard lock(segment_sync_failure.mutex); auto& state = segment_sync_failure;
        if (state.armed && state.after_payload && offset >= 64)
        {
            std::array<char, PATH_MAX> path{}; auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
            auto const n = ::readlink(descriptor.c_str(), path.data(), path.size());
            std::string_view const name(path.data(), n > 0 ? size_t(n) : 0);
            if (name.find(state.path) != std::string_view::npos) state.payload = true;
        }
    }
    {
        std::lock_guard lock(summary_write_observer.mutex); auto& state = summary_write_observer;
        if (state.armed && offset >= 64)
        {
            std::array<char, PATH_MAX> path{};
            auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
            auto const n = ::readlink(descriptor.c_str(), path.data(), path.size());
            std::string_view const name(path.data(), n > 0 ? size_t(n) : 0);
            if (name.find(state.namespace_name + "/run-1-") != std::string_view::npos)
            { state.bytes += count; state.ready.notify_all(); }
        }
    }
    {
        std::unique_lock lock(target_pipeline_barrier.mutex); auto& state = target_pipeline_barrier;
        if (state.armed && !offset) state.enabled = true;
        if (state.armed && state.observe_stage)
        {
            std::array<char, PATH_MAX> path{};
            auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
            auto const n = ::readlink(descriptor.c_str(), path.data(), path.size());
            std::string_view const name(path.data(), n > 0 ? size_t(n) : 0);
            if (offset >= 64 && name.find("/.xchplot2-external-") != std::string_view::npos && name.find("/run-") != std::string_view::npos)
            { state.stage_bytes += count; state.ready.notify_all(); }
            if (name.starts_with(state.stage_checkpoint_prefix)) { state.stage_checkpoint = true; state.ready.notify_all(); }
        }
        struct stat file{};
        if (state.armed && state.enabled && !::fstat(fd, &file) &&
            file.st_dev == state.target_device && file.st_ino == state.target_inode)
        {
            state.write_blocked = true; state.ready.notify_all();
            if (!state.ready.wait_for(lock, std::chrono::seconds(10), [&] { return state.released; })) state.timed_out = true;
            if (state.fail_write) { errno = EIO; return -1; }
        }
    }
    {
        std::unique_lock lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
        // Reservation/index headers separate sizing from real pipeline work.
        if (state.armed && !offset) state.enabled = true;
        if (state.armed && !state.read && uint64_t(offset) >= state.write_offset && offset)
        {
            if (!state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.blocked; })) state.timed_out = true;
            state.observed = true; state.ready.notify_all();
            if (state.fail) { errno = EIO; return -1; }
        }
    }
    if (slow_checkpoint_writes.load())
    {
        std::array<char, PATH_MAX> path{};
        auto const descriptor = "/proc/self/fd/" + std::to_string(fd);
        auto const size = ::readlink(descriptor.c_str(), path.data(), path.size());
        if (size > 0 && std::string_view(path.data(), size_t(size)).find("/checkpoint.partial.") != std::string_view::npos)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (fail_next_write.load())
    {
        auto const remaining = writes_before_failure.load();
        if (remaining) writes_before_failure = remaining - 1;
        else if (fail_next_write.exchange(false)) { errno = EIO; return -1; }
    }
    return __real_pwrite(fd, bytes, count, offset);
}
#endif

void io_trace(bool enabled)
{
    require(!::setenv("POS2GPU_PHASE_TIMING", enabled ? "1" : "0", 1), "cannot select I/O trace test mode");
    std::string pattern = (fs::temp_directory_path() / "grow io trace-XXXXXX").string();
    external_detail::Descriptor file(::mkstemp(pattern.data()));
    struct Cleanup { std::string path; ~Cleanup() { ::unlink(path.c_str()); } } cleanup{pattern};
    std::ostringstream output;
    struct Capture
    {
        std::streambuf* previous;
        explicit Capture(std::streambuf* buffer) : previous(std::cerr.rdbuf(buffer)) {}
        ~Capture() { std::cerr.rdbuf(previous); }
    } capture(output.rdbuf());
    std::string const owner = "trace owner";
    std::array<uint8_t, 4> bytes{1, 2, 3, 4}, read{};
    Stats stats; size_t expected_writes = 0;
    constexpr std::array roles{"contribution", "checkpoint_manifest", "checkpoint_summary", "scratch_ownership", "final_output", "index"};
    for (auto role : roles)
    {
        external_detail::write(file.fd, 0, bytes, &stats, {role, &owner}); ++expected_writes;
        external_detail::sync(file.fd, &stats, {role, &owner});
    }
#ifdef GROW_TEST_WRAP_PWRITE
    interrupt_next_write = true;
    external_detail::write(file.fd, 0, bytes, &stats, {"final_output", &owner}); expected_writes += 2;
#endif
    external_detail::read(file.fd, 0, read, &stats, {"final_output", &owner});
    require(read == bytes, "I/O tracing changed file bytes");
    auto fails = [](auto operation) { try { operation(); } catch (std::runtime_error const&) { return true; } return false; };
    require(fails([&] { external_detail::write(-1, 0, bytes, &stats, {"final_output", &owner}); }), "traced write error disappeared"); ++expected_writes;
    require(fails([&] { external_detail::read(-1, 0, read, &stats, {"final_output", &owner}); }), "traced read error disappeared");
    require(fails([&] { external_detail::sync(-1, &stats, {"final_output", &owner}); }), "traced sync error disappeared");
    auto gate = std::make_shared<std::recursive_mutex>();
    std::promise<void> locked; auto ready = locked.get_future();
    std::jthread holder([&] { std::lock_guard guard(*gate); locked.set_value(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
    ready.get();
    {
        auto guard = external_detail::target_lock(gate, file.fd, owner);
        require(guard.owns_lock(), "traced target gate was not acquired");
        auto recursive = external_detail::target_lock(gate, file.fd, owner);
        require(recursive.owns_lock(), "traced recursive target gate changed ownership");
    }
    require(!external_detail::target_lock({}, file.fd, owner).owns_lock(), "absent target gate acquired a lock");
    errno = EDOM;
    external_detail::IOTrace unavailable(-1, "diagnostic", {"test", &owner});
    require(errno == EDOM, "diagnostic identity lookup changed errno");
    auto const now = std::chrono::steady_clock::now(); unavailable.report(now, now, 0, 0);
    require(errno == EDOM, "diagnostic report changed errno");
    require(stats.write_calls == expected_writes && stats.read_calls == 2 && stats.syncs == roles.size() + 1,
        "tracing changed syscall attempt counters");
    require(stats.written_bytes == bytes.size() * (roles.size()
#ifdef GROW_TEST_WRAP_PWRITE
        + 1
#endif
        ) && stats.read_bytes == bytes.size(), "tracing changed successful byte counters");
    auto const text = output.str();
    if (!enabled) { require(text.empty(), "disabled I/O tracing emitted diagnostics"); return; }
    size_t writes = 0, reads = 0, syncs = 0, locks = 0, errors = 0, interrupts = 0;
    std::istringstream lines(text); std::string line;
    while (std::getline(lines, line))
    {
        require(line.starts_with("[grow-io-timing] ") && line.find("owner=\"trace owner\"") != std::string::npos,
            "I/O trace record lost its owner or atomic line boundary");
        auto field = [&](std::string const& name)
        {
            auto const first = line.find(" " + name + "="); require(first != std::string::npos, "missing I/O trace field");
            return std::stoull(line.substr(first + name.size() + 2));
        };
        require(field("end_ns") >= field("begin_ns"), "I/O trace has a reversed interval");
#ifdef __linux__
        require(field("tid") == uint64_t(::getpid()), "I/O trace lost its Linux task identity");
#endif
        writes += line.find("op=pwrite ") != std::string::npos;
        reads += line.find("op=pread ") != std::string::npos;
        syncs += line.find("op=fsync ") != std::string::npos;
        locks += line.find("op=target_lock ") != std::string::npos;
        errors += field("error") == EBADF; interrupts += field("error") == EINTR;
    }
    require(writes == stats.write_calls && reads == stats.read_calls && syncs == stats.syncs && locks == 2 && errors == 3,
        "I/O trace omitted or duplicated syscall/target-lock samples");
#ifdef GROW_TEST_WRAP_PWRITE
    require(interrupts == 1, "I/O trace omitted the interrupted write attempt");
#endif
    for (auto role : roles) require(text.find(std::string("role=") + role + " ") != std::string::npos, "I/O role attribution disappeared");
    require(text.find("path=\"" + pattern + "\"") != std::string::npos && text.find("inode=0 ") != std::string::npos,
        "I/O trace lost its resolved or unavailable file identity");
}

void checkpoint_buffer()
{
    // The bounded sink must retain the old ostream field and checksum bytes,
    // including native padding, embedded zeroes and an empty summary table.
    std::ostringstream old(std::ios::binary); old.exceptions(std::ios::badbit | std::ios::failbit);
    external_detail::CheckpointBuffer buffer(8192);
    auto fields = [&](auto& out)
    {
        external_detail::put_string(out, "xchplot2-external-2");
        external_detail::put_string(out, std::string_view("memo\0bytes", 10));
        external_detail::put(out, uint64_t(UINT64_MAX)); external_detail::put(out, uint16_t(65535));
        external_detail::put(out, uint32_t(17)); external_detail::put(out, uint64_t(0));
        std::array<uint8_t, 4096> summary{};
        for (size_t i = 0; i < summary.size(); ++i) summary[i] = uint8_t(i);
        out.write(reinterpret_cast<char const*>(summary.data()), summary.size());
        out.write(reinterpret_cast<char const*>(summary.data()), 0);
    };
    fields(old); fields(buffer);
    auto const body = old.str(); auto const expected = group_tools::hex(grow_digest({reinterpret_cast<uint8_t const*>(body.data()), body.size()})) + '\n' + body;
    auto const record = buffer.record();
    require(record.size() == expected.size() && std::equal(record.begin(), record.end(), reinterpret_cast<uint8_t const*>(expected.data())),
        "bounded external checkpoint serialization changed persisted bytes");
    external_detail::CheckpointBuffer exact(record.size()); fields(exact);
    require(std::ranges::equal(exact.record(), record), "exact external checkpoint buffer rejected its boundary");
    bool rejected = false;
    try { external_detail::CheckpointBuffer short_buffer(record.size() - 1); fields(short_buffer); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "external checkpoint buffer exceeded its bound");

    external_detail::CheckpointInput view(body);
    require(view.rdbuf()->in_avail() == std::streamsize(body.size()) &&
        external_detail::get_string(view) == "xchplot2-external-2" &&
        external_detail::get_string(view) == std::string_view("memo\0bytes", 10) &&
        external_detail::get<uint64_t>(view) == UINT64_MAX && external_detail::get<uint16_t>(view) == UINT16_MAX &&
        external_detail::get<uint32_t>(view) == 17 && !external_detail::get<uint64_t>(view),
        "bounded external checkpoint input changed field parsing");
    view.ignore(4096);
    require(!view.rdbuf()->in_avail(), "bounded external checkpoint input exposed bytes outside its body");
    rejected = false;
    try { (void)external_detail::get<uint8_t>(view); }
    catch (std::ios_base::failure const&) { rejected = true; }
    require(rejected, "bounded external checkpoint input accepted a truncated field");
    std::string shared = "a"; external_detail::CheckpointInput shared_view(shared); shared[0] = 'b';
    require(external_detail::get<char>(shared_view) == 'b', "external checkpoint input copied its record");
}

void packing_io_policy()
{
    StorageDevice const ssd{"ssd", {"fast"}, true, false}, hdd{"hdd", {"slow"}, false, true};
    StorageDevice const other_ssd{"other-ssd", {"other-fast"}, true, false};
    StorageDevice const ssd_alias{"ssd-alias", {"fast", "other-fast"}, true, false};
    StorageDevice const shared{"shared", {"fast", "slow"}, false, false}, unknown{"unknown", {}};
    require(!external_detail::independent_packing_io(ssd, {&ssd, 1}), "shared SSD packing admitted competing I/O");
    require(external_detail::independent_packing_io(ssd, {&other_ssd, 1}), "independent SSD packing drives were serialized");
    require(!external_detail::independent_packing_io(ssd, {&ssd_alias, 1}) &&
        !external_detail::independent_packing_io(ssd_alias, {&other_ssd, 1}), "composite SSD aliases admitted competing packing I/O");
    std::array const sources{other_ssd, ssd};
    require(!external_detail::independent_packing_io(ssd, sources), "a later shared packing source escaped device admission");
    require(external_detail::independent_packing_io(hdd, {&ssd, 1}), "independent packing drives were serialized");
    require(!external_detail::independent_packing_io(hdd, {&hdd, 1}), "shared HDD packing admitted competing I/O");
    require(!external_detail::independent_packing_io(hdd, {&shared, 1}), "partially shared packing drives admitted competing I/O");
    require(!external_detail::independent_packing_io(unknown, {&ssd, 1}) &&
        !external_detail::independent_packing_io(ssd, {&unknown, 1}), "unknown packing topology admitted competing I/O");
}

Identity initial_identity(uint8_t k = 18)
{
    Identity value; value.header.magic = PlotGroupFile::MAGIC; value.header.version = PlotGroupFile::FORMAT_VERSION;
    value.header.group_id = PlotGroupId(std::string(64, 'a')); value.header.k = k; value.header.strength = 2;
    value.memo = {0, 17, 255}; value.header.memo_length = uint8_t(value.memo.size());
    value.header.chunk_index_offset = sizeof(Header) + value.memo.size(); return value;
}
std::vector<std::vector<uint64_t>> members(size_t count, uint64_t fragments = 8, uint8_t k = 18)
{
    std::vector<std::vector<uint64_t>> result(count);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k), range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t member = 0; member < count; ++member)
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
            for (uint64_t i = 0; i < fragments; ++i)
            {
                uint64_t const value = chunk * range + i * (range / fragments) + member + 7;
                result[member].push_back(value); if (i == 3) result[member].push_back(value);
            }
    return result;
}
std::vector<uint8_t> contents(fs::path const& path)
{ auto in = group_tools::input(path.string()); return {std::istreambuf_iterator<char>(in), {}}; }
Config config()
{
    Config value; value.capacity_bytes = 64ull * 1024 * 1024; value.window_bytes = 32768;
    value.codec_bytes = 8ull * 1024 * 1024; value.checkpoint_bytes = 65536; value.io_overlap = false; value.workers = 3; return value;
}
ExternalConfig external(fs::path const& root)
{
    return {{{(root / "scratch-a").string(), 16ull * 1024 * 1024}, {(root / "scratch-b").string(), 16ull * 1024 * 1024}}, 32768, 2};
}
Stats complete(fs::path const& path, Members input, ExternalConfig ext, bool resume, Config cfg = config())
{
    ExternalEngine engine(path.string(), cfg, ext, resume, initial_identity()); engine.confirm_identity();
    for (size_t i = engine.committed_member_count(); engine.can_append() && i < input.size(); ++i)
        require(engine.append(input.subspan(i, 1)).accepted, "external fixture append rejected");
    engine.finish(); return engine.stats();
}
std::string ownership_token(fs::path const& path)
{
    auto const checkpoint = contents(fs::path(path.string() + ".external") / "checkpoint");
    std::istringstream in(std::string(checkpoint.begin() + 65, checkpoint.end()), std::ios::binary);
    in.exceptions(std::ios::badbit | std::ios::failbit);
    for (unsigned i = 0; i < 3; ++i) (void)external_detail::get_string(in);
    return external_detail::get_string(in);
}
fs::path retired_path(fs::path const& path)
{ return fs::path(path.string() + ".external.retired-" + ownership_token(path)); }
std::vector<fs::path> summary_segments(fs::path const& path, ExternalConfig const& ext, uint64_t id)
{
    std::vector<fs::path> result;
    for (auto const& grant : ext.scratch)
        for (auto const& entry : fs::directory_iterator(fs::path(grant.directory) / (".xchplot2-external-" + ownership_token(path))))
            if (entry.path().filename().string().starts_with("run-" + std::to_string(id) + "-")) result.push_back(entry.path());
    return result;
}
void bulk_checkpoints(fs::path const& root, Members input, ExternalConfig ext)
{
    auto const path = root / "bulk-checkpoint";
    auto const checkpoint = fs::path(path.string() + ".external") / "checkpoint";
    auto const chunks = PlotGroupFile::getChunkCountForK(18);
    auto const table_bytes = chunks * (sizeof(ChunkSummary) + sizeof(uint64_t));
    std::vector<uint64_t> prefix_reservation;
    {
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const initial = ExternalEngine::inspect(path.string());
        require(std::ranges::all_of(initial.reserved_bytes, [](uint64_t bytes) { return !bytes; }) &&
            fs::file_size(checkpoint) < table_bytes, "initial external ownership manifest allocated a bulk summary");
        require(engine.append(input.subspan(0, 1)).accepted, "bulk summary fixture rejected");
        auto const saved = ExternalEngine::inspect(path.string());
        prefix_reservation = saved.reserved_bytes;
        auto const old = summary_segments(path, ext, 1);
        require(!old.empty() && fs::file_size(checkpoint) < table_bytes && saved.members == 1 &&
            saved.reserved_bytes == engine.scratch_reserved_bytes(), "bulk summary was inline or escaped scratch accounting");
        bool rejected = false;
        try { (void)ExternalEngine::inspect(path.string(), fs::file_size(checkpoint) * 8); }
        catch (std::exception const&) { rejected = true; }
        require(rejected, "small manifest inspection omitted its summary hashing buffer admission");
        require(engine.append(input.subspan(1, 1)).accepted, "bulk summary replacement rejected");
        require(std::ranges::none_of(old, [](fs::path const& part) { return fs::exists(part); }) &&
            !summary_segments(path, ext, 3).empty(), "durable manifest did not replace and reclaim its prior summary generation");
    }
    auto const parts = summary_segments(path, ext, 3); require(!parts.empty(), "bulk summary generation is missing");
#ifdef GROW_TEST_WRAP_PWRITE
    {
        auto const partial = fs::path(path.string() + ".external") / "checkpoint.partial.foreign";
        { std::ofstream out(partial, std::ios::binary); out << "foreign oversized partial"; }
        auto const bytes = ExternalEngine::metadata_bytes(18, ext) / 2 - chunks * sizeof(uint64_t) + 1;
        fs::resize_file(partial, bytes); oversized_partial_reads = 0; observe_oversized_partial = true;
        { ExternalEngine recovered(path.string(), config(), ext, true); }
        observe_oversized_partial = false;
        require(!oversized_partial_reads && fs::file_size(partial) == bytes,
            "orphan partial inspection allocated beyond the manifest bound or changed a foreign file");
        fs::remove(partial);
    }
#endif
    {
        ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity(); engine.start_packing();
        require(std::ranges::none_of(parts, [](fs::path const& part) { return fs::exists(part); }),
            "packing retained its durable staging summary");
        engine.finish(); require(engine.release_completed(), "completed bulk summary retained scratch files");
    }
    {
        auto const expected = root / "bulk-reference";
        Engine engine(expected.string(), config(), false, initial_identity()); engine.confirm_identity();
        require(engine.append(input.subspan(0, 2)).accepted && contents(path) == contents(expected),
            "bulk summary persistence changed ordinary v2 output");
    }
    {
        auto tight = ext;
        for (size_t i = 0; i < tight.scratch.size(); ++i) tight.scratch[i].capacity_bytes = prefix_reservation[i];
        auto const tight_path = root / "bulk-pair-tight", expected = root / "bulk-pair-reference";
        ExternalEngine engine(tight_path.string(), config(), tight, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input.subspan(0, 1)).accepted && engine.stats().sized_chunks && engine.stats().reclaimed_bytes,
            "summary-pair admission omitted exact-size retry after reclaiming its conservative reservation");
        require(!engine.append(input.subspan(1, 1)).accepted && engine.committed_member_count() == 1,
            "summary-pair capacity exhaustion acknowledged an unreserved generation");
        engine.finish(); require(engine.release_completed(), "summary-pair capacity rejection retained provisional files");
        Engine reference(expected.string(), config(), false, initial_identity()); reference.confirm_identity();
        require(reference.append(input.subspan(0, 1)).accepted && contents(tight_path) == contents(expected),
            "summary-pair capacity rejection changed its acknowledged ordinary prefix");
    }
    // Planning changes only an operation-local candidate. A read or late
    // summary failure must preserve the saved prefix and allow the same
    // engine to retry, including exact final sizing against prior source bytes.
    {
        auto const retry_path = root / "bulk-candidate-retry";
        auto const record_path = fs::path(retry_path.string() + ".external") / "checkpoint";
        auto cfg = config(); cfg.io_overlap = true;
        // Vary the final region until the complete upstream index has a
        // partial final byte. Rounding each independent size field would then
        // reject this fixture at its exact capacity.
        std::vector<std::vector<uint64_t>> retry_raw(2);
        for (size_t member = 0; member < retry_raw.size(); ++member)
        { auto const raw = input[member].raw(); retry_raw[member].assign(raw.begin(), raw.end()); }
        uint64_t const range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        ChunkCodec codec(18, cfg.codec_bytes);
        auto chunk_size = [&](uint64_t chunk)
        {
            std::array<FragmentRegion, 2> regions;
            for (size_t member = 0; member < retry_raw.size(); ++member)
            {
                auto const& raw = retry_raw[member];
                auto const first = std::lower_bound(raw.begin(), raw.end(), chunk * range);
                auto const last = std::lower_bound(first, raw.end(), (chunk + 1) * range);
                regions[member] = std::span<uint64_t const>(first, last);
            }
            return uint64_t(codec.encoded_size(chunk, {}, 0, regions));
        };
        std::vector<uint64_t> sizes(size_t(chunks), 0);
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) sizes[size_t(chunk)] = chunk_size(chunk);
        BitWriter canonical; gsz_encode(sizes, uint16_t(retry_raw.size()), canonical);
        for (unsigned added = 0; !(canonical.bitCount() % 8) && added < 256; ++added)
        {
            retry_raw.back().push_back(retry_raw.back().back() + 1);
            sizes.back() = chunk_size(chunks - 1);
            canonical.clear(); gsz_encode(sizes, uint16_t(retry_raw.size()), canonical);
        }
        require(canonical.bitCount() % 8, "candidate exact fixture has no partial canonical index byte");
        Members retry_input(retry_raw);
        auto const expected = root / "bulk-candidate-reference";
        { Engine reference(expected.string(), cfg, false, initial_identity()); reference.confirm_identity();
          require(reference.append(retry_input).accepted, "candidate exact reference rejected"); }
        cfg.capacity_bytes = fs::file_size(expected) + ExternalEngine::metadata_bytes(18, ext);
        cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext);
        {
            ExternalEngine engine(retry_path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
            require(engine.append(retry_input.subspan(0, 1)).accepted, "candidate retry prefix rejected");
        }
        ExternalEngine engine(retry_path.string(), cfg, ext, true); engine.confirm_identity();
        auto const before = contents(retry_path), record = contents(record_path);
        auto const reserved = engine.scratch_reserved_bytes();
#ifdef GROW_TEST_WRAP_PWRITE
        {
            std::lock_guard lock(summary_read_observer.mutex); auto& state = summary_read_observer;
            state.prefix = ".xchplot2-external-" + ownership_token(retry_path) + "/run-1-";
            state.bytes = state.maximum = 0; state.armed = state.fail = true;
        }
        bool failed = false;
        try { (void)engine.append(retry_input.subspan(1, 1)); } catch (std::runtime_error const&) { failed = true; }
        {
            std::lock_guard lock(summary_read_observer.mutex); auto& state = summary_read_observer;
            state.armed = false;
            require(failed && !state.fail && state.bytes && state.maximum <= 65536,
                "candidate summary read did not fail within its bounded buffer");
        }
        require(engine.committed_member_count() == 1 && contents(retry_path) == before && contents(record_path) == record &&
            engine.scratch_reserved_bytes() == reserved, "failed candidate summary read changed its committed prefix");
#endif
        auto const source = retry_input[1].raw(); std::vector<uint64_t> damaged(source.begin(), source.end());
        damaged.erase(std::lower_bound(damaged.begin(), damaged.end(), (chunks - 1) * range), damaged.end());
        std::vector<Member> invalid; invalid.emplace_back(std::move(damaged)); invalid.back().compact(18);
        bool rejected = false;
        try { (void)engine.append(invalid); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected && engine.committed_member_count() == 1 && contents(retry_path) == before && contents(record_path) == record &&
            engine.scratch_reserved_bytes() == reserved, "late candidate planning failure changed its committed prefix");
#ifdef GROW_TEST_WRAP_PWRITE
        {
            std::lock_guard lock(summary_read_observer.mutex); auto& state = summary_read_observer;
            state.bytes = state.maximum = 0; state.armed = true;
        }
#endif
        auto const sized = engine.stats().sized_chunks;
        auto const appended = engine.append(retry_input.subspan(1, 1));
        require(appended.accepted && appended.attempted_final_bytes == fs::file_size(expected) && engine.stats().sized_chunks > sized,
            "candidate summary retry did not preserve exact prior source sizes");
#ifdef GROW_TEST_WRAP_PWRITE
        {
            std::lock_guard lock(summary_read_observer.mutex); auto& state = summary_read_observer;
            state.armed = false;
            require(state.bytes == table_bytes && state.maximum <= 65536,
                "candidate summary was not reloaded through one bounded authenticated pass");
        }
#endif
        engine.finish(); require(engine.release_completed() && contents(retry_path) == contents(expected),
            "candidate summary retry changed ordinary output or retained scratch");
        auto denied = cfg; --denied.capacity_bytes;
        auto const denied_path = root / "bulk-candidate-capacity";
        auto const denied_record = fs::path(denied_path.string() + ".external") / "checkpoint";
        ExternalEngine limited(denied_path.string(), denied, ext, false, initial_identity()); limited.confirm_identity();
        require(limited.append(retry_input.subspan(0, 1)).accepted, "candidate capacity prefix rejected");
        auto const prefix = contents(denied_path), prefix_record = contents(denied_record);
        auto const prefix_reserved = limited.scratch_reserved_bytes();
        require(!limited.append(retry_input.subspan(1, 1)).accepted && limited.committed_member_count() == 1 &&
            contents(denied_path) == prefix && contents(denied_record) == prefix_record && limited.scratch_reserved_bytes() == prefix_reserved,
            "candidate exact capacity rejection changed its committed prefix");
        limited.finish(); require(limited.release_completed() && contents(denied_path) == contents(root / "bulk-pair-reference"),
            "candidate exact capacity rejection changed ordinary prefix bytes");
    }
    for (std::string const boundary : {"external-summary-reserved:2", "external-summary-flushed:2",
        "external-run-before-checkpoint:2", "external-run-after-checkpoint:2", "external-summary-retired"})
    {
        auto const interrupted = root / ("bulk-" + boundary); auto const child = ::fork(); require(child >= 0, "fork failed");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", boundary.c_str(), 1);
            (void)complete(interrupted, input.subspan(0, 2), ext, false); std::_Exit(0);
        }
        int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "bulk summary recovery boundary was not reached: " + boundary);
        auto const saved = ExternalEngine::inspect(interrupted.string());
        {
            ExternalEngine engine(interrupted.string(), config(), ext, true);
            require(saved.reserved_bytes == engine.scratch_reserved_bytes(), "unconfirmed bulk generations escaped scratch accounting");
            engine.confirm_identity();
            for (size_t member = engine.committed_member_count(); member < 2; ++member)
                require(engine.append(input.subspan(member, 1)).accepted, "bulk summary recovery rejected the unacknowledged suffix");
            engine.finish(); require(engine.release_completed(), "bulk summary recovery retained owned scratch");
        }
        require(contents(interrupted) == contents(path), "bulk summary crash recovery changed ordinary output: " + boundary);
    }
    for (std::string const damage : {"replacement", "missing", "checksum", "ownership", "payload"})
    {
        auto const damaged_path = root / ("bulk-" + damage);
        auto const record_path = fs::path(damaged_path.string() + ".external") / "checkpoint";
        {
            ExternalEngine engine(damaged_path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
            require(engine.append(input.subspan(0, 1)).accepted, "damaged bulk summary fixture rejected");
        }
        auto damaged_parts = summary_segments(damaged_path, ext, 1);
        require(!damaged_parts.empty(), "damaged bulk summary generation is missing");
        auto const before = contents(damaged_path);
        if (damage == "missing" || damage == "replacement")
        {
            fs::rename(damaged_parts.front(), damaged_parts.front().string() + ".held");
            if (damage == "replacement") { std::ofstream replacement(damaged_parts.front(), std::ios::binary); replacement << "unowned replacement"; }
        }
        else if (damage == "ownership" || damage == "payload")
        {
            std::fstream file(damaged_parts.front(), std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(damage == "ownership" ? 0 : 64); file.put('x'); file.flush();
        }
        else
        {
            // Change only the manifest's summary digest, retaining valid
            // file stamps and the manifest envelope checksum.
            std::ranges::sort(damaged_parts, {}, [](fs::path const& part)
            { auto const name = part.filename().string(); return std::stoul(name.substr(name.rfind('-') + 1)); });
            std::vector<uint8_t> table; table.reserve(size_t(table_bytes));
            for (auto const& part : damaged_parts)
            {
                auto const file = contents(part); size_t const bytes = std::min(file.size() - 64, size_t(table_bytes) - table.size());
                table.insert(table.end(), file.begin() + 64, file.begin() + 64 + bytes);
            }
            require(table.size() == table_bytes, "bulk checksum fixture lost table coverage");
            auto const digest = grow_digest(table); auto record = contents(record_path);
            auto const found = std::search(record.begin() + 65, record.end(), digest.begin(), digest.end());
            require(found != record.end(), "bulk checksum fixture has no digest reference"); *found ^= 1;
            auto const envelope = group_tools::hex(grow_digest(std::span<uint8_t const>(record).subspan(65)));
            std::copy(envelope.begin(), envelope.end(), record.begin());
            std::ofstream out(record_path, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<char const*>(record.data()), record.size());
        }
        auto const saved_record = contents(record_path); bool rejected = false;
        try { (void)ExternalEngine::inspect(damaged_path.string()); } catch (std::exception const&) { rejected = true; }
        require(rejected && contents(damaged_path) == before && contents(record_path) == saved_record,
            "inspection accepted or modified a damaged bulk summary: " + damage);
        rejected = false;
        try { ExternalEngine engine(damaged_path.string(), config(), ext, true); } catch (std::exception const&) { rejected = true; }
        require(rejected && contents(damaged_path) == before && contents(record_path) == saved_record,
            "resume accepted or modified a damaged bulk summary: " + damage);
        auto const directory = ".xchplot2-external-" + ownership_token(damaged_path);
        for (auto const& grant : ext.scratch) fs::remove_all(fs::path(grant.directory) / directory);
        fs::remove_all(damaged_path.string() + ".external"); fs::remove(damaged_path);
    }
#ifdef GROW_TEST_WRAP_PWRITE
    struct FailureCase { char const* name; uint64_t run; unsigned skip; bool payload; };
    for (auto const& failure : {FailureCase{"contribution-first", 2, 0, false}, FailureCase{"contribution-later", 2, 1, false},
        FailureCase{"summary-first", 3, 0, false}, FailureCase{"summary-later", 3, 1, false}, FailureCase{"summary-body", 3, 0, true}})
    {
        auto const failed_path = root / (std::string("bulk-sync-") + failure.name);
        std::vector<uint64_t> before;
        {
            ExternalEngine engine(failed_path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
            require(engine.append(input.subspan(0, 1)).accepted, "bulk sync failure fixture rejected"); before = engine.scratch_reserved_bytes();
            {
                std::lock_guard lock(segment_sync_failure.mutex); auto& state = segment_sync_failure;
                state.path = ".xchplot2-external-" + ownership_token(failed_path) + "/run-" + std::to_string(failure.run) + "-";
                state.remaining = failure.skip; state.after_payload = failure.payload; state.payload = false; state.armed = true;
            }
            bool rejected = false;
            try { (void)engine.append(input.subspan(1, 1)); } catch (std::exception const&) { rejected = true; }
            { std::lock_guard lock(segment_sync_failure.mutex); require(!segment_sync_failure.armed, "bulk segment sync failure was not reached"); }
            auto const saved = ExternalEngine::inspect(failed_path.string());
            require(rejected && saved.members == 1 && saved.reserved_bytes == engine.scratch_reserved_bytes(),
                "failed bulk sync changed acknowledged members or lost physical scratch charges");
            for (size_t i = 0; i < before.size(); ++i)
                require(saved.reserved_bytes[i] >= before[i] && (failure.run != 2 || saved.reserved_bytes[i] == before[i]),
                    "failed reservation subtracted a previous contribution or summary grant");
        }
        {
            ExternalEngine engine(failed_path.string(), config(), ext, true); engine.confirm_identity();
            require(engine.scratch_reserved_bytes() == before && engine.append(input.subspan(1, 1)).accepted,
                "bulk sync recovery did not reclaim only its unacknowledged generation");
            engine.finish(); require(engine.release_completed() && contents(failed_path) == contents(path),
                "bulk sync recovery changed ordinary bytes or retained owned scratch");
        }
    }
    // Scratch metadata must finish while another target owner holds its gate;
    // only publication of the small manifest waits for that resource.
    auto const gated_path = root / "bulk-gated";
    ext.target_gate = std::make_shared<std::recursive_mutex>();
    ExternalEngine engine(gated_path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
    {
        std::lock_guard lock(summary_write_observer.mutex); auto& state = summary_write_observer;
        state.namespace_name = ".xchplot2-external-" + ownership_token(gated_path); state.bytes = 0; state.armed = true;
    }
    std::unique_lock gate(*ext.target_gate);
    auto operation = std::async(std::launch::async, [&] { return engine.append(input.subspan(0, 1)).accepted; });
    bool observed;
    {
        std::unique_lock lock(summary_write_observer.mutex); auto& state = summary_write_observer;
        observed = state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.bytes == table_bytes; });
    }
    bool const waiting = operation.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready;
    gate.unlock(); bool const accepted = operation.get();
    { std::lock_guard lock(summary_write_observer.mutex); summary_write_observer.armed = false; }
    require(observed && waiting && accepted, "bulk summary scratch I/O waited for the target manifest gate");
    engine.finish(); require(engine.release_completed(), "gated bulk summary retained scratch files");
#endif
}
void packing_transition(fs::path const& root, Members input, ExternalConfig ext,
                        std::vector<uint8_t> const& reference)
{
    auto cfg = config(); using Phase = ExternalEngine::MemoryPhase;
    auto const stage = ExternalEngine::phase_memory_bytes(18, cfg, ext, Phase::staging, uint16_t(input.size()));
    auto const pack = ExternalEngine::phase_memory_bytes(18, cfg, ext, Phase::packing);
    auto const publication = ExternalEngine::phase_memory_bytes(18, cfg, ext, Phase::publication);
    require(stage == ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(input.size())) && stage > pack && pack > publication &&
        pack == ExternalEngine::phase_memory_bytes(18, cfg, ext, Phase::packing, UINT16_MAX),
        "external phase reservations lost summary release or retained incoming batch storage");
    cfg.memory_limit_bytes = stage; ext.target_gate = std::make_shared<std::recursive_mutex>();
    auto const path = root / "packing-transition";
    {
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.memory_phase() == Phase::staging && engine.required_memory(uint16_t(input.size())) == stage && engine.append(input).accepted,
            "external staging phase lost its original admission");
        // The owner can prepare on the existing kind of queued writer, and
        // release its old grant only after joining that operation.
        auto preparation = std::async(std::launch::async, [&] { engine.start_packing(); }); preparation.get();
        require(!engine.can_append() && engine.memory_phase() == Phase::packing && engine.required_memory() == pack &&
            ExternalEngine::inspect(path.string()).finishing, "external phase preparation did not durably release the summary");
        engine.start_packing();
    }
    cfg.memory_limit_bytes = pack;
    {
        ExternalEngine engine(path.string(), cfg, ext, true); engine.confirm_identity();
        require(engine.memory_phase() == Phase::packing && engine.required_memory() == pack,
            "external packing recovery retained the staging reservation");
        engine.finish();
        require(engine.memory_phase() == Phase::publication && engine.required_memory() == publication && contents(path) == reference,
            "external lower-budget packing changed ordinary output");
    }
}
void summary_planning(fs::path const& root, Members input, Members compact, ExternalConfig const& ext,
                      std::vector<uint8_t> const& reference)
{
    auto const directory = root / "summary-planning"; fs::create_directory(directory);
    std::optional<std::pair<uint64_t, uint64_t>> admission;
    for (unsigned workers : {1u, 3u})
    {
        auto cfg = config(); cfg.workers = workers; cfg.window_bytes = 4096;
        cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(compact.size()));
        auto const path = directory / std::to_string(workers);
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto const appended = engine.append(compact);
        require(appended.accepted && !engine.stats().sized_chunks, "immutable external planning lost conservative admission");
        auto const bounds = std::pair(appended.attempted_final_bytes, appended.attempted_peak_bytes);
        if (!admission) admission = bounds;
        require(bounds == *admission, "parallel external summary reduction changed admission");
        engine.finish(); require(contents(path) == reference, "parallel external summary planning changed ordinary bytes");
    }
    // The bounded parallel summary path must be exercised independently of
    // contribution encoding, and leave the acknowledged prefix recoverable.
    auto cfg = config(); cfg.window_bytes = 4096;
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(compact.size()));
    auto const interrupted = directory / "interrupted";
    auto child = ::fork(); require(child >= 0, "fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "external-summary-window", 1);
        ExternalEngine engine(interrupted.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        (void)engine.append(compact); std::_Exit(0);
    }
    int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "immutable external planning did not use a parallel summary window");
    require(!ExternalEngine::inspect(interrupted.string()).members, "parallel summary interruption acknowledged members");
    {
        ExternalEngine engine(interrupted.string(), cfg, ext, true); engine.confirm_identity();
        require(engine.append(compact).accepted, "interrupted summary planning could not retry"); engine.finish();
    }
    require(contents(interrupted) == reference, "interrupted summary planning changed ordinary bytes");

    // Raw and borrowed inputs retain the original serial scan even when the
    // codec pool can encode their contributions with multiple workers.
    std::vector<Member> borrowed;
    for (size_t member = 0; member < input.size(); ++member) borrowed.push_back(Member::borrowed(input[member].raw()));
    for (bool view : {false, true})
    {
        auto const path = directory / (view ? "borrowed" : "raw");
        child = ::fork(); require(child >= 0, "fork failed");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", "external-summary-window", 1);
            ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
            require(engine.append(view ? Members(borrowed) : input).accepted, "serial external summary input rejected");
            engine.finish(); std::_Exit(0);
        }
        status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
            "raw or borrowed external input used parallel summary planning");
        require(contents(path) == reference, "serial external summary input changed ordinary bytes");
    }

    // A failure in a later window must drain every queued reader before the
    // window descriptors die, then permit a valid retry on the same engine.
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18), range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<Member> invalid;
    for (size_t member = 0; member < input.size(); ++member)
    {
        auto const raw = input[member].raw(); std::vector<uint64_t> damaged(raw.begin(), raw.end());
        if (!member) damaged.erase(std::lower_bound(damaged.begin(), damaged.end(), (chunks - 1) * range), damaged.end());
        invalid.emplace_back(std::move(damaged)); invalid.back().compact(18);
    }
    auto const failed = directory / "failed";
    {
        ExternalEngine engine(failed.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto const before = contents(failed), checkpoint = contents(fs::path(failed.string() + ".external") / "checkpoint");
        bool rejected = false;
        try { (void)engine.append(invalid); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected && !engine.committed_member_count() && contents(failed) == before &&
            contents(fs::path(failed.string() + ".external") / "checkpoint") == checkpoint,
            "failed parallel external summary changed acknowledged data");
        require(engine.append(compact).accepted, "failed parallel external summary prevented a valid retry"); engine.finish();
    }
    require(contents(failed) == reference, "failed parallel external summary retry changed ordinary bytes");

    // Compact planning keeps the exact canonical-index fallback near final
    // capacity and observes the same workspace boundary as other inputs.
    for (unsigned workers : {1u, 3u})
    {
        auto exact = cfg; exact.workers = workers;
        exact.capacity_bytes = reference.size() + ExternalEngine::metadata_bytes(18, ext);
        auto const path = directory / ("exact-" + std::to_string(workers));
        ExternalEngine engine(path.string(), exact, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(compact).accepted && engine.stats().sized_chunks, "compact external exact capacity fallback rejected fitting bytes");
        engine.finish(); require(contents(path) == reference, "compact external exact capacity fallback changed ordinary bytes");
    }
    auto denied = cfg; --denied.memory_limit_bytes; bool rejected = false;
    try
    {
        ExternalEngine engine((directory / "memory-denied").string(), denied, ext, false, initial_identity());
        engine.confirm_identity(); (void)engine.append(compact);
    }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "parallel external planning bypassed workspace admission");

    // Bound one chunk's descriptors even when they exceed the I/O window and
    // its sparse data cannot be represented by the canonical size index.
    auto sparse = members(size_t(cfg.window_bytes / sizeof(FragmentRegion) + 1), 3);
    std::vector<Member> many;
    for (auto& member : sparse) { many.emplace_back(std::move(member)); many.back().compact(18); }
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(many.size()));
    auto const oversized = directory / "oversized";
    {
        ExternalEngine engine(oversized.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto const before = contents(oversized); rejected = false;
        try { rejected = !engine.append(many).accepted; } catch (UnencodableChunkSize const&) { rejected = true; }
        require(rejected && !engine.committed_member_count() && contents(oversized) == before,
            "oversized external summary descriptors bypassed bounded admission");
        require(engine.append(compact).accepted, "oversized external summary prevented a valid retry"); engine.finish();
    }
    require(contents(oversized) == reference, "oversized external summary retry changed ordinary bytes");
}
void packing_index(fs::path const& root, ExternalConfig const& ext)
{
    // This index exceeds the old flush threshold. Keep a long fixed interval
    // so completed chunks, rather than frequent checkpoints, fill the spool.
    auto const directory = root / "packing-index"; fs::create_directory(directory);
    auto cfg = config(); cfg.window_bytes = 1024 * 1024; cfg.checkpoint_bytes = cfg.capacity_bytes;
    auto const identity = initial_identity(22); auto raw = members(2, 8, 22);
    std::vector<Member> compact;
    for (auto& member : raw) { compact.emplace_back(std::move(member)); compact.back().compact(22); }
    Members const input(compact);
    auto const expected = directory / "reference";
    { Engine engine(expected.string(), cfg, false, identity); engine.confirm_identity();
      require(engine.append(input).accepted, "large index reference rejected"); }
    auto const reference = contents(expected);
    // The unaligned smaller threshold crosses words and keeps a partial byte
    // between bulk flushes; the larger buffer keeps the index to a checkpoint.
    for (uint64_t const window : {uint64_t(65537), uint64_t(1024 * 1024)})
    for (int const error : {0, EINTR, EIO})
    {
        cfg.window_bytes = window;
        cfg.memory_limit_bytes = ExternalEngine::memory_bytes(22, cfg, ext, uint16_t(input.size()));
        auto const path = directory / ("index-" + std::to_string(window) + "-" + std::to_string(error)); bool failed = false;
        {
            ExternalEngine engine(path.string(), cfg, ext, false, identity); engine.confirm_identity();
            require(engine.append(input).accepted, "large external index input rejected");
#ifdef GROW_TEST_WRAP_PWRITE
            struct Restore { ~Restore() { observe_index_writes = false; index_write_error = 0; } } restore;
            largest_index_write = 0; index_write_error = error; observe_index_writes = true;
#endif
            try { engine.finish(); } catch (std::runtime_error const&) { failed = true; }
#ifdef GROW_TEST_WRAP_PWRITE
            require(largest_index_write > 65536 && !index_write_error,
                "external packing did not coalesce its index or reach the injected write error");
            require(failed == (error == EIO), "external buffered index changed write retry or failure handling");
#else
            require(!failed, "external buffered index failed");
#endif
            require(engine.packing_checkpoint_bytes() == cfg.checkpoint_bytes && !engine.checkpoint_tuning_changes(),
                "index buffering changed the explicit checkpoint interval");
        }
        if (failed)
        {
            auto const saved = ExternalEngine::inspect(path.string());
            require(saved.finishing && saved.members == input.size(), "failed buffered index lost the durable contribution prefix");
            // The index buffer is runtime workspace. A smaller admitted window
            // on resume must reconstruct the same stream and carry bits.
            auto resumed = cfg; resumed.window_bytes = 32768;
            ExternalEngine engine(path.string(), resumed, ext, true); engine.confirm_identity(); engine.finish();
        }
        require(contents(path) == reference, "buffered external index or its recovery changed ordinary bytes");
    }
    auto denied = cfg; denied.memory_limit_bytes = ExternalEngine::memory_bytes(22, cfg, ext) - 1; bool rejected = false;
    try { ExternalEngine engine((directory / "memory-denied").string(), denied, ext, false, identity); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "external index buffer bypassed workspace admission");
}
void packing_checkpoints(fs::path const& root, Members input, ExternalConfig const& ext,
                         std::vector<uint8_t> const& reference)
{
    uint64_t const mib = 1024 * 1024, current = 4 * mib, ceiling = PackingCheckpointHistory::maximum_bytes;
    PackingCheckpointHistory slow, fast, noisy, overlapped;
    require(slow.select(current, ceiling, 64 * ceiling) == current, "checkpoint policy changed without completed observations");
    slow.add({current, 1000, 800}); fast.add({current, 1000, 1});
    require(slow.select(current, UINT64_MAX, 64 * ceiling) == PackingCheckpointHistory::forecast_bytes &&
        slow.select(current, 16 * mib, 64 * ceiling) == 16 * mib,
        "checkpoint policy ignored measured overhead or its automatic ceiling");
    require(fast.select(current, ceiling, 64 * ceiling) == current && slow.select(current, ceiling, current) == current,
        "checkpoint policy changed without a material remaining saving");
    overlapped.add({current, 1000, 800, 100000});
    require(overlapped.select(current, ceiling, 64 * ceiling) == current,
        "checkpoint policy hid completed codec work behind durability waits");
    for (size_t i = 0; i < noisy.samples.size(); ++i) noisy.add({current, 1000, 1});
    noisy.add({current, 100000, 99000});
    require(noisy.select(current, ceiling, 64 * ceiling) == current, "isolated checkpoint latency replaced stable timing evidence");

    uint64_t const established = PackingCheckpointHistory::forecast_bytes, trial = 2 * established;
    PackingCheckpointHistory growing, regressed, unproven;
    for (auto history : {&growing, &regressed, &unproven})
    {
        // Small-interval observations cannot extrapolate beyond the existing
        // envelope; first complete a real interval at that frontier.
        history->add({current, 1000, 800});
        require(history->select(established, ceiling, 64 * ceiling) == established,
            "large checkpoint trial reused a small-interval latency forecast");
        for (unsigned i = 0; i < 2; ++i)
        {
            history->add({established, 1000, 800});
            require(history->select(established, ceiling, 64 * ceiling) == established,
                "large checkpoint trial accepted an isolated baseline observation");
        }
        history->add({established, 1000, 800});
        require(history->select(established, ceiling, 64 * ceiling) == trial,
            "profitable large checkpoint trial was not bounded to its next step");
    }
    for (size_t i = 0; i < growing.trial_rates.size(); ++i)
    {
        growing.add({trial, i == 1 ? 120000u : 1200u, 800});
        regressed.add({trial, 3000, 2600}); unproven.add({trial, 2000, 1600});
        bool const complete = i + 1 == growing.trial_rates.size();
        require(growing.select(trial, ceiling, 64 * ceiling) == (complete ? ceiling : trial),
            "large checkpoint growth skipped completed trial evidence");
        require(regressed.select(trial, ceiling, 64 * ceiling) == (complete ? established : trial) &&
            unproven.select(trial, ceiling, 64 * ceiling) == (complete ? established : trial),
            "large checkpoint trial retained a regressed or unproven interval");
    }
    regressed.add({established, 1000, 800});
    require(regressed.select(established, ceiling, 64 * ceiling) == established,
        "failed checkpoint trial repeatedly increased replay and delayed reclamation");
    PackingCheckpointHistory short_tail, limited;
    for (unsigned i = 0; i < 3; ++i)
    { short_tail.add({established, 1000, 800}); limited.add({established, 1000, 800}); }
    require(short_tail.select(established, ceiling, trial) == established &&
        limited.select(established, established, 64 * ceiling) == established,
        "large checkpoint trial ignored its remaining evidence or admitted ceiling");
    PackingCheckpointHistory confidence;
    for (size_t i = 0; i < confidence.samples.size(); ++i) confidence.add({current, 1000, 800});
    for (unsigned i = 0; i < 3; ++i) confidence.add({established, 1000, 240});
    require(confidence.select(established, ceiling, 64 * ceiling) == established,
        "small-interval history reduced the large checkpoint confidence margin");
    confidence.add({established, 1000, 240});
    require(confidence.select(established, ceiling, 64 * ceiling) == trial,
        "sustained current-size evidence did not qualify a material checkpoint trial");

    auto const directory = root / "packing-checkpoints"; fs::create_directory(directory);
    auto cfg = config(); cfg.checkpoint_bytes = 4096;
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(input.size()));
    uint64_t fixed_checkpoints = 0;
    for (bool automatic : {false, true})
    {
        auto const path = directory / (automatic ? "automatic" : "fixed");
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted, "checkpoint policy input rejected");
        if (automatic) engine.tune_packing_checkpoints(UINT64_MAX);
        auto const before = engine.stats().checkpoints;
#ifdef GROW_TEST_WRAP_PWRITE
        struct Restore { ~Restore() { slow_checkpoint_writes = false; } } restore;
        slow_checkpoint_writes = true;
#endif
        engine.finish();
        require(contents(path) == reference && engine.packing_checkpoint_bytes() <= ceiling,
            "checkpoint coalescing changed ordinary bytes or exceeded its redo ceiling");
        if (!automatic)
        {
            fixed_checkpoints = engine.stats().checkpoints - before;
            require(engine.packing_checkpoint_bytes() == cfg.checkpoint_bytes && !engine.checkpoint_tuning_changes(),
                "fixed external checkpoint interval was tuned");
        }
#ifdef GROW_TEST_WRAP_PWRITE
        else require(engine.stats().checkpoints - before < fixed_checkpoints && engine.checkpoint_tuning_changes(),
            "completed destination checkpoint costs did not coalesce durability work");
#endif
    }
    // Interrupt after an automatic checkpoint observation. Its interval and
    // timing evidence are process-local; a resumed explicit interval wins.
    auto const interrupted = directory / "interrupted"; auto const child = ::fork(); require(child >= 0, "fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "external-pack-window-checkpoint", 1);
        auto frequent = cfg; frequent.checkpoint_bytes = 1;
        ExternalEngine engine(interrupted.string(), frequent, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted, "adaptive checkpoint recovery input rejected");
        engine.tune_packing_checkpoints(UINT64_MAX);
#ifdef GROW_TEST_WRAP_PWRITE
        slow_checkpoint_writes = true;
#endif
        engine.finish(); std::_Exit(0);
    }
    int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "adaptive checkpoint recovery did not interrupt a packing window");
    auto const saved = ExternalEngine::inspect(interrupted.string());
    require(saved.finishing && saved.members == input.size(), "adaptive checkpoint changed acknowledged membership");
    auto explicit_config = cfg; explicit_config.checkpoint_bytes = 16384;
    {
        ExternalEngine engine(interrupted.string(), explicit_config, ext, true); engine.confirm_identity();
        require(engine.packing_checkpoint_bytes() == explicit_config.checkpoint_bytes && !engine.checkpoint_tuning_changes(),
            "resumed explicit checkpoint interval reused automatic timing evidence");
        engine.finish();
        require(engine.packing_checkpoint_bytes() == explicit_config.checkpoint_bytes && !engine.checkpoint_tuning_changes(),
            "resumed explicit checkpoint interval changed during packing");
    }
    require(contents(interrupted) == reference, "adaptive checkpoint recovery changed ordinary bytes");
}
void codec_windows(fs::path const& root, Members input, ExternalConfig const& ext,
                   std::vector<uint8_t> const& reference, std::string const& name, bool overlap)
{
    auto cfg = config(); --cfg.window_bytes; cfg.io_overlap = overlap;
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(input.size()));
    auto const normal = root / (name + "-windows");
    {
        ExternalEngine engine(normal.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted, "windowed batch append rejected"); engine.finish();
        require(engine.stats().contribution_workers > 1 && engine.stats().packing_workers > 1,
            "windowed fixture did not admit parallel codecs");
        require(bool(engine.stats().overlap_windows) == overlap && bool(engine.stats().packing_overlap_windows) == overlap,
            "external codec overlap did not follow the requested policy");
    }
    require(contents(normal) == reference, "windowed batch changed ordinary bytes");
    // The failpoint is conditional on a window containing more chunks than
    // workers. Replay the interrupted immutable batch after a durable prefix.
    auto const append = root / (name + "-append-window");
    {
        ExternalEngine engine(append.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input.subspan(0, 1)).accepted, "window prefix append rejected");
    }
    auto child = ::fork(); require(child >= 0, "fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "external-contribution-window", 1);
        ExternalEngine engine(append.string(), cfg, ext, true); engine.confirm_identity();
        (void)engine.append(input.subspan(1)); std::_Exit(0);
    }
    int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "contribution did not gather more chunks than workers");
    require(ExternalEngine::inspect(append.string()).members == 1, "unacknowledged window changed the durable member prefix");
    complete(append, input, ext, true, cfg);
    require(contents(append) == reference, "windowed contribution replay changed ordinary bytes");
    auto const pack = root / (name + "-pack-window");
    {
        ExternalEngine engine(pack.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        for (size_t i = 0; i < input.size(); ++i)
            require(engine.append(input.subspan(i, 1)).accepted, "window packing input rejected");
    }
    child = ::fork(); require(child >= 0, "fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "external-pack-window-checkpoint", 1);
        auto frequent = cfg; frequent.checkpoint_bytes = 1;
        ExternalEngine engine(pack.string(), frequent, ext, true); engine.confirm_identity(); engine.finish(); std::_Exit(0);
    }
    status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "packing did not checkpoint inside a window wider than its workers");
    auto const progress = ExternalEngine::inspect(pack.string());
    require(progress.finishing && progress.members == input.size(), "mid-window checkpoint changed acknowledged membership");
    complete(pack, {}, ext, true, cfg);
    require(contents(pack) == reference, "mid-window packing recovery changed ordinary bytes");
}
void pipeline_admission(fs::path const& root, Members input, ExternalConfig const& ext,
                        std::vector<uint8_t> const& reference)
{
    // A descriptor can be larger than its I/O window. Keep one admitted task
    // per source lane and exercise a single codec worker with overlap on/off.
    auto sparse_raw = members(24, 3); Members sparse(sparse_raw);
    { Engine engine((root / "sparse-reference").string(), config(), false, initial_identity()); engine.confirm_identity();
      require(engine.append(sparse).accepted, "sparse ordinary reference did not fit"); }
    auto const sparse_reference = contents(root / "sparse-reference");
    for (bool const overlap : {false, true})
    {
        auto cfg = config(); cfg.window_bytes = 511; cfg.workers = 1; cfg.io_overlap = overlap;
        cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(sparse.size()));
        auto const path = root / (overlap ? "small-overlap" : "small-serial");
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(sparse).accepted, "small-window contribution did not fit its admitted descriptors"); engine.finish();
        require(bool(engine.stats().overlap_windows) == overlap && bool(engine.stats().packing_overlap_windows) == overlap &&
            engine.stats().packing_workers == 1 && contents(path) == sparse_reference,
            "small-window pipeline changed bytes or its requested overlap policy");
    }
    // The aggregate codec allowance may fit a chunk without fitting its
    // retained predecessor. That case must preserve the serial fallback.
    uint64_t required = ChunkCodec::estimate_base();
    std::vector<FragmentRegion> regions(input.size()); std::vector<size_t> cursors(input.size(), 0);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18), range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        uint64_t symbols = 0;
        for (size_t member = 0; member < input.size(); ++member)
        {
            auto const view = input[member];
            if (view.compact()) regions[member] = view.region(chunk);
            else
            {
                auto const raw = view.raw(); size_t const first = cursors[member];
                while (cursors[member] < raw.size() && raw[cursors[member]] < (chunk + 1) * range) ++cursors[member];
                regions[member] = view.region(chunk, first, cursors[member] - first);
            }
            symbols += regions[member].size();
        }
        auto const detail = ChunkCodec::fresh_summary(18, chunk, regions);
        required = std::max({required, ChunkCodec::estimate_work(18, 0, symbols),
            ChunkCodec::estimate_work(18, ChunkCodec::size_bounds(detail).second, 0)});
    }
    auto cfg = config(); cfg.codec_bytes = required; cfg.io_overlap = true;
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(input.size()));
    auto const fallback = root / "codec-fallback";
    {
        ExternalEngine engine(fallback.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted, "tight aggregate codec rejected a fitting contribution"); engine.finish();
        require(!engine.stats().overlap_windows && !engine.stats().packing_overlap_windows &&
            engine.stats().contribution_workers == 1 && engine.stats().packing_workers == 1 && contents(fallback) == reference,
            "tight codec allowance did not preserve serial fallback");
    }
    auto denied = config(); denied.io_overlap = true;
    denied.memory_limit_bytes = ExternalEngine::memory_bytes(18, denied, ext, uint16_t(input.size())) - 1;
    bool rejected = false;
    try
    {
        ExternalEngine engine((root / "pipeline-memory-denied").string(), denied, ext, false, initial_identity()); engine.confirm_identity();
        (void)engine.append(input);
    }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "external pipeline descriptors bypassed memory admission");
}
void worker_retuning(fs::path const& root, Members input, ExternalConfig const& ext,
                     std::vector<uint8_t> const& reference)
{
    auto cfg = config(); cfg.workers = 1;
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext);
    {
        ExternalEngine bounded((root / "worker-retuning-denied").string(), cfg, ext, false, initial_identity()); bounded.confirm_identity();
        bool rejected = false;
        try { bounded.set_workers(3); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected && bounded.append(input.subspan(0, 1)).accepted && bounded.stats().contribution_workers == 1,
            "external task metadata bypassed its original worker admission");
    }
    auto envelope = cfg; envelope.workers = 3;
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, envelope, ext, uint16_t(input.size()));
    ExternalEngine engine((root / "worker-retuning").string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
    require(engine.append(input.subspan(0, 1)).accepted, "retuned external first contribution rejected");
    bool rejected = false;
    try { engine.set_workers(0); } catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "external retune accepted no workers");
    engine.set_workers(3);
    require(engine.append(input.subspan(1)).accepted && engine.stats().contribution_workers > 1,
        "external contribution retune did not expand beyond its initial share");
    engine.set_workers(1); engine.finish();
    require(engine.stats().packing_workers == 1 && contents(root / "worker-retuning") == reference,
        "external packing retune changed bytes or retained the previous worker share");
}
#ifdef GROW_TEST_WRAP_PWRITE
void independent_io_barriers(fs::path const& root, ExternalConfig const& ext, Members input,
                             Config cfg, std::vector<uint8_t> const& expected)
{
    auto const target = storage_device(root);
    for (auto const& grant : ext.scratch)
    {
        auto const source = storage_device(grant.directory);
        if (!external_detail::independent_packing_io(target, {&source, 1}))
        { std::cout << "independent packing I/O barriers require admitted storage topology\n"; return; }
    }
    for (unsigned failure : {0u, 1u, 2u})
    {
        auto const path = root / ("independent-io-" + std::to_string(failure));
        auto shared = ext; shared.target_gate = std::make_shared<std::recursive_mutex>();
        uint64_t contribution_bytes;
        {
            ExternalEngine engine(path.string(), cfg, shared, false, initial_identity()); engine.confirm_identity();
            require(engine.append(input).accepted, "independent packing source did not fit");
            contribution_bytes = engine.contribution_bytes();
        }
        auto const stage_path = fs::path(path.string() + "-next");
        auto stage = std::make_unique<ExternalEngine>(stage_path.string(), cfg, shared, false, initial_identity()); stage->confirm_identity();
        struct stat file{}; require(!::stat(path.c_str(), &file), "cannot inspect independent target fixture");
        {
            std::lock_guard lock(target_pipeline_barrier.mutex); auto& state = target_pipeline_barrier;
            state.armed = true; state.enabled = state.released = state.codec_blocked = state.write_blocked = false;
            state.read_observed = state.timed_out = false; state.calls = 0;
            state.observe_stage = state.stage_checkpoint = false; state.stage_bytes = 0; state.stage_expected_bytes = contribution_bytes + PlotGroupFile::getChunkCountForK(18) * (sizeof(ChunkSummary) + sizeof(uint64_t));
            state.stage_checkpoint_prefix = stage_path.string() + ".external/checkpoint.partial.";
            state.fail_read = failure == 1; state.fail_write = failure == 2;
            state.target_device = file.st_dev; state.target_inode = file.st_ino;
            // A source packet consumes its whole reader window plus a byte.
            // This read belongs to the source after the blocked codec window.
            state.read_offset = 64 + cfg.window_bytes * 4;
        }
        auto operation = std::async(std::launch::async, [&]
        {
            try { ExternalEngine engine(path.string(), cfg, shared, true); engine.confirm_identity(); engine.finish(); return false; }
            catch (std::runtime_error const&) { return true; }
        });
        bool observed, outstanding, staged, checkpoint_blocked, checkpoint_emitted, timed_out;
        {
            std::unique_lock lock(target_pipeline_barrier.mutex); auto& state = target_pipeline_barrier;
            observed = state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.read_observed; });
            outstanding = operation.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
            state.observe_stage = true;
        }
        auto contribution = std::async(std::launch::async, [&] { return stage->append(input).accepted; });
        {
            std::unique_lock lock(target_pipeline_barrier.mutex); auto& state = target_pipeline_barrier;
            staged = state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.stage_bytes == state.stage_expected_bytes; });
            checkpoint_blocked = !state.ready.wait_for(lock, std::chrono::seconds(1), [&] { return state.stage_checkpoint; }) &&
                contribution.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
            state.released = true; state.ready.notify_all();
        }
        bool const rejected = operation.get();
        bool const accepted = contribution.get();
        {
            std::lock_guard lock(target_pipeline_barrier.mutex); auto& state = target_pipeline_barrier;
            timed_out = state.timed_out; checkpoint_emitted = state.stage_checkpoint; state.armed = false;
        }
        require(observed && outstanding && staged && checkpoint_blocked && checkpoint_emitted && accepted && !timed_out && rejected == bool(failure),
            "independent read/encode/write pipeline did not retain its windows and serialize target metadata");
        complete(path, input, shared, true, cfg);
        require(contents(path) == expected, "independent I/O recovery changed canonical bytes");
        stage->finish(); stage.reset();
        require(contents(stage_path) == expected, "concurrent external contribution changed canonical bytes");
    }
}
void pipeline_barriers(fs::path const& root, ExternalConfig const& ext)
{
    auto raw = members(2, 16); std::vector<Member> storage;
    for (auto const& value : raw) { storage.emplace_back(std::vector<uint64_t>(value)); storage.back().compact(18); }
    Members input(storage);
    { Engine engine((root / "barrier-reference").string(), config(), false, initial_identity()); engine.confirm_identity();
      require(engine.append(input).accepted, "barrier reference did not fit"); }
    auto const expected = contents(root / "barrier-reference");
    auto cfg = config(); cfg.workers = 1; cfg.io_overlap = true;
    std::vector<FragmentRegion> regions;
    for (size_t member = 0; member < input.size(); ++member) regions.push_back(input[member].region(0));
    ChunkCodec codec(18, cfg.codec_bytes);
    uint64_t const encoded = codec.encode(0, {}, 0, regions).size();
    // The first contribution packet fills its writer, while a single source
    // chunk fits. Descriptor/source admission makes these one-chunk windows.
    cfg.window_bytes = encoded + sizeof(uint64_t) - 1;
    require(cfg.window_bytes >= sizeof(Header) &&
        cfg.window_bytes < sizeof(ChunkInput) + sizeof(uint64_t) + sizeof(EncodedBlock) + regions.size() * sizeof(FragmentRegion) &&
        cfg.window_bytes < encoded * 2, "barrier fixture did not isolate a single source task");
    cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, uint16_t(input.size()));
    independent_io_barriers(root, ext, input, cfg, expected);
    for (bool const packing : {false, true})
        for (bool const failure : {false, true})
        {
            auto const name = std::string(packing ? "barrier-pack" : "barrier-append") + (failure ? "-failed" : "-success");
            auto const path = root / name;
            if (packing)
            {
                ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
                require(engine.append(input).accepted, "barrier packing source did not fit");
            }
            auto engine = std::make_unique<ExternalEngine>(path.string(), cfg, ext, packing, initial_identity()); engine->confirm_identity();
            {
                std::lock_guard lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
                state.armed = true; state.enabled = state.released = state.blocked = state.observed = state.timed_out = false;
                state.read = false; state.fail = failure; state.calls = 0; state.hold = packing ? 2 : 1;
                state.write_offset = packing ? engine->identity().header.chunk_index_offset : 64;
            }
            auto operation = std::async(std::launch::async, [&]
            {
                bool rejected = false;
                try
                {
                    if (packing) engine->finish(); else require(engine->append(input).accepted, "barrier contribution did not fit");
                }
                catch (std::runtime_error const&) { rejected = true; }
                return rejected;
            });
            bool observed, blocked, outstanding, timed_out;
            {
                std::unique_lock lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
                observed = state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.observed; });
                blocked = state.blocked; outstanding = operation.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
                state.released = true; state.ready.notify_all();
            }
            bool const rejected = operation.get();
            {
                std::lock_guard lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
                timed_out = state.timed_out; state.armed = false;
            }
            engine.reset();
            require(observed && blocked && outstanding && !timed_out && rejected == failure,
                "external encode/write barrier did not retain a live producer: " + name +
                " observed=" + std::to_string(observed) + " blocked=" + std::to_string(blocked) +
                " outstanding=" + std::to_string(outstanding) + " timeout=" + std::to_string(timed_out) +
                " rejected=" + std::to_string(rejected));
            auto const saved = ExternalEngine::inspect(path.string());
            require(saved.members == (packing || !failure ? input.size() : 0), "barrier write changed acknowledged membership");
            complete(path, input, ext, true, cfg);
            require(contents(path) == expected, "barrier writer recovery changed canonical bytes");
        }
    // Hold the first packing encode while the foreground reads its following
    // source. The reader and checkpoint state remain on that foreground.
    for (bool const failure : {false, true})
    {
        auto const path = root / (failure ? "barrier-read-failed" : "barrier-read-success");
        {
            ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
            require(engine.append(input).accepted, "read barrier packing source did not fit");
        }
        {
            std::lock_guard lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
            state.armed = true; state.enabled = state.released = state.blocked = state.observed = state.timed_out = false;
            state.read = true; state.fail = failure; state.calls = state.hold = 0;
            // One source packet occupies a full reader buffer and one byte
            // of its next buffer, so this read belongs to the following task.
            state.read_offset = 64 + cfg.window_bytes * 2;
        }
        auto operation = std::async(std::launch::async, [&]
        {
            try { ExternalEngine engine(path.string(), cfg, ext, true); engine.confirm_identity(); engine.finish(); return false; }
            catch (std::runtime_error const&) { return true; }
        });
        bool observed, blocked, outstanding, timed_out;
        {
            std::unique_lock lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
            observed = state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.observed; });
            blocked = state.blocked; outstanding = operation.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
            state.released = true; state.ready.notify_all();
        }
        bool const rejected = operation.get();
        {
            std::lock_guard lock(pipeline_barrier.mutex); auto& state = pipeline_barrier;
            timed_out = state.timed_out; state.armed = false;
        }
        require(observed && blocked && outstanding && !timed_out && rejected == failure,
            "external read-ahead did not retain a live encode through success or failure");
        complete(path, input, ext, true, cfg);
        require(contents(path) == expected, "barrier reader recovery changed canonical bytes");
    }
}
#endif
void retirement(fs::path const& root, Members input, ExternalConfig const& ext, std::vector<uint8_t> const& reference)
{
    // Before the single rename, the complete canonical checkpoint remains
    // resumable and no retired directory has been introduced.
    {
        auto const path = root / "before-retirement"; complete(path, input, ext, false);
        auto const holder = retired_path(path); auto const checkpoint = fs::path(path.string() + ".external") / "checkpoint";
        auto const saved = contents(checkpoint); auto const child = ::fork(); require(child >= 0, "fork failed");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", "external-handoff-before-retirement", 1);
            ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity(); engine.release_completed(); std::_Exit(0);
        }
        int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86, "pre-retirement boundary was not reached");
        require(ExternalEngine::pending(path.string()) && !fs::exists(holder) && contents(checkpoint) == saved && contents(path) == reference,
            "pre-retirement interruption changed ownership or output");
        require(ExternalEngine::inspect(path.string()).complete, "pre-retirement checkpoint could not be inspected");
        { ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity(); engine.finish(); require(engine.release_completed(), "pre-retirement recovery did not hand off"); }
        require(!ExternalEngine::pending(path.string()) && contents(path) == reference, "pre-retirement recovery changed ordinary bytes");
    }
    // Every occupied destination is preserved, including an empty directory
    // that a replacing rename would otherwise silently discard.
    {
        auto const path = root / "occupied-retirement"; complete(path, input, ext, false);
        auto const holder = retired_path(path); auto const checkpoint = fs::path(path.string() + ".external") / "checkpoint";
        auto const saved = contents(checkpoint); auto const sentinel = root / "retirement-sentinel";
        { std::ofstream out(sentinel, std::ios::binary); out << "user data"; }
        auto const sentinel_bytes = contents(sentinel);
        ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity();
        for (std::string const kind : {"empty-directory", "directory", "file", "symlink"})
        {
            if (kind == "empty-directory" || kind == "directory")
            {
                fs::create_directory(holder);
                if (kind == "directory") fs::copy_file(sentinel, holder / "user");
            }
            else if (kind == "file") fs::copy_file(sentinel, holder);
            else fs::create_symlink(sentinel, holder);
            struct stat before{}, after{}; require(!::lstat(holder.c_str(), &before), "cannot inspect occupied fixture");
            bool rejected = false;
            try { (void)engine.release_completed(); } catch (std::exception const&) { rejected = true; }
            require(rejected && !::lstat(holder.c_str(), &after) && before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
                before.st_mode == after.st_mode && ExternalEngine::pending(path.string()) && contents(checkpoint) == saved &&
                contents(path) == reference && contents(sentinel) == sentinel_bytes,
                "external retirement replaced an occupied destination or damaged its checkpoint");
            if (kind == "directory") require(contents(holder / "user") == sentinel_bytes, "occupied directory contents changed");
            if (kind == "file") require(contents(holder) == sentinel_bytes, "occupied file changed");
            if (kind == "symlink") require(fs::read_symlink(holder) == sentinel, "occupied symlink changed");
            fs::remove_all(holder);
        }
#ifdef GROW_TEST_WRAP_PWRITE
        reject_atomic_retirement = true; bool rejected = false; std::string reason;
        try { (void)engine.release_completed(); } catch (std::exception const& error) { rejected = true; reason = error.what(); }
        require(rejected && reason.find(std::strerror(ENOSYS)) != reason.npos && !reject_atomic_retirement &&
            retirement_flags == RENAME_NOREPLACE && !fs::exists(holder) &&
            ExternalEngine::pending(path.string()) && contents(checkpoint) == saved && contents(path) == reference,
            "unavailable no-replace rename mutated the source or used a replacing fallback");
#endif
        require(engine.release_completed() && !ExternalEngine::pending(path.string()) && contents(path) == reference,
            "retirement could not retry after a safely rejected destination");
    }
    complete(root / "handoff-ordinary", input, ext, false);
    // A completed protocol can hand off without leaving an active sidecar.
    {
        ExternalEngine engine((root / "handoff-ordinary").string(), config(), ext, true); engine.confirm_identity();
        require(engine.release_completed() && !ExternalEngine::pending((root / "handoff-ordinary").string()), "completed external protocol did not hand off");
    }
    { Engine engine((root / "handoff-ordinary").string(), config(), true); engine.confirm_identity();
      require(engine.identity().header.group_size == input.size(), "ordinary handoff lost group members"); }
    std::vector<std::string> handoffs{"external-handoff-retired"};
#ifdef __linux__
    if (!::setxattr(root.c_str(), "user.xchplot2.test", "1", 1, XATTR_CREATE))
    {
        require(!::removexattr(root.c_str(), "user.xchplot2.test"), "cannot remove test attribute");
        handoffs.push_back("external-handoff-checkpoint-removed");
    }
#endif
    for (auto const& boundary : handoffs)
    {
        auto const path = root / boundary; complete(path, input, ext, false);
        auto const child = ::fork(); require(child >= 0, "fork failed");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", boundary.c_str(), 1);
            ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity(); engine.release_completed(); std::_Exit(0);
        }
        int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86, "handoff interruption boundary was not reached");
        require(!ExternalEngine::pending(path.string()), "retired checkpoint blocks ordinary resume");
        fs::path retired;
        for (auto const& entry : fs::directory_iterator(root))
            if (entry.path().filename().string().starts_with(path.filename().string() + ".external.retired-")) retired = entry.path();
        require(!retired.empty() && ExternalEngine::inspect_retired(path.string(), retired.string()).complete,
            "interrupted retired checkpoint could not authenticate its namespace");
        Engine engine(path.string(), config(), true); engine.confirm_identity(); require(contents(path) == reference, "interrupted handoff changed ordinary bytes");
        require(engine.append(input.subspan(0, 1)).accepted && ExternalEngine::inspect_retired(path.string(), retired.string()).complete,
            "later ordinary append invalidated terminal metadata ownership");
        auto const forged = root / (path.filename().string() + ".external.retired-" + std::string(64, 'f'));
        fs::create_directory(forged);
        if (fs::exists(retired / "checkpoint")) fs::copy_file(retired / "checkpoint", forged / "checkpoint");
#ifdef __linux__
        else
        {
            ssize_t const size = ::getxattr(retired.c_str(), "user.xchplot2.external-retired", nullptr, 0);
            require(size > 0, "retired tombstone has no ownership receipt"); std::vector<char> receipt(size_t(size), 0);
            require(::getxattr(retired.c_str(), "user.xchplot2.external-retired", receipt.data(), receipt.size()) == size &&
                !::setxattr(forged.c_str(), "user.xchplot2.external-retired", receipt.data(), receipt.size(), XATTR_CREATE), "cannot copy test receipt");
        }
#endif
        bool rejected = false;
        try { (void)ExternalEngine::inspect_retired(path.string(), forged.string()); } catch (std::exception const&) { rejected = true; }
        require(rejected && fs::exists(forged), "retired inspection trusted a copied prefixed directory");
    }
#ifdef GROW_TEST_WRAP_PWRITE
    {
        auto const path = root / "receipt-unavailable"; complete(path, input, ext, false); uint64_t retained_metadata = 0;
        { ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity(); reject_retired_receipt = true;
          require(engine.release_completed() && !reject_retired_receipt, "unavailable receipt did not preserve checkpoint proof");
          retained_metadata = engine.remaining_metadata_bytes(); }
        fs::path retired; for (auto const& entry : fs::directory_iterator(root))
            if (entry.path().filename().string().starts_with(path.filename().string() + ".external.retired-")) retired = entry.path();
        require(!retired.empty() && fs::exists(retired / "checkpoint") &&
            ExternalEngine::inspect_retired(path.string(), retired.string()).complete, "unavailable xattrs lost terminal ownership proof");
        auto const inspection = ExternalEngine::inspect_retired(path.string(), retired.string());
        require(retained_metadata && retained_metadata == inspection.metadata_occupied_bytes &&
            inspection.metadata_reserved_bytes == inspection.metadata_occupied_bytes &&
            inspection.peak_bytes == inspection.target_bytes + inspection.metadata_occupied_bytes,
            "retired metadata was credited as reclaimed or charged as future construction workspace");
    }
#endif
}
void retained_runs(fs::path const& root, Members input, ExternalConfig ext,
                   std::vector<uint8_t> const& reference)
{
    ext.packing_inputs = 8;
    auto const path = root / "retained-runs";
    {
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        for (size_t i = 0; i < input.size(); ++i)
            require(engine.append(input.subspan(i, 1)).accepted, "retained contribution rejected");
        require(!engine.stats().merge_passes, "fitting final readers were copied before packing");
    }
    auto const saved = ExternalEngine::inspect(path.string());
    require(saved.external.packing_inputs == ext.packing_inputs && saved.external.fan_in == ext.fan_in,
        "external inspection lost the separate reader and compaction settings");
    auto changed = ext; changed.packing_inputs = 4; bool rejected = false;
    try { ExternalEngine engine(path.string(), config(), changed, true); }
    catch (std::exception const&) { rejected = true; }
    require(rejected, "external resume changed the saved reader settings");
    auto const stats = complete(path, input, ext, true);
    require(!stats.merge_passes && stats.merge_inputs == input.size() && contents(path) == reference,
        "retained final readers changed ordinary bytes or forced an unnecessary copy");

    // Use ordinary member density so the large group remains within the
    // upstream chunk-index encoding range as well as the run-table bound.
    auto many_raw = members(67, 64); Members many(many_raw);
    auto const expected = root / "run-limit-reference";
    { Engine engine(expected.string(), config(), false, initial_identity()); engine.confirm_identity();
      require(engine.append(many).accepted, "run-limit reference rejected"); }
    ext.packing_inputs = 64;
    ext.segment_bytes = 1024 * 1024;
    for (auto& grant : ext.scratch) grant.capacity_bytes = 64ull * 1024 * 1024;
    auto const limited = root / "run-limit";
    ExternalEngine engine(limited.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
    require(engine.append(many.subspan(0, 3)).accepted, "run-limit initial contribution rejected");
    uint64_t prefix = engine.contribution_bytes();
    for (size_t i = 3; i < many.size(); ++i)
    {
        require(engine.append(many.subspan(i, 1)).accepted, "run-limit contribution rejected");
        if (i == 3) prefix += engine.contribution_bytes();
    }
    require(engine.stats().merge_passes == 1 &&
        engine.stats().phases[size_t(WorkPhase::merge)].read_bytes < prefix,
        "run-limit compaction copied the growing prefix instead of a smaller adjacent set");
    engine.finish(); require(contents(limited) == contents(expected), "forced retained-run compaction changed ordinary bytes");
}
// A stalled actual write/fsync on one output must return all codec CPU permits
// before another output performs serial planning and parallel encoding.
void shared_storage_cpu(fs::path const& root, Members input)
{
#ifdef GROW_TEST_WRAP_PWRITE
    for (bool sync : {false, true}) for (bool overlap : {false, true})
    {
        auto cfg = config(); cfg.workers = 1; cfg.io_overlap = overlap;
        cfg.cpu_scheduler = std::make_shared<CodecScheduler>(cfg.workers);
        auto const suffix = std::string(sync ? "sync" : "write") + (overlap ? "-overlap" : "-serial");
        auto const blocked_path = root / ("shared-cpu-blocked-" + suffix);
        auto const other_path = root / ("shared-cpu-other-" + suffix);
        Engine first(blocked_path.string(), cfg, false, initial_identity()); first.confirm_identity();
        {
            std::lock_guard lock(shared_storage_barrier.mutex); auto& state = shared_storage_barrier;
            state.path = blocked_path.string(); state.sync = sync;
            state.armed = true; state.blocked = state.released = state.timed_out = false;
        }
        auto blocked = std::async(std::launch::async, [&] { return first.append(input); });
        bool observed;
        {
            std::unique_lock lock(shared_storage_barrier.mutex); auto& state = shared_storage_barrier;
            observed = state.ready.wait_for(lock, std::chrono::seconds(5), [&] { return state.blocked; });
        }
        auto other = std::async(std::launch::async, [&]
        {
            auto grant = external(root);
            ExternalEngine second(other_path.string(), cfg, grant, false, initial_identity()); second.confirm_identity();
            require(second.append(input).accepted, "independent shared CPU contribution rejected");
            return second.finish();
        });
        bool const progressed = other.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        bool const outstanding = blocked.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
        bool timed_out;
        {
            std::lock_guard lock(shared_storage_barrier.mutex); auto& state = shared_storage_barrier;
            timed_out = state.timed_out; state.released = true; state.armed = false;
        }
        shared_storage_barrier.ready.notify_all();
        auto const appended = blocked.get(); auto const finished = other.get();
        require(observed && progressed && outstanding && !timed_out && appended.accepted && finished.members == input.size(),
            "blocked output storage retained shared compute admission");
        require(contents(blocked_path) == contents(other_path), "shared compute changed ordinary/external v2 bytes");
    }
#else
    (void)root; (void)input;
#endif
}

void prepared_inputs(fs::path const& root, std::vector<std::vector<uint64_t>> const& raw,
                     ExternalConfig const& ext, std::vector<uint8_t> const& reference)
{
    auto feed = [&](ExternalEngine::PreparedInput& input, bool packed)
    {
        for (size_t member = 0; member < raw.size(); ++member)
        {
            auto const values = std::span<uint64_t const>(raw[member]); input.begin_member(uint16_t(member), values.size());
            if (packed)
            {
                std::vector<uint32_t> boundaries(size_t(PlotGroupFile::getChunkCountForK(18)) + 1);
                size_t first = 0;
                for (size_t chunk = 0; chunk < boundaries.size(); ++chunk)
                { while (first < values.size() && (values[first] >> 24) < chunk) ++first; boundaries[chunk] = uint32_t(first); }
                input.append_packed({boundaries, {}, {values.size(), 0}});
                BitWriter bits; for (auto value : values) bits.append(value & ((1ull << 24) - 1), 24);
                std::vector<uint64_t> words((values.size() * 24 + 63) / 64);
                std::memcpy(words.data(), bits.asBytes().data(), bits.asBytes().size());
                for (size_t first = 0; first < words.size();)
                {
                    size_t const count = std::min(size_t(17), words.size() - first);
                    input.append_packed({{}, std::span<uint64_t const>(words).subspan(first, count), {}}); first += count;
                }
            }
            else for (size_t first = 0; first < values.size();)
            {
                size_t const count = std::min(size_t(13), values.size() - first);
                input.append_raw(values.subspan(first, count)); first += count;
            }
            input.finish_member();
        }
        input.seal();
    };
    {
        ExternalEngine engine((root / "prepared-batch-overflow").string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        std::vector<Member> oversized; oversized.reserve(size_t(UINT16_MAX) + 2);
        for (size_t member = 0; member < size_t(UINT16_MAX) + 2; ++member) oversized.push_back(Member::borrowed(raw.front()));
        bool rejected = false; try { engine.append(oversized); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected && !engine.committed_member_count(), "ordinary batch membership narrowed before validation");
    }
    for (bool packed : {false, true}) for (bool overlap : {false, true})
    {
        auto cfg = config(); cfg.io_overlap = overlap;
        auto const path = root / (std::string("prepared-") + (packed ? "packed" : "raw") + (overlap ? "-overlap" : "-serial"));
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "prepared input reservation rejected");
        require(input->reserved_bytes() >= input->written_bytes(), "prepared input reservation undercounts payload");
        feed(*input, packed); require(input->sealed(), "prepared input did not seal");
        require(engine.append(*input).accepted && engine.batch_resources().packets == 1,
            "prepared input lost grouped contribution batching");
        require(input->read_bytes() >= input->written_bytes(), "prepared input read accounting omits a source scan");
        input.reset(); // Committed contribution/summary ownership belongs to the engine.
        engine.finish(); require(contents(path) == reference, "prepared input output differs");
    }
    {
        auto const path = root / "prepared-discard";
        auto engine = std::make_unique<ExternalEngine>(path.string(), config(), ext, false, initial_identity()); engine->confirm_identity();
        auto const before = engine->scratch_reserved_bytes();
        auto input = engine->prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "discard reservation rejected");
        input->begin_member(0, raw.front().size()); input->append_raw(std::span(raw.front()).first(13)); input.reset();
        require(engine->scratch_reserved_bytes() == before, "discarded partial input retains scratch");
        input = engine->prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "detached reservation rejected");
        engine.reset(); input.reset(); // Teardown detaches the provisional owner safely.
    }
    {
        ExternalEngine engine((root / "prepared-wide-bound").string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        require(!engine.prepare_input(1, uint64_t(UINT32_MAX) + 1) && engine.scratch_reserved_bytes() == before,
            "unsupported prepared boundary width did not preserve ordinary fallback");
    }
    {
        auto tight = ext; for (auto& grant : tight.scratch) grant.capacity_bytes = 4096;
        ExternalEngine engine((root / "prepared-tight").string(), config(), tight, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        require(!engine.prepare_input(uint16_t(raw.size()), raw.front().size()), "unreserved prepared input admitted");
        require(engine.scratch_reserved_bytes() == before, "failed preflight retains scratch");
    }
    {
        auto const path = root / "prepared-exact"; auto cfg = config();
        cfg.capacity_bytes = reference.size() + ExternalEngine::metadata_bytes(18, ext);
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "exact input reservation rejected");
        feed(*input, false); require(engine.append(*input).accepted, "prepared exact output capacity rejected");
        engine.finish(); require(contents(path) == reference, "prepared exact output differs");
    }
    {
        auto const path = root / "prepared-rejected-prefix"; auto cfg = config();
        auto prefix_path = root / "prepared-prefix-reference";
        { Engine engine(prefix_path.string(), config(), false, initial_identity()); engine.confirm_identity();
          require(engine.append(Members(raw).subspan(0, 1)).accepted, "prepared prefix reference rejected"); }
        auto const prefix = contents(prefix_path);
        cfg.capacity_bytes = prefix.size() + ExternalEngine::metadata_bytes(18, ext);
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "prefix input reservation rejected");
        feed(*input, false); require(!engine.append(*input).accepted, "oversized prepared batch accepted");
        require(engine.append(*input, 1).accepted, "prepared fitting prefix rejected");
        engine.finish(); require(contents(path) == prefix, "prepared prefix output differs");
    }
#ifdef GROW_TEST_WRAP_PWRITE
    {
        auto const path = root / "prepared-write-error";
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "write error input reservation rejected");
        input->begin_member(0, raw.front().size()); writes_before_failure = 0; fail_next_write = true;
        bool rejected = false;
        try { input->append_raw(raw.front()); input->finish_member(); input->seal(); }
        catch (std::runtime_error const&) { rejected = true; }
        fail_next_write = false; require(rejected, "prepared input write error disappeared"); input.reset();
        require(engine.scratch_reserved_bytes() == before, "failed prepared write retains scratch");
    }
#endif
    for (bool packed : {false, true})
    {
        auto const path = root / (packed ? "prepared-bad-packed" : "prepared-bad-raw");
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        auto input = engine.prepare_input(1, raw.front().size()); require(bool(input), "malformed input reservation rejected");
        input->begin_member(0, raw.front().size()); bool rejected = false;
        try
        {
            if (packed)
            {
                std::vector<uint32_t> boundaries(size_t(PlotGroupFile::getChunkCountForK(18)) + 1, 0);
                boundaries.back() = uint32_t(raw.front().size()); boundaries[1] = boundaries.back();
                input->append_packed({boundaries, {}, {raw.front().size(), 0}});
            }
            else
            {
                std::array<uint64_t, 2> unsorted{17, 16}; input->append_raw(unsorted);
            }
        }
        catch (std::runtime_error const&) { rejected = true; }
        require(rejected, "invalid prepared transport was accepted");
        rejected = false; try { input->finish_member(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, "failed prepared transport was reusable");
        input.reset(); require(engine.scratch_reserved_bytes() == before, "invalid input retained scratch");
    }
#ifdef GROW_TEST_WRAP_PWRITE
    {
        auto const path = root / "prepared-append-error";
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "append error reservation rejected");
        feed(*input, false); writes_before_failure = 0; fail_next_write = true;
        OrderedWriter worker([] {}); auto ready = worker.submit([&] { engine.append(*input); });
        bool rejected = false; try { ready.get(); } catch (std::runtime_error const&) { rejected = true; }
        fail_next_write = false; require(rejected, "prepared contribution write failure disappeared");
        input.reset(); require(engine.scratch_reserved_bytes() == before, "failed prepared contribution retained scratch");
        require(engine.append(Members(raw)).accepted, "ordinary append after failed prepared contribution rejected");
        engine.finish(); require(contents(path) == reference, "ordinary recovery from prepared failure differs");
    }
#endif
#ifdef GROW_TEST_WRAP_PWRITE
    {
        auto const path = root / "prepared-read-error";
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "read error reservation rejected");
        feed(*input, true);
        {
            std::lock_guard lock(summary_read_observer.mutex); auto& state = summary_read_observer;
            state.prefix = ".xchplot2-external-"; state.fail = state.armed = true;
        }
        bool rejected = false; try { engine.append(*input); } catch (std::runtime_error const&) { rejected = true; }
        { std::lock_guard lock(summary_read_observer.mutex); summary_read_observer.armed = false; }
        require(rejected && !engine.committed_member_count(), "prepared source read failure lost the durable frontier");
        require(engine.append(*input).accepted, "prepared source retry after read failure rejected");
        input.reset(); engine.finish(); require(contents(path) == reference, "prepared source retry output differs");
    }
#endif
    {
        ExternalEngine engine((root / "prepared-bad-tail").string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes(); auto input = engine.prepare_input(1, 1);
        require(bool(input), "tail input reservation rejected"); input->begin_member(0, 1);
        std::vector<uint32_t> boundaries(size_t(PlotGroupFile::getChunkCountForK(18)) + 1, 0); boundaries.back() = 1;
        input->append_packed({boundaries, {}, {1, 0}}); std::array<uint64_t, 1> tail{1ull << 63};
        input->append_packed({{}, tail, {}}); bool rejected = false;
        try { input->finish_member(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, "nonzero prepared packed tail was accepted"); input.reset();
        require(engine.scratch_reserved_bytes() == before, "invalid prepared tail retained scratch");
    }
#ifdef GROW_TEST_WRAP_PWRITE
    {
        auto const path = root / "prepared-codec-error";
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "codec error reservation rejected");
        feed(*input, false); fail_next_codec = true;
        OrderedWriter worker([] {}); auto ready = worker.submit([&] { engine.append(*input); });
        bool rejected = false; try { ready.get(); } catch (std::runtime_error const&) { rejected = true; }
        fail_next_codec = false; require(rejected, "prepared codec worker exception disappeared");
        input.reset(); require(engine.scratch_reserved_bytes() == before, "failed codec worker retained prepared scratch");
        require(engine.append(Members(raw)).accepted, "ordinary append after failed prepared codec rejected");
        engine.finish(); require(contents(path) == reference, "ordinary recovery from prepared codec failure differs");
    }
#endif
#ifdef GROW_TEST_WRAP_PWRITE
    for (bool packed : {false, true})
    {
        auto cfg = config(); cfg.cpu_scheduler = std::make_shared<CodecScheduler>(1);
        ExternalEngine engine((root / (packed ? "prepared-packed-cpu-gate" : "prepared-raw-cpu-gate")).string(), cfg, ext, false, initial_identity());
        engine.confirm_identity();
        auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "CPU gate input reservation rejected");
        {
            std::lock_guard lock(prepared_write_barrier.mutex); auto& state = prepared_write_barrier;
            state.armed = true; state.blocked = state.released = state.timed_out = false;
        }
        struct Release
        {
            ~Release()
            {
                std::lock_guard lock(prepared_write_barrier.mutex); prepared_write_barrier.released = true;
                prepared_write_barrier.ready.notify_all();
            }
        };
        bool blocked = false, progressed = false;
        {
            auto pending = std::async(std::launch::async, [&] { feed(*input, packed); });
            Release release;
            {
                std::unique_lock lock(prepared_write_barrier.mutex);
                blocked = prepared_write_barrier.ready.wait_for(lock, std::chrono::seconds(5), [&] { return prepared_write_barrier.blocked; });
            }
            auto other = cfg.cpu_scheduler->submit([] {});
            progressed = other.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
            {
                std::lock_guard lock(prepared_write_barrier.mutex); prepared_write_barrier.released = true;
                prepared_write_barrier.ready.notify_all();
            }
            pending.get(); other.get();
        }
        require(blocked && progressed && !prepared_write_barrier.timed_out,
            "blocked prepared storage retained the shared CPU permit");
        require(engine.append(*input).accepted, "CPU gate prepared append rejected");
        input.reset(); engine.finish();
    }
#endif
    for (auto const* point : {"external-input-sealed", "external-contribution-flushed", "external-run-after-checkpoint"})
    {
        auto const path = root / (std::string("prepared-crash-") + point);
        auto const child = ::fork(); require(child >= 0, "cannot fork prepared input crash test");
        if (!child)
        {
            try
            {
                ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
                auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "crash reservation rejected");
                require(!::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1), "cannot set prepared input failpoint");
                feed(*input, false); engine.append(*input); std::_Exit(0);
            }
            catch (...) { std::_Exit(1); }
        }
        int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "prepared input failpoint did not interrupt");
        ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity();
        if (!engine.committed_member_count())
        {
            auto input = engine.prepare_input(uint16_t(raw.size()), raw.front().size()); require(bool(input), "resumed prepared reservation rejected");
            feed(*input, true); require(engine.append(*input).accepted, "resumed prepared append rejected");
        }
        engine.finish(); require(contents(path) == reference, "resumed prepared output differs");
    }
}

void contribution_inputs(fs::path const& root, std::vector<std::vector<uint64_t>> const& raw,
                         ExternalConfig const& ext, std::vector<uint8_t> const& reference)
{
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18), range = 1ull << 24;
    std::vector<std::vector<std::vector<uint8_t>>> encoded(raw.size());
    ChunkCodec codec(18, config().codec_bytes);
    for (size_t member = 0; member < raw.size(); ++member)
    {
        size_t first = 0;
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
        {
            size_t last = first;
            while (last < raw[member].size() && raw[member][last] < (chunk + 1) * range) ++last;
            FragmentRegion const region(std::span(raw[member]).subspan(first, last - first));
            auto bytes = codec.encode(chunk, {}, 0, {&region, 1});
            encoded[member].emplace_back(bytes.begin(), bytes.end()); first = last;
        }
    }
    auto rejects = [](auto&& action)
    {
        bool rejected = false; try { action(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, "invalid contribution input was accepted");
    };
    auto feed_member = [&](ExternalEngine::ContributionInput& input, uint16_t member, size_t offset)
    {
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
            input.append_chunk(member, chunk, encoded[offset + member][size_t(chunk)]);
        input.finish_member(member);
    };
    auto feed = [&](ExternalEngine::ContributionInput& input, size_t offset, bool concurrent)
    {
        std::vector<std::future<void>> pending;
        for (uint16_t member = input.size(); member-- > 0;)
        {
            if (concurrent) pending.push_back(std::async(std::launch::async,
                [&, member] { feed_member(input, member, offset); }));
            else feed_member(input, member, offset);
        }
        for (auto& future : pending) future.get();
        input.seal();
    };
    auto feed_group = [&](ExternalEngine::ContributionInput& input, uint16_t member, uint16_t count, size_t offset)
    {
        ChunkCodec joiner(18, config().codec_bytes);
        std::vector<EncodedChunk> parts(count);
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
        {
            for (uint16_t i = 0; i < count; ++i) parts[i] = {encoded[offset + member + i][size_t(chunk)], 1};
            auto const bytes = joiner.join(chunk, parts);
            input.append_chunk(member, chunk, bytes, count);
        }
        input.finish_member(member, count);
    };
    auto feed_groups = [&](ExternalEngine::ContributionInput& input, size_t offset, bool concurrent)
    {
        std::vector<std::future<void>> pending;
        for (uint16_t first = 0; first < input.size();)
        {
            uint16_t const count = std::min<uint16_t>(first ? 3 : 2, input.size() - first);
            if (concurrent) pending.push_back(std::async(std::launch::async,
                [&, first, count] { feed_group(input, first, count, offset); }));
            else feed_group(input, first, count, offset);
            first += count;
        }
        for (auto& future : pending) future.get();
        input.seal();
    };
    for (bool concurrent : {false, true}) for (bool existing : {false, true})
    {
        auto const path = root / (std::string("contribution-grouped-") + (concurrent ? "concurrent-" : "serial-") + (existing ? "append" : "fresh"));
        auto cfg = config(); cfg.cpu_scheduler = std::make_shared<CodecScheduler>(3);
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        uint16_t const prior = existing ? 2 : 0;
        if (prior) require(engine.append(Members(raw).subspan(0, prior)).accepted, "grouped contribution prefix rejected");
        auto input = engine.prepare_contributions(uint16_t(raw.size() - prior), raw.front().size());
        require(bool(input), "grouped contribution reservation rejected");
        auto const reserved = input->reserved_bytes();
        feed_groups(*input, prior, concurrent);
        require(input->reserved_bytes() <= reserved && input->written_bytes() <= input->reserved_bytes(),
            "grouped contribution exceeded its original reservation");
        require(engine.append(*input).accepted, "grouped contribution append rejected");
        input.reset(); engine.finish(); require(contents(path) == reference, "grouped contributions changed canonical bytes");
    }
    for (bool split : {false, true})
    {
        auto const path = root / (split ? "contribution-grouped-split" : "contribution-grouped-retry");
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_contributions(uint16_t(raw.size()), raw.front().size());
        require(bool(input), "grouped retry reservation rejected");
        auto const reserved = input->reserved_bytes();
        input->reset_member(0, 2);
        std::array<EncodedChunk, 2> parts{{{encoded[0][0], 1}, {encoded[1][0], 1}}};
        auto const bytes = codec.join(0, parts); input->append_chunk(0, 0, bytes, 2);
        rejects([&] { input->append_chunk(1, 0, encoded[1][0]); });
        rejects([&] { input->reset_member(0, 3); });
        rejects([&] { input->finish_member(0); });
        rejects([&] { input->seal(); });
        std::vector<std::future<void>> pending;
        for (uint16_t member = 2; member < input->size(); ++member)
            pending.push_back(std::async(std::launch::async, [&, member] { feed_member(*input, member, 0); }));
        input->reset_member(0, 2);
        if (split) { feed_member(*input, 0, 0); feed_member(*input, 1, 0); }
        else feed_group(*input, 0, 2, 0);
        for (auto& future : pending) future.get();
        input->seal();
        require(input->reserved_bytes() <= reserved, "grouped retry exceeded its reserved capacity");
        require(engine.append(*input).accepted, "grouped retry append rejected");
        engine.finish(); require(contents(path) == reference, "grouped retry disturbed another stream");
    }
    for (uint16_t count : {1, 2})
    {
        auto const prefix_path = root / ("contribution-grouped-prefix-reference-" + std::to_string(count));
        { Engine engine(prefix_path.string(), config(), false, initial_identity()); engine.confirm_identity();
          require(engine.append(Members(raw).subspan(0, count)).accepted, "grouped prefix reference rejected"); }
        auto const prefix = contents(prefix_path); auto cfg = config();
        cfg.capacity_bytes = prefix.size() + ExternalEngine::metadata_bytes(18, ext);
        auto const path = root / ("contribution-grouped-prefix-" + std::to_string(count));
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_contributions(uint16_t(raw.size()), raw.front().size());
        require(bool(input), "grouped prefix reservation rejected"); feed_groups(*input, 0, false);
        require(!engine.append(*input).accepted && !engine.committed_member_count(), "oversized grouped batch committed");
        if (count == 1)
        {
            require(!engine.append(*input, count).accepted && !engine.committed_member_count(),
                "prefix cutting a grouped run committed");
            input.reset();
            require(engine.append(Members(raw).subspan(0, count)).accepted, "ordinary fitting-tail regeneration rejected");
        }
        else require(engine.append(*input, count).accepted, "aligned grouped prefix rejected");
        engine.finish(); require(contents(path) == prefix, "grouped capacity boundary changed canonical prefix bytes");
    }
    for (bool concurrent : {false, true}) for (bool existing : {false, true})
    {
        auto const path = root / (std::string("contribution-") + (concurrent ? "concurrent-" : "serial-") + (existing ? "append" : "fresh"));
        auto cfg = config(); cfg.cpu_scheduler = std::make_shared<CodecScheduler>(3);
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        uint16_t const prior = existing ? 2 : 0;
        if (prior) require(engine.append(Members(raw).subspan(0, prior)).accepted, "ordinary contribution prefix rejected");
        auto input = engine.prepare_contributions(uint16_t(raw.size() - prior), raw.front().size());
        require(bool(input), "contribution input reservation rejected");
        rejects([&] { engine.append(Members(raw).subspan(0, 1)); });
        rejects([&] { engine.start_packing(); });
        feed(*input, prior, concurrent);
        require(input->written_bytes() <= input->reserved_bytes() && engine.committed_member_count() == prior,
            "received contribution exceeded reservation or committed before acceptance");
        require(engine.append(*input).accepted, "valid contribution input rejected");
        input.reset(); engine.finish(); require(contents(path) == reference, "received contributions changed ordinary group bytes");
    }
    {
        auto const path = root / "contribution-member-retry";
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_contributions(uint16_t(raw.size()), raw.front().size());
        require(bool(input), "contribution retry reservation rejected");
        uint64_t const reserved = input->reserved_bytes();
        std::vector<std::future<void>> pending;
        for (uint16_t member = 1; member < input->size(); ++member)
            pending.push_back(std::async(std::launch::async, [&, member] { feed_member(*input, member, 0); }));
        for (uint64_t chunk = 0; chunk < 7; ++chunk) input->append_chunk(0, chunk, encoded[0][size_t(chunk)]);
        input->reset_member(0);
        require(input->reserved_bytes() == reserved, "partial contribution retry changed its scratch reservation");
        auto invalid = encoded[0][0]; invalid.push_back(0);
        rejects([&] { input->append_chunk(0, 0, invalid); });
        input->reset_member(0); feed_member(*input, 0, 0);
        for (auto& future : pending) future.get();
        input->seal(); rejects([&] { input->reset_member(0); });
        require(engine.append(*input).accepted, "restarted contribution input rejected");
        engine.finish(); require(contents(path) == reference, "member retry disturbed another contribution");
    }
    {
        auto const path = root / "contribution-exact"; auto cfg = config();
        cfg.capacity_bytes = reference.size() + ExternalEngine::metadata_bytes(18, ext);
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_contributions(uint16_t(raw.size()), raw.front().size()); require(bool(input), "exact contribution preflight rejected");
        feed(*input, 0, false); require(engine.append(*input).accepted, "exact contribution capacity rejected");
        engine.finish(); require(contents(path) == reference, "exact contribution output differs");
    }
    {
        auto const prefix_path = root / "contribution-prefix-reference";
        { Engine engine(prefix_path.string(), config(), false, initial_identity()); engine.confirm_identity();
          require(engine.append(Members(raw).subspan(0, 1)).accepted, "contribution prefix reference rejected"); }
        auto const prefix = contents(prefix_path); auto cfg = config();
        cfg.capacity_bytes = prefix.size() + ExternalEngine::metadata_bytes(18, ext);
        auto const path = root / "contribution-prefix";
        ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        auto input = engine.prepare_contributions(uint16_t(raw.size()), raw.front().size()); require(bool(input), "prefix contribution preflight rejected");
        feed(*input, 0, false); require(!engine.append(*input).accepted && !engine.committed_member_count(), "oversized received batch committed");
        require(engine.append(*input, 1).accepted, "fitting received prefix rejected");
        engine.finish(); require(contents(path) == prefix, "received prefix output differs");
    }
    {
        auto tight = ext; for (auto& grant : tight.scratch) grant.capacity_bytes = 4096;
        ExternalEngine engine((root / "contribution-no-space").string(), config(), tight, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        require(!engine.prepare_contributions(1, raw.front().size()) && engine.scratch_reserved_bytes() == before,
            "contribution preflight consumed unreserved scratch");
    }
    {
        auto cfg = config();
        cfg.memory_limit_bytes = ExternalEngine::memory_bytes(18, cfg, ext, 1) +
            ExternalEngine::contribution_input_memory_bytes(cfg, 1) - 1;
        ExternalEngine engine((root / "contribution-no-memory").string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        rejects([&] { engine.prepare_contributions(1, raw.front().size()); });
    }
    for (auto const* kind : {"order", "duplicate", "trailing", "truncated", "symbols", "incomplete"})
    {
        ExternalEngine engine((root / (std::string("contribution-invalid-") + kind)).string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        auto input = engine.prepare_contributions(1, std::string_view(kind) == "symbols" ? 1 : raw.front().size());
        require(bool(input), "invalid contribution fixture reservation rejected");
        auto bytes = encoded.front().front();
        if (std::string_view(kind) == "order") rejects([&] { input->append_chunk(0, 1, bytes); });
        else if (std::string_view(kind) == "duplicate")
        { input->append_chunk(0, 0, bytes); rejects([&] { input->append_chunk(0, 0, bytes); }); }
        else if (std::string_view(kind) == "incomplete")
        { input->append_chunk(0, 0, bytes); rejects([&] { input->finish_member(0); }); rejects([&] { input->seal(); }); }
        else
        {
            if (std::string_view(kind) == "trailing") bytes.push_back(0);
            if (std::string_view(kind) == "truncated") bytes.pop_back();
            rejects([&] { input->append_chunk(0, 0, bytes); });
            rejects([&] { input->append_chunk(0, 0, encoded.front().front()); });
        }
        input.reset(); require(engine.scratch_reserved_bytes() == before && !engine.committed_member_count(),
            "invalid contribution retained scratch or changed the committed frontier");
    }
    {
        auto engine = std::make_unique<ExternalEngine>((root / "contribution-detach").string(), config(), ext, false, initial_identity()); engine->confirm_identity();
        auto input = engine->prepare_contributions(1, raw.front().size()); require(bool(input), "contribution detach preflight rejected");
        input->append_chunk(0, 0, encoded.front().front()); engine.reset(); input.reset();
    }
#ifdef GROW_TEST_WRAP_PWRITE
    {
        auto const path = root / "contribution-write-error";
        ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        auto const before = engine.scratch_reserved_bytes();
        auto input = engine.prepare_contributions(1, raw.front().size()); require(bool(input), "write failure contribution preflight rejected");
        fail_next_write = true; rejects([&] { feed_member(*input, 0, 0); }); fail_next_write = false;
        input.reset(); require(engine.scratch_reserved_bytes() == before, "failed contribution write retains scratch");
        require(engine.append(Members(raw)).accepted, "ordinary append after contribution write failure rejected");
        engine.finish(); require(contents(path) == reference, "contribution failure changed later ordinary output");
    }
#endif
    for (bool grouped : {false, true})
    for (auto const* point : {"external-contribution-input-sealed", "external-run-before-checkpoint", "external-run-after-checkpoint"})
    {
        auto const path = root / (std::string(grouped ? "contribution-grouped-crash-" : "contribution-crash-") + point);
        auto const child = ::fork(); require(child >= 0, "cannot fork contribution recovery test");
        if (!child)
        {
            try
            {
                ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
                auto input = engine.prepare_contributions(uint16_t(raw.size()), raw.front().size()); require(bool(input), "crash contribution preflight rejected");
                require(!::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1), "cannot set contribution recovery failpoint");
                if (grouped) feed_groups(*input, 0, false); else feed(*input, 0, false);
                engine.append(*input); std::_Exit(0);
            }
            catch (...) { std::_Exit(1); }
        }
        int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "contribution failpoint did not interrupt");
        ExternalEngine engine(path.string(), config(), ext, true); engine.confirm_identity();
        if (!engine.committed_member_count())
        {
            auto input = engine.prepare_contributions(uint16_t(raw.size()), raw.front().size()); require(bool(input), "resumed contribution preflight rejected");
            if (grouped) feed_groups(*input, 0, true); else feed(*input, 0, false);
            require(engine.append(*input).accepted, "resumed contribution append rejected");
        }
        engine.finish(); require(contents(path) == reference, "contribution recovery changed ordinary bytes");
    }
}

int main(int argc, char** argv)
try
{
    if (argc == 2 && (std::string(argv[1]) == "--io-trace-enabled" || std::string(argv[1]) == "--io-trace-disabled"))
    {
        io_trace(std::string(argv[1]) == "--io-trace-enabled"); std::cout << "external I/O tracing tests passed\n"; return 0;
    }
    bool const independent_io_only = argc == 3 && std::string(argv[1]) == "--independent-io-only";
    require(argc == 1 || independent_io_only || (argc == 2 && (std::string(argv[1]) == "--retirement-only" || std::string(argv[1]) == "--windows-only" || std::string(argv[1]) == "--planning-only" || std::string(argv[1]) == "--checkpoints-only" || std::string(argv[1]) == "--bulk-checkpoints-only" || std::string(argv[1]) == "--prepared-input-only" || std::string(argv[1]) == "--contribution-input-only")),
        "usage: grow-external-test [--retirement-only|--windows-only|--planning-only|--checkpoints-only|--bulk-checkpoints-only|--prepared-input-only|--contribution-input-only|--independent-io-only SCRATCH_PARENT]");
    if (independent_io_only)
    {
#ifdef GROW_TEST_WRAP_PWRITE
        auto const output_parent = fs::absolute(argv[0]).parent_path(), scratch_parent = fs::canonical(argv[2]);
        require(fs::is_directory(scratch_parent), "independent I/O scratch parent must be a directory");
        require_disk_scratch(output_parent); require_disk_scratch(scratch_parent);
        auto const target = storage_device(output_parent), source = storage_device(scratch_parent);
        require(external_detail::independent_packing_io(target, {&source, 1}),
            "independent I/O tests require known physically disjoint output and scratch devices");
        std::string name = (output_parent / "external-io-test-XXXXXX").string();
        require(::mkdtemp(name.data()), "cannot create independent I/O output test directory");
        fs::path const root(name); struct Cleanup { fs::path root; ~Cleanup() { fs::remove_all(root); } } output_cleanup{root};
        name = (scratch_parent / "external-io-scratch-XXXXXX").string();
        require(::mkdtemp(name.data()), "cannot create independent I/O scratch test directory");
        fs::path const scratch(name); Cleanup scratch_cleanup{scratch};
        fs::create_directory(scratch / "scratch-a"); fs::create_directory(scratch / "scratch-b");
        packing_io_policy(); pipeline_barriers(root, external(scratch));
        std::cout << "external independent I/O tests passed\n"; return 0;
#else
        throw std::runtime_error("independent I/O tests require syscall wrapper support");
#endif
    }
    bool const retirement_only = argc == 2 && std::string(argv[1]) == "--retirement-only";
    bool const windows_only = argc == 2 && std::string(argv[1]) == "--windows-only";
    bool const planning_only = argc == 2 && std::string(argv[1]) == "--planning-only";
    bool const checkpoints_only = argc == 2 && std::string(argv[1]) == "--checkpoints-only";
    auto const scratch_parent = fs::absolute(argv[0]).parent_path(); require_disk_scratch(scratch_parent);
    std::string pattern = (scratch_parent / "external-test-XXXXXX").string(); require(::mkdtemp(pattern.data()), "cannot create test directory");
    fs::path const root(pattern); fs::create_directory(root / "scratch-a"); fs::create_directory(root / "scratch-b");
    auto raw = members(retirement_only || windows_only ? 2 : 8); Members input(raw);
    { Engine engine((root / "reference").string(), config(), false, initial_identity()); engine.confirm_identity(); require(engine.append(input).accepted, "reference append rejected"); }
    auto const reference = contents(root / "reference"); auto ext = external(root);
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--contribution-input-only")) contribution_inputs(root, raw, ext, reference);
    if (argc == 2 && std::string(argv[1]) == "--contribution-input-only")
    { fs::remove_all(root); std::cout << "external contribution input tests passed\n"; return 0; }
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--prepared-input-only")) prepared_inputs(root, raw, ext, reference);
    if (argc == 2 && std::string(argv[1]) == "--prepared-input-only")
    { fs::remove_all(root); std::cout << "external prepared input tests passed\n"; return 0; }
    checkpoint_buffer(); packing_io_policy();
    if (argc == 1) shared_storage_cpu(root, input);
    bool const bulk_only = argc == 2 && std::string(argv[1]) == "--bulk-checkpoints-only";
    if (argc == 1 || bulk_only) bulk_checkpoints(root, input, ext);
    if (bulk_only)
    { fs::remove_all(root); std::cout << "external bulk checkpoint tests passed\n"; return 0; }
    if (argc == 1 || checkpoints_only) packing_index(root, ext);
    packing_transition(root, input, ext, reference);
    if (retirement_only)
    {
        retirement(root, input, ext, reference); fs::remove_all(root); std::cout << "external retirement tests passed\n"; return 0;
    }
    std::vector<Member> compact; for (auto const& member : raw) { compact.emplace_back(std::vector<uint64_t>(member)); compact.back().compact(18); }
    if (!windows_only && !checkpoints_only) summary_planning(root, input, Members(compact), ext, reference);
    if (!windows_only && !planning_only) packing_checkpoints(root, Members(compact), ext, reference);
    if (planning_only)
    {
        fs::remove_all(root); std::cout << "external summary planning tests passed\n"; return 0;
    }
    if (checkpoints_only)
    {
        fs::remove_all(root); std::cout << "external packing checkpoint tests passed\n"; return 0;
    }
#ifdef GROW_TEST_WRAP_PWRITE
    pipeline_barriers(root, ext);
#endif
    for (bool const overlap : {false, true})
    {
        auto const suffix = overlap ? "-overlap" : "-serial";
        codec_windows(root, input, ext, reference, std::string("raw") + suffix, overlap);
        codec_windows(root, Members(compact), ext, reference, std::string("compact") + suffix, overlap);
    }
    pipeline_admission(root, Members(compact), ext, reference);
    worker_retuning(root, Members(compact), ext, reference);
    if (windows_only)
    {
        fs::remove_all(root); std::cout << "external codec window tests passed\n"; return 0;
    }
    retained_runs(root, input, ext, reference);
    complete(root / "raw", input, ext, false); require(contents(root / "raw") == reference, "external raw output differs from ordinary bytes");
    auto const inspected = ExternalEngine::inspect((root / "raw").string());
    require(inspected.complete && inspected.members == input.size(), "external completion inspection changed progress");
    complete(root / "raw", input, ext, true); require(contents(root / "raw") == reference, "completed external resume changed bytes");
    complete(root / "compact", Members(compact), ext, false); require(contents(root / "compact") == reference, "external compact output differs");
    {
        auto boundary = initial_identity(); boundary.header.group_id = PlotGroupId(std::string(64, 'f'));
        boundary.header.meta_group = 255; boundary.header.strength = 4; boundary.memo.resize(255);
        for (size_t i = 0; i < boundary.memo.size(); ++i) boundary.memo[i] = uint8_t(i);
        boundary.header.memo_length = uint8_t(boundary.memo.size());
        { Engine engine((root / "boundary-reference").string(), config(), false, boundary); engine.confirm_identity(); require(engine.append(input).accepted, "boundary reference rejected"); }
        { ExternalEngine engine((root / "boundary").string(), config(), ext, false, boundary); engine.confirm_identity(); require(engine.append(input).accepted, "boundary external rejected"); engine.finish(); }
        require(contents(root / "boundary") == contents(root / "boundary-reference"), "external group/meta identity or memo changed ordinary bytes");
    }
    require(ram_backed_filesystem(0x01021994u) && ram_backed_filesystem(0x858458f6u) &&
        ram_backed_filesystem(0x958458f6u) && !ram_backed_filesystem(0xef53), "scratch filesystem classification changed");
#ifdef __linux__
    {
        struct statfs filesystem{};
        if (!::statfs("/dev/shm", &filesystem) && ram_backed_filesystem(uint64_t(filesystem.f_type)))
        {
            auto ram = ext; ram.scratch = {{"/dev/shm", 16ull * 1024 * 1024}}; bool rejected = false;
            try { ExternalEngine engine((root / "ram").string(), config(), ram, false, initial_identity()); }
            catch (std::exception const&) { rejected = true; }
            require(rejected && !fs::exists(root / "ram"), "RAM-backed scratch created an external target before rejection");
        }
    }
#endif
    auto serial = config(); serial.workers = 1; serial.release_cache = true;
    auto const serial_stats = complete(root / "serial", input, ext, false, serial);
    require(serial_stats.packing_workers == 1 && contents(root / "serial") == reference, "serial external packing changed ordinary bytes");
    {
        ExternalEngine engine((root / "buffered").string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted, "buffered fixture rejected"); uint64_t segments = 0;
        for (auto const& grant : ext.scratch) for (auto const& entry : fs::recursive_directory_iterator(grant.directory))
            if (entry.is_regular_file() && entry.path().filename().string().starts_with("run-")) ++segments;
        engine.finish(); auto const& stats = engine.stats(); uint64_t const window = config().window_bytes;
        require(stats.read_calls <= (stats.read_bytes + window - 1) / window + segments + 2,
            "external preread calls exceed the bounded payload/window and segment crossings");
        require(stats.packing_workers > 1 && stats.contribution_workers > 1 && contents(root / "buffered") == reference,
            "parallel external codec did not preserve ordinary bytes");
    }
    for (bool const overlap : {false, true})
        for (std::string const boundary : {"external-initialized-before-checkpoint", "external-initialized-after-checkpoint", "external-contribution-flushed", "external-run-before-checkpoint", "external-run-after-checkpoint", "external-merge-before-checkpoint", "external-merge-after-checkpoint", "external-pack-initialized-before-checkpoint", "external-pack-initialized-after-checkpoint", "external-pack-flushed", "external-pack-before-checkpoint", "external-pack-after-checkpoint", "external-pack-after-checkpoint:2", "external-segment-unlinked", "external-index-flushed", "external-ready-before-checkpoint", "external-ready-after-checkpoint", "external-header-flushed", "external-published-before-checkpoint", "external-published-after-checkpoint", "external-retired-before-checkpoint"})
        {
            auto cfg = config(); cfg.io_overlap = overlap;
            auto const path = root / (std::string(overlap ? "overlap-" : "") + boundary); auto const child = ::fork(); require(child >= 0, "fork failed");
            if (!child)
            {
                ::setenv("XCHPLOT2_GROW_FAILPOINT", boundary.c_str(), 1);
                try { complete(path, input, ext, false, cfg); std::_Exit(0); }
                catch (std::exception const& e) { std::cerr << e.what() << '\n'; std::_Exit(1); }
            }
            int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86, "external recovery boundary was not reached: " + boundary);
            auto before = ExternalEngine::inspect(path.string()); (void)before;
            complete(path, input, ext, true, cfg); require(contents(path) == reference, "external recovery changed output: " + boundary);
        }
    // Exact final-capacity admission must use the canonical index length.
    {
        auto cfg = config(); cfg.capacity_bytes = reference.size() + ExternalEngine::metadata_bytes(18, ext);
        ExternalEngine engine((root / "exact").string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted && engine.stats().sized_chunks, "exact external capacity fallback rejected a fitting group");
        engine.finish(); require(contents(root / "exact") == reference, "exact external capacity changed ordinary bytes");
    }
    // A full scratch grant publishes the acknowledged prefix and leaves the
    // rejected resident member available for the controller's next group.
    std::vector<uint64_t> first_reservation;
    {
        ExternalEngine engine((root / "profile").string(), config(), ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input.subspan(0, 1)).accepted, "external prefix profile rejected"); first_reservation = engine.scratch_reserved_bytes();
        require(first_reservation[0] && first_reservation[1], "external run did not distribute segments across scratch grants"); engine.finish();
    }
    auto prefix_ext = ext;
    for (size_t i = 0; i < prefix_ext.scratch.size(); ++i) prefix_ext.scratch[i].capacity_bytes = first_reservation[i];
    {
        ExternalEngine engine((root / "prefix").string(), config(), prefix_ext, false, initial_identity()); engine.confirm_identity();
        require(engine.append(input.subspan(0, 1)).accepted, "exact scratch admission rejected its measured prefix");
        require(!engine.append(input.subspan(1, 1)).accepted && engine.committed_member_count() == 1, "scratch exhaustion acknowledged a rejected member");
        engine.finish(); require(contents(root / "prefix") == contents(root / "profile"), "full scratch could not publish its committed prefix");
    }
    retirement(root, input, ext, reference);
    auto tight = ext; for (auto& grant : tight.scratch) grant.capacity_bytes = 4096;
    { ExternalEngine engine((root / "tight").string(), config(), tight, false, initial_identity()); engine.confirm_identity();
      auto const refused = engine.append(input); require(!refused.accepted && !engine.committed_member_count(), "scratch exhaustion acknowledged members"); }
    // Empty fallback has its own durable transition before restoring the inode.
    {
        auto const path = root / "empty";
        { ExternalEngine engine(path.string(), config(), tight, false, initial_identity()); engine.confirm_identity();
          require(!engine.append(input).accepted && !engine.finish().members && engine.release_completed(), "empty external fallback failed"); }
        require(!ExternalEngine::pending(path.string()) && !fs::file_size(path), "empty external handoff retained bootstrap bytes");
        Engine engine(path.string(), config(), true, initial_identity()); engine.confirm_identity();
        require(engine.append(input).accepted && contents(path) == reference, "ordinary retry lost the resident member after empty fallback");
    }
    for (std::string const boundary : {"external-empty-before-checkpoint", "external-empty-after-checkpoint", "external-empty-ready-after-checkpoint"})
    {
        auto const path = root / boundary; auto const child = ::fork(); require(child >= 0, "fork failed");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", boundary.c_str(), 1);
            ExternalEngine engine(path.string(), config(), tight, false, initial_identity()); engine.confirm_identity();
            require(!engine.append(input).accepted, "empty fixture unexpectedly fit"); engine.finish(); std::_Exit(0);
        }
        int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86, "empty boundary not reached");
        (void)ExternalEngine::inspect(path.string(), config().memory_limit_bytes);
        { ExternalEngine engine(path.string(), config(), tight, true); engine.confirm_identity(); engine.finish(); engine.release_completed(); }
        require(!ExternalEngine::pending(path.string()) && !fs::file_size(path), "empty recovery did not restore the original inode");
    }
#ifdef GROW_TEST_WRAP_PWRITE
    for (bool const packing : {false, true})
    {
        auto const path = root / (packing ? "failed-pack" : "failed-append"); bool failed = false;
        auto cfg = config(); cfg.cpu_scheduler = std::make_shared<CodecScheduler>(cfg.workers);
        {
            ExternalEngine engine(path.string(), cfg, ext, false, initial_identity()); engine.confirm_identity();
            require(engine.append(input.subspan(0, 1)).accepted, "write-failure fixture rejected"); writes_before_failure = packing ? 1 : 0; fail_next_write = true;
            try { if (packing) engine.finish(); else engine.append(input.subspan(1, 1)); }
            catch (std::exception const&) { failed = true; }
        }
        require(failed && !fail_next_write, "external writer syscall failure was not injected");
        auto const saved = ExternalEngine::inspect(path.string());
        complete(path, input.subspan(0, packing ? 1 : input.size()), ext, true, cfg);
        require(contents(path) == (packing ? contents(root / "profile") : reference), "write-failure recovery changed acknowledged members");
        require(saved.members == 1, "write failure changed acknowledged member count");
    }
#endif
    // Inspection parses its tail and rejects an insufficient allocation budget.
    {
        auto const path = root / "compact", checkpoint = fs::path(path.string() + ".external") / "checkpoint";
        auto const before = contents(path); bool rejected = false;
        try { (void)ExternalEngine::inspect(path.string(), 1); } catch (std::exception const&) { rejected = true; }
        require(rejected && contents(path) == before, "inspection bypassed its memory budget");
        auto original = contents(checkpoint), corrupt = original; corrupt.back() ^= 1;
        auto const hash = group_tools::hex(grow_digest(std::span<uint8_t const>(corrupt).subspan(65)));
        std::copy(hash.begin(), hash.end(), corrupt.begin());
        { std::ofstream out(checkpoint, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<char const*>(corrupt.data()), corrupt.size()); }
        rejected = false; try { (void)ExternalEngine::inspect(path.string()); } catch (std::exception const&) { rejected = true; }
        require(rejected && contents(path) == before, "inspection accepted a malformed late checkpoint field");
        { std::ofstream out(checkpoint, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<char const*>(original.data()), original.size()); }
    }
    // A replacement checkpoint is preserved even after a provisional run was encoded.
    {
        auto const path = root / "checkpoint-replaced", checkpoint = fs::path(path.string() + ".external") / "checkpoint";
        auto const held = fs::path(checkpoint.string() + ".held");
        {
            ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity();
            require(engine.append(input.subspan(0, 1)).accepted, "checkpoint replacement fixture rejected");
            fs::rename(checkpoint, held);
            { std::ofstream out(checkpoint, std::ios::binary); out << "user replacement"; }
            auto const before = contents(checkpoint); bool rejected = false;
            try { (void)engine.append(input.subspan(1, 1)); } catch (std::exception const&) { rejected = true; }
            require(rejected && contents(checkpoint) == before && fs::exists(held), "atomic save overwrote a replacement checkpoint");
        }
        fs::remove(checkpoint); fs::rename(held, checkpoint); complete(path, input, ext, true);
        require(contents(path) == reference, "restored checkpoint lost acknowledged members");
    }
    // Missing required scratch must fail before trimming an uncommitted tail.
    {
        auto const path = root / "missing";
        { ExternalEngine engine(path.string(), config(), ext, false, initial_identity()); engine.confirm_identity(); require(engine.append(input.subspan(0, 1)).accepted, "missing-input fixture rejected"); }
        fs::path segment;
        for (auto const& grant : ext.scratch) for (auto const& entry : fs::recursive_directory_iterator(grant.directory))
            if (entry.is_regular_file() && entry.path().filename().string().starts_with("run-")) segment = entry.path();
        require(!segment.empty(), "missing-input fixture has no segment"); fs::rename(segment, segment.string() + ".held");
        { std::ofstream tail(path, std::ios::binary | std::ios::app); tail << "uncommitted tail"; }
        auto const before = contents(path); bool rejected = false;
        try { ExternalEngine engine(path.string(), config(), ext, true); }
        catch (std::exception const&) { rejected = true; }
        require(rejected && contents(path) == before && fs::exists(segment.string() + ".held"), "missing scratch recovery mutated saved data");
    }
    // Existing files and canonical directory aliases are never adopted.
    {
        auto alias = ext; alias.scratch.push_back(ext.scratch.front()); bool rejected = false;
        try { ExternalEngine engine((root / "alias").string(), config(), alias, false, initial_identity()); }
        catch (std::exception const&) { rejected = true; }
        require(rejected && !fs::exists(root / "alias"), "duplicate scratch alias created an external destination");
        auto const before = contents(root / "reference"); rejected = false;
        try { ExternalEngine engine((root / "reference").string(), config(), ext, false, initial_identity()); }
        catch (std::exception const&) { rejected = true; }
        require(rejected && contents(root / "reference") == before, "external creation overwrote an occupied destination");
    }
    // Moving the same inodes under replacement parents must still fail.
    for (bool const scratch : {false, true})
    {
        auto const base = root / (scratch ? "scratch-parent" : "target-parent");
        fs::create_directories(base / "out"); fs::create_directory(base / "scratch");
        ExternalConfig local{{{(base / "scratch").string(), 16ull * 1024 * 1024}}, 32768, 2};
        auto const path = base / "out" / "group";
        { ExternalEngine engine(path.string(), config(), local, false, initial_identity()); engine.confirm_identity(); require(engine.append(input.subspan(0, 1)).accepted, "parent fixture rejected"); }
        auto const parent = base / (scratch ? "scratch" : "out"), held = base / "held";
        fs::rename(parent, held); fs::create_directory(parent);
        std::vector<fs::path> entries; for (auto const& entry : fs::directory_iterator(held)) entries.push_back(entry.path());
        for (auto const& entry : entries) fs::rename(entry, parent / entry.filename());
        auto const before = contents(path); bool rejected = false;
        try { ExternalEngine engine(path.string(), config(), local, true); } catch (std::exception const&) { rejected = true; }
        require(rejected && contents(path) == before, "replacement parent was accepted or mutated output");
    }
    fs::remove_all(root); std::cout << "external engine tests passed\n"; return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
