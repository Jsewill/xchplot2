// Native persistence for the external-assembly experiment. Sync the replacement
// data and checkpoint before reclaiming any source. No best-effort barriers.
#pragma once
#include "host/GroupFormat.hpp"
#include "host/GroupFile.hpp"
#include "host/GroupConstructionIdentity.hpp"

#include <cerrno>
#include <cstdlib>
#include <tuple>

namespace stream_state
{
namespace file_io = group_grow::file_io;
using group_tools::require;
namespace fs = std::filesystem;

// Deterministic abrupt exits for recovery tests, without running destructors.
// Optional :N selects the Nth occurrence of the named persistence boundary.
void failpoint(std::string const& name)
{
    char const* env = std::getenv("XCHPLOT2_STREAM_FAILPOINT");
    if (!env) return;
    std::string setting(env);
    auto const colon = setting.find(':');
    if (setting.substr(0, colon) != name) return;
    static uint64_t hits = 0;
    uint64_t const target = colon == setting.npos ? 1 : group_tools::number(setting.substr(colon + 1).c_str());
    if (++hits == target) std::_Exit(86);
}
struct Descriptor
{
    int fd;
    explicit Descriptor(int fd) : fd(fd) { require(fd >= 0, "cannot open durability/lock handle"); }
    ~Descriptor() { file_io::close_file(fd); }
    Descriptor(Descriptor const&) = delete;
    Descriptor& operator=(Descriptor const&) = delete;
};
void sync_path(fs::path const& path, bool directory = false)
{
    Descriptor handle(file_io::open_file(path.c_str(), (directory ? file_io::read_only | file_io::directory : file_io::read_write) | file_io::close_on_exec | file_io::no_follow));
    require(file_io::flush(handle.fd) == 0, "cannot sync " + file_io::path_text(path));
}
struct WorkLock
{
    Descriptor handle;
    explicit WorkLock(fs::path const& work)
        : handle(file_io::open_file((work / "lock").c_str(), file_io::create | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600))
    {
        require(file_io::lock_file(handle.fd, file_io::lock_exclusive | file_io::lock_nonblocking) == 0, "work directory is already in use");
    }
};
std::string digest(std::string const& text)
{
    struct sha256 context;
    std::array<uint8_t, 32> result{};
    sha256_init(&context);
    sha256_update(&context, text.data(), text.size());
    sha256_sum(&context, result.data());
    return group_tools::hex(result);
}
// Immutable revisions disable replacement even if another file appears while
// their private temporary file is being written.
void atomic_save(fs::path const& path, std::string const& text, bool replace = true)
{
    std::string temporary = file_io::path_text(path) + ".partial.XXXXXX";
    Descriptor handle(file_io::temp_file(temporary.data()));
    try
    {
        size_t written = 0;
        while (written < text.size())
        {
            file_io::Count const n = file_io::write_file(handle.fd, text.data() + written, text.size() - written);
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "cannot write checkpoint");
            written += size_t(n);
        }
        require(file_io::flush(handle.fd) == 0, "cannot sync checkpoint");
#ifdef _WIN32
        require(!file_io::rename_file(file_io::native_path(temporary), path, replace && fs::exists(file_io::native_path(path))), "cannot publish checkpoint");
#else
        if (replace && fs::exists(file_io::native_path(path))) fs::rename(temporary, path);
        else
        {
            // Initial checkpoint creation also checks the no-overwrite
            // publication mechanism before any raw input becomes reclaimable.
            fs::create_hard_link(temporary, path);
            fs::remove(temporary);
        }
#endif
        sync_path(path.parent_path(), true);
    }
    catch (...)
    {
        (void)file_io::unlink_file(temporary);
        throw;
    }
}

struct Stamp
{
    uint64_t device = 0, inode = 0, bytes = 0;
    int64_t mtime = 0, ctime = 0;
    uint32_t mtime_ns = 0, ctime_ns = 0;
    auto fields() { return std::tie(device, inode, bytes, mtime, ctime, mtime_ns, ctime_ns); }
    bool operator==(Stamp const&) const = default;
};
Stamp stamp(fs::path const& path)
{
    file_io::Stat value{};
    require(file_io::inspect_link(path.c_str(), &value) == 0 && file_io::is_regular(value.st_mode),
            "expected regular file: " + file_io::path_text(path));
#ifdef __APPLE__
    auto const modified = value.st_mtimespec, changed = value.st_ctimespec;
#else
    auto const modified = value.st_mtim, changed = value.st_ctim;
#endif
    return {uint64_t(value.st_dev), uint64_t(value.st_ino), uint64_t(value.st_size),
            modified.tv_sec, changed.tv_sec, uint32_t(modified.tv_nsec), uint32_t(changed.tv_nsec)};
}
// A preflight file size is an allocation bound, even if the file grows or is
// replaced before it is opened. Never follow a changed input to its new EOF.
std::string read_exact(std::istream& in, uint64_t bytes)
{
    require(bytes <= std::numeric_limits<size_t>::max() &&
        bytes <= uint64_t(std::numeric_limits<std::streamsize>::max()), "saved state exceeds the read size limit");
    std::string text(size_t(bytes), '\0');
    in.read(text.data(), std::streamsize(bytes));
    require(uint64_t(in.gcount()) == bytes && !in.bad(), "saved state was truncated while loading");
    require(in.peek() == std::char_traits<char>::eof() && !in.bad(), "saved state grew while loading");
    return text;
}
bool same_output(Stamp const& saved, Stamp const& current)
{
    // Publication changes link count/ctime, but must not change contents/mtime.
    return saved.device == current.device && saved.inode == current.inode &&
        saved.bytes == current.bytes && saved.mtime == current.mtime && saved.mtime_ns == current.mtime_ns;
}
struct Stats
{
    uint64_t input_bytes = 0, raw_payload_read_bytes = 0;
    uint64_t staged_bytes = 0, merge_written_bytes = 0, run_read_bytes = 0;
    uint64_t group_bytes = 0, index_bytes = 0;
    uint64_t work_bytes = 0, peak_work_bytes = 0, reclaimed_bytes = 0;
    uint64_t peak_modeled_bytes = 0, merge_passes = 0, initial_runs = 0;
    uint64_t max_active_members = 0, max_active_runs = 0;
    uint64_t remaining_raw_bytes = 0, reclaimed_raw_bytes = 0, peak_total_file_bytes = 0;

    auto fields()
    {
        return std::tie(input_bytes, raw_payload_read_bytes, staged_bytes, merge_written_bytes,
            run_read_bytes, group_bytes, index_bytes, work_bytes, peak_work_bytes, reclaimed_bytes,
            peak_modeled_bytes, merge_passes, initial_runs, max_active_members, max_active_runs,
            remaining_raw_bytes, reclaimed_raw_bytes, peak_total_file_bytes);
    }
    void peak()
    {
        peak_work_bytes = std::max(peak_work_bytes, work_bytes);
        peak_total_file_bytes = std::max(peak_total_file_bytes, work_bytes + remaining_raw_bytes);
    }
    void add(uint64_t bytes) { work_bytes += bytes; peak(); }
    void remove(fs::path const& path)
    {
        uint64_t const bytes = stamp(path).bytes;
        require(!file_io::unlink_file(path), "cannot reclaim temporary segment");
        work_bytes -= bytes;
        reclaimed_bytes += bytes;
    }
};
struct Run
{
    uint32_t id = 0, first = 0, count = 0;
    uint64_t bytes = 0;
    auto fields() { return std::tie(id, first, count, bytes); }
    bool operator==(Run const&) const = default;
};
fs::path segment_path(fs::path const& work, Run const& run, uint64_t segment)
{
    return work / ("run-" + std::to_string(run.id) + "-" + std::to_string(segment) + ".tmp");
}
struct Config
{
    uint32_t batch = 0, fan = 0;
    uint64_t buffer = 0, segment = 0;
    bool keep = false, consume = false;
    auto fields() { return std::tie(batch, fan, buffer, segment, keep, consume); }
    bool operator==(Config const&) const = default;
};
enum class Phase { stage, merge, final, ready, done };
struct Active
{
    Run output;
    uint64_t chunk = 0, index_bytes = 0;
    uint32_t carry_bits = 0, carry_value = 0;
    std::vector<Run> inputs;
    std::vector<uint64_t> positions;
};
struct State
{
    unsigned version = 2;
    std::string producer_build = STREAM_IDENTITY;
    Config config;
    Phase phase = Phase::stage;
    uint32_t staged = 0, next_id = 0, merge_first = 0;
    Stats stats;
    std::vector<Stamp> raw;
    std::vector<Run> runs, next_runs;
    bool active = false;
    Active operation;
    Stamp completed;
};
void validate_graph(State const& state, uint64_t members, uint64_t chunks)
{
    auto coverage = [&](std::vector<Run> const& runs, uint64_t expected)
    {
        uint64_t next = 0;
        for (auto run : runs)
        {
            require(run.first == next && run.count > 0 && run.id < state.next_id && run.bytes > 0,
                    "checkpoint run coverage is inconsistent");
            next += run.count;
        }
        require(next == expected, "checkpoint run coverage is incomplete");
    };
    if (state.phase == Phase::done)
    {
        require(!state.active && state.runs.empty() && state.next_runs.empty() && state.staged == members,
                "invalid completed checkpoint");
    }
    else
    {
        coverage(state.runs, state.phase == Phase::stage ? state.staged : members);
        uint64_t const merged = state.merge_first == state.runs.size() ? members :
            state.runs[state.merge_first].first;
        coverage(state.next_runs, state.phase == Phase::merge ? merged : 0);
        if (state.active)
        {
            auto const& active = state.operation;
            require(active.output.id < state.next_id && active.output.count && !active.inputs.empty(),
                    "invalid active checkpoint output");
            uint64_t next = active.output.first;
            for (size_t i = 0; i < active.inputs.size(); ++i)
            {
                auto const& run = active.inputs[i];
                require(run.first == next && run.id != active.output.id &&
                            (!active.chunk ? active.positions[i] == 0 : active.positions[i] > 0) &&
                            (active.chunk != chunks || active.positions[i] == run.bytes),
                        "checkpoint chunk/input coverage is inconsistent");
                next += run.count;
            }
            require(next == active.output.first + uint64_t(active.output.count),
                    "checkpoint output membership mismatch");
            if (state.phase == Phase::final || state.phase == Phase::ready)
                require(active.output.first == 0 && active.output.count == members &&
                            active.output.bytes == state.stats.group_bytes &&
                            active.index_bytes == state.stats.index_bytes,
                        "checkpoint final file lengths are inconsistent");
        }
    }
    if (state.phase >= Phase::ready)
        require(state.completed.bytes == state.stats.group_bytes && state.completed.bytes > 51,
                "invalid completed file length");
}
template<typename T> void write_fields(std::ostream& out, T value)
{
    std::apply([&](auto&... field) { ((out << field << ' '), ...); }, value.fields());
    out << '\n';
}
template<typename T> void read_fields(std::istream& in, T& value)
{
    std::apply([&](auto&... field) { ((in >> field), ...); }, value.fields());
}
void save_state(fs::path const& work, State const& state, std::string const& identity,
                std::string const& label)
{
    std::ostringstream out;
    out << "xchplot2-stream-" << state.version << '\n' << state.producer_build << '\n';
    if (state.version == 2)
    {
        auto const compatibility = group_grow::construction_compatibility({});
        out << group_grow::construction_build_text(compatibility.algorithm) << '\n'
            << group_tools::hex(compatibility.representation) << '\n';
    }
    out << identity << '\n';
    write_fields(out, state.config);
    out << int(state.phase) << ' ' << state.staged << ' ' << state.next_id << ' '
        << state.merge_first << '\n';
    write_fields(out, state.stats);
    out << state.raw.size() << '\n';
    for (auto value : state.raw) write_fields(out, value);
    for (auto const* runs : {&state.runs, &state.next_runs})
    {
        out << runs->size() << '\n';
        for (auto value : *runs) write_fields(out, value);
    }
    out << state.active << '\n';
    if (state.active)
    {
        auto const& active = state.operation;
        write_fields(out, active.output);
        out << active.chunk << ' ' << active.index_bytes << ' ' << active.carry_bits << ' '
            << active.carry_value << '\n' << active.inputs.size() << '\n';
        for (size_t i = 0; i < active.inputs.size(); ++i)
        {
            write_fields(out, active.inputs[i]);
            out << active.positions[i] << '\n';
        }
    }
    write_fields(out, state.completed);
    std::string const body = out.str();
    // Include newly created data filenames before making the new cursors durable.
    sync_path(work, true);
    failpoint(label + "-before-checkpoint");
    atomic_save(work / "checkpoint", digest(body) + '\n' + body);
    failpoint(label + "-after-checkpoint");
}
State load_state(fs::path const& work, Config const& config, std::string const& identity,
                 uint64_t members, uint64_t limit, uint64_t chunks)
{
    auto const path = work / "checkpoint";
    auto const saved_stamp = stamp(path);
    require(saved_stamp.bytes < limit / 8, "checkpoint exceeds RAM budget");
    auto file = group_tools::input(file_io::path_text(path));
    auto body = read_exact(file, saved_stamp.bytes);
    require(stamp(path) == saved_stamp, "checkpoint changed while loading");
    require(body.size() >= 65 && body[64] == '\n', "invalid checkpoint checksum");
    auto const hash = body.substr(0, 64);
    body.erase(0, 65);
    require(hash == digest(body), "checkpoint checksum mismatch");
    std::istringstream in(body);
    in.exceptions(std::ios::badbit | std::ios::failbit);
    std::string format, source, job;
    in >> format >> source;
    bool const legacy = format == "xchplot2-stream-1";
    group_grow::check_construction_format(legacy || format == "xchplot2-stream-2", "stream");
    if (legacy) group_grow::check_construction_build(source, STREAM_IDENTITY, "stream");
    else
    {
        std::string algorithm, representation;
        in >> algorithm >> representation;
        require(algorithm == group_grow::construction_algorithm_identity,
            "stream checkpoint algorithm is incompatible; resume with a build supporting its recorded algorithm or restart this construction");
        group_grow::ConstructionCompatibility compatibility;
        compatibility.algorithm = group_grow::construction_build(algorithm);
        auto const bytes = group_tools::hex(representation);
        require(bytes.size() == compatibility.representation.size(), "invalid stream checkpoint representation identity");
        std::copy(bytes.begin(), bytes.end(), compatibility.representation.begin());
        group_grow::check_construction_compatibility(compatibility, group_grow::construction_compatibility({}), "stream");
    }
    in >> job;
    require(job == identity, "checkpoint group input identity mismatch");
    State state;
    state.version = legacy ? 1 : 2; state.producer_build = source;
    read_fields(in, state.config);
    require(state.config == config, "checkpoint assembly settings mismatch");
    int phase;
    in >> phase >> state.staged >> state.next_id >> state.merge_first;
    require(phase >= 0 && phase <= int(Phase::done) && state.staged <= members &&
                state.next_id <= members * 2, "invalid checkpoint progress");
    state.phase = Phase(phase);
    read_fields(in, state.stats);
    size_t count;
    in >> count;
    require(count == members, "checkpoint raw membership mismatch");
    state.raw.resize(count);
    for (auto& value : state.raw) read_fields(in, value);
    for (auto* runs : {&state.runs, &state.next_runs})
    {
        in >> count;
        require(count <= members, "invalid checkpoint run count");
        runs->resize(count);
        for (auto& value : *runs)
        {
            read_fields(in, value);
            require(value.count && value.first + uint64_t(value.count) <= members &&
                        value.id < state.next_id && value.bytes > 0, "invalid checkpoint run");
        }
    }
    require(state.merge_first <= state.runs.size(), "invalid checkpoint merge position");
    in >> state.active;
    if (state.active)
    {
        auto& active = state.operation;
        read_fields(in, active.output);
        in >> active.chunk >> active.index_bytes >> active.carry_bits >> active.carry_value >> count;
        require(count > 0 && count <= config.fan && active.chunk <= chunks &&
                    active.carry_bits < 8 && active.carry_value < (1u << active.carry_bits),
                "invalid checkpoint operation");
        active.inputs.resize(count);
        active.positions.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            read_fields(in, active.inputs[i]);
            in >> active.positions[i];
            require(active.positions[i] <= active.inputs[i].bytes && active.inputs[i].count &&
                        active.inputs[i].first + uint64_t(active.inputs[i].count) <= members &&
                        active.inputs[i].id < state.next_id, "invalid checkpoint input cursor");
        }
    }
    read_fields(in, state.completed);
    in.exceptions(std::ios::badbit);
    std::string extra;
    require(!(in >> extra) && in.eof(), "unexpected checkpoint fields");
    require((state.phase == Phase::stage && !state.active) || state.staged == members,
            "checkpoint skipped raw staging");
    require((state.phase != Phase::final && state.phase != Phase::ready) || state.active,
            "checkpoint missing final operation");
    validate_graph(state, members, chunks);
    return state;
}

void validate_run(fs::path const& work, Run const& run, uint64_t position, uint64_t segment)
{
    if (position == run.bytes) return;
    for (uint64_t offset = position / segment * segment; offset < run.bytes; offset += segment)
    {
        uint64_t const bytes = std::min(segment, run.bytes - offset);
        require(stamp(segment_path(work, run, offset / segment)).bytes == bytes,
                "committed temporary segment is truncated or changed");
    }
}
void validate_run_prefix(fs::path const& work, Run const& run, uint64_t segment)
{
    for (uint64_t offset = 0; offset < run.bytes; offset += segment)
    {
        uint64_t const bytes = stamp(segment_path(work, run, offset / segment)).bytes;
        require(bytes >= std::min(segment, run.bytes - offset) && bytes <= segment,
                "committed merge output is truncated");
    }
}
void validate_prefix(fs::path const& path, uint64_t bytes)
{
    require(!bytes || (fs::exists(file_io::native_path(path)) && stamp(path).bytes >= bytes), "committed output is truncated");
    if (fs::exists(file_io::native_path(path))) (void)stamp(path);
}
void truncate_prefix(fs::path const& path, uint64_t bytes)
{
    validate_prefix(path, bytes);
    if (fs::exists(file_io::native_path(path))) fs::resize_file(path, bytes);
}
void rollback_run(fs::path const& work, Run const& run, uint64_t segment, Stats& stats)
{
    std::string const prefix = "run-" + std::to_string(run.id) + "-";
    for (auto const& entry : fs::directory_iterator(file_io::native_path(work)))
    {
        auto const name = file_io::path_text(entry.path().filename());
        if (!name.starts_with(prefix) || !name.ends_with(".tmp")) continue;
        auto const index = group_tools::number(name.substr(prefix.size(), name.size() - prefix.size() - 4).c_str());
        uint64_t const complete = run.bytes / segment;
        if (index < complete) continue;
        if (index == complete && run.bytes % segment)
        {
            uint64_t const kept = run.bytes % segment;
            uint64_t const old = stamp(entry.path()).bytes;
            truncate_prefix(entry.path(), kept);
            stats.work_bytes -= old - kept;
        }
        else stats.remove(entry.path());
    }
}
uint64_t work_file_bytes(fs::path const& work)
{
    uint64_t bytes = 0;
    for (auto const& entry : fs::directory_iterator(file_io::native_path(work)))
    {
        auto const name = file_io::path_text(entry.path().filename());
        if ((name.starts_with("run-") && name.ends_with(".tmp")) || name == "index.tmp" ||
            name == "group.gplot" || (name == "group.gplot.partial" && !fs::exists(file_io::native_path(work / "group.gplot"))))
            bytes += stamp(entry.path()).bytes;
    }
    return bytes;
}
} // namespace stream_state
