// A locked, immutable owner inode and a bounded heartbeat distinguish current
// local ownership from durable observations left by a stopped coordinator.
#pragma once
#include "host/GroupFarmUpdate.hpp"
#include "host/Random.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>

namespace group_grow
{
inline constexpr uint64_t farming_owner_record_bytes = 4096;
inline uint64_t farming_owner_reservation(uint64_t alignment)
{
    auto const record = farming_update_aligned(farming_owner_record_bytes, alignment);
    require(record <= UINT64_MAX / 4, "farming owner reservation overflow"); return record * 4;
}
inline uint64_t farming_owner_memory_bound(std::filesystem::path const& root)
{ return grow_add(64 * 1024, farming_metadata_memory_bound(root)); }
namespace farming_owner_detail
{
namespace fs = std::filesystem;
inline constexpr char filename[] = "grow-farm.owner", heartbeat_filename[] = "grow-farm.heartbeat";
inline uint64_t now_ms()
{
    auto const value = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    require(value >= 0, "invalid farming owner observation clock"); return uint64_t(value);
}
inline bool filename_matches(std::string const& name)
{
    if (name == filename || name == heartbeat_filename) return true;
    for (auto const* base : {filename, heartbeat_filename})
    {
        auto const prefix = std::string(base) + ".partial.";
        if (!name.starts_with(prefix)) continue;
        auto const suffix = std::string_view(name).substr(prefix.size());
        return suffix.size() == 6 && std::all_of(suffix.begin(), suffix.end(), [](unsigned char c)
        { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); });
    }
    return false;
}
inline FarmingStateFile read_record(fs::path const& path, uint64_t memory_limit)
{
    for (unsigned attempt = 0; ; ++attempt)
    {
        auto const before = stream_state::stamp(path);
        require(before.bytes <= farming_owner_record_bytes, "farming owner metadata exceeds its read bound");
        try { return read_farming_state(path, memory_limit); }
        catch (std::runtime_error const&)
        { if (attempt >= 2 || stream_state::stamp(path) == before) throw; }
    }
}
inline uint64_t charge(file_io::Stat const& info)
{
    require(file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
        info.st_size >= 0 && info.st_blocks >= 0 && uint64_t(info.st_blocks) <= UINT64_MAX / 512,
        "invalid farming owner metadata allocation");
    return std::max(uint64_t(info.st_size), uint64_t(info.st_blocks) * 512);
}
inline uint64_t charge(fs::path const& path)
{
    file_io::Stat info{}; require(!file_io::inspect_link(path, &info), "cannot inspect farming owner metadata allocation");
    return charge(info);
}
inline std::string checked(std::string body)
{
    body += stream_state::digest(body) + '\n';
    require(body.size() <= farming_owner_record_bytes, "farming owner record exceeds its bound"); return body;
}
struct Record
{
    std::string job, fingerprint, directory, instance, host, boot, pid_namespace, process_start;
    uint64_t pid = 0;
    int descriptor = -1;
};
inline std::string text(Record const& value)
{
    require(farming_update_digest(value.job) && farming_update_digest(value.fingerprint) && farming_update_digest(value.instance) &&
        !value.directory.empty() && value.directory.size() <= 256 && value.host.size() <= 256 && value.boot.size() <= 256 &&
        value.pid_namespace.size() <= 256 && value.process_start.size() <= 256 && value.pid && value.descriptor >= -1,
        "invalid farming owner identity");
    std::ostringstream out;
    out << "ordinary-group-farming-owner-v2\n" << std::quoted(value.job) << ' ' << std::quoted(value.fingerprint) << ' '
        << std::quoted(value.directory) << ' ' << std::quoted(value.instance) << '\n'
        << std::quoted(value.host) << ' ' << std::quoted(value.boot) << ' ' << std::quoted(value.pid_namespace) << ' '
        << value.pid << ' ' << std::quoted(value.process_start) << ' ' << value.descriptor << '\n';
    return out.str();
}
inline Record parse(std::string const& body)
{
    require(body.size() <= farming_owner_record_bytes, "farming owner record exceeds its bound");
    std::istringstream in(body); std::string magic; std::getline(in, magic); Record result;
    in >> std::quoted(result.job) >> std::quoted(result.fingerprint) >> std::quoted(result.directory) >> std::quoted(result.instance)
        >> std::quoted(result.host) >> std::quoted(result.boot) >> std::quoted(result.pid_namespace) >> result.pid >> std::quoted(result.process_start);
    require(in && (magic == "ordinary-group-farming-owner-v1" || magic == "ordinary-group-farming-owner-v2"), "unsupported farming owner record");
    if (magic == "ordinary-group-farming-owner-v2") in >> result.descriptor;
    require(bool(in), "incomplete farming owner descriptor binding");
    in >> std::ws; require(in.eof(), "unexpected farming owner data"); (void)text(result); return result;
}
struct Heartbeat
{
    std::string instance, control, state = "running";
    uint64_t at = 0, sequence = 1;
};
inline std::string text(Heartbeat const& value)
{
    require(farming_update_digest(value.instance) && farming_update_digest(value.control) && value.at && value.sequence &&
        (value.state == "running" || value.state == "stopped"), "invalid farming owner heartbeat");
    std::ostringstream out; out << "ordinary-group-farming-heartbeat-v1\n" << std::quoted(value.instance) << ' '
        << std::quoted(value.control) << ' ' << value.at << ' ' << value.sequence << ' ' << std::quoted(value.state) << '\n'; return out.str();
}
inline Heartbeat parse_heartbeat(std::string const& body)
{
    require(body.size() <= farming_owner_record_bytes, "farming heartbeat exceeds its bound");
    std::istringstream in(body); std::string magic; std::getline(in, magic); Heartbeat result;
    in >> std::quoted(result.instance) >> std::quoted(result.control) >> result.at >> result.sequence >> std::quoted(result.state);
    require(in && magic == "ordinary-group-farming-heartbeat-v1", "unsupported farming owner heartbeat");
    in >> std::ws; require(in.eof(), "unexpected farming heartbeat data"); (void)text(result); return result;
}
#if defined(__linux__)
inline std::string kernel_identity()
{
    std::ifstream input("/proc/sys/kernel/random/boot_id"); std::string result;
    std::getline(input, result); return input && result.size() <= 256 ? result : std::string{};
}
inline std::string pid_namespace(uint64_t pid)
{
    if (!pid || pid > uint64_t(INT_MAX)) return {};
    std::array<char, 256> value{}; auto const path = "/proc/" + std::to_string(pid) + "/ns/pid";
    auto const bytes = ::readlink(path.c_str(), value.data(), value.size());
    return bytes > 0 && size_t(bytes) < value.size() ? std::string(value.data(), size_t(bytes)) : std::string{};
}
inline std::string process_start(uint64_t pid)
{
    if (!pid || pid > uint64_t(INT_MAX)) return {};
    std::ifstream input("/proc/" + std::to_string(pid) + "/stat"); std::string line;
    std::getline(input, line); auto const command_end = line.rfind(')');
    if (!input || line.size() > 4096 || command_end == line.npos) return {};
    std::istringstream fields(line.substr(command_end + 1)); std::string value;
    for (unsigned field = 3; field <= 22; ++field)
    {
        fields >> value;
        if (!fields || (field == 3 && (value == "Z" || value == "X"))) return {};
    }
    return value.find_first_not_of("0123456789") == value.npos ? value : std::string{};
}
#elif defined(_WIN32)
inline std::string process_start(uint64_t pid)
{
    if (!pid || pid > MAXDWORD) return {};
    HANDLE process = ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, DWORD(pid));
    if (!process) return {};
    FILETIME created{}, exited{}, kernel{}, user{}; std::string result;
    if (::GetProcessTimes(process, &created, &exited, &kernel, &user) && ::WaitForSingleObject(process, 0) == WAIT_TIMEOUT)
        result = std::to_string((uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime);
    ::CloseHandle(process); return result;
}
inline std::wstring mutex_name(std::string const& instance)
{ return L"Local\\xchplot2-farm-owner-" + std::wstring(instance.begin(), instance.end()); }
struct LocalProof
{
    HANDLE handle = nullptr;
    ~LocalProof() { if (handle) { ::ReleaseMutex(handle); ::CloseHandle(handle); } }
};
#endif
inline Record local_record(FarmingState const& state, size_t lane)
{
    require(lane < state.directories.size(), "farming owner destination is outside the saved job");
    Record value; value.job = state.job_id; value.fingerprint = state.fingerprint;
    value.directory = state.directories[lane].directory_identity; value.host = profile_detail::host_name();
    std::array<uint8_t, 32> nonce{};
    pos2gpu::read_random_bytes(nonce.data(), nonce.size()); value.instance = group_tools::hex(nonce);
#ifdef _WIN32
    value.pid = ::GetCurrentProcessId(); value.process_start = process_start(value.pid);
#elif defined(__linux__)
    value.pid = uint64_t(::getpid()); value.boot = kernel_identity(); value.pid_namespace = pid_namespace(value.pid);
    value.process_start = process_start(value.pid);
#else
    value.pid = uint64_t(::getpid());
#endif
    return value;
}
inline bool local_process(Record const& record, stream_state::Stamp const& owner_stamp)
{
#ifdef _WIN32
    (void)owner_stamp;
    HANDLE proof = ::OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE, mutex_name(record.instance).c_str());
    if (!proof) return false;
    auto const observed = ::WaitForSingleObject(proof, 0);
    if (observed == WAIT_OBJECT_0 || observed == WAIT_ABANDONED) ::ReleaseMutex(proof);
    ::CloseHandle(proof);
    return (observed == WAIT_TIMEOUT || record.pid == ::GetCurrentProcessId()) && !record.process_start.empty() &&
        process_start(record.pid) == record.process_start;
#elif defined(__linux__)
    bool const process = !record.boot.empty() && record.boot == kernel_identity() && !record.pid_namespace.empty() &&
        record.pid_namespace == pid_namespace(uint64_t(::getpid())) && record.pid_namespace == pid_namespace(record.pid) &&
        !record.process_start.empty() && process_start(record.pid) == record.process_start;
    if (!process || record.descriptor < 0) return false;
    auto const base = "/proc/" + std::to_string(record.pid);
    auto const descriptor = base + "/fd/" + std::to_string(record.descriptor);
    auto same_inode = [&]
    {
        file_io::Stat info{};
        return !file_io::inspect_path(descriptor, &info) && file_io::is_regular(info.st_mode) &&
            uint64_t(info.st_dev) == owner_stamp.device && uint64_t(info.st_ino) == owner_stamp.inode;
    };
    if (!same_inode()) return false;
    std::ifstream input(base + "/fdinfo/" + std::to_string(record.descriptor), std::ios::binary);
    std::array<char, 4096> data{}; input.read(data.data(), std::streamsize(data.size())); auto const bytes = input.gcount();
    if (input.bad() || !input.eof() || bytes <= 0 || uint64_t(bytes) >= data.size()) return false;
    std::istringstream lines(std::string(data.data(), size_t(bytes))); std::string line;
    bool writable = false, locked = false;
    while (std::getline(lines, line))
    {
        if (line.starts_with("flags:"))
        {
            uint64_t flags = 0; std::istringstream value(line.substr(6)); value >> std::oct >> flags; value >> std::ws;
            writable = value.eof() && !value.bad() && (flags & O_ACCMODE) == O_RDWR;
        }
        else if (line.starts_with("lock:"))
        {
            std::string label, index, kind, advisory, mode; uint64_t pid = 0; std::istringstream value(line);
            value >> label >> index >> kind >> advisory >> mode >> pid;
            locked = locked || (value && kind == "FLOCK" && advisory == "ADVISORY" && mode == "WRITE" && pid == record.pid);
        }
    }
    return writable && locked && same_inode() && process_start(record.pid) == record.process_start;
#else
    (void)record; (void)owner_stamp; return false;
#endif
}
} // namespace farming_owner_detail
inline uint64_t farming_owner_allocation(FarmingMetadataDirectory const& metadata)
{
    metadata.check(); if (!metadata.root_pin || (!metadata.legacy && !metadata.private_pin)) return 0;
    uint64_t bytes = 0; size_t files = 0;
    for (auto const& entry : std::filesystem::directory_iterator(metadata.path))
        if (farming_owner_detail::filename_matches(file_io::path_text(entry.path().filename())))
        {
            require(++files <= 1024, "farming owner metadata exceeds its file bound");
            file_io::Stat current{};
            int const inspected = file_io::inspect_link(entry.path(), &current);
            if (inspected && errno == ENOENT && file_io::path_text(entry.path().filename()).find(".partial.") != std::string::npos) continue;
            require(!inspected, "cannot inspect farming owner metadata allocation");
            bytes = grow_add(bytes, farming_owner_detail::charge(current));
        }
    metadata.check(); return bytes;
}
inline void admit_farming_owner_record(FarmingMetadataDirectory const& metadata, uint64_t reserved, uint64_t headroom,
    std::string const& contents)
{
    auto const used = farming_owner_allocation(metadata); file_io::Space space{};
    require(!file_io::space_path(metadata.root, &space) && space.f_frsize && space.f_bavail <= UINT64_MAX / space.f_frsize,
        "cannot inspect farming owner metadata allowance");
    auto const replacement = farming_update_aligned(contents.size(), space.f_frsize);
    require(used <= reserved && replacement <= reserved - used, "farming owner replacement exceeds its reserved metadata allowance");
    auto const available = space.f_bavail * space.f_frsize;
    require(headroom <= available && replacement <= available - headroom, "farming owner metadata would consume declared headroom");
}
inline void check_farming_owner_allowance(FarmingMetadataDirectory const& metadata, uint64_t reserved, uint64_t headroom)
{
    require(farming_owner_allocation(metadata) <= reserved, "farming owner allocation exceeded its metadata grant");
    file_io::Space space{};
    require(!file_io::space_path(metadata.root, &space) && space.f_frsize && space.f_bavail <= UINT64_MAX / space.f_frsize &&
        space.f_bavail * space.f_frsize >= headroom, "farming owner publication consumed declared filesystem headroom");
}
struct FarmingOwnerHealth
{
    std::string status = "not-observed", instance, host, control_digest, last_state;
    uint64_t heartbeat_at_ms = 0, heartbeat_sequence = 0;
    bool checked = false, lock_held = false, local_process_proven = false, heartbeat_current = false;
};
inline FarmingOwnerHealth observe_farming_owner(FarmingMetadataDirectory const& metadata, FarmingState const& state,
    size_t lane, uint64_t memory_limit, std::optional<std::string> const& approved_source = {})
{
    using namespace farming_owner_detail;
    require(lane < state.directories.size(), "farming owner destination is outside the saved job");
    metadata.check(); FarmingOwnerHealth result; auto const path = metadata.path / filename;
    if (!metadata.root_pin || (!metadata.legacy && !metadata.private_pin) || !farm_metadata_detail::present(path))
    { result.checked = true; result.status = "not-recorded"; return result; }
    auto const saved = read_record(path, memory_limit); auto const owner = parse(saved.body);
    auto const& identity = state.directories[lane].directory_identity;
    bool const relocated = owner.directory != identity && approved_source && owner.directory == *approved_source;
    require(owner.job == state.job_id && owner.fingerprint == state.fingerprint &&
        (owner.directory == identity || relocated) && identity == farming_directory_identity(metadata.root),
        "farming owner belongs to a different job or native destination");
    farm::inventory_detail::Handle handle(file_io::open_file(path, file_io::read_only | file_io::close_on_exec | file_io::no_follow));
    auto const pinned = farm::inventory_detail::stat(handle);
    require(uint64_t(pinned.st_dev) == saved.stamp.device && uint64_t(pinned.st_ino) == saved.stamp.inode && pinned.st_nlink == 1,
        "farming owner changed before its lock observation");
    if (file_io::lock_file(handle.fd, file_io::lock_exclusive | file_io::lock_nonblocking))
    { require(errno == EWOULDBLOCK || errno == EAGAIN || errno == EBUSY, "cannot observe farming owner lock"); result.lock_held = true; }
    handle = farm::inventory_detail::Handle{};
    result.checked = true; result.instance = owner.instance; result.host = owner.host;
    result.local_process_proven = !relocated && result.lock_held && local_process(owner, saved.stamp);
    auto const heartbeat_path = metadata.path / heartbeat_filename;
    if (farm_metadata_detail::present(heartbeat_path))
    {
        auto const heartbeat_file = read_record(heartbeat_path, memory_limit); auto const heartbeat = parse_heartbeat(heartbeat_file.body);
        if (heartbeat.instance == owner.instance)
        {
            result.control_digest = heartbeat.control; result.last_state = heartbeat.state;
            result.heartbeat_at_ms = heartbeat.at; result.heartbeat_sequence = heartbeat.sequence;
            auto const now = now_ms(); result.heartbeat_current = heartbeat.at <= now && now - heartbeat.at <= 10000;
        }
    }
    result.status = relocated ? (result.lock_held ? "relocated-active-owner" : "relocated-stale-owner") :
        !result.lock_held ? (result.last_state == "stopped" ? "stopped" : "stale-owner") :
        !result.local_process_proven ? "remote-or-unknown-owner" : !result.heartbeat_current ? "unresponsive-owner" :
        result.last_state == "stopped" ? "stopping" : "live";
    require(stream_state::stamp(path) == saved.stamp, "farming owner changed during its lock observation");
    metadata.check(); return result;
}
// All source/target construction locks and the relocation request fence stay
// held across this handoff and the succeeding plan publication. A copied
// observation never becomes proof that the target has a live writer.
inline void handoff_farming_owner(FarmingMetadataDirectory const& metadata, FarmingState const& state, size_t lane,
    std::string const& source_identity, uint64_t reserved_bytes, uint64_t memory_limit)
{
    using namespace farming_owner_detail;
    metadata.check(); auto const path = metadata.path / filename;
    if (!farm_metadata_detail::present(path)) return;
    require(farming_owner_memory_bound(metadata.root) < memory_limit, "farming owner native handoff exceeds --max-ram");
    file_io::Space alignment{};
    require(!file_io::space_path(metadata.root, &alignment) && reserved_bytes >= farming_owner_reservation(alignment.f_frsize),
        "farming owner native handoff lacks its reserved metadata allowance");
    auto health = observe_farming_owner(metadata, state, lane, memory_limit, source_identity);
    for (unsigned retry = 0; health.lock_held && retry < 3; ++retry)
    { std::this_thread::sleep_for(std::chrono::milliseconds(5)); health = observe_farming_owner(metadata, state, lane, memory_limit, source_identity); }
    require(!health.lock_held, "native farming owner handoff cannot replace an active ownership inode");
    auto const file = read_record(path, memory_limit); auto owner = parse(file.body);
    if (owner.directory == state.directories[lane].directory_identity) return;
    require(owner.directory == source_identity, "native farming owner handoff lacks its approved source identity");
    owner.directory = state.directories[lane].directory_identity; owner.boot.clear(); owner.pid_namespace.clear(); owner.process_start.clear(); owner.descriptor = -1;
    Heartbeat heartbeat; heartbeat.instance = owner.instance; heartbeat.control = stream_state::digest("");
    auto const heartbeat_path = metadata.path / heartbeat_filename;
    if (farm_metadata_detail::present(heartbeat_path))
    {
        heartbeat = parse_heartbeat(read_record(heartbeat_path, memory_limit).body);
        require(heartbeat.instance == owner.instance, "copied farming heartbeat belongs to another ownership instance");
    }
    require(heartbeat.sequence < UINT64_MAX, "farming owner native handoff sequence exhausted");
    ++heartbeat.sequence; heartbeat.at = now_ms(); heartbeat.state = "stopped";
    auto const stopped = checked(text(heartbeat)); admit_farming_owner_record(metadata, reserved_bytes, state.headroom, stopped);
    require(stream_state::stamp(path) == file.stamp, "copied farming owner changed before native handoff");
    stream_state::atomic_save(heartbeat_path, stopped); check_farming_owner_allowance(metadata, reserved_bytes, state.headroom);
    grow_failpoint("farm-owner-handoff-heartbeat");
    auto const rebound = checked(text(owner)); admit_farming_owner_record(metadata, reserved_bytes, state.headroom, rebound);
    require(stream_state::stamp(path) == file.stamp, "copied farming owner changed during native handoff");
    stream_state::atomic_save(path, rebound); check_farming_owner_allowance(metadata, reserved_bytes, state.headroom);
    metadata.check(); grow_failpoint("farm-owner-handoff-published");
}

class FarmingOwner
{
    FarmingMetadataDirectory metadata_;
    farming_owner_detail::Record owner_;
    farming_owner_detail::Heartbeat heartbeat_;
    farm::inventory_detail::Handle owner_handle_;
    uint64_t reserved_, headroom_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::thread thread_;
    bool stopping_ = false;
#ifdef _WIN32
    farming_owner_detail::LocalProof proof_;
#endif
    void admit(std::string const& contents)
    {
        admit_farming_owner_record(metadata_, reserved_, headroom_, contents);
    }
    void publish_heartbeat(std::string_view state)
    {
        using namespace farming_owner_detail;
        require(heartbeat_.sequence < UINT64_MAX, "farming heartbeat sequence exhausted");
        heartbeat_.at = now_ms(); heartbeat_.state = state; ++heartbeat_.sequence;
        auto const contents = checked(text(heartbeat_)); admit(contents); metadata_.check();
        stream_state::atomic_save(metadata_.path / heartbeat_filename, contents); metadata_.check();
        check_farming_owner_allowance(metadata_, reserved_, headroom_);
    }
public:
    // The caller holds this destination's DirectoryLock for the complete
    // lifetime. This observational lock never replaces construction authority.
    FarmingOwner(FarmingMetadataDirectory const& metadata, FarmingState const& state, size_t lane,
        std::string control_digest, uint64_t reserved_bytes, uint64_t memory_limit, std::optional<std::string> const& approved_source = {}) :
        metadata_(metadata.root, metadata.legacy), owner_(farming_owner_detail::local_record(state, lane)),
        reserved_(reserved_bytes), headroom_(state.headroom)
    {
        using namespace farming_owner_detail;
        require(farming_owner_memory_bound(metadata.root) < memory_limit, "farming owner exceeds its retained-memory allowance");
        file_io::Space space{}; require(!file_io::space_path(metadata.root, &space), "cannot inspect farming owner metadata alignment");
        require(reserved_ >= farming_owner_reservation(space.f_frsize),
            "saved job has no farming owner metadata allowance; review plan --upgrade-job before enabling live ownership observations");
        metadata_.create(); metadata_.check();
        (void)farming_owner_allocation(metadata_);
        for (auto const& entry : std::filesystem::directory_iterator(metadata_.path))
        {
            auto const name = file_io::path_text(entry.path().filename());
            if (filename_matches(name) && name.find(".partial.") != name.npos)
            { (void)charge(entry.path()); require(!file_io::unlink_file(entry.path()), "cannot reclaim interrupted farming owner metadata"); }
        }
        auto prior = observe_farming_owner(metadata_, state, lane, memory_limit, approved_source);
        for (unsigned retry = 0; prior.lock_held && retry < 3; ++retry)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            prior = observe_farming_owner(metadata_, state, lane, memory_limit, approved_source);
        }
        require(!prior.lock_held, "farming owner is already held; preserve its active coordinator");
#ifdef _WIN32
        proof_.handle = ::CreateMutexW(nullptr, TRUE, mutex_name(owner_.instance).c_str());
        require(proof_.handle && ::GetLastError() != ERROR_ALREADY_EXISTS, "cannot create local farming owner proof");
#endif
        heartbeat_.instance = owner_.instance; heartbeat_.control = std::move(control_digest);
        owner_.descriptor = INT_MAX;
        auto const preflight = checked(text(owner_)); admit(preflight);
        auto const path = metadata_.path / filename;
        std::string temporary = file_io::path_text(path) + ".partial.XXXXXX";
        farm::inventory_detail::Handle handle(file_io::temp_file(temporary.data()));
        try
        {
#ifndef _WIN32
            require(!::fcntl(handle.fd, F_SETFD, FD_CLOEXEC), "cannot protect farming owner descriptor from child execution");
#endif
            owner_.descriptor = handle.fd; auto const contents = checked(text(owner_));
            require(contents.size() <= preflight.size(), "farming owner descriptor exceeded its preflight bound");
            require(!file_io::lock_file(handle.fd, file_io::lock_exclusive | file_io::lock_nonblocking), "cannot lock farming owner record");
            size_t written = 0;
            while (written < contents.size())
            {
                auto const count = file_io::write_file(handle.fd, contents.data() + written, contents.size() - written);
                if (count < 0 && errno == EINTR) continue;
                require(count > 0, "cannot write farming owner record"); written += size_t(count);
            }
            require(!file_io::flush(handle.fd), "cannot sync farming owner record"); metadata_.check();
            require(farming_owner_allocation(metadata_) <= reserved_, "staged farming owner exceeded its metadata grant");
            require(!file_io::rename_file(file_io::native_path(temporary), path, farm_metadata_detail::present(path)), "cannot publish farming owner record");
            owner_handle_ = std::move(handle); stream_state::sync_path(metadata_.path, true); metadata_.check();
            publish_heartbeat("running");
            thread_ = std::thread([this]
            {
                std::unique_lock lock(mutex_);
                while (!ready_.wait_for(lock, std::chrono::seconds(1), [&] { return stopping_; }))
                    try { publish_heartbeat("running"); } catch (...) { /* A failed heartbeat becomes visibly stale. */ }
            });
        }
        catch (...)
        {
            (void)file_io::unlink_file(temporary);
            throw;
        }
    }
    FarmingOwner(FarmingOwner const&) = delete;
    FarmingOwner& operator=(FarmingOwner const&) = delete;
    ~FarmingOwner()
    {
        { std::lock_guard lock(mutex_); stopping_ = true; } ready_.notify_all();
        if (thread_.joinable()) thread_.join();
        try { publish_heartbeat("stopped"); } catch (...) { /* The released inode lock still proves the owner stopped. */ }
    }
};
} // namespace group_grow
