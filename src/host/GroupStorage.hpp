// Storage limits shared by single-file growth and destination scheduling.
#pragma once
#include "host/GroupEngine.hpp"
#include "host/Utf8.hpp"
#include <cstring>
#include <condition_variable>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <set>
#include <span>
#include <utility>
#ifdef _WIN32
#include <winioctl.h>
#include <nvme.h>
#else
#include <sys/resource.h>
#endif
#ifdef __linux__
#include <linux/btrfs.h>
#include <sys/sysmacros.h>
#endif

namespace group_grow
{
inline bool ram_backed_filesystem(uint64_t type)
{
    // Match TempFile's low-word Linux filesystem-magic check without adding a
    // host-library dependency to the standalone ordinary-group tools.
    auto const magic = uint32_t(type);
    return magic == 0x01021994u || magic == 0x858458f6u || magic == 0x958458f6u;
}
inline void require_disk_scratch(std::filesystem::path const& directory)
{
#ifdef __linux__
    struct statfs filesystem{};
    require(!::statfs(directory.c_str(), &filesystem), "cannot inspect external scratch filesystem");
    require(!ram_backed_filesystem(uint64_t(filesystem.f_type)),
        "external scratch must use a disk-backed filesystem; RAM-backed scratch bypasses the memory budget and durable recovery");
#elif defined(_WIN32)
    file_io::Stat value{};
    require(!file_io::inspect_link(directory, &value) && file_io::is_directory(value.st_mode),
        "external scratch requires a local, uncompressed NTFS directory without reparse points");
#else
    (void)directory;
#endif
}
inline uint64_t filesystem_file_bound(uint64_t type, uint64_t block_bytes)
{
    // Linux ext4_max_size() leaves one block below the 32-bit extent limit.
    // This is an upper bound, not a promise about inode features, quotas or
    // free space; physical allocation admission still runs for every change.
    if (type == 0xef53)
    {
        require(block_bytes >= 1024 && block_bytes <= 65536 && std::has_single_bit(block_bytes),
                "invalid ext4 filesystem block size");
        return uint64_t(UINT32_MAX) * block_bytes;
    }
    return uint64_t(INT64_MAX);
}

inline uint64_t maximum_file_bytes(std::string const& path)
{
    uint64_t limit = uint64_t(INT64_MAX);
#ifdef __linux__
    struct statfs filesystem{};
    require(!::statfs(path.c_str(), &filesystem), "cannot inspect destination file-size limit");
    limit = filesystem_file_bound(uint64_t(filesystem.f_type), uint64_t(filesystem.f_bsize));
#endif
#ifndef _WIN32
    struct rlimit resource{};
    require(!::getrlimit(RLIMIT_FSIZE, &resource), "cannot inspect process file-size limit");
    if (resource.rlim_cur != RLIM_INFINITY) limit = std::min(limit, uint64_t(resource.rlim_cur));
#endif
    return limit;
}

struct StorageDevice
{
    std::string filesystem;
    std::vector<std::string> drives;
    // Each flag requires every underlying drive to have that known type.
    // Both remain false for mixed media or an unknown rotation classification.
    bool nonrotating = false, rotating = false;
};

// Runtime feedback, never allocation authority or a saved scratch setting.
// A permit covers one native I/O operation, so a recovered owner spanning many
// drives can always drain without changing any of its retained capacity grants.
class DiskIoGovernor
{
public:
    using Clock = std::chrono::steady_clock;
    struct Counters
    {
        uint64_t reads = 0, writes = 0, read_ms = 0, write_ms = 0;
        uint64_t inflight = 0, busy_ms = 0, queue_ms = 0;
        uint64_t read_sectors = 0, write_sectors = 0;
    };
    using Sampler = std::function<std::optional<Counters>(std::string const&)>;
    struct Status
    {
        size_t drives = 0, active = 0, limit = 1;
        bool farming_known = false, pressured = false;
        uint64_t completed_bytes = 0, waits = 0;
    };
    explicit DiskIoGovernor(Sampler sampler = sample_device) : sampler_(std::move(sampler)) { pending_.reserve(4096); }

    void register_device(StorageDevice const& device, std::optional<uint64_t> filesystem = {})
    {
        std::lock_guard lock(mutex_);
        if (!(device.nonrotating && !device.drives.empty()))
        {
            if (device.drives.empty()) register_drive(unknown_drive);
            else for (auto const& name : device.drives) register_drive(name);
        }
        if (filesystem)
        { require(filesystems_.contains(*filesystem) || filesystems_.size() < 4096, "too many disk feedback filesystems"); filesystems_.try_emplace(*filesystem, device); }
    }
    void register_directory(std::filesystem::path const& directory);
    // Fresh installed-reader challenge completion times, not RPC/process wall
    // times. The declared deadline remains the resource gate's responsibility.
    void observe_farming(uint64_t response_ns, uint64_t deadline_ns = 0, bool missed = false,
                         Clock::time_point now = Clock::now())
    {
        std::lock_guard lock(mutex_);
        farming_at_ = now; farming_response_ = response_ns; farming_deadline_ = deadline_ns;
        bool const late = missed || (deadline_ns && response_ns >= deadline_ns / 2);
        if (late) pressure(now);
        changed_.notify_all();
    }
    void observe_pressure(bool pressured, Clock::time_point now = Clock::now())
    {
        std::lock_guard lock(mutex_);
        if (pressured) pressure(now);
        changed_.notify_all();
    }
    // Sampling and selection are bounded by the registered physical leaves.
    // Inactive-drive pressure is distinguishable from our own traffic; busy
    // counters during construction alone never prove a foreign farming load.
    void poll(Clock::time_point now = Clock::now())
    { std::lock_guard lock(mutex_); poll_locked(now); }

    long double routing_cost(StorageDevice const& device, Clock::time_point now = Clock::now())
    {
        if (device.nonrotating || device.drives.empty()) return 0;
        std::lock_guard lock(mutex_); poll_locked(now);
        long double result = -1;
        for (auto const& name : device.drives) if (auto at = drives_.find(name); at != drives_.end())
        {
            auto const& drive = at->second;
            // An unused/long-deferred physical drive receives a bounded probe,
            // including a slow drive; observations are not permanent bans.
            bool const busy = now < drive.foreign_until;
            bool const probe = drive.selected ? selection_ - drive.selected >= 2 * drives_.size() : !busy;
            result = std::max(result, probe ? -1.L : (busy ? 1000000.L : 0.L) + std::max(drive.ns_per_byte, drive.device_ns_per_byte));
        }
        return result;
    }
    void selected(StorageDevice const& device)
    {
        std::lock_guard lock(mutex_); ++selection_;
        for (auto const& name : device.drives) if (auto at = drives_.find(name); at != drives_.end()) at->second.selected = selection_;
    }
    class Permit
    {
    public:
        Permit() = default;
        Permit(Permit&& other) noexcept
            : governor_(std::exchange(other.governor_, nullptr)), device_(other.device_), bytes_(other.bytes_) {}
        Permit& operator=(Permit&& other) noexcept
        {
            if (this != &other)
            {
                if (governor_) governor_->complete(*device_, 0, 0);
                governor_ = std::exchange(other.governor_, nullptr); device_ = other.device_; bytes_ = other.bytes_;
            }
            return *this;
        }
        ~Permit() { if (governor_) governor_->complete(*device_, 0, 0); }
        Permit(Permit const&) = delete;
        Permit& operator=(Permit const&) = delete;
        size_t bytes() const { return bytes_; }
        void complete(uint64_t bytes, uint64_t elapsed_ns, Clock::time_point now = Clock::now())
        { if (auto* governor = std::exchange(governor_, nullptr)) governor->complete(*device_, bytes, elapsed_ns, now); }
    private:
        friend class DiskIoGovernor;
        Permit(DiskIoGovernor* governor, StorageDevice const* device, size_t bytes)
            : governor_(governor), device_(device), bytes_(bytes) {}
        DiskIoGovernor* governor_ = nullptr;
        StorageDevice const* device_ = nullptr;
        size_t bytes_ = 0;
    };
    Permit acquire(StorageDevice const& device, size_t requested, std::function<void()> const& check_running = {})
    {
        if (device.nonrotating && !device.drives.empty()) return Permit(nullptr, &device, requested);
        register_device(device);
        std::unique_lock lock(mutex_);
        require(pending_.size() < 4096, "too many queued disk operations");
        uint64_t const ticket = ++ticket_; pending_.push_back({ticket, &device});
        try
        {
            for (;;)
            {
                if (check_running) { lock.unlock(); check_running(); lock.lock(); }
                auto const now = Clock::now(); poll_locked(now);
                if (queued_turn(ticket, now)) if (auto permit = try_acquire_locked(device, requested, now))
                {
                    erase_ticket(ticket); changed_.notify_all(); return std::move(*permit);
                }
                auto wake = now + std::chrono::milliseconds(10);
                for_drives(device, [&](Drive& drive) { if (drive.next > now) wake = std::min(wake, drive.next); });
                changed_.wait_until(lock, wake);
            }
        }
        catch (...)
        {
            if (!lock.owns_lock()) lock.lock();
            erase_ticket(ticket); changed_.notify_all(); throw;
        }
    }
    std::optional<Permit> try_acquire(StorageDevice const& device, size_t requested, Clock::time_point now = Clock::now())
    {
        if (device.nonrotating && !device.drives.empty()) return Permit(nullptr, &device, requested);
        register_device(device); std::lock_guard lock(mutex_); poll_locked(now);
        if (!pending_.empty()) return {};
        return try_acquire_locked(device, requested, now);
    }
    Permit acquire(int descriptor, size_t requested, std::function<void()> const& check_running = {})
    {
        file_io::Stat value{};
        StorageDevice const* device = &unknown_device_;
        {
            std::lock_guard lock(mutex_);
            if (!file_io::inspect_fd(descriptor, &value))
                if (auto at = filesystems_.find(uint64_t(value.st_dev)); at != filesystems_.end()) device = &at->second;
        }
        return acquire(*device, requested, check_running);
    }
    Status status(Clock::time_point now = Clock::now()) const
    {
        std::lock_guard lock(mutex_);
        return {drives_.size(), active_, limit_, farming_known(now), now < pressure_until_, completed_bytes_, waits_};
    }
    uint64_t memory_bytes() const
    {
        std::lock_guard lock(mutex_); uint64_t bytes = sizeof(*this) + 2 * pending_.capacity() * sizeof(Pending);
        for (auto const& [name, drive] : drives_) { (void)drive; bytes += sizeof(Drive) + sizeof(std::string) + 4 * sizeof(void*) + name.capacity() + 1; }
        for (auto const& [id, device] : filesystems_)
        {
            (void)id; bytes += sizeof(StorageDevice) + 5 * sizeof(void*) + device.filesystem.capacity() + 1 + device.drives.capacity() * sizeof(std::string);
            for (auto const& drive : device.drives) bytes += drive.capacity() + 1;
        }
        return bytes;
    }
    static std::optional<Counters> sample_device(std::string const& name)
    {
#ifdef __linux__
        // Only kernel-discovered major:minor leaves form sysfs paths. NVMe
        // subsystem names and unresolved/foreign identities remain unknown.
        if (name.empty() || name.find(':') == name.npos || name.find_first_not_of("0123456789:") != name.npos) return {};
        std::ifstream input(file_io::native_path("/sys/dev/block") / name / "stat");
        Counters value; uint64_t merged_reads, merged_writes;
        if (!(input >> value.reads >> merged_reads >> value.read_sectors >> value.read_ms >> value.writes >> merged_writes >>
              value.write_sectors >> value.write_ms >> value.inflight >> value.busy_ms >> value.queue_ms)) return {};
        return value;
#else
        (void)name; return {};
#endif
    }
private:
    static constexpr char const* unknown_drive = "unresolved";
    struct Drive
    {
        std::optional<Counters> counters;
        Clock::time_point sampled{}, own_at{}, next{}, foreign_until{};
        long double ns_per_byte = 0, device_ns_per_byte = 0;
        uint64_t selected = 0;
        size_t active = 0;
    };
    std::map<std::string, Drive> drives_;
    std::map<uint64_t, StorageDevice> filesystems_;
    StorageDevice unknown_device_;
    Sampler sampler_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    struct Pending { uint64_t ticket; StorageDevice const* device; };
    std::vector<Pending> pending_;
    Clock::time_point sampled_{}, window_{}, pressure_until_{}, trial_after_{}, farming_at_{};
    size_t active_ = 0, limit_ = 1;
    uint64_t completed_bytes_ = 0, window_bytes_ = 0, waits_ = 0, selection_ = 0, ticket_ = 0;
    uint64_t farming_response_ = 0, farming_deadline_ = 0;
    long double baseline_rate_ = 0;
    bool waiting_ = false, trial_ = false;
    void register_drive(std::string const& name)
    { require(drives_.contains(name) || drives_.size() < 4096, "too many disk feedback resources"); drives_.try_emplace(name); }
    template<class Operation> void for_drives(StorageDevice const& device, Operation operation)
    {
        if (device.drives.empty()) operation(drives_.at(unknown_drive));
        else for (auto const& name : device.drives) operation(drives_.at(name));
    }
    void erase_ticket(uint64_t ticket)
    { pending_.erase(std::find_if(pending_.begin(), pending_.end(), [&](Pending const& p) { return p.ticket == ticket; })); }
    bool ready(StorageDevice const& device, Clock::time_point now)
    {
        bool blocked = active_ >= limit_;
        for_drives(device, [&](Drive& drive) { blocked = blocked || drive.active || now < drive.next; });
        return !blocked;
    }
    bool queued_turn(uint64_t ticket, Clock::time_point now)
    {
        // An older eligible request wins. A waiter for one busy physical disk
        // must not leave other independent disks idle behind it.
        for (auto const& pending : pending_)
        {
            if (pending.ticket == ticket) return true;
            if (ready(*pending.device, now)) return false;
        }
        return false;
    }
    std::optional<Permit> try_acquire_locked(StorageDevice const& device, size_t requested, Clock::time_point now)
    {
        bool local = false, foreign = false;
        for_drives(device, [&](Drive& drive)
        { local = local || drive.active || now < drive.next; foreign = foreign || now < drive.foreign_until; });
        if (active_ >= limit_ || local)
        {
            ++waits_;
            if (active_ >= limit_ && !local) waiting_ = true;
            return {};
        }
        if (foreign && now >= pressure_until_) pressure(now);
        ++active_; size_t bytes = requested;
        for_drives(device, [&](Drive& drive)
        {
            ++drive.active; drive.own_at = now;
            // Aim for a short syscall, bounded even before a measured rate
            // exists. Cached timings are tuning evidence only.
            uint64_t const quantum = now < pressure_until_ ? 5000000 : 20000000;
            long double const service = std::max(drive.ns_per_byte, drive.device_ns_per_byte);
            uint64_t const ceiling = drive.device_ns_per_byte ? 4 * 1024 * 1024 : 1024 * 1024;
            uint64_t const measured = service > 0 ? uint64_t(std::clamp<long double>(
                quantum / service, 65536, ceiling)) : ceiling;
            bytes = std::min(bytes, size_t(measured));
        });
        return Permit(this, &device, bytes);
    }
    bool farming_known(Clock::time_point now) const
    { return farming_at_ != Clock::time_point{} && now >= farming_at_ && now - farming_at_ <= std::chrono::seconds(30); }
    bool farming_timely(Clock::time_point now) const
    { return farming_known(now) && farming_deadline_ && farming_response_ < farming_deadline_ / 2; }
    void pressure(Clock::time_point now)
    {
        // Multiplicative decrease and a hold avoid repeated grow/shrink cycles.
        limit_ = std::max<size_t>(1, limit_ / 2); pressure_until_ = now + std::chrono::seconds(2);
        trial_after_ = now + std::chrono::seconds(5); trial_ = false; baseline_rate_ = 0;
    }
    void poll_locked(Clock::time_point now)
    {
        if (sampled_ != Clock::time_point{} && now >= sampled_ && now - sampled_ < std::chrono::milliseconds(250)) return;
        sampled_ = now;
        if (!farming_timely(now)) limit_ = std::min<size_t>(limit_, 2);
        for (auto& [name, drive] : drives_)
        {
            auto current = sampler_ ? sampler_(name) : std::optional<Counters>{};
            if (current && drive.counters && current->busy_ms >= drive.counters->busy_ms &&
                current->read_sectors >= drive.counters->read_sectors && current->write_sectors >= drive.counters->write_sectors)
            {
                // Sector and busy counters see completed block work below the
                // page cache. Do not let fast cached writes erase that service
                // cost or create an unbounded deferred writeback queue.
                long double const sectors = static_cast<long double>(current->read_sectors - drive.counters->read_sectors) +
                    (current->write_sectors - drive.counters->write_sectors);
                if (sectors && current->busy_ms > drive.counters->busy_ms)
                {
                    long double const service = (current->busy_ms - drive.counters->busy_ms) * 1000000.L / (sectors * 512.L);
                    drive.device_ns_per_byte = drive.device_ns_per_byte ? (drive.device_ns_per_byte * 3 + service) / 4 : service;
                }
            }
            else drive.device_ns_per_byte = 0;
            if (current && drive.counters && now > drive.sampled && drive.own_at < drive.sampled &&
                !drive.active && now - drive.own_at > std::chrono::seconds(1) &&
                current->busy_ms >= drive.counters->busy_ms && current->queue_ms >= drive.counters->queue_ms)
            {
                auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - drive.sampled).count();
                if (ms > 0 && (static_cast<long double>(current->busy_ms - drive.counters->busy_ms) / ms >= .75L ||
                    static_cast<long double>(current->queue_ms - drive.counters->queue_ms) / ms >= 1.L))
                    drive.foreign_until = now + std::chrono::seconds(1);
            }
            drive.counters = current; drive.sampled = now;
        }
        if (window_ == Clock::time_point{} || now < window_) { window_ = now; window_bytes_ = completed_bytes_; return; }
        if (now - window_ < std::chrono::seconds(2)) return;
        long double const ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - window_).count();
        long double const rate = (completed_bytes_ - window_bytes_) / ns;
        if (now >= pressure_until_ && now >= trial_after_ && rate > 0)
        {
            if (trial_)
            {
                // Whole-window throughput also sees a shared HBA/controller
                // ceiling without assuming one independent lane per disk.
                if (rate < baseline_rate_ * 1.05L)
                { limit_ = std::max<size_t>(1, limit_ - 1); trial_after_ = now + std::chrono::seconds(5); }
                trial_ = false;
            }
            else if (waiting_ && limit_ < std::min(drives_.size(), farming_timely(now) ? drives_.size() : size_t(2)))
            { baseline_rate_ = rate; ++limit_; trial_ = true; }
        }
        window_ = now; window_bytes_ = completed_bytes_; waiting_ = false;
    }
    void complete(StorageDevice const& device, uint64_t bytes, uint64_t elapsed_ns, Clock::time_point now = Clock::now())
    {
        std::lock_guard lock(mutex_);
        --active_; completed_bytes_ = grow_add(completed_bytes_, bytes);
        for_drives(device, [&](Drive& drive)
        {
            --drive.active; drive.own_at = now;
            if (bytes && elapsed_ns)
            {
                long double const rate = static_cast<long double>(elapsed_ns) / bytes;
                drive.ns_per_byte = drive.ns_per_byte ? (drive.ns_per_byte * 3 + rate) / 4 : rate;
            }
            // Leave regular gaps for foreign random reads. Unknown farming
            // feedback retains the larger reserve; pressure remains finite.
            auto const divisor = now < pressure_until_ || now < drive.foreign_until ? 1 : farming_timely(now) ? 8 : 4;
            long double const physical = bytes * drive.device_ns_per_byte;
            long double const service = std::max(static_cast<long double>(elapsed_ns), physical);
            long double const deferred = std::max(0.L, physical - elapsed_ns);
            uint64_t const delay = bytes && !drive.device_ns_per_byte ?
                uint64_t(std::min(50000000.L, bytes * 20000000.L / (1024 * 1024))) :
                uint64_t(std::clamp<long double>(deferred + service / divisor, 0, 50000000));
            drive.next = now + std::chrono::nanoseconds(delay);
        });
        changed_.notify_all();
    }
};

// Paths and Linux device numbers can change across boots. Pair the persistent
// filesystem identity with the directory's native file ID when recovering a job.
inline std::string farming_directory_identity(std::filesystem::path const& path)
{
    file_io::Stat info{};
    require(!file_io::inspect_path(path.c_str(), &info) && file_io::is_directory(info.st_mode),
        "cannot identify farming destination directory");
    std::ostringstream value;
#ifdef __linux__
    struct statfs filesystem{};
    require(!::statfs(path.c_str(), &filesystem), "cannot identify farming destination filesystem");
    value << std::hex << uint64_t(filesystem.f_type) << ':';
    if (unsigned(filesystem.f_type) == 0x9123683eu)
    {
        // Btrfs statfs IDs can include the current device number. Its UUID and
        // subvolume ID retain the directory identity across mount cycles.
        int const fd = file_io::open_file(path.c_str(), file_io::read_only | file_io::directory | file_io::close_on_exec);
        require(fd >= 0, "cannot open farming destination directory");
        struct Close { int fd; ~Close() { file_io::close_file(fd); } } close{fd};
        struct btrfs_ioctl_fs_info_args btrfs{};
        struct btrfs_ioctl_get_subvol_info_args subvolume{};
        require(!::ioctl(fd, BTRFS_IOC_FS_INFO, &btrfs) && !::ioctl(fd, BTRFS_IOC_GET_SUBVOL_INFO, &subvolume),
            "cannot identify farming destination subvolume");
        for (auto byte : btrfs.fsid) value << std::setw(2) << std::setfill('0') << unsigned(byte);
        value << ':' << subvolume.treeid;
    }
    else for (auto byte : std::as_bytes(std::span(&filesystem.f_fsid, 1)))
        value << std::setw(2) << std::setfill('0') << std::to_integer<unsigned>(byte);
#elif defined(_WIN32)
    std::wstring mount(32768, L'\0');
    std::array<wchar_t, 64> name{};
    require(::GetVolumePathNameW(path.c_str(), mount.data(), DWORD(mount.size())) &&
        ::GetVolumeNameForVolumeMountPointW(mount.c_str(), name.data(), DWORD(name.size())),
        "cannot identify farming destination volume");
    value << file_io::path_text(std::filesystem::path(name.data()));
#else
    value << uint64_t(info.st_dev);
#endif
    value << ':' << uint64_t(info.st_ino);
    return value.str();
}

inline bool farming_mount_within(std::filesystem::path const& mount, std::filesystem::path const& root)
{
    auto m = mount.begin(), r = root.begin();
    for (; r != root.end(); ++r, ++m)
    {
        if (m == mount.end()) return false;
#ifdef _WIN32
        auto const a = m->native(), b = r->native();
        if (::CompareStringOrdinal(a.c_str(), int(a.size()), b.c_str(), int(b.size()), TRUE) != CSTR_EQUAL) return false;
#else
        if (*m != *r) return false;
#endif
    }
    return true;
}

struct FarmingDestinationDiscovery
{
    std::vector<std::filesystem::path> selected;
    std::vector<std::pair<std::filesystem::path, std::string>> skipped;
};

inline bool farming_path_excluded(std::filesystem::path const& path,
    std::span<std::filesystem::path const> exclusions)
{
    auto const absolute = std::filesystem::absolute(path).lexically_normal();
    auto const canonical = std::filesystem::weakly_canonical(absolute);
    for (auto const& excluded : exclusions)
        if (farming_mount_within(absolute, std::filesystem::absolute(excluded).lexically_normal()) ||
            farming_mount_within(canonical, std::filesystem::weakly_canonical(excluded))) return true;
    return false;
}

inline std::vector<std::filesystem::path> farming_mount_paths(std::istream& mounts,
    std::vector<std::pair<std::filesystem::path, std::string>>* skipped = nullptr)
{
    std::vector<std::filesystem::path> paths;
    for (std::string line; std::getline(mounts, line);)
    {
        std::istringstream row(line);
        std::string id, parent, device, root, mount, options;
        if (!(row >> id >> parent >> device >> root >> mount >> options)) continue;
        std::string decoded;
        for (size_t i = 0; i < mount.size(); ++i)
        {
            if (mount[i] == '\\' && i + 3 < mount.size() &&
                mount[i + 1] >= '0' && mount[i + 1] <= '7' && mount[i + 2] >= '0' && mount[i + 2] <= '7' &&
                mount[i + 3] >= '0' && mount[i + 3] <= '7')
            {
                decoded.push_back(char((mount[i + 1] - '0') * 64 + (mount[i + 2] - '0') * 8 + mount[i + 3] - '0'));
                i += 3;
            }
            else decoded.push_back(mount[i]);
        }
        auto path = file_io::native_path(decoded);
        if (options == "rw" || options.starts_with("rw,")) paths.push_back(std::move(path));
        else if (skipped) skipped->emplace_back(std::move(path), "read-only mount");
    }
    require(!mounts.bad(), "cannot read mounted filesystem list");
    return paths;
}

inline FarmingDestinationDiscovery farming_destination_discovery(std::filesystem::path const& under = {})
{
    namespace fs = std::filesystem;
    FarmingDestinationDiscovery result;
    auto& candidates = result.selected;
    std::optional<fs::path> scope;
    if (!under.empty())
    {
        require(fs::is_directory(under), "--discover-drives requires an existing directory: " + file_io::path_text(under));
        scope = fs::canonical(under).lexically_normal();
    }
    auto selected = [&](fs::path const& path) -> std::optional<fs::path>
    {
        std::error_code error;
        auto const canonical = fs::canonical(path, error);
        if (error)
        {
            if (!scope || farming_mount_within(path.lexically_normal(), *scope))
                result.skipped.emplace_back(path, "cannot resolve mount path: " + error.message());
            return {};
        }
        if (scope && !farming_mount_within(canonical, *scope)) return {};
        return canonical;
    };
#ifdef __linux__
    std::ifstream mounts("/proc/self/mountinfo");
    require(mounts.is_open(), "cannot open mounted filesystem list");
    std::vector<std::pair<fs::path, std::string>> skipped;
    auto const paths = farming_mount_paths(mounts, &skipped);
    for (auto const& [path, reason] : skipped)
    {
        if (auto const canonical = selected(path)) result.skipped.emplace_back(*canonical, reason);
    }
    for (auto const& path : paths)
    {
        auto const canonical = selected(path); if (!canonical) continue;
        auto skip = [&](std::string reason) { result.skipped.emplace_back(*canonical, std::move(reason)); };
        auto const decoded = file_io::path_text(*canonical);
        if (!scope && (decoded == "/" || decoded == "/home" || decoded == "/usr" || decoded == "/var" ||
            decoded == "/boot" || decoded.starts_with("/boot/") || decoded == "/efi"))
        { skip("system storage omitted from automatic suggestions"); continue; }
        struct statfs filesystem{};
        if (::statfs(path.c_str(), &filesystem)) { skip("cannot inspect mounted filesystem"); continue; }
        if (ram_backed_filesystem(uint64_t(filesystem.f_type))) { skip("RAM-backed filesystem"); continue; }
        // These are suggestions for persistent local plot storage. Explicit
        // destinations still go through the normal filesystem validation.
        if (unsigned(filesystem.f_type) != 0xef53u && unsigned(filesystem.f_type) != 0x58465342u &&
            unsigned(filesystem.f_type) != 0x9123683eu && unsigned(filesystem.f_type) != 0x5346544eu)
        { skip("unsupported filesystem for automatic discovery"); continue; }
        if (::access(path.c_str(), W_OK | X_OK)) { skip("mount directory is not writable and searchable"); continue; }
        candidates.push_back(*canonical);
    }
#elif defined(_WIN32)
    std::array<wchar_t, 32768> windows{};
    auto const length = ::GetWindowsDirectoryW(windows.data(), DWORD(windows.size()));
    wchar_t const system = length && length < windows.size() ? wchar_t(std::towupper(windows[0])) : L'C';
    std::array<wchar_t, 64> volume{};
    HANDLE const volumes = ::FindFirstVolumeW(volume.data(), DWORD(volume.size()));
    require(volumes != INVALID_HANDLE_VALUE, "cannot list mounted volumes");
    struct Close { HANDLE handle; ~Close() { ::FindVolumeClose(handle); } } close{volumes};
    do
    {
        auto const kind = ::GetDriveTypeW(volume.data());
        std::array<wchar_t, 32> filesystem{};
        DWORD flags = 0;
        bool const inspected = ::GetVolumeInformationW(volume.data(), nullptr, 0, nullptr, nullptr, &flags,
            filesystem.data(), DWORD(filesystem.size()));
        DWORD length = 0;
        (void)::GetVolumePathNamesForVolumeNameW(volume.data(), nullptr, 0, &length);
        require(length <= 1024 * 1024, "mounted volume path list exceeds its bound");
        if (!length) continue;
        std::vector<wchar_t> paths(length);
        require(::GetVolumePathNamesForVolumeNameW(volume.data(), paths.data(), length, &length),
            "cannot read mounted volume paths");
        for (auto path = paths.data(); *path; path += std::wcslen(path) + 1)
        {
            fs::path const mounted(path);
            auto const canonical = selected(mounted); if (!canonical) continue;
            std::string reason;
            if (!scope && std::towupper(path[0]) == system) reason = "system storage omitted from automatic suggestions";
            else if (kind != DRIVE_FIXED && kind != DRIVE_REMOVABLE) reason = "unsupported drive type for automatic discovery";
            else if (!inspected) reason = "cannot inspect mounted filesystem";
            else if (flags & FILE_READ_ONLY_VOLUME) reason = "read-only volume";
            else if (std::wstring_view(filesystem.data()) != L"NTFS") reason = "automatic discovery requires NTFS";
            if (!reason.empty()) result.skipped.emplace_back(*canonical, std::move(reason));
            else candidates.push_back(*canonical);
        }
    } while (::FindNextVolumeW(volumes, volume.data(), DWORD(volume.size())));
    require(::GetLastError() == ERROR_NO_MORE_FILES, "cannot finish listing mounted volumes");
#else
    require(!scope, "--discover-drives requires Linux or Windows mount discovery");
#endif
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    // Bind mounts of the same directory must not become duplicate destinations.
    std::map<std::string, fs::path> identities;
    std::erase_if(candidates, [&](auto const& path)
    {
        try
        {
            auto const [previous, inserted] = identities.emplace(farming_directory_identity(path), path);
            if (inserted) return false;
            result.skipped.emplace_back(path, "same destination as " + file_io::path_text(previous->second));
        }
        catch (std::runtime_error const& error) { result.skipped.emplace_back(path, error.what()); }
        return true;
    });
    std::sort(result.skipped.begin(), result.skipped.end());
    result.skipped.erase(std::unique(result.skipped.begin(), result.skipped.end()), result.skipped.end());
    return result;
}

inline std::vector<std::filesystem::path> farming_destination_candidates(std::filesystem::path const& under = {})
{
    auto result = farming_destination_discovery(under);
    require(under.empty() || !result.selected.empty(), "no writable supported mounted storage at or below " + file_io::path_text(under));
    return std::move(result.selected);
}

namespace storage_detail
{
inline std::string nvme_subsystem_name(std::span<uint8_t const> field)
{
    if (field.size() != 256) return {};
    auto end = std::find(field.begin(), field.end(), uint8_t(0));
    auto const length = size_t(end - field.begin());
    if (end == field.end() || !length || length > 223) return {};
    std::string_view name(reinterpret_cast<char const*>(field.data()), length);
    if (!name.starts_with("nqn.") || !pos2gpu::valid_utf8(name) ||
        std::any_of(name.begin(), name.end(), [](unsigned char c) { return c <= 0x20 || c == 0x7f; })) return {};
    // Conservative subset of NVMe Base Specification, "NVMe Qualified Names":
    // date/domain names with ASCII DNS labels (including IDNA), no ASCII
    // whitespace/control bytes, and the UUID form. Other shapes stay unknown.
    // https://nvmexpress.org/specifications/
    // Compare accepted NQNs byte-for-byte; case/Unicode normalization changes identity.
    constexpr std::string_view uuid_prefix = "nqn.2014-08.org.nvmexpress:uuid:";
    if (name.starts_with(uuid_prefix))
    {
        auto const uuid = name.substr(uuid_prefix.size());
        if (uuid.size() != 36) return {};
        for (size_t i = 0; i < uuid.size(); ++i)
            if (i == 8 || i == 13 || i == 18 || i == 23)
            { if (uuid[i] != '-') return {}; }
            else if (!((uuid[i] >= '0' && uuid[i] <= '9') || (uuid[i] >= 'a' && uuid[i] <= 'f') ||
                       (uuid[i] >= 'A' && uuid[i] <= 'F'))) return {};
        return std::string(name);
    }
    if (name.size() <= 12 || name[8] != '-' || name[11] != '.') return {};
    for (auto i : {4, 5, 6, 7, 9, 10}) if (name[i] < '0' || name[i] > '9') return {};
    unsigned const month = unsigned(name[9] - '0') * 10 + unsigned(name[10] - '0');
    if (name.substr(4, 4) == "0000" || month < 1 || month > 12) return {};
    auto const colon = name.find(':', 12);
    if (colon == name.npos || colon + 1 == name.size()) return {};
    auto const domain = name.substr(12, colon - 12);
    auto alphanumeric = [](unsigned char c)
    { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); };
    size_t labels = 0;
    for (size_t first = 0; first < domain.size();)
    {
        auto last = domain.find('.', first); if (last == domain.npos) last = domain.size();
        if (last == first || last - first > 63 || !alphanumeric(domain[first]) || !alphanumeric(domain[last - 1])) return {};
        for (size_t i = first; i < last; ++i) if (!alphanumeric(domain[i]) && domain[i] != '-') return {};
        ++labels; first = last + 1;
    }
    if (labels < 2 || domain.back() == '.') return {};
    constexpr std::string_view reserved = "org.nvmexpress";
    if (domain.size() == reserved.size() && std::equal(domain.begin(), domain.end(), reserved.begin(), [](char a, char b)
        { return a == b || (a >= 'A' && a <= 'Z' && a - 'A' + 'a' == b); })) return {};
    return std::string(name);
}
#ifdef _WIN32
inline std::string nvme_subsystem_reply(std::span<uint8_t const> encoded)
{
    if (encoded.size() < sizeof(STORAGE_PROTOCOL_DATA_DESCRIPTOR)) return {};
    STORAGE_PROTOCOL_DATA_DESCRIPTOR descriptor{}; std::memcpy(&descriptor, encoded.data(), sizeof(descriptor));
    auto const& protocol = descriptor.ProtocolSpecificData;
    constexpr size_t base = offsetof(STORAGE_PROTOCOL_DATA_DESCRIPTOR, ProtocolSpecificData);
    if (descriptor.Version != sizeof(descriptor) || descriptor.Size < sizeof(descriptor) || descriptor.Size > encoded.size() ||
        protocol.ProtocolType != ProtocolTypeNvme || protocol.DataType != NVMeDataTypeIdentify ||
        protocol.ProtocolDataRequestValue != NVME_IDENTIFY_CNS_CONTROLLER || protocol.ProtocolDataOffset < sizeof(protocol) ||
        protocol.ProtocolDataOffset > encoded.size() - base) return {};
    size_t const offset = base + protocol.ProtocolDataOffset;
    if (protocol.ProtocolDataLength < sizeof(NVME_IDENTIFY_CONTROLLER_DATA) || protocol.ProtocolDataLength > encoded.size() - offset) return {};
    NVME_IDENTIFY_CONTROLLER_DATA controller{}; std::memcpy(&controller, encoded.data() + offset, sizeof(controller));
    if (!controller.VID || !controller.NN) return {};
    return nvme_subsystem_name(controller.SUBNQN);
}
inline std::string nvme_subsystem(HANDLE disk)
{
    // Microsoft's protocol-specific Identify query returns one fixed controller
    // payload. Unsupported drivers or malformed replies leave topology unknown.
    constexpr size_t size = offsetof(STORAGE_PROPERTY_QUERY, AdditionalParameters) +
        sizeof(STORAGE_PROTOCOL_SPECIFIC_DATA) + sizeof(NVME_IDENTIFY_CONTROLLER_DATA);
    static_assert(alignof(STORAGE_PROPERTY_QUERY) <= alignof(STORAGE_PROTOCOL_DATA_DESCRIPTOR));
    static_assert(alignof(STORAGE_PROTOCOL_SPECIFIC_DATA) <= alignof(STORAGE_PROTOCOL_DATA_DESCRIPTOR));
    static_assert(offsetof(STORAGE_PROPERTY_QUERY, AdditionalParameters) % alignof(STORAGE_PROTOCOL_SPECIFIC_DATA) == 0);
    alignas(STORAGE_PROTOCOL_DATA_DESCRIPTOR) std::array<uint8_t, size> encoded{};
    auto query = reinterpret_cast<STORAGE_PROPERTY_QUERY*>(encoded.data());
    query->PropertyId = StorageAdapterProtocolSpecificProperty; query->QueryType = PropertyStandardQuery;
    auto protocol = reinterpret_cast<STORAGE_PROTOCOL_SPECIFIC_DATA*>(query->AdditionalParameters);
    protocol->ProtocolType = ProtocolTypeNvme; protocol->DataType = NVMeDataTypeIdentify;
    protocol->ProtocolDataRequestValue = NVME_IDENTIFY_CNS_CONTROLLER;
    protocol->ProtocolDataOffset = sizeof(*protocol); protocol->ProtocolDataLength = sizeof(NVME_IDENTIFY_CONTROLLER_DATA);
    DWORD returned = 0;
    if (!::DeviceIoControl(disk, IOCTL_STORAGE_QUERY_PROPERTY, encoded.data(), DWORD(encoded.size()),
        encoded.data(), DWORD(encoded.size()), &returned, nullptr) || returned > encoded.size()) return {};
    return nvme_subsystem_reply(std::span<uint8_t const>(encoded).first(returned));
}
#endif
} // namespace storage_detail

// Follow partitions, device-mapper and RAID slaves to the underlying devices.
// Btrfs uses an anonymous st_dev, so obtain its filesystem UUID and device
// links through the kernel's filesystem-info ioctl and sysfs interface.
// Windows maps a volume's extents to physical disk numbers. Volume GUIDs
// identify filesystems, never evidence that their backing disks are separate.
inline StorageDevice storage_device(std::filesystem::path const& directory)
{
    namespace fs = std::filesystem;
    file_io::Stat info{}; require(!file_io::inspect_path(directory.c_str(), &info), "cannot inspect destination device");
    StorageDevice result{std::to_string(uint64_t(info.st_dev)), {}};
#ifdef __linux__
    std::vector<fs::path> roots;
    int fd = file_io::open_file(directory.c_str(), file_io::read_only | file_io::directory | file_io::close_on_exec);
    require(fd >= 0, "cannot open destination device directory");
    struct Close { int fd; ~Close() { file_io::close_file(fd); } } close{fd};
    struct statfs filesystem{}; require(!::fstatfs(fd, &filesystem), "cannot inspect destination filesystem device");
    if (unsigned(filesystem.f_type) == 0x9123683eu)
    {
        struct btrfs_ioctl_fs_info_args btrfs{};
        if (!::ioctl(fd, BTRFS_IOC_FS_INFO, &btrfs))
        {
            std::ostringstream name; name << std::hex << std::setfill('0');
            for (size_t i = 0; i < sizeof(btrfs.fsid); ++i)
            {
                if (i == 4 || i == 6 || i == 8 || i == 10) name << '-';
                name << std::setw(2) << unsigned(btrfs.fsid[i]);
            }
            result.filesystem = "btrfs:" + name.str();
            std::error_code error;
            auto devices = fs::directory_iterator(file_io::native_path("/sys/fs/btrfs") / name.str() / "devices", error);
            if (!error) for (auto const& device : devices) roots.push_back(device.path());
        }
    }
    else roots.push_back(file_io::native_path("/sys/dev/block") /
        (std::to_string(major(info.st_dev)) + ":" + std::to_string(minor(info.st_dev))));
    std::set<std::string> visited, leaves;
    bool unknown = roots.empty(), nonrotating = true, all_rotating = true;
    std::function<void(fs::path, unsigned)> walk = [&](fs::path path, unsigned depth)
    {
        std::error_code error; path = fs::canonical(file_io::native_path(path), error);
        if (error || depth > 32) { unknown = true; return; }
        if (fs::exists(file_io::native_path(path / "partition"), error)) path = path.parent_path();
        if (error) { unknown = true; return; }
        if (!visited.insert(file_io::path_text(path)).second) return;
        auto slaves = fs::directory_iterator(file_io::native_path(path / "slaves"), error);
        if (error) { unknown = true; return; }
        bool child = false;
        for (auto const& slave : slaves) { child = true; walk(slave.path(), depth + 1); }
        if (child) return;
        std::ifstream dev(path / "dev"); std::string device;
        if (!(dev >> device)) { unknown = true; return; }
        std::ifstream rotational(path / "queue/rotational"); unsigned rotating = 1;
        if (!(rotational >> rotating)) { nonrotating = false; all_rotating = false; }
        else
        {
            if (rotating) nonrotating = false;
            else all_rotating = false;
        }
        // All namespaces/paths belonging to one NVMe subsystem share a writer.
        std::ifstream nqn(path.parent_path() / "subsysnqn"); std::string subsystem;
        if (nqn >> subsystem) device = "nvme:" + subsystem;
        leaves.insert(device);
    };
    for (auto const& root : roots) walk(root, 0);
    if (!unknown && !leaves.empty())
    {
        result.drives.assign(leaves.begin(), leaves.end());
        result.nonrotating = nonrotating;
        result.rotating = all_rotating;
    }
#elif defined(_WIN32)
    std::error_code error; auto path = fs::absolute(directory, error);
    if (error) return result;
    std::wstring mount(32768, L'\0');
    if (!::GetVolumePathNameW(path.c_str(), mount.data(), DWORD(mount.size()))) return result;
    std::array<wchar_t, 64> name{};
    if (!::GetVolumeNameForVolumeMountPointW(mount.c_str(), name.data(), DWORD(name.size()))) return result;
    std::wstring volume(name.data());
    if (volume.empty() || volume.back() != L'\\') return result;
    auto identity = fs::path(volume).u8string();
    result.filesystem = "windows:" + std::string(reinterpret_cast<char const*>(identity.data()), identity.size());
    volume.pop_back(); // Opening the volume itself requires no trailing slash.
    struct NativeFile
    {
        HANDLE value;
        ~NativeFile() { if (value != INVALID_HANDLE_VALUE) ::CloseHandle(value); }
    } source{::CreateFileW(volume.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr)};
    if (source.value == INVALID_HANDLE_VALUE) return result;
    constexpr DWORD maximum_extents = 1024;
    std::vector<uint8_t> encoded(sizeof(VOLUME_DISK_EXTENTS));
    std::set<DWORD> disks;
    for (unsigned attempt = 0; attempt < 3; ++attempt)
    {
        DWORD returned = 0;
        auto extents = reinterpret_cast<VOLUME_DISK_EXTENTS*>(encoded.data());
        bool const complete = ::DeviceIoControl(source.value, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
            nullptr, 0, encoded.data(), DWORD(encoded.size()), &returned, nullptr);
        if (!complete && ::GetLastError() != ERROR_MORE_DATA) return result;
        if (returned < sizeof(DWORD) || !extents->NumberOfDiskExtents || extents->NumberOfDiskExtents > maximum_extents) return result;
        size_t const required = offsetof(VOLUME_DISK_EXTENTS, Extents) + size_t(extents->NumberOfDiskExtents) * sizeof(DISK_EXTENT);
        if (!complete)
        {
            if (required <= encoded.size()) return result;
            encoded.resize(required); continue;
        }
        if (returned < required || returned > encoded.size()) return result;
        for (DWORD index = 0; index < extents->NumberOfDiskExtents; ++index)
        {
            auto const& extent = extents->Extents[index];
            if (extent.StartingOffset.QuadPart < 0 || extent.ExtentLength.QuadPart <= 0) return result;
            disks.insert(extent.DiskNumber);
        }
        break;
    }
    if (disks.empty()) return result;
    bool nonrotating = true, all_rotating = true;
    std::vector<std::string> drives;
    for (auto number : disks)
    {
        auto device_name = L"\\\\.\\PhysicalDrive" + std::to_wstring(number);
        NativeFile disk{::CreateFileW(device_name.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr)};
        if (disk.value == INVALID_HANDLE_VALUE) return result;
        STORAGE_PROPERTY_QUERY query{}; query.PropertyId = StorageDeviceProperty; query.QueryType = PropertyStandardQuery;
        STORAGE_DESCRIPTOR_HEADER header{}; DWORD returned = 0;
        if (!::DeviceIoControl(disk.value, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), &header, sizeof(header), &returned, nullptr) ||
            returned < sizeof(header) || header.Size < sizeof(STORAGE_DEVICE_DESCRIPTOR) || header.Size > 65536) return result;
        std::vector<uint8_t> description(header.Size);
        if (!::DeviceIoControl(disk.value, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), description.data(), DWORD(description.size()), &returned, nullptr) ||
            returned < sizeof(STORAGE_DEVICE_DESCRIPTOR) || returned > description.size()) return result;
        auto device = reinterpret_cast<STORAGE_DEVICE_DESCRIPTOR const*>(description.data());
        if (device->Size < sizeof(STORAGE_DEVICE_DESCRIPTOR) || device->Size > returned || device->Version < sizeof(STORAGE_DEVICE_DESCRIPTOR)) return result;
        // Virtual disks and storage arrays can share backing media despite
        // having distinct disk numbers. Their unresolved topology serializes.
        if (device->BusType == BusTypeUnknown || device->BusType == BusTypeVirtual || device->BusType == BusTypeFileBackedVirtual ||
            device->BusType == BusTypeSpaces || device->BusType == BusTypeRAID || device->BusType >= BusTypeMax) return result;
        if (device->BusType == BusTypeNvme)
        {
            auto subsystem = storage_detail::nvme_subsystem(disk.value);
            if (subsystem.empty()) return result;
            drives.push_back("windows:nvme:" + subsystem);
        }
        query.PropertyId = StorageDeviceSeekPenaltyProperty;
        DEVICE_SEEK_PENALTY_DESCRIPTOR seek{};
        if (!::DeviceIoControl(disk.value, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), &seek, sizeof(seek), &returned, nullptr) ||
            returned < sizeof(seek) || seek.Version < sizeof(seek) || seek.Size < sizeof(seek) || seek.IncursSeekPenalty > 1)
        { nonrotating = false; all_rotating = false; }
        else if (seek.IncursSeekPenalty) nonrotating = false;
        else all_rotating = false;
        drives.push_back("windows:disk:" + std::to_string(number));
    }
    result.drives = std::move(drives); result.nonrotating = nonrotating; result.rotating = all_rotating;
#endif
    return result;
}

inline void DiskIoGovernor::register_directory(std::filesystem::path const& directory)
{
    file_io::Stat value{};
    require(!file_io::inspect_path(directory.c_str(), &value), "cannot bind disk feedback directory");
    register_device(storage_device(directory), uint64_t(value.st_dev));
}

inline std::vector<size_t> writer_groups(std::span<StorageDevice const> devices)
{
    std::vector<size_t> groups(devices.size(), 0);
    // Unknown topology might overlap any known device. Serialize the whole
    // job in that case rather than infer physical independence from st_dev.
    if (std::any_of(devices.begin(), devices.end(), [](auto const& item) { return item.drives.empty(); })) return groups;
    for (size_t i = 0; i < groups.size(); ++i) groups[i] = i;
    for (size_t i = 0; i < devices.size(); ++i)
        for (size_t j = 0; j < i; ++j)
        {
            bool overlap = false;
            for (auto const& drive : devices[i].drives)
                if (std::find(devices[j].drives.begin(), devices[j].drives.end(), drive) != devices[j].drives.end()) overlap = true;
            if (!overlap || groups[i] == groups[j]) continue;
            size_t const from = groups[i], to = groups[j];
            for (auto& group : groups) if (group == from) group = to;
        }
    std::vector<size_t> names;
    for (auto& group : groups)
    {
        auto found = std::find(names.begin(), names.end(), group);
        if (found == names.end()) { names.push_back(group); group = names.size() - 1; }
        else group = size_t(found - names.begin());
    }
    return groups;
}
} // namespace group_grow
