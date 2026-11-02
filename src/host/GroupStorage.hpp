// Storage limits shared by single-file growth and destination scheduling.
#pragma once
#include "host/GroupEngine.hpp"
#include <set>
#ifdef _WIN32
#include <winioctl.h>
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
    bool nonrotating = true, all_rotating = true, nvme = false;
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
        nvme = nvme || device->BusType == BusTypeNvme;
        query.PropertyId = StorageDeviceSeekPenaltyProperty;
        DEVICE_SEEK_PENALTY_DESCRIPTOR seek{};
        if (!::DeviceIoControl(disk.value, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), &seek, sizeof(seek), &returned, nullptr) ||
            returned < sizeof(seek) || seek.Version < sizeof(seek) || seek.Size < sizeof(seek) || seek.IncursSeekPenalty > 1)
        { nonrotating = false; all_rotating = false; }
        else if (seek.IncursSeekPenalty) nonrotating = false;
        else all_rotating = false;
        drives.push_back("windows:disk:" + std::to_string(number));
    }
    // Namespace disk numbers do not establish independent NVMe subsystems.
    // Keep their physical overlap keys and serialize all unresolved NVMe.
    if (nvme) drives.push_back("windows:nvme");
    result.drives = std::move(drives); result.nonrotating = nonrotating; result.rotating = all_rotating;
#endif
    return result;
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
