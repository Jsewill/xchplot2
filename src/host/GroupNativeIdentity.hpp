// Prospective object evidence for native owners that survive mount changes.
// Capture only from already pinned, no-follow descriptors. A filesystem token
// never authorizes an object without its complete generation-bearing handle.
#pragma once
#include "host/GroupFile.hpp"
#include <array>
#include <span>
#ifdef __linux__
#include <linux/btrfs.h>
#include <linux/fs.h>
#include <linux/magic.h>
#include <sys/ioctl.h>
#include <sys/vfs.h>
#endif

namespace group_grow::file_io
{
#ifdef __linux__
// Linux UAPI include/uapi/linux/fs.h: FS_IOC_GETFSUUID is available on newer
// kernels than some supported build sysroots. Unsupported kernels fail ioctl.
struct DurableFilesystemUuid { uint8_t len; uint8_t uuid[16]; };
inline constexpr auto durable_filesystem_uuid_ioctl = _IOR(0x15, 0, DurableFilesystemUuid);
static_assert(sizeof(DurableFilesystemUuid) == 17);
#ifdef FS_IOC_GETFSUUID
static_assert(sizeof(DurableFilesystemUuid) == sizeof(struct fsuuid2));
static_assert(durable_filesystem_uuid_ioctl == FS_IOC_GETFSUUID);
#endif
#endif
struct DurableIdentity
{
    enum class Scheme : uint8_t { none, linux_handle, windows_file_id };
    Scheme scheme = Scheme::none;
    std::string filesystem, object;
    uint32_t handle_type = 0;
    bool operator==(DurableIdentity const&) const = default;
};
inline std::string identity_hex(std::span<uint8_t const> bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result; result.reserve(bytes.size() * 2);
    for (auto byte : bytes) { result += digits[byte >> 4]; result += digits[byte & 15]; }
    return result;
}
inline bool valid_durable_identity(DurableIdentity const& value)
{
    auto hex = [](std::string_view text, size_t maximum)
    { return !text.empty() && text.size() <= maximum && text.size() % 2 == 0 &&
        text.find_first_not_of("0123456789abcdef") == text.npos &&
        text.find_first_not_of('0') != text.npos && text.find_first_not_of('f') != text.npos; };
    if (value.scheme == DurableIdentity::Scheme::none)
        return value.filesystem.empty() && value.object.empty() && !value.handle_type;
    if (value.filesystem.empty() || value.filesystem.size() > 256)
        return false;
    if (value.scheme == DurableIdentity::Scheme::linux_handle)
    {
        std::string_view const filesystem = value.filesystem;
        auto uuid = [&](std::string_view prefix)
        { return filesystem.starts_with(prefix) && filesystem.size() == prefix.size() + 32 && hex(filesystem.substr(prefix.size()), 32); };
        bool const fs = uuid("btrfs:") || uuid("ext4:") || uuid("xfs:") ||
            (filesystem.starts_with("tmpfs:") && filesystem.size() == std::string_view("tmpfs:").size() + 16 && hex(filesystem.substr(6), 16));
        return fs && value.handle_type && hex(value.object, 256);
    }
    if (value.scheme != DurableIdentity::Scheme::windows_file_id || value.handle_type ||
        !hex(value.object, 32) || value.object.size() != 32 || !value.filesystem.starts_with("ntfs:")) return false;
    auto const serial = value.filesystem.find(':', 5);
    if (serial != 41 || serial + 1 == value.filesystem.size()) return false;
    std::string volume;
    for (size_t index = 5; index < serial; ++index)
    {
        char const character = value.filesystem[index];
        if (index == 13 || index == 18 || index == 23 || index == 28) { if (character != '-') return false; }
        else
        {
            if (std::string_view("0123456789abcdef").find(character) == std::string_view::npos) return false;
            volume += character;
        }
    }
    return hex(volume, 32) && value.filesystem.find_first_not_of("0123456789", serial + 1) == value.filesystem.npos;
}
inline bool has_durable_identity(DurableIdentity const& value)
{ return value.scheme != DurableIdentity::Scheme::none && valid_durable_identity(value); }

inline int capture_durable_identity(int fd, DurableIdentity* out)
{
    *out = {};
    Stat info{};
    if (inspect_fd(fd, &info)) return -1;
    if (!is_regular(info.st_mode) && !is_directory(info.st_mode)) { errno = ENOTSUP; return -1; }
    DurableIdentity result;
#ifdef __linux__
    struct statfs filesystem{};
    if (::fstatfs(fd, &filesystem)) return -1;
    if (uint32_t(filesystem.f_type) == BTRFS_SUPER_MAGIC)
    {
        struct btrfs_ioctl_fs_info_args btrfs{};
        if (::ioctl(fd, BTRFS_IOC_FS_INFO, &btrfs)) return -1;
        result.filesystem = "btrfs:" + identity_hex(btrfs.fsid);
    }
    else if (uint32_t(filesystem.f_type) == TMPFS_MAGIC)
    {
        // tmpfs has no disk UUID. Its fsid identifies this live instance;
        // surviving files cannot cross a reboot into a new tmpfs instance.
        auto const* bytes = reinterpret_cast<uint8_t const*>(&filesystem.f_fsid);
        result.filesystem = "tmpfs:" + identity_hex({bytes, sizeof(filesystem.f_fsid)});
    }
    else if (uint32_t(filesystem.f_type) == EXT4_SUPER_MAGIC || uint32_t(filesystem.f_type) == XFS_SUPER_MAGIC)
    {
        DurableFilesystemUuid uuid{};
        if (::ioctl(fd, durable_filesystem_uuid_ioctl, &uuid)) return -1;
        if (uuid.len != sizeof(uuid.uuid)) { errno = ENOTSUP; return -1; }
        result.filesystem = (uint32_t(filesystem.f_type) == EXT4_SUPER_MAGIC ? "ext4:" : "xfs:") + identity_hex(uuid.uuid);
    }
    else { errno = ENOTSUP; return -1; }
    // Linux export handles include the filesystem's object generation. Never
    // resolve this authority with open_by_handle_at; compare pinned FDs only.
    constexpr unsigned maximum_handle_bytes = 128;
    alignas(file_handle) std::array<uint8_t, sizeof(file_handle) + maximum_handle_bytes> storage{};
    auto* handle = reinterpret_cast<file_handle*>(storage.data());
    handle->handle_bytes = maximum_handle_bytes; int mount = 0;
    if (::name_to_handle_at(fd, "", handle, &mount, AT_EMPTY_PATH)) return -1;
    if (!handle->handle_bytes || handle->handle_bytes > maximum_handle_bytes || handle->handle_type <= 0) { errno = ENOTSUP; return -1; }
    result.scheme = DurableIdentity::Scheme::linux_handle;
    result.handle_type = uint32_t(handle->handle_type);
    result.object = identity_hex({handle->f_handle, handle->handle_bytes});
#elif defined(_WIN32)
    // inspect_fd above enforces local NTFS and rejects reparse points and
    // unsupported native attributes before either identity can be captured.
    FILE_ID_INFO native{};
    if (!::GetFileInformationByHandleEx(handle(fd), FileIdInfo, &native, sizeof(native))) return failed();
    std::wstring path(32768, L'\0');
    DWORD const count = ::GetFinalPathNameByHandleW(handle(fd), path.data(), DWORD(path.size()), VOLUME_NAME_GUID);
    if (!count || count >= path.size()) return failed(); path.resize(count);
    constexpr std::wstring_view prefix = L"\\\\?\\Volume{";
    auto const end = path.find(L"}\\");
    if (path.compare(0, prefix.size(), prefix) || end == path.npos) { errno = ENOTSUP; return -1; }
    std::string volume;
    for (auto character : std::wstring_view(path).substr(prefix.size(), end - prefix.size()))
    {
        if (character >= L'A' && character <= L'F') character += L'a' - L'A';
        if (!((character >= L'0' && character <= L'9') || (character >= L'a' && character <= L'f') || character == L'-'))
        { errno = ENOTSUP; return -1; }
        volume += char(character);
    }
    result.scheme = DurableIdentity::Scheme::windows_file_id;
    result.filesystem = "ntfs:" + volume + ':' + std::to_string(native.VolumeSerialNumber);
    result.object = identity_hex(native.FileId.Identifier);
#else
    errno = ENOTSUP; return -1;
#endif
    if (!has_durable_identity(result)) { errno = ENOTSUP; return -1; }
    *out = std::move(result); return 0;
}
inline bool matches_durable_identity(int fd, DurableIdentity const& saved)
{
    DurableIdentity current;
    return has_durable_identity(saved) && !capture_durable_identity(fd, &current) && current == saved;
}
} // namespace group_grow::file_io
