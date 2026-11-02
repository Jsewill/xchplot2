// Native file operations used by ordinary-group construction and recovery.
// Linux wrappers preserve the syscall boundaries exercised by fault tests.
#pragma once
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#include <shlobj.h>
#include <winioctl.h>
#include <sys/stat.h>
#ifdef interface
#undef interface
#endif
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/uio.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/falloc.h>
#endif
#endif

namespace group_grow::file_io
{
namespace fs = std::filesystem;
inline fs::path native_path(char const* path) { return fs::path(reinterpret_cast<char8_t const*>(path)); }
inline fs::path native_path(std::string const& path) { return native_path(path.c_str()); }
inline fs::path const& native_path(fs::path const& path) { return path; }
inline std::string path_text(fs::path const& path)
{ auto const text = path.u8string(); return {reinterpret_cast<char const*>(text.data()), text.size()}; }
#ifdef _WIN32
inline fs::path native_path(wchar_t const* path) { return fs::path(path); }
using Offset = int64_t;
using Count = intptr_t;
inline constexpr size_t max_transfer = MAXDWORD;
inline constexpr int read_only = _O_RDONLY, read_write = _O_RDWR, create = _O_CREAT, exclusive = _O_EXCL;
inline constexpr int directory = 1 << 24, no_follow = 1 << 25, close_on_exec = _O_NOINHERIT;
inline constexpr int lock_exclusive = 2, lock_nonblocking = 4, keep_size = 1, punch_hole = 2;
struct Timestamp { int64_t tv_sec = 0; long tv_nsec = 0; };
struct Stat
{
    uint64_t st_dev = 0, st_ino = 0, st_nlink = 0;
    int64_t st_size = 0, st_blocks = 0, st_blksize = 0;
    unsigned st_mode = 0;
    Timestamp st_mtim, st_ctim;
};
struct Space { uint64_t f_bsize = 0, f_frsize = 0, f_blocks = 0, f_bavail = 0; };
struct IoVec { void* iov_base; size_t iov_len; };
inline bool is_regular(unsigned mode) { return mode == _S_IFREG; }
inline bool is_directory(unsigned mode) { return mode == _S_IFDIR; }
inline int failed(DWORD error = ::GetLastError())
{
    switch (error)
    {
    case ERROR_FILE_NOT_FOUND: case ERROR_PATH_NOT_FOUND: errno = ENOENT; break;
    case ERROR_FILE_EXISTS: case ERROR_ALREADY_EXISTS: errno = EEXIST; break;
    case ERROR_DISK_FULL: case ERROR_HANDLE_DISK_FULL: errno = ENOSPC; break;
    case ERROR_ACCESS_DENIED: case ERROR_PRIVILEGE_NOT_HELD: errno = EACCES; break;
    case ERROR_SHARING_VIOLATION: case ERROR_LOCK_VIOLATION: errno = EBUSY; break;
    case ERROR_NOT_SUPPORTED: case ERROR_INVALID_FUNCTION: errno = ENOTSUP; break;
    case ERROR_INVALID_PARAMETER: errno = EINVAL; break;
    case ERROR_DIR_NOT_EMPTY: errno = ENOTEMPTY; break;
    case ERROR_NOT_SAME_DEVICE: errno = EXDEV; break;
    default: errno = EIO; break;
    }
    return -1;
}
inline HANDLE handle(int fd) { return reinterpret_cast<HANDLE>(::_get_osfhandle(fd)); }
inline long page_size() { SYSTEM_INFO info{}; ::GetSystemInfo(&info); return long(info.dwPageSize); }
inline bool ntfs(HANDLE file)
{
    wchar_t name[32]{}; DWORD flags = 0;
    if (!::GetVolumeInformationByHandleW(file, nullptr, 0, nullptr, nullptr, &flags, name, DWORD(std::size(name))))
    { failed(); return false; }
    if (std::wstring_view(name) != L"NTFS" || !(flags & FILE_PERSISTENT_ACLS)) { errno = ENOTSUP; return false; }
    std::wstring final(32768, L'\0');
    DWORD const length = ::GetFinalPathNameByHandleW(file, final.data(), DWORD(final.size()), VOLUME_NAME_DOS);
    if (!length || length >= final.size()) { failed(); return false; }
    // Recovery identities use NTFS's exact volume serial and file index.
    // Remote volumes and filesystems with larger IDs have a different model.
    if (final.compare(0, 8, L"\\\\?\\UNC\\") == 0) { errno = ENOTSUP; return false; }
    return true;
}
inline Timestamp timestamp(LARGE_INTEGER value)
{
    constexpr int64_t epoch = 116444736000000000ll, ticks = 10000000;
    int64_t const time = value.QuadPart - epoch;
    auto seconds = time / ticks; auto fraction = time % ticks;
    if (fraction < 0) { --seconds; fraction += ticks; }
    return {seconds, long(fraction * 100)};
}
inline int inspect_handle(HANDLE file, Stat* out)
{
    BY_HANDLE_FILE_INFORMATION identity{}; FILE_BASIC_INFO basic{}; FILE_STANDARD_INFO allocation{};
    FILE_ID_INFO native_identity{};
    if (!::GetFileInformationByHandle(file, &identity) ||
        !::GetFileInformationByHandleEx(file, FileBasicInfo, &basic, sizeof(basic)) ||
        !::GetFileInformationByHandleEx(file, FileStandardInfo, &allocation, sizeof(allocation)) ||
        !::GetFileInformationByHandleEx(file, FileIdInfo, &native_identity, sizeof(native_identity))) return failed();
    if (!ntfs(file)) return -1;
    if (identity.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_SPARSE_FILE | FILE_ATTRIBUTE_COMPRESSED | FILE_ATTRIBUTE_ENCRYPTED))
    { errno = ENOTSUP; return -1; }
    uint64_t inode = 0; std::memcpy(&inode, native_identity.FileId.Identifier, sizeof(inode));
    for (size_t index = sizeof(inode); index < sizeof(native_identity.FileId.Identifier); ++index)
        if (native_identity.FileId.Identifier[index]) { errno = ENOTSUP; return -1; }
    DWORD sectors = 0, bytes = 0, free = 0, total = 0;
    std::wstring name(32768, L'\0');
    DWORD const count = ::GetFinalPathNameByHandleW(file, name.data(), DWORD(name.size()), VOLUME_NAME_DOS);
    if (!count || count >= name.size()) return failed(); name.resize(count);
    std::wstring volume(32768, L'\0');
    if (!::GetVolumePathNameW(name.c_str(), volume.data(), DWORD(volume.size())) ||
        !::GetDiskFreeSpaceW(volume.c_str(), &sectors, &bytes, &free, &total)) return failed();
    *out = {native_identity.VolumeSerialNumber, inode,
        identity.nNumberOfLinks, allocation.EndOfFile.QuadPart, allocation.AllocationSize.QuadPart / 512,
        int64_t(sectors) * bytes, unsigned(allocation.Directory ? _S_IFDIR : _S_IFREG),
        timestamp(basic.LastWriteTime), timestamp(basic.ChangeTime)};
    return 0;
}
inline int inspect_fd(int fd, Stat* out) { return inspect_handle(handle(fd), out); }
template <typename Path> inline int inspect_link(Path const& path, Stat* out)
{
    HANDLE file = ::CreateFileW(native_path(path).c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (file == INVALID_HANDLE_VALUE) return failed();
    int const result = inspect_handle(file, out); int const saved = errno; ::CloseHandle(file); errno = saved; return result;
}
template <typename Path> inline int inspect_path(Path const& path, Stat* out) { return inspect_link(path, out); }
inline int space_handle(HANDLE file, Space* out)
{
    if (!ntfs(file)) return -1;
    std::wstring path(32768, L'\0'); DWORD const count = ::GetFinalPathNameByHandleW(file, path.data(), DWORD(path.size()), VOLUME_NAME_DOS);
    if (!count || count >= path.size()) return failed(); path.resize(count);
    std::wstring volume(32768, L'\0'); DWORD sectors = 0, bytes = 0, available = 0, total = 0;
    ULARGE_INTEGER user_free{}, all{}, free{};
    if (!::GetVolumePathNameW(path.c_str(), volume.data(), DWORD(volume.size())) ||
        !::GetDiskFreeSpaceW(volume.c_str(), &sectors, &bytes, &available, &total) ||
        !::GetDiskFreeSpaceExW(volume.c_str(), &user_free, &all, &free)) return failed();
    uint64_t const unit = uint64_t(sectors) * bytes;
    if (!unit) { errno = EINVAL; return -1; }
    *out = {unit, unit, all.QuadPart / unit, user_free.QuadPart / unit}; return 0;
}
inline int space_fd(int fd, Space* out) { return space_handle(handle(fd), out); }
template <typename Path> inline int space_path(Path const& path, Space* out)
{
    HANDLE file = ::CreateFileW(native_path(path).c_str(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (file == INVALID_HANDLE_VALUE) return failed();
    int const result = space_handle(file, out); int const saved = errno; ::CloseHandle(file); errno = saved; return result;
}
template <typename Path> inline int open_file(Path const& path, int flags, unsigned = 0600)
{
    auto const name = native_path(path); HANDLE file = INVALID_HANDLE_VALUE;
    bool const dir = flags & directory;
    DWORD const access = dir ? FILE_READ_ATTRIBUTES : GENERIC_READ | ((flags & read_write) ? GENERIC_WRITE : 0);
    DWORD const attributes = FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_WRITE_THROUGH | (dir ? FILE_FLAG_BACKUP_SEMANTICS : FILE_FLAG_OVERLAPPED);
    try
    {
        if (flags & create)
        {
            // Inspect the existing parent before creating a destination.
            Stat parent{}; if (inspect_link(name.parent_path().empty() ? fs::path(L".") : name.parent_path(), &parent)) return -1;
            file = pos2gpu::create_private_file_handle(name, access, flags & exclusive ? CREATE_NEW : OPEN_ALWAYS, attributes);
        }
        else file = ::CreateFileW(name.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, attributes, nullptr);
    }
    catch (std::system_error const& error) { return failed(DWORD(error.code().value())); }
    catch (std::exception const&) { errno = EACCES; return -1; }
    if (file == INVALID_HANDLE_VALUE) return failed();
    Stat value{};
    if (inspect_handle(file, &value) || is_directory(value.st_mode) != dir)
    { int const saved = errno; ::CloseHandle(file); errno = saved ? saved : EINVAL; return -1; }
    int const fd = ::_open_osfhandle(reinterpret_cast<intptr_t>(file), _O_BINARY | _O_NOINHERIT | (flags & read_write ? _O_RDWR : _O_RDONLY));
    if (fd < 0) ::CloseHandle(file); return fd;
}
inline std::mutex locks_mutex;
inline std::unordered_map<int, HANDLE> directory_locks;
inline int close_file(int fd)
{
    { std::lock_guard lock(locks_mutex); auto found = directory_locks.find(fd);
      if (found != directory_locks.end()) { ::CloseHandle(found->second); directory_locks.erase(found); } }
    return ::_close(fd);
}
inline int lock_file(int fd, int)
{
    Stat value{}; if (inspect_fd(fd, &value)) return -1;
    HANDLE file = handle(fd); bool const dir = is_directory(value.st_mode);
    if (dir)
    {
        PWSTR base = nullptr;
        if (FAILED(::SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &base))) { errno = EACCES; return -1; }
        fs::path path = fs::path(base) / (L"xchplot2-group-" + std::to_wstring(value.st_dev) + L"-" + std::to_wstring(value.st_ino) + L".lock");
        ::CoTaskMemFree(base);
        try { file = pos2gpu::create_private_file_handle(path, GENERIC_READ | GENERIC_WRITE, OPEN_ALWAYS, 0, FILE_SHARE_READ | FILE_SHARE_WRITE); }
        catch (std::system_error const& error) { return failed(DWORD(error.code().value())); }
        catch (...) { errno = EACCES; return -1; }
    }
    // Existing upstream readers never touch this sentinel beyond the allowed
    // signed file range. Windows byte-range locks are not advisory locks.
    OVERLAPPED range{}; range.Offset = UINT32_MAX; range.OffsetHigh = INT32_MAX;
    if (!::LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &range))
    { DWORD const error = ::GetLastError(); if (dir) ::CloseHandle(file); return failed(error); }
    if (dir) { std::lock_guard lock(locks_mutex); directory_locks.emplace(fd, file); }
    return 0;
}
inline Count positioned(int fd, void* data, size_t bytes, Offset offset, bool writing)
{
    if (offset < 0) { errno = EINVAL; return -1; }
    OVERLAPPED operation{}; operation.Offset = DWORD(uint64_t(offset)); operation.OffsetHigh = DWORD(uint64_t(offset) >> 32);
    operation.hEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!operation.hEvent) return failed();
    DWORD transferred = 0; DWORD const count = DWORD(std::min(bytes, max_transfer));
    BOOL success = writing ? ::WriteFile(handle(fd), data, count, &transferred, &operation) : ::ReadFile(handle(fd), data, count, &transferred, &operation);
    DWORD error = success ? ERROR_SUCCESS : ::GetLastError();
    if (!success && error == ERROR_IO_PENDING)
    { success = ::GetOverlappedResult(handle(fd), &operation, &transferred, TRUE); error = success ? ERROR_SUCCESS : ::GetLastError(); }
    ::CloseHandle(operation.hEvent);
    if (!success) { if (!writing && error == ERROR_HANDLE_EOF) return 0; return failed(error); }
    return Count(transferred);
}
inline Count read_at(int fd, void* data, size_t bytes, Offset offset) { return positioned(fd, data, bytes, offset, false); }
inline Count write_at(int fd, void const* data, size_t bytes, Offset offset) { return positioned(fd, const_cast<void*>(data), bytes, offset, true); }
inline Count writev_at(int fd, IoVec const* vectors, int count, Offset offset)
{
    Count total = 0;
    for (int index = 0; index < count; ++index)
    { auto const written = write_at(fd, vectors[index].iov_base, vectors[index].iov_len, offset + total);
      if (written < 0) return total ? total : -1; total += written; if (size_t(written) != vectors[index].iov_len) break; }
    return total;
}
inline Count write_file(int fd, void const* data, size_t bytes)
{
    auto const offset = ::_lseeki64(fd, 0, SEEK_CUR); if (offset < 0) return -1;
    auto const written = write_at(fd, data, bytes, offset);
    if (written > 0 && ::_lseeki64(fd, offset + written, SEEK_SET) < 0) return -1;
    return written;
}
inline int resize_file(int fd, Offset bytes)
{
    if (bytes < 0) { errno = EINVAL; return -1; }
    FILE_STANDARD_INFO current{};
    if (!::GetFileInformationByHandleEx(handle(fd), FileStandardInfo, &current, sizeof(current))) return failed();
    FILE_END_OF_FILE_INFO value{}; value.EndOfFile.QuadPart = bytes;
    if (!::SetFileInformationByHandle(handle(fd), FileEndOfFileInfo, &value, sizeof(value))) return failed();
    // NTFS otherwise retains excess allocation until close, invalidating the
    // controller's saved file guards. Growing still keeps preflight reserves.
    if (bytes <= current.EndOfFile.QuadPart)
    {
        FILE_ALLOCATION_INFO allocation{}; allocation.AllocationSize.QuadPart = bytes;
        if (!::SetFileInformationByHandle(handle(fd), FileAllocationInfo, &allocation, sizeof(allocation))) return failed();
    }
    return 0;
}
inline int allocate_file(int fd, int mode, Offset offset, Offset bytes)
{
    if (offset < 0 || bytes < 0 || bytes > INT64_MAX - offset) { errno = EINVAL; return -1; }
    Stat current{}; if (inspect_fd(fd, &current)) return -1;
    int64_t const end = offset + bytes;
    int64_t allocation = end;
    if (mode & punch_hole)
    {
        if (mode != (punch_hole | keep_size) || offset < current.st_size) { errno = ENOTSUP; return -1; }
        allocation = offset; // release only the optional allocation past EOF
    }
    else
    {
        if (mode != 0 && mode != keep_size) { errno = ENOTSUP; return -1; }
        allocation = std::max(allocation, current.st_blocks * 512);
    }
    FILE_ALLOCATION_INFO value{}; value.AllocationSize.QuadPart = std::max(allocation, current.st_size);
    if (!::SetFileInformationByHandle(handle(fd), FileAllocationInfo, &value, sizeof(value))) return failed();
    if (!mode && current.st_size < end) return resize_file(fd, end);
    return 0;
}
inline int flush(int fd)
{
    Stat value{}; if (inspect_fd(fd, &value)) return -1;
    // Windows namespace operations below are write-through requests. There
    // is no POSIX directory-fsync emulation or ignored FlushFileBuffers error.
    if (is_directory(value.st_mode)) return 0;
    return ::FlushFileBuffers(handle(fd)) ? 0 : failed();
}
template <typename Path> inline int rename_file(Path const& from, Path const& to, bool replace = true)
{ return ::MoveFileExW(native_path(from).c_str(), native_path(to).c_str(), MOVEFILE_WRITE_THROUGH | (replace ? MOVEFILE_REPLACE_EXISTING : 0)) ? 0 : failed(); }
template <typename Path> inline int remove_path(Path const& path, bool dir)
{
    HANDLE file = ::CreateFileW(native_path(path).c_str(), DELETE | FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_WRITE_THROUGH | (dir ? FILE_FLAG_BACKUP_SEMANTICS : 0), nullptr);
    if (file == INVALID_HANDLE_VALUE) return failed();
    Stat value{}; if (inspect_handle(file, &value) || is_directory(value.st_mode) != dir)
    { int const saved = errno; ::CloseHandle(file); errno = saved ? saved : EINVAL; return -1; }
    struct Disposition { DWORD flags; } disposition{1u | 2u};
    // FileDispositionInfoEx and POSIX deletion are required, with no weaker
    // DeleteFile fallback. Closing this handle removes the namespace link.
    BOOL const deleted = ::SetFileInformationByHandle(file, FILE_INFO_BY_HANDLE_CLASS(21), &disposition, sizeof(disposition));
    DWORD const error = deleted ? ERROR_SUCCESS : ::GetLastError();
    BOOL const closed = ::CloseHandle(file); if (!deleted) return failed(error); return closed ? 0 : failed();
}
template <typename Path> inline int unlink_file(Path const& path) { return remove_path(path, false); }
template <typename Path> inline int remove_directory(Path const& path) { return remove_path(path, true); }
template <typename Path> inline int make_directory(Path const& path, unsigned = 0700)
{
    auto const name = native_path(path); Stat parent{};
    if (inspect_link(name.parent_path().empty() ? fs::path(L".") : name.parent_path(), &parent)) return -1;
    try { pos2gpu::create_private_directory(name); return 0; }
    catch (std::system_error const& error) { return failed(DWORD(error.code().value())); }
    catch (...) { errno = EACCES; return -1; }
}
inline int temp_file(char* pattern)
{
    std::string path(pattern); if (path.size() < 6 || path.substr(path.size() - 6) != "XXXXXX") { errno = EINVAL; return -1; }
    std::random_device random;
    for (unsigned attempt = 0; attempt < 32; ++attempt)
    {
        for (size_t index = path.size() - 6; index < path.size(); ++index) path[index] = "0123456789abcdef"[random() % 16];
        int const fd = open_file(path, create | exclusive | read_write | close_on_exec | no_follow);
        if (fd >= 0) { std::memcpy(pattern, path.c_str(), path.size() + 1); return fd; }
        if (errno != EEXIST) return -1;
    }
    errno = EEXIST; return -1;
}
#else
using Stat = struct stat;
using Space = struct statvfs;
using IoVec = struct iovec;
using Offset = off_t;
using Count = ssize_t;
inline constexpr size_t max_transfer = SSIZE_MAX;
inline constexpr int read_only = O_RDONLY, read_write = O_RDWR, create = O_CREAT, exclusive = O_EXCL;
inline constexpr int directory = O_DIRECTORY, no_follow = O_NOFOLLOW, close_on_exec = O_CLOEXEC;
inline constexpr int lock_exclusive = LOCK_EX, lock_nonblocking = LOCK_NB;
#ifdef __linux__
inline constexpr int keep_size = FALLOC_FL_KEEP_SIZE, punch_hole = FALLOC_FL_PUNCH_HOLE;
#endif
inline bool is_regular(unsigned mode) { return S_ISREG(mode); }
inline bool is_directory(unsigned mode) { return S_ISDIR(mode); }
inline long page_size() { return ::sysconf(_SC_PAGESIZE); }
template <typename Path> inline int open_file(Path const& path, int flags, unsigned mode = 0600) { return ::open(native_path(path).c_str(), flags, mode); }
inline int close_file(int fd) { return ::close(fd); }
inline int lock_file(int fd, int flags) { return ::flock(fd, flags); }
template <typename Path> inline int inspect_path(Path const& path, Stat* out) { return ::stat(native_path(path).c_str(), out); }
template <typename Path> inline int inspect_link(Path const& path, Stat* out) { return ::lstat(native_path(path).c_str(), out); }
inline int inspect_fd(int fd, Stat* out) { return ::fstat(fd, out); }
template <typename Path> inline int space_path(Path const& path, Space* out) { return ::statvfs(native_path(path).c_str(), out); }
inline int space_fd(int fd, Space* out) { return ::fstatvfs(fd, out); }
inline Count read_at(int fd, void* data, size_t bytes, Offset offset) { return ::pread(fd, data, bytes, offset); }
inline Count write_at(int fd, void const* data, size_t bytes, Offset offset) { return ::pwrite(fd, data, bytes, offset); }
inline Count writev_at(int fd, IoVec const* vectors, int count, Offset offset) { return ::pwritev(fd, vectors, count, offset); }
inline Count write_file(int fd, void const* data, size_t bytes) { return ::write(fd, data, bytes); }
inline int resize_file(int fd, Offset bytes) { return ::ftruncate(fd, bytes); }
inline int flush(int fd) { return ::fsync(fd); }
#ifdef __linux__
inline int allocate_file(int fd, int mode, Offset offset, Offset bytes) { return ::fallocate(fd, mode, offset, bytes); }
#endif
template <typename Path> inline int rename_file(Path const& from, Path const& to, bool replace = true)
{
    if (replace) return ::rename(native_path(from).c_str(), native_path(to).c_str());
#ifdef __linux__
    return ::renameat2(AT_FDCWD, native_path(from).c_str(), AT_FDCWD, native_path(to).c_str(), RENAME_NOREPLACE);
#else
    errno = ENOTSUP; return -1;
#endif
}
template <typename Path> inline int unlink_file(Path const& path) { return ::unlink(native_path(path).c_str()); }
template <typename Path> inline int remove_directory(Path const& path) { return ::rmdir(native_path(path).c_str()); }
template <typename Path> inline int make_directory(Path const& path, unsigned mode = 0700) { return ::mkdir(native_path(path).c_str(), mode); }
inline int temp_file(char* pattern) { return ::mkstemp(pattern); }
#endif
} // namespace group_grow::file_io
