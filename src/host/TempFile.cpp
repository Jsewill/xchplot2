// TempFile.cpp — see header for design.

#include "host/TempFile.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#else
#include <sys/mman.h>
#include <sys/statvfs.h>  // statvfs — free_space
#include <sys/vfs.h>    // statfs / struct statfs — dir_is_ram_backed
#include <unistd.h>
#endif

namespace pos2gpu {

#ifdef _WIN32
namespace {

// Each operation owns an event and offset. Sharing a seek position would race
// when SpillEngine writes disjoint ranges from several worker threads.
DWORD transfer_at(int fd, std::uint64_t offset, void* data, std::size_t bytes, bool write)
{
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(offset);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32);
    operation.hEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!operation.hEvent)
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "TempFile event");
    struct Cleanup {
        HANDLE event;
        ~Cleanup() { ::CloseHandle(event); }
    } cleanup{operation.hEvent};
    HANDLE const file = reinterpret_cast<HANDLE>(::_get_osfhandle(fd));
    DWORD transferred = 0;
    DWORD const count = static_cast<DWORD>(std::min<std::size_t>(bytes, MAXDWORD));
    BOOL ok = write ? ::WriteFile(file, data, count, &transferred, &operation)
                    : ::ReadFile(file, data, count, &transferred, &operation);
    if (!ok && ::GetLastError() == ERROR_IO_PENDING)
        ok = ::GetOverlappedResult(file, &operation, &transferred, TRUE);
    if (!ok) {
        DWORD const error = ::GetLastError();
        if (!write && error == ERROR_HANDLE_EOF) return 0;
        throw std::system_error(static_cast<int>(error), std::system_category(),
            write ? "TempFile::pwrite_at" : "TempFile::pread_at");
    }
    return transferred;
}

} // namespace
#endif

std::string TempFile::resolve_dir(std::string_view explicit_dir)
{
    if (!explicit_dir.empty()) return std::string(explicit_dir);
    if (char const* p = std::getenv("XCHPLOT2_TEMP_DIR"); p && *p) return p;
    if (char const* p = std::getenv("TMPDIR");            p && *p) return p;
#ifdef _WIN32
    return std::filesystem::temp_directory_path().string();
#else
    return "/tmp";
#endif
}

bool TempFile::dir_is_ram_backed(std::string const& dir)
{
#ifdef _WIN32
    std::error_code error;
    auto const path = std::filesystem::absolute(std::filesystem::path(resolve_dir(dir)), error);
    if (error) return false;
    wchar_t volume[MAX_PATH]{};
    if (!::GetVolumePathNameW(path.c_str(), volume, MAX_PATH)) return false;
    return ::GetDriveTypeW(volume) == DRIVE_RAMDISK;
#else
    std::string const resolved = resolve_dir(dir);
    struct statfs st {};
    if (::statfs(resolved.c_str(), &st) != 0) {
        return false;  // can't probe — do not block spilling on an unknown fs
    }
    // RAM-backed filesystem magics (linux/magic.h), hardcoded so the check
    // has no dependency on that header across toolchains. Compare the low 32
    // bits: f_type's width and signedness vary by platform, but every magic
    // is a 32-bit constant, so a truncating cast matches without sign-
    // extension surprises.
    unsigned const     fsmagic         = static_cast<unsigned>(st.f_type);
    constexpr unsigned kTmpfsMagic     = 0x01021994u;
    constexpr unsigned kRamfsMagic     = 0x858458f6u;
    constexpr unsigned kHugetlbfsMagic = 0x958458f6u;
    return fsmagic == kTmpfsMagic
        || fsmagic == kRamfsMagic
        || fsmagic == kHugetlbfsMagic;
#endif
}

std::string TempFile::dir_problem(std::string const& dir)
{
    std::string const resolved = resolve_dir(dir);
    try {
        // Construct-and-destroy is the probe: it exercises exactly the
        // mkstemp the spill will do later, so a read-only mount, a full
        // filesystem or an ACL that stat/access would wave through is caught
        // here instead of mid-plot. The file is unlinked at construction and
        // closed by the destructor, so nothing is left behind.
        TempFile probe(resolved);
        return {};
    } catch (std::exception const& e) {
        return e.what();
    }
}

void TempFile::bump_high_water(std::uint64_t end) noexcept
{
    // Relaxed is enough: nothing is published through this value — it is a
    // diagnostic ceiling, not a handshake. The loop is what matters, because
    // several workers can be raising it at once.
    std::uint64_t cur = high_water_.load(std::memory_order_relaxed);
    while (end > cur &&
           !high_water_.compare_exchange_weak(cur, end,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
        // cur was reloaded by compare_exchange_weak; retry unless someone
        // else already pushed the mark past `end`.
    }
}

TempFile::TempFile(std::string_view dir)
{
    std::string base = resolve_dir(dir);
    if (base.back() == '/') base.pop_back();
    std::string templ = base + "/xchplot2-spill-XXXXXX";
    std::string buf(templ);
#ifdef _WIN32
    fd_ = create_private_temp(buf, FILE_FLAG_OVERLAPPED | FILE_FLAG_DELETE_ON_CLOSE);
    path_ = std::move(buf);
#else
    fd_ = ::mkstemp(buf.data());
    if (fd_ < 0) {
        int const e = errno;
        throw std::runtime_error(
            "TempFile: mkstemp(" + templ + ") failed: " + std::strerror(e));
    }
    path_ = buf;
    // Unlink immediately so the file disappears on crash.
    if (::unlink(path_.c_str()) != 0) {
        int const e = errno;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error(
            "TempFile: unlink(" + path_ + ") failed: " + std::strerror(e));
    }
#endif
}

TempFile::~TempFile()
{
    unmap();
    if (fd_ >= 0) {
#ifdef _WIN32
        ::_close(fd_);
#else
        ::close(fd_);
#endif
        fd_ = -1;
    }
}

TempFile::TempFile(TempFile&& other) noexcept
    : fd_(other.fd_)
    , path_(std::move(other.path_))
    , high_water_(other.high_water_.load(std::memory_order_relaxed))
    , map_(other.map_)
    , map_bytes_(other.map_bytes_)
{
    other.fd_ = -1;
    other.high_water_.store(0, std::memory_order_relaxed);
    other.map_ = nullptr;
    other.map_bytes_ = 0;
}

TempFile& TempFile::operator=(TempFile&& other) noexcept
{
    if (this != &other) {
        unmap();
        if (fd_ >= 0) {
#ifdef _WIN32
            ::_close(fd_);
#else
            ::close(fd_);
#endif
        }
        fd_         = other.fd_;
        path_       = std::move(other.path_);
        high_water_.store(other.high_water_.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
        map_        = other.map_;
        map_bytes_  = other.map_bytes_;
        other.fd_ = -1;
        other.high_water_.store(0, std::memory_order_relaxed);
        other.map_ = nullptr;
        other.map_bytes_ = 0;
    }
    return *this;
}

std::uint64_t TempFile::free_space(std::string const& dir)
{
    std::string const resolved = resolve_dir(dir);
#ifdef _WIN32
    ULARGE_INTEGER available{};
    if (!::GetDiskFreeSpaceExW(std::filesystem::path(resolved).c_str(), &available, nullptr, nullptr)) return 0;
    return available.QuadPart;
#else
    struct statvfs st {};
    if (::statvfs(resolved.c_str(), &st) != 0) return 0;   // unknown
    // f_bavail, not f_bfree: the latter counts blocks reserved for root,
    // which this process cannot have. Quoting those would let the check pass
    // on a filesystem that is already full for everyone but root.
    return std::uint64_t(st.f_bavail) * std::uint64_t(st.f_frsize);
#endif
}

void TempFile::preallocate(std::uint64_t bytes)
{
    if (bytes == 0 || fd_ < 0) return;
#if defined(_WIN32)
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max()))
        throw std::runtime_error("TempFile::preallocate: size exceeds the file offset range");
    HANDLE const file = reinterpret_cast<HANDLE>(::_get_osfhandle(fd_));
    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = static_cast<LONGLONG>(bytes);
    FILE_END_OF_FILE_INFO end{};
    end.EndOfFile.QuadPart = static_cast<LONGLONG>(bytes);
    if (!::SetFileInformationByHandle(file, FileAllocationInfo, &allocation, sizeof(allocation))
        || !::SetFileInformationByHandle(file, FileEndOfFileInfo, &end, sizeof(end)))
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                               "TempFile::preallocate(" + std::to_string(bytes) + "): " + path_);
#elif defined(__linux__)
    if (::fallocate(fd_, 0, 0, static_cast<off_t>(bytes)) == 0) return;
    int const e = errno;
    // Not every filesystem implements it (network mounts, some FUSE, older
    // kernels). That is not an error — the file just grows on demand as it
    // always did, so degrade quietly rather than refusing to spill.
    if (e == EOPNOTSUPP || e == ENOSYS || e == EINVAL) return;
    throw std::runtime_error(
        "TempFile::preallocate(" + std::to_string(bytes) + ") failed on " +
        path_ + ": " + std::strerror(e) +
        (e == ENOSPC
            ? ". The temp dir cannot hold this spill table — point --temp-dir "
              "(or XCHPLOT2_TEMP_DIR) at a filesystem with more free space."
            : ""));
#else
    (void)bytes;   // no portable non-zeroing preallocation; grow on demand
#endif
}

void* TempFile::map(std::size_t bytes)
{
    if (map_) {
        throw std::runtime_error(
            "TempFile::map: already mapped (one mapping per TempFile)");
    }
    if (bytes == 0) return nullptr;
    // Reserve the blocks before mapping. ftruncate alone gives a SPARSE file,
    // and writing to a mapped page that the filesystem then cannot back
    // raises SIGBUS — a bare crash, mid-plot, with nothing in the log to say
    // the disk filled up. Reserving up front turns that into a plain ENOSPC
    // error here, before the mapping exists. Quietly does nothing where
    // fallocate is unsupported, which is the old (sparse) behaviour.
    preallocate(bytes);
#ifdef _WIN32
    HANDLE const file = reinterpret_cast<HANDLE>(::_get_osfhandle(fd_));
    HANDLE const mapping = ::CreateFileMappingW(file, nullptr, PAGE_READWRITE,
        static_cast<DWORD>(std::uint64_t(bytes) >> 32), static_cast<DWORD>(bytes), nullptr);
    if (!mapping)
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "TempFile::map");
    void* p = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
    DWORD const error = ::GetLastError();
    ::CloseHandle(mapping); // the view retains the mapping until unmap()
    if (!p) throw std::system_error(static_cast<int>(error), std::system_category(), "TempFile::map");
#else
    // Size the file so the whole mapping is backed — touching a mapped
    // page past EOF would raise SIGBUS otherwise.
    if (::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
        int const e = errno;
        throw std::runtime_error(
            "TempFile::ftruncate(" + std::to_string(bytes) + ") failed: " +
            std::strerror(e));
    }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
        int const e = errno;
        throw std::runtime_error(
            "TempFile::mmap(" + std::to_string(bytes) + ") failed: " +
            std::strerror(e));
    }
#endif
    map_       = p;
    map_bytes_ = bytes;
    bump_high_water(bytes);
    return p;
}

void TempFile::unmap() noexcept
{
    if (map_) {
#ifdef _WIN32
        ::UnmapViewOfFile(map_);
#else
        ::munmap(map_, map_bytes_);
#endif
        map_       = nullptr;
        map_bytes_ = 0;
    }
}

void TempFile::pwrite_at(std::uint64_t offset, void const* data, std::size_t bytes)
{
    auto const* p = static_cast<unsigned char const*>(data);
    std::size_t remaining = bytes;
    std::uint64_t cur = offset;
    while (remaining > 0) {
#ifdef _WIN32
        auto const n = transfer_at(fd_, cur, const_cast<unsigned char*>(p), remaining, true);
#else
        ssize_t const n = ::pwrite(fd_, p, remaining, static_cast<off_t>(cur));
        if (n < 0) {
            if (errno == EINTR) continue;
            int const e = errno;
            throw std::runtime_error(
                "TempFile::pwrite_at(" + std::to_string(offset) + ", " +
                std::to_string(bytes) + ") failed: " + std::strerror(e));
        }
#endif
        if (n == 0) {
            throw std::runtime_error(
                "TempFile::pwrite_at: zero-byte write (disk full?)");
        }
        p         += n;
        cur       += static_cast<std::uint64_t>(n);
        remaining -= static_cast<std::size_t>(n);
    }
    bump_high_water(offset + bytes);
}

void TempFile::pread_at(std::uint64_t offset, void* data, std::size_t bytes)
{
    auto* p = static_cast<unsigned char*>(data);
    std::size_t remaining = bytes;
    std::uint64_t cur = offset;
    while (remaining > 0) {
#ifdef _WIN32
        auto const n = transfer_at(fd_, cur, p, remaining, false);
#else
        ssize_t const n = ::pread(fd_, p, remaining, static_cast<off_t>(cur));
        if (n < 0) {
            if (errno == EINTR) continue;
            int const e = errno;
            throw std::runtime_error(
                "TempFile::pread_at(" + std::to_string(offset) + ", " +
                std::to_string(bytes) + ") failed: " + std::strerror(e));
        }
#endif
        if (n == 0) {
            throw std::runtime_error(
                "TempFile::pread_at: short read at offset " +
                std::to_string(cur) + " (file size " +
                std::to_string(high_water_.load(std::memory_order_relaxed)) + ")");
        }
        p         += n;
        cur       += static_cast<std::uint64_t>(n);
        remaining -= static_cast<std::size_t>(n);
    }
}

} // namespace pos2gpu
