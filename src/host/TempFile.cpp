// TempFile.cpp — see header for design.

#include "host/TempFile.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/statvfs.h>  // statvfs — free_space
#include <sys/vfs.h>    // statfs / struct statfs — dir_is_ram_backed
#include <unistd.h>

namespace pos2gpu {

std::string TempFile::resolve_dir(std::string_view explicit_dir)
{
    if (!explicit_dir.empty()) return std::string(explicit_dir);
    if (char const* p = std::getenv("XCHPLOT2_TEMP_DIR"); p && *p) return p;
    if (char const* p = std::getenv("TMPDIR");            p && *p) return p;
    return "/tmp";
}

bool TempFile::dir_is_ram_backed(std::string const& dir)
{
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

TempFile::TempFile(std::string_view dir)
{
    std::string base = resolve_dir(dir);
    if (base.back() == '/') base.pop_back();
    std::string templ = base + "/xchplot2-spill-XXXXXX";
    std::string buf(templ);
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
}

TempFile::~TempFile()
{
    unmap();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

TempFile::TempFile(TempFile&& other) noexcept
    : fd_(other.fd_)
    , path_(std::move(other.path_))
    , high_water_(other.high_water_)
    , map_(other.map_)
    , map_bytes_(other.map_bytes_)
{
    other.fd_ = -1;
    other.high_water_ = 0;
    other.map_ = nullptr;
    other.map_bytes_ = 0;
}

TempFile& TempFile::operator=(TempFile&& other) noexcept
{
    if (this != &other) {
        unmap();
        if (fd_ >= 0) ::close(fd_);
        fd_         = other.fd_;
        path_       = std::move(other.path_);
        high_water_ = other.high_water_;
        map_        = other.map_;
        map_bytes_  = other.map_bytes_;
        other.fd_ = -1;
        other.high_water_ = 0;
        other.map_ = nullptr;
        other.map_bytes_ = 0;
    }
    return *this;
}

std::uint64_t TempFile::free_space(std::string const& dir)
{
    std::string const resolved = resolve_dir(dir);
    struct statvfs st {};
    if (::statvfs(resolved.c_str(), &st) != 0) return 0;   // unknown
    // f_bavail, not f_bfree: the latter counts blocks reserved for root,
    // which this process cannot have. Quoting those would let the check pass
    // on a filesystem that is already full for everyone but root.
    return std::uint64_t(st.f_bavail) * std::uint64_t(st.f_frsize);
}

void TempFile::preallocate(std::uint64_t bytes)
{
    if (bytes == 0 || fd_ < 0) return;
#if defined(__linux__)
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
    map_       = p;
    map_bytes_ = bytes;
    if (bytes > high_water_) high_water_ = bytes;
    return p;
}

void TempFile::unmap() noexcept
{
    if (map_) {
        ::munmap(map_, map_bytes_);
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
        ssize_t const n = ::pwrite(fd_, p, remaining, static_cast<off_t>(cur));
        if (n < 0) {
            if (errno == EINTR) continue;
            int const e = errno;
            throw std::runtime_error(
                "TempFile::pwrite_at(" + std::to_string(offset) + ", " +
                std::to_string(bytes) + ") failed: " + std::strerror(e));
        }
        if (n == 0) {
            throw std::runtime_error(
                "TempFile::pwrite_at: zero-byte write (disk full?)");
        }
        p         += n;
        cur       += static_cast<std::uint64_t>(n);
        remaining -= static_cast<std::size_t>(n);
    }
    std::uint64_t const end = offset + bytes;
    if (end > high_water_) high_water_ = end;
}

void TempFile::pread_at(std::uint64_t offset, void* data, std::size_t bytes)
{
    auto* p = static_cast<unsigned char*>(data);
    std::size_t remaining = bytes;
    std::uint64_t cur = offset;
    while (remaining > 0) {
        ssize_t const n = ::pread(fd_, p, remaining, static_cast<off_t>(cur));
        if (n < 0) {
            if (errno == EINTR) continue;
            int const e = errno;
            throw std::runtime_error(
                "TempFile::pread_at(" + std::to_string(offset) + ", " +
                std::to_string(bytes) + ") failed: " + std::strerror(e));
        }
        if (n == 0) {
            throw std::runtime_error(
                "TempFile::pread_at: short read at offset " +
                std::to_string(cur) + " (file size " +
                std::to_string(high_water_) + ")");
        }
        p         += n;
        cur       += static_cast<std::uint64_t>(n);
        remaining -= static_cast<std::size_t>(n);
    }
}

} // namespace pos2gpu
