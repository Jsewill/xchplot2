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
#include <unistd.h>

namespace pos2gpu {

std::string TempFile::resolve_dir(std::string_view explicit_dir)
{
    if (!explicit_dir.empty()) return std::string(explicit_dir);
    if (char const* p = std::getenv("XCHPLOT2_TEMP_DIR"); p && *p) return p;
    if (char const* p = std::getenv("TMPDIR");            p && *p) return p;
    return "/tmp";
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
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

TempFile::TempFile(TempFile&& other) noexcept
    : fd_(other.fd_)
    , path_(std::move(other.path_))
    , high_water_(other.high_water_)
{
    other.fd_ = -1;
    other.high_water_ = 0;
}

TempFile& TempFile::operator=(TempFile&& other) noexcept
{
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_         = other.fd_;
        path_       = std::move(other.path_);
        high_water_ = other.high_water_;
        other.fd_ = -1;
        other.high_water_ = 0;
    }
    return *this;
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
