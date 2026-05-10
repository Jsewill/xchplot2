// TempFile.hpp — POSIX-anonymous temp file with positional read/write.
//
// Task #26 disk-fallback foundation. Self-contained primitive: opens a
// unique-named file at construction (mkstemp), unlinks it immediately
// so it disappears on process exit even on crash, and supports
// thread-safe positional I/O via pread/pwrite.
//
// Path resolution order (when caller passes empty `dir`):
//   1. $XCHPLOT2_TEMP_DIR
//   2. $TMPDIR
//   3. /tmp
//
// The file is automatically removed when the TempFile destructor runs.
// On crash the kernel reclaims the inode at process exit because the
// directory entry is already unlinked at construction.
//
// Not yet wired into any tier — future low-VRAM streaming work will
// use this to spill cap-sized intermediate buffers (e.g. d_t1_meta in
// the source-tile gather rewrite) when host pinned can't grow further.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace pos2gpu {

class TempFile {
public:
    // Open a fresh anonymous temp file. `dir` overrides the env-based
    // resolution; pass empty to use $XCHPLOT2_TEMP_DIR / $TMPDIR / /tmp.
    explicit TempFile(std::string_view dir = "");
    ~TempFile();

    TempFile(TempFile const&) = delete;
    TempFile& operator=(TempFile const&) = delete;
    TempFile(TempFile&& other) noexcept;
    TempFile& operator=(TempFile&& other) noexcept;

    // Thread-safe positional write. Throws on short writes or errors.
    void pwrite_at(std::uint64_t offset, void const* data, std::size_t bytes);

    // Thread-safe positional read. Throws on short reads (EOF before
    // `bytes` consumed) or errors.
    void pread_at(std::uint64_t offset, void* data, std::size_t bytes);

    // High-water mark — max(end-offset) ever written.
    std::uint64_t size() const noexcept { return high_water_; }

    // Underlying file path (unlinked already; useful for diagnostics
    // via /proc/<pid>/fd/<fd> on Linux).
    std::string const& path() const noexcept { return path_; }

    int fd() const noexcept { return fd_; }

    // For tests / diagnostics — returns the directory the file lives in.
    static std::string resolve_dir(std::string_view explicit_dir);

private:
    int           fd_         = -1;
    std::string   path_;
    std::uint64_t high_water_ = 0;
};

} // namespace pos2gpu
