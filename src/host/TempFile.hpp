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
// Backs the host-RAM disk-offload path: SpillEngine/SpillBuffer stream the
// cold cap-sized tables (h_t1_meta, h_t3, h_t2_meta, h_t2_xbits) through one
// of these, and h_frags takes a map() instead.

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

    // Pageable, file-backed home for a CPU-touched buffer (the host-RAM
    // disk-offload; see the README's "Host RAM and disk-offload").
    // ftruncate()s the file to `bytes` and MAP_SHARED-maps it, returning a host
    // pointer the CPU (and pageable-host DMA) can use as a drop-in
    // replacement for a pinned allocation. Unlike pinned pages, these
    // are reclaimable: under memory pressure the kernel writes dirty
    // pages back to THIS temp file (not swap) and evicts clean ones, so
    // the resident set can fall below `bytes`. Only ONE mapping per
    // TempFile; call unmap() (or let the destructor run) to release it.
    // NOT usable for buffers a device KERNEL writes via USM-host — the
    // mapping is not device-accessible; use SpillBuffer for those.
    void* map(std::size_t bytes);

    // Release the mapping created by map(). Idempotent.
    void unmap() noexcept;

    // Reserve `bytes` of blocks for this file NOW, before anything is
    // written. Two reasons, and the second is the one that matters:
    //
    //  - EXTENTS. A spill file is created, grown to a couple of GiB and
    //    deleted once per plot. Growing it a 32 MiB pwrite at a time makes
    //    the allocator find room over and over, on a filesystem that is
    //    simultaneously being churned by the previous plot's deletion.
    //    Reserving the whole range in one call gives it the chance to place
    //    the file contiguously.
    //
    //  - ENOSPC WHEN IT IS STILL CHEAP. Without this, a temp dir that cannot
    //    hold the spill fails on some pwrite deep inside T2, minutes into a
    //    batch, as "zero-byte write (disk full?)". With it, the failure lands
    //    at table setup with the size that could not be reserved.
    //
    // Does NOT make the file non-sparse: unwritten ranges still read as
    // zeros, so SpillCoverage stays load-bearing. Silently does nothing on a
    // filesystem without fallocate support (the file simply grows on demand,
    // which is the old behaviour); throws only on a real failure such as
    // ENOSPC.
    void preallocate(std::uint64_t bytes);

    // Bytes available in `dir` (after resolve_dir) to an unprivileged
    // writer, or 0 when statvfs cannot answer. Callers treat 0 as "unknown,
    // do not block on it" — the same stance dir_is_ram_backed takes, and for
    // the same reason: an unprobeable filesystem must not veto a spill that
    // would have worked.
    static std::uint64_t free_space(std::string const& dir);

    // High-water mark — max(end-offset) ever written.
    std::uint64_t size() const noexcept { return high_water_; }

    // Underlying file path (unlinked already; useful for diagnostics
    // via /proc/<pid>/fd/<fd> on Linux).
    std::string const& path() const noexcept { return path_; }

    int fd() const noexcept { return fd_; }

    // For tests / diagnostics — returns the directory the file lives in.
    static std::string resolve_dir(std::string_view explicit_dir);

    // True when the filesystem hosting `dir` (after resolve_dir) keeps file
    // contents in RAM — tmpfs, ramfs, or hugetlbfs. Spilling there consumes
    // the very RAM a --max-host-ram budget is meant to bound, so callers use
    // this to refuse a RAM-backed spill target before doing any heavy work.
    // A zram/zswap SWAP device backing a real disk filesystem is NOT flagged:
    // only the mount's own fs magic is inspected, so files that actually live
    // on btrfs/ext4 pass even when the system swaps to compressed RAM.
    // Returns false if statfs() fails — an unprobeable fs must not block
    // spilling. That includes a dir that does not exist, so this is NOT a
    // usability check; pair it with dir_problem().
    static bool dir_is_ram_backed(std::string const& dir);

    // "" when a spill file can actually be created in `dir` (after
    // resolve_dir), otherwise a human-readable reason.
    //
    // Probes by doing the real thing — create a temp file and delete it —
    // rather than stat/access, which miss read-only mounts, full
    // filesystems and ACLs, and would hand back a pass that fails minutes
    // later. Without this, a mistyped --temp-dir (the flag the tmpfs error
    // tells users to reach for) sails past the guard and dies deep in the
    // pipeline with a raw mkstemp errno, after the batch has begun.
    static std::string dir_problem(std::string const& dir);

private:
    int           fd_         = -1;
    std::string   path_;
    std::uint64_t high_water_ = 0;
    void*         map_        = nullptr;   // MAP_SHARED region, if map() was called
    std::size_t   map_bytes_  = 0;
};

} // namespace pos2gpu
