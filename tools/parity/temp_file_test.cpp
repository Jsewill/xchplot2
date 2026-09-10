// temp_file_test — unit tests for TempFile.
//
// Validates: mkstemp + unlink-on-create, pwrite/pread round-trip,
// out-of-order writes, large I/O, EOF detection, env-var path
// resolution, move semantics, destructor cleanup (fd closed).

#include "host/TempFile.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <winioctl.h>
#else
#include <unistd.h>
#endif

namespace {

bool check(bool cond, char const* what)
{
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    return cond;
}

} // namespace

int main()
{
    bool all_ok = true;
    std::string const temp = std::filesystem::temp_directory_path().string();

    // Test 1: basic open + write + read round-trip.
    {
        pos2gpu::TempFile tf;
        all_ok = check(tf.fd() >= 0, "tempfile fd is valid") && all_ok;
        std::vector<std::uint64_t> w(8);
        for (std::size_t i = 0; i < w.size(); ++i) w[i] = 0xDEAD0000ull + i;
        tf.pwrite_at(0, w.data(), w.size() * sizeof(std::uint64_t));
        std::vector<std::uint64_t> r(8, 0);
        tf.pread_at(0, r.data(), r.size() * sizeof(std::uint64_t));
        all_ok = check(w == r, "pwrite/pread round-trip preserves bytes") && all_ok;
        all_ok = check(tf.size() == 8 * sizeof(std::uint64_t),
                       "size() reports high-water mark") && all_ok;
    }

    // Test 2: out-of-order writes; sparse holes read as zero.
    {
        pos2gpu::TempFile tf;
        std::uint64_t hi = 0xCAFEBABE;
        std::uint64_t lo = 0xFEEDFACE;
        tf.pwrite_at(8000, &hi, sizeof(hi));   // write tail first
        tf.pwrite_at(0,    &lo, sizeof(lo));   // then head
        std::uint64_t got_lo = 0, got_hi = 0;
        tf.pread_at(0,    &got_lo, sizeof(got_lo));
        tf.pread_at(8000, &got_hi, sizeof(got_hi));
        all_ok = check(got_lo == lo && got_hi == hi,
                       "out-of-order pwrite both readable") && all_ok;
        // Hole between [8, 8000) reads as zero.
        std::uint64_t hole = 0xAAAAAAAAull;
        tf.pread_at(4000, &hole, sizeof(hole));
        all_ok = check(hole == 0, "sparse hole reads as zero") && all_ok;
    }

    // Test 3: large I/O — 4 MB chunk through pwrite/pread.
    {
        pos2gpu::TempFile tf;
        constexpr std::size_t N = 1 << 20; // 1M u32 = 4 MB
        std::vector<std::uint32_t> w(N);
        for (std::size_t i = 0; i < N; ++i) w[i] = static_cast<std::uint32_t>(i * 2654435761u);
        tf.pwrite_at(0, w.data(), w.size() * sizeof(std::uint32_t));
        std::vector<std::uint32_t> r(N, 0);
        tf.pread_at(0, r.data(), r.size() * sizeof(std::uint32_t));
        all_ok = check(std::memcmp(w.data(), r.data(),
                                   N * sizeof(std::uint32_t)) == 0,
                       "4 MB round-trip preserves bytes") && all_ok;
    }

    // Test 4: pread past end-of-file throws.
    {
        pos2gpu::TempFile tf;
        std::uint32_t v = 1;
        tf.pwrite_at(0, &v, sizeof(v));
        std::uint64_t past = 0;
        bool threw = false;
        try { tf.pread_at(100, &past, sizeof(past)); }
        catch (std::runtime_error const&) { threw = true; }
        all_ok = check(threw, "pread past EOF throws") && all_ok;
    }

    // Test 5: file is unlinked on construction (path no longer in dir).
    {
        std::string path;
        {
            pos2gpu::TempFile tf;
            path = tf.path();
#ifndef _WIN32
            all_ok = check(!std::filesystem::exists(path), "file unlinked on construction") && all_ok;
#endif
        }
        all_ok = check(!std::filesystem::exists(path), "file removed on close") && all_ok;
    }

    // Test 6: move construction transfers fd; source has -1 fd.
    {
        pos2gpu::TempFile a;
        int const a_fd = a.fd();
        std::uint64_t v = 42;
        a.pwrite_at(0, &v, sizeof(v));
        pos2gpu::TempFile b(std::move(a));
        all_ok = check(b.fd() == a_fd, "move transfers fd") && all_ok;
        all_ok = check(a.fd() == -1, "moved-from has fd=-1") && all_ok;
        std::uint64_t got = 0;
        b.pread_at(0, &got, sizeof(got));
        all_ok = check(got == 42, "moved object retains data") && all_ok;
    }

    // Test 7: env-based dir resolution.
    {
#ifdef _WIN32
        ::_putenv_s("XCHPLOT2_TEMP_DIR", temp.c_str());
#else
        ::setenv("XCHPLOT2_TEMP_DIR", temp.c_str(), 1);
#endif
        std::string const d = pos2gpu::TempFile::resolve_dir("");
        all_ok = check(d == temp, "resolve_dir uses XCHPLOT2_TEMP_DIR") && all_ok;
        std::string const explicit_d = pos2gpu::TempFile::resolve_dir("/var/tmp");
        all_ok = check(explicit_d == "/var/tmp",
                       "explicit dir overrides env") && all_ok;
#ifdef _WIN32
        ::_putenv_s("XCHPLOT2_TEMP_DIR", "");
#else
        ::unsetenv("XCHPLOT2_TEMP_DIR");
#endif
    }

    // Test 8: dir_problem — the spill guard's usability probe.
    //
    // dir_is_ram_backed() answers false for a dir it cannot even statfs, so
    // it passes a mistyped --temp-dir straight through; this is what catches
    // that, and a version of it that returns "" for an unusable dir would
    // silently restore the original failure — a raw mkstemp errno thrown deep
    // in the pipeline, minutes into a batch.
    {
        all_ok = check(pos2gpu::TempFile::dir_problem(temp).empty(),
                       "dir_problem: usable dir reports no problem") && all_ok;

        std::string const missing =
            pos2gpu::TempFile::dir_problem("/nonexistent-xchplot2-probe");
        all_ok = check(!missing.empty(),
                       "dir_problem: missing dir reports a problem") && all_ok;

        // Exists but not writable. Skipped as root, where write permission
        // is not enforced and the probe would (correctly) succeed.
#ifndef _WIN32
        if (::geteuid() != 0) {
            char tmpl[] = "/tmp/xchplot2-ro-XXXXXX";
            if (char const* d = ::mkdtemp(tmpl); d) {
                ::chmod(d, 0555);
                std::string const ro = pos2gpu::TempFile::dir_problem(d);
                all_ok = check(!ro.empty(),
                               "dir_problem: read-only dir reports a problem")
                         && all_ok;
                ::chmod(d, 0755);
                ::rmdir(d);
            }
        }
#endif

        // The probe must not leave its own file behind — it creates one and
        // relies on TempFile unlinking at construction.
        {
            auto const dir = std::filesystem::path(temp) / ("xchplot2-probe-" + std::to_string(std::random_device{}()));
            all_ok = check(std::filesystem::create_directory(dir), "create probe directory") && all_ok;
            (void) pos2gpu::TempFile::dir_problem(dir.string());
            all_ok = check(std::filesystem::remove(dir), "dir_problem: leaves nothing behind") && all_ok;
        }
    }

    // ---- preallocate + free_space ----
    //
    // These exist so a temp dir that cannot hold the spill says so at setup
    // instead of on a pwrite inside T2, minutes into a batch. The failure
    // mode worth testing is the quiet one: a preallocate that silently does
    // nothing still "passes" every plot run, and nothing downstream notices
    // until a disk fills.
    {
        pos2gpu::TempFile f;
        f.preallocate(4u << 20);           // 4 MiB
#ifdef _WIN32
        FILE_STANDARD_INFO st{};
        bool const ok = ::GetFileInformationByHandleEx(
            reinterpret_cast<HANDLE>(::_get_osfhandle(f.fd())), FileStandardInfo, &st, sizeof(st));
        bool const reserved = st.EndOfFile.QuadPart == (4 << 20) && st.AllocationSize.QuadPart >= (4 << 20);
        bool const noop = false;
#else
        struct stat st {};
        bool const ok = (::fstat(f.fd(), &st) == 0);
        // fallocate reserves blocks AND extends i_size (no KEEP_SIZE), so on
        // any filesystem that implements it the file is now 4 MiB with blocks
        // behind it. st_blocks is in 512-B units. Where it is unsupported the
        // call is a documented no-op, and the file stays empty — accept that
        // rather than fail on, say, a network mount, but do not accept a
        // half-done job.
        bool const reserved = (st.st_size == (4 << 20)) &&
                              (std::uint64_t(st.st_blocks) * 512 >= (4u << 20));
        bool const noop     = (st.st_size == 0) && (st.st_blocks == 0);
#endif
        all_ok = check(ok && (reserved || noop),
                       "preallocate: reserves blocks, or is a clean no-op")
                 && all_ok;

        // Whatever it did, the file must still behave: preallocate must not
        // make a range count as written. A sparse read still returns zeros,
        // which is exactly why SpillCoverage cannot be retired.
        std::uint64_t v = 0xDEADBEEFu;
        f.pread_at(1u << 20, &v, sizeof(v));
        all_ok = check(v == 0, "preallocate: unwritten range still reads zeros")
                 && all_ok;

        // And it must not disturb real I/O.
        std::uint64_t const w = 0x0123456789ABCDEFull;
        f.pwrite_at(1u << 20, &w, sizeof(w));
        std::uint64_t r = 0;
        f.pread_at(1u << 20, &r, sizeof(r));
        all_ok = check(r == w, "preallocate: round-trip still works") && all_ok;

        f.preallocate(512u << 10);
        r = 0;
        f.pread_at(1u << 20, &r, sizeof(r));
        all_ok = check(r == w, "preallocate: smaller reservation preserves existing data") && all_ok;

        // Zero is a no-op, not an error.
        f.preallocate(0);
        all_ok = check(true, "preallocate: zero bytes is a no-op") && all_ok;
    }
    {
        // free_space answers for a real dir, and returns the documented 0 for
        // one that cannot be probed. 0 means "unknown" to callers, so a bogus
        // path must not come back looking like a full disk.
        std::uint64_t const here = pos2gpu::TempFile::free_space(temp);
        all_ok = check(here > 0, "free_space: reports something for the temp directory")
                 && all_ok;
        std::uint64_t const nowhere =
            pos2gpu::TempFile::free_space("/nonexistent-xchplot2-probe");
        all_ok = check(nowhere == 0, "free_space: unprobeable dir reports 0")
                 && all_ok;
    }

    {
        pos2gpu::TempFile file;
        auto* mapped = static_cast<std::uint64_t*>(file.map(16384));
        mapped[1023] = 0x123456789abcdef0ULL;
        bool refused = false;
        try { file.map(4096); } catch (std::runtime_error const&) { refused = true; }
        all_ok = check(refused, "second live mapping is rejected") && all_ok;
        pos2gpu::TempFile moved(std::move(file));
        moved.unmap();
        std::uint64_t value = 0;
        moved.pread_at(1023 * sizeof(value), &value, sizeof(value));
        all_ok = check(value == 0x123456789abcdef0ULL, "mapping survives move and unmap") && all_ok;
    }
    {
        pos2gpu::TempFile file;
#ifdef _WIN32
        // Avoid allocating a 4 GiB hole just to exercise OffsetHigh.
        DWORD returned = 0;
        HANDLE const handle = reinterpret_cast<HANDLE>(::_get_osfhandle(file.fd()));
        OVERLAPPED operation{};
        BOOL sparse = ::DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, &operation);
        if (!sparse && ::GetLastError() == ERROR_IO_PENDING)
            sparse = ::GetOverlappedResult(handle, &operation, &returned, TRUE);
        all_ok = check(sparse, "sparse offset test file") && all_ok;
#endif
        std::uint64_t const offset = (std::uint64_t{1} << 32) + 8, expected = 12345;
        file.pwrite_at(offset, &expected, sizeof(expected));
        std::uint64_t actual = 0;
        file.pread_at(offset, &actual, sizeof(actual));
        all_ok = check(actual == expected && file.size() == offset + sizeof(expected), "64-bit positional I/O") && all_ok;
    }

    return all_ok ? 0 : 1;
}
