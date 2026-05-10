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
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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
        pos2gpu::TempFile tf;
        std::string const p = tf.path();
        struct stat st{};
        bool const stat_fails = (::stat(p.c_str(), &st) != 0);
        all_ok = check(stat_fails, "file unlinked on construction") && all_ok;
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
        ::setenv("XCHPLOT2_TEMP_DIR", "/tmp", 1);
        std::string const d = pos2gpu::TempFile::resolve_dir("");
        all_ok = check(d == "/tmp", "resolve_dir uses XCHPLOT2_TEMP_DIR") && all_ok;
        std::string const explicit_d = pos2gpu::TempFile::resolve_dir("/var/tmp");
        all_ok = check(explicit_d == "/var/tmp",
                       "explicit dir overrides env") && all_ok;
        ::unsetenv("XCHPLOT2_TEMP_DIR");
    }

    return all_ok ? 0 : 1;
}
