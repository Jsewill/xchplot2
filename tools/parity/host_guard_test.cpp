// host_guard_test — HostGuard's redzone arithmetic.
//
// The dangerous failure mode of a canary is that it never fires: a clean
// soak then "proves" there is no overrun when in fact nothing was ever
// being checked. A green end-to-end plot run cannot tell those apart, so
// the deliberate-overrun cases below are the only thing standing behind
// any conclusion drawn from the guard.
//
// HostGuard reads its config once, from the environment, in a function-
// local static — so this test re-execs itself with XCHPLOT2_HOST_GUARD set
// rather than trying to reconfigure a live instance.
//
// Pure arithmetic over malloc'd memory, no SYCL/GPU — runs anywhere.

#include "host/HostGuard.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

// A guarded allocation over plain malloc, mirroring how GpuPipeline wraps
// sycl::malloc_host.
struct Alloc {
    void*  base = nullptr;
    void*  user = nullptr;
    size_t bytes = 0;

    Alloc(size_t n, char const* what) : bytes(n)
    {
        base = std::malloc(n + 2 * pos2gpu::host_guard_pad());
        user = pos2gpu::host_guard_arm(base, n, what);
    }
    void* release(char const* where)
    {
        void* b = pos2gpu::host_guard_disarm(user, where);
        user = nullptr;
        return b;
    }
    ~Alloc() { if (user) pos2gpu::host_guard_disarm(user, "~Alloc"); std::free(base); }
};

}  // namespace

int main(int argc, char** argv)
{
    // Re-exec once with the guard enabled (see file header).
    if (argc < 2 || std::string(argv[1]) != "--armed") {
        setenv("XCHPLOT2_HOST_GUARD", "1", 1);
        std::vector<char*> av{argv[0], const_cast<char*>("--armed"), nullptr};
        execv(argv[0], av.data());
        std::perror("execv");
        return 1;
    }

    auto& g = pos2gpu::HostGuard::instance();
    check(g.enabled(),                "guard enables from XCHPLOT2_HOST_GUARD");
    check(g.pad_bytes() == (1u << 20), "default redzone is 1 MiB per side");

    {   // Clean lifetime: no damage, and the user pointer is offset.
        Alloc a(4096, "clean");
        check(a.user == static_cast<char*>(a.base) + g.pad_bytes(),
                                       "user pointer sits past the head redzone");
        check(g.live_count() == 1,     "armed allocation is tracked");
        std::memset(a.user, 0xAB, a.bytes);   // legal: entirely in bounds
        uint64_t const before = g.damage_count();
        g.check_all("clean");
        check(g.damage_count() == before, "in-bounds writes do not trip the guard");
        a.release("clean-free");
        check(g.live_count() == 0,     "disarm drops the tracking entry");
        check(g.damage_count() == before, "a clean free reports nothing");
    }

    {   // One byte past the end — the cursor-runs-long case.
        Alloc a(4096, "overrun");
        uint64_t const before = g.damage_count();
        static_cast<char*>(a.user)[a.bytes] = 0x7F;
        g.check_all("overrun");
        check(g.damage_count() == before + 1, "a 1-byte overrun is detected");
    }

    {   // One byte before the start.
        Alloc a(4096, "underrun");
        uint64_t const before = g.damage_count();
        static_cast<char*>(a.user)[-1] = 0x7F;
        g.check_all("underrun");
        check(g.damage_count() == before + 1, "a 1-byte underrun is detected");
    }

    {   // Zeroing the redzone must count as damage: a stray memset is the
        // most likely real-world overrun, and a zero-tolerant pattern would
        // sail straight through it.
        Alloc a(4096, "zeroed");
        uint64_t const before = g.damage_count();
        std::memset(static_cast<char*>(a.user) + a.bytes, 0, 64);
        g.check_all("zeroed");
        check(g.damage_count() == before + 1, "a zeroing overrun is detected");
    }

    {   // Two allocations must not share a pattern, or copying one buffer
        // over another would look clean.
        Alloc a(4096, "distinct-a");
        Alloc b(4096, "distinct-b");
        uint64_t const before = g.damage_count();
        std::memcpy(static_cast<char*>(b.user) + b.bytes,
                    static_cast<char*>(a.user) + a.bytes, 4096);
        g.check_all("distinct");
        check(g.damage_count() == before + 1,
              "another allocation's redzone does not pass as this one's");
    }

    {   // A shifted copy of this buffer's own redzone must not pass either.
        Alloc a(4096, "shifted");
        uint64_t const before = g.damage_count();
        char* tail = static_cast<char*>(a.user) + a.bytes;
        std::memmove(tail, tail + 8, 4096);
        g.check_all("shifted");
        check(g.damage_count() == before + 1,
              "a shifted copy of the same redzone is detected");
    }

    {   // Damage is reported once, not at every later checkpoint.
        Alloc a(4096, "repaint");
        static_cast<char*>(a.user)[a.bytes] = 0x7F;
        g.check_all("repaint-1");
        uint64_t const after_first = g.damage_count();
        g.check_all("repaint-2");
        check(g.damage_count() == after_first,
              "a reported overrun is not re-reported at the next checkpoint");
    }

    {   // Pointers the guard never armed must pass through untouched, so
        // mixed guarded/unguarded frees stay safe.
        int stack_local = 0;
        check(pos2gpu::host_guard_disarm(&stack_local, "foreign") == &stack_local,
              "an unarmed pointer passes through disarm unchanged");
        check(pos2gpu::host_guard_disarm(nullptr, "null") == nullptr,
              "disarm(nullptr) is a no-op");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall good\n", failures);
    return failures ? 1 : 0;
}
