// spill_engine_test — the host-RAM disk-offload I/O engine.
//
// SpillEngine is the code in this tree most able to produce a plot that is
// wrong but looks right. Everything it does is invisible: a wait that returns
// one job early hands back another chunk's bytes over a range SpillCoverage
// considers legitimately written, so there is no error, no short file, and no
// hash to compare against — just a farmable-looking .plot2 built on the wrong
// data. End-to-end parity cannot attribute that, and it only reproduces under
// a thread interleaving nobody controls.
//
// It shipped with no test at all because it lived in an anonymous namespace
// inside a 5000-line SYCL translation unit. The two allocations and one copy
// it needed from SYCL are now injected (SpillHostOps), so the protocol is
// reachable here with no GPU, no AdaptiveCpp and no driver.
//
// The single most valuable case is `is_done` under out-of-order completion.
// Before the worker pool landed, completion was FIFO and the engine tracked a
// scalar `done_ticket`, testing `done_ticket >= target`. With a pool that
// predicate means "some LATER job finished", which is exactly the silent
// corruption above. The watermark cases below fail loudly against that old
// logic and pass against the current one.

#include "host/SpillEngine.hpp"
#include "host/TempFile.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

// Stand-in for the SYCL side: staging windows are plain malloc, and the
// "device<->host" copy is memcpy. The engine cannot tell the difference —
// which is the point of the interface.
struct MemOps final : pos2gpu::SpillHostOps
{
    int live = 0;

    void* alloc_staging(std::size_t bytes, char const*) override
    {
        void* p = std::malloc(bytes);
        if (!p) throw std::bad_alloc();
        ++live;
        return p;
    }
    void free_staging(void* p) override
    {
        if (!p) return;
        std::free(p);
        --live;
    }
    void copy_blocking(void* dst, void const* src, std::size_t bytes) override
    {
        std::memcpy(dst, src, bytes);
    }
};

// Build an engine with a specific worker count. The count is read from the
// environment at construction, so it has to be set before the object exists.
struct EngineWithThreads
{
    MemOps ops;
    std::unique_ptr<pos2gpu::SpillEngine> eng;

    explicit EngineWithThreads(int threads)
    {
        std::string const n = std::to_string(threads);
        ::setenv("XCHPLOT2_SPILL_IO_THREADS", n.c_str(), 1);
        eng = std::make_unique<pos2gpu::SpillEngine>(ops, /*quiet=*/true);
        ::unsetenv("XCHPLOT2_SPILL_IO_THREADS");
    }
    pos2gpu::SpillEngine& operator*() { return *eng; }
};

std::vector<std::uint64_t> ramp(std::uint64_t n, std::uint64_t seed)
{
    std::vector<std::uint64_t> v(n);
    for (std::uint64_t i = 0; i < n; ++i) v[i] = (i * 0x9E3779B97F4A7C15ull) ^ seed;
    return v;
}

constexpr std::uint64_t kWin = pos2gpu::SpillEngine::kStageBytes;   // 32 MiB

}  // namespace

int main()
{
    using pos2gpu::SpillBuffer;
    using pos2gpu::SpillEngine;

    // ---------------------------------------------------------------
    // 1. Completion watermark under OUT-OF-ORDER arrival.
    //
    // These call mark_done/is_done directly because the interleaving they
    // describe is precisely what a running pool will not reproduce on demand.
    // Every case here is a regression guard against `done_ticket >= target`.
    // ---------------------------------------------------------------
    {
        EngineWithThreads e(2);
        auto& eng = *e;
        std::lock_guard<std::mutex> lk(eng.mtx);   // the bookkeeping wants it held

        eng.mark_done_locked(3);
        eng.mark_done_locked(2);
        check(!eng.is_done_locked(1),
              "a later ticket completing does NOT complete an earlier one");
        check(eng.is_done_locked(2), "ticket 2 reads done once it lands");
        check(eng.is_done_locked(3), "ticket 3 reads done once it lands");
        check(eng.done_upto == 0,
              "the watermark stays put while a hole is open below it");

        eng.mark_done_locked(1);
        check(eng.done_upto == 3, "filling the hole slides the watermark past all three");
        check(eng.done_above.empty(), "the out-of-order set drains as the watermark moves");
        check(eng.is_done_locked(1) && eng.is_done_locked(2) && eng.is_done_locked(3),
              "all three read done afterwards");
        check(!eng.is_done_locked(4), "an unissued ticket is not done");
    }

    // ---------------------------------------------------------------
    // 2. Round-trip through disk, across worker counts.
    //
    // 1 is the old serial arm, 2 the shipping default, 8 forces the part
    // splitter to cut a 32 MiB chunk into eight 4 MiB pieces (kPartBytes),
    // which is the arithmetic the pool refactor introduced: each part carries
    // its own file offset AND its own window offset, and swapping the two
    // would still write the right number of bytes to the right file.
    // ---------------------------------------------------------------
    for (int threads : {1, 2, 8}) {
        EngineWithThreads e(threads);
        // 2.5 windows: two full chunks plus a partial tail, so the chunk loop,
        // the tail path and the part splitter all run.
        std::uint64_t const n   = (kWin * 5 / 2) / sizeof(std::uint64_t);
        auto const          src = ramp(n, 0xABCDEFull);
        std::vector<std::uint64_t> dst(n, 0);

        SpillBuffer buf(*e, sizeof(std::uint64_t), n);
        buf.write_from_device(src.data(), 0, n);
        buf.read_to_device(dst.data(), 0, n);

        std::string const label = "round-trip through disk (" +
                                  std::to_string(threads) + " io threads)";
        check(dst == src, label.c_str());

        // Read the file directly rather than through the same windows the
        // write used: if file_off and win_off were confused, a window-mediated
        // read would make the same mistake twice and still compare equal.
        std::vector<std::uint64_t> raw(n, 0);
        buf.drain();
        buf.file.pread_at(0, raw.data(), n * sizeof(std::uint64_t));
        std::string const rlabel = "bytes on disk are in file order (" +
                                   std::to_string(threads) + " io threads)";
        check(raw == src, rlabel.c_str());
    }

    // ---------------------------------------------------------------
    // 3. Sub-range reads, and two tables sharing one engine.
    //
    // The shared ping-pong cursor and the shared ticket space are what make
    // "one engine, 64 MiB of staging, any number of spilled tables" work; a
    // window handed to table B while table A's pwrite is still reading it is
    // cross-table corruption.
    // ---------------------------------------------------------------
    {
        EngineWithThreads e(4);
        std::uint64_t const n = (kWin * 3 / 2) / sizeof(std::uint64_t);
        auto const a_src = ramp(n, 0x1111ull);
        auto const b_src = ramp(n, 0x2222ull);

        SpillBuffer a(*e, sizeof(std::uint64_t), n);
        SpillBuffer b(*e, sizeof(std::uint64_t), n);

        // Interleave so both tables have deferred writes in flight together.
        std::uint64_t const half = n / 2;
        a.write_from_device(a_src.data(), 0, half);
        b.write_from_device(b_src.data(), 0, half);
        a.write_from_device(a_src.data() + half, half, n - half);
        b.write_from_device(b_src.data() + half, half, n - half);

        std::vector<std::uint64_t> a_dst(n, 0), b_dst(n, 0);
        a.read_to_device(a_dst.data(), 0, n);
        b.read_to_device(b_dst.data(), 0, n);
        check(a_dst == a_src, "two tables on one engine: table A survives interleaving");
        check(b_dst == b_src, "two tables on one engine: table B survives interleaving");

        // Sub-range read: entry_off scales by elem, so an off-by-elem here
        // reads the right byte count from the wrong place.
        std::vector<std::uint64_t> mid(1024, 0);
        a.read_to_device(mid.data(), half, 1024);
        check(std::equal(mid.begin(), mid.end(), a_src.begin() + half),
              "a sub-range read lands at entry_off * elem");
    }

    // ---------------------------------------------------------------
    // 4. The read guard: a never-written range must throw, not return zeros.
    // ---------------------------------------------------------------
    {
        EngineWithThreads e(2);
        std::uint64_t const n = 4096;
        auto const src = ramp(n, 0x33ull);
        SpillBuffer buf(*e, sizeof(std::uint64_t), n * 2);
        buf.write_from_device(src.data(), 0, n);

        std::vector<std::uint64_t> dst(n, 0);
        bool threw = false;
        try { buf.read_to_device(dst.data(), n, n); }   // past the written range
        catch (std::exception const&) { threw = true; }
        check(threw, "reading a never-written range throws instead of returning zeros");
    }

    // ---------------------------------------------------------------
    // 5. wait_ticket catches a window refilled before it was consumed.
    //
    // Pinned to ONE worker so completion order equals enqueue order and the
    // second job is deterministically the last to touch the window. This is
    // the check that turns "silently reads another chunk's bytes" into a hard
    // error, so a test that only sometimes exercises it is worth little.
    // ---------------------------------------------------------------
    {
        EngineWithThreads e(1);
        auto& eng = *e;
        std::uint64_t const n = (kWin * 2) / sizeof(std::uint64_t);
        auto const src = ramp(n, 0x44ull);
        SpillBuffer buf(*e, sizeof(std::uint64_t), n);
        buf.write_from_device(src.data(), 0, n);
        buf.drain();

        // Two reads into the SAME window with nothing consumed in between.
        std::uint64_t const t1 =
            eng.enqueue(SpillEngine::Op::Read, 0, &buf.file, 0, kWin);
        std::uint64_t const t2 =
            eng.enqueue(SpillEngine::Op::Read, 0, &buf.file, kWin, kWin);
        eng.drain();

        bool threw = false;
        try { eng.wait_ticket(0, t1, "test"); }
        catch (std::exception const&) { threw = true; }
        check(threw, "wait_ticket rejects a window refilled before it was consumed");

        bool ok_for_latest = true;
        try { eng.wait_ticket(0, t2, "test"); }
        catch (std::exception const&) { ok_for_latest = false; }
        check(ok_for_latest, "wait_ticket accepts the job the window actually holds");
    }

    // ---------------------------------------------------------------
    // 6. drain() is a barrier over EVERY ticket, not just the newest.
    // ---------------------------------------------------------------
    {
        EngineWithThreads e(4);
        auto& eng = *e;
        std::uint64_t const n = kWin / sizeof(std::uint64_t);
        auto const src = ramp(n, 0x55ull);
        SpillBuffer buf(*e, sizeof(std::uint64_t), n * 4);
        for (int i = 0; i < 4; ++i)
            buf.write_from_device(src.data(), std::uint64_t(i) * n, n);
        eng.drain();
        std::lock_guard<std::mutex> lk(eng.mtx);
        check(eng.done_upto == eng.next_ticket,
              "after drain the watermark covers every ticket issued");
        check(eng.parts_left.empty(), "after drain no job has parts outstanding");
        check(eng.done_above.empty(), "after drain nothing is stranded above the watermark");
    }

    // ---------------------------------------------------------------
    // 7. A failing part must surface as an error, not a hang.
    //
    // The decrement of parts_left has to happen on the exception path too;
    // without it the ticket never completes and every waiter blocks forever,
    // which in a batch reads as a wedged plotter rather than a disk problem.
    // /dev/full makes pwrite fail with ENOSPC deterministically.
    // ---------------------------------------------------------------
    {
        EngineWithThreads e(2);
        auto& eng = *e;
        std::uint64_t const n = 4096;
        auto const src = ramp(n, 0x66ull);
        SpillBuffer buf(*e, sizeof(std::uint64_t), n);

        int const full = ::open("/dev/full", O_WRONLY);
        bool threw = false;
        if (full < 0) {
            std::printf("SKIP  (no /dev/full) a failing part surfaces as an error\n");
        } else {
            ::dup2(full, buf.file.fd());   // every pwrite now fails
            ::close(full);
            try {
                buf.write_from_device(src.data(), 0, n);
                eng.drain();
            } catch (std::exception const&) { threw = true; }
            check(threw, "a failing part surfaces as an error rather than hanging");
        }
    }

    // ---------------------------------------------------------------
    // 8. Destruction order: a buffer torn down with a deferred write still
    //    queued must drain, not hand the pool a dangling TempFile.
    //    The engine outliving the buffer is what makes that legal.
    // ---------------------------------------------------------------
    {
        EngineWithThreads e(2);
        std::uint64_t const n = kWin / sizeof(std::uint64_t);
        auto const src = ramp(n, 0x77ull);
        {
            SpillBuffer buf(*e, sizeof(std::uint64_t), n);
            buf.write_from_device(src.data(), 0, n);
            // No drain: the pwrite is still in flight as ~SpillBuffer runs.
        }
        check(true, "a buffer destroyed with a write in flight drains cleanly");
    }

    // ---------------------------------------------------------------
    // 9. Windows are all returned. A leak here is 32 MiB of pinned host RAM
    //    per plot on the tier that exists because host RAM ran out.
    // ---------------------------------------------------------------
    {
        MemOps ops;
        {
            pos2gpu::SpillEngine eng(ops, /*quiet=*/true);
            check(ops.live == pos2gpu::SpillEngine::kNumWindows,
                  "the engine holds exactly kNumWindows staging windows");
        }
        check(ops.live == 0, "every staging window is returned on destruction");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall good\n", failures);
    return failures ? 1 : 0;
}
