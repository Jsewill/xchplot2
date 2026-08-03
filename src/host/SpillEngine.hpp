// SpillEngine.hpp — host-RAM disk-offload: generalized overlapped/
// double-buffered spill.
//
// (User-facing shape: the README's "Host RAM and disk-offload"; the budget
// policy that selects what lands here is HostRamPolicy.hpp.) Two cooperating
// pieces:
//
//   SpillEngine — ONE per pipeline invocation. Owns the background I/O
//     worker pool, the TWO ping-pong staging windows (64 MiB pinned
//     total), and a single monotonic ticket space. EVERY spilled
//     table shares this one engine, so the pinned staging cost stays
//     fixed at 64 MiB no matter how many tables spill — that is the
//     whole point of the generalization over P1.5's per-table windows.
//
//   SpillBuffer — one per spilled table (h_t1_meta, h_t3, ...). Owns
//     ONLY a TempFile (its own on-disk backing) plus its element size;
//     it holds no windows and no threads. All of its device<->disk
//     traffic flows through the shared engine's windows. The spilled
//     tables have disjoint access phases (h_t1_meta in T1, h_t3 in T3),
//     so the shared windows are never contended; the per-slot / drain
//     waits serialize any cross-table op and preserve the exact
//     byte-for-byte semantics of the original single-table path — the
//     ping-pong cursor and slot tickets are shared, so a deferred write
//     always lands before its window is repopulated, across tables too.
//
// Semantics (unchanged from P1.5, now table-agnostic):
//   - WRITES are DEFERRED + ping-pong: D2H into a free window (wait),
//     hand it to the pool for pwrite, return. The pwrite overlaps the
//     caller's next kernel. A window is not reused until its prior op
//     completes (per-slot wait).
//   - READS are double-buffered: drain pending writes, then pipeline
//     pread(chunk+1) with H2D(chunk).
//   - The streaming-partition source-tile read (T1 sort) is driven via
//     SpillTileReader (see gpu/SpillTileReader.hpp), also double-buffered.
//   - A drain() barrier lands all outstanding writes before a re-read.
// Gated per-table by the budget policy (BatchPlotter) / the legacy
// XCHPLOT2_SPILL_T1META flag; default OFF is byte-identical to the
// all-pinned path. XCHPLOT2_SPILL_NO_OVERLAP=1 forces synchronous I/O
// (A/B measurement only).
//
// WHY THIS IS A HEADER AND NOT PART OF GpuPipeline.cpp
// ----------------------------------------------------
// It used to be an anonymous-namespace struct inside a 5000-line SYCL
// translation unit, which made it unreachable by any test. That is backwards
// relative to risk: the ticket protocol below is the code most able to produce
// a SILENTLY wrong plot — a wait that returns before its own data lands hands
// back another chunk's bytes over a range SpillCoverage considers written, and
// no end-to-end hash can attribute that. Meanwhile the pieces that DID have
// tests (interval arithmetic, budget arithmetic, canaries) are the ones whose
// failures are loud.
//
// The only things the engine needed from SYCL were two staging allocations and
// a blocking device<->host copy, so those are injected as SpillHostOps.
// GpuPipeline.cpp supplies the USM implementation; spill_engine_test supplies a
// std::memcpy one and hammers the protocol with no GPU present.

#pragma once

#include "gpu/SpillTileReader.hpp"
#include "host/SpillCoverage.hpp"
#include "host/TempFile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pos2gpu {

// The engine's whole dependency on the GPU stack, which is two allocations and
// one blocking copy. Implemented over SYCL USM in the pipeline and over plain
// memcpy in the test.
//
// `copy_blocking` MUST have completed when it returns: the caller immediately
// hands the window to an I/O thread (write path) or to the consumer (read
// path), and an outstanding async copy would be a data race against both.
struct SpillHostOps {
    virtual ~SpillHostOps() = default;

    // Host-visible staging window, `bytes` long. Pinned in the real
    // implementation; the name is for the allocation-tracking stats.
    virtual void* alloc_staging(std::size_t bytes, char const* what) = 0;
    virtual void  free_staging(void* p) = 0;

    // Blocking copy between a device allocation and a staging window, either
    // direction. Both pointers are readable/writable by the implementation.
    virtual void  copy_blocking(void* dst, void const* src, std::size_t bytes) = 0;
};

struct SpillEngine {
    // 32 MiB per window. TWO windows = 64 MiB pinned resident, SHARED by
    // all spilled tables. Each window also bounds the streaming-partition
    // source tile (32 MiB / 8 B = 4M u64 entries), see SpillBuffer.
    static constexpr uint64_t kStageBytes = 32ULL << 20;
    // TWO windows, and four was MEASURED not to be worth it (2026-08-02).
    // The triple partition needs a u64 and a u32 tile resident together, so it
    // consumes both windows and has no cross-tile prefetch; four windows would
    // let tile t+1 load while tile t partitions. Built, byte-parity clean,
    // A/B'd at k=28 Tiny over 3 reps: spill stall 8.08 s (4 windows) vs 8.45 s
    // (2) — a 0.37 s difference against a 0.81-1.43 s spread, i.e. nothing.
    // The mechanism explains it: ONE I/O worker thread served the engine, and
    // both preads of a pair are already submitted before either is waited on,
    // so the queue is 2 deep and the worker never idled. A deeper queue cannot
    // make a serial worker faster. The lever for the ~17%-of-wall spill stall
    // is I/O PARALLELISM, not more windows — and 64 MiB of extra pinned
    // staging is the wrong thing to spend on the tier that exists because RAM
    // is scarce.
    //
    // That parallelism is now here, and it deliberately did NOT come from more
    // worker threads pulling whole jobs off the queue. The number of jobs that
    // can be in flight is bounded by kNumWindows, not by the worker count:
    // every caller waits for a window before reusing it, so with two windows
    // at most two jobs exist at once and a pool of eight threads would leave
    // six idle. Splitting each JOB across the workers by byte range is not
    // bounded that way — see kPartBytes and worker_loop.
    static constexpr int      kNumWindows = 2;

    // Smallest slice of a job worth handing to its own thread. A job is cut
    // into one part per worker, floored at this size: at the default two
    // threads a full 32 MiB chunk becomes two 16 MiB pwrites, and the floor
    // only binds on the tail chunk of a table (or at high thread counts,
    // where it caps the split at kStageBytes/kPartBytes = 8 parts). All parts
    // of a job read the SAME window at disjoint byte ranges, which is safe
    // without any extra staging memory, and is the whole reason this beats
    // adding windows. Below a few MiB the syscall and wake-up overhead stops
    // being worth it, and the tail chunk should not be shredded into slivers.
    static constexpr size_t   kPartBytes = 4ull << 20;

    SpillHostOps*     ops = nullptr;
    uint8_t*          win[kNumWindows] = {};   // byte windows; typed by SpillBuffer

    enum class Op { Write, Read };
    // A byte range of one JOB, which is what a worker actually executes. A job
    // is the caller-visible unit of I/O — window `slot` <-> `file` at
    // `off_bytes` — and is atomic from the caller's point of view: its ticket
    // completes only when every part of it has. `win_off` is the offset inside
    // the window; the file offset is the job's plus the same amount, so parts
    // are disjoint in both spaces.
    struct Part { Op op; int slot; pos2gpu::TempFile* file;
                  uint64_t file_off; size_t win_off; size_t bytes;
                  uint64_t ticket; };

    std::vector<std::thread> io_threads;
    std::mutex              mtx;
    std::condition_variable cv_work;                 // workers wait for parts
    std::condition_variable cv_done;                 // producers wait for completion
    std::deque<Part>        parts;
    bool                    stopping = false;
    // Parts of each still-incomplete job that have not finished yet. A job's
    // ticket completes when its entry reaches zero.
    std::unordered_map<uint64_t, unsigned> parts_left;
    uint64_t                next_ticket = 0;         // last enqueued
    // Completion is no longer monotonic in ticket order: N workers finish
    // parts of different jobs in whatever order the kernel schedules them, so
    // job 7 can land before job 6. `done_upto` is the watermark below which
    // EVERY ticket is complete, and `done_above` holds the completed tickets
    // that are still above it. is_done() answers for any single ticket.
    //
    // Getting this wrong is not a performance bug. The old code's
    // `done_ticket >= target` meant "my job is done" only because one FIFO
    // worker made completion order equal enqueue order; with a pool it would
    // mean "some LATER job is done", a wait that returns before its own data
    // has landed, and a silently wrong plot.
    uint64_t                done_upto = 0;
    std::set<uint64_t>      done_above;
    uint64_t                slot_ticket[kNumWindows] = {};  // last job ENQUEUED against each window
    // Last job the worker actually COMPLETED into each window, and the
    // ticket a reader is waiting to consume from it. These exist because
    // wait_slot() waits on slot_ticket[slot] — the LATEST job queued for
    // that window, not the one the caller is about to read. The two are
    // the same only while callers consume each window before resubmitting
    // to it. If that discipline ever breaks, wait_slot() returns happily
    // and hands back a DIFFERENT chunk's bytes: the range is fully written,
    // so SpillCoverage sees nothing wrong, and the result is a silently
    // wrong plot. wait_ticket() below turns that into a hard error for one
    // integer compare, so it stays on in release builds.
    uint64_t                win_last_done[kNumWindows]   = {};
    uint64_t                pending_ticket[kNumWindows]  = {};
    int                     write_slot = 0;          // ping-pong cursor for writes (SHARED across tables)
    std::string             io_error;                // first worker error, re-raised on wait/drain
    bool                    overlap = true;          // XCHPLOT2_SPILL_NO_OVERLAP=1 => synchronous (A/B measure only)
    // XCHPLOT2_SPILL_VERIFY=1: round-trip every written chunk (pread it back
    // into the other window and memcmp) before moving on. Debug only — it
    // serialises the pipeline. This exists to split "the spill I/O itself
    // corrupts data" from "the spill only perturbs the timing of a bug that
    // lives elsewhere", which no amount of end-to-end plot hashing can
    // distinguish. Throws on the first mismatch, naming file+offset.
    bool                    verify  = false;
    uint64_t                verified_chunks = 0;
    // XCHPLOT2_SPILL_IO_DELAY_US=N: the worker sleeps N microseconds before
    // each job. Race amplification, debug only. Every ordering rule in here is
    // "the main thread must not touch a window / a file range until the I/O
    // thread has finished with it"; stretching each I/O turns any violation of
    // that from a rare timing coincidence into a near-certain one. A fault
    // that reproduces under delay and stops reproducing after a fix is much
    // stronger evidence than a soak at the natural ~5% rate.
    unsigned                io_delay_us = 0;
    uint64_t                blocked_ns = 0;          // wall the main thread spent stalled on disk I/O (this plot)
    // Bytes this plot actually moved to and from the temp dir. Counted in the
    // worker after the syscall returns, so they are I/O that happened, not
    // I/O that was planned.
    //
    // They exist because the modelled figure was wrong and nothing caught it:
    // the budget message assumed every spilled table is written once and read
    // once, when the meta tables are sorted IN PLACE and make five passes. A
    // user sizing an NVMe's endurance from that was told about half the truth.
    // HostRamPolicy now models the passes per table, but a model that mirrors
    // control flow in another TU can drift silently — so the run reports what
    // it really did, and the two can be compared.
    std::atomic<uint64_t>   bytes_written{0};
    std::atomic<uint64_t>   bytes_read{0};
    // StreamingPinnedScratch::quiet — the stall line below is per PLOT, so on
    // a long batch it is the noisiest thing the pipeline emits. Suppressed
    // under -q; XCHPLOT2_SPILL_VERIFY's tally is not, because asking for the
    // verify pass IS asking for its verdict.
    bool                    quiet   = false;
    // Workers in the pool. XCHPLOT2_SPILL_IO_THREADS overrides; 1 restores the
    // single serial worker exactly (one part per job, FIFO), which is the A/B
    // arm for any measurement here.
    int                     num_io_threads = kDefaultIoThreads;

    // TWO, and four was measured not to beat it. k=28 Tiny, --max-host-ram
    // min, 6 reps per arm, spill stall in seconds:
    //
    //   1 thread   mean 7.925   sigma 0.607   (7.22 .. 8.60)
    //   2 threads  mean 6.620   sigma 0.562   (5.95 .. 7.64)
    //   4 threads  mean 6.858   sigma 0.407   (6.41 .. 7.58)
    //
    // 1 -> 2 is -1.305 s, -16.5%. Quoted as "about 2.2 sigma" when it landed,
    // which undersells it: that compares the difference of MEANS against a
    // single arm's sample spread. The two-sample statistic is
    // 1.305 / sqrt(0.607^2/6 + 0.562^2/6) = 3.86, and every 2-thread rep but
    // one sits below every 1-thread rep. 2 -> 4 is 0.238 s in the WRONG
    // direction (0.84 by the same statistic), i.e. nothing.
    //
    // CAVEAT ON PROVENANCE: these reps were taken on a box whose GPU was not
    // idle. The spill stall is disk-bound so the ranking is unlikely to move,
    // but the absolute seconds are not a clean baseline — re-measure before
    // quoting them anywhere user-facing.
    //
    // That ceiling is the same one the four-window experiment hit (see
    // kNumWindows), and it is consistent: this NVMe stops rewarding
    // concurrency somewhere around two outstanding operations, so neither
    // more windows nor more threads buys a third. Anyone testing on a device
    // with a deeper sweet spot — a RAID, a datacentre drive — should try
    // XCHPLOT2_SPILL_IO_THREADS higher and measure before assuming it helps.
    static constexpr int    kDefaultIoThreads = 2;

    SpillEngine(SpillHostOps& ops_, bool quiet_)
        : ops(&ops_), quiet(quiet_)
    {
        if (char const* v = std::getenv("XCHPLOT2_SPILL_NO_OVERLAP"); v && v[0] == '1')
            overlap = false;
        if (char const* v = std::getenv("XCHPLOT2_SPILL_VERIFY"); v && v[0] == '1')
            verify = true;
        if (char const* v = std::getenv("XCHPLOT2_SPILL_IO_DELAY_US"); v && v[0])
            io_delay_us = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        if (char const* v = std::getenv("XCHPLOT2_SPILL_IO_THREADS"); v && v[0]) {
            int const n = std::atoi(v);
            if (n >= 1 && n <= 32) num_io_threads = n;
        }
        for (int i = 0; i < kNumWindows; ++i) {
            win[i] = static_cast<uint8_t*>(
                ops->alloc_staging(kStageBytes, "h_spill_stage_window"));
        }
        for (int i = 0; i < num_io_threads; ++i)
            io_threads.emplace_back([this] { worker_loop(); });
    }

    ~SpillEngine() {
        try { drain(); } catch (...) { /* destructor: swallow late I/O errors */ }
        {
            std::lock_guard<std::mutex> lk(mtx);
            stopping = true;
            cv_work.notify_all();
        }
        for (auto& t : io_threads) if (t.joinable()) t.join();
        if (!quiet) {
            double const gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
            uint64_t const w = bytes_written.load(std::memory_order_relaxed);
            uint64_t const r = bytes_read.load(std::memory_order_relaxed);
            std::fprintf(stderr,
                "[spill] this plot: %.2f GiB temp-dir traffic (%.2f W / %.2f R), "
                "pipeline stalled %.2f s on disk I/O (overlap=%s)\n",
                double(w + r) * gib, double(w) * gib, double(r) * gib,
                blocked_ns / 1e9, overlap ? "on" : "off");
        }
        if (verify)
            std::fprintf(stderr,
                "[spill] verify: %llu chunks round-tripped clean\n",
                (unsigned long long)verified_chunks);
        for (int i = 0; i < kNumWindows; ++i)
            if (win[i]) ops->free_staging(win[i]);
    }

    SpillEngine(SpillEngine const&)            = delete;
    SpillEngine& operator=(SpillEngine const&) = delete;

    // ---- completion bookkeeping (call with `mtx` held) ----

    // Has this specific ticket completed? Not "has anything at least this
    // recent completed" — with a worker pool those are different questions,
    // and only this one is safe to consume data on.
    bool is_done_locked(uint64_t ticket) const {
        return ticket <= done_upto || done_above.count(ticket) != 0;
    }

    // Record a completed ticket and slide the watermark over any run of
    // completions that now sits directly above it. Keeps `done_above` bounded
    // by the number of jobs actually in flight, which is at most kNumWindows.
    void mark_done_locked(uint64_t ticket) {
        done_above.insert(ticket);
        while (done_above.erase(done_upto + 1) != 0) ++done_upto;
    }

    // ---- background workers ----
    void worker_loop() {
        for (;;) {
            Part p;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_work.wait(lk, [this] { return stopping || !parts.empty(); });
                if (parts.empty()) return;          // stopping and drained
                p = parts.front();
                parts.pop_front();
            }
            if (io_delay_us)   // debug race amplification; see io_delay_us
                std::this_thread::sleep_for(std::chrono::microseconds(io_delay_us));
            try {
                if (p.op == Op::Write) {
                    p.file->pwrite_at(p.file_off, win[p.slot] + p.win_off, p.bytes);
                    bytes_written.fetch_add(p.bytes, std::memory_order_relaxed);
                } else {
                    p.file->pread_at (p.file_off, win[p.slot] + p.win_off, p.bytes);
                    bytes_read.fetch_add(p.bytes, std::memory_order_relaxed);
                }
            } catch (std::exception const& e) {
                std::lock_guard<std::mutex> lk(mtx);
                if (io_error.empty()) io_error = e.what();
            }
            {
                // The counter must be decremented on the failure path too, or
                // one bad part leaves its ticket forever incomplete and every
                // waiter on it deadlocks instead of seeing io_error.
                std::lock_guard<std::mutex> lk(mtx);
                auto it = parts_left.find(p.ticket);
                if (it != parts_left.end() && --it->second == 0) {
                    parts_left.erase(it);
                    mark_done_locked(p.ticket);
                    // Whose bytes win[slot] holds now. Set only when the LAST
                    // part lands, so a half-filled window is never advertised
                    // as ready.
                    win_last_done[p.slot] = p.ticket;
                }
                cv_done.notify_all();
            }
        }
    }

    // Enqueue one disk op against window `slot` targeting `file`. The
    // window MUST already be safe to touch (caller did wait_slot + D2H
    // for writes; the read path enqueues only into a window whose last
    // H2D has completed).
    //
    // Split across the pool by byte range. The parts touch disjoint ranges of
    // both the window and the file, so they need no coordination with each
    // other — the only thing that has to be atomic is the TICKET, which
    // completes when the last part does.
    uint64_t enqueue(Op op, int slot, pos2gpu::TempFile* file,
                     uint64_t off_bytes, size_t bytes) {
        std::lock_guard<std::mutex> lk(mtx);
        uint64_t const t = ++next_ticket;
        // At least one part even for a zero-byte job, so the ticket always has
        // something to complete it.
        size_t const per   = std::max<size_t>(kPartBytes,
                                              (bytes + num_io_threads - 1) /
                                              std::max(1, num_io_threads));
        unsigned     count = 0;
        for (size_t off = 0; off < bytes; off += per) {
            size_t const n = std::min(per, bytes - off);
            parts.push_back({op, slot, file, off_bytes + off, off, n, t});
            ++count;
        }
        if (count == 0) {                       // bytes == 0
            parts.push_back({op, slot, file, off_bytes, 0, 0, t});
            count = 1;
        }
        parts_left[t] = count;
        slot_ticket[slot] = t;
        if (count == 1) cv_work.notify_one();
        else            cv_work.notify_all();
        return t;
    }

    void rethrow_locked() {
        if (!io_error.empty()) throw std::runtime_error("SpillEngine I/O error: " + io_error);
    }

    // Block until window `slot`'s most recent op has completed.
    void wait_slot(int slot) {
        auto const t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mtx);
        uint64_t const target = slot_ticket[slot];
        cv_done.wait(lk, [this, target] { return is_done_locked(target); });
        blocked_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count();
        rethrow_locked();
    }

    // Block until the SPECIFIC job `ticket` has completed, and verify that
    // window `slot` still holds ITS bytes rather than a later job's. Use
    // this, not wait_slot, whenever the point of the wait is to consume
    // what a particular read produced. See win_last_done above.
    void wait_ticket(int slot, uint64_t ticket, char const* what) {
        auto const t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this, ticket] { return is_done_locked(ticket); });
        blocked_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count();
        rethrow_locked();
        if (win_last_done[slot] == ticket) return;
        throw std::runtime_error(
            std::string("SpillEngine::") + what + ": staging window " +
            std::to_string(slot) + " was refilled before its contents were "
            "consumed — waited for job " + std::to_string(ticket) +
            " but the window now holds job " + std::to_string(win_last_done[slot]) +
            ". Reading it would substitute one chunk's bytes for another's and "
            "silently corrupt the plot; failing loudly instead.");
    }

    // Block until every enqueued op has completed (write-back barrier).
    // `done_upto >= target` and not is_done_locked(target): this has to mean
    // EVERY ticket up to the barrier, not just the newest one, and with a pool
    // the newest can land first.
    void drain() {
        auto const t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mtx);
        uint64_t const target = next_ticket;
        cv_done.wait(lk, [this, target] { return done_upto >= target; });
        blocked_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count();
        rethrow_locked();
    }
};

// One spilled table. Owns its TempFile + element size; borrows the
// shared SpillEngine for all windows/threads/tickets. Method names match
// the original single-table object so the call sites are unchanged.
struct SpillBuffer {
    SpillEngine*      eng  = nullptr;
    pos2gpu::TempFile file;                          // mkstemp+unlink; honors XCHPLOT2_TEMP_DIR
    size_t            elem = 1;                       // element width in bytes

    // `max_entries` is the table's full capacity, not the count it will end
    // up holding — the file is reserved for the worst case so a disk that
    // cannot hold this table says so HERE, at setup, rather than on a pwrite
    // somewhere inside T2 with the batch already running. See
    // TempFile::preallocate.
    SpillBuffer(SpillEngine& e, size_t elem_bytes, uint64_t max_entries)
        : eng(&e), elem(elem_bytes)
    {
        file.preallocate(max_entries * uint64_t(elem_bytes));
    }
    SpillBuffer(SpillBuffer const&)            = delete;
    SpillBuffer& operator=(SpillBuffer const&) = delete;

    // Queued jobs hold a RAW pointer to `file`. Destroying this buffer with a
    // deferred pwrite still in the queue would hand an I/O thread a dangling
    // TempFile — a use-after-free that would read as data corruption. Every
    // present call site happens to drain first (the reads do it implicitly);
    // draining here makes that a property of the type instead of an unwritten
    // rule the next caller has to know.
    //
    // Draining here is itself only safe while the engine outlives the buffer.
    // The pipeline declares the engine FIRST and the buffers after, so
    // reverse-order destruction of those locals gets it right — an ordering
    // nothing in the type system enforces, which is why spill_engine_test
    // pins it.
    ~SpillBuffer() {
        if (!eng) return;
        try { eng->drain(); } catch (...) { /* dtor: late I/O errors are lost */ }
    }

    // Which byte ranges of `file` have been written. See SpillCoverage.hpp for
    // why a read of an unwritten range is the dangerous case (sparse hole ->
    // zeros -> silently short plot, not an error).
    pos2gpu::SpillCoverage coverage;

    void note_written(uint64_t off, uint64_t len) { coverage.note_written(off, len); }

    // Throws unless [off, off+len) is fully covered by prior writes.
    void require_written(uint64_t off, uint64_t len, char const* what) const {
        if (coverage.covered(off, len)) return;
        throw std::runtime_error(
            std::string("SpillBuffer::") + what + ": read of NEVER-WRITTEN range in " +
            file.path() + " — want [" + std::to_string(off) + ", " +
            std::to_string(off + len) + "), covered only to " +
            std::to_string(coverage.covered_to(off)) +
            ". A sparse-hole read returns zeros and would silently corrupt the "
            "plot; failing loudly instead.");
    }

    // Entries that fit one window, and the tile count for the streaming-
    // partition read so each tile fits a window (see SpillTileReader).
    uint64_t win_entries() const { return SpillEngine::kStageBytes / elem; }
    uint64_t tile_count_for(uint64_t count) const {
        uint64_t const we = win_entries();
        return (count + we - 1) / we;
    }
    void drain() { eng->drain(); }

    // ---- WRITE: device buffer -> disk, deferred + ping-pong ----
    void write_from_device(void const* d_src, uint64_t entry_off, uint64_t n) {
        uint8_t const* src = static_cast<uint8_t const*>(d_src);
        uint64_t const nb  = n * elem;
        uint64_t const base = entry_off * elem;
        for (uint64_t done = 0; done < nb; ) {
            uint64_t const c    = std::min<uint64_t>(SpillEngine::kStageBytes, nb - done);
            int const      slot = eng->write_slot;
            eng->wait_slot(slot);                                       // prior op on this window done
            eng->ops->copy_blocking(eng->win[slot], src + done, c);     // D2H into window
            eng->enqueue(SpillEngine::Op::Write, slot, &file, base + done, c);  // async pwrite
            note_written(base + done, c);
            if (!eng->overlap) eng->wait_slot(slot);                    // A/B measure: block on the pwrite
            if (eng->verify) verify_chunk(slot, base + done, c);        // debug: round-trip check
            eng->write_slot ^= 1;
            done += c;
        }
    }

    // Debug (XCHPLOT2_SPILL_VERIFY=1): confirm the chunk just written to
    // `off_bytes` reads back byte-identical. win[slot] still holds the source
    // bytes, so pread the range into the other window and memcmp the two.
    // Serialising by construction — the point is a verdict, not throughput.
    void verify_chunk(int slot, uint64_t off_bytes, size_t bytes) {
        int const vslot = slot ^ 1;
        eng->wait_slot(slot);                       // the pwrite has landed
        eng->wait_slot(vslot);                      // the other window is idle
        eng->enqueue(SpillEngine::Op::Read, vslot, &file, off_bytes, bytes);
        eng->wait_slot(vslot);
        if (std::memcmp(eng->win[slot], eng->win[vslot], bytes) != 0) {
            throw std::runtime_error(
                "SpillBuffer verify: readback mismatch in " + file.path() +
                " at byte offset " + std::to_string(off_bytes) +
                " (" + std::to_string(bytes) + " B) after " +
                std::to_string(eng->verified_chunks) + " good chunks");
        }
        ++eng->verified_chunks;
    }

    // ---- READ: disk -> device buffer, double-buffered ----
    void read_to_device(void* d_dst, uint64_t entry_off, uint64_t n) {
        if (n == 0) return;
        uint8_t*       dst  = static_cast<uint8_t*>(d_dst);
        uint64_t const nb   = n * elem;
        uint64_t const base = entry_off * elem;
        eng->drain();                                                   // pending writes must land first
        require_written(base, nb, "read_to_device");
        if (!eng->overlap) {                                           // A/B measure: serial pread -> H2D
            for (uint64_t done = 0; done < nb; ) {
                uint64_t const c = std::min<uint64_t>(SpillEngine::kStageBytes, nb - done);
                uint64_t const t =
                    eng->enqueue(SpillEngine::Op::Read, 0, &file, base + done, c);
                eng->wait_ticket(0, t, "read_to_device");
                eng->ops->copy_blocking(dst + done, eng->win[0], c);
                done += c;
            }
            return;
        }
        // tk[slot] = the read whose bytes that window is holding for us.
        // Waiting on the ticket rather than the window is what makes a
        // double-submit an error instead of a wrong chunk (see wait_ticket).
        uint64_t tk[SpillEngine::kNumWindows] = {0, 0};
        int slot = 0;
        uint64_t const c0 = std::min<uint64_t>(SpillEngine::kStageBytes, nb);
        tk[slot] = eng->enqueue(SpillEngine::Op::Read, slot, &file, base, c0);
        for (uint64_t done = 0; done < nb; ) {
            uint64_t const c         = std::min<uint64_t>(SpillEngine::kStageBytes, nb - done);
            uint64_t const next_done = done + c;
            int const      next_slot = slot ^ 1;
            if (next_done < nb) {                                       // prefetch next chunk
                uint64_t const nc = std::min<uint64_t>(SpillEngine::kStageBytes, nb - next_done);
                tk[next_slot] = eng->enqueue(SpillEngine::Op::Read, next_slot, &file,
                                             base + next_done, nc);
            }
            eng->wait_ticket(slot, tk[slot], "read_to_device");         // this chunk's pread done
            eng->ops->copy_blocking(dst + done, eng->win[slot], c);     // H2D
            done = next_done;
            slot = next_slot;
        }
    }

    // C-callback view onto the shared engine for the streaming-partition
    // primitive (different TU). Only the u64 h_t1_meta table uses this,
    // so the windows are reinterpreted as u64. See gpu/SpillTileReader.hpp.
    static void thunk_submit(void* ctx, int slot, uint64_t off_bytes, uint64_t bytes) {
        auto* b = static_cast<SpillBuffer*>(ctx);
        b->require_written(off_bytes, bytes, "tile_reader");
        // Remember which job owns this window so thunk_wait can confirm the
        // partition consumes THAT tile and not a later prefetch.
        b->eng->pending_ticket[slot] =
            b->eng->enqueue(SpillEngine::Op::Read, slot, &b->file, off_bytes,
                            static_cast<size_t>(bytes));
    }
    static void thunk_wait(void* ctx, int slot) {
        auto* e = static_cast<SpillBuffer*>(ctx)->eng;
        e->wait_ticket(slot, e->pending_ticket[slot], "tile_reader");
    }
    SpillTileReader tile_reader() {
        // The partition is about to stream the table we just finished writing,
        // and thunk_submit (unlike read_to_device) has no drain of its own.
        // Land every deferred pwrite here so the contract belongs to the type
        // rather than to a drain the one call site remembers to make.
        eng->drain();
        SpillTileReader r;
        r.ctx         = this;
        r.win[0]      = reinterpret_cast<uint64_t*>(eng->win[0]);
        r.win[1]      = reinterpret_cast<uint64_t*>(eng->win[1]);
        r.win_entries = SpillEngine::kStageBytes / sizeof(uint64_t);
        r.overlap     = eng->overlap;
        r.submit      = &thunk_submit;
        r.wait        = &thunk_wait;
        return r;
    }
};

}  // namespace pos2gpu
