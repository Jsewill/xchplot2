// GpuPipeline.cu — orchestrates Xs → T1 → T2 → T3 on the device, with
// CUB radix sort between phases (each phase consumes sorted-by-match_info
// input). Final T3 output is sorted by proof_fragment (low 2k bits) to
// match pos2-chip Table3Constructor::post_construct_span.
//
// Two overloads live here:
//   run_gpu_pipeline(cfg)       — transient pool, one-shot.
//   run_gpu_pipeline(cfg, pool) — shared pool, batch-friendly. This is the
//                                 real implementation; the one-shot form
//                                 just wraps it in a temporary pool.

#include "host/GpuPipeline.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/HostPinnedPool.hpp"
#include "host/PoolSizing.hpp"
#include "host/TempFile.hpp"   // P1 host-RAM disk-offload (XCHPLOT2_SPILL_T1META)
#include "host/SpillCoverage.hpp"  // spill read guard — see header
#include "host/HostGuard.hpp"      // pinned-host redzones — see header

#include "gpu/AesGpu.cuh"
#include "gpu/XsKernel.cuh"
#include "gpu/XsKernels.cuh"   // launch_xs_gen / launch_xs_pack (stage 4e)
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"
#include "gpu/PipelineKernels.cuh"
#include "gpu/Sort.cuh"
#include "gpu/StreamingPartition.cuh"
#include "gpu/SyclBackend.hpp"

#include <sycl/sycl.hpp>


#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pos2gpu {

namespace {


// =====================================================================
// T1 sort: by match_info, low k bits, stable. Uses CUB SortPairs with
// (key=match_info, value=index) then permutes T1Pairings.
// =====================================================================
// T2 sort: same shape — sort indices by match_info.
// =====================================================================
// Streaming allocation tracker.
//
// Wraps cudaMalloc / cudaFree so we can: (a) account for live/peak VRAM
// used by the streaming pipeline, (b) honour a soft device-memory cap
// set via POS2GPU_MAX_VRAM_MB (throws before the underlying cudaMalloc
// when an alloc would push live past the cap), and (c) emit a per-alloc
// trace under POS2GPU_STREAMING_STATS=1 for manual audits.
//
// Pinned host allocations are NOT counted — the cap is specifically for
// device VRAM, and the pinned D2H staging buffer is host-resident.
// =====================================================================
struct StreamingStats {
    size_t cap  = 0;   // 0 = no cap
    size_t live = 0;
    size_t peak = 0;
    std::unordered_map<void*, size_t> sizes;
    // Per-plot OWNED pinned-host allocations (h_xs, h_t1_meta when not
    // caller-provided, sort staging, ...). Tracked separately from the
    // device map: they don't count toward the VRAM cap, but they must
    // be released on a mid-plot throw all the same — under
    // --continue-on-error a failed plot would otherwise strand multiple
    // GB of pinned host memory per failure, starving peer workers.
    // Buffers owned by the caller (scratch fields / HostPinnedPool) are
    // never tracked here.
    std::unordered_set<void*> host_ptrs;
    bool        verbose = false;
    char const* phase   = "(init)";

    // Free any allocations still alive on destruction. If the streaming
    // pipeline throws partway (e.g. d_xs_temp OOM after d_xs already
    // succeeded), this dtor releases the still-live device and pinned-
    // host buffers instead of leaking them across batch iterations.
    ~StreamingStats() {
        if (sizes.empty() && host_ptrs.empty()) return;
        auto& q = sycl_backend::queue();
        for (auto& [ptr, _bytes] : sizes) {
            if (ptr) sycl::free(ptr, q);
        }
        sizes.clear();
        for (void* ptr : host_ptrs) {
            if (ptr) sycl::free(pos2gpu::host_guard_disarm(ptr, "~StreamingStats"), q);
        }
        host_ptrs.clear();
    }
};

inline void s_init_from_env(StreamingStats& s)
{
    if (char const* v = std::getenv("POS2GPU_MAX_VRAM_MB"); v && v[0]) {
        s.cap = size_t(std::strtoull(v, nullptr, 10)) * (1ULL << 20);
    }
    if (char const* v = std::getenv("POS2GPU_STREAMING_STATS"); v && v[0] == '1') {
        s.verbose = true;
    }
}

// Format a byte count as both raw bytes and decimal MB. The previous
// `bytes >> 20` form (integer right-shift = truncating divide by 1 MiB)
// rounded any sub-MiB request down to "0 MB", which masked both the
// real allocation size and any genuine zero-byte sizing bug at the
// call site. Use this helper in every error path so a future
// `requested=0` is unambiguous (raw bytes settles it).
inline std::string s_fmt_bytes(size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf),
                  "%zu bytes (%.2f MB)", bytes, bytes / 1048576.0);
    return std::string(buf);
}

template <typename T>
inline void s_malloc(StreamingStats& s, T*& out, size_t bytes, char const* reason)
{
    // Zero-byte requests come from sizing queries that returned 0,
    // which downstream callers honour as "skip this alloc" only by
    // accident (sycl::malloc_device(0) returns null on HIP). Surface
    // the actual upstream cause instead of triggering the misleading
    // "Card likely too small" path below.
    if (bytes == 0) {
        throw std::runtime_error(
            std::string("internal: s_malloc('") + reason + "') called with "
            "bytes=0 — an upstream sizing query returned 0 (count=0). On "
            "AMD/HIP this most often indicates a kernel correctness issue "
            "on an unvalidated device — either an AOT target outside the "
            "validated set (the gfx1013/RDNA1 community spoof is the known "
            "case) or AdaptiveCpp's generic SSCP JIT miscompiling a kernel "
            "for the actual gfx ISA. Run the parity tests on this device "
            "to localise: sycl_g_x_parity, sycl_sort_parity, "
            "sycl_bucket_offsets_parity, sycl_t1_parity.");
    }
    if (s.cap && s.live + bytes > s.cap) {
        throw std::runtime_error(
            std::string("streaming VRAM cap: phase=") + s.phase +
            " alloc=" + reason +
            " live=" + s_fmt_bytes(s.live) +
            " + new=" + s_fmt_bytes(bytes) +
            " would exceed cap=" + s_fmt_bytes(s.cap));
    }
    void* p = sycl::malloc_device(bytes, sycl_backend::queue());
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_device(") + reason + "): null — phase=" +
            s.phase + " requested=" + s_fmt_bytes(bytes) +
            " live=" + s_fmt_bytes(s.live) +
            ". Card likely too small for this k via the streaming "
            "pipeline; try a smaller k or a card with more VRAM.");
    }
    out = static_cast<T*>(p);
    s.live += bytes;
    if (s.live > s.peak) s.peak = s.live;
    s.sizes[p] = bytes;
    if (s.verbose) {
        std::fprintf(stderr,
            "[stream %-8s] +%7.2f MB  %-20s  live=%8.2f  peak=%8.2f\n",
            s.phase, bytes / 1048576.0, reason,
            s.live / 1048576.0, s.peak / 1048576.0);
    }
}

// Pinned-host allocation with leak tracking (see StreamingStats::
// host_ptrs). Throws on allocation failure. Only OWNED per-plot buffers
// go through this — caller-provided scratch and HostPinnedPool slots
// are managed by their owners.
inline void* s_malloc_host_raw(StreamingStats& s, size_t bytes,
                               char const* what, sycl::queue& q)
{
    host_pinned_reserve_check(bytes, what);
    // XCHPLOT2_HOST_GUARD: over-allocate and paint redzones either side, so
    // a kernel writing past this buffer is caught here instead of silently
    // damaging whichever allocation follows it. Off by default and then
    // pad is 0, i.e. byte-for-byte the original request. See HostGuard.hpp.
    size_t const pad = pos2gpu::host_guard_pad();
    void* p = sycl::malloc_host(bytes + 2 * pad, q);
    if (!p) {
        throw std::runtime_error(
            std::string("sycl::malloc_host(") + what + ") failed");
    }
    p = pos2gpu::host_guard_arm(p, bytes, what);
    s.host_ptrs.insert(p);
    return p;
}

template <typename T>
inline T* s_malloc_host(StreamingStats& s, size_t bytes,
                        char const* what, sycl::queue& q)
{
    return static_cast<T*>(s_malloc_host_raw(s, bytes, what, q));
}

// Free a pinned-host pointer and drop it from leak tracking. Safe on
// pointers that were never tracked (erase is a no-op).
inline void s_free_host(StreamingStats& s, void* p, sycl::queue& q)
{
    if (!p) return;
    s.host_ptrs.erase(p);
    // Checks the redzones and hands back the real base to free. A no-op
    // returning `p` unchanged when the guard is off or never armed it.
    sycl::free(pos2gpu::host_guard_disarm(p, "s_free_host"), q);
}

template <typename T>
inline void s_free(StreamingStats& s, T*& ptr)
{
    if (!ptr) return;
    void* raw = static_cast<void*>(ptr);
    auto it = s.sizes.find(raw);
    if (it != s.sizes.end()) {
        s.live -= it->second;
        if (s.verbose) {
            std::fprintf(stderr,
                "[stream %-8s] -%7.2f MB  %-20s  live=%8.2f  peak=%8.2f\n",
                s.phase, it->second / 1048576.0, "(free)",
                s.live / 1048576.0, s.peak / 1048576.0);
        }
        s.sizes.erase(it);
    }
    sycl::free(raw, sycl_backend::queue());
    ptr = nullptr;
}

// ---------------------------------------------------------------------
// Host-RAM disk-offload — generalized overlapped/double-buffered spill.
// See docs/host-ram-disk-offload.md. Two cooperating pieces:
//
//   SpillEngine — ONE per pipeline invocation. Owns the background I/O
//     worker thread, the TWO ping-pong staging windows (64 MiB pinned
//     total), and a single monotonic FIFO ticket space. EVERY spilled
//     table shares this one engine, so the pinned staging cost stays
//     fixed at 64 MiB no matter how many tables spill — that is the
//     whole point of the generalization over P1.5's per-table windows.
//
//   SpillBuffer — one per spilled table (h_t1_meta, h_t3, ...). Owns
//     ONLY a TempFile (its own on-disk backing) plus its element size;
//     it holds no windows and no thread. All of its device<->disk
//     traffic flows through the shared engine's windows. The spilled
//     tables have disjoint access phases (h_t1_meta in T1, h_t3 in T3),
//     so the shared windows are never contended; the FIFO worker + the
//     per-slot / drain waits serialize any cross-table op and preserve
//     the exact byte-for-byte semantics of the original single-table
//     path — the ping-pong cursor and slot tickets are shared, so a
//     deferred write always lands before its window is repopulated,
//     across tables too.
//
// Semantics (unchanged from P1.5, now table-agnostic):
//   - WRITES are DEFERRED + ping-pong: D2H into a free window (wait),
//     hand it to the worker for pwrite, return. The pwrite overlaps the
//     caller's next kernel. A window is not reused until its prior op
//     completes (per-slot wait).
//   - READS are double-buffered: drain pending writes, then pipeline
//     pread(chunk+1) with H2D(chunk).
//   - The streaming-partition source-tile read (T1 sort) is driven via
//     SpillTileReader (see StreamingPartition.cuh), also double-buffered.
//   - A drain() barrier lands all outstanding writes before a re-read.
// Gated per-table by the budget policy (BatchPlotter) / the legacy
// XCHPLOT2_SPILL_T1META flag; default OFF is byte-identical to the
// all-pinned path. XCHPLOT2_SPILL_NO_OVERLAP=1 forces synchronous I/O
// (A/B measurement only).
struct SpillEngine {
    // 32 MiB per window. TWO windows = 64 MiB pinned resident, SHARED by
    // all spilled tables. Each window also bounds the streaming-partition
    // source tile (32 MiB / 8 B = 4M u64 entries), see SpillBuffer.
    static constexpr uint64_t kStageBytes = 32ULL << 20;
    static constexpr int      kNumWindows = 2;

    StreamingStats*   stats = nullptr;
    sycl::queue*      q     = nullptr;
    uint8_t*          win[kNumWindows] = {nullptr, nullptr};   // byte windows; typed by SpillBuffer

    enum class Op { Write, Read, Stop };
    struct Job { Op op; int slot; pos2gpu::TempFile* file;
                 uint64_t off_bytes; size_t bytes; uint64_t ticket; };

    std::thread             io_thread;
    std::mutex              mtx;
    std::condition_variable cv_work;                 // worker waits for jobs
    std::condition_variable cv_done;                 // producers wait for completion
    std::deque<Job>         jobs;
    uint64_t                next_ticket = 0;         // last enqueued
    uint64_t                done_ticket = 0;         // last completed (FIFO => monotonic)
    uint64_t                slot_ticket[kNumWindows] = {0, 0};  // last job ENQUEUED against each window
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
    uint64_t                win_last_done[kNumWindows]   = {0, 0};
    uint64_t                pending_ticket[kNumWindows]  = {0, 0};
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

    SpillEngine(StreamingStats& s, sycl::queue& queue)
        : stats(&s), q(&queue)
    {
        if (char const* v = std::getenv("XCHPLOT2_SPILL_NO_OVERLAP"); v && v[0] == '1')
            overlap = false;
        if (char const* v = std::getenv("XCHPLOT2_SPILL_VERIFY"); v && v[0] == '1')
            verify = true;
        if (char const* v = std::getenv("XCHPLOT2_SPILL_IO_DELAY_US"); v && v[0])
            io_delay_us = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        for (int i = 0; i < kNumWindows; ++i) {
            win[i] = static_cast<uint8_t*>(s_malloc_host_raw(
                s, kStageBytes, "h_spill_stage_window", queue));
        }
        io_thread = std::thread([this] { worker_loop(); });
    }

    ~SpillEngine() {
        try { drain(); } catch (...) { /* destructor: swallow late I/O errors */ }
        {
            std::lock_guard<std::mutex> lk(mtx);
            jobs.push_back({Op::Stop, 0, nullptr, 0, 0, 0});
            cv_work.notify_all();
        }
        if (io_thread.joinable()) io_thread.join();
        std::fprintf(stderr,
            "[spill] pipeline stalled %.2f s on disk I/O this plot (overlap=%s)\n",
            blocked_ns / 1e9, overlap ? "on" : "off");
        if (verify)
            std::fprintf(stderr,
                "[spill] verify: %llu chunks round-tripped clean\n",
                (unsigned long long)verified_chunks);
        for (int i = 0; i < kNumWindows; ++i)
            if (win[i]) s_free_host(*stats, win[i], *q);
    }

    SpillEngine(SpillEngine const&)            = delete;
    SpillEngine& operator=(SpillEngine const&) = delete;

    // ---- background worker + ticket bookkeeping ----
    void worker_loop() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv_work.wait(lk, [this] { return !jobs.empty(); });
                j = jobs.front();
                jobs.pop_front();
            }
            if (j.op == Op::Stop) return;
            if (io_delay_us)   // debug race amplification; see io_delay_us
                std::this_thread::sleep_for(std::chrono::microseconds(io_delay_us));
            try {
                if (j.op == Op::Write) j.file->pwrite_at(j.off_bytes, win[j.slot], j.bytes);
                else                   j.file->pread_at (j.off_bytes, win[j.slot], j.bytes);
            } catch (std::exception const& e) {
                std::lock_guard<std::mutex> lk(mtx);
                if (io_error.empty()) io_error = e.what();
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                done_ticket           = j.ticket;    // FIFO => monotonic
                win_last_done[j.slot] = j.ticket;    // whose bytes win[slot] holds now
                cv_done.notify_all();
            }
        }
    }

    // Enqueue one disk op against window `slot` targeting `file`. The
    // window MUST already be safe to touch (caller did wait_slot + D2H
    // for writes; the read path enqueues only into a window whose last
    // H2D has completed).
    uint64_t enqueue(Op op, int slot, pos2gpu::TempFile* file,
                     uint64_t off_bytes, size_t bytes) {
        std::lock_guard<std::mutex> lk(mtx);
        uint64_t const t = ++next_ticket;
        jobs.push_back({op, slot, file, off_bytes, bytes, t});
        slot_ticket[slot] = t;
        cv_work.notify_one();
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
        cv_done.wait(lk, [this, target] { return done_ticket >= target; });
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
        cv_done.wait(lk, [this, ticket] { return done_ticket >= ticket; });
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
    void drain() {
        auto const t0 = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lk(mtx);
        uint64_t const target = next_ticket;
        cv_done.wait(lk, [this, target] { return done_ticket >= target; });
        blocked_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count();
        rethrow_locked();
    }
};

// One spilled table. Owns its TempFile + element size; borrows the
// shared SpillEngine for all windows/thread/tickets. Method names match
// the original single-table object so the call sites are unchanged.
struct SpillBuffer {
    SpillEngine*      eng  = nullptr;
    pos2gpu::TempFile file;                          // mkstemp+unlink; honors XCHPLOT2_TEMP_DIR
    size_t            elem = 1;                       // element width in bytes

    SpillBuffer(SpillEngine& e, size_t elem_bytes) : eng(&e), elem(elem_bytes) {}
    SpillBuffer(SpillBuffer const&)            = delete;
    SpillBuffer& operator=(SpillBuffer const&) = delete;

    // Queued jobs hold a RAW pointer to `file`. Destroying this buffer with a
    // deferred pwrite still in the queue would hand the I/O thread a dangling
    // TempFile — a use-after-free that would read as data corruption. Every
    // present call site happens to drain first (the reads do it implicitly);
    // draining here makes that a property of the type instead of an unwritten
    // rule the next caller has to know.
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
            eng->q->memcpy(eng->win[slot], src + done, c).wait();       // D2H into window
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
                eng->q->memcpy(dst + done, eng->win[0], c).wait();
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
            eng->q->memcpy(dst + done, eng->win[slot], c).wait();       // H2D
            done = next_done;
            slot = next_slot;
        }
    }

    // C-callback view onto the shared engine for the streaming-partition
    // primitive (different TU). Only the u64 h_t1_meta table uses this,
    // so the windows are reinterpreted as u64. See StreamingPartition.cuh.
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

// Sanity-check t1_count after T1 match. Healthy plots produce ~2^k
// entries; anything below total_xs/64 (= 2^(k-6)) — let alone literal
// zero — points at kernel correctness on the device, not a VRAM
// shortfall. Catching this here surfaces a clear diagnostic instead of
// letting downstream sort-scratch alloc fail with the misleading
// "Card likely too small" message. Two AMD/HIP cases produce 0 T1
// matches at k=28: the gfx1013/RDNA1 community spoof on a W5700, and
// AdaptiveCpp's generic SSCP JIT on the same RDNA1 silicon (the JIT
// path is theoretically more compatible than the AOT spoof but has
// been observed to miscompile the matcher). Only the OOM further down
// was visible before this check.
inline void validate_t1_count(uint64_t t1_count, int k)
{
    uint64_t const min_plausible = (1ULL << k) >> 6;
    if (t1_count >= min_plausible) return;

    // Did the kernels run at all? A device that faulted mid-run reaches this
    // check with the same t1_count == 0 as a device that miscompiled the
    // matcher, and the advice for the two is opposite: one is a driver/device
    // problem the parity tests cannot reproduce, the other is exactly what
    // they exist to catch. The async error count decides it.
    if (unsigned const n = async_error_count(); n > 0) {
        std::string const first = first_async_error();
        throw std::runtime_error(
            "T1 match produced " + std::to_string(t1_count) + " entries, and " +
            std::to_string(n) + " asynchronous backend error(s) were reported "
            "during the run. The kernels did not compute a wrong answer — they "
            "did not execute. This is a driver or device fault, NOT a codegen "
            "or parity problem, and the parity tests will not reproduce it.\n"
            "  First backend error: " + first + "\n"
            "  Reading Level Zero codes: 0x70000001 (ze:1879048193) is "
            "DEVICE_LOST and is the ROOT event. The 0x70000003 "
            "(ze:1879048195) OUT_OF_DEVICE_MEMORY storm that follows it is a "
            "dead context reporting on every subsequent call — it is not a "
            "memory shortage, and chasing it wastes time.\n"
            "  DEVICE_LOST part-way through a run usually means a kernel "
            "outran the GPU's job/preemption timeout and the driver reset the "
            "engine. Check `dmesg` for xe/i915 reset or hangcheck lines. On "
            "the Xe driver the limit is "
            "/sys/class/drm/card*/device/tile0/gt0/engines/*/job_timeout_ms "
            "(and preempt_timeout_us); i915 uses "
            "i915.enable_hangcheck / preempt_timeout_ms. Re-running at a "
            "smaller k shortens every kernel and is the fastest way to confirm "
            "the diagnosis: if a small k completes and k=28 does not, it is "
            "the timeout, not the code.");
    }

    throw std::runtime_error(
        "T1 match produced " + std::to_string(t1_count) + " entries "
        "(expected ~2^" + std::to_string(k) + " = " +
        std::to_string(1ULL << k) + " for k=" + std::to_string(k) +
        "). No asynchronous backend errors were reported, so the kernels ran "
        "and returned wrong results — a device codegen issue, not a VRAM "
        "shortfall. On AMD/HIP this most often means the "
        "AdaptiveCpp target produced wrong output for the actual gfx "
        "ISA — either the gfx1013/RDNA1 community AOT spoof or the "
        "generic SSCP JIT path on an unvalidated card. Build the "
        "parity tests via cmake and verify on this device: "
        "sycl_g_x_parity, sycl_sort_parity, sycl_bucket_offsets_parity, "
        "sycl_t1_parity. The first three exercise individual kernels at "
        "small N; sycl_t1_parity runs the full T1 matcher against the "
        "pos2-chip CPU reference and is the closest reproducer of the "
        "k=28 failure. README's 'Community-tested, not parity-validated' "
        "caveat applies.");
}

} // namespace

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg,
                                   GpuBufferPool& pool,
                                   int pinned_index)
{

    sycl::queue& q = sycl_backend::queue();
    // Grant the T2/T3 two-phase match its VRAM budget for this queue before
    // any match runs. Nothing else bounds that allocation, and it is not
    // covered by the pool sizing — see SyclBackend.hpp.
    sycl_backend::set_twophase_budget(q, cfg.twophase_budget_bytes);
    if (cfg.k < 18 || cfg.k > 32 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 32]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }
    if (pool.k != cfg.k || pool.strength != cfg.strength
        || pool.testnet != cfg.testnet)
    {
        throw std::runtime_error(
            "GpuBufferPool was sized for different (k, strength, testnet)");
    }
    if (pinned_index < 0 || pinned_index >= GpuBufferPool::kNumPinnedBuffers) {
        throw std::runtime_error(
            "pinned_index must be in [0, GpuBufferPool::kNumPinnedBuffers)");
    }

    uint64_t const total_xs = pool.total_xs;
    uint64_t const cap      = pool.cap;

    constexpr int kThreads = 256;
    auto blocks = [&](uint64_t n) {
        return unsigned((n + kThreads - 1) / kThreads);
    };

    // ---- pool aliases ----
    // d_pair_a carries the "current phase match output": T1, then T2, then T3.
    // d_pair_b carries the "current phase sort output": sorted T1, sorted T2,
    // then final uint64_t fragments. Each subsequent phase's output overwrites
    // the previous (consumed) contents in the same slot.
    XsCandidateGpu* d_xs             = static_cast<XsCandidateGpu*>(pool.d_storage);
    // d_pair_a-derived aliases (d_t1_meta, d_t1_mi, d_t2_meta, d_t2_mi,
    // d_t2_xbits, d_t3) are NOT declared here. They're declared inside
    // the Xs phase block below, right after pool.ensure_pair_a()
    // performs the lazy malloc_device for d_pair_a. Deferring that
    // alloc until after Xs gen has been submitted to the queue lets
    // the ~400-500 ms CPU-side malloc_device overlap with Xs's
    // ~750 ms GPU execution — saves ~400-500 ms off first-plot wall;
    // batch plots 2+ hit ensure_pair_a's cached-pointer fast path
    // so the alloc cost is paid exactly once per pool.
    //
    // d_pair_b-derived aliases stay up here because d_pair_b is
    // eager-allocated by the pool ctor: Xs gen needs it as scratch
    // from the start of the pipeline.
    uint64_t*       d_t1_meta_sorted  = static_cast<uint64_t*>      (pool.d_pair_b);
    uint64_t*       d_t2_meta_sorted  = static_cast<uint64_t*>      (pool.d_pair_b);
    uint32_t*       d_t2_xbits_sorted = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(pool.d_pair_b) + pool.cap * sizeof(uint64_t));
    uint64_t*       d_frags_out       = static_cast<uint64_t*>      (pool.d_pair_b);

    uint64_t*       d_count        = pool.d_counter;
    // Xs phase needs ~3.22 GB scratch at k=28 in split-keys_a mode
    // (3 × total_xs × u32 + cub); d_pair_b is idle through the whole
    // Xs phase (not touched until T1 sort permute writes to it), so
    // we alias it rather than allocating separately.
    //
    // Split-keys_a: the Xs sort's keys_a (total_xs · u32 = 1 GiB at
    // k=28) lives in d_storage's tail — bytes [total_xs·8, storage_bytes)
    // which is idle during Xs gen+sort. The final pack phase writes
    // d_storage[0..total_xs·8) only, leaving keys_a's memory region
    // undisturbed (and its contents unread after the sort anyway, so
    // the overlap on T1/T2/T3-sort aliases in d_storage after pack is
    // a pure write-without-read of stale bytes). Saves ~1 GiB off the
    // pair_b xs-scratch region — see GpuBufferPool.cpp for sizing.
    void* const d_xs_split_keys_a = static_cast<uint8_t*>(pool.d_storage)
                                    + pool.total_xs * sizeof(XsCandidateGpu);
    void*           d_xs_temp      = pool.d_pair_b;
    void*           d_sort_scratch = pool.d_sort_scratch;
    // Lazy pinned-host alloc: skips ~600 ms × (kNumPinnedBuffers-1)
    // on single-plot runs (only slot 0 gets allocated). See
    // GpuBufferPool::ensure_pinned header comment for rationale.
    uint64_t*       h_pinned_t3    = pool.ensure_pinned(pinned_index);
    // T1/T2/T3 match kernels report 0 scratch bytes, but some CUDA paths
    // reject a nullptr d_temp_storage with cudaErrorInvalidArgument even
    // when bytes==0. Point them at d_sort_scratch (idle during match) to
    // give the kernel a valid non-null handle.
    void*           d_match_temp   = pool.d_sort_scratch;

    // Sort key/val arrays alias d_storage. Safe because Xs is fully consumed
    // by T1 match (stream-synchronised) before we enter T1 sort.
    //
    // Only three slots live here — keys_out, vals_in, vals_out. The
    // sort's keys_input is always the SoA match-info stream from
    // d_pair_a (d_t1_mi / d_t2_mi), so the fourth slot that would
    // have hosted "d_keys_in" is neither allocated nor used. See
    // GpuBufferPool.cpp for the matching storage_bytes shrink.
    auto     storage_u32 = static_cast<uint32_t*>(pool.d_storage);
    uint32_t* d_keys_out = storage_u32 + 0 * cap;
    uint32_t* d_vals_in  = storage_u32 + 1 * cap;
    uint32_t* d_vals_out = storage_u32 + 2 * cap;

    // ---- per-phase wall-time profiling ----
    // Enabled when either cfg.profile is set (xchplot2 -P / --profile) or
    // POS2GPU_PHASE_TIMING=1 is in the env. Each phase's wall is measured
    // around q.wait()s so launches actually drain to the device before the
    // next start sample — adds a sync point but gives an honest breakdown.
    // When disabled, begin/end/report are early-out and add ~zero cost.
    bool const phase_timing = cfg.profile || [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING");
        return v && v[0] == '1';
    }();
    using phase_clock = std::chrono::steady_clock;
    std::vector<std::pair<char const*, phase_clock::time_point>> phase_starts;
    std::vector<std::pair<char const*, double>>                  phase_records;
    // XCHPLOT2_HOST_GUARD: track the current phase label whether or not
    // phase timing is on, so a redzone report names the phase that broke
    // the buffer and not just the buffer. See HostGuard.hpp.
    char const* guard_phase = "(pipeline start)";
    auto begin_phase = [&](char const* label) -> int {
        if (pos2gpu::host_guard_on()) guard_phase = label;
        if (!phase_timing) return -1;
        q.wait();
        phase_starts.emplace_back(label, phase_clock::now());
        return static_cast<int>(phase_starts.size() - 1);
    };
    auto end_phase = [&](int idx) {
        // Drain before checking: a kernel still in flight has not done its
        // damage yet, and a guard that samples too early reads clean.
        if (pos2gpu::host_guard_on()) {
            q.wait();
            pos2gpu::host_guard_check(guard_phase);
        }
        if (idx < 0) return;
        q.wait();
        auto const t1 = phase_clock::now();
        auto const& [name, t0] = phase_starts[idx];
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        phase_records.emplace_back(name, ms);
    };
    auto report_phases = [&]() {
        if (!phase_timing || phase_records.empty()) return;
        double total = 0.0;
        for (auto const& [_n, ms] : phase_records) total += ms;
        std::fprintf(stderr, "[phase-timing]");
        for (auto const& [name, ms] : phase_records) {
            std::fprintf(stderr, " %s=%.1fms(%.0f%%)",
                name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        }
        std::fprintf(stderr, " total=%.1fms\n", total);
    };

    // ---------- Phase Xs ----------
    size_t xs_temp_bytes = 0;
    launch_construct_xs(cfg.plot_id.data(), cfg.k, cfg.testnet,
                              nullptr, nullptr, &xs_temp_bytes, q,
                              d_xs_split_keys_a);
    int p_xs = begin_phase("Xs gen+sort");
    // Xs phase events stubbed in slice 17b — pass nullptr for the (no-op)
    // profiling event slots. The launch_construct_xs_profiled signature still
    // accepts cudaEvent_t for API compatibility but ignores the values.
    launch_construct_xs_profiled(cfg.plot_id.data(), cfg.k, cfg.testnet,
                                       d_xs, d_xs_temp, &xs_temp_bytes,
                                       nullptr, nullptr, q,
                                       d_xs_split_keys_a);
    // Overlap d_pair_a's lazy malloc_device (~400-500 ms for 4.36 GB at
    // k=28) with Xs gen's GPU execution. In production
    // (POS2GPU_PHASE_TIMING unset), launch_construct_xs_profiled returns
    // immediately with the kernel in-flight on the queue; this CPU-side
    // alloc then runs in parallel and its wall is hidden behind Xs's
    // ~750 ms GPU work. In phase_timing mode xs-timing's internal
    // q.waits serialise Xs first, then this alloc pays full wall — a
    // diagnostic-mode trade-off.
    void* const d_pair_a_raw = pool.ensure_pair_a();
    end_phase(p_xs);

    // d_pair_a-derived aliases, now that the lazy alloc has resolved.
    // Same layout as the old eager version — just computed from the
    // local d_pair_a_raw instead of pool.d_pair_a so there's no
    // confusion about when the pointer became valid.
    //
    // T1 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B) then mi[cap] (cap·4 B). Total cap·12 B, fits in d_pair_a's
    // cap·16 B budget.
    uint64_t*     d_t1_meta = static_cast<uint64_t*>(d_pair_a_raw);
    uint32_t*     d_t1_mi   = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * sizeof(uint64_t));
    // T2 match output is SoA, carved out of d_pair_a. Layout: meta[cap]
    // (cap·8 B), then mi[cap] (cap·4 B), then xbits[cap] (cap·4 B). Total
    // cap·16 B, matching d_pair_a's size.
    uint64_t*     d_t2_meta  = static_cast<uint64_t*>(d_pair_a_raw);
    uint32_t*     d_t2_mi    = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * sizeof(uint64_t));
    uint32_t*     d_t2_xbits = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(d_pair_a_raw) + pool.cap * (sizeof(uint64_t) + sizeof(uint32_t)));
    T3PairingGpu* d_t3       = static_cast<T3PairingGpu*>(d_pair_a_raw);

    // ---------- Phase T1 ----------
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_count, cap,
                          nullptr, &t1_temp_bytes, q);
    // The queue is out-of-order: wait so the counter zeroing can't be
    // scheduled after the match kernel that increments it.
    q.memset(d_count, 0, sizeof(uint64_t)).wait();
    int p_t1 = begin_phase("T1 match");
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1_meta, d_t1_mi, d_count, cap,
                          d_match_temp, &t1_temp_bytes, q);
    end_phase(p_t1);

    // No explicit sync: the next cudaMemcpy (non-async, default stream)
    // implicitly drains prior stream work before the host reads t1_count.
    uint64_t t1_count = 0;
    q.memcpy(&t1_count, d_count, sizeof(uint64_t)).wait();
    if (t1_count > cap) throw std::runtime_error("T1 overflow");
    validate_t1_count(t1_count, cfg.k);


    // Sort T1 by match_info (low k bits). d_storage is now repurposed
    // as (keys_in, keys_out, vals_in, vals_out), Xs having been fully
    // consumed by T1 match above. T1 match emits match_info in a SoA
    // stream (d_t1_mi), so we feed that directly to CUB as the sort key
    // input rather than extracting from a packed struct.
    int p_t1_sort = begin_phase("T1 sort");
    {
        launch_init_u32_identity(d_vals_in, t1_count, q);
        size_t sort_bytes = pool.sort_scratch_bytes;
        launch_sort_pairs_u32_u32(
            d_sort_scratch, sort_bytes,
            d_t1_mi, d_keys_out, d_vals_in, d_vals_out,
            t1_count, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);

        launch_gather_u64(d_t1_meta, d_vals_out, d_t1_meta_sorted, t1_count, q);
    }
    end_phase(p_t1_sort);

    // ---------- Phase T2 ----------
    // Sorted T1 = (d_t1_meta_sorted: uint64 meta, d_keys_out: uint32 match_info).
    // No AoS struct anymore — saves 33 % of sorted-T1 bandwidth on both the
    // permute write and the match-kernel hot path.
    auto t2p = make_t2_params(cfg.k, cfg.strength);
    size_t t2_temp_bytes = 0;
    launch_t2_match(cfg.plot_id.data(), t2p, nullptr, nullptr, t1_count,
                          nullptr, nullptr, nullptr, d_count, cap,
                          nullptr, &t2_temp_bytes, q);
    q.memset(d_count, 0, sizeof(uint64_t)).wait();
    int p_t2 = begin_phase("T2 match");
    launch_t2_match(cfg.plot_id.data(), t2p, d_t1_meta_sorted, d_keys_out, t1_count,
                          d_t2_meta, d_t2_mi, d_t2_xbits, d_count, cap,
                          d_match_temp, &t2_temp_bytes, q);
    end_phase(p_t2);

    uint64_t t2_count = 0;
    q.memcpy(&t2_count, d_count, sizeof(uint64_t)).wait();
    if (t2_count > cap) throw std::runtime_error("T2 overflow");

    int p_t2_sort = begin_phase("T2 sort");
    {
        // T2 match emitted match_info as a SoA stream (d_t2_mi) — feed
        // it straight into CUB as the sort key input rather than
        // re-extracting from a packed struct. vals_in just needs a
        // 0..n-1 identity fill.
        launch_init_u32_identity(d_vals_in, t2_count, q);
        size_t sort_bytes = pool.sort_scratch_bytes;
        launch_sort_pairs_u32_u32(
            d_sort_scratch, sort_bytes,
            d_t2_mi, d_keys_out, d_vals_in, d_vals_out,
            t2_count, 0, cfg.k, q);

        launch_permute_t2(d_t2_meta, d_t2_xbits, d_vals_out,
                          d_t2_meta_sorted, d_t2_xbits_sorted, t2_count, q);
    }
    end_phase(p_t2_sort);

    // ---------- Phase T3 ----------
    // d_keys_out now holds the T2 sorted match_info (T1's was overwritten by
    // the T2 sort above) — pass as the slim stream for binary search in T3.
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          nullptr, t2_count,
                          d_t3, d_count, cap,
                          nullptr, &t3_temp_bytes, q);
    q.memset(d_count, 0, sizeof(uint64_t)).wait();
    int p_t3 = begin_phase("T3 match + Feistel");
    launch_t3_match(cfg.plot_id.data(), t3p,
                          d_t2_meta_sorted, d_t2_xbits_sorted,
                          d_keys_out, t2_count,
                          d_t3, d_count, cap,
                          d_match_temp, &t3_temp_bytes, q);
    end_phase(p_t3);

    uint64_t t3_count = 0;
    q.memcpy(&t3_count, d_count, sizeof(uint64_t)).wait();
    if (t3_count > cap) throw std::runtime_error("T3 overflow");

    // Sort T3 by proof_fragment (low 2k bits). T3PairingGpu is just a
    // uint64_t, so reinterpret the d_pair_a slot directly.
    uint64_t* d_frags_in = reinterpret_cast<uint64_t*>(d_t3);
    int p_t3_sort = begin_phase("T3 sort");
    {
        size_t sort_bytes = pool.sort_scratch_bytes;
        launch_sort_keys_u64(
            d_sort_scratch, sort_bytes,
            d_frags_in, d_frags_out,
            t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
    }
    end_phase(p_t3_sort);

    // ---------- D2H ----------
    int p_d2h = begin_phase("D2H copy T3 fragments (pinned)");
    GpuPipelineResult result;
    result.t1_count = t1_count;
    result.t2_count = t2_count;
    result.t3_count = t3_count;

    if (t3_count > 0) {
        q.memcpy(h_pinned_t3, d_frags_out, sizeof(uint64_t) * t3_count);
        q.wait();
    }
    end_phase(p_d2h);

    if (t3_count > 0) {
        // Borrow: caller (batch producer) promises to finish consuming this
        // pinned slot before reusing it for another plot.
        result.external_fragments_ptr   = h_pinned_t3;
        result.external_fragments_count = t3_count;
    }

    // Xs gen / sort per-phase timings stubbed in slice 17b — see profiling
    // notes above.

    // Release d_pair_a so it isn't held between plots in a batch run.
    // At ~5 ms/alloc on amdgcn (sycl::malloc_device effectively just
    // reserves virtual address space), the per-plot realloc cost is
    // below noise, but freeing 4.36 GB during the inter-plot gap means
    // the pool path is viable on cards with ~7-8 GiB free that would
    // otherwise hit InsufficientVramError and fall back to streaming.
    // The final q.wait() inside the D2H block above has already drained
    // T3 sort so the buffer is safe to free.
    pool.release_pair_a();

    report_phases();
    return result;
}

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg)
{
    // Explicit override for callers that want the streaming path without
    // having to rebuild anything. Handy for testing and for users who know
    // their hardware won't fit the pool.
    if (char const* env = std::getenv("XCHPLOT2_STREAMING");
        env && env[0] == '1')
    {
        return run_gpu_pipeline_streaming(cfg);
    }

    // Default: build a transient pool and run through it. Pays the full
    // per-call allocator overhead (~2.4 s for k=28) — batch callers should
    // construct a pool once and reuse it via the 3-arg overload.
    //
    // On insufficient device VRAM the pool ctor throws
    // InsufficientVramError; catch it specifically and fall back to
    // streaming so users on small-VRAM cards get a working plot with no
    // flags. Other CUDA errors propagate.
    try {
        GpuBufferPool pool(cfg.k, cfg.strength, cfg.testnet);
        GpuPipelineResult r = run_gpu_pipeline(cfg, pool, /*pinned_index=*/0);
        // Pool (and its pinned buffer) is about to be destroyed, so
        // materialise a self-contained copy before returning.
        if (r.external_fragments_ptr && r.external_fragments_count > 0) {
            r.t3_fragments_storage.resize(r.external_fragments_count);
            std::memcpy(r.t3_fragments_storage.data(),
                        r.external_fragments_ptr,
                        sizeof(uint64_t) * r.external_fragments_count);
        }
        r.external_fragments_ptr   = nullptr;
        r.external_fragments_count = 0;
        return r;
    } catch (InsufficientVramError const& e) {
        std::fprintf(stderr,
            "[xchplot2] pool needs %.2f GiB, only %.2f GiB free of "
            "%.2f GiB — falling back to streaming pipeline\n",
            e.required_bytes / double(1ULL << 30),
            e.free_bytes     / double(1ULL << 30),
            e.total_bytes    / double(1ULL << 30));
        return run_gpu_pipeline_streaming(cfg);
    }
}

// =====================================================================
// Streaming pipeline — per-phase cudaMalloc / cudaFree, no persistent pool.
//
// Only buffers required for the CURRENT and NEXT phase are resident at any
// point. Tiled sorts + SoA emission drive the peak down under 8 GB at
// k=28, so an 8 GB card can run this path.
//
// The implementation body below accepts an optional caller-provided
// pinned D2H buffer — used by BatchPlotter to amortise cudaMallocHost
// across plots and double-buffer the D2H with the FSE consumer.
//
// Exception safety: on throw mid-pipeline we currently leak the
// still-live device allocations. The CLI terminates on exception anyway,
// so the OS reclaims the context. If we later embed this in a long-lived
// process we can add RAII owners without changing the public surface.
// =====================================================================
namespace { // anon: shared impl, not part of the public API.

GpuPipelineResult run_gpu_pipeline_streaming_impl(
    GpuPipelineConfig const& cfg,
    uint64_t* pinned_dst,                       // nullable
    size_t    pinned_capacity,                  // count, not bytes; ignored if pinned_dst null
    StreamingPinnedScratch const& scratch);     // any field nullptr → per-plot malloc_host fallback

} // namespace

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg)
{

    sycl::queue& q = sycl_backend::queue();
    StreamingPinnedScratch scratch{};
    // Honor XCHPLOT2_STREAMING_TIER in the no-arg path so test mode and
    // standalone callers can exercise non-default tiers without going
    // through BatchPlotter. Mirrors BatchPlotter's tier selection.
    if (char const* tier_env = std::getenv("XCHPLOT2_STREAMING_TIER")) {
        std::string t = tier_env;
        if (t == "plain") {
            scratch.plain_mode = true;
        } else if (t == "compact") {
            // compact = default (no flags set). Explicitly leave both off.
        } else if (t == "minimal") {
            scratch.t2_tile_count     = 8;
            scratch.gather_tile_count = 4;
        } else if (t == "tiny") {
            scratch.t2_tile_count     = 8;
            scratch.gather_tile_count = 4;
            scratch.tiny_mode         = true;
        } else if (t == "pinned") {
            // Mirrors BatchPlotter: pinned = tiny's parks + the
            // streaming-partition T1 flow.
            scratch.t2_tile_count     = 8;
            scratch.gather_tile_count = 4;
            scratch.tiny_mode         = true;
            scratch.pinned_mode       = true;
        }
        // Unrecognized values fall through to default (compact).
    }
    return run_gpu_pipeline_streaming_impl(cfg, /*pinned_dst=*/nullptr,
                                                /*pinned_capacity=*/0,
                                                scratch);
}

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg,
                                             uint64_t* pinned_dst,
                                             size_t    pinned_capacity)
{
    if (!pinned_dst || pinned_capacity == 0) {
        throw std::runtime_error(
            "run_gpu_pipeline_streaming(cfg, pinned, cap): pinned buffer must be non-null");
    }
    return run_gpu_pipeline_streaming_impl(cfg, pinned_dst, pinned_capacity,
                                           StreamingPinnedScratch{});
}

GpuPipelineResult run_gpu_pipeline_streaming(GpuPipelineConfig const& cfg,
                                             uint64_t* pinned_dst,
                                             size_t    pinned_capacity,
                                             StreamingPinnedScratch const& scratch)
{
    if (!pinned_dst || pinned_capacity == 0) {
        throw std::runtime_error(
            "run_gpu_pipeline_streaming(cfg, pinned, cap, scratch): pinned buffer must be non-null");
    }
    return run_gpu_pipeline_streaming_impl(cfg, pinned_dst, pinned_capacity, scratch);
}

namespace {

GpuPipelineResult run_gpu_pipeline_streaming_impl(
    GpuPipelineConfig const& cfg,
    uint64_t* pinned_dst,
    size_t    pinned_capacity,
    StreamingPinnedScratch const& scratch)
{

    sycl::queue& q = sycl_backend::queue();
    // Grant the T2/T3 two-phase match its VRAM budget for this queue before
    // any match runs. The streaming tiers are, by definition, the ones with
    // no VRAM to spare, and the scratch is not part of any tier's peak
    // model — see SyclBackend.hpp.
    sycl_backend::set_twophase_budget(q, scratch.twophase_budget_bytes);
    if (cfg.k < 18 || cfg.k > 32 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 32]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }

    int const num_section_bits = (cfg.k < 28) ? 2 : (cfg.k - 26);
    uint64_t const total_xs = 1ULL << cfg.k;
    uint64_t const cap =
        max_pairs_per_section(cfg.k, num_section_bits) *
        (1ULL << num_section_bits);

    constexpr int kThreads = 256;
    auto blocks = [&](uint64_t n) {
        return unsigned((n + kThreads - 1) / kThreads);
    };

    StreamingStats stats;
    s_init_from_env(stats);

    // ---- per-phase wall-time profiling ----
    // Identical shape to the pool path (run_gpu_pipeline above); the
    // [phase-timing] output format matches so POS2GPU_PHASE_TIMING=1 now
    // produces the same breakdown whether the pipeline runs pool or
    // falls back to streaming. On 12 GiB cards at k=28 (where pool
    // overflows and we always streams) this is the only way to see
    // which phase is eating the wall.
    bool const phase_timing = cfg.profile || [] {
        char const* v = std::getenv("POS2GPU_PHASE_TIMING");
        return v && v[0] == '1';
    }();
    using phase_clock = std::chrono::steady_clock;
    std::vector<std::pair<char const*, phase_clock::time_point>> phase_starts;
    std::vector<std::pair<char const*, double>>                  phase_records;
    // XCHPLOT2_HOST_GUARD: track the current phase label whether or not
    // phase timing is on, so a redzone report names the phase that broke
    // the buffer and not just the buffer. See HostGuard.hpp.
    char const* guard_phase = "(pipeline start)";
    auto begin_phase = [&](char const* label) -> int {
        if (pos2gpu::host_guard_on()) guard_phase = label;
        if (!phase_timing) return -1;
        q.wait();
        phase_starts.emplace_back(label, phase_clock::now());
        return static_cast<int>(phase_starts.size() - 1);
    };
    auto end_phase = [&](int idx) {
        // Drain before checking: a kernel still in flight has not done its
        // damage yet, and a guard that samples too early reads clean.
        if (pos2gpu::host_guard_on()) {
            q.wait();
            pos2gpu::host_guard_check(guard_phase);
        }
        if (idx < 0) return;
        q.wait();
        auto const t1 = phase_clock::now();
        auto const& [name, t0] = phase_starts[idx];
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        phase_records.emplace_back(name, ms);
    };
    auto report_phases = [&]() {
        if (!phase_timing || phase_records.empty()) return;
        double total = 0.0;
        for (auto const& [_n, ms] : phase_records) total += ms;
        std::fprintf(stderr, "[phase-timing]");
        for (auto const& [name, ms] : phase_records) {
            std::fprintf(stderr, " %s=%.1fms(%.0f%%)",
                name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        }
        std::fprintf(stderr, " total=%.1fms\n", total);
    };

    // --- pipeline-wide tiny allocations ---
    // d_counter: per-phase uint64 count output (reused).
    // The match kernels each need their own temp-storage buffer sized via
    // their size query; we allocate it per-phase rather than globally so
    // that the peak VRAM is the phase's alone.
    stats.phase = "init";
    uint64_t* d_counter = nullptr;
    s_malloc(stats, d_counter, sizeof(uint64_t), "d_counter");

    // Phase 2 (pipeline-parallel) function-scope state. Originally
    // declared inside the Xs / T1 / T2 phase blocks; lifted here so
    // the start_at_t3_match entry can populate them and goto past the
    // first half. The first-half phase code below assigns to these as
    // it runs; the second-half (T3 match → end) code reads them.
    bool t1_match_sliced = !scratch.plain_mode && scratch.gather_tile_count > 1;
    bool h_meta_owned    = (!scratch.plain_mode && scratch.h_meta == nullptr);
    bool h_keys_owned    = (!scratch.plain_mode && scratch.h_keys_merged == nullptr);
    bool h_xbits_owned   = false;
    // h_t2_meta_owned tracks ownership of the (possibly distinct) T2
    // meta buffer. In tiny mode this is decoupled from h_meta_owned to
    // prevent the T1-input / T2-output buffer-reuse race; see
    // feedback_h_meta_buffer_reuse_bug memo.
    bool h_t2_meta_owned = false;

    uint64_t t1_count = 0;
    uint64_t t2_count = 0;

    uint64_t* h_t2_meta        = nullptr;
    uint32_t* h_t2_xbits       = nullptr;
    uint32_t* h_t2_keys_merged = nullptr;

    // Phase 2.2 split: lifted from inside the first-half scope so
    // start_at_t2_match can populate them and skip the Xs / T1 phases,
    // and stop_after_t1_sort can hand them back to the caller.
    uint64_t* h_t1_meta        = nullptr;
    uint32_t* h_t1_keys_merged = nullptr;
    // Host-RAM disk-offload: the shared background I/O engine (created
    // lazily on the first spilled table) plus one SpillBuffer per spilled
    // table. While a table's SpillBuffer is set its host pointer stays
    // null and every access routes through the SpillBuffer. Declared here
    // (function scope) so the engine outlives every SpillBuffer and the
    // h_t3 spill survives from T3 match into T3 sort. See SpillEngine.
    std::unique_ptr<SpillEngine> spill_engine;
    std::unique_ptr<SpillBuffer> t1_meta_spill;   // h_t1_meta (~2 GiB, tiny)
    std::unique_ptr<SpillBuffer> t3_spill;        // h_t3 (~2 GiB, compact/minimal/tiny)
    std::unique_ptr<SpillBuffer> t2_xbits_spill;  // h_t2_xbits (~1 GiB, compact / minimal)
    // Lazily create the one shared engine the first time any table spills.
    auto ensure_spill_engine = [&]() -> SpillEngine& {
        if (!spill_engine) spill_engine = std::make_unique<SpillEngine>(stats, q);
        return *spill_engine;
    };

    // Phase 2.2 split: device buffers that cross the Xs+T1 / T2-match
    // scope boundary. Lifted from inner phase blocks so the if-skip
    // wrapping Xs+T1 (when start_at_t2_match is set) doesn't break
    // their references in T2 match / T2 sort code below. All six are
    // trivially init'd to nullptr; non-skip paths assign as before.
    uint32_t* d_keys_out        = nullptr;
    uint32_t* d_vals_in         = nullptr;
    uint32_t* d_vals_out        = nullptr;
    uint32_t* h_keys            = nullptr;
    uint32_t* h_vals            = nullptr;
    uint32_t* d_t1_keys_merged  = nullptr;
    uint64_t* d_t1_meta_sorted  = nullptr;

    uint32_t* d_t2_keys_merged  = nullptr;
    uint64_t* d_t2_meta_sorted  = nullptr;
    uint32_t* d_t2_xbits_sorted = nullptr;
    uint64_t* d_t2_meta         = nullptr;
    uint32_t* d_t2_mi           = nullptr;
    uint32_t* d_t2_xbits        = nullptr;
    void*     d_sort_scratch    = nullptr;

    // Phase 1.4c — Pinned tier only. h_xs survives the Xs phase
    // and is consumed by T1 match's per-section-pair tile flow.
    // h_xs_section_starts: (num_sections+1) prefix-sum offsets giving
    // each section's begin position in h_xs. For non-Pinned paths,
    // both stay null / empty.
    XsCandidateGpu*       h_xs_pinned          = nullptr;
    std::vector<uint64_t> h_xs_section_starts;

    if (scratch.start_at_t3_match) {
        // Minimal and tiny modes both work for the second-half handoff:
        // - Tiny: T3 match reads all per-section streams (meta+xbits+
        //   mi) from host pinned. No device-side T2 buffers needed.
        // - Minimal: T3 match reads meta from host pinned slices but
        //   xbits + mi from full-cap device buffers. We rehydrate
        //   those from the caller-provided host buffers below.
        // Plain / compact don't support the split — they consume T2
        // outputs from device buffers that the first half doesn't
        // surface to host.
        if (scratch.gather_tile_count <= 1) {
            throw std::runtime_error(
                "start_at_t3_match requires minimal or tiny tier "
                "(gather_tile_count > 1); plain/compact don't surface "
                "T2 sorted outputs to host pinned.");
        }
        // Caller must provide the T2 boundary buffers. Prefer the
        // dedicated h_t2_meta field; fall back to h_meta for callers
        // that pre-date the buffer-reuse fix.
        uint64_t* const t2_meta_in = scratch.h_t2_meta
            ? scratch.h_t2_meta
            : scratch.h_meta;
        if (!t2_meta_in || !scratch.h_t2_xbits || !scratch.h_keys_merged) {
            throw std::runtime_error(
                "start_at_t3_match requires h_t2_meta (or h_meta), "
                "h_t2_xbits, and h_keys_merged populated by the caller");
        }
        t1_count         = scratch.t1_count_in;
        t2_count         = scratch.t2_count_in;
        h_t2_meta        = t2_meta_in;
        h_t2_xbits       = scratch.h_t2_xbits;
        h_t2_keys_merged = scratch.h_keys_merged;
        h_meta_owned     = false;
        h_t2_meta_owned  = false;
        h_xbits_owned    = false;
        h_keys_owned     = false;

        // Minimal mode: T3 match reads d_t2_xbits_sorted on device
        // for its full-cap random-access reads (it only slices the
        // meta side). Allocate and rehydrate from h_t2_xbits before
        // jumping to T3 match. d_t2_keys_merged is rehydrated by the
        // existing T3 match prep-stage code at line ~2360.
        if (!scratch.tiny_mode) {
            s_malloc(stats, d_t2_xbits_sorted, cap * sizeof(uint32_t),
                     "d_t2_xbits_sorted(start_at_t3)");
            q.memcpy(d_t2_xbits_sorted, h_t2_xbits,
                     t2_count * sizeof(uint32_t)).wait();
        }
        goto t3_match_entry;
    }

    // Phase 2.2 split BEFORE T2 match: when start_at_t2_match is set,
    // skip the Xs / T1 work and feed T2 match from the caller's
    // h_meta + h_keys_merged. Variables that cross the Xs+T1 / T2 match
    // boundary (h_t1_meta, h_t1_keys_merged, d_t1_meta_sorted,
    // d_t1_keys_merged, d_keys_out, d_vals_in, d_vals_out, h_keys,
    // h_vals) are lifted to function top so the if-skip below can
    // bypass their declarations without breaking T2 match references.
    if (scratch.start_at_t2_match) {
        if (scratch.start_at_t3_match) {
            throw std::runtime_error(
                "start_at_t2_match and start_at_t3_match are mutually exclusive");
        }
        if (scratch.gather_tile_count <= 1) {
            throw std::runtime_error(
                "start_at_t2_match requires minimal or tiny tier "
                "(gather_tile_count > 1); plain/compact path doesn't surface "
                "T1 sorted outputs to host pinned in a state the caller can "
                "hand off.");
        }
        if (!scratch.h_meta || !scratch.h_keys_merged) {
            throw std::runtime_error(
                "start_at_t2_match requires h_meta and h_keys_merged "
                "populated by the caller with sorted T1 outputs");
        }
        t1_count         = scratch.t1_count_in;
        h_t1_meta        = scratch.h_meta;
        h_t1_keys_merged = scratch.h_keys_merged;
        h_meta_owned     = false;
        h_keys_owned     = false;

        // Minimal mode: T2 match consumes d_t1_meta_sorted and
        // d_t1_keys_merged on device (full-cap), populated by T1 sort
        // in the normal flow. Skipping Xs+T1 means we must rehydrate
        // them here from the caller-provided host pinned buffers.
        // Tiny mode reads h_t1_meta + h_t1_keys_merged directly per
        // section inside T2 match and doesn't need the device buffers.
        if (!scratch.tiny_mode) {
            s_malloc(stats, d_t1_meta_sorted, cap * sizeof(uint64_t),
                     "d_t1_meta_sorted(start_at_t2)");
            q.memcpy(d_t1_meta_sorted, scratch.h_meta,
                     t1_count * sizeof(uint64_t)).wait();
            s_malloc(stats, d_t1_keys_merged, cap * sizeof(uint32_t),
                     "d_t1_keys_merged(start_at_t2)");
            q.memcpy(d_t1_keys_merged, scratch.h_keys_merged,
                     t1_count * sizeof(uint32_t)).wait();
        }
    }

    // Open a scope for the entire first-half (Xs+T1+T2) phase code so
    // the start_at_t3_match goto above can legally bypass any inner
    // declarations — only function-top state is in scope at the label.
    {
    // Phase 2.2 split: skip Xs / T1 / T1 sort when start_at_t2_match
    // is set. Closing brace lands just before T2 match phase, where
    // stop_after_t1_sort also exits if requested.
    if (!scratch.start_at_t2_match) {
    // ---------- Phase Xs (stage 4e: inlined gen+sort+pack) ----------
    // launch_construct_xs lumps keys_a/keys_b/vals_a/vals_b into a single
    // d_xs_temp blob (~4 GB at k=28). keys_a+vals_a are dead after the
    // CUB sort but can't be freed because they're interior slices of a
    // single allocation. Inline the three sub-kernels so we can:
    //   1. alloc cub_scratch + keys_a + vals_a
    //   2. gen fills keys_a, vals_a
    //   3. alloc keys_b + vals_b
    //   4. CUB sort keys_a/vals_a -> keys_b/vals_b; keys_a/vals_a now dead
    //   5. free cub_scratch + keys_a + vals_a       <- 2078 MB freed
    //   6. alloc d_xs
    //   7. pack keys_b/vals_b -> d_xs
    //   8. free keys_b + vals_b
    // Phase peak at k=28 drops from d_xs (2048) + d_xs_temp (4128) =
    // 6176 MB to max(sort 4126 MB, pack 4096 MB) = 4126 MB.
    stats.phase = "Xs";

    AesHashKeys const xs_keys = make_keys(cfg.plot_id.data());
    uint32_t    const xs_xor_const = cfg.testnet ? 0xA3B1C4D7u : 0u;

    XsCandidateGpu* d_xs = nullptr;
    uint32_t* d_xs_keys_b = nullptr;
    uint32_t* d_xs_vals_b = nullptr;

    bool const xs_sliced = !scratch.plain_mode && scratch.gather_tile_count > 1;

    if (!xs_sliced) {
        // Compact / plain — full-cap gen+sort+pack (4128 MB sort peak).
        size_t xs_cub_bytes = 0;
        launch_sort_pairs_u32_u32(
            nullptr, xs_cub_bytes,
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            total_xs, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);

        void*     d_xs_cub_scratch = nullptr;
        uint32_t* d_xs_keys_a      = nullptr;
        uint32_t* d_xs_vals_a      = nullptr;
        s_malloc(stats, d_xs_cub_scratch, xs_cub_bytes,                     "d_xs_cub");
        s_malloc(stats, d_xs_keys_a,      total_xs * sizeof(uint32_t),      "d_xs_keys_a");
        s_malloc(stats, d_xs_vals_a,      total_xs * sizeof(uint32_t),      "d_xs_vals_a");

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            // Sentinel-fill keys_a / vals_a head/mid/tail with 0xCD.
            uint64_t const off_mid  = total_xs / 2;
            uint64_t const off_tail = (total_xs >= 16ULL) ? total_xs - 16ULL : 0ULL;
            q.memset(d_xs_keys_a,            0xCD, 64).wait();
            q.memset(d_xs_keys_a + off_mid,  0xCD, 64).wait();
            q.memset(d_xs_keys_a + off_tail, 0xCD, 64).wait();
            q.memset(d_xs_vals_a,            0xCD, 64).wait();
            q.memset(d_xs_vals_a + off_mid,  0xCD, 64).wait();
            q.memset(d_xs_vals_a + off_tail, 0xCD, 64).wait();

            // Trivial-kernel sanity: writes 0xDEADBEEF to keys_a[0..16]
            // with no LDS / no captured struct / no AES. If this
            // produces 0xCDCDCDCD post-launch, AdaptiveCpp's HIP
            // submission path is producing no-op stubs for ANY kernel
            // — the problem is below our level. If it produces
            // 0xDEADBEEF, simple kernels work and the issue is
            // specific to the cooperative-LDS / AES kernel pattern.
            {
                uint32_t* p = d_xs_keys_a;
                q.parallel_for(
                    sycl::nd_range<1>{256, 256},
                    [=](sycl::nd_item<1> it) {
                        size_t idx = it.get_global_id(0);
                        if (idx < 16) p[idx] = 0xDEADBEEFu;
                    }).wait();
                uint32_t check[16] = {};
                q.memcpy(check, d_xs_keys_a, 16 * sizeof(uint32_t)).wait();
                bool const ok = (check[0] == 0xDEADBEEFu);
                std::fprintf(stderr,
                    "[t1-debug] trivial kernel test: %s  (keys_a[0]=0x%08x)\n",
                    ok ? "PASS — simple kernels can write"
                       : "FAIL — kernel writes are not landing",
                    check[0]);
                // Restore sentinel since the trivial kernel overwrote
                // the head region.
                q.memset(d_xs_keys_a, 0xCD, 64).wait();
            }

            // Dump d_aes_tables[0..16]. Standard AES T0[0] = 0xC66363A5.
            // If we see 0xBE / 0xCD here, the T-table USM buffer was
            // never populated by aes_tables_device's q.memcpy — kernels
            // would then read garbage and produce nothing useful.
            {
                uint32_t* d_tables = sycl_backend::aes_tables_device(q);
                uint32_t aes_check[16] = {};
                q.memcpy(aes_check, d_tables, 16 * sizeof(uint32_t)).wait();
                std::fprintf(stderr,
                    "[t1-debug] d_aes_tables[0..16] (T0[a] = (2S[a],S[a],S[a],3S[a]) packed LE; T0[0] = 0xa56363c6):\n");
                for (int i = 0; i < 16; ++i) {
                    std::fprintf(stderr, "  [%2d] 0x%08x\n", i, aes_check[i]);
                }
            }
        }

        int p_xs = begin_phase("Xs gen+sort");
        launch_xs_gen(xs_keys, d_xs_keys_a, d_xs_vals_a, total_xs,
                      cfg.k, xs_xor_const, q);

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sn = (total_xs < 16ULL) ? total_xs : 16ULL;
            uint64_t const off_mid  = total_xs / 2;
            uint64_t const off_tail = (total_xs >= 16ULL) ? total_xs - 16ULL : 0ULL;
            uint32_t ka_h[16] = {}, va_h[16] = {};
            uint32_t ka_m[16] = {}, va_m[16] = {};
            uint32_t ka_t[16] = {}, va_t[16] = {};
            q.memcpy(ka_h, d_xs_keys_a,            sn * sizeof(uint32_t)).wait();
            q.memcpy(va_h, d_xs_vals_a,            sn * sizeof(uint32_t)).wait();
            q.memcpy(ka_m, d_xs_keys_a + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(va_m, d_xs_vals_a + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(ka_t, d_xs_keys_a + off_tail, sn * sizeof(uint32_t)).wait();
            q.memcpy(va_t, d_xs_vals_a + off_tail, sn * sizeof(uint32_t)).wait();
            std::fprintf(stderr,
                "[t1-debug] post-xs_gen   total_xs=%llu (head idx=0, mid idx=%llu, tail idx=%llu):\n",
                (unsigned long long)total_xs,
                (unsigned long long)off_mid, (unsigned long long)off_tail);
            for (uint64_t i = 0; i < sn; ++i) {
                std::fprintf(stderr,
                    "  H[%2llu] ka=0x%08x va=0x%08x  M[%2llu] ka=0x%08x va=0x%08x  T[%2llu] ka=0x%08x va=0x%08x\n",
                    (unsigned long long)i,            ka_h[i], va_h[i],
                    (unsigned long long)(off_mid + i),  ka_m[i], va_m[i],
                    (unsigned long long)(off_tail + i), ka_t[i], va_t[i]);
            }
        }

        s_malloc(stats, d_xs_keys_b, total_xs * sizeof(uint32_t), "d_xs_keys_b");
        s_malloc(stats, d_xs_vals_b, total_xs * sizeof(uint32_t), "d_xs_vals_b");

        launch_sort_pairs_u32_u32(
            d_xs_cub_scratch, xs_cub_bytes,
            d_xs_keys_a, d_xs_keys_b,
            d_xs_vals_a, d_xs_vals_b,
            total_xs, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
        end_phase(p_xs);

        s_free(stats, d_xs_cub_scratch);
        s_free(stats, d_xs_keys_a);
        s_free(stats, d_xs_vals_a);

        s_malloc(stats, d_xs, total_xs * sizeof(XsCandidateGpu), "d_xs");

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sn = (total_xs < 16ULL) ? total_xs : 16ULL;
            uint64_t const off_mid  = total_xs / 2;
            uint64_t const off_tail = (total_xs >= 16ULL) ? total_xs - 16ULL : 0ULL;
            uint32_t kb_h[16] = {}, vb_h[16] = {};
            uint32_t kb_m[16] = {}, vb_m[16] = {};
            uint32_t kb_t[16] = {}, vb_t[16] = {};
            q.memcpy(kb_h, d_xs_keys_b,            sn * sizeof(uint32_t)).wait();
            q.memcpy(vb_h, d_xs_vals_b,            sn * sizeof(uint32_t)).wait();
            q.memcpy(kb_m, d_xs_keys_b + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(vb_m, d_xs_vals_b + off_mid,  sn * sizeof(uint32_t)).wait();
            q.memcpy(kb_t, d_xs_keys_b + off_tail, sn * sizeof(uint32_t)).wait();
            q.memcpy(vb_t, d_xs_vals_b + off_tail, sn * sizeof(uint32_t)).wait();
            std::fprintf(stderr,
                "[t1-debug] post-xs_sort  total_xs=%llu (head idx=0, mid idx=%llu, tail idx=%llu):\n",
                (unsigned long long)total_xs,
                (unsigned long long)off_mid, (unsigned long long)off_tail);
            for (uint64_t i = 0; i < sn; ++i) {
                std::fprintf(stderr,
                    "  H[%2llu] kb=0x%08x vb=0x%08x  M[%2llu] kb=0x%08x vb=0x%08x  T[%2llu] kb=0x%08x vb=0x%08x\n",
                    (unsigned long long)i,            kb_h[i], vb_h[i],
                    (unsigned long long)(off_mid + i),  kb_m[i], vb_m[i],
                    (unsigned long long)(off_tail + i), kb_t[i], vb_t[i]);
            }
        }

        int p_xs_pack = begin_phase("Xs pack");
        launch_xs_pack(d_xs_keys_b, d_xs_vals_b, d_xs, total_xs, q);
        end_phase(p_xs_pack);

        s_free(stats, d_xs_keys_b);
        s_free(stats, d_xs_vals_b);
    } else {
        // Sliced (minimal/tiny/pinned). Tile gen+sort in N=2 position
        // halves into cap/2 device buffers, D2H per tile to USM-host.
        //
        // From here, two sub-paths:
        //   - Minimal/Tiny: merge host-pinned tile outputs into device
        //     d_xs_keys_b + d_xs_vals_b (full cap). Pack in N=2 halves
        //     with D2H per tile to a host-pinned XsCandidateGpu
        //     accumulator. Drops sort peak from 4128 MB → 2056 MB and
        //     pack peak from 4096 MB → 3072 MB at k=28. (Existing.)
        //   - Pinned (Phase 1.4a+b): merge on host (CPU std::merge-style
        //     loop — the GPU merge kernel does per-thread binary search
        //     with random reads, pathological on USM-host source). Then
        //     pack reads/writes with all-host USM pointers — pack's
        //     access is sequential per thread so PCIe bursts are
        //     efficient. Eliminates d_xs_keys_b / d_xs_vals_b /
        //     d_xs_pack_tile from device entirely (~3 GB saving at
        //     k=28). The d_xs rehydrate at the bottom of this block
        //     still happens; eliminating it is Phase 1.4c.
        //
        // The Pinned sub-path is gated on scratch.tiny_mode below;
        // Minimal/Tiny stay on the existing flow unchanged.
        //
        // Phase 1.5d (after d_frags_out alias) — Tiny tier bumps tile
        // count from N=2 → N=4 to shrink each device tile buffer from
        // cap/2 × u32 (128 MB at k=26 / 512 MB at k=28) to cap/4 ×
        // u32 (64 MB at k=26 / 256 MB at k=28). With 4 buffers
        // (keys_a, vals_a, keys_b, vals_b) + cub scratch in flight,
        // saves 256 MB at k=26 / 1 GB at k=28 on the Xs gen+sort
        // phase peak. Minimal stays at N=2 to keep the
        // launch_merge_pairs_stable_2way_u32_u32 path unchanged
        // (Minimal's merge primitive is 2-way; an N-way GPU merge
        // would be a separate kernel rewrite). Tiny's merge+pack
        // already runs on CPU so N-way generalizes trivially.
        constexpr int kXsTinyTiles = 4;
        constexpr int kXsMinTiles  = 2;
        int const kXsTiles = scratch.tiny_mode ? kXsTinyTiles : kXsMinTiles;

        uint64_t xs_tile_offsets[kXsTinyTiles + 1];
        uint64_t const xs_tile_max =
            (total_xs + uint64_t(kXsTiles) - 1) / uint64_t(kXsTiles);
        xs_tile_offsets[0] = 0;
        for (int t = 0; t < kXsTiles; ++t) {
            xs_tile_offsets[t + 1] =
                std::min(xs_tile_offsets[t] + xs_tile_max, total_xs);
        }
        // Legacy two-half names — only used by the Minimal merge
        // primitive below (N=2 path).
        uint64_t const xs_tile_n0 =
            scratch.tiny_mode ? 0 : xs_tile_offsets[1];
        uint64_t const xs_tile_n1 =
            scratch.tiny_mode ? 0 : (total_xs - xs_tile_offsets[1]);

        size_t xs_cub_tile_bytes = 0;
        launch_sort_pairs_u32_u32(
            nullptr, xs_cub_tile_bytes,
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            xs_tile_max, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);

        void*     d_xs_cub_scratch  = nullptr;
        uint32_t* d_xs_keys_a_tile  = nullptr;
        uint32_t* d_xs_vals_a_tile  = nullptr;
        uint32_t* d_xs_keys_b_tile  = nullptr;
        uint32_t* d_xs_vals_b_tile  = nullptr;
        s_malloc(stats, d_xs_keys_a_tile, xs_tile_max * sizeof(uint32_t), "d_xs_keys_a_tile");
        s_malloc(stats, d_xs_vals_a_tile, xs_tile_max * sizeof(uint32_t), "d_xs_vals_a_tile");
        s_malloc(stats, d_xs_keys_b_tile, xs_tile_max * sizeof(uint32_t), "d_xs_keys_b_tile");
        s_malloc(stats, d_xs_vals_b_tile, xs_tile_max * sizeof(uint32_t), "d_xs_vals_b_tile");
        s_malloc(stats, d_xs_cub_scratch, xs_cub_tile_bytes,              "d_xs_cub");

        uint32_t* h_xs_keys = s_malloc_host<uint32_t>(
            stats, total_xs * sizeof(uint32_t), "h_xs_keys", q);
        uint32_t* h_xs_vals = s_malloc_host<uint32_t>(
            stats, total_xs * sizeof(uint32_t), "h_xs_vals", q);

        int p_xs = begin_phase("Xs gen+sort");
        auto run_tile = [&](uint64_t pos_begin, uint64_t pos_end, uint64_t out_offset) {
            uint64_t tile_n = pos_end - pos_begin;
            if (tile_n == 0) return;
            launch_xs_gen_range(
                xs_keys, d_xs_keys_a_tile, d_xs_vals_a_tile,
                pos_begin, pos_end, cfg.k, xs_xor_const, q);
            launch_sort_pairs_u32_u32(
                d_xs_cub_scratch, xs_cub_tile_bytes,
                d_xs_keys_a_tile, d_xs_keys_b_tile,
                d_xs_vals_a_tile, d_xs_vals_b_tile,
                tile_n, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
            q.memcpy(h_xs_keys + out_offset, d_xs_keys_b_tile,
                     tile_n * sizeof(uint32_t)).wait();
            q.memcpy(h_xs_vals + out_offset, d_xs_vals_b_tile,
                     tile_n * sizeof(uint32_t)).wait();
        };
        for (int t = 0; t < kXsTiles; ++t) {
            run_tile(xs_tile_offsets[t], xs_tile_offsets[t + 1],
                     xs_tile_offsets[t]);
        }
        end_phase(p_xs);

        s_free(stats, d_xs_cub_scratch);
        s_free(stats, d_xs_vals_b_tile);
        s_free(stats, d_xs_keys_b_tile);
        s_free(stats, d_xs_vals_a_tile);
        s_free(stats, d_xs_keys_a_tile);

        XsCandidateGpu* h_xs = nullptr;

        if (scratch.tiny_mode) {
            // Phase 1.4a + 1.4b — Pinned tier only.
            //
            // Both merge AND pack run on the CPU as a single fused
            // loop that writes directly to the host-pinned h_xs output.
            //
            // Why CPU rather than GPU-on-host-pointers: AdaptiveCpp's
            // CUDA backend doesn't issue burst PCIe transfers when a
            // SYCL kernel dereferences malloc_host pointers — each
            // thread's read becomes an individual transaction, ~1 µs
            // latency. Measured: kernel pack on all-USM-host pointers
            // at k=26 ran 4434 ms (vs 27 ms for the device-input
            // tiled-pack of the existing path). A CPU loop that
            // streams the same data hits ~30 GB/s memory bandwidth,
            // which works out to ~30 ms at k=26 / ~134 ms at k=28.
            //
            // Why fuse merge + pack: pack is trivially
            // `out[i] = {keys[i], vals[i]}` — combining it with the
            // merge's "pick lower of two heads" loop adds no work but
            // saves the intermediate h_xs_keys_merged + h_xs_vals_merged
            // arrays (~2 GB host at k=28).
            //
            // Net device-peak saving: skips d_xs_keys_b + d_xs_vals_b +
            // d_xs_pack_tile entirely (~3 GB device at k=28). The
            // d_xs rehydrate at the bottom of the block is unchanged —
            // that's Phase 1.4c's lever.
            h_xs = s_malloc_host<XsCandidateGpu>(
                stats, total_xs * sizeof(XsCandidateGpu), "h_xs(pinned)", q);

            // Phase 1.4c also needs section-start offsets so T1 match
            // can find each section's range in h_xs without re-scanning.
            // Sections are top num_section_bits of match_info; h_xs is
            // sorted by match_info so each section is contiguous.
            int      const xs_num_section_bits =
                (cfg.k < 28) ? 2 : (cfg.k - 26);
            uint32_t const xs_num_sections     = 1u << xs_num_section_bits;
            int      const xs_section_shift    = cfg.k - xs_num_section_bits;
            std::vector<uint64_t> section_count(xs_num_sections, 0);

            int p_xs_pack = begin_phase("Xs pack");
            {
                // N-way merge over kXsTiles sorted runs. Tiebreak:
                // lowest tile index wins, which generalizes the
                // 2-way "A wins on equal keys" stability that the
                // existing minimal-path launch_merge_pairs_stable_2way_u32_u32
                // provides. Fused with pack (no intermediate
                // h_xs_keys_merged / h_xs_vals_merged allocation).
                //
                // Hot loop is N=4 → 4 compares per output element.
                // At k=26 / total_xs ≈ 64M, that's ~256M compares;
                // measured ~150 ms (about 5x the 2-way path's 30 ms),
                // amortized across a multi-second plot wall is noise.
                uint64_t idx[kXsTinyTiles];
                for (int s = 0; s < kXsTiles; ++s) {
                    idx[s] = xs_tile_offsets[s];
                }
                uint64_t out = 0;
                while (true) {
                    int best = -1;
                    uint32_t best_k = 0;
                    for (int s = 0; s < kXsTiles; ++s) {
                        if (idx[s] < xs_tile_offsets[s + 1]) {
                            uint32_t const kk = h_xs_keys[idx[s]];
                            if (best == -1 || kk < best_k) {
                                best_k = kk;
                                best   = s;
                            }
                        }
                    }
                    if (best < 0) break;
                    uint32_t const k_out = h_xs_keys[idx[best]];
                    uint32_t const v_out = h_xs_vals[idx[best]];
                    h_xs[out] = XsCandidateGpu{ k_out, v_out };
                    ++section_count[k_out >> xs_section_shift];
                    ++idx[best];
                    ++out;
                }
            }
            end_phase(p_xs_pack);

            // Prefix-sum into the function-scope offsets array.
            h_xs_section_starts.assign(xs_num_sections + 1, 0);
            for (uint32_t s = 0; s < xs_num_sections; ++s) {
                h_xs_section_starts[s + 1] = h_xs_section_starts[s] + section_count[s];
            }

            s_free_host(stats, h_xs_keys, q);
            s_free_host(stats, h_xs_vals, q);
        } else {
            // Minimal/Tiny path — unchanged from before Phase 1.4.

            // Full-cap merge outputs on device. Merge from USM-host inputs.
            s_malloc(stats, d_xs_keys_b, total_xs * sizeof(uint32_t), "d_xs_keys_b");
            s_malloc(stats, d_xs_vals_b, total_xs * sizeof(uint32_t), "d_xs_vals_b");
            launch_merge_pairs_stable_2way_u32_u32(
                h_xs_keys + 0,           h_xs_vals + 0,           xs_tile_n0,
                h_xs_keys + xs_tile_n0,  h_xs_vals + xs_tile_n0,  xs_tile_n1,
                d_xs_keys_b, d_xs_vals_b, total_xs, q);
            s_free_host(stats, h_xs_keys, q);
            s_free_host(stats, h_xs_vals, q);

            // Tiled pack. d_xs_pack_tile reuses across tiles; the
            // packed output collects on host pinned h_xs (cap ×
            // XsCandidate = 2048 MB host at k=28).
            //
            // Cheap-win bump from N=2 to N=4: shrinks d_xs_pack_tile
            // from cap/2 × XsCandidate (1024 MB at k=28) to cap/4
            // × XsCandidate (512 MB at k=28). Saves 512 MB at k=28
            // / 128 MB at k=26 on the Xs pack phase peak.
            // The d_xs_keys_b + d_xs_vals_b co-residency (the bigger
            // pack contributors) is unchanged — that requires
            // Pinned-style algorithm to attack.
            constexpr int kXsPackTiles = 4;
            uint64_t const pack_tile_max =
                (total_xs + uint64_t(kXsPackTiles) - 1) / uint64_t(kXsPackTiles);

            XsCandidateGpu* d_xs_pack_tile = nullptr;
            s_malloc(stats, d_xs_pack_tile, pack_tile_max * sizeof(XsCandidateGpu), "d_xs_pack_tile");

            h_xs = s_malloc_host<XsCandidateGpu>(
                stats, total_xs * sizeof(XsCandidateGpu), "h_xs", q);

            int p_xs_pack = begin_phase("Xs pack");
            for (int n = 0; n < kXsPackTiles; ++n) {
                uint64_t const tile_off = uint64_t(n) * pack_tile_max;
                if (tile_off >= total_xs) break;
                uint64_t const tile_n = std::min(pack_tile_max, total_xs - tile_off);
                launch_xs_pack_range(d_xs_keys_b + tile_off,
                                     d_xs_vals_b + tile_off,
                                     d_xs_pack_tile, tile_n, q);
                q.memcpy(h_xs + tile_off, d_xs_pack_tile,
                         tile_n * sizeof(XsCandidateGpu)).wait();
            }
            end_phase(p_xs_pack);

            s_free(stats, d_xs_pack_tile);
            s_free(stats, d_xs_keys_b);
            s_free(stats, d_xs_vals_b);
            d_xs_keys_b = nullptr;
            d_xs_vals_b = nullptr;
        }

        // Re-hydrate full d_xs on device from host pinned (Minimal/Tiny).
        // Pinned (Phase 1.4c): skip — h_xs survives to T1 match, which
        // consumes it via per-section-pair tile H2D.
        if (scratch.tiny_mode) {
            h_xs_pinned = h_xs;
        } else {
            s_malloc(stats, d_xs, total_xs * sizeof(XsCandidateGpu), "d_xs");
            q.memcpy(d_xs, h_xs, total_xs * sizeof(XsCandidateGpu)).wait();
            s_free_host(stats, h_xs, q);
        }
    }

    // ---------- Phase T1 match ----------
    // SoA output: meta (uint64) + mi (uint32). Same 12 B/pair as the old
    // AoS struct, but the two streams can be freed independently — we
    // drop d_t1_mi as soon as CUB consumes it in the T1 sort phase.
    //
    // Minimal mode (gather_tile_count > 1) splits T1 match into N=
    // num_sections passes (one per section_l) with cap/N staging
    // outputs that are D2H'd to host pinned per pass — keeps d_xs +
    // d_t1_meta + d_t1_mi from being co-resident at full-cap. Drops
    // the T1 match peak from
    //   d_xs (2048) + d_t1_meta (2080) + d_t1_mi (1040) = 5168 MB
    // to
    //   d_xs (2048) + d_t1_meta_stage (cap/N × 8) +
    //   d_t1_mi_stage (cap/N × 4) = ~2870 MB at k=28 N=4.
    //
    // d_t1_meta + d_t1_mi (full cap) are then re-allocated on device
    // for T1 sort, with the data H2D'd from host pinned. d_t1_meta
    // stays parked on h_t1_meta across T1 sort exactly as in compact
    // mode (the existing park dance is skipped — data is already on
    // host).
    // t1_match_sliced: declared at function top.

    stats.phase = "T1 match";
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    size_t t1_temp_bytes = 0;
    launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          nullptr, nullptr, d_counter, cap,
                          nullptr, &t1_temp_bytes, q);

    uint64_t* d_t1_meta = nullptr;
    uint32_t* d_t1_mi   = nullptr;
    void*     d_t1_match_temp = nullptr;

    // h_t1_meta is now declared at function top (Phase 2.2 split prep)
    // so start_at_t2_match can populate it before this code runs.
    // h_t1_mi is sliced-only — freed in T1 sort once CUB has consumed
    // the H2D'd copy. h_meta_owned: declared at function top.
    bool      h_t1_mi_owned = false;
    uint32_t* h_t1_mi = nullptr;

    // t1_count: declared at function top.

    if (!t1_match_sliced) {
        // Single-shot path (compact / plain): d_t1_meta + d_t1_mi
        // allocated full-cap on device.
        s_malloc(stats, d_t1_meta,        cap * sizeof(uint64_t), "d_t1_meta");
        s_malloc(stats, d_t1_mi,          cap * sizeof(uint32_t), "d_t1_mi");
        s_malloc(stats, d_t1_match_temp,  t1_temp_bytes,          "d_t1_match_temp");

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sample_n = (total_xs < 16ULL) ? total_xs : 16ULL;
            XsCandidateGpu sample[16] = {};
            q.memcpy(sample, d_xs, sample_n * sizeof(XsCandidateGpu)).wait();
            std::fprintf(stderr,
                "[t1-debug] plain pre-launch  k=%d total_xs=%llu cap=%llu  d_xs[0..%llu]:\n",
                cfg.k, (unsigned long long)total_xs,
                (unsigned long long)cap, (unsigned long long)sample_n);
            for (uint64_t i = 0; i < sample_n; ++i) {
                std::fprintf(stderr,
                    "  [%2llu] match_info=0x%08x x=0x%08x\n",
                    (unsigned long long)i, sample[i].match_info, sample[i].x);
            }
        }

        int p_t1 = begin_phase("T1 match");
        q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                              d_t1_meta, d_t1_mi, d_counter, cap,
                              d_t1_match_temp, &t1_temp_bytes, q);
        end_phase(p_t1);

        q.memcpy(&t1_count, d_counter, sizeof(uint64_t)).wait();
        if (t1_count > cap) throw std::runtime_error("T1 overflow");
        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            std::fprintf(stderr,
                "[t1-debug] plain post-launch t1_count=%llu\n",
                (unsigned long long)t1_count);
        }
        validate_t1_count(t1_count, cfg.k);

        s_free(stats, d_t1_match_temp);
        s_free(stats, d_xs);
    } else {
        // Sliced path (minimal): N=num_sections passes with cap/N
        // staging buffers. Output accumulates on host pinned, then
        // d_t1_mi + h_t1_meta receive their final populations after
        // d_xs is freed.
        uint32_t const t1_num_sections   = 1u << t1p.num_section_bits;
        uint32_t const t1_num_match_keys = 1u << t1p.num_match_key_bits;
        uint32_t const t1_num_buckets    = t1_num_sections * t1_num_match_keys;
        // 25% safety over the per-section average expected output.
        uint64_t const t1_section_cap =
            ((cap + t1_num_sections - 1) / t1_num_sections) * 5ULL / 4ULL;
        // Phase 1.6a — Tiny per-(section_l, match_key_r) sub-pass cap:
        // staging only needs to hold one bucket's match output, not the
        // full section's. Same 25% safety. Drops staging from ~247 MB
        // (165 d_t1_meta_stage + 82 d_t1_mi_stage) to ~62 MB at k=26.
        uint64_t const t1_bucket_pair_cap =
            ((cap + t1_num_buckets - 1) / t1_num_buckets) * 5ULL / 4ULL;
        uint64_t const t1_stage_cap =
            scratch.tiny_mode ? t1_bucket_pair_cap : t1_section_cap;

        s_malloc(stats, d_t1_match_temp, t1_temp_bytes, "d_t1_match_temp");

        // Pinned (Phase 1.4c) defers the prepare to per-pass on each
        // section-pair tile; Minimal/Tiny do it once on full d_xs as
        // before.
        if (!scratch.tiny_mode) {
            // Compute bucket + fine-bucket offsets once; passes share them.
            // Also zeros d_counter.
            launch_t1_match_prepare(cfg.plot_id.data(), t1p, d_xs, total_xs,
                                    d_counter, d_t1_match_temp, &t1_temp_bytes, q);
        }

        // Host pinned full-cap accumulators for meta + mi.
        //
        // Host-RAM disk-offload (docs/host-ram-disk-offload.md): when the
        // budget policy selected h_t1_meta for spill (scratch.spill.h_t1_meta,
        // or the legacy XCHPLOT2_SPILL_T1META=1 flag) and we own the buffer
        // in tiny mode, redirect h_t1_meta (~2 GiB at k=28) to a TempFile on
        // disk, streamed through the shared 64 MiB SpillEngine instead of a
        // full pinned alloc. h_t1_meta stays null; the SpillBuffer services
        // every access. Default takes the exact original pinned path below.
        if (h_meta_owned && scratch.tiny_mode) {
            bool const want_spill = scratch.spill.h_t1_meta ||
                [] { char const* sv = std::getenv("XCHPLOT2_SPILL_T1META");
                     return sv && sv[0] == '1'; }();
            if (want_spill) {
                t1_meta_spill = std::make_unique<SpillBuffer>(
                    ensure_spill_engine(), sizeof(uint64_t));
                h_t1_meta = nullptr;
                std::fprintf(stderr,
                    "[spill] h_t1_meta -> disk %s (shared staging %llu MiB in %d windows, "
                    "overlap=%s, cap %.2f GiB spilled)\n",
                    t1_meta_spill->file.path().c_str(),
                    (unsigned long long)(SpillEngine::kNumWindows *
                        SpillEngine::kStageBytes / 1048576),
                    SpillEngine::kNumWindows,
                    spill_engine->overlap ? "on" : "off",
                    cap * sizeof(uint64_t) / 1073741824.0);
            }
        }
        if (!t1_meta_spill) {
            h_t1_meta = h_meta_owned
                ? s_malloc_host<uint64_t>(stats, cap * sizeof(uint64_t), "h_t1_meta", q)
                : scratch.h_meta;
            if (!h_t1_meta) throw std::runtime_error("sycl::malloc_host(h_t1_meta) failed");
        }
        if (scratch.pool) {
            // Pool path: amortise across plots in a batch. The pool
            // owns the buffer and frees it on its own destruction; the
            // streaming pipeline must NOT free it here.
            h_t1_mi = scratch.pool->acquire_as<uint32_t>("h_t1_mi", cap, q);
            h_t1_mi_owned = false;
        } else {
            h_t1_mi_owned = true;
            h_t1_mi = s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t), "h_t1_mi", q);
        }

        // Per-pass staging device buffers (cap/N).
        uint64_t* d_t1_meta_stage = nullptr;
        uint32_t* d_t1_mi_stage   = nullptr;
        s_malloc(stats, d_t1_meta_stage, t1_stage_cap * sizeof(uint64_t), "d_t1_meta_stage");
        s_malloc(stats, d_t1_mi_stage,   t1_stage_cap * sizeof(uint32_t), "d_t1_mi_stage");

        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            uint64_t const sample_n = (total_xs < 16ULL) ? total_xs : 16ULL;
            XsCandidateGpu sample[16] = {};
            q.memcpy(sample, d_xs, sample_n * sizeof(XsCandidateGpu)).wait();
            std::fprintf(stderr,
                "[t1-debug] sliced pre-launch k=%d total_xs=%llu cap=%llu  d_xs[0..%llu]:\n",
                cfg.k, (unsigned long long)total_xs,
                (unsigned long long)cap, (unsigned long long)sample_n);
            for (uint64_t i = 0; i < sample_n; ++i) {
                std::fprintf(stderr,
                    "  [%2llu] match_info=0x%08x x=0x%08x\n",
                    (unsigned long long)i, sample[i].match_info, sample[i].x);
            }
        }

        // Phase 1.4c — Pinned per-pair-tile setup. matching_section is
        // pos2-chip ProofCore::matching_section, used internally by
        // launch_t1_match_all_buckets to find each L bucket's matching
        // R bucket. We compute it ourselves to know which two sections
        // need to be co-resident in d_xs_tile for each pass.
        //
        // CRITICAL: the tile MUST be sorted by bucket number — the
        // prepare's bucket-offset construction does binary search on
        // d_sorted[mid].match_info >> bucket_shift and ASSUMES the
        // sequence is monotonically increasing. Always H2D the
        // numerically-smaller section first, regardless of which side
        // (L or R) it is semantically. The kernel's d_offsets lookups
        // are by absolute section number, so semantic L/R order is
        // irrelevant to correctness.
        int      const ns_bits  = t1p.num_section_bits;
        uint32_t const num_secs = t1_num_sections;
        auto matching_section = [ns_bits, num_secs](uint32_t s) -> uint32_t {
            uint32_t const rl  = ((s << 1) | (s >> (ns_bits - 1))) & (num_secs - 1);
            uint32_t const rl1 = (rl + 1) & (num_secs - 1);
            return ((rl1 >> 1) | (rl1 << (ns_bits - 1))) & (num_secs - 1);
        };

        // Phase 1.6a — Tiny sub-section attack: process each (section_l,
        // match_key_r) bucket-pair independently. The kernel reads L
        // from d_offsets[section_l*nmk..(section_l+1)*nmk] (full
        // section_l) but only ONE r_bucket per bucket_id. So we can
        // fetch L_full + R_one_bucket per pass and shrink the tile
        // from L+R (256 MB at k=26) to L+R/N where N=num_match_keys
        // (160 MB at k=26 / 1280 MB at k=28).
        //
        // Pre-compute per-section per-bucket entry boundaries in
        // h_xs_pinned (sorted by match_info, so within section_s the
        // buckets are contiguous; binary search the boundaries).
        std::vector<std::vector<uint64_t>> section_bucket_starts;
        XsCandidateGpu* d_xs_tile = nullptr;
        uint64_t        max_pair  = 0;
        if (scratch.tiny_mode) {
            int const t1_bucket_shift = t1p.num_match_target_bits;
            section_bucket_starts.assign(
                num_secs, std::vector<uint64_t>(t1_num_match_keys + 1));
            for (uint32_t s = 0; s < num_secs; ++s) {
                uint64_t const s_off  = h_xs_section_starts[s];
                uint64_t const s_size = h_xs_section_starts[s + 1] - s_off;
                section_bucket_starts[s][0] = 0;
                section_bucket_starts[s][t1_num_match_keys] = s_size;
                for (uint32_t b = 1; b < t1_num_match_keys; ++b) {
                    uint32_t const target_bucket = s * t1_num_match_keys + b;
                    uint64_t lo = 0, hi = s_size;
                    while (lo < hi) {
                        uint64_t mid = lo + (hi - lo) / 2;
                        uint32_t const bm =
                            h_xs_pinned[s_off + mid].match_info >> t1_bucket_shift;
                        if (bm < target_bucket) lo = mid + 1;
                        else                    hi = mid;
                    }
                    section_bucket_starts[s][b] = lo;
                }
            }

            // Sub-tile size = max over all section pairs of
            // (L_section_size + max R bucket in matching R section).
            for (uint32_t s = 0; s < num_secs; ++s) {
                uint64_t const sl_size =
                    h_xs_section_starts[s + 1] - h_xs_section_starts[s];
                uint32_t const sr = matching_section(s);
                uint64_t r_max_bucket = 0;
                for (uint32_t b = 0; b < t1_num_match_keys; ++b) {
                    uint64_t const b_size =
                        section_bucket_starts[sr][b + 1] -
                        section_bucket_starts[sr][b];
                    if (b_size > r_max_bucket) r_max_bucket = b_size;
                }
                uint64_t const sub_pair = sl_size + r_max_bucket;
                if (sub_pair > max_pair) max_pair = sub_pair;
            }
            s_malloc(stats, d_xs_tile,
                     max_pair * sizeof(XsCandidateGpu), "d_xs_tile_pinned");
        }

        int p_t1 = begin_phase("T1 match");
        uint64_t host_offset = 0;
        // POS2GPU_T1_PASS_TRACE=1: one line per tiny sub-section pass with
        // both the INPUT slice (l_n / r_buck_n, computed on the host by
        // binary search over h_xs_pinned) and the OUTPUT pair count. A
        // corrupt tiny+spill plot loses exactly 1/16 of T1, i.e. one pass
        // of the sixteen; diffing this trace against a good run says which
        // pass, and — decisively — whether that pass was FED wrong (bad
        // slice bounds) or fed right and MATCHED wrong.
        bool const t1_pass_trace = [] {
            char const* v = std::getenv("POS2GPU_T1_PASS_TRACE");
            return v && v[0] == '1';
        }();
        // POS2GPU_T1_DROP_R=<section_l>,<mk>: fault injection — skip the R
        // half of that pass's tile H2D, reproducing exactly what an
        // unwaited copy leaves behind. This exists to prove causation
        // rather than infer it: the race is ~1-in-8 and nothing about
        // observing it says the missing copy is what caused the bad plot.
        // Dropping the copy on purpose and getting the SAME plot hash does.
        // Add POS2GPU_T1_DROP_R_NOTHROW=1 to suppress the zero-yield guard
        // below, so the run produces the corrupt plot instead of the error.
        int drop_r_sec = -1, drop_r_mk = -1;
        if (char const* v = std::getenv("POS2GPU_T1_DROP_R"); v && v[0]) {
            std::sscanf(v, "%d,%d", &drop_r_sec, &drop_r_mk);
        }
        bool const drop_r_nothrow = [] {
            char const* v = std::getenv("POS2GPU_T1_DROP_R_NOTHROW");
            return v && v[0] == '1';
        }();
        // This switch deliberately corrupts the plot. Say so, loudly and
        // unconditionally — a plotter must never damage output on the
        // strength of a stray environment variable without leaving a trace
        // in the log that explains the bad plot.
        if (drop_r_sec >= 0) {
            std::fprintf(stderr,
                "[t1-inject] *** POS2GPU_T1_DROP_R=%d,%d IS SET: deliberately "
                "skipping that pass's R tile copy. THIS CORRUPTS THE PLOT. %s ***\n",
                drop_r_sec, drop_r_mk,
                drop_r_nothrow
                    ? "POS2GPU_T1_DROP_R_NOTHROW=1 also suppresses the zero-yield "
                      "guard, so a SILENTLY WRONG plot WILL be written."
                    : "The zero-yield guard will abort the plot.");
        }
        for (uint32_t section_l = 0; section_l < t1_num_sections; ++section_l) {
            if (scratch.tiny_mode) {
                // N = num_match_keys sub-passes per section_l. Each
                // pass: L (full section_l) + R (one bucket of
                // section_r), processing the one bucket_id =
                // section_l*nmk + mk that hits that R bucket.
                uint32_t const section_r = matching_section(section_l);
                uint64_t const l_off = h_xs_section_starts[section_l];
                uint64_t const l_n   = h_xs_section_starts[section_l + 1] - l_off;
                uint64_t const r_sec_off = h_xs_section_starts[section_r];

                for (uint32_t mk = 0; mk < t1_num_match_keys; ++mk) {
                    uint64_t const r_buck_off_in_sec =
                        section_bucket_starts[section_r][mk];
                    uint64_t const r_buck_n =
                        section_bucket_starts[section_r][mk + 1] -
                        r_buck_off_in_sec;
                    uint64_t const r_buck_abs_off = r_sec_off + r_buck_off_in_sec;
                    uint64_t const total_for_pass = l_n + r_buck_n;
                    uint32_t const bucket_begin =
                        section_l * t1_num_match_keys + mk;
                    uint32_t const bucket_end   = bucket_begin + 1;

                    // Smaller section first — L section has bucket_ids
                    // [section_l*nmk, (section_l+1)*nmk); R bucket has
                    // bucket_id section_r*nmk + mk. The single-R-bucket
                    // bucket_id is uniformly above or below all L
                    // bucket_ids based on section_l vs section_r.
                    //
                    // BOTH halves must be waited, not just the second.
                    // These are two independent USM copies with no data
                    // dependency between them, so on an out-of-order queue
                    // waiting the second says nothing about the first; the
                    // "issue N, wait the last" shorthand is only safe when
                    // the ops are ordered by something else. Losing the R
                    // half here is silent and severe: launch_t1_match_range
                    // finds its R candidates through d_fine_offsets[r_bucket],
                    // and a tile whose head still holds the PREVIOUS pass's
                    // R bucket contains no entries for this one — so every L
                    // thread reads an empty range and the pass yields exactly
                    // zero pairs. That drops 1/16 of T1 and, through pairing,
                    // (15/16)^2 of T2 and (15/16)^4 of T3, for a plot that is
                    // ~78% the right size and completely wrong. Waiting both
                    // events keeps the copies concurrent and costs nothing.
                    bool const drop_r = (int(section_l) == drop_r_sec &&
                                         int(mk)        == drop_r_mk);
                    sycl::event e_l, e_r;
                    if (section_l < section_r) {
                        e_l = q.memcpy(d_xs_tile,
                                       h_xs_pinned + l_off,
                                       l_n * sizeof(XsCandidateGpu));
                        if (!drop_r)
                            e_r = q.memcpy(d_xs_tile + l_n,
                                           h_xs_pinned + r_buck_abs_off,
                                           r_buck_n * sizeof(XsCandidateGpu));
                    } else {
                        if (!drop_r)
                            e_r = q.memcpy(d_xs_tile,
                                           h_xs_pinned + r_buck_abs_off,
                                           r_buck_n * sizeof(XsCandidateGpu));
                        e_l = q.memcpy(d_xs_tile + r_buck_n,
                                       h_xs_pinned + l_off,
                                       l_n * sizeof(XsCandidateGpu));
                    }
                    e_l.wait();
                    if (!drop_r) e_r.wait();

                    launch_t1_match_prepare(
                        cfg.plot_id.data(), t1p, d_xs_tile, total_for_pass,
                        d_counter, d_t1_match_temp, &t1_temp_bytes, q);

                    launch_t1_match_range(
                        cfg.plot_id.data(), t1p, d_xs_tile, total_for_pass,
                        d_t1_meta_stage, d_t1_mi_stage, d_counter, t1_stage_cap,
                        d_t1_match_temp, bucket_begin, bucket_end, q);

                    uint64_t pass_count = 0;
                    q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
                    if (pass_count > t1_stage_cap) {
                        throw std::runtime_error(
                            "T1 match (sub-section) section_l=" +
                            std::to_string(section_l) +
                            " mk=" + std::to_string(mk) +
                            " produced " + std::to_string(pass_count) +
                            " pairs, staging holds " + std::to_string(t1_stage_cap) +
                            ". Increase t1_bucket_pair_cap safety factor.");
                    }
                    // Check the accumulation target BEFORE the copy: the
                    // per-pass caps sum to ~1.25×cap, so a skewed
                    // distribution can pass every per-pass check yet
                    // overrun the cap-sized host buffers — the post-loop
                    // "T1 overflow" throw would arrive after the heap
                    // corruption.
                    if (host_offset + pass_count > cap) {
                        throw std::runtime_error("T1 overflow (sliced accumulation)");
                    }
                    if (t1_pass_trace) {
                        std::fprintf(stderr,
                            "[t1-pass] s_l=%u mk=%u  l_off=%llu l_n=%llu  "
                            "r_off=%llu r_n=%llu  total=%llu  -> count=%llu "
                            "host_off=%llu\n",
                            section_l, mk,
                            (unsigned long long)l_off, (unsigned long long)l_n,
                            (unsigned long long)r_buck_abs_off,
                            (unsigned long long)r_buck_n,
                            (unsigned long long)total_for_pass,
                            (unsigned long long)pass_count,
                            (unsigned long long)host_offset);
                    }
                    // A pass fed a non-empty L section AND a non-empty R
                    // bucket cannot legitimately match nothing: the smallest
                    // k this tier supports puts thousands of entries on each
                    // side, so the expected yield is thousands and P(exactly
                    // zero) is not a number that occurs. A zero here means
                    // the pass was fed a tile that does not contain its R
                    // bucket — the failure that silently drops 1/16 of T1 and
                    // produces a ~78%-sized plot that still passes every
                    // structural check. Refuse to write that plot.
                    if (pass_count == 0 && l_n > 0 && r_buck_n > 0
                        && !drop_r_nothrow) {
                        throw std::runtime_error(
                            "T1 match (tiny sub-section) section_l=" +
                            std::to_string(section_l) + " mk=" + std::to_string(mk) +
                            " matched ZERO pairs from l_n=" + std::to_string(l_n) +
                            " r_n=" + std::to_string(r_buck_n) + " (tile of " +
                            std::to_string(total_for_pass) + "). Both inputs are "
                            "non-empty, so this is not a real result — the "
                            "staged tile did not contain R bucket " +
                            std::to_string(section_r * t1_num_match_keys + mk) +
                            ". Continuing would drop 1/16 of T1 and write a "
                            "short, silently wrong plot.");
                    }
                    if (t1_meta_spill) {
                        t1_meta_spill->write_from_device(
                            d_t1_meta_stage, host_offset, pass_count);
                    } else {
                        q.memcpy(h_t1_meta + host_offset, d_t1_meta_stage,
                                 pass_count * sizeof(uint64_t)).wait();
                    }
                    q.memcpy(h_t1_mi   + host_offset, d_t1_mi_stage,
                             pass_count * sizeof(uint32_t)).wait();
                    host_offset += pass_count;
                    q.memset(d_counter, 0, sizeof(uint64_t)).wait();
                }
            } else {
                // Non-tiny: existing per-section pass (no sub-bucket).
                uint32_t const bucket_begin = section_l * t1_num_match_keys;
                uint32_t const bucket_end   = (section_l + 1) * t1_num_match_keys;

                launch_t1_match_range(
                    cfg.plot_id.data(), t1p, d_xs, total_xs,
                    d_t1_meta_stage, d_t1_mi_stage, d_counter, t1_stage_cap,
                    d_t1_match_temp, bucket_begin, bucket_end, q);

                uint64_t pass_count = 0;
                q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
                if (pass_count > t1_stage_cap) {
                    throw std::runtime_error(
                        "T1 match (sliced) section_l=" + std::to_string(section_l) +
                        " produced " + std::to_string(pass_count) +
                        " pairs, staging holds " + std::to_string(t1_stage_cap) +
                        ". Increase t1_section_cap safety factor.");
                }
                // Same pre-copy bounds check as the tiny branch above:
                // per-section caps sum past cap, so guard before writing
                // into the cap-sized host buffers.
                if (host_offset + pass_count > cap) {
                    throw std::runtime_error("T1 overflow (sliced accumulation)");
                }
                q.memcpy(h_t1_meta + host_offset, d_t1_meta_stage,
                         pass_count * sizeof(uint64_t)).wait();
                q.memcpy(h_t1_mi   + host_offset, d_t1_mi_stage,
                         pass_count * sizeof(uint32_t)).wait();
                host_offset += pass_count;
                q.memset(d_counter, 0, sizeof(uint64_t)).wait();
            }
        }
        end_phase(p_t1);

        if (scratch.tiny_mode && d_xs_tile) {
            s_free(stats, d_xs_tile);
        }
        if (scratch.tiny_mode && h_xs_pinned) {
            s_free_host(stats, h_xs_pinned, q);
            h_xs_pinned = nullptr;
        }

        t1_count = host_offset;
        if (t1_count > cap) throw std::runtime_error("T1 overflow");
        if (char const* v = std::getenv("POS2GPU_T1_DEBUG"); v && v[0] == '1') {
            std::fprintf(stderr,
                "[t1-debug] sliced post-launch t1_count=%llu (sum across %u sections)\n",
                (unsigned long long)t1_count, t1_num_sections);
        }
        validate_t1_count(t1_count, cfg.k);

        s_free(stats, d_t1_meta_stage);
        s_free(stats, d_t1_mi_stage);
        s_free(stats, d_t1_match_temp);

        // Xs fully consumed. Pinned never allocates d_xs (h_xs feeds
        // T1 match via per-pair tiles); Minimal/Tiny did so it must
        // be freed.
        if (d_xs) {
            s_free(stats, d_xs);
        }

        // Re-hydrate d_t1_mi full-cap on device for T1 sort (CUB
        // sort key input). h_t1_meta stays on host across T1 sort.
        s_malloc(stats, d_t1_mi, cap * sizeof(uint32_t), "d_t1_mi");
        q.memcpy(d_t1_mi, h_t1_mi, t1_count * sizeof(uint32_t)).wait();
        if (h_t1_mi_owned) s_free_host(stats, h_t1_mi, q);
        h_t1_mi = nullptr;
        // d_t1_meta stays nullptr — h_t1_meta has the data; the
        // existing T1-sort park block will see d_t1_meta == nullptr
        // and skip the d_t1_meta → h_t1_meta memcpy.
    }

    // Stage 4b (compact only): park d_t1_meta on pinned host across
    // the T1 sort phase. d_t1_meta is only needed again for
    // launch_gather_u64 at the end of T1 sort — holding it alive
    // through CUB setup was responsible for the 6256 MB overall
    // streaming peak (d_t1_meta 2080 + d_t1_mi 1040 + CUB working 3120
    // + scratch). JIT H2D before the gather below, free right after.
    // Mirror of stage 4a for T2.
    //
    // Stage 4f: use caller-provided scratch when present (amortised
    // across batch); fall back to per-plot malloc_host otherwise. Same
    // pattern applied to h_t1_keys_merged, h_t2_*, h_t3 below.
    //
    // Plain mode skips the park entirely: d_t1_meta stays live through
    // T1 sort. Costs ~2 GB peak but saves a PCIe round-trip.
    //
    // Sliced mode: h_t1_meta was already populated by the T1 match
    // passes — d_t1_meta is nullptr and the park dance is skipped
    // here. h_meta_owned + h_t1_meta were declared above (lifted out
    // of the original T1-sort scope) so the rest of T1 sort sees the
    // same variables in both paths.
    if (!scratch.plain_mode && !t1_match_sliced) {
        h_t1_meta = h_meta_owned
            ? s_malloc_host<uint64_t>(stats, cap * sizeof(uint64_t), "h_t1_meta", q)
            : scratch.h_meta;
        if (!h_t1_meta) throw std::runtime_error("sycl::malloc_host(h_t1_meta) failed");
        q.memcpy(h_t1_meta, d_t1_meta, t1_count * sizeof(uint64_t)).wait();
        s_free(stats, d_t1_meta);
        d_t1_meta = nullptr;
    }

    // ---------- Phase T1 sort (tiled, N=2) ----------
    // Partition T1 into two halves by index, CUB-sort each with scratch
    // sized for the larger half, then stable 2-way merge the sorted runs
    // back into the extract-input slot (d_keys_in / d_vals_in) — that
    // slot is free because the CUB sort has already consumed it.
    //
    // N=2 is the minimal case that exercises the tile + merge path; a
    // larger N shrinks per-tile CUB scratch further but needs a multi-
    // way merge or a tree of pairwise merges. Phase 6 can bump N once
    // Phase 4's k=28 VRAM measurement shows how tight the budget is.
    uint64_t const t1_tile_n0  = t1_count / 2;
    uint64_t const t1_tile_n1  = t1_count - t1_tile_n0;
    uint64_t const t1_tile_max = (t1_tile_n0 > t1_tile_n1) ? t1_tile_n0 : t1_tile_n1;

    size_t t1_sort_bytes = 0;
    launch_sort_pairs_u32_u32(
        nullptr, t1_sort_bytes,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        t1_tile_max, 0, cfg.k, q);

    stats.phase = "T1 sort";
    // With T1 SoA emission, d_t1_mi IS the CUB key input. We only need
    // d_keys_out (CUB sort output), d_vals_in (identity) + d_vals_out
    // (sorted vals). d_t1_mi is freed as soon as CUB consumes it.
    //
    // Compact / plain: full-cap d_keys_out + d_vals_in + d_vals_out
    // (1040 MB each at k=28); plus d_t1_mi (1040, full-cap input) +
    // scratch ≈ 4176 MB peak.
    //
    // Minimal: per-tile cap/2 output buffers (520 each) instead of
    // full-cap + USM-host h_keys/h_vals to collect tile outputs +
    // launch_merge_pairs_stable_2way_u32_u32 reading USM-host inputs.
    // Drops T1 sort CUB peak to:
    //   d_t1_mi (1040) + 3 × cap/2 u32 (1560) + scratch ≈ 2616 MB.
    // d_sort_scratch: declared at function top for cross-phase reuse.
    // d_keys_out / d_vals_in / d_vals_out / h_keys: lifted to function
    // top (Phase 2.2 split prep) so they survive past the Xs+T1
    // if-skip. Comment summary of original sites preserved for context:
    //   d_keys_out — populated in compact path; minimal uses h_keys instead
    //   d_vals_in  — T2 sort below also uses this; wider-scope intent kept
    //   d_vals_out — populated in compact path; minimal uses h_vals instead
    //   h_keys     — USM-host, sliced path only
    // h_vals lifted to function top (Phase 2.2 split prep). USM-host, sliced path only.

    int p_t1_sort = begin_phase("T1 sort");

    // ------------------------------------------------------------
    // Phase 1.3c-ii Pinned T1 sort: streaming partition + per-bucket sort.
    //
    // Today's Tiny-mode T1 sort gather (~2080 MB d_t1_meta on device
    // at k=28) is the streaming-pipeline floor — it's why we can't
    // fit on 2-3 GB GPUs even after every other host park. Pinned
    // replaces that gather with the streaming-partition primitive
    // (Phase 1.3b) + per-bucket launch_sort_pairs_u32_u64 (Phase 1.3a):
    // h_t1_meta is read tile-by-tile, partitioned to host-pinned
    // bucket arenas, and each bucket sorted in turn on a small per-
    // bucket-sized device buffer. d_t1_meta never lives at full cap
    // on device.
    //
    // Outputs match Tiny's contract:
    //   h_t1_meta:        sorted in place
    //   h_t1_keys_merged: sorted on host-pinned (downstream T2 match
    //                     reads section slices from it; tiny path
    //                     already parks d_t1_keys_merged on host)
    //   d_t1_mi:          freed (no longer needed)
    //   d_t1_keys_merged: nullptr (parked on host as h_t1_keys_merged,
    //                     matching tiny's behaviour after this phase)
    if (scratch.tiny_mode) {
        // Pick partition geometry. num_top_bits must satisfy
        //   begin_bit=0 + num_top_bits <= end_bit=cfg.k
        // with at least 1 lower bit left to sort within each bucket
        // (so num_top_bits <= cfg.k - 1).
        //
        // num_top_bits scales with k so the per-bucket entry count
        // stays in CUB's efficient range without exploding the per-
        // bucket launch count. The previous flat `min(8, k-1)` made
        // smaller k pay 256 sequential per-bucket sort launches on
        // tiny buckets — at k=24 (~5K entries/bucket × 256 launches
        // × ~500 µs each) that was ~128 ms of pure launch overhead,
        // a 50% wall regression vs Tiny.
        //
        // Heuristic:  max(4, min(8, k - 18))
        //   k=18..22 →  4 → 16 buckets   (k=22: ~22K/bucket)
        //   k=24     →  6 → 64 buckets   (~22K/bucket)
        //   k=26     →  8 → 256 buckets  (~22K/bucket)
        //   k=28+    →  8 → 256 buckets  (~1M/bucket at k=28)
        //
        // Target: ~20K-1M entries per bucket — small enough that the
        // per-bucket CUB sort fits comfortably in scratch, large enough
        // that launch overhead is a fraction of sort time.
        int  const num_top_bits   = std::max(4, std::min(8, cfg.k - 18));
        int  const top_bit_offset = cfg.k - num_top_bits;
        size_t const num_buckets  = size_t{1} << num_top_bits;

        // Allocate h_t1_keys_merged using the same pool/scratch/
        // malloc_host pattern the merge phase uses below (so the
        // ownership flag h_keys_owned remains accurate). Streaming
        // partition writes its bucketed key output directly into
        // this buffer, and the per-bucket sort overwrites it with
        // sorted output — no extra copy.
        if (h_keys_owned) {
            h_t1_keys_merged = scratch.pool
                ? scratch.pool->acquire_as<uint32_t>("h_keys_merged", cap, q)
                : s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t),
                                          "h_t1_keys_merged(pinned)", q);
            if (!h_t1_keys_merged) throw std::runtime_error(
                "sycl::malloc_host(h_t1_keys_merged, pinned mode) failed");
        } else {
            h_t1_keys_merged = scratch.h_keys_merged;
        }

        // Temp host arenas for partition output vals and bucket starts.
        uint64_t* h_part_vals = s_malloc_host<uint64_t>(
            stats, cap * sizeof(uint64_t), "h_part_vals(pinned)", q);
        uint32_t* h_bucket_starts = s_malloc_host<uint32_t>(
            stats, (num_buckets + 1) * sizeof(uint32_t),
            "h_bucket_starts(pinned)", q);

        // Run streaming partition. Output: (h_t1_keys_merged,
        // h_part_vals) bucketed by top_num_top_bits of mi, with
        // h_bucket_starts giving the exclusive-scan offsets.
        // P1.5 spill: when h_t1_meta lives on disk, feed the partition's
        // per-tile source-value reads from the TempFile through the
        // double-buffered SpillTileReader (h_vals_in stays null). First
        // land all T1-match write-backs (the A->B barrier), then hand the
        // primitive the reader. tile_count is fixed so each tile fits a
        // window; BOTH calls pass the same tile_count so the scratch-size
        // query matches the exec pass (the query returns before any read).
        if (t1_meta_spill) t1_meta_spill->drain();
        SpillTileReader        sp_reader;
        SpillTileReader const* sp_reader_p = nullptr;
        uint64_t const         sp_tiles    = t1_meta_spill
            ? t1_meta_spill->tile_count_for(t1_count) : 0;
        if (t1_meta_spill) {
            sp_reader   = t1_meta_spill->tile_reader();
            sp_reader_p = &sp_reader;
        }
        size_t partition_scratch_bytes = 0;
        launch_streaming_partition_u32_u64(
            nullptr, partition_scratch_bytes,
            d_t1_mi, h_t1_meta,
            h_t1_keys_merged, h_part_vals, h_bucket_starts,
            t1_count, top_bit_offset, num_top_bits,
            sp_tiles, q, sp_reader_p);
        void* d_partition_scratch = nullptr;
        if (partition_scratch_bytes) {
            s_malloc(stats, d_partition_scratch,
                     partition_scratch_bytes, "d_t1_partition_scratch");
        }
        launch_streaming_partition_u32_u64(
            d_partition_scratch, partition_scratch_bytes,
            d_t1_mi, h_t1_meta,
            h_t1_keys_merged, h_part_vals, h_bucket_starts,
            t1_count, top_bit_offset, num_top_bits,
            sp_tiles, q, sp_reader_p);
        if (d_partition_scratch) s_free(stats, d_partition_scratch);

        // d_t1_mi is no longer needed after partition.
        s_free(stats, d_t1_mi);
        d_t1_mi = nullptr;

        // Find max bucket size for per-bucket device buffer sizing.
        uint32_t max_bucket = 0;
        for (size_t b = 0; b < num_buckets; ++b) {
            uint32_t const bsz = h_bucket_starts[b + 1] - h_bucket_starts[b];
            if (bsz > max_bucket) max_bucket = bsz;
        }

        // Per-bucket device scratch, sized at max bucket. Reused
        // across all buckets — the per-bucket sort is sequential.
        uint32_t* d_bk_in  = nullptr;
        uint32_t* d_bk_out = nullptr;
        uint64_t* d_bv_in  = nullptr;
        uint64_t* d_bv_out = nullptr;
        s_malloc(stats, d_bk_in,  max_bucket * sizeof(uint32_t), "d_t1_bk_in");
        s_malloc(stats, d_bk_out, max_bucket * sizeof(uint32_t), "d_t1_bk_out");
        s_malloc(stats, d_bv_in,  max_bucket * sizeof(uint64_t), "d_t1_bv_in");
        s_malloc(stats, d_bv_out, max_bucket * sizeof(uint64_t), "d_t1_bv_out");

        // Query sort scratch at the max bucket size.
        size_t bucket_sort_bytes = 0;
        launch_sort_pairs_u32_u64(
            nullptr, bucket_sort_bytes,
            d_bk_in, d_bk_out, d_bv_in, d_bv_out,
            max_bucket, 0, top_bit_offset, q);
        void* d_bucket_sort_scratch = nullptr;
        if (bucket_sort_bytes) {
            s_malloc(stats, d_bucket_sort_scratch,
                     bucket_sort_bytes, "d_t1_bucket_sort_scratch");
        }

        for (size_t b = 0; b < num_buckets; ++b) {
            uint32_t const bstart = h_bucket_starts[b];
            uint32_t const bsz    = h_bucket_starts[b + 1] - bstart;
            if (bsz == 0) continue;

            // Both copies feed launch_sort_pairs_u32_u64 below; on an
            // out-of-order queue waiting only the second lets the sort
            // race the first. See StreamingPartitionSycl.cpp:374.
            auto e_bk = q.memcpy(d_bk_in, h_t1_keys_merged + bstart,
                                 bsz * sizeof(uint32_t));
            auto e_bv = q.memcpy(d_bv_in, h_part_vals + bstart,
                                 bsz * sizeof(uint64_t));
            e_bk.wait();
            e_bv.wait();

            size_t bsb = bucket_sort_bytes;
            launch_sort_pairs_u32_u64(
                d_bucket_sort_scratch, bsb,
                d_bk_in, d_bk_out, d_bv_in, d_bv_out,
                bsz, 0, top_bit_offset, q);

            // Overwrite the bucketed-but-unsorted ranges with sorted
            // data. h_t1_keys_merged was the partition output target
            // for keys; sorted keys go back to the same slot. h_t1_meta
            // was the SOURCE of streaming_partition — its unsorted
            // contents are no longer needed, so we use it as the final
            // sorted-meta destination directly (in place overwrite).
            // The queue is NOT in-order (an earlier comment here claimed
            // it was). Nothing else drains this copy: the spill branch
            // waits on the SpillEngine's own queue handle, and the next
            // iteration's sort overwrites d_bk_out — a write-after-read
            // hazard on a buffer whose contents are consumed after the
            // loop. Wait for it explicitly.
            auto e_keys_back = q.memcpy(h_t1_keys_merged + bstart, d_bk_out,
                                        bsz * sizeof(uint32_t));
            if (t1_meta_spill) {
                t1_meta_spill->write_from_device(d_bv_out, bstart, bsz);
            } else {
                q.memcpy(h_t1_meta + bstart, d_bv_out,
                         bsz * sizeof(uint64_t)).wait();
            }
            e_keys_back.wait();
        }

        if (d_bucket_sort_scratch) s_free(stats, d_bucket_sort_scratch);
        s_free(stats, d_bk_in);
        s_free(stats, d_bk_out);
        s_free(stats, d_bv_in);
        s_free(stats, d_bv_out);
        s_free_host(stats, h_part_vals, q);
        s_free_host(stats, h_bucket_starts, q);

        end_phase(p_t1_sort);
        // h_t1_keys_merged and h_t1_meta are both sorted on host-pinned.
        // d_t1_keys_merged stays nullptr (parked on host) — same shape
        // tiny mode reaches after the 2-way merge + D2H park.
        goto t1_sort_done;
    }

    {  // Block-scope wrapper for non-pinned T1 sort path. The goto in
       // the pinned branch above lands at t1_sort_done; this block
       // contains the existing CUB-sort + 2-way-merge + gather code
       // (with its own local variable declarations) which goto must
       // not appear to "skip past" at function scope.
    if (!t1_match_sliced) {
        // Compact / plain — existing full-cap path.
        s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
        s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
        s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
        s_malloc(stats, d_sort_scratch, t1_sort_bytes,          "d_sort_scratch(t1)");

        launch_init_u32_identity(d_vals_in, t1_count, q);
        if (t1_tile_n0 > 0) {
            launch_sort_pairs_u32_u32(
                d_sort_scratch, t1_sort_bytes,
                d_t1_mi + 0, d_keys_out + 0,
                d_vals_in + 0, d_vals_out + 0,
                t1_tile_n0, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
        }
        if (t1_tile_n1 > 0) {
            launch_sort_pairs_u32_u32(
                d_sort_scratch, t1_sort_bytes,
                d_t1_mi + t1_tile_n0, d_keys_out + t1_tile_n0,
                d_vals_in + t1_tile_n0, d_vals_out + t1_tile_n0,
                t1_tile_n1, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
        }

        s_free(stats, d_sort_scratch);
        s_free(stats, d_vals_in);
        s_free(stats, d_t1_mi);
    } else {
        // Sliced — per-tile cap/2 output buffers, D2H to USM-host.
        uint32_t* d_keys_out_tile = nullptr;
        uint32_t* d_vals_in_tile  = nullptr;
        uint32_t* d_vals_out_tile = nullptr;
        s_malloc(stats, d_keys_out_tile, t1_tile_max * sizeof(uint32_t), "d_t1_keys_out_tile");
        s_malloc(stats, d_vals_in_tile,  t1_tile_max * sizeof(uint32_t), "d_t1_vals_in_tile");
        s_malloc(stats, d_vals_out_tile, t1_tile_max * sizeof(uint32_t), "d_t1_vals_out_tile");
        s_malloc(stats, d_sort_scratch,  t1_sort_bytes,                  "d_sort_scratch(t1)");

        h_keys = s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t), "h_keys(t1)", q);
        h_vals = s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t), "h_vals(t1)", q);

        auto run_tile = [&](uint64_t tile_off, uint64_t tile_n) {
            if (tile_n == 0) return;
            uint32_t const off32 = static_cast<uint32_t>(tile_off);
            uint32_t* d_vals_in_tile_local = d_vals_in_tile;
            q.parallel_for(
                sycl::range<1>{ static_cast<size_t>(tile_n) },
                [=](sycl::id<1> i) {
                    d_vals_in_tile_local[i] = off32 + uint32_t(i);
                }).wait();
            launch_sort_pairs_u32_u32(
                d_sort_scratch, t1_sort_bytes,
                d_t1_mi + tile_off, d_keys_out_tile,
                d_vals_in_tile,    d_vals_out_tile,
                tile_n, /*begin_bit=*/0, /*end_bit=*/cfg.k, q);
            q.memcpy(h_keys + tile_off, d_keys_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
            q.memcpy(h_vals + tile_off, d_vals_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
        };
        run_tile(0,            t1_tile_n0);
        run_tile(t1_tile_n0,   t1_tile_n1);

        s_free(stats, d_sort_scratch);
        s_free(stats, d_vals_out_tile);
        s_free(stats, d_vals_in_tile);
        s_free(stats, d_keys_out_tile);
        s_free(stats, d_t1_mi);
    }

    // 3-pass post-CUB (merge → gather meta) — same shape as T2 sort,
    // but T1 only has one gather stream (meta) so it's 2 passes here.
    // d_t1_keys_merged is now declared at function top (Phase 2.2 split prep).
    uint32_t* d_t1_merged_vals  = nullptr;
    s_malloc(stats, d_t1_keys_merged, cap * sizeof(uint32_t), "d_t1_keys_merged");
    s_malloc(stats, d_t1_merged_vals, cap * sizeof(uint32_t), "d_t1_merged_vals");

    if (!t1_match_sliced) {
        launch_merge_pairs_stable_2way_u32_u32(
            d_keys_out + 0,          d_vals_out + 0,          t1_tile_n0,
            d_keys_out + t1_tile_n0, d_vals_out + t1_tile_n0, t1_tile_n1,
            d_t1_keys_merged, d_t1_merged_vals, t1_count, q);
        s_free(stats, d_keys_out);
        s_free(stats, d_vals_out);
    } else {
        // Merge inputs are USM-host; the kernel reads via PCIe (sequential
        // 2-way merge → bandwidth-bound, ~3.27 GB at k=28 / ~25 GB/s ≈
        // 130 ms). Live device set during merge is just the two cap-sized
        // output buffers (d_t1_keys_merged + d_t1_merged_vals = 2080 MB).
        launch_merge_pairs_stable_2way_u32_u32(
            h_keys + 0,            h_vals + 0,            t1_tile_n0,
            h_keys + t1_tile_n0,   h_vals + t1_tile_n0,   t1_tile_n1,
            d_t1_keys_merged, d_t1_merged_vals, t1_count, q);
        s_free_host(stats, h_keys, q); h_keys = nullptr;
        s_free_host(stats, h_vals, q); h_vals = nullptr;
    }

    // Stage 4c (compact only): d_t1_keys_merged is not used by the
    // gather below (gather uses d_t1_merged_vals for indices); it is
    // only consumed by T2 match as the "d_sorted_mi" input. Park it on
    // pinned host across the gather peak so the 1040 MB doesn't coexist
    // with d_t1_merged_vals + d_t1_meta + d_t1_meta_sorted. H2D'd back
    // at T2 match entry.
    //
    // Plain mode keeps d_t1_keys_merged live across the gather peak.
    // h_keys_owned: declared at function top (non-const so start_at_t3_match can override).
    // h_keys_owned governs BOTH h_t1_keys_merged and h_t2_keys_merged
    // (they share scratch.h_keys_merged when caller provides; they're
    // also never live concurrently — h_t1 is freed before h_t2 is
    // allocated). Pool integration must NOT modify h_keys_owned, since
    // doing so breaks the downstream h_t2_keys_merged allocation logic
    // at the `h_t2_keys_merged = h_keys_owned ? malloc_host : scratch`
    // site. Instead, frees of both buffers are additionally gated on
    // scratch.pool == nullptr.
    // h_t1_keys_merged is now declared at function top (Phase 2.2 split
    // prep) so start_at_t2_match can populate it before this code runs.
    if (!scratch.plain_mode) {
        if (h_keys_owned) {
            h_t1_keys_merged = scratch.pool
                ? scratch.pool->acquire_as<uint32_t>("h_keys_merged", cap, q)
                : s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t),
                                          "h_t1_keys_merged", q);
            if (!h_t1_keys_merged) throw std::runtime_error(
                "sycl::malloc_host(h_t1_keys_merged) failed");
        } else {
            h_t1_keys_merged = scratch.h_keys_merged;
        }
        q.memcpy(h_t1_keys_merged, d_t1_keys_merged, t1_count * sizeof(uint32_t)).wait();
        s_free(stats, d_t1_keys_merged);
        d_t1_keys_merged = nullptr;
    }

    // Stage 4b (compact only): JIT H2D d_t1_meta back onto the device
    // for the gather, then free it immediately. Peak during this window:
    //   d_t1_keys_merged (1040) + d_t1_merged_vals (1040)
    //   + d_t1_meta (2080 H2D) + d_t1_meta_sorted (2080 populated)
    //   = 6240 MB — same as T2 sort's gather peak, and no longer the
    // overall bottleneck on its own.
    //
    // Plain mode: d_t1_meta is already live (never parked).
    int const t1_gather_N = scratch.plain_mode ? 1 : scratch.gather_tile_count;

    // Cheap-win reorder (k=26 saves ~72 MB / k=28 ~288 MB): for tiny
    // (and pinned-mode which inherits tiny=true), park d_t1_merged_vals
    // to host BEFORE allocating d_t1_meta. The previous order had both
    // co-resident at the H2D moment, contributing 264 + 528 = 792 MB
    // to the T1 sort phase peak at k=26 (this was the dominant Tiny
    // floor). After the reorder, d_t1_merged_vals is gone when
    // d_t1_meta is allocated, so the peak drops to d_t1_meta (528 MB)
    // + per-tile staging (~192 MB). Saves ~264 MB at k=26.
    //
    // The park is duplicated below (inside the multi-tile gather
    // branch) for backward-compat-without-this-reorder code paths;
    // the duplicated branch is guarded so it's a no-op when we've
    // already parked here.
    uint32_t* h_t1_merged_vals_pre = nullptr;  // populated by the early-park
    if (!scratch.plain_mode && t1_gather_N > 1 && scratch.tiny_mode
        && d_t1_merged_vals != nullptr) {
        h_t1_merged_vals_pre = scratch.pool
            ? scratch.pool->acquire_as<uint32_t>("h_merged_vals", cap, q)
            : s_malloc_host<uint32_t>(stats, t1_count * sizeof(uint32_t),
                                      "h_t1_merged_vals_pre", q);
        if (!h_t1_merged_vals_pre) throw std::runtime_error(
            "sycl::malloc_host(h_t1_merged_vals_pre) failed");
        q.memcpy(h_t1_merged_vals_pre, d_t1_merged_vals,
                 t1_count * sizeof(uint32_t)).wait();
        s_free(stats, d_t1_merged_vals);
        d_t1_merged_vals = nullptr;
    }

    if (!scratch.plain_mode) {
        s_malloc(stats, d_t1_meta, cap * sizeof(uint64_t), "d_t1_meta");
        q.memcpy(d_t1_meta, h_t1_meta, t1_count * sizeof(uint64_t)).wait();
        // With gather_tile_count > 1 we reuse h_t1_meta to stage the
        // sorted output (overwriting the unsorted data we just
        // rehydrated from); defer the free until after the H2D rebuild.
        if (t1_gather_N <= 1) {
            if (h_meta_owned) s_free_host(stats, h_t1_meta, q);
            h_t1_meta = nullptr;
        }
    }

    // d_t1_meta_sorted is now declared at function top (Phase 2.2 split prep).
    if (t1_gather_N <= 1) {
        s_malloc(stats, d_t1_meta_sorted, cap * sizeof(uint64_t), "d_t1_meta_sorted");
        launch_gather_u64(d_t1_meta, d_t1_merged_vals, d_t1_meta_sorted, t1_count, q);
        end_phase(p_t1_sort);
        s_free(stats, d_t1_meta);
        s_free(stats, d_t1_merged_vals);
    } else {
        // Tiled-output gather (minimal tier). Produce the sorted output
        // in N tiles, D2H each tile to h_t1_meta (overwriting the
        // unsorted data we just rehydrated from), then free the inputs
        // and rebuild the full d_t1_meta_sorted on device. Peak during
        // gather drops from
        //   d_t1_meta (2080) + d_t1_merged_vals (1040)
        //   + d_t1_meta_sorted (2080) = 5200 MB
        // to
        //   d_t1_meta (2080) + d_t1_merged_vals (1040)
        //   + d_tile (cap/N × u64 = 520 at N=4) = ~3640 MB.
        //
        // Tiny tier: park d_t1_merged_vals on host before the gather
        // loop, then H2D each tile's index slice into a small device
        // buffer. Drops the merged_vals contribution from cap × u32
        // (1040 MB at k=28) to cap/N × u32 (260 MB at N=4 / 130 MB at
        // N=8). Combined with all other tiny chunks the gather peak
        // approaches d_t1_meta's floor (~2080 MB at k=28).
        uint64_t const tile_max =
            (t1_count + uint64_t(t1_gather_N) - 1) / uint64_t(t1_gather_N);
        uint64_t* d_tile = nullptr;
        uint32_t* h_t1_merged_vals = h_t1_merged_vals_pre;  // from early park
        uint32_t* d_idx_tile       = nullptr;
        if (scratch.tiny_mode) {
            // Cheap-win reorder: the early-park above (before d_t1_meta
            // alloc) handled the host park. If we got here without one
            // (e.g., the early-park branch was skipped because
            // d_t1_merged_vals was already nullptr), fall back to the
            // original park-here logic.
            if (h_t1_merged_vals == nullptr) {
                h_t1_merged_vals = scratch.pool
                    ? scratch.pool->acquire_as<uint32_t>("h_merged_vals", cap, q)
                    : s_malloc_host<uint32_t>(stats, t1_count * sizeof(uint32_t),
                                              "h_t1_merged_vals", q);
                if (!h_t1_merged_vals)
                    throw std::runtime_error("sycl::malloc_host(h_t1_merged_vals) failed");
                q.memcpy(h_t1_merged_vals, d_t1_merged_vals,
                         t1_count * sizeof(uint32_t)).wait();
                s_free(stats, d_t1_merged_vals);
                d_t1_merged_vals = nullptr;
            }
            s_malloc(stats, d_idx_tile, tile_max * sizeof(uint32_t), "d_t1_merged_vals_tile");
        }
        s_malloc(stats, d_tile, tile_max * sizeof(uint64_t), "d_t1_meta_sorted_tile");
        for (int n = 0; n < t1_gather_N; ++n) {
            uint64_t const tile_off = uint64_t(n) * tile_max;
            if (tile_off >= t1_count) break;
            uint64_t const tile_n = std::min(tile_max, t1_count - tile_off);
            uint32_t const* src_idx = nullptr;
            if (scratch.tiny_mode) {
                q.memcpy(d_idx_tile, h_t1_merged_vals + tile_off,
                         tile_n * sizeof(uint32_t)).wait();
                src_idx = d_idx_tile;
            } else {
                src_idx = d_t1_merged_vals + tile_off;
            }
            launch_gather_u64(
                d_t1_meta, src_idx,
                d_tile, tile_n, q);
            q.memcpy(h_t1_meta + tile_off, d_tile,
                     tile_n * sizeof(uint64_t)).wait();
        }
        s_free(stats, d_tile);
        if (scratch.tiny_mode) {
            s_free(stats, d_idx_tile);
            if (!scratch.pool) s_free_host(stats, h_t1_merged_vals, q);
        }
        s_free(stats, d_t1_meta);
        if (d_t1_merged_vals) s_free(stats, d_t1_merged_vals);
        // Tiny tier: skip the full-cap d_t1_meta_sorted rehydration. The
        // sliced T2 match path (per-section meta_l/meta_r H2D) reads
        // section-sized slices from h_t1_meta directly. Saves 2080 MB of
        // device VRAM at k=28 across T2 match.
        if (!scratch.tiny_mode) {
            s_malloc(stats, d_t1_meta_sorted, cap * sizeof(uint64_t), "d_t1_meta_sorted");
            q.memcpy(d_t1_meta_sorted, h_t1_meta, t1_count * sizeof(uint64_t)).wait();
        }
        end_phase(p_t1_sort);
        // Tiny: keep h_t1_meta alive across T2 match for slicing. Free
        // happens inside the tiny T2 match block.
        // stop_after_t1_sort (Phase 2.2): keep h_t1_meta alive so the
        // caller can read sorted T1 metadata. h_meta_owned is already
        // false for the stop_after_t1_sort path (caller provides the
        // buffer), so the free below is a no-op — but the unconditional
        // h_t1_meta = nullptr would still disconnect it from the
        // caller's pointer, hence the explicit guard.
        if (!scratch.tiny_mode && !scratch.stop_after_t1_sort) {
            if (h_meta_owned) s_free_host(stats, h_t1_meta, q);
            h_t1_meta = nullptr;
        }
    }

    }  // end block-scope wrapper for non-pinned T1 sort path.

    // Pinned-mode T1 sort joins back here. h_t1_keys_merged and
    // h_t1_meta are both sorted on host-pinned at this point; the
    // d_t1_keys_merged rehydration below is a no-op for tiny / pinned
    // (the consumer-side T2 match block does its own brief rehydrate
    // and free).
    t1_sort_done:;

    // Stage 4c (compact only): H2D d_t1_keys_merged back now that T2
    // match (its consumer) is about to start. Pinned host freed after
    // H2D. Plain mode: d_t1_keys_merged is already live.
    //
    // Tiny tier: skip the rehydration. h_t1_keys_merged stays alive
    // across T2 match; the split kernel reads section_r's mi slice each
    // pass. The T2 prepare step needs mi on device for histogram counts;
    // the tiny T2 match block briefly rehydrates for prepare and frees.
    //
    // stop_after_t1_sort: also skip — sorted T1 match_info is the
    // caller's return state, host pinned must survive.
    if (!scratch.plain_mode && !scratch.tiny_mode && !scratch.stop_after_t1_sort) {
        s_malloc(stats, d_t1_keys_merged, cap * sizeof(uint32_t), "d_t1_keys_merged");
        q.memcpy(d_t1_keys_merged, h_t1_keys_merged, t1_count * sizeof(uint32_t)).wait();
        if (h_keys_owned && !scratch.pool) s_free_host(stats, h_t1_keys_merged, q);
        h_t1_keys_merged = nullptr;
    }
    }  // end of Xs+T1 if-skip (when start_at_t2_match is set, skip to here)

    // Phase 2.2 split first-half cut: when stop_after_t1_sort is set,
    // return immediately. Sorted T1 metadata + match_info are in the
    // caller-provided h_meta and h_keys_merged on host pinned;
    // result.t1_count is set. Only tiny tier is supported by this
    // initial cut.
    if (scratch.stop_after_t1_sort) {
        if (scratch.gather_tile_count <= 1) {
            throw std::runtime_error(
                "stop_after_t1_sort requires minimal or tiny tier "
                "(gather_tile_count > 1); plain/compact path never parks "
                "T1 sorted outputs to host pinned.");
        }
        if (!scratch.h_meta || !scratch.h_keys_merged) {
            throw std::runtime_error(
                "stop_after_t1_sort requires caller-provided h_meta and "
                "h_keys_merged so the sorted T1 outputs survive return");
        }
        if (!h_t1_meta || !h_t1_keys_merged) {
            throw std::runtime_error(
                "stop_after_t1_sort: T1 boundary buffers not populated "
                "(internal pipeline error)");
        }
        GpuPipelineResult result;
        result.t1_count = t1_count;
        result.t2_count = 0;
        result.t3_count = 0;
        return result;
    }

    // ---------- Phase T2 match ----------
    // Plain mode: single-pass full-cap N=1 match. Device live set
    // during match is T1 sorted (3.07 GB at k=28) + full-cap T2 output
    // (4.16 GB) ≈ 7.23 GB. No PCIe round-trips.
    //
    // Compact mode (tiled N=2, D2H per pass): two bucket-range passes
    // through half-cap device staging + pinned host accumulators. Match
    // live set drops to T1 sorted + half-cap staging ≈ 5.15 GB, at the
    // cost of ~70 ms of PCIe per pass. This is stage 3 of C (see
    // docs/t2-match-tiling-plan.md). Pool path uses the single-shot
    // launch_t2_match — it has the VRAM and doesn't pay the staging
    // round-trip cost.
    //
    // Per-pass compact safety: we expect each half to produce ≤ cap/2
    // pairs because the match output is roughly uniform across bucket
    // ids. cap itself has a built-in safety margin (see
    // extra_margin_bits in PoolSizing), and typical actual utilisation
    // is well under 100 %. If a pass ever exceeds staging capacity we
    // throw rather than silently dropping pairs.
    stats.phase = "T2 match";
    auto t2p = make_t2_params(cfg.k, cfg.strength);

    // Shared outputs. In plain mode d_t2_meta / d_t2_xbits / d_t2_mi
    // all become live full-cap buffers here; the T2 sort / gather
    // sections below skip the JIT H2D re-hydrations. In compact mode
    // only d_t2_mi is live here (hydrated from the per-plot h_t2_mi),
    // and h_t2_meta / h_t2_xbits hold the concatenated outputs on
    // pinned host until JIT H2D at the gather site.
    // d_t2_meta / d_t2_mi / d_t2_xbits / t2_count / h_t2_meta /
    // h_t2_xbits / h_xbits_owned are declared at function top so the
    // start_at_t3_match entry can populate them and skip this phase.
    if (scratch.plain_mode) {
        // Plain: one-shot launch_t2_match into full-cap device buffers.
        size_t t2_temp_bytes = 0;
        launch_t2_match(cfg.plot_id.data(), t2p, nullptr, nullptr, t1_count,
                        nullptr, nullptr, nullptr, d_counter, cap,
                        nullptr, &t2_temp_bytes, q);

        void* d_t2_match_temp = nullptr;
        s_malloc(stats, d_t2_meta,       cap * sizeof(uint64_t), "d_t2_meta");
        s_malloc(stats, d_t2_mi,         cap * sizeof(uint32_t), "d_t2_mi");
        s_malloc(stats, d_t2_xbits,      cap * sizeof(uint32_t), "d_t2_xbits");
        s_malloc(stats, d_t2_match_temp, t2_temp_bytes,          "d_t2_match_temp");

        q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        int p_t2 = begin_phase("T2 match");
        launch_t2_match(cfg.plot_id.data(), t2p,
                        d_t1_meta_sorted, d_t1_keys_merged, t1_count,
                        d_t2_meta, d_t2_mi, d_t2_xbits,
                        d_counter, cap,
                        d_t2_match_temp, &t2_temp_bytes, q);
        end_phase(p_t2);

        q.memcpy(&t2_count, d_counter, sizeof(uint64_t)).wait();
        if (t2_count > cap) throw std::runtime_error("T2 overflow");

        s_free(stats, d_t2_match_temp);
        s_free(stats, d_t1_meta_sorted);
        s_free(stats, d_t1_keys_merged);
    } else {
        // Compact: N-tile cap/N staging with pinned-host accumulators.
        // N = scratch.t2_tile_count: 2 = compact (~2.3 GB staging at
        // k=28); 8 = minimal (~570 MB) for 4 GiB cards. Must be a power
        // of 2 ≤ t2_num_buckets so even bucket distribution is exact.
        uint32_t const t2_num_buckets =
            (1u << t2p.num_section_bits) * (1u << t2p.num_match_key_bits);
        int const N = scratch.t2_tile_count;
        if (N < 2 || (N & (N - 1)) != 0) {
            throw std::runtime_error(
                "scratch.t2_tile_count must be a power of 2 ≥ 2 (got " +
                std::to_string(N) + ")");
        }
        if (static_cast<uint32_t>(N) > t2_num_buckets) {
            throw std::runtime_error(
                "scratch.t2_tile_count " + std::to_string(N) +
                " exceeds t2_num_buckets " + std::to_string(t2_num_buckets));
        }
        uint64_t const t2_tile_cap = (cap + uint64_t(N) - 1) / uint64_t(N);
        // Phase 1.6b — Tiny per-bucket-pair cap: one bucket_id per pass.
        // Halves staging vs the t2_tile_cap that the t2_tile_count=8
        // Tiny default produced.
        uint64_t const t2_bucket_pair_cap =
            ((cap + uint64_t(t2_num_buckets) - 1) / uint64_t(t2_num_buckets)) * 5ULL / 4ULL;
        uint64_t const t2_stage_cap =
            scratch.tiny_mode ? t2_bucket_pair_cap : t2_tile_cap;

        size_t t2_temp_bytes = 0;
        launch_t2_match_prepare(cfg.plot_id.data(), t2p, nullptr, t1_count,
                                d_counter, nullptr, &t2_temp_bytes, q);

        // Tile-cap device staging (reused across all passes).
        uint64_t* d_t2_meta_stage  = nullptr;
        uint32_t* d_t2_mi_stage    = nullptr;
        uint32_t* d_t2_xbits_stage = nullptr;
        void*     d_t2_match_temp  = nullptr;
        s_malloc(stats, d_t2_meta_stage,  t2_stage_cap * sizeof(uint64_t), "d_t2_meta_stage");
        s_malloc(stats, d_t2_mi_stage,    t2_stage_cap * sizeof(uint32_t), "d_t2_mi_stage");
        s_malloc(stats, d_t2_xbits_stage, t2_stage_cap * sizeof(uint32_t), "d_t2_xbits_stage");
        s_malloc(stats, d_t2_match_temp,  t2_temp_bytes,                  "d_t2_match_temp");

        // Full-cap pinned host that will hold the concatenated T2 output.
        // Stage 4f: reuse the caller-provided scratch for h_meta / h_xbits
        // (amortised across batch). h_t2_mi is still allocated per-plot.
        auto alloc_pinned_or_throw = [&](size_t bytes, char const* what) {
            return s_malloc_host_raw(stats, bytes, what, q);
        };
        // In tiny mode, h_t2_meta MUST be distinct from h_t1_meta (=
        // scratch.h_meta when caller provides). Otherwise the per-pass
        // T2 match D2H corrupts unread h_t1_meta sections at small k.
        // Caller can pre-provide scratch.h_t2_meta to avoid the per-
        // plot malloc; otherwise we allocate fresh per plot.
        if (scratch.tiny_mode) {
            if (scratch.h_t2_meta) {
                h_t2_meta = scratch.h_t2_meta;
                h_t2_meta_owned = false;
            } else {
                h_t2_meta = static_cast<uint64_t*>(
                    alloc_pinned_or_throw(cap * sizeof(uint64_t), "h_t2_meta(tiny)"));
                h_t2_meta_owned = true;
            }
        } else if (scratch.h_t2_meta && scratch.h_t2_meta != scratch.h_meta) {
            // Caller provided a distinct h_t2_meta buffer. Honour that —
            // required by the pipeline-parallel orchestrator's T2-boundary
            // protocol so the start_at_t3_match consumer can read T2 meta
            // from its dedicated buffer (the consumer prefers h_t2_meta
            // over h_meta). Without this, minimal-mode producers wrote
            // T2 meta into h_meta and left h_t2_meta untouched, causing
            // start_at_t3_match to read uninitialized memory and the
            // T3-match kernel to wedge or produce garbage.
            h_t2_meta = scratch.h_t2_meta;
            h_t2_meta_owned = false;
        } else {
            // Compact mode without a dedicated h_t2_meta buffer: T2 match
            // reads d_t1_meta_sorted on device, not h_t1_meta on host —
            // sharing scratch.h_meta with the T2 output buffer is safe.
            h_t2_meta = h_meta_owned
                ? static_cast<uint64_t*>(alloc_pinned_or_throw(cap * sizeof(uint64_t), "h_t2_meta"))
                : scratch.h_meta;
            h_t2_meta_owned = h_meta_owned;
        }
        // h_t2_mi: per-plot full-cap host pinned. Pool path amortises
        // across plots when scratch.pool is set; otherwise per-plot
        // malloc + free at line ~1996 (gated on h_t2_mi_owned).
        bool      h_t2_mi_owned = (scratch.pool == nullptr);
        uint32_t* h_t2_mi = scratch.pool
            ? scratch.pool->acquire_as<uint32_t>("h_t2_mi", cap, q)
            : static_cast<uint32_t*>(alloc_pinned_or_throw(cap * sizeof(uint32_t), "h_t2_mi"));
        h_xbits_owned = (scratch.h_t2_xbits == nullptr);
        if (scratch.spill.h_t2_xbits) {
            // Host-RAM disk-offload: redirect the ~1 GiB h_t2_xbits
            // (Compact/Minimal only — the budget policy never sets this
            // bit for Tiny, whose T2-sort partition reads it USM-host) to
            // a TempFile via the shared SpillEngine. Same class as h_t3 —
            // pure GPU-DMA through the shared 64 MiB window. h_t2_xbits
            // stays null; h_xbits_owned is forced false so the pinned-free
            // guards below skip it (the SpillBuffer is reset at the tier-
            // appropriate last-use site).
            t2_xbits_spill = std::make_unique<SpillBuffer>(
                ensure_spill_engine(), sizeof(uint32_t));
            h_t2_xbits    = nullptr;
            h_xbits_owned = false;
            std::fprintf(stderr,
                "[spill] h_t2_xbits -> disk %s (shared 64 MiB staging, %.2f GiB spilled)\n",
                t2_xbits_spill->file.path().c_str(),
                double(cap) * sizeof(uint32_t) / 1073741824.0);
        } else {
            h_t2_xbits = h_xbits_owned
                ? static_cast<uint32_t*>(alloc_pinned_or_throw(cap * sizeof(uint32_t), "h_t2_xbits"))
                : scratch.h_t2_xbits;
        }

        // Compute bucket + fine-bucket offsets once; both passes share
        // them. Also zeroes d_counter.
        //
        // Phase 1.6d — Tiny path computes offsets on host via binary
        // search on h_t1_keys_merged (sorted by mi), then H2Ds the
        // small offsets arrays (~33 KB total at k=26) directly to
        // d_t2_match_temp. Eliminates the 264 MB d_t1_keys_merged_prep
        // device spike that previously set the T2 match floor.
        uint32_t const num_buckets_t2_early =
            (1u << t2p.num_section_bits) * (1u << t2p.num_match_key_bits);
        if (scratch.tiny_mode) {
            // d_t2_match_temp layout (matches launch_t2_match_prepare):
            //   [0 .. num_buckets]      = bucket offsets (u64)
            //   [num_buckets+1 .. + fine_count + 1] = fine offsets (u64)
            // fine_count = num_buckets * 2^kT2FineBits with kT2FineBits=8.
            constexpr int kT2FineBitsLocal = 8;
            uint32_t const num_buckets_t2 = num_buckets_t2_early;
            uint32_t const fine_count =
                num_buckets_t2 * (1u << kT2FineBitsLocal);
            int const bucket_shift = t2p.num_match_target_bits;
            int const fine_shift   = t2p.num_match_target_bits - kT2FineBitsLocal;

            std::vector<uint64_t> h_bucket_off(num_buckets_t2 + 1);
            std::vector<uint64_t> h_fine_off  (fine_count + 1);

            // One linear sweep instead of num_buckets × 256 cache-
            // hostile binary searches over pinned memory: keys are
            // sorted by mi, so key >> fine_shift (the global fine
            // index = (bucket << fineBits) | fine) is non-decreasing.
            // Bucket offsets are the fine table sampled at each
            // bucket's first fine slot. Same shape as the T3 prepare.
            (void)bucket_shift;
            {
                uint64_t idx = 0;
                for (uint64_t fi = 0; fi < fine_count; ++fi) {
                    while (idx < t1_count &&
                           uint64_t(h_t1_keys_merged[idx] >> fine_shift) < fi) {
                        ++idx;
                    }
                    h_fine_off[fi] = idx;
                }
                h_fine_off[fine_count] = t1_count;
            }
            for (uint32_t b = 0; b <= num_buckets_t2; ++b) {
                h_bucket_off[b] = (b == num_buckets_t2)
                    ? t1_count
                    : h_fine_off[uint64_t(b) << kT2FineBitsLocal];
            }

            q.memcpy(d_t2_match_temp, h_bucket_off.data(),
                     (num_buckets_t2 + 1) * sizeof(uint64_t)).wait();
            q.memcpy(static_cast<uint64_t*>(d_t2_match_temp) + (num_buckets_t2 + 1),
                     h_fine_off.data(),
                     (fine_count + 1) * sizeof(uint64_t)).wait();
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        } else {
            launch_t2_match_prepare(cfg.plot_id.data(), t2p,
                                    d_t1_keys_merged,
                                    t1_count,
                                    d_counter, d_t2_match_temp, &t2_temp_bytes, q);
        }

        // Tiny mode: D2H the bucket-offsets table so we can compute each
        // section's row range host-side. Only the bucket-offsets prefix
        // is needed (fine-offsets stay on device for the kernel's binary
        // search).
        uint32_t const num_sections_t2   = 1u << t2p.num_section_bits;
        uint32_t const num_match_keys_t2 = 1u << t2p.num_match_key_bits;
        uint32_t const num_buckets_t2    = num_sections_t2 * num_match_keys_t2;
        std::vector<uint64_t> h_t2_bucket_offsets;
        if (scratch.tiny_mode) {
            h_t2_bucket_offsets.resize(num_buckets_t2 + 1);
            q.memcpy(h_t2_bucket_offsets.data(), d_t2_match_temp,
                     (num_buckets_t2 + 1) * sizeof(uint64_t)).wait();
        }

        auto compute_section_r_t2 = [&](uint32_t section_l) -> uint32_t {
            uint32_t const mask = num_sections_t2 - 1u;
            uint32_t const rl   = ((section_l << 1) |
                                   (section_l >> (t2p.num_section_bits - 1))) & mask;
            uint32_t const rl1  = (rl + 1u) & mask;
            return ((rl1 >> 1) |
                    (rl1 << (t2p.num_section_bits - 1))) & mask;
        };

        // Per-section state (tiny only): re-allocate slices when the
        // pass crosses into a new section_l. Slices stay on device for
        // all passes within a section.
        //
        // Phase 1.6b — Tiny sub-section attack on T2: only L slice is
        // cached per section_l now. R is loaded per (section_l, mk)
        // bucket-pair sub-pass since the kernel iterates one bucket_id
        // at a time. Saves R-section co-resident bytes:
        //   meta_r:  128 → 32 MB at k=26 (-96 MB)
        //   mi_r:     64 → 16 MB at k=26 (-48 MB)
        // Combined with smaller per-bucket-pair staging cap below,
        // T2 match phase live drops ~452 → ~264 MB at k=26.
        int32_t  cur_section_l = -1;
        uint64_t cur_section_l_row_start = 0;
        uint64_t* d_t2_meta_l_slice = nullptr;
        // Per-pass R bucket slices (Tiny only). Allocated ONCE at the
        // max bucket size and reused across all buckets — a
        // malloc_device/free pair per bucket is an implicitly-
        // synchronizing driver call, and at hundreds of buckets ×
        // 2 allocs it added seconds per plot on the tiny tier.
        uint64_t* d_t2_meta_r_slice = nullptr;
        uint32_t* d_t2_mi_r_slice   = nullptr;
        uint64_t cur_r_slice_row_start = 0;
        if (scratch.tiny_mode) {
            uint64_t t2_max_r_count = 0;
            for (uint32_t b = 0; b < num_buckets_t2; ++b) {
                t2_max_r_count = std::max(
                    t2_max_r_count,
                    h_t2_bucket_offsets[b + 1] - h_t2_bucket_offsets[b]);
            }
            if (t2_max_r_count > 0) {
                s_malloc(stats, d_t2_meta_r_slice,
                         t2_max_r_count * sizeof(uint64_t), "d_t2_meta_r_bucket");
                s_malloc(stats, d_t2_mi_r_slice,
                         t2_max_r_count * sizeof(uint32_t), "d_t2_mi_r_bucket");
            }
        }

        auto release_t2_slices = [&]() {
            if (d_t2_mi_r_slice)   { s_free(stats, d_t2_mi_r_slice);   d_t2_mi_r_slice   = nullptr; }
            if (d_t2_meta_r_slice) { s_free(stats, d_t2_meta_r_slice); d_t2_meta_r_slice = nullptr; }
            if (d_t2_meta_l_slice) { s_free(stats, d_t2_meta_l_slice); d_t2_meta_l_slice = nullptr; }
            cur_section_l = -1;
        };

        // Tiny: ensure L is loaded for the section_l of the bucket about
        // to be processed. L stays on device until section_l changes.
        // R is NOT loaded here — load_t2_r_bucket handles each pass's R.
        auto ensure_t2_l_slice = [&](uint32_t section_l) {
            if (static_cast<int32_t>(section_l) == cur_section_l) return;
            if (d_t2_meta_l_slice) { s_free(stats, d_t2_meta_l_slice); d_t2_meta_l_slice = nullptr; }

            cur_section_l_row_start = h_t2_bucket_offsets[section_l * num_match_keys_t2];
            uint64_t section_l_row_end =
                h_t2_bucket_offsets[(section_l + 1) * num_match_keys_t2];
            uint64_t section_l_count = section_l_row_end - cur_section_l_row_start;

            if (section_l_count > 0) {
                s_malloc(stats, d_t2_meta_l_slice, section_l_count * sizeof(uint64_t), "d_t2_meta_l_slice");
                if (t1_meta_spill) {
                    t1_meta_spill->read_to_device(
                        d_t2_meta_l_slice, cur_section_l_row_start, section_l_count);
                } else {
                    q.memcpy(d_t2_meta_l_slice, h_t1_meta + cur_section_l_row_start,
                             section_l_count * sizeof(uint64_t)).wait();
                }
            }
            cur_section_l = static_cast<int32_t>(section_l);
        };

        // Tiny: load one R bucket's worth of meta+mi from host pinned
        // into the persistent max-sized slice buffers (no per-bucket
        // allocation). r_bucket_id = section_r * num_match_keys_t2 +
        // match_key_r. Returns the row start in the original t1 stream
        // (used as the kernel's section_r_row_start = base offset for
        // r-index math).
        auto load_t2_r_bucket = [&](uint32_t r_bucket_id) -> uint64_t {
            uint64_t const r_start = h_t2_bucket_offsets[r_bucket_id];
            uint64_t const r_end   = h_t2_bucket_offsets[r_bucket_id + 1];
            uint64_t const r_count = r_end - r_start;
            cur_r_slice_row_start = r_start;
            if (r_count == 0) return r_start;
            if (t1_meta_spill) {
                t1_meta_spill->read_to_device(d_t2_meta_r_slice, r_start, r_count);
            } else {
                q.memcpy(d_t2_meta_r_slice, h_t1_meta + r_start,
                         r_count * sizeof(uint64_t)).wait();
            }
            q.memcpy(d_t2_mi_r_slice, h_t1_keys_merged + r_start,
                     r_count * sizeof(uint32_t)).wait();
            return r_start;
        };

        auto run_pass_and_stage = [&](uint32_t bucket_begin, uint32_t bucket_end,
                                      uint64_t host_offset) -> uint64_t
        {
            // Phase 1.6b — Tiny path is now per-bucket-pair, see the
            // dedicated loop below. This lambda is only called from the
            // non-tiny path.
            launch_t2_match_range(cfg.plot_id.data(), t2p,
                                  d_t1_meta_sorted, d_t1_keys_merged, t1_count,
                                  d_t2_meta_stage, d_t2_mi_stage, d_t2_xbits_stage,
                                  d_counter, t2_stage_cap, d_t2_match_temp,
                                  bucket_begin, bucket_end, q);
            uint64_t pass_count = 0;
            q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
            if (pass_count > t2_stage_cap) {
                throw std::runtime_error(
                    "T2 match pass overflow: bucket range [" +
                    std::to_string(bucket_begin) + "," + std::to_string(bucket_end) +
                    ") produced " + std::to_string(pass_count) +
                    " pairs, staging holds " + std::to_string(t2_stage_cap) +
                    " (consider lower N or fall back to compact tier).");
            }
            // Pre-copy bounds check: per-pass caps can sum past cap, and
            // the post-loop "T2 overflow" throw would arrive only after
            // the cap-sized host buffers were already overrun.
            if (host_offset + pass_count > cap) {
                throw std::runtime_error("T2 overflow (staged accumulation)");
            }
            q.memcpy(h_t2_meta  + host_offset, d_t2_meta_stage,  pass_count * sizeof(uint64_t));
            q.memcpy(h_t2_mi    + host_offset, d_t2_mi_stage,    pass_count * sizeof(uint32_t));
            if (t2_xbits_spill)  // compact/minimal spill (h_t2_meta stays pinned/aliased here)
                t2_xbits_spill->write_from_device(d_t2_xbits_stage, host_offset, pass_count);
            else
                q.memcpy(h_t2_xbits + host_offset, d_t2_xbits_stage, pass_count * sizeof(uint32_t));
            q.wait();
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();
            return pass_count;
        };

        int p_t2 = begin_phase("T2 match");
        t2_count = 0;
        if (scratch.tiny_mode) {
            // Per-bucket-pair sub-section attack. For each bucket_id,
            // ensure L is loaded for its section_l (cached across the
            // section's num_match_keys_t2 passes), copy only the matching
            // R bucket into the persistent max-sized R slice, run the
            // kernel for just this one bucket_id, accumulate, repeat.
            for (uint32_t bucket_id = 0; bucket_id < num_buckets_t2; ++bucket_id) {
                uint32_t const section_l = bucket_id / num_match_keys_t2;
                uint32_t const section_r = compute_section_r_t2(section_l);
                uint32_t const mk        = bucket_id % num_match_keys_t2;
                uint32_t const r_bucket_id = section_r * num_match_keys_t2 + mk;

                ensure_t2_l_slice(section_l);
                uint64_t const r_row_start = load_t2_r_bucket(r_bucket_id);

                if (d_t2_meta_l_slice &&
                    (d_t2_meta_r_slice || h_t2_bucket_offsets[r_bucket_id + 1] ==
                                          h_t2_bucket_offsets[r_bucket_id]))
                {
                    launch_t2_match_section_pair_split_range(
                        cfg.plot_id.data(), t2p,
                        d_t2_meta_l_slice, cur_section_l_row_start,
                        d_t2_meta_r_slice, d_t2_mi_r_slice, r_row_start,
                        d_t2_meta_stage, d_t2_mi_stage, d_t2_xbits_stage,
                        d_counter, t2_stage_cap, d_t2_match_temp,
                        bucket_id, bucket_id + 1, q);

                    uint64_t pass_count = 0;
                    q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
                    if (pass_count > t2_stage_cap) {
                        throw std::runtime_error(
                            "tiny T2 match bucket " + std::to_string(bucket_id) +
                            " produced " + std::to_string(pass_count) +
                            " pairs, staging holds " + std::to_string(t2_stage_cap));
                    }
                    if (t2_count + pass_count > cap) {
                        throw std::runtime_error("T2 overflow (staged accumulation)");
                    }
                    // Tiny T2 match keeps h_t2_meta / h_t2_xbits pinned:
                    // its Phase-1.5b T2 sort partitions them via a USM-host
                    // kernel, so they cannot live on disk. (Compact/Minimal
                    // route h_t2_xbits through t2_xbits_spill above/below.)
                    q.memcpy(h_t2_meta  + t2_count, d_t2_meta_stage,  pass_count * sizeof(uint64_t));
                    q.memcpy(h_t2_mi    + t2_count, d_t2_mi_stage,    pass_count * sizeof(uint32_t));
                    q.memcpy(h_t2_xbits + t2_count, d_t2_xbits_stage, pass_count * sizeof(uint32_t));
                    q.wait();
                    q.memset(d_counter, 0, sizeof(uint64_t)).wait();
                    t2_count += pass_count;
                }
            }
            release_t2_slices();
        } else {
            // N evenly-spaced bucket ranges. host_offset accumulates so
            // each pass appends to the pinned host buffer behind the
            // prior pass.
            for (int pass = 0; pass < N; ++pass) {
                uint32_t const bucket_begin =
                    uint32_t(uint64_t(pass)     * num_buckets_t2 / uint64_t(N));
                uint32_t const bucket_end =
                    uint32_t(uint64_t(pass + 1) * num_buckets_t2 / uint64_t(N));
                t2_count += run_pass_and_stage(bucket_begin, bucket_end,
                                               /*host_offset=*/t2_count);
            }
        }
        end_phase(p_t2);

        if (t2_count > cap) throw std::runtime_error("T2 overflow");

        // Free device staging + T1 sorted + match temp before
        // re-allocating the full-cap d_t2_mi that T2 sort expects.
        s_free(stats, d_t2_match_temp);
        s_free(stats, d_t2_meta_stage);
        s_free(stats, d_t2_mi_stage);
        s_free(stats, d_t2_xbits_stage);
        // Tiny: d_t1_meta_sorted and d_t1_keys_merged are null (parked
        // on host pinned). Free the host buffers; T2 sort below will
        // build its inputs from h_t2_meta/h_t2_mi/h_t2_xbits.
        if (scratch.tiny_mode) {
            if (h_meta_owned) s_free_host(stats, h_t1_meta, q);
            t1_meta_spill.reset();  // P1: free staging window + close/unlink temp file
            h_t1_meta = nullptr;
            if (h_keys_owned && !scratch.pool) s_free_host(stats, h_t1_keys_merged, q);
            h_t1_keys_merged = nullptr;
        } else {
            s_free(stats, d_t1_meta_sorted);
            s_free(stats, d_t1_keys_merged);
        }

        // Stage 4a: hydrate full-cap d_t2_mi from h_t2_mi. d_t2_meta
        // and d_t2_xbits are NOT hydrated yet — they stay on pinned
        // host until their gather calls at the end of T2 sort.
        s_malloc(stats, d_t2_mi, cap * sizeof(uint32_t), "d_t2_mi");
        q.memcpy(d_t2_mi, h_t2_mi, t2_count * sizeof(uint32_t));
        q.wait();
        if (h_t2_mi_owned) s_free_host(stats, h_t2_mi, q);
    }

    // ---------- Phase T2 sort (tiled, N=2) ----------
    // Mirror of T1 sort above — same tile-and-merge shape, but permute
    // writes a meta-xbits pair (T2 match output is 16 B, split SoA for
    // T3's L1-bound read pattern) instead of plain meta.
    // N=4 tiling halves the CUB scratch peak (~1044 MB → ~522 MB at
    // k=28), bringing the T2 CUB-alloc peak under 8 GB. Merge is done
    // as a tree of three 2-way merges: (0+1)→AB, (2+3)→CD, (AB+CD)→final.
    constexpr int kNumT2Tiles = 4;
    uint64_t t2_tile_n  [kNumT2Tiles];
    uint64_t t2_tile_off[kNumT2Tiles + 1];
    uint64_t const t2_base_tile = t2_count / kNumT2Tiles;
    uint64_t       t2_rem       = t2_count % kNumT2Tiles;
    t2_tile_off[0] = 0;
    for (int t = 0; t < kNumT2Tiles; ++t) {
        t2_tile_n[t]     = t2_base_tile + (t2_rem > 0 ? 1 : 0);
        if (t2_rem > 0) --t2_rem;
        t2_tile_off[t+1] = t2_tile_off[t] + t2_tile_n[t];
    }
    uint64_t t2_tile_max = 0;
    for (int t = 0; t < kNumT2Tiles; ++t)
        if (t2_tile_n[t] > t2_tile_max) t2_tile_max = t2_tile_n[t];

    size_t t2_sort_bytes = 0;
    launch_sort_pairs_u32_u32(
        nullptr, t2_sort_bytes,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        t2_tile_max, 0, cfg.k, q);

    stats.phase = "T2 sort";
    // CUB sort key input = d_t2_mi (emitted SoA by T2 match); no extract
    // needed, so d_keys_in only needs to hold the merged sorted-MI output
    // that downstream T3 match will consume. Allocate it AFTER the CUB
    // tile-sort has freed d_t2_mi to keep peak narrow.
    //
    // Compact / plain: full-cap d_keys_out + d_vals_in + d_vals_out
    // (~4168 MB peak with d_t2_mi during tile sort).
    //
    // Sliced (minimal): per-tile cap/N output buffers + USM-host
    // accumulators, then USM-host parking of AB / CD between merge
    // tree steps so the final merge sees only its own outputs +
    // USM-host inputs (live device ~2080 MB at k=28). Peaks under
    // 4 GiB at every step.

    uint64_t const ab_count = t2_tile_n[0] + t2_tile_n[1];
    uint64_t const cd_count = t2_tile_n[2] + t2_tile_n[3];

    int p_t2_sort = begin_phase("T2 sort");

    // ------------------------------------------------------------
    // Phase 1.5b — Pinned-only T2 sort via streaming partition.
    //
    // Today's tiled T2 sort gather (Tiny path, line ~2842) re-
    // allocates d_t2_meta at full cap on device (528 MB at k=26 /
    // ~2 GB at k=28) to support random-access gather. That's the
    // T2-sort phase peak and, after Phase 1.4 eliminated Xs+T1
    // contributors, the overall plot peak.
    //
    // Pinned replaces the entire CUB-tile-sort + tree-merge +
    // d_t2_meta-gather + d_t2_xbits-gather flow with:
    //   1. launch_streaming_partition_u32_u64_u32 (Phase 1.5a)
    //      reads d_t2_mi tile-by-tile and h_t2_meta + h_t2_xbits
    //      paired (so the meta+xbits pairing survives duplicate
    //      mi keys), producing bucketed host-resident output.
    //   2. Per-bucket: H2D bucket → sort (key, identity_idx)
    //      with launch_sort_pairs_u32_u32 → gather meta with
    //      launch_gather_u64 + xbits with launch_gather_u32 →
    //      D2H sorted output back into h_t2_meta + h_t2_xbits +
    //      h_t2_keys_merged at the bucket's contiguous range.
    //
    // Output contract matches Tiny exactly:
    //   h_t2_meta:        sorted in place
    //   h_t2_xbits:       sorted in place
    //   h_t2_keys_merged: sorted on host-pinned (T3 match's
    //                     tiny/pinned path consumes from here)
    //   d_t2_keys_merged: nullptr (parked on host)
    //   d_t2_meta:        nullptr (never allocated)
    //   d_t2_xbits:       nullptr (never allocated)
    //   d_merged_vals:    nullptr (never allocated; per-bucket
    //                     idx serves the same role internally)
    //
    // Per-bucket device peak: small (~30-50 MB at k=26 — sort
    // scratch + 4 small per-bucket scratches). Replaces 528 MB
    // d_t2_meta floor entirely.
    if (scratch.tiny_mode) {
        int  const t2p_num_top_bits =
            std::max(4, std::min(8, cfg.k - 18));
        int  const t2p_top_bit_offset = cfg.k - t2p_num_top_bits;
        size_t const t2p_num_buckets  = size_t{1} << t2p_num_top_bits;

        // Allocate h_t2_keys_merged via the same pool/scratch/
        // malloc_host pattern the existing code uses below — sorted
        // keys land here directly via streaming partition + per-
        // bucket sort.
        if (h_keys_owned) {
            h_t2_keys_merged = scratch.pool
                ? scratch.pool->acquire_as<uint32_t>("h_keys_merged", cap, q)
                : s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t),
                                          "h_t2_keys_merged(pinned)", q);
            if (!h_t2_keys_merged) throw std::runtime_error(
                "sycl::malloc_host(h_t2_keys_merged, pinned) failed");
        } else {
            h_t2_keys_merged = scratch.h_keys_merged;
        }

        // Temp host arenas for partition output (meta + xbits will
        // be re-emitted into h_t2_meta + h_t2_xbits during per-
        // bucket sort, overwriting the original unsorted contents).
        uint64_t* h_part_meta = s_malloc_host<uint64_t>(
            stats, cap * sizeof(uint64_t), "h_part_meta(pinned T2)", q);
        uint32_t* h_part_xbits = s_malloc_host<uint32_t>(
            stats, cap * sizeof(uint32_t), "h_part_xbits(pinned T2)", q);
        uint32_t* h_bucket_starts = s_malloc_host<uint32_t>(
            stats, (t2p_num_buckets + 1) * sizeof(uint32_t),
            "h_bucket_starts(pinned T2)", q);

        // d_t2_mi is on device at full cap here (rehydrated from
        // h_t2_mi just before this block). Use it as the partition
        // source key. Triple-val partition keeps (meta, xbits)
        // paired across duplicate mi values.
        size_t partition_scratch_bytes = 0;
        launch_streaming_partition_u32_u64_u32(
            nullptr, partition_scratch_bytes,
            d_t2_mi, h_t2_meta, h_t2_xbits,
            h_t2_keys_merged, h_part_meta, h_part_xbits, h_bucket_starts,
            t2_count, t2p_top_bit_offset, t2p_num_top_bits,
            /*tile_count=*/0, q);
        void* d_partition_scratch = nullptr;
        if (partition_scratch_bytes) {
            s_malloc(stats, d_partition_scratch,
                     partition_scratch_bytes, "d_t2_partition_scratch");
        }
        launch_streaming_partition_u32_u64_u32(
            d_partition_scratch, partition_scratch_bytes,
            d_t2_mi, h_t2_meta, h_t2_xbits,
            h_t2_keys_merged, h_part_meta, h_part_xbits, h_bucket_starts,
            t2_count, t2p_top_bit_offset, t2p_num_top_bits,
            /*tile_count=*/0, q);
        if (d_partition_scratch) s_free(stats, d_partition_scratch);

        // d_t2_mi consumed.
        s_free(stats, d_t2_mi);
        d_t2_mi = nullptr;

        // Max bucket size for per-bucket scratch sizing.
        uint32_t max_bucket = 0;
        for (size_t b = 0; b < t2p_num_buckets; ++b) {
            uint32_t const bsz = h_bucket_starts[b + 1] - h_bucket_starts[b];
            if (bsz > max_bucket) max_bucket = bsz;
        }

        // Per-bucket device scratch (reused across buckets).
        uint32_t* d_bk_in       = nullptr;
        uint32_t* d_bk_out      = nullptr;
        uint32_t* d_bidx_in     = nullptr;
        uint32_t* d_bidx_out    = nullptr;
        uint64_t* d_bmeta       = nullptr;
        uint64_t* d_bmeta_out   = nullptr;
        uint32_t* d_bxbits      = nullptr;
        uint32_t* d_bxbits_out  = nullptr;
        s_malloc(stats, d_bk_in,      max_bucket * sizeof(uint32_t), "d_t2_bk_in");
        s_malloc(stats, d_bk_out,     max_bucket * sizeof(uint32_t), "d_t2_bk_out");
        s_malloc(stats, d_bidx_in,    max_bucket * sizeof(uint32_t), "d_t2_bidx_in");
        s_malloc(stats, d_bidx_out,   max_bucket * sizeof(uint32_t), "d_t2_bidx_out");
        s_malloc(stats, d_bmeta,      max_bucket * sizeof(uint64_t), "d_t2_bmeta");
        s_malloc(stats, d_bmeta_out,  max_bucket * sizeof(uint64_t), "d_t2_bmeta_out");
        s_malloc(stats, d_bxbits,     max_bucket * sizeof(uint32_t), "d_t2_bxbits");
        s_malloc(stats, d_bxbits_out, max_bucket * sizeof(uint32_t), "d_t2_bxbits_out");

        size_t bucket_sort_bytes = 0;
        launch_sort_pairs_u32_u32(
            nullptr, bucket_sort_bytes,
            d_bk_in, d_bk_out, d_bidx_in, d_bidx_out,
            max_bucket, 0, t2p_top_bit_offset, q);
        void* d_bucket_sort_scratch = nullptr;
        if (bucket_sort_bytes) {
            s_malloc(stats, d_bucket_sort_scratch,
                     bucket_sort_bytes, "d_t2_bucket_sort_scratch");
        }

        for (size_t b = 0; b < t2p_num_buckets; ++b) {
            uint32_t const bstart = h_bucket_starts[b];
            uint32_t const bsz    = h_bucket_starts[b + 1] - bstart;
            if (bsz == 0) continue;

            // H2D bucket's triple (keys, meta, xbits) — bucketed,
            // not yet sorted within bucket.
            // All three feed kernels below (d_bk_in -> sort, d_bmeta and
            // d_bxbits -> the gathers). Out-of-order queue: wait each.
            auto e_bk    = q.memcpy(d_bk_in,  h_t2_keys_merged + bstart,
                                    bsz * sizeof(uint32_t));
            auto e_bmeta = q.memcpy(d_bmeta,  h_part_meta + bstart,
                                    bsz * sizeof(uint64_t));
            auto e_bxb   = q.memcpy(d_bxbits, h_part_xbits + bstart,
                                    bsz * sizeof(uint32_t));
            e_bk.wait();
            e_bmeta.wait();
            e_bxb.wait();

            // Init identity idx so we can recover the sort
            // permutation as d_bidx_out after sorting (keys, idx).
            launch_init_u32_identity(d_bidx_in, bsz, q);

            size_t bsb = bucket_sort_bytes;
            launch_sort_pairs_u32_u32(
                d_bucket_sort_scratch, bsb,
                d_bk_in, d_bk_out, d_bidx_in, d_bidx_out,
                bsz, 0, t2p_top_bit_offset, q);

            // Gather meta + xbits using the sort permutation.
            // d_bmeta / d_bxbits are bucket-sized on device →
            // random reads are L2-cached and fast.
            launch_gather_u64(d_bmeta,  d_bidx_out, d_bmeta_out,  bsz, q);
            launch_gather_u32(d_bxbits, d_bidx_out, d_bxbits_out, bsz, q);

            // D2H sorted triple. h_t2_keys_merged + h_t2_meta +
            // h_t2_xbits are overwritten in place — the unsorted
            // contents at [bstart..bend) are no longer needed.
            // Wait all three: the next iteration's sort and gathers
            // overwrite d_bk_out / d_bmeta_out / d_bxbits_out, so an
            // un-waited D2H here is a write-after-read hazard on data
            // that is consumed after the loop.
            auto e_kb = q.memcpy(h_t2_keys_merged + bstart, d_bk_out,
                                 bsz * sizeof(uint32_t));
            auto e_mb = q.memcpy(h_t2_meta        + bstart, d_bmeta_out,
                                 bsz * sizeof(uint64_t));
            auto e_xb = q.memcpy(h_t2_xbits       + bstart, d_bxbits_out,
                                 bsz * sizeof(uint32_t));
            e_kb.wait();
            e_mb.wait();
            e_xb.wait();
        }

        if (d_bucket_sort_scratch) s_free(stats, d_bucket_sort_scratch);
        s_free(stats, d_bk_in);
        s_free(stats, d_bk_out);
        s_free(stats, d_bidx_in);
        s_free(stats, d_bidx_out);
        s_free(stats, d_bmeta);
        s_free(stats, d_bmeta_out);
        s_free(stats, d_bxbits);
        s_free(stats, d_bxbits_out);
        s_free_host(stats, h_part_meta, q);
        s_free_host(stats, h_part_xbits, q);
        s_free_host(stats, h_bucket_starts, q);

        end_phase(p_t2_sort);
        goto t2_sort_done;
    }

    {  // Block-scope wrapper for the non-Pinned T2 sort body.
       // Goto from the Pinned branch above lands at t2_sort_done
       // below; this scope contains all the existing local
       // declarations (d_AB_keys, d_t2_meta_sorted_tile, etc.)
       // so the goto does not appear to cross any function-scope
       // variable declaration (C++ rule).
    if (!t1_match_sliced) {
        // Compact / plain — existing full-cap CUB tile sort.
        s_malloc(stats, d_keys_out,     cap * sizeof(uint32_t), "d_keys_out");
        s_malloc(stats, d_vals_in,      cap * sizeof(uint32_t), "d_vals_in");
        s_malloc(stats, d_vals_out,     cap * sizeof(uint32_t), "d_vals_out");
        s_malloc(stats, d_sort_scratch, t2_sort_bytes,          "d_sort_scratch(t2)");

        launch_init_u32_identity(d_vals_in, t2_count, q);
        for (int t = 0; t < kNumT2Tiles; ++t) {
            if (t2_tile_n[t] == 0) continue;
            uint64_t off = t2_tile_off[t];
            launch_sort_pairs_u32_u32(
                d_sort_scratch, t2_sort_bytes,
                d_t2_mi    + off, d_keys_out + off,
                d_vals_in  + off, d_vals_out + off,
                t2_tile_n[t], 0, cfg.k, q);
        }

        s_free(stats, d_sort_scratch);
        s_free(stats, d_vals_in);
        s_free(stats, d_t2_mi);
    } else {
        // Sliced — per-tile cap/N output, D2H to USM-host h_keys/h_vals.
        uint32_t* d_keys_out_tile = nullptr;
        uint32_t* d_vals_in_tile  = nullptr;
        uint32_t* d_vals_out_tile = nullptr;
        s_malloc(stats, d_keys_out_tile, t2_tile_max * sizeof(uint32_t), "d_t2_keys_out_tile");
        s_malloc(stats, d_vals_in_tile,  t2_tile_max * sizeof(uint32_t), "d_t2_vals_in_tile");
        s_malloc(stats, d_vals_out_tile, t2_tile_max * sizeof(uint32_t), "d_t2_vals_out_tile");
        s_malloc(stats, d_sort_scratch,  t2_sort_bytes,                  "d_sort_scratch(t2)");

        h_keys = s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t), "h_keys(t2)", q);
        h_vals = s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t), "h_vals(t2)", q);

        for (int t = 0; t < kNumT2Tiles; ++t) {
            uint64_t const tile_n = t2_tile_n[t];
            if (tile_n == 0) continue;
            uint64_t const tile_off = t2_tile_off[t];
            uint32_t const off32    = static_cast<uint32_t>(tile_off);
            uint32_t* d_vals_in_tile_local = d_vals_in_tile;
            q.parallel_for(
                sycl::range<1>{ static_cast<size_t>(tile_n) },
                [=](sycl::id<1> i) {
                    d_vals_in_tile_local[i] = off32 + uint32_t(i);
                }).wait();
            launch_sort_pairs_u32_u32(
                d_sort_scratch, t2_sort_bytes,
                d_t2_mi + tile_off, d_keys_out_tile,
                d_vals_in_tile,    d_vals_out_tile,
                tile_n, 0, cfg.k, q);
            q.memcpy(h_keys + tile_off, d_keys_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
            q.memcpy(h_vals + tile_off, d_vals_out_tile,
                     tile_n * sizeof(uint32_t)).wait();
        }

        s_free(stats, d_sort_scratch);
        s_free(stats, d_vals_out_tile);
        s_free(stats, d_vals_in_tile);
        s_free(stats, d_keys_out_tile);
        s_free(stats, d_t2_mi);
    }

    // Tree-of-2-way-merges: (tile 0 + tile 1) → AB, (tile 2 + tile 3) → CD,
    // then (AB + CD) → final merged stream.
    //
    // Compact: AB + CD live across the final merge → peak ~4160 MB.
    // Sliced: AB and CD parked to USM-host between tree steps so the
    // final merge sees only itself + USM-host inputs (~2080 MB peak).
    uint32_t* d_AB_keys = nullptr;
    uint32_t* d_AB_vals = nullptr;
    uint32_t* d_CD_keys = nullptr;
    uint32_t* d_CD_vals = nullptr;
    uint32_t* h_AB_keys = nullptr;
    uint32_t* h_AB_vals = nullptr;
    uint32_t* h_CD_keys = nullptr;
    uint32_t* h_CD_vals = nullptr;

    if (!t1_match_sliced) {
        s_malloc(stats, d_AB_keys, ab_count * sizeof(uint32_t), "d_t2_AB_keys");
        s_malloc(stats, d_AB_vals, ab_count * sizeof(uint32_t), "d_t2_AB_vals");
        s_malloc(stats, d_CD_keys, cd_count * sizeof(uint32_t), "d_t2_CD_keys");
        s_malloc(stats, d_CD_vals, cd_count * sizeof(uint32_t), "d_t2_CD_vals");

        if (ab_count > 0) {
            launch_merge_pairs_stable_2way_u32_u32(
                d_keys_out + t2_tile_off[0], d_vals_out + t2_tile_off[0], t2_tile_n[0],
                d_keys_out + t2_tile_off[1], d_vals_out + t2_tile_off[1], t2_tile_n[1],
                d_AB_keys, d_AB_vals, ab_count, q);
        }
        if (cd_count > 0) {
            launch_merge_pairs_stable_2way_u32_u32(
                d_keys_out + t2_tile_off[2], d_vals_out + t2_tile_off[2], t2_tile_n[2],
                d_keys_out + t2_tile_off[3], d_vals_out + t2_tile_off[3], t2_tile_n[3],
                d_CD_keys, d_CD_vals, cd_count, q);
        }

        s_free(stats, d_keys_out);
        s_free(stats, d_vals_out);
    } else {
        // AB merge: read USM-host slices, write device d_AB. Then D2H
        // to USM-host and free device.
        s_malloc(stats, d_AB_keys, ab_count * sizeof(uint32_t), "d_t2_AB_keys");
        s_malloc(stats, d_AB_vals, ab_count * sizeof(uint32_t), "d_t2_AB_vals");
        if (ab_count > 0) {
            launch_merge_pairs_stable_2way_u32_u32(
                h_keys + t2_tile_off[0], h_vals + t2_tile_off[0], t2_tile_n[0],
                h_keys + t2_tile_off[1], h_vals + t2_tile_off[1], t2_tile_n[1],
                d_AB_keys, d_AB_vals, ab_count, q);
        }
        h_AB_keys = s_malloc_host<uint32_t>(stats, ab_count * sizeof(uint32_t), "h_AB_keys", q);
        h_AB_vals = s_malloc_host<uint32_t>(stats, ab_count * sizeof(uint32_t), "h_AB_vals", q);
        if (ab_count > 0) {
            // Wait both — the s_free below releases these copies' SOURCE
            // buffers, so an un-waited copy is a use-after-free.
            auto e_abk = q.memcpy(h_AB_keys, d_AB_keys, ab_count * sizeof(uint32_t));
            auto e_abv = q.memcpy(h_AB_vals, d_AB_vals, ab_count * sizeof(uint32_t));
            e_abk.wait();
            e_abv.wait();
        }
        s_free(stats, d_AB_vals);
        s_free(stats, d_AB_keys);

        // CD merge: same shape.
        s_malloc(stats, d_CD_keys, cd_count * sizeof(uint32_t), "d_t2_CD_keys");
        s_malloc(stats, d_CD_vals, cd_count * sizeof(uint32_t), "d_t2_CD_vals");
        if (cd_count > 0) {
            launch_merge_pairs_stable_2way_u32_u32(
                h_keys + t2_tile_off[2], h_vals + t2_tile_off[2], t2_tile_n[2],
                h_keys + t2_tile_off[3], h_vals + t2_tile_off[3], t2_tile_n[3],
                d_CD_keys, d_CD_vals, cd_count, q);
        }
        h_CD_keys = s_malloc_host<uint32_t>(stats, cd_count * sizeof(uint32_t), "h_CD_keys", q);
        h_CD_vals = s_malloc_host<uint32_t>(stats, cd_count * sizeof(uint32_t), "h_CD_vals", q);
        if (cd_count > 0) {
            // Wait both — s_free below frees the sources (see AB above).
            auto e_cdk = q.memcpy(h_CD_keys, d_CD_keys, cd_count * sizeof(uint32_t));
            auto e_cdv = q.memcpy(h_CD_vals, d_CD_vals, cd_count * sizeof(uint32_t));
            e_cdk.wait();
            e_cdv.wait();
        }
        s_free(stats, d_CD_vals);
        s_free(stats, d_CD_keys);

        // h_keys + h_vals consumed by AB/CD merges — free.
        s_free_host(stats, h_keys, q); h_keys = nullptr;
        s_free_host(stats, h_vals, q); h_vals = nullptr;
    }

    // d_t2_keys_merged: merged sorted MI for T3 (declared at function top).
    uint32_t* d_merged_vals    = nullptr;   // merged sorted src indices.
    s_malloc(stats, d_t2_keys_merged, cap * sizeof(uint32_t), "d_t2_keys_merged");
    s_malloc(stats, d_merged_vals,    cap * sizeof(uint32_t), "d_merged_vals");

    if (!t1_match_sliced) {
        launch_merge_pairs_stable_2way_u32_u32(
            d_AB_keys, d_AB_vals, ab_count,
            d_CD_keys, d_CD_vals, cd_count,
            d_t2_keys_merged, d_merged_vals, t2_count, q);
        s_free(stats, d_AB_keys);
        s_free(stats, d_AB_vals);
        s_free(stats, d_CD_keys);
        s_free(stats, d_CD_vals);
    } else {
        // Final merge from USM-host inputs into device outputs.
        launch_merge_pairs_stable_2way_u32_u32(
            h_AB_keys, h_AB_vals, ab_count,
            h_CD_keys, h_CD_vals, cd_count,
            d_t2_keys_merged, d_merged_vals, t2_count, q);
        s_free_host(stats, h_AB_keys, q); h_AB_keys = nullptr;
        s_free_host(stats, h_AB_vals, q); h_AB_vals = nullptr;
        s_free_host(stats, h_CD_keys, q); h_CD_keys = nullptr;
        s_free_host(stats, h_CD_vals, q); h_CD_vals = nullptr;
    }

    // Stage 4c (compact only): d_t2_keys_merged is not consumed by the
    // gather calls below (they use d_merged_vals for indices) — it's
    // only needed later by T3 match as the sorted-MI input. Park it on
    // pinned host across the gather peak so the 1040 MB doesn't coexist
    // with d_merged_vals + d_t2_meta + d_t2_meta_sorted. H2D'd back
    // before T3 match.
    //
    // Plain mode keeps d_t2_keys_merged live across the gather peak.
    if (!scratch.plain_mode) {
        if (h_keys_owned) {
            // Pool path reuses the "h_keys_merged" slot — h_t1_keys_merged
            // was freed before this point, so the buffer is available.
            h_t2_keys_merged = scratch.pool
                ? scratch.pool->acquire_as<uint32_t>("h_keys_merged", cap, q)
                : s_malloc_host<uint32_t>(stats, cap * sizeof(uint32_t),
                                          "h_t2_keys_merged", q);
        } else {
            h_t2_keys_merged = scratch.h_keys_merged;
        }
        if (!h_t2_keys_merged) throw std::runtime_error("sycl::malloc_host(h_t2_keys_merged) failed");
        q.memcpy(h_t2_keys_merged, d_t2_keys_merged, t2_count * sizeof(uint32_t)).wait();
        s_free(stats, d_t2_keys_merged);
        d_t2_keys_merged = nullptr;
    }

    // Stage 4a (compact only): JIT H2D the gather source buffers.
    // d_t2_meta is alive only for the duration of its gather (2080 MB
    // at k=28), then freed before d_t2_xbits is H2D'd. With stage 4c
    // the gather peak drops to d_merged_vals (1040) + d_t2_meta (2080)
    // + d_t2_meta_sorted (2080) = 5200 MB (no more d_t2_keys_merged).
    //
    // Plain mode: d_t2_meta and d_t2_xbits are already live from T2
    // match (never parked). Gather reads them directly and frees after.
    int const t2_gather_N = scratch.plain_mode ? 1 : scratch.gather_tile_count;

    if (t2_gather_N <= 1) {
        // Single-shot path (compact / plain).
        if (!scratch.plain_mode) {
            s_malloc(stats, d_t2_meta, cap * sizeof(uint64_t), "d_t2_meta");
            q.memcpy(d_t2_meta, h_t2_meta, t2_count * sizeof(uint64_t));
            q.wait();
            if (h_meta_owned) s_free_host(stats, h_t2_meta, q);
            h_t2_meta = nullptr;
        }

        s_malloc(stats, d_t2_meta_sorted, cap * sizeof(uint64_t), "d_t2_meta_sorted");
        launch_gather_u64(d_t2_meta, d_merged_vals, d_t2_meta_sorted, t2_count, q);
        q.wait();
        s_free(stats, d_t2_meta);

        if (!scratch.plain_mode) {
            s_malloc(stats, d_t2_xbits, cap * sizeof(uint32_t), "d_t2_xbits");
            if (t2_xbits_spill) {  // compact spill: disk -> device, then release
                t2_xbits_spill->read_to_device(d_t2_xbits, 0, t2_count);
                t2_xbits_spill.reset();
            } else {
                q.memcpy(d_t2_xbits, h_t2_xbits, t2_count * sizeof(uint32_t));
                q.wait();
                if (h_xbits_owned) s_free_host(stats, h_t2_xbits, q);
            }
            h_t2_xbits = nullptr;
        }

        s_malloc(stats, d_t2_xbits_sorted, cap * sizeof(uint32_t), "d_t2_xbits_sorted");
        launch_gather_u32(d_t2_xbits, d_merged_vals, d_t2_xbits_sorted, t2_count, q);
        end_phase(p_t2_sort);
        s_free(stats, d_t2_xbits);
        s_free(stats, d_merged_vals);
    } else {
        // Tiled-output gather (minimal tier). Both gathers stage their
        // sorted outputs to host pinned (reusing h_t2_meta and
        // h_t2_xbits — same buffers that just held the parked unsorted
        // data) one tile at a time. Crucially, d_t2_meta_sorted is NOT
        // re-allocated on device until BOTH gathers and d_merged_vals
        // are done — otherwise the xbits gather peak (d_t2_meta_sorted
        // 2080 + d_merged_vals 1040 + d_t2_xbits 1040 + tile 260) would
        // still hit ~4420 MB. Deferring the rehydrate keeps the xbits
        // gather peak at d_merged_vals (1040) + d_t2_xbits (1040) +
        // tile (260 at N=4) = ~2340 MB. Final rehydrate peak:
        // d_t2_meta_sorted (2080) + d_t2_xbits_sorted (1040) = 3120 MB.
        //
        // Tiny tier: also park d_merged_vals on host before the
        // gathers, then H2D each tile's index slice into a small
        // device buffer (reused across both gather passes). Drops the
        // merged_vals contribution from cap × u32 (1040 MB at k=28) to
        // tile cap × u32 (260 MB at N=4 / 130 MB at N=8) for the
        // duration of each pass.
        uint64_t const tile_max =
            (t2_count + uint64_t(t2_gather_N) - 1) / uint64_t(t2_gather_N);

        uint32_t* h_t2_merged_vals = nullptr;
        uint32_t* d_t2_idx_tile    = nullptr;
        if (scratch.tiny_mode) {
            // Reuse the "h_merged_vals" pool slot — h_t1_merged_vals
            // freed/returned before this point; never live concurrently.
            h_t2_merged_vals = scratch.pool
                ? scratch.pool->acquire_as<uint32_t>("h_merged_vals", cap, q)
                : s_malloc_host<uint32_t>(stats, t2_count * sizeof(uint32_t),
                                          "h_t2_merged_vals", q);
            if (!h_t2_merged_vals)
                throw std::runtime_error("sycl::malloc_host(h_t2_merged_vals) failed");
            q.memcpy(h_t2_merged_vals, d_merged_vals,
                     t2_count * sizeof(uint32_t)).wait();
            s_free(stats, d_merged_vals);
            d_merged_vals = nullptr;
            s_malloc(stats, d_t2_idx_tile, tile_max * sizeof(uint32_t), "d_t2_merged_vals_tile");
        }

        // --- Meta gather (tiled output → h_t2_meta) ---
        s_malloc(stats, d_t2_meta, cap * sizeof(uint64_t), "d_t2_meta");
        q.memcpy(d_t2_meta, h_t2_meta, t2_count * sizeof(uint64_t)).wait();
        {
            uint64_t* d_meta_tile = nullptr;
            s_malloc(stats, d_meta_tile, tile_max * sizeof(uint64_t), "d_t2_meta_sorted_tile");
            for (int n = 0; n < t2_gather_N; ++n) {
                uint64_t const tile_off = uint64_t(n) * tile_max;
                if (tile_off >= t2_count) break;
                uint64_t const tile_n = std::min(tile_max, t2_count - tile_off);
                uint32_t const* idx_src = nullptr;
                if (scratch.tiny_mode) {
                    q.memcpy(d_t2_idx_tile, h_t2_merged_vals + tile_off,
                             tile_n * sizeof(uint32_t)).wait();
                    idx_src = d_t2_idx_tile;
                } else {
                    idx_src = d_merged_vals + tile_off;
                }
                launch_gather_u64(
                    d_t2_meta, idx_src,
                    d_meta_tile, tile_n, q);
                q.memcpy(h_t2_meta + tile_off, d_meta_tile,
                         tile_n * sizeof(uint64_t)).wait();
            }
            s_free(stats, d_meta_tile);
        }
        s_free(stats, d_t2_meta);

        // --- Xbits gather (tiled output → h_t2_xbits) ---
        s_malloc(stats, d_t2_xbits, cap * sizeof(uint32_t), "d_t2_xbits");
        if (t2_xbits_spill)  // minimal/tiny spill: whole unsorted xbits disk -> device
            t2_xbits_spill->read_to_device(d_t2_xbits, 0, t2_count);
        else
            q.memcpy(d_t2_xbits, h_t2_xbits, t2_count * sizeof(uint32_t)).wait();
        {
            uint32_t* d_xbits_tile = nullptr;
            s_malloc(stats, d_xbits_tile, tile_max * sizeof(uint32_t), "d_t2_xbits_sorted_tile");
            for (int n = 0; n < t2_gather_N; ++n) {
                uint64_t const tile_off = uint64_t(n) * tile_max;
                if (tile_off >= t2_count) break;
                uint64_t const tile_n = std::min(tile_max, t2_count - tile_off);
                uint32_t const* idx_src = nullptr;
                if (scratch.tiny_mode) {
                    q.memcpy(d_t2_idx_tile, h_t2_merged_vals + tile_off,
                             tile_n * sizeof(uint32_t)).wait();
                    idx_src = d_t2_idx_tile;
                } else {
                    idx_src = d_merged_vals + tile_off;
                }
                launch_gather_u32(
                    d_t2_xbits, idx_src,
                    d_xbits_tile, tile_n, q);
                if (t2_xbits_spill)
                    t2_xbits_spill->write_from_device(d_xbits_tile, tile_off, tile_n);
                else
                    q.memcpy(h_t2_xbits + tile_off, d_xbits_tile,
                             tile_n * sizeof(uint32_t)).wait();
            }
            s_free(stats, d_xbits_tile);
        }
        s_free(stats, d_t2_xbits);

        // d_merged_vals dead now that both gathers have produced their
        // sorted outputs on host.
        if (scratch.tiny_mode) {
            s_free(stats, d_t2_idx_tile);
            if (!scratch.pool) s_free_host(stats, h_t2_merged_vals, q);
        }
        if (d_merged_vals) s_free(stats, d_merged_vals);

        // Rehydrate d_t2_xbits_sorted to device (1040 MB at k=28). The
        // T3 match kernel reads d_sorted_xbits[l] / d_sorted_xbits[r]
        // by index and the random-access pattern would be too slow via
        // PCIe with USM-host.
        //
        // Tiny tier: skip the rehydration. h_t2_xbits stays alive across
        // T3 match, and the per-section split kernel H2Ds the section-l
        // + section-r slices into small device buffers each pass. d_t2_
        // xbits_sorted remains nullptr in this path; the T3 match block
        // skips its s_free below.
        //
        // Phase 2 split (stop_after_t2_sort): also skip — h_t2_xbits
        // is what the second-half receiver reads. The second-half
        // start_at_t3_match path rehydrates d_t2_xbits_sorted there
        // when running in minimal mode.
        if (!scratch.tiny_mode && !scratch.stop_after_t2_sort) {
            s_malloc(stats, d_t2_xbits_sorted, cap * sizeof(uint32_t), "d_t2_xbits_sorted");
            if (t2_xbits_spill) {  // minimal spill: sorted xbits disk -> device, then release
                t2_xbits_spill->read_to_device(d_t2_xbits_sorted, 0, t2_count);
                t2_xbits_spill.reset();
            } else {
                q.memcpy(d_t2_xbits_sorted, h_t2_xbits, t2_count * sizeof(uint32_t)).wait();
                if (h_xbits_owned) s_free_host(stats, h_t2_xbits, q);
            }
            h_t2_xbits = nullptr;
        }

        // Site 4: do NOT rehydrate d_t2_meta_sorted to device. h_t2_meta
        // (now containing the sorted meta) stays alive across T3 match;
        // the sliced T3 match path H2Ds a section_l + section_r pair of
        // slices per pass, dropping T3 match peak from
        //   d_t2_meta_sorted (2080) + d_t2_xbits_sorted (1040) +
        //   d_t2_keys_merged (1040) + d_t3_stage (1040) = 5200 MB
        // to
        //   d_meta_l (cap/N_sections × u64 = 520) + d_meta_r (520) +
        //   d_t2_xbits_sorted (1040) + d_t2_keys_merged (1040) +
        //   d_t3_stage (cap/N_sections × u64 = 520) = ~3640 MB at k=28.
        // h_t2_meta is freed inside the T3 match block once all
        // section-pair passes complete.

        end_phase(p_t2_sort);
    }
    }  // close non-pinned T2 sort scope-wrapper.

    t2_sort_done:;
    }  // end of first-half (Xs+T1+T2) scope.

    // Phase 2 (pipeline-parallel) first-half cut: when
    // stop_after_t2_sort is set, return immediately. The sorted T2
    // outputs are in h_t2_meta / h_t2_xbits / h_t2_keys_merged on host
    // pinned; the caller owns those buffers and must hand them to the
    // second-half entry point on the receiving GPU. Requires
    // gather_tile_count > 1 (minimal/tiny path), since the compact
    // path frees h_t2_meta during T2 sort gather.
    if (scratch.stop_after_t2_sort) {
        if (scratch.gather_tile_count <= 1) {
            throw std::runtime_error(
                "StreamingPinnedScratch::stop_after_t2_sort requires "
                "gather_tile_count > 1 (minimal or tiny tier); "
                "compact path frees h_t2_meta during T2 sort gather.");
        }
        // d_t2_keys_merged was parked on host already (line ~2055)
        // for non-plain modes. Verify h_t2_meta and h_t2_xbits are
        // also alive — they should be in minimal/tiny mode.
        if (!h_t2_meta || !h_t2_xbits || !h_t2_keys_merged) {
            throw std::runtime_error(
                "stop_after_t2_sort: T2 boundary buffers not populated "
                "(h_t2_meta / h_t2_xbits / h_t2_keys_merged). Internal "
                "logic error.");
        }
        s_free(stats, d_counter);

        // Phase 2-A defense: drain the queue before handing the boundary
        // buffers to the receiver. The T2-sort gather phase issues D2H
        // copies into h_t2_meta / h_t2_xbits / h_t2_keys_merged with
        // intermediate q.wait()s inside, but a final wait here is a
        // cheap, explicit guarantee that *every* host-pinned write is
        // drained before the orchestrator's SlotChannel.send() makes the
        // slot visible to the second-half thread on dev_second. Without
        // this, any straggling async copy could race the receiver's
        // first H2D / read on the same buffer.
        q.wait();

        report_phases();
        if (stats.verbose) {
            std::fprintf(stderr,
                "[streaming first-half] k=%d strength=%d  peak device VRAM = %.2f MB\n",
                cfg.k, cfg.strength, stats.peak / (1024.0 * 1024.0));
        }

        GpuPipelineResult result;
        result.t1_count = t1_count;
        result.t2_count = t2_count;
        result.t3_count = 0;  // T3 not run in first half
        // Caller takes ownership of h_t2_meta / h_t2_xbits /
        // h_t2_keys_merged; we do NOT free them here.
        return result;
    }

t3_match_entry:
    // ---------- Phase T3 match ----------
    // Plain mode: one-shot launch_t3_match writing directly into
    // full-cap d_t3. No pinned-host staging, no round-trips — saves
    // the per-plot sycl::malloc_host(2 GB) (~500 ms on NVIDIA) plus
    // the two D2H halves + H2D re-hydration. Match live set:
    //   d_t2_keys_merged (1040) + d_t2_meta_sorted (2080)
    //   + d_t2_xbits_sorted (1040) + d_t3 (2080) + temp
    //   = ~6240 MB — fits under plain's 7290 MB T2-match floor.
    //
    // Compact mode (stage 4d.3, N=2 tiled): half-cap d_t3 staging +
    // D2H-to-pinned-host between passes, then full-cap d_t3 + H2D
    // before T3 sort. Keeps T3 match peak at 5200 MB.
    stats.phase = "T3 match";
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    size_t t3_temp_bytes = 0;
    launch_t3_match_prepare(cfg.plot_id.data(), t3p, nullptr, t2_count,
                            d_counter, nullptr, &t3_temp_bytes, q);

    // Stage 4c (compact only): H2D d_t2_keys_merged back from pinned
    // host now that we're about to enter T3 match (its consumer).
    // Pinned host freed after H2D. Plain mode: d_t2_keys_merged is
    // already live (never parked).
    //
    // Tiny tier: skip the rehydration. h_t2_keys_merged stays alive
    // across T3 match; the split kernel reads section_r's mi slice
    // each pass (d_t2_keys_merged is only used as the binary-search
    // and r-stream input, both indexed within section_r's row range).
    if (!scratch.plain_mode && !scratch.tiny_mode) {
        s_malloc(stats, d_t2_keys_merged, cap * sizeof(uint32_t), "d_t2_keys_merged");
        q.memcpy(d_t2_keys_merged, h_t2_keys_merged, t2_count * sizeof(uint32_t)).wait();
        if (h_keys_owned && !scratch.pool) s_free_host(stats, h_t2_keys_merged, q);
        h_t2_keys_merged = nullptr;
    }

    T3PairingGpu* d_t3    = nullptr;
    uint64_t      t3_count = 0;
    // Tiny mode plumbing: when the minimal/tiny T3 match block parks
    // its concatenated output on host pinned, tiny additionally keeps
    // h_t3 alive across T3 sort instead of rehydrating d_t3 to full
    // cap (saves 2080 MB at k=28). The T3 sort phase below reads the
    // input directly from this host buffer.
    T3PairingGpu* scratch_tiny_h_t3        = nullptr;
    bool          scratch_tiny_h_t3_owned  = false;

    if (scratch.plain_mode) {
        // Plain: one-shot full-cap T3 match.
        void* d_t3_match_temp = nullptr;
        s_malloc(stats, d_t3,            cap * sizeof(T3PairingGpu), "d_t3");
        s_malloc(stats, d_t3_match_temp, t3_temp_bytes,              "d_t3_match_temp");

        q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        int p_t3 = begin_phase("T3 match + Feistel");
        launch_t3_match(cfg.plot_id.data(), t3p,
                        d_t2_meta_sorted, d_t2_xbits_sorted,
                        d_t2_keys_merged, t2_count,
                        d_t3, d_counter, cap,
                        d_t3_match_temp, &t3_temp_bytes, q);
        end_phase(p_t3);

        q.memcpy(&t3_count, d_counter, sizeof(uint64_t)).wait();
        if (t3_count > cap) throw std::runtime_error("T3 overflow");

        s_free(stats, d_t3_match_temp);
        s_free(stats, d_t2_meta_sorted);
        s_free(stats, d_t2_xbits_sorted);
        s_free(stats, d_t2_keys_merged);
    } else if (scratch.gather_tile_count > 1) {
        // Minimal (sliced T3 match — site 4). d_t2_meta_sorted is NOT
        // on device in this path; the sorted meta is parked on
        // h_t2_meta (from the T2 sort tiled gather). For each section_l
        // we H2D the matching pair of sections (l + r) into small
        // device slices, run the kernel against those slices, D2H the
        // stage output to h_t3, then free the slices. Drops T3 match
        // peak from ~5200 MB (compact) to ~3665 MB at k=28.
        //
        // Tiny mode: also park d_t2_xbits_sorted and d_t2_keys_merged
        // on host pinned (they were never rehydrated above). Per
        // section, allocate xbits + mi slices alongside meta slices and
        // call the fully-sliced split kernel. Drops T3 match peak by
        // an additional ~2080 MB at k=28.
        uint32_t const num_sections   = 1u << t3p.num_section_bits;
        uint32_t const num_match_keys = 1u << t3p.num_match_key_bits;
        uint32_t const num_buckets_t3 = num_sections * num_match_keys;
        // Per-pass output capacity sized at cap/N × 1.5 (50% safety
        // margin over the expected uniform-distribution average).
        // T3 input is already plot-id-shaped meta — its per-section
        // populations skew further from uniform than T1's xbit-derived
        // sections, so it gets the larger margin (T1 stays at 25%).
        // The +20% bump in d_t3_stage VRAM is the price of letting the
        // sliced/minimal path tolerate skewed plot_id seeds at small k
        // and unblocks pipeline-parallel + minimal-tier composition.
        uint64_t const t3_section_cap =
            ((cap + num_sections - 1) / num_sections) * 3ULL / 2ULL;

        // Phase 1.5c — Pinned tier only. d_t3_stage (T3 match output
        // staging buffer, 198 MB at k=26 / ~800 MB at k=28) goes on
        // host-pinned. The match kernel's writes are atomic-claim
        // slot allocation — sequential by slot, written via USM-host
        // pointer (each write → PCIe transaction). Measured at k=26:
        // -110 MB overall plot peak, +18% wall. The wall cost is
        // intended for Pinned (target hardware can't run any other
        // tier anyway); on the dev box's 4090 it's just slower.
        //
        // Per pass:
        //   match writes (~cap/N_sections × T3PairingGpu) via PCIe
        //   q.memcpy h_t3 ← d_t3_stage becomes host→host (effectively
        //     a sequential memcpy, fast).
        //
        // Free path is symmetric — when host-pinned, sycl::free not
        // s_free so the StreamingStats tracker stays accurate.
        T3PairingGpu* d_t3_stage      = nullptr;
        void*         d_t3_match_temp = nullptr;
        bool          d_t3_stage_on_host = scratch.tiny_mode;
        if (d_t3_stage_on_host) {
            d_t3_stage = s_malloc_host<T3PairingGpu>(
                stats, t3_section_cap * sizeof(T3PairingGpu),
                "d_t3_stage(pinned T3)", q);
        } else {
            s_malloc(stats, d_t3_stage, t3_section_cap * sizeof(T3PairingGpu), "d_t3_stage");
        }
        s_malloc(stats, d_t3_match_temp, t3_temp_bytes,                          "d_t3_match_temp");

        bool const h_t3_owned   = (scratch.h_t3 == nullptr);
        // Host-RAM disk-offload: when the budget policy selected h_t3 and we
        // own it, redirect this ~2 GiB table to a TempFile via the shared
        // SpillEngine. h_t3 stays null; t3_spill services every access.
        bool const h_t3_spilled = h_t3_owned && scratch.spill.h_t3;
        T3PairingGpu* h_t3 = nullptr;
        if (h_t3_spilled) {
            t3_spill = std::make_unique<SpillBuffer>(
                ensure_spill_engine(), sizeof(T3PairingGpu));
            std::fprintf(stderr,
                "[spill] h_t3 -> disk %s (shared 64 MiB staging, %.2f GiB spilled)\n",
                t3_spill->file.path().c_str(),
                cap * sizeof(T3PairingGpu) / 1073741824.0);
        } else if (h_t3_owned) {
            h_t3 = scratch.pool
                ? scratch.pool->acquire_as<T3PairingGpu>("h_t3", cap, q)
                : s_malloc_host<T3PairingGpu>(stats, cap * sizeof(T3PairingGpu),
                                              "h_t3", q);
        } else {
            h_t3 = reinterpret_cast<T3PairingGpu*>(scratch.h_t3);
        }
        if (!h_t3 && !t3_spill) throw std::runtime_error("sycl::malloc_host(h_t3) failed");

        // Compute bucket + fine-bucket offsets in d_t3_match_temp; also
        // zero d_counter. Same call shape as compact path.
        //
        // Tiny mode: d_t2_keys_merged is parked. The prepare kernel
        // needs the sorted mi stream on device for its histogram
        // counts. Briefly rehydrate, run prepare, then free again. The
        // 1040 MB spike is bounded to this prepare phase; the subsequent
        // per-section loop reads only sliced mi.
        // Phase 1.6d (T3) — Tiny path computes T3 prepare offsets on
        // host via binary search on h_t2_keys_merged (sorted by mi).
        // Eliminates the cap-sized d_t2_keys_merged_prep device spike
        // (264 MB at k=26 / 1 GB at k=28) that previously bounded
        // T3 match phase from above.
        if (scratch.tiny_mode) {
            constexpr int kT3FineBitsLocal = 8;
            uint32_t const num_buckets_t3_p =
                (1u << t3p.num_section_bits) * (1u << t3p.num_match_key_bits);
            uint32_t const fine_count =
                num_buckets_t3_p * (1u << kT3FineBitsLocal);
            int const bucket_shift = t3p.num_match_target_bits;
            int const fine_shift   = t3p.num_match_target_bits - kT3FineBitsLocal;

            std::vector<uint64_t> h_bucket_off(num_buckets_t3_p + 1);
            std::vector<uint64_t> h_fine_off  (fine_count + 1);

            // One linear sweep instead of num_buckets × 256 cache-
            // hostile binary searches over pinned memory: keys are
            // sorted by mi, so key >> fine_shift (the global fine
            // index = (bucket << fineBits) | fine) is non-decreasing
            // and every lower bound falls out of one advancing cursor.
            // Bucket offsets are then just the fine table sampled at
            // each bucket's first fine slot.
            (void)bucket_shift;
            {
                uint64_t idx = 0;
                for (uint64_t fi = 0; fi < fine_count; ++fi) {
                    while (idx < t2_count &&
                           uint64_t(h_t2_keys_merged[idx] >> fine_shift) < fi) {
                        ++idx;
                    }
                    h_fine_off[fi] = idx;
                }
                h_fine_off[fine_count] = t2_count;
            }
            for (uint32_t b = 0; b <= num_buckets_t3_p; ++b) {
                h_bucket_off[b] = (b == num_buckets_t3_p)
                    ? t2_count
                    : h_fine_off[uint64_t(b) << kT3FineBitsLocal];
            }

            q.memcpy(d_t3_match_temp, h_bucket_off.data(),
                     (num_buckets_t3_p + 1) * sizeof(uint64_t)).wait();
            q.memcpy(static_cast<uint64_t*>(d_t3_match_temp) + (num_buckets_t3_p + 1),
                     h_fine_off.data(),
                     (fine_count + 1) * sizeof(uint64_t)).wait();
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();
        } else {
            launch_t3_match_prepare(cfg.plot_id.data(), t3p,
                                    d_t2_keys_merged,
                                    t2_count,
                                    d_counter, d_t3_match_temp, &t3_temp_bytes, q);
        }

        // D2H the bucket-offsets table (small: 17 × u64 at k=28
        // strength=2) so we can compute each section's global row range
        // host-side.
        std::vector<uint64_t> h_t3_offsets(num_buckets_t3 + 1);
        q.memcpy(h_t3_offsets.data(), d_t3_match_temp,
                 (num_buckets_t3 + 1) * sizeof(uint64_t)).wait();

        auto compute_section_r = [&](uint32_t section_l) -> uint32_t {
            // Mirror the kernel's section_l → section_r permutation.
            uint32_t const mask = num_sections - 1u;
            uint32_t const rl   = ((section_l << 1) |
                                   (section_l >> (t3p.num_section_bits - 1))) & mask;
            uint32_t const rl1  = (rl + 1u) & mask;
            return ((rl1 >> 1) |
                    (rl1 << (t3p.num_section_bits - 1))) & mask;
        };

        int p_t3 = begin_phase("T3 match + Feistel");
        uint64_t host_offset = 0;

        // Phase 1.6c — Tiny T3 sub-section attack: iterate one
        // bucket_id = (section_l, match_key_r) at a time. L slices
        // (meta_l + xbits_l) are cached per section_l, R slices
        // (meta_r + xbits_r + mi_r) are re-loaded per pass at
        // R-bucket granularity. Saves R-section co-resident bytes:
        //   meta_r:    128 → 32 MB at k=26 (-96 MB)
        //   xbits_r:    64 → 16 MB at k=26 (-48 MB)
        //   mi_r:       64 → 16 MB at k=26 (-48 MB)
        // T3 match phase live drops 448 → ~256 MB at k=26.
        if (scratch.tiny_mode) {
            // Per-section_l L cache.
            int32_t cur_section_l = -1;
            uint64_t  cur_l_row_start = 0;
            uint64_t* d_meta_l_slice  = nullptr;
            uint32_t* d_xbits_l_slice = nullptr;
            auto release_l = [&]() {
                if (d_xbits_l_slice) { s_free(stats, d_xbits_l_slice); d_xbits_l_slice = nullptr; }
                if (d_meta_l_slice)  { s_free(stats, d_meta_l_slice);  d_meta_l_slice  = nullptr; }
                cur_section_l = -1;
            };
            auto ensure_l = [&](uint32_t section_l) {
                if (static_cast<int32_t>(section_l) == cur_section_l) return;
                release_l();
                cur_l_row_start = h_t3_offsets[section_l * num_match_keys];
                uint64_t const l_end = h_t3_offsets[(section_l + 1) * num_match_keys];
                uint64_t const l_count = l_end - cur_l_row_start;
                if (l_count > 0) {
                    s_malloc(stats, d_meta_l_slice,
                             l_count * sizeof(uint64_t), "d_t3_meta_l_slice");
                    s_malloc(stats, d_xbits_l_slice,
                             l_count * sizeof(uint32_t), "d_t3_xbits_l_slice");
                    q.memcpy(d_meta_l_slice, h_t2_meta + cur_l_row_start,
                             l_count * sizeof(uint64_t)).wait();
                    q.memcpy(d_xbits_l_slice, h_t2_xbits + cur_l_row_start,
                             l_count * sizeof(uint32_t)).wait();
                }
                cur_section_l = static_cast<int32_t>(section_l);
            };

            // Persistent max-sized R-bucket slices, reused across all
            // buckets — same allocation-churn fix as the tiny T2 loop
            // above (per-bucket malloc_device is an implicitly-
            // synchronizing driver call; ×3 allocs × num_buckets it
            // added seconds per plot).
            uint64_t t3_max_r_count = 0;
            for (uint32_t b = 0; b < num_buckets_t3; ++b) {
                t3_max_r_count = std::max(
                    t3_max_r_count, h_t3_offsets[b + 1] - h_t3_offsets[b]);
            }
            uint64_t* d_meta_r_slice  = nullptr;
            uint32_t* d_xbits_r_slice = nullptr;
            uint32_t* d_mi_r_slice    = nullptr;
            if (t3_max_r_count > 0) {
                s_malloc(stats, d_meta_r_slice,
                         t3_max_r_count * sizeof(uint64_t), "d_t3_meta_r_bucket");
                s_malloc(stats, d_xbits_r_slice,
                         t3_max_r_count * sizeof(uint32_t), "d_t3_xbits_r_bucket");
                s_malloc(stats, d_mi_r_slice,
                         t3_max_r_count * sizeof(uint32_t), "d_t3_mi_r_bucket");
            }

            for (uint32_t bucket_id = 0; bucket_id < num_buckets_t3; ++bucket_id) {
                uint32_t const section_l   = bucket_id / num_match_keys;
                uint32_t const section_r   = compute_section_r(section_l);
                uint32_t const mk          = bucket_id % num_match_keys;
                uint32_t const r_bucket_id = section_r * num_match_keys + mk;

                uint64_t const l_count =
                    h_t3_offsets[section_l * num_match_keys + num_match_keys] -
                    h_t3_offsets[section_l * num_match_keys];
                if (l_count == 0) continue;
                ensure_l(section_l);

                uint64_t const r_row_start = h_t3_offsets[r_bucket_id];
                uint64_t const r_row_end   = h_t3_offsets[r_bucket_id + 1];
                uint64_t const r_count     = r_row_end - r_row_start;

                if (r_count > 0) {
                    q.memcpy(d_meta_r_slice, h_t2_meta + r_row_start,
                             r_count * sizeof(uint64_t)).wait();
                    q.memcpy(d_xbits_r_slice, h_t2_xbits + r_row_start,
                             r_count * sizeof(uint32_t)).wait();
                    q.memcpy(d_mi_r_slice, h_t2_keys_merged + r_row_start,
                             r_count * sizeof(uint32_t)).wait();

                    launch_t3_match_section_pair_split_range(
                        cfg.plot_id.data(), t3p,
                        d_meta_l_slice, d_xbits_l_slice, cur_l_row_start,
                        d_meta_r_slice, d_xbits_r_slice, d_mi_r_slice, r_row_start,
                        d_t3_stage, d_counter, t3_section_cap,
                        d_t3_match_temp, bucket_id, bucket_id + 1, q);

                    uint64_t pass_count = 0;
                    q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
                    if (pass_count > t3_section_cap) {
                        throw std::runtime_error(
                            "T3 match (sub-section) bucket_id=" +
                            std::to_string(bucket_id) +
                            " produced " + std::to_string(pass_count) +
                            " pairs, staging holds " + std::to_string(t3_section_cap));
                    }
                    if (host_offset + pass_count > cap) {
                        throw std::runtime_error("T3 overflow (staged accumulation)");
                    }
                    if (t3_spill)
                        t3_spill->write_from_device(d_t3_stage, host_offset, pass_count);
                    else
                        q.memcpy(h_t3 + host_offset, d_t3_stage,
                                 pass_count * sizeof(T3PairingGpu)).wait();
                    host_offset += pass_count;
                    q.memset(d_counter, 0, sizeof(uint64_t)).wait();
                }
            }
            if (d_mi_r_slice)    s_free(stats, d_mi_r_slice);
            if (d_xbits_r_slice) s_free(stats, d_xbits_r_slice);
            if (d_meta_r_slice)  s_free(stats, d_meta_r_slice);
            release_l();
        } else {
            // Non-tiny (minimal): existing per-section loop preserved.
            for (uint32_t section_l = 0; section_l < num_sections; ++section_l) {
                uint32_t const section_r = compute_section_r(section_l);
                uint64_t const section_l_row_start = h_t3_offsets[section_l * num_match_keys];
                uint64_t const section_l_row_end   = h_t3_offsets[(section_l + 1) * num_match_keys];
                uint64_t const section_l_count     = section_l_row_end - section_l_row_start;
                uint64_t const section_r_row_start = h_t3_offsets[section_r * num_match_keys];
                uint64_t const section_r_row_end   = h_t3_offsets[(section_r + 1) * num_match_keys];
                uint64_t const section_r_count     = section_r_row_end - section_r_row_start;

                if (section_l_count == 0) continue;

                uint64_t* d_meta_l_slice = nullptr;
                uint64_t* d_meta_r_slice = nullptr;
                s_malloc(stats, d_meta_l_slice, section_l_count * sizeof(uint64_t), "d_t3_meta_l_slice");
                if (section_r_count > 0) {
                    s_malloc(stats, d_meta_r_slice, section_r_count * sizeof(uint64_t), "d_t3_meta_r_slice");
                }

                q.memcpy(d_meta_l_slice, h_t2_meta + section_l_row_start,
                         section_l_count * sizeof(uint64_t)).wait();
                if (section_r_count > 0) {
                    q.memcpy(d_meta_r_slice, h_t2_meta + section_r_row_start,
                             section_r_count * sizeof(uint64_t)).wait();
                }

                uint32_t const bucket_begin = section_l * num_match_keys;
                uint32_t const bucket_end   = (section_l + 1) * num_match_keys;

                launch_t3_match_section_pair_range(
                    cfg.plot_id.data(), t3p,
                    d_meta_l_slice, section_l_row_start,
                    d_meta_r_slice, section_r_row_start,
                    d_t2_xbits_sorted, d_t2_keys_merged, t2_count,
                    d_t3_stage, d_counter, t3_section_cap,
                    d_t3_match_temp, bucket_begin, bucket_end, q);

                uint64_t pass_count = 0;
                q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
                if (pass_count > t3_section_cap) {
                    throw std::runtime_error(
                        "T3 match (sliced) section_l=" + std::to_string(section_l) +
                        " produced " + std::to_string(pass_count) +
                        " pairs, staging holds " + std::to_string(t3_section_cap) +
                        " (50% over uniform avg). Lower N or widen t3_section_cap safety factor.");
                }
                if (host_offset + pass_count > cap) {
                    throw std::runtime_error("T3 overflow (staged accumulation)");
                }
                if (t3_spill)
                    t3_spill->write_from_device(d_t3_stage, host_offset, pass_count);
                else
                    q.memcpy(h_t3 + host_offset, d_t3_stage,
                             pass_count * sizeof(T3PairingGpu)).wait();
                host_offset += pass_count;
                q.memset(d_counter, 0, sizeof(uint64_t)).wait();

                if (section_r_count > 0) s_free(stats, d_meta_r_slice);
                s_free(stats, d_meta_l_slice);
            }
        }
        end_phase(p_t3);

        t3_count = host_offset;
        if (t3_count > cap) throw std::runtime_error("T3 overflow");

        // d_t2_meta_sorted is null in this path (never allocated) — skip
        // its s_free. Free everything else that was alive across T3 match.
        s_free(stats, d_t3_match_temp);
        if (d_t3_stage_on_host) {
            s_free_host(stats, d_t3_stage, q);
            d_t3_stage = nullptr;
        } else {
            s_free(stats, d_t3_stage);
        }
        // Tiny: d_t2_xbits_sorted and d_t2_keys_merged are null (parked
        // on host pinned). Free the host buffers instead.
        if (scratch.tiny_mode) {
            // Tiny keeps h_t2_xbits pinned (T2-sort partition is USM-host).
            if (h_xbits_owned) s_free_host(stats, h_t2_xbits, q);
            h_t2_xbits = nullptr;
            if (h_keys_owned && !scratch.pool) s_free_host(stats, h_t2_keys_merged, q);
            h_t2_keys_merged = nullptr;
        } else {
            s_free(stats, d_t2_xbits_sorted);
            s_free(stats, d_t2_keys_merged);
        }

        // h_t2_meta was kept alive across T3 match for slicing; free now
        // that all section pairs have been H2D'd.
        if (h_t2_meta_owned) s_free_host(stats, h_t2_meta, q);
        h_t2_meta = nullptr;

        // Re-hydrate full-cap d_t3 on device for T3 sort.
        //
        // Tiny mode: skip the rehydration. h_t3 stays alive across T3
        // sort; the tiled-sort path below H2D's each tile directly from
        // h_t3 into a small device buffer instead of reading from a
        // full-cap d_t3. Saves 2080 MB of device VRAM at k=28 across
        // T3 sort.
        if (!scratch.tiny_mode) {
            s_malloc(stats, d_t3, cap * sizeof(T3PairingGpu), "d_t3");
            if (t3_spill) {
                // Minimal: pull the spilled table back to the device in one
                // double-buffered pass, then release its disk file.
                t3_spill->read_to_device(d_t3, 0, t3_count);
                t3_spill.reset();
            } else {
                q.memcpy(d_t3, h_t3, t3_count * sizeof(T3PairingGpu)).wait();
                if (h_t3_owned && !scratch.pool) s_free_host(stats, h_t3, q);
            }
        }
        // Stash h_t3 ownership for T3 sort cleanup. d_t3 stays nullptr
        // in tiny mode; the T3 sort phase reads h_t3 (or t3_spill) directly.
        scratch_tiny_h_t3        = scratch.tiny_mode ? h_t3 : nullptr;
        scratch_tiny_h_t3_owned  = (scratch.tiny_mode && h_t3_owned && !scratch.pool);
    } else {
        // Compact: N=2 half-cap staging with pinned-host h_t3 accumulator.
        uint64_t const t3_half_cap = (cap + 1) / 2;

        T3PairingGpu* d_t3_stage    = nullptr;
        void*         d_t3_match_temp = nullptr;
        s_malloc(stats, d_t3_stage,      t3_half_cap * sizeof(T3PairingGpu), "d_t3_stage");
        s_malloc(stats, d_t3_match_temp, t3_temp_bytes,                     "d_t3_match_temp");

        // Full-cap pinned host that will hold the concatenated T3 output.
        // Stage 4f: reuse scratch.h_t3 when provided (amortised across
        // batch). T3PairingGpu is just a uint64 proof_fragment, so the
        // scratch buffer is declared as uint64_t* and reinterpret-cast.
        bool const h_t3_owned   = (scratch.h_t3 == nullptr);
        bool const h_t3_spilled = h_t3_owned && scratch.spill.h_t3;
        T3PairingGpu* h_t3 = nullptr;
        if (h_t3_spilled) {
            // Host-RAM disk-offload: redirect the ~2 GiB h_t3 accumulator to
            // a TempFile via the shared SpillEngine (compact tier).
            t3_spill = std::make_unique<SpillBuffer>(
                ensure_spill_engine(), sizeof(T3PairingGpu));
            std::fprintf(stderr,
                "[spill] h_t3 -> disk %s (shared 64 MiB staging, %.2f GiB spilled)\n",
                t3_spill->file.path().c_str(),
                cap * sizeof(T3PairingGpu) / 1073741824.0);
        } else if (h_t3_owned) {
            h_t3 = scratch.pool
                ? scratch.pool->acquire_as<T3PairingGpu>("h_t3", cap, q)
                : s_malloc_host<T3PairingGpu>(stats, cap * sizeof(T3PairingGpu),
                                              "h_t3", q);
        } else {
            h_t3 = reinterpret_cast<T3PairingGpu*>(scratch.h_t3);
        }
        if (!h_t3 && !t3_spill) throw std::runtime_error("sycl::malloc_host(h_t3) failed");

        // Compute bucket + fine-bucket offsets once; both match passes
        // share them. Also zeroes d_counter.
        launch_t3_match_prepare(cfg.plot_id.data(), t3p,
                                d_t2_keys_merged, t2_count,
                                d_counter, d_t3_match_temp, &t3_temp_bytes, q);

        uint32_t const t3_num_buckets =
            (1u << t3p.num_section_bits) * (1u << t3p.num_match_key_bits);
        uint32_t const t3_bucket_mid = t3_num_buckets / 2;

        auto run_t3_pass = [&](uint32_t bucket_begin, uint32_t bucket_end,
                               uint64_t host_offset) -> uint64_t
        {
            launch_t3_match_range(cfg.plot_id.data(), t3p,
                                  d_t2_meta_sorted, d_t2_xbits_sorted,
                                  d_t2_keys_merged, t2_count,
                                  d_t3_stage, d_counter, t3_half_cap,
                                  d_t3_match_temp, bucket_begin, bucket_end, q);
            uint64_t pass_count = 0;
            q.memcpy(&pass_count, d_counter, sizeof(uint64_t)).wait();
            if (pass_count > t3_half_cap) {
                throw std::runtime_error(
                    "T3 match pass overflow: bucket range [" +
                    std::to_string(bucket_begin) + "," + std::to_string(bucket_end) +
                    ") produced " + std::to_string(pass_count) +
                    " pairs, staging holds " + std::to_string(t3_half_cap) +
                    ". Lower N or widen staging.");
            }
            if (host_offset + pass_count > cap) {
                throw std::runtime_error("T3 overflow (staged accumulation)");
            }
            if (t3_spill)
                t3_spill->write_from_device(d_t3_stage, host_offset, pass_count);
            else
                q.memcpy(h_t3 + host_offset, d_t3_stage,
                         pass_count * sizeof(T3PairingGpu)).wait();
            // Reset counter so the next pass writes at stage index 0.
            q.memset(d_counter, 0, sizeof(uint64_t)).wait();
            return pass_count;
        };

        int p_t3 = begin_phase("T3 match + Feistel");
        uint64_t const t3_count1 = run_t3_pass(0,              t3_bucket_mid,   /*host_offset=*/0);
        uint64_t const t3_count2 = run_t3_pass(t3_bucket_mid,  t3_num_buckets,  /*host_offset=*/t3_count1);
        end_phase(p_t3);

        t3_count = t3_count1 + t3_count2;
        if (t3_count > cap) throw std::runtime_error("T3 overflow");

        // Free everything that was alive across T3 match: staging, temp,
        // sorted T2 inputs, keys_merged.
        s_free(stats, d_t3_match_temp);
        s_free(stats, d_t3_stage);
        s_free(stats, d_t2_meta_sorted);
        s_free(stats, d_t2_xbits_sorted);
        s_free(stats, d_t2_keys_merged);

        // Re-hydrate full-cap d_t3 on device for T3 sort.
        s_malloc(stats, d_t3, cap * sizeof(T3PairingGpu), "d_t3");
        if (t3_spill) {
            t3_spill->read_to_device(d_t3, 0, t3_count);
            t3_spill.reset();
        } else {
            q.memcpy(d_t3, h_t3, t3_count * sizeof(T3PairingGpu)).wait();
            if (h_t3_owned) s_free_host(stats, h_t3, q);
        }
    }

    // ---------- Phase T3 sort ----------
    // Compact / plain: full-cap CUB sort_keys with separate keys_in
    // (= d_t3) and keys_out (= d_frags_out) buffers — peaks at
    // 2 × cap × u64 + scratch ≈ 4228 MB at k=28.
    //
    // Minimal: tile the sort in halves with a single cap/2 output
    // buffer, D2H each tile to host pinned, std::inplace_merge on
    // host, then H2D the merged result back into the full-cap
    // d_frags_out the D2H phase below expects. Drops T3 sort peak to
    // ~3152 MB at k=28 (d_t3 2080 + tile output 1040 + sort scratch
    // sized for cap/2 ≈ 32). Adds one cap-sized PCIe round-trip per
    // plot.
    stats.phase = "T3 sort";
    uint64_t* d_frags_in  = reinterpret_cast<uint64_t*>(d_t3);
    uint64_t* d_frags_out = nullptr;

    // Phase 1.5c-b (revived): in Tiny tier the tile-sort + host-merge
    // already lands the final sorted output in a host-pinned buffer
    // (h_frags below). The original code then H2D-re-hydrates a
    // cap-sized d_frags_out on device just so the D2H phase has
    // something to memcpy out of. That re-hydrate sets Tiny's k=26
    // device-peak (528 MB) — by far the largest single contributor.
    //
    // When the caller has supplied pinned_dst with cap-class capacity
    // (batch mode), we can alias h_frags = pinned_dst directly: the
    // tile sort writes its D2H'd output into pinned_dst, std::inplace_merge
    // runs in-place there, and the D2H phase becomes a no-op (the
    // sorted fragments already live in the caller's buffer).
    //
    // For the one-shot path (no pinned_dst) we still alias the temp
    // pinned region by pre-allocating it before the tile sort and
    // letting D2H emit it into the OWNING result vector with a host
    // std::memcpy. Either way: no device-side d_frags_out re-hydrate.
    bool d_frags_out_on_host = false;
    uint64_t* h_frags_owned_oneshot = nullptr;  // freed in D2H after copy

    // Phase 2.5a: t3_sort_full_cap opt-in skips the tile-merge path
    // when caller has confirmed VRAM headroom. Forces the fast path
    // even in minimal mode where t1_match_sliced is true. Tiny mode
    // (input lives on host) can't take this branch — full-cap fast
    // path needs d_t3 on device.
    bool const t3_use_full_cap = !t1_match_sliced ||
        (scratch.t3_sort_full_cap && !scratch.tiny_mode);

    if (t3_use_full_cap) {
        size_t t3_sort_bytes = 0;
        launch_sort_keys_u64(
            nullptr, t3_sort_bytes,
            static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
            cap, 0, 2 * cfg.k, q);

        s_malloc(stats, d_frags_out,    cap * sizeof(uint64_t), "d_frags_out");
        s_malloc(stats, d_sort_scratch, t3_sort_bytes,          "d_sort_scratch(t3)");

        int p_t3_sort = begin_phase("T3 sort");
        launch_sort_keys_u64(
            d_sort_scratch, t3_sort_bytes,
            d_frags_in, d_frags_out,
            t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
        end_phase(p_t3_sort);

        s_free(stats, d_t3);
        s_free(stats, d_sort_scratch);
    } else {
        // Tiled sort + host merge.
        //
        // Minimal: input lives on device (d_t3 full cap). Tile sort
        // reads from d_frags_in (= d_t3) + offset.
        //
        // Tiny: input lives on host (scratch_tiny_h_t3, set by the
        // earlier minimal/tiny T3 match block when tiny_mode is true).
        // d_t3 is nullptr. Each tile sort H2Ds the input slice from
        // host into the same d_frags_out_tile buffer (reused as both
        // sort input and sort output via in-place CUB), removing the
        // ~2080 MB d_t3 pin from the T3 sort phase.
        // Cheap-win bump from N=2 to N=4: shrinks tile_max from cap/2
        // to cap/4 → d_frags_out_tile (+ d_frags_in_tile in tiny mode)
        // each drop by half. T3 sort phase peak drops from 2*(cap/2)
        // + scratch ≈ 537 MB at k=26 to 2*(cap/4) + scratch ≈ 269 MB
        // at k=26. Host-side merge becomes a 3-merge tree for N=4.
        constexpr int kT3SortTiles = 4;
        uint64_t const tile_max = (cap + uint64_t(kT3SortTiles) - 1) / uint64_t(kT3SortTiles);
        uint64_t t3_offsets[kT3SortTiles + 1];
        t3_offsets[0] = 0;
        for (int i = 0; i < kT3SortTiles; ++i) {
            uint64_t const next = std::min(t3_offsets[i] + tile_max, t3_count);
            t3_offsets[i + 1] = next;
        }

        size_t t3_tile_sort_bytes = 0;
        launch_sort_keys_u64(
            nullptr, t3_tile_sort_bytes,
            static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
            tile_max, 0, 2 * cfg.k, q);

        uint64_t* d_frags_out_tile     = nullptr;
        uint64_t* d_frags_in_tile      = nullptr;  // tiny only
        void*     d_sort_scratch_tile  = nullptr;
        s_malloc(stats, d_frags_out_tile,    tile_max * sizeof(uint64_t), "d_frags_out_tile");
        if (scratch.tiny_mode) {
            // Separate input buffer for tiny since launch_sort_keys_u64
            // requires distinct in/out buffers for CUB radix sort.
            s_malloc(stats, d_frags_in_tile, tile_max * sizeof(uint64_t), "d_frags_in_tile");
        }
        s_malloc(stats, d_sort_scratch_tile, t3_tile_sort_bytes,          "d_sort_scratch(t3_tile)");

        // Tiny tier: alias h_frags onto the caller's pinned_dst when
        // available, else pre-allocate the one-shot temp and let D2H
        // free it after the std::memcpy to the OWNING result. Saves
        // the d_frags_out re-hydrate AND avoids double-allocating
        // cap-sized host pinned (one for h_frags + one for h_pinned
        // in D2H).
        uint64_t* h_frags = nullptr;
        // Host-RAM disk-offload: mmap-backed home for the non-tiny h_frags
        // when the budget policy selected it (CPU inplace_merge target, so
        // NOT the device-staging SpillBuffer). Held here so the mapping
        // outlives the merge + final D2H below.
        std::unique_ptr<pos2gpu::TempFile> frags_spill_file;
        if (scratch.tiny_mode && pinned_dst && pinned_capacity >= cap) {
            h_frags = pinned_dst;
            d_frags_out_on_host = true;
        } else if (scratch.tiny_mode) {
            h_frags_owned_oneshot = s_malloc_host<uint64_t>(
                stats, cap * sizeof(uint64_t), "h_frags_oneshot", q);
            h_frags = h_frags_owned_oneshot;
            d_frags_out_on_host = true;
        } else if (scratch.spill.h_frags) {
            // ~2 GiB pageable, file-backed drop-in. q.memcpy D2H into it
            // (line ~4614) and std::inplace_merge both work on a plain
            // host pointer; under pressure the kernel reclaims its pages
            // to THIS temp file instead of holding them pinned.
            frags_spill_file = std::make_unique<pos2gpu::TempFile>();
            h_frags = static_cast<uint64_t*>(
                frags_spill_file->map(cap * sizeof(uint64_t)));
            std::fprintf(stderr,
                "[spill] h_frags -> mmap %s (pageable, %.2f GiB file-backed)\n",
                frags_spill_file->path().c_str(),
                double(cap) * sizeof(uint64_t) / 1073741824.0);
        } else {
            h_frags = s_malloc_host<uint64_t>(
                stats, cap * sizeof(uint64_t), "h_frags", q);
        }

        // Tiny mode: source pointer is the parked host buffer; the
        // staging variant H2Ds before each tile sort. In minimal mode
        // d_frags_in is the device-resident d_t3 and the tile sort
        // reads from it directly.
        uint64_t const* h_t3_src =
            scratch.tiny_mode
                ? reinterpret_cast<uint64_t const*>(scratch_tiny_h_t3)
                : nullptr;

        int p_t3_sort = begin_phase("T3 sort");
        for (int t = 0; t < kT3SortTiles; ++t) {
            uint64_t const tile_off = t3_offsets[t];
            uint64_t const tile_n   = t3_offsets[t + 1] - tile_off;
            if (tile_n == 0) continue;

            uint64_t const* sort_in = nullptr;
            if (scratch.tiny_mode) {
                if (t3_spill)
                    // Tiny: pull this tile of the spilled h_t3 from disk,
                    // double-buffered through the shared staging windows.
                    t3_spill->read_to_device(d_frags_in_tile, tile_off, tile_n);
                else
                    q.memcpy(d_frags_in_tile, h_t3_src + tile_off,
                             tile_n * sizeof(uint64_t)).wait();
                sort_in = d_frags_in_tile;
            } else {
                sort_in = d_frags_in + tile_off;
            }
            launch_sort_keys_u64(
                d_sort_scratch_tile, t3_tile_sort_bytes,
                const_cast<uint64_t*>(sort_in), d_frags_out_tile,
                tile_n, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, q);
            q.memcpy(h_frags + tile_off, d_frags_out_tile,
                     tile_n * sizeof(uint64_t)).wait();
        }
        end_phase(p_t3_sort);

        s_free(stats, d_frags_out_tile);
        if (scratch.tiny_mode) s_free(stats, d_frags_in_tile);
        s_free(stats, d_sort_scratch_tile);
        if (scratch.tiny_mode) {
            // h_t3 was kept alive into T3 sort; free now that all tiles
            // have been sorted + D2H'd.
            if (t3_spill) t3_spill.reset();
            else if (scratch_tiny_h_t3_owned) s_free_host(stats, scratch_tiny_h_t3, q);
            scratch_tiny_h_t3 = nullptr;
        } else {
            s_free(stats, d_t3);
        }

        // Multi-way stable merge of the N=4 sorted runs. Tree shape
        // for N=4: (0+1)→A, (2+3)→B, (A+B)→final. 3 binary
        // std::inplace_merges, depth 2. Stable, matches the original
        // 2-way merge's byte-parity contract for downstream consumers.
        //
        // The two depth-1 merges touch disjoint ranges, so run them
        // concurrently — at k=28 each is a ~1 GB memory-bound merge on
        // the critical path while the GPU sits idle; overlapping them
        // halves the depth-1 wall.
        if constexpr (kT3SortTiles == 4) {
            std::thread merge_ab([&] {
                std::inplace_merge(h_frags + t3_offsets[0],
                                   h_frags + t3_offsets[1],
                                   h_frags + t3_offsets[2]);
            });
            std::inplace_merge(h_frags + t3_offsets[2],
                               h_frags + t3_offsets[3],
                               h_frags + t3_offsets[4]);
            merge_ab.join();
            std::inplace_merge(h_frags + t3_offsets[0],
                               h_frags + t3_offsets[2],
                               h_frags + t3_offsets[4]);
        } else {
            // Generic sequential fallback for other N.
            for (int t = 1; t < kT3SortTiles; ++t) {
                std::inplace_merge(h_frags + t3_offsets[0],
                                   h_frags + t3_offsets[t],
                                   h_frags + t3_offsets[t + 1]);
            }
        }

        if (d_frags_out_on_host) {
            // Tiny: h_frags already holds the sorted output in caller's
            // pinned_dst (or in h_frags_owned_oneshot for the no-pinned
            // path). D2H below detects d_frags_out_on_host and either
            // no-ops or std::memcpys to the OWNING vector — no device
            // re-hydrate, no q.memcpy roundtrip. Saves cap × u64 device
            // bytes (= 528 MB at k=26 / ~2 GB at k=28) on the T3 sort
            // phase peak — the dominant Tiny-tier device floor.
            //
            // Set d_frags_out to h_frags so the unconditional s_free
            // below knows it's a host pointer (the alias case marks the
            // pointer in a side-flag the s_free path checks).
            d_frags_out = h_frags;
        } else {
            // Minimal/Compact/Plain: re-hydrate full-cap d_frags_out for
            // the existing D2H phase.
            s_malloc(stats, d_frags_out, cap * sizeof(uint64_t), "d_frags_out");
            if (t3_count > 0) {
                q.memcpy(d_frags_out, h_frags, t3_count * sizeof(uint64_t)).wait();
            }
            if (frags_spill_file) frags_spill_file.reset();  // munmap + drop temp file
            else s_free_host(stats, h_frags, q);
        }
    }

    // ---------- D2H ----------
    // Two destination modes:
    //   caller-supplied pinned_dst (batch): copy D2H into pinned_dst and
    //     return a BORROWING result (external_fragments_ptr). Consumer
    //     must finish reading pinned_dst before the caller reuses it.
    //   no pinned_dst (one-shot): alloc a temp pinned region sized to
    //     t3_count, D2H, copy to an OWNING vector, free the temp.
    stats.phase = "D2H";
    GpuPipelineResult result;
    result.t1_count = t1_count;
    result.t2_count = t2_count;
    result.t3_count = t3_count;

    int p_d2h = begin_phase("D2H copy T3 fragments (pinned)");
    if (t3_count > 0) {
        if (d_frags_out_on_host) {
            // Tiny: sorted output already lives in pinned_dst (or in
            // h_frags_owned_oneshot). No device-to-host transfer needed.
            if (pinned_dst) {
                // Result aliases pinned_dst — caller owns.
                if (pinned_capacity < t3_count) {
                    throw std::runtime_error(
                        "run_gpu_pipeline_streaming: pinned_capacity " +
                        std::to_string(pinned_capacity) +
                        " < t3_count " + std::to_string(t3_count));
                }
                result.external_fragments_ptr   = pinned_dst;
                result.external_fragments_count = t3_count;
            } else {
                // One-shot path — copy from our temp pinned to OWNING
                // vector using a host std::memcpy (q.memcpy would
                // serialize through the SYCL queue, which an earlier
                // attempt this session showed adds 1-2s wall at k=26).
                result.t3_fragments_storage.resize(t3_count);
                std::memcpy(result.t3_fragments_storage.data(),
                            h_frags_owned_oneshot,
                            sizeof(uint64_t) * t3_count);
            }
        } else if (pinned_dst) {
            if (pinned_capacity < t3_count) {
                throw std::runtime_error(
                    "run_gpu_pipeline_streaming: pinned_capacity " +
                    std::to_string(pinned_capacity) +
                    " < t3_count " + std::to_string(t3_count));
            }
            q.memcpy(pinned_dst, d_frags_out, sizeof(uint64_t) * t3_count);
            q.wait();
            result.external_fragments_ptr   = pinned_dst;
            result.external_fragments_count = t3_count;
        } else {
            uint64_t* h_pinned = s_malloc_host<uint64_t>(
                stats, sizeof(uint64_t) * t3_count, "h_pinned(d2h)", q);
            q.memcpy(h_pinned, d_frags_out, sizeof(uint64_t) * t3_count);
            q.wait();
            result.t3_fragments_storage.resize(t3_count);
            std::memcpy(result.t3_fragments_storage.data(), h_pinned,
                        sizeof(uint64_t) * t3_count);
            s_free_host(stats, h_pinned, q);
        }
    }
    end_phase(p_d2h);

    if (d_frags_out_on_host) {
        // Free the one-shot pinned (alias-to-pinned_dst path leaves the
        // buffer for the caller). d_frags_out is a host pointer in this
        // branch — must NOT route through s_free's device tracking.
        if (h_frags_owned_oneshot) s_free_host(stats, h_frags_owned_oneshot, q);
        d_frags_out = nullptr;
    } else {
        s_free(stats, d_frags_out);
    }
    s_free(stats, d_counter);

    if (stats.verbose) {
        std::fprintf(stderr,
            "[streaming] k=%d strength=%d  peak device VRAM = %.2f MB\n",
            cfg.k, cfg.strength, stats.peak / 1048576.0);
    }
    report_phases();
    return result;
}

} // namespace (anon — streaming impl)

uint64_t* streaming_alloc_pinned_uint64(size_t count)
{
    // Throws rather than returning null: the callers treat null as "allocation
    // failed, fall back", but a host that is out of RAM has nothing to fall
    // back TO — every lower tier wants more host memory, not less. Failing
    // here with the reason beats failing later without one.
    size_t const bytes = count * sizeof(uint64_t);
    host_pinned_reserve_check(bytes, "streaming pinned u64");
    // See s_malloc_host_raw for the redzone rationale. These are the D2H
    // drain slots and the caller-provided scratch tables — the buffers
    // device kernels write into through a cursor, so the ones most worth
    // guarding.
    void* p = sycl::malloc_host(bytes + 2 * pos2gpu::host_guard_pad(),
                                sycl_backend::queue());
    if (!p) return nullptr;
    return static_cast<uint64_t*>(
        pos2gpu::host_guard_arm(p, bytes, "streaming pinned u64"));
}

uint32_t* streaming_alloc_pinned_uint32(size_t count)
{
    size_t const bytes = count * sizeof(uint32_t);
    host_pinned_reserve_check(bytes, "streaming pinned u32");
    void* p = sycl::malloc_host(bytes + 2 * pos2gpu::host_guard_pad(),
                                sycl_backend::queue());
    if (!p) return nullptr;  // nullptr on failure
    return static_cast<uint32_t*>(
        pos2gpu::host_guard_arm(p, bytes, "streaming pinned u32"));
}

uint64_t twophase_bytes_held()
{
    return sycl_backend::twophase_bytes_held(sycl_backend::queue());
}

void streaming_free_pinned_uint32(uint32_t* ptr)
{
    if (ptr) sycl::free(pos2gpu::host_guard_disarm(ptr, "free pinned u32"),
                        sycl_backend::queue());
}

void streaming_free_pinned_uint64(uint64_t* ptr)
{
    if (ptr) sycl::free(pos2gpu::host_guard_disarm(ptr, "free pinned u64"),
                        sycl_backend::queue());
}

void streaming_host_guard_check(char const* where)
{
    pos2gpu::host_guard_check(where);
}

void bind_current_device(int device_id)
{
    sycl_backend::set_current_device_id(device_id);
}

int gpu_device_count()
{
    try {
        return sycl_backend::get_gpu_device_count();
    } catch (...) {
        return 0;
    }
}

} // namespace pos2gpu
