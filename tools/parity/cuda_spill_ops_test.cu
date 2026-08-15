// cuda_spill_ops_test — the spill engine driven over REAL CUDA memory.
//
// spill_engine_test already hammers the ticket protocol with a std::memcpy
// backend and no device. That proves the protocol; it cannot prove the CUDA
// backing, which is the half a port actually gets wrong: a copy whose direction
// is inferred correctly on the write path and wrongly on the read path, staging
// that is pageable rather than pinned, or a copy that has not landed when
// copy_blocking returns.
//
// So this runs the real SpillEngine + SpillBuffer with CudaSpillHostOps
// against real device allocations, and checks the only thing that ultimately
// matters: what comes back off the disk is byte-for-byte what went down to it.
//
// Deliberately independent of the plotting pipeline. It needs a GPU but not a
// plot, so it is unaffected by the k=28 streaming non-determinism that
// currently blocks byte-parity on this branch — which is exactly why it can be
// landed and trusted before that is fixed.

#include "host/CudaSpillHostOps.hpp"
#include "host/SpillEngine.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, char const* what)
{
    std::printf("%-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

#define CUDA_OK(expr)                                                          \
    do {                                                                       \
        cudaError_t const e_ = (expr);                                         \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__,      \
                         #expr, cudaGetErrorString(e_));                       \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

// A pattern that is sensitive to BOTH a wrong offset and a wrong length: every
// entry encodes its own index, so a shifted or truncated round trip shows up as
// a specific mismatching index rather than as zeros.
std::vector<std::uint64_t> pattern(std::uint64_t n, std::uint64_t salt)
{
    std::vector<std::uint64_t> v(n);
    for (std::uint64_t i = 0; i < n; ++i)
        v[i] = (i * 0x9E3779B97F4A7C15ull) ^ (salt * 0xD1B54A32D192ED03ull) ^ (i + 1);
    return v;
}

}  // namespace

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device available - skipping\n");
        return 0;   // not a failure: this test is device-gated by design
    }

    using pos2gpu::CudaSpillHostOps;
    using pos2gpu::SpillBuffer;
    using pos2gpu::SpillEngine;

    // Deliberately spans several staging windows (32 MiB / 8 B = 4M entries),
    // and is NOT a whole multiple of one, so the tail chunk is exercised too.
    constexpr std::uint64_t kWinEntries = SpillEngine::kStageBytes / sizeof(std::uint64_t);
    std::uint64_t const     n           = kWinEntries * 2 + 12345;

    CudaSpillHostOps ops{/*stream=*/nullptr};
    SpillEngine      eng(ops, /*quiet=*/true);

    std::uint64_t* d_src = nullptr;
    std::uint64_t* d_dst = nullptr;
    CUDA_OK(cudaMalloc(&d_src, n * sizeof(std::uint64_t)));
    CUDA_OK(cudaMalloc(&d_dst, n * sizeof(std::uint64_t)));

    auto const host_src = pattern(n, 7);
    CUDA_OK(cudaMemcpy(d_src, host_src.data(), n * sizeof(std::uint64_t),
                       cudaMemcpyHostToDevice));
    // Poison the destination so a read that silently does nothing cannot pass.
    CUDA_OK(cudaMemset(d_dst, 0xA5, n * sizeof(std::uint64_t)));

    // ---- whole-buffer round trip, device -> disk -> device ----
    {
        SpillBuffer buf(eng, sizeof(std::uint64_t), n);
        buf.write_from_device(d_src, 0, n);
        buf.drain();                      // land every deferred pwrite
        buf.read_to_device(d_dst, 0, n);

        std::vector<std::uint64_t> back(n, 0);
        CUDA_OK(cudaMemcpy(back.data(), d_dst, n * sizeof(std::uint64_t),
                           cudaMemcpyDeviceToHost));
        check(back == host_src, "round trip: device -> disk -> device is exact");
    }

    // ---- partial range, at a non-window-aligned offset ----
    //
    // The offset arithmetic is per-entry on the caller's side and per-byte
    // inside the engine, and a spill that is correct only at offset 0 is a
    // spill that corrupts every table after the first pass.
    {
        std::uint64_t const off = kWinEntries + 777;
        std::uint64_t const len = kWinEntries / 3;

        SpillBuffer buf(eng, sizeof(std::uint64_t), n);
        buf.write_from_device(d_src + off, off, len);
        buf.drain();

        CUDA_OK(cudaMemset(d_dst, 0x5A, n * sizeof(std::uint64_t)));
        buf.read_to_device(d_dst + off, off, len);

        std::vector<std::uint64_t> back(len, 0);
        CUDA_OK(cudaMemcpy(back.data(), d_dst + off, len * sizeof(std::uint64_t),
                           cudaMemcpyDeviceToHost));
        bool ok = true;
        for (std::uint64_t i = 0; i < len && ok; ++i)
            ok = (back[i] == host_src[off + i]);
        check(ok, "partial range at unaligned offset round trips exactly");
    }

    // ---- two tables share the one engine's windows ----
    //
    // The whole point of the SpillEngine/SpillBuffer split is that staging
    // stays at 64 MiB no matter how many tables spill. If the shared ping-pong
    // cursor lets one table's deferred write land in the other's window, this
    // is where it shows.
    {
        std::uint64_t const m = kWinEntries + 999;
        auto const a_src = pattern(m, 11);
        auto const b_src = pattern(m, 29);

        std::uint64_t* d_a = nullptr;
        std::uint64_t* d_b = nullptr;
        CUDA_OK(cudaMalloc(&d_a, m * sizeof(std::uint64_t)));
        CUDA_OK(cudaMalloc(&d_b, m * sizeof(std::uint64_t)));
        CUDA_OK(cudaMemcpy(d_a, a_src.data(), m * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_b, b_src.data(), m * sizeof(std::uint64_t), cudaMemcpyHostToDevice));

        SpillBuffer a(eng, sizeof(std::uint64_t), m);
        SpillBuffer b(eng, sizeof(std::uint64_t), m);

        // Interleaved on purpose — sequential writes would not exercise the
        // shared cursor.
        a.write_from_device(d_a, 0, m);
        b.write_from_device(d_b, 0, m);
        a.drain();
        b.drain();

        CUDA_OK(cudaMemset(d_a, 0, m * sizeof(std::uint64_t)));
        CUDA_OK(cudaMemset(d_b, 0, m * sizeof(std::uint64_t)));
        a.read_to_device(d_a, 0, m);
        b.read_to_device(d_b, 0, m);

        std::vector<std::uint64_t> a_back(m, 0), b_back(m, 0);
        CUDA_OK(cudaMemcpy(a_back.data(), d_a, m * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(b_back.data(), d_b, m * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        check(a_back == a_src && b_back == b_src,
              "two tables sharing one engine do not cross-contaminate");

        CUDA_OK(cudaFree(d_a));
        CUDA_OK(cudaFree(d_b));
    }

    // ---- the read guard still fires over the CUDA backend ----
    //
    // Spill temp files are SPARSE: a pread of a never-written range succeeds
    // and returns ZEROS, which is indistinguishable from real data downstream.
    // SpillCoverage is what turns that into a hard error, and a guard that has
    // quietly stopped firing looks exactly like a healthy run.
    {
        SpillBuffer buf(eng, sizeof(std::uint64_t), n);
        buf.write_from_device(d_src, 0, kWinEntries);   // only the first window
        buf.drain();
        bool threw = false;
        try {
            buf.read_to_device(d_dst, kWinEntries, kWinEntries);  // never written
        } catch (std::exception const&) {
            threw = true;
        }
        check(threw, "reading a never-written range throws, not returns zeros");
    }

    // Staging is the engine's two windows and nothing more, however many
    // SpillBuffers were created above.
    check(ops.staging_live == SpillEngine::kNumWindows * SpillEngine::kStageBytes,
          "staging stayed at kNumWindows * kStageBytes across 4 tables");

    CUDA_OK(cudaFree(d_src));
    CUDA_OK(cudaFree(d_dst));

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall good\n", failures);
    return failures ? 1 : 0;
}
