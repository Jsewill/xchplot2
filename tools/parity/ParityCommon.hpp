// ParityCommon.hpp — shared harness helpers for the parity tests.
//
// Keeps the PRNG seed shape, mismatch-reporting format, and the CUDA
// error-check macro consistent across every `*_parity` / `*_bench`
// binary in this directory. The audit that motivated this header
// found ~170 lines of verbatim copy-paste across 7-9 files (same
// derive_plot_id, same Stats/compare shape, same CHECK macro).
//
// Plain-header (inline) so .cu and .cpp TUs can both include it
// without changing the existing CMake layout. No library target
// needed.

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// CUDA error-check macro. Only meaningful inside a .cu TU (where
// cuda_runtime.h is in scope). Guarded behind __CUDACC__ so the
// header can still be included from plain .cpp parity tests for
// derive_plot_id / Stats / compare without pulling in CUDA.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define PARITY_CHECK(call) do {                                              \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(err));           \
        std::exit(2);                                                        \
    }                                                                        \
} while (0)
#endif

namespace pos2gpu::parity {

// Deterministic mixing from a 32-bit seed to a 32-byte plot_id. Not
// cryptographic — just spreads bits so parity tests for distinct seeds
// exercise non-trivially different plot_ids. Golden-ratio + splitmix-
// style step.
inline std::array<uint8_t, 32> derive_plot_id(uint32_t seed)
{
    std::array<uint8_t, 32> id{};
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ uint64_t(seed) * 0x100000001B3ULL;
    for (std::size_t i = 0; i < id.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        id[i] = static_cast<uint8_t>(s >> 56);
    }
    return id;
}

// Mismatch counter with pretty-print of the first 5 errors per
// (seed, label). Keeps test output useful when a regression lands:
// you see which labelled comparison first diverges and at what
// index, without a multi-thousand-line fault log.
struct Stats {
    uint64_t total      = 0;
    uint64_t mismatches = 0;
    bool ok() const { return mismatches == 0; }
};

// Cmp is any `bool(uint64_t i)` — returns true when host index i
// agrees between CPU reference and GPU result.
template <typename Cmp>
Stats compare(uint64_t n, Cmp const& cmp, char const* label, uint32_t seed)
{
    Stats s;
    s.total = n;
    for (uint64_t i = 0; i < n; ++i) {
        if (!cmp(i)) {
            if (s.mismatches < 5) {
                std::printf("  [seed=%u %s] MISMATCH at i=%llu\n",
                            seed, label,
                            static_cast<unsigned long long>(i));
            }
            ++s.mismatches;
        }
    }
    return s;
}

} // namespace pos2gpu::parity
