// GpuBufferPool.cu — queries per-phase scratch sizes once and allocates
// worst-case-sized persistent buffers.

#include "host/GpuBufferPool.hpp"
#include "host/PoolSizing.hpp"

#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pos2gpu {

namespace {

// Variadic so the preprocessor doesn't choke on template-argument commas
// in e.g. cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(...).
#define POOL_CHECK(...) do {                                             \
    cudaError_t err = (__VA_ARGS__);                                     \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(std::string("GpuBufferPool CUDA: ") +   \
                                 cudaGetErrorString(err));               \
    }                                                                    \
} while (0)

// Format a byte count as "<N> bytes (<N.NN> MB)" for diagnostics. The
// raw byte count surfaces sub-MiB requests that would otherwise round
// to "0 MB"; the MB form keeps human readability for the > 1 MiB case.
inline std::string fmt_alloc_bytes(size_t bytes)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%zu bytes (%.2f MB)",
                  bytes, double(bytes) / (1024.0 * 1024.0));
    return std::string(buf);
}

// cudaMalloc wrapper that includes the requested byte count + the
// underlying CUDA error name/string in the diagnostic. The pool's
// preflight cudaMemGetInfo check fails most attempts cleanly with
// the "needs ~X.X GiB, only Y.Y GiB free" InsufficientVramError
// shape, but a later cudaMalloc can race with another GPU consumer
// (compositor spike, transient driver activity) and surface CUDA:2
// (cudaErrorMemoryAllocation) — without this wrapper the caller saw
// only "GpuBufferPool CUDA: out of memory" with no clue which alloc
// or how big. Mirrors main's d7d2748 sycl_alloc_device_or_throw fix.
template <typename T>
inline void pool_alloc(T*& out, size_t bytes, char const* what)
{
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&out), bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMalloc(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") failed: " +
            cudaGetErrorName(err) + " (" + cudaGetErrorString(err) +
            "). Likely transient OOM — check `nvidia-smi` for other "
            "GPU consumers, or set POS2GPU_MAX_VRAM_MB lower if VRAM "
            "is shared with display/compositor.");
    }
}

inline void pool_alloc_host(void** out, size_t bytes, char const* what)
{
    cudaError_t err = cudaMallocHost(out, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMallocHost(") + what + ", " +
            fmt_alloc_bytes(bytes) + ") failed: " +
            cudaGetErrorName(err) + " (" + cudaGetErrorString(err) +
            "). Pinned-host alloc failed — system RAM exhausted, or "
            "the pinned-memory cgroup/ulimit is below the requested "
            "size.");
    }
}

} // namespace

GpuBufferPool::GpuBufferPool(int k_, int strength_, bool testnet_)
    : k(k_), strength(strength_), testnet(testnet_)
{
    int const num_section_bits = (k < 28) ? 2 : (k - 26);
    total_xs = 1ULL << k;
    cap      = max_pairs_per_section(k, num_section_bits) * (1ULL << num_section_bits);

    // d_storage must hold EITHER total_xs XsCandidateGpu (8 B each) OR four
    // cap-sized uint32 key/val arrays during sort. Cast everything to size_t
    // so std::max's template deduction finds one common type.
    storage_bytes = std::max(
        static_cast<size_t>(total_xs) * sizeof(XsCandidateGpu),
        static_cast<size_t>(cap) * 4 * sizeof(uint32_t));

    // d_pair_a holds the *match output* of the current phase: T1 SoA
    // (meta·8 B + mi·4 B = 12 B), T2 SoA (meta·8 B + mi·4 B + xbits·4 B =
    // 16 B), then T3 (T3PairingGpu, 8 B). Worst case is T2 at 16 B/entry.
    // It does NOT alias the Xs construction scratch — that's d_pair_b.
    pair_a_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(T1PairingGpu),
        static_cast<size_t>(cap) * sizeof(T2PairingGpu),
        static_cast<size_t>(cap) * sizeof(T3PairingGpu),
        static_cast<size_t>(cap) * sizeof(uint64_t),
    });

    // d_pair_b holds the *sort output* of the current phase (sorted T1
    // meta, sorted T2 meta+xbits, T3 frags) AND the Xs construction
    // scratch (~4.4 GB at k=28: 4 × total_xs uint32s + radix temp). Sized
    // to the max of those — at k=28 the Xs scratch dominates by ~3 GB
    // over the largest sorted output (cap·12 B for T2's meta+xbits).
    uint8_t dummy_plot_id[32] = {};
    POOL_CHECK(launch_construct_xs(dummy_plot_id, k, testnet,
                                   nullptr, nullptr, &xs_temp_bytes));
    pair_b_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(uint64_t),                          // sorted T1 meta
        static_cast<size_t>(cap) * (sizeof(uint64_t) + sizeof(uint32_t)),     // sorted T2 meta+xbits
        static_cast<size_t>(cap) * sizeof(uint64_t),                          // T3 frags out
        xs_temp_bytes,                                                        // Xs aliased scratch
    });

    // Query CUB sort scratch sizes (largest across T1/T2/T3 sorts).
    size_t s_pairs = 0;
    POOL_CHECK(cub::DeviceRadixSort::SortPairs<uint32_t, uint32_t>(
        nullptr, s_pairs,
        static_cast<uint32_t const*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t const*>(nullptr), static_cast<uint32_t*>(nullptr),
        cap, 0, k, nullptr));
    size_t s_keys = 0;
    POOL_CHECK(cub::DeviceRadixSort::SortKeys<uint64_t>(
        nullptr, s_keys,
        static_cast<uint64_t const*>(nullptr), static_cast<uint64_t*>(nullptr),
        cap, 0, 2 * k, nullptr));
    sort_scratch_bytes = std::max(s_pairs, s_keys);

    pinned_bytes = cap * sizeof(uint64_t);

    // Check free VRAM before attempting allocation so we can give a useful
    // diagnostic instead of a generic cudaErrorMemoryAllocation. The margin
    // covers CUDA driver/context state, CUB internal scratch, AES T-tables,
    // and other small runtime allocations.
    {
        size_t const required_device =
            storage_bytes + pair_a_bytes + pair_b_bytes + sort_scratch_bytes + sizeof(uint64_t);
        size_t const margin = 512ULL * 1024 * 1024; // 512 MB
        size_t free_b = 0, total_b = 0;
        POOL_CHECK(cudaMemGetInfo(&free_b, &total_b));
        if (free_b < required_device + margin) {
            auto to_gib = [](size_t b) { return b / double(1ULL << 30); };
            InsufficientVramError e(
                "GpuBufferPool: insufficient device VRAM for k=" +
                std::to_string(k) + " strength=" + std::to_string(strength) +
                "; need ~" + std::to_string(to_gib(required_device + margin)).substr(0, 5) +
                " GiB (pool " + std::to_string(to_gib(required_device)).substr(0, 5) +
                " GiB + ~0.5 GiB runtime), only " +
                std::to_string(to_gib(free_b)).substr(0, 5) +
                " GiB free of " + std::to_string(to_gib(total_b)).substr(0, 5) +
                " GiB total. Use a smaller k or a GPU with more VRAM.");
            e.required_bytes = required_device + margin;
            e.free_bytes     = free_b;
            e.total_bytes    = total_b;
            throw e;
        }
    }

    if (getenv("POS2GPU_POOL_DEBUG")) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::fprintf(stderr,
            "[pool] k=%d strength=%d cap=%llu total_xs=%llu "
            "free=%.2fGB total=%.2fGB\n",
            k, strength, (unsigned long long)cap, (unsigned long long)total_xs,
            free_b/1e9, total_b/1e9);
        std::fprintf(stderr,
            "[pool] sizes: storage=%.2fGB pair_a=%.2fGB pair_b=%.2fGB "
            "xs_temp(alias→pair_b)=%.2fGB sort_scratch=%.2fGB pinned=%.2fGB\n",
            storage_bytes/1e9, pair_a_bytes/1e9, pair_b_bytes/1e9,
            xs_temp_bytes/1e9, sort_scratch_bytes/1e9, pinned_bytes/1e9);
    }

    // Wrap allocations so a mid-sequence failure (e.g. d_pair_b OOM after
    // d_storage + d_pair_a have already succeeded) frees the pre-allocated
    // buffers instead of leaking ~10 GB of device VRAM and ~7 GB of host
    // pinned memory per failed pool ctor across a batch retry loop.
    auto cleanup_partial = [&]{
        if (d_storage)       { cudaFree(d_storage);      d_storage      = nullptr; }
        if (d_pair_a)        { cudaFree(d_pair_a);       d_pair_a       = nullptr; }
        if (d_pair_b)        { cudaFree(d_pair_b);       d_pair_b       = nullptr; }
        if (d_sort_scratch)  { cudaFree(d_sort_scratch); d_sort_scratch = nullptr; }
        if (d_counter)       { cudaFree(d_counter);      d_counter      = nullptr; }
        for (int i = 0; i < kNumPinnedBuffers; ++i) {
            if (h_pinned_t3[i]) { cudaFreeHost(h_pinned_t3[i]); h_pinned_t3[i] = nullptr; }
        }
    };
    try {
        pool_alloc(d_storage,      storage_bytes,      "d_storage");
        pool_alloc(d_pair_a,       pair_a_bytes,       "d_pair_a");
        pool_alloc(d_pair_b,       pair_b_bytes,       "d_pair_b");
        pool_alloc(d_sort_scratch, sort_scratch_bytes, "d_sort_scratch");
        pool_alloc(d_counter,      sizeof(uint64_t),   "d_counter");
        for (int i = 0; i < kNumPinnedBuffers; ++i) {
            pool_alloc_host(reinterpret_cast<void**>(&h_pinned_t3[i]),
                            pinned_bytes,
                            ("h_pinned_t3[" + std::to_string(i) + "]").c_str());
        }
    } catch (...) {
        cleanup_partial();
        throw;
    }
}

GpuBufferPool::~GpuBufferPool()
{
    if (d_storage)       cudaFree(d_storage);
    if (d_pair_a)        cudaFree(d_pair_a);
    if (d_pair_b)        cudaFree(d_pair_b);
    if (d_sort_scratch)  cudaFree(d_sort_scratch);
    if (d_counter)       cudaFree(d_counter);
    for (int i = 0; i < kNumPinnedBuffers; ++i) {
        if (h_pinned_t3[i]) cudaFreeHost(h_pinned_t3[i]);
    }
}

} // namespace pos2gpu
