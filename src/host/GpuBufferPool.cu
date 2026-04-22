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

    // d_pair_*: worst case across T1 (12 B), T2 (16 B), T3 (8 B), uint64
    // frags (8 B), AND the aliased Xs scratch. Xs wants ~4.34 GB at k=28 —
    // d_pair_b is reused for it, so the buffer must be sized to fit either
    // the largest pairing struct OR the Xs construction scratch (4 *
    // total_xs uint32s plus the radix-sort temp).
    uint8_t dummy_plot_id[32] = {};
    POOL_CHECK(launch_construct_xs(dummy_plot_id, k, testnet,
                                   nullptr, nullptr, &xs_temp_bytes));
    pair_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(T1PairingGpu),
        static_cast<size_t>(cap) * sizeof(T2PairingGpu),
        static_cast<size_t>(cap) * sizeof(T3PairingGpu),
        static_cast<size_t>(cap) * sizeof(uint64_t),
        xs_temp_bytes,
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
            storage_bytes + 2 * pair_bytes + sort_scratch_bytes + sizeof(uint64_t);
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
            "[pool] sizes: storage=%.2fGB pair=%.2fGB xs_temp(alias)=%.2fGB "
            "sort_scratch=%.2fGB pinned=%.2fGB\n",
            storage_bytes/1e9, pair_bytes/1e9, xs_temp_bytes/1e9,
            sort_scratch_bytes/1e9, pinned_bytes/1e9);
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
        POOL_CHECK(cudaMalloc(&d_storage,      storage_bytes));
        POOL_CHECK(cudaMalloc(&d_pair_a,       pair_bytes));
        POOL_CHECK(cudaMalloc(&d_pair_b,       pair_bytes));
        POOL_CHECK(cudaMalloc(&d_sort_scratch, sort_scratch_bytes));
        POOL_CHECK(cudaMalloc(&d_counter,      sizeof(uint64_t)));
        for (int i = 0; i < kNumPinnedBuffers; ++i) {
            POOL_CHECK(cudaMallocHost(&h_pinned_t3[i], pinned_bytes));
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
