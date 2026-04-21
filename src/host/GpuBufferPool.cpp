// GpuBufferPool.cu — queries per-phase scratch sizes once and allocates
// worst-case-sized persistent buffers. Slice 13 migrated the device and
// pinned-host allocations from the cudaMalloc / cudaMallocHost family to
// sycl::malloc_device / sycl::malloc_host on the shared SYCL queue;
// cudaMemGetInfo is left as-is because it's a context-level query that
// works regardless of which runtime is doing the allocations (SYCL +
// CUDA host code share the same primary CUDA context).

#include "host/GpuBufferPool.hpp"
#include "gpu/Sort.cuh"
#include "gpu/SyclBackend.hpp"
#include "host/PoolSizing.hpp"

#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pos2gpu {

namespace {


// Allocate `bytes` of device memory on `q` and check for null. The cap-and-
// throw helpers in GpuPipeline.cu are streaming-pipeline specific; the pool
// just allocates worst-case sizes once at construction so a one-line wrap
// suffices.
inline void* sycl_alloc_device_or_throw(size_t bytes, sycl::queue& q,
                                        char const* what)
{
    void* p = sycl::malloc_device(bytes, q);
    if (!p) {
        throw std::runtime_error(std::string("sycl::malloc_device(") + what + ") failed");
    }
    return p;
}

inline void* sycl_alloc_host_or_throw(size_t bytes, sycl::queue& q,
                                      char const* what)
{
    void* p = sycl::malloc_host(bytes, q);
    if (!p) {
        throw std::runtime_error(std::string("sycl::malloc_host(") + what + ") failed");
    }
    return p;
}

} // namespace

GpuBufferPool::GpuBufferPool(int k_, int strength_, bool testnet_)
    : k(k_), strength(strength_), testnet(testnet_)
{
    sycl::queue& q = sycl_backend::queue();

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
    // we alias d_pair_b for that, so the buffer must be sized to fit either
    // the largest pairing struct OR the Xs construction scratch (which is
    // 4 × total_xs uint32s plus the radix-sort temp). The CUB sort scratch
    // alone is ~8 × total_xs, which often exceeds the pairing-only budget.
    uint8_t dummy_plot_id[32] = {};
    launch_construct_xs(dummy_plot_id, k, testnet,
                                   nullptr, nullptr, &xs_temp_bytes, q);
    pair_bytes = std::max({
        static_cast<size_t>(cap) * sizeof(T1PairingGpu),
        static_cast<size_t>(cap) * sizeof(T2PairingGpu),
        static_cast<size_t>(cap) * sizeof(T3PairingGpu),
        static_cast<size_t>(cap) * sizeof(uint64_t),
        xs_temp_bytes,
    });

    // Query CUB sort scratch sizes (largest across T1/T2/T3 sorts).
    size_t s_pairs = 0;
    launch_sort_pairs_u32_u32(
        nullptr, s_pairs,
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        cap, 0, k, q);
    size_t s_keys = 0;
    launch_sort_keys_u64(
        nullptr, s_keys,
        static_cast<uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        cap, 0, 2 * k, q);
    sort_scratch_bytes = std::max(s_pairs, s_keys);

    pinned_bytes = cap * sizeof(uint64_t);

    // Check VRAM before attempting allocation so we can give a useful
    // diagnostic instead of a generic allocation failure. The margin covers
    // GPU driver/context state, sort scratch, AES T-tables, and other small
    // runtime allocations.
    //
    // SYCL has no portable free-memory query, so slice 17c approximates
    // free_b == total_b. The actual sycl::malloc_device call will throw if
    // VRAM is exhausted; the diagnostic message is just less precise about
    // how much of the total is already consumed by other processes.
    {
        size_t const required_device =
            storage_bytes + 2 * pair_bytes + sort_scratch_bytes + sizeof(uint64_t);
        size_t const margin = 512ULL * 1024 * 1024; // 512 MB
        size_t const total_b =
            q.get_device().get_info<sycl::info::device::global_mem_size>();
        size_t const free_b = total_b;  // approximation — see comment above
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
        size_t const total_b =
            q.get_device().get_info<sycl::info::device::global_mem_size>();
        std::fprintf(stderr,
            "[pool] k=%d strength=%d cap=%llu total_xs=%llu "
            "total=%.2fGB (free unavailable in SYCL build)\n",
            k, strength, (unsigned long long)cap, (unsigned long long)total_xs,
            total_b/1e9);
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
        if (d_storage)       { sycl::free(d_storage,      q); d_storage      = nullptr; }
        if (d_pair_a)        { sycl::free(d_pair_a,       q); d_pair_a       = nullptr; }
        if (d_pair_b)        { sycl::free(d_pair_b,       q); d_pair_b       = nullptr; }
        if (d_sort_scratch)  { sycl::free(d_sort_scratch, q); d_sort_scratch = nullptr; }
        if (d_counter)       { sycl::free(d_counter,      q); d_counter      = nullptr; }
        for (int i = 0; i < kNumPinnedBuffers; ++i) {
            if (h_pinned_t3[i]) { sycl::free(h_pinned_t3[i], q); h_pinned_t3[i] = nullptr; }
        }
    };
    try {
        d_storage      = sycl_alloc_device_or_throw(storage_bytes,      q, "d_storage");
        d_pair_a       = sycl_alloc_device_or_throw(pair_bytes,         q, "d_pair_a");
        d_pair_b       = sycl_alloc_device_or_throw(pair_bytes,         q, "d_pair_b");
        d_sort_scratch = sycl_alloc_device_or_throw(sort_scratch_bytes, q, "d_sort_scratch");
        d_counter      = static_cast<uint64_t*>(
            sycl_alloc_device_or_throw(sizeof(uint64_t),                q, "d_counter"));
        for (int i = 0; i < kNumPinnedBuffers; ++i) {
            h_pinned_t3[i] = static_cast<uint64_t*>(
                sycl_alloc_host_or_throw(pinned_bytes,                  q, "h_pinned_t3"));
        }
    } catch (...) {
        cleanup_partial();
        throw;
    }
}

GpuBufferPool::~GpuBufferPool()
{
    sycl::queue& q = sycl_backend::queue();
    if (d_storage)       sycl::free(d_storage,      q);
    if (d_pair_a)        sycl::free(d_pair_a,       q);
    if (d_pair_b)        sycl::free(d_pair_b,       q);
    if (d_sort_scratch)  sycl::free(d_sort_scratch, q);
    if (d_counter)       sycl::free(d_counter,      q);
    for (int i = 0; i < kNumPinnedBuffers; ++i) {
        if (h_pinned_t3[i]) sycl::free(h_pinned_t3[i], q);
    }
}

} // namespace pos2gpu
