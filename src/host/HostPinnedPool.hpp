// HostPinnedPool.hpp — name-keyed cache of host-pinned buffers.
//
// Motivation: the streaming pipeline does ~52 sycl::malloc_host calls per
// plot. In a batch (or pipeline-plot) workload most of those buffers have
// the same shape across plots, so reallocating each time burns driver
// time and pinned memory pages. HostPinnedPool keeps the buffers alive
// across plots via named slots — same name returns the same pointer (or
// a grown version of it) on subsequent plots.
//
// Lifetime: pool owns every buffer it allocates and frees them in its
// destructor. A `release_all()` call only marks slots reusable for a
// future plot; the physical memory stays pinned. This trades a small
// steady-state pinned-RAM overhead for zero per-plot driver cost on the
// covered allocations.
//
// Concurrency: the pool itself is *not* thread-safe. The streaming
// pipeline is single-threaded per device, and the pipeline-parallel
// orchestrator's stage 1 and stage 2 threads each get their own pool.
//
// Sizing policy: sizes are memorized per name; on a request larger than
// the cached size, the existing buffer is freed and a fresh one is
// allocated. Sizes never shrink — once a plot needs `N` bytes, the pool
// keeps `N` bytes available for that name forever.

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace pos2gpu {

class HostPinnedPool {
public:
    HostPinnedPool() = default;
    ~HostPinnedPool();

    HostPinnedPool(HostPinnedPool const&) = delete;
    HostPinnedPool& operator=(HostPinnedPool const&) = delete;
    HostPinnedPool(HostPinnedPool&&) = delete;
    HostPinnedPool& operator=(HostPinnedPool&&) = delete;

    // Returns a host-pinned buffer of at least `bytes` bytes for the
    // slot named `name`. If the slot exists and is large enough, the
    // existing pointer is returned (no allocation). If it exists but is
    // too small, the old buffer is freed and a fresh one is allocated.
    // If it doesn't exist, a fresh buffer is allocated.
    //
    // Throws std::runtime_error if sycl::malloc_host returns nullptr.
    void* acquire(std::string_view name, std::size_t bytes, sycl::queue& q);

    // Typed convenience wrapper.
    template <typename T>
    T* acquire_as(std::string_view name, std::size_t count, sycl::queue& q)
    {
        return static_cast<T*>(acquire(name, count * sizeof(T), q));
    }

    // Mark the pool's slots as reusable for the next plot. Currently a
    // no-op since slots are always reusable by the same name; reserved
    // for a future high-water-mark policy.
    void release_all() noexcept {}

    // Total bytes currently pinned across all slots. For diagnostics.
    std::size_t live_bytes() const noexcept;

    // Number of named slots. For tests.
    std::size_t slot_count() const noexcept;

private:
    struct Slot {
        void*        ptr     = nullptr;
        std::size_t  bytes   = 0;
        sycl::queue* alloc_q = nullptr; // queue used to allocate; reused for free
    };
    std::unordered_map<std::string, Slot> slots_;
};

} // namespace pos2gpu
