// MultiGpuShardBufferPool.hpp — per-shard device buffer cache that
// persists across plots in a batch run.
//
// The sharded pipeline allocates the same set of large device buffers
// for every plot — d_full_xs (~2 GB at k=28 for the replicated Xs),
// d_t1/t2/t3 staging (1–3 GB each), the sort-output buffers, etc.
// Without a pool, each plot pays the malloc + free cost; on real
// multi-GPU runs that's tens to hundreds of milliseconds per plot.
//
// This pool caches device allocations keyed by a string label. Each
// `ensure<T>(name, n)` returns a T* of capacity ≥ n, reusing the
// previous allocation if it's already large enough. On size growth
// the slot is freed and reallocated. The pool's destructor frees
// every cached allocation.
//
// Pools are owned by run_batch_sharded (one per shard) and threaded
// into MultiGpuShardContext. When a pool is attached, the pipeline
// routes its per-phase allocations through ensure<T>() instead of
// raw sycl::malloc_device. When it isn't (e.g. the parity tests),
// the pipeline falls back to per-plot malloc + free.
//
// Thread-safety: not thread-safe. Each shard owns its own pool; no
// cross-shard sharing.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include <sycl/sycl.hpp>

namespace pos2gpu {

class ShardBufferPool {
public:
    ShardBufferPool() = default;
    explicit ShardBufferPool(sycl::queue* q) : q_(q) {}
    ShardBufferPool(ShardBufferPool const&) = delete;
    ShardBufferPool& operator=(ShardBufferPool const&) = delete;
    ShardBufferPool(ShardBufferPool&&) = default;
    ShardBufferPool& operator=(ShardBufferPool&&) = default;
    ~ShardBufferPool() { clear(); }

    void attach(sycl::queue* q) noexcept { q_ = q; }

    // Returns a T* of capacity >= n, reusing a previous allocation
    // under the same label if it's already large enough. Capacity
    // growth triggers a free + alloc; downsizing is a no-op (we keep
    // the larger buffer for future plots).
    template <class T>
    T* ensure(std::string_view name, std::uint64_t n)
    {
        Slot& slot = slots_[std::string(name)];
        std::size_t const bytes = static_cast<std::size_t>(n) * sizeof(T);
        if (slot.bytes < bytes) {
            if (slot.ptr) sycl::free(slot.ptr, *q_);
            slot.ptr   = bytes == 0 ? nullptr
                                    : sycl::malloc_device(bytes, *q_);
            slot.bytes = bytes;
        }
        return static_cast<T*>(slot.ptr);
    }

    // Untyped variant for the match-prepare scratch (sycl::malloc_device
    // returns void*).
    void* ensure_bytes(std::string_view name, std::size_t bytes)
    {
        Slot& slot = slots_[std::string(name)];
        if (slot.bytes < bytes) {
            if (slot.ptr) sycl::free(slot.ptr, *q_);
            slot.ptr   = bytes == 0 ? nullptr
                                    : sycl::malloc_device(bytes, *q_);
            slot.bytes = bytes;
        }
        return slot.ptr;
    }

    // Pinned-host equivalent. The replicate-full step in each match
    // phase needs a multi-GB pinned host bounce buffer; without
    // pooling, every phase pays a fresh malloc_host (which page-locks
    // the range — non-trivial cost, especially on Linux). Pinned host
    // memory is process-wide regardless of which device queue did the
    // alloc, so a single per-pool host slot is sufficient.
    template <class T>
    T* ensure_host(std::string_view name, std::uint64_t n)
    {
        return static_cast<T*>(
            ensure_host_bytes(name, static_cast<std::size_t>(n) * sizeof(T)));
    }

    void* ensure_host_bytes(std::string_view name, std::size_t bytes)
    {
        Slot& slot = host_slots_[std::string(name)];
        if (slot.bytes < bytes) {
            if (slot.ptr) sycl::free(slot.ptr, *q_);
            slot.ptr   = bytes == 0 ? nullptr
                                    : sycl::malloc_host(bytes, *q_);
            slot.bytes = bytes;
        }
        return slot.ptr;
    }

    // Free every cached slot. Called from the destructor; useful
    // explicitly when the caller wants to reclaim VRAM between
    // batches without destroying the pool object.
    void clear() noexcept
    {
        if (!q_) return;
        for (auto& [name, slot] : slots_) {
            if (slot.ptr) sycl::free(slot.ptr, *q_);
        }
        slots_.clear();
        for (auto& [name, slot] : host_slots_) {
            if (slot.ptr) sycl::free(slot.ptr, *q_);
        }
        host_slots_.clear();
    }

    // Total bytes currently held by all cached slots — useful for
    // debug/log reporting. Includes both device and pinned-host slots.
    std::size_t total_bytes() const noexcept
    {
        std::size_t total = 0;
        for (auto const& [name, slot] : slots_)      total += slot.bytes;
        for (auto const& [name, slot] : host_slots_) total += slot.bytes;
        return total;
    }

private:
    struct Slot {
        void*       ptr   = nullptr;
        std::size_t bytes = 0;
    };

    sycl::queue*                q_ = nullptr;
    std::map<std::string, Slot> slots_;
    std::map<std::string, Slot> host_slots_;
};

} // namespace pos2gpu
