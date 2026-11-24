// HostPinnedPool.cpp — see header for design.

#include "host/HostPinnedPool.hpp"
#include "host/GpuCleanup.hpp"

#include "host/GpuBufferPool.hpp"   // host_pinned_reserve_check

#include <stdexcept>
#include <string>

namespace pos2gpu {

HostPinnedPool::~HostPinnedPool()
{
    for (auto& [name, s] : slots_) {
        if (s.ptr && s.alloc_q) {
            release_gpu_memory(name.c_str(), s.bytes, &GpuCleanupCharges::host, [&] { sycl::free(s.ptr, *s.alloc_q); });
        }
    }
}

void* HostPinnedPool::acquire(std::string_view name, std::size_t bytes,
                              sycl::queue& q)
{
    if (bytes == 0) {
        throw std::runtime_error(
            "HostPinnedPool::acquire('" + std::string(name) + "'): bytes must be > 0");
    }
    std::string key(name);
    auto it = slots_.find(key);
    if (it != slots_.end() && it->second.bytes >= bytes) {
        return it->second.ptr;
    }
    if (it != slots_.end() && it->second.ptr && it->second.alloc_q) {
        auto* owned = std::exchange(it->second.ptr, nullptr);
        checked_gpu_free(std::exchange(it->second.bytes, 0), &GpuCleanupCharges::host,
            [&] { sycl::free(owned, *it->second.alloc_q); });
    }
    // After the free above, so a grown slot is charged only its delta against
    // what the host actually has left.
    host_pinned_reserve_check(bytes, std::string(name).c_str());
    void* p = sycl::malloc_host(bytes, q);
    if (!p) {
        throw std::runtime_error(
            "HostPinnedPool::acquire('" + std::string(name) + "'): "
            "sycl::malloc_host(" + std::to_string(bytes) + ") returned nullptr");
    }
    if (it == slots_.end()) {
        slots_.emplace(std::move(key), Slot{p, bytes, &q});
    } else {
        it->second.ptr     = p;
        it->second.bytes   = bytes;
        it->second.alloc_q = &q;
    }
    return p;
}

std::size_t HostPinnedPool::live_bytes() const noexcept
{
    std::size_t total = 0;
    for (auto const& [_, s] : slots_) total += s.bytes;
    return total;
}

std::size_t HostPinnedPool::slot_count() const noexcept
{
    return slots_.size();
}

} // namespace pos2gpu
