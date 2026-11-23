// Destruction can reach a failed driver while the process still owns other workers.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <utility>

namespace pos2gpu
{
struct GpuCleanupCharges
{
    uint64_t host = 0, device = 0;
};
inline thread_local GpuCleanupCharges* gpu_cleanup_charges = nullptr;

class ObserveGpuCleanup
{
    GpuCleanupCharges* previous_;
public:
    explicit ObserveGpuCleanup(GpuCleanupCharges& charges) noexcept
        : previous_(std::exchange(gpu_cleanup_charges, &charges)) {}
    ~ObserveGpuCleanup() { gpu_cleanup_charges = previous_; }
    ObserveGpuCleanup(ObserveGpuCleanup const&) = delete;
    ObserveGpuCleanup& operator=(ObserveGpuCleanup const&) = delete;
};

// A driver may already have consumed an allocation when a free reports failure.
// Retain its charge and preserve the original exception for ordinary releases.
template<class Free>
void checked_gpu_free(uint64_t bytes, uint64_t GpuCleanupCharges::*kind, Free&& free)
{
    try { free(); }
    catch (...) {
        if (auto* charges = gpu_cleanup_charges) {
            auto& retained = charges->*kind;
            retained += std::min(bytes, UINT64_MAX - retained);
        }
        throw;
    }
}

template<class Free>
void release_gpu_memory(char const* name, uint64_t bytes, uint64_t GpuCleanupCharges::*kind, Free&& free) noexcept
{
    try { checked_gpu_free(bytes, kind, std::forward<Free>(free)); }
    catch (std::exception const& error) { std::fprintf(stderr, "%s cleanup failed: %s\n", name, error.what()); }
    catch (...) { std::fprintf(stderr, "%s cleanup failed: unknown exception\n", name); }
}
}
