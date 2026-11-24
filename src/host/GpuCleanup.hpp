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
    uint64_t unwinding_host = 0, unwinding_device = 0;
    bool terminal = false;
    bool (*reported_loss)() noexcept = nullptr;

    static void add(uint64_t& charge, uint64_t bytes) noexcept
    { charge += std::min(bytes, UINT64_MAX - charge); }

    void confirm_loss() noexcept
    {
        terminal = true;
        add(host, std::exchange(unwinding_host, 0));
        add(device, std::exchange(unwinding_device, 0));
    }
    void clear_unwind() noexcept { unwinding_host = unwinding_device = 0; }
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

// AdaptiveCpp can report a failed native deallocation asynchronously and
// return normally. Once a worker's context is lost, that return cannot prove
// the pages reusable. Before a synchronous loss is classified by its caller,
// keep a provisional total for the allocations released during stack unwind.
template<class Free>
void checked_gpu_free(uint64_t bytes, uint64_t GpuCleanupCharges::*kind, Free&& free, bool partial = false)
{
    auto* charges = gpu_cleanup_charges;
    try { free(); }
    catch (...) {
        if (charges) GpuCleanupCharges::add(charges->*kind, bytes);
        throw;
    }
    if (!charges) return;
    if (charges->terminal || (charges->reported_loss && charges->reported_loss()))
        GpuCleanupCharges::add(charges->*kind, bytes);
    else if (partial || std::uncaught_exceptions())
        GpuCleanupCharges::add(kind == &GpuCleanupCharges::host ? charges->unwinding_host : charges->unwinding_device, bytes);
}

template<class Free>
void release_gpu_memory(char const* name, uint64_t bytes, uint64_t GpuCleanupCharges::*kind, Free&& free, bool partial = false) noexcept
{
    try { checked_gpu_free(bytes, kind, std::forward<Free>(free), partial); }
    catch (std::exception const& error) { std::fprintf(stderr, "%s cleanup failed: %s\n", name, error.what()); }
    catch (...) { std::fprintf(stderr, "%s cleanup failed: unknown exception\n", name); }
}
}
