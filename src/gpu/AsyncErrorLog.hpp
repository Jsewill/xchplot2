// AsyncErrorLog.hpp — process-wide tally of asynchronous backend errors.
//
// Deliberately free of any SYCL include so non-GPU code (the plot writer, in
// particular) can ask "did the device fault during this run?" without taking a
// dependency on the backend. SyclBackend.hpp's async_error_handler is the only
// writer; everything else reads.
//
// Why this exists: a backend that fails every kernel launch and a backend that
// runs every kernel but miscompiles it converge on the SAME downstream symptom
// — a phase that produced too little output — and the correct response to each
// is the opposite of the other. One is a driver/device fault that parity tests
// cannot reproduce; the other is exactly what parity tests are for. Without a
// record that launches failed, every consumer downstream is left guessing, and
// they guessed wrong: an Intel Arc that lost its device mid-run was reported as
// an AMD codegen bug, twice, in two different places.

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <functional>
#include <initializer_list>
#include <vector>
#include <mutex>
#include <string>
#include <string_view>

namespace pos2gpu {

struct AsyncErrorLog {
    std::atomic<unsigned> count{0};
    std::mutex            mu;
    std::string           first, device_loss;   // guarded by mu
    std::string_view      loss_backend;         // native backend of device_loss, guarded by mu
    std::string           backend;              // immutable while registered
    std::atomic<bool>     generating{false};
    std::function<void()> wake;
};

// A compute lane owns its diagnostics across its queue and output threads.
// Ordinary plotters continue to share the process-wide error record.
inline std::shared_ptr<AsyncErrorLog>& scoped_async_errors()
{
    thread_local std::shared_ptr<AsyncErrorLog> value;
    return value;
}
struct ScopedAsyncErrors {
    std::shared_ptr<AsyncErrorLog> previous;
    explicit ScopedAsyncErrors(std::shared_ptr<AsyncErrorLog> value) noexcept
        : previous(std::move(scoped_async_errors())) { scoped_async_errors() = std::move(value); }
    ~ScopedAsyncErrors() { scoped_async_errors() = std::move(previous); }
    ScopedAsyncErrors(ScopedAsyncErrors const&) = delete;
    ScopedAsyncErrors& operator=(ScopedAsyncErrors const&) = delete;
};

inline std::string_view lost_gpu_backend(std::string_view error) noexcept
{
    auto has = [&](std::initializer_list<char const*> names) {
        for (auto name : names) if (error.find(name) != error.npos) return true;
        return false;
    };
    if (has({"ZE_RESULT_ERROR_DEVICE_LOST", "0x70000001", "ze:1879048193)"})) return "level_zero";
    if (has({"CUDA_ERROR_CONTEXT_IS_DESTROYED", "CUDA_ERROR_ILLEGAL_ADDRESS", "CUDA_ERROR_LAUNCH_FAILED",
             "cudaErrorIllegalAddress", "cudaErrorLaunchFailure", "CUDA:700)", "CUDA:709)", "CUDA:719)"})) return "cuda";
    if (has({"hipErrorContextIsDestroyed", "hipErrorIllegalAddress", "hipErrorLaunchFailure",
             "HIP:700)", "HIP:709)", "HIP:719)"})) return "hip";
    return {};
}
inline bool is_gpu_device_loss(std::string_view message) noexcept { return !lost_gpu_backend(message).empty(); }

struct WorkerErrorRegistration
{
    static auto& registry()
    {
        struct Registry { std::mutex mutex; std::vector<AsyncErrorLog*> workers; };
        static Registry value;
        return value;
    }
    AsyncErrorLog& log;
    WorkerErrorRegistration(AsyncErrorLog& entry, std::string backend, std::function<void()> wake) : log(entry)
    {
        log.backend = std::move(backend); log.wake = std::move(wake);
        auto& list = registry(); std::lock_guard lock(list.mutex); list.workers.push_back(&log);
    }
    ~WorkerErrorRegistration()
    {
        auto& list = registry(); std::lock_guard lock(list.mutex); std::erase(list.workers, &log); log.wake = {};
    }
    WorkerErrorRegistration(WorkerErrorRegistration const&) = delete;
    WorkerErrorRegistration& operator=(WorkerErrorRegistration const&) = delete;
};

inline AsyncErrorLog& async_errors()
{
    static AsyncErrorLog log;
    return scoped_async_errors() ? *scoped_async_errors() : log;
}

// Record one async backend error. noexcept: diagnostics must never be the
// thing that kills a run.
inline void record_async_error(char const* what) noexcept
{
    try {
        auto& reported = async_errors();
        auto const backend = what ? lost_gpu_backend(what) : std::string_view{};
        auto record = [&](AsyncErrorLog& log) {
            {
                std::lock_guard lock(log.mu);
                if (log.first.empty() && what) log.first = what;
                if (log.device_loss.empty() && !backend.empty())
                {
                    log.loss_backend = backend;
                    auto const message = std::string_view(what);
                    log.device_loss = message.substr(message.size() > 240 ? message.size() - 240 : 0);
                }
                log.count.fetch_add(1, std::memory_order_relaxed);
            }
            if (!backend.empty() && log.wake) log.wake();
        };
        if (backend.empty()) { record(reported); return; }
        // AdaptiveCpp pops a process-wide async list into the queue that happens
        // to wait first. A native error names its backend, not necessarily that
        // queue's device. Fence all in-flight lanes of that backend when ambiguous.
        auto& list = WorkerErrorRegistration::registry(); std::lock_guard lock(list.mutex);
        bool any = false, reporting_lane = false;
        for (auto* log : list.workers) if (log->backend == backend && log->generating.load()) {
            record(*log); any = true; reporting_lane |= log == &reported;
        }
        if (!reporting_lane && reported.backend == backend) { record(reported); any = true; }
        if (!any) for (auto* log : list.workers) if (log->backend == backend) { record(*log); any = true; }
        if (reported.backend.empty()) record(reported);
        else if (!any) { ScopedAsyncErrors fallback(nullptr); record(async_errors()); }
    } catch (...) {
    }
}

inline unsigned async_error_count() noexcept
{
    return async_errors().count.load(std::memory_order_relaxed);
}

inline std::string async_device_loss(std::string_view backend = {})
{
    auto& log = async_errors();
    if (!log.count.load(std::memory_order_relaxed)) return {};
    std::lock_guard lock(log.mu);
    return backend.empty() || log.loss_backend == backend ? log.device_loss : std::string{};
}

inline void clear_worker_allocation_error()
{
    auto& log = async_errors();
    std::lock_guard lock(log.mu);
    if (!log.device_loss.empty()) return;
    for (auto marker : {"OUT_OF_DEVICE_MEMORY", "OUT_OF_HOST_MEMORY", "OUT_OF_MEMORY", "OutOfMemory",
                        "MemoryAllocation", "ze:1879048195)", "ze:1879048194)", "CUDA:2)", "HIP:2)"})
        if (log.first.find(marker) != std::string::npos)
        { log.first.clear(); log.count.store(0, std::memory_order_relaxed); return; }
}

inline std::string first_async_error()
{
    auto& log = async_errors();
    try {
        std::lock_guard<std::mutex> lk(log.mu);
        return log.first;
    } catch (...) {
        return {};
    }
}

}  // namespace pos2gpu
