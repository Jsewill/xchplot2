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

#include <atomic>
#include <mutex>
#include <string>

namespace pos2gpu {

struct AsyncErrorLog {
    std::atomic<unsigned> count{0};
    std::mutex            mu;
    std::string           first;   // guarded by mu
};

inline AsyncErrorLog& async_errors()
{
    static AsyncErrorLog log;
    return log;
}

// Record one async backend error. noexcept: diagnostics must never be the
// thing that kills a run.
inline void record_async_error(char const* what) noexcept
{
    auto& log = async_errors();
    log.count.fetch_add(1, std::memory_order_relaxed);
    try {
        std::lock_guard<std::mutex> lk(log.mu);
        if (log.first.empty() && what) log.first = what;
    } catch (...) {
    }
}

inline unsigned async_error_count() noexcept
{
    return async_errors().count.load(std::memory_order_relaxed);
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
