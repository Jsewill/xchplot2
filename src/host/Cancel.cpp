// Cancel.cpp — implementation of the SIGINT/SIGTERM/SIGHUP cancel flag.

#include "host/Cancel.hpp"

#include <atomic>
#include <csignal>

#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>  // write(2)
#endif

namespace pos2gpu {

namespace {

// Lock-free std::atomic is both async-signal-safe (per [support.signal])
// and safe for cross-thread access — request_cancel() is called
// concurrently from several worker threads, which `volatile
// sig_atomic_t` (safe only between a handler and the thread it
// interrupts) did not cover.
//
// Two separate flags, deliberately:
//   g_signal_count — bumped ONLY by the signal handler. The second
//     same-signal receipt escalates to a hard kill.
//   g_soft_cancel  — set by request_cancel() (e.g. a worker hitting
//     ENOSPC asks peers to drain). It must NOT count toward the
//     escalation: with a single counter, one programmatic cancel meant
//     the user's FIRST Ctrl-C hard-killed the process mid-plot.
std::atomic<int>  g_signal_count{0};
std::atomic<bool> g_soft_cancel{false};

void write_stderr_safe(char const* msg, std::size_t len) noexcept
{
#if defined(__unix__) || defined(__APPLE__)
    // write(2) is async-signal-safe; std::fprintf is not.
    ssize_t const rc = ::write(2, msg, len);
    (void)rc;  // nothing useful to do if stderr is gone
#else
    (void)msg;
    (void)len;
#endif
}

extern "C" void cancel_handler(int sig) noexcept
{
    // On the second receipt, restore the default disposition and re-raise
    // so the process dies immediately. Prevents a hung plotter from
    // needing kill -9 when the user insists.
    if (g_signal_count.fetch_add(1, std::memory_order_relaxed) >= 1) {
        std::signal(sig, SIG_DFL);
        std::raise(sig);
        return;
    }
    static char const msg[] =
        "\n[xchplot2] cancel requested — finishing current plot then "
        "stopping. Press Ctrl-C again to abort immediately.\n";
    write_stderr_safe(msg, sizeof(msg) - 1);
}

} // namespace

void install_cancel_signal_handlers()
{
    std::signal(SIGINT,  cancel_handler);
    std::signal(SIGTERM, cancel_handler);
    // SIGHUP — sent when the controlling terminal disappears (SSH
    // disconnect, terminal closed). Without explicit handling, the
    // default disposition kills the process immediately, leaving any
    // in-flight CUDA contexts and kernels improperly torn down.
    // That path is a *suspected* cause of host-wide CUDA driver-
    // state corruption observed across runpod containers in 2026-05:
    // NVML still reports healthy GPUs but cuInit returns
    // cudaErrorInitializationError in every subsequent process.
    // Routing SIGHUP through the same cooperative cancel flag lets
    // the batch loop finish its current plot and drain CUDA cleanly
    // before exit. Hypothesis to verify: does this stop the wedge
    // from recurring after SSH-backgrounded plot runs?
#if defined(SIGHUP)
    std::signal(SIGHUP, cancel_handler);
#endif
    // SIGXFSZ → ignore. Linux sends this in addition to write(2)
    // returning EFBIG when RLIMIT_FSIZE is exceeded; the default
    // disposition kills the process before our writer's RAII guard
    // can remove the .partial file. Ignoring the signal lets
    // write() return EFBIG cleanly, ofstream sets failbit, our
    // throw chain in PlotFileWriterParallel fires, and the
    // PartialGuard destructor unlinks the partial as designed.
    // Same shape as ENOSPC handling, just a different trigger.
#if defined(SIGXFSZ)
    std::signal(SIGXFSZ, SIG_IGN);
#endif
    // SIGPIPE → ignore. The writer + reader stream paths use write()
    // not pipes, but downstream notify-style hooks (e.g. logging to
    // a fifo) shouldn't kill the process. Cheap insurance.
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
}

bool cancel_requested() noexcept
{
    return g_soft_cancel.load(std::memory_order_relaxed)
        || g_signal_count.load(std::memory_order_relaxed) > 0;
}

void request_cancel() noexcept
{
    // Separate soft flag — see the g_soft_cancel comment above for why
    // this must not feed the signal handler's escalation count.
    g_soft_cancel.store(true, std::memory_order_relaxed);
}

void reset_cancel_for_tests() noexcept
{
    g_signal_count.store(0, std::memory_order_relaxed);
    g_soft_cancel.store(false, std::memory_order_relaxed);
}

} // namespace pos2gpu
