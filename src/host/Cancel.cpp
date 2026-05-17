// Cancel.cpp — implementation of the SIGINT/SIGTERM/SIGHUP cancel flag.

#include "host/Cancel.hpp"

#include <csignal>

#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>  // write(2)
#endif

namespace pos2gpu {

namespace {

// sig_atomic_t is the one type C/C++ guarantee is safe to read/write from
// a signal handler without synchronization concerns. The count lets us
// turn the second same-signal receipt into a hard kill, so a user whose
// cooperative shutdown is stuck can still escape with a second Ctrl-C.
volatile std::sig_atomic_t g_cancel_count = 0;

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
    if (g_cancel_count >= 1) {
        std::signal(sig, SIG_DFL);
        std::raise(sig);
        return;
    }
    g_cancel_count = 1;
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
}

bool cancel_requested() noexcept
{
    return g_cancel_count > 0;
}

void reset_cancel_for_tests() noexcept
{
    g_cancel_count = 0;
}

} // namespace pos2gpu
