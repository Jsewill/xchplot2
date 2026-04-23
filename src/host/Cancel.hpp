// Cancel.hpp — SIGINT/SIGTERM handling for long-running batches.
//
// install_cancel_signal_handlers() installs handlers that set an
// async-signal-safe flag on first receipt and restore the default
// disposition on second receipt (so double-Ctrl-C kills hard).
//
// cancel_requested() is cheap enough to call from tight loops.

#pragma once

namespace pos2gpu {

// Install SIGINT + SIGTERM handlers. Idempotent — safe to call more than
// once. First signal sets the cancel flag and prints a one-line notice
// via write(2) (async-signal-safe). Second signal of the same type
// re-raises with the default disposition, terminating the process.
void install_cancel_signal_handlers();

// True if a cancelling signal has been received since program start
// (or since reset_cancel_for_tests()).
bool cancel_requested() noexcept;

// Testing hook — clear the flag. Not intended for production code.
void reset_cancel_for_tests() noexcept;

} // namespace pos2gpu
