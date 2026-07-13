// DeviceIds.hpp — synthetic device-id sentinels shared between the
// CLI / BatchPlotter (host code) and the per-thread device binder.
// Real GPU ids are 0..N-1; negative values are reserved for selectors
// that don't correspond to a numbered device.
//
// On main this header lives alongside SyclBackend.hpp and is consumed
// by the SYCL queue() routing. The cuda-only branch has no SYCL
// backend — only kCpuDeviceId is used here, by BatchPlotter, to
// dispatch a CPU worker through pos2-chip's Plotter (see CpuPlotter.cpp).

#pragma once

#include <string>

namespace pos2gpu {

// The thread-local CUDA device starts unbound; bind_current_device()
// in BatchPlotter pins it to a specific GPU index when the worker
// is GPU-backed. -1 means "use whatever CUDA picks by default".
inline constexpr int kDefaultGpuId = -1;

// Routes BatchPlotter to dispatch the worker through pos2-chip's CPU
// Plotter (no CUDA calls, no GPU at runtime). Set by --cpu or by
// passing `cpu` as a token in --devices.
inline constexpr int kCpuDeviceId = -2;

// The sentinels above are an internal encoding and must never reach a log
// line: printing device_ids with a bare %d renders the CPU worker as "-2",
// which reads as a mangled flag rather than a device. Everything user-facing
// goes through here.
inline std::string device_label(int device_id)
{
    if (device_id == kCpuDeviceId)   return "cpu";
    if (device_id == kDefaultGpuId)  return "gpu";
    return "gpu" + std::to_string(device_id);
}

} // namespace pos2gpu
