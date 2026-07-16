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
// Plotter (no CUDA calls, no GPU at runtime) on host NUMA node 0. Set
// by `cpu` / `cpu0` in --devices, or by --cpu.
//
// Historically this was "the CPU", singular — correct, because every
// box we ran on had one node. It is now node 0 specifically, and the
// value is unchanged so a single-node host encodes exactly as before.
inline constexpr int kCpuDeviceId = -2;

// CPU node n encodes as kCpuDeviceId - n: node 0 = -2, node 1 = -3, ...
// Growing DOWNWARD keeps node 0 on the historical -2 and leaves -1
// (kDefaultGpuId) alone. Negative ids below -1 are therefore CPU nodes
// and nothing else — do not add a third sentinel family here without
// changing this encoding, because is_cpu_device() claims the whole range.
inline constexpr int cpu_device_id(int numa_node)
{
    return kCpuDeviceId - numa_node;
}

inline constexpr bool is_cpu_device(int device_id)
{
    return device_id <= kCpuDeviceId;
}

// Which NUMA node a CPU device id names. Undefined for non-CPU ids.
inline constexpr int cpu_numa_node(int device_id)
{
    return kCpuDeviceId - device_id;
}

// The sentinels above are an internal encoding and must never reach a log
// line: printing device_ids with a bare %d renders a CPU worker as "-2",
// which reads as a mangled flag rather than a device. Everything user-facing
// goes through here.
//
// CPU nodes render "cpu0", "cpu1" — symmetric with "gpu0", "gpu1", and the
// same spelling --devices accepts back. A single-node host still shows "cpu0"
// rather than a bare "cpu": one host having one node is not a reason to give
// its worker a different KIND of name than a two-node host's would get.
inline std::string device_label(int device_id)
{
    if (device_id == kDefaultGpuId) return "gpu";
    if (is_cpu_device(device_id))   return "cpu" + std::to_string(cpu_numa_node(device_id));
    return "gpu" + std::to_string(device_id);
}

} // namespace pos2gpu
