// DeviceIds.hpp — synthetic device-id sentinels shared between the
// CLI / BatchPlotter (host code) and SyclBackend (per-thread queue
// routing). Real GPU ids are 0..N-1; negative values are reserved
// for selectors that don't correspond to a numbered device.
//
// Lives in src/gpu/ rather than src/host/ because SyclBackend.hpp
// (which can't include host-side headers) is the authoritative
// consumer; BatchPlotter / cli.cpp pull the same constants from
// here so the two sides agree on the encoding.

#pragma once

#include <string>

namespace pos2gpu {

// Default thread-local value of sycl_backend::current_device_id_ref().
// queue() picks sycl::gpu_selector_v in this case — the single-device
// zero-config path users see when --devices is not passed.
inline constexpr int kDefaultGpuId = -1;

// A CPU plotter on host NUMA node 0. BatchPlotter pushes this into
// device_ids when `cpu` / `cpu0` appears in --devices, so the
// multi-device fan-out treats a CPU node like just-another-device.
//
// Historically this was "the CPU", singular — correct, because every
// box we ran on had one node. It is now node 0 specifically, and the
// value is unchanged so a single-node host encodes exactly as before.
inline constexpr int kCpuDeviceId = -2;

// CPU node n encodes as kCpuDeviceId - n: node 0 = -2, node 1 = -3, ...
// Growing DOWNWARD keeps node 0 on the historical -2 and leaves -1
// (kDefaultGpuId) alone. Negative ids below -1 are therefore CPU nodes and
// nothing else — do not add a third sentinel family here without changing
// this encoding, because is_cpu_device() below claims the whole range.
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
