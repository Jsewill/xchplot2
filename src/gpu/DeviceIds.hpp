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

// Routes queue() to sycl::cpu_selector_v — AdaptiveCpp's OMP backend
// on the CPU build path (ACPP_TARGETS=omp). BatchPlotter pushes this
// into device_ids when --cpu (or `cpu` in --devices) is requested,
// so the multi-device fan-out treats CPU like just-another-device.
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
