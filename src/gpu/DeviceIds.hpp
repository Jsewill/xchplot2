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

} // namespace pos2gpu
