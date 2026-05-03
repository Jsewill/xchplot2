// SyclDeviceList.hpp — plain-types declaration for `xchplot2 devices`
// (and any other consumer that needs to enumerate GPU devices without
// pulling <sycl/sycl.hpp> into its TU).
//
// cli.cpp is compiled by g++ with -Werror, and including SyclBackend.hpp
// drags in AdaptiveCpp's libkernel/host/builtins.hpp which has a
// narrowing-conversion warning that gets escalated to an error. Keeping
// this header SYCL-free lets non-acpp TUs query the device list via the
// implementation in SyclDeviceList.cpp (compiled by acpp).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pos2gpu {

struct GpuDeviceInfo {
    std::size_t   id;
    std::string   name;
    std::string   backend;          // "cuda" / "hip" / "level_zero" / "opencl" / "?"
    bool          is_cuda_backend;  // true iff backend == sycl::backend::cuda
    std::uint64_t vram_bytes;
    unsigned      cu_count;         // max_compute_units
};

// Enumerate every visible SYCL GPU device. Order matches what
// `--devices N` uses for index lookup, so the printed `[N]` is a
// drop-in for that flag.
std::vector<GpuDeviceInfo> list_gpu_devices();

} // namespace pos2gpu
