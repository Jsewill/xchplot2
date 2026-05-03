// SyclDeviceList.cpp — implementation of list_gpu_devices().
// Compiled by acpp via add_sycl_to_target so the SYCL headers are in
// scope here; the public-facing header (SyclDeviceList.hpp) carries
// only plain types for non-acpp consumers like cli.cpp.

#include "gpu/SyclDeviceList.hpp"
#include "gpu/SyclBackend.hpp"

namespace pos2gpu {

std::vector<GpuDeviceInfo> list_gpu_devices()
{
    std::vector<GpuDeviceInfo> out;
    auto devs = sycl_backend::usable_gpu_devices();
    out.reserve(devs.size());
    for (std::size_t i = 0; i < devs.size(); ++i) {
        auto const& d = devs[i];
        GpuDeviceInfo info{};
        info.id              = i;
        info.name            = d.get_info<sycl::info::device::name>();
        info.vram_bytes      = d.get_info<sycl::info::device::global_mem_size>();
        info.cu_count        = static_cast<unsigned>(
                                   d.get_info<sycl::info::device::max_compute_units>());
        info.is_cuda_backend = false;
        switch (d.get_backend()) {
            case sycl::backend::cuda:
                info.backend = "cuda";
                info.is_cuda_backend = true;
                break;
            case sycl::backend::hip:
                info.backend = "hip";
                break;
            case sycl::backend::level_zero:
                info.backend = "level_zero";
                break;
            default:
                info.backend = "?";
                break;
        }
        out.push_back(std::move(info));
    }
    return out;
}

} // namespace pos2gpu
