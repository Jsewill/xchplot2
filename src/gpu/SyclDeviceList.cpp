// SyclDeviceList.cpp — implementation of list_gpu_devices().
// Compiled by acpp via add_sycl_to_target so the SYCL headers are in
// scope here; the public-facing header (SyclDeviceList.hpp) carries
// only plain types for non-acpp consumers like cli.cpp.

#include "gpu/SyclDeviceList.hpp"
#include "gpu/SyclBackend.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <iomanip>
#include <sstream>
#include <string_view>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pos2gpu {
namespace {

std::string pci_identity(std::string_view address)
{
    // CUDA and HIP use domain:bus:device.function. Normalize optional wider
    // domain padding and reject incomplete/overflowed responses.
    std::array<unsigned, 4> fields{};
    for (size_t i = 0; i < fields.size(); ++i) {
        auto const separator = i < 2 ? ':' : '.';
        auto const end = i == 3 ? address.size() : address.find(separator);
        if (end == address.npos || !end) return {};
        auto const result = std::from_chars(address.data(), address.data() + end, fields[i], 16);
        if (result.ec != std::errc{} || result.ptr != address.data() + end) return {};
        address.remove_prefix(end + (i != 3));
    }
    if (!address.empty() || fields[0] > 65535 || fields[1] > 255 || fields[2] > 31 || fields[3] > 7) return {};
    std::ostringstream out;
    out << "pci:" << std::hex << std::setfill('0') << std::setw(4) << fields[0] << ':'
        << std::setw(2) << fields[1] << ':' << std::setw(2) << fields[2] << '.' << fields[3];
    return out.str();
}

std::string physical_identity(sycl::device const& device)
{
    auto const backend = device.get_backend();
    if (backend != sycl::backend::cuda && backend != sycl::backend::hip) return {};
    // AdaptiveCpp's native CUDA/HIP interop uses this device's runtime ID.
    // Counting devices of the same vendor would lose visibility-mask mapping.
    int const native = device.AdaptiveCpp_device_id().get_id();
    if (native < 0) return {};
    char const* symbol = backend == sycl::backend::cuda ? "cudaDeviceGetPCIBusId" : "hipDeviceGetPCIBusId";
#ifdef _WIN32
    using Query = int (__stdcall *)(char*, int, int);
    std::array<wchar_t const*, 3> const names = backend == sycl::backend::cuda
        ? std::array{L"cudart64_13.dll", L"cudart64_12.dll", L"cudart64_110.dll"}
        : std::array{L"amdhip64_7.dll", L"amdhip64_6.dll", L"amdhip64.dll"};
    for (auto name : names) {
        auto const module = ::GetModuleHandleW(name);
        if (!module) continue;
        auto const query = reinterpret_cast<Query>(::GetProcAddress(module, symbol));
#else
    using Query = int (*)(char*, int, int);
    std::array<char const*, 4> const names = backend == sycl::backend::cuda
        ? std::array{"libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"}
        : std::array{"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5"};
    for (auto name : names) {
        // The enumerated backend already owns its runtime. Query that loaded
        // library, without loading another installation for an identity guess.
        void* const module = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        if (!module) continue;
        struct Close { void* value; ~Close() { ::dlclose(value); } } close{module};
        auto const query = reinterpret_cast<Query>(::dlsym(module, symbol));
#endif
        if (!query) continue;
        std::array<char, 64> address{};
        if (query(address.data(), int(address.size()), native)) return {};
        auto const end = std::find(address.begin(), address.end(), '\0');
        if (end == address.end()) return {};
        return pci_identity(std::string_view(address.data(), size_t(end - address.begin())));
    }
    return {};
}
} // namespace

std::string gpu_physical_identity(int device_id)
{
    auto const devices = sycl_backend::usable_gpu_devices();
    if (device_id < 0 || size_t(device_id) >= devices.size()) return {};
    return physical_identity(devices[size_t(device_id)]);
}

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
        info.host_unified_memory = d.get_info<sycl::info::device::host_unified_memory>();
        info.cu_count        = static_cast<unsigned>(
                                   d.get_info<sycl::info::device::max_compute_units>());
        info.auto_dispatchable = sycl_backend::is_auto_dispatchable(d);
        info.physical_identity = physical_identity(d);
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
