#include "host/VramProbe.hpp"
#include <cstring>
#include <windows.h>
#include <wrl/client.h>
#include <initguid.h>
#include <dxcore_interface.h>
#include <hip/hip_runtime_api.h>

namespace pos2gpu {
std::optional<size_t> windows_hip_worker_budget(HMODULE runtime, int ordinal, size_t hip_free, size_t hip_total, size_t resident)
{
#if HIP_VERSION_MAJOR >= 6
    using Properties = hipError_t (*)(hipDeviceProp_tR0600*, int);
    auto const properties = reinterpret_cast<Properties>(::GetProcAddress(runtime, "hipGetDevicePropertiesR0600"));
    hipDeviceProp_tR0600 device{};
    if (!properties || properties(&device, ordinal) != hipSuccess) return {};
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(device.luid));
    std::memcpy(&luid, device.luid, sizeof(luid));
    if (!luid.LowPart && !luid.HighPart) return {};

    struct Library {
        HMODULE handle = ::LoadLibraryExW(L"dxcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        ~Library() { if (handle) ::FreeLibrary(handle); }
    } library;
    if (!library.handle) return {};
    using CreateFactory = HRESULT (WINAPI *)(REFIID, void**);
    auto const create = reinterpret_cast<CreateFactory>(::GetProcAddress(library.handle, "DXCoreCreateAdapterFactory"));
    Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> factory;
    Microsoft::WRL::ComPtr<IDXCoreAdapter> adapter;
    if (!create || FAILED(create(IID_IDXCoreAdapterFactory, reinterpret_cast<void**>(factory.GetAddressOf()))) ||
        FAILED(factory->GetAdapterByLuid(luid, IID_IDXCoreAdapter, reinterpret_cast<void**>(adapter.GetAddressOf())))) return {};

    // Some HIP 6 drivers return a LUID with no node mask. A single physical
    // node on the matching logical adapter identifies node 0 unambiguously.
    uint32_t node_count = 0;
    if (FAILED(adapter->GetProperty(DXCoreAdapterProperty::PhysicalAdapterCount, &node_count))) return {};
    auto const node = windows_hip_worker_node(device.luidDeviceNodeMask, node_count);
    if (!node) return {};

    uint64_t dedicated = 0;
    DXCoreAdapterMemoryBudget budget{};
    DXCoreMemoryUsage usage{};
    DXCoreAdapterMemoryBudgetNodeSegmentGroup const local{*node, DXCoreSegmentGroup::Local};
    DXCoreMemoryQueryInput const physical{*node, DXCoreMemoryType::Dedicated};
    if (FAILED(adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory, &dedicated)) ||
        FAILED(adapter->QueryState(DXCoreAdapterState::AdapterMemoryBudget, &local, &budget)) ||
        FAILED(adapter->QueryState(DXCoreAdapterState::AdapterMemoryUsageBytes, &physical, &usage))) return {};
    return windows_hip_worker_headroom(hip_free, budget.budget, budget.currentUsage,
        (std::min<uint64_t>)(dedicated, hip_total), (std::max)(usage.committed, usage.resident), resident);
#else
    return {};
#endif
}
} // namespace pos2gpu
