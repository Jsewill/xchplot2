#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
namespace sycl {
namespace info::device { struct global_mem_size {}; }
struct device { template<class T> std::uint64_t get_info() const {return 8ULL<<30;} };
struct queue { void wait() {} };
struct exception : std::runtime_error {using std::runtime_error::runtime_error;};
enum class allocation_failure { none, null, exception };
inline allocation_failure fail_allocation = allocation_failure::none;
inline std::mutex allocation_mutex;
inline std::map<void*, bool> allocations;
inline void* allocate(std::size_t n, bool device)
{
    if (fail_allocation == allocation_failure::exception) throw exception("injected allocation failure");
    if (fail_allocation == allocation_failure::null) return nullptr;
    auto* p = std::malloc(n);
    if (p) {
        std::lock_guard lock(allocation_mutex);
        allocations.emplace(p, device);
    }
    return p;
}
inline bool is_device_allocation(void* p)
{
    std::lock_guard lock(allocation_mutex);
    return allocations.at(p);
}
inline void* malloc_host(std::size_t n,queue&){return allocate(n, false);}
inline void* malloc_device(std::size_t n,queue&){return allocate(n, true);}
template<class T> T* malloc_device(std::size_t n,queue& q){return static_cast<T*>(malloc_device(n*sizeof(T),q));}
inline void free(void* p,queue&)
{
    if (!p) return;
    {
        std::lock_guard lock(allocation_mutex);
        if (allocations.erase(p) != 1) std::abort();
    }
    std::free(p);
}
}
