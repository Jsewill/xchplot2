#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
namespace sycl {
namespace info::device { struct global_mem_size {}; }
struct device { template<class T> std::uint64_t get_info() const {return 8ULL<<30;} };
struct queue { void wait() {} };
struct exception : std::runtime_error {using std::runtime_error::runtime_error;};
inline void* malloc_host(std::size_t n,queue&){return std::malloc(n);}
inline void* malloc_device(std::size_t n,queue&){return std::malloc(n);}
template<class T> T* malloc_device(std::size_t n,queue&){return static_cast<T*>(std::malloc(n*sizeof(T)));}
inline void free(void* p,queue&){std::free(p);}
}
