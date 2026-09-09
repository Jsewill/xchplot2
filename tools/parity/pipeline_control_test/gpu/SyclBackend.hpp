#pragma once
#include <sycl/sycl.hpp>
#include <vector>
namespace pos2gpu::sycl_backend {
inline std::vector<sycl::device> usable_gpu_devices(){return {{},{},{}};}
inline sycl::queue& queue(){thread_local sycl::queue q;return q;}
}
