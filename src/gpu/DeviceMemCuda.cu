// DeviceMemCuda.cu — real free/total VRAM query for the NVIDIA path.
//
// SYCL has no portable free-memory query and AdaptiveCpp exposes none, so
// query_device_memory() in GpuBufferPool.cpp reported the device *total* as
// "free". Every tier and pool sizing decision was therefore made against a
// number that ignored the CUDA context, the display server, and every other
// process on the card.
//
// cudaMemGetInfo is a context-level query that sees all of it, and SYCL +
// CUDA host code share the same primary CUDA context (see the note at the
// top of GpuBufferPool.cpp — the pool used to call this directly, and lost
// it when the file was migrated from .cu to .cpp and the CUDA runtime call
// became unavailable in a SYCL-only TU). What cudaMemGetInfo reports is
// what sycl::malloc_device will actually be able to hand out.
//
// Returns false on any CUDA error, leaving the caller on its
// global_mem_size fallback, so a driver-less or non-NVIDIA host degrades to
// the old behaviour rather than failing the run.

#include <cuda_runtime.h>

#include <cstddef>

namespace pos2gpu {

bool cuda_query_device_memory(int device_ordinal,
                              std::size_t& free_bytes,
                              std::size_t& total_bytes)
{
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    // Only touch the thread's current device when we actually have to.
    // Worker threads in the multi-GPU fan-out are each bound to their own
    // device, and cudaMemGetInfo reports for whichever device is current.
    bool const switch_device = (device_ordinal >= 0 && device_ordinal != prev);
    if (switch_device && cudaSetDevice(device_ordinal) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    std::size_t f = 0;
    std::size_t t = 0;
    cudaError_t const rc = cudaMemGetInfo(&f, &t);

    if (switch_device) {
        cudaSetDevice(prev);
    }
    // Clear any sticky error so a failure here cannot resurface later as a
    // spurious fault inside an unrelated kernel launch.
    cudaGetLastError();

    if (rc != cudaSuccess || t == 0) return false;

    free_bytes  = f;
    total_bytes = t;
    return true;
}

} // namespace pos2gpu
