// Machine-readable CI inventory and k=28 budgets from the production backend.
#include "host/GpuBufferPool.hpp"
#ifdef XCHPLOT2_GPU_CI_SYCL
#include "gpu/SyclDeviceList.hpp"
#else
#include "gpu/CudaDeviceList.hpp"
#include "host/VramBudget.hpp"
#include <cuda_runtime_api.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    try {
        if (argc != 2)
            throw std::runtime_error("usage: gpu_ci_info cuda|hip|level_zero");
        if (std::getenv("POS2GPU_MAX_VRAM_MB"))
            throw std::runtime_error("CI inventory requires uncapped physical VRAM");
#ifdef XCHPLOT2_GPU_CI_SYCL
        auto const devices = pos2gpu::list_gpu_devices();
#else
        auto const query = pos2gpu::list_cuda_devices();
        if (!query.error.empty()) throw std::runtime_error(query.error);
        auto const& devices = query.devices;
#endif
        if (devices.size() != 1)
            throw std::runtime_error("GPU CI requires exactly one visible GPU; isolate the runner device");
#ifdef XCHPLOT2_GPU_CI_SYCL
        std::string const backend = devices.front().backend;
#else
        std::string const backend = "cuda";
#endif
        if (backend != argv[1])
            throw std::runtime_error("Expected " + std::string(argv[1]) + ", found " + backend);
#ifdef XCHPLOT2_GPU_CI_SYCL
        size_t const free_bytes = pos2gpu::query_device_memory().free_bytes;
#else
        size_t free_bytes = 0, total_bytes = 0;
        auto const status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
        if (!free_bytes || !devices.front().vram_bytes)
            throw std::runtime_error("GPU memory inventory is empty");
        std::fprintf(stderr, "GPU: %s (%s)\n", devices.front().name.c_str(), backend.c_str());
        std::printf("backend=%s\ntotal_bytes=%llu\nfree_bytes=%zu\nmargin_bytes=%zu\n",
                    backend.c_str(),
                    static_cast<unsigned long long>(devices.front().vram_bytes),
                    free_bytes, pos2gpu::vram_safety_margin());
#ifdef XCHPLOT2_GPU_CI_SYCL
        std::printf("tiny=%zu\nminimal=%zu\ncompact=%zu\nplain=%zu\n",
                    pos2gpu::streaming_tiny_peak_bytes(28),
                    pos2gpu::streaming_minimal_peak_bytes(28),
                    pos2gpu::streaming_peak_bytes(28),
                    pos2gpu::streaming_plain_peak_bytes(28));
        std::printf("pinned=%zu\n", pos2gpu::streaming_pinned_peak_bytes(28));
        std::printf("spill_tiers=tiny,pinned,minimal,compact\n");
#else
        using pos2gpu::StreamingTier;
        std::printf("tiny=%zu\nminimal=%zu\ncompact=%zu\nplain=%zu\n",
                    size_t(pos2gpu::streaming_base_peak_bytes(28, StreamingTier::Tiny)),
                    size_t(pos2gpu::streaming_base_peak_bytes(28, StreamingTier::Minimal)),
                    size_t(pos2gpu::streaming_base_peak_bytes(28, StreamingTier::Compact)),
                    size_t(pos2gpu::streaming_base_peak_bytes(28, StreamingTier::Plain)));
        // HostRamPolicy: Compact uses SpillBuffer; Minimal uses file mappings.
        // Tiny can reduce drain slots, but cannot spill its GPU-visible host data.
        std::printf("spill_tiers=minimal,compact\n");
#endif
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "GPU CI inventory failed: %s\n", e.what());
        return 1;
    }
}
