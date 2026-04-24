// xs_bench — wall-clock CPU XsConstructor vs GPU launch_construct_xs at
// k=24 (16M entries) and k=26 (64M entries). Single-shot per size; not
// statistically rigorous, but enough to confirm there's a real speedup to
// chase further down the pipeline.

#include "gpu/AesGpu.cuh"
#include "gpu/SyclBackend.hpp"
#include "gpu/XsKernel.cuh"

#include "plot/TableConstructorGeneric.hpp"
#include "pos/ProofParams.hpp"

#include "ParityCommon.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory_resource>
#include <vector>

#define CHECK(call) do {                                                                     \
    cudaError_t err = (call);                                                                \
    if (err != cudaSuccess) {                                                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                   \
                     cudaGetErrorString(err));                                               \
        std::exit(2);                                                                        \
    }                                                                                        \
} while (0)

using pos2gpu::parity::derive_plot_id;

static double bench_cpu(uint8_t const* plot_id, int k)
{
    uint64_t total = 1ULL << k;
    ProofParams params(plot_id, static_cast<uint8_t>(k), uint8_t{2}, uint8_t{0});
    XsConstructor xs_ctor(params);

    std::vector<Xs_Candidate> out(total), tmp(total);
    std::vector<std::byte> scratch_storage(size_t{256} * 1024 * 1024);
    std::pmr::monotonic_buffer_resource mr(scratch_storage.data(), scratch_storage.size());

    auto t0 = std::chrono::steady_clock::now();
    auto sp = xs_ctor.construct(std::span<Xs_Candidate>(out),
                                std::span<Xs_Candidate>(tmp), mr);
    auto t1 = std::chrono::steady_clock::now();
    (void)sp;
    return std::chrono::duration<double>(t1 - t0).count();
}

static double bench_gpu(uint8_t const* plot_id, int k)
{
    uint64_t total = 1ULL << k;
    pos2gpu::XsCandidateGpu* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, sizeof(pos2gpu::XsCandidateGpu) * total));

    size_t temp_bytes = 0;
    pos2gpu::launch_construct_xs(plot_id, k, false, nullptr, nullptr, &temp_bytes, pos2gpu::sycl_backend::queue());
    void* d_temp = nullptr;
    CHECK(cudaMalloc(&d_temp, temp_bytes));

    // Warm up to amortise context init.
    pos2gpu::launch_construct_xs(plot_id, k, false, d_out, d_temp, &temp_bytes, pos2gpu::sycl_backend::queue());
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::steady_clock::now();
    pos2gpu::launch_construct_xs(plot_id, k, false, d_out, d_temp, &temp_bytes, pos2gpu::sycl_backend::queue());
    CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    CHECK(cudaFree(d_temp));
    CHECK(cudaFree(d_out));
    return std::chrono::duration<double>(t1 - t0).count();
}

int main()
{
    pos2gpu::initialize_aes_tables();

    auto plot_id = derive_plot_id(42);

    for (int k : {20, 22, 24, 26}) {
        uint64_t total = 1ULL << k;
        double cpu_s = bench_cpu(plot_id.data(), k);
        double gpu_s = bench_gpu(plot_id.data(), k);
        std::printf("k=%d  N=%-12llu  CPU=%7.3fs  GPU=%7.3fs  speedup=%5.1fx\n",
                    k, static_cast<unsigned long long>(total),
                    cpu_s, gpu_s, cpu_s / gpu_s);
    }
    return 0;
}
