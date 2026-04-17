// xs_parity — proves the GPU Xs construction (g(x) for all x in [0, 2^k),
// then stable radix sort by match_info) is byte-identical to pos2-chip's
// CPU XsConstructor::construct.
//
// Compares Xs_Candidate[2^k] arrays byte-for-byte. Pass criterion: every
// (match_info, x) pair matches in order.

#include "gpu/AesGpu.cuh"
#include "gpu/XsKernel.cuh"

// pos2-chip headers for the CPU reference.
#include "plot/TableConstructorGeneric.hpp"
#include "pos/ProofParams.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory_resource>
#include <vector>

namespace {

#define CHECK(call) do {                                                                     \
    cudaError_t err = (call);                                                                \
    if (err != cudaSuccess) {                                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                \
                     cudaGetErrorString(err));                                               \
        std::exit(2);                                                                        \
    }                                                                                        \
} while (0)

std::array<uint8_t, 32> derive_plot_id(uint32_t seed)
{
    std::array<uint8_t, 32> id{};
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ uint64_t(seed) * 0x100000001B3ULL;
    for (size_t i = 0; i < id.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        id[i] = static_cast<uint8_t>(s >> 56);
    }
    return id;
}

bool run_for(uint32_t seed, int k, bool testnet)
{
    auto plot_id = derive_plot_id(seed);
    uint64_t const total = 1ULL << k;

    std::printf("[seed=%u  k=%d  testnet=%d  N=%llu]\n",
                seed, k, int(testnet), static_cast<unsigned long long>(total));

    // ---- CPU reference ----
    ProofParams params(plot_id.data(),
                       static_cast<uint8_t>(k),
                       /*strength=*/uint8_t{2},
                       testnet ? uint8_t{1} : uint8_t{0});
    XsConstructor xs_ctor(params);

    std::vector<Xs_Candidate> cpu_out_buf(total);
    std::vector<Xs_Candidate> cpu_tmp_buf(total);

    // pos2-chip's XsConstructor::construct needs a pmr scratch. Use the
    // simplest std::pmr::monotonic_buffer_resource backed by a flat buffer
    // — generous size so internal RadixSort's per-thread counts fit.
    std::vector<std::byte> scratch_storage(64 * 1024 * 1024);
    std::pmr::monotonic_buffer_resource scratch_mr(
        scratch_storage.data(), scratch_storage.size());

    auto cpu_span = xs_ctor.construct(
        std::span<Xs_Candidate>(cpu_out_buf),
        std::span<Xs_Candidate>(cpu_tmp_buf),
        scratch_mr);

    if (cpu_span.size() != total) {
        std::printf("  CPU returned unexpected size: %zu (want %llu)\n",
                    cpu_span.size(), static_cast<unsigned long long>(total));
        return false;
    }

    // ---- GPU ----
    pos2gpu::XsCandidateGpu* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, sizeof(pos2gpu::XsCandidateGpu) * total));

    size_t temp_bytes = 0;
    auto err = pos2gpu::launch_construct_xs(
        plot_id.data(), k, testnet,
        /*d_out=*/nullptr,
        /*d_temp_storage=*/nullptr,
        &temp_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "  query temp_bytes failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    void* d_temp = nullptr;
    CHECK(cudaMalloc(&d_temp, temp_bytes));

    err = pos2gpu::launch_construct_xs(
        plot_id.data(), k, testnet, d_out, d_temp, &temp_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "  launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_temp); cudaFree(d_out);
        return false;
    }
    CHECK(cudaDeviceSynchronize());

    std::vector<pos2gpu::XsCandidateGpu> gpu_out(total);
    CHECK(cudaMemcpy(gpu_out.data(), d_out,
                     sizeof(pos2gpu::XsCandidateGpu) * total,
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_temp));
    CHECK(cudaFree(d_out));

    // ---- compare ----
    static_assert(sizeof(Xs_Candidate) == sizeof(pos2gpu::XsCandidateGpu),
                  "Xs_Candidate layout drift");

    auto* gpu_as_cpu = reinterpret_cast<Xs_Candidate const*>(gpu_out.data());

    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < total; ++i) {
        if (cpu_span[i].match_info != gpu_as_cpu[i].match_info ||
            cpu_span[i].x          != gpu_as_cpu[i].x)
        {
            if (mismatches < 5) {
                std::printf("  MISMATCH at i=%llu  cpu=(mi=0x%08x x=%u)  gpu=(mi=0x%08x x=%u)\n",
                            static_cast<unsigned long long>(i),
                            cpu_span[i].match_info, cpu_span[i].x,
                            gpu_as_cpu[i].match_info, gpu_as_cpu[i].x);
            }
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::printf("  OK  %llu / %llu match (Xs_Candidate full struct)\n",
                    static_cast<unsigned long long>(total),
                    static_cast<unsigned long long>(total));
        return true;
    } else {
        std::printf("  FAIL  %llu mismatches / %llu\n",
                    static_cast<unsigned long long>(mismatches),
                    static_cast<unsigned long long>(total));
        return false;
    }
}

} // namespace

int main()
{
    pos2gpu::initialize_aes_tables();

    bool all_ok = true;
    // Sweep a few seeds at k=18 (small, fast). Add k=20 for one to confirm
    // larger sizes work end-to-end.
    for (uint32_t seed : {1u, 2u, 0xCAFE'BABEu}) {
        all_ok = run_for(seed, /*k=*/18, /*testnet=*/false) && all_ok;
    }
    all_ok = run_for(/*seed=*/7u, /*k=*/18, /*testnet=*/true) && all_ok;
    all_ok = run_for(/*seed=*/9u, /*k=*/20, /*testnet=*/false) && all_ok;

    std::printf("\n==> %s\n", all_ok ? "ALL OK" : "FAIL");
    return all_ok ? 0 : 1;
}
