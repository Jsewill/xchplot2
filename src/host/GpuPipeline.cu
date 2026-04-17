// GpuPipeline.cu — orchestrates Xs → T1 → T2 → T3 on the device, with
// CUB radix sort between phases (each phase consumes sorted-by-match_info
// input). Final T3 output is sorted by proof_fragment (low 2k bits) to
// match pos2-chip Table3Constructor::post_construct_span.

#include "host/GpuPipeline.hpp"

#include "gpu/AesGpu.cuh"
#include "gpu/XsKernel.cuh"
#include "gpu/T1Kernel.cuh"
#include "gpu/T2Kernel.cuh"
#include "gpu/T3Kernel.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pos2gpu {

namespace {

#define CHECK(call) do {                                                 \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        throw std::runtime_error(std::string("CUDA: ") +                 \
                                 cudaGetErrorString(err));               \
    }                                                                    \
} while (0)

template <typename T>
T* dmalloc(size_t count) {
    T* p = nullptr;
    CHECK(cudaMalloc(&p, sizeof(T) * count));
    return p;
}

template <typename T>
void dfree(T* p) { if (p) cudaFree(p); }

// max_pairs_per_section_possible from pos2-chip TableConstructorGeneric.hpp:23
// Returns the per-section worst-case pair count; multiply by num_sections
// for the full per-phase capacity.
size_t max_pairs_per_section(int k, int num_section_bits)
{
    int extra_margin_bits = 8 - ((28 - k) / 2);
    return (1ULL << (k - num_section_bits)) + (1ULL << (k - extra_margin_bits));
}

// =====================================================================
// T1 sort: by match_info, low k bits, stable. Uses CUB SortPairs with
// (key=match_info, value=index) then permutes T1Pairings.
// =====================================================================

__global__ void permute_t1(
    T1PairingGpu const* __restrict__ src,
    uint32_t const* __restrict__ indices,
    T1PairingGpu* __restrict__ dst,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = src[indices[idx]];
}

__global__ void extract_t1_keys(
    T1PairingGpu const* __restrict__ src,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ vals_out,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    keys_out[idx] = src[idx].match_info;
    vals_out[idx] = uint32_t(idx);
}

// =====================================================================
// T2 sort: same shape — sort indices by match_info.
// =====================================================================

__global__ void permute_t2(
    T2PairingGpu const* __restrict__ src,
    uint32_t const* __restrict__ indices,
    T2PairingGpu* __restrict__ dst,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = src[indices[idx]];
}

__global__ void extract_t2_keys(
    T2PairingGpu const* __restrict__ src,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ vals_out,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (idx >= count) return;
    keys_out[idx] = src[idx].match_info;
    vals_out[idx] = uint32_t(idx);
}

// =====================================================================
// T3 sort: T3PairingGpu is just uint64_t proof_fragment — SortKeys directly.
// =====================================================================

} // namespace

GpuPipelineResult run_gpu_pipeline(GpuPipelineConfig const& cfg)
{
    if (cfg.k < 18 || cfg.k > 32 || (cfg.k & 1) != 0) {
        throw std::runtime_error("k must be even in [18, 32]");
    }
    if (cfg.strength < 2) {
        throw std::runtime_error("strength must be >= 2");
    }

    int const num_section_bits = (cfg.k < 28) ? 2 : (cfg.k - 26);
    uint64_t const total_xs    = 1ULL << cfg.k;
    size_t const cap           = max_pairs_per_section(cfg.k, num_section_bits)
                               * (1ULL << num_section_bits);

    constexpr int kThreads = 256;
    auto blocks = [&](uint64_t n) {
        return unsigned((n + kThreads - 1) / kThreads);
    };

    cudaStream_t stream = nullptr; // default stream

    // ---------- Phase Xs ----------
    XsCandidateGpu* d_xs = dmalloc<XsCandidateGpu>(total_xs);
    void* d_xs_temp = nullptr;
    size_t xs_temp_bytes = 0;
    CHECK(launch_construct_xs(cfg.plot_id.data(), cfg.k, cfg.testnet,
                              nullptr, nullptr, &xs_temp_bytes));
    d_xs_temp = nullptr;
    CHECK(cudaMalloc(&d_xs_temp, xs_temp_bytes));
    CHECK(launch_construct_xs(cfg.plot_id.data(), cfg.k, cfg.testnet,
                              d_xs, d_xs_temp, &xs_temp_bytes, stream));

    // ---------- Phase T1 ----------
    auto t1p = make_t1_params(cfg.k, cfg.strength);
    T1PairingGpu* d_t1 = dmalloc<T1PairingGpu>(cap);
    uint64_t* d_t1_count = dmalloc<uint64_t>(1);
    void* d_t1_temp = nullptr;
    size_t t1_temp_bytes = 0;
    CHECK(launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1, d_t1_count, cap,
                          nullptr, &t1_temp_bytes));
    CHECK(cudaMalloc(&d_t1_temp, t1_temp_bytes));
    CHECK(launch_t1_match(cfg.plot_id.data(), t1p, d_xs, total_xs,
                          d_t1, d_t1_count, cap,
                          d_t1_temp, &t1_temp_bytes, stream));
    CHECK(cudaStreamSynchronize(stream));

    uint64_t t1_count = 0;
    CHECK(cudaMemcpy(&t1_count, d_t1_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t1_count > cap) {
        throw std::runtime_error("T1 overflow");
    }
    // Free Xs and its temp now — never needed again
    dfree(d_xs); dfree(static_cast<uint8_t*>(d_xs_temp));
    dfree(d_t1_temp);

    // Sort T1 by match_info (low k bits)
    T1PairingGpu* d_t1_sorted = dmalloc<T1PairingGpu>(t1_count);
    {
        uint32_t* d_keys_in   = dmalloc<uint32_t>(t1_count);
        uint32_t* d_keys_out  = dmalloc<uint32_t>(t1_count);
        uint32_t* d_vals_in   = dmalloc<uint32_t>(t1_count);
        uint32_t* d_vals_out  = dmalloc<uint32_t>(t1_count);
        extract_t1_keys<<<blocks(t1_count), kThreads, 0, stream>>>(
            d_t1, d_keys_in, d_vals_in, t1_count);
        CHECK(cudaGetLastError());

        size_t sort_bytes = 0;
        CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, sort_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            t1_count, /*begin_bit=*/0, /*end_bit=*/cfg.k, stream));
        void* d_sort_scratch = nullptr;
        CHECK(cudaMalloc(&d_sort_scratch, sort_bytes));
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, sort_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            t1_count, /*begin_bit=*/0, /*end_bit=*/cfg.k, stream));

        permute_t1<<<blocks(t1_count), kThreads, 0, stream>>>(
            d_t1, d_vals_out, d_t1_sorted, t1_count);
        CHECK(cudaGetLastError());
        CHECK(cudaStreamSynchronize(stream));

        cudaFree(d_sort_scratch);
        dfree(d_keys_in); dfree(d_keys_out); dfree(d_vals_in); dfree(d_vals_out);
    }
    dfree(d_t1); dfree(d_t1_count);

    // ---------- Phase T2 ----------
    auto t2p = make_t2_params(cfg.k, cfg.strength);
    T2PairingGpu* d_t2 = dmalloc<T2PairingGpu>(cap);
    uint64_t* d_t2_count = dmalloc<uint64_t>(1);
    void* d_t2_temp = nullptr;
    size_t t2_temp_bytes = 0;
    CHECK(launch_t2_match(cfg.plot_id.data(), t2p, d_t1_sorted, t1_count,
                          d_t2, d_t2_count, cap,
                          nullptr, &t2_temp_bytes));
    CHECK(cudaMalloc(&d_t2_temp, t2_temp_bytes));
    CHECK(launch_t2_match(cfg.plot_id.data(), t2p, d_t1_sorted, t1_count,
                          d_t2, d_t2_count, cap,
                          d_t2_temp, &t2_temp_bytes, stream));
    CHECK(cudaStreamSynchronize(stream));

    uint64_t t2_count = 0;
    CHECK(cudaMemcpy(&t2_count, d_t2_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t2_count > cap) {
        throw std::runtime_error("T2 overflow");
    }
    dfree(d_t1_sorted); dfree(d_t2_temp);

    // Sort T2 by match_info (low k bits)
    T2PairingGpu* d_t2_sorted = dmalloc<T2PairingGpu>(t2_count);
    {
        uint32_t* d_keys_in  = dmalloc<uint32_t>(t2_count);
        uint32_t* d_keys_out = dmalloc<uint32_t>(t2_count);
        uint32_t* d_vals_in  = dmalloc<uint32_t>(t2_count);
        uint32_t* d_vals_out = dmalloc<uint32_t>(t2_count);
        extract_t2_keys<<<blocks(t2_count), kThreads, 0, stream>>>(
            d_t2, d_keys_in, d_vals_in, t2_count);
        CHECK(cudaGetLastError());

        size_t sort_bytes = 0;
        CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, sort_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            t2_count, 0, cfg.k, stream));
        void* d_sort_scratch = nullptr;
        CHECK(cudaMalloc(&d_sort_scratch, sort_bytes));
        CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_scratch, sort_bytes,
            d_keys_in, d_keys_out, d_vals_in, d_vals_out,
            t2_count, 0, cfg.k, stream));

        permute_t2<<<blocks(t2_count), kThreads, 0, stream>>>(
            d_t2, d_vals_out, d_t2_sorted, t2_count);
        CHECK(cudaGetLastError());
        CHECK(cudaStreamSynchronize(stream));

        cudaFree(d_sort_scratch);
        dfree(d_keys_in); dfree(d_keys_out); dfree(d_vals_in); dfree(d_vals_out);
    }
    dfree(d_t2); dfree(d_t2_count);

    // ---------- Phase T3 ----------
    auto t3p = make_t3_params(cfg.k, cfg.strength);
    T3PairingGpu* d_t3 = dmalloc<T3PairingGpu>(cap);
    uint64_t* d_t3_count = dmalloc<uint64_t>(1);
    void* d_t3_temp = nullptr;
    size_t t3_temp_bytes = 0;
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p, d_t2_sorted, t2_count,
                          d_t3, d_t3_count, cap,
                          nullptr, &t3_temp_bytes));
    CHECK(cudaMalloc(&d_t3_temp, t3_temp_bytes));
    CHECK(launch_t3_match(cfg.plot_id.data(), t3p, d_t2_sorted, t2_count,
                          d_t3, d_t3_count, cap,
                          d_t3_temp, &t3_temp_bytes, stream));
    CHECK(cudaStreamSynchronize(stream));

    uint64_t t3_count = 0;
    CHECK(cudaMemcpy(&t3_count, d_t3_count, sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    if (t3_count > cap) {
        throw std::runtime_error("T3 overflow");
    }
    dfree(d_t2_sorted); dfree(d_t3_temp);

    // Sort T3 by proof_fragment (low 2k bits) — matches CPU's
    // post_construct_span sort key.
    uint64_t* d_frags_in  = nullptr;
    uint64_t* d_frags_out = nullptr;
    {
        // T3PairingGpu IS just { uint64_t proof_fragment }, so reinterpret
        // d_t3 as a uint64_t array directly.
        d_frags_in  = reinterpret_cast<uint64_t*>(d_t3);
        d_frags_out = dmalloc<uint64_t>(t3_count);

        size_t sort_bytes = 0;
        CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr, sort_bytes,
            d_frags_in, d_frags_out,
            t3_count, /*begin_bit=*/0, /*end_bit=*/2 * cfg.k, stream));
        void* d_sort_scratch = nullptr;
        CHECK(cudaMalloc(&d_sort_scratch, sort_bytes));
        CHECK(cub::DeviceRadixSort::SortKeys(
            d_sort_scratch, sort_bytes,
            d_frags_in, d_frags_out,
            t3_count, 0, 2 * cfg.k, stream));
        CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_sort_scratch);
    }

    // Copy sorted T3 fragments to host
    GpuPipelineResult result;
    result.t1_count = t1_count;
    result.t2_count = t2_count;
    result.t3_count = t3_count;
    result.t3_fragments.resize(t3_count);
    if (t3_count > 0) {
        CHECK(cudaMemcpy(result.t3_fragments.data(), d_frags_out,
                         sizeof(uint64_t) * t3_count, cudaMemcpyDeviceToHost));
    }
    dfree(d_t3); dfree(d_t3_count); dfree(d_frags_out);

    return result;
}

} // namespace pos2gpu
