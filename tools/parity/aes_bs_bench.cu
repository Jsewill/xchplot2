// aes_bs_bench — standalone throughput comparison between the bit-sliced
// warp-parallel run_rounds_bs32_warp and the current T-table run_rounds on
// the GPU. This is the go/no-go gate for the match_all_buckets integration:
// if the BS path isn't at least ~1.5× the T-table per-state throughput, the
// rewrite doesn't pay for itself and we pivot to DP4A instead.

#include "gpu/AesGpu.cuh"
#include "gpu/AesGpuBitsliced.cuh"
#include "gpu/AesHashGpu.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

using pos2gpu::AesState;

#define CHECK(expr) do {                                                 \
    cudaError_t err_ = (expr);                                           \
    if (err_ != cudaSuccess) {                                           \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,      \
                     #expr, cudaGetErrorString(err_));                   \
        std::exit(1);                                                    \
    }                                                                    \
} while (0)

namespace {

__global__ void bench_bs32_run_rounds(AesState const* __restrict__ in,
                                       AesState const* __restrict__ key_pairs,
                                       AesState* __restrict__ out,
                                       uint64_t n_batches, int rounds)
{
    uint64_t batch = uint64_t(blockIdx.x) * (blockDim.x / 32)
                   + (threadIdx.x / 32);
    uint32_t lane  = threadIdx.x & 31;
    if (batch >= n_batches) return;
    AesState my = in[batch * 32 + lane];
    AesState k1 = key_pairs[batch * 2 + 0];
    AesState k2 = key_pairs[batch * 2 + 1];
    uint32_t bs[4], k1_bs[4], k2_bs[4];
    pos2gpu::bs32_pack(my, bs);
    pos2gpu::make_bs32_round_key(k1, k1_bs);
    pos2gpu::make_bs32_round_key(k2, k2_bs);
    pos2gpu::run_rounds_bs32_warp(bs, k1_bs, k2_bs, rounds);
    AesState result;
    pos2gpu::bs32_unpack(bs, result);
    out[batch * 32 + lane] = result;
}

// Shared-memory T-table baseline: this is what match_all_buckets actually
// uses today. Each block loads the 4 KB sT once, then every thread runs
// run_rounds_smem on its own state.
__global__ void bench_tt_run_rounds_smem(AesState const* __restrict__ in,
                                          AesState const* __restrict__ key_pairs,
                                          AesState* __restrict__ out,
                                          uint64_t n, int rounds)
{
    __shared__ uint32_t sT[4 * 256];
    pos2gpu::load_aes_tables_smem(sT);
    __syncthreads();

    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    pos2gpu::AesHashKeys keys;
    keys.round_key_1 = key_pairs[(tid / 32) * 2 + 0];
    keys.round_key_2 = key_pairs[(tid / 32) * 2 + 1];
    out[tid] = pos2gpu::run_rounds_smem(in[tid], keys, rounds, sT);
}

} // namespace

int main(int argc, char** argv)
{
    pos2gpu::initialize_aes_tables();
    CHECK(cudaDeviceSynchronize());

    // 4 M states by default; override via argv[1].
    uint64_t n_states = 4ull << 20;
    if (argc > 1) n_states = std::strtoull(argv[1], nullptr, 10);
    n_states = (n_states + 31) & ~uint64_t(31);  // round up to multiple of 32
    uint64_t n_batches = n_states / 32;
    int rounds = pos2gpu::kAesPairingRounds;

    std::printf("aes_bs_bench  n_states=%llu  rounds/hash=%d (2x = %d AES rounds)\n",
                static_cast<unsigned long long>(n_states), rounds, rounds * 2);

    std::mt19937_64 rng(0x50C0A53C0C0A);
    std::vector<AesState> h_in(n_states), h_keys(n_batches * 2);
    for (auto& s : h_in)   for (int j = 0; j < 4; ++j) s.w[j] = static_cast<uint32_t>(rng());
    for (auto& k : h_keys) for (int j = 0; j < 4; ++j) k.w[j] = static_cast<uint32_t>(rng());

    AesState *d_in = nullptr, *d_keys = nullptr, *d_bs_out = nullptr, *d_tt_out = nullptr;
    CHECK(cudaMalloc(&d_in,     n_states   * sizeof(AesState)));
    CHECK(cudaMalloc(&d_keys,   n_batches  * 2 * sizeof(AesState)));
    CHECK(cudaMalloc(&d_bs_out, n_states   * sizeof(AesState)));
    CHECK(cudaMalloc(&d_tt_out, n_states   * sizeof(AesState)));
    CHECK(cudaMemcpy(d_in,   h_in.data(),   n_states   * sizeof(AesState), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_keys, h_keys.data(), n_batches * 2 * sizeof(AesState), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    // ---- T-table path: 1 thread / state, 256 threads / block ----
    constexpr int kTTThreads = 256;
    uint64_t tt_blocks = (n_states + kTTThreads - 1) / kTTThreads;

    // Warmup
    bench_tt_run_rounds_smem<<<tt_blocks, kTTThreads>>>(d_in, d_keys, d_tt_out, n_states, rounds);
    CHECK(cudaDeviceSynchronize());

    // Timed (3 runs, take min)
    float tt_ms_min = 1e30f;
    for (int run = 0; run < 3; ++run) {
        CHECK(cudaEventRecord(e0));
        bench_tt_run_rounds_smem<<<tt_blocks, kTTThreads>>>(d_in, d_keys, d_tt_out, n_states, rounds);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < tt_ms_min) tt_ms_min = ms;
    }

    // ---- BS-32 path: 1 warp / 32 states, 4 warps / block ----
    constexpr int kBSThreads = 128;
    uint64_t bs_blocks = (n_batches + (kBSThreads / 32) - 1) / (kBSThreads / 32);

    bench_bs32_run_rounds<<<bs_blocks, kBSThreads>>>(d_in, d_keys, d_bs_out, n_batches, rounds);
    CHECK(cudaDeviceSynchronize());

    float bs_ms_min = 1e30f;
    for (int run = 0; run < 3; ++run) {
        CHECK(cudaEventRecord(e0));
        bench_bs32_run_rounds<<<bs_blocks, kBSThreads>>>(d_in, d_keys, d_bs_out, n_batches, rounds);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < bs_ms_min) bs_ms_min = ms;
    }

    // ---- Parity sanity: compare a handful of outputs ----
    std::vector<AesState> h_bs(n_states), h_tt(n_states);
    CHECK(cudaMemcpy(h_bs.data(), d_bs_out, n_states * sizeof(AesState), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_tt.data(), d_tt_out, n_states * sizeof(AesState), cudaMemcpyDeviceToHost));
    bool all_ok = true;
    for (uint64_t i = 0; i < 1024 && i < n_states; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (h_bs[i].w[j] != h_tt[i].w[j]) { all_ok = false; break; }
        }
        if (!all_ok) break;
    }

    double bs_rate = double(n_states) / (bs_ms_min * 1e-3);
    double tt_rate = double(n_states) / (tt_ms_min * 1e-3);
    double speedup = bs_rate / tt_rate;

    std::printf("  T-table run_rounds    : %.3f ms   %.2f Mhash/s\n",
                tt_ms_min, tt_rate / 1e6);
    std::printf("  BS-32   run_rounds    : %.3f ms   %.2f Mhash/s\n",
                bs_ms_min, bs_rate / 1e6);
    std::printf("  speedup (BS / T-table): %.3fx   parity-sanity=%s\n",
                speedup, all_ok ? "ok" : "FAIL");

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_bs_out));
    CHECK(cudaFree(d_tt_out));
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));

    return all_ok ? 0 : 1;
}
