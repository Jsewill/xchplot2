// t1_debug — verify a single (x_l, x_r) pair through both CPU and GPU paths.
// Used to find where T1 matching diverges.

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"

#include "pos/aes/AesHash.hpp"
#include "pos/ProofConstants.hpp"
#include "pos/ProofParams.hpp"
#include "pos/ProofCore.hpp"

#include "ParityCommon.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstdint>

// Suspected GPU-only pair from previous parity run, seed=1, k=18, strength=2:
//   x_l=77100, x_r=230247, match_info=0x00000

namespace {

using pos2gpu::parity::derive_plot_id;

__global__ void test_kernel(
    pos2gpu::AesHashKeys keys,
    uint32_t x_l, uint32_t x_r,
    uint32_t match_key_r,
    int k,
    uint32_t target_mask,
    uint32_t* out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out[0] = pos2gpu::g_x(keys, x_l, k, 16);                      // g(x_l)
    out[1] = pos2gpu::g_x(keys, x_r, k, 16);                      // g(x_r)
    out[2] = pos2gpu::matching_target(keys, 1, match_key_r,
                                      uint64_t(x_l), 0) & target_mask; // target for x_l
    out[3] = pos2gpu::matching_target(keys, 1, match_key_r,
                                      uint64_t(x_r), 0) & target_mask; // target for x_r (would be wrong direction but for sanity)
    auto p = pos2gpu::pairing(keys, uint64_t(x_l), uint64_t(x_r), 0);
    out[4] = p.r[0];
    out[5] = p.r[3] & 0x3;  // test_result for T1
}

void show_one(uint32_t seed, int k, uint32_t x_l, uint32_t x_r, uint32_t match_key_r)
{
    auto plot_id = derive_plot_id(seed);
    int section_bits = (k < 28) ? 2 : (k - 26);
    int match_key_bits = 2;
    int target_bits = k - section_bits - match_key_bits;
    uint32_t target_mask = (1u << target_bits) - 1u;

    // CPU
    AesHash cpu(plot_id.data(), k);
    uint32_t cpu_g_xl = cpu.g_x<true>(x_l, 16);
    uint32_t cpu_g_xr = cpu.g_x<true>(x_r, 16);
    uint32_t cpu_target_xl = cpu.matching_target<true>(1, match_key_r, uint64_t(x_l), 0) & target_mask;
    uint32_t cpu_target_xr = cpu.matching_target<true>(1, match_key_r, uint64_t(x_r), 0) & target_mask;
    auto cpu_pair = cpu.pairing<true>(uint64_t(x_l), uint64_t(x_r), 0);

    // GPU
    pos2gpu::initialize_aes_tables();
    pos2gpu::AesHashKeys keys = pos2gpu::make_keys(plot_id.data());
    uint32_t* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(uint32_t) * 8);
    test_kernel<<<1, 1>>>(keys, x_l, x_r, match_key_r, k, target_mask, d_out);
    cudaDeviceSynchronize();
    uint32_t h_out[8] = {};
    cudaMemcpy(h_out, d_out, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    std::printf("=== seed=%u k=%d x_l=%u x_r=%u match_key_r=%u ===\n", seed, k, x_l, x_r, match_key_r);
    std::printf("g(x_l):           CPU=0x%05x  GPU=0x%05x  %s\n",
                cpu_g_xl, h_out[0], cpu_g_xl == h_out[0] ? "OK" : "MISMATCH");
    std::printf("section_l (g>>%d):  CPU=%u   (g(x_r)>>%d)=%u\n",
                k - section_bits, cpu_g_xl >> (k - section_bits),
                k - section_bits, h_out[1] >> (k - section_bits));
    std::printf("g(x_r):           CPU=0x%05x  GPU=0x%05x  %s\n",
                cpu_g_xr, h_out[1], cpu_g_xr == h_out[1] ? "OK" : "MISMATCH");
    std::printf("target_for_xl:    CPU=0x%05x  GPU=0x%05x  %s\n",
                cpu_target_xl, h_out[2], cpu_target_xl == h_out[2] ? "OK" : "MISMATCH");
    std::printf("R target (g(x_r) & mask): CPU=0x%05x   does it match target_for_xl? %s\n",
                cpu_g_xr & target_mask,
                (cpu_g_xr & target_mask) == cpu_target_xl ? "YES" : "no");
    std::printf("pairing.r[0]:     CPU=0x%08x GPU=0x%08x  %s\n",
                cpu_pair.r[0], h_out[4], cpu_pair.r[0] == h_out[4] ? "OK" : "MISMATCH");
    std::printf("pairing.r[3] & 3 (test_result): CPU=%u  GPU=%u  %s\n",
                cpu_pair.r[3] & 0x3, h_out[5],
                (cpu_pair.r[3] & 0x3) == h_out[5] ? "OK" : "MISMATCH");
    std::printf("\n");
}

} // namespace

int main()
{
    // Sample GPU-only pair from earlier parity run.
    // x_l=77100 (g=0x20650, section=2, mk=0)
    // x_r=230247 (g=0x1b9df, section=1, mk=2)  → so this pair is in section_l=2's iteration,
    //   and to be emitted, match_key_r must = 2 (R's match_key), and target_l for mk=2
    //   must equal g(x_r) & 14_bit_mask = 0x039df.
    for (uint32_t mk = 0; mk < 4; ++mk) {
        show_one(/*seed=*/1u, /*k=*/18, /*x_l=*/77100, /*x_r=*/230247, mk);
    }
    return 0;
}
