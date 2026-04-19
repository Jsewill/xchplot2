// aes_bs_parity — proves aesenc_round_bs (no T-tables) is byte-identical
// to aesenc_round (T-table) for random states and keys.
//
// Day 1 of the BS-AES work: establishes the parity harness. aesenc_round_bs
// currently uses an explicit S-box lookup + explicit MixColumns; later passes
// replace the lookup with the Boyar-Peralta 113-gate bit-logic network and
// add warp-parallel 32-way batching. Every later change must keep this test
// passing.

#include "gpu/AesGpu.cuh"
#include "gpu/AesGpuBitsliced.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/AesSBoxBP.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

using pos2gpu::AesState;
using pos2gpu::aesenc_round;
using pos2gpu::aesenc_round_bs;
using pos2gpu::initialize_aes_tables;
using pos2gpu::initialize_aes_bs_tables;

#define CHECK(expr) do {                                                 \
    cudaError_t err_ = (expr);                                           \
    if (err_ != cudaSuccess) {                                           \
        std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,      \
                     #expr, cudaGetErrorString(err_));                   \
        std::exit(1);                                                    \
    }                                                                    \
} while (0)

namespace {

__global__ void tt_round_kernel(AesState const* in, AesState const* keys,
                                 AesState* out, uint64_t n)
{
    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = aesenc_round(in[tid], keys[tid]);
}

__global__ void bs_round_kernel(AesState const* in, AesState const* keys,
                                 AesState* out, uint64_t n)
{
    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[tid] = aesenc_round_bs(in[tid], keys[tid]);
}

// Transpose identity: pack 32 states into bit-sliced form, unpack, verify
// the lane-local state is bitwise identical. Launch config: one warp per
// 32-state batch.
__global__ void bs32_roundtrip_kernel(AesState const* in, AesState* out,
                                       uint64_t n_batches)
{
    uint64_t batch    = uint64_t(blockIdx.x) * (blockDim.x / 32)
                      + (threadIdx.x / 32);
    uint32_t lane     = threadIdx.x & 31;
    if (batch >= n_batches) return;
    AesState my = in[batch * 32 + lane];
    uint32_t planes[4];
    pos2gpu::bs32_pack(my, planes);
    AesState rt;
    pos2gpu::bs32_unpack(planes, rt);
    out[batch * 32 + lane] = rt;
}

// Apply sub_bytes_bs32 to 32 states per warp and write back.
__global__ void bs32_sub_bytes_kernel(AesState const* in, AesState* out,
                                       uint64_t n_batches)
{
    uint64_t batch = uint64_t(blockIdx.x) * (blockDim.x / 32)
                   + (threadIdx.x / 32);
    uint32_t lane  = threadIdx.x & 31;
    if (batch >= n_batches) return;
    AesState my = in[batch * 32 + lane];
    uint32_t bs[4];
    pos2gpu::bs32_pack(my, bs);
    pos2gpu::sub_bytes_bs32(bs);
    AesState result;
    pos2gpu::bs32_unpack(bs, result);
    out[batch * 32 + lane] = result;
}

__global__ void bs32_shift_rows_kernel(AesState const* in, AesState* out,
                                        uint64_t n_batches)
{
    uint64_t batch = uint64_t(blockIdx.x) * (blockDim.x / 32)
                   + (threadIdx.x / 32);
    uint32_t lane  = threadIdx.x & 31;
    if (batch >= n_batches) return;
    AesState my = in[batch * 32 + lane];
    uint32_t bs[4];
    pos2gpu::bs32_pack(my, bs);
    pos2gpu::shift_rows_bs32(bs);
    AesState result;
    pos2gpu::bs32_unpack(bs, result);
    out[batch * 32 + lane] = result;
}

__global__ void bs32_ark_kernel(AesState const* in, AesState const* keys,
                                 AesState* out, uint64_t n_batches)
{
    uint64_t batch = uint64_t(blockIdx.x) * (blockDim.x / 32)
                   + (threadIdx.x / 32);
    uint32_t lane  = threadIdx.x & 31;
    if (batch >= n_batches) return;
    AesState my  = in[batch * 32 + lane];
    AesState key = keys[batch]; // one key per batch; broadcast to 32 states
    uint32_t bs[4], kbs[4];
    pos2gpu::bs32_pack(my, bs);
    pos2gpu::make_bs32_round_key(key, kbs);
    pos2gpu::add_round_key_bs32(bs, kbs);
    AesState result;
    pos2gpu::bs32_unpack(bs, result);
    out[batch * 32 + lane] = result;
}

__global__ void bs32_mix_columns_kernel(AesState const* in, AesState* out,
                                         uint64_t n_batches)
{
    uint64_t batch = uint64_t(blockIdx.x) * (blockDim.x / 32)
                   + (threadIdx.x / 32);
    uint32_t lane  = threadIdx.x & 31;
    if (batch >= n_batches) return;
    AesState my = in[batch * 32 + lane];
    uint32_t bs[4];
    pos2gpu::bs32_pack(my, bs);
    pos2gpu::mix_columns_bs32(bs);
    AesState result;
    pos2gpu::bs32_unpack(bs, result);
    out[batch * 32 + lane] = result;
}

// Full warp-parallel round: pack, aesenc_round_bs32_warp with a broadcast key,
// unpack. Compared against 32 scalar aesenc_round invocations per batch.
__global__ void bs32_round_kernel(AesState const* in, AesState const* keys,
                                   AesState* out, uint64_t n_batches)
{
    uint64_t batch = uint64_t(blockIdx.x) * (blockDim.x / 32)
                   + (threadIdx.x / 32);
    uint32_t lane  = threadIdx.x & 31;
    if (batch >= n_batches) return;
    AesState my  = in[batch * 32 + lane];
    AesState key = keys[batch];
    uint32_t bs[4], kbs[4];
    pos2gpu::bs32_pack(my, bs);
    pos2gpu::make_bs32_round_key(key, kbs);
    pos2gpu::aesenc_round_bs32_warp(bs, kbs);
    AesState result;
    pos2gpu::bs32_unpack(bs, result);
    out[batch * 32 + lane] = result;
}

// N rounds warp-parallel with alternating keys.
__global__ void bs32_run_rounds_kernel(AesState const* in,
                                        AesState const* key_pairs,
                                        AesState* out,
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

// Scalar reference: one thread per state, T-table run_rounds.
__global__ void tt_run_rounds_kernel(AesState const* in,
                                      AesState const* key_pairs,
                                      AesState* out,
                                      uint64_t n, int rounds)
{
    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    pos2gpu::AesHashKeys keys;
    keys.round_key_1 = key_pairs[(tid / 32) * 2 + 0];
    keys.round_key_2 = key_pairs[(tid / 32) * 2 + 1];
    out[tid] = pos2gpu::run_rounds(in[tid], keys, rounds);
}

// For each x in [0,256) compare bp_sbox(x) to the Rijndael S-box populated
// in constant memory. One thread per byte; result is bp[x] and ref[x] so the
// host can diagnose any mismatches.
__global__ void bp_sbox_exhaustive_kernel(uint8_t* bp_out, uint8_t* ref_out)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= 256) return;
    bp_out[x]  = pos2gpu::bp_sbox(uint8_t(x));
    ref_out[x] = pos2gpu::kAesSBox[x];
}

// Cross-check the bit-sliced instantiation: evaluate bp_sbox_circuit<uint32_t>
// on 8 × 32 = 256 inputs packed as bit-planes (covering every byte value),
// unpack, compare to kAesSBox. Validates the template, the NOT-mask, and the
// bit-plane conventions. Single-thread kernel — correctness not throughput.
__global__ void bp_sbox_bitsliced_all_kernel(uint8_t* bp_out, uint8_t* ref_out)
{
    for (int group = 0; group < 8; ++group) {
        // Inputs x = group*32 .. group*32 + 31.
        uint32_t u[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t plane = 0;
            for (int j = 0; j < 32; ++j) {
                uint32_t x = uint32_t(group * 32 + j);
                plane |= ((x >> (7 - k)) & 1u) << j;
            }
            u[k] = plane;
        }
        uint32_t s[8];
        pos2gpu::bp_sbox_circuit<uint32_t>(
            u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            0xFFFFFFFFu);
        for (int j = 0; j < 32; ++j) {
            uint8_t y = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                y |= uint8_t(((s[k] >> j) & 1u) << (7 - k));
            }
            int idx = group * 32 + j;
            bp_out[idx]  = y;
            ref_out[idx] = pos2gpu::kAesSBox[idx];
        }
    }
}

} // namespace

int main()
{
    initialize_aes_tables();
    initialize_aes_bs_tables();
    CHECK(cudaDeviceSynchronize());

    constexpr uint64_t kN = 1ull << 20; // 1 M independent round tests
    std::mt19937_64 rng(0xB1756C1CED0123ULL);

    std::vector<AesState> h_in(kN), h_keys(kN);
    for (uint64_t i = 0; i < kN; ++i) {
        for (int j = 0; j < 4; ++j) {
            h_in[i].w[j]   = static_cast<uint32_t>(rng());
            h_keys[i].w[j] = static_cast<uint32_t>(rng());
        }
    }

    AesState *d_in = nullptr, *d_keys = nullptr, *d_tt = nullptr, *d_bs = nullptr;
    CHECK(cudaMalloc(&d_in,   kN * sizeof(AesState)));
    CHECK(cudaMalloc(&d_keys, kN * sizeof(AesState)));
    CHECK(cudaMalloc(&d_tt,   kN * sizeof(AesState)));
    CHECK(cudaMalloc(&d_bs,   kN * sizeof(AesState)));
    CHECK(cudaMemcpy(d_in,   h_in.data(),   kN * sizeof(AesState), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_keys, h_keys.data(), kN * sizeof(AesState), cudaMemcpyHostToDevice));

    constexpr int kThreads = 256;
    uint64_t blocks = (kN + kThreads - 1) / kThreads;

    tt_round_kernel<<<blocks, kThreads>>>(d_in, d_keys, d_tt, kN);
    bs_round_kernel<<<blocks, kThreads>>>(d_in, d_keys, d_bs, kN);
    CHECK(cudaDeviceSynchronize());

    std::vector<AesState> h_tt(kN), h_bs(kN);
    CHECK(cudaMemcpy(h_tt.data(), d_tt, kN * sizeof(AesState), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_bs.data(), d_bs, kN * sizeof(AesState), cudaMemcpyDeviceToHost));

    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < kN; ++i) {
        if (std::memcmp(&h_tt[i], &h_bs[i], sizeof(AesState)) != 0) {
            if (mismatches < 4) {
                std::printf("  mismatch at i=%llu:\n",
                            static_cast<unsigned long long>(i));
                std::printf("    in  : %08x %08x %08x %08x\n",
                            h_in[i].w[0], h_in[i].w[1], h_in[i].w[2], h_in[i].w[3]);
                std::printf("    key : %08x %08x %08x %08x\n",
                            h_keys[i].w[0], h_keys[i].w[1], h_keys[i].w[2], h_keys[i].w[3]);
                std::printf("    tt  : %08x %08x %08x %08x\n",
                            h_tt[i].w[0], h_tt[i].w[1], h_tt[i].w[2], h_tt[i].w[3]);
                std::printf("    bs  : %08x %08x %08x %08x\n",
                            h_bs[i].w[0], h_bs[i].w[1], h_bs[i].w[2], h_bs[i].w[3]);
            }
            ++mismatches;
        }
    }

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_keys));
    CHECK(cudaFree(d_tt));
    CHECK(cudaFree(d_bs));

    std::printf("  aesenc_round_bs vs aesenc_round: %llu / %llu  %s\n",
                static_cast<unsigned long long>(kN - mismatches),
                static_cast<unsigned long long>(kN),
                mismatches == 0 ? "OK" : "FAIL");

    // ------ bs32 pack/unpack round-trip ------
    constexpr uint64_t kBatches = 8192; // 8192 * 32 = 256K states
    constexpr uint64_t kStates  = kBatches * 32;
    std::vector<AesState> h_tin(kStates);
    for (auto& s : h_tin) {
        for (int j = 0; j < 4; ++j) s.w[j] = static_cast<uint32_t>(rng());
    }
    AesState *d_tin = nullptr, *d_tout = nullptr;
    CHECK(cudaMalloc(&d_tin,  kStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_tout, kStates * sizeof(AesState)));
    CHECK(cudaMemcpy(d_tin, h_tin.data(), kStates * sizeof(AesState),
                     cudaMemcpyHostToDevice));
    constexpr int kThreadsRt = 128; // 4 warps per block
    uint64_t blocks_rt = (kBatches + (kThreadsRt / 32) - 1) / (kThreadsRt / 32);
    bs32_roundtrip_kernel<<<blocks_rt, kThreadsRt>>>(d_tin, d_tout, kBatches);
    CHECK(cudaDeviceSynchronize());
    std::vector<AesState> h_tout(kStates);
    CHECK(cudaMemcpy(h_tout.data(), d_tout, kStates * sizeof(AesState),
                     cudaMemcpyDeviceToHost));
    uint64_t rt_mismatches = 0;
    for (uint64_t i = 0; i < kStates; ++i) {
        if (std::memcmp(&h_tin[i], &h_tout[i], sizeof(AesState)) != 0) {
            if (rt_mismatches < 4) {
                std::printf("  bs32 round-trip mismatch at i=%llu (batch %llu lane %llu)\n",
                            static_cast<unsigned long long>(i),
                            static_cast<unsigned long long>(i / 32),
                            static_cast<unsigned long long>(i % 32));
                std::printf("    in : %08x %08x %08x %08x\n",
                            h_tin[i].w[0], h_tin[i].w[1], h_tin[i].w[2], h_tin[i].w[3]);
                std::printf("    out: %08x %08x %08x %08x\n",
                            h_tout[i].w[0], h_tout[i].w[1], h_tout[i].w[2], h_tout[i].w[3]);
            }
            ++rt_mismatches;
        }
    }
    CHECK(cudaFree(d_tin));
    CHECK(cudaFree(d_tout));
    std::printf("  bs32 pack/unpack round-trip:     %llu / %llu  %s\n",
                static_cast<unsigned long long>(kStates - rt_mismatches),
                static_cast<unsigned long long>(kStates),
                rt_mismatches == 0 ? "OK" : "FAIL");

    // ------ Boyar-Peralta S-box: exhaustive 256-input test ------
    uint8_t *d_bp = nullptr, *d_ref = nullptr;
    CHECK(cudaMalloc(&d_bp,  256));
    CHECK(cudaMalloc(&d_ref, 256));
    bp_sbox_exhaustive_kernel<<<1, 256>>>(d_bp, d_ref);
    CHECK(cudaDeviceSynchronize());
    uint8_t h_bp[256], h_ref[256];
    CHECK(cudaMemcpy(h_bp,  d_bp,  256, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_ref, d_ref, 256, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_bp));
    CHECK(cudaFree(d_ref));
    uint64_t bp_mismatches = 0;
    for (int x = 0; x < 256; ++x) {
        if (h_bp[x] != h_ref[x]) {
            if (bp_mismatches < 8) {
                std::printf("  BP S-box mismatch: x=0x%02x  bp=0x%02x  ref=0x%02x\n",
                            x, h_bp[x], h_ref[x]);
            }
            ++bp_mismatches;
        }
    }
    std::printf("  BP S-box (scalar) vs kAesSBox:   %llu / 256  %s\n",
                static_cast<unsigned long long>(256 - bp_mismatches),
                bp_mismatches == 0 ? "OK" : "FAIL");

    // ------ BP S-box bit-sliced instantiation ------
    CHECK(cudaMalloc(&d_bp,  256));
    CHECK(cudaMalloc(&d_ref, 256));
    bp_sbox_bitsliced_all_kernel<<<1, 1>>>(d_bp, d_ref);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_bp,  d_bp,  256, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_ref, d_ref, 256, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_bp));
    CHECK(cudaFree(d_ref));
    uint64_t bs_bp_mismatches = 0;
    for (int x = 0; x < 256; ++x) {
        if (h_bp[x] != h_ref[x]) {
            if (bs_bp_mismatches < 8) {
                std::printf("  BP S-box<uint32> mismatch: x=0x%02x  bs=0x%02x  ref=0x%02x\n",
                            x, h_bp[x], h_ref[x]);
            }
            ++bs_bp_mismatches;
        }
    }
    std::printf("  BP S-box (bit-sliced) vs kAesSBox: %llu / 256  %s\n",
                static_cast<unsigned long long>(256 - bs_bp_mismatches),
                bs_bp_mismatches == 0 ? "OK" : "FAIL");

    // ------ sub_bytes_bs32 vs scalar SubBytes ------
    // Generate random 32-state batches, pack→sub_bytes_bs32→unpack, compare
    // each byte of each output state against kSBoxHost[input_byte].
    constexpr uint64_t kSubBatches = 4096; // 128K states
    constexpr uint64_t kSubStates  = kSubBatches * 32;
    std::vector<AesState> h_sbin(kSubStates);
    for (auto& s : h_sbin) {
        for (int j = 0; j < 4; ++j) s.w[j] = static_cast<uint32_t>(rng());
    }
    AesState *d_sbin = nullptr, *d_sbout = nullptr;
    CHECK(cudaMalloc(&d_sbin,  kSubStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_sbout, kSubStates * sizeof(AesState)));
    CHECK(cudaMemcpy(d_sbin, h_sbin.data(), kSubStates * sizeof(AesState),
                     cudaMemcpyHostToDevice));
    uint64_t blocks_sb = (kSubBatches + 3) / 4; // 4 warps/block
    bs32_sub_bytes_kernel<<<blocks_sb, 128>>>(d_sbin, d_sbout, kSubBatches);
    CHECK(cudaDeviceSynchronize());
    std::vector<AesState> h_sbout(kSubStates);
    CHECK(cudaMemcpy(h_sbout.data(), d_sbout, kSubStates * sizeof(AesState),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sbin));
    CHECK(cudaFree(d_sbout));
    // Need the host-side S-box to compare against.
    static constexpr uint8_t kSBox[256] = {
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    };
    uint64_t sb_mismatches = 0;
    for (uint64_t i = 0; i < kSubStates; ++i) {
        AesState expected;
        uint8_t const* inb  = reinterpret_cast<uint8_t const*>(&h_sbin[i]);
        uint8_t*       outb = reinterpret_cast<uint8_t*>(&expected);
        for (int b = 0; b < 16; ++b) outb[b] = kSBox[inb[b]];
        if (std::memcmp(&expected, &h_sbout[i], sizeof(AesState)) != 0) {
            if (sb_mismatches < 4) {
                uint8_t const* gb = reinterpret_cast<uint8_t const*>(&h_sbout[i]);
                std::printf("  sub_bytes_bs32 mismatch at state %llu (lane %llu)\n",
                            static_cast<unsigned long long>(i),
                            static_cast<unsigned long long>(i % 32));
                std::printf("    in : "); for (int b=0;b<16;++b) std::printf("%02x ", inb[b]); std::printf("\n");
                std::printf("    exp: "); for (int b=0;b<16;++b) std::printf("%02x ", outb[b]); std::printf("\n");
                std::printf("    got: "); for (int b=0;b<16;++b) std::printf("%02x ", gb[b]); std::printf("\n");
            }
            ++sb_mismatches;
        }
    }
    std::printf("  sub_bytes_bs32 vs scalar SubBytes: %llu / %llu  %s\n",
                static_cast<unsigned long long>(kSubStates - sb_mismatches),
                static_cast<unsigned long long>(kSubStates),
                sb_mismatches == 0 ? "OK" : "FAIL");

    // ------ shift_rows_bs32 vs scalar ShiftRows ------
    // Reuse h_sbin as the input; recompute scalar expected per state.
    CHECK(cudaMalloc(&d_sbin,  kSubStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_sbout, kSubStates * sizeof(AesState)));
    CHECK(cudaMemcpy(d_sbin, h_sbin.data(), kSubStates * sizeof(AesState),
                     cudaMemcpyHostToDevice));
    bs32_shift_rows_kernel<<<blocks_sb, 128>>>(d_sbin, d_sbout, kSubBatches);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_sbout.data(), d_sbout, kSubStates * sizeof(AesState),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sbin));
    CHECK(cudaFree(d_sbout));
    uint64_t sr_mismatches = 0;
    for (uint64_t i = 0; i < kSubStates; ++i) {
        AesState expected;
        uint8_t const* inb  = reinterpret_cast<uint8_t const*>(&h_sbin[i]);
        uint8_t*       outb = reinterpret_cast<uint8_t*>(&expected);
        for (int b = 0; b < 16; ++b) {
            int c = b >> 2, r = b & 3;
            int b_old = ((c + r) & 3) * 4 + r;
            outb[b] = inb[b_old];
        }
        if (std::memcmp(&expected, &h_sbout[i], sizeof(AesState)) != 0) {
            ++sr_mismatches;
        }
    }
    std::printf("  shift_rows_bs32 vs scalar:         %llu / %llu  %s\n",
                static_cast<unsigned long long>(kSubStates - sr_mismatches),
                static_cast<unsigned long long>(kSubStates),
                sr_mismatches == 0 ? "OK" : "FAIL");

    // ------ add_round_key_bs32 vs scalar AddRoundKey ------
    std::vector<AesState> h_keys_bs(kSubBatches);
    for (auto& k : h_keys_bs) {
        for (int j = 0; j < 4; ++j) k.w[j] = static_cast<uint32_t>(rng());
    }
    AesState *d_keys_bs = nullptr;
    CHECK(cudaMalloc(&d_sbin,  kSubStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_sbout, kSubStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_keys_bs,  kSubBatches * sizeof(AesState)));
    CHECK(cudaMemcpy(d_sbin, h_sbin.data(), kSubStates * sizeof(AesState),
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_keys_bs, h_keys_bs.data(), kSubBatches * sizeof(AesState),
                     cudaMemcpyHostToDevice));
    bs32_ark_kernel<<<blocks_sb, 128>>>(d_sbin, d_keys_bs, d_sbout, kSubBatches);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_sbout.data(), d_sbout, kSubStates * sizeof(AesState),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sbin));
    CHECK(cudaFree(d_sbout));
    CHECK(cudaFree(d_keys_bs));
    uint64_t ark_mismatches = 0;
    for (uint64_t i = 0; i < kSubStates; ++i) {
        AesState const& key = h_keys_bs[i / 32];
        AesState expected;
        for (int j = 0; j < 4; ++j) expected.w[j] = h_sbin[i].w[j] ^ key.w[j];
        if (std::memcmp(&expected, &h_sbout[i], sizeof(AesState)) != 0) {
            ++ark_mismatches;
        }
    }
    std::printf("  add_round_key_bs32 vs scalar:      %llu / %llu  %s\n",
                static_cast<unsigned long long>(kSubStates - ark_mismatches),
                static_cast<unsigned long long>(kSubStates),
                ark_mismatches == 0 ? "OK" : "FAIL");

    // ------ mix_columns_bs32 vs scalar MixColumns ------
    auto scalar_xtime = [](uint8_t x) -> uint8_t {
        return uint8_t((x << 1) ^ ((x & 0x80) ? 0x1B : 0));
    };
    auto scalar_mix_columns = [&](uint8_t s[16]) {
        uint8_t t[16];
        for (int i = 0; i < 16; ++i) t[i] = s[i];
        for (int cc = 0; cc < 4; ++cc) {
            uint8_t a0 = t[cc*4+0], a1 = t[cc*4+1], a2 = t[cc*4+2], a3 = t[cc*4+3];
            s[cc*4+0] = uint8_t(scalar_xtime(a0) ^ scalar_xtime(a1) ^ a1 ^ a2 ^ a3);
            s[cc*4+1] = uint8_t(a0 ^ scalar_xtime(a1) ^ scalar_xtime(a2) ^ a2 ^ a3);
            s[cc*4+2] = uint8_t(a0 ^ a1 ^ scalar_xtime(a2) ^ scalar_xtime(a3) ^ a3);
            s[cc*4+3] = uint8_t(scalar_xtime(a0) ^ a0 ^ a1 ^ a2 ^ scalar_xtime(a3));
        }
    };

    CHECK(cudaMalloc(&d_sbin,  kSubStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_sbout, kSubStates * sizeof(AesState)));
    CHECK(cudaMemcpy(d_sbin, h_sbin.data(), kSubStates * sizeof(AesState),
                     cudaMemcpyHostToDevice));
    bs32_mix_columns_kernel<<<blocks_sb, 128>>>(d_sbin, d_sbout, kSubBatches);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_sbout.data(), d_sbout, kSubStates * sizeof(AesState),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sbin));
    CHECK(cudaFree(d_sbout));
    uint64_t mc_mismatches = 0;
    for (uint64_t i = 0; i < kSubStates; ++i) {
        AesState expected = h_sbin[i];
        scalar_mix_columns(reinterpret_cast<uint8_t*>(&expected));
        if (std::memcmp(&expected, &h_sbout[i], sizeof(AesState)) != 0) {
            if (mc_mismatches < 4) {
                uint8_t const* inb = reinterpret_cast<uint8_t const*>(&h_sbin[i]);
                uint8_t const* eb  = reinterpret_cast<uint8_t const*>(&expected);
                uint8_t const* gb  = reinterpret_cast<uint8_t const*>(&h_sbout[i]);
                std::printf("  mix_columns_bs32 mismatch at state %llu (lane %llu)\n",
                            static_cast<unsigned long long>(i),
                            static_cast<unsigned long long>(i % 32));
                std::printf("    in : "); for (int bb=0;bb<16;++bb) std::printf("%02x ", inb[bb]); std::printf("\n");
                std::printf("    exp: "); for (int bb=0;bb<16;++bb) std::printf("%02x ", eb[bb]); std::printf("\n");
                std::printf("    got: "); for (int bb=0;bb<16;++bb) std::printf("%02x ", gb[bb]); std::printf("\n");
            }
            ++mc_mismatches;
        }
    }
    std::printf("  mix_columns_bs32 vs scalar:        %llu / %llu  %s\n",
                static_cast<unsigned long long>(kSubStates - mc_mismatches),
                static_cast<unsigned long long>(kSubStates),
                mc_mismatches == 0 ? "OK" : "FAIL");

    // ------ aesenc_round_bs32_warp vs 32 × aesenc_round (scalar T-table) ------
    CHECK(cudaMalloc(&d_sbin,   kSubStates  * sizeof(AesState)));
    CHECK(cudaMalloc(&d_sbout,  kSubStates  * sizeof(AesState)));
    CHECK(cudaMalloc(&d_keys_bs, kSubBatches * sizeof(AesState)));
    CHECK(cudaMemcpy(d_sbin,    h_sbin.data(),    kSubStates * sizeof(AesState), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_keys_bs, h_keys_bs.data(), kSubBatches * sizeof(AesState), cudaMemcpyHostToDevice));
    bs32_round_kernel<<<blocks_sb, 128>>>(d_sbin, d_keys_bs, d_sbout, kSubBatches);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_sbout.data(), d_sbout, kSubStates * sizeof(AesState), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sbin));
    CHECK(cudaFree(d_sbout));
    CHECK(cudaFree(d_keys_bs));
    // Build expected by running the T-table aesenc_round on each state.
    std::vector<AesState> h_expected(kSubStates);
    AesState *d_e_in = nullptr, *d_e_keys = nullptr, *d_e_out = nullptr;
    // One key per 32 states — replicate to a per-state array for the scalar kernel.
    std::vector<AesState> h_keys_flat(kSubStates);
    for (uint64_t i = 0; i < kSubStates; ++i) h_keys_flat[i] = h_keys_bs[i / 32];
    CHECK(cudaMalloc(&d_e_in,   kSubStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_e_keys, kSubStates * sizeof(AesState)));
    CHECK(cudaMalloc(&d_e_out,  kSubStates * sizeof(AesState)));
    CHECK(cudaMemcpy(d_e_in,   h_sbin.data(),       kSubStates * sizeof(AesState), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_e_keys, h_keys_flat.data(),  kSubStates * sizeof(AesState), cudaMemcpyHostToDevice));
    uint64_t blocks_e = (kSubStates + 255) / 256;
    tt_round_kernel<<<blocks_e, 256>>>(d_e_in, d_e_keys, d_e_out, kSubStates);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_expected.data(), d_e_out, kSubStates * sizeof(AesState), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_e_in));
    CHECK(cudaFree(d_e_keys));
    CHECK(cudaFree(d_e_out));
    uint64_t round_mismatches = 0;
    for (uint64_t i = 0; i < kSubStates; ++i) {
        if (std::memcmp(&h_expected[i], &h_sbout[i], sizeof(AesState)) != 0) {
            if (round_mismatches < 4) {
                uint8_t const* inb = reinterpret_cast<uint8_t const*>(&h_sbin[i]);
                uint8_t const* kb  = reinterpret_cast<uint8_t const*>(&h_keys_flat[i]);
                uint8_t const* eb  = reinterpret_cast<uint8_t const*>(&h_expected[i]);
                uint8_t const* gb  = reinterpret_cast<uint8_t const*>(&h_sbout[i]);
                std::printf("  aesenc_round_bs32_warp mismatch at state %llu (lane %llu)\n",
                            static_cast<unsigned long long>(i),
                            static_cast<unsigned long long>(i % 32));
                std::printf("    in : "); for (int bb=0;bb<16;++bb) std::printf("%02x ", inb[bb]); std::printf("\n");
                std::printf("    key: "); for (int bb=0;bb<16;++bb) std::printf("%02x ", kb[bb]); std::printf("\n");
                std::printf("    exp: "); for (int bb=0;bb<16;++bb) std::printf("%02x ", eb[bb]); std::printf("\n");
                std::printf("    got: "); for (int bb=0;bb<16;++bb) std::printf("%02x ", gb[bb]); std::printf("\n");
            }
            ++round_mismatches;
        }
    }
    std::printf("  aesenc_round_bs32_warp vs T-table: %llu / %llu  %s\n",
                static_cast<unsigned long long>(kSubStates - round_mismatches),
                static_cast<unsigned long long>(kSubStates),
                round_mismatches == 0 ? "OK" : "FAIL");

    // ------ run_rounds_bs32_warp vs scalar run_rounds (16 round pairs) ------
    constexpr int   kRounds    = pos2gpu::kAesPairingRounds;
    constexpr uint64_t kRRBatches = 1024;                    // 32 K states
    constexpr uint64_t kRRStates  = kRRBatches * 32;
    std::vector<AesState> h_rrin(kRRStates), h_rrkeys(kRRBatches * 2);
    for (auto& s : h_rrin)   for (int j=0;j<4;++j) s.w[j] = static_cast<uint32_t>(rng());
    for (auto& k : h_rrkeys) for (int j=0;j<4;++j) k.w[j] = static_cast<uint32_t>(rng());
    AesState *d_rrin = nullptr, *d_rrkeys = nullptr;
    AesState *d_rrbs_out = nullptr, *d_rrtt_out = nullptr;
    CHECK(cudaMalloc(&d_rrin,     kRRStates   * sizeof(AesState)));
    CHECK(cudaMalloc(&d_rrkeys,   kRRBatches * 2 * sizeof(AesState)));
    CHECK(cudaMalloc(&d_rrbs_out, kRRStates   * sizeof(AesState)));
    CHECK(cudaMalloc(&d_rrtt_out, kRRStates   * sizeof(AesState)));
    CHECK(cudaMemcpy(d_rrin,   h_rrin.data(),   kRRStates * sizeof(AesState), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rrkeys, h_rrkeys.data(), kRRBatches * 2 * sizeof(AesState), cudaMemcpyHostToDevice));
    uint64_t bs_blocks = (kRRBatches + 3) / 4;  // 4 warps / block
    bs32_run_rounds_kernel<<<bs_blocks, 128>>>(d_rrin, d_rrkeys, d_rrbs_out, kRRBatches, kRounds);
    uint64_t tt_blocks = (kRRStates + 255) / 256;
    tt_run_rounds_kernel<<<tt_blocks, 256>>>(d_rrin, d_rrkeys, d_rrtt_out, kRRStates, kRounds);
    CHECK(cudaDeviceSynchronize());
    std::vector<AesState> h_rrbs(kRRStates), h_rrtt(kRRStates);
    CHECK(cudaMemcpy(h_rrbs.data(), d_rrbs_out, kRRStates * sizeof(AesState), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_rrtt.data(), d_rrtt_out, kRRStates * sizeof(AesState), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_rrin));
    CHECK(cudaFree(d_rrkeys));
    CHECK(cudaFree(d_rrbs_out));
    CHECK(cudaFree(d_rrtt_out));
    uint64_t rr_mismatches = 0;
    for (uint64_t i = 0; i < kRRStates; ++i) {
        if (std::memcmp(&h_rrbs[i], &h_rrtt[i], sizeof(AesState)) != 0) {
            ++rr_mismatches;
        }
    }
    std::printf("  run_rounds_bs32_warp vs T-table:   %llu / %llu  %s\n",
                static_cast<unsigned long long>(kRRStates - rr_mismatches),
                static_cast<unsigned long long>(kRRStates),
                rr_mismatches == 0 ? "OK" : "FAIL");

    if (mismatches == 0 && rt_mismatches == 0 && bp_mismatches == 0
        && bs_bp_mismatches == 0 && sb_mismatches == 0
        && sr_mismatches == 0 && ark_mismatches == 0
        && mc_mismatches == 0 && round_mismatches == 0
        && rr_mismatches == 0) {
        std::printf("\n==> ALL OK\n");
        return 0;
    }
    std::printf("\n==> FAIL\n");
    return 1;
}
