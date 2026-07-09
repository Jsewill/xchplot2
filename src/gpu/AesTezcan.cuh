// AesTezcan.cuh — bank-conflict-aware T-table AES round variants following
// Cihangir Tezcan's "Breakthrough AES Performance on CUDA Devices"
// (https://ieeexplore.ieee.org/document/9422754).
//
// Two tricks are combined:
//
//  1) Single T-table + __byte_perm for T1/T2/T3. The standard AES T-tables
//     are cyclic byte rotations of T0:
//       T1[b] = ror(T0[b],  8)
//       T2[b] = ror(T0[b], 16)
//       T3[b] = ror(T0[b], 24)
//     Storing only T0 cuts the table footprint 4x. The __byte_perm
//     intrinsic does the rotation in 1 instruction; on Ada the added
//     ALU is cheap vs the shared-memory port pressure it saves.
//
//  2) Bank-replicated T0 with `sT0[256][BANK_SIZE]`. Each warp thread
//     reads `sT0[byte_val][threadIdx.x & (BANK_SIZE - 1)]`, so 32 warp
//     lanes land on 32 distinct banks regardless of `byte_val`. With
//     BANK_SIZE = 32 this eliminates bank conflicts entirely at the
//     cost of 32x the smem (32 KB/block vs 1 KB). Intermediate values
//     (4, 8) trade smem for residual conflicts.
//
// These variants are bit-identical to pos2gpu::aesenc_round_smem; they
// differ only in memory layout and access pattern. Parity-verified in
// tools/parity/aes_tezcan_bench.cu before any benchmarking.
//
// Hash-layer wrappers (g_x / pairing / matching_target / run_rounds) live
// in AesHashGpu.cuh so this header stays free of that include cycle.

#pragma once

#include "gpu/AesGpu.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace pos2gpu {

// Load T0 into shared memory, replicated BANK_SIZE times along the
// second dimension. Layout: sT0[byte_val * BANK_SIZE + replica].
template<int BANK_SIZE>
__device__ __forceinline__ void load_aes_t0_smem_rep(uint32_t* sT0)
{
    int tid = threadIdx.x;
    int stride = blockDim.x;
    #pragma unroll 1
    for (int i = tid; i < 256; i += stride) {
        uint32_t v = kAesT0[i];
        #pragma unroll
        for (int r = 0; r < BANK_SIZE; ++r) {
            sT0[i * BANK_SIZE + r] = v;
        }
    }
}

// One AES round: T0 lookup + __byte_perm for T1/T2/T3.
// BANK_SIZE must be a power of two in [1, 32].
template<int BANK_SIZE>
__device__ __forceinline__ AesState aesenc_round_smem_tezcan(
    AesState s, AesState const& key, uint32_t const* __restrict__ sT0)
{
    static_assert(BANK_SIZE >= 1 && BANK_SIZE <= 32,
                  "BANK_SIZE must be in [1, 32]");
    static_assert((BANK_SIZE & (BANK_SIZE - 1)) == 0,
                  "BANK_SIZE must be a power of two");

    int lane = (BANK_SIZE > 1) ? (threadIdx.x & (BANK_SIZE - 1)) : 0;

    auto load = [sT0, lane](uint32_t byte_val) -> uint32_t {
        if (BANK_SIZE == 1) return sT0[byte_val];
        return sT0[byte_val * BANK_SIZE + lane];
    };
    auto byte = [](uint32_t w, int n) -> uint32_t {
        return (w >> (8 * n)) & 0xFFu;
    };

    // Byte-permutation rotations over the standard Rijndael T0 layout
    // used by pos2-chip / pos2-gpu:
    //   T1[b] = ror24(T0[b]) = byte_perm(T0[b], T0[b], 0x2103)
    //   T2[b] = ror16(T0[b]) = byte_perm(T0[b], T0[b], 0x1032)
    //   T3[b] = ror8 (T0[b]) = byte_perm(T0[b], T0[b], 0x0321)
    //
    // Tezcan's published zero-fill shift trick (0x4321, 0x5432, 0x6543)
    // only works with a transposed T0 layout; we stick with the standard
    // layout and pay the identical 1-instruction PRMT cost for a real
    // rotation via __byte_perm(x, x, sel).
    AesState out;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint32_t v0 = load(byte(s.w[c],         0));
        uint32_t t1 = load(byte(s.w[(c + 1) & 3], 1));
        uint32_t t2 = load(byte(s.w[(c + 2) & 3], 2));
        uint32_t t3 = load(byte(s.w[(c + 3) & 3], 3));
        uint32_t v1 = __byte_perm(t1, t1, 0x2103);  // T1 = ror24(T0)
        uint32_t v2 = __byte_perm(t2, t2, 0x1032);  // T2 = ror16(T0)
        uint32_t v3 = __byte_perm(t3, t3, 0x0321);  // T3 = ror8 (T0)
        out.w[c] = v0 ^ v1 ^ v2 ^ v3 ^ key.w[c];
    }
    return out;
}

} // namespace pos2gpu
