// AesGpu.cuh — single-round AES on the device, semantically equivalent to
// Intel's _mm_aesenc_si128(state, key):
//
//     A = ShiftRows(state)
//     B = SubBytes(A)
//     C = MixColumns(B)
//     out = C XOR key
//
// We use the standard 4-table T-box construction. Each of T0..T3 packs
// MixColumns(SubBytes(x)) for one column rotation. One round therefore
// becomes 16 byte-loads + 12 XORs after applying ShiftRows by index
// permutation.
//
// The state is laid out as four little-endian uint32 words, matching how
// the CPU code loads it via rx_set_int_vec_i128(i3, i2, i1, i0):
//   state.w[0] = i0  (bytes  0.. 3)
//   state.w[1] = i1  (bytes  4.. 7)
//   state.w[2] = i2  (bytes  8..11)
//   state.w[3] = i3  (bytes 12..15)
//
// Cross-check against pos2-chip/src/pos/aes/intrin_portable.h which
// defines `rx_aesenc_vec_i128 _mm_aesenc_si128`.
//
// Backend portability:
//
// The SYCL path (compiled by acpp/clang in non-CUDA mode) cannot see
// __constant__ memory, threadIdx, or __device__ markup. The pieces it
// needs — aesenc_round_smem, set_int_vec_i128, load_state_le, and the
// AesState struct itself — are decorated with the portable macros from
// PortableAttrs.hpp and stay outside the __CUDACC__ gate. The constant-
// memory T-tables, the aesenc_round variant that reads them, and
// load_aes_tables_smem (uses threadIdx) are CUDA-only.

#pragma once

#include "gpu/PortableAttrs.hpp"

#include <cstdint>

#if defined(__CUDACC__)
  #include <cuda_runtime.h>
#endif

namespace pos2gpu {

#if defined(__CUDACC__)
// AES T-tables in constant memory. Defined in AesGpu.cu, populated by
// initialize_aes_tables() at startup.
__device__ __constant__ extern uint32_t kAesT0[256];
__device__ __constant__ extern uint32_t kAesT1[256];
__device__ __constant__ extern uint32_t kAesT2[256];
__device__ __constant__ extern uint32_t kAesT3[256];
#endif

struct AesState {
    uint32_t w[4];
};

// Load 16 bytes (little-endian) into an AesState.
POS2_HOST_DEVICE_INLINE AesState load_state_le(uint8_t const* bytes)
{
    AesState s;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        s.w[i] = uint32_t(bytes[4*i + 0])
               | (uint32_t(bytes[4*i + 1]) << 8)
               | (uint32_t(bytes[4*i + 2]) << 16)
               | (uint32_t(bytes[4*i + 3]) << 24);
    }
    return s;
}

#if defined(__CUDACC__)
// One AES round equivalent to _mm_aesenc_si128(state, key), reading the
// T-tables from constant memory. CUDA-only because __constant__ has no
// SYCL equivalent — the SYCL path uses aesenc_round_smem with tables
// preloaded into local memory.
__device__ __forceinline__ AesState aesenc_round(AesState s, AesState const& key)
{
    auto byte = [](uint32_t w, int n) -> uint32_t {
        return (w >> (8 * n)) & 0xFFu;
    };

    AesState out;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint32_t v = kAesT0[byte(s.w[c],         0)]
                   ^ kAesT1[byte(s.w[(c + 1) & 3], 1)]
                   ^ kAesT2[byte(s.w[(c + 2) & 3], 2)]
                   ^ kAesT3[byte(s.w[(c + 3) & 3], 3)];
        out.w[c] = v ^ key.w[c];
    }
    return out;
}
#endif

// Convenience: load an i128 from four little-endian 32-bit ints, matching
// rx_set_int_vec_i128(i3, i2, i1, i0).
POS2_HOST_DEVICE_INLINE AesState set_int_vec_i128(int32_t i3, int32_t i2, int32_t i1, int32_t i0)
{
    AesState s;
    s.w[0] = static_cast<uint32_t>(i0);
    s.w[1] = static_cast<uint32_t>(i1);
    s.w[2] = static_cast<uint32_t>(i2);
    s.w[3] = static_cast<uint32_t>(i3);
    return s;
}

// Initialize the constant-memory T-tables on first use. Must be called once
// per program from host code before any kernel that touches AesGpu runs.
// Implemented in AesGpu.cu (CUDA TU only).
void initialize_aes_tables();

// =========================================================================
// Shared-memory variant. Each kernel block can call load_aes_tables_smem
// at start to populate per-block 4×256 uint32 tables (16 KB shared mem)
// from constant memory; subsequent rounds read from SRAM, avoiding the
// constant-memory broadcast serialization that hurts when each warp lane
// looks up a different byte value.
//
// Usage in a kernel with blockDim.x ∈ [128, 1024]:
//
//   __shared__ uint32_t sT[4 * 256];
//   load_aes_tables_smem(sT);
//   __syncthreads();
//   AesState state = ...;
//   state = aesenc_round_smem(state, round_key, sT);
//
// The SYCL path uses the same aesenc_round_smem (pointer-based, fully
// portable) but provides its own loader — local_accessor + nd_item barrier
// in place of __shared__ + __syncthreads — and supplies the table data
// from a USM buffer initialised from AesTables.inl on the host side.
// =========================================================================

#if defined(__CUDACC__)
__device__ __forceinline__ void load_aes_tables_smem(uint32_t* sT)
{
    // sT layout: [T0|T1|T2|T3], 256 entries each (4096 entries total).
    int tid = threadIdx.x;
    int stride = blockDim.x;
    #pragma unroll 1
    for (int i = tid; i < 256; i += stride) {
        sT[0 * 256 + i] = kAesT0[i];
        sT[1 * 256 + i] = kAesT1[i];
        sT[2 * 256 + i] = kAesT2[i];
        sT[3 * 256 + i] = kAesT3[i];
    }
}
#endif

POS2_DEVICE_INLINE AesState aesenc_round_smem(
    AesState s, AesState const& key, uint32_t const* __restrict__ sT)
{
    auto byte = [](uint32_t w, int n) -> uint32_t {
        return (w >> (8 * n)) & 0xFFu;
    };
    AesState out;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint32_t v = sT[0 * 256 + byte(s.w[c],         0)]
                   ^ sT[1 * 256 + byte(s.w[(c + 1) & 3], 1)]
                   ^ sT[2 * 256 + byte(s.w[(c + 2) & 3], 2)]
                   ^ sT[3 * 256 + byte(s.w[(c + 3) & 3], 3)];
        out.w[c] = v ^ key.w[c];
    }
    return out;
}

} // namespace pos2gpu
