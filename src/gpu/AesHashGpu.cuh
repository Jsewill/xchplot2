// AesHashGpu.cuh — device-side mirror of pos2-chip's AesHash class.
// One round-key pair derived from a 32-byte plot_id; functions g_x,
// pairing, matching_target, chain that consume it.
//
// Cross-reference:
//   pos2-chip/src/pos/aes/AesHash.hpp
//
// The CPU code uses 16 alternating rounds (round_key_1, round_key_2). We
// keep the same round count constants here so a single binary can be a
// drop-in for the CPU code.
//
// Backend portability:
//
// The `_smem` family (run_rounds_smem, g_x_smem, pairing_smem,
// matching_target_smem, chain_smem) is fully pointer-driven (table
// pointer passed as an argument) and decorated with portable macros, so
// it compiles under both nvcc and acpp/clang. The non-smem family reads
// the constant-memory T-tables directly via aesenc_round and is
// therefore CUDA-only.

#pragma once

#include "gpu/AesGpu.cuh"
#include "gpu/PortableAttrs.hpp"

#include <cstdint>

namespace pos2gpu {

constexpr int kAesGRounds              = 16;
constexpr int kAesPairingRounds        = 16;
constexpr int kAesMatchingTargetRounds = 16;
constexpr int kAesChainingRounds       = 16;

struct AesHashKeys {
    AesState round_key_1;
    AesState round_key_2;
};

// Build the two round keys from a 32-byte plot_id, matching
// load_plot_id_as_aes_key in AesHash.hpp.
POS2_HOST_DEVICE inline AesHashKeys make_keys(uint8_t const* plot_id_bytes)
{
    AesHashKeys k;
    k.round_key_1 = load_state_le(plot_id_bytes + 0);
    k.round_key_2 = load_state_le(plot_id_bytes + 16);
    return k;
}

#if defined(__CUDACC__)
// One full alternating round-pair. The CPU loop is:
//   for r in 0..Rounds: state = aesenc(state, k1); state = aesenc(state, k2);
// CUDA-only: calls aesenc_round which reads constant-memory T-tables.
__device__ __forceinline__ AesState run_rounds(AesState state, AesHashKeys const& keys, int rounds)
{
    #pragma unroll 2
    for (int r = 0; r < rounds; ++r) {
        state = aesenc_round(state, keys.round_key_1);
        state = aesenc_round(state, keys.round_key_2);
    }
    return state;
}

// g_x: input is uint32_t x in i0; output is bottom k bits of state.x.
// Mirrors AesHash::g_x.
__device__ __forceinline__ uint32_t g_x(AesHashKeys const& keys, uint32_t x, int k, int rounds = kAesGRounds)
{
    AesState s = set_int_vec_i128(0, 0, 0, static_cast<int32_t>(x));
    s = run_rounds(s, keys, rounds);
    return s.w[0] & ((1u << k) - 1u);
}
#endif

// pairing: load (meta_l_lo, meta_l_hi, meta_r_lo, meta_r_hi) into i0..i3,
// run AES_PAIRING_ROUNDS << extra_rounds_bits, return all 4 u32s.
// Mirrors AesHash::pairing<Soft>.
struct Result128 { uint32_t r[4]; };

#if defined(__CUDACC__)
__device__ __forceinline__ Result128 pairing(
    AesHashKeys const& keys,
    uint64_t meta_l, uint64_t meta_r,
    int extra_rounds_bits = 0)
{
    int32_t i0 = static_cast<int32_t>(meta_l & 0xFFFFFFFFu);
    int32_t i1 = static_cast<int32_t>((meta_l >> 32) & 0xFFFFFFFFu);
    int32_t i2 = static_cast<int32_t>(meta_r & 0xFFFFFFFFu);
    int32_t i3 = static_cast<int32_t>((meta_r >> 32) & 0xFFFFFFFFu);
    AesState s = set_int_vec_i128(i3, i2, i1, i0);
    int rounds = kAesPairingRounds << extra_rounds_bits;
    s = run_rounds(s, keys, rounds);
    Result128 out;
    out.r[0] = s.w[0];
    out.r[1] = s.w[1];
    out.r[2] = s.w[2];
    out.r[3] = s.w[3];
    return out;
}

// matching_target: load (table_id, match_key, meta_lo, meta_hi) into i0..i3,
// run AES_MATCHING_TARGET_ROUNDS << extra_rounds_bits, return state.x.
// Mirrors AesHash::matching_target<Soft>.
__device__ __forceinline__ uint32_t matching_target(
    AesHashKeys const& keys,
    uint32_t table_id, uint32_t match_key, uint64_t meta,
    int extra_rounds_bits = 0)
{
    int32_t i0 = static_cast<int32_t>(table_id);
    int32_t i1 = static_cast<int32_t>(match_key);
    int32_t i2 = static_cast<int32_t>(meta & 0xFFFFFFFFu);
    int32_t i3 = static_cast<int32_t>((meta >> 32) & 0xFFFFFFFFu);
    AesState s = set_int_vec_i128(i3, i2, i1, i0);
    int rounds = kAesMatchingTargetRounds << extra_rounds_bits;
    s = run_rounds(s, keys, rounds);
    return s.w[0];
}

// chain: 64-bit input in i0/i1, AES_CHAINING_ROUNDS rounds, return lo|hi.
// Mirrors AesHash::chain<Soft>.
__device__ __forceinline__ uint64_t chain(AesHashKeys const& keys, uint64_t input)
{
    int32_t i0 = static_cast<int32_t>(input & 0xFFFFFFFFu);
    int32_t i1 = static_cast<int32_t>((input >> 32) & 0xFFFFFFFFu);
    AesState s = set_int_vec_i128(0, 0, i1, i0);
    s = run_rounds(s, keys, kAesChainingRounds);
    return uint64_t(s.w[0]) | (uint64_t(s.w[1]) << 32);
}
#endif // __CUDACC__

// =========================================================================
// Shared-memory T-table variants. Use after load_aes_tables_smem(sT) +
// __syncthreads() in CUDA, or after a SYCL local_accessor + barrier in
// SYCL. All five functions mirror their constant-memory peers above;
// only the inner aesenc_round_smem call (and the table pointer arg)
// differ. Fully portable — compile under both backends.
// =========================================================================

POS2_DEVICE_INLINE AesState run_rounds_smem(
    AesState state, AesHashKeys const& keys, int rounds, uint32_t const* __restrict__ sT)
{
    #pragma unroll 2
    for (int r = 0; r < rounds; ++r) {
        state = aesenc_round_smem(state, keys.round_key_1, sT);
        state = aesenc_round_smem(state, keys.round_key_2, sT);
    }
    return state;
}

POS2_DEVICE_INLINE uint32_t g_x_smem(
    AesHashKeys const& keys, uint32_t x, int k,
    uint32_t const* __restrict__ sT, int rounds = kAesGRounds)
{
    AesState s = set_int_vec_i128(0, 0, 0, static_cast<int32_t>(x));
    s = run_rounds_smem(s, keys, rounds, sT);
    return s.w[0] & ((1u << k) - 1u);
}

POS2_DEVICE_INLINE Result128 pairing_smem(
    AesHashKeys const& keys,
    uint64_t meta_l, uint64_t meta_r,
    uint32_t const* __restrict__ sT,
    int extra_rounds_bits = 0)
{
    int32_t i0 = static_cast<int32_t>(meta_l & 0xFFFFFFFFu);
    int32_t i1 = static_cast<int32_t>((meta_l >> 32) & 0xFFFFFFFFu);
    int32_t i2 = static_cast<int32_t>(meta_r & 0xFFFFFFFFu);
    int32_t i3 = static_cast<int32_t>((meta_r >> 32) & 0xFFFFFFFFu);
    AesState s = set_int_vec_i128(i3, i2, i1, i0);
    int rounds = kAesPairingRounds << extra_rounds_bits;
    s = run_rounds_smem(s, keys, rounds, sT);
    Result128 out;
    out.r[0] = s.w[0]; out.r[1] = s.w[1];
    out.r[2] = s.w[2]; out.r[3] = s.w[3];
    return out;
}

POS2_DEVICE_INLINE uint32_t matching_target_smem(
    AesHashKeys const& keys,
    uint32_t table_id, uint32_t match_key, uint64_t meta,
    uint32_t const* __restrict__ sT,
    int extra_rounds_bits = 0)
{
    int32_t i0 = static_cast<int32_t>(table_id);
    int32_t i1 = static_cast<int32_t>(match_key);
    int32_t i2 = static_cast<int32_t>(meta & 0xFFFFFFFFu);
    int32_t i3 = static_cast<int32_t>((meta >> 32) & 0xFFFFFFFFu);
    AesState s = set_int_vec_i128(i3, i2, i1, i0);
    int rounds = kAesMatchingTargetRounds << extra_rounds_bits;
    s = run_rounds_smem(s, keys, rounds, sT);
    return s.w[0];
}

POS2_DEVICE_INLINE uint64_t chain_smem(
    AesHashKeys const& keys, uint64_t input,
    uint32_t const* __restrict__ sT)
{
    int32_t i0 = static_cast<int32_t>(input & 0xFFFFFFFFu);
    int32_t i1 = static_cast<int32_t>((input >> 32) & 0xFFFFFFFFu);
    AesState s = set_int_vec_i128(0, 0, i1, i0);
    s = run_rounds_smem(s, keys, kAesChainingRounds, sT);
    return uint64_t(s.w[0]) | (uint64_t(s.w[1]) << 32);
}

} // namespace pos2gpu
