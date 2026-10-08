// FeistelCipherGpu.cuh — device-side mirror of pos2-chip's FeistelCipher.
// Pure arithmetic, no external state. Used by T3 to encode proof fragments.
//
// Cross-reference: pos2-chip/src/pos/FeistelCipher.hpp

#pragma once

#include <cuda_runtime.h>
#include "pos/FeistelCipher.hpp"

#include <cstdint>

namespace pos2gpu {

struct FeistelKey {
    uint32_t k;
    FeistelCipher::FullRoundKey round_key;
};

// The SHA-256 schedule runs once per plot on the host. Kernels receive the
// twelve mixing words, exactly as extracted by the pinned CPU reference.
inline FeistelKey make_feistel_key(uint8_t const* plot_id, int k)
{
    FeistelCipher const cipher(plot_id, static_cast<uint32_t>(k));
    return {cipher.k_, cipher.round_key_};
}

__host__ __device__ inline uint64_t feistel_rotate_left(uint64_t value, uint64_t shift, uint64_t bit_length)
{
    if (shift > bit_length) shift = bit_length;
    uint64_t mask = (bit_length == 64 ? ~0ULL : ((1ULL << bit_length) - 1));
    return ((value << shift) & mask) | (value >> (bit_length - shift));
}

struct FeistelResultGpu { uint64_t left, right; };

__host__ __device__ inline FeistelResultGpu feistel_round(
    FeistelKey const& fk, uint64_t left, uint64_t right, FeistelCipher::RoundKey round_key)
{
    int k = fk.k;
    uint64_t bitmask = (k == 64 ? ~0ULL : ((1ULL << k) - 1));
    uint64_t a = right;
    uint64_t b = round_key.b;
    uint64_t c = round_key.c;
    uint64_t d = round_key.d;

    a = (a + b) & bitmask;
    d = feistel_rotate_left(d ^ a, 16, k);
    c = (c + d) & bitmask;
    b = feistel_rotate_left(b ^ c, 12, k);

    a = (a + b) & bitmask;
    d = feistel_rotate_left(d ^ a, 8, k);
    c = (c + d) & bitmask;
    b = feistel_rotate_left(b ^ c, 7, k);

    FeistelResultGpu res;
    res.left  = right;
    res.right = (left ^ b) & bitmask;
    return res;
}

__host__ __device__ inline uint64_t feistel_encrypt(FeistelKey const& fk, uint64_t input_value)
{
    int k = fk.k;
    uint64_t bitmask = (k == 64 ? ~0ULL : ((1ULL << k) - 1));
    uint64_t left  = (input_value >> k) & bitmask;
    uint64_t right = input_value & bitmask;
    for (int r = 0; r < 4; ++r) {
        auto const round_key = fk.round_key.round[r];
        FeistelResultGpu res = feistel_round(fk, left, right, round_key);
        left  = res.left;
        right = res.right;
    }
    return (left << k) | right;
}

} // namespace pos2gpu
