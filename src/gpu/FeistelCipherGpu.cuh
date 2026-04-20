// FeistelCipherGpu.cuh — device-side mirror of pos2-chip's FeistelCipher.
// Pure arithmetic, no external state. Used by T3 to encode proof fragments.
//
// Cross-reference: pos2-chip/src/pos/FeistelCipher.hpp

#pragma once

#include "gpu/PortableAttrs.hpp"

#include <cstdint>

namespace pos2gpu {

struct FeistelKey {
    uint8_t plot_id[32];
    int k;
    int rounds;
};

POS2_HOST_DEVICE_INLINE FeistelKey make_feistel_key(uint8_t const* plot_id, int k, int rounds = 4)
{
    FeistelKey fk;
    fk.k = k;
    fk.rounds = rounds;
    #pragma unroll
    for (int i = 0; i < 32; ++i) fk.plot_id[i] = plot_id[i];
    return fk;
}

POS2_HOST_DEVICE_INLINE uint64_t feistel_rotate_left(uint64_t value, uint64_t shift, uint64_t bit_length)
{
    if (shift > bit_length) shift = bit_length;
    uint64_t mask = (bit_length == 64 ? ~0ULL : ((1ULL << bit_length) - 1));
    return ((value << shift) & mask) | (value >> (bit_length - shift));
}

POS2_HOST_DEVICE_INLINE uint64_t feistel_slice_key(FeistelKey const& fk, int start_bit, int num_bits)
{
    int start_byte    = start_bit / 8;
    int bit_offset    = start_bit % 8;
    int needed_bytes  = (bit_offset + num_bits + 7) / 8;
    if (start_byte + needed_bytes > 32) return 0;

    uint64_t key_segment = 0;
    for (int i = 0; i < needed_bytes; ++i)
        key_segment = (key_segment << 8) | uint64_t(fk.plot_id[start_byte + i]);
    int total_bits   = needed_bytes * 8;
    int shift_amount = total_bits - bit_offset - num_bits;
    uint64_t mask = (num_bits >= 64 ? ~0ULL : ((1ULL << num_bits) - 1));
    return (key_segment >> shift_amount) & mask;
}

POS2_HOST_DEVICE_INLINE uint64_t feistel_round_key(FeistelKey const& fk, int round_num)
{
    int half_length    = fk.k;
    int bits_for_round = 3 * half_length;
    int start_bit      = 0;
    if (fk.rounds > 1)
        start_bit = (round_num * (256 - 3 * half_length)) / (fk.rounds - 1);
    return feistel_slice_key(fk, start_bit, bits_for_round);
}

struct FeistelResultGpu { uint64_t left, right; };

POS2_HOST_DEVICE_INLINE FeistelResultGpu feistel_round(
    FeistelKey const& fk, uint64_t left, uint64_t right, uint64_t round_key)
{
    int k = fk.k;
    uint64_t bitmask = (k == 64 ? ~0ULL : ((1ULL << k) - 1));
    uint64_t a = right;
    uint64_t b = round_key & bitmask;
    uint64_t c = (round_key >> k) & bitmask;
    uint64_t d = (round_key >> (2 * k)) & bitmask;

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

POS2_HOST_DEVICE_INLINE uint64_t feistel_encrypt(FeistelKey const& fk, uint64_t input_value)
{
    int k = fk.k;
    uint64_t bitmask = (k == 64 ? ~0ULL : ((1ULL << k) - 1));
    uint64_t left  = (input_value >> k) & bitmask;
    uint64_t right = input_value & bitmask;
    for (int r = 0; r < fk.rounds; ++r) {
        uint64_t round_key = feistel_round_key(fk, r);
        FeistelResultGpu res = feistel_round(fk, left, right, round_key);
        left  = res.left;
        right = res.right;
    }
    return (left << k) | right;
}

} // namespace pos2gpu
