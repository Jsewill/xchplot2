// AesHashBsSycl.hpp — sub_group-cooperative bit-sliced AES hash for SYCL.
//
// Cross-reference:
//   src/gpu/AesGpuBitsliced.cuh  (CUDA original, 32-lane warp-coop)
//   src/gpu/AesHashGpu.cuh       (CUDA T-table API; _smem family)
//   src/gpu/AesSBoxBP.cuh        (Boyar-Peralta S-box circuit, shared)
//
// Exports sub_group-cooperative equivalents of g_x_smem / pairing_smem /
// matching_target_smem. Each kernel thread holds one state; 32 threads in
// a sub_group cooperate on 32 parallel AES computations, using only bit
// ops + sub_group shuffles — no T-table LDS lookups, which is what makes
// the bitsliced path win on amdgcn under AdaptiveCpp's HIP backend.
//
// Preconditions for callers:
//   - Kernel MUST be launched with reqd_sub_group_size(32) (wave32 on
//     RDNA2, warp32 on NVIDIA; both native). The shuffle/ballot math is
//     hard-coded for 32 lanes.
//   - ALL 32 lanes of the sub_group must participate in every call.
//     Lanes with no real work should pass dummy inputs, do the call,
//     then return afterwards.

#pragma once

#include "gpu/AesGpu.cuh"
#include "gpu/AesHashGpu.cuh"
#include "gpu/AesSBoxBP.cuh"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace pos2gpu {

// ---------- low-level sub_group primitives ----------

inline uint32_t bs_shfl(sycl::sub_group const& sg, uint32_t x, int lane)
{
    return sycl::select_from_group(sg, x, lane);
}

// Ballot: 32 lanes each contribute one bit, collected into a single
// uint32 mask (bit l of the result == lane l's predicate).
//
// Fast path on AdaptiveCpp's HIP target: __builtin_amdgcn_ballot_w32
// lowers to a single v_cmp + s_mov on RDNA2/3 — one native amdgcn
// instruction instead of the log-n reduction the portable fallback
// compiles to. This is the critical piece for bitsliced AES to win
// on amdgcn: bs32_pack calls ballot 128× per hash, so a 5× speedup
// per call is the difference between a +23 % regression (the first
// attempt with reduce_over_group<bit_or>) and a net win.
//
// Wave-size caveat: we hard-code _w32 because gfx1031 (RDNA2) is
// wave32 and the entire bitsliced scheme is wave32-only (reqd_sub_
// group_size(32) on the kernels, 32-way pack/unpack layout). Using
// _w64 on a wave32 target miscompiles — LLVM issue #62477.
//
// Recipe source: AdaptiveCpp doc/hip-source-interop.md — use
// __acpp_if_target_hip(...) so the amdgcn builtin only materialises
// during the HIP device pass; the host / SSCP path uses the portable
// SYCL reduction fallback.
inline uint32_t bs_ballot(sycl::sub_group const& sg, bool pred)
{
#if defined(__AMDGCN__) || defined(__HIP_DEVICE_COMPILE__)
    return static_cast<uint32_t>(__builtin_amdgcn_ballot_w32(pred));
#else
    uint32_t lane = sg.get_local_linear_id();
    uint32_t bit  = pred ? (1u << lane) : 0u;
    return sycl::reduce_over_group(sg, bit, sycl::bit_or<uint32_t>{});
#endif
}

// ---------- 32-way pack / unpack ----------
//
// Bit-plane layout matches AesGpuBitsliced.cuh:
//   plane p (0..127) has bit l = bit p of lane l's scalar state.
//   thread t owns planes { 4t, 4t+1, 4t+2, 4t+3 }.

inline void bs32_pack(sycl::sub_group const& sg,
                      AesState const& my, uint32_t out[4])
{
    uint32_t lane = sg.get_local_linear_id();
    for (int p = 0; p < 128; ++p) {
        int byte_idx    = p >> 3;
        int bit_in_byte = p & 7;
        int word_idx    = byte_idx >> 2;
        int byte_in_w   = byte_idx & 3;
        uint32_t bit = (my.w[word_idx] >> (8 * byte_in_w + bit_in_byte)) & 1u;
        uint32_t plane = bs_ballot(sg, bit != 0u);
        if (lane == uint32_t(p >> 2)) {
            out[p & 3] = plane;
        }
    }
}

inline void bs32_unpack(sycl::sub_group const& sg,
                        uint32_t const in[4], AesState& my)
{
    uint32_t lane = sg.get_local_linear_id();
    my.w[0] = my.w[1] = my.w[2] = my.w[3] = 0u;
    for (int p = 0; p < 128; ++p) {
        int owner = p >> 2;
        int slot  = p & 3;
        uint32_t plane = bs_shfl(sg, in[slot], owner);
        uint32_t bit = (plane >> lane) & 1u;
        int byte_idx    = p >> 3;
        int bit_in_byte = p & 7;
        int word_idx    = byte_idx >> 2;
        int byte_in_w   = byte_idx & 3;
        my.w[word_idx] |= bit << (8 * byte_in_w + bit_in_byte);
    }
}

// ---------- round key materialisation ----------
//
// All 32 states share the same key, so each bit-plane of a bit-sliced
// key is either all-ones or all-zeros. No cross-lane communication.

inline void make_bs32_round_key(sycl::sub_group const& sg,
                                AesState const& key, uint32_t key_bs[4])
{
    uint32_t lane = sg.get_local_linear_id();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int p = 4 * int(lane) + i;
        int byte_idx    = p >> 3;
        int bit_in_byte = p & 7;
        int word_idx    = byte_idx >> 2;
        int byte_in_w   = byte_idx & 3;
        uint32_t bit = (key.w[word_idx] >> (8 * byte_in_w + bit_in_byte)) & 1u;
        key_bs[i] = bit ? 0xFFFFFFFFu : 0u;
    }
}

inline void add_round_key_bs32(uint32_t bs[4], uint32_t const key_bs[4])
{
    bs[0] ^= key_bs[0]; bs[1] ^= key_bs[1];
    bs[2] ^= key_bs[2]; bs[3] ^= key_bs[3];
}

// ---------- ShiftRows ----------
//
// Each lane fetches its own output byte from a single source lane. The
// permutation preserves bit-within-byte index, so one shuffle per plane.

inline void shift_rows_bs32(sycl::sub_group const& sg, uint32_t bs[4])
{
    uint32_t lane  = sg.get_local_linear_id();
    int is_hi = int(lane) & 1;
    int b     = int(lane) >> 1;
    int c     = b >> 2;
    int r     = b & 3;
    int b_old = ((c + r) & 3) * 4 + r;
    int owner = 2 * b_old + is_hi;
    uint32_t n0 = bs_shfl(sg, bs[0], owner);
    uint32_t n1 = bs_shfl(sg, bs[1], owner);
    uint32_t n2 = bs_shfl(sg, bs[2], owner);
    uint32_t n3 = bs_shfl(sg, bs[3], owner);
    bs[0] = n0; bs[1] = n1; bs[2] = n2; bs[3] = n3;
}

// ---------- MixColumns ----------
//
// See AesGpuBitsliced.cuh for the algebraic derivation. 14 shuffles per
// lane (12 same-half column mates + 2 cross-half boundary bits).

inline void mix_columns_bs32(sycl::sub_group const& sg, uint32_t bs[4])
{
    uint32_t lane = sg.get_local_linear_id();
    int is_hi    = int(lane) & 1;
    int b        = int(lane) >> 1;
    int c        = b >> 2;
    int r        = b & 3;
    int partner  = int(lane) ^ 1;
    int col_base = 8 * c;
    int r1 = (r + 1) & 3;
    int r2 = (r + 2) & 3;
    int r3 = (r + 3) & 3;
    int L1 = col_base + 2 * r1 + is_hi;
    int L2 = col_base + 2 * r2 + is_hi;
    int L3 = col_base + 2 * r3 + is_hi;
    int L1_other = col_base + 2 * r1 + (is_hi ^ 1);

    uint32_t r1_0 = bs_shfl(sg, bs[0], L1);
    uint32_t r1_1 = bs_shfl(sg, bs[1], L1);
    uint32_t r1_2 = bs_shfl(sg, bs[2], L1);
    uint32_t r1_3 = bs_shfl(sg, bs[3], L1);
    uint32_t r2_0 = bs_shfl(sg, bs[0], L2);
    uint32_t r2_1 = bs_shfl(sg, bs[1], L2);
    uint32_t r2_2 = bs_shfl(sg, bs[2], L2);
    uint32_t r2_3 = bs_shfl(sg, bs[3], L2);
    uint32_t r3_0 = bs_shfl(sg, bs[0], L3);
    uint32_t r3_1 = bs_shfl(sg, bs[1], L3);
    uint32_t r3_2 = bs_shfl(sg, bs[2], L3);
    uint32_t r3_3 = bs_shfl(sg, bs[3], L3);

    uint32_t t_0 = bs[0] ^ r1_0;
    uint32_t t_1 = bs[1] ^ r1_1;
    uint32_t t_2 = bs[2] ^ r1_2;
    uint32_t t_3 = bs[3] ^ r1_3;

    uint32_t t_boundary = bs_shfl(sg, bs[3], partner)
                        ^ bs_shfl(sg, bs[3], L1_other);

    uint32_t xt_0, xt_1, xt_2, xt_3;
    if (is_hi) {
        xt_0 = t_boundary ^ t_3;
        xt_1 = t_0;
        xt_2 = t_1;
        xt_3 = t_2;
    } else {
        xt_0 = t_boundary;
        xt_1 = t_0 ^ t_boundary;
        xt_2 = t_1;
        xt_3 = t_2 ^ t_boundary;
    }

    bs[0] = xt_0 ^ r1_0 ^ r2_0 ^ r3_0;
    bs[1] = xt_1 ^ r1_1 ^ r2_1 ^ r3_1;
    bs[2] = xt_2 ^ r1_2 ^ r2_2 ^ r3_2;
    bs[3] = xt_3 ^ r1_3 ^ r2_3 ^ r3_3;
}

// ---------- SubBytes via Boyar-Peralta bitsliced S-box ----------
//
// Threads 2b and 2b+1 cooperate on byte b: they swap their four planes
// once, run the 113-gate BP circuit redundantly, then keep the four
// outputs for their own half of the byte.

inline void sub_bytes_bs32(sycl::sub_group const& sg, uint32_t bs[4])
{
    uint32_t lane = sg.get_local_linear_id();
    int is_hi   = int(lane) & 1;
    int partner = int(lane) ^ 1;

    uint32_t peer0 = bs_shfl(sg, bs[0], partner);
    uint32_t peer1 = bs_shfl(sg, bs[1], partner);
    uint32_t peer2 = bs_shfl(sg, bs[2], partner);
    uint32_t peer3 = bs_shfl(sg, bs[3], partner);

    uint32_t U0, U1, U2, U3, U4, U5, U6, U7;
    if (is_hi) {
        U0 = bs[3]; U1 = bs[2]; U2 = bs[1]; U3 = bs[0];
        U4 = peer3; U5 = peer2; U6 = peer1; U7 = peer0;
    } else {
        U0 = peer3; U1 = peer2; U2 = peer1; U3 = peer0;
        U4 = bs[3]; U5 = bs[2]; U6 = bs[1]; U7 = bs[0];
    }

    uint32_t S0, S1, S2, S3, S4, S5, S6, S7;
    bp_sbox_circuit<uint32_t>(U0, U1, U2, U3, U4, U5, U6, U7,
                               S0, S1, S2, S3, S4, S5, S6, S7,
                               0xFFFFFFFFu);

    if (is_hi) {
        bs[3] = S0; bs[2] = S1; bs[1] = S2; bs[0] = S3;
    } else {
        bs[3] = S4; bs[2] = S5; bs[1] = S6; bs[0] = S7;
    }
}

// ---------- full round + round loop ----------

inline void aesenc_round_bs32(sycl::sub_group const& sg,
                              uint32_t bs[4], uint32_t const key_bs[4])
{
    shift_rows_bs32(sg, bs);
    sub_bytes_bs32(sg, bs);
    mix_columns_bs32(sg, bs);
    add_round_key_bs32(bs, key_bs);
}

inline void run_rounds_bs32(sycl::sub_group const& sg,
                            uint32_t bs[4],
                            uint32_t const k1_bs[4],
                            uint32_t const k2_bs[4],
                            int rounds)
{
    #pragma unroll 2
    for (int r = 0; r < rounds; ++r) {
        aesenc_round_bs32(sg, bs, k1_bs);
        aesenc_round_bs32(sg, bs, k2_bs);
    }
}

// ---------- high-level wrappers matching AesHashGpu.cuh ----------
//
// Each wrapper must be called uniformly across the sub_group. The return
// value is per-lane (this lane's result); callers collect per-lane values
// into their own output buffers as usual.

// g_x_bs32 — bitsliced equivalent of g_x_smem(keys, x, k). Each lane
// contributes its own `x`, returns bottom k bits of state.w[0] for this
// lane's x.
inline uint32_t g_x_bs32(sycl::sub_group const& sg,
                         AesHashKeys const& keys, uint32_t x, int k,
                         int rounds = kAesGRounds)
{
    AesState in = set_int_vec_i128(0, 0, 0, static_cast<int32_t>(x));
    uint32_t bs[4], k1_bs[4], k2_bs[4];
    bs32_pack(sg, in, bs);
    make_bs32_round_key(sg, keys.round_key_1, k1_bs);
    make_bs32_round_key(sg, keys.round_key_2, k2_bs);
    run_rounds_bs32(sg, bs, k1_bs, k2_bs, rounds);
    AesState out;
    bs32_unpack(sg, bs, out);
    return out.w[0] & ((1u << k) - 1u);
}

// matching_target_bs32 — bitsliced equivalent of matching_target_smem.
// (table_id, match_key) are typically sub_group-uniform in the match
// kernels; only `meta` varies per lane. That's fine — bitslicing doesn't
// require per-lane inputs to differ.
inline uint32_t matching_target_bs32(sycl::sub_group const& sg,
                                     AesHashKeys const& keys,
                                     uint32_t table_id, uint32_t match_key,
                                     uint64_t meta,
                                     int extra_rounds_bits = 0)
{
    int32_t i0 = static_cast<int32_t>(table_id);
    int32_t i1 = static_cast<int32_t>(match_key);
    int32_t i2 = static_cast<int32_t>(meta & 0xFFFFFFFFu);
    int32_t i3 = static_cast<int32_t>((meta >> 32) & 0xFFFFFFFFu);
    AesState in = set_int_vec_i128(i3, i2, i1, i0);
    uint32_t bs[4], k1_bs[4], k2_bs[4];
    bs32_pack(sg, in, bs);
    make_bs32_round_key(sg, keys.round_key_1, k1_bs);
    make_bs32_round_key(sg, keys.round_key_2, k2_bs);
    int rounds = kAesMatchingTargetRounds << extra_rounds_bits;
    run_rounds_bs32(sg, bs, k1_bs, k2_bs, rounds);
    AesState out;
    bs32_unpack(sg, bs, out);
    return out.w[0];
}

// pairing_bs32 — bitsliced equivalent of pairing_smem. Kept for
// completeness / future use; the current match kernels keep the inner
// loop on T-table pairing because the inner trip count is data-dependent
// (per-lane window size varies), which is awkward to bit-slice without
// a batch-collect prepass.
inline Result128 pairing_bs32(sycl::sub_group const& sg,
                              AesHashKeys const& keys,
                              uint64_t meta_l, uint64_t meta_r,
                              int extra_rounds_bits = 0)
{
    int32_t i0 = static_cast<int32_t>(meta_l & 0xFFFFFFFFu);
    int32_t i1 = static_cast<int32_t>((meta_l >> 32) & 0xFFFFFFFFu);
    int32_t i2 = static_cast<int32_t>(meta_r & 0xFFFFFFFFu);
    int32_t i3 = static_cast<int32_t>((meta_r >> 32) & 0xFFFFFFFFu);
    AesState in = set_int_vec_i128(i3, i2, i1, i0);
    uint32_t bs[4], k1_bs[4], k2_bs[4];
    bs32_pack(sg, in, bs);
    make_bs32_round_key(sg, keys.round_key_1, k1_bs);
    make_bs32_round_key(sg, keys.round_key_2, k2_bs);
    int rounds = kAesPairingRounds << extra_rounds_bits;
    run_rounds_bs32(sg, bs, k1_bs, k2_bs, rounds);
    AesState out;
    bs32_unpack(sg, bs, out);
    Result128 r{};
    r.r[0] = out.w[0]; r.r[1] = out.w[1];
    r.r[2] = out.w[2]; r.r[3] = out.w[3];
    return r;
}

} // namespace pos2gpu
