// AesGpuBitsliced.cuh — bit-sliced-ready AES round on the device.
//
// This is the day-1 scaffold for the BS-AES rewrite. The round is
// decomposed into explicit ShiftRows → SubBytes → MixColumns → AddRoundKey
// with no T-tables. On this first pass SubBytes is still an S-box lookup
// from constant memory (byte-addressable, not bit-logic) — the point of
// day 1 is to prove the non-T-table round structure is bit-exact vs the
// existing T-table form. Day 2 swaps the S-box for the Boyar-Peralta
// 113-gate bitwise network and introduces warp-parallel 32-way batching.
//
// Invariant: aesenc_round_bs(s, k) must equal aesenc_round(s, k) for all s,k.
// Verified by tools/parity/aes_bs_parity.

#pragma once

#include "gpu/AesGpu.cuh"
#include "gpu/AesSBoxBP.cuh"
#include <cstdint>

namespace pos2gpu {

// Rijndael S-box in constant memory. Populated by initialize_aes_bs_tables().
// Separate from AesGpu.cu's internal kSBox (which is anonymous-namespace host
// state) so that device code can index it directly.
__device__ __constant__ extern uint8_t kAesSBox[256];

void initialize_aes_bs_tables();

// =========================================================================
// 32-way warp-cooperative bit-sliced layout.
//
// A warp of 32 threads holds 32 AES states simultaneously. Bit position p
// of state s (0..127, LSB-first within each byte) is stored at bit `s` of
// bit-plane `p`. Each thread owns 4 contiguous bit-planes:
//
//   thread t owns bit-planes { 4t, 4t+1, 4t+2, 4t+3 } for t in [0, 32).
//
// The contiguous mapping clusters all 8 bit-planes of byte b into threads
// 2b and 2b+1. S-box operations (per byte) thus touch at most two lanes.
//
// Pack/unpack transpose 32 scalar AesStates (one per lane) to/from the
// bit-sliced form. Each is O(128) warp ops but amortizes across 32×32 =
// 1024 state bits, so ≈ 2.5 cycles per state-bit.
// =========================================================================

__device__ __forceinline__ void bs32_pack(AesState const& my, uint32_t out[4])
{
    int const lane = threadIdx.x & 31;
    for (int p = 0; p < 128; ++p) {
        int byte_idx    = p >> 3;
        int bit_in_byte = p & 7;
        int word_idx    = byte_idx >> 2;
        int byte_in_w   = byte_idx & 3;
        uint32_t bit    = (my.w[word_idx] >> (8 * byte_in_w + bit_in_byte)) & 1u;
        uint32_t plane  = __ballot_sync(0xFFFFFFFFu, bit != 0u);
        if (lane == (p >> 2)) {
            out[p & 3] = plane;
        }
    }
}

__device__ __forceinline__ void bs32_unpack(uint32_t const in[4], AesState& my)
{
    int const lane = threadIdx.x & 31;
    my.w[0] = my.w[1] = my.w[2] = my.w[3] = 0u;
    for (int p = 0; p < 128; ++p) {
        int owner      = p >> 2;
        int slot       = p & 3;
        uint32_t plane = __shfl_sync(0xFFFFFFFFu, in[slot], owner);
        uint32_t bit   = (plane >> lane) & 1u;
        int byte_idx    = p >> 3;
        int bit_in_byte = p & 7;
        int word_idx    = byte_idx >> 2;
        int byte_in_w   = byte_idx & 3;
        my.w[word_idx] |= bit << (8 * byte_in_w + bit_in_byte);
    }
}

// =========================================================================
// make_bs32_round_key — materialise a bit-sliced round key from a scalar
// AesState. All 32 states share the same key, so each bit-plane is either
// 0xFFFFFFFF (key bit = 1) or 0x00000000 (key bit = 0).
// =========================================================================

__device__ __forceinline__ void make_bs32_round_key(AesState const& key,
                                                     uint32_t key_bs[4])
{
    int const lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int p           = 4 * lane + i;
        int byte_idx    = p >> 3;
        int bit_in_byte = p & 7;
        int word_idx    = byte_idx >> 2;
        int byte_in_w   = byte_idx & 3;
        uint32_t bit    = (key.w[word_idx] >> (8 * byte_in_w + bit_in_byte)) & 1u;
        key_bs[i] = bit ? 0xFFFFFFFFu : 0u;
    }
}

__device__ __forceinline__ void add_round_key_bs32(uint32_t bs[4],
                                                    uint32_t const key_bs[4])
{
    bs[0] ^= key_bs[0];
    bs[1] ^= key_bs[1];
    bs[2] ^= key_bs[2];
    bs[3] ^= key_bs[3];
}

// =========================================================================
// shift_rows_bs32 — AES ShiftRows via warp shuffle.
//
// ShiftRows permutes bytes: new_byte[c*4+r] = old_byte[((c+r) & 3)*4 + r].
// Because the permutation preserves the bit-within-byte index, each thread
// only needs to fetch its own slot `i` from a single source lane, so the
// whole operation is exactly four __shfl_sync per thread.
// =========================================================================

__device__ __forceinline__ void shift_rows_bs32(uint32_t bs[4])
{
    int const lane    = threadIdx.x & 31;
    int const is_hi   = lane & 1;       // 0 = low half, 1 = high half
    int const b       = lane >> 1;       // output byte index 0..15
    int const c       = b >> 2;          // column
    int const r       = b & 3;           // row
    int const b_old   = ((c + r) & 3) * 4 + r;  // source byte
    int const owner   = 2 * b_old + is_hi;      // source lane

    uint32_t n0 = __shfl_sync(0xFFFFFFFFu, bs[0], owner);
    uint32_t n1 = __shfl_sync(0xFFFFFFFFu, bs[1], owner);
    uint32_t n2 = __shfl_sync(0xFFFFFFFFu, bs[2], owner);
    uint32_t n3 = __shfl_sync(0xFFFFFFFFu, bs[3], owner);
    bs[0] = n0;  bs[1] = n1;  bs[2] = n2;  bs[3] = n3;
}

// =========================================================================
// mix_columns_bs32 — AES MixColumns on 32 parallel states.
//
// Per column of 4 bytes a0..a3 (bytes c*4+0 .. c*4+3, threads 8c..8c+7):
//   out[r] = xtime(a_r ^ a_{r+1}) ^ a_{r+1} ^ a_{r+2} ^ a_{r+3}
// (derived from the MixColumns matrix, using that xtime is linear over GF(2)
// so xtime(a) ^ xtime(b) = xtime(a ^ b)).
//
// Each thread's shuffle footprint:
//   - 12 same-half column-mate shuffles (a_{r+1}, a_{r+2}, a_{r+3} × 4 slots)
//   -  2 cross-half shuffles for the single "boundary bit" of t = a_r ^ a_{r+1}
//      that xtime needs from the other half (t[7] for low-half, t[3] for high).
// Total: 14 warp shuffles per thread. Output is written in-place into bs[].
// =========================================================================

__device__ __forceinline__ void mix_columns_bs32(uint32_t bs[4])
{
    uint32_t const mask = 0xFFFFFFFFu;
    int const lane     = threadIdx.x & 31;
    int const is_hi    = lane & 1;
    int const b        = lane >> 1;
    int const c        = b >> 2;
    int const r        = b & 3;
    int const partner  = lane ^ 1;
    int const col_base = 8 * c;

    int const r1 = (r + 1) & 3;
    int const r2 = (r + 2) & 3;
    int const r3 = (r + 3) & 3;

    int const L1 = col_base + 2 * r1 + is_hi;      // same-half row-(r+1)
    int const L2 = col_base + 2 * r2 + is_hi;      // same-half row-(r+2)
    int const L3 = col_base + 2 * r3 + is_hi;      // same-half row-(r+3)
    int const L1_other = col_base + 2 * r1 + (is_hi ^ 1);

    uint32_t r1_0 = __shfl_sync(mask, bs[0], L1);
    uint32_t r1_1 = __shfl_sync(mask, bs[1], L1);
    uint32_t r1_2 = __shfl_sync(mask, bs[2], L1);
    uint32_t r1_3 = __shfl_sync(mask, bs[3], L1);
    uint32_t r2_0 = __shfl_sync(mask, bs[0], L2);
    uint32_t r2_1 = __shfl_sync(mask, bs[1], L2);
    uint32_t r2_2 = __shfl_sync(mask, bs[2], L2);
    uint32_t r2_3 = __shfl_sync(mask, bs[3], L2);
    uint32_t r3_0 = __shfl_sync(mask, bs[0], L3);
    uint32_t r3_1 = __shfl_sync(mask, bs[1], L3);
    uint32_t r3_2 = __shfl_sync(mask, bs[2], L3);
    uint32_t r3_3 = __shfl_sync(mask, bs[3], L3);

    // t = a_r ^ a_{r+1}. My half of t's bits is bs[i] ^ r1_i.
    uint32_t t_0 = bs[0] ^ r1_0;
    uint32_t t_1 = bs[1] ^ r1_1;
    uint32_t t_2 = bs[2] ^ r1_2;
    uint32_t t_3 = bs[3] ^ r1_3;

    // Boundary bit: the single bit of t from the *other* half xtime needs.
    // For is_hi=0, that's t[7]; for is_hi=1, that's t[3]. Both live in bs[3]
    // of the partner thread (of a_r) and of L1_other (of a_{r+1}).
    uint32_t t_boundary = __shfl_sync(mask, bs[3], partner)
                        ^ __shfl_sync(mask, bs[3], L1_other);

    // xtime(t) bit-planes for my half.
    //   Low half (bits 0..3):
    //     xt[0]=t[7], xt[1]=t[0]^t[7], xt[2]=t[1], xt[3]=t[2]^t[7]
    //   High half (bits 4..7):
    //     xt[4]=t[3]^t[7], xt[5]=t[4], xt[6]=t[5], xt[7]=t[6]
    //     (here t[3] = t_boundary, t[4..7] = t_0..t_3 in my registers;
    //      t[7] = t_3 — so xt[4] = t_boundary ^ t_3)
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

    // out = xt ^ a_{r+1} ^ a_{r+2} ^ a_{r+3}
    bs[0] = xt_0 ^ r1_0 ^ r2_0 ^ r3_0;
    bs[1] = xt_1 ^ r1_1 ^ r2_1 ^ r3_1;
    bs[2] = xt_2 ^ r1_2 ^ r2_2 ^ r3_2;
    bs[3] = xt_3 ^ r1_3 ^ r2_3 ^ r3_3;
}

// =========================================================================
// sub_bytes_bs32 — 32-way warp-cooperative SubBytes via Boyar-Peralta.
//
// Threads 2b and 2b+1 cooperate on byte b of the state:
//   - thread 2b   owns planes { 8b+0, 8b+1, 8b+2, 8b+3 } (bits 0..3 of byte b)
//   - thread 2b+1 owns planes { 8b+4, 8b+5, 8b+6, 8b+7 } (bits 4..7 of byte b)
// Each thread shuffles its partner's four planes in, assembles the full 8-bit
// input in paper convention (U0 = MSB = plane 8b+7), runs the BP circuit
// redundantly across the pair, then keeps only the four outputs for its own
// half. One __shfl_sync per partner plane (4 in + 0 out — outputs stay local).
// =========================================================================

__device__ __forceinline__ void sub_bytes_bs32(uint32_t bs[4])
{
    int const lane    = threadIdx.x & 31;
    int const is_hi   = lane & 1;
    int const partner = lane ^ 1;

    uint32_t peer0 = __shfl_sync(0xFFFFFFFFu, bs[0], partner);
    uint32_t peer1 = __shfl_sync(0xFFFFFFFFu, bs[1], partner);
    uint32_t peer2 = __shfl_sync(0xFFFFFFFFu, bs[2], partner);
    uint32_t peer3 = __shfl_sync(0xFFFFFFFFu, bs[3], partner);

    // Paper convention: U0 = MSB of byte (bit 7), U7 = LSB (bit 0).
    // My own planes hold bits {4,5,6,7} if high half, else {0,1,2,3}, LSB at
    // bs[0] and MSB-of-half at bs[3]. So plane p within my half maps to:
    //   high half: bs[0]=bit4, bs[1]=bit5, bs[2]=bit6, bs[3]=bit7
    //   low  half: bs[0]=bit0, bs[1]=bit1, bs[2]=bit2, bs[3]=bit3
    uint32_t U0, U1, U2, U3, U4, U5, U6, U7;
    if (is_hi) {
        U0 = bs[3];   U1 = bs[2];   U2 = bs[1];   U3 = bs[0];
        U4 = peer3;   U5 = peer2;   U6 = peer1;   U7 = peer0;
    } else {
        U0 = peer3;   U1 = peer2;   U2 = peer1;   U3 = peer0;
        U4 = bs[3];   U5 = bs[2];   U6 = bs[1];   U7 = bs[0];
    }

    uint32_t S0, S1, S2, S3, S4, S5, S6, S7;
    bp_sbox_circuit<uint32_t>(U0, U1, U2, U3, U4, U5, U6, U7,
                               S0, S1, S2, S3, S4, S5, S6, S7,
                               0xFFFFFFFFu);

    // Keep the half of the output that lives on this thread.
    if (is_hi) {
        // I own bits 4..7; S0=bit7, S1=bit6, S2=bit5, S3=bit4
        bs[3] = S0;   bs[2] = S1;   bs[1] = S2;   bs[0] = S3;
    } else {
        // I own bits 0..3; S4=bit3, S5=bit2, S6=bit1, S7=bit0
        bs[3] = S4;   bs[2] = S5;   bs[1] = S6;   bs[0] = S7;
    }
}

// =========================================================================
// aesenc_round_bs32_warp — one AES round on 32 parallel states.
//   ShiftRows → SubBytes → MixColumns → AddRoundKey
// Matches the semantics of aesenc_round / aesenc_round_bs when run on each
// of the 32 states independently. All sub-primitives are __forceinline__ so
// this is a single fused kernel body.
// =========================================================================

__device__ __forceinline__
void aesenc_round_bs32_warp(uint32_t bs[4], uint32_t const key_bs[4])
{
    shift_rows_bs32(bs);
    sub_bytes_bs32(bs);
    mix_columns_bs32(bs);
    add_round_key_bs32(bs, key_bs);
}

// Alternating-key round loop, mirror of run_rounds_smem / run_rounds.
// `rounds` is the CPU-side round-pair count (16 for AES-128 in pos2-chip),
// producing `2 * rounds` AES rounds total.
__device__ __forceinline__
void run_rounds_bs32_warp(uint32_t bs[4],
                           uint32_t const k1_bs[4],
                           uint32_t const k2_bs[4],
                           int rounds)
{
    #pragma unroll 2
    for (int r = 0; r < rounds; ++r) {
        aesenc_round_bs32_warp(bs, k1_bs);
        aesenc_round_bs32_warp(bs, k2_bs);
    }
}

// =========================================================================
// Reference per-thread AES round (scalar S-box lookup, explicit mix/shift).
// Day-1 scaffold; to be replaced by warp-parallel BP round.
// =========================================================================

__device__ __forceinline__ AesState aesenc_round_bs(AesState s, AesState const& key)
{
    auto byte = [] (uint32_t w, int n) -> uint8_t {
        return static_cast<uint8_t>((w >> (8 * n)) & 0xFFu);
    };
    auto xtime = [] (uint8_t x) -> uint8_t {
        return static_cast<uint8_t>((x << 1) ^ ((x & 0x80) ? 0x1B : 0));
    };

    // ShiftRows: new_byte[c*4 + r] = old_byte[((c + r) & 3) * 4 + r].
    // Fold the SubBytes lookup in the same pass.
    uint8_t sb[16];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            uint8_t raw = byte(s.w[(c + r) & 3], r);
            sb[c * 4 + r] = kAesSBox[raw];
        }
    }

    // MixColumns per column, XOR with round key.
    AesState out;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = sb[c * 4 + 0];
        uint8_t a1 = sb[c * 4 + 1];
        uint8_t a2 = sb[c * 4 + 2];
        uint8_t a3 = sb[c * 4 + 3];
        uint8_t t0 = uint8_t(xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3);
        uint8_t t1 = uint8_t(a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3);
        uint8_t t2 = uint8_t(a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3);
        uint8_t t3 = uint8_t(xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3));
        uint32_t col = uint32_t(t0)
                     | (uint32_t(t1) << 8)
                     | (uint32_t(t2) << 16)
                     | (uint32_t(t3) << 24);
        out.w[c] = col ^ key.w[c];
    }
    return out;
}

} // namespace pos2gpu
