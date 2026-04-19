// AesSBoxBP.cuh — Boyar-Peralta AES S-box circuit.
//
// Reference: Joan Boyar and René Peralta, "A small depth-16 circuit for the
// AES S-box" (ACISP 2012). 113 XOR/AND/NOT gates, depth 16.
//
// The circuit is parameterised on a bit-type T:
//   - T = uint8_t with NOT-mask N = 1   : per-byte scalar S-box, useful for
//                                         per-byte parity testing against the
//                                         Rijndael S-box table.
//   - T = uint32_t with N = 0xFFFFFFFFu : 32-way bit-sliced; each argument is
//                                         a bit-plane holding bit k across 32
//                                         parallel AES states. One evaluation
//                                         computes 32 S-boxes in parallel.
//
// Bit ordering (matches the paper): U0 is the MSB of the input byte, U7 the
// LSB. Similarly S0 is the MSB of the output byte.
//
// Validated on device for every x in [0, 256) against the Rijndael S-box
// populated in AesGpuBitsliced.cu; see tools/parity/aes_bs_parity.

#pragma once

#include <cstdint>

namespace pos2gpu {

template <typename T>
__host__ __device__ __forceinline__
void bp_sbox_circuit(T U0, T U1, T U2, T U3, T U4, T U5, T U6, T U7,
                     T& S0, T& S1, T& S2, T& S3,
                     T& S4, T& S5, T& S6, T& S7,
                     T N)
{
    // ---- Top linear transform (23 XORs) ----
    T y14 = U3 ^ U5;
    T y13 = U0 ^ U6;
    T y9  = U0 ^ U3;
    T y8  = U0 ^ U5;
    T t0  = U1 ^ U2;
    T y1  = t0 ^ U7;
    T y4  = y1 ^ U3;
    T y12 = y13 ^ y14;
    T y2  = y1 ^ U0;
    T y5  = y1 ^ U6;
    T y3  = y5 ^ y8;
    T t1  = U4 ^ y12;
    T y15 = t1 ^ U5;
    T y20 = t1 ^ U1;
    T y6  = y15 ^ U7;
    T y10 = y15 ^ t0;
    T y11 = y20 ^ y9;
    T y7  = U7 ^ y11;
    T y17 = y10 ^ y11;
    T y19 = y10 ^ y8;
    T y16 = t0 ^ y11;
    T y21 = y13 ^ y16;
    T y18 = U0 ^ y16;

    // ---- Middle non-linear (32 ANDs + XORs) ----
    T t2  = y12 & y15;
    T t3  = y3  & y6;
    T t4  = t3  ^ t2;
    T t5  = y4  & U7;
    T t6  = t5  ^ t2;
    T t7  = y13 & y16;
    T t8  = y5  & y1;
    T t9  = t8  ^ t7;
    T t10 = y2  & y7;
    T t11 = t10 ^ t7;
    T t12 = y9  & y11;
    T t13 = y14 & y17;
    T t14 = t13 ^ t12;
    T t15 = y8  & y10;
    T t16 = t15 ^ t12;
    T t17 = t4  ^ t14;
    T t18 = t6  ^ t16;
    T t19 = t9  ^ t14;
    T t20 = t11 ^ t16;
    T t21 = t17 ^ y20;
    T t22 = t18 ^ y19;
    T t23 = t19 ^ y21;
    T t24 = t20 ^ y18;

    T t25 = t21 ^ t22;
    T t26 = t21 & t23;
    T t27 = t24 ^ t26;
    T t28 = t25 & t27;
    T t29 = t28 ^ t22;
    T t30 = t23 ^ t24;
    T t31 = t22 ^ t26;
    T t32 = t31 & t30;
    T t33 = t32 ^ t24;
    T t34 = t23 ^ t33;
    T t35 = t27 ^ t33;
    T t36 = t24 & t35;
    T t37 = t36 ^ t34;
    T t38 = t27 ^ t36;
    T t39 = t29 & t38;
    T t40 = t25 ^ t39;

    T t41 = t40 ^ t37;
    T t42 = t29 ^ t33;
    T t43 = t29 ^ t40;
    T t44 = t33 ^ t37;
    T t45 = t42 ^ t41;

    T z0  = t44 & y15;
    T z1  = t37 & y6;
    T z2  = t33 & U7;
    T z3  = t43 & y16;
    T z4  = t40 & y1;
    T z5  = t29 & y7;
    T z6  = t42 & y11;
    T z7  = t45 & y17;
    T z8  = t41 & y10;
    T z9  = t44 & y12;
    T z10 = t37 & y3;
    T z11 = t33 & y4;
    T z12 = t43 & y13;
    T z13 = t40 & y5;
    T z14 = t29 & y2;
    T z15 = t42 & y9;
    T z16 = t45 & y14;
    T z17 = t41 & y8;

    // ---- Bottom linear transform (30 XORs + 4 NOTs folded into affine) ----
    T tc1  = z15 ^ z16;
    T tc2  = z10 ^ tc1;
    T tc3  = z9  ^ tc2;
    T tc4  = z0  ^ z2;
    T tc5  = z1  ^ z0;
    T tc6  = z3  ^ z4;
    T tc7  = z12 ^ tc4;
    T tc8  = z7  ^ tc6;
    T tc9  = z8  ^ tc7;
    T tc10 = tc8 ^ tc9;
    T tc11 = tc6 ^ tc5;
    T tc12 = z3  ^ z5;
    T tc13 = z13 ^ tc1;
    T tc14 = tc4 ^ tc12;
    S3     = tc3 ^ tc11;
    T tc16 = z6  ^ tc8;
    T tc17 = z14 ^ tc10;
    T tc18 = tc13 ^ tc14;
    S7     = z12 ^ tc18 ^ N;
    T tc20 = z15 ^ tc16;
    T tc21 = tc2 ^ z11;
    S0     = tc3 ^ tc16;
    S6     = tc10 ^ tc18 ^ N;
    S4     = tc14 ^ S3;
    S1     = S3 ^ tc16 ^ N;
    T tc26 = tc17 ^ tc20;
    S2     = tc26 ^ z17 ^ N;
    S5     = tc21 ^ tc17;
}

__host__ __device__ __forceinline__
uint8_t bp_sbox(uint8_t x)
{
    uint8_t U0 = uint8_t((x >> 7) & 1u);
    uint8_t U1 = uint8_t((x >> 6) & 1u);
    uint8_t U2 = uint8_t((x >> 5) & 1u);
    uint8_t U3 = uint8_t((x >> 4) & 1u);
    uint8_t U4 = uint8_t((x >> 3) & 1u);
    uint8_t U5 = uint8_t((x >> 2) & 1u);
    uint8_t U6 = uint8_t((x >> 1) & 1u);
    uint8_t U7 = uint8_t((x >> 0) & 1u);
    uint8_t S0, S1, S2, S3, S4, S5, S6, S7;
    bp_sbox_circuit<uint8_t>(U0, U1, U2, U3, U4, U5, U6, U7,
                              S0, S1, S2, S3, S4, S5, S6, S7,
                              uint8_t(1u));
    return uint8_t((S0 << 7) | (S1 << 6) | (S2 << 5) | (S3 << 4)
                 | (S4 << 3) | (S5 << 2) | (S6 << 1) | (S7 << 0));
}

} // namespace pos2gpu
