// pos2_keygen.h — C interface to the pos2_keygen Rust staticlib.
//
// Exposes a single function that derives the plot_id and memo for a v2
// Chia plot from caller-supplied farmer + pool keys. Wraps chia-rs
// (chia-bls + chia-protocol) so the output is byte-equivalent to what
// `chia plots create --v2` produces.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return codes
#define POS2_OK                  0
#define POS2_BAD_FARMER_PK      -1
#define POS2_BAD_POOL_KEY       -2
#define POS2_BAD_POOL_KIND      -3
#define POS2_MEMO_BUF_TOO_SMALL -4
#define POS2_BAD_SEED           -5
#define POS2_BAD_ADDRESS        -6
#define POS2_BAD_HRP            -7

// pool_kind values
#define POS2_POOL_PK  0  // pool_key_ptr points to 48 bytes (G1 public key)
#define POS2_POOL_PH  1  // pool_key_ptr points to 32 bytes (puzzle hash)

// Derive a v2 plot's plot_id + memo from caller-supplied keys and seed.
//
// Inputs:
//   seed_ptr, seed_len         : >= 32 bytes of entropy for the master SK.
//   farmer_pk_ptr              : 48 bytes (G1 compressed).
//   pool_key_ptr + pool_kind   : 48B pool PK (POS2_POOL_PK) or 32B pool
//                                contract puzzle hash (POS2_POOL_PH).
//                                PH mode includes the taproot term in
//                                plot_public_key.
//   strength, plot_index, meta_group : v2 proof-of-space parameters.
//
// Outputs:
//   out_plot_id                : 32 bytes.
//   out_memo_buf, inout_memo_len:
//       caller provides buffer + its capacity via *inout_memo_len;
//       on POS2_OK the function writes `pool_key_or_ph || farmer_pk ||
//       master_sk_bytes` (112 B in PH mode, 128 B in PK mode) and updates
//       *inout_memo_len to the bytes written. Returns
//       POS2_MEMO_BUF_TOO_SMALL (and writes the required size to
//       *inout_memo_len) if the buffer is too small.
int pos2_keygen_derive_plot(
    const uint8_t* seed_ptr, size_t seed_len,
    const uint8_t* farmer_pk_ptr,
    const uint8_t* pool_key_ptr, int pool_kind,
    uint8_t strength, uint16_t plot_index, uint8_t meta_group,
    uint8_t* out_plot_id,
    uint8_t* out_memo_buf, size_t* inout_memo_len);

// Decode a Chia bech32m address ("xch1..." mainnet or "txch1..." testnet)
// into a 32-byte puzzle hash. `address` must be a NUL-terminated C string.
// Returns POS2_OK / POS2_BAD_ADDRESS / POS2_BAD_HRP.
int pos2_keygen_decode_address(
    const char* address,
    uint8_t out_puzzle_hash[32]);

// Derive a deterministic per-plot seed from a caller-supplied 32-byte base
// seed and a plot index, as SHA256(base_seed || idx_le_u64). Lets `--seed`
// drive reproducible multi-plot batches.
int pos2_keygen_derive_subseed(
    const uint8_t base_seed[32],
    uint64_t idx,
    uint8_t out_seed[32]);

#ifdef __cplusplus
} // extern "C"
#endif
