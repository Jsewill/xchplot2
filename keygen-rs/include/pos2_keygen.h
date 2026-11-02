// pos2_keygen.h — C interface to the pos2_keygen Rust staticlib.
//
// Derives the plot_id, shared group identity, and memo for a v2
// Chia plot from caller-supplied farmer + pool keys. Wraps chia-rs
// (chia-bls + chia-protocol), using PoS2 1.0 identities and the V2 taproot
// derivation in chia-blockchain PR #21484.

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
#define POS2_BAD_HASH_INPUT     -8

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
//                                Both pool modes include the V2 taproot term.
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

// Shared group identity and memo using the current key/taproot derivation.
// Same inputs/outputs as derive_plot, without per-member index/meta_group.
int pos2_keygen_derive_group(
    const uint8_t* seed_ptr, size_t seed_len,
    const uint8_t* farmer_pk_ptr,
    const uint8_t* pool_key_ptr, int pool_kind,
    uint8_t strength, uint8_t out_group_id[32],
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

// General SHA-256 helper, independent of key derivation. Hashes the exact
// concatenation of caller-owned byte segments in array order, without adding
// lengths, separators, or copying the input into a combined buffer.
typedef struct pos2_sha256_segment {
    const uint8_t* data;
    size_t len;
} pos2_sha256_segment;

// segments may be NULL only when segment_count is zero. A segment's data may
// be NULL only when its len is zero. All nonempty regions must be readable and
// the descriptor array properly aligned; out_digest points to 32 writable bytes.
// Returns POS2_OK or POS2_BAD_HASH_INPUT for null pointers, invalid alignment,
// lengths beyond the addressable slice range or SHA-256 message-length limit.
// On an error, out_digest is unchanged. Input memory remains caller-owned.
int pos2_sha256_segments(
    const pos2_sha256_segment* segments, size_t segment_count,
    uint8_t out_digest[32]);

// Streaming equivalent for bounded, reusable input buffers. The reader returns
// 1 with the next segment, 0 at EOF, or a negative value on failure. Other return
// values are invalid. Each segment is consumed before the next reader call, so
// its storage need only remain readable and unchanged until that call. The
// reader must not throw or unwind across this C boundary. context is passed
// through unchanged and may be NULL if the reader supports it.
typedef int (*pos2_sha256_reader)(void* context, pos2_sha256_segment* next);

// reader and out_digest must be non-NULL. Segment lengths have the same limits
// as pos2_sha256_segments. Returns POS2_OK or POS2_BAD_HASH_INPUT; the output is
// unchanged on invalid metadata or reader failure, including after partial input.
int pos2_sha256_stream(
    pos2_sha256_reader reader, void* context, uint8_t out_digest[32]);

// Process-local incremental SHA state for consumers that hash while producing
// their input. Handles own their allocation and must be freed exactly once;
// clones are independent, and digest does not consume or modify its state.
// Handles must originate from new/clone and may not be used after free. These
// native states are not a serialization format and must not be written to disk.
typedef struct pos2_sha256_state pos2_sha256_state;

// Return NULL on allocation failure; clone(NULL) also returns NULL. Each state
// allocation has a conservative bound reported by state_bytes(), including an
// allocator allowance. Callers also account for clone/digest temporary storage.
pos2_sha256_state* pos2_sha256_state_new(void);
pos2_sha256_state* pos2_sha256_state_clone(const pos2_sha256_state* state);
size_t pos2_sha256_state_bytes(void);
// free(NULL) is harmless. No other operation may overlap free on the same state.
void pos2_sha256_state_free(pos2_sha256_state* state);

// Returns POS2_OK or POS2_BAD_HASH_INPUT. NULL state, NULL nonempty data,
// unaddressable lengths and cumulative SHA message-length overflow are rejected
// without modifying state. Empty data may be NULL. Input remains caller-owned
// and is consumed synchronously before return. Input/output buffers must not
// overlap the opaque state allocation. State access must be serialized.
int pos2_sha256_state_update(pos2_sha256_state* state, const uint8_t* data, size_t len);
// NULL state/output is rejected; on failure the output is unchanged. On success
// out_digest points to 32 writable bytes, and the state remains reusable.
int pos2_sha256_state_digest(const pos2_sha256_state* state, uint8_t out_digest[32]);

#ifdef __cplusplus
} // extern "C"
#endif
