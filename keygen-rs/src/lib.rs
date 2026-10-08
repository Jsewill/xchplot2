// pos2_keygen — C-callable shim around chia-bls
// that derives a v2 plot's plot_id and memo from caller-supplied farmer +
// pool keys plus a 32-byte master-SK seed. The GPU plotter uses the returned
// plot_id / memo to drive the existing batch path.
//
// chia-bls supplies BLS12-381 arithmetic and EIP-2333 HD derivation.
// This shim sequences PoS2 1.0 identity hashing, the V2 taproot contract in
// chia-blockchain PR #21484, and the existing memo layout.

use chia_bls::{PublicKey, SecretKey};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Result codes returned across the FFI boundary.
// ---------------------------------------------------------------------------
pub const POS2_OK: i32 = 0;
pub const POS2_BAD_FARMER_PK: i32 = -1;
pub const POS2_BAD_POOL_KEY: i32 = -2;
pub const POS2_BAD_POOL_KIND: i32 = -3;
pub const POS2_MEMO_BUF_TOO_SMALL: i32 = -4;
pub const POS2_BAD_SEED: i32 = -5;
pub const POS2_BAD_ADDRESS: i32 = -6;
pub const POS2_BAD_HRP: i32 = -7;

// pool_kind values.
pub const POS2_POOL_PK: i32 = 0; // pool_key_or_ph points to 48 bytes (G1)
pub const POS2_POOL_PH: i32 = 1; // pool_key_or_ph points to 32 bytes (hash)

fn master_sk_to_local_sk(master: &SecretKey) -> SecretKey {
    // Mirrors chia-blockchain's chia/wallet/derive_keys.py::master_sk_to_local_sk,
    // which is the hardened path m/12381/8444/3/0. chia-bls has hardened wallet
    // and pool helpers but not this specific "local" (plot-key) path, so
    // inline the four derive_hardened steps.
    master
        .derive_hardened(12381)
        .derive_hardened(8444)
        .derive_hardened(3)
        .derive_hardened(0)
}

fn generate_taproot_sk(local_pk: &PublicKey, farmer_pk: &PublicKey) -> SecretKey {
    // V2 taproot: SHA256((local_pk + farmer_pk) || farmer_pk).
    // chia-blockchain PR #21484 deliberately omits the standalone local key.
    let sum = local_pk + farmer_pk;
    let mut msg = Vec::with_capacity(48 * 2);
    msg.extend_from_slice(&sum.to_bytes());
    msg.extend_from_slice(&farmer_pk.to_bytes());

    let mut hasher = Sha256::new();
    hasher.update(&msg);
    let seed: [u8; 32] = hasher.finalize().into();
    SecretKey::from_seed(&seed)
}

fn generate_plot_public_key(local_pk: &PublicKey, farmer_pk: &PublicKey) -> PublicKey {
    // Every V2 plot includes taproot, including pool-public-key plots.
    let taproot_pk = generate_taproot_sk(local_pk, farmer_pk).public_key();
    local_pk + farmer_pk + &taproot_pk
}

// Chia's protocol crate also requires the CLVM runtime. Keep only the plot-ID
// encoding here and compare it with that crate in tests. Integers in Chia's
// streamable format are big-endian, unlike the subseed index below.
#[cfg(test)]
fn compute_plot_id_v2(
    strength: u8,
    plot_pk: &PublicKey,
    pool_key: &[u8],
    plot_index: u16,
    meta_group: u8,
) -> [u8; 32] {
    let mut id = Sha256::new();
    id.update(compute_plot_group_id_v2(strength, plot_pk, pool_key));
    id.update(plot_index.to_be_bytes());
    id.update([meta_group]);
    id.finalize().into()
}

fn compute_plot_group_id_v2(strength: u8, plot_pk: &PublicKey, pool_key: &[u8]) -> [u8; 32] {
    let mut group = Sha256::new();
    group.update([strength]);
    group.update(plot_pk.to_bytes());
    group.update(pool_key);
    group.finalize().into()
}

/// Derives a v2 plot's plot_id and memo from caller-supplied keys.
///
/// Inputs:
/// - `seed_ptr` / `seed_len`: the random entropy that becomes the master
///    secret key (>= 32 bytes). Caller is responsible for generating it
///    (e.g. `/dev/urandom` or RDRAND); we do not touch the system RNG so
///    the caller fully controls determinism.
/// - `farmer_pk_ptr`: 48-byte G1 (compressed) public key of the farmer.
/// - `pool_key_ptr` / `pool_kind`: either a 48-byte pool public key
///    (`pool_kind = POS2_POOL_PK`) or a 32-byte pool contract puzzle hash
///    (`pool_kind = POS2_POOL_PH`). Both forms use the V2 taproot term.
/// - `strength`, `plot_index`, `meta_group`: v2 proof-of-space parameters
///    that feed compute_plot_id_v2.
///
/// Outputs:
/// - `out_plot_id`: exactly 32 bytes of the computed v2 plot id.
/// - `out_memo_buf` / `*inout_memo_len`: memo bytes are
///    `pool_key_or_ph || farmer_pk || master_sk_bytes` — 112 bytes in
///    pool-PH mode, 128 bytes in pool-PK mode. Caller passes the buffer
///    size in `*inout_memo_len`; on success it's overwritten with the
///    bytes written. Returns `POS2_MEMO_BUF_TOO_SMALL` if the caller's
///    buffer is too small.
///
/// # Safety
/// All pointers must be non-null and point to readable/writable memory of
/// at least the sizes described above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_keygen_derive_plot(
    seed_ptr: *const u8,
    seed_len: usize,
    farmer_pk_ptr: *const u8, // 48 bytes
    pool_key_ptr: *const u8,  // 48 bytes (pool_pk) or 32 bytes (pool_ph)
    pool_kind: i32,
    strength: u8,
    plot_index: u16,
    meta_group: u8,
    out_plot_id: *mut u8,       // 32 bytes written
    out_memo_buf: *mut u8,      // caller-owned buffer
    inout_memo_len: *mut usize, // in: capacity; out: bytes written
) -> i32 {
    let mut group_id = [0u8; 32];
    let rc = unsafe {
        pos2_keygen_derive_group(
            seed_ptr,
            seed_len,
            farmer_pk_ptr,
            pool_key_ptr,
            pool_kind,
            strength,
            group_id.as_mut_ptr(),
            out_memo_buf,
            inout_memo_len,
        )
    };
    if rc != POS2_OK {
        return rc;
    }
    let mut id = Sha256::new();
    id.update(group_id);
    id.update(plot_index.to_be_bytes());
    id.update([meta_group]);
    unsafe {
        std::ptr::copy_nonoverlapping(id.finalize().as_ptr(), out_plot_id, 32);
    }
    POS2_OK
}

/// Derive shared keys, memo and group identity using the current protocol.
///
/// # Safety
/// Pointers have the same requirements as `pos2_keygen_derive_plot`;
/// `out_group_id` points to 32 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_keygen_derive_group(
    seed_ptr: *const u8,
    seed_len: usize,
    farmer_pk_ptr: *const u8,
    pool_key_ptr: *const u8,
    pool_kind: i32,
    strength: u8,
    out_group_id: *mut u8,
    out_memo_buf: *mut u8,
    inout_memo_len: *mut usize,
) -> i32 {
    if seed_len < 32 {
        return POS2_BAD_SEED;
    }
    let seed: &[u8] = unsafe { std::slice::from_raw_parts(seed_ptr, seed_len) };

    let farmer_pk_bytes: &[u8; 48] = match unsafe { (farmer_pk_ptr as *const [u8; 48]).as_ref() } {
        Some(b) => b,
        None => return POS2_BAD_FARMER_PK,
    };
    let farmer_pk = match PublicKey::from_bytes(farmer_pk_bytes) {
        Ok(pk) => pk,
        Err(_) => return POS2_BAD_FARMER_PK,
    };

    let pool_key_slice: &[u8] = match pool_kind {
        x if x == POS2_POOL_PK => {
            let bytes: &[u8; 48] = match unsafe { (pool_key_ptr as *const [u8; 48]).as_ref() } {
                Some(b) => b,
                None => return POS2_BAD_POOL_KEY,
            };
            if PublicKey::from_bytes(bytes).is_err() {
                return POS2_BAD_POOL_KEY;
            }
            &bytes[..]
        }
        x if x == POS2_POOL_PH => {
            let bytes: &[u8; 32] = match unsafe { (pool_key_ptr as *const [u8; 32]).as_ref() } {
                Some(b) => b,
                None => return POS2_BAD_POOL_KEY,
            };
            &bytes[..]
        }
        _ => return POS2_BAD_POOL_KIND,
    };

    let master_sk = SecretKey::from_seed(seed);
    let local_sk = master_sk_to_local_sk(&master_sk);
    let local_pk = local_sk.public_key();

    let plot_pk = generate_plot_public_key(&local_pk, &farmer_pk);

    let group_id = compute_plot_group_id_v2(strength, &plot_pk, pool_key_slice);

    let master_sk_bytes = master_sk.to_bytes();
    let memo_len = pool_key_slice.len() + 48 /* farmer_pk */ + master_sk_bytes.len();

    let capacity = unsafe { *inout_memo_len };
    if capacity < memo_len {
        unsafe { *inout_memo_len = memo_len };
        return POS2_MEMO_BUF_TOO_SMALL;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(group_id.as_ptr(), out_group_id, 32);
        let dst = out_memo_buf;
        std::ptr::copy_nonoverlapping(pool_key_slice.as_ptr(), dst, pool_key_slice.len());
        std::ptr::copy_nonoverlapping(farmer_pk_bytes.as_ptr(), dst.add(pool_key_slice.len()), 48);
        std::ptr::copy_nonoverlapping(
            master_sk_bytes.as_ptr(),
            dst.add(pool_key_slice.len() + 48),
            master_sk_bytes.len(),
        );
        *inout_memo_len = memo_len;
    }

    POS2_OK
}

/// Decode a Chia bech32m address ("xch1..." mainnet, "txch1..." testnet) into
/// the 32-byte puzzle hash bytes.
///
/// # Safety
/// `address_ptr` must be a NUL-terminated C string. `out_puzzle_hash` must
/// point to at least 32 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_keygen_decode_address(
    address_ptr: *const core::ffi::c_char,
    out_puzzle_hash: *mut u8,
) -> i32 {
    if address_ptr.is_null() || out_puzzle_hash.is_null() {
        return POS2_BAD_ADDRESS;
    }
    let cstr = unsafe { core::ffi::CStr::from_ptr(address_ptr) };
    let s = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return POS2_BAD_ADDRESS,
    };

    let decoded = match bech32::primitives::decode::CheckedHrpstring::new::<bech32::Bech32m>(s) {
        Ok(x) => x,
        Err(_) => return POS2_BAD_ADDRESS,
    };
    let hrp = decoded.hrp();
    let h = hrp.as_str();
    if h != "xch" && h != "txch" {
        return POS2_BAD_HRP;
    }
    let data: Vec<u8> = decoded.byte_iter().collect();
    if data.len() != 32 {
        return POS2_BAD_ADDRESS;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), out_puzzle_hash, 32);
    }
    POS2_OK
}

/// Deterministically derive a 32-byte subseed for plot `idx` from a caller's
/// base seed. Matches SHA256(base_seed || idx_little_endian_u64), so callers
/// who record a base seed can reproduce any plot in the batch. Keeps
/// per-plot master SKs independent even when the user supplied a fixed seed
/// for the whole run.
///
/// # Safety
/// Both pointers must be valid 32-byte regions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_keygen_derive_subseed(
    base_seed: *const u8, // 32 bytes
    idx: u64,
    out_seed: *mut u8, // 32 bytes
) -> i32 {
    if base_seed.is_null() || out_seed.is_null() {
        return POS2_BAD_SEED;
    }
    let base = unsafe { std::slice::from_raw_parts(base_seed, 32) };
    let mut h = Sha256::new();
    h.update(base);
    h.update(idx.to_le_bytes());
    let digest = h.finalize();
    unsafe {
        std::ptr::copy_nonoverlapping(digest.as_ptr(), out_seed, 32);
    }
    POS2_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_ids_match_chia_protocol() {
        let plot_pk = SecretKey::from_seed(&[0x11; 32]).public_key();
        let pool_pk = SecretKey::from_seed(&[0x22; 32]).public_key();
        let contract = chia_protocol::Bytes32::new([0x33; 32]);
        for strength in [1, 2, 32, 63] {
            for index in [0, 1, 255, 256, u16::MAX] {
                for group in [0, 1, u8::MAX] {
                    for pool in [false, true] {
                        let bytes = pool_pk.to_bytes();
                        let key: &[u8] = if pool { &bytes } else { contract.as_ref() };
                        let expected = chia_protocol::compute_plot_id_v2(
                            strength,
                            &plot_pk,
                            pool.then_some(&pool_pk),
                            (!pool).then_some(&contract),
                            index,
                            group,
                        );
                        assert_eq!(
                            compute_plot_id_v2(strength, &plot_pk, key, index, group),
                            expected.to_bytes()
                        );
                        assert_eq!(
                            compute_plot_group_id_v2(strength, &plot_pk, key),
                            chia_protocol::compute_plot_group_id_v2(
                                strength,
                                &plot_pk,
                                pool.then_some(&pool_pk),
                                (!pool).then_some(&contract),
                            )
                            .to_bytes()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn v2_taproot_matches_upstream_key_vector() {
        // chia-blockchain PR #21484, test_v2_plot_loading.py.
        let master = SecretKey::from_seed(&[b'1'; 32]);
        let farmer = SecretKey::from_seed(&[b'2'; 32]).public_key();
        let local = master_sk_to_local_sk(&master).public_key();
        assert_eq!(
            hex::encode(generate_plot_public_key(&local, &farmer).to_bytes()),
            "8144c8ef9f89911efdc979145ef674978e5158d6afa818942aa48b183d9a863a6e7535258b3e95e34087598e3476c798"
        );
    }

    #[test]
    fn group_ffi_shares_memo_and_derives_member_ids_for_both_pool_kinds() {
        let seed = [0xaa; 32];
        let farmer = SecretKey::from_seed(&[0xbb; 32]).public_key();
        let pool = SecretKey::from_seed(&[0xcc; 32]).public_key();
        let contract = [0xdd; 32];
        for kind in [POS2_POOL_PK, POS2_POOL_PH] {
            let pool_bytes = pool.to_bytes();
            let pool_key = if kind == POS2_POOL_PK {
                &pool_bytes[..]
            } else {
                &contract[..]
            };
            let mut group_id = [0; 32];
            let mut group_memo = [0; 128];
            let mut group_len = group_memo.len();
            assert_eq!(
                unsafe {
                    pos2_keygen_derive_group(
                        seed.as_ptr(),
                        seed.len(),
                        farmer.to_bytes().as_ptr(),
                        pool_key.as_ptr(),
                        kind,
                        2,
                        group_id.as_mut_ptr(),
                        group_memo.as_mut_ptr(),
                        &mut group_len,
                    )
                },
                POS2_OK
            );
            let master = SecretKey::from_seed(&seed);
            let plot_pk =
                generate_plot_public_key(&master_sk_to_local_sk(&master).public_key(), &farmer);
            assert_eq!(
                group_id,
                chia_protocol::compute_plot_group_id_v2(
                    2,
                    &plot_pk,
                    (kind == POS2_POOL_PK).then_some(&pool),
                    (kind == POS2_POOL_PH).then_some(&chia_protocol::Bytes32::new(contract))
                )
                .to_bytes()
            );
            for index in [0u16, 1, u16::MAX] {
                let mut id = [0; 32];
                let mut memo = [0; 128];
                let mut len = memo.len();
                assert_eq!(
                    unsafe {
                        pos2_keygen_derive_plot(
                            seed.as_ptr(),
                            seed.len(),
                            farmer.to_bytes().as_ptr(),
                            pool_key.as_ptr(),
                            kind,
                            2,
                            index,
                            7,
                            id.as_mut_ptr(),
                            memo.as_mut_ptr(),
                            &mut len,
                        )
                    },
                    POS2_OK
                );
                assert_eq!(len, group_len);
                assert_eq!(memo, group_memo);
                assert_eq!(id, compute_plot_id_v2(2, &plot_pk, pool_key, index, 7));
            }
        }
    }

    // Same inputs must produce identical plot_id + memo.
    #[test]
    fn deterministic_same_seed() {
        let seed = [0xAA_u8; 32];
        let farmer_pk = SecretKey::from_seed(&[0xBB_u8; 32]).public_key().to_bytes();
        let pool_ph = [0xCC_u8; 32];

        let mut pid1 = [0u8; 32];
        let mut memo1 = vec![0u8; 128];
        let mut mlen1: usize = memo1.len();
        let rc1 = unsafe {
            pos2_keygen_derive_plot(
                seed.as_ptr(),
                seed.len(),
                farmer_pk.as_ptr(),
                pool_ph.as_ptr(),
                POS2_POOL_PH,
                2,
                0,
                0,
                pid1.as_mut_ptr(),
                memo1.as_mut_ptr(),
                &mut mlen1,
            )
        };
        assert_eq!(rc1, POS2_OK);
        memo1.truncate(mlen1);

        let mut pid2 = [0u8; 32];
        let mut memo2 = vec![0u8; 128];
        let mut mlen2: usize = memo2.len();
        let rc2 = unsafe {
            pos2_keygen_derive_plot(
                seed.as_ptr(),
                seed.len(),
                farmer_pk.as_ptr(),
                pool_ph.as_ptr(),
                POS2_POOL_PH,
                2,
                0,
                0,
                pid2.as_mut_ptr(),
                memo2.as_mut_ptr(),
                &mut mlen2,
            )
        };
        assert_eq!(rc2, POS2_OK);
        memo2.truncate(mlen2);

        assert_eq!(pid1, pid2);
        assert_eq!(memo1, memo2);
        assert_eq!(memo1.len(), 32 + 48 + 32); // ph-mode memo
    }

    #[test]
    fn subseed_is_deterministic_and_index_dependent() {
        let base = [0x11_u8; 32];
        let mut a = [0u8; 32];
        let mut a2 = [0u8; 32];
        let mut b = [0u8; 32];
        let r1 = unsafe { pos2_keygen_derive_subseed(base.as_ptr(), 0, a.as_mut_ptr()) };
        let r2 = unsafe { pos2_keygen_derive_subseed(base.as_ptr(), 0, a2.as_mut_ptr()) };
        let r3 = unsafe { pos2_keygen_derive_subseed(base.as_ptr(), 1, b.as_mut_ptr()) };
        assert_eq!(r1, POS2_OK);
        assert_eq!(r2, POS2_OK);
        assert_eq!(r3, POS2_OK);
        assert_eq!(a, a2);
        assert_ne!(a, b);
    }

    #[test]
    fn decode_address_accepts_xch_and_txch() {
        // Produce an "xch1..." address programmatically (so we don't need to
        // depend on a specific well-known vector) and round-trip through
        // our decoder.
        use bech32::{Bech32m, Hrp};
        let ph = [0x42_u8; 32];
        let hrp = Hrp::parse("xch").unwrap();
        let addr = bech32::encode::<Bech32m>(hrp, &ph).unwrap();
        let c = std::ffi::CString::new(addr).unwrap();

        let mut out = [0u8; 32];
        let rc = unsafe { pos2_keygen_decode_address(c.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(rc, POS2_OK);
        assert_eq!(out, ph);

        // Rejects unknown HRP.
        let bad = std::ffi::CString::new("btc1qaaaaaaaaaaaa").unwrap();
        let mut dummy = [0u8; 32];
        let rc = unsafe { pos2_keygen_decode_address(bad.as_ptr(), dummy.as_mut_ptr()) };
        assert_ne!(rc, POS2_OK);
    }
    #[test]
    fn decode_address_rejects_legacy_checksum() {
        let address =
            bech32::encode::<bech32::Bech32>(bech32::Hrp::parse("xch").unwrap(), &[0x42; 32])
                .unwrap();
        let address = std::ffi::CString::new(address).unwrap();
        let mut output = [0x55; 32];
        assert_eq!(
            unsafe { pos2_keygen_decode_address(address.as_ptr(), output.as_mut_ptr()) },
            POS2_BAD_ADDRESS
        );
        assert_eq!(output, [0x55; 32]);
    }
}
