// pos2_keygen — C-callable shim around chia 0.42 (chia-bls + chia-protocol)
// that derives a v2 plot's plot_id and memo from caller-supplied farmer +
// pool keys plus a 32-byte master-SK seed. The GPU plotter uses the returned
// plot_id / memo to drive the existing batch path.
//
// The heavy lifting (BLS12-381 arithmetic, EIP-2333 HD derivation, Chia's
// taproot construction, compute_plot_id_v2 hashing) lives in chia-rs; this
// crate just sequences the calls and lays out the memo bytes the same way
// chia-blockchain's create_v2_plots does, so the resulting plots are
// byte-identical to `chia plots create --v2`.

use chia::bls::{PublicKey, SecretKey};
use chia::protocol::{Bytes32, compute_plot_id_v2};
use chia::sha2::Sha256;

// ---------------------------------------------------------------------------
// Result codes returned across the FFI boundary.
// ---------------------------------------------------------------------------
pub const POS2_OK: i32                 = 0;
pub const POS2_BAD_FARMER_PK: i32      = -1;
pub const POS2_BAD_POOL_KEY: i32       = -2;
pub const POS2_BAD_POOL_KIND: i32      = -3;
pub const POS2_MEMO_BUF_TOO_SMALL: i32 = -4;
pub const POS2_BAD_SEED: i32           = -5;

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
    // Same construction as chia-blockchain's _generate_taproot_sk: the seed is
    // std_hash( (local_pk + farmer_pk) || local_pk || farmer_pk ).
    let sum = local_pk + farmer_pk;
    let mut msg = Vec::with_capacity(48 * 3);
    msg.extend_from_slice(&sum.to_bytes());
    msg.extend_from_slice(&local_pk.to_bytes());
    msg.extend_from_slice(&farmer_pk.to_bytes());

    let mut hasher = Sha256::new();
    hasher.update(&msg);
    let seed: [u8; 32] = hasher.finalize().into();
    SecretKey::from_seed(&seed)
}

fn generate_plot_public_key(
    local_pk: &PublicKey,
    farmer_pk: &PublicKey,
    include_taproot: bool,
) -> PublicKey {
    if include_taproot {
        let taproot_sk = generate_taproot_sk(local_pk, farmer_pk);
        let taproot_pk = taproot_sk.public_key();
        local_pk + farmer_pk + &taproot_pk
    } else {
        local_pk + farmer_pk
    }
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
///    (`pool_kind = POS2_POOL_PH`). When a puzzle hash is used, the plot
///    public key is constructed with the taproot term — matching
///    chia-blockchain's include_taproot flag.
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
    out_plot_id: *mut u8,    // 32 bytes written
    out_memo_buf: *mut u8,   // caller-owned buffer
    inout_memo_len: *mut usize, // in: capacity; out: bytes written
) -> i32 {
    if seed_len < 32 {
        return POS2_BAD_SEED;
    }
    let seed: &[u8] = unsafe { std::slice::from_raw_parts(seed_ptr, seed_len) };

    let farmer_pk_bytes: &[u8; 48] =
        match unsafe { (farmer_pk_ptr as *const [u8; 48]).as_ref() } {
            Some(b) => b,
            None => return POS2_BAD_FARMER_PK,
        };
    let farmer_pk = match PublicKey::from_bytes(farmer_pk_bytes) {
        Ok(pk) => pk,
        Err(_) => return POS2_BAD_FARMER_PK,
    };

    let (pool_pk_opt, pool_ph_opt, pool_key_slice): (
        Option<PublicKey>,
        Option<Bytes32>,
        &[u8],
    ) = match pool_kind {
        x if x == POS2_POOL_PK => {
            let bytes: &[u8; 48] =
                match unsafe { (pool_key_ptr as *const [u8; 48]).as_ref() } {
                    Some(b) => b,
                    None => return POS2_BAD_POOL_KEY,
                };
            let pk = match PublicKey::from_bytes(bytes) {
                Ok(pk) => pk,
                Err(_) => return POS2_BAD_POOL_KEY,
            };
            (Some(pk), None, &bytes[..])
        }
        x if x == POS2_POOL_PH => {
            let bytes: &[u8; 32] =
                match unsafe { (pool_key_ptr as *const [u8; 32]).as_ref() } {
                    Some(b) => b,
                    None => return POS2_BAD_POOL_KEY,
                };
            let ph: Bytes32 = (*bytes).into();
            (None, Some(ph), &bytes[..])
        }
        _ => return POS2_BAD_POOL_KIND,
    };

    let master_sk = SecretKey::from_seed(seed);
    let local_sk  = master_sk_to_local_sk(&master_sk);
    let local_pk  = local_sk.public_key();

    let include_taproot = pool_ph_opt.is_some();
    let plot_pk = generate_plot_public_key(&local_pk, &farmer_pk, include_taproot);

    let plot_id: Bytes32 = compute_plot_id_v2(
        strength,
        &plot_pk,
        pool_pk_opt.as_ref(),
        pool_ph_opt.as_ref(),
        plot_index,
        meta_group,
    );

    let master_sk_bytes = master_sk.to_bytes();
    let memo_len = pool_key_slice.len() + 48 /* farmer_pk */ + master_sk_bytes.len();

    let capacity = unsafe { *inout_memo_len };
    if capacity < memo_len {
        unsafe { *inout_memo_len = memo_len };
        return POS2_MEMO_BUF_TOO_SMALL;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(plot_id.as_ref().as_ptr(), out_plot_id, 32);
        let dst = out_memo_buf;
        std::ptr::copy_nonoverlapping(pool_key_slice.as_ptr(), dst, pool_key_slice.len());
        std::ptr::copy_nonoverlapping(
            farmer_pk_bytes.as_ptr(),
            dst.add(pool_key_slice.len()),
            48,
        );
        std::ptr::copy_nonoverlapping(
            master_sk_bytes.as_ptr(),
            dst.add(pool_key_slice.len() + 48),
            master_sk_bytes.len(),
        );
        *inout_memo_len = memo_len;
    }

    POS2_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same inputs must produce identical plot_id + memo.
    #[test]
    fn deterministic_same_seed() {
        let seed      = [0xAA_u8; 32];
        let farmer_pk = SecretKey::from_seed(&[0xBB_u8; 32]).public_key().to_bytes();
        let pool_ph   = [0xCC_u8; 32];

        let mut pid1 = [0u8; 32];
        let mut memo1 = vec![0u8; 128];
        let mut mlen1: usize = memo1.len();
        let rc1 = unsafe {
            pos2_keygen_derive_plot(
                seed.as_ptr(), seed.len(),
                farmer_pk.as_ptr(),
                pool_ph.as_ptr(), POS2_POOL_PH,
                2, 0, 0,
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
                seed.as_ptr(), seed.len(),
                farmer_pk.as_ptr(),
                pool_ph.as_ptr(), POS2_POOL_PH,
                2, 0, 0,
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
}
