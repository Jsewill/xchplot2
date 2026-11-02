// pos2_keygen — C-callable shim around chia-bls
// that derives a v2 plot's plot_id and memo from caller-supplied farmer +
// pool keys plus a 32-byte master-SK seed. The GPU plotter uses the returned
// plot_id / memo to drive the existing batch path.
//
// chia-bls supplies BLS12-381 arithmetic and EIP-2333 HD derivation.
// This shim sequences PoS2 1.0 identity hashing, the V2 taproot contract in
// chia-blockchain PR #21500, and the existing memo layout.

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
pub const POS2_BAD_HASH_INPUT: i32 = -8;

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
    // chia-blockchain PR #21500 deliberately omits the standalone local key.
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

/// Caller-owned bytes supplied to the general SHA-256 helper.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pos2Sha256Segment {
    pub data: *const u8,
    pub len: usize,
}

/// Hash the exact concatenation of byte segments, independent of key derivation.
/// No lengths or separators are inserted, and input bytes are never coalesced.
/// Metadata is validated before reading any segment data. On rejection, the
/// output remains unchanged. A null array is accepted for an empty segment list,
/// and a null data pointer is accepted for an empty segment.
///
/// # Safety
/// For valid metadata, `segments` must point to an aligned, initialized array
/// of `segment_count` descriptors. Every nonempty data region must be readable
/// for its declared length, and `out_digest` must point to 32 writable bytes.
/// All input regions must remain valid and unmodified throughout this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_sha256_segments(
    segments: *const Pos2Sha256Segment,
    segment_count: usize,
    out_digest: *mut u8,
) -> i32 {
    if out_digest.is_null()
        || segment_count > isize::MAX as usize / std::mem::size_of::<Pos2Sha256Segment>()
        || (segment_count != 0 && (segments.is_null() || !segments.is_aligned()))
    {
        return POS2_BAD_HASH_INPUT;
    }
    let segments = if segment_count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(segments, segment_count) }
    };
    let mut total_bytes = 0u64;
    for segment in segments {
        if segment.len > isize::MAX as usize || (segment.len != 0 && segment.data.is_null()) {
            return POS2_BAD_HASH_INPUT;
        }
        let Ok(len) = u64::try_from(segment.len) else {
            return POS2_BAD_HASH_INPUT;
        };
        total_bytes = match total_bytes.checked_add(len) {
            Some(total) if total <= u64::MAX / 8 => total,
            _ => return POS2_BAD_HASH_INPUT,
        };
    }
    let mut hash = Sha256::new();
    for segment in segments {
        if segment.len != 0 {
            hash.update(unsafe { std::slice::from_raw_parts(segment.data, segment.len) });
        }
    }
    let digest = hash.finalize();
    unsafe { std::ptr::copy_nonoverlapping(digest.as_ptr(), out_digest, 32) };
    POS2_OK
}

/// Hash segments from a reader that may reuse its storage between calls.
/// The reader returns 1 for a segment, 0 at EOF, and a negative value on failure.
/// Invalid metadata or any other reader result leaves the output unchanged.
///
/// # Safety
/// The reader must not unwind and must accept the supplied context and a writable
/// descriptor. Each returned nonempty region must remain readable and unmodified
/// until the next reader call. `out_digest` must point to 32 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_sha256_stream(
    reader: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut Pos2Sha256Segment) -> i32>,
    context: *mut std::ffi::c_void,
    out_digest: *mut u8,
) -> i32 {
    let Some(reader) = reader else {
        return POS2_BAD_HASH_INPUT;
    };
    if out_digest.is_null() {
        return POS2_BAD_HASH_INPUT;
    }
    let mut hash = Sha256::new();
    let mut total_bytes = 0u64;
    loop {
        let mut segment = Pos2Sha256Segment {
            data: std::ptr::null(),
            len: 0,
        };
        match unsafe { reader(context, &mut segment) } {
            0 => break,
            1 => {}
            _ => return POS2_BAD_HASH_INPUT,
        }
        if segment.len > isize::MAX as usize || (segment.len != 0 && segment.data.is_null()) {
            return POS2_BAD_HASH_INPUT;
        }
        let Ok(len) = u64::try_from(segment.len) else {
            return POS2_BAD_HASH_INPUT;
        };
        total_bytes = match total_bytes.checked_add(len) {
            Some(total) if total <= u64::MAX / 8 => total,
            _ => return POS2_BAD_HASH_INPUT,
        };
        if segment.len != 0 {
            hash.update(unsafe { std::slice::from_raw_parts(segment.data, segment.len) });
        }
    }
    let digest = hash.finalize();
    unsafe { std::ptr::copy_nonoverlapping(digest.as_ptr(), out_digest, 32) };
    POS2_OK
}

// Opaque to C callers. The states never leave the current process, so the
// accelerated sha2 implementation can keep its native representation.
struct Sha256State {
    hash: Sha256,
    bytes: u64,
}

unsafe fn allocate_sha256_state(state: Sha256State) -> *mut std::ffi::c_void {
    let layout = std::alloc::Layout::new::<Sha256State>();
    let allocation = unsafe { std::alloc::alloc(layout) }.cast::<Sha256State>();
    if allocation.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { allocation.write(state) };
    allocation.cast()
}

/// Allocate a process-local SHA state, or return null on allocation failure.
#[unsafe(no_mangle)]
pub extern "C" fn pos2_sha256_state_new() -> *mut std::ffi::c_void {
    unsafe {
        allocate_sha256_state(Sha256State {
            hash: Sha256::new(),
            bytes: 0,
        })
    }
}

/// Clone a valid live state; null input or allocation failure returns null.
///
/// # Safety
/// A nonnull state must be a live handle from new/clone and must not be mutated
/// or freed during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_sha256_state_clone(
    state: *const std::ffi::c_void,
) -> *mut std::ffi::c_void {
    if state.is_null() {
        return std::ptr::null_mut();
    }
    let state = unsafe { &*state.cast::<Sha256State>() };
    unsafe {
        allocate_sha256_state(Sha256State {
            hash: state.hash.clone(),
            bytes: state.bytes,
        })
    }
}

/// Conservative state-allocation size including a fixed allocator allowance.
#[unsafe(no_mangle)]
pub extern "C" fn pos2_sha256_state_bytes() -> usize {
    std::mem::size_of::<Sha256State>() + 64
}

/// Free an owned state once; null is accepted.
///
/// # Safety
/// A nonnull state must be a live handle from new/clone with no concurrent users.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_sha256_state_free(state: *mut std::ffi::c_void) {
    if !state.is_null() {
        let state = state.cast::<Sha256State>();
        unsafe {
            std::ptr::drop_in_place(state);
            std::alloc::dealloc(state.cast(), std::alloc::Layout::new::<Sha256State>());
        }
    }
}

/// Update a live state synchronously, preserving it on invalid metadata.
///
/// # Safety
/// State must be a live new/clone handle with exclusive access. Every nonempty
/// input must be readable for len bytes and must not overlap the state itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_sha256_state_update(
    state: *mut std::ffi::c_void,
    data: *const u8,
    len: usize,
) -> i32 {
    if state.is_null() || len > isize::MAX as usize || (len != 0 && data.is_null()) {
        return POS2_BAD_HASH_INPUT;
    }
    let Ok(length) = u64::try_from(len) else {
        return POS2_BAD_HASH_INPUT;
    };
    let state = unsafe { &mut *state.cast::<Sha256State>() };
    let bytes = match state.bytes.checked_add(length) {
        Some(bytes) if bytes <= u64::MAX / 8 => bytes,
        _ => return POS2_BAD_HASH_INPUT,
    };
    if len != 0 {
        state
            .hash
            .update(unsafe { std::slice::from_raw_parts(data, len) });
    }
    state.bytes = bytes;
    POS2_OK
}

/// Return a digest without consuming the state; errors preserve the output.
///
/// # Safety
/// State must be a live new/clone handle and may not be mutated or freed during
/// the call. A nonnull output must point to writable storage for the digest and
/// must not overlap the opaque state allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pos2_sha256_state_digest(
    state: *const std::ffi::c_void,
    out_digest: *mut u8,
) -> i32 {
    if state.is_null() || out_digest.is_null() {
        return POS2_BAD_HASH_INPUT;
    }
    let state = unsafe { &*state.cast::<Sha256State>() };
    let digest = state.hash.clone().finalize();
    unsafe { std::ptr::copy_nonoverlapping(digest.as_ptr(), out_digest, 32) };
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
        // chia-blockchain PR #21500, test_v2_plot_loading.py.
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

    #[test]
    fn sha256_segments_matches_known_vectors() {
        for (input, expected) in [
            (
                &b""[..],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                &b"abc"[..],
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                &b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"[..],
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
        ] {
            let segment = Pos2Sha256Segment {
                data: input.as_ptr(),
                len: input.len(),
            };
            let mut output = [0; 32];
            assert_eq!(
                unsafe { pos2_sha256_segments(&segment, 1, output.as_mut_ptr()) },
                POS2_OK
            );
            assert_eq!(hex::encode(output), expected);
        }
        let input = vec![b'a'; 1_000_000];
        let segment = Pos2Sha256Segment {
            data: input.as_ptr(),
            len: input.len(),
        };
        let mut output = [0; 32];
        assert_eq!(
            unsafe { pos2_sha256_segments(&segment, 1, output.as_mut_ptr()) },
            POS2_OK
        );
        assert_eq!(
            hex::encode(output),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha256_segments_preserves_boundaries_and_empty_segments() {
        let input: Vec<u8> = (0..4096).map(|value| (value % 256) as u8).collect();
        let expected = Sha256::digest(&input);
        for split in [0, 1, 7, 55, 56, 63, 64, 65, input.len()] {
            let segments = [
                Pos2Sha256Segment {
                    data: std::ptr::null(),
                    len: 0,
                },
                Pos2Sha256Segment {
                    data: input[..split].as_ptr(),
                    len: split,
                },
                Pos2Sha256Segment {
                    data: std::ptr::null(),
                    len: 0,
                },
                Pos2Sha256Segment {
                    data: input[split..].as_ptr(),
                    len: input.len() - split,
                },
            ];
            let mut output = [0; 32];
            assert_eq!(
                unsafe {
                    pos2_sha256_segments(segments.as_ptr(), segments.len(), output.as_mut_ptr())
                },
                POS2_OK
            );
            assert_eq!(output.as_slice(), expected.as_slice());
        }
        let mut output = [0; 32];
        assert_eq!(
            unsafe { pos2_sha256_segments(std::ptr::null(), 0, output.as_mut_ptr()) },
            POS2_OK
        );
        assert_eq!(
            hex::encode(output),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_segments_rejects_invalid_metadata_without_changing_output() {
        let input = b"abc";
        let segment = Pos2Sha256Segment {
            data: input.as_ptr(),
            len: input.len(),
        };
        assert_eq!(
            unsafe { pos2_sha256_segments(&segment, 1, std::ptr::null_mut()) },
            POS2_BAD_HASH_INPUT
        );
        let misaligned = (&segment as *const Pos2Sha256Segment)
            .cast::<u8>()
            .wrapping_add(1)
            .cast::<Pos2Sha256Segment>();
        for (segments, count) in [
            (std::ptr::null(), 1),
            (&segment as *const Pos2Sha256Segment, usize::MAX),
            (misaligned, 1),
        ] {
            let mut output = [0x55; 32];
            assert_eq!(
                unsafe { pos2_sha256_segments(segments, count, output.as_mut_ptr()) },
                POS2_BAD_HASH_INPUT
            );
            assert_eq!(output, [0x55; 32]);
        }
        for invalid in [
            Pos2Sha256Segment {
                data: std::ptr::null(),
                len: 1,
            },
            Pos2Sha256Segment {
                data: input.as_ptr(),
                len: usize::MAX,
            },
        ] {
            let segments = [segment, invalid];
            let mut output = [0x55; 32];
            assert_eq!(
                unsafe {
                    pos2_sha256_segments(segments.as_ptr(), segments.len(), output.as_mut_ptr())
                },
                POS2_BAD_HASH_INPUT
            );
            assert_eq!(output, [0x55; 32]);
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn sha256_segments_rejects_message_length_overflow_before_reading_data() {
        let input = b"abc";
        let segments = [
            Pos2Sha256Segment {
                data: input.as_ptr(),
                len: (u64::MAX / 8) as usize,
            },
            Pos2Sha256Segment {
                data: input.as_ptr(),
                len: 1,
            },
        ];
        let mut output = [0x55; 32];
        assert_eq!(
            unsafe { pos2_sha256_segments(segments.as_ptr(), segments.len(), output.as_mut_ptr()) },
            POS2_BAD_HASH_INPUT
        );
        assert_eq!(output, [0x55; 32]);
    }

    #[test]
    fn sha256_states_clone_prefixes_and_digest_without_consuming() {
        let state = pos2_sha256_state_new();
        assert!(!state.is_null());
        assert!(pos2_sha256_state_bytes() >= std::mem::size_of::<Sha256State>());
        unsafe {
            assert_eq!(pos2_sha256_state_update(state, b"abc".as_ptr(), 3), POS2_OK);
            let prefix = pos2_sha256_state_clone(state);
            assert!(!prefix.is_null());
            assert_eq!(pos2_sha256_state_update(state, b"def".as_ptr(), 3), POS2_OK);
            assert_eq!(
                pos2_sha256_state_update(prefix, b"xyz".as_ptr(), 3),
                POS2_OK
            );
            for (handle, expected) in [(state, b"abcdef"), (prefix, b"abcxyz")] {
                let mut output = [0x55; 32];
                assert_eq!(
                    pos2_sha256_state_digest(handle, output.as_mut_ptr()),
                    POS2_OK
                );
                assert_eq!(output.as_slice(), Sha256::digest(expected).as_slice());
                let mut repeated = [0x55; 32];
                assert_eq!(
                    pos2_sha256_state_digest(handle, repeated.as_mut_ptr()),
                    POS2_OK
                );
                assert_eq!(repeated, output);
                assert_eq!(
                    pos2_sha256_state_update(handle, std::ptr::null(), 0),
                    POS2_OK
                );
                assert_eq!(pos2_sha256_state_update(handle, b"!".as_ptr(), 1), POS2_OK);
                let mut extended = expected.to_vec();
                extended.push(b'!');
                assert_eq!(
                    pos2_sha256_state_digest(handle, output.as_mut_ptr()),
                    POS2_OK
                );
                assert_eq!(output.as_slice(), Sha256::digest(extended).as_slice());
            }
            pos2_sha256_state_free(prefix);
            pos2_sha256_state_free(state);
            pos2_sha256_state_free(std::ptr::null_mut());
            assert!(pos2_sha256_state_clone(std::ptr::null()).is_null());
        }
    }

    #[test]
    fn sha256_states_preserve_state_and_output_on_invalid_input() {
        let state = pos2_sha256_state_new();
        assert!(!state.is_null());
        let mut output = [0x55; 32];
        unsafe {
            assert_eq!(
                pos2_sha256_state_update(std::ptr::null_mut(), b"abc".as_ptr(), 3),
                POS2_BAD_HASH_INPUT
            );
            assert_eq!(
                pos2_sha256_state_digest(std::ptr::null(), output.as_mut_ptr()),
                POS2_BAD_HASH_INPUT
            );
            assert_eq!(output, [0x55; 32]);
            assert_eq!(
                pos2_sha256_state_digest(state, std::ptr::null_mut()),
                POS2_BAD_HASH_INPUT
            );
            assert_eq!(pos2_sha256_state_update(state, b"abc".as_ptr(), 3), POS2_OK);
            let mut lengths = vec![usize::MAX];
            if usize::BITS == 64 {
                lengths.push((u64::MAX / 8) as usize);
            }
            assert_eq!(
                pos2_sha256_state_update(state, std::ptr::null(), 1),
                POS2_BAD_HASH_INPUT
            );
            for length in lengths {
                assert_eq!(
                    pos2_sha256_state_update(state, b"abc".as_ptr(), length),
                    POS2_BAD_HASH_INPUT
                );
                assert_eq!(
                    pos2_sha256_state_digest(state, output.as_mut_ptr()),
                    POS2_OK
                );
                assert_eq!(output.as_slice(), Sha256::digest(b"abc").as_slice());
            }
            pos2_sha256_state_free(state);
        }
    }

    struct HashReader {
        segments: Vec<Vec<u8>>,
        scratch: Vec<u8>,
        next: usize,
        end_status: i32,
        invalid_tail: Option<Pos2Sha256Segment>,
    }

    unsafe extern "C" fn read_hash_segment(
        context: *mut std::ffi::c_void,
        next: *mut Pos2Sha256Segment,
    ) -> i32 {
        let reader = unsafe { &mut *context.cast::<HashReader>() };
        if let Some(input) = reader.segments.get(reader.next) {
            reader.scratch.clear();
            reader.scratch.extend_from_slice(input);
            unsafe {
                *next = Pos2Sha256Segment {
                    data: reader.scratch.as_ptr(),
                    len: reader.scratch.len(),
                };
            }
            reader.next += 1;
            1
        } else if let Some(invalid) = reader.invalid_tail.take() {
            unsafe { *next = invalid };
            1
        } else {
            reader.end_status
        }
    }

    #[test]
    fn sha256_stream_consumes_reused_buffers_and_empty_segments() {
        let input: Vec<u8> = (0..4096).map(|value| (value % 256) as u8).collect();
        for len in [0, 3, 55, 56, 63, 64, 65, input.len()] {
            for chunk_size in [1, 7, 55, 64, 65, input.len()] {
                let mut segments = vec![vec![]];
                segments.extend(input[..len].chunks(chunk_size).map(Vec::from));
                segments.push(vec![]);
                let mut reader = HashReader {
                    segments,
                    scratch: Vec::with_capacity(chunk_size),
                    next: 0,
                    end_status: 0,
                    invalid_tail: None,
                };
                let mut output = [0x55; 32];
                assert_eq!(
                    unsafe {
                        pos2_sha256_stream(
                            Some(read_hash_segment),
                            (&mut reader as *mut HashReader).cast(),
                            output.as_mut_ptr(),
                        )
                    },
                    POS2_OK
                );
                assert_eq!(output.as_slice(), Sha256::digest(&input[..len]).as_slice());
            }
        }
        unsafe extern "C" fn empty_reader(
            _context: *mut std::ffi::c_void,
            _next: *mut Pos2Sha256Segment,
        ) -> i32 {
            0
        }
        let mut output = [0x55; 32];
        assert_eq!(
            unsafe {
                pos2_sha256_stream(
                    Some(empty_reader),
                    std::ptr::null_mut(),
                    output.as_mut_ptr(),
                )
            },
            POS2_OK
        );
        assert_eq!(output.as_slice(), Sha256::digest(b"").as_slice());
    }

    #[test]
    fn sha256_stream_preserves_output_on_reader_and_metadata_errors() {
        let mut output = [0x55; 32];
        assert_eq!(
            unsafe { pos2_sha256_stream(None, std::ptr::null_mut(), output.as_mut_ptr()) },
            POS2_BAD_HASH_INPUT
        );
        assert_eq!(output, [0x55; 32]);
        assert_eq!(
            unsafe {
                pos2_sha256_stream(
                    Some(read_hash_segment),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            POS2_BAD_HASH_INPUT
        );
        let input = b"abc";
        let invalid_segments = [
            Pos2Sha256Segment {
                data: std::ptr::null(),
                len: 1,
            },
            Pos2Sha256Segment {
                data: input.as_ptr(),
                len: usize::MAX,
            },
            #[cfg(target_pointer_width = "64")]
            Pos2Sha256Segment {
                data: input.as_ptr(),
                len: (u64::MAX / 8) as usize + 1,
            },
            #[cfg(target_pointer_width = "64")]
            Pos2Sha256Segment {
                data: input.as_ptr(),
                len: (u64::MAX / 8) as usize,
            },
        ];
        for (status, invalid) in [(-1, None), (2, None), (i32::MIN, None)].into_iter().chain(
            invalid_segments
                .into_iter()
                .map(|segment| (0, Some(segment))),
        ) {
            let mut reader = HashReader {
                segments: vec![input.to_vec()],
                scratch: Vec::with_capacity(input.len()),
                next: 0,
                end_status: status,
                invalid_tail: invalid,
            };
            assert_eq!(
                unsafe {
                    pos2_sha256_stream(
                        Some(read_hash_segment),
                        (&mut reader as *mut HashReader).cast(),
                        output.as_mut_ptr(),
                    )
                },
                POS2_BAD_HASH_INPUT
            );
            assert_eq!(output, [0x55; 32]);
        }
    }
}
