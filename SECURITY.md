# Security Policy

## Reporting a vulnerability

Email **<abraham.sewill@proton.me>** with a description of the issue and
steps to reproduce. Please do not open a public GitHub issue for
security-sensitive reports.

## Scope — what counts for a plotter

xchplot2 is a client-side plot builder. It handles:

- Farmer and pool public keys provided on the command line.
- Optional `--seed` entropy that derives per-plot subseeds; a weak
  or reused seed lets an attacker who observes plot IDs correlate
  plots to the same master key.
- BLS key parsing via the
  [`chia-bls` Rust crate](https://crates.io/crates/chia-bls) through
  `keygen-rs`.
- Per-plot private keys, included in plot memos and saved job manifests.
- Large file writes into caller-supplied output directories.
- Optional compute pool identities, TLS connections and LAN discovery.
  Enrolled workers can compute for several plotting hosts; trust is granted
  to the pool, not to one exclusive host pair.

`plot` saves identities before starting, including for unseeded jobs. The
`xchplot2-job-*.tsv` files contain the memo and its private plot key material;
they are intended persistent recovery data. `--manifest` selects another
path, and `batch` can read the same format. Keep manifests and plot memos
private; redact them before sharing logs or reproductions. Retain a manifest
while recovery may be needed. Deleting it can prevent recovery of an unseeded
job's identities.

New manifests and their publication temporaries use owner-only permissions
on Linux and protected owner-only ACLs on Windows, and are not allowed to replace another job's manifest. Report
permission, disclosure, or publication failures through the channel above.
See [plotting and recovery](REFERENCE.md#plotting-and-recovery) for behavior.

Compute enrollment and invitations contain private TLS keys and must be kept
private. Only the pool creator keeps the signing key. Each invitation assigns
a distinct machine identity; do not clone it across workers. Discovered
advertisements do not establish trust: work connections authenticate through
the enrolled CA. Workers receive task identities and bounded fragment/codec
data, not storage paths, seeds or private plot memos. Trust enrolled workers
to compute correctly; bounded parsing is not verification of every result's
mathematical provenance. See [shared compute](REFERENCE.md#shared-compute)
for interface selection, credential renewal and signed revocation policies.
Revocation is not instantaneous on disconnected machines: they must receive
the newer signed policy by reconnecting to an updated peer or importing it.
A renewed certificate preserves the stable node ID; a revoked node cannot
renew itself back into the pool.

Relevant threat model items we want to hear about:

- **Key handling:** any path where farmer/pool key bytes or the
  master seed or private plot keys leak beyond their intended memo/manifest
  storage, including through logs, temporary-file permissions, or crash dumps.
- **File-path handling:** any way a crafted `-o` / `out_dir` / memo
  string escapes the intended output directory or overwrites files
  outside it (path traversal, symlink races). This includes races around temporary files and atomic publication.
- **Manifest parsing:** malformed `batch` manifests that cause
  out-of-bounds reads, arbitrary allocation, or unchecked sign
  conversion.
- **Shared compute:** authentication or lease-fencing bypasses, cross-owner
  task substitution, unbounded network allocations, interface restrictions
  bypassed, or private enrollment material disclosed.
- **Build-time supply chain:** tampering paths in
  `scripts/install-deps.sh`, `Containerfile`, `compose.yaml`, or
  the FetchContent targets (pos2-chip, AdaptiveCpp, mdns).

## Explicitly out of scope

- Proof-of-space soundness and the v2 PoS algorithm itself —
  report those upstream in
  [`pos2-chip`](https://github.com/Chia-Network/pos2-chip).
- Consensus, farming, or wallet behavior — those belong in
  [`chia-blockchain`](https://github.com/Chia-Network/chia-blockchain)
  and [`chia_rs`](https://github.com/Chia-Network/chia_rs).
- Performance regressions on exotic GPUs — file as a normal bug.

## Response

Acknowledgement within a week. Fixes for in-scope issues land on
`main` (and the `cuda-only` branch if applicable) with credit in the
commit message unless you prefer otherwise.
