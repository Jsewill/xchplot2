# Security Policy

## Reporting a vulnerability

Email **abraham.sewill@proton.me** with a description of the issue and
steps to reproduce. Please do not open a public GitHub issue for
security-sensitive reports.

## Scope — what counts for a plotter

xchplot2 is a client-side plot builder. It handles:

- Farmer and pool public keys provided on the command line.
- Optional `--seed` entropy that derives per-plot subseeds; a weak
  or reused seed lets an attacker who observes plot IDs correlate
  plots to the same master key.
- BLS key parsing via the
  [`chia` Rust crate](https://crates.io/crates/chia) through
  `keygen-rs`.
- Large file writes into caller-supplied output directories.

Relevant threat model items we want to hear about:

- **Key handling:** any path where farmer/pool key bytes or the
  master seed leak into logs, temporary files, crash dumps, or
  the plot file itself beyond the documented memo payload.
- **File-path handling:** any way a crafted `-o` / `out_dir` / memo
  string escapes the intended output directory or overwrites files
  outside it (path traversal, symlink races). The atomic
  `.partial` + rename is safe by design; report if you can break it.
- **Manifest parsing:** malformed `batch` manifests that cause
  out-of-bounds reads, arbitrary allocation, or unchecked sign
  conversion.
- **Build-time supply chain:** tampering paths in
  `scripts/install-deps.sh`, `Containerfile`, `compose.yaml`, or
  the FetchContent targets (pos2-chip, AdaptiveCpp).

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
