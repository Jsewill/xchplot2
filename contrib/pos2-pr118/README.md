# Preparation for pos2-chip PR #118

[PR #118](https://github.com/Chia-Network/pos2-chip/pull/118) proposes plot
groups as the supported proving format. It was open and awaiting review on
2026-09-09. This compatibility check pins its current head,
[`2d737fa40dd579431c48fcf4b584b59365ad1791`](https://github.com/Chia-Network/pos2-chip/commit/2d737fa40dd579431c48fcf4b584b59365ad1791).

This is an opt-in interoperability test and migration plan. Normal builds
still use `b0da7aa7bec3974833d651173a6d9953c21eb808` and produce `.plot2`
files. Production grouped plotting is not implemented here. Upstream's
[`PlotGroupFile::writeData`](https://github.com/Chia-Network/pos2-chip/blob/2d737fa40dd579431c48fcf4b584b59365ad1791/src/plot/PlotFile.hpp#L525)
only creates single-plot groups for tests; the production grouping tool is
outside this PR.

## Run the compatibility check

Use an existing xchplot2 executable from either the main or CUDA-only branch.
The separate reference build needs CMake, Git, and a C++20 compiler. It fetches
the pinned PR into its own build directory and reuses our existing solver
candidate-buffer patch.

```sh
cmake -S contrib/pos2-pr118 -B build-pr118 -DCMAKE_BUILD_TYPE=Release
cmake --build build-pr118 --parallel 4
python3 contrib/pos2-pr118/check.py \
  build/tools/xchplot2/xchplot2 build-pr118/pos2_pr118_check
```

The default runs the CPU batch path. Add `--gpu` to exercise `gpu0` with the
Plain tier instead; that requires a working GPU and the plotter's normal
runtime environment. It does not substitute a CPU for a missing GPU.

For an already checked-out upstream revision, configure with
`-DPOS2_PR118_DIR=/absolute/path/to/pos2-chip`. Use a separate build directory
when evaluating a new revision, and record its commit alongside the results.

The test generates four temporary k=18 fixtures with plot indices 0, 1,
4660, and 65535, meta groups 0, 7, and 255, and strengths 2 and 4. It checks:

- Plot ID derivation against Python's SHA-256 of
  `group_id || big_endian_u16(plot_index) || u8(meta_group)`.
- Exact raw fragment equality between current xchplot2 and the PR's CPU
  plotter, including duplicate counts.
- The new `.gplot` header, memo, and a complete fragment round-trip against
  the deduplicated raw stream. Deduplication is intentional upstream.
- Group proving, per-plot index propagation, solving, and full proof
  validation for 32 deterministic challenges per fixture. Each fixture
  must yield at least one validated proof.

Only a private copy of each freshly generated fixture gets its raw header
adapted from v1/plot ID to v2/group ID. Files are removed when the test exits.
This fixture adapter is not a converter for arbitrary existing plots, and
these small test groups are not presented as farmable production output.
The test supplies synthetic group IDs; farmer/pool key and memo derivation
remain a separate migration requirement below.

Verified on 2026-09-09 with GCC 15.3: CPU and GPU runs passed for both
branches, with 129 validated full group proofs per run (516 total). GPU
runs used an NVIDIA GeForce RTX 4090 with driver 610.57.04. These results
cover k=18 single-plot test groups and the Plain GPU tier; production group
sizes, other tiers, AMD, and Intel hardware remain outside this check.

## Required changes before adopting the PR

| Area | Current behavior | Required migration |
| --- | --- | --- |
| Key generation | `keygen-rs` already computes the proposed per-plot ID via chia-protocol, but exposes only that ID and the memo. | Expose the group ID without changing the existing per-plot ID calculation; check both pool-PK and pool-contract cases against upstream vectors. |
| Batch identity | `plot -n N` creates fresh keys and a memo for each plot. An incrementing index alone does not make these plots a group. | Generate keys/memo once per group; carry group ID separately from the derived plot ID through `BatchEntry` and plot options. Pack production groups from index zero. Nonzero index bases are an upstream single-plot test facility. |
| Raw writer | `PlotFileWriterParallel.cpp` writes `pos2`, format v1, with a per-plot ID. | Raw v2 carries the group ID. Preserve exact fragment output and the existing bounded header checks, identity checks, temporary files, and durability barriers. |
| Group output | There is no production group assembler. | Integrate the approved grouping tool/format, including its compressed chunk index and fragment deduplication. Publish only complete groups; retain raw inputs until the group is durable and validated. Resume must check the complete group identity and membership. |
| Proof APIs | `ProofParams`, `Prover`, and a validator constructed from per-plot parameters. | Use `PlotGroupParams` for challenge selection and `PlotProofParams` for plotting/solving. Consume `GroupProver` results with their `plot_index`; validate each full proof with group parameters and that index. |
| Build integration | Legacy proof headers and header-only CPU dependency surface. | Update the renamed `ChunkCompression.hpp`, changed AES/span constructors, and compile upstream's `src/pos/sha/sha256.c`. The existing solver patch applies to the pinned PR unchanged. |
| Testnet mode | `-T` changes the Xs hash in CPU/GPU plotting and has dedicated parity vectors. | The PR removes this mode. Reject it explicitly in the new format path and update its callers/tests; never silently ignore it. |
| Validation and rollout | Hosted CI and the existing optional hardware suites target the current pin and format. | Recheck the approved revision, cover grouped identity/resume/corruption failures, run CPU and every GPU tier/spill path against the new reference, then update the dependency pin and installation matrices in both branches. |

The relevant code is in `keygen-rs/src/lib.rs`, `tools/xchplot2/cli.cpp`,
`src/host/BatchManifest.cpp`, `BatchPlotter.cpp`, `CpuPlotter.cpp`,
`GpuPlotter.cpp`, and `PlotFileWriterParallel.cpp`. GPU table construction
still takes the derived per-plot ID; it must not receive the group ID.

Do not merely rename old `.plot2` files or reinterpret their IDs as group
IDs. The raw format version, identity semantics, proving context, and
group compression all change. Production migration of existing plots
needs a separate decision once the upstream format and grouping tool settle.
