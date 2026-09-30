# Preparation for pos2-chip PR #118

[PR #118](https://github.com/Chia-Network/pos2-chip/pull/118) proposes plot
groups as the supported proving format. It remains open on 2026-09-30.
This opt-in build pins its current head,
[`e27db9557ec611572ebec3526375e1789a708acc`](https://github.com/Chia-Network/pos2-chip/commit/e27db9557ec611572ebec3526375e1789a708acc).

This is an opt-in interoperability test and migration plan. Normal builds
still use `b0da7aa7bec3974833d651173a6d9953c21eb808` and produce `.plot2`
files. The experimental job runner below assembles actual multi-member groups
using the pinned reader's format. Upstream's
[`PlotGroupFile::writeData`](https://github.com/Chia-Network/pos2-chip/blob/e27db9557ec611572ebec3526375e1789a708acc/src/plot/PlotFile.hpp#L523)
still only creates single-plot groups for tests; our assembler does not call it.
These results establish interoperability with that public reader, not settled
farmer compatibility. The companion [chia-rs PR #1517](https://github.com/Chia-Network/chia_rs/pull/1517)
and [chia-blockchain PR #21396](https://github.com/Chia-Network/chia-blockchain/pull/21396)
remain drafts, with the latter identifying a pending taproot update.

## Run the compatibility check

Use an existing xchplot2 executable from either the main or CUDA-only branch.
The separate reference build needs CMake, Git, Cargo, and a C++20 compiler.
It fetches the pinned PR into its own build directory and reuses our existing
solver candidate-buffer patch.

```sh
cmake -S contrib/pos2-pr118 -B build-pr118 -DCMAKE_BUILD_TYPE=Release
cmake --build build-pr118 --parallel 2
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

Refreshed on 2026-09-30 with GCC 16.2: CPU and RTX 4090 Plain GPU runs passed
all four fixtures and 129 full proofs each. The production dependency pin
remains unchanged.

## Experimental grouped jobs

The separate build also produces `pos2_pr118_group` and builds the existing
Rust key shim. Use the job runner for a group packed from member index zero:

```sh
python3 contrib/pos2-pr118/group.py \
  build/tools/xchplot2/xchplot2 build-pr118/pos2_pr118_group \
  --farmer-pk "$FARMER_PK" --pool-ph "$POOL_PH" \
  --group-size 2 --k 28 --devices gpu0 --tier plain \
  --max-host-ram 18G --max-group-ram 512 --out /plots/experimental.gplot
```

`--pool-pk` is supported instead of `--pool-ph`. The default device is CPU;
GPU selection is explicit and never falls back to CPU. Key generation uses
released chia-protocol/chia-bls 0.48's existing pool-PK and pool-contract
taproot contract. It generates one seed and memo per group, exposes the group
hash, and derives each member ID as SHA-256 of the group ID, big-endian u16
index, and u8 meta group. `--seed` makes the group reproducible. k must be even
in 18..28; `--testnet` is explicitly rejected because this PR removes it.

The runner publishes a private `.gplot.job.json` before plotting and keeps its
private raw directory beside the output. These contain secret plot keys.
On POSIX they use owner-only file/directory permissions. Each member runs
through the existing batch plotter. Assembly validates its ID, memo,
parameters, index, chunk bounds, decoded counts, sorted fragment ranges, and
membership before encoding. Deduplication is within each member only.
Regions must be nonempty for every member, as required by the pinned reader;
unsupported region/index encodings fail before publication.

Add `--resume` to the same command after interruption. The runner re-derives
and compares the saved group ID, memo and complete membership, checks existing
raw files before reusing them, and finishes missing members. A completed
group is checked against every raw fragment, then all returned qualities
from 32 deterministic challenges are solved and validated with their member
indices. Every member must yield a full proof. It writes and verifies a
temporary group, fsyncs it, publishes without overwriting another group, and
syncs the parent directory. Raw inputs are retained even after success.
Invalid existing outputs are rejected and preserved.
The saved job also records the reference's actual upstream Git revision plus
a SHA-256 content identifier for the upstream headers/SHA/FSE sources and
solver patch it uses.
Source trees without Git metadata use that content identifier alone. Changing
the reference source identity rejects resume instead of attributing a local
override to the default dependency pin.

`--max-group-ram MIB` limits modeled assembly and reader buffers, not process
RSS or plotting RAM. The model reserves 24 bytes per grouped chunk for indexes
(96 MiB at k=28), each member's raw index and one decoded raw chunk, transient
raw decode arrays, group compression buffers, and bounded reader buffers.
It checks file-controlled lengths/counts before allocating. Groups that exceed
the budget are rejected. Full-proof solver/chainer allocations, standard
library overhead, and OS caches are outside this modeled cap. Solver work
runs one member at a time. `--max-host-ram` controls the existing plotter
separately.

Run the regression check (small pool-PK/pool-contract groups, a 64-member
group, completed and partial resume, corruption, private artifacts, error
redaction, and RAM rejection):

```sh
python3 contrib/pos2-pr118/group-test.py \
  build/tools/xchplot2/xchplot2 build-pr118/pos2_pr118_group
# GPU coverage of the same jobs:
python3 contrib/pos2-pr118/group-test.py \
  build/tools/xchplot2/xchplot2 build-pr118/pos2_pr118_group --devices gpu0
cargo test --manifest-path keygen-rs/Cargo.toml --locked
```

The C++ `identity`, `prepare`, `check-raw`, `assemble`, and `verify` commands are internal
parts of this runner. Direct `assemble` writes its supplied temporary path;
the Python runner owns cleanup, durability and publication.

Verified on 2026-09-30: the regression passed on CPU and RTX 4090 Plain GPU,
including every member of a 64-member k=18 group. A two-member k=28 GPU group
passed exact round-trip checks across all 4,194,304 grouped chunks and 86 full
proofs (43 per member). The complete GPU job took 59.60 s with an 18 GiB plotting
budget in a 22 GiB container. A separate host assembly plus verification took
50.75 s, peaked at 295,944 KiB RSS (289.01 MiB), and reproduced the published
group byte for byte under the 512 MiB modeled buffer cap. Solver/chainer memory
is included in that measured RSS, although excluded from the modeled cap.
These measurements cover this hardware, two k=28 members, and 64 k=18 members;
larger k=28 groups and other GPU tiers/vendors remain unverified.

## Required changes before adopting the PR

| Area | Current behavior | Required migration |
| --- | --- | --- |
| Key generation | `keygen-rs` exposes the existing group hash through an additional API, used by the experimental runner. Tests cover both pool kinds against released chia-protocol. | Recheck the approved farmer/taproot contract before production adoption. |
| Batch identity | `plot -n N` creates fresh keys and a memo for each plot. An incrementing index alone does not make these plots a group. | Generate keys/memo once per group; carry group ID separately from the derived plot ID through `BatchEntry` and plot options. Pack production groups from index zero. Nonzero index bases are an upstream single-plot test facility. |
| Raw writer | `PlotFileWriterParallel.cpp` writes `pos2`, format v1, with a per-plot ID. | Raw v2 carries the group ID. Preserve exact fragment output and the existing bounded header checks, identity checks, temporary files, and durability barriers. |
| Group output | The opt-in runner assembles multi-member groups with compressed chunk indexes, per-member deduplication, durable publication and complete resume identity checks. | Adopt an approved format/tool contract and validate production hardware and group sizes before enabling the main CLI. |
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
