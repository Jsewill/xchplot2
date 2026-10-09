# PoS2 1.0 compatibility and grouped jobs

Normal builds and this separate reference build pin
[pos2-chip 1.0.0, `ff89d3e`](https://github.com/Chia-Network/pos2-chip/commit/ff89d3e61f4315311a7f975409a6433e67f699ff).
This includes merged plot groups ([#118](https://github.com/Chia-Network/pos2-chip/pull/118)),
the incompatible Feistel cipher ([#119](https://github.com/Chia-Network/pos2-chip/pull/119)),
and corrected decompression buffer sizing ([#120](https://github.com/Chia-Network/pos2-chip/pull/120)).
The directory/tool names retain `pr118` for existing scripts.

`plot`, `bench`, and `test` now write official version 2 `.gplot` files with
one member at index zero. The experimental runner below assembles multiple
members. Files starting at nonzero indices still use the 0x82 extension;
normal xchplot2 and upstream readers reject that extension.

The companion [chia-rs PR #1517](https://github.com/Chia-Network/chia_rs/pull/1517)
and [chia-blockchain PR #21396](https://github.com/Chia-Network/chia-blockchain/pull/21396)
remain drafts as of 2026-10-08. Key generation implements the V2 taproot change
in [chia-blockchain PR #21484](https://github.com/Chia-Network/chia-blockchain/pull/21484),
including its independent public-key test vector. These checks establish
format/proof interoperability, not released farmer compatibility.

## Run the compatibility check

Use an updated xchplot2 executable from either the main or CUDA-only branch.
The separate reference build needs CMake, Git, Cargo, and a C++20 compiler.
It fetches the pinned revision into its own build directory and reuses our existing
solver candidate-buffer patch.

```sh
cmake -S contrib/pos2-pr118 -B build-pr118 -DCMAKE_BUILD_TYPE=Release
cmake --build build-pr118 --parallel 2
python3 contrib/pos2-pr118/check.py \
  build/tools/xchplot2/xchplot2 build-pr118/pos2_pr118_check
```

The default runs the CPU batch path at k=18. Use `--k 22` or `--k 28`
for larger fixtures, with `TMPDIR` on a disk that has space for raw and group
files. Add `--gpu` to exercise `gpu0` with the
Plain tier instead; that requires a working GPU and the plotter's normal
runtime environment. It does not substitute a CPU for a missing GPU.

For an already checked-out upstream revision, configure with
`-DPOS2_PR118_DIR=/absolute/path/to/pos2-chip`. Use a separate build directory
when evaluating a new revision, and record its commit alongside the results.

The test generates four temporary fixtures with plot indices 0, 1,
4660, and 65535, meta groups 0, 7, and 255, and strengths 2 and 4. It checks:

- Plot ID derivation against Python's SHA-256 of
  `group_id || big_endian_u16(plot_index) || u8(meta_group)`.
- Exact raw fragment equality between current xchplot2 and upstream's CPU
  plotter, including duplicate counts.
- Byte-for-byte native single-member `.gplot` equality with the upstream writer.
- The `.gplot` header, memo, and a complete fragment round-trip against
  the deduplicated raw stream. Deduplication is intentional upstream.
- Group proving, per-plot index propagation, solving, and full proof
  validation for 32 deterministic challenges per fixture. Each fixture
  must yield at least one validated proof.

Fixtures are written directly as raw-v2 members; no legacy header conversion
is performed. Files are removed when the test exits. Synthetic IDs are not
farming keys; the Rust shim separately checks key and memo derivation.

### PoS2 1.0 validation

Checked on 2026-10-08 against `ff89d3e` on Linux x86-64 with an RTX 4090
and NVIDIA driver 610.57.04:

- Main: 39 CTest checks, a separate 11-test host-only build, and seven Rust
  key tests passed. Native CUDA: all 18 CTest checks passed. Cargo release
  builds linked successfully for both branches.
- Main passed all four upstream fixtures on CPU and GPU at k=18. Both
  branches passed all four GPU fixtures at k=22, and main also passed all
  four at k=28. These included exact raw fragments, native single-member
  group bytes, and full proofs.
- The correctness suite passed CPU byte parity and positive full-proof
  samples for the k=28 group and both k=18 raw boundary vectors. Main used
  all five tiers and four spill variants; native CUDA used its four tiers
  and Compact/Minimal spill variants. The host budget was 14 GiB, with
  explicit spill cases using `min`.
- Grouped-job regressions passed on main CPU/GPU and native CUDA GPU,
  including 64-member k=18 groups, multiple files, private saved jobs,
  partial resume, identity mismatches, corruption, and RAM-budget rejection.
- Official two-member groups at k=22 and k=28 passed full-proof validation
  in both production executables, using the assembler's 512 MiB modeled
  buffer cap.

These were correctness checks on a shared desktop GPU. They do not qualify
isolated VRAM limits or physical small cards. AMD, Intel, ARM64, and Windows
were not exercised for this migration.

### Historical validation

The historical results below used earlier formats and ciphers. They are not
PoS2 1.0 qualifications.

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
Rust key shim. Use the job runner for a group starting at member index zero:

```sh
python3 contrib/pos2-pr118/group.py \
  build/tools/xchplot2/xchplot2 build-pr118/pos2_pr118_group \
  --farmer-pk "$FARMER_PK" --pool-ph "$POOL_PH" \
  --group-size 2 --k 28 --devices gpu0 --tier plain \
  --max-host-ram 18G --max-group-ram 512 --out /plots/experimental.gplot
```

`--pool-pk` is supported instead of `--pool-ph`. The default device is CPU;
GPU selection is explicit and never falls back to CPU. Key generation uses
released chia-protocol/chia-bls 0.50 for BLS and identity hashing, with
the V2 taproot hash SHA-256 of `(local_pk + farmer_pk) || farmer_pk` for both
pool-public-key and pool-contract modes. It generates one seed and memo per group, exposes the group
hash, and derives each member ID as SHA-256 of the group ID, big-endian u16
index, and u8 meta group. New jobs generate fresh random keys by default, so
repeating the same plotting parameters produces distinct group and member IDs.
`--seed` makes the group reproducible. k must be even
in 18..28; `--testnet` is explicitly rejected because PoS2 1.0 removes it.

Add `--files N` to write multiple files sharing one group ID and memo.
`--group-size` is the number of members in each file. For example,
`--out /plots/example.gplot --group-size 2 --files 3` writes six unique members
in `example.gplot`, `example-001.gplot`, and `example-002.gplot`. Each file
uses the same meta group and a distinct member-index range: 0–1, 2–3, and 4–5
in this example. `--plot-index` sets the first member's index (default 0);
the last member's index must be at most 65535. `--meta-group` stays explicit
and defaults to 0 for every file. It controls challenge scheduling and is
independent of how the group is divided into files. Each file is plotted,
assembled, and verified before starting the next, using the same per-file
memory limits. Raw member files are still retained.

The pinned upstream format has no stored starting index. Files with a nonzero
starting index therefore use experimental version **0x82**, with a little-endian
u16 starting index immediately after the existing 51-byte header and before
the memo. Their chunk encoding and local member slots are unchanged. The
[reader patch](plot-index-base.patch) maps those local slots to the stored
member indices when deriving IDs and returning proofs. It rejects index ranges
that exceed 65535. Files starting at zero still use upstream version 2.

Only `pos2_pr118_group` uses this patch; `pos2_pr118_check` retains the unmodified
upstream reader. Rebuild the grouped-job tool before using `--files` or
`--plot-index`. **Unmodified upstream readers and farmers cannot read the
0x82 extension.** Farmer support requires adopting the reader extension; these
files are experimental output, not qualified farming plots.

The runner publishes a private `.gplot.job.json` before plotting and keeps its
private raw directory beside each output. These contain secret plot keys.
On POSIX they use owner-only file/directory permissions. Each member runs
through the existing batch plotter. Assembly validates its ID, memo,
parameters, index, chunk bounds, decoded counts, sorted fragment ranges, and
membership before encoding. Deduplication is within each member only.
Regions must be nonempty for every member, as required by the pinned reader;
unsupported region/index encodings fail before publication.

Add `--resume` to the same command after interruption. The runner re-derives
and compares the saved group ID, memo and complete membership, checks existing
raw files before reusing them, and finishes missing members. For multiple
files, the first file's saved job records the file count and shared seed;
resume preserves that seed even if later files have not started. Keep all
saved jobs and raw directories, and resume with the original `--out`,
`--files`, `--plot-index`, and other identity parameters. A completed
group is checked against every raw fragment, then all returned qualities
from 32 deterministic challenges are solved and validated with their member
indices. Every member must yield a full proof. It writes and verifies a
temporary group, fsyncs it, publishes without overwriting another group, and
syncs the parent directory. Raw inputs are retained even after success.
Invalid existing outputs are rejected and preserved.
The saved job also records the reference's actual upstream Git revision plus
a SHA-256 content identifier for the upstream headers/SHA/FSE sources and
solver and member-index reader patches it uses.
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

Run the regression check (small pool-PK/pool-contract groups, multiple files
with shared keys and unique member IDs, a 64-member group, completed and
partial resume, corruption, private artifacts, error redaction, and RAM
rejection):

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

Historical (pre-1.0), 2026-09-30: the regression passed on CPU and RTX 4090 Plain GPU,
including every member of a 64-member k=18 group. A two-member k=28 GPU group
passed exact round-trip checks across all 4,194,304 grouped chunks and 86 full
proofs (43 per member). The complete GPU job took 59.60 s with an 18 GiB plotting
budget in a 22 GiB container. A separate host assembly plus verification took
50.75 s, peaked at 295,944 KiB RSS (289.01 MiB), and reproduced the published
group byte for byte under the 512 MiB modeled buffer cap. Solver/chainer memory
is included in that measured RSS, although excluded from the modeled cap.
These measurements cover this hardware, two members per k=28 file, and 64
members per k=18 file; larger k=28 files and other GPU tiers/vendors remain
unverified.

Historical (pre-1.0), 2026-10-01 with GCC 16.2 and RTX 4090 Plain: CPU/GPU k=18
regressions cover multiple files, distinct member IDs, resume, corruption,
and index 65535. Two-file jobs with two members per file at k=22 and k=28
passed CPU/GPU raw byte comparisons, complete grouped-fragment round-trips,
and 206 full proofs. The k=22 job used indices 65532–65535 and meta group 255;
the k=28 job used indices 0–3 and meta group 0. Unmodified readers rejected
the 0x82 files, and the patched reader rejected overflowing index ranges.

## Migration boundaries

The production writer retains the existing CPU reference boundary, shared
compression pool, exclusive temporary files, fsync, atomic publication, and
bounded resume checks. GPU kernels receive derived member IDs. Manifests
persist group IDs plus explicit `gplot-v2` or `raw-v2` format names; legacy
boolean/testnet manifests fail before plotting. Group verification uses each
returned member index for solving and full-proof validation. Raw-v2 `.plot2`
files are temporary assembler inputs, not the supported farming format.

Do not rename or convert old `.plot2` files or earlier experimental groups.
The Feistel change alters every plot's fragments, so these files and saved
jobs require replotting even where a group header still says version 2.
The experimental runner's recorded upstream source identity prevents resuming
an older job against this pin. Official groups start at member index zero;
0x82 files need the separate patched reader and remain experimental.
