# Contributing to xchplot2

This branch uses native CUDA.

## Building and running tests

Install the branch's [build dependencies](INSTALL.md), then configure and
build all CMake targets:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The full test set requires the matching GPU backend. For host checks with
no GPU or GPU toolchain, run `scripts/test/host-tests.sh`.
`xchplot2 parity-check --dir build/tools/parity` runs the available
`*_parity` and `*_test` executables and reports each failure's output.

The parity binaries under `tools/parity/` are the correctness gate:

- AES, Xs, T1, T2, and T3 tests check agreement with pos2-chip's CPU reference.
- Backend-specific sort and bucket tests cover vendor kernels and scratch.
- `plot_file_parity` covers the writer/reader round-trip.
- Host tests cover memory budgets, spill storage, recovery, and reporting.

Kernel, sort, and plot-format changes must pass parity at k=22 and k=28.
Output bytes must remain identical to the pinned CPU reference. The
[GPU suites](#gpu-ci) cover tiers, spill variants, and capacity boundaries;
passing a build or a software cap does not certify other physical hardware.

After a functional change, spot-check a real output with full proofs:

```bash
xchplot2 verify /path/to/output.plot2 --full --trials 100
```

Default `verify` samples quality chains; `--full` also reconstructs and
validates full proofs. An empty sample fails. Sampling does not inspect every
part of a file, so use a matching CPU output for byte parity. For example,
this synthetic testnet fixture uses the same ID, memo, and plot parameters:

```bash
PLOT_ID=$(printf 'ab%.0s' {1..32})
MEMO=$(printf '00%.0s' {1..112})
xchplot2 test 28 "$PLOT_ID" 2 0 0 -T -m "$MEMO" -o ref -N ref.plot2
printf '28 2 0 0 1 %s %s out gpu.plot2\n' "$PLOT_ID" "$MEMO" > m.tsv
xchplot2 batch m.tsv --tier tiny
sha256sum ref/ref.plot2 out/gpu.plot2
```

The hashes must match. Use a tier and spill configuration appropriate to
the changed path, and a real disk with `--temp-dir` for spill checks.
These synthetic fixtures are not farmable plots.

## Architecture

```text
src/gpu/                  GPU kernels and sort backends
src/host/
├── GpuPipeline           Xs → T1 → T2 → T3 orchestration
├── GpuBufferPool         persistent device buffers and host drain slots
├── BatchPlotter          workers, memory admission, and writer queues
├── BatchManifest         saved identities and recovery manifests
└── PlotFileWriterParallel  CPU reference boundary, writer, and verification
tools/xchplot2/           CLI and shell completions
tools/parity/             parity and host tests
keygen-rs/                BLS key derivation, plot IDs, and memo encoding
```

Memory responsibilities are split between [VramBudget.hpp](src/host/VramBudget.hpp)
for base tier peaks, `GpuBufferPool` for backend allocation requirements,
and `BatchPlotter` for worker admission and host spill policy. Each worker
must fit its own budget. Keep the driver watchdog and allocation accounting
consistent; an allocation trace alone is not a physical VRAM measurement.

Pool buffers persist across plots. Streaming tiers progressively tile work
and keep more intermediate data on the host. Pinned scratch can be reused
only after its previous consumer is finished. Writer/drain queues bound
the number of in-flight plots and prevent early buffer reuse.

The user-facing tier models are in [REFERENCE.md](REFERENCE.md#memory-requirements).
Measured throughput and memory use are in
[BENCHMARKS.md](BENCHMARKS.md).

## GPU CI

Hosted PR jobs lint and build without GPU hardware. `GPU hardware` runs only
on trusted `main` / `cuda-only` pushes, schedules, and manual dispatches:

After registering the runners below, set the Actions repository variable
`GPU_RUNNERS_READY` to `true` to enable automatic push and scheduled runs.
Until then, automatic GPU jobs are skipped. Manual dispatch remains available
and requires the matching runners.

| Suite | When | Checks |
| --- | --- | --- |
| `quick` | Branch pushes; manual | All CTest tests, then k=18 CPU byte parity and 100 full-proof challenges for every tier and disk-spill variant |
| `vram` | Daily, 04:17 UTC; manual | Quick CTest tests, three k=28 plots at each tier's budget, rejection 1 MiB below it, then k=28 CPU byte parity and full proofs for every tier and spill variant |
| `physical` | Sunday, 07:47 UTC; manual | Actual 2/4/6/8 GiB capacity, every k=28 tier that fits, plus three uncapped auto-tier plots with full-proof verification |

The default branch schedules both `main` and `cuda-only`, because
[GitHub schedules run only on the default branch](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
Pushes to `cuda-only` also run that branch's quick suite directly.
Both branches must contain the GPU CI changes before enabling the schedule.

The SYCL branch tests five tiers and four disk variants. Native CUDA tests
four tiers, Compact's spill engine, and Minimal's file mappings; native Tiny
cannot spill its GPU-visible host tables.

Provide Linux x64 self-hosted runners with these additional labels:

| Label | Hardware / toolchain |
| --- | --- |
| `gpu-cuda` | NVIDIA, CUDA toolkit, AdaptiveCpp with its CUDA backend; enough free VRAM for every k=28 tier |
| `gpu-hip` | AMD, ROCm, AdaptiveCpp with its HIP backend; enough free VRAM for every k=28 tier |
| `gpu-level-zero` | Intel, Level Zero runtime and Sysman, AdaptiveCpp with its Level Zero backend and LLVM SPIR-V translator; enough free VRAM for every k=28 tier |
| `gpu-cuda-2gb`, `gpu-cuda-4gb`, `gpu-cuda-6gb`, `gpu-cuda-8gb` | Physical NVIDIA cards with those capacities; use CUDA 12.9 for Pascal cards |
| `gpu-hip-4gb`, `gpu-level-zero-8gb` | Physical 4 GiB AMD and 8 GiB Intel cards with the same vendor toolchains as above |

The weekly physical matrix covers these four NVIDIA capacities plus AMD 4 GiB
and Intel 8 GiB on `main`, and all four NVIDIA capacities on `cuda-only`. Set the Actions variable
`GPU_PHYSICAL_MATRIX` to match a different fleet, including AMD and Intel:

```json
{"include":[{"backend":"cuda","ref":"main","runner":"my-2gb-nvidia","vram_mib":2048},{"backend":"hip","ref":"main","runner":"my-4gb-amd","vram_mib":4096},{"backend":"level_zero","ref":"main","runner":"my-8gb-intel","vram_mib":8192},{"backend":"cuda","ref":"cuda-only","runner":"my-2gb-nvidia","vram_mib":2048}]}
```

Keep entries for the capacities and vendors you intend to certify. Missing
runner labels leave jobs queued; no GPU or a wrong backend fails the job.
Physical capacity comes from the device inventory, so a software-capped 24 GiB
GPU cannot pass a 2 GiB lane. A physical card with no fitting k=28 tier fails.

Install the project's build dependencies before registering runners: CMake
3.24+, a supported C++ compiler, Rust, Python 3.11+, and the matching GPU
toolchain. `scripts/install-deps.sh --gpu nvidia|amd|intel` covers the SYCL
build; follow [INSTALL.md](INSTALL.md) for vendor driver setup. Pin runner images and
toolchain versions, and update them deliberately. Put custom toolchain paths
in the runner service environment (`PATH`, `CMAKE_PREFIX_PATH`, `CUDACXX`,
`CXX`, `CUDAHOSTCXX`, `LD_LIBRARY_PATH` as needed).

Use dedicated ephemeral runners with one visible GPU per job and sufficient
host RAM and fast scratch storage; 64 GiB host RAM and 40 GiB scratch are a
useful starting point for the k=28 suites. Configure device visibility in the
runner environment (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, or
`ZE_AFFINITY_MASK`). Jobs set the AdaptiveCpp backend mask themselves and fail
if zero or multiple GPUs remain visible. No PR workflow targets these hosts.
The harness serializes GPU tests per host; other GPU workloads, including a
desktop, can still inflate driver memory measurements.

`gpu_ci_info` obtains each k=28 floor from the production backend, including
sort scratch, and records physical/free VRAM and the safety margin. The
boundary script requires a positive driver-reported peak within the allowed
budget, independently of allocator bookkeeping. `POS2GPU_VRAM_MARGIN_MB`
remains available for driver/hardware calibration; record why a runner needs
a different value. Other plotting overrides and self-test bypasses are
removed by the harness.

Run the same checks locally after building all CMake targets. This branch
uses `--backend cuda`; HIP and Level Zero commands apply to `main`:

Temporary plots and spill files use the checkout's filesystem by default.
Use `--scratch /path/to/disk` to select another existing directory. RAM-backed
filesystems such as tmpfs cannot validate disk spilling and are rejected by
the plotter.

```bash
python3 scripts/test/gpu-ci-test.py
python3 scripts/test/gpu-ci.py build --backend cuda --suite quick --logs /tmp/gpu-quick
python3 scripts/test/gpu-ci.py build --backend hip --suite vram --logs /tmp/gpu-vram
python3 scripts/test/gpu-ci.py build --backend level_zero --suite physical --physical-vram-mib 8192 --logs /tmp/gpu-physical
```

Actions retain build logs, device inventory, CTest JUnit results, per-tier
boundary/rejection logs, proof logs, and a JSON summary for 14 days. Generated
plots are temporary; byte comparisons and reference hashes are recorded.

## Pinned testnet farming fixture

`contrib/testnet-farming.patch` targets chia-blockchain commit `39f8bec88`
(2.7.0 Checkpoint Merge). It fixes the v2 service wiring, proof challenge,
and dependency issues present at that revision. This fixture is not a claim
about the current state of upstream farming support.

```bash
git clone https://github.com/Chia-Network/chia-blockchain
cd chia-blockchain
git checkout 39f8bec88
git apply /path/to/xchplot2/contrib/testnet-farming.patch
```

The patch header explains its changes. The separate
[pos2-chip PR #118 compatibility notes](contrib/pos2-pr118/README.md) record
the proposed grouped format and the pinned revision checked for it.

## Documentation checks

README is the short entry point. Keep installation recipes in `INSTALL.md`,
operating options in `REFERENCE.md`, and dated measurements in `BENCHMARKS.md`.
Document each branch's actual behavior; native CUDA and SYCL have different
memory budgets and spill support. Record GPU, source revision, sample count,
warmup, units, and measurement date with new results. Label older results
whose provenance is incomplete.

The existing Markdown job lints tracked documentation, including section
fragments. Local links are checked without contacting external websites:

```bash
python3 scripts/test/docs.py
```

When moving sections, update incoming links and keep the useful README
entry headings. `docs/` is ignored local material; do not publish it as part
of a documentation move.

## Binary releases

The release workflow builds the standalone CMake executable in
`ci/release/Containerfile`: Ubuntu 22.04, CUDA 12.9.1, CMake 3.28.3, and
Rust 1.98.1. CUDA targets are explicit in `scripts/build-release.sh`; keep
the compatibility requirements in `INSTALL.md` and the archive README in
sync when changing them. `cargo-about` collects the Rust dependency licenses
and fails on unresolved licenses.

Build the same archive locally with Docker or Podman:

```bash
podman build -t xchplot2-release -f ci/release/Containerfile ci/release
podman run --rm -v "$PWD:/src" xchplot2-release bash scripts/build-release.sh
```

The Linux archive and its SHA-256 checksum are written to `build/release/dist/`.
For native Windows, install the [Windows build tools](INSTALL.md#windows)
and PowerShell 7.3+, then run:

```powershell
rustup toolchain install 1.98.1 --profile minimal
rustup default 1.98.1
cargo install --locked --features cli cargo-about --version 0.9.2
./scripts/build-release.ps1
python scripts/test/release.py build/release-windows/dist/xchplot2-0.12.0-windows-x86_64-cuda.zip
```

The PowerShell script loads the Visual Studio 2022 x64 environment when
needed, builds all targets with CUDA 12.9.1, runs the host CTest subset, and
writes a ZIP and checksum to `build/release-windows/dist/`. The extracted
Windows check also exercises Unicode paths, real key generation, Ctrl-Break,
resume, and publication failure. CI runs it with toolkit libraries removed
from `PATH`. Windows GPU plotting and spill behavior need qualification on
Windows hardware before the archive is advertised for those devices.
For affected CUDA 12.x headers, CMake applies NVIDIA's
[64-bit PTX operand fix](https://github.com/NVIDIA/cccl/commit/270f4100dceeb6345f74fd374695e78bb0a48082)
to a build-local copy; `BUILDINFO.txt` records the backport. The installed
toolkit stays intact.

PR and manual runs retain both platforms' archives as workflow artifacts. Pushing a
`vVERSION-cuda-only` tag creates a draft GitHub release; publish it after
qualifying the extracted archive on the supported GPUs. Do not rebuild
between qualification and publication.

The workflow runs host tests and tests the extracted archive on Ubuntu 22.04
without a GPU toolkit. Run `scripts/test/release.py ARCHIVE.tar.gz` to repeat
the archive, CPU plotting, and full-proof check. For GPU qualification, use
the packaged executable for k=22 and k=28 CPU byte comparisons, full proofs,
and the tier, spill, and recovery checks described above. Record qualification
in the release notes; keep generated plots and detailed logs out of the tree.
`gpu-ci.py --binary /path/to/extracted/bin/xchplot2` uses the package for
plotting and verification while retaining the build's parity and inventory
tools. Both must come from the same source revision and toolchain.

## Commit style

Short imperative subjects, lowercase scope prefix, no trailing period:

```
gpu: split xs-sort keys_a to d_storage tail — drops pool VRAM min ~1.3 GB
docs: tighten streaming peak (~7.3 GB measured), add AMD row
CMakeLists: re-enable -O3 for SYCL TUs
```

Body paragraphs explain *why* (what invariant was wrong, what the
measurement was, what alternative was considered and why it was
rejected). The *what* is in the diff.

## Scope of changes

- Keep unrelated refactors out of correctness or performance commits.
- Performance changes should cite before/after numbers on a named GPU
  at a specified `k`.
- New runtime knobs go in `REFERENCE.md`'s
  [environment table](REFERENCE.md#environment-variables) so users can discover them.

## PRs

The `main` branch carries the SYCL/AdaptiveCpp port; the
[`cuda-only`](https://github.com/Jsewill/xchplot2/tree/cuda-only)
branch is the native CUDA path for NVIDIA. Keep shared CLI, recovery,
and documentation behavior aligned where it applies, and validate backend
changes on the branch's supported hardware.

## Reporting bugs

Open an issue with:

- Exact command line and the full stderr output.
- GPU model and VRAM (`nvidia-smi -L`).
- Build flavor: container (`CUDA_ARCH`), Cargo, or CMake; include toolkit
  and driver versions.
- Whether parity tests pass on your build.
