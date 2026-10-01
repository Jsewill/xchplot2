# Contributing to xchplot2

Thanks for taking the time. A few notes to keep review loops short.

## Building and running tests

Install the branch's [build dependencies](INSTALL.md), then configure and
build all CMake targets:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The full test set requires the matching GPU backend. For host checks with
no GPU or GPU toolchain, configure the host-only CMake targets:

```bash
cmake -B build-host -S . -DXCHPLOT2_HOST_TESTS_ONLY=ON
cmake --build build-host --parallel
ctest --test-dir build-host --output-on-failure
```

`scripts/test/host-tests.sh` runs these same targets in a temporary build
and accepts `thread` or `address` for sanitizer runs.
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

## Install CI

`install-matrix` runs the dependency installer on fresh OS images, followed
by `cargo install --path . --locked`, a complete CMake CLI build, and the
existing CPU-safe plot/proof tests. It checks that the requested AdaptiveCpp
backend exists and that Intel's SPIR-V translator runs. No installed
toolchains are restored from cache in these jobs.

| Platform | NVIDIA | AMD | Intel |
| --- | --- | --- | --- |
| Ubuntu 24.04 | Yes | Yes | Yes |
| Debian 13 | Yes | Yes | No compute-runtime package in stable or backports |
| Fedora 44 | Yes | Yes | Yes |
| Arch rolling | Yes | Yes | Yes |
| Ubuntu 24.04 on WSL2 / Windows Server 2025 | Yes | Yes | Yes |

The same check is runnable locally with
`bash scripts/test/native-install.sh nvidia` (or `amd` / `intel`). It installs
system packages and builds AdaptiveCpp, just like the public installer.
Additional Fedora AMD and Ubuntu NVIDIA/Intel jobs pass `--no-acpp` and
exercise Cargo's automatic AdaptiveCpp build and install to `~/.local`.
Both the installed Cargo executable and the CMake executable run
`scripts/test/recovery.py`: real CPU plots interrupted by Linux SIGINT and
SIGTERM, saved manifest and completed-file preservation on resume, forced
publication failure, partial cleanup, and full proofs. Windows uses the same
harness with Ctrl-Break and keeps its Unicode benchmark checks.
CI runs on PRs, `main` pushes, manual dispatch, and weekly to catch package
repository changes. The existing container and CUDA architecture matrices
remain separate coverage. The [binary release job](#binary-releases) checks
native Windows separately.

These are build and install checks, not GPU driver or hardware certification.
Hosted WSL2 jobs also lack GPUs; actual WSL GPU support depends on the card
and its Windows driver. Vendor kernel execution and production plotting use
the GPU suites below.

## GPU CI

Hosted PR jobs lint and build without GPU hardware. `GPU hardware` runs only
on trusted `main` / `cuda-only` pushes, schedules, and manual dispatches:

Register runners and set `GPU_CORRECTNESS_MATRIX` to the installed fleet before
setting the Actions repository variable `GPU_RUNNERS_READY` to `true`.
That enables automatic push and daily runs. Without a correctness matrix,
the original NVIDIA, AMD, Intel, and `cuda-only` lanes remain the default.
Automatic weekly physical runs additionally require `GPU_PHYSICAL_MATRIX`;
manual physical dispatch retains the default matrix below. Until readiness
is enabled, automatic GPU jobs are skipped. Manual dispatch requires matching
runners.

| Suite | When | Checks |
| --- | --- | --- |
| `quick` | Branch pushes; manual | All CTest tests, then k=18 CPU byte parity and 100 full-proof challenges for every tier and disk-spill variant |
| `correctness` | Manual; shared GPU | All CTest tests, then k=28 CPU byte parity and full proofs for every tier and spill variant, with the production VRAM caps |
| `vram` | Daily, 04:17 UTC; manual | Quick CTest tests, three k=28 plots at each tier's budget, rejection 1 MiB below it, then k=28 CPU byte parity and full proofs for every tier and spill variant |
| `physical` | Sunday, 07:47 UTC; manual | Actual 2/4/6/8 GiB capacity, every k=28 tier that fits, plus three uncapped auto-tier plots with full-proof verification |

`quick` and `correctness` record device-wide memory measurements as diagnostics.
Other applications can change those counters, so they do not establish the
plotter's memory use on a shared GPU. Allocation and admission guards remain
active. `vram` and `physical` enforce the device-wide memory assertion and
require an isolated GPU; the summary records which policy ran. A correctness
pass does not qualify memory usage.

Every suite also compares two k=18 mainnet vectors against CPU output on each
selected tier and spill variant: strength 3 with an all-`ff` ID, maximum
index/meta group (65535/255), and a patterned 255-byte memo; strength 4 with an
ID containing bytes `00` through `1f` and index/meta group 1/1. Each path must
report positive solved full proofs. The summary records vector parameters,
reference hashes, and proof counts. Production k=28 checks and original log
names remain intact. Existing kernel parity tests cover testnet; the legacy
file header does not select testnet parameters for full-file verification.

By default, the default branch schedules both `main` and `cuda-only`, because
[GitHub schedules run only on the default branch](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
Pushes to `cuda-only` also run that branch's quick suite directly.
A configured fleet matrix can narrow this coverage. Both branches must contain
the GPU CI changes before enabling a schedule that tests both.

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

The default physical matrix covers these four NVIDIA capacities plus AMD 4 GiB
and Intel 8 GiB on `main`, and all four NVIDIA capacities on `cuda-only`.
Set `GPU_PHYSICAL_MATRIX` to the installed fleet to enable weekly physical
runs, including AMD and Intel:

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

For a fleet containing only an NVIDIA 24 GiB card with about 32 GiB host RAM,
start with this `GPU_CORRECTNESS_MATRIX` value:

```json
{"include":[{"backend":"cuda","ref":"main","runner":"gpu-cuda","max_host_ram_gib":18}]}
```

This enables NVIDIA `main` coverage only. Add other branches and vendors after
their runner toolchains and hardware pass; leave `GPU_PHYSICAL_MATRIX` unset
until real small cards are installed. Each matrix entry can specify
`max_host_ram_gib`; zero or omission uses the plotter's default host budget.
The explicit budget survives the harness's environment cleanup and applies
to boundary and normal plots; spill cases still use `min`.

On a designated Linux GPU host, download and verify the runner using the
repository's **Settings → Actions → Runners → New self-hosted runner**
instructions. Install it in a separate directory owned by the runner user,
with no personal credentials. For one supervised job, use the fresh
registration token from that page and run from that directory:

```bash
./config.sh --url https://github.com/Jsewill/xchplot2 \
    --token "$RUNNER_REGISTRATION_TOKEN" --unattended --ephemeral \
    --name xchplot2-cuda --labels gpu-cuda --work work
systemd-run --user --unit=xchplot2-gpu-runner --collect --wait --same-dir \
    --property=MemoryMax=22G --property=MemorySwapMax=0 \
    --property=CPUQuota=400% --setenv=CUDA_VISIBLE_DEVICES=0 \
    --setenv=PATH="$PATH" "$PWD/run.sh"
```

The 18 GiB plot budget bounds modelled allocations; the 22 GiB cgroup limit
separately bounds the whole runner and its child processes, with no swap.
Keep `RUNNER_TEMP` on real disk with space for the build and at least 40 GiB
of plotting scratch. The workflow places plots and spills in its isolated
temporary directory. Stop this transient runner with
`systemctl --user stop xchplot2-gpu-runner.service`.

[Ephemeral runners](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#ephemeral-runners-for-autoscaling)
accept one job and automatically deregister. Preserve their `_diag` logs,
then provision a clean runner for the next job. A regular schedule needs that
runner lifecycle supplied by the designated fleet operator; a single launch
does not provide continuing capacity. Confirm a manual `quick` and `vram`
run and retained artifacts before enabling `GPU_RUNNERS_READY`. Each job
still checks its actual backend and capacity, and missing vendors remain
uncertified.

`gpu_ci_info` obtains each k=28 floor from the production backend, including
sort scratch, and records physical/free VRAM and the safety margin. The
boundary script requires a positive driver-reported peak within the allowed
budget, independently of allocator bookkeeping. `POS2GPU_VRAM_MARGIN_MB`
remains available for driver/hardware calibration; record why a runner needs
a different value. Other plotting overrides and self-test bypasses are
removed by the harness.

Run the same checks locally after building all CMake targets:

Temporary plots and spill files use the checkout's filesystem by default.
Use `--max-host-ram-gib 18` to budget 18 GiB for boundary and normal tier
plots; explicit spill cases still use `min`. This limits modelled pinned and
anonymous allocations, not total RSS. A container limit can separately bound
total memory use.

Use `--scratch /path/to/disk` to select another existing directory. RAM-backed
filesystems such as tmpfs cannot validate disk spilling and are rejected by
the plotter.

```bash
python3 scripts/test/gpu-ci-test.py
python3 scripts/test/gpu-ci.py build --backend cuda --suite quick --logs /tmp/gpu-quick
python3 scripts/test/gpu-ci.py build --backend cuda --suite correctness --logs /tmp/gpu-correctness
python3 scripts/test/gpu-ci.py build --backend hip --suite vram --logs /tmp/gpu-vram
python3 scripts/test/gpu-ci.py build --backend level_zero --suite physical --physical-vram-mib 8192 --logs /tmp/gpu-physical
```

Actions retain build logs, device inventory, CTest JUnit results, per-tier
boundary/rejection logs, proof logs, and a JSON summary for 14 days. Generated
plots are temporary; byte comparisons and reference hashes are recorded.
The summary records failed stages, exception messages, and subprocess command
and exit code when available, including archive and inventory failures.
NVIDIA runs additionally retain best-effort driver XML snapshots at suite
start and failure, including process-memory evidence. Watchdog measurements
are sampled device-wide changes in free memory; they cannot attribute memory
to this process, and endpoint snapshots can miss transient external usage.
Use `correctness` on a shared desktop. Reserve strict `vram` and `physical`
qualification for isolated GPUs; retrying until a shared-device counter passes
does not establish memory attribution.

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

The release workflow builds Linux archives and experimental Windows ZIPs
for x86-64 and ARM64 through the standalone CMake executable and CPack.
The x86-64 Linux image pins Ubuntu 24.04, AdaptiveCpp
25.10, Rust 1.98.1, CUDA 12.9.1, ROCm 7.1.1, and LLVM 20, with Level Zero.
ROCm 7.1.1 provides the LLVM 20 runtime compiler needed by the shared generic
build; the former AMD-only archive used ROCm 6.2 with LLVM 18.
Keep `INSTALL.md` and the archive README in sync with these pins.

Build the Linux archive locally:

```bash
podman build -t xchplot2-release -f ci/release/Containerfile ci/release
podman run --rm -v "$PWD:/src" xchplot2-release bash scripts/build-release.sh
```

The Linux release matrix builds natively on x86-64 and ARM64 and checks
each extracted archive on its matching runtime. On ARM64, build the same
Containerfile with `--build-arg BASE_IMAGE=ubuntu:26.04`; Ubuntu 26.04 supplies
ARM64 HIP packages, and the image builds the Level Zero loader from source.
CUDA 12.9's math declarations are adjusted for the newer glibc so the
release retains pre-Turing code generation.
Both architectures include CUDA, HIP, and Level Zero; ARM64 also includes
OpenCL. The ARM64 archive therefore has a newer glibc baseline. Hosted
checks do not qualify GPU drivers or hardware.

Artifacts are written to `build/release-linux/dist/`. `acpp --acpp-deploy`
collects CPU, CUDA, and HIP runtime/JIT dependencies; the build script adds
Level Zero, checks all three GPU backends, collects license notices, and
makes library paths relative.

PR and manual runs retain workflow artifacts. Manual runs can select one
platform; PR and tag runs build both. A `vVERSION` tag creates a
draft GitHub release; publish it after qualifying the extracted archives on
the supported GPUs. Do not rebuild between qualification and publication.
The Linux job checks extraction, bundled libraries, offline AMD and Intel
kernel compilation, the packaged SYCL JIT through `hellosycl`, CPU/SYCL plot
byte parity, and full proofs in an image without development toolchains.

The Windows x86-64 job uses VS 2022, LLVM/Clang 20.1.8, CUDA 12.9.1, and HIP
SDK 6.4.2. ARM64 uses Visual Studio's 14.44 tools and CUDA 13.4.2, with CUDA,
Level Zero, and OpenCL backends; AMD's Windows HIP SDK has no ARM64 runtime.
Both build AdaptiveCpp into LLVM using
`ci/release/build-adaptivecpp-windows.ps1`, including the Level Zero loader
and LLVM-SPIRV translator. The cached install tree is invalidated when
compiler sources or options change. `scripts/build-release.ps1` collects
runtime DLLs and notices, builds all targets, runs the host checks, and
writes one combined `build/release-windows/dist/*-windows-ARCH-sycl.zip`.
It also requires `cargo-about` 0.9.2 and the toolchain in `ACPP_PREFIX`.
The toolchain applies `contrib/adaptivecpp-windows-hip.patch` for upstream
device-IR fixes and `contrib/adaptivecpp-windows-level-zero.patch` for
Windows headers, linking, DLL installation, and integrated SPIR-V builds.
ARM64 also applies `contrib/adaptivecpp-windows-opencl.patch` for the backend
DLL's install location and builds Khronos's OpenCL loader. Toolchain caches
are separate for each host architecture. The archive tests check native
executable architecture and run without the development SDKs.
The ARM64 runner's side-by-side 14.44 toolset keeps its STL compatible with
LLVM 20; the default Visual Studio 2026 STL needs a newer Clang.
Windows uses `generic;omp`: generic GPU kernels and precompiled CPU kernels,
because the Windows CPU JIT needs Visual Studio static CRT libraries.
The archive test hides the compiler and both GPU SDK installations, clears
SDK paths, loads bundled DLLs and LLVM tools, compiles an AMD `gfx1031`
kernel, translates Intel SPIR-V, dispatches a CPU kernel, and runs the
shared CPU recovery check in `scripts/test/recovery.py`. CUDA and HIP backend DLL loading
additionally requires their graphics drivers, so those two checks run only
when their drivers are installed. The ZIP includes the compiler and
redistributable runtime dependencies; users need only their graphics driver.
These checks do not qualify Windows GPU execution.

Run `scripts/test/release.py ARCHIVE.tar.gz` (or `ARCHIVE.zip` on Windows)
for the archive and CPU checks. It also compares the SYCL plotting pipeline
on OpenMP against the CPU reference, covering CUDA-enabled builds without
an NVIDIA driver.
Add `--sycl-probe build/release-linux/tools/sanity/hellosycl` to test kernel execution
(use `build/release-windows/tools/sanity/hellosycl.exe` on Windows).
For GPU qualification on Linux, run the existing GPU suite against the
exact archive and its matching release build:

```bash
python3 scripts/test/gpu-ci.py build/release-linux --backend cuda --suite quick \
    --archive build/release-linux/dist/ARCHIVE.tar.gz --logs /tmp/release-cuda
```

The adjacent `ARCHIVE.tar.gz.sha256` must match, and the archive's
`BUILDINFO.txt` must equal the build's file before any inventory or test
executable runs. Keep the original release build's inventory and parity tools;
a separately compiled build with a different source revision or toolchain
cannot qualify the archive. The packaged runtime libraries take precedence
over installed toolchain libraries for these checks.
Use `correctness` for k=28 archive byte comparisons and proofs on a shared GPU,
and record memory qualification separately. Run the `vram` and `physical` suites
on isolated supported GPUs as described above
for k=28, every fitting tier, and spill variants. Perform the required k=22
byte comparisons and full proofs with the same extracted archive as well.
The `--binary` option remains
available for an already extracted executable, but does not record archive
identity or enforce the matching `BUILDINFO.txt` check.

Attach `summary.json` and relevant inventory, parity, boundary, and proof logs
to the release qualification notes. Archive runs record its name, SHA256,
`BUILDINFO.txt`, UTC start time, host platform, and pass status in the summary.
Keep generated plots out of the tree. These commands require GPU hardware;
there is no automatic release GPU gate until the required runners are configured.

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
branch is the original CUDA-only path, preserved as the most-tested
NVIDIA configuration. A PR that only helps NVIDIA may still land on
`main`, but don't regress parity on AMD (`gfx1031`) along the way.

## Reporting bugs

Open an issue with:

- Exact command line and the full stderr output.
- GPU vendor + model + VRAM (`nvidia-smi -L` / `rocminfo | grep gfx`).
- Build flavor: container (service name + `ACPP_GFX` / `CUDA_ARCH`),
  native `scripts/install-deps.sh`, or `cargo install`.
- Whether parity tests pass on your build.
