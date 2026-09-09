# xchplot2 reference (`main`)

[README](README.md) · [Installation](INSTALL.md) · [Benchmarks](BENCHMARKS.md)

[Plotting and recovery](#plotting-and-recovery) · [Configuration](#configuration-and-argument-files) ·
[Devices](#devices-and-cpu-workers) ·
[Benchmarking](#benchmarking) · [Memory](#memory-requirements) ·
[Disk-offload](#host-ram-and-disk-offload) · [Other commands](#lower-level-subcommands) ·
[Environment variables](#environment-variables)

## Plotting and recovery

```bash
xchplot2 plot -k 28 -n 10 \
    -f <farmer-pk> \
    -c <pool-contract-address> \
    -o <output-dir>
```

| Option | Meaning |
|---|---|
| `-k`, `--k K` | Plot size; default 28 |
| `-n`, `--num N` | Number of plots; default 1 |
| `-s`, `--strength S` | Proof strength; default 2 |
| `-f`, `--farmer-pk HEX` | Farmer public key, 96 hex characters |
| `-p`, `--pool-pk HEX` | Pool public key, 96 hex characters |
| `--pool-ph HEX` | Pool puzzle hash, 64 hex characters |
| `-c`, `--pool-contract-address ADDRESS` | Pool contract address, `xch1...` or `txch1...` |
| `-o`, `--out DIR` | Output directory; default current directory |
| `-i`, `--plot-index N` | Starting plot index; default 0, incremented per plot |
| `-g`, `--meta-group N` | Meta-group field; default 0 |
| `-S`, `--seed HEX` | Optional 64 hex characters for reproducible identities |
| `-T`, `--testnet` | Use testnet proof parameters |
| `-v`, `--verbose` | Print additional worker and plotting details |

Supply the farmer key and one of the pool key, puzzle hash, or contract
address forms. See [plot indices and meta groups](#plot-indices-and-meta-groups)
for the grouping distinction. Full help: `xchplot2 --help`.

On a host that is short of RAM for the tier its GPU lands on, add
`--temp-dir <path>` to choose where the automatic disk-offload writes
(`--max-host-ram` to bound it explicitly, `--no-auto-spill` to turn it
off) — see [Host RAM and disk-offload](#host-ram-and-disk-offload).

Before plotting starts, `plot` saves the prepared identities and keys in an
`xchplot2-job-*.tsv` manifest in the output directory. Use `--manifest FILE`
to choose its location. Each new job keeps its own manifest; another job
cannot overwrite it. Manifests contain private plot keys and are created
with owner-only permissions on Linux.

Repeat the same `plot` command with `--resume` (or `--skip-existing`) to
recover its saved job, including when no `--seed` was supplied. If several
saved jobs match, select one with `--manifest FILE`. The original keys,
plot parameters, count, and output directory must match; devices and memory
settings can change. A fixed `--seed` still reconstructs the same identities.
An ordinary run without `--resume` starts a new job.

Alternatively, resume directly from the saved manifest without repeating
the key or plot arguments:

```bash
xchplot2 batch /path/to/job.tsv --resume
```

Resume skips files only after checking their header identity, memo, chunk
index, and file bounds. `--continue-on-error` logs per-plot failures and
continues; both options work in `plot` and `batch` modes. Manifest paths may
be double-quoted to include spaces, quotes, or backslashes; line breaks in
paths are unsupported.

An aggregate progress line updates after each plot completes:

```
[batch] progress: plot 3/10 done (30.0%, 2.41 s/plot avg, 0.000373 TiB/s, fully plotted in ~17s)
```

On a terminal it rewrites itself in place and is on by default; when
stderr is redirected it prints one line per plot and defaults to off.
`--progress` / `--no-progress` force it either way. The ETA is
plots-based (average s/plot × remaining), formatted as
hours/minutes/seconds when the estimate exceeds one hour.

`-q`/`--quiet` suppresses info-level stderr output — the progress line,
end-of-run summaries, and streaming-tier notes. Warnings and errors
still print. `plot` writes each absolute output path to stdout after that
file is published or validated by resume, in completion order. Failed or
unstarted plots are never listed, and completed paths remain available if
a later plot fails. Exit status is 0 for completion, 1 for argument errors,
2 for an exception, 3 for per-plot failures, or 4 for unfinished work.
`-q` and `-v` are mutually exclusive.

Plots are written to an exclusively created `<name>.plot2.partial.XXXXXX`
file, flushed through its original file descriptor, and atomically renamed on
completion. The final name is published only after writing succeeds.
A first `Ctrl-C` asks the plotter to
finish the plot in flight and stop; a second hard-kills. Concurrent writers
use separate temporary files; the last completed rename wins. Resume checks
the header identity, memo, chunk index, and file bounds before skipping a plot.

### Batch manifests

`batch` consumes the manifest saved by `plot`, or one prepared with existing
plot identities. Device, tier, memory, progress, and recovery options work
as they do for `plot`:

```bash
xchplot2 batch /path/to/job.tsv --devices gpu --resume
```

Each non-comment line has nine whitespace-separated fields in this order:

```text
k strength plot_index meta_group testnet plot_id_hex memo_hex out_dir out_name
```

`plot_id_hex` is 64 hex characters; `memo_hex` encodes up to 255 bytes
(`""` represents an empty memo). `testnet` accepts `0`, `1`, `false`, or
`true`. Memo and path fields accept double quotes, with backslash escapes
for quotes and backslashes. `out_name` must be a filename, not a path;
relative `out_dir` paths resolve from the working directory. Blank lines
and `#` comments are ignored. Invalid rows are rejected before plotting.
Prefer the automatically saved manifest for recovery; see
[Security](SECURITY.md) before sharing manifests containing private keys.

## Configuration and argument files

`--config FILE` loads a configuration file. Without it, xchplot2 looks for
`$HOME/.config/xchplot2/config.toml`. The supported syntax is a small TOML
subset: named sections and scalar `key = value` entries, with double-quoted
strings and `#` or `;` comments. Arrays, nested tables, and multiline strings
are unsupported.

Use long option names without `--`. `[defaults]` applies to every command;
a section such as `[plot]` or `[bench]` overrides those defaults. Explicit
command-line flags override the same options from the file. Put options
that only apply to one command in that command's section:

```toml
[plot]
out = "/mnt/plots"
num = 10

[bench]
k = 28
num = 10
warmup = 2
devices = "gpu"
keep = false
```

Save this as `plotter.toml`, then supply the remaining arguments normally:

```bash
xchplot2 plot --config plotter.toml -f <farmer-pk> -c <pool-contract-address>
xchplot2 bench --config plotter.toml -o /scratch --num 3
```

### Argument files

`@FILE` inserts whitespace-separated arguments from a file. For example,
save these lines in `bench.args`:

```text
# Reusable benchmark options
--k 28
--num 10
--warmup 2
--devices gpu
```

```bash
xchplot2 bench @bench.args -o /scratch --num 3
```

Later flags override earlier values. `@~/bench.args` expands the leading
`~/` in the argument-file path. Inside the file, `#` starts a comment;
shell quoting, variable expansion, and nested argument files are unsupported.
Use the configuration file or directly quoted CLI arguments for paths with
spaces. Pass `--config` directly on the command line, not inside an argument file.

## Plot indices and meta groups

Both are v2 PoS fields and default to 0.
`<plot-index>` (u16) is the within-group identifier; `plot -n N`
uses it as the base and increments per plot (so `-i 0 -n 1000`
produces plots with `plot_index` 0..999).
`<meta-group>` (u8) is a challenge-isolation boundary — plots with
different meta_group values are guaranteed never to pass the same
challenge.

The grouped-plot format is proposed in
[pos2-chip PR #118](https://github.com/Chia-Network/pos2-chip/pull/118).
xchplot2 currently produces one `.plot2` file per plot using its existing
dependency pin. A group must share its group ID and memo; `plot -n N`
currently generates independent keys for each plot, so incrementing the
index alone does not form a group. The
[opt-in compatibility check and migration plan](contrib/pos2-pr118/README.md)
track preparation for the proposed format.

## Devices and CPU workers

`xchplot2 devices` lists the IDs accepted by `--devices`. With no selection,
plotting uses one GPU. CPU plotting is opt-in and uses pos2-chip's CPU
implementation. Combine selectors with commas:

| Selector | Workers |
|---|---|
| `0` or `gpu0` | One GPU |
| `gpu` | Every visible GPU |
| `cpu` | Every CPU NUMA node |
| `cpu1` | One CPU NUMA node |
| `all` or `gpu,cpu` | Every GPU and CPU NUMA node |

```bash
xchplot2 plot ... --devices gpu
xchplot2 plot ... --devices 0,2,cpu
xchplot2 plot ... --cpu
xchplot2 plot ... --devices cpu --cpu-workers 2
```

Ordinarily, `plot` and `batch` distribute independent plots through a shared queue,
with one worker per GPU. Repeated selectors are deduplicated. Each GPU
chooses a tier from its own free VRAM; host memory scales with the workers
and their tiers. Storage, CPU compression, and PCIe bandwidth are shared.

### CPU worker count and NUMA

`--cpu-workers auto|max|N|off` sets the count **per selected NUMA node**.
Selecting a nonzero count also opts in to CPU plotting. `auto` starts from
four workers per node when CPU-only, or one per node beside a GPU, then
reduces the count to fit available RAM. `max` is bounded by RAM and core count;
`0` or `off` disables CPU workers even with `--devices all`.

Each CPU worker needs its own working set. Before each new plot, it checks
available RAM again and waits for a peer to free memory if necessary. It
retires after `XCHPLOT2_CPU_WAIT_SECS` (default 300) if memory does not return.
`XCHPLOT2_CPU_RESERVE_MB` keeps RAM available for other work.

CPU workers run at nice +10 by default and are pinned to their selected NUMA
node. Pinning controls locality, not pos2-chip's thread count; the CPU plotter
can still oversubscribe the node. `-v` reports the actual affinity mask:

```text
[cpu1#0] pinned to NUMA node 1 (cpus 32-63)
```

For a running plotter, inspect all thread masks with:

```bash
grep -h Cpus_allowed_list /proc/$(pgrep -n xchplot2)/task/*/status | sort | uniq -c
```

`XCHPLOT2_CPU_NO_PIN=1` disables pinning for comparison. Measure on the actual
machine before raising concurrency; historical CPU and mixed GPU/CPU results
are retained in [BENCHMARKS.md](BENCHMARKS.md#earlier-cpu-and-spill-measurements).

### Batch completion and worker rates

Multi-worker jobs report each worker's rate and suggested batch sizes that
reduce idle time at the end. A tail guard stops a slower worker from starting
a plot when faster workers are expected to finish the remaining queue sooner.
It applies to CPUs and GPUs. `XCHPLOT2_TAIL_GUARD=0` disables the guard.
Rate estimates and already-running plots can still leave a tail.

Shared stderr lines may interleave between workers. The stdout paths identify
successfully completed outputs; see [plotting and recovery](#plotting-and-recovery).

### Per-GPU streaming tier

Append `:tier` to a GPU selector to override its tier:

```bash
xchplot2 plot ... --devices gpu,2:tiny
xchplot2 plot ... --devices all,2:tiny
xchplot2 plot ... --devices gpu:tiny,2:plain
xchplot2 plot ... --devices gpu:tiny,2:auto
```

Precedence, highest first:

1. A specific GPU's `<id>:<tier>` token.
2. `gpu:<tier>` or `all:<tier>`.
3. A non-auto global `--tier`, then `XCHPLOT2_STREAMING_TIER`.
4. Automatic selection from free VRAM.

The specific `:auto` override restores automatic selection for that GPU.
Global `--tier auto` leaves the environment default in effect; unset
`XCHPLOT2_STREAMING_TIER` to restore global automatic selection.
CPU tier suffixes, unknown tiers, and conflicting tiers for the same GPU ID
are rejected. See [memory requirements](#memory-requirements) for this branch's
tiers. `scripts/test-multi-gpu.sh` checks parsing and, when enough GPUs are
visible, runs a multi-GPU smoke test.

### Single-plot multi-GPU strategies

`--shard-plot` makes the selected GPUs cooperate on each plot. Peer transport
is the default; `--host-bounce` uses explicit host-pinned staging when the
peer staging does not fit:

```bash
xchplot2 plot ... --devices 0,1 --shard-plot
xchplot2 plot ... --devices 0,1 --shard-plot --host-bounce
```

Sharding is experimental and GPU-only. The independent work queue was faster
on the previously measured PCIe host; see the
[earlier multi-GPU results](BENCHMARKS.md#earlier-multi-gpu-measurements).

`--strategy auto|work-queue|pipeline|shard` explicitly selects the strategy.
Auto normally chooses the work queue; with multiple devices it chooses a
pipeline if the smallest GPU's total capacity is below Tiny's base peak.
This is a strategy heuristic; actual memory admission can still refuse it.
The pipeline requires two or three GPUs and assigns different stages to
them. `--pipeline-plot` opts in directly, and
`--pipeline-stage-tiers tiny:minimal` sets one tier per stage (two or three
colon-separated `tiny` or `minimal` values). Pipeline plotting rejects
`--max-host-ram`; select `--strategy work-queue` when a host-memory cap is needed.
`--pipeline-depth N` controls the preallocated boundary-buffer count
(default 2); increasing it uses more pinned host RAM. These strategy and
stage controls apply to `plot` and `batch`. `--prefer-peer-copy` is a
deprecated no-op alias because peer transport is already the default.

## Benchmarking

`bench` measures how fast your hardware plots by writing synthetic
unfarmable `.plot2` files (random plot_ids, no keys), then reports
steady-state throughput in TiB/s, TiB/hour, TiB/day, and TiB/month
(30-day basis):

```bash
# Quick smoke (k=18 finishes in seconds on most GPUs)
xchplot2 bench -k 18 -n 3 -o /tmp

# Measure one GPU at k=28 after two warmup plots
xchplot2 bench -k 28 -n 10 --warmup 2 --devices 0 -o /scratch

# Also run a tmpfs pass to isolate compute from disk I/O
xchplot2 bench -k 28 -o /scratch --compute-only
```

| Option | Meaning |
|---|---|
| `-k K`, `-s S`, `-T` | Plot size, strength, and testnet parameters; defaults 28, 2, and mainnet |
| `-n N`, `--num N` | Measured plot count used to size the queue; default 10 |
| `--warmup W` | Initial completions excluded per worker; default 1 |
| `-o DIR`, `--out DIR` | Directory for real output writes; default current directory |
| `--devices SPEC`, `--cpu`, `--cpu-workers N` | Select the [devices and CPU workers](#devices-and-cpu-workers) to measure |
| `--tier T` | Force a [streaming tier](#memory-requirements), even when the pool fits |
| `--max-host-ram SIZE`, `--temp-dir DIR`, `--no-auto-spill` | Use the normal [host memory and spill policy](#host-ram-and-disk-offload) |
| `--keep` | Retain synthetic output files on disk for inspection |
| `--compute-only` | Add a second pass using RAM-backed output when available |
| `--target-size TiB` | Estimate time to fill this capacity instead of the output directory's free space |
| `-v`, `-q` | More worker detail, or quieter informational output |

The queue contains `(warmup + num) × workers` plots. Faster workers can
complete more of that queue; `-n` does not guarantee equal per-worker counts.
Each worker's warmup is excluded separately, and the report shows the
steady-state window and per-worker rates. Increase `-n` if a short run
cannot measure a slower worker.

Every pass includes FSE compression and real file writes. The second
`--compute-only` pass uses tmpfs if there is enough room; otherwise it warns
and reports a `compute+cache` pass in the output directory. Inspect that label
before interpreting the difference as disk overhead.

Bench removes its generated files by default. `--keep` retains disk output;
temporary tmpfs output is always removed to release the RAM. Results and
time-to-fill estimates are printed to stderr. See [BENCHMARKS.md](BENCHMARKS.md)
for the project's recorded measurements and comparison methodology.

## Memory requirements

At k=28, auto selection tries the persistent pool, then Plain, Compact,
Minimal, and Tiny. Pinned is a separate manual tier, not a smaller automatic fallback.
The owning driver supplies free VRAM after context creation. A missing
reading or insufficient `peak + buffer` causes refusal before allocation.

The streaming base peaks below come from [VramBudget.hpp](src/host/VramBudget.hpp).
They are allocation models, not the desktop driver deltas or host RSS in
[BENCHMARKS.md](BENCHMARKS.md#current-throughput-and-memory).

| Tier | Base peak, MiB | Base + default 128 MiB buffer |
|---|---:|---:|
| Plain | 7,290 | 7,418 |
| Compact | 5,200 | 5,328 |
| Minimal | 3,900 | 4,028 |
| Tiny | 1,100 | 1,228 |
| Pinned (manual) | 1,150 | 1,278 |

The selected backend's sort scratch can raise these base peaks. On roomy
GPUs, Plain/Compact can receive up to 780 MiB of optional two-phase match
scratch and Minimal up to 1,170 MiB. The grant uses memory left after the
base peak, backend adjustment, and buffer; it shrinks on a tighter card.
A roomy card's measured use is therefore not a required tier floor.

The `vram:` diagnostic shows the selected allowance. `bench` samples driver
memory throughout the run and fails if use exceeds the budget.
`POS2GPU_VRAM_MARGIN_MB` changes the buffer; desktop activity may need more
than the default. Models scale with k.

Lower VRAM tiers generally need more host RAM. Use the current per-tier
[host RSS measurements](BENCHMARKS.md#current-throughput-and-memory) for
planning, with room for the OS and other workers. Host admission uses a
separate model of pinned and anonymous memory, plus a reserve. Eligible
storage can be moved to disk when that model does not fit.

Memory quantities in this guide use MiB/GiB. Environment names ending in
`_MB` take MiB counts; existing diagnostic output may label them `MB`.
For boundary tests and the limits of software caps, see
[CONTRIBUTING.md](CONTRIBUTING.md#gpu-ci).

## Host RAM and disk-offload

Lower VRAM tiers keep more intermediate data in host RAM. A tier can fit
the GPU and still exceed the host's available memory.

xchplot2 models each tier's host peak and checks it before allocating
anything. When the tier does not fit, it **tries to spill eligible cold tables to a
usable temporary directory** by default. If the remaining working set still does not fit, it refuses.
Automatic spill is used only when admission would otherwise fail.

```bash
# Nothing to do — this is the default. A host that is short on RAM
# spills, announces it, and plots.
xchplot2 plot -k 28 -n 10 -f <farmer-pk> -c <pool-address>

# Put the spill somewhere specific (a fast NVMe, not the array).
xchplot2 plot ... --temp-dir /mnt/nvme/xchplot2-spill

# Cap the pinned+anonymous host peak yourself. Accepts 8G / 8192M /
# raw bytes, or `min` for "spill everything this tier can".
xchplot2 plot ... --max-host-ram 8G
xchplot2 plot ... --max-host-ram min

# Prefer a clear refusal over a slower plot.
xchplot2 plot ... --no-auto-spill
```

`--auto-spill` re-enables automatic spilling after an earlier
`--no-auto-spill` flag or configuration setting. The environment override
`XCHPLOT2_NO_AUTO_SPILL=1` still disables it; unset that variable to re-enable it.

What gets spilled, in order, largest first — and only as far as the
budget requires:

1. the cold cap-sized tables, streamed through two shared 32 MiB staging
   windows (64 MiB in total, however many tables spill) so the disk I/O
   overlaps compute. Which tables depends on the tier: `tiny` and `pinned` route
   `h_t1_meta`, `h_t3`, `h_t2_meta`, `h_t2_xbits`; `compact` and
   `minimal` route `h_t3`, `h_frags`, `h_t2_xbits`;
2. the D2H drain slots, 3 → 1, **last**. Spilling a table costs disk
   I/O per plot; a drain slot costs producer/consumer overlap across
   plots, which is the more expensive of the two in a batch.

Earlier spill RSS and timing measurements are retained in
[BENCHMARKS.md](BENCHMARKS.md#earlier-cpu-and-spill-measurements). They predate
the current pinned scratch reuse and are not current minimum-RAM guarantees.

Notes:

- **The temp dir must be real disk.** `/tmp` is tmpfs on most systemd
  distributions, i.e. RAM — spilling there consumes the memory the
  budget exists to bound. xchplot2 refuses a RAM-backed temp dir up
  front (override with `XCHPLOT2_ALLOW_RAM_TEMP_DIR=1` to allow RAM-backed spill), and checks the dir exists and is writable before
  starting rather than failing mid-batch.
- **Budget ~7.1 GiB of free space** for tiny at k=28, ~5.1 GiB for
  compact or minimal — the latter two route `h_frags` as well, a full
  2.03 GiB table even though it makes no engine I/O. This is checked
  against the temp dir before the batch starts, and it is per *worker*:
  each GPU in a multi-GPU run spills into its own files, so size the dir
  for the number of cards you are plotting with.
- **Repeated sort passes increase temporary I/O.** Inspect the
  `[spill] this plot:` line for your configuration and size the drive's
  endurance from its write volume. Historical I/O counts are in
  [BENCHMARKS.md](BENCHMARKS.md#sycl-spill).
- **`--max-host-ram` bounds the unswappable class** — pinned plus
  anonymous, the class that gets a process OOM-killed. One table
  (`h_frags`, on compact/minimal) is spilled as a file-backed mapping
  instead: those bytes leave the dangerous class but stay resident
  until the kernel needs them back, so they still show in RSS. Where
  the two differ, the log reports both.
- Files are unlinked at creation, so a crash cannot leave them behind,
  and each is `fallocate`d as it is created — a disk that fills anyway
  fails at once with its size rather than part-way through a table.

## Lower-level subcommands

### Single test plot

```bash
xchplot2 test <k> <plot-id-hex> [strength] [plot-index] [meta-group] [verbose]
```

This accepts a raw 64-character hex plot ID. It uses CPU phases by default;
`-G` / `--gpu-all` selects all available GPU phases, while `--gpu-t1`,
`--gpu-t2`, and `--gpu-t3` select individual phases. `-P` / `--profile`
prints phase timings. Use `-m` / `--memo HEX`, `-o` / `--out DIR`,
`-N` / `--out-name NAME`, and `-T` / `--testnet` for output and test parameters.
The [CPU-reference fixture](CONTRIBUTING.md#building-and-running-tests)
shows a matching raw-ID test and GPU batch. Arbitrary IDs and memos do not
produce farmable plots.

### Verification

```bash
xchplot2 verify /path/to/NAME.plot2 --full --trials 100
```

`verify` checks file structure, then samples quality chains for N random
challenges (default 100). `--full` also reconstructs and cryptographically
validates a full proof for every returned chain, failing if any cannot be
validated. Both modes fail on an empty sample. Sampling does not validate
every part of a plot; use byte comparison with a CPU reference for parity.

### Parity checks

```bash
xchplot2 parity-check --dir build/tools/parity
```

Runs the available `*_parity` and `*_test` executables, prints each result
and failure output, and returns nonzero if a test fails. The default
directory is `./build/tools/parity`. Build the tests first; see
[CONTRIBUTING.md](CONTRIBUTING.md#building-and-running-tests) for CMake,
CTest, hardware requirements, and CPU-reference byte comparisons.

### Shell completions

`xchplot2 completions bash|zsh|fish` writes a completion script to stdout.
For Bash, load it in the current shell or add this line to `~/.bashrc`:

```bash
source <(xchplot2 completions bash)
```

For zsh, save the output as `_xchplot2` in a completion directory:

```zsh
mkdir -p ~/.zsh/completions
xchplot2 completions zsh > ~/.zsh/completions/_xchplot2
```

Add `fpath=(~/.zsh/completions $fpath)` to `~/.zshrc` before its `compinit`
call. If completion is not initialized there, follow it with
`autoload -Uz compinit` and `compinit`.

For fish, install the generated script in its completion directory:

```fish
mkdir -p ~/.config/fish/completions
xchplot2 completions fish > ~/.config/fish/completions/xchplot2.fish
```

Regenerate saved scripts after upgrading. Completion suggestions cover
common commands and options; use this reference for the complete command guide.

## Troubleshooting

- **Listing visible GPUs**: `xchplot2 devices` prints id, name, backend,
  VRAM, compute-unit count, and which sort path each device will use
  (CUB on cuda-backend devices when this build links CUB; SortSycl
  otherwise). Use the printed `[N]` index with `--devices N` for
  `plot` / `batch`.

- **"tier X needs ~N GiB of HOST RAM"**, or the plotter is
  OOM-killed mid-batch: this is host memory, not VRAM, and a *lower*
  `--tier` costs **more** of it, not less — the ladder buys VRAM with
  host RAM. Do not respond by forcing a smaller tier. Either give the
  spill a real disk (`--temp-dir /mnt/nvme/...`, and note `/tmp` is
  usually tmpfs, which is refused), or free host RAM. The error names
  the floor the spill can reach, so if it says the gap cannot be
  closed, the box genuinely needs more RAM for that card. See [Host RAM
  and disk-offload](#host-ram-and-disk-offload).

- **Hybrid hosts (NVIDIA + AMD/Intel on the same box)**: a single
  binary handles all visible GPUs. `xchplot2 plot --devices gpu`
  spawns a worker per GPU (use `--devices all` to also add a CPU
  worker); each worker picks the right sort backend at queue
  construction (CUB on NVIDIA, hand-rolled SYCL radix on AMD/Intel)
  via the runtime dispatcher in `SortDispatch.cpp`. No rebuild
  required to add a second-vendor card.

- **`[AdaptiveCpp Warning] [backend_loader] Could not load library:
  /opt/adaptivecpp/lib/hipSYCL/librt-backend-cuda.so (libcudart.so.11.0:
  cannot open shared object file)`**: cosmetic only — AdaptiveCpp
  built with CUDA backend support but no CUDA runtime to load. Happens
  when AdaptiveCpp was installed out-of-band rather than via
  `scripts/install-deps.sh --gpu amd` (which sets
  `-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE`). To suppress without a
  rebuild: `export ACPP_VISIBILITY_MASK='hip;omp'` so AdaptiveCpp skips
  the CUDA backend probe entirely.

- **`T1 match produced 0 entries`** on RDNA1 (`gfx1010` / `gfx1011` /
  `gfx1012`, including the Radeon Pro W5700 / RX 5700 XT). The
  community `gfx1013` AOT-spoof default was observed to silently
  compile no-op kernel stubs on at least one W5700 + ROCm 6 +
  AdaptiveCpp 25.10 host. Cargo now defaults to `ACPP_TARGETS=generic`
  (SSCP JIT) for RDNA1, while the container script retains the spoof.
  Follow [AMD target selection](INSTALL.md#amd-target-selection) for the
  build path and explicit overrides. The startup self-
  test in `SyclBackend::queue()` catches the no-op-kernel case at
  queue construction with a clear exception, so this surfaces
  immediately rather than as empty pipeline output minutes in.

- **`CUB ... invalid argument`** mid-pipeline, or
  **`sycl_backend::queue: device id 0 out of range (found 0 usable
  GPU device(s))`** with `--devices N` while the default selector
  finds a GPU: pre-`762fde2` symptoms of CUB-only sort being
  dispatched against an AMD/Intel device (or being filtered out of
  the device list). The runtime sort dispatcher fixes both — `git
  pull && cargo install --path . --force` to upgrade.

- **Deep-pipeline diagnostics**: set `POS2GPU_T1_DEBUG=1` for verbose
  per-stage dumps (Xs gen / sort intermediates, T1 match input/output
  samples, AES T-table sanity). Useful when the symptom isn't on the
  list above and you want to localize where the data goes wrong.

## Environment variables

| Variable                      | Effect                                                                  |
|-------------------------------|-------------------------------------------------------------------------|
| `XCHPLOT2_BUILD_CUDA=ON\|OFF` | Override the build-time CUB / nvcc-TU switch. Default is vendor-aware (NVIDIA → ON; AMD / Intel → OFF; no GPU → `nvcc`-presence). Force `OFF` on dual-toolchain hosts (CUDA + ROCm) where you want the SYCL-only build. |
| `XCHPLOT2_STREAMING=1`        | Force the low-VRAM streaming pipeline even when the pool would fit.     |
| `XCHPLOT2_STREAMING_TIER=plain\|compact\|minimal\|tiny\|pinned` | Force a streaming tier even when the pool fits. A non-auto `--tier` takes precedence. See [memory requirements](#memory-requirements). |
| `XCHPLOT2_MAX_HOST_RAM=8G\|8192M\|<bytes>\|min` | Cap the streaming path's unswappable (pinned + anonymous) host peak, spilling the cold cap-sized tables to `--temp-dir` largest-first — and then the D2H drain slots, 3 → 1 — until the model fits. `min` takes everything this tier can. Equivalent CLI flag: `--max-host-ram`. See [Host RAM and disk-offload](#host-ram-and-disk-offload). |
| `XCHPLOT2_TEMP_DIR=/path`     | Where spilled tables live. Falls back to `$TMPDIR`, then `/tmp`. **Must be real disk** — a RAM-backed dir is refused up front, since spilling to tmpfs consumes the memory the budget exists to bound. Equivalent CLI flag: `--temp-dir`. |
| `XCHPLOT2_NO_AUTO_SPILL=1`    | Refuse to plot when the tier does not fit host RAM, instead of spilling automatically. The automatic spill only ever fires where the run would otherwise be rejected, so this trades a slower plot for a clear error. Equivalent CLI flag: `--no-auto-spill`. |
| `XCHPLOT2_ALLOW_RAM_TEMP_DIR=1` | Allow a RAM-backed spill directory with a warning. The spilled bytes consume RAM outside the stated host budget. No override is needed for a disk-backed directory. |
| `XCHPLOT2_HOST_RESERVE_MB=N`  | Host RAM left for the rest of the box when admitting a streaming tier. Default 512. |
| `XCHPLOT2_DRAIN_SLOTS=N`      | Pin the D2H drain slot count (1-3) and skip the budget's automatic reduction. Fewer slots means less pinned host memory and less producer/consumer overlap. |
| `XCHPLOT2_SPILL_IO_THREADS=N` | Worker threads per spill I/O job, 1-32. Default 2. Threads share the staging memory; measure on the target drive before increasing the count. `1` processes the job serially. |
| `XCHPLOT2_HOST_FREE_MB=N`     | Testing only — make the host report at most N MB free, so the host-RAM guard and the automatic spill can be exercised on a box that is not actually short. Clamps only: it can never talk the plotter past a real shortage. |
| `POS2GPU_MAX_VRAM_MB=N` | Cap the free-VRAM query and enforce the selected budget. This tests admission and the watchdog on the current GPU; it does not emulate another physical card. |
| `POS2GPU_VRAM_MARGIN_MB=N` | Buffer beyond the selected VRAM peak, in MiB. Default 128. Raise it when other activity takes VRAM after admission. |
| `POS2GPU_ASSERT_VRAM=1`       | Fail (not just warn) when a path's true peak exceeds what it declared. Armed by `bench`. |
| `POS2GPU_STREAMING_STATS=1`   | Log every streaming-path `malloc_device` / `free`.                      |
| `POS2GPU_POOL_DEBUG=1`        | Log pool allocation sizes at construction.                              |
| `POS2GPU_PHASE_TIMING=1`      | Per-phase wall-time breakdown (Xs / sort / T1 / T2 / T3) on stderr.     |
| `ACPP_GFX=gfxXXXX` | AMD container AOT target; Cargo detects the GPU separately. See [AMD target selection](INSTALL.md#amd-target-selection). |
| `ACPP_TARGETS=...` | Override Cargo target selection, such as `generic` or `hip:gfx1031`. For direct CMake use `-DACPP_TARGETS=...`. |
| `CUDA_ARCHITECTURES=89`    | Override the CUDA arch autodetected from `nvidia-smi`.                  |
| `CUDA_PATH=/path/to/cuda`     | Override the CUDA Toolkit root for linking (default: `/opt/cuda`, `/usr/local/cuda`). Useful on JetPack / non-standard installs. |
| `CUDA_HOME=/path/to/cuda`     | Fallback for `CUDA_PATH` — same effect.                                 |
| `POS2_CHIP_DIR=/path`         | Build-time: point at a local pos2-chip checkout instead of FetchContent.|
| `XCHPLOT2_TEST_GPU_COUNT=N`   | Override `scripts/test-multi-gpu.sh`'s auto-detected GPU count (forces run / skip without consulting `nvidia-smi`). |
| `XCHPLOT2_CPU_AUTO_WORKERS=N` | Override the automatic CPU worker starting count, before the host-RAM cap. |
| `XCHPLOT2_CPU_RESERVE_MB=N` | Additional host RAM reserved from CPU workers, in MiB. |
| `XCHPLOT2_CPU_WAIT_SECS=N` | Maximum wait for host RAM before retiring a CPU worker; default 300 seconds. |
| `XCHPLOT2_CPU_NICE=N` | CPU worker nice value; default 10. |
| `XCHPLOT2_CPU_NO_PIN=1` | Disable CPU NUMA affinity for comparison. |
| `XCHPLOT2_CPU_WORKERS_UNGATED=1` | Bypass CPU host-RAM admission checks; an allocation failure can terminate the batch. |
| `XCHPLOT2_TAIL_GUARD=0` | Disable the slower-worker tail guard. |
