# xchplot2 reference (`main`)

[README](README.md) · [Installation](INSTALL.md) · [Benchmarks](BENCHMARKS.md)

[Plotting and recovery](#plotting-and-recovery) · [Configuration](#configuration-and-argument-files) ·
[Groups and drive filling](#grouped-plotting-and-drive-filling) · [Shared compute](#shared-compute) ·
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
| `-k`, `--k K` | 28 for farming (default); even sizes 18..26 for testing |
| `-n`, `--num N` | Number of independent output files; default 1 |
| `--group-members N` | Members per file, 1..65535; default 1 |
| `--compute auto\|off` | Use saved compute pool membership when present; default `auto` |
| `-s`, `--strength S` | Proof strength; default 2 |
| `-f`, `--farmer-pk HEX` | Farmer public key, 96 hex characters |
| `-p`, `--pool-pk HEX` | Pool public key, 96 hex characters |
| `--pool-ph HEX` | Pool puzzle hash, 64 hex characters |
| `-c`, `--pool-contract-address ADDRESS` | Pool contract address, `xch1...` or `txch1...` |
| `-o`, `--out DIR` | Output directory; default current directory |
| `-d`, `--devices SPEC` | Select [GPUs and CPU workers](#devices-and-cpu-workers) |
| `-M`, `--max-host-ram SIZE` | Host-memory budget, such as `8G` or `512M` |
| `-t`, `--temp-dir DIR` | Temporary storage for spill or grouped construction |
| `-j`, `--workers N` | Shared compression CPU allowance for grouped construction |
| `-r`, `--resume` | Recover the saved job and skip validated completed files |
| `-i`, `--plot-index N` | Member index; must be 0 for single-member groups |
| `-g`, `--meta-group N` | Meta-group field; default 0 |
| `-S`, `--seed HEX` | Optional 64 hex characters for reproducible identities |
| `-v`, `--verbose` | Print additional worker and plotting details |

Supply the farmer key and one of the pool key, puzzle hash, or contract
address forms. See [plot indices and meta groups](#plot-indices-and-meta-groups)
for the grouping distinction. Full help: `xchplot2 --help`.
Only k28 is valid for farming. Smaller sizes exercise construction, recovery
and proof validation in tests; their output is not farmable.

On a host that is short of RAM for the tier its GPU lands on, add
`--temp-dir <path>` to choose where the automatic disk-offload writes
(`--max-host-ram` to bound it explicitly, `--no-auto-spill` to turn it
off) — see [Host RAM and disk-offload](#host-ram-and-disk-offload).

Before plotting starts, `plot` saves the prepared identities and keys in an
`xchplot2-job-*.tsv` manifest in the output directory. Use `--manifest FILE`
to choose its location. Each new job keeps its own manifest; another job
cannot overwrite it. Manifests contain private plot keys and are created
with owner-only permissions on Linux and protected owner-only ACLs on Windows.

Repeat the same `plot` command with `-r`/`--resume` (or `--skip-existing`) to
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

Commands show concise results by default. `-v`/`--verbose` adds diagnostics,
assumptions and worker details; `-q`/`--quiet` suppresses informational messages.
Quiet mode retains command results, output paths, warnings and errors. JSON
reports retain their fields regardless of verbosity. Use `COMMAND --help`
for common options and `COMMAND --help-all` for the extended plotting options.

An aggregate progress line updates after each plot completes:

```
[batch] progress: plot 3/10 done (30.0%, 2.41 s/plot avg, 0.000373 TiB/s, fully plotted in ~17s)
```

On a terminal it rewrites itself in place and is on by default; when
stderr is redirected it prints one line per plot and defaults to off.
`--progress` / `--no-progress` force it either way. The ETA is
plots-based (average s/plot × remaining), formatted as
hours/minutes/seconds when the estimate exceeds one hour.

Quiet mode suppresses automatic progress and plotting summaries. Explicit
`--progress` still enables progress; `--no-progress` does not disable verbose
details. `plot` writes each absolute output path to stdout after that
file is published or validated by resume, in completion order. Failed or
unstarted plots are never listed, and completed paths remain available if
a later plot fails. Exit status is 0 for completion, 1 for argument errors,
2 for an exception, 3 for per-plot failures, or 4 for unfinished work.
`-q` and `-v` are mutually exclusive.

Plots are written to an exclusively created `<name>.gplot.partial.XXXXXX`
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

Ordinary single-member and raw manifests have nine whitespace-separated fields in this order:

```text
k strength plot_index meta_group format group_id_hex memo_hex out_dir out_name
```

`format` is `gplot-v2` for a single-member group (index 0), or `raw-v2`
for a temporary member consumed by the group assembler. `group_id_hex` is
64 hex characters; the member ID is derived from it, the index, and meta group.
`memo_hex` encodes up to 255 bytes (`""` represents an empty memo).
`plot --group-members N` and automatic shared plotting save `gplot-v2-grouped`
rows with a tenth field, the member count. This explicit format prevents older
binaries from silently treating a grouped job as single-member output. Grouped
rows start at index zero; `batch` selects grouped construction for that format.
Grouped, ordinary single-member and raw jobs require separate manifests.
Legacy manifests with a testnet boolean and per-plot ID are rejected.
Memo and path fields accept double quotes, with backslash escapes
for quotes and backslashes. `out_name` must be a filename, not a path;
relative `out_dir` paths resolve from the working directory. Blank lines
and `#` comments are ignored. Invalid rows are rejected before plotting.
Prefer the automatically saved manifest for recovery; see
[Security](SECURITY.md) before sharing manifests containing private keys.

## Grouped plotting and drive filling

These commands are included in normal Linux Cargo, CMake and container builds,
and in experimental native Windows CMake builds. Windows grouped construction
requires local NTFS volumes with ordinary, uncompressed, unencrypted files;
ReFS, network volumes and sparse files are rejected. Native Windows support
remains experimental; see the [platform limitations](INSTALL.md#windows).
Windows retains compact retirement records beside completed external jobs
for recovery. Keep them with the output until recovery is no longer needed.
Unknown disk topology is scheduled conservatively. Windows NVMe destinations
sharing a known subsystem share a writer group; distinct known subsystems can
run independently. Unsupported or unknown NVMe identities retain conservative
whole-job serialization.

The grouped engine supports one local CPU or GPU producer, or a shared compute
pool. Select `--devices cpu`, `--devices gpu0`, or `--backend auto`; automatic
selection can fall back to CPU. Multi-device selectors, CPU NUMA selectors,
`--cpu-workers`, sharding, pipeline strategies, `--continue-on-error`, and
`--max-host-ram min` belong to the single-member batch engine. Use
`--compute off` to keep that engine for ordinary plotting after enrolling a
machine in shared compute.

Create three independent files, each containing four members:

```bash
xchplot2 plot -n 3 --group-members 4 \
    -f "$FARMER_PK" -c "$POOL_ADDRESS" -o /plots
```

`-n` always counts files; `--group-members` counts members within each file.
The normal private job manifest and `plot --resume` / `batch --resume`
interfaces retain the prepared identities. New unseeded jobs get fresh keys.

For a directory layout, pass the destination directories as trailing arguments.
Shell globs, brace expansion and quoted Bash arrays work normally:

```bash
xchplot2 plan -- /disk-{a,b}/plots
xchplot2 fill --plan-only -- /disk-a/plots /disk-b/plots
# Quoted arrays preserve spaces in destination names:
destinations=(/disk-a/plots "/disk-b/plot files")
xchplot2 plan -- "${destinations[@]}"
```

Without `--save-plan`, both forms perform the same read-only preview. They need
existing destination directories but no keys, create no files or locks, and
start no compute services.
The default report summarizes the destinations, planned files and members,
storage allowance and intended harvester. `-v` adds model assumptions and
resource details; `--json` emits the machine-readable report. An infeasible
layout reports its reason and exits unsuccessfully. `--help` shows the common workflow;
`--help-all` includes recovery, tuning and shared-compute options.

To execute a reviewed layout, save a keyless plan:

```bash
xchplot2 plan /disk-a/plots /disk-b/plots --save-plan farm.plan
xchplot2 fill farm.plan -f "$FARMER_PK" -c "$POOL_ADDRESS"
```

`--save-plan` also works with `fill --plan-only`. It creates only the requested
plan and refuses to overwrite an existing file. The plan contains the
resolved harvester inputs, initial assignments, destination identities, quotas
and fill policy, with no plot keys. `plan farm.plan` inspects it without
plotting; `-F FILE` or `--from-plan FILE` is an explicit alternative for either command.
Execution preserves those assignments and checks the current storage,
available space and joint farming workload before creating the private job.
The reviewed capacity must still be available; additional free space does not
enlarge its quota. Execution options such as plotting RAM, workers and scratch
may change. Different layout inputs, destination selectors or a changed source
job require a new review. New previews and completed-job recovery require
matching plan schema, farming model, key derivation and numeric representation.
Legacy plans and in-progress construction checkpoints retain their
original-build requirement.

With no directories, `plan` lists candidate mounted data volumes with available
space and detected media type. It excludes system mounts, RAM filesystems,
read-only volumes and unsupported filesystems. These are suggestions; select
the intended directories explicitly before filling them.
One command accepts up to 1,024 destination directories. Directories on the
same filesystem share its available capacity; adding another directory there
does not add disk space.

Use `--paths-from FILE` for one literal path per line, including paths with
spaces. Blank lines are ignored; paths resolve from the current working
directory. Use `--paths0-from FILE` for NUL-delimited paths. Either accepts `-`
to read standard input. Lists do not expand shell expressions or comments.
An empty explicit list is an error. `--` makes subsequent arguments literal,
including names beginning with `-` or `@`:

```bash
xchplot2 plan --paths-from destinations.txt
printf '%s\0' /disk-*/plots | xchplot2 plan --paths0-from -
```

To build the layout, supply the farmer public key and one pool destination.
The plot seed is generated automatically; `-S "$SEED"` remains available for
reproducible identities:

```bash
xchplot2 fill -f "$FARMER_PK" -c "$POOL_ADDRESS" -- /disk-{a,b}/plots
# Resume the entire job through either member, without repeating keys or paths:
xchplot2 fill -r /disk-b/plots
```

`fill` always plans a farming layout, including with an explicit
`--group-members N`. Omit that option or use `auto` to select file membership
from the farming profile. Files start at index zero; the planner assigns
shared-ID families and meta groups across physical storage resources.
`--max-groups N` limits files per directory, and `--max-total-members N`
limits membership per directory. Without those limits, the initial layout uses
available capacity. `--capacity-bytes SIZE` caps aggregate output capacity
per filesystem; `--headroom-bytes SIZE` leaves space free (default 64 MiB).
Sizes accept bytes or suffixes such as `512M` and `14G`.

Every member directory contains a private, checksummed `grow-farm.plan` with
the seed, farmer/pool identity, complete destination list, limits and farming
profile. Each copy describes the whole job. New jobs store shared storage
settings and the resolved farming configuration once within each copy; the
capacity allowance includes recovery files and space for their replacements.
Keep these files with the outputs; the seed derives private plot keys
and must not be shared. It is not a wallet seed. A temporary `grow-farm.plan.next`
may remain after interruption; keep it for recovery too. New farming plans
record their creating build as provenance and can resume completed ordinary
groups with a build supporting the same plan schema, farming model,
key-derivation identity and numeric representation. In-progress construction
checkpoints and legacy plans still require their original build. A compatible
plan does not make an incompatible construction checkpoint resumable.

`plan --metadata-only PLAN` reads a saved keyless preview, and
`plan --metadata-only --manifest FILE` reads one private job copy. This mode
reports saved assignments and compatibility without probing destinations,
starting services or inspecting construction checkpoints. It can inspect a
supported legacy plan from another build without making that plan executable.
The report does not establish current availability or farming readiness.

To assess several managed jobs under one harvester budget, refresh a keyless
summary for each job, then assess the summaries together:

```bash
xchplot2 plan --manifest /disk-a/plots/grow-farm.plan --save-plan job-a.plan
xchplot2 plan --manifest /disk-b/plots/grow-farm.plan --save-plan job-b.plan
xchplot2 plan --assess-jobs job-a.plan job-b.plan --json
```

Refreshing requires all of that job's destinations and a recovered revision.
The summary retains its initial forecast and admitted member extents, including
expansions. Joint assessment includes those full promises even for offline
destinations. It shares the declared CPU, lookup-worker, cache and reader-memory
budgets, keeps independent jobs' identity families independent, and combines
known storage sharing conservatively. Duplicate jobs and destination aliases
are rejected. Harvester service inputs, plot parameters and meta constraints
must match; saved group-size choices and layout objectives may differ.

This mode reads the supplied summaries and checks their explicit destination
paths for topology. It does not open private source manifests, plots or
checkpoints, and does not create an execution plan. Its result covers saved
forecasts, not a measurement of current load. Refresh summaries after filling
or expanding a job before drawing current-load conclusions. Unresolved offline
topology is shared conservatively; outside plots and services remain unmodeled.

Resume searches every supplied directory and loads the full job. Directory
order and omitted arguments do not change its scope. `fill -r` from a member
directory, or `fill -r --manifest /disk-b/plots/grow-farm.plan`, also works.
Different jobs, conflicting revisions, corrupt copies or mismatching explicit
keys are rejected before output changes. Resume audits existing outputs,
preserves completed files, and recovers partial construction. Plotting RAM,
workers and compute selection can change; the saved identity namespace, farming
profile, initial assignments and quotas stay fixed.

The initial member count is a size forecast. As compression leaves room, new
jobs can extend the final unsealed file or admit another file within the
reviewed limits. Each addition must fit the combined job's farming workload,
exact construction space, file-size limit, headroom and scratch allowance.
Compatible changes on different destinations can share a saved revision.
Replacement copies can be staged concurrently on independent drives; every
destination must have a durable copy before the revision is published and
construction relying on it proceeds.
Completed files stay unchanged, and resume retains admitted additions and the
recorded stop reason. Construction methods and configured I/O windows can
stop filling before the output budget is exhausted, so some unused capacity
can remain. Existing jobs retain their saved fill policy.
`metadata-capacity` means another extent would exceed the job's reserved
recovery metadata. That limit can also leave space unused; resume does not
retry an addition already recorded as stopped.

To add another destination, preview it together with any existing member,
then explicitly extend the job:

```bash
xchplot2 plan /disk-b/plots /disk-c/plots --save-plan expansion.plan
xchplot2 fill expansion.plan
```

The preview includes all saved destinations. Existing plot identities and
assignments stay fixed; the new layout must fit the combined farm's CPU,
memory, storage and response limits. An expansion can be infeasible even when
the new disk has room. The assessment covers plots belonging to this job;
unmanaged plots consume disk space but their farming workload is not modeled.
`fill --extend /disk-b/plots /disk-c/plots` plans and executes an expansion
directly. All original destinations must be available for expansion. A new revision is
saved across the members before new plotting starts; interrupted publication
is completed with ordinary `fill -r`.

If a saved directory is unavailable or its mountpoint now refers to a different
directory, normal resume stops and lists it. `fill -r --allow-missing DIR`
continues already admitted construction on available members and exits
with status 4 while other members remain unavailable. It cannot change the
job, admit another extent, seal an unfinished file or commit an incompletely
staged revision. Restore the original mounts
and resume normally to finish the whole job.
Before new external construction, all fixed-job destinations must be available
to account for preserved scratch and its ownership records. This does not
prevent read-only inspection or resuming available members without new scratch.

For an intentional move, keep the output and saved job together, take the old
path offline, then use `fill -r --relocate OLD NEW`. The new directory locates
the job; another existing member or `--manifest FILE` can also serve that purpose.
All members must be available after applying the relocation. Replacement
outputs are audited before the new location is saved.

### Growing an existing group

```bash
xchplot2 grow /plots/example.gplot --max-members 8
```

`grow` requires an existing ordinary v2 group file and recovers interrupted
construction before extending it. It reads the file's identity and parameters;
keys are unnecessary. `--max-members N` is the desired total membership, with
a maximum of 65535. Capacity and construction limits can stop growth earlier.
Use `plot` to create a new named job, or `fill` to manage a directory layout.

### Construction and recovery

`-M`/`--max-host-ram SIZE` bounds modeled construction, producer, transport and
codec buffers across the job. The default is 14,000,000,000 bytes.
`-j`/`--workers N` bounds the shared compression CPU allowance; zero or omission
uses the available logical CPU count. The operating system, libraries and
file cache are outside that buffer model. Separate plotting processes each
have their own construction budget.

Automatic construction chooses among incremental rewriting, staging,
merging and external contributions according to available RAM and scratch.
`--construction auto|incremental|staged|merged|external` forces a method when
needed. `--batch-members N` bounds construction batches independently of the
final file membership. `--member-storage compact|raw` selects retained
fragments; compact is the default.

Repeat `-t`/`--temp-dir DIR` to offer scratch on additional filesystems.
Destination directories are excluded from automatic scratch selection unless
`--destination-temp on|ssd|hdd` permits them; the default is `off`. The media
filters require known, uniformly matching drives. Explicit scratch directories
remain available regardless of that policy. `--temp-capacity-bytes SIZE` caps
scratch per filesystem. External construction needs permitted scratch.

Writers sharing a physical drive serialize their I/O. Independent drives can
progress together within the shared RAM and CPU limits. `--io-overlap on|off`,
`--generation-overlap auto|on|off`, and `--cache-release on|off` control overlap
and clean-cache release. `--window-bytes SIZE` and `--checkpoint-bytes SIZE`
set fixed I/O and durability windows instead of the automatic choices.

Construction checkpoints are private and tied to the same build. Retain the
in-progress file, plan and any owned scratch/checkpoint directory together;
do not move or edit them during recovery. A file being constructed is not a
farmable completed output. Stop gracefully with Ctrl-C, then repeat the
owning command with `--resume` (`grow` resumes implicitly). Completed output
paths appear on stdout after publication or a successful resume audit.
`--json` replaces human output with the plan or construction metrics;
`--quiet` suppresses informational stderr and `--verbose` adds construction
details and final metrics. Aggregate member progress uses `--progress` and
`--no-progress`, with the same terminal defaults as `plot`.

### Farming layout

Automatic directory planning chooses farming group size and meta families
together. The plotting machine's RAM, GPU speed and temporary disks determine
how to construct that layout; the planner does not use them as the final
harvester's CPU or cache capacity. It uses the destination's physical-resource
topology and a declared farming profile. The default preview reports the
profile's source, CPU and memory allowances and initial file counts.
`--verbose` adds response estimates, assumptions, detected configuration,
identity families, resource sharing and workload estimates.

File sizes are estimates: compressed output can be slightly larger or smaller,
and a small difference can decide whether one more whole member fits. Filling
checks actual storage use before admitting more work. Temporary construction
and recovery space is a separate allowance, so it can limit filling even when
the final files would fit.

Use `--harvester local` when plotting on the intended harvester, or
`--harvester FILE` to use a profile from another machine. To transfer a profile,
export it on the intended harvester and copy it to the plotter:

```bash
# On the intended harvester; no destination or plotting keys are needed:
xchplot2 plan --harvester local --save-profile harvester.json
# Copy harvester.json to the plotter, then review its destinations:
xchplot2 plan /disk-a/plots --harvester harvester.json --save-plan farm.plan
xchplot2 fill farm.plan -f "$FARMER_PK" -c "$POOL_ADDRESS"
```

`--save-profile` writes only the requested profile, refuses to overwrite an
existing file and needs no destination. Explicit farming overrides are included
in the export. A saved plan already contains its harvester profile; execution
does not need the profile file again. A file literally named `local` can be
selected as `--harvester ./local`.

For repeated use, set the harvester in the existing
[configuration file](#configuration-and-argument-files), so normal plan and
fill commands need no selection flag:

```toml
[plan]
harvester = "/path/to/harvester.json"

[fill]
harvester = "/path/to/harvester.json"
```

Use `harvester = "local"` on the intended harvester. Command-line
`--harvester` overrides the saved selection. Without a selection, the planner
uses its declared defaults and reports them as assumptions.

Detection observes this process's CPU affinity and available memory, reads
the local Chia configuration, and attempts read-only authenticated harvester
RPCs through the installed `chia` executable. It starts no Chia services.
Use `--chia-root DIR` and `--chia-executable FILE` when needed. The default root
follows `CHIA_ROOT`, then `~/.chia/mainnet`. RPC detection is limited to a
configured loopback endpoint; missing tools, unsupported configuration or an
offline service are reported as unavailable, not as an empty farm.

The public [Chia development implementation](https://github.com/Chia-Network/chia-blockchain/tree/f94da8ed38485c7c1cb5f9a3d252d26f68ad6efb/chia/harvester)
uses the configured harvester lookup thread pool for v2 partial proofs.
Detection can use that observed concurrency. It does not establish finalized
PoS2 CPU, memory or cache defaults. Available hardware is not measured spare
capacity: CPU/memory budgets and service costs remain declared assumptions.
Explicit `--farming-*` options override imported or detected inputs for a new
job; for example, lower `--farming-cores` and `--farming-memory-bytes` to leave
resources for other services. Saved jobs keep the resolved profile and its
provenance on resume.

The JSON report's `farming_profile` can be imported either on its own or inside
the full report. It carries harvester inputs, hostname, observation time and
count-only loaded-plot snapshots; it imports no destination assignments or
plot keys. Snapshots do not establish refresh completion or the cost of
unmanaged plots. The assessment covers this job only, even when other plots
were detected. Reserve an allowance for outside work and qualify the intended
reader before relying on the estimate.

To sample the pinned group reader on explicit ordinary plots, run calibration on
the intended reader machine and save its observations with an editable profile:

```bash
xchplot2 plan --calibrate /plots/example.gplot --calibration-samples 16 \
  --save-profile reader-profile.json --json
xchplot2 plan /new/plots --harvester reader-profile.json --save-plan farm.plan
```

Calibration reads the supplied files without changing them. It reports fresh
and retained-reader quality lookups, foreground reader CPU time, and separate
read-only storage observations. Storage observations remain report-only; they
do not replace bandwidth, seek or concurrency assumptions. OS cache state is
uncontrolled, so opening a fresh reader does not establish cold physical reads.
Calibration does not solve full proofs or measure spare harvester capacity.

Existing CPU inputs only increase from the scoped samples and margin. These
are editable, conditional service envelopes, not measured worst-case bounds.
The saved profile records the sampled k, strength, reader, method and margin,
with its own calibration hostname/time. An imported baseline's configuration,
inventory and hardware snapshot keep their original hostname/time. Planning
rejects a calibrated profile for a different k, strength or reader; test-size
profiles require matching explicit test-size options rather than default k28.

Supply at most 64 plots with the same k and strength. Samples default to 16 per
plot and are capped at 1024. `--calibration-timeout-ms` defaults to 30000 and
`--calibration-memory-bytes` to `512M`, per reader worker. The worker cap is
admitted alongside the parent's buffers under `--max-host-ram`; its platform
limit covers address space on Linux or committed process memory on Windows,
and does not promise an equal RSS or filesystem-cache limit. Use `--json` or
`--save-profile FILE`; saving refuses an existing file. Harvester imports and
explicit farming inputs can supply the baseline. Calibration samples only the
supplied plots.

Defaults are deliberately independent of the plotting host:

| Farming input | Default | Override |
| --- | --- | --- |
| Objective | Minimize recurring dynamic work within the hard response deadline; latency breaks ties | `--farming-objective energy\|latency` |
| Eligible groups per physical storage resource | At most 1 per filter check | `--farming-max-eligible-groups N` |
| Relative CPU and I/O work weights | 1 and 1 | `--farming-cpu-work-weight N`, `--farming-io-work-weight N` |
| Concurrent CPU work | 4 cores | `--farming-cores N` |
| Synchronous lookups | 4 workers | `--farming-lookup-workers N` |
| Reader working memory | 512 MiB | `--farming-memory-bytes N` |
| Index cache | Stateless reader, no cache | `--farming-reader stateless\|cached`, `--farming-index-cache-bytes N` |
| Per-member CPU time | 4 ms for decoding, chaining and reader work | `--farming-chain-us-per-member N` |
| Index/open CPU allowance | 50 ns per expanded entry | `--farming-index-ns-per-entry N` |
| Hard modeled quality-response deadline | 8 seconds | `--farming-deadline-ms N` |
| Filter horizon | Base bits 9 and 8, adjusted for strength and capped at 13 | `--farming-filter-bits N`, `--farming-future-filter-bits N` set effective bits directly |

The response deadline is an operational latency limit, not a consensus deadline.
The future profile models one scheduled filter reduction without assigning a
calendar activation date. The default cadence is 75/8 seconds. The default CPU
coefficients are assumed service envelopes, not measurements of the user's
harvester. Deadline admission is conditional on those envelopes covering the
reader and storage's actual service costs.

Cold-reader admission reserves up to eight compressed bytes per index entry,
the valid format's maximum codeword size. Capacity forecasting uses an expected
index size separately. This avoids treating the average compressed index as a
worst-case read bound.

Detected HDDs use a 10 ms access cost, 250 MB/s transfers and one active read;
detected SSDs use 0.2 ms, 500 MB/s and up to four active reads. Unknown devices
use 20 ms, 125 MB/s and one active read, with shared resources serialized
conservatively. These are model inputs, not storage benchmarks. Shared transfer
bandwidth, synchronous lookup workers and harvester CPU capacity constrain the
modeled job. Per-file member processing remains serial.

Larger files reduce repeated index work and disk accesses, but processing more
members serially makes each lookup longer. Smaller files shorten individual
lookups while increasing the repeated work. The planner compares balanced
integer layouts within the declared storage, CPU, worker and memory limits.
An efficient average workload alone does not establish a response deadline.

Meta groups change eligibility correlations, so choose them together with file
size. The [filter implementation](https://github.com/Chia-Network/chia-blockchain/blob/main/chia/types/blockchain_format/proof_of_space.py)
compares a hash of the group ID and filter challenge with the signage-point
index XOR the meta value, masked by the effective filter. Distinct masked meta
values exclude simultaneous eligibility only within the same group ID.
Rotating meta numbers while generating unrelated group IDs does not provide
that guarantee. When the effective filter shrinks, some meta values alias.
Window scheduling also correlates neighboring signage points; a planner must
evaluate consecutive challenges, including carried-over work.

Within one shared-ID family, balance member counts across its usable meta
slots. With one file per slot and no other eligible work on the drive, this
minimizes the largest activation. Ordinary files start at member index zero:
reusing both the group ID and meta value in another such file duplicates member
identities. Exhausted meta slots require a new identity family. Assign families
per physical storage resource, keeping that assignment stable on resume.

Measure or explicitly model the actual reader. The pinned k28 C++ reader
expands a chunk index to 32 MiB per resident group reader. The pinned C API
constructs a reader for each quality request, so it instead pays index loading
and decoding on each lookup. The on-disk index is compressed and data dependent;
32 MiB describes the expanded index, not a fixed read size. It also reads the
selected chunks for all members and chains them before returning the group's
qualities. Index-cache policy,
decoding/chaining work and total farm concurrency therefore matter alongside
disk access. Do not infer farming CPU or cache capacity from the plotting host.

The cached profile assumes persistent readers loaded before farming. It removes
index-loading costs only when the entire expanded index set fits the declared
cache; otherwise it uses cold costs. The working-memory budget includes that
cache and concurrent reader scratch. Repeated lookups on one persistent reader
are serialized. These memory estimates constrain the profile; they are not a
strict RSS bound on the farmer's variable proof and fragment allocations.

Deadline admission considers every identity family that could be eligible
together, including families on different drives. It bounds CPU, lookup-worker
and physical I/O contention for that burst. The bound must fit both the response
deadline and the signage-point interval, so admitted work finishes before the
next arrival. The model assumes work-conserving queues with FIFO admission to
CPU and read resources. Same-family meta exclusion is retained under both
modeled filter settings. A finite sample with no late responses cannot replace
this admission check.

`--farming-objective energy`, the default, minimizes an expected dynamic-work
proxy: weighted CPU-active seconds plus I/O-active seconds per signage point.
The positive `--farming-cpu-work-weight` and `--farming-io-work-weight` values
express their relative costs. The defaults compare active time with equal
weights. This proxy does not measure joules or account for idle power, spin-down
policies or memory power. `--farming-objective latency` instead minimizes the
response bound, with work as a tie-break. Both modes must satisfy the same hard
deadline and eligibility limits.

The JSON report includes `response_bound_seconds`, `feasible` and
`infeasibility_reason`, along with `dynamic_work_seconds_per_sp` and its CPU/I/O
components. `max_eligible_groups` is the configured per-resource limit;
`certified_max_eligible_groups` is the assessed maximum. An infeasible profile
produces a report and an error exit, without publishing a new plan or starting
plot production. Adding destinations must pass a joint assessment of the new
and existing assignments using the saved identities and profile.

The search includes every balanced file count within its bounds. It compares
joint file counts and refines individual directories until no such move
improves the result. Mixed farms do not receive an exhaustive comparison of all
possible assignments. Work and memory limits are enforced explicitly. Selection
is conditional on the declared service envelopes and search coverage; it does
not establish measured energy use or physical farming performance. Exact
file-format and construction admission still apply when plotting.
[Chia's farming estimates](https://docs.chia.net/chia-blockchain/consensus/proof-of-space-2.0/new-proof-farming-requirements/)
provide context for the tradeoff; qualify the intended storage and reader before
relying on a latency bound.

### Replacing an existing farm

Ordinary `fill` uses available space without deleting existing plots. Replacing
an existing farm requires an explicit inventory, a saved migration review and
execution of that review. Execution needs a compatible Chia harvester, farmer,
started PoS2 solver and synced, activated full node, with complete resource
bounds for the selected construction backend. Linux and native Windows use
native file claims and bounded worker processes. A platform without the required
telemetry or process controls refuses admission. See the
[compatibility guide](contrib/pos2-pr118/README.md) for PoS2 reader support.

Replacement routes include solo pool contracts, external pooling and OG pool
public keys. For an external pool, the farmer must report the configured public
route, supported pooling protocol, initialized authentication parameters and
current difficulty. The review must cover that protocol and a conservative
minimum difficulty. These RPC observations do not prove pool registration, a
current authentication token, possession of an OG signing key or acceptance of
a future proof. The replacement's exact public-key route is acknowledged after
publication; migration does not wait for a random winning proof or pool partial.
The selected membership and contract hash must already be appropriate for the
pool. The [current Pool2 protocol](https://github.com/Chia-Network/pool2-reference#how-it-works)
uses a RewardPuzzle route tied to the launcher ID and versioned authentication;
an older contract address is not evidence of that membership. Migration does
not create or upgrade a Plot NFT or change the wallet's pool membership.

Inventory every loaded plot root of each configured source and target reader,
and include existing destination data. The optional installed-reader arguments
add authenticated observations from the existing services:

```bash
xchplot2 plan --inventory /disk-a/plots /disk-b/plots \
  --chia-root /home/farmer/.chia/mainnet --chia-executable /opt/chia/bin/chia \
  --save-plan inventory.plan --json
```

This reads native file identities, allocated and logical sizes, supported
public headers and storage sharing. It exports no plot memos, private keys or
seeds and starts no farming services. Without `--save-plan`, it only reports
the inventory. Saving refuses an existing file. Paths, public identifiers and
hostnames remain in the report, so review those before sharing it.

File names and extensions do not establish a format or deletion eligibility.
Supported PoS1 headers can become retirement candidates after rechecking the
same native files and parsed public header, including ordinary files currently
unloaded from the reader. A present loaded record must agree with those fields.
Proprietary PoS1 containers can also become candidates when
an explicitly selected installed reader reports consistent public plot metadata
for that exact native object. Filename and size alone are insufficient: native
identity, timestamps and the observed header bytes are pinned, and the reader's
plot public key and pool route must bind its reported plot ID. This identifies
the reviewed object without reconstructing old proofs or claiming its contents
are valid. Unknown files without that metadata, links, aliases and unreadable
or changed objects stay protected. Existing PoS2 files remain part of the
workload and are not PoS1 retirement candidates.

`--farm-adapter chia|gigahorse|drchia` selects the installed reader interface;
`--chia-executable` must name its actual executable. Each source phase captures
and fences a coherent bounded reader registry for the selected batch. After
construction and verification, migration refreshes source metadata before
retirement and checks each reviewed source and target reader's loaded scope
before each claim and unlink. Unexpected arrivals stop retirement. Reader,
configuration or reported version changes invalidate the review, apart from
the reviewed source-directory removal described below. DrChia builds
without a version RPC record explicitly labelled CLI version provenance instead;
that is not an independently reported running-service version. Unloaded ordinary
files record the selected reader's route context separately from their public
header identity; this does not claim that the reader loaded or decoded them.
Native source claims retain the exact reviewed object across interruption and recovery.
Registry calls, catalog traversal and native object reads have explicit time,
memory and output limits. Retirement requires no vendor proof checker or GPU
reconstruction of the old plot.

For different source and replacement readers, add a `farm_adapters` object
beside `farm_resources` in the local `--harvester` profile:

```json
{
  "schema": 1,
  "target_route": "new-chia",
  "routes": [
    {"id": "old-gigahorse", "kind": "gigahorse",
     "root": "/home/farmer/.chia/gigahorse", "executable": "/opt/gigahorse/chia",
     "source_roots": ["/disk-a/plots"]},
    {"id": "new-chia", "kind": "chia",
     "root": "/home/farmer/.chia/mainnet", "executable": "/opt/chia/bin/chia",
     "source_roots": []}
  ]
}
```

Each source root must match one explicit route. Pass this local profile during
inventory, review and execution; saved public plans do not authorize executable
paths. The resource declarations must also match the saved review on resume.
An unexpected loaded plot stops progress until the catalog and workload are
reviewed again. During execution, only the catalog and replacements already
published by the private journal may appear in those readers' loaded registries.
Explicitly classified metadata artifacts remain protected; a reader reporting
one as a loaded plot contradicts its zero-workload classification and is refused.

For authenticated remote services, a route can also declare `host` and
`services`, keyed by `harvester`, `farmer`, `full_node` and `solver`. Each service
entry supplies an absolute local `root` containing its existing private-CA CLI
configuration, `rpc_host` matching that configuration's `self_hostname`, and
`host` naming its physical resource host. Harvester, full-node and solver entries
also supply `node_id`, the service TLS identity seen by the farmer, and
`farmer_node_id`, the reciprocal farmer identity seen by that service. The
farmer's public and private connection identities can differ. These local
bindings select the authenticated contexts; a saved public plan imports none of
the executable paths or authentication authority.

Replacement and source paths must identify the same accessible native files for
the controller and selected harvester, such as an explicitly shared filesystem
with matching paths. RPC supplies public loaded metadata, not remote native
inode attestation. A matching pathname on an unrelated remote filesystem does
not establish the controller's exact replacement. Configure the actual shared
file namespace and physical storage/resource mappings before review.

The inventory contains an editable `farm_resources` template with unknown
capacities and costs. Complete it for the actual mixed farm before reviewing a
migration. Use the existing [harvester profile workflow](#farming-layout) for
the intended PoS2 reader, and include the completed `farm_resources` object in
the profile supplied through `--harvester FILE`. Keep each inventory entry's
`workload_id` consistent with the profile's counts. Opaque data also needs a
workload reservation unless explicitly classified as `non-plot` with a reason.
That classification never makes the file a retirement candidate.

| Profile part | Required scope |
| --- | --- |
| `resources` | Physical host, CPU, host memory, GPU compute and memory, storage, controllers, network and solvers actually shared by the work. Aliases identify one resource, not extra capacity. Memory capacities are bytes; other capacities are concurrent service lanes. |
| `workloads` | Format, decoder/version, backend, pool difficulty, current and future quality/full-proof bursts, response deadline, provenance and complete resource costs for each existing cohort. |
| `uses` | Resident and transient memory, full carried-in queue bounds, and either explicit reserved lanes or bounded service times. Service times are microseconds; memory use is bytes. |
| `native` | Bounds for the selected construction and verification settings, physical output/scratch/source mappings, controller and RPC supervision, and increasing member cohorts with separate new-farming and full-proof reserves. |
| `gate` | Maximum telemetry age and the recovery interval/sample count required before admitting work again. |

`native.service_hosts` maps service roles to their physical resource hosts;
omitted roles use `native.host`. These labels must match the local adapter
bindings. For external pooling, set `native.min_pool_difficulty` to the lowest
difficulty covered by the full-solver envelope and `native.pool_protocol_version`
to the configured protocol. A live difficulty below that floor, or a different
protocol, stops further retirement. Higher difficulty remains within the
reviewed arrival bound. Solo and OG routes leave both values zero.

Additional loaded farmer-side harvesters require reviewed aggregate workloads with
`external_service_node_id`, `external_service_host`, `plot_count` bounding loaded
files, and `external_service_members` bounding their full member population.
Their costs cover the receiver host and the shared farmer/solver resources.
These unscanned service reservations remain charged throughout migration and
cannot be retired as inventoried files. Unknown receivers or populations beyond
the reviewed bounds stop progress.

Unknown costs or capacities block admission. Declared reservations must cover
the entire workload; changing a flag to `bounded` does not measure that bound.
Partial-proof calibration does not supply a full-proof solver reserve. GPU
names and ordinals are not physical identities: supported CUDA/HIP selections
report their PCI identity with `devices --verbose`. A shared GPU needs a
complete conditional service envelope; it has no guaranteed preemption or
isolation from another application's queues. Imported remote resource profiles
can support forecasts but do not grant control over remote services.

Native envelopes must explicitly cover future filters and the controller's
memory in each phase (`covers_future_filter` and
`phase_totals_include_controller`). Construction, proof verification and RPC
supervision run sequentially. The proof worker has a memory and time limit;
`verification_cpu_threads` can additionally limit its CPU affinity.
Each phase's CPU envelope includes its concurrent controller or monitor and
capped worker, within the inherited CPU set.
GPU session envelopes cover the full inherited CPU set in every phase:
retained runtime helpers are not constrained by the host worker limit.
They also require an explicit `standing_gpu_bytes` resident-memory bound in
every phase for retained contexts, runtime caches and tables. Supply a complete
declared or measured bound; it is additional to the allocation grant, and the
VRAM safety margin does not substitute for it. GPU preflight can perform bounded
active device probes.
Construction and verification service costs cover the entire file up to
`max_group_members`, including all internal batches and windows.
`max_batch_members` limits an individual construction batch; a measured cost
for one batch is insufficient for a larger file. An operation timeout does not
measure or enforce its per-resource service cost.
Declared bounds must cover every permitted execution setting, including smaller
worker, batch and memory limits; measurements at one setting do not establish
that envelope.
For resources marked `conditional_shared`, admission adds all declared work and
queued blocking time and checks response deadlines. This permits conditional
sharing of a CPU or disk lane; it does not establish a measured service bound.

Shared compute also needs a reviewed migration envelope. `native.compute_workers`
maps each worker's authenticated `tls_identity` to its physical `host` and,
where used, `gpu_identity`. Declare the configured `cpu_threads`, `slots`,
`host_bytes`, `gpu_bytes`, standing host/GPU memory and permitted `allowed_kinds`
(`generate-cpu`, `generate-gpu` or `encode`). Each worker's `reservation` covers
its entire service, including work from other owners; idle status supplies no
extra capacity. The host-memory partition already includes the worker baseline.
`native.compute_owner_identity`, `compute_broker_identity` and
`compute_broker_host` bind the owner and broker; `compute_services` reserves the
persistent session/codec and broker resources. These are explicit reviewed
physical mappings, not remote hardware discovery or permission to reconfigure a
foreign worker. Missing mappings or grants beyond the envelope block admission.
Protected migration gives the owner/client/codec and the worker independent
CPU gates. Its reviewed envelope reserves both workloads in full, so a local
offer may exceed `--workers` when physical capacity and complete phase bounds
permit it. Ordinary plotting retains its shared owner/worker gate.
Local GPU worker reservations also cover the complete inherited CPU affinity
for runtime helpers, as well as offered task CPUs and GPU allocation/residency.

Review destinations and the completed profile, then save the resulting plan:

```bash
xchplot2 plan /disk-a/plots /disk-b/plots --replot inventory.plan \
  --harvester mixed-farm.json --compute off --devices cpu \
  --chia-root /home/farmer/.chia/mainnet --chia-executable /opt/chia/bin/chia \
  --save-plan migration.plan --json
xchplot2 fill migration.plan --compute off --devices cpu \
  --harvester mixed-farm.json \
  --chia-root /home/farmer/.chia/mainnet --chia-executable /opt/chia/bin/chia \
  -f "$FARMER_PK" -c "$POOL_ADDRESS"
```

The review fixes the selected source files, replacement batches, destination
identities and resource assumptions. Inspect the old formats, k sizes, file
counts and allocated bytes alongside the new k size and member counts.
Physical bytes do not imply equivalent farming capacity, rewards or fees;
unavailable value estimates remain unknown. A smaller compressed file need
not use fewer farming resources.

`--reclaim-before-bytes SIZE` belongs to the review and defaults to zero.
Zero requires each mapped replacement to be ready before its sources are
removed. A positive value permits that many allocated source bytes to be
retired while their mapped replacements are not yet ready, across interruptions
and restarts. Removal is irreversible: the new plot cannot reconstruct an old
one. This limit measures physical bytes, not lost income or equal farming
value. Temporary space selected with `--temp-dir` is construction scratch;
it is not automatically a farmable replacement.
External construction with pre-delete credit also requires a joint output and
scratch admission. The review and execution check the actual temporary roots,
filesystem sharing, full output/checkpoint reservations and outstanding deletion
debt before claiming any source. An inadequate `--temp-capacity-bytes` or
unavailable scratch root refuses the deletion; credit cannot stand in for a
complete scratch grant.

Each batch must fit conservative output, construction, recovery and headroom
allowances. Execution checks actual available space again. A rename or
quarantine frees no capacity; open handles, snapshots or filesystem behavior
can delay its release after deletion. Extra free space does not enlarge the
reviewed replacement schedule.

For replacement-first retirement, execution seals and checks the exact replacement,
runs bounded full-proof samples that must yield a valid complete proof, and
publishes a new immutable `.plot2` file. Sampling does not inspect every chain
or prove every byte of the file correct.
The configured harvester must report that exact path with matching plot
identity, public-key route, size and modification time. The target network
must be synced, active for PoS2 and compatible with the replacement parameters.
The connected farmer must acknowledge the same exact plot and key route in a
completed plot synchronization. Multi-member replacements require its exact
reported group membership. Matched peer observations bind the selected
harvester, farmer, full node and solver, using explicit reciprocal TLS identities
for remote links. Other receivers must fit their reviewed service reservations.
A bounded phase captures the complete loaded prefix once. Before each protected
retirement action it rechecks network activation and sync, service connections,
completed plot-sync generations, pool route/difficulty and native paths of all
acknowledged replacements. Construction or proof work invalidates that phase;
changed generations require a fresh exact prefix audit. Completed synchronization
can remain valid while services are idle. Imported JSON cannot assert readiness.
Missing capabilities, changed populations or stale observations pause work.
An authorized pre-deletion still requires the live target network and fresh
readiness of previously credited replacements.
General service and configured public pool-route prerequisites apply before private
preparation and pre-deletion; exact new plot/key acknowledgement follows
publication. Missing or changed reviewed farmer/solver versions block execution
while network-only forecasts remain available.

Resource admission includes the old farm, completed replacements, construction,
verification and separate full-proof demand, under current and future filter
conditions. Retiring a plot does not release a foreign decoder's resident
memory or queued work. Telemetry restricts native admissions and interrupts
construction or proof work if protected headroom is lost. Construction saves a
checkpoint; a paused proof attempt restarts against the sealed replacement
within its original verification deadline. Telemetry does not stop or
reconfigure the farmer, harvester, remote solver or their services.
The reviewed envelope must also allow fresh observations after a restart while
conservatively retaining the original arrivals until their removal is confirmed.
The first complete target farming and full-proof cohort is reserved even when
no replacement exists, covering the services that must already be running.

Keep the original `migration.plan`, every private `grow-replot.plan` copy and
any `.next` file, receipts and owned temporary files. The private journal
contains the seed and assignments from preparation onward. Resume with the
same reviewed plan and installed-reader arguments; saved private state restores
the keys. All required source and destination roots must be available and must
still identify the same directories. A source mismatch stops retirement while
preserving the claimed file for inspection. On Windows, a reader that holds a
source without delete sharing must unload it before the native claim can open.
Keep the reader's RPC service running. Recovery of a batch with durable
retirement progress can retain its previously validated metadata for those exact
now-unloaded native objects. A later opaque batch without previously validated
reader metadata must be loaded and freshly acknowledged before its own retirement
begins. Ordinary supported public headers can identify unloaded files directly.

New reviews can record a removal-compatible fingerprint for conventional Chia
configuration. Once the private journal has durable retirement intent, removing
that reviewed source directory from the reader's configuration is permitted.
Keep all replacement directories configured and preserve every other setting,
executable and service binding. Removing an ancestor of a replacement directory
is refused.
This also applies when the source and target share that configuration; fresh
target acknowledgements remain required. Directory additions are refused.
Native configuration rewrites may reorder keys or remove comments without
changing the other settings. Older reviews and unsupported configuration syntax
retain the exact-configuration requirement; keep or restore the reviewed
configuration to resume those jobs.

Migration currently requires unchanged native file and device identifiers;
device renumbering after a remount or reboot can prevent resume even when the
filesystem's persistent identity is unchanged.

After all replacements and retirements pass the final audit, the job publishes
ordinary `grow-farm.plan` copies. Completed replacement files retain their
identities and fixed member counts; ordinary recovery never grows them.
An unfinished migration must finish through its original review before ordinary
fill or extension can operate on its destinations.
Reassess any remaining legacy workload before extending a completed ordinary job.

## Shared compute

A compute pool is a set of trusted machines that share generation and encoding
capacity. It is independent of the Chia pool destination used for plot keys.
A plotting host owns its files, checkpoints and publication. A broker assigns
leases; workers return computed fragments directly to the plotting host.
Workers never receive destination paths, farmer/pool keys, seeds or private
plot memos. Enroll only machines you trust to compute correctly.

On the first machine, create a pool and an invitation for another machine:

```bash
xchplot2 compute init --name storage-a
xchplot2 compute invite worker-b.invite --name worker-b
```

Transfer the invitation privately to that machine, then run there:

```bash
xchplot2 compute join worker-b.invite
xchplot2 compute worker
```

Create a separate invitation for each additional machine. An invitation holds
that machine's private TLS identity, not the pool's signing key. Delete
transferred copies after joining. Enrollment is pool membership: every member
can share with multiple plotting hosts. It is not an exclusive machine pair.

After enrollment, `plot`, `fill`, and `grow` use shared compute automatically.
Each job starts an embedded broker unless a dedicated broker is configured.
Its local executor also offers bounded capacity while the job runs; nearby
workers discover the broker and join. Several brokers can use one worker at
once. The worker enforces one combined CPU, RAM, GPU and task-slot budget
across their requests. One executor per enrolled machine identity prevents
concurrent local jobs from advertising the same capacity twice.

Run `compute worker` to keep offering capacity independently of a plotting
job; it stays in the foreground until stopped. Save defaults for future jobs:

```bash
xchplot2 compute configure --cpus 4 --host-bytes 8G --gpu-bytes 4G
xchplot2 compute status
```

A standalone worker defaults to half the logical CPUs, half physical RAM
capped at 16 GiB, one task slot, and automatic CPU/GPU selection. Automatic
GPU capacity is 80% of the smaller of reported free and total VRAM. A plotting
job partitions its own host budget between the embedded service, local work,
construction and optional remote encoding, honoring saved offered limits.
A worker registers with up to 128 brokers by default. Save another limit with
`compute configure --max-brokers N`, or override it with
`compute worker --max-brokers N` (1..4096). Each registration consumes reserved
transport RAM; automatic local executors lower their registration limit to
fit the offered memory. Dedicated workers enforce the saved or explicit limit;
when lowering their host RAM, lower `--max-brokers` too if startup reports an
insufficient transport budget. For example, a 512 MiB CPU worker can use
`compute configure --max-brokers 16`.
Status reports deferred brokers when the limit is
reached. Idle registrations rotate to admit waiting brokers; granted or active
work keeps its registration until its reservation is released.
For large registries or slow links, raise each dedicated broker's `--worker-ms`
above the full refresh cycle; the default registration expiry is 30 seconds.
Refreshes use bounded concurrency, so an oversized limit can otherwise cause
idle registrations to expire between visits.
Explicit service flags override saved defaults. Admission and retries adapt
to available workers and measured task/transfer cost; discovery alone does
not guarantee that a worker has enough capacity for a particular task.

`compute status` authenticates live brokers and workers and summarizes their
availability and work. Add `-v` to see saved defaults, certificate expiry,
identities, owners, jobs, queue wait reasons, offered resources, broker
reservations and actual worker usage. A worker shared by
several brokers enforces its total limit locally; a broker's reservation view
is not a pool-wide capacity measurement.

```bash
xchplot2 compute status --json
xchplot2 compute status --broker 192.168.1.10:18444 --limit 256
xchplot2 compute pause
xchplot2 compute drain
xchplot2 compute resume
```

`pause` stops new task admission while accepted tasks finish. `drain` also
changes to `drained` after accepted work finishes; `resume` reopens admission.
Only the worker's own enrolled identity can control it. These settings last
for that worker process. Use `--worker HOST:PORT` to query or control a worker
when discovery is unavailable. Explicit endpoints select the services to
query; otherwise status uses saved broker settings and LAN discovery.
`--limit` bounds each broker section (default 128, maximum 4096);
`--timeout-ms` bounds network queries (default 5000). JSON includes errors and
truncation indicators. Live pages can change while being read.

Discovery advertisements are untrusted hints. Status and work connections
require mutual TLS in the enrolled pool. Use the same build on collaborating
machines; incompatible compute formats are not scheduled. Upgrade brokers,
workers and plotting hosts together, then restart their services. Preserve
the saved enrollment directory when upgrading.

### Background workers and brokers

Save resource and network defaults with `compute configure`, then install a
service for this enrolled machine:

```bash
xchplot2 compute service print
xchplot2 compute service install
xchplot2 compute service status
xchplot2 compute service restart
xchplot2 compute service remove
```

The default role is `worker`; add `--role broker` for a dedicated broker.
`print` displays the definition without installing or starting it. `install`
starts the service and arranges subsequent startup. `start`, `stop`, `restart`
and `remove` act only on the service generated for this identity. Use
`compute drain` and wait for `drained` before stopping if accepted tasks should
finish first. Restart after changing saved defaults or replacing the executable.

Linux uses a systemd user service; logout behavior follows the system's user
manager policy. Windows uses Task Scheduler at user login, without elevation,
and requires that user to remain logged in. The native Windows service path
is experimental: rapid stop/start/restart can leave the worker stopped on
x86-64. Check `compute service status` and `compute status` after restarting.

### Networks and dedicated brokers

LAN discovery uses mDNS/DNS-SD on active multicast interfaces. It excludes
loopback and point-to-point VPNs unless selected explicitly. Sharing listeners
accept the selected networks plus loopback. Interface changes are refreshed
automatically. IPv4 and IPv6 are supported; write IPv6 endpoints as
`[address]:port`, with a scope on link-local addresses such as
`[fe80::1234%enp3s0]:18444`. On Windows, select an adapter by its name or
interface index; `lo` selects loopback on either platform. Choose interfaces
explicitly when needed:

```bash
xchplot2 compute configure --interface enp3s0 --interface wg0
xchplot2 compute configure --interface auto
```

Discovery stays on the local link. Routed networks and VPNs can use a direct
broker address. Run a dedicated broker on an enrolled machine, then configure
plotting hosts and workers to use it:

```bash
# Dedicated machine; default TCP port 18444:
xchplot2 compute broker
# On participating machines:
xchplot2 compute configure --broker 192.168.1.10:18444
# Return a plotting host to an embedded broker per job:
xchplot2 compute configure --broker auto
```

`compute worker --broker HOST:PORT` can repeat direct addresses and also
browse discovered brokers. `--discovery off` disables discovery; a standalone
worker then needs a direct broker. Broker and worker `--listen` / `--advertise`
options set fixed reachable endpoints. Firewalls must permit UDP 5353 for
discovery and the chosen TCP listener ports; automatic embedded services and
workers choose available TCP ports. Use explicit interfaces for a VPN rather
than exposing the services publicly.

The saved setup is in `$XDG_CONFIG_HOME/xchplot2/compute`, falling back to
`$HOME/.config/xchplot2/compute`; Windows instead defaults to
`%LOCALAPPDATA%\xchplot2\compute` when `XDG_CONFIG_HOME` is unset.
`compute ... --directory DIR` or plotting's
`--compute-directory DIR` selects another setup. Preserve it for recovery:
the stable node identity scopes fixed seeds, so the same supplied seed on two
enrolled plotting machines creates different plot identities. Private saved
manifests can be resumed directly with `batch` without re-deriving keys.
Old ordinary manifests continue to use their original engine; use
`plot --compute off --resume` when resuming one after enrollment.

### Overrides and interruption

`--compute off` disables automatic sharing for a plotting command.
`--compute-local off` uses pool workers without offering local capacity.
`--compute-broker HOST:PORT` overrides the saved broker for one job.
Advanced direct connections can still supply `--compute-owner-id`,
`--compute-ca`, `--compute-cert` and `--compute-key`; the broker and worker
help list their corresponding TLS flags. No enrollment files are needed for
that explicit interface. It does not keep a signed membership registry or
propagate policy updates; use saved enrollment for managed revocation and
renewal. An enrolled peer still applies its own policy to direct connections.

`--compute-owner-id ID` also overrides an enrolled machine's storage-owner
namespace and keeps a stable broker owner for the broker's `--owner-share ID:WEIGHT`
setting. Use a distinct explicit ID for each concurrent job. Without this
override, automatic jobs get separate temporary broker owners while retaining
the machine's stable key namespace.

`--compute-inflight N` bounds outstanding work, `--compute-codec-ram auto|SIZE`
controls the optional encoding partition (`0` disables it), and
`--compute-timeout-ms N` / `--compute-socket-timeout-ms N` bound task and socket
waits. The defaults are one hour per task and 30 seconds per socket operation.
`--compute-frame-bytes SIZE` bounds each transport frame. Local offer overrides
are `--compute-offer-cpus`, `--compute-offer-ram`, `--compute-offer-vram`,
`--compute-offer-slots`, and `--compute-offer-backend`. `compute worker --help`
lists standalone task, thread, window and transport limits.

Workers may arrive or leave while jobs run. Lost or expired work is retried
without publishing incomplete results. Available local compute permits
fallback during broker outages. A storage-only job waits within its task
deadline, then reports failure while preserving recovery state. Broker
restarts fence old leases; plot commits remain on their storage owners.
Stopping a job also stops any broker and executor it owns. Other local jobs
can take over the executor and retry unfinished work.

### Membership renewal and revocation

Enrollment files and invitations use owner-only permissions. The pool creator
holds the signing authority. Inspect its registry and revoke a stable node ID:

```bash
xchplot2 compute members
xchplot2 compute revoke NODE_ID
xchplot2 compute policy-export pool.policy
```

Revocation applies to that node, including replacement certificates. Signed
policy updates propagate over authenticated connections and are checked on
existing connections before further operations. Disconnected members cannot
learn a new revocation until they reconnect or import the policy with
`compute policy-import pool.policy`. Distribute it to every remaining member
when excluding a machine. The registry holds up to 1024 stable nodes. Members
created by an older build may be absent from the registry until renewal;
the authority can still revoke their node IDs explicitly.

To rotate a machine's certificate while preserving its node ID and seeded
plot namespace, run these commands on the indicated machines:

```bash
# Machine being renewed:
xchplot2 compute renew-request machine.request
# Pool creator, after receiving that request:
xchplot2 compute renew-sign machine.request machine.replacement
# Original machine, after receiving the replacement:
xchplot2 compute renew-import machine.replacement
```

Restart the renewed machine's services to use the new credentials. Repeat the
same command and file paths if interrupted; renewal retains the pending key
and signed response for recovery. Certificates last at most one year and
cannot outlive the pool authority. An expired leaf can be renewed while the
saved private key and authority remain valid. The authority lasts ten years;
its expiry or loss of its signing key requires a new pool and re-enrollment.
Lost machine credentials also require re-enrollment unless a private backup
is available. Keep plot manifests for recovery of existing output identities.

## Configuration and argument files

`--config FILE` loads a configuration file. Without it, xchplot2 looks for
`$HOME/.config/xchplot2/config.toml` (Windows: `%APPDATA%\xchplot2\config.toml`).
The supported syntax is a small TOML
subset: named sections and scalar `key = value` entries, with double-quoted
strings and `#` or `;` comments. Arrays, nested tables, and multiline strings
are unsupported.

Use long option names without `--`. `[defaults]` applies to every command;
a section such as `[plot]`, `[fill]` or `[bench]` overrides those defaults.
Compute service sections use `[compute.broker]` and `[compute.worker]`. Explicit
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

Numeric arguments must consume the whole value: `28junk` is rejected rather
than interpreted as `28`. Decimal integers retain optional signs and leading
whitespace; size arguments retain their documented units and fractional forms.

Later flags override earlier values. `@~/bench.args` expands the leading
`~/` in the argument-file path. Inside the file, `#` starts a comment;
shell quoting, variable expansion, and nested argument files are unsupported.
Use the configuration file or directly quoted CLI arguments for paths with
spaces. For `plan` and `fill` destination lists, use
[`--paths-from` or `--paths0-from`](#grouped-plotting-and-drive-filling).
Pass `--config` directly on the command line, not inside an argument file.

## Plot indices and meta groups

Both are v2 PoS fields and default to 0. `plot -n N` writes N independent
`.gplot` groups with fresh keys. `--group-members M` chooses M members per
file; every file starts at member index zero.
`--seed` and saved-job resume preserve those identities. The meta group (u8)
selects a challenge-isolation boundary. Every member of a group shares its
group ID and memo; its ID is SHA-256 of `group_id || u16_be(index) || meta_group`.

The official version 2 format from
[pos2-chip PR #118](https://github.com/Chia-Network/pos2-chip/pull/118)
stores a group size and assumes member indices start at zero. Installed
`plot`, `fill` and `grow` commands write this ordinary format. The separate
[experimental grouped-job runner](contrib/pos2-pr118/README.md#experimental-grouped-jobs)
retains shared-ID files with nonzero starting indices. Those files require
the explicitly marked 0x82 reader extension, which unmodified upstream
readers cannot read.

PoS2 1.0 changes the Feistel cipher as well as the file format. Old raw plots,
earlier experimental groups, and saved jobs require replotting; renaming or
rewriting their headers is insufficient. `--testnet` / `-T` are rejected:
PoS2 no longer has a separate testnet Xs hash. Both pool modes now use the
V2 taproot derivation in
[chia-blockchain PR #21500](https://github.com/Chia-Network/chia-blockchain/pull/21500).
[Farmer plot-group integration](https://github.com/Chia-Network/chia-blockchain/pull/21566)
merged on 2026-10-28. Development integration and format/proof interoperability
do not establish support in a released farmer or network activation.

## Devices and CPU workers

`xchplot2 devices` lists the IDs accepted by `--devices`. Without an enrolled
compute pool or grouped construction, plotting uses one GPU by default. CPU plotting is opt-in and uses pos2-chip's CPU
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

`XCHPLOT2_CPU_NO_PIN=1` disables pinning for comparison. Use
[`bench`](#benchmarking) on your machine before increasing worker counts.

### Batch completion and worker rates

With `-v`, multi-worker jobs report each worker's rate and suggested batch
sizes that reduce idle time at the end. A tail guard stops a slower worker from starting
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

Sharding is experimental and GPU-only. Compare it with the independent work
queue using [`bench`](#benchmarking) on your hardware.

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
unfarmable `.gplot` files (random group IDs, no keys). The default report shows
seconds per plot, TiB/hour and a write-time estimate. `-v` adds TiB/s, TiB/day,
TiB/month (30-day basis) and detailed measurements. Use k28 to measure farming
throughput; smaller sizes report test-size results:

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
| `-k K`, `-s S` | Plot size and strength; defaults 28 and 2 |
| `-n N`, `--num N` | Measured plot count used to size the queue; default 10 |
| `--warmup W` | Initial completions excluded per worker; default 1 |
| `-o DIR`, `--out DIR` | Directory for real output writes; default current directory |
| `-d`/`--devices SPEC`, `--cpu`, `--cpu-workers N` | Select the [devices and CPU workers](#devices-and-cpu-workers) to measure |
| `--tier T` | Force a [streaming tier](#memory-requirements), even when the pool fits |
| `-M`/`--max-host-ram SIZE`, `-t`/`--temp-dir DIR`, `--no-auto-spill` | Use the normal [host memory and spill policy](#host-ram-and-disk-offload) |
| `--keep` | Retain synthetic output files on disk for inspection |
| `--json` | Also emit a JSON report to stdout for saving or comparing runs |
| `--compute-only` | Add a second pass using RAM-backed output when available |
| `--target-size TiB` | Estimate time to fill this capacity instead of the output directory's free space |
| `-v`, `-q` | More worker detail, or quieter informational output |

The queue contains `(warmup + num) × workers` plots. Faster workers can
complete more of that queue; `-n` does not guarantee equal per-worker counts.
Each worker's warmup is excluded separately. `-v` adds the steady-state
window and detailed per-worker measurements to the report. Increase `-n` if a short run
cannot measure a slower worker.

Every pass includes FSE compression and real file writes. The second
`--compute-only` pass uses tmpfs if there is enough room; otherwise it warns
and reports a `compute+cache` pass in the output directory. Inspect that label
before interpreting the difference as disk overhead.

Bench removes its generated files by default. `--keep` retains disk output;
temporary tmpfs output is always removed to release the RAM. Results and
time-to-fill estimates are printed to stderr. See [BENCHMARKS.md](BENCHMARKS.md)
for the project's recorded measurements and comparison methodology.

Use `xchplot2 bench -k 28 -o /scratch --json > bench.json` to save the
measurements. The report includes the build version, plot parameters, queue
size and warmup, output directory, requested tier and host RAM policy, plus
per-pass throughput, measurement windows, worker labels, actual recorded
pipelines, and per-worker counts and interval spread. An empty pipeline means
that the worker did not record one; the requested tier is not an assertion
about the pipeline actually used. `workers_unmeasured` retains the human
report's lower-bound warning. The second pass is labeled `tmpfs` or
`compute+cache` as appropriate. Human output and progress remain on stderr;
without `--json`, stdout stays unchanged. Valid UTF-8 is preserved; invalid
filesystem bytes are represented individually as `\u00XX` escapes. Compare
runs under similar device
load: concurrent GPU use can change results, and this report does not diagnose
why a rate changed.

## Memory requirements

At k=28, auto selection tries the persistent pool, then Plain, Compact,
Minimal, and Tiny. Pinned is a separate manual tier, not a smaller automatic fallback.
The owning driver supplies free VRAM after context creation. A missing
reading or insufficient `peak + buffer` causes refusal before allocation.

Integrated Intel GPUs share host memory. Admission uses the driver's free
host-memory reading, bounded by physical host RAM and the runtime's allocation
cap. CPU allocations and cache activity affect those counters, so they cannot
measure dedicated GPU memory use. `POS2GPU_ASSERT_VRAM=1` is unavailable on
these devices; use `POS2GPU_ASSERT_VRAM=0` for throughput benchmarks. CPU byte
parity and full-proof checks still apply to plotting correctness.
See the [integrated Intel limitation](#troubleshooting) before production use.

The streaming base peaks below come from [VramBudget.hpp](src/host/VramBudget.hpp).
They are allocation models, not the desktop driver deltas or host RSS in
[BENCHMARKS.md](BENCHMARKS.md#gpu-results).

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

Lower VRAM tiers generally need more host RAM. Set `--max-host-ram` to leave
room for the OS and other workers; see [host RAM and disk-offload](#host-ram-and-disk-offload).
Host admission uses a separate model of pinned and anonymous memory, plus a
reserve. Eligible storage can be moved to disk when that model does not fit.

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
  endurance from its write volume.
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
xchplot2 test <k> <group-id-hex> [strength] [plot-index] [meta-group] [verbose]
```

This accepts a 64-character hex group ID and derives its member ID. It uses CPU phases by default;
`-G` / `--gpu-all` selects all available GPU phases, while `--gpu-t1`,
`--gpu-t2`, and `--gpu-t3` select individual phases. `-P` / `--profile`
prints phase timings. Use `-m` / `--memo HEX`, `-o` / `--out DIR`,
`-N` / `--out-name NAME` for output parameters. `--raw` writes a temporary
raw member and allows indices 0..65535; group output requires index 0.
Use `-v`/`--verbose` for diagnostics; the positional verbose argument remains
supported, with an explicit flag taking precedence.
The [CPU-reference fixture](CONTRIBUTING.md#building-and-running-tests)
shows a matching group-ID test and GPU batch. Arbitrary IDs and memos do not
produce farmable plots.

### Verification

```bash
xchplot2 verify /path/to/NAME.gplot --full --trials 100
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
and failure output, and returns nonzero if a test fails. `-v` includes timings
and output from successful tests; `-q` retains the final counts and failures. The default
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

- **Integrated Intel Level Zero plotting**: this path can produce invalid
  plots and is not supported for production. Use `--devices cpu` on these
  systems.

- **Intel Arc B580 sort hangs**: on Fedora 44 with AdaptiveCpp 25.10 and
  Intel compute-runtime 26.22, disable direct submission. Apply the workaround to the plotting process:

  ```bash
  ACPP_VISIBILITY_MASK=ze NEOReadDebugKeys=1 EnableDirectSubmission=0 \
      xchplot2 bench -k 28 --devices 0 -o /scratch
  ```

  These Intel driver settings are not set automatically. Use an up-to-date
  build for the Level Zero disk-spill synchronization fix.
  With Tiny/Pinned on this stack, set `POS2GPU_VRAM_MARGIN_MB=1024` to
  accommodate the additional driver-reported VRAM use.

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
| `NEOReadDebugKeys=1`, `EnableDirectSubmission=0` | Intel driver workaround for B580 submission hangs. See [troubleshooting](#troubleshooting). |
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
| `POS2GPU_VRAM_MARGIN_MB=N` | Buffer beyond the selected modeled VRAM peak, in MiB. Default 128. Raise it when other activity takes VRAM after admission. |
| `POS2GPU_ASSERT_VRAM=1` | Fail when the watchdog's sampled device-wide decrease in free VRAM exceeds the declared budget, including its margin. Other processes can affect this measurement; repeat on an idle GPU before attributing a failure to the model. Armed by `bench`. |
| `POS2GPU_STREAMING_STATS=1`   | Log every streaming-path `malloc_device` / `free`.                      |
| `POS2GPU_POOL_DEBUG=1`        | Log pool allocation sizes at construction.                              |
| `POS2GPU_PHASE_TIMING=1`      | Per-phase wall-time breakdown (Xs / sort / T1 / T2 / T3) on stderr. The [group construction engine](#construction-and-recovery) also reports output transfers, host CPU time and controller waits. Diagnostic synchronization can change overlap; these are not kernel-busy measurements. |
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
