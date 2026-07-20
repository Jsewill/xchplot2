// BatchPlotter.hpp — staggered multi-plot pipeline.
//
// One producer thread runs the GPU pipeline back-to-back; one consumer
// thread runs the (already-parallelised) FSE compression + plot file
// write. A bounded queue of depth 1 between them lets GPU compute for
// plot N+1 overlap CPU FSE for plot N.
//
// Steady-state per-plot wall time = max(GPU_compute, CPU_FSE) instead of
// (GPU_compute + CPU_FSE). For k=28 strength=2 on the current build that's
// roughly 3 s vs 7 s — about 2x throughput.

#pragma once

#include "host/BenchStats.hpp"
#include "host/GpuPlotter.hpp"
#include "host/MultiGpuPipelineParallel.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace pos2gpu {

struct BatchEntry {
    int k = 28;
    int strength = 2;
    int plot_index = 0;
    int meta_group = 0;
    bool testnet = false;
    std::array<uint8_t, 32> plot_id{};
    std::vector<uint8_t> memo;
    std::string out_dir;
    std::string out_name;
};

struct BatchResult {
    size_t plots_written = 0;
    size_t plots_skipped = 0;  // present + skipped via BatchOptions::skip_existing
    size_t plots_failed  = 0;  // raised an exception under BatchOptions::continue_on_error
    double total_wall_seconds = 0.0;
    std::uint64_t bytes_written = 0;
    // Per-finished-plot wall offset in completion order, in seconds from the
    // RUN's epoch — one steady_clock origin shared by every worker, handed to
    // each slice by run_batch(). It used to be each worker's own start, which
    // made the merged list below meaningless: a GPU worker's origin is taken
    // after its pool construction and pinned-host allocation, so its offsets
    // ran seconds ahead of a CPU worker's on the same wall clock, and sorting
    // them together produced a timeline that never happened.
    std::vector<double> completion_seconds;  // merged + sorted in run_batch()
    // When this worker began its first plot (device init + pool construction
    // sit before it). Only load-bearing under --warmup 0; otherwise a warmup
    // completion is the epoch and init is excluded by construction.
    double work_start_seconds = 0.0;
    // Per-worker breakdown, in device order. run_batch() populates this for
    // every strategy — single-worker strategies produce exactly one entry.
    // Throughput must be derived from this, never from the merged list above:
    // workers do NOT finish an equal share of a work-queue, so per-worker
    // warmup exclusion and the drain tail are invisible once merged. See
    // BenchStats.hpp.
    std::vector<WorkerTimeline> workers;
};

// Options controlling batch behavior.
//   verbose           — per-plot progress on stderr
//   skip_existing     — if an output .plot2 already exists (and passes a
//                       lightweight magic/size check), skip the plot
//                       instead of overwriting it
//   continue_on_error — catch per-plot exceptions and log rather than
//                       aborting the batch; plots_failed in the result
//                       counts how many skipped this way
//   device_ids        — explicit list of GPU device ids to use. When empty
//                       and use_all_devices is false, run on a single
//                       device picked by the default SYCL gpu_selector_v
//                       (zero-configuration, pre-multi-GPU behavior).
//                       With multiple ids, the batch is spread across
//                       workers — one thread per device, each with its
//                       own GpuBufferPool and producer/consumer channel.
//                       Plots are NOT partitioned up front: workers race
//                       to pull the next entry off a shared queue, so a
//                       worker takes plots in proportion to its speed and
//                       a 3x-faster GPU finishes roughly 3x the plots of
//                       its CPU peer. Nothing here hands each worker an
//                       equal share, and callers must not assume one.
//   use_all_devices   — enumerate all SYCL GPU devices at runtime and
//                       use them. Overrides device_ids. Useful when the
//                       caller doesn't know the host's device count up
//                       front (e.g. `--devices all` on the CLI).
//   cpu_workers       — how many CPU workers run ON EACH selected CPU node.
//                       A COUNT, not a selection: whether the CPU is in the
//                       batch at all is cpu_selected() — see cpu_node_ids /
//                       use_all_cpu_nodes / cpu_opt_in. So this is consulted
//                       only once something has asked for the CPU, and its
//                       default is what to do THEN, not by default.
//
//                       Per node rather than per host so it means one thing on
//                       every box. On the single-socket hosts this was tuned on
//                       the two are the same number, so nothing changed.
//
//                       Encoded as kCpuWorkers* sentinels or a positive count:
//                         kCpuWorkersAuto (-1, the default once selected) — the
//                             throughput knee (~4 per node), then trimmed to what
//                             host RAM holds. Beside a GPU the knee is 1 per
//                             node instead — see "Why a knee" below.
//                         kCpuWorkersMax  (-2) — as many as fit in RAM, capped at
//                             the core count (more only oversubscribes). The
//                             "as many as make sense" escape hatch.
//                         0 — none, whatever --devices asked for. This is the one
//                             way a count speaks to selection, and it is what
//                             `--no-cpu` used to be.
//                         N > 0 — exactly N per node (still RAM-trimmed).
//                       Each worker is one CPU-node id in device_ids — see
//                       cpu_device_id() in src/gpu/DeviceIds.hpp — so they are
//                       ordinary work-queue workers. Set via
//                       `--cpu-workers auto|max|N|off`.
//
//                       Why a knee and not "fill RAM" by default: pos2-chip's
//                       plotter is memory-latency-bound, so concurrent plots
//                       interleave each other's stalls — but each already fans
//                       out to every core, so the gain plateaus fast. Aggregate
//                       steady-state, 32-thread 5950X at k=28: 52.28 s/plot at
//                       N=1, 43.85 at N=2 (+19%), 41.69 at N=4 (+25%), flat after
//                       — the 3rd and 4th together buy ~5%. Past the knee you only
//                       oversubscribe (at k=22, RAM alone would permit ~500
//                       workers, each spawning 32 threads). So auto caps at the
//                       knee; kCpuWorkersMax (RAM- and core-bounded) is there if
//                       you want to push past it.
//
//                       Those are `--devices cpu` numbers, and beside a GPU they
//                       are the wrong curve: they measure what extra workers ADD
//                       and cannot see what they COST, which is the GPU worker's
//                       host-side FSE consumer. Measured 2026-07-17 on an RTX
//                       4090 + 5950X at k=28, the knee of 4 ran a batch 2.39x
//                       SLOWER than the GPU alone (336.4 s vs 140.8 s for 55
//                       plots) as the GPU's own rate fell 2.56 → 4.23 s/plot; 1
//                       worker was a wash (2.57). So auto uses a knee of 1 per
//                       node whenever a GPU worker is in the batch. Both effects
//                       that make 4 wrong ease as the GPU slows, so a weak-GPU
//                       host may want more — measure it with
//                       XCHPLOT2_CPU_AUTO_WORKERS before assuming.
//
//                       It also costs a full copy of the plotter's working set
//                       per worker (12.14 GiB at k=28), which is why
//                       resolve_batch_devices trims the count against host RAM,
//                       subtracting each GPU worker's host footprint first.
//   streaming_tier    — optional manual override for the streaming
//                       pipeline tier (when the GPU pool doesn't fit).
//                       Accepted values: "plain" (~7.24 GB floor at k=28,
//                       ~10-15% faster), "compact" (~5.33 GB floor, fits
//                       on tight 8 GB cards), "minimal" (~3.76 GB floor,
//                       4 GB cards), "tiny" (~1.5 GB floor target, sub-
//                       4 GB cards — scaffolding only as of this commit;
//                       runs the minimal-tier path until tighter slicing
//                       lands). Empty string = auto (pick the largest
//                       tier that fits free VRAM). Equivalent to
//                       XCHPLOT2_STREAMING_TIER env var but settable via
//                       --tier on the CLI; the struct field takes
//                       precedence over the env var.
//   shard_plot        — opt in to single-plot multi-GPU mode. Default
//                       (false) keeps the existing work-queue dispatch:
//                       N workers, one plot each, round-robin. With
//                       shard_plot=true, the workers form a "team" and
//                       run plots one at a time, each owning a shard
//                       of every plot. Phase 1 scaffold lands the
//                       option but only supports N=1 (no-op fall-through
//                       to single-GPU); N > 1 throws a clear error
//                       until later phases implement the real sharding.
//                       See docs/multi-gpu-single-plot-*.md for the
//                       design.
//   shard_strategy    — partition strategy when shard_plot is on.
//                       Reserved for the multi-GPU work; "bucket" is
//                       the planned default (output-bucket partition
//                       with distributed radix sort). "section_l" is
//                       the alternative (input-section partition with
//                       gather-sort-scatter). Ignored when N=1.
// Phase 2.4 batch-strategy picker. Auto = pick at runtime based on
// device VRAM and k. WorkQueue = N independent plotters round-robin
// (default for equal-VRAM PCIe-only rigs where each card fits the
// pool). PipelinePlot = N-stage split (--pipeline-plot semantics).
// ShardPlot = single-plot multi-GPU (--shard-plot semantics).
enum class BatchStrategy {
    Auto,
    WorkQueue,
    PipelinePlot,
    ShardPlot,
};

// Sentinels for BatchOptions::cpu_workers. Negative so a positive value is always
// an explicit exact count. See the cpu_workers doc above.
inline constexpr int kCpuWorkersAuto = -1;  // knee (~4), RAM-trimmed — the default
inline constexpr int kCpuWorkersMax  = -2;  // as many as fit in RAM, capped at cores

struct BatchOptions {
    bool verbose           = false;
    bool skip_existing     = false;
    bool continue_on_error = false;
    std::vector<int> device_ids;
    bool use_all_devices   = false;
    // Set by the CLI whenever --devices was given at all — even --devices cpu,
    // which leaves device_ids empty. It is the ONLY thing that tells an explicit
    // CPU-only selection apart from "no --devices at all": both reach
    // resolve_batch_devices with an empty device_ids, but only the latter should
    // have the default GPU materialised alongside it.
    bool devices_specified = false;

    // CPU selection, mirroring the GPU pair above: `cpu` / `all` set
    // use_all_cpu_nodes the way `gpu` sets use_all_devices, and `cpu0` / `cpu1`
    // fill cpu_node_ids the way `gpu0` / `0` fill device_ids. Selection only —
    // how many plots run on a selected node is cpu_workers, below. Both are
    // owned by --devices and reset by a later one, exactly like the GPU pair.
    std::vector<int> cpu_node_ids;
    bool use_all_cpu_nodes = false;
    // "The CPU is in this batch", asked for by something OTHER than a --devices
    // token: `--cpu`, or naming a count with `--cpu-workers N|auto|max`.
    //
    // Separate from the two fields above because it has a different owner, and
    // that is what makes the flags commute. --devices REPLACES the selection it
    // owns, so if this lived in use_all_cpu_nodes then `--cpu-workers 4
    // --devices gpu` would silently drop the CPU while `--devices gpu
    // --cpu-workers 4` kept it. Same flags, different order, different rig.
    bool cpu_opt_in = false;

    // Is the CPU in this batch at all? Ask the selection — cpu_workers is a
    // count and answers a different question. `--cpu-workers 0` is the one way
    // a count speaks to selection: it is how you say "none".
    bool cpu_selected() const {
        return (use_all_cpu_nodes || !cpu_node_ids.empty() || cpu_opt_in)
               && cpu_workers != 0;
    }
    int  cpu_workers       = kCpuWorkersAuto;  // per selected node; 0 = off
    std::string streaming_tier;

    // Per-GPU tier override populated by the --devices `<id>:<tier>`
    // suffix syntax. Keyed by CUDA device id; value is "plain" /
    // "compact" / "minimal" / "tiny" / "auto". A "auto" entry means
    // "explicitly opt back to auto-pick" — wins over both
    // `all_gpus_tier` and the global `streaming_tier`. See README
    // multi-GPU section for syntax examples.
    std::map<int, std::string> per_device_tier;

    // Tier set via the `gpu:<tier>` / `all:<tier>` shorthand on
    // --devices. Applies to every GPU that wasn't given an explicit
    // `<id>:<tier>` override above. Wins over the global
    // `streaming_tier` for those devices.
    std::string all_gpus_tier;
    // Phase 2.4: explicit strategy (Auto = picker decides). Legacy
    // shard_plot / pipeline_plot bool fields below are still honoured
    // for backward compatibility and act as explicit overrides if
    // strategy == Auto.
    BatchStrategy strategy = BatchStrategy::Auto;
    bool shard_plot        = false;
    std::string shard_strategy = "bucket";
    // Pipeline-parallel mode (Phase 2.1d): split each plot at the T2-
    // sort boundary across exactly two devices. device_ids[0] runs
    // Xs / T1 / T2; device_ids[1] runs T3 / Frag. Plots are pipelined
    // (depth=2) so plot N's stage 2 overlaps plot N+1's stage 1. On
    // PCIe-only hosts the two stages contend for host bandwidth and
    // throughput is below work-queue; the value is correctness on
    // heterogeneous rigs and per-plot latency on NVLink-equipped
    // hosts. Mutually exclusive with shard_plot.
    bool pipeline_plot     = false;
    // Per-stage streaming tier for the pipeline-plot path. Empty =
    // auto-pick per stage from each device's VRAM (Minimal when it
    // fits with headroom, else Tiny). When non-empty, size must match
    // device_ids.size() (currently 2 or 3). Tiny is the smallest
    // streaming tier (~3.2 GB at k=28); Minimal trades device VRAM
    // for fewer PCIe round-trips per stage (~3.76 GB but ~20%
    // faster). Ignored when pipeline_plot is false.
    std::vector<PipelineStageTier> pipeline_tiers;
    // Phase 2.2h: depth (in-flight plots per boundary) for the
    // pipelined batch. Default 2 = one plot overlapped per stage.
    // Higher values amortise pipeline fill/drain over more plots; at
    // k=28 each extra depth slot is ~6 GB pinned host per boundary.
    int pipeline_depth = 2;
    // When true and shard_plot is on, the distributed sorts route data
    // via direct device-to-device memcpy (Peer transport). On NVLink
    // hosts this stays on the fabric; on PCIe-only hosts the SYCL/CUDA
    // backend resolves D2D as an implicit single host bounce, which is
    // still ~one fewer copy than the explicit two-bounce HostBounce
    // path. Equivalent on a single-GPU dev box (peer-on-same-context =
    // ordinary device memcpy).
    //
    // Default flipped to true after k=28 measurements on 2× RTX 4000
    // Ada showed Peer at ~9.2 s/plot vs HostBounce at ~14.0 s/plot
    // (PCIe-only). Set to false (CLI: `--host-bounce`) on tight-VRAM
    // (<10 GB) cards at large k where the per-source staging cost
    // matters: Peer allocates source-side staging sized to the source
    // shard's full input count (~1.6 GB/shard for u32_u32 at k=28; up
    // to ~3.2 GB/shard for u32_u64+u32 in T2's sort).
    bool prefer_peer_copy  = true;

    // Aggregate progress: prints a one-liner after each plot completes
    // showing "N/M done (%, avg s/plot, TiB/s, fully-plotted ETA)";
    // rewritten in place when stderr is a TTY. Independent of verbose
    // (which is finer-grained per-phase noise). Fires in all three
    // batch strategies (work-queue, shard-plot, pipeline-plot). The
    // CLI defaults this to isatty(stderr); `--progress`/`--no-progress`
    // force it either way.
    bool progress          = false;

    // Quiet (CLI: -q/--quiet): suppress info-level stderr lines —
    // streaming-tier selection notes, multi-device worker banner.
    // Warnings, errors, and per-plot FAILED lines always print.
    // Does not imply --no-progress; the CLI wires that separately.
    bool quiet             = false;
};

// Parse a manifest file in the format described in tools/xchplot2/main.cpp
// (tab-separated, one plot per line). Throws std::runtime_error on bad input.
std::vector<BatchEntry> parse_manifest(std::string const& path);

// Run the staggered pipeline. Producer/consumer share a queue of depth 1.
// The first plot pays the full GPU+FSE cost; subsequent plots overlap.
BatchResult run_batch(std::vector<BatchEntry> const& entries,
                      BatchOptions const& opts);

// Legacy bool-verbose shim kept for source-compat with older callsites.
inline BatchResult run_batch(std::vector<BatchEntry> const& entries,
                             bool verbose = false)
{
    BatchOptions opts;
    opts.verbose = verbose;
    return run_batch(entries, opts);
}

// Phase 2.4 auto-strategy picker. Heuristic:
//   N == 1                              → WorkQueue
//   smallest device VRAM < tiny peak    → PipelinePlot (work-queue won't fit)
//   else                                → WorkQueue
// shard_plot is never auto-selected (niche; remains explicit opt-in).
//
// `reason_out`, if non-null, gets a short human-readable string for
// `[strategy] auto-picked X because Y` verbose printout.
//
// `vram_for_device` is the injectable VRAM lookup (test seam). The
// default-form overload uses sycl_backend's usable_gpu_devices.
struct StrategyPickInputs {
    std::vector<int> device_ids;
    int              k;
};
BatchStrategy select_strategy(
    StrategyPickInputs const&                inputs,
    std::function<std::uint64_t(int)> const& vram_for_device,
    std::string*                             reason_out = nullptr);

BatchStrategy select_strategy(
    StrategyPickInputs const& inputs,
    std::string*              reason_out = nullptr);

// Names for a worker list, one per entry, in the same order.
//
// device_label() on its own stopped being unique the moment --cpu-workers let
// one device back N workers: four of them all called "cpu" makes an interleaved
// log unreadable and a per-worker bench table meaningless. Repeats get a
// #ordinal suffix ("cpu#0", "cpu#1"); anything that appears once is left exactly
// as device_label() had it, so existing single-CPU logs do not churn.
//
// The bench uses this too — its per-worker block and the batch's log prefixes
// have to agree on what a worker is called, or they cannot be read together.
std::vector<std::string> worker_labels(std::vector<int> const& device_ids);

// Peak resident host memory of ONE CPU worker plotting at this k.
//
// pos2-chip's Plotter, not ours: no streaming tier can shrink it (the CPU
// branch of run_batch_slice returns long before the tier machinery), so this
// is a hard, fixed cost per concurrent CPU plot and the only defence against
// N of them is to not start them. Measured, not modelled from first
// principles — see the anchors in the .cpp.
std::uint64_t cpu_worker_peak_bytes(int k);

// Resolve BatchOptions' device selection to the concrete id list
// run_batch will use (use_all_devices → enumerate, device_ids →
// as-given, cpu_workers → append that many kCpuDeviceId entries;
// empty → default selector).
//
// NOT pure: it probes host RAM and caps cpu_workers at what actually fits
// (see cpu_worker_peak_bytes — 12.13 GiB each at k=28, and the OOM killer
// takes the GPU workers' in-flight plots down with it). It is deterministic
// within a process, though, because the free-RAM probe is taken once and
// cached: run_batch and the bench's own sizing call must agree on the worker
// count, and re-probing would let the CPU workers' own RSS change the answer
// underneath them. Callers that need the list cannot drift from run_batch.
//
// `gate_note`, if non-null, receives a human-readable line whenever the gate
// changed the count — so run_batch can print it exactly once rather than
// every caller printing it.
std::vector<int> resolve_batch_devices(BatchOptions const& opts,
                                       int                 k,
                                       std::string*        gate_note = nullptr);

// Filter a resolved device list down to the devices that can actually run a
// kernel, so one unusable GPU doesn't abort a run the others could finish.
// Each id is probed on its own thread (the SYCL queue is thread_local); ids
// that throw are omitted and, if `dropped` is non-null, appended to it as
// "<label>: <reason>". Negative ids — CPU nodes and the default-GPU sentinel —
// pass through unprobed. Probing costs a queue construction and its first-use
// JIT per device, so callers should only do it when a fallback exists.
std::vector<int> usable_batch_devices(std::vector<int> const&   device_ids,
                                      std::vector<std::string>* dropped = nullptr);

// Resolve the effective strategy the same way run_batch does
// (explicit opts.strategy > legacy shard/pipeline bools > heuristic).
BatchStrategy resolve_batch_strategy(BatchOptions const& opts,
                                     std::vector<int> const& device_ids,
                                     int k,
                                     std::string* reason_out = nullptr);

// Number of concurrently-plotting workers run_batch will spawn for
// these options (shard-plot and pipeline-plot act as one team = 1).
std::size_t batch_worker_count(BatchOptions const& opts, int k);

} // namespace pos2gpu
