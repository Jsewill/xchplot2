// xchplot2 — standalone Chia v2 plot creator on GPU. Three modes:
//   test  : low-level single-plot harness (caller supplies plot_id + memo).
//   batch : drive a TSV manifest of pre-computed plots through the GPU
//           pipeline with producer/consumer staggering.
//   plot  : full standalone — derives plot_id + memo from caller-supplied
//           BLS keys via the keygen-rs Rust shim, then dispatches through
//           batch internally. The "real" entrypoint for users.

#include "gpu/SyclDeviceList.hpp" // list_gpu_devices() — backs the
                                  // `devices` subcommand below. Plain
                                  // types only; the SYCL include lives
                                  // in SyclDeviceList.cpp (acpp-built).
#include "gpu/DeviceIds.hpp"   // device_label() — never print a raw device id
#include "host/GpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "host/NumaTopology.hpp"  // host_numa_nodes() — one `devices` row per CPU node
#include "host/Cancel.hpp"
#include "host/ConfigFile.hpp"
#include "host/PlotFileWriterParallel.hpp"
#include "pos2_keygen.h" // Rust shim for plot_id + memo derivation

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>  // isatty — progress defaults to on for interactive runs

namespace {

// Tri-state --progress resolution: explicit --progress/--no-progress
// wins; otherwise default to on when stderr is a TTY (where the line
// rewrites in place) and off when redirected or under -q/--quiet.
// -1 = auto, 0 = forced off, 1 = forced on.
bool resolve_progress(int tri, bool quiet)
{
    if (tri >= 0) return tri != 0;
    return !quiet && ::isatty(::fileno(stderr)) != 0;
}

void print_usage(char const* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " test <k> <plot_id_hex> [strength] [plot_index] [meta_group] [verbose]\n"
        << "         [-T|--testnet] [-o|--out DIR] [-m|--memo HEX] [-N|--out-name NAME]\n"
        << "         [--gpu-t1] [--gpu-t2] [--gpu-t3] [-G|--gpu-all] [-P|--profile]\n"
        << "  " << prog << " batch <manifest.tsv> [-v|--verbose] [-q|--quiet]\n"
        << "         [--skip-existing] [--continue-on-error]\n"
        << "         [--progress|--no-progress] [--devices <SPEC>]\n"
        << "    Manifest: one plot per non-empty/non-# line, whitespace-separated:\n"
        << "      k strength plot_index meta_group testnet plot_id_hex memo_hex out_dir out_name\n"
        << "    Runs GPU compute and CPU FSE in a producer/consumer pipeline so they overlap\n"
        << "    across consecutive plots. ~2x throughput vs separate `test` invocations.\n"
        << "  " << prog << " bench [-k K] [-s S] [-n N] [-o DIR] [--devices SPEC]\n"
        << "         [--tier T] [--cpu] [--warmup W] [--keep] [-T|--testnet]\n"
        << "         [--target-size TiB] [--compute-only] [-q|--quiet]\n"
        << "    Measure plotting throughput (TiB/hour, TiB/day, TiB/month) on\n"
        << "    synthetic unfarmable plots (default: 1 warmup + 10 measured\n"
        << "    plots/worker). Writes real .plot2 files unless --keep is set;\n"
        << "    deletes them on exit by default.\n"
        << "  " << prog << " plot -k K -n N -f HEX  ( -p HEX | --pool-ph HEX | -c xch1... )\n"
        << "         [-s S] [-o DIR] [-T] [-i N] [-g N] [-S HEX] [-v] [-q]\n"
        << "         [--skip-existing] [--continue-on-error]\n"
        << "    Standalone farmable plot(s): derives plot_id + memo internally\n"
        << "    from the keys via chia-rs, then batches through the GPU pipeline.\n"
        << "    -f, --farmer-pk HEX             : 96 hex chars (48 B G1 public key).\n"
        << "    -p, --pool-pk HEX               : 96 hex chars. Pool public key mode.\n"
        << "        --pool-ph HEX               : 64 hex chars (raw puzzle hash).\n"
        << "    -c, --pool-contract-address ADR : Chia bech32m address (xch1.../txch1...);\n"
        << "                                      decoded internally to a 32-byte hash.\n"
        << "    -k, --k K                       : k size (default 28).\n"
        << "    -n, --num N                     : number of plots to create.\n"
        << "    -s, --strength S                : v2 PoS strength (default 2).\n"
        << "    -o, --out DIR                   : output directory.\n"
        << "    -i, --plot-index N              : base v2 PoS plot_index (default 0); increments per plot.\n"
        << "    -g, --meta-group N              : v2 PoS meta_group field (default 0).\n"
        << "    -S, --seed HEX                  : optional 64 hex chars of master-SK\n"
        << "                                      entropy. Per-plot seed = SHA256(seed || i).\n"
        << "                                      Reproducible across runs. Defaults to\n"
        << "                                      fresh /dev/urandom per plot.\n"
        << "    -T, --testnet                   : testnet proof parameters.\n"
        << "    -v, --verbose                   : per-plot progress on stderr.\n"
        << "    -q, --quiet                     : suppress info-level stderr output\n"
        << "                                      (progress line, summaries, tier\n"
        << "                                      notes). Warnings/errors and the\n"
        << "                                      stdout path listing still print.\n"
        << "    --progress / --no-progress      : force the aggregate progress line\n"
        << "                                      on/off. Default: on when stderr is\n"
        << "                                      a terminal (rewrites in place),\n"
        << "                                      off when redirected or with -q.\n"
    << "    --skip-existing                 : skip plots whose output file is already a\n"
        << "                                      complete .plot2 (magic + non-trivial size).\n"
        << "    --continue-on-error             : log per-plot failures and keep going\n"
        << "                                      instead of aborting the batch.\n"
        << "    --devices SPEC                  : multi-device. SPEC is a comma\n"
        << "                                      list mixing any of:\n"
        << "                                        all       — every GPU + every CPU node\n"
        << "                                        gpu       — every visible GPU\n"
        << "                                        cpu       — every CPU node (slow)\n"
        << "                                        gpu0,gpu2 — those GPUs\n"
        << "                                        cpu1      — that CPU (NUMA) node only\n"
        << "                                        0,1,3     — explicit GPU ids (== gpu0,..)\n"
        << "                                      e.g. gpu,cpu == all. This flag names\n"
        << "                                      DEVICES; --cpu-workers says how many\n"
        << "                                      plots run on each, so `cpu,cpu` is just\n"
        << "                                      `cpu` — repeats dedup, as `0,0` does.\n"
        << "                                      Any GPU selector accepts a `:tier`\n"
        << "                                      suffix to pin the streaming tier for\n"
        << "                                      that device(s). Tier ∈ plain | compact\n"
        << "                                      | minimal | tiny | pinned | auto.\n"
        << "                                      Per-GPU override wins over gpu:<tier>\n"
        << "                                      wins over global --tier. Examples:\n"
        << "                                        gpu,2:tiny      all GPUs auto, 2 = tiny\n"
        << "                                        all,2:tiny      all GPUs + CPU; 2 = tiny\n"
        << "                                        gpu:tiny,2:plain    all tiny, 2 = plain\n"
        << "                                        gpu:tiny,2:auto     all tiny, 2 auto\n"
        << "                                      `cpu:<tier>` is rejected (no tier on\n"
        << "                                      CPU worker).\n"
        << "                                      Omitted = single device via default\n"
        << "                                      SYCL selector (zero-config).\n"
        << "    CPU plotting is OPT-IN: ask for it with `--devices cpu` / `--devices all`\n"
        << "    (or --cpu, which adds it to whatever GPUs are already selected). CPU\n"
        << "    plots run alongside the GPUs, niced below them, and on a multi-socket\n"
        << "    host each worker is pinned to its own NUMA node.\n"
        << "    --cpu                           : add CPU workers to the current selection\n"
        << "                                      (every node), without naming --devices.\n"
        << "    --cpu-workers auto|max|N|off    : how many CPU plots run concurrently\n"
        << "                                      ON EACH selected node. Naming any count\n"
        << "                                      but 0 also opts the CPU in.\n"
        << "                                        auto (default) — the throughput knee\n"
        << "                                            (~4), then trimmed to host RAM.\n"
        << "                                        max            — as many as fit in RAM,\n"
        << "                                                         capped at core count.\n"
        << "                                        N              — exactly N per node.\n"
        << "                                        off            — none, overriding\n"
        << "                                                         --devices cpu.\n"
        << "                                      The CPU plotter is memory-latency-bound,\n"
        << "                                      so concurrent plots interleave each\n"
        << "                                      other's stalls — but each already uses\n"
        << "                                      every core, so it plateaus by ~4 (k=28:\n"
        << "                                      N=2 +19%, N=4 +25%, flat after). Each\n"
        << "                                      needs its own working set (12.1 GiB at\n"
        << "                                      k=28), so the count is RAM-trimmed and\n"
        << "                                      it tells you what it picked.\n"
        << "    --shard-plot                    : EXPERIMENTAL — opt in to single-plot\n"
        << "                                      multi-GPU. Each plot is processed by\n"
        << "                                      ALL --devices cooperatively (one plot\n"
        << "                                      at a time, sharded across GPUs)\n"
        << "                                      instead of the default work-queue\n"
        << "                                      (one plot per GPU). Drop the flag for\n"
        << "                                      the default work-queue behaviour.\n"
        << "    --host-bounce                   : with --shard-plot, force the distributed\n"
        << "                                      sort traffic through host-pinned bounce\n"
        << "                                      instead of the default Peer transport.\n"
        << "                                      Use on tight-VRAM (<10 GB) cards at\n"
        << "                                      large k where the Peer path's per-\n"
        << "                                      source staging (~1.6 GB/shard for u32\n"
        << "                                      pairs at k=28; up to ~3.2 GB/shard\n"
        << "                                      for T2's u32_u64+u32) doesn't fit.\n"
        << "                                      The Peer default is faster on every\n"
        << "                                      tested topology — NVLink hosts route\n"
        << "                                      direct, PCIe-only hosts get implicit\n"
        << "                                      single-bounce via SYCL/CUDA D2D, still\n"
        << "                                      one copy fewer than HostBounce.\n"
        << "    --prefer-peer-copy              : deprecated alias; Peer is the default\n"
        << "                                      now, this flag is a no-op kept for\n"
        << "                                      backward-compat with existing scripts.\n"
        << "    --tier plain|compact|minimal|tiny|auto : force streaming pipeline tier\n"
        << "                                      when GPU pool doesn't fit. plain =\n"
        << "                                      ~7.24 GB floor (k=28), faster.\n"
        << "                                      compact = ~5.33 GB floor, fits on\n"
        << "                                      tight 8 GB cards. minimal = ~3.83 GB\n"
        << "                                      floor, fits on 4 GiB cards (extra\n"
        << "                                      PCIe round-trips during T2 match).\n"
        << "                                      tiny = ~3.2 GB floor at k=28, fits\n"
        << "                                      4 GB cards comfortably. Parks every\n"
        << "                                      cap-sized intermediate (T1 meta /\n"
        << "                                      T1 keys / T2 meta / T2 xbits / T2\n"
        << "                                      keys / d_t3) on host pinned and\n"
        << "                                      reads section-sized slices into\n"
        << "                                      device for each match/sort pass.\n"
        << "                                      Slower than minimal due to extra\n"
        << "                                      cap-sized PCIe round-trips.\n"
        << "                                      auto (default) = pick the largest\n"
        << "                                      tier that fits. Equivalent to\n"
        << "                                      XCHPLOT2_STREAMING_TIER env var;\n"
        << "                                      CLI flag wins if both set.\n"
        << "  " << prog << " verify <plotfile> [--trials N]\n"
        << "    Open <plotfile> and run N random challenges through the CPU prover.\n"
        << "    Zero proofs across a sensible sample (>=100) strongly indicates a\n"
        << "    corrupt plot. Default N=100.\n"
        << "  " << prog << " parity-check [--dir PATH]\n"
        << "    Run every *_parity binary in PATH and summarize PASS/FAIL.\n"
        << "    Default PATH is ./build/tools/parity. Build the tests with\n"
        << "    `cmake --build <build-dir>` first. Useful for post-refactor\n"
        << "    regression screening.\n"
        << "  " << prog << " devices\n"
        << "    List every visible SYCL GPU device + the host CPU plotter\n"
        << "    with id, name, backend, capacity, and which sort path the\n"
        << "    runtime dispatcher will route a worker to (CUB on cuda-\n"
        << "    backend devices when this build links CUB, otherwise SortSycl).\n"
        << "    Use the printed [N] / [cpu] index with --devices in plot/batch.\n"
        << "\n"
        << "  test-mode positional args:\n"
        << "    <k>            : even integer in [18, 32]\n"
        << "    <plot_id_hex>  : 64 hex characters\n"
        << "    [strength]     : optional, defaults to 2\n"
        << "    [plot_index]   : optional, defaults to 0\n"
        << "    [meta_group]   : optional, defaults to 0\n"
        << "    [verbose]      : optional, 0/1, default 0\n"
        << "  test-mode flags:\n"
        << "    -T, --testnet      : use testnet proof parameters\n"
        << "    -o, --out DIR      : output directory, defaults to .\n"
        << "    -m, --memo HEX     : memo bytes (hex); required for farmable plots\n"
        << "    -N, --out-name NAME: override output filename (basename only)\n"
        << "        --gpu-tN       : run phase N on GPU (T1/T2/T3); default CPU\n"
        << "    -G, --gpu-all      : run all phases on GPU (where implemented)\n"
        << "    -P, --profile      : print per-phase device-time breakdown\n"
        << "\n"
        << "  Environment variables:\n"
        << "    XCHPLOT2_STREAMING=1          force the low-VRAM streaming pipeline even\n"
        << "                                  when the persistent pool would fit.\n"
        << "    POS2GPU_MAX_VRAM_MB=N         cap the pool/streaming VRAM query to N MB\n"
        << "                                  (useful for testing the streaming fallback).\n"
        << "    POS2GPU_STREAMING_STATS=1     log every streaming-path alloc / free.\n"
        << "    POS2GPU_POOL_DEBUG=1          log pool allocation sizes at construction.\n"
        << "    POS2GPU_PHASE_TIMING=1        per-phase wall-time breakdown on stderr.\n"
        << "    ACPP_GFX=gfxXXXX              AMD only — required at build time to AOT\n"
        << "                                  for the right amdgcn ISA (see README).\n";
}

bool parse_hex_bytes(std::string const& s, std::vector<uint8_t>& out)
{
    if (s.size() % 2 != 0) return false;
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    out.clear();
    out.reserve(s.size() / 2);
    for (size_t i = 0; i < s.size(); i += 2) {
        int hi = val(s[i]);
        int lo = val(s[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return true;
}

bool parse_hex(std::string const& s, std::array<uint8_t, 32>& out)
{
    if (s.size() != 64) return false;
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < 32; ++i) {
        int hi = val(s[2*i]);
        int lo = val(s[2*i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

// Read exactly `n` bytes of entropy from /dev/urandom. Throws on failure.
void read_urandom(uint8_t* out, size_t n)
{
    std::ifstream f("/dev/urandom", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open /dev/urandom");
    f.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(n));
    if (f.gcount() != static_cast<std::streamsize>(n)) {
        throw std::runtime_error("short read from /dev/urandom");
    }
}

// Parse a --devices value into BatchOptions.
//
// Accepted forms:
//   "all"              → use every GPU visible at runtime (sets
//                        use_all_devices; device_ids stays empty).
//   "0"                → use only GPU id 0.
//   "0,2,3"            → use these specific device ids, in sorted order.
//
// Zero-configuration default (no flag) produces device_ids.empty() and
// use_all_devices=false — which triggers the single-device
// gpu_selector_v path, identical to pre-multi-GPU behavior.
//
// --cpu-workers N: how many CPU plots run concurrently ON EACH SELECTED CPU NODE.
//
// Per node, not per host, so it means the same thing on a 1-socket box as on a
// 4-socket one — and on the 1-socket box (every rig this was tuned on) per-node
// and per-host are the same number, so nothing about it has changed.
//
// Accepts a count or a keyword:
//   auto        — the throughput knee (~4), RAM-trimmed. The default WHEN the
//                 CPU is selected; the CPU itself is opt-in (`--devices cpu`,
//                 `--devices all`, or `--cpu`).
//   max         — as many as fit in RAM, capped at the core count.
//   off / none / 0 — no CPU workers, whatever --devices asked for.
//   N           — exactly N (still RAM-trimmed).
//
// Naming any count but zero is itself a request for the CPU: `--cpu-workers 4`
// on its own means the CPU is in, on every node. Zero means it is out — that is
// the one way a count speaks to selection, and it is what `--no-cpu` used to be.
//
// N is an ASK, not a promise: run_batch caps it at what host RAM actually holds
// (12.1 GiB per worker at k=28) and says so when it does.
//
// Returns false on malformed input (caller prints usage + exits 1).
bool parse_cpu_workers_arg(std::string const& s, pos2gpu::BatchOptions& opts)
{
    // Selecting the CPU here, rather than in the --devices parser, is what lets
    // the two flags commute — see BatchOptions::cpu_opt_in.
    auto opt_in = [&](int n) { opts.cpu_workers = n; opts.cpu_opt_in = true; return true; };

    if (s == "auto")               return opt_in(pos2gpu::kCpuWorkersAuto);
    if (s == "max")                return opt_in(pos2gpu::kCpuWorkersMax);
    if (s == "off" || s == "none") { opts.cpu_workers = 0; return true; }

    char* endp = nullptr;
    long const v = std::strtol(s.c_str(), &endp, 10);
    if (endp == s.c_str() || *endp != '\0' || v < 0 || v > 64) {
        std::cerr << "Error: --cpu-workers expects an integer in [0, 64], or "
                     "'auto' / 'max' / 'off' (got '" << s << "')\n";
        return false;
    }
    // Assign, don't max: this is the explicit, specific flag. `--cpu-workers 0`
    // means "no, actually, none" and must not resolve to the auto default — nor
    // opt the CPU in, which is why it does not go through opt_in().
    if (v == 0) { opts.cpu_workers = 0; return true; }
    return opt_in(static_cast<int>(v));
}

// Returns false on malformed input (caller prints usage + exits 1).
bool parse_devices_arg(std::string const& s, pos2gpu::BatchOptions& opts)
{
    // Accept comma-separated mix of:
    //   "all"             → all GPUs + CPU worker
    //   "gpu"             → all GPUs
    //   "cpu"             → CPU worker (CPU-only; suppresses the implicit
    //                       default GPU that a bare run would otherwise add)
    //   "<int>"           → explicit GPU id
    //   "gpu:<tier>"      → all GPUs default to <tier>
    //   "all:<tier>"      → all GPUs default to <tier>, plus CPU worker
    //   "<int>:<tier>"    → explicit GPU id + per-GPU tier override
    //
    // <tier> is one of plain/compact/minimal/tiny/auto. The "auto"
    // sentinel explicitly opts back into auto-pick — useful for
    // overriding `gpu:<tier>` or global `--tier` on one specific GPU.
    //
    // Compose freely: `gpu,2:tiny` = all GPUs auto-pick except GPU 2
    // which uses tiny. `gpu:tiny,2:plain` = all GPUs tiny except GPU 2
    // which uses plain. `cpu:<tier>` is rejected — CPU worker has no
    // streaming tier.
    auto is_valid_tier = [](std::string const& t) {
        return t == "plain" || t == "compact" || t == "minimal" ||
               t == "tiny"  || t == "pinned"  || t == "auto";
    };
    auto bad = [&](char const* why) {
        std::cerr << "Error: --devices: " << why
                  << " (token list: \"" << s << "\")\n";
        return false;
    };

    // A --devices list REPLACES any earlier one, so every selection it can
    // express resets here — CPU included. Leaving the CPU selection standing
    // would make `--devices all --devices gpu` keep a CPU nobody asked for.
    // cpu_workers is deliberately NOT reset: it is a count, not a selection,
    // and an explicit --cpu-workers must survive a later --devices.
    opts.device_ids.clear();
    opts.use_all_devices = false;
    opts.per_device_tier.clear();
    opts.all_gpus_tier.clear();
    opts.cpu_node_ids.clear();
    opts.use_all_cpu_nodes = false;
    bool any_token = false;
    bool any_gpu_token = false;
    size_t start = 0;
    while (start <= s.size()) {
        size_t const end = s.find(',', start);
        std::string const tok = s.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (tok.empty()) return false;
        any_token = true;

        std::string selector = tok;
        std::string tier_suffix;
        if (size_t const colon = tok.find(':'); colon != std::string::npos) {
            selector    = tok.substr(0, colon);
            tier_suffix = tok.substr(colon + 1);
            if (tier_suffix.empty())  return bad("empty tier after `:`");
            if (!is_valid_tier(tier_suffix)) {
                return bad("invalid tier (expect plain|compact|minimal|tiny|auto)");
            }
        }

        // "gpu7" / "cpu1" — a kind plus an index. Bare "7" is the historical
        // spelling of "gpu7" and still parses, below.
        auto indexed = [](std::string const& tok, char const* kind) -> std::optional<int> {
            std::string const prefix(kind);
            if (tok.size() <= prefix.size() || tok.compare(0, prefix.size(), prefix) != 0) {
                return std::nullopt;
            }
            std::string const digits = tok.substr(prefix.size());
            char* endp = nullptr;
            long const v = std::strtol(digits.c_str(), &endp, 10);
            if (endp == digits.c_str() || *endp != '\0' || v < 0 || v > 1023) {
                return std::nullopt;
            }
            return static_cast<int>(v);
        };

        auto add_gpu = [&](int id) -> bool {
            opts.device_ids.push_back(id);
            any_gpu_token = true;
            if (!tier_suffix.empty()) {
                auto const it = opts.per_device_tier.find(id);
                if (it != opts.per_device_tier.end() && it->second != tier_suffix) {
                    return bad("same GPU id given two different :tier overrides");
                }
                opts.per_device_tier[id] = tier_suffix;
            }
            return true;
        };

        if (selector == "all") {
            opts.use_all_devices  = true;
            opts.use_all_cpu_nodes = true;  // "all" means every GPU *and* every CPU node
            any_gpu_token = true;
            if (!tier_suffix.empty()) opts.all_gpus_tier = tier_suffix;
        } else if (selector == "gpu") {
            opts.use_all_devices = true;
            any_gpu_token = true;
            if (!tier_suffix.empty()) opts.all_gpus_tier = tier_suffix;
        } else if (selector == "cpu") {
            if (!tier_suffix.empty()) return bad("cpu token cannot carry a tier");
            // Every CPU node, exactly as `gpu` means every GPU. It is a
            // SELECTION, not a count: --cpu-workers says how many plots run per
            // node. So repeats dedup — `cpu,cpu` names the same hardware twice
            // and cannot mean more of it, any more than `0,0` means two cards.
            opts.use_all_cpu_nodes = true;
        } else if (auto const node = indexed(selector, "cpu")) {
            if (!tier_suffix.empty()) return bad("cpu token cannot carry a tier");
            opts.cpu_node_ids.push_back(*node);
        } else if (auto const id = indexed(selector, "gpu")) {
            if (!add_gpu(*id)) return false;
        } else {
            char* endp = nullptr;
            long const v = std::strtol(selector.c_str(), &endp, 10);
            if (endp == selector.c_str() || *endp != '\0' || v < 0 || v > 1023) {
                return bad("unrecognised device token "
                           "(expect all|gpu|cpu|gpu<n>|cpu<n>|<id>)");
            }
            if (!add_gpu(static_cast<int>(v))) return false;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (!any_token) return false;
    bool const any_cpu_token = opts.use_all_cpu_nodes || !opts.cpu_node_ids.empty();
    if (!any_gpu_token && !any_cpu_token) return false;

    // `cpu` covers every node, so listing nodes as well is redundant.
    if (opts.use_all_cpu_nodes) opts.cpu_node_ids.clear();

    auto dedup = [](std::vector<int>& v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    dedup(opts.device_ids);
    dedup(opts.cpu_node_ids);
    // Any successful --devices — even a CPU-only one that left device_ids empty —
    // is an explicit selection. resolve_batch_devices keys off this to avoid
    // materialising a default GPU on top of an explicit CPU-only request.
    opts.devices_specified = true;
    return true;
}

std::string plot_id_to_filename(int k, std::array<uint8_t, 32> const& plot_id)
{
    // Match chia plots create's v2 filename scheme: plot-k{size}-{id}.plot2
    static char const hex[] = "0123456789abcdef";
    std::string out = "plot-k" + std::to_string(k) + "-";
    out.reserve(out.size() + 64 + 6);
    for (uint8_t b : plot_id) {
        out += hex[b >> 4];
        out += hex[b & 0xF];
    }
    out += ".plot2";
    return out;
}

constexpr double kGiBBytes = 1024.0 * 1024.0 * 1024.0;
constexpr double kTibBytes = kGiBBytes * 1024.0;

std::string bytes_to_hex(std::array<uint8_t, 32> const& id)
{
    static char const hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t b : id) {
        out += hex[b >> 4];
        out += hex[b & 0xF];
    }
    return out;
}

std::string format_duration_dh(double seconds)
{
    if (seconds < 0.0) seconds = 0.0;
    // long long, not int: a slow rig with a big --target-size overflows a 32-bit
    // seconds count (k=18 at ~5.6e-7 TiB/s and --target-size 2000 is ~3.5e9 s),
    // and the signed overflow printed "-24855d -20h".
    long long const total_s = static_cast<long long>(
        std::min(seconds, 9.0e15));
    long long const d = total_s / 86400;
    int const h = static_cast<int>((total_s % 86400) / 3600);
    int const m = static_cast<int>((total_s % 3600) / 60);
    int const s = static_cast<int>(total_s % 60);
    char buf[48];
    if (d > 0) {
        std::snprintf(buf, sizeof(buf), "%lldd %dh", d, h);
    } else if (h > 0) {
        // Carry the minutes. Truncating to whole hours threw away up to 59 of
        // them against a base of one, so a 1 h 53 m disk fill printed as "~1h" —
        // off by 47%, and wrong in the flattering direction on exactly the
        // single-GPU rigs where the fill is 1-2 h.
        std::snprintf(buf, sizeof(buf), "%dh%02dm", h, m);
    } else if (m > 0) {
        std::snprintf(buf, sizeof(buf), "%dm%02ds", m, s);
    } else {
        std::snprintf(buf, sizeof(buf), "%ds", s);
    }
    return buf;
}

// Render one worker's share of a sized batch as "gpu0 43, cpu#1 6".
std::string format_split(std::vector<pos2gpu::WorkerSplit> const& split,
                         std::vector<std::string> const& names)
{
    std::string out;
    for (auto const& w : split) {
        if (!out.empty()) out += ", ";
        // worker_index, never the position in `split` — see WorkerSplit.
        out += (w.worker_index < names.size() ? names[w.worker_index]
                                              : pos2gpu::device_label(w.device_id));
        out += " " + std::to_string(w.plots);
    }
    return out;
}

// The batch sizes that leave nobody waiting on a peer. Shared by bench and the
// plot/batch run summary so the two give the same advice in the same words.
//
// `device_ids` is the FULL worker list, including any the model dropped for
// want of a rate: worker_labels() numbers repeats by position in it.
void print_batch_sizing(char const* prefix,
                        pos2gpu::OptimalBatch const& opt,
                        std::vector<int> const& device_ids)
{
    if (!opt.valid) return;

    auto const names = pos2gpu::worker_labels(device_ids);
    auto const& w = opt.workable;

    // Matched workers land together at one plot apiece — there is no trade-off
    // to explain and no second number to offer.
    if (opt.coincide) {
        std::fprintf(stderr,
            "%s   optimal batch: multiples of %zu — %s — every worker lands "
            "together\n",
            prefix, w.plots, format_split(w.split, names).c_str());
    } else {
        std::fprintf(stderr,
            "%s   optimal batch: multiples of %zu — %s — every worker lands "
            "within %.2f s (%.1f%% idle)\n",
            prefix, w.plots, format_split(w.split, names).c_str(),
            w.spread_seconds, 100.0 * w.idle_fraction);
        if (opt.exact.valid) {
            // Worth a second line because the workable size's error is a
            // FRACTION: double the batch and you double the seconds wasted. The
            // exact size is the one that stays tight however many you run.
            //
            // Quote the percentage, not the seconds. The two sizes are different
            // batches, so their absolute waits are not comparable — a 2-plot
            // batch idling 0.01 s and an 83-plot batch idling 0.01 s look
            // identical spelled that way, and the second is 40x the better deal.
            // The fraction is the part that holds still.
            std::fprintf(stderr,
                "%s     for a near-exact landing use multiples of %zu — %s — "
                "%.2f%% idle\n",
                prefix, opt.exact.plots,
                format_split(opt.exact.split, names).c_str(),
                100.0 * opt.exact.idle_fraction);
        }
    }

    if (opt.workers_unrated > 0) {
        std::fprintf(stderr,
            "%s     (%zu worker%s had no rate yet and %s left out of this — run "
            "more plots to size around %s)\n",
            prefix, opt.workers_unrated, opt.workers_unrated == 1 ? "" : "s",
            opt.workers_unrated == 1 ? "was" : "were",
            opt.workers_unrated == 1 ? "it" : "them");
    }
}

// Why a worker has no rate, in the operator's terms. Read off WorkerStats::why
// rather than assumed: both printers used to hardcode "too few plots", which is
// the wrong reason — and the wrong advice — for a worker excluded by the window
// rather than by its plot count.
char const* unmeasured_reason(pos2gpu::Unmeasured why)
{
    switch (why) {
        case pos2gpu::Unmeasured::NoPlots:
            return "no plots finished";
        case pos2gpu::Unmeasured::AllWarmup:
            return "none survived the warmup drop";
        case pos2gpu::Unmeasured::AllPastWindow:
            return "every plot landed after a peer ran dry";
        case pos2gpu::Unmeasured::Degenerate:
            return "measurement window collapsed";
        case pos2gpu::Unmeasured::No:
            break;
    }
    return "no rate";
}

// Per-worker rates for a plot/batch run, plus the sizing advice they imply.
//
// The live progress block shows the same rates while the run is going, but it
// only exists on a TTY — a run redirected to a log never saw it, and that is
// exactly the run someone reads afterwards to decide how to size the next one.
//
// warmup=0 + WholeRun: unlike bench, this reports the run that actually
// happened rather than a steady-state estimate, so no plot is excluded.
// compute_bench_stats still anchors on work_start_seconds, keeping device init
// and pool construction — which are not per-plot costs — out of the rates.
//
// WholeRun is load-bearing, not a nicety. Under the bench's FullQueue window a
// batch of 4 plots across 4 workers reported three of them as unmeasurable and
// printed no sizing at all: every worker finished exactly one plot, the window
// shut at the earliest of those four completions, and the other three landed
// 0.3 s past it. See RateWindow in BenchStats.hpp.
void print_run_workers(char const* prefix, pos2gpu::BatchResult const& res)
{
    if (res.workers.size() < 2) return;  // nothing to compare or balance

    auto const stats = pos2gpu::compute_bench_stats(
        res.workers, 0, pos2gpu::RateWindow::WholeRun);
    if (!stats.valid) return;

    std::vector<int> ids;
    ids.reserve(res.workers.size());
    for (auto const& w : res.workers) ids.push_back(w.device_id);
    auto const names = pos2gpu::worker_labels(ids);

    std::fprintf(stderr, "%s per-worker:\n", prefix);
    std::vector<double> rates(stats.workers.size(), 0.0);
    for (std::size_t i = 0; i < stats.workers.size(); ++i) {
        auto const& w = stats.workers[i];
        char const* name = i < names.size() ? names[i].c_str() : "?";
        if (!w.measured) {
            // WholeRun clips nothing, so in practice this is a worker still on
            // its first plot when the batch ended.
            std::fprintf(stderr, "%s   %-5s %s\n",
                         prefix, name, unmeasured_reason(w.why));
            continue;
        }
        rates[i] = w.s_per_plot;
        char tail[80] = {0};
        if (w.idle_tail_seconds >= 0.05) {
            std::snprintf(tail, sizeof(tail), " — then idle %.1f s waiting on a peer",
                          w.idle_tail_seconds);
        }
        std::fprintf(stderr, "%s   %-5s %zu plot%s — %.2f s/plot%s\n",
                     prefix, name, w.plots_total, w.plots_total == 1 ? "" : "s",
                     w.s_per_plot, tail);
    }

    print_batch_sizing(prefix, pos2gpu::pick_batch_sizing(rates, ids), ids);
}

void print_run_summary(char const* prefix, pos2gpu::BatchResult const& res)
{
    double const per = res.plots_written
        ? res.total_wall_seconds / double(res.plots_written) : 0.0;
    std::cerr << prefix << " wrote " << res.plots_written << " plots in "
              << res.total_wall_seconds << " s (" << per << " s/plot)";
    if (res.plots_skipped) std::cerr << "; skipped " << res.plots_skipped;
    if (res.plots_failed)  std::cerr << "; failed "  << res.plots_failed;
    if (res.bytes_written > 0 && res.total_wall_seconds > 0.0) {
        double const gib = static_cast<double>(res.bytes_written) / kGiBBytes;
        double const tib_per_hour =
            (static_cast<double>(res.bytes_written) / kTibBytes) /
            (res.total_wall_seconds / 3600.0);
        std::cerr << "; " << gib << " GiB written ("
                  << tib_per_hour << " TiB/hour effective)";
    }
    std::cerr << "\n";
    // After the aggregate, never beside it: the s/plot above is the batch's,
    // and a per-worker name next to a batch number is the confusion the
    // progress prefix already had to be fixed for (see BatchPlotter's
    // progress_prefix).
    std::cerr.flush();
    print_run_workers(prefix, res);
}

// Pick the roomiest tmpfs, not the first one that happens to exist. The old
// order took $XDG_RUNTIME_DIR ahead of /dev/shm — but systemd sizes /run/user at
// 10% of RAM and caps it, while /dev/shm gets 50%, so it reached for the smaller
// of the two for a pass that writes the entire plot set at once and deletes
// nothing until the end.
std::string resolve_tmpfs_dir()
{
    std::string    best;
    std::uintmax_t best_avail = 0;
    auto consider = [&](std::string const& dir) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) return;
        auto const sp = std::filesystem::space(dir, ec);
        if (ec) return;
        if (sp.available > best_avail) {
            best_avail = sp.available;
            best = dir;
        }
    };
    if (char const* x = std::getenv("XDG_RUNTIME_DIR"); x && x[0]) consider(x);
    consider("/dev/shm");
    return best;
}

// A scratch dir on the roomiest tmpfs that we have actually proven we can write
// to, or "" to say there isn't one.
//
// The old code appended a FIXED "/xchplot2-bench" and called create_directories.
// On /dev/shm that is a shared, world-visible path: whoever creates it first owns
// it, and create_directories then quietly SUCCEEDS for everyone else (the
// directory exists — that is all it checks) until the first plot open() dies with
// EACCES. On this dev box a root-owned /dev/shm/xchplot2-bench from July 8 made
// every --compute-only run fail that way, and it failed only after the e2e pass
// had already been paid for. Namespace the dir per user + process, and prove the
// write before handing it back.
std::string prepare_tmpfs_scratch()
{
    std::string const base = resolve_tmpfs_dir();
    if (base.empty()) return {};

    std::string const dir = base + "/xchplot2-bench-"
        + std::to_string(static_cast<unsigned>(::getuid())) + "-"
        + std::to_string(static_cast<unsigned>(::getpid()));

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return {};

    auto const probe = std::filesystem::path(dir) / ".probe";
    {
        std::ofstream f(probe, std::ios::binary);
        if (!f) {
            std::filesystem::remove_all(dir, ec);
            return {};
        }
    }
    std::filesystem::remove(probe, ec);
    return dir;
}

struct BenchMeasurement {
    pos2gpu::BatchResult result;
    std::vector<std::filesystem::path> paths;
    std::vector<std::uint64_t> plot_sizes;
    pos2gpu::BenchStats stats;   // per-worker + aggregate; see BenchStats.hpp
    double rate_tib_s = 0.0;
    double s_per_plot = 0.0;
    double gib_per_plot = 0.0;
    double size_gib_min = 0.0;
    double size_gib_max = 0.0;
};

// warmup_per_worker, NOT a global count. This used to take `warmup * workers`
// and drop that many plots off the front of the merged timeline, which is only
// the same thing when every worker runs at the same speed. On a GPU+CPU rig it
// dropped the GPU's first two plots and kept the CPU's cold-start one.
BenchMeasurement analyze_bench_run(
    pos2gpu::BatchResult const& res,
    std::vector<std::filesystem::path> const& paths,
    std::size_t warmup_per_worker)
{
    BenchMeasurement out;
    out.result = res;
    out.paths = paths;
    out.plot_sizes.reserve(paths.size());
    for (auto const& p : paths) {
        out.plot_sizes.push_back(
            static_cast<std::uint64_t>(std::filesystem::file_size(p)));
    }

    if (out.plot_sizes.size() != res.plots_written) {
        throw std::runtime_error(
            "bench internal error: path count != plots_written");
    }

    std::uint64_t const tally_bytes = [&] {
        std::uint64_t sum = 0;
        for (auto s : out.plot_sizes) sum += s;
        return sum;
    }();
    if (tally_bytes != res.bytes_written) {
        std::fprintf(stderr,
            "[bench] WARNING: writer byte tally (%llu) != file_size sum "
            "(%llu) — unexpected write path\n",
            static_cast<unsigned long long>(res.bytes_written),
            static_cast<unsigned long long>(tally_bytes));
    }

    // Throughput comes from the per-worker timelines, never from the merged
    // completion list: a work-queue gives no worker a predictable share, so
    // once the lists are merged there is no way to tell a worker's cold plot
    // from a peer's steady one, nor to see the drain tail where a worker that
    // has emptied the queue sits idle while a slower peer finishes. Both of
    // those land inside the measured window if you average the merged gaps.
    out.stats = pos2gpu::compute_bench_stats(res.workers, warmup_per_worker);
    if (!out.stats.valid) {
        throw std::runtime_error(
            "bench: no worker finished a measurable plot after its own warmup "
            "exclusion (increase -n or reduce --warmup)");
    }
    out.s_per_plot = out.stats.s_per_plot;

    out.size_gib_min = static_cast<double>(out.plot_sizes.front()) / kGiBBytes;
    out.size_gib_max = out.size_gib_min;
    std::uint64_t size_sum = 0;
    for (auto s : out.plot_sizes) {
        double const gib = static_cast<double>(s) / kGiBBytes;
        out.size_gib_min = std::min(out.size_gib_min, gib);
        out.size_gib_max = std::max(out.size_gib_max, gib);
        size_sum += s;
    }
    out.gib_per_plot = static_cast<double>(size_sum) /
        (kGiBBytes * static_cast<double>(out.plot_sizes.size()));

    // plot_sizes is in entry order, not completion order, so the sizes
    // of the measured plots specifically can't be identified. Use the
    // mean size across the run instead — on-disk size at fixed k is
    // near-constant, and this keeps the rate consistent with the
    // s/plot and GiB/plot figures printed alongside it.
    double const mean_plot_bytes =
        static_cast<double>(size_sum) / double(out.plot_sizes.size());

    // Derive the rate from the aggregate s/plot rather than recomputing it from
    // a wall span, so every figure the bench prints (s/plot, GiB/plot, TiB/s,
    // TiB/day, disk-fill ETA) descends from one number and none can disagree
    // with the others. For a single worker this is identically the old
    // bytes x N / wall — same value, same digits.
    out.rate_tib_s = out.s_per_plot > 0.0
        ? (mean_plot_bytes / kTibBytes) / out.s_per_plot
        : 0.0;

    return out;
}

void print_bench_measurement(char const* label,
                           BenchMeasurement const& m,
                           int k,
                           char const* suffix = nullptr)
{
    double const tib_hour = m.rate_tib_s * 3600.0;
    double const tib_day  = tib_hour * 24.0;
    double const tib_month = tib_day * 30.0;
    // %.3g, not %.6f. The README's own smoke test (-k 18) runs at ~5.6e-7 TiB/s,
    // which %.6f rounds to a printed "0.000000" — a zero rate — while the three
    // fields beside it (already %.3g) carry three real digits off the same
    // number. One line, contradicting itself. At k=28 both spellings agree.
    std::fprintf(stderr,
        "[bench] %s: %.3g TiB/s | %.3g TiB/hour | %.3g TiB/day | "
        "%.3g TiB/month (30d)",
        label, m.rate_tib_s, tib_hour, tib_day, tib_month);
    if (suffix && suffix[0]) std::fprintf(stderr, "  [%s]", suffix);
    std::fprintf(stderr, "\n");
}

// Per-worker breakdown. On a work-queue this is the only honest view of a
// multi-device run: the plot counts show how unevenly the queue actually split,
// each worker's σ is the real per-plot spread (the merged timeline's gaps are
// an interleaving artifact), and the aggregate steady-state is the SUM of the
// rates printed here.
void print_bench_workers(pos2gpu::BenchStats const& stats,
                         std::size_t warmup_per_worker,
                         std::size_t queue_len,
                         std::size_t worker_count)
{
    // "raise -n" is useless advice on its own: the queue is sized as if the
    // workers were equal, and they never are, so a user has no way to guess how
    // far to raise it. We DO know — this worker retired `plots_total` out of a
    // `queue_len` queue, so scale that up to the warmup + 3 it needs and say the
    // number. On a fast GPU + CPU rig the honest answer is startlingly large,
    // which is itself the finding.
    auto suggest_n = [&](std::size_t plots_total) -> int {
        std::size_t const want = warmup_per_worker + 3;
        double const got = static_cast<double>(std::max<std::size_t>(plots_total, 1));
        double const need_queue =
            static_cast<double>(queue_len) * (static_cast<double>(want) / got);
        double const n = need_queue / static_cast<double>(worker_count)
                       - static_cast<double>(warmup_per_worker);
        return static_cast<int>(std::ceil(std::max(1.0, n)));
    };

    std::fprintf(stderr, "[bench] per-worker steady-state:\n");
    // Same naming rule the batch's log prefixes use, from the same function, so
    // the two can be read side by side. It matters here: with --cpu-workers 4
    // every one of these lines would otherwise be called "cpu", and the block's
    // whole purpose is telling workers apart.
    std::vector<int> worker_devices;
    worker_devices.reserve(stats.workers.size());
    for (auto const& w : stats.workers) worker_devices.push_back(w.device_id);
    std::vector<std::string> const names = pos2gpu::worker_labels(worker_devices);

    for (std::size_t wi = 0; wi < stats.workers.size(); ++wi) {
        auto const& w = stats.workers[wi];
        std::string const& name = names[wi];
        if (!w.measured) {
            // A deeper queue answers both live reasons — it leaves more plots
            // past the warmup drop AND holds the window open longer — so the
            // -n suggestion stands whichever one fired.
            std::fprintf(stderr,
                "[bench]   %-5s %zu plot%s finished — %s; excluded from the "
                "aggregate (needs about -n %d to be measurable here)\n",
                name.c_str(), w.plots_total, w.plots_total == 1 ? "" : "s",
                unmeasured_reason(w.why), suggest_n(w.plots_total));
            continue;
        }
        char dropped[64] = {0};
        if (w.past_window > 0) {
            std::snprintf(dropped, sizeof(dropped), ", %zu past window",
                          w.past_window);
        }
        char tail[96] = {0};
        if (w.idle_tail_seconds >= 0.05) {
            std::snprintf(tail, sizeof(tail),
                          " — then idle %.1f s while a peer drained the queue",
                          w.idle_tail_seconds);
        }
        // "interval", not "plot time". A GPU worker is a producer/consumer
        // pipeline over a depth-1 queue, so what we stamp is when a plot lands
        // on disk, not how long it took to make. Starve the consumer (a CPU
        // co-worker eating every core), then free it, and it retires a staged
        // plot in ~1 s — a real run showed "min 1.01" on a GPU that cannot
        // produce a plot in under 6.6 s. The mean is exact regardless (the
        // intervals tile the window by construction); the spread is a contention
        // signal, and a bare "min" invited reading it as a 1-second plot.
        std::fprintf(stderr,
            "[bench]   %-5s %zu plots (%zu warmup, %zu measured%s) — "
            "%.2f s/plot (interval min %.2f / max %.2f, σ %.2f)%s\n",
            name.c_str(), w.plots_total, w.warmup_dropped, w.plots_measured,
            dropped, w.s_per_plot, w.interval_min, w.interval_max,
            w.interval_stddev, tail);
        if (w.plots_measured < 3) {
            std::fprintf(stderr,
                "[bench]   WARNING: %s measured on only %zu plot%s — its rate is "
                "noisy and it feeds the aggregate; try -n %d\n",
                name.c_str(), w.plots_measured,
                w.plots_measured == 1 ? "" : "s", suggest_n(w.plots_total));
        }
    }
    std::fprintf(stderr,
        "[bench]   steady window: %.1f s → %.1f s (%.1f s, every worker busy) — "
        "the aggregate below is the sum of the rates above\n",
        stats.window_begin, stats.window_end,
        stats.window_end - stats.window_begin);

    // The rates above are exactly what sizing a batch needs, and a bench run is
    // where someone has just measured them. An unmeasured worker's s_per_plot is
    // 0, which pick_batch_sizing reads as "no rate" and drops.
    std::vector<double> rates;
    rates.reserve(stats.workers.size());
    for (auto const& w : stats.workers) rates.push_back(w.s_per_plot);
    print_batch_sizing("[bench]", pos2gpu::pick_batch_sizing(rates, worker_devices),
                       worker_devices);
}

std::vector<pos2gpu::BatchEntry> build_bench_entries(
    int k, int strength, bool testnet, int count, std::string const& out_dir)
{
    std::vector<pos2gpu::BatchEntry> entries;
    entries.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        pos2gpu::BatchEntry e;
        e.k = k;
        e.strength = strength;
        e.plot_index = 0;
        e.meta_group = 0;
        e.testnet = testnet;
        read_urandom(e.plot_id.data(), e.plot_id.size());
        e.out_dir = out_dir;
        e.out_name = "bench-" + bytes_to_hex(e.plot_id) + ".plot2";
        entries.push_back(std::move(e));
    }
    return entries;
}

void cleanup_bench_files(std::vector<std::filesystem::path> const& paths)
{
    for (auto const& p : paths) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
}

BenchMeasurement run_bench_pass(
    int k, int strength, bool testnet, int count,
    std::string const& out_dir,
    pos2gpu::BatchOptions const& opts,
    std::size_t warmup_plots,
    bool keep)
{
    auto entries = build_bench_entries(k, strength, testnet, count, out_dir);
    std::vector<std::filesystem::path> paths;
    paths.reserve(entries.size());
    for (auto const& e : entries) {
        paths.push_back(std::filesystem::path(e.out_dir) / e.out_name);
    }
    pos2gpu::BatchOptions run_opts = opts;
    run_opts.progress = !opts.quiet;
    run_opts.skip_existing = false;
    run_opts.continue_on_error = false;
    // A benchmark is exactly where a VRAM regression should be caught, so the
    // watchdog's check is fatal here by default: if a tier's real peak exceeds
    // what its model promised, the run fails instead of quietly publishing a
    // throughput number for a configuration that will OOM on a smaller card.
    // The 0 flag leaves an explicit POS2GPU_ASSERT_VRAM=0 from the user alone.
    setenv("POS2GPU_ASSERT_VRAM", "1", 0);
    try {
        auto const res = pos2gpu::run_batch(entries, run_opts);
        if (res.plots_failed > 0 || res.plots_written == 0) {
            throw std::runtime_error("bench plotting pass failed");
        }
        return analyze_bench_run(res, paths, warmup_plots);
    } catch (...) {
        // Don't leave partial bench output behind on failure (the files
        // are unfarmable junk; --keep only applies to successful runs).
        if (!keep) cleanup_bench_files(paths);
        throw;
    }
}

// Expand `@argfile` tokens in argv. The convention is: any argv
// element starting with `@` is replaced by the whitespace-tokenised
// contents of the named file. Lines starting with `#` are ignored.
// Tilde expansion: a leading `~/` becomes $HOME/.
//
// Example ~/.xchplot2:
//     # default farmer + pool keys + output dir
//     --farmer-pk 0a1b2c... (96 hex)
//     --pool-contract xch1abc...
//     --out /mnt/plots
//     --tier minimal
//
// Then: `xchplot2 plot @~/.xchplot2 --num 10`
std::vector<std::string> expand_argfiles(int argc, char* argv[])
{
    std::vector<std::string> out;
    out.reserve(argc);
    char const* home = std::getenv("HOME");
    auto resolve_path = [&](std::string p) -> std::string {
        if (home && p.size() >= 2 && p[0] == '~' && p[1] == '/') {
            return std::string(home) + p.substr(1);
        }
        return p;
    };
    for (int i = 0; i < argc; ++i) {
        std::string tok = argv[i];
        if (tok.size() < 2 || tok[0] != '@') {
            out.push_back(std::move(tok));
            continue;
        }
        std::ifstream in(resolve_path(tok.substr(1)));
        if (!in) {
            std::cerr << "Error: cannot open argfile '" << tok.substr(1) << "'\n";
            out.push_back(std::move(tok));
            continue;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (auto h = line.find('#'); h != std::string::npos) {
                line = line.substr(0, h);
            }
            std::istringstream is(line);
            std::string word;
            while (is >> word) out.push_back(std::move(word));
        }
    }
    return out;
}

} // namespace

extern "C" int xchplot2_main(int argc, char* argv[])
{
    pos2gpu::install_cancel_signal_handlers();

    // Layer 1: TOML-subset config from `--config PATH` or
    // `~/.config/xchplot2/config.toml`. Schema:
    //   [defaults]              keys applied to every subcommand
    //   [batch] / [plot] / ...  per-subcommand overrides
    // Each key/value becomes a CLI flag and is injected into argv
    // BETWEEN the user's positional args and their flags so the
    // existing per-subcommand parsers see the same positional shape
    // and user-provided flags take precedence (last-wins).
    std::string config_path;
    int strip_argc = argc;
    std::vector<char*> argv_stripped;
    {
        argv_stripped.reserve(argc);
        for (int i = 0; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--config" && i + 1 < argc) {
                config_path = argv[i + 1];
                ++i;
                continue;
            }
            argv_stripped.push_back(argv[i]);
        }
        strip_argc = static_cast<int>(argv_stripped.size());
    }
    if (config_path.empty()) {
        if (char const* home = std::getenv("HOME")) {
            std::string const default_path =
                std::string(home) + "/.config/xchplot2/config.toml";
            std::ifstream probe(default_path);
            if (probe) config_path = default_path;
        }
    }
    std::vector<std::string> config_tokens;
    if (!config_path.empty()) {
        pos2gpu::ConfigFile cfg;
        try {
            cfg = pos2gpu::ConfigFile::load(config_path);
        } catch (std::exception const& ex) {
            std::cerr << "Error: parsing " << config_path << ": "
                      << ex.what() << "\n";
            return 1;
        }
        std::string subcmd = (strip_argc >= 2) ? std::string(argv_stripped[1]) : "";
        auto emit_section = [&](std::string const& name) {
            for (auto const& [k, v] : cfg.section_view(name)) {
                auto const as_bool = cfg.get_bool(name, k);
                std::string const flag =
                    (k.size() > 0 && k[0] == '-') ? k : ("--" + k);
                if (as_bool) {
                    if (*as_bool) {
                        config_tokens.push_back(flag);
                    } else if (flag.size() > 2 && flag.substr(0, 2) == "--") {
                        // false → --no-XXX so the CLI can re-enable
                        // via --XXX (last-wins). Short flags don't
                        // have a canonical negation form; skip them.
                        config_tokens.push_back("--no-" + flag.substr(2));
                    }
                    continue;
                }
                config_tokens.push_back(flag);
                config_tokens.push_back(v);
            }
        };
        emit_section("defaults");
        if (!subcmd.empty()) emit_section(subcmd);
    }

    // Layer 2: expand @argfile tokens. Inject config tokens between
    // user positional args and user flags so positional parsers
    // (batch's `argv[2] = manifest` etc.) still index correctly and
    // user-provided flags appear after config-provided ones.
    std::vector<std::string> expanded;
    if (!config_tokens.empty()) {
        int flag_start = strip_argc;
        for (int i = 2; i < strip_argc; ++i) {
            std::string const a = argv_stripped[i];
            if (a.size() >= 2 && a[0] == '-' && a[1] != '\0') {
                flag_start = i;
                break;
            }
        }
        std::vector<char*> combined;
        combined.reserve(strip_argc + config_tokens.size());
        for (int i = 0; i < flag_start; ++i) combined.push_back(argv_stripped[i]);
        std::vector<std::string> ct_owned = config_tokens;
        for (auto& t : ct_owned) combined.push_back(t.data());
        for (int i = flag_start; i < strip_argc; ++i) combined.push_back(argv_stripped[i]);
        expanded = expand_argfiles(static_cast<int>(combined.size()), combined.data());
    } else {
        expanded = expand_argfiles(strip_argc, argv_stripped.data());
    }
    std::vector<char*> argv_owned;
    argv_owned.reserve(expanded.size());
    for (auto& s : expanded) argv_owned.push_back(s.data());
    argc = static_cast<int>(argv_owned.size());
    argv = argv_owned.data();

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "devices") {
        // Enumerate every visible SYCL GPU device + the CPU plotter
        // (always available via AdaptiveCpp's OpenMP host backend).
        // Reports id, name, backend, capacity, and which sort path
        // the runtime dispatcher will route a worker on this device
        // to (CUB on cuda-backend queues when this build links the
        // CUB sort path; SortSycl otherwise — see SortDispatch.cpp).
        // Use the printed `[gpu0]` / `[cpu0]` index with `--devices`.
        auto devices = pos2gpu::list_gpu_devices();
        auto const cpu_nodes = pos2gpu::host_numa_nodes();
        // "1 CPU" was hard-coded, which a multi-socket host makes a lie — it
        // gets one selectable row per node, and the header has to count them.
        std::printf("Visible devices (%zu GPU + %zu CPU node%s):\n",
                    devices.size(), cpu_nodes.size(),
                    cpu_nodes.size() == 1 ? "" : "s");
        for (auto const& d : devices) {
            std::size_t vram_mb =
                static_cast<std::size_t>(d.vram_bytes / (1024ull * 1024ull));
#ifdef XCHPLOT2_HAVE_CUB
            char const* sort_hint = d.is_cuda_backend ? "CUB" : "SYCL";
#else
            char const* sort_hint = "SYCL";
#endif
            // One tag column shared with the CPU rows below, wide enough for the
            // longest tag either side prints ("[cpu0]"). The old hard-coded
            // "[%zu]   " also silently shunted the name column right on any host
            // with a two-digit GPU id.
            std::string const tag = "[" + std::to_string(d.id) + "]";
            std::printf("  %-7s%-32s backend=%-10s vram=%5zu MB  CUs=%-4u  sort:%s\n",
                        tag.c_str(), d.name.c_str(), d.backend.c_str(),
                        vram_mb, d.cu_count, sort_hint);
        }
        // One row per CPU node, because those are the ids --devices accepts.
        // hardware_concurrency() returns 0 when it can't figure the count out
        // (rare), in which case print "?".
        unsigned const threads = std::thread::hardware_concurrency();
        for (auto const& node : cpu_nodes) {
            // An empty cpu list means the kernel had no NUMA sysfs and
            // host_numa_nodes synthesised a node meaning "the machine" — so the
            // machine's thread count is the honest answer. (A single-socket host
            // with CONFIG_NUMA does list its cpus, and lands here with the same
            // number by a different route.)
            std::size_t const node_threads =
                node.cpus.empty() ? threads : node.cpus.size();
            std::string const tag = "[" + pos2gpu::device_label(
                pos2gpu::cpu_device_id(node.node_id)) + "]";
            std::string const name = cpu_nodes.size() > 1
                ? "Host CPU plotter (NUMA node " + std::to_string(node.node_id) + ")"
                : std::string("Host CPU plotter");
            if (node_threads == 0) {
                std::printf("  %-7s%-32s backend=%-10s threads=  ?            sort:SYCL  (1-2 orders slower than GPU)\n",
                            tag.c_str(), name.c_str(), "omp");
            } else {
                std::printf("  %-7s%-32s backend=%-10s threads=%-4zu           sort:SYCL  (1-2 orders slower than GPU)\n",
                            tag.c_str(), name.c_str(), "omp", node_threads);
            }
        }
        if (devices.empty()) {
            std::printf("\nNo GPU devices visible to AdaptiveCpp / SYCL.\n"
                        "Check rocminfo / nvidia-smi, ACPP_VISIBILITY_MASK, and that the\n"
                        "relevant SYCL backend was built into AdaptiveCpp.\n"
                        "The CPU plotter is always available via `--devices cpu` or `--cpu`.\n");
        } else {
            std::printf("\nUse `--devices gpu0` (or bare `0`) for a specific GPU,\n"
                        "     `--devices gpu` for every GPU,\n"
                        "     `--devices cpu` for every CPU node (opt-in; slow),\n"
                        "     `--devices cpu0` for one CPU node,\n"
                        "     `--devices all` for every GPU + every CPU node,\n"
                        "  or any comma combination (e.g. `0,2,cpu`).\n"
                        "--devices names DEVICES; `--cpu-workers N` sets how many\n"
                        "plots run on each CPU node.\n");
        }
        return 0;
    }

    if (mode == "bench") {
        int k = 28;
        int strength = 2;
        int measured = 10;
        int warmup = 1;
        bool keep = false;
        bool compute_only = false;
        bool testnet = false;
        std::string out_dir = ".";
        double target_size_tib = -1.0;
        pos2gpu::BatchOptions opts{};

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](int more) -> bool {
                if (i + more >= argc) {
                    std::cerr << "Error: " << a << " requires " << more << " arg(s)\n";
                    return false;
                }
                return true;
            };
            if      ((a == "--k" || a == "-k") && need(1)) k = std::atoi(argv[++i]);
            else if ((a == "--strength" || a == "-s") && need(1)) strength = std::atoi(argv[++i]);
            else if ((a == "--num" || a == "-n") && need(1)) measured = std::atoi(argv[++i]);
            else if (a == "--warmup" && need(1)) warmup = std::atoi(argv[++i]);
            else if ((a == "--out" || a == "-o") && need(1)) out_dir = argv[++i];
            else if (a == "--keep") keep = true;
            else if (a == "--compute-only") compute_only = true;
            else if (a == "--quiet" || a == "-q") opts.quiet = true;
            else if (a == "--no-quiet") opts.quiet = false;
            else if (a == "--testnet" || a == "-T") testnet = true;
            else if (a == "--target-size" && need(1)) {
                target_size_tib = std::atof(argv[++i]);
                if (target_size_tib <= 0.0) {
                    std::cerr << "Error: --target-size must be > 0 TiB\n";
                    return 1;
                }
            }
            else if (a == "-v" || a == "--verbose") opts.verbose = true;
            // CPU is opt-in. --cpu asks for it on every node without naming
            // --devices; it says nothing about HOW MANY, so it never disturbs an
            // explicit --cpu-workers N in either direction.
            else if (a == "--cpu") opts.cpu_opt_in = true;
            else if (a == "--cpu-workers" && need(1)) {
                if (!parse_cpu_workers_arg(argv[++i], opts)) return 1;
            }
            else if (a == "--shard-plot") opts.shard_plot = true;
            else if (a == "--pipeline-plot") opts.pipeline_plot = true;
            else if (a == "--tier" && need(1)) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal"
                    && t != "tiny" && t != "pinned" && t != "auto") {
                    std::cerr << "Error: --tier expects "
                                 "plain|compact|minimal|tiny|pinned|auto\n";
                    return 1;
                }
                opts.streaming_tier = (t == "auto") ? "" : t;
            }
            else if (a == "--devices" && need(1)) {
                if (!parse_devices_arg(argv[++i], opts)) {
                    std::cerr << "Error: invalid --devices value\n";
                    return 1;
                }
            }
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        if (k < 18 || k > 32 || (k % 2) != 0) {
            std::cerr << "Error: -k must be an even integer in [18, 32]\n";
            return 1;
        }
        if (measured < 1) {
            std::cerr << "Error: -n must be >= 1\n";
            return 1;
        }
        if (warmup < 0) {
            std::cerr << "Error: --warmup must be >= 0\n";
            return 1;
        }

        std::size_t const worker_count = pos2gpu::batch_worker_count(opts, k);
        // Per worker, not a global total: each worker has its own cold-start
        // plot to exclude. The queue size is still sized off the worker count.
        std::size_t const warmup_per_worker = static_cast<std::size_t>(warmup);
        int const plot_count = (warmup + measured) * static_cast<int>(worker_count);

        // Hoisted out of the try so the catch can still sweep them: once a pass
        // SUCCEEDS its plots are on disk, and run_bench_pass only removes the
        // output of the pass that threw. A --compute-only run whose second pass
        // died therefore returned straight out of the catch and abandoned the
        // first pass's entire set — 10-30 GiB at k=28.
        BenchMeasurement e2e{};
        BenchMeasurement compute{};
        std::string      tmpfs_scratch;  // per-run dir to remove, if we made one

        // Both passes' plots, and the tmpfs dir we minted for the second, on every
        // exit path. Deleting the e2e output before measuring free space is also
        // load-bearing: the disk-fill estimate must not be reduced by the bench's
        // own just-written output.
        auto sweep = [&] {
            if (!keep) {
                cleanup_bench_files(e2e.paths);
                cleanup_bench_files(compute.paths);
            }
            // The tmpfs scratch goes even under --keep. Those plots are a second
            // copy of the same synthetic set, written to RAM only so the pass
            // could be timed without a disk; "keeping" them means pinning 10-30
            // GiB of RAM until reboot, which is not what anyone means by --keep.
            // The e2e set on the real disk is the one worth keeping.
            if (!tmpfs_scratch.empty()) {
                std::error_code ec;
                std::filesystem::remove_all(tmpfs_scratch, ec);
            }
        };

        try {
            if (!opts.quiet) {
                if (worker_count == 1) {
                    std::fprintf(stderr,
                        "[bench] warmup: %d plot/worker (excluded). measured: %d plots/worker.\n",
                        warmup, measured);
                } else {
                    // Say what -n actually does. It is neither a run total (the
                    // queue holds worker_count times more than that) nor a
                    // per-worker share (nobody hands out shares) — it sizes the
                    // queue, and the queue then pays out by speed.
                    std::fprintf(stderr,
                        "[bench] warmup: %d plot/worker (excluded). queue: %d plots "
                        "for %zu workers.\n", warmup, plot_count, worker_count);
                    std::fprintf(stderr,
                        "[bench]   -n %d sizes the queue at (%d warmup + %d) x %zu "
                        "workers; the work-queue then hands each plot to whoever is "
                        "free, so a fast worker takes more than %d and a slow one "
                        "fewer — see the split below.\n",
                        measured, warmup, measured, worker_count, measured);
                }
            }

            e2e = run_bench_pass(
                k, strength, testnet, plot_count, out_dir, opts,
                warmup_per_worker, keep);

            std::size_t const ran = e2e.stats.workers.size();
            if (ran > 1) {
                print_bench_workers(e2e.stats, warmup_per_worker,
                                    static_cast<std::size_t>(plot_count),
                                    worker_count);
            }

            // A worker that finished too few plots to measure still did work
            // and still contended for the host — it just can't be credited. The
            // aggregate is then strictly below what the rig sustains, and must
            // not be quoted as if it were the answer.
            char bound[80] = {0};
            if (e2e.stats.workers_unmeasured > 0) {
                std::snprintf(bound, sizeof(bound),
                    "  [LOWER BOUND — %zu worker(s) unmeasured]",
                    e2e.stats.workers_unmeasured);
            }
            std::fprintf(stderr,
                "[bench] steady-state: %.2f s/plot, %.3f GiB/plot (k=%d, %zu worker%s)%s\n",
                e2e.s_per_plot, e2e.gib_per_plot, k, ran,
                ran == 1 ? "" : "s", bound);

            if (ran == 1) {
                auto const& w = e2e.stats.workers.front();
                std::fprintf(stderr,
                    "[bench]   per-plot spread: size %.3f GiB (min %.3f / max %.3f), "
                    "interval %.2f s (min %.2f / max %.2f, σ %.2f)\n",
                    e2e.gib_per_plot, e2e.size_gib_min, e2e.size_gib_max,
                    w.s_per_plot, w.interval_min, w.interval_max,
                    w.interval_stddev);
            } else {
                // No aggregate interval spread here: the gap between successive
                // completions across workers is short when two finish together
                // and long across a slow worker's plot, so it measures the
                // interleaving, not the plotter. The per-worker σ above is the
                // real spread. Size spread is still meaningful.
                std::fprintf(stderr,
                    "[bench]   per-plot spread: size %.3f GiB (min %.3f / max %.3f)\n",
                    e2e.gib_per_plot, e2e.size_gib_min, e2e.size_gib_max);
            }

            print_bench_measurement("end-to-end", e2e, k);

            std::string compute_label;
            if (compute_only) {
                // The compute-only pass writes the SAME plot_count plots, and
                // nothing is deleted until the whole run ends — so the tmpfs has
                // to hold the entire set at once. /dev/shm defaults to half of
                // RAM: at k=28 a 2-worker -n 10 run is 22 x 0.92 GiB = 20 GiB,
                // which ENOSPCs a 32 GB box partway through a pass that already
                // cost minutes. The e2e pass just measured the real plot size, so
                // check before spending the time, not during.
                double const need_bytes =
                    e2e.gib_per_plot * kGiBBytes * static_cast<double>(plot_count);
                std::string compute_dir = prepare_tmpfs_scratch();
                if (!compute_dir.empty()) {
                    double const avail = static_cast<double>(
                        std::filesystem::space(compute_dir).available);
                    if (need_bytes > avail) {
                        std::fprintf(stderr,
                            "[bench] WARNING: compute-only needs %.1f GiB in %s "
                            "but only %.1f GiB is free\n",
                            need_bytes / kGiBBytes, compute_dir.c_str(),
                            avail / kGiBBytes);
                        std::error_code rm_ec;
                        std::filesystem::remove_all(compute_dir, rm_ec);
                        compute_dir.clear();
                    }
                }
                if (compute_dir.empty()) {
                    compute_dir = out_dir;
                    compute_label = "compute+cache";
                    std::fprintf(stderr,
                        "[bench] WARNING: no usable tmpfs — compute-only uses "
                        "%s and may reflect page cache, not RAM\n",
                        compute_dir.c_str());
                } else {
                    compute_label = "tmpfs";
                    tmpfs_scratch = compute_dir;  // remove the dir, not just its plots
                }
                compute = run_bench_pass(
                    k, strength, testnet, plot_count, compute_dir, opts,
                    warmup_per_worker, keep);
                print_bench_measurement("compute-only", compute, k,
                                        compute_label.c_str());
                if (e2e.s_per_plot > 0.0 && compute.s_per_plot > 0.0) {
                    // From the two s/plot figures directly, NOT from the two
                    // rates. rate = mean_plot_bytes / s_per_plot, and the passes
                    // mint fresh random plot_ids, so a rate ratio carries a
                    // stray (B_e2e / B_compute) size factor. Disk overhead is a
                    // few percent, so even a sub-percent size difference is a
                    // large relative error in it — large enough to flip the sign.
                    double const overhead_pct =
                        100.0 * (1.0 - compute.s_per_plot / e2e.s_per_plot);
                    // A negative "overhead" is not a disk that gives time back:
                    // it means the two passes differ by more than the effect being
                    // measured. Printing "~-2.3%" invited reading it as a result.
                    if (overhead_pct >= 0.0) {
                        std::fprintf(stderr,
                            "[bench]   disk overhead: ~%.1f%% of wall\n",
                            overhead_pct);
                    } else {
                        std::fprintf(stderr,
                            "[bench]   disk overhead: none measurable — the "
                            "in-RAM pass came out %.1f%% slower, so run-to-run "
                            "noise here exceeds the disk's cost\n",
                            -overhead_pct);
                    }
                }
            }

            // Delete before measuring free space so the fill estimate
            // isn't reduced by the bench's own just-written output.
            sweep();
            if (keep) {
                for (auto const& p : e2e.paths) {
                    std::fprintf(stderr, "[bench] kept %s\n", p.c_str());
                }
                // ...but only the ones sweep() actually left behind: the
                // compute-only set is gone if it lived in the tmpfs scratch.
                std::size_t dropped = 0;
                for (auto const& p : compute.paths) {
                    std::error_code ec;
                    if (std::filesystem::exists(p, ec)) {
                        std::fprintf(stderr, "[bench] kept %s\n", p.c_str());
                    } else {
                        ++dropped;
                    }
                }
                if (dropped > 0) {
                    std::fprintf(stderr,
                        "[bench] note: %zu compute-only plot(s) not kept — they "
                        "were written to RAM (tmpfs) and would have held it "
                        "until reboot\n", dropped);
                }
            }

            double const fill_bytes = (target_size_tib > 0.0)
                ? target_size_tib * kTibBytes
                : static_cast<double>(
                    std::filesystem::space(out_dir).available);
            if (e2e.rate_tib_s > 0.0 && fill_bytes > 0.0) {
                double const fill_s = fill_bytes / (e2e.rate_tib_s * kTibBytes);
                std::fprintf(stderr,
                    "[bench] %s: %.3g TiB %s — fully plotted in ~%s (end-to-end rate)\n",
                    out_dir.c_str(),
                    fill_bytes / kTibBytes,
                    target_size_tib > 0.0 ? "target" : "free",
                    format_duration_dh(fill_s).c_str());
            }
            return 0;
        } catch (std::exception const& e) {
            sweep();
            std::cerr << "[bench] FAILED: " << e.what() << "\n";
            return 2;
        }
    }

    if (mode == "batch") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        std::string manifest = argv[2];
        pos2gpu::BatchOptions opts{};
        int progress_tri = -1;  // -1 auto (TTY), 0 off, 1 on
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "-v" || a == "--verbose")        opts.verbose = true;
            else if (a == "--no-verbose")                  opts.verbose = false;
            else if (a == "-q" || a == "--quiet")          opts.quiet = true;
            else if (a == "--no-quiet")                    opts.quiet = false;
            else if (a == "--progress")                    progress_tri = 1;
            else if (a == "--no-progress")                 progress_tri = 0;
            else if (a == "--skip-existing"
                  || a == "--resume")                      opts.skip_existing = true;
            else if (a == "--no-skip-existing"
                  || a == "--no-resume")                   opts.skip_existing = false;
            else if (a == "--continue-on-error")           opts.continue_on_error = true;
            else if (a == "--no-continue-on-error")        opts.continue_on_error = false;
            else if (a == "--cpu")                         opts.cpu_opt_in = true;
            else if (a == "--cpu-workers" && i + 1 < argc) {
                if (!parse_cpu_workers_arg(argv[++i], opts)) return 1;
            }
            else if (a == "--shard-plot")                  opts.shard_plot = true;
            else if (a == "--no-shard-plot")               opts.shard_plot = false;
            else if (a == "--pipeline-plot")               opts.pipeline_plot = true;
            else if (a == "--no-pipeline-plot")            opts.pipeline_plot = false;
            else if (a == "--pipeline-depth" && i + 1 < argc) {
                int const d = std::atoi(argv[++i]);
                if (d < 1) { std::cerr << "Error: --pipeline-depth must be >= 1\n"; return 1; }
                opts.pipeline_depth = d;
            }
            else if (a == "--strategy" && i + 1 < argc) {
                std::string const s = argv[++i];
                if      (s == "auto")          opts.strategy = pos2gpu::BatchStrategy::Auto;
                else if (s == "work-queue" ||
                         s == "workqueue")     opts.strategy = pos2gpu::BatchStrategy::WorkQueue;
                else if (s == "pipeline" ||
                         s == "pipeline-plot") opts.strategy = pos2gpu::BatchStrategy::PipelinePlot;
                else if (s == "shard" ||
                         s == "shard-plot")    opts.strategy = pos2gpu::BatchStrategy::ShardPlot;
                else {
                    std::cerr << "Error: --strategy expects auto|work-queue|"
                                 "pipeline|shard (got '" << s << "')\n";
                    return 1;
                }
            }
            else if (a == "--pipeline-stage-tiers" && i + 1 < argc) {
                std::string const spec = argv[++i];
                auto parse_tier = [](std::string const& t)
                    -> std::optional<pos2gpu::PipelineStageTier> {
                    if (t == "tiny" || t == "Tiny")
                        return pos2gpu::PipelineStageTier::Tiny;
                    if (t == "minimal" || t == "Minimal" || t == "min")
                        return pos2gpu::PipelineStageTier::Minimal;
                    return std::nullopt;
                };
                std::vector<pos2gpu::PipelineStageTier> tiers;
                std::size_t pos = 0;
                while (pos <= spec.size()) {
                    auto colon = spec.find(':', pos);
                    auto end = (colon == std::string::npos) ? spec.size() : colon;
                    auto tok = spec.substr(pos, end - pos);
                    auto t = parse_tier(tok);
                    if (!t) {
                        std::cerr << "Error: --pipeline-stage-tiers tiers must be "
                                     "'tiny' or 'minimal' (got '" << tok << "')\n";
                        return 1;
                    }
                    tiers.push_back(*t);
                    if (colon == std::string::npos) break;
                    pos = colon + 1;
                }
                if (tiers.size() != 2 && tiers.size() != 3) {
                    std::cerr << "Error: --pipeline-stage-tiers expects "
                                 "'STAGE1:STAGE2' or 'STAGE1:STAGE2:STAGE3' "
                                 "(got " << tiers.size() << " stages)\n";
                    return 1;
                }
                opts.pipeline_tiers = std::move(tiers);
            }
            else if (a == "--host-bounce")                 opts.prefer_peer_copy = false;
            else if (a == "--prefer-peer-copy")            { /* now the default, kept as a no-op alias */ }
            else if (a == "--tier" && i + 1 < argc) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal"
                    && t != "tiny" && t != "pinned" && t != "auto") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', 'tiny', 'pinned', or 'auto' (got '"
                              << t << "')\n";
                    return 1;
                }
                opts.streaming_tier = (t == "auto") ? "" : t;
            }
            else if (a == "--devices" && i + 1 < argc) {
                if (!parse_devices_arg(argv[++i], opts)) {
                    std::cerr << "Error: --devices expects 'all', 'cpu', or a "
                                 "comma-separated list of device ids "
                                 "(got '" << argv[i] << "')\n";
                    return 1;
                }
            }
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        if (opts.quiet && opts.verbose) {
            std::cerr << "Error: -q/--quiet and -v/--verbose are mutually "
                         "exclusive\n";
            return 1;
        }
        opts.progress = resolve_progress(progress_tri, opts.quiet);
        try {
            auto entries = pos2gpu::parse_manifest(manifest);
            if (!opts.quiet) {
                std::cerr << "[batch] " << entries.size() << " plots queued\n";
            }
            auto res = pos2gpu::run_batch(entries, opts);
            if (!opts.quiet) print_run_summary("[batch]", res);
            return (res.plots_failed > 0) ? 3 : 0;
        } catch (std::exception const& e) {
            std::cerr << "[batch] FAILED: " << e.what() << "\n";
            return 2;
        }
    }

    if (mode == "verify") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        std::string plotfile = argv[2];
        size_t trials = 100;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if ((a == "--trials" || a == "-n") && i + 1 < argc) {
                long v = std::atol(argv[++i]);
                if (v <= 0) {
                    std::cerr << "Error: --trials must be > 0\n";
                    return 1;
                }
                trials = static_cast<size_t>(v);
            } else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        try {
            std::cerr << "[verify] " << plotfile << ": running " << trials
                      << " random challenges\n";
            auto res = pos2gpu::verify_plot_file(plotfile, trials);
            std::cerr << "[verify] " << res.trials << " trials, "
                      << res.challenges_with_proof << " with >=1 proof, "
                      << res.proofs_found << " proofs total\n";
            if (res.proofs_found == 0) {
                std::cerr << "[verify] FAIL: no proofs produced — plot is "
                             "likely corrupt\n";
                return 4;
            }
            std::cerr << "[verify] OK\n";
            return 0;
        } catch (std::exception const& e) {
            std::cerr << "[verify] FAILED: " << e.what() << "\n";
            return 2;
        }
    }

    if (mode == "parity-check") {
        std::string dir = "./build/tools/parity";
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if ((a == "--dir" || a == "-d") && i + 1 < argc) {
                dir = argv[++i];
            } else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        // Glob every *_parity binary in `dir`. Same code path works for
        // both branches — main ships sycl_*_parity extras that cuda-only
        // doesn't, and the wildcard picks up whichever actually exists.
        std::vector<std::filesystem::path> tests;
        std::error_code ec;
        if (std::filesystem::is_directory(dir, ec)) {
            for (auto const& entry :
                 std::filesystem::directory_iterator(dir, ec))
            {
                auto const name = entry.path().filename().string();
                constexpr char const kSuffix[] = "_parity";
                constexpr size_t kLen = sizeof(kSuffix) - 1;
                bool const ends =
                    name.size() >= kLen &&
                    name.compare(name.size() - kLen, kLen, kSuffix) == 0;
                if (ends && entry.is_regular_file(ec)) {
                    tests.push_back(entry.path());
                }
            }
        }
        if (tests.empty()) {
            std::cerr << "No `*_parity` binaries found under " << dir << ".\n"
                         "Build them first:\n"
                         "  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release\n"
                         "  cmake --build build --parallel\n"
                         "Then re-run from the repo root, or pass --dir <path>.\n";
            return 2;
        }
        std::sort(tests.begin(), tests.end());

        int pass = 0, fail = 0;
        std::cerr << "==> parity tests (" << tests.size() << " found in "
                  << dir << ")\n";
        for (auto const& test : tests) {
            auto const name = test.filename().string();
            std::string const log_path =
                "/tmp/xchplot2-parity-" + name + ".log";
            // Redirecting through the shell: `test` is a path we
            // generated ourselves from a directory listing — no user-
            // controlled shell metachars reach this string.
            std::string const cmd =
                test.string() + " >" + log_path + " 2>&1";
            auto const t0 = std::chrono::steady_clock::now();
            int const rc = std::system(cmd.c_str());
            auto const ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
            if (rc == 0) {
                std::fprintf(stderr, "  PASS  %-32s  (%.1f ms)\n",
                             name.c_str(), ms);
                ++pass;
            } else {
                std::fprintf(stderr,
                             "  FAIL  %-32s  (exit %d; log: %s)\n",
                             name.c_str(), rc, log_path.c_str());
                ++fail;
            }
        }
        std::fprintf(stderr, "\n==> %d passed, %d failed\n", pass, fail);
        return fail > 0 ? 1 : 0;
    }

    if (mode == "plot") {
        // Standalone farmable-plot path: derive plot_id + memo internally.
        int k = 28;
        int strength = 2;
        int num = 1;
        int plot_index_base = 0;
        int meta_group = 0;
        bool testnet = false;
        bool verbose = false;
        bool skip_existing = false;
        bool continue_on_error = false;
        std::string out_dir = ".";
        std::string farmer_pk_hex, pool_pk_hex, pool_ph_hex, pool_addr;
        std::string seed_hex;
        std::vector<int> plot_device_ids;
        bool plot_use_all_devices = false;
        bool plot_devices_specified = false;
        int  plot_cpu_workers     = pos2gpu::kCpuWorkersAuto;  // count, once selected
        bool plot_cpu_opt_in      = false;   // --cpu / --cpu-workers N; CPU is opt-in
        std::vector<int> plot_cpu_node_ids;  // --devices cpu0,cpu1
        bool plot_use_all_cpu_nodes = false; // --devices cpu / all
        bool plot_shard_plot      = false;
        int  plot_progress_tri    = -1;  // -1 auto (TTY), 0 off, 1 on
        bool plot_quiet           = false;
        bool plot_pipeline_plot   = false;
        int  plot_pipeline_depth  = 2;
        pos2gpu::BatchStrategy plot_strategy = pos2gpu::BatchStrategy::Auto;
        bool plot_prefer_peer_copy = true;  // default flipped — Peer is faster on every tested topology; --host-bounce opts back to the explicit two-bounce path.
        std::string plot_streaming_tier;
        std::map<int, std::string> plot_per_device_tier;
        std::string plot_all_gpus_tier;
        std::vector<pos2gpu::PipelineStageTier> plot_pipeline_tiers;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](int more) -> bool {
                if (i + more >= argc) {
                    std::cerr << "Error: " << a << " requires " << more << " arg(s)\n";
                    return false;
                }
                return true;
            };
            if      ((a == "--k"          || a == "-k") && need(1)) k        = std::atoi(argv[++i]);
            else if ((a == "--num"        || a == "-n") && need(1)) num      = std::atoi(argv[++i]);
            else if ((a == "--strength"   || a == "-s") && need(1)) strength = std::atoi(argv[++i]);
            else if ((a == "--out"        || a == "-o") && need(1)) out_dir  = argv[++i];
            else if ((a == "--farmer-pk"  || a == "-f") && need(1)) farmer_pk_hex = argv[++i];
            else if ((a == "--pool-pk"    || a == "-p") && need(1)) pool_pk_hex   = argv[++i];
            else if  (a == "--pool-ph"                  && need(1)) pool_ph_hex   = argv[++i];
            else if ((a == "--pool-contract-address" || a == "-c") && need(1)) pool_addr = argv[++i];
            else if ((a == "--plot-index" || a == "-i") && need(1)) plot_index_base = std::atoi(argv[++i]);
            else if ((a == "--meta-group" || a == "-g") && need(1)) meta_group      = std::atoi(argv[++i]);
            else if ((a == "--seed"       || a == "-S") && need(1)) seed_hex        = argv[++i];
            else if  (a == "--testnet"    || a == "-T") testnet = true;
            else if  (a == "--no-testnet")              testnet = false;
            else if  (a == "-v" || a == "--verbose")    verbose = true;
            else if  (a == "--no-verbose")              verbose = false;
            else if  (a == "-q" || a == "--quiet")      plot_quiet = true;
            else if  (a == "--no-quiet")                plot_quiet = false;
            else if  (a == "--progress")                plot_progress_tri = 1;
            else if  (a == "--no-progress")             plot_progress_tri = 0;
            else if  (a == "--skip-existing"
                   || a == "--resume")                  skip_existing = true;
            else if  (a == "--no-skip-existing"
                   || a == "--no-resume")               skip_existing = false;
            else if  (a == "--continue-on-error")       continue_on_error = true;
            else if  (a == "--no-continue-on-error")    continue_on_error = false;
            else if  (a == "--cpu")                     plot_cpu_opt_in = true;
            else if  (a == "--cpu-workers" && need(1)) {
                pos2gpu::BatchOptions tmp;
                if (!parse_cpu_workers_arg(argv[++i], tmp)) return 1;
                plot_cpu_workers = tmp.cpu_workers;
                // Naming a count is a request for the CPU — carry that, or
                // `plot --cpu-workers 4` would set a count for a CPU it never
                // selected and plot on the GPU alone.
                plot_cpu_opt_in |= tmp.cpu_opt_in;
            }
            else if  (a == "--shard-plot")              plot_shard_plot = true;
            else if  (a == "--no-shard-plot")           plot_shard_plot = false;
            else if  (a == "--pipeline-plot")           plot_pipeline_plot = true;
            else if  (a == "--no-pipeline-plot")        plot_pipeline_plot = false;
            else if  (a == "--pipeline-depth" && need(1)) {
                int const d = std::atoi(argv[++i]);
                if (d < 1) { std::cerr << "Error: --pipeline-depth must be >= 1\n"; return 1; }
                plot_pipeline_depth = d;
            }
            else if  (a == "--strategy" && need(1)) {
                std::string const s = argv[++i];
                if      (s == "auto")          plot_strategy = pos2gpu::BatchStrategy::Auto;
                else if (s == "work-queue" ||
                         s == "workqueue")     plot_strategy = pos2gpu::BatchStrategy::WorkQueue;
                else if (s == "pipeline" ||
                         s == "pipeline-plot") plot_strategy = pos2gpu::BatchStrategy::PipelinePlot;
                else if (s == "shard" ||
                         s == "shard-plot")    plot_strategy = pos2gpu::BatchStrategy::ShardPlot;
                else {
                    std::cerr << "Error: --strategy expects auto|work-queue|"
                                 "pipeline|shard (got '" << s << "')\n";
                    return 1;
                }
            }
            else if  (a == "--pipeline-stage-tiers" && need(1)) {
                std::string const spec = argv[++i];
                auto parse_tier = [](std::string const& t)
                    -> std::optional<pos2gpu::PipelineStageTier> {
                    if (t == "tiny" || t == "Tiny")
                        return pos2gpu::PipelineStageTier::Tiny;
                    if (t == "minimal" || t == "Minimal" || t == "min")
                        return pos2gpu::PipelineStageTier::Minimal;
                    return std::nullopt;
                };
                std::vector<pos2gpu::PipelineStageTier> tiers;
                std::size_t pos = 0;
                while (pos <= spec.size()) {
                    auto colon = spec.find(':', pos);
                    auto end = (colon == std::string::npos) ? spec.size() : colon;
                    auto tok = spec.substr(pos, end - pos);
                    auto t = parse_tier(tok);
                    if (!t) {
                        std::cerr << "Error: --pipeline-stage-tiers tiers must be "
                                     "'tiny' or 'minimal' (got '" << tok << "')\n";
                        return 1;
                    }
                    tiers.push_back(*t);
                    if (colon == std::string::npos) break;
                    pos = colon + 1;
                }
                if (tiers.size() != 2 && tiers.size() != 3) {
                    std::cerr << "Error: --pipeline-stage-tiers expects "
                                 "'STAGE1:STAGE2' or 'STAGE1:STAGE2:STAGE3' "
                                 "(got " << tiers.size() << " stages)\n";
                    return 1;
                }
                plot_pipeline_tiers = std::move(tiers);
            }
            else if  (a == "--host-bounce")             plot_prefer_peer_copy = false;
            else if  (a == "--prefer-peer-copy")        { /* now the default, kept as a no-op alias */ }
            else if  (a == "--tier" && need(1)) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal"
                    && t != "tiny" && t != "pinned" && t != "auto") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', 'tiny', 'pinned', or 'auto' (got '"
                              << t << "')\n";
                    return 1;
                }
                plot_streaming_tier = (t == "auto") ? "" : t;
            }
            else if  (a == "--devices" && need(1)) {
                pos2gpu::BatchOptions tmp;
                if (!parse_devices_arg(argv[++i], tmp)) {
                    std::cerr << "Error: --devices expects 'all', 'gpu', 'cpu', "
                                 "'gpu<n>', 'cpu<n>', or a comma-separated list "
                                 "of device ids (got '" << argv[i] << "')\n";
                    return 1;
                }
                plot_device_ids       = std::move(tmp.device_ids);
                plot_use_all_devices  = tmp.use_all_devices;
                plot_devices_specified = true;
                // The CPU SELECTION is --devices' to replace; the count is not,
                // so plot_cpu_workers is untouched here. plot_cpu_opt_in is also
                // untouched — it belongs to --cpu / --cpu-workers, and letting a
                // later --devices clear it would make those flags order-dependent.
                plot_cpu_node_ids       = std::move(tmp.cpu_node_ids);
                plot_use_all_cpu_nodes  = tmp.use_all_cpu_nodes;
                plot_per_device_tier  = std::move(tmp.per_device_tier);
                plot_all_gpus_tier    = std::move(tmp.all_gpus_tier);
            }
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        if (plot_quiet && verbose) {
            std::cerr << "Error: -q/--quiet and -v/--verbose are mutually "
                         "exclusive\n";
            return 1;
        }
        if (farmer_pk_hex.empty()) {
            std::cerr << "Error: --farmer-pk is required\n";
            return 1;
        }
        // Exactly one pool source.
        int const pool_specs = int(!pool_pk_hex.empty())
                             + int(!pool_ph_hex.empty())
                             + int(!pool_addr.empty());
        if (pool_specs == 0) {
            std::cerr << "Error: a pool destination is required — pick one of "
                         "--pool-pk, --pool-ph, --pool-contract-address\n";
            return 1;
        }
        if (pool_specs > 1) {
            std::cerr << "Error: --pool-pk, --pool-ph, and --pool-contract-address "
                         "are mutually exclusive (saw " << pool_specs << ")\n";
            return 1;
        }
        if (num < 1) {
            std::cerr << "Error: --num must be >= 1\n";
            return 1;
        }
        if (plot_index_base < 0 || plot_index_base > 0xFFFF) {
            std::cerr << "Error: --plot-index must be in [0, 65535]\n";
            return 1;
        }
        // plot_index auto-increments across `-n N`; reject upfront if the
        // final plot's plot_index would exceed the u16 range.
        if (plot_index_base + num - 1 > 0xFFFF) {
            std::cerr << "Error: --plot-index + (--num - 1) exceeds 65535 "
                         "(base=" << plot_index_base << ", num=" << num << ")\n";
            return 1;
        }
        if (meta_group < 0 || meta_group > 0xFF) {
            std::cerr << "Error: --meta-group must be in [0, 255]\n";
            return 1;
        }

        std::vector<uint8_t> farmer_pk;
        if (!parse_hex_bytes(farmer_pk_hex, farmer_pk) || farmer_pk.size() != 48) {
            std::cerr << "Error: --farmer-pk must be 96 hex chars (48 bytes)\n";
            return 1;
        }

        std::vector<uint8_t> pool_key;
        int pool_kind = POS2_POOL_PH; // default unused; set in branches below
        if (!pool_pk_hex.empty()) {
            if (!parse_hex_bytes(pool_pk_hex, pool_key) || pool_key.size() != 48) {
                std::cerr << "Error: --pool-pk must be 96 hex chars (48 bytes)\n";
                return 1;
            }
            pool_kind = POS2_POOL_PK;
        } else if (!pool_ph_hex.empty()) {
            if (!parse_hex_bytes(pool_ph_hex, pool_key) || pool_key.size() != 32) {
                std::cerr << "Error: --pool-ph must be 64 hex chars (32 bytes)\n";
                return 1;
            }
            pool_kind = POS2_POOL_PH;
        } else {
            // --pool-contract-address (bech32m); decode via Rust shim.
            pool_key.assign(32, 0);
            int rc = pos2_keygen_decode_address(pool_addr.c_str(), pool_key.data());
            if (rc != POS2_OK) {
                std::cerr << "Error: --pool-contract-address invalid (rc=" << rc
                          << "; expected xch1.../txch1...)\n";
                return 1;
            }
            pool_kind = POS2_POOL_PH;
        }

        // Optional reproducible-build base seed.
        std::array<uint8_t, 32> base_seed{};
        bool have_base_seed = false;
        if (!seed_hex.empty()) {
            if (!parse_hex(seed_hex, base_seed)) {
                std::cerr << "Error: --seed must be 64 hex chars\n";
                return 1;
            }
            have_base_seed = true;
        }

        try {
            std::vector<pos2gpu::BatchEntry> entries;
            entries.reserve(static_cast<size_t>(num));
            for (int i = 0; i < num; ++i) {
                uint8_t seed[32];
                if (have_base_seed) {
                    int rc = pos2_keygen_derive_subseed(
                        base_seed.data(),
                        static_cast<uint64_t>(i),
                        seed);
                    if (rc != POS2_OK) {
                        std::cerr << "Error: subseed derivation failed (rc=" << rc << ")\n";
                        return 2;
                    }
                } else {
                    read_urandom(seed, sizeof(seed));
                }

                uint8_t plot_id[32];
                std::vector<uint8_t> memo(128);
                size_t memo_len = memo.size();
                // plot_index increments per plot so a single `plot -n N`
                // run produces plots with distinct plot_index values —
                // this is the within-group identifier the grouped-file
                // layout planned in pos2-chip will expect.
                uint16_t const plot_index_i =
                    static_cast<uint16_t>(plot_index_base + i);
                int rc = pos2_keygen_derive_plot(
                    seed, sizeof(seed),
                    farmer_pk.data(),
                    pool_key.data(), pool_kind,
                    static_cast<uint8_t>(strength),
                    plot_index_i,
                    static_cast<uint8_t>(meta_group),
                    plot_id,
                    memo.data(), &memo_len);
                if (rc != POS2_OK) {
                    std::cerr << "Error: pos2_keygen_derive_plot failed (rc=" << rc << ")\n";
                    return 2;
                }
                memo.resize(memo_len);

                pos2gpu::BatchEntry e;
                e.k          = k;
                e.strength   = strength;
                e.plot_index = plot_index_base + i;
                e.meta_group = meta_group;
                e.testnet    = testnet;
                std::copy(plot_id, plot_id + 32, e.plot_id.begin());
                e.memo       = std::move(memo);
                e.out_dir    = out_dir;
                e.out_name   = plot_id_to_filename(k, e.plot_id);
                entries.push_back(std::move(e));

                if (verbose) {
                    std::cerr << "[plot] prepared " << (i + 1) << "/" << num
                              << " " << e.out_name << "\n";
                }
            }

            pos2gpu::BatchOptions opts{};
            opts.verbose           = verbose;
            opts.skip_existing     = skip_existing;
            opts.continue_on_error = continue_on_error;
            opts.device_ids        = plot_device_ids;
            opts.use_all_devices   = plot_use_all_devices;
            opts.devices_specified = plot_devices_specified;
            opts.cpu_workers       = plot_cpu_workers;
            opts.cpu_node_ids      = plot_cpu_node_ids;
            opts.use_all_cpu_nodes = plot_use_all_cpu_nodes;
            opts.cpu_opt_in        = plot_cpu_opt_in;
            opts.shard_plot        = plot_shard_plot;
            opts.pipeline_plot          = plot_pipeline_plot;
            opts.pipeline_depth         = plot_pipeline_depth;
            opts.strategy               = plot_strategy;
            opts.pipeline_tiers         = plot_pipeline_tiers;
            opts.prefer_peer_copy  = plot_prefer_peer_copy;
            opts.streaming_tier    = plot_streaming_tier;
            opts.per_device_tier   = plot_per_device_tier;
            opts.all_gpus_tier     = plot_all_gpus_tier;
            opts.quiet             = plot_quiet;
            opts.progress          = resolve_progress(plot_progress_tri, plot_quiet);
            auto res = pos2gpu::run_batch(entries, opts);
            if (!plot_quiet) print_run_summary("[plot]", res);
            // stdout path listing is the machine-readable result — kept
            // under -q so scripts can still consume it.
            for (auto const& e : entries) {
                std::cout << out_dir << "/" << e.out_name << "\n";
            }
            return (res.plots_failed > 0) ? 3 : 0;
        } catch (std::exception const& e) {
            std::cerr << "[plot] FAILED: " << e.what() << "\n";
            return 2;
        }
    }

    if (mode == "completions") {
        // Shell completion script generator.
        //   xchplot2 completions bash | source /dev/stdin
        //   xchplot2 completions zsh  > _xchplot2 + add to fpath
        //   xchplot2 completions fish > ~/.config/fish/completions/xchplot2.fish
        if (argc < 3) {
            std::cerr << "Error: xchplot2 completions <bash|zsh|fish>\n";
            return 1;
        }
        std::string const shell = argv[2];
        if (shell == "bash") {
            std::cout << R"(# bash completion for xchplot2 — source this from ~/.bashrc:
#     source <(xchplot2 completions bash)
_xchplot2() {
    local cur prev words cword
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    local subcmds="batch plot bench test devices parity-check completions"
    local tiers="plain compact minimal tiny pinned auto"
    local devices_tokens="all gpu cpu 0 1 2 3"
    case "${prev}" in
        --tier)            COMPREPLY=( $(compgen -W "${tiers}" -- "$cur") ); return 0 ;;
        --devices)         COMPREPLY=( $(compgen -W "${devices_tokens}" -- "$cur") ); return 0 ;;
        -o|--out)          COMPREPLY=( $(compgen -d -- "$cur") ); return 0 ;;
        -f|--farmer-pk|-p|--pool-pk|--pool-ph|-c|--seed|-S) return 0 ;;
        completions)       COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") ); return 0 ;;
    esac
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${subcmds}" -- "$cur") )
        return 0
    fi
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "-v --verbose -q --quiet --progress --no-progress --cpu --cpu-workers --tier --devices --shard-plot --pipeline-plot --host-bounce --skip-existing --resume --config -k -n -f -p -c -o -T -i -g -S --help" -- "$cur") )
        return 0
    fi
}
complete -F _xchplot2 xchplot2
)";
        } else if (shell == "zsh") {
            std::cout << R"(#compdef xchplot2
_xchplot2() {
    local -a subcmds
    subcmds=(batch:"Run a manifest of plots" plot:"Single-plot farmable mode" bench:"Measure plotting throughput" test:"Single test plot" devices:"List available GPU/CPU" parity-check:"Run parity tests" completions:"Emit shell completion script")
    if (( CURRENT == 2 )); then
        _describe 'subcommand' subcmds
        return
    fi
    _arguments \
        '--tier[Streaming tier]:tier:(plain compact minimal tiny pinned auto)' \
        '--devices[Device selector]:spec:(all gpu cpu 0 1 2 3)' \
        '--progress[Force aggregate progress line on]' \
        '--no-progress[Force aggregate progress line off]' \
        '-v[Verbose]' '--verbose[Verbose]' \
        '-q[Quiet — suppress info-level output]' '--quiet[Quiet — suppress info-level output]' \
        '--cpu[Add CPU worker]' \
        '--cpu-workers[CPU plots to run concurrently: auto|max|N|off (default auto, RAM-gated)]:count:(auto max off)' \
        '--shard-plot[Single-plot multi-GPU]' \
        '--pipeline-plot[Pipeline-parallel multi-stage]' \
        '-o[Output dir]:dir:_files -/' \
        '*:: :->args'
}
_xchplot2 "$@"
)";
        } else if (shell == "fish") {
            std::cout << R"(# fish completion for xchplot2 — install at:
#     ~/.config/fish/completions/xchplot2.fish
complete -c xchplot2 -f
complete -c xchplot2 -n '__fish_use_subcommand' -a 'batch'         -d 'Run a manifest of plots'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'plot'          -d 'Single-plot farmable mode'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'bench'         -d 'Measure plotting throughput'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'test'          -d 'Single test plot'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'devices'       -d 'List available GPU/CPU'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'parity-check'  -d 'Run parity tests'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'completions'   -d 'Emit shell completion script'
complete -c xchplot2 -l tier      -x -a 'plain compact minimal tiny pinned auto'  -d 'Streaming tier'
complete -c xchplot2 -l devices   -x -a 'all gpu cpu 0 1 2 3'                      -d 'Device selector'
complete -c xchplot2 -l progress  -d 'Force aggregate progress line on'
complete -c xchplot2 -l no-progress -d 'Force aggregate progress line off'
complete -c xchplot2 -s v -l verbose -d 'Verbose'
complete -c xchplot2 -s q -l quiet -d 'Quiet — suppress info-level output'
complete -c xchplot2 -l cpu       -d 'Add CPU worker'
complete -c xchplot2 -l shard-plot     -d 'Single-plot multi-GPU (experimental)'
complete -c xchplot2 -l pipeline-plot  -d 'Pipeline-parallel multi-stage'
complete -c xchplot2 -s o -l out  -r -d 'Output dir'
complete -c xchplot2 -n "__fish_seen_subcommand_from completions" -a 'bash zsh fish'
)";
        } else {
            std::cerr << "Error: unknown shell '" << shell
                      << "' (expect bash, zsh, or fish)\n";
            return 1;
        }
        return 0;
    }

    if (mode != "test") {
        print_usage(argv[0]);
        return 1;
    }

    pos2gpu::GpuPlotOptions opts{};
    std::string output_dir = ".";

    // Strip flags from argv into a positional vector.
    std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--testnet"    || a == "-T") opts.testnet = true;
        else if  (a == "--gpu-t1") opts.t1 = pos2gpu::PhaseStrategy::Gpu;
        else if  (a == "--gpu-t2") opts.t2 = pos2gpu::PhaseStrategy::Gpu;
        else if  (a == "--gpu-t3") opts.t3 = pos2gpu::PhaseStrategy::Gpu;
        else if  (a == "--gpu-all"   || a == "-G") {
            opts.t1 = opts.t2 = opts.t3 = pos2gpu::PhaseStrategy::Gpu;
        }
        else if  (a == "--profile"   || a == "-P") opts.profile = true;
        else if ((a == "--out"       || a == "-o") && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if ((a == "--memo"      || a == "-m") && i + 1 < argc) {
            std::string memo_hex = argv[++i];
            if (!parse_hex_bytes(memo_hex, opts.memo)) {
                std::cerr << "Error: --memo must be even-length hex\n";
                return 1;
            }
        }
        else if ((a == "--out-name"  || a == "-N") && i + 1 < argc) {
            opts.out_name = argv[++i];
        }
        else {
            pos.push_back(a);
        }
    }

    if (pos.size() < 2) {
        print_usage(argv[0]);
        return 1;
    }

    opts.k = std::atoi(pos[0].c_str());
    if (!parse_hex(pos[1], opts.plot_id)) {
        std::cerr << "Error: plot_id must be 64 hex characters\n";
        return 1;
    }
    if (pos.size() >= 3) opts.strength    = std::atoi(pos[2].c_str());
    if (pos.size() >= 4) opts.plot_index  = std::atoi(pos[3].c_str());
    if (pos.size() >= 5) opts.meta_group  = std::atoi(pos[4].c_str());
    if (pos.size() >= 6) opts.verbose     = std::atoi(pos[5].c_str()) != 0;

    if (opts.testnet) {
        std::cout << "TESTNET plot — will NOT be valid on mainnet.\n";
    }

    try {
        std::string out = pos2gpu::plot_to_file(opts, output_dir);
        std::cout << "Wrote: " << out << "\n";
    } catch (std::exception const& e) {
        std::cerr << "Plotting failed: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
