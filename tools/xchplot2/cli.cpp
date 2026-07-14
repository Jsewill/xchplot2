// xchplot2 — standalone Chia v2 plot creator on GPU. Three modes:
//   test  : low-level single-plot harness (caller supplies plot_id + memo).
//   batch : drive a TSV manifest of pre-computed plots through the GPU
//           pipeline with producer/consumer staggering.
//   plot  : full standalone — derives plot_id + memo from caller-supplied
//           BLS keys via the keygen-rs Rust shim, then dispatches through
//           batch internally. The "real" entrypoint for users.

#include "gpu/DeviceIds.hpp"       // device_label() — never print a raw device id
#include "gpu/CudaDeviceList.hpp"  // list_cuda_devices() — backs the
                                   // `devices` subcommand below.
                                   // Plain-types header; the
                                   // cuda_runtime.h call lives in
                                   // CudaDeviceList.cu so cli.cpp
                                   // (compiled by g++) doesn't need
                                   // the CUDA include path.
#include "host/Cancel.hpp"
#include "host/ConfigFile.hpp"
#include "host/GpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "pos2_keygen.h" // Rust shim for plot_id + memo derivation

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
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
        << "         [--progress|--no-progress] [--devices SPEC]\n"
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
        << "    --devices SPEC                  : multi-device. SPEC is a comma\n"
        << "                                      list mixing any of:\n"
        << "                                        all       — every GPU + CPU\n"
        << "                                        gpu       — every visible CUDA GPU\n"
        << "                                        cpu       — CPU worker only (slow)\n"
        << "                                        0,1,3     — explicit GPU ids\n"
        << "                                      e.g. gpu,cpu == all.\n"
        << "                                      Any GPU selector accepts a `:tier`\n"
        << "                                      suffix to pin the streaming tier for\n"
        << "                                      that device(s). Tier ∈ plain | compact\n"
        << "                                      | minimal | tiny | auto. Per-GPU\n"
        << "                                      override wins over gpu:<tier> wins\n"
        << "                                      over global --tier. Examples:\n"
        << "                                        gpu,2:tiny     all GPUs auto, 2 = tiny\n"
        << "                                        all,2:tiny     all GPUs + CPU; 2 = tiny\n"
        << "                                        gpu:tiny,2:plain   all tiny, 2 = plain\n"
        << "                                        gpu:tiny,2:auto    all tiny, 2 auto\n"
        << "                                      `cpu:<tier>` is rejected (no streaming\n"
        << "                                      tier on the CPU worker).\n"
        << "                                      Omitted = single device via the\n"
        << "                                      CUDA-default device (zero-config).\n"
        << "    --cpu                           : add a CPU worker alongside the\n"
        << "                                      selected GPUs (or use CPU only when\n"
        << "                                      no GPU is selected). Goes through\n"
        << "                                      pos2-chip's Plotter — no CUDA calls,\n"
        << "                                      works on hosts with no GPU at all.\n"
        << "                                      Plotting is 1-2 orders of magnitude\n"
        << "                                      slower than GPU.\n"
        << "    --shard-plot                    : EXPERIMENTAL — opt in to single-plot\n"
        << "                                      multi-GPU. Each plot is processed by\n"
        << "                                      ALL --devices cooperatively (one plot\n"
        << "                                      at a time, sharded across GPUs)\n"
        << "                                      instead of the default work-queue\n"
        << "                                      (one plot per GPU). Phase 1 ships\n"
        << "                                      the surface area only; N>1 throws\n"
        << "                                      until the partition + multi-GPU sort\n"
        << "                                      land. Drop the flag for the existing\n"
        << "                                      multi-plot behaviour.\n"
        << "    --tier plain|compact|minimal|tiny|auto\n"
        << "                                    : force streaming pipeline tier when\n"
        << "                                      GPU pool doesn't fit.\n"
        << "                                        plain   = ~7.42 GiB floor, fastest\n"
        << "                                        compact = ~5.33 GiB floor, fits 8 GiB\n"
        << "                                        minimal = ~3.7 GiB floor, fits 4 GiB\n"
        << "                                                  (estimated; please report\n"
        << "                                                   actual fit on real 4 GiB)\n"
        << "                                        tiny    = ~1.1 GiB floor, sub-2 GiB\n"
        << "                                                  cards (P620 2GB, GTX 1050 2GB)\n"
        << "                                        auto    = pick largest that fits\n"
        << "                                      Equivalent to XCHPLOT2_STREAMING_TIER\n"
        << "                                      env var; CLI flag wins if both set.\n"
        << "  " << prog << " parity-check [--dir PATH]\n"
        << "    Run every *_parity binary in PATH (default: ./build/tools/parity)\n"
        << "    and summarize PASS/FAIL. Build the tests with `cmake --build\n"
        << "    <build-dir>` first. Useful for post-refactor regression screening.\n"
        << "  " << prog << " devices\n"
        << "    List every visible CUDA device + the host CPU plotter,\n"
        << "    with id, name, VRAM/threads, and per-GPU SM count + compute\n"
        << "    capability. Use the printed [N] / [cpu] index with --devices.\n"
        << "\n"
        << "  test-mode positional args:\n"
        << "    <k>            : even integer in [18, 30]\n"
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
        << "    -P, --profile      : print per-phase device-time breakdown\n";
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
//   "all"              → use every CUDA device visible at runtime
//                        (sets use_all_devices; device_ids stays empty).
//   "0"                → use only GPU id 0.
//   "0,2,3"            → use these specific device ids, in sorted order.
//
// Zero-configuration default (no flag) produces device_ids.empty() and
// use_all_devices=false — which triggers the single-device path on the
// CUDA-default device (identical to pre-multi-GPU behavior).
//
// Returns false on malformed input (caller prints usage + exits 1).
bool parse_devices_arg(std::string const& s, pos2gpu::BatchOptions& opts)
{
    // Accept comma-separated mix of:
    //   "all"             → all GPUs + CPU worker
    //   "gpu"             → all GPUs
    //   "cpu"             → CPU worker
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
               t == "tiny"  || t == "auto";
    };
    auto bad = [&](char const* why) {
        std::cerr << "Error: --devices: " << why
                  << " (token list: \"" << s << "\")\n";
        return false;
    };

    opts.device_ids.clear();
    opts.per_device_tier.clear();
    opts.all_gpus_tier.clear();
    bool any_token = false;
    bool any_gpu_token = false;
    size_t start = 0;
    while (start <= s.size()) {
        size_t const end = s.find(',', start);
        std::string const tok = s.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (tok.empty()) return false;
        any_token = true;

        // Split on `:` into selector + optional tier suffix.
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

        if (selector == "all") {
            opts.use_all_devices = true;
            opts.include_cpu = true;
            any_gpu_token = true;
            if (!tier_suffix.empty()) opts.all_gpus_tier = tier_suffix;
        } else if (selector == "gpu") {
            opts.use_all_devices = true;
            any_gpu_token = true;
            if (!tier_suffix.empty()) opts.all_gpus_tier = tier_suffix;
        } else if (selector == "cpu") {
            if (!tier_suffix.empty()) return bad("cpu token cannot carry a tier");
            opts.include_cpu = true;
        } else {
            char* endp = nullptr;
            long const v = std::strtol(selector.c_str(), &endp, 10);
            if (endp == selector.c_str() || *endp != '\0' || v < 0 || v > 1023) {
                return bad("unrecognised device token (expect all|gpu|cpu|<id>)");
            }
            int const id = static_cast<int>(v);
            opts.device_ids.push_back(id);
            any_gpu_token = true;
            if (!tier_suffix.empty()) {
                // Reject duplicate IDs with conflicting tier overrides.
                auto const it = opts.per_device_tier.find(id);
                if (it != opts.per_device_tier.end() && it->second != tier_suffix) {
                    return bad("same GPU id given two different :tier overrides");
                }
                opts.per_device_tier[id] = tier_suffix;
            }
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (!any_token) return false;
    if (!any_gpu_token && !opts.include_cpu) return false;
    std::sort(opts.device_ids.begin(), opts.device_ids.end());
    opts.device_ids.erase(
        std::unique(opts.device_ids.begin(), opts.device_ids.end()),
        opts.device_ids.end());
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
    int const total_s = static_cast<int>(seconds);
    int const d = total_s / 86400;
    int const h = (total_s % 86400) / 3600;
    int const m = (total_s % 3600) / 60;
    int const s = total_s % 60;
    char buf[32];
    if (d > 0) {
        std::snprintf(buf, sizeof(buf), "%dd %dh", d, h);
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

void print_run_summary(char const* prefix, pos2gpu::BatchResult const& res)
{
    double const per = res.plots_written
        ? res.total_wall_seconds / double(res.plots_written) : 0.0;
    std::cerr << prefix << " wrote " << res.plots_written << " plots in "
              << res.total_wall_seconds << " s (" << per << " s/plot)";
    if (res.plots_skipped) std::cerr << "; skipped " << res.plots_skipped;
    if (res.bytes_written > 0 && res.total_wall_seconds > 0.0) {
        double const gib = static_cast<double>(res.bytes_written) / kGiBBytes;
        double const tib_per_hour =
            (static_cast<double>(res.bytes_written) / kTibBytes) /
            (res.total_wall_seconds / 3600.0);
        std::cerr << "; " << gib << " GiB written ("
                  << tib_per_hour << " TiB/hour effective)";
    }
    std::cerr << "\n";
}

std::string resolve_tmpfs_dir()
{
    if (char const* x = std::getenv("XDG_RUNTIME_DIR"); x && x[0]) {
        std::error_code ec;
        if (std::filesystem::is_directory(x, ec)) return x;
    }
    std::error_code ec;
    if (std::filesystem::is_directory("/dev/shm", ec)) return "/dev/shm";
    return {};
}

struct BenchMeasurement {
    pos2gpu::BatchResult result;
    std::vector<std::filesystem::path> paths;
    std::vector<std::uint64_t> plot_sizes;
    double rate_tib_s = 0.0;
    double s_per_plot = 0.0;
    double gib_per_plot = 0.0;
    double size_gib_min = 0.0;
    double size_gib_max = 0.0;
    pos2gpu::BenchStats stats;   // per-worker + aggregate; see BenchStats.hpp
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
    (void)k;
    double const tib_hour = m.rate_tib_s * 3600.0;
    double const tib_day  = tib_hour * 24.0;
    double const tib_month = tib_day * 30.0;
    std::fprintf(stderr,
        "[bench] %s: %.6f TiB/s | %.3g TiB/hour | %.3g TiB/day | "
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
                         std::size_t warmup_per_worker)
{
    std::fprintf(stderr, "[bench] per-worker steady-state:\n");
    for (auto const& w : stats.workers) {
        std::string const name = pos2gpu::device_label(w.device_id);
        if (!w.measured) {
            std::fprintf(stderr,
                "[bench]   %-5s %zu plot%s finished — too few to measure past a "
                "%zu-plot warmup; excluded from the aggregate (raise -n)\n",
                name.c_str(), w.plots_total, w.plots_total == 1 ? "" : "s",
                warmup_per_worker);
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
                "noisy and it feeds the aggregate; raise -n\n",
                name.c_str(), w.plots_measured,
                w.plots_measured == 1 ? "" : "s");
        }
    }
    std::fprintf(stderr,
        "[bench]   steady window: %.1f s → %.1f s (%.1f s, every worker busy) — "
        "the aggregate below is the sum of the rates above\n",
        stats.window_begin, stats.window_end,
        stats.window_end - stats.window_begin);
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
    try {
        auto const res = pos2gpu::run_batch(entries, run_opts);
        if (res.plots_written == 0) {
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
// contents of the named file (`@~/.xchplot2` etc). Lines starting
// with `#` and blank lines are ignored. Tokens are separated by
// any whitespace (spaces, tabs, newlines) — quoting isn't supported
// in v1 (keep your values unquoted; --memo / --pool-ph etc. are hex
// or simple strings that don't need it).
//
// Example ~/.xchplot2:
//     # default farmer + pool keys + output dir
//     --farmer-pk 0a1b2c... (96 hex)
//     --pool-contract xch1abc...
//     --out /mnt/plots
//     --tier minimal
//
// Then: `xchplot2 plot @~/.xchplot2 --num 10`
//
// Tilde expansion: a leading `~/` in the path becomes $HOME/. No
// other shell expansions are performed (callers can fully qualify
// the path if needed).
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
            // Pass through so the actual parser surfaces the error too.
            out.push_back(std::move(tok));
            continue;
        }
        std::string line;
        while (std::getline(in, line)) {
            // Strip comments
            if (auto h = line.find('#'); h != std::string::npos) {
                line = line.substr(0, h);
            }
            // Whitespace-split each line
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

    // Layer 1: structured TOML-subset config from `--config PATH` or
    // `~/.config/xchplot2/config.toml` if it exists. Schema:
    //   [defaults]              keys applied to every subcommand
    //   [batch] / [plot] / ...  per-subcommand overrides
    // Each key/value is translated to a CLI flag and prepended to
    // argv, so the existing per-subcommand parsers process them
    // exactly as if they'd been typed. Real CLI args still win
    // because they come AFTER in argv (parsers process left-to-
    // right; last assignment wins for non-accumulating flags).
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
        // Pick the subcommand-specific section if present (argv[1]
        // is the subcommand). Apply [defaults] first then override
        // with [<subcmd>] so per-subcommand values win.
        std::string subcmd = (strip_argc >= 2) ? std::string(argv_stripped[1]) : "";
        auto emit_section = [&](std::string const& name) {
            for (auto const& [k, v] : cfg.section_view(name)) {
                // Bool true  → bare flag (`--progress`).
                // Bool false → `--no-progress` so the parser flips the
                //              same opts.progress bit. Important for
                //              config-set defaults the user wants to
                //              override on CLI: without the `--no-`
                //              path a config-set bool was permanently
                //              on (the only way to "negate" a flag is
                //              to not pass it, but config-injection
                //              forces it in every invocation).
                // Other      → flag + value pair.
                auto const as_bool = cfg.get_bool(name, k);
                std::string const flag =
                    (k.size() > 0 && k[0] == '-') ? k : ("--" + k);
                if (as_bool) {
                    if (*as_bool) {
                        config_tokens.push_back(flag);
                    } else {
                        // Emit `--no-XXX`. Only works for the `--`
                        // (long) form — short flags (`-v`) don't have
                        // a canonical negation, so skip the emit and
                        // rely on default-off behaviour.
                        if (flag.size() > 2 && flag.substr(0, 2) == "--") {
                            config_tokens.push_back("--no-" + flag.substr(2));
                        }
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

    // Layer 2: expand @argfile tokens (in argv AND in config-derived
    // tokens — a config value can itself reference @some/other/file).
    // The expansion's purpose is unchanged from before the config
    // layer landed: drop common args in a file, prefix invocations
    // with `@~/.xchplot2`.
    //
    // Inject config tokens between user positional args and user
    // flags so positional parsers (e.g. batch's `argv[2] = manifest`)
    // still see the right positionals, and user-provided flags appear
    // AFTER config-provided ones (last-wins → user overrides config).
    std::vector<std::string> expanded;
    if (!config_tokens.empty()) {
        // Find the first user arg that looks like a flag (starts with
        // `-` and isn't a bare `-`). Everything before it is positional,
        // everything from it onward is user flags.
        int flag_start = strip_argc;  // default: no flags → append
        for (int i = 2; i < strip_argc; ++i) {
            std::string const a = argv_stripped[i];
            if (a.size() >= 2 && a[0] == '-' && a[1] != '\0') {
                flag_start = i;
                break;
            }
        }
        std::vector<char*> combined;
        combined.reserve(strip_argc + config_tokens.size());
        // [prog] [subcmd] [positional args...]
        for (int i = 0; i < flag_start; ++i) {
            combined.push_back(argv_stripped[i]);
        }
        // <config flags>
        std::vector<std::string> ct_owned = config_tokens;
        for (auto& t : ct_owned) combined.push_back(t.data());
        // [user flags...]
        for (int i = flag_start; i < strip_argc; ++i) {
            combined.push_back(argv_stripped[i]);
        }
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
        // Enumerate every visible CUDA device + the host CPU plotter
        // (always available via --cpu / --devices cpu). Reports id,
        // name, VRAM/threads, and per-GPU SM count + compute capability.
        // Use the printed `[N]` / `[cpu]` index with `--devices`.
        auto query = pos2gpu::list_cuda_devices();
        if (!query.error.empty()) {
            std::printf("CUDA error: %s\n", query.error.c_str());
            std::printf("(no driver / no NVIDIA GPU / mismatched libcuda?)\n");
            std::printf("The CPU plotter is always available via `--devices cpu` or `--cpu`.\n");
            return 1;
        }
        std::printf("Visible devices (%zu GPU + 1 CPU):\n", query.devices.size());
        for (auto const& d : query.devices) {
            std::size_t vram_mb =
                static_cast<std::size_t>(d.vram_bytes / (1024ull * 1024ull));
            std::printf("  [%d]   %-32s vram=%5zu MB  SMs=%-3d  CC=%d.%d\n",
                        d.id, d.name.c_str(), vram_mb,
                        d.sm_count, d.cc_major, d.cc_minor);
        }
        unsigned threads = std::thread::hardware_concurrency();
        if (threads == 0) {
            std::printf("  [cpu] %-32s threads=  ?                          (1-2 orders slower than GPU)\n",
                        "Host CPU plotter");
        } else {
            std::printf("  [cpu] %-32s threads=%-4u                       (1-2 orders slower than GPU)\n",
                        "Host CPU plotter", threads);
        }
        if (query.devices.empty()) {
            std::printf("\nNo CUDA devices visible.\n"
                        "Check `nvidia-smi -L` and that the driver is loaded.\n"
                        "The CPU plotter is always available via `--devices cpu` or `--cpu`.\n");
        } else {
            std::printf("\nUse `--devices N` (id) for a specific GPU,\n"
                        "     `--devices gpu` for every GPU,\n"
                        "     `--devices cpu` for the host CPU only,\n"
                        "     `--devices all` for every GPU + CPU,\n"
                        "  or any comma combination (e.g. `0,2,cpu`).\n");
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
            else if (a == "--cpu") opts.include_cpu = true;
            else if (a == "--shard-plot") opts.shard_plot = true;
            else if (a == "--tier" && need(1)) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal"
                    && t != "tiny" && t != "auto") {
                    std::cerr << "Error: --tier expects plain|compact|minimal|tiny|auto\n";
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

        if (k < 18 || k > 30 || (k % 2) != 0) {
            std::cerr << "Error: -k must be an even integer in [18, 30]\n";
            return 1;
        }
        if (strength < 2 || strength > 63) {
            std::cerr << "Error: --strength must be in [2, 63]\n";
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

        std::size_t const worker_count = pos2gpu::batch_worker_count(opts);
        // Per worker, not a global total: each worker has its own cold-start
        // plot to exclude. The queue size is still sized off the worker count.
        std::size_t const warmup_per_worker = static_cast<std::size_t>(warmup);
        int const plot_count = (warmup + measured) * static_cast<int>(worker_count);

        try {
            // Fail the bench if a streaming tier outgrows the peak its floor is
            // derived from — the floors are only honest while that holds. Costs
            // nothing (no driver calls); setenv does not clobber an explicit 0.
            setenv("POS2GPU_ASSERT_VRAM", "1", 0);

            if (!opts.quiet) {
                if (worker_count == 1) {
                    std::fprintf(stderr,
                        "[bench] warmup: %d plot/worker (excluded). measured: %d plots/worker.\n",
                        warmup, measured);
                } else {
                    std::fprintf(stderr,
                        "[bench] warmup: %d plot/worker (excluded). queue: %d plots "
                        "for %zu workers.\n", warmup, plot_count, worker_count);
                    std::fprintf(stderr,
                        "[bench]   the work-queue hands each plot to whoever is free, "
                        "so -n is a target for the run, not a per-worker share — see "
                        "the split below.\n");
                }
            }

            BenchMeasurement e2e = run_bench_pass(
                k, strength, testnet, plot_count, out_dir, opts,
                warmup_per_worker, keep);

            std::size_t const ran = e2e.stats.workers.size();
            if (ran > 1) print_bench_workers(e2e.stats, warmup_per_worker);

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

            BenchMeasurement compute{};
            std::string compute_label;
            if (compute_only) {
                std::string compute_dir = resolve_tmpfs_dir();
                if (compute_dir.empty()) {
                    compute_dir = out_dir;
                    compute_label = "compute+cache";
                    std::fprintf(stderr,
                        "[bench] WARNING: no tmpfs available — compute-only uses "
                        "%s and may reflect page cache, not RAM\n",
                        compute_dir.c_str());
                } else {
                    compute_dir += "/xchplot2-bench";
                    std::filesystem::create_directories(compute_dir);
                    compute_label = "tmpfs";
                }
                compute = run_bench_pass(
                    k, strength, testnet, plot_count, compute_dir, opts,
                    warmup_per_worker, keep);
                print_bench_measurement("compute-only", compute, k,
                                        compute_label.c_str());
                if (e2e.rate_tib_s > 0.0 && compute.rate_tib_s > 0.0) {
                    double const overhead_pct =
                        100.0 * (1.0 - e2e.rate_tib_s / compute.rate_tib_s);
                    std::fprintf(stderr,
                        "[bench]   disk overhead: ~%.1f%% of wall\n", overhead_pct);
                }
            }

            // Delete before measuring free space so the fill estimate
            // isn't reduced by the bench's own just-written output.
            if (!keep) {
                cleanup_bench_files(e2e.paths);
                cleanup_bench_files(compute.paths);
            } else {
                for (auto const& p : e2e.paths) {
                    std::fprintf(stderr, "[bench] kept %s\n", p.c_str());
                }
                for (auto const& p : compute.paths) {
                    std::fprintf(stderr, "[bench] kept %s\n", p.c_str());
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
            if      (a == "-v" || a == "--verbose") opts.verbose = true;
            else if (a == "--no-verbose")           opts.verbose = false;
            else if (a == "-q" || a == "--quiet")   opts.quiet = true;
            else if (a == "--no-quiet")             opts.quiet = false;
            else if (a == "--progress")             progress_tri = 1;
            else if (a == "--no-progress")          progress_tri = 0;
            else if (a == "--skip-existing"
                  || a == "--resume")               opts.skip_existing = true;
            else if (a == "--no-skip-existing"
                  || a == "--no-resume")            opts.skip_existing = false;
            else if (a == "--cpu")                  opts.include_cpu = true;
            else if (a == "--no-cpu")               opts.include_cpu = false;
            else if (a == "--shard-plot")           opts.shard_plot = true;
            else if (a == "--no-shard-plot")        opts.shard_plot = false;
            else if (a == "--tier" && i + 1 < argc) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal" &&
                    t != "tiny" && t != "auto") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', 'tiny', or 'auto' (got '"
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
            return 0;
        } catch (std::exception const& e) {
            std::cerr << "[batch] FAILED: " << e.what() << "\n";
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
        std::string out_dir = ".";
        std::string farmer_pk_hex, pool_pk_hex, pool_ph_hex, pool_addr;
        std::string seed_hex;
        std::vector<int> plot_device_ids;
        bool plot_use_all_devices = false;
        bool plot_include_cpu     = false;
        bool plot_shard_plot      = false;
        int  plot_progress_tri    = -1;  // -1 auto (TTY), 0 off, 1 on
        bool plot_quiet           = false;
        bool plot_skip_existing   = false;
        std::string plot_streaming_tier;
        std::map<int, std::string> plot_per_device_tier;
        std::string plot_all_gpus_tier;

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
                   || a == "--resume")                  plot_skip_existing = true;
            else if  (a == "--no-skip-existing"
                   || a == "--no-resume")               plot_skip_existing = false;
            else if  (a == "--cpu")                     plot_include_cpu = true;
            else if  (a == "--no-cpu")                  plot_include_cpu = false;
            else if  (a == "--shard-plot")              plot_shard_plot = true;
            else if  (a == "--no-shard-plot")           plot_shard_plot = false;
            else if  (a == "--tier" && need(1)) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal" &&
                    t != "tiny" && t != "auto") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', 'tiny', or 'auto' (got '"
                              << t << "')\n";
                    return 1;
                }
                plot_streaming_tier = (t == "auto") ? "" : t;
            }
            else if  (a == "--devices" && need(1)) {
                pos2gpu::BatchOptions tmp;
                if (!parse_devices_arg(argv[++i], tmp)) {
                    std::cerr << "Error: --devices expects 'all', 'cpu', or a "
                                 "comma-separated list of device ids "
                                 "(got '" << argv[i] << "')\n";
                    return 1;
                }
                plot_device_ids       = std::move(tmp.device_ids);
                plot_use_all_devices  = tmp.use_all_devices;
                if (tmp.include_cpu) plot_include_cpu = true;
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
        if (pool_specs != 1) {
            std::cerr << "Error: exactly one of --pool-pk, --pool-ph, "
                         "--pool-contract-address is required\n";
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
        if (strength < 2 || strength > 63) {
            std::cerr << "Error: --strength must be in [2, 63]\n";
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
            opts.verbose          = verbose;
            opts.quiet            = plot_quiet;
            opts.progress         = resolve_progress(plot_progress_tri, plot_quiet);
            opts.skip_existing    = plot_skip_existing;
            opts.device_ids       = plot_device_ids;
            opts.use_all_devices  = plot_use_all_devices;
            opts.include_cpu      = plot_include_cpu;
            opts.shard_plot       = plot_shard_plot;
            opts.streaming_tier   = plot_streaming_tier;
            opts.per_device_tier  = plot_per_device_tier;
            opts.all_gpus_tier    = plot_all_gpus_tier;
            auto res = pos2gpu::run_batch(entries, opts);
            if (!plot_quiet) print_run_summary("[plot]", res);
            // stdout path listing is the machine-readable result — kept
            // under -q so scripts can still consume it.
            for (auto const& e : entries) {
                std::cout << out_dir << "/" << e.out_name << "\n";
            }
            return 0;
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
        // Coverage focuses on the most-used surfaces: subcommands,
        // --tier values, --devices tokens (with :tier suffix awareness
        // limited to the static value list), and the boolean flags.
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
    local tiers="plain compact minimal tiny auto"
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
        COMPREPLY=( $(compgen -W "-v --verbose -q --quiet --progress --no-progress --cpu --tier --devices --shard-plot --skip-existing --resume --config -k -n -f -p -c -o -T -i -g -S --help" -- "$cur") )
        return 0
    fi
}
complete -F _xchplot2 xchplot2
)";
        } else if (shell == "zsh") {
            std::cout << R"(#compdef xchplot2
# zsh completion for xchplot2 — add this directory to fpath and
# autoload xchplot2's completion function, then 'autoload -U compinit; compinit'.
_xchplot2() {
    local -a subcmds tiers
    subcmds=(batch:"Run a manifest of plots" plot:"Single-plot farmable mode" bench:"Measure plotting throughput" test:"Single test plot" devices:"List available GPU/CPU" parity-check:"Run parity tests" completions:"Emit shell completion script")
    tiers=(plain compact minimal tiny auto)
    case "$state" in
        cmds) _describe 'command' subcmds ;;
    esac
    if (( CURRENT == 2 )); then
        _describe 'subcommand' subcmds
        return
    fi
    _arguments \
        '--tier[Streaming tier]:tier:(plain compact minimal tiny auto)' \
        '--devices[Device selector]:spec:(all gpu cpu 0 1 2 3)' \
        '--progress[Force aggregate progress line on]' \
        '--no-progress[Force aggregate progress line off]' \
        '-q[Quiet — suppress info-level output]' '--quiet[Quiet — suppress info-level output]' \
        '-v[Verbose]' '--verbose[Verbose]' \
        '--cpu[Add CPU worker]' \
        '--shard-plot[Single-plot multi-GPU]' \
        '-o[Output dir]:dir:_files -/' \
        '*:: :->args'
}
_xchplot2 "$@"
)";
        } else if (shell == "fish") {
            std::cout << R"(# fish completion for xchplot2 — install at:
#     ~/.config/fish/completions/xchplot2.fish
complete -c xchplot2 -f
# Subcommands
complete -c xchplot2 -n '__fish_use_subcommand' -a 'batch'         -d 'Run a manifest of plots'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'plot'          -d 'Single-plot farmable mode'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'bench'         -d 'Measure plotting throughput'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'test'          -d 'Single test plot'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'devices'       -d 'List available GPU/CPU'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'parity-check'  -d 'Run parity tests'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'completions'   -d 'Emit shell completion script'
# Flags
complete -c xchplot2 -l tier      -x -a 'plain compact minimal tiny auto'  -d 'Streaming tier'
complete -c xchplot2 -l devices   -x -a 'all gpu cpu 0 1 2 3'              -d 'Device selector'
complete -c xchplot2 -l progress  -d 'Force aggregate progress line on'
complete -c xchplot2 -l no-progress -d 'Force aggregate progress line off'
complete -c xchplot2 -s q -l quiet -d 'Quiet — suppress info-level output'
complete -c xchplot2 -s v -l verbose -d 'Verbose'
complete -c xchplot2 -l cpu       -d 'Add CPU worker'
complete -c xchplot2 -l shard-plot -d 'Single-plot multi-GPU (experimental)'
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
