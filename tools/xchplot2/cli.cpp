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
#include "device_options.hpp"
#include "gpu/DeviceIds.hpp"   // device_label() — never print a raw device id
#include "host/GpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "group_commands.hpp"
#ifdef XCHPLOT2_GROUP_COMMANDS
#include "host/ComputeSession.hpp"
#endif
#include "host/NumaTopology.hpp"  // host_numa_nodes() — one `devices` row per CPU node
#include "host/Cancel.hpp"
#include "host/ConfigFile.hpp"
#include "host/Random.hpp"
#include "host/Json.hpp"
#include "BuildVersion.hpp"
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
#include <cctype>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#include <io.h>
#else
#include <unistd.h>  // isatty — progress defaults to on for interactive runs
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace {

void set_env(char const* name, char const* value, bool overwrite = true)
{
    if (!overwrite && std::getenv(name)) return;
#ifdef _WIN32
    int const rc = ::_putenv_s(name, value);
#else
    int const rc = ::setenv(name, value, 1);
#endif
    if (rc != 0) throw std::runtime_error(std::string("cannot set environment variable ") + name);
}

int run_parity_test(std::string const& path, std::FILE* log)
{
#ifdef _WIN32
    int saved_out = _dup(1), saved_err = _dup(2);
    if (saved_out < 0 || saved_err < 0) {
        if (saved_out >= 0) _close(saved_out);
        if (saved_err >= 0) _close(saved_err);
        return 127;
    }
    std::fflush(nullptr);
    bool const redirected = _dup2(_fileno(log), 1) == 0 && _dup2(_fileno(log), 2) == 0;
    char const* args[] = {"parity-test", nullptr};
    auto const rc = redirected ? _spawnv(_P_WAIT, path.c_str(), args) : -1;
    _dup2(saved_out, 1); _dup2(saved_err, 2);
    _close(saved_out); _close(saved_err);
    return rc < 0 ? 127 : static_cast<int>(rc);
#else
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return 127;
    int rc = posix_spawn_file_actions_adddup2(&actions, fileno(log), STDOUT_FILENO);
    if (rc == 0) rc = posix_spawn_file_actions_adddup2(&actions, fileno(log), STDERR_FILENO);
    pid_t pid = 0;
    char* args[] = {const_cast<char*>(path.c_str()), nullptr};
    if (rc == 0) rc = posix_spawn(&pid, path.c_str(), &actions, nullptr, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    if (rc != 0) return 127;
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return 127;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
}

// Keep decimal signs/leading whitespace accepted by strtol, but consume the
// entire argument and reject overflow before narrowing to the destination.
template<class T>
bool parse_integer_arg(std::string const& s, T& out, std::string const& name)
{
    char* end = nullptr;
    errno = 0;
    long long const value = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || end != s.c_str() + s.size() || errno == ERANGE ||
        value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max()) {
        std::cerr << "Error: " << name << " expects an integer (got '" << s << "')\n";
        return false;
    }
    out = static_cast<T>(value);
    return true;
}

// Parse a --max-host-ram value: "8G"/"8GiB"/"8g", "8192M"/"8192MiB",
// "512K", "4T", a raw byte count, or "min"/"0" (== spill everything
// routable, lowest floor). Binary units (G == GiB). Returns false on a
// malformed value; on success sets `out_bytes` (0 for min).
bool parse_host_ram_arg(std::string const& raw, std::uint64_t& out_bytes)
{
    std::string s = raw;
    for (auto& c : s) c = static_cast<char>(std::tolower((unsigned char)c));
    if (s == "min") { out_bytes = 0; return true; }
    if (s.empty()) return false;

    // Split trailing unit suffix (k/m/g/t, optional 'i', optional 'b').
    std::size_t pos = 0;
    long double value = 0.0L;
    try { value = std::stold(s, &pos); }
    catch (...) { return false; }
    if (!std::isfinite(value) || value < 0.0L) return false;

    std::string unit = s.substr(pos);
    std::uint64_t mult = 1;
    if (!unit.empty()) {
        char u = unit[0];
        if      (u == 'k') mult = 1ULL << 10;
        else if (u == 'm') mult = 1ULL << 20;
        else if (u == 'g') mult = 1ULL << 30;
        else if (u == 't') mult = 1ULL << 40;
        else if (u == 'b') mult = 1;
        else return false;
        // Allow "gib"/"gb"/"g"; reject anything else.
        std::string rest = unit.substr(1);
        if (!(rest.empty() || rest == "b" || rest == "ib")) return false;
    }
    value *= static_cast<long double>(mult);
    if (!std::isfinite(value) || value >= std::ldexp(1.0L, 64)) return false;
    out_bytes = static_cast<std::uint64_t>(value);
    return true;
}

// Apply the XCHPLOT2_MAX_HOST_RAM env fallback when --max-host-ram was not
// passed on the command line. The flag always wins.
void resolve_host_ram_env(pos2gpu::BatchOptions& o)
{
    // XCHPLOT2_NO_AUTO_SPILL turns off the automatic rescue. Read it first and
    // unconditionally: it is meaningful with or without an explicit budget.
    if (char const* v = std::getenv("XCHPLOT2_NO_AUTO_SPILL"); v && v[0]) {
        std::string const s = v;
        if (s == "1" || s == "true" || s == "yes" || s == "on")
            o.auto_host_ram_spill = false;
    }
    if (o.has_max_host_ram) return;
    if (char const* v = std::getenv("XCHPLOT2_MAX_HOST_RAM"); v && v[0]) {
        std::uint64_t b = 0;
        if (parse_host_ram_arg(v, b)) { o.has_max_host_ram = true; o.max_host_ram = b; }
    }
}

// Tri-state --progress resolution: explicit --progress/--no-progress
// wins; otherwise default to on when stderr is a TTY (where the line
// rewrites in place) and off when redirected or under -q/--quiet.
// -1 = auto, 0 = forced off, 1 = forced on.
bool resolve_progress(int tri, bool quiet)
{
    if (tri >= 0) return tri != 0;
#ifdef _WIN32
    return !quiet && ::_isatty(::_fileno(stderr)) != 0;
#else
    return !quiet && ::isatty(::fileno(stderr)) != 0;
#endif
}

void print_usage(char const* prog, std::ostream& out = std::cerr)
{
    out
        << "Usage:\n"
        << "  " << prog << " plan [OPTIONS] [--] [DIR...]\n"
        << "  " << prog << " fill [OPTIONS] [-r|--resume] [--] [DIR...]\n"
        << "  " << prog << " status [DIR|PLAN|--manifest FILE] [--watch] [--inspect] [--live]\n"
        << "  " << prog << " pause|drain|resume|retire [DIR|--manifest FILE] [--destination DIR]...\n"
        << "  " << prog << " grow FILE [--max-members N] [OPTIONS]\n"
        << "  " << prog << " compute COMMAND [OPTIONS]\n"
        << "    Farming-layout preview, capacity-driven plotting, group growth and shared compute.\n"
        << "    Use COMMAND --help for its options. Previews write only requested plan or profile exports.\n"
        << "  " << prog << " test <k> <group_id_hex> [strength] [plot_index] [meta_group] [verbose]\n"
        << "         [--raw] [-o|--out DIR] [-m|--memo HEX] [-N|--out-name NAME]\n"
        << "         [--gpu-t1] [--gpu-t2] [--gpu-t3] [-G|--gpu-all] [-P|--profile]\n"
        << "  " << prog << " batch <manifest.tsv> [-v|--verbose] [-q|--quiet]\n"
        << "         [-r|--resume] [--continue-on-error]\n"
        << "         [--progress|--no-progress] [-d|--devices <SPEC>]\n"
        << "    Manifest: one plot per non-empty/non-# line, whitespace-separated:\n"
        << "      k strength plot_index meta_group format group_id_hex memo_hex out_dir out_name\n"
        << "    The gplot-v2-grouped format adds a final member count.\n"
        << "    Runs GPU compute and CPU FSE in a producer/consumer pipeline so they overlap\n"
        << "    across consecutive plots. ~2x throughput vs separate `test` invocations.\n"
        << "  " << prog << " bench [-k K] [-s S] [-n N] [-o DIR] [-d|--devices SPEC]\n"
        << "         [--tier T] [--cpu] [--warmup W] [--keep]\n"
        << "         [--target-size TiB] [--compute-only] [--json] [-q|--quiet]\n"
        << "    Measure plotting throughput (TiB/hour, TiB/day, TiB/month) on\n"
        << "    synthetic unfarmable plots (default queue: 1 warmup + 10 measured\n"
        << "    plots/worker). Always writes real .gplot files; deletes them\n"
        << "    on exit unless --keep is set.\n"
        << "  " << prog << " plot -k K -n N -f HEX  ( -p HEX | --pool-ph HEX | -c xch1... )\n"
        << "         [-s S] [-o DIR] [-i N] [-g N] [-S HEX] [-v] [-q]\n"
        << "         [-r|--resume] [--manifest FILE] [--continue-on-error]\n"
        << "         [--group-members N] [--construction auto|incremental|staged|merged|external]\n"
        << "         [--compute auto|off] [--compute-local on|off] [--compute-directory DIR]\n"
        << "    Create farmable plot groups from the supplied farmer and pool keys.\n"
        << "    Saved pool membership enables automatic shared compute.\n"
        << "    -f, --farmer-pk HEX             : 96 hex chars (48 B G1 public key).\n"
        << "    -p, --pool-pk HEX               : 96 hex chars. Pool public key mode.\n"
        << "        --pool-ph HEX               : 64 hex chars (raw puzzle hash).\n"
        << "    -c, --pool-contract-address ADR : Chia bech32m address (xch1.../txch1...);\n"
        << "                                      decoded internally to a 32-byte hash.\n"
        << "    -k, --k K                       : k size: 28 for farming (default); smaller sizes for testing.\n"
        << "    -n, --num N                     : number of plots to create.\n"
        << "    --group-members N               : members per file, 1..65535.\n"
        << "                                      -n still counts files; default is one member per file.\n"
        << "    -j, --workers N                 : shared plotting CPU budget for grouped jobs.\n"
        << "    -s, --strength S                : v2 PoS strength (default 2).\n"
        << "    -o, --out DIR                   : output directory.\n"
        << "    --manifest FILE                 : save/reuse this job manifest (default: unique\n"
        << "                                      xchplot2-job-*.tsv in the output directory).\n"
        << "    -i, --plot-index N              : member index (must be 0 for single-member groups).\n"
        << "    -g, --meta-group N              : v2 PoS meta_group field (default 0).\n"
        << "    -S, --seed HEX                  : optional 64 hex chars of master-SK\n"
        << "                                      entropy. Per-plot seed = SHA256(seed || i).\n"
        << "                                      Reproducible across runs. Defaults to\n"
        << "                                      a fresh random seed per plot.\n"
        << "    -v, --verbose                   : per-plot progress on stderr.\n"
        << "    -q, --quiet                     : suppress info-level stderr output\n"
        << "                                      (progress line, summaries, tier\n"
        << "                                      notes). Warnings/errors and the\n"
        << "                                      stdout path listing still print.\n"
        << "    --progress / --no-progress      : force the aggregate progress line\n"
        << "                                      on/off. Default: on when stderr is\n"
        << "                                      a terminal (rewrites in place),\n"
        << "                                      off when redirected or with -q.\n"
        << "    -r, --resume, --skip-existing   : recover the saved plot job and skip files\n"
        << "                                      with matching identity, memo, and valid bounds.\n"
        << "    --continue-on-error             : log per-plot failures and keep going\n"
        << "                                      instead of aborting the batch.\n"
        << "    -d, --devices SPEC              : multi-device. SPEC is a comma\n"
        << "                                      list mixing any of:\n"
        << "                                        all       — every GPU + every CPU node\n"
        << "                                        gpu       — every visible GPU\n"
        << "                                        cpu       — every CPU node\n"
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
        << "                                        auto (default) — 4 per node CPU-only,\n"
        << "                                            1 beside a GPU, trimmed to host RAM.\n"
        << "                                        max            — as many as fit in RAM,\n"
        << "                                                         capped at core count.\n"
        << "                                        N              — exactly N per node.\n"
        << "                                        off            — none, overriding\n"
        << "                                                         --devices cpu.\n"
        << "                                      Each worker needs its own host working set.\n"
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
        << "    --tier plain|compact|minimal|tiny|pinned|auto\n"
        << "                                    : force a streaming tier even if the pool fits.\n"
        << "                                      Every tier must fit its peak plus VRAM buffer.\n"
        << "                                      Lower VRAM tiers use more host RAM.\n"
        << "                                      Pinned is manual; Tiny is the automatic floor.\n"
        << "                                      auto leaves environment/default selection in effect.\n"
        << "                                      A non-auto CLI tier overrides XCHPLOT2_STREAMING_TIER.\n"
        << "    -M, --max-host-ram SIZE         : cap the streaming path's pinned + anonymous\n"
        << "                                      host footprint by spilling its large\n"
        << "                                      cold tables (h_t1_meta, h_t3, ...) to\n"
        << "                                      a temp-dir file, largest-first, until\n"
        << "                                      the modelled resident host peak fits.\n"
        << "                                      'min' = spill everything routable\n"
        << "                                      (lowest floor). Unset = spill only\n"
        << "                                      if the tier does not fit host RAM\n"
        << "                                      (see --no-auto-spill). Env:\n"
        << "                                      XCHPLOT2_MAX_HOST_RAM (flag wins).\n"
        << "                                      SIZE accepts 8G, 8192M, bytes or 'min'.\n"
        << "    -t, --temp-dir PATH            : where spilled tables live (sets\n"
        << "                                      XCHPLOT2_TEMP_DIR). Prefer fast NVMe.\n"
        << "    --no-auto-spill               : refuse to plot when the tier does not\n"
        << "                                      fit host RAM, instead of spilling to\n"
        << "                                      --temp-dir automatically. Auto only\n"
        << "                                      fires where the run would otherwise\n"
        << "                                      be a hard error, so it never slows a\n"
        << "                                      run that already works. Env:\n"
        << "                                      XCHPLOT2_NO_AUTO_SPILL=1.\n"
        << "  " << prog << " verify <plotfile> [--trials N] [--full]\n"
        << "    Sample quality chains for N random challenges (default 100).\n"
        << "    --full also solves and validates full proofs; this costs more CPU time.\n"
        << "    Zero proofs across a sensible sample (>=100) strongly indicates a\n"
        << "    corrupt plot. Default N=100.\n"
        << "  " << prog << " parity-check [--dir PATH]\n"
        << "    Run every *_parity and *_test binary in PATH and summarize\n"
        << "    PASS/FAIL.\n"
        << "    Default PATH is ./build/tools/parity. Build the tests with\n"
        << "    `cmake --build <build-dir>` first. Useful for post-refactor\n"
        << "    regression screening.\n"
        << "  " << prog << " devices\n"
        << "    List every visible SYCL GPU device + the host CPU plotter\n"
        << "    with id, name, backend, capacity, and which sort path the\n"
        << "    runtime dispatcher will route a worker to (CUB on cuda-\n"
        << "    backend devices when this build links CUB, otherwise SortSycl).\n"
        << "    Use the printed [N] / [cpu] index with --devices in plot/batch.\n"
        << "  " << prog << " completions <bash|zsh|fish>\n"
        << "    Print a shell completion script to stdout.\n"
        << "\n"
        << "  Reusable options:\n"
        << "    -V, --version  print the version and source revision.\n"
        << "    --config FILE  load settings from named command sections. Default:\n"
#ifdef _WIN32
        << "                   %APPDATA%/xchplot2/config.toml\n"
#else
        << "                   $HOME/.config/xchplot2/config.toml\n"
#endif
        << "    @FILE          insert whitespace-separated arguments (no shell quoting).\n"
        << "\n"
        << "  test-mode positional args:\n"
        << "    <k>            : even integer in [18, 28]\n"
        << "    <group_id_hex>  : 64 hex characters\n"
        << "    [strength]     : optional, defaults to 2\n"
        << "    [plot_index]   : optional, defaults to 0\n"
        << "    [meta_group]   : optional, defaults to 0\n"
        << "    [verbose]      : optional, 0/1, default 0\n"
        << "  test-mode flags:\n"
        << "        --raw         : write a temporary raw member for group assembly\n"
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
        << "    POS2GPU_MAX_VRAM_MB=N         cap free VRAM and enforce the selected budget\n"
        << "                                  (MiB; tests this GPU, not another card).\n"
        << "    POS2GPU_STREAMING_STATS=1     log every streaming-path alloc / free.\n"
        << "    POS2GPU_POOL_DEBUG=1          log pool allocation sizes at construction.\n"
        << "    POS2GPU_PHASE_TIMING=1        per-phase wall-time breakdown on stderr.\n"
        << "    ACPP_GFX=gfxXXXX              AMD container AOT target; Cargo detects the\n"
        << "                                  GPU separately (see INSTALL.md).\n";
}

void print_command_help(char const* prog, std::string_view command, std::ostream& out = std::cout)
{
    out << "Usage: " << prog << ' ' << command;
    if (command == "plot")
        out << " -f FARMER_KEY -c POOL_ADDRESS [OPTIONS]\n"
            "Create plots and save a resumable job. Farming uses k=28; other sizes are test-only.\n"
            "  -n, --num N           plots to create\n"
            "  -o, --out DIR         output directory\n"
            "  -d, --devices SPEC    cpu, gpu, all, or selected device IDs\n"
            "  -M, --max-host-ram N  host RAM budget (for example, 14G)\n"
            "  -r, --resume          continue a saved job\n"
            "  --manifest FILE      choose the saved job file\n"
            "  --pool-ph HEX        pool hash instead of a contract address; --pool-pk also supported\n";
    else if (command == "batch")
        out << " MANIFEST [OPTIONS]\n"
            "Run or resume a saved plotting job.\n"
            "  -r, --resume          preserve completed plots and recover unfinished work\n"
            "  -d, --devices SPEC    cpu, gpu, all, or selected device IDs\n"
            "  -M, --max-host-ram N  host RAM budget (for example, 14G)\n";
    else if (command == "bench")
        out << " [OPTIONS]\n"
            "Measure plotting throughput using disposable synthetic plots.\n"
            "  -k N                 test size, default 28; only k=28 represents farming\n"
            "  -n N                 measured queue allowance per worker, default 10\n"
            "  --warmup N           warmup plots per worker, default 1\n"
            "  -d, --devices SPEC   cpu, gpu, all, or selected device IDs\n"
            "  -o, --out DIR        benchmark output directory\n"
            "  --keep               retain benchmark files\n"
            "  --compute-only       also measure a RAM-backed pass when available\n"
            "  --json               machine-readable measurements on stdout\n";
    else if (command == "test")
        out << " K GROUP_ID [STRENGTH] [INDEX] [META] [VERBOSE] [OPTIONS]\n"
            "Low-level test harness; use plot or fill for farming.\n"
            "  -o, --out DIR        output directory\n"
            "  -m, --memo HEX       caller-supplied memo\n"
            "  -N, --out-name NAME  output filename\n"
            "  --raw               write a temporary group member\n"
            "  -G, --gpu-all       run all phases on the GPU\n"
            "  -P, --profile       request phase timing\n";
    else if (command == "verify")
        out << " [OPTIONS] [--] FILE\n"
            "Sample quality chains; use --full to solve and validate full proofs.\n"
            "  -n, --trials N      random challenges, default 100\n"
            "  --full              validate full proofs\n";
    else if (command == "devices")
        out << " [OPTIONS]\nList selectable GPUs and CPU nodes.\n";
    else if (command == "parity-check")
        out << " [OPTIONS]\nRun available parity and host tests; failures include their output.\n"
            "  --dir PATH          test binary directory, default ./build/tools/parity\n";
    out << "  -v, --verbose       show diagnostics and detailed measurements\n"
        "  -q, --quiet         suppress informational output; keep results and errors\n";
    if (command == "plot" || command == "batch" || command == "bench")
        out << "  --progress / --no-progress  override automatic terminal progress\n";
    out << "  --config FILE       load command settings\nUse " << prog << ' ' << command
        << " --help-all for the complete reference.\n";
}

bool command_help(char const* prog, std::string_view command, std::string_view option)
{
    if (option == "--help" || option == "-h") { print_command_help(prog, command); return true; }
    if (option == "--help-all") { print_usage(prog, std::cout); return true; }
    return false;
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
    errno = 0;
    long const v = std::strtol(s.c_str(), &endp, 10);
    if (endp == s.c_str() || endp != s.c_str() + s.size() || errno == ERANGE || v < 0 || v > 64) {
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


std::string plot_id_to_filename(int k, std::array<uint8_t, 32> const& plot_id)
{
    // The sole member ID includes meta_group as well as the group identity.
    static char const hex[] = "0123456789abcdef";
    std::string out = "plot-k" + std::to_string(k) + "-";
    out.reserve(out.size() + 64 + 6);
    for (uint8_t b : plot_id) {
        out += hex[b >> 4];
        out += hex[b & 0xF];
    }
    out += ".gplot";
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

void print_run_summary(char const* prefix, pos2gpu::BatchResult const& res, bool verbose)
{
    double const per = res.plots_written
        ? res.total_wall_seconds / double(res.plots_written) : 0.0;
    std::cerr << prefix << " wrote " << res.plots_written << " plots in "
              << res.total_wall_seconds << " s";
    if (res.plots_written) std::cerr << " (" << per << " s/plot)";
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
    if (verbose) print_run_workers(prefix, res);
}

int batch_exit_code(pos2gpu::BatchResult const& res, std::size_t requested)
{
    if (res.plots_failed) return 3;
    return res.plots_written + res.plots_skipped < requested ? 4 : 0;
}

// Pick the roomiest tmpfs, not the first one that happens to exist. The old
// order took $XDG_RUNTIME_DIR ahead of /dev/shm — but systemd sizes /run/user at
// 10% of RAM and caps it, while /dev/shm gets 50%, so it reached for the smaller
// of the two for a pass that writes the entire plot set at once and deletes
// nothing until the end.
#ifndef _WIN32
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
#endif

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
#ifdef _WIN32
    return {};  // The caller uses its existing compute+cache fallback without tmpfs.
#else
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
#endif
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

// JSON is opt-in on stdout; the human report remains on stderr.
using pos2gpu::write_json_string;

void write_bench_json_pass(std::ostream& out, BenchMeasurement const& m,
                           std::string const& label)
{
    out << "{\"label\":";
    write_json_string(out, label);
    out << ",\"wall_seconds\":" << m.result.total_wall_seconds
        << ",\"plots_written\":" << m.result.plots_written
        << ",\"s_per_plot\":" << m.s_per_plot
        << ",\"gib_per_plot\":" << m.gib_per_plot
        << ",\"size_gib_min\":" << m.size_gib_min
        << ",\"size_gib_max\":" << m.size_gib_max
        << ",\"tib_per_second\":" << m.rate_tib_s
        << ",\"window_begin_seconds\":" << m.stats.window_begin
        << ",\"window_end_seconds\":" << m.stats.window_end
        << ",\"plots_measured\":" << m.stats.plots_measured
        << ",\"workers_unmeasured\":" << m.stats.workers_unmeasured
        << ",\"workers\":[";
    auto const labels = pos2gpu::worker_labels([&] {
        std::vector<int> ids;
        for (auto const& w : m.stats.workers) ids.push_back(w.device_id);
        return ids;
    }());
    for (std::size_t i = 0; i < m.stats.workers.size(); ++i) {
        auto const& w = m.stats.workers[i];
        if (i) out << ',';
        out << "{\"device\":";
        write_json_string(out, labels[i]);
        out << ",\"pipeline\":";
        write_json_string(out, m.result.workers[i].pipeline);
        out << ",\"plots_total\":" << w.plots_total
            << ",\"warmup_dropped\":" << w.warmup_dropped
            << ",\"past_window\":" << w.past_window
            << ",\"plots_measured\":" << w.plots_measured
            << ",\"measured\":" << (w.measured ? "true" : "false")
            << ",\"s_per_plot\":" << w.s_per_plot
            << ",\"interval_min_seconds\":" << w.interval_min
            << ",\"interval_max_seconds\":" << w.interval_max
            << ",\"interval_stddev_seconds\":" << w.interval_stddev << '}';
    }
    out << "]}";
}

void print_bench_measurement(char const* label,
                           BenchMeasurement const& m,
                           int k,
                           bool verbose,
                           char const* suffix = nullptr)
{
    double const tib_hour = m.rate_tib_s * 3600.0;
    double const tib_day  = tib_hour * 24.0;
    double const tib_month = tib_day * 30.0;
    // %.3g, not %.6f. The README's own smoke test (-k 18) runs at ~5.6e-7 TiB/s,
    // which %.6f rounds to a printed "0.000000" — a zero rate — while the three
    // fields beside it (already %.3g) carry three real digits off the same
    // number. One line, contradicting itself. At k=28 both spellings agree.
    if (verbose) std::fprintf(stderr,
        "[bench] %s: %.3g TiB/s | %.3g TiB/hour | %.3g TiB/day | %.3g TiB/month (30d)",
        label, m.rate_tib_s, tib_hour, tib_day, tib_month);
    else std::fprintf(stderr, "[bench] %s: %.3g TiB/hour", label, tib_hour);
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
                         std::size_t worker_count,
                         bool verbose, bool quiet)
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

    if (!quiet) std::fprintf(stderr, "[bench] per-worker steady-state:\n");
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
        if (verbose) std::fprintf(stderr,
            "[bench]   %-5s %zu plots (%zu warmup, %zu measured%s) — "
            "%.2f s/plot (interval min %.2f / max %.2f, σ %.2f)%s\n",
            name.c_str(), w.plots_total, w.warmup_dropped, w.plots_measured,
            dropped, w.s_per_plot, w.interval_min, w.interval_max,
            w.interval_stddev, tail);
        else if (!quiet) std::fprintf(stderr, "[bench]   %-5s %.2f s/plot (%zu measured plots)\n",
            name.c_str(), w.s_per_plot, w.plots_measured);
        if (w.plots_measured < 3) {
            std::fprintf(stderr,
                "[bench]   WARNING: %s measured on only %zu plot%s — its rate is "
                "noisy and it feeds the aggregate; try -n %d\n",
                name.c_str(), w.plots_measured,
                w.plots_measured == 1 ? "" : "s", suggest_n(w.plots_total));
        }
    }
    if (verbose) std::fprintf(stderr,
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
    if (verbose) print_batch_sizing("[bench]", pos2gpu::pick_batch_sizing(rates, worker_devices), worker_devices);
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
        pos2gpu::read_random_bytes(e.group_id.data(), e.group_id.size());
        e.plot_id = pos2gpu::plot_id_for_group(e.group_id, 0, 0);
        e.out_dir = out_dir;
        e.out_name = "bench-" + bytes_to_hex(e.group_id) + ".gplot";
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
    run_opts.skip_existing = false;
    run_opts.continue_on_error = false;
    // A benchmark is exactly where a VRAM regression should be caught, so the
    // watchdog's check is fatal here by default: if a tier's real peak exceeds
    // what its model promised, the run fails instead of quietly publishing a
    // throughput number for a configuration that will OOM on a smaller card.
    // Preserve an explicit POS2GPU_ASSERT_VRAM=0 from the user.
    set_env("POS2GPU_ASSERT_VRAM", "1", false);
    resolve_host_ram_env(run_opts);
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
#ifdef _WIN32
    if (!home) home = std::getenv("USERPROFILE");
#endif
    auto resolve_path = [&](std::string p) -> std::string {
        if (home && p.size() >= 2 && p[0] == '~' && p[1] == '/') {
            return std::string(home) + p.substr(1);
        }
        return p;
    };
    bool group = argc > 1 && pos2gpu::is_group_command(argv[1]);
    bool literal = false;
    bool path_list = false;
    unsigned values = 0;
    auto append = [&](std::string token) {
        if (path_list && !token.starts_with('-')) {}
        else if (values) --values;
        else if (!literal) {
            path_list = group && pos2gpu::group_path_list_option(token);
            if (token == "--") literal = true;
            else if (group && !path_list) values = pos2gpu::group_option_values(token);
        }
        out.push_back(std::move(token));
        if (out.size() == 2) group = pos2gpu::is_group_command(out[1]);
    };
    for (int i = 0; i < argc; ++i) {
        std::string tok = argv[i];
        if (i == 0) { out.push_back(std::move(tok)); continue; }
        if (literal || values || (path_list && !tok.starts_with('-')) || tok.size() < 2 || tok[0] != '@') {
            append(std::move(tok));
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
            if (auto h = line.find('#'); h != std::string::npos) line = line.substr(0, h);
            std::istringstream is(line);
            std::string word;
            while (is >> word) append(std::move(word));
        }
    }
    return out;
}

bool group_value_option(std::string_view option)
{
    static constexpr std::string_view options[] = {
        "--backend", "--device", "--max-ram", "--workers", "-j", "--batch-members",
        "--capacity-bytes", "--headroom-bytes", "--leave-free", "--temp-capacity-bytes", "--destination-temp",
        "--construction", "--member-storage", "--window-bytes", "--checkpoint-bytes",
        "--io-overlap", "--generation-overlap", "--cache-release"
    };
    return option == "--compute" || option.starts_with("--compute-") ||
        std::find(std::begin(options), std::end(options), option) != std::end(options);
}

std::vector<std::string> group_arguments(std::vector<std::string> const& input)
{
    std::vector<std::string> result;
    for (size_t i = 0; i < input.size(); ++i) {
        auto key = input[i];
        if (key == "--") {
            result.insert(result.end(), input.begin() + i, input.end());
            break;
        }
        if (pos2gpu::group_path_list_option(key)) {
            result.push_back(key);
            while (i + 1 < input.size() && !input[i + 1].starts_with('-')) result.push_back(input[++i]);
            continue;
        }
        if (key == "--relocate") {
            if (i + 2 >= input.size()) throw std::invalid_argument("--relocate requires OLD and NEW paths");
            result.insert(result.end(), {key, input[i + 1], input[i + 2]});
            i += 2;
            continue;
        }
        if (key == "--max-host-ram" || key == "--max-ram" || key == "-M" || key == "--leave-free" || key.ends_with("-bytes") ||
            key == "--compute-offer-ram" || key == "--compute-offer-vram" || key == "--compute-codec-ram") {
            if (++i == input.size()) throw std::invalid_argument(key + " requires a size");
            if (key == "--compute-codec-ram" && input[i] == "auto") {
                result.insert(result.end(), {key, "auto"}); continue;
            }
            uint64_t bytes = 0;
            if (!parse_host_ram_arg(input[i], bytes) ||
                ((key == "--max-host-ram" || key == "--max-ram" || key == "-M") && !bytes))
                throw std::invalid_argument(key + " requires a positive size such as 8G or 512M; group construction does not spill to min");
            result.insert(result.end(), {(key == "--max-host-ram" || key == "-M") ? "--max-ram" : key, std::to_string(bytes)});
            continue;
        }
        result.push_back(key);
        // Copy option values verbatim. A path or credential named like an
        // option must never be interpreted as a second option here.
        if (pos2gpu::group_option_values(key) && i + 1 < input.size()) result.push_back(input[++i]);
    }
    return result;
}

bool group_commands_available()
{
#ifdef XCHPLOT2_GROUP_COMMANDS
    return true;
#else
    std::cerr << "Error: this build does not include grouped construction and shared compute\n";
    return false;
#endif
}

} // namespace

extern "C" int xchplot2_main(int argc, char* argv[])
{
    pos2gpu::install_cancel_signal_handlers();
    if (argc > 1 && (std::string_view(argv[1]) == "--version" || std::string_view(argv[1]) == "-V")) {
        std::cout << "xchplot2 " << XCHPLOT2_VERSION << " (" << XCHPLOT2_REVISION << ")\n";
        return 0;
    }
    // Direct help must remain available when a saved config needs repair.
    // Inspect only the first option here; command parsers handle later options
    // without mistaking their values or literal filenames for help requests.
    if (argc > 1 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h" ||
        std::string_view(argv[1]) == "--help-all")) {
        print_usage(argv[0], std::cout); return 0;
    }
    if (argc > 2) {
        std::string_view const command = argv[1];
        if ((command == "plot" || command == "batch" || command == "bench" || command == "test" ||
             command == "verify" || command == "devices" || command == "parity-check") &&
            command_help(argv[0], command, argv[2])) return 0;
    }

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
        bool group = argc > 1 && pos2gpu::is_group_command(argv[1]);
        bool literal = false;
        unsigned values = 0;
        for (int i = 0; i < argc; ++i) {
            std::string a = argv[i];
            if (values) --values;
            else if (!literal) {
                if (a == "--config" && i + 1 < argc) {
                    config_path = argv[++i];
                    continue;
                }
                if (a == "--") literal = true;
                else if (group && i >= 2) values = pos2gpu::group_option_values(a);
            }
            argv_stripped.push_back(argv[i]);
            if (argv_stripped.size() == 2) group = pos2gpu::is_group_command(a);
        }
        strip_argc = static_cast<int>(argv_stripped.size());
    }
    if (config_path.empty()) {
#ifdef _WIN32
        if (char const* data = std::getenv("APPDATA")) {
            auto const default_path = std::filesystem::path(data) / "xchplot2/config.toml";
            if (std::filesystem::exists(default_path)) config_path = default_path.string();
        }
#else
        if (char const* home = std::getenv("HOME")) {
            std::string const default_path =
                std::string(home) + "/.config/xchplot2/config.toml";
            std::ifstream probe(default_path);
            if (probe) config_path = default_path;
        }
#endif
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
                std::string const flag =
                    (k.size() > 0 && k[0] == '-') ? k : ("--" + k);
                // Values such as devices="0", trials=1 and cpu-workers=off
                // are arguments. Coerce only options that actually take a bool.
                static constexpr std::string_view bool_flags[] = {
                    "--verbose", "-v", "--quiet", "-q", "--progress",
                    "--testnet", "-T", "--skip-existing", "--resume", "-r",
                    "--continue-on-error", "--cpu", "--shard-plot",
                    "--pipeline-plot", "--host-bounce", "--auto-spill",
                    "--no-auto-spill", "--keep", "--compute-only", "--full", "--json", "--plan-only",
                    "--extend", "--update", "--cancel-update", "--policy-only", "--upgrade-job", "--status", "--watch", "--inspect", "--live", "--revalidate", "--allow-missing", "--metadata-only", "--assess-jobs", "--calibrate", "--inventory"
                };
                std::string const bool_flag = flag.starts_with("--no-") ? "--" + flag.substr(5) : flag;
                auto const as_bool = std::find(std::begin(bool_flags), std::end(bool_flags), bool_flag)
                    != std::end(bool_flags) ? cfg.get_bool(name, k) : std::nullopt;
                if (as_bool) {
                    if (*as_bool) {
                        config_tokens.push_back(flag);
                    } else if (flag.size() > 2 && flag.substr(0, 2) == "--") {
                        // false → --no-XXX so the CLI can re-enable
                        // via --XXX (last-wins). Short flags don't
                        // have a canonical negation form; skip them.
                        config_tokens.push_back(flag.starts_with("--no-")
                            ? "--" + flag.substr(5) : "--no-" + flag.substr(2));
                    }
                    continue;
                }
                config_tokens.push_back(flag);
                config_tokens.push_back(v);
            }
        };
        emit_section("defaults");
        if (!subcmd.empty()) emit_section(subcmd);
        if (subcmd == "compute" && strip_argc >= 3)
            emit_section(subcmd + "." + argv_stripped[2]);
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

    if (mode == "--help" || mode == "-h" || mode == "--help-all") {
        print_usage(argv[0], std::cout);
        return 0;
    }
    if (pos2gpu::is_group_command(mode)) {
        if (!group_commands_available()) return 1;
#ifdef XCHPLOT2_GROUP_COMMANDS
        try {
            return pos2gpu::run_group_command(mode, group_arguments({expanded.begin() + 2, expanded.end()}));
        } catch (std::exception const& error) {
            std::cerr << '[' << mode << "] " << error.what() << '\n'; return 1;
        }
#endif
    }
    if (mode == "compute") {
        bool help = false;
        for (int i = 2; i < argc; ++i) {
            std::string_view const option = argv[i];
            if (option == "--help" || option == "-h" || option == "--help-all") { help = true; break; }
            if (option != "--verbose" && option != "-v" && option != "--no-verbose" &&
                option != "--quiet" && option != "-q" && option != "--no-quiet") break;
        }
        if (argc < 3 || help) {
            std::cout << "Usage: xchplot2 compute COMMAND [OPTIONS]\n"
                      << "  setup      Enroll, configure and start a helper in one command\n"
                      << "  init       Create a pool on the first machine\n"
                      << "  invite     Create a private invitation for another machine\n"
                      << "  join       Enroll this machine from an invitation\n"
                      << "  configure  Save resource, network and dedicated broker defaults\n"
                      << "  status     Show live workers, plotting job owners and tasks\n"
                      << "  pause / drain / resume  Control this machine's running worker\n"
                      << "  service    Manage a background worker or broker\n"
                      << "  members / revoke  Inspect or revoke pool membership\n"
                      << "  renew-request / renew-sign / renew-import  Rotate a node's credentials\n"
                      << "  policy-export / policy-import  Synchronize signed membership offline\n"
                      << "  broker  Schedule shared compute tasks for plotting jobs\n"
                      << "  worker  Share CPU/GPU capacity with plotting hosts\n"
                      << "Use xchplot2 compute COMMAND --help for connection and resource options.\n";
            return argc < 3 ? 1 : 0;
        }
        if (!group_commands_available()) return 1;
#ifdef XCHPLOT2_GROUP_COMMANDS
        try {
            std::vector<std::string> values(expanded.begin() + 2, expanded.end());
            for (size_t i = 1; i < values.size(); ++i) {
                if (values[i] != "--host-bytes" && values[i] != "--gpu-bytes" &&
                    values[i] != "--max-task-bytes" && values[i] != "--max-frame-bytes") continue;
                auto const option = values[i];
                uint64_t bytes = 0;
                if (++i == values.size() || !parse_host_ram_arg(values[i], bytes))
                    throw std::invalid_argument(option + " requires a byte count or a size such as 8G");
                values[i] = std::to_string(bytes);
            }
            std::vector<char*> arguments; for (auto& value : values) arguments.push_back(value.data());
            if (values.front() == "broker") return pos2gpu::run_compute_broker(int(arguments.size()), arguments.data());
            if (values.front() == "worker") return pos2gpu::run_compute_worker(int(arguments.size()), arguments.data());
            if (values.front() == "service") return pos2gpu::run_compute_service(int(arguments.size()), arguments.data());
            if (values.front() == "status" || values.front() == "pause" || values.front() == "drain" || values.front() == "resume")
                return pos2gpu::run_compute_status(int(arguments.size()), arguments.data());
            if (values.front() == "members" || values.front() == "revoke" || values.front().starts_with("renew-") || values.front().starts_with("policy-"))
                return pos2gpu::run_compute_membership(int(arguments.size()), arguments.data());
            return pos2gpu::run_compute_setup(int(arguments.size()), arguments.data());
        } catch (std::exception const& error) { std::cerr << "compute: " << error.what() << '\n'; return 1; }
#endif
        return 1;
    }

    if (mode == "devices") {
        bool verbose = false, quiet = false;
        for (int i = 2; i < argc; ++i) {
            std::string_view const option = argv[i];
            if (command_help(argv[0], mode, option)) return 0;
            if (option == "--verbose" || option == "-v") verbose = true;
            else if (option == "--no-verbose") verbose = false;
            else if (option == "--quiet" || option == "-q") quiet = true;
            else if (option == "--no-quiet") quiet = false;
            else { std::cerr << "Error: unknown argument: " << option << '\n'; return 1; }
        }
        if (quiet && verbose) {
            std::cerr << "Error: -q/--quiet and -v/--verbose are mutually exclusive\n"; return 1;
        }
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
        if (!quiet) std::printf("Visible devices (%zu GPU + %zu CPU node%s):\n",
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
            // A tiny iGPU is still targetable by explicit index, but auto
            // dispatch (--devices gpu/all, zero-config) skips it — say so.
            char const* auto_note = d.auto_dispatchable
                ? "" : "  (auto-dispatch skips this — explicit --devices only)";
            std::printf("  %-7s%-32s %5zu MiB VRAM", tag.c_str(), d.name.c_str(), vram_mb);
            if (verbose) std::printf("  backend=%s CUs=%u sort:%s physical=%s", d.backend.c_str(), d.cu_count, sort_hint,
                d.physical_identity.empty() ? "unknown" : d.physical_identity.c_str());
            std::printf("%s\n", auto_note);
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
                std::printf("  %-7s%-32s ? threads", tag.c_str(), name.c_str());
            } else {
                std::printf("  %-7s%-32s %zu threads", tag.c_str(), name.c_str(), node_threads);
            }
            if (verbose) std::printf("  backend=omp sort:SYCL");
            std::printf("\n");
        }
        if (devices.empty() && !quiet) {
            std::printf("\nNo GPU devices visible. CPU plotting remains available; use -v for driver diagnostics.\n");
        }
        if (devices.empty() && verbose) {
            std::printf("\nNo GPU devices visible to AdaptiveCpp / SYCL.\n"
#ifdef _WIN32
                        "Check your Intel, AMD, or NVIDIA graphics driver in Device Manager,\n"
                        "keep the complete release bin directory together, and check\n"
                        "ACPP_VISIBILITY_MASK. GPU toolkits are not required by the release.\n"
#else
                        "Check rocminfo / nvidia-smi, ACPP_VISIBILITY_MASK, and that the\n"
                        "relevant SYCL backend was built into AdaptiveCpp.\n"
#endif
                        "The CPU plotter is always available via `--devices cpu` or `--cpu`.\n");
        } else if (verbose) {
            std::printf("\nUse `--devices gpu0` (or bare `0`) for a specific GPU,\n"
                        "     `--devices gpu` for every plottable GPU (tiny iGPUs skipped),\n"
                        "     `--devices cpu` for every CPU node (opt-in),\n"
                        "     `--devices cpu0` for one CPU node,\n"
                        "     `--devices all` for every GPU + every CPU node,\n"
                        "  or any comma combination (e.g. `0,2,cpu`).\n"
                        "--devices names DEVICES; `--cpu-workers N` sets how many\n"
                        "plots run on each CPU node.\n");
        }
        if (!quiet && !verbose) std::printf("Use --devices with a listed ID, gpu, cpu, or all on plotting commands.\n");
        return 0;
    }

    if (mode == "bench") {
        int k = 28;
        int strength = 2;
        int measured = 10;
        int warmup = 1;
        bool keep = false;
        bool json = false;
        bool compute_only = false;
        bool testnet = false;
        std::string out_dir = ".";
        double target_size_tib = -1.0;
        pos2gpu::BatchOptions opts{};
        int progress_tri = -1;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (command_help(argv[0], mode, a)) return 0;
            auto need = [&](int more) -> bool {
                if (i + more >= argc) {
                    std::cerr << "Error: " << a << " requires " << more << " arg(s)\n";
                    return false;
                }
                return true;
            };
            if      ((a == "--k" || a == "-k") && need(1)) { if (!parse_integer_arg(argv[++i], k, a)) return 1; }
            else if ((a == "--strength" || a == "-s") && need(1)) { if (!parse_integer_arg(argv[++i], strength, a)) return 1; }
            else if ((a == "--num" || a == "-n") && need(1)) { if (!parse_integer_arg(argv[++i], measured, a)) return 1; }
            else if (a == "--warmup" && need(1)) { if (!parse_integer_arg(argv[++i], warmup, a)) return 1; }
            else if ((a == "--out" || a == "-o") && need(1)) out_dir = argv[++i];
            else if (a == "--json") json = true;
            else if (a == "--no-json") json = false;
            else if (a == "--keep") keep = true;
            else if (a == "--no-keep") keep = false;
            else if (a == "--compute-only") compute_only = true;
            else if (a == "--no-compute-only") compute_only = false;
            else if (a == "--quiet" || a == "-q") opts.quiet = true;
            else if (a == "--no-quiet") opts.quiet = false;
            else if (a == "--testnet" || a == "-T") {
                std::cerr << "Error: PoS2 1.0 removes testnet-specific plots; omit --testnet\n";
                return 1;
            }
            else if (a == "--no-testnet") testnet = false;
            else if (a == "--target-size" && need(1)) {
                std::string const value = argv[++i];
                char* end = nullptr;
                errno = 0;
                target_size_tib = std::strtod(value.c_str(), &end);
                if (end == value.c_str() || end != value.c_str() + value.size() ||
                    errno == ERANGE || !std::isfinite(target_size_tib) || target_size_tib <= 0.0) {
                    std::cerr << "Error: --target-size must be > 0 TiB\n";
                    return 1;
                }
            }
            else if (a == "-v" || a == "--verbose") opts.verbose = true;
            else if (a == "--no-verbose") opts.verbose = false;
            else if (a == "--progress") progress_tri = 1;
            else if (a == "--no-progress") progress_tri = 0;
            // CPU is opt-in. --cpu asks for it on every node without naming
            // --devices; it says nothing about HOW MANY, so it never disturbs an
            // explicit --cpu-workers N in either direction.
            else if (a == "--cpu") { opts.cpu_opt_in = true; if (opts.cpu_workers == 0) opts.cpu_workers = pos2gpu::kCpuWorkersAuto; }
            else if (a == "--no-cpu") { opts.cpu_opt_in = false; opts.cpu_workers = 0; }
            else if (a == "--cpu-workers" && need(1)) {
                if (!parse_cpu_workers_arg(argv[++i], opts)) return 1;
            }
            else if (a == "--shard-plot") opts.shard_plot = true;
            else if (a == "--no-shard-plot") opts.shard_plot = false;
            else if (a == "--pipeline-plot") opts.pipeline_plot = true;
            else if (a == "--no-pipeline-plot") opts.pipeline_plot = false;
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
            else if ((a == "--max-host-ram" || a == "-M") && need(1)) {
                std::uint64_t b = 0;
                if (!parse_host_ram_arg(argv[++i], b)) {
                    std::cerr << "Error: --max-host-ram expects a size like "
                                 "8G / 8192M / <bytes>, or 'min' (got '"
                              << argv[i] << "')\n";
                    return 1;
                }
                opts.has_max_host_ram = true;
                opts.max_host_ram     = b;
            }
            else if ((a == "--temp-dir" || a == "-t") && need(1)) {
                set_env("XCHPLOT2_TEMP_DIR", argv[++i]);
            }
            else if (a == "--auto-spill") opts.auto_host_ram_spill = true;
            else if (a == "--no-auto-spill") {
                opts.auto_host_ram_spill = false;
            }
            else if ((a == "--devices" || a == "-d") && need(1)) {
                if (!pos2gpu::parse_devices_arg(argv[++i], opts)) {
                    std::cerr << "Error: invalid --devices value\n";
                    return 1;
                }
            }
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_command_help(argv[0], mode, std::cerr);
                return 1;
            }
        }

        if (k < 18 || k > 28 || (k % 2) != 0) {
            std::cerr << "Error: -k must be an even integer in [18, 28]\n";
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
        if (opts.quiet && opts.verbose) {
            std::cerr << "Error: -q/--quiet and -v/--verbose are mutually exclusive\n"; return 1;
        }
        opts.progress = resolve_progress(progress_tri, opts.quiet);

        std::size_t const worker_count = pos2gpu::batch_worker_count(opts, k);
        // Per worker, not a global total: each worker has its own cold-start
        // plot to exclude. The queue size is still sized off the worker count.
        std::size_t const warmup_per_worker = static_cast<std::size_t>(warmup);
        if (worker_count == 0 || worker_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            measured > std::numeric_limits<int>::max() / static_cast<int>(worker_count) - warmup) {
            std::cerr << "Error: benchmark queue size exceeds the supported range\n";
            return 1;
        }
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
                    if (opts.verbose) std::fprintf(stderr,
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
                                    worker_count, opts.verbose, opts.quiet);
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
            if (k != 28) std::fprintf(stderr, "[bench] k=%d is test-only; use k=28 for farming measurements\n", k);

            if (opts.verbose && ran == 1) {
                auto const& w = e2e.stats.workers.front();
                std::fprintf(stderr,
                    "[bench]   per-plot spread: size %.3f GiB (min %.3f / max %.3f), "
                    "interval %.2f s (min %.2f / max %.2f, σ %.2f)\n",
                    e2e.gib_per_plot, e2e.size_gib_min, e2e.size_gib_max,
                    w.s_per_plot, w.interval_min, w.interval_max,
                    w.interval_stddev);
            } else if (opts.verbose) {
                // No aggregate interval spread here: the gap between successive
                // completions across workers is short when two finish together
                // and long across a slow worker's plot, so it measures the
                // interleaving, not the plotter. The per-worker σ above is the
                // real spread. Size spread is still meaningful.
                std::fprintf(stderr,
                    "[bench]   per-plot spread: size %.3f GiB (min %.3f / max %.3f)\n",
                    e2e.gib_per_plot, e2e.size_gib_min, e2e.size_gib_max);
            }

            print_bench_measurement("end-to-end", e2e, k, opts.verbose);

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
                // Pin the second pass to the pipeline the first one actually
                // ran. The tier is auto-picked per worker from free VRAM at the
                // moment that worker starts, so nothing guarantees two passes
                // of the same binary on the same idle card agree — and when
                // they don't, the wall-time difference is the tier's, not the
                // disk's. Measured on a 4090 squeezed to ~9 GiB free at k=28:
                // pass 1 plain, pass 2 compact, and the "disk overhead" line
                // then reported the plain->compact gap with a straight face.
                //
                // Only GPU workers carry a tier, and only a streaming tier is
                // forceable (there is no --tier pool), so this pins what it can
                // and the check after the pass catches whatever it could not.
                pos2gpu::BatchOptions compute_opts = opts;
                {
                    std::string tier;
                    bool uniform = true;
                    for (auto const& w : e2e.result.workers) {
                        if (w.pipeline.empty() || w.pipeline == "cpu") continue;
                        if (w.pipeline == "pool") { uniform = false; break; }
                        if (tier.empty())          tier = w.pipeline;
                        else if (tier != w.pipeline) { uniform = false; break; }
                    }
                    if (uniform && !tier.empty()) {
                        compute_opts.streaming_tier = tier;
                        compute_opts.per_device_tier.clear();
                        compute_opts.all_gpus_tier.clear();
                    }
                }
                compute = run_bench_pass(
                    k, strength, testnet, plot_count, compute_dir, compute_opts,
                    warmup_per_worker, keep);
                print_bench_measurement("compute-only", compute, k, opts.verbose,
                                        compute_label.c_str());

                // Did the pin hold? A pool/streaming split cannot be pinned at
                // all, and a forced tier can still be overridden by its own
                // floor check. Compare per device rather than as a set: two
                // workers swapping tiers between passes is still a difference.
                std::string mismatch;
                {
                    std::map<int, std::string> before;
                    for (auto const& w : e2e.result.workers) before[w.device_id] = w.pipeline;
                    for (auto const& w : compute.result.workers) {
                        auto it = before.find(w.device_id);
                        if (it == before.end() || it->second == w.pipeline) continue;
                        if (!mismatch.empty()) mismatch += ", ";
                        mismatch += "dev " + std::to_string(w.device_id) + " " +
                                    (it->second.empty() ? "?" : it->second) + "->" +
                                    (w.pipeline.empty() ? "?" : w.pipeline);
                    }
                }
                if (!mismatch.empty()) {
                    std::fprintf(stderr,
                        "[bench]   disk overhead: NOT COMPARABLE — the two passes "
                        "ran different pipelines (%s), so their wall-time "
                        "difference is that change, not the disk\n",
                        mismatch.c_str());
                } else if (e2e.s_per_plot > 0.0 && compute.s_per_plot > 0.0) {
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
                    std::fprintf(stderr, "[bench] kept %s\n", p.string().c_str());
                }
                // ...but only the ones sweep() actually left behind: the
                // compute-only set is gone if it lived in the tmpfs scratch.
                std::size_t dropped = 0;
                for (auto const& p : compute.paths) {
                    std::error_code ec;
                    if (std::filesystem::exists(p, ec)) {
                        std::fprintf(stderr, "[bench] kept %s\n", p.string().c_str());
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
                    "[bench] %s: %.3g TiB %s — estimated write time ~%s (%s rate)\n",
                    out_dir.c_str(),
                    fill_bytes / kTibBytes,
                    target_size_tib > 0.0 ? "target" : "free",
                    format_duration_dh(fill_s).c_str(), k == 28 ? "end-to-end" : "test-size");
            }
            if (json) {
                std::ostringstream report;
                report.precision(std::numeric_limits<double>::max_digits10);
                report << "{\"version\":";
#ifdef XCHPLOT2_VERSION
                write_json_string(report, XCHPLOT2_VERSION);
#else
                write_json_string(report, "unknown");
#endif
                report << ",\"k\":" << k << ",\"strength\":" << strength
                       << ",\"testnet\":" << (testnet ? "true" : "false")
                       << ",\"warmup_per_worker\":" << warmup
                       << ",\"num\":" << measured << ",\"queue_plots\":" << plot_count
                       << ",\"output_directory\":";
                write_json_string(report, out_dir);
                report << ",\"requested_tier\":";
                write_json_string(report, opts.streaming_tier.empty() ? "auto" : opts.streaming_tier);
                auto report_opts = opts;
                resolve_host_ram_env(report_opts);
                report << ",\"max_host_ram_bytes\":";
                if (report_opts.has_max_host_ram) report << report_opts.max_host_ram;
                else report << "null";
                report << ",\"auto_host_ram_spill\":" << (report_opts.auto_host_ram_spill ? "true" : "false")
                       << ",\"passes\":[";
                write_bench_json_pass(report, e2e, "end-to-end");
                if (compute_only) {
                    report << ',';
                    write_bench_json_pass(report, compute, compute_label);
                }
                report << "]}\n";
                std::cout << report.str();
            }
            return 0;
        } catch (std::exception const& e) {
            sweep();
            std::cerr << "[bench] FAILED: " << e.what() << "\n";
            return 2;
        }
    }

    if (mode == "batch") {
        if (argc < 3) { print_command_help(argv[0], mode, std::cerr); return 1; }
        std::string manifest;
        pos2gpu::BatchOptions opts{};
        std::vector<std::string> group_options;
        bool group_requested = false, literal = false;
        int progress_tri = -1;  // -1 auto (TTY), 0 off, 1 on
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (!literal && a == "--") { literal = true; continue; }
            if (!literal && command_help(argv[0], mode, a)) return 0;
            if (literal || !a.starts_with('-')) {
                if (!manifest.empty()) { std::cerr << "Error: unexpected argument: " << a << '\n'; return 1; }
                manifest = a; continue;
            }
            if (group_value_option(a)) {
                if (i + 1 == argc) { std::cerr << "Error: " << a << " requires a value\n"; return 1; }
                group_requested = true;
                group_options.insert(group_options.end(), {a, argv[++i]});
            }
            else if (a == "-v" || a == "--verbose")        opts.verbose = true;
            else if (a == "--no-verbose")                  opts.verbose = false;
            else if (a == "-q" || a == "--quiet")          opts.quiet = true;
            else if (a == "--no-quiet")                    opts.quiet = false;
            else if (a == "--progress")                    progress_tri = 1;
            else if (a == "--no-progress")                 progress_tri = 0;
            else if (a == "--skip-existing"
                  || a == "--resume" || a == "-r")         opts.skip_existing = true;
            else if (a == "--no-skip-existing"
                  || a == "--no-resume")                   opts.skip_existing = false;
            else if (a == "--continue-on-error")           opts.continue_on_error = true;
            else if (a == "--no-continue-on-error")        opts.continue_on_error = false;
            else if (a == "--cpu")                         { opts.cpu_opt_in = true; if (opts.cpu_workers == 0) opts.cpu_workers = pos2gpu::kCpuWorkersAuto; }
            else if (a == "--no-cpu") { opts.cpu_opt_in = false; opts.cpu_workers = 0; }
            else if (a == "--cpu-workers" && i + 1 < argc) {
                if (!parse_cpu_workers_arg(argv[++i], opts)) return 1;
            }
            else if (a == "--shard-plot")                  opts.shard_plot = true;
            else if (a == "--no-shard-plot")               opts.shard_plot = false;
            else if (a == "--pipeline-plot")               opts.pipeline_plot = true;
            else if (a == "--no-pipeline-plot")            opts.pipeline_plot = false;
            else if (a == "--pipeline-depth" && i + 1 < argc) {
                int d = 0;
                if (!parse_integer_arg(argv[++i], d, a)) return 1;
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
            else if (a == "--no-host-bounce") opts.prefer_peer_copy = true;
            else if (a == "--prefer-peer-copy")            { /* now the default, kept as a no-op alias */ }
            else if (a == "--tier" && i + 1 < argc) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal"
                    && t != "tiny" && t != "pinned" && t != "auto" && t != "pool") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', 'tiny', 'pinned', or 'auto' (got '"
                              << t << "')\n";
                    return 1;
                }
                opts.streaming_tier = (t == "auto") ? "" : t;
            }
            else if ((a == "--max-host-ram" || a == "-M") && i + 1 < argc) {
                std::uint64_t b = 0;
                if (!parse_host_ram_arg(argv[++i], b)) {
                    std::cerr << "Error: --max-host-ram expects a size like "
                                 "8G / 8192M / <bytes>, or 'min' (got '"
                              << argv[i] << "')\n";
                    return 1;
                }
                opts.has_max_host_ram = true;
                opts.max_host_ram     = b;
            }
            else if ((a == "--temp-dir" || a == "-t") && i + 1 < argc) {
                group_options.insert(group_options.end(), {a, argv[++i]});
                set_env("XCHPLOT2_TEMP_DIR", argv[i]);
            }
            else if (a == "--auto-spill") opts.auto_host_ram_spill = true;
            else if (a == "--no-auto-spill") {
                opts.auto_host_ram_spill = false;
            }
            else if ((a == "--devices" || a == "-d") && i + 1 < argc) {
                if (!pos2gpu::parse_devices_arg(argv[++i], opts)) {
                    std::cerr << "Error: --devices expects 'all', 'cpu', or a "
                                 "comma-separated list of device ids "
                                 "(got '" << argv[i] << "')\n";
                    return 1;
                }
            }
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_command_help(argv[0], mode, std::cerr);
                return 1;
            }
        }
        if (opts.quiet && opts.verbose) {
            std::cerr << "Error: -q/--quiet and -v/--verbose are mutually "
                         "exclusive\n";
            return 1;
        }
        if (manifest.empty()) { print_command_help(argv[0], mode, std::cerr); return 1; }
        opts.progress = resolve_progress(progress_tri, opts.quiet);
        try {
            auto entries = pos2gpu::parse_manifest(manifest);
            if (!opts.quiet) {
                std::cerr << "[batch] " << entries.size() << " plots queued\n";
            }
            resolve_host_ram_env(opts);
            bool const grouped = std::any_of(entries.begin(), entries.end(), [](auto const& e) { return e.grouped; });
            pos2gpu::BatchResult res;
            if (grouped) {
                if (!group_commands_available()) return 1;
#ifdef XCHPLOT2_GROUP_COMMANDS
                auto grouped_options = pos2gpu::group_batch_options(opts, group_arguments(group_options));
                grouped_options.command = "batch";
                res = pos2gpu::run_group_batch(entries, opts, std::move(grouped_options));
#endif
            } else {
                if (group_requested || opts.streaming_tier == "pool")
                    throw std::invalid_argument("group construction options require a grouped manifest created with plot --group-members");
                res = pos2gpu::run_batch(entries, opts);
            }
            if (!opts.quiet) print_run_summary("[batch]", res, opts.verbose);
            return batch_exit_code(res, entries.size());
        } catch (std::exception const& e) {
            std::cerr << "[batch] FAILED: " << e.what() << "\n";
            return 2;
        }
    }

    if (mode == "verify") {
        std::string plotfile;
        size_t trials = 100;
        bool full = false, verbose = false, quiet = false, literal = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (!literal && a == "--") { literal = true; continue; }
            if (!literal && command_help(argv[0], mode, a)) return 0;
            if (!literal && (a == "--trials" || a == "-n") && i + 1 < argc) {
                long v = 0;
                if (!parse_integer_arg(argv[++i], v, a)) return 1;
                if (v <= 0) {
                    std::cerr << "Error: --trials must be > 0\n";
                    return 1;
                }
                trials = static_cast<size_t>(v);
            } else if (!literal && a == "--full") full = true;
            else if (!literal && a == "--no-full") full = false;
            else if (!literal && (a == "--verbose" || a == "-v")) verbose = true;
            else if (!literal && a == "--no-verbose") verbose = false;
            else if (!literal && (a == "--quiet" || a == "-q")) quiet = true;
            else if (!literal && a == "--no-quiet") quiet = false;
            else if ((literal || !a.starts_with('-')) && plotfile.empty()) plotfile = a;
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_command_help(argv[0], mode, std::cerr);
                return 1;
            }
        }
        if (plotfile.empty()) { print_command_help(argv[0], mode, std::cerr); return 1; }
        if (quiet && verbose) {
            std::cerr << "Error: -q/--quiet and -v/--verbose are mutually exclusive\n"; return 1;
        }
        try {
            if (verbose) std::cerr << "[verify] " << plotfile << ": running " << trials
                      << " random challenges\n";
            auto res = pos2gpu::verify_plot_file(plotfile, trials, full);
            if (!quiet) std::cerr << "[verify] " << res.trials << " trials, "
                      << res.challenges_with_proof << " with >=1 quality chain, "
                      << res.proofs_found << " quality chains total\n";
            if (res.proofs_found == 0) {
                std::cerr << "[verify] FAIL: no proofs produced — plot is "
                             "likely corrupt\n";
                return 4;
            }
            if (full) std::cerr << "[verify] " << res.full_proofs_validated << " full proofs validated\n";
            std::cerr << (full ? "[verify] OK (full proof validation)\n"
                              : "[verify] OK (quality-chain sample; use --full to solve and validate)\n");
            return 0;
        } catch (std::exception const& e) {
            std::cerr << "[verify] FAILED: " << e.what() << "\n";
            return 2;
        }
    }

    if (mode == "parity-check") {
        std::string dir = "./build/tools/parity";
        bool verbose = false, quiet = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (command_help(argv[0], mode, a)) return 0;
            if ((a == "--dir" || a == "-d") && i + 1 < argc) {
                dir = argv[++i];
            } else if (a == "--verbose" || a == "-v") verbose = true;
            else if (a == "--no-verbose") verbose = false;
            else if (a == "--quiet" || a == "-q") quiet = true;
            else if (a == "--no-quiet") quiet = false;
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_command_help(argv[0], mode, std::cerr);
                return 1;
            }
        }
        if (quiet && verbose) {
            std::cerr << "Error: -q/--quiet and -v/--verbose are mutually exclusive\n"; return 1;
        }

        // Glob every *_parity AND *_test binary in `dir`. Same code path works
        // for both branches — main ships sycl_*_parity extras that cuda-only
        // doesn't, and the wildcard picks up whichever actually exists.
        //
        // `_test` is not decoration. The suffix used to be `_parity` alone,
        // which quietly excluded every host test — the whole host-RAM spill
        // suite among them. This command is what a user runs to answer "is
        // this build good?", and it answered without executing a line of the
        // spill path. A test runner that silently skips tests is worse than no
        // test runner, because it is believed. CI covers these via
        // scripts/test/host-tests.sh; this is the interactive half.
        auto has_suffix = [](std::string const& s, char const* suf) {
            size_t const n = std::strlen(suf);
            return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
        };
        std::vector<std::filesystem::path> tests;
        std::error_code ec;
        if (std::filesystem::is_directory(dir, ec)) {
            for (auto const& entry :
                 std::filesystem::directory_iterator(dir, ec))
            {
                auto name = entry.path().filename().string();
#ifdef _WIN32
                if (entry.path().extension() != ".exe") continue;
                name = entry.path().stem().string();
#endif
                if ((has_suffix(name, "_parity") || has_suffix(name, "_test"))
                    && entry.is_regular_file(ec)) {
                    tests.push_back(entry.path());
                }
            }
        }
        if (tests.empty()) {
            std::cerr << "No `*_parity` / `*_test` binaries found under " << dir << ".\n"
                         "Build them first:\n"
                         "  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release\n"
                         "  cmake --build build --parallel\n"
                         "Then re-run from the repo root, or pass --dir <path>.\n";
            return 2;
        }
        std::sort(tests.begin(), tests.end());

        int pass = 0, fail = 0;
        if (!quiet) std::cerr << "==> parity + host tests (" << tests.size() << " found in "
                  << dir << ")\n";
        for (auto const& test : tests) {
            auto const name = test.filename().string();
            // tmpfile creates a private, exclusively opened log; paths never
            // pass through a shell, even when they contain spaces/metacharacters.
            auto close_log = [](std::FILE* f) { std::fclose(f); };
            std::unique_ptr<std::FILE, decltype(close_log)> log(std::tmpfile(), close_log);
            if (!log) { std::perror("parity log"); return 1; }
            auto const t0 = std::chrono::steady_clock::now();
            int const rc = run_parity_test(test.string(), log.get());
            auto const ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count();
            if (rc == 0) {
                if (verbose) std::fprintf(stderr, "  PASS  %-32s  (%.1f ms)\n", name.c_str(), ms);
                else if (!quiet) std::fprintf(stderr, "  PASS  %s\n", name.c_str());
                if (verbose) {
                    std::rewind(log.get());
                    char output[4096];
                    while (auto n = std::fread(output, 1, sizeof(output), log.get()))
                        std::fwrite(output, 1, n, stderr);
                }
                ++pass;
            } else {
                std::fprintf(stderr,
                             "  FAIL  %-32s  (exit %d)\n", name.c_str(), rc);
                std::rewind(log.get());
                char output[4096];
                while (auto n = std::fread(output, 1, sizeof(output), log.get()))
                    std::fwrite(output, 1, n, stderr);
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
        int group_members = 1;
        bool group_requested = false;
        std::vector<std::string> group_options;
        int plot_index_base = 0;
        int meta_group = 0;
        bool testnet = false;
        bool verbose = false;
        bool skip_existing = false;
        bool continue_on_error = false;
        std::string out_dir = ".";
        std::string farmer_pk_hex, pool_pk_hex, pool_ph_hex, pool_addr;
        std::string seed_hex;
        std::string manifest_path;
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
        bool          plot_has_max_host_ram = false;
        std::uint64_t plot_max_host_ram     = 0;
        bool          plot_no_auto_spill    = false;
        std::vector<pos2gpu::PipelineStageTier> plot_pipeline_tiers;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (command_help(argv[0], mode, a)) return 0;
            auto need = [&](int more) -> bool {
                if (i + more >= argc) {
                    std::cerr << "Error: " << a << " requires " << more << " arg(s)\n";
                    return false;
                }
                return true;
            };
            if (a == "--group-members" && need(1)) {
                if (!parse_integer_arg(argv[++i], group_members, a)) return 1;
                group_requested = true;
            }
            else if (group_value_option(a) && need(1)) {
                if (a != "--compute" && a != "--compute-directory") group_requested = true;
                group_options.insert(group_options.end(), {a, argv[++i]});
            }
            else if ((a == "--k"          || a == "-k") && need(1)) { if (!parse_integer_arg(argv[++i], k, a)) return 1; }
            else if ((a == "--num"        || a == "-n") && need(1)) { if (!parse_integer_arg(argv[++i], num, a)) return 1; }
            else if ((a == "--strength"   || a == "-s") && need(1)) { if (!parse_integer_arg(argv[++i], strength, a)) return 1; }
            else if ((a == "--out"        || a == "-o") && need(1)) out_dir  = argv[++i];
            else if ((a == "--farmer-pk"  || a == "-f") && need(1)) farmer_pk_hex = argv[++i];
            else if ((a == "--pool-pk"    || a == "-p") && need(1)) pool_pk_hex   = argv[++i];
            else if  (a == "--pool-ph"                  && need(1)) pool_ph_hex   = argv[++i];
            else if ((a == "--pool-contract-address" || a == "-c") && need(1)) pool_addr = argv[++i];
            else if ((a == "--plot-index" || a == "-i") && need(1)) { if (!parse_integer_arg(argv[++i], plot_index_base, a)) return 1; }
            else if ((a == "--meta-group" || a == "-g") && need(1)) { if (!parse_integer_arg(argv[++i], meta_group, a)) return 1; }
            else if ((a == "--seed"       || a == "-S") && need(1)) seed_hex        = argv[++i];
            else if  (a == "--manifest" && need(1)) manifest_path = argv[++i];
            else if (a == "--testnet" || a == "-T") {
                std::cerr << "Error: PoS2 1.0 removes testnet-specific plots; omit --testnet\n";
                return 1;
            }
            else if  (a == "--no-testnet")              testnet = false;
            else if  (a == "-v" || a == "--verbose")    verbose = true;
            else if  (a == "--no-verbose")              verbose = false;
            else if  (a == "-q" || a == "--quiet")      plot_quiet = true;
            else if  (a == "--no-quiet")                plot_quiet = false;
            else if  (a == "--progress")                plot_progress_tri = 1;
            else if  (a == "--no-progress")             plot_progress_tri = 0;
            else if  (a == "--skip-existing"
                   || a == "--resume" || a == "-r")     skip_existing = true;
            else if  (a == "--no-skip-existing"
                   || a == "--no-resume")               skip_existing = false;
            else if  (a == "--continue-on-error")       continue_on_error = true;
            else if  (a == "--no-continue-on-error")    continue_on_error = false;
            else if  (a == "--cpu") { plot_cpu_opt_in = true; if (plot_cpu_workers == 0) plot_cpu_workers = pos2gpu::kCpuWorkersAuto; }
            else if (a == "--no-cpu") { plot_cpu_opt_in = false; plot_cpu_workers = 0; }
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
                int d = 0;
                if (!parse_integer_arg(argv[++i], d, a)) return 1;
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
            else if (a == "--no-host-bounce") plot_prefer_peer_copy = true;
            else if  (a == "--prefer-peer-copy")        { /* now the default, kept as a no-op alias */ }
            else if  (a == "--tier" && need(1)) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal"
                    && t != "tiny" && t != "pinned" && t != "auto" && t != "pool") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', 'tiny', 'pinned', or 'auto' (got '"
                              << t << "')\n";
                    return 1;
                }
                plot_streaming_tier = (t == "auto") ? "" : t;
            }
            else if ((a == "--max-host-ram" || a == "-M") && need(1)) {
                std::uint64_t b = 0;
                if (!parse_host_ram_arg(argv[++i], b)) {
                    std::cerr << "Error: --max-host-ram expects a size like "
                                 "8G / 8192M / <bytes>, or 'min' (got '"
                              << argv[i] << "')\n";
                    return 1;
                }
                plot_has_max_host_ram = true;
                plot_max_host_ram     = b;
            }
            else if ((a == "--temp-dir" || a == "-t") && need(1)) {
                group_options.insert(group_options.end(), {a, argv[++i]});
                set_env("XCHPLOT2_TEMP_DIR", argv[i]);
            }
            else if  (a == "--auto-spill") plot_no_auto_spill = false;
            else if  (a == "--no-auto-spill") {
                plot_no_auto_spill = true;
            }
            else if ((a == "--devices" || a == "-d") && need(1)) {
                pos2gpu::BatchOptions tmp;
                if (!pos2gpu::parse_devices_arg(argv[++i], tmp)) {
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
                print_command_help(argv[0], mode, std::cerr);
                return 1;
            }
        }

#ifdef XCHPLOT2_GROUP_COMMANDS
        try {
            std::string compute = "auto"; std::filesystem::path compute_directory;
            for (size_t i = 0; i + 1 < group_options.size(); i += 2) {
                if (group_options[i] == "--compute") compute = group_options[i + 1];
                if (group_options[i] == "--compute-directory")
                    compute_directory = std::filesystem::path(reinterpret_cast<char8_t const*>(group_options[i + 1].c_str()));
            }
            if (compute != "auto" && compute != "off") throw std::invalid_argument("--compute requires auto or off");
            if (!group_requested && compute != "off") group_requested = pos2gpu::saved_compute_available(compute_directory);
        } catch (std::exception const& error) { std::cerr << "compute: " << error.what() << '\n'; return 1; }
#endif
        if (group_members < 1 || group_members > 65535) {
            std::cerr << "Error: --group-members must be in [1, 65535]; use fill for automatic group sizing\n";
            return 1;
        }
        if (group_requested && !group_commands_available()) return 1;
        if (!group_requested && plot_streaming_tier == "pool") {
            std::cerr << "Error: an explicit pool tier requires --group-members; ordinary plotting selects it automatically\n";
            return 1;
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
        if (plot_index_base != 0) {
            std::cerr << "Error: ordinary groups start at --plot-index 0; use --group-members to choose membership\n";
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
            if (plot_no_auto_spill) opts.auto_host_ram_spill = false;
            opts.has_max_host_ram  = plot_has_max_host_ram;
            opts.max_host_ram      = plot_max_host_ram;
            opts.quiet             = plot_quiet;
            opts.progress          = resolve_progress(plot_progress_tri, plot_quiet);
            resolve_host_ram_env(opts);
            pos2gpu::GroupOptions parsed_group_options;
            if (group_requested) {
                group_options = group_arguments(group_options);
#ifdef XCHPLOT2_GROUP_COMMANDS
                parsed_group_options = pos2gpu::group_batch_options(opts, group_options);
                if (have_base_seed && !parsed_group_options.pool_broker.empty())
                    base_seed = pos2gpu::compute_owner_seed(base_seed, parsed_group_options.pool_owner);
#endif
            }
            out_dir = std::filesystem::absolute(out_dir).lexically_normal().string();
#ifdef XCHPLOT2_GROUP_COMMANDS
            pos2gpu::check_group_output_directory(out_dir);
            if (!manifest_path.empty())
                pos2gpu::check_group_output_directory(std::filesystem::absolute(manifest_path).lexically_normal().parent_path());
#endif
            std::filesystem::create_directories(out_dir);
            std::filesystem::path job_path = manifest_path;
            std::vector<pos2gpu::BatchEntry> entries;
            auto memo_prefix = pool_key;
            memo_prefix.insert(memo_prefix.end(), farmer_pk.begin(), farmer_pk.end());
            auto matches_request = [&](std::vector<pos2gpu::BatchEntry> const& job) {
                if (job.size() != static_cast<std::size_t>(num)) return false;
                for (std::size_t i = 0; i < job.size(); ++i) {
                    auto const& e = job[i];
                    if (e.k != k || e.strength != strength || e.testnet != testnet ||
                        e.raw || e.plot_index != 0 || e.grouped != group_requested || e.group_members != group_members ||
                        e.meta_group != meta_group ||
                        std::filesystem::weakly_canonical(e.out_dir) !=
                            std::filesystem::weakly_canonical(out_dir) ||
                        e.memo.size() != memo_prefix.size() + 32 ||
                        !std::equal(memo_prefix.begin(), memo_prefix.end(), e.memo.begin()))
                        return false;
                }
                return true;
            };
            // A seed already reconstructs the job exactly. Random jobs must
            // recover their saved identities before any new keys are made.
            if (skip_existing && !have_base_seed) {
                if (!job_path.empty() && std::filesystem::exists(job_path)) {
                    entries = pos2gpu::parse_manifest(job_path.string());
                    if (!matches_request(entries))
                        throw std::invalid_argument("manifest does not match the requested keys, plot parameters, or output directory");
                } else if (job_path.empty()) {
                    bool found_job = false;
                    for (auto const& file : std::filesystem::directory_iterator(out_dir)) {
                        auto const name = file.path().filename().string();
                        if (!name.starts_with("xchplot2-job-") || !name.ends_with(".tsv")) continue;
                        found_job = true;
                        auto candidate = pos2gpu::parse_manifest(file.path().string());
                        if (!matches_request(candidate)) continue;
                        if (!entries.empty())
                            throw std::invalid_argument("multiple matching jobs; select --manifest FILE or run batch FILE --resume");
                        entries = std::move(candidate);
                        job_path = file.path();
                    }
                    if (found_job && entries.empty())
                        throw std::invalid_argument("no saved job matches this request; use the original plot parameters or start a new job without --resume");
                }
            }
            bool const recovered = !entries.empty();
            entries.reserve(static_cast<size_t>(num));
            for (int i = 0; !recovered && i < num; ++i) {
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
                    pos2gpu::read_random_bytes(seed, sizeof(seed));
                }

                uint8_t group_id[32];
                std::vector<uint8_t> memo(128);
                size_t memo_len = memo.size();
                int rc = pos2_keygen_derive_group(
                    seed, sizeof(seed), farmer_pk.data(), pool_key.data(), pool_kind,
                    static_cast<uint8_t>(strength), group_id, memo.data(), &memo_len);
                if (rc != POS2_OK) {
                    std::cerr << "Error: pos2_keygen_derive_group failed (rc=" << rc << ")\n";
                    return 2;
                }
                memo.resize(memo_len);

                pos2gpu::BatchEntry e;
                e.k          = k;
                e.strength   = strength;
                e.plot_index = 0;
                e.grouped = group_requested;
                e.group_members = group_members;
                e.meta_group = meta_group;
                e.testnet    = testnet;
                std::copy(group_id, group_id + 32, e.group_id.begin());
                e.plot_id = pos2gpu::plot_id_for_group(e.group_id, 0, uint8_t(meta_group));
                e.memo       = std::move(memo);
                e.out_dir    = out_dir;
                e.out_name   = plot_id_to_filename(k, e.plot_id);
                entries.push_back(std::move(e));

                if (verbose) {
                    std::cerr << "[plot] prepared " << (i + 1) << "/" << num
                              << " " << entries.back().out_name << "\n";
                }
            }
            if (job_path.empty()) {
                job_path = std::filesystem::path(out_dir) /
                    ("xchplot2-job-" + bytes_to_hex(entries.front().plot_id)
                     + "-k" + std::to_string(k) + "-n" + std::to_string(num)
                     + (group_requested ? "-m" + std::to_string(group_members) : "")
                     + (testnet ? "-testnet" : "") + ".tsv");
            }
            pos2gpu::write_manifest(job_path.string(), entries);
            if (!plot_quiet)
                std::cerr << "[plot] " << (recovered ? "resuming" : "saved")
                          << " job: " << job_path.string()
                          << " (reuse with batch FILE --resume)\n";

            std::mutex output_mutex;
            opts.on_plot_ready = [&](pos2gpu::BatchEntry const& entry) {
                std::lock_guard lock(output_mutex);
                std::cout << (std::filesystem::path(entry.out_dir) / entry.out_name).string()
                          << '\n' << std::flush;
            };
            pos2gpu::BatchResult res;
#ifdef XCHPLOT2_GROUP_COMMANDS
            if (group_requested) res = pos2gpu::run_group_batch(entries, opts, std::move(parsed_group_options));
            else
#endif
                res = pos2gpu::run_batch(entries, opts);
            if (!plot_quiet) print_run_summary("[plot]", res, verbose);
            return batch_exit_code(res, entries.size());
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
    local cur prev words cword path
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    local subcmds="plan plot fill status pause drain resume retire grow compute batch bench test devices verify parity-check completions"
    local tiers="plain compact minimal tiny pinned auto"
    local devices_tokens="all gpu cpu 0 1 2 3"
    if [[ "${COMP_WORDS[1]}" == plan || "${COMP_WORDS[1]}" == fill || "${COMP_WORDS[1]}" == status || "${COMP_WORDS[1]}" == pause || "${COMP_WORDS[1]}" == drain || "${COMP_WORDS[1]}" == resume || "${COMP_WORDS[1]}" == retire ]]; then
        local i option paths="" values=0 literal=0
        for ((i=2; i<COMP_CWORD; ++i)); do
            if [[ -n "$paths" && "${COMP_WORDS[i]}" != -* ]]; then
                continue
            elif ((values)); then
                ((--values))
            else
                paths=""
                option="${COMP_WORDS[i]}"
                case "${COMP_WORDS[i]}" in
                    --) literal=1; break ;;
                    --relocate) values=2 ;;
                    --temp-dirs|--exclude-destination|--farm-config) paths="$option" ;;
                    --replot|--no-replot|--discover-farm|--no-discover-farm) ;;
                    --policy-only|--no-policy-only|--upgrade-job|--no-upgrade-job) ;;
                    -r|--resume|--no-resume|--plan-only|--no-plan-only|--metadata-only|--no-metadata-only|--assess-jobs|--no-assess-jobs|--inventory|--no-inventory|--status|--no-status|--watch|--no-watch|--inspect|--no-inspect|--live|--no-live|--revalidate|--no-revalidate|--calibrate|--no-calibrate|--extend|--no-extend|--update|--no-update|--cancel-update|--no-cancel-update|--allow-missing|--no-allow-missing|--json|--no-json|-q|--quiet|--no-quiet|-v|--verbose|--no-verbose|--progress|--no-progress|-h|--help|--help-all) ;;
                    -*) values=1 ;;
                esac
            fi
        done
        if ((literal)); then
            while IFS= read -r path; do COMPREPLY+=("$path"); done < <(compgen -f -- "$cur")
            return 0
        fi
        if [[ -n "$paths" && "$cur" != -* ]]; then
            local kind=-d
            [[ "$paths" == --farm-config ]] && kind=-f
            while IFS= read -r path; do COMPREPLY+=("$path"); done < <(compgen "$kind" -- "$cur")
            return 0
        fi
        if ((values)) && [[ "$option" == --relocate ]]; then
            while IFS= read -r path; do COMPREPLY+=("$path"); done < <(compgen -d -- "$cur")
            return 0
        fi
    fi
    case "${prev}" in
        --tier)            COMPREPLY=( $(compgen -W "${tiers}" -- "$cur") ); return 0 ;;
        -d|--devices)      COMPREPLY=( $(compgen -W "${devices_tokens}" -- "$cur") ); return 0 ;;
        --compute)         COMPREPLY=( $(compgen -W "auto off" -- "$cur") ); return 0 ;;
        --compute-local|--discovery|--io-overlap|--cache-release) COMPREPLY=( $(compgen -W "on off" -- "$cur") ); return 0 ;;
        --backend|--compute-offer-backend) COMPREPLY=( $(compgen -W "auto cpu gpu" -- "$cur") ); return 0 ;;
        --construction)    COMPREPLY=( $(compgen -W "auto incremental staged merged external" -- "$cur") ); return 0 ;;
        --alternative)     COMPREPLY=( $(compgen -W "memory eligible-groups deadline cpu-cores lookup-workers capacity" -- "$cur") ); return 0 ;;
        --replot-verification) COMPREPLY=( $(compgen -W "structural sampled strict" -- "$cur") ); return 0 ;;
        --replot-confirmation) COMPREPLY=( $(compgen -W "directory file" -- "$cur") ); return 0 ;;
        -o|--out|-t|--temp-dir|--relocate|--discover-drives|--mounts-under|--destination|--chia-root|--directory|--compute-directory|--replot-source-root|--replot-exclude-root)
            while IFS= read -r path; do COMPREPLY+=("$path"); done < <(compgen -d -- "$cur")
            return 0 ;;
        -M|--max-host-ram|--leave-free|--headroom-bytes|-j|--workers|--farm-adapter-timeout-ms|--replot-proof-first-groups|--replot-proof-every-groups) return 0 ;;
        --harvester)
            COMPREPLY=( $(compgen -W "local" -- "$cur") )
            while IFS= read -r path; do COMPREPLY+=("$path"); done < <(compgen -f -- "$cur")
            return 0 ;;
        --farm-adapter) COMPREPLY=( $(compgen -W "chia gigahorse drchia" -- "$cur") ); return 0 ;;
        --manifest|--save-plan|-F|--from-plan|--replot-inventory|--save-profile|--chia-executable|--paths-from|--paths0-from|--config|--out-name|--dir|--ca|--cert|--key|--tls-ca|--tls-cert|--tls-key|--compute-ca|--compute-cert|--compute-key)
            while IFS= read -r path; do COMPREPLY+=("$path"); done < <(compgen -f -- "$cur")
            return 0 ;;
        -f|--farmer-pk|-p|--pool-pk|--pool-ph|-c|--seed|-S) return 0 ;;
        compute)           COMPREPLY=( $(compgen -W "setup init invite join configure members revoke renew-request renew-sign renew-import policy-export policy-import status pause drain resume service broker worker" -- "$cur") ); return 0 ;;
        service)           if [[ "${COMP_WORDS[1]}" == compute ]]; then COMPREPLY=( $(compgen -W "install start stop restart status remove print" -- "$cur") ); return 0; fi ;;
        --role)            if [[ "${COMP_WORDS[2]}" == setup ]]; then COMPREPLY=( $(compgen -W "plotter worker" -- "$cur") ); else COMPREPLY=( $(compgen -W "worker broker" -- "$cur") ); fi; return 0 ;;
        completions)       COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") ); return 0 ;;
    esac
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${subcmds} --version -V --help -h" -- "$cur") )
        return 0
    fi
    if [[ "${COMP_WORDS[1]}" == verify ]]; then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=( $(compgen -W "--trials -n --full --no-full -v --verbose --no-verbose -q --quiet --no-quiet --config -h --help --help-all" -- "$cur") )
        elif [[ "$prev" != --trials && "$prev" != -n ]]; then
            COMPREPLY=( $(compgen -f -- "$cur") )
        fi
        return 0
    fi
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "-v --verbose --no-verbose -q --quiet --no-quiet --progress --no-progress --cpu --cpu-workers --tier --devices --shard-plot --pipeline-plot --host-bounce --skip-existing --resume --manifest --config --strategy --pipeline-depth --pipeline-stage-tiers --max-host-ram --temp-dir --auto-spill --no-auto-spill --continue-on-error --warmup --keep --compute-only --target-size --json --trials --full --no-full --k --num --strength --plot-index --meta-group --out --farmer-pk --pool-pk --pool-ph --pool-contract-address --seed --raw --memo --out-name --profile --gpu-all --gpu-t1 --gpu-t2 --gpu-t3 --dir --compute --compute-directory --compute-local --directory --name --interface --discovery --role --worker --limit --max-brokers --ca --cert --key --generation-threads --max-task-bytes --max-window-chunks --max-batch-members --task-timeout-ms --tls-timeout-ms --group-members --plan-only --alternative --upgrade-job --no-upgrade-job --policy-only --no-policy-only --metadata-only --no-metadata-only --assess-jobs --no-assess-jobs --inventory --no-inventory --status --no-status --watch --no-watch --inspect --no-inspect --live --no-live --destination --revalidate --no-revalidate --replot --no-replot --replot-inventory --replot-confirmation --discover-farm --no-discover-farm --farm-config --exclude-destination --temp-dirs --replot-verification --replot-proof-first-groups --replot-proof-every-groups --replot-source-root --replot-exclude-root --replot-max-source-files --replot-max-source-bytes --reclaim-before-bytes --farm-adapter --farm-adapter-timeout-ms --calibrate --no-calibrate --calibration-samples --calibration-timeout-ms --calibration-memory-bytes --save-plan -F --from-plan --save-profile --resource-setting --harvester --chia-root --chia-executable --help-all --paths-from --paths0-from --discover-drives --mounts-under --extend --update --cancel-update --allow-missing --relocate --construction --backend --device --max-ram --workers --batch-members --max-members --max-groups --max-total-members --capacity-bytes --leave-free --headroom-bytes --destination-temp --temp-capacity-bytes --member-storage --window-bytes --checkpoint-bytes --io-overlap --generation-overlap --cache-release --farming-reader --farming-objective --farming-cores --farming-lookup-workers --farming-memory-bytes --farming-index-cache-bytes --farming-filter-bits --farming-future-filter-bits --farming-deadline-ms --farming-chain-us-per-member --farming-index-ns-per-entry --farming-cpu-work-weight --farming-io-work-weight --farming-max-eligible-groups --compute-broker --compute-owner-id --compute-ca --compute-cert --compute-key --compute-inflight --compute-codec-ram --compute-timeout-ms --compute-socket-timeout-ms --compute-frame-bytes --compute-listen --compute-advertise --compute-offer-ram --compute-offer-cpus --compute-offer-vram --compute-offer-slots --compute-offer-backend --listen --advertise --broker --tls-ca --tls-cert --tls-key --host-bytes --cpus --gpu-bytes --slots --timeout-ms --lease-ms --worker-ms --control-threads --max-connections --idle-ms --owner-share --max-workers --max-owners --max-tasks --max-frame-bytes -k -n -s -f -p -c -o -i -g -S -d -t -M -r -j --help" -- "$cur") )
        return 0
    fi
    if [[ "${COMP_WORDS[1]}" == plan || "${COMP_WORDS[1]}" == fill || "${COMP_WORDS[1]}" == status || "${COMP_WORDS[1]}" == pause || "${COMP_WORDS[1]}" == drain || "${COMP_WORDS[1]}" == resume || "${COMP_WORDS[1]}" == retire ]]; then
        while IFS= read -r path; do COMPREPLY+=("$path"); done < <(compgen -f -- "$cur")
    fi
}
complete -F _xchplot2 xchplot2
)";
        } else if (shell == "zsh") {
            std::cout << R"(#compdef xchplot2
_xchplot2() {
    local -a subcmds
    local state
    local positional='*:: :->args'
    subcmds=(plan:"Preview farming layout" fill:"Fill destination drives" status:"Inspect farm progress" pause:"Pause at a safe checkpoint" drain:"Finish started files, then stop" resume:"Continue paused or drained work" retire:"Detach destinations, keep plots" grow:"Extend an existing group" compute:"Share compute" batch:"Run a manifest of plots" plot:"Single-plot farmable mode" bench:"Measure plotting throughput" test:"Single test plot" devices:"List available GPU/CPU" verify:"Verify a plot file" parity-check:"Run parity tests" completions:"Emit shell completion script")
    if (( CURRENT == 2 )); then
        _arguments '-V[Print version and source revision]' '--version[Print version and source revision]' \
            '-h[Show help]' '--help[Show help]' '1:subcommand:->command'
        if [[ "$state" == command ]]; then _describe 'subcommand' subcmds; fi
        return
    fi
    if [[ "${words[2]}" == compute && $CURRENT -eq 3 ]]; then
        _values 'compute command' setup init invite join configure members revoke renew-request renew-sign renew-import policy-export policy-import status pause drain resume service broker worker
        return
    fi
    if [[ "${words[2]}" == compute && "${words[3]}" == service && $CURRENT -eq 4 ]]; then
        _values 'service command' install start stop restart status remove print
        return
    fi
    if [[ "${words[2]}" == verify ]]; then
        _arguments '--trials[Random challenges]:count:' '-n[Random challenges]:count:' \
            '--full[Validate full proofs]' '--no-full[Quality-chain sample]' \
            '-v[Additional verification details]' '--verbose[Additional verification details]' '--no-verbose[Concise output]' \
            '-q[Suppress informational output]' '--quiet[Suppress informational output]' '--no-quiet[Show informational output]' \
            '-h[Show help]' '--help[Show help]' '--help-all[Show the complete reference]' \
            '--config[Configuration file]:file:_files' '2:plot file:_files'
        return
    fi
    if [[ "${words[2]}" == plan || "${words[2]}" == fill || "${words[2]}" == status || "${words[2]}" == pause || "${words[2]}" == drain || "${words[2]}" == resume || "${words[2]}" == retire ]]; then
        positional='*:destination directory or saved plan:_files'
    fi
    _arguments \
        '--compute[Shared compute auto or off]:value:' \
        '--compute-directory[Saved compute setup directory]:value:' \
        '--compute-local[Offer local compute on or off]:value:' \
        '--directory[Saved compute setup directory]:value:' \
        '--role[Compute setup or background service role]:role:(plotter worker broker)' \
        '--worker[Worker status or control endpoint]:address:' \
        '--limit[Rows per broker status section]:count:' \
        '--max-brokers[Worker broker registration limit]:count:' \
        '--name[Machine display name]:value:' \
        '--interface[Sharing network interface]:value:' \
        '--discovery[LAN discovery on or off]:value:' \
        '--ca[Trusted worker CA]:value:' \
        '--cert[Worker certificate]:value:' \
        '--key[Worker private key]:value:' \
        '--generation-threads[CPU threads per generation task]:value:' \
        '--max-task-bytes[Codec and generation scratch limit in bytes]:value:' \
        '--max-window-chunks[Maximum chunks per codec task]:value:' \
        '--max-batch-members[Maximum members per generation task]:value:' \
        '--task-timeout-ms[Task timeout in milliseconds]:value:' \
        '--tls-timeout-ms[TLS operation timeout in milliseconds]:value:' \
        '--group-members[Members per file]:value:' \
        '--plan-only[Preview without writing]' \
        '--metadata-only[Inspect saved plan metadata without live checks]' \
        '--no-metadata-only[Restore live plan checks]' \
        '--assess-jobs[Check saved plans together against one harvester budget]' \
        '--inventory[Inspect explicit existing farm roots]' \
        '--no-inventory[Restore ordinary plan mode]' \
        '--replot[Replace eligible old plots; plan previews, fill executes]' \
        '--no-replot[Use ordinary plotting]' \
        '--replot-inventory[Optional saved inventory]:file:_files' \
        '--replot-confirmation[Farming acknowledgement scope]:scope:(directory file)' \
        '--discover-farm[Include configured farm directories]' \
        '--no-discover-farm[Use supplied destinations]' \
        '*--farm-config[Additional reader roots or configuration files]:path:_files' \
        '*--exclude-destination[Directories protected from output, scratch and retirement]:directory:_files -/' \
        '*--temp-dirs[Temporary directories, separated by spaces]:directory:_files -/' \
        '--replot-verification[Structural checks, plus proof samples for selected or all groups]:mode:(structural sampled strict)' \
        '--replot-proof-first-groups[Proof-check the first N groups with the sampled policy]:count:' \
        '--replot-proof-every-groups[Proof-check every Nth group across destinations with the sampled policy; 0 disables]:count:' \
        '--status[Inspect saved farm progress]' \
        '--watch[Refresh status until interrupted]' '--no-watch[Print status once]' \
        '--inspect[Scan outputs and check recovery checkpoints]' '--no-inspect[Read saved status]' \
        '--live[Query shared compute and migration service readiness]' '--no-live[Use saved status only]' \
        '*--destination[Saved destination to control]:directory:_files -/' \
        '--policy-only[Save the control setting without starting an idle job]' \
        '--revalidate[Review updated migration resources or verification policy]' \
        '*--replot-source-root[Exact inventory root to select]:dir:_files -/' \
        '*--replot-exclude-root[Exact inventory root to protect]:dir:_files -/' \
        '--replot-max-source-files[Maximum selected old files]:count:' \
        '--replot-max-source-bytes[Maximum selected allocated old-file bytes]:size:' \
        '--farm-adapter[Installed native farm adapter]:adapter:(chia gigahorse drchia)' \
        '--farm-adapter-timeout-ms[Inventory or migration adapter operation deadline]:milliseconds:' \
        '--reclaim-before-bytes[Allocated-byte limit for early deletion before replacements are ready]:size:' \
        '--no-assess-jobs[Restore ordinary plan mode]' \
        '--calibrate[Measure explicitly supplied plot files]' \
        '--no-calibrate[Disable reader calibration]' \
        '--calibration-samples[Quality lookups per plot]:count:' \
        '--calibration-timeout-ms[Reader worker deadline]:milliseconds:' \
        '--calibration-memory-bytes[Reader worker memory limit]:size:' \
        '--paths-from[Literal destination list]:file:_files' \
        '--paths0-from[NUL-delimited destination list]:file:_files' \
        '*--discover-drives[Discover mounted storage below a directory]:directory:_files -/' \
        '*--mounts-under[Alias for discover-drives]:directory:_files -/' \
        '--extend[Add destinations or refresh farm capacity]' \
        '--update[Revise future farm policy]' \
        '--cancel-update[Cancel a queued farm update]' \
        '--allow-missing[Resume available destinations]' \
        '--relocate[Move a saved destination]:old directory:_files -/:new directory:_files -/' \
        '--construction[Group construction method]:value:' \
        '--backend[CPU/GPU plot generation]:value:' \
        '--device[GPU device]:value:' \
        '--max-ram[Combined plotting RAM budget]:value:' \
        '-j[Shared plotting CPU budget]:count:' '--workers[Shared plotting CPU budget]:count:' \
        '--batch-members[Construction batch limit]:value:' \
        '--max-members[Maximum total file membership]:value:' \
        '--max-groups[File limit per directory]:value:' \
        '--max-total-members[Member limit per directory]:value:' \
        '--capacity-bytes[Output capacity]:value:' \
        '--leave-free[Space to leave free; binary unit suffixes accepted]:size:' \
        '--headroom-bytes[Alias for leave-free]:size:' \
        '--destination-temp[Allow temporary files on destinations]:value:' \
        '--temp-capacity-bytes[Temporary storage limit per filesystem]:value:' \
        '--member-storage[Temporary member storage format]:value:' \
        '--window-bytes[I/O window size in bytes]:value:' \
        '--checkpoint-bytes[Target bytes between recovery checkpoints]:value:' \
        '--io-overlap[I/O overlap]:value:' \
        '--generation-overlap[Overlap generation with construction]:value:' \
        '--cache-release[Release clean file cache]:value:' \
        '--save-plan[Save a plan without private plot keys]:file:_files' \
        '--alternative[Review assessed resource choices, or capacity alone]:names:(memory eligible-groups deadline cpu-cores lookup-workers capacity)' \
        '--upgrade-job[Review additional space for recovery records on a stopped job]' \
        '-F[Use a saved plan]:file:_files' '--from-plan[Use a saved plan]:file:_files' \
        '--save-profile[Export reusable harvester or migration setup]:file:_files' \
        '*--resource-setting[Declare a remaining inventory/replot bound by stable ID]:setting:' \
        '--help-all[All planning and recovery options]' \
        "--harvester[Intended harvester]:harvester:_alternative 'local\:local harvester\:(local)' 'profile\:profile file\:_files'" \
        '--chia-root[Local Chia configuration root]:directory:_files -/' \
        '--chia-executable[Installed Chia CLI]:file:_files' \
        '--farming-reader[Final reader profile]:value:' \
        '--farming-objective[Layout objective]:value:' \
        '--farming-cores[Harvester CPU core limit]:value:' \
        '--farming-lookup-workers[Concurrent harvester lookups]:value:' \
        '--farming-memory-bytes[Reader RAM limit in bytes]:value:' \
        '--farming-index-cache-bytes[Reader index cache in bytes]:value:' \
        '--farming-filter-bits[Current filter]:value:' \
        '--farming-future-filter-bits[Future filter]:value:' \
        '--farming-deadline-ms[Response deadline in milliseconds]:value:' \
        '--farming-chain-us-per-member[CPU cost per member in microseconds]:value:' \
        '--farming-index-ns-per-entry[Index decode cost per entry in nanoseconds]:value:' \
        '--farming-cpu-work-weight[CPU work weight]:value:' \
        '--farming-io-work-weight[I/O work weight]:value:' \
        '--farming-max-eligible-groups[Eligible groups per resource]:value:' \
        '--compute-broker[Broker address]:value:' \
        '--compute-owner-id[Stable plotting job owner ID]:value:' \
        '--compute-ca[Trusted compute CA]:value:' \
        '--compute-cert[Compute certificate]:value:' \
        '--compute-key[Compute private key]:value:' \
        '--compute-inflight[Task concurrency ceiling]:value:' \
        '--compute-codec-ram[Remote codec RAM]:value:' \
        '--compute-timeout-ms[Task timeout in milliseconds]:value:' \
        '--compute-socket-timeout-ms[Socket timeout in milliseconds]:value:' \
        '--compute-frame-bytes[Maximum network message size in bytes]:value:' \
        '--compute-listen[Local worker listener]:value:' \
        '--compute-advertise[Local worker address]:value:' \
        '--compute-offer-ram[Local shared RAM]:value:' \
        '--compute-offer-cpus[Local shared CPU]:value:' \
        '--compute-offer-vram[Local shared VRAM]:value:' \
        '--compute-offer-slots[Local task slots]:value:' \
        '--compute-offer-backend[Local CPU/GPU plot generation]:value:' \
        '--listen[Listener address and port]:value:' \
        '--advertise[Address and port reachable by plotting hosts]:value:' \
        '--broker[Broker address and port]:value:' \
        '--tls-ca[Trusted certificate authority file]:value:' \
        '--tls-cert[Service certificate file]:value:' \
        '--tls-key[Service private key file]:value:' \
        '--host-bytes[Offered host RAM limit in bytes]:value:' \
        '--cpus[Offered CPU thread limit]:value:' \
        '--gpu-bytes[Offered GPU RAM limit in bytes]:value:' \
        '--slots[Maximum active worker tasks]:value:' \
        '--timeout-ms[Query or connection timeout in milliseconds]:value:' \
        '--lease-ms[Renewable task assignment duration in milliseconds]:value:' \
        '--worker-ms[Worker registration expiry in milliseconds]:value:' \
        '--control-threads[Concurrent broker request handlers]:value:' \
        '--max-connections[Maximum active and idle control connections]:value:' \
        '--idle-ms[Idle connection timeout in milliseconds]:value:' \
        '--owner-share[Relative scheduling share as owner-ID\:weight]:value:' \
        '--max-workers[Maximum registered workers]:value:' \
        '--max-owners[Maximum plotting job owners]:value:' \
        '--max-tasks[Maximum queued, active and recent task records]:value:' \
        '--max-frame-bytes[Maximum network message size in bytes]:value:' \
        '--tier[Streaming tier]:tier:(plain compact minimal tiny pinned auto)' \
        '-d[Device selector]:spec:(all gpu cpu 0 1 2 3)' '--devices[Device selector]:spec:(all gpu cpu 0 1 2 3)' \
        '--progress[Force aggregate progress line on]' \
        '--no-progress[Force aggregate progress line off]' \
        '-v[Verbose]' '--verbose[Verbose]' '--no-verbose[Concise output]' \
        '-q[Quiet — suppress info-level output]' '--quiet[Quiet — suppress info-level output]' '--no-quiet[Show informational output]' \
        '-r[Resume a saved job]' '--resume[Resume a saved job]' \
        '--cpu[Add CPU worker]' \
        '--cpu-workers[CPU plots to run concurrently: auto|max|N|off (default auto, RAM-gated)]:count:(auto max off)' \
        '--shard-plot[Single-plot multi-GPU]' \
        '--pipeline-plot[Pipeline-parallel multi-stage]' \
        '--manifest[Saved job]:file:_files' \
        '--config[Configuration file]:file:_files' \
        '--strategy[Batch strategy]:strategy:(auto work-queue pipeline shard)' \
        '--pipeline-depth[Pipeline depth]:count:' \
        '--pipeline-stage-tiers[Pipeline stage tiers]:tiers:' \
        '-M[Host RAM budget]:size:' '--max-host-ram[Host RAM budget]:size:' \
        '-t[Spill directory]:dir:_files -/' '--temp-dir[Spill directory]:dir:_files -/' \
        '--no-auto-spill[Refuse automatic spill]' \
        '--warmup[Warmup plots per worker]:count:' \
        '--keep[Retain benchmark plots]' '--compute-only[Add RAM-backed pass]' \
        '--target-size[Target size in TiB]:size:' '--json[Benchmark JSON on stdout]' \
        '-k[Plot size]:k:' '-n[Plot count]:count:' '-s[Strength]:strength:' \
        '-f[Farmer public key]:hex:' '-p[Pool public key]:hex:' \
        '--pool-ph[Pool puzzle hash]:hex:' '-c[Pool contract address]:address:' \
        '-i[Plot index]:index:' '-g[Meta group]:group:' '-S[Seed]:hex:' \
        '--raw[Temporary raw group member (test only)]' \
        '-o[Output dir]:dir:_files -/' \
        "$positional"
}
_xchplot2 "$@"
)";
        } else if (shell == "fish") {
            std::cout << R"(# fish completion for xchplot2 — install at:
#     ~/.config/fish/completions/xchplot2.fish
complete -c xchplot2 -f
complete -c xchplot2 -n '__fish_use_subcommand' -s V -l version -d 'Print version and source revision'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'plan' -d 'Preview farming layout'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'fill' -d 'Fill destination drives'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'status' -d 'Inspect farm progress'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'pause' -d 'Pause at a safe checkpoint'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'drain' -d 'Finish started files, then stop'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'resume' -d 'Continue paused or drained work'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'retire' -d 'Detach destinations, keep plots'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'grow' -d 'Extend an existing group'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'compute' -d 'Share compute'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute; and not __fish_seen_subcommand_from setup init invite join configure members revoke renew-request renew-sign renew-import policy-export policy-import status pause drain resume service broker worker' -a 'setup init invite join configure members revoke renew-request renew-sign renew-import policy-export policy-import status pause drain resume service broker worker'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute -r -d 'Shared compute auto or off'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-directory -r -d 'Saved compute setup directory'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-local -r -d 'Offer local compute on or off'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute; and __fish_seen_subcommand_from service; and not __fish_seen_subcommand_from install start stop restart status remove print' -a 'install start stop restart status remove print'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute; and __fish_seen_subcommand_from setup' -l role -r -a 'plotter worker' -d 'Compute setup role'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute; and __fish_seen_subcommand_from service' -l role -r -a 'worker broker' -d 'Background service role'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l worker -r -d 'Worker status or control endpoint'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l limit -r -d 'Rows per broker status section'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-brokers -r -d 'Worker broker registration limit'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l directory -r -d 'Saved compute setup directory'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l name -r -d 'Machine display name'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l interface -r -d 'Sharing network interface'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l discovery -r -d 'LAN discovery on or off'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l ca -r -d 'Trusted worker CA'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l cert -r -d 'Worker certificate'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l key -r -d 'Worker private key'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l generation-threads -r -d 'CPU threads per generation task'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-task-bytes -r -d 'Codec and generation scratch limit in bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-window-chunks -r -d 'Maximum chunks per codec task'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-batch-members -r -d 'Maximum members per generation task'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l task-timeout-ms -r -d 'Task timeout in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l tls-timeout-ms -r -d 'TLS operation timeout in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l group-members -r -d 'Members per file'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l plan-only -d 'Preview without writing'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l farm-adapter -r -a 'chia gigahorse drchia' -d 'Installed native farm adapter'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l farm-adapter-timeout-ms -r -d 'Inventory or migration adapter operation deadline'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan' -l inventory -d 'Inspect explicit farm roots'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan' -l alternative -r -a 'memory eligible-groups deadline cpu-cores lookup-workers capacity' -d 'Review assessed resource choices, or capacity alone'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan' -l upgrade-job -d 'Review additional space for recovery records on a stopped job'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot -d 'Replace eligible old plots; plan previews, fill executes'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l no-replot -d 'Use ordinary plotting'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-inventory -r -F -d 'Optional saved inventory'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-confirmation -r -a 'directory file' -d 'Farming acknowledgement scope'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l discover-farm -d 'Include configured farm directories'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l no-discover-farm -d 'Use supplied destinations'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l farm-config -r -F -d 'Additional reader roots or configuration files'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l exclude-destination -r -a '(__fish_complete_directories)' -d 'Directories protected from output, scratch and retirement'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill grow' -l temp-dirs -r -a '(__fish_complete_directories)' -d 'Temporary directories, separated by spaces'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-verification -r -a 'structural sampled strict' -d 'Structural checks, plus proof samples for selected or all groups'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-proof-first-groups -r -d 'Proof-check the first N groups with the sampled policy'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-proof-every-groups -r -d 'Proof-check every Nth group across destinations with the sampled policy; 0 disables'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l status -d 'Inspect saved farm progress'
complete -c xchplot2 -n '__fish_seen_subcommand_from status plan fill; and not __fish_seen_subcommand_from compute' -l watch -d 'Refresh status until interrupted'
complete -c xchplot2 -n '__fish_seen_subcommand_from status plan fill; and not __fish_seen_subcommand_from compute' -l inspect -d 'Scan outputs and check recovery checkpoints'
complete -c xchplot2 -n '__fish_seen_subcommand_from status plan fill; and not __fish_seen_subcommand_from compute' -l live -d 'Query shared compute and migration service readiness'
complete -c xchplot2 -n '__fish_seen_subcommand_from status plan fill; and not __fish_seen_subcommand_from compute' -l timeout-ms -r -d 'Total live query time in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from pause drain resume retire; and not __fish_seen_subcommand_from compute' -l destination -r -a '(__fish_complete_directories)' -d 'Saved destination to control'
complete -c xchplot2 -n '__fish_seen_subcommand_from pause drain resume retire; and not __fish_seen_subcommand_from compute' -l policy-only -d 'Save the control setting without starting an idle job'
complete -c xchplot2 -n '__fish_seen_subcommand_from status pause drain resume retire; and not __fish_seen_subcommand_from compute' -l manifest -r -F -d 'Saved farming job'
complete -c xchplot2 -n '__fish_seen_subcommand_from status pause drain resume retire; and not __fish_seen_subcommand_from compute' -a '(__fish_complete_path)' -d 'Saved job directory or plan'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan' -l revalidate -d 'Review updated migration resources or verification policy'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-source-root -r -a '(__fish_complete_directories)' -d 'Exact inventory root to select'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-exclude-root -r -a '(__fish_complete_directories)' -d 'Exact inventory root to protect'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-max-source-files -r -d 'Maximum selected old files'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l replot-max-source-bytes -r -d 'Maximum selected allocated old-file bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l reclaim-before-bytes -r -d 'Allocated-byte limit for early deletion before replacements are ready'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l save-plan -r -F -d 'Save a plan without private plot keys'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -s F -l from-plan -r -F -d 'Use a saved plan'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l save-profile -r -F -d 'Export reusable harvester or migration setup'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l resource-setting -r -d 'Declare an inventory/replot bound by stable ID'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill grow plot batch bench test verify devices parity-check compute' -l help-all -d 'Full command reference'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l harvester -r -F -a local -d 'Local harvester or profile file'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l chia-root -r -a '(__fish_complete_directories)' -d 'Local Chia configuration root'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l chia-executable -r -F -d 'Installed Chia CLI'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l paths-from -r -d 'Literal destination list'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l paths0-from -r -d 'NUL-delimited destination list'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l discover-drives -r -a '(__fish_complete_directories)' -d 'Discover mounted storage below a directory'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l mounts-under -r -a '(__fish_complete_directories)' -d 'Alias for discover-drives'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l extend -d 'Add destinations or refresh farm capacity'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l update -d 'Revise future farm policy'
complete -c xchplot2 -n '__fish_seen_subcommand_from fill' -l cancel-update -d 'Cancel a queued farm update'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l allow-missing -d 'Resume available destinations'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -l relocate -r -d 'Old and new destination directories'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill' -a '(__fish_complete_path)' -d 'Destination directory or saved plan'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l construction -r -d 'Group construction method'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l backend -r -d 'CPU/GPU plot generation'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l device -r -d 'GPU device'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l max-ram -r -d 'Combined plotting RAM budget'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -s j -l workers -r -d 'Shared plotting CPU budget'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l batch-members -r -d 'Construction batch limit'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l max-members -r -d 'Maximum total file membership'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l max-groups -r -d 'File limit per directory'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l max-total-members -r -d 'Member limit per directory'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l capacity-bytes -r -d 'Output capacity'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l leave-free -r -d 'Space to leave free; binary unit suffixes accepted'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l headroom-bytes -r -d 'Alias for leave-free'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l destination-temp -r -d 'Allow temporary files on destinations'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l temp-capacity-bytes -r -d 'Temporary storage limit per filesystem'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l member-storage -r -d 'Temporary member storage format'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l window-bytes -r -d 'I/O window size in bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l checkpoint-bytes -r -d 'Target bytes between recovery checkpoints'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l io-overlap -r -d 'I/O overlap'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l generation-overlap -r -d 'Overlap generation with construction'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l cache-release -r -d 'Release clean file cache'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-reader -r -d 'Final reader profile'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-objective -r -d 'Layout objective'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-cores -r -d 'Harvester CPU core limit'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-lookup-workers -r -d 'Concurrent harvester lookups'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-memory-bytes -r -d 'Reader RAM limit in bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-index-cache-bytes -r -d 'Reader index cache in bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-filter-bits -r -d 'Current filter'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-future-filter-bits -r -d 'Future filter'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-deadline-ms -r -d 'Response deadline in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-chain-us-per-member -r -d 'CPU cost per member in microseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-index-ns-per-entry -r -d 'Index decode cost per entry in nanoseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-cpu-work-weight -r -d 'CPU work weight'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-io-work-weight -r -d 'I/O work weight'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l farming-max-eligible-groups -r -d 'Eligible groups per resource'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-broker -r -d 'Broker address'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-owner-id -r -d 'Stable plotting job owner ID'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-ca -r -d 'Trusted compute CA'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-cert -r -d 'Compute certificate'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-key -r -d 'Compute private key'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-inflight -r -d 'Task concurrency ceiling'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-codec-ram -r -d 'Remote codec RAM'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-timeout-ms -r -d 'Task timeout in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-socket-timeout-ms -r -d 'Socket timeout in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-frame-bytes -r -d 'Maximum network message size in bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-listen -r -d 'Local worker listener'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-advertise -r -d 'Local worker address'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-offer-ram -r -d 'Local shared RAM'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-offer-cpus -r -d 'Local shared CPU'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-offer-vram -r -d 'Local shared VRAM'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-offer-slots -r -d 'Local task slots'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -l compute-offer-backend -r -d 'Local CPU/GPU plot generation'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l listen -r -d 'Listener address and port'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l advertise -r -d 'Address and port reachable by plotting hosts'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l broker -r -d 'Broker address and port'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l tls-ca -r -d 'Trusted certificate authority file'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l tls-cert -r -d 'Service certificate file'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l tls-key -r -d 'Service private key file'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l host-bytes -r -d 'Offered host RAM limit in bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l cpus -r -d 'Offered CPU thread limit'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l gpu-bytes -r -d 'Offered GPU RAM limit in bytes'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l slots -r -d 'Maximum active worker tasks'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l backend -r -d 'CPU/GPU plot generation'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l device -r -d 'GPU device index'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l timeout-ms -r -d 'Query or connection timeout in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l lease-ms -r -d 'Renewable task assignment duration in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l worker-ms -r -d 'Worker registration expiry in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l control-threads -r -d 'Concurrent broker request handlers'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-connections -r -d 'Maximum active and idle control connections'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l idle-ms -r -d 'Idle connection timeout in milliseconds'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l owner-share -r -d 'Relative scheduling share as owner-ID:weight'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-workers -r -d 'Maximum registered workers'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-owners -r -d 'Maximum plotting job owners'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-tasks -r -d 'Maximum queued, active and recent task records'
complete -c xchplot2 -n '__fish_seen_subcommand_from compute' -l max-frame-bytes -r -d 'Maximum network message size in bytes'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'batch'         -d 'Run a manifest of plots'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'plot'          -d 'Single-plot farmable mode'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'bench'         -d 'Measure plotting throughput'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'test'          -d 'Single test plot'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'verify'        -d 'Verify a plot file'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'devices'       -d 'List available GPU/CPU'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'parity-check'  -d 'Run parity tests'
complete -c xchplot2 -n '__fish_use_subcommand' -a 'completions'   -d 'Emit shell completion script'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l tier      -x -a 'plain compact minimal tiny pinned auto'  -d 'Streaming tier'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch bench' -s d -l devices -x -a 'all gpu cpu 0 1 2 3' -d 'Device selector'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l manifest  -r -d 'Saved job'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l progress  -d 'Force aggregate progress line on'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l no-progress -d 'Force aggregate progress line off'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill grow plot batch bench test verify devices parity-check compute' -s v -l verbose -d 'Verbose'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill grow plot batch bench test verify devices parity-check compute' -l no-verbose -d 'Concise output'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill grow plot batch bench test verify devices parity-check compute' -s q -l quiet -d 'Quiet — suppress info-level output'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan fill grow plot batch bench test verify devices parity-check compute' -l no-quiet -d 'Show informational output'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l cpu-workers -x -a 'auto max off' -d 'CPU workers per node'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l cpu       -d 'Add CPU worker'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l shard-plot     -d 'Single-plot multi-GPU (experimental)'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l pipeline-plot  -d 'Pipeline-parallel multi-stage'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s o -l out  -r -d 'Output dir'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s k -l k -x -d 'Plot size'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s s -l strength -x -d 'Strength'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s f -l farmer-pk -x -d 'Farmer public key'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s p -l pool-pk -x -d 'Pool public key'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s c -l pool-contract-address -x -d 'Pool contract address'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s i -l plot-index -x -d 'Plot index'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s g -l meta-group -x -d 'Meta group'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s S -l seed -x -d 'Seed'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -s n -l num -x -d 'Plot count'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l pool-ph -x -d 'Pool puzzle hash'
complete -c xchplot2 -n '__fish_seen_subcommand_from test' -l raw -d 'Temporary raw group member'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch' -s r -l resume -d 'Resume a saved job'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l skip-existing -d 'Skip completed plots'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l continue-on-error -d 'Continue after plot errors'
complete -c xchplot2 -l config -r -d 'Configuration file'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch bench' -s M -l max-host-ram -x -d 'Host RAM budget'
complete -c xchplot2 -n '__fish_seen_subcommand_from plan plot fill grow batch bench' -s t -l temp-dir -r -d 'Spill directory'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l no-auto-spill -d 'Refuse automatic spill'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l strategy -x -a 'auto work-queue pipeline shard' -d 'Batch strategy'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l pipeline-depth -x -d 'Pipeline depth'
complete -c xchplot2 -n 'not __fish_seen_subcommand_from verify' -l pipeline-stage-tiers -x -d 'Pipeline stage tiers'
complete -c xchplot2 -n '__fish_seen_subcommand_from bench' -l warmup -x -d 'Warmup plots per worker'
complete -c xchplot2 -n '__fish_seen_subcommand_from bench' -l keep -d 'Retain benchmark plots'
complete -c xchplot2 -n '__fish_seen_subcommand_from bench' -l compute-only -d 'Add RAM-backed pass'
complete -c xchplot2 -n '__fish_seen_subcommand_from bench' -l target-size -x -d 'Target size in TiB'
complete -c xchplot2 -n '__fish_seen_subcommand_from bench' -l json -d 'Benchmark JSON on stdout'
complete -c xchplot2 -n '__fish_seen_subcommand_from verify' -s n -l trials -x -d 'Random challenges'
complete -c xchplot2 -n '__fish_seen_subcommand_from verify' -l full -d 'Validate full proofs'
complete -c xchplot2 -n '__fish_seen_subcommand_from verify' -l no-full -d 'Quality-chain sample'
complete -c xchplot2 -n '__fish_seen_subcommand_from verify' -F
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
    std::optional<bool> verbose_flag;
    bool quiet = false, literal = false;

    // Strip flags from argv into a positional vector.
    std::vector<std::string> pos;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (literal) { pos.push_back(a); continue; }
        if (a == "--") { literal = true; continue; }
        if (command_help(argv[0], mode, a)) return 0;
        if (a == "--verbose" || a == "-v") { verbose_flag = true; continue; }
        if (a == "--no-verbose") { verbose_flag = false; continue; }
        if (a == "--quiet" || a == "-q") { quiet = true; continue; }
        if (a == "--no-quiet") { quiet = false; continue; }
        if (a == "--testnet" || a == "-T") {
            std::cerr << "Error: PoS2 1.0 removes testnet-specific plots; omit --testnet\n";
            return 1;
        }
        else if (a == "--no-testnet") opts.testnet = false;
        else if (a == "--raw") opts.raw = true;
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
        print_command_help(argv[0], mode, std::cerr);
        return 1;
    }
    if (pos.size() > 6) {
        std::cerr << "Error: unexpected argument: " << pos[6] << '\n'; return 1;
    }

    if (!parse_integer_arg(pos[0], opts.k, "k")) return 1;
    if (!parse_hex(pos[1], opts.group_id)) {
        std::cerr << "Error: group_id must be 64 hex characters\n";
        return 1;
    }
    if (pos.size() >= 3 && !parse_integer_arg(pos[2], opts.strength, "strength")) return 1;
    if (pos.size() >= 4 && !parse_integer_arg(pos[3], opts.plot_index, "plot_index")) return 1;
    if (pos.size() >= 5 && !parse_integer_arg(pos[4], opts.meta_group, "meta_group")) return 1;
    if (pos.size() >= 6) {
        int verbose = 0;
        if (!parse_integer_arg(pos[5], verbose, "verbose")) return 1;
        opts.verbose = verbose != 0;
    }
    if (verbose_flag) opts.verbose = *verbose_flag;
    if (quiet && opts.verbose) {
        std::cerr << "Error: -q/--quiet and -v/--verbose are mutually exclusive\n"; return 1;
    }

    if (opts.plot_index < 0 || opts.plot_index > 65535 ||
        opts.meta_group < 0 || opts.meta_group > 255 || (!opts.raw && opts.plot_index != 0)) {
        std::cerr << "Error: group files require index 0; raw member indices must be 0..65535 and meta group 0..255\n";
        return 1;
    }
    opts.plot_id = pos2gpu::plot_id_for_group(opts.group_id, uint16_t(opts.plot_index), uint8_t(opts.meta_group));

    try {
        std::string out = pos2gpu::plot_to_file(opts, output_dir);
        std::cout << "Wrote: " << out << "\n";
    } catch (std::exception const& e) {
        std::cerr << "Plotting failed: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
