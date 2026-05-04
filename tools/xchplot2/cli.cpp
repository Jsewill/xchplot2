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
#include "host/GpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "host/Cancel.hpp"
#include "host/PlotFileWriterParallel.hpp"
#include "pos2_keygen.h" // Rust shim for plot_id + memo derivation

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void print_usage(char const* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " test <k> <plot_id_hex> [strength] [plot_index] [meta_group] [verbose]\n"
        << "         [-T|--testnet] [-o|--out DIR] [-m|--memo HEX] [-N|--out-name NAME]\n"
        << "         [--gpu-t1] [--gpu-t2] [--gpu-t3] [-G|--gpu-all] [-P|--profile]\n"
        << "  " << prog << " batch <manifest.tsv> [-v|--verbose]\n"
        << "         [--skip-existing] [--continue-on-error]\n"
        << "         [--devices <SPEC>]\n"
        << "    Manifest: one plot per non-empty/non-# line, whitespace-separated:\n"
        << "      k strength plot_index meta_group testnet plot_id_hex memo_hex out_dir out_name\n"
        << "    Runs GPU compute and CPU FSE in a producer/consumer pipeline so they overlap\n"
        << "    across consecutive plots. ~2x throughput vs separate `test` invocations.\n"
        << "  " << prog << " plot -k K -n N -f HEX  ( -p HEX | --pool-ph HEX | -c xch1... )\n"
        << "         [-s S] [-o DIR] [-T] [-i N] [-g N] [-S HEX] [-v]\n"
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
    << "    --skip-existing                 : skip plots whose output file is already a\n"
        << "                                      complete .plot2 (magic + non-trivial size).\n"
        << "    --continue-on-error             : log per-plot failures and keep going\n"
        << "                                      instead of aborting the batch.\n"
        << "    --devices SPEC                  : multi-device. SPEC is a comma\n"
        << "                                      list mixing any of:\n"
        << "                                        all       — every GPU + CPU\n"
        << "                                        gpu       — every visible GPU\n"
        << "                                        cpu       — CPU worker only (slow)\n"
        << "                                        0,1,3     — explicit GPU ids\n"
        << "                                      e.g. gpu,cpu == all.\n"
        << "                                      Omitted = single device via default\n"
        << "                                      SYCL selector (zero-config).\n"
        << "    --cpu                           : add a CPU worker alongside the\n"
        << "                                      selected GPUs (or use CPU only when\n"
        << "                                      no GPU is selected). Plotting on CPU\n"
        << "                                      is 1-2 orders of magnitude slower\n"
        << "                                      than GPU; intended for GPU-less\n"
        << "                                      hosts or as an extra worker.\n"
        << "    --tier plain|compact|minimal|auto : force streaming pipeline tier\n"
        << "                                      when GPU pool doesn't fit. plain =\n"
        << "                                      ~7.24 GB floor (k=28), faster.\n"
        << "                                      compact = ~5.33 GB floor, fits on\n"
        << "                                      tight 8 GB cards. minimal = ~3.83 GB\n"
        << "                                      floor, fits on 4 GiB cards (extra\n"
        << "                                      PCIe round-trips during T2 match).\n"
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
// Returns false on malformed input (caller prints usage + exits 1).
bool parse_devices_arg(std::string const& s, pos2gpu::BatchOptions& opts)
{
    // Accept comma-separated mix of:
    //   "all"      → every GPU + the CPU worker
    //   "gpu"      → every visible GPU only
    //   "cpu"      → the CPU worker only
    //   "<int>"    → opts.device_ids.push_back(int)  (real GPU index)
    // "cpu" alone is OK; otherwise at least one GPU token is required.
    opts.device_ids.clear();
    bool any_token = false;
    bool any_gpu_token = false;
    size_t start = 0;
    while (start <= s.size()) {
        size_t const end = s.find(',', start);
        std::string const tok = s.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (tok.empty()) return false;
        any_token = true;
        if (tok == "all") {
            opts.use_all_devices = true;
            opts.include_cpu = true;
            any_gpu_token = true;
        } else if (tok == "gpu") {
            opts.use_all_devices = true;
            any_gpu_token = true;
        } else if (tok == "cpu") {
            opts.include_cpu = true;
        } else {
            char* endp = nullptr;
            long const v = std::strtol(tok.c_str(), &endp, 10);
            if (endp == tok.c_str() || *endp != '\0' || v < 0 || v > 1023) {
                return false;
            }
            opts.device_ids.push_back(static_cast<int>(v));
            any_gpu_token = true;
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

} // namespace

extern "C" int xchplot2_main(int argc, char* argv[])
{
    pos2gpu::install_cancel_signal_handlers();

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
        // Use the printed `[N]` / `[cpu]` index with `--devices`.
        auto devices = pos2gpu::list_gpu_devices();
        std::printf("Visible devices (%zu GPU + 1 CPU):\n", devices.size());
        for (auto const& d : devices) {
            std::size_t vram_mb =
                static_cast<std::size_t>(d.vram_bytes / (1024ull * 1024ull));
#ifdef XCHPLOT2_HAVE_CUB
            char const* sort_hint = d.is_cuda_backend ? "CUB" : "SYCL";
#else
            char const* sort_hint = "SYCL";
#endif
            std::printf("  [%zu]   %-32s backend=%-10s vram=%5zu MB  CUs=%-4u  sort:%s\n",
                        d.id, d.name.c_str(), d.backend.c_str(),
                        vram_mb, d.cu_count, sort_hint);
        }
        // CPU row. hardware_concurrency() returns 0 when it can't
        // figure out the count (rare), in which case print "?".
        unsigned threads = std::thread::hardware_concurrency();
        if (threads == 0) {
            std::printf("  [cpu] %-32s backend=%-10s threads=  ?            sort:SYCL  (1-2 orders slower than GPU)\n",
                        "Host CPU plotter", "omp");
        } else {
            std::printf("  [cpu] %-32s backend=%-10s threads=%-4u           sort:SYCL  (1-2 orders slower than GPU)\n",
                        "Host CPU plotter", "omp", threads);
        }
        if (devices.empty()) {
            std::printf("\nNo GPU devices visible to AdaptiveCpp / SYCL.\n"
                        "Check rocminfo / nvidia-smi, ACPP_VISIBILITY_MASK, and that the\n"
                        "relevant SYCL backend was built into AdaptiveCpp.\n"
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

    if (mode == "batch") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        std::string manifest = argv[2];
        pos2gpu::BatchOptions opts{};
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "-v" || a == "--verbose")        opts.verbose = true;
            else if (a == "--skip-existing")               opts.skip_existing = true;
            else if (a == "--continue-on-error")           opts.continue_on_error = true;
            else if (a == "--cpu")                         opts.include_cpu = true;
            else if (a == "--tier" && i + 1 < argc) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal" && t != "auto") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', or 'auto' (got '" << t << "')\n";
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
        try {
            auto entries = pos2gpu::parse_manifest(manifest);
            std::cerr << "[batch] " << entries.size() << " plots queued\n";
            auto res = pos2gpu::run_batch(entries, opts);
            double per = res.plots_written
                ? res.total_wall_seconds / double(res.plots_written) : 0;
            std::cerr << "[batch] wrote " << res.plots_written << " plots in "
                      << res.total_wall_seconds << " s ("
                      << per << " s/plot)";
            if (res.plots_skipped) std::cerr << "; skipped " << res.plots_skipped;
            if (res.plots_failed)  std::cerr << "; failed "  << res.plots_failed;
            std::cerr << "\n";
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
        bool plot_include_cpu     = false;
        std::string plot_streaming_tier;

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
            else if  (a == "-v" || a == "--verbose")    verbose = true;
            else if  (a == "--skip-existing")           skip_existing = true;
            else if  (a == "--continue-on-error")       continue_on_error = true;
            else if  (a == "--cpu")                     plot_include_cpu = true;
            else if  (a == "--tier" && need(1)) {
                std::string t = argv[++i];
                if (t != "plain" && t != "compact" && t != "minimal" && t != "auto") {
                    std::cerr << "Error: --tier expects 'plain', 'compact', "
                                 "'minimal', or 'auto' (got '" << t << "')\n";
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
                plot_device_ids      = std::move(tmp.device_ids);
                plot_use_all_devices = tmp.use_all_devices;
                if (tmp.include_cpu) plot_include_cpu = true;
            }
            else {
                std::cerr << "Error: unknown argument: " << a << "\n";
                print_usage(argv[0]);
                return 1;
            }
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
            opts.include_cpu       = plot_include_cpu;
            opts.streaming_tier    = plot_streaming_tier;
            auto res = pos2gpu::run_batch(entries, opts);
            double per = res.plots_written
                ? res.total_wall_seconds / double(res.plots_written) : 0;
            std::cerr << "[plot] wrote " << res.plots_written << " plots in "
                      << res.total_wall_seconds << " s ("
                      << per << " s/plot)";
            if (res.plots_skipped) std::cerr << "; skipped " << res.plots_skipped;
            if (res.plots_failed)  std::cerr << "; failed "  << res.plots_failed;
            std::cerr << "\n";
            for (auto const& e : entries) {
                std::cout << out_dir << "/" << e.out_name << "\n";
            }
            return (res.plots_failed > 0) ? 3 : 0;
        } catch (std::exception const& e) {
            std::cerr << "[plot] FAILED: " << e.what() << "\n";
            return 2;
        }
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
