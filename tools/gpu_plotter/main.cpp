// gpu_plotter — CLI wrapper around pos2gpu::plot_to_file. Mirrors
// pos2-chip/src/tools/plotter/plotter_main.cpp's argument surface so it
// can be substituted in scripts.

#include "host/GpuPlotter.hpp"
#include "host/BatchPlotter.hpp"
#include "pos2_keygen.h" // Rust shim for plot_id + memo derivation

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(char const* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " test <k> <plot_id_hex> [strength] [plot_index] [meta_group] [verbose] "
        << "[--testnet] [--out DIR] [--memo HEX] [--out-name NAME] "
        << "[--gpu-t1] [--gpu-t2] [--gpu-t3] [--gpu-all]\n"
        << "  " << prog << " batch <manifest.tsv> [-v]\n"
        << "    Manifest: one plot per non-empty/non-# line, whitespace-separated:\n"
        << "      k strength plot_index meta_group testnet plot_id_hex memo_hex out_dir out_name\n"
        << "    Runs GPU compute and CPU FSE in a producer/consumer pipeline so they overlap\n"
        << "    across consecutive plots. ~2x throughput vs separate `test` invocations.\n"
        << "  " << prog << " plot --k K --num N --farmer-pk HEX\n"
        << "         ( --pool-pk HEX | --pool-ph HEX | --pool-contract-address xch1... )\n"
        << "         [--strength S] [--out DIR] [--testnet]\n"
        << "         [--plot-index N] [--meta-group N] [--seed HEX] [-v]\n"
        << "    Standalone farmable plot(s): derives plot_id + memo internally\n"
        << "    from the keys via chia-rs, then batches through the GPU pipeline.\n"
        << "    --farmer-pk HEX             : 96 hex chars (48 B G1 public key).\n"
        << "    --pool-pk HEX               : 96 hex chars. Pool public key mode.\n"
        << "    --pool-ph HEX               : 64 hex chars (raw puzzle hash).\n"
        << "    --pool-contract-address ADR : Chia bech32m address (xch1.../txch1...);\n"
        << "                                  decoded internally to a 32-byte hash.\n"
        << "    --num N                     : number of plots to create.\n"
        << "    --plot-index N              : v2 PoS plot_index field (default 0).\n"
        << "    --meta-group N              : v2 PoS meta_group field (default 0).\n"
        << "    --seed HEX                  : optional 64 hex chars of master-SK\n"
        << "                                  entropy. Per-plot seed = SHA256(seed || i).\n"
        << "                                  Reproducible across runs. Defaults to\n"
        << "                                  fresh /dev/urandom per plot.\n"
        << "\n"
        << "    <k>            : even integer in [18, 32]\n"
        << "    <plot_id_hex>  : 64 hex characters\n"
        << "    [strength]     : optional, defaults to 2\n"
        << "    [plot_index]   : optional, defaults to 0\n"
        << "    [meta_group]   : optional, defaults to 0\n"
        << "    [verbose]      : optional, 0/1, default 0\n"
        << "    --testnet      : use testnet proof parameters\n"
        << "    --out DIR      : output directory, defaults to .\n"
        << "    --memo HEX     : memo bytes (hex); required for farmable plots\n"
        << "    --out-name NAME: override output filename (basename only)\n"
        << "    --gpu-tN       : run phase N on GPU (T1/T2/T3); default CPU\n"
        << "    --gpu-all      : run all phases on GPU (where implemented)\n";
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

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "batch") {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        std::string manifest = argv[2];
        bool verbose = false;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-v" || a == "--verbose") verbose = true;
        }
        try {
            auto entries = pos2gpu::parse_manifest(manifest);
            std::cerr << "[batch] " << entries.size() << " plots queued\n";
            auto res = pos2gpu::run_batch(entries, verbose);
            double per = res.plots_written ? res.total_wall_seconds / res.plots_written : 0;
            std::cerr << "[batch] wrote " << res.plots_written << " plots in "
                      << res.total_wall_seconds << " s ("
                      << per << " s/plot)\n";
            return 0;
        } catch (std::exception const& e) {
            std::cerr << "[batch] FAILED: " << e.what() << "\n";
            return 2;
        }
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

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](int more) -> bool {
                if (i + more >= argc) {
                    std::cerr << "Error: " << a << " requires " << more << " arg(s)\n";
                    return false;
                }
                return true;
            };
            if (a == "--k"          && need(1)) k        = std::atoi(argv[++i]);
            else if (a == "--num"   && need(1)) num      = std::atoi(argv[++i]);
            else if (a == "--strength"  && need(1)) strength = std::atoi(argv[++i]);
            else if (a == "--out"       && need(1)) out_dir  = argv[++i];
            else if (a == "--farmer-pk" && need(1)) farmer_pk_hex = argv[++i];
            else if (a == "--pool-pk"   && need(1)) pool_pk_hex   = argv[++i];
            else if (a == "--pool-ph"   && need(1)) pool_ph_hex   = argv[++i];
            else if (a == "--pool-contract-address" && need(1)) pool_addr = argv[++i];
            else if (a == "--plot-index" && need(1)) plot_index_base = std::atoi(argv[++i]);
            else if (a == "--meta-group" && need(1)) meta_group      = std::atoi(argv[++i]);
            else if (a == "--seed"       && need(1)) seed_hex        = argv[++i];
            else if (a == "--testnet")  testnet = true;
            else if (a == "-v" || a == "--verbose") verbose = true;
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
                int rc = pos2_keygen_derive_plot(
                    seed, sizeof(seed),
                    farmer_pk.data(),
                    pool_key.data(), pool_kind,
                    static_cast<uint8_t>(strength),
                    static_cast<uint16_t>(plot_index_base),
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
                e.plot_index = plot_index_base;
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

            auto res = pos2gpu::run_batch(entries, verbose);
            double per = res.plots_written
                ? res.total_wall_seconds / double(res.plots_written) : 0;
            std::cerr << "[plot] wrote " << res.plots_written << " plots in "
                      << res.total_wall_seconds << " s ("
                      << per << " s/plot)\n";
            for (auto const& e : entries) {
                std::cout << out_dir << "/" << e.out_name << "\n";
            }
            return 0;
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
        if (a == "--testnet") opts.testnet = true;
        else if (a == "--gpu-t1") opts.t1 = pos2gpu::PhaseStrategy::Gpu;
        else if (a == "--gpu-t2") opts.t2 = pos2gpu::PhaseStrategy::Gpu;
        else if (a == "--gpu-t3") opts.t3 = pos2gpu::PhaseStrategy::Gpu;
        else if (a == "--gpu-all") {
            opts.t1 = opts.t2 = opts.t3 = pos2gpu::PhaseStrategy::Gpu;
        }
        else if (a == "--profile") opts.profile = true;
        else if (a == "--out" && i + 1 < argc) {
            output_dir = argv[++i];
        }
        else if (a == "--memo" && i + 1 < argc) {
            std::string memo_hex = argv[++i];
            if (!parse_hex_bytes(memo_hex, opts.memo)) {
                std::cerr << "Error: --memo must be even-length hex\n";
                return 1;
            }
        }
        else if (a == "--out-name" && i + 1 < argc) {
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
