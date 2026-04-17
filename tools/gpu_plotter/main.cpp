// gpu_plotter — CLI wrapper around pos2gpu::plot_to_file. Mirrors
// pos2-chip/src/tools/plotter/plotter_main.cpp's argument surface so it
// can be substituted in scripts.

#include "host/GpuPlotter.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2 || std::string(argv[1]) != "test") {
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
