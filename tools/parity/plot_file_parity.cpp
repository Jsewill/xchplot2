// plot_file_parity — round-trip test for the parallel plot-file writer.
//
// Synthesises a sorted uint64 fragment stream, writes it via
// `write_plot_file_parallel`, reads it back via `read_plot_file_fragments`,
// and asserts bit-exact equality. Closes the correctness gap introduced
// by the parallel chunkify + coarse-task fan-out in the writer, which
// was only indirectly exercised before (via "file opens" checks).
//
// CPU-side test; no GPU required.

#include "host/PlotFileWriterParallel.hpp"
#include "host/BatchPlotter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

std::array<uint8_t, 32> derive_plot_id(uint32_t seed)
{
    std::array<uint8_t, 32> id{};
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ uint64_t(seed) * 0x100000001B3ULL;
    for (size_t i = 0; i < id.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        id[i] = static_cast<uint8_t>(s >> 56);
    }
    return id;
}

// Runs pos2-chip's CPU plotter to produce a real, spec-valid T3
// fragment stream for the given plot_id at k,strength. Synthetic
// fragments don't satisfy the delta-fits-in-one-byte constraint of
// ChunkCompressor::deltifyAndStubProofFragments across all counts;
// using real output sidesteps that entirely.
std::vector<uint64_t> real_fragments(std::array<uint8_t, 32> const& plot_id,
                                     int k, int strength)
{
    return pos2gpu::run_cpu_plotter_to_fragments(
        plot_id.data(),
        static_cast<uint8_t>(k),
        static_cast<uint8_t>(strength),
        /*testnet=*/uint8_t{0},
        /*verbose=*/false);
}

bool run_one(char const* label, uint32_t seed, int k, int strength)
{
    auto const id        = derive_plot_id(seed);
    auto const fragments = real_fragments(id, k, strength);

    std::printf("[%s seed=%u k=%d strength=%d count=%zu]\n",
                label, seed, k, strength, fragments.size());

    std::filesystem::path tmp =
        std::filesystem::temp_directory_path() /
        ("xch_rt_s" + std::to_string(seed) + "_s" + std::to_string(strength) + ".plot2");

    std::vector<uint8_t> memo;  // empty
    pos2gpu::write_plot_file_parallel(
        tmp.string(),
        std::span<uint64_t const>(fragments),
        id.data(),
        static_cast<uint8_t>(k),
        static_cast<uint8_t>(strength),
        /*testnet=*/uint8_t{0},
        /*index=*/uint16_t{0},
        /*meta_group=*/uint8_t{0},
        std::span<uint8_t const>(memo),
        /*thread_count=*/0);

    std::vector<uint64_t> roundtrip = pos2gpu::read_plot_file_fragments(tmp.string());

    std::error_code ec;
    std::filesystem::remove(tmp, ec);

    std::printf("  wrote %zu, read back %zu\n", fragments.size(), roundtrip.size());
    if (roundtrip.size() != fragments.size()) {
        std::printf("  FAIL: size mismatch\n");
        return false;
    }
    uint64_t mismatches = 0;
    for (size_t i = 0; i < fragments.size(); ++i) {
        if (roundtrip[i] != fragments[i]) {
            if (mismatches < 5) {
                std::printf("  MISMATCH i=%zu: wrote=0x%llx read=0x%llx\n",
                            i,
                            static_cast<unsigned long long>(fragments[i]),
                            static_cast<unsigned long long>(roundtrip[i]));
            }
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        std::printf("  OK %zu fragments bit-exact round-trip\n", fragments.size());
        return true;
    }
    std::printf("  FAIL %llu / %zu mismatches\n",
                static_cast<unsigned long long>(mismatches), fragments.size());
    return false;
}

bool file_safety()
{
    auto dir = std::filesystem::temp_directory_path();
    std::random_device random;
    do { dir = std::filesystem::temp_directory_path() / ("xch_safety_" + std::to_string(random())); }
    while (!std::filesystem::create_directory(dir));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    } cleanup{dir};
    auto const path = dir / "plot.plot2", victim = dir / "sentinel";
    { std::ofstream out(victim); out << "preserve me"; }
#ifndef _WIN32
    std::filesystem::create_symlink(victim, path.string() + ".partial");
#endif
    pos2gpu::BatchEntry entry;
    entry.k = 18; entry.plot_id = derive_plot_id(1); entry.memo = {1, 2, 3};
    auto const fragments = real_fragments(entry.plot_id, entry.k, entry.strength);
    auto write = [&] {
        pos2gpu::write_plot_file_parallel(path.string(), fragments, entry.plot_id.data(),
            18, 2, 0, 0, 0, entry.memo);
    };
    write();
    std::ifstream in(victim);
    std::string sentinel; std::getline(in, sentinel);
    if (sentinel != "preserve me" || std::filesystem::is_symlink(path)) return false;
    if (!pos2gpu::plot_file_matches(path.string(), entry)) return false;
    auto wrong = entry; wrong.plot_id[0] ^= 1;
    if (pos2gpu::plot_file_matches(path.string(), wrong)) return false;
    wrong = entry; wrong.memo.push_back(4);
    if (pos2gpu::plot_file_matches(path.string(), wrong)) return false;
    wrong.memo.clear();
    if (pos2gpu::plot_file_matches(path.string(), wrong)) return false;
    std::exception_ptr errors[2];
    std::thread writers[2];
    for (int i = 0; i < 2; ++i) writers[i] = std::thread([&, i] {
        try { write(); } catch (...) { errors[i] = std::current_exception(); }
    });
    for (auto& writer : writers) writer.join();
    if (errors[0] || errors[1] || pos2gpu::read_plot_file_fragments(path.string()) != fragments) return false;
    auto const verification = pos2gpu::verify_plot_file(path.string(), 100, true);
    if (verification.proofs_found == 0 || verification.full_proofs_validated != verification.proofs_found) return false;
    auto corrupt_u64 = [&](std::streamoff offset) {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        uint64_t const impossible = ~uint64_t{0};
        file.seekp(offset); file.write(reinterpret_cast<char const*>(&impossible), sizeof(impossible));
        file.close();
        bool const rejected = !pos2gpu::plot_file_matches(path.string(), entry);
        write();
        return rejected;
    };
    // Attacker-controlled chunk count and offset must be rejected before use.
    if (!corrupt_u64(43 + entry.memo.size()) || !corrupt_u64(51 + entry.memo.size())) return false;
    auto const size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 1);
    if (pos2gpu::plot_file_matches(path.string(), entry)) return false;
    { std::ofstream junk(path); junk << "pos2" << std::string(60, 'x'); }
    if (pos2gpu::plot_file_matches(path.string(), entry)) return false;
    std::printf("  OK exclusive temp, concurrent writers, resume identity/bounds, and %zu full proofs\n",
                verification.full_proofs_validated);
    return true;
}

} // namespace

int main()
{
    bool all_ok = true;

    // Round-trip over varied plot ids at k=18, strength=2 — uses the real
    // CPU plotter to produce fragment streams that are guaranteed valid
    // inputs to the compression format.
    for (uint32_t seed : {1u, 2u, 17u, 42u, 0xCAFEBABEu, 0xDEADBEEFu}) {
        all_ok = run_one("k18s2",   seed, 18, 2) && all_ok;
    }
    // Boundary seeds.
    for (uint32_t seed : {0u, 0xFFFFFFFFu, 0x80000000u}) {
        all_ok = run_one("boundary", seed, 18, 2) && all_ok;
    }
    // Different strength, same k.
    for (uint32_t seed : {1u, 17u}) {
        all_ok = run_one("k18s4",   seed, 18, 4) && all_ok;
    }

    all_ok = file_safety() && all_ok;
    std::printf("\n==> %s\n", all_ok ? "ALL OK" : "FAIL");
    return all_ok ? 0 : 1;
}
