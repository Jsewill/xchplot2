// PlotFileWriterParallel.hpp — drop-in replacement for
// pos2-chip/src/plot/PlotFile.hpp::PlotFile::writeData(PlotData, ...) but
// compresses chunks in parallel via std::async.
//
// Output bytes are byte-identical to PlotFile::writeData.
//
// Two functions live here, both implemented in PlotFileWriterParallel.cpp.
// That .cpp is the SOLE TU in pos2-gpu that includes pos2-chip's plot/* and
// pos/ProofParams.hpp headers — keeping it that way avoids the multiple-
// definition link errors caused by non-inline soft_aesenc / soft_aesdec
// in pos2-chip's pos/aes/soft_aes.hpp. Other TUs talk to us via raw bytes
// and never see those types directly.
//
// Takes a span<uint64_t const> instead of PlotData so callers can pass
// pinned-host memory directly, avoiding a ~1 s pinned→heap memcpy per plot
// in batch mode.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace pos2gpu {

// Writes a v2 .plot2 file. Returns total bytes written.
//
// `t3_fragments` must already be sorted by proof_fragment (low 2k bits) —
// matching what GpuPipeline / pos2-chip's CPU plotter produce.
//
// `plot_id_32` is a 32-byte plot id; (k, strength, testnet) are the v2
// proof params. We accept raw bytes here (rather than a ProofParams ref)
// so this header doesn't drag pos2-chip headers into our other TUs.
//
// `thread_count == 0` uses std::thread::hardware_concurrency().
size_t write_plot_file_parallel(
    std::string const& filename,
    std::span<uint64_t const> t3_fragments,
    uint8_t const* plot_id_32,
    uint8_t  k,
    uint8_t  strength,
    uint8_t  testnet,
    uint16_t index,
    uint8_t  meta_group,
    std::span<uint8_t const> memo,
    unsigned thread_count = 0);

// Run pos2-chip's CPU `Plotter` end-to-end and return the sorted T3
// proof_fragment vector. Encapsulated here so other TUs don't need to
// include plot/Plotter.hpp (and through it pos/aes/soft_aes.hpp).
std::vector<uint64_t> run_cpu_plotter_to_fragments(
    uint8_t const* plot_id_32,
    uint8_t k,
    uint8_t strength,
    uint8_t testnet,
    bool    verbose);

// Reads a .plot2 file written by `write_plot_file_parallel` (or
// pos2-chip's CPU writer) and returns the concatenated decompressed
// T3 proof fragments in on-disk order. Used by plot_file_parity to
// verify write + read round-trip without exposing pos2-chip's
// plot/PlotFile.hpp to other TUs.
std::vector<uint64_t> read_plot_file_fragments(std::string const& filename);

} // namespace pos2gpu
