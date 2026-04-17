// PlotFileWriterParallel.hpp — drop-in replacement for
// pos2-chip/src/plot/PlotFile.hpp::PlotFile::writeData(PlotData, ...) but
// compresses chunks in parallel via std::async. The serial reference
// at PlotFile.hpp lines 51..156 takes ~3 s for k=28 single-threaded.
// Per-chunk FSE compression is independent — easy parallel win.
//
// Output bytes are byte-identical to PlotFile::writeData.

#pragma once

#include "plot/ChunkCompressor.hpp"
#include "plot/PlotData.hpp"
#include "plot/PlotFile.hpp"
#include "plot/PlotIO.hpp"
#include "pos/ProofParams.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pos2gpu {

inline size_t write_plot_file_parallel(
    std::string const& filename,
    PlotData const& data,
    ProofParams const& params,
    uint16_t const index,
    uint8_t const meta_group,
    std::span<uint8_t const> const memo,
    unsigned thread_count = 0)
{
    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 4;
    }

    // Build chunked representation (cheap; single pass over fragments)
    uint64_t const range_per_chunk = (1ULL << (params.get_k() + PlotFile::CHUNK_SPAN_RANGE_BITS));
    ChunkedProofFragments chunked
        = ChunkedProofFragments::convertToChunkedProofFragments(data, range_per_chunk);

    uint64_t const num_chunks = static_cast<uint64_t>(chunked.proof_fragments_chunks.size());
    int const stub_bits = params.get_k() - PlotFile::MINUS_STUB_BITS;

    // Parallel chunk compression. Each chunk's compressProofFragments call
    // is independent and CPU-bound — perfect for std::async.
    std::vector<std::vector<uint8_t>> compressed(num_chunks);
    {
        std::vector<std::future<void>> tasks;
        // Simple fan-out: fire one async task per chunk, but cap concurrency
        // by waiting in batches.
        for (uint64_t start = 0; start < num_chunks; start += thread_count) {
            uint64_t end = std::min<uint64_t>(start + thread_count, num_chunks);
            tasks.clear();
            tasks.reserve(end - start);
            for (uint64_t i = start; i < end; ++i) {
                tasks.emplace_back(std::async(std::launch::async, [&, i] {
                    uint64_t start_range = i * range_per_chunk;
                    compressed[i] = ChunkCompressor::compressProofFragments(
                        chunked.proof_fragments_chunks[i], start_range, stub_bits);
                }));
            }
            for (auto& f : tasks) f.get();
        }
    }

    // Serial write phase — file I/O is sequential anyway.
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open " + filename);

    out.write("pos2", 4);
    uint8_t const ver = PlotFile::FORMAT_VERSION;
    out.write(reinterpret_cast<char const*>(&ver), 1);
    out.write(reinterpret_cast<char const*>(params.get_plot_id_bytes()), 32);

    uint8_t const k_byte = static_cast<uint8_t>(params.get_k());
    uint8_t const mkb    = static_cast<uint8_t>(params.get_match_key_bits());
    out.write(reinterpret_cast<char const*>(&k_byte), 1);
    out.write(reinterpret_cast<char const*>(&mkb),    1);
    out.write(reinterpret_cast<char const*>(&index), 2);
    out.write(reinterpret_cast<char const*>(&meta_group), 1);

    uint8_t const memo_size = static_cast<uint8_t>(memo.size());
    out.write(reinterpret_cast<char const*>(&memo_size), 1);
    out.write(reinterpret_cast<char const*>(memo.data()), memo.size());

    out.write(reinterpret_cast<char const*>(&num_chunks), sizeof(num_chunks));
    if (!out) throw std::runtime_error("Failed to write chunk count to " + filename);

    std::streampos offsets_start_pos = out.tellp();
    uint64_t zero = 0;
    for (uint64_t i = 0; i < num_chunks; ++i) {
        out.write(reinterpret_cast<char const*>(&zero), sizeof(zero));
    }
    if (!out) throw std::runtime_error("Failed to write chunk offset placeholders to " + filename);

    std::vector<uint64_t> offsets(num_chunks);
    for (uint64_t i = 0; i < num_chunks; ++i) {
        offsets[i] = static_cast<uint64_t>(out.tellp());
        writeVector(out, compressed[i]);
        if (!out) {
            throw std::runtime_error(
                "Failed to write chunk " + std::to_string(i) + " to " + filename);
        }
    }

    size_t bytes_written = static_cast<size_t>(out.tellp());

    out.seekp(offsets_start_pos);
    if (!out) throw std::runtime_error("Failed to seek to chunk offsets in " + filename);
    for (uint64_t i = 0; i < num_chunks; ++i) {
        out.write(reinterpret_cast<char const*>(&offsets[i]), sizeof(offsets[i]));
    }
    if (!out) throw std::runtime_error("Failed to write chunk offsets to " + filename);
    out.seekp(0, std::ios::end);

    return bytes_written;
}

} // namespace pos2gpu
