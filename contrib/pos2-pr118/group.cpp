// Experimental PR #118 multi-member writer. The production format stays unchanged.
#include "plot/PlotFile.hpp"
#include "pos/ProofValidator.hpp"
#include "pos2_keygen.h"
#include "prove/Prover.hpp"
#include "solve/Solver.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

namespace
{
void require(bool condition, std::string const& message)
{
    if (!condition)
        throw std::runtime_error(message);
}
uint64_t number(char const* text)
{
    std::string s(text);
    require(!s.empty() && s.find_first_not_of("0123456789") == s.npos, "invalid integer");
    return std::stoull(s);
}
std::vector<uint8_t> hex(std::string const& text)
{
    require(text.size() % 2 == 0, "invalid hex length");
    auto digit = [](char c)
    {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        throw std::runtime_error("invalid hex");
    };
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < text.size(); i += 2)
        bytes.push_back((digit(text[i]) << 4) | digit(text[i + 1]));
    return bytes;
}
std::string hex(std::span<uint8_t const> bytes)
{
    return Utils::bytesToHex(bytes);
}
template <typename T> T read(std::ifstream& in)
{
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
std::ifstream input(std::string const& path)
{
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open " + path);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    return in;
}
struct Raw
{
    std::string path;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> fragments;
    uint64_t cached = std::numeric_limits<uint64_t>::max();
};
struct Job
{
    PlotGroupParams params;
    std::vector<uint8_t> memo;
    std::vector<Raw> members;
    uint64_t budget;
    uint64_t fixed;
    uint64_t cached_bytes = 0;

    void room(uint64_t extra) const
    {
        uint64_t const used = fixed + cached_bytes;
        require(used <= budget && extra <= budget - used, "group RAM budget exceeded");
    }
    std::span<uint64_t const> region(size_t member, uint64_t chunk, uint64_t live = 0)
    {
        auto& raw = members.at(member);
        uint64_t const region_range = 1ull
                                      << (params.get_k() + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        uint64_t const raw_chunk =
            chunk >> (PlotFile::CHUNK_SPAN_RANGE_BITS - PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        if (raw.cached != raw_chunk)
        {
            cached_bytes -= raw.fragments.capacity() * 8;
            std::vector<uint64_t>().swap(raw.fragments);
            auto in = input(raw.path);
            in.seekg(raw.offsets.at(raw_chunk));
            uint64_t const bytes = read<uint64_t>(in);
            room(bytes + live);
            require(bytes >= 12, "raw chunk too small");
            uint32_t const count = read<uint32_t>(in);
            uint32_t const fse_size = read<uint32_t>(in);
            uint32_t const stub_size = read<uint32_t>(in);
            require(uint64_t(fse_size) + stub_size + 12 == bytes, "invalid raw chunk lengths");
            require(stub_size == (uint64_t(count) * (params.get_k() - 2) + 7) / 8,
                    "invalid raw stub length");
            room(bytes + uint64_t(count) * 32 + live);
            in.seekg(raw.offsets.at(raw_chunk) + 8);
            std::vector<uint8_t> compressed(bytes);
            in.read(reinterpret_cast<char*>(compressed.data()), bytes);
            require(count > 0, "empty raw chunk");
            raw.fragments = ChunkCompressor::decompressProofFragments(
                compressed,
                raw_chunk * (1ull << (params.get_k() + PlotFile::CHUNK_SPAN_RANGE_BITS)),
                params.get_k() - 2);
            room(raw.fragments.capacity() * 8 + live);
            cached_bytes += raw.fragments.capacity() * 8;
            require(raw.fragments.size() == count &&
                        std::is_sorted(raw.fragments.begin(), raw.fragments.end()),
                    "invalid raw fragment order");
            uint64_t const start =
                raw_chunk * (1ull << (params.get_k() + PlotFile::CHUNK_SPAN_RANGE_BITS));
            require(raw.fragments.front() >= start &&
                        raw.fragments.back() < start + (1ull << (params.get_k() + 16)),
                    "raw fragments outside chunk");
            raw.cached = raw_chunk;
        }
        uint64_t const start = chunk * region_range;
        auto first = std::lower_bound(raw.fragments.begin(), raw.fragments.end(), start);
        auto last = std::lower_bound(first, raw.fragments.end(), start + region_range);
        require(first != last,
                "upstream grouped format requires a nonempty region for every member");
        return {first, last};
    }
};
Job load_job(std::string const& path, uint64_t budget, bool allow_missing = false)
{
    require(std::filesystem::file_size(path) < budget / 16,
            "group input manifest exceeds RAM budget");
    auto in = input(path);
    int k, strength, meta;
    std::string group_hex, memo_hex;
    in >> k >> strength >> meta >> group_hex >> memo_hex;
    require(k >= 18 && k <= 28 && meta >= 0 && meta <= 255 && strength >= 2 && strength <= 63,
            "invalid experimental group parameters (k must be 18..28)");
    PlotGroupParams params{PlotGroupId(group_hex), uint8_t(k), uint8_t(strength), uint8_t(meta)};
    Job job{params, hex(memo_hex), {}, budget, PlotGroupFile::getChunkCountForK(k) * 24};
    require(job.memo.size() <= 255, "memo too long");
    require(job.fixed < budget, "group index exceeds RAM budget");
    // The file-size cap above bounds text allocation before std::quoted parses.
    in.exceptions(std::ios::badbit);
    std::string name;
    while (in >> std::quoted(name))
    {
        require(job.members.size() < 65535 && name.size() <= 4096,
                "too many members or path too long");
        job.members.push_back(Raw{name, {}, {}, std::numeric_limits<uint64_t>::max()});
    }
    require(in.eof() && !job.members.empty(), "invalid or empty member list");
    uint64_t const max_raw_count = 1ull << (k - PlotFile::CHUNK_SPAN_RANGE_BITS);
    job.fixed += job.members.size() * (max_raw_count * 8 + 4096);
    job.room(0);
    for (size_t member = 0; member < job.members.size(); ++member)
    {
        auto& raw = job.members[member];
        if (allow_missing && !std::filesystem::exists(raw.path))
            continue;
        auto file = input(raw.path);
        uint64_t const size = std::filesystem::file_size(raw.path);
        require(read<std::array<char, 4>>(file) == std::array<char, 4>{'p', 'o', 's', '2'} &&
                    read<uint8_t>(file) == 1,
                "expected current raw v1 plot");
        auto const id = read<std::array<uint8_t, 32>>(file);
        require(PlotId(id) == params.get_plot_params_for_index(uint16_t(member)).get_plot_id(),
                "raw member ID mismatch");
        require(read<uint8_t>(file) == k && read<uint8_t>(file) == strength &&
                    read<uint16_t>(file) == member && read<uint8_t>(file) == meta,
                "raw member parameters/index mismatch");
        require(read<uint8_t>(file) == job.memo.size(), "raw memo length mismatch");
        std::vector<uint8_t> memo(job.memo.size());
        file.read(reinterpret_cast<char*>(memo.data()), memo.size());
        require(memo == job.memo, "raw member memo mismatch");
        uint64_t const count = read<uint64_t>(file);
        require(count > 0 && count <= max_raw_count, "raw chunk count exceeds bounds");
        require(size >= 51 + memo.size() + count * 8, "truncated raw index");
        raw.offsets.resize(count);
        file.read(reinterpret_cast<char*>(raw.offsets.data()), count * 8);
        uint64_t end = 51 + memo.size() + count * 8;
        for (uint64_t offset : raw.offsets)
        {
            require(offset == end && offset <= size && size - offset >= 8,
                    "invalid raw chunk offset");
            file.seekg(offset);
            uint64_t const bytes = read<uint64_t>(file);
            require(bytes <= size - offset - 8, "truncated raw chunk");
            end = offset + bytes + 8;
        }
        require(end == size, "unexpected raw trailing data");
    }
    return job;
}
// Same normalization as pinned PlotGroupFile::createFSECTable (private upstream).
std::unique_ptr<FSE_CTable, decltype(&POS2_FSE_freeCTable)> compression_table()
{
    std::array<short, 256> norm{};
    std::array<double, 178> weights{};
    double total = 0;
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = std::exp(-double(i) / 256.0);
        total += weights[i];
    }
    int assigned = 0;
    for (size_t i = 0; i < weights.size(); ++i)
    {
        norm[i] = std::max(short(1), short(weights[i] / total * 2048 + 0.5));
        assigned += norm[i];
    }
    norm[0] += short(2048 - assigned);
    std::unique_ptr<FSE_CTable, decltype(&POS2_FSE_freeCTable)> table(
        POS2_FSE_createCTable(177, 11), POS2_FSE_freeCTable);
    require(bool(table), "cannot allocate FSE table");
    require(!POS2_FSE_isError(POS2_FSE_buildCTable(table.get(), norm.data(), 177, 11)),
            "cannot build FSE table");
    return table;
}
void assemble(Job& job, std::string const& output_path)
{
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    require(bool(out), "cannot create group temporary file");
    out.exceptions(std::ios::badbit | std::ios::failbit);
    PlotGroupFile::Header header{PlotGroupFile::MAGIC,
                                 PlotGroupFile::FORMAT_VERSION,
                                 job.params.get_plot_group_id(),
                                 uint8_t(job.params.get_k()),
                                 uint8_t(job.params.get_strength()),
                                 uint16_t(job.members.size()),
                                 uint8_t(job.params.get_meta_group()),
                                 0,
                                 uint8_t(job.memo.size())};
    out.write(reinterpret_cast<char*>(&header), sizeof(header));
    out.write(reinterpret_cast<char*>(job.memo.data()), job.memo.size());
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(header.k);
    uint64_t const range = 1ull << (header.k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    uint64_t const threshold = 178ull << (header.k - 8);
    auto table = compression_table();
    std::vector<uint64_t> sizes(chunks);
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        uint64_t const begin = uint64_t(out.tellp());
        std::vector<uint8_t> high;
        BitWriter bits;
        uint64_t previous = chunk * range;
        for (size_t member = 0; member < job.members.size(); ++member)
        {
            auto fragments = job.region(member, chunk, high.capacity() * 64);
            job.room((high.size() + fragments.size()) * 64);
            bool first = true;
            uint64_t last = 0;
            for (uint64_t fragment : fragments)
            {
                if (!first && fragment == last)
                    continue;
                first = false;
                last = fragment;
                uint64_t const value = fragment + member * range;
                uint64_t const delta = value - previous;
                previous = value;
                uint64_t const quotient = delta / threshold;
                uint64_t const remainder = delta % threshold;
                require(quotient + 1 + header.k - 8 <= 64, "unencodable fragment delta");
                uint64_t const unary = (1ull << quotient) - 1;
                bits.append((unary << (header.k - 8)) |
                                (remainder & ((1ull << (header.k - 8)) - 1)),
                            uint32_t(quotient + 1 + header.k - 8));
                high.push_back(uint8_t(remainder >> (header.k - 8)));
            }
        }
        job.room(high.size() * 64);
        std::vector<uint8_t> ans(POS2_FSE_compressBound(high.size()));
        size_t const size = POS2_FSE_compress_usingCTable(ans.data(), ans.size(), high.data(),
                                                          high.size(), table.get());
        require(size > 0 && !POS2_FSE_isError(size), "cannot compress grouped region");
        uint64_t leb = size;
        do
        {
            uint8_t byte = (leb & 0x7f) | (leb > 0x7f ? 0x80 : 0);
            out.put(char(byte));
            leb >>= 7;
        } while (leb);
        out.write(reinterpret_cast<char*>(ans.data()), size);
        auto bytes = bits.asBytes();
        out.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
        sizes[chunk] = uint64_t(out.tellp()) - begin;
    }
    header.chunk_index_offset = uint64_t(out.tellp());
    BitWriter index;
    uint32_t const rice = uint32_t(gsz_rice_k_for_g(header.group_size));
    for (uint64_t size : sizes)
    {
        int64_t const delta = int64_t(size) - int64_t(gsz_expected_for_g(header.group_size));
        uint64_t const magnitude = delta < 0 ? uint64_t(-delta) : uint64_t(delta);
        require((magnitude >> rice) + rice + 2 <= 64,
                "group chunk index cannot encode this group size");
    }
    gsz_encode(sizes, header.group_size, index);
    auto bytes = index.asBytes();
    out.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
    out.seekp(0);
    out.write(reinterpret_cast<char*>(&header), sizeof(header));
    out.close();
}
void verify(Job& job, std::string const& path)
{
    auto file = input(path);
    auto const header = read<PlotGroupFile::Header>(file);
    uint64_t const size = std::filesystem::file_size(path);
    require(header.magic == PlotGroupFile::MAGIC &&
                header.version == PlotGroupFile::FORMAT_VERSION &&
                header.group_id == job.params.get_plot_group_id() &&
                header.k == job.params.get_k() && header.strength == job.params.get_strength() &&
                header.meta_group == job.params.get_meta_group() &&
                header.group_size == job.members.size() && header.memo_length == job.memo.size(),
            "group identity/membership mismatch");
    std::vector<uint8_t> memo(header.memo_length);
    file.read(reinterpret_cast<char*>(memo.data()), memo.size());
    require(memo == job.memo, "group memo mismatch");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(header.k);
    uint64_t const begin = sizeof(header) + memo.size();
    require(header.chunk_index_offset >= begin && header.chunk_index_offset < size &&
                size - header.chunk_index_offset <= chunks * 16,
            "invalid grouped index bounds");
    job.room((size - header.chunk_index_offset) + chunks * 8);
    std::vector<uint64_t> packed((size - header.chunk_index_offset + 7) / 8);
    file.seekg(header.chunk_index_offset);
    file.read(reinterpret_cast<char*>(packed.data()), size - header.chunk_index_offset);
    BitReader reader(packed, (size - header.chunk_index_offset) * 8);
    std::vector<uint64_t> sizes(chunks);
    gsz_decode(sizes, header.group_size, reader);
    uint64_t end = begin;
    for (uint64_t bytes : sizes)
    {
        require(bytes > 0 && bytes <= header.chunk_index_offset - end,
                "invalid grouped chunk size");
        // Bound all allocations in upstream readChunk before calling it.
        job.room(2 * bytes + uint64_t(header.group_size) * 64 * 64);
        end += bytes;
    }
    require(end == header.chunk_index_offset, "group chunk index does not cover payload");
    std::vector<uint64_t>().swap(packed);
    std::vector<uint64_t>().swap(sizes);
    GroupProver prover(path);
    auto& group = prover.getPlotGroup();
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        std::vector<std::vector<ProofFragment>> decoded(header.group_size);
        group.readChunk(chunk, decoded);
        uint64_t live = 0;
        for (auto const& fragments : decoded)
            live += fragments.capacity() * 8;
        for (size_t member = 0; member < job.members.size(); ++member)
        {
            auto expected = job.region(member, chunk, live);
            size_t offset = 0;
            bool first = true;
            uint64_t previous = 0;
            for (auto fragment : expected)
            {
                if (!first && previous == fragment)
                    continue;
                first = false;
                previous = fragment;
                require(offset < decoded[member].size() && decoded[member][offset++] == fragment,
                        "group payload/member mismatch");
            }
            require(offset == decoded[member].size(), "group payload has extra fragments");
        }
    }
    for (auto& raw : job.members)
        std::vector<uint64_t>().swap(raw.fragments);
    job.cached_bytes = 0;
    std::vector<size_t> validated(header.group_size);
    // Solve one member at a time: full-proof RAM does not grow with group size.
    for (uint8_t trial = 0; trial < 32; ++trial)
    {
        std::array<uint8_t, 32> challenge{};
        challenge.back() = trial;
        for (auto const& qualities : prover.prove(challenge))
        {
            require(qualities.plot_index < header.group_size, "proof member outside group");
            size_t const member = qualities.plot_index;
            auto params = job.params.get_plot_params_for_index(uint16_t(member));
            Solver solver(params);
            ProofFragmentCodec codec(params);
            ProofValidator validator(job.params, uint16_t(member));
            for (auto const& quality : qualities.quality_chains)
            {
                std::array<uint32_t, TOTAL_XS_IN_PROOF / 2> x_bits{};
                size_t index = 0;
                for (auto fragment : quality.chain_links)
                    for (auto x : codec.get_x_bits_from_proof_fragment(fragment))
                    {
                        require(index < x_bits.size(), "invalid quality width");
                        x_bits[index++] = x;
                    }
                require(index == x_bits.size(), "incomplete quality");
                bool valid = false;
                for (auto const& proof : solver.solve(x_bits))
                {
                    auto links = validator.validate_full_proof(proof, challenge);
                    if (links && *links == quality.chain_links)
                    {
                        valid = true;
                        break;
                    }
                }
                require(valid, "quality has no valid full proof");
                ++validated[member];
            }
        }
    }
    for (size_t member = 0; member < job.members.size(); ++member)
    {
        require(validated[member] > 0, "member has no full proof in 32 challenges");
        std::cout << "PASS group member=" << member << " full proofs=" << validated[member] << '\n';
    }
}
void prepare(int argc, char** argv)
{
    require(argc == 6, "prepare K STRENGTH META_GROUP COUNT (seed, farmer PK, pool key on stdin)");
    std::string seed_hex, farmer_hex, pool_hex;
    require(bool(std::cin >> seed_hex >> farmer_hex >> pool_hex), "missing preparation keys");
    auto seed = hex(seed_hex), farmer = hex(farmer_hex), pool = hex(pool_hex);
    uint64_t const k = number(argv[2]), strength = number(argv[3]), meta = number(argv[4]),
                   count = number(argv[5]);
    require(seed.size() == 32 && farmer.size() == 48 && (pool.size() == 32 || pool.size() == 48) &&
                strength <= 63 && meta <= 255 && count >= 1 && count <= 65535 && k >= 18 && k <= 28,
            "invalid prepare parameters");
    std::array<uint8_t, 32> id{};
    std::array<uint8_t, 128> memo{};
    size_t memo_size = memo.size();
    int rc = pos2_keygen_derive_group(seed.data(), seed.size(), farmer.data(), pool.data(),
                                      pool.size() == 48 ? POS2_POOL_PK : POS2_POOL_PH,
                                      uint8_t(strength), id.data(), memo.data(), &memo_size);
    require(rc == POS2_OK, "key derivation failed rc=" + std::to_string(rc));
    PlotGroupParams params{PlotGroupId(id), uint8_t(k), uint8_t(strength), uint8_t(meta)};
    std::cout << hex(id) << ' ' << hex(std::span(memo).first(memo_size)) << '\n';
    for (uint64_t i = 0; i < count; ++i)
        std::cout << params.get_plot_params_for_index(uint16_t(i)).get_plot_id().to_string()
                  << '\n';
}
} // namespace
int main(int argc, char** argv)
try
{
    require(std::endian::native == std::endian::little, "upstream format requires little endian");
    require(argc >= 2, "identity|prepare|assemble|verify");
    std::string const mode = argv[1];
    if (mode == "identity")
    {
        require(argc == 2, "identity takes no parameters");
        std::cout << POS2_PR118_IDENTITY << '\n';
    }
    else if (mode == "prepare")
        prepare(argc, argv);
    else
    {
        require((mode == "assemble" || mode == "verify" || mode == "check-raw") && argc == 5,
                "assemble|verify|check-raw INPUTS GROUP_FILE RAM_MIB");
        uint64_t const mib = number(argv[4]);
        require(mib >= 16 && mib <= 1048576, "RAM budget must be 16..1048576 MiB");
        auto job = load_job(argv[2], mib * 1024 * 1024, mode == "check-raw");
        if (mode == "check-raw")
            return 0;
        if (mode == "assemble")
            assemble(job, argv[3]);
        verify(job, argv[3]);
    }
    return 0;
}
catch (std::exception const& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
