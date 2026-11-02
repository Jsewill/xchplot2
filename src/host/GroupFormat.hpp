// Shared input helpers and canonical PoS2 group encoding for the experimental tools.
#pragma once
#include "plot/PlotFile.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>

namespace group_tools
{
inline void require(bool condition, char const* message)
{
    if (!condition)
        throw std::runtime_error(message);
}
inline void require(bool condition, std::string const& message)
{
    if (!condition)
        throw std::runtime_error(message);
}
inline uint64_t number(char const* text)
{
    std::string s(text);
    require(!s.empty() && s.find_first_not_of("0123456789") == s.npos, "invalid integer");
    return std::stoull(s);
}
inline std::vector<uint8_t> hex(std::string const& text)
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
inline std::string hex(std::span<uint8_t const> bytes)
{
    return Utils::bytesToHex(bytes);
}
template <typename T> T read(std::ifstream& in)
{
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}
inline std::ifstream input(std::string const& path)
{
    std::ifstream in(std::filesystem::path(reinterpret_cast<char8_t const*>(path.c_str())), std::ios::binary);
    require(bool(in), "cannot open " + path);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    return in;
}
// Same normalization as pinned PlotGroupFile::createFSECTable (private upstream).
inline std::array<short, 256> compression_counts()
{
    static auto const counts = []
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
        return norm;
    }();
    return counts;
}
inline std::unique_ptr<FSE_CTable, decltype(&POS2_FSE_freeCTable)> compression_table()
{
    auto norm = compression_counts();
    std::unique_ptr<FSE_CTable, decltype(&POS2_FSE_freeCTable)> table(
        POS2_FSE_createCTable(177, 11), POS2_FSE_freeCTable);
    require(bool(table), "cannot allocate FSE table");
    require(!POS2_FSE_isError(POS2_FSE_buildCTable(table.get(), norm.data(), 177, 11)),
            "cannot build FSE table");
    return table;
}
// Same delta coding as the pinned writer. The existing assembler and the
// streaming prototype share this boundary; temporary containers are separate.
inline void append_delta(uint64_t delta, uint8_t k, std::vector<uint8_t>& high, BitWriter& bits)
{
    uint64_t const threshold = 178ull << (k - 8);
    uint64_t const quotient = delta / threshold;
    uint64_t const remainder = delta % threshold;
    require(quotient + 1 + k - 8 <= 64, "unencodable fragment delta");
    uint64_t const unary = (1ull << quotient) - 1;
    bits.append((unary << (k - 8)) | (remainder & ((1ull << (k - 8)) - 1)),
                uint32_t(quotient + 1 + k - 8));
    high.push_back(uint8_t(remainder >> (k - 8)));
}
} // namespace group_tools
