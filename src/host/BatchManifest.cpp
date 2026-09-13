#include "host/BatchPlotter.hpp"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#include "host/WindowsFile.hpp"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pos2gpu {

void validate_batch_entry(BatchEntry const& e)
{
    if (e.k < 18 || e.k > 32 || (e.k & 1))
        throw std::invalid_argument("k must be even in [18, 32]");
    int const sections = e.k < 28 ? 2 : e.k - 26;
    if (e.strength < 2 || e.strength > e.k - sections - 1)
        throw std::invalid_argument("strength must be in [2, k - section_bits - 1]");
    if (e.plot_index < 0 || e.plot_index > 65535)
        throw std::invalid_argument("plot index must be in [0, 65535]");
    if (e.meta_group < 0 || e.meta_group > 255)
        throw std::invalid_argument("meta group must be in [0, 255]");
    if (e.memo.size() > 255) throw std::invalid_argument("memo exceeds 255 bytes");
    if (e.out_dir.empty()) throw std::invalid_argument("output directory is empty");
    if (e.out_name.empty() || e.out_name == "." || e.out_name == ".." ||
        e.out_name.find_first_of("/\\") != std::string::npos ||
        std::filesystem::path(e.out_name).has_root_path())
        throw std::invalid_argument("output name must be a filename within the output directory");
#ifdef _WIN32
    if (e.out_name.find_first_of("<>:\"|?*") != std::string::npos
        || e.out_name.back() == '.' || e.out_name.back() == ' ')
        throw std::invalid_argument("output name contains characters unsupported by Windows");
#endif
}

namespace {

bool parse_hex(std::string const& s, std::vector<uint8_t>& out)
{
    if (s.size() % 2) return false;
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    out.clear();
    out.reserve(s.size() / 2);
    for (size_t i = 0; i < s.size(); i += 2) {
        int hi = val(s[i]), lo = val(s[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back(uint8_t((hi << 4) | lo));
    }
    return true;
}

bool parse_hex_array32(std::string const& s, std::array<uint8_t, 32>& out)
{
    std::vector<uint8_t> tmp;
    if (!parse_hex(s, tmp) || tmp.size() != 32) return false;
    std::copy(tmp.begin(), tmp.end(), out.begin());
    return true;
}

} // namespace

std::vector<BatchEntry> parse_manifest(std::string const& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open manifest: " + path);

    std::vector<BatchEntry> out;
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        auto const first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream is(line);
        BatchEntry e;
        std::string testnet_s, plot_id_s, memo_s;
        if (!(is >> e.k >> e.strength >> e.plot_index >> e.meta_group
                 >> testnet_s >> plot_id_s >> std::quoted(memo_s)
                 >> std::quoted(e.out_dir) >> std::quoted(e.out_name))) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) +
                                     ": expected 9 whitespace-separated fields "
                                     "(k strength plot_index meta_group testnet "
                                     "plot_id_hex memo_hex out_dir out_name)");
        }
        std::transform(testnet_s.begin(), testnet_s.end(), testnet_s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (testnet_s != "0" && testnet_s != "1" && testnet_s != "true" && testnet_s != "false")
            throw std::invalid_argument("manifest line " + std::to_string(line_no) + ": invalid testnet boolean");
        e.testnet = testnet_s == "1" || testnet_s == "true";
        is >> std::ws;
        if (is.peek() != std::char_traits<char>::eof() && is.peek() != '#')
            throw std::invalid_argument("manifest line " + std::to_string(line_no) + ": trailing fields");
        if (!parse_hex_array32(plot_id_s, e.plot_id)) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) +
                                     ": plot_id must be 64 hex chars");
        }
        if (!parse_hex(memo_s, e.memo) || e.memo.size() > 255) {
            throw std::runtime_error("manifest line " + std::to_string(line_no) +
                                     ": memo invalid hex or > 255 bytes");
        }
        try { validate_batch_entry(e); }
        catch (std::exception const& ex) {
            throw std::invalid_argument("manifest line " + std::to_string(line_no) + ": " + ex.what());
        }
        out.push_back(std::move(e));
    }
    return out;
}

void write_manifest(std::string const& path, std::vector<BatchEntry> const& entries)
{
    if (entries.empty()) throw std::invalid_argument("cannot save an empty plot job");
    auto hex = [](std::span<uint8_t const> bytes) {
        std::string text;
        for (auto b : bytes) {
            text += "0123456789abcdef"[b >> 4];
            text += "0123456789abcdef"[b & 15];
        }
        return text;
    };
    std::ostringstream data;
    data << "# xchplot2 job manifest; contains private plot keys\n";
    for (auto const& e : entries) {
        validate_batch_entry(e);
        if (e.out_dir.find_first_of("\r\n") != std::string::npos ||
            e.out_name.find_first_of("\r\n") != std::string::npos)
            throw std::invalid_argument("manifest paths cannot contain line breaks");
        data << e.k << ' ' << e.strength << ' ' << e.plot_index << ' ' << e.meta_group
             << ' ' << e.testnet << ' ' << hex(e.plot_id) << ' ' << std::quoted(hex(e.memo))
             << ' ' << std::quoted(e.out_dir) << ' ' << std::quoted(e.out_name) << '\n';
    }
    auto const text = data.str();
    // Follow the plot writer's exclusive temporary-file + durability barrier.
    // Publication must not replace a concurrently saved job.
    std::string partial = path + ".partial.XXXXXX";
#ifdef _WIN32
    int const fd = create_private_temp(partial);
#else
    int const fd = ::mkstemp(partial.data());
#endif
    if (fd < 0) throw std::runtime_error("cannot create manifest: " + path);
    struct Cleanup {
        std::string const& path;
        std::FILE* file;
        ~Cleanup() {
            if (file) std::fclose(file);
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{partial, nullptr};
#ifdef _WIN32
    cleanup.file = ::_fdopen(fd, "wb");
    if (!cleanup.file) ::_close(fd);
#else
    cleanup.file = ::fdopen(fd, "wb");
    if (!cleanup.file) ::close(fd);
#endif
    if (!cleanup.file || std::fwrite(text.data(), 1, text.size(), cleanup.file) != text.size()
        || std::fflush(cleanup.file) != 0)
        throw std::runtime_error("cannot write manifest: " + path);
#ifdef _WIN32
    if (::_commit(fd) != 0)
#else
    if (::fsync(fd) != 0)
#endif
        throw std::runtime_error("cannot sync manifest: " + path);
    if (std::fclose(std::exchange(cleanup.file, nullptr)) != 0)
        throw std::runtime_error("cannot close manifest: " + path);
    std::error_code ec;
#ifdef _WIN32
    if (!::MoveFileExW(std::filesystem::path(partial).c_str(),
                       std::filesystem::path(path).c_str(), MOVEFILE_WRITE_THROUGH))
        ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
#elif defined(__linux__)
    if (::renameat2(AT_FDCWD, partial.c_str(), AT_FDCWD, path.c_str(), RENAME_NOREPLACE) != 0)
        ec = std::error_code(errno, std::generic_category());
#else
    std::filesystem::create_hard_link(partial, path, ec);
#endif
    if (ec && (ec != std::errc::file_exists || parse_manifest(path) != entries)) {
        throw std::runtime_error("cannot publish manifest " + path + ": " + ec.message()
                                 + "; use --resume or choose another --manifest path");
    }
#ifndef _WIN32
    auto dir = std::filesystem::path(path).parent_path();
    if (dir.empty()) dir = ".";
    int const parent = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (parent >= 0) { ::fsync(parent); ::close(parent); }
#endif
}

} // namespace pos2gpu
