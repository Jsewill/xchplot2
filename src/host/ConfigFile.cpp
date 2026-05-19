// ConfigFile.cpp — implementation of the TOML-subset config reader.
//
// Parser is a single-pass, line-oriented walk: trim → comment-strip →
// dispatch on first character (`[` section, otherwise key = value).
// Quoted strings handle backslash-escaped `"` and `\\` only — no
// unicode escapes, no multi-line variants. Unquoted values stop at
// the first whitespace or comment marker.

#include "host/ConfigFile.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pos2gpu {

std::map<std::string, std::string> const ConfigFile::kEmptySection_{};

namespace {

inline void trim(std::string& s)
{
    auto const not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

// Strip an unquoted-comment tail (`#` or `;` to end of line). Comments
// inside `"..."` strings are NOT stripped. Returns the cleaned line.
std::string strip_comment(std::string const& line)
{
    bool in_string = false;
    bool escape    = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char const c = line[i];
        if (in_string) {
            if (escape) { escape = false; continue; }
            if (c == '\\') { escape = true; continue; }
            if (c == '"')  { in_string = false; continue; }
        } else {
            if (c == '"')                    in_string = true;
            else if (c == '#' || c == ';')   return line.substr(0, i);
        }
    }
    return line;
}

// Parse a quoted string starting at line[start] (line[start] == '"').
// Sets `end` to one past the closing quote. Handles `\"` and `\\`.
std::string parse_quoted(std::string const& line, size_t start, size_t& end)
{
    std::string out;
    out.reserve(line.size() - start);
    bool escape = false;
    for (size_t i = start + 1; i < line.size(); ++i) {
        char const c = line[i];
        if (escape) {
            out.push_back(c);
            escape = false;
        } else if (c == '\\') {
            escape = true;
        } else if (c == '"') {
            end = i + 1;
            return out;
        } else {
            out.push_back(c);
        }
    }
    throw std::runtime_error("unterminated quoted string");
}

} // namespace

ConfigFile ConfigFile::load(std::string const& path)
{
    ConfigFile cfg;
    std::ifstream in(path);
    if (!in) return cfg;  // missing file → empty config (intentional)

    std::string section = "";  // top-of-file keys live in section ""
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        std::string s = strip_comment(line);
        trim(s);
        if (s.empty()) continue;

        // [section] header
        if (s.front() == '[') {
            if (s.back() != ']') {
                throw std::runtime_error(
                    path + ":" + std::to_string(line_no) +
                    ": unterminated section header");
            }
            section = s.substr(1, s.size() - 2);
            trim(section);
            if (section.empty()) {
                throw std::runtime_error(
                    path + ":" + std::to_string(line_no) +
                    ": empty section name");
            }
            if (cfg.data_.find(section) == cfg.data_.end()) {
                cfg.order_.push_back(section);
            }
            cfg.data_[section];  // ensure entry exists even if section is empty
            continue;
        }

        // key = value
        auto const eq = s.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_no) +
                ": expected `key = value` or `[section]`");
        }
        std::string key = s.substr(0, eq);
        trim(key);
        if (key.empty()) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_no) +
                ": empty key before `=`");
        }

        // Find start of value (after `=` + any whitespace)
        size_t v_start = eq + 1;
        while (v_start < s.size() && std::isspace(static_cast<unsigned char>(s[v_start]))) ++v_start;
        std::string value;
        if (v_start < s.size() && s[v_start] == '"') {
            size_t v_end = 0;
            try {
                value = parse_quoted(s, v_start, v_end);
            } catch (std::exception const& ex) {
                throw std::runtime_error(
                    path + ":" + std::to_string(line_no) + ": " + ex.what());
            }
            // After the closing quote we expect only whitespace (the
            // comment was already stripped). Anything else is a typo.
            while (v_end < s.size() && std::isspace(static_cast<unsigned char>(s[v_end]))) ++v_end;
            if (v_end != s.size()) {
                throw std::runtime_error(
                    path + ":" + std::to_string(line_no) +
                    ": trailing characters after quoted value");
            }
        } else {
            value = s.substr(v_start);
            trim(value);
        }

        if (cfg.data_.find(section) == cfg.data_.end() && !section.empty()) {
            cfg.order_.push_back(section);
        }
        cfg.data_[section][std::move(key)] = std::move(value);
    }
    return cfg;
}

std::optional<std::string> ConfigFile::get(std::string const& section,
                                           std::string const& key) const
{
    auto const sec_it = data_.find(section);
    if (sec_it == data_.end()) return std::nullopt;
    auto const k_it = sec_it->second.find(key);
    if (k_it == sec_it->second.end()) return std::nullopt;
    return k_it->second;
}

std::optional<bool> ConfigFile::get_bool(std::string const& section,
                                         std::string const& key) const
{
    auto const raw = get(section, key);
    if (!raw) return std::nullopt;
    std::string s = *raw;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (s == "true"  || s == "1" || s == "yes" || s == "on")  return true;
    if (s == "false" || s == "0" || s == "no"  || s == "off") return false;
    return std::nullopt;
}

std::optional<long> ConfigFile::get_int(std::string const& section,
                                        std::string const& key) const
{
    auto const raw = get(section, key);
    if (!raw) return std::nullopt;
    try {
        size_t pos = 0;
        long const v = std::stol(*raw, &pos);
        if (pos != raw->size()) return std::nullopt;
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

std::map<std::string, std::string> const& ConfigFile::section_view(
    std::string const& section) const
{
    auto const it = data_.find(section);
    if (it == data_.end()) return kEmptySection_;
    return it->second;
}

} // namespace pos2gpu
