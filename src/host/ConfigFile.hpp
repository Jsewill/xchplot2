// ConfigFile.hpp — minimal TOML-subset config reader.
//
// What it parses:
//   * Line comments: `#` or `;` to end-of-line.
//   * Section headers: `[section_name]`.
//   * Key-value pairs: `key = value` (whitespace around `=` fine).
//   * String values: quoted `"foo"` or unquoted `foo` (no whitespace
//     allowed in unquoted form — wrap in quotes if it has spaces).
//   * Integer / float / bool: parsed as type when the textual form
//     matches; otherwise the raw string is preserved and the caller
//     can coerce on its own.
//
// What it does NOT parse:
//   * Nested tables, arrays of tables, inline tables.
//   * Multi-line strings (basic or literal).
//   * Datetimes.
//   * Array values (`[1, 2, 3]`). Pass space-or-comma-separated tokens
//     in a string and the caller can split.
//
// Why hand-rolled instead of vendoring toml++: the actual surface
// we need is "structured @argfile" — key=value lookup grouped by
// section. Full TOML is overkill, and vendoring ~5kLOC of library
// for that footprint is more cost than the feature's worth.
//
// Lookup model: `Config::get(section, key)` returns the value as a
// string (caller-coerced). `Config::section(name)` returns the
// subset for that section. `Config::all_sections()` enumerates the
// section names encountered.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace pos2gpu {

class ConfigFile {
public:
    // Parse a config file from `path`. Throws on syntax error with a
    // line-numbered message. Returns an empty ConfigFile (no sections,
    // no keys) when the file doesn't exist — callers can probe
    // ~/.config/xchplot2/config.toml unconditionally without a
    // separate exists() check.
    static ConfigFile load(std::string const& path);

    // Returns the value for `[section] key = ...`. `std::nullopt` if
    // either the section or the key is absent. Section "" is the
    // pre-header (top-of-file) keys, conventionally "[defaults]" by
    // user convention but stored here as "" if literally before any
    // `[section]` header.
    std::optional<std::string> get(std::string const& section,
                                   std::string const& key) const;

    // Convenience: bool / int coercion via the same get() lookup.
    // Bool accepts {true, false, 1, 0, yes, no, on, off} case-
    // insensitive. Int accepts decimal. Returns nullopt on absent
    // or unparseable; the caller is responsible for fall-through.
    std::optional<bool> get_bool(std::string const& section,
                                 std::string const& key) const;
    std::optional<long> get_int (std::string const& section,
                                 std::string const& key) const;

    // Direct access to a section's whole key→value map. Empty map
    // when the section is absent.
    std::map<std::string, std::string> const& section_view(
        std::string const& section) const;

    // Section names actually present in the file (in insertion order).
    std::vector<std::string> const& all_sections() const { return order_; }

    // True when load() returned an empty config — useful for distinguishing
    // "no config file" from "empty config file".
    bool empty() const { return data_.empty(); }

private:
    std::map<std::string, std::map<std::string, std::string>> data_;
    std::vector<std::string> order_;
    static std::map<std::string, std::string> const kEmptySection_;
};

} // namespace pos2gpu
