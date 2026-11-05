// Public, portable harvester inputs. Construction identities stay in private job state.
#pragma once
#include "host/GroupFarming.hpp"
#include "host/Json.hpp"
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <istream>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <type_traits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
extern char** environ;
#endif

namespace group_grow
{
inline constexpr char reader_calibration_method[] = "pos2-group-reference-reader-gsz-1";
struct ReaderCalibrationScope
{
    uint8_t k = 28, strength = 2;
    FarmingReader reader = FarmingReader::stateless;
    std::string method = reader_calibration_method;
    long double margin = 2;
    std::string host_name, captured_at;
    bool operator==(ReaderCalibrationScope const&) const = default;
};
inline void validate_reader_calibration_scope(ReaderCalibrationScope const& scope)
{
    if (scope.k < 18 || scope.k > 28 || (scope.k & 1) || scope.strength < 2 || scope.strength > scope.k - 3 ||
        (scope.reader != FarmingReader::stateless && scope.reader != FarmingReader::cached) ||
        scope.method != reader_calibration_method || !std::isfinite(scope.margin) || scope.margin < 1)
        throw std::runtime_error("unsupported or invalid reader calibration scope");
}
struct FarmingProfileMetadata
{
    std::string source = "assumed", captured_at, chia_version, host_name;
    std::string config_status = "not inspected", inventory_status = "unavailable";
    unsigned detected_logical_cpus = 0;
    uint64_t detected_available_memory_bytes = 0;
    bool inventory_available = false;
    uint64_t inventory_v1_files = 0, inventory_v2_files = 0, inventory_v2_members = 0;
    uint64_t inventory_failed_files = 0, inventory_no_key_files = 0;
    std::optional<ReaderCalibrationScope> reader_calibration;
    std::vector<std::string> assumptions{
        "CPU and memory budgets are model inputs, not Chia defaults or measured spare capacity.",
        "Reader CPU costs and storage service costs are assumed envelopes, not calibration measurements.",
        "Assessment covers this plotting job; other plots and services are not included."};
    bool operator==(FarmingProfileMetadata const&) const = default;
};
struct FarmingProfile : FarmingProfileMetadata
{
    FarmingConfig config;
};
inline void validate_reader_calibration_scope(FarmingProfileMetadata const& profile, FarmingConfig const& config)
{
    if (!profile.reader_calibration) return;
    auto const& scope = *profile.reader_calibration; validate_reader_calibration_scope(scope);
    if (scope.k != config.k || scope.strength != config.strength || scope.reader != config.reader)
        throw std::runtime_error("reader calibration scope differs from the requested k, strength or reader");
}
struct ChiaDetectionOptions
{
    std::filesystem::path root, executable;
    std::optional<unsigned> lookup_workers;
    std::chrono::milliseconds timeout{10000};
    uint64_t max_output_bytes = 16 * 1024 * 1024;
    uint64_t max_memory_bytes = 256 * 1024 * 1024;
};
inline uint64_t farming_profile_metadata_memory(FarmingProfileMetadata const& profile)
{
    uint64_t bytes = sizeof(profile);
    auto add = [&](uint64_t amount) { bytes = amount > UINT64_MAX - bytes ? UINT64_MAX : bytes + amount; };
    auto string = [&](std::string const& value) { add(value.capacity()); add(1); };
    string(profile.source); string(profile.captured_at); string(profile.chia_version); string(profile.host_name);
    string(profile.config_status); string(profile.inventory_status);
    if (profile.assumptions.capacity() > (UINT64_MAX - bytes) / sizeof(std::string)) return UINT64_MAX;
    add(profile.assumptions.capacity() * sizeof(std::string));
    for (auto const& assumption : profile.assumptions) string(assumption);
    if (profile.reader_calibration)
    { string(profile.reader_calibration->method); string(profile.reader_calibration->host_name); string(profile.reader_calibration->captured_at); }
    return bytes;
}
inline uint64_t farming_profile_memory(FarmingProfile const& profile)
{
    auto const bytes = farming_profile_metadata_memory(profile);
    auto const config_bytes = sizeof(FarmingProfile) - sizeof(FarmingProfileMetadata);
    return bytes > UINT64_MAX - config_bytes ? UINT64_MAX : bytes + config_bytes;
}
namespace profile_detail
{
inline void profile_require(bool condition, char const* message)
{ if (!condition) throw std::runtime_error(message); }
inline constexpr size_t profile_bytes = 16 * 1024 * 1024;
inline constexpr size_t config_bytes = 1024 * 1024;
inline constexpr uint64_t parsing_memory_bytes = 256 * 1024 * 1024;
struct Json
{
    enum Kind { null, boolean, number, string, array, object } kind = null;
    std::string scalar;
    std::vector<Json> items;
    std::map<std::string, Json> fields;
};
// This boundary consumes both our public profile and Chia's documented JSON RPC
// output. Bound bytes, nesting and nodes before accepting any model inputs.
class JsonReader
{
    std::string_view text_;
    size_t position_ = 0, nodes_ = 0;
    uint64_t memory_limit_, memory_used_, node_limit_;
    void whitespace()
    { while (position_ < text_.size() && (text_[position_] == ' ' || text_[position_] == '\n' || text_[position_] == '\r' || text_[position_] == '\t')) ++position_; }
    char take()
    { profile_require(position_ < text_.size(), "truncated farming profile JSON"); return text_[position_++]; }
    bool consume(char value)
    { whitespace(); if (position_ == text_.size() || text_[position_] != value) return false; ++position_; return true; }
    unsigned hex()
    {
        unsigned value = 0;
        for (unsigned i = 0; i < 4; ++i)
        {
            char const c = take(); unsigned digit;
            if (c >= '0' && c <= '9') digit = unsigned(c - '0');
            else if (c >= 'a' && c <= 'f') digit = unsigned(c - 'a') + 10;
            else if (c >= 'A' && c <= 'F') digit = unsigned(c - 'A') + 10;
            else throw std::runtime_error("invalid farming profile JSON escape");
            value = (value << 4) | digit;
        }
        return value;
    }
    static void utf8(std::string& out, unsigned value)
    {
        if (value < 0x80) out += char(value);
        else if (value < 0x800) { out += char(0xc0 | (value >> 6)); out += char(0x80 | (value & 63)); }
        else if (value < 0x10000) { out += char(0xe0 | (value >> 12)); out += char(0x80 | ((value >> 6) & 63)); out += char(0x80 | (value & 63)); }
        else { out += char(0xf0 | (value >> 18)); out += char(0x80 | ((value >> 12) & 63)); out += char(0x80 | ((value >> 6) & 63)); out += char(0x80 | (value & 63)); }
    }
    std::string string()
    {
        profile_require(take() == '"', "expected farming profile JSON string"); std::string result;
        for (;;)
        {
            unsigned char const c = static_cast<unsigned char>(take());
            if (c == '"') return result;
            profile_require(c >= 32, "control character in farming profile JSON string");
            if (c != '\\')
            {
                if (c < 0x80) { result += char(c); continue; }
                unsigned continuation = 0, value = 0, minimum = 0;
                if (c >= 0xc2 && c <= 0xdf) { continuation = 1; value = c & 31; minimum = 0x80; }
                else if (c >= 0xe0 && c <= 0xef) { continuation = 2; value = c & 15; minimum = 0x800; }
                else if (c >= 0xf0 && c <= 0xf4) { continuation = 3; value = c & 7; minimum = 0x10000; }
                else throw std::runtime_error("invalid farming profile JSON UTF-8");
                for (unsigned i = 0; i < continuation; ++i)
                { unsigned char const next = static_cast<unsigned char>(take()); profile_require((next & 0xc0) == 0x80, "invalid farming profile JSON UTF-8"); value = (value << 6) | (next & 63); }
                profile_require(value >= minimum && value <= 0x10ffff && (value < 0xd800 || value > 0xdfff), "invalid farming profile JSON UTF-8");
                utf8(result, value); continue;
            }
            switch (take())
            {
            case '"': result += '"'; break;
            case '\\': result += '\\'; break;
            case '/': result += '/'; break;
            case 'b': result += '\b'; break;
            case 'f': result += '\f'; break;
            case 'n': result += '\n'; break;
            case 'r': result += '\r'; break;
            case 't': result += '\t'; break;
            case 'u':
            {
                unsigned value = hex();
                if (value >= 0xd800 && value <= 0xdbff)
                {
                    profile_require(take() == '\\' && take() == 'u', "missing farming profile JSON surrogate");
                    unsigned const low = hex();
                    profile_require(low >= 0xdc00 && low <= 0xdfff, "invalid farming profile JSON surrogate");
                    value = 0x10000 + ((value - 0xd800) << 10) + low - 0xdc00;
                }
                else profile_require(value < 0xdc00 || value > 0xdfff, "invalid farming profile JSON surrogate");
                utf8(result, value); break;
            }
            default: throw std::runtime_error("invalid farming profile JSON escape");
            }
        }
    }
    Json value(unsigned depth)
    {
        profile_require(depth <= 32 && ++nodes_ <= node_limit_, "farming profile JSON exceeds its structure limit");
        // Covers node temporaries, map allocation/key storage and vector growth
        // including the old vector while a larger allocation is constructed.
        static_assert(sizeof(Json) <= 128);
        profile_require(512 <= memory_limit_ - memory_used_, "farming profile JSON exceeds its parsing memory budget");
        memory_used_ += 512;
        whitespace(); profile_require(position_ < text_.size(), "truncated farming profile JSON"); Json result;
        if (text_[position_] == '{')
        {
            ++position_; result.kind = Json::object;
            if (consume('}')) return result;
            do
            {
                whitespace(); auto key = string(); profile_require(consume(':'), "missing farming profile JSON colon");
                auto child = value(depth + 1);
                profile_require(result.fields.emplace(std::move(key), std::move(child)).second, "duplicate farming profile JSON key");
                if (consume('}')) return result;
            } while (consume(','));
            throw std::runtime_error("invalid farming profile JSON object");
        }
        if (text_[position_] == '[')
        {
            ++position_; result.kind = Json::array;
            if (consume(']')) return result;
            do { result.items.push_back(value(depth + 1)); if (consume(']')) return result; } while (consume(','));
            throw std::runtime_error("invalid farming profile JSON array");
        }
        if (text_[position_] == '"') { result.kind = Json::string; result.scalar = string(); return result; }
        for (auto const literal : {std::string_view("true"), std::string_view("false"), std::string_view("null")})
            if (text_.substr(position_, literal.size()) == literal)
            {
                position_ += literal.size(); result.kind = literal == "null" ? Json::null : Json::boolean;
                result.scalar = literal; return result;
            }
        size_t const start = position_;
        if (text_[position_] == '-') ++position_;
        profile_require(position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9', "invalid farming profile JSON value");
        if (text_[position_] == '0') ++position_;
        else while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
        auto digits = [&]
        {
            size_t const before = position_;
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
            profile_require(position_ != before, "invalid farming profile JSON number");
        };
        if (position_ < text_.size() && text_[position_] == '.') { ++position_; digits(); }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E'))
        { ++position_; if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_; digits(); }
        result.kind = Json::number; result.scalar = text_.substr(start, position_ - start); return result;
    }
public:
    explicit JsonReader(std::string_view text, uint64_t memory_limit = parsing_memory_bytes, uint64_t node_limit = 262144)
        : text_(text), memory_limit_(memory_limit), memory_used_(0), node_limit_(std::min(node_limit, memory_limit / 512))
    {
        // Input capture capacity plus decoded strings can each grow beyond
        // their used length. Charge both before allocating parsed structure.
        profile_require(text.size() <= memory_limit / 4, "farming profile JSON exceeds its parsing memory budget");
        memory_used_ = text.size() * 4;
    }
    Json read()
    { auto result = value(0); whitespace(); profile_require(position_ == text_.size(), "trailing farming profile JSON data"); return result; }
};
inline std::string read_text(std::istream& in, uint64_t limit)
{
    std::string text; char buffer[4096];
    while (in.read(buffer, sizeof(buffer)) || in.gcount())
    {
        profile_require(uint64_t(in.gcount()) <= limit - text.size(), "farming profile input exceeds its byte limit");
        text.append(buffer, size_t(in.gcount()));
    }
    profile_require(in.eof() && !in.bad(), "cannot read farming profile input"); return text;
}
inline Json const* field(Json const& object, char const* name)
{ profile_require(object.kind == Json::object, "expected farming profile JSON object"); auto at = object.fields.find(name); return at == object.fields.end() ? nullptr : &at->second; }
inline uint64_t integer(Json const& value)
{
    profile_require(value.kind == Json::number, "expected farming profile integer"); uint64_t result = 0;
    auto const parsed = std::from_chars(value.scalar.data(), value.scalar.data() + value.scalar.size(), result);
    profile_require(parsed.ec == std::errc{} && parsed.ptr == value.scalar.data() + value.scalar.size(), "invalid farming profile integer"); return result;
}
inline long double real(Json const& value)
{
    profile_require(value.kind == Json::number, "expected farming profile number");
    std::istringstream in(value.scalar); in.imbue(std::locale::classic()); long double result = 0; in >> result;
    profile_require(in && in.peek() == std::char_traits<char>::eof() && std::isfinite(result), "invalid farming profile number"); return result;
}
inline std::string text(Json const& value)
{ profile_require(value.kind == Json::string && value.scalar.size() <= 4096, "invalid farming profile text"); return value.scalar; }
inline bool boolean(Json const& value)
{ profile_require(value.kind == Json::boolean, "expected farming profile boolean"); return value.scalar == "true"; }
inline void quote(std::ostream& out, std::string const& value)
{
    pos2gpu::write_json_string(out, value);
}
inline void validate(FarmingConfig const& config)
{ farming_detail::validate(config, {}); }
} // namespace profile_detail

namespace profile_detail
{
inline FarmingProfile parse_farming_profile(Json const& document)
{
    auto const* nested = field(document, "farming_profile"); auto const& object = nested ? *nested : document;
    FarmingProfile result; auto& c = result.config;
    auto const* schema = field(object, "schema_version");
    profile_require(schema && (integer(*schema) == 1 || integer(*schema) == 2), "unsupported or missing farming profile schema_version");
    auto const* model = field(object, "model");
    profile_require(model && text(*model) == farming_model_identity, "farming profile model differs from this build");
    auto set_integer = [&](char const* name, auto& target)
    {
        if (auto p = field(object, name))
        {
            auto const value = integer(*p);
            profile_require(value <= uint64_t(std::numeric_limits<std::remove_reference_t<decltype(target)>>::max()), "farming profile integer overflows");
            target = std::remove_reference_t<decltype(target)>(value);
        }
    };
    auto set_real = [&](char const* name, long double& target) { if (auto p = field(object, name)) target = real(*p); };
    auto set_text = [&](char const* name, std::string& target) { if (auto p = field(object, name)) target = text(*p); };
    if (auto p = field(object, "reader"))
    { auto const value = text(*p); profile_require(value == "stateless" || value == "cached", "invalid farming profile reader"); c.reader = value == "cached" ? FarmingReader::cached : FarmingReader::stateless; }
    if (auto p = field(object, "objective"))
    { auto const value = text(*p); profile_require(value == "energy" || value == "latency", "invalid farming profile objective"); c.objective = value == "latency" ? FarmingObjective::latency : FarmingObjective::energy; }
    set_integer("harvester_cores", c.harvester_cores); set_integer("lookup_workers", c.lookup_workers);
    set_integer("memory_bytes", c.memory_bytes); set_integer("index_cache_bytes", c.index_cache_bytes);
    set_integer("filter_bits", c.filter_bits); set_integer("future_filter_bits", c.future_filter_bits);
    set_integer("max_eligible_groups", c.max_eligible_families_per_resource);
    set_integer("lookup_memory_per_member", c.lookup_memory_per_member);
    set_real("deadline_seconds", c.deadline_seconds); set_real("challenge_seconds", c.challenge_seconds);
    set_real("chain_seconds_per_member", c.chain_seconds_per_member); set_real("index_seconds_per_entry", c.index_seconds_per_entry);
    set_real("compressed_index_bytes_per_entry", c.compressed_index_bytes_per_entry);
    set_real("cpu_work_weight", c.cpu_work_weight); set_real("io_work_weight", c.io_work_weight);
    set_text("source", result.source); set_text("captured_at", result.captured_at); set_text("chia_version", result.chia_version);
    set_text("host_name", result.host_name);
    set_text("config_status", result.config_status); set_text("inventory_status", result.inventory_status);
    set_integer("detected_logical_cpus", result.detected_logical_cpus);
    set_integer("detected_available_memory_bytes", result.detected_available_memory_bytes);
    if (auto p = field(object, "inventory_available")) result.inventory_available = boolean(*p);
    set_integer("inventory_v1_files", result.inventory_v1_files); set_integer("inventory_v2_files", result.inventory_v2_files);
    set_integer("inventory_v2_members", result.inventory_v2_members); set_integer("inventory_failed_files", result.inventory_failed_files);
    set_integer("inventory_no_key_files", result.inventory_no_key_files);
    if (result.inventory_available)
        for (auto name : {"inventory_v1_files", "inventory_v2_files", "inventory_v2_members", "inventory_failed_files", "inventory_no_key_files"})
            profile_require(field(object, name), "incomplete farming profile inventory snapshot");
    if (auto p = field(object, "assumptions"))
    {
        profile_require(p->kind == Json::array && p->items.size() <= 64, "invalid farming profile assumptions");
        result.assumptions.clear(); for (auto const& item : p->items) result.assumptions.push_back(text(item));
    }
    auto const* calibration = field(object, "reader_calibration");
    profile_require((integer(*schema) == 2) == bool(calibration), "reader calibration scope requires farming profile schema_version 2");
    if (calibration)
    {
        profile_require(calibration->kind == Json::object, "invalid reader calibration scope");
        for (auto name : {"k", "strength", "reader", "method", "margin", "host_name", "captured_at"})
            profile_require(field(*calibration, name), "incomplete reader calibration scope");
        ReaderCalibrationScope scope;
        auto const k = integer(*field(*calibration, "k")), strength = integer(*field(*calibration, "strength"));
        profile_require(k <= UINT8_MAX && strength <= UINT8_MAX, "reader calibration parameters overflow");
        scope.k = uint8_t(k); scope.strength = uint8_t(strength);
        auto const reader = text(*field(*calibration, "reader"));
        profile_require(reader == "stateless" || reader == "cached", "invalid calibrated reader");
        scope.reader = reader == "cached" ? FarmingReader::cached : FarmingReader::stateless;
        scope.method = text(*field(*calibration, "method")); scope.margin = real(*field(*calibration, "margin"));
        scope.host_name = text(*field(*calibration, "host_name")); scope.captured_at = text(*field(*calibration, "captured_at"));
        validate_reader_calibration_scope(scope); result.reader_calibration = std::move(scope);
        c.k = result.reader_calibration->k; c.strength = result.reader_calibration->strength;
    }
    validate_reader_calibration_scope(result, c); validate(c); return result;
}
} // namespace profile_detail
inline FarmingProfile read_farming_profile(std::istream& in, uint64_t max_memory_bytes = profile_detail::parsing_memory_bytes)
{
    using namespace profile_detail;
    return parse_farming_profile(JsonReader(read_text(in, std::min<uint64_t>(profile_bytes, max_memory_bytes / 4)), max_memory_bytes).read());
}
inline std::optional<FarmingProfile> read_replot_farming_profile(std::istream& in, uint64_t max_memory_bytes = profile_detail::parsing_memory_bytes)
{
    using namespace profile_detail;
    auto const document = JsonReader(read_text(in, std::min<uint64_t>(profile_bytes, max_memory_bytes / 4)), max_memory_bytes).read();
    // Resource-only migration inputs do not claim an ordinary reader profile.
    // Their resource policy is validated separately by the migration planner.
    auto const* kind = field(document, "kind");
    bool const inventory = kind && kind->kind == Json::string && kind->scalar == "farm-inventory";
    bool const ordinary = field(document, "farming_profile") || field(document, "model") ||
        (field(document, "schema_version") && !inventory);
    auto const* schema = field(document, "schema"); auto const* resources = field(document, "resources");
    auto const* workloads = field(document, "workloads");
    bool const raw_resources = schema && schema->kind == Json::number && resources && resources->kind == Json::array &&
        workloads && workloads->kind == Json::array;
    if (!ordinary && (field(document, "farm_resources") || raw_resources)) return {};
    return parse_farming_profile(document);
}
inline void write_farming_profile_json(std::ostream& out, FarmingProfile const& value)
{
    using profile_detail::quote;
    profile_detail::validate(value.config);
    validate_reader_calibration_scope(value, value.config);
    std::ostringstream text; text.imbue(std::locale::classic());
    text << std::setprecision(std::numeric_limits<long double>::max_digits10);
    auto const& c = value.config;
    text << "{\"schema_version\":" << (value.reader_calibration ? 2 : 1) << ",\"model\":"; quote(text, farming_model_identity);
    text << ",\"reader\":"; quote(text, c.reader == FarmingReader::cached ? "cached" : "stateless");
    text << ",\"objective\":"; quote(text, c.objective == FarmingObjective::latency ? "latency" : "energy");
    text << ",\"harvester_cores\":" << c.harvester_cores << ",\"lookup_workers\":" << c.lookup_workers
        << ",\"memory_bytes\":" << c.memory_bytes << ",\"index_cache_bytes\":" << c.index_cache_bytes
        << ",\"filter_bits\":" << unsigned(c.filter_bits) << ",\"future_filter_bits\":" << unsigned(c.future_filter_bits)
        << ",\"max_eligible_groups\":" << c.max_eligible_families_per_resource
        << ",\"lookup_memory_per_member\":" << c.lookup_memory_per_member
        << ",\"deadline_seconds\":" << c.deadline_seconds << ",\"challenge_seconds\":" << c.challenge_seconds
        << ",\"chain_seconds_per_member\":" << c.chain_seconds_per_member << ",\"index_seconds_per_entry\":" << c.index_seconds_per_entry
        << ",\"compressed_index_bytes_per_entry\":" << c.compressed_index_bytes_per_entry
        << ",\"cpu_work_weight\":" << c.cpu_work_weight << ",\"io_work_weight\":" << c.io_work_weight;
    auto string = [&](char const* name, std::string const& contents) { text << ",\"" << name << "\":"; quote(text, contents); };
    string("source", value.source); string("captured_at", value.captured_at); string("chia_version", value.chia_version);
    string("host_name", value.host_name);
    string("config_status", value.config_status); string("inventory_status", value.inventory_status);
    if (value.reader_calibration)
    {
        auto const& scope = *value.reader_calibration;
        text << ",\"reader_calibration\":{\"k\":" << unsigned(scope.k) << ",\"strength\":" << unsigned(scope.strength)
            << ",\"reader\":"; quote(text, scope.reader == FarmingReader::cached ? "cached" : "stateless");
        text << ",\"method\":"; quote(text, scope.method); text << ",\"margin\":" << scope.margin << ",\"host_name\":";
        quote(text, scope.host_name); text << ",\"captured_at\":"; quote(text, scope.captured_at); text << '}';
    }
    text << ",\"detected_logical_cpus\":" << value.detected_logical_cpus
        << ",\"detected_available_memory_bytes\":" << value.detected_available_memory_bytes
        << ",\"inventory_available\":" << (value.inventory_available ? "true" : "false")
        << ",\"inventory_v1_files\":" << value.inventory_v1_files << ",\"inventory_v2_files\":" << value.inventory_v2_files
        << ",\"inventory_v2_members\":" << value.inventory_v2_members << ",\"inventory_failed_files\":" << value.inventory_failed_files
        << ",\"inventory_no_key_files\":" << value.inventory_no_key_files << ",\"assumptions\":[";
    bool comma = false; for (auto const& assumption : value.assumptions)
    { if (comma) text << ','; comma = true; quote(text, assumption); }
    text << "]}"; out << text.str();
}

namespace profile_detail
{
inline std::wstring windows_argument(std::wstring_view value)
{
    std::wstring result = L"\""; size_t slashes = 0;
    for (wchar_t c : value)
    {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\'); slashes = 0; result += c;
    }
    result.append(slashes * 2, L'\\'); return result + L'"';
}
struct ProcessResult
{ bool started = false, timed_out = false, limit_exceeded = false; int exit_code = -1; std::string output; };
inline ProcessResult run_chia(std::filesystem::path const& executable, std::filesystem::path const& root,
    char const* endpoint, std::chrono::milliseconds timeout, uint64_t limit)
{
    ProcessResult result; auto const deadline = std::chrono::steady_clock::now() + timeout;
#ifdef _WIN32
    HANDLE read = nullptr, write = nullptr; SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    if (!::CreatePipe(&read, &write, &attributes, 0)) return result;
    ::SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
    std::wstring command = windows_argument(executable.wstring()) + L" --root-path " + windows_argument(root.wstring()) + L" rpc harvester ";
    for (char const* c = endpoint; *c; ++c) command += wchar_t(*c);
    STARTUPINFOW startup{}; startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write; startup.hStdError = write; startup.hStdInput = ::GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    result.started = ::CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    ::CloseHandle(write);
    if (!result.started) { ::CloseHandle(read); return result; }
    bool done = false;
    for (;;)
    {
        if (std::chrono::steady_clock::now() >= deadline) { result.timed_out = true; break; }
        DWORD available = 0;
        if (::PeekNamedPipe(read, nullptr, 0, nullptr, &available, nullptr) && available)
        {
            char buffer[4096]; DWORD count = 0;
            if (::ReadFile(read, buffer, std::min<DWORD>(available, sizeof(buffer)), &count, nullptr) && count)
            {
                if (count > limit - result.output.size()) { result.limit_exceeded = true; break; }
                result.output.append(buffer, count); continue;
            }
        }
        done = ::WaitForSingleObject(process.hProcess, 0) == WAIT_OBJECT_0;
        if (done) break;
        ::WaitForSingleObject(process.hProcess, 20);
    }
    if (!done) { ::TerminateProcess(process.hProcess, 1); ::WaitForSingleObject(process.hProcess, INFINITE); }
    DWORD code = 0; if (::GetExitCodeProcess(process.hProcess, &code)) result.exit_code = int(code);
    ::CloseHandle(process.hThread); ::CloseHandle(process.hProcess); ::CloseHandle(read);
#else
    int descriptors[2]; if (::pipe(descriptors)) return result;
    ::fcntl(descriptors[0], F_SETFD, FD_CLOEXEC); ::fcntl(descriptors[1], F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions)) { ::close(descriptors[0]); ::close(descriptors[1]); return result; }
    int error = ::posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO);
    if (!error) error = ::posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDERR_FILENO);
    if (!error) error = ::posix_spawn_file_actions_addclose(&actions, descriptors[0]);
    if (!error) error = ::posix_spawn_file_actions_addclose(&actions, descriptors[1]);
    std::vector<std::string> arguments{executable.string(), "--root-path", root.string(), "rpc", "harvester", endpoint};
    std::vector<char*> argv; for (auto& argument : arguments) argv.push_back(argument.data()); argv.push_back(nullptr);
    pid_t child = 0;
    if (!error) error = ::posix_spawnp(&child, argv.front(), &actions, nullptr, argv.data(), environ);
    ::posix_spawn_file_actions_destroy(&actions); ::close(descriptors[1]);
    if (error) { ::close(descriptors[0]); return result; }
    result.started = true; ::fcntl(descriptors[0], F_SETFL, O_NONBLOCK);
    int status = 0; bool done = false, eof = false;
    for (;;)
    {
        if (std::chrono::steady_clock::now() >= deadline) { result.timed_out = true; break; }
        char buffer[4096]; auto const count = ::read(descriptors[0], buffer, sizeof(buffer));
        if (count > 0)
        {
            if (uint64_t(count) > limit - result.output.size()) { result.limit_exceeded = true; break; }
            result.output.append(buffer, size_t(count)); continue;
        }
        if (!count) eof = true;
        if (!done)
        {
            auto const waited = ::waitpid(child, &status, WNOHANG);
            if (waited == child) done = true;
            else if (waited < 0 && errno != EINTR) break;
        }
        if (done && eof) break;
        pollfd descriptor{descriptors[0], POLLIN, 0}; ::poll(&descriptor, 1, 20);
    }
    if (!done)
    {
        ::kill(child, SIGKILL);
        while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    }
    if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
    ::close(descriptors[0]);
#endif
    return result;
}
inline std::string process_status(ProcessResult const& value)
{
    if (!value.started) return "unavailable: Chia CLI could not be started";
    if (value.timed_out) return "unavailable: Chia RPC timed out";
    if (value.limit_exceeded) return "unavailable: Chia RPC output exceeded its byte limit";
    if (value.exit_code) return "unavailable: Chia RPC failed";
    return {};
}
struct ChiaConfig
{ std::optional<unsigned> workers; bool local_rpc = false; std::string rpc_hostname, status; };
inline std::string_view trim(std::string_view value)
{
    while (!value.empty() && (value.front() == ' ' || value.front() == '\r')) value.remove_prefix(1);
    while (!value.empty() && (value.back() == ' ' || value.back() == '\r')) value.remove_suffix(1);
    return value;
}
// Intentionally not a YAML implementation: only conventional block mappings
// with a plain integer num_threads are accepted. Aliases/merges/flow mappings
// that could change these values leave detection explicitly unsupported.
inline ChiaConfig read_chia_config(std::istream& input, uint64_t max_memory_bytes = parsing_memory_bytes)
{
    ChiaConfig result; result.status = "unsupported: expected conventional harvester.num_threads YAML scalar";
    auto unsupported = [&] { result.workers.reset(); result.local_rpc = false; result.rpc_hostname.clear(); return result; };
    auto contents = read_text(input, std::min<uint64_t>(config_bytes, max_memory_bytes / 4)); std::istringstream lines(std::move(contents));
    bool harvester = false, seen_harvester = false, seen_workers = false, seen_host = false;
    size_t direct_indent = 0;
    for (std::string line; std::getline(lines, line);)
    {
        auto value = trim(line); if (value.empty() || value.front() == '#') continue;
        if (value.front() == '\t' || line.find('\t') != line.npos || value == "---" || value == "..." || value.starts_with("%")) return unsupported();
        size_t const indent = line.find_first_not_of(' ');
        if (!indent)
        {
            harvester = false;
            auto const colon = value.find(':');
            if (colon == value.npos || value.substr(0, colon).find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != value.npos)
                return unsupported();
            if (value.starts_with("<<:")) return unsupported();
            if (value.starts_with("harvester:"))
            {
                if (seen_harvester) return unsupported();
                seen_harvester = true;
                auto const tail = trim(value.substr(10));
                if (!tail.empty() && tail.front() != '#') return unsupported();
                harvester = true; direct_indent = 0;
            }
            else if (value.starts_with("self_hostname:"))
            {
                if (seen_host) return unsupported();
                seen_host = true;
                auto host = trim(value.substr(14));
                if (host.starts_with("&self_hostname ")) host = trim(host.substr(15));
                auto const comment = host.find(" #"); if (comment != host.npos) host = trim(host.substr(0, comment));
                if (host.size() >= 2 && ((host.front() == '"' && host.back() == '"') || (host.front() == '\'' && host.back() == '\'')))
                    host = host.substr(1, host.size() - 2);
                if (host.empty() || host.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:") != host.npos) return unsupported();
                result.rpc_hostname = host;
                result.local_rpc = host == "localhost" || host == "127.0.0.1" || host == "::1";
            }
            continue;
        }
        if (!harvester) continue;
        if (!direct_indent) direct_indent = indent;
        if (indent < direct_indent) return unsupported();
        if (indent != direct_indent) continue;
        if (!value.starts_with("- "))
        {
            auto const colon = value.find(':');
            if (colon == value.npos || value.substr(0, colon).find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != value.npos)
                return unsupported();
        }
        if (value.starts_with("<<:")) return unsupported();
        if (!value.starts_with("num_threads:")) continue;
        if (seen_workers) return unsupported();
        seen_workers = true;
        if (value.size() <= 12 || value[12] != ' ') return unsupported();
        auto worker_text = trim(value.substr(12));
        auto const comment = worker_text.find(" #"); if (comment != worker_text.npos) worker_text = trim(worker_text.substr(0, comment));
        unsigned workers = 0; auto const parsed = std::from_chars(worker_text.data(), worker_text.data() + worker_text.size(), workers);
        if (parsed.ec != std::errc{} || parsed.ptr != worker_text.data() + worker_text.size() || !workers || workers > 1024) return unsupported();
        result.workers = workers;
    }
    if (!seen_harvester || !seen_workers) { result.workers.reset(); return result; }
    result.status = "observed harvester.num_threads; running service may require restart after config changes";
    return result;
}
inline unsigned logical_cpus()
{
#ifdef _WIN32
    DWORD_PTR process = 0, system = 0;
    if (::GetProcessAffinityMask(::GetCurrentProcess(), &process, &system)) return unsigned(std::popcount(process));
#elif defined(__linux__)
    cpu_set_t cpus; CPU_ZERO(&cpus);
    if (!::sched_getaffinity(0, sizeof(cpus), &cpus)) return unsigned(CPU_COUNT(&cpus));
#endif
    return std::thread::hardware_concurrency();
}
inline uint64_t available_memory()
{
#ifdef _WIN32
    MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
    if (::GlobalMemoryStatusEx(&memory)) return memory.ullAvailPhys;
#elif defined(__linux__)
    std::ifstream in("/proc/meminfo"); std::string key, unit; uint64_t value;
    while (in >> key >> value >> unit)
        if (key == "MemAvailable:" && value <= UINT64_MAX / 1024) return value * 1024;
#endif
    return 0;
}
inline std::string host_name()
{
#ifdef _WIN32
    wchar_t name[MAX_COMPUTERNAME_LENGTH + 1]{}; DWORD size = MAX_COMPUTERNAME_LENGTH + 1;
    if (!::GetComputerNameW(name, &size)) return {};
    int const bytes = ::WideCharToMultiByte(CP_UTF8, 0, name, int(size), nullptr, 0, nullptr, nullptr);
    if (!bytes) return {};
    std::string result(size_t(bytes), '\0');
    if (!::WideCharToMultiByte(CP_UTF8, 0, name, int(size), result.data(), bytes, nullptr, nullptr)) return {};
    return result;
#else
    char name[256]{};
    if (::gethostname(name, sizeof(name) - 1)) return {};
    return name;
#endif
}
inline std::filesystem::path chia_root(std::filesystem::path supplied)
{
    if (supplied.empty()) { if (auto value = std::getenv("CHIA_ROOT")) supplied = value; }
#ifdef _WIN32
    auto const home = std::getenv("USERPROFILE");
#else
    auto const home = std::getenv("HOME");
#endif
    if (supplied.empty() && home && *home) supplied = std::filesystem::path(home) / ".chia/mainnet";
    auto const encoded = supplied.generic_u8string(); std::string const value(encoded.begin(), encoded.end());
    if (home && value == "~") supplied = std::filesystem::path(home);
    else if (home && value.starts_with("~/")) supplied = std::filesystem::path(home) / value.substr(2);
    return supplied.empty() ? supplied : std::filesystem::absolute(supplied).lexically_normal();
}
inline void inventory(Json const& object, FarmingProfile& profile)
{
    auto const* success = field(object, "success"); profile_require(success && boolean(*success), "Chia inventory RPC was unsuccessful");
    auto const* plots = field(object, "plots"); profile_require(plots && plots->kind == Json::array, "missing Chia plot inventory");
    uint64_t v1 = 0, v2 = 0, members = 0;
    for (auto const& plot : plots->items)
    {
        auto const* size = field(plot, "size"); profile_require(size, "missing Chia inventory plot size"); auto const encoded = integer(*size);
        profile_require(encoded <= 255, "invalid Chia inventory plot size");
        if (encoded & 0x80)
        {
            profile_require((encoded & 0x7f) >= 2 && (encoded & 0x7f) <= 17, "unsupported Chia inventory plot strength");
            auto const* group_size = field(plot, "group_size");
            profile_require(group_size, "Chia inventory predates plot-group membership support"); auto const count = integer(*group_size);
            profile_require(count && count <= 65535 && count <= UINT64_MAX - members, "invalid Chia inventory group_size");
            ++v2; members += count;
        }
        else { profile_require(encoded >= 18 && encoded <= 50, "unsupported Chia inventory plot size"); ++v1; }
    }
    auto count = [&](char const* name)
    {
        auto const* list = field(object, name); profile_require(list && list->kind == Json::array, "missing Chia inventory failure list");
        for (auto const& path : list->items) profile_require(path.kind == Json::string, "invalid Chia inventory failure list");
        return uint64_t(list->items.size());
    };
    profile.inventory_failed_files = count("failed_to_open_filenames"); profile.inventory_no_key_files = count("not_found_filenames");
    profile.inventory_v1_files = v1; profile.inventory_v2_files = v2; profile.inventory_v2_members = members;
    profile.inventory_available = true;
    profile.inventory_status = "loaded-plot snapshot only; refresh completion and outside workload costs are unknown";
}
} // namespace profile_detail

inline FarmingProfile detect_farming_profile(ChiaDetectionOptions const& options = {})
{
    using namespace profile_detail;
    profile_require(options.timeout.count() > 0 && options.max_output_bytes && options.max_output_bytes <= 64 * 1024 * 1024 && options.max_memory_bytes,
        "invalid Chia detection timeout, output limit or parsing memory budget");
    FarmingProfile result; result.source = "local-detection";
    result.assumptions.push_back("Local hardware observations use this process's affinity and available memory; they do not measure the running harvester's grant or spare capacity.");
    auto now = std::time(nullptr); std::tm utc{};
#ifdef _WIN32
    ::gmtime_s(&utc, &now);
#else
    ::gmtime_r(&now, &utc);
#endif
    char timestamp[32]{}; std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc); result.captured_at = timestamp;
    result.host_name = host_name();
    result.detected_logical_cpus = logical_cpus(); result.detected_available_memory_bytes = available_memory();
    if (result.detected_logical_cpus) result.config.harvester_cores = std::min(result.config.harvester_cores, result.detected_logical_cpus);
    if (options.lookup_workers)
    {
        profile_require(*options.lookup_workers && *options.lookup_workers <= 1024, "invalid explicit harvester lookup worker budget");
        result.config.lookup_workers = *options.lookup_workers;
    }
    auto const root = chia_root(options.root);
    if (root.empty()) { result.config_status = "unavailable: Chia root could not be located"; return result; }
    std::ifstream config_file(root / "config/config.yaml");
    if (!config_file) { result.config_status = "unavailable: Chia config was not found or could not be read"; return result; }
    ChiaConfig config;
    try { config = read_chia_config(config_file, options.max_memory_bytes); }
    catch (std::exception const&) { result.config_status = "unsupported: Chia config exceeds supported detection input"; return result; }
    result.config_status = config.status;
    if (config.workers && !options.lookup_workers) result.config.lookup_workers = *config.workers;
    if (options.lookup_workers) result.config_status += "; using explicit lookup worker budget";
    if (!config.local_rpc) { result.inventory_status = "unavailable: Chia self_hostname is not a supported loopback address"; return result; }
    auto executable = options.executable.empty() ? std::filesystem::path("chia") : options.executable;
    uint64_t const capture_limit = std::min(options.max_output_bytes, options.max_memory_bytes / 4);
    if (!capture_limit) { result.inventory_status = "unavailable: detection memory budget is too small"; return result; }
    auto version = run_chia(executable, root, "get_version", options.timeout, capture_limit);
    result.inventory_status = process_status(version); if (!result.inventory_status.empty()) return result;
    try
    {
        auto const document = JsonReader(version.output, options.max_memory_bytes).read(); auto const* success = field(document, "success");
        profile_require(success && boolean(*success), "Chia version RPC was unsuccessful");
        auto const* value = field(document, "version"); profile_require(value, "missing Chia version"); result.chia_version = text(*value);
    }
    catch (std::exception const& error)
    {
        result.inventory_status = std::string_view(error.what()).find("memory budget") != std::string_view::npos ?
            "unavailable: Chia version response exceeds the detection parsing memory budget" : "unavailable: invalid Chia version RPC response";
        return result;
    }
    std::string{}.swap(version.output);
    auto const plots = run_chia(executable, root, "get_plots", options.timeout, capture_limit);
    result.inventory_status = process_status(plots); if (!result.inventory_status.empty()) return result;
    try { inventory(JsonReader(plots.output, options.max_memory_bytes).read(), result); }
    catch (std::exception const& error)
    {
        result.inventory_status = std::string_view(error.what()).find("memory budget") != std::string_view::npos ?
            "unavailable: Chia inventory exceeds the detection parsing memory budget" : "unavailable: invalid or unsupported Chia plot inventory response";
    }
    return result;
}
} // namespace group_grow
