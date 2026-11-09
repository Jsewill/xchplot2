// Observations from an explicitly selected installed harvester. The CLI uses
// its configured private-CA client authentication and explicit local service bindings.
#pragma once
#include "host/GroupFarmInventory.hpp"
#include "host/GroupFarmAdapterProcess.hpp"
#include <cctype>

namespace group_grow::farm
{
struct FarmServiceUnavailable : std::runtime_error
{ using std::runtime_error::runtime_error; };
inline void require_service_available(bool condition, std::string const& reason)
{ if (!condition) throw FarmServiceUnavailable(reason); }
enum class HarvesterAdapterKind { chia, gigahorse, drchia };
struct FarmServiceBinding
{
    std::filesystem::path root;
    std::string rpc_host, host, node_id, farmer_node_id;
};
struct HarvesterAdapterOptions;
class RevalidationReaderAllowance
{
    using Reader = std::tuple<std::string_view, std::string_view, std::string_view>;
    FarmInventory const& inventory_;
    std::map<InventoryPlot const*, ReplotReaderObservation const*> expected_;
    std::map<std::string_view, std::set<Reader>, std::less<>> readers_;
public:
    static uint64_t memory_bound(FarmInventory const& inventory, std::span<ReplotReaderObservation const> observations,
        std::span<uint64_t const> nonactive)
    {
        using namespace resource_detail;
        return add(sizeof(RevalidationReaderAllowance) + 4096, add(multiply(inventory.plots.size(), 256),
            add(multiply(observations.size(), 128), multiply(nonactive.size(), 128))));
    }
    RevalidationReaderAllowance(FarmInventory const& inventory, std::span<ReplotReaderObservation const> observations,
        std::span<uint64_t const> native_audited_nonactive, InventoryLimits const& limits = {}) : inventory_(inventory)
    {
        inventory_detail::require(observations.size() <= inventory.roots.size() && native_audited_nonactive.size() <= inventory.plots.size() &&
            memory_bound(inventory, observations, native_audited_nonactive) <= limits.memory_bytes, "revalidation reader allowance exceeds its retained grant");
        std::map<size_t, ReplotReaderObservation const*> roots;
        for (auto const& observation : observations)
        {
            inventory_detail::require(observation.root < inventory.roots.size() && roots.emplace(size_t(observation.root), &observation).second &&
                (observation.reader == "chia" || observation.reader == "gigahorse" || observation.reader == "drchia") &&
                !observation.version.empty() && inventory_detail::hex(observation.identity, 32) && !observation.source.empty() && !observation.captured_at.empty(),
                "invalid or duplicate current revalidation reader observation");
        }
        for (auto index : native_audited_nonactive)
        {
            inventory_detail::require(index < inventory.plots.size(), "revalidation reader allowance has an invalid native-audited source");
            auto const& plot = inventory.plots[size_t(index)]; auto at = roots.find(plot.root);
            inventory_detail::require(at != roots.end() && expected_.emplace(&plot, at->second).second,
                "native-audited nonactive source lacks a unique current reader observation");
        }
        for (auto const& plot : inventory.plots)
        {
            if (auto current = expected(plot)) readers_[plot.harvester].emplace(current->identity, current->version, current->reader);
            else if (!plot.reader_identity.empty()) readers_[plot.harvester].emplace(plot.reader_identity, plot.reader_version, plot.reader);
        }
    }
    FarmInventory const& inventory() const { return inventory_; }
    ReplotReaderObservation const* expected(InventoryPlot const& plot) const
    { auto at = expected_.find(&plot); return at == expected_.end() ? nullptr : at->second; }
    template<class Check> void for_each_reader(std::string const& route, Check check) const
    {
        if (route.empty()) for (auto const& [id, readers] : readers_)
        { (void)id; for (auto const& reader : readers) std::apply(check, reader); }
        else
        {
            if (auto at = readers_.find(""); at != readers_.end()) for (auto const& reader : at->second) std::apply(check, reader);
            if (auto at = readers_.find(route); at != readers_.end()) for (auto const& reader : at->second) std::apply(check, reader);
        }
    }
};
class RetirementConfigurationAllowance
{
    FarmInventory const& inventory_;
    std::map<std::string, std::set<std::string>> directories_;
    std::map<std::string, std::set<std::string>> retained_;
    uint64_t limit_;
public:
    static uint64_t memory_bound(FarmInventory const& inventory, std::span<FarmingDirectory const> targets = {})
    {
        uint64_t bytes = sizeof(RetirementConfigurationAllowance) + 4096;
        for (auto const& root : inventory.roots)
        {
            inventory_detail::require(bytes <= UINT64_MAX - 4096 && root.path.capacity() <= (UINT64_MAX - bytes - 4096) / 2, "retirement allowance memory bound overflows");
            bytes += 2 * root.path.capacity() + 4096;
        }
        for (auto const& target : targets)
        {
            inventory_detail::require(bytes <= UINT64_MAX - 4096 && target.directory.capacity() <= (UINT64_MAX - bytes - 4096) / 2, "retirement allowance target memory bound overflows");
            bytes += 2 * target.directory.capacity() + 4096;
        }
        return bytes;
    }
    explicit RetirementConfigurationAllowance(FarmInventory const& inventory, InventoryLimits const& limits = {}) : inventory_(inventory), limit_(limits.memory_bytes)
    { inventory_detail::require(memory_bound(inventory) <= limits.memory_bytes, "retirement allowance exceeds its retained grant"); }
    void authorize(HarvesterAdapterOptions const& options, InventoryRoot const& root);
    void retain(HarvesterAdapterOptions const& options, std::string const& directory);
    FarmInventory const& inventory() const { return inventory_; }
    bool permits(std::string const& configuration, std::string const& directory) const
    {
        auto held = retained_.find(configuration);
        if (held != retained_.end()) for (auto const& target : held->second)
        {
            auto const parent = file_io::native_path(directory).lexically_normal(), child = file_io::native_path(target).lexically_normal();
            auto part = parent.begin(), descendant = child.begin(); bool contains = true;
            for (; part != parent.end(); ++part, ++descendant)
            {
                if (descendant == child.end()) { contains = false; break; }
#ifdef _WIN32
                auto left = part->native(), right = descendant->native();
                std::replace(left.begin(), left.end(), L'/', L'\\'); std::replace(right.begin(), right.end(), L'/', L'\\');
                if (::CompareStringOrdinal(left.data(), int(left.size()), right.data(), int(right.size()), TRUE) != CSTR_EQUAL)
                { contains = false; break; }
#else
                if (*part != *descendant) { contains = false; break; }
#endif
            }
            if (contains) return false;
        }
        auto at = directories_.find(configuration); return at != directories_.end() && at->second.contains(directory);
    }
};
struct HarvesterAdapterOptions
{
    HarvesterAdapterKind kind = HarvesterAdapterKind::chia;
    std::filesystem::path executable, root;
    std::string host, harvester, solver, rpc_host;
    std::map<std::string, FarmServiceBinding> services;
    // Local execution authority, derived only after the private retirement
    // journal has been locked and validated. Never imported from a review.
    RetirementConfigurationAllowance const* retirement = nullptr;
    // Bound by the caller only after its locked native audit proves these
    // exact indices are durable nonactive sources. Public mappings alone
    // never authorize a different reader for an active source.
    RevalidationReaderAllowance const* revalidation = nullptr;
    // Private operation budget. The caller separately reserves retained
    // catalog/plan memory before offering this additional process/parse grant.
    InventoryLimits limits;
    uint64_t max_peak_age_seconds = 600;
};
inline void RetirementConfigurationAllowance::authorize(HarvesterAdapterOptions const& options, InventoryRoot const& root)
{
    inventory_detail::require(std::find_if(inventory_.roots.begin(), inventory_.roots.end(), [&](auto const& reviewed)
        { return &reviewed == &root; }) != inventory_.roots.end(), "retirement allowance requires a borrowed reviewed source root");
    auto path = options.root.lexically_normal();
    if (path != path.root_path() && path.filename().empty()) path = path.parent_path();
    inventory_detail::require(path.is_absolute(), "retirement allowance requires the explicit local reader root");
    if (auto binding = options.services.find("harvester"); binding != options.services.end())
        inventory_detail::require(binding->second.root == options.root, "retirement allowance source reader binding is inconsistent");
    directories_[stream_state::digest(file_io::path_text(path))].insert(root.path);
}
inline void RetirementConfigurationAllowance::retain(HarvesterAdapterOptions const& options, std::string const& directory)
{
    inventory_detail::bounded_text(directory); auto const path = file_io::native_path(directory);
    inventory_detail::require(path.is_absolute() && file_io::path_text(path.lexically_normal()) == directory, "retained target directory must be normalized and absolute");
    auto root = options.root.lexically_normal();
    if (root != root.root_path() && root.filename().empty()) root = root.parent_path();
    retained_[stream_state::digest(file_io::path_text(root))].insert(directory);
    uint64_t bytes = memory_bound(inventory_);
    for (auto const& [configuration, directories] : retained_)
    {
        (void)configuration;
        for (auto const& entry : directories)
        { inventory_detail::require(bytes <= limit_ && entry.capacity() <= (limit_ - bytes) / 2 && 4096 <= limit_ - bytes - 2 * entry.capacity(), "retained target directories exceed their memory grant"); bytes += 2 * entry.capacity() + 4096; }
    }
}
inline std::map<std::string, std::string> target_service_hosts(HarvesterAdapterOptions const& options)
{
    std::map<std::string, std::string> result;
    auto const local = profile_detail::host_name();
    for (auto const* service : {"harvester", "farmer", "full_node", "solver"})
    {
        auto found = options.services.find(service);
        result.emplace(service, found == options.services.end() ? (options.host.empty() ? local : options.host) : found->second.host);
    }
    return result;
}
struct FarmAdapterRoute
{
    std::string id;
    HarvesterAdapterOptions options;
    std::vector<std::string> source_roots;
};
struct FarmAdapterTable
{
    std::string target_route;
    std::vector<FarmAdapterRoute> routes;
};
// Runtime-only allowance for exact publications already authorized by the
// private journal. A future planned name is not a loaded-workload reservation.
using PublishedPlotAllowance = std::function<bool(std::string_view, std::string_view)>;
struct ReplacementExpectation
{
    std::filesystem::path path;
    InventoryFileIdentity identity;
    std::string group_id, farmer_public_key, pool_public_key, pool_contract_puzzle_hash, network, genesis_challenge;
    std::string target_route, reader_version, reader_identity, network_version, farmer_version, solver_version;
    uint8_t k = 0, strength = 0, meta_group = 0;
    uint16_t members = 0;
};
struct ReplacementReadiness
{
    bool ready = false, retryable = false;
    std::string status = "unavailable", reason, version, network, host, captured_at, method;
    InventoryFileIdentity identity;
};
struct TargetNetworkReadiness
{
    bool ready = false, retryable = false;
    NetworkProjection observed;
    std::string reason;
};
struct TargetServiceExpectation
{
    std::string farmer_public_key, pool_public_key, pool_contract_puzzle_hash;
};
struct TargetPoolState
{
    std::string mode, pool_url, launcher_id, owner_public_key, target_puzzle_hash;
    uint64_t protocol_version = 0, difficulty = 0, authentication_token_timeout = 0;
    bool external = false;
    bool operator==(TargetPoolState const&) const = default;
};
struct OtherHarvesterObservation
{
    std::string node_id, peer_host;
    uint64_t files = 0, members = 0;
};
struct TargetPrerequisites
{
    bool ready = false, retryable = false;
    NetworkProjection observed;
    std::string reason, pool_mode;
    TargetPoolState pool;
    std::vector<OtherHarvesterObservation> other_harvesters;
};
inline uint64_t target_readiness_supervision_bytes(InventoryLimits const& limits)
{
    inventory_detail::validate_limits(limits);
    // Two retained DOMs: G/4. Two indexes each reserve 4096 bytes per
    // entry, at most G each. One concurrent RPC: child G/2, capture growth
    // G/32, new parse/DOM G/8. Total 93G/32 < 3G; capped peer views fit MiB.
    inventory_detail::require(limits.memory_bytes <= (UINT64_MAX - 1024 * 1024) / 3, "target readiness memory bound overflows");
    return 3 * limits.memory_bytes + 1024 * 1024;
}
// Runtime-only evidence returned by the private migration publication protocol.
// Public inventory/profile JSON cannot import this or acknowledge readiness.
struct ReplacementPublication
{
    std::filesystem::path path;
    InventoryFileIdentity identity;
    std::string review_digest;
    uint64_t batch = 0;
};
struct SourceRetirementObservation
{
    bool no_loaded_record = false;
    std::string source, captured_at, reason;
    std::chrono::steady_clock::time_point observed;
};
struct SourceRetirementSnapshot
{
    std::vector<bool> no_loaded_records, lookup_bounded;
    std::string source, captured_at;
    std::chrono::steady_clock::time_point observed;
};
namespace adapter_detail
{
using inventory_detail::require;
using Json = profile_detail::Json;
inline std::string hex_string(Json const& value, size_t bytes)
{
    auto text = profile_detail::text(value); if (text.starts_with("0x")) text.erase(0, 2);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    require(inventory_detail::hex(text, bytes), "native farm adapter returned an invalid public identifier"); return text;
}
// Canonical bounded public RPC projection, including loaded population and
// failure lists. Retain only its digest between the phase's capture/final fence.
inline std::string registry_identity(Json const& document)
{
    struct sha256 hash; sha256_init(&hash);
    auto append = [&](std::string_view value) { sha256_update(&hash, value.data(), value.size()); };
    auto visit = [&](auto&& self, Json const& value) -> void
    {
        append(std::to_string(int(value.kind))); append(":"); append(std::to_string(value.scalar.size())); append(":"); append(value.scalar);
        for (auto const& [key, child] : value.fields)
        { append(std::to_string(key.size())); append(":"); append(key); self(self, child); }
        append("["); for (auto const& child : value.items) self(self, child); append("]");
    };
    visit(visit, document); std::array<uint8_t, 32> digest{}; sha256_sum(&hash, digest.data()); return group_tools::hex(digest);
}
inline std::string version(Json const& value)
{ return inventory_detail::string(value, "version"); }
inline char const* name(HarvesterAdapterKind kind)
{
    switch (kind) { case HarvesterAdapterKind::chia: return "chia"; case HarvesterAdapterKind::gigahorse: return "gigahorse"; case HarvesterAdapterKind::drchia: return "drchia"; }
    throw std::runtime_error("unsupported native harvester adapter");
}
inline std::string group_id_from_loaded_key(std::string const& public_key, std::string const& pool, uint8_t strength)
{
    require(inventory_detail::hex(public_key, 48) && (inventory_detail::hex(pool, 32) || inventory_detail::hex(pool, 48)), "invalid loaded plot public key/route");
    // Same streamable encoding as the pinned keygen-rs group identity; its
    // protocol-crate parity tests cover this exact order and strength byte.
    auto key = group_tools::hex(public_key), route = group_tools::hex(pool);
    std::vector<uint8_t> input; input.reserve(1 + key.size() + route.size()); input.push_back(strength);
    input.insert(input.end(), key.begin(), key.end()); input.insert(input.end(), route.begin(), route.end()); return group_tools::hex(grow_digest(input));
}
inline std::string legacy_id_from_loaded_key(std::string const& public_key, std::string const& pool)
{
    require(inventory_detail::hex(public_key, 48) && (inventory_detail::hex(pool, 32) || inventory_detail::hex(pool, 48)), "invalid native public plot key/route");
    // chia.types.blockchain_format.proof_of_space.calculate_plot_id_pk/ph.
    auto bytes = group_tools::hex(pool), key = group_tools::hex(public_key); bytes.insert(bytes.end(), key.begin(), key.end()); return group_tools::hex(grow_digest(bytes));
}
inline std::string implementation_identity(std::filesystem::path const& path, InventoryLimits const& limits,
    std::chrono::steady_clock::time_point until)
{
    inventory_detail::RootHandle root(path.parent_path()); inventory_detail::SourceHandle source(root, file_io::path_text(path.filename()));
    auto const before = inventory_detail::stamp(inventory_detail::stat(source.handle));
    require(before.bytes <= 512 * 1024 * 1024, "native farm executable exceeds its bounded fingerprint input");
    struct sha256 hash; sha256_init(&hash); std::array<uint8_t, 65536> buffer{}; uint64_t at = 0;
    while (at < before.bytes)
    {
        inventory_detail::check(limits, until); size_t bytes = size_t(std::min<uint64_t>(buffer.size(), before.bytes - at));
        auto count = file_io::read_at(source.handle.fd, buffer.data(), bytes, at);
        if (count < 0 && errno == EINTR) continue; require(count > 0, "native farm executable changed while fingerprinting");
        sha256_update(&hash, buffer.data(), size_t(count)); at += uint64_t(count);
    }
    require(inventory_detail::stamp(inventory_detail::stat(source.handle)) == before && stream_state::stamp(path) == before,
        "native farm executable changed while fingerprinting");
    std::array<uint8_t, 32> digest{}; sha256_sum(&hash, digest.data()); return group_tools::hex(digest);
}
// This is optional retirement compatibility, not the configuration parser
// used to admit RPC. Accept only conventional block YAML, with bounded aliases
// to already completed nodes. Unsupported syntax keeps the strict byte identity.
struct RetirementConfig
{ std::string identity; std::vector<std::string> directories; };
class RetirementConfigReader
{
    struct Node
    {
        char kind = 's'; size_t depth = 1; std::string scalar, identity;
        std::map<std::string, std::shared_ptr<Node>> fields;
        std::vector<std::shared_ptr<Node>> items;
    };
    struct Line { size_t indent; std::string text; };
    std::vector<Line> lines_;
    std::map<std::string, std::shared_ptr<Node>> anchors_;
    size_t at_ = 0, nodes_ = 0;
    uint64_t bytes_ = 0, limit_;
    void charge(uint64_t bytes)
    { require(bytes <= limit_ - std::min(bytes_, limit_), "retirement configuration exceeds its parse grant"); bytes_ += bytes; }
    static std::string_view trim(std::string_view value) { return profile_detail::trim(value); }
    static bool sequence(std::string_view value) { return value == "-" || value.starts_with("- "); }
    static size_t colon(std::string_view value)
    {
        char quote = 0;
        for (size_t i = 0; i < value.size(); ++i)
        {
            char const c = value[i];
            if (quote == '"' && c == '\\') { require(i + 1 < value.size(), "incomplete YAML escape"); ++i; continue; }
            if (quote == '\'' && c == '\'' && i + 1 < value.size() && value[i + 1] == '\'') { ++i; continue; }
            if (quote) { if (c == quote) quote = 0; continue; }
            if (c == '\'' || c == '"') { quote = c; continue; }
            if (c == ':' && (i + 1 == value.size() || value[i + 1] == ' ')) return i;
        }
        require(!quote, "unterminated YAML scalar"); return value.npos;
    }
    static std::string scalar(std::string_view value)
    {
        value = trim(value);
        if (value.empty() || value == "~" || value == "null" || value == "Null" || value == "NULL") return "n";
        if (value.front() == '\'' || value.front() == '"')
        {
            require(value.size() >= 2 && value.back() == value.front(), "unsupported YAML quoted scalar");
            if (value.front() == '"')
            { auto parsed = profile_detail::JsonReader(std::string(value), 8 * value.size() + 1024).read(); return "s" + profile_detail::text(parsed); }
            std::string result = "s";
            for (size_t i = 1; i + 1 < value.size(); ++i)
            {
                if (value[i] == '\'') { require(i + 2 < value.size() && value[i + 1] == '\'', "unsupported YAML quoted scalar"); ++i; }
                result += value[i];
            }
            return result;
        }
        for (auto text : {"true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"}) if (value == text) return "b1";
        for (auto text : {"false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF"}) if (value == text) return "b0";
        require(value != "-" && value != ":" && !value.starts_with("- ") &&
            value.find_first_of("[]{}!|>\"'`@") == value.npos && value.front() != '?' && value.front() != '%' &&
            value.front() != '&' && value.front() != '*' && value.find(": ") == value.npos, "unsupported YAML scalar syntax");
        auto digits = value; bool const negative = digits.front() == '-';
        if (digits.front() == '+' || digits.front() == '-') digits.remove_prefix(1);
        bool const numeric = !digits.empty() && (std::isdigit(static_cast<unsigned char>(digits.front())) || digits.front() == '.');
        if (numeric && digits.find_first_not_of("0123456789") == digits.npos)
        {
            require(digits.size() == 1 || digits.front() != '0', "ambiguous YAML integer representation");
            return std::string("i") + (negative && digits != "0" ? "-" : "") + std::string(digits);
        }
        // PyYAML's ordinary float resolver requires a decimal point and an
        // explicit sign on an exponent. Other numeric-looking syntax is not
        // accepted as compatibility authority.
        if (numeric)
        {
            size_t i = 0; while (i < digits.size() && std::isdigit(static_cast<unsigned char>(digits[i]))) ++i;
            if (i < digits.size() && digits[i] == '.')
            {
                ++i; while (i < digits.size() && std::isdigit(static_cast<unsigned char>(digits[i]))) ++i;
                bool valid = i > 1 || (i == 1 && digits.front() != '.');
                if (i < digits.size() && (digits[i] == 'e' || digits[i] == 'E'))
                {
                    ++i; valid = valid && i < digits.size() && (digits[i] == '+' || digits[i] == '-');
                    if (valid) { ++i; size_t const start = i; while (i < digits.size() && std::isdigit(static_cast<unsigned char>(digits[i]))) ++i; valid = i > start; }
                }
                if (valid && i == digits.size())
                {
                    auto text = value; if (text.front() == '+') text.remove_prefix(1);
                    double number = 0; auto parsed = std::from_chars(text.data(), text.data() + text.size(), number);
                    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && std::isfinite(number), "unsupported YAML float");
                    std::ostringstream out; out << 'f' << std::hexfloat << number; return out.str();
                }
            }
            require(digits.find_first_of("_:-") == digits.npos && !digits.starts_with("0x") && !digits.starts_with("0X") &&
                !digits.starts_with("0o") && !digits.starts_with("0O") && !digits.starts_with("0b") && !digits.starts_with("0B"),
                "ambiguous YAML numeric scalar");
        }
        auto nonfinite = value; if (nonfinite.front() == '+' || nonfinite.front() == '-') nonfinite.remove_prefix(1);
        require(nonfinite != ".inf" && nonfinite != ".Inf" && nonfinite != ".INF" && nonfinite != ".nan" && nonfinite != ".NaN" && nonfinite != ".NAN",
            "unsupported YAML nonfinite scalar");
        return "s" + std::string(value);
    }
    std::shared_ptr<Node> node(char kind)
    { require(++nodes_ <= 16384, "retirement configuration exceeds its node bound"); charge(sizeof(Node) + 256); auto result = std::make_shared<Node>(); result->kind = kind; return result; }
    std::shared_ptr<Node> value(std::string_view text, size_t indent, size_t depth, bool mapping_value = false)
    {
        require(depth <= 64, "retirement configuration exceeds its depth bound"); text = trim(text); std::string anchor;
        if (text.starts_with('&'))
        {
            auto const end = text.find(' '); anchor = text.substr(1, end == text.npos ? text.size() - 1 : end - 1);
            require(!anchor.empty() && anchor.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == anchor.npos && !anchors_.contains(anchor), "invalid YAML anchor");
            text = end == text.npos ? std::string_view{} : trim(text.substr(end));
        }
        std::shared_ptr<Node> result;
        if (text.starts_with('*'))
        {
            require(anchor.empty(), "unsupported anchored YAML alias"); auto found = anchors_.find(std::string(text.substr(1)));
            require(found != anchors_.end(), "forward, cyclic or unknown YAML alias"); charge(64); return found->second;
        }
        if (text == "!!set {}") result = node('u');
        else if (text == "!!set")
        {
            require(at_ < lines_.size() && lines_[at_].indent > indent, "unsupported YAML set"); result = block(lines_[at_].indent, depth + 1);
            require(result->kind == 'm', "unsupported YAML set structure");
            for (auto const& [key, member] : result->fields)
                require(key.starts_with('s') && member->kind == 's' && member->scalar == "n", "unsupported YAML set member");
            result->kind = 'u';
        }
        else if (text.empty() && at_ < lines_.size() && (lines_[at_].indent > indent ||
            (mapping_value && lines_[at_].indent == indent && sequence(lines_[at_].text)))) result = block(lines_[at_].indent, depth + 1);
        else if (text == "[]") result = node('l');
        else if (text == "{}") result = node('m');
        else { result = node('s'); result->scalar = scalar(text); charge(2 * result->scalar.capacity()); }
        if (!anchor.empty()) { charge(anchor.capacity() + 128); anchors_.emplace(std::move(anchor), result); }
        return result;
    }
    void field(Node& map, std::string_view text, size_t indent, size_t depth)
    {
        auto const split = colon(text); require(split != text.npos, "unsupported YAML block mapping"); auto key = scalar(trim(text.substr(0, split)));
        require(key.starts_with('s') && key != "s<<", "YAML mapping keys must be unique strings without merges"); charge(2 * key.capacity() + 128);
        auto child = value(text.substr(split + 1), indent, depth, true);
        require(child->depth < 64, "retirement configuration alias depth exceeds its bound");
        map.depth = std::max(map.depth, child->depth + 1);
        require(map.fields.emplace(std::move(key), std::move(child)).second, "duplicate YAML mapping key");
    }
    std::shared_ptr<Node> block(size_t indent, size_t depth)
    {
        require(depth <= 64, "retirement configuration exceeds its depth bound");
        bool const list = sequence(lines_[at_].text); auto result = node(list ? 'l' : 'm');
        while (at_ < lines_.size() && lines_[at_].indent == indent && sequence(lines_[at_].text) == list)
        {
            auto const text = lines_[at_++].text;
            if (!list) { field(*result, text, indent, depth); continue; }
            auto rest = trim(std::string_view(text).substr(1));
            if (colon(rest) != rest.npos)
            {
                auto row = node('m'); field(*row, rest, indent + 2, depth + 1);
                while (at_ < lines_.size() && lines_[at_].indent == indent + 2 && !sequence(lines_[at_].text))
                { auto next = lines_[at_++].text; field(*row, next, indent + 2, depth + 1); }
                result->items.push_back(std::move(row));
            }
            else result->items.push_back(value(rest, indent, depth));
            require(result->items.back()->depth < 64, "retirement configuration alias depth exceeds its bound");
            result->depth = std::max(result->depth, result->items.back()->depth + 1);
            charge(64);
        }
        return result;
    }
    std::string digest(Node& value, bool root = false)
    {
        if (!root && !value.identity.empty()) return value.identity;
        std::string material(1, value.kind);
        if (value.kind == 's') material += value.scalar;
        for (auto const& item : value.items) material += digest(*item);
        for (auto const& [key, child] : value.fields)
        {
            material += stream_state::digest(key);
            if (root && key == "sharvester")
            {
                require(child->kind == 'm', "unsupported harvester config"); std::string fields = "m";
                for (auto const& [name, nested] : child->fields)
                { fields += stream_state::digest(name); fields += name == "splot_directories" ? stream_state::digest("retirement-plot-directories-v1") : digest(*nested); }
                material += stream_state::digest(fields);
            }
            else material += digest(*child);
        }
        charge(material.capacity()); auto result = stream_state::digest(material); if (!root) value.identity = result; return result;
    }
public:
    explicit RetirementConfigReader(std::string_view contents, uint64_t limit) : limit_(limit)
    {
        // YAML treats these raw Unicode characters as line breaks, including
        // inside quoted text. JSON-style escapes instead denote literal text.
        for (auto separator : {"\xc2\x85", "\xe2\x80\xa8", "\xe2\x80\xa9"})
            require(contents.find(separator) == contents.npos, "unsupported raw YAML Unicode line separator");
        for (size_t i = 0; i < contents.size(); ++i) if (contents[i] == '\r')
            require(i + 1 < contents.size() && contents[i + 1] == '\n', "unsupported bare YAML carriage return");
        charge(contents.size()); std::istringstream input{std::string(contents)};
        for (std::string line; std::getline(input, line);)
        {
            require(line.find('\t') == line.npos, "unsupported YAML indentation"); char quote = 0;
            for (size_t i = 0; i < line.size(); ++i)
            {
                char const c = line[i];
                if (quote == '"' && c == '\\') { require(i + 1 < line.size(), "incomplete YAML escape"); ++i; continue; }
                if (quote == '\'' && c == '\'' && i + 1 < line.size() && line[i + 1] == '\'') { ++i; continue; }
                if (quote) { if (c == quote) quote = 0; continue; }
                if (c == '\'' || c == '"') quote = c;
                else if (c == '#' && (!i || line[i - 1] == ' ')) { line.resize(i); break; }
            }
            require(!quote, "unsupported multiline YAML scalar"); auto text = trim(line); if (text.empty()) continue;
            require(text != "---" && text != "..." && !text.starts_with('%'), "unsupported YAML directive");
            charge(2 * line.capacity() + sizeof(Line) + 64); lines_.push_back({line.find_first_not_of(' '), std::string(text)});
        }
    }
    RetirementConfig read(bool strict_paths = true)
    {
        require(!lines_.empty() && !lines_[0].indent, "unsupported YAML root"); auto root = block(0, 0);
        require(at_ == lines_.size() && root->kind == 'm', "unsupported YAML block structure");
        auto harvester = root->fields.find("sharvester"); require(harvester != root->fields.end() && harvester->second->kind == 'm', "missing harvester configuration");
        auto dirs = harvester->second->fields.find("splot_directories"); require(dirs != harvester->second->fields.end() && dirs->second->kind == 'l', "unsupported plot directory list");
        RetirementConfig result;
        for (auto const& dir : dirs->second->items)
        {
            require(dir->kind == 's' && dir->scalar.starts_with('s'), "plot directory is not text"); auto path = dir->scalar.substr(1);
            if (strict_paths) require(file_io::native_path(path).is_absolute() && file_io::path_text(file_io::native_path(path).lexically_normal()) == path &&
                std::find(result.directories.begin(), result.directories.end(), path) == result.directories.end(), "plot directories are not unique normalized absolute paths");
            result.directories.push_back(std::move(path));
        }
        std::sort(result.directories.begin(), result.directories.end()); result.identity = digest(*root, true); return result;
    }
};
inline RetirementConfig retirement_config(std::string_view contents, uint64_t limit)
{
    try { return RetirementConfigReader(contents, limit).read(); }
    catch (std::exception const&) { return {}; }
}
class Session
{
    HarvesterAdapterOptions options_;
    std::chrono::steady_clock::time_point until_;
    inventory_detail::RootHandle root_;
    inventory_detail::SourceHandle config_;
    stream_state::Stamp config_stamp_, executable_stamp_;
    std::string config_digest_, reader_identity_;
    unsigned configured_lookup_workers_ = 0;
    RetirementReaderContext retirement_;
    std::map<std::string, std::unique_ptr<Session>> services_;
public:
    explicit Session(HarvesterAdapterOptions const& options)
        : options_(options), until_(calibration_process::deadline(options.limits.timeout_ms)), root_(options.root), config_(root_, "config/config.yaml")
    {
        inventory_detail::validate_limits(options.limits);
        require(options.limits.memory_bytes >= 32 * 1024 * 1024, "native farm adapter private process/parse grant is too small");
        require(options.root.is_absolute() && options.executable.is_absolute(), "native farm adapter requires explicit absolute executable and Chia root");
        config_stamp_ = inventory_detail::stamp(inventory_detail::stat(config_.handle));
        require(config_stamp_.bytes <= std::min<uint64_t>(1024 * 1024, options.limits.memory_bytes / 32), "native farm adapter config exceeds its byte bound");
        std::string contents(size_t(config_stamp_.bytes), '\0'); size_t at = 0;
        while (at < contents.size())
        {
            auto count = file_io::read_at(config_.handle.fd, contents.data() + at, contents.size() - at, at);
            if (count < 0 && errno == EINTR) continue; require(count > 0, "native farm adapter config changed while reading"); at += size_t(count);
        }
        config_digest_ = stream_state::digest(contents); std::istringstream input(contents);
        auto parsed = profile_detail::read_chia_config(input, options.limits.memory_bytes / 4);
        require(parsed.workers.has_value() && (parsed.local_rpc || (!options.rpc_host.empty() && parsed.rpc_hostname == options.rpc_host && !options.host.empty())),
            "native farm adapter requires conventional configuration and an explicit host binding for authenticated remote RPC");
        require(options.rpc_host.empty() || parsed.rpc_hostname == options.rpc_host, "configured native RPC host differs from its local service binding");
        configured_lookup_workers_ = *parsed.workers;
        executable_stamp_ = stream_state::stamp(options.executable);
        auto implementation = implementation_identity(options.executable, options.limits, until_);
        reader_identity_ = stream_state::digest(file_io::path_text(options.executable) + '\n' +
            implementation + '\n' + file_io::path_text(root_.path) + '\n' + config_digest_ + '\n' + options.rpc_host + '\n' + options.host);
        auto conventional = retirement_config(contents, options.limits.memory_bytes / 4);
        bool compatible = !conventional.identity.empty() && conventional.directories.size() <= options.limits.max_roots;
        std::string retirement_identity = "farm-reader-retirement-1\n" +
            file_io::path_text(options.executable) + '\n' + implementation + '\n' + file_io::path_text(root_.path) + '\n' +
            conventional.identity + '\n' + options.rpc_host + '\n' + options.host;
        std::map<std::string, std::vector<std::string>> configurations;
        if (compatible) configurations.emplace(stream_state::digest(file_io::path_text(root_.path)), std::move(conventional.directories));
        std::string bindings, retirement_bindings;
        for (auto const& [service, binding] : options.services)
        {
            require(service == "harvester" || service == "farmer" || service == "full_node" || service == "solver", "unsupported native service binding");
            require(!binding.host.empty() && !binding.rpc_host.empty() && binding.root.is_absolute(), "native service binding lacks its explicit RPC/resource host or local config root");
            bindings += service + '\n' + binding.node_id + '\n' + binding.farmer_node_id + '\n' + binding.host + '\n';
            retirement_bindings += service + '\n' + binding.node_id + '\n' + binding.farmer_node_id + '\n' + binding.host + '\n';
            if (service == "harvester")
            {
                require(binding.root == options.root && binding.rpc_host == parsed.rpc_hostname && binding.host == options.host, "harvester service binding differs from the selected reader context");
                continue;
            }
            auto child = options; child.services.clear(); child.root = binding.root; child.rpc_host = binding.rpc_host; child.host = binding.host;
            auto context = std::make_unique<Session>(child); bindings += context->reader_identity() + '\n';
            auto const& observed = context->retirement_context(); compatible = compatible && !observed.retirement_identity.empty();
            retirement_bindings += observed.retirement_identity + '\n';
            for (auto const& config : observed.configurations)
            {
                auto [at, fresh] = configurations.emplace(config.root_identity, config.plot_directories);
                compatible = compatible && (fresh || at->second == config.plot_directories);
            }
            services_.emplace(service, std::move(context));
        }
        if (!bindings.empty()) reader_identity_ = stream_state::digest(reader_identity_ + '\n' + bindings);
        if (compatible)
        {
            retirement_.reader_identity = reader_identity_;
            retirement_.retirement_identity = stream_state::digest(retirement_identity + '\n' + retirement_bindings);
            for (auto& [root, directories] : configurations) retirement_.configurations.push_back({root, std::move(directories)});
        }
        unchanged();
    }
    void running() const
    {
        require(!options_.limits.cancelled || !options_.limits.cancelled(), "farm inventory cancelled");
        require_service_available(std::chrono::steady_clock::now() < until_, "native farm adapter deadline expired");
    }
    void unchanged() const
    {
        running();
        for (auto const& [service, context] : services_) { (void)service; context->unchanged(); }
        require(inventory_detail::stamp(inventory_detail::stat(config_.handle)) == config_stamp_ &&
            stream_state::stamp(options_.root / "config" / "config.yaml") == config_stamp_ &&
            stream_state::stamp(options_.executable) == executable_stamp_, "native farm adapter executable or configuration changed");
    }
    Json rpc(AdapterOperation operation, uint32_t height = 0)
    {
        unchanged();
        char const* service = "harvester";
        switch (operation)
        {
        case AdapterOperation::full_node_version: case AdapterOperation::full_node_network: case AdapterOperation::full_node_constants:
        case AdapterOperation::full_node_state: case AdapterOperation::full_node_block_record: case AdapterOperation::full_node_connections: service = "full_node"; break;
        case AdapterOperation::farmer_version: case AdapterOperation::farmer_network: case AdapterOperation::farmer_connections:
        case AdapterOperation::farmer_harvesters: case AdapterOperation::farmer_harvesters_summary: case AdapterOperation::farmer_pool_state: service = "farmer"; break;
        case AdapterOperation::solver_version: case AdapterOperation::solver_network: case AdapterOperation::solver_connections: case AdapterOperation::solver_state: service = "solver"; break;
        default: break;
        }
        if (auto found = services_.find(service); found != services_.end()) return found->second->rpc(operation, height);
        auto const remaining = std::chrono::duration_cast<std::chrono::milliseconds>(until_ - std::chrono::steady_clock::now()).count();
        require(remaining > 0, "native farm adapter deadline expired");
        AdapterCommand command; command.operation = operation; command.root = options_.root; command.executable = options_.executable;
        command.height = height;
        // DrChia's published Chia 2.2-based RPC exposes no get_version.
        // Record CLI implementation provenance honestly; get_plots supplies
        // the fresh authoritative public metadata, not a made-up version RPC.
        if (operation == AdapterOperation::harvester_version && options_.kind == HarvesterAdapterKind::drchia)
            command.operation = AdapterOperation::reader_cli_version;
        // Reserve parent capture/JSON structure separately from the child OS cap.
        calibration_process::Limits limits{options_.limits.memory_bytes / 2, std::min(uint64_t(remaining), options_.limits.timeout_ms),
            std::min(options_.limits.max_output_bytes, options_.limits.memory_bytes / 64), options_.limits.cancelled};
        AdapterResult result;
        try { result = run_farm_adapter(command, limits); }
        catch (calibration_process::DeadlineExpired const&)
        { throw FarmServiceUnavailable("native " + std::string(service) + " RPC deadline expired"); }
        require_service_available(result.exit_code == 0, "native " + std::string(service) + " RPC failed (native output withheld)"); unchanged();
        if (command.operation == AdapterOperation::reader_cli_version)
        {
            auto text = std::string_view(result.output);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
            require(!text.empty() && text.size() <= 128 && text.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.+_-") == text.npos,
                "native reader CLI returned invalid version provenance (output withheld)");
            Json parsed; parsed.kind = Json::object; Json success; success.kind = Json::boolean; success.scalar = "true";
            Json version; version.kind = Json::string; version.scalar = "cli:" + std::string(text);
            parsed.fields.emplace("success", std::move(success)); parsed.fields.emplace("version", std::move(version)); return parsed;
        }
        auto parsed = profile_detail::JsonReader(result.output, options_.limits.memory_bytes / 8, options_.limits.memory_bytes / 4096).read();
        auto const* success = profile_detail::field(parsed, "success");
        require_service_available(success && profile_detail::boolean(*success), "native " + std::string(service) + " RPC did not report success"); return parsed;
    }
    std::string source() const
    {
        return std::string(name(options_.kind)) + ":" + file_io::path_text(options_.executable) + ":config-sha256=" + config_digest_;
    }
    std::string const& reader_identity() const { return reader_identity_; }
    unsigned configured_lookup_workers() const { return configured_lookup_workers_; }
    RetirementReaderContext const& retirement_context() const { return retirement_; }
    bool matches_reader_identity(std::string_view reviewed) const
    {
        if (reviewed == reader_identity_) return true;
        if (!options_.retirement || retirement_.retirement_identity.empty()) return false;
        for (auto const& reader : options_.retirement->inventory().retirement_readers)
        {
            if (reader.reader_identity != reviewed || reader.retirement_identity != retirement_.retirement_identity ||
                reader.configurations.size() != retirement_.configurations.size()) continue;
            bool removed = false, matches = true;
            for (auto const& current : retirement_.configurations)
            {
                auto found = std::find_if(reader.configurations.begin(), reader.configurations.end(), [&](auto const& config)
                    { return config.root_identity == current.root_identity; });
                if (found == reader.configurations.end()) { matches = false; break; }
                for (auto const& directory : current.plot_directories)
                    if (std::find(found->plot_directories.begin(), found->plot_directories.end(), directory) == found->plot_directories.end()) matches = false;
                for (auto const& directory : found->plot_directories)
                    if (std::find(current.plot_directories.begin(), current.plot_directories.end(), directory) == current.plot_directories.end())
                    { removed = true; if (!options_.retirement->permits(current.root_identity, directory)) matches = false; }
            }
            if (matches && removed) return true;
        }
        return false;
    }
    void begin_inventory_files(uint64_t entries)
    {
        unchanged(); until_ = inventory_detail::catalog_deadline(options_.limits, entries);
        for (auto& [service, context] : services_) { (void)service; context->begin_inventory_files(entries); }
    }
    InventoryLimits remaining_limits() const
    {
        unchanged(); auto result = options_.limits;
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(until_ - std::chrono::steady_clock::now()).count();
        require(left > 0, "native inventory deadline expired"); result.timeout_ms = std::min(uint64_t(left), options_.limits.timeout_ms); return result;
    }
};
inline bool matches_reviewed_reader(Session const& session, HarvesterAdapterOptions const& options,
    InventoryPlot const& plot, std::string const& version)
{
    if (options.revalidation) if (auto current = options.revalidation->expected(plot))
        return session.matches_reader_identity(current->identity) && current->version == version && current->reader == name(options.kind);
    return session.matches_reader_identity(plot.reader_identity) && plot.reader_version == version && plot.reader == name(options.kind);
}
inline NetworkProjection projection(Session& session)
{
    NetworkProjection result; result.source = session.source(); result.captured_at = inventory_detail::timestamp();
    result.version = version(session.rpc(AdapterOperation::full_node_version));
    result.reader_version = version(session.rpc(AdapterOperation::harvester_version)); result.reader_identity = session.reader_identity();
    auto network = session.rpc(AdapterOperation::full_node_network); result.network = inventory_detail::string(network, "network_name");
    auto response = session.rpc(AdapterOperation::full_node_constants); auto const& constants = inventory_detail::required_field(response, "constants");
    result.genesis_challenge = hex_string(inventory_detail::required_field(constants, "GENESIS_CHALLENGE"), 32);
    result.hard_fork2_height = inventory_detail::integer(constants, "HARD_FORK2_HEIGHT");
    result.min_strength = inventory_detail::integer(constants, "MIN_PLOT_STRENGTH"); result.max_strength = inventory_detail::integer(constants, "MAX_PLOT_STRENGTH");
    result.plot_size = inventory_detail::integer(constants, "PLOT_SIZE_V2");
    require(!result.network.empty() && result.min_strength <= result.max_strength && result.max_strength <= 255 && result.plot_size >= 18 && result.plot_size <= 50,
        "native network constants have unsupported v2 parameters");
    result.available = true; return result;
}
inline uint64_t active_network(Session& session, HarvesterAdapterOptions const& options, ReplacementExpectation const& expected, NetworkProjection const& projection)
{
    require(projection.available && expected.network == projection.network && expected.genesis_challenge == projection.genesis_challenge,
        "replacement target network/genesis differs from the reviewed live network");
    require(expected.k == projection.plot_size && expected.strength >= projection.min_strength && expected.strength <= projection.max_strength,
        "replacement parameters are outside the live network's consensus bounds");
    auto harvester_network = session.rpc(AdapterOperation::harvester_network);
    require(inventory_detail::string(harvester_network, "network_name") == expected.network, "harvester and full node report different networks");
    auto response = session.rpc(AdapterOperation::full_node_state); auto const& state = inventory_detail::required_field(response, "blockchain_state");
    auto const& sync = inventory_detail::required_field(state, "sync");
    require_service_available(inventory_detail::boolean(sync, "synced") && !inventory_detail::boolean(sync, "sync_mode"), "replacement requires a synced full node outside sync mode");
    auto const& peak = inventory_detail::required_field(state, "peak");
    auto const last_transaction = inventory_detail::integer(peak, "prev_transaction_block_height");
    require(peak.kind == Json::object && last_transaction > projection.hard_fork2_height && last_transaction <= UINT32_MAX,
        "replacement PoS2 consensus is not active at the observed last transaction height");
    auto const& peak_time = inventory_detail::required_field(peak, "timestamp"); uint64_t timestamp = 0;
    if (peak_time.kind != Json::null) timestamp = profile_detail::integer(peak_time);
    else
    {
        // A non-transaction peak has no timestamp. Ask for the actual previous
        // transaction block rather than treating a healthy synced node as old.
        auto block = session.rpc(AdapterOperation::full_node_block_record, uint32_t(last_transaction));
        auto const& record = inventory_detail::required_field(block, "block_record");
        require(inventory_detail::integer(record, "height") == last_transaction, "native node returned a different transaction block");
        timestamp = inventory_detail::integer(record, "timestamp");
    }
    auto const now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    require_service_available(options.max_peak_age_seconds && now >= 0 && timestamp <= uint64_t(now) && uint64_t(now) - timestamp <= options.max_peak_age_seconds,
        "replacement full-node peak has no sufficiently recent transaction timestamp");
    return timestamp;
}
inline uint64_t reviewed_target(Session& session, HarvesterAdapterOptions const& options, NetworkProjection const& reviewed,
    NetworkProjection const& observed, uint8_t k, uint8_t strength)
{
    require(reviewed.available && session.matches_reader_identity(reviewed.reader_identity) && reviewed.reader_version == observed.reader_version &&
        reviewed.version == observed.version && (reviewed.target_route.empty() || reviewed.target_route == options.harvester),
        "target reader/configuration/route changed since review; sources must be preserved");
    ReplacementExpectation expected; expected.k = k; expected.strength = strength; expected.network = reviewed.network;
    expected.genesis_challenge = reviewed.genesis_challenge; return active_network(session, options, expected, observed);
}
inline void require_publication(ReplacementExpectation const& expected, ReplacementPublication const& publication)
{
    require(publication.path == expected.path && publication.identity == expected.identity &&
        inventory_detail::hex(publication.review_digest, 32) && expected.path.extension() == ".plot2",
        "replacement requires the private journal's immutable first .plot2 publication receipt");
}
inline long double service_time(Json const& object, char const* field, bool positive = false)
{
    auto const value = profile_detail::real(inventory_detail::required_field(object, field));
    auto const now = std::chrono::duration<long double>(std::chrono::system_clock::now().time_since_epoch()).count();
    require((positive ? value > 0 : value >= 0) && value <= now, "target service timestamp is missing, invalid or in the future");
    return value;
}
inline std::string loopback_peer(std::string_view host)
{
    if (host == "127.0.0.1" || host == "::ffff:127.0.0.1") return "127.0.0.1";
    if (host == "::1") return "::1";
    return {};
}
struct ServiceConnection
{
    std::string node, host;
    uint64_t type = 0, local = 0, server = 0, peer = 0;
    long double created = 0;
    bool operator==(ServiceConnection const&) const = default;
};
inline std::vector<ServiceConnection> service_connections(Session& session, AdapterOperation operation)
{
    auto response = session.rpc(operation); auto const& rows = inventory_detail::required_field(response, "connections");
    require(rows.kind == Json::array && rows.items.size() <= 128, "target service connection view exceeds its bounded peer count");
    std::vector<ServiceConnection> result; result.reserve(rows.items.size());
    for (auto const& row : rows.items)
    {
        session.running(); ServiceConnection value;
        value.node = hex_string(inventory_detail::required_field(row, "node_id"), 32); value.host = inventory_detail::string(row, "peer_host");
        inventory_detail::bounded_text(value.host, 64); value.type = inventory_detail::integer(row, "type");
        value.local = inventory_detail::integer(row, "local_port"); value.server = inventory_detail::integer(row, "peer_server_port");
        value.peer = inventory_detail::integer(row, "peer_port"); value.created = service_time(row, "creation_time", true);
        (void)service_time(row, "last_message_time");
        require(value.type >= 1 && value.type <= 8 && (value.local || operation == AdapterOperation::harvester_connections) &&
            value.local <= 65535 && value.server <= 65535 && value.peer && value.peer <= 65535,
            "target service connection has unsupported role or port metadata");
        require(std::find(result.begin(), result.end(), value) == result.end(), "target service reports duplicate connection records");
        result.push_back(std::move(value));
    }
    return result;
}
inline std::pair<ServiceConnection, ServiceConnection> service_pair(std::vector<ServiceConnection> const& first,
    std::vector<ServiceConnection> const& second, uint64_t first_role, uint64_t second_role, bool allow_first_unadvertised = false,
    std::string const& first_peer = {}, std::string const& second_peer = {})
{
    std::optional<std::pair<ServiceConnection, ServiceConnection>> result;
    require(first_peer.empty() == second_peer.empty(), "native peer identity binding must identify both sides of a service link");
    for (auto const& a : first) for (auto const& b : second)
        if (a.type == second_role && b.type == first_role &&
            (first_peer.empty() ? (!loopback_peer(a.host).empty() && loopback_peer(a.host) == loopback_peer(b.host) &&
                a.server && (b.server || allow_first_unadvertised) && a.local == b.server && b.local == a.server) :
                (a.node == first_peer && b.node == second_peer)))
        {
            require(!result, "target service connection pairing is ambiguous"); result = std::pair{a, b};
        }
    require_service_available(result.has_value(), "target services lack a matched active peer connection; remote links require explicit reciprocal TLS node identities"); return *result;
}
inline void service_network(Session& session, AdapterOperation operation, NetworkProjection const& reviewed)
{
    auto response = session.rpc(operation);
    require(inventory_detail::string(response, "network_name") == reviewed.network &&
        hex_string(inventory_detail::required_field(response, "genesis_challenge"), 32) == reviewed.genesis_challenge,
        "target farmer or solver is configured for a different network");
}
class TargetServices
{
    Session& session_;
    Json harvesters_;
    Json const* receiver_ = nullptr;
    std::array<ServiceConnection, 3> farmer_links_;
    long double generation_ = 0;
    struct ReceiverGeneration { std::string host; uint64_t port = 0, files = 0; long double generation = 0; };
    std::map<std::string, ReceiverGeneration> other_generations_;
    std::string pool_hash_, pool_key_, farmer_version_, solver_version_, reviewed_network_;
    void pool_route()
    {
        if (!pool_key_.empty()) { pool.mode = "og-public-key-route"; pool_mode = pool.mode; return; }
        auto response = session_.rpc(AdapterOperation::farmer_pool_state); auto const& pools = inventory_detail::required_field(response, "pool_state");
        require(pools.kind == Json::array && pools.items.size() <= 128, "target farmer pool-state view exceeds its bound");
        bool selected = false; TargetPoolState observed;
        for (auto const& row : pools.items)
        {
            session_.running(); auto const id = hex_string(inventory_detail::required_field(row, "p2_singleton_puzzle_hash"), 32);
            if (id != pool_hash_) continue;
            require(!selected, "target farmer reports duplicate pool-contract routes"); selected = true;
            auto const& config = inventory_detail::required_field(row, "pool_config");
            require(hex_string(inventory_detail::required_field(config, "p2_singleton_puzzle_hash"), 32) == id, "target farmer pool-contract route is inconsistent");
            observed.pool_url = inventory_detail::string(config, "pool_url");
            inventory_detail::bounded_text(observed.pool_url, 2048);
            auto const* version = profile_detail::field(config, "version"); observed.protocol_version = version ? profile_detail::integer(*version) : 1;
            require(observed.protocol_version == 1 || observed.protocol_version == 2, "target farmer reports an unsupported pooling protocol version");
            // These are public pool route fields in both Plot NFT and
            // RewardPuzzle v2. They identify the route, not the auth secret.
            observed.launcher_id = hex_string(inventory_detail::required_field(config, "launcher_id"), 32);
            observed.owner_public_key = hex_string(inventory_detail::required_field(config, "owner_public_key"), 48);
            observed.target_puzzle_hash = hex_string(inventory_detail::required_field(config, "target_puzzle_hash"), 32);
            observed.external = !observed.pool_url.empty();
            observed.mode = observed.external ? "configured-external-pool" : "configured-self-pooling";
            if (observed.external)
            {
                require(observed.pool_url.starts_with("https://") || (reviewed_network_ != "mainnet" && observed.pool_url.starts_with("http://")),
                    "target farmer pool URL does not satisfy its network transport policy");
                observed.difficulty = inventory_detail::integer(row, "current_difficulty");
                observed.authentication_token_timeout = inventory_detail::integer(row, "authentication_token_timeout");
                require(observed.difficulty && observed.authentication_token_timeout && observed.authentication_token_timeout <= 255,
                    "target farmer has not initialized pool difficulty/authentication parameters");
                // Upstream initializes difficulty from public pool_info. Its
                // update schedule and counters do not attest registration or
                // possession of a current v2 bearer token. No partial from a
                // particular file, startup partial or arbitrary age is required.
            }
        }
        require(selected, "target farmer has no configured route matching the supplied pool-contract hash");
        if (!pool.mode.empty())
        {
            auto previous = pool; previous.difficulty = observed.difficulty;
            previous.authentication_token_timeout = observed.authentication_token_timeout;
            require(previous == observed, "target pool route or protocol changed during readiness observation");
        }
        pool = std::move(observed); pool_mode = pool.mode;
    }
    Json const& receiver(Json const& response) const
    {
        auto const& rows = inventory_detail::required_field(response, "harvesters");
        require(rows.kind == Json::array && rows.items.size() <= 128, "target farmer harvester view exceeds its bounded count");
        Json const* found = nullptr;
        for (auto const& row : rows.items)
        {
            session_.running(); auto const& connection = inventory_detail::required_field(row, "connection");
            if (hex_string(inventory_detail::required_field(connection, "node_id"), 32) != farmer_links_[0].node) continue;
            auto const host = inventory_detail::string(connection, "host");
            require(!found && (host == farmer_links_[0].host || (!loopback_peer(host).empty() && loopback_peer(host) == loopback_peer(farmer_links_[0].host))) &&
                inventory_detail::integer(connection, "port") == farmer_links_[0].peer,
                "target farmer receiver differs from its matched live harvester connection"); found = &row;
        }
        require_service_available(found, "target farmer has no synchronized receiver for the connected selected harvester");
        require_service_available(inventory_detail::required_field(*found, "syncing").kind == Json::null, "target farmer harvester synchronization is still in progress");
        (void)service_time(*found, "last_sync_time", true); return *found;
    }
public:
    std::string pool_mode;
    TargetPoolState pool;
    std::vector<OtherHarvesterObservation> other_harvesters;
    TargetServices(Session& session, HarvesterAdapterOptions const& options, NetworkProjection const& reviewed,
        NetworkProjection& observed, TargetServiceExpectation const& expected) : session_(session)
    {
        pool_hash_ = expected.pool_contract_puzzle_hash; pool_key_ = expected.pool_public_key; reviewed_network_ = reviewed.network;
        farmer_version_ = reviewed.farmer_version; solver_version_ = reviewed.solver_version;
        require(inventory_detail::hex(expected.farmer_public_key, 48), "target service prerequisite requires the supplied public farmer key");
        require(expected.pool_public_key.empty() != expected.pool_contract_puzzle_hash.empty(), "target service prerequisite requires exactly one public pool route");
        require(pool_key_.empty() ? inventory_detail::hex(pool_hash_, 32) : inventory_detail::hex(pool_key_, 48), "target public pool route is invalid");
        require(!reviewed.farmer_version.empty() && !reviewed.solver_version.empty(), "target farmer/solver versions were not observed during review; refresh inventory and migration review with these services available");
        observed.farmer_version = version(session.rpc(AdapterOperation::farmer_version)); observed.solver_version = version(session.rpc(AdapterOperation::solver_version));
        require(observed.farmer_version == reviewed.farmer_version && observed.solver_version == reviewed.solver_version,
            "target farmer or solver service version changed since review");
        service_network(session, AdapterOperation::farmer_network, reviewed); service_network(session, AdapterOperation::solver_network, reviewed);
        auto harvester = service_connections(session, AdapterOperation::harvester_connections);
        auto farmer = service_connections(session, AdapterOperation::farmer_connections);
        auto node = service_connections(session, AdapterOperation::full_node_connections);
        auto solver = service_connections(session, AdapterOperation::solver_connections);
        // NodeType values and advertised-port semantics are from the pinned
        // Chia peer protocol. Farmer TLS identities can differ by channel.
        auto pair = [&](auto const& first, auto const& second, uint64_t first_role, uint64_t second_role, char const* service, bool reverse = false)
        {
            auto found = options.services.find(service);
            if (found == options.services.end()) return service_pair(first, second, first_role, second_role, reverse);
            auto const& binding = found->second;
            require(inventory_detail::hex(binding.node_id, 32) && inventory_detail::hex(binding.farmer_node_id, 32),
                "explicit service link requires reciprocal reviewed TLS node identities");
            return service_pair(first, second, first_role, second_role, reverse,
                reverse ? binding.farmer_node_id : binding.node_id, reverse ? binding.node_id : binding.farmer_node_id);
        };
        farmer_links_[0] = pair(harvester, farmer, 2, 3, "harvester", true).second;
        farmer_links_[1] = pair(farmer, node, 3, 1, "full_node").first;
        farmer_links_[2] = pair(farmer, solver, 3, 8, "solver").first;
        {
            auto state = session.rpc(AdapterOperation::solver_state);
            require_service_available(profile_detail::boolean(inventory_detail::required_field(state, "started")), "target PoS2 solver has not completed startup");
        }
        pool_route();
        harvesters_ = session.rpc(AdapterOperation::farmer_harvesters); receiver_ = &receiver(harvesters_);
        generation_ = service_time(*receiver_, "last_sync_time", true);
        auto const& rows = inventory_detail::required_field(harvesters_, "harvesters");
        for (auto const& row : rows.items)
        {
            auto const& plots = inventory_detail::required_field(row, "plots");
            require(plots.kind == Json::array && plots.items.size() <= options.limits.max_files, "target farmer plot view exceeds its bound");
            if (&row == receiver_) continue;
            require_service_available(inventory_detail::required_field(row, "syncing").kind == Json::null,
                "another harvester synchronization is incomplete; even an empty partial view has unknown workload");
            (void)service_time(row, "last_sync_time", true);
            if (plots.items.empty()) continue;
            auto const& connection = inventory_detail::required_field(row, "connection");
            OtherHarvesterObservation other;
            other.node_id = hex_string(inventory_detail::required_field(connection, "node_id"), 32);
            other.peer_host = inventory_detail::string(connection, "host"); other.files = plots.items.size();
            for (auto const& plot : plots.items)
            {
                auto const size = inventory_detail::integer(plot, "size");
                auto const members = (size & 0x80) ? inventory_detail::integer(plot, "group_size") : 1;
                require(size <= 255 && members && members <= 65535 && members <= UINT64_MAX - other.members,
                    "other harvester reports unsupported member population");
                other.members += members;
            }
            require(other_generations_.emplace(other.node_id, ReceiverGeneration{other.peer_host,
                inventory_detail::integer(connection, "port"), other.files, service_time(row, "last_sync_time", true)}).second,
                "target farmer reports duplicate harvester receivers");
            other_harvesters.push_back(std::move(other));
        }
        observed.service_status = "live farmer/node/solver, completed selected-harvester sync and public pool route observed; signing-key possession and external pool authentication/acceptance are not exposed by these RPCs";
    }
    TargetServices(TargetServices const&) = delete;
    TargetServices& operator=(TargetServices const&) = delete;
    Json const& plots() const { return inventory_detail::required_field(*receiver_, "plots"); }
    void finish()
    {
        auto farmer = service_connections(session_, AdapterOperation::farmer_connections);
        for (auto const& link : farmer_links_)
            require_service_available(std::find(farmer.begin(), farmer.end(), link) != farmer.end(), "target service connection changed during the readiness snapshot");
        {
            auto summary = session_.rpc(AdapterOperation::farmer_harvesters_summary);
            auto const& selected = receiver(summary);
            require_service_available(service_time(selected, "last_sync_time", true) == generation_, "target farmer synchronization generation changed during the readiness snapshot");
            size_t observed = 0;
            for (auto const& row : inventory_detail::required_field(summary, "harvesters").items)
            {
                if (&row == &selected) continue;
                require_service_available(inventory_detail::required_field(row, "syncing").kind == Json::null,
                    "another harvester began synchronization during readiness observation; its workload is unknown");
                (void)service_time(row, "last_sync_time", true);
                auto const files = inventory_detail::integer(row, "plots"); if (!files) continue;
                auto const& connection = inventory_detail::required_field(row, "connection");
                auto const id = hex_string(inventory_detail::required_field(connection, "node_id"), 32);
                auto found = other_generations_.find(id);
                require(found != other_generations_.end() && found->second.files == files &&
                    found->second.host == inventory_detail::string(connection, "host") && found->second.port == inventory_detail::integer(connection, "port") &&
                    found->second.generation == service_time(row, "last_sync_time", true),
                    "another harvester's observed workload changed during readiness observation");
                ++observed;
            }
            require(observed == other_generations_.size(), "another harvester receiver disappeared during readiness observation");
        }
        {
            auto state = session_.rpc(AdapterOperation::solver_state);
            require_service_available(profile_detail::boolean(inventory_detail::required_field(state, "started")), "target PoS2 solver stopped during readiness observation");
        }
        pool_route();
        require(version(session_.rpc(AdapterOperation::farmer_version)) == farmer_version_ &&
            version(session_.rpc(AdapterOperation::solver_version)) == solver_version_, "target farmer or solver service version changed during readiness observation");
        session_.unchanged();
    }
};
} // namespace adapter_detail

// The catalog is immutable throughout an invocation. Keep its path and reader
// lookups once while each protected observation still obtains fresh native RPC
// data. The caller includes memory_bound() in its retained controller grant.
class RegistryScopeIndex
{
    using Reader = std::tuple<std::string_view, std::string_view, std::string_view>;
    FarmInventory const& inventory_;
    std::multimap<std::string, InventoryPlot const*, std::less<>> paths_;
    std::map<std::string_view, std::set<Reader>, std::less<>> readers_;
    static std::string filename(FarmInventory const& inventory, InventoryPlot const& plot)
    {
        auto value = file_io::path_text(file_io::native_path(inventory.roots.at(plot.root).path) /
            inventory_detail::relative(plot.relative_path));
#ifdef _WIN32
        std::replace(value.begin(), value.end(), '\\', '/');
#endif
        return value;
    }
    static bool nonplot(InventoryPlot const& plot)
    { return plot.status == "non-plot" && !plot.header_validated && !plot.reason.empty(); }
public:
    static uint64_t memory_bound(FarmInventory const& inventory)
    {
        uint64_t bytes = sizeof(RegistryScopeIndex) + 4096;
        for (auto const& plot : inventory.plots)
        {
            // A full-path node, reader tuple and (at most) one reader group.
            // All metadata except the normalized full path stays borrowed.
            uint64_t const capacity = nonplot(plot) ? 0 : filename(inventory, plot).capacity();
            inventory_detail::require(capacity <= UINT64_MAX - 4096 && capacity + 4096 <= UINT64_MAX - bytes,
                "registry scope index memory bound overflows");
            bytes += capacity + 4096;
        }
        return bytes;
    }
    RegistryScopeIndex(FarmInventory const& inventory, InventoryLimits const& limits = {}) : inventory_(inventory)
    {
        inventory_detail::validate_limits(limits);
        inventory_detail::require(inventory.plots.size() <= limits.max_files && memory_bound(inventory) <= limits.memory_bytes,
            "registry scope index exceeds its retained controller grant");
        for (auto const& plot : inventory.plots)
        {
            if (!plot.reader_identity.empty()) readers_[plot.harvester].emplace(plot.reader_identity, plot.reader_version, plot.reader);
            if (!nonplot(plot)) paths_.emplace(filename(inventory, plot), &plot);
        }
    }
    FarmInventory const& inventory() const { return inventory_; }
    void require_reader(adapter_detail::Session& session, HarvesterAdapterOptions const& options, std::string const& version) const
    {
        if (options.revalidation)
        {
            inventory_detail::require(&options.revalidation->inventory() == &inventory_, "revalidation reader allowance belongs to another catalog");
            options.revalidation->for_each_reader(options.harvester, [&](std::string_view identity, std::string_view observed_version, std::string_view reader)
            { inventory_detail::require(session.matches_reader_identity(identity) && observed_version == version && reader == adapter_detail::name(options.kind),
                "current revalidation reader implementation changed since review"); });
            return;
        }
        auto check = [&](auto const& readers)
        {
            for (auto const& [identity, observed_version, reader] : readers)
                inventory_detail::require(session.matches_reader_identity(identity) && observed_version == version &&
                    reader == adapter_detail::name(options.kind), "registry scope reader implementation changed since review");
        };
        if (options.harvester.empty()) for (auto const& [route, readers] : readers_) { (void)route; check(readers); }
        else
        {
            if (auto at = readers_.find(""); at != readers_.end()) check(at->second);
            if (auto at = readers_.find(options.harvester); at != readers_.end()) check(at->second);
        }
    }
    void require_scope(adapter_detail::Session& session, profile_detail::Json const& plots,
        HarvesterAdapterOptions const& options, PublishedPlotAllowance const& published = {}) const
    {
        inventory_detail::require(plots.kind == profile_detail::Json::array && plots.items.size() <= options.limits.max_files,
            "loaded registry scope exceeds its bounded lookup grant");
        for (auto const& native : plots.items)
        {
            session.running(); auto name = inventory_detail::string(native, "filename");
            auto const id = adapter_detail::hex_string(inventory_detail::required_field(native, "plot_id"), 32);
#ifdef _WIN32
            std::replace(name.begin(), name.end(), '\\', '/');
#endif
            auto const [first, last] = paths_.equal_range(name); bool covered = false;
            for (auto at = first; at != last; ++at)
            {
                auto const& plot = *at->second;
                if ((plot.harvester.empty() || options.harvester.empty() || plot.harvester == options.harvester) &&
                    (plot.plot_id.empty() || plot.plot_id == id)) { covered = true; break; }
            }
            inventory_detail::require(covered || (published && published(name, id)),
                "loaded plot is outside the reviewed reader/catalog scope; inventory every loaded source root and review the complete workload");
        }
    }
};

inline FarmAdapterTable parse_farm_adapters(profile_detail::Json const& document, InventoryLimits const& limits = {})
{
    using namespace inventory_detail;
    validate_limits(limits); FarmAdapterTable result;
    auto const* supplied = profile_detail::field(document, "farm_adapters"); if (!supplied) return result;
    auto shape = [&](profile_detail::Json const& value, std::initializer_list<std::string_view> fields)
    {
        require(value.kind == profile_detail::Json::object, "expected local farm adapter object");
        for (auto const& [key, child] : value.fields) { (void)child; require(std::find(fields.begin(), fields.end(), key) != fields.end(), "unsupported local farm adapter field"); }
    };
    shape(*supplied, {"schema", "target_route", "routes"}); require(integer(*supplied, "schema") == 1, "unsupported local farm adapter schema");
    result.target_route = string(*supplied, "target_route"); auto const& routes = required_field(*supplied, "routes");
    require(routes.kind == profile_detail::Json::array && !routes.items.empty() && routes.items.size() <= 64,
        "local farm adapter route count exceeds its bound");
    std::set<std::string> ids, roots;
    auto path = [&](std::string const& text)
    {
        bounded_text(text); auto value = file_io::native_path(text);
        require(value.is_absolute(), "local farm adapter paths must be explicit absolute paths"); value = value.lexically_normal();
        if (value != value.root_path() && value.filename().empty()) value = value.parent_path(); return value;
    };
    for (auto const& value : routes.items)
    {
        shape(value, {"id", "kind", "root", "executable", "source_roots", "host", "services"});
        FarmAdapterRoute route; route.id = string(value, "id");
        require(!route.id.empty() && route.id.size() <= 128 && route.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") == route.id.npos &&
            ids.insert(route.id).second, "invalid or duplicate local farm adapter route ID");
        auto kind = string(value, "kind");
        if (kind == "chia") route.options.kind = HarvesterAdapterKind::chia;
        else if (kind == "gigahorse") route.options.kind = HarvesterAdapterKind::gigahorse;
        else if (kind == "drchia") route.options.kind = HarvesterAdapterKind::drchia;
        else throw std::runtime_error("unsupported local farm adapter kind");
        route.options.root = path(string(value, "root")); route.options.executable = path(string(value, "executable"));
        auto const& sources = required_field(value, "source_roots");
        require(sources.kind == profile_detail::Json::array && sources.items.size() <= limits.max_roots, "local source route mapping exceeds its bound");
        for (auto const& source : sources.items)
        {
            auto normalized = file_io::path_text(path(profile_detail::text(source)));
            require(roots.insert(normalized).second, "ambiguous local farm source root routing"); route.source_roots.push_back(std::move(normalized));
        }
        route.options.limits = limits; route.options.host = profile_detail::host_name(); route.options.harvester = route.id;
        if (auto const* host = profile_detail::field(value, "host")) { route.options.host = profile_detail::text(*host); bounded_text(route.options.host, 256); require(!route.options.host.empty(), "farm adapter host binding is empty"); }
        if (auto const* services = profile_detail::field(value, "services"))
        {
            shape(*services, {"harvester", "farmer", "full_node", "solver"});
            for (auto const& [service, entry] : services->fields)
            {
                shape(entry, {"root", "rpc_host", "host", "node_id", "farmer_node_id"});
                FarmServiceBinding binding; binding.root = path(string(entry, "root")); binding.rpc_host = string(entry, "rpc_host"); binding.host = string(entry, "host");
                bounded_text(binding.host, 256); bounded_text(binding.rpc_host, 256);
                require(!binding.host.empty() && !binding.rpc_host.empty(), "service RPC/resource host binding is empty");
                if (auto const* id = profile_detail::field(entry, "node_id")) binding.node_id = adapter_detail::hex_string(*id, 32);
                if (auto const* id = profile_detail::field(entry, "farmer_node_id")) binding.farmer_node_id = adapter_detail::hex_string(*id, 32);
                require(service == "farmer" || (!binding.node_id.empty() && !binding.farmer_node_id.empty()), "service link requires both reciprocal TLS identities");
                if (service == "harvester")
                { require(binding.host == route.options.host, "harvester resource host differs from its route"); route.options.root = binding.root; route.options.rpc_host = binding.rpc_host; }
                route.options.services.emplace(service, std::move(binding));
            }
        }
        result.routes.push_back(std::move(route));
    }
    require(ids.contains(result.target_route), "local farm adapter target_route is missing"); return result;
}
inline FarmAdapterTable read_farm_adapters(std::istream& in, InventoryLimits const& limits = {})
{
    inventory_detail::validate_limits(limits); auto text = profile_detail::read_text(in, std::min(limits.max_output_bytes, limits.memory_bytes / 16));
    auto parsed = profile_detail::JsonReader(text, limits.memory_bytes / 2).read(); return parse_farm_adapters(parsed, limits);
}
// Local profile export only. Public inventories and migration reviews retain
// reader provenance but never serialize executable/configuration authority.
inline void write_farm_adapters_json(std::ostream& out, FarmAdapterTable const& table)
{
    inventory_detail::require(!table.routes.empty() && table.routes.size() <= 64, "invalid local farm adapter export scope");
    auto quote = [&](std::string const& text) { pos2gpu::write_json_string(out, text); };
    out << "{\"schema\":1,\"target_route\":"; quote(table.target_route); out << ",\"routes\":["; bool comma = false;
    for (auto const& route : table.routes)
    {
        if (comma) out << ','; comma = true;
        auto const& options = route.options;
        inventory_detail::require(options.root.is_absolute() && options.executable.is_absolute(), "local farm adapter export requires explicit absolute reader/configuration paths");
        out << "{\"id\":"; quote(route.id); out << ",\"kind\":";
        quote(options.kind == HarvesterAdapterKind::gigahorse ? "gigahorse" : options.kind == HarvesterAdapterKind::drchia ? "drchia" : "chia");
        out << ",\"root\":"; quote(file_io::path_text(options.root)); out << ",\"executable\":"; quote(file_io::path_text(options.executable));
        out << ",\"host\":"; quote(options.host); out << ",\"source_roots\":["; bool source_comma = false;
        for (auto const& root : route.source_roots) { if (source_comma) out << ','; source_comma = true; quote(root); }
        out << ']';
        if (!options.services.empty())
        {
            out << ",\"services\":{"; bool service_comma = false;
            for (auto const& [role, binding] : options.services)
            {
                if (service_comma) out << ','; service_comma = true; quote(role); out << ":{\"root\":"; quote(file_io::path_text(binding.root));
                out << ",\"rpc_host\":"; quote(binding.rpc_host); out << ",\"host\":"; quote(binding.host);
                if (!binding.node_id.empty()) { out << ",\"node_id\":"; quote(binding.node_id); }
                if (!binding.farmer_node_id.empty()) { out << ",\"farmer_node_id\":"; quote(binding.farmer_node_id); }
                out << '}';
            }
            out << '}';
        }
        out << '}';
    }
    out << "]}";
}
inline HarvesterAdapterOptions resolve_source(FarmAdapterTable const& table, InventoryRoot const& root, InventoryPlot const& plot)
{
    FarmAdapterRoute const* found = nullptr;
    auto const path = file_io::path_text(file_io::native_path(root.path).lexically_normal());
    for (auto const& route : table.routes)
        if (std::find(route.source_roots.begin(), route.source_roots.end(), path) != route.source_roots.end())
        { inventory_detail::require(!found, "ambiguous local farm source route"); found = &route; }
    inventory_detail::require(found && (plot.harvester.empty() || plot.harvester == found->id),
        "missing or changed reviewed source route; supply the matching explicit local profile");
    return found->options;
}
inline HarvesterAdapterOptions resolve_target(FarmAdapterTable const& table, std::string const& expected_route = {})
{
    inventory_detail::require(!table.target_route.empty() && (expected_route.empty() || expected_route == table.target_route),
        "missing or changed reviewed target route; supply the matching explicit local profile");
    FarmAdapterRoute const* found = nullptr;
    for (auto const& route : table.routes) if (route.id == table.target_route)
    { inventory_detail::require(!found, "ambiguous local farm target route"); found = &route; }
    inventory_detail::require(found, "local farm target route is unavailable"); return found->options;
}

inline void record_retirement_reader(FarmInventory& inventory, adapter_detail::Session const& session)
{
    auto const& context = session.retirement_context(); if (context.retirement_identity.empty()) return;
    auto found = std::find_if(inventory.retirement_readers.begin(), inventory.retirement_readers.end(), [&](auto const& reader)
        { return reader.reader_identity == context.reader_identity; });
    if (found == inventory.retirement_readers.end()) inventory.retirement_readers.push_back(context);
}
inline NetworkProjection observe_network_projection(HarvesterAdapterOptions const& options, FarmInventory* inventory = nullptr)
{
    NetworkProjection result;
    try { adapter_detail::Session session(options); result = adapter_detail::projection(session); session.unchanged(); if (inventory) record_retirement_reader(*inventory, session); }
    catch (std::exception const& error)
    {
        if (options.limits.cancelled && options.limits.cancelled()) throw;
        result.source = std::string("unavailable: ") + error.what(); result.captured_at = inventory_detail::timestamp(); return result;
    }
    // Optional planning provenance is separate from network availability and
    // never asserts service readiness. Execution requires these real versions.
    try
    {
        adapter_detail::Session session(options);
        inventory_detail::require(session.reader_identity() == result.reader_identity, "target reader/configuration changed during service observation");
        auto farmer = adapter_detail::version(session.rpc(AdapterOperation::farmer_version));
        auto solver = adapter_detail::version(session.rpc(AdapterOperation::solver_version)); session.unchanged();
        result.farmer_version = std::move(farmer); result.solver_version = std::move(solver);
        result.service_status = "farmer/solver versions observed; live prerequisites not checked";
    }
    catch (std::exception const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.service_status = std::string("unavailable: ") + error.what(); }
    return result;
}
namespace adapter_detail
{
inline void require_reviewed_reader(Session& session, FarmInventory const& inventory, HarvesterAdapterOptions const& options,
    std::string const& version, RegistryScopeIndex const* index = nullptr)
{
    if (index)
    {
        inventory_detail::require(&index->inventory() == &inventory, "registry scope index belongs to another catalog");
        index->require_reader(session, options, version); return;
    }
    if (options.revalidation)
    {
        inventory_detail::require(&options.revalidation->inventory() == &inventory, "revalidation reader allowance belongs to another catalog");
        options.revalidation->for_each_reader(options.harvester, [&](std::string_view identity, std::string_view observed_version, std::string_view reader)
        { inventory_detail::require(session.matches_reader_identity(identity) && observed_version == version && reader == name(options.kind),
            "current revalidation reader implementation changed since review"); });
        return;
    }
    for (auto const& plot : inventory.plots) if (!plot.reader_identity.empty() &&
        (options.harvester.empty() || plot.harvester.empty() || options.harvester == plot.harvester))
        inventory_detail::require(session.matches_reader_identity(plot.reader_identity) && plot.reader_version == version &&
            plot.reader == name(options.kind), "registry scope reader implementation changed since review");
}
inline void require_plot_scope(Session& session, Json const& plots, FarmInventory const& inventory,
    HarvesterAdapterOptions const& options, PublishedPlotAllowance const& published = {}, RegistryScopeIndex const* index = nullptr)
{
    if (index)
    {
        inventory_detail::require(&index->inventory() == &inventory, "registry scope index belongs to another catalog");
        index->require_scope(session, plots, options, published); return;
    }
    inventory_detail::require(plots.kind == Json::array && plots.items.size() <= options.limits.max_files &&
        inventory.plots.size() <= options.limits.memory_bytes / 4096, "loaded registry scope exceeds its bounded lookup grant");
    // Reference the retained catalog's relative paths instead of duplicating
    // every full path. The existing root count is bounded independently.
    std::map<std::pair<size_t, std::string_view>, InventoryPlot const*> paths;
    for (auto const& plot : inventory.plots)
        if (!(plot.status == "non-plot" && !plot.header_validated && !plot.reason.empty()) &&
            (plot.harvester.empty() || options.harvester.empty() || plot.harvester == options.harvester))
            paths.emplace(std::pair(plot.root, std::string_view(plot.relative_path)), &plot);
    for (auto const& native : plots.items)
    {
        session.running(); auto filename = inventory_detail::string(native, "filename");
        auto const id = hex_string(inventory_detail::required_field(native, "plot_id"), 32);
#ifdef _WIN32
        std::replace(filename.begin(), filename.end(), '\\', '/');
#endif
        bool covered = false;
        for (size_t root = 0; root < inventory.roots.size(); ++root)
        {
            auto prefix = inventory.roots[root].path;
#ifdef _WIN32
            std::replace(prefix.begin(), prefix.end(), '\\', '/');
#endif
            bool const separator = !prefix.empty() && (prefix.back() == '/' || prefix.back() == '\\');
            size_t const offset = prefix.size() + (separator ? 0 : 1);
            if (filename.size() <= offset || filename.compare(0, prefix.size(), prefix) ||
                (!separator && filename[prefix.size()] != '/' && filename[prefix.size()] != '\\')) continue;
            auto const found = paths.find({root, std::string_view(filename).substr(offset)});
            if (found != paths.end() && (found->second->plot_id.empty() || found->second->plot_id == id)) { covered = true; break; }
        }
        inventory_detail::require(covered || (published && published(filename, id)),
            "loaded plot is outside the reviewed reader/catalog scope; inventory every loaded source root and review the complete workload");
    }
}
inline void require_registry_scope(Session& session, Json const& response, FarmInventory const& inventory,
    HarvesterAdapterOptions const& options, PublishedPlotAllowance const& published = {}, RegistryScopeIndex const* index = nullptr)
{ require_plot_scope(session, inventory_detail::required_field(response, "plots"), inventory, options, published, index); }
} // namespace adapter_detail
inline void observe_registry_scope(HarvesterAdapterOptions const& options, FarmInventory const& inventory,
    PublishedPlotAllowance const& published = {}, RegistryScopeIndex const* index = nullptr)
{
    adapter_detail::Session session(options); auto const version = adapter_detail::version(session.rpc(AdapterOperation::harvester_version));
    adapter_detail::require_reviewed_reader(session, inventory, options, version, index);
    auto response = session.rpc(AdapterOperation::harvester_plots);
    adapter_detail::require_registry_scope(session, response, inventory, options, published, index);
    inventory_detail::require(adapter_detail::version(session.rpc(AdapterOperation::harvester_version)) == version,
        "source reader version changed during the registry scope observation");
    session.unchanged();
}
inline TargetNetworkReadiness observe_target_network(HarvesterAdapterOptions const& options, NetworkProjection const& reviewed, uint8_t k, uint8_t strength)
{
    TargetNetworkReadiness result;
    try
    {
        adapter_detail::Session session(options); result.observed = adapter_detail::projection(session);
        result.observed.target_route = reviewed.target_route;
        (void)adapter_detail::reviewed_target(session, options, reviewed, result.observed, k, strength);
        session.unchanged(); result.ready = true;
        result.reason = "same reviewed target reader/network is synced, active and compatible; this attests no replacement file";
    }
    catch (FarmServiceUnavailable const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.ready = false; result.retryable = true; result.reason = error.what(); }
    catch (std::exception const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.ready = false; result.reason = error.what(); }
    return result;
}
inline TargetPrerequisites observe_target_prerequisites(HarvesterAdapterOptions const& options, NetworkProjection const& reviewed,
    uint8_t k, uint8_t strength, TargetServiceExpectation const& expected)
{
    TargetPrerequisites result;
    try
    {
        adapter_detail::Session session(options); result.observed = adapter_detail::projection(session);
        result.observed.target_route = reviewed.target_route;
        (void)adapter_detail::reviewed_target(session, options, reviewed, result.observed, k, strength);
        adapter_detail::TargetServices services(session, options, reviewed, result.observed, expected); services.finish();
        result.ready = true; result.pool_mode = services.pool_mode; result.pool = services.pool; result.other_harvesters = services.other_harvesters;
        result.reason = result.observed.service_status + "; exact new plot/key acknowledgement awaits publication";
    }
    catch (FarmServiceUnavailable const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.ready = false; result.retryable = true; result.reason = error.what(); }
    catch (std::exception const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.reason = error.what(); }
    return result;
}
namespace adapter_detail
{
inline void observe_inventory(FarmInventory& inventory, HarvesterAdapterOptions const& options, Session& session,
    std::string const& version, Json const& response, bool complete_registry)
{
    inventory_detail::validate(inventory, options.limits);
    for (auto const* text : {&options.host, &options.harvester, &options.solver}) inventory_detail::bounded_text(*text);
    uint64_t retained = farm_inventory_memory(inventory);
    auto const& plots = inventory_detail::required_field(response, "plots");
    inventory_detail::require(plots.kind == profile_detail::Json::array && plots.items.size() <= options.limits.max_files, "native plot inventory exceeds its bound");
    if (complete_registry) adapter_detail::require_registry_scope(session, response, inventory, options);
    inventory_detail::require(inventory.plots.size() <= options.limits.memory_bytes / 4096,
        "native inventory lookup exceeds its private memory grant");
    std::map<std::string, size_t> paths;
    std::set<size_t> loaded;
    std::vector<bool> duplicate(inventory.plots.size(), false);
    for (size_t i = 0; i < inventory.plots.size(); ++i)
    {
        auto& plot = inventory.plots[i];
        if (plot.harvester.empty() || options.harvester.empty() || plot.harvester == options.harvester)
        {
            auto const before = inventory_detail::plot_memory(plot); plot.native_identified = false;
            if (!plot.header_validated && plot.status == "native-observed")
            { plot.status = "protected"; plot.reason = "previous reader metadata is not a fresh loaded-file observation"; }
            auto const after = inventory_detail::plot_memory(plot);
            if (after >= before) retained += after - before; else retained -= before - after;
        }
        auto filename = file_io::path_text(file_io::native_path(inventory.roots.at(plot.root).path) / inventory_detail::relative(plot.relative_path));
#ifdef _WIN32
        std::replace(filename.begin(), filename.end(), '\\', '/');
#endif
        paths.emplace(std::move(filename), i);
        if (inventory.plots[i].alias_of) { duplicate[i] = true; duplicate[*inventory.plots[i].alias_of] = true; }
    }
    // Native object reads share a count-bounded catalog grant. No proof
    // reconstruction is required to retire an exact reviewed object.
    if (complete_registry) session.begin_inventory_files(inventory.plots.size());
    for (auto const& native : plots.items)
    {
        session.running();
        auto filename = inventory_detail::string(native, "filename"); auto at = paths.end();
#ifdef _WIN32
        std::replace(filename.begin(), filename.end(), '\\', '/');
#endif
        at = paths.find(filename);
        if (at == paths.end()) continue;
        inventory_detail::require(loaded.insert(at->second).second, "native reader reports duplicate source path records");
        auto& plot = inventory.plots[at->second];
        if (!plot.harvester.empty() && !options.harvester.empty() && plot.harvester != options.harvester) continue;
        inventory_detail::require(retained + 128 * 1024 < options.limits.memory_bytes / 2,
            "native inventory observation exceeds its retained catalog grant");
        auto const before_memory = inventory_detail::plot_memory(plot);
        auto const current = observe_inventory_file(inventory.roots[plot.root], plot.relative_path, session.remaining_limits());
        session.running();
        inventory_detail::require(matches_inventory_identity(plot.identity, current.identity) && inventory_detail::integer(native, "file_size") == plot.identity.stamp.bytes &&
            inventory_detail::integer(native, "time_modified") == uint64_t(plot.identity.stamp.mtime), "native loaded plot differs from the observed file");
        auto const id = adapter_detail::hex_string(inventory_detail::required_field(native, "plot_id"), 32);
        inventory_detail::require(plot.plot_id.empty() || plot.plot_id == id, "native reader and public header disagree on plot identity");
        inventory_detail::require(plot.reader_identity.empty() || (session.matches_reader_identity(plot.reader_identity) && plot.reader_version == version &&
            plot.reader == adapter_detail::name(options.kind)), "native reader/configuration implementation changed since review; create a new reviewed inventory/profile");
        plot.reader = adapter_detail::name(options.kind); plot.reader_version = version; plot.plot_id = id;
        plot.reader_identity = session.reader_identity();
        auto const size = inventory_detail::integer(native, "size");
        inventory_detail::require(size <= 255, "native reader returned invalid plot parameters");
        if (plot.header_validated) inventory_detail::require(size == (plot.algorithm == PlotAlgorithm::pos1 ? plot.k : uint64_t(0x80 | plot.strength)),
            "native reader and public header plot parameters disagree");
        if (!plot.header_validated && size >= 18 && size <= 50) { plot.algorithm = PlotAlgorithm::pos1; plot.k = uint8_t(size); }
        if (auto const* compression = profile_detail::field(native, "compression_level"))
        {
            auto level = profile_detail::integer(*compression); inventory_detail::require(level <= 255, "native reader returned invalid compression level");
            if (!plot.header_validated) { plot.compression_level = uint8_t(level); plot.compression_family = "native-reported; format unknown"; }
            else inventory_detail::require(level == plot.compression_level, "native reader and public header compression disagree");
        }
        for (auto const& [name, target, bytes] : {std::tuple{"pool_public_key", &plot.pool_public_key, size_t(48)},
            std::tuple{"pool_contract_puzzle_hash", &plot.pool_contract_puzzle_hash, size_t(32)}})
            if (auto const* key = profile_detail::field(native, name); key && key->kind != profile_detail::Json::null)
            {
                auto value = adapter_detail::hex_string(*key, bytes);
                inventory_detail::require(target->empty() || *target == value, "native reader and public header pool route disagree"); *target = std::move(value);
            }
        inventory_detail::require(plot.pool_public_key.empty() != plot.pool_contract_puzzle_hash.empty(), "native reader must identify exactly one public pool route");
        plot.plot_public_key = adapter_detail::hex_string(inventory_detail::required_field(native, "plot_public_key"), 48);
        if (plot.algorithm == PlotAlgorithm::pos1)
            inventory_detail::require(adapter_detail::legacy_id_from_loaded_key(plot.plot_public_key,
                plot.pool_public_key.empty() ? plot.pool_contract_puzzle_hash : plot.pool_public_key) == plot.plot_id,
                "native public plot key/pool route does not bind the observed legacy plot ID");
        plot.harvester = options.harvester; plot.solver = options.solver;
        if (!options.host.empty()) plot.host = options.host;
        if (plot.algorithm == PlotAlgorithm::pos1) { plot.native_identified = true; plot.members = 1; }
        if (!plot.header_validated && plot.algorithm == PlotAlgorithm::pos1)
        {
            // The reader supplies public metadata; the native file stamp binds
            // the reviewed object. This does not attest old proof validity or
            // invent a proprietary header grammar.
            plot.native_identified = true;
            if (plot.identity.links == 1 && !duplicate[at->second]) plot.status = "native-observed";
            plot.reason = "exact native file with reader-reported public metadata; old proof validity untested";
        }
        auto const after_memory = inventory_detail::plot_memory(plot);
        if (after_memory >= before_memory) retained += after_memory - before_memory; else retained -= before_memory - after_memory;
    }
    for (size_t i = 0; i < inventory.plots.size(); ++i)
    {
        auto& plot = inventory.plots[i];
        if (loaded.contains(i) || !plot.header_validated || plot.algorithm != PlotAlgorithm::pos1 ||
            (plot.container != PlotContainer::chiapos_v1 && plot.container != PlotContainer::chiapos_v2) ||
            (!plot.harvester.empty() && !options.harvester.empty() && plot.harvester != options.harvester)) continue;
        session.running();
        inventory_detail::require(plot.reader_identity.empty() || (session.matches_reader_identity(plot.reader_identity) &&
            plot.reader_version == version && plot.reader == adapter_detail::name(options.kind)),
            "ordinary source reader/configuration implementation changed since review");
        (void)recheck_inventory_source(inventory.roots.at(plot.root), plot, session.remaining_limits());
        auto const before = inventory_detail::plot_memory(plot);
        plot.reader = adapter_detail::name(options.kind); plot.reader_version = version; plot.reader_identity = session.reader_identity();
        plot.harvester = options.harvester; plot.solver = options.solver; if (!options.host.empty()) plot.host = options.host;
        if (plot.status != "protected") plot.reason = "ordinary public header; reader registry reports no loaded record; route context observed, old proof validity untested";
        auto const after = inventory_detail::plot_memory(plot);
        if (after >= before) retained += after - before; else retained -= before - after;
        inventory_detail::require(retained + 128 * 1024 < options.limits.memory_bytes / 2,
            "ordinary source context exceeds its retained catalog grant");
    }
    inventory_detail::require(adapter_detail::version(session.rpc(AdapterOperation::harvester_version)) == version,
        "native reader service version changed during catalog observation");
    session.unchanged();
}
} // namespace adapter_detail
inline void observe_harvester_inventory(FarmInventory& inventory, HarvesterAdapterOptions const& options, bool complete_registry = true,
    std::vector<ResourcePreparationFact>* preparation = nullptr)
{
    adapter_detail::Session session(options);
    auto const version = adapter_detail::version(session.rpc(AdapterOperation::harvester_version));
    auto response = session.rpc(AdapterOperation::harvester_plots);
    adapter_detail::observe_inventory(inventory, options, session, version, response, complete_registry);
    record_retirement_reader(inventory, session);
    if (preparation)
    {
        inventory_detail::require(preparation->size() < max_preparation_entries, "too many reader configuration observations");
        preparation->push_back({"readers[" + session.reader_identity() + "].configured_num_threads", std::to_string(session.configured_lookup_workers()),
            "observed", session.source(), "Selected config harvester.num_threads; running service may need restart. Not an OS CPU cap, whole-service queue bound or spare-resource grant."});
    }
}
inline void observe_harvester_inventory(FarmInventory& inventory, FarmAdapterTable const& table,
    std::vector<ResourcePreparationFact>* preparation = nullptr)
{
    std::set<std::string> selected;
    for (auto& plot : inventory.plots)
    { auto options = resolve_source(table, inventory.roots.at(plot.root), plot); plot.harvester = options.harvester; selected.insert(options.harvester); }
    for (auto const& route : table.routes) if (selected.contains(route.id)) observe_harvester_inventory(inventory, route.options, true, preparation);
    auto const target = resolve_target(table);
    if (!selected.contains(table.target_route)) observe_registry_scope(target, inventory);
    inventory.network = observe_network_projection(target, &inventory);
    inventory.network.target_route = table.target_route;
}
// One authoritative reader snapshot for a route and phase. The caller pins
// every selected native object before claiming it; this object retains only
// bounded public metadata and never grants deletion permission by itself.
class SourceReadinessSnapshot
{
    HarvesterAdapterOptions options_;
    adapter_detail::Session session_;
    FarmInventory current_;
    std::string version_, registry_identity_;
    FarmInventory const& inventory_;
    RegistryScopeIndex const* scope_ = nullptr;
    PublishedPlotAllowance published_;
    std::map<std::pair<std::string, std::string>, size_t> paths_;
    bool finished_ = false, allow_unloaded_ = false;
public:
    SourceReadinessSnapshot(HarvesterAdapterOptions const& options, FarmInventory const& inventory, std::span<uint64_t const> sources,
        PublishedPlotAllowance const& published = {}, bool allow_unloaded = false, RegistryScopeIndex const* scope = nullptr)
        : options_(options), session_(options), inventory_(inventory), scope_(scope), published_(published), allow_unloaded_(allow_unloaded)
    {
        inventory_detail::require(!sources.empty() && sources.size() <= options.limits.max_files &&
            sources.size() <= options.limits.memory_bytes / 32768, "source snapshot exceeds its retained metadata grant");
        current_.version = inventory.version; current_.host = inventory.host; current_.captured_at = inventory_detail::timestamp(); current_.producer_build = GROW_IDENTITY;
        current_.roots = inventory.roots; current_.complete = true;
        for (auto index : sources)
        {
            inventory_detail::require(index < inventory.plots.size(), "source snapshot index is invalid");
            auto const& reviewed = inventory.plots[size_t(index)]; auto plot = reviewed;
            inventory_detail::require(plot.root < inventory.roots.size() && plot.identity.links == 1 && !plot.alias_of &&
                (options.harvester.empty() || plot.harvester.empty() || options.harvester == plot.harvester),
                "source snapshot crosses a reviewed route or lacks unique native identity");
            auto const key = std::pair(current_.roots[plot.root].path, plot.relative_path);
            inventory_detail::require(paths_.emplace(key, current_.plots.size()).second, "source snapshot repeats a path");
            if (options.revalidation) if (auto current = options.revalidation->expected(reviewed))
            { plot.reader = current->reader; plot.reader_version = current->version; plot.reader_identity = current->identity; }
            plot.native_identified = false; current_.plots.push_back(std::move(plot));
        }
        uint64_t index_bytes = 0;
        for (auto const& [key, index] : paths_)
        {
            (void)index;
            auto const bytes = uint64_t(key.first.capacity()) + key.second.capacity() + 256;
            inventory_detail::require(bytes <= options.limits.memory_bytes - index_bytes, "source snapshot path index exceeds its private grant");
            index_bytes += bytes;
        }
        auto charge = [&]
        { inventory_detail::require(farm_inventory_memory(current_) <= options.limits.memory_bytes / 2 &&
            index_bytes <= options.limits.memory_bytes / 2 - farm_inventory_memory(current_), "source snapshot copied roots, metadata and path index exceed their retained grant"); };
        charge();
        version_ = adapter_detail::version(session_.rpc(AdapterOperation::harvester_version));
        adapter_detail::require_reviewed_reader(session_, inventory_, options_, version_, scope_);
        auto response = session_.rpc(AdapterOperation::harvester_plots);
        adapter_detail::require_registry_scope(session_, response, inventory, options_, published, scope_);
        registry_identity_ = adapter_detail::registry_identity(response);
        adapter_detail::observe_inventory(current_, options_, session_, version_, response, false); charge();
    }
    InventoryPlot check(InventoryRoot const& root, InventoryPlot const& reviewed)
    {
        inventory_detail::require(!finished_, "source snapshot is already finished"); session_.running();
        auto at = paths_.find({root.path, reviewed.relative_path});
        inventory_detail::require(at != paths_.end(), "source is absent from the route snapshot");
        auto const& current = current_.plots.at(at->second);
        if (!current.native_identified && reviewed.header_validated && reviewed.algorithm == PlotAlgorithm::pos1 &&
            (reviewed.container == PlotContainer::chiapos_v1 || reviewed.container == PlotContainer::chiapos_v2))
        {
            inventory_detail::require(reviewed.reader_identity.empty() || adapter_detail::matches_reviewed_reader(session_, options_, reviewed, version_),
                "previously observed source reader/configuration changed since review");
            auto observed = observe_inventory_file(root, reviewed.relative_path, session_.remaining_limits());
            inventory_detail::require(matches_inventory_identity(reviewed.identity, observed.identity) && observed.header_validated && observed.algorithm == reviewed.algorithm &&
                observed.container == reviewed.container && observed.k == reviewed.k && observed.plot_id == reviewed.plot_id &&
                observed.farmer_public_key == reviewed.farmer_public_key && observed.pool_public_key == reviewed.pool_public_key &&
                observed.pool_contract_puzzle_hash == reviewed.pool_contract_puzzle_hash && observed.compression_level == reviewed.compression_level,
                "unloaded ordinary source public header or native object differs from review");
            observed.root = reviewed.root; observed.harvester = reviewed.harvester; observed.workload_id = reviewed.workload_id;
            observed.reason = "exact reviewed native object and ordinary public header; currently unloaded, no fresh reader acknowledgement or old proof validity claimed";
            return observed;
        }
        if (!current.native_identified && allow_unloaded_)
        {
            inventory_detail::require(reviewed.native_identified && reviewed.algorithm == PlotAlgorithm::pos1 && reviewed.k >= 18 && reviewed.k <= 50 &&
                (reviewed.pool_public_key.empty() != reviewed.pool_contract_puzzle_hash.empty()) &&
                adapter_detail::legacy_id_from_loaded_key(reviewed.plot_public_key,
                    reviewed.pool_public_key.empty() ? reviewed.pool_contract_puzzle_hash : reviewed.pool_public_key) == reviewed.plot_id &&
                adapter_detail::matches_reviewed_reader(session_, options_, reviewed, version_),
                "unloaded source lacks the exact previously reviewed reader metadata");
            (void)recheck_inventory_source(root, reviewed, session_.remaining_limits());
            auto observed = reviewed;
            observed.reason = "exact reviewed native object is currently unloaded; public identity retained from the authorized review, old proof validity untested";
            return observed;
        }
        inventory_detail::require(current.reader_identity == session_.reader_identity() && current.reader_version == version_ &&
            current.reader == adapter_detail::name(options_.kind) && matches_inventory_identity(reviewed.identity, current.identity) &&
            current.plot_id == reviewed.plot_id && current.k == reviewed.k && current.algorithm == reviewed.algorithm &&
            (reviewed.plot_public_key.empty() || current.plot_public_key == reviewed.plot_public_key) && current.pool_public_key == reviewed.pool_public_key &&
            current.pool_contract_puzzle_hash == reviewed.pool_contract_puzzle_hash && current.compression_level == reviewed.compression_level &&
            current.native_identified && (reviewed.header_validated || current.status == "native-observed"),
            "fresh authoritative source metadata differs from the reviewed native object");
        (void)recheck_inventory_source(root, reviewed, session_.remaining_limits()); return current;
    }
    void finish()
    {
        inventory_detail::require(!finished_, "source snapshot is already finished");
        inventory_detail::require(adapter_detail::version(session_.rpc(AdapterOperation::harvester_version)) == version_,
            "source reader version changed during the batch snapshot");
        auto response = session_.rpc(AdapterOperation::harvester_plots);
        adapter_detail::require_registry_scope(session_, response, inventory_, options_, published_, scope_);
        inventory_detail::require(adapter_detail::registry_identity(response) == registry_identity_,
            "source loaded population or reader metadata changed during the batch snapshot");
        session_.unchanged(); finished_ = true;
    }
};
namespace adapter_detail
{
inline void require_source_resource_binding(HarvesterAdapterOptions const& options, InventoryPlot const& plot)
{
    // Before reader identification an opaque row's host is native file
    // locality, not an observed remote decoder binding. Supported headers and
    // previously identified rows retain every known physical source label.
    if (plot.header_validated || plot.native_identified || !plot.reader_identity.empty())
    {
        auto const host = options.host.empty() ? profile_detail::host_name() : options.host;
        inventory_detail::require(plot.host.empty() || plot.host == host, "revalidation changed an original source physical host binding");
    }
    inventory_detail::require(plot.solver.empty() || plot.solver == options.solver, "revalidation changed an original source solver binding");
}
inline void require_matching_catalog_metadata(Json const& native, InventoryPlot const& reviewed)
{
    using namespace inventory_detail;
    auto const id = hex_string(required_field(native, "plot_id"), 32);
    require(reviewed.plot_id.empty() || id == reviewed.plot_id, "fresh reader changed an original plot identity");
    if (reviewed.identity.stamp.inode)
        require(integer(native, "file_size") == reviewed.identity.stamp.bytes && integer(native, "time_modified") == uint64_t(reviewed.identity.stamp.mtime),
            "fresh reader metadata differs from the original native file");
    auto const size = integer(native, "size"); require(size <= 255, "fresh reader returned invalid plot parameters");
    if (reviewed.algorithm == PlotAlgorithm::pos1 && reviewed.k) require(size == reviewed.k, "fresh reader changed original PoS1 parameters");
    if (reviewed.algorithm == PlotAlgorithm::pos2 && reviewed.strength) require(size == uint64_t(0x80 | reviewed.strength), "fresh reader changed original PoS2 parameters");
    if (auto const* compression = profile_detail::field(native, "compression_level"))
    {
        if (reviewed.header_validated || reviewed.native_identified)
            require(profile_detail::integer(*compression) == reviewed.compression_level, "fresh reader changed original compression metadata");
    }
    else require(!reviewed.native_identified || reviewed.header_validated || !reviewed.compression_level,
        "fresh reader cannot support the original opaque compression metadata");
    for (auto const& [name, value, bytes] : {std::tuple{"plot_public_key", &reviewed.plot_public_key, size_t(48)},
        std::tuple{"pool_public_key", &reviewed.pool_public_key, size_t(48)},
        std::tuple{"pool_contract_puzzle_hash", &reviewed.pool_contract_puzzle_hash, size_t(32)}})
    {
        auto const* field = profile_detail::field(native, name);
        if (!value->empty()) require(field && field->kind != Json::null && hex_string(*field, bytes) == *value, "fresh reader changed an original public key or pool route");
        else if (name != std::string_view("plot_public_key") && (!reviewed.pool_public_key.empty() || !reviewed.pool_contract_puzzle_hash.empty()))
            require(!field || field->kind == Json::null, "fresh reader added an incompatible original pool route");
    }
    if (reviewed.algorithm == PlotAlgorithm::pos2) if (auto const* members = profile_detail::field(native, "group_size"))
        require(profile_detail::integer(*members) == reviewed.members, "fresh reader changed an original group membership");
}
} // namespace adapter_detail
inline uint64_t replot_reader_refresh_memory_bound(FarmInventory const& inventory)
{
    using namespace resource_detail;
    return add(4096, add(multiply(inventory.plots.size(), 256), multiply(inventory.roots.size(), 4096)));
}
namespace adapter_detail
{
struct ReplotReaderSnapshot
{
    std::vector<ReplotReaderObservation> readers;
    std::vector<RetirementReaderContext> retirement;
    NetworkProjection network;
};
// Builder refresh and successor activation share the complete native catalog
// comparison. Validation retains the caller's native-audited allowances and
// requires reviewed tuples; a fresh builder observation clears those aliases.
inline ReplotReaderSnapshot observe_replot_reader_observations(FarmInventory const& inventory,
    FarmAdapterTable const& table, HarvesterAdapterOptions const& fallback, std::span<uint64_t const> native_audited_nonactive,
    PublishedPlotAllowance const& published, RegistryScopeIndex const* scope, uint64_t working_memory_bytes, bool require_reviewed)
{
    using namespace inventory_detail;
    require(replot_reader_refresh_memory_bound(inventory) <= (working_memory_bytes ? working_memory_bytes : fallback.limits.memory_bytes),
        "reader refresh exceeds its caller-reserved working memory grant");
    if (scope) require(&scope->inventory() == &inventory, "reader refresh registry index belongs to another catalog");
    std::vector<bool> nonactive(inventory.plots.size(), false), loaded(inventory.plots.size(), false);
    for (auto index : native_audited_nonactive)
    { require(index < nonactive.size() && !nonactive[size_t(index)], "reader refresh has an invalid or duplicate native-audited source"); nonactive[size_t(index)] = true; }
    struct Route { HarvesterAdapterOptions options; std::vector<size_t> roots, plots; bool target = false; };
    std::map<std::string, Route> routes; std::vector<std::string> root_routes(inventory.roots.size());
    auto clean = [require_reviewed](HarvesterAdapterOptions options)
    { if (!require_reviewed) { options.retirement = nullptr; options.revalidation = nullptr; } return options; };
    auto target = clean(table.routes.empty() ? fallback : resolve_target(table, inventory.network.target_route));
    require(inventory.network.target_route.empty() || target.harvester == inventory.network.target_route, "revalidation changed the target route");
    routes.emplace(target.harvester, Route{target, {}, {}, true});
    auto nonplot = [](InventoryPlot const& plot) { return plot.status == "non-plot" && !plot.header_validated && !plot.reason.empty(); };
    std::vector<bool> data_roots(inventory.roots.size(), false);
    for (auto const& plot : inventory.plots) if (!nonplot(plot)) data_roots.at(plot.root) = true;
    for (size_t root = 0; root < inventory.roots.size(); ++root)
    {
        if (!data_roots[root]) continue;
        InventoryPlot probe; probe.root = root;
        auto options = clean(table.routes.empty() ? fallback : resolve_source(table, inventory.roots[root], probe));
        root_routes[root] = options.harvester;
        auto [at, fresh] = routes.emplace(options.harvester, Route{options, {}, {}, false});
        (void)fresh; at->second.roots.push_back(root);
    }
    for (size_t i = 0; i < inventory.plots.size(); ++i)
    {
        auto const& plot = inventory.plots[i]; if (nonplot(plot)) continue;
        auto& route = routes.at(root_routes.at(plot.root));
        require(plot.harvester.empty() || plot.harvester == route.options.harvester, "revalidation changed an original source route");
        adapter_detail::require_source_resource_binding(route.options, plot);
        route.plots.push_back(i);
    }
    std::vector<ReplotReaderObservation> observations; std::vector<RetirementReaderContext> retirement;
    std::string target_identity, target_version;
    for (auto const& [id, route] : routes)
    {
        (void)id; auto const& options = route.options; adapter_detail::Session session(options);
        auto const version = adapter_detail::version(session.rpc(AdapterOperation::harvester_version));
        if (require_reviewed)
        {
            require_reviewed_reader(session, inventory, options, version, scope);
            if (route.target) require(session.matches_reader_identity(inventory.network.reader_identity) && version == inventory.network.reader_version,
                "current target reader differs from the reviewed successor");
        }
        auto response = session.rpc(AdapterOperation::harvester_plots);
        auto const allowance = route.target ? published : PublishedPlotAllowance{};
        adapter_detail::require_registry_scope(session, response, inventory, options, allowance, scope);
        auto const registry = adapter_detail::registry_identity(response);
        std::map<std::pair<size_t, std::string_view>, size_t> paths;
        for (auto index : route.plots) paths.emplace(std::pair(inventory.plots[index].root, std::string_view(inventory.plots[index].relative_path)), index);
        for (auto const& native : required_field(response, "plots").items)
        {
            auto filename = string(native, "filename");
#ifdef _WIN32
            std::replace(filename.begin(), filename.end(), '\\', '/');
#endif
            for (auto root : route.roots)
            {
                auto prefix = inventory.roots[root].path;
#ifdef _WIN32
                std::replace(prefix.begin(), prefix.end(), '\\', '/');
#endif
                bool const separator = !prefix.empty() && (prefix.back() == '/' || prefix.back() == '\\'); size_t const offset = prefix.size() + !separator;
                if (filename.size() <= offset || filename.compare(0, prefix.size(), prefix) || (!separator && filename[prefix.size()] != '/')) continue;
                auto at = paths.find({root, std::string_view(filename).substr(offset)}); if (at == paths.end()) continue;
                require(!loaded[at->second], "fresh reader repeats original source metadata"); loaded[at->second] = true;
                adapter_detail::require_matching_catalog_metadata(native, inventory.plots[at->second]);
            }
        }
        for (auto index : route.plots)
        {
            auto const& plot = inventory.plots[index];
            require(nonactive[index] || !plot.native_identified || plot.header_validated || loaded[index],
                "active opaque source lacks fresh matching reader metadata; reload its original reader before revalidation");
        }
        require(adapter_detail::version(session.rpc(AdapterOperation::harvester_version)) == version, "reader version changed during revalidation");
        auto final = session.rpc(AdapterOperation::harvester_plots);
        adapter_detail::require_registry_scope(session, final, inventory, options, allowance, scope);
        require(adapter_detail::registry_identity(final) == registry, "reader catalog changed during revalidation"); session.unchanged();
        auto const captured_at = timestamp();
        for (auto root : route.roots) observations.push_back({uint64_t(root), adapter_detail::name(options.kind), version, session.reader_identity(), session.source(), captured_at});
        auto const& context = session.retirement_context();
        if (!context.retirement_identity.empty() && std::none_of(retirement.begin(), retirement.end(), [&](auto const& old) { return old.reader_identity == context.reader_identity; })) retirement.push_back(context);
        if (route.target) { target_identity = session.reader_identity(); target_version = version; }
    }
    FarmInventory network_context; auto network = observe_network_projection(target, &network_context); network.target_route = inventory.network.target_route;
    auto const& prior = inventory.network;
    require(network.available && network.reader_identity == target_identity && network.reader_version == target_version && network.network == prior.network &&
        network.genesis_challenge == prior.genesis_challenge && network.hard_fork2_height == prior.hard_fork2_height && network.min_strength == prior.min_strength &&
        network.max_strength == prior.max_strength && network.plot_size == prior.plot_size, "revalidation changed original network/consensus parameters or the current target reader");
    if (require_reviewed) require(network.version == prior.version, "current full-node version differs from the reviewed successor");
    for (auto const& context : network_context.retirement_readers) if (std::none_of(retirement.begin(), retirement.end(), [&](auto const& old) { return old.reader_identity == context.reader_identity; })) retirement.push_back(context);
    return {std::move(observations), std::move(retirement), std::move(network)};
}
} // namespace adapter_detail
// The caller first audits EVERY original native object/path under its stopped
// job locks, including protected rows and durable claim/absence names. This
// helper only refreshes observations in a provisional successor; it grants no
// claim, unlink, reservation release or unchecked identification capability.
// Borrowed registry indexes must be rebuilt after the active tuples change.
inline std::vector<ReplotReaderObservation> refresh_replot_reader_observations(FarmInventory& inventory,
    FarmAdapterTable const& table, HarvesterAdapterOptions const& fallback, std::span<uint64_t const> native_audited_nonactive,
    PublishedPlotAllowance const& published = {}, RegistryScopeIndex const* scope = nullptr, uint64_t working_memory_bytes = 0)
{
    auto snapshot = adapter_detail::observe_replot_reader_observations(inventory, table, fallback, native_audited_nonactive,
        published, scope, working_memory_bytes, false);
    std::vector<bool> nonactive(inventory.plots.size(), false);
    for (auto index : native_audited_nonactive) nonactive[size_t(index)] = true;
    std::map<size_t, ReplotReaderObservation const*> current;
    for (auto const& observation : snapshot.readers) current.emplace(size_t(observation.root), &observation);
    for (size_t i = 0; i < inventory.plots.size(); ++i) if (!nonactive[i])
    {
        auto& plot = inventory.plots[i]; if (plot.status == "non-plot" && !plot.header_validated && !plot.reason.empty()) continue;
        auto const& observation = *current.at(plot.root);
        // Preserve entirely unknown/protected identification as unknown.
        if (!plot.reader_identity.empty() || plot.header_validated)
        { plot.reader = observation.reader; plot.reader_version = observation.version; plot.reader_identity = observation.identity; }
    }
    inventory.network = std::move(snapshot.network); inventory.retirement_readers = std::move(snapshot.retirement);
    return std::move(snapshot.readers);
}
// Before activating a public successor, repeat the builder's complete catalog
// check against the native-audited origin. This is read-only and preserves
// legitimate removal-only configuration and nonactive-reader allowances.
inline void validate_replot_reader_observations(FarmInventory const& inventory,
    FarmAdapterTable const& table, HarvesterAdapterOptions const& fallback, std::span<uint64_t const> native_audited_nonactive,
    PublishedPlotAllowance const& published = {}, RegistryScopeIndex const* scope = nullptr, uint64_t working_memory_bytes = 0)
{
    (void)adapter_detail::observe_replot_reader_observations(inventory, table, fallback, native_audited_nonactive,
        published, scope, working_memory_bytes, true);
}
// Every batch receives its own absence result from the same fresh route
// registry. No result survives a protected phase or grants retirement itself.
inline SourceRetirementSnapshot observe_source_batches_retired(HarvesterAdapterOptions const& options,
    FarmInventory const& inventory, std::span<std::span<uint64_t const> const> batches,
    PublishedPlotAllowance const& published = {}, RegistryScopeIndex const* scope = nullptr)
{
    inventory_detail::require(!batches.empty() && batches.size() <= options.limits.max_files && batches.size() <= options.limits.memory_bytes / 1024,
        "source registry batches exceed their bounded lookup memory");
    SourceRetirementSnapshot result; result.captured_at = inventory_detail::timestamp();
    result.no_loaded_records.resize(batches.size(), false);
    result.lookup_bounded.reserve(batches.size());
    {
        adapter_detail::Session session(options); result.source = session.source();
        for (auto sources : batches)
        {
            session.running();
            result.lookup_bounded.push_back(!sources.empty() && sources.size() <= options.limits.max_files && sources.size() <= options.limits.memory_bytes / 32768);
            if (!result.lookup_bounded.back()) continue;
            for (auto index : sources)
            {
                session.running();
                inventory_detail::require(index < inventory.plots.size(), "source registry batch index is invalid"); auto const& p = inventory.plots[size_t(index)];
                inventory_detail::require(p.root < inventory.roots.size() && !p.plot_id.empty() &&
                    (options.harvester.empty() || p.harvester.empty() || options.harvester == p.harvester), "source registry batch crosses a reviewed route or lacks plot identity");
            }
        }
        auto version = adapter_detail::version(session.rpc(AdapterOperation::harvester_version));
        adapter_detail::require_reviewed_reader(session, inventory, options, version, scope);
        for (size_t batch = 0; batch < batches.size(); ++batch) if (result.lookup_bounded[batch]) for (auto index : batches[batch])
        {
            session.running();
            auto const& p = inventory.plots[size_t(index)];
            bool const header_only = p.reader_identity.empty() && p.header_validated && p.algorithm == PlotAlgorithm::pos1 &&
                (p.container == PlotContainer::chiapos_v1 || p.container == PlotContainer::chiapos_v2);
            inventory_detail::require(header_only || adapter_detail::matches_reviewed_reader(session, options, p, version),
                "source registry reader implementation changed or was not reviewed; retain old reservations");
        }
        auto response = session.rpc(AdapterOperation::harvester_plots); auto const& plots = inventory_detail::required_field(response, "plots");
        inventory_detail::require(plots.kind == profile_detail::Json::array && plots.items.size() <= options.limits.max_files,
            "native source registry snapshot exceeds its bound");
        adapter_detail::require_registry_scope(session, response, inventory, options, published, scope);
        // Keep lookup retention at the original per-workload bound, even when
        // the whole route has many more sources. Only native RPC and its DOM
        // are shared; each workload scans that same coherent snapshot.
        std::set<std::string> loaded_paths;
        for (auto const& p : plots.items)
        {
            session.running(); auto path = inventory_detail::string(p, "filename");
#ifdef _WIN32
            std::replace(path.begin(), path.end(), '\\', '/');
#endif
            inventory_detail::require(loaded_paths.insert(path).second, "native source registry repeats a source path");
        }
        // Discard duplicate-check path copies before building a source lookup.
        loaded_paths.clear();
        for (size_t batch = 0; batch < batches.size(); ++batch)
        {
            if (!result.lookup_bounded[batch]) continue;
            session.running(); std::set<std::string> paths, ids;
            for (auto index : batches[batch])
            {
                session.running(); auto const& p = inventory.plots[size_t(index)];
                auto path = file_io::path_text(file_io::native_path(inventory.roots[p.root].path) / inventory_detail::relative(p.relative_path));
#ifdef _WIN32
                std::replace(path.begin(), path.end(), '\\', '/');
#endif
                paths.insert(std::move(path)); ids.insert(p.plot_id);
            }
            bool loaded = false;
            for (auto const& p : plots.items)
            {
                session.running(); auto path = inventory_detail::string(p, "filename");
#ifdef _WIN32
                std::replace(path.begin(), path.end(), '\\', '/');
#endif
                if (paths.contains(path) || ids.contains(adapter_detail::hex_string(inventory_detail::required_field(p, "plot_id"), 32)))
                { loaded = true; break; }
            }
            result.no_loaded_records[batch] = !loaded;
        }
        inventory_detail::require(adapter_detail::version(session.rpc(AdapterOperation::harvester_version)) == version,
            "source registry reader version changed during absence observation; retain old reservations");
        session.unchanged(); result.observed = std::chrono::steady_clock::now();
    }
    return result;
}
inline SourceRetirementObservation observe_sources_retired(HarvesterAdapterOptions const& options, FarmInventory const& inventory,
    std::span<uint64_t const> sources, PublishedPlotAllowance const& published = {}, RegistryScopeIndex const* scope = nullptr)
{
    try
    {
        std::array<std::span<uint64_t const>, 1> batches{sources};
        auto snapshot = observe_source_batches_retired(options, inventory, batches, published, scope);
        SourceRetirementObservation result; result.no_loaded_record = snapshot.no_loaded_records.front();
        result.source = std::move(snapshot.source); result.captured_at = std::move(snapshot.captured_at); result.observed = snapshot.observed;
        result.reason = !snapshot.lookup_bounded.front() ? "source registry batch exceeds its bounded lookup memory; old arrivals remain reserved" :
            result.no_loaded_record ? "one fresh native registry snapshot has no batch source paths/IDs; queued/resident/opaque reservations remain separate" :
                "a batch source path or plot ID is still registered; old arrivals remain reserved";
        return result;
    }
    catch (std::exception const& error)
    {
        if (options.limits.cancelled && options.limits.cancelled()) throw;
        SourceRetirementObservation result; result.captured_at = inventory_detail::timestamp(); result.reason = error.what(); return result;
    }
}
inline SourceRetirementObservation observe_source_retired(HarvesterAdapterOptions const& options, InventoryRoot const& root, InventoryPlot const& plot)
{
    FarmInventory inventory; inventory.version = file_io::has_durable_identity(root.durable) ? 2 : 1;
    inventory.roots.push_back(root); inventory.plots.push_back(plot); inventory.plots[0].root = 0;
    std::array<uint64_t, 1> indices{0}; return observe_sources_retired(options, inventory, indices);
}
// One bounded live observation for one prefix audit. Stream expectations into
// check(), and finish before every retirement action while this phase lease
// remains live. Construction/proof phases invalidate it. It is never saved.
class TargetReadinessSnapshot
{
    HarvesterAdapterOptions options_;
    adapter_detail::Session session_;
    NetworkProjection projection_;
    std::string reviewed_reader_identity_;
    adapter_detail::Json response_;
    TargetServiceExpectation service_;
    std::unique_ptr<adapter_detail::TargetServices> services_;
    // Keys refer to immutable JSON storage, avoiding another full path copy.
    struct LoadedPlot
    {
        adapter_detail::Json const* plot;
        std::optional<stream_state::Stamp> checked;
        LoadedPlot(adapter_detail::Json const* value) : plot(value) {}
    };
    std::map<std::string_view, LoadedPlot, std::less<>> loaded_;
    std::map<std::string_view, adapter_detail::Json const*, std::less<>> farmer_loaded_;
    uint64_t peak_timestamp_ = 0;
    uint8_t k_ = 0, strength_ = 0;
    bool healthy_ = true;
    void current() const
    {
        session_.running(); inventory_detail::require(healthy_, "target readiness snapshot contains a failed replacement check");
        auto const now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        require_service_available(now >= 0 && peak_timestamp_ <= uint64_t(now) && uint64_t(now) - peak_timestamp_ <= options_.max_peak_age_seconds,
            "target readiness snapshot's transaction peak became stale");
    }
public:
    TargetReadinessSnapshot(HarvesterAdapterOptions const& options, NetworkProjection const& reviewed, uint8_t k, uint8_t strength,
        TargetServiceExpectation const& service, FarmInventory const* inventory = nullptr, PublishedPlotAllowance const& published = {},
        RegistryScopeIndex const* scope = nullptr)
        : options_(options), session_(options_), projection_(adapter_detail::projection(session_)), reviewed_reader_identity_(reviewed.reader_identity), service_(service), k_(k), strength_(strength)
    {
        projection_.target_route = reviewed.target_route;
        peak_timestamp_ = adapter_detail::reviewed_target(session_, options_, reviewed, projection_, k, strength);
        services_ = std::make_unique<adapter_detail::TargetServices>(session_, options_, reviewed, projection_, service_);
        response_ = session_.rpc(AdapterOperation::harvester_plots);
        if (inventory) adapter_detail::require_registry_scope(session_, response_, *inventory, options_, published, scope);
        if (inventory) adapter_detail::require_plot_scope(session_, services_->plots(), *inventory, options_, published, scope);
        // JSON already owns/costs its strings and nodes. The additional index
        // has no path copies and reserves at least 4096 bytes per entry, well
        // above its map-node storage, within the existing private grant.
        auto index = [&](adapter_detail::Json const& plots, auto& target)
        {
            inventory_detail::require(plots.kind == adapter_detail::Json::array && plots.items.size() <= options_.limits.max_files &&
                plots.items.size() <= options_.limits.memory_bytes / 4096, "target readiness index exceeds its bounded memory/count grant");
            for (auto const& plot : plots.items)
            {
                session_.running(); auto const& filename = inventory_detail::required_field(plot, "filename");
                inventory_detail::require(filename.kind == adapter_detail::Json::string, "target service plot filename is not text");
                inventory_detail::bounded_text(filename.scalar);
                inventory_detail::require(target.emplace(filename.scalar, &plot).second, "target service reports duplicate replacement paths");
            }
        };
        index(inventory_detail::required_field(response_, "plots"), loaded_); index(services_->plots(), farmer_loaded_);
        current(); session_.unchanged();
    }
    TargetReadinessSnapshot(TargetReadinessSnapshot const&) = delete;
    TargetReadinessSnapshot& operator=(TargetReadinessSnapshot const&) = delete;
    TargetReadinessSnapshot(TargetReadinessSnapshot&&) = delete;
    TargetReadinessSnapshot& operator=(TargetReadinessSnapshot&&) = delete;
    TargetPoolState const& pool_state() const { return services_->pool; }
    std::vector<OtherHarvesterObservation> const& other_harvesters() const { return services_->other_harvesters; }
    void finish()
    {
        try
        {
            current(); ReplacementExpectation expected; expected.k = k_; expected.strength = strength_;
            inventory_detail::require(adapter_detail::version(session_.rpc(AdapterOperation::harvester_version)) == projection_.reader_version &&
                adapter_detail::version(session_.rpc(AdapterOperation::full_node_version)) == projection_.version,
                "target harvester or full-node version changed during the readiness phase");
            expected.network = projection_.network; expected.genesis_challenge = projection_.genesis_challenge;
            peak_timestamp_ = adapter_detail::active_network(session_, options_, expected, projection_);
            services_->finish();
            for (auto const& [path, entry] : loaded_) if (entry.checked)
            {
                session_.running();
                inventory_detail::require(stream_state::stamp(file_io::native_path(path)) == *entry.checked,
                    "previously acknowledged replacement native path changed or disappeared during the readiness phase");
            }
            session_.unchanged();
        }
        catch (...) { healthy_ = false; throw; }
    }
    ReplacementReadiness check(ReplacementExpectation const& expected, ReplacementPublication const& publication)
    {
        ReplacementReadiness result; result.host = options_.host.empty() ? profile_detail::host_name() : options_.host;
        result.captured_at = projection_.captured_at; result.method = "authenticated-configured-native-rpc";
        result.network = projection_.network; result.version = projection_.reader_version;
        try
        {
            current(); adapter_detail::require_publication(expected, publication);
            inventory_detail::require(expected.k == k_ && expected.strength == strength_ && expected.network == projection_.network &&
                expected.genesis_challenge == projection_.genesis_challenge && expected.reader_identity == reviewed_reader_identity_ &&
                expected.reader_version == projection_.reader_version && expected.network_version == projection_.version && expected.target_route == projection_.target_route &&
                expected.farmer_version == projection_.farmer_version && expected.solver_version == projection_.solver_version &&
                expected.farmer_public_key == service_.farmer_public_key && expected.pool_public_key == service_.pool_public_key &&
                expected.pool_contract_puzzle_hash == service_.pool_contract_puzzle_hash,
                "replacement differs from the reviewed target snapshot parameters/provenance");
            inventory_detail::RootHandle root(expected.path.parent_path());
            auto file = inventory_detail::observe(root, file_io::path_text(expected.path.filename()), options_.limits);
            inventory_detail::require(matches_inventory_identity(expected.identity, file.identity) && file.header_validated && file.algorithm == PlotAlgorithm::pos2 &&
                file.plot_id == expected.group_id && file.k == expected.k && file.strength == expected.strength && file.meta_group == expected.meta_group && file.members == expected.members &&
                file.farmer_public_key == expected.farmer_public_key && file.pool_public_key == expected.pool_public_key && file.pool_contract_puzzle_hash == expected.pool_contract_puzzle_hash,
                "replacement file/header/public keys differ from the reviewed exact output");
            auto const at = loaded_.find(file_io::path_text(expected.path));
            auto const farmer_at = farmer_loaded_.find(file_io::path_text(expected.path));
            require_service_available(at != loaded_.end(), "exact replacement is not loaded by the configured harvester");
            require_service_available(farmer_at != farmer_loaded_.end(), "exact replacement has not reached the connected farmer's completed plot synchronization");
            for (auto const* record : {at->second.plot, farmer_at->second})
            {
                auto const& p = *record;
                inventory_detail::require(adapter_detail::hex_string(inventory_detail::required_field(p, "plot_id"), 32) == expected.group_id &&
                    inventory_detail::integer(p, "file_size") == expected.identity.stamp.bytes && inventory_detail::integer(p, "size") == uint64_t(0x80 | expected.strength) &&
                    inventory_detail::integer(p, "time_modified") == uint64_t(expected.identity.stamp.mtime), "native harvester replacement ID/parameters differ");
                if (auto const* meta = profile_detail::field(p, "meta_group"))
                    inventory_detail::require(profile_detail::integer(*meta) == expected.meta_group, "native harvester replacement meta group differs");
                if (auto const* count = profile_detail::field(p, "group_size"))
                    inventory_detail::require(profile_detail::integer(*count) == expected.members, "target service replacement group membership differs");
                else inventory_detail::require(record != farmer_at->second || expected.members == 1,
                    "connected farmer lacks exact multi-member group metadata; this replacement is unsupported");
                auto const* pool = profile_detail::field(p, expected.pool_public_key.empty() ? "pool_contract_puzzle_hash" : "pool_public_key");
                inventory_detail::require(pool && adapter_detail::hex_string(*pool, expected.pool_public_key.empty() ? 32 : 48) ==
                    (expected.pool_public_key.empty() ? expected.pool_contract_puzzle_hash : expected.pool_public_key), "native harvester replacement pool route differs");
                auto const public_key = adapter_detail::hex_string(inventory_detail::required_field(p, "plot_public_key"), 48);
                inventory_detail::require(adapter_detail::group_id_from_loaded_key(public_key,
                    expected.pool_public_key.empty() ? expected.pool_contract_puzzle_hash : expected.pool_public_key, expected.strength) == expected.group_id,
                    "native harvester plot public key does not bind the reviewed farmer/pool/group identity");
            }
            auto after = inventory_detail::observe(root, file_io::path_text(expected.path.filename()), options_.limits);
            inventory_detail::require(matches_inventory_identity(expected.identity, after.identity), "published replacement changed during live readiness observation");
            at->second.checked = expected.identity.stamp;
            current(); result.identity = file.identity; result.ready = true; result.status = "ready";
            result.reason = "new immutable publication matches the live harvester and connected farmer's completed sync, key route and solver prerequisites; finish the complete audit before retirement";
        }
        catch (FarmServiceUnavailable const& error)
        { healthy_ = false; if (options_.limits.cancelled && options_.limits.cancelled()) throw; result.ready = false; result.retryable = true; result.reason = error.what(); }
        catch (std::exception const& error)
        { healthy_ = false; if (options_.limits.cancelled && options_.limits.cancelled()) throw; result.ready = false; result.status = "unavailable"; result.reason = error.what(); }
        return result;
    }
};
inline ReplacementReadiness observe_replacement_readiness(HarvesterAdapterOptions const& options, ReplacementExpectation const& expected,
    std::optional<ReplacementPublication> const& publication = {})
{
    ReplacementReadiness result; result.host = options.host.empty() ? profile_detail::host_name() : options.host;
    result.captured_at = inventory_detail::timestamp(); result.method = "authenticated-configured-native-rpc";
    try
    {
        inventory_detail::require(publication.has_value(), "replacement requires the private journal's immutable first .plot2 publication receipt");
        adapter_detail::require_publication(expected, *publication);
        NetworkProjection reviewed; reviewed.available = true; reviewed.network = expected.network; reviewed.genesis_challenge = expected.genesis_challenge;
        reviewed.reader_identity = expected.reader_identity; reviewed.reader_version = expected.reader_version;
        reviewed.version = expected.network_version; reviewed.target_route = expected.target_route;
        reviewed.farmer_version = expected.farmer_version; reviewed.solver_version = expected.solver_version;
        TargetServiceExpectation service{expected.farmer_public_key, expected.pool_public_key, expected.pool_contract_puzzle_hash};
        TargetReadinessSnapshot snapshot(options, reviewed, expected.k, expected.strength, service);
        result = snapshot.check(expected, *publication); if (result.ready) snapshot.finish();
    }
    catch (FarmServiceUnavailable const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.ready = false; result.retryable = true; result.reason = error.what(); }
    catch (std::exception const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.ready = false; result.status = "unavailable"; result.reason = error.what(); }
    return result;
}
} // namespace group_grow::farm
