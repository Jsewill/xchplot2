// Observations from an explicitly selected installed harvester. The CLI uses
// its configured private-CA client authentication and explicit local service bindings.
#pragma once
#include "host/GroupFarmInventory.hpp"
#include "host/GroupFarmAdapterProcess.hpp"

namespace group_grow::farm
{
enum class HarvesterAdapterKind { chia, gigahorse, drchia };
struct FarmServiceBinding
{
    std::filesystem::path root;
    std::string rpc_host, host, node_id, farmer_node_id;
};
struct HarvesterAdapterOptions
{
    HarvesterAdapterKind kind = HarvesterAdapterKind::chia;
    std::filesystem::path executable, root;
    std::string host, harvester, solver, rpc_host;
    std::map<std::string, FarmServiceBinding> services;
    // Private operation budget. The caller separately reserves retained
    // catalog/plan memory before offering this additional process/parse grant.
    InventoryLimits limits;
    uint64_t max_peak_age_seconds = 600;
};
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
    bool ready = false;
    std::string status = "unavailable", reason, version, network, host, captured_at, method;
    InventoryFileIdentity identity;
};
struct TargetNetworkReadiness
{
    bool ready = false;
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
    bool ready = false;
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
class Session
{
    HarvesterAdapterOptions options_;
    std::chrono::steady_clock::time_point until_;
    inventory_detail::RootHandle root_;
    inventory_detail::SourceHandle config_;
    stream_state::Stamp config_stamp_, executable_stamp_;
    std::string config_digest_, reader_identity_;
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
        executable_stamp_ = stream_state::stamp(options.executable);
        reader_identity_ = stream_state::digest(file_io::path_text(options.executable) + '\n' +
            implementation_identity(options.executable, options.limits, until_) + '\n' + file_io::path_text(root_.path) + '\n' + config_digest_ + '\n' + options.rpc_host + '\n' + options.host);
        std::string bindings;
        for (auto const& [service, binding] : options.services)
        {
            require(service == "harvester" || service == "farmer" || service == "full_node" || service == "solver", "unsupported native service binding");
            require(!binding.host.empty() && !binding.rpc_host.empty() && binding.root.is_absolute(), "native service binding lacks its explicit RPC/resource host or local config root");
            bindings += service + '\n' + binding.node_id + '\n' + binding.farmer_node_id + '\n' + binding.host + '\n';
            if (service == "harvester")
            {
                require(binding.root == options.root && binding.rpc_host == parsed.rpc_hostname && binding.host == options.host, "harvester service binding differs from the selected reader context");
                continue;
            }
            auto child = options; child.services.clear(); child.root = binding.root; child.rpc_host = binding.rpc_host; child.host = binding.host;
            auto context = std::make_unique<Session>(child); bindings += context->reader_identity() + '\n';
            services_.emplace(service, std::move(context));
        }
        if (!bindings.empty()) reader_identity_ = stream_state::digest(reader_identity_ + '\n' + bindings);
        unchanged();
    }
    void running() const
    { inventory_detail::check(options_.limits, until_); }
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
        auto result = run_farm_adapter(command, limits);
        require(result.exit_code == 0, "native farm RPC failed (native output withheld)"); unchanged();
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
        require(success && profile_detail::boolean(*success), "native farm RPC did not report success"); return parsed;
    }
    std::string source() const
    {
        return std::string(name(options_.kind)) + ":" + file_io::path_text(options_.executable) + ":config-sha256=" + config_digest_;
    }
    std::string const& reader_identity() const { return reader_identity_; }
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
    require(inventory_detail::boolean(sync, "synced") && !inventory_detail::boolean(sync, "sync_mode"), "replacement requires a synced full node outside sync mode");
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
    require(options.max_peak_age_seconds && now >= 0 && timestamp <= uint64_t(now) && uint64_t(now) - timestamp <= options.max_peak_age_seconds,
        "replacement full-node peak has no sufficiently recent transaction timestamp");
    return timestamp;
}
inline uint64_t reviewed_target(Session& session, HarvesterAdapterOptions const& options, NetworkProjection const& reviewed,
    NetworkProjection const& observed, uint8_t k, uint8_t strength)
{
    require(reviewed.available && reviewed.reader_identity == observed.reader_identity && reviewed.reader_version == observed.reader_version &&
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
    require(result.has_value(), "target services lack a matched active peer connection; remote links require explicit reciprocal TLS node identities"); return *result;
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
        require(found, "target farmer has no synchronized receiver for the connected selected harvester");
        require(inventory_detail::required_field(*found, "syncing").kind == Json::null, "target farmer harvester synchronization is still in progress");
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
            require(profile_detail::boolean(inventory_detail::required_field(state, "started")), "target PoS2 solver has not completed startup");
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
            require(inventory_detail::required_field(row, "syncing").kind == Json::null,
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
            require(std::find(farmer.begin(), farmer.end(), link) != farmer.end(), "target service connection changed during the readiness snapshot");
        {
            auto summary = session_.rpc(AdapterOperation::farmer_harvesters_summary);
            auto const& selected = receiver(summary);
            require(service_time(selected, "last_sync_time", true) == generation_, "target farmer synchronization generation changed during the readiness snapshot");
            size_t observed = 0;
            for (auto const& row : inventory_detail::required_field(summary, "harvesters").items)
            {
                if (&row == &selected) continue;
                require(inventory_detail::required_field(row, "syncing").kind == Json::null,
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
            require(profile_detail::boolean(inventory_detail::required_field(state, "started")), "target PoS2 solver stopped during readiness observation");
        }
        pool_route();
        require(version(session_.rpc(AdapterOperation::farmer_version)) == farmer_version_ &&
            version(session_.rpc(AdapterOperation::solver_version)) == solver_version_, "target farmer or solver service version changed during readiness observation");
        session_.unchanged();
    }
};
} // namespace adapter_detail

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

inline NetworkProjection observe_network_projection(HarvesterAdapterOptions const& options)
{
    NetworkProjection result;
    try { adapter_detail::Session session(options); result = adapter_detail::projection(session); session.unchanged(); }
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
inline void require_plot_scope(Session& session, Json const& plots, FarmInventory const& inventory,
    HarvesterAdapterOptions const& options, PublishedPlotAllowance const& published = {})
{
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
    HarvesterAdapterOptions const& options, PublishedPlotAllowance const& published = {})
{ require_plot_scope(session, inventory_detail::required_field(response, "plots"), inventory, options, published); }
} // namespace adapter_detail
inline void observe_registry_scope(HarvesterAdapterOptions const& options, FarmInventory const& inventory,
    PublishedPlotAllowance const& published = {})
{
    adapter_detail::Session session(options); auto const version = adapter_detail::version(session.rpc(AdapterOperation::harvester_version));
    for (auto const& plot : inventory.plots) if (!plot.reader_identity.empty() &&
        (options.harvester.empty() || plot.harvester.empty() || options.harvester == plot.harvester))
        inventory_detail::require(plot.reader_identity == session.reader_identity() && plot.reader_version == version &&
            plot.reader == adapter_detail::name(options.kind), "registry scope reader implementation changed since review");
    auto response = session.rpc(AdapterOperation::harvester_plots);
    adapter_detail::require_registry_scope(session, response, inventory, options, published); session.unchanged();
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
        inventory_detail::require(current.identity == plot.identity && inventory_detail::integer(native, "file_size") == plot.identity.stamp.bytes &&
            inventory_detail::integer(native, "time_modified") == uint64_t(plot.identity.stamp.mtime), "native loaded plot differs from the observed file");
        auto const id = adapter_detail::hex_string(inventory_detail::required_field(native, "plot_id"), 32);
        inventory_detail::require(plot.plot_id.empty() || plot.plot_id == id, "native reader and public header disagree on plot identity");
        inventory_detail::require(plot.reader_identity.empty() || (plot.reader_identity == session.reader_identity() && plot.reader_version == version &&
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
        inventory_detail::require(plot.reader_identity.empty() || (plot.reader_identity == session.reader_identity() &&
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
inline void observe_harvester_inventory(FarmInventory& inventory, HarvesterAdapterOptions const& options, bool complete_registry = true)
{
    adapter_detail::Session session(options);
    auto const version = adapter_detail::version(session.rpc(AdapterOperation::harvester_version));
    auto response = session.rpc(AdapterOperation::harvester_plots);
    adapter_detail::observe_inventory(inventory, options, session, version, response, complete_registry);
}
inline void observe_harvester_inventory(FarmInventory& inventory, FarmAdapterTable const& table)
{
    std::set<std::string> selected;
    for (auto& plot : inventory.plots)
    { auto options = resolve_source(table, inventory.roots.at(plot.root), plot); plot.harvester = options.harvester; selected.insert(options.harvester); }
    for (auto const& route : table.routes) if (selected.contains(route.id)) observe_harvester_inventory(inventory, route.options);
    auto const target = resolve_target(table);
    if (!selected.contains(table.target_route)) observe_registry_scope(target, inventory);
    inventory.network = observe_network_projection(target);
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
    PublishedPlotAllowance published_;
    std::map<std::pair<std::string, std::string>, size_t> paths_;
    bool finished_ = false, allow_unloaded_ = false;
public:
    SourceReadinessSnapshot(HarvesterAdapterOptions const& options, FarmInventory const& inventory, std::span<uint64_t const> sources,
        PublishedPlotAllowance const& published = {}, bool allow_unloaded = false)
        : options_(options), session_(options), inventory_(inventory), published_(published), allow_unloaded_(allow_unloaded)
    {
        inventory_detail::require(!sources.empty() && sources.size() <= options.limits.max_files &&
            sources.size() <= options.limits.memory_bytes / 32768, "source snapshot exceeds its retained metadata grant");
        current_.host = inventory.host; current_.captured_at = inventory_detail::timestamp(); current_.producer_build = GROW_IDENTITY;
        current_.roots = inventory.roots; current_.complete = true;
        for (auto index : sources)
        {
            inventory_detail::require(index < inventory.plots.size(), "source snapshot index is invalid");
            auto plot = inventory.plots[size_t(index)];
            inventory_detail::require(plot.root < inventory.roots.size() && plot.identity.links == 1 && !plot.alias_of &&
                (options.harvester.empty() || plot.harvester.empty() || options.harvester == plot.harvester),
                "source snapshot crosses a reviewed route or lacks unique native identity");
            auto const key = std::pair(current_.roots[plot.root].path, plot.relative_path);
            inventory_detail::require(paths_.emplace(key, current_.plots.size()).second, "source snapshot repeats a path");
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
        auto response = session_.rpc(AdapterOperation::harvester_plots);
        adapter_detail::require_registry_scope(session_, response, inventory, options_, published);
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
            inventory_detail::require(reviewed.reader_identity.empty() || (reviewed.reader_identity == session_.reader_identity() &&
                reviewed.reader_version == version_ && reviewed.reader == adapter_detail::name(options_.kind)),
                "previously observed source reader/configuration changed since review");
            auto observed = observe_inventory_file(root, reviewed.relative_path, session_.remaining_limits());
            inventory_detail::require(observed.identity == reviewed.identity && observed.header_validated && observed.algorithm == reviewed.algorithm &&
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
                reviewed.reader_identity == session_.reader_identity() && reviewed.reader_version == version_ && reviewed.reader == adapter_detail::name(options_.kind),
                "unloaded source lacks the exact previously reviewed reader metadata");
            (void)recheck_inventory_source(root, reviewed, session_.remaining_limits());
            auto observed = reviewed;
            observed.reason = "exact reviewed native object is currently unloaded; public identity retained from the authorized review, old proof validity untested";
            return observed;
        }
        inventory_detail::require(current.reader_identity == session_.reader_identity() && current.reader_version == version_ &&
            current.reader == adapter_detail::name(options_.kind) && current.identity == reviewed.identity &&
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
        adapter_detail::require_registry_scope(session_, response, inventory_, options_, published_);
        inventory_detail::require(adapter_detail::registry_identity(response) == registry_identity_,
            "source loaded population or reader metadata changed during the batch snapshot");
        session_.unchanged(); finished_ = true;
    }
};
inline SourceRetirementObservation observe_sources_retired(HarvesterAdapterOptions const& options, FarmInventory const& inventory,
    std::span<uint64_t const> sources, PublishedPlotAllowance const& published = {})
{
    SourceRetirementObservation result; result.captured_at = inventory_detail::timestamp();
    try
    {
        inventory_detail::require(!sources.empty() && sources.size() <= options.limits.max_files && sources.size() <= options.limits.memory_bytes / 32768,
            "source registry batch exceeds its bounded lookup memory");
        std::set<std::string> paths, ids;
        for (auto index : sources)
        {
            inventory_detail::require(index < inventory.plots.size(), "source registry batch index is invalid"); auto const& p = inventory.plots[size_t(index)];
            inventory_detail::require(p.root < inventory.roots.size() && !p.plot_id.empty() &&
                (options.harvester.empty() || p.harvester.empty() || options.harvester == p.harvester), "source registry batch crosses a reviewed route or lacks plot identity");
            paths.insert(file_io::path_text(file_io::native_path(inventory.roots[p.root].path) / inventory_detail::relative(p.relative_path))); ids.insert(p.plot_id);
        }
        adapter_detail::Session session(options); result.source = session.source();
        auto version = adapter_detail::version(session.rpc(AdapterOperation::harvester_version));
        for (auto index : sources)
        {
            auto const& p = inventory.plots[size_t(index)];
            bool const header_only = p.reader_identity.empty() && p.header_validated && p.algorithm == PlotAlgorithm::pos1 &&
                (p.container == PlotContainer::chiapos_v1 || p.container == PlotContainer::chiapos_v2);
            inventory_detail::require(header_only || (p.reader_identity == session.reader_identity() && p.reader_version == version &&
                p.reader == adapter_detail::name(options.kind)), "source registry reader implementation changed or was not reviewed; retain old reservations");
        }
        auto response = session.rpc(AdapterOperation::harvester_plots); auto const& plots = inventory_detail::required_field(response, "plots");
        inventory_detail::require(plots.kind == profile_detail::Json::array && plots.items.size() <= options.limits.max_files,
            "native source registry snapshot exceeds its bound");
        adapter_detail::require_registry_scope(session, response, inventory, options, published);
        for (auto const& p : plots.items)
            if (paths.contains(inventory_detail::string(p, "filename")) ||
                ids.contains(adapter_detail::hex_string(inventory_detail::required_field(p, "plot_id"), 32)))
            { result.reason = "a batch source path or plot ID is still registered; old arrivals remain reserved"; return result; }
        inventory_detail::require(adapter_detail::version(session.rpc(AdapterOperation::harvester_version)) == version,
            "source registry reader version changed during absence observation; retain old reservations");
        session.unchanged(); result.observed = std::chrono::steady_clock::now(); result.no_loaded_record = true;
        result.reason = "one fresh native registry snapshot has no batch source paths/IDs; queued/resident/opaque reservations remain separate";
    }
    catch (std::exception const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.no_loaded_record = false; result.reason = error.what(); }
    return result;
}
inline SourceRetirementObservation observe_source_retired(HarvesterAdapterOptions const& options, InventoryRoot const& root, InventoryPlot const& plot)
{
    FarmInventory inventory; inventory.roots.push_back(root); inventory.plots.push_back(plot); inventory.plots[0].root = 0;
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
        inventory_detail::require(now >= 0 && peak_timestamp_ <= uint64_t(now) && uint64_t(now) - peak_timestamp_ <= options_.max_peak_age_seconds,
            "target readiness snapshot's transaction peak became stale");
    }
public:
    TargetReadinessSnapshot(HarvesterAdapterOptions const& options, NetworkProjection const& reviewed, uint8_t k, uint8_t strength,
        TargetServiceExpectation const& service, FarmInventory const* inventory = nullptr, PublishedPlotAllowance const& published = {})
        : options_(options), session_(options_), projection_(adapter_detail::projection(session_)), service_(service), k_(k), strength_(strength)
    {
        projection_.target_route = reviewed.target_route;
        peak_timestamp_ = adapter_detail::reviewed_target(session_, options_, reviewed, projection_, k, strength);
        services_ = std::make_unique<adapter_detail::TargetServices>(session_, options_, reviewed, projection_, service_);
        response_ = session_.rpc(AdapterOperation::harvester_plots);
        if (inventory) adapter_detail::require_registry_scope(session_, response_, *inventory, options_, published);
        if (inventory) adapter_detail::require_plot_scope(session_, services_->plots(), *inventory, options_, published);
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
                expected.genesis_challenge == projection_.genesis_challenge && expected.reader_identity == projection_.reader_identity &&
                expected.reader_version == projection_.reader_version && expected.network_version == projection_.version && expected.target_route == projection_.target_route &&
                expected.farmer_version == projection_.farmer_version && expected.solver_version == projection_.solver_version &&
                expected.farmer_public_key == service_.farmer_public_key && expected.pool_public_key == service_.pool_public_key &&
                expected.pool_contract_puzzle_hash == service_.pool_contract_puzzle_hash,
                "replacement differs from the reviewed target snapshot parameters/provenance");
            inventory_detail::RootHandle root(expected.path.parent_path());
            auto file = inventory_detail::observe(root, file_io::path_text(expected.path.filename()), options_.limits);
            inventory_detail::require(file.identity == expected.identity && file.header_validated && file.algorithm == PlotAlgorithm::pos2 &&
                file.plot_id == expected.group_id && file.k == expected.k && file.strength == expected.strength && file.meta_group == expected.meta_group && file.members == expected.members &&
                file.farmer_public_key == expected.farmer_public_key && file.pool_public_key == expected.pool_public_key && file.pool_contract_puzzle_hash == expected.pool_contract_puzzle_hash,
                "replacement file/header/public keys differ from the reviewed exact output");
            auto const at = loaded_.find(file_io::path_text(expected.path));
            auto const farmer_at = farmer_loaded_.find(file_io::path_text(expected.path));
            inventory_detail::require(at != loaded_.end(), "exact replacement is not loaded by the configured harvester");
            inventory_detail::require(farmer_at != farmer_loaded_.end(), "exact replacement has not reached the connected farmer's completed plot synchronization");
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
            inventory_detail::require(after.identity == expected.identity, "published replacement changed during live readiness observation");
            at->second.checked = expected.identity.stamp;
            current(); result.identity = file.identity; result.ready = true; result.status = "ready";
            result.reason = "new immutable publication matches the live harvester and connected farmer's completed sync, key route and solver prerequisites; finish the complete audit before retirement";
        }
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
    catch (std::exception const& error)
    { if (options.limits.cancelled && options.limits.cancelled()) throw; result.ready = false; result.status = "unavailable"; result.reason = error.what(); }
    return result;
}
} // namespace group_grow::farm
