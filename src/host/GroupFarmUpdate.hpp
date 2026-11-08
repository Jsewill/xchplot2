// A durable, keyless request for the coordinator's next recovery boundary.
#pragma once
#include "host/GroupFarmState.hpp"
#include "host/GroupFarmMetadata.hpp"
#include "host/GroupFarmControlState.hpp"
#include "host/GroupStorage.hpp"
#include <map>
#include <set>
#include <type_traits>

namespace group_grow
{
struct FarmingUpdate
{
    unsigned version = 3;
    std::string id, job_id, source_policy;
    uint64_t source_revision = 0, generation = 0;
    FarmingConfig config;
    std::optional<FarmingProfileMetadata> profile;
    uint64_t capacity = 0, headroom = 0, max_groups = UINT64_MAX, max_total_members = UINT64_MAX;
    std::vector<std::filesystem::path> directories;
    std::vector<std::string> directory_identities;
    std::string action;
    std::vector<std::string> action_destinations;
    std::map<std::string, std::string> expected;
    std::vector<std::string> supersedes;
};
inline std::string farming_policy_digest(FarmingState state)
{
    // Progress revisions must not invalidate an otherwise current request.
    // Keys, topology, limits, forecasts and cohort identity assignments remain
    // bound; only durable progress and the request receipt are omitted.
    state.revision = 1; state.parent_digest.clear(); state.applied_update.clear(); state.request_floor = 0; state.ancestors.clear();
    for (auto& d : state.directories) { d.extents.clear(); d.sealed_files = 0; d.fill_stop.clear(); }
    return stream_state::digest(farming_state_text(state));
}
inline std::string farming_update_text(FarmingUpdate const& update)
{
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<long double>::max_digits10)
        << "ordinary-group-farming-update-v" << update.version << '\n' << std::quoted(update.id) << ' ' << std::quoted(update.job_id) << ' '
        << std::quoted(update.source_policy) << '\n';
    farming_config_write(out, update.config);
    out << update.profile.has_value() << '\n';
    if (update.profile) out << farming_profile_metadata_text(*update.profile, 7);
    out << update.capacity << ' ' << update.headroom << ' ' << update.max_groups << ' ' << update.max_total_members << '\n'
        << update.directories.size() << '\n';
    require(update.directory_identities.size() == update.directories.size(), "farming update lacks destination identities");
    for (size_t i = 0; i < update.directories.size(); ++i)
        out << std::quoted(file_io::path_text(update.directories[i])) << ' ' << std::quoted(update.directory_identities[i]) << '\n';
    if (update.version >= 2)
    {
        out << std::quoted(update.action) << ' ' << update.action_destinations.size() << '\n';
        for (auto const& identity : update.action_destinations) out << std::quoted(identity) << '\n';
    }
    if (update.version >= 3)
    {
        out << update.source_revision << ' ' << update.generation << ' ' << update.expected.size() << '\n';
        for (auto const& [field, value] : update.expected) out << std::quoted(field) << ' ' << std::quoted(value) << '\n';
        out << update.supersedes.size() << '\n';
        for (auto const& id : update.supersedes) out << std::quoted(id) << '\n';
    }
    return out.str();
}
inline bool farming_update_digest(std::string const& value)
{ return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == value.npos; }
inline std::string farming_update_id(FarmingUpdate update)
{ update.id.clear(); return stream_state::digest(farming_update_text(update)); }
template<class Visitor> inline void farming_update_fields_visit(FarmingUpdate& target,
    FarmingUpdate const& source, Visitor&& visit)
{
#define GROW_UPDATE_FIELD(name, member) visit(name, target.member, source.member)
    GROW_UPDATE_FIELD("k", config.k); GROW_UPDATE_FIELD("strength", config.strength);
    GROW_UPDATE_FIELD("farming-filter-bits", config.filter_bits);
    GROW_UPDATE_FIELD("farming-future-filter-bits", config.future_filter_bits);
    GROW_UPDATE_FIELD("farming-reader", config.reader); GROW_UPDATE_FIELD("farming-objective", config.objective);
    GROW_UPDATE_FIELD("farming-cores", config.harvester_cores); GROW_UPDATE_FIELD("farming-lookup-workers", config.lookup_workers);
    GROW_UPDATE_FIELD("farming-index-cache-bytes", config.index_cache_bytes); GROW_UPDATE_FIELD("farming-memory-bytes", config.memory_bytes);
    GROW_UPDATE_FIELD("farming-max-eligible-groups", config.max_eligible_families_per_resource);
    GROW_UPDATE_FIELD("farming-cpu-work-weight", config.cpu_work_weight); GROW_UPDATE_FIELD("farming-io-work-weight", config.io_work_weight);
    GROW_UPDATE_FIELD("farming-deadline-ms", config.deadline_seconds); GROW_UPDATE_FIELD("farming-challenge-period", config.challenge_seconds);
    GROW_UPDATE_FIELD("group-members", config.group_members_override); GROW_UPDATE_FIELD("meta-slots", config.meta_slots_override);
    GROW_UPDATE_FIELD("meta-group", config.meta_group_override); GROW_UPDATE_FIELD("fixed-meta", config.fixed_meta);
    GROW_UPDATE_FIELD("farming-chain-us-per-member", config.chain_seconds_per_member); GROW_UPDATE_FIELD("farming-index-ns-per-entry", config.index_seconds_per_entry);
    GROW_UPDATE_FIELD("farming-index-entry-bytes", config.compressed_index_bytes_per_entry);
    GROW_UPDATE_FIELD("farming-lookup-member-bytes", config.lookup_memory_per_member);
    GROW_UPDATE_FIELD("capacity-bytes", capacity); GROW_UPDATE_FIELD("leave-free", headroom);
    GROW_UPDATE_FIELD("max-groups", max_groups); GROW_UPDATE_FIELD("max-total-members", max_total_members);
    GROW_UPDATE_FIELD("harvester-profile", profile);
#undef GROW_UPDATE_FIELD
}
template<class Value> inline std::string farming_update_field_text(Value const& value)
{
    std::ostringstream out; out << std::setprecision(std::numeric_limits<long double>::max_digits10);
    if constexpr (std::is_same_v<Value, std::optional<FarmingProfileMetadata>>)
    { out << value.has_value() << '\n'; if (value) out << farming_profile_metadata_text(*value, 7); }
    else if constexpr (std::is_enum_v<Value> || std::is_same_v<Value, uint8_t>) out << unsigned(value);
    else out << value;
    return out.str();
}
inline std::string farming_update_control(std::string_view action)
{ return action == "resume" ? "active" : action == "pause" ? "paused" : action == "drain" ? "draining" : "retired"; }
inline std::map<std::string, std::string> farming_update_fields(FarmingUpdate const& update)
{
    std::map<std::string, std::string> result;
    auto target = update;
    farming_update_fields_visit(target, update, [&](auto const* name, auto const&, auto const& value)
    { result.emplace(name, farming_update_field_text(value)); });
    require(update.directories.size() == update.directory_identities.size(), "farming update lacks destination identities");
    for (size_t i = 0; i < update.directories.size(); ++i)
        result.emplace("destination:" + update.directory_identities[i], file_io::path_text(update.directories[i]));
    if (!update.action.empty())
    {
        auto const control = farming_update_control(update.action);
        if (update.action_destinations.empty()) result.emplace("control", control);
        else for (auto const& identity : update.action_destinations) result.emplace("control:" + identity, control);
    }
    return result;
}
inline FarmingUpdate farming_update_snapshot(FarmingState const& state)
{
    FarmingUpdate result;
    result.job_id = state.job_id; result.source_policy = farming_policy_digest(state); result.source_revision = state.policy_revision;
    result.config = state.config; result.profile = state.profile_metadata;
    result.capacity = state.capacity; result.headroom = state.headroom;
    result.max_groups = state.max_groups; result.max_total_members = state.max_total_members;
    for (auto const& directory : state.directories)
    {
        result.directories.push_back(file_io::native_path(directory.directory));
        result.directory_identities.push_back(directory.directory_identity);
    }
    return result;
}
inline std::map<std::string, std::string> farming_update_state_fields(FarmingState const& state)
{
    auto result = farming_update_fields(farming_update_snapshot(state));
    result.emplace("control", state.control);
    for (auto const& directory : state.directories) result.emplace("control:" + directory.directory_identity, directory.control);
    return result;
}
inline bool farming_update_control_only(FarmingUpdate const& update)
{
    return update.version >= 3 && !update.action.empty() && !update.expected.empty() &&
        std::all_of(update.expected.begin(), update.expected.end(), [](auto const& field)
        { return field.first == "control" || field.first.starts_with("control:"); });
}
inline void farming_update_preconditions(FarmingUpdate& update, FarmingState const& state,
    std::set<std::string> const& explicit_fields = {})
{
    auto before = farming_update_state_fields(state), after = farming_update_fields(update);
    update.source_policy = farming_policy_digest(state); update.source_revision = state.policy_revision;
    update.expected.clear();
    for (auto const& [field, desired] : after)
        if (before[field] != desired || explicit_fields.contains(field)) update.expected.emplace(field, before[field]);
}
inline constexpr std::string_view farming_update_choices_prefix = "ordinary-group-farming-field-values-v1\n";
inline std::set<std::string> farming_update_expected_values(std::string const& expected)
{
    if (!expected.starts_with(farming_update_choices_prefix)) return {expected};
    std::istringstream in(expected.substr(farming_update_choices_prefix.size()));
    size_t count = 0; in >> count;
    require(in && count && count <= 16, "invalid farming update field alternatives");
    std::set<std::string> values;
    for (size_t i = 0; i < count; ++i)
    {
        std::string value; in >> std::quoted(value);
        require(in && !value.starts_with(farming_update_choices_prefix) && values.insert(std::move(value)).second,
            "invalid farming update field alternative");
    }
    in >> std::ws; require(in.eof(), "unexpected farming update field alternatives");
    return values;
}
inline std::string farming_update_expected_union(std::string const& before, std::string const& after)
{
    auto values = farming_update_expected_values(before);
    auto const incoming = farming_update_expected_values(after); values.insert(incoming.begin(), incoming.end());
    if (values.size() == 1) return *values.begin();
    require(values.size() <= 16, "too many competing values for one farming update field; resolve the waiting requests first");
    std::ostringstream out; out << farming_update_choices_prefix << values.size() << '\n';
    for (auto const& value : values) out << std::quoted(value) << '\n';
    return out.str();
}
inline FarmingUpdate farming_update_rebase(FarmingUpdate const& request, FarmingState const& current)
{
    require(request.job_id == current.job_id, "queued update belongs to a different farming job");
    if (request.version < 3)
    {
        require(request.source_policy == farming_policy_digest(current),
            "legacy queued update is stale; submit a fresh fill request");
        return request;
    }
    require(request.source_revision <= current.policy_revision,
        "queued update requires a newer policy revision; recover its saved members before continuing");
    require(request.source_revision != current.policy_revision || request.source_policy == farming_policy_digest(current),
        "queued update source differs at the same policy revision; refresh status and submit a fresh request");
    auto before = farming_update_state_fields(current), desired = farming_update_fields(request);
    for (auto const& [field, expected] : request.expected)
    {
        auto const value = desired.find(field);
        require(value != desired.end(), "queued update has an unknown field: " + field);
        require(farming_update_expected_values(expected).contains(before[field]) || before[field] == value->second,
            "queued update conflicts with changed field " + field + "; inspect status and submit a fresh request");
    }
    auto result = farming_update_snapshot(current);
    result.id = request.id; result.generation = request.generation; result.expected = request.expected; result.supersedes = request.supersedes;
    farming_update_fields_visit(result, request, [&](auto const* name, auto& target, auto const& value)
    { if (request.expected.contains(name)) target = value; });
    std::set<std::filesystem::path> paths(result.directories.begin(), result.directories.end());
    for (size_t i = 0; i < request.directories.size(); ++i)
    {
        if (!request.expected.contains("destination:" + request.directory_identities[i]) ||
            std::find(result.directory_identities.begin(), result.directory_identities.end(), request.directory_identities[i]) != result.directory_identities.end()) continue;
        require(paths.insert(request.directories[i]).second, "queued update conflicts with destination path " + file_io::path_text(request.directories[i]));
        result.directories.push_back(request.directories[i]); result.directory_identities.push_back(request.directory_identities[i]);
    }
    result.action = request.action; result.action_destinations = request.action_destinations;
    return result;
}
struct FarmingUpdateFile
{
    FarmingUpdate update;
    stream_state::Stamp stamp;
};
inline FarmingUpdate parse_farming_update(std::string const& body)
{
    std::istringstream in(body); std::string magic; std::getline(in, magic);
    require(magic == "ordinary-group-farming-update-v1" || magic == "ordinary-group-farming-update-v2" ||
        magic == "ordinary-group-farming-update-v3",
        "unsupported farming update request");
    FarmingUpdate result;
    result.version = magic == "ordinary-group-farming-update-v1" ? 1 : magic == "ordinary-group-farming-update-v2" ? 2 : 3;
    in >> std::quoted(result.id) >> std::quoted(result.job_id) >> std::quoted(result.source_policy);
    require(in && farming_update_digest(result.id) && farming_update_digest(result.job_id) && farming_update_digest(result.source_policy),
        "invalid farming update identity");
    result.config = farming_config_read(in); farming_detail::validate(result.config, {});
    bool profile = false; in >> profile;
    if (profile)
    {
        result.profile = farming_profile_metadata_read(in, 7);
        validate_reader_calibration_scope(*result.profile, result.config);
    }
    size_t count = 0;
    in >> result.capacity >> result.headroom >> result.max_groups >> result.max_total_members >> count;
    require(in && count && count <= 1024 && result.max_groups && result.max_total_members, "invalid farming update limits");
    for (size_t i = 0; i < count; ++i)
    {
        std::string text, identity; in >> std::quoted(text) >> std::quoted(identity);
        auto directory = file_io::native_path(text);
        require(in && directory.is_absolute() && !identity.empty() && identity.size() <= 256, "invalid farming update destination");
        require(std::find(result.directories.begin(), result.directories.end(), directory) == result.directories.end() &&
            std::find(result.directory_identities.begin(), result.directory_identities.end(), identity) == result.directory_identities.end(),
            "duplicate farming update destination");
        result.directories.push_back(std::move(directory));
        result.directory_identities.push_back(std::move(identity));
    }
    if (result.version >= 2)
    {
        size_t targets = 0;
        in >> std::quoted(result.action) >> targets;
        require(in && targets <= count && (result.action.empty() || result.action == "pause" || result.action == "drain" ||
            result.action == "resume" || result.action == "retire") && (!targets || !result.action.empty()),
            "invalid farming control request");
        for (size_t i = 0; i < targets; ++i)
        {
            std::string identity; in >> std::quoted(identity);
            require(in && std::find(result.directory_identities.begin(), result.directory_identities.end(), identity) != result.directory_identities.end() &&
                std::find(result.action_destinations.begin(), result.action_destinations.end(), identity) == result.action_destinations.end(),
                "farming control target is not a distinct destination of this job");
            result.action_destinations.push_back(std::move(identity));
        }
    }
    if (result.version >= 3)
    {
        size_t fields = 0;
        in >> result.source_revision >> result.generation >> fields;
        require(in && result.generation && fields <= 64 + count, "invalid farming update preconditions");
        auto const desired = farming_update_fields(result);
        for (size_t i = 0; i < fields; ++i)
        {
            std::string field, value; in >> std::quoted(field) >> std::quoted(value);
            require(in && field.size() <= 512 && value.size() <= (farming_profile_metadata_bytes + std::string_view("1\n").size()) * 16 && desired.contains(field),
                "invalid farming update field precondition");
            (void)farming_update_expected_values(value);
            require(result.expected.emplace(std::move(field), std::move(value)).second, "duplicate farming update field precondition");
        }
        size_t replaced = 0; in >> replaced;
        require(in && replaced <= 128, "too many superseded farming updates");
        for (size_t i = 0; i < replaced; ++i)
        {
            std::string id; in >> std::quoted(id);
            require(in && farming_update_digest(id) && id != result.id &&
                (result.supersedes.empty() || result.supersedes.back() < id), "invalid superseded farming update identity");
            result.supersedes.push_back(std::move(id));
        }
        require(result.id == farming_update_id(result), "farming update content identity mismatch");
    }
    in >> std::ws; require(in.eof(), "unexpected farming update data");
    return result;
}
inline FarmingUpdateFile load_farming_update(std::filesystem::path const& path, uint64_t memory_limit)
{
    auto const file = read_farming_state(path, memory_limit);
    auto update = parse_farming_update(file.body);
    require(stream_state::stamp(path) == file.stamp, "farming update changed while loading");
    return {std::move(update), file.stamp};
}

inline constexpr size_t farming_update_slots = 4, farming_update_history = 128;
inline constexpr size_t farming_update_receipt_bytes = 4096, farming_update_reason_bytes = 1024;
inline uint64_t farming_update_aligned(uint64_t bytes, uint64_t alignment)
{
    require(alignment, "invalid farming update allocation alignment");
    auto const remainder = bytes % alignment;
    return remainder ? grow_add(bytes, alignment - remainder) : bytes;
}
inline uint64_t farming_update_reservation(uint64_t request_bytes, uint64_t alignment,
    size_t slots = farming_update_slots)
{
    // Every unresolved request retains its acceptance and one terminal result.
    // Floor publication and one private atomic replacement also need space.
    auto const request = farming_update_aligned(request_bytes, alignment);
    auto const receipt = farming_update_aligned(farming_update_receipt_bytes, alignment);
    auto const per_request = grow_add(request, grow_add(receipt, receipt));
    require(slots <= UINT64_MAX / per_request, "farming update reservation overflow");
    return grow_add(slots * per_request, grow_add(receipt, std::max(request, receipt)));
}
inline uint64_t farming_control_reservation(FarmingState const& state, uint64_t alignment)
{
    auto const record = farming_update_aligned(grow_add(farming_control_size_bound(state), 65), alignment);
    require(record <= UINT64_MAX / 2, "farming control reservation overflow"); return record * 2;
}
struct FarmingUpdateSpaceError : std::runtime_error
{
    uint64_t const required, available;
    FarmingUpdateSpaceError(uint64_t needed, uint64_t remaining) :
        std::runtime_error("farming update needs " + std::to_string(needed) + " metadata bytes; " +
            std::to_string(remaining) + " reserved bytes remain; retry after the coordinator reaches a recovery boundary"),
        required(needed), available(remaining) {}
};
inline uint64_t farming_update_allocation(std::filesystem::path const& metadata, bool controls_only = false)
{
    file_io::Stat directory{};
    if (file_io::inspect_link(metadata, &directory))
    { require(errno == ENOENT, "cannot inspect farming update directory"); return 0; }
    require(file_io::is_directory(directory.st_mode), "invalid farming update directory");
    uint64_t bytes = 0; size_t files = 0;
    for (auto const& entry : std::filesystem::directory_iterator(metadata))
    {
        auto const name = file_io::path_text(entry.path().filename());
        if (!name.starts_with("grow-farm.update")) continue;
        if (controls_only && name != "grow-farm.update.controls" && name != "grow-farm.update.controls.next" &&
            !name.starts_with("grow-farm.update.controls.partial.") && !name.starts_with("grow-farm.update.controls.next.partial.")) continue;
        require(++files <= farming_update_history * 4 + 16, "farming update metadata exceeds its file bound");
        file_io::Stat info{};
        int const inspected = file_io::inspect_link(entry.path(), &info);
        if (inspected && errno == ENOENT && (name == "grow-farm.update.controls.next" || name.find(".partial.") != name.npos)) continue;
        bool private_links = info.st_nlink == 1;
        auto const split = name.find(".partial.");
        if (!inspected && info.st_nlink == 2 && split != name.npos)
        {
            file_io::Stat published{};
            private_links = !file_io::inspect_link(metadata / name.substr(0, split), &published) &&
                file_io::is_regular(published.st_mode) && published.st_dev == info.st_dev && published.st_ino == info.st_ino;
        }
        require(!inspected && file_io::is_regular(info.st_mode) && private_links &&
            info.st_size >= 0 && info.st_blocks >= 0 && uint64_t(info.st_blocks) <= UINT64_MAX / 512,
            "invalid farming update allocation");
        bytes = grow_add(bytes, std::max(uint64_t(info.st_size), uint64_t(info.st_blocks) * 512));
    }
    return bytes;
}
struct FarmingUpdateReceipt
{
    std::string id, job_id, request_digest, status, reason, successor;
    uint64_t generation = 0;
};
inline bool farming_update_terminal(std::string_view status)
{ return status == "refused" || status == "applied" || status == "cancelled" || status == "superseded"; }
inline std::string farming_update_receipt_text(FarmingUpdateReceipt const& receipt)
{
    require(farming_update_digest(receipt.id) && farming_update_digest(receipt.job_id) && farming_update_digest(receipt.request_digest) &&
        (receipt.status == "accepted" || farming_update_terminal(receipt.status)) && receipt.reason.size() <= farming_update_reason_bytes &&
        (receipt.status == "superseded" ? farming_update_digest(receipt.successor) : receipt.successor.empty()),
        "invalid farming update receipt");
    std::ostringstream out;
    out << "ordinary-group-farming-update-receipt-v1\n" << std::quoted(receipt.id) << ' ' << std::quoted(receipt.job_id) << ' '
        << receipt.generation << ' ' << std::quoted(receipt.request_digest) << '\n'
        << std::quoted(receipt.status) << ' ' << std::quoted(receipt.reason) << ' ' << std::quoted(receipt.successor) << '\n';
    return out.str();
}
inline FarmingUpdateReceipt parse_farming_update_receipt(std::string const& body)
{
    require(body.size() <= farming_update_receipt_bytes, "farming update receipt exceeds its bound");
    std::istringstream in(body); std::string magic; std::getline(in, magic);
    FarmingUpdateReceipt receipt;
    in >> std::quoted(receipt.id) >> std::quoted(receipt.job_id) >> receipt.generation >> std::quoted(receipt.request_digest)
        >> std::quoted(receipt.status) >> std::quoted(receipt.reason) >> std::quoted(receipt.successor);
    require(in && magic == "ordinary-group-farming-update-receipt-v1", "unsupported farming update receipt");
    in >> std::ws; require(in.eof(), "unexpected farming update receipt data");
    (void)farming_update_receipt_text(receipt); return receipt;
}
struct FarmingUpdateRecord
{
    FarmingUpdateFile file;
    std::optional<FarmingUpdateReceipt> receipt;
    std::string body, digest;
};

// Request locks cover every available native member in one stable order. A
// coordinator may apply next() only while it additionally owns all members'
// construction locks. A disconnected subset can accept intent, never policy.
class FarmingUpdateMailbox
{
    struct Replica
    {
        std::unique_ptr<FarmingMetadataDirectory> metadata;
        int lock = -1;
        ~Replica() { if (lock >= 0) file_io::close_file(lock); }
    };
    FarmingState state_;
    uint64_t memory_;
    bool acquired_ = false;
    bool terminal_ = false;
    std::vector<std::unique_ptr<Replica>> replicas_;
    std::vector<FarmingControlNativeBinding> control_bindings_;
    mutable uint64_t floor_ = 0;
    mutable bool legacy_floor_ = false;
    static std::string request_name(std::string const& id) { return "grow-farm.update-" + id + ".request"; }
    static std::string receipt_name(std::string const& id, std::string const& status)
    { return "grow-farm.update-" + id + '.' + status; }
    static bool record_name(std::string const& name)
    {
        if (name == "grow-farm.update" || name == "grow-farm.update.error" || name == "grow-farm.update.floor" ||
            name == "grow-farm.update.controls" || name == "grow-farm.update.controls.next") return true;
        constexpr std::string_view prefix = "grow-farm.update-";
        if (!name.starts_with(prefix) || name.size() <= prefix.size() + 64) return false;
        auto const id = name.substr(prefix.size(), 64), suffix = name.substr(prefix.size() + 64);
        return farming_update_digest(id) && (suffix == ".request" || suffix == ".accepted" || suffix == ".refused" ||
            suffix == ".applied" || suffix == ".cancelled" || suffix == ".superseded");
    }
    static std::string checked(std::string body) { body += stream_state::digest(body) + '\n'; return body; }
    void check() const { for (auto const& replica : replicas_) replica->metadata->check(); }
    void require_writer() const { require(locked(), "farming update operation requires every available request lock"); check(); }
    void cleanup_partials()
    {
        require_writer();
        for (auto const& replica : replicas_)
        {
            for (auto const& entry : std::filesystem::directory_iterator(replica->metadata->path))
            {
                auto const name = file_io::path_text(entry.path().filename());
                constexpr std::string_view partial = ".partial.";
                auto const split = name.find(partial);
                if (split == name.npos || !record_name(name.substr(0, split))) continue;
                auto const nonce = std::string_view(name).substr(split + partial.size());
                if (nonce.size() != std::string_view("XXXXXX").size() || std::any_of(nonce.begin(), nonce.end(), [](unsigned char c)
                    { return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')); })) continue;
                file_io::Stat temporary{};
                require(!file_io::inspect_link(entry.path(), &temporary) && file_io::is_regular(temporary.st_mode) &&
                    (temporary.st_nlink == 1 || temporary.st_nlink == 2), "invalid interrupted farming update publication");
                if (temporary.st_nlink == 2)
                {
                    file_io::Stat published{};
                    require(!file_io::inspect_link(replica->metadata->path / file_io::native_path(name.substr(0, split)), &published) &&
                        file_io::is_regular(published.st_mode) && published.st_dev == temporary.st_dev && published.st_ino == temporary.st_ino,
                        "interrupted farming update has an unrelated hard link");
                }
                require(!file_io::unlink_file(entry.path()), "cannot clean interrupted farming update publication");
            }
            stream_state::sync_path(replica->metadata->path, true); replica->metadata->check();
        }
    }
    void read_floor() const
    {
        floor_ = state_.request_floor;
        for (auto const& replica : replicas_)
        {
            auto const path = replica->metadata->path / "grow-farm.update.floor";
            if (!farm_metadata_detail::present(path)) continue;
            auto const saved = read_farming_state(path, memory_);
            std::istringstream in(saved.body); std::string magic, job; uint64_t floor = 0; bool legacy = false;
            std::getline(in, magic); in >> std::quoted(job) >> floor >> legacy;
            require(in && magic == "ordinary-group-farming-update-floor-v1" && job == state_.job_id,
                "invalid farming update history floor");
            in >> std::ws; require(in.eof(), "unexpected farming update floor data");
            floor_ = std::max(floor_, floor); legacy_floor_ = legacy_floor_ || legacy;
        }
    }
    uint64_t alignment(Replica const& replica) const
    {
        file_io::Space space{}; require(!file_io::space_path(replica.metadata->root, &space) && space.f_frsize,
            "cannot inspect farming update allocation units");
        return space.f_frsize;
    }
    void admit(Replica const& replica, uint64_t bytes, bool control = false) const
    {
        uint64_t const updates = farming_update_allocation(replica.metadata->path);
        uint64_t used = grow_add(replica.metadata->allocation(), updates);
        uint64_t reserved = state_.manifest_reserved_bytes;
        uint64_t update_reserved = state_.update_reserved_bytes;
        uint64_t control_reserved = state_.version >= 9 ? farming_control_reservation(state_, alignment(replica)) : 0;
        bool control_separated = state_.version >= 9;
        uint64_t progress_reserved = state_.progress_reserved_bytes, owner_reserved = state_.owner_reserved_bytes, largest_plan = 0;
        size_t plans = 0;
        bool separated = state_.version >= 8;
        for (auto const* name : {"grow-farm.plan", "grow-farm.plan.next"})
        {
            auto const path = replica.metadata->path / name;
            if (!farm_metadata_detail::present(path)) continue;
            file_io::Stat info{};
            require(!file_io::inspect_link(path, &info) && file_io::is_regular(info.st_mode) && info.st_nlink == 1 &&
                info.st_size >= 0 && info.st_blocks >= 0 && uint64_t(info.st_blocks) <= UINT64_MAX / 512,
                "invalid farming update source allocation");
            auto const allocation = std::max(uint64_t(info.st_size), uint64_t(info.st_blocks) * 512);
            used = grow_add(used, allocation); largest_plan = std::max(largest_plan, allocation); ++plans;
            auto const source = load_farming_state(path, memory_);
            require(source.job_id == state_.job_id, "farming update source job changed");
            reserved = std::max(reserved, source.manifest_reserved_bytes);
            if (source.version >= 8)
            {
                separated = true; update_reserved = std::max(update_reserved, source.update_reserved_bytes);
                progress_reserved = std::max(progress_reserved, source.progress_reserved_bytes);
                owner_reserved = std::max(owner_reserved, source.owner_reserved_bytes);
            }
            if (source.version >= 9)
            {
                control_separated = true;
                control_reserved = std::max(control_reserved, farming_control_reservation(source, alignment(replica)));
            }
        }
        uint64_t available = reserved > used ? reserved - used : 0;
        if (separated)
        {
            auto const protected_bytes = grow_add(grow_add(progress_reserved, owner_reserved), plans < 2 ? largest_plan : 0);
            available = available > protected_bytes ? available - protected_bytes : 0;
            if (control_separated)
            {
                auto const controls = farming_update_allocation(replica.metadata->path, true);
                require(controls <= updates, "farming control allocation exceeds total request bookkeeping");
                auto const category_reserved = control ? std::min(control_reserved, update_reserved) :
                    update_reserved > control_reserved ? update_reserved - control_reserved : 0;
                auto const category_used = control ? controls : updates - controls;
                require(category_used <= category_reserved, "farming request metadata exceeds its protected control or policy allowance; review plan --upgrade-job");
                available = std::min(available, category_reserved - category_used);
            }
            else available = std::min(available, update_reserved > updates ? update_reserved - updates : 0);
        }
        auto const needed = farming_update_aligned(bytes, alignment(replica));
        if (needed > available) throw FarmingUpdateSpaceError(needed, available);
        file_io::Space space{}; require(!file_io::space_path(replica.metadata->root, &space), "cannot inspect farming update free space");
        require(space.f_bavail <= UINT64_MAX / space.f_frsize, "farming update free-space overflow");
        auto const free = space.f_bavail * space.f_frsize;
        require(free >= state_.headroom && needed <= free - state_.headroom,
            "farming update would consume the required free-space allowance");
    }
    void publish(std::string const& name, std::string const& body)
    {
        auto const contents = checked(body);
        for (auto const& replica : replicas_)
        {
            auto const path = replica->metadata->path / name;
            if (farm_metadata_detail::present(path))
            { require(read_farming_state(path, memory_).body == body, "conflicting immutable farming update record"); continue; }
            admit(*replica, contents.size());
            replica->metadata->check(); stream_state::atomic_save(path, contents, false); replica->metadata->check();
        }
    }
    bool committed(std::string const& id) const
    {
        for (auto const& replica : replicas_)
            for (auto const* name : {"grow-farm.plan", "grow-farm.plan.next"})
            {
                auto const path = replica->metadata->path / name;
                if (!farm_metadata_detail::present(path)) continue;
                if (load_farming_state(path, memory_).applied_update == id) return true;
            }
        return false;
    }
    void converge(std::vector<FarmingUpdateRecord> const& records)
    {
        require_writer();
        for (auto const& record : records)
        {
            publish(request_name(record.file.update.id), record.body);
            if (record.receipt) publish(receipt_name(record.file.update.id, record.receipt->status), farming_update_receipt_text(*record.receipt));
        }
    }
    void resolve_superseded(std::vector<FarmingUpdateRecord> const& saved)
    {
        require_writer(); require(complete(), "resolving competing farming updates requires every policy member");
        for (auto const& record : saved)
        {
            if (record.receipt && farming_update_terminal(record.receipt->status)) continue;
            auto successor = saved.end();
            for (auto at = saved.begin(); at != saved.end(); ++at)
            {
                if (std::find(at->file.update.supersedes.begin(), at->file.update.supersedes.end(), record.file.update.id) == at->file.update.supersedes.end()) continue;
                require(at->file.update.generation > record.file.update.generation, "cyclic farming update supersession");
                if (successor == saved.end()) successor = at;
            }
            if (successor != saved.end()) finish(record.file.update.id, "superseded", "replaced by a later accepted request", successor->file.update.id);
        }
    }
    std::string replace_controls(FarmingControlState const& value, std::vector<std::optional<stream_state::Stamp>> const& expected)
    {
        require_writer(); require(expected.size() == replicas_.size(), "incomplete farming control replacement stamps");
        auto const body = farming_control_state_text(value), contents = checked(body);
        require(contents.size() <= grow_add(farming_control_size_bound(state_), 65), "farming control snapshot exceeds its allocation bound");
        (void)parse_farming_controls(state_, body, memory_);
        for (size_t i = 0; i < replicas_.size(); ++i)
        {
            auto const& replica = replicas_[i]; auto const path = replica->metadata->path / "grow-farm.update.controls";
            auto const pending = replica->metadata->path / "grow-farm.update.controls.next";
            if (farm_metadata_detail::present(pending))
            {
                require(stream_state::stamp(pending).bytes <= grow_add(farming_control_handoff_size_bound(state_, control_bindings_), 65),
                    "interrupted farming control exceeds its read bound");
                (void)parse_farming_controls(state_, read_farming_state(pending, memory_).body, memory_, control_bindings_);
                require(!file_io::unlink_file(pending), "cannot reclaim unpublished farming control snapshot");
                stream_state::sync_path(replica->metadata->path, true);
            }
            auto unchanged = [&]
            {
                require(expected[i] ? farm_metadata_detail::present(path) && stream_state::stamp(path) == *expected[i] :
                    !farm_metadata_detail::present(path), "farming control snapshot changed before replacement");
            };
            unchanged();
            if (expected[i] && read_farming_state(path, memory_).body == body) continue;
            admit(*replica, contents.size(), true); replica->metadata->check();
            auto const staged = stage_farming_state(path, contents, false);
            require(farming_update_allocation(replica->metadata->path) <= state_.update_reserved_bytes,
                "staged farming controls exceed their reserved update metadata allowance");
            file_io::Space staged_space{};
            require(!file_io::space_path(replica->metadata->root, &staged_space) && staged_space.f_frsize &&
                staged_space.f_bavail <= UINT64_MAX / staged_space.f_frsize &&
                staged_space.f_bavail * staged_space.f_frsize >= state_.headroom,
                "staged farming controls consumed declared filesystem headroom");
            grow_failpoint("farm-control-before-publication");
            unchanged(); replica->metadata->check();
            require(!file_io::rename_file(staged, path, expected[i].has_value()), "cannot publish farming control snapshot");
            stream_state::sync_path(replica->metadata->path, true); replica->metadata->check();
            grow_failpoint("farm-control-published");
        }
        return stream_state::digest(body);
    }
public:
    FarmingUpdateMailbox(FarmingState const& state, std::optional<bool> legacy, bool acquire, uint64_t memory = UINT64_MAX,
        std::span<std::filesystem::path const> runtime_paths = {}, std::span<FarmingControlNativeBinding const> control_bindings = {}) :
        state_(state), memory_(memory), control_bindings_(control_bindings.begin(), control_bindings.end())
    {
        if (!control_bindings_.empty()) (void)farming_control_source_shape(state_, control_bindings_);
        std::map<std::string, std::filesystem::path> ordered;
        require(runtime_paths.empty() || runtime_paths.size() >= state.directories.size(), "farming update runtime bindings are incomplete");
        bool const has_members = std::any_of(state.directories.begin(), state.directories.end(), farming_member);
        terminal_ = !has_members;
        if (terminal_) return;
        for (size_t i = 0; i < state.directories.size(); ++i)
        {
            auto const& directory = state.directories[i];
            if (has_members && !farming_member(directory)) continue;
            require(!directory.directory_identity.empty() && directory.directory_identity.size() <= 256 &&
                ordered.emplace(directory.directory_identity, runtime_paths.empty() ? file_io::native_path(directory.directory) : runtime_paths[i]).second,
                "invalid farming update member identity");
        }
        for (auto const& [identity, root] : ordered)
        {
            auto metadata = std::make_unique<FarmingMetadataDirectory>(root, legacy);
            if (!metadata->root_pin || !metadata->recorded) continue;
            if (farming_directory_identity(root) != identity) continue;
            auto source = metadata->path / "grow-farm.plan";
            if (!farm_metadata_detail::present(source)) source = metadata->path / "grow-farm.plan.next";
            auto const saved = load_farming_state(source, memory_);
            require(saved.job_id == state.job_id, "farming update member belongs to another job");
            auto replica = std::make_unique<Replica>(); replica->metadata = std::move(metadata);
            replicas_.push_back(std::move(replica));
        }
        require(!replicas_.empty(), "no available matching farming member can retain an update");
        if (acquire)
        {
            acquired_ = true;
            for (auto const& replica : replicas_)
            {
                auto const path = replica->metadata->path / "grow-farm.update.lock";
                int const fd = file_io::open_file(path, file_io::create | file_io::read_write | file_io::close_on_exec | file_io::no_follow, 0600);
                require(fd >= 0, "cannot open farming update lock");
                file_io::Stat info{};
                if (file_io::inspect_fd(fd, &info) || !file_io::is_regular(info.st_mode) || info.st_nlink != 1)
                { file_io::close_file(fd); throw std::runtime_error("invalid farming update lock"); }
                if (file_io::lock_file(fd, file_io::lock_exclusive | file_io::lock_nonblocking))
                { file_io::close_file(fd); acquired_ = false; break; }
                replica->lock = fd;
            }
            if (!acquired_) for (auto const& replica : replicas_) if (replica->lock >= 0)
            { file_io::close_file(replica->lock); replica->lock = -1; }
        }
        check(); if (acquired_) cleanup_partials(); read_floor();
    }
    bool locked() const { return acquired_; }
    bool terminal() const { return terminal_; }
    bool complete() const
    {
        auto const members = size_t(std::count_if(state_.directories.begin(), state_.directories.end(), farming_member));
        return !members || replicas_.size() == members;
    }
    std::filesystem::path manifest() const
    {
        require(!replicas_.empty(), "retired farming job has no writable policy member");
        auto const path = replicas_.front()->metadata->path / "grow-farm.plan";
        return farm_metadata_detail::present(path) ? path : replicas_.front()->metadata->path / "grow-farm.plan.next";
    }
    uint64_t floor() const { read_floor(); return floor_; }
    FarmingControlState control_state(std::vector<std::optional<stream_state::Stamp>>* stamps = nullptr) const
    {
        check();
        FarmingControlState result;
        result.job_id = state_.job_id; result.fingerprint = state_.fingerprint;
        result.native_members = farming_control_native_members(state_);
        result.native_binding = farming_control_native_digest(state_);
        result.source_revision = state_.policy_revision; result.source_policy = farming_policy_digest(state_);
        if (stamps) stamps->clear();
        for (auto const& replica : replicas_)
        {
            auto const path = replica->metadata->path / "grow-farm.update.controls";
            if (!farm_metadata_detail::present(path)) { if (stamps) stamps->push_back({}); continue; }
            require(stream_state::stamp(path).bytes <= grow_add(farming_control_handoff_size_bound(state_, control_bindings_), 65),
                "farming control snapshot exceeds its read bound");
            FarmingStateFile file;
            for (unsigned attempt = 0; ; ++attempt)
            {
                auto const before = stream_state::stamp(path);
                try { file = read_farming_state(path, memory_); break; }
                catch (std::runtime_error const&)
                {
                    if (attempt >= 2 || stream_state::stamp(path) == before) throw;
                }
            }
            auto const incoming = parse_farming_controls(state_, file.body, memory_, control_bindings_);
            result = merge_farming_controls(state_, result, incoming);
            if (stamps) stamps->push_back(file.stamp);
        }
        check(); return result;
    }
    std::string submit_control(std::string_view action, std::vector<std::string> const& destinations = {})
    {
        if (terminal_)
        {
            require(action == "pause" || action == "drain" || action == "resume" || action == "retire", "invalid local farming control action");
            for (auto const& destination : destinations)
                require(std::any_of(state_.directories.begin(), state_.directories.end(),
                    [&](auto const& directory) { return directory.directory_identity == destination; }), "farming control target is not a saved destination");
            return stream_state::digest("retired farming intent " + state_.job_id);
        }
        require_writer();
        std::vector<std::optional<stream_state::Stamp>> expected;
        auto current = control_state(&expected);
        std::string actor;
        for (auto const& replica : replicas_)
        {
            auto const identity = farming_directory_identity(replica->metadata->root);
            auto const member = std::find_if(state_.directories.begin(), state_.directories.end(),
                [&](auto const& directory) { return directory.directory_identity == identity && farming_member(directory); });
            if (member != state_.directories.end() && (actor.empty() || identity < actor)) actor = identity;
        }
        require(!actor.empty(), "no active native member can retain farming control intent");
        require(action == "pause" || action == "drain" || action == "resume" || action == "retire", "invalid local farming control action");
        auto const desired = farming_update_control(action);
        auto same = [&](std::string const& field)
        {
            auto const at = current.fields.find(field);
            return at != current.fields.end() && at->second.size() == 1 && at->second.front().value == desired;
        };
        bool const repeated = destinations.empty() ? same("control") :
            std::all_of(destinations.begin(), destinations.end(), [&](auto const& destination) { return same("control:" + destination); });
        auto next = repeated ? current : advance_farming_controls(state_, std::move(current), actor, action, destinations, farming_policy_digest(state_));
        return replace_controls(next, expected);
    }
    std::string handoff_controls()
    {
        if (terminal_) return {};
        require_writer(); require(complete(), "native farming control handoff requires every active member");
        require(!control_bindings_.empty(), "native farming control handoff lacks approved source bindings");
        std::vector<std::optional<stream_state::Stamp>> expected;
        auto const current = control_state(&expected);
        if (current.context.empty()) return {};
        return replace_controls(current, expected);
    }
    std::vector<FarmingUpdateRecord> records() const
    {
        if (terminal_) return {};
        check(); read_floor();
        std::map<std::string, FarmingUpdateRecord> requests;
        std::map<std::string, FarmingUpdateReceipt> receipts;
        std::vector<std::pair<std::string, std::string>> old_errors;
        uint64_t bytes = 0; size_t files = 0;
        for (auto const& replica : replicas_)
        {
            for (auto const& entry : std::filesystem::directory_iterator(replica->metadata->path))
            {
                auto const name = file_io::path_text(entry.path().filename());
                bool const old = name == "grow-farm.update";
                if (!old && (!name.starts_with("grow-farm.update-") || name.find(".partial.") != name.npos)) continue;
                require(++files <= replicas_.size() * (farming_update_history * 3 + 1), "farming update record count exceeds its bound");
                file_io::Stat info{};
                require(!file_io::inspect_link(entry.path(), &info) && file_io::is_regular(info.st_mode) && info.st_nlink == 1,
                    "invalid farming update record object");
                auto const saved = read_farming_state(entry.path(), memory_);
                if (old || name.ends_with(".request"))
                {
                    FarmingUpdateFile file{parse_farming_update(saved.body), saved.stamp};
                    require(file.update.job_id == state_.job_id && (old || name == request_name(file.update.id)),
                        "farming update request name or job differs from its content");
                    auto at = requests.find(file.update.id);
                    if (at != requests.end()) require(at->second.body == saved.body, "conflicting copies of one farming update request");
                    else
                    {
                        require(requests.size() < farming_update_history, "farming update history needs complete-member compaction");
                        bytes = grow_add(bytes, saved.body.size());
                        require(bytes < memory_ / 32, "farming update union exceeds --max-ram");
                        auto const id = file.update.id;
                        requests.emplace(id, FarmingUpdateRecord{std::move(file), {}, saved.body, saved.digest});
                    }
                }
                else
                {
                    auto receipt = parse_farming_update_receipt(saved.body);
                    require(receipt.job_id == state_.job_id && name == receipt_name(receipt.id, receipt.status), "invalid farming update receipt binding");
                    auto [at, inserted] = receipts.emplace(receipt.id, receipt);
                    if (inserted)
                    {
                        require(receipts.size() <= farming_update_history, "too many farming update receipts");
                        bytes = grow_add(bytes, saved.body.size());
                        require(bytes < memory_ / 32, "farming update union exceeds --max-ram");
                    }
                    else
                    {
                        if (at->second.status == receipt.status) require(farming_update_receipt_text(at->second) == saved.body,
                            "conflicting copies of one farming update receipt");
                        else if (at->second.status == "accepted") at->second = receipt;
                        else if (receipt.status != "accepted") require(farming_update_receipt_text(at->second) == saved.body,
                            "conflicting terminal farming update receipts");
                    }
                }
            }
            auto const error = replica->metadata->path / "grow-farm.update.error";
            if (farm_metadata_detail::present(error))
            {
                auto const saved = read_farming_state(error, memory_);
                std::istringstream in(saved.body); std::string magic, id, reason; std::getline(in, magic);
                in >> std::quoted(id) >> std::quoted(reason);
                require(in && magic == "ordinary-group-farming-update-error-v1", "invalid legacy farming update refusal");
                in >> std::ws; require(in.eof() && reason.size() <= 4096, "invalid legacy farming update refusal data");
                old_errors.emplace_back(std::move(id), std::move(reason));
            }
        }
        for (auto const& [id, reason] : old_errors) if (auto at = requests.find(id); at != requests.end())
        {
            FarmingUpdateReceipt receipt{id, state_.job_id, at->second.digest, "refused", reason.substr(0, farming_update_reason_bytes), {}, at->second.file.update.generation};
            auto [previous, inserted] = receipts.emplace(id, receipt);
            if (!inserted && previous->second.status == "accepted") previous->second = receipt;
            else if (!inserted) require(farming_update_receipt_text(previous->second) == farming_update_receipt_text(receipt),
                "legacy refusal conflicts with a farming update receipt");
        }
        for (auto const& [id, receipt] : receipts)
            require(requests.contains(id) || (receipt.generation ? receipt.generation <= floor_ : legacy_floor_),
                "farming update receipt lacks its immutable request");
        std::vector<FarmingUpdateRecord> result;
        for (auto& [id, record] : requests)
        {
            if (auto at = receipts.find(id); at != receipts.end())
            {
                require(at->second.request_digest == record.digest && at->second.generation == record.file.update.generation,
                    "farming update receipt differs from its request");
                record.receipt = at->second;
            }
            result.push_back(std::move(record));
        }
        std::sort(result.begin(), result.end(), [](auto const& a, auto const& b)
        { return std::tie(a.file.update.generation, a.file.update.id) < std::tie(b.file.update.generation, b.file.update.id); });
        check(); return result;
    }
    std::optional<FarmingUpdateReceipt> receipt(std::string const& id) const
    {
        for (auto const& record : records()) if (record.file.update.id == id)
            return record.receipt.value_or(FarmingUpdateReceipt{id, state_.job_id, record.digest, "accepted", {}, {}, record.file.update.generation});
        return {};
    }
    std::vector<FarmingUpdateRecord> pending() const
    {
        auto saved = records();
        std::set<std::string> replaced;
        for (auto const& record : saved)
            replaced.insert(record.file.update.supersedes.begin(), record.file.update.supersedes.end());
        std::erase_if(saved, [&](auto const& record)
        {
            return (record.receipt && farming_update_terminal(record.receipt->status)) || replaced.contains(record.file.update.id) ||
                record.file.update.id == state_.applied_update;
        });
        return saved;
    }
    bool actionable() const { return complete() && !pending().empty(); }
    void finish(std::string const& id, std::string_view status, std::string_view reason = {}, std::string_view successor = {})
    {
        require_writer();
        auto const saved = records();
        auto const at = std::find_if(saved.begin(), saved.end(), [&](auto const& record) { return record.file.update.id == id; });
        require(at != saved.end(), "farming update result lacks its immutable request");
        if (at->receipt && farming_update_terminal(at->receipt->status))
        {
            require(at->receipt->status == status && at->receipt->successor == successor, "farming update already has another terminal result");
            converge(saved); return;
        }
        require(status == "accepted" || farming_update_terminal(status), "invalid farming update result state");
        if (status == "refused" || status == "superseded") require(complete(), "resolving competing farming updates requires every policy member");
        if (status != "accepted" && status != "applied") require(!committed(id), "farming update is already staged or committed; resume its publication");
        if (status == "applied") require(committed(id), "farming update has no staged or committed policy receipt");
        FarmingUpdateReceipt result{id, state_.job_id, at->digest, std::string(status), std::string(reason.substr(0, farming_update_reason_bytes)),
            std::string(successor), at->file.update.generation};
        converge(saved); publish(receipt_name(id, result.status), farming_update_receipt_text(result));
    }
    std::optional<FarmingUpdateFile> next(FarmingState const& current)
    {
        require_writer(); require(complete(), "pending farming update is waiting for unavailable policy members");
        require(current.job_id == state_.job_id && current.directories.size() == state_.directories.size() &&
            std::equal(current.directories.begin(), current.directories.end(), state_.directories.begin(),
                [](auto const& a, auto const& b) { return a.directory_identity == b.directory_identity; }),
            "farming update membership changed; reopen its mailbox before selecting another request");
        auto saved = records(); converge(saved);
        // Replacements record their causality in the immutable request first.
        // Only the complete lock holder emits a superseded terminal receipt:
        // disconnected publishers can then replace the same predecessor
        // without writing contradictory outcomes on separate partitions.
        resolve_superseded(saved);
        saved = records();
        for (auto const& record : saved)
        {
            if (record.receipt && farming_update_terminal(record.receipt->status)) continue;
            auto const& update = record.file.update;
            if ((update.version >= 3 && update.generation <= floor_) || (update.version < 3 && legacy_floor_))
            {
                if (committed(update.id)) finish(update.id, "applied");
                else finish(update.id, "refused", "request predates compacted history; refresh status and submit a fresh request");
                continue;
            }
            if (current.applied_update == update.id) return record.file;
            try { return FarmingUpdateFile{farming_update_rebase(update, current), record.file.stamp}; }
            catch (std::runtime_error const& error) { finish(update.id, "refused", error.what()); }
        }
        return {};
    }
    bool cancel()
    {
        if (terminal_) return false;
        require_writer(); bool found = false;
        for (auto const& record : records()) if (!record.receipt || !farming_update_terminal(record.receipt->status))
        { finish(record.file.update.id, "cancelled"); found = true; }
        return found;
    }
    void compact()
    {
        require_writer(); if (!complete()) return;
        auto saved = records(); converge(saved); read_floor();
        uint64_t ceiling = floor_, unresolved = UINT64_MAX; bool old_pending = false;
        for (auto const& record : saved)
        {
            auto const& update = record.file.update;
            if (!record.receipt || !farming_update_terminal(record.receipt->status))
            { if (update.version < 3) old_pending = true; else unresolved = std::min(unresolved, update.generation); }
            else ceiling = std::max(ceiling, update.generation);
        }
        if (unresolved != UINT64_MAX) ceiling = std::min(ceiling, unresolved - 1);
        if (ceiling < floor_) return;
        std::ostringstream out;
        out << "ordinary-group-farming-update-floor-v1\n" << std::quoted(state_.job_id) << ' ' << ceiling << ' ' << !old_pending << '\n';
        auto const contents = checked(out.str());
        // Complete-member request locks exclude every publisher. No request or
        // result is removed until every native member durably acknowledges it.
        for (auto const& replica : replicas_)
        {
            auto const path = replica->metadata->path / "grow-farm.update.floor";
            admit(*replica, contents.size()); replica->metadata->check();
            stream_state::atomic_save(path, contents); replica->metadata->check();
        }
        floor_ = ceiling; legacy_floor_ = !old_pending;
        for (auto const& record : saved)
        {
            auto const& update = record.file.update;
            if (!record.receipt || !farming_update_terminal(record.receipt->status) ||
                (update.version >= 3 ? update.generation > floor_ : !legacy_floor_)) continue;
            for (auto const& replica : replicas_)
            {
                for (auto const& name : {request_name(update.id), receipt_name(update.id, "accepted"), receipt_name(update.id, "refused"),
                    receipt_name(update.id, "applied"), receipt_name(update.id, "cancelled"), receipt_name(update.id, "superseded")})
                {
                    auto const path = replica->metadata->path / name;
                    if (farm_metadata_detail::present(path)) require(!file_io::unlink_file(path), "cannot compact farming update history");
                }
                auto const legacy = replica->metadata->path / "grow-farm.update";
                if (farm_metadata_detail::present(legacy) && load_farming_update(legacy, memory_).update.id == update.id)
                {
                    require(!file_io::unlink_file(legacy), "cannot compact legacy farming update");
                    auto const error = replica->metadata->path / "grow-farm.update.error";
                    if (farm_metadata_detail::present(error)) require(!file_io::unlink_file(error), "cannot compact legacy farming update refusal");
                }
                stream_state::sync_path(replica->metadata->path, true); replica->metadata->check();
            }
        }
        // Interrupted pruning can leave a terminal receipt after its request
        // was removed. The acknowledged floor is its bounded tombstone.
        for (auto const& replica : replicas_)
        {
            for (auto const& entry : std::filesystem::directory_iterator(replica->metadata->path))
            {
                auto const name = file_io::path_text(entry.path().filename());
                if (!name.starts_with("grow-farm.update-") || name.ends_with(".request") || name.find(".partial.") != name.npos) continue;
                auto const receipt = parse_farming_update_receipt(read_farming_state(entry.path(), memory_).body);
                if (receipt.generation <= floor_ && !farm_metadata_detail::present(replica->metadata->path / request_name(receipt.id)))
                    require(!file_io::unlink_file(entry.path()), "cannot compact orphan farming update receipt");
            }
            stream_state::sync_path(replica->metadata->path, true); replica->metadata->check();
        }
    }
    std::string submit(FarmingUpdate update)
    {
        require_writer();
        auto saved = records();
        if (complete()) { resolve_superseded(saved); saved = records(); }
        if (complete() && saved.size() >= farming_update_slots) { compact(); saved = records(); }
        size_t waiting = 0; uint64_t generation = floor();
        auto const desired = farming_update_fields(update);
        auto const user_fields = update.expected;
        std::set<std::string> replaced, contested;
        std::map<std::string, std::string> waiting_values;
        for (auto const& record : saved)
            replaced.insert(record.file.update.supersedes.begin(), record.file.update.supersedes.end());
        auto active = [&](FarmingUpdateRecord const& record)
        { return (!record.receipt || !farming_update_terminal(record.receipt->status)) && !replaced.contains(record.file.update.id); };
        auto const unresolved = std::count_if(saved.begin(), saved.end(), [](auto const& record)
        { return !record.receipt || !farming_update_terminal(record.receipt->status); });
        for (auto const& record : saved)
        {
            auto const& old = record.file.update;
            generation = std::max(generation, old.generation);
            if (!active(record)) continue;
            ++waiting;
            if (old.version < 3) continue;
            if (farming_update_control_only(old) != farming_update_control_only(update)) continue;
            auto const previous = farming_update_fields(old);
            for (auto const& [field, expected] : old.expected)
            {
                auto const [at, inserted] = waiting_values.emplace(field, previous.at(field));
                if (!inserted && at->second != previous.at(field)) contested.insert(field);
            }
            bool same = update.source_policy == old.source_policy;
            for (auto const& [field, expected] : update.expected)
                same = same && old.expected.contains(field) &&
                    previous.contains(field) && previous.at(field) == desired.at(field);
            if (same) return old.id;
        }
        require(size_t(unresolved) < farming_update_slots,
            "too many unresolved farming updates; resolve or cancel a waiting request before submitting another");
        for (auto const& record : saved)
        {
            auto const& old = record.file.update;
            if (old.version < 3 || !active(record) || farming_update_control_only(old) != farming_update_control_only(update) ||
                std::any_of(old.expected.begin(), old.expected.end(), [&](auto const& field)
                { return contested.contains(field.first) && !user_fields.contains(field.first); })) continue;
            FarmingUpdate merged;
            try { merged = farming_update_rebase(old, state_); }
            catch (std::runtime_error const&) { continue; }
            bool const same_scope = std::set<std::string>(old.action_destinations.begin(), old.action_destinations.end()) ==
                std::set<std::string>(update.action_destinations.begin(), update.action_destinations.end());
            bool const compatible_action = old.action.empty() || update.action.empty() || same_scope ||
                (old.action == update.action && !old.action_destinations.empty() && !update.action_destinations.empty());
            if (!compatible_action)
            {
                auto const previous = farming_update_fields(old);
                // A payload carries one action. Retain different actions on
                // different scopes as ordered requests, admitting both the
                // original value and the already reviewed predecessor value.
                for (auto& [field, expected] : update.expected) if (old.expected.contains(field))
                    expected = farming_update_expected_union(expected, previous.at(field));
                continue;
            }
            std::set<std::string> fields;
            for (auto const& [field, expected] : old.expected) fields.insert(field);
            for (auto const& [field, expected] : update.expected) fields.insert(field);
            farming_update_fields_visit(merged, update, [&](auto const* field, auto& target, auto const& value)
            { if (update.expected.contains(field)) target = value; });
            for (size_t i = 0; i < update.directories.size(); ++i)
                if (std::find(merged.directory_identities.begin(), merged.directory_identities.end(), update.directory_identities[i]) == merged.directory_identities.end())
                { merged.directories.push_back(update.directories[i]); merged.directory_identities.push_back(update.directory_identities[i]); }
            if (!update.action.empty())
            {
                merged.action = update.action;
                if (old.action.empty() || update.action_destinations.empty() || old.action != update.action) merged.action_destinations = update.action_destinations;
                else for (auto const& identity : update.action_destinations)
                    if (std::find(merged.action_destinations.begin(), merged.action_destinations.end(), identity) == merged.action_destinations.end())
                        merged.action_destinations.push_back(identity);
            }
            merged.supersedes = update.supersedes;
            merged.supersedes.push_back(old.id);
            std::sort(merged.supersedes.begin(), merged.supersedes.end());
            merged.supersedes.erase(std::unique(merged.supersedes.begin(), merged.supersedes.end()), merged.supersedes.end());
            farming_update_preconditions(merged, state_, fields);
            auto preserve_choices = [&](auto const& prior)
            {
                for (auto const& [field, expected] : prior)
                    if (expected.starts_with(farming_update_choices_prefix) && merged.expected.contains(field))
                        merged.expected[field] = farming_update_expected_union(merged.expected[field], expected);
            };
            preserve_choices(old.expected); preserve_choices(update.expected);
            update = std::move(merged);
        }
        size_t remaining = waiting;
        for (auto const& record : saved)
            if (active(record) &&
                std::find(update.supersedes.begin(), update.supersedes.end(), record.file.update.id) != update.supersedes.end()) --remaining;
        require(remaining < farming_update_slots, "too many unresolved farming updates; resolve or cancel a waiting request before submitting another");
        require(generation < UINT64_MAX, "farming update generation exhausted");
        update.version = 3; update.job_id = state_.job_id; update.generation = generation + 1;
        update.id = farming_update_id(update);
        auto const body = farming_update_text(update);
        check_farming_state_size(body.size(), memory_);
        (void)parse_farming_update(body);
        uint64_t union_bytes = grow_add(body.size(), farming_update_receipt_bytes);
        for (auto const& record : saved) union_bytes = grow_add(union_bytes, grow_add(record.body.size(), farming_update_receipt_bytes));
        require(union_bytes < memory_ / 32, "farming update union exceeds --max-ram");
        for (auto const& replica : replicas_)
        {
            auto const required = grow_add(farming_update_aligned(body.size() + 65, alignment(*replica)),
                grow_add(farming_update_aligned(farming_update_receipt_bytes, alignment(*replica)),
                    farming_update_aligned(farming_update_receipt_bytes, alignment(*replica))));
            admit(*replica, required);
        }
        converge(saved); publish(request_name(update.id), body); finish(update.id, "accepted");
        if (complete()) resolve_superseded(records());
        return update.id;
    }
};
} // namespace group_grow
