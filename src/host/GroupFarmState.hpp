// Private directory farming jobs. Progress remains in each group's durable
// construction protocol; replicated job revisions retain keys and assignments.
#pragma once
#include "host/GroupEngine.hpp"
#include "host/GroupFarming.hpp"
#include "host/GroupHarvesterProfile.hpp"
// Only the shared durability, checksum and file-stamp primitives are used.
// The unrelated stream checkpoint functions still require their build macro.
#ifndef STREAM_IDENTITY
#define STREAM_IDENTITY GROW_IDENTITY
#define GROW_FARM_STREAM_IDENTITY
#endif
#include "host/GroupCheckpoint.hpp"
#ifdef GROW_FARM_STREAM_IDENTITY
#undef STREAM_IDENTITY
#undef GROW_FARM_STREAM_IDENTITY
#endif

namespace group_grow
{
// Bump this when owner/base-seed derivation, family/meta identity schedules or
// the Rust group-key derivation changes. Construction checkpoints have separate
// format, algorithm and native-layout guards; legacy records retain GROW_IDENTITY.
inline constexpr char farming_key_derivation_identity[] = "directory-farming-keys-1";
inline constexpr char farming_cohort_model_identity[] = "group-farming-4";
inline constexpr char farming_current_model_identity[] = "group-farming-5";
inline constexpr char farming_cohort_key_identity[] = "directory-farming-keys-2";
enum class FarmingStateLoad { recovery, metadata_only };
inline constexpr size_t farming_profile_metadata_bytes = 65536;
inline void check_farming_state_size(uint64_t bytes, uint64_t memory_limit)
{ require(bytes < memory_limit / 16, "saved farming plan exceeds --max-ram"); }
struct FarmingCohort
{
    uint64_t first_file = 0;
    FarmingConfig config;
    FarmingDrivePlan layout;
    uint64_t identity_resources = 1, identity_resource = 0, identity_epoch = 0;
    bool operator==(FarmingCohort const&) const = default;
};
struct FarmingDirectory
{
    std::string directory, filesystem;
    uint64_t capacity_bytes = 0, resource = 0;
    FarmingDrivePlan layout;
    FarmingDestination storage;
    std::string directory_identity;
    uint64_t identity_resources = 1, identity_resource = 0, identity_epoch = 0;
    // Initial forecasts retain the identity schedule. Live capacity admission
    // records exact targets separately, sealing them before publication.
    std::vector<uint16_t> extents;
    uint64_t sealed_files = 0;
    std::string fill_stop;
    // Earlier cohorts end at the following cohort's first file. The current
    // layout starts at layout_first; extents retain absolute file ordinals.
    uint64_t layout_first = 0, policy_files = 0;
    std::vector<FarmingCohort> cohorts;
    std::string control = "active";
};
struct FarmingState
{
    unsigned version = 6;
    // Preserve the creating build as provenance and in every revision digest.
    // Schema/model/key derivation and numeric representation define v6 plan
    // compatibility independently.
    std::string producer_build = GROW_IDENTITY;
    FarmingConfig config;
    uint8_t identity_k = 28, identity_strength = 2;
    // Identifies the durable live request incorporated by this policy revision.
    std::string applied_update;
    uint64_t policy_revision = 0;
    std::string control = "active";
    uint64_t request_floor = 0;
    uint64_t manifest_reserved_bytes = 0;
    uint64_t update_reserved_bytes = 0, progress_reserved_bytes = 0;
    std::string fingerprint;
    FarmingPlan estimate;
    std::vector<FarmingDirectory> directories;
    std::string job_id, parent_digest;
    uint64_t revision = 1;
    std::string seed, seed_owner, farmer, pool;
    bool pool_public_key = false;
    bool fixed_extent = false;
    bool capacity_fill = false;
    std::optional<FarmingProfileMetadata> profile_metadata;
    // Earlier revisions retain their original public JSON and canonical bytes.
    std::string profile_json;
    uint64_t capacity = 0, headroom = 64 * 1024 * 1024;
    uint64_t max_groups = UINT64_MAX, max_total_members = UINT64_MAX;
};
inline char const* farming_state_model(FarmingState const& state)
{ return state.version >= 8 ? farming_current_model_identity : state.version >= 7 ? farming_cohort_model_identity : farming_model_identity; }
inline bool farming_control_valid(std::string_view value)
{ return value == "active" || value == "paused" || value == "draining" || value == "retired"; }
inline char const* farming_state_keys(FarmingState const& state)
{ return state.version >= 7 ? farming_cohort_key_identity : farming_key_derivation_identity; }
inline FarmingCohort farming_cohort(FarmingState const& state, FarmingDirectory const& d, uint64_t ordinal)
{
    if (ordinal >= d.layout_first)
        return {d.layout_first, state.config, d.layout, d.identity_resources, d.identity_resource, d.identity_epoch};
    for (size_t i = d.cohorts.size(); i-- > 0;)
        if (ordinal >= d.cohorts[i].first_file) return d.cohorts[i];
    throw std::runtime_error("saved farming cohort does not cover its file ordinal");
}
inline uint64_t farming_layout_files(FarmingDirectory const& d)
{ return grow_add(d.layout_first, d.layout.files); }
inline uint64_t farming_layout_members(FarmingDirectory const& d)
{
    uint64_t result = d.layout.total_members;
    for (size_t i = 0; i < std::min<uint64_t>(d.layout_first, d.extents.size()); ++i)
        result = grow_add(result, d.extents[i]);
    return result;
}
inline bool same_farming_assignment(FarmingState const& before, FarmingDirectory const& a,
    FarmingState const& after, FarmingDirectory const& b, uint64_t ordinal)
{
    auto const x = farming_cohort(before, a, ordinal), y = farming_cohort(after, b, ordinal);
    return x.config.k == y.config.k && x.config.strength == y.config.strength &&
        x.identity_resources == y.identity_resources && x.identity_resource == y.identity_resource &&
        x.identity_epoch == y.identity_epoch &&
        x.layout.family(ordinal - x.first_file) == y.layout.family(ordinal - y.first_file) &&
        x.layout.meta(ordinal - x.first_file) == y.layout.meta(ordinal - y.first_file);
}
inline bool valid_farming_policy_revision(FarmingState const& before, FarmingState const& after)
{
    if (after.version < 7 || after.version > 8 || after.version < before.version ||
        after.policy_revision != after.revision || after.policy_revision <= before.policy_revision ||
        !before.capacity_fill || !after.capacity_fill || !after.fixed_extent ||
        after.directories.size() < before.directories.size() || before.producer_build != after.producer_build ||
        after.identity_k != (before.version >= 7 ? before.identity_k : before.config.k) ||
        after.identity_strength != (before.version >= 7 ? before.identity_strength : before.config.strength)) return false;
    for (size_t lane = 0; lane < before.directories.size(); ++lane)
    {
        auto const& a = before.directories[lane]; auto const& b = after.directories[lane];
        if (b.policy_files < a.sealed_files || b.policy_files > a.extents.size() || b.policy_files > b.extents.size() ||
            (after.version < 8 ? a.sealed_files != b.sealed_files :
                (b.sealed_files < a.sealed_files || b.sealed_files > b.policy_files))) return false;
        for (uint64_t file = 0; file < b.policy_files; ++file)
        {
            bool const newly_sealed = after.version >= 8 && file >= a.sealed_files && file < b.sealed_files;
            if (!b.extents[size_t(file)] || (newly_sealed ? b.extents[size_t(file)] > a.extents[size_t(file)] :
                a.extents[size_t(file)] != b.extents[size_t(file)]) || !same_farming_assignment(before, a, after, b, file)) return false;
        }
    }
    return true;
}
inline FarmingAssessment farming_assess_state(FarmingState const& state,
    std::vector<std::vector<uint16_t>> const& members, uint64_t memory_limit, FarmingAdjustments* adjustments = nullptr)
{
    require(members.size() == state.directories.size(), "farming assessment destination count differs");
    farming_detail::PlanningMemory memory{memory_limit};
    size_t rows = state.directories.size();
    for (auto const& d : state.directories) rows += d.cohorts.size();
    require(rows <= 4096, "farming cohort assessment exceeds its work bound");
    memory.add(rows * (sizeof(FarmingConfig) + sizeof(FarmingDrivePlan) + sizeof(FarmingDestination) + sizeof(std::vector<uint16_t>)) * 2);
    for (auto const& lane : members) memory.add(lane.size() * sizeof(uint16_t));
    std::vector<FarmingConfig> configs; configs.reserve(rows);
    std::vector<FarmingDrivePlan> layouts; layouts.reserve(rows);
    std::vector<FarmingDestination> destinations; destinations.reserve(rows);
    std::vector<std::vector<uint16_t>> extents; extents.reserve(rows);
    auto envelope = state.config;
    envelope.filter_bits = uint8_t(farming_filter_bits(state.config));
    envelope.future_filter_bits = uint8_t(farming_filter_bits(state.config, true));
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        auto const& d = state.directories[lane];
        for (size_t i = 0; i <= d.cohorts.size(); ++i)
        {
            auto cohort = i < d.cohorts.size() ? d.cohorts[i] : farming_cohort(state, d, d.layout_first);
            auto const first = std::min<uint64_t>(cohort.first_file, members[lane].size());
            auto const last = std::min<uint64_t>(i < d.cohorts.size() ?
                (i + 1 < d.cohorts.size() ? d.cohorts[i + 1].first_file : d.layout_first) : members[lane].size(), members[lane].size());
            // All cohorts share the current harvester and filter assumptions.
            // Legacy revisions retain their global worst-case service envelope;
            // current revisions assess each cohort's own service costs together.
            cohort.config.filter_bits = state.config.filter_bits;
            cohort.config.future_filter_bits = state.config.future_filter_bits;
            if (state.version < 8 && last > first)
            {
                envelope.k = std::max(envelope.k, cohort.config.k);
                envelope.filter_bits = std::min(envelope.filter_bits, uint8_t(farming_filter_bits(cohort.config)));
                envelope.future_filter_bits = std::min(envelope.future_filter_bits, uint8_t(farming_filter_bits(cohort.config, true)));
                envelope.chain_seconds_per_member = std::max(envelope.chain_seconds_per_member, cohort.config.chain_seconds_per_member);
                envelope.index_seconds_per_entry = std::max(envelope.index_seconds_per_entry, cohort.config.index_seconds_per_entry);
                envelope.compressed_index_bytes_per_entry = std::max(envelope.compressed_index_bytes_per_entry, cohort.config.compressed_index_bytes_per_entry);
                envelope.lookup_memory_per_member = std::max(envelope.lookup_memory_per_member, cohort.config.lookup_memory_per_member);
            }
            configs.push_back(cohort.config); layouts.push_back(cohort.layout); destinations.push_back(d.storage);
            extents.emplace_back(members[lane].begin() + first, members[lane].begin() + last);
        }
    }
    farming_detail::validate(envelope, {});
    if (adjustments)
    {
        *adjustments = farming_adjustments(envelope, destinations, layouts, extents, memory.limit - memory.used, configs, state.version >= 8);
        return adjustments->baseline;
    }
    return farming_detail::assess(envelope, destinations, layouts, &extents, memory, nullptr, configs, state.version >= 8);
}
inline bool same_farming_profile_metadata(FarmingState const& before, FarmingState const& after)
{ return before.profile_metadata == after.profile_metadata && before.profile_json == after.profile_json; }
inline std::string farming_profile_metadata_text(FarmingProfileMetadata const& profile, unsigned version = 5)
{
    require(version >= 6 || !profile.reader_calibration, "scoped reader calibration requires farming plan v6");
    uint64_t bytes = 0;
    auto string = [&](std::string const& value) { bytes = grow_add(bytes, value.size()); };
    string(profile.source); string(profile.captured_at); string(profile.chia_version); string(profile.host_name);
    string(profile.config_status); string(profile.inventory_status);
    if (profile.reader_calibration)
    {
        validate_reader_calibration_scope(*profile.reader_calibration); string(profile.reader_calibration->method);
        string(profile.reader_calibration->host_name); string(profile.reader_calibration->captured_at);
    }
    for (auto const& assumption : profile.assumptions) string(assumption);
    require(bytes <= farming_profile_metadata_bytes && profile.assumptions.size() <= 64,
        "saved farming profile metadata exceeds its bound");
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<long double>::max_digits10);
    out << std::quoted(profile.source) << ' ' << std::quoted(profile.captured_at) << ' '
        << std::quoted(profile.chia_version) << ' ' << std::quoted(profile.host_name) << '\n'
        << std::quoted(profile.config_status) << ' ' << std::quoted(profile.inventory_status) << '\n'
        << profile.detected_logical_cpus << ' ' << profile.detected_available_memory_bytes << ' '
        << profile.inventory_available << ' ' << profile.inventory_v1_files << ' ' << profile.inventory_v2_files << ' '
        << profile.inventory_v2_members << ' ' << profile.inventory_failed_files << ' ' << profile.inventory_no_key_files << '\n'
        << profile.assumptions.size();
    for (auto const& assumption : profile.assumptions) out << ' ' << std::quoted(assumption);
    out << '\n';
    if (version >= 6)
    {
        out << profile.reader_calibration.has_value();
        if (profile.reader_calibration)
        {
            auto const& scope = *profile.reader_calibration;
            out << ' ' << unsigned(scope.k) << ' ' << unsigned(scope.strength) << ' ' << unsigned(scope.reader)
                << ' ' << std::quoted(scope.method) << ' ' << scope.margin
                << ' ' << std::quoted(scope.host_name) << ' ' << std::quoted(scope.captured_at);
        }
        out << '\n';
    }
    auto text = out.str();
    require(text.size() <= farming_profile_metadata_bytes, "saved farming profile metadata exceeds its bound");
    return text;
}
inline FarmingProfileMetadata farming_profile_metadata_read(std::istream& in, unsigned version = 5)
{
    FarmingProfileMetadata profile;
    size_t count = 0;
    in >> std::quoted(profile.source) >> std::quoted(profile.captured_at) >> std::quoted(profile.chia_version)
        >> std::quoted(profile.host_name) >> std::quoted(profile.config_status) >> std::quoted(profile.inventory_status)
        >> profile.detected_logical_cpus >> profile.detected_available_memory_bytes >> profile.inventory_available
        >> profile.inventory_v1_files >> profile.inventory_v2_files >> profile.inventory_v2_members
        >> profile.inventory_failed_files >> profile.inventory_no_key_files >> count;
    require(in && count <= 64, "invalid saved farming profile metadata");
    profile.assumptions.clear(); profile.assumptions.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        std::string assumption; in >> std::quoted(assumption);
        require(bool(in), "incomplete saved farming profile metadata");
        profile.assumptions.push_back(std::move(assumption));
    }
    if (version >= 6)
    {
        bool calibrated = false; in >> calibrated;
        require(bool(in), "invalid saved reader calibration presence");
        if (calibrated)
        {
            ReaderCalibrationScope scope; unsigned k = 0, strength = 0, reader = 0;
            in >> k >> strength >> reader >> std::quoted(scope.method) >> scope.margin
                >> std::quoted(scope.host_name) >> std::quoted(scope.captured_at);
            require(in && k <= UINT8_MAX && strength <= UINT8_MAX && reader <= 1,
                "invalid saved reader calibration scope");
            scope.k = uint8_t(k); scope.strength = uint8_t(strength); scope.reader = FarmingReader(reader);
            validate_reader_calibration_scope(scope); profile.reader_calibration = std::move(scope);
        }
    }
    (void)farming_profile_metadata_text(profile, version);
    return profile;
}
inline void set_farming_state_profile(FarmingState& state, FarmingProfile const& profile)
{
    require(state.version >= 5 && profile.config == state.config,
        "saved farming profile differs from its configuration");
    profile_detail::validate(profile.config);
    validate_reader_calibration_scope(profile, state.config);
    (void)farming_profile_metadata_text(profile, state.version);
    state.profile_metadata = profile;
    state.profile_json.clear();
}
inline std::optional<FarmingProfile> farming_state_profile(FarmingState const& state,
    uint64_t memory_limit = profile_detail::parsing_memory_bytes)
{
    if (state.version < 5)
    {
        if (state.profile_json.empty()) return std::nullopt;
        std::istringstream in(state.profile_json);
        auto profile = read_farming_profile(in, memory_limit);
        require(!profile.reader_calibration, "scoped reader calibration requires farming plan v6");
        return profile;
    }
    if (!state.profile_metadata) return std::nullopt;
    require(state.version >= 6 || !state.profile_metadata->reader_calibration,
        "scoped reader calibration requires farming plan v6");
    require(farming_profile_metadata_memory(*state.profile_metadata) < memory_limit,
        "saved farming profile metadata exceeds its parsing memory budget");
    FarmingProfile profile;
    static_cast<FarmingProfileMetadata&>(profile) = *state.profile_metadata;
    profile.config = state.config;
    profile_detail::validate(profile.config);
    validate_reader_calibration_scope(profile, profile.config);
    return profile;
}
inline void initialize_farming_extents(FarmingState& state, size_t first = 0)
{
    if (!state.capacity_fill) return;
    for (size_t lane = first; lane < state.directories.size(); ++lane)
    {
        auto& d = state.directories[lane];
        require(d.layout.files <= farming_detail::maximum_files, "farming extent exceeds its file work bound");
        d.extents.clear(); d.extents.reserve(size_t(d.layout.files));
        for (uint64_t file = 0; file < d.layout.files; ++file) d.extents.push_back(d.layout.file_members(file));
        d.sealed_files = 0; d.fill_stop.clear();
    }
}
inline uint64_t farming_extent_memory(FarmingState const& state)
{
    uint64_t bytes = grow_add(state.profile_json.capacity(), state.producer_build.capacity());
    if (state.profile_metadata) bytes = grow_add(bytes, farming_profile_metadata_memory(*state.profile_metadata));
    for (auto const& d : state.directories)
        bytes = grow_add(bytes, grow_add(d.extents.capacity() * sizeof(uint16_t),
            grow_add(d.cohorts.capacity() * sizeof(FarmingCohort), d.fill_stop.capacity())));
    return bytes;
}
inline uint64_t farming_revision_margin(FarmingState const& state)
{
    // Every lane can seal its current extent and record its terminal reason,
    // including decimal count growth, after the last successful admission.
    return grow_add(128, state.directories.size() * 64);
}
inline bool farming_revision_fits(FarmingState const& state, uint64_t bytes)
{
    uint64_t const auxiliary = state.version >= 8 ? grow_add(state.update_reserved_bytes, state.progress_reserved_bytes) : 0;
    if (auxiliary > state.manifest_reserved_bytes) return false;
    uint64_t const capacity = (state.manifest_reserved_bytes - auxiliary) / (state.version == 7 ? 4 : 2);
    return bytes <= capacity && farming_revision_margin(state) <= capacity - bytes;
}
inline bool valid_farming_extent_revision(FarmingState const& before, FarmingState const& after)
{
    if (before.capacity_fill != after.capacity_fill || before.version != after.version ||
        before.producer_build != after.producer_build || !same_farming_profile_metadata(before, after) ||
        (before.version >= 7 && (before.identity_k != after.identity_k || before.identity_strength != after.identity_strength)) ||
        before.applied_update != after.applied_update || before.policy_revision != after.policy_revision ||
        before.control != after.control || before.request_floor != after.request_floor ||
        before.update_reserved_bytes != after.update_reserved_bytes || before.progress_reserved_bytes != after.progress_reserved_bytes) return false;
    for (size_t lane = 0; lane < before.directories.size(); ++lane)
    {
        auto const& a = before.directories[lane]; auto const& b = after.directories[lane];
        if (a.cohorts != b.cohorts || a.layout_first != b.layout_first || a.policy_files != b.policy_files ||
            a.control != b.control) return false;
        if (b.sealed_files < a.sealed_files || (before.version < 8 && b.sealed_files > a.sealed_files + 1) ||
            b.extents.size() < a.sealed_files || b.extents.size() > a.extents.size() + 1 ||
            (!a.fill_stop.empty() && (a.fill_stop != b.fill_stop || a.extents != b.extents))) return false;
        bool const stopped = !b.fill_stop.empty();
        if (stopped && b.sealed_files != b.extents.size()) return false;
        if (b.extents.size() < a.extents.size() && b.fill_stop != "capacity" && !(before.version >= 8 &&
            (b.fill_stop == "farming-limit" || b.fill_stop == "group-limit" || b.fill_stop == "total-member-limit" ||
             b.fill_stop == "metadata-capacity"))) return false;
        if (b.extents.size() > a.extents.size() && (a.sealed_files != a.extents.size() || b.extents.back() != 1)) return false;
        for (size_t file = 0; file < a.extents.size(); ++file)
        {
            if (file < a.sealed_files && a.extents[file] != b.extents[file]) return false;
            if (file >= b.extents.size()) continue;
            if (a.extents[file] == b.extents[file]) continue;
            if (file != a.sealed_files && !(before.version >= 8 && file < b.sealed_files)) return false;
            if (b.extents[file] > a.extents[file])
            {
                if (file + 1 < farming_layout_files(a) || b.sealed_files != a.sealed_files ||
                    unsigned(b.extents[file]) != unsigned(a.extents[file]) + 1) return false;
            }
            else if (before.version < 8 ? b.sealed_files != a.sealed_files + 1 : file >= b.sealed_files) return false;
        }
    }
    return true;
}
inline void farming_config_write(std::ostream& out, FarmingConfig const& c)
{
    out << unsigned(c.k) << ' ' << unsigned(c.strength) << ' ' << unsigned(c.filter_bits) << ' '
        << unsigned(c.future_filter_bits) << ' ' << unsigned(c.reader) << ' ' << unsigned(c.objective) << ' '
        << c.harvester_cores << ' ' << c.lookup_workers << ' '
        << c.index_cache_bytes << ' ' << c.memory_bytes << ' ' << c.deadline_seconds << ' '
        << c.challenge_seconds << ' ' << c.group_members_override << ' ' << c.meta_slots_override << ' '
        << unsigned(c.meta_group_override) << ' ' << c.fixed_meta << ' ' << c.chain_seconds_per_member << ' '
        << c.index_seconds_per_entry << ' ' << c.compressed_index_bytes_per_entry << ' ' << c.lookup_memory_per_member << ' '
        << c.cpu_work_weight << ' ' << c.io_work_weight << ' ' << c.max_eligible_families_per_resource << '\n';
}
inline FarmingConfig farming_config_read(std::istream& in)
{
    FarmingConfig c;
    unsigned k, strength, filter, future, reader, objective, members, slots, meta;
    in >> k >> strength >> filter >> future >> reader >> objective >> c.harvester_cores >> c.lookup_workers >> c.index_cache_bytes >>
        c.memory_bytes >> c.deadline_seconds >> c.challenge_seconds >> members >> slots >> meta >> c.fixed_meta >> c.chain_seconds_per_member >> c.index_seconds_per_entry >>
        c.compressed_index_bytes_per_entry >> c.lookup_memory_per_member >> c.cpu_work_weight >> c.io_work_weight >>
        c.max_eligible_families_per_resource;
    require(in && k <= 255 && strength <= 255 && filter <= 255 && future <= 255 && reader <= 1 && objective <= 1 &&
        members <= 65535 && slots <= 256 && meta <= 255 && c.harvester_cores && c.lookup_workers && c.lookup_workers <= 1024 && c.memory_bytes &&
        c.cpu_work_weight > 0 && std::isfinite(c.cpu_work_weight) && c.io_work_weight > 0 && std::isfinite(c.io_work_weight) &&
        c.max_eligible_families_per_resource &&
        c.deadline_seconds > 0 && c.challenge_seconds > 0, "invalid saved farming profile");
    c.k = uint8_t(k); c.strength = uint8_t(strength); c.filter_bits = uint8_t(filter);
    c.future_filter_bits = uint8_t(future); c.reader = FarmingReader(reader); c.objective = FarmingObjective(objective);
    c.group_members_override = uint16_t(members); c.meta_slots_override = uint16_t(slots);
    c.meta_group_override = uint8_t(meta);
    return c;
}
// Storage defaults and overrides contain only the fields previously repeated
// in every directory. Its physical resource remains the directory's resource.
inline constexpr unsigned farming_storage_fields = 511;
inline unsigned farming_storage_overrides(FarmingDestination const& value, FarmingDestination const& defaults)
{
    return unsigned(value.known_device != defaults.known_device) |
        (unsigned(value.nonrotating != defaults.nonrotating) << 1) |
        (unsigned(value.read_bytes_per_second != defaults.read_bytes_per_second) << 2) |
        (unsigned(value.read_latency_seconds != defaults.read_latency_seconds) << 3) |
        (unsigned(value.read_concurrency != defaults.read_concurrency) << 4) |
        (unsigned(value.capacity_bytes != defaults.capacity_bytes) << 5) |
        (unsigned(value.max_file_bytes != defaults.max_file_bytes) << 6) |
        (unsigned(value.max_total_members != defaults.max_total_members) << 7) |
        (unsigned(value.max_groups != defaults.max_groups) << 8);
}
inline void farming_storage_write(std::ostream& out, FarmingDestination const& value, unsigned mask)
{
    out << mask;
    if (mask & 1) out << ' ' << value.known_device;
    if (mask & 2) out << ' ' << value.nonrotating;
    if (mask & 4) out << ' ' << value.read_bytes_per_second;
    if (mask & 8) out << ' ' << value.read_latency_seconds;
    if (mask & 16) out << ' ' << value.read_concurrency;
    if (mask & 32) out << ' ' << value.capacity_bytes;
    if (mask & 64) out << ' ' << value.max_file_bytes;
    if (mask & 128) out << ' ' << value.max_total_members;
    if (mask & 256) out << ' ' << value.max_groups;
    out << '\n';
}
inline FarmingDestination farming_storage_read(std::istream& in, FarmingDestination value = {}, bool complete = false)
{
    unsigned mask = 0; in >> mask;
    require(in && mask <= farming_storage_fields && (!complete || mask == farming_storage_fields),
        "invalid saved farming storage overrides");
    if (mask & 1) in >> value.known_device;
    if (mask & 2) in >> value.nonrotating;
    if (mask & 4) in >> value.read_bytes_per_second;
    if (mask & 8) in >> value.read_latency_seconds;
    if (mask & 16) in >> value.read_concurrency;
    if (mask & 32) in >> value.capacity_bytes;
    if (mask & 64) in >> value.max_file_bytes;
    if (mask & 128) in >> value.max_total_members;
    if (mask & 256) in >> value.max_groups;
    require(bool(in), "incomplete saved farming storage overrides");
    return value;
}
inline std::string farming_state_text(FarmingState const& state, bool keyless = false)
{
    require(state.version >= 3 && state.version <= 8, "unsupported saved farming plan version");
    require(state.profile_json.size() <= farming_profile_metadata_bytes,
        "saved farming profile metadata exceeds its bound");
    require(state.version < 5 || state.profile_json.empty(), "saved farming profile duplicates its configuration");
    if (state.profile_metadata)
    {
        require(state.version >= 6 || !state.profile_metadata->reader_calibration,
            "scoped reader calibration requires farming plan v6");
        validate_reader_calibration_scope(*state.profile_metadata, state.config);
    }
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<long double>::max_digits10);
    out << "ordinary-group-farming-v" << state.version << '\n' << std::quoted(state.producer_build) << '\n'
        << std::quoted(farming_state_model(state)) << '\n';
    if (state.version >= 6) out << std::quoted(farming_state_keys(state)) << ' '
        << std::numeric_limits<long double>::radix << ' ' << std::numeric_limits<long double>::digits << ' '
        << std::numeric_limits<long double>::max_exponent << '\n';
    out << std::quoted(keyless ? std::string(64, '0') : state.fingerprint) << '\n';
    out << std::quoted(state.job_id) << ' ' << state.revision << ' ' << std::quoted(state.parent_digest) << '\n'
        << std::quoted(keyless ? std::string(64, '0') : state.seed) << ' ' << std::quoted(keyless ? std::string{} : state.seed_owner)
        << ' ' << std::quoted(keyless ? std::string(96, '0') : state.farmer) << ' '
        << std::quoted(keyless ? std::string(state.pool_public_key ? 96 : 64, '0') : state.pool)
        << ' ' << state.pool_public_key << ' ' << state.fixed_extent << '\n'
        << state.capacity << ' ' << state.headroom << ' ' << state.max_groups << ' ' << state.max_total_members << '\n';
    if (state.version == 4) out << state.capacity_fill << ' ' << std::quoted(state.profile_json) << '\n';
    if (state.version >= 5)
    {
        out << state.capacity_fill << ' ' << state.profile_metadata.has_value() << '\n';
        if (state.profile_metadata) out << farming_profile_metadata_text(*state.profile_metadata, state.version);
    }
    out << state.manifest_reserved_bytes << '\n';
    farming_config_write(out, state.config);
    if (state.version >= 7) out << unsigned(state.identity_k) << ' ' << unsigned(state.identity_strength) << ' '
        << std::quoted(state.applied_update) << ' ' << state.policy_revision << '\n';
    if (state.version >= 8) out << std::quoted(state.control) << ' ' << state.request_floor << ' '
        << state.update_reserved_bytes << ' ' << state.progress_reserved_bytes << '\n';
    auto const& e = state.estimate;
    out << e.feasible << ' ' << e.index_bytes << ' ' << e.reader_memory_bytes << ' ' << e.planning_memory_bytes << ' '
        << e.candidates_evaluated << ' ' << e.cpu_utilization << ' ' << e.worker_utilization << '\n'
        << std::quoted(e.reason) << ' ' << e.response_bound_seconds << ' ' << e.dynamic_work_seconds_per_sp << ' '
        << e.cpu_work_seconds_per_sp << ' ' << e.io_work_seconds_per_sp << ' '
        << e.max_eligible_families_per_resource << '\n' << state.directories.size() << '\n';
    FarmingDestination const defaults = state.directories.empty() ? FarmingDestination{} : state.directories.front().storage;
    if (state.version >= 5) farming_storage_write(out, defaults, farming_storage_fields);
    for (auto const& d : state.directories)
    {
        auto const& p = d.layout;
        out << std::quoted(d.directory) << ' ' << std::quoted(d.filesystem) << ' ' << d.capacity_bytes << ' ' << d.resource << '\n'
            << std::quoted(d.directory_identity) << ' ' << d.identity_resources << ' ' << d.identity_resource << ' '
            << d.identity_epoch << ' ' << p.schedule << '\n'
            << p.total_members << ' ' << p.files << ' ' << p.families << ' ' << p.group_bytes << ' '
            << p.group_members << ' ' << p.meta_slots << ' ' << unsigned(p.meta_bits) << ' '
            << unsigned(p.meta_base) << ' ' << p.fixed_members << ' ' << p.resource_rank << ' ' << p.resource_lanes << '\n';
        if (state.version >= 5) farming_storage_write(out, d.storage, farming_storage_overrides(d.storage, defaults));
        else out << d.storage.known_device << ' ' << d.storage.nonrotating << ' ' << d.storage.read_bytes_per_second << ' '
            << d.storage.read_latency_seconds << ' ' << d.storage.read_concurrency << '\n'
            << d.storage.capacity_bytes << ' ' << d.storage.max_file_bytes << ' ' << d.storage.max_total_members << ' '
            << d.storage.max_groups << '\n';
        if (state.version >= 4)
        {
            size_t runs = 0;
            for (size_t file = 0; file < d.extents.size(); ++file)
                if (!file || d.extents[file] != d.extents[file - 1]) ++runs;
            out << d.sealed_files << ' ' << std::quoted(d.fill_stop) << ' ' << d.extents.size() << ' ' << runs;
            for (size_t first = 0; first < d.extents.size();)
            {
                size_t last = first + 1;
                while (last < d.extents.size() && d.extents[last] == d.extents[first]) ++last;
                out << ' ' << d.extents[first] << ' ' << last - first; first = last;
            }
            out << '\n';
        }
        if (state.version >= 7)
        {
            out << d.layout_first << ' ' << d.policy_files << ' ' << d.cohorts.size() << '\n';
            for (auto const& cohort : d.cohorts)
            {
                out << cohort.first_file << ' ' << cohort.identity_resources << ' ' << cohort.identity_resource << ' '
                    << cohort.identity_epoch << '\n';
                farming_config_write(out, cohort.config);
                auto const& p = cohort.layout;
                out << p.schedule << ' ' << p.total_members << ' ' << p.files << ' ' << p.families << ' ' << p.group_bytes << ' '
                    << p.group_members << ' ' << p.meta_slots << ' ' << unsigned(p.meta_bits) << ' ' << unsigned(p.meta_base) << ' '
                    << p.fixed_members << ' ' << p.resource_rank << ' ' << p.resource_lanes << '\n';
            }
        }
        if (state.version >= 8) out << std::quoted(d.control) << '\n';
    }
    return out.str();
}
inline std::filesystem::path stage_farming_state(std::filesystem::path const& path, std::string const& contents, bool failpoints = true)
{
    auto const pending = file_io::native_path(file_io::path_text(path) + ".next");
    require(!std::filesystem::exists(pending), "pending farming revision must be recovered first");
    stream_state::atomic_save(pending, contents, false);
    if (failpoints) grow_failpoint("farm-plan-before-publication");
    return pending;
}
inline std::filesystem::path stage_farming_state(std::filesystem::path const& path, FarmingState const& state)
{
    auto contents = farming_state_text(state);
    auto const digest = stream_state::digest(contents);
    contents += digest + '\n';
    return stage_farming_state(path, contents);
}
inline void publish_farming_state(std::filesystem::path const& pending, std::filesystem::path const& path, bool replace)
{
    // The controller holds all member directory locks, checks the parent
    // revision, and stages every copy before replacing any published copy.
    require(!file_io::rename_file(pending, path, replace), "cannot publish farming plan");
    stream_state::sync_path(path.parent_path(), true);
    grow_failpoint("farm-plan-published");
}
struct FarmingStateFile
{
    std::string body, digest;
    stream_state::Stamp stamp;
};
inline FarmingStateFile read_farming_state(std::filesystem::path const& path, uint64_t memory_limit)
{
    auto const stamp = stream_state::stamp(path);
    check_farming_state_size(stamp.bytes, memory_limit);
    auto file = group_tools::input(file_io::path_text(path));
    auto contents = stream_state::read_exact(file, stamp.bytes);
    require(contents.size() >= 65 && contents.back() == '\n', "invalid farming plan checksum");
    auto const digest = contents.substr(contents.size() - 65, 64);
    contents.resize(contents.size() - 65);
    require(digest == stream_state::digest(contents), "farming plan checksum mismatch");
    require(stream_state::stamp(path) == stamp, "farming plan changed while loading");
    return {std::move(contents), digest, stamp};
}
inline FarmingState parse_farming_state(std::string const& body, uint64_t memory_limit = UINT64_MAX,
    FarmingStateLoad mode = FarmingStateLoad::recovery)
{
    require(body.size() < memory_limit / 4, "saved farming state exceeds its parsing memory budget");
    std::istringstream in(body);
    std::string version, build, model;
    std::getline(in, version); in >> std::quoted(build);
    require(version == "ordinary-group-farming-v3" || version == "ordinary-group-farming-v4" ||
        version == "ordinary-group-farming-v5" || version == "ordinary-group-farming-v6" || version == "ordinary-group-farming-v7" ||
        version == "ordinary-group-farming-v8",
        "unsupported saved farming plan version");
    FarmingState state;
    state.version = version == "ordinary-group-farming-v3" ? 3 : version == "ordinary-group-farming-v4" ? 4 :
        version == "ordinary-group-farming-v5" ? 5 : version == "ordinary-group-farming-v6" ? 6 :
        version == "ordinary-group-farming-v7" ? 7 : 8;
    require(!build.empty() && build.size() <= 128, "invalid saved farming producer build");
    require(state.version >= 6 || mode == FarmingStateLoad::metadata_only || build == GROW_IDENTITY,
        "farming plan requires its original grow build; plan --metadata-only can inspect its saved metadata");
    state.producer_build = std::move(build);
    in >> std::quoted(model); require(model == farming_state_model(state), "saved farming model differs from this build");
    if (state.version >= 6)
    {
        std::string keys; int radix = 0, digits = 0, exponent = 0;
        in >> std::quoted(keys);
        require(in && keys == farming_state_keys(state), "saved farming key derivation differs from this build");
        in >> radix >> digits >> exponent;
        require(in && radix == std::numeric_limits<long double>::radix && digits == std::numeric_limits<long double>::digits &&
            exponent == std::numeric_limits<long double>::max_exponent,
            "saved farming numeric representation is unsupported by this build; values were not loaded");
    }
    in >> std::quoted(state.fingerprint) >> std::quoted(state.job_id) >> state.revision >> std::quoted(state.parent_digest)
        >> std::quoted(state.seed) >> std::quoted(state.seed_owner) >> std::quoted(state.farmer) >> std::quoted(state.pool)
        >> state.pool_public_key >> state.fixed_extent >> state.capacity >> state.headroom >> state.max_groups >> state.max_total_members;
    if (state.version == 4)
    {
        in >> state.capacity_fill >> std::quoted(state.profile_json);
        require(in && state.profile_json.size() <= farming_profile_metadata_bytes, "saved farming profile metadata exceeds its bound");
    }
    if (state.version >= 5)
    {
        bool metadata = false; in >> state.capacity_fill >> metadata;
        require(bool(in), "invalid saved farming profile metadata");
        if (metadata) state.profile_metadata = farming_profile_metadata_read(in, state.version);
    }
    in >> state.manifest_reserved_bytes;
    auto encoded = [](std::string const& value, size_t bytes)
    { return value.size() == bytes * 2 && value.find_first_not_of("0123456789abcdefABCDEF") == value.npos; };
    require(in && encoded(state.job_id, 32) && encoded(state.fingerprint, 32) && state.revision &&
        (state.revision == 1 ? state.parent_digest.empty() : encoded(state.parent_digest, 32)) &&
        encoded(state.seed, 32) && encoded(state.farmer, 48) && encoded(state.pool, state.pool_public_key ? 48 : 32) &&
        state.seed_owner.size() <= 128 && state.max_groups && state.max_total_members && (!state.capacity_fill || state.fixed_extent),
        "invalid saved farming identity or limits");
    state.config = farming_config_read(in);
    state.identity_k = state.config.k; state.identity_strength = state.config.strength;
    if (state.version >= 7)
    {
        unsigned k = 0, strength = 0;
        in >> k >> strength >> std::quoted(state.applied_update) >> state.policy_revision;
        require(in && k >= 18 && k <= 28 && !(k & 1) && strength >= 2 && strength <= std::min(17u, k - 3) &&
            (state.applied_update.empty() || encoded(state.applied_update, 32)) && state.capacity_fill &&
            state.policy_revision && state.policy_revision <= state.revision,
            "invalid saved farming policy identity");
        state.identity_k = uint8_t(k); state.identity_strength = uint8_t(strength);
    }
    if (state.version >= 8)
    {
        in >> std::quoted(state.control) >> state.request_floor >> state.update_reserved_bytes >> state.progress_reserved_bytes;
        require(in && farming_control_valid(state.control) && state.update_reserved_bytes <= state.manifest_reserved_bytes &&
            state.progress_reserved_bytes <= state.manifest_reserved_bytes - state.update_reserved_bytes,
            "invalid saved farming job control or metadata reservation");
    }
    if (state.profile_metadata) validate_reader_calibration_scope(*state.profile_metadata, state.config);
    auto& e = state.estimate;
    auto real = [&]
    {
        std::string value; in >> value; size_t used = 0;
        auto const result = std::stold(value, &used);
        require(used == value.size(), "invalid saved farming estimate"); return result;
    };
    in >> e.feasible >> e.index_bytes >> e.reader_memory_bytes >> e.planning_memory_bytes >> e.candidates_evaluated;
    e.cpu_utilization = real(); e.worker_utilization = real();
    in >> std::quoted(e.reason); e.response_bound_seconds = real(); e.dynamic_work_seconds_per_sp = real();
    e.cpu_work_seconds_per_sp = real(); e.io_work_seconds_per_sp = real();
    in >> e.max_eligible_families_per_resource;
    size_t count = 0; in >> count;
    require(in && count && count <= 1024, "invalid saved farming destination count");
    uint64_t expanded = grow_add(sizeof(FarmingState), grow_add(body.size() * 4, count * sizeof(FarmingDirectory)));
    if (state.profile_metadata) expanded = grow_add(expanded, farming_profile_metadata_memory(*state.profile_metadata));
    require(expanded < memory_limit, "saved farming state exceeds its parsing memory budget");
    state.directories.resize(count);
    FarmingDestination const defaults = state.version >= 5 ? farming_storage_read(in, {}, true) : FarmingDestination{};
    uint64_t extent_count = 0;
    for (auto& d : state.directories)
    {
        auto& p = d.layout; unsigned members, slots, bits, meta, rank, lanes;
        in >> std::quoted(d.directory) >> std::quoted(d.filesystem) >> d.capacity_bytes >> d.resource >>
            std::quoted(d.directory_identity) >> d.identity_resources >> d.identity_resource >> d.identity_epoch >> p.schedule >>
            p.total_members >> p.files >> p.families >> p.group_bytes >> members >> slots >> bits >> meta >> p.fixed_members >> rank >> lanes;
        if (state.version >= 5) d.storage = farming_storage_read(in, defaults);
        else in >> d.storage.known_device >> d.storage.nonrotating >>
            d.storage.read_bytes_per_second >> d.storage.read_latency_seconds >> d.storage.read_concurrency >>
            d.storage.capacity_bytes >> d.storage.max_file_bytes >> d.storage.max_total_members >> d.storage.max_groups;
        require(in && !d.directory.empty() && members <= 65535 && slots && slots <= 256 &&
            bits <= 8 && meta <= 255 && d.resource < count && lanes && lanes <= count && rank < lanes &&
            !d.directory_identity.empty() && d.identity_resources && d.identity_resources <= count &&
            d.identity_resource < d.identity_resources && d.identity_epoch < state.revision &&
            ((p.files && p.total_members && p.families) || (!p.files && !p.total_members && !p.families)),
            "invalid saved farming layout");
        p.group_members = uint16_t(members); p.meta_slots = uint16_t(slots);
        p.meta_bits = uint8_t(bits); p.meta_base = uint8_t(meta); p.resource_rank = uint16_t(rank); p.resource_lanes = uint16_t(lanes);
        d.storage.resource = d.resource;
        if (state.version >= 4)
        {
            size_t files = 0, runs = 0; in >> d.sealed_files >> std::quoted(d.fill_stop) >> files >> runs;
            require(in && files <= farming_detail::maximum_files - extent_count &&
                (state.version >= 8 || files <= state.max_groups) && runs <= files &&
                bool(runs) == bool(files) &&
                d.sealed_files <= files && (state.capacity_fill || (!files && !d.sealed_files && d.fill_stop.empty())) &&
                (d.fill_stop.empty() || d.fill_stop == "capacity" || d.fill_stop == "farming-limit" ||
                 d.fill_stop == "group-limit" || d.fill_stop == "total-member-limit" || d.fill_stop == "metadata-capacity"),
                "invalid saved farming extents");
            uint64_t const extent_bytes = files * sizeof(uint16_t);
            require(extent_bytes <= memory_limit - expanded, "saved farming extents exceed their parsing memory budget");
            expanded = grow_add(expanded, extent_bytes);
            extent_count += files; d.extents.reserve(files);
            uint64_t total = 0;
            for (size_t run = 0; run < runs; ++run)
            {
                unsigned extent = 0; size_t length = 0; in >> extent >> length;
                require(in && extent && extent <= UINT16_MAX &&
                    (state.version >= 7 || !state.config.group_members_override || extent <= state.config.group_members_override) && length &&
                    length <= files - d.extents.size(),
                    "invalid saved farming member extent");
                for (size_t member = 0; member < length; ++member)
                {
                    auto const file = d.extents.size();
                    require(state.version >= 7 || file + 1 >= p.files || extent == p.file_members(file) ||
                        (!d.fill_stop.empty() && file + 1 == files && extent < p.file_members(file)),
                        "saved farming extent changed an initial assignment");
                    d.extents.push_back(uint16_t(extent)); farming_detail::add(total, extent);
                }
            }
            require(d.extents.size() == files && (state.version >= 8 || total <= state.max_total_members),
                "saved farming extents exceed their total member limit");
            require(!state.capacity_fill || (!d.fill_stop.empty() ? d.sealed_files == files : state.version >= 7 || files >= p.files),
                "saved farming extent omitted an initial assignment");
        }
        if (state.version >= 7)
        {
            size_t cohorts = 0;
            in >> d.layout_first >> d.policy_files >> cohorts;
            require(in && cohorts <= 1024 && d.layout_first <= d.policy_files && d.policy_files <= d.extents.size() &&
                (cohorts != 0 || d.layout_first == 0) && (!d.fill_stop.empty() || farming_layout_files(d) <= d.extents.size()),
                "invalid saved farming cohort boundaries");
            auto const bytes = cohorts * sizeof(FarmingCohort);
            require(bytes < memory_limit - expanded, "saved farming cohorts exceed their parsing memory budget");
            expanded = grow_add(expanded, bytes); d.cohorts.reserve(cohorts);
            for (size_t i = 0; i < cohorts; ++i)
            {
                FarmingCohort cohort;
                in >> cohort.first_file >> cohort.identity_resources >> cohort.identity_resource >> cohort.identity_epoch;
                cohort.config = farming_config_read(in);
                auto& p = cohort.layout;
                in >> p.schedule >> p.total_members >> p.files >> p.families >> p.group_bytes >> members >> slots >> bits >> meta
                    >> p.fixed_members >> rank >> lanes;
                require(in && members <= UINT16_MAX && slots && slots <= 256 && bits <= 8 && meta <= 255 && lanes && lanes <= count && rank < lanes &&
                    cohort.identity_resources && cohort.identity_resources <= count && cohort.identity_resource < cohort.identity_resources &&
                    cohort.identity_epoch < state.revision && cohort.first_file <= d.layout_first &&
                    (!i ? cohort.first_file == 0 : cohort.first_file >= d.cohorts.back().first_file),
                    "invalid saved farming cohort identity");
                p.group_members = uint16_t(members); p.meta_slots = uint16_t(slots); p.meta_bits = uint8_t(bits);
                p.meta_base = uint8_t(meta); p.resource_rank = uint16_t(rank); p.resource_lanes = uint16_t(lanes);
                farming_detail::validate(cohort.config, {});
                d.cohorts.push_back(std::move(cohort));
            }
            for (uint64_t file = d.policy_files; file < d.extents.size(); ++file)
                require(!state.config.group_members_override || d.extents[size_t(file)] <= state.config.group_members_override,
                    "future farming extent exceeds its revised group limit");
        }
        if (state.version >= 8)
        {
            in >> std::quoted(d.control);
            require(in && farming_control_valid(d.control), "invalid saved farming destination control");
        }
        state.estimate.destinations.push_back(p);
    }
    in >> std::ws; require(in.eof(), "unexpected saved farming plan data");
    return state;
}
inline FarmingState load_farming_state(std::filesystem::path const& path, uint64_t memory_limit,
    FarmingStateLoad mode = FarmingStateLoad::recovery)
{
    auto const file = read_farming_state(path, memory_limit);
    auto state = parse_farming_state(file.body, memory_limit, mode);
    require(stream_state::stamp(path) == file.stamp, "farming plan changed while loading");
    return state;
}
} // namespace group_grow
