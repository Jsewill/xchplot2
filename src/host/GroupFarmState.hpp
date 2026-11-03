// Private directory farming jobs. Progress remains in each group's durable
// construction protocol; replicated job revisions retain keys and assignments.
#pragma once
#include "host/GroupEngine.hpp"
#include "host/GroupFarming.hpp"
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
struct FarmingDirectory
{
    std::string directory, filesystem;
    uint64_t capacity_bytes = 0, resource = 0;
    FarmingDrivePlan layout;
    FarmingDestination storage;
    std::string directory_identity;
    uint64_t identity_resources = 1, identity_resource = 0, identity_epoch = 0;
};
struct FarmingState
{
    FarmingConfig config;
    uint64_t manifest_reserved_bytes = 0;
    std::string fingerprint;
    FarmingPlan estimate;
    std::vector<FarmingDirectory> directories;
    std::string job_id, parent_digest;
    uint64_t revision = 1;
    std::string seed, seed_owner, farmer, pool;
    bool pool_public_key = false;
    bool fixed_extent = false;
    uint64_t capacity = 0, headroom = 64 * 1024 * 1024;
    uint64_t max_groups = UINT64_MAX, max_total_members = UINT64_MAX;
};
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
inline std::string farming_state_text(FarmingState const& state)
{
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<long double>::max_digits10);
    out << "ordinary-group-farming-v3\n" << std::quoted(GROW_IDENTITY) << '\n'
        << std::quoted(farming_model_identity) << '\n' << std::quoted(state.fingerprint) << '\n';
    out << std::quoted(state.job_id) << ' ' << state.revision << ' ' << std::quoted(state.parent_digest) << '\n'
        << std::quoted(state.seed) << ' ' << std::quoted(state.seed_owner) << ' ' << std::quoted(state.farmer) << ' '
        << std::quoted(state.pool) << ' ' << state.pool_public_key << ' ' << state.fixed_extent << '\n'
        << state.capacity << ' ' << state.headroom << ' ' << state.max_groups << ' ' << state.max_total_members << '\n';
    out << state.manifest_reserved_bytes << '\n';
    farming_config_write(out, state.config);
    auto const& e = state.estimate;
    out << e.feasible << ' ' << e.index_bytes << ' ' << e.reader_memory_bytes << ' ' << e.planning_memory_bytes << ' '
        << e.candidates_evaluated << ' ' << e.cpu_utilization << ' ' << e.worker_utilization << '\n'
        << std::quoted(e.reason) << ' ' << e.response_bound_seconds << ' ' << e.dynamic_work_seconds_per_sp << ' '
        << e.cpu_work_seconds_per_sp << ' ' << e.io_work_seconds_per_sp << ' '
        << e.max_eligible_families_per_resource << '\n' << state.directories.size() << '\n';
    for (auto const& d : state.directories)
    {
        auto const& p = d.layout;
        out << std::quoted(d.directory) << ' ' << std::quoted(d.filesystem) << ' ' << d.capacity_bytes << ' ' << d.resource << '\n'
            << std::quoted(d.directory_identity) << ' ' << d.identity_resources << ' ' << d.identity_resource << ' '
            << d.identity_epoch << ' ' << p.schedule << '\n'
            << p.total_members << ' ' << p.files << ' ' << p.families << ' ' << p.group_bytes << ' '
            << p.group_members << ' ' << p.meta_slots << ' ' << unsigned(p.meta_bits) << ' '
            << unsigned(p.meta_base) << ' ' << p.fixed_members << ' ' << p.resource_rank << ' ' << p.resource_lanes << '\n'
            << d.storage.known_device << ' ' << d.storage.nonrotating << ' ' << d.storage.read_bytes_per_second << ' '
            << d.storage.read_latency_seconds << ' ' << d.storage.read_concurrency << '\n'
            << d.storage.capacity_bytes << ' ' << d.storage.max_file_bytes << ' ' << d.storage.max_total_members << ' '
            << d.storage.max_groups << '\n';
    }
    return out.str();
}
inline std::filesystem::path stage_farming_state(std::filesystem::path const& path, std::string const& contents)
{
    auto const pending = file_io::native_path(file_io::path_text(path) + ".next");
    require(!std::filesystem::exists(pending), "pending farming revision must be recovered first");
    stream_state::atomic_save(pending, contents, false);
    grow_failpoint("farm-plan-before-publication");
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
    require(stamp.bytes < memory_limit / 16, "saved farming plan exceeds --max-ram");
    auto file = group_tools::input(file_io::path_text(path));
    std::string contents{std::istreambuf_iterator<char>(file), {}};
    require(contents.size() >= 65 && contents.back() == '\n', "invalid farming plan checksum");
    auto const digest = contents.substr(contents.size() - 65, 64);
    contents.resize(contents.size() - 65);
    require(digest == stream_state::digest(contents), "farming plan checksum mismatch");
    require(stream_state::stamp(path) == stamp, "farming plan changed while loading");
    return {std::move(contents), digest, stamp};
}
inline FarmingState parse_farming_state(std::string const& body)
{
    std::istringstream in(body);
    std::string version, build, model;
    std::getline(in, version); in >> std::quoted(build);
    require(version == "ordinary-group-farming-v3" && build == GROW_IDENTITY,
        "farming plan requires its original grow build");
    FarmingState state;
    in >> std::quoted(model); require(model == farming_model_identity, "saved farming model differs from this build");
    in >> std::quoted(state.fingerprint) >> std::quoted(state.job_id) >> state.revision >> std::quoted(state.parent_digest)
        >> std::quoted(state.seed) >> std::quoted(state.seed_owner) >> std::quoted(state.farmer) >> std::quoted(state.pool)
        >> state.pool_public_key >> state.fixed_extent >> state.capacity >> state.headroom >> state.max_groups >> state.max_total_members
        >> state.manifest_reserved_bytes;
    auto encoded = [](std::string const& value, size_t bytes)
    { return value.size() == bytes * 2 && value.find_first_not_of("0123456789abcdefABCDEF") == value.npos; };
    require(in && encoded(state.job_id, 32) && encoded(state.fingerprint, 32) && state.revision &&
        (state.revision == 1 ? state.parent_digest.empty() : encoded(state.parent_digest, 32)) &&
        encoded(state.seed, 32) && encoded(state.farmer, 48) && encoded(state.pool, state.pool_public_key ? 48 : 32) &&
        state.seed_owner.size() <= 128 && state.max_groups && state.max_total_members,
        "invalid saved farming identity or limits");
    state.config = farming_config_read(in);
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
    state.directories.resize(count);
    for (auto& d : state.directories)
    {
        auto& p = d.layout; unsigned members, slots, bits, meta, rank, lanes;
        in >> std::quoted(d.directory) >> std::quoted(d.filesystem) >> d.capacity_bytes >> d.resource >>
            std::quoted(d.directory_identity) >> d.identity_resources >> d.identity_resource >> d.identity_epoch >> p.schedule >>
            p.total_members >> p.files >> p.families >> p.group_bytes >> members >> slots >> bits >> meta >> p.fixed_members >> rank >> lanes >> d.storage.known_device >> d.storage.nonrotating >>
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
        state.estimate.destinations.push_back(p);
    }
    in >> std::ws; require(in.eof(), "unexpected saved farming plan data");
    return state;
}
inline FarmingState load_farming_state(std::filesystem::path const& path, uint64_t memory_limit)
{
    auto const file = read_farming_state(path, memory_limit);
    auto state = parse_farming_state(file.body);
    require(stream_state::stamp(path) == file.stamp, "farming plan changed while loading");
    return state;
}
} // namespace group_grow
