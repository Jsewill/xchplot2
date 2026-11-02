// Immutable directory farming assignments. Progress remains in each group's
// existing durable construction protocol; this checkpoint contains no seed.
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
};
struct FarmingState
{
    FarmingConfig config;
    uint64_t manifest_reserved_bytes = 0;
    std::string fingerprint;
    FarmingPlan estimate;
    std::vector<FarmingDirectory> directories;
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
    out << "ordinary-group-farming-v2\n" << std::quoted(GROW_IDENTITY) << '\n'
        << std::quoted(farming_model_identity) << '\n' << std::quoted(state.fingerprint) << '\n';
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
            << p.total_members << ' ' << p.files << ' ' << p.families << ' ' << p.group_bytes << ' '
            << p.group_members << ' ' << p.meta_slots << ' ' << unsigned(p.meta_bits) << ' '
            << unsigned(p.meta_base) << ' ' << p.fixed_members << ' ' << p.resource_rank << ' ' << p.resource_lanes << '\n'
            << d.storage.known_device << ' ' << d.storage.nonrotating << ' ' << d.storage.read_bytes_per_second << ' '
            << d.storage.read_latency_seconds << ' ' << d.storage.read_concurrency << '\n';
    }
    return out.str();
}
inline std::filesystem::path stage_farming_state(std::filesystem::path const& path, FarmingState const& state)
{
    auto const text = farming_state_text(state);
    std::string temporary = file_io::path_text(path) + ".pending-XXXXXX";
    stream_state::Descriptor owned(file_io::temp_file(temporary.data()));
    try
    {
        // The private name belongs to this invocation. Final publication below
        // uses the same exclusive hard-link operation as initial checkpoints.
        stream_state::atomic_save(file_io::native_path(temporary), text + stream_state::digest(text) + '\n');
        grow_failpoint("farm-plan-before-publication");
        return file_io::native_path(temporary);
    }
    catch (...)
    {
        (void)file_io::unlink_file(temporary); throw;
    }
}
inline void publish_farming_state(std::filesystem::path const& pending, std::filesystem::path const& path)
{
#ifdef _WIN32
    require(!file_io::rename_file(pending, path, false), "cannot publish farming plan");
    stream_state::sync_path(path.parent_path(), true);
    grow_failpoint("farm-plan-published");
#else
    std::filesystem::create_hard_link(pending, path);
    stream_state::sync_path(path.parent_path(), true);
    grow_failpoint("farm-plan-published");
    require(std::filesystem::remove(pending), "cannot retire farming-plan publication file");
    stream_state::sync_path(path.parent_path(), true);
#endif
}
inline FarmingState load_farming_state(std::filesystem::path const& path, uint64_t memory_limit)
{
    auto const stamp = stream_state::stamp(path);
    require(stamp.bytes < memory_limit / 16, "saved farming plan exceeds --max-ram");
    auto file = group_tools::input(file_io::path_text(path));
    std::string contents{std::istreambuf_iterator<char>(file), {}};
    require(contents.size() >= 65 && contents.back() == '\n', "invalid farming plan checksum");
    auto const body = contents.substr(0, contents.size() - 65);
    require(contents.substr(contents.size() - 65, 64) == stream_state::digest(body), "farming plan checksum mismatch");
    std::istringstream in(body);
    std::string version, build, model;
    std::getline(in, version); in >> std::quoted(build);
    require(version == "ordinary-group-farming-v2" && build == GROW_IDENTITY,
        "farming plan requires its original grow build");
    FarmingState state;
    in >> std::quoted(model); require(model == farming_model_identity, "saved farming model differs from this build");
    in >> std::quoted(state.fingerprint) >> state.manifest_reserved_bytes; state.config = farming_config_read(in);
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
            p.total_members >> p.files >> p.families >> p.group_bytes >> members >> slots >> bits >> meta >> p.fixed_members >> rank >> lanes >> d.storage.known_device >> d.storage.nonrotating >>
            d.storage.read_bytes_per_second >> d.storage.read_latency_seconds >> d.storage.read_concurrency;
        require(in && !d.directory.empty() && members <= 65535 && slots && slots <= 256 &&
            bits <= 8 && meta <= 255 && d.resource < count && lanes && lanes <= count && rank < lanes &&
            ((p.files && p.total_members && p.families) || (!p.files && !p.total_members && !p.families)),
            "invalid saved farming layout");
        p.group_members = uint16_t(members); p.meta_slots = uint16_t(slots);
        p.meta_bits = uint8_t(bits); p.meta_base = uint8_t(meta); p.resource_rank = uint16_t(rank); p.resource_lanes = uint16_t(lanes);
        d.storage.resource = d.resource; d.storage.capacity_bytes = d.capacity_bytes;
        state.estimate.destinations.push_back(p);
    }
    in >> std::ws; require(in.eof(), "unexpected saved farming plan data");
    require(stream_state::stamp(path) == stamp, "farming plan changed while loading");
    return state;
}
} // namespace group_grow
