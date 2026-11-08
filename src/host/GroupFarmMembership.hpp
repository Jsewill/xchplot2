// Durable membership is narrower than the historical allocation obligation.
#pragma once
#include "host/GroupFarmState.hpp"

namespace group_grow
{
inline bool farming_legacy_retirement_inputs(FarmingState const& before, FarmingState const& current)
{
    if (before.version != 8 || !before.capacity_fill || !before.fixed_extent ||
        before.revision == UINT64_MAX || current.revision != before.revision + 1 ||
        current.parent_digest.size() != 64 || current.parent_digest.find_first_not_of("0123456789abcdefABCDEF") != current.parent_digest.npos ||
        current.producer_build != before.producer_build || current.config != before.config ||
        !same_farming_profile_metadata(before, current) || current.identity_k != before.identity_k ||
        current.identity_strength != before.identity_strength || current.applied_update != before.applied_update ||
        current.control != before.control || current.request_floor != before.request_floor ||
        current.manifest_reserved_bytes != before.manifest_reserved_bytes || current.update_reserved_bytes != before.update_reserved_bytes ||
        current.progress_reserved_bytes != before.progress_reserved_bytes || current.fingerprint != before.fingerprint ||
        current.job_id != before.job_id || current.seed != before.seed || current.seed_owner != before.seed_owner ||
        current.farmer != before.farmer || current.pool != before.pool || current.pool_public_key != before.pool_public_key ||
        current.fixed_extent != before.fixed_extent || current.capacity_fill != before.capacity_fill ||
        current.capacity != before.capacity || current.headroom != before.headroom || current.max_groups != before.max_groups ||
        current.max_total_members != before.max_total_members || current.directories.size() != before.directories.size()) return false;
    for (size_t lane = 0; lane < before.directories.size(); ++lane)
    {
        auto const& a = before.directories[lane]; auto const& b = current.directories[lane];
        if (!farming_member(a) || !farming_member(b) || a.directory != b.directory || a.filesystem != b.filesystem ||
            a.capacity_bytes != b.capacity_bytes || a.resource != b.resource || a.layout != b.layout || a.storage != b.storage ||
            a.directory_identity != b.directory_identity || a.identity_resources != b.identity_resources ||
            a.identity_resource != b.identity_resource || a.identity_epoch != b.identity_epoch || a.layout_first != b.layout_first ||
            a.cohorts != b.cohorts || a.control != b.control || b.detached_at || b.detached_source_revision ||
            !b.detached_source_digest.empty() || b.detached_acknowledged || b.detached_reclaimed ||
            b.policy_files < a.policy_files || b.policy_files > a.extents.size() || b.policy_files > b.extents.size() ||
            b.extents.size() > a.extents.size() || b.sealed_files < a.sealed_files || b.sealed_files > b.extents.size() ||
            (!b.fill_stop.empty() && b.sealed_files != b.extents.size()) ||
            (b.extents.size() < a.extents.size() && b.fill_stop.empty()) ||
            (!a.fill_stop.empty() && (a.fill_stop != b.fill_stop || a.extents != b.extents || a.sealed_files != b.sealed_files))) return false;
        if (!b.fill_stop.empty() && b.fill_stop != "capacity" && b.fill_stop != "farming-limit" && b.fill_stop != "group-limit" &&
            b.fill_stop != "total-member-limit" && b.fill_stop != "metadata-capacity") return false;
        for (size_t file = 0; file < b.extents.size(); ++file)
            if (!b.extents[file] || b.extents[file] > a.extents[file] ||
                ((file < a.sealed_files || file >= b.sealed_files) && b.extents[file] != a.extents[file])) return false;
    }
    return true;
}
inline bool valid_farming_retirement_upgrade(FarmingState const& before, FarmingState const& current)
{
    if (!farming_legacy_retirement_inputs(before, current) || current.version != 9 ||
        current.policy_revision != current.revision || current.owner_reserved_bytes || !current.shared_scratch_possible ||
        current.conservative_storage_topology || current.ancestors.size() != 1 ||
        current.ancestors.front().first != before.revision || current.ancestors.front().second != current.parent_digest) return false;
    for (size_t lane = 0; lane < before.directories.size(); ++lane)
        if (current.directories[lane].grant_extents != before.directories[lane].extents ||
            current.directories[lane].grant_config != std::optional<FarmingConfig>{before.config}) return false;
    return true;
}
inline void initialize_farming_retirement_upgrade(FarmingState const& before, FarmingState& current)
{
    require(current.version == 8 && farming_legacy_retirement_inputs(before, current),
        "legacy retirement upgrade changed an original storage, identity or farming grant; retirement was not applied");
    current.version = 9; current.policy_revision = current.revision;
    current.owner_reserved_bytes = 0; current.shared_scratch_possible = true; current.conservative_storage_topology = false;
    current.ancestors = {{before.revision, current.parent_digest}};
    current.estimate = before.estimate;
    for (size_t lane = 0; lane < current.directories.size(); ++lane)
    {
        current.directories[lane].grant_extents = before.directories[lane].extents;
        current.directories[lane].grant_config = before.config;
    }
    require(valid_farming_retirement_upgrade(before, current), "legacy retirement upgrade exceeded its original grants; retirement was not applied");
}
inline void initialize_farming_grants(FarmingState& state, size_t first = 0)
{
    require(state.version >= 9 && state.capacity_fill, "independent farming grants require a current capacity-fill job");
    for (size_t lane = first; lane < state.directories.size(); ++lane)
    {
        auto& directory = state.directories[lane];
        if (!farming_member(directory)) continue;
        directory.grant_extents = directory.extents;
        directory.grant_config = state.config;
    }
}
inline bool farming_grant_covers(FarmingState const& state, size_t lane, uint64_t file, uint16_t members)
{
    if (state.version < 9 || lane >= state.directories.size() || !members) return false;
    auto const& directory = state.directories[lane];
    return farming_member(directory) && file < directory.grant_extents.size() && members <= directory.grant_extents[size_t(file)];
}
inline void farming_record_parent(FarmingState& state, uint64_t revision, std::string const& digest)
{
    require(state.version >= 9 && revision && revision == state.revision - 1 && digest.size() == 64 &&
        digest.find_first_not_of("0123456789abcdefABCDEF") == digest.npos,
        "invalid farming revision parent");
    // Every still-required replica participates in each policy publication.
    // Only detached members can remain older; their exact source anchors are
    // retained with their obligations instead of growing a lifetime log.
    state.ancestors.clear();
    state.ancestors.emplace_back(revision, digest);
}
inline bool farming_detached_replica_matches(FarmingState const& older, FarmingState const& current, size_t lane)
{
    if (older.version < 9 || current.version < 9 || lane >= older.directories.size() || lane >= current.directories.size() ||
        older.job_id != current.job_id || older.fingerprint != current.fingerprint || older.seed != current.seed ||
        older.seed_owner != current.seed_owner || older.farmer != current.farmer || older.pool != current.pool ||
        older.pool_public_key != current.pool_public_key || older.identity_k != current.identity_k ||
        older.identity_strength != current.identity_strength) return false;
    auto const& source = older.directories[lane]; auto const& target = current.directories[lane];
    if (farming_member(source) || farming_member(target) || source.detached_at != target.detached_at ||
        source.detached_source_revision != target.detached_source_revision || source.detached_source_digest != target.detached_source_digest ||
        source.directory_identity != target.directory_identity || source.directory != target.directory ||
        source.resource != target.resource || source.filesystem != target.filesystem) return false;
    // Acknowledged sources are accepted only for native ownership inspection:
    // their old grant never becomes policy or construction authority again.
    if (target.detached_acknowledged) return true;
    return source.capacity_bytes == target.capacity_bytes && source.storage == target.storage && source.layout == target.layout &&
        source.layout_first == target.layout_first && source.cohorts == target.cohorts &&
        source.identity_resources == target.identity_resources && source.identity_resource == target.identity_resource &&
        source.identity_epoch == target.identity_epoch && source.grant_extents == target.grant_extents &&
        source.grant_config == target.grant_config;
}
inline void detach_farming_member(FarmingState& state, size_t lane)
{
    require(state.version >= 9 && lane < state.directories.size(), "detachment requires a current farming member");
    auto& directory = state.directories[lane];
    if (!farming_member(directory)) return;
    require(directory.grant_config.has_value() && directory.grant_extents.size() >= directory.extents.size(),
        "farming member has no retained allocation obligation");
    require(state.revision > 1 && !state.parent_digest.empty(), "farming detachment lacks its acknowledged source revision");
    directory.membership = "detached"; directory.detached_at = state.revision; directory.control = "retired";
    directory.detached_source_revision = state.revision - 1; directory.detached_source_digest = state.parent_digest;
    // No acknowledgement is inferred here. A disconnected old writer still
    // owns every promised byte and member until its own native lock is held.
}
// Storage grants are byte quotas, not an assertion that every possible
// member has the planner's estimated size. The construction controller keeps
// their native capacity and scratch guards. This larger service envelope lets
// an available drive spend compact-output savings within its own byte quota
// without asking unavailable peers to approve a new family or member.
inline FarmingAssessment expand_farming_grants(FarmingState& state, uint64_t memory_limit)
{
    require(state.version >= 9 && state.capacity_fill, "independent farming grants require a current capacity-fill job");
    uint64_t files = 0, retained = state.directories.size() * sizeof(std::vector<uint16_t>);
    std::vector<std::vector<uint16_t>> members; members.reserve(state.directories.size());
    for (auto const& directory : state.directories)
    {
        require(directory.grant_config.has_value() && directory.grant_extents.size() <= farming_detail::maximum_files - files,
            "farming allocation grants exceed their file work bound");
        files += directory.grant_extents.size();
        retained = grow_add(retained, (directory.grant_extents.size() + 1) * sizeof(uint16_t) * 2);
        require(retained < memory_limit, "farming allocation grants exceed their planning memory budget");
        members.push_back(directory.grant_extents);
    }
    auto assess = [&] { return farming_assess_state(state, members, memory_limit - retained); };
    auto accepted = assess();
    require(accepted.feasible, "saved farming allocation obligations exceed the revised harvester limits");
    auto eligible = [&](size_t lane)
    {
        auto const& directory = state.directories[lane];
        return farming_member(directory) && directory.control == "active" && directory.fill_stop.empty() &&
            directory.capacity_bytes > state.manifest_reserved_bytes;
    };
    auto maximum = [&](size_t lane, size_t file)
    {
        auto const& config = *state.directories[lane].grant_config;
        auto const bound = config.group_members_override ? config.group_members_override : UINT16_MAX;
        auto const& grant = members[lane];
        uint64_t total = 0; for (auto value : grant) total = grow_add(total, value);
        return total >= state.max_total_members || grant[file] >= bound ? grant[file] :
            uint16_t(grant[file] + std::min<uint64_t>(bound - grant[file], state.max_total_members - total));
    };
    auto extend = [&](size_t lane, size_t file, uint16_t high)
    {
        auto& grant = members[lane]; auto low = grant[file];
        auto const& config = *state.directories[lane].grant_config;
        while (low < high)
        {
            auto const candidate = uint16_t(unsigned(low) + (unsigned(high) - low + 1) / 2);
            grant[file] = candidate;
            if (!farming_detail::shape(config, candidate)) { high = uint16_t(candidate - 1); continue; }
            auto result = assess();
            if (result.feasible) { low = candidate; accepted = std::move(result); }
            else high = uint16_t(candidate - 1);
        }
        grant[file] = low;
    };
    auto extend_terminal = [&](size_t lane, bool minimum)
    {
        auto const& directory = state.directories[lane];
        if (directory.extents.empty() || directory.extents.size() <= directory.sealed_files ||
            directory.extents.size() <= directory.policy_files) return;
        auto const file = directory.extents.size() - 1;
        auto const high = maximum(lane, file);
        extend(lane, file, minimum ? uint16_t(std::min<unsigned>(high, unsigned(members[lane][file]) + 1)) : high);
    };
    // Reserve a next member before another ordinal so a partial terminal keeps
    // priority under an explicit total-member limit. The forecast's automatic
    // group size is not a member cap; format, shape and joint admission are.
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
        if (eligible(lane)) extend_terminal(lane, true);
    // Admit every feasible lane's next ordinal before maximizing automatic
    // groups, so one lane cannot spend the peers' minimum future rights.
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        if (!eligible(lane)) continue;
        auto const& directory = state.directories[lane]; auto& grant = members[lane];
        uint64_t total = 0; for (auto value : grant) total = grow_add(total, value);
        if (files < farming_detail::maximum_files && grant.size() <= directory.extents.size() && grant.size() < state.max_groups &&
            total < state.max_total_members)
        {
            grant.push_back(1); auto result = assess();
            if (!result.feasible) grant.pop_back();
            else { ++files; accepted = std::move(result); }
        }
    }
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        if (!eligible(lane)) continue;
        extend_terminal(lane, false);
        auto& grant = members[lane];
        if (grant.size() > state.directories[lane].extents.size()) extend(lane, grant.size() - 1, maximum(lane, grant.size() - 1));
    }
    for (size_t lane = 0; lane < state.directories.size(); ++lane) state.directories[lane].grant_extents = std::move(members[lane]);
    return accepted;
}
inline size_t farming_member_count(FarmingState const& state)
{
    return size_t(std::count_if(state.directories.begin(), state.directories.end(),
        [](auto const& directory) { return farming_member(directory); }));
}
inline void acknowledge_farming_detachment(FarmingState& state, size_t lane)
{
    require(state.version >= 9 && lane < state.directories.size() && !farming_member(state.directories[lane]) &&
        state.directories[lane].detached_at < state.revision, "farming retirement must durably detach before acknowledgement");
    // The caller owns the exact native writer lock and has already published
    // its incompatible-schema tombstone. Acknowledgement itself retains the
    // full allocation; only a later policy may reclaim an acknowledged grant.
    state.directories[lane].detached_acknowledged = true;
}
struct FarmingMembershipObligations
{
    size_t detached = 0, unfenced = 0;
    uint64_t bytes = 0, files = 0, members = 0;
};
inline FarmingMembershipObligations farming_membership_obligations(FarmingState const& state)
{
    FarmingMembershipObligations result;
    for (auto const& directory : state.directories) if (!farming_member(directory))
    {
        ++result.detached;
        if (directory.detached_acknowledged) continue;
        ++result.unfenced; result.bytes = grow_add(result.bytes, directory.capacity_bytes);
        result.files = grow_add(result.files, directory.grant_extents.size());
        for (auto members : directory.grant_extents) result.members = grow_add(result.members, members);
    }
    return result;
}
inline char const* farming_independent_policy_reason()
{
    return "an unavailable detached writer retains its storage and farming grants; larger or reshaped shared allocations require "
        "the original writer's durable retirement acknowledgement or a separately reviewed job";
}
inline bool valid_farming_independent_policy_revision(FarmingState const& before, FarmingState const& after)
{
    if (before.version < 9 || after.version < 9 || after.directories.size() < before.directories.size() ||
        before.shared_scratch_possible != after.shared_scratch_possible ||
        before.conservative_storage_topology != after.conservative_storage_topology) return false;
    for (size_t lane = 0; lane < before.directories.size(); ++lane)
    {
        auto const& source = before.directories[lane]; auto const& target = after.directories[lane];
        if (farming_reclaimed_tombstone(source) && source != target) return false;
        auto storage = target.storage;
        bool const reclaim = !farming_member(source) && source.detached_acknowledged;
        if (reclaim)
        {
            if (storage.capacity_bytes > source.storage.capacity_bytes || storage.max_groups > source.storage.max_groups ||
                storage.max_total_members > source.storage.max_total_members) return false;
            storage.capacity_bytes = source.storage.capacity_bytes; storage.max_groups = source.storage.max_groups;
            storage.max_total_members = source.storage.max_total_members;
        }
        if (target.capacity_bytes > source.capacity_bytes || target.grant_config != source.grant_config ||
            target.grant_extents.size() > source.grant_extents.size() || target.resource != source.resource ||
            target.filesystem != source.filesystem || target.directory_identity != source.directory_identity ||
            storage != source.storage) return false;
        for (size_t file = 0; file < target.grant_extents.size(); ++file)
            if (target.grant_extents[file] > source.grant_extents[file] ||
                !same_farming_assignment(before, source, after, target, file)) return false;
    }
    for (size_t lane = before.directories.size(); lane < after.directories.size(); ++lane)
        if (after.directories[lane].capacity_bytes || !after.directories[lane].grant_extents.empty()) return false;
    // A partition may consume only its original delegated storage and service
    // rights. It cannot mint a new namespace or spend unassigned shared work
    // simply by removing the other participants from its replica set.
    return true;
}
} // namespace group_grow
