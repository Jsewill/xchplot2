// Private, replicated migration progress. The ordinary farming state owns all
// replacement keys and assignments from the first durable revision onward.
#pragma once
#include "host/GroupFarmState.hpp"
#include "host/GroupFarmInventory.hpp"
#include "host/GroupFarmMetadata.hpp"
#include <array>
#include <functional>
#include <cstdlib>

namespace group_grow
{
enum class ReplotPhase : uint8_t { preparing, replacing, handoff, complete };
enum class ReplotRetirement : uint8_t { active, claim_intent, claimed, retired };
enum class ReplotVerificationMode : uint8_t { structural, sampled, strict };
struct ReplotVerification
{
    ReplotVerificationMode mode = ReplotVerificationMode::structural;
    uint64_t first_groups = 0, every_groups = 0;
    bool operator==(ReplotVerification const&) const = default;
};
inline char const* replot_verification_name(ReplotVerificationMode mode)
{
    switch (mode)
    {
    case ReplotVerificationMode::structural: return "structural";
    case ReplotVerificationMode::sampled: return "sampled";
    case ReplotVerificationMode::strict: return "strict";
    }
    throw std::runtime_error("invalid migration verification mode");
}
inline void validate_replot_verification(ReplotVerification const& policy)
{
    (void)replot_verification_name(policy.mode);
    require(policy.mode == ReplotVerificationMode::sampled ? policy.first_groups || policy.every_groups :
        !policy.first_groups && !policy.every_groups, "migration proof sampling requires sampled mode and a nonzero group schedule");
}
inline bool replot_requires_proof(ReplotVerification const& policy, uint64_t batch)
{
    validate_replot_verification(policy);
    return policy.mode == ReplotVerificationMode::strict || (policy.mode == ReplotVerificationMode::sampled &&
        (batch < policy.first_groups || (policy.every_groups && batch % policy.every_groups == policy.every_groups - 1)));
}
inline ReplotVerification replot_verification_policy(std::optional<ReplotVerification> const& policy)
{ return policy.value_or(ReplotVerification{ReplotVerificationMode::strict}); }
inline void write_replot_verification(std::ostream& out, ReplotVerification const& policy)
{
    validate_replot_verification(policy);
    out << replot_verification_name(policy.mode) << ' ' << policy.first_groups << ' ' << policy.every_groups << '\n';
}
inline ReplotVerification read_replot_verification(std::istream& in)
{
    ReplotVerification result; std::string mode; in >> mode >> result.first_groups >> result.every_groups;
    require(bool(in), "incomplete migration verification policy");
    if (mode == "structural") result.mode = ReplotVerificationMode::structural;
    else if (mode == "sampled") result.mode = ReplotVerificationMode::sampled;
    else { require(mode == "strict", "invalid migration verification mode"); result.mode = ReplotVerificationMode::strict; }
    validate_replot_verification(result); return result;
}
struct ReplotCandidateProgress
{
    ReplotRetirement state = ReplotRetirement::active;
    bool predelete = false;
    bool operator==(ReplotCandidateProgress const&) const = default;
};
inline constexpr size_t replot_maximum_batch_sources = 1024;
struct ReplotPredeleteCandidate
{
    uint64_t allocated_bytes = 0;
    bool eligible = false;
};
struct ReplotPredeleteSelection
{
    std::vector<size_t> sources;
    bool feasible = false;
    bool search_exhausted = false;
};
inline constexpr size_t replot_predelete_combination_checks = 32;
inline constexpr size_t replot_predelete_fallback_source_visits = 4 * replot_maximum_batch_sources;
inline constexpr size_t replot_predelete_fallback_candidate_visits =
    (replot_maximum_batch_sources + 1) * (replot_predelete_combination_checks + 1);
inline uint64_t replot_predelete_memory_bound(size_t count)
{
    require(count <= replot_maximum_batch_sources, "migration early-deletion memory exceeds its batch limit");
    return count * (sizeof(ReplotPredeleteCandidate) + 4 * sizeof(size_t)) +
        sizeof(std::array<bool, replot_maximum_batch_sources>) + 5 * sizeof(std::vector<size_t>) + sizeof(size_t);
}
// Forecast and runtime supply their own space/resource projections. Only whole
// eligible files can join the reviewed order, within the same outstanding debt.
// Fits must rebuild its trial projection: fallback trials can remove earlier
// optional choices. Mandatory pending sources and their full debt never change.
template<class Candidate, class Fits>
ReplotPredeleteSelection select_replot_predeletions(size_t count, uint64_t allowance, uint64_t outstanding,
    std::vector<size_t> selected, Candidate candidate, Fits fits)
{
    require(count <= replot_maximum_batch_sources && outstanding <= allowance && selected.size() <= count,
        "invalid migration early-deletion selection limits");
    std::array<bool, replot_maximum_batch_sources> included{};
    for (auto source : selected)
    {
        require(source < count && !included[source], "migration source pending early deletion is duplicated or outside its batch");
        included[source] = true;
    }
    selected.reserve(count);
    if (fits(std::span<size_t const>(selected))) return {std::move(selected), true};
    // Zero is replacement-first policy, including files whose native allocated
    // byte count is zero. Previously committed pending rows remain mandatory.
    if (!allowance) return {std::move(selected), false};
    auto const mandatory = selected; auto const mandatory_debt = outstanding;
    std::vector<ReplotPredeleteCandidate> candidates(count);
    for (size_t source = 0; source < count; ++source)
    {
        if (included[source]) continue;
        auto const current = candidates[source] = candidate(source);
        if (!current.eligible || current.allocated_bytes > allowance - outstanding) continue;
        outstanding += current.allocated_bytes; selected.push_back(source);
        if (fits(std::span<size_t const>(selected))) return {std::move(selected), true};
    }
    auto const greedy = std::move(selected); selected = mandatory; selected.reserve(count);
    size_t visits = 0, candidate_visits = 0, combinations = 0;
    auto failure = [&](bool exhausted)
    { selected.resize(mandatory.size()); return ReplotPredeleteSelection{std::move(selected), false, exhausted}; };
    auto evaluate = [&]() -> std::optional<bool>
    {
        auto const cost = std::max(size_t(1), selected.size());
        if (cost > replot_predelete_fallback_source_visits - visits) return {};
        visits += cost; return fits(std::span<size_t const>(selected));
    };
    auto tried_greedy_prefix = [&]
    { return selected.size() <= greedy.size() && std::equal(selected.begin(), selected.end(), greedy.begin()); };
    // Try every affordable singleton before bounded combination search. With
    // no pending sources this covers the whole batch with one probe per source;
    // a late useful file cannot be hidden by an earlier small debt commitment.
    for (size_t source = 0; source < count; ++source)
    {
        ++candidate_visits; auto const current = candidates[source];
        if (included[source] || !current.eligible || current.allocated_bytes > allowance - mandatory_debt) continue;
        selected.push_back(source);
        if (!tried_greedy_prefix())
        {
            auto feasible = evaluate();
            if (!feasible) return failure(true);
            if (*feasible) return {std::move(selected), true};
        }
        selected.resize(mandatory.size());
    }
    // Enumerate legal combinations without recursion or per-node allocation.
    // Bound both expensive feasibility calls/source projections and cheap
    // traversal of skipped or unaffordable branches. Exhaustion is a search
    // limit, never evidence that no legal subset could satisfy the caller.
    std::vector<size_t> next(count + 1, 0); size_t depth = 0; outstanding = mandatory_debt;
    for (;;)
    {
        bool descended = false;
        while (next[depth] < count)
        {
            if (candidate_visits == replot_predelete_fallback_candidate_visits) return failure(true);
            ++candidate_visits; auto const source = next[depth]++; auto const current = candidates[source];
            if (included[source] || !current.eligible || current.allocated_bytes > allowance - outstanding) continue;
            selected.push_back(source); outstanding += current.allocated_bytes;
            if (depth && !tried_greedy_prefix())
            {
                if (combinations == replot_predelete_combination_checks) return failure(true);
                ++combinations; auto feasible = evaluate();
                if (!feasible) return failure(true);
                if (*feasible) return {std::move(selected), true};
            }
            ++depth; next[depth] = source + 1; descended = true; break;
        }
        if (descended) continue;
        if (!depth) return failure(false);
        --depth; outstanding -= candidates[selected.back()].allocated_bytes; selected.pop_back();
    }
}
struct ReplotBatch
{
    uint64_t lane = 0, ordinal = 0, first_source = 0, sources = 0;
    uint64_t final_bound_bytes = 0, workspace_bound_bytes = 0;
    uint64_t receipt_allowance_bytes = 0, receipt_staging_bytes = 0;
};
struct ReplotSpace
{
    uint64_t free_bytes = 0, reclaimable_allocated_bytes = 0, final_bound_bytes = 0, workspace_bound_bytes = 0;
    uint64_t metadata_allowance_bytes = 0;
};
// Live free space already excludes allocated, identity-checked metadata. Reserve
// only the allowance that has not yet been allocated; never credit other files.
inline uint64_t replot_metadata_remaining(uint64_t allowance, uint64_t allocated)
{ return allowance - std::min(allowance, allocated); }
inline void write_replot_native_identity(std::ostream& out, file_io::DurableIdentity const& identity)
{
    require(file_io::valid_durable_identity(identity), "invalid migration native identity");
    out << unsigned(identity.scheme) << ' ' << std::quoted(identity.filesystem) << ' ' << identity.handle_type << ' ' << std::quoted(identity.object) << '\n';
}
inline file_io::DurableIdentity read_replot_native_identity(std::istream& in)
{
    unsigned scheme = 0; file_io::DurableIdentity result;
    in >> scheme >> std::quoted(result.filesystem) >> result.handle_type >> std::quoted(result.object);
    require(in && scheme <= unsigned(file_io::DurableIdentity::Scheme::windows_file_id), "invalid migration native identity representation");
    result.scheme = file_io::DurableIdentity::Scheme(scheme);
    require(file_io::valid_durable_identity(result), "invalid migration native identity"); return result;
}
struct ReplotReview
{
    // Canonical public observations are embedded once, in this immutable file.
    std::string inventory_json, resources_json;
    FarmingState farm;
    uint64_t reclaim_before_bytes = 0;
    uint64_t memory_limit_bytes = 0;
    std::vector<uint64_t> sources;
    std::vector<ReplotBatch> batches;
    std::vector<ReplotSpace> space;
    // Destination order, followed by additional explicitly reviewed scratch
    // roots. Existing source roots remain owned by the embedded inventory.
    std::vector<farm::InventoryRoot> owned_roots;
    std::vector<std::string> scratch_directories;
    // Absence retains the exact legacy review bytes and mandatory full proofs.
    std::optional<ReplotVerification> verification;
    bool directory_confirmation = false;
};
inline uint64_t replot_native_root_memory(ReplotReview const& review)
{
    uint64_t bytes = review.owned_roots.capacity() * sizeof(farm::InventoryRoot) + review.scratch_directories.capacity() * sizeof(std::string);
    for (auto const& root : review.owned_roots)
    {
        for (auto const* text : {&root.path, &root.directory_identity, &root.storage.filesystem, &root.durable.filesystem, &root.durable.object}) bytes += 2 * text->capacity();
        bytes += root.storage.drives.capacity() * sizeof(std::string); for (auto const& drive : root.storage.drives) bytes += 2 * drive.capacity();
    }
    for (auto const& path : review.scratch_directories) bytes += 2 * path.capacity(); return bytes;
}
inline void validate_replot_review(ReplotReview const& review)
{
    if (review.verification) validate_replot_verification(*review.verification);
    require(!review.directory_confirmation || review.verification.has_value(), "directory confirmation requires an explicit verification policy");
    auto const& state = review.farm;
    require(state.version == 6 && state.fixed_extent && !state.capacity_fill && state.estimate.feasible &&
        !state.directories.empty() && !review.inventory_json.empty() && !review.resources_json.empty(), "invalid replot review");
    require(state.fingerprint == std::string(64, '0') && state.seed == std::string(64, '0') && state.seed_owner.empty() &&
        state.farmer == std::string(96, '0') && state.pool == std::string(state.pool_public_key ? 96 : 64, '0'),
        "replot review contains private identity fields");
    require(!review.sources.empty() && review.sources.size() <= farming_detail::maximum_files &&
        review.batches.size() <= farming_detail::maximum_files && review.space.size() == state.directories.size() &&
        review.memory_limit_bytes >= 16 * 1024 * 1024, "replot review exceeds its work or memory bounds");
    uint64_t batch = 0, source = 0;
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        auto const& directory = state.directories[lane];
        require(file_io::native_path(directory.directory).is_absolute() && directory.extents.empty() && !directory.sealed_files,
            "replot review must retain fixed, unmodified ordinary assignments");
        for (uint64_t ordinal = 0; ordinal < directory.layout.files; ++ordinal, ++batch)
        {
            require(batch < review.batches.size(), "replot review is missing a replacement batch");
            auto const& current = review.batches[size_t(batch)];
            require(current.lane == lane && current.ordinal == ordinal && current.first_source == source &&
                current.sources <= replot_maximum_batch_sources && current.sources <= review.sources.size() - source && current.final_bound_bytes,
                "invalid replot batch assignment");
            source += current.sources;
        }
    }
    require(batch == review.batches.size() && source == review.sources.size(), "replot review has unassigned sources or replacements");
    auto selected = review.sources; std::sort(selected.begin(), selected.end());
    require(std::adjacent_find(selected.begin(), selected.end()) == selected.end(), "duplicate replot candidate");
    require(review.owned_roots.size() <= 1024 && review.scratch_directories.size() <= 1024 &&
        replot_native_root_memory(review) < review.memory_limit_bytes / 4 &&
        (!review.owned_roots.empty() || review.scratch_directories.empty()), "invalid migration native root scope");
    if (!review.owned_roots.empty())
    {
        std::vector<std::string> paths; paths.reserve(state.directories.size() + review.scratch_directories.size());
        std::set<std::string> seen, scratch; bool durable = false;
        for (auto const& directory : state.directories)
        { require(seen.insert(directory.directory).second, "migration destination root repeats"); paths.push_back(directory.directory); }
        for (auto const& path : review.scratch_directories)
        {
            farm::inventory_detail::bounded_text(path);
            require(file_io::native_path(path).is_absolute() && file_io::path_text(file_io::native_path(path).lexically_normal()) == path && scratch.insert(path).second,
                "invalid or duplicate reviewed migration scratch root");
            if (seen.insert(path).second) paths.push_back(path);
        }
        require(paths.size() == review.owned_roots.size(), "migration native roots differ from its reviewed destination and scratch scope");
        std::set<std::pair<uint64_t, uint64_t>> native;
        for (size_t index = 0; index < paths.size(); ++index)
        {
            auto const& root = review.owned_roots[index];
            for (auto const* text : {&root.path, &root.directory_identity, &root.storage.filesystem}) farm::inventory_detail::bounded_text(*text);
            require(root.path == paths[index] && !root.directory_identity.empty() && !root.storage.filesystem.empty() && root.inode &&
                native.emplace(root.device, root.inode).second && root.storage.drives.size() <= 128 && file_io::valid_durable_identity(root.durable),
                "invalid or contradictory migration native root owner");
            for (auto const& drive : root.storage.drives) farm::inventory_detail::bounded_text(drive);
            durable = durable || file_io::has_durable_identity(root.durable);
            if (index < state.directories.size()) require(root.directory_identity == state.directories[index].directory_identity &&
                root.storage.filesystem == state.directories[index].filesystem, "migration native destination association changed");
        }
        require(durable, "migration native root schema has no prospective object evidence");
    }
}
inline std::string replot_review_text(ReplotReview const& review)
{
    validate_replot_review(review);
    std::ostringstream out;
    out << (review.directory_confirmation ? "ordinary-group-replot-review-v4\n" : review.verification ? "ordinary-group-replot-review-v3\n" : review.owned_roots.empty() ? "ordinary-group-replot-review-v1\n" : "ordinary-group-replot-review-v2\n") << std::quoted(review.inventory_json) << '\n'
        << std::quoted(review.resources_json) << '\n' << std::quoted(farming_state_text(review.farm, true)) << '\n'
        << review.reclaim_before_bytes << ' ' << review.memory_limit_bytes << ' ' << review.sources.size();
    for (auto index : review.sources) out << ' ' << index;
    out << '\n' << review.batches.size() << '\n';
    for (auto const& batch : review.batches)
        out << batch.lane << ' ' << batch.ordinal << ' ' << batch.first_source << ' ' << batch.sources << ' '
            << batch.final_bound_bytes << ' ' << batch.workspace_bound_bytes << ' ' << batch.receipt_allowance_bytes << ' ' << batch.receipt_staging_bytes << '\n';
    out << review.space.size() << '\n';
    for (auto const& space : review.space)
        out << space.free_bytes << ' ' << space.reclaimable_allocated_bytes << ' ' << space.final_bound_bytes << ' ' << space.workspace_bound_bytes << ' '
            << space.metadata_allowance_bytes << '\n';
    if (!review.owned_roots.empty() || review.verification)
    {
        out << review.scratch_directories.size(); for (auto const& path : review.scratch_directories) out << ' ' << std::quoted(path); out << '\n';
        out << review.owned_roots.size() << '\n';
        for (auto const& root : review.owned_roots)
        {
            out << std::quoted(root.path) << ' ' << std::quoted(root.directory_identity) << ' ' << root.device << ' ' << root.inode << ' '
                << std::quoted(root.storage.filesystem) << ' ' << root.storage.nonrotating << ' ' << root.storage.rotating << ' ' << root.storage.drives.size();
            for (auto const& drive : root.storage.drives) out << ' ' << std::quoted(drive); out << '\n'; write_replot_native_identity(out, root.durable);
        }
    }
    if (review.verification) write_replot_verification(out, *review.verification);
    return out.str();
}
inline ReplotReview parse_replot_review(std::string const& body, uint64_t memory_limit)
{
    require(body.size() < memory_limit / 16, "replot review exceeds its parsing memory budget");
    std::istringstream in(body); std::string version, farm; std::getline(in, version);
    require(version == "ordinary-group-replot-review-v1" || version == "ordinary-group-replot-review-v2" ||
        version == "ordinary-group-replot-review-v3" || version == "ordinary-group-replot-review-v4", "expected a saved replot review");
    ReplotReview result; size_t count = 0;
    result.directory_confirmation = version == "ordinary-group-replot-review-v4";
    in >> std::quoted(result.inventory_json) >> std::quoted(result.resources_json) >> std::quoted(farm)
        >> result.reclaim_before_bytes >> result.memory_limit_bytes >> count;
    require(in && count && count <= farming_detail::maximum_files && count < memory_limit / 128, "invalid replot candidate count");
    result.farm = parse_farming_state(farm, memory_limit / 4);
    result.sources.resize(count); for (auto& index : result.sources) in >> index;
    in >> count;
    require(in && count <= farming_detail::maximum_files && count < memory_limit / (128 * sizeof(ReplotBatch)), "invalid replot batch count");
    result.batches.resize(count);
    for (auto& batch : result.batches)
        in >> batch.lane >> batch.ordinal >> batch.first_source >> batch.sources >> batch.final_bound_bytes >> batch.workspace_bound_bytes
            >> batch.receipt_allowance_bytes >> batch.receipt_staging_bytes;
    in >> count; require(in && count == result.farm.directories.size(), "invalid replot capacity observations");
    result.space.resize(count);
    for (auto& space : result.space) in >> space.free_bytes >> space.reclaimable_allocated_bytes >> space.final_bound_bytes >> space.workspace_bound_bytes
        >> space.metadata_allowance_bytes;
    if (version != "ordinary-group-replot-review-v1")
    {
        in >> count; require(in && count <= 1024 && count < memory_limit / 8192, "migration reviewed scratch scope exceeds its memory bound");
        result.scratch_directories.resize(count); for (auto& path : result.scratch_directories) in >> std::quoted(path);
        in >> count; require(in && (count || version == "ordinary-group-replot-review-v3" || result.directory_confirmation) && count <= 1024 && count < memory_limit / 32768, "migration native root scope exceeds its memory bound");
        result.owned_roots.resize(count);
        for (auto& root : result.owned_roots)
        {
            size_t drives = 0;
            in >> std::quoted(root.path) >> std::quoted(root.directory_identity) >> root.device >> root.inode >> std::quoted(root.storage.filesystem)
                >> root.storage.nonrotating >> root.storage.rotating >> drives;
            require(in && drives <= 128 && drives < memory_limit / 8192, "migration native drive scope exceeds its memory bound");
            root.storage.drives.resize(drives); for (auto& drive : root.storage.drives) in >> std::quoted(drive);
            root.durable = read_replot_native_identity(in);
        }
        require(replot_native_root_memory(result) < memory_limit / 4, "migration native roots exceed their expanded parsing memory bound");
    }
    if (version == "ordinary-group-replot-review-v3" || result.directory_confirmation) result.verification = read_replot_verification(in);
    require(bool(in), "incomplete replot review"); in >> std::ws; require(in.eof(), "unexpected replot review data");
    validate_replot_review(result); return result;
}
struct ReplotConstructionEvidence
{
    uint64_t batch = 0;
    std::string engine, source_build, inspection_digest;
    stream_state::Stamp stage;
    bool operator==(ReplotConstructionEvidence const&) const = default;
};
struct ReplotRevalidation
{
    // Flat successors retain the original bytes once. Progress and receipts
    // remain bound to that origin; only these reviewed observations can vary.
    std::string origin_review, inventory_json, resources_json;
    std::string parent_policy_digest, journal_digest;
    std::vector<farm::ReplotReaderObservation> readers;
    std::optional<ReplotConstructionEvidence> construction;
    // A successor repeats the current override; its embedded origin is immutable.
    std::optional<ReplotVerification> verification;
};
inline void validate_replot_construction_binding(std::optional<ReplotConstructionEvidence> const& reviewed,
    std::optional<ReplotConstructionEvidence> const& current)
{
    require(current == reviewed,
        "construction checkpoint or native stage changed since review; repeat --revalidate at the current stopped boundary");
}
inline void validate_replot_reader_changes(farm::FarmInventory const& origin, farm::FarmInventory const& current,
    farm::InventoryLimits const& limits = {})
{
    require(origin.plots.size() == current.plots.size(), "revalidation changed the original inventory entry count");
    auto const original = farm::farm_inventory_text(origin, limits);
    (void)farm::farm_inventory_text(current, limits);
    auto comparable = current;
    for (size_t index = 0; index < comparable.plots.size(); ++index)
    {
        auto& plot = comparable.plots[index]; auto const& saved = origin.plots[index];
        plot.reader = saved.reader; plot.reader_identity = saved.reader_identity; plot.reader_version = saved.reader_version;
    }
    auto& network = comparable.network; auto const& saved = origin.network;
    network.source = saved.source; network.version = saved.version; network.captured_at = saved.captured_at;
    network.reader_version = saved.reader_version; network.reader_identity = saved.reader_identity;
    network.farmer_version = saved.farmer_version; network.solver_version = saved.solver_version; network.service_status = saved.service_status;
    comparable.retirement_readers = origin.retirement_readers;
    require(original == farm::farm_inventory_text(comparable, limits),
        "revalidation changed original native objects, network, routes, workloads or protected inventory accounting");
    require(current.network.available && !current.network.reader_identity.empty() && !current.network.reader_version.empty(),
        "revalidation requires fresh target reader/network observations");
    for (auto const& plot : current.plots) if (!plot.reader_identity.empty())
        require(!plot.reader.empty() && !plot.reader_version.empty(), "revalidation has incomplete source reader observations");
}
inline void validate_replot_nonactive_provenance(farm::FarmInventory const& origin, farm::FarmInventory const& current,
    std::span<uint64_t const> nonactive)
{
    for (auto index : nonactive)
    {
        auto const& original = origin.plots.at(size_t(index)); auto const& active = current.plots.at(size_t(index));
        require(original.reader == active.reader && original.reader_version == active.reader_version && original.reader_identity == active.reader_identity,
            "revalidation cannot relabel original durable source identification as a new reader observation");
    }
}
inline std::string replot_revalidation_text(ReplotRevalidation const& value)
{
    require(!value.origin_review.empty() && !value.inventory_json.empty() && !value.resources_json.empty() &&
        (value.parent_policy_digest.empty() || (value.parent_policy_digest.size() == 64 &&
            value.parent_policy_digest.find_first_not_of("0123456789abcdef") == value.parent_policy_digest.npos)) &&
        value.journal_digest.size() == 64 && value.journal_digest.find_first_not_of("0123456789abcdef") == value.journal_digest.npos,
        "invalid reviewed replot revalidation");
    require(value.origin_review.starts_with("ordinary-group-replot-review-v1\n") || value.origin_review.starts_with("ordinary-group-replot-review-v2\n") ||
        value.origin_review.starts_with("ordinary-group-replot-review-v3\n") || value.origin_review.starts_with("ordinary-group-replot-review-v4\n"),
        "revalidation must embed one original review, never a nested successor");
    require(value.readers.size() <= 1024, "revalidation reader observations exceed their root bound");
    std::set<uint64_t> roots;
    for (auto const& reader : value.readers)
        require(roots.insert(reader.root).second && (reader.reader == "chia" || reader.reader == "gigahorse" || reader.reader == "drchia") &&
            !reader.version.empty() && reader.version.size() <= 4096 && reader.identity.size() == 64 &&
            reader.identity.find_first_not_of("0123456789abcdef") == reader.identity.npos && !reader.source.empty() && reader.source.size() <= 8192 &&
            !reader.captured_at.empty() && reader.captured_at.size() <= 128, "invalid revalidation reader observation");
    if (value.construction)
    {
        auto const& saved = *value.construction;
        require(saved.batch < farming_detail::maximum_files && (saved.engine == "incremental" || saved.engine == "staged" ||
            saved.engine == "merged" || saved.engine == "external") && !saved.source_build.empty() && saved.source_build.size() <= 128 &&
            saved.inspection_digest.size() == 64 && saved.inspection_digest.find_first_not_of("0123456789abcdef") == saved.inspection_digest.npos &&
            saved.stage.inode && saved.stage.mtime_ns < 1000000000 && saved.stage.ctime_ns < 1000000000,
            "invalid reviewed construction checkpoint binding");
    }
    if (value.verification) validate_replot_verification(*value.verification);
    std::ostringstream out; out << (value.verification ? "ordinary-group-replot-revalidation-v3\n" : value.construction ? "ordinary-group-replot-revalidation-v2\n" : "ordinary-group-replot-revalidation-v1\n")
        << std::quoted(value.origin_review) << '\n'
        << std::quoted(value.inventory_json) << '\n' << std::quoted(value.resources_json) << '\n'
        << std::quoted(value.parent_policy_digest) << ' ' << std::quoted(value.journal_digest) << '\n' << value.readers.size() << '\n';
    for (auto const& reader : value.readers) out << reader.root << ' ' << std::quoted(reader.reader) << ' ' << std::quoted(reader.version) << ' '
        << std::quoted(reader.identity) << ' ' << std::quoted(reader.source) << ' ' << std::quoted(reader.captured_at) << '\n';
    if (value.verification) out << bool(value.construction) << '\n';
    if (value.construction)
    {
        auto const& saved = *value.construction; auto const& stamp = saved.stage;
        out << saved.batch << ' ' << std::quoted(saved.engine) << ' ' << std::quoted(saved.source_build) << ' ' << std::quoted(saved.inspection_digest)
            << ' ' << stamp.device << ' ' << stamp.inode << ' ' << stamp.bytes << ' ' << stamp.mtime << ' ' << stamp.ctime
            << ' ' << stamp.mtime_ns << ' ' << stamp.ctime_ns << '\n';
    }
    if (value.verification) write_replot_verification(out, *value.verification);
    return out.str();
}
inline ReplotRevalidation parse_replot_revalidation(std::string const& body, uint64_t memory_limit)
{
    require(body.size() < memory_limit / 16, "replot revalidation exceeds its parsing memory budget");
    std::istringstream in(body); std::string version; std::getline(in, version);
    require(version == "ordinary-group-replot-revalidation-v1" || version == "ordinary-group-replot-revalidation-v2" ||
        version == "ordinary-group-replot-revalidation-v3", "expected a reviewed replot revalidation");
    ReplotRevalidation result; in >> std::quoted(result.origin_review) >> std::quoted(result.inventory_json) >> std::quoted(result.resources_json)
        >> std::quoted(result.parent_policy_digest) >> std::quoted(result.journal_digest);
    size_t count = 0; in >> count;
    require(in && count <= 1024 && count < memory_limit / (128 * sizeof(farm::ReplotReaderObservation)), "revalidation reader count exceeds its memory bound");
    result.readers.resize(count);
    for (auto& reader : result.readers) in >> reader.root >> std::quoted(reader.reader) >> std::quoted(reader.version) >> std::quoted(reader.identity)
        >> std::quoted(reader.source) >> std::quoted(reader.captured_at);
    bool construction = version == "ordinary-group-replot-revalidation-v2";
    if (version == "ordinary-group-replot-revalidation-v3")
    { unsigned present = 0; in >> present; require(in && present <= 1, "invalid reviewed construction presence"); construction = present; }
    if (construction)
    {
        auto& saved = result.construction.emplace(); auto& stamp = saved.stage;
        in >> saved.batch >> std::quoted(saved.engine) >> std::quoted(saved.source_build) >> std::quoted(saved.inspection_digest)
            >> stamp.device >> stamp.inode >> stamp.bytes >> stamp.mtime >> stamp.ctime >> stamp.mtime_ns >> stamp.ctime_ns;
    }
    if (version == "ordinary-group-replot-revalidation-v3") result.verification = read_replot_verification(in);
    require(bool(in), "incomplete replot revalidation"); in >> std::ws; require(in.eof(), "unexpected replot revalidation data");
    (void)replot_revalidation_text(result); return result;
}
// Keep the retirement/receipt prefix contiguous. Later admitted outputs retain
// their own exact construction/publication evidence until that prefix reaches
// them. Unconfirmed entries grant no ordinary source retirement authority;
// durable readiness and immutable completion evidence fence retirement.
inline constexpr size_t replot_pending_output_limit = 3;
struct ReplotQueuedOutput
{
    uint64_t batch = 0;
    std::optional<stream_state::Stamp> verified_stage, published, ready;
    file_io::DurableIdentity verified_stage_identity, published_identity, ready_identity;
    std::optional<bool> full_proof_validated;
    std::vector<ReplotCandidateProgress> retirements;
    bool operator==(ReplotQueuedOutput const&) const = default;
};
struct ReplotLaneProgress
{
    uint64_t baseline_files = 0, files = 0;
    std::string baseline_digest, digest;
    bool operator==(ReplotLaneProgress const&) const = default;
};
struct ReplotJournal
{
    std::string review_digest, parent_digest, receipt_digest;
    std::string policy_digest;
    uint64_t revision = 1, batch = 0, retired_prefix = 0;
    ReplotPhase phase = ReplotPhase::preparing;
    FarmingState farm;
    // Only the active batch is replicated. The immutable reviewed catalog is
    // required for resume; completed batches have chained per-output receipts.
    std::vector<ReplotCandidateProgress> retirements;
    std::optional<stream_state::Stamp> verified_stage, published, ready;
    file_io::DurableIdentity verified_stage_identity, published_identity, ready_identity;
    // Legacy sealed records required a positive full proof. Explicit evidence
    // records the work actually completed, independently of successor policy.
    std::optional<bool> full_proof_validated;
    // Once enabled, preserve the schema even after the queue drains so old
    // bytes remain canonical and its authority never disappears on restart.
    bool queued_publications = false;
    std::vector<ReplotQueuedOutput> queued;
    std::string authority_directory, authority_identity, authority_host, authority_user;
    uint64_t authority_revision = 0;
    // Each destination advances a contiguous immutable completion chain.
    // Completed outputs leave the bounded unconfirmed queue, without forcing
    // an unavailable earlier directory to advance the global receipt prefix.
    std::vector<ReplotLaneProgress> completed_lanes;
};
inline bool replot_journal_native_evidence(ReplotJournal const& journal)
{ return file_io::has_durable_identity(journal.verified_stage_identity) || file_io::has_durable_identity(journal.published_identity) || file_io::has_durable_identity(journal.ready_identity); }
inline char const* replot_phase_name(ReplotPhase phase)
{
    switch (phase)
    {
    case ReplotPhase::preparing: return "preparing";
    case ReplotPhase::replacing: return "replacing";
    case ReplotPhase::handoff: return "handoff";
    case ReplotPhase::complete: return "complete";
    }
    throw std::runtime_error("invalid replot phase");
}
inline char const* replot_retirement_name(ReplotRetirement state)
{
    switch (state)
    {
    case ReplotRetirement::active: return "active";
    case ReplotRetirement::claim_intent: return "claim-intent";
    case ReplotRetirement::claimed: return "claimed";
    case ReplotRetirement::retired: return "retired";
    }
    throw std::runtime_error("invalid replot retirement state");
}
inline bool replot_digest(std::string const& value)
{ return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == value.npos; }
inline uint64_t replot_output_count(FarmingState const& state)
{
    uint64_t count = 0;
    for (auto const& directory : state.directories)
    {
        require(directory.layout.files <= farming_detail::maximum_files - count, "replot output count exceeds its bound");
        count += directory.layout.files;
    }
    return count;
}
inline std::pair<size_t, uint64_t> replot_batch_position(FarmingState const& state, uint64_t batch)
{
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        if (batch < state.directories[lane].layout.files) return {lane, batch};
        batch -= state.directories[lane].layout.files;
    }
    throw std::runtime_error("migration batch exceeds its fixed destination assignments");
}
inline bool replot_lane_completed(ReplotJournal const& journal, uint64_t batch)
{
    if (journal.completed_lanes.empty() || batch >= replot_output_count(journal.farm)) return false;
    auto const [lane, ordinal] = replot_batch_position(journal.farm, batch);
    return ordinal < journal.completed_lanes.at(lane).files;
}
inline uint64_t replot_completed_members(ReplotJournal const& journal)
{
    using namespace farm::resource_detail;
    uint64_t result = 0;
    for (size_t lane = 0; lane < journal.completed_lanes.size(); ++lane)
    {
        auto const& layout = journal.farm.directories[lane].layout; auto const files = journal.completed_lanes[lane].files;
        if (!files) continue;
        if (layout.fixed_members) result = add(result, std::min(layout.total_members, multiply(files, layout.group_members)));
        else { group_tools::require(layout.files, "completed migration destination has no fixed file layout");
            result = add(result, add(multiply(files, layout.total_members / layout.files), std::min(files, layout.total_members % layout.files))); }
    }
    return result;
}
inline bool replot_source_completed(ReplotReview const& review, ReplotJournal const& journal, uint64_t position)
{
    uint64_t prefix = 0;
    for (size_t lane = 0; lane < journal.completed_lanes.size(); ++lane)
    {
        auto const files = journal.completed_lanes[lane].files;
        if (files)
        {
            auto const& first = review.batches.at(size_t(prefix)); auto const& last = review.batches.at(size_t(prefix + files - 1));
            if (position >= first.first_source && position < last.first_source + last.sources) return true;
        }
        prefix += journal.farm.directories[lane].layout.files;
    }
    return false;
}
inline bool replot_output_matches(stream_state::Stamp const& saved, file_io::DurableIdentity const& saved_identity,
    stream_state::Stamp current, file_io::DurableIdentity const& current_identity, bool rename_allowed = false)
{
    if (!file_io::valid_durable_identity(saved_identity) || !file_io::valid_durable_identity(current_identity)) return false;
    if (file_io::has_durable_identity(saved_identity))
    {
        if (saved_identity != current_identity) return false;
        current.device = saved.device;
    }
    return rename_allowed ? stream_state::same_output(saved, current) : saved == current;
}
inline void validate_replot_journal(ReplotJournal const& journal)
{
    require(journal.authority_directory.empty() ? (!journal.authority_revision && journal.authority_identity.empty() &&
        journal.authority_host.empty() && journal.authority_user.empty()) : journal.queued_publications && journal.authority_revision &&
        journal.authority_revision <= journal.revision && file_io::native_path(journal.authority_directory).is_absolute() &&
        !journal.authority_identity.empty() && !journal.authority_host.empty() && !journal.authority_user.empty(), "invalid migration controller authority");
    require(journal.queued.size() < replot_pending_output_limit && (journal.queued.empty() || journal.queued_publications),
        "migration publication queue exceeds its durable bound");
    require(journal.completed_lanes.empty() || (journal.queued_publications && journal.completed_lanes.size() == journal.farm.directories.size()), "migration completion lanes omit a fixed destination");
    for (size_t lane = 0; lane < journal.completed_lanes.size(); ++lane)
    {
        auto const& progress = journal.completed_lanes[lane];
        require(progress.baseline_files <= progress.files && progress.files <= journal.farm.directories[lane].layout.files &&
            (progress.files ? replot_digest(progress.digest) : progress.digest.empty()) &&
            (progress.baseline_files ? replot_digest(progress.baseline_digest) : progress.baseline_digest.empty()),
            "invalid migration destination completion prefix");
    }
    uint64_t previous = journal.batch;
    for (auto const& output : journal.queued)
    {
        require(journal.phase == ReplotPhase::replacing && output.batch >= journal.batch && output.batch < replot_output_count(journal.farm) &&
            (&output == &journal.queued.front() || output.batch > previous) &&
            (output.batch != journal.batch || (!journal.verified_stage && !journal.published && !journal.ready)),
            "migration publication queue repeats or overlaps active authority");
        require(!replot_lane_completed(journal, output.batch), "completed migration output retained an unconfirmed queue slot");
        require(!output.full_proof_validated || output.verified_stage, "queued migration proof lacks its checked stage");
        require(output.retirements.size() <= replot_maximum_batch_sources, "queued migration retirement count exceeds its bound");
        for (auto const& progress : output.retirements)
            require(unsigned(progress.state) <= unsigned(ReplotRetirement::retired) &&
                (progress.state == ReplotRetirement::active ? !progress.predelete : output.ready || progress.predelete),
                "queued migration source retirement lacks bounded early-deletion intent");
        for (auto pair : {std::pair{&output.verified_stage, &output.verified_stage_identity}, std::pair{&output.published, &output.published_identity}, std::pair{&output.ready, &output.ready_identity}})
            require(file_io::valid_durable_identity(*pair.second) && (!file_io::has_durable_identity(*pair.second) ||
                (*pair.first && (**pair.first).inode)), "queued migration native evidence lacks its output stamp");
        require(!output.published || (output.verified_stage && replot_output_matches(*output.verified_stage, output.verified_stage_identity,
            *output.published, output.published_identity, true)), "queued migration publication lacks its checked stage");
        require(!output.ready || (!journal.completed_lanes.empty() && output.published && replot_output_matches(*output.published, output.published_identity,
            *output.ready, output.ready_identity)), "queued migration readiness lacks its exact immutable publication");
        previous = output.batch;
    }
    require(!journal.full_proof_validated || journal.verified_stage.has_value(), "migration proof evidence has no structurally checked stage");
    require(replot_digest(journal.review_digest) && (journal.policy_digest.empty() || (replot_digest(journal.policy_digest) && journal.revision > 1)) && journal.revision &&
        ((journal.revision == 1 && journal.parent_digest.empty()) ||
            (journal.revision > 1 && replot_digest(journal.parent_digest))), "invalid replot journal revision");
    require(journal.farm.version == 6 && journal.farm.fixed_extent && !journal.farm.capacity_fill &&
        journal.farm.estimate.feasible && !journal.farm.directories.empty(), "replot journal requires a fixed ordinary farming job");
    require(journal.batch <= replot_output_count(journal.farm) &&
        journal.retirements.size() <= replot_maximum_batch_sources &&
        journal.retired_prefix <= farming_detail::maximum_files && unsigned(journal.phase) <= unsigned(ReplotPhase::complete) &&
        (journal.batch ? replot_digest(journal.receipt_digest) : journal.receipt_digest.empty()),
        "invalid replot journal progress");
    for (auto value : journal.retirements)
        require(unsigned(value.state) <= unsigned(ReplotRetirement::retired) &&
            (value.state != ReplotRetirement::active || !value.predelete), "invalid replot retirement state");
    for (auto const pair : {std::pair{&journal.verified_stage, &journal.verified_stage_identity},
        std::pair{&journal.published, &journal.published_identity}, std::pair{&journal.ready, &journal.ready_identity}})
        require(file_io::valid_durable_identity(*pair.second) &&
            (!file_io::has_durable_identity(*pair.second) || (*pair.first && (**pair.first).inode)),
            "migration native evidence lacks its durable output stamp");
    auto consistent = [](auto const& later, auto const& a, auto const& b)
    { return !later || !file_io::has_durable_identity(a) || a == b; };
    require(consistent(journal.published, journal.verified_stage_identity, journal.published_identity) &&
        consistent(journal.ready, journal.published_identity, journal.ready_identity) &&
        consistent(journal.ready, journal.verified_stage_identity, journal.ready_identity), "migration publication changed or omitted its durable native object");
    require(!journal.published || (journal.verified_stage && replot_output_matches(*journal.verified_stage, journal.verified_stage_identity,
        *journal.published, journal.published_identity, true)),
        "replot publication lacks its verified staged identity");
    require(!journal.ready || (journal.published && replot_output_matches(*journal.published, journal.published_identity,
        *journal.ready, journal.ready_identity)), "replot readiness lacks its immutable publication");
    if (journal.phase == ReplotPhase::preparing)
    {
        require(!journal.batch && !journal.retired_prefix && !journal.verified_stage && !journal.published && !journal.ready &&
            std::all_of(journal.retirements.begin(), journal.retirements.end(), [](auto value) { return value.state == ReplotRetirement::active; }),
            "replot preparation contains destructive progress");
    }
    if (journal.phase >= ReplotPhase::handoff)
        require(journal.batch == replot_output_count(journal.farm) && !journal.verified_stage && !journal.published && !journal.ready && journal.retirements.empty(),
            "replot handoff has unfinished replacements or retirements");
}
inline std::string replot_journal_text(ReplotJournal const& journal)
{
    validate_replot_journal(journal);
    std::ostringstream out;
    bool const durable = replot_journal_native_evidence(journal);
    out << (!journal.completed_lanes.empty() ? "ordinary-group-replot-journal-v6\n" : journal.queued_publications ? "ordinary-group-replot-journal-v5\n" : journal.full_proof_validated ? "ordinary-group-replot-journal-v4\n" : durable ? "ordinary-group-replot-journal-v3\n" : journal.policy_digest.empty() ? "ordinary-group-replot-journal-v1\n" : "ordinary-group-replot-journal-v2\n") << std::quoted(journal.review_digest) << ' '
        << journal.revision << ' ' << std::quoted(journal.parent_digest) << ' ' << unsigned(journal.phase) << ' ' << journal.batch << ' '
        << journal.retired_prefix << ' ' << std::quoted(journal.receipt_digest) << '\n'
        << std::quoted(farming_state_text(journal.farm)) << '\n' << journal.retirements.size();
    for (auto value : journal.retirements) out << ' ' << unsigned(value.state) << ' ' << value.predelete;
    out << '\n';
    for (auto const* value : {&journal.verified_stage, &journal.published, &journal.ready})
    {
        out << value->has_value();
        if (*value) { auto const& stamp = **value; out << ' ' << stamp.device << ' ' << stamp.inode << ' ' << stamp.bytes << ' '
            << stamp.mtime << ' ' << stamp.ctime << ' ' << stamp.mtime_ns << ' ' << stamp.ctime_ns; }
        out << '\n';
    }
    if (journal.queued_publications || durable || !journal.policy_digest.empty() || journal.full_proof_validated) out << std::quoted(journal.policy_digest) << '\n';
    if (journal.queued_publications || durable || journal.full_proof_validated) for (auto const* identity : {&journal.verified_stage_identity, &journal.published_identity, &journal.ready_identity})
        write_replot_native_identity(out, *identity);
    if (journal.queued_publications)
    {
        out << journal.full_proof_validated.has_value();
        if (journal.full_proof_validated) out << ' ' << *journal.full_proof_validated;
        out << '\n' << journal.queued.size() << '\n';
        for (auto const& output : journal.queued)
        {
            out << output.batch << ' ' << output.retirements.size();
            for (auto const& progress : output.retirements) out << ' ' << unsigned(progress.state) << ' ' << progress.predelete;
            out << '\n';
            for (auto const* value : {&output.verified_stage, &output.published})
            {
                out << value->has_value();
                if (*value) { auto const& stamp = **value; out << ' ' << stamp.device << ' ' << stamp.inode << ' ' << stamp.bytes << ' '
                    << stamp.mtime << ' ' << stamp.ctime << ' ' << stamp.mtime_ns << ' ' << stamp.ctime_ns; }
                out << '\n';
            }
            write_replot_native_identity(out, output.verified_stage_identity); write_replot_native_identity(out, output.published_identity);
            out << output.full_proof_validated.has_value();
            if (output.full_proof_validated) out << ' ' << *output.full_proof_validated;
            out << '\n';
            if (!journal.completed_lanes.empty())
            {
                out << output.ready.has_value();
                if (output.ready) { auto const& stamp = *output.ready; out << ' ' << stamp.device << ' ' << stamp.inode << ' ' << stamp.bytes << ' '
                    << stamp.mtime << ' ' << stamp.ctime << ' ' << stamp.mtime_ns << ' ' << stamp.ctime_ns; }
                out << '\n'; write_replot_native_identity(out, output.ready_identity);
            }
        }
        out << std::quoted(journal.authority_directory) << ' ' << std::quoted(journal.authority_identity) << ' '
            << std::quoted(journal.authority_host) << ' ' << std::quoted(journal.authority_user) << ' ' << journal.authority_revision << '\n';
        if (!journal.completed_lanes.empty())
        {
            out << journal.completed_lanes.size() << '\n';
            for (auto const& lane : journal.completed_lanes) out << lane.baseline_files << ' ' << lane.files << ' '
                << std::quoted(lane.baseline_digest) << ' ' << std::quoted(lane.digest) << '\n';
        }
    }
    else if (journal.full_proof_validated) out << *journal.full_proof_validated << '\n';
    return out.str();
}
inline ReplotJournal parse_replot_journal(std::string const& body, uint64_t memory_limit)
{
    require(body.size() < memory_limit / 8, "replot journal exceeds its parsing memory budget");
    std::istringstream in(body); std::string version, farm; unsigned phase = 0;
    std::getline(in, version); require(version == "ordinary-group-replot-journal-v1" || version == "ordinary-group-replot-journal-v2" ||
        version == "ordinary-group-replot-journal-v3" || version == "ordinary-group-replot-journal-v4" || version == "ordinary-group-replot-journal-v5" || version == "ordinary-group-replot-journal-v6", "unsupported replot journal version");
    ReplotJournal result; size_t count = 0;
    in >> std::quoted(result.review_digest) >> result.revision >> std::quoted(result.parent_digest) >> phase >> result.batch
        >> result.retired_prefix >> std::quoted(result.receipt_digest) >> std::quoted(farm);
    require(in && phase <= unsigned(ReplotPhase::complete), "invalid replot journal header"); result.phase = ReplotPhase(phase);
    result.farm = parse_farming_state(farm, memory_limit / 2);
    in >> count;
    require(in && count <= replot_maximum_batch_sources && count < memory_limit / 64, "replot retirement count exceeds its bound");
    result.retirements.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        unsigned value = 0; bool predelete = false; in >> value >> predelete;
        require(in && value <= unsigned(ReplotRetirement::retired), "invalid replot retirement state");
        result.retirements.push_back({ReplotRetirement(value), predelete});
    }
    for (auto* value : {&result.verified_stage, &result.published, &result.ready})
    {
        bool ready = false; in >> ready;
        if (ready)
        {
            stream_state::Stamp stamp;
            in >> stamp.device >> stamp.inode >> stamp.bytes >> stamp.mtime >> stamp.ctime >> stamp.mtime_ns >> stamp.ctime_ns;
            require(in && stamp.bytes && stamp.mtime_ns < 1000000000 && stamp.ctime_ns < 1000000000,
                "invalid replot replacement stamp"); *value = stamp;
        }
        require(bool(in), "incomplete replot replacement publication record");
    }
    require(bool(in), "incomplete replot readiness record");
    if (version != "ordinary-group-replot-journal-v1")
    { in >> std::quoted(result.policy_digest); require(in && (result.policy_digest.empty() ? version != "ordinary-group-replot-journal-v2" : replot_digest(result.policy_digest)), "invalid replot active policy digest"); }
    if (version == "ordinary-group-replot-journal-v3" || version == "ordinary-group-replot-journal-v4" || version == "ordinary-group-replot-journal-v5" || version == "ordinary-group-replot-journal-v6")
    {
        for (auto* identity : {&result.verified_stage_identity, &result.published_identity, &result.ready_identity}) *identity = read_replot_native_identity(in);
        require(version != "ordinary-group-replot-journal-v3" || replot_journal_native_evidence(result), "migration native journal schema has no object evidence");
    }
    if (version == "ordinary-group-replot-journal-v4")
    { unsigned proof = 0; in >> proof; require(in && proof <= 1, "invalid migration full-proof evidence"); result.full_proof_validated = bool(proof); }
    if (version == "ordinary-group-replot-journal-v5" || version == "ordinary-group-replot-journal-v6")
    {
        result.queued_publications = true;
        bool evidence = false; in >> evidence;
        if (evidence) { bool proof = false; in >> proof; result.full_proof_validated = proof; }
        in >> count; require(in && count < replot_pending_output_limit, "migration publication queue exceeds its parsing bound");
        result.queued.resize(count);
        for (auto& output : result.queued)
        {
            size_t retirements = 0; in >> output.batch >> retirements;
            require(in && retirements <= replot_maximum_batch_sources && retirements < memory_limit / 64, "queued migration retirements exceed their parsing bound");
            output.retirements.resize(retirements);
            for (auto& progress : output.retirements)
            {
                unsigned value = 0; in >> value >> progress.predelete;
                require(in && value <= unsigned(ReplotRetirement::retired), "invalid queued migration retirement"); progress.state = ReplotRetirement(value);
            }
            for (auto* value : {&output.verified_stage, &output.published})
            {
                bool present = false; in >> present;
                if (present)
                {
                    stream_state::Stamp stamp;
                    in >> stamp.device >> stamp.inode >> stamp.bytes >> stamp.mtime >> stamp.ctime >> stamp.mtime_ns >> stamp.ctime_ns;
                    require(in && stamp.bytes && stamp.mtime_ns < 1000000000 && stamp.ctime_ns < 1000000000, "invalid queued migration output stamp");
                    *value = stamp;
                }
            }
            output.verified_stage_identity = read_replot_native_identity(in); output.published_identity = read_replot_native_identity(in);
            in >> evidence; if (evidence) { bool proof = false; in >> proof; output.full_proof_validated = proof; }
            if (version == "ordinary-group-replot-journal-v6")
            {
                bool ready = false; in >> ready;
                if (ready) { stream_state::Stamp stamp; in >> stamp.device >> stamp.inode >> stamp.bytes >> stamp.mtime >> stamp.ctime >> stamp.mtime_ns >> stamp.ctime_ns;
                    require(in && stamp.bytes && stamp.mtime_ns < 1000000000 && stamp.ctime_ns < 1000000000, "invalid queued migration ready stamp"); output.ready = stamp; }
                output.ready_identity = read_replot_native_identity(in);
            }
            require(bool(in), "incomplete queued migration output");
        }
        in >> std::quoted(result.authority_directory) >> std::quoted(result.authority_identity) >> std::quoted(result.authority_host)
            >> std::quoted(result.authority_user) >> result.authority_revision;
        require(bool(in), "incomplete migration controller authority");
        if (version == "ordinary-group-replot-journal-v6")
        {
            in >> count; require(in && count == result.farm.directories.size() && count < memory_limit / 256, "migration completion lanes exceed their parsing bound");
            result.completed_lanes.resize(count);
            for (auto& lane : result.completed_lanes) in >> lane.baseline_files >> lane.files >> std::quoted(lane.baseline_digest) >> std::quoted(lane.digest);
            require(bool(in), "incomplete migration destination completion prefixes");
        }
    }
    in >> std::ws; require(in.eof(), "unexpected replot journal data");
    validate_replot_journal(result); return result;
}
inline bool valid_replot_revision(ReplotJournal const& before, ReplotJournal const& after)
{
    try { validate_replot_journal(before); validate_replot_journal(after); }
    catch (std::runtime_error const&) { return false; }
    if (before.revision == UINT64_MAX || after.revision != before.revision + 1 ||
        after.parent_digest != stream_state::digest(replot_journal_text(before)) ||
        before.review_digest != after.review_digest || farming_state_text(before.farm) != farming_state_text(after.farm)) return false;
    if (before.completed_lanes != after.completed_lanes)
    {
        auto unchanged = after; unchanged.revision = before.revision; unchanged.parent_digest = before.parent_digest;
        if (before.completed_lanes.empty())
        {
            if (after.completed_lanes.size() != before.farm.directories.size()) return false;
            uint64_t prefix = 0;
            for (size_t lane = 0; lane < after.completed_lanes.size(); ++lane)
            {
                auto const& progress = after.completed_lanes[lane]; auto const files = before.farm.directories[lane].layout.files;
                auto const completed = before.batch > prefix ? std::min(files, before.batch - prefix) : 0;
                if (progress.baseline_files != completed || progress.files != completed || progress.baseline_digest != progress.digest) return false;
                prefix += files;
            }
        }
        else
        {
            if (before.completed_lanes.size() != after.completed_lanes.size()) return false;
            std::optional<size_t> completed_lane;
            for (size_t lane = 0; lane < before.completed_lanes.size(); ++lane)
            {
                auto const& a = before.completed_lanes[lane]; auto const& b = after.completed_lanes[lane];
                if (a == b) continue;
                if (completed_lane || a.files == UINT64_MAX || b.files != a.files + 1 || b.baseline_files != a.baseline_files ||
                    b.baseline_digest != a.baseline_digest || b.digest == a.digest) return false;
                completed_lane = lane;
            }
            if (!completed_lane) return false;
            auto const lane = *completed_lane; uint64_t batch = before.completed_lanes[lane].files;
            for (size_t previous = 0; previous < lane; ++previous) batch += before.farm.directories[previous].layout.files;
            if (batch == before.batch && before.ready && std::all_of(before.retirements.begin(), before.retirements.end(), [](auto const& p) { return p.state == ReplotRetirement::retired; }))
            { if (after.queued != before.queued) return false; }
            else
            {
                auto const at = std::find_if(before.queued.begin(), before.queued.end(), [&](auto const& output) { return output.batch == batch; });
                if (at == before.queued.end() || !at->ready || !std::all_of(at->retirements.begin(), at->retirements.end(), [](auto const& p) { return p.state == ReplotRetirement::retired; })) return false;
                auto expected = before.queued; expected.erase(expected.begin() + (at - before.queued.begin()));
                if (after.queued != expected) return false;
                unchanged.queued = before.queued;
            }
        }
        unchanged.completed_lanes = before.completed_lanes;
        return replot_journal_text(unchanged) == replot_journal_text(before);
    }
    if (before.authority_directory != after.authority_directory || before.authority_identity != after.authority_identity ||
        before.authority_host != after.authority_host || before.authority_user != after.authority_user || before.authority_revision != after.authority_revision)
    {
        if (!before.authority_directory.empty() || after.authority_directory.empty() || after.authority_revision != after.revision || before.phase >= ReplotPhase::handoff) return false;
        auto unchanged = after; unchanged.revision = before.revision; unchanged.parent_digest = before.parent_digest;
        unchanged.authority_directory.clear(); unchanged.authority_identity.clear(); unchanged.authority_host.clear(); unchanged.authority_user.clear(); unchanged.authority_revision = 0;
        unchanged.queued_publications = before.queued_publications;
        return replot_journal_text(unchanged) == replot_journal_text(before);
    }
    if (before.queued_publications && !after.queued_publications) return false;
    if (before.queued != after.queued || before.queued_publications != after.queued_publications)
    {
        auto unchanged = after; unchanged.revision = before.revision; unchanged.parent_digest = before.parent_digest;
        unchanged.queued_publications = before.queued_publications; unchanged.queued = before.queued;
        // Adoption transfers exactly one queued object's evidence to the active
        // batch; a crash before/after this revision retains one authority.
        if (!before.queued.empty() && before.queued.front().batch == before.batch &&
            after.queued.size() + 1 == before.queued.size() &&
            std::equal(after.queued.begin(), after.queued.end(), before.queued.begin() + 1))
        {
            auto const& output = before.queued.front();
            if (before.verified_stage || before.published || before.ready || after.ready != output.ready ||
                after.verified_stage != output.verified_stage || after.published != output.published ||
                after.verified_stage_identity != output.verified_stage_identity || after.published_identity != output.published_identity || after.ready_identity != output.ready_identity ||
                after.full_proof_validated != output.full_proof_validated || after.retirements != output.retirements) return false;
            unchanged.verified_stage = before.verified_stage; unchanged.published = before.published; unchanged.ready = before.ready;
            unchanged.verified_stage_identity = before.verified_stage_identity; unchanged.published_identity = before.published_identity; unchanged.ready_identity = before.ready_identity;
            unchanged.full_proof_validated = before.full_proof_validated; unchanged.retirements = before.retirements;
        }
        else if (after.queued.size() == before.queued.size() + 1)
        {
            auto expected = after.queued;
            auto const added = std::find_if(expected.begin(), expected.end(), [&](auto const& output)
                { return std::none_of(before.queued.begin(), before.queued.end(), [&](auto const& previous) { return previous.batch == output.batch; }); });
            if (added == expected.end()) return false;
            auto const output = *added; expected.erase(added);
            if (expected != before.queued) return false;
            if (output.batch <= before.batch || output.verified_stage || output.published || output.ready || output.full_proof_validated ||
                std::any_of(output.retirements.begin(), output.retirements.end(), [](auto const& progress)
                    { return progress.state != ReplotRetirement::active || progress.predelete; })) return false;
        }
        else if (after.queued.size() == before.queued.size())
        {
            size_t changes = 0;
            for (size_t index = 0; index < before.queued.size(); ++index)
            {
                auto const& a = before.queued[index]; auto const& b = after.queued[index];
                if (a == b) continue;
                if (a.batch != b.batch || a.retirements.size() != b.retirements.size()) return false;
                auto changed = b;
                size_t retirement_changes = 0;
                for (size_t source = 0; source < a.retirements.size(); ++source)
                {
                    auto const old = a.retirements[source], next = b.retirements[source];
                    if (old == next) continue;
                    if (unsigned(next.state) != unsigned(old.state) + 1 || (!next.predelete && !a.ready) ||
                        (old.state != ReplotRetirement::active && old.predelete != next.predelete)) return false;
                    ++retirement_changes;
                }
                if (retirement_changes)
                {
                    if (retirement_changes != 1) return false;
                    changed.retirements = a.retirements;
                    if (changed != a) return false;
                }
                else if (!a.ready && b.ready)
                {
                    changed.ready = a.ready; changed.ready_identity = a.ready_identity;
                    if (changed != a) return false;
                }
                else if ((!a.verified_stage && b.published) || (a.verified_stage && (a.verified_stage != b.verified_stage ||
                    a.verified_stage_identity != b.verified_stage_identity || a.full_proof_validated != b.full_proof_validated)) ||
                    (a.published && (a.published != b.published || a.published_identity != b.published_identity)) || a.ready != b.ready || a.ready_identity != b.ready_identity ||
                    (!a.verified_stage && !b.verified_stage) ||
                    (a.verified_stage.has_value() == b.verified_stage.has_value() && a.published.has_value() == b.published.has_value())) return false;
                ++changes;
            }
            if (changes != 1 && (before.queued_publications || !after.queued.empty())) return false;
        }
        else return false;
        return replot_journal_text(unchanged) == replot_journal_text(before);
    }
    if (before.policy_digest != after.policy_digest)
    {
        // The coordinator separately binds this one transition to an explicit
        // successor and audits every retained object at its reviewed boundary.
        return !after.policy_digest.empty() && before.phase < ReplotPhase::handoff && after.phase == before.phase &&
            after.batch == before.batch && after.retired_prefix == before.retired_prefix && after.receipt_digest == before.receipt_digest &&
            after.retirements == before.retirements && after.verified_stage == before.verified_stage && after.published == before.published && after.ready == before.ready &&
            after.verified_stage_identity == before.verified_stage_identity && after.published_identity == before.published_identity && after.ready_identity == before.ready_identity &&
            after.full_proof_validated == before.full_proof_validated && after.queued == before.queued;
    }
    if (after.phase < before.phase || unsigned(after.phase) > unsigned(before.phase) + 1 ||
        after.batch < before.batch || after.batch > before.batch + 1) return false;
    // A completed receipt is durable before advancing the prefix. Its content
    // and exact next batch come from the immutable review at the controller.
    if (after.batch == before.batch + 1)
    {
        return before.phase == ReplotPhase::replacing && after.phase == before.phase && (before.ready || replot_lane_completed(before, before.batch)) && !after.verified_stage && !after.published && !after.ready && !after.full_proof_validated &&
            after.retired_prefix == before.retired_prefix + before.retirements.size() &&
            replot_digest(after.receipt_digest) && after.receipt_digest != before.receipt_digest &&
            (std::all_of(before.retirements.begin(), before.retirements.end(), [](auto value) { return value.state == ReplotRetirement::retired; }) ||
                (replot_lane_completed(before, before.batch) && std::all_of(before.retirements.begin(), before.retirements.end(), [](auto value) { return value.state == ReplotRetirement::active && !value.predelete; }))) &&
            std::all_of(after.retirements.begin(), after.retirements.end(), [](auto value) { return value.state == ReplotRetirement::active && !value.predelete; });
    }
    if (before.retired_prefix != after.retired_prefix || before.receipt_digest != after.receipt_digest ||
        before.retirements.size() != after.retirements.size()) return false;
    if (before.verified_stage && before.full_proof_validated != after.full_proof_validated) return false;
    size_t changes = size_t(after.phase != before.phase);
    for (auto const pair : {std::pair{&before.verified_stage, &after.verified_stage}, std::pair{&before.published, &after.published},
        std::pair{&before.ready, &after.ready}})
    {
        if (*pair.first != *pair.second)
        { if (*pair.first || !*pair.second || before.phase != ReplotPhase::replacing) return false; ++changes; }
    }
    auto native_transition = [](auto const& a, auto const& b, auto const& identity_a, auto const& identity_b)
    { return identity_a == identity_b || (!a && b); };
    if (!native_transition(before.verified_stage, after.verified_stage, before.verified_stage_identity, after.verified_stage_identity) ||
        !native_transition(before.published, after.published, before.published_identity, after.published_identity) ||
        !native_transition(before.ready, after.ready, before.ready_identity, after.ready_identity)) return false;
    for (size_t index = 0; index < before.retirements.size(); ++index)
    {
        auto const a = before.retirements[index], b = after.retirements[index];
        if (a == b) continue;
        if (before.phase != ReplotPhase::replacing || unsigned(b.state) != unsigned(a.state) + 1 ||
            (a.state != ReplotRetirement::active && a.predelete != b.predelete) ||
            (a.state == ReplotRetirement::active && !before.ready && !b.predelete)) return false;
        ++changes;
    }
    return changes == 1;
}
inline void validate_replot_progress(ReplotReview const& review, ReplotJournal const& journal)
{
    validate_replot_journal(journal);
    require(!review.verification || !journal.verified_stage || journal.full_proof_validated.has_value(),
        "explicit migration policy lacks its completed verification evidence");
    require(!review.verification || !journal.verified_stage || !journal.policy_digest.empty() ||
        !replot_requires_proof(*review.verification, journal.batch) || journal.full_proof_validated.value_or(false),
        "selected migration proof has no positive completed evidence");
    auto policy = journal.farm;
    policy.seed_owner.clear(); policy.pool_public_key = review.farm.pool_public_key;
    policy.fingerprint = policy.seed = std::string(64, '0'); policy.farmer = std::string(96, '0');
    policy.pool = std::string(policy.pool_public_key ? 96 : 64, '0');
    require(farming_state_text(policy) == farming_state_text(review.farm), "private migration assignments differ from the original review");
    require(journal.batch <= review.batches.size(), "private migration batch exceeds its reviewed assignment");
    for (auto const& output : journal.queued)
    {
        require(output.batch < review.batches.size() && output.retirements.size() == review.batches[size_t(output.batch)].sources && (!review.verification || !output.verified_stage || output.full_proof_validated.has_value()),
            "queued migration output lacks its reviewed verification evidence");
        require(!review.verification || !output.verified_stage || !journal.policy_digest.empty() ||
            !replot_requires_proof(*review.verification, output.batch) || output.full_proof_validated.value_or(false),
            "queued migration sample lacks its completed full proof");
    }
    bool const active = journal.batch < review.batches.size();
    auto const first = active ? review.batches[size_t(journal.batch)].first_source : review.sources.size();
    auto const count = active ? review.batches[size_t(journal.batch)].sources : 0;
    require(journal.retired_prefix == first && journal.retirements.size() == count,
        "private migration retirement progress differs from its reviewed source mapping");
}
inline ReplotJournal next_replot_revision(ReplotJournal const& current, std::string const& digest,
    std::function<void(ReplotJournal&)> const& change)
{
    auto next = current; require(next.revision < UINT64_MAX, "migration revision exhausted");
    ++next.revision; next.parent_digest = digest; change(next);
    require(valid_replot_revision(current, next), "invalid replot journal update");
    return next;
}
inline uint64_t replot_journal_allocation(uint64_t journal_bytes, uint64_t farming_bytes, uint64_t unit)
{
    using namespace farm::resource_detail;
    group_tools::require(unit, "invalid migration metadata alignment");
    auto align = [&](uint64_t bytes) { return multiply(bytes / unit + bool(bytes % unit), unit); };
    return add(add(multiply(align(add(journal_bytes, 65)), 2), multiply(align(add(farming_bytes, 65)), 2)), multiply(unit, 4));
}
inline bool replot_local_bindings_fit(uint64_t journal_bytes, uint64_t farming_bytes, uint64_t local_bytes, uint64_t unit, uint64_t allowance)
{
    using namespace farm::resource_detail;
    auto const required = replot_journal_allocation(journal_bytes, farming_bytes, unit);
    auto const bytes = add(local_bytes, 65);
    auto const local = add(multiply(multiply(bytes / unit + bool(bytes % unit), unit), 2), multiply(unit, 2));
    return required <= allowance && local <= allowance - required;
}
inline bool replot_policy_boundary_matches(ReplotJournal const& journal, std::string const& journal_digest, ReplotRevalidation const& successor)
{
    return journal.phase < ReplotPhase::handoff && successor.parent_policy_digest == journal.policy_digest && successor.journal_digest == journal_digest;
}
inline bool replot_policy_activation(ReplotJournal const& journal, std::string const& journal_digest,
    std::string const& policy_digest, ReplotRevalidation const* successor)
{
    if (journal.policy_digest == policy_digest) return false;
    require(successor && !policy_digest.empty(), "migration has an active reviewed successor; use its matching --from-plan file");
    require(replot_policy_boundary_matches(journal, journal_digest, *successor),
        "migration advanced after this successor was reviewed; save a new revalidation at the current stopped boundary");
    return true;
}
inline char const* replot_policy_binding(ReplotJournal const& journal, std::string const& journal_digest,
    std::string const& policy_digest, ReplotRevalidation const* successor)
{
    if (journal.policy_digest == policy_digest) return policy_digest.empty() ? "origin" : "active";
    if (policy_digest.empty()) return "origin-with-active-successor";
    if (!successor || successor->parent_policy_digest != journal.policy_digest) return "different-active-policy";
    return replot_policy_boundary_matches(journal, journal_digest, *successor) ? "pending" : "stale-boundary";
}
struct ReplotReceipt
{
    std::string review_digest, previous_digest;
    uint64_t batch = 0, first_candidate = 0, candidates = 0;
    stream_state::Stamp replacement;
    file_io::DurableIdentity replacement_identity;
    std::optional<bool> full_proof_validated;
    bool lane_chain = false;
};
inline std::string replot_receipt_text(ReplotReceipt const& receipt)
{
    require(replot_digest(receipt.review_digest) && (receipt.lane_chain ? receipt.previous_digest.empty() || replot_digest(receipt.previous_digest) :
            receipt.batch ? replot_digest(receipt.previous_digest) : receipt.previous_digest.empty()) && (!receipt.lane_chain || receipt.full_proof_validated.has_value()) &&
        receipt.candidates <= replot_maximum_batch_sources && receipt.first_candidate <= farming_detail::maximum_files - receipt.candidates &&
        receipt.replacement.bytes && file_io::valid_durable_identity(receipt.replacement_identity) &&
        (!file_io::has_durable_identity(receipt.replacement_identity) || receipt.replacement.inode), "invalid replot receipt");
    auto const& stamp = receipt.replacement;
    std::ostringstream out;
    bool const durable = file_io::has_durable_identity(receipt.replacement_identity);
    out << (receipt.lane_chain ? "ordinary-group-replot-receipt-v4\n" : receipt.full_proof_validated ? "ordinary-group-replot-receipt-v3\n" : durable ? "ordinary-group-replot-receipt-v2\n" : "ordinary-group-replot-receipt-v1\n") << std::quoted(receipt.review_digest) << ' '
        << receipt.batch << ' ' << std::quoted(receipt.previous_digest) << ' ' << receipt.first_candidate << ' ' << receipt.candidates << '\n'
        << stamp.device << ' ' << stamp.inode << ' ' << stamp.bytes << ' ' << stamp.mtime << ' ' << stamp.ctime << ' '
        << stamp.mtime_ns << ' ' << stamp.ctime_ns << '\n';
    if (durable || receipt.full_proof_validated) write_replot_native_identity(out, receipt.replacement_identity);
    if (receipt.full_proof_validated) out << *receipt.full_proof_validated << '\n';
    return out.str();
}
inline ReplotReceipt parse_replot_receipt(std::string const& body)
{
    require(body.size() <= 4096, "replot receipt exceeds its bound");
    std::istringstream in(body); std::string version; std::getline(in, version);
    require(version == "ordinary-group-replot-receipt-v1" || version == "ordinary-group-replot-receipt-v2" ||
        version == "ordinary-group-replot-receipt-v3" || version == "ordinary-group-replot-receipt-v4", "unsupported replot receipt version");
    ReplotReceipt result; result.lane_chain = version == "ordinary-group-replot-receipt-v4"; auto& stamp = result.replacement;
    in >> std::quoted(result.review_digest) >> result.batch >> std::quoted(result.previous_digest) >> result.first_candidate >> result.candidates
        >> stamp.device >> stamp.inode >> stamp.bytes >> stamp.mtime >> stamp.ctime >> stamp.mtime_ns >> stamp.ctime_ns;
    require(in && stamp.mtime_ns < 1000000000 && stamp.ctime_ns < 1000000000, "invalid replot receipt stamp");
    if (version != "ordinary-group-replot-receipt-v1")
    { result.replacement_identity = read_replot_native_identity(in); require(version == "ordinary-group-replot-receipt-v3" || result.lane_chain || file_io::has_durable_identity(result.replacement_identity), "migration native receipt schema has no object evidence"); }
    if (version == "ordinary-group-replot-receipt-v3" || result.lane_chain)
    { unsigned proof = 0; in >> proof; require(in && proof <= 1, "invalid migration receipt full-proof evidence"); result.full_proof_validated = bool(proof); }
    in >> std::ws; require(in.eof(), "unexpected replot receipt data");
    (void)replot_receipt_text(result); return result;
}
inline void validate_replot_receipt(ReplotReview const& review, std::string const& review_digest,
    uint64_t batch, std::string const& previous, ReplotReceipt const& receipt)
{
    auto const& plan = review.batches.at(size_t(batch));
    require(!review.verification || receipt.full_proof_validated.has_value(),
        "explicit migration receipt lacks its completed verification evidence");
    require(!receipt.lane_chain && receipt.review_digest == review_digest && receipt.batch == batch && receipt.previous_digest == previous &&
        receipt.first_candidate == plan.first_source && receipt.candidates == plan.sources,
        "migration receipt chain differs from the review");
}
inline void complete_replot_batch(ReplotReview const& review, ReplotReceipt const& receipt,
    std::string const& receipt_digest, ReplotJournal& state)
{
    validate_replot_progress(review, state);
    validate_replot_receipt(review, state.review_digest, state.batch, state.receipt_digest, receipt);
    require(state.ready && *state.ready == receipt.replacement && state.ready_identity == receipt.replacement_identity &&
        state.full_proof_validated == receipt.full_proof_validated && replot_digest(receipt_digest) &&
        std::all_of(state.retirements.begin(), state.retirements.end(), [](auto value) { return value.state == ReplotRetirement::retired; }),
        "migration batch cannot complete before its replacement and retirements");
    state.retired_prefix += state.retirements.size(); ++state.batch; state.receipt_digest = receipt_digest;
    state.verified_stage.reset(); state.published.reset(); state.ready.reset(); state.retirements.clear();
    state.full_proof_validated.reset();
    state.verified_stage_identity = {}; state.published_identity = {}; state.ready_identity = {};
    if (state.batch < review.batches.size()) state.retirements.resize(size_t(review.batches[size_t(state.batch)].sources));
}
inline void validate_replot_completion(ReplotReview const& review, ReplotJournal const& state, uint64_t batch,
    std::string const& previous, ReplotReceipt const& receipt)
{
    auto const& plan = review.batches.at(size_t(batch));
    require(receipt.lane_chain && receipt.review_digest == state.review_digest && receipt.batch == batch &&
        receipt.first_candidate == plan.first_source && receipt.candidates == plan.sources && receipt.previous_digest == previous &&
        receipt.full_proof_validated.has_value(), "migration destination completion differs from its immutable assignment or receipt chain");
}
inline void complete_replot_deferred_batch(ReplotReview const& review, ReplotReceipt const& completed, ReplotReceipt const& receipt,
    std::string const& receipt_digest, ReplotJournal& state)
{
    validate_replot_progress(review, state);
    validate_replot_receipt(review, state.review_digest, state.batch, state.receipt_digest, receipt);
    require(replot_lane_completed(state, state.batch) && completed.lane_chain && completed.batch == state.batch &&
        completed.review_digest == state.review_digest && completed.first_candidate == receipt.first_candidate && completed.candidates == receipt.candidates &&
        completed.replacement == receipt.replacement && completed.replacement_identity == receipt.replacement_identity &&
        completed.full_proof_validated == receipt.full_proof_validated && replot_digest(receipt_digest),
        "global migration receipt lacks its exact confirmed destination completion");
    require(state.ready || (!state.verified_stage && !state.published && std::all_of(state.retirements.begin(), state.retirements.end(),
        [](auto const& progress) { return progress.state == ReplotRetirement::active && !progress.predelete; })), "deferred completion contradicts active source retirement authority");
    state.retired_prefix += state.retirements.size(); ++state.batch; state.receipt_digest = receipt_digest;
    state.verified_stage.reset(); state.published.reset(); state.ready.reset(); state.retirements.clear(); state.full_proof_validated.reset();
    state.verified_stage_identity = {}; state.published_identity = {}; state.ready_identity = {};
    if (state.batch < review.batches.size()) state.retirements.resize(size_t(review.batches[size_t(state.batch)].sources));
}
// Local convenience defaults are separate from portable observations and never
// replace reader, resource, native-object or journal authorization checks.
struct ReplotLocalBindings
{
    std::string review_digest, job_id, host, user;
    std::string adapter, root, executable, profile;
    std::string policy_digest;
};
inline std::string replot_local_bindings_text(ReplotLocalBindings const& local)
{
    require(replot_digest(local.review_digest) && replot_digest(local.job_id) && !local.host.empty() && !local.user.empty() &&
        (local.adapter == "chia" || local.adapter == "gigahorse" || local.adapter == "drchia") &&
        (local.policy_digest.empty() || replot_digest(local.policy_digest)), "invalid private replot bindings");
    for (auto const* path : {&local.root, &local.executable, &local.profile})
        require(path->empty() || file_io::native_path(*path).is_absolute(), "private replot binding paths must be absolute");
    std::ostringstream out; out << "ordinary-group-replot-local-v1\n" << std::quoted(local.review_digest) << ' '
        << std::quoted(local.job_id) << ' ' << std::quoted(local.host) << ' ' << std::quoted(local.user) << '\n'
        << std::quoted(local.adapter) << ' ' << std::quoted(local.root) << ' ' << std::quoted(local.executable) << ' ' << std::quoted(local.profile)
        << ' ' << std::quoted(local.policy_digest) << '\n';
    auto result = out.str(); require(result.size() <= 8192, "private replot bindings exceed their allocation allowance"); return result;
}
inline ReplotLocalBindings parse_replot_local_bindings(std::string const& body)
{
    require(body.size() <= 8192, "private replot bindings exceed their allocation allowance");
    std::istringstream in(body); std::string version; std::getline(in, version);
    require(version == "ordinary-group-replot-local-v1", "unsupported private replot bindings version");
    ReplotLocalBindings result; in >> std::quoted(result.review_digest) >> std::quoted(result.job_id) >> std::quoted(result.host) >> std::quoted(result.user)
        >> std::quoted(result.adapter) >> std::quoted(result.root) >> std::quoted(result.executable) >> std::quoted(result.profile) >> std::quoted(result.policy_digest);
    require(bool(in), "incomplete private replot bindings"); in >> std::ws; require(in.eof(), "unexpected private replot binding data");
    (void)replot_local_bindings_text(result); return result;
}
// Direct fill keeps its immutable review and local connection defaults in one
// owner-only record. A crash before the first journal cannot lose half of the
// preparation. Portable reviews continue to exclude these local bindings.
struct ReplotJob
{
    std::string review, profile;
    ReplotLocalBindings bindings;
    std::string seed, seed_owner, farmer, pool;
    bool pool_public_key = false;
};
inline std::string replot_job_text(ReplotJob const& job)
{
    require(stream_state::digest(job.review) == job.bindings.review_digest && !job.profile.empty(),
        "private migration job does not match its review");
    require(replot_digest(job.seed) && farm::inventory_detail::hex(job.farmer, 48) &&
        farm::inventory_detail::hex(job.pool, job.pool_public_key ? 48 : 32), "invalid private migration key identity");
    std::ostringstream out; out << "ordinary-group-replot-job-v1\n" << std::quoted(job.review) << '\n'
        << std::quoted(job.profile) << '\n' << std::quoted(replot_local_bindings_text(job.bindings)) << '\n'
        << std::quoted(job.seed) << ' ' << std::quoted(job.seed_owner) << ' ' << std::quoted(job.farmer) << ' '
        << std::quoted(job.pool) << ' ' << job.pool_public_key << '\n';
    return out.str();
}
inline ReplotJob parse_replot_job(std::string const& body, uint64_t memory_limit)
{
    check_farming_state_size(body.size(), memory_limit);
    std::istringstream in(body); std::string version, bindings; std::getline(in, version);
    require(version == "ordinary-group-replot-job-v1", "unsupported private migration job version");
    ReplotJob job; in >> std::quoted(job.review) >> std::quoted(job.profile) >> std::quoted(bindings)
        >> std::quoted(job.seed) >> std::quoted(job.seed_owner) >> std::quoted(job.farmer) >> std::quoted(job.pool) >> job.pool_public_key;
    require(bool(in), "incomplete private migration job"); in >> std::ws; require(in.eof(), "unexpected private migration job data");
    job.bindings = parse_replot_local_bindings(bindings);
    require(stream_state::digest(job.review) == job.bindings.review_digest && !job.profile.empty(),
        "private migration job does not match its review");
    require(replot_digest(job.seed) && farm::inventory_detail::hex(job.farmer, 48) &&
        farm::inventory_detail::hex(job.pool, job.pool_public_key ? 48 : 32), "invalid private migration key identity");
    return job;
}
inline std::filesystem::path replot_authority_directory(std::string const& review_digest)
{
    require(replot_digest(review_digest), "invalid migration authority identity");
#ifdef _WIN32
    auto const* base = ::_wgetenv(L"LOCALAPPDATA");
    require(base && *base, "LOCALAPPDATA is required for durable migration coordination");
    return std::filesystem::path(base) / "xchplot2" / "migrations" / review_digest;
#else
    auto const* state = std::getenv("XDG_STATE_HOME");
    if (state && *state) return file_io::native_path(state) / "xchplot2" / "migrations" / review_digest;
    auto const* home = std::getenv("HOME"); require(home && *home, "HOME or XDG_STATE_HOME is required for durable migration coordination");
    return file_io::native_path(home) / ".local" / "state" / "xchplot2" / "migrations" / review_digest;
#endif
}
inline std::string replot_authority_user()
{
#ifdef _WIN32
    pos2gpu::PrivateFileSecurity security; std::wstring const sid(security.sid_text); return {sid.begin(), sid.end()};
#else
    return std::to_string(::geteuid());
#endif
}
struct ReplotJournalReplica
{
    std::filesystem::path path;
    std::shared_ptr<FarmingMetadataDirectory> metadata;
    std::optional<stream_state::Stamp> published, pending;
    std::string digest, pending_digest;
    bool available = true;
};
struct ReplotJournalSet
{
    ReplotJournal state;
    std::string digest;
    std::vector<ReplotJournalReplica> replicas;
    bool present = false, authority_missing = false;
    std::optional<ReplotJournalReplica> authority;
};
inline ReplotJournalSet load_replot_journals(std::span<std::filesystem::path const> directories,
    std::string const& review_digest, uint64_t memory_limit, bool allow_missing = false)
{
    require(!directories.empty() && directories.size() <= 1024, "invalid replot replica scope");
    ReplotJournalSet result;
    bool const legacy = farming_metadata_legacy(directories);
    auto authority_root = replot_authority_directory(review_digest);
    auto authority_metadata = std::make_shared<FarmingMetadataDirectory>(authority_root, false);
    if (authority_metadata->root_pin)
    {
        ReplotJournalReplica replica; replica.metadata = authority_metadata; replica.path = authority_metadata->path / "grow-replot.plan";
        for (bool pending : {false, true})
        {
            auto const path = pending ? file_io::native_path(file_io::path_text(replica.path) + ".next") : replica.path;
            file_io::Stat info{};
            if (file_io::inspect_link(path, &info)) { require(errno == ENOENT, "cannot inspect migration controller journal"); continue; }
            auto const file = read_farming_state(path, memory_limit / 4); auto state = parse_replot_journal(file.body, memory_limit / 4);
            require(state.review_digest == review_digest && state.authority_directory == file_io::path_text(authority_root) &&
                state.authority_identity == farming_directory_identity(authority_root) && state.authority_host == profile_detail::host_name() &&
                state.authority_user == replot_authority_user(), "migration controller authority changed its native directory, host or user");
            if (result.present)
                require(state.revision == result.state.revision ? file.digest == result.digest : valid_replot_revision(result.state, state),
                    "migration controller journal has conflicting or rolled-back revisions");
            result.state = std::move(state); result.digest = file.digest; result.present = true;
            if (pending) { replica.pending = file.stamp; replica.pending_digest = file.digest; }
            else { replica.published = file.stamp; replica.digest = file.digest; }
        }
        if (result.present) result.authority = std::move(replica);
    }
    bool const authoritative = result.authority.has_value();
    std::optional<ReplotJournal> previous;
    uint64_t minimum = UINT64_MAX, maximum = 0;
    for (auto const& directory : directories)
    {
        ReplotJournalReplica replica;
        replica.metadata = std::make_shared<FarmingMetadataDirectory>(directory, legacy);
        replica.path = replica.metadata->path / "grow-replot.plan";
        replica.available = bool(replica.metadata->root_pin);
        require(replica.available || allow_missing || authoritative, "all replot destinations must be available before controller authority is established");
        if (!replica.available) { result.replicas.push_back(std::move(replica)); continue; }
        for (bool pending : {false, true})
        {
            auto const path = pending ? file_io::native_path(file_io::path_text(replica.path) + ".next") : replica.path;
            file_io::Stat info{};
            if (file_io::inspect_link(path, &info)) { require(errno == ENOENT, "cannot inspect replot journal replica"); continue; }
            auto file = read_farming_state(path, memory_limit / 4);
            auto state = parse_replot_journal(file.body, memory_limit / 4);
            require(state.review_digest == review_digest, "replot journal belongs to another reviewed migration");
            if (authoritative)
            {
                require(state.revision <= result.state.revision, "destination journal is newer than its controller authority; preserve all copies and inspect the fork");
                require(state.revision < result.state.authority_revision ? state.authority_directory.empty() :
                    state.authority_directory == result.state.authority_directory && state.authority_identity == result.state.authority_identity &&
                    state.authority_host == result.state.authority_host && state.authority_user == result.state.authority_user && state.authority_revision == result.state.authority_revision,
                    "destination journal belongs to another migration controller authority");
                require(state.revision != result.state.revision || file.digest == result.digest, "destination journal conflicts with its controller authority");
                require(farming_state_text(state.farm) == farming_state_text(result.state.farm) && state.batch <= result.state.batch &&
                    state.retired_prefix <= result.state.retired_prefix && state.phase <= result.state.phase, "destination journal contradicts controller progress");
                if (!state.completed_lanes.empty())
                {
                    require(state.completed_lanes.size() == result.state.completed_lanes.size(), "destination completion authority is newer than its controller");
                    for (size_t lane = 0; lane < state.completed_lanes.size(); ++lane)
                    {
                        auto const& prior = state.completed_lanes[lane]; auto const& current = result.state.completed_lanes[lane];
                        require(prior.baseline_files == current.baseline_files && prior.baseline_digest == current.baseline_digest &&
                            prior.files <= current.files && (prior.files != current.files || prior.digest == current.digest),
                            "destination completion prefix conflicts with or is newer than its controller authority");
                    }
                }
                if (pending) { replica.pending = file.stamp; replica.pending_digest = std::move(file.digest); }
                else { replica.published = file.stamp; replica.digest = std::move(file.digest); }
                continue;
            }
            require(!result.present || state.revision == result.state.revision ||
                (state.revision > result.state.revision ? state.revision - result.state.revision == 1 : result.state.revision - state.revision == 1),
                "replot journal replicas have an invalid revision gap");
            minimum = std::min(minimum, state.revision); maximum = std::max(maximum, state.revision);
            require(maximum - minimum <= 1, "replot journal replicas span unrelated revisions");
            if (!result.present || state.revision > result.state.revision)
            {
                if (result.present) previous = std::move(result.state);
                result.state = std::move(state); result.digest = file.digest; result.present = true;
            }
            else if (state.revision == result.state.revision)
                require(file.digest == result.digest, "conflicting replot journal revisions");
            else if (previous)
                require(replot_journal_text(*previous) == file.body, "conflicting replot parent revisions");
            else previous = std::move(state);
            if (pending) { replica.pending = file.stamp; replica.pending_digest = std::move(file.digest); }
            else { replica.published = file.stamp; replica.digest = std::move(file.digest); }
        }
        result.replicas.push_back(std::move(replica));
    }
    if (previous) require(valid_replot_revision(*previous, result.state), "invalid replot journal parent transition");
    result.authority_missing = result.present && !result.state.authority_directory.empty() && !result.authority;
    require(!result.authority_missing || allow_missing, "migration controller authority is unavailable; restore its saved native state directory before recovery");
    return result;
}
inline bool replot_journals_unchanged(ReplotJournalSet const& saved)
{
    try
    {
        std::vector<ReplotJournalReplica const*> replicas;
        if (saved.authority) replicas.push_back(&*saved.authority);
        for (auto const& replica : saved.replicas) replicas.push_back(&replica);
        for (auto const* entry : replicas)
        {
            auto const& replica = *entry;
            if (!replica.available) continue;
            replica.metadata->check();
            for (bool pending : {false, true})
            {
                auto const path = pending ? file_io::native_path(file_io::path_text(replica.path) + ".next") : replica.path;
                auto const& expected = pending ? replica.pending : replica.published; file_io::Stat info{};
                if (expected) { if (stream_state::stamp(path) != *expected) return false; }
                else if (!file_io::inspect_link(path, &info) || errno != ENOENT) return false;
            }
        }
        return true;
    }
    catch (std::exception const&) { return false; }
}
inline void publish_replot_journals(ReplotJournalSet& saved, ReplotJournal const& state,
    uint64_t memory_limit, std::function<void()> const& check)
{
    validate_replot_journal(state);
    auto body = replot_journal_text(state); auto const digest = stream_state::digest(body);
    require(!saved.present || digest == saved.digest || valid_replot_revision(saved.state, state), "invalid replot journal update");
    require(saved.present || (state.revision == 1 && state.phase == ReplotPhase::preparing), "replot journal must begin with preparation");
    body += digest + '\n'; check_farming_state_size(body.size(), memory_limit / 4);
    auto inspect = [](auto const& path, std::optional<stream_state::Stamp> const& expected)
    {
        file_io::Stat info{};
        if (expected) require(stream_state::stamp(path) == *expected, "replot journal changed during migration");
        else require(file_io::inspect_link(path, &info) && errno == ENOENT, "unrelated replot journal appeared during migration");
    };
    require(!saved.authority_missing && (state.authority_directory.empty() || saved.authority), "migration publication requires its durable controller authority");
    check();
    std::vector<ReplotJournalReplica*> replicas;
    if (saved.authority) replicas.push_back(&*saved.authority);
    for (auto& replica : saved.replicas) if (replica.available) replicas.push_back(&replica);
    for (auto* entry : replicas)
    {
        auto& replica = *entry; replica.metadata->create();
        auto const pending = file_io::native_path(file_io::path_text(replica.path) + ".next");
        inspect(replica.path, replica.published); inspect(pending, replica.pending);
        if (replica.pending && replica.pending_digest != digest && saved.authority && &replica != &*saved.authority)
        {
            require(!file_io::unlink_file(pending), "cannot reclaim stale destination journal staging after controller reconciliation");
            stream_state::sync_path(pending.parent_path(), true); replica.pending.reset(); replica.pending_digest.clear();
        }
        if (replica.pending)
            require(replica.pending_digest == digest, "recover pending replot journal before a new revision");
        else if (!replica.published || replica.digest != digest)
        {
            stage_farming_state(replica.path, body, false);
            replica.pending = stream_state::stamp(pending); replica.pending_digest = digest;
        }
    }
    grow_failpoint("replot-journal-all-staged");
    for (auto* entry : replicas) if (entry->pending)
    {
        auto& replica = *entry;
        check(); replica.metadata->check(); auto const pending = file_io::native_path(file_io::path_text(replica.path) + ".next");
        inspect(replica.path, replica.published); inspect(pending, replica.pending);
        publish_farming_state(pending, replica.path, replica.published.has_value());
        replica.published = stream_state::stamp(replica.path); replica.digest = digest;
        replica.pending.reset(); replica.pending_digest.clear();
        grow_failpoint("replot-journal-one-published");
    }
    saved.state = state; saved.digest = digest; saved.present = true;
}
} // namespace group_grow
