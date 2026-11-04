// Private, replicated migration progress. The ordinary farming state owns all
// replacement keys and assignments from the first durable revision onward.
#pragma once
#include "host/GroupFarmState.hpp"
#include <functional>

namespace group_grow
{
enum class ReplotPhase : uint8_t { preparing, replacing, handoff, complete };
enum class ReplotRetirement : uint8_t { active, claim_intent, claimed, retired };
struct ReplotCandidateProgress
{
    ReplotRetirement state = ReplotRetirement::active;
    bool predelete = false;
    bool operator==(ReplotCandidateProgress const&) const = default;
};
inline constexpr size_t replot_maximum_batch_sources = 1024;
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
};
inline void validate_replot_review(ReplotReview const& review)
{
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
}
inline std::string replot_review_text(ReplotReview const& review)
{
    validate_replot_review(review);
    std::ostringstream out;
    out << "ordinary-group-replot-review-v1\n" << std::quoted(review.inventory_json) << '\n'
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
    return out.str();
}
inline ReplotReview parse_replot_review(std::string const& body, uint64_t memory_limit)
{
    require(body.size() < memory_limit / 16, "replot review exceeds its parsing memory budget");
    std::istringstream in(body); std::string version, farm; std::getline(in, version);
    require(version == "ordinary-group-replot-review-v1", "expected a saved replot review");
    ReplotReview result; size_t count = 0;
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
    require(bool(in), "incomplete replot review"); in >> std::ws; require(in.eof(), "unexpected replot review data");
    validate_replot_review(result); return result;
}
struct ReplotJournal
{
    std::string review_digest, parent_digest, receipt_digest;
    uint64_t revision = 1, batch = 0, retired_prefix = 0;
    ReplotPhase phase = ReplotPhase::preparing;
    FarmingState farm;
    // Only the active batch is replicated. The immutable reviewed catalog is
    // required for resume; completed batches have chained per-output receipts.
    std::vector<ReplotCandidateProgress> retirements;
    std::optional<stream_state::Stamp> verified_stage, published, ready;
};
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
inline void validate_replot_journal(ReplotJournal const& journal)
{
    require(replot_digest(journal.review_digest) && journal.revision &&
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
    require(!journal.published || (journal.verified_stage && stream_state::same_output(*journal.verified_stage, *journal.published)),
        "replot publication lacks its verified staged identity");
    require(!journal.ready || (journal.published && *journal.ready == *journal.published), "replot readiness lacks its immutable publication");
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
    out << "ordinary-group-replot-journal-v1\n" << std::quoted(journal.review_digest) << ' '
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
    return out.str();
}
inline ReplotJournal parse_replot_journal(std::string const& body, uint64_t memory_limit)
{
    require(body.size() < memory_limit / 8, "replot journal exceeds its parsing memory budget");
    std::istringstream in(body); std::string version, farm; unsigned phase = 0;
    std::getline(in, version); require(version == "ordinary-group-replot-journal-v1", "unsupported replot journal version");
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
    if (after.phase < before.phase || unsigned(after.phase) > unsigned(before.phase) + 1 ||
        after.batch < before.batch || after.batch > before.batch + 1) return false;
    // A completed receipt is durable before advancing the prefix. Its content
    // and exact next batch come from the immutable review at the controller.
    if (after.batch == before.batch + 1)
    {
        return before.phase == ReplotPhase::replacing && after.phase == before.phase && before.ready && !after.verified_stage && !after.published && !after.ready &&
            after.retired_prefix == before.retired_prefix + before.retirements.size() &&
            replot_digest(after.receipt_digest) && after.receipt_digest != before.receipt_digest &&
            std::all_of(before.retirements.begin(), before.retirements.end(), [](auto value) { return value.state == ReplotRetirement::retired; }) &&
            std::all_of(after.retirements.begin(), after.retirements.end(), [](auto value) { return value.state == ReplotRetirement::active && !value.predelete; });
    }
    if (before.retired_prefix != after.retired_prefix || before.receipt_digest != after.receipt_digest ||
        before.retirements.size() != after.retirements.size()) return false;
    size_t changes = size_t(after.phase != before.phase);
    for (auto const pair : {std::pair{&before.verified_stage, &after.verified_stage}, std::pair{&before.published, &after.published},
        std::pair{&before.ready, &after.ready}})
    {
        if (*pair.first != *pair.second)
        { if (*pair.first || !*pair.second || before.phase != ReplotPhase::replacing) return false; ++changes; }
    }
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
struct ReplotReceipt
{
    std::string review_digest, previous_digest;
    uint64_t batch = 0, first_candidate = 0, candidates = 0;
    stream_state::Stamp replacement;
};
inline std::string replot_receipt_text(ReplotReceipt const& receipt)
{
    require(replot_digest(receipt.review_digest) && (receipt.batch ? replot_digest(receipt.previous_digest) : receipt.previous_digest.empty()) &&
        receipt.candidates <= replot_maximum_batch_sources && receipt.first_candidate <= farming_detail::maximum_files - receipt.candidates &&
        receipt.replacement.bytes, "invalid replot receipt");
    auto const& stamp = receipt.replacement;
    std::ostringstream out;
    out << "ordinary-group-replot-receipt-v1\n" << std::quoted(receipt.review_digest) << ' '
        << receipt.batch << ' ' << std::quoted(receipt.previous_digest) << ' ' << receipt.first_candidate << ' ' << receipt.candidates << '\n'
        << stamp.device << ' ' << stamp.inode << ' ' << stamp.bytes << ' ' << stamp.mtime << ' ' << stamp.ctime << ' '
        << stamp.mtime_ns << ' ' << stamp.ctime_ns << '\n';
    return out.str();
}
inline ReplotReceipt parse_replot_receipt(std::string const& body)
{
    require(body.size() <= 4096, "replot receipt exceeds its bound");
    std::istringstream in(body); std::string version; std::getline(in, version);
    require(version == "ordinary-group-replot-receipt-v1", "unsupported replot receipt version");
    ReplotReceipt result; auto& stamp = result.replacement;
    in >> std::quoted(result.review_digest) >> result.batch >> std::quoted(result.previous_digest) >> result.first_candidate >> result.candidates
        >> stamp.device >> stamp.inode >> stamp.bytes >> stamp.mtime >> stamp.ctime >> stamp.mtime_ns >> stamp.ctime_ns;
    require(in && stamp.mtime_ns < 1000000000 && stamp.ctime_ns < 1000000000, "invalid replot receipt stamp");
    in >> std::ws; require(in.eof(), "unexpected replot receipt data");
    (void)replot_receipt_text(result); return result;
}
struct ReplotJournalReplica
{
    std::filesystem::path path;
    std::optional<stream_state::Stamp> published, pending;
    std::string digest, pending_digest;
};
struct ReplotJournalSet
{
    ReplotJournal state;
    std::string digest;
    std::vector<ReplotJournalReplica> replicas;
    bool present = false;
};
inline ReplotJournalSet load_replot_journals(std::span<std::filesystem::path const> directories,
    std::string const& review_digest, uint64_t memory_limit)
{
    require(!directories.empty() && directories.size() <= 1024, "invalid replot replica scope");
    ReplotJournalSet result;
    std::optional<ReplotJournal> previous;
    uint64_t minimum = UINT64_MAX, maximum = 0;
    for (auto const& directory : directories)
    {
        require(std::filesystem::is_directory(directory), "all replot destinations must be available");
        ReplotJournalReplica replica; replica.path = directory / "grow-replot.plan";
        for (bool pending : {false, true})
        {
            auto const path = pending ? file_io::native_path(file_io::path_text(replica.path) + ".next") : replica.path;
            file_io::Stat info{};
            if (file_io::inspect_link(path, &info)) { require(errno == ENOENT, "cannot inspect replot journal replica"); continue; }
            auto file = read_farming_state(path, memory_limit / 4);
            auto state = parse_replot_journal(file.body, memory_limit / 4);
            require(state.review_digest == review_digest, "replot journal belongs to another reviewed migration");
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
    return result;
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
    check();
    for (auto& replica : saved.replicas)
    {
        auto const pending = file_io::native_path(file_io::path_text(replica.path) + ".next");
        inspect(replica.path, replica.published); inspect(pending, replica.pending);
        if (replica.pending)
            require(replica.pending_digest == digest, "recover pending replot journal before a new revision");
        else if (!replica.published || replica.digest != digest)
        {
            stage_farming_state(replica.path, body, false);
            replica.pending = stream_state::stamp(pending); replica.pending_digest = digest;
        }
    }
    grow_failpoint("replot-journal-all-staged");
    for (auto& replica : saved.replicas) if (replica.pending)
    {
        check(); auto const pending = file_io::native_path(file_io::path_text(replica.path) + ".next");
        inspect(replica.path, replica.published); inspect(pending, replica.pending);
        publish_farming_state(pending, replica.path, replica.published.has_value());
        replica.published = stream_state::stamp(replica.path); replica.digest = digest;
        replica.pending.reset(); replica.pending_digest.clear();
        grow_failpoint("replot-journal-one-published");
    }
    saved.state = state; saved.digest = digest; saved.present = true;
}
} // namespace group_grow
