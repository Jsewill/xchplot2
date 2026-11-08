// Local completion of immutable, globally admitted destination grants. The
// controller holds DirectoryLock for every save/retirement; observations are
// last-known status, never authority to extend a grant or report live health.
#pragma once
#include "host/GroupFarmUpdate.hpp"

namespace group_grow
{
struct FarmingProgressSummary
{
    std::string observed_at, writer_state = "unknown";
    uint64_t completed_files = 0, completed_members = 0, pending_files = 0;
    uint64_t charged_bytes = 0, reserved_bytes = 0;
    bool operator==(FarmingProgressSummary const&) const = default;
};
struct FarmingProgressAssignment
{
    uint64_t first_file = 0, identity_resources = 1, identity_resource = 0, identity_epoch = 0;
    uint8_t k = 28, strength = 2;
    FarmingDrivePlan layout;
    bool operator==(FarmingProgressAssignment const&) const = default;
};
struct FarmingProgress
{
    std::string job_id, fingerprint, directory_identity, source_policy;
    uint64_t source_revision = 0, source_policy_revision = 0, sequence = 1, capacity_bytes = 0, progress_reserved_bytes = 0, headroom_bytes = 0;
    std::vector<uint16_t> granted_extents, extents;
    uint64_t granted_sealed_files = 0, sealed_files = 0;
    std::string granted_fill_stop, fill_stop;
    std::vector<FarmingProgressAssignment> assignments;
    FarmingProgressSummary summary;
};
enum class FarmingProgressOverlay { applied, covered };
struct FarmingProgressFile
{
    FarmingProgress progress;
    stream_state::Stamp stamp;
    uint64_t allocated_bytes = 0;
};
struct FarmingProgressNativeBinding
{
    std::string source_identity, target_identity;
};
namespace farming_progress_detail
{
namespace fs = std::filesystem;
inline constexpr char filename[] = "grow-farm.progress";
inline bool stop(std::string const& value)
{
    return value.empty() || value == "capacity" || value == "farming-limit" || value == "group-limit" ||
        value == "total-member-limit" || value == "metadata-capacity";
}
inline void text(std::string const& value, size_t limit)
{ require(value.size() <= limit && value.find('\0') == value.npos, "invalid farming progress text"); }
inline uint64_t members(std::span<uint16_t const> extents, uint64_t files)
{
    require(files <= extents.size(), "farming progress sealed prefix exceeds its grants");
    uint64_t total = 0;
    for (uint64_t file = 0; file < files; ++file) farming_detail::add(total, extents[size_t(file)]);
    return total;
}
inline FarmingProgressAssignment assignment(FarmingCohort const& value)
{
    FarmingProgressAssignment result;
    result.first_file = value.first_file; result.identity_resources = value.identity_resources;
    result.identity_resource = value.identity_resource; result.identity_epoch = value.identity_epoch;
    result.k = value.config.k; result.strength = value.config.strength;
    result.layout.schedule = value.layout.schedule; result.layout.meta_slots = value.layout.meta_slots;
    result.layout.meta_bits = value.layout.meta_bits; result.layout.meta_base = value.layout.meta_base;
    result.layout.resource_rank = value.layout.resource_rank; result.layout.resource_lanes = value.layout.resource_lanes;
    return result;
}
inline std::vector<FarmingProgressAssignment> assignments(FarmingState const& state, FarmingDirectory const& directory, uint64_t files)
{
    std::vector<FarmingProgressAssignment> result; result.reserve(directory.cohorts.size() + 1);
    for (size_t i = 0; i <= directory.cohorts.size(); ++i)
    {
        auto const cohort = i < directory.cohorts.size() ? directory.cohorts[i] : farming_cohort(state, directory, directory.layout_first);
        auto const last = std::min(files, i < directory.cohorts.size() ?
            (i + 1 < directory.cohorts.size() ? directory.cohorts[i + 1].first_file : directory.layout_first) : files);
        if (cohort.first_file < last) result.push_back(assignment(cohort));
    }
    return result;
}
inline bool same_assignment(FarmingProgressAssignment const& a, FarmingProgressAssignment const& b, uint64_t file)
{
    return a.k == b.k && a.strength == b.strength && a.identity_resources == b.identity_resources &&
        a.identity_resource == b.identity_resource && a.identity_epoch == b.identity_epoch &&
        a.layout.family(file - a.first_file) == b.layout.family(file - b.first_file) &&
        a.layout.meta(file - a.first_file) == b.layout.meta(file - b.first_file);
}
inline bool protected_assignments(FarmingState const& state, FarmingDirectory const& directory,
    FarmingProgress const& progress, uint64_t files)
{
    if (!files) return true;
    auto current = assignments(state, directory, files);
    if (current.empty() || progress.assignments.empty()) return false;
    size_t before = 0, after = 0;
    for (uint64_t file = 0; file < files; ++file)
    {
        while (before + 1 < progress.assignments.size() && progress.assignments[before + 1].first_file <= file) ++before;
        while (after + 1 < current.size() && current[after + 1].first_file <= file) ++after;
        if (!same_assignment(progress.assignments[before], current[after], file)) return false;
    }
    return true;
}
inline void validate(FarmingProgress const& value)
{
    require(farming_update_digest(value.job_id) && farming_update_digest(value.source_policy) &&
        !value.fingerprint.empty() && !value.directory_identity.empty() && value.source_revision && value.sequence &&
        value.source_policy_revision <= value.source_revision, "invalid farming progress binding");
    text(value.fingerprint, 256); text(value.directory_identity, 256); text(value.summary.observed_at, 128);
    text(value.summary.writer_state, 128); require(!value.summary.writer_state.empty(), "farming progress lacks writer observation state");
    require(value.granted_extents.size() <= farming_detail::maximum_files && value.extents.size() <= value.granted_extents.size() &&
        value.granted_sealed_files <= value.sealed_files && value.sealed_files <= value.extents.size() &&
        value.granted_sealed_files <= value.granted_extents.size() && stop(value.granted_fill_stop) && stop(value.fill_stop) &&
        (value.fill_stop.empty() || value.sealed_files == value.extents.size()) &&
        (value.extents.size() == value.granted_extents.size() || !value.fill_stop.empty()),
        "farming progress exceeds its globally admitted prefix");
    require(value.granted_fill_stop.empty() || (value.granted_fill_stop == value.fill_stop &&
        value.granted_extents == value.extents && value.granted_sealed_files == value.sealed_files),
        "farming progress reopened a stopped grant");
    for (size_t file = 0; file < value.granted_extents.size(); ++file)
    {
        require(value.granted_extents[file], "farming progress contains an empty global grant");
        if (file >= value.extents.size()) continue;
        auto const extent = value.extents[file];
        require(extent && extent <= value.granted_extents[file] &&
            (file < value.sealed_files || extent == value.granted_extents[file]) &&
            (file >= value.granted_sealed_files || extent == value.granted_extents[file]),
            "farming progress enlarged or changed an immutable member grant");
    }
    require(value.assignments.size() <= 1025 && bool(value.assignments.size()) == bool(value.granted_extents.size()),
        "invalid farming progress identity coverage");
    for (size_t i = 0; i < value.assignments.size(); ++i)
    {
        auto const& row = value.assignments[i]; auto const& plan = row.layout;
        FarmingConfig config; config.k = row.k; config.strength = row.strength; farming_detail::validate(config, {});
        require((!i ? !row.first_file : value.assignments[i - 1].first_file < row.first_file) &&
            row.first_file < value.granted_extents.size() && row.identity_resources && row.identity_resources <= 1024 &&
            row.identity_resource < row.identity_resources && row.identity_epoch < value.source_revision &&
            plan.resource_lanes && plan.resource_lanes <= 1024 && plan.resource_rank < plan.resource_lanes &&
            plan.meta_bits <= 8 && plan.meta_slots && plan.meta_slots <= (1u << plan.meta_bits),
            "invalid farming progress identity assignment");
    }
    require(value.summary.completed_files <= value.sealed_files &&
        value.summary.completed_members == members(value.extents, value.summary.completed_files) &&
        value.summary.pending_files <= value.extents.size() - value.summary.completed_files &&
        value.summary.charged_bytes <= value.capacity_bytes &&
        value.summary.reserved_bytes <= value.capacity_bytes - value.summary.charged_bytes,
        "farming progress observation contradicts its durable prefix or capacity grant");
}
inline bool immutable(FarmingProgress const& a, FarmingProgress const& b)
{
    return a.job_id == b.job_id && a.fingerprint == b.fingerprint && a.directory_identity == b.directory_identity &&
        a.source_policy == b.source_policy && a.source_revision == b.source_revision && a.source_policy_revision == b.source_policy_revision &&
        a.capacity_bytes == b.capacity_bytes && a.progress_reserved_bytes == b.progress_reserved_bytes && a.headroom_bytes == b.headroom_bytes &&
        a.granted_extents == b.granted_extents &&
        a.granted_sealed_files == b.granted_sealed_files && a.granted_fill_stop == b.granted_fill_stop && a.assignments == b.assignments;
}
inline bool monotone(FarmingProgress const& before, FarmingProgress const& after)
{
    if (!immutable(before, after) || before.sequence == UINT64_MAX || after.sequence != before.sequence + 1 ||
        after.sealed_files < before.sealed_files || after.extents.size() > before.extents.size() ||
        (!before.fill_stop.empty() && (after.fill_stop != before.fill_stop || after.extents != before.extents))) return false;
    for (size_t file = 0; file < before.sealed_files; ++file)
        if (after.extents[file] != before.extents[file]) return false;
    return true;
}
inline bool native_binding(FarmingProgress const& value, std::string const& target,
    std::optional<FarmingProgressNativeBinding> const& binding)
{
    return binding ? !binding->source_identity.empty() && !binding->target_identity.empty() &&
        value.directory_identity == binding->source_identity && target == binding->target_identity : value.directory_identity == target;
}
inline void binding(FarmingState const& state, size_t lane, FarmingProgress const& value, std::string const& policy = {},
    std::optional<FarmingProgressNativeBinding> const& native = {})
{
    validate(value);
    require(state.version >= 8 && lane < state.directories.size() && state.job_id == value.job_id && state.fingerprint == value.fingerprint &&
        native_binding(value, state.directories[lane].directory_identity, native) && value.source_revision <= state.revision,
        "farming progress belongs to a foreign job, destination or future revision");
    require((policy.empty() ? farming_policy_digest(state) : policy) == value.source_policy || state.policy_revision > value.source_revision,
        "farming progress belongs to a foreign source policy");
}
inline bool covered(FarmingState const& state, size_t lane, FarmingProgress const& value)
{
    auto const& current = state.directories[lane];
    if (current.sealed_files < value.sealed_files || current.extents.size() < value.sealed_files ||
        !protected_assignments(state, current, value, value.sealed_files)) return false;
    for (size_t file = 0; file < value.sealed_files; ++file)
        if (current.extents[file] != value.extents[file]) return false;
    return value.fill_stop.empty() || (current.fill_stop == value.fill_stop && current.extents == value.extents) ||
        state.policy_revision > value.source_revision;
}
inline bool present(fs::path const& path)
{
    file_io::Stat info{};
    if (!file_io::inspect_link(path, &info)) return true;
    require(errno == ENOENT, "cannot inspect farming progress record"); return false;
}
inline uint64_t charge(fs::path const& path)
{
    file_io::Stat info{};
    require(!file_io::inspect_link(path, &info) && file_io::is_regular(info.st_mode) && info.st_size >= 0 &&
        info.st_blocks >= 0 && uint64_t(info.st_blocks) <= UINT64_MAX / 512, "invalid farming progress allocation");
    return std::max(uint64_t(info.st_size), uint64_t(info.st_blocks) * 512);
}
inline std::vector<fs::path> partials(FarmingMetadataDirectory const& metadata)
{
    std::vector<fs::path> result;
    if (!metadata.root_pin || (!metadata.legacy && !metadata.private_pin)) return result;
    for (auto const& entry : fs::directory_iterator(metadata.path))
    {
        auto const name = file_io::path_text(entry.path().filename());
        if (name.starts_with(std::string(filename) + ".partial.") || name.starts_with(std::string(filename) + ".next.partial."))
        {
            require(result.size() < 1024, "too many interrupted farming progress writes");
            (void)charge(entry.path()); result.push_back(entry.path());
        }
    }
    return result;
}
inline void remove_unpublished(FarmingMetadataDirectory const& metadata)
{
    auto paths = partials(metadata);
    auto const pending = metadata.path / (std::string(filename) + ".next");
    if (present(pending)) { (void)charge(pending); paths.push_back(pending); }
    for (auto const& path : paths) require(!file_io::unlink_file(path), "cannot remove interrupted farming progress write");
    if (!paths.empty()) stream_state::sync_path(metadata.path, true);
}
} // namespace farming_progress_detail

inline FarmingProgress capture_farming_progress(FarmingState const& state, size_t lane, std::string const& policy = {})
{
    require(state.version >= 8 && lane < state.directories.size(), "local farming progress requires a current saved job");
    auto const& directory = state.directories[lane]; FarmingProgress result;
    result.job_id = state.job_id; result.fingerprint = state.fingerprint; result.directory_identity = directory.directory_identity;
    result.source_policy = policy.empty() ? farming_policy_digest(state) : policy;
    result.source_revision = state.revision; result.source_policy_revision = state.policy_revision;
    result.capacity_bytes = directory.capacity_bytes; result.progress_reserved_bytes = state.progress_reserved_bytes;
    result.headroom_bytes = state.headroom;
    result.granted_extents = result.extents = directory.extents;
    result.granted_sealed_files = result.sealed_files = directory.sealed_files;
    result.granted_fill_stop = result.fill_stop = directory.fill_stop;
    result.assignments = farming_progress_detail::assignments(state, directory, directory.extents.size());
    farming_progress_detail::validate(result); return result;
}
inline FarmingProgress advance_farming_progress(FarmingProgress const& previous, FarmingDirectory const& directory,
    FarmingProgressSummary summary)
{
    farming_progress_detail::validate(previous);
    require(previous.sequence < UINT64_MAX && directory.directory_identity == previous.directory_identity,
        "farming progress writer changed destination or exhausted its sequence");
    auto result = previous; ++result.sequence;
    result.extents = directory.extents; result.sealed_files = directory.sealed_files; result.fill_stop = directory.fill_stop;
    result.summary = std::move(summary); farming_progress_detail::validate(result);
    require(farming_progress_detail::monotone(previous, result), "farming progress replacement is not monotone"); return result;
}
inline FarmingProgressOverlay validate_farming_progress(FarmingState const& state, size_t lane, FarmingProgress const& progress,
    std::string const& policy = {})
{
    using namespace farming_progress_detail;
    binding(state, lane, progress, policy);
    if (covered(state, lane, progress)) return FarmingProgressOverlay::covered;
    auto const& current = state.directories[lane];
    require(current.sealed_files >= progress.granted_sealed_files && current.sealed_files <= progress.sealed_files &&
        current.extents.size() >= progress.granted_extents.size() && current.capacity_bytes >= progress.capacity_bytes &&
        state.progress_reserved_bytes >= progress.progress_reserved_bytes &&
        (progress.fill_stop.empty() || current.extents.size() == progress.granted_extents.size()) &&
        current.fill_stop.empty() && protected_assignments(state, current, progress, progress.granted_extents.size()),
        "farming progress no longer has its globally protected assignments and capacity grants");
    for (size_t file = 0; file < progress.granted_extents.size(); ++file)
        require(current.extents[file] == (file < current.sealed_files ? progress.extents[file] : progress.granted_extents[file]),
            "farming progress conflicts with a changed global member grant");
    return FarmingProgressOverlay::applied;
}
inline FarmingProgressOverlay overlay_farming_progress(FarmingState& state, size_t lane, FarmingProgress const& progress,
    std::string const& policy = {})
{
    auto const checked = validate_farming_progress(state, lane, progress, policy);
    if (checked == FarmingProgressOverlay::covered) return checked;
    auto& current = state.directories[lane];
    for (size_t file = 0; file < progress.extents.size(); ++file) current.extents[file] = progress.extents[file];
    if (!progress.fill_stop.empty()) { current.extents.resize(progress.extents.size()); current.fill_stop = progress.fill_stop; }
    current.sealed_files = progress.sealed_files;
    return FarmingProgressOverlay::applied;
}
inline uint64_t farming_progress_size_bound(FarmingState const& state, size_t lane)
{
    require(lane < state.directories.size() && state.directories[lane].extents.size() <= farming_detail::maximum_files &&
        state.directories[lane].cohorts.size() <= 1024, "farming progress size exceeds its work bound");
    auto const& directory = state.directories[lane];
    return 8192 + directory.extents.size() * 16 + (directory.cohorts.size() + 1) * 512;
}
inline uint64_t farming_progress_memory_bound(FarmingState const& state, size_t lane)
{
    auto const bytes = farming_progress_size_bound(state, lane); auto const& directory = state.directories[lane];
    return 64 * 1024 + bytes * 4 + directory.extents.size() * sizeof(uint16_t) * 4 +
        (directory.cohorts.size() + 1) * sizeof(FarmingProgressAssignment) * 4;
}
inline uint64_t farming_progress_memory_bound(FarmingProgress const& progress)
{
    farming_progress_detail::validate(progress);
    uint64_t const bytes = 8192 + progress.granted_extents.size() * 16 + progress.assignments.size() * 512;
    return 64 * 1024 + bytes * 4 +
        (progress.granted_extents.capacity() + progress.extents.capacity()) * sizeof(uint16_t) * 2 +
        progress.assignments.capacity() * sizeof(FarmingProgressAssignment) * 4;
}
inline std::string farming_progress_text(FarmingProgress const& value)
{
    farming_progress_detail::validate(value);
    std::ostringstream out;
    out << "ordinary-group-farming-progress-v1\n" << std::quoted(farming_current_model_identity) << ' '
        << std::quoted(farming_cohort_key_identity) << '\n' << std::quoted(value.job_id) << ' ' << std::quoted(value.fingerprint) << ' '
        << std::quoted(value.directory_identity) << ' ' << std::quoted(value.source_policy) << '\n'
        << value.source_revision << ' ' << value.source_policy_revision << ' ' << value.sequence << ' ' << value.capacity_bytes << ' '
        << value.progress_reserved_bytes << ' ' << value.headroom_bytes << '\n';
    auto extents = [&](std::vector<uint16_t> const& values, uint64_t sealed, std::string const& reason)
    {
        out << sealed << ' ' << std::quoted(reason) << ' ' << values.size() << '\n';
        for (auto value : values) out << value << ' ';
        out << '\n';
    };
    extents(value.granted_extents, value.granted_sealed_files, value.granted_fill_stop);
    extents(value.extents, value.sealed_files, value.fill_stop);
    out << value.assignments.size() << '\n';
    for (auto const& row : value.assignments)
    {
        auto const& p = row.layout;
        out << row.first_file << ' ' << row.identity_resources << ' ' << row.identity_resource << ' ' << row.identity_epoch << ' '
            << unsigned(row.k) << ' ' << unsigned(row.strength) << ' ' << p.schedule << ' ' << p.meta_slots << ' '
            << unsigned(p.meta_bits) << ' ' << unsigned(p.meta_base) << ' ' << p.resource_rank << ' ' << p.resource_lanes << '\n';
    }
    auto const& summary = value.summary;
    out << std::quoted(summary.observed_at) << ' ' << std::quoted(summary.writer_state) << ' ' << summary.completed_files << ' '
        << summary.completed_members << ' ' << summary.pending_files << ' ' << summary.charged_bytes << ' ' << summary.reserved_bytes << '\n';
    return out.str();
}
inline FarmingProgress parse_farming_progress(std::string const& body, uint64_t memory_limit)
{
    require(body.size() < memory_limit / 4, "farming progress exceeds its parsing memory budget");
    std::istringstream in(body); std::string magic, model, keys; std::getline(in, magic); in >> std::quoted(model) >> std::quoted(keys);
    require(magic == "ordinary-group-farming-progress-v1" && model == farming_current_model_identity && keys == farming_cohort_key_identity,
        "unsupported farming progress journal");
    FarmingProgress result;
    in >> std::quoted(result.job_id) >> std::quoted(result.fingerprint) >> std::quoted(result.directory_identity) >> std::quoted(result.source_policy)
        >> result.source_revision >> result.source_policy_revision >> result.sequence >> result.capacity_bytes >> result.progress_reserved_bytes >> result.headroom_bytes;
    uint64_t expanded = body.size() * 4 + 64 * 1024;
    require(expanded <= memory_limit, "farming progress exceeds its parsing memory budget");
    auto extents = [&](std::vector<uint16_t>& values, uint64_t& sealed, std::string& reason)
    {
        uint64_t count = 0; in >> sealed >> std::quoted(reason) >> count;
        require(in && count <= farming_detail::maximum_files && count * sizeof(uint16_t) <= memory_limit - expanded,
            "farming progress extents exceed their parsing memory budget");
        expanded += count * sizeof(uint16_t); values.reserve(size_t(count));
        for (uint64_t file = 0; file < count; ++file)
        {
            unsigned value = 0; in >> value;
            require(in && value && value <= UINT16_MAX, "invalid farming progress member extent"); values.push_back(uint16_t(value));
        }
    };
    extents(result.granted_extents, result.granted_sealed_files, result.granted_fill_stop);
    extents(result.extents, result.sealed_files, result.fill_stop);
    uint64_t rows = 0; in >> rows;
    require(in && rows <= 1025 && rows * sizeof(FarmingProgressAssignment) <= memory_limit - expanded,
        "farming progress identity rows exceed their parsing memory budget");
    result.assignments.reserve(size_t(rows));
    for (uint64_t i = 0; i < rows; ++i)
    {
        FarmingProgressAssignment row; auto& p = row.layout;
        unsigned k = 0, strength = 0, slots = 0, bits = 0, meta = 0, rank = 0, lanes = 0;
        in >> row.first_file >> row.identity_resources >> row.identity_resource >> row.identity_epoch >> k >> strength >> p.schedule >> slots
            >> bits >> meta >> rank >> lanes;
        require(in && k <= UINT8_MAX && strength <= UINT8_MAX && slots <= UINT16_MAX && bits <= 8 && meta <= UINT8_MAX &&
            rank <= UINT16_MAX && lanes <= UINT16_MAX, "invalid farming progress identity row");
        row.k = uint8_t(k); row.strength = uint8_t(strength); p.meta_slots = uint16_t(slots); p.meta_bits = uint8_t(bits);
        p.meta_base = uint8_t(meta); p.resource_rank = uint16_t(rank); p.resource_lanes = uint16_t(lanes); result.assignments.push_back(row);
    }
    auto& summary = result.summary;
    in >> std::quoted(summary.observed_at) >> std::quoted(summary.writer_state) >> summary.completed_files >> summary.completed_members
        >> summary.pending_files >> summary.charged_bytes >> summary.reserved_bytes;
    require(bool(in), "incomplete farming progress journal"); in >> std::ws; require(in.eof(), "unexpected farming progress data");
    farming_progress_detail::validate(result); return result;
}
inline uint64_t farming_progress_allocation(FarmingMetadataDirectory const& metadata)
{
    using namespace farming_progress_detail;
    metadata.check(); uint64_t bytes = 0; auto const path = metadata.path / filename;
    if (present(path)) bytes = charge(path);
    auto const pending = metadata.path / (std::string(filename) + ".next");
    if (present(pending)) farming_detail::add(bytes, charge(pending));
    for (auto const& partial : partials(metadata)) farming_detail::add(bytes, charge(partial));
    metadata.check(); return bytes;
}
inline std::optional<FarmingProgressFile> load_farming_progress(FarmingMetadataDirectory const& metadata, uint64_t memory_limit,
    std::optional<FarmingProgressNativeBinding> const& native = {})
{
    using namespace farming_progress_detail;
    metadata.check(); auto const path = metadata.path / filename;
    if (!present(path)) return std::nullopt;
    auto file = read_farming_state(path, memory_limit); auto progress = parse_farming_progress(file.body, memory_limit);
    require(native_binding(progress, farming_directory_identity(metadata.root), native), "farming progress destination identity changed");
    auto const allocated = farming_progress_allocation(metadata);
    require(stream_state::stamp(path) == file.stamp, "farming progress changed while loading"); metadata.check();
    return FarmingProgressFile{std::move(progress), file.stamp, allocated};
}
inline FarmingProgressFile save_farming_progress(FarmingMetadataDirectory& metadata, FarmingProgress const& progress,
    uint64_t memory_limit, std::optional<stream_state::Stamp> expected_stamp = {})
{
    using namespace farming_progress_detail;
    // DirectoryLock is held by the caller throughout the compare and replace.
    validate(progress); metadata.create(); metadata.check();
    require(progress.directory_identity == farming_directory_identity(metadata.root), "farming progress writer changed destination identity");
    auto const path = metadata.path / filename;
    if (expected_stamp)
    {
        auto current = load_farming_progress(metadata, memory_limit);
        require(current && current->stamp == *expected_stamp && monotone(current->progress, progress),
            "farming progress changed before monotone publication");
    }
    else require(progress.sequence == 1 && !present(path), "farming progress must resume its existing local sequence");
    auto body = farming_progress_text(progress); auto const checksum = stream_state::digest(body); body += checksum + '\n';
    check_farming_state_size(body.size(), memory_limit); remove_unpublished(metadata); metadata.check();
    file_io::Space space{};
    require(!file_io::space_path(metadata.path, &space), "cannot inspect farming progress allocation alignment");
    auto const unit = std::max(uint64_t(space.f_bsize), uint64_t(space.f_frsize));
    require(unit && farming_detail::ceiling(body.size(), unit) <= UINT64_MAX / unit, "farming progress allocation bound overflows");
    auto const replacement = farming_detail::ceiling(body.size(), unit) * unit;
    require(space.f_frsize && uint64_t(space.f_bavail) <= UINT64_MAX / uint64_t(space.f_frsize),
        "invalid farming progress free-space accounting");
    auto const available = uint64_t(space.f_bavail) * uint64_t(space.f_frsize);
    require(progress.headroom_bytes <= available && replacement <= available - progress.headroom_bytes,
        "farming progress replacement needs " + std::to_string(replacement) + " bytes with " + std::to_string(progress.headroom_bytes) +
        " bytes of declared headroom; available free space is " + std::to_string(available) + " bytes");
    auto const current_bytes = present(path) ? charge(path) : 0;
    require(current_bytes <= progress.progress_reserved_bytes && replacement <= progress.progress_reserved_bytes - current_bytes,
        "farming progress replacement needs " + std::to_string(current_bytes) + " current bytes plus " + std::to_string(replacement) +
        " staged bytes; globally reserved allowance is " + std::to_string(progress.progress_reserved_bytes) + " bytes");
    if (expected_stamp) require(stream_state::stamp(path) == *expected_stamp, "farming progress changed before publication");
    auto const pending = stage_farming_state(path, body, false);
    try
    {
        metadata.check();
        require(charge(pending) <= progress.progress_reserved_bytes - current_bytes,
            "staged farming progress allocation exceeds its globally reserved metadata grant");
        file_io::Space staged_space{};
        require(!file_io::space_path(metadata.path, &staged_space) && staged_space.f_frsize &&
            uint64_t(staged_space.f_bavail) <= UINT64_MAX / uint64_t(staged_space.f_frsize) &&
            uint64_t(staged_space.f_bavail) * uint64_t(staged_space.f_frsize) >= progress.headroom_bytes,
            "staged farming progress consumed declared filesystem headroom");
        if (expected_stamp) require(stream_state::stamp(path) == *expected_stamp, "farming progress changed during staging");
        require(!file_io::rename_file(pending, path, expected_stamp.has_value()), "cannot publish farming progress");
        stream_state::sync_path(metadata.path, true); metadata.check();
    }
    catch (...)
    {
        if (present(pending)) { require(!file_io::unlink_file(pending), "cannot reclaim refused farming progress staging"); stream_state::sync_path(metadata.path, true); }
        throw;
    }
    return {progress, stream_state::stamp(path), farming_progress_allocation(metadata)};
}
inline void retire_farming_progress(FarmingMetadataDirectory const& metadata, FarmingState const& state, size_t lane,
    FarmingProgressFile const& file, std::string const& policy = {}, std::optional<FarmingProgressNativeBinding> const& native = {})
{
    using namespace farming_progress_detail;
    // All affected DirectoryLocks remain held. Ordinary retirement follows
    // global publication; relocation first persists every successor copy so
    // its acknowledged prefix survives retiring the old native binding.
    binding(state, lane, file.progress, policy, native);
    require(covered(state, lane, file.progress), "global farming state does not incorporate this local progress");
    metadata.check(); auto const path = metadata.path / filename;
    require(native_binding(file.progress, farming_directory_identity(metadata.root), native) && stream_state::stamp(path) == file.stamp,
        "farming progress changed before retirement");
    require(!file_io::unlink_file(path), "cannot retire incorporated farming progress");
    remove_unpublished(metadata); stream_state::sync_path(metadata.path, true); metadata.check();
}
} // namespace group_grow
