// Keyless, explicitly saved farming reviews. Execution still uses the private
// replicated job and each group's existing construction/recovery protocol.
#pragma once
#include "host/GroupFarmState.hpp"
#include "host/GroupHarvesterProfile.hpp"

namespace group_grow
{
struct FarmingPreview
{
    FarmingState state;
    std::string source_digest;
    std::filesystem::path source;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> relocations;
};
inline std::string farming_preview_text(FarmingState const& state, std::string const& source_digest,
    std::filesystem::path const& source,
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> const& relocations)
{
    std::ostringstream out;
    out << "ordinary-group-farming-preview-v1\n"
        << std::quoted(source_digest) << ' ' << std::quoted(file_io::path_text(source))
        << ' ' << relocations.size() << '\n';
    for (auto const& [old, replacement] : relocations)
        out << std::quoted(file_io::path_text(old)) << ' ' << std::quoted(file_io::path_text(replacement)) << '\n';
    out << std::quoted(farming_state_text(state, true)) << '\n';
    return out.str();
}
inline std::string farming_preview_text(FarmingPreview const& preview)
{ return farming_preview_text(preview.state, preview.source_digest, preview.source, preview.relocations); }
inline FarmingPreview parse_farming_preview(std::string const& body, uint64_t memory_limit = 256 * 1024 * 1024,
    FarmingStateLoad mode = FarmingStateLoad::recovery)
{
    std::istringstream in(body);
    std::string version, source, state;
    std::getline(in, version);
    require(version == "ordinary-group-farming-preview-v1", "expected a saved keyless farming preview");
    FarmingPreview preview;
    size_t count = 0;
    in >> std::quoted(preview.source_digest) >> std::quoted(source) >> count;
    require(in && count <= 1024 && ((source.empty() && preview.source_digest.empty()) ||
        (!source.empty() && preview.source_digest.size() == 64 &&
            preview.source_digest.find_first_not_of("0123456789abcdef") == std::string::npos)),
        "invalid farming preview source");
    preview.source = file_io::native_path(source);
    require(source.empty() || preview.source.is_absolute(), "farming preview source must be absolute");
    for (size_t i = 0; i < count; ++i)
    {
        std::string old, replacement; in >> std::quoted(old) >> std::quoted(replacement);
        auto a = file_io::native_path(old), b = file_io::native_path(replacement);
        require(in && !source.empty() && a.is_absolute() && b.is_absolute(), "invalid farming preview relocation");
        preview.relocations.emplace_back(std::move(a), std::move(b));
    }
    in >> std::quoted(state);
    require(bool(in), "incomplete farming preview");
    in >> std::ws; require(in.eof(), "unexpected farming preview data");
    preview.state = parse_farming_state(state, memory_limit, mode);
    require(preview.state.fixed_extent && preview.state.estimate.feasible,
        "farming preview does not contain an executable managed layout");
    require(preview.state.fingerprint == std::string(64, '0') && preview.state.seed == std::string(64, '0') &&
        preview.state.seed_owner.empty() && preview.state.farmer == std::string(96, '0') &&
        preview.state.pool == std::string(preview.state.pool_public_key ? 96 : 64, '0'),
        "farming preview contains private identity fields");
    for (auto const& directory : preview.state.directories)
        require(file_io::native_path(directory.directory).is_absolute(), "farming preview destination must be absolute");
    (void)farming_state_profile(preview.state, std::min(memory_limit / 8, uint64_t(256 * 1024 * 1024)));
    return preview;
}
struct FarmingPreviewFile
{
    FarmingPreview preview;
    stream_state::Stamp stamp;
    std::string digest;
};
inline FarmingPreviewFile load_farming_preview(std::filesystem::path const& path, uint64_t memory_limit,
    FarmingStateLoad mode = FarmingStateLoad::recovery)
{
    auto const file = read_farming_state(path, memory_limit);
    auto preview = parse_farming_preview(file.body, memory_limit, mode);
    require(stream_state::stamp(path) == file.stamp, "farming preview changed while loading");
    return {std::move(preview), file.stamp, file.digest};
}
} // namespace group_grow
