// Construction compatibility is an explicit contract, independent of source
// provenance. Bump the algorithm identity when canonical encoding, member
// ordering/deduplication, contribution packets, or replay semantics change.
// Incompatible checkpoint layouts also require a new engine schema.
#pragma once
#include "host/GroupFormat.hpp"

#include <climits>
#include <cstring>
#include <initializer_list>

#ifndef GROW_CONSTRUCTION_ALGORITHM_IDENTITY
#define GROW_CONSTRUCTION_ALGORITHM_IDENTITY "ordinary-group-construction-1"
#endif

namespace group_grow
{
inline constexpr char construction_algorithm_identity[] = GROW_CONSTRUCTION_ALGORITHM_IDENTITY;
static_assert(sizeof(construction_algorithm_identity) <= 64);

struct ConstructionCheckpointIdentity
{
    std::string format, producer_build, algorithm, representation;
    bool legacy_exact_build = false;
};
struct ConstructionPathBinding
{
    std::string saved_root, current_root, directory_identity;
};

// Byte arrays keep this extension independent of native structure padding.
// The digest covers the native scalar representation and the engine's saved
// layouts, so an ABI change cannot gain authority from a matching algorithm.
struct ConstructionCompatibility
{
    std::array<char, 64> algorithm{};
    std::array<uint8_t, 32> representation{};
    bool operator==(ConstructionCompatibility const&) const = default;
};

inline std::array<char, 64> construction_build(std::string_view source)
{
    std::array<char, 64> result{};
    std::copy_n(source.begin(), std::min(source.size(), result.size()), result.begin());
    return result;
}
inline std::string construction_build_text(std::array<char, 64> const& source)
{ return {source.begin(), std::find(source.begin(), source.end(), '\0')}; }

inline ConstructionCompatibility construction_compatibility(std::initializer_list<uint64_t> layout)
{
    using H = PlotGroupFile::Header;
    ConstructionCompatibility result;
    std::copy(std::begin(construction_algorithm_identity), std::end(construction_algorithm_identity), result.algorithm.begin());
    struct sha256 context;
    sha256_init(&context);
    auto bytes = [&](auto const& value)
    { sha256_update(&context, reinterpret_cast<uint8_t const*>(&value), sizeof(value)); };
    constexpr char domain[] = "xchplot2-construction-native-1";
    bytes(domain);
    bytes(uint64_t(0x0102030405060708)); bytes(int64_t(-2)); bytes(uint32_t(0x01020304));
    bytes(uint16_t(0x0102)); bytes(false); bytes(true);
    for (uint64_t value : {uint64_t(CHAR_BIT), uint64_t(sizeof(unsigned)), uint64_t(sizeof(bool)),
        uint64_t(alignof(uint64_t)), uint64_t(alignof(uint32_t)), uint64_t(sizeof(H)), uint64_t(alignof(H)),
        uint64_t(offsetof(H, magic)), uint64_t(offsetof(H, version)), uint64_t(offsetof(H, group_id)),
        uint64_t(offsetof(H, k)), uint64_t(offsetof(H, strength)), uint64_t(offsetof(H, meta_group)),
        uint64_t(offsetof(H, group_size)), uint64_t(offsetof(H, memo_length)), uint64_t(offsetof(H, chunk_index_offset))}) bytes(value);
    for (auto value : layout) bytes(value);
    sha256_sum(&context, result.representation.data());
    return result;
}
inline void check_construction_compatibility(ConstructionCompatibility const& saved,
    ConstructionCompatibility const& expected, std::string const& engine)
{
    group_tools::require(saved.algorithm == expected.algorithm, engine +
        " checkpoint algorithm is incompatible; preserve the plot, checkpoints and temporary files, "
        "then resume with a build supporting the recorded algorithm. See REFERENCE.md, Grouped plotting and drive filling");
    group_tools::require(saved.representation == expected.representation, engine +
        " checkpoint native representation is incompatible; preserve the plot, checkpoints and temporary files, "
        "then resume on a matching platform and build. See REFERENCE.md, Grouped plotting and drive filling");
}
class LegacyConstructionBuildError : public std::runtime_error
{
public:
    std::string const producer_build, engine;
    LegacyConstructionBuildError(std::string saved, std::string source) :
        std::runtime_error(source + " legacy checkpoint requires its original exact build (" + saved +
            "); preserve the plot, checkpoints and temporary files, then resume with that binary. "
            "See REFERENCE.md, Grouped plotting and drive filling"),
        producer_build(std::move(saved)), engine(std::move(source)) {}
};
inline void check_construction_build(std::string const& saved, std::string_view expected, std::string const& engine)
{
    if (saved != expected) throw LegacyConstructionBuildError(saved, engine);
}
inline void check_construction_build(std::array<char, 64> const& saved, std::string_view expected, std::string const& engine)
{
    if (saved != construction_build(expected)) throw LegacyConstructionBuildError(construction_build_text(saved), engine);
}
inline void check_construction_format(bool supported, std::string const& engine)
{
    group_tools::require(supported, engine +
        " checkpoint format is unsupported; preserve the plot, checkpoints and temporary files, "
        "then resume with a build supporting the recorded format. See REFERENCE.md, Grouped plotting and drive filling");
}
inline ConstructionCheckpointIdentity construction_checkpoint_identity(std::string format, std::string producer,
    bool legacy, ConstructionCompatibility const& compatibility)
{
    return {std::move(format), std::move(producer), legacy ? std::string{} : construction_build_text(compatibility.algorithm),
        legacy ? std::string{} : group_tools::hex(compatibility.representation), legacy};
}
}
