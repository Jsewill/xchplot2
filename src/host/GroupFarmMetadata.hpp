// Destination bookkeeping has one private namespace. Fixed legacy records
// retain their original location; discovery never creates or moves anything.
#pragma once
#include "host/GroupFarmFile.hpp"
#include <memory>
#include <optional>
#include <span>

namespace group_grow
{
namespace farm_metadata_detail
{
namespace fs = std::filesystem;
using farm::inventory_detail::require;
inline bool present(fs::path const& path)
{
    file_io::Stat info{};
    if (!file_io::inspect_link(path, &info)) return true;
    require(errno == ENOENT, "cannot inspect farming metadata: " + file_io::path_text(path)); return false;
}
inline bool records(fs::path const& directory)
{
    bool found = false;
    for (auto const* name : {"grow-farm.plan", "grow-farm.plan.next", "grow-replot.plan", "grow-replot.plan.next"})
        found = present(directory / name) || found;
    return found;
}
inline void private_directory(farm::inventory_detail::RootHandle const& parent,
    farm::inventory_detail::RootHandle const& directory)
{
    auto const root = farm::inventory_detail::stat(parent.handle), value = farm::inventory_detail::stat(directory.handle);
    require(value.st_dev == root.st_dev, "farming metadata directory must remain on its destination filesystem");
#ifdef _WIN32
    pos2gpu::validate_private_file_handle(file_io::handle(directory.handle.fd), true);
#else
    require(value.st_uid == ::geteuid() && (value.st_mode & 0777) == 0700,
        "farming metadata directory must be owned by this user with owner-only permissions");
#endif
}
inline void same_directory(farm::inventory_detail::RootHandle const& saved)
{
    farm::inventory_detail::RootHandle current(saved.path);
    auto const before = farm::inventory_detail::stat(saved.handle), after = farm::inventory_detail::stat(current.handle);
    require(before.st_dev == after.st_dev && before.st_ino == after.st_ino,
        "farming metadata directory changed during the operation");
}
inline void create_directory(farm::inventory_detail::RootHandle const& root, std::string const& name)
{
#ifdef _WIN32
    try { pos2gpu::create_private_directory(root.path / file_io::native_path(name)); }
    catch (std::system_error const& error)
    { require(error.code().value() == ERROR_ALREADY_EXISTS, "cannot create private farming metadata directory"); }
#else
    require(!::mkdirat(root.handle.fd, name.c_str(), 0700) || errno == EEXIST,
        "cannot create private farming metadata directory");
#endif
    require(!file_io::flush(root.handle.fd), "cannot sync private farming metadata directory");
}
inline uint64_t path_memory(fs::path const& path)
{
    require(path.native().capacity() <= (UINT64_MAX / 4 - sizeof(fs::path)) / sizeof(fs::path::value_type),
        "farming metadata path memory overflow");
    return 4 * (sizeof(fs::path) + path.native().capacity() * sizeof(fs::path::value_type));
}
inline uint64_t handle_memory(fs::path const& path)
{
    uint64_t bytes = sizeof(farm::inventory_detail::RootHandle) + path_memory(path);
#ifdef _WIN32
    // RootHandle retains every no-follow ancestor; vector growth is bounded
    // by twice the number of path components.
    bytes += 2 * (size_t(std::distance(path.begin(), path.end())) + 1) * sizeof(farm::inventory_detail::Handle);
#endif
    return bytes;
}
inline bool reserved_component(fs::path const& component)
{
#ifdef _WIN32
    auto const& value = component.native(); constexpr std::wstring_view name = L".xchplot2";
    return ::CompareStringOrdinal(value.data(), int(value.size()), name.data(), int(name.size()), TRUE) == CSTR_EQUAL;
#else
    return component == ".xchplot2";
#endif
}
} // namespace farm_metadata_detail

struct FarmingMetadataDirectory
{
    std::filesystem::path root, path;
    bool legacy = false, recorded = false;
    std::unique_ptr<farm::inventory_detail::RootHandle> root_pin, private_pin;
    explicit FarmingMetadataDirectory(std::filesystem::path directory, std::optional<bool> selected = {}) :
        root(std::move(directory)), path(root / ".xchplot2")
    {
        using namespace farm_metadata_detail;
        if (!present(root)) { legacy = selected.value_or(false); if (legacy) path = root; return; }
        root_pin = std::make_unique<farm::inventory_detail::RootHandle>(root);
        if (present(path))
        {
            private_pin = std::make_unique<farm::inventory_detail::RootHandle>(path);
            private_directory(*root_pin, *private_pin);
        }
        bool const old = records(root), current = private_pin && records(path);
        require(!old || !current, "destination contains both legacy and private farming metadata; preserve both and resolve the conflict");
        recorded = old || current; legacy = selected.value_or(old);
        require(!recorded || legacy == old, "farming destinations contain conflicting metadata layouts");
        if (legacy) path = root;
    }
    void check() const
    {
        using namespace farm_metadata_detail;
        if (!root_pin) { require(!present(root), "unavailable farming metadata destination appeared"); return; }
        same_directory(*root_pin);
        if (private_pin)
        {
            same_directory(*private_pin); private_directory(*root_pin, *private_pin);
        }
        else require(!present(root / ".xchplot2"), "private farming metadata directory appeared during the operation");
        require(!records(legacy ? root / ".xchplot2" : root), "a conflicting farming metadata layout appeared during the operation");
    }
    void create()
    {
        using namespace farm_metadata_detail;
        check(); require(bool(root_pin), "farming metadata destination is unavailable");
        if (legacy || private_pin) return;
        create_directory(*root_pin, ".xchplot2");
        private_pin = std::make_unique<farm::inventory_detail::RootHandle>(path);
        private_directory(*root_pin, *private_pin); check();
    }
    void require_fresh() const
    {
        check();
        if (!legacy && !recorded && private_pin)
            farm_metadata_detail::require(std::filesystem::is_empty(path),
                "reserved .xchplot2 directory contains bookkeeping without its saved job; preserve it and recover that job first");
    }
    uint64_t allocation() const
    {
        if (legacy || !private_pin) return 0;
        check(); auto const info = farm::inventory_detail::stat(private_pin->handle);
        farm_metadata_detail::require(info.st_blocks >= 0 && uint64_t(info.st_blocks) <= UINT64_MAX / 512,
            "invalid farming metadata directory allocation");
        return uint64_t(info.st_blocks) * 512;
    }
};
inline bool farming_metadata_legacy(std::span<std::filesystem::path const> directories, bool existing = false)
{
    std::optional<bool> selected;
    for (auto const& path : directories)
    {
        FarmingMetadataDirectory directory(path);
        if (!directory.recorded) continue;
        farm_metadata_detail::require(!selected || *selected == directory.legacy,
            "farming destinations contain conflicting metadata layouts");
        selected = directory.legacy;
    }
    return selected.value_or(existing);
}
inline std::filesystem::path farming_metadata_path(std::filesystem::path const& directory,
    std::string const& name, std::optional<bool> legacy = {})
{
    if (legacy) return (*legacy ? directory : directory / ".xchplot2") / name;
    return FarmingMetadataDirectory(directory).path / name;
}
inline bool farming_metadata_namespace(std::filesystem::path const& directory)
{
    auto reserved = [](std::filesystem::path const& path)
    {
        for (auto const& component : path)
            if (farm_metadata_detail::reserved_component(component)) return true;
        return false;
    };
    auto const path = std::filesystem::absolute(directory).lexically_normal();
    return reserved(path) || reserved(std::filesystem::weakly_canonical(path));
}
inline void require_farming_review_path(std::filesystem::path const& directory, std::filesystem::path const& output)
{
    auto const parent = std::filesystem::weakly_canonical(directory / ".xchplot2");
    auto const child = std::filesystem::weakly_canonical(output);
    auto part = parent.begin(), descendant = child.begin(); bool contains = true;
    for (; part != parent.end(); ++part, ++descendant)
    {
        if (descendant == child.end()) { contains = false; break; }
#ifdef _WIN32
        auto left = part->native(), right = descendant->native();
        std::replace(left.begin(), left.end(), L'/', L'\\'); std::replace(right.begin(), right.end(), L'/', L'\\');
        if (::CompareStringOrdinal(left.data(), int(left.size()), right.data(), int(right.size()), TRUE) != CSTR_EQUAL)
        { contains = false; break; }
#else
        if (*part != *descendant) { contains = false; break; }
#endif
    }
    farm_metadata_detail::require(!contains, "save portable reviews outside the reserved .xchplot2 recovery directory");
}
inline std::filesystem::path farming_replot_directory(std::filesystem::path const& directory,
    std::string const& job, std::optional<bool> legacy = {})
{
    farm_metadata_detail::require(job.size() == 64 && job.find_first_not_of("0123456789abcdef") == job.npos,
        "invalid farming metadata job nonce");
    bool const old = legacy.value_or(false);
    if (legacy) return old ? directory / (".xchplot2-replot-" + job) : directory / ".xchplot2" / ("replot-" + job);
    FarmingMetadataDirectory metadata(directory);
    return farming_replot_directory(directory, job, metadata.legacy);
}
struct FarmingReplotDirectory
{
    FarmingMetadataDirectory metadata;
    std::filesystem::path path;
    std::unique_ptr<farm::inventory_detail::RootHandle> pin;
    FarmingReplotDirectory(std::filesystem::path const& root, std::string const& job, bool create,
        std::optional<bool> legacy = {}) : metadata(root, legacy), path(farming_replot_directory(root, job, metadata.legacy))
    {
        using namespace farm_metadata_detail;
        if (create)
        {
            metadata.create();
            create_directory(metadata.legacy ? *metadata.root_pin : *metadata.private_pin, file_io::path_text(path.filename()));
        }
        if (present(path))
        {
            require(metadata.root_pin && (metadata.legacy || metadata.private_pin), "private migration metadata parent changed");
            pin = std::make_unique<farm::inventory_detail::RootHandle>(path);
            private_directory(metadata.legacy ? *metadata.root_pin : *metadata.private_pin, *pin);
        }
        check();
    }
    void check() const
    {
        using namespace farm_metadata_detail;
        metadata.check();
        if (pin) { same_directory(*pin); private_directory(metadata.legacy ? *metadata.root_pin : *metadata.private_pin, *pin); }
        else require(!present(path), "private migration metadata directory appeared during the operation");
    }
    uint64_t allocation() const
    {
        check(); if (!pin) return 0; auto const info = farm::inventory_detail::stat(pin->handle);
        farm_metadata_detail::require(info.st_blocks >= 0 && uint64_t(info.st_blocks) <= UINT64_MAX / 512,
            "invalid private migration directory allocation");
        return uint64_t(info.st_blocks) * 512;
    }
};
inline uint64_t farming_metadata_memory_bound(std::filesystem::path const& root, std::string const& job = {})
{
    using namespace farm_metadata_detail;
    auto const metadata = root / ".xchplot2";
    uint64_t bytes = sizeof(FarmingMetadataDirectory) + 2 * sizeof(void*) + path_memory(root) + path_memory(metadata) +
        handle_memory(root) + handle_memory(metadata);
    if (!job.empty())
    {
        // A journal replica and its job directory own independent root pins.
        auto const child = root / (".xchplot2-replot-" + job);
        require(bytes <= UINT64_MAX / 2 && path_memory(child) <= UINT64_MAX / 8,
            "migration metadata memory overflow");
        bytes = 2 * bytes + sizeof(FarmingReplotDirectory) + path_memory(child) + handle_memory(child);
    }
    return bytes;
}
} // namespace group_grow
