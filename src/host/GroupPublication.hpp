// The requested name stays stable in checkpoints. Only its same-directory
// construction file is private; publication never copies the plot payload.
#pragma once
#include "host/GroupFile.hpp"
#include <stdexcept>

namespace group_grow::publication
{
namespace fs = std::filesystem;
inline void require(bool value, std::string const& message)
{ if (!value) throw std::runtime_error(message); }

template<class Path> std::string partial(Path const& value)
{ return file_io::path_text(file_io::native_path(value)) + ".partial"; }

// Migration already owns a separate private stage and publication protocol.
template<class Path> bool enabled(Path const& value)
{ return file_io::native_path(value).extension() != ".building"; }

template<class Path> std::string path(Path const& value)
{
    auto const name = file_io::path_text(file_io::native_path(value));
    if (!enabled(value)) return name;
    file_io::Stat staged{}, final{};
    int const exists = file_io::inspect_link(partial(value), &staged);
    if (exists) { require(errno == ENOENT, "cannot inspect private group output"); return name; }
    require(file_io::inspect_link(name, &final) && errno == ENOENT,
        "both the group output and its .partial file exist; preserve both and resolve the conflict before resuming");
    return partial(value);
}
template<class Path> bool pending(Path const& value)
{ return path(value) != file_io::path_text(file_io::native_path(value)); }
template<class Path> bool exists(Path const& value)
{
    file_io::Stat info{};
    int const status = file_io::inspect_link(path(value), &info);
    require(!status || errno == ENOENT, "cannot inspect group output");
    return !status;
}

// Keep the syscall-style error contract for cleanup paths, including destructors.
template<class Path> int inspect(Path const& value, file_io::Stat* out) noexcept
{
    try { return file_io::inspect_link(path(value), out); }
    catch (...) { errno = EINVAL; return -1; }
}
template<class Path> int remove(Path const& value) noexcept
{
    try { return file_io::unlink_file(path(value)); }
    catch (...) { errno = EINVAL; return -1; }
}
template<class Path> int open(Path const& value, int flags, bool private_output = false, unsigned mode = 0600)
{
    auto name = path(value);
    if (private_output && enabled(value) && (flags & file_io::create))
    {
        file_io::Stat info{};
        if (!file_io::inspect_link(name, &info)) { errno = EEXIST; return -1; }
        if (errno != ENOENT) return -1;
        name = partial(value);
    }
    return file_io::open_file(name, flags, mode);
}

inline void rename(std::string const& from, std::string const& to)
{
    // The construction owner retains its exclusive file lock across this
    // operation. A no-replace rename rejects a competing destination name.
    file_io::Stat before{}, after{}, parent_before{}, parent_after{};
    require(!file_io::inspect_link(from, &before) && file_io::is_regular(before.st_mode) && before.st_nlink == 1,
        "group publication requires its exclusively owned regular file");
    auto parent = file_io::native_path(from).parent_path(); if (parent.empty()) parent = ".";
    int const directory = file_io::open_file(parent, file_io::read_only | file_io::directory | file_io::no_follow | file_io::close_on_exec);
    require(directory >= 0, "cannot pin group publication directory");
    struct Close { int fd; ~Close() { file_io::close_file(fd); } } close{directory};
    require(!file_io::inspect_fd(directory, &parent_before) && !file_io::inspect_link(parent, &parent_after) &&
        parent_before.st_dev == parent_after.st_dev && parent_before.st_ino == parent_after.st_ino,
        "group publication directory changed");
    int const renamed = file_io::rename_file(from, to, false);
    int const error = renamed ? errno : 0;
    require(!renamed, "cannot rename group output without replacing an occupied path: " + std::string(std::strerror(error)));
    require(!file_io::inspect_link(to, &after) && file_io::is_regular(after.st_mode) && after.st_nlink == 1 &&
        before.st_dev == after.st_dev && before.st_ino == after.st_ino && before.st_size == after.st_size &&
        before.st_mtim.tv_sec == after.st_mtim.tv_sec && before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
        !file_io::inspect_link(parent, &parent_after) && parent_before.st_dev == parent_after.st_dev && parent_before.st_ino == parent_after.st_ino,
        "group output or parent changed during publication");
    require(!file_io::flush(directory), "cannot sync group publication directory");
}
template<class Path> void hide(Path const& value)
{
    if (!enabled(value) || pending(value)) return;
    rename(file_io::path_text(file_io::native_path(value)), partial(value));
}
template<class Path> void publish(Path const& value)
{
    if (!pending(value)) return;
    rename(partial(value), file_io::path_text(file_io::native_path(value)));
}
} // namespace group_grow::publication
