// Native, no-follow file pins shared by inventory and saved-plan type checks.
// Keep this layer independent of plot readers and vendor headers.
#pragma once
#include "host/GroupNativeIdentity.hpp"
#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace group_grow::farm
{
struct InventoryRoot;
namespace inventory_detail
{
namespace fs = std::filesystem;
inline void require(bool value, std::string const& message)
{ if (!value) throw std::runtime_error(message); }
inline void bounded_text(std::string const& text, size_t bound = 4096)
{ require(text.size() <= bound && text.find('\0') == text.npos, "farm inventory text exceeds its bound"); }
inline fs::path relative(std::string const& value)
{
    bounded_text(value); auto result = file_io::native_path(value);
    require(!result.empty() && !result.is_absolute() && !result.has_root_name(), "inventory path must be relative");
    for (auto const& component : result)
    {
        require(component != "." && component != ".." && !component.empty(), "invalid inventory path component");
#ifdef _WIN32
        require(component.native().find(L':') == std::wstring::npos, "alternate data streams cannot identify an inventory source");
#endif
    }
    return result;
}
struct Handle
{
    int fd = -1;
    Handle() = default;
    explicit Handle(int value) : fd(value) { require(fd >= 0, "cannot open farm inventory source without following links"); }
    Handle(Handle const&) = delete; Handle& operator=(Handle const&) = delete;
    Handle(Handle&& other) noexcept : fd(std::exchange(other.fd, -1)) {}
    Handle& operator=(Handle&& other) noexcept
    { if (this != &other) { if (fd >= 0) file_io::close_file(fd); fd = std::exchange(other.fd, -1); } return *this; }
    ~Handle() { if (fd >= 0) file_io::close_file(fd); }
};
inline file_io::Stat stat(Handle const& handle)
{ file_io::Stat result{}; require(!file_io::inspect_fd(handle.fd, &result), "cannot inspect farm inventory handle"); return result; }
#ifdef _WIN32
inline Handle windows_open(fs::path const& path, bool directory, bool retirement = false)
{
    // Held ancestors deny delete/rename. Held files also deny writes, so an
    // adapter observes the same object throughout its bounded operation.
    HANDLE native = ::CreateFileW(path.c_str(), GENERIC_READ | (retirement ? GENERIC_WRITE | DELETE : 0),
        directory ? FILE_SHARE_READ | FILE_SHARE_WRITE : FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | (directory ? FILE_FLAG_BACKUP_SEMANTICS : FILE_FLAG_OVERLAPPED) |
            (retirement ? FILE_FLAG_WRITE_THROUGH : 0), nullptr);
    if (native == INVALID_HANDLE_VALUE)
    {
        auto const error = ::GetLastError();
        throw std::runtime_error(retirement && error == ERROR_SHARING_VIOLATION ?
            "source is open without retirement sharing; unload the source root, keep its reader RPC available, and resume the saved migration" :
            "cannot pin farm inventory source (Windows error " + std::to_string(error) + ')');
    }
    int fd = ::_open_osfhandle(reinterpret_cast<intptr_t>(native), _O_RDONLY | _O_BINARY);
    if (fd < 0) { ::CloseHandle(native); throw std::runtime_error("cannot retain farm inventory handle"); }
    Handle result(fd); auto const s = stat(result);
    require(directory ? file_io::is_directory(s.st_mode) : file_io::is_regular(s.st_mode), "unsupported farm inventory file type");
    return result;
}
#endif
struct RootHandle
{
    fs::path path;
    Handle handle;
#ifdef _WIN32
    std::vector<Handle> ancestors;
#endif
    explicit RootHandle(fs::path const& input)
    {
        path = fs::absolute(input).lexically_normal();
        if (path != path.root_path() && path.filename().empty()) path = path.parent_path();
#ifdef _WIN32
        fs::path current = path.root_path(); ancestors.push_back(windows_open(current, true));
        for (auto const& part : path.relative_path()) { current /= part; ancestors.push_back(windows_open(current, true)); }
        handle = windows_open(path, true);
#else
        handle = Handle(::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        for (auto const& component : path.relative_path())
            handle = Handle(::openat(handle.fd, component.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
#endif
    }
    void matches(InventoryRoot const& root) const;
};
struct SourceHandle
{
    Handle handle;
    Handle parent;
    std::string basename;
#ifdef _WIN32
    std::vector<Handle> ancestors;
#endif
    SourceHandle(RootHandle const& root, std::string const& name, bool retirement = false)
    {
        auto const value = relative(name);
        basename = file_io::path_text(value.filename());
#ifdef _WIN32
        auto current = root.path;
        for (auto const& part : value.parent_path()) { current /= part; ancestors.push_back(windows_open(current, true)); }
        parent = windows_open(current, true);
        handle = windows_open(root.path / value, false, retirement);
#else
        parent = Handle(::fcntl(root.handle.fd, F_DUPFD_CLOEXEC, 3));
        for (auto const& part : value.parent_path())
            parent = Handle(::openat(parent.fd, part.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        handle = Handle(::openat(parent.fd, value.filename().c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW));
        require(file_io::is_regular(stat(handle).st_mode), "inventory source is not a regular file");
#endif
    }
};

inline std::string saved_plan_header(fs::path const& path)
{
    // Use the same native pins as inventory: opening a FIFO must not precede
    // the regular-file check, and ancestors/leaf must never follow links.
    auto observed = [&](fs::path const& name)
    {
        file_io::Stat value{};
        require(!file_io::inspect_link(name, &value) && file_io::is_regular(value.st_mode) && value.st_size >= 0,
            "expected regular file: " + file_io::path_text(name));
        return value;
    };
    auto fields = [](file_io::Stat const& value)
    {
#ifdef __APPLE__
        auto const modified = value.st_mtimespec, changed = value.st_ctimespec;
#else
        auto const modified = value.st_mtim, changed = value.st_ctim;
#endif
        return std::tuple{uint64_t(value.st_dev), uint64_t(value.st_ino), uint64_t(value.st_size),
            modified.tv_sec, changed.tv_sec, uint32_t(modified.tv_nsec), uint32_t(changed.tv_nsec)};
    };
    auto const saved = fields(observed(path));
    RootHandle root(fs::absolute(path).parent_path()); SourceHandle source(root, file_io::path_text(path.filename()));
    require(fields(stat(source.handle)) == saved, "saved plan changed before its type check");
    std::array<char, 64> line{}; size_t read = 0;
    while (read < line.size())
    {
        auto const count = file_io::read_at(source.handle.fd, line.data() + read, line.size() - read, read);
        if (count < 0 && errno == EINTR) continue;
        require(count >= 0, "cannot read saved plan type"); if (!count) break; read += size_t(count);
    }
    require(fields(stat(source.handle)) == saved && fields(observed(path)) == saved, "saved plan changed during its type check");
    auto const first = std::string_view(line.data(), read);
    auto const header = first.substr(0, first.find('\n'));
    return std::string(header);
}
inline bool is_replot_review(fs::path const& path)
{
    auto const header = saved_plan_header(path);
    return header == "ordinary-group-replot-review-v1" || header == "ordinary-group-replot-review-v2" || header == "ordinary-group-replot-review-v3" ||
        header == "ordinary-group-replot-review-v4" || header == "ordinary-group-replot-job-v1" ||
        header == "ordinary-group-replot-revalidation-v1" || header == "ordinary-group-replot-revalidation-v2" || header == "ordinary-group-replot-revalidation-v3";
}
} // namespace inventory_detail
} // namespace group_grow::farm
