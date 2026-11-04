// Retirement is authorized by a private journal intent, never by a catalog.
// Claim a pinned source into an owner-only same-filesystem namespace, validate
// the claimed native object, then unlink only that private name.
#pragma once
#include "host/GroupFarmInventory.hpp"
#include "host/GroupFarmMigration.hpp"

namespace group_grow::farm
{
inline std::string replot_claim_directory(std::string const& job)
{
    inventory_detail::require(replot_digest(job), "invalid retirement job nonce");
    return ".xchplot2-retire-" + job;
}
inline std::string replot_claim_name(uint64_t source)
{
    std::ostringstream out; out << "source-" << std::setw(20) << std::setfill('0') << source; return out.str();
}
inline std::string replot_claim_path(std::string const& job, uint64_t source)
{ return replot_claim_directory(job) + '/' + replot_claim_name(source); }
#ifdef __linux__
namespace retirement_detail
{
inline inventory_detail::Handle private_directory(int root, std::string const& name, bool create)
{
    if (create && ::mkdirat(root, name.c_str(), 0700))
        inventory_detail::require(errno == EEXIST, "cannot create private retirement directory");
    inventory_detail::Handle result(::openat(root, name.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    auto const directory = inventory_detail::stat(result); file_io::Stat parent{};
    inventory_detail::require(!file_io::inspect_fd(root, &parent) && directory.st_dev == parent.st_dev &&
        directory.st_uid == ::geteuid() && (directory.st_mode & 0777) == 0700,
        "retirement directory must be owner-only and on the selected source filesystem");
    return result;
}
inline bool exists(int root, std::string const& relative)
{
    file_io::Stat info{};
    if (!::fstatat(root, relative.c_str(), &info, AT_SYMLINK_NOFOLLOW)) return true;
    inventory_detail::require(errno == ENOENT, "cannot inspect retirement source or claim"); return false;
}
} // namespace retirement_detail
#elif defined(_WIN32)
namespace retirement_detail
{
inline std::filesystem::path handle_path(int fd)
{
    std::wstring value(32768, L'\0');
    DWORD const count = ::GetFinalPathNameByHandleW(file_io::handle(fd), value.data(), DWORD(value.size()), VOLUME_NAME_DOS);
    inventory_detail::require(count && count < value.size(), "cannot resolve pinned retirement directory");
    value.resize(count); return std::filesystem::path(value);
}
inline inventory_detail::Handle private_directory(int root, std::string const& name, bool create)
{
    auto const path = handle_path(root) / inventory_detail::relative(name);
    if (create)
    {
        try { pos2gpu::create_private_directory(path); }
        catch (std::system_error const& error)
        { inventory_detail::require(error.code().value() == ERROR_ALREADY_EXISTS, "cannot create private retirement directory"); }
    }
    auto result = inventory_detail::windows_open(path, true);
    pos2gpu::validate_private_file_handle(file_io::handle(result.fd), true);
    file_io::Stat parent{}; auto const directory = inventory_detail::stat(result);
    inventory_detail::require(!file_io::inspect_fd(root, &parent) && directory.st_dev == parent.st_dev,
        "retirement directory must remain on the selected source filesystem");
    return result;
}
inline bool exists(int root, std::string const& relative)
{
    file_io::Stat value{};
    if (!file_io::inspect_link(handle_path(root) / inventory_detail::relative(relative), &value)) return true;
    inventory_detail::require(errno == ENOENT, "cannot inspect retirement source or claim"); return false;
}
inline void rename_claim(InventorySourcePin const& pin, int directory, std::string const& name)
{
    auto const filename = file_io::native_path(name).native();
    std::vector<uint8_t> buffer(sizeof(FILE_RENAME_INFO) + filename.size() * sizeof(wchar_t), uint8_t{});
    auto* rename = reinterpret_cast<FILE_RENAME_INFO*>(buffer.data());
    rename->ReplaceIfExists = FALSE; rename->RootDirectory = file_io::handle(directory);
    rename->FileNameLength = DWORD(filename.size() * sizeof(wchar_t));
    std::memcpy(rename->FileName, filename.data(), rename->FileNameLength);
    // FileRenameInfo operates on this DELETE-access handle, with a simple name
    // relative to the held target directory. A collision never replaces a file.
    inventory_detail::require(::SetFileInformationByHandle(file_io::handle(pin.source_fd()), FileRenameInfo,
        rename, DWORD(buffer.size())), "cannot claim retirement source without replacing another file");
    // NTFS write-through handles flush metadata changes including rename;
    // FlushFileBuffers additionally commits this file's cached metadata before
    // the controller can publish its claimed revision (CreateFileW/File Caching).
    inventory_detail::require(!file_io::flush(pin.source_fd()), "cannot flush retirement claim; retain it for recovery");
}
} // namespace retirement_detail
#endif
inline InventoryFileIdentity claim_replot_source(InventoryRoot const& root, InventoryPlot const& plot,
    std::string const& job, uint64_t source, InventoryLimits const& limits, bool recovering = false)
{
#if defined(__linux__) || defined(_WIN32)
    inventory_detail::require(plot.identity.links == 1 && !plot.alias_of, "linked or aliased sources cannot be retired independently");
    auto const relative = replot_claim_path(job, source);
    inventory_detail::RootHandle root_pin(file_io::native_path(root.path)); root_pin.matches(root);
    bool const original = retirement_detail::exists(root_pin.handle.fd, plot.relative_path);
    bool const claimed = retirement_detail::exists(root_pin.handle.fd, relative);
    inventory_detail::require(original != claimed, original ? "retirement source and claim both exist; no file was removed" :
        "retirement source and claim are both missing");
    InventorySourcePin pin(root, plot, limits, claimed ? relative : recovering ? plot.relative_path : "", true);
    auto directory = retirement_detail::private_directory(pin.root_fd(), replot_claim_directory(job), !claimed);
    if (!claimed)
    {
        pin.recheck_parent(plot.relative_path);
#ifdef _WIN32
        retirement_detail::rename_claim(pin, directory.fd, replot_claim_name(source));
#else
        inventory_detail::require(!::renameat2(pin.source_parent_fd(), pin.source_basename().c_str(), directory.fd,
            replot_claim_name(source).c_str(), RENAME_NOREPLACE), "cannot claim retirement source without replacing another file");
#endif
        inventory_detail::require(!file_io::flush(pin.source_parent_fd()) && !file_io::flush(directory.fd) && !file_io::flush(pin.root_fd()),
            "cannot sync retirement claim; retain it for recovery");
        grow_failpoint("replot-source-renamed");
    }
    inventory_detail::require(pin.matches_claim(relative, limits), "claimed source identity changed; preserve the claim and inspect it");
#ifdef _WIN32
    return pin.identity(relative, limits);
#else
    return observe_inventory_file(root, relative, limits).identity;
#endif
#else
    (void)root; (void)plot; (void)job; (void)source; (void)limits; (void)recovering;
    throw std::runtime_error("migration retirement requires a supported native handle-bound claim protocol");
#endif
}
inline bool replot_claim_missing(InventoryRoot const& root, InventoryPlot const& plot, std::string const& job, uint64_t source)
{
#if defined(__linux__) || defined(_WIN32)
    inventory_detail::RootHandle pin(file_io::native_path(root.path)); pin.matches(root);
    return !retirement_detail::exists(pin.handle.fd, plot.relative_path) && !retirement_detail::exists(pin.handle.fd, replot_claim_path(job, source));
#else
    (void)root; (void)plot; (void)job; (void)source; return false;
#endif
}
inline void unlink_replot_claim(InventoryRoot const& root, InventoryPlot const& plot, std::string const& job,
    uint64_t source, InventoryLimits const& limits, std::function<void()> const& before_unlink)
{
#if defined(__linux__) || defined(_WIN32)
    auto const relative = replot_claim_path(job, source); InventorySourcePin pin(root, plot, limits, relative, true);
    auto directory = retirement_detail::private_directory(pin.root_fd(), replot_claim_directory(job), false);
    auto const held = inventory_detail::stat(directory), parent = [&] { file_io::Stat value{};
        inventory_detail::require(!file_io::inspect_fd(pin.source_parent_fd(), &value), "cannot inspect claimed source parent"); return value; }();
    inventory_detail::require(held.st_dev == parent.st_dev && held.st_ino == parent.st_ino && pin.matches_claim(relative, limits),
        "retirement claim escaped its private directory");
    before_unlink();
    pin.recheck_parent(relative);
    inventory_detail::require(pin.matches_claim(relative, limits), "retirement claim changed during live readiness checks");
    grow_failpoint("replot-before-unlink");
#ifdef _WIN32
    struct Disposition { DWORD flags; } disposition{1u | 2u};
    inventory_detail::require(::SetFileInformationByHandle(file_io::handle(pin.source_fd()), FILE_INFO_BY_HANDLE_CLASS(21),
        &disposition, sizeof(disposition)), "cannot unlink validated retirement claim with native POSIX semantics");
    // POSIX disposition removes the link when this exact handle closes. No
    // pathname delete fallback may act on a substituted claim.
    inventory_detail::require(!pin.close_source(), "cannot close native retirement disposition; retain the journal for recovery");
#else
    inventory_detail::require(!::unlinkat(directory.fd, replot_claim_name(source).c_str(), 0), "cannot unlink validated retirement claim");
#endif
    inventory_detail::require(!file_io::flush(directory.fd), "cannot sync retired source; retain the journal for recovery");
    grow_failpoint("replot-source-unlinked");
#else
    (void)root; (void)plot; (void)job; (void)source; (void)limits; (void)before_unlink;
    throw std::runtime_error("migration retirement requires a supported native handle-bound claim protocol");
#endif
}
} // namespace group_grow::farm
