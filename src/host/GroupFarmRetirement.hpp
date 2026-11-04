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
#endif
inline InventoryFileIdentity claim_replot_source(InventoryRoot const& root, InventoryPlot const& plot,
    std::string const& job, uint64_t source, InventoryLimits const& limits)
{
#ifdef __linux__
    inventory_detail::require(plot.identity.links == 1 && !plot.alias_of, "linked or aliased sources cannot be retired independently");
    auto const relative = replot_claim_path(job, source);
    inventory_detail::RootHandle root_pin(file_io::native_path(root.path)); root_pin.matches(root);
    bool const original = retirement_detail::exists(root_pin.handle.fd, plot.relative_path);
    bool const claimed = retirement_detail::exists(root_pin.handle.fd, relative);
    inventory_detail::require(original != claimed, original ? "retirement source and claim both exist; no file was removed" :
        "retirement source and claim are both missing");
    InventorySourcePin pin(root, plot, limits, claimed ? relative : "");
    auto directory = retirement_detail::private_directory(pin.root_fd(), replot_claim_directory(job), !claimed);
    if (!claimed)
    {
        pin.recheck_parent(plot.relative_path);
        inventory_detail::require(!::renameat2(pin.source_parent_fd(), pin.source_basename().c_str(), directory.fd,
            replot_claim_name(source).c_str(), RENAME_NOREPLACE), "cannot claim retirement source without replacing another file");
        inventory_detail::require(!file_io::flush(pin.source_parent_fd()) && !file_io::flush(directory.fd) && !file_io::flush(pin.root_fd()),
            "cannot sync retirement claim; retain it for recovery");
        grow_failpoint("replot-source-renamed");
    }
    inventory_detail::require(pin.matches_claim(relative, limits), "claimed source identity changed; preserve the claim and inspect it");
    return observe_inventory_file(root, relative, limits).identity;
#else
    (void)root; (void)plot; (void)job; (void)source; (void)limits;
    throw std::runtime_error("migration retirement requires the supported native Linux claim protocol");
#endif
}
inline bool replot_claim_missing(InventoryRoot const& root, InventoryPlot const& plot, std::string const& job, uint64_t source)
{
#ifdef __linux__
    inventory_detail::RootHandle pin(file_io::native_path(root.path)); pin.matches(root);
    return !retirement_detail::exists(pin.handle.fd, plot.relative_path) && !retirement_detail::exists(pin.handle.fd, replot_claim_path(job, source));
#else
    (void)root; (void)plot; (void)job; (void)source; return false;
#endif
}
inline void unlink_replot_claim(InventoryRoot const& root, InventoryPlot const& plot, std::string const& job,
    uint64_t source, InventoryLimits const& limits, std::function<void()> const& before_unlink)
{
#ifdef __linux__
    auto const relative = replot_claim_path(job, source); InventorySourcePin pin(root, plot, limits, relative);
    auto directory = retirement_detail::private_directory(pin.root_fd(), replot_claim_directory(job), false);
    auto const held = inventory_detail::stat(directory), parent = [&] { file_io::Stat value{};
        inventory_detail::require(!file_io::inspect_fd(pin.source_parent_fd(), &value), "cannot inspect claimed source parent"); return value; }();
    inventory_detail::require(held.st_dev == parent.st_dev && held.st_ino == parent.st_ino && pin.matches_claim(relative, limits),
        "retirement claim escaped its private directory");
    before_unlink();
    pin.recheck_parent(relative);
    inventory_detail::require(pin.matches_claim(relative, limits), "retirement claim changed during live readiness checks");
    grow_failpoint("replot-before-unlink");
    inventory_detail::require(!::unlinkat(directory.fd, replot_claim_name(source).c_str(), 0), "cannot unlink validated retirement claim");
    inventory_detail::require(!file_io::flush(directory.fd), "cannot sync retired source; retain the journal for recovery");
    grow_failpoint("replot-source-unlinked");
#else
    (void)root; (void)plot; (void)job; (void)source; (void)limits; (void)before_unlink;
    throw std::runtime_error("migration retirement requires the supported native Linux claim protocol");
#endif
}
} // namespace group_grow::farm
