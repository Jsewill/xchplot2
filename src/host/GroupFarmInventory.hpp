// Public, bounded observations of explicitly selected farm roots. A catalog is
// never authority to unlink a path; retirement pins and rechecks its native ID.
#pragma once
#include "host/GroupFarmFile.hpp"
#include "host/GroupFarmState.hpp"
#include "host/GroupStorage.hpp"
#include "host/GroupFarmResources.hpp"
#include <functional>
#ifndef _WIN32
#include <dirent.h>
#endif

namespace group_grow::farm
{
enum class PlotAlgorithm { unknown, pos1, pos2 };
enum class PlotContainer { unknown, chiapos_v1, chiapos_v2, pos2_group };
struct InventoryLimits
{
    uint64_t memory_bytes = 256 * 1024 * 1024, max_files = 100000;
    uint64_t max_header_bytes = 4096, timeout_ms = 30000, max_output_bytes = 16 * 1024 * 1024;
    size_t max_roots = 128;
    std::function<bool()> cancelled;
};
struct InventoryFileIdentity
{
    stream_state::Stamp stamp;
    uint64_t allocated_bytes = 0, links = 0, header_bytes = 0;
    std::string header_digest;
    file_io::DurableIdentity durable;
    bool operator==(InventoryFileIdentity const&) const = default;
};
struct InventoryRoot
{
    std::string path, directory_identity;
    StorageDevice storage;
    uint64_t device = 0, inode = 0;
    file_io::DurableIdentity durable;
};
struct InventoryPlot
{
    size_t root = 0;
    std::string relative_path;
    InventoryFileIdentity identity;
    PlotAlgorithm algorithm = PlotAlgorithm::unknown;
    PlotContainer container = PlotContainer::unknown;
    uint8_t k = 0, strength = 0, compression_level = 0, meta_group = 0;
    uint16_t members = 0;
    std::string compression_family, plot_id, farmer_public_key, pool_public_key, pool_contract_puzzle_hash, plot_public_key;
    std::string reader, reader_version, host, harvester, solver, workload_id, status, reason;
    // CLI bytes/root/config plus observed application version; the RPC does
    // not expose a digest of all modules loaded by a separate running service.
    std::string reader_identity, checker_identity;
    // Structural header validation is not a complete proof or decoder check.
    bool header_validated = false;
    // Reader-reported identity bound to the exact observed native file. This
    // is not old proof validation or an imported deletion capability.
    bool native_identified = false;
    std::optional<size_t> alias_of;
};
struct NetworkProjection
{
    std::string network, genesis_challenge, source, version, captured_at, target_route, reader_version, reader_identity;
    std::string farmer_version, solver_version, service_status;
    uint64_t hard_fork2_height = 0, min_strength = 0, max_strength = 0, plot_size = 0;
    bool available = false;
};
struct RetirementReaderConfiguration
{
    std::string root_identity;
    std::vector<std::string> plot_directories;
};
struct RetirementReaderContext
{
    std::string reader_identity, retirement_identity;
    std::vector<RetirementReaderConfiguration> configurations;
};
// A successor review's current route snapshot. This never replaces the origin
// identification provenance of an already claimed or retired source file.
struct ReplotReaderObservation
{
    uint64_t root = 0;
    std::string reader, version, identity, source, captured_at;
};
struct FarmInventory
{
    unsigned version = 1;
    std::string host, captured_at, producer_build;
    std::vector<InventoryRoot> roots;
    std::vector<InventoryPlot> plots;
    NetworkProjection network;
    // Optional reviewed compatibility for removal of explicitly authorized old
    // source directories. Older catalogs retain their strict reader identity.
    std::vector<RetirementReaderContext> retirement_readers;
    std::optional<Profile> resource_template;
    std::optional<ResourcePreparation> resource_preparation;
    bool complete = false;
};
namespace inventory_detail
{
inline void RootHandle::matches(InventoryRoot const& root) const
{
    auto const s = stat(handle); RootHandle current(path); auto const now = stat(current.handle);
    bool const durable = file_io::has_durable_identity(root.durable);
    require(file_io::valid_durable_identity(root.durable) && (uint64_t(s.st_dev) == root.device || durable) && uint64_t(s.st_ino) == root.inode &&
        s.st_dev == now.st_dev && s.st_ino == now.st_ino &&
        (!durable || (file_io::matches_durable_identity(handle.fd, root.durable) &&
            file_io::matches_durable_identity(current.handle.fd, root.durable))) &&
        file_io::path_text(path) == root.path && farming_directory_identity(path) == root.directory_identity,
        "farm inventory root identity changed");
}
inline std::string timestamp()
{
    auto const now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm value{};
#ifdef _WIN32
    gmtime_s(&value, &now);
#else
    gmtime_r(&now, &value);
#endif
    std::ostringstream out; out << std::put_time(&value, "%Y-%m-%dT%H:%M:%SZ"); return out.str();
}
inline void check(InventoryLimits const& limits, std::chrono::steady_clock::time_point deadline)
{
    require(!limits.cancelled || !limits.cancelled(), "farm inventory cancelled");
    require(std::chrono::steady_clock::now() < deadline, "farm inventory deadline exceeded");
}
inline void validate_limits(InventoryLimits const& l)
{
    require(l.memory_bytes >= 1024 * 1024 && l.max_files && l.max_files <= 1000000 &&
        l.max_roots && l.max_roots <= 128 && l.max_header_bytes >= 512 && l.max_header_bytes <= 65536 &&
        l.timeout_ms && l.timeout_ms <= 3600000 && l.max_output_bytes && l.max_output_bytes <= l.memory_bytes / 4,
        "invalid farm inventory limits");
}
inline std::chrono::steady_clock::time_point catalog_deadline(InventoryLimits const& limits, uint64_t entries)
{
    validate_limits(limits);
    require(entries <= limits.max_files && entries < UINT64_MAX && entries + 1 <= UINT64_MAX / limits.timeout_ms,
        "farm catalog operation allowance overflows its entry bound");
    // One bounded operation per entry plus metadata/final provenance work.
    // This does not enlarge an individual RPC or native checker grant.
    uint64_t const milliseconds = (entries + 1) * limits.timeout_ms;
    auto const now = std::chrono::steady_clock::now();
    auto const available = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::time_point::max() - now).count();
    require(available > 0 && milliseconds <= uint64_t(available), "farm catalog deadline exceeds the monotonic clock range");
    return now + std::chrono::milliseconds(milliseconds);
}
inline bool hex(std::string const& text, size_t bytes)
{ return text.size() == bytes * 2 && text.find_first_not_of("0123456789abcdef") == text.npos; }
inline stream_state::Stamp stamp(file_io::Stat const& s)
{
#ifdef __APPLE__
    auto const m = s.st_mtimespec, c = s.st_ctimespec;
#else
    auto const m = s.st_mtim, c = s.st_ctim;
#endif
    require(s.st_size >= 0, "negative farm inventory file size");
    return {uint64_t(s.st_dev), uint64_t(s.st_ino), uint64_t(s.st_size), m.tv_sec, c.tv_sec,
        uint32_t(m.tv_nsec), uint32_t(c.tv_nsec)};
}
inline uint64_t number(std::span<uint8_t const> data, size_t position, size_t bytes, bool little = false)
{
    require(position <= data.size() && bytes <= data.size() - position && bytes <= 8, "truncated plot header");
    uint64_t value = 0; for (size_t i = 0; i < bytes; ++i) value = (value << 8) | data[position + (little ? bytes - i - 1 : i)]; return value;
}
inline void memo(InventoryPlot& plot, std::span<uint8_t const> data)
{
    require(data.size() == 112 || data.size() == 128, "unsupported public plot memo layout");
    size_t const pool = data.size() - 80;
    if (pool == 32) plot.pool_contract_puzzle_hash = group_tools::hex(data.first(pool));
    else plot.pool_public_key = group_tools::hex(data.first(pool));
    plot.farmer_public_key = group_tools::hex(data.subspan(pool, 48));
    // The final local master secret is deliberately never copied or exported.
}
inline void header(InventoryPlot& p, std::span<uint8_t const> data)
{
    auto match = [&](std::string_view text) { return data.size() >= text.size() && !std::memcmp(data.data(), text.data(), text.size()); };
    if (match(PlotGroupFile::MAGIC_STR))
    {
        require(data.size() >= sizeof(PlotGroupFile::Header), "truncated ordinary group header");
        PlotGroupFile::Header h{}; std::memcpy(&h, data.data(), sizeof(h));
        require(h.version == PlotGroupFile::FORMAT_VERSION && h.k >= 18 && h.k <= 32 && h.strength <= 32 &&
            h.group_size && h.chunk_index_offset >= sizeof(h) + h.memo_length &&
            h.chunk_index_offset < p.identity.stamp.bytes && h.memo_length <= data.size() - sizeof(h), "invalid ordinary group header");
        p.algorithm = PlotAlgorithm::pos2; p.container = PlotContainer::pos2_group;
        p.plot_id = h.group_id.to_string(); p.k = h.k; p.strength = h.strength; p.meta_group = h.meta_group; p.members = h.group_size;
        memo(p, data.subspan(sizeof(h), h.memo_length)); p.reader = "pos2-group"; p.reader_version = "2";
        p.compression_family = "pos2-strength";
    }
    else if (match(std::string_view("Proof of Space Plot\0", 19)) || match("PLOT"))
    {
        bool const v2 = match("PLOT"); size_t at = v2 ? 8 : 19;
        if (v2) require(number(data, 4, 4, true) == 2, "unsupported chiapos container version");
        require(at + 33 <= data.size(), "truncated chiapos identity");
        p.algorithm = PlotAlgorithm::pos1; p.container = v2 ? PlotContainer::chiapos_v2 : PlotContainer::chiapos_v1;
        p.plot_id = group_tools::hex(data.subspan(at, 32)); at += 32; p.k = data[at++];
        require(p.k >= 18 && p.k <= 50, "unsupported chiapos plot size");
        if (!v2)
        {
            auto const length = number(data, at, 2); at += 2;
            require(length == 4 && at + length <= data.size() && !std::memcmp(data.data() + at, "v1.0", 4), "unsupported chiapos format description"); at += size_t(length);
        }
        auto const length = number(data, at, 2); at += 2;
        require(length <= data.size() - at, "truncated chiapos memo"); memo(p, data.subspan(at, size_t(length))); at += size_t(length);
        if (v2)
        {
            auto const flags = number(data, at, 4, true); at += 4;
            require((flags & ~uint64_t(1)) == 0, "unknown chiapos container flags");
            if (flags & 1) { p.compression_level = uint8_t(number(data, at, 1)); ++at; }
        }
        require(at + 80 <= data.size(), "truncated chiapos table pointers");
        uint64_t previous = 0;
        for (unsigned i = 0; i < 10; ++i)
        {
            auto const offset = number(data, at + i * 8, 8);
            // Compressed tables can be omitted (zero). Nonempty tables retain
            // their ordered, bounded upstream offsets.
            if (offset) { require(offset >= at + 80 && offset >= previous && offset < p.identity.stamp.bytes, "invalid chiapos table pointer"); previous = offset; }
        }
        require(previous != 0, "empty chiapos table index");
        p.members = 1; p.reader = "chiapos"; p.reader_version = v2 ? "container-2" : "v1.0";
        p.compression_family = p.compression_level ? "bladebit" : "uncompressed";
    }
    else { p.status = "protected"; p.reason = "unknown container; an installed supported reader must identify it"; return; }
    p.header_validated = true; p.status = "observed"; p.reason = "public header only; complete validity and farming resources are unverified";
}
inline InventoryPlot observe_handle(Handle const& handle, std::string const& relative_path, InventoryLimits const& limits)
{
    auto const before = stat(handle);
    InventoryPlot result; result.relative_path = relative_path; result.identity.stamp = stamp(before);
    require(before.st_blocks >= 0 && uint64_t(before.st_blocks) <= UINT64_MAX / 512, "invalid allocated file size");
    result.identity.allocated_bytes = uint64_t(before.st_blocks) * 512; result.identity.links = uint64_t(before.st_nlink);
    uint64_t const bytes = std::min(result.identity.stamp.bytes, limits.max_header_bytes);
    std::vector<uint8_t> data(size_t(bytes), uint8_t{}); size_t at = 0;
    while (at < data.size())
    {
        auto const count = file_io::read_at(handle.fd, data.data() + at, data.size() - at, at);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "farm inventory header changed while reading"); at += size_t(count);
    }
    result.identity.header_bytes = bytes; result.identity.header_digest = group_tools::hex(grow_digest(data));
    (void)file_io::capture_durable_identity(handle.fd, &result.identity.durable);
    require(stamp(stat(handle)) == result.identity.stamp, "farm inventory source changed while reading");
    try { header(result, data); }
    catch (std::exception const& error)
    {
        // Partially parsed metadata cannot confer candidate status.
        auto identity = result.identity; result = {}; result.relative_path = relative_path; result.identity = std::move(identity);
        result.status = "protected"; result.reason = error.what();
    }
    if (result.identity.links != 1) { result.status = "protected"; result.reason = "multiple links; allocated bytes are not independently reclaimable"; }
    return result;
}
inline InventoryPlot observe(RootHandle const& root, std::string const& relative_path, InventoryLimits const& limits)
{ SourceHandle source(root, relative_path); return observe_handle(source.handle, relative_path, limits); }
inline uint64_t plot_memory(InventoryPlot const& p)
{
    uint64_t bytes = sizeof(p) + 256; // Native-identity map node and allocator overhead.
    for (auto const* s : {&p.relative_path, &p.identity.header_digest, &p.identity.durable.filesystem, &p.identity.durable.object, &p.compression_family, &p.plot_id, &p.farmer_public_key,
        &p.pool_public_key, &p.pool_contract_puzzle_hash, &p.plot_public_key, &p.reader, &p.reader_version, &p.reader_identity, &p.checker_identity,
        &p.host, &p.harvester, &p.solver, &p.workload_id, &p.status, &p.reason})
        bytes += 2 * s->capacity();
    return bytes;
}
inline uint64_t memory(FarmInventory const& value)
{
    uint64_t bytes = sizeof(value) + value.roots.capacity() * sizeof(InventoryRoot) + value.plots.capacity() * sizeof(InventoryPlot);
    auto charge = [&](std::string const& s) { require(s.capacity() <= (UINT64_MAX - bytes) / 2, "farm inventory memory overflow"); bytes += 2 * s.capacity(); };
    charge(value.host); charge(value.captured_at); charge(value.producer_build);
    for (auto const& r : value.roots) { charge(r.path); charge(r.directory_identity); charge(r.storage.filesystem);
        charge(r.durable.filesystem); charge(r.durable.object); for (auto const& d : r.storage.drives) charge(d); }
    for (auto const* s : {&value.network.network, &value.network.genesis_challenge, &value.network.source, &value.network.version, &value.network.captured_at,
        &value.network.target_route, &value.network.reader_version, &value.network.reader_identity,
        &value.network.farmer_version, &value.network.solver_version, &value.network.service_status}) charge(*s);
    if (value.resource_template) bytes += profile_memory(*value.resource_template);
    if (value.resource_preparation) bytes = resource_detail::add(bytes, resource_preparation_memory(*value.resource_preparation));
    for (auto const& p : value.plots) bytes += plot_memory(p) - sizeof(p);
    bytes += value.retirement_readers.capacity() * sizeof(RetirementReaderContext);
    for (auto const& reader : value.retirement_readers)
    {
        charge(reader.reader_identity); charge(reader.retirement_identity);
        bytes += reader.configurations.capacity() * sizeof(RetirementReaderConfiguration);
        for (auto const& config : reader.configurations)
        {
            charge(config.root_identity); bytes += config.plot_directories.capacity() * sizeof(std::string);
            for (auto const& directory : config.plot_directories) charge(directory);
        }
    }
    return bytes;
}
} // namespace inventory_detail

inline InventoryPlot observe_inventory_file(InventoryRoot const& root, std::string const& relative_path, InventoryLimits const& limits = {})
{
    inventory_detail::validate_limits(limits); inventory_detail::RootHandle handle(file_io::native_path(root.path)); handle.matches(root);
    auto result = inventory_detail::observe(handle, relative_path, limits); handle.matches(root); return result;
}
inline bool same_inventory_contents(InventoryFileIdentity const& a, InventoryFileIdentity const& b)
{
    return stream_state::same_output(a.stamp, b.stamp) && a.allocated_bytes == b.allocated_bytes && a.links == b.links &&
        a.header_bytes == b.header_bytes && a.header_digest == b.header_digest;
}
// Directional comparison: fresh evidence cannot upgrade a legacy saved row.
// Callers first bind the current root and obtain current from its pinned file.
inline bool matches_inventory_identity(InventoryFileIdentity const& saved, InventoryFileIdentity const& current, bool renamed = false)
{
    if (!file_io::valid_durable_identity(saved.durable) || !file_io::valid_durable_identity(current.durable)) return false;
    auto observed = current.stamp;
    bool const durable = file_io::has_durable_identity(saved.durable);
    if (durable && saved.durable != current.durable) return false;
    if (durable) observed.device = saved.stamp.device;
    return (renamed ? stream_state::same_output(saved.stamp, observed) : saved.stamp == observed) &&
        saved.allocated_bytes == current.allocated_bytes && saved.links == current.links &&
        saved.header_bytes == current.header_bytes && saved.header_digest == current.header_digest;
}
class InventorySourcePin
{
    InventoryRoot root_;
    InventoryPlot plot_;
    inventory_detail::RootHandle root_handle_;
    inventory_detail::SourceHandle source_;
public:
    InventorySourcePin(InventoryRoot const& root, InventoryPlot const& plot, InventoryLimits const& limits = {}, std::string current_relative = {}, bool retirement = false)
        : root_(root), plot_(plot), root_handle_(file_io::native_path(root.path)),
          source_(root_handle_, current_relative.empty() ? plot.relative_path : current_relative, retirement)
    {
        inventory_detail::validate_limits(limits); root_handle_.matches(root_);
        auto const current = inventory_detail::observe_handle(source_.handle, current_relative.empty() ? plot.relative_path : current_relative, limits);
        inventory_detail::require(matches_inventory_identity(plot.identity, current.identity, !current_relative.empty()),
            "farm inventory source changed before retirement pin");
        recheck_parent(current_relative.empty() ? plot.relative_path : current_relative);
    }
    InventoryFileIdentity identity(std::string const& relative, InventoryLimits const& limits) const
    { return inventory_detail::observe_handle(source_.handle, relative, limits).identity; }
    int close_source() { return file_io::close_file(std::exchange(source_.handle.fd, -1)); }
    int root_fd() const { return root_handle_.handle.fd; }
    int source_fd() const { return source_.handle.fd; }
    int source_parent_fd() const { return source_.parent.fd; }
    std::string const& source_basename() const { return source_.basename; }
    void recheck_parent(std::string const& current_relative_path) const
    {
        root_handle_.matches(root_);
#ifdef _WIN32
        file_io::Stat current{}, parent{};
        auto const path = root_handle_.path / inventory_detail::relative(current_relative_path);
        inventory_detail::require(!file_io::inspect_link(path, &current) && !file_io::inspect_link(path.parent_path(), &parent),
            "cannot inspect pinned retirement namespace");
        auto const held_parent = inventory_detail::stat(source_.parent);
        inventory_detail::require(parent.st_dev == held_parent.st_dev && parent.st_ino == held_parent.st_ino &&
            inventory_detail::stamp(current) == inventory_detail::stamp(inventory_detail::stat(source_.handle)),
            "farm retirement parent or source path changed");
#else
        inventory_detail::SourceHandle current(root_handle_, current_relative_path);
        auto const a = inventory_detail::stat(source_.parent), b = inventory_detail::stat(current.parent);
        inventory_detail::require(a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
            inventory_detail::stamp(inventory_detail::stat(current.handle)) == inventory_detail::stamp(inventory_detail::stat(source_.handle)),
            "farm retirement parent or source path changed");
#endif
    }
    bool matches_claim(std::string const& claimed_relative_path, InventoryLimits const& limits = {}) const
    {
        root_handle_.matches(root_);
        auto const held = inventory_detail::stamp(inventory_detail::stat(source_.handle));
#ifdef _WIN32
        file_io::Stat name{};
        if (file_io::inspect_link(root_handle_.path / inventory_detail::relative(claimed_relative_path), &name) ||
            inventory_detail::stamp(name) != held) return false;
        auto const claimed = inventory_detail::observe_handle(source_.handle, claimed_relative_path, limits);
#else
        auto const claimed = inventory_detail::observe(root_handle_, claimed_relative_path, limits);
#endif
        return stream_state::same_output(held, claimed.identity.stamp) && matches_inventory_identity(plot_.identity, claimed.identity, true);
    }
};
inline InventoryFileIdentity recheck_inventory_source(InventoryRoot const& root, InventoryPlot const& plot, InventoryLimits const& limits = {})
{
    auto current = observe_inventory_file(root, plot.relative_path, limits);
    inventory_detail::require(matches_inventory_identity(plot.identity, current.identity), "farm inventory source identity changed"); return current.identity;
}
inline uint64_t farm_inventory_memory(FarmInventory const& value) { return inventory_detail::memory(value); }

namespace inventory_detail
{
inline char const* name(PlotAlgorithm value)
{
    switch (value) { case PlotAlgorithm::unknown: return "unknown"; case PlotAlgorithm::pos1: return "pos1"; case PlotAlgorithm::pos2: return "pos2"; }
    throw std::runtime_error("unsupported inventory algorithm");
}
inline char const* name(PlotContainer value)
{
    switch (value) { case PlotContainer::unknown: return "unknown"; case PlotContainer::chiapos_v1: return "chiapos-v1";
        case PlotContainer::chiapos_v2: return "chiapos-v2"; case PlotContainer::pos2_group: return "pos2-group-v2"; }
    throw std::runtime_error("unsupported inventory container");
}
inline void validate(FarmInventory const& v, InventoryLimits const& limits)
{
    validate_limits(limits); require((v.version == 1 || v.version == 2) && !v.roots.empty() && v.roots.size() <= limits.max_roots &&
        v.plots.size() <= limits.max_files && memory(v) < limits.memory_bytes / 2, "farm inventory exceeds its schema or memory bound");
    for (auto const* text : {&v.host, &v.captured_at, &v.producer_build, &v.network.network, &v.network.genesis_challenge,
        &v.network.source, &v.network.version, &v.network.captured_at, &v.network.target_route, &v.network.reader_version, &v.network.reader_identity,
        &v.network.farmer_version, &v.network.solver_version, &v.network.service_status}) bounded_text(*text);
    require(!v.network.available || (!v.network.network.empty() && hex(v.network.genesis_challenge, 32)), "invalid inventory network projection");
    if (v.resource_preparation) validate_resource_preparation(*v.resource_preparation);
    require(v.retirement_readers.size() <= limits.max_roots, "inventory retirement reader count exceeds its bound");
    std::set<std::string> readers;
    for (auto const& reader : v.retirement_readers)
    {
        require(hex(reader.reader_identity, 32) && hex(reader.retirement_identity, 32) && readers.insert(reader.reader_identity).second &&
            !reader.configurations.empty() && reader.configurations.size() <= 5, "invalid inventory retirement reader context");
        std::set<std::string> configurations;
        for (auto const& config : reader.configurations)
        {
            require(hex(config.root_identity, 32) && configurations.insert(config.root_identity).second &&
                config.plot_directories.size() <= limits.max_roots, "invalid inventory retirement configuration");
            std::set<std::string> directories;
            for (auto const& directory : config.plot_directories)
            {
                bounded_text(directory); auto path = file_io::native_path(directory);
                require(path.is_absolute() && file_io::path_text(path.lexically_normal()) == directory && directories.insert(directory).second,
                    "invalid reviewed retirement directory");
            }
        }
    }
    std::set<std::pair<uint64_t, uint64_t>> roots; bool durable = false;
    for (auto const& r : v.roots)
    {
        bounded_text(r.path); bounded_text(r.directory_identity); bounded_text(r.storage.filesystem);
        require(file_io::native_path(r.path).is_absolute() && r.inode && roots.emplace(r.device, r.inode).second &&
            r.storage.drives.size() <= 128 && file_io::valid_durable_identity(r.durable), "invalid or duplicate farm inventory root");
        durable = durable || file_io::has_durable_identity(r.durable);
        for (auto const& d : r.storage.drives) bounded_text(d);
    }
    std::map<std::pair<uint64_t, uint64_t>, size_t> identities;
    std::set<std::pair<size_t, std::string>> paths;
    for (size_t i = 0; i < v.plots.size(); ++i)
    {
        auto const& p = v.plots[i]; auto const& id = p.identity;
        (void)relative(p.relative_path); (void)name(p.algorithm); (void)name(p.container);
        require(p.root < v.roots.size() && paths.emplace(p.root, p.relative_path).second && (!p.alias_of || *p.alias_of < i) && id.header_bytes <= limits.max_header_bytes &&
            id.header_bytes <= id.stamp.bytes && (id.header_digest.empty() || hex(id.header_digest, 32)) &&
            id.stamp.mtime_ns < 1000000000 && id.stamp.ctime_ns < 1000000000 && file_io::valid_durable_identity(id.durable), "invalid farm inventory file identity");
        if (file_io::has_durable_identity(id.durable))
        {
            auto const& root = v.roots[p.root];
            require(file_io::has_durable_identity(root.durable) && root.durable.scheme == id.durable.scheme &&
                root.durable.filesystem == id.durable.filesystem && root.device == id.stamp.device,
                "inventory file evidence differs from its native root");
            durable = true;
        }
        for (auto const* text : {&p.compression_family, &p.plot_id, &p.farmer_public_key, &p.pool_public_key, &p.pool_contract_puzzle_hash, &p.plot_public_key,
            &p.reader, &p.reader_version, &p.reader_identity, &p.checker_identity, &p.host, &p.harvester, &p.solver, &p.workload_id, &p.status, &p.reason}) bounded_text(*text);
        require((p.plot_id.empty() || hex(p.plot_id, 32)) && (p.farmer_public_key.empty() || hex(p.farmer_public_key, 48)) &&
            (p.pool_public_key.empty() || hex(p.pool_public_key, 48)) && (p.pool_contract_puzzle_hash.empty() || hex(p.pool_contract_puzzle_hash, 32)) &&
            (p.plot_public_key.empty() || hex(p.plot_public_key, 48)),
            "invalid inventory public identifier");
        if (id.stamp.inode)
        {
            auto [at, fresh] = identities.emplace(std::pair(id.stamp.device, id.stamp.inode), i);
            require(fresh ? !p.alias_of : p.alias_of && *p.alias_of == at->second &&
                same_inventory_contents(v.plots[at->second].identity, id) && p.status == "protected" && v.plots[at->second].status == "protected",
                "inventory aliases disagree with native identities");
        }
        else require(!p.alias_of, "unknown inventory file cannot certify an alias");
        require(!p.header_validated || (p.algorithm != PlotAlgorithm::unknown && p.container != PlotContainer::unknown &&
            !p.plot_id.empty() && !p.farmer_public_key.empty() &&
            (p.pool_public_key.empty() != p.pool_contract_puzzle_hash.empty()) && !id.header_digest.empty()), "incomplete inventory public header");
        require(!p.native_identified || (p.algorithm == PlotAlgorithm::pos1 && !p.plot_id.empty() && p.k >= 18 && p.k <= 50 &&
            !p.reader.empty() && !p.reader_version.empty() && hex(p.reader_identity, 32) && !p.plot_public_key.empty() &&
            (p.pool_public_key.empty() != p.pool_contract_puzzle_hash.empty()) && !id.header_digest.empty()), "incomplete native inventory observation");
    }
    require(v.version == (durable ? 2u : 1u), "farm inventory native evidence differs from its schema");
}
inline void write_durable_identity(std::ostream& out, file_io::DurableIdentity const& value)
{
    require(file_io::has_durable_identity(value), "invalid durable native identity");
    out << "{\"scheme\":" << unsigned(value.scheme) << ",\"filesystem\":"; profile_detail::quote(out, value.filesystem);
    out << ",\"handle_type\":" << value.handle_type << ",\"object\":"; profile_detail::quote(out, value.object); out << '}';
}
inline void write_identity(std::ostream& out, InventoryFileIdentity const& id)
{
    auto const& s = id.stamp;
    out << "{\"device\":" << s.device << ",\"inode\":" << s.inode << ",\"bytes\":" << s.bytes << ",\"mtime\":" << s.mtime
        << ",\"ctime\":" << s.ctime << ",\"mtime_ns\":" << s.mtime_ns << ",\"ctime_ns\":" << s.ctime_ns
        << ",\"allocated_bytes\":" << id.allocated_bytes << ",\"links\":" << id.links << ",\"header_bytes\":" << id.header_bytes << ",\"header_digest\":";
    profile_detail::quote(out, id.header_digest);
    if (file_io::has_durable_identity(id.durable)) { out << ",\"durable_identity\":"; write_durable_identity(out, id.durable); }
    out << '}';
}
using Json = profile_detail::Json;
inline Json const& required_field(Json const& object, char const* key)
{ auto const* value = profile_detail::field(object, key); require(value, std::string("missing inventory field: ") + key); return *value; }
inline std::string string(Json const& object, char const* key) { return profile_detail::text(required_field(object, key)); }
inline uint64_t integer(Json const& object, char const* key) { return profile_detail::integer(required_field(object, key)); }
inline bool boolean(Json const& object, char const* key) { return profile_detail::boolean(required_field(object, key)); }
inline int64_t signed_integer(Json const& object, char const* key)
{
    auto const& value = required_field(object, key); require(value.kind == Json::number, "invalid inventory time"); int64_t result = 0;
    auto p = std::from_chars(value.scalar.data(), value.scalar.data() + value.scalar.size(), result);
    require(p.ec == std::errc{} && p.ptr == value.scalar.data() + value.scalar.size(), "invalid inventory time"); return result;
}
inline file_io::DurableIdentity read_durable_identity(Json const& object, unsigned version)
{
    auto const* value = profile_detail::field(object, "durable_identity"); if (!value) return {};
    require(version == 2 && value->kind == Json::object && value->fields.size() == 4, "invalid durable native identity schema");
    auto const scheme = integer(*value, "scheme"), type = integer(*value, "handle_type");
    require(scheme <= unsigned(file_io::DurableIdentity::Scheme::windows_file_id) && type <= UINT32_MAX, "invalid durable native identity representation");
    file_io::DurableIdentity result{file_io::DurableIdentity::Scheme(scheme), string(*value, "filesystem"), string(*value, "object"), uint32_t(type)};
    require(file_io::has_durable_identity(result), "invalid durable native identity"); return result;
}
inline InventoryFileIdentity read_identity(Json const& j, unsigned version = 1)
{
    InventoryFileIdentity result; auto& s = result.stamp;
    s.device = integer(j, "device"); s.inode = integer(j, "inode"); s.bytes = integer(j, "bytes"); s.mtime = signed_integer(j, "mtime"); s.ctime = signed_integer(j, "ctime");
    auto m = integer(j, "mtime_ns"), c = integer(j, "ctime_ns"); require(m < 1000000000 && c < 1000000000, "invalid inventory time precision");
    s.mtime_ns = uint32_t(m); s.ctime_ns = uint32_t(c); result.allocated_bytes = integer(j, "allocated_bytes");
    result.links = integer(j, "links"); result.header_bytes = integer(j, "header_bytes"); result.header_digest = string(j, "header_digest");
    result.durable = read_durable_identity(j, version); return result;
}
class OutputLimit : public std::streambuf
{
    std::streambuf* target_;
    uint64_t remaining_;
protected:
    std::streamsize xsputn(char const* text, std::streamsize count) override
    {
        require(count >= 0 && uint64_t(count) <= remaining_, "farm inventory exceeds its output byte bound");
        auto const written = target_->sputn(text, count); if (written > 0) remaining_ -= uint64_t(written); return written;
    }
    int_type overflow(int_type c) override
    {
        if (traits_type::eq_int_type(c, traits_type::eof())) return traits_type::not_eof(c);
        require(remaining_, "farm inventory exceeds its output byte bound");
        auto const result = target_->sputc(traits_type::to_char_type(c));
        if (!traits_type::eq_int_type(result, traits_type::eof())) --remaining_; return result;
    }
public:
    OutputLimit(std::ostream& target, uint64_t bytes) : target_(target.rdbuf()), remaining_(bytes) {}
};
} // namespace inventory_detail

inline void write_farm_inventory_json(std::ostream& destination, FarmInventory const& v, InventoryLimits const& limits = {})
{
    inventory_detail::validate(v, limits);
    inventory_detail::OutputLimit buffer(destination, limits.max_output_bytes); std::ostream out(&buffer);
    auto quote = [&](std::string const& text) { profile_detail::quote(out, text); };
    auto text = [&](char const* key, std::string const& value) { out << ",\"" << key << "\":"; quote(value); };
    out << "{\"schema_version\":" << v.version << ",\"kind\":\"farm-inventory\""; text("host", v.host); text("captured_at", v.captured_at); text("producer_build", v.producer_build);
    out << ",\"complete\":" << (v.complete ? "true" : "false") << ",\"network\":{\"available\":" << (v.network.available ? "true" : "false");
    text("network", v.network.network); text("genesis_challenge", v.network.genesis_challenge); text("source", v.network.source); text("version", v.network.version); text("captured_at", v.network.captured_at);
    text("target_route", v.network.target_route); text("reader_version", v.network.reader_version); text("reader_identity", v.network.reader_identity);
    text("farmer_version", v.network.farmer_version); text("solver_version", v.network.solver_version); text("service_status", v.network.service_status);
    out << ",\"hard_fork2_height\":" << v.network.hard_fork2_height << ",\"min_strength\":" << v.network.min_strength
        << ",\"max_strength\":" << v.network.max_strength << ",\"plot_size\":" << v.network.plot_size << "},\"roots\":[";
    bool comma = false;
    for (auto const& r : v.roots)
    {
        if (comma) out << ','; comma = true; out << "{\"path\":"; quote(r.path); text("directory_identity", r.directory_identity);
        text("filesystem", r.storage.filesystem); out << ",\"device\":" << r.device << ",\"inode\":" << r.inode
            << ",\"nonrotating\":" << (r.storage.nonrotating ? "true" : "false") << ",\"rotating\":" << (r.storage.rotating ? "true" : "false") << ",\"drives\":[";
        bool drive_comma = false; for (auto const& d : r.storage.drives) { if (drive_comma) out << ','; drive_comma = true; quote(d); } out << ']';
        if (file_io::has_durable_identity(r.durable)) { out << ",\"durable_identity\":"; inventory_detail::write_durable_identity(out, r.durable); }
        out << '}';
    }
    out << "],\"plots\":["; comma = false;
    for (auto const& p : v.plots)
    {
        if (comma) out << ','; comma = true;
        out << "{\"root\":" << p.root; text("relative_path", p.relative_path); out << ",\"identity\":"; inventory_detail::write_identity(out, p.identity);
        text("algorithm", inventory_detail::name(p.algorithm)); text("container", inventory_detail::name(p.container));
        out << ",\"k\":" << unsigned(p.k) << ",\"strength\":" << unsigned(p.strength) << ",\"meta_group\":" << unsigned(p.meta_group)
            << ",\"members\":" << p.members << ",\"compression_level\":" << unsigned(p.compression_level)
            << ",\"header_validated\":" << (p.header_validated ? "true" : "false") << ",\"alias_of\":";
        if (p.alias_of) out << *p.alias_of; else out << "null";
        out << ",\"native_identified\":" << (p.native_identified ? "true" : "false");
        text("compression_family", p.compression_family); text("plot_id", p.plot_id); text("farmer_public_key", p.farmer_public_key);
        text("pool_public_key", p.pool_public_key); text("pool_contract_puzzle_hash", p.pool_contract_puzzle_hash);
        text("plot_public_key", p.plot_public_key);
        text("reader", p.reader); text("reader_version", p.reader_version); text("host", p.host); text("harvester", p.harvester); text("solver", p.solver);
        text("reader_identity", p.reader_identity); text("checker_identity", p.checker_identity);
        text("workload_id", p.workload_id); text("status", p.status); text("reason", p.reason); out << '}';
    }
    out << ']';
    if (!v.retirement_readers.empty())
    {
        out << ",\"retirement_readers\":["; bool reader_comma = false;
        for (auto const& reader : v.retirement_readers)
        {
            if (reader_comma) out << ','; reader_comma = true; out << "{\"reader_identity\":"; quote(reader.reader_identity);
            text("retirement_identity", reader.retirement_identity); out << ",\"configurations\":["; bool config_comma = false;
            for (auto const& config : reader.configurations)
            {
                if (config_comma) out << ','; config_comma = true; out << "{\"root_identity\":"; quote(config.root_identity);
                out << ",\"plot_directories\":["; bool directory_comma = false;
                for (auto const& directory : config.plot_directories)
                { if (directory_comma) out << ','; directory_comma = true; quote(directory); }
                out << "]}";
            }
            out << "]}";
        }
        out << ']';
    }
    if (v.resource_template) { out << ",\"farm_resources\":"; write_profile_json(out, *v.resource_template); }
    if (v.resource_preparation) { out << ",\"resource_preparation\":"; write_resource_preparation_json(out, *v.resource_preparation); }
    out << '}'; inventory_detail::require(bool(out), "cannot write farm inventory");
}
inline std::string farm_inventory_text(FarmInventory const& value, InventoryLimits const& limits = {})
{
    std::ostringstream out; write_farm_inventory_json(out, value, limits); auto text = out.str();
    inventory_detail::require(text.size() <= limits.max_output_bytes, "farm inventory exceeds its output byte bound"); return text;
}
inline FarmInventory read_farm_inventory(std::istream& in, InventoryLimits const& limits = {})
{
    using namespace inventory_detail;
    validate_limits(limits); auto input = profile_detail::read_text(in, limits.max_output_bytes);
    auto j = profile_detail::JsonReader(input, limits.memory_bytes / 2, limits.memory_bytes / 1024).read();
    auto const version = integer(j, "schema_version");
    require((version == 1 || version == 2) && string(j, "kind") == "farm-inventory", "unsupported farm inventory schema");
    FarmInventory result; result.version = unsigned(version); result.host = string(j, "host"); result.captured_at = string(j, "captured_at"); result.producer_build = string(j, "producer_build");
    result.complete = boolean(j, "complete"); auto const& n = required_field(j, "network"); result.network.available = boolean(n, "available");
    result.network.network = string(n, "network"); result.network.genesis_challenge = string(n, "genesis_challenge"); result.network.source = string(n, "source");
    result.network.version = string(n, "version"); result.network.captured_at = string(n, "captured_at"); result.network.hard_fork2_height = integer(n, "hard_fork2_height");
    result.network.target_route = string(n, "target_route"); result.network.reader_version = string(n, "reader_version"); result.network.reader_identity = string(n, "reader_identity");
    for (auto const& [name, target] : {std::pair{"farmer_version", &result.network.farmer_version},
        std::pair{"solver_version", &result.network.solver_version}, std::pair{"service_status", &result.network.service_status}})
        if (auto const* value = profile_detail::field(n, name)) *target = profile_detail::text(*value);
    result.network.min_strength = integer(n, "min_strength"); result.network.max_strength = integer(n, "max_strength"); result.network.plot_size = integer(n, "plot_size");
    if (auto const* readers = profile_detail::field(j, "retirement_readers"))
    {
        require(readers->kind == Json::array && readers->items.size() <= limits.max_roots, "inventory retirement reader count exceeds its bound");
        auto shape = [](Json const& object, std::initializer_list<std::string_view> names)
        {
            require(object.kind == Json::object, "expected retirement reader context object");
            for (auto const& [key, child] : object.fields)
            { (void)child; require(std::find(names.begin(), names.end(), key) != names.end(), "unsupported retirement reader context field"); }
        };
        for (auto const& value : readers->items)
        {
            shape(value, {"reader_identity", "retirement_identity", "configurations"});
            RetirementReaderContext reader; reader.reader_identity = string(value, "reader_identity"); reader.retirement_identity = string(value, "retirement_identity");
            auto const& configs = required_field(value, "configurations"); require(configs.kind == Json::array && configs.items.size() <= 5, "retirement configuration count exceeds its bound");
            for (auto const& config : configs.items)
            {
                shape(config, {"root_identity", "plot_directories"});
                RetirementReaderConfiguration context; context.root_identity = string(config, "root_identity");
                auto const& dirs = required_field(config, "plot_directories"); require(dirs.kind == Json::array && dirs.items.size() <= limits.max_roots, "retirement directory count exceeds its bound");
                for (auto const& dir : dirs.items) context.plot_directories.push_back(profile_detail::text(dir));
                reader.configurations.push_back(std::move(context));
            }
            result.retirement_readers.push_back(std::move(reader));
        }
    }
    auto const& roots = required_field(j, "roots"); auto const& plots = required_field(j, "plots");
    require(roots.kind == Json::array && !roots.items.empty() && roots.items.size() <= limits.max_roots &&
        plots.kind == Json::array && plots.items.size() <= limits.max_files &&
        plots.items.size() <= limits.memory_bytes / (4 * sizeof(InventoryPlot)), "farm inventory count exceeds its bound");
    result.roots.reserve(roots.items.size()); result.plots.reserve(plots.items.size());
    for (auto const& r : roots.items)
    {
        InventoryRoot root; root.path = string(r, "path"); root.directory_identity = string(r, "directory_identity"); root.device = integer(r, "device"); root.inode = integer(r, "inode");
        root.storage.filesystem = string(r, "filesystem"); root.storage.nonrotating = boolean(r, "nonrotating"); root.storage.rotating = boolean(r, "rotating");
        root.durable = read_durable_identity(r, result.version);
        auto const& drives = required_field(r, "drives"); require(drives.kind == Json::array && drives.items.size() <= 128, "invalid inventory physical drives");
        for (auto const& d : drives.items) root.storage.drives.push_back(profile_detail::text(d)); result.roots.push_back(std::move(root));
    }
    for (auto const& p : plots.items)
    {
        InventoryPlot plot; plot.root = integer(p, "root"); plot.relative_path = string(p, "relative_path"); plot.identity = read_identity(required_field(p, "identity"), result.version);
        auto algorithm = string(p, "algorithm"); auto container = string(p, "container");
        bool found = false; for (auto a : {PlotAlgorithm::unknown, PlotAlgorithm::pos1, PlotAlgorithm::pos2}) if (algorithm == name(a)) { plot.algorithm = a; found = true; }
        require(found, "unsupported inventory algorithm"); found = false;
        for (auto c : {PlotContainer::unknown, PlotContainer::chiapos_v1, PlotContainer::chiapos_v2, PlotContainer::pos2_group}) if (container == name(c)) { plot.container = c; found = true; }
        require(found, "unsupported inventory container");
        auto byte = [&](char const* key) { auto value = integer(p, key); require(value <= 255, "invalid inventory byte parameter"); return uint8_t(value); };
        plot.k = byte("k"); plot.strength = byte("strength"); plot.meta_group = byte("meta_group"); plot.compression_level = byte("compression_level");
        auto members = integer(p, "members"); require(members <= 65535, "invalid inventory group size"); plot.members = uint16_t(members);
        plot.header_validated = boolean(p, "header_validated"); auto const& alias = required_field(p, "alias_of"); if (alias.kind != Json::null) plot.alias_of = profile_detail::integer(alias);
        plot.native_identified = boolean(p, "native_identified");
        plot.compression_family = string(p, "compression_family"); plot.plot_id = string(p, "plot_id"); plot.farmer_public_key = string(p, "farmer_public_key");
        plot.pool_public_key = string(p, "pool_public_key"); plot.pool_contract_puzzle_hash = string(p, "pool_contract_puzzle_hash");
        plot.plot_public_key = string(p, "plot_public_key");
        plot.reader = string(p, "reader"); plot.reader_version = string(p, "reader_version"); plot.host = string(p, "host"); plot.harvester = string(p, "harvester");
        plot.reader_identity = string(p, "reader_identity"); plot.checker_identity = string(p, "checker_identity");
        plot.solver = string(p, "solver"); plot.workload_id = string(p, "workload_id"); plot.status = string(p, "status"); plot.reason = string(p, "reason");
        result.plots.push_back(std::move(plot));
    }
    if (profile_detail::field(j, "farm_resources")) result.resource_template = parse_profile(j);
    if (auto const* preparation = profile_detail::field(j, "resource_preparation")) result.resource_preparation = parse_resource_preparation(*preparation);
    validate(result, limits); return result;
}
inline FarmInventory parse_farm_inventory(std::string const& text, InventoryLimits const& limits = {})
{ std::istringstream in(text); return read_farm_inventory(in, limits); }

// These are the selected adapters' physical-host declarations, after their
// authenticated observations. Peer addresses are deliberately not accepted.
struct FarmResourceBindings
{
    std::map<std::string, std::string> target_hosts;
    std::map<std::string, std::set<std::string>> workload_hosts;
};
inline Profile make_inventory_resource_template(FarmInventory const& inventory,
    std::vector<Remaining> const& counts, FarmResourceBindings const& bindings = {})
{
    using namespace resource_detail;
    std::vector<Resource> resources; std::set<std::string> hosts{inventory.host};
    for (auto const& [role, host] : bindings.target_hosts) { (void)role; hosts.insert(host); }
    for (auto const& [workload, physical] : bindings.workload_hosts)
    { (void)workload; hosts.insert(physical.begin(), physical.end()); }
    for (auto const& host : hosts)
    {
        text(host, true); resources.push_back({"cpu:" + host, host, {}, ResourceKind::cpu});
        resources.push_back({"memory:" + host, host, {}, ResourceKind::host_memory});
    }
    std::set<std::string> filesystems;
    for (auto const& root : inventory.roots) if (filesystems.insert(root.storage.filesystem).second)
    {
        Resource resource; resource.id = "storage:" + root.storage.filesystem; resource.host = inventory.host; resource.kind = ResourceKind::storage;
        if (root.storage.drives.empty()) resource.aliases.push_back("storage:unknown:" + inventory.host);
        else for (auto const& drive : root.storage.drives) resource.aliases.push_back("drive:" + drive);
        resources.push_back(std::move(resource));
    }
    auto result = make_profile_template(std::move(resources), counts, inventory.captured_at, inventory.producer_build);
    Graph graph(result);
    struct ObservedValue
    {
        std::string value; bool mixed = false;
        void add(std::string current) { if (value.empty()) value = std::move(current); else mixed = mixed || value != current; }
    };
    struct ObservedWorkload { ObservedValue format, reader, version; std::set<std::string> resources; };
    std::map<std::string, ObservedWorkload> observations;
    for (auto const& workload : result.workloads) observations.emplace(workload.id, ObservedWorkload{});
    for (auto const& plot : inventory.plots) if (!plot.alias_of)
    {
        auto at = observations.find(plot.workload_id); if (at == observations.end()) continue;
        auto& observed = at->second;
        observed.resources.insert(graph.resources[graph.find("storage:" + inventory.roots.at(plot.root).storage.filesystem)].id);
        if (plot.header_validated || plot.native_identified)
            observed.format.add(std::string(inventory_detail::name(plot.algorithm)) + '/' + inventory_detail::name(plot.container));
        else observed.format.add("unknown");
        observed.reader.add(plot.reader_identity.empty() ? "unknown" : plot.reader);
        observed.version.add(plot.reader_identity.empty() ? "unknown" : plot.reader_version);
    }
    for (auto& workload : result.workloads)
    {
        auto& observed = observations.at(workload.id); auto& used = observed.resources;
        if (!observed.format.mixed && !observed.format.value.empty()) workload.format = observed.format.value;
        if (!observed.reader.mixed && !observed.reader.value.empty()) workload.decoder = observed.reader.value;
        if (!observed.version.mixed && !observed.version.value.empty()) workload.decoder_version = observed.version.value;
        if (auto at = bindings.workload_hosts.find(workload.id); at != bindings.workload_hosts.end()) for (auto const& host : at->second)
        { used.insert(graph.resources[graph.find("cpu:" + host)].id); used.insert(graph.resources[graph.find("memory:" + host)].id); }
        workload.uses.clear();
        for (auto const& id : used) { Use use; use.resource = id; workload.uses.push_back(std::move(use)); }
    }
    auto& native = result.native; native.host = inventory.host; native.service_hosts = bindings.target_hosts;
    if (inventory.network.available) native.k = inventory.network.plot_size;
    for (auto const& resource : graph.resources) if (resource.kind == ResourceKind::storage) native.source_resources.push_back(resource.id);
    // The inventory roots are source paths, not an implicit output/scratch
    // selection. Populate only CPU/RAM and source supervision edges we know.
    auto local = [&](std::vector<Demand>& values)
    { for (auto const& id : {"cpu:" + inventory.host, "memory:" + inventory.host}) { Demand demand; demand.resource = id; values.push_back(std::move(demand)); } };
    local(native.construction); local(native.verification); local(native.supervision);
    for (auto const& id : native.source_resources) { Demand demand; demand.resource = id; native.supervision.push_back(std::move(demand)); }
    validate(result); return result;
}
inline ResourcePreparation prepare_farm_inventory_resources(FarmInventory const& inventory,
    FarmResourceBindings const& bindings = {}, bool reused_declarations = false)
{
    using namespace resource_detail;
    require(inventory.resource_template.has_value(), "resource preparation requires an inventory resource profile");
    auto const& profile = *inventory.resource_template; auto result = prepare_resource_declarations(profile);
    auto fact = [&](std::string path, std::string value, std::string basis, std::string source, std::string scope)
    {
        require(result.facts.size() < max_preparation_entries, "too many resource preparation observations");
        result.facts.push_back({std::move(path), std::move(value), std::move(basis), std::move(source), std::move(scope)});
    };
    fact("farm_resources", profile.source, reused_declarations ? "declared" : "unknown", profile.producer,
        reused_declarations ? "Saved declarations copied unchanged; observations below do not revalidate their worst-case bounds or physical scope." :
            "Editable topology skeleton; no resource capacity, foreign queue bound or full-proof reserve has been established.");
    fact("local.logical_cpus", std::to_string(profile_detail::logical_cpus()), "observed", "local-os",
        "This process's affinity when available, otherwise hardware concurrency; not the running services' CPU grant or spare capacity.");
    fact("local.available_memory_bytes", std::to_string(profile_detail::available_memory()), "observed", "local-os",
        "Current OS availability snapshot; zero means unavailable. Not an offered budget, service memory bound or persistent capacity guarantee.");
    for (auto const& [role, host] : bindings.target_hosts)
    {
        fact("target.service_hosts." + role, host, "declared", "selected-authenticated-adapter-binding",
            "Physical host binding selected for this service; peer IP addresses are not physical resource identities.");
        auto at = profile.native.service_hosts.find(role); auto const& declared = at == profile.native.service_hosts.end() ? profile.native.host : at->second;
        if (reused_declarations && declared != host) result.missing.push_back({"physical-bindings", "farm_resources.native.service_hosts." + role,
            "Selected adapter binding differs from the saved physical host scope; review that scope and its CPU/RAM/solver/network demands before planning."});
    }
    std::set<std::string> workload_ids;
    for (auto const& workload : profile.workloads) workload_ids.insert(workload.id);
    std::map<std::string, std::pair<std::string, bool>> observed;
    auto observe = [&](std::string path, std::string value)
    { auto [at, fresh] = observed.emplace(std::move(path), std::pair{value, false}); if (!fresh) at->second.second = at->second.second || at->second.first != value; };
    for (auto const& plot : inventory.plots) if (!plot.alias_of && workload_ids.contains(plot.workload_id))
    {
        auto const path = "inventory.workloads[" + plot.workload_id + "]";
        observe(path + ".format", std::string(inventory_detail::name(plot.algorithm)) + '/' + inventory_detail::name(plot.container));
        if (plot.header_validated || plot.native_identified) observe(path + ".parameters", "k=" + std::to_string(plot.k) + ";strength=" +
            std::to_string(plot.strength) + ";compression=" + std::to_string(plot.compression_level));
        if (!plot.reader_identity.empty())
        {
            observe(path + ".reader", plot.reader + ':' + plot.reader_version);
            observe(path + ".reader_identity", plot.reader_identity);
        }
    }
    for (auto const& [path, values] : observed)
    {
        // Workload labels may aggregate different native readers. Never pick
        // one reader or imply that one declaration covers all of them.
        fact(path, values.second ? "mixed; inspect matching inventory.plots records" : values.first, "observed",
            "inventory-public-header-and-selected-reader", "Exact observed metadata/provenance only; no old proof validity, decoder cost or full-proof bound is inferred.");
    }
    Graph graph(profile);
    std::set<size_t> source_resources;
    for (auto const& root : inventory.roots)
    {
        auto const id = "storage:" + root.storage.filesystem;
        fact("inventory.storage[" + id + "]", root.path, "observed", "native-filesystem-and-drive-topology",
            "Local source filesystem path; drive aliases describe observed sharing, and unknown topology remains conservatively shared.");
        bool covered = graph.names.contains(id);
        if (covered)
        {
            auto const index = graph.find(id); auto const& resource = graph.resources[index];
            covered = resource.host == inventory.host && resource.kind == ResourceKind::storage;
            auto alias = [&](std::string const& name) { auto at = graph.names.find(name); return at != graph.names.end() && at->second == index; };
            if (root.storage.drives.empty()) covered = covered && alias("storage:unknown:" + inventory.host);
            else for (auto const& drive : root.storage.drives) covered = covered && alias("drive:" + drive);
            if (covered) source_resources.insert(index);
        }
        if (!covered) result.missing.push_back({"physical-bindings", "farm_resources.resources",
            "Bind " + id + " and its observed drive aliases to this local physical storage resource; a new label cannot supply additional capacity."});
    }
    std::set<size_t> declared_sources;
    for (auto const& id : profile.native.source_resources) declared_sources.insert(graph.find(id));
    if (declared_sources != source_resources) result.missing.push_back({"physical-bindings", "farm_resources.native.source_resources",
        "Map all inventoried source filesystems (including shared drive aliases) to the reviewed source supervision scope; inventory does not select output or scratch mappings."});
    bool remote = std::any_of(bindings.target_hosts.begin(), bindings.target_hosts.end(), [&](auto const& item) { return item.second != inventory.host; });
    for (auto const& [workload, hosts] : bindings.workload_hosts)
    { (void)workload; remote = remote || std::any_of(hosts.begin(), hosts.end(), [&](auto const& host) { return host != inventory.host; }); }
    if (remote && std::none_of(profile.resources.begin(), profile.resources.end(), [](auto const& resource) { return resource.kind == ResourceKind::network; }))
        result.missing.push_back({"physical-bindings", "farm_resources.resources", "Declare physical network resources and their bounds for the selected remote service paths; host bindings alone do not establish network service capacity."});
    validate_resource_preparation(result); return result;
}

inline FarmInventory scan_farm_inventory(std::span<std::filesystem::path const> paths, InventoryLimits const& limits = {})
{
    using namespace inventory_detail;
    validate_limits(limits); require(!paths.empty() && paths.size() <= limits.max_roots, "invalid farm inventory root count");
    auto const deadline = catalog_deadline(limits, limits.max_files);
    FarmInventory result; result.host = profile_detail::host_name(); result.captured_at = timestamp(); result.producer_build = GROW_IDENTITY;
    result.roots.reserve(paths.size());
    uint64_t retained = memory(result), enumerated = 0;
    std::set<std::pair<uint64_t, uint64_t>> roots;
    std::map<std::pair<uint64_t, uint64_t>, size_t> files;
    for (auto const& path : paths)
    {
        check(limits, deadline); RootHandle root(path); auto const info = stat(root.handle);
        require(roots.emplace(uint64_t(info.st_dev), uint64_t(info.st_ino)).second, "farm inventory contains duplicate root identities");
        InventoryRoot recorded{file_io::path_text(root.path), farming_directory_identity(root.path), storage_device(root.path), uint64_t(info.st_dev), uint64_t(info.st_ino)};
        (void)file_io::capture_durable_identity(root.handle.fd, &recorded.durable);
        if (file_io::has_durable_identity(recorded.durable)) result.version = 2;
        root.matches(recorded); size_t const index = result.roots.size();
        retained += 2 * (recorded.path.capacity() + recorded.directory_identity.capacity() + recorded.storage.filesystem.capacity() +
            recorded.durable.filesystem.capacity() + recorded.durable.object.capacity()) + 256;
        for (auto const& d : recorded.storage.drives) retained += 2 * d.capacity() + sizeof(std::string);
        require(retained + 65536 < limits.memory_bytes / 2, "farm inventory roots exceed its memory budget");
        result.roots.push_back(recorded);
        std::vector<std::string> pending{std::string{}};
        while (!pending.empty())
        {
            check(limits, deadline); std::string current = std::move(pending.back()); pending.pop_back();
            std::vector<std::string> names;
#ifndef _WIN32
            Handle directory(::dup(root.handle.fd));
            if (!current.empty()) for (auto const& part : relative(current))
                directory = Handle(::openat(directory.fd, part.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
            DIR* native = ::fdopendir(::dup(directory.fd)); require(native, "cannot enumerate farm inventory directory");
            struct CloseDir { DIR* value; ~CloseDir() { ::closedir(value); } } close{native};
            for (;;)
            {
                check(limits, deadline); errno = 0; auto const* entry = ::readdir(native);
                if (!entry) { require(!errno, "farm inventory directory enumeration failed"); break; }
                std::string_view name(entry->d_name); if (name == "." || name == "..") continue;
                require(enumerated < limits.max_files, "farm inventory exceeds its total entry bound"); ++enumerated;
                require(names.size() + result.plots.size() + pending.size() < limits.max_files &&
                    retained + (names.size() + pending.size() + 1) * 16384 + 65536 < limits.memory_bytes / 2,
                    "farm inventory exceeds its entry or memory budget");
                names.emplace_back(name);
            }
#else
            auto directory = root.path / file_io::native_path(current);
            for (auto const& entry : fs::directory_iterator(directory))
            {
                check(limits, deadline);
                require(enumerated < limits.max_files, "farm inventory exceeds its total entry bound"); ++enumerated;
                require(names.size() + result.plots.size() + pending.size() < limits.max_files &&
                    retained + (names.size() + pending.size() + 1) * 16384 + 65536 < limits.memory_bytes / 2,
                    "farm inventory exceeds its entry or memory budget");
                names.push_back(file_io::path_text(entry.path().filename()));
            }
#endif
            std::sort(names.begin(), names.end());
            for (auto const& name : names)
            {
                check(limits, deadline); std::string const relative_name = current.empty() ? name : current + '/' + name;
                bounded_text(relative_name); file_io::Stat info{};
#ifndef _WIN32
                bool const inspected = !::fstatat(directory.fd, name.c_str(), &info, AT_SYMLINK_NOFOLLOW);
#else
                bool const inspected = !file_io::inspect_link(root.path / file_io::native_path(relative_name), &info);
#endif
                if (inspected && file_io::is_directory(info.st_mode))
                {
                    require(uint64_t(info.st_dev) == recorded.device, "farm inventory must name mounted child filesystems as separate explicit roots");
                    pending.push_back(relative_name); continue;
                }
                require(result.plots.size() < limits.max_files &&
                    retained + (names.size() + pending.size() + 1) * 16384 + 65536 < limits.memory_bytes / 2, "farm inventory exceeds its memory budget");
                if (result.plots.size() == result.plots.capacity())
                {
                    auto const capacity = std::min<uint64_t>(limits.max_files, std::max<uint64_t>(32, result.plots.capacity() * 2));
                    require(capacity * sizeof(InventoryPlot) + retained + (names.size() + pending.size() + 1) * 16384 + 65536 < limits.memory_bytes / 2,
                        "farm inventory vector growth exceeds its memory budget");
                    retained += (capacity - result.plots.capacity()) * sizeof(InventoryPlot); result.plots.reserve(size_t(capacity));
                }
                InventoryPlot plot; plot.relative_path = relative_name; plot.root = index; plot.host = result.host;
                try { plot = observe(root, relative_name, limits); plot.root = index; plot.host = result.host; }
                catch (std::exception const&)
                {
                    plot.status = "protected"; plot.reason = "unreadable, changed, linked or unsupported native file; occupancy is not reclaimable";
                    if (inspected && file_io::is_regular(info.st_mode)) { plot.identity.stamp = stamp(info); plot.identity.links = info.st_nlink;
                        if (info.st_blocks >= 0 && uint64_t(info.st_blocks) <= UINT64_MAX / 512) plot.identity.allocated_bytes = uint64_t(info.st_blocks) * 512; }
                }
                if (!file_io::has_durable_identity(recorded.durable)) plot.identity.durable = {};
                if (plot.identity.stamp.inode)
                {
                    auto [at, fresh] = files.emplace(std::pair(plot.identity.stamp.device, plot.identity.stamp.inode), result.plots.size());
                    if (!fresh)
                    {
                        plot.alias_of = at->second; plot.status = "protected"; plot.reason = "duplicate native file identity";
                        auto& original = result.plots[at->second]; original.status = "protected"; original.reason = "duplicate native file identity"; retained += 512;
                    }
                }
                retained += plot_memory(plot) - sizeof(plot);
                result.plots.push_back(std::move(plot));
            }
        }
        root.matches(recorded);
    }
    result.complete = true; return result;
}
} // namespace group_grow::farm
