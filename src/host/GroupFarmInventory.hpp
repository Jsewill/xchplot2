// Public, bounded observations of explicitly selected farm roots. A catalog is
// never authority to unlink a path; retirement pins and rechecks its native ID.
#pragma once
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
    bool operator==(InventoryFileIdentity const&) const = default;
};
struct InventoryRoot
{
    std::string path, directory_identity;
    StorageDevice storage;
    uint64_t device = 0, inode = 0;
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
    // Observation only: execution repeats the supported native exact-file proof
    // check. Imported public JSON cannot grant a native deletion capability.
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
struct FarmInventory
{
    unsigned version = 1;
    std::string host, captured_at, producer_build;
    std::vector<InventoryRoot> roots;
    std::vector<InventoryPlot> plots;
    NetworkProjection network;
    std::optional<Profile> resource_template;
    bool complete = false;
};
namespace inventory_detail
{
namespace fs = std::filesystem;
inline void require(bool value, std::string const& message)
{ if (!value) throw std::runtime_error(message); }
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
inline void bounded_text(std::string const& text, size_t bound = 4096)
{ require(text.size() <= bound && text.find('\0') == text.npos, "farm inventory text exceeds its bound"); }
inline fs::path relative(std::string const& value)
{
    bounded_text(value); auto result = file_io::native_path(value);
    require(!result.empty() && !result.is_absolute() && !result.has_root_name(), "inventory path must be relative");
    for (auto const& component : result)
        require(component != "." && component != ".." && !component.empty(), "invalid inventory path component");
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
#ifdef _WIN32
inline Handle windows_open(fs::path const& path, bool directory)
{
    // Held ancestors deny delete/rename. Held files also deny writes, so an
    // adapter observes the same object throughout its bounded operation.
    HANDLE native = ::CreateFileW(path.c_str(), GENERIC_READ,
        directory ? FILE_SHARE_READ | FILE_SHARE_WRITE : FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | (directory ? FILE_FLAG_BACKUP_SEMANTICS : FILE_FLAG_OVERLAPPED), nullptr);
    require(native != INVALID_HANDLE_VALUE, "cannot pin farm inventory source");
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
    void matches(InventoryRoot const& root) const
    {
        auto const s = stat(handle); RootHandle current(path); auto const now = stat(current.handle);
        require(uint64_t(s.st_dev) == root.device && uint64_t(s.st_ino) == root.inode &&
            s.st_dev == now.st_dev && s.st_ino == now.st_ino &&
            file_io::path_text(path) == root.path && farming_directory_identity(path) == root.directory_identity,
            "farm inventory root identity changed");
    }
};
struct SourceHandle
{
    Handle handle;
    Handle parent;
    std::string basename;
#ifdef _WIN32
    std::vector<Handle> ancestors;
#endif
    SourceHandle(RootHandle const& root, std::string const& name)
    {
        auto const value = relative(name);
        basename = file_io::path_text(value.filename());
#ifdef _WIN32
        auto current = root.path;
        for (auto const& part : value.parent_path()) { current /= part; ancestors.push_back(windows_open(current, true)); }
        handle = windows_open(root.path / value, false);
#else
        parent = Handle(::fcntl(root.handle.fd, F_DUPFD_CLOEXEC, 3));
        for (auto const& part : value.parent_path())
            parent = Handle(::openat(parent.fd, part.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        handle = Handle(::openat(parent.fd, value.filename().c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW));
        require(file_io::is_regular(stat(handle).st_mode), "inventory source is not a regular file");
#endif
    }
};
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
            h.group_size && h.meta_group <= 8 && h.chunk_index_offset >= sizeof(h) + h.memo_length &&
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
inline InventoryPlot observe(RootHandle const& root, std::string const& relative_path, InventoryLimits const& limits)
{
    SourceHandle source(root, relative_path); auto const before = stat(source.handle);
    InventoryPlot result; result.relative_path = relative_path; result.identity.stamp = stamp(before);
    require(before.st_blocks >= 0 && uint64_t(before.st_blocks) <= UINT64_MAX / 512, "invalid allocated file size");
    result.identity.allocated_bytes = uint64_t(before.st_blocks) * 512; result.identity.links = uint64_t(before.st_nlink);
    uint64_t const bytes = std::min(result.identity.stamp.bytes, limits.max_header_bytes);
    std::vector<uint8_t> data(size_t(bytes), uint8_t{}); size_t at = 0;
    while (at < data.size())
    {
        auto const count = file_io::read_at(source.handle.fd, data.data() + at, data.size() - at, at);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "farm inventory header changed while reading"); at += size_t(count);
    }
    result.identity.header_bytes = bytes; result.identity.header_digest = group_tools::hex(grow_digest(data));
    require(stamp(stat(source.handle)) == result.identity.stamp, "farm inventory source changed while reading");
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
inline uint64_t plot_memory(InventoryPlot const& p)
{
    uint64_t bytes = sizeof(p) + 256; // Native-identity map node and allocator overhead.
    for (auto const* s : {&p.relative_path, &p.identity.header_digest, &p.compression_family, &p.plot_id, &p.farmer_public_key,
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
    for (auto const& r : value.roots) { charge(r.path); charge(r.directory_identity); charge(r.storage.filesystem); for (auto const& d : r.storage.drives) charge(d); }
    for (auto const* s : {&value.network.network, &value.network.genesis_challenge, &value.network.source, &value.network.version, &value.network.captured_at,
        &value.network.target_route, &value.network.reader_version, &value.network.reader_identity,
        &value.network.farmer_version, &value.network.solver_version, &value.network.service_status}) charge(*s);
    if (value.resource_template) bytes += profile_memory(*value.resource_template);
    for (auto const& p : value.plots) bytes += plot_memory(p) - sizeof(p);
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
class InventorySourcePin
{
    InventoryRoot root_;
    InventoryPlot plot_;
    inventory_detail::RootHandle root_handle_;
    inventory_detail::SourceHandle source_;
public:
    InventorySourcePin(InventoryRoot const& root, InventoryPlot const& plot, InventoryLimits const& limits = {}, std::string current_relative = {})
        : root_(root), plot_(plot), root_handle_(file_io::native_path(root.path)),
          source_(root_handle_, current_relative.empty() ? plot.relative_path : current_relative)
    {
        inventory_detail::validate_limits(limits); root_handle_.matches(root_);
        auto const held = inventory_detail::stamp(inventory_detail::stat(source_.handle));
        auto const current = inventory_detail::observe(root_handle_, current_relative.empty() ? plot.relative_path : current_relative, limits);
        inventory_detail::require((current_relative.empty() ? held == plot.identity.stamp : stream_state::same_output(plot.identity.stamp, held)) &&
            (current_relative.empty() ? current.identity == plot.identity : same_inventory_contents(plot.identity, current.identity)),
            "farm inventory source changed before retirement pin");
    }
    int root_fd() const { return root_handle_.handle.fd; }
    int source_fd() const { return source_.handle.fd; }
    int source_parent_fd() const { return source_.parent.fd; }
    std::string const& source_basename() const { return source_.basename; }
    void recheck_parent(std::string const& current_relative_path) const
    {
        root_handle_.matches(root_);
        inventory_detail::SourceHandle current(root_handle_, current_relative_path);
        auto const a = inventory_detail::stat(source_.parent), b = inventory_detail::stat(current.parent);
        inventory_detail::require(a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
            inventory_detail::stamp(inventory_detail::stat(current.handle)) == inventory_detail::stamp(inventory_detail::stat(source_.handle)),
            "farm retirement parent or source path changed");
    }
    bool matches_claim(std::string const& claimed_relative_path, InventoryLimits const& limits = {}) const
    {
        root_handle_.matches(root_);
        auto const held = inventory_detail::stamp(inventory_detail::stat(source_.handle));
        auto const claimed = inventory_detail::observe(root_handle_, claimed_relative_path, limits);
        return stream_state::same_output(plot_.identity.stamp, held) && same_inventory_contents(plot_.identity, claimed.identity);
    }
};
inline InventoryFileIdentity recheck_inventory_source(InventoryRoot const& root, InventoryPlot const& plot, InventoryLimits const& limits = {})
{
    auto current = observe_inventory_file(root, plot.relative_path, limits);
    inventory_detail::require(current.identity == plot.identity, "farm inventory source identity changed"); return current.identity;
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
    validate_limits(limits); require(v.version == 1 && !v.roots.empty() && v.roots.size() <= limits.max_roots &&
        v.plots.size() <= limits.max_files && memory(v) < limits.memory_bytes / 2, "farm inventory exceeds its schema or memory bound");
    for (auto const* text : {&v.host, &v.captured_at, &v.producer_build, &v.network.network, &v.network.genesis_challenge,
        &v.network.source, &v.network.version, &v.network.captured_at, &v.network.target_route, &v.network.reader_version, &v.network.reader_identity,
        &v.network.farmer_version, &v.network.solver_version, &v.network.service_status}) bounded_text(*text);
    require(!v.network.available || (!v.network.network.empty() && hex(v.network.genesis_challenge, 32)), "invalid inventory network projection");
    std::set<std::pair<uint64_t, uint64_t>> roots;
    for (auto const& r : v.roots)
    {
        bounded_text(r.path); bounded_text(r.directory_identity); bounded_text(r.storage.filesystem);
        require(file_io::native_path(r.path).is_absolute() && r.inode && roots.emplace(r.device, r.inode).second &&
            r.storage.drives.size() <= 128, "invalid or duplicate farm inventory root");
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
            id.stamp.mtime_ns < 1000000000 && id.stamp.ctime_ns < 1000000000, "invalid farm inventory file identity");
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
            !p.reader.empty() && !p.reader_version.empty() && hex(p.reader_identity, 32) && hex(p.checker_identity, 32) && !id.header_digest.empty()), "incomplete native inventory observation");
    }
}
inline void write_identity(std::ostream& out, InventoryFileIdentity const& id)
{
    auto const& s = id.stamp;
    out << "{\"device\":" << s.device << ",\"inode\":" << s.inode << ",\"bytes\":" << s.bytes << ",\"mtime\":" << s.mtime
        << ",\"ctime\":" << s.ctime << ",\"mtime_ns\":" << s.mtime_ns << ",\"ctime_ns\":" << s.ctime_ns
        << ",\"allocated_bytes\":" << id.allocated_bytes << ",\"links\":" << id.links << ",\"header_bytes\":" << id.header_bytes << ",\"header_digest\":";
    profile_detail::quote(out, id.header_digest); out << '}';
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
inline InventoryFileIdentity read_identity(Json const& j)
{
    InventoryFileIdentity result; auto& s = result.stamp;
    s.device = integer(j, "device"); s.inode = integer(j, "inode"); s.bytes = integer(j, "bytes"); s.mtime = signed_integer(j, "mtime"); s.ctime = signed_integer(j, "ctime");
    auto m = integer(j, "mtime_ns"), c = integer(j, "ctime_ns"); require(m < 1000000000 && c < 1000000000, "invalid inventory time precision");
    s.mtime_ns = uint32_t(m); s.ctime_ns = uint32_t(c); result.allocated_bytes = integer(j, "allocated_bytes");
    result.links = integer(j, "links"); result.header_bytes = integer(j, "header_bytes"); result.header_digest = string(j, "header_digest"); return result;
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
    out << "{\"schema_version\":1,\"kind\":\"farm-inventory\""; text("host", v.host); text("captured_at", v.captured_at); text("producer_build", v.producer_build);
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
        bool drive_comma = false; for (auto const& d : r.storage.drives) { if (drive_comma) out << ','; drive_comma = true; quote(d); } out << "]}";
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
    if (v.resource_template) { out << ",\"farm_resources\":"; write_profile_json(out, *v.resource_template); }
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
    require(integer(j, "schema_version") == 1 && string(j, "kind") == "farm-inventory", "unsupported farm inventory schema");
    FarmInventory result; result.host = string(j, "host"); result.captured_at = string(j, "captured_at"); result.producer_build = string(j, "producer_build");
    result.complete = boolean(j, "complete"); auto const& n = required_field(j, "network"); result.network.available = boolean(n, "available");
    result.network.network = string(n, "network"); result.network.genesis_challenge = string(n, "genesis_challenge"); result.network.source = string(n, "source");
    result.network.version = string(n, "version"); result.network.captured_at = string(n, "captured_at"); result.network.hard_fork2_height = integer(n, "hard_fork2_height");
    result.network.target_route = string(n, "target_route"); result.network.reader_version = string(n, "reader_version"); result.network.reader_identity = string(n, "reader_identity");
    for (auto const& [name, target] : {std::pair{"farmer_version", &result.network.farmer_version},
        std::pair{"solver_version", &result.network.solver_version}, std::pair{"service_status", &result.network.service_status}})
        if (auto const* value = profile_detail::field(n, name)) *target = profile_detail::text(*value);
    result.network.min_strength = integer(n, "min_strength"); result.network.max_strength = integer(n, "max_strength"); result.network.plot_size = integer(n, "plot_size");
    auto const& roots = required_field(j, "roots"); auto const& plots = required_field(j, "plots");
    require(roots.kind == Json::array && !roots.items.empty() && roots.items.size() <= limits.max_roots &&
        plots.kind == Json::array && plots.items.size() <= limits.max_files &&
        plots.items.size() <= limits.memory_bytes / (4 * sizeof(InventoryPlot)), "farm inventory count exceeds its bound");
    result.roots.reserve(roots.items.size()); result.plots.reserve(plots.items.size());
    for (auto const& r : roots.items)
    {
        InventoryRoot root; root.path = string(r, "path"); root.directory_identity = string(r, "directory_identity"); root.device = integer(r, "device"); root.inode = integer(r, "inode");
        root.storage.filesystem = string(r, "filesystem"); root.storage.nonrotating = boolean(r, "nonrotating"); root.storage.rotating = boolean(r, "rotating");
        auto const& drives = required_field(r, "drives"); require(drives.kind == Json::array && drives.items.size() <= 128, "invalid inventory physical drives");
        for (auto const& d : drives.items) root.storage.drives.push_back(profile_detail::text(d)); result.roots.push_back(std::move(root));
    }
    for (auto const& p : plots.items)
    {
        InventoryPlot plot; plot.root = integer(p, "root"); plot.relative_path = string(p, "relative_path"); plot.identity = read_identity(required_field(p, "identity"));
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
    validate(result, limits); return result;
}
inline FarmInventory parse_farm_inventory(std::string const& text, InventoryLimits const& limits = {})
{ std::istringstream in(text); return read_farm_inventory(in, limits); }

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
        root.matches(recorded); size_t const index = result.roots.size();
        retained += 2 * (recorded.path.capacity() + recorded.directory_identity.capacity() + recorded.storage.filesystem.capacity()) + 256;
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
