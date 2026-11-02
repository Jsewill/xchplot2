// One job-wide capacity ledger for destination growth and external scratch.
// Free space is snapshotted before mutation, as for destination admission.
// Protocol discovery, exact allocation and durable cleanup remain engine-owned.
#pragma once
#include "host/GroupExternal.hpp"
#include <tuple>
#include <utility>

namespace group_grow
{
struct ScratchRoot
{
    std::string directory;
    StorageDevice device;
    uint64_t available_bytes = 0, alignment = 0, file_limit = 0;
    uint64_t directory_device = 0, directory_inode = 0;

    static ScratchRoot inspect(std::filesystem::path const& path)
    {
        auto const directory = std::filesystem::canonical(file_io::native_path(path));
        file_io::Stat value{}; file_io::Space space{};
        long const page = file_io::page_size();
        require(!file_io::inspect_path(directory.c_str(), &value) && file_io::is_directory(value.st_mode) && value.st_blksize > 0 &&
            !file_io::space_path(directory.c_str(), &space) && space.f_frsize && space.f_bsize && page > 0,
            "cannot inspect scratch/destination directory");
        require(uint64_t(space.f_bavail) <= UINT64_MAX / uint64_t(space.f_frsize), "scratch capacity overflow");
        return {file_io::path_text(directory), storage_device(directory), uint64_t(space.f_bavail) * uint64_t(space.f_frsize),
            std::max({uint64_t(space.f_frsize), uint64_t(space.f_bsize), uint64_t(value.st_blksize), uint64_t(page)}),
            maximum_file_bytes(file_io::path_text(directory)), uint64_t(value.st_dev), uint64_t(value.st_ino)};
    }
    void check() const
    {
        file_io::Stat value{};
        require(!file_io::inspect_path(directory.c_str(), &value) && file_io::is_directory(value.st_mode) &&
            uint64_t(value.st_dev) == directory_device && uint64_t(value.st_ino) == directory_inode,
            "scratch/destination directory changed during the run");
    }
};

class ScratchPool
{
public:
    // Snapshot input also permits resource-policy tests with synthetic device
    // identities; ordinary callers use inspect() on the supplied directories.
    ScratchPool(std::vector<ScratchRoot> roots, uint64_t headroom, uint64_t capacity = 0)
        : headroom_(headroom), capacity_(capacity)
    {
        require(!roots.empty() && roots.size() <= 1024, "scratch root count must be in 1..1024");
        roots_.reserve(roots.size()); filesystems_.reserve(roots.size());
        for (auto& root : roots) add_root(std::move(root), true);
    }
    ScratchPool(std::span<std::filesystem::path const> directories, uint64_t headroom, uint64_t capacity = 0)
        : headroom_(headroom), capacity_(capacity)
    {
        require(!directories.empty() && directories.size() <= 1024, "scratch root count must be in 1..1024");
        roots_.reserve(directories.size()); filesystems_.reserve(directories.size());
        for (auto const& directory : directories)
        {
            require_disk_scratch(directory);
            add_root(ScratchRoot::inspect(file_io::native_path(directory)), true);
        }
    }
    ScratchPool(ScratchPool const&) = delete;
    ScratchPool& operator=(ScratchPool const&) = delete;

    // Register every saved output and its sidecar peak before seal_recovery().
    // Only physical blocks already owned at registration earn a free-space
    // credit. Newly allocated blocks remain covered by the original bound.
    bool try_reserve_output(uint64_t owner, ScratchRoot directory, uint64_t peak,
                            uint64_t allocated = 0, bool saved = false)
    {
        std::lock_guard lock(mutex_);
        size_t const root = add_root(std::move(directory), false);
        auto& entry = find_or_add(owner); auto& filesystem = fs_for(root);
        require(!entry.io, "cannot resize a destination during I/O");
        require(!saved || !sealed_ || entry.output.has_value(), "restore saved output before sealing resource admission");
        saved = saved || entry.restored;
        if (!filesystem.headroom_available) return false;
        if (allocated > peak) return false;
        if (entry.output)
        {
            require(entry.output->root == root, "destination reservation directory changed");
            if (peak < entry.output->allocated || (entry.output->saved && peak < entry.output->peak)) return false;
            uint64_t const old = entry.output->peak - entry.output->allocated;
            uint64_t const next = peak - entry.output->allocated;
            if (next > old && next - old > free_bytes(filesystem)) return false;
            filesystem.reserved = filesystem.reserved - old + next;
            entry.output->peak = peak; entry.output->saved = entry.output->saved || saved;
        }
        else
        {
            uint64_t const additional = peak - allocated;
            if (additional > free_bytes(filesystem)) return false;
            filesystem.reserved = grow_add(filesystem.reserved, additional);
            entry.output = Output{root, peak, allocated, saved};
        }
        return true;
    }
    bool try_reserve_output(uint64_t owner, std::filesystem::path const& directory, uint64_t peak,
                            uint64_t allocated = 0, bool saved = false)
    { return try_reserve_output(owner, ScratchRoot::inspect(file_io::native_path(directory)), peak, allocated, saved); }

    uint64_t available_for_output(uint64_t owner, ScratchRoot directory)
    {
        std::lock_guard lock(mutex_);
        size_t const root = add_root(std::move(directory), false);
        auto& filesystem = fs_for(root); uint64_t room = free_bytes(filesystem);
        if (auto* entry = find(owner); entry && entry->output)
        {
            require(entry->output->root == root, "destination reservation directory changed");
            room = grow_add(room, entry->output->peak);
        }
        return room;
    }
    uint64_t available_for_output(uint64_t owner, std::filesystem::path const& directory)
    { return available_for_output(owner, ScratchRoot::inspect(file_io::native_path(directory))); }

    void restore(uint64_t owner, ExternalConfig const& external, std::span<uint64_t const> allocated)
    {
        std::lock_guard lock(mutex_);
        require(!sealed_ && external.scratch.size() == allocated.size() && !external.scratch.empty(),
            "restore saved scratch before sealing resource admission");
        auto& entry = find_or_add(owner);
        require(entry.external.scratch.empty() && !entry.io, "scratch ownership was already reserved");
        std::vector<size_t> roots; roots.reserve(external.scratch.size());
        for (size_t i = 0; i < external.scratch.size(); ++i)
        {
            auto const& grant = external.scratch[i]; size_t const root = supplied_root(grant.directory);
            require(std::find(roots.begin(), roots.end(), root) == roots.end(), "saved scratch aliases the same directory");
            require(grant.capacity_bytes && allocated[i] <= grant.capacity_bytes,
                "saved physical scratch allocation exceeds its grant");
            roots.push_back(root);
        }
        entry.external = external; entry.roots = std::move(roots);
        entry.allocated.assign(allocated.begin(), allocated.end()); entry.restored = true;
        if (entry.output) entry.output->saved = true;
        for (size_t i = 0; i < entry.roots.size(); ++i)
        {
            auto& filesystem = fs_for(entry.roots[i]); uint64_t const quota = external.scratch[i].capacity_bytes;
            filesystem.reserved = grow_add(filesystem.reserved, quota - allocated[i]);
            filesystem.scratch = grow_add(filesystem.scratch, quota);
        }
    }
    void seal_recovery()
    {
        std::lock_guard lock(mutex_);
        require(!sealed_, "scratch recovery admission was already sealed");
        for (auto const& entry : owners_) require(entry.external.scratch.empty() || entry.output.has_value(),
            "saved scratch requires a destination peak reservation");
        for (auto const& filesystem : filesystems_)
            require(filesystem.reserved <= filesystem.budget && (!filesystem.scratch || filesystem.headroom_available) &&
                (!capacity_ || filesystem.scratch <= capacity_),
                "saved scratch and destination reservations exceed available space or --temp-capacity-bytes");
        for (auto const& root : roots_) root.value.check();
        sealed_ = true;
    }
    // Recovery can register roots that the current invocation has not enabled
    // for new jobs. Keep their capacity and I/O obligations in the same ledger.
    void restrict_grants(std::span<std::filesystem::path const> directories)
    {
        std::lock_guard lock(mutex_);
        require(!sealed_, "select fresh scratch roots before sealing resource admission");
        std::vector<size_t> allowed; allowed.reserve(directories.size());
        for (auto const& directory : directories) allowed.push_back(supplied_root(file_io::path_text(directory)));
        for (size_t i = 0; i < roots_.size(); ++i)
            roots_[i].grantable = std::find(allowed.begin(), allowed.end(), i) != allowed.end();
    }
    // A terminal retired owner may preserve changed files. They need no future
    // allocation and earn no free-space credit, but still consume the job's
    // per-filesystem scratch cap until the user removes them.
    void preserve(ScratchRoot directory, uint64_t occupied)
    {
        std::lock_guard lock(mutex_);
        require(!sealed_, "inspect retired scratch before sealing resource admission");
        directory.check(); size_t const root = add_root(std::move(directory), false);
        fs_for(root).scratch = grow_add(fs_for(root).scratch, occupied);
    }

    std::optional<ExternalConfig> grant(uint64_t owner, uint64_t desired, uint64_t minimum,
                                        uint64_t segment_bytes, unsigned fan_in, bool automatic = false,
                                        unsigned packing_inputs = 0)
    {
        std::lock_guard lock(mutex_);
        require(sealed_ && minimum && desired >= minimum && segment_bytes > 64 &&
            segment_bytes <= uint64_t(INT64_MAX) && fan_in >= 2 && fan_in <= 64 &&
            (!packing_inputs || (packing_inputs >= fan_in && packing_inputs <= 64)), "invalid fresh scratch admission");
        auto* entry = find(owner);
        require(entry && entry->output && entry->external.scratch.empty() && !entry->io,
            "fresh scratch requires an idle destination reservation");
        ExternalConfig external; external.segment_bytes = segment_bytes; external.fan_in = fan_in;
        std::vector<size_t> selected; std::vector<size_t> seen;
        std::vector<size_t> order, ssds;
        order.reserve(roots_.size()); ssds.reserve(roots_.size());
        for (size_t i = 0; i < roots_.size(); ++i)
        {
            order.push_back(i);
            if (roots_[i].value.device.nonrotating && !roots_[i].value.device.drives.empty()) ssds.push_back(i);
        }
        uint64_t remaining = desired;
        auto const& destination = roots_[entry->output->root].value.device;
        // Quotas are a conservative backlog proxy, not measured bandwidth.
        // Rank only known SSD slots; unknown and rotating storage retain the
        // established supplied-order and physical-exclusion policy.
        auto pressure = [&](size_t root)
        {
            uint64_t reserved = 0; size_t active = 0;
            for (auto const& other : owners_) if (other.id != owner) for (size_t n = 0; n < other.roots.size(); ++n)
                if (overlap(roots_[root].value.device, roots_[other.roots[n]].value.device))
                { reserved = grow_add(reserved, other.external.scratch[n].capacity_bytes); if (other.io) ++active; }
            return std::tuple{overlap(destination, roots_[root].value.device), active, reserved};
        };
        std::sort(ssds.begin(), ssds.end(), [&](size_t left, size_t right)
        {
            auto const a = pressure(left), b = pressure(right);
            return a == b ? left < right : a < b;
        });
        size_t next_ssd = 0;
        for (auto& index : order) if (roots_[index].value.device.nonrotating && !roots_[index].value.device.drives.empty())
            index = ssds[next_ssd++];
        // Prefer unused rotating drives before sharing a partial grant.
        for (unsigned speed : {0u, 1u}) for (bool sharing : {false, true})
            for (size_t const i : order)
        {
            if (!remaining) break;
            auto const& root = roots_[i];
            if (!root.grantable || std::find(seen.begin(), seen.end(), root.filesystem) != seen.end()) continue;
            if (automatic ? unsigned(!root.value.device.nonrotating) != speed : speed != 0) continue;
            if (automatic && destination.nonrotating && !root.value.device.drives.empty() &&
                !root.value.device.nonrotating) continue;
            if (shared_rotating(i, owner) != sharing) continue;
            seen.push_back(root.filesystem); root.value.check();
            if (segment_bytes > root.value.file_limit) continue;
            auto const& filesystem = filesystems_[root.filesystem];
            uint64_t room = free_bytes(filesystem);
            if (capacity_) room = std::min(room, capacity_ - std::min(capacity_, filesystem.scratch));
            uint64_t const quota = std::min(remaining, room) / root.value.alignment * root.value.alignment;
            if (!quota) continue;
            selected.push_back(i); external.scratch.push_back({root.value.directory, quota}); remaining -= quota;
        }
        if (desired - remaining < minimum) return {};
        // Retaining more readers avoids copying packets that the final packer
        // can already join. Only known SSD scratch gets the larger reader set;
        // the merge arity and the aggregate reader buffer remain unchanged.
        bool const nonrotating = std::all_of(selected.begin(), selected.end(), [&](size_t i)
            { auto const& device = roots_[i].value.device; return device.nonrotating && !device.drives.empty(); });
        external.packing_inputs = nonrotating && packing_inputs ? packing_inputs : fan_in;
        for (size_t i = 0; i < selected.size(); ++i)
        {
            auto& filesystem = fs_for(selected[i]); uint64_t const quota = external.scratch[i].capacity_bytes;
            filesystem.reserved = grow_add(filesystem.reserved, quota); filesystem.scratch = grow_add(filesystem.scratch, quota);
        }
        entry->external = external; entry->roots = std::move(selected);
        entry->allocated.assign(entry->roots.size(), 0); return external;
    }

    class IoLease
    {
    public:
        IoLease(IoLease&& other) noexcept : pool_(std::exchange(other.pool_, nullptr)), owner_(other.owner_) {}
        IoLease& operator=(IoLease&& other) noexcept
        { if (this != &other) { reset(); pool_ = std::exchange(other.pool_, nullptr); owner_ = other.owner_; } return *this; }
        ~IoLease() { reset(); }
        IoLease(IoLease const&) = delete;
        IoLease& operator=(IoLease const&) = delete;
        void reset() noexcept { if (pool_) { pool_->release_io(owner_); pool_ = nullptr; } }
    private:
        friend class ScratchPool;
        IoLease(ScratchPool* pool, uint64_t owner) : pool_(pool), owner_(owner) {}
        ScratchPool* pool_;
        uint64_t owner_;
    };
    std::optional<IoLease> try_io(uint64_t owner)
    { return acquire_io(owner, false); }
    // External contribution staging performs bulk I/O only on its scratch.
    // The caller must serialize its target metadata with every owner of the
    // same physical target resources. This lease never permits final packing
    // or ordinary output writes; those retain the full try_io() exclusion.
    std::optional<IoLease> try_stage_io(uint64_t owner)
    { return acquire_io(owner, true); }
    // Compare two normal/full leases from the admitted resource snapshots,
    // independent of their current lease modes. This grants no I/O; try_io()
    // still checks live owners and directory identities before acquisition.
    bool io_compatible(uint64_t left, uint64_t right) const
    {
        std::lock_guard lock(mutex_);
        if (!sealed_ || left == right) return false;
        auto const* a = find(left); auto const* b = find(right);
        return a && b && a->output && b->output && !conflicts(*a, *b, false, false);
    }

    // Call only after the engine has durably removed all owned scratch files.
    // Keeping the output lease permits a handoff to ordinary same-file growth.
    void release_scratch_after_cleanup(uint64_t owner)
    {
        std::lock_guard lock(mutex_); auto* entry = find(owner);
        require(sealed_ && entry && !entry->io, "cannot release scratch during I/O or recovery admission"); release_scratch(*entry);
    }
    void release_after_cleanup(uint64_t owner, uint64_t retained_output_bytes)
    {
        std::lock_guard lock(mutex_); auto* entry = find(owner);
        require(sealed_ && entry && entry->output && !entry->io && retained_output_bytes <= entry->output->peak,
            "invalid completed destination release");
        release_scratch(*entry); auto const output = *entry->output; auto& filesystem = fs_for(output.root);
        filesystem.reserved -= output.peak - output.allocated;
        if (retained_output_bytes >= output.allocated) filesystem.budget -= retained_output_bytes - output.allocated;
        else filesystem.budget = grow_add(filesystem.budget, output.allocated - retained_output_bytes);
        owners_.erase(owners_.begin() + (entry - owners_.data()));
    }
    // A fresh rejected candidate has not written files and needs no cleanup.
    void cancel(uint64_t owner)
    {
        std::lock_guard lock(mutex_); auto* entry = find(owner);
        require(entry && !entry->io && !entry->restored && (!entry->output || !entry->output->saved),
            "cannot cancel saved scratch/destination ownership");
        release_scratch(*entry);
        if (entry->output) fs_for(entry->output->root).reserved -= entry->output->peak - entry->output->allocated;
        owners_.erase(owners_.begin() + (entry - owners_.data()));
    }
    uint64_t memory_bytes() const
    {
        std::lock_guard lock(mutex_);
        // Reallocation can keep a previous backing array alive while moving
        // its entries. Owned strings/vectors transfer their storage separately.
        uint64_t bytes = sizeof(*this) + 2 * (roots_.capacity() * sizeof(Root) +
            filesystems_.capacity() * sizeof(Filesystem) + owners_.capacity() * sizeof(Owner));
        for (auto const& root : roots_)
        {
            bytes = grow_add(bytes, root.value.directory.capacity() + 1 + root.value.device.filesystem.capacity() + 1 +
                root.value.device.drives.capacity() * sizeof(std::string));
            for (auto const& drive : root.value.device.drives) bytes = grow_add(bytes, drive.capacity() + 1);
            // One in-progress grant can own selected/seen tables and both
            // candidate/result grant arrays before their storage is moved.
            bytes = grow_add(bytes, 6 * sizeof(size_t) + 2 * sizeof(ScratchGrant) + 2 * (root.value.directory.capacity() + 1));
        }
        for (auto const& filesystem : filesystems_) bytes = grow_add(bytes, filesystem.name.capacity() + 1);
        for (auto const& entry : owners_)
        {
            bytes = grow_add(bytes, entry.external.scratch.capacity() * sizeof(ScratchGrant) +
                entry.roots.capacity() * sizeof(size_t) + entry.allocated.capacity() * sizeof(uint64_t));
            for (auto const& grant : entry.external.scratch) bytes = grow_add(bytes, grant.directory.capacity() + 1);
        }
        return bytes;
    }

private:
    struct Root { ScratchRoot value; size_t filesystem; bool supplied, grantable; };
    struct Filesystem { std::string name; uint64_t budget = 0, reserved = 0, scratch = 0; bool headroom_available = true; };
    struct Output { size_t root; uint64_t peak, allocated; bool saved; };
    struct Owner
    {
        uint64_t id;
        std::optional<Output> output;
        ExternalConfig external;
        std::vector<size_t> roots;
        std::vector<uint64_t> allocated;
        bool restored = false, io = false, stage_io = false;
    };
    uint64_t headroom_, capacity_;
    bool sealed_ = false;
    std::vector<Root> roots_;
    std::vector<Filesystem> filesystems_;
    std::vector<Owner> owners_;
    mutable std::mutex mutex_;
    static uint64_t free_bytes(Filesystem const& filesystem)
    { return filesystem.headroom_available && filesystem.budget > filesystem.reserved ? filesystem.budget - filesystem.reserved : 0; }
    Filesystem& fs_for(size_t root) { return filesystems_[roots_[root].filesystem]; }
    Owner* find(uint64_t id)
    { for (auto& owner : owners_) if (owner.id == id) return &owner; return nullptr; }
    Owner const* find(uint64_t id) const
    { for (auto const& owner : owners_) if (owner.id == id) return &owner; return nullptr; }
    Owner& find_or_add(uint64_t id)
    {
        if (auto* found = find(id)) return *found;
        require(owners_.size() < 1024, "too many active scratch owners"); owners_.push_back(Owner{id}); return owners_.back();
    }
    size_t add_root(ScratchRoot root, bool supplied)
    {
        require(!root.directory.empty() && !root.device.filesystem.empty() && root.alignment, "invalid scratch directory snapshot");
        for (size_t i = 0; i < roots_.size(); ++i) if (roots_[i].value.directory == root.directory ||
            (roots_[i].value.directory_device == root.directory_device && roots_[i].value.directory_inode == root.directory_inode))
        {
            require(roots_[i].value.directory_device == root.directory_device && roots_[i].value.directory_inode == root.directory_inode &&
                roots_[i].value.device.filesystem == root.device.filesystem && roots_[i].value.device.drives == root.device.drives &&
                roots_[i].value.device.nonrotating == root.device.nonrotating &&
                roots_[i].value.device.rotating == root.device.rotating, "scratch/destination directory changed during discovery");
            if (!sealed_)
            {
                fs_for(i).budget = std::min(fs_for(i).budget, root.available_bytes > headroom_ ? root.available_bytes - headroom_ : 0);
                fs_for(i).headroom_available = fs_for(i).headroom_available && root.available_bytes >= headroom_;
            }
            roots_[i].supplied = roots_[i].supplied || supplied;
            roots_[i].grantable = roots_[i].grantable || supplied; return i;
        }
        size_t filesystem = 0;
        while (filesystem < filesystems_.size() && filesystems_[filesystem].name != root.device.filesystem) ++filesystem;
        uint64_t const budget = root.available_bytes > headroom_ ? root.available_bytes - headroom_ : 0;
        if (filesystem == filesystems_.size()) filesystems_.push_back({root.device.filesystem, budget, 0, 0, root.available_bytes >= headroom_});
        else if (!sealed_)
        {
            filesystems_[filesystem].budget = std::min(filesystems_[filesystem].budget, budget);
            filesystems_[filesystem].headroom_available = filesystems_[filesystem].headroom_available && root.available_bytes >= headroom_;
        }
        roots_.push_back({std::move(root), filesystem, supplied, supplied}); return roots_.size() - 1;
    }
    size_t supplied_root(std::string const& directory) const
    {
        auto const supplied = ScratchRoot::inspect(file_io::native_path(directory));
        for (size_t i = 0; i < roots_.size(); ++i) if (roots_[i].supplied &&
            roots_[i].value.directory_device == supplied.directory_device && roots_[i].value.directory_inode == supplied.directory_inode)
        { roots_[i].value.check(); return i; }
        throw std::runtime_error("saved scratch directory was not supplied");
    }
    static bool overlap(StorageDevice const& left, StorageDevice const& right)
    {
        if (left.drives.empty() || right.drives.empty()) return true;
        for (auto const& drive : left.drives)
            if (std::find(right.drives.begin(), right.drives.end(), drive) != right.drives.end()) return true;
        return false;
    }
    bool shared_rotating(size_t root, uint64_t owner) const
    {
        auto const& left = roots_[root].value.device;
        for (auto const& other : owners_) if (other.id != owner) for (auto index : other.roots)
        {
            auto const& right = roots_[index].value.device;
            if (overlap(left, right) && (left.drives.empty() || right.drives.empty() || !left.nonrotating || !right.nonrotating)) return true;
        }
        return false;
    }
    bool stage_independent(Owner const& entry) const
    {
        if (entry.roots.empty()) return false;
        auto const& target = roots_[entry.output->root].value.device;
        for (auto root : entry.roots)
            if (overlap(target, roots_[root].value.device)) return false;
        return true;
    }
    bool conflicts(Owner const& left, Owner const& right, bool left_stage_io, bool right_stage_io) const
    {
        auto const& a = roots_[left.output->root].value.device; auto const& b = roots_[right.output->root].value.device;
        if (overlap(a, b) && (!(left_stage_io || right_stage_io) || a.drives.empty() || b.drives.empty())) return true;
        for (auto root : left.roots) if (overlap(roots_[root].value.device, b)) return true;
        for (auto root : right.roots) if (overlap(a, roots_[root].value.device)) return true;
        for (auto l : left.roots) for (auto r : right.roots)
        {
            auto const& x = roots_[l].value.device; auto const& y = roots_[r].value.device;
            if (overlap(x, y) && (x.drives.empty() || y.drives.empty() || !x.nonrotating || !y.nonrotating)) return true;
        }
        return false;
    }
    std::optional<IoLease> acquire_io(uint64_t owner, bool stage_io)
    {
        std::lock_guard lock(mutex_);
        require(sealed_, "restore all saved scratch before admitting I/O");
        auto* entry = find(owner); require(entry && entry->output && !entry->io, "invalid scratch I/O owner");
        roots_[entry->output->root].value.check();
        for (auto root : entry->roots) roots_[root].value.check();
        if (stage_io && !stage_independent(*entry)) return {};
        for (auto const& other : owners_) if (other.io && conflicts(*entry, other, stage_io, other.stage_io)) return {};
        entry->io = true; entry->stage_io = stage_io; return IoLease(this, owner);
    }
    void release_io(uint64_t owner) noexcept
    { std::lock_guard lock(mutex_); if (auto* entry = find(owner)) { entry->io = false; entry->stage_io = false; } }
    void release_scratch(Owner& entry)
    {
        for (size_t i = 0; i < entry.roots.size(); ++i)
        {
            auto& filesystem = fs_for(entry.roots[i]); uint64_t const quota = entry.external.scratch[i].capacity_bytes;
            filesystem.reserved -= quota - entry.allocated[i]; filesystem.scratch -= quota;
            filesystem.budget = grow_add(filesystem.budget, entry.allocated[i]);
        }
        entry.external.scratch.clear(); entry.roots.clear(); entry.allocated.clear(); entry.restored = false;
    }
};
} // namespace group_grow
