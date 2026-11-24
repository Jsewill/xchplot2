// One job-wide capacity ledger for destination growth and external scratch.
// Free space is snapshotted before mutation, as for destination admission.
// Protocol discovery, exact allocation and durable cleanup remain engine-owned.
#pragma once
#include "host/GroupExternal.hpp"
#include <functional>
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

    void set_io_governor(std::shared_ptr<DiskIoGovernor> governor)
    {
        std::lock_guard lock(mutex_);
        require(!sealed_, "bind disk feedback before resource admission");
        io_governor_ = std::move(governor);
        if (io_governor_) for (auto const& root : roots_)
            io_governor_->register_device(root.value.device, root.value.directory_device);
        bandwidth_ = io_governor_;
        if (!bandwidth_ && std::count_if(roots_.begin(), roots_.end(), [](Root const& root)
            { return root.value.device.rotating && !root.value.device.drives.empty(); }) > 1)
        {
            bandwidth_ = std::make_shared<DiskIoGovernor>(DiskIoGovernor::sample_device, 0);
            for (auto const& root : roots_) bandwidth_->register_device(root.value.device);
        }
    }
    std::shared_ptr<DiskIoGovernor> bandwidth_feedback() const { return bandwidth_; }
    uint64_t storage_snapshot_bytes() const
    {
        std::lock_guard lock(mutex_);
        uint64_t bytes = 0;
        for (auto const& root : roots_) bytes = grow_add(bytes, group_grow::storage_snapshot_bytes(root.value.device));
        return bytes;
    }
    void bind_runtime(ExternalConfig& external) const
    { external.io_governor = io_governor_; external.bandwidth = bandwidth_; }
    bool offers_scratch(std::string const& filesystem) const
    {
        std::lock_guard lock(mutex_);
        return std::any_of(roots_.begin(), roots_.end(), [&](Root const& root)
            { return root.grantable && root.value.device.filesystem == filesystem; });
    }
    std::vector<StorageDevice> scratch_devices(uint64_t owner) const
    {
        std::lock_guard lock(mutex_);
        std::vector<StorageDevice> devices;
        if (auto const* entry = find(owner))
            for (auto root : entry->roots) devices.push_back(roots_[root].value.device);
        return devices;
    }
    // A candidate may be priced against the existing owners before it has
    // a reservation of its own. Resolve the same discovery snapshots.
    std::vector<StorageDevice> scratch_devices(ExternalConfig const& config) const
    {
        std::lock_guard lock(mutex_);
        std::vector<StorageDevice> devices; devices.reserve(config.scratch.size());
        for (auto const& grant : config.scratch)
        {
            auto at = std::find_if(roots_.begin(), roots_.end(),
                [&](auto const& root) { return root.value.directory == grant.directory; });
            require(at != roots_.end(), "unknown candidate scratch directory");
            devices.push_back(at->value.device);
        }
        return devices;
    }

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

    uint64_t available_for_output(uint64_t owner, ScratchRoot directory, bool after_fresh_scratch = false)
    {
        std::lock_guard lock(mutex_);
        size_t const root = add_root(std::move(directory), false);
        auto& filesystem = fs_for(root); uint64_t room = free_bytes(filesystem);
        if (auto* entry = find(owner); entry && entry->output)
        {
            require(entry->output->root == root, "destination reservation directory changed");
            room = grow_add(room, entry->output->peak);
        }
        // A queued farm output may wait for temporary grants made by this
        // run. Restored grants can belong to preserved or offline jobs.
        if (after_fresh_scratch) for (auto const& entry : owners_) if (!entry.restored)
            for (size_t i = 0; i < entry.roots.size(); ++i)
                if (roots_[entry.roots[i]].filesystem == roots_[root].filesystem)
                    room = grow_add(room, entry.external.scratch[i].capacity_bytes);
        return room;
    }
    uint64_t available_for_output(uint64_t owner, std::filesystem::path const& directory, bool after_fresh_scratch = false)
    { return available_for_output(owner, ScratchRoot::inspect(file_io::native_path(directory)), after_fresh_scratch); }

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
        entry.external = external; bind_runtime(entry.external); entry.roots = std::move(roots);
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
    // Retain a farming journal's admitted replacement allowance without
    // creating an I/O owner or lending it to temporary plot construction.
    // Already allocated bytes are absent from the free-space snapshot.
    void reserve_metadata(ScratchRoot directory, uint64_t peak, uint64_t allocated)
    {
        std::lock_guard lock(mutex_);
        require(!sealed_ && allocated <= peak, "invalid farming metadata reservation");
        directory.check(); size_t const root = add_root(std::move(directory), false);
        auto& filesystem = fs_for(root);
        filesystem.reserved = grow_add(filesystem.reserved, peak - allocated);
    }
    void reserve_metadata(std::filesystem::path const& directory, uint64_t peak, uint64_t allocated)
    { reserve_metadata(ScratchRoot::inspect(file_io::native_path(directory)), peak, allocated); }

    std::optional<ExternalConfig> grant(uint64_t owner, uint64_t desired, uint64_t minimum,
                                        uint64_t segment_bytes, unsigned fan_in, bool automatic = false,
                                        unsigned packing_inputs = 0, uint64_t read_buffer_bytes = 0,
                                        size_t parallel_roots = 1, size_t open_slots = SIZE_MAX,
                                        std::function<uint64_t(ExternalConfig&)> const& configure_stripes = {},
                                        bool preview = false, bool spare_spindle = false, long double successor_rate = 0)
    {
        std::lock_guard lock(mutex_);
        require(sealed_ && minimum && desired >= minimum && segment_bytes > 64 &&
            segment_bytes <= uint64_t(INT64_MAX) && fan_in >= 2 && fan_in <= 64 &&
            (!packing_inputs || (packing_inputs >= fan_in && packing_inputs <= 64)), "invalid fresh scratch admission");
        auto* entry = find(owner);
        require(entry && entry->output && entry->external.scratch.empty() && !entry->io,
            "fresh scratch requires an idle destination reservation");
        ExternalConfig external; external.segment_bytes = segment_bytes; external.fan_in = fan_in;
        bind_runtime(external);
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
        // Rank known SSDs by their outstanding capacity and active writers.
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
        if (io_governor_ && io_governor_->limits_io())
        {
            // Keep SSD positions and their established backlog ranking. HDD
            // capacity still comes from this ledger; live feedback orders only
            // otherwise eligible physical roots, including periodic probes.
            std::vector<size_t> rotating, positions;
            std::vector<std::pair<bool, long double>> costs(roots_.size());
            for (size_t i = 0; i < order.size(); ++i)
            {
                auto const index = order[i]; auto const& device = roots_[index].value.device;
                if (device.nonrotating) continue;
                bool const pressured = io_governor_->scoped() && io_governor_->pressured(&device);
                costs[index] = {pressured, !io_governor_->scoped() || pressured ? io_governor_->routing_cost(device) : 0.L};
                positions.push_back(i); rotating.push_back(index);
            }
            std::stable_sort(rotating.begin(), rotating.end(), [&](size_t a, size_t b) { return costs[a] < costs[b]; });
            for (size_t i = 0; i < positions.size(); ++i) order[positions[i]] = rotating[i];
        }
        if (spare_spindle)
        {
            // An optional receiver can wait a long time for its first byte.
            // Keep queued rotational writers available even on an unstriped grant.
            std::erase_if(order, [&](size_t root)
            {
                auto const& disk = roots_[root].value.device;
                bool const own = roots_[root].filesystem == roots_[entry->output->root].filesystem ||
                    (!disk.drives.empty() && !destination.drives.empty() && overlap(destination, disk));
                return !disk.nonrotating && (shared_rotating(root, owner) ||
                    (!own && std::any_of(owners_.begin(), owners_.end(), [&](auto const& other)
                    { return other.id != owner && other.output && overlap(disk, roots_[other.output->root].value.device); })));
            });
        }
        if (parallel_roots > 1)
        {
            uint64_t const extent = std::min(segment_bytes, read_buffer_bytes ? std::max(uint64_t(1), read_buffer_bytes / 8) : segment_bytes);
            size_t const width = spare_spindle ? 1 : std::min<uint64_t>(parallel_roots, std::max<uint64_t>(1, desired / extent));
            for (auto [i, quota] : rotating_shares(owner, destination, order, desired, segment_bytes, automatic, width, open_slots, successor_rate))
            {
                auto const& root = roots_[i]; root.value.check();
                selected.push_back(i); seen.push_back(root.filesystem);
                external.scratch.push_back({root.value.directory, quota}); remaining -= quota;
            }
            external.striped = selected.size() > 1;
        }
        // Prefer unused rotating drives before sharing a partial grant.
        for (unsigned speed : {0u, 1u}) for (bool sharing : {false, true})
            for (size_t const i : order)
        {
            if (!remaining) break;
            auto const& root = roots_[i];
            if (spare_spindle && !root.value.device.nonrotating && !selected.empty()) continue;
            if (!root.grantable || std::find(seen.begin(), seen.end(), root.filesystem) != seen.end()) continue;
            if (automatic ? unsigned(!root.value.device.nonrotating) != speed : speed != 0) continue;
            if (automatic && destination.nonrotating && !root.value.device.drives.empty() &&
                !root.value.device.nonrotating) continue;
            if (shared_rotating(i, owner) != sharing) continue;
            seen.push_back(root.filesystem); root.value.check();
            if (segment_bytes > root.value.file_limit) continue;
            uint64_t const quota = std::min(remaining, grant_room(i)) / root.value.alignment * root.value.alignment;
            if (!quota) continue;
            selected.push_back(i); external.scratch.push_back({root.value.directory, quota}); remaining -= quota;
        }
        if (desired - remaining < minimum) return {};
        // The selected roots determine the extent layout and the RAM left for
        // readers. Use that admission before deciding which runs to retain.
        if (external.striped && configure_stripes) read_buffer_bytes = configure_stripes(external);
        // Retaining more readers avoids copying packets that the final packer
        // can already join. Known HDDs can also retain more when the requested
        // aggregate cache leaves large refills; unknown storage keeps the
        // original gate. Compaction arity and saved admission stay independent.
        bool const nonrotating = std::all_of(selected.begin(), selected.end(), [&](size_t i)
            { auto const& device = roots_[i].value.device; return device.nonrotating && !device.drives.empty(); });
        bool const known = std::all_of(selected.begin(), selected.end(), [&](size_t i)
            { return !roots_[i].value.device.drives.empty(); });
        uint64_t const readers = nonrotating ? packing_inputs : known ?
            std::min<uint64_t>(packing_inputs, read_buffer_bytes / (16ull * 1024 * 1024)) : 0;
        external.packing_inputs = unsigned(std::max<uint64_t>(fan_in, readers));
        if (preview) return external;
        for (size_t i = 0; i < selected.size(); ++i)
        {
            auto& filesystem = fs_for(selected[i]); uint64_t const quota = external.scratch[i].capacity_bytes;
            filesystem.reserved = grow_add(filesystem.reserved, quota); filesystem.scratch = grow_add(filesystem.scratch, quota);
            if (io_governor_) io_governor_->selected(roots_[selected[i]].value.device);
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
    // A lane journal writes only its target. A producer can acquire its
    // admitted scratch again between network frames, so check its whole grant.
    bool target_io_compatible(StorageDevice const& target, uint64_t owner) const
    {
        std::lock_guard lock(mutex_);
        auto const* entry = find(owner);
        if (!sealed_ || !entry || !entry->output || overlap(target, roots_[entry->output->root].value.device)) return false;
        return std::none_of(entry->roots.begin(), entry->roots.end(), [&](size_t root)
        { return overlap(target, roots_[root].value.device); });
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
        if (io_governor_) bytes = grow_add(bytes, io_governor_->memory_bytes());
        if (bandwidth_ && bandwidth_ != io_governor_) bytes = grow_add(bytes, bandwidth_->memory_bytes());
        for (auto const& root : roots_)
        {
            bytes = grow_add(bytes, root.value.directory.capacity() + 1 + root.value.device.filesystem.capacity() + 1 +
                root.value.device.drives.capacity() * sizeof(std::string));
            for (auto const& drive : root.value.device.drives) bytes = grow_add(bytes, drive.capacity() + 1);
            // One in-progress grant can own selected/seen tables and both
            // candidate/result grant arrays before their storage is moved.
            bytes = grow_add(bytes, 8 * sizeof(size_t) + 2 * sizeof(long double) + 2 * sizeof(ScratchGrant) + 2 * (root.value.directory.capacity() + 1));
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
    std::shared_ptr<DiskIoGovernor> io_governor_;
    std::shared_ptr<DiskIoGovernor> bandwidth_;
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
        if (io_governor_) io_governor_->register_device(root.device, root.directory_device);
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
    uint64_t grant_room(size_t i) const
    {
        auto const& fs = filesystems_[roots_[i].filesystem]; auto const alignment = roots_[i].value.alignment;
        uint64_t room = free_bytes(fs);
        if (capacity_) room = std::min(room, capacity_ - std::min(capacity_, fs.scratch));
        return room / alignment * alignment;
    }
    // Reserve the spindles needed by simultaneous output writers and balance
    // the remaining drives by throughput before splitting a capacity grant.
    std::vector<std::pair<size_t, uint64_t>> rotating_shares(uint64_t owner, StorageDevice const& destination,
        std::span<size_t const> order, uint64_t desired, uint64_t segment_bytes, bool automatic, size_t width, size_t open_slots, long double successor_rate) const
    {
        struct Candidate { size_t root; uint64_t room; long double rate; bool lookup_pressure; };
        std::vector<Candidate> disks;
        size_t waiting = 1;
        std::vector<size_t> outputs;
        for (auto const& other : owners_) if (other.id != owner && other.output && (other.io || !other.external.scratch.empty()))
            outputs.push_back(other.output->root);
        // A queued file behind the same spindle will use that writer later.
        // Reserve the remaining slots for destinations that can run alongside it.
        for (auto const& other : owners_)
        {
            if (waiting >= open_slots) break;
            if (other.id == owner || !other.output || other.io || !other.external.scratch.empty()) continue;
            auto const& target = roots_[other.output->root].value.device;
            if (shared_rotating(other.output->root, owner) || overlap(destination, target) || std::any_of(outputs.begin(), outputs.end(),
                [&](size_t output) { return overlap(roots_[output].value.device, target); })) continue;
            outputs.push_back(other.output->root); ++waiting;
        }
        for (size_t i : order)
        {
            auto const& root = roots_[i];
            if (!root.grantable || segment_bytes > root.value.file_limit || !grant_room(i)) continue;
            if (!root.value.device.rotating || root.value.device.drives.empty())
            {
                if (root.value.device.nonrotating && (disks.empty() || automatic)) return {};
                continue;
            }
            if ((automatic && destination.nonrotating) || shared_rotating(i, owner) ||
                std::any_of(outputs.begin(), outputs.end(), [&](size_t output)
                { return overlap(root.value.device, roots_[output].value.device); }) ||
                std::any_of(disks.begin(), disks.end(), [&](Candidate const& other)
                { return roots_[other.root].filesystem == root.filesystem || overlap(roots_[other.root].value.device, root.value.device); })) continue;
            disks.push_back({i, grant_room(i), bandwidth_ ? bandwidth_->bandwidth(root.value.device) : 0,
                io_governor_ && io_governor_->scoped() && io_governor_->pressured(&root.value.device)});
        }
        // The output already has sequential writes of its own. Use it for
        // scratch only when the dedicated disks cannot supply the grant.
        uint64_t dedicated = 0;
        for (auto const& disk : disks) if (!overlap(destination, roots_[disk.root].value.device))
            dedicated += std::min(desired - dedicated, disk.room);
        if (dedicated >= desired) std::erase_if(disks, [&](Candidate const& disk)
            { return overlap(destination, roots_[disk.root].value.device); });
        uint64_t timely_room = 0;
        for (auto const& disk : disks) if (!disk.lookup_pressure) timely_room += std::min(desired - timely_room, disk.room);
        if (timely_room == desired) std::erase_if(disks, [](auto const& disk) { return disk.lookup_pressure; });
        // Give unmeasured roots an equal first trial. A slow/busy root
        // with a measured rate gets correspondingly less work.
        long double total_rate = 0; size_t measured = 0;
        for (auto const& disk : disks) if (disk.rate > 0) { total_rate += disk.rate; ++measured; }
        long double const typical = measured ? total_rate / measured : 1;
        for (auto& disk : disks) if (!disk.rate) disk.rate = typical;
        std::stable_sort(disks.begin(), disks.end(), [](auto const& a, auto const& b) { return a.rate > b.rate; });
        // A successor can open while an older, narrow stripe is finishing.
        // Include occupied spindles when pricing its sustainable share. Taking
        // every free spindle here would leave the following writer just the
        // narrow stripe when that owner releases it.
        long double fleet_rate = 0;
        for (auto const& disk : disks) fleet_rate += disk.rate;
        size_t sharing = waiting;
        bool backlog = false;
        for (auto const& other : owners_) if (other.id != owner && other.output)
        {
            if (other.roots.empty()) { if (!other.io) backlog = true; continue; }
            if (!std::all_of(other.roots.begin(), other.roots.end(), [&](size_t i)
                { return roots_[i].value.device.rotating && !roots_[i].value.device.drives.empty(); })) continue;
            ++sharing;
            for (auto i : other.roots)
            {
                auto const rate = bandwidth_ ? bandwidth_->bandwidth(roots_[i].value.device) : 0;
                fleet_rate += rate > 0 ? rate : typical;
            }
        }
        // Balance the available bandwidth of simultaneous owners. One
        // fast spindle can serve the same work as several slower ones.
        std::vector<long double> loads(std::min(waiting, disks.size()));
        std::erase_if(disks, [&](Candidate const& disk)
        {
            auto least = std::min_element(loads.begin(), loads.end());
            *least += disk.rate; return least != loads.begin();
        });
        size_t selected = std::min(disks.size(), width);
        if (backlog && sharing > waiting && selected > 1)
        {
            auto const fair = fleet_rate / sharing;
            long double kept = 0;
            uint64_t room = 0;
            for (size_t i = 0; i < selected; ++i)
            {
                kept += disks[i].rate;
                room += std::min(desired - room, disks[i].room);
                if (room == desired && (i + 1 == selected || kept + disks[i + 1].rate / 2 >= fair))
                { selected = i + 1; break; }
            }
        }
        // When this fills the last open slot, a peer can finish before this
        // stripe is released. Leave a queued writer a target if the rest of
        // the stripe still has the requested space and can feed the output.
        if (open_slots == 1 && !outputs.empty())
        {
            auto const output_rate = bandwidth_ ? bandwidth_->bandwidth(destination) : 0;
            auto const required_rate = output_rate > 0 ? output_rate : typical;
            StorageDevice const* successor = nullptr;
            long double least = 0;
            struct PendingWriter { StorageDevice const* target = nullptr; uint64_t room = 0; long double kept = 0, lost = 0; };
            PendingWriter next;
            bool several = false;
            for (auto const& other : owners_)
            {
                if (other.id == owner || !other.output || other.io || !other.external.scratch.empty()) continue;
                auto const& target = roots_[other.output->root].value.device;
                if (overlap(destination, target) || shared_rotating(other.output->root, owner) ||
                    std::any_of(outputs.begin(), outputs.end(), [&](size_t output)
                    { return overlap(target, roots_[output].value.device); })) continue;
                uint64_t room = 0; long double kept = 0, lost = 0;
                for (auto const& disk : std::span(disks).first(selected))
                    if (overlap(target, roots_[disk.root].value.device)) lost += disk.rate;
                    else { kept += disk.rate; room += std::min(desired - room, disk.room); }
                if (!lost) { successor = nullptr; next = {}; break; }
                if (room == desired && kept >= required_rate && (!successor || lost < least))
                { successor = &target; least = lost; }
                if (!next.target) next = {&target, room, kept, lost};
                else if (!overlap(*next.target, target)) several = true;
            }
            auto replacement = disks.end();
            if (!successor && next.target && !several)
            {
                // The fair stripe can take the only runnable successor's
                // spindle. Try the ranked slack before lending that spindle.
                auto const sufficient_rate = std::min(required_rate, next.kept + next.lost);
                replacement = std::find_if(disks.begin() + selected, disks.end(), [&](Candidate const& disk)
                {
                    return disk.room >= desired - next.room && next.kept + disk.rate >= sufficient_rate &&
                        std::none_of(owners_.begin(), owners_.end(), [&](Owner const& other)
                        { return other.output && overlap(roots_[disk.root].value.device, roots_[other.output->root].value.device); });
                });
                if (replacement != disks.end()) successor = next.target;
            }
            if (successor)
            {
                auto end = std::remove_if(disks.begin(), disks.begin() + selected, [&](Candidate const& disk)
                { return overlap(*successor, roots_[disk.root].value.device); });
                if (replacement != disks.end()) *end++ = *replacement;
                selected = end - disks.begin();
            }
        }
        // A qualified same-output successor can receive on different HDDs while this
        // stripe packs. Split only this owner's share, with enough measured service
        // and complete capacity on both sides. Unknown rates keep the ordinary grant.
        if (successor_rate > 0 && selected > 1 && bandwidth_)
        {
            auto stripe = std::span(disks).first(selected);
            long double total = 0;
            for (auto const& disk : stripe)
            {
                if (overlap(destination, roots_[disk.root].value.device) ||
                    !bandwidth_->bandwidth(roots_[disk.root].value.device)) { total = 0; break; }
                total += disk.rate;
            }
            long double rate = 0;
            uint64_t room = 0;
            for (size_t i = 0; total && i + 1 < selected; ++i)
            {
                rate += disks[i].rate; room += std::min(desired - room, disks[i].room);
                if (room < desired || rate < std::max(successor_rate, total / 2) || total - rate < successor_rate) continue;
                uint64_t spare = 0;
                for (auto const& disk : stripe.subspan(i + 1)) spare += std::min(desired - spare, disk.room);
                if (spare == desired) selected = i + 1;
                break;
            }
        }
        disks.resize(selected);
        std::stable_sort(disks.begin(), disks.end(), [](auto const& a, auto const& b) { return a.room / a.rate < b.room / b.rate; });
        total_rate = 0; for (auto const& disk : disks) total_rate += disk.rate;
        std::vector<std::pair<size_t, uint64_t>> shares;
        uint64_t remaining = desired;
        for (size_t i = 0; i < disks.size(); ++i)
        {
            auto const& disk = disks[i];
            auto const& root = roots_[disk.root];
            uint64_t const share = i + 1 == disks.size() ? remaining : uint64_t(remaining * (disk.rate / total_rate));
            uint64_t const quota = std::min(disk.room, share) / root.value.alignment * root.value.alignment;
            total_rate -= disk.rate;
            if (!quota) continue;
            shares.emplace_back(disk.root, quota); remaining -= quota;
        }
        return shares;
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
