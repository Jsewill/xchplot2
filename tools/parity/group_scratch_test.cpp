#include "host/GroupScratch.hpp"
#include <iostream>
#include <random>

using namespace group_grow;
namespace fs = std::filesystem;
constexpr uint64_t unit = 4096;

template<class Operation> void rejected(Operation operation, std::string const& message)
{
    bool failed = false;
    try { operation(); } catch (std::exception const&) { failed = true; }
    require(failed, message);
}
ScratchRoot root(fs::path const& path, std::string filesystem, std::vector<std::string> drives,
                 bool nonrotating = false, uint64_t available = 100 * unit)
{
    auto value = ScratchRoot::inspect(path);
    value.device = {std::move(filesystem), std::move(drives), nonrotating};
    value.available_bytes = available; value.alignment = unit; value.file_limit = 1024 * unit;
    return value;
}
uint64_t quota(ExternalConfig const& config)
{
    uint64_t bytes = 0; for (auto const& grant : config.scratch) bytes = grow_add(bytes, grant.capacity_bytes); return bytes;
}
void shared_capacity(fs::path const& directory)
{
    auto a = root(directory / "a", "shared", {"one"});
    auto b = root(directory / "b", "shared", {"one"});
    auto alias = a; alias.directory = (directory / "alias-a").string();
    ScratchPool pool({a, b, alias}, 10 * unit, 60 * unit);
    require(pool.try_reserve_output(0, a, 20 * unit) && pool.try_reserve_output(1, b, unit), "shared output reservation failed");
    rejected([&] { (void)pool.grant(0, unit, unit, unit, 2); }, "fresh grant preceded saved ownership admission");
    pool.seal_recovery();
    auto first = pool.grant(0, 80 * unit, unit, unit, 2);
    require(first && first->scratch.size() == 1 && quota(*first) == 60 * unit,
        "aliases or sibling directories multiplied filesystem space or the scratch cap");
    auto const memory = pool.memory_bytes();
    require(!pool.grant(1, 20 * unit, 10 * unit, unit, 2) && pool.memory_bytes() == memory,
        "failed grant retained a partial capacity lease");
    require(pool.available_for_output(0, a) == 29 * unit && !pool.try_reserve_output(0, a, 30 * unit),
        "scratch and final-output obligations used independent free-space ledgers");
    require(pool.try_reserve_output(0, a, 29 * unit), "joint output/scratch admission omitted available bytes");
    pool.release_scratch_after_cleanup(0);
    auto second = pool.grant(1, 60 * unit, 60 * unit, unit, 2);
    require(second && quota(*second) == 60 * unit, "durably cleaned fresh scratch was not available to the next owner");
    pool.cancel(1); pool.cancel(0);

    ScratchPool preserved({a}, 10 * unit, 60 * unit);
    preserved.preserve(a, 20 * unit);
    require(preserved.try_reserve_output(0, a, 20 * unit), "preserved-file output reservation failed");
    preserved.seal_recovery();
    auto capped = preserved.grant(0, 60 * unit, unit, unit, 2);
    require(capped && quota(*capped) == 40 * unit && preserved.available_for_output(0, a) == 50 * unit,
        "retired occupancy was ignored by the scratch cap or credited as reclaimable space");
    preserved.cancel(0);
}
void output_adjustment(fs::path const& directory)
{
    auto a = root(directory / "a", "shared", {"one"});
    ScratchPool pool({a}, 0);
    require(pool.try_reserve_output(0, a, 80 * unit), "fresh destination reservation failed"); pool.seal_recovery();
    require(!pool.grant(0, 40 * unit, 40 * unit, unit, 2), "shared scratch overcommitted a destination peak");
    require(pool.try_reserve_output(0, a, 60 * unit), "fresh output could not make room for useful scratch");
    auto grant = pool.grant(0, 40 * unit, 40 * unit, unit, 2);
    require(grant && quota(*grant) == 40 * unit, "joint fresh output and scratch capacity was not admitted");
    auto sibling = root(directory / "b", "shared", {"one"}, false, 20 * unit);
    (void)pool.available_for_output(1, sibling); pool.release_scratch_after_cleanup(0);
    require(pool.available_for_output(1, sibling) == 40 * unit,
        "later directory discovery charged the job's own growth a second time");
}
void saved_capacity(fs::path const& directory)
{
    auto scratch = root(directory / "a", "scratch", {"scratch-drive"}, false, 10 * unit);
    auto target = root(directory / "c", "target", {"target-drive"});
    ScratchPool pool({scratch}, 0, 100 * unit);
    require(pool.try_reserve_output(0, target, 20 * unit, 5 * unit, true), "saved output physical credit was rejected");
    ExternalConfig saved{{{scratch.directory, 80 * unit}}, unit, 2};
    std::array<uint64_t, 1> occupied{75 * unit}; pool.restore(0, saved, occupied);
    require(pool.try_reserve_output(1, target, unit), "second output reservation failed"); pool.seal_recovery();
    require(!pool.try_reserve_output(0, target, 19 * unit, 5 * unit), "saved output reservation was silently reduced");
    require(!pool.grant(1, 6 * unit, 6 * unit, unit, 2), "saved scratch's remaining physical obligation was not retained");
    require(pool.available_for_output(2, scratch, true) == pool.available_for_output(2, scratch),
        "a future output counted preserved scratch as reclaimable capacity");
    rejected([&] { pool.cancel(0); }, "saved ownership could be cancelled before durable cleanup");
    pool.release_scratch_after_cleanup(0);
    auto next = pool.grant(1, 80 * unit, 80 * unit, unit, 2);
    require(next && quota(*next) == 80 * unit, "cleaned saved physical allocation was not returned to the ledger");
    pool.release_after_cleanup(0, 10 * unit);
    require(pool.available_for_output(1, target) == 95 * unit, "completed output net growth was not retained");
    pool.cancel(1);

    ScratchPool cap({scratch}, 0, 64 * unit);
    require(cap.try_reserve_output(0, target, 20 * unit, 5 * unit, true), "saved cap fixture output failed");
    cap.restore(0, saved, occupied);
    rejected([&] { cap.seal_recovery(); }, "saved grants exceeded the aggregate user scratch cap");

    ScratchPool no_credit({scratch}, 0);
    require(no_credit.try_reserve_output(0, target, unit), "no-credit fixture output failed");
    std::array<uint64_t, 1> sparse{unit}; no_credit.restore(0, saved, sparse);
    rejected([&] { no_credit.seal_recovery(); }, "logical scratch occupancy was credited as physical allocation");
}
void saved_only_roots(fs::path const& directory)
{
    auto saved_root = root(directory / "a", "saved-ssd", {"ssd"}, true);
    auto enabled = root(directory / "b", "enabled-hdd", {"hdd"});
    auto target = root(directory / "c", "target", {"target"});
    ScratchPool pool({saved_root, enabled}, 0);
    require(pool.try_reserve_output(0, target, unit, 0, true) &&
        pool.try_reserve_output(1, target, unit) && pool.try_reserve_output(2, target, unit),
        "saved-only root output reservation failed");
    ExternalConfig saved{{{saved_root.directory, 20 * unit}}, unit, 2};
    std::array<uint64_t, 1> occupied{unit}; pool.restore(0, saved, occupied);
    std::array<fs::path, 1> allowed{enabled.directory}; pool.restrict_grants(allowed); pool.seal_recovery();
    require(pool.offers_scratch(enabled.device.filesystem) && !pool.offers_scratch(saved_root.device.filesystem) &&
        !pool.offers_scratch(target.device.filesystem), "queued outputs lent space outside the fresh scratch policy");
    require(pool.try_io(0).has_value(), "restricting fresh grants blocked saved scratch recovery");
    auto next = pool.grant(1, 20 * unit, unit, unit, 2, true);
    require(next && next->scratch.size() == 1 && next->scratch[0].directory == enabled.directory,
        "new job selected a faster saved-only root outside the enabled policy");
    pool.release_scratch_after_cleanup(0);
    require(!pool.grant(2, 90 * unit, 90 * unit, unit, 2, true),
        "released saved-only root became eligible when enabled storage could not fit");
    rejected([&] { pool.restrict_grants(allowed); }, "fresh root policy changed after sealing admission");
    pool.cancel(2); pool.cancel(1); pool.release_after_cleanup(0, 0);

    ScratchPool aliases({saved_root}, 0);
    require(aliases.try_reserve_output(0, target, unit), "allowed alias output reservation failed");
    std::array<fs::path, 1> alias{directory / "alias-a"}; aliases.restrict_grants(alias); aliases.seal_recovery();
    require(aliases.grant(0, unit, unit, unit, 2).has_value(), "enabled scratch alias lost its original directory grant");

    ScratchPool frozen({saved_root, enabled}, 0);
    require(frozen.try_reserve_output(0, target, unit, 0, true) && frozen.try_reserve_output(1, target, unit),
        "independent saved-scratch output reservation failed");
    frozen.restore(0, saved, occupied); frozen.restrict_grants({}); frozen.seal_recovery();
    require(frozen.try_io(0).has_value() && !frozen.grant(1, unit, unit, unit, 2),
        "independent scratch restriction blocked exact recovery or spent an unavailable writer's possible quota");
    frozen.release_scratch_after_cleanup(0);
    require(!frozen.grant(1, unit, unit, unit, 2), "local scratch cleanup reopened unaccounted shared storage grants");

    ScratchPool journals({saved_root}, 0);
    journals.reserve_metadata(saved_root, 20 * unit, 5 * unit);
    require(journals.try_reserve_output(0, saved_root, 20 * unit), "metadata-protected output reservation failed");
    journals.seal_recovery();
    auto bounded = journals.grant(0, 100 * unit, unit, unit, 2);
    require(bounded && quota(*bounded) == 65 * unit,
        "scratch borrowed the durable journal's unused replacement allowance or credited allocated metadata twice");
    rejected([&] { journals.reserve_metadata(saved_root, unit, 0); }, "journal promise changed after scratch admission");
}
void saved_aliases(fs::path const& directory)
{
    auto a = root(directory / "a", "shared", {"one"});
    auto b = root(directory / "b", "shared", {"one"});
    auto target = root(directory / "c", "target", {"target"});
    ScratchPool pool({a, b}, 0, 64 * unit);
    require(pool.try_reserve_output(0, target, unit), "saved alias output failed");
    ExternalConfig shared{{{a.directory, 40 * unit}, {b.directory, 40 * unit}}, unit, 2};
    std::array<uint64_t, 2> occupied{30 * unit, 30 * unit}; pool.restore(0, shared, occupied);
    rejected([&] { pool.seal_recovery(); }, "saved sibling scratch roots bypassed the filesystem cap");

    ScratchPool alias({a}, 0);
    ExternalConfig duplicate{{{a.directory, 20 * unit}, {(directory / "alias-a").string(), 20 * unit}}, unit, 2};
    std::array<uint64_t, 2> duplicate_allocated{10 * unit, 10 * unit};
    rejected([&] { alias.restore(0, duplicate, duplicate_allocated); }, "aliased saved scratch earned duplicate physical credit");
    ExternalConfig missing{{{b.directory, 20 * unit}}, unit, 2};
    std::array<uint64_t, 1> one{unit};
    rejected([&] { alias.restore(1, missing, one); }, "an unsupplied saved scratch directory was admitted");
}
void topology(fs::path const& directory, bool nonrotating, bool unknown)
{
    auto scratch = root(directory / "a", "scratch", unknown ? std::vector<std::string>{} : std::vector<std::string>{"ssd"}, nonrotating);
    auto first = root(directory / "c", "target-a", {"hdd-a"});
    auto second = root(directory / "d", "target-b", {"hdd-b"});
    auto shared_destination = root(directory / "e", "target-partition", {"hdd-a"});
    auto scratch_destination = root(directory / "f", "scratch-partition", {"ssd"}, true);
    ScratchPool pool({scratch}, 0);
    require(pool.try_reserve_output(0, first, unit) && pool.try_reserve_output(1, second, unit) &&
        pool.try_reserve_output(2, shared_destination, unit) && pool.try_reserve_output(3, scratch_destination, unit),
        "topology output reservations failed");
    require(!pool.try_reserve_output(4, first, UINT64_MAX), "inactive output fixture unexpectedly reserved capacity");
    ScratchPool const& view = pool;
    require(!view.io_compatible(0, 1) && !view.target_io_compatible(second.device, 0),
        "I/O compatibility preceded saved ownership admission");
    pool.seal_recovery();
    require(pool.grant(0, 30 * unit, 30 * unit, unit, 2).has_value() &&
        pool.grant(1, 30 * unit, 30 * unit, unit, 2).has_value(), "shared scratch quotas failed");
    auto const memory = pool.memory_bytes(), capacity = pool.available_for_output(0, first);
    bool const compatible = nonrotating && !unknown;
    require(view.target_io_compatible(second.device, 0) == !unknown &&
        !view.target_io_compatible(shared_destination.device, 0) && !view.target_io_compatible(scratch_destination.device, 0) &&
        !view.target_io_compatible(second.device, 4) && !view.target_io_compatible(second.device, 5),
        "a lane journal must exclude aliased, unbound or unknown drives and admit an independent target");
    require(view.io_compatible(0, 1) == compatible && view.io_compatible(1, 0) == compatible,
        "full I/O compatibility disagreed with shared scratch topology");
    require(!view.io_compatible(0, 2) && !view.io_compatible(2, 0) &&
        !view.io_compatible(0, 3) && !view.io_compatible(3, 0),
        "full I/O compatibility ignored shared output or scratch-versus-output resources");
    require(!view.io_compatible(0, 0) && !view.io_compatible(0, 4) && !view.io_compatible(4, 0) &&
        !view.io_compatible(0, 5) && !view.io_compatible(5, 0),
        "I/O compatibility admitted a duplicate, absent or unreserved output owner");
    require(pool.memory_bytes() == memory && pool.available_for_output(0, first) == capacity,
        "I/O compatibility changed resource ownership or capacity");
    auto first_io = pool.try_io(0); require(first_io.has_value(), "first I/O lease failed");
    require(view.io_compatible(0, 1) == compatible, "a live full lease changed snapshot compatibility");
    auto second_io = pool.try_io(1);
    require(second_io.has_value() == (nonrotating && !unknown), "scratch topology sharing policy was not enforced");
    require(!pool.try_io(2), "different partitions of a destination drive ran concurrently");
    require(!pool.try_io(3), "scratch-vs-destination overlap was allowed on shared SSD storage");
    rejected([&] { pool.release_scratch_after_cleanup(0); }, "active I/O ownership was released");
    first_io.reset();
    if (!second_io) second_io = pool.try_io(1);
    require(second_io.has_value(), "serialized scratch did not become available after completion");
    second_io.reset();
    require(pool.try_io(2).has_value(), "ordinary destination remained blocked after I/O completion");
    pool.cancel(0); pool.cancel(1); pool.cancel(2); pool.cancel(3); pool.cancel(4);
}
void staged_topology(fs::path const& directory, bool nonrotating, bool unknown)
{
    auto scratch = root(directory / "a", "scratch", unknown ? std::vector<std::string>{} :
        std::vector<std::string>{"scratch"}, nonrotating);
    auto target = root(directory / "c", "target", {"target"});
    auto sibling = root(directory / "d", "target-partition", {"target"});
    ScratchPool pool({scratch}, 0);
    require(pool.try_reserve_output(0, target, unit) && pool.try_reserve_output(1, sibling, unit) &&
        pool.try_reserve_output(2, sibling, unit), "staging output reservations failed");
    rejected([&] { (void)pool.try_stage_io(0); }, "staging preceded saved ownership admission");
    pool.seal_recovery();
    require(pool.grant(0, 20 * unit, 20 * unit, unit, 2).has_value() &&
        pool.grant(1, 20 * unit, 20 * unit, unit, 2).has_value(), "staging scratch grants failed");
    require(!pool.try_stage_io(2), "an owner without external scratch obtained a staging lease");
    auto const before = pool.available_for_output(0, target);
    auto packed = pool.try_io(0); require(packed.has_value(), "packing lease failed");
    require(!pool.try_io(1), "ordinary writers sharing an output drive ran concurrently");
    auto staged = pool.try_stage_io(1);
    require(staged.has_value() == (nonrotating && !unknown),
        "staging and packing ignored shared scratch topology");
    require(!pool.io_compatible(0, 1) && !pool.io_compatible(1, 0),
        "a live stage lease weakened the query's full-output exclusion");
    packed.reset();
    if (staged)
    {
        packed = pool.try_io(0);
        require(packed.has_value(), "packing behind an independent staging lease was asymmetric");
        packed.reset();
        std::optional<ScratchPool::IoLease> moved(std::move(staged)); staged.reset();
        rejected([&] { pool.cancel(1); }, "a moved staging lease lost its owner exclusion");
        rejected([&] { pool.release_scratch_after_cleanup(1); }, "active staging scratch was released");
        rejected([&] { (void)pool.try_reserve_output(1, sibling, 2 * unit); }, "active staging output was resized");
        rejected([&] { (void)pool.try_io(1); }, "staging owner obtained overlapping lease modes");
        moved.reset();
    }
    auto first = pool.try_stage_io(0);
    require(first.has_value() == !unknown, "staging admitted unknown or rejected independent scratch");
    if (first)
    {
        auto second = pool.try_stage_io(1);
        require(second.has_value() == nonrotating, "staging owners shared rotating bulk scratch");
        require(!pool.io_compatible(0, 1) && !pool.io_compatible(1, 0),
            "live stage modes changed full I/O compatibility");
    }
    first.reset();
    require(pool.available_for_output(0, target) == before, "I/O phase leases changed capacity reservations");
    require(pool.try_io(1).has_value(), "staging completion did not restore ordinary I/O admission");
    pool.cancel(0); pool.cancel(1); pool.cancel(2);
}
void staged_aliases(fs::path const& directory)
{
    auto scratch = root(directory / "a", "scratch", {"scratch"}, true);
    auto other_scratch = root(directory / "b", "other-scratch", {"other"}, true);
    auto target = root(directory / "c", "target", {"target"});
    auto scratch_target = root(directory / "d", "scratch-target", {"scratch"}, true);
    auto mixed_target = root(directory / "e", "mixed-target", {"target", "scratch"});
    auto unknown_target = root(directory / "f", "unknown-target", {});
    ScratchPool pool({scratch, other_scratch}, 0);
    for (auto const& [owner, output] : std::array<std::pair<uint64_t, ScratchRoot>, 5>{{
        {0, target}, {1, target}, {2, scratch_target}, {3, mixed_target}, {4, unknown_target}}})
        require(pool.try_reserve_output(owner, output, unit), "staging alias output reservation failed");
    std::array<uint64_t, 1> occupied{0};
    pool.restore(0, {{{other_scratch.directory, 10 * unit}}, unit, 2}, occupied);
    pool.restore(1, {{{scratch.directory, 10 * unit}}, unit, 2}, occupied);
    pool.restore(2, {{{other_scratch.directory, 10 * unit}}, unit, 2}, occupied);
    pool.restore(3, {{{scratch.directory, 10 * unit}}, unit, 2}, occupied);
    pool.restore(4, {{{scratch.directory, 10 * unit}}, unit, 2}, occupied);
    pool.seal_recovery();
    require(!pool.try_stage_io(3), "partially overlapping stage target and scratch were treated as independent");
    require(!pool.try_stage_io(4), "unknown staging target was treated as independent");
    auto first = pool.try_stage_io(1); require(first.has_value(), "alias fixture staging lease failed");
    require(!pool.try_io(2) && !pool.try_stage_io(2), "another output overlapped staging bulk scratch");
    require(!pool.try_io(3) && !pool.try_io(4), "mixed or unknown output bypassed staging scratch exclusion");
    first.reset();
    auto packed = pool.try_io(2); require(packed.has_value(), "reverse alias packing lease failed");
    require(!pool.try_stage_io(1), "staging bulk scratch overlapped another active output");
    packed.reset();
    // A staging target also cannot disturb another owner's bulk scratch.
    auto ordinary = pool.try_io(1); require(ordinary.has_value(), "reverse scratch alias lease failed");
    require(!pool.try_stage_io(2), "staging metadata target overlapped another owner's scratch");
    ordinary.reset();
    for (uint64_t owner = 0; owner < 5; ++owner) pool.release_after_cleanup(owner, 0);
}
void saved_staging(fs::path const& directory)
{
    auto scratch = root(directory / "a", "scratch", {"scratch"}, true);
    auto target = root(directory / "c", "target", {"target"});
    ScratchPool pool({scratch}, 0);
    ExternalConfig saved{{{scratch.directory, 20 * unit}}, unit, 2};
    std::array<uint64_t, 1> occupied{unit};
    for (uint64_t owner : {0u, 1u})
    {
        require(pool.try_reserve_output(owner, target, 2 * unit, unit, true), "saved staging output failed");
        pool.restore(owner, saved, occupied);
    }
    pool.seal_recovery();
    auto packed = pool.try_io(0); auto staged = pool.try_stage_io(1);
    require(packed && staged, "saved consecutive owners could not resume pack and independent staging");
    rejected([&] { pool.release_after_cleanup(0, unit); }, "packing owner was released before joining I/O");
    rejected([&] { pool.cancel(1); }, "saved staging owner was cancelled before cleanup");
    packed.reset(); pool.release_after_cleanup(0, unit);
    rejected([&] { pool.release_after_cleanup(1, unit); }, "earlier cleanup released the following owner's lease");
    staged.reset(); pool.release_after_cleanup(1, unit);
}
void staged_distributed(fs::path const& directory)
{
    auto first = root(directory / "a", "first-scratch", {"first"});
    auto second = root(directory / "b", "second-scratch", {"second"});
    auto target = root(directory / "c", "target", {"target"});
    ScratchPool pool({first, second}, 0);
    for (uint64_t owner : {0u, 1u, 2u})
        require(pool.try_reserve_output(owner, target, unit), "distributed staging output failed");
    std::array<uint64_t, 1> occupied{0}; std::array<uint64_t, 2> both{0, 0};
    pool.restore(0, {{{first.directory, 10 * unit}}, unit, 2}, occupied);
    pool.restore(1, {{{second.directory, 10 * unit}}, unit, 2}, occupied);
    pool.restore(2, {{{first.directory, 10 * unit}, {second.directory, 10 * unit}}, unit, 2}, both);
    pool.seal_recovery();
    auto packed = pool.try_io(0); auto staged = pool.try_stage_io(1);
    require(packed && staged, "independent rotating scratch could not stage beside a shared-target pack");
    require(!pool.try_stage_io(2), "a partially shared rotating scratch grant bypassed exclusion");
    packed.reset(); staged.reset();
    auto first_stage = pool.try_stage_io(0); auto second_stage = pool.try_stage_io(1);
    require(first_stage && second_stage, "independent rotating staging owners were needlessly serialized");
    first_stage.reset(); second_stage.reset();
    for (uint64_t owner : {0u, 1u, 2u}) pool.release_after_cleanup(owner, 0);
}
void distributed(fs::path const& directory)
{
    auto a = root(directory / "a", "scratch-a", {"drive-a"}, false, 40 * unit);
    auto b = root(directory / "b", "scratch-b", {"drive-b"}, false, 40 * unit);
    auto output = root(directory / "c", "target", {"target"});
    ScratchPool pool({a, b}, 0);
    require(pool.try_reserve_output(0, output, unit) && pool.try_reserve_output(1, output, unit), "distributed output reservation failed");
    pool.seal_recovery(); auto first = pool.grant(0, 30 * unit, 30 * unit, unit, 2);
    require(first && first->scratch.size() == 1 && first->scratch[0].directory == a.directory,
        "fresh grants needlessly claimed every physical scratch drive");
    auto next = pool.grant(1, 30 * unit, 30 * unit, unit, 2);
    require(next && quota(*next) == 30 * unit && next->scratch.size() == 1 && next->scratch[0].directory == b.directory,
        "a second owner shared rotating scratch while an independent drive was available");
}
void automatic_storage(fs::path const& directory)
{
    auto hdd = root(directory / "a", "hdd", {"hdd"});
    auto ssd = root(directory / "b", "ssd", {"ssd"}, true);
    auto target = root(directory / "c", "target", {"target"}, true);
    ScratchPool preferred({hdd, ssd}, 0);
    require(preferred.try_reserve_output(0, target, unit), "automatic output reservation failed");
    preferred.seal_recovery(); auto grant = preferred.grant(0, 20 * unit, unit, unit, 2, true, 8);
    require(grant && grant->scratch.size() == 1 && grant->scratch[0].directory == ssd.directory,
        "automatic scratch selected rotating storage ahead of a supplied SSD");
    require(grant->fan_in == 2 && grant->packing_inputs == 8,
        "SSD run retention changed compaction arity or ignored the admitted reader set");
    ScratchPool slower({hdd}, 0);
    require(slower.try_reserve_output(0, target, unit), "slower scratch output reservation failed");
    slower.seal_recovery();
    require(!slower.grant(0, 20 * unit, unit, unit, 2, true), "automatic scratch slowed a known SSD destination");
    auto rotating = slower.grant(0, 20 * unit, unit, unit, 2, false, 8);
    require(rotating && rotating->packing_inputs == rotating->fan_in,
        "explicit rotating scratch ignored the storage choice or expanded its reader set");
    auto rotating_target = target; rotating_target.device.nonrotating = false;
    for (auto const [buffers, readers] : {std::pair{16ull, 2u}, {64ull, 4u}, {128ull, 8u}, {1024ull, 8u}})
    {
        ScratchPool buffered({hdd}, 0);
        require(buffered.try_reserve_output(0, rotating_target, unit), "buffered HDD output reservation failed");
        buffered.seal_recovery();
        auto retained = buffered.grant(0, 20 * unit, unit, unit, 2, true, 8, buffers * 1024 * 1024);
        require(retained && retained->fan_in == 2 && retained->packing_inputs == readers,
            "HDD run retention ignored its reader buffer or changed compaction arity");
    }
    auto unknown = ssd; unknown.device.drives.clear();
    ScratchPool unclassified({unknown}, 0);
    require(unclassified.try_reserve_output(0, target, unit), "unknown scratch output reservation failed");
    unclassified.seal_recovery(); auto conservative = unclassified.grant(0, 20 * unit, unit, unit, 2, true, 8, 128ull * 1024 * 1024);
    require(conservative && conservative->packing_inputs == conservative->fan_in,
        "unknown scratch expanded its reader set");
}
void guards(fs::path const& directory)
{
    auto a = root(directory / "a", "scratch", {"one"});
    ScratchPool pool({a}, 0);
    fs::rename(directory / "a", directory / "old-a"); fs::create_directory(directory / "a");
    rejected([&] { pool.seal_recovery(); }, "replaced scratch root was accepted before modification");
    fs::remove(directory / "a"); fs::rename(directory / "old-a", directory / "a");
}
void headroom(fs::path const& directory)
{
    auto full = root(directory / "a", "full", {"one"}, false, unit);
    auto output = root(directory / "c", "target", {"two"});
    ScratchPool pool({full}, 2 * unit);
    require(!pool.try_reserve_output(0, full, 4 * unit, 4 * unit, true),
        "saved physical credit bypassed filesystem headroom");
    require(pool.try_reserve_output(1, output, unit), "an unused full temp filesystem blocked another destination");
    pool.seal_recovery(); require(!pool.grant(1, unit, unit, unit, 2), "scratch used filesystem headroom");
}
void many_owners(fs::path const& directory)
{
    auto scratch = root(directory / "a", "shared-ssd", {"ssd"}, true, 1024 * unit);
    ScratchPool pool({scratch}, 0); pool.seal_recovery();
    std::vector<ScratchPool::IoLease> leases;
    for (uint64_t i = 0; i < 64; ++i)
    {
        auto const path = directory / ("destination-" + std::to_string(i)); fs::create_directory(path);
        auto target = root(path, "filesystem-" + std::to_string(i), {"drive-" + std::to_string(i)});
        require(pool.try_reserve_output(i, target, unit), "many-owner output admission failed");
        require(pool.grant(i, unit, unit, unit, 2).has_value(), "many-owner shared SSD grant failed");
        auto lease = pool.try_io(i); require(lease.has_value(), "independent final writers could not share known SSD scratch");
        leases.push_back(std::move(*lease));
    }
    require(pool.memory_bytes() > sizeof(ScratchPool), "pool metadata allocations were omitted from RAM accounting");
    leases.clear();
    for (uint64_t i = 0; i < 64; ++i)
    {
        pool.cancel(i); auto const path = directory / ("destination-" + std::to_string(i));
        require(fs::is_empty(path), "many-owner admission wrote an output"); fs::remove(path);
    }
}
void ssd_backlog(fs::path const& directory)
{
    auto a = root(directory / "a", "scratch-a", {"ssd-a"}, true, 1024 * unit);
    auto alias = root(directory / "b", "scratch-partition", {"ssd-a"}, true, 1024 * unit);
    auto b = root(directory / "e", "scratch-b", {"ssd-b"}, true, 1024 * unit);
    auto target = root(directory / "c", "output", {"output"});
    ScratchPool pool({a, alias, b}, 0);
    require(pool.try_reserve_output(0, target, unit) && pool.try_reserve_output(1, target, unit),
        "SSD backlog fixture output admission failed");
    pool.seal_recovery();
    auto first = pool.grant(0, 20 * unit, 20 * unit, unit, 2);
    auto next = pool.grant(1, 20 * unit, 20 * unit, unit, 2);
    require(first && next && first->scratch.front().directory == a.directory && next->scratch.front().directory == b.directory,
        "SSD routing ignored a physical sibling's reserved backlog");
    pool.cancel(0); pool.cancel(1);

    ScratchPool balanced({a, b}, 0); balanced.seal_recovery();
    uint64_t a_bytes = 0, b_bytes = 0;
    for (uint64_t i = 0; i < 60; ++i)
    {
        auto const path = directory / ("balanced-destination-" + std::to_string(i)); fs::create_directory(path);
        auto output = root(path, "balanced-filesystem-" + std::to_string(i), {"balanced-drive-" + std::to_string(i)});
        require(balanced.try_reserve_output(i, output, unit), "many-destination backlog output admission failed");
        auto grant = balanced.grant(i, unit, unit, unit, 2);
        require(grant && quota(*grant) == unit, "backlog ranking changed an exact capacity grant");
        if (grant->scratch.front().directory == a.directory) a_bytes += quota(*grant);
        else if (grant->scratch.front().directory == b.directory) b_bytes += quota(*grant);
        else throw std::runtime_error("backlog routing selected an unsupplied root");
        require(a_bytes <= b_bytes + unit && b_bytes <= a_bytes + unit,
            "known independent SSD reservations did not balance conservatively");
    }
    for (uint64_t i = 0; i < 60; ++i)
    {
        balanced.cancel(i); fs::remove(directory / ("balanced-destination-" + std::to_string(i)));
    }

    a.device.drives.clear(); b.device.drives.clear();
    ScratchPool unknown({a, b}, 0);
    require(unknown.try_reserve_output(0, target, unit) && unknown.try_reserve_output(1, target, unit),
        "unknown backlog fixture output admission failed");
    unknown.seal_recovery();
    require(unknown.grant(0, unit, unit, unit, 2).has_value(), "unknown first grant failed");
    auto fallback = unknown.grant(1, unit, unit, unit, 2);
    require(fallback && fallback->scratch.front().directory == a.directory,
        "unknown scratch acquired an invented physical routing advantage");
    auto io = unknown.try_io(0);
    require(io && !unknown.try_io(1), "backlog ranking weakened unknown-resource I/O exclusion");
    io.reset(); unknown.cancel(0); unknown.cancel(1);
}
void rotating_bandwidth(fs::path const& directory)
{
    std::vector<ScratchRoot> disks;
    for (std::string name : {"a", "b", "c", "d"})
    {
        disks.push_back(root(directory / name, name, {name}));
        disks.back().device.rotating = true;
    }
    auto first = root(directory / "e", "output-1", {"output-1"});
    auto second = root(directory / "f", "output-2", {"output-2"});
    ScratchPool probe(disks, 0);
    require(probe.try_reserve_output(0, first, unit), "probe output reservation failed");
    for (size_t i = 1; i < disks.size(); ++i)
        require(probe.try_reserve_output(i, disks[i - 1], unit), "queued writer reservation failed");
    probe.seal_recovery();
    auto const room = probe.available_for_output(0, first);
    auto const candidate = probe.grant(0, 16 * unit, 16 * unit, unit, 2, true, 4, unit, 1, 1, {}, true, true);
    require(candidate && candidate->scratch.size() == 1 && candidate->scratch.front().directory == disks.back().directory &&
        probe.scratch_devices(0).empty() && probe.available_for_output(0, first) == room &&
        probe.scratch_devices(*candidate).front().drives == disks.back().device.drives,
        "pricing an optional route claimed capacity or included an already queued writer in its stripe");
    auto const commit = probe.grant(0, 16 * unit, 16 * unit, unit, 2, true, 4, unit, 4, 1, {}, false, true);
    require(commit && commit->scratch.front().directory == candidate->scratch.front().directory &&
        probe.try_io(1).has_value(), "the optional receiver stranded a ready writer after its checked grant");
    require(!probe.grant(1, 120 * unit, 120 * unit, unit, 2, true, 4, unit, 4, 1, {}, true, true),
        "a narrow optional grant occupied extra spindles to meet its capacity request");
    auto observe = std::make_shared<DiskIoGovernor>(DiskIoGovernor::Sampler{});
    ScratchPool pool(disks, 0); pool.set_io_governor(observe);
    for (size_t i = 0; i < disks.size(); ++i) observe->observe_transfer(disks[i].device, (4 - i) * unit, unit);
    require(pool.try_reserve_output(0, first, unit) && pool.try_reserve_output(1, second, unit), "HDD output admission failed");
    pool.seal_recovery();
    auto a = pool.grant(0, 50 * unit, 50 * unit, unit, 2, false, 0, 0, 4, 2);
    auto b = pool.grant(1, 50 * unit, 50 * unit, unit, 2, false, 0, 0, 4, 1);
    auto check = [&](ExternalConfig const& grant, size_t root, uint64_t bytes)
    { return std::any_of(grant.scratch.begin(), grant.scratch.end(), [&](auto const& part)
        { return part.directory == disks[root].directory && part.capacity_bytes == bytes; }); };
    require(a && b && a->striped && b->striped && pool.io_compatible(0, 1) &&
        check(*a, 0, 40 * unit) && check(*a, 3, 10 * unit) &&
        check(*b, 1, 30 * unit) && check(*b, 2, 20 * unit),
        "unequal HDDs failed to balance simultaneous owner bandwidth and proportional capacity");
    pool.cancel(0); pool.cancel(1);

    ScratchPool rotating_owners(disks, 0);
    require(rotating_owners.try_reserve_output(0, first, unit) && rotating_owners.try_reserve_output(1, second, unit) &&
        rotating_owners.try_reserve_output(2, first, unit), "rotating output reservations failed");
    rotating_owners.seal_recovery();
    auto narrow = rotating_owners.grant(0, 30 * unit, 30 * unit, unit, 2);
    auto replacement = rotating_owners.grant(1, 30 * unit, 30 * unit, unit, 2, false, 0, 0, 4, 1);
    require(narrow && narrow->scratch.size() == 1 && replacement && replacement->scratch.size() == 2 &&
        quota(*replacement) == 30 * unit && rotating_owners.io_compatible(0, 1),
        "a replacement stripe took the next writer's share of the available HDDs");
    rotating_owners.cancel(0); rotating_owners.cancel(1); rotating_owners.cancel(2);

    auto writer = second; writer.device.rotating = true;
    struct SuccessorCase { uint64_t demand; uint64_t writer_rate; bool free_target; };
    for (auto [demand, writer_rate, free_target] : std::array<SuccessorCase, 3>{{
        {30 * unit, 1, true}, {150 * unit, 1, false}, {30 * unit, 4, false}}})
    {
        auto rates = std::make_shared<DiskIoGovernor>(DiskIoGovernor::Sampler{});
        ScratchPool tail(disks, 0); tail.set_io_governor(rates);
        require(tail.try_reserve_output(0, first, unit) && tail.try_reserve_output(1, writer, unit) &&
            tail.try_reserve_output(2, disks[1], 0), "queued successor outputs were not admitted");
        for (auto const& disk : disks) rates->observe_transfer(disk.device, unit, unit);
        rates->observe_transfer(writer.device, writer_rate * unit, unit);
        tail.seal_recovery();
        auto running = tail.grant(0, 30 * unit, 30 * unit, unit, 2, false, 0, 0, 4, 2);
        auto stripe = tail.grant(1, demand, demand, unit, 2, false, 0, 0, 4, 1);
        require(running && stripe && quota(*stripe) == demand && tail.io_compatible(0, 1) &&
            tail.target_io_compatible(disks[1].device, 1) == free_target,
            "a scratch stripe must free the next writer when enough capacity and bandwidth remain");
    }

    struct ReplacementCase { uint64_t room, spare_rate, writer_rate; bool free_target; };
    for (auto [room, spare_rate, writer_rate, free_target] : std::array<ReplacementCase, 4>{{
        {100 * unit, 3, 3, true}, {100 * unit, 4, 5, true},
        {100 * unit, 3, 4, false}, {20 * unit, 3, 3, false}}})
    {
        auto roots = disks;
        roots[2].available_bytes = room;
        auto rates = std::make_shared<DiskIoGovernor>(DiskIoGovernor::Sampler{});
        ScratchPool tail(roots, 0); tail.set_io_governor(rates);
        require(tail.try_reserve_output(0, first, unit) && tail.try_reserve_output(1, writer, unit) &&
            tail.try_reserve_output(2, roots[0], 0) && tail.try_reserve_output(3, roots[1], 0) &&
            tail.try_reserve_output(4, roots[2], unit), "replacement scratch fixture output admission failed");
        tail.seal_recovery();
        tail.release_after_cleanup(4, unit);
        auto running = tail.grant(0, 30 * unit, 30 * unit, unit, 2);
        auto running_io = tail.try_io(0);
        require(running && running_io && check(*running, 0, 30 * unit) && !tail.try_io(2),
            "replacement scratch fixture did not block the earlier queued output");
        rates->observe_transfer(roots[0].device, unit, unit);
        rates->observe_transfer(roots[1].device, 4 * unit, unit);
        rates->observe_transfer(roots[2].device, spare_rate * unit, unit);
        rates->observe_transfer(roots[3].device, unit, unit);
        rates->observe_transfer(writer.device, writer_rate * unit, unit);
        auto stripe = tail.grant(1, 30 * unit, 30 * unit, unit, 2, false, 0, 0, 4, 1);
        auto stripe_io = tail.try_io(1);
        require(stripe && stripe_io && stripe->scratch.size() == 1 &&
            check(*stripe, free_target ? 2 : 1, 30 * unit) && bool(tail.try_io(3)) == free_target,
            "scratch failed to preserve a runnable output within the stripe bandwidth and capacity");
    }

    ScratchPool queued(disks, 0);
    require(queued.try_reserve_output(0, first, unit) && queued.try_reserve_output(1, first, unit) &&
        queued.try_reserve_output(2, disks.front(), unit), "queued physical output admission failed");
    queued.seal_recovery();
    auto parallel = queued.grant(0, 50 * unit, 50 * unit, unit, 2, false, 0, 0, 4, 2);
    require(parallel && parallel->striped && queued.io_compatible(0, 2) && !queued.io_compatible(0, 1),
        "a queued file on the same output took the scratch bandwidth of an independent writer");

    ScratchPool farm(disks, 0);
    for (size_t i = 0; i < disks.size(); ++i)
        require(farm.try_reserve_output(i, disks[i], i ? 0 : 80 * unit), "farm output registration failed");
    farm.seal_recovery();
    auto paired = farm.grant(0, 60 * unit, 60 * unit, unit, 2, true, 0, 0, 4, 2);
    require(paired && paired->scratch.size() == 1 && farm.io_compatible(0, 1),
        "a second eligible output lost its independent writer or scratch bandwidth");
    farm.release_scratch_after_cleanup(0);
    auto scratch = farm.grant(0, 60 * unit, 60 * unit, unit, 2, true, 0, 0, 4, 1);
    require(scratch && scratch->striped && scratch->scratch.size() == 3 &&
        farm.available_for_output(1, disks[1]) == 80 * unit && farm.available_for_output(1, disks[1], true) == 100 * unit &&
        !farm.io_compatible(0, 1) && !farm.try_reserve_output(1, disks[1], 100 * unit),
        "idle output disks failed to lend scratch exclusively to a striped farm group");
    farm.release_after_cleanup(0, unit);
    require(farm.try_reserve_output(1, disks[1], 100 * unit), "cleaned scratch could not become a full output");
    auto rotated = farm.grant(1, 60 * unit, 60 * unit, unit, 2, true, 0, 0, 4, 1);
    require(rotated && rotated->scratch.size() == 3 && check(*rotated, 0, 20 * unit),
        "a completed output could not contribute scratch to the next group");

    for (auto const [limit, tight, readers] : {std::tuple{24u, false, 16u}, {6u, false, 6u}, {24u, true, 4u}})
    {
        Config work; work.memory_limit_bytes = 4ull << 30;
        ScratchPool striped(disks, 0);
        require(striped.try_reserve_output(0, first, unit), "striped reader fixture output admission failed");
        striped.seal_recovery();
        auto layout = [&](ExternalConfig& external)
        {
            if (tight)
            {
                auto serial = external; serial.striped = false;
                work.memory_limit_bytes = std::max(
                    ExternalEngine::phase_memory_bytes(26, work, serial, ExternalEngine::MemoryPhase::staging, UINT16_MAX),
                    ExternalEngine::phase_memory_bytes(26, work, serial, ExternalEngine::MemoryPhase::packing));
            }
            return ExternalEngine::striped_reader_bytes(26, work, external);
        };
        auto grant = striped.grant(0, 80 * unit, 80 * unit, unit, 4, false, limit, 8 * work.window_bytes, 4, 1, layout);
        require(grant && grant->striped && grant->scratch.size() == 4 && grant->packing_inputs == readers,
            "striped run retention did not respect the admitted reader cache and membership limit");
    }

    auto alias = disks.front(); alias.directory = first.directory; alias.directory_inode = first.directory_inode;
    alias.device.filesystem = "other-partition";
    disks.push_back(alias);
    ScratchPool alone(disks, 0);
    require(alone.try_reserve_output(0, second, unit) && alone.try_reserve_output(1, second, unit), "queued HDD output admission failed");
    alone.seal_recovery();
    auto wide = alone.grant(0, 100 * unit, 100 * unit, unit, 2, true, 0, 0, 8, 1);
    require(wide && wide->scratch.size() == 4 && quota(*wide) == 100 * unit,
        "a queued owner stranded bandwidth or a partition multiplied one HDD");

    using namespace std::chrono;
    std::array<DiskIoGovernor::Counters, 2> counters{};
    DiskIoGovernor shared([&](std::string const& name) { return std::optional{counters[name == "a" ? 0 : 1]}; });
    std::array<StorageDevice, 2> pair{disks[0].device, disks[1].device};
    for (auto const& device : pair) shared.register_device(device);
    auto const begin = DiskIoGovernor::Clock::now(); shared.poll(begin);
    counters[0].read_sectors = 200000; counters[0].busy_ms = 1000;
    require(!shared.concurrent_bandwidth(pair, begin + seconds(1)), "an isolated drive supplied aggregate controller bandwidth");
    counters[0].read_sectors += 100000; counters[0].busy_ms += 1000;
    counters[1].write_sectors = 150000; counters[1].busy_ms = 1000;
    require(std::abs(shared.concurrent_bandwidth(pair, begin + seconds(2)) - 250000 * 512.L / 1000000000) < 1e-15L,
        "shared paths were priced at isolated rates despite concurrent block measurements");
    counters[0].read_sectors += 100000; counters[0].busy_ms += 500;
    auto const concurrent = shared.concurrent_bandwidth(pair, begin + seconds(3));
    auto const available = shared.bandwidth(pair[0], begin + seconds(3));
    auto const observed = shared.observed_bulk_bandwidth(pair[0].drives.front(), begin + seconds(3));
    require(observed && observed->first >= observed->second && std::abs(observed->second - available) < 1e-15L &&
        !shared.observed_bulk_bandwidth("unobserved"), "read-only disk telemetry disagreed with scheduler bandwidth");
    require(concurrent > 0 && available < shared.bandwidth(pair[0], begin + seconds(5)),
        "a partial foreign load failed to reserve its bandwidth or erased a joint measurement");

    // Both disks complete our writes. A competing sequential reader on one
    // of them cannot turn its extra sectors into bandwidth for this plot.
    DiskIoGovernor competing([&](std::string const& name) { return std::optional{counters[name == "a" ? 0 : 1]}; });
    for (auto const& device : pair) competing.register_device(device);
    competing.poll(begin);
    constexpr uint64_t mib = 1024 * 1024;
    for (size_t i = 0; i < pair.size(); ++i)
    {
        competing.observe_transfer(pair[i], 16 * mib, 0, DiskIoGovernor::Transfer::write, begin + milliseconds(100));
        counters[i].write_sectors += 16 * mib / 512;
        counters[i].busy_ms += i ? 100 : 1000;
    }
    counters[0].read_sectors += 144 * mib / 512;
    auto const busy = competing.bandwidth(pair[0], begin + seconds(1));
    require(busy > 0 && busy < competing.bandwidth(pair[1], begin + seconds(1)) / 5,
        "another reader's throughput was credited to an active scratch writer");
    competing.observe_transfer(pair[1], 32 * mib, 0, DiskIoGovernor::Transfer::write, begin + milliseconds(1100));
    competing.poll(begin + seconds(2));
    counters[1].write_sectors += 32 * mib / 512; counters[1].busy_ms += 1000;
    auto const writeback = competing.bandwidth(pair[1], begin + seconds(3));
    competing.observe_transfer(pair[1], 32 * mib, 1000000, DiskIoGovernor::Transfer::flush, begin + milliseconds(3100));
    counters[1].write_sectors += 32 * mib / 512; counters[1].busy_ms += 1000;
    require(writeback > 0 && competing.bandwidth(pair[1], begin + seconds(4)) < writeback / 5,
        "deferred own writeback or its flush hid a subsequent outside writer");
}
void disk_feedback(fs::path const& directory)
{
    using Clock = DiskIoGovernor::Clock;
    using namespace std::chrono;
    auto const begin = Clock::time_point(seconds(100));
    StorageDevice a{"a", {"disk-a"}}, alias{"alias", {"disk-a"}}, b{"b", {"disk-b"}}, c{"c", {"disk-c"}};
    StorageDevice ssd{"ssd", {"ssd"}, true};
    DiskIoGovernor governor([](std::string const&) { return std::optional<DiskIoGovernor::Counters>{}; });
    for (auto const* device : {&a, &alias, &b, &c, &ssd}) governor.register_device(*device);
    governor.observe_farming(10000000, 1000000000, false, begin);
    auto first = governor.try_acquire(a, 8 * 1024 * 1024, begin);
    require(first && first->bytes() <= 1024 * 1024 && !governor.try_acquire(b, unit, begin),
        "cold disk feedback admitted an unbounded burst or an unmeasured working set");
    first->complete(1024 * 1024, 10000000, begin + milliseconds(10));
    governor.poll(begin + seconds(2));
    require(governor.status(begin + seconds(2)).limit == 2, "completed throughput and independent backlog did not trial another HDD");
    first = governor.try_acquire(a, unit, begin + seconds(2));
    auto second = governor.try_acquire(b, unit, begin + seconds(2));
    require(first && second && !governor.try_acquire(alias, unit, begin + seconds(2)),
        "independent HDDs did not overlap or aliases doubled physical concurrency");
    auto flash = governor.try_acquire(ssd, 8 * 1024 * 1024, begin + seconds(2));
    require(flash && flash->bytes() == 8 * 1024 * 1024 && governor.status(begin + seconds(2)).drives == 3,
        "farm disk feedback changed known SSD transfers or counted physical aliases twice");
    first->complete(2 * 1024 * 1024, 10000000, begin + seconds(2));
    second->complete(2 * 1024 * 1024, 10000000, begin + seconds(2));
    governor.poll(begin + seconds(4));
    require(governor.status(begin + seconds(4)).limit == 2, "a successful throughput trial was discarded");
    first = governor.try_acquire(a, unit, begin + seconds(4));
    second = governor.try_acquire(b, unit, begin + seconds(4));
    require(first && second && !governor.try_acquire(c, unit, begin + seconds(4)), "HDD trial exceeded its current working set");
    first->complete(2 * 1024 * 1024, 10000000, begin + seconds(4));
    second->complete(2 * 1024 * 1024, 10000000, begin + seconds(4));
    governor.poll(begin + seconds(6));
    require(governor.status(begin + seconds(6)).limit == 3, "healthy farm feedback did not expand gradually");
    governor.observe_farming(600000000, 1000000000, false, begin + seconds(6));
    require(governor.status(begin + seconds(6)).limit == 1 && governor.status(begin + seconds(6)).pressured,
        "real farming response pressure did not reduce rotating disk activity");
    first = governor.try_acquire(a, unit, begin + seconds(7));
    require(first && !governor.try_acquire(b, unit, begin + seconds(7)), "pressure blocked all forward progress or admitted another HDD");
    first->complete(2 * 1024 * 1024, 10000000, begin + seconds(7));
    governor.observe_farming(10000000, 1000000000, false, begin + seconds(8));
    governor.poll(begin + seconds(8));
    require(governor.status(begin + seconds(8)).limit == 1, "one healthy sample bypassed recovery hold");
    first = governor.try_acquire(a, unit, begin + seconds(10));
    require(first && !governor.try_acquire(b, unit, begin + seconds(10)), "recovery working set escaped its current bound");
    first->complete(2 * 1024 * 1024, 10000000, begin + seconds(10));
    governor.poll(begin + seconds(12));
    require(governor.status(begin + seconds(12)).limit == 2, "safe completed work could not recover after pressure");
    first = governor.try_acquire(a, unit, begin + seconds(12));
    second = governor.try_acquire(b, unit, begin + seconds(12));
    require(first && second, "recovered independent disks were not usable");
    first->complete(2 * 1024 * 1024, 10000000, begin + seconds(12));
    second->complete(2 * 1024 * 1024, 10000000, begin + seconds(12));
    governor.poll(begin + seconds(14)); // Accept the recovery trial.
    first = governor.try_acquire(a, unit, begin + seconds(14));
    second = governor.try_acquire(b, unit, begin + seconds(14));
    require(first && second && !governor.try_acquire(c, unit, begin + seconds(14)), "controller trial escaped its current bound");
    first->complete(2 * 1024 * 1024, 10000000, begin + seconds(14));
    second->complete(2 * 1024 * 1024, 10000000, begin + seconds(14));
    governor.poll(begin + seconds(16));
    first = governor.try_acquire(a, unit, begin + seconds(16));
    second = governor.try_acquire(b, unit, begin + seconds(16));
    auto third = governor.try_acquire(c, unit, begin + seconds(16));
    require(first && second && third, "bounded controller throughput trial did not start");
    first->complete(1024 * 1024, 10000000, begin + seconds(16));
    second->complete(1024 * 1024, 10000000, begin + seconds(16));
    third->complete(1024 * 1024, 10000000, begin + seconds(16));
    governor.poll(begin + seconds(18));
    require(governor.status(begin + seconds(18)).limit == 2, "additional disks without a throughput gain ignored a shared-controller ceiling");
    governor.poll(begin + seconds(60));
    require(!governor.status(begin + seconds(60)).farming_known && governor.status(begin + seconds(60)).limit <= 2,
        "stale farming feedback retained an expanded disk working set");

    DiskIoGovernor::Counters writeback;
    DiskIoGovernor cached([&](std::string const&) { return std::optional(writeback); });
    auto buffered = cached.try_acquire(a, 1024 * 1024, begin);
    require(buffered.has_value(), "cached-write fixture was not admitted");
    auto const initial_quantum = buffered->bytes(); buffered->complete(initial_quantum, 1000, begin + milliseconds(1));
    writeback.busy_ms = 100; writeback.write_sectors = 2048; writeback.queue_ms = 100;
    cached.poll(begin + milliseconds(250));
    buffered = cached.try_acquire(a, 8 * 1024 * 1024, begin + milliseconds(250));
    require(buffered && buffered->bytes() < initial_quantum && !cached.status(begin + milliseconds(250)).pressured,
        "kernel service behind cached writes was ignored or our own I/O became foreign farm pressure");
    buffered->complete(buffered->bytes(), 1000, begin + milliseconds(250));
    require(!cached.try_acquire(a, unit, begin + milliseconds(251)) && cached.try_acquire(a, unit, begin + milliseconds(500)),
        "cached writes bypassed physical service pacing or self-saturation blocked recovery indefinitely");

    // Busy counters only on an inactive drive are foreign-load evidence.
    // A continuously active own writer must not permanently throttle itself.
    std::map<std::string, DiskIoGovernor::Counters> samples{{"disk-a", {}}, {"disk-b", {}}};
    auto live = std::make_shared<DiskIoGovernor>([&](std::string const& name) -> std::optional<DiskIoGovernor::Counters>
    { if (auto at = samples.find(name); at != samples.end()) return at->second; return {}; });
    live->register_device(a); live->register_device(b); live->selected(a); live->selected(b); live->poll();
    samples["disk-a"].busy_ms = 1000000; samples["disk-a"].queue_ms = 1000000;
    std::this_thread::sleep_for(milliseconds(300)); live->poll();
    require(live->routing_cost(a) > live->routing_cost(b),
        "live foreign disk queue pressure did not change scratch routing");
    auto ra = root(directory / "a", "a", a.drives), rb = root(directory / "b", "b", b.drives);
    ScratchPool pool({ra, rb}, 0); pool.set_io_governor(live);
    require(pool.try_reserve_output(0, root(directory / "c", "target", {"target"}), unit), "feedback output reservation failed");
    pool.seal_recovery();
    auto grant = pool.grant(0, 20 * unit, 20 * unit, unit, 2);
    require(grant && quota(*grant) == 20 * unit && grant->scratch.front().directory == rb.directory && grant->io_governor == live,
        "runtime disk feedback changed full scratch capacity or lost its actual-I/O hook");
    auto const retained = quota(*grant); live->observe_pressure(true);
    require(quota(*grant) == retained, "farming pressure changed retained checkpoint quotas");
    for (size_t i = 0; i < 2 * live->status().drives; ++i) live->selected(b);
    require(live->routing_cost(a) < live->routing_cost(b), "a slow/contended physical drive was permanently starved");
    pool.cancel(0);

    DiskIoGovernor cancellable;
    auto held = cancellable.acquire(a, unit);
    std::atomic<bool> cancel{false}; std::promise<void> waiting; bool signalled = false;
    auto waiter = std::async(std::launch::async, [&]
    {
        try
        {
            auto permit = cancellable.acquire(b, unit, [&]
            {
                if (!signalled) { waiting.set_value(); signalled = true; }
                if (cancel.load()) throw ConstructionInterrupted();
            });
            return false;
        }
        catch (ConstructionInterrupted const&) { return true; }
    });
    require(waiting.get_future().wait_for(seconds(5)) == std::future_status::ready, "disk cancellation waiter never started");
    cancel.store(true);
    require(waiter.wait_for(seconds(5)) == std::future_status::ready && waiter.get(), "queued disk I/O could not drain on cancellation");
    held.complete(unit, 0);
    require(cancellable.try_acquire(b, unit).has_value(), "cancelled FIFO disk ticket blocked its successor permanently");

    DiskIoGovernor independent;
    auto warm = independent.try_acquire(a, unit, begin);
    require(warm && !independent.try_acquire(b, unit, begin), "independent FIFO fixture did not observe cold contention");
    warm->complete(unit, 0, begin); independent.poll(begin + seconds(2));
    auto slow = independent.acquire(a, unit);
    std::promise<void> queued; bool notified = false;
    auto same_disk = std::async(std::launch::async, [&]
    {
        auto permit = independent.acquire(alias, unit, [&]
        { if (!notified) { queued.set_value(); notified = true; } });
    });
    require(queued.get_future().wait_for(seconds(5)) == std::future_status::ready, "slow-drive FIFO waiter never started");
    auto other_disk = std::async(std::launch::async, [&] { auto permit = independent.acquire(b, unit); });
    bool const progressed = other_disk.wait_for(seconds(5)) == std::future_status::ready;
    slow.complete(unit, 0); same_disk.get(); other_disk.get();
    require(progressed, "a queued slow-drive request stalled a ready independent HDD");
}
void nvme_topology()
{
    auto field = [](std::string_view name)
    {
        std::array<uint8_t, 256> encoded{};
        require(name.size() < encoded.size(), "NQN test fixture is too large");
        std::copy(name.begin(), name.end(), encoded.begin()); return encoded;
    };
    std::string const first = "nqn.2026-11.com.example:namespace-\xc3\xa9-\xf0\x9f\x8c\xb1";
    std::string const second = "nqn.2014-08.org.nvmexpress:uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6";
    require(storage_detail::nvme_subsystem_name(field(first)) == first && storage_detail::nvme_subsystem_name(field(second)) == second,
        "documented Unicode/domain or UUID subsystem identity was changed");
    for (auto const& invalid : std::vector<std::string>{
        "", "independent-controller", "nqn.", "nqn.0000-11.com.example:value", "nqn.2026-00.com.example:value",
        "nqn.2026-13.com.example:value", "nqn.2026-11.example:value", "nqn.2026-11.com..example:value",
        "nqn.2026-11.com.example.:value", "nqn.2026-11.com.-example:value", "nqn.2026-11.com.example",
        "nqn.2026-11.com.example:", "nqn.2026-11.com.example:value\n", "nqn.2026-11.com.example:value with space",
        "nqn.2026-11.com.example:\xed\xa0\x80", "nqn.2026-11.com.example:\xe2\x82",
        "nqn.2014-08.org.nvmexpress:arbitrary", "nqn.2014-08.ORG.NVMEXPRESS:arbitrary",
        "nqn.2014-08.org.nvmexpress:uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bfz",
        "nqn.2014-08.org.nvmexpress:uuid:f81d4fae_7dec-11d0-a765-00a0c91e6bf6"})
        require(storage_detail::nvme_subsystem_name(field(invalid)).empty(), "malformed NQN established independent storage");
    auto maximum = std::string("nqn.2026-11.com.example:"); maximum.append(223 - maximum.size(), 'x');
    require(storage_detail::nvme_subsystem_name(field(maximum)) == maximum &&
        storage_detail::nvme_subsystem_name(field(maximum + 'x')).empty(), "NQN byte limit was not enforced");
    auto unterminated = field(first); unterminated.fill('x');
    auto const complete = field(first);
    require(storage_detail::nvme_subsystem_name(unterminated).empty() &&
        storage_detail::nvme_subsystem_name(std::span(complete).first(223)).empty(),
        "unterminated or truncated subsystem identity was accepted");
    std::vector<StorageDevice> devices{
        {"a", {"windows:disk:0", "windows:nvme:" + first}},
        {"b", {"windows:disk:1", "windows:nvme:" + first}},
        {"c", {"windows:disk:2", "windows:nvme:" + second}},
        {"alias-a", {"windows:disk:0", "windows:nvme:" + first}}};
    auto groups = writer_groups(devices);
    require(groups[0] == groups[1] && groups[0] == groups[3] && groups[0] != groups[2],
        "NVMe namespace/partition aliases or distinct subsystems received incorrect writer groups");
    devices.push_back({"unresolved", {}}); groups = writer_groups(devices);
    require(std::all_of(groups.begin(), groups.end(), [&](size_t group) { return group == groups.front(); }),
        "unresolved NVMe topology bypassed conservative whole-job serialization");
#ifdef _WIN32
    constexpr size_t offset = sizeof(STORAGE_PROTOCOL_DATA_DESCRIPTOR);
    std::vector<uint8_t> reply(offset + sizeof(NVME_IDENTIFY_CONTROLLER_DATA));
    STORAGE_PROTOCOL_DATA_DESCRIPTOR descriptor{}; descriptor.Version = descriptor.Size = sizeof(descriptor);
    auto& protocol = descriptor.ProtocolSpecificData; protocol.ProtocolType = ProtocolTypeNvme; protocol.DataType = NVMeDataTypeIdentify;
    protocol.ProtocolDataRequestValue = NVME_IDENTIFY_CNS_CONTROLLER;
    protocol.ProtocolDataOffset = sizeof(protocol); protocol.ProtocolDataLength = sizeof(NVME_IDENTIFY_CONTROLLER_DATA);
    NVME_IDENTIFY_CONTROLLER_DATA controller{}; controller.VID = 1; controller.NN = 1;
    std::copy(first.begin(), first.end(), controller.SUBNQN);
    auto encode = [&]
    { std::memcpy(reply.data(), &descriptor, sizeof(descriptor)); std::memcpy(reply.data() + offset, &controller, sizeof(controller)); };
    encode(); require(storage_detail::nvme_subsystem_reply(reply) == first, "valid Windows Identify reply lost subsystem identity");
    require(storage_detail::nvme_subsystem_reply(std::span(reply).first(sizeof(descriptor) - 1)).empty(),
        "short Windows Identify descriptor was accepted");
    for (unsigned invalid = 0; invalid < 10; ++invalid)
    {
        auto changed = descriptor;
        switch (invalid)
        {
        case 0: changed.Version = 0; break;
        case 1: changed.Size = 0; break;
        case 2: changed.Size = DWORD(reply.size() + 1); break;
        case 3: changed.ProtocolSpecificData.ProtocolType = ProtocolTypeUnknown; break;
        case 4: changed.ProtocolSpecificData.DataType = NVMeDataTypeLogPage; break;
        case 5: changed.ProtocolSpecificData.ProtocolDataRequestValue = NVME_IDENTIFY_CNS_SPECIFIC_NAMESPACE; break;
        case 6: changed.ProtocolSpecificData.ProtocolDataOffset = 0; break;
        case 7: changed.ProtocolSpecificData.ProtocolDataOffset = UINT32_MAX; break;
        case 8: changed.ProtocolSpecificData.ProtocolDataLength = UINT32_MAX; break;
        case 9: changed.ProtocolSpecificData.ProtocolDataLength = sizeof(NVME_IDENTIFY_CONTROLLER_DATA) - 1; break;
        }
        std::memcpy(reply.data(), &changed, sizeof(changed));
        require(storage_detail::nvme_subsystem_reply(reply).empty(), "malformed Windows Identify descriptor established independence");
    }
    encode(); require(storage_detail::nvme_subsystem_reply(std::span(reply).first(reply.size() - 1)).empty(), "truncated controller payload was accepted");
    controller.VID = 0; encode(); require(storage_detail::nvme_subsystem_reply(reply).empty(), "missing controller vendor was accepted");
    controller.VID = 1; controller.NN = 0; encode();
    require(storage_detail::nvme_subsystem_reply(reply).empty(), "missing controller namespaces were accepted");
#endif
}
int main() try
{
    nvme_topology();
#ifdef _WIN32
    fs::path directory;
    std::random_device random;
    for (unsigned attempt = 0; attempt < 32; ++attempt)
    {
        directory = fs::temp_directory_path() / ("xchplot2-scratch-test-" + std::to_string(random()) + "-" + std::to_string(random()));
        if (!file_io::make_directory(directory)) break;
        require(errno == EEXIST, "cannot create private scratch pool test directory"); directory.clear();
    }
    require(!directory.empty(), "cannot create unique scratch pool test directory");
#else
    std::string pattern = "/tmp/xchplot2-scratch-test-XXXXXX";
    require(::mkdtemp(pattern.data()), "cannot create scratch pool test directory");
    fs::path const directory(pattern);
#endif
    for (std::string const name : {"a", "b", "c", "d", "e", "f"}) fs::create_directory(directory / name);
    fs::create_directory_symlink(directory / "a", directory / "alias-a");
    shared_capacity(directory); output_adjustment(directory); saved_capacity(directory); saved_only_roots(directory); saved_aliases(directory);
    topology(directory, true, false); topology(directory, false, false); topology(directory, true, true);
    staged_topology(directory, true, false); staged_topology(directory, false, false); staged_topology(directory, true, true);
    staged_aliases(directory); saved_staging(directory); staged_distributed(directory);
    distributed(directory); automatic_storage(directory); guards(directory); headroom(directory); many_owners(directory); ssd_backlog(directory);
    rotating_bandwidth(directory); disk_feedback(directory);
    for (std::string const name : {"a", "b", "c", "d", "e", "f"})
        require(fs::is_empty(directory / name), "capacity discovery or admission wrote to a supplied directory");
    fs::remove_all(directory);
    std::cout << "shared scratch capacity, saved ownership and physical-drive admission checks passed\n";
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
