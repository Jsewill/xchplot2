#include "host/GroupMerge.hpp"
#include "host/GroupWriter.hpp"
#include "host/GroupPolicy.hpp"
#include "host/GroupFarmMigration.hpp"
#include "host/GroupFarmRetirement.hpp"
#include <iostream>
#include <sstream>
#include <sys/wait.h>

// Upstream and group-common headers define non-inline helpers, so the
// controller fixture shares this translation unit with the engine fixtures.
#include "host/GroupPlotter.cpp"
#include "../xchplot2/group_commands.hpp"

using namespace group_grow;
namespace fs = std::filesystem;

// Inject an actual writer syscall failure while a contribution producer may
// still borrow its next window. The executable alone wraps pwrite at link time.
#ifdef GROW_TEST_WRAP_PWRITE
std::atomic<bool> fail_write{false};
std::atomic<uint64_t> fail_write_begin{0}, fail_write_end{0};
std::atomic<bool> hold_background_write{false}, background_write_active{false};
std::atomic<bool> background_write_timed_out{false};
std::thread::id producer_thread;
std::promise<void>* background_write_started = nullptr;
std::shared_future<void> background_write_release;
std::atomic<bool> hold_background_check{false}, background_check_after{false};
std::string background_check_target;
std::atomic<bool> replace_unlinked{false};
std::string replacement_target;
std::function<void(int, char const*, int, char const*)> retirement_rename_hook;
extern "C" int __real_renameat2(int, char const*, int, char const*, unsigned);
extern "C" int __wrap_renameat2(int from, char const* old_name, int to, char const* new_name, unsigned flags)
{
    if (retirement_rename_hook)
    {
        auto hook = std::move(retirement_rename_hook); retirement_rename_hook = {};
        hook(from, old_name, to, new_name);
    }
    return __real_renameat2(from, old_name, to, new_name, flags);
}
extern "C" int __real_unlink(char const*);
extern "C" int __wrap_unlink(char const* path)
{
    int const result = __real_unlink(path);
    if (!result && replace_unlinked.load(std::memory_order_acquire) && replacement_target == path &&
        replace_unlinked.exchange(false, std::memory_order_acq_rel))
    {
        int const fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0)
        {
            constexpr char bytes[] = "replacement";
            (void)::write(fd, bytes, sizeof(bytes) - 1); ::close(fd);
        }
    }
    return result;
}
extern "C" ssize_t __real_pwrite(int, void const*, size_t, off_t);
extern "C" ssize_t __wrap_pwrite(int fd, void const* bytes, size_t count, off_t offset)
{
    if (std::this_thread::get_id() != producer_thread && hold_background_write.exchange(false, std::memory_order_acq_rel))
    {
        background_write_active.store(true, std::memory_order_release);
        background_write_started->set_value();
        if (background_write_release.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
            background_write_timed_out.store(true, std::memory_order_release);
        background_write_active.store(false, std::memory_order_release);
    }
    if (fail_write.load(std::memory_order_acquire) && uint64_t(offset) >= fail_write_begin.load(std::memory_order_relaxed) &&
        uint64_t(offset) < fail_write_end.load(std::memory_order_relaxed) && fail_write.exchange(false, std::memory_order_acq_rel))
    { errno = EIO; return -1; }
    return __real_pwrite(fd, bytes, count, offset);
}
extern "C" int __real_lstat(char const*, struct stat*);
extern "C" int __wrap_lstat(char const* path, struct stat* info)
{
    bool const captured = background_check_after.load(std::memory_order_acquire) &&
        std::this_thread::get_id() != producer_thread && path == background_check_target;
    int const result = captured ? __real_lstat(path, info) : 0;
    if ((!captured || !result) && std::this_thread::get_id() != producer_thread && path == background_check_target &&
        hold_background_check.exchange(false, std::memory_order_acq_rel))
    {
        background_write_active.store(true, std::memory_order_release);
        background_write_started->set_value();
        if (background_write_release.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
            background_write_timed_out.store(true, std::memory_order_release);
        background_write_active.store(false, std::memory_order_release);
    }
    return captured ? result : __real_lstat(path, info);
}
#endif

void test_producer_options()
{
    using namespace grow_controller_test;
    auto parse = [](std::initializer_list<char const*> options)
    {
        std::vector<char*> command;
        for (auto option : options) command.push_back(const_cast<char*>(option));
        return pos2gpu::parse_group_options(int(command.size()), command.data());
    };
    auto const defaults = parse({"grow", "group.gplot"});
    require(defaults.generation_overlap && defaults.generation_overlap_auto &&
        defaults.batch == defaults.max_members, "producer defaults do not allow admitted automatic work");
    auto const automatic = parse({"grow", "group.gplot", "--generation-overlap", "auto"});
    require(automatic.generation_overlap && automatic.generation_overlap_auto,
        "explicit automatic overlap differs from the default");
    auto const enabled = parse({"grow", "group.gplot", "--generation-overlap", "on", "--batch-members", "7"});
    require(enabled.generation_overlap && !enabled.generation_overlap_auto && enabled.batch == 7,
        "explicit producer overrides were not retained");
    auto const disabled = parse({"grow", "group.gplot", "--generation-overlap", "on", "--generation-overlap", "off"});
    require(!disabled.generation_overlap && !disabled.generation_overlap_auto,
        "explicit overlap off did not override an earlier request");
    bool rejected = false;
    try { (void)parse({"grow", "group.gplot", "--generation-overlap", "invalid"}); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "unknown producer overlap policy was accepted");

    Arguments args;
    args.member_storage = "raw"; args.max_ram = 5 * mib;
    Backend backend; backend.generation_overlap = true;
    require(backend.admit_gpu_generation(args, mib, 5 * mib, 4 * mib) && !backend.generation_overlap,
        "automatic raw admission rejected a fitting borrowed GPU output");
    require(backend.admit_gpu_generation(args, 0, 5 * mib, 4 * mib) && !backend.generation_overlap,
        "backend reuse re-enabled overlap while borrowed output may be live");
    backend.generation_overlap = true; args.generation_overlap_auto = false;
    require(!backend.admit_gpu_generation(args, mib, 5 * mib, 4 * mib) && backend.generation_overlap,
        "explicit overlap silently changed to borrowed output");
    backend.generation_overlap = false;
    require(backend.admit_gpu_generation(args, mib, 5 * mib, 4 * mib),
        "explicit serial output failed exact borrowed admission");
    backend.generation_overlap = true; args.generation_overlap_auto = true;
    require(!backend.admit_gpu_generation(args, mib, 5 * mib, 4 * mib + 1),
        "borrowed fallback exceeded the RAM budget");
    backend.generation_overlap = true; args.member_storage = "compact";
    require(!backend.admit_gpu_generation(args, mib, 5 * mib, 4 * mib) && backend.generation_overlap,
        "compact generation incorrectly used a raw borrowed-output allowance");
    backend.generation_overlap = true; args.member_storage = "raw";
    require(backend.admit_gpu_generation(args, 0, 5 * mib, 4 * mib) && backend.generation_overlap,
        "exact owning output admission unnecessarily disabled overlap");
    require(backend.admit_gpu_generation(args, 0, 5 * mib, 4 * mib, 2 * mib, 2 * mib) && !backend.generation_overlap,
        "live headroom did not admit a fitting raw borrowed fallback");
    backend.generation_overlap = true;
    require(backend.admit_gpu_generation(args, 0, 5 * mib, 4 * mib, 3 * mib, 2 * mib) && backend.generation_overlap,
        "exact live owning headroom unnecessarily disabled overlap");
    require(!backend.admit_gpu_generation(args, 0, 5 * mib, 4 * mib, 2 * mib - 1, 2 * mib),
        "live borrowed fallback exceeded available headroom");
    backend.generation_overlap = true; args.generation_overlap_auto = false;
    require(!backend.admit_gpu_generation(args, 0, 5 * mib, 4 * mib, 2 * mib, 2 * mib) && backend.generation_overlap,
        "live headroom silently disabled explicit overlap");
    args.max_ram = 8 * mib; args.generation_overlap_auto = true; backend.generation_overlap = true;
    require(backend.admit_gpu_generation(args, 0, 8 * mib, 6 * mib) && backend.generation_overlap,
        "ordinary workspace did not admit the initial owning representation");
    require(backend.admit_gpu_generation(args, 2 * mib, 8 * mib, 6 * mib) && !backend.generation_overlap,
        "increased engine or retained-member workspace did not re-admit borrowed output");
    backend.generation_overlap = true; args.generation_overlap_auto = false;
    require(!backend.admit_gpu_generation(args, 2 * mib, 8 * mib, 6 * mib) && backend.generation_overlap,
        "increased workspace silently disabled explicit overlap");
}

void test_destination_temporary()
{
    using namespace grow_controller_test;
    auto parse = [](std::initializer_list<char const*> options)
    {
        std::vector<char*> command;
        for (auto option : options) command.push_back(const_cast<char*>(option));
        return pos2gpu::parse_group_options(int(command.size()), command.data());
    };
    auto const defaults = parse({"grow", "group.gplot"});
    require(defaults.destination_temp == "off", "destination scratch was enabled by default");
    auto const ssd = parse({"grow", "group.gplot", "--destination-temp", "ssd"});
    auto const hdd = parse({"grow", "group.gplot", "--destination-temp", "hdd"});
    auto const on = parse({"grow", "group.gplot", "--destination-temp", "on",
        "--construction", "external", "--temp-capacity-bytes", "1048576"});
    auto const off = parse({"grow", "group.gplot", "--destination-temp", "on", "--destination-temp", "off"});
    require(ssd.destination_temp == "ssd" && hdd.destination_temp == "hdd" && on.destination_temp == "on" &&
        on.temporary_capacity == mib && off.destination_temp == "off", "destination temporary options changed");
    bool rejected = false;
    try { (void)parse({"grow", "group.gplot", "--destination-temp", "auto"}); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "unknown destination temporary policy was accepted");
    rejected = false;
    try { (void)parse({"grow", "group.gplot", "--destination-temp", "all"}); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "obsolete destination temporary policy spelling was accepted");
    rejected = false;
    try { (void)parse({"grow", "group.gplot", "--temp-capacity-bytes", "1048576"}); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "temporary cap without enabled storage was accepted");

    std::vector<fs::path> roots;
    StorageDevice const fast{"fast-fs", {"nvme"}, true}, slow{"slow-fs", {"hdd"}, false, true},
        unknown{"unknown-fs", {}, false, false}, mixed{"mixed-fs", {"nvme", "hdd"}, false, false};
    add_destination_temporary(roots, defaults.destination_temp, "fast", fast);
    require(roots.empty(), "default policy promoted a known SSD destination");
    roots.emplace_back("explicit");
    for (auto const& [path, device] : std::array<std::pair<fs::path, StorageDevice>, 4>{
            std::pair{fs::path("fast"), fast}, {fs::path("slow"), slow}, {fs::path("unknown"), unknown},
            {fs::path("mixed"), mixed}})
        add_destination_temporary(roots, ssd.destination_temp, path, device);
    require(roots == std::vector<fs::path>{"explicit", "fast"},
        "SSD policy removed explicit storage or included rotating/unknown storage");
    std::vector<fs::path> rotating_roots;
    for (auto const& [path, device] : std::array<std::pair<fs::path, StorageDevice>, 4>{
            std::pair{fs::path("fast"), fast}, {fs::path("slow"), slow}, {fs::path("unknown"), unknown},
            {fs::path("mixed"), mixed}})
        add_destination_temporary(rotating_roots, hdd.destination_temp, path, device);
    require(rotating_roots == std::vector<fs::path>{"slow"},
        "HDD policy included nonrotating, mixed or unclassified storage");
    add_destination_temporary(roots, on.destination_temp, "slow", slow);
    add_destination_temporary(roots, on.destination_temp, "unknown", unknown);
    add_destination_temporary(roots, on.destination_temp, "fast", fast);
    add_destination_temporary(roots, off.destination_temp, "disabled", fast);
    require(roots == std::vector<fs::path>{"explicit", "fast", "slow", "unknown"},
        "on/off destination policy changed stable root selection or retained duplicates");
}

void metadata_options()
{
    auto parse = [](std::string_view action, std::initializer_list<char const*> options)
    {
        std::vector<char*> command;
        for (auto option : options) command.push_back(const_cast<char*>(option));
        return pos2gpu::parse_group_options(int(command.size()), command.data(), action);
    };
    auto const positional = parse("plan", {"plan", "--metadata-only", "saved.plan", "--json"});
    auto const preview = parse("plan", {"plan", "--metadata-only", "--from-plan", "saved.plan"});
    auto const private_job = parse("plan", {"plan", "--metadata-only", "--manifest", "private.plan"});
    require(positional.metadata_only && positional.plan_only && positional.directory_farm && positional.json &&
        positional.from_plan == preview.from_plan && !private_job.manifest.empty() &&
        !positional.automatic_compute && !private_job.farming_profile,
        "metadata inspection did not retain its read-only source scope");
    auto const joint = parse("plan", {"plan", "--assess-jobs", "first.plan", "second.plan", "--json"});
    require(joint.assess_jobs && joint.plan_only && joint.assessment_plans.size() == 2 &&
        joint.target.empty() && joint.destinations.empty() && !joint.automatic_compute && !joint.farming_profile,
        "joint assessment did not retain its read-only source scope");
    auto reject = [&](std::string_view action, std::initializer_list<char const*> options)
    {
        bool rejected = false;
        try { (void)parse(action, options); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, "metadata inspection accepted a live action or override");
    };
    reject("fill", {"fill", "--metadata-only", "saved.plan"});
    reject("grow", {"grow", "--metadata-only", "saved.gplot"});
    reject("plan", {"plan", "--metadata-only"});
    reject("plan", {"plan", "--metadata-only", "saved.plan", "another.plan"});
    reject("plan", {"plan", "--metadata-only", "--manifest", "private.plan", "saved.plan"});
    reject("plan", {"plan", "--metadata-only", "saved.plan", "--save-plan", "replacement.plan"});
    reject("plan", {"plan", "--metadata-only", "saved.plan", "--harvester", "local"});
    reject("plan", {"plan", "--metadata-only", "saved.plan", "--paths-from", "unread-list"});
    reject("plan", {"plan", "--metadata-only", "saved.plan", "--farming-cores", "1"});
    reject("fill", {"fill", "--assess-jobs", "saved.plan"});
    reject("plan", {"plan", "--assess-jobs"});
    reject("plan", {"plan", "--assess-jobs", "saved.plan", "--metadata-only"});
    reject("plan", {"plan", "--assess-jobs", "saved.plan", "--save-plan", "replacement.plan"});
    reject("plan", {"plan", "--assess-jobs", "saved.plan", "--harvester", "local"});
    reject("plan", {"plan", "--assess-jobs", "saved.plan", "--calibrate"});
    reject("fill", {"fill", "unread.plot2", "--replot-proof-solves", "1"});
    reject("plan", {"plan", "--calibrate", "unread.plot2", "--replot-proof-solves", "1"});
#ifndef XCHPLOT2_GROUP_COMMANDS
    reject("plan", {"plan", "--calibrate", "not-read.gplot", "--json"});
    reject("plan", {"plan", "--calibrate", "--calibration-worker", "not-read.gplot", "--json"});
#endif
}

void temporary_admission(std::filesystem::path const& root, group_grow::Identity const& initial)
{
    using namespace grow_controller_test;
    auto directory = group_grow::ScratchRoot::inspect(root);
    directory.available_bytes = 128 * mib;
    group_grow::ScratchPool pool(std::vector{directory}, 0);
    require(pool.try_reserve_output(0, directory, directory.available_bytes), "joint output reservation failed");
    pool.seal_recovery();
    Arguments args; args.max_ram = 256 * mib; args.workers = 2; args.window = 65536;
    auto config = configuration(args, directory.available_bytes, mib);
    auto const grant = temporary_grant(pool, 0, root, identity_from(initial), 64, 1, config, false);
    require(grant && config.capacity_bytes < directory.available_bytes,
        "shared filesystem could not jointly admit final output and temporary contributions");
    uint64_t scratch = 0; for (auto const& part : grant->scratch) scratch += part.capacity_bytes;
    require(config.capacity_bytes + scratch <= directory.available_bytes,
        "joint output and scratch admission exceeded the shared filesystem");
    pool.release_scratch_after_cleanup(0);
    require(pool.try_reserve_output(0, root, directory.available_bytes),
        "ordinary handoff could not restore the original capacity after scratch cleanup");
    pool.cancel(0);
}

void replot_external_projection(fs::path const& directory, group_grow::Identity const& initial)
{
    using namespace grow_controller_test;
    Arguments args; args.max_ram = 256 * mib; args.workers = 2; args.batch = 4;
    args.construction = "external"; args.window = 65536; args.headroom = 0;
    auto const requested = identity_from(initial); auto target = group_grow::ScratchRoot::inspect(directory);
    target.available_bytes = 2 * 1024 * mib;
    auto const capacity = sum(canonical_bound(requested, 4), args.max_ram);
    require(replot_external_capacity(args, requested, 4, capacity, directory / "projection.gplot", {target}, target, 0),
        "ordinary scratch policy could not admit a projected external migration");
    auto empty = target; empty.available_bytes = 0;
    require(!replot_external_capacity(args, requested, 4, capacity, directory / "projection.gplot", {empty}, empty, 0),
        "external migration admitted scratch/output without actual or reviewed prospective space");
    auto missing_output = target; missing_output.available_bytes = 0;
    require(!replot_external_capacity(args, requested, 4, capacity, directory / "projection.gplot", {target}, missing_output, 0),
        "scratch alias multiplied projected filesystem space");
    auto changed = target; ++changed.directory_inode;
    bool refused = false;
    try { (void)replot_external_capacity(args, requested, 4, capacity, directory / "projection.gplot", {changed}, changed, 0); }
    catch (std::runtime_error const&) { refused = true; }
    require(refused, "projected scratch admission accepted a changed native directory");
    auto capped = args; capped.temporary_capacity = 1;
    require(!replot_external_capacity(capped, requested, 4, capacity, directory / "projection.gplot", {target}, target, 0),
        "projected external migration ignored the ordinary scratch cap");
}

// Exercise actual controller lifecycle with synthetic known drive identities;
// production topology discovery has no test override.
void test_construction_history(std::filesystem::path const& root, group_grow::Members members,
    group_grow::Identity const& initial, group_grow::Config config)
{
    using namespace grow_controller_test;
    Arguments args;
    args.backend = "cpu"; args.member_storage = "raw";
    args.max_ram = 536870912; args.batch = 1; args.workers = 2;
    args.window = config.window_bytes = mib;
    args.checkpoint_bytes = config.checkpoint_bytes = 16 * mib;
    config.capacity_bytes = 256 * mib; config.memory_limit_bytes = args.max_ram - mib;
    group_grow::ConstructionHistories histories(4);
    group_grow::ConstructionContext context{{"history-test-drive"}, initial.header.k, initial.header.strength, initial.header.meta_group,
        initial.memo.size(), args.max_ram, args.window, args.checkpoint_bytes, config.codec_bytes,
        config.workers, false, config.io_overlap, config.release_cache};
    auto source = histories.find(context);
    Backend backend;
    auto const requested = identity_from(initial);
    auto select = [&](Construction& engine, Arguments const& options)
    {
        uint64_t const reserved = engine.memory_bytes(1, false), fixed = sum(reserved, mib);
        Producer producer{requested, nullptr, backend, options, options.max_ram, fixed, fixed};
        return automatic_selection(engine, producer, 64, 1, reserved, 0, 0, true);
    };
    Construction first((root / "history-first").string(), config, false, initial, "auto");
    first.attach_history(source); first.confirm_identity();
    require(!select(first, args) && !first.can_select(), "first group bypassed conservative initial construction");
    require(first.append(members.subspan(0, 1)).accepted, "first history append did not fit");
    first.select_merged();
    for (size_t i = 1; i < members.size(); ++i)
        require(first.append(members.subspan(i, 1)).accepted, "history contribution did not fit");
    auto const published = first.finish();
    auto const completed = source->snapshot();
    require(completed.count && completed.contributions.count && completed.packing.count &&
        completed.layout.count && completed.shapes.count,
        "completed construction omitted append, packing, layout or canonical evidence");

    // Retain the real completed work/call counts and canonical shape above.
    // Like construction_selection(), make the policy decision independent of
    // scheduler, filesystem-cache and sanitizer timing: unit byte/codec rates,
    // no sync delay. The subsequent handoffs and publication still perform I/O.
    {
        std::lock_guard lock(source->mutex);
        auto timing = [](group_grow::ConstructionSamples& samples)
        {
            for (size_t i = 0; i < samples.count; ++i)
            {
                auto& sample = samples.samples[i];
                sample.read_ns = sample.read_bytes; sample.write_ns = sample.written_bytes;
                sample.sync_ns = sample.decode_ns = 0;
                sample.codec_ns = uint64_t(std::ceil(sample.codec_bytes / std::max(1u, sample.workers)));
                sample.elapsed_ns = sample.read_ns + sample.write_ns + sample.codec_ns;
            }
        };
        timing(source->history); timing(source->history.contributions);
        timing(source->history.packing); timing(source->history.layout);
    }

    Construction next((root / "history-next").string(), config, false, initial, "auto");
    next.attach_history(source); next.confirm_identity();
    auto const selected = select(next, args);
    require(selected && selected->merged && next.can_select(),
        "a later compatible group could not select merged construction before its first append");
    auto explicit_args = args; explicit_args.window_set = explicit_args.checkpoint_set = true;
    if (auto constrained = select(next, explicit_args))
        require(constrained->config.window_bytes == config.window_bytes &&
            constrained->config.checkpoint_bytes == config.checkpoint_bytes,
            "inherited observations overrode explicit I/O constraints");
    next.retune(selected->config); next.select_merged();
    require(next.append(members).accepted, "empty merged handoff lost exact append admission");
    auto const inherited = next.finish();
    require(inherited.final_bytes == published.final_bytes && inherited.members == published.members,
        "inherited construction changed the published group");
    auto contents = [](fs::path const& path)
    { auto in = input(path.string()); return std::vector<uint8_t>{std::istreambuf_iterator<char>(in), {}}; };
    require(contents(root / "history-first") == contents(root / "history-next"),
        "inherited construction changed ordinary v2 bytes");

    auto const limited_path = root / "history-capacity";
    {
        Construction bounded(limited_path.string(), config, false, initial);
        bounded.attach_history(source); bounded.confirm_identity();
        require(bounded.append(members).accepted, "capacity history fixture did not fit");
        bounded.finish();
    }
    auto tight = config; tight.capacity_bytes = fs::file_size(limited_path);
    {
        Construction bounded(limited_path.string(), tight, true, initial);
        bounded.attach_history(source); bounded.confirm_identity();
        auto const before = source->snapshot();
        auto const refused = bounded.append(members.subspan(0, 1));
        auto const after = source->snapshot();
        require(!refused.accepted && after.count == before.count &&
            after.contributions.count == before.contributions.count &&
            bounded.finish().members == members.size() && contents(limited_path) == contents(root / "history-first"),
            "capacity rejection trained a successful batch or changed the completed prefix");
    }

    auto narrow = config; narrow.capacity_bytes = 4 * mib;
    {
        auto const path = root / "history-retired-capacity";
        fs::copy_file(limited_path, path);
        uint64_t const metadata = 4096;
        bool rejected = false;
        try
        {
            Construction bounded(path.string(), tight, true, initial, "incremental", {}, metadata);
            bounded.confirm_identity();
        }
        catch (std::runtime_error const&) { rejected = true; }
        require(rejected && contents(path) == contents(limited_path),
            "retired metadata bypassed capacity admission or changed a rejected output");
        auto admitted = tight; admitted.capacity_bytes += metadata;
        Construction bounded(path.string(), admitted, true, initial, "incremental", {}, metadata);
        bounded.confirm_identity();
        auto const finished = bounded.finish();
        require(finished.peak_file_bytes == admitted.capacity_bytes && finished.workspace_bytes == metadata,
            "retired metadata disappeared from idle construction accounting");
        bounded.resize_capacity(config.capacity_bytes + metadata);
        auto const appended = bounded.append(members.subspan(0, 1));
        require(appended.accepted && appended.result.workspace_bytes >= metadata &&
            appended.result.peak_file_bytes <= config.capacity_bytes + metadata &&
            bounded.config.capacity_bytes == config.capacity_bytes,
            "capacity expansion reused space occupied by retired metadata");
    }
    {
        Construction fallback((root / "history-empty-fallback").string(), narrow, false, initial, "auto");
        fallback.attach_history(source); fallback.confirm_identity(); fallback.select_merged();
        auto const before = source->snapshot();
        auto const accepted = fallback.append(members.subspan(0, 1));
        auto const after = source->snapshot();
        require(accepted.accepted && !fallback.is_staging() && fallback.automatic_fallbacks == 1 &&
            after.contributions.count == before.contributions.count && fallback.finish().members == 1,
            "empty inherited construction could not fall back to exact ordinary admission");
    }

    // Saved zero-byte owners keep their inode when closed. Both handoffs
    // must accept that same inode without opening an unrelated replacement.
    auto const retained_path = root / "history-retained-empty";
    { std::ofstream file(retained_path); }
    Construction retained(retained_path.string(), narrow, true, initial, "auto");
    retained.attach_history(source); retained.confirm_identity(); retained.select_merged();
    require(retained.append(members.subspan(0, 1)).accepted && !retained.is_staging() &&
        retained.automatic_fallbacks == 1 && retained.finish().members == 1,
        "retained empty construction could not hand off or fall back through its saved inode");

#ifdef GROW_TEST_WRAP_PWRITE
    // Inject the replacement exactly after the old owner's unlink, for both
    // minimum-reserve and contribution-admission rejection of an empty merge.
    for (uint64_t capacity : {2 * config.window_bytes, 4 * config.window_bytes})
    {
        auto const path = root / ("history-fallback-replacement-" + std::to_string(capacity));
        auto cfg = config; cfg.capacity_bytes = capacity;
        Construction replaced(path.string(), cfg, false, initial, "auto");
        replaced.attach_history(source); replaced.confirm_identity(); replaced.select_merged();
        replacement_target = path.string(); replace_unlinked.store(true);
        bool rejected = false;
        try { replaced.append(members.subspan(0, 1)); } catch (std::exception const&) { rejected = true; }
        bool const injected = !replace_unlinked.exchange(false);
        std::string const expected = "replacement";
        require(rejected && injected && contents(path) == std::vector<uint8_t>(expected.begin(), expected.end()),
            "empty merge fallback adopted or removed a replacement after close");
    }
#endif

    // A replacement path survives closing the old empty inode, so the fresh
    // merge open must fail exclusively without adopting or deleting that file.
    auto const conflict = root / "history-conflict";
    Construction replaced(conflict.string(), config, false, initial, "auto");
    replaced.attach_history(source); replaced.confirm_identity();
    fs::rename(conflict, root / "history-original-empty");
    { std::ofstream file(conflict); file << "replacement"; }
    auto const replacement = contents(conflict);
    bool rejected = false;
    try { replaced.select_merged(); } catch (std::exception const&) { rejected = true; }
    require(rejected && contents(conflict) == replacement,
        "empty construction handoff adopted or removed a replacement target");

    // The normal same-drive writer serializes appends while an idle destination
    // can inspect shared evidence. Histories therefore need their own lock.
    context.drives = {"history-live-drive"}; auto live_source = histories.find(context);
    Construction live_first((root / "history-live-first").string(), config, false, initial);
    Construction live_next((root / "history-live-next").string(), config, false, initial);
    live_first.attach_history(live_source); live_next.attach_history(live_source);
    live_first.confirm_identity(); live_next.confirm_identity();
    group_grow::OrderedWriter writer([] {});
    auto writing = writer.submit([&] { require(live_first.append(members).accepted, "live history append did not fit"); });
    while (writing.wait_for(std::chrono::seconds(0)) != std::future_status::ready) live_next.refresh_history();
    writing.get();
    auto following = writer.submit([&] { require(live_next.append(members).accepted, "second live history append did not fit"); });
    following.get();
    require(live_source->snapshot().count == 2, "same-drive destinations lost completed writer observations");

#ifdef GROW_TEST_WRAP_PWRITE
    context.drives = {"history-failed-drive"}; auto failed_source = histories.find(context);
    auto const failed_path = root / "history-failed";
    {
        Construction failed(failed_path.string(), config, false, initial, "merged");
        failed.attach_history(failed_source); failed.confirm_identity();
        require(failed.append(members).accepted, "failed-finalization fixture did not fit");
        auto const before = failed_source->snapshot();
        fail_write_begin.store(0); fail_write_end.store(UINT64_MAX); fail_write.store(true);
        bool failed_finish = false;
        try { failed.finish(); } catch (std::exception const&) { failed_finish = true; }
        fail_write.store(false);
        auto const after = failed_source->snapshot();
        require(failed_finish && !after.packing.count && !after.layout.count && !after.shapes.count &&
            after.contributions.count == before.contributions.count,
            "failed finalization trained successful construction evidence");
    }
    {
        Construction resumed(failed_path.string(), config, true, initial, "incremental");
        resumed.attach_history(failed_source); resumed.confirm_identity();
        require(resumed.is_staging(), "requested construction overrode the saved recovery protocol");
        auto const recovered = resumed.finish(); auto const after = failed_source->snapshot();
        require(recovered.members == members.size() && !after.packing.count && !after.layout.count && !after.shapes.count &&
            contents(failed_path) == contents(root / "history-first"),
            "recovery trained incomplete phase timings or changed canonical output");
    }
#endif
}

Identity identity()
{
    Identity result;
    result.header.magic = PlotGroupFile::MAGIC; result.header.version = PlotGroupFile::FORMAT_VERSION;
    result.header.group_id = PlotGroupId(std::string(64, 'a')); result.header.k = 18; result.header.strength = 2;
    result.memo = {0, 17, 255}; result.header.memo_length = uint8_t(result.memo.size());
    result.header.chunk_index_offset = sizeof(Header) + result.memo.size(); return result;
}
Config configuration()
{
    Config result;
    result.capacity_bytes = 64ull * 1024 * 1024; result.window_bytes = 32768;
    result.checkpoint_bytes = 65536; result.codec_bytes = 8ull * 1024 * 1024;
    result.workers = 2; result.io_overlap = false;
    result.memory_limit_bytes = std::max(Engine::memory_bytes(18, result, 16), MergeEngine::memory_bytes(18, result, 16));
    return result;
}
std::vector<std::vector<uint64_t>> members(size_t count = 8, uint64_t symbols = 64)
{
    std::vector<std::vector<uint64_t>> result(count);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18), range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t member = 0; member < count; ++member)
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
            for (uint64_t i = 0; i < symbols; ++i)
            {
                uint64_t const value = chunk * range + i * (range / symbols) + member + 7;
                result[member].push_back(value); if (i == 3) result[member].push_back(value);
            }
    return result;
}
std::vector<uint8_t> contents(fs::path const& path)
{ auto in = group_tools::input(path.string()); return {std::istreambuf_iterator<char>(in), {}}; }
void direct(fs::path const& path, Members input, Config cfg)
{
    Engine output(path.string(), cfg, false, identity()); output.confirm_identity();
    require(output.append(input).accepted, "direct merge test reference did not fit");
}
Result complete(fs::path const& path, Members input, bool resume, Config cfg, size_t batch = 1)
{
    if (resume && fs::file_size(path) >= sizeof(Header) && !MergeEngine::pending(path.string()))
    {
        auto in = group_tools::input(path.string()); auto header = group_tools::read<Header>(in);
        if (header.version == 0xf1)
        {
            Engine output(path.string(), cfg, true); output.confirm_identity(); output.resume({});
        }
    }
    MergeEngine output(path.string(), cfg, resume, identity()); output.confirm_identity();
    if (!output.can_append()) return output.finish();
    while (output.committed_member_count() < input.size())
    {
        size_t const first = output.committed_member_count(), count = std::min(batch, input.size() - first);
        auto result = output.append(input.subspan(first, count));
        require(result.accepted, "merged test contribution rejected: " + result.reason);
    }
    return output.finish();
}
void parity(fs::path const& root, Members input, Config cfg)
{
    auto reference = root / "reference.gplot"; direct(reference, input, cfg); auto expected = contents(reference);
    for (size_t batch : {size_t(1), size_t(3), input.size()})
    {
        auto path = root / ("merged-" + std::to_string(batch));
        auto result = complete(path, input, false, cfg, batch);
        require(result.final_bytes == expected.size() && result.peak_file_bytes <= cfg.capacity_bytes && contents(path) == expected,
                "balanced merge differs from canonical bytes");
        auto const before = fs::last_write_time(path); complete(path, input, true, cfg);
        require(fs::last_write_time(path) == before && contents(path) == expected, "completed merge resume changed output");
        fs::remove(path);
    }
    auto extended = root / "extended.gplot"; direct(extended, input.subspan(0, 1), cfg);
    complete(extended, input, true, cfg); require(contents(extended) == expected, "ordinary merge extension differs"); fs::remove(extended);
    auto staged = root / "bounded.gplot";
    {
        MergeEngine output(staged.string(), cfg, false, identity()); output.confirm_identity();
        for (size_t member = 0; member < input.size(); ++member)
            require(output.append(input.subspan(member, 1)).accepted, "bounded merge did not fit");
        auto result = output.finish(); auto const& stats = output.stats();
        uint64_t const passes = (input.size() + cfg.merge_inputs - 3) / (cfg.merge_inputs - 1);
        require(stats.merge_passes < passes && stats.reclaimed_bytes > result.final_bytes &&
            stats.max_active_runs <= MergeEngine::packing_inputs(cfg) && stats.read_bytes && stats.written_bytes && stats.sync_ns,
            "merge did not perform bounded, measured space recycling");
        if (input.size() <= MergeEngine::packing_inputs(cfg))
            require(!stats.merge_passes && stats.max_active_runs == input.size(), "final packing materialized an unnecessary run");
        auto const& packing = stats.phases[size_t(WorkPhase::pack)];
        auto const& layout = stats.phases[size_t(WorkPhase::layout)];
        require(packing.read_bytes && packing.written_bytes && packing.read_calls && packing.write_calls && packing.syncs &&
            layout.written_bytes && layout.write_calls && layout.syncs && !layout.codec_ns,
            "completed finalization phases did not retain their own storage work");
        if (cfg.io_overlap)
            require(stats.contribution_workers > 1 && stats.packing_overlap_windows && stats.layout_batches < stats.layout_moves,
                    "merge did not parallelize contributions, overlap packing and batch placement");
        // Retained-run slack can exceed this small output's size with the
        // larger overlap transaction. capacity_pressure() checks that case
        // below the duplicate-output bound with forced compaction instead.
        if (!cfg.io_overlap)
            require(result.peak_file_bytes < result.final_bytes * 2, "merge retained a full duplicate output");
    }
    require(contents(staged) == expected, "recycling changed final output"); fs::remove(staged);
}
void retuning(fs::path const& root, Members input, Config cfg)
{
    auto path = root / "retuning.gplot"; direct(path, input.subspan(0, 1), cfg);
    auto const before = contents(path);
    Engine output(path.string(), cfg, true, identity()); output.confirm_identity();
    output.retune(output.largest_chunk(), cfg.checkpoint_bytes);
    bool rejected = false;
    try { (void)output.append(input.subspan(1)); }
    catch (WindowBudgetError const&) { rejected = true; }
    require(rejected && contents(path) == before && !output.identity().pending,
            "automatic window sizing failure changed the ordinary source");
    output.retune(cfg.window_bytes, cfg.checkpoint_bytes);
    require(output.append(input.subspan(1)).accepted && contents(path) == contents(root / "reference.gplot"),
            "restoring the reserved window changed canonical bytes");
}
void ordinary_prefix(fs::path const& root, Config cfg)
{
    // A bounded reader set must compact the new suffix rather than repeatedly
    // copying its larger ordinary prefix. Vary the last contribution sizes.
    auto input = members(12); auto path = root / "large-prefix.gplot";
    auto reference = root / "large-prefix-reference.gplot";
    direct(reference, input, cfg); direct(path, Members(input).subspan(0, 8), cfg);
    {
        MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity();
        for (size_t count : {size_t(1), size_t(2), size_t(1)})
            require(output.append(Members(input).subspan(output.committed_member_count(), count)).accepted,
                    "small ordinary-prefix contribution did not fit");
        require(output.stats().max_active_runs <= MergeEngine::packing_inputs(cfg), "prefix compaction exceeded reader budget");
    }
    {
        MergeState saved(path.string(), cfg, true, identity());
        require(saved.state.runs[0].ordinary && saved.state.runs[0].count == 8,
                "suffix compaction unnecessarily rewrote the ordinary prefix");
    }
    complete(path, input, true, cfg);
    require(contents(path) == contents(reference), "partial/changing suffix batches changed canonical bytes");
    fs::remove(path); fs::remove(reference);
}
void balanced_traffic(fs::path const& root, Config cfg)
{
    // Borrow the same valid member payload for many contributions, keeping
    // resident memory small while exposing repeated unbalanced suffix copies.
    auto source = members(1, 96); std::vector<Member> input;
    for (size_t member = 0; member < 128; ++member) input.push_back(Member::borrowed(source[0]));
    auto reference = root / "traffic-reference.gplot";
    cfg.capacity_bytes = 256 * 1024 * 1024; cfg.window_bytes = 65536;
    cfg.memory_limit_bytes = std::max(Engine::memory_bytes(18, cfg, uint16_t(input.size())),
        MergeEngine::memory_bytes(18, cfg, uint16_t(input.size())));
    direct(reference, input, cfg);
    for (unsigned fan : {2u, 4u})
    {
        cfg.merge_inputs = fan; auto path = root / ("traffic-" + std::to_string(fan)); uint64_t read_bytes = 0;
        {
            MergeEngine output(path.string(), cfg, false, identity()); output.confirm_identity();
            for (size_t member = 0; member < input.size(); ++member)
                require(output.append(Members(input).subspan(member, 1)).accepted, "balanced traffic contribution did not fit");
            read_bytes = output.stats().read_bytes;
        }
        {
            MergeState saved(path.string(), cfg, true, identity());
            uint64_t const tables = input.size() * PlotGroupFile::getChunkCountForK(18) * (sizeof(ChunkSummary) + sizeof(uint64_t));
            unsigned const bits = std::bit_width(input.size() - 1);
            unsigned const levels = fan == 4 ? (bits + 1) / 2 : bits;
            require(saved.state.run_count <= MergeState::max_runs &&
                read_bytes <= saved.state.live_bytes * levels + tables,
                "deferred runs repeatedly rewrote an unbalanced suffix");
        }
        complete(path, input, true, cfg);
        require(contents(path) == contents(reference), "balanced deferred traffic changed canonical bytes"); fs::remove(path);
    }
    fs::remove(reference);
}
void interruption(fs::path const& root, Members input, Config cfg, std::string const& point, bool extend = false)
{
    auto expected = contents(root / "reference.gplot");
    auto path = root / "interrupted.gplot"; fs::remove(path);
    if (extend) direct(path, input.subspan(0, 1), cfg);
    auto const child = ::fork(); require(child >= 0, "cannot fork merge recovery test");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", point.c_str(), 1);
        try { complete(path, input, extend, cfg); }
        catch (std::exception const& error) { std::cerr << point << ": " << error.what() << '\n'; std::_Exit(1); }
        std::_Exit(0);
    }
    int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "merge failpoint was not reached: " + point);
    auto resumed = cfg;
    // Changing concurrency or raising the transaction request must not
    // invalidate the reserve committed by the interrupted writer.
    if (point == "merge-pack-committed:2") { resumed.workers = 1; resumed.checkpoint_bytes *= 8; }
    if (point == "merge-copy-committed:2") resumed.merge_inputs = cfg.merge_inputs == 4 ? 2 : 4;
    if (point == "merge-contribution-partial" || point == "merge-contribution-flushed")
    {
        MergeEngine output(path.string(), resumed, true, identity());
        require(output.committed_member_count() == (extend ? 1 : 0),
                "provisional contribution survived without its batch checkpoint");
    }
    complete(path, input, true, resumed);
    require(contents(path) == expected, "merge crash recovery differs: " + point); fs::remove(path);
}
void recovery(fs::path const& root, Members input, Config cfg)
{
    std::vector<std::string> points{
        "merge-allocated", "merge-bootstrap", "merge-locator-partial", "merge-initialized-committed",
        "merge-checkpoint-partial:2", "merge-run-flushed:2", "merge-run-committed:3",
        "merge-finishing-committed", "merge-pack-start-committed", "merge-pack-flushed:2", "merge-pack-committed:2",
        "merge-index-flushed", "merge-layout-ready-committed", "merge-layout-located",
        "merge-layout-partial:3", "merge-layout-move-flushed:2", "merge-layout-move-committed:3",
        "merge-layout-complete-committed", "merge-final-header-partial", "merge-finalizing",
        "merge-retire-partial", "merge-retired", "merge-truncated", "merge-published"};
    for (auto const& point : points) interruption(root, input, cfg, point);
    for (auto const* point : {"merge-locator-partial", "merge-initialized-committed",
                              "merge-pack-committed:2", "merge-index-flushed", "merge-layout-partial", "merge-retired"})
        interruption(root, input, cfg, point, true);
    // A final group within the reader bound needs no packet-copy pass. Force
    // input pressure separately so both saved merge fan-ins still recover.
    auto more = members(MergeEngine::packing_inputs(cfg) + cfg.merge_inputs);
    auto const pressure = root / "pressure"; fs::create_directory(pressure);
    direct(pressure / "reference.gplot", more, cfg);
    for (auto const* point : {"merge-start-committed", "merge-copy-flushed:2", "merge-copy-committed:2", "merge-complete-committed"})
        interruption(pressure, more, cfg, point);
    interruption(pressure, more, cfg, "merge-copy-committed:2", true);
    interruption(pressure, more, cfg, "merge-run-committed:" + std::to_string(more.size()));
}
void capacity_pressure(fs::path const& root, Members input, Config cfg)
{
    cfg.capacity_bytes = 9 * 1024 * 1024;
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, 16);
    auto const path = root / "pressure-capacity.gplot";
    {
        MergeEngine output(path.string(), cfg, false, identity()); output.confirm_identity();
        for (size_t member = 0; member < input.size(); ++member)
            require(output.append(input.subspan(member, 1)).accepted, "capacity-pressure contribution did not fit");
        require(output.stats().merge_passes && output.stats().max_active_runs <= MergeEngine::packing_inputs(cfg),
                "capacity pressure did not compact the bounded run set");
        auto const result = output.finish();
        require(result.peak_file_bytes <= cfg.capacity_bytes && cfg.capacity_bytes < result.final_bytes * 2 &&
            output.stats().reclaimed_bytes > result.final_bytes, "capacity compaction retained duplicate output space");
    }
    require(contents(path) == contents(root / "reference.gplot"), "capacity compaction changed canonical bytes");
    fs::remove(path);
    interruption(root, input, cfg, "merge-copy-committed:2");
}

std::pair<uint64_t, uint64_t> layout_fixture(fs::path const& path, Config cfg,
    std::span<uint8_t const> expected, unsigned order, uint64_t block_bytes = 0, uint64_t reserve_bytes = 0)
{
    MergeState arena(path.string(), cfg, false, identity()); arena.confirm();
    auto record = arena.layout(block_bytes ? block_bytes : MergeEngine::block_bytes(cfg));
    arena.initialize(record, arena.required_bytes(record, grow_add(expected.size(), reserve_bytes)));
    auto& state = arena.state;
    std::memcpy(&state.target, expected.data(), sizeof(Header)); state.members = state.target.group_size;
    state.finishing = true; state.final_bytes = expected.size(); state.phase = MergeState::Phase::layout;
    state.layout_blocks = uint32_t(MergeState::align(expected.size(), state.block_bytes) / state.block_bytes);
    uint32_t const count = state.layout_blocks, tail = state.block_limit - count;
    require(tail >= 2 && count > tail + 3 && expected.size() % state.block_bytes,
            "layout fixture needs spare blocks and a partial final block");
    std::fill(arena.map.begin(), arena.map.end(), MergeState::free_block);
    for (uint32_t i = 0; i < count; ++i) arena.map[i] = i;
    uint64_t moves = count, bytes = expected.size();
    if (order < 2)
    {
        for (uint32_t i = 0; i < count; ++i) arena.map[i] = i ? i - 1 : order ? count : count - 1;
        if (!order) { ++moves; bytes += state.block_bytes; }
    }
    else if (order == 2)
    {
        // A placed prefix, a cycle, and open chains whose sources occupy
        // every tail slot. The only initial vacancies are final positions.
        state.layout_done = 2;
        for (uint32_t i = 2; i < count - tail; ++i) arena.map[i] = i == 2 ? count - tail - 1 : i - 1;
        for (uint32_t i = count - tail; i < count; ++i) arena.map[i] = count + i - (count - tail);
        moves = count - 1; bytes -= state.block_bytes;
    }
    else if (order == 3)
    {
        for (uint32_t i = 0; i < tail; ++i) arena.map[i] = count + i;
        moves = tail; bytes = uint64_t(tail) * state.block_bytes;
    }
    else if (order == 4)
    {
        // Independent long chains must advance together rather than syncing
        // each dependent step of one chain before starting the next chain.
        for (uint32_t i = 0; i < count; ++i) arena.map[i] = i + tail;
    }
    else
    {
        // Several cycles can share existing durable tail spares. Include
        // logical block zero and leave the short final block in a cycle.
        for (uint32_t i = 0; i + 1 < count; i += 2)
        { arena.map[i] = i + 1; arena.map[i + 1] = i; }
        if (count % 2)
        {
            arena.map[count - 3] = count - 2; arena.map[count - 2] = count - 1; arena.map[count - 1] = count - 3;
        }
        moves += count / 2;
        bytes += uint64_t(count / 2) * state.block_bytes;
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        uint64_t const offset = uint64_t(i) * state.block_bytes;
        auto block = expected.subspan(size_t(offset), size_t(std::min(state.block_bytes, expected.size() - offset)));
        arena.write(uint64_t(arena.map[i]) * state.block_bytes, block);
        if (!i)
        {
            Header marker = state.target; marker.version = MergeState::construction_version; marker.chunk_index_offset = state.anchor;
            arena.value(uint64_t(arena.map[i]) * state.block_bytes, marker);
        }
    }
    arena.relocate_checkpoint(); arena.validate(); return {moves, bytes};
}
void layout(fs::path const& root, Members input, Config cfg)
{
    auto reference = root / "layout-reference.gplot"; direct(reference, input.subspan(0, 1), cfg);
    auto expected = contents(reference); fs::remove(reference);
    auto path = root / "layout.gplot";
    cfg.capacity_bytes = cfg.window_bytes * 64;
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, 16);
    require(MergeEngine::block_bytes(cfg) < cfg.checkpoint_bytes,
            "layout fixture must exercise the existing smaller-block policy");
    for (unsigned order : {0u, 1u, 2u, 3u, 4u, 5u})
    {
        auto const [moves, bytes] = layout_fixture(path, cfg, expected, order);
        {
            MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity(); output.finish();
            require(output.stats().layout_moves == moves && output.stats().layout_bytes == bytes,
                    "layout did not follow displacement chains and cycles");
            if (order == 3) require(output.stats().layout_batches < moves, "independent layout moves were not batched");
            if (order >= 4) require(output.stats().layout_batches <= (moves + 1) / 2,
                    "independent chains or cycles did not share durable checkpoints");
        }
        require(contents(path) == expected, "permuted layout changed canonical bytes"); fs::remove(path);
        for (auto const* point : {"merge-layout-partial", "merge-layout-written", "merge-layout-partial:3",
            "merge-layout-written:3", "merge-layout-move-flushed:2", "merge-checkpoint-partial:2",
            "merge-layout-move-committed:2"})
        {
            if (order == 3) break;
            layout_fixture(path, cfg, expected, order);
            auto const child = ::fork(); require(child >= 0, "cannot fork layout recovery test");
            if (!child)
            {
                ::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1);
                try { complete(path, input.subspan(0, 1), true, cfg); }
                catch (std::exception const& error) { std::cerr << point << ": " << error.what() << '\n'; std::_Exit(1); }
                std::_Exit(0);
            }
            int status = 0;
            require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
                    "layout failpoint was not reached: " + std::string(point));
            complete(path, input.subspan(0, 1), true, cfg);
            require(contents(path) == expected, "permuted layout recovery differs: " + std::string(point)); fs::remove(path);
        }
    }
}
void checkpoint_geometry(fs::path const& root, Members input, Config cfg)
{
    auto reference = root / "checkpoint-reference.gplot"; direct(reference, input, cfg);
    auto const expected = contents(reference); fs::remove(reference);
    auto const path = root / "checkpoint.gplot";
    uint64_t const reserve = 1024 * 1024;
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, uint16_t(input.size()));
    for (uint64_t block : {uint64_t(65536), uint64_t(131072), uint64_t(262144)})
    {
        for (unsigned order : {1u, 4u})
        {
            uint64_t previous = 0;
            for (uint64_t transaction : {uint64_t(262144), uint64_t(1048576)})
            {
                cfg.checkpoint_bytes = transaction;
                auto const [moves, bytes] = layout_fixture(path, cfg, expected, order, block, reserve);
                uint64_t batches = 0;
                {
                    MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity(); output.finish();
                    require(output.stats().layout_moves == moves && output.stats().layout_bytes == bytes && bytes == expected.size(),
                            "checkpoint geometry changed layout copy bytes");
                    batches = output.stats().layout_batches;
                    require(order != 1 || batches == moves, "dependent layout chain reused an uncommitted source");
                }
                require(contents(path) == expected, "checkpoint geometry changed canonical bytes"); fs::remove(path);
                if (previous && order == 4) require(batches < previous, "larger transaction did not batch independent chains");
                std::cout << "layout geometry block=" << block << " transaction=" << transaction << " dependent=" << (order == 1)
                          << " bytes=" << bytes << " moves=" << moves << " batches=" << batches << '\n';
                previous = batches;
            }
        }
    }
    // A saved layout may have a different block geometry from new builds.
    // Raising the request must preserve both its blocks and replay reserve.
    cfg.checkpoint_bytes = 262144;
    auto const [moves, bytes] = layout_fixture(path, cfg, expected, 4, 65536, reserve);
    uint64_t batches = 0;
    {
        MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity(); output.finish();
        batches = output.stats().layout_batches;
    }
    require(contents(path) == expected, "saved checkpoint geometry changed canonical bytes"); fs::remove(path);
    layout_fixture(path, cfg, expected, 4, 65536, reserve);
    cfg.checkpoint_bytes = 1048576;
    {
        MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity(); output.finish();
        require(output.stats().layout_batches == batches && output.stats().layout_moves == moves && output.stats().layout_bytes == bytes,
                "larger resume request exceeded the saved transaction reserve");
    }
    require(contents(path) == expected, "larger resume request changed canonical bytes"); fs::remove(path);
}
void placement_costs()
{
    require(merge_placement_bytes(4096, 8192, 8200, 0) == 8 &&
        merge_placement_bytes(4096, 8192, 8200, 9000) == 808 &&
        merge_placement_bytes(4096, 8192, 8200, 12288) == 4096 &&
        merge_placement_bytes(4096, 8192, 9300, 9000) == 1108,
            "placement credited bytes beyond its known prefix or proven terminal bound");
    Stats stats; stats.read_bytes = stats.written_bytes = 100;
    stats.read_ns = stats.write_ns = 100; stats.syncs = 2; stats.sync_ns = 20;
    require(prefer_merge_placement(stats, 100, 30, 1), "profitable placement checkpoint was rejected");
    require(!prefer_merge_placement(stats, 100, 200, 1), "placement omitted its checkpoint slot write");
    require(!prefer_merge_placement(stats, 10, 30, 1), "partial output received a full-block copy saving");
    require(!prefer_merge_placement(stats, 100, 30, 180), "placement ignored measured map/hash checkpoint cost");
    require(!prefer_merge_placement(stats, 100, 30, 171), "placement accepted less than its required saving");
    require(!prefer_merge_placement(stats, 100, 30, 0), "placement accepted missing checkpoint observations");
    stats.syncs = 0;
    require(!prefer_merge_placement(stats, 100, 30, 1), "placement accepted missing sync observations");
}
void placement(fs::path const& root, Config cfg)
{
    auto path = root / "placement.gplot";
    for (bool tail_available : {true, false})
    {
        MergeState arena(path.string(), cfg, false, identity()); arena.confirm();
        auto record = arena.layout(cfg.window_bytes);
        arena.initialize(record, arena.required_bytes(record, cfg.window_bytes * 16));
        auto& state = arena.state;
        state.live_bytes = state.block_bytes * 8;
        uint32_t const begin = uint32_t(state.live_bytes / state.block_bytes);
        if (!tail_available)
            for (uint32_t i = begin; i < state.block_limit; ++i) arena.map[i] = MergeState::reserved_block;
        MergeState::Run run;
        MergeState::Writer output(arena, run, cfg.window_bytes, true);
        std::vector<uint8_t> bytes(size_t(state.block_bytes * 3), 17); output.write(bytes); output.flush();
        auto blocks = arena.blocks(run);
        require(blocks.size() == 3 && blocks[2] == 2 &&
            (tail_available ? blocks[0] >= begin : blocks[0] == begin - 1 && blocks[1] == begin - 2),
                "canonical placement stole an imminent final position or failed its capacity fallback");
        auto const saved = arena.map;
        require(!arena.releasing(run, state.block_bytes - 1, blocks[0]) &&
            arena.releasing(run, state.block_bytes, blocks[0]) &&
            !arena.releasing(run, state.block_bytes, blocks[1]) && !arena.releasing(run, run.bytes, 0) &&
            arena.map == saved && !run.consumed && !run.released,
                "placement query freed a live or partial input block");
        output.write(std::span(bytes).first(1)); output.flush();
        auto const terminal = arena.blocks(run).back();
        require(!arena.releasing(run, run.bytes - 1, terminal) && arena.releasing(run, run.bytes, terminal),
                "placement query did not preserve the partial terminal input block");
        arena.release(run, run.bytes);
        require(!arena.releasing(run, run.bytes, terminal), "placement query revisited an already released terminal block");
        fs::remove(path);
    }
}

void packing_windows(fs::path const& root, Config cfg)
{
    auto dense = members(24, 96); auto reference = root / "packing-window-reference.gplot";
    cfg.window_bytes = 32768; cfg.checkpoint_bytes = 65536;
    cfg.memory_limit_bytes = 1024ull * 1024 * 1024;
    direct(reference, dense, cfg); auto const expected = contents(reference);
    Engine ordinary(reference.string(), cfg, true, identity());
    require(ordinary.largest_chunk() > 4096, "packing window fixture must cross several saved blocks");
    for (unsigned workers : {1u, 2u})
    for (bool overlap : {false, true})
    {
        auto path = root / "packing-window.gplot"; auto initial = cfg;
        initial.window_bytes = 4096; initial.checkpoint_bytes = 4096;
        initial.workers = 1; initial.io_overlap = false;
        {
            MergeEngine output(path.string(), initial, false, identity()); output.confirm_identity();
            require(output.append(Members(dense).subspan(0, 8)).accepted, "small-block packing input did not fit");
        }
        cfg.workers = workers; cfg.io_overlap = overlap;
        complete(path, dense, true, cfg, 16);
        require(contents(path) == expected, "changed packing windows or concurrency changed canonical bytes");
        fs::remove(path);
    }
    fs::remove(reference);
}

void placement_recovery(fs::path const& root, Config cfg)
{
    auto dense = members(8, 96); auto reference = root / "placement-reference.gplot";
    cfg.window_bytes = 32768; cfg.checkpoint_bytes = 262144;
    cfg.memory_limit_bytes = 1024ull * 1024 * 1024;
    direct(reference, dense, cfg); auto const expected = contents(reference);
    for (unsigned workers : {1u, 2u})
    for (auto const* point : {"merge-pack-placement-flushed", "merge-pack-placement-committed"})
    {
        auto path = root / "placement-recovery.gplot";
        {
            MergeState arena(path.string(), cfg, false, identity()); arena.confirm();
            auto const record = arena.layout(4096);
            arena.initialize(record, arena.required_bytes(record, 0)); arena.validate();
        }
        cfg.workers = workers; cfg.io_overlap = workers > 1;
        {
            MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity();
            require(output.append(dense).accepted, "placement recovery input did not fit");
        }
        auto const child = ::fork(); require(child >= 0, "cannot fork placement recovery test");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1);
            try
            {
                MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity();
                // Synthetic slow-read evidence makes this policy branch
                // deterministic; the underlying Stats object is mutable.
                auto& stats = const_cast<Stats&>(output.stats());
                stats.read_bytes = 1; stats.read_ns = 1000000000000000ull;
                output.finish();
            }
            catch (std::exception const& error) { std::cerr << point << ": " << error.what() << '\n'; std::_Exit(1); }
            std::_Exit(0);
        }
        int status = 0;
        require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
                "placement recovery failpoint was not reached: " + std::string(point));
        complete(path, dense, true, cfg);
        require(contents(path) == expected, "placement checkpoint recovery changed canonical bytes"); fs::remove(path);
    }
    fs::remove(reference);
}

void placement_bounds(fs::path const& root, Config cfg)
{
    auto dense = members(8, 96); auto reference = root / "placement-bound-reference.gplot";
    cfg.window_bytes = 32768; cfg.checkpoint_bytes = 262144;
    cfg.memory_limit_bytes = 1024ull * 1024 * 1024; cfg.workers = 1; cfg.io_overlap = false;
    direct(reference, dense, cfg); auto const expected = contents(reference);
    Engine ordinary(reference.string(), cfg, true, identity());
    Stats evidence; evidence.read_bytes = evidence.written_bytes = 1000000000;
    evidence.read_ns = 1000000000000000ull; evidence.write_ns = 40000000000000ull;
    evidence.syncs = evidence.sync_ns = 2;
    for (unsigned mode : {0u, 1u, 2u})
    {
        auto path = root / "placement-bound.gplot";
        {
            MergeState arena(path.string(), cfg, false, identity()); arena.confirm();
            auto const record = arena.layout(4096);
            require(prefer_merge_placement(evidence, record.block_bytes, record.slot_bytes, 1) &&
                !prefer_merge_placement(evidence, ordinary.largest_chunk(), record.slot_bytes, 1),
                    "placement bound fixture must distinguish whole-block and current-chunk savings");
            arena.initialize(record, arena.required_bytes(record, 0));
        }
        auto const child = ::fork(); require(child >= 0, "cannot fork placement bound test");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", "merge-pack-placement-flushed", 1);
            auto finish = [&](MergeEngine& output)
            {
                // Only policy evidence is synthetic. A real committed
                // contribution establishes the transient size bound.
                auto& stats = const_cast<Stats&>(output.stats());
                stats.read_bytes = evidence.read_bytes; stats.read_ns = evidence.read_ns;
                stats.written_bytes = evidence.written_bytes; stats.write_ns = evidence.write_ns;
                stats.syncs = evidence.syncs; stats.sync_ns = evidence.sync_ns;
                output.finish();
            };
            try
            {
                {
                    MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity();
                    require(output.append(dense).accepted, "placement bound input did not fit");
                    if (mode == 1)
                    {
                        timespec const times[2]{{0, UTIME_OMIT}, {1, 0}};
                        require(!::utimensat(AT_FDCWD, path.c_str(), times, 0), "cannot change placement source timestamp");
                    }
                    if (mode != 2) finish(output);
                }
                if (mode == 2)
                {
                    MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity(); finish(output);
                }
            }
            catch (std::exception const& error) { std::cerr << error.what() << '\n'; std::_Exit(1); }
            std::_Exit(0);
        }
        int status = 0;
        require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == (mode ? 0 : 86),
                "placement used an invalidated bound or failed to retain its committed bound");
        if (!mode) complete(path, dense, true, cfg);
        require(contents(path) == expected, "placement bound invalidation changed canonical bytes"); fs::remove(path);
    }
    fs::remove(reference);
}

void single_pass(fs::path const& root, Members input, Config cfg)
{
    std::vector<Member> compact;
    for (size_t i = 0; i < input.size(); ++i)
    {
        compact.emplace_back(std::vector<uint64_t>(input[i].raw().begin(), input[i].raw().end()));
        compact.back().compact(18);
    }
    auto path = root / "single-pass.gplot";
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18);
    {
        MergeEngine output(path.string(), cfg, false, identity()); output.confirm_identity();
        for (size_t member = 0; member < compact.size(); ++member)
        {
            auto const sized = output.stats().sized_chunks, encoded = output.stats().encode_chunks;
            require(output.append(Members(compact).subspan(member, 1)).accepted,
                    "bounded compact contribution did not fit");
            require(output.stats().sized_chunks == sized && output.stats().encode_chunks == encoded + chunks,
                    "bounded compact contribution repeated FSE sizing");
        }
        output.finish();
    }
    require(contents(path) == contents(root / "reference.gplot"), "single-pass compact contribution changed canonical bytes");
    fs::remove(path);

    // Borrowed/raw views keep the full sizing path. Resumed or changed saved
    // sources also require their exact summary-to-payload validation.
    {
        Member borrowed = Member::borrowed(input[0].raw());
        MergeEngine output(path.string(), cfg, false, identity()); output.confirm_identity();
        require(output.append(Members(&borrowed, 1)).accepted && output.stats().sized_chunks >= chunks,
                "borrowed contribution bypassed exact sizing");
    }
    {
        MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity();
        require(output.append(Members(compact).subspan(1, 1)).accepted && output.stats().sized_chunks >= chunks,
                "resumed contribution bypassed exact source validation");
        auto const sized = output.stats().sized_chunks;
        fs::last_write_time(path, fs::last_write_time(path) + std::chrono::seconds(1));
        require(output.append(Members(compact).subspan(2, 1)).accepted && output.stats().sized_chunks > sized,
                "changed saved source bypassed exact validation");
    }
    fs::remove(path);

    auto tight = cfg;
    tight.window_bytes = 4096; tight.checkpoint_bytes = 4096;
    for (;;)
    {
        tight.memory_limit_bytes = MergeEngine::memory_bytes(18, tight, 16);
        auto const measured = complete(path, input, false, tight, input.size()); fs::remove(path);
        if (measured.peak_file_bytes == tight.capacity_bytes) break;
        require(measured.peak_file_bytes < tight.capacity_bytes, "compact capacity fixture exceeded its cap");
        tight.capacity_bytes = measured.peak_file_bytes;
    }
    tight.capacity_bytes -= tight.window_bytes;
    tight.memory_limit_bytes = MergeEngine::memory_bytes(18, tight, 16);
    {
        MergeEngine output(path.string(), tight, false, identity()); output.confirm_identity();
        auto excess_raw = members(input.size() * 2);
        std::vector<Member> excess;
        for (auto& raw : excess_raw)
        { excess.emplace_back(std::move(raw)); excess.back().compact(18); }
        auto const before = contents(path); auto const stamp = fs::last_write_time(path);
        require(!output.append(excess).accepted && contents(path) == before && fs::last_write_time(path) == stamp,
                "compact capacity rejection changed the committed destination");
        auto const sized = output.stats().sized_chunks;
        require(output.append(compact).accepted && output.stats().sized_chunks >= sized + chunks,
                "near-capacity compact contribution did not retain exact sizing fallback");
        auto const result = output.finish();
        require(result.peak_file_bytes <= tight.capacity_bytes, "compact fallback exceeded strict capacity");
    }
    require(contents(path) == contents(root / "reference.gplot"), "compact sizing fallback changed canonical bytes");
    fs::remove(path);
    // Both boundaries follow actual provisional file writes, while the
    // previous run checkpoint still owns the whole batch.
    for (auto const* point : {"merge-contribution-partial", "merge-contribution-flushed"})
    {
        interruption(root, compact, cfg, point);
        interruption(root, compact, cfg, point, true);
    }
}

void contribution_pipeline(fs::path const& root, Config cfg)
{
    auto const directory = root / "contributions"; fs::create_directory(directory);
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18);
    uint64_t const range = 1ull << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    std::vector<std::vector<uint64_t>> raw(3);
    for (size_t member = 0; member < raw.size(); ++member)
        for (uint64_t chunk = 0; chunk < chunks; ++chunk)
        {
            uint64_t const count = chunk % 17 ? 32 : 256;
            for (uint64_t symbol = 0; symbol < count; ++symbol)
            {
                uint64_t const value = chunk * range + symbol * (range / count) + member + 7;
                raw[member].push_back(value);
                if (symbol == 3) raw[member].push_back(value);
            }
        }
    cfg.window_bytes = 4096; cfg.checkpoint_bytes = 65536;
    cfg.memory_limit_bytes = std::max(Engine::memory_bytes(18, cfg, uint16_t(raw.size())),
        MergeEngine::memory_bytes(18, cfg, uint16_t(raw.size())));
    auto const reference = directory / "reference.gplot"; direct(reference, raw, cfg);
    auto const expected = contents(reference);
    std::vector<Member> compact;
    for (auto const& member : raw) { compact.emplace_back(std::vector<uint64_t>(member)); compact.back().compact(18); }
    std::optional<std::pair<uint64_t, uint64_t>> admission;
    for (unsigned workers : {1u, 4u})
        for (bool overlap : {false, true})
        {
            cfg.workers = workers; cfg.io_overlap = overlap;
            cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, uint16_t(compact.size()));
            auto const path = directory / (std::to_string(workers) + (overlap ? "-overlap.gplot" : "-serial.gplot"));
            MergeEngine output(path.string(), cfg, false, identity()); output.confirm_identity();
            auto const result = output.append(compact);
            require(result.accepted && !output.stats().sized_chunks,
                    "immutable contribution did not retain single-pass admission");
            auto const bounds = std::pair(result.attempted_final_bytes, result.attempted_peak_bytes);
            if (!admission) admission = bounds;
            require(bounds == *admission, "parallel summary reduction changed capacity admission");
            require(bool(output.stats().overlap_windows) == (overlap && workers > 1),
                    "contribution overlap did not follow the requested worker/window policy");
            auto const final = output.finish();
            require(final.peak_file_bytes <= cfg.capacity_bytes && contents(path) == expected,
                    "bounded contribution pipeline changed canonical bytes or exceeded capacity");
        }

    // A chunk that needs the aggregate codec allowance must still work with
    // overlap requested. Use the codec's admission estimate for this fixture.
    std::vector<FragmentRegion> regions(compact.size());
    uint64_t required = ChunkCodec::estimate_base();
    for (uint64_t chunk = 0; chunk < chunks; ++chunk)
    {
        uint64_t symbols = 0;
        for (size_t member = 0; member < compact.size(); ++member)
        { regions[member] = compact[member].view().region(chunk); symbols += regions[member].size(); }
        auto const detail = ChunkCodec::fresh_summary(18, chunk, regions);
        auto const most = ChunkCodec::size_bounds(detail).second;
        required = std::max({required, ChunkCodec::estimate_work(18, 0, symbols),
            ChunkCodec::estimate_work(18, most + 8, 0)});
    }
    cfg.codec_bytes = required; cfg.workers = 4; cfg.io_overlap = true;
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, uint16_t(compact.size()));
    auto const fallback = directory / "codec-fallback.gplot";
    {
        MergeEngine output(fallback.string(), cfg, false, identity()); output.confirm_identity();
        require(output.append(compact).accepted && !output.stats().overlap_windows &&
            output.stats().contribution_workers == 1, "tight codec budget did not retain serial contribution fallback");
        output.finish();
    }
    require(contents(fallback) == expected, "tight codec budget changed packed output");

    // Fail a later bounded summary window, then reuse the same destination.
    // All queued readers must finish before their borrowed regions go away.
    auto damaged = raw;
    damaged[0].erase(std::lower_bound(damaged[0].begin(), damaged[0].end(), (chunks - 1) * range), damaged[0].end());
    std::vector<Member> invalid;
    for (auto& member : damaged) { invalid.emplace_back(std::move(member)); invalid.back().compact(18); }
    cfg = configuration(); cfg.workers = 4; cfg.io_overlap = true; cfg.window_bytes = 4096;
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, uint16_t(compact.size()));
    auto const failed = directory / "summary-failure.gplot";
    {
        MergeEngine output(failed.string(), cfg, false, identity()); output.confirm_identity();
        bool rejected = false;
        try { (void)output.append(invalid); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected && !output.committed_member_count() && !fs::file_size(failed),
                "failed parallel summary scan changed the destination");
        require(output.append(compact).accepted, "parallel summary failure prevented a valid retry"); output.finish();
    }
    require(contents(failed) == expected, "parallel summary retry changed canonical bytes");

#ifdef GROW_TEST_WRAP_PWRITE
    auto const write_failed = directory / "writer-failure.gplot";
    {
        MergeEngine output(write_failed.string(), cfg, false, identity()); output.confirm_identity();
        require(output.append(Members(compact).subspan(0, 1)).accepted, "writer failure prefix did not fit");
    }
    std::pair<uint64_t, uint64_t> failure_range;
    {
        MergeState saved(write_failed.string(), cfg, true, identity());
        auto const free = std::find(saved.map.begin(), saved.map.begin() + saved.state.block_limit, MergeState::free_block);
        require(free != saved.map.begin() + saved.state.block_limit, "writer failure fixture has no free block");
        uint64_t const begin = uint64_t(free - saved.map.begin()) * saved.state.block_bytes;
        require(cfg.window_bytes < saved.state.block_bytes, "writer failure fixture needs multiple writes per block");
        // Let the first full provisional buffer reach disk before failing a
        // later write while another bounded encode has been scheduled.
        failure_range = {begin + cfg.window_bytes, begin + saved.state.block_bytes};
    }
    {
        MergeEngine output(write_failed.string(), cfg, true, identity()); output.confirm_identity();
        struct DisableFailure { ~DisableFailure() { fail_write.store(false, std::memory_order_release); } } disable;
        fail_write_begin.store(failure_range.first, std::memory_order_relaxed);
        fail_write_end.store(failure_range.second, std::memory_order_relaxed);
        fail_write.store(true, std::memory_order_release); bool rejected = false;
        try { (void)output.append(Members(compact).subspan(1)); }
        catch (std::runtime_error const& error)
        { rejected = std::string_view(error.what()).starts_with("cannot write merged destination"); }
        require(rejected && !fail_write.load(std::memory_order_acquire) && output.stats().overlap_windows && output.committed_member_count() == 1,
                "provisional writer failure did not drain a pending contribution window");
    }
    complete(write_failed, compact, true, cfg);
    require(contents(write_failed) == expected, "provisional writer failure changed resumed canonical bytes");
#endif

    auto denied = cfg; --denied.memory_limit_bytes;
    bool rejected = false;
    try
    {
        MergeEngine output((directory / "memory-denied.gplot").string(), denied, false, identity());
        output.confirm_identity(); (void)output.append(compact);
    }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "bounded contribution descriptors bypassed memory admission");

    // These exits occur after provisional writes while the following encode
    // may still be running; recovery must discard the whole uncommitted run.
    for (auto const* point : {"merge-contribution-partial", "merge-contribution-flushed"})
    {
        interruption(directory, compact, cfg, point);
        interruption(directory, compact, cfg, point, true);
    }

    // One task's member descriptors can exceed a tiny I/O window. Its batch
    // metadata is bounded even when the chunk cannot use the size-index code.
    auto sparse = members(size_t(cfg.window_bytes / sizeof(FragmentRegion) + 1), 3);
    std::vector<Member> many;
    for (auto const& member : sparse) { many.emplace_back(std::vector<uint64_t>(member)); many.back().compact(18); }
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, uint16_t(many.size()));
    auto const sparse_path = directory / "sparse.gplot";
    {
        MergeEngine output(sparse_path.string(), cfg, false, identity()); output.confirm_identity();
        bool rejected = false;
        try { rejected = !output.append(many).accepted; } catch (UnencodableChunkSize const&) { rejected = true; }
        require(rejected && !output.committed_member_count() && !fs::file_size(sparse_path),
                "oversized task descriptors bypassed bounded chunk admission");
        require(output.append(compact).accepted, "oversized task descriptors prevented a valid retry"); output.finish();
    }
    require(contents(sparse_path) == expected, "oversized task descriptors changed the subsequent output");
}

template<class F> void rejects_unchanged(fs::path const& path, F&& operation)
{
    auto const before = contents(path); auto const stamp = fs::last_write_time(path);
    bool rejected = false;
    try { operation(); } catch (std::runtime_error const&) { rejected = true; }
    require(rejected && contents(path) == before && fs::last_write_time(path) == stamp,
            "invalid merged recovery changed committed bytes");
}
void admission(fs::path const& root, Members input, Config cfg)
{
    auto path = root / "capacity.gplot";
    auto const measured = complete(path, input, false, cfg);
    fs::remove(path);
    cfg.capacity_bytes = measured.peak_file_bytes;
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, 16);
    auto const tight = complete(path, input, false, cfg);
    require(tight.peak_file_bytes <= cfg.capacity_bytes && contents(path) == contents(root / "reference.gplot"),
            "near-capacity merge differs");
    fs::remove(path);
    for (auto const* point : {"merge-pack-committed:2", "merge-layout-partial"})
        interruption(root, input, cfg, point);

    // Capacity denial of a complete ordinary group is read-only, including
    // when its trailing recovery metadata cannot be admitted at all.
    direct(path, input.subspan(0, 1), configuration());
    cfg.capacity_bytes = fs::file_size(path); cfg.memory_limit_bytes = configuration().memory_limit_bytes;
    auto const before = contents(path); auto const stamp = fs::last_write_time(path);
    {
        MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity();
        require(!output.append(input.subspan(1, 1)).accepted, "full ordinary destination admitted merge workspace");
    }
    require(contents(path) == before && fs::last_write_time(path) == stamp, "capacity denial changed ordinary group");
    fs::remove(path);

    for (unsigned corruption : {0u, 1u, 2u, 3u})
    {
        cfg = configuration();
        {
            MergeEngine output(path.string(), cfg, false, identity()); output.confirm_identity();
            require(output.append(input.subspan(0, 1)).accepted, "corruption fixture did not fit");
        }
        auto smaller = cfg; smaller.codec_bytes = 1;
        rejects_unchanged(path, [&] { MergeEngine output(path.string(), smaller, true, identity()); });
        {
            MergeState arena(path.string(), cfg, true, identity()); arena.confirm();
            if (corruption == 1)
            {
                arena.map[arena.state.runs[0].head] = arena.state.runs[0].head;
                arena.checkpoint("test-corrupt-map");
            }
            else if (corruption == 0)
            {
                uint64_t const offset = uint64_t(arena.state.summary.head) * arena.state.block_bytes;
                auto byte = arena.value<uint8_t>(offset); byte ^= 1; arena.value(offset, byte); arena.sync();
            }
            else if (corruption == 2)
            {
                for (unsigned slot = 0; slot < 2; ++slot)
                {
                    uint64_t const offset = arena.state.anchor + 2 * arena.alignment() + slot * arena.state.slot_bytes;
                    auto byte = arena.value<uint8_t>(offset); byte ^= 1; arena.value(offset, byte);
                }
                arena.sync();
            }
            else
            {
                arena.state.chunk = 1; arena.checkpoint("test-corrupt-phase");
            }
        }
        rejects_unchanged(path, [&]
        {
            MergeEngine output(path.string(), cfg, true, identity()); output.confirm_identity();
            (void)output.append(input.subspan(1, 1));
        });
        fs::remove(path);
    }
}
void writers(fs::path const& root)
{
    std::array<StorageDevice, 5> drives{{{"a", {"one"}}, {"b", {"two"}}, {"c", {"one", "three"}},
        {"d", {"three"}}, {"e", {"four"}}}};
    auto groups = writer_groups(drives);
    require(groups[0] == groups[2] && groups[2] == groups[3] && groups[0] != groups[1] && groups[1] != groups[4],
            "overlapping physical drives got independent writers");
    drives[4].drives.clear(); groups = writer_groups(drives);
    require(std::all_of(groups.begin(), groups.end(), [](auto group) { return group == 0; }),
            "unknown physical topology was treated as independent");
    auto const device = storage_device(root);
    require(!device.filesystem.empty(), "destination filesystem identity is missing");
    std::vector<int> order;
    OrderedWriter writer([] {});
    std::promise<void> started, release; auto signal = release.get_future().share();
    auto first = writer.submit([&, signal] { started.set_value(); signal.wait(); order.push_back(1); });
    started.get_future().get();
    auto second = writer.submit([&] { order.push_back(2); });
    require(!writer.has_room() && writer.busy(), "ordered writer did not bound its pending queue");
    bool rejected = false;
    try { (void)writer.submit([] {}); } catch (std::exception const&) { rejected = true; }
    require(rejected, "ordered writer accepted an unbounded queue");
    {
        OrderedWriter independent([] {});
        auto separate = independent.submit([] {}); separate.get();
        require(first.wait_for(std::chrono::seconds(0)) != std::future_status::ready,
                "independent drive work waited for another drive");
    }
    release.set_value(); first.get(); second.get();
    require(order == std::vector<int>({1, 2}), "drive writes were reordered");
    auto failed = writer.submit([] { throw std::runtime_error("writer failure"); });
    bool propagated = false;
    try { failed.get(); } catch (std::runtime_error const& error) { propagated = std::string_view(error.what()) == "writer failure"; }
    require(propagated, "destination writer lost its failure");
}
void group_prefetch(fs::path const& root, Members input)
{
    using namespace grow_controller_test;
    auto const requested = identity_from(identity());
    Fixtures fixture{requested, {}, 0};
    for (uint16_t index = 0; index < 4; ++index)
    {
        auto const path = root / ("prefetch-member-" + std::to_string(index));
        PlotData data; data.t3_proof_fragments.assign(input[index].raw().begin(), input[index].raw().end());
        fixture.bytes = sum(fixture.bytes, PlotFile::writeData(path.string(), data, requested.params, index, requested.memo));
        fixture.paths.push_back(path);
    }
    Arguments args; args.max_ram = 512 * mib; args.batch = args.max_members = 4;
    args.workers = 2; args.window = 65536; args.checkpoint_bytes = mib;
    args.construction = "merged"; args.member_storage = "compact";
    Backend model; model.mode = "fixture"; model.generation_overlap = true;
    Producer producer{requested, &fixture, model, args, args.max_ram, mib, mib};
    GroupPrefetch partial;
    require(prefetch_group(partial, producer, requested, 1, 4, 0, "test prefetch member ", [&]
        { return partial.members.size() == 2; }) == 2 && partial.hash && partial.hash->size() == 2 &&
        producer.fill_stop == Producer::FillStop::requested && !fill_wait_reason(producer.fill_stop),
        "next group producer did not stop at a completed packing boundary");
    require(!prefetch_group(partial, producer, requested, 2, 4, 0, "unexpected prefetch ", [] { return false; }) &&
        partial.owner == 1 && partial.members.size() == 2, "next group queue retained more than one batch");
    std::vector<Member> transferred; std::unique_ptr<MemberBatchHash> transferred_hash;
    partial.take(transferred, transferred_hash, 1, 1);
    require(transferred.size() == 1 && transferred[0].compact() && partial.members.empty() && !partial.identity,
        "smaller next engine admission lost owning members or retained the old batch");
    GroupPrefetch denied;
    producer.budget = sum(producer.fixed, producer.next_peak(0)) - 1;
    require(!prefetch_group(denied, producer, requested, 1, 4, 0, "unexpected prefetch ", [] { return false; }) &&
        denied.members.empty(), "next group prefetch ignored insufficient RAM");
    producer.budget = args.max_ram;
    for (auto const* mode : {"cpu", "idle"})
    {
        model.mode = mode;
        require(!prefetch_group(denied, producer, requested, 1, 4, 0, "unexpected prefetch ", [] { return false; }),
            "next group prefetch enabled an unavailable or CPU producer");
    }
    model.mode = "fixture"; args.generation_overlap = false;
    require(!group_prefetch_enabled(args, model), "next group prefetch ignored overlap off");
    args.generation_overlap = true; args.member_storage = "raw";
    require(!group_prefetch_enabled(args, model), "next group prefetch enabled a raw borrowed-output handoff");
    args.member_storage = "compact"; model.generation_overlap = false;
    require(!group_prefetch_enabled(args, model), "next group prefetch ignored the effective overlap policy");

#ifdef GROW_TEST_WRAP_PWRITE
    // The final writer is held at its first real syscall. Fixture loading uses
    // the normal owning producer on this thread while the old engine packs.
    for (bool failure : {false, true})
    {
        std::unique_ptr<Backend> backend;
        GroupPrefetch next;
        auto const current = root / (failure ? "prefetch-failed-current" : "prefetch-current");
        auto const following = root / (failure ? "prefetch-failed-next" : "prefetch-next");
        std::promise<void> started, release;
        auto started_signal = started.get_future();
        producer_thread = std::this_thread::get_id(); background_write_started = &started;
        background_write_release = release.get_future().share();
        hold_background_write.store(true, std::memory_order_release);
        bool rejected = false, injected = false;
        try
        {
            auto const result = run_group(args, current, &fixture, requested, 4, 128 * mib, 0, backend,
                nullptr, {}, {}, nullptr,
                [&](uint64_t fixed, uint16_t cap, std::future<Result>& pending)
                {
                    started_signal.get();
                    struct Release { std::promise<void>& signal; ~Release() { signal.set_value(); } } unblock{release};
                    uint16_t const wanted = std::min(cap, uint16_t(4));
                    auto const next_config = grow_controller_test::configuration(args, 128 * mib, mib);
                    Producer next_producer{requested, &fixture, *backend, args, args.max_ram,
                        sum(fixed, prefetch_memory(args, requested, next_config, wanted, storage_device(root).nonrotating)), fixed};
                    auto const generated = prefetch_group(next, next_producer, requested, 1, wanted, 0,
                        "test next group member ", [&]
                        { return pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
                    require(generated == wanted && generated > 1 && background_write_active.load(std::memory_order_acquire) &&
                        !fs::exists(following), "next owning batch did not advance during blocked final packing");
                    next.metadata_bytes = sum(product(wanted, 2 * sizeof(Member)), batch_hash_bytes(args, wanted));
                    require(next_producer.peak <= args.max_ram, "next group batch exceeded combined RAM admission");
                    if (failure)
                    {
                        fail_write_begin.store(0); fail_write_end.store(UINT64_MAX);
                        fail_write.store(true, std::memory_order_release); injected = true;
                    }
                    return wait_bit(ControllerWait::batch_full);
                });
            require(!failure && result.members == 4 && result.generated_members == 4 && next.members.size() == 4,
                "final packing did not preserve the completed current and next batches");
            auto const generated = backend->generated_members;
            auto const handoff = run_group(args, following, &fixture, requested, 4, 128 * mib, 0, backend,
                nullptr, {}, {}, &next);
            require(handoff.members == 4 && !handoff.generated_members && backend->generated_members == generated &&
                next.members.empty() && contents(current) == contents(following),
                "next group handoff regenerated members or changed the published bytes");
        }
        catch (std::runtime_error const&) { if (!failure) throw; rejected = true; }
        require(rejected == failure && (!failure || (injected && !fail_write.load() && next.members.size() == 4)) &&
            !background_write_active.load(std::memory_order_acquire) && !hold_background_write.load(std::memory_order_acquire),
            "failed final packing did not drain its writer");
        fail_write.store(false); background_write_started = nullptr; background_write_release = {};
        next.clear();
    }
#endif
    // Find the ordinary receiver's exact admission boundary. Its inherited
    // owning payload replaces the generated prefix rather than coexisting
    // with another complete prefix and a new generation.
    uint64_t const base = sum(sum(fixture.bytes, sizeof(ConstructionHistory)), mib);
    uint64_t const generation = producer.next_peak(0);
    uint64_t const storage = member_storage_bound(args, requested);
    auto ordinary_fits = [&](uint64_t budget)
    {
        auto options = args; options.max_ram = budget;
        uint64_t const caller = sum(base, sum(2 * sizeof(Member), batch_hash_bytes(options, 1)));
        if (caller >= budget) return false;
        auto const cfg = grow_controller_test::configuration(options, 128 * mib, caller);
        uint64_t const fixed = sum(base, prefetch_memory(options, requested, cfg, 4, storage_device(root).nonrotating));
        return sum(fixed, sum(generation, product(3, storage))) <= budget;
    };
    uint64_t low = mib, high = args.max_ram;
    require(ordinary_fits(high), "constrained receiver fixture has no ordinary admission");
    while (low < high)
    {
        uint64_t const middle = low + (high - low) / 2;
        if (ordinary_fits(middle)) high = middle;
        else low = middle + 1;
    }
    args.max_ram = low;
    std::unique_ptr<Backend> ordinary_backend;
    auto const reference = root / "prefetch-constrained-reference";
    auto const ordinary = run_group(args, reference, &fixture, requested, 4, 128 * mib, 0, ordinary_backend);
    require(ordinary.max_batch_members == 4 && ordinary.generated_members == 4,
        "constrained ordinary receiver did not admit its normal full batch");
    auto handoff = [&](uint16_t prefetched_cap, uint16_t count, uint16_t receiver_cap, std::string const& name)
    {
        auto backend = std::make_unique<Backend>();
        backend->mode = "fixture"; backend->generation_overlap = true;
        Producer staged{requested, &fixture, *backend, args, args.max_ram, mib, mib};
        GroupPrefetch batch;
        require(prefetch_group(batch, staged, requested, 1, prefetched_cap, 0, "test constrained prefetch ", [&]
            { return batch.members.size() == count; }) == count, "constrained prefetch did not retain its requested members");
        batch.metadata_bytes = sum(product(prefetched_cap, 2 * sizeof(Member)), batch_hash_bytes(args, prefetched_cap));
        auto const generated = backend->generated_members;
        auto options = args; options.batch = receiver_cap;
        auto const path = root / name;
        auto const received = run_group(options, path, &fixture, requested, 4, 128 * mib, 0, backend,
            nullptr, {}, {}, &batch);
        require(received.members == 4 && received.max_batch_members == receiver_cap &&
            received.generated_members == 4 - count && backend->generated_members == generated + received.generated_members &&
            batch.members.empty() && received.peak_modeled_bytes <= args.max_ram && contents(path) == contents(reference),
            "constrained receiver shrank its ordinary batch, regenerated inherited members or changed output bytes");
    };
    handoff(4, 4, 4, "prefetch-constrained-full");
    handoff(2, 2, 4, "prefetch-constrained-larger");
    handoff(4, 2, 2, "prefetch-constrained-smaller");

    // Destination selection keeps the old metadata in fixed until take().
    // Its incoming payload replaces a prefix; another destination's payload
    // must still reduce the available generation room.
    {
        auto const caller = sum(base, sum(2 * sizeof(Member), batch_hash_bytes(args, 1)));
        auto const cfg = grow_controller_test::configuration(args, 128 * mib, caller);
        Construction engine((root / "prefetch-selection").string(), cfg, false, identity(), "auto");
        ConstructionSample shape; shape.member_bytes = ordinary.final_bytes / ordinary.members;
        engine.history.shapes.add(shape);
        uint64_t const engine_bytes = engine.memory_bytes(4, true);
        uint64_t const metadata = sum(product(4, 2 * sizeof(Member)), batch_hash_bytes(args, 4));
        uint64_t const fixed = sum(sum(base, engine_bytes), metadata);
        uint64_t const budget = sum(fixed, sum(generation, product(3, storage)));
        Backend backend; backend.mode = "fixture"; backend.generation_overlap = true;
        Producer receiver{requested, &fixture, backend, args, budget, fixed, fixed};
        auto normal = automatic_selection(engine, receiver, 4, 4, engine_bytes, 0, 0, true);
        receiver.fixed = sum(fixed, metadata);
        auto inherited = automatic_selection(engine, receiver, 4, 4, engine_bytes, 0, 0, false,
            product(2, storage), metadata);
        receiver.fixed = fixed;
        auto queued = automatic_selection(engine, receiver, 4, 4, engine_bytes, storage, 0, false);
        require(normal && inherited && queued && normal->batch == 4 && inherited->batch == normal->batch &&
            queued->batch < normal->batch,
            "destination selection confused its incoming batch with other queued payload or freed its metadata early");
    }

    // A changed next-group reservation must preserve ordinary serial admission.
    // Model an obsolete metadata reservation without allocating a large queue.
    auto backend = std::make_unique<Backend>();
    backend->mode = "fixture"; backend->generation_overlap = true;
    Producer staged{requested, &fixture, *backend, args, args.max_ram, mib, mib};
    GroupPrefetch dropped;
    require(prefetch_group(dropped, staged, requested, 1, 4, 0, "test obsolete prefetch ", [] { return false; }) == 4,
        "obsolete reservation fixture did not fill its owning batch");
    auto const before = backend->generated_members;
    dropped.metadata_bytes = args.max_ram;
    auto const retried = run_group(args, root / "prefetch-serial-retry", &fixture, requested, 4,
        128 * mib, 0, backend, nullptr, {}, {}, &dropped);
    require(retried.members == 4 && retried.generated_members == 4 && dropped.members.empty() &&
        backend->generated_members == before + retried.generated_members && retried.peak_modeled_bytes <= args.max_ram,
        "speculative reservation prevented a valid serial next group or lost discarded generation accounting");
}
Result finish_with_cpu_record(grow_controller_test::Construction& engine, bool eligible, bool body = true)
{
    using namespace grow_controller_test;
    std::ostringstream diagnostic;
    Result result;
    {
        struct Restore
        {
            std::streambuf* previous;
            ~Restore() { std::cerr.rdbuf(previous); }
        } restore{std::cerr.rdbuf(diagnostic.rdbuf())};
        result = engine.finish();
    }
    auto const output = diagnostic.str(); std::cerr << output;
    if (controller_profile())
    {
        auto const record = output.find("phase=external_pack_body history_eligible=" + std::to_string(eligible));
        require(body ? record != std::string::npos : output.find("phase=external_pack_body") == std::string::npos,
            "packing CPU diagnostic omitted performed work, misstated eligibility or reported zero-work publication");
    }
    return result;
}

void successor_lookahead(fs::path const& root, Members input,
    grow_controller_test::Fixtures const& fixture, Config cfg)
{
#ifdef GROW_TEST_WRAP_PWRITE
    using namespace grow_controller_test;
    auto const requested = identity_from(identity());
    Arguments args; args.max_ram = 512 * mib; args.batch = 2; args.workers = 2;
    args.member_storage = "compact"; args.window = cfg.window_bytes; args.checkpoint_bytes = cfg.checkpoint_bytes;
    cfg.workers = 1; cfg.capacity_bytes = 128 * mib; cfg.memory_limit_bytes = args.max_ram;
    for (auto const* mode : {"drain", "consumer-failure", "producer-failure", "rejected",
            "memory", "other-retained", "off", "raw", "cpu"})
    {
        auto options = args;
        bool const memory = std::string_view(mode) == "memory" || std::string_view(mode) == "other-retained";
        bool const serial = memory || std::string_view(mode) == "off" || std::string_view(mode) == "raw" ||
            std::string_view(mode) == "cpu";
        if (std::string_view(mode) == "off") options.generation_overlap = false;
        if (std::string_view(mode) == "raw") options.member_storage = "raw";
        auto directory = root / (std::string("successor-lookahead-") + mode); fs::create_directory(directory);
        auto scratch = directory / "scratch"; fs::create_directory(scratch);
        auto targets = directory / "targets"; fs::create_directory(targets);
        auto scratch_info = ScratchRoot::inspect(scratch);
        scratch_info.device = {"lookahead-scratch-fs", {"lookahead-ssd"}, true, false};
        auto target_info = ScratchRoot::inspect(targets);
        target_info.device = {"lookahead-target-fs", {"lookahead-hdd"}, false, true};
        ScratchPool pool({scratch_info}, 0);
        require(pool.try_reserve_output(0, target_info, cfg.capacity_bytes), "lookahead output did not fit");
        pool.seal_recovery();
        bool const rejected = std::string_view(mode) == "rejected";
        auto grant = rejected ? pool.grant(0, 4096, 4096, 4096, 2) : pool.grant(0, 128 * mib, 128 * mib, mib, 2);
        require(bool(grant), "lookahead scratch did not fit");
        GroupSuccessor next(targets / "next", requested, 4, cfg.capacity_bytes);
        next.owner = 0; next.workers = 1; next.batch_cap = memory ? 1 : 2; next.scratch = grant;
        next.io = pool.try_stage_io(0); require(bool(next.io), "lookahead stage lease was denied");
        next.engine = std::make_unique<Construction>(next.path.string(), cfg, false, identity(), "external", grant);
        next.engine->confirm_identity();
        auto source = std::make_shared<SharedConstructionHistory>();
        source->successor_batches.prepare(next.batch_cap); source->successor_batch_workers = next.workers;
        next.engine->demand_history = source;
        next.engine_bytes = sum(next.engine->memory_bytes(next.batch_cap, options.member_storage == "compact"),
            sum(product(next.batch_cap, 2 * sizeof(Member)), batch_hash_bytes(options, next.batch_cap)));
        next.fixed = sum(sum(fixture.bytes, next.engine_bytes), sum(pool.memory_bytes(), 4 * mib));
        Backend backend; backend.mode = std::string_view(mode) == "cpu" ? "cpu" : "fixture";
        backend.generation_overlap = true;
        std::promise<void> started, release;
        auto started_signal = started.get_future();
        producer_thread = std::this_thread::get_id(); background_write_started = &started;
        background_write_release = release.get_future().share(); background_write_timed_out.store(false);
        background_check_target = next.path.string();
        hold_background_check.store(rejected); hold_background_write.store(!rejected);
        struct Release
        {
            std::promise<void>& signal; bool sent = false;
            void send() { if (!sent) { sent = true; signal.set_value(); } }
            ~Release() { send(); }
        } unblock{release};
        if (serial)
        {
            uint64_t const retained = std::string_view(mode) == "other-retained" ? 7 * mib : 0;
            if (memory)
            {
                Producer probe{requested, &fixture, backend, options, options.max_ram, next.fixed, next.fixed};
                options.max_ram = sum(sum(next.fixed, retained), probe.next_peak(0));
            }
            unblock.send(); // An unexpected async write reports through the assertion, without blocking.
            stage_successor(next, options, backend, retained, [] { return false; }, &fixture);
            require(hold_background_write.load() && !background_write_timed_out.load() &&
                next.metrics.members == 4 && next.metrics.generated_members == 4 &&
                next.metrics.peak_modeled_bytes <= options.max_ram,
                "successor ignored serial policy or retained-memory admission");
            hold_background_write.store(false); background_write_started = nullptr; background_write_release = {};
            background_check_target.clear(); next.close(); continue;
        }
        unsigned observations = 0;
        bool observed = false, failed = false;
        try
        {
            stage_successor(next, options, backend, 7 * mib, [&]
            {
                if (backend.generated_members != 2) return false;
                if (!observations++) return false; // The immutable candidate is ready to launch.
                if (observations == 2)
                {
                    require(started_signal.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
                        "successor append did not start on its CPU consumer");
                    require(!next.engine->external_batches.sizes[0].planning.count &&
                        !source->successor_batches.sizes[0].encoding.count,
                        "successor published batch timing before its immutable CPU consumer joined");
                    return false; // Generate a member while the real consumer is held.
                }
                observed = background_write_active.load(std::memory_order_acquire);
                if (std::string_view(mode) == "consumer-failure")
                {
                    fail_write_begin.store(0); fail_write_end.store(UINT64_MAX); fail_write.store(true);
                }
                unblock.send();
                if (std::string_view(mode) == "producer-failure") throw std::runtime_error("successor producer failure");
                return true; // The predecessor finished: drain, without generating further members.
            }, &fixture);
        }
        catch (std::runtime_error const&) { failed = true; }
        unblock.send();
        require(observed && !background_write_active.load(std::memory_order_acquire) &&
            !background_write_timed_out.load(std::memory_order_acquire) &&
            !hold_background_write.load() && !hold_background_check.load(),
            "successor did not overlap generation or drain its held CPU consumer");
        bool const consumer_failure = std::string_view(mode) == "consumer-failure";
        bool const producer_failure = std::string_view(mode) == "producer-failure";
        require(failed == (consumer_failure || producer_failure) && (!consumer_failure || !fail_write.load()),
            "successor lost its consumer or producer failure");
        fail_write.store(false); background_write_started = nullptr; background_write_release = {};
        background_check_target.clear();
        auto const saved = ExternalEngine::inspect(next.path.string());
        if (rejected)
        {
            require(next.deferred && !next.metrics.members && next.metrics.generated_members == 3 &&
                !saved.members && !saved.finishing && next.engine->external->can_append(),
                "rejected successor retained an obsolete lookahead or invoked full target packing");
            require(!source->successor_batches.sizes[0].planning.count &&
                !source->successor_batches.sizes[1].encoding.count,
                "rejected successor trained the next group's batch policy");
        }
        else if (producer_failure)
        {
            require(saved.members == 2 && !saved.finishing && !next.metrics.members,
                "producer unwind did not drain the immutable append to its durable frontier");
            require(!source->successor_batches.sizes[0].planning.count,
                "producer failure trained an unobserved successor append");
        }
        else if (consumer_failure)
        {
            require(!saved.members && !saved.finishing, "failed successor append acknowledged an uncommitted queue");
            require(!source->successor_batches.sizes[0].encoding.count,
                "consumer failure trained an uncommitted successor append");
        }
        else
        {
            require(next.metrics.members == 3 && next.metrics.generated_members == 3 && saved.members == 3 &&
                !saved.finishing && next.metrics.max_batch_members == 2 && next.metrics.peak_modeled_bytes <= args.max_ram,
                "predecessor stop lost an owning lookahead, generated further members or exceeded RAM admission");
            require(source->successor_batches.sizes[0].planning.count == 1 &&
                source->successor_batches.sizes[1].encoding.count == 1 &&
                source->successor_batches.source_members == saved.members && source->successor_batches.source_bytes,
                "joined successor did not retain accepted batch timings and source-shape provenance");
            source->successor_batches.half_attempted = true;
            source->successor_batches.selected = 1;
            next.engine->external_batches = {}; // A paused lane's old snapshot must not win.
            auto const generated = backend.generated_members;
            stage_successor(next, args, backend, 7 * mib, [] { return false; }, &fixture);
            require(next.metrics.members == 4 && next.metrics.generated_members == 4 && backend.generated_members == generated + 1,
                "successor reentry regenerated its drained prefix");
            require(source->successor_batches.half_attempted && source->successor_batches.selected == 1 &&
                source->successor_batches.sizes[0].planning.count == 1 &&
                source->successor_batches.sizes[1].encoding.count == 2,
                "a stale compatible lane overwrote its shared trial, selection or completed observations");
            ++source->successor_batch_workers;
            stage_successor(next, args, backend, 7 * mib, [] { return true; }, &fixture);
            require(!next.engine->external_batches.ceiling && !next.engine->external_batches.half_attempted &&
                !next.engine->external_batches.sizes[0].planning.count,
                "successor reused batch timings from a different compression-worker grant");
            next.io.reset();
            auto io = pool.try_io(0); require(bool(io), "lookahead finish could not acquire ordinary writer authority");
            next.engine->prepare_finish(); next.engine->finish();
            auto const reference = directory / "reference"; direct(reference, input.subspan(0, 4), cfg);
            require(contents(next.path) == contents(reference), "successor lookahead changed member order or ordinary v2 bytes");
        }
        next.close();
    }
#endif
}

void successor_staging(fs::path const& root, Members input, Config cfg)
{
    using namespace grow_controller_test;
    auto const requested = identity_from(identity());
    auto scratch = root / "successor-scratch"; fs::create_directory(scratch);
    auto targets = root / "successor-targets"; fs::create_directory(targets);
    auto const current_path = group_path(targets, 0), next_path = group_path(targets, 1);
    Fixtures fixture{requested, {}, 0};
    for (uint16_t index = 0; index < 4; ++index)
    {
        auto const path = root / ("successor-member-" + std::to_string(index));
        PlotData data; data.t3_proof_fragments.assign(input[index].raw().begin(), input[index].raw().end());
        fixture.bytes = sum(fixture.bytes, PlotFile::writeData(path.string(), data, requested.params, index, requested.memo));
        fixture.paths.push_back(path);
    }
    successor_lookahead(root, input, fixture, cfg);
    auto scratch_info = ScratchRoot::inspect(scratch);
    scratch_info.device = {"successor-scratch-fs", {"successor-ssd"}, true, false};
    auto target_info = ScratchRoot::inspect(targets);
    target_info.device = {"successor-target-fs", {"successor-hdd"}, false, true};
    ScratchPool pool({scratch_info}, 0);
    require(pool.try_reserve_output(0, target_info, 128 * mib) &&
        pool.try_reserve_output(1, target_info, 128 * mib), "successor test could not reserve both outputs");
    pool.seal_recovery();
    auto old_scratch = pool.grant(0, 128 * mib, 128 * mib, mib, 2);
    auto next_scratch = pool.grant(1, 128 * mib, 128 * mib, mib, 2);
    require(old_scratch && next_scratch, "successor test could not reserve independent contribution scratch");
    auto gate = std::make_shared<std::recursive_mutex>();
    old_scratch->target_gate = next_scratch->target_gate = gate;
    cfg.workers = 1; cfg.capacity_bytes = 128 * mib; cfg.memory_limit_bytes = 512 * mib;
    Construction old(current_path.string(), cfg, false, identity(), "external", old_scratch);
    old.confirm_identity();
    require(old.append(input.subspan(0, 4)).accepted, "successor test could not stage the current group");
    std::optional<fs::path> old_owned;
    for (auto const& entry : fs::directory_iterator(scratch))
        if (entry.is_directory()) { old_owned = entry.path(); break; }
    require(old_owned.has_value(), "successor test has no current scratch namespace");
    auto old_io = pool.try_io(0); require(bool(old_io), "successor test could not acquire the old final writer");
    GroupSuccessor next(next_path, requested, 4, 128 * mib);
    next.owner = 1; next.workers = 1; next.batch_cap = 1; next.scratch = next_scratch;
    next.io = pool.try_stage_io(1); require(bool(next.io), "independent successor stage was blocked by the old pack lease");
    next.engine = std::make_unique<Construction>(next.path.string(), cfg, false, identity(), "external", next.scratch);
    next.engine->confirm_identity();
    Arguments args; args.max_ram = 512 * mib; args.batch = 1; args.workers = 2;
    args.member_storage = "compact"; args.window = cfg.window_bytes; args.checkpoint_bytes = cfg.checkpoint_bytes;
    Backend backend; backend.mode = "fixture"; backend.generation_overlap = true;
    next.engine_bytes = sum(next.engine->memory_bytes(1, true),
        sum(2 * sizeof(Member), batch_hash_bytes(args, 1)));
    next.fixed = sum(sum(sum(fixture.bytes, old.memory_bytes(4, true)), next.engine_bytes),
        sum(pool.memory_bytes(), 4 * mib));
    require(next.fixed < args.max_ram, "successor fixture has no modeled producer headroom");
    auto tight = args; tight.max_ram = next.fixed;
    require(stage_successor(next, tight, backend, 0, [] { return false; }, &fixture) == wait_bit(ControllerWait::producer_memory) &&
        !next.metrics.members && !next.metrics.generated_members,
        "blocked successor generation did not expose its observed RAM-admission reason");
    std::promise<void> release;
    auto signal = release.get_future().share();
    group_grow::Result old_result;
    OrderedWriter writer([] {});
    auto pending = writer.submit([&] { signal.wait(); old_result = old.finish(); });
    struct Release { std::promise<void>& value; bool sent = false; ~Release() { if (!sent) value.set_value(); } } unblock{release};
    // Reenter stage service to model yielding to a different ready drive. Each
    // call retains the previous durable prefix and the same explicit grants.
    stage_successor(next, args, backend, 0, [&] { return next.metrics.members >= 2; }, &fixture);
    require(next.metrics.members >= 2 && next.metrics.members <= 3 && next.metrics.batches == next.metrics.members &&
        pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready,
        "successor did not retain its prefix while the prior final task remained live");
    auto const first_peak = next.metrics.peak_modeled_bytes;
    auto const full_reason = stage_successor(next, args, backend, 0, [] { return false; }, &fixture);
    auto const saved = ExternalEngine::inspect(next.path.string());
    require(full_reason == wait_bit(ControllerWait::group_full) && next.metrics.members == 4 && next.metrics.batches == 4 && next.metrics.max_batch_members == 1 &&
        next.metrics.generated_members == 4 && next.metrics.peak_modeled_bytes >= first_peak &&
        next.metrics.peak_modeled_bytes <= args.max_ram && saved.members == 4 && !saved.finishing &&
        saved.target_bytes < next.metrics.final_bytes && next.engine->external->can_append(),
        "successor did not durably stage beyond one RAM batch without packing its target");
    // A stopped lane must retain the pending job without keeping an inactive
    // engine in the global construction count or hiding its completed neighbor.
    { std::ofstream file(*old_owned / "preserved"); file.put('!'); }
    next.io.reset(); next.workers = 0;
    unblock.sent = true; release.set_value(); pending.get(); old_io.reset();
    auto const old_workspace = old.memory_bytes(4, true);
    old.release_external();
    require(!old.scratch_cleanup_complete && fs::exists(*old_owned / "preserved"),
        "successor terminal fixture did not preserve changed current scratch");
    old.external.reset();
    auto contents = [](fs::path const& path)
    { auto in = group_tools::input(path.string()); return std::vector<uint8_t>{std::istreambuf_iterator<char>(in), {}}; };
    struct SavedFile { std::vector<uint8_t> bytes; struct stat stamp; };
    std::map<fs::path, SavedFile> next_files;
    auto retain = [&](fs::path const& path)
    { next_files.emplace(path, SavedFile{contents(path), Construction::snapshot(path.string())}); };
    auto const checkpoint = fs::path(next.path.string() + ".external") / "checkpoint";
    retain(next.path); retain(checkpoint);
    for (auto const& entry : fs::directory_iterator(scratch)) if (entry.is_directory() && entry.path() != *old_owned)
        for (auto const& segment : fs::directory_iterator(entry.path())) if (segment.is_regular_file()) retain(segment.path());
    next.guard = snapshot(next.path, 1);
    output_auxiliary(next.path, next.guard, args.max_ram);
    next.guard.members = next.guard.pending_members = uint16_t(next.metrics.members); next.guard.pending = true;
    FarmingRun farm; farm.memory_limit = args.max_ram; farm.lanes.resize(1);
    FarmingDrivePlan layout; layout.fixed_members = true; layout.group_members = 4; layout.meta_slots = 1;
    FarmingDestination storage; storage.capacity_bytes = 128 * mib;
    farm.state.directories.push_back({targets.string(), "test", storage.capacity_bytes, 0, layout, storage});
    auto current = snapshot(current_path, 0); output_auxiliary(current_path, current, args.max_ram);
    current.members = current.pending_members = 4;
    auto& lane = farm.lanes[0]; lane.outputs = {current, next.guard};
    lane.reserved_members = lane.next_reserved_members = 4; lane.successor_ordinal = 1;
    auto remaining_root = root / "successor-remaining"; fs::create_directory(remaining_root);
    auto remaining_info = ScratchRoot::inspect(remaining_root);
    remaining_info.device = {"remaining-fs", {"remaining-drive"}, false, true};
    auto const pool_before = pool.memory_bytes();
    require(pool.try_reserve_output(3, remaining_info, 128 * mib), "remaining output reservation did not fit");
    uint64_t fixed = sum(next.fixed - old_workspace, pool.memory_bytes() - pool_before);
    size_t active = 1;
    auto memory = [&](unsigned, uint16_t) { return Engine::memory_bytes(18, cfg, 1); };
    uint64_t const base = fixed - next.engine_bytes, budget = sum(base, memory(1, 1));
    auto const plan = destination_plan(budget, base, 0, 0, 1, 1, 1, 1, memory,
        [](size_t active, unsigned, uint16_t) { return 1.0L / active; });
    require(plan.active && active >= plan.active && sum(fixed, memory(1, 1)) > budget,
        "terminal fixture did not retain a blocking active/workspace grant");
    auto const quota = pool.available_for_output(1, target_info), pool_memory = pool.memory_bytes();
    auto const staged_stats = next.engine->stats();
    auto const staged_waits = next.metrics.controller_waits.nanoseconds;
    auto const preserved = detach_successor(next, fixed, active, lane.successor_ordinal);
    lane.reserved_members = lane.next_reserved_members = 0;
    require(!next.engine && !next.io && next.detached && !lane.successor_ordinal && active < plan.active &&
        sum(fixed, memory(1, 1)) <= budget && pool.memory_bytes() == pool_memory &&
        pool.available_for_output(1, target_info) == quota && farm.membership(0, 1) == 4,
        "terminal successor kept a live grant or lost its durable reservation/certificate extent");
    require(preserved.members == 4 && preserved.final_bytes == next.guard.bytes &&
        preserved.controller_waits.nanoseconds == staged_waits &&
        preserved.generated_members == 4 && preserved.batches == 4 && preserved.staged_batches == 4 &&
        preserved.written_bytes == staged_stats.written_bytes && preserved.scratch_peak_bytes &&
        !preserved.scratch_released && preserved.peak_file_bytes >= next.guard.charge(),
        "terminal successor omitted acknowledged stage work or reported canonical bytes as published");
    for (auto const& [path, before] : next_files)
    {
        Construction::unchanged(before.stamp, path.string());
        require(contents(path) == before.bytes, "terminal successor changed checkpoint, segment or target bytes");
    }
    bool repeated = false;
    try { (void)detach_successor(next, fixed, active, lane.successor_ordinal); }
    catch (std::runtime_error const&) { repeated = true; }
    require(repeated && active < plan.active && sum(fixed, memory(1, 1)) <= budget,
        "terminal successor accounted its work or workspace twice");
    farm.check();
    { std::ofstream file(checkpoint, std::ios::binary | std::ios::trunc); file.put('!'); }
    bool audited = false;
    try { farm.check(); } catch (std::runtime_error const&) { audited = true; }
    { std::ofstream file(checkpoint, std::ios::binary | std::ios::trunc);
      auto const& bytes = next_files.at(checkpoint).bytes;
      file.write(reinterpret_cast<char const*>(bytes.data()), std::streamsize(bytes.size())); }
    require(audited, "detached successor retained a live checkpoint audit exemption");
    farm.check();
    auto remaining_cfg = cfg; remaining_cfg.memory_limit_bytes = budget - fixed;
    auto remaining_io = pool.try_io(3); require(bool(remaining_io), "remaining construction lost I/O admission");
    Construction remaining((remaining_root / "remaining.gplot").string(), remaining_cfg, false, identity());
    remaining.confirm_identity();
    require(remaining.append(input.subspan(0, 1)).accepted && remaining.finish().members == 1,
        "remaining construction could not finish within released workspace and active admission");
    remaining_io.reset();
    pool.release_after_cleanup(3, allocated_bytes(remaining_root / "remaining.gplot"));
    GroupSuccessor recovered(next.path, requested, 4, 128 * mib);
    recovered.guard = next.guard; recovered.recovery = ExternalEngine::inspect(next.path.string());
    auto saved_ordinal = std::optional<uint64_t>(1);
    auto const saved_fixed = fixed, saved_active = active;
    auto const saved_work = detach_successor(recovered, fixed, active, saved_ordinal);
    require(fixed == saved_fixed && active == saved_active && !saved_ordinal && saved_work.recovered &&
        saved_work.members == 4 && !saved_work.generated_members && !saved_work.written_bytes,
        "saved terminal successor released nonexistent workspace or counted historical work twice");
    auto resumed_cfg = cfg; resumed_cfg.workers = 2;
    auto next_io = pool.try_io(1); require(bool(next_io), "successor recovery did not acquire full I/O ownership");
    Construction resumed(next.path.string(), resumed_cfg, true, identity(), "external", next_scratch);
    auto pack_history = std::make_shared<SharedConstructionHistory>(); resumed.attach_history(pack_history);
    resumed.confirm_identity();
    require(resumed.staged_recovery && resumed.external->can_append(), "successor fixture did not reopen its acknowledged staging prefix");
    resumed.prepare_finish(); resumed.prepare_finish();
    require(resumed.fresh_external_pack && resumed.member_count() == 4 && finish_with_cpu_record(resumed, true).members == old_result.members &&
        contents(next.path) == contents(current_path) && backend.generated_members == 4,
        "successor recovery regenerated its acknowledged prefix or changed ordinary v2 bytes");
    require(pack_history->snapshot().external_packing.count == 1 && !pack_history->snapshot().shapes.count,
        "reopened staged successor lost its whole-pack rate or relaxed conservative recovery shape sampling");
    resumed.release_external(); next_io.reset();
    pool.release_after_cleanup(1, sum(allocated_bytes(next.path), resumed.remaining_metadata_bytes));

    // A stage-only singleton rejection cannot invoke Construction's full-pack
    // fallback. Its empty recoverable target and checkpoint remain untouched.
    require(pool.try_reserve_output(2, target_info, 128 * mib), "rejected successor output was not reserved");
    auto tiny = pool.grant(2, 4096, 4096, 4096, 2);
    require(bool(tiny), "rejected successor scratch fixture was not granted");
    tiny->target_gate = gate;
    GroupSuccessor rejected(targets / "rejected", requested, 1, 128 * mib);
    rejected.owner = 2; rejected.workers = rejected.batch_cap = 1; rejected.scratch = tiny;
    rejected.io = pool.try_stage_io(2); require(bool(rejected.io), "rejected successor lost stage ownership");
    rejected.engine = std::make_unique<Construction>(rejected.path.string(), cfg, false, identity(), "external", tiny);
    rejected.engine->confirm_identity();
    rejected.fixed = sum(fixture.bytes, sum(rejected.engine->memory_bytes(1, true), 4 * mib));
    auto const before = fs::file_size(rejected.path);
    auto const rejected_reason = stage_successor(rejected, args, backend, 0, [] { return false; }, &fixture);
    require(rejected_reason == wait_bit(ControllerWait::scratch_capacity) && rejected.deferred && !rejected.metrics.members && rejected.engine->external &&
        rejected.engine->external->can_append() && fs::file_size(rejected.path) == before &&
        ExternalEngine::inspect(rejected.path.string()).members == 0,
        "stage-only rejection performed full target packing or lost its recoverable owner");
    rejected.close();
    {
        // Known contribution consumers omit the optional producer cache. A
        // scratch rejection must still hash the compact batch through ordinary
        // exact admission after its recoverable empty external handoff.
        std::vector<Member> compact;
        compact.emplace_back(std::vector<uint64_t>(input[0].raw().begin(), input[0].raw().end()));
        compact.back().compact(18);
        Construction fallback(rejected.path.string(), cfg, true, identity(), "external", tiny);
        fallback.confirm_identity();
        require(fallback.append(compact, nullptr).accepted && fallback.incremental && !fallback.external && fallback.fallbacks,
            "hash-free external rejection could not use ordinary exact admission");
        fallback.finish();
        auto const reference = targets / "hash-free-reference"; direct(reference, input.subspan(0, 1), cfg);
        require(contents(rejected.path) == contents(reference), "hash-free fallback changed ordinary v2 bytes");
    }
    for (auto const* mode : {"incremental", "merged", "staged"})
    {
        auto policy = args; policy.construction = mode;
        GroupSuccessor denied(targets / (std::string("policy-") + mode), requested, 1, 128 * mib);
        auto const before = pool.memory_bytes();
        require(!prepare_successor(denied, policy, backend, pool, 3, next.fixed, 0, 1, gate, &fixture) &&
            !fs::exists(denied.path) && pool.memory_bytes() == before,
            "successor staging overrode an explicit new-group construction policy");
    }
}

void destination_admission()
{
    using namespace grow_controller_test;
    std::vector<StorageDevice> devices;
    for (size_t i = 0; i < 64; ++i)
        devices.push_back({std::to_string(i), {"drive-" + std::to_string(i)}});
    auto groups = writer_groups(devices);
    size_t const independent = *std::max_element(groups.begin(), groups.end()) + 1;
    require(independent == devices.size(), "independent destination drives were combined");
    uint64_t const budget = 14000000000ull, metadata = devices.size() * mib;
    uint64_t const generation = 4ull * 1024 * mib, member = 512 * mib;
    auto memory = [](unsigned workers, uint16_t batch)
    { return 256 * mib + workers * mib + uint64_t(batch) * 4096; };
    auto neutral = [](size_t active, unsigned, uint16_t) { return 1.0L / active; };
    auto const bounded = destination_plan(budget, metadata, generation, member, independent, 32, 64, 1, memory, neutral);
    require(bounded.active > 1 && bounded.active < devices.size() && bounded.batch && bounded.workers,
        "many destinations did not activate a bounded useful subset");
    require(bounded.active * bounded.workers <= 32 && metadata + generation + bounded.active *
        memory(bounded.workers, bounded.batch) + (bounded.active * bounded.batch - 1) * member <= budget,
        "destination leases overcommitted combined CPU or RAM");
    auto const plentiful = destination_plan(128ull * 1024 * mib, metadata, generation, member,
        independent, 32, 64, 1, memory, neutral);
    require(plentiful.active >= bounded.active && plentiful.batch >= bounded.batch,
        "additional RAM did not admit more destination work");
    auto const expensive_rewrites = destination_plan(budget, metadata, generation, member,
        independent, 32, 16, 1, memory, [](size_t active, unsigned, uint16_t batch) { return (batch < 16 ? 10000.0L : 1.0L) / active; });
    require(expensive_rewrites.active < bounded.active && expensive_rewrites.batch == 16,
        "observed rewrite costs did not favor fewer larger batches");
    devices.back().drives.clear(); groups = writer_groups(devices);
    auto const unknown = destination_plan(budget, metadata, generation, member,
        *std::max_element(groups.begin(), groups.end()) + 1, 32, 64, 1, memory, neutral);
    require(unknown.active == 1 && unknown.workers == 32, "unknown topology lost serialization or CPU capacity");
    auto const impossible = destination_plan(metadata + generation, metadata, generation, member,
        independent, 32, 64, 1, memory, neutral);
    require(!impossible.active, "destination activation ignored the next producer peak");
    auto const recovery = destination_plan(budget, metadata, generation, member,
        independent, 32, 1, 3, memory, neutral);
    require(recovery.active && recovery.batch >= 3, "saved recovery membership was reduced to the new batch bound");

    Config worker_config; worker_config.window_bytes = 65536; worker_config.codec_bytes = 4 * mib;
    auto worker_memory = [&](unsigned count, uint16_t batch)
    {
        auto cfg = worker_config; cfg.workers = count;
        return Engine::memory_bytes(18, cfg, batch);
    };
    uint64_t const worker_room = worker_memory(12, 1);
    require(worker_room < worker_memory(24, 1), "worker admission fixture lost its descriptor boundary");
    auto const fewer_workers = destination_plan(metadata + generation + worker_room, metadata, generation,
        member, 1, 24, 1, 1, worker_memory, neutral);
    require(fewer_workers.active == 1 && fewer_workers.workers == 12 && fewer_workers.batch == 1,
        "destination admission discarded a fitting intermediate worker grant");
    require(!destination_plan(metadata + generation + worker_memory(1, 1) - 1, metadata, generation,
        member, 1, 24, 1, 1, worker_memory, neutral).active,
        "worker fallback admitted a destination below its singleton workspace");

    // A successor can keep the former working batch while another owner uses
    // the genuinely released RAM. The reference ceiling is never a lease.
    uint16_t const maximum = 8, former = 6;
    uint64_t const retained_member = 8 * mib;
    auto workspace = [](uint16_t count) { return 4 * mib + uint64_t(count) * 4096; };
    uint64_t const live_room = sum(workspace(maximum), product(maximum - 1, retained_member));
    uint64_t const former_room = sum(workspace(former), product(former - 1, retained_member));
    uint64_t const released = live_room - former_room;
    auto const batch = successor_batch_cap(maximum, live_room, released, retained_member, workspace);
    require(batch == former, "successor admission did not preserve the derived pre-transition batch ceiling");
    uint64_t const fixed = sum(workspace(batch), product(batch - 1, retained_member));
    auto next_memory = [&](unsigned, uint16_t) { return released; };
    require(destination_plan(live_room, fixed, 0, 0, 1, 1, 1, 1, next_memory, neutral).active &&
        !destination_plan(live_room, sum(fixed, released), 0, 0, 1, 1, 1, 1, next_memory, neutral).active,
        "successor admission retained phantom workspace instead of leaving the released RAM available");
    require(successor_batch_cap(maximum, live_room, 0, retained_member, workspace) == maximum,
        "successor admission changed without a predecessor release");
    require(successor_batch_cap(maximum, live_room, live_room, retained_member, workspace) == 1,
        "zero reference admission disabled a newly fitting successor singleton");
    require(!successor_batch_cap(maximum, workspace(1) - 1, live_room, retained_member, workspace),
        "successor singleton fallback bypassed actual workspace admission");
}

void execution_planning()
{
    using namespace grow_controller_test;
    GenerationHistory generation;
    require(generation.member_ns() == 0, "cold generation acquired an invented rate");
    for (unsigned i = 0; i < 8; ++i) generation.add(4, 400);
    generation.add(1, 10000);
    require(generation.member_ns() == 100, "generation history lost its bounded robust rate");
    std::vector<ExecutionLane> lanes;
    for (size_t i = 0; i < 60; ++i)
    {
        ExecutionWork work; work.generation = 1000; work.cpu = 240; work.destination = 100;
        work.batches = 98; work.known = work.io_overlap = true;
        lanes.push_back({work, i});
    }
    auto memory = [](unsigned, uint16_t batch) { return mib + uint64_t(batch) * 4096; };
    auto select = [&](size_t resources)
    {
        return destination_plan(14000000000ull, mib, mib, mib, lanes.size(), 24, 16, 1, memory,
            [&](size_t active, unsigned workers, uint16_t)
            { return pipeline_execution_estimate(lanes, active, 24, workers, resources, true).completion; });
    };
    auto const producer_bound = select(lanes.size());
    require(producer_bound.active == 1,
        "shared generation was divided by destination count instead of limiting useful concurrency");
    for (auto& lane : lanes) { lane.work.generation = 1; lane.work.destination = 1000; }
    auto const independent = select(lanes.size());
    require(independent.active > 1, "independent measured outputs did not admit useful overlap");
    for (auto& lane : lanes) lane.destination = 0;
    require(select(1).active == 1, "partitions multiplied one physical output's execution capacity");
    for (size_t i = 0; i < lanes.size(); ++i)
    {
        lanes[i].destination = i; lanes[i].scratch = lanes.size();
        lanes[i].work.destination = 1; lanes[i].work.scratch_read = lanes[i].work.scratch_write = 1000;
    }
    require(select(lanes.size() + 1).active == 1, "shared scratch demand was divided by active destinations");
    lanes.back().work.known = false;
    require(!pipeline_execution_estimate(lanes, 1, 24, 24, lanes.size() + 1, true).known,
        "an unmeasured destination inherited another drive's service rate");
    lanes.back().work.known = true; lanes.back().scratch = SIZE_MAX;
    require(!pipeline_execution_estimate(lanes, 1, 24, 24, lanes.size() + 1, true).known,
        "unknown scratch attribution acquired an execution estimate");
    std::array<ExecutionReady, 3> ready{{{7, 0, 100, true}, {8, 0, 300, true}, {9, 0, 200, true}}};
    require(execution_generation_order(ready) == 8, "generation did not start the longest measured downstream tail");
    ready.back().known = false;
    require(execution_generation_order(ready) == 7, "unknown consumers changed the safe rotating generation order");
    ready.back().deferrals = ready.size();
    require(execution_generation_order(ready) == 9, "measured generation order starved a deferred destination");

    Config cfg; cfg.capacity_bytes = 1000000000000ull; cfg.window_bytes = mib;
    cfg.checkpoint_bytes = 16 * mib; cfg.workers = 24; cfg.merge_inputs = 4;
    auto const large = construction_external_forecast(28, cfg, 0, 1, 98, 1000000000, 1000000000, 4, 64);
    auto const narrow = construction_external_forecast(28, cfg, 0, 1, 98, 1000000000, 1000000000, 4, 4);
    auto const smaller = construction_external_forecast(28, cfg, 0, 1, 48, 1000000000, 1000000000, 4, 64);
    require(large.members == 98 && smaller.members == 48 && large.batches > smaller.batches &&
        large.merged.read < narrow.merged.read && large.merged.write < narrow.merged.write,
        "representative grouping and retained-reader choices lost their run and merge costs");
    long double const summary_bytes = static_cast<long double>(PlotGroupFile::getChunkCountForK(28)) *
        (sizeof(ChunkSummary) + sizeof(uint64_t));
    require(smaller.merged.read == smaller.members * 1000000000.0L + (smaller.batches - 1) * summary_bytes,
        "external final packing invented a compaction below its hard run limit");
    std::array<ConstructionRun, 4> unequal{{{1, 100}, {2, 200}, {3, 300}, {4, 400}}};
    auto const unequal_tail = construction_external_forecast(28, cfg, 10, 1, 11, 100, 100, 4, 4, unequal);
    require(unequal_tail.merged.read == summary_bytes + 1100 && unequal_tail.packing.read == 1100,
        "external forecast compacted unequal adjacent tail ranges or lost the cached committed runs");
    ConstructionHistory history;
    ConstructionSample sample; sample.elapsed_ns = 1000; sample.written_bytes = 100;
    sample.write_ns = 100; sample.read_bytes = 100; sample.read_ns = 100;
    sample.codec_bytes = 100; sample.codec_ns = 100; sample.workers = 2;
    sample.cpu_ns = 100; sample.serial_cpu_ns = 20; sample.cpu_measured = sample.serial_cpu_measured = true;
    sample.io_attributed = true;
    sample.scratch_io = {100, 100, 10, 100, 100, 1, 1, 1};
    sample.destination_io = {10, 100, 10, 10, 100, 1, 1, 1};
    history.execution_contributions.add(sample); history.execution_packing.add(sample);
    require(!history.count && construction_execution_work(large, history, 1000, true, true, true).known,
        "external execution estimates still required ordinary append history");
    history.execution_packing = {};
    require(!construction_execution_work(large, history, 1000, true, true, true).known,
        "missing completed external packing acquired a final-output rate");
    history.execution_packing.add(sample);
    history.execution_packing.samples[0].cpu_measured = false;
    require(!construction_execution_work(large, history, 1000, true, true, true).known,
        "missing thread CPU time was replaced by codec wall time");

    require(streamed_input_candidate(1, 2, true, true, true, false, false) &&
        !streamed_input_candidate(2, 2, true, true, true, false, false) &&
        !streamed_input_candidate(1, 2, false, true, true, false, false) &&
        !streamed_input_candidate(1, 2, true, false, true, false, false) &&
        !streamed_input_candidate(1, 2, true, true, false, false, false) &&
        !streamed_input_candidate(1, 2, true, true, true, true, false) &&
        !streamed_input_candidate(1, 2, true, true, true, false, true),
        "compact-input cold admission weakened physical, capacity, recovery or singleton-progress gates");
    auto retained_memory = [](unsigned workers, uint16_t batch)
    { return 256 * mib + workers * mib + uint64_t(batch) * 4096; };
    auto input_memory = [&](unsigned workers, uint16_t batch) { return sum(retained_memory(workers, batch), 16 * mib); };
    auto neutral = [](size_t active, unsigned, uint16_t) { return 1.0L / active; };
    uint64_t const budget = 14000000000ull, fixed = 60 * mib, producer = 4ull * 1024 * mib, member = 512 * mib;
    auto const ordinary = destination_plan(budget, fixed, producer, member, 60, 24, 16, 1, retained_memory, neutral);
    auto const streamed = destination_plan(budget, fixed, producer, 0, 60, 24, 16, 1, input_memory,
        [&](size_t active, unsigned workers, uint16_t batch)
        {
            return input_memory(workers, batch) <= (budget - fixed - producer) / active ?
                1.0L / active : std::numeric_limits<long double>::infinity();
        });
    require(streamed.active > ordinary.active && fixed + producer + streamed.active *
        input_memory(streamed.workers, streamed.batch) <= budget,
        "bounded compact input did not admit another writer while preserving an ordinary singleton producer");
    auto const useful = destination_plan(budget, fixed, producer, member, 60, 24, 16, 1, retained_memory,
        [](size_t active, unsigned, uint16_t batch) { return (batch < 16 ? 10000.0L : 1.0L) / active; });
    auto scratch_bytes = [](uint16_t count) { return 2 * mib + uint64_t(count) * 4 * mib; };
    uint64_t const scratch_capacity = sum(scratch_bytes(4), useful.batch);
    uint16_t const scratch_batch = admitted_batch_cap(useful.batch, 1, scratch_capacity, 0, scratch_bytes);
    auto const scratch_fitting = destination_plan(budget, fixed, producer, 0, 60, 24, scratch_batch, 1, input_memory,
        [&](size_t active, unsigned workers, uint16_t batch)
        {
            return input_memory(workers, batch) <= (budget - fixed - producer) / active ?
                1.0L / active : std::numeric_limits<long double>::infinity();
        });
    require(useful.active && scratch_batch && scratch_batch < useful.batch && scratch_fitting.active > useful.active &&
        scratch_fitting.batch <= scratch_batch && scratch_bytes(scratch_fitting.batch) <= scratch_capacity,
        "a smaller scratch-fitting batch lost the ordinary useful ceiling or rejected additional streamed writers");
    require(!admitted_batch_cap(useful.batch, 1, scratch_bytes(1) - 1, 0, scratch_bytes),
        "compact-input batch admission ignored scratch too small for an ordinary singleton");
    require(execution_singleton_fallback(100, 60, 10, 30) && !execution_singleton_fallback(100, 60, 11, 30) &&
        !execution_singleton_fallback(100, 101, 0, 0) && !execution_singleton_fallback(100, 60, 0, 41),
        "a heterogeneous peer consumed another owner's ordinary singleton fallback headroom");
    std::array<ExecutionLane, 2> mixed;
    for (auto& lane : mixed) { lane.work.generation = 1; lane.work.batches = 1; lane.work.known = lane.work.io_overlap = true; }
    mixed[0].destination = 0; mixed[0].scratch = 1; mixed[0].work.scratch_write = 1000;
    mixed[1].destination = 1; mixed[1].work.destination = 1000;
    auto const same_nvme = pipeline_execution_estimate(mixed, 2, 24, 12, 2, true);
    require(same_nvme.known && same_nvme.completion >= mixed[0].work.scratch_write + mixed[1].work.destination,
        "NVMe output and another owner's scratch multiplied the same physical resource");
    StreamingInputCohort cohort(3);
    require(cohort.stream_allowed(0) && !cohort.stream_allowed(1), "cold compact input split an existing cohort");
    cohort.admit(0, 98, 10, true, 100); cohort.admit(1, 48, 11, false, 110);
    cohort.complete(0, 98, 1000, 200, 1);
    require(!cohort.may_admit(2) && !cohort.stream_allowed(1), "compact-input cohort did not drain at durable completion");
    cohort.complete(1, 48, 500, 300, 0);
    require(cohort.calibrating() && !cohort.stream_allowed(0) && cohort.may_admit(0) && !cohort.may_admit(2),
        "compact-input trial skipped its matching ordinary baseline");
    cohort.admit(0, 98, 10, false, 400); cohort.complete(0, 98, 1000, 600, 0);
    require(!cohort.may_admit(0) && cohort.may_admit(1), "baseline admitted extra work for an already completed owner");
    cohort.admit(1, 48, 11, false, 610); cohort.complete(1, 48, 500, 900, 0);
    require(std::string(cohort.status()) == "accepted" && cohort.stream_allowed(1),
        "faster durable compact-input cohort was not retained");
    require(cohort.context_allowed(0, 98, 10) && !cohort.context_allowed(0, 48, 10) &&
        !cohort.context_allowed(0, 98, 11) && !cohort.context_allowed(2, 98, 10),
        "learned compact-input retention escaped its owner, group work or physical/options context");
    StreamingInputCohort slower(1);
    slower.admit(0, 98, 10, true, 100); slower.complete(0, 98, 1000, 500, 0);
    slower.admit(0, 98, 10, false, 600); slower.complete(0, 98, 1000, 700, 0);
    require(std::string(slower.status()) == "rejected" && !slower.stream_allowed(0),
        "slower compact input was retained after a completed ordinary baseline");
    StreamingInputCohort changed(1);
    changed.admit(0, 98, 10, true, 100); changed.complete(0, 98, 1000, 200, 0);
    changed.admit(0, 48, 10, false, 300);
    require(std::string(changed.status()) == "uncalibrated" && changed.may_admit(0) && !changed.stream_allowed(0),
        "different group work manufactured a comparable compact-input baseline");
    StreamingInputCohort resources(1);
    resources.admit(0, 98, 10, true, 100); resources.complete(0, 98, 1000, 200, 0);
    resources.admit(0, 98, 11, false, 300);
    require(std::string(resources.status()) == "uncalibrated", "different physical context trained compact-input retention");
    StreamingInputCohort incomplete(1);
    incomplete.admit(0, 98, 10, true, 100); incomplete.complete(0, 98, 1000, 200, 0);
    require(std::string(incomplete.status()) == "uncalibrated" && !incomplete.stream_allowed(0),
        "an unfinished baseline was reported as a learned compact-input gain");
    incomplete.cancel();
    require(!incomplete.calibrating() && incomplete.may_admit(0), "cancelled compact-input calibration blocked ordinary progress");
    StreamingInputCohort failed(1);
    failed.admit(0, 98, 10, true, 100); failed.complete(0, 1, 10, 200, 0);
    require(!failed.stream_allowed(0) && std::string(failed.status()) == "uncalibrated",
        "partial or failed group work trained a complete durable compact-input cohort");
    StreamingInputCohort preflight(1);
    preflight.admit(0, 98, 10, true, 100); preflight.complete(0, 98, 1000, 200, 0, false);
    require(std::string(preflight.status()) == "uncalibrated" && !preflight.stream_allowed(0),
        "a declined input preflight was reported as a measured streaming cohort");
    StreamingInputCohort missing_peer(2);
    missing_peer.admit(0, 98, 10, true, 100, 2); missing_peer.complete(0, 98, 1000, 200, 0);
    require(std::string(missing_peer.status()) == "uncalibrated" && !missing_peer.stream_allowed(0),
        "a predicted extra writer that never admitted trained compact-input retention");
}
void worker_grants(fs::path const& root, Members input, Config cfg)
{
    using namespace grow_controller_test;
    cfg.cpu_scheduler = std::make_shared<CodecScheduler>(4);
    auto contents = [](fs::path const& path)
    { auto in = group_tools::input(path.string()); return std::vector<uint8_t>{std::istreambuf_iterator<char>(in), {}}; };
    std::vector<uint8_t> expected;
    for (auto const* mode : {"incremental", "merged", "staged"})
    {
        auto const path = root / (std::string("worker-grant-") + mode);
        Construction engine(path.string(), cfg, false, identity(), mode);
        engine.confirm_identity();
        auto envelope = cfg; envelope.workers = 4;
        auto const admitted = engine.memory_for(envelope, 2, false, false);
        for (unsigned count : {1u, 4u})
        {
            engine.set_workers(count, 2, false, admitted);
            require(engine.config.workers == count && engine.append(input.subspan(count == 1 ? 0 : 2, 2)).accepted,
                "idle construction could not change its task worker grant");
        }
        bool rejected = false;
        try { engine.set_workers(0, 2, false, admitted); }
        catch (std::runtime_error const&) { rejected = true; }
        require(rejected && engine.config.workers == 4, "invalid worker grant changed construction state");
        engine.finish();
        auto const bytes = contents(path);
        if (expected.empty()) expected = bytes;
        else require(bytes == expected, "task worker grants changed ordinary v2 output bytes");
    }
    // A serial merge lease cannot silently acquire the parallel I/O window.
    auto serial = cfg; serial.workers = 1; serial.io_overlap = true;
    auto const admitted = MergeEngine::memory_bytes(18, serial, 2);
    Construction bounded((root / "worker-grant-memory").string(), serial, false, identity(), "merged");
    auto const before = bounded.config;
    bool rejected = false;
    try { bounded.set_workers(4, 2, false, admitted); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected && bounded.config.workers == before.workers,
        "parallel worker grant exceeded the admitted merge workspace");
    // A saved arena retains its own map geometry even with a larger runtime
    // capacity. Worker admission must price that restored geometry.
    auto const saved_path = root / "worker-grant-restored-arena";
    {
        Construction saved(saved_path.string(), cfg, false, identity(), "merged");
        saved.confirm_identity();
        require(saved.append(input.subspan(0, 2)).accepted, "worker resume fixture did not stage its prefix");
    }
    auto larger = cfg; larger.capacity_bytes = 64ull * 1024 * mib;
    Construction resumed(saved_path.string(), larger, true, identity(), "merged");
    auto resumed_grant = larger; resumed_grant.workers = 4;
    auto const saved_lease = resumed.memory_for(resumed_grant, 2, false, false);
    require(MergeEngine::memory_bytes(18, larger, 2) > saved_lease,
        "worker resume fixture did not distinguish saved and maximum arena maps");
    resumed.confirm_identity();
    resumed.set_workers(4, 2, false, saved_lease);
    require(resumed.append(input.subspan(2, 2)).accepted && resumed.finish().members == 4 &&
        contents(saved_path) == expected, "restored arena could not retune workers within its admitted workspace");
}
void cache_frontiers(fs::path const& root, Config cfg)
{
    cfg.release_cache = true;
    auto const path = root / "cache-frontiers.gplot";
    MergeState arena(path.string(), cfg, false, identity()); arena.confirm();
    auto record = arena.layout(cfg.window_bytes);
    uint64_t const block = record.block_bytes;
    arena.initialize(record, arena.required_bytes(record, 12 * block));
    auto& source = arena.state.runs[0]; auto& neighbor = arena.state.runs[1];
    source.count = 1; neighbor.first = 1; neighbor.count = 1;
    arena.state.run_count = 2; arena.state.members = arena.state.target.group_size = 2;
    MergeState::Writer output(arena, source, cfg.window_bytes), other(arena, neighbor, cfg.window_bytes);
    std::vector<uint8_t> expected(size_t(3 * block + 41));
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = uint8_t(i * 17 + i / block);
    auto const bytes = std::span<uint8_t const>(expected);
    output.write(bytes.first(size_t(block))); output.flush(); arena.checkpoint("cache-first"); output.evict();
    // Interleave a second stream so the physical source map is fragmented.
    other.write(bytes.first(size_t(block + 17))); other.flush();
    output.write(bytes.subspan(size_t(block), size_t(block + 29))); output.flush();
    arena.checkpoint("cache-second"); output.evict(); other.evict();
    output.write(bytes.subspan(size_t(2 * block + 29))); output.flush();
    arena.checkpoint("cache-third"); output.evict(); output.evict();

    MergeState::Reader input(arena, source, 2 * block);
    std::vector<uint8_t> actual(expected.size());
    input.read(std::span(actual).first(size_t(block + 17)));
    uint32_t const retired = source.head;
    arena.release(source, input.position()); arena.checkpoint("cache-consumed");
    // Cache advice must use the reader's saved physical map after retirement,
    // and leave its separately prefetched, unconsumed bytes available.
    input.evict(input.position());
    require(arena.allocate() == retired, "cache fixture did not recycle its consumed source");
    arena.value(uint64_t(retired) * block, uint64_t(0x12345678));
    input.read(std::span(actual).subspan(size_t(block + 17)));
    require(actual == expected && !input.remaining(), "cache frontier changed fragmented input bytes");
    input.evict(input.position()); input.evict(input.position());
    bool rejected = false;
    try { input.evict(input.position() - 1); } catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "merge cache frontier moved backwards");
}
void construction_histories()
{
    ConstructionHistories histories(8);
    ConstructionContext context{{"8:16"}, 18, 2, 0, 3, 536870912, 1048576, 16777216, 8388608, 2, true, true, true};
    auto first = histories.find(context);
    require(first && first->snapshot().count == 0 && first->snapshot().shapes.count == 0,
            "a new physical drive inherited unrelated construction observations");
    ConstructionSample sample{1000, 0, 0, 0, 0, 1, 0, 100, 1, 1000};
    first->history.add(sample);
    require(histories.find(context) == first && histories.find(context)->snapshot().count == 1,
            "compatible groups on the same physical drive lost completed evidence");
    uint64_t const before_lookup = histories.memory_bytes();
    auto unobserved = context; unobserved.workers = 1;
    require(histories.find(context, false) == first && !histories.find(unobserved, false) &&
        histories.memory_bytes() == before_lookup,
        "candidate history lookup allocated or crossed the worker compatibility boundary");
    auto other = context; other.drives = {"8:32"};
    require(histories.find(other) != first && histories.find(other)->snapshot().count == 0,
            "unrelated physical drives shared construction observations");
    other.drives.clear(); require(!histories.find(other), "unknown topology shared construction history");
    other = context; other.k = 20;
    require(histories.find(other) != first, "different plot parameters shared construction observations");
    other = context; other.compact = false;
    require(histories.find(other) != first, "different producer settings shared construction observations");
    other = context; other.checkpoint /= 2;
    require(histories.find(other) != first, "different explicit I/O settings shared construction observations");
    other = context; other.scratch_drives = {"8:48"};
    auto const demand = histories.find(other);
    require(demand != first, "external scratch shared destination-only construction observations");
    auto phase = other; phase.workers = 1;
    uint64_t const demand_memory = histories.memory_bytes();
    require(histories.reuse_external(demand, phase) == demand && !histories.find(phase, false) &&
        histories.memory_bytes() == demand_memory,
        "changed phase grant lost compatible CPU evidence or weakened ordinary worker isolation");
    std::array<ConstructionContext, 8> incompatible{phase, phase, phase, phase, phase, phase, phase, phase};
    incompatible[0].scratch_drives = {"8:64"}; incompatible[1].drives.clear();
    incompatible[2].k = 20; incompatible[3].max_ram /= 2; incompatible[4].io_overlap = false;
    incompatible[5].window /= 2; incompatible[6].checkpoint /= 2; incompatible[7].codec /= 2;
    for (auto const& candidate : incompatible)
        require(!histories.reuse_external(demand, candidate), "incompatible external phase context reused CPU evidence");

    auto cfg = configuration(); cfg.workers = 1;
    ConstructionForecast forecast;
    forecast.batches = 4; forecast.incremental.codec = 100;
    forecast.merged.codec = 10; forecast.merged.syncs = 2;
    forecast.finalization.codec = forecast.packing.codec = 8;
    forecast.finalization.syncs = forecast.layout.syncs = 2;
    require(prefer_merged(forecast, cfg, first->history), "completed codec evidence did not price merged work");
    auto const codec = construction_rates(first->history).codec;
    ConstructionSample layout{100000, 0, 0, 100000, 0, 1, 1};
    first->history.layout.add(layout);
    require(!prefer_merged(forecast, cfg, first->history) && construction_rates(first->history).codec == codec &&
        construction_rates(first->history.layout).codec == 0,
        "layout timing entered the codec model or failed to price finalization");
    first->history.layout = {};
    auto packing = sample; packing.codec_ns *= 100; packing.elapsed_ns = packing.codec_ns;
    first->history.packing.add(packing);
    require(!prefer_merged(forecast, cfg, first->history) && construction_rates(first->history).codec == codec,
        "packing throughput failed to price its own codec work separately");

    auto const ordinary = construction_forecast(18, cfg, 1, 654187, 1, 64);
    ConstructionHistory initial_history; initial_history.add(sample);
    auto const base_cost = construction_cost(ordinary.merged, cfg, construction_rates(initial_history));
    require(std::abs(construction_merged_cost(ordinary, cfg, initial_history) - base_cost) <=
        std::max(1.0L, std::abs(base_cost)) * std::numeric_limits<long double>::epsilon() * 16,
        "phase separation changed the first group's unobserved finalization estimate");

    auto const unknown = construction_forecast(18, cfg, 0, 0, 1, 64);
    auto const inherited = construction_forecast(18, cfg, 0, 0, 1, 64, 654187);
    require(!unknown.batches && inherited.batches && inherited.members == 64 && inherited.incremental.read > 0,
        "fresh-group forecasting ignored compatible canonical size evidence");
    auto const single = construction_forecast(18, cfg, 0, 0, 64, 64, 654187);
    require(single.batches == 1 && single.incremental.read == 0 && !prefer_merged(single, cfg, first->history),
        "inherited history changed fresh single-pass construction admission");
    auto fresh_config = cfg; fresh_config.checkpoint_bytes = 16 * 1024 * 1024;
    auto const fresh = construction_forecast(18, fresh_config, 0, 0, 1, 1, 654187);
    require(fresh.incremental.syncs == 11,
        "fresh ordinary forecast imposed a nonexistent old-source checkpoint frontier");
    ConstructionSample shape; shape.member_bytes = 654187; shape.chunk_bytes = 4096;
    for (size_t i = 0; i < first->history.shapes.samples.size(); ++i) first->history.shapes.add(shape);
    auto outlier = shape; outlier.member_bytes *= 100;
    first->history.shapes.add(outlier);
    require(construction_shape(first->history.shapes) == shape.member_bytes,
        "one skewed group replaced the canonical size history");
    for (size_t i = 0; i < first->history.shapes.samples.size(); ++i) first->history.shapes.add(outlier);
    require(construction_shape(first->history.shapes) == outlier.member_bytes,
        "canonical size history did not adapt to sustained change");
}
void phase_worker_demand()
{
    ConstructionHistory history;
    unsigned const workers = 8, neutral = workers / 2;
    require(external_staging_workers(workers, history) == neutral && !external_staging_workers(1, history),
        "unobserved phase demand changed the conservative worker split");
    ConstructionSample busy; busy.elapsed_ns = 1000000; busy.workers = 4;
    busy.cpu_ns = busy.elapsed_ns * busy.workers; busy.cpu_measured = true;
    auto idle = busy; idle.cpu_ns = 1;
    for (unsigned i = 0; i < 4; ++i)
    { history.external_staging.add(busy); history.external_packing.add(idle); }
    require(external_staging_workers(workers, history) == neutral + 1,
        "sustained measured staging demand did not receive its bounded extra worker");
    history.overlap_workers = workers; history.overlap_staging_workers = neutral + 1;
    history.external_packing.add(busy);
    require(external_staging_workers(workers, history) == neutral + 1,
        "one conflicting sample defeated phase-demand hysteresis");
    for (unsigned i = 0; i < 4; ++i)
    { history.external_staging.add(idle); history.external_packing.add(busy); }
    require(external_staging_workers(workers, history) == neutral - 1,
        "sustained opposite phase demand did not reverse the bounded grant");
    auto missing = busy; missing.cpu_measured = false; history.external_staging.add(missing);
    require(external_staging_workers(workers, history) == neutral,
        "unsupported thread CPU evidence masqueraded as zero demand");
}
void joined_worker_cpu(Members input)
{
    Member compact(std::vector<uint64_t>(input[0].raw().begin(), input[0].raw().end())); compact.compact(18);
    std::array<FragmentRegion, 1> regions{compact.view().region(0)};
    std::array<ChunkInput, 1> tasks{ChunkInput{0, {}, 0, regions}};
    std::array<ChunkSummary, 1> summaries;
    std::array<uint64_t, 1> lengths;
    ParallelEncoder encoder(18, 2, 8 * 1024 * 1024);
    encoder.summarize(tasks, summaries); auto const summarized = encoder.cpu_time();
    encoder.encode(tasks, lengths); auto const encoded = encoder.cpu_time();
    bool const supported = thread_cpu_nanoseconds().has_value();
    require(encoded.measured == supported && (!supported ||
        (summarized.nanoseconds && encoded.nanoseconds > summarized.nanoseconds && lengths[0])),
        "joined summary/encoding jobs omitted CPU evidence or reported unsupported zero demand");
}
void construction_selection()
{
    auto cfg = configuration();
    cfg.capacity_bytes = 256ull * 1024 * 1024; cfg.window_bytes = 1024 * 1024;
    cfg.checkpoint_bytes = 16 * 1024 * 1024; cfg.workers = 4;
    require(!prefer_merged(construction_forecast(18, cfg, 1, 654187, 64, 64), cfg),
            "one remaining batch selected merging");
    require(prefer_merged(construction_forecast(18, cfg, 1, 654187, 1, 64), cfg),
            "many repeated rewrites did not select merging");
    auto short_growth = construction_forecast(18, cfg, 1, 654187, 1, 3);
    ConstructionSample sync_bound{100000000, 0, 1000000, 99000000, 0, 654187, 11, 654187, 4};
    ConstructionSample codec_bound{100000000, 0, 1000000, 0, 0, 654187, 11, 654187, 4};
    // Parallel joining saves codec work even for this short growth, while
    // its checkpoint saving is too small to justify a sync-bound switch.
    require(!prefer_merged(short_growth, cfg, sync_bound) && prefer_merged(short_growth, cfg, codec_bound),
            "construction choice ignored measured storage and codec costs");
    cfg.window_bytes = 16 * 1024 * 1024; cfg.capacity_bytes = 64 * 1024 * 1024;
    require(!prefer_merged(construction_forecast(18, cfg, 1, 654187, 1, 64), cfg),
            "construction choice ignored merge reserve");
    cfg.window_bytes = 1024 * 1024; cfg.capacity_bytes = 128ull * 1024 * 1024 * 1024;
    require(!prefer_merged(construction_forecast(28, cfg, 1, 1000000000, 1, 5), cfg) &&
        prefer_merged(construction_forecast(28, cfg, 1, 1000000000, 1, 64), cfg),
        "k28 construction choice lost the short/long growth distinction");
    require(prefer_merged(construction_forecast(28, cfg, 60, 60000000000, 1, 64), cfg),
            "construction choice ignored the existing ordinary prefix");
    auto const two = construction_forecast(28, cfg, 1, 1000000000, 1, 64);
    cfg.merge_inputs = 4;
    auto const four_way = construction_forecast(28, cfg, 1, 1000000000, 1, 64);
    require(four_way.merged.read < two.merged.read && four_way.merged.write < two.merged.write,
            "four-input forecast did not reduce merge traffic");
    uint64_t const widest = 16 * 1024 * 1024;
    require(four_input_candidate(cfg, true, true, widest, 0) &&
        !four_input_candidate(cfg, false, false, widest, 4) &&
        !four_input_candidate(cfg, false, true, widest, 4),
        "merge input selection ignored device evidence or rotating preread width");
    cfg.window_bytes = widest;
    require(!four_input_candidate(cfg, false, true, widest, 0) &&
        four_input_candidate(cfg, false, true, widest, 1),
        "rotating merge selection ignored completed timings");
    cfg.merge_inputs = 2;
    auto const hdd_two = construction_forecast(28, cfg, 10, 10000000000ull, 9, 64);
    cfg.merge_inputs = 4;
    auto const hdd_four = construction_forecast(28, cfg, 10, 10000000000ull, 9, 64);
    ConstructionSample streaming;
    streaming.elapsed_ns = 2000000000; streaming.read_ns = streaming.write_ns = 1000000000;
    streaming.read_bytes = streaming.written_bytes = 100 * widest;
    streaming.read_calls = streaming.write_calls = 100; streaming.workers = cfg.workers;
    ConstructionHistory streaming_history; streaming_history.add(streaming);
    require(prefer_four_inputs(hdd_four, hdd_two, cfg, streaming_history),
        "rotating merge selection ignored substantial sequential traffic savings");
    auto latency = streaming;
    latency.read_ns *= 10; latency.read_bytes *= 8; latency.write_ns /= 100;
    latency.elapsed_ns = latency.read_ns + latency.write_ns;
    ConstructionHistory latency_history; latency_history.add(latency);
    require(!prefer_four_inputs(hdd_four, hdd_two, cfg, latency_history) &&
        !prefer_four_inputs(hdd_two, hdd_two, cfg, streaming_history),
        "rotating merge selection ignored refill cost or its saving margin");
    ConstructionHistory history;
    for (unsigned i = 0; i < 4; ++i) history.add(sync_bound);
    auto const stable = construction_rates(history);
    auto delayed = sync_bound; delayed.sync_ns *= 100; delayed.elapsed_ns *= 100;
    history.add(delayed);
    require(construction_rates(history).sync == stable.sync, "one delayed flush replaced the timing history");
    for (unsigned i = 0; i < 8; ++i) history.add(codec_bound);
    require(construction_rates(history).sync == 0, "timing history did not adapt to sustained change");
    cfg.capacity_bytes = 64 * 1024 * 1024 / 3; cfg.window_bytes = 16 * 1024 * 1024;
    auto const oversized = construction_forecast(18, cfg, 1, 640000, 1, 16);
    cfg.window_bytes = 4 * 1024 * 1024;
    auto const fitting = construction_forecast(18, cfg, 1, 640000, 1, 16);
    require(fitting.incremental.syncs < oversized.incremental.syncs,
            "ordinary forecast ignored a redo window that cannot fit near capacity");
    cfg.window_bytes = 65536; cfg.checkpoint_bytes = 16 * 1024 * 1024;
    cfg.capacity_bytes = 128ull * 1024 * 1024 * 1024;
    auto const large_blocks = construction_forecast(18, cfg, 1, 654187, 1, 16);
    cfg.capacity_bytes = 64 * 1024 * 1024;
    auto const small_blocks = construction_forecast(18, cfg, 1, 654187, 1, 16);
    require(small_blocks.finalization.syncs > large_blocks.finalization.syncs,
            "layout forecast ignored dependent moves between smaller blocks");
    cfg.window_bytes = 16 * 1024 * 1024; cfg.capacity_bytes = 128ull * 1024 * 1024 * 1024;
    cfg.checkpoint_bytes = 256 * 1024 * 1024;
    auto const block = MergeEngine::block_bytes(cfg), memory = MergeEngine::memory_bytes(28, cfg, 64);
    auto const short_transaction = construction_forecast(28, cfg, 1, 1000000000, 1, 64);
    cfg.checkpoint_bytes = 1024 * 1024 * 1024;
    auto const long_transaction = construction_forecast(28, cfg, 1, 1000000000, 1, 64);
    require(block == 256 * 1024 * 1024 && MergeEngine::block_bytes(cfg) == block &&
        MergeEngine::memory_bytes(28, cfg, 64) == memory,
        "longer merge transactions enlarged blocks or codec memory");
    require(long_transaction.minimum_merge_bytes > short_transaction.minimum_merge_bytes &&
        long_transaction.merged.syncs < short_transaction.merged.syncs,
        "checkpoint forecast ignored the reserve and sync tradeoff");
    auto const rates = construction_rates(sync_bound);
    require(construction_cost(long_transaction.merged, cfg, rates) < construction_cost(short_transaction.merged, cfg, rates),
        "checkpoint selection ignored measured durability cost");
    cfg.capacity_bytes = 7ull * 1024 * 1024 * 1024 / 2;
    auto const oversized_transaction = construction_forecast(28, cfg, 1, 1000000000, 1, 8);
    cfg.checkpoint_bytes = 256 * 1024 * 1024;
    auto const fitting_transaction = construction_forecast(28, cfg, 1, 1000000000, 1, 8);
    require(oversized_transaction.minimum_merge_bytes > cfg.capacity_bytes &&
        fitting_transaction.minimum_merge_bytes <= cfg.capacity_bytes &&
        !prefer_merged(oversized_transaction, cfg, sync_bound),
        "longer checkpoint bypassed merge space admission");
}
void external_batch_selection()
{
    using namespace grow_controller_test;
    Arguments args; args.batch = 8; args.member_storage = "compact";
    auto const requested = identity_from(identity());
    Backend backend; backend.mode = "gpu"; backend.generation_overlap = true;
    backend.generated_members = 8; backend.generation_ns = 8000000000ull;
    // No GPU is constructed or called. The controller uses its real member
    // bound and a modeled producer peak to exercise the actual policy helper.
    Producer producer{requested, nullptr, backend, args, args.max_ram, mib, 0};
    auto const storage = member_storage_bound(args, requested);
    producer.budget = sum(sum(producer.fixed, producer.next_peak(8)), product(7, storage));
    group_grow::ExternalEngine::BatchResources resources;
    resources.members = 8; resources.runs = resources.packets = 1;
    resources.bytes = 8 * mib; resources.source_bytes = mib / 16;
    resources.packet_bytes = 8; resources.readers = 64;
    auto cfg = configuration(); cfg.window_bytes = 16 * mib;
    ExternalBatchHistory history;
    auto select = [&](ExternalBatchHistory& observations, uint16_t maximum = 64, uint64_t retained = 0)
    { return external_batch_size(observations, resources, producer, cfg, maximum, 8, retained, 0); };
    auto sample = [](uint64_t seconds)
    {
        group_grow::ConstructionSample value; value.elapsed_ns = seconds * 1000000000ull;
        value.written_bytes = 8 * mib; value.codec_bytes = 8 * mib;
        return value;
    };
    auto add = [&](ExternalBatchHistory& observations, uint16_t count, uint64_t planning, uint64_t encoding)
    { observations.add(count, sample(planning), sample(encoding)); };
    require(select(history) == 8 && history.ceiling == 8 && !history.half_attempted,
        "external batching probed before observing its maximum batch");
    add(history, 8, 8, 8);
    auto clone = history;
    require(select(clone) == 4 && clone.half_attempted && !history.half_attempted,
        "ambiguous external planning did not permit one isolated half-batch trial");
    require(select(clone) == 8 && clone.half_attempted,
        "an unobserved or rejected external trial was repeated");
    require(select(history) == 4, "external batching did not try its fitting overlap candidate");
    for (unsigned i = 0; i < 4; ++i) add(history, 4, 4, 4);
    require(select(history) == 4 && history.selected == 4,
        "measured external overlap did not select the faster batch");
    add(history, 4, 1000, 1000);
    require(select(history) == 4, "one external timing outlier overturned the measured median");
    auto hysteresis = history; hysteresis.sizes[1].planning = {}; hysteresis.sizes[1].encoding = {};
    add(hysteresis, 4, 5, 5);
    auto undecided = hysteresis; undecided.selected = 8;
    require(select(hysteresis) == 4 && select(undecided) == 8,
        "external batch selection lost its improvement margin or incumbent hysteresis");
    hysteresis.sizes[1].planning = {}; hysteresis.sizes[1].encoding = {};
    add(hysteresis, 4, 12, 12);
    require(select(hysteresis) == 8, "slower measured external half batches were retained");
    auto serial = history; serial.sizes[0].planning = {}; serial.sizes[0].encoding = {};
    serial.sizes[1].planning = {}; serial.sizes[1].encoding = {}; serial.half_attempted = false;
    add(serial, 8, 64, 1);
    require(select(serial) == 8 && !serial.half_attempted,
        "serial external scan overhead was treated as free codec throughput");

    auto guarded = history; guarded.half_attempted = false;
    guarded.sizes[1].planning = {}; guarded.sizes[1].encoding = {};
    resources.readers = 8;
    require(select(guarded) == 8 && !guarded.half_attempted,
        "half batching crossed the external retained-reader limit");
    resources.readers = 64;
    auto const original = resources;
    resources.members = resources.packets = 64; resources.source_bytes = 1;
    require(select(guarded, 128) == 8 && !guarded.half_attempted,
        "half batching reduced the descriptor-limited final packing window");
    resources = original;
    auto refill = guarded; refill.sizes[0].encoding = {};
    auto buffered = refill;
    auto slow_write = sample(8); slow_write.write_ns = slow_write.elapsed_ns; slow_write.write_calls = 1;
    buffered.sizes[0].encoding.add(slow_write);
    require(select(buffered) == 4 && buffered.half_attempted,
        "buffered write waits were charged as unmeasured reader refill latency");
    auto larger_payload = buffered;
    auto expanded = sample(4); expanded.codec_bytes = 128 * mib;
    larger_payload.add(4, sample(4), expanded);
    require(select(larger_payload) == 8,
        "smaller batches ignored their observed additional final packing input");
    auto slow_read = sample(8); slow_read.read_ns = slow_read.elapsed_ns;
    slow_read.read_bytes = 8 * mib; slow_read.read_calls = 1;
    refill.sizes[0].encoding.add(slow_read);
    require(select(refill) == 8 && !refill.half_attempted,
        "extra external packet reads and reader refills were omitted");
    auto const budget = producer.budget;
    producer.budget = sum(sum(producer.fixed, producer.next_peak(8)), product(6, storage));
    require(select(guarded) == 8 && !guarded.half_attempted,
        "external overlap admitted two batches without their complete member headroom");
    producer.budget = budget;
    require(select(guarded, 64, storage) == 8 && !guarded.half_attempted,
        "external overlap ignored another destination's retained member");
    args.batch = 2;
    require(select(guarded) == 2 && !guarded.half_attempted,
        "an explicit upper bound already fitting two batches was needlessly halved");
    args.batch = 8;
    require(select(guarded, uint16_t(resources.members + 3)) == 3,
        "external batching split or exceeded the final partial batch");
    for (auto const& mode : {"cpu", "idle", "fixture"})
    {
        backend.mode = mode;
        require(select(guarded) == 8 && !guarded.half_attempted,
            "external batching assumed an available GPU for another backend");
    }
    backend.mode = "gpu"; backend.generation_overlap = false;
    require(select(guarded) == 8 && !guarded.half_attempted,
        "external batching ignored effective overlap off");
    backend.generation_overlap = true; args.generation_overlap = false;
    require(select(guarded) == 8, "external batching ignored explicit overlap off");
    args.generation_overlap = true; args.member_storage = "raw";
    require(select(guarded) == 8, "external batching changed raw borrowed-output admission");

    // A short group cannot train its own smaller-batch trial after its first
    // full append. Compatible successor history must reach the next group's
    // first selection, without changing the cold group's large-batch policy.
    args.member_storage = "compact"; args.batch = 9;
    backend.generated_members = 9; backend.generation_ns = 18000000000ull;
    producer.budget = sum(sum(producer.fixed, producer.next_peak(0)), product(8, storage));
    resources = {}; resources.packet_bytes = 8; resources.readers = 64;
    ExternalBatchHistory successor;
    auto next = [&](ExternalBatchHistory& observations)
    { return external_batch_size(observations, resources, producer, cfg, 16, 9, 0, 0); };
    require(next(successor) == 9 && !successor.half_attempted,
        "cold short successor selected a smaller batch without completed evidence");
    add(successor, 9, 9, 9);
    successor.source_members = 9; successor.source_bytes = 9 * (mib / 16);
    resources.members = 9; resources.runs = resources.packets = 1;
    resources.bytes = 9 * mib; resources.source_bytes = successor.source_bytes;
    require(next(successor) == 7 && !successor.half_attempted && successor.ceiling == 9,
        "short successor lost its large baseline while draining its final partial batch");
    resources = {}; resources.packet_bytes = 8; resources.readers = 64;
    auto missing_shape = successor; missing_shape.source_members = 0; missing_shape.source_bytes = 0;
    require(next(missing_shape) == 9 && !missing_shape.half_attempted,
        "fresh successor invented a source shape for its packing forecast");
    require(next(successor) == 4 && successor.half_attempted,
        "compatible short successor could not try overlap at its first append");
    require(next(successor) == 9,
        "fresh successor repeated an interrupted trial without a completed smaller append");
    add(successor, 4, 4, 4);
    require(next(successor) == 4 && successor.selected == 4,
        "completed short-successor evidence did not retain its faster batch");
    successor.sizes[1].planning = {}; successor.sizes[1].encoding = {};
    add(successor, 4, 9, 9);
    require(next(successor) == 9 && successor.selected == 9,
        "more successor overlap concealed the total cost of slower repeated appends");
    successor.prepare(8);
    require(!successor.half_attempted && !successor.source_members && !successor.source_bytes &&
        !successor.sizes[0].planning.count && !successor.sizes[1].encoding.count,
        "changed successor ceiling retained incompatible timing or shape evidence");
}

void external_finish_grants(fs::path const& root, Members input, Config cfg)
{
    using namespace grow_controller_test;
    auto directory = root / "packing-grant-scratch"; fs::create_directory(directory);
    ExternalConfig external; external.scratch.push_back({directory.string(), 128 * mib}); external.segment_bytes = mib;
    cfg.capacity_bytes = 128 * mib; cfg.memory_limit_bytes = 512 * mib;
    auto const path = root / "packing-grant", reference = root / "packing-grant-reference";
    direct(reference, input.subspan(0, 2), cfg);
    uint64_t packing = 0;
    {
        Construction engine(path.string(), cfg, false, identity(), "external", external); engine.confirm_identity();
        require(engine.append(input.subspan(0, 2)).accepted, "packing grant fixture did not stage");
        auto const staging = engine.memory_bytes(2, true);
        require(staging >= Engine::memory_bytes(18, cfg, 2, 0, 2),
            "staging released the ordinary fallback envelope before terminal packing");
        {
            auto prepared = engine.external->prepare_input(2, fragment_bound(identity_from(identity()).params) / sizeof(uint64_t));
            require(bool(prepared), "worker grant fixture could not reserve bounded input");
            auto requested = engine.config; requested.workers = cfg.workers + 1;
            auto const workspace = engine.memory_for(requested, 2, true, false);
            auto const input_bytes = ExternalEngine::input_memory_bytes(cfg, 2);
            require(staging < workspace && workspace <= sum(staging, input_bytes),
                "worker grant fixture did not distinguish codec and live input reservations");
            bool rejected = false;
            try { engine.set_workers(requested.workers, 2, true, staging); }
            catch (std::runtime_error const&) { rejected = true; }
            require(rejected && engine.config.workers == cfg.workers && engine.member_count() == 2,
                "worker retuning consumed live input's separate workspace grant");
            engine.set_workers(requested.workers, 2, true, workspace);
            engine.set_workers(cfg.workers, 2, true, staging);
        }
        OrderedWriter writer([] {});
        auto transition = writer.submit([&] { engine.prepare_finish(); }); transition.get();
        packing = engine.memory_bytes(2, true);
        require(packing < staging && packing == engine.external->required_memory(2) && !engine.can_append() &&
            ExternalEngine::inspect(path.string()).finishing,
            "joined durable packing transition did not return its released RAM grant");
        auto requested = engine.config; requested.workers = cfg.workers + 1;
        auto const retuned = engine.memory_for(requested, 2, true, false);
        require(retuned < staging, "packing worker metadata reacquired the ordinary fallback envelope");
        engine.set_workers(requested.workers, 2, true, retuned);
        require(engine.memory_for(engine.config, 2, true, false) <= retuned,
            "packing worker retune reacquired an unrelated ordinary envelope");
        bool rejected = false;
        try { (void)engine.finish_staging(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected && engine.external && engine.terminal_packing && engine.memory_bytes(2, true) == retuned,
            "terminal packing reopened ordinary admission after giving up its workspace grant");
    }
    Construction resumed(path.string(), cfg, true, identity(), "external", external); resumed.confirm_identity();
    require(resumed.memory_bytes(2, true) >= packing,
        "recovered controller skipped its ordinary handoff reservation");
    resumed.prepare_finish();
    require(resumed.memory_bytes(2, true) == packing && resumed.finish().members == 2,
        "recovered terminal pack did not reacquire its explicit live phase grant");
    auto contents = [](fs::path const& path)
    { auto in = group_tools::input(path.string()); return std::vector<uint8_t>{std::istreambuf_iterator<char>(in), {}}; };
    require(contents(path) == contents(reference), "phase RAM grants changed ordinary v2 output bytes");
    resumed.release_external();
    auto source = std::make_shared<SharedConstructionHistory>();
    Construction trained((root / "packing-grant-trained").string(), cfg, false, identity(), "external", external);
    trained.attach_history(source); trained.confirm_identity();
    require(trained.append(input.subspan(0, 2)).accepted, "CPU phase observation fixture did not stage");
    trained.prepare_finish(); trained.finish();
    auto const observed = source->snapshot();
    bool const supported = thread_cpu_nanoseconds().has_value();
    require(observed.external_staging.count == 1 && observed.external_packing.count == 1 &&
        observed.external_staging.samples[0].cpu_measured == supported && observed.external_packing.samples[0].cpu_measured == supported &&
        (!supported || (observed.external_staging.samples[0].cpu_ns && observed.external_packing.samples[0].cpu_ns)) &&
        !observed.count && !observed.contributions.count && !observed.packing.count &&
        observed.execution_contributions.count && observed.execution_packing.count &&
        observed.execution_contributions.samples[0].io_attributed && observed.execution_packing.samples[0].io_attributed,
        "joined external CPU phases omitted attributed execution evidence or contaminated ordinary append history");
    finish_with_cpu_record(trained, false, false);
    require(source->snapshot().external_packing.count == observed.external_packing.count,
        "zero-work publication added a whole-pack observation");
    trained.release_external();

    // Interrupt after the first emitted chunk is durably checkpointed. Its
    // resumed suffix still reports CPU work, but cannot train a whole pack.
    auto const interrupted = root / "packing-grant-interrupted";
    auto const child = ::fork(); require(child >= 0, "packing CPU recovery fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "external-pack-after-checkpoint", 1);
        auto frequent = cfg; frequent.checkpoint_bytes = 1;
        Construction engine(interrupted.string(), frequent, false, identity(), "external", external);
        engine.confirm_identity(); require(engine.append(input.subspan(0, 2)).accepted, "packing CPU recovery input did not stage");
        engine.prepare_finish(); engine.finish(); std::_Exit(0);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "packing CPU recovery did not interrupt an emitted prefix");
    auto partial_history = std::make_shared<SharedConstructionHistory>();
    Construction partial(interrupted.string(), cfg, true, identity(), "external", external);
    partial.attach_history(partial_history); partial.confirm_identity();
    require(partial.staged_recovery && !partial.external->can_append(), "packing CPU recovery did not restore the interrupted pack phase");
    partial.prepare_finish();
    require(!partial.fresh_external_pack && finish_with_cpu_record(partial, false).members == 2 &&
        partial.external->packing_body_ns() && !partial_history->snapshot().external_packing.count &&
        !partial_history->snapshot().shapes.count && contents(interrupted) == contents(reference),
        "partial packing trained a whole-phase rate, omitted runtime work or changed ordinary v2 bytes");
    partial.release_external();
}

void external_batch_observations(fs::path const& root, Members members, Config cfg)
{
    using namespace grow_controller_test;
    auto directory = root / "external-batch-scratch"; fs::create_directory(directory);
    ExternalConfig external; external.scratch.push_back({directory.string(), 128 * mib});
    external.segment_bytes = mib; external.packing_inputs = 8;
    cfg.capacity_bytes = 128 * mib; cfg.memory_limit_bytes = 512 * mib;
    auto source = std::make_shared<SharedConstructionHistory>();
    source->successor_batches.prepare(4); source->successor_batches.half_attempted = true;
    source->successor_batch_workers = 2;
    Construction engine((root / "external-batch-observations").string(), cfg, false, identity(), "external", external);
    engine.attach_history(source); engine.external_batches.prepare(2); engine.confirm_identity();
    auto const before = engine.external->stats();
    require(engine.append(members.subspan(0, 2)).accepted, "external observation contribution did not fit");
    require(!engine.external_batches.sizes[0].planning.count,
        "external worker published observations before its consumer joined");
    engine.observe_external(2, before);
    auto const shape = engine.external->batch_resources();
    require(shape.members == 2 && shape.runs == 1 && shape.packets == 1 && shape.bytes &&
        shape.source_bytes && shape.packet_bytes == 8 && shape.readers == 8,
        "external batch resource snapshot omitted its live packing state");
    require(engine.external_batches.sizes[0].planning.count == 1 &&
        engine.external_batches.sizes[0].encoding.count == 1 && !source->snapshot().count &&
        !source->snapshot().contributions.count, "external timings contaminated same-file construction history");
    require(source->successor_batch_workers == 2 && source->successor_batches.ceiling == 4 &&
        source->successor_batches.half_attempted && !source->successor_batches.sizes[0].planning.count,
        "foreground external observations overwrote a successor's different worker context");
    engine.shape_recorded = true;
    auto grant = cfg; grant.workers = cfg.workers + 1;
    engine.set_workers(grant.workers, 2, false, engine.memory_for(grant, 2, false, false));
    require(!engine.external_batches.ceiling && !engine.shape_recorded && !engine.shared_history,
        "worker retune retained incompatible external batch timings or shape context");
    engine.finish_staging();
    require(!engine.external && !engine.external_batches.ceiling && !engine.external_batches.sizes[0].planning.count &&
        !source->snapshot().count && !source->snapshot().contributions.count,
        "external timing evidence survived ordinary handoff");
}

void bounded_saved_reads(fs::path const& root)
{
    using namespace grow_controller_test;
    auto reject = [](auto operation, char const* message)
    {
        bool rejected = false;
        try { operation(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, message);
    };
    for (auto const& text : {std::string{}, std::string("checkpoint bytes")})
    {
        std::istringstream in(text);
        require(stream_state::read_exact(in, text.size()) == text, "bounded saved-state read changed its bytes");
    }
    std::istringstream short_input("short"), trailing_input("prefix with more bytes");
    reject([&] { (void)stream_state::read_exact(short_input, 16); }, "bounded saved-state read accepted truncation");
    reject([&] { (void)stream_state::read_exact(trailing_input, 6); }, "bounded saved-state read accepted growth");
    // An input that never reaches EOF represents an appender staying ahead of
    // a reader. Only the admitted prefix and a non-consuming probe may run.
    struct GrowingInput : std::streambuf
    {
        size_t extracted = 0, probes = 0;
        std::streamsize xsgetn(char* out, std::streamsize count) override
        { std::fill_n(out, size_t(count), 'x'); extracted += size_t(count); return count; }
        int_type underflow() override { ++probes; return traits_type::to_int_type('x'); }
    } buffer;
    std::istream growing(&buffer);
    reject([&] { (void)stream_state::read_exact(growing, 32); }, "bounded saved-state read followed an appender");
    require(buffer.extracted == 32 && buffer.probes == 1, "saved-state growth escaped the admitted read bound");

    auto const work = root / "bounded-stream-state"; fs::create_directory(work);
    stream_state::State state; state.config = {1, 2, 4096, 4096, false, false}; state.raw.resize(1);
    stream_state::save_state(work, state, "bounded-state-fixture", "bounded-state-fixture");
    auto const checkpoint = work / "checkpoint";
    auto const bytes = contents(checkpoint);
    std::string const saved(bytes.begin(), bytes.end());
    auto const restored = stream_state::load_state(work, state.config, "bounded-state-fixture", 1, mib, 1);
    require(restored.config == state.config && restored.raw.size() == state.raw.size(),
        "bounded stream checkpoint read changed recovery state");
    auto replace = [&](std::string const& text) { stream_state::atomic_save(checkpoint, text); };
    replace(saved.substr(0, saved.size() - 1));
    reject([&] { (void)stream_state::load_state(work, state.config, "bounded-state-fixture", 1, mib, 1); },
        "stream checkpoint accepted truncated bytes");
    replace(saved + "extra");
    reject([&] { (void)stream_state::load_state(work, state.config, "bounded-state-fixture", 1, mib, 1); },
        "stream checkpoint accepted trailing bytes");
    replace(std::string(512, 'a') + '\n' + saved.substr(65));
    reject([&] { (void)stream_state::load_state(work, state.config, "bounded-state-fixture", 1, mib, 1); },
        "stream checkpoint accepted an oversized checksum line");
    replace(saved);
    reject([&] { (void)stream_state::load_state(work, state.config, "bounded-state-fixture", 1, 8, 1); },
        "stream checkpoint ignored its read budget");
}

FarmingState farming_format_fixture(unsigned version)
{
    FarmingState state; state.version = version;
    state.fingerprint = std::string(64, '0'); state.job_id = std::string(64, 'a');
    state.seed = std::string(64, 'b'); state.farmer = std::string(96, 'c'); state.pool = std::string(64, 'd');
    state.seed_owner = "owner"; state.fixed_extent = true;
    state.capacity = 256; state.headroom = 32; state.max_groups = 8; state.max_total_members = 16;
    state.manifest_reserved_bytes = 4096;
    if (version >= 4) state.capacity_fill = true;
    if (version == 4) state.profile_json = "{}";
    auto& c = state.config;
    c.k = 18; c.strength = 2; c.filter_bits = 9; c.future_filter_bits = 8;
    c.reader = FarmingReader::cached; c.objective = FarmingObjective::latency;
    c.harvester_cores = 3; c.lookup_workers = 4; c.index_cache_bytes = 1024; c.memory_bytes = 8192;
    c.deadline_seconds = 8; c.challenge_seconds = 16; c.group_members_override = c.meta_slots_override = 2;
    c.meta_group_override = 1; c.fixed_meta = true; c.chain_seconds_per_member = 0.125L;
    c.index_seconds_per_entry = 0.25L; c.compressed_index_bytes_per_entry = 8; c.lookup_memory_per_member = 64;
    c.cpu_work_weight = 2; c.io_work_weight = 3; c.max_eligible_families_per_resource = 4;
    auto& e = state.estimate;
    e.feasible = true; e.index_bytes = 5; e.reader_memory_bytes = 6; e.planning_memory_bytes = 7;
    e.candidates_evaluated = 8; e.cpu_utilization = 0.5L; e.worker_utilization = 0.25L;
    e.reason = "fixture"; e.response_bound_seconds = 1; e.dynamic_work_seconds_per_sp = 2;
    e.cpu_work_seconds_per_sp = 3; e.io_work_seconds_per_sp = 4; e.max_eligible_families_per_resource = 1;
    for (size_t lane = 0; lane < 2; ++lane)
    {
        FarmingDirectory d; d.directory = lane ? "/b" : "/a"; d.filesystem = "fs";
        d.capacity_bytes = lane ? 96 : 128; d.resource = lane;
        d.directory_identity = lane ? "identity-b" : "identity-a";
        d.identity_resources = 2; d.identity_resource = lane;
        auto& p = d.layout;
        p.total_members = 3; p.files = 2; p.families = 1; p.group_bytes = 4096;
        p.group_members = p.meta_slots = 2; p.meta_bits = 8; p.meta_base = 1; p.fixed_members = true;
        auto& s = d.storage;
        s.resource = lane; s.known_device = true; s.read_bytes_per_second = 128;
        s.read_latency_seconds = 0.5L; s.read_concurrency = 1; s.capacity_bytes = d.capacity_bytes;
        s.max_file_bytes = lane ? 48 : 64; s.max_total_members = 16; s.max_groups = 8;
        if (version >= 4) { d.extents = {2, 1}; d.sealed_files = 1; }
        state.estimate.destinations.push_back(p); state.directories.push_back(std::move(d));
    }
    return state;
}

void replot_journal_protocol(fs::path const& root)
{
    using grow_controller_test::mib;
    ReplotJournal journal; journal.review_digest = std::string(64, 'a');
    journal.farm = farming_format_fixture(6); journal.farm.capacity_fill = false;
    for (auto& directory : journal.farm.directories) { directory.extents.clear(); directory.sealed_files = 0; }
    journal.retirements.resize(2);
    ReplotReview review; review.inventory_json = "{}"; review.resources_json = "{}";
    review.memory_limit_bytes = 16 * grow_controller_test::mib;
    review.farm = parse_farming_state(farming_state_text(journal.farm, true), 16 * mib);
    review.sources = {0, 1}; review.space.resize(review.farm.directories.size());
    uint64_t source = 0;
    for (size_t lane = 0; lane < review.farm.directories.size(); ++lane)
        for (uint64_t ordinal = 0; ordinal < review.farm.directories[lane].layout.files; ++ordinal)
        {
            uint64_t const count = source < review.sources.size() ? 1 : 0;
            review.batches.push_back({lane, ordinal, source, count, 4096, 8192, 16384, 4096}); source += count;
        }
    for (auto& space : review.space) space.metadata_allowance_bytes = 65536;
    auto const review_body = replot_review_text(review);
    require(replot_review_text(parse_replot_review(review_body, 16 * mib)) == review_body,
        "replot review changed canonical bytes");
    auto roundtrip = [&](ReplotJournal const& value)
    {
        auto const body = replot_journal_text(value);
        require(replot_journal_text(parse_replot_journal(body, 16 * mib)) == body, "replot journal changed canonical bytes");
    };
    roundtrip(journal);
    auto advance = [&](auto change)
    {
        auto next = journal; ++next.revision; next.parent_digest = stream_state::digest(replot_journal_text(journal)); change(next);
        validate_replot_journal(next); require(valid_replot_revision(journal, next), "legal replot revision rejected");
        roundtrip(next); journal = std::move(next);
    };
    advance([](auto& value) { value.phase = ReplotPhase::replacing; });
    auto early_retirement = journal; ++early_retirement.revision;
    early_retirement.parent_digest = stream_state::digest(replot_journal_text(journal));
    early_retirement.retirements[0].state = ReplotRetirement::claim_intent;
    require(!valid_replot_revision(journal, early_retirement), "replot retirement lacked replacement or explicit predelete state");
    advance([](auto& value) { value.retirements[0] = {ReplotRetirement::claim_intent, true}; });
    advance([](auto& value) { value.retirements[0].state = ReplotRetirement::claimed; });
    advance([](auto& value) { value.retirements[0].state = ReplotRetirement::retired; });
    advance([](auto& value) { value.verified_stage = stream_state::Stamp{1, 2, 4096, 3, 4, 5, 6}; });
    advance([](auto& value) { value.published = value.verified_stage; });
    advance([](auto& value) { value.ready = value.published; });
    advance([](auto& value) { value.retirements[1].state = ReplotRetirement::claim_intent; });
    advance([](auto& value) { value.retirements[1].state = ReplotRetirement::claimed; });
    advance([](auto& value) { value.retirements[1].state = ReplotRetirement::retired; });
    ReplotReceipt receipt{journal.review_digest, {}, 0, 0, 2, *journal.ready};
    auto const body = replot_receipt_text(receipt);
    require(replot_receipt_text(parse_replot_receipt(body)) == body, "replot receipt changed canonical bytes");
    advance([&](auto& value)
    {
        ++value.batch; value.retired_prefix += value.retirements.size(); value.retirements.clear(); value.verified_stage.reset(); value.published.reset(); value.ready.reset();
        value.receipt_digest = stream_state::digest(body);
    });
    auto invalid = journal; ++invalid.revision; invalid.parent_digest = stream_state::digest(replot_journal_text(journal));
    invalid.farm.seed = std::string(64, 'e');
    require(!valid_replot_revision(journal, invalid), "replot revision changed plot identities");
    invalid = journal; ++invalid.revision; invalid.parent_digest = stream_state::digest(replot_journal_text(journal));
    invalid.retired_prefix = 0;
    require(!valid_replot_revision(journal, invalid), "replot revision forgot outstanding retirement progress");
    auto reject = [](auto operation, char const* message)
    { bool failed = false; try { operation(); } catch (std::runtime_error const&) { failed = true; } require(failed, message); };
    reject([&] { (void)parse_replot_journal(replot_journal_text(journal), 64); }, "replot journal ignored parser budget");
    reject([&] { (void)parse_replot_receipt(body + "extra"); }, "replot receipt accepted trailing data");
    invalid = journal; invalid.retirements.resize(replot_maximum_batch_sources + 1);
    reject([&] { (void)replot_journal_text(invalid); }, "replot journal retained an unbounded active batch");
    invalid = journal; invalid.phase = ReplotPhase::complete;
    reject([&] { (void)replot_journal_text(invalid); }, "replot handoff skipped unfinished replacements");
    for (auto const* point : {"replot-journal-all-staged", "replot-journal-one-published"})
    {
        auto const base = root / point; fs::create_directory(base);
        std::vector<fs::path> directories{base / "a", base / "b"};
        for (auto const& path : directories) fs::create_directory(path);
        ReplotJournal initial; initial.review_digest = std::string(64, 'b'); initial.farm = journal.farm; initial.retirements.resize(1);
        auto replicas = load_replot_journals(directories, initial.review_digest, 16 * mib);
        require(!replicas.present, "fresh replot journal unexpectedly exists");
        publish_replot_journals(replicas, initial, 16 * mib, [] {});
        auto next = initial; ++next.revision; next.parent_digest = replicas.digest; next.phase = ReplotPhase::replacing;
        auto const child = ::fork(); require(child >= 0, "cannot fork replot journal recovery test");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1);
            publish_replot_journals(replicas, next, 16 * mib, [] {}); std::_Exit(1);
        }
        int status = 0;
        require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "replot journal did not stop at its durability boundary");
        auto recovered = load_replot_journals(directories, initial.review_digest, 16 * mib);
        require(recovered.state.revision == next.revision && recovered.digest == stream_state::digest(replot_journal_text(next)),
            "replot journal selected an older interrupted revision");
        publish_replot_journals(recovered, recovered.state, 16 * mib, [] {});
        for (auto const& replica : recovered.replicas)
            require(replica.digest == recovered.digest && replica.published && !replica.pending,
                "replot journal recovery did not finish every replica");
        reject([&] { (void)load_replot_journals(directories, std::string(64, 'c'), 16 * mib); },
            "replot journal accepted another review's authority");
    }
}

void replot_capacity_prefix()
{
    using namespace grow_controller_test;
    require(replot_metadata_remaining(64, 0) == 64 && replot_metadata_remaining(64, 24) == 40 &&
        !replot_metadata_remaining(64, 64) && !replot_metadata_remaining(64, 80),
        "live migration metadata reservation omitted future allocations or charged current blocks twice");
    require(sum(replot_metadata_remaining(64, 24), replot_metadata_remaining(64, 8)) == 96,
        "shared-filesystem migration replicas failed to reserve every unallocated metadata allowance");
    ReplotInputs input;
    input.review.farm.headroom = 0;
    FarmingDirectory destination; destination.filesystem = "output-fs";
    input.review.farm.directories = {destination};
    input.review.space = {{96, 96, 128, 32, 8}};
    input.review.sources = {0, 1};
    input.review.batches = {{0, 0, 0, 1, 64, 32, 4, 4}, {0, 1, 1, 1, 64, 32, 4, 4}};
    farm::InventoryRoot root; root.storage.filesystem = "output-fs";
    input.inventory.roots = {root}; input.inventory.plots.resize(2);
    input.inventory.plots[0].identity.allocated_bytes = 32;
    input.inventory.plots[1].identity.allocated_bytes = 64;
    require(!replot_prefix_capacity(input).empty(),
        "replacement-first prefix spent source bytes before its mapped replacement was ready");
    input.review.reclaim_before_bytes = 36;
    require(!replot_prefix_capacity(input).empty(), "pre-delete limit admitted a fraction of an indivisible old file");
    input.review.reclaim_before_bytes = 64;
    require(replot_prefix_capacity(input).empty(),
        "ready replacement failed to release its outstanding pre-delete allowance for the next batch");
    auto receipts = input;
    for (auto& batch : receipts.review.batches) batch.receipt_allowance_bytes = 12;
    require(!replot_prefix_capacity(receipts).empty(), "later replacement ignored retained receipt allocations");
    auto elsewhere = input; elsewhere.inventory.roots[0].storage.filesystem = "other-fs";
    elsewhere.review.reclaim_before_bytes = 1024;
    require(!replot_prefix_capacity(elsewhere).empty(), "retiring another filesystem supplied output free space");
    auto resources = input; resources.review.space[0].free_bytes = 1024;
    std::vector<std::vector<uint64_t>> early;
    auto requires_retirement = [&](size_t batch, auto const&, std::span<uint64_t const> selected)
    { return selected.size() == 1 && selected.front() == resources.review.sources[batch]; };
    require(replot_prefix_capacity(resources, requires_retirement, {"output-fs"}, &early).empty() &&
        early == std::vector<std::vector<uint64_t>>{{0}, {1}},
        "resource-driven retirement did not use the same bounded indivisible source mapping with ample free disk space");
    resources.review.reclaim_before_bytes = 31;
    require(!replot_prefix_capacity(resources, requires_retirement, {"output-fs"}, &early).empty() && early.front().empty(),
        "resource-driven retirement exceeded the physical early-deletion cap");
    auto no_journal = input; no_journal.review.space[0].free_bytes = 7;
    no_journal.review.reclaim_before_bytes = 1024;
    require(!replot_prefix_capacity(no_journal).empty(), "retirement began without space for durable private state");
}

void replot_destination_namespace(fs::path const& root)
{
    using namespace grow_controller_test; using namespace group_grow::farm;
    auto const destination = root / "replot-namespace", empty = root / "replot-empty";
    fs::create_directory(destination); fs::create_directory(empty);
    Arguments args; args.max_ram = 512 * mib;
    std::array<fs::path, 1> roots{destination};
    auto catalog = scan_farm_inventory(roots, replot_inventory_limits(args));
    replot_destination_coverage(args, catalog, empty, {});
    auto const file = destination / "opaque-farm-data";
    { std::ofstream out(file, std::ios::binary); out << "protected existing data"; }
    auto reject = [&](auto operation, char const* message)
    { bool failed = false; try { operation(); } catch (std::runtime_error const&) { failed = true; } require(failed, message); };
    reject([&] { replot_destination_coverage(args, catalog, destination, {}); },
        "migration accepted existing destination data omitted from inventory");
    catalog = scan_farm_inventory(roots, replot_inventory_limits(args));
    replot_destination_coverage(args, catalog, destination, {});
    { std::ofstream out(file, std::ios::app); out << " changed"; }
    reject([&] { replot_destination_coverage(args, catalog, destination, {}); },
        "migration accepted a changed inventoried destination object");
    catalog = scan_farm_inventory(roots, replot_inventory_limits(args));
    auto const metadata = destination / "explicit-inventory-input";
    { std::ofstream out(metadata, std::ios::binary); out << "typed caller-validated catalog fixture"; }
    std::array<ReplotMetadataFile, 1> supplied{{{metadata, stream_state::stamp(metadata)}}};
    replot_destination_coverage(args, catalog, destination, supplied, &catalog);
    require(catalog.plots.size() == 2 && replot_nonplot(catalog.plots.back()),
        "explicit typed metadata failed to retain a protected native-stamped observation");
    replot_destination_coverage(args, catalog, destination, {});
    { std::ofstream out(metadata, std::ios::app); out << " stale"; }
    reject([&] { replot_destination_coverage(args, catalog, destination, supplied); },
        "changed metadata received an unverified destination exemption");
}

void replot_retirement_protocol(fs::path const& root)
{
#ifdef __linux__
    using namespace group_grow::farm;
    auto const base = root / "replot-retirement"; fs::create_directory(base);
    std::string const job(64, 'd');
    auto reject = [](auto operation, char const* message)
    { bool failed = false; try { operation(); } catch (std::runtime_error const&) { failed = true; } require(failed, message); };
    auto write = [](fs::path const& path, std::string const& bytes)
    { std::ofstream out(path, std::ios::binary); out << bytes; require(bool(out), "cannot write retirement fixture"); };
    auto read = [](fs::path const& path)
    { auto bytes = contents(path); return std::string(bytes.begin(), bytes.end()); };
    auto fixture = [&](std::string const& name)
    {
        auto const directory = base / name; fs::create_directories(directory / "nested");
        // The low-level retirement protocol operates on an already-authorized
        // native object. Format eligibility is covered by inventory/controller
        // fixtures; these bytes make accidental substitution easy to detect.
        write(directory / "nested" / "source.plot", "reviewed source bytes");
        std::array<fs::path, 1> roots{directory}; auto inventory = scan_farm_inventory(roots);
        require(inventory.plots.size() == 1, "retirement fixture has an unexpected inventory");
        return inventory;
    };
    auto claim_path = [&](FarmInventory const& inventory)
    { return file_io::native_path(inventory.roots[0].path) / replot_claim_path(job, 0); };
    auto original_path = [](FarmInventory const& inventory)
    { return file_io::native_path(inventory.roots[0].path) / inventory.plots[0].relative_path; };
    auto claim = [&](FarmInventory const& inventory)
    { return claim_replot_source(inventory.roots[0], inventory.plots[0], job, 0, {}); };
    auto unlink = [&](FarmInventory const& inventory, std::function<void()> gate)
    { unlink_replot_claim(inventory.roots[0], inventory.plots[0], job, 0, {}, gate); };
    auto normal = fixture("normal"); auto const observed = normal.plots[0].identity;
    require(same_inventory_contents(claim(normal), observed) && !fs::exists(original_path(normal)) &&
        read(claim_path(normal)) == "reviewed source bytes", "claim lost or replaced the reviewed source");
    require(same_inventory_contents(claim(normal), observed), "claim could not recover an interrupted rename");
    reject([&] { unlink(normal, [] { throw std::runtime_error("not ready"); }); }, "retirement ignored a fresh readiness refusal");
    require(read(claim_path(normal)) == "reviewed source bytes", "failed readiness removed the claim");
    unsigned gates = 0; unlink(normal, [&] { ++gates; });
    require(gates == 1 && replot_claim_missing(normal.roots[0], normal.plots[0], job, 0), "retirement did not recheck readiness or finish its claim");

    auto changed = fixture("changed-source"); auto const original = original_path(changed);
    fs::rename(original, original.parent_path() / "saved.plot"); write(original, "reviewed source bytes");
    reject([&] { (void)claim(changed); }, "same-byte replacement inode was claimed as the old source");
    require(read(original) == "reviewed source bytes" && fs::exists(original.parent_path() / "saved.plot") &&
        !fs::exists(claim_path(changed)), "source mismatch changed a file");

    auto collision = fixture("claim-collision"); fs::create_directory(claim_path(collision).parent_path());
    require(!::chmod(claim_path(collision).parent_path().c_str(), 0700), "cannot protect fixture claim directory");
    write(claim_path(collision), "unrelated claim");
    reject([&] { (void)claim(collision); }, "claim overwrote a pre-existing file");
    require(read(claim_path(collision)) == "unrelated claim" && read(original_path(collision)) == "reviewed source bytes",
        "claim collision did not preserve both files");

    auto unprotected = fixture("unprotected-claim"); fs::create_directory(claim_path(unprotected).parent_path());
    require(!::chmod(claim_path(unprotected).parent_path().c_str(), 0755), "cannot set fixture claim permissions");
    reject([&] { (void)claim(unprotected); }, "retirement accepted a claim directory accessible to other users");
    require(fs::exists(original_path(unprotected)), "unprotected claim directory lost the original");

    auto alias = fixture("new-hardlink"); fs::create_hard_link(original_path(alias), original_path(alias).parent_path() / "alias.plot");
    reject([&] { (void)claim(alias); }, "source gained a hardlink after review but was still claimed");
    require(fs::exists(original_path(alias)), "hardlink rejection removed the source");
    auto changed_claim = fixture("changed-claim"); (void)claim(changed_claim);
    reject([&] { unlink(changed_claim, [&]
        {
            fs::rename(claim_path(changed_claim), claim_path(changed_claim).parent_path() / "saved-claim");
            write(claim_path(changed_claim), "substituted claim");
        }); }, "claim substituted during live readiness was unlinked");
    require(read(claim_path(changed_claim)) == "substituted claim" &&
        read(claim_path(changed_claim).parent_path() / "saved-claim") == "reviewed source bytes",
        "changed claim did not preserve both native objects");

    auto changed_root = fixture("changed-root"); auto const root_path = file_io::native_path(changed_root.roots[0].path);
    fs::rename(root_path, base / "old-root"); fs::create_directories(root_path / "nested"); write(original_path(changed_root), "new mount contents");
    reject([&] { (void)claim(changed_root); }, "retirement accepted a different root directory");
    require(read(original_path(changed_root)) == "new mount contents" &&
        read(base / "old-root" / "nested" / "source.plot") == "reviewed source bytes", "root mismatch removed a file");

#ifdef GROW_TEST_WRAP_PWRITE
    auto swapped = fixture("rename-substitution");
    retirement_rename_hook = [&](int parent, char const* name, int, char const*)
    {
        require(!::renameat(parent, name, parent, "saved-before-rename"), "cannot inject source substitution");
        write(original_path(swapped), "substituted source");
    };
    reject([&] { (void)claim(swapped); }, "source substituted between pin and rename was accepted");
    require(!retirement_rename_hook && read(claim_path(swapped)) == "substituted source" &&
        read(original_path(swapped).parent_path() / "saved-before-rename") == "reviewed source bytes",
        "failed native identity check did not preserve both files");

    auto moved_parent = fixture("moved-parent"); auto const moved_root = file_io::native_path(moved_parent.roots[0].path);
    fs::create_directory(moved_root / "other"); write(moved_root / "other" / "source.plot", "unrelated symlink target");
    retirement_rename_hook = [&](int, char const*, int, char const*)
    {
        fs::rename(moved_root / "nested", moved_root / "parked");
        fs::create_directory_symlink(moved_root / "other", moved_root / "nested");
    };
    require(same_inventory_contents(claim(moved_parent), moved_parent.plots[0].identity) && !retirement_rename_hook &&
        read(claim_path(moved_parent)) == "reviewed source bytes" &&
        read(moved_root / "other" / "source.plot") == "unrelated symlink target",
        "retirement followed a swapped ancestor instead of its pinned parent");
    unlink(moved_parent, [] {});
    require(read(moved_root / "other" / "source.plot") == "unrelated symlink target", "retirement deleted through a substituted ancestor");
#endif

    for (auto const* point : {"replot-source-renamed", "replot-before-unlink", "replot-source-unlinked"})
    {
        auto interrupted = fixture(point); auto const child = ::fork(); require(child >= 0, "cannot fork retirement recovery fixture");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1);
            (void)claim(interrupted); unlink(interrupted, [] {}); std::_Exit(1);
        }
        int status = 0;
        require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "retirement missed its injected durability boundary");
        bool const removed = std::string_view(point) == "replot-source-unlinked";
        require(!fs::exists(original_path(interrupted)) &&
            replot_claim_missing(interrupted.roots[0], interrupted.plots[0], job, 0) == removed,
            "retirement recovery confused claimed and removed sources");
        if (!removed)
        {
            require(same_inventory_contents(claim(interrupted), interrupted.plots[0].identity), "interrupted claim lost its native identity");
            unsigned fresh = 0; unlink(interrupted, [&] { ++fresh; });
            require(fresh == 1, "recovered retirement skipped its fresh readiness check");
        }
        require(replot_claim_missing(interrupted.roots[0], interrupted.plots[0], job, 0), "retirement recovery left a source behind");
    }
#else
    (void)root;
#endif
}

void farming_saved_reads(fs::path const& root)
{
    using namespace grow_controller_test;
    auto reject = [](auto operation, char const* message)
    {
        bool rejected = false;
        try { operation(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, message);
    };
    auto const state = farming_format_fixture(5);
    auto const plan = farming_state_text(state), preview = farming_preview_text(state, {}, {}, {});
    for (auto const& body : {plan, preview})
    {
        auto const path = root / (body == plan ? "bounded-farming-plan" : "bounded-farming-preview");
        auto const saved = body + stream_state::digest(body) + '\n';
        stream_state::atomic_save(path, saved);
        require(read_farming_state(path, mib).body == body, "bounded farming read changed canonical bytes");
        if (body == plan) require(load_farming_state(path, mib).job_id == state.job_id,
            "bounded private farming read changed its identity");
        else require(farming_state_text(load_farming_preview(path, mib).preview.state, true) == farming_state_text(state, true),
            "bounded preview read changed its managed layout");
        stream_state::atomic_save(path, saved.substr(0, saved.size() - 1));
        reject([&] { (void)read_farming_state(path, mib); }, "farming read accepted truncated checksum bytes");
        stream_state::atomic_save(path, saved + "extra");
        reject([&] { (void)read_farming_state(path, mib); }, "farming read accepted trailing checksum bytes");
        stream_state::atomic_save(path, saved);
        reject([&] { (void)read_farming_state(path, 16); }, "farming read ignored its read budget");
    }
}

void farming_compatibility_format(fs::path const& root)
{
    using namespace grow_controller_test;
    auto reject = [](auto operation, char const* message)
    {
        bool rejected = false;
        try { operation(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, message);
    };
    for (unsigned version : {3u, 4u, 5u, 6u})
    {
        auto state = farming_format_fixture(version);
        auto const original = farming_state_text(state);
        require(farming_state_text(parse_farming_state(original)) == original,
            "farming compatibility changed canonical saved bytes");
        state.producer_build = std::string(64, 'e');
        auto const foreign = farming_state_text(state);
        if (version < 6)
            reject([&] { (void)parse_farming_state(foreign); }, "legacy recovery accepted a foreign producer build");
        else require(farming_state_text(parse_farming_state(foreign)) == foreign,
            "compatible farming plan did not preserve foreign producer provenance");
        auto inspected = parse_farming_state(foreign, mib, FarmingStateLoad::metadata_only);
        require(inspected.version == version && inspected.producer_build == state.producer_build &&
            farming_state_text(inspected) == foreign, "metadata inspection migrated a legacy revision");
        auto const path = root / ("foreign-farming-v" + std::to_string(version));
        stream_state::atomic_save(path, foreign + stream_state::digest(foreign) + '\n');
        auto const before = contents(path);
        require(load_farming_state(path, mib, FarmingStateLoad::metadata_only).producer_build == state.producer_build &&
            contents(path) == before, "metadata inspection changed its private source");
        if (version >= 5)
        {
            Arguments args; args.command = "plan"; args.metadata_only = args.plan_only = args.json = true;
            args.manifest = path; args.max_ram = mib;
            struct Capture
            {
                std::ostringstream text; std::streambuf* previous = std::cout.rdbuf(text.rdbuf());
                ~Capture() { std::cout.rdbuf(previous); }
            };
            auto const report = [&]
            {
                Capture capture; require(print_farming_metadata(args) == 0, "metadata report failed");
                return capture.text.str();
            }();
            auto const json = profile_detail::JsonReader(report).read();
            require(profile_detail::boolean(*profile_detail::field(json, "metadata_only")) &&
                !profile_detail::boolean(*profile_detail::field(json, "live_readiness_checked")) &&
                profile_detail::boolean(*profile_detail::field(json, "plan_compatible")) == (version >= 6) &&
                report.find(state.seed) == report.npos && report.find(state.farmer) == report.npos &&
                contents(path) == before, "metadata report exposed keys, changed its source or claimed live readiness");
        }
        auto preview_state = state;
        preview_state.profile_json.clear(); // The legacy format fixture uses opaque profile bytes.
        auto const preview = farming_preview_text(preview_state, {}, {}, {});
        if (version < 6)
            reject([&] { (void)parse_farming_preview(preview); }, "legacy preview recovery accepted a foreign producer build");
        require(parse_farming_preview(preview, mib, FarmingStateLoad::metadata_only).state.producer_build == state.producer_build,
            "metadata inspection could not read a foreign-build preview");
    }
    auto const state = farming_format_fixture(6);
    auto replace = [&](std::string const& from, std::string const& to)
    {
        auto text = farming_state_text(state); auto const at = text.find(from);
        require(at != text.npos, "missing compatibility fixture field"); text.replace(at, from.size(), to); return text;
    };
    for (auto const& text : {replace("ordinary-group-farming-v6", "ordinary-group-farming-v7"),
        replace(farming_model_identity, "unknown-farming-model"),
        replace(farming_key_derivation_identity, "unknown-farming-key-derivation")})
        for (auto mode : {FarmingStateLoad::recovery, FarmingStateLoad::metadata_only})
            reject([&] { (void)parse_farming_state(text, mib, mode); },
                "unsupported farming compatibility was accepted for recovery or inspection");
    auto numeric = farming_state_text(state);
    auto const key_end = numeric.find(farming_key_derivation_identity) + std::string_view(farming_key_derivation_identity).size() + 1;
    numeric.replace(key_end, numeric.find('\n', key_end) - key_end, " 0 0 0");
    for (auto mode : {FarmingStateLoad::recovery, FarmingStateLoad::metadata_only})
        reject([&] { (void)parse_farming_state(numeric, mib, mode); },
            "unsupported numeric representation was normalized during recovery or inspection");
    auto changed = state; changed.producer_build = std::string(64, 'f');
    require(!valid_farming_extent_revision(state, changed), "farming revision rewrote its producer provenance");
}

void farming_joint_forecasts(fs::path const& root)
{
    using namespace grow_controller_test;
    auto const base = root / "joint-forecasts"; fs::create_directory(base);
    auto save = [&](fs::path const& path, FarmingPreview const& preview)
    { auto text = farming_preview_text(preview); stream_state::atomic_save(path, text + stream_state::digest(text) + '\n'); };
    auto fixture = [&](std::string const& name, char id, FarmingConfig config, uint16_t membership = 1)
    {
        auto const directory = base / name; fs::create_directory(directory);
        FarmingState state; state.job_id.assign(64, id); state.fixed_extent = state.capacity_fill = true;
        state.config = config; state.max_groups = 8; state.max_total_members = 32;
        FarmingDirectory lane; lane.directory = directory.string(); lane.directory_identity = farming_directory_identity(directory);
        lane.filesystem = storage_device(directory).filesystem; lane.capacity_bytes = 64 * mib;
        lane.storage.capacity_bytes = lane.capacity_bytes; lane.storage.resource = 0;
        lane.storage.read_bytes_per_second = 1e12L; lane.storage.read_latency_seconds = 1e-9L; lane.storage.read_concurrency = 1;
        lane.layout.files = lane.layout.families = 1; lane.layout.total_members = membership;
        lane.layout.group_members = config.group_members_override ? config.group_members_override : membership;
        lane.layout.fixed_members = bool(config.group_members_override);
        lane.layout.meta_bits = uint8_t(std::min({8u, farming_filter_bits(config), farming_filter_bits(config, true)}));
        lane.layout.meta_base = config.meta_group_override; lane.extents = {membership};
        state.estimate.destinations = {lane.layout};
        static_cast<FarmingAssessment&>(state.estimate) = farming_assess(config, {lane.storage}, {lane.layout}, {{membership}});
        require(state.estimate.feasible, "single-job joint fixture is infeasible");
        state.directories.push_back(std::move(lane));
        FarmingProfile profile; profile.config = config; profile.source = "fixture-" + name;
        profile.captured_at = "2026-11-03T00:00:00Z"; profile.host_name = name; set_farming_state_profile(state, profile);
        FarmingPreview preview{std::move(state), std::string(64, 'e'), base / (name + "-private-source"), {}};
        auto const path = base / (name + ".plan"); save(path, preview); return path;
    };
    auto options = [&](std::initializer_list<fs::path> paths)
    {
        Arguments args; args.command = "plan"; args.plan_only = args.assess_jobs = true;
        args.assessment_plans.assign(paths); args.max_ram = 64 * mib; return args;
    };
    auto reject = [&](Arguments const& args, char const* message)
    {
        bool rejected = false;
        try { (void)farming_joint_assessment(args); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, message);
    };
    FarmingConfig config; config.k = 18;
    auto const a = fixture("namespace-a", '1', config), b = fixture("namespace-b", '2', config);
    auto const before_a = contents(a), before_b = contents(b);
    auto const separate = farming_joint_assessment(options({a}));
    auto const joint = farming_joint_assessment(options({a, b}));
    require(separate.assessment.feasible && !joint.assessment.feasible &&
        joint.assessment.max_eligible_families_per_resource == 2 && joint.resources == 1 &&
        joint.sources[0].profile_source != joint.sources[1].profile_source &&
        joint.sources[0].saved.digest == read_farming_state(a, 64 * mib).digest &&
        contents(a) == before_a && contents(b) == before_b &&
        fs::is_empty(base / "namespace-a") && fs::is_empty(base / "namespace-b"),
        "joint assessment merged independent families, ignored sharing, compared provenance or wrote files");
    reject(options({a, a}), "joint assessment accepted duplicate summaries");
    auto duplicate = load_farming_preview(b, 64 * mib).preview;
    duplicate.state.job_id = load_farming_preview(a, 64 * mib).preview.state.job_id;
    duplicate.state.revision = 2; duplicate.state.parent_digest.assign(64, 'f');
    auto const duplicate_path = base / "duplicate-job.plan"; save(duplicate_path, duplicate);
    reject(options({a, duplicate_path}), "joint assessment counted revisions of the same job independently");
    auto alias = load_farming_preview(b, 64 * mib).preview;
    alias.state.directories = load_farming_preview(a, 64 * mib).preview.state.directories;
    auto const alias_path = base / "alias.plan"; save(alias_path, alias);
    reject(options({a, alias_path}), "joint assessment accepted duplicate destination aliases");

    config.max_eligible_families_per_resource = 4; config.harvester_cores = config.lookup_workers = 1;
    config.index_seconds_per_entry = 0; config.chain_seconds_per_member = 5;
    auto const cpu_a = fixture("cpu-a", '3', config), cpu_b = fixture("cpu-b", '4', config);
    auto const cpu = farming_joint_assessment(options({cpu_a, cpu_b}));
    require(!cpu.assessment.feasible && cpu.assessment.reason.find("response deadline") != std::string::npos,
        "joint assessment gave each job its own CPU grant");
    auto changed = load_farming_preview(cpu_b, 64 * mib).preview;
    ++changed.state.config.harvester_cores; changed.state.profile_metadata.reset();
    auto const changed_path = base / "different-profile.plan"; save(changed_path, changed);
    reject(options({cpu_a, changed_path}), "joint assessment normalized incompatible saved service profiles");

    config.chain_seconds_per_member = 0.004L; config.reader = FarmingReader::cached;
    config.index_cache_bytes = uint64_t(1) << (config.k - 3);
    FarmingDestination storage; storage.capacity_bytes = 64 * mib;
    FarmingDrivePlan layout; layout.files = layout.total_members = layout.group_members = layout.families = 1;
    layout.meta_bits = uint8_t(std::min({8u, farming_filter_bits(config), farming_filter_bits(config, true)}));
    config.memory_bytes = farming_assess(config, {storage}, {layout}, {{1}}).reader_memory_bytes;
    auto const memory_a = fixture("memory-a", '5', config), memory_b = fixture("memory-b", '6', config);
    auto const shared_memory = farming_joint_assessment(options({memory_a, memory_b}));
    require(!shared_memory.assessment.feasible && shared_memory.assessment.reason.find("working memory") != std::string::npos,
        "joint assessment duplicated the declared cache or reader-memory grant");

    config.reader = FarmingReader::stateless; config.memory_bytes = 512 * mib;
    auto const forecast_a = fixture("offline-a", '7', config, 2), forecast_b = fixture("offline-b", '8', config);
    auto first = load_farming_preview(forecast_a, 64 * mib).preview;
    first.state.directories[0].extents = {1}; save(forecast_a, first);
    auto second = load_farming_preview(forecast_b, 64 * mib).preview;
    second.state.directories[0].extents = {3}; save(forecast_b, second);
    fs::rename(base / "offline-a", base / "offline-a-unavailable");
    auto const offline = farming_joint_assessment(options({forecast_a, forecast_b}));
    require(offline.offline_destinations == 1 && offline.resources == 1 &&
        offline.sources[0].members == 2 && offline.sources[1].members == 3,
        "joint assessment dropped an offline forecast or a larger admitted extent");
    config.group_members_override = 2;
    auto const groups_a = fixture("group-size-a", '9', config, 2);
    config.group_members_override = 3; config.objective = FarmingObjective::latency;
    auto const groups_b = fixture("group-size-b", 'a', config, 3);
    require(farming_joint_assessment(options({groups_a, groups_b})).assessment.feasible,
        "joint assessment rejected compatible saved group-size or objective choices");
    auto limited = options({a}); limited.max_ram = 3 * mib;
    require(farming_joint_assessment(limited).planning_memory_bytes <= limited.max_ram,
        "single joint source exceeded its bounded parser/model grant");
    limited.assessment_plans.push_back(b);
    reject(limited, "joint assessment applied --max-ram independently to each source");
    auto excessive = first; excessive.state.capacity_fill = false; excessive.state.directories[0].extents.clear();
    excessive.state.directories[0].layout.files = farming_detail::maximum_files + 1;
    excessive.state.directories[0].layout.total_members = farming_detail::maximum_files + 1;
    excessive.state.directories[0].layout.group_members = 1;
    auto const excessive_path = base / "excessive-files.plan"; save(excessive_path, excessive);
    reject(options({excessive_path}), "joint assessment expanded a forecast beyond the file work bound");
}

void farming_reader_scope()
{
    using namespace grow_controller_test;
    auto state = farming_format_fixture(6); state.config.fixed_meta = false;
    FarmingProfile profile; profile.config = state.config; profile.source = "foreign baseline snapshot";
    profile.host_name = "foreign harvester"; profile.captured_at = "2026-10-01T00:00:00Z";
    profile.chia_version = "foreign snapshot"; profile.inventory_available = true; profile.inventory_v1_files = 7;
    ReaderCalibrationScope scope; scope.k = state.config.k; scope.strength = state.config.strength;
    scope.reader = state.config.reader; scope.margin = std::nextafter(2.0L, 3.0L);
    scope.host_name = "local calibration machine"; scope.captured_at = "2026-11-03T00:00:00Z";
    profile.reader_calibration = scope;
    set_farming_state_profile(state, profile);
    auto const text = farming_state_text(state);
    auto const restored = parse_farming_state(text);
    auto const recovered = farming_state_profile(restored);
    require(recovered && recovered->reader_calibration == profile.reader_calibration &&
        recovered->source == profile.source &&
        recovered->host_name == profile.host_name && recovered->captured_at == profile.captured_at &&
        recovered->chia_version == profile.chia_version && recovered->inventory_v1_files == profile.inventory_v1_files &&
        farming_state_text(restored) == text, "scoped reader calibration changed across private recovery");
    std::ostringstream json; write_farming_profile_json(json, *recovered);
    std::istringstream in(json.str()); auto const exported = read_farming_profile(in);
    require(exported.reader_calibration == profile.reader_calibration,
        "re-exported farming profile lost its reader calibration scope");
    auto foreign = state; foreign.producer_build.assign(64, 'e');
    auto const foreign_text = farming_state_text(foreign);
    require(farming_state_text(parse_farming_state(foreign_text)) == foreign_text &&
        farming_state_profile(parse_farming_state(foreign_text))->reader_calibration == profile.reader_calibration,
        "compatible foreign producer changed scoped calibration or canonical bytes");
    auto reject = [](auto operation, char const* message)
    {
        bool rejected = false;
        try { operation(); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, message);
    };
    auto legacy = state; legacy.version = 5;
    reject([&] { (void)farming_state_text(legacy); }, "legacy farming serialization discarded a reader scope");
    reject([&] { (void)farming_state_profile(legacy); }, "legacy farming reconstruction accepted a reader scope");
    legacy.profile_metadata.reset(); auto const before = farming_state_text(legacy);
    reject([&] { set_farming_state_profile(legacy, profile); }, "scoped profile attached to a legacy farming plan");
    require(farming_state_text(legacy) == before, "rejected legacy profile attachment changed its saved state");
    auto mismatch = state; mismatch.config.k = 28;
    reject([&] { (void)farming_state_text(mismatch); }, "private state serialized reader costs for a different k");
    reject([&] { (void)farming_state_profile(mismatch); }, "private profile reconstructed reader costs for a different k");
    mismatch = state; mismatch.config.reader = FarmingReader::stateless;
    reject([&] { (void)farming_state_profile(mismatch); }, "private profile reconstructed reader costs for a different reader");
    Arguments args; args.k = profile.config.k; args.strength = profile.config.strength;
    args.farming_profile = std::make_shared<FarmingProfile const>(profile);
    require(farming_configuration(args).k == profile.config.k, "matching reader scope was rejected");
    args.k.reset();
    reject([&] { (void)farming_configuration(args); }, "k18 reader calibration silently applied to default k28");
    args.k = profile.config.k; args.farming_reader = "stateless";
    reject([&] { (void)farming_configuration(args); }, "cached reader calibration silently applied to stateless lookup");
}

void farming_legacy_format()
{
    // These literal legacy records come from the pre-existing serializer.
    // Runtime build/model identities remain pinned by their production macros.
    for (unsigned version : {3u, 4u})
    {
        auto state = farming_format_fixture(version);
        std::ostringstream expected;
        expected << "ordinary-group-farming-v" << version << '\n' << std::quoted(GROW_IDENTITY) << '\n'
            << std::quoted(farming_model_identity) << '\n' << std::quoted(std::string(64, '0')) << '\n'
            << std::quoted(std::string(64, 'a')) << " 1 \"\"\n"
            << std::quoted(std::string(64, 'b')) << " \"owner\" " << std::quoted(std::string(96, 'c')) << ' '
            << std::quoted(std::string(64, 'd')) << " 0 1\n256 32 8 16\n";
        if (version == 4) expected << "1 \"{}\"\n";
        expected << "4096\n18 2 9 8 1 1 3 4 1024 8192 8 16 2 2 1 1 0.125 0.25 8 64 2 3 4\n"
            << "1 5 6 7 8 0.5 0.25\n\"fixture\" 1 2 3 4 1\n2\n"
            << "\"/a\" \"fs\" 128 0\n\"identity-a\" 2 0 0 0\n3 2 1 4096 2 2 8 1 1 0 1\n1 0 128 0.5 1\n128 64 16 8\n";
        if (version == 4) expected << "1 \"\" 2 2 2 1 1 1\n";
        expected << "\"/b\" \"fs\" 96 1\n\"identity-b\" 2 1 0 0\n3 2 1 4096 2 2 8 1 1 0 1\n1 0 128 0.5 1\n96 48 16 8\n";
        if (version == 4) expected << "1 \"\" 2 2 2 1 1 1\n";
        auto const text = farming_state_text(state), legacy = expected.str();
        require(text == legacy && stream_state::digest(text) == stream_state::digest(legacy),
            "legacy farming serialization changed its canonical revision digest");
        auto const restored = parse_farming_state(text);
        require(restored.version == version && farming_state_text(restored) == legacy,
            "legacy farming roundtrip migrated or changed a saved revision");
        require(farming_state_text(parse_farming_state(text + '\n')) == legacy,
            "legacy noncanonical whitespace changed the canonical saved revision");
    }
}

void farming_compact_format()
{
    auto state = farming_format_fixture(5);
    state.config.fixed_meta = false;
    auto& different = state.directories.back().storage;
    different.known_device = false; different.nonrotating = true;
    different.read_bytes_per_second = 192; different.read_latency_seconds = 0.25L; different.read_concurrency = 3;
    different.max_total_members = 15; different.max_groups = 7;
    state.config.chain_seconds_per_member = std::nextafter(0.125L, 1.0L);
    state.config.index_seconds_per_entry = std::nextafter(0.25L, 1.0L);
    FarmingProfile profile; profile.config = state.config;
    profile.source = "detected"; profile.captured_at = "2026-11-03T00:00:00Z";
    profile.chia_version = "fixture version"; profile.host_name = "host é \"quoted\"";
    profile.config_status = "configuration read"; profile.inventory_status = "complete";
    profile.detected_logical_cpus = 12; profile.detected_available_memory_bytes = 32768;
    profile.inventory_available = true; profile.inventory_v1_files = 11; profile.inventory_v2_files = 12;
    profile.inventory_v2_members = 13; profile.inventory_failed_files = 14; profile.inventory_no_key_files = 15;
    profile.assumptions = {"line one\nline two", "path \\ quote \"", "Unicode é"};
    set_farming_state_profile(state, profile);
    auto const text = farming_state_text(state);
    auto restored = parse_farming_state(text);
    require(restored.version == 5 && restored.profile_json.empty() && restored.config == state.config &&
        same_farming_profile_metadata(state, restored) && farming_state_text(restored) == text,
        "compact farming roundtrip changed configuration, provenance or canonical bytes");
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
        require(restored.directories[lane].storage == state.directories[lane].storage &&
            restored.directories[lane].layout == state.directories[lane].layout &&
            restored.directories[lane].extents == state.directories[lane].extents,
            "compact storage overrides changed a heterogeneous drive or frozen assignment");
    auto const public_profile = farming_state_profile(restored);
    std::ostringstream before, after;
    write_farming_profile_json(before, profile); write_farming_profile_json(after, *public_profile);
    require(before.str() == after.str(), "compact private metadata changed public profile JSON or provenance");
    auto changed = restored;
    changed.config.lookup_workers = 7;
    require(farming_state_profile(changed)->config.lookup_workers == 7,
        "restored public profile retained a second authoritative farming configuration");
    require(farming_state_text(parse_farming_state(text + '\n')) == text,
        "compact noncanonical whitespace changed the canonical saved revision");

    auto reject = [](auto action, char const* message)
    {
        bool refused = false;
        try { action(); } catch (std::exception const&) { refused = true; }
        require(refused, message);
    };
    reject([&] { (void)farming_state_profile(restored, 1); },
        "compact profile restoration bypassed its memory budget");
    reject([&] { (void)parse_farming_state(text, text.size() * 4); },
        "compact parsing bypassed its input memory budget");
    reject([&] { auto p = profile; ++p.config.lookup_workers; set_farming_state_profile(changed, p); },
        "compact profile accepted a conflicting farming configuration");
    reject([&] { auto p = profile; p.assumptions.resize(65); set_farming_state_profile(state, p); },
        "compact profile accepted an unbounded assumption count");
    reject([&] { auto p = profile; p.assumptions = {std::string(farming_profile_metadata_bytes, 'x')};
        set_farming_state_profile(state, p); }, "compact profile accepted metadata beyond its byte bound");

    auto replace = [&](std::string const& from, std::string const& to)
    {
        auto malformed = text; auto const position = malformed.find(from);
        require(position != std::string::npos, "compact malformed-record fixture is missing");
        malformed.replace(position, from.size(), to); return malformed;
    };
    std::string const defaults = "511 1 0 128 0.5 1 128 64 16 8\n";
    reject([&] { (void)parse_farming_state(replace(defaults, "0\n")); },
        "compact storage accepted an incomplete shared record");
    reject([&] { (void)parse_farming_state(replace(defaults, "512\n")); },
        "compact storage accepted an unknown shared-field mask");
    reject([&] { (void)parse_farming_state(replace(defaults, "4294967296\n")); },
        "compact storage accepted an overflowing shared-field mask");
    reject([&] { (void)parse_farming_state(replace(defaults, "")); },
        "compact storage accepted a missing shared record");
    std::string const overrides = "511 0 1 192 0.25 3 96 48 15 7\n";
    reject([&] { (void)parse_farming_state(replace(overrides, "512\n")); },
        "compact storage accepted unknown per-drive override fields");
    reject([&] { (void)parse_farming_state(replace(overrides, "511 0 1 192 0.25 3 18446744073709551616 48 15 7\n")); },
        "compact storage accepted an overflowing per-drive field");
    auto const position = text.find(overrides);
    require(position != std::string::npos, "compact overrides are missing from the heterogeneous fixture");
    reject([&] { (void)parse_farming_state(text.substr(0, position) + "511 0 1 192\n"); },
        "compact storage accepted truncated per-drive override values");
}

void controller_wakeups()
{
    ControllerWake wake;
    auto const observed = wake.generation();
    require(!wake.wait_for(observed, std::chrono::milliseconds(0)), "idle controller fabricated a notification");
    wake.notify();
    require(wake.wait_for(observed, std::chrono::milliseconds(0)),
        "controller lost a notification between its readiness scan and wait");
    for (bool fail : {false, true})
    {
        auto const before = wake.generation();
        std::promise<void> notified, release;
        auto notification = notified.get_future(); auto released = release.get_future().share();
        std::atomic<bool> joined = false;
        auto pending = NotifiedFuture<int>::submit(wake, [fail]
        {
            if (fail) throw std::runtime_error("injected receiver failure");
            return 7;
        }, [&](auto task)
        {
            return std::async(std::launch::async, [&, task = std::move(task)]() mutable
            {
                std::exception_ptr failure;
                try { task(); } catch (...) { failure = std::current_exception(); }
                notified.set_value(); released.wait(); joined = true;
                if (failure) std::rethrow_exception(failure);
            });
        });
        bool const notified_before_return = notification.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
        bool const woken = wake.wait_for(before, std::chrono::milliseconds(0));
        bool const ready = pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        release.set_value();
        bool propagated = false;
        try { propagated = pending.get() == 7 && !fail; }
        catch (std::runtime_error const& error)
        { propagated = fail && std::string(error.what()) == "injected receiver failure"; }
        require(notified_before_return && woken && ready && propagated && joined && !pending.valid(),
            "receiver notification preceded result publication or lost failure/borrower cleanup");
    }
}

void replot_publication_names(fs::path const& root, Members input)
{
    using namespace grow_controller_test;
    auto const directory = root / "replot-publication-names"; fs::create_directory(directory);
    auto cfg = configuration(); cfg.capacity_bytes = 128 * mib; cfg.memory_limit_bytes = 512 * mib;
    auto const ordinary = group_path(directory, 0); direct(ordinary, input.subspan(0, 2), cfg);
    auto const before = contents(ordinary); auto const job = std::string(64, 'a');
    auto const published = group_path(directory, 0, job); fs::rename(ordinary, published);
    auto const saved = snapshot(published, 0);
    require(saved.published_plot2 && group_tools::hex(saved.publication_job) == job && group_output_path(directory, saved) == published &&
        rollover_entry(directory, published, 512 * mib) == 0 && contents(published) == before,
        "migration publication changed bytes or lost its canonical name");
    Arguments args; args.max_ram = 512 * mib; args.workers = 1; args.max_groups = 4; args.max_total_members = 8; args.headroom = 0;
    auto fixture = [&]
    {
        FarmingRun run; run.state = farming_format_fixture(6); run.state.capacity_fill = false; run.state.job_id = job;
        run.state.directories.resize(1); auto& d = run.state.directories.front(); d.directory = directory.string();
        d.capacity_bytes = 128 * mib; d.layout.files = 1; d.layout.total_members = d.layout.group_members = 2;
        d.extents.clear(); d.sealed_files = 0; run.lanes.resize(1); run.lanes.front().outputs.push_back(saved);
        run.available = {1}; run.memory_limit = args.max_ram; run.metadata = mib; return run;
    };
    std::vector<fs::path> const directories{directory}; std::vector<StorageDevice> const devices{{"fs", {"drive"}}};
    std::vector<size_t> const physical{0}; std::map<std::string, uint64_t> const spaces{{"fs", 128 * mib}};
    auto valid = fixture(); valid.audit_outputs(args, nullptr, directories, devices, physical, spaces, 1);
    require(valid.lanes.front().ordinal == 1 && valid.path(0) == group_path(directory, 1),
        "complete published fixed group was scheduled for regrowth");
    auto wrong = fixture(); wrong.state.job_id = std::string(64, 'b'); bool rejected = false;
    try { wrong.audit_outputs(args, nullptr, directories, devices, physical, spaces, 1); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected && contents(published) == before, "wrong migration output nonce was accepted or changed");
    fs::copy_file(published, ordinary); rejected = false;
    try { check_outputs(directory, std::span(&saved, 1), args.max_ram); } catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "ordinary and published aliases for one ordinal were accepted");
}

void farming_parallel_audits(fs::path const& root, Members input)
{
    using namespace grow_controller_test;
    std::vector<fs::path> directories{root / "audit-first", root / "audit-second"};
    std::vector<StorageDevice> devices{{"first-fs", {"first-drive"}}, {"second-fs", {"second-drive"}}};
    std::vector<size_t> const physical{0, 1}, shared{0, 0};
    std::map<std::string, uint64_t> spaces{{"first-fs", 128 * mib}, {"second-fs", 128 * mib}};
    auto write_cfg = configuration(); write_cfg.capacity_bytes = 128 * mib; write_cfg.memory_limit_bytes = 512 * mib;
    for (auto const& directory : directories)
    { fs::create_directory(directory); direct(group_path(directory, 0), input.subspan(0, 2), write_cfg); }
    auto const first_path = group_path(directories[0], 0), second_path = group_path(directories[1], 0);
    auto const first_bytes = contents(first_path), second_bytes = contents(second_path);
    Arguments args; args.max_ram = 512 * mib; args.window = 65536; args.checkpoint_bytes = mib;
    args.workers = 2; args.max_groups = 4; args.max_total_members = 8; args.headroom = 0;
    auto fixture = [&]
    {
        FarmingRun run; run.state = farming_format_fixture(5); run.state.capacity_fill = false;
        run.metadata = 2 * mib; run.memory_limit = args.max_ram; run.metadata_workers = 2;
        run.lanes.resize(directories.size()); run.available.assign(directories.size(), 1);
        for (size_t lane = 0; lane < directories.size(); ++lane)
        {
            auto& d = run.state.directories[lane]; d.directory = directories[lane].string(); d.capacity_bytes = 128 * mib;
            d.layout.files = 1; d.layout.total_members = d.layout.group_members = 2;
            run.lanes[lane].outputs.push_back(snapshot(group_path(directories[lane], 0), 0));
        }
        return run;
    };
    require(!Engine::pending(first_path.string()) && Engine::pending((root / "missing-audit").string()),
        "ordinary audit probe did not distinguish a complete file from an ambiguous path");
    auto parallel = fixture();
    std::array<std::promise<void>, 2> started;
    auto first = started[0].get_future(), second = started[1].get_future();
    std::promise<void> release; auto released = release.get_future().share();
    FarmingStageSubmit submit = [&](size_t lane, std::function<void()> operation)
    {
        return std::async(std::launch::async, [&, lane, operation = std::move(operation)]
        { started[lane].set_value(); released.wait(); operation(); });
    };
    auto auditing = std::async(std::launch::async, [&]
    { parallel.audit_outputs(args, nullptr, directories, devices, physical, spaces, directories.size(), submit); });
    bool const overlap = first.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
        second.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    release.set_value(); auditing.get();
    require(overlap && parallel.planning_peak <= args.max_ram && parallel.audited.read_bytes &&
        parallel.lanes[0].completed_members == 2 && parallel.lanes[1].completed_members == 2,
        "bounded independent-drive audits lost overlap, results or RAM accounting");
    size_t submissions = 0;
    FarmingStageSubmit counted = [&](size_t, std::function<void()> operation)
    { ++submissions; return std::async(std::launch::async, std::move(operation)); };
    auto serial = fixture(); serial.audit_outputs(args, nullptr, directories, devices, shared, spaces, directories.size(), counted);
    require(!submissions && serial.audited.read_bytes == parallel.audited.read_bytes,
        "shared physical storage launched parallel audits or changed complete index coverage");
    auto small = args; small.max_ram = 96 * mib; small.window = 32 * mib;
    auto bounded = fixture(); bounded.audit_outputs(small, nullptr, directories, devices, physical, spaces, directories.size(), counted);
    require(!submissions && bounded.lanes[0].completed_members == 2 && bounded.lanes[1].completed_members == 2,
        "tight RAM rejected the original serial audit envelope or launched unadmitted peers");
    auto restore = [](fs::path const& path, std::vector<uint8_t> const& bytes)
    { std::ofstream out(path, std::ios::binary | std::ios::trunc); out.write(reinterpret_cast<char const*>(bytes.data()), bytes.size()); };
    { std::ofstream out(first_path, std::ios::binary | std::ios::app); out.put('!'); }
    require(!Engine::pending(first_path.string()), "orphan-tail fixture unexpectedly has a durable checkpoint");
    auto orphan = fixture(); orphan.audit_outputs(args, nullptr, directories, devices, physical, spaces, directories.size());
    require(orphan.lanes[0].outputs[0].pending && orphan.lanes[0].reserved_members == 2 &&
        fs::file_size(first_path) == first_bytes.size() + 1 && orphan.lanes[1].completed_members == 2,
        "parallel ordinary discovery rejected or changed an orphan initialization tail");
    restore(first_path, first_bytes);
    auto set_byte = [](fs::path const& path, size_t offset, uint8_t value)
    { std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out); out.seekp(offset); out.put(char(value)); };
    set_byte(first_path, offsetof(Header, version), 0xf1);
    require(Engine::pending(first_path.string()), "finalizing marker was admitted as a complete ordinary audit");
    auto finalizing = fixture(); finalizing.audit_outputs(args, nullptr, directories, devices, physical, spaces, directories.size(), counted);
    require(!submissions && finalizing.lanes[0].outputs[0].pending,
        "finalizing recovery lost its serial admission or pending identity");
    restore(first_path, first_bytes);
    auto const child = ::fork(); require(child >= 0, "pending audit fixture fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "initialized", 1);
        Engine output(first_path.string(), write_cfg, true); output.confirm_identity();
        (void)output.append(input.subspan(2, 4)); std::_Exit(0);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86 &&
        Engine::pending(first_path.string()), "ordinary checkpoint footer escaped the conservative audit probe");
    auto const pending_bytes = contents(first_path);
    auto pending = fixture(); pending.state.directories[0].layout.total_members = pending.state.directories[0].layout.group_members = 6;
    pending.audit_outputs(args, nullptr, directories, devices, physical, spaces, directories.size(), counted);
    require(!submissions && pending.lanes[0].outputs[0].pending && pending.lanes[0].reserved_members == 6 &&
        contents(first_path) == pending_bytes, "pending batch audit lost its full original memory envelope or changed recovery bytes");
    restore(first_path, first_bytes);
    set_byte(first_path, offsetof(Header, version), 0xfe);
    auto corrupt = fixture(); bool rejected = false;
    try { corrupt.audit_outputs(args, nullptr, directories, devices, physical, spaces, directories.size(), counted); }
    catch (std::runtime_error const&) { rejected = true; }
    require(rejected && !submissions, "corrupt private marker bypassed serial protocol validation");
    restore(first_path, first_bytes);
    auto failing = fixture();
    std::promise<void> failed, finish_peer;
    auto failure = failed.get_future(); auto peer = finish_peer.get_future().share();
    std::atomic<bool> joined = false;
    FarmingStageSubmit fail = [&](size_t lane, std::function<void()> operation)
    {
        return std::async(std::launch::async, [&, lane, operation = std::move(operation)]
        {
            if (!lane) { peer.wait(); operation(); joined = true; return; }
            set_byte(second_path, offsetof(Header, k), 28);
            try { operation(); } catch (...) { failed.set_value(); throw; }
        });
    };
    auto failed_audit = std::async(std::launch::async, [&]
    {
        try { failing.audit_outputs(args, nullptr, directories, devices, physical, spaces, directories.size(), fail); }
        catch (std::runtime_error const& error) { return std::string(error.what()); }
        return std::string{};
    });
    bool const observed_failure = failure.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    bool const draining = failed_audit.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    finish_peer.set_value(); auto const error = failed_audit.get();
    require(observed_failure && draining && joined && error == "group index/windows/codec exceed the memory budget",
        "changed-header audit exceeded its actual-k memory bound or escaped before its peer joined");
    restore(second_path, second_bytes);
}

void farming_replica_staging(fs::path const& root)
{
    using namespace grow_controller_test;
    auto state = farming_format_fixture(5);
    state.directories.push_back(state.directories.back());
    std::vector<size_t> const physical{0, 0, 1};
    std::vector<uint8_t> const selected{1, 1, 1};
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        auto const directory = root / ("replica-stage-" + std::to_string(lane));
        fs::create_directory(directory); state.directories[lane].directory = directory.string();
    }
    auto body = farming_state_text(state); body += stream_state::digest(body) + '\n';
    std::vector<std::optional<stream_state::Stamp>> stamps(selected.size());
    std::promise<void> failed_submission, release;
    auto failed = failed_submission.get_future(); auto released = release.get_future().share();
    std::atomic<bool> joined = false;
    FarmingStageSubmit fail_submit = [&](size_t first, std::function<void()> operation) -> std::future<void>
    {
        if (first)
        {
            failed_submission.set_value();
            throw std::runtime_error("injected replica submission failure");
        }
        return std::async(std::launch::async, [&, operation = std::move(operation)]
        { released.wait(); operation(); joined.store(true, std::memory_order_release); });
    };
    auto worker = std::async(std::launch::async, [&]
    {
        try { stage_farming_replicas(state, physical, body, selected, stamps, 2, fail_submit); }
        catch (std::runtime_error const& error) { return std::string(error.what()); }
        return std::string{};
    });
    bool const reached = failed.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    bool const waiting = worker.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
    release.set_value(); auto const error = worker.get();
    require(reached && waiting && joined.load(std::memory_order_acquire) &&
        error == "injected replica submission failure", "replica staging escaped without joining submitted work");
    require(stamps[0] && stamps[1] && !stamps[2],
        "replica submission failure skipped its shared-resource chain or fabricated an unstaged stamp");
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        auto const directory = file_io::native_path(state.directories[lane].directory);
        require(!fs::exists(directory / "grow-farm.plan"), "replica staging published before its all-staged barrier");
        if (lane < 2)
        {
            auto const bytes = contents(directory / "grow-farm.plan.next");
            require(bytes == std::vector<uint8_t>(body.begin(), body.end()),
                "replica staging failure left an incomplete private copy");
            fs::remove(directory / "grow-farm.plan.next");
        }
    }
    std::vector<size_t> submissions;
    std::array<std::promise<void>, 2> started;
    auto first_started = started[0].get_future(), second_started = started[1].get_future();
    std::promise<void> run_chains; auto chains_released = run_chains.get_future().share();
    std::atomic<unsigned> active{0};
    FarmingStageSubmit submit = [&](size_t first, std::function<void()> operation)
    {
        submissions.push_back(first);
        return std::async(std::launch::async, [&, first, operation = std::move(operation)]
        {
            active.fetch_add(1, std::memory_order_acq_rel); started[first ? 1 : 0].set_value();
            chains_released.wait(); operation(); active.fetch_sub(1, std::memory_order_acq_rel);
        });
    };
    auto staging = std::async(std::launch::async, [&]
    { stage_farming_replicas(state, physical, body, selected, stamps, 2, submit); });
    bool const overlap = first_started.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
        second_started.wait_for(std::chrono::seconds(10)) == std::future_status::ready && active.load(std::memory_order_acquire) == 2;
    run_chains.set_value(); staging.get();
    require(overlap && submissions == std::vector<size_t>{0, 2},
        "replica staging created parallel operations for the same physical resource");
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        auto const path = file_io::native_path(state.directories[lane].directory) / "grow-farm.plan.next";
        require(stamps[lane] && stream_state::stamp(path) == *stamps[lane] &&
            contents(path) == std::vector<uint8_t>(body.begin(), body.end()),
            "replica all-staged barrier omitted a complete durable copy");
    }
    for (size_t lane : {0u, 1u})
        fs::remove(file_io::native_path(state.directories[lane].directory) / "grow-farm.plan.next");
    std::promise<void> operation_failed, finish_peer;
    auto operation_failure = operation_failed.get_future(); auto peer_finished = finish_peer.get_future().share();
    joined = false;
    FarmingStageSubmit fail_operation = [&](size_t first, std::function<void()> operation)
    {
        return std::async(std::launch::async, [&, first, operation = std::move(operation)]
        {
            if (!first) { peer_finished.wait(); operation(); joined = true; return; }
            try { operation(); }
            catch (...) { operation_failed.set_value(); throw; }
        });
    };
    auto failing = std::async(std::launch::async, [&]
    {
        try { stage_farming_replicas(state, physical, body, selected, stamps, 2, fail_operation); }
        catch (std::runtime_error const& error) { return std::string(error.what()); }
        return std::string{};
    });
    bool const failed_operation = operation_failure.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    bool const joined_before_return = failing.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
    finish_peer.set_value(); auto const operation_error = failing.get();
    require(failed_operation && joined_before_return && joined &&
        operation_error.find("pending farming revision must be recovered") != std::string::npos,
        "replica worker failure escaped before another physical resource finished staging");

    auto capped = farming_format_fixture(5);
    auto const common = capped.directories.front(); capped.directories.resize(5, common);
    std::vector<size_t> const independent{0, 1, 2, 3, 4};
    std::vector<uint8_t> const all(independent.size(), 1);
    std::vector<std::optional<stream_state::Stamp>> capped_stamps(independent.size());
    for (size_t lane = 0; lane < capped.directories.size(); ++lane)
    {
        auto const directory = root / ("replica-capped-" + std::to_string(lane)); fs::create_directory(directory);
        auto& d = capped.directories[lane]; d.directory = directory.string(); d.resource = d.storage.resource = lane;
        d.identity_resources = independent.size(); d.identity_resource = lane;
    }
    auto capped_body = farming_state_text(capped); capped_body += stream_state::digest(capped_body) + '\n';
    std::array<std::promise<void>, 2> admitted;
    auto admitted_first = admitted[0].get_future(), admitted_second = admitted[1].get_future();
    std::promise<void> unblock, excess; auto unblocked = unblock.get_future().share(), excessive = excess.get_future().share();
    std::atomic<unsigned> running{0}, maximum{0}; std::atomic<bool> signaled{false};
    FarmingStageSubmit bounded = [&](size_t first, std::function<void()> operation)
    {
        return std::async(std::launch::async, [&, first, operation = std::move(operation)]
        {
            auto const count = running.fetch_add(1, std::memory_order_acq_rel) + 1;
            auto observed = maximum.load(std::memory_order_relaxed);
            while (observed < count && !maximum.compare_exchange_weak(observed, count, std::memory_order_relaxed)) {}
            if (first < admitted.size()) admitted[first].set_value();
            if (count > 2 && !signaled.exchange(true)) excess.set_value();
            unblocked.wait(); operation(); running.fetch_sub(1, std::memory_order_acq_rel);
        });
    };
    auto capped_work = std::async(std::launch::async, [&]
    { stage_farming_replicas(capped, independent, capped_body, all, capped_stamps, 2, bounded); });
    bool const both_admitted = admitted_first.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
        admitted_second.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    bool const over_limit = excessive.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
    unblock.set_value(); capped_work.get();
    require(both_admitted && !over_limit && maximum.load() == 2 && !running.load(),
        "injected replica submission exceeded its worker cap or lost independent-disk overlap");
    require(std::all_of(capped_stamps.begin(), capped_stamps.end(), [](auto const& stamp) { return stamp.has_value(); }),
        "bounded replica staging omitted a later resource chain");
    for (auto const& directory : capped.directories)
        fs::remove(file_io::native_path(directory.directory) / "grow-farm.plan.next");
    std::promise<void> slow_started, later_started, release_slow;
    auto slow = slow_started.get_future(), later = later_started.get_future();
    auto slow_release = release_slow.get_future().share();
    FarmingStageSubmit unbalanced = [&](size_t first, std::function<void()> operation)
    {
        return std::async(std::launch::async, [&, first, operation = std::move(operation)]
        {
            if (!first) { slow_started.set_value(); slow_release.wait(); }
            if (first == 2) later_started.set_value();
            operation();
        });
    };
    auto rolling = std::async(std::launch::async, [&]
    { stage_farming_replicas(capped, independent, capped_body, all, capped_stamps, 2, unbalanced); });
    bool const refilled = slow.wait_for(std::chrono::seconds(10)) == std::future_status::ready &&
        later.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    bool const still_staging = rolling.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    release_slow.set_value(); rolling.get();
    require(refilled && still_staging,
        "replica staging left an admitted resource slot idle behind a slower independent drive");
    std::vector<size_t> attempted;
    FarmingStageSubmit conflicting = [&](size_t first, std::function<void()> operation)
    { attempted.push_back(first); return std::async(std::launch::async, std::move(operation)); };
    bool stopped = false;
    try { stage_farming_replicas(capped, independent, capped_body, all, capped_stamps, 2, conflicting); }
    catch (std::runtime_error const& error)
    { stopped = std::string(error.what()).find("pending farming revision must be recovered") != std::string::npos; }
    require(stopped && (attempted == std::vector<size_t>{0} || attempted == std::vector<size_t>{0, 1}),
        "failed bounded replica staging admitted a later resource after observing failure");
}

void farming_preview_handoff(fs::path const& root)
{
    using namespace grow_controller_test;
    auto const directory = root / "saved-preview-handoff";
    auto const a = directory / "disk a", b = directory / "disk b", path = directory / "review.plan";
    fs::create_directories(a); fs::create_directory(b);
    Arguments options; options.command = "plan"; options.directory_farm = options.plan_only = true;
    options.target = a; options.destinations.push_back(b); options.save_plan = path;
    options.k = 18; options.group_members = 1; options.max_groups = options.max_total_members = 1;
    options.farming_max_eligible_groups = 4; options.max_ram = 512 * mib;
    options.capacity = 64 * mib;
    options.seed = std::string(64, 'a'); options.farmer = std::string(96, 'b'); options.pool_pk = std::string(96, 'c');
    auto job = farming_arguments(options);
    {
        auto reserved = options; reserved.save_plan = a / "grow-farm.plan";
        bool refused = false;
        try { (void)farming_run(reserved, nullptr, job); }
        catch (std::exception const& error) { refused = std::string(error.what()).find("filename is reserved") != std::string::npos; }
        require(refused && fs::is_empty(a), "saved preview claimed the private job recovery filename");
    }
    {
        FarmingProfile public_profile;
        public_profile.assumptions.assign(64, std::string(4096, 'x'));
        std::ostringstream json; write_farming_profile_json(json, public_profile);
        std::istringstream input(json.str());
        auto profile = std::make_shared<FarmingProfile const>(read_farming_profile(input));
        for (bool preview : {false, true})
        {
            auto oversized = options; oversized.command = preview ? "plan" : "fill";
            oversized.plan_only = preview; oversized.save_plan.clear(); oversized.farming_profile = profile;
            auto source = farming_arguments(oversized);
            bool refused = false;
            try { (void)farming_run(oversized, nullptr, source); }
            catch (std::exception const& error) { refused = std::string(error.what()).find("profile metadata exceeds its bound") != std::string::npos; }
            require(refused && fs::is_empty(a) && fs::is_empty(b),
                "oversized aggregate profile metadata reached farming publication");
        }

        public_profile.assumptions.resize(14);
        Arguments memory = options; memory.save_plan.clear(); memory.target.clear(); memory.destinations.clear();
        memory.farming_profile = std::make_shared<FarmingProfile const>(public_profile);
        for (unsigned i = 0; i < 16; ++i)
        {
            auto const destination = directory / ("serialization-memory-" + std::to_string(i));
            fs::create_directory(destination);
            if (!i) memory.target = destination;
            else memory.destinations.push_back(destination);
        }
        auto source = farming_arguments(memory);
        auto reference = farming_run(memory, nullptr, source);
        auto const text = farming_state_text(reference->state);
        // This leaves room for the search and retained inputs, but not the
        // serialization buffers that must coexist while sizing private copies.
        memory.max_ram = sum(reference->metadata, product(text.size(), 3));
        bool refused = false;
        try { (void)farming_run(memory, nullptr, source); }
        catch (std::exception const& error) { refused = std::string(error.what()).find("reservation serialization exceeds --max-ram") != std::string::npos; }
        require(refused && fs::is_empty(memory.target), "private planning omitted its serialization working memory");
    }
    auto planned = farming_run(options, nullptr, job);
    require(planned->feasible() && fs::is_empty(a) && fs::is_empty(b), "saving preparation wrote destination files");
    {
        // An independent maximum-identity serialization places the metadata
        // exactly at an allocation boundary, then crosses it with one byte.
        // Key form and escaped owner/parent fields must never move that edge.
        uint64_t const alignment = 4096;
        auto boundary = planned->state; boundary.version = 4;
        boundary.pool.assign(96, '0'); boundary.seed_owner.assign(128, '"');
        boundary.parent_digest.assign(64, '0'); boundary.revision = 2;
        uint64_t const extra = sum(65, std::max(uint64_t(1024), farming_revision_margin(boundary)));
        auto const initial = farming_state_text(boundary);
        uint64_t const padding = (alignment - sum(initial.size(), extra) % alignment) % alignment;
        boundary.profile_json.append(size_t(padding), ' ');
        uint64_t const expected = product(sum(farming_state_text(boundary).size(), extra), 2);
        for (bool crossed : {false, true})
            for (bool public_key : {false, true})
                for (bool parent : {false, true})
                    for (auto const& owner : {std::string{}, std::string("owner"), std::string(128, 'x'),
                        std::string(128, '\\'), std::string(128, '"')})
                    {
                        auto state = boundary;
                        state.pool.assign(public_key ? 96 : 64, 'a'); state.pool_public_key = public_key;
                        state.seed_owner = owner; state.parent_digest = parent ? std::string(64, 'b') : "";
                        state.revision = parent ? 2 : 1;
                        if (crossed) state.profile_json.push_back(' ');
                        auto const text = farming_state_text(state);
                        uint64_t const reserved = farming_manifest_reservation(state, text.size(), alignment);
                        require(reserved == (crossed ? sum(expected, product(alignment, 2)) : expected),
                            "farming reservation changed at an allocation boundary with its key or escaped owner form");
                        state.manifest_reserved_bytes = reserved;
                        require(farming_revision_fits(state, sum(farming_state_text(state).size(), 65)),
                            "maximum escaped owner consumed the reserved revision margin");
                    }
    }
    save_farming_review(options, job, *planned);
    auto const original = contents(path);
    std::string const encoded(original.begin(), original.end());
    require(encoded.find(*options.seed) == std::string::npos && encoded.find(*options.farmer) == std::string::npos &&
        encoded.find(*options.pool_pk) == std::string::npos, "saved farming preview leaked supplied identity fields");
    auto const saved = load_farming_preview(path, options.max_ram);
    require(saved.preview.state.seed == std::string(64, '0') && saved.preview.state.seed_owner.empty() &&
        saved.preview.state.capacity_fill, "saved preview lost its keyless capacity policy");
    bool refused = false;
    try { save_farming_review(options, job, *planned); }
    catch (std::exception const&) { refused = true; }
    require(refused && contents(path) == original, "saving a farming preview replaced an existing file");

    auto inspect = [&]
    {
        Arguments args; args.command = "fill"; args.directory_farm = args.plan_only = true;
        args.from_plan = path; args.max_ram = options.max_ram;
        auto source = farming_arguments(args);
        return farming_run(args, nullptr, source);
    };
    auto reviewed = inspect();
    require(reviewed->feasible() && reviewed->admission && fs::is_empty(a) && fs::is_empty(b),
        "saved preview inspection skipped frozen assessment or wrote destination files");
    for (size_t i = 0; i < planned->state.directories.size(); ++i)
        require(reviewed->state.directories[i].layout == planned->state.directories[i].layout &&
            reviewed->state.directories[i].capacity_bytes == planned->state.directories[i].capacity_bytes &&
            reviewed->state.directories[i].extents == planned->state.directories[i].extents,
            "saved preview inspection reoptimized reviewed assignments");
    {
        Arguments changed; changed.command = "fill"; changed.directory_farm = changed.plan_only = true;
        changed.from_plan = path; changed.max_ram = options.max_ram;
        auto source = farming_arguments(changed);
        ++source.preview->preview.state.directories.front().layout.schedule;
        refused = false;
        try { (void)farming_run(changed, nullptr, source); }
        catch (std::exception const& error) { refused = std::string(error.what()).find("initial assignment differs") != std::string::npos; }
        require(refused && fs::is_empty(a) && fs::is_empty(b), "inconsistent reviewed identity schedule reached publication");
    }
    {
        Arguments changed; changed.command = "fill"; changed.directory_farm = true;
        changed.from_plan = path; changed.max_ram = options.max_ram;
        auto source = farming_arguments(changed);
        auto& state = source.preview->preview.state; state.manifest_reserved_bytes = 0;
        for (auto& destination : state.directories) destination.storage.capacity_bytes = destination.capacity_bytes;
        refused = false;
        try { (void)farming_run(changed, nullptr, source); }
        catch (std::exception const& error) { refused = std::string(error.what()).find("exceeds its reserved metadata") != std::string::npos; }
        require(refused && fs::is_empty(a) && fs::is_empty(b), "insufficient reviewed metadata reservation reached private publication");
    }

    auto const unrelated = directory / "unrelated allocation";
    {
        auto bytes = std::string(2 * mib, '\0');
        pos2gpu::read_random_bytes(reinterpret_cast<uint8_t*>(bytes.data()), bytes.size());
        stream_state::atomic_save(unrelated, bytes, false);
    }
    refused = false;
    std::string capacity_error;
    try
    {
        Arguments changed; changed.command = "fill"; changed.directory_farm = changed.plan_only = true;
        changed.from_plan = path; changed.max_ram = options.max_ram;
        auto source = farming_arguments(changed);
        // Anchor this reviewed quota to the observed unrelated allocation,
        // independently of other tests creating/removing files on this disk.
        file_io::Space space{}; file_io::Stat allocation{};
        require(!file_io::space_path(a.c_str(), &space) && !file_io::inspect_link(unrelated.c_str(), &allocation),
            "cannot inspect unrelated capacity change");
        uint64_t const available = product(space.f_bavail, space.f_frsize);
        uint64_t const occupied = std::max(uint64_t(allocation.st_size), product(uint64_t(allocation.st_blocks), 512));
        auto& state = source.preview->preview.state;
        // This case deliberately checks changed physical free space rather
        // than the handoff fixture's fixed, concurrency-independent cap.
        changed.capacity = state.capacity = 0;
        uint64_t const quota = (sum(available, occupied) - state.headroom) / state.directories.size();
        for (auto& destination : state.directories)
        {
            destination.capacity_bytes = quota;
            destination.storage.capacity_bytes = quota - state.manifest_reserved_bytes;
        }
        (void)farming_run(changed, nullptr, source);
    }
    catch (std::exception const& error) { capacity_error = error.what(); refused = capacity_error.find("capacity is no longer available") != std::string::npos; }
    require(refused && fs::is_empty(a) && fs::is_empty(b), "unrelated storage growth bypassed saved-preview capacity freshness: " + capacity_error);
    fs::remove(unrelated);

    auto const previous = directory / "old disk a";
    fs::rename(a, previous); fs::create_directory(a);
    refused = false;
    try { (void)inspect(); }
    catch (std::exception const& error) { refused = std::string(error.what()).find("storage identity or topology changed") != std::string::npos; }
    require(refused && fs::is_empty(a) && fs::is_empty(b), "replaced destination bypassed saved-preview identity freshness");
    fs::remove(a); fs::rename(previous, a);

    Arguments executing; executing.command = "fill"; executing.directory_farm = true;
    executing.from_plan = path; executing.max_ram = options.max_ram;
    auto admitted = farming_arguments(executing);
    auto published = farming_run(executing, nullptr, admitted);
    require(published->feasible() && fs::exists(a / "grow-farm.plan") && fs::exists(b / "grow-farm.plan"),
        "saved-preview execution did not use ordinary replicated publication");
    published.reset();
    refused = false;
    try { (void)inspect(); }
    catch (std::exception const& error) { refused = std::string(error.what()).find("stale") != std::string::npos; }
    require(refused, "fresh-job preview accepted a subsequently published job");

    auto source_options = options; source_options.save_plan = directory / "saved-job-review.plan";
    auto source_job = farming_arguments(source_options);
    auto source_run = farming_run(source_options, nullptr, source_job);
    save_farming_review(source_options, source_job, *source_run);
    Arguments source_inspection; source_inspection.command = "fill";
    source_inspection.directory_farm = source_inspection.plan_only = true;
    source_inspection.from_plan = source_options.save_plan; source_inspection.max_ram = options.max_ram;
    auto source_review = farming_arguments(source_inspection);
    auto source_checked = farming_run(source_inspection, nullptr, source_review);
    require(source_checked->feasible() && source_checked->admission,
        "saved-job preview lost its own allocation allowance or frozen assessment");
    auto const source_bytes = contents(source_options.save_plan);
    stream_state::atomic_save(source_options.save_plan, std::string(source_bytes.begin(), source_bytes.end()));
    refused = false;
    try { source_review.check(); }
    catch (std::exception const& error) { refused = std::string(error.what()).find("preview changed") != std::string::npos; }
    require(refused, "changed preview stamp bypassed source admission");

    auto revised = *source_job.saved;
    ++revised.revision; revised.parent_digest = source_job.digest;
    auto revision = farming_state_text(revised); revision += stream_state::digest(revision) + '\n';
    stream_state::atomic_save(a / "grow-farm.plan", revision);
    stream_state::atomic_save(b / "grow-farm.plan", revision);
    source_inspection.target.clear(); source_inspection.destinations.clear(); source_inspection.manifest.clear();
    source_inspection.extend = false;
    refused = false;
    try { (void)farming_arguments(source_inspection); }
    catch (std::exception const& error) { refused = std::string(error.what()).find("stale") != std::string::npos; }
    require(refused && contents(a / "grow-farm.plan") == std::vector<uint8_t>(revision.begin(), revision.end()),
        "changed source revision was accepted or rewritten by a saved preview");

    auto text = farming_preview_text(saved.preview) + "unexpected\n";
    refused = false;
    try { (void)parse_farming_preview(text); }
    catch (std::exception const& error) { refused = std::string(error.what()).find("unexpected") != std::string::npos; }
    require(refused, "saved preview parser accepted trailing data");
}

void farming_capacity_extents(fs::path const& root)
{
    using namespace grow_controller_test;
    auto const directory = root / "capacity-extents"; fs::create_directory(directory);
    FarmingRun farm; farm.memory_limit = 512 * mib; farm.metadata = 2 * mib;
    auto& state = farm.state; state.fixed_extent = state.capacity_fill = true;
    state.config.k = 18; state.capacity = 5110000; state.headroom = 0; state.manifest_reserved_bytes = 8192;
    state.job_id = state.fingerprint = state.seed = state.pool = std::string(64, '0');
    state.farmer = std::string(96, '0');
    FarmingDestination storage; storage.capacity_bytes = state.capacity - state.manifest_reserved_bytes;
    storage.resource = 0; storage.known_device = storage.nonrotating = true;
    state.estimate = farming_plan(state.config, {storage});
    FarmingDirectory assignment; assignment.directory = directory.string(); assignment.filesystem = "test";
    assignment.capacity_bytes = state.capacity; assignment.storage = storage;
    assignment.layout = state.estimate.destinations[0]; assignment.directory_identity = "test:capacity";
    state.directories.push_back(assignment); initialize_farming_extents(state);
    farm.metadata += farming_extent_memory(state); farm.lanes.resize(1); farm.available = {1};
    auto const manifest = directory / "grow-farm.plan";
    auto text = farming_state_text(state); stream_state::atomic_save(manifest, text + stream_state::digest(text) + '\n');
    farm.manifests.emplace_back(manifest, stream_state::stamp(manifest));
    farm.lanes[0].manifest_charge = farm.lanes[0].completed_charge = snapshot(manifest, 0).charge();
    auto const original = state;
    require(state.directories[0].extents == std::vector<uint16_t>{7}, "capacity fixture forecast differs");
    require(farm.reserve_extent(0, 0, 8, farm.metadata), "feasible capacity member was not admitted");
    auto admitted = load_farming_state(manifest, farm.memory_limit);
    require(admitted.directories[0].extents == std::vector<uint16_t>{8} &&
        admitted.directories[0].layout == original.directories[0].layout &&
        valid_farming_extent_revision(original, admitted), "capacity admission changed the original assignment");
    auto child = ::fork(); require(child >= 0, "capacity revision fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "farm-extent-all-staged", 1);
        (void)farm.reserve_extent(0, 0, 9, farm.metadata); ::_exit(0);
    }
    int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "capacity revision interruption did not reach its durable fence");
    require(load_farming_state(manifest, farm.memory_limit).directories[0].extents == std::vector<uint16_t>{8} &&
        load_farming_state(directory / "grow-farm.plan.next", farm.memory_limit).directories[0].extents == std::vector<uint16_t>{9},
        "unpublished capacity intent replaced the previous exact extent");
    auto const pending = load_farming_state(directory / "grow-farm.plan.next", farm.memory_limit);
    require(valid_farming_extent_revision(admitted, pending), "staged capacity intent is not recoverable");
    require(!file_io::unlink_file(directory / "grow-farm.plan.next"), "cannot retire test capacity intent");
    farm.seal_extent(0, 0, 7, "capacity", farm.metadata);
    auto sealed = load_farming_state(manifest, farm.memory_limit);
    require(sealed.directories[0].sealed_files == 1 && sealed.directories[0].fill_stop == "capacity" &&
        sealed.directories[0].extents == std::vector<uint16_t>{7} && valid_farming_extent_revision(admitted, sealed),
        "rejected provisional member was not durably sealed at its actual extent");
    auto changed = sealed; changed.directories[0].extents[0] = 8;
    require(!valid_farming_extent_revision(sealed, changed), "sealed capacity membership was mutable");

    FarmingRun repeated; repeated.state = original; repeated.state.version = 4; repeated.memory_limit = farm.memory_limit;
    repeated.metadata = 2 * mib; repeated.lanes.resize(1); repeated.available = {1};
    repeated.metadata += farming_extent_memory(repeated.state);
    auto const compact = root / "compact-extents"; fs::create_directory(compact);
    repeated.state.directories[0].directory = compact.string();
    auto const compact_manifest = compact / "grow-farm.plan";
    auto save_compact = [&]
    {
        auto const body = farming_state_text(repeated.state);
        stream_state::atomic_save(compact_manifest, body + stream_state::digest(body) + '\n');
        repeated.manifests = {{compact_manifest, stream_state::stamp(compact_manifest)}};
        repeated.lanes[0].manifest_charge = repeated.lanes[0].completed_charge = snapshot(compact_manifest, 0).charge();
    };
    save_compact(); auto const initial_bytes = farming_state_text(repeated.state).size();
    repeated.seal_extent(0, 0, 7, "", repeated.metadata);
    for (size_t file = 1; file <= 200; ++file)
    {
        repeated.lanes[0].ordinal = file;
        require(repeated.reserve_extent(0, file, 1, repeated.metadata), "repeated small suffix exhausted compact metadata");
        repeated.seal_extent(0, file, 1, "", repeated.metadata);
        repeated.lanes[0].reserved_members = 0;
    }
    auto const compact_body = farming_state_text(repeated.state);
    require(compact_body.size() < initial_bytes + 128 &&
        parse_farming_state(compact_body).directories[0].extents == repeated.state.directories[0].extents,
        "equal suffix extents were not compactly preserved");
    bool memory_rejected = false;
    try { (void)parse_farming_state(compact_body, compact_body.size() * 4 + sizeof(FarmingState) + sizeof(FarmingDirectory) + 1); }
    catch (std::exception const&) { memory_rejected = true; }
    require(memory_rejected, "compact extent expansion bypassed its parsing memory budget");
    repeated.lanes[0].ordinal = repeated.state.directories[0].extents.size();
    auto const old_memory = farming_extent_memory(repeated.state);
    FarmingProfile profile; profile.config = repeated.state.config;
    std::ostringstream profile_text; write_farming_profile_json(profile_text, profile);
    repeated.state.profile_json = profile_text.str();
    auto const required = farming_state_text(repeated.state).size() + 65 + farming_revision_margin(repeated.state);
    require(required < repeated.state.manifest_reserved_bytes / 2, "metadata edge fixture has no padding room");
    repeated.state.profile_json.append(repeated.state.manifest_reserved_bytes / 2 - required + 1, ' ');
    repeated.metadata = repeated.metadata - old_memory + farming_extent_memory(repeated.state);
    save_compact(); bool metadata_rejected = false;
    auto const before_stop = repeated.state.directories[0].extents;
    require(!repeated.reserve_extent(0, repeated.lanes[0].ordinal, 1, repeated.metadata, &metadata_rejected) &&
        metadata_rejected && repeated.state.directories[0].extents == before_stop,
        "metadata capacity admission modified a rejected suffix");
    repeated.stop_filling(0, "metadata-capacity", repeated.metadata);
    auto const terminal = load_farming_state(compact_manifest, repeated.memory_limit);
    require(terminal.directories[0].fill_stop == "metadata-capacity" && terminal.directories[0].extents == before_stop,
        "metadata capacity stop did not retain a recoverable exact extent");

    FarmingRun partial; partial.memory_limit = farm.memory_limit; partial.metadata = 2 * mib;
    partial.state = original; auto shared = original.directories[0].storage;
    partial.state.estimate = farming_plan(partial.state.config, {shared, shared});
    partial.state.directories.push_back(partial.state.directories[0]);
    for (size_t lane = 0; lane < partial.state.directories.size(); ++lane)
        partial.state.directories[lane].layout = partial.state.estimate.destinations[lane];
    initialize_farming_extents(partial.state);
    partial.state.directories[1].extents[0] = 8;
    partial.metadata += farming_extent_memory(partial.state);
    partial.lanes.resize(2); partial.available = {0, 1}; partial.certify(partial.metadata);
    require(partial.admission && partial.admission->feasible, "full-job missing-member fixture was not admitted");
    auto const full_assessment = *partial.admission;
    partial.select_available();
    require(!partial.all_replicas && partial.covered(), "missing-member selection lost its full-job certificate");
    partial.certify(partial.metadata);
    require(partial.admission->dynamic_work_seconds_per_sp == full_assessment.dynamic_work_seconds_per_sp &&
        partial.admission->response_bound_seconds == full_assessment.response_bound_seconds,
        "missing-member recovery replaced the full-job assessment with a subset");
    partial.lanes[0].reserved_members = 9;
    bool new_admission_rejected = false;
    try { partial.certify(partial.metadata); }
    catch (std::exception const&) { new_admission_rejected = true; }
    require(new_admission_rejected, "missing-member recovery admitted an unreviewed larger extent");

    FarmingRun hard; hard.state = original; hard.memory_limit = farm.memory_limit; hard.metadata = 2 * mib;
    hard.state.config.group_members_override = 1;
    hard.state.config.max_eligible_families_per_resource = farming_detail::maximum_files;
    auto hard_storage = shared; hard_storage.capacity_bytes = uint64_t(1) << 50;
    hard_storage.max_groups = hard_storage.max_total_members = farming_detail::maximum_files;
    hard.state.estimate = farming_plan(hard.state.config, {hard_storage});
    require(hard.state.estimate.feasible && hard.state.estimate.destinations[0].files == farming_detail::maximum_files,
        "hard extent-count fixture did not reach its file bound");
    hard.state.directories[0].layout = hard.state.estimate.destinations[0];
    hard.state.directories[0].storage = hard_storage;
    initialize_farming_extents(hard.state); hard.lanes.resize(1); hard.available = {1};
    hard.state.directories[0].sealed_files = hard.lanes[0].ordinal = farming_detail::maximum_files;
    auto const hard_size = hard.state.directories[0].extents.size(); metadata_rejected = false;
    require(!hard.reserve_extent(0, hard.lanes[0].ordinal, 1, hard.metadata, &metadata_rejected) && !metadata_rejected &&
        hard.state.directories[0].extents.size() == hard_size && hard.state.revision == original.revision,
        "hard file-count exhaustion mutated or repeatedly admitted a suffix");

    auto input = members(2); auto cfg = configuration();
    auto const window_path = root / "capacity-window.gplot"; direct(window_path, Members(input).subspan(0, 1), cfg);
    auto const source_bytes = contents(window_path);
    auto small = cfg;
    {
        Engine inspect(window_path.string(), cfg, true, identity()); inspect.confirm_identity();
        small.window_bytes = inspect.largest_chunk();
    }
    {
        Construction engine(window_path.string(), small, true, identity(), "incremental"); engine.confirm_identity();
        auto const rejected = engine.append(Members(input).subspan(1), nullptr, true);
        require(!rejected.accepted && rejected.reason.find("source-window") != std::string::npos &&
            !engine.identity().pending && contents(window_path) == source_bytes,
            "new top-up window rejection started or changed a durable transaction");
        bool ordinary_rejected = false;
        try { (void)engine.append(Members(input).subspan(1)); }
        catch (WindowBudgetError const&) { ordinary_rejected = true; }
        require(ordinary_rejected && contents(window_path) == source_bytes,
            "top-up window handling changed ordinary initial-work rejection");
    }
    auto const pending_path = root / "capacity-window-pending.gplot";
    direct(pending_path, Members(input).subspan(0, 1), cfg);
    child = ::fork(); require(child >= 0, "pending window fixture fork failed");
    if (!child)
    {
        ::setenv("XCHPLOT2_GROW_FAILPOINT", "marked", 1);
        Engine output(pending_path.string(), cfg, true, identity()); output.confirm_identity();
        (void)output.append(Members(input).subspan(1)); ::_exit(0);
    }
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
        "pending window fixture did not create its durable transaction");
    auto const pending_bytes = contents(pending_path);
    {
        Engine output(pending_path.string(), cfg, true, identity()); output.confirm_identity();
        bool pending_rejected = false;
        try { (void)output.append(Members(input).subspan(1), nullptr, true); }
        catch (std::exception const&) { pending_rejected = true; }
        require(pending_rejected && output.identity().pending && contents(pending_path) == pending_bytes,
            "optional candidate rejection abandoned pending construction");
        (void)output.resume(Members(input).subspan(1));
    }
    auto const window_reference = root / "capacity-window-reference.gplot"; direct(window_reference, input, cfg);
    require(contents(pending_path) == contents(window_reference), "pending window fixture lost ordinary recovery parity");
}

void farming_extent_admission(fs::path const& root)
{
    using namespace grow_controller_test;
    auto make_farm = [&](char const* name, uint64_t files, uint64_t family_limit)
    {
        FarmingRun farm; farm.memory_limit = 512 * mib; farm.metadata = 2 * mib;
        auto& config = farm.state.config; config.k = 18; config.group_members_override = 1;
        config.fixed_meta = true; config.max_eligible_families_per_resource = family_limit;
        farm.state.fingerprint = std::string(64, '0'); farm.state.manifest_reserved_bytes = 4096;
        farm.state.job_id = farm.state.seed = farm.state.pool = std::string(64, '0');
        farm.state.farmer = std::string(96, '0');
        std::vector<FarmingDestination> destinations(2);
        std::vector<FarmingDrivePlan> layouts(2);
        for (size_t lane = 0; lane < layouts.size(); ++lane)
        {
            auto& storage = destinations[lane]; storage.capacity_bytes = 128 * mib;
            storage.resource = 0; storage.known_device = storage.nonrotating = true;
            auto& layout = layouts[lane]; layout.files = layout.total_members = files;
            layout.group_members = layout.meta_slots = 1; layout.fixed_members = true;
            layout.resource_rank = uint16_t(lane); layout.resource_lanes = 2;
            layout.families = layout.family(files - 1) + 1;
            auto const directory = root / (std::string(name) + "-" + std::to_string(lane));
            fs::create_directory(directory);
            farm.state.directories.push_back({directory.string(), "test", storage.capacity_bytes, 0, layout, storage});
            farm.state.directories.back().directory_identity = "test:" + std::to_string(lane);
        }
        auto const initial = farming_assess(config, destinations, layouts,
            std::vector<std::vector<uint16_t>>(2, std::vector<uint16_t>(size_t(files), 1)));
        require(initial.feasible, "frozen farming admission fixture is infeasible");
        static_cast<FarmingAssessment&>(farm.state.estimate) = initial;
        farm.state.estimate.destinations = layouts; farm.lanes.resize(2);
        return farm;
    };
    auto bounded = make_farm("bounded", 2, 4);
    bounded.state.fixed_extent = true;
    auto& terminal = bounded.state.directories[0].layout;
    terminal.group_members = 2; terminal.total_members = 3;
    bounded.lanes[0].ordinal = 1; bounded.lanes[0].completed_members = 2;
    Arguments options;
    require(bounded.maximum(options, 0) == 1 && bounded.file_maximum(0, 2) == 0,
        "managed recovery enlarged a partial terminal file or added an unplanned file");
    bounded.lanes[0].ordinal = 2;
    require(!bounded.maximum(options, 0), "managed recovery exceeded its saved extent");

    auto promised = make_farm("promised", 2, 4);
    promised.lanes[0].outputs.resize(2);
    for (auto& output : promised.lanes[0].outputs) output.members = 1;
    promised.lanes[0].ordinal = 2; promised.lanes[0].reserved_members = 1;
    promised.certify(promised.metadata);
    require(!promised.feasible() && promised.admission &&
        promised.admission->max_eligible_families_per_resource == 5,
        "fast-lane extension spent another lane's unbuilt forecast budget");

    auto joint = make_farm("joint", 1, 3);
    for (auto& lane : joint.lanes)
    { lane.outputs.resize(1); lane.outputs[0].members = 1; lane.ordinal = 1; }
    joint.lanes[0].reserved_members = 1; joint.certify(joint.metadata);
    require(joint.feasible() && joint.covered(), "safe first farming extension was not certified");
    auto const memory = joint.certificate_memory();
    joint.certify(joint.metadata);
    require(joint.covered() && joint.certificate_memory() == memory,
        "covered farming extent lost its reusable global certificate");
    joint.lanes[0].reserved_members = 0; joint.lanes[1].reserved_members = 1;
    joint.certify(joint.metadata);
    require(joint.feasible(), "safe alternate farming extension was not certified");
    joint.lanes[0].reserved_members = 1;
    require(!joint.covered(), "independent farming certificates were combined without a joint assessment");
    joint.certify(joint.metadata);
    require(!joint.feasible() && joint.admission->max_eligible_families_per_resource == 4,
        "joint farming reservations exceeded their frozen family constraint");

    auto saved_capacity = [&](char const* name, uint64_t limit, bool first_active = false)
    {
        auto farm = make_farm(name, 1, limit);
        farm.state.fixed_extent = farm.state.capacity_fill = true;
        farm.state.manifest_reserved_bytes = 8192;
        initialize_farming_extents(farm.state);
        farm.metadata += farming_extent_memory(farm.state);
        for (size_t lane = 0; lane < farm.lanes.size(); ++lane)
        {
            auto& current = farm.lanes[lane];
            current.outputs.resize(1); current.outputs[0].members = 1;
            bool const active = first_active && !lane;
            current.ordinal = current.completed_members = active ? 0 : 1;
            current.reserved_members = active ? 1 : 0;
            farm.state.directories[lane].sealed_files = active ? 0 : 1;
        }
        auto body = farming_state_text(farm.state); body += stream_state::digest(body) + '\n';
        for (size_t lane = 0; lane < farm.lanes.size(); ++lane)
        {
            auto const manifest = file_io::native_path(farm.state.directories[lane].directory) / "grow-farm.plan";
            stream_state::atomic_save(manifest, body);
            farm.manifests.emplace_back(manifest, stream_state::stamp(manifest));
            farm.lanes[lane].manifest_charge = farm.lanes[lane].completed_charge = snapshot(manifest, 0).charge();
        }
        return farm;
    };
    using Change = FarmingRun::ExtentChange;
    using Kind = Change::Kind;
    auto batched = saved_capacity("coalesced-compatible", 4);
    auto const before_batch = batched.state;
    std::vector<Change> additions{{Kind::reserve, 0, 1, 1}, {Kind::reserve, 1, 1, 1}};
    batched.revise_extents(additions, batched.metadata);
    require(additions[0].accepted && additions[1].accepted && batched.state.revision == before_batch.revision + 1 &&
        batched.state.parent_digest == stream_state::digest(farming_state_text(before_batch)) &&
        valid_farming_extent_revision(before_batch, batched.state) && batched.covered() && batched.feasible(),
        "compatible independent-lane farming additions did not commit one jointly admitted revision");
    for (size_t lane = 0; lane < batched.lanes.size(); ++lane)
    {
        auto const manifest = file_io::native_path(batched.state.directories[lane].directory) / "grow-farm.plan";
        auto const restored = load_farming_state(manifest, batched.memory_limit);
        require(restored.directories[lane].extents == std::vector<uint16_t>{1, 1} &&
            restored.directories[lane].layout == before_batch.directories[lane].layout &&
            farming_state_text(restored) == farming_state_text(batched.state),
            "coalesced farming publication omitted an addition or changed a frozen identity");
    }
    auto constrained = saved_capacity("coalesced-constrained", 3);
    auto const constrained_before = constrained.state;
    additions = {{Kind::reserve, 0, 1, 1}, {Kind::reserve, 1, 1, 1}};
    constrained.revise_extents(additions, constrained.metadata);
    require(additions[0].accepted && !additions[1].accepted && constrained.feasible() && constrained.covered() &&
        constrained.state.revision == constrained_before.revision + 1 &&
        constrained.state.directories[0].extents == std::vector<uint16_t>{1, 1} &&
        constrained.state.directories[1].extents == std::vector<uint16_t>{1} &&
        constrained.admission->max_eligible_families_per_resource == 3 &&
        valid_farming_extent_revision(constrained_before, constrained.state),
        "joint batch rejection lost a fitting first admission or combined incompatible certificates");
    auto const stable = farming_state_text(constrained.state);
    auto const stamps = constrained.manifests;
    for (auto changes : {std::vector<Change>{{Kind::seal, 0, 1, 1}, {Kind::reserve, 0, 2, 1}},
                         std::vector<Change>{{Kind::reserve, 0, 1, 1}, {Kind::reserve, 1, 9, 1}}})
    {
        bool refused = false;
        try { constrained.revise_extents(changes, constrained.metadata); }
        catch (std::runtime_error const&) { refused = true; }
        require(refused && farming_state_text(constrained.state) == stable && constrained.manifests == stamps,
            "duplicate or invalid coalesced transition changed a valid saved farming revision");
    }
    additions = {{Kind::reserve, 1, 1, 1}};
    bool memory_refused = false;
    try { constrained.revise_extents(additions, constrained.memory_limit); }
    catch (FarmingResourceError const& error) { memory_refused = error.memory_budget; }
    require(memory_refused && !additions[0].accepted && farming_state_text(constrained.state) == stable &&
        constrained.manifests == stamps && constrained.feasible() && constrained.covered(),
        "coalesced assessment memory failure retained a provisional admission or changed the saved revision");
    auto mixed = saved_capacity("coalesced-seal-reserve", 3, true);
    auto const mixed_before = mixed.state;
    std::vector<Change> seal_and_add{{Kind::seal, 0, 0, 1, "total-member-limit"}, {Kind::reserve, 1, 1, 1}};
    mixed.revise_extents(seal_and_add, mixed.metadata);
    require(seal_and_add[1].accepted && mixed.state.revision == mixed_before.revision + 1 &&
        mixed.state.directories[0].sealed_files == 1 && mixed.state.directories[0].fill_stop == "total-member-limit" &&
        mixed.state.directories[1].extents == std::vector<uint16_t>{1, 1} && mixed.feasible() && mixed.covered() &&
        mixed.admission->max_eligible_families_per_resource == 3 && valid_farming_extent_revision(mixed_before, mixed.state),
        "mixed completion and reservation lost the sealed lane's promised demand or joint admission");
    for (auto const& reason : {std::string("group-limit"), std::string("total-member-limit"), std::string{}})
    {
        auto limited = saved_capacity(("coalesced-metadata-" + (reason.empty() ? "capacity" : reason)).c_str(), 4);
        limited.state.manifest_reserved_bytes = 2 * (farming_state_text(limited.state).size() + 65 +
            farming_revision_margin(limited.state) - 1);
        auto body = farming_state_text(limited.state); body += stream_state::digest(body) + '\n';
        for (size_t lane = 0; lane < limited.manifests.size(); ++lane)
        {
            auto& copy = limited.manifests[lane]; stream_state::atomic_save(copy.first, body);
            copy.second = stream_state::stamp(copy.first);
            limited.lanes[lane].manifest_charge = limited.lanes[lane].completed_charge = snapshot(copy.first, 0).charge();
        }
        std::vector<Change> stop{{Kind::reserve, 0, 1, 1, reason}}; stop[0].rejected = Kind::stop;
        limited.revise_extents(stop, limited.metadata);
        require(!stop[0].accepted && stop[0].metadata_rejected &&
            limited.state.directories[0].fill_stop == (reason.empty() ? "metadata-capacity" : reason),
            "metadata fallback changed explicit terminal-limit reason precedence");
    }

    auto saved = make_farm("saved", 1, 3);
    saved.state.version = 5; // Retain strict legacy replica-build coverage.
    auto const manifest = root / "farming-admission.plan";
    auto const text = farming_state_text(saved.state);
    stream_state::atomic_save(manifest, text + stream_state::digest(text) + '\n');
    FarmingRun resumed; resumed.state = load_farming_state(manifest, saved.memory_limit);
    resumed.memory_limit = saved.memory_limit; resumed.metadata = saved.metadata; resumed.lanes.resize(2);
    for (size_t lane = 0; lane < resumed.lanes.size(); ++lane)
    {
        require(resumed.state.directories[lane].storage.resource == 0,
            "saved farming storage lost its shared physical-resource assignment");
        resumed.lanes[lane].outputs.resize(1); resumed.lanes[lane].outputs[0].members = 1;
        resumed.lanes[lane].ordinal = 1;
    }
    resumed.lanes[0].reserved_members = 1; resumed.certify(resumed.metadata);
    require(resumed.feasible() && resumed.admission && resumed.admission->max_eligible_families_per_resource == 3,
        "resumed shared-resource farming extension was not assessed beyond its forecast");

    auto const replica = file_io::native_path(saved.state.directories.back().directory) / "grow-farm.plan";
    auto const locator = file_io::native_path(saved.state.directories.front().directory) / "grow-farm.plan";
    auto const digest = stream_state::digest(text);
    stream_state::atomic_save(locator, text + digest + '\n');
    stream_state::atomic_save(replica, text + digest + '\n');
    auto const original_replica = contents(locator);
    auto discover = [&]
    {
        Arguments args; args.plan_only = true; args.target = locator.parent_path(); args.max_ram = saved.memory_limit;
        return farming_arguments(args);
    };
    auto duplicates = discover();
    require(duplicates.saved && duplicates.digest == digest && duplicates.directories.size() == saved.state.directories.size(),
        "identical farming replicas changed the discovered job");
    auto const formatted = text + '\n';
    stream_state::atomic_save(replica, formatted + stream_state::digest(formatted) + '\n');
    auto noncanonical = discover();
    require(noncanonical.saved && noncanonical.digest == digest && !noncanonical.pending,
        "equivalent noncanonical farming replica changed the canonical revision");
    for (auto const& copy : noncanonical.copies) if (copy.stamp)
        require(copy.digest == digest, "noncanonical farming replica retained its raw checksum as revision identity");
    stream_state::atomic_save(replica, text + std::string(64, '0') + '\n');
    bool corrupt_replica = false;
    try { (void)discover(); }
    catch (std::runtime_error const& error)
    { corrupt_replica = std::string(error.what()).find("checksum mismatch") != std::string::npos; }
    require(corrupt_replica && contents(locator) == original_replica,
        "later farming replica bypassed its checksum check");
    auto foreign_build = text;
    auto const build = std::string(GROW_IDENTITY);
    foreign_build.replace(foreign_build.find(build), build.size(), "different grow build");
    stream_state::atomic_save(replica, foreign_build + stream_state::digest(foreign_build) + '\n');
    bool foreign_replica = false;
    try { (void)discover(); }
    catch (std::runtime_error const& error)
    { foreign_replica = std::string(error.what()).find("original grow build") != std::string::npos; }
    require(foreign_replica && contents(locator) == original_replica,
        "later farming replica bypassed its saved-build validation");

    auto const exclusive_directory = root / "exclusive-farming-publication";
    fs::create_directory(exclusive_directory);
    auto const exclusive_path = exclusive_directory / "grow-farm.plan";
    FarmingJob::Copy exclusive_copy{exclusive_path, {}, {}, false};
    exclusive_copy.check();
    auto const exclusive_pending = stage_farming_state(exclusive_path, saved.state);
    auto const prepared = contents(exclusive_pending);
    bool exclusive_rejected = false;
    try { stream_state::atomic_save(exclusive_pending, "conflicting pending revision\n", false); }
    catch (std::exception const&) { exclusive_rejected = true; }
    require(exclusive_rejected && contents(exclusive_pending) == prepared,
        "exclusive farming staging replaced an existing pending revision");
    // A competing file appears after the controller validated its absence.
    std::string const competing = "concurrently saved job\n";
    stream_state::atomic_save(exclusive_path, competing);
    exclusive_rejected = false;
    try { publish_farming_state(exclusive_pending, exclusive_path, exclusive_copy.stamp.has_value()); }
    catch (std::runtime_error const&) { exclusive_rejected = true; }
    require(exclusive_rejected && contents(exclusive_pending) == prepared &&
        contents(exclusive_path) == std::vector<uint8_t>(competing.begin(), competing.end()),
        "exclusive farming publication replaced a concurrently appearing job");
    for (auto const& entry : fs::directory_iterator(exclusive_directory))
        require(entry.path() == exclusive_path || entry.path() == exclusive_pending,
            "failed exclusive farming publication retained a private temporary file");
    publish_farming_state(exclusive_pending, exclusive_path, true);
    require(contents(exclusive_path) == prepared && !fs::exists(exclusive_pending),
        "validated farming-parent replacement failed");

    auto partial = make_farm("partial-topology", 1, 3);
    for (auto& directory : partial.state.directories)
    {
        directory.directory_identity = farming_directory_identity(directory.directory);
        directory.resource = directory.storage.resource = 1;
        directory.layout.schedule = 1;
        directory.identity_resources = 2; directory.identity_resource = 1;
    }
    partial.state.estimate.destinations.clear();
    for (auto const& directory : partial.state.directories)
        partial.state.estimate.destinations.push_back(directory.layout);
    auto absent = partial.state.directories.front();
    absent.directory = (root / "offline-topology").string(); absent.directory_identity = "offline";
    absent.resource = absent.storage.resource = 0;
    absent.layout.schedule = 0; absent.layout.resource_rank = 0; absent.layout.resource_lanes = 1;
    absent.identity_resource = 0;
    absent.layout.files = absent.layout.total_members = absent.layout.families = 0;
    partial.state.directories.push_back(absent);
    partial.state.estimate.destinations.push_back(absent.layout);
    partial.state.fixed_extent = true;
    FarmingJob job; job.saved = partial.state;
    for (auto const& directory : partial.state.directories) job.directories.push_back(directory.directory);
    Arguments preview; preview.plan_only = true; preview.max_ram = 512 * mib;
    auto available = farming_run(preview, nullptr, job);
    require(available->available[0] && available->available[1] && !available->available[2],
        "partial topology compared renumbered resource IDs instead of available resource relationships");
    job.saved->directories[1].resource = job.saved->directories[1].storage.resource = 0;
    bool changed_topology = false;
    try { (void)farming_run(preview, nullptr, job); }
    catch (std::runtime_error const& error)
    { changed_topology = std::string(error.what()).find("storage topology changed") != std::string::npos; }
    require(changed_topology, "offline destination bypassed changed storage topology among available members");

#ifdef GROW_TEST_WRAP_PWRITE
    for (bool pending_source : {true, false})
    {
        auto const directory = root / (pending_source ? "changed-pending" : "appeared-parent");
        auto const left = directory / "a", right = directory / "b";
        fs::create_directories(left); fs::create_directory(right);
        Arguments options; options.command = "fill"; options.directory_farm = true;
        options.target = left; options.destinations.push_back(right);
        options.k = 18; options.group_members = 1; options.meta = 0;
        options.farming_max_eligible_groups = 4;
        options.capacity = 64 * mib; options.headroom = 0; options.max_ram = 512 * mib;
        options.max_groups = options.max_total_members = 1;
        auto job = farming_arguments(options);
        auto const pending = right / "grow-farm.plan.next", published = left / "grow-farm.plan";
        std::promise<void> started, release;
        auto ready = started.get_future();
        producer_thread = std::this_thread::get_id();
        background_write_started = &started; background_write_release = release.get_future().share();
        background_check_target = pending.string();
        background_write_timed_out = false; background_check_after = true; hold_background_check = true;
        auto writer = std::async(std::launch::async, [&]
        {
            try
            {
                auto run = farming_run(options, nullptr, job);
                if (!run->feasible()) return run->infeasibility_reason();
            }
            catch (std::runtime_error const& error) { return std::string(error.what()); }
            return std::string{};
        });
        bool const paused = ready.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
        auto const changed = pending_source ? pending : published;
        std::string const replacement = "unrelated replacement\n";
        if (paused) stream_state::atomic_save(changed, replacement);
        release.set_value();
        auto const error = writer.get();
        hold_background_check = false; background_check_after = false; background_check_target.clear();
        background_write_started = nullptr;
        require(paused && !background_write_timed_out &&
            error.find(pending_source ? "pending farming revision changed" : "farming job appeared") != std::string::npos,
            "concurrent farming metadata change bypassed its publication audit: " + error);
        require(!fs::exists(right / "grow-farm.plan") && (!pending_source || !fs::exists(published)),
            "farming metadata conflict published part of the job before rejection");
        require(contents(changed) == std::vector<uint8_t>(replacement.begin(), replacement.end()),
            "farming publication overwrote concurrently replaced metadata");
    }
#endif
}

void output_reservations(fs::path const& root, Members input, Config cfg)
{
    using namespace grow_controller_test;
    auto const directory = root / "output-reservations", scratch = root / "output-reservation-scratch";
    fs::create_directory(directory); fs::create_directory(scratch);
    auto const path = directory / "external.gplot";
    cfg.capacity_bytes = 128 * mib; cfg.memory_limit_bytes = 512 * mib;
    ExternalConfig external; external.scratch.push_back({scratch.string(), 128 * mib});
    external.segment_bytes = mib;
    Construction engine(path.string(), cfg, false, identity(), "external", external);
    engine.confirm_identity();
    require(engine.append(input.subspan(0, 1)).accepted, "output reservation contribution did not fit");
    auto const inspected = ExternalEngine::inspect(path.string(), cfg.memory_limit_bytes);
    require(inspected.metadata_reserved_bytes >= inspected.metadata_occupied_bytes,
        "output reservation fixture exceeded its metadata envelope");
    auto output = snapshot(path, 0); output.reservation = UINT64_MAX;
    require(output_auxiliary(path, output, cfg.memory_limit_bytes).has_value(),
        "active output reservation was not discovered");
    require(output.charge() == inspected.peak_bytes,
        "active external metadata was omitted or counted twice in aggregate capacity");
    (void)output_auxiliary(path, output, cfg.memory_limit_bytes);
    require(output.charge() == inspected.peak_bytes, "repeated active inspection accumulated a reservation");

    // Keep an unrelated file in the owned namespace so ordinary handoff must
    // authenticate and retain its retired metadata, as for changed scratch.
    std::optional<fs::path> owned;
    for (auto const& entry : fs::directory_iterator(scratch))
        if (entry.is_directory()) { owned = entry.path(); break; }
    require(owned.has_value(), "output reservation fixture has no scratch namespace");
    { std::ofstream file(*owned / "preserved"); file.put('!'); }
    (void)engine.finish(); engine.release_external();
    require(!engine.scratch_cleanup_complete, "changed scratch did not retain retired metadata");
    auto const completed = snapshot(path, 0);
    output.bytes = completed.bytes; output.allocated = completed.allocated;
    std::optional<fs::path> holder;
    for (auto const& entry : fs::directory_iterator(directory))
        if (entry.path().filename().string().starts_with(path.filename().string() + ".external.retired-"))
        { holder = entry.path(); break; }
    require(holder.has_value(), "completed output has no authenticated retired holder");
    auto const retired = ExternalEngine::inspect_retired(path.string(), holder->string(), cfg.memory_limit_bytes);
    require(!output_auxiliary(path, output, cfg.memory_limit_bytes) && !output.reservation &&
        output.auxiliary == retired.metadata_reserved_bytes && output.retired == retired.metadata_reserved_bytes &&
        output.charge() == grow_add(std::max(completed.bytes, completed.allocated), retired.metadata_reserved_bytes),
        "retired metadata retained the old target peak or changed ordinary output accounting");

    auto const ordinary = directory / "ordinary.gplot";
    complete(ordinary, input.subspan(0, 1), false, cfg);
    auto plain = snapshot(ordinary, 0); plain.reservation = UINT64_MAX;
    require(!output_auxiliary(ordinary, plain, cfg.memory_limit_bytes) && !plain.reservation && !plain.auxiliary &&
        plain.charge() == std::max(plain.bytes, plain.allocated),
        "ordinary output inspection retained an unrelated reservation or metadata charge");
}

void farming_active_audit(fs::path const& root, Members input, Config cfg)
{
    using namespace grow_controller_test;
    auto const directory = root / "farming-active-audit", scratch = root / "farming-active-scratch";
    fs::create_directory(directory); fs::create_directory(scratch);
    auto const previous = group_path(directory, 0), current = group_path(directory, 1);
    cfg.capacity_bytes = 128 * mib; cfg.memory_limit_bytes = 512 * mib;
    complete(previous, input.subspan(0, 1), false, cfg);
    ExternalConfig external; external.scratch.push_back({scratch.string(), 128 * mib});
    external.segment_bytes = mib;
    {
        Construction engine(current.string(), cfg, false, identity(), "external", external);
        engine.confirm_identity();
        require(engine.append(input.subspan(0, 1)).accepted, "farming audit contribution did not fit");
    }
    FarmingRun farm; farm.memory_limit = cfg.memory_limit_bytes;
    farm.state.directories.push_back({directory.string()}); farm.lanes.resize(1);
    auto& lane = farm.lanes.front(); lane.ordinal = 1;
    lane.outputs = {snapshot(previous, 0), snapshot(current, 1)};
    farm.check();
    std::array<uint8_t, 1> const active{1};

    // An inspector can load an earlier checkpoint immediately before the
    // writer durably advances it and deletes a consumed segment. Recreate
    // that reader view without depending on thread or disk timing.
    std::optional<fs::path> segment;
    for (auto const& entry : fs::recursive_directory_iterator(scratch))
        if (entry.path().filename().string().starts_with("run-")) { segment = entry.path(); break; }
    require(segment && fs::remove(*segment), "farming audit fixture has no external segment");
    farm.check(active);
    bool rejected = false;
    try { farm.check(); }
    catch (std::runtime_error const& error)
    { rejected = std::string_view(error.what()) == "required external segment is missing"; }
    require(rejected, "inactive farming audit skipped a missing required segment");

    auto rejects = [&](char const* message)
    {
        bool rejected = false;
        try { farm.check(active); } catch (std::runtime_error const&) { rejected = true; }
        require(rejected, message);
    };
    auto const previous_work = previous.string() + ".external";
    fs::create_directory(previous_work);
    rejects("active lane skipped another group's external metadata audit");
    fs::remove(previous_work);
    auto const unexpected = current.string() + ".external.unrecognized";
    fs::create_directory(unexpected);
    rejects("active farming audit accepted an unrecognized output suffix");
    fs::remove(unexpected);
    auto const malformed = current.string() + ".external.retired-invalid";
    fs::create_directory(malformed);
    rejects("active farming audit accepted a malformed retired holder name");
    fs::remove(malformed);
    auto const unplanned = group_path(directory, 2);
    { std::ofstream output(unplanned); output.put('!'); }
    rejects("active farming audit accepted an unplanned ordinary output"); fs::remove(unplanned);

    auto const retired = current.string() + ".external.retired-" + std::string(64, 'f');
    fs::rename(current.string() + ".external", retired);
    fs::remove(fs::path(retired) / "checkpoint");
    farm.check(active);
    rejected = false;
    try { farm.check(); } catch (std::runtime_error const&) { rejected = true; }
    require(rejected, "inactive farming audit skipped unauthenticated retired metadata");
    { std::ofstream output(previous, std::ios::app); output.put('!'); }
    rejects("active farming audit skipped a changed completed output");
}

void prepared_producer(fs::path const& root, Members input)
{
    using namespace grow_controller_test;
    auto const requested = identity_from(identity());
    Fixtures fixture{requested, {}, 0};
    for (uint16_t index = 0; index < 4; ++index)
    {
        auto const path = root / ("prepared-producer-member-" + std::to_string(index));
        PlotData data; data.t3_proof_fragments.assign(input[index].raw().begin(), input[index].raw().end());
        fixture.bytes = sum(fixture.bytes, PlotFile::writeData(path.string(), data, requested.params, index, requested.memo));
        fixture.paths.push_back(path);
    }
    auto const scratch = root / "prepared-producer-scratch"; fs::create_directory(scratch);
    ExternalConfig ext{{{scratch.string(), 128 * mib}}, mib, 2};
    Arguments args; args.backend = "cpu"; args.max_ram = 512 * mib; args.batch = args.max_members = 4;
    args.window = 65536; args.workers = 2; args.checkpoint_bytes = mib;
    auto scheduler = std::make_shared<CodecScheduler>(unsigned(args.workers));
    uint64_t const metadata = sum(fixture.bytes, sum(mib, CodecScheduler::memory_bytes(unsigned(args.workers))));
    auto cfg = grow_controller_test::configuration(args, 128 * mib, metadata); cfg.cpu_scheduler = scheduler;
    auto const path = root / "prepared-producer.gplot", reference = root / "prepared-producer-reference.gplot";
    { Engine engine(reference.string(), cfg, false, identity()); engine.confirm_identity();
      require(engine.append(input.subspan(0, 4)).accepted, "prepared producer reference rejected"); }
    Construction construction(path.string(), cfg, false, identity(), "external", ext); construction.confirm_identity();
    auto const fixed = sum(metadata, sum(construction.memory_bytes(4, true), ExternalEngine::input_memory_bytes(cfg, 4)));
    Backend backend; Producer producer{requested, &fixture, backend, args, args.max_ram, fixed, fixed};
    auto prepared = construction.external->prepare_input(4, fragment_bound(requested.params) / sizeof(uint64_t));
    require(bool(prepared), "prepared producer reservation rejected");
    producer.fill_input(*prepared, 0, 0, "test bounded fixture member ");
    require(prepared->sealed() && backend.generated_members == 4 && producer.peak <= args.max_ram,
        "prepared fixture producer lost completion or admission");
    OrderedWriter worker([] {}); auto ready = worker.submit([&]
    { require(construction.append(*prepared, 4).accepted, "prepared Construction append rejected"); });
    ready.get(); require(construction.staged_batches == 1 && !construction.shape_recorded,
        "prepared Construction append bypassed accounting");
    prepared.reset(); construction.finish();
    require(contents(path) == contents(reference), "prepared fixture producer changed canonical group bytes");
    auto const prefix_path = root / "prepared-producer-prefix.gplot", prefix_reference = root / "prepared-producer-prefix-reference.gplot";
    { Engine engine(prefix_reference.string(), cfg, false, identity()); engine.confirm_identity();
      require(engine.append(input.subspan(0, 2)).accepted, "prepared producer prefix reference rejected"); }
    Construction prefix(prefix_path.string(), cfg, false, identity(), "external", ext); prefix.confirm_identity();
    Producer prefix_producer{requested, &fixture, backend, args, args.max_ram, fixed, fixed};
    auto prefix_input = prefix.external->prepare_input(4, fragment_bound(requested.params) / sizeof(uint64_t));
    require(bool(prefix_input), "prepared producer prefix reservation rejected"); size_t polls = 0;
    auto const generated = prefix_producer.fill_input(*prefix_input, 0, 0, "test bounded prefix member ", [&] { return ++polls == 2; });
    require(generated == 2 && prefix_input->size() == 2 && prefix_input->sealed() &&
        prefix_producer.fill_stop == Producer::FillStop::requested, "prepared producer did not seal its completed stop prefix");
    auto prefix_ready = worker.submit([&]
    { require(prefix.append(*prefix_input, 2).accepted, "prepared Construction prefix append rejected"); });
    prefix_ready.get(); prefix_input.reset(); prefix.finish();
    require(contents(prefix_path) == contents(prefix_reference), "stopped prepared producer changed prefix bytes");

}

int main(int argc, char** argv) try
{
    require(argc == 1 || (argc == 2 && (std::string(argv[1]) == "--prefetch-only" ||
        std::string(argv[1]) == "--farming-audit-only" || std::string(argv[1]) == "--replot-capacity-only" || std::string(argv[1]) == "--prepared-input-only")), "usage: grow-merge-test [--prefetch-only|--farming-audit-only|--replot-capacity-only|--prepared-input-only]");
    bool const prefetch_only = argc == 2 && std::string(argv[1]) == "--prefetch-only";
    bool const farming_audit_only = argc == 2 && std::string(argv[1]) == "--farming-audit-only";
    test_producer_options();
    controller_wakeups();
    test_destination_temporary();
    metadata_options();
    external_batch_selection();
    phase_worker_demand();
    require(filesystem_file_bound(0xef53, 4096) == uint64_t(UINT32_MAX) * 4096 &&
        filesystem_file_bound(0x9123683e, 4096) == uint64_t(INT64_MAX), "filesystem file-size admission differs");
    auto name = (fs::temp_directory_path() / "xchplot2-merge-XXXXXX").string();
    require(::mkdtemp(name.data()), "cannot create merge test directory");
    fs::path root(name); struct Cleanup { fs::path root; ~Cleanup() { fs::remove_all(root); } } cleanup{root};
    auto input = members(); auto cfg = configuration();
    if (argc == 2 && std::string(argv[1]) == "--replot-capacity-only")
    {
        temporary_admission(root, identity()); replot_external_projection(root, identity()); replot_capacity_prefix();
        std::cout << "projected migration output/scratch admission checks passed\n"; return 0;
    }
    joined_worker_cpu(input);
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--prepared-input-only"))
    {
        auto const parent = fs::absolute(argv[0]).parent_path(); require_disk_scratch(parent);
        auto name = (parent / "prepared-producer-XXXXXX").string();
        require(::mkdtemp(name.data()), "cannot create prepared producer test directory");
        Cleanup external_cleanup{fs::path(name)}; prepared_producer(external_cleanup.root, input);
        if (argc != 1) { std::cout << "bounded fixture producer and Construction checks passed\n"; return 0; }
    }
    if (farming_audit_only)
    {
        auto const parent = fs::absolute(argv[0]).parent_path(); require_disk_scratch(parent);
        auto name = (parent / "farming-audit-XXXXXX").string();
        require(::mkdtemp(name.data()), "cannot create farming audit test directory");
        Cleanup external_cleanup{fs::path(name)};
        bounded_saved_reads(external_cleanup.root);
        farming_saved_reads(external_cleanup.root);
        replot_journal_protocol(external_cleanup.root);
        replot_capacity_prefix();
        replot_destination_namespace(external_cleanup.root);
        replot_retirement_protocol(external_cleanup.root);
        replot_publication_names(external_cleanup.root, input);
        farming_compatibility_format(external_cleanup.root);
        farming_joint_forecasts(external_cleanup.root);
        farming_reader_scope();
        farming_legacy_format();
        farming_compact_format();
        farming_replica_staging(external_cleanup.root);
        farming_parallel_audits(external_cleanup.root, input);
        farming_capacity_extents(external_cleanup.root);
        farming_extent_admission(external_cleanup.root);
        farming_preview_handoff(external_cleanup.root);
        farming_active_audit(external_cleanup.root, input, cfg);
        output_reservations(external_cleanup.root, input, cfg);
        successor_staging(external_cleanup.root, input, cfg);
        external_finish_grants(external_cleanup.root, input, cfg);
        std::cout << "farming extent admission, ownership and inactive metadata audit checks passed\n"; return 0;
    }
    writers(root); group_prefetch(root, input); destination_admission(); execution_planning();
    if (prefetch_only) { std::cout << "next-group prefetch and receiver admission checks passed\n"; return 0; }
    construction_selection(); construction_histories(); cache_frontiers(root, cfg); temporary_admission(root, identity()); replot_external_projection(root, identity());
    {
        // External scratch needs physical allocation, while the existing
        // same-file cases may keep using a RAM-backed temporary directory.
        auto const parent = fs::absolute(argv[0]).parent_path(); require_disk_scratch(parent);
        auto name = (parent / "external-observation-XXXXXX").string();
        require(::mkdtemp(name.data()), "cannot create external observation test directory");
        Cleanup external_cleanup{fs::path(name)};
        successor_staging(external_cleanup.root, input, cfg);
        external_finish_grants(external_cleanup.root, input, cfg);
        external_batch_observations(external_cleanup.root, input, cfg);
        replot_journal_protocol(external_cleanup.root);
        replot_capacity_prefix();
        replot_destination_namespace(external_cleanup.root);
        replot_retirement_protocol(external_cleanup.root);
        replot_publication_names(external_cleanup.root, input);
        bounded_saved_reads(external_cleanup.root);
        farming_saved_reads(external_cleanup.root);
        farming_compatibility_format(external_cleanup.root);
        farming_joint_forecasts(external_cleanup.root);
        farming_reader_scope();
        farming_legacy_format();
        farming_compact_format();
        farming_replica_staging(external_cleanup.root);
        farming_parallel_audits(external_cleanup.root, input);
        farming_capacity_extents(external_cleanup.root);
        farming_extent_admission(external_cleanup.root);
        farming_preview_handoff(external_cleanup.root);
        farming_active_audit(external_cleanup.root, input, cfg);
        output_reservations(external_cleanup.root, input, cfg);
    }
    test_construction_history(root, input, identity(), cfg); parity(root, input, cfg);
    retuning(root, input, cfg); worker_grants(root, input, cfg); placement_costs(); placement(root, cfg); packing_windows(root, cfg);
    placement_recovery(root, cfg); placement_bounds(root, cfg); layout(root, input, cfg); checkpoint_geometry(root, input, cfg);
    single_pass(root, input, cfg);
    contribution_pipeline(root, cfg);
    ordinary_prefix(root, cfg); balanced_traffic(root, cfg);
    admission(root, input, cfg); recovery(root, input, cfg);
    auto four = root / "four-inputs"; fs::create_directory(four); cfg.merge_inputs = 4;
    cfg.memory_limit_bytes = MergeEngine::memory_bytes(18, cfg, 16);
    parity(four, input, cfg); admission(four, input, cfg); recovery(four, input, cfg);
    auto overlap = root / "overlap"; fs::create_directory(overlap);
    cfg.io_overlap = true; cfg.window_bytes = 65536; cfg.checkpoint_bytes = 1024 * 1024;
    cfg.capacity_bytes = 12 * 1024 * 1024;
    cfg.memory_limit_bytes = std::max(Engine::memory_bytes(18, cfg, 16), MergeEngine::memory_bytes(18, cfg, 16));
    parity(overlap, input, cfg);
    capacity_pressure(overlap, input, cfg);
    for (auto const* point : {"merge-pack-committed:2", "merge-layout-partial", "merge-layout-move-committed:2"})
        interruption(overlap, input, cfg, point);
    std::cout << "same-file balanced merge parity, recycling and recovery checks passed\n";
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
