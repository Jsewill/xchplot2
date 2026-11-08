// Included by group_merge_test after the controller's DirectoryLock fixture.
#pragma once
#include "host/GroupFarmProgress.hpp"
#include <sys/wait.h>

namespace group_farm_progress_test
{
using namespace group_grow;
using grow_controller_test::DirectoryLock;
using grow_controller_test::mib;
namespace fs = std::filesystem;
inline void check(bool value, char const* message) { require(value, message); }
template<class Function> inline void rejects(Function action, char const* message)
{
    bool caught = false; try { action(); } catch (std::runtime_error const&) { caught = true; } check(caught, message);
}
inline FarmingState fixture(fs::path const& root)
{
    FarmingState state; state.version = 8; state.policy_revision = state.revision = 1;
    state.config.k = state.identity_k = 18; state.config.group_members_override = 2;
    state.capacity_fill = state.fixed_extent = true; state.capacity = 64 * mib; state.headroom = 0;
    state.manifest_reserved_bytes = 256 * 1024; state.progress_reserved_bytes = 64 * 1024;
    state.job_id = std::string(64, 'a'); state.fingerprint = std::string(64, 'b');
    state.seed = std::string(64, 'c'); state.farmer = std::string(96, 'd'); state.pool = std::string(64, 'e');
    for (auto const* name : {"left", "right"})
    {
        auto const path = root / name; fs::create_directory(path);
        FarmingDirectory directory; directory.directory = file_io::path_text(path);
        directory.directory_identity = farming_directory_identity(path); directory.capacity_bytes = 32 * mib;
        directory.storage.capacity_bytes = directory.capacity_bytes; directory.storage.known_device = true;
        directory.resource = directory.identity_resource = state.directories.size(); directory.identity_resources = 2;
        directory.layout.schedule = directory.resource; directory.layout.meta_slots = 256;
        directory.layout.files = 3; directory.layout.group_members = 2; directory.layout.total_members = 6;
        directory.layout.fixed_members = true; directory.extents = {2, 2, 2};
        state.directories.push_back(std::move(directory));
    }
    return state;
}
inline std::string checksummed(FarmingProgress const& value)
{ auto body = farming_progress_text(value); return body + stream_state::digest(body) + '\n'; }
inline void run(fs::path const& parent)
{
    auto const root = parent / "local-farming-progress"; fs::create_directory(root);
    auto state = fixture(root); auto const global_bytes = farming_state_text(state);
    auto const policy = farming_policy_digest(state); auto const memory = 64ull * 1024 * 1024;
    auto initial = capture_farming_progress(state, 0, policy);
    check(initial.sequence == 1 && !initial.summary.completed_files && !initial.summary.pending_files &&
        initial.summary.observed_at.empty(), "initial local progress invented an observation");
    check(checksummed(initial).size() <= farming_progress_size_bound(state, 0) &&
        farming_progress_memory_bound(state, 0) >= 64 * 1024,
        "local progress size or fixed memory allowance was understated");
    check(farming_progress_text(parse_farming_progress(farming_progress_text(initial), memory)) == farming_progress_text(initial),
        "local progress canonical bytes changed during reload");
    rejects([&] { parse_farming_progress(farming_progress_text(initial), 1); }, "local progress parsing ignored its memory budget");
    auto const path = file_io::native_path(state.directories[0].directory);
    DirectoryLock lock(path); FarmingMetadataDirectory metadata(path, false);
    check(!load_farming_progress(metadata, memory) && !fs::exists(path / ".xchplot2"),
        "read-only progress lookup created bookkeeping");
    auto saved = save_farming_progress(metadata, initial, memory);
    check(saved.allocated_bytes && farming_progress_allocation(metadata) == saved.allocated_bytes,
        "local progress allocation was not charged");
    auto loaded = load_farming_progress(metadata, memory);
    check(loaded && loaded->stamp == saved.stamp && loaded->progress.source_policy == policy,
        "saved local progress lost its source binding");
    auto observed = state.directories[0]; observed.extents[0] = 1; observed.sealed_files = 1;
    FarmingProgressSummary summary; summary.observed_at = "2026-11-08T12:00:00Z"; summary.writer_state = "running";
    summary.pending_files = 1; summary.charged_bytes = 4096; summary.reserved_bytes = 8192;
    auto sealed = advance_farming_progress(initial, observed, summary);
    check(!sealed.summary.completed_files && sealed.sealed_files == 1,
        "sealing an unfinished output invented completed payload");
    saved = save_farming_progress(metadata, sealed, memory, saved.stamp);
    auto current = state;
    check(overlay_farming_progress(current, 0, sealed, policy) == FarmingProgressOverlay::applied &&
        current.directories[0].extents[0] == 1 && current.directories[0].sealed_files == 1 &&
        current.directories[1].sealed_files == 0 && current.revision == state.revision && current.parent_digest == state.parent_digest &&
        farming_state_text(state) == global_bytes,
        "local progress changed another grant or the immutable global parent");
    auto cached = current;
    check(overlay_farming_progress(cached, 0, sealed, policy) == FarmingProgressOverlay::covered,
        "already incorporated local progress was not recognized");
    auto bad = sealed; bad.job_id = std::string(64, 'f');
    rejects([&] { auto target = state; overlay_farming_progress(target, 0, bad); }, "foreign local progress job was accepted");
    bad = sealed; bad.directory_identity = state.directories[1].directory_identity;
    rejects([&] { auto target = state; overlay_farming_progress(target, 0, bad); }, "foreign local progress destination was accepted");
    bad = sealed; bad.source_policy = std::string(64, 'f');
    rejects([&] { auto target = state; overlay_farming_progress(target, 0, bad); }, "foreign local progress policy was accepted");
    bad = sealed; ++bad.source_revision;
    rejects([&] { auto target = state; overlay_farming_progress(target, 0, bad); }, "future local progress revision was accepted");
    bad = sealed; ++bad.granted_extents[1]; ++bad.extents[1];
    rejects([&] { auto target = state; overlay_farming_progress(target, 0, bad); }, "local progress enlarged a saved grant");
    bad = sealed; bad.extents[1] = 3;
    rejects([&] { farming_progress_text(bad); }, "local progress grew membership beyond its immutable grant");
    bad = sealed; bad.summary.completed_files = 2; bad.summary.completed_members = 3;
    rejects([&] { farming_progress_text(bad); }, "local observation claimed completion beyond its sealed prefix");
    bad = sealed; bad.summary.pending_files = bad.extents.size() + 1;
    rejects([&] { farming_progress_text(bad); }, "local observation claimed impossible active outputs");
    bad = sealed; bad.summary.charged_bytes = bad.capacity_bytes; bad.summary.reserved_bytes = 1;
    rejects([&] { farming_progress_text(bad); }, "local observation exceeded its destination capacity grant");
    auto changed = state; ++changed.directories[0].identity_epoch;
    rejects([&] { overlay_farming_progress(changed, 0, sealed, policy); }, "local progress adopted a changed key namespace");
    changed = state; ++changed.revision; changed.directories[0].extents[1] = 1;
    rejects([&] { overlay_farming_progress(changed, 0, sealed, policy); }, "local progress ignored a changed global target");
    auto compatible = state; ++compatible.revision;
    check(overlay_farming_progress(compatible, 0, sealed, policy) == FarmingProgressOverlay::applied,
        "compatible older local progress revision was rejected");
    auto paused = state; paused.control = paused.directories[0].control = "paused";
    paused.policy_revision = paused.revision = 2;
    check(overlay_farming_progress(paused, 0, sealed) == FarmingProgressOverlay::applied &&
        paused.control == "paused" && paused.directories[0].control == "paused" && paused.directories[0].extents.size() == 3,
        "local progress reopened a paused policy or discarded its grants");
    auto stale = current; stale.policy_revision = stale.revision = 2; stale.control = "retired";
    check(overlay_farming_progress(stale, 0, sealed) == FarmingProgressOverlay::covered && stale.control == "retired",
        "dominated stale progress overrode a newer policy");
    stale.directories[0].extents[0] = 2;
    rejects([&] { overlay_farming_progress(stale, 0, sealed); }, "conflicting stale completion was silently discarded");
    summary.completed_files = summary.completed_members = 1; summary.pending_files = 0;
    summary.observed_at = "2026-11-08T12:01:00Z";
    auto completed = advance_farming_progress(sealed, observed, summary);
    rejects([&] { save_farming_progress(metadata, completed, memory, loaded->stamp); },
        "stale local writer bypassed compare-before-publication fencing");
    auto const published = metadata.path / farming_progress_detail::filename;
    auto const preserved = checksummed(sealed);
    stream_state::atomic_save(published, preserved.substr(0, preserved.size() - 1), true);
    rejects([&] { load_farming_progress(metadata, memory); }, "truncated local progress checksum was accepted");
    stream_state::atomic_save(published, preserved, true); loaded = load_farming_progress(metadata, memory);
    auto corrupt = preserved; corrupt[0] = 'x'; stream_state::atomic_save(published, corrupt, true);
    rejects([&] { load_farming_progress(metadata, memory); }, "corrupt local progress was accepted");
    stream_state::atomic_save(published, preserved, true); loaded = load_farming_progress(metadata, memory);
    stream_state::atomic_save(metadata.path / "grow-farm.progress.next", std::string("torn unpublished record"), false);
    stream_state::atomic_save(metadata.path / "grow-farm.progress.next.partial.interrupted", std::string("torn temporary record"), false);
    auto interrupted = load_farming_progress(metadata, memory);
    check(interrupted && interrupted->progress.sequence == sealed.sequence && interrupted->allocated_bytes > loaded->allocated_bytes,
        "torn staging replaced committed progress or escaped allocation charging");
    saved = save_farming_progress(metadata, completed, memory, loaded->stamp);
    check(!fs::exists(metadata.path / "grow-farm.progress.next") &&
        !fs::exists(metadata.path / "grow-farm.progress.next.partial.interrupted"),
        "local writer retained abandoned publication staging");
    auto stopped_directory = observed; stopped_directory.extents.resize(1); stopped_directory.fill_stop = "capacity";
    auto stopped = advance_farming_progress(completed, stopped_directory, summary);
    auto stopped_state = state;
    check(overlay_farming_progress(stopped_state, 0, stopped, policy) == FarmingProgressOverlay::applied &&
        stopped_state.directories[0].fill_stop == "capacity" && stopped_state.directories[0].extents.size() == 1,
        "local terminal progress did not conservatively release an unused suffix");
    auto extended = state; ++extended.revision; extended.directories[0].extents.push_back(2);
    rejects([&] { overlay_farming_progress(extended, 0, stopped, policy); },
        "old terminal progress discarded newly saved global grants");
    check(extended.directories[0].extents.size() == 4 && !extended.directories[0].sealed_files,
        "refused old terminal progress changed the newer global prefix");
    for (auto const* reason : {"capacity", "group-limit", "total-member-limit", "farming-limit", "metadata-capacity"})
    {
        auto terminal_directory = observed; terminal_directory.extents.resize(1); terminal_directory.fill_stop = reason;
        auto terminal_progress = advance_farming_progress(completed, terminal_directory, summary);
        auto converged = state; overlay_farming_progress(converged, 0, terminal_progress, policy); ++converged.revision;
        check(valid_farming_extent_revision(state, converged), "valid local terminal prefix could not converge globally");
        auto nonterminal = converged; nonterminal.directories[0].fill_stop.clear();
        check(!valid_farming_extent_revision(state, nonterminal), "nonterminal local trim was accepted as global progress");
        auto legacy = state; legacy.version = 7; auto legacy_next = converged; legacy_next.version = 7;
        check(valid_farming_extent_revision(legacy, legacy_next) == (std::string_view(reason) == "capacity"),
            "local terminal convergence relaxed a legacy trim rule");
    }
    auto prior_seal = state; prior_seal.directories[0].sealed_files = 1;
    auto changed_seal = prior_seal; changed_seal.directories[0].sealed_files = 2;
    changed_seal.directories[0].extents = {1, 2}; changed_seal.directories[0].fill_stop = "group-limit";
    check(!valid_farming_extent_revision(prior_seal, changed_seal), "terminal convergence changed an already sealed member count");
    saved = save_farming_progress(metadata, stopped, memory, saved.stamp);
    auto reopened = stopped_directory; reopened.fill_stop.clear(); reopened.extents.push_back(2);
    rejects([&] { advance_farming_progress(stopped, reopened, summary); }, "local progress reopened a stopped capacity grant");
    rejects([&] { retire_farming_progress(metadata, state, 0, saved, policy); },
        "unincorporated local completion was retired");
    retire_farming_progress(metadata, stopped_state, 0, saved, policy);
    check(!load_farming_progress(metadata, memory) && !farming_progress_allocation(metadata),
        "incorporated local progress was not durably retired");
    auto right = capture_farming_progress(state, 1, policy); right.progress_reserved_bytes = 1;
    DirectoryLock right_lock(file_io::native_path(state.directories[1].directory));
    FarmingMetadataDirectory right_metadata(file_io::native_path(state.directories[1].directory), false);
    rejects([&] { save_farming_progress(right_metadata, right, memory); }, "local progress bypassed its reserved metadata allowance");
    check(!load_farming_progress(right_metadata, memory), "refused local journal was published");
    auto one_copy_state = state; one_copy_state.progress_reserved_bytes = saved.allocated_bytes;
    right = capture_farming_progress(one_copy_state, 1);
    auto one_copy = save_farming_progress(right_metadata, right, memory);
    auto refreshed = advance_farming_progress(right, state.directories[1], right.summary);
    rejects([&] { save_farming_progress(right_metadata, refreshed, memory, one_copy.stamp); },
        "local progress replacement spent the already charged record's allowance twice");
    auto unchanged = load_farming_progress(right_metadata, memory);
    check(unchanged && unchanged->stamp == one_copy.stamp && unchanged->progress.sequence == 1,
        "refused local progress replacement changed the published record");
    auto headroom_state = state;
    auto const headroom_path = root / "headroom"; fs::create_directory(headroom_path);
    headroom_state.directories[0].directory = file_io::path_text(headroom_path);
    headroom_state.directories[0].directory_identity = farming_directory_identity(headroom_path);
    DirectoryLock headroom_lock(headroom_path); FarmingMetadataDirectory headroom_metadata(headroom_path, false); headroom_metadata.create();
    file_io::Space free{}; check(!file_io::space_path(headroom_path, &free), "cannot inspect progress headroom fixture");
    headroom_state.headroom = uint64_t(free.f_bavail) * uint64_t(free.f_frsize);
    auto no_space = capture_farming_progress(headroom_state, 0);
    rejects([&] { save_farming_progress(headroom_metadata, no_space, memory); },
        "local journal publication consumed declared filesystem headroom");
    check(!load_farming_progress(headroom_metadata, memory) && !farming_progress_allocation(headroom_metadata),
        "headroom rejection left a published or staged local journal");
    headroom_state.headroom = 0;
    auto with_space = capture_farming_progress(headroom_state, 0);
    auto admitted = save_farming_progress(headroom_metadata, with_space, memory);
    check(admitted.progress.headroom_bytes == 0, "local journal ignored its captured headroom policy");
    auto large_state = state; large_state.revision = large_state.policy_revision = 2048;
    auto& large_directory = large_state.directories[0]; large_directory.extents.assign(4096, 2);
    for (uint64_t cohort = 0; cohort < 1023; ++cohort)
    {
        auto row = farming_cohort(state, state.directories[0], 0);
        row.first_file = cohort * 4; row.identity_epoch = cohort; row.layout.schedule = cohort;
        large_directory.cohorts.push_back(row);
    }
    large_directory.layout_first = 4092; large_directory.identity_epoch = 1023; large_directory.layout.schedule = 1023;
    auto large = capture_farming_progress(large_state, 0);
    auto const large_body = farming_progress_text(large);
    check(large_body.size() + 65 <= farming_progress_size_bound(large_state, 0) &&
        farming_progress_text(parse_farming_progress(large_body, farming_progress_memory_bound(large_state, 0))) == large_body,
        "large cohort progress exceeded its declared serialization or memory bound");
    large.granted_extents.resize(farming_detail::maximum_files + 1, 2);
    rejects([&] { farming_progress_text(large); }, "local progress exceeded its permanent file work bound");
    auto former = state; former.config.max_eligible_families_per_resource = 1000000;
    former.config.deadline_seconds = former.config.challenge_seconds = 1000000;
    former.capacity = uint64_t(1) << 51;
    former.directories[0].capacity_bytes = former.directories[0].storage.capacity_bytes = uint64_t(1) << 50;
    former.directories[0].extents.assign(131072, 2);
    former.progress_reserved_bytes = 2 * farming_update_aligned(farming_progress_size_bound(former, 0), 4096);
    former.manifest_reserved_bytes += former.progress_reserved_bytes;
    auto former_grant = capture_farming_progress(former, 0);
    auto former_terminal = former.directories[0]; former_terminal.sealed_files = 1;
    former_terminal.extents.resize(1); former_terminal.fill_stop = "capacity";
    auto former_progress = advance_farming_progress(former_grant, former_terminal, {});
    auto former_loaded = parse_farming_progress(farming_progress_text(former_progress), memory);
    auto incorporated = former; overlay_farming_progress(incorporated, 0, former_loaded); ++incorporated.revision;
    auto const retained_bound = farming_progress_memory_bound(former_loaded);
    check(retained_bound > farming_progress_memory_bound(incorporated, 0) &&
        retained_bound >= former_loaded.granted_extents.capacity() * sizeof(uint16_t) &&
        overlay_farming_progress(incorporated, 0, former_loaded) == FarmingProgressOverlay::covered,
        "covered old journal lost its larger retained grant and parsing memory allowance");
    std::cout << "local farming progress grants, checksums, monotone publication and last-observed status passed\n";
}
inline void relocation(fs::path const& parent, Members input, Config cfg)
{
    using namespace grow_controller_test;
    auto bytes = [](fs::path const& path)
    {
        auto file = group_tools::input(file_io::path_text(path));
        return std::string(std::istreambuf_iterator<char>(file), {});
    };
    for (bool copied : {false, true})
        for (auto const* point : {"", "farm-relocation-all-staged", "farm-relocation-progress-retired:1", "farm-plan-published:1", "early-staging"})
        {
            auto const root = parent / (std::string("progress-relocation-") + (copied ? "copy-" : "rename-") +
                (std::string_view(point).empty() ? "normal" : point));
            fs::create_directory(root);
            auto const first = root / "first", second = root / "second", moved = root / "moved";
            fs::create_directory(first); fs::create_directory(second);
            Arguments args; args.command = "fill"; args.directory_farm = true; args.target = first; args.destinations = {second};
            args.max_ram = 512 * mib; args.headroom = 0; args.capacity = 128 * mib;
            args.k = 18; args.strength = 2; args.group_members = 2; args.max_groups = 1; args.max_total_members = 2;
            args.farming_max_eligible_groups = 8; args.backend = "cpu"; args.workers = args.batch = 1;
            args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, 'b');
            args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
            Keys keys(args); auto initial_job = farming_arguments(args); auto initial = farming_run(args, &keys, initial_job);
            check(initial->feasible(), "relocation progress fixture was infeasible");
            cfg.capacity_bytes = 64 * mib; cfg.memory_limit_bytes = args.max_ram;
            for (size_t lane = 0; lane < initial->lanes.size(); ++lane)
            {
                Construction output(group_path(lane ? second : first, 0).string(), cfg, false,
                    initial_identity(initial->identity(keys, lane, 0)), "incremental");
                output.confirm_identity(); check(output.append(input.subspan(0, 1)).accepted, "relocation contribution was rejected");
                (void)output.finish();
            }
            // Leave the completed one-member targets only in the local
            // journals; the immutable global source still grants two.
            initial->all_replicas = false;
            for (size_t lane = 0; lane < initial->lanes.size(); ++lane)
                initial->seal_extent(lane, 0, 1, "capacity", initial->metadata);
            initial.reset();
            auto const first_payload = bytes(group_path(first, 0)), second_payload = bytes(group_path(second, 0));
            if (copied)
            {
                fs::copy(first, moved, fs::copy_options::recursive);
                fs::rename(first, root / "detached-original");
            }
            else fs::rename(first, moved);
            if (copied)
            {
                auto foreign = args; foreign.target = moved; foreign.destinations.clear(); foreign.resume = foreign.resume_set = true;
                rejects([&] { auto job = farming_arguments(foreign); (void)farming_run(foreign, &keys, job); },
                    "copied observations gained native authority without explicit relocation");
            }
            auto const first_plan = bytes(farming_metadata_path(moved, "grow-farm.plan", false));
            auto const second_plan = bytes(farming_metadata_path(second, "grow-farm.plan", false));
            auto const first_progress = bytes(farming_metadata_path(moved, "grow-farm.progress", false));
            auto const second_progress = bytes(farming_metadata_path(second, "grow-farm.progress", false));
            auto review_args = args; review_args.command = "plan"; review_args.plan_only = true;
            review_args.target = second; review_args.destinations.clear(); review_args.relocations = {{first, moved}};
            review_args.save_plan = root / "relocation.plan";
            auto reviewed_job = farming_arguments(review_args); auto review = farming_run(review_args, &keys, reviewed_job);
            check(review->feasible() && review->state.policy_revision == review->state.revision,
                "explicit relocation did not produce a reviewed policy handoff");
            save_farming_review(review_args, reviewed_job, *review); review.reset();
            check(bytes(farming_metadata_path(moved, "grow-farm.plan", false)) == first_plan &&
                bytes(farming_metadata_path(second, "grow-farm.plan", false)) == second_plan &&
                bytes(farming_metadata_path(moved, "grow-farm.progress", false)) == first_progress &&
                bytes(farming_metadata_path(second, "grow-farm.progress", false)) == second_progress,
                "read-only relocation preview changed source plans or committed local progress");
            auto execute = [&]
            {
                Arguments requested; requested.command = "fill"; requested.directory_farm = true;
                requested.from_plan = review_args.save_plan; requested.max_ram = args.max_ram;
                requested.backend = "cpu"; requested.workers = requested.batch = 1;
                auto job = farming_arguments(requested); auto run = farming_run(requested, &keys, job);
                check(run->feasible(), "reviewed relocation was not executable");
            };
            if (!std::string_view(point).empty())
            {
                auto const child = ::fork(); check(child >= 0, "cannot fork relocation handoff");
                if (!child)
                {
                    ::setenv("XCHPLOT2_GROW_FAILPOINT", std::string_view(point) == "early-staging" ? "farm-relocation-all-staged" : point, 1);
                    execute(); std::_Exit(0);
                }
                int status = 0;
                check(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
                    "relocation did not reach its durable handoff boundary");
                if (std::string_view(point) == "early-staging")
                    check(!file_io::unlink_file(farming_metadata_path(moved, "grow-farm.plan.next", false)),
                        "cannot retain an incomplete relocation preparation fixture");
                auto inspect = args; inspect.command = "plan"; inspect.plan_only = true; inspect.target = moved;
                inspect.destinations.clear(); inspect.resume = inspect.resume_set = true;
                auto source = farming_arguments(inspect); auto readonly = farming_run(inspect, &keys, source);
                check(readonly->feasible(), "read-only pending relocation lost authenticated progress authority"); readonly.reset();
                auto resume = args; resume.target = moved; resume.destinations.clear(); resume.resume = resume.resume_set = true;
                auto source_job = farming_arguments(resume); auto recovered = farming_run(resume, &keys, source_job);
                check(recovered->feasible(), "relocation handoff could not recover without repeating --relocate");
            }
            else execute();
            auto const after = load_farming_state(farming_metadata_path(moved, "grow-farm.plan", false), args.max_ram);
            check(farming_state_text(after) == farming_state_text(load_farming_state(farming_metadata_path(second, "grow-farm.plan", false), args.max_ram)) &&
                after.directories[0].directory_identity == farming_directory_identity(moved) &&
                after.directories[0].sealed_files == 1 && after.directories[0].extents[0] == 1 &&
                after.directories[1].sealed_files == 1 && after.directories[1].extents[0] == 1 &&
                bytes(group_path(moved, 0)) == first_payload && bytes(group_path(second, 0)) == second_payload,
                "relocation handoff lost a committed prefix, payload or converged native authority");
            FarmingMetadataDirectory current(moved, false);
            auto progress = load_farming_progress(current, args.max_ram);
            check(progress && progress->progress.directory_identity == after.directories[0].directory_identity,
                "relocation retained foreign-native observations under the new global state");
        }
    std::cout << "reviewed native relocation and interrupted progress handoff passed\n";
}
} // namespace group_farm_progress_test
