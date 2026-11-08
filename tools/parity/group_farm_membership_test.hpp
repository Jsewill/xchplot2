// Included by the controller's existing farming audit fixture.
#pragma once
#include "host/GroupFarmMembership.hpp"

namespace group_farm_membership_test
{
using namespace group_grow;
using group_farm_progress_test::check;
using group_farm_progress_test::rejects;
namespace fs = std::filesystem;
inline void legacy_missing_seal(fs::path const& parent)
{
    using namespace grow_controller_test;
    auto const root = parent / "legacy-known-missing-seal"; fs::create_directory(root);
    auto state = group_farm_progress_test::fixture(root); state.revision = 3; state.parent_digest = std::string(64, 'f');
    state.update_reserved_bytes = 64 * 1024;
    for (auto& directory : state.directories) directory.layout.families = 1;
    state.directories[1].sealed_files = 1; state.directories[1].policy_files = 0;
    auto const body = farming_state_text(state); auto const digest = stream_state::digest(body);
    auto const left = file_io::native_path(state.directories[0].directory);
    auto const right = file_io::native_path(state.directories[1].directory), offline = root / "offline";
    for (auto const& path : {left, right})
    {
        FarmingMetadataDirectory metadata(path, false); metadata.create();
        stream_state::atomic_save(metadata.path / "grow-farm.plan", body + digest + '\n', false);
    }
    auto const old_missing = stream_state::stamp(farming_metadata_path(right, "grow-farm.plan", false));
    fs::rename(right, offline);
    {
        FarmingUpdateMailbox mailbox(state, false, true, 64 * mib);
        check(mailbox.locked(), "legacy missing-seal fixture could not retain retirement intent");
        (void)mailbox.submit_control("retire", {state.directories[1].directory_identity});
    }
    FarmingJob job; job.saved = state; job.digest = digest; job.published = true; job.legacy_metadata = false;
    job.directories = {left, right};
    auto const manifest = farming_metadata_path(left, "grow-farm.plan", false);
    job.copies.push_back({manifest, stream_state::stamp(manifest), digest, false});
    FarmingRun run; run.state = state; run.metadata = 2 * mib; run.memory_limit = 64 * mib;
    run.lanes.resize(2); run.available = {1, 0}; run.all_replicas = false; run.bound_directories = job.directories;
    run.locks.resize(2); run.locks[0] = std::make_unique<DirectoryLock>(left);
    for (auto const& path : job.directories) run.metadata_directories.push_back(std::make_shared<FarmingMetadataDirectory>(path, false));
    file_io::Space space{}; check(!file_io::space_path(left, &space), "cannot inspect legacy retirement alignment");
    Arguments args; args.max_ram = run.memory_limit;
    settle_farming_membership(args, job, run, std::max(uint64_t(space.f_bsize), uint64_t(space.f_frsize)));
    check(run.membership_changed && run.state.version == 9 && !farming_member(run.state.directories[1]) &&
        run.state.directories[1].policy_files == state.directories[1].sealed_files &&
        run.state.directories[1].grant_extents == state.directories[1].extents &&
        run.state.directories[1].capacity_bytes == state.directories[1].capacity_bytes &&
        stream_state::stamp(farming_metadata_path(offline, "grow-farm.plan", false)) == old_missing &&
        read_farming_state(manifest, run.memory_limit).body == body,
        "partial legacy retirement lost a known missing seal, altered old metadata or released its original grant");
}
inline void stale_detached_preparation(fs::path const& parent, bool reclaimed = false)
{
    using namespace grow_controller_test;
    auto const root = parent / (reclaimed ? "reclaimed-metadata-growth" : "detached-preparation"); fs::create_directory(root);
    auto const left = root / "left", right = root / "right"; fs::create_directory(left); fs::create_directory(right);
    Arguments args; args.command = "fill"; args.directory_farm = true; args.target = left; args.destinations = {right};
    args.max_ram = 512 * mib; args.headroom = 0; args.capacity = 128 * mib;
    args.k = 18; args.strength = 2; args.group_members = 2; args.max_groups = 2; args.max_total_members = 4;
    args.farming_max_eligible_groups = 8; args.backend = "cpu"; args.workers = args.batch = 1;
    args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, 'b');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    Keys keys(args); auto initial_job = farming_arguments(args); auto initial = farming_run(args, &keys, initial_job);
    auto state = initial->state; initial.reset();
    auto advance = [&]
    {
        auto const digest = stream_state::digest(farming_state_text(state)); auto const revision = state.revision;
        ++state.revision; state.policy_revision = state.revision; state.parent_digest = digest;
        farming_record_parent(state, revision, digest);
    };
    auto save = [&](fs::path const& directory, char const* name)
    {
        auto body = farming_state_text(state); body += stream_state::digest(body) + '\n';
        stream_state::atomic_save(farming_metadata_path(directory, name, false), body, true);
    };
    advance(); detach_farming_member(state, 1);
    advance();
    if (reclaimed) acknowledge_farming_detachment(state, 1);
    advance();
    if (reclaimed)
    {
        auto& directory = state.directories[1]; directory.extents.clear(); directory.grant_extents.clear();
        directory.capacity_bytes = state.manifest_reserved_bytes; directory.storage.capacity_bytes = 0;
        directory.detached_reclaimed = true;
    }
    save(left, "grow-farm.plan"); save(right, "grow-farm.plan");
    auto const retained_plan = stream_state::stamp(farming_metadata_path(right, "grow-farm.plan", false));
    auto const retained_progress = stream_state::stamp(farming_metadata_path(right, "grow-farm.progress", false));
    auto const owner_path = farming_metadata_path(right, "grow-farm.owner", false);
    auto const retained_owner = fs::exists(owner_path) ? std::optional{stream_state::stamp(owner_path)} : std::nullopt;
    auto const retained_capacity = state.directories[1].capacity_bytes;
    advance(); auto const obsolete = state;
    save(left, "grow-farm.plan");
    if (!reclaimed) save(right, "grow-farm.plan.next");
    advance();
    if (reclaimed) state.manifest_reserved_bytes += 65536;
    save(left, "grow-farm.plan");
    auto const source = read_farming_state(farming_metadata_path(left, "grow-farm.plan", false), args.max_ram);
    args.target = left; args.destinations.clear(); args.resume = args.resume_set = true;
    auto job = farming_arguments(args);
    check(job.saved && !job.pending && job.saved->revision == state.revision,
        "an authenticated older detached preparation changed the current policy head");
    auto resumed = farming_run(args, &keys, job);
    if (reclaimed)
    {
        check(resumed->state.revision == state.revision && resumed->state.directories[1].capacity_bytes == retained_capacity &&
            stream_state::stamp(farming_metadata_path(right, "grow-farm.plan", false)) == retained_plan &&
            stream_state::stamp(farming_metadata_path(right, "grow-farm.progress", false)) == retained_progress &&
            (retained_owner ? fs::exists(owner_path) && stream_state::stamp(owner_path) == *retained_owner : !fs::exists(owner_path)),
            "later global metadata growth rewrote or enlarged an already reclaimed retirement tombstone");
        return;
    }
    check(resumed->state.revision == state.revision &&
        read_farming_state(farming_metadata_path(right, "grow-farm.plan", false), args.max_ram).digest == source.digest &&
        !fs::exists(farming_metadata_path(right, "grow-farm.plan.next", false)),
        "an obsolete detached preparation prevented publication of the current policy");
    resumed.reset();
    auto foreign = obsolete; foreign.directories[1].detached_source_digest = std::string(64, 'f');
    auto foreign_body = farming_state_text(foreign); foreign_body += stream_state::digest(foreign_body) + '\n';
    auto const pending = farming_metadata_path(right, "grow-farm.plan.next", false);
    stream_state::atomic_save(pending, foreign_body, false);
    auto const foreign_stamp = stream_state::stamp(pending);
    rejects([&] { (void)farming_arguments(args); }, "an unrelated detached preparation gained authority or was silently cleaned");
    check(stream_state::stamp(pending) == foreign_stamp, "refusing an unrelated detached preparation deleted its evidence");
}
inline void predetach_preparation(fs::path const& parent, bool invalid_edge)
{
    using namespace grow_controller_test;
    auto const root = parent / (invalid_edge ? "invalid-predetach-lineage" : "predetach-lineage"); fs::create_directory(root);
    auto const left = root / "left", right = root / "right"; fs::create_directory(left); fs::create_directory(right);
    Arguments args; args.command = "fill"; args.directory_farm = true; args.target = left; args.destinations = {right};
    args.max_ram = 512 * mib; args.headroom = 0; args.capacity = 128 * mib;
    args.k = 18; args.strength = 2; args.group_members = 2; args.max_groups = 2; args.max_total_members = 4;
    args.farming_max_eligible_groups = 8; args.backend = "cpu"; args.workers = args.batch = 1;
    args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, 'b');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    Keys keys(args); auto initial_job = farming_arguments(args); auto initial = farming_run(args, &keys, initial_job);
    auto state = initial->state; initial.reset();
    auto advance = [&]
    {
        auto const digest = stream_state::digest(farming_state_text(state)); auto const revision = state.revision;
        ++state.revision; state.policy_revision = state.revision; state.parent_digest = digest;
        farming_record_parent(state, revision, digest);
    };
    auto save = [&](fs::path const& directory, char const* name)
    {
        auto body = farming_state_text(state); body += stream_state::digest(body) + '\n';
        stream_state::atomic_save(farming_metadata_path(directory, name, false), body, true);
    };
    advance();
    if (invalid_edge)
    {
        auto& directory = state.directories[1]; directory.sealed_files = directory.policy_files = 1;
        directory.extents[0] = 1;
    }
    save(left, "grow-farm.plan"); save(right, "grow-farm.plan");
    advance();
    if (invalid_edge) state.directories[1].extents[0] = 2;
    auto const anchored = stream_state::digest(farming_state_text(state));
    save(left, "grow-farm.plan"); save(right, "grow-farm.plan.next");
    advance(); detach_farming_member(state, 1); save(left, "grow-farm.plan");
    args.target = left; args.destinations.clear(); args.resume = args.resume_set = true;
    if (invalid_edge)
    {
        auto const primary = farming_metadata_path(right, "grow-farm.plan", false);
        auto const pending = farming_metadata_path(right, "grow-farm.plan.next", false);
        auto const primary_stamp = stream_state::stamp(primary), pending_stamp = stream_state::stamp(pending);
        rejects([&] { (void)farming_arguments(args); }, "a staged source anchor bypassed an invalid protected-prefix transition");
        check(stream_state::stamp(primary) == primary_stamp && stream_state::stamp(pending) == pending_stamp,
            "invalid intermediate lineage rejection changed its recovery evidence");
        return;
    }
    auto job = farming_arguments(args);
    check(job.saved && !job.pending && job.saved->directories[1].detached_source_digest == anchored,
        "an exact staged detachment source did not authenticate its older published parent");
    auto resumed = farming_run(args, &keys, job);
    check(resumed->state.revision == state.revision &&
        read_farming_state(farming_metadata_path(right, "grow-farm.plan", false), args.max_ram).digest == job.digest &&
        !fs::exists(farming_metadata_path(right, "grow-farm.plan.next", false)),
        "returning pre-detachment primary and staged source did not reconcile to the current policy");
}
inline void connected_grant_exhaustion(fs::path const& parent, bool model_limit)
{
    using namespace grow_controller_test;
    auto const root = parent / (model_limit ? "connected-model-limit" : "connected-grant-renewal"); fs::create_directory(root);
    Arguments args; args.command = "fill"; args.directory_farm = true; args.target = root; args.plan_only = true;
    args.max_ram = 512 * mib; args.headroom = 0; args.capacity = 4 * mib;
    args.k = 18; args.strength = 2; args.group_members = 1; args.backend = "cpu"; args.workers = args.batch = 1;
    args.window = args.checkpoint_bytes = 64 * 1024; args.construction = "incremental"; args.quiet = true;
    args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, 'b');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    if (model_limit) args.meta = 0; // One eligible family cannot admit a second fixed-meta file.
    Keys keys(args); auto initial_job = farming_arguments(args); auto planned = farming_run(args, &keys, initial_job);
    auto const original = planned->state; auto state = original; planned.reset();
    // A smaller, jointly admitted forecast with the same ample byte quota
    // reaches the finite delegation boundary without plotting a whole drive.
    ++state.revision; state.policy_revision = state.revision;
    state.parent_digest = stream_state::digest(farming_state_text(original));
    farming_record_parent(state, original.revision, state.parent_digest);
    auto& directory = state.directories[0]; directory.extents = {1};
    directory.grant_extents = model_limit ? std::vector<uint16_t>{1} : std::vector<uint16_t>{1, 1};
    directory.layout.files = directory.layout.total_members = directory.layout.families = 1;
    check(valid_farming_policy_revision(original, state), "connected grant fixture changed a protected identity or allocation");
    static_cast<FarmingAssessment&>(state.estimate) = farming_assess_state(state, {directory.grant_extents}, args.max_ram);
    check(state.estimate.feasible, "connected grant fixture exceeded its joint harvester admission");
    auto const body = farming_state_text(state);
    check(farming_state_text(parse_farming_state(body, args.max_ram)) == body, "connected grant fixture is not canonical");
    {
        DirectoryLock lock(root); FarmingMetadataDirectory metadata(root, false); metadata.create();
        stream_state::atomic_save(metadata.path / "grow-farm.plan", body + stream_state::digest(body) + '\n', false);
    }
    args.plan_only = false; args.resume = args.resume_set = true;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    args.cancelled = [deadline] { return std::chrono::steady_clock::now() >= deadline; };
    check(run_farming(args) == 0, "connected grant renewal did not finish at a durable boundary");
    auto const saved = read_farming_state(farming_metadata_path(root, "grow-farm.plan", false), args.max_ram);
    auto completed = parse_farming_state(saved.body, args.max_ram);
    FarmingMetadataDirectory metadata(root, false); auto const progress = load_farming_progress(metadata, args.max_ram);
    check(bool(progress), "connected grant renewal lost its local progress");
    (void)overlay_farming_progress(completed, 0, progress->progress);
    auto const& final = completed.directories[0];
    check(!progress->progress.summary.pending_files && progress->progress.summary.completed_files == final.sealed_files,
        "connected grant renewal left an unfinished output");
    check(std::all_of(final.extents.begin(), final.extents.end(), [](auto members) { return members == 1; }),
        "connected grant renewal ignored the explicit group-member cap");
    if (model_limit)
    {
        check(final.sealed_files == 1 && final.fill_stop == "farming-limit" && saved.body == body,
            "refused connected grant renewal published an infeasible policy or lost its completed output");
        auto const output = group_path(root, 0); auto const stamp = stream_state::stamp(output);
        check(run_farming(args) == 0 && stream_state::stamp(output) == stamp &&
            read_farming_state(metadata.path / "grow-farm.plan", args.max_ram).body == body,
            "a stopped model-limit resume changed its completed payload or retried the refused policy");
    }
    else
        check(final.sealed_files > directory.grant_extents.size() && final.fill_stop == "capacity",
            "connected grant exhaustion stopped before its unchanged byte quota was filled");
}
inline void prepared_expansion(fs::path const& parent)
{
    using namespace grow_controller_test;
    auto const root = parent / "prepared-expansion"; fs::create_directory(root);
    auto const left = root / "left", right = root / "right", added = root / "added";
    for (auto const& directory : {left, right, added}) fs::create_directory(directory);
    Arguments args; args.command = "fill"; args.directory_farm = true; args.target = left; args.destinations = {right};
    args.max_ram = 512 * mib; args.headroom = 0; args.capacity = 16 * mib;
    args.k = 18; args.strength = 2; args.group_members = args.max_groups = args.max_total_members = 1;
    args.farming_max_eligible_groups = 8;
    args.backend = "cpu"; args.workers = args.batch = 1; args.window = args.checkpoint_bytes = 64 * 1024;
    args.construction = "incremental"; args.quiet = true;
    args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, 'b');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    check(run_farming(args) == 0, "prepared expansion fixture did not complete its original members");
    auto const left_output = stream_state::stamp(group_path(left, 0)), right_output = stream_state::stamp(group_path(right, 0));
    auto review_args = args; review_args.target = added; review_args.destinations = {right, left};
    review_args.plan_only = review_args.update = true;
    Keys keys(review_args); auto review_job = farming_arguments(review_args); auto reviewed = farming_run(review_args, &keys, review_job);
    auto const state = reviewed->state; auto const body = farming_state_text(state); auto const digest = stream_state::digest(body);
    check(reviewed->feasible() && state.directories.size() == 3 && state.policy_revision == state.revision,
        "prepared expansion fixture omitted its jointly admitted successor policy");
    reviewed.reset();
    // Reproduce complete preparation followed by only partial publication.
    // The added destination already has the new head; old journals remain.
    for (auto const& directory : {left, right, added})
    {
        DirectoryLock lock(directory); FarmingMetadataDirectory metadata(directory, false); metadata.create();
        stream_state::atomic_save(metadata.path / "grow-farm.plan.next", body + digest + '\n', false);
        if (directory != right) publish_farming_state(metadata.path / "grow-farm.plan.next", metadata.path / "grow-farm.plan", directory != added);
    }
    args.target = added; args.destinations.clear(); args.resume = args.resume_set = true;
    check(run_farming(args) == 0, "partially published expansion did not recover through its added destination");
    check(stream_state::stamp(group_path(left, 0)) == left_output && stream_state::stamp(group_path(right, 0)) == right_output,
        "prepared expansion recovery changed an original completed payload");
    for (size_t lane = 0; lane < state.directories.size(); ++lane)
    {
        auto const directory = file_io::native_path(state.directories[lane].directory);
        FarmingMetadataDirectory metadata(directory, false); auto const progress = load_farming_progress(metadata, args.max_ram);
        check(read_farming_state(metadata.path / "grow-farm.plan", args.max_ram).digest == digest && progress &&
            progress->progress.source_policy == farming_policy_digest(state) &&
            progress->progress.summary.completed_files == 1 && !progress->progress.summary.pending_files &&
            !fs::exists(metadata.path / "grow-farm.plan.next"),
            "prepared expansion recovery did not bind incorporated journals to the canonical published head");
    }
}
inline void offline_grant_return(fs::path const& parent, bool restart)
{
    using namespace grow_controller_test;
    auto const root = parent / (restart ? "offline-grant-restart" : "offline-grant-return"); fs::create_directory(root);
    auto const left = root / "left", right = root / "right", offline = root / "offline";
    fs::create_directory(left); fs::create_directory(right);
    Arguments args; args.command = "fill"; args.directory_farm = true; args.target = left; args.destinations = {right};
    args.plan_only = true; args.max_ram = 512 * mib; args.headroom = 0; args.capacity = 8 * mib;
    args.k = 18; args.strength = 2; args.group_members = 1; args.backend = "cpu"; args.workers = args.batch = 1;
    args.window = args.checkpoint_bytes = 64 * 1024; args.construction = "incremental"; args.quiet = true;
    args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, 'b');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    Keys keys(args); auto initial_job = farming_arguments(args); auto planned = farming_run(args, &keys, initial_job);
    auto const original = planned->state; auto state = original; planned.reset();
    ++state.revision; state.policy_revision = state.revision;
    state.parent_digest = stream_state::digest(farming_state_text(original));
    farming_record_parent(state, original.revision, state.parent_digest);
    for (auto& directory : state.directories)
    {
        directory.extents = {1}; directory.grant_extents = {1, 1};
        directory.layout.files = directory.layout.total_members = directory.layout.families = 1;
    }
    // The returning peer has no fresh work to trigger a different renewal.
    state.directories[1].control = "paused";
    check(valid_farming_policy_revision(original, state), "offline grant fixture changed a protected assignment");
    static_cast<FarmingAssessment&>(state.estimate) = farming_assess_state(state, {{1, 1}, {1, 1}}, args.max_ram);
    check(state.estimate.feasible, "offline grant fixture exceeded its joint harvester admission");
    auto const body = farming_state_text(state);
    for (auto const& directory : {left, right})
    {
        DirectoryLock lock(directory); FarmingMetadataDirectory metadata(directory, false); metadata.create();
        stream_state::atomic_save(metadata.path / "grow-farm.plan", body + stream_state::digest(body) + '\n', false);
    }
    fs::rename(right, offline); args.plan_only = false; args.destinations.clear();
    pid_t child = -1;
    auto start = [&]
    {
        child = ::fork(); check(child >= 0, "cannot fork the offline grant coordinator");
        if (!child)
        {
            auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            args.cancelled = [deadline] { return std::chrono::steady_clock::now() >= deadline; };
            try { std::_Exit(run_farming(args)); }
            catch (std::exception const& error) { std::cerr << error.what() << std::endl; std::_Exit(1); }
        }
    };
    auto stop = [&]
    {
        if (child > 0) { (void)::kill(child, SIGKILL); int status = 0; (void)::waitpid(child, &status, 0); child = -1; }
    };
    auto wait_for_stop = [&](uint64_t after)
    {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (std::chrono::steady_clock::now() < deadline)
        {
            int status = 0;
            if (::waitpid(child, &status, WNOHANG) == child) { child = -1; check(false, "automatic offline fill exited before its peer returned"); }
            try
            {
                FarmingMetadataDirectory metadata(left, false); auto progress = load_farming_progress(metadata, args.max_ram);
                if (progress && progress->progress.sequence > after && progress->progress.summary.completed_files == 2 &&
                    !progress->progress.summary.pending_files && progress->progress.fill_stop == "farming-limit")
                    return progress->progress.sequence;
            }
            catch (std::runtime_error const&) {} // Retry an observation replaced by the live local writer.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error("offline fill did not durably exhaust its finite grant");
    };
    try
    {
        start(); auto sequence = wait_for_stop(0);
        auto const first = stream_state::stamp(group_path(left, 0)), second = stream_state::stamp(group_path(left, 1));
        check(read_farming_state(farming_metadata_path(left, "grow-farm.plan", false), args.max_ram).body == body &&
            read_farming_state(farming_metadata_path(offline, "grow-farm.plan", false), args.max_ram).body == body,
            "offline grant exhaustion rewrote the replicated policy");
        if (restart)
        {
            stop(); auto explicit_resume = args; explicit_resume.resume = explicit_resume.resume_set = explicit_resume.allow_missing = true;
            check(run_farming(explicit_resume) == 4 && stream_state::stamp(group_path(left, 0)) == first &&
                stream_state::stamp(group_path(left, 1)) == second,
                "explicit missing-peer resume waited automatically or changed completed payloads");
            FarmingMetadataDirectory metadata(left, false); sequence = load_farming_progress(metadata, args.max_ram)->progress.sequence;
            start(); (void)wait_for_stop(sequence);
        }
        fs::rename(offline, right);
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        int status = 0; bool exited = false;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (::waitpid(child, &status, WNOHANG) == child) { child = -1; exited = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        check(exited && WIFEXITED(status) && WEXITSTATUS(status) == 0, "returned peer did not release the saved grant wait");
        FarmingMetadataDirectory metadata(left, false); auto const progress = load_farming_progress(metadata, args.max_ram);
        check(progress && progress->progress.summary.completed_files > 2 && progress->progress.fill_stop == "capacity" &&
            stream_state::stamp(group_path(left, 0)) == first && stream_state::stamp(group_path(left, 1)) == second &&
            read_farming_state(metadata.path / "grow-farm.plan", args.max_ram).digest ==
                read_farming_state(farming_metadata_path(right, "grow-farm.plan", false), args.max_ram).digest,
            "automatic fill forgot exhausted grants after a peer return or coordinator restart");
    }
    catch (...) { stop(); throw; }
}
inline void run(fs::path const& parent)
{
    auto const root = parent / "farming-membership"; fs::create_directory(root);
    auto state = group_farm_progress_test::fixture(root); state.version = 9;
    for (auto& directory : state.directories) directory.layout.families = 1;
    initialize_farming_grants(state);
    auto const before = state;
    auto legacy = before; legacy.version = 8; legacy.ancestors.clear(); legacy.owner_reserved_bytes = 0;
    legacy.shared_scratch_possible = legacy.conservative_storage_topology = false;
    for (auto& directory : legacy.directories) { directory.grant_extents.clear(); directory.grant_config.reset(); }
    auto const legacy_bytes = farming_state_text(legacy);
    auto audited = legacy; ++audited.revision; audited.parent_digest = stream_state::digest(legacy_bytes);
    audited.directories[0].extents[0] = 1; audited.directories[0].sealed_files = 1;
    initialize_farming_retirement_upgrade(legacy, audited);
    check(valid_farming_retirement_upgrade(legacy, audited) && audited.shared_scratch_possible && !audited.owner_reserved_bytes &&
        audited.manifest_reserved_bytes == legacy.manifest_reserved_bytes && audited.update_reserved_bytes == legacy.update_reserved_bytes &&
        audited.progress_reserved_bytes == legacy.progress_reserved_bytes && audited.directories[0].extents[0] == 1 &&
        audited.directories[0].grant_extents == legacy.directories[0].extents && farming_state_text(legacy) == legacy_bytes,
        "partial legacy retirement expanded, refunded or rewrote an original grant");
    for (unsigned change = 0; change < 5; ++change)
    {
        auto altered = legacy; ++altered.revision; altered.parent_digest = stream_state::digest(legacy_bytes);
        if (change == 0) ++altered.directories[0].capacity_bytes;
        if (change == 1) ++altered.config.group_members_override;
        if (change == 2) ++altered.directories[0].layout.schedule;
        if (change == 3) altered.directories[0].extents.push_back(1);
        if (change == 4) ++altered.manifest_reserved_bytes;
        rejects([&] { initialize_farming_retirement_upgrade(legacy, altered); },
            "partial legacy retirement changed storage, service, identity or metadata rights");
        check(altered.version == 8, "refused legacy retirement left a partially upgraded plan");
    }
    auto sealed_legacy = legacy; sealed_legacy.directories[0].sealed_files = 1;
    auto changed_seal = sealed_legacy; ++changed_seal.revision;
    changed_seal.parent_digest = stream_state::digest(farming_state_text(sealed_legacy));
    changed_seal.directories[0].extents[0] = 1;
    rejects([&] { initialize_farming_retirement_upgrade(sealed_legacy, changed_seal); },
        "partial legacy retirement changed an already sealed completed membership");
    auto const assessment = expand_farming_grants(state, 64 * 1024 * 1024);
    check(assessment.feasible && state.directories[0].grant_extents.size() > before.directories[0].extents.size() &&
        state.directories[1].grant_extents.size() > before.directories[1].extents.size() &&
        state.directories[0].extents == before.directories[0].extents &&
        state.directories[0].capacity_bytes == before.directories[0].capacity_bytes,
        "upfront farming grants omitted additional ordinals or changed byte quotas and forecasts");
    auto const expanded = state.directories[0].grant_extents;
    (void)expand_farming_grants(state, 64 * 1024 * 1024);
    check(state.directories[0].grant_extents == expanded, "reservation sizing repeatedly enlarged an independent grant");
    auto joint = before; joint.config.fixed_meta = true; joint.config.group_members_override = 1;
    joint.config.max_eligible_families_per_resource = 3;
    for (size_t lane = 0; lane < joint.directories.size(); ++lane)
    {
        auto& directory = joint.directories[lane]; directory.extents = {1}; directory.layout.files = directory.layout.total_members = 1;
        directory.layout.group_members = directory.layout.meta_slots = 1;
        directory.layout.resource_rank = uint16_t(lane); directory.layout.resource_lanes = 2;
        directory.resource = directory.storage.resource = 0; directory.layout.schedule = 0;
    }
    initialize_farming_grants(joint); auto const constrained = expand_farming_grants(joint, 64 * 1024 * 1024);
    check(constrained.feasible && joint.directories[0].grant_extents.size() == 2 && joint.directories[1].grant_extents.size() == 1,
        "independent lane envelopes combined incompatible individual farming assessments");
    auto partial = before; partial.directories[0].extents.back() = 1;
    initialize_farming_grants(partial); (void)expand_farming_grants(partial, 64 * 1024 * 1024);
    check(partial.directories[0].grant_extents[2] == 2 && partial.directories[0].extents[2] == 1,
        "independent farming grant omitted terminal member growth beyond its forecast");
    auto automatic = before; automatic.config.group_members_override = 0;
    for (auto& directory : automatic.directories)
    {
        directory.extents = {7}; directory.layout.files = 1; directory.layout.total_members = directory.layout.group_members = 7;
        directory.layout.fixed_members = false;
    }
    initialize_farming_grants(automatic); auto const automatic_assessment = expand_farming_grants(automatic, 64 * 1024 * 1024);
    check(automatic_assessment.feasible && std::all_of(automatic.directories.begin(), automatic.directories.end(), [](auto const& directory)
        { return directory.extents == std::vector<uint16_t>{7} && directory.grant_extents.size() == 2 && directory.grant_extents[0] >= 8; }),
        "automatic forecast size capped terminal growth or consumed a peer's next ordinal");
    auto const automatic_grants = automatic.directories;
    (void)expand_farming_grants(automatic, 64 * 1024 * 1024);
    check(automatic.directories == automatic_grants, "repeated automatic grant sizing changed an admitted envelope");
    auto explicit_limit = automatic; explicit_limit.config.group_members_override = 7;
    initialize_farming_grants(explicit_limit); (void)expand_farming_grants(explicit_limit, 64 * 1024 * 1024);
    check(std::all_of(explicit_limit.directories.begin(), explicit_limit.directories.end(), [](auto const& directory)
        { return directory.grant_extents[0] == 7 && std::all_of(directory.grant_extents.begin(), directory.grant_extents.end(),
            [](auto members) { return members <= 7; }); }), "automatic grant growth ignored an explicit group-member cap");
    auto total_limit = automatic; total_limit.max_total_members = 8;
    initialize_farming_grants(total_limit); (void)expand_farming_grants(total_limit, 64 * 1024 * 1024);
    check(std::all_of(total_limit.directories.begin(), total_limit.directories.end(), [](auto const& directory)
        { return directory.grant_extents == std::vector<uint16_t>{8}; }),
        "independent grants exceeded a total-member cap or displaced a terminal top-up with another ordinal");
    auto const body = farming_state_text(state);
    auto restored = parse_farming_state(body, 64 * 1024 * 1024);
    check(farming_state_text(restored) == body && restored.directories[0].grant_config == state.directories[0].grant_config,
        "independent farming grants changed during canonical reload");
    auto initial = capture_farming_progress(state, 0);
    check(initial.version == 2 && initial.granted_extents == state.directories[0].grant_extents &&
        initial.extents == state.directories[0].extents &&
        farming_progress_text(parse_farming_progress(farming_progress_text(initial), 64 * 1024 * 1024)) == farming_progress_text(initial),
        "independent progress lost its larger immutable allocation envelope");
    auto observed = state.directories[0]; observed.extents.push_back(1);
    auto advanced = advance_farming_progress(initial, observed, {});
    auto target = state; overlay_farming_progress(target, 0, advanced);
    check(target.directories[0].extents == observed.extents && target.directories[1].extents == state.directories[1].extents &&
        target.revision == state.revision && farming_policy_digest(target) == farming_policy_digest(state),
        "local grant spending changed a peer or the source policy");
    rejects([&] { auto bad = advanced; bad.extents.back() = uint16_t(bad.granted_extents.back() + 1); farming_progress_text(bad); },
        "local progress exceeded its preapproved service envelope");
    rejects([&] { auto bad = advanced; bad.extents.push_back(1); farming_progress_text(bad); },
        "local progress allocated an unapproved ordinal");
    auto missing = state; ++missing.revision; missing.parent_digest = stream_state::digest(body);
    farming_record_parent(missing, state.revision, missing.parent_digest); missing.policy_revision = missing.revision;
    detach_farming_member(missing, 1);
    check(!farming_member(missing.directories[1]) && farming_member_count(missing) == 1 &&
        missing.directories[1].grant_extents == state.directories[1].grant_extents &&
        missing.directories[1].capacity_bytes == state.directories[1].capacity_bytes &&
        valid_farming_policy_revision(state, missing) && valid_farming_independent_policy_revision(state, missing),
        "detachment released or changed an unavailable writer's outstanding grants");
    auto topology = state; topology.config.max_eligible_families_per_resource = 1;
    std::vector<std::vector<uint16_t>> service;
    for (auto const& directory : topology.directories) service.push_back(directory.grant_extents);
    auto const baseline = farming_assess_state(topology, service, 64 * 1024 * 1024);
    auto unchanged_detach = missing; unchanged_detach.config.max_eligible_families_per_resource = 1;
    auto const unchanged_service = farming_assess_state(unchanged_detach, service, 64 * 1024 * 1024);
    check(baseline.feasible && unchanged_service.feasible && baseline.response_bound_seconds == unchanged_service.response_bound_seconds,
        "plain detachment changed an already certified physical topology");
    unchanged_detach.conservative_storage_topology = true;
    auto const folded = farming_assess_state(unchanged_detach, service, 64 * 1024 * 1024);
    check(!folded.feasible && folded.max_eligible_families_per_resource == 2 &&
        unchanged_detach.directories[1].resource == missing.directories[1].resource &&
        same_farming_assignment(missing, missing.directories[1], unchanged_detach, unchanged_detach.directories[1], 0),
        "unproved retired storage topology split physical service or changed an identity assignment");
    auto old_writer = capture_farming_progress(state, 1);
    auto old_directory = state.directories[1]; old_directory.extents.push_back(1);
    auto old_progress = advance_farming_progress(old_writer, old_directory, {});
    auto returned = missing;
    check(overlay_farming_progress(returned, 1, old_progress) == FarmingProgressOverlay::applied &&
        returned.directories[1].extents == old_directory.extents && !farming_member(returned.directories[1]),
        "a returning detached writer's local allocation was discarded or reactivated");
    auto incorporated = state; incorporated.policy_revision = ++incorporated.revision;
    incorporated.directories[1].extents = old_directory.extents;
    incorporated.directories[1].policy_files = old_directory.extents.size();
    check(valid_farming_policy_revision(state, incorporated),
        "a later reviewed policy could not preserve locally delegated output ordinals");
    auto premature = missing; premature.policy_revision = ++premature.revision;
    premature.directories[1].detached_acknowledged = true; premature.directories[1].grant_extents.clear();
    check(!valid_farming_policy_revision(missing, premature), "retirement reclaimed a grant in its acknowledgement revision");
    auto acknowledged = missing; auto const detached_digest = stream_state::digest(farming_state_text(missing));
    acknowledged.policy_revision = ++acknowledged.revision; acknowledged.parent_digest = detached_digest;
    farming_record_parent(acknowledged, missing.revision, detached_digest); acknowledge_farming_detachment(acknowledged, 1);
    check(valid_farming_policy_revision(missing, acknowledged) && farming_membership_obligations(missing).unfenced == 1 &&
        !farming_membership_obligations(acknowledged).unfenced &&
        acknowledged.directories[1].grant_extents == missing.directories[1].grant_extents,
        "retirement acknowledgement discarded an outstanding grant before durable fencing");
    premature = missing; premature.policy_revision = ++premature.revision;
    premature.directories[1].detached_acknowledged = premature.directories[1].detached_reclaimed = true;
    check(!valid_farming_policy_revision(missing, premature), "retirement skipped the durably acknowledged predecessor before reclamation");
    auto malformed = missing; malformed.directories[1].detached_reclaimed = true;
    rejects([&] { (void)parse_farming_state(farming_state_text(malformed), 64 * 1024 * 1024); },
        "saved reclamation omitted its durable retirement acknowledgement");
    auto reclaimed = acknowledged; reclaimed.policy_revision = ++reclaimed.revision;
    reclaimed.parent_digest = stream_state::digest(farming_state_text(acknowledged));
    farming_record_parent(reclaimed, acknowledged.revision, reclaimed.parent_digest);
    reclaimed.directories[1].grant_extents.clear(); reclaimed.directories[1].extents.clear();
    reclaimed.directories[1].capacity_bytes = reclaimed.manifest_reserved_bytes;
    reclaimed.directories[1].storage.capacity_bytes = 0;
    reclaimed.directories[1].storage.max_groups = reclaimed.directories[1].storage.max_total_members = 0;
    reclaimed.directories[1].detached_reclaimed = true;
    check(valid_farming_policy_revision(acknowledged, reclaimed) &&
        valid_farming_independent_policy_revision(acknowledged, reclaimed) &&
        farming_state_text(parse_farming_state(farming_state_text(reclaimed), 64 * 1024 * 1024)) == farming_state_text(reclaimed),
        "acknowledged retirement could not reclaim an unused allocation while preserving its native slot");
    auto frozen = reclaimed; frozen.policy_revision = ++frozen.revision;
    ++frozen.manifest_reserved_bytes;
    check(valid_farming_policy_revision(reclaimed, frozen) && farming_reclaimed_tombstone(frozen.directories[1]),
        "policy-wide metadata growth changed an immutable reclaimed directory");
    for (unsigned change = 0; change < 6; ++change)
    {
        auto edited = frozen; auto& directory = edited.directories[1];
        if (change == 0) directory.detached_reclaimed = false;
        if (change == 1) directory.capacity_bytes = frozen.manifest_reserved_bytes;
        if (change == 2) directory.fill_stop = "capacity";
        if (change == 3) directory.layout.schedule = 99;
        if (change == 4) directory.policy_files = 1;
        if (change == 5) directory.control = "active";
        check(!valid_farming_policy_revision(reclaimed, edited) && !valid_farming_independent_policy_revision(reclaimed, edited),
            "a later policy edited or reactivated a frozen reclaimed tombstone");
    }
    auto next = missing;
    for (unsigned i = 0; i < 1100; ++i)
    {
        auto const digest = stream_state::digest(farming_state_text(next)); auto const revision = next.revision;
        ++next.revision; next.parent_digest = digest; next.policy_revision = next.revision;
        farming_record_parent(next, revision, digest);
    }
    check(next.ancestors.size() == 1 && next.directories[1].detached_source_revision == state.revision &&
        next.directories[1].detached_source_digest == stream_state::digest(body) &&
        farming_state_text(parse_farming_state(farming_state_text(next), 64 * 1024 * 1024)) == farming_state_text(next),
        "farming ancestry lost a missing member source or imposed a lifetime revision limit");
    check(farming_detached_replica_matches(missing, next, 1) && farming_detached_replica_matches(missing, reclaimed, 1),
        "a returning detached tombstone lost its retained source certificate");
    auto foreign = missing; ++foreign.directories[1].grant_extents.back();
    check(!farming_detached_replica_matches(foreign, next, 1), "a returning detached tombstone changed an unfenced allocation certificate");
    foreign = missing; foreign.directories[1].detached_source_digest = std::string(64, 'f');
    check(!farming_detached_replica_matches(foreign, reclaimed, 1), "acknowledged retirement accepted an unrelated source certificate");
    auto enlarged = next; ++enlarged.directories[0].grant_extents.back();
    check(!valid_farming_independent_policy_revision(next, enlarged),
        "a detached partition gained authority over unreserved shared service");
    enlarged = next; ++enlarged.directories[0].capacity_bytes;
    check(!valid_farming_independent_policy_revision(next, enlarged),
        "a detached partition gained authority over unreserved shared storage");
    enlarged = next; enlarged.directories[1].grant_extents.clear();
    check(!valid_farming_policy_revision(next, enlarged), "policy retirement refunded an unfenced missing writer");
    enlarged = next; enlarged.directories[1].membership = "member";
    check(!valid_farming_policy_revision(next, enlarged), "a returning old member silently rejoined detached membership");

    // An available drive persists new ordinals without opening the absent
    // peer, changing a global parent, or consuming a different byte grant.
    using grow_controller_test::FarmingRun;
    FarmingRun farm; farm.state = state; farm.memory_limit = 64 * 1024 * 1024; farm.metadata = 2 * 1024 * 1024;
    farm.lanes.resize(2); farm.available = {1, 0}; farm.all_replicas = false; farm.progress.resize(2);
    auto const path = file_io::native_path(state.directories[0].directory);
    farm.locks.resize(2); farm.locks[0] = std::make_unique<grow_controller_test::DirectoryLock>(path);
    farm.metadata_directories.resize(2); farm.metadata_directories[0] = std::make_shared<FarmingMetadataDirectory>(path, false);
    farm.progress[0] = save_farming_progress(*farm.metadata_directories[0], initial, farm.memory_limit);
    farm.lanes[0].completed_charge = farm.lanes[0].manifest_charge = farm.manifest_charge = farm.progress[0]->allocated_bytes;
    auto const ordinal = state.directories[0].extents.size();
    check(farm.reserve_extent(0, ordinal, 1, farm.metadata) && farm.state.revision == state.revision &&
        farm.state.directories[1].extents == state.directories[1].extents &&
        !fs::exists(file_io::native_path(state.directories[1].directory) / ".xchplot2"),
        "independent ordinal admission waited for or wrote to an absent peer");
    auto saved = load_farming_progress(*farm.metadata_directories[0], farm.memory_limit);
    auto resume = state; overlay_farming_progress(resume, 0, saved->progress);
    check(resume.directories[0].extents.size() == farm.state.directories[0].extents.size() &&
        resume.directories[0].extents.back() == 1, "independent ordinal allocation was not durable");
    check(!farm.reserve_extent(0, ordinal + 1, 1, farm.metadata), "independent ordinal admission exceeded its durable grant");
    farm.scratch_grants_frozen = true;
    check(!farm.recovery_only(), "ordinary no-scratch farming lost independent allocation");
    farm.state.shared_scratch_possible = true;
    auto const frozen_extents = farm.state.directories[0].extents; auto const frozen_stamp = farm.progress[0]->stamp;
    check(farm.recovery_only() && !farm.reserve_extent(0, ordinal, 2, farm.metadata) &&
        farm.state.directories[0].extents == frozen_extents && farm.progress[0]->stamp == frozen_stamp,
        "an unavailable scratch writer's outstanding quota permitted new local spending");
    auto disabled = farm.state; disabled.policy_revision = ++disabled.revision; disabled.shared_scratch_possible = false;
    check(!valid_farming_policy_revision(farm.state, disabled), "a later policy forgot previously delegated shared scratch grants");
    stale_detached_preparation(root); stale_detached_preparation(root, true);
    predetach_preparation(root, false); predetach_preparation(root, true);
    legacy_missing_seal(root);
    connected_grant_exhaustion(root, false); connected_grant_exhaustion(root, true);
    prepared_expansion(root);
    offline_grant_return(root, false); offline_grant_return(root, true);
    std::cout << "farming membership, conserved missing-writer obligations and independent allocation grants passed\n";
}
} // namespace group_farm_membership_test
