// Included after the controller fixture and its shared farming-state fixture.
#pragma once
#include "host/GroupFarmControl.hpp"
#include "host/GroupFarmOwner.hpp"
#include <csignal>

namespace group_farm_control_test
{
using namespace group_grow;
using grow_controller_test::DirectoryLock;
using group_farm_progress_test::check;
using group_farm_progress_test::rejects;
namespace fs = std::filesystem;
struct ChildGuard
{
    pid_t pid = -1;
    ~ChildGuard()
    {
        if (pid <= 0) return;
        (void)::kill(pid, SIGKILL); int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    }
};
inline std::string bytes(fs::path const& path)
{ std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}; }
inline void save(FarmingState const& state)
{
    auto body = farming_state_text(state); body += stream_state::digest(body) + '\n';
    for (auto const& d : state.directories)
    {
        FarmingMetadataDirectory metadata(file_io::native_path(d.directory), false); metadata.create();
        stream_state::atomic_save(metadata.path / "grow-farm.plan", body);
    }
}
inline void idle_control_wakeup(fs::path const& parent)
{
    using namespace grow_controller_test;
    auto const root = parent / "idle-farming-controls", left = root / "left", right = root / "right", offline = root / "offline";
    fs::create_directory(root); fs::create_directory(left); fs::create_directory(right);
    Arguments args; args.command = "fill"; args.directory_farm = true; args.target = left; args.destinations = {right};
    args.plan_only = true; args.max_ram = 512 * mib; args.capacity = 8 * mib; args.headroom = 0;
    args.k = 18; args.strength = 2; args.group_members = 1; args.max_groups = args.max_total_members = 2;
    args.backend = "cpu"; args.workers = args.batch = 1; args.window = args.checkpoint_bytes = 64 * 1024;
    args.construction = "incremental"; args.quiet = true; args.seed = std::string(64, 'a'); args.pool_ph = std::string(64, 'b');
    args.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    Keys keys(args); auto job = farming_arguments(args); auto planned = farming_run(args, &keys, job);
    check(planned->feasible(), "idle control fixture lacks a feasible saved policy");
    save(planned->state); planned.reset(); fs::rename(right, offline); fs::create_directory(right);
    args.plan_only = false; args.destinations.clear();
    auto const child = ::fork(); check(child >= 0, "cannot fork idle control coordinator");
    if (!child)
    {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        args.cancelled = [deadline] { return std::chrono::steady_clock::now() >= deadline; };
        try { std::_Exit(run_farming(args)); }
        catch (std::exception const& error) { std::cerr << error.what() << std::endl; std::_Exit(1); }
    }
    ChildGuard cleanup{child};
    auto wait_for = [&](auto&& ready, char const* failure)
    {
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline)
        {
            int status = 0;
            if (::waitpid(child, &status, WNOHANG) == child) { cleanup.pid = -1; check(false, failure); }
            try { if (ready()) return; }
            catch (std::runtime_error const&) {} // Retry an atomic progress or owner replacement.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        throw std::runtime_error(failure);
    };
    FarmingMetadataDirectory metadata(left, false);
    auto stopped = [&]
    {
        auto const state = load_farming_state(metadata.path / "grow-farm.plan", args.max_ram);
        auto const progress = load_farming_progress(metadata, args.max_ram);
        return progress && progress->progress.summary.completed_members == 2 && !progress->progress.summary.pending_files &&
            observe_farming_owner(metadata, state, 0, args.max_ram).status == "stopped";
    };
    wait_for(stopped, "automatic coordinator did not enter its missing-peer idle wait");
    auto const first = stream_state::stamp(group_path(left, 0)), second = stream_state::stamp(group_path(left, 1));
    auto const original = bytes(metadata.path / "grow-farm.plan");
    for (auto const* action : {"pause", "resume"})
    {
        auto control = args; control.command = control.farm_action = action; control.resume = control.policy_only = true;
        check(run_farming(control) == 0, "policy-only control failed while a coordinator was idle");
        auto const state = load_farming_state(metadata.path / "grow-farm.plan", args.max_ram);
        FarmingUpdateMailbox mailbox(state, false, false, args.max_ram);
        auto const digest = farming_control_view(state, mailbox.records(), mailbox.control_state()).digest;
        wait_for([&]
        {
            auto const owner = observe_farming_owner(metadata, state, 0, args.max_ram);
            return owner.status == "stopped" && owner.control_digest == digest;
        }, "idle coordinator did not acknowledge compact control intent");
        check(bytes(metadata.path / "grow-farm.plan") == original, "idle compact control changed replicated policy bytes");
    }
    auto const stable_progress = stream_state::stamp(metadata.path / "grow-farm.progress");
    auto const stable_owner = stream_state::stamp(metadata.path / "grow-farm.owner");
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    check(stream_state::stamp(metadata.path / "grow-farm.progress") == stable_progress &&
        stream_state::stamp(metadata.path / "grow-farm.owner") == stable_owner, "unchanged idle metadata caused repeated controller recovery");
    // A policy publisher may finish while the idle observer is descheduled.
    // Restore the peer only for that fenced publication, then remove it
    // before resuming the observer so native availability remains unchanged.
    check(!::kill(child, SIGSTOP), "cannot stop idle policy observer");
    int stopped_status = 0;
    check(::waitpid(child, &stopped_status, WUNTRACED) == child && WIFSTOPPED(stopped_status), "idle policy observer did not stop");
    fs::remove(right); fs::rename(offline, right);
    auto policy = args; policy.update = true; policy.headroom = 4096; policy.headroom_set = true;
    check(run_farming(policy) == 0, "connected policy publication failed in idle observer fixture");
    auto const changed = load_farming_state(metadata.path / "grow-farm.plan", args.max_ram);
    check(changed.headroom == policy.headroom && bytes(metadata.path / "grow-farm.plan") != original,
        "idle policy fixture did not publish a newer saved policy");
    auto const sequence = load_farming_progress(metadata, args.max_ram)->progress.sequence;
    fs::rename(right, offline); fs::create_directory(right);
    auto const offline_plan = bytes(farming_metadata_path(offline, "grow-farm.plan", false));
    check(!::kill(child, SIGCONT), "cannot resume idle policy observer");
    wait_for([&]
    {
        auto const progress = load_farming_progress(metadata, args.max_ram);
        return progress && progress->progress.sequence > sequence && stopped();
    }, "idle coordinator did not reload a policy published without a lasting availability change");
    auto retirement = args; retirement.command = retirement.farm_action = "retire";
    retirement.resume = true; retirement.action_destinations = {right};
    check(run_farming(retirement) == 0, "idle missing-peer retirement failed");
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    int status = 0; bool exited = false;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (::waitpid(child, &status, WNOHANG) == child) { cleanup.pid = -1; exited = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    check(exited && WIFEXITED(status) && !WEXITSTATUS(status), "retired missing peer kept the original coordinator waiting for physical return");
    auto const retired = load_farming_state(metadata.path / "grow-farm.plan", args.max_ram);
    check(farming_member_count(retired) == 1 && !farming_member(retired.directories[1]) && fs::is_empty(right) &&
        bytes(farming_metadata_path(offline, "grow-farm.plan", false)) == offline_plan &&
        stream_state::stamp(group_path(left, 0)) == first && stream_state::stamp(group_path(left, 1)) == second,
        "idle recovery rewrote completed payloads, missing metadata or a replacement mountpoint");
}
inline void run(fs::path const& parent)
{
    auto const root = parent / "local-farming-controls"; fs::create_directory(root);
    auto state = group_farm_progress_test::fixture(root);
    for (auto& directory : state.directories) directory.layout.families = 1;
    uint64_t const memory = 64ull * 1024 * 1024;
    state.manifest_reserved_bytes = 2 * 1024 * 1024;
    state.update_reserved_bytes = farming_update_reservation(8192, 4096) + farming_control_reservation(state, 4096);
    state.farmer = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    grow_controller_test::Arguments key_args; key_args.seed = state.seed; key_args.farmer = state.farmer; key_args.pool_ph = state.pool;
    key_args.k = state.identity_k; key_args.strength = state.identity_strength;
    state.fingerprint = grow_controller_test::farming_fingerprint(grow_controller_test::Keys(key_args));
    save(state);
    auto const left = file_io::native_path(state.directories[0].directory), right = file_io::native_path(state.directories[1].directory);
    auto const a = state.directories[0].directory_identity, b = state.directories[1].directory_identity;
    {
        auto const manifest = farming_metadata_path(left, "grow-farm.plan", false); auto const original = bytes(manifest);
        auto staged = state; ++staged.revision; staged.policy_revision = staged.revision; staged.headroom = 1;
        staged.parent_digest = stream_state::digest(farming_state_text(state));
        check(valid_farming_policy_revision(state, staged), "control pending-revision fixture lacks valid policy lineage");
        auto const pending = stage_farming_state(manifest, staged);
        grow_controller_test::Arguments request; request.command = "pause"; request.target = left;
        request.directory_farm = request.resume = true; request.farm_action = "pause"; request.max_ram = memory;
        auto const job = grow_controller_test::farming_arguments(request);
        check(job.pending && job.saved->revision == staged.revision &&
            !grow_controller_test::queue_farming_update(request, job).empty(), "staged valid revision blocked local pause intent");
        check(bytes(manifest) == original && fs::exists(pending), "local pause published or discarded an unrelated pending policy revision");
        state = std::move(staged); save(state); check(!file_io::unlink_file(pending), "cannot complete staged control fixture publication");
        FarmingUpdateMailbox mailbox(state, false, true, memory); auto const first = mailbox.submit_control("resume");
        auto const current = farming_metadata_path(left, "grow-farm.update.controls", false); auto const stamp = stream_state::stamp(current);
        check(mailbox.submit_control("resume") == first && stream_state::stamp(current) == stamp,
            "repeating active resume changed its causal identity or interrupted the current owner unnecessarily");
    }
    auto const plan_before = bytes(farming_metadata_path(left, "grow-farm.plan", false));
    auto const policy = farming_policy_digest(state);
    FarmingControlState empty;
    { FarmingUpdateMailbox mailbox(state, false, false, memory); empty = mailbox.control_state(); }
    auto paused = advance_farming_controls(state, empty, a, "pause", {a}, policy);
    auto resumed = advance_farming_controls(state, empty, b, "resume", {a}, policy);
    auto concurrent = merge_farming_controls(state, paused, resumed);
    check(farming_control_state_text(concurrent) == farming_control_state_text(merge_farming_controls(state, resumed, paused)),
        "concurrent control merge depends on member observation order");
    auto view = farming_control_view(state, {}, concurrent);
    check(view.effective(0) == "paused" && view.conflicts == std::vector<std::string>{"control:" + a},
        "concurrent resume bypassed an unobserved pause");
    auto resolved = advance_farming_controls(state, concurrent, a, "resume", {a}, policy);
    check(farming_control_view(state, {}, merge_farming_controls(state, resolved, paused)).effective(0) == "active",
        "returning stale pause replayed after an explicit causal resolution");
    auto unrelated = advance_farming_controls(state, paused, a, "drain", {b}, policy);
    auto joined = merge_farming_controls(state, unrelated, resumed);
    check(farming_control_view(state, {}, joined).effective(0) == "paused" &&
        farming_control_view(state, {}, joined).effective(1) == "draining",
        "global causal context erased a retained control on an unrelated destination");
    {
        auto many = state; many.directories.clear(); auto const many_root = root / "many-actors"; fs::create_directory(many_root);
        for (unsigned actor = 0; actor < 6; ++actor)
        {
            auto directory = state.directories.front(); auto const path = many_root / ("actor-" + std::to_string(actor)); fs::create_directory(path);
            directory.directory = file_io::path_text(path); directory.directory_identity = farming_directory_identity(path);
            directory.resource = directory.identity_resource = directory.layout.schedule = actor; directory.identity_resources = 6;
            many.directories.push_back(std::move(directory));
        }
        many.update_reserved_bytes = farming_update_reservation(8192, 4096) + farming_control_reservation(many, 4096); save(many);
        FarmingControlState common;
        { FarmingUpdateMailbox mailbox(many, false, false, memory); common = mailbox.control_state(); }
        auto const selected = many.directories.front().directory_identity; auto const source = farming_policy_digest(many);
        std::vector<FarmingControlState> branches;
        for (size_t actor = 0; actor < many.directories.size(); ++actor)
        {
            branches.push_back(advance_farming_controls(many, common, many.directories[actor].directory_identity,
                actor % 2 ? "resume" : "pause", {selected}, source));
            auto const body = farming_control_state_text(branches.back());
            stream_state::atomic_save(farming_metadata_path(file_io::native_path(many.directories[actor].directory),
                "grow-farm.update.controls", false), body + stream_state::digest(body) + '\n');
        }
        {
            FarmingUpdateMailbox mailbox(many, false, true, memory); auto const reunited = mailbox.control_state();
            check(reunited.fields.at("control:" + selected).size() > farming_update_slots &&
                farming_control_view(many, {}, reunited).effective(0) == "paused", "reunited native actors were limited by policy queue slots");
            check(farming_control_state_text(reunited).size() <= farming_control_size_bound(many) &&
                farming_control_state_text(parse_farming_controls(many, farming_control_state_text(reunited), memory)) == farming_control_state_text(reunited),
                "many-actor control reconciliation exceeded its deterministic size or parser bound");
            (void)mailbox.submit_control("resume", {selected});
        }
        auto const old = farming_control_state_text(branches.front());
        stream_state::atomic_save(farming_metadata_path(file_io::native_path(many.directories.front().directory), "grow-farm.update.controls", false),
            old + stream_state::digest(old) + '\n');
        FarmingUpdateMailbox mailbox(many, false, false, memory); auto const reconciled = farming_control_view(many, {}, mailbox.control_state());
        check(reconciled.effective(0) == "active" && reconciled.conflicts.empty(), "large causal frontier could not be explicitly resolved without stale replay");
    }
    auto foreign = concurrent; foreign.context["foreign-native-member"] = 1;
    rejects([&] { (void)parse_farming_controls(state, farming_control_state_text(foreign), memory); },
        "control snapshot adopted a foreign actor");
    foreign = concurrent; foreign.job_id = std::string(64, 'f');
    rejects([&] { (void)merge_farming_controls(state, foreign, empty); }, "control snapshot adopted a foreign job");
    rejects([&] { (void)parse_farming_controls(state, farming_control_state_text(concurrent), 1); },
        "control snapshot ignored its parsing memory grant");
    { FarmingUpdateMailbox mailbox(state, false, true, memory); (void)mailbox.submit_control("pause"); }
    auto const offline = root / "offline-right"; fs::rename(right, offline);
    for (unsigned command = 0; command < 64; ++command)
    {
        FarmingUpdateMailbox mailbox(state, false, true, memory);
        check(mailbox.locked() && !mailbox.complete(), "local controls required an unrelated missing drive");
        auto const* action = command % 3 == 0 ? "pause" : command % 3 == 1 ? "resume" : "drain";
        (void)mailbox.submit_control(action);
        view = farming_control_view(state, mailbox.records(), mailbox.control_state());
        check(view.control == farming_update_control(action) && mailbox.records().empty(),
            "sequential local control retained unresolved policy requests");
        grow_controller_test::FarmingRun writer; writer.state = state; writer.lanes.resize(state.directories.size()); writer.controls = view;
        grow_controller_test::Arguments args;
        check(writer.maximum(args, 0) == (view.control == "active" ? 2 : 0) &&
            writer.accepts_future(0) == (view.control == "active"), "local control did not fence healthy-drive future work");
        writer.lanes[0].outputs.emplace_back();
        check(writer.maximum(args, 0) == (view.control == "paused" ? 0 : 2),
            "local drain stopped a started file or local pause admitted another member");
        FarmingMetadataDirectory metadata(left, false);
        check(farming_update_allocation(metadata.path) <= farming_control_reservation(state, 4096),
            "repeated offline-peer controls grew beyond a bounded current and replacement snapshot");
    }
    check(bytes(farming_metadata_path(left, "grow-farm.plan", false)) == plan_before,
        "local control changed immutable global policy bytes");
    fs::rename(offline, right);
    auto const snapshot_before = bytes(farming_metadata_path(left, "grow-farm.update.controls", false));
    { FarmingUpdateMailbox mailbox(state, false, false, memory); view = farming_control_view(state, mailbox.records(), mailbox.control_state()); }
    check(bytes(farming_metadata_path(left, "grow-farm.update.controls", false)) == snapshot_before,
        "read-only control observation rewrote or converged a stale peer");
    check(view.control == "paused", "returning stale replica replayed an earlier global control");
    for (auto const* point : {"farm-control-before-publication", "farm-control-published"})
    {
        auto const child = ::fork(); check(child >= 0, "cannot fork control replacement crash fixture");
        if (!child)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1);
            FarmingUpdateMailbox mailbox(state, false, true, memory); (void)mailbox.submit_control("drain"); std::_Exit(0);
        }
        int status = 0;
        check(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "control replacement did not reach its durable crash boundary");
        { FarmingUpdateMailbox read_only(state, false, false, memory); (void)read_only.control_state(); }
        { FarmingUpdateMailbox recovery(state, false, true, memory); (void)recovery.submit_control("resume"); }
        { FarmingUpdateMailbox read_only(state, false, false, memory); view = farming_control_view(state, read_only.records(), read_only.control_state()); }
        check(view.control == "active" && !fs::exists(farming_metadata_path(left, "grow-farm.update.controls.next", false)) &&
            !fs::exists(farming_metadata_path(right, "grow-farm.update.controls.next", false)),
            "control replacement recovery replayed unpublished intent or retained staging");
    }
    {
        FarmingUpdateMailbox held(state, false, true, memory);
        auto const child = ::fork(); check(child >= 0, "cannot fork competing control publisher");
        if (!child)
        {
            FarmingUpdateMailbox contender(state, false, true, memory); std::_Exit(contender.locked() ? 1 : 0);
        }
        int status = 0; check(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
            "a competing process bypassed the available-member control locks");
    }
    {
        FarmingUpdateMailbox mailbox(state, false, true, memory); (void)mailbox.submit_control("pause", {a});
        check(!mailbox.cancel(), "policy cancellation invented a compact control request");
        check(farming_control_view(state, mailbox.records(), mailbox.control_state()).effective(0) == "paused",
            "policy cancellation silently undid durable local pause intent");
    }
    {
        auto const pressure_root = root / "protected-control-quota"; fs::create_directory(pressure_root);
        auto pressure = group_farm_progress_test::fixture(pressure_root); pressure.version = 9;
        for (auto& directory : pressure.directories)
        { directory.layout.families = 1; directory.grant_config = pressure.config; directory.grant_extents = directory.extents; }
        pressure.manifest_reserved_bytes = 2 * 1024 * 1024;
        auto const policy_allowance = farming_update_reservation(4096, 4096);
        auto const control_allowance = farming_control_reservation(pressure, 4096);
        pressure.update_reserved_bytes = policy_allowance + control_allowance; save(pressure);
        auto const path = file_io::native_path(pressure.directories.front().directory);
        auto const peer = file_io::native_path(pressure.directories.back().directory); fs::rename(peer, pressure_root / "offline-peer");
        FarmingUpdateMailbox mailbox(pressure, false, true, memory);
        auto oversized = farming_update_snapshot(pressure); ++oversized.capacity;
        oversized.profile.emplace(); oversized.profile->assumptions.emplace_back();
        oversized.id = std::string(64, '0'); oversized.generation = 1; farming_update_preconditions(oversized, pressure);
        auto const base = farming_update_text(oversized).size() + 65;
        auto const receipts = farming_update_aligned(farming_update_receipt_bytes, 4096) * 2;
        auto const request_bytes = policy_allowance - receipts + 4096;
        check(request_bytes > base && request_bytes + receipts <= pressure.update_reserved_bytes,
            "policy-pressure fixture does not cross only the protected control slice");
        oversized.profile->assumptions.front().assign(size_t(request_bytes - base), 'x');
        bool refused = false;
        try { (void)mailbox.submit(oversized); }
        catch (FarmingUpdateSpaceError const& error) { refused = error.required > error.available; }
        check(refused && mailbox.records().empty(), "policy request borrowed the locally reserved control snapshot allowance");
        for (size_t request = 0; request < farming_update_slots; ++request)
        {
            auto update = farming_update_snapshot(pressure); update.capacity += request + 1;
            farming_update_preconditions(update, pressure); (void)mailbox.submit(update);
        }
        auto const policies = mailbox.records(); check(policies.size() == farming_update_slots, "policy pressure did not retain its unavailable-peer queue");
        (void)mailbox.submit_control("pause");
        check(farming_control_view(pressure, mailbox.records(), mailbox.control_state()).control == "paused" &&
            farming_update_allocation(farming_metadata_path(path, "grow-farm.plan", false).parent_path(), true) <= control_allowance &&
            mailbox.records().size() == policies.size(), "full policy queue prevented a healthy member from persisting local pause");
        (void)mailbox.submit_control("resume");
        check(farming_control_view(pressure, mailbox.records(), mailbox.control_state()).control == "active",
            "full policy queue prevented a causal local resume replacement");
    }
    FarmingMetadataDirectory metadata(left, false);
    auto const allowance = farming_owner_reservation(4096);
    auto const owner_before = observe_farming_owner(metadata, state, 0, memory);
    check(owner_before.status == "not-recorded" && !fs::exists(metadata.path / "grow-farm.owner"),
        "read-only ownership inspection created bookkeeping or invented a live owner");
    rejects([&] { FarmingOwner refused(metadata, state, 0, view.digest, 0, memory); },
        "owner heartbeat borrowed unreserved metadata from a legacy job");
    int ready[2]{}; check(!::pipe(ready), "cannot create owner fixture readiness pipe");
    auto const child = ::fork(); check(child >= 0, "cannot fork live owner fixture");
    if (!child)
    {
        ::close(ready[0]); DirectoryLock writer(left);
        FarmingOwner owner(metadata, state, 0, view.digest, allowance, memory);
        char marker = 'y'; if (::write(ready[1], &marker, 1) != 1) std::_Exit(2);
        ::close(ready[1]); for (;;) ::pause();
    }
    ChildGuard owner_child{child};
    ::close(ready[1]); char marker = 0; check(::read(ready[0], &marker, 1) == 1 && marker == 'y', "owner fixture did not start"); ::close(ready[0]);
    auto health = observe_farming_owner(metadata, state, 0, memory);
    check(health.status == "live" && health.lock_held && health.local_process_proven && health.heartbeat_current,
        "status did not distinguish the current local owner from saved progress");
    auto diagnosis = [&](FarmingOwnerHealth const& owner, std::string control = "active", std::string acknowledgement = "effective")
    {
        return grow_controller_test::farming_status_diagnosis("ready", "", "continue with fill DIR", control,
            acknowledgement, "ready", owner, true, false);
    };
    auto live_diagnosis = diagnosis(health);
    check(live_diagnosis.status == "running" && !live_diagnosis.blocking && live_diagnosis.evidence.find("last-known") != std::string::npos,
        "current ownership incorrectly promoted saved progress to a current construction inspection");
    auto pending_control = diagnosis(health, "draining", "pending-boundary");
    check(pending_control.status == "control-pending" && pending_control.blocking && pending_control.affected_work.find("already started") != std::string::npos,
        "unacknowledged drain intent was reported as completed drain or lost started work");
    auto const owner_stamp = stream_state::stamp(metadata.path / "grow-farm.owner");
    {
        auto body = farming_owner_detail::read_record(metadata.path / "grow-farm.owner", memory).body;
        check(body.starts_with("ordinary-group-farming-owner-v2\n") && farming_owner_detail::parse(body).descriptor >= 0,
            "current owner did not persist its descriptor lifetime binding");
        body.replace(0, std::string_view("ordinary-group-farming-owner-v2").size(), "ordinary-group-farming-owner-v1");
        body.erase(body.rfind(' ')); body += '\n';
        auto const legacy = farming_owner_detail::parse(body);
        check(legacy.descriptor == -1, "legacy owner acquired an invented descriptor binding");
#ifdef __linux__
        check(!farming_owner_detail::local_process(legacy, owner_stamp), "legacy PID metadata proved an owner's current object lifetime");
#endif
    }
    rejects([&] { DirectoryLock contender(left); }, "a concurrent writer bypassed the live destination owner");
    (void)observe_farming_owner(metadata, state, 0, memory);
    check(stream_state::stamp(metadata.path / "grow-farm.owner") == owner_stamp && farming_owner_allocation(metadata) <= allowance,
        "read-only owner status changed its immutable record or exceeded its fixed metadata grant");
    check(!::kill(child, SIGKILL), "cannot kill owner fixture"); int status = 0;
    check(::waitpid(child, &status, 0) == child && WIFSIGNALED(status), "owner fixture did not stop");
    owner_child.pid = -1;
    health = observe_farming_owner(metadata, state, 0, memory);
    check(health.status == "stale-owner" && !health.lock_held && !health.local_process_proven,
        "killed owner remained live through its old PID or heartbeat timestamp");
    auto killed_diagnosis = diagnosis(health);
    check(killed_diagnosis.status == "stale-owner" && killed_diagnosis.next.find("fill DIR") != std::string::npos &&
        killed_diagnosis.condition.find("lock is released") != std::string::npos,
        "killed owner's saved ready record produced live readiness or no safe recovery action");
    {
        DirectoryLock writer(left); FarmingOwner replacement(metadata, state, 0, view.digest, allowance, memory);
        check(observe_farming_owner(metadata, state, 0, memory).status == "live", "stale owner prevented an authorized replacement");
    }
    check(observe_farming_owner(metadata, state, 0, memory).status == "stopped", "clean owner shutdown was reported as current or crashed");
#ifdef __linux__
    {
        int signal[2]{}; check(!::pipe(signal), "cannot create stale-owner reader fixture readiness pipe");
        auto const reader = ::fork(); check(reader >= 0, "cannot fork stale-owner reader fixture");
        if (!reader)
        {
            ::close(signal[0]);
            try
            {
                int descriptor = -1;
                {
                    DirectoryLock writer(left);
                    FarmingOwner owner(metadata, state, 0, view.digest, allowance, memory);
                    descriptor = farming_owner_detail::parse(farming_owner_detail::read_record(metadata.path / "grow-farm.owner", memory).body).descriptor;
                    check(!::chmod(metadata.path.c_str(), 0500), "cannot prevent stopped-heartbeat publication in fixture");
                }
                check(!::chmod(metadata.path.c_str(), 0700), "cannot restore fixture metadata permissions");
                auto const heartbeat = farming_owner_detail::parse_heartbeat(farming_owner_detail::read_record(metadata.path / "grow-farm.heartbeat", memory).body);
                check(heartbeat.state == "running" && descriptor >= 0, "stale owner fixture did not retain its unreported shutdown");
                farm::inventory_detail::Handle probe(file_io::open_file(metadata.path / "grow-farm.owner",
                    file_io::read_only | file_io::close_on_exec | file_io::no_follow));
                check(::dup2(probe.fd, descriptor) == descriptor && !file_io::lock_file(descriptor,
                    file_io::lock_exclusive | file_io::lock_nonblocking), "cannot overlap reader probe with stale descriptor identity");
                char marker = 'y'; if (::write(signal[1], &marker, 1) != 1) std::_Exit(3);
                ::close(signal[1]); for (;;) ::pause();
            }
            catch (...) { std::_Exit(3); }
        }
        ChildGuard cleanup{reader}; ::close(signal[1]); char marker = 0;
        check(::read(signal[0], &marker, 1) == 1 && marker == 'y', "stale-owner reader fixture did not start"); ::close(signal[0]);
        auto const stale = observe_farming_owner(metadata, state, 0, memory);
        check(stale.lock_held && stale.heartbeat_current && !stale.local_process_proven && stale.status != "live",
            "reader lock made a still-alive former owner's unreported shutdown look live");
        check(!::kill(reader, SIGKILL), "cannot stop stale-owner reader fixture"); int result = 0;
        check(::waitpid(reader, &result, 0) == reader && WIFSIGNALED(result), "stale-owner reader fixture did not stop"); cleanup.pid = -1;
    }
#endif
    {
        FarmingControlState source_controls;
        { FarmingUpdateMailbox mailbox(state, false, false, memory); source_controls = mailbox.control_state(); }
        auto expanded = state; auto added = state.directories.back(); auto const added_path = root / "added-native-member"; fs::create_directory(added_path);
        added.directory = file_io::path_text(added_path); added.directory_identity = farming_directory_identity(added_path);
        expanded.directories.push_back(added); ++expanded.revision; expanded.policy_revision = expanded.revision;
        check(farming_control_view(expanded, {}, parse_farming_controls(expanded, farming_control_state_text(source_controls), memory)).effective(0) == "paused",
            "an ordinary appended member invalidated existing native control intent");
        auto const moved = root / "explicit-control-relocation"; fs::copy(left, moved, fs::copy_options::recursive);
        auto target = state; target.directories[0].directory = file_io::path_text(moved);
        target.directories[0].directory_identity = farming_directory_identity(moved); ++target.revision; target.policy_revision = target.revision;
        target.parent_digest = stream_state::digest(farming_state_text(state));
        std::vector<FarmingControlNativeBinding> bindings{{a, target.directories[0].directory_identity}};
        std::vector<fs::path> paths{moved, right};
        rejects([&] { FarmingUpdateMailbox copied(target, false, false, memory, paths); (void)copied.control_state(); },
            "copied controls gained foreign-native authority without an approved handoff");
        FarmingMetadataDirectory destination(moved, false);
        rejects([&] { (void)observe_farming_owner(destination, target, 0, memory); },
            "copied owner observation gained foreign-native authority without approval");
        check(observe_farming_owner(destination, target, 0, memory, a).status == "relocated-stale-owner",
            "approved copied observation was reported as a current live owner");
        DirectoryLock original_lock(left), target_lock(moved), peer_lock(right);
        {
            farm::inventory_detail::Handle active(file_io::open_file(destination.path / "grow-farm.owner",
                file_io::read_only | file_io::close_on_exec | file_io::no_follow));
            check(!file_io::lock_file(active.fd, file_io::lock_exclusive | file_io::lock_nonblocking), "cannot lock copied owner fixture");
            rejects([&] { FarmingOwner refused(destination, target, 0, view.digest, allowance, memory, a); },
                "approved native mapping bypassed a held copied ownership inode");
        }
        auto const crash = ::fork(); check(crash >= 0, "cannot fork control native handoff fixture");
        if (!crash)
        {
            ::setenv("XCHPLOT2_GROW_FAILPOINT", "farm-control-published", 1);
            FarmingUpdateMailbox handoff(target, false, true, memory, paths, bindings); (void)handoff.handoff_controls(); std::_Exit(0);
        }
        int result = 0; check(::waitpid(crash, &result, 0) == crash && WIFEXITED(result) && WEXITSTATUS(result) == 86,
            "native control handoff missed its partial-publication boundary");
        {
            FarmingUpdateMailbox handoff(target, false, true, memory, paths, bindings);
            check(farming_control_view(target, {}, handoff.control_state()).effective(0) == "paused",
                "partial native control publication lost the destination's pause");
            (void)handoff.handoff_controls();
            handoff_farming_owner(destination, target, 0, a, allowance, memory);
            save(target);
        }
        {
            FarmingUpdateMailbox current(target, false, false, memory, paths);
            auto const controls = current.control_state();
            check(farming_control_view(target, {}, controls).effective(0) == "paused" &&
                !controls.context.contains(a) && controls.fields.contains("control:" + target.directories[0].directory_identity),
                "completed native handoff retained an old actor or lost scoped intent");
        }
        check(observe_farming_owner(destination, target, 0, memory).status == "stopped",
            "completed native owner handoff still required the old source lineage");
        for (auto const* point : {"farm-owner-handoff-heartbeat", "farm-owner-handoff-published"})
        {
            fs::copy_file(metadata.path / "grow-farm.owner", destination.path / "grow-farm.owner", fs::copy_options::overwrite_existing);
            fs::copy_file(metadata.path / "grow-farm.heartbeat", destination.path / "grow-farm.heartbeat", fs::copy_options::overwrite_existing);
            auto const child = ::fork(); check(child >= 0, "cannot fork owner native handoff fixture");
            if (!child)
            {
                ::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1);
                handoff_farming_owner(destination, target, 0, a, allowance, memory); std::_Exit(0);
            }
            check(::waitpid(child, &result, 0) == child && WIFEXITED(result) && WEXITSTATUS(result) == 86,
                "native owner handoff missed its durable publication boundary");
            handoff_farming_owner(destination, target, 0, a, allowance, memory);
            check(observe_farming_owner(destination, target, 0, memory).status == "stopped",
                "interrupted native owner handoff invented live ownership or lost recovery");
        }
        auto retired = target;
        for (auto& directory : retired.directories)
        { directory.membership = "detached"; directory.control = "retired"; directory.detached_acknowledged = directory.detached_reclaimed = true; }
        auto const frozen = bytes(destination.path / "grow-farm.update.controls");
        auto const frozen_stamp = stream_state::stamp(destination.path / "grow-farm.update.controls");
        FarmingUpdateMailbox terminal(retired, false, true, memory, paths);
        check(terminal.terminal() && terminal.records().empty() &&
            farming_control_view(retired, {}, terminal.control_state()).control == "retired", "fully retired metadata regained historical control authority");
        auto legacy = farming_update_snapshot(retired); legacy.action = "resume"; farming_update_preconditions(legacy, retired, {"control"});
        legacy.generation = 1; legacy.id = farming_update_id(legacy);
        auto const legacy_body = farming_update_text(legacy);
        FarmingUpdateRecord old_resume{{legacy, {}}, {}, legacy_body, stream_state::digest(legacy_body)};
        auto const retired_view = farming_control_view(retired, {old_resume});
        check(retired_view.control == "retired" && retired_view.effective(0) == "retired",
            "legacy queued resume overrode durable detached membership");
        for (auto const* action : {"resume", "pause", "drain", "retire"}) (void)terminal.submit_control(action);
        check(bytes(destination.path / "grow-farm.update.controls") == frozen &&
            stream_state::stamp(destination.path / "grow-farm.update.controls") == frozen_stamp,
            "repeat control rewrote a frozen retired metadata snapshot");
        rejects([&] { (void)terminal.submit_control("resume", {"foreign-native-member"}); }, "terminal idempotence accepted a foreign destination");
    }
    {
        DirectoryLock writer(left);
        auto remote = farming_owner_detail::local_record(state, 0); remote.boot = "another-kernel-boot";
        stream_state::atomic_save(metadata.path / "grow-farm.owner", farming_owner_detail::checked(farming_owner_detail::text(remote)));
        farming_owner_detail::Heartbeat heartbeat;
        heartbeat.instance = remote.instance; heartbeat.control = view.digest; heartbeat.at = farming_owner_detail::now_ms();
        stream_state::atomic_save(metadata.path / "grow-farm.heartbeat", farming_owner_detail::checked(farming_owner_detail::text(heartbeat)));
        farm::inventory_detail::Handle proof(file_io::open_file(metadata.path / "grow-farm.owner",
            file_io::read_only | file_io::close_on_exec | file_io::no_follow));
        check(!file_io::lock_file(proof.fd, file_io::lock_exclusive | file_io::lock_nonblocking), "cannot hold remote owner fixture inode");
        health = observe_farming_owner(metadata, state, 0, memory);
        check(health.status == "remote-or-unknown-owner" && health.lock_held && health.heartbeat_current && !health.local_process_proven,
            "a matching PID and fresh remote heartbeat were reported as proven local liveness");
        auto unknown = diagnosis(health);
        check(unknown.status == "owner-unconfirmed" && unknown.blocking && unknown.next.find("preserve") != std::string::npos,
            "remote or unproven ownership supplied local liveness or unsafe lock-removal guidance");
    }
    auto other_job = state; other_job.job_id = std::string(64, 'f');
    rejects([&] { (void)observe_farming_owner(metadata, other_job, 0, memory); }, "owner status accepted a foreign job binding");
    idle_control_wakeup(parent);
    std::cout << "bounded local farming controls and live ownership checks passed\n";
}
} // namespace group_farm_control_test
