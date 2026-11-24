// As in grow-worker-test, include the implementation to exercise bounded
// private ownership without adding a production cache-testing interface.
#include "host/ComputeWorker.cpp"
#include "host/ComputeSession.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/ComputeDiscovery.hpp"
#include "compute_pool_fixture.hpp"
#include <deque>
#include <iostream>

using namespace group_grow;
using group_tools::require;

namespace
{
template <typename Operation> void rejects(Operation&& operation)
{
    bool rejected = false;
    try { operation(); } catch (std::exception const&) { rejected = true; }
    require(rejected, "client accepted an expired task or invalid retry identity");
}

void retry_pressure()
{
    worker_detail::TaskRegistry registry;
    WorkerTaskIdentity identity("long-running-member");
    auto connection = registry.acquire(identity, "owner-session");
    uint64_t const sequence = connection->sequence;
    ++connection->next_attempt;
    for (size_t i = 0; i <= worker_detail::TaskRegistry::maximum; ++i)
    {
        WorkerTaskIdentity completed("completed-" + std::to_string(i));
        (void)registry.acquire(completed, "owner-session");
    }
    // The old bug evicted this state after connection teardown because its
    // submission timestamp predated all the intervening completed work.
    connection.reset();
    WorkerTaskIdentity following("following");
    (void)registry.acquire(following, "owner-session");
    auto retry = registry.acquire(identity, "owner-session");
    require(retry->sequence == sequence && retry->next_attempt == 2,
        "cache pressure discarded the logical member's retry sequence or attempt");
    auto copied = identity;
    require(registry.acquire(copied, "owner-session") == retry,
        "copying a retry handle detached its pinned task state");
    worker_detail::TaskRegistry another;
    rejects([&] { (void)another.acquire(identity, "owner-session"); });
    copied.task_id = "different-member";
    rejects([&] { (void)registry.acquire(copied, "owner-session"); });
    registry.reset();
    auto restarted = registry.acquire(identity, "new-owner-session");
    require(restarted != retry && restarted->sequence == 1 && restarted->next_attempt == 1,
        "a new owner session retained an obsolete retry sequence");

    worker_detail::TaskRegistry bounded;
    std::vector<WorkerTaskIdentity> active;
    active.reserve(worker_detail::TaskRegistry::maximum);
    for (size_t i = 0; i < worker_detail::TaskRegistry::maximum; ++i)
    {
        active.emplace_back("active-" + std::to_string(i));
        (void)bounded.acquire(active.back(), "bounded-session");
    }
    WorkerTaskIdentity overflow("overflow");
    rejects([&] { (void)bounded.acquire(overflow, "bounded-session"); });
}

void planning_feedback()
{
    using Planning = worker_detail::GenerationPlanning;
    using Output = pool::GenerationOutput;
    // Each planner carries its fixed worker histories. Keep the independent
    // scenarios off the test thread's stack, as the client does in its Impl.
    std::deque<Planning> planners;
    auto new_planner = [&]() -> Planning& { return planners.emplace_back(1000); };
    auto const now = Clock::now(); pool::Endpoint route{"127.0.0.1", 49101};
    WorkerTaskObservation fast; fast.worker_identity = "local-worker"; fast.worker_instance = "first";
    fast.kind = pool::Kind::GenerateCpu; fast.k = 18; fast.strength = 2; fast.work_units = 1;
    fast.local_direct = true; fast.execution_ns = 400; fast.transfer_ns = 100;
    auto hint = [&](Planning& planning, uint8_t k = 18, uint8_t strength = 2, pool::Kind kind = pool::Kind::GenerateCpu,
        Output output = Output::Packed)
    { return planning.suggested(k, strength, kind, output, now); };
    auto observe = [&](Planning& planning, WorkerTaskObservation const& value, pool::Endpoint const& endpoint)
    {
        auto const selected = planning.selected(value, endpoint, now);
        planning.observed(value, endpoint, selected, now); planning.released(value.worker_identity, selected);
    };
    auto train = [&](Planning& planning, WorkerTaskObservation const& value, pool::Endpoint const& endpoint)
    { for (unsigned i = 0; i < 3; ++i) observe(planning, value, endpoint); };
    auto& planning = new_planner();
    auto missing = fast; missing.transfer_ns = 0; train(planning, missing, route);
    require(hint(planning).batch == 1 && hint(planning).concurrency == 2,
        "missing delivery phase evidence trained a generation policy");
    observe(planning, fast, route); observe(planning, fast, route);
    require(hint(planning).batch == 1, "one noisy completion changed generation batching");
    observe(planning, fast, route);
    require(hint(planning).batch == 2 && hint(planning).concurrency == 3,
        "consistent generation observations did not train bounded hints");
    auto pressure = fast; pressure.execution_ns = 100; pressure.transfer_ns = 400;
    for (unsigned i = 0; i < 6; ++i) observe(planning, i & 1 ? fast : pressure, route);
    require(hint(planning).batch == 2 && hint(planning).concurrency == 3,
        "alternating result pressure bypassed planning hysteresis");
    require(hint(planning, 20).batch == 1 && hint(planning, 18, 3).batch == 1 &&
        hint(planning, 18, 2, pool::Kind::GenerateGpu).batch == 1,
        "generation hints crossed a plot parameter or backend boundary");
    auto slow = pressure; slow.worker_identity = "remote-worker"; slow.local_direct = false;
    train(planning, slow, route);
    require(hint(planning).batch == 2 && hint(planning).concurrency == 3,
        "a slow remote minority pinned the local worker's learned policy");
    auto gpu = fast; gpu.worker_identity = "gpu-worker"; gpu.kind = pool::Kind::GenerateGpu;
    train(planning, gpu, route);
    planning.failed(fast, route, planning.selected(fast, route, now), now);
    require(hint(planning).batch == 1 && hint(planning, 18, 2, pool::Kind::GenerateGpu).batch == 2,
        "a CPU worker failure changed another backend's policy");
    auto hybrid = gpu; hybrid.kind = pool::Kind::GenerateCpu; train(planning, hybrid, route);
    planning.failed(hybrid, route, planning.selected(hybrid, route, now), now);
    require(hint(planning, 18, 2, pool::Kind::GenerateGpu).batch == 2,
        "one backend's failure changed the same worker's other backend policy");
    require(hint(planning, 18, 2, pool::Kind::GenerateAny).batch == 1,
        "mixed-backend aggregation ignored its active worker majority");

    auto& replacement = new_planner(); train(replacement, fast, route);
    auto changed = fast; changed.worker_instance = "replacement";
    observe(replacement, changed, route);
    require(hint(replacement).batch == 1, "a replacement worker inherited an old incarnation's feedback");
    observe(replacement, changed, route); observe(replacement, changed, route);
    require(hint(replacement).batch == 2, "replacement worker could not learn its own feedback");
    auto relocated = route; ++relocated.port; observe(replacement, changed, relocated);
    require(hint(replacement).batch == 1, "a changed authenticated route inherited old feedback");
    changed.local_direct = false; observe(replacement, changed, relocated);
    require(hint(replacement).batch == 1, "a TLS route inherited local delivery votes");
    require(replacement.suggested(18, 2, pool::Kind::GenerateCpu, Output::Packed, now + std::chrono::milliseconds(1001)).concurrency == 2,
        "an idle worker retained stale pressure beyond the task timeout");

    auto& bounded = new_planner();
    for (unsigned i = 0; i < 256; ++i) observe(bounded, fast, route);
    require(hint(bounded).batch == pool::max_generation_batch_members && hint(bounded).concurrency == 64,
        "generation feedback exceeded or failed to reach its bounded policy");
    auto older_workload = fast; older_workload.k = 20; train(bounded, older_workload, route);
    auto retired_selection = bounded.selected(older_workload, route, now);
    for (size_t i = 0; i < Planning::maximum; ++i)
    {
        auto fresh = fast; fresh.worker_identity = "fresh-" + std::to_string(i);
        observe(bounded, fresh, route);
    }
    bounded.observed(older_workload, route, retired_selection, now);
    bounded.failed(older_workload, route, retired_selection, now);
    require(hint(bounded, 20).batch == 1, "late feedback resurrected a worker after bounded history eviction");
    auto& invalid = new_planner(); auto oversized = fast; oversized.worker_identity.assign(65, 'x');
    train(invalid, oversized, route); auto invalid_route = route; invalid_route.host.assign(254, 'x');
    train(invalid, fast, invalid_route);
    require(hint(invalid).batch == 1, "invalid protocol keys entered fixed planning storage");

    auto& late = new_planner(); auto old_selection = late.selected(fast, route, now);
    train(late, fast, route); changed = fast; changed.worker_instance = "new-process";
    train(late, changed, route);
    for (unsigned i = 0; i < 3; ++i) late.observed(pressure, route, old_selection, now);
    late.failed(fast, route, old_selection, now);
    require(hint(late).batch == 2 && hint(late).concurrency == 3,
        "an old incarnation's late completion or failure replaced current feedback");
    auto before_relocation = late.selected(changed, route, now);
    train(late, changed, relocated); train(late, changed, route);
    late.observed(changed, route, before_relocation, now); late.failed(changed, route, before_relocation, now);
    require(hint(late).batch == 2 && hint(late).concurrency == 3,
        "returning to an earlier route resurrected a fenced selection's feedback");
    auto expired_selection = late.selected(changed, route, now);
    auto const expired = now + std::chrono::milliseconds(1001);
    late.observed(changed, route, expired_selection, expired); late.failed(changed, route, expired_selection, expired);
    require(late.suggested(18, 2, pool::Kind::GenerateCpu, Output::Packed, expired).batch == 1,
        "a delayed completion revived a peer after its monotonic observation timeout");

    auto& measured = new_planner();
    auto remote = slow; remote.generation_output = Output::Contributions; remote.worker_slots = 1;
    remote.generation = {300, 100, 80, 120};
    auto price = [&](unsigned lanes, unsigned batch = 1, Output output = Output::Contributions)
    { return measured.estimate(18, 2, remote.kind, output, lanes, batch, now); };
    observe(measured, remote, route); observe(measured, remote, route);
    require(!price(2).receivers, "two samples priced an untrained remote receiver");
    observe(measured, remote, route);
    require(price(2).member_ns == 400 && price(2).receivers == 1 && !price(2, 1, Output::Packed).receivers &&
        hint(measured, 18, 2, remote.kind, Output::Contributions).batch == 2,
        "a remote singleton did not price its measured service and trial a coencoded batch");
    auto pair = remote; pair.work_units = 2; pair.generation = {700, 300, 250, 250};
    train(measured, pair, route);
    require(hint(measured, 18, 2, remote.kind, Output::Contributions).batch == 1,
        "a slower measured batch displaced the faster singleton");
    {
        auto& output_modes = new_planner();
        auto fresh = remote; fresh.kind = pool::Kind::GenerateGpu; fresh.generation_output = Output::Fresh;
        auto estimate = [&](Output output, unsigned batch)
        { return output_modes.estimate(18, 2, fresh.kind, output, 1, batch, now); };
        train(output_modes, fresh, route);
        require(estimate(Output::Fresh, 1).member_ns == 400 && !estimate(Output::Fresh, 2).receivers &&
            !estimate(Output::Contributions, 1).receivers && output_modes.probe_contribution_route(18, 2, now) &&
            output_modes.suggested(18, 2, fresh.kind, Output::Contributions, now).batch == 1 &&
            output_modes.dispatch_width(18, 2, fresh.kind, Output::Contributions, 3, 2, 1, 3, 2, now) == 2,
            "fresh group observations changed ordinary contribution sizing or priced an unmeasured group size");
        auto ordinary = fresh; ordinary.generation_output = Output::Contributions;
        ordinary.generation = {700, 300, 250, 250}; train(output_modes, ordinary, route);
        fresh.work_units = 2; train(output_modes, fresh, route);
        require(std::abs(estimate(Output::Fresh, 2).member_ns - 200) < 1e-6L &&
            std::abs(estimate(Output::Contributions, 2).member_ns - 1000) < 1e-6L,
            "fresh group delivery changed the measured ordinary contribution rate");
        auto const failed = output_modes.selected(fresh, route, now);
        output_modes.failed(fresh, route, failed, now);
        require(!estimate(Output::Fresh, 2).receivers && std::abs(estimate(Output::Contributions, 2).member_ns - 1000) < 1e-6L,
            "fresh group failure erased matching ordinary timings");
    }
    auto peer = remote; peer.worker_identity = "second-timed-worker";
    train(measured, peer, route);
    require(price(2).receivers == 2 && price(2).member_ns == 200 && price(1).member_ns == 400,
        "independent worker service did not improve the two-receiver forecast");
    {
        auto& wave = new_planner();
        auto const kind = pool::Kind::GenerateGpu;
        auto quick = remote; quick.kind = kind; quick.work_units = 2; quick.independent_members = true;
        quick.generation = {1800, 200, 200, 0};
        auto medium = peer; medium.kind = kind; medium.generation = {2400, 100, 100, 0};
        auto slow_wave = medium; slow_wave.worker_identity = "slow-wave-worker"; slow_wave.generation = {9900, 100, 100, 0};
        auto sizing = [&](uint64_t members)
        { return wave.estimate(18, 2, kind, Output::Contributions, 3, 3, now, members); };
        auto const probe = wave.selected(slow_wave, route, now);
        observe(wave, quick, route);
        auto const first = sizing(9);
        require(first.receivers == 1 && first.measured_slots == 1 && std::abs(first.member_ns - 1000) < 1e-6L &&
            !sizing(0).receivers && !wave.estimate(20, 2, kind, Output::Contributions, 3, 3, now, 9).receivers &&
            !wave.estimate(18, 2, kind, Output::Packed, 3, 3, now, 9).receivers &&
            !wave.estimate(18, 2, kind, Output::Contributions, 3, 1, now, 9).receivers,
            "a first completion did not price only its matching ready work");
        observe(wave, medium, route);
        require(sizing(9).receivers == 2 && sizing(9).measured_slots == 2,
            "a pending unmeasured probe suppressed the measured routes' scratch sizing");
        wave.observed(slow_wave, route, probe, now); wave.released(slow_wave.worker_identity, probe);
        auto const short_wave = sizing(9), long_wave = sizing(30);
        require(short_wave.receivers == 2 && short_wave.measured_slots == 3 &&
            std::abs(short_wave.member_ns - 5000.L / 7) < 1e-6L &&
            long_wave.receivers == 3 && long_wave.measured_slots == 3 && long_wave.member_ns < short_wave.member_ns,
            "the slow receiver changed the short scratch wave or was ignored for a longer wave");
    }
    peer.generation = remote.generation = {10, 990, 990, 10};
    auto& shared = new_planner();
    remote.worker_slots = 8; train(shared, remote, route);
    auto const limited = shared.estimate(18, 2, remote.kind, Output::Contributions, 8, 1, now);
    require(limited.receivers == 2 && std::abs(limited.member_ns - 990) < 1e-6L,
        "multiple worker slots multiplied shared link bandwidth");
    auto& prefetched = new_planner();
    remote.generation.frame_read_ns = remote.generation.frame_wait_ns;
    remote.generation.frame_wait_ns = 10;
    train(prefetched, remote, route);
    auto const overlapping = prefetched.estimate(18, 2, remote.kind, Output::Contributions, 8, 1, now);
    require(overlapping.receivers == limited.receivers && overlapping.member_ns == limited.member_ns,
        "read-ahead hid the shared link cost from the planner");
    auto compute_bound = peer; compute_bound.worker_slots = 2; compute_bound.generation = {550, 550, 550, 0};
    train(shared, compute_bound, route);
    auto const unequal = shared.estimate(18, 2, remote.kind, Output::Contributions, 2, 1, now);
    require(unequal.receivers == 2 && std::abs(unequal.member_ns - 550) < 1e-6L,
        "the individually fastest slots crowded out a faster combination on the shared link");
    auto& mixed_link = new_planner();
    auto link_heavy = peer; link_heavy.worker_slots = 2; link_heavy.generation = {100, 900, 900, 0};
    auto link_light = compute_bound; link_light.worker_identity = "light-worker";
    link_light.generation = {1200, 300, 300, 0};
    train(mixed_link, link_heavy, route); train(mixed_link, link_light, route);
    // Six sevenths of a receiver on the first worker and eight sevenths on
    // the second use two receivers and exactly one link.
    auto const partial = mixed_link.estimate(18, 2, remote.kind, Output::Contributions, 2, 1, now);
    require(partial.receivers == 2 && std::abs(partial.member_ns - 10500.L / 17) < 1e-6L,
        "the forecast did not time-share workers at the shared link limit");
    {
        auto& adapters = new_planner();
        auto first = link_heavy, second = link_heavy; second.worker_identity = "second-adapter-worker";
        first.ingress_interface = second.ingress_interface = 7; train(adapters, first, route); train(adapters, second, route);
        auto estimate = [&] { return adapters.estimate(18, 2, first.kind, Output::Contributions, 2, 1, now); };
        require(std::abs(estimate().member_ns - 900) < 1e-6L, "two workers on one adapter consumed two link budgets");
        second.ingress_interface = 9; train(adapters, second, route);
        require(std::abs(estimate().member_ns - 500) < 1e-6L && estimate().receivers == 2,
            "workers on distinct physical adapters could not receive in parallel");
        // The fast adapter saturates part way through a second receiver.
        // Give the remaining receiver time to the other physical link.
        second.generation = {1500, 500, 500, 0}; second.ingress_interface = 11; train(adapters, second, route);
        long double const expected = 1 / (1.L / 900 + (2 - 1000.L / 900) / 2000);
        require(std::abs(estimate().member_ns - expected) < 1e-6L,
            "separate links consumed more than the shared owner receiver budget");
        second.ingress_interface = 0; second.generation = first.generation; train(adapters, second, route);
        require(std::abs(estimate().member_ns - 900) < 1e-6L, "an unresolved path was assumed independent of a physical adapter");
        first.ingress_interface = 3; second.ingress_interface = 3;
        first.generation.frame_read_ns = second.generation.frame_read_ns = 100;
        first.generation.receiver_ns = second.generation.receiver_ns = 200;
        train(adapters, first, route); train(adapters, second, route);
        require(std::abs(estimate().member_ns - 550) < 1e-6L, "receiver stalls were charged again as link occupancy");
    }
    {
        auto& widths = new_planner(); auto packed = remote; packed.generation_output = Output::Packed; packed.connect_ns = 50;
        packed.generation = {70, 30, 30, 0}; train(widths, packed, route);
        packed.work_units = 2; packed.generation = {100, 20, 20, 0}; train(widths, packed, route);
        auto price = [&](unsigned width) { return widths.estimate(18, 2, packed.kind, Output::Packed, 1, width, now); };
        require(price(1).member_ns == 150 && price(2).member_ns == 85 && price(2).measured_packed_batch &&
            price(3).member_ns > 0 && !price(3).measured_packed_batch,
            "packed widths lost their delivery cost or treated other widths as an exact measured batch");
    }
    auto& dual_backend = new_planner();
    link_heavy.worker_slots = 1;
    auto efficient_cpu = link_heavy; efficient_cpu.generation = {1080, 120, 120, 0};
    link_heavy.kind = pool::Kind::GenerateGpu;
    train(dual_backend, link_heavy, route); train(dual_backend, efficient_cpu, route);
    link_heavy.worker_identity = "gpu-only-worker"; train(dual_backend, link_heavy, route);
    auto const combined = dual_backend.estimate(18, 2, pool::Kind::GenerateAny, Output::Contributions, 3, 1, now);
    require(combined.receivers == 2 && combined.measured_slots == 2 &&
        std::abs(combined.member_ns - 6000.L / 11) < 1e-6L,
        "a hybrid worker lost its low-link backend or offered the same slot twice");
    // Independently enumerate integer allocations and all time shares between
    // two allocations that straddle the link limit, for several receiver caps.
    auto& three_workers = new_planner();
    std::array<unsigned, 3> const service{1000, 1000, 1500}, wire{900, 650, 300};
    for (unsigned i = 0; i < service.size(); ++i)
    {
        auto sample = peer; sample.worker_identity = "worker-" + std::to_string(i); sample.worker_slots = 2;
        sample.generation = {service[i] - wire[i], wire[i], wire[i], 0}; train(three_workers, sample, route);
    }
    long double previous = 0; unsigned needed = 0;
    for (unsigned limit = 1; limit <= 6; ++limit)
    {
        struct Load { long double rate = 0, link = 0; };
        std::vector<Load> loads;
        for (unsigned a = 0; a <= 2; ++a) for (unsigned b = 0; b <= 2; ++b) for (unsigned c = 0; c <= 2; ++c)
        {
            if (a + b + c > limit) continue;
            Load load; std::array<unsigned, 3> const slots{a, b, c};
            for (size_t i = 0; i < slots.size(); ++i)
            {
                auto const rate = static_cast<long double>(slots[i]) / service[i];
                load.rate += rate; load.link += rate * wire[i];
            }
            loads.push_back(load);
        }
        long double best = 0;
        for (auto const& low : loads) if (low.link <= 1)
        {
            best = std::max(best, low.rate);
            for (auto const& high : loads) if (high.link > 1)
                best = std::max(best, low.rate + (high.rate - low.rate) * (1 - low.link) / (high.link - low.link));
        }
        if (best > previous + 1e-12L) { needed = limit; previous = best; }
        auto const estimate = three_workers.estimate(18, 2, remote.kind, Output::Contributions, limit, 1, now);
        require(estimate.receivers == needed && std::abs(estimate.member_ns - 1 / best) < 1e-6L,
            "worker rate or receiver count differed from the feasible allocation search");
    }
    auto& direct = new_planner();
    auto local = compute_bound; local.local_direct = true; local.generation_output = Output::Contributions;
    local.execution_ns = 1000; local.transfer_ns = 100; local.worker_slots = 1;
    remote.worker_slots = 1; train(direct, remote, route); train(direct, local, route);
    auto const local_and_remote = direct.estimate(18, 2, remote.kind, Output::Contributions, 2, 1, now);
    require(local_and_remote.receivers == 2 && std::abs(local_and_remote.member_ns - 11000.L / 21) < 1e-6L,
        "a co-located worker suppressed the remote forecast or consumed network bandwidth");
    auto other_backend = local; other_backend.kind = pool::Kind::GenerateGpu;
    observe(direct, other_backend, route);
    require(direct.estimate(18, 2, pool::Kind::GenerateAny, Output::Contributions, 2, 1, now).member_ns == local_and_remote.member_ns,
        "a cold backend hid the other backend's measured service on the same worker");
    measured.failed(remote, route, measured.selected(remote, route, now), now);
    require(price(2).receivers == 1 && price(2).measured_slots == 1 && price(2).member_ns == 400,
        "a failed route must leave the other worker's measured service available");

    auto& fleet = new_planner();
    require(fleet.probe_contribution_route(18, 2, now), "a cold GPU route was assumed to have a delivery sample");
    auto pipeline = pair; pipeline.kind = pool::Kind::GenerateGpu; pipeline.work_units = 3;
    pipeline.independent_members = true; pipeline.generation = {750, 150, 75, 50};
    auto width = [&] { return fleet.dispatch_width(18, 2, pool::Kind::GenerateGpu, Output::Contributions, 3, 2, 2, 3, 2, now); };
    auto packed = pipeline; packed.independent_members = false;
    packed.generation_output = Output::Packed; packed.work_units = 1;
    train(fleet, packed, route);
    require(fleet.probe_contribution_route(18, 2, now), "packed delivery suppressed an unmeasured contribution probe");
    auto& legacy = new_planner(); auto untimed = pipeline; untimed.generation = {}; observe(legacy, untimed, route);
    require(!legacy.probe_contribution_route(18, 2, now) && !legacy.estimate(18, 2, pipeline.kind, Output::Contributions, 2, 3, now).receivers,
        "an untimed delivery kept probing or supplied an unmeasured link rate");
    train(fleet, pipeline, route);
    auto& whole_group = new_planner();
    train(whole_group, pipeline, route);
    require(!whole_group.joint_contribution_members(18, 2, 2, now, 3) &&
        whole_group.joint_contribution_members(18, 2, 3, now, 3) == 3 &&
        !whole_group.joint_contribution_members(18, 2, 3, now, 3),
        "independent destinations fragmented a whole-group trial or duplicated its assignment");
    require(!fleet.probe_contribution_route(18, 2, now) && fleet.probe_contribution_route(20, 2, now) &&
        fleet.probe_contribution_route(18, 2, now + std::chrono::milliseconds(1001)),
        "a GPU route sample crossed workload or lifetime boundaries");
    auto failed = fleet.selected(pipeline, route, now);
    fleet.failed(pipeline, route, failed, now); fleet.released(pipeline.worker_identity, failed);
    require(fleet.probe_contribution_route(18, 2, now), "a failed route kept a cold contribution batch open");
    train(fleet, pipeline, route);
    require(width() == 2 && fleet.estimate(18, 2, pipeline.kind, Output::Contributions, 2, 3, now).member_ns == 300 &&
        !fleet.estimate(18, 2, pipeline.kind, Output::Contributions, 2, 1, now).receivers &&
        fleet.suggested(18, 2, pipeline.kind, Output::Contributions, now).batch == 1,
        "independent pipeline samples must price the route without training joint encoding or hiding an unknown slot");
    auto singleton = peer; singleton.kind = pipeline.kind; singleton.generation = {1350, 150, 100, 80};
    auto single = pipeline; single.independent_members = false; single.work_units = 1; single.generation = {400, 50, 25, 25};
    {
        auto& optional = new_planner();
        auto joint_width = [&](unsigned maximum = 2) { return optional.joint_contribution_members(18, 2, maximum, now); };
        require(!joint_width(), "a cold GPU split a ready independent batch");
        observe(optional, single, route);
        require(!joint_width(), "a singleton prevented the first independently pipelined batch");
        train(optional, pipeline, route);
        require(!joint_width(1) && joint_width() == 2 && !joint_width(3) && !joint_width(),
            "concurrent destinations queued extra joint trials while the first was pending");
        auto joint = single; joint.work_units = 2; joint.generation = {1200, 300, 150, 100};
        observe(optional, joint, route);
        require(!joint_width() && joint_width(3) == 3, "a finished joint trial prevented comparing the next ready width");
        auto faster_joint = joint; faster_joint.worker_identity = "fast-joint"; faster_joint.generation = {80, 20, 10, 10};
        auto other_pipe = pipeline; other_pipe.worker_identity = faster_joint.worker_identity;
        train(optional, faster_joint, route); train(optional, other_pipe, route);
        require(joint_width() == 2, "a faster free joint encoder was ignored");
        auto wider = faster_joint; wider.work_units = 3; wider.generation = {600, 300, 30, 30};
        observe(optional, wider, route);
        require(joint_width(3) == 2, "a wider joint batch displaced the faster smaller batch");
        auto const busy = optional.selected(faster_joint, route, now);
        require(!joint_width(), "an occupied joint encoder hid the free worker's faster pipeline");
        optional.released(faster_joint.worker_identity, busy);
        require(!optional.joint_contribution_members(18, 2, 2, now + std::chrono::milliseconds(1001)),
            "expired GPU measurements changed a fresh contribution choice");

        auto& coalesced = new_planner();
        train(coalesced, pipeline, route);
        auto const occupied = coalesced.selected(pipeline, route, now);
        require(!coalesced.joint_contribution_members(18, 2, 2, now),
            "an unmeasured joint batch queued behind an occupied worker");
        coalesced.released(pipeline.worker_identity, occupied);
        require(coalesced.suggested(18, 2, pipeline.kind, Output::Contributions, now).batch == 1 &&
            coalesced.joint_contribution_members(18, 2, 2, now) == 2,
            "coalesced singleton requests left a ready destination waiting for a singleton sample");
        observe(coalesced, joint, route);
        require(!coalesced.joint_contribution_members(18, 2, 2, now),
            "an independently delivered destination continued splitting after a slower joint trial");
        auto& mixed_routes = new_planner();
        train(mixed_routes, pipeline, route);
        auto slow_joint = joint; slow_joint.worker_identity = "joint-only";
        observe(mixed_routes, slow_joint, route);
        require(!mixed_routes.joint_contribution_members(18, 2, 2, now),
            "joint calibration repeated because the broker assigned the trial to another GPU");
        auto occupied_pipeline = mixed_routes.selected(pipeline, route, now);
        require(mixed_routes.joint_contribution_members(18, 2, 2, now) == 2 &&
            !mixed_routes.joint_contribution_members(18, 2, 2, now + std::chrono::nanoseconds(600)) &&
            mixed_routes.joint_contribution_members(18, 2, 2, now + std::chrono::nanoseconds(901)) == 2,
            "a busy pipeline must justify its wait without displacing a free encoder after an overrun");
        mixed_routes.released(pipeline.worker_identity, occupied_pipeline);

        auto& assigned = new_planner();
        train(assigned, pipeline, route);
        auto slow_single = single; slow_single.worker_identity = "slow-uplink";
        slow_single.generation = {1100, 700, 500, 300};
        observe(assigned, slow_single, route);
        require(assigned.joint_contribution_members(18, 2, 3, now) == 2,
            "initial joint calibration took three members before pricing two");
        auto slow_pair = slow_single; slow_pair.work_units = 2;
        auto selected_slow = assigned.selected(slow_pair, route, now);
        require(assigned.repartition_contributions(slow_pair, now) == 2,
            "a joint trial occupied a slow worker longer than the whole independent batch");
        assigned.released(slow_pair.worker_identity, selected_slow);
        require(!assigned.joint_contribution_members(18, 2, 3, now),
            "idle slow route repeatedly attracted the same unstarted calibration");
        selected_slow = assigned.selected(slow_single, route, now);
        require(assigned.joint_contribution_members(18, 2, 3, now) == 2,
            "a busy slow worker prevented the fast worker from testing joint encoding");
        assigned.released(slow_pair.worker_identity, selected_slow);
        auto fast_pair = single; fast_pair.work_units = 2;
        auto selected_fast = assigned.selected(fast_pair, route, now);
        require(!assigned.repartition_contributions(fast_pair, now), "a fast joint calibration was rejected");
        assigned.released(fast_pair.worker_identity, selected_fast);
        auto fast_busy = assigned.selected(pipeline, route, now);
        require(assigned.repartition_contributions(slow_pair, now) == 2 &&
            !assigned.repartition_contributions(slow_pair, now + std::chrono::nanoseconds(901)),
            "a busy pipeline needs a known remaining service time to displace an actual lease");
        auto slow_stream = slow_single; slow_stream.independent_members = true; slow_stream.work_units = 3;
        require(assigned.repartition_contributions(slow_stream, now) == 1 &&
            !assigned.repartition_contributions(slow_stream, now + std::chrono::nanoseconds(901)),
            "a slow worker held the full independent batch when another GPU could finish the remainder sooner");
        assigned.released(pipeline.worker_identity, fast_busy);
        require(!assigned.repartition_contributions(slow_stream, now, now + std::chrono::milliseconds(1)) &&
            assigned.repartition_contributions(slow_stream, now, now) == 1,
            "an independent delivery lost its full batch during a long fill or retained it at the tail");
        require(assigned.repartition_contributions(slow_stream, now) == 1 &&
            !assigned.repartition_contributions(pipeline, now) && !assigned.repartition_contributions(slow_single, now),
            "partitioning changed the fast pipeline or tried to split a singleton");
        auto competitive = slow_pair; competitive.generation = {100, 100, 100, 40};
        observe(assigned, competitive, route);
        require(!assigned.repartition_contributions(slow_pair, now),
            "a measured faster joint delivery was displaced by the worker's old singleton cost");

        auto& pipeline_only = new_planner();
        train(pipeline_only, pipeline, route);
        auto prior = slow_stream; prior.generation = {3600, 1800, 600, 300};
        observe(pipeline_only, prior, route);
        require(pipeline_only.repartition_contributions(slow_pair, now) == 2 &&
            !pipeline_only.joint_contribution_members(18, 2, 2, now),
            "a worker first measured in a pipeline took or repeatedly requested a slower joint trial");

        auto& queued = new_planner();
        train(queued, pipeline, route);
        require(!queued.defer_queued_joint(18, 2, 2, now) && queued.joint_contribution_members(18, 2, 3, now) == 2 &&
            queued.defer_queued_joint(18, 2, 2, now) && !queued.defer_queued_joint(18, 2, 3, now),
            "a queued trial held an idle measured pipeline or changed another width");
        fast_busy = queued.selected(pipeline, route, now);
        require(!queued.defer_queued_joint(18, 2, 2, now), "a queued trial assumed an occupied alternative could start immediately");
        queued.released(pipeline.worker_identity, fast_busy);
        require(!queued.joint_contribution_members(18, 2, 3, now) &&
            !queued.defer_queued_joint(18, 2, 2, now + std::chrono::milliseconds(1001)),
            "a withdrawn calibration immediately retried or outlived its history");

        auto& balanced = new_planner();
        auto second = pipeline; second.worker_identity = "second-pipeline"; second.generation = {1200, 150, 100, 50};
        train(balanced, second, route);
        auto wide = second; wide.work_units = 8;
        require(!balanced.repartition_contributions(wide, now), "a wide stream was split without another measured pipeline");
        train(balanced, pipeline, route);
        require(balanced.repartition_contributions(wide, now) == 3,
            "a slower free GPU did not receive its lowest-completion-time share");
        fast_busy = balanced.selected(pipeline, route, now);
        require(balanced.repartition_contributions(wide, now) == 4,
            "the busy GPU's unfinished delivery was omitted from a two-way split");
        balanced.released(pipeline.worker_identity, fast_busy);

        auto& abandoned = new_planner();
        train(abandoned, pipeline, route);
        require(abandoned.joint_contribution_members(18, 2, 2, now) == 2, "missing initial joint trial");
        auto later = now + std::chrono::milliseconds(900);
        auto refreshed = abandoned.selected(pipeline, route, later);
        abandoned.observed(pipeline, route, refreshed, later); abandoned.released(pipeline.worker_identity, refreshed);
        require(!abandoned.joint_contribution_members(18, 2, 2, later) &&
            abandoned.joint_contribution_members(18, 2, 2, now + std::chrono::milliseconds(1001)) == 2,
            "an undelivered joint trial suppressed new calibration past its deadline");

        auto& bounded = new_planner();
        auto triple = joint; triple.work_units = 3; triple.generation = {300, 150, 50, 50};
        train(bounded, single, route); train(bounded, joint, route); train(bounded, triple, route);
        require(bounded.estimate(18, 2, pipeline.kind, Output::Contributions, 1, 2, now).member_ns == 450 &&
            bounded.estimate(18, 2, pipeline.kind, Output::Contributions, 1, 3, now).member_ns == 150,
            "a narrower destination inherited a joint rate slower than its measured singleton");
        auto slower_pipe = pipeline; slower_pipe.generation = {1800, 300, 150, 100};
        train(bounded, slower_pipe, route);
        require(bounded.joint_contribution_members(18, 2, 2, now) == 1,
            "a faster measured singleton was forced back into a pipeline");
    }
    for (bool singleton_first : {false, true})
    {
        auto& mixed = new_planner();
        if (singleton_first) observe(mixed, single, route);
        observe(mixed, pipeline, route); observe(mixed, pipeline, route);
        if (!singleton_first) observe(mixed, single, route);
        require(mixed.estimate(18, 2, pipeline.kind, Output::Contributions, 2, 3, now).member_ns == 350 &&
            !mixed.estimate(18, 2, pipeline.kind, Output::Contributions, 2, 1, now).receivers &&
            mixed.suggested(18, 2, pipeline.kind, Output::Contributions, now).batch == 1,
            "singleton and independent delivery samples must jointly price the pipeline without qualifying a singleton or joint batch");
    }
    train(fleet, singleton, route);
    auto const one_receiver = fleet.estimate(18, 2, pipeline.kind, Output::Contributions, 1, 3, now);
    auto const two_receivers = fleet.estimate(18, 2, pipeline.kind, Output::Contributions, 2, 3, now);
    require(one_receiver.member_ns == 300 && one_receiver.measured_slots == 2 &&
        two_receivers.receivers == 2 && std::abs(two_receivers.member_ns - 250) < 1e-6L,
        "the forecast must select the fastest single route and combine unequal rates when both can run");
    require(width() == 3, "the faster free GPU did not receive a full batch");
    auto const active_fast = fleet.selected(pipeline, route, now);
    require(width() == 1, "the slower free route received a full batch while the faster GPU was occupied");
    require(fleet.dispatch_width(18, 2, pool::Kind::GenerateGpu, Output::Contributions, 3, 2, 1, 3, 2, now, 3) == 3 &&
        fleet.dispatch_width(18, 2, pool::Kind::GenerateGpu, Output::Contributions, 3, 2, 1, 3, 2, now + std::chrono::nanoseconds(2000), 3) == 2,
        "a singleton-only route changed the batch queued for the independently eligible slot");
    auto const active_slow = fleet.selected(singleton, route, now);
    require(width() == 3 &&
        fleet.dispatch_width(18, 2, pool::Kind::GenerateGpu, Output::Contributions, 3, 2, 2, 3, 2, now + std::chrono::nanoseconds(2000)) == 2,
        "a queued independent batch must use a known next delivery and fall back when it runs late");
    fleet.released(pipeline.worker_identity, active_fast); fleet.released(singleton.worker_identity, active_slow);
    require(width() == 3, "completed work left a GPU slot occupied");
    auto const old_route = fleet.selected(pipeline, route, now);
    train(fleet, pipeline, relocated);
    require(width() == 3, "a changed route inherited its predecessor's outstanding lease");
    auto const new_route = fleet.selected(pipeline, relocated, now);
    fleet.released(pipeline.worker_identity, old_route);
    require(width() == 1, "a late release freed the replacement route's running slot");
    fleet.released(pipeline.worker_identity, new_route);

    auto& resident = new_planner();
    auto packed_width = [&] { return resident.dispatch_width(18, 2, pool::Kind::GenerateAny, Output::Packed, 3, 2, 2, 3, 1, now); };
    auto fast_packed = single, slow_packed = singleton;
    fast_packed.generation_output = slow_packed.generation_output = Output::Packed;
    train(resident, fast_packed, route); train(resident, singleton, route);
    require(packed_width() == 1, "a contribution route supplied an unmeasured packed batch");
    train(resident, slow_packed, route);
    auto cpu = slow_packed; cpu.kind = pool::Kind::GenerateCpu; cpu.worker_identity = fast_packed.worker_identity;
    train(resident, cpu, route);
    require(packed_width() == 3, "the faster packed route lost its batch or counted its CPU and GPU slots twice");
    auto active = resident.selected(fast_packed, route, now);
    require(packed_width() == 1, "an occupied fast worker assigned its full packed batch to a slower route");
    auto queued_width = [&](Clock::time_point at)
    { return resident.dispatch_width(18, 2, pool::Kind::GenerateAny, Output::Packed, 3, 3, 2, 3, 1, at); };
    require(queued_width(now) == 3 && queued_width(now + std::chrono::nanoseconds(2000)) == 1,
        "an extra packed destination lost an earlier delivery or relied on an overdue worker");
    auto busy = resident.selected(slow_packed, route, now);
    require(packed_width() == 3, "a queued packed request lost the batch of the first predicted worker");
    require(resident.dispatch_width(18, 2, pool::Kind::GenerateAny, Output::Packed, 3, 2, 2, 3, 1,
        now + std::chrono::nanoseconds(2000)) == 1, "an overdue worker committed a queued packed batch");
    resident.released(slow_packed.worker_identity, busy);
    resident.released(fast_packed.worker_identity, active);

    auto& receiving = new_planner();
    fast_packed.generation = {600, 400, 400, 0}; fast_packed.connect_ns = 0; fast_packed.worker_slots = 1;
    auto roster = std::make_shared<pool::GenerationRoster>();
    for (auto const& worker : {fast_packed, slow_packed})
        roster->routes.push_back({worker.worker_identity, worker.worker_instance, route, {}, {}, 0, 1});
    {
        auto& pair = new_planner();
        pair.capacity(18, 2, roster, 0, now);
        auto quick = fast_packed, helper = slow_packed;
        quick.execution_ns = 300; quick.transfer_ns = 100;
        helper.execution_ns = 4000; helper.transfer_ns = 4000;
        helper.generation = {4000, 4000, 4000, 0};
        train(pair, quick, route); train(pair, helper, route);
        auto ordinary = [&](unsigned paid)
        {
            WorkerPackedRoute target;
            auto const active = pair.selected(quick, route, now);
            auto const n = pair.dispatch_width(18, 2, quick.kind, Output::Packed, 3, 3, 2, 3, 1, now, 0, &target, true);
            pair.released(quick.worker_identity, active);
            require(n == 3 && target.route.identity == quick.worker_identity && target.paid_width == paid,
                "the slower singleton route inherited a batch sized for the earlier release");
        };
        ordinary(0);
        auto whole = quick; whole.work_units = 3; whole.execution_ns = 750;
        observe(pair, whole, route); ordinary(3);
        auto hybrid = std::make_shared<pool::GenerationRoster>(*roster); hybrid->routes.back().cpu_slots = 1;
        pair.capacity(18, 2, hybrid, pair.selections(), now);
        WorkerPackedRoute unpriced;
        require(pair.dispatch_width(18, 2, pool::Kind::GenerateAny, Output::Packed, 3, 3, 2, 3, 1, now, 0, &unpriced, true) == 3 &&
            unpriced.route.identity.empty(), "a measured GPU priced the unmeasured CPU backend on the same worker");
        pair.capacity(18, 2, roster, pair.selections(), now);
        helper.execution_ns = 200; helper.transfer_ns = 200; observe(pair, helper, route);
        WorkerPackedRoute open;
        require(pair.dispatch_width(18, 2, quick.kind, Output::Packed, 3, 3, 2, 3, 1, now, 0, &open, true) == 3 &&
            open.route.identity.empty(), "a competitive measured singleton was pinned behind the incumbent");
    }
    receiving.capacity(18, 2, roster, 0, now); train(receiving, fast_packed, route);
    auto selected = receiving.selected(slow_packed, route, now);
    pool::Lease first; first.worker_identity = slow_packed.worker_identity;
    first.task.kind = slow_packed.kind; first.task.k = 18; first.task.strength = 2; first.task.work_units = 1;
    WorkerPackedRoute planned;
    auto width_after = [&](unsigned ns, unsigned slots = 2)
    { return receiving.dispatch_width(18, 2, pool::Kind::GenerateGpu, Output::Packed, 3, 3, slots, 3, 1,
        now + std::chrono::nanoseconds(ns), 0, &planned); };
    require(width_after(6000) == 1, "a cold grant still connecting to its worker justified a packed batch");
    receiving.receiving_packed(first, selected, now + std::chrono::nanoseconds(2000));
    require(width_after(4000) == 1, "a short first transfer was assumed to outlast a measured whole batch");
    require(width_after(6000) == 3 && planned.route.identity == fast_packed.worker_identity,
        "a long first singleton transfer kept a measured worker on singletons");
    pool::Lease granted; granted.task.kind = fast_packed.kind; granted.worker_identity = fast_packed.worker_identity;
    granted.worker_instance = fast_packed.worker_instance; granted.endpoint = route;
    require(planned.matches(granted), "the measured worker lost its sized packed batch");
    granted.endpoint = relocated;
    require(!planned.matches(granted), "a changed data endpoint kept a batch sized for the previous endpoint");
    granted.endpoint = route; granted.worker_identity = slow_packed.worker_identity;
    require(!planned.matches(granted), "a different worker received the wide batch after sizing");
    auto in_progress = receiving.selected(fast_packed, route, now + std::chrono::nanoseconds(5900));
    require(width_after(6000) == 3 && width_after(7000) == 1, "the measured worker's release horizon was ignored");
    receiving.released(fast_packed.worker_identity, in_progress);
    auto spare = std::make_shared<pool::GenerationRoster>(*roster); spare->routes.back().gpu_slots = 2;
    receiving.capacity(18, 2, spare, receiving.selections(), now);
    require(width_after(6000, 3) == 1, "an unmeasured worker with a second free slot was excluded from packed sizing");
    receiving.capacity(18, 2, roster, receiving.selections(), now); receiving.released(slow_packed.worker_identity, selected);
    require(width_after(6000) == 1, "a completed first singleton still occupied its worker");
    selected = receiving.selected(slow_packed, route, now + std::chrono::nanoseconds(10000));
    auto const sent = now + std::chrono::nanoseconds(15000);
    require(width_after(20000) == 1, "an unconfirmed request reserved the measured worker's batch");
    receiving.receiving_packed(first, selected, sent);
    require(width_after(20000) == 3, "the validated prelude discarded service already spent on the singleton");
    receiving.released(slow_packed.worker_identity, selected);
    auto moved = std::make_shared<pool::GenerationRoster>(*roster); moved->routes.back().endpoint = relocated;
    auto replaced = receiving.selected(slow_packed, relocated, now); receiving.receiving_packed(first, selected, now);
    receiving.capacity(18, 2, moved, receiving.selections(), now);
    require(width_after(6000) == 1, "a stale member prelude from the old route occupied its replacement");
    receiving.released(slow_packed.worker_identity, replaced);

    {
        auto& joining = new_planner();
        auto extra = slow_packed; extra.worker_identity = "arriving-packed-worker";
        auto mixed = std::make_shared<pool::GenerationRoster>(*roster);
        mixed->routes.push_back({extra.worker_identity, extra.worker_instance, route, {}, {}, 0, 1});
        joining.capacity(18, 2, mixed, 0, now);
        train(joining, fast_packed, route);
        auto slower = slow_packed; slower.generation = {8000, 1000, 1000, 0};
        train(joining, slower, route);
        auto trial = joining.selected(extra, route, now);
        pool::Lease probe = first; probe.worker_identity = extra.worker_identity;
        joining.receiving_packed(probe, trial, now);
        auto width = [&](unsigned ns)
        { return joining.dispatch_width(18, 2, pool::Kind::GenerateGpu, Output::Packed, 3, 3, 3, 3, 1,
            now + std::chrono::nanoseconds(ns), 0, &planned); };
        require(width(100) == 1 && width(10000) == 3 && planned.route.identity == fast_packed.worker_identity,
            "a second measured helper kept a fast route on singletons during another worker's long first transfer");
        auto active_fast = joining.selected(fast_packed, route, now + std::chrono::nanoseconds(9700));
        require(width(10000) == 3 && planned.route.identity == fast_packed.worker_identity,
            "a wide batch stopped following the earlier measured release when an unpriced slot was occupied");
        auto prior_slow = joining.selected(slower, route, now + std::chrono::nanoseconds(4000));
        auto next_slow = joining.selected(slower, route, now + std::chrono::nanoseconds(5960));
        joining.observed(slower, route, prior_slow, now + std::chrono::nanoseconds(6000));
        joining.released(slower.worker_identity, prior_slow);
        require(width(10000) == 3 && planned.route.identity == fast_packed.worker_identity,
            "an overlapped receipt on a slower measured worker broke a batch that beats its next singleton");
        require(width(12000) == 1, "an overdue fast worker borrowed another worker's release estimate");
        joining.released(slower.worker_identity, next_slow);
        joining.released(fast_packed.worker_identity, active_fast);
        auto changed = std::make_shared<pool::GenerationRoster>(*mixed); changed->routes.back().endpoint = relocated;
        joining.capacity(18, 2, changed, joining.selections(), now);
        require(width(10000) == 1, "a replaced worker inherited its predecessor's occupied first transfer");
    }

    {
        auto& costs = new_planner();
        auto routes = std::make_shared<pool::GenerationRoster>(*roster);
        auto third = slow_packed; third.worker_identity = "wireless";
        routes->routes.push_back({third.worker_identity, third.worker_instance, route, {}, {}, 0, 1});
        costs.capacity(18, 2, routes, 0, now);
        auto quick = fast_packed, medium = slow_packed, whole = quick;
        quick.elapsed_ns = 3'500'000'000; medium.elapsed_ns = 25'000'000'000; third.elapsed_ns = 111'000'000'000;
        whole.work_units = 3; whole.elapsed_ns = 8'000'000'000;
        whole.generation = {}; // Older workers still deliver the measured owner receipt.
        auto wave = [&](unsigned width, WorkerPackedRoute const& target = {}, unsigned receivers = 3)
        { return costs.publication_trial_ns(18, 2, pool::Kind::GenerateGpu, Output::Packed, width, 3,
            receivers, *routes, target, now); };
        observe(costs, quick, route); observe(costs, whole, route);
        WorkerPackedRoute target{routes->routes.front(), pool::Kind::GenerateGpu};
        require(!wave(1) && wave(3, target) == whole.elapsed_ns && !wave(2, target),
            "a fast worker priced unmeasured singleton peers or an unobserved split");
        observe(costs, medium, route); observe(costs, third, route);
        require(wave(1) == third.elapsed_ns && wave(1, {}, 1) == 3.L * third.elapsed_ns && !wave(3),
            "a fast fleet rate concealed the slowest receipt or an unpriced whole-group route");
        auto changed = std::make_shared<pool::GenerationRoster>(*routes); changed->routes.back().endpoint = relocated;
        costs.capacity(18, 2, changed, costs.selections(), now);
        require(!wave(1) && !costs.publication_trial_ns(18, 2, pool::Kind::GenerateGpu, Output::Packed, 1, 3, 3, *changed, {}, now),
            "a replacement route inherited an earlier publication trial price");
    }
    auto& final_group = new_planner();
    final_group.capacity(18, 2, roster, 0, now);
    auto triple = fast_packed; triple.work_units = 3; triple.generation = {1200, 600, 600, 0};
    auto alternate = slow_packed; alternate.generation = {4000, 4000, 4000, 0}; alternate.connect_ns = 0;
    train(final_group, triple, route); train(final_group, alternate, route);
    auto earlier = [&](Clock::time_point at, pool::Kind kind = pool::Kind::GenerateAny)
    { return final_group.packed_before_singletons(18, 2, kind, 3, planned, at); };
    auto running = final_group.selected(triple, route, now);
    require(earlier(now) && planned.route.identity == triple.worker_identity,
        "a final measured batch was split although it beats a singleton on every alternative");
    require(!earlier(now + std::chrono::nanoseconds(2000)), "an overdue worker kept the final batch");
    auto packed_pair = triple; packed_pair.work_units = 2; packed_pair.generation = {750, 450, 450, 0};
    train(final_group, packed_pair, route);
    final_group.released(triple.worker_identity, running);
    running = final_group.selected(triple, route, now);
    std::shared_ptr<void> assigned_singleton;
    auto spare_fits = [&](unsigned ns) { return final_group.useful_packed_singleton(18, 2, pool::Kind::GenerateGpu, 3, 2,
        now, now + std::chrono::nanoseconds(ns), assigned_singleton); };
    require(!spare_fits(0) && !spare_fits(7000) && spare_fits(12000) && !spare_fits(12000),
        "packed work did not preserve the final fast batch or use the measured spare over a fitting independent horizon");
    assigned_singleton.reset();
    auto lost = std::make_shared<pool::GenerationRoster>(*roster); lost->routes.pop_back();
    final_group.capacity(18, 2, lost, final_group.selections(), now);
    require(!spare_fits(12000), "a withdrawn packed worker still supplied an idle singleton");
    final_group.capacity(18, 2, roster, final_group.selections(), now);
    {
        auto& shared_link = new_planner();
        shared_link.capacity(18, 2, roster, 0, now);
        auto full = triple, shorter = packed_pair, helper = alternate;
        full.generation = {1, 1800, 1800, 0}; shorter.generation = {1, 1200, 1200, 0};
        helper.generation = {1, 8000, 8000, 0};
        full.ingress_interface = shorter.ingress_interface = helper.ingress_interface = 7;
        train(shared_link, full, route); train(shared_link, shorter, route); train(shared_link, helper, route);
        shared_link.selected(full, route, now);
        require(!shared_link.useful_packed_singleton(18, 2, pool::Kind::GenerateGpu, 3, 2, now,
            now + std::chrono::nanoseconds(12000), assigned_singleton), "a saturated shared ingress funded another packed receiver");
    }
    final_group.released(triple.worker_identity, running);
    auto dual = std::make_shared<pool::GenerationRoster>(*roster); dual->routes.back().cpu_slots = 1;
    final_group.capacity(18, 2, dual, final_group.selections(), now);
    require(!earlier(now) && earlier(now, pool::Kind::GenerateGpu),
        "an unmeasured eligible backend was excluded from a final packed group");
    final_group.capacity(18, 2, roster, final_group.selections(), now);
    alternate.generation = {250, 250, 250, 0};
    for (unsigned i = 0; i < 16; ++i) observe(final_group, alternate, route);
    require(!earlier(now), "a faster singleton was excluded from a final packed group");
}

struct RunningBroker
{
    pool::BrokerServer server;
    std::atomic<bool> stop{false};
    std::thread thread;
    explicit RunningBroker(pool::BrokerOptions options)
        : server(std::move(options)), thread([this] { server.serve(stop); }) {}
    ~RunningBroker() { stop.store(true); thread.join(); }
};

struct StalledPeer
{
    pool::Listener listener;
    std::mutex mutex;
    std::condition_variable ready;
    bool release = false;
    std::atomic<bool> accepted{false};
    std::exception_ptr failure;
    std::thread thread;
    explicit StalledPeer(pool::TlsOptions tls)
        : listener({"127.0.0.1", 0}, tls), thread([this]
        {
            try
            {
                for (;;)
                {
                    auto channel = listener.accept();
                    if (!channel) { std::lock_guard lock(mutex); if (release) return; continue; }
                    (void)channel.receive(); accepted.store(true); ready.notify_all();
                    std::unique_lock lock(mutex); ready.wait(lock, [&] { return release; }); return;
                }
            }
            catch (...) { failure = std::current_exception(); ready.notify_all(); }
        }) {}
    void unblock()
    { { std::lock_guard lock(mutex); release = true; } ready.notify_all(); }
    ~StalledPeer() { unblock(); thread.join(); }
};

WorkerCancellation deadline()
{
    WorkerCancellation result;
    result.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
    return result;
}
template <typename Operation> void bounded_failure(Operation&& operation)
{
    auto const begin = std::chrono::steady_clock::now();
    rejects(std::forward<Operation>(operation));
    require(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1),
        "client waited for the longer socket timeout after its task deadline");
}

void confirmed_generation_prior()
{
    using Planning = worker_detail::GenerationPlanning;
    pool::Endpoint endpoint{"192.0.2.1", 4000};
    WorkerTaskObservation task; task.worker_identity = "worker"; task.worker_instance = "active";
    task.k = 18; task.strength = 2; task.work_units = 1; task.kind = pool::Kind::GenerateGpu;
    task.worker_slots = 1; task.execution_ns = 100; task.transfer_ns = 20; task.connect_ns = 5;
    task.ingress_interface = 7; task.generation = {100, 20, 20, 10, 20};
    auto routes = std::make_shared<pool::GenerationRoster>();
    routes->routes.push_back({"worker","active",endpoint,"host","device",0,1});
    unsigned lookups = 0;
    Planning planner(10000, [&](auto const& observed, auto const& route, uint64_t compute, uint64_t wire) {
        ++lookups;
        require(route.instance == observed.worker_instance && compute == 100 && wire > 0, "incorrect dated calibration input");
        return WorkerGenerationPrior{90,120,15,25,std::chrono::seconds(1),2,12345};
    });
    auto forecast = [&](unsigned width = 1) { return planner.estimate(18, 2, task.kind, task.generation_output, 1, width); };
    auto receive = [&] {
        auto const ticket = planner.selected(task, endpoint);
        planner.observed(task, endpoint, ticket); planner.released(task.worker_identity, ticket);
    };
    planner.capacity(18, 2, routes, planner.selections());
    require(!forecast().member_ns && !lookups, "a saved prior licensed an unobserved route");
    receive();
    auto const confirmed = forecast();
    require(confirmed.dated_prior && confirmed.prior_min_samples == 2 && confirmed.prior_oldest_utc == 12345 &&
        std::abs(confirmed.member_ns - 150) < 0.001 && !forecast(2).member_ns && lookups == 1,
        "a confirmed 120 ns worker ceiling plus 25 ns wire and 5 ns setup did not price the exact width");
    require(!planner.estimate(18, 2, task.kind, task.generation_output, 1, 1, Clock::now() + std::chrono::seconds(2)).member_ns,
        "a long-running owner kept a confirmed prior after its dated source expired");
    auto changed = std::make_shared<pool::GenerationRoster>(*routes); changed->routes.front().gpu_slots = 2;
    planner.capacity(18, 2, changed, planner.selections());
    require(!forecast().member_ns, "a changed offer kept the previous dated forecast");
    task.generation.frame_read_ns = task.generation.send_ns = 40; receive();
    require(!forecast().member_ns, "a current link slower than the dated range accepted the prior");
    receive();
    require(forecast().member_ns > 0 && !forecast().dated_prior, "live samples failed to replace older evidence");
    task.ingress_interface = 0; receive();
    require(!forecast().member_ns && lookups == 3, "an unknown physical ingress reused the dated route");
    task.ingress_interface = 7; task.generation.frame_read_ns = task.generation.send_ns = 20; receive();
    require(forecast().dated_prior, "restoring the measured ingress failed to reconfirm the prior");
    auto const failed = planner.selected(task, endpoint); planner.failed(task, endpoint, failed);
    planner.released(task.worker_identity, failed);
    require(!forecast().member_ns, "a failed delivery left the dated route available for the following assignment");
}
void live_generation_routes()
{
    using Planning = worker_detail::GenerationPlanning;
    auto const now = Clock::now();
    pool::Endpoint old{"127.0.0.1", 49111}, added{"127.0.0.1", 49112};
    WorkerTaskObservation value; value.worker_identity = "first"; value.worker_instance = "instance-a";
    value.kind = pool::Kind::GenerateGpu; value.k = 18; value.strength = 2;
    value.generation_output = pool::GenerationOutput::Contributions; value.work_units = value.worker_slots = 1;
    value.execution_ns = value.transfer_ns = 100; value.generation = {100, 100, 100, 100};
    pool::GenerationRoute first{value.worker_identity, value.worker_instance, old, {}, {}, 0, 1};
    pool::GenerationRoute second{"second", "instance-b", added, {}, {}, 0, 1};
    Planning planning(10000);
    auto offers = [&](std::vector<pool::GenerationRoute> routes, bool complete = true)
    {
        auto roster = std::make_shared<pool::GenerationRoster>(); roster->routes = std::move(routes); roster->complete = complete;
        planning.capacity(18, 2, roster, planning.selections(), now);
    };
    auto observe = [&](WorkerTaskObservation const& task, pool::Endpoint const& route)
    {
        auto const selection = planning.selected(task, route, now);
        planning.observed(task, route, selection, now); planning.released(task.worker_identity, selection);
    };
    auto probe = [&] { return planning.probe_contribution_route(18, 2, now); };
    auto measured = [&] { return planning.estimate(18, 2, value.kind, value.generation_output, 3, 1, now, 6).measured_slots; };
    offers({first}); observe(value, old);
    require(!probe() && measured() == 1, "one delivery did not price the existing GPU");
    offers({first, second});
    require(probe() && measured() == 1, "adding a GPU borrowed the established worker's delivery history");
    auto newcomer = value; newcomer.worker_identity = second.identity; newcomer.worker_instance = second.instance;
    auto wide = newcomer; wide.work_units = 3; wide.independent_members = true;
    auto selection = planning.selected(wide, added, now);
    require(planning.repartition_contributions(wide, now) == 1, "a late GPU was given an entire unmeasured pipeline");
    planning.released(wide.worker_identity, selection);
    selection = planning.selected(newcomer, added, now);
    require(!probe(), "a pending first delivery kept throttling established GPU pipelines");
    planning.observed(newcomer, added, selection, now); planning.released(newcomer.worker_identity, selection);
    require(!probe() && measured() == 2, "a useful first delivery did not price the new route");
    for (size_t i = 0; i <= Planning::maximum; ++i) observe(value, old);
    require(!probe() && measured() == 2, "other assignments repeatedly retried a slow but currently offered worker");
    offers({second}, false);
    require(measured() == 2, "a truncated listing evicted an unlisted measured worker");
    offers({second});
    require(measured() == 1, "withdrawn generation capacity stayed in the forecast");
    offers({first, second});
    require(!probe() && measured() == 2, "resuming the same route forced an unnecessary cold trial");
    auto late = planning.selected(value, old, now);
    ++first.endpoint.port; offers({first, second});
    require(probe() && measured() == 1, "an equal-slot endpoint replacement kept the old delivery cost");
    planning.observed(value, old, late, now); planning.released(value.worker_identity, late);
    require(measured() == 1, "an old route's late receipt reactivated its cost");
}

void network_deadlines()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "client-ca");
    auto identity = pool_test::issue(directory.path, "client-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::validate_tls_credentials(tls);
    auto other_key = pool_test::issue(directory.path, "admission-wrong-key", &ca);
    auto mismatched_key = tls; mismatched_key.key_file = pool_test::utf8_path(other_key.key_file);
    rejects([&] { pool::validate_tls_credentials(mismatched_key); });
    {
        pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
        auto broker = std::make_unique<RunningBroker>(options);
        WorkerClientConfig config; config.broker = options.listen = broker->server.endpoint(); config.tls = tls;
        config.owner_id = "idle-discovery"; WorkerClient client(config);
        require(bool(client.generation_capacity(18, 2).roster), "cold discovery did not open its owner session");
        broker.reset(); broker = std::make_unique<RunningBroker>(options);
        auto const until = Clock::now() + std::chrono::seconds(5);
        while (!client.generation_capacity(20, 2).roster && Clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        require(bool(client.generation_capacity(20, 2).roster), "idle discovery never recovered after a broker restart");
        pool::BrokerClient control(config.broker, tls);
        auto replacement = control.open_owner(config.owner_id);
        require(!client.generation_capacity(22, 2).roster, "advisory discovery displaced a new owner in the same broker epoch");
        control.close_owner(replacement);
    }
    {
        StalledPeer peer(tls);
        WorkerClientConfig config; config.broker = peer.listener.endpoint(); config.tls = tls;
        config.owner_id = "client-cpu-wait-deadline";
        config.cpu_scheduler = std::make_shared<CodecScheduler>(1);
        WorkerClient client(config);
        std::array<uint64_t, 3> fragments{1, 2, 3};
        std::span<uint64_t const> region(fragments);
        WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
        std::future<void> pending;
        bool completed;
        {
            CodecScheduler::Lease pressure(config.cpu_scheduler.get());
            auto cancel = deadline();
            pending = std::async(std::launch::async, [&, cancel]
            { bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); }); });
            completed = pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
        }
        pending.get();
        require(completed && !peer.accepted.load(), "client CPU admission ignored the task deadline");
    }
    {
        StalledPeer peer(tls);
        WorkerClientConfig config; config.broker = peer.listener.endpoint(); config.tls = tls;
        config.owner_id = "lazy-open-deadline";
        WorkerClient client(config);
        require(!peer.accepted.load(), "client construction opened an owner session before a task");
        std::array<uint8_t, 32> id{};
        auto cancel = deadline();
        bounded_failure([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, cancel); });
        require(peer.accepted.load(), "lazy-owner deadline test never reached authenticated control");
    }
    {
        StalledPeer peer(tls);
        WorkerClientConfig config; config.broker = peer.listener.endpoint(); config.tls = tls;
        config.owner_id = "sequence-wait-deadline";
        WorkerClient client(config); std::array<uint8_t, 32> id{};
        auto first = std::async(std::launch::async, [&]
        { rejects([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}); }); });
        {
            std::unique_lock lock(peer.mutex);
            require(peer.ready.wait_for(lock, std::chrono::seconds(1), [&] { return peer.accepted.load(); }),
                "first owner-open request did not reach the stalled broker");
        }
        auto cancel = deadline();
        bounded_failure([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, cancel); });
        peer.unblock(); first.get();
    }
    {
        pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
        RunningBroker broker(options);
        StalledPeer peer(tls);
        pool::WorkerOffer offer; offer.instance = pool::random_token(); offer.compatibility = worker_compatibility();
        offer.codec_compatibility = worker_compatibility(pool::Kind::Pack);
        offer.endpoint = peer.listener.endpoint(); offer.capacity = {1, 1, uint64_t(128) << 20, 0};
        offer.capabilities = pool::capability(pool::Kind::Pack);
        pool::BrokerClient control(broker.server.endpoint(), tls); control.register_worker(offer);
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "direct-result-deadline";
        WorkerClient client(config);
        std::array<uint64_t, 3> fragments{1, 2, 3};
        std::span<uint64_t const> region(fragments);
        WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
        auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        require(peer.accepted.load(), "direct deadline test never reached the leased TLS peer");
    }
}

void independent_receivers()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "receiver-ca");
    auto identity = pool_test::issue(directory.path, "receiver-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.max_frame_bytes = 1u << 20; tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    pool::Listener peer({"127.0.0.1", 0}, tls);
    pool::BrokerClient control(broker.server.endpoint(), tls);
    pool::WorkerOffer offer; offer.endpoint = peer.endpoint(); offer.instance = pool::random_token();
    offer.compatibility = worker_compatibility(); offer.capabilities = pool::capability(pool::Kind::GenerateGpu);
    offer.generation_threads = 2; offer.independent_k_mask = 1u << 18;
    offer.capacity = {2, 1, uint64_t(128) << 20, 1024};
    offer.generation_profiles = {{pool::Kind::GenerateGpu, 18, 2, 2, offer.capacity, offer.capacity.host_bytes, 4}};
    control.register_worker(offer);
    WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
    config.owner_id = "independent-receivers"; config.generation_kind = pool::Kind::GenerateGpu;
    config.contribution_read_ahead = true; config.max_active_generation = 3;
    WorkerClient client(config);
    require(client.generation_capacity(18, 2).batch == 1, "fixture must admit only singleton joint encoding");
    std::array<std::array<uint8_t, 32>, 2> joint_ids{};
    bool repartition = false;
    try
    {
        WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::seconds(2);
        client.contributions(18, 2, joint_ids, [](auto, auto, auto) {}, cancel);
    }
    catch (pool::UnsupportedResources const&) { repartition = true; }
    require(repartition, "a joint batch that only fits after splitting waited for its deadline");
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(18);
    auto worker = std::async(std::launch::async, [&]
    {
        for (unsigned count : {1u, 2u, 1u, 1u, 2u, 2u})
        {
            auto channel = peer.accept(10000); require(bool(channel), "missing independent receiver task");
            auto frame = channel.receive(); pool::Reader request(frame.payload);
            auto lease = pool::read_lease(request);
            require(frame.type == pool::Message::Work && lease.task.work_units == count &&
                lease.task.generation_output == pool::GenerationOutput::Contributions &&
                lease.task.independent_members == (count > 1) && control.begin(lease, channel.peer_identity()),
                "receiver cohort did not share its GPU lease");
            (void)request.u32();
            require(request.u8() == uint8_t(worker_detail::Operation::Contributions) && request.u8() == 18 &&
                request.u8() == 2 && request.u16() == count, "invalid receiver cohort request");
            pool::Writer out(tls.max_frame_bytes); out.u8(uint8_t(worker_detail::DataKind::Contributions));
            for (unsigned i = 0; i < count; ++i)
            {
                auto const id = request.bytes(32);
                for (uint64_t n = 0; n < chunks; ++n)
                { out.u64(i * chunks + n); out.u16(1); out.bytes(id.first(1)); }
            }
            request.finish();
            channel.send(pool::Message::Data, out.data()); control.finish(lease);
            channel.send(pool::Message::End, {});
        }
    });
    // Keep the first request in its receiver until the two independent
    // callers are admitted. The shared frame crosses their member boundary.
    std::promise<void> started;
    auto run = [&](uint8_t tag)
    {
        WorkerCancellation cancel; cancel.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        std::array<uint8_t, 32> id{tag}; uint64_t next = 0;
        try {
            client.contributions(18, 2, {&id, 1}, [](auto, auto, auto) {}, cancel, {},
                [&](uint64_t first, auto window, uint16_t members)
                {
                    require(first == next && members == 1 && window.size() <= chunks - next, "receiver window crossed a member");
                    if (!tag && !next)
                    {
                        started.set_value();
                        while (client.stats().peak_active_generation < 3)
                        { cancel.check(); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
                    }
                    for (auto bytes : window) require(bytes.size() == 1 && bytes.front() == tag, "misrouted receiver data");
                    next += window.size();
                    if (tag == 1) throw WorkerTaskDeclined();
                }, {}, false, tag < 3);
        } catch (WorkerTaskDeclined const&) { require(tag == 1, "a full destination declined another member"); return; }
        require(tag != 1 && next == chunks, "independent member was truncated or declined incorrectly");
    };
    auto ready = started.get_future();
    auto first = std::async(std::launch::async, run, 0);
    require(ready.wait_for(std::chrono::seconds(10)) == std::future_status::ready, "initial receiver did not start");
    auto full = std::async(std::launch::async, run, 1), writable = std::async(std::launch::async, run, 2);
    first.get(); full.get(); writable.get();
    auto exact = std::async(std::launch::async, run, 3), other_exact = std::async(std::launch::async, run, 4);
    exact.get(); other_exact.get();
    require(client.stats().independent_members == 2 && client.stats().completed_contributions == 4,
        "the receiver changed an independently batched or explicitly single-member delivery");
    joint_ids[0][0] = 6; joint_ids[1][0] = 7;
    uint64_t received = 0;
    auto window = [&](uint64_t ordinal, std::span<std::span<uint8_t const> const> bytes, uint16_t members)
    {
        require(ordinal == received && members == 1, "adaptive lease changed its contribution framing");
        for (auto part : bytes)
        {
            require(part.size() == 1 && part.front() == joint_ids.at(received / chunks)[0], "adaptive lease changed its plot ID");
            ++received;
        }
    };
    auto sink = [&](uint64_t ordinal, std::span<uint8_t const> part, uint16_t members) { window(ordinal, {&part, 1}, members); };
    require(client.try_contributions(18, 2, joint_ids, sink, {}, {}, window, true) == 0 && received == 2 * chunks,
        "a sole eligible GPU failed to deliver its complete independent batch");
    bool declined = false;
    try { client.try_contributions(18, 2, joint_ids, sink, {}, {}, [](auto, auto, auto) { throw WorkerTaskDeclined(); }, true); }
    catch (WorkerTaskDeclined const&) { declined = true; }
    require(declined, "a destination rejection was reported as an unstarted lease split");
    worker.get();

    auto other_identity = pool_test::issue(directory.path, "second-gpu", &ca);
    auto other_tls = pool_test::tls(ca, other_identity);
    pool::Listener other_peer({"127.0.0.1", 0}, other_tls);
    offer.endpoint = other_peer.endpoint(); offer.instance = pool::random_token();
    pool::BrokerClient other(broker.server.endpoint(), other_tls);
    other.register_worker(offer);
    config.owner_id = "two-gpu-receivers";
    WorkerClient shared(config);
    require(shared.generation_capacity(18, 2).slots == 2 &&
        shared.independent_contribution_width(18, 2) == 1 && shared.independent_contribution_width(18, 2, 2) == 1 &&
        shared.independent_contribution_width(18, 2, 3) == 2 && shared.independent_contribution_width(18, 2, 8) == 3 &&
        shared.independent_contribution_width(18, 2, 3, 2) == 3,
        "ordered batches must leave work for the second slot within the owner's admission limit");

    other.unregister_worker(offer.instance);
    offer.instance = pool::random_token(); offer.independent_k_mask = 0;
    offer.capacity = {4, 2, uint64_t(256) << 20, 2048};
    other.register_worker(offer);
    config.owner_id = "mixed-output-receivers";
    WorkerClient mixed(config);
    auto const mixed_capacity = mixed.generation_capacity(18, 2);
    require(mixed_capacity.slots == 3 &&
        mixed_capacity.availability.independent_width(UINT32_MAX) == 4 &&
        !mixed_capacity.availability.independent_width(0) &&
        mixed.independent_contribution_width(18, 2, 3) == 1 &&
        mixed.independent_contribution_width(18, 2, 3, 2) == 3 &&
        mixed.independent_contribution_width(18, 2, 3) == 1,
        "a singleton-only worker disabled another destination's capable GPU or received an incompatible batch");
}

void assigned_local_and_fencing()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "local-ca");
    auto identity = pool_test::issue(directory.path, "local-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    auto data_tls = tls; data_tls.max_frame_bytes = 4096;
    WorkerConfig offer; offer.broker = broker.server.endpoint(); offer.listen = {"127.0.0.1", 0}; offer.tls = data_tls;
    offer.capacity = {2, 4, uint64_t(2) << 30, 0}; offer.generation_threads = 1;
    offer.available_resources = [capacity = offer.capacity](unsigned) { return capacity; };
    WorkerService service(offer); service.start();
    WorkerClientConfig config; config.broker = offer.broker; config.tls = data_tls;
    config.owner_id = "direct-local-owner"; config.local_service = &service;
    std::vector<WorkerTaskObservation> observations;
    config.observe = [&](auto const& value) { observations.push_back(value); };
    WorkerClient client(config);
    auto const generation_slots = client.generation_capacity(18, 2).slots;
    require(generation_slots && generation_slots < offer.capacity.slots, "fixture requires CPU-limited generation capacity");
    std::array<uint8_t, 32> id{0x91};
    std::vector<std::array<uint8_t, 32>> oversized(pool::max_generation_batch_members + 1);
    rejects([&] { client.contributions(18, 2, oversized, [](auto, auto, auto) {}); });
    std::unique_ptr<CompactMemberReceiver> receiver;
    uint64_t sink_ns = 0;
    client.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
    {
        if (!receiver)
        {
            auto const begin = Clock::now(); std::this_thread::sleep_for(std::chrono::milliseconds(10));
            sink_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
            receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count));
        }
        receiver->append(tile);
    });
    require(bool(receiver), "assigned local generation produced no member");
    auto generated = receiver->finish();
    require(generated.size(), "assigned local generation produced an empty member");
    auto stats = client.stats();
    require(stats.local_tasks == 1 && !stats.input_bytes && !stats.output_bytes &&
        observations.size() == 1 && observations.front().local_direct && observations.front().execution_ns &&
        observations.front().worker_slots == generation_slots &&
        sink_ns && observations.front().transfer_ns >= sink_ns,
        "ordinary local assignment used wire payload or omitted its observation");
    WorkerTaskObservation remote_observation;
    auto remote_config = config; remote_config.local_service = nullptr; remote_config.owner_id = "remote-observation-owner";
    remote_config.observe = [&](auto const& value) { remote_observation = value; };
    WorkerClient remote(remote_config); sink_ns = 0;
    remote.generate(18, 2, id, [&](auto const&, uint64_t)
    {
        if (sink_ns) return;
        auto const begin = Clock::now(); std::this_thread::sleep_for(std::chrono::milliseconds(10));
        sink_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
    });
    auto const timing = remote.stats();
    require(!remote_observation.local_direct && remote_observation.execution_ns &&
        remote_observation.member_id_count == 1 && remote_observation.member_ids[0] == id &&
        sink_ns && remote_observation.transfer_ns >= sink_ns && timing.timed_generation_tasks == 1 &&
        timing.generation.worker_ns && timing.generation.send_ns && timing.generation.frame_wait_ns &&
        timing.generation.receiver_ns >= sink_ns && remote_observation.worker_slots == generation_slots &&
        !stats.timed_generation_tasks,
        "TLS result consumption used a different sink observation boundary");

    std::array<uint64_t, 3> fragments{1, 2, 3}; std::span<uint64_t const> region(fragments);
    WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
    ChunkCodec codec(18, uint64_t(1) << 20);
    auto canonical = codec.encode(0, {}, 0, {&region, 1});
    std::vector<uint8_t> expected(canonical.begin(), canonical.end());
    WorkerTaskOptions per_call;
    bool selected = false, observed = false;
    per_call.selected = [&](auto const& value) { selected = value.local_direct; return true; };
    per_call.observe = [&](auto const& value) { observed = value.local_direct && !value.worker_instance.empty(); };
    auto encoded = client.encode(18, {&chunk, 1}, uint64_t(1) << 20, {}, {}, {}, per_call);
    require(encoded == std::vector<std::vector<uint8_t>>{expected} && selected && observed,
        "direct local codec or per-call worker metadata differs from canonical output");
    stats = client.stats();
    require(stats.local_tasks == 2 && !stats.input_bytes && !stats.output_bytes,
        "direct local codec counted a wire transfer");
    per_call.selected = [](auto const&) { return false; };
    bool declined = false;
    try { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, {}, {}, {}, per_call); }
    catch (WorkerTaskDeclined const&) { declined = true; }
    require(declined && client.stats().failed_tasks == stats.failed_tasks && client.stats().local_tasks == stats.local_tasks,
        "an optional selection decline counted as a worker execution or failure");

    // Fence between successful receipt/validation and its final acknowledgement.
    // A best-effort transport exception must not swallow an explicit rejection.
    pool::BrokerClient controller(offer.broker, tls);
    bool stale = false;
    try
    {
        (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, {}, {}, [&](auto result)
        {
            require(result.front() == expected, "fencing fixture received incorrect local bytes");
            (void)controller.open_owner(config.owner_id);
        });
    }
    catch (pool::StaleDelivery const&) { stale = true; }
    require(stale && client.stats().local_tasks == stats.local_tasks,
        "explicit fenced completion was accepted as a successful local result");
    require(client.encode(18, {&chunk, 1}, uint64_t(1) << 20).front() == expected,
        "client did not refresh its fenced owner session for later work");

    // Changing public generation parameters creates a distinct bounded model.
    uint64_t chunks = 0;
    std::array<std::array<uint8_t, 32>, 2> ids{id, id};
    client.contributions(18, 3, ids, [&](uint64_t ordinal, auto bytes, uint16_t members)
    { require(ordinal == chunks++ && !bytes.empty() && members == ids.size(), "changed-workload batch was truncated"); });
    require(chunks == PlotGroupFile::getChunkCountForK(18) && client.suggested_batch_members(18, 3, 2, pool::GenerationOutput::Contributions) <= 2 &&
        client.suggested_generation_concurrency(18, 3, 1, pool::GenerationOutput::Contributions) == 1,
        "adaptive worker suggestions exceeded the caller's storage or RAM ceiling");
    config.local_service = nullptr; config.owner_id = "minimum-frame-wire-owner";
    WorkerClient wire(config); chunks = 0;
    for (auto* owner : {&client, &wire})
    {
        auto const before = owner->stats(); bool received = false, declined = false;
        try { owner->contributions(18, 3, {&id, 1}, [&](auto ordinal, auto, auto)
            { require(!ordinal, "capacity probe missed its first chunk"); received = true; throw WorkerTaskDeclined(); }); }
        catch (WorkerTaskDeclined const&) { declined = true; }
        auto const after = owner->stats();
        require(received && declined && after.failed_tasks == before.failed_tasks &&
            after.completed_contributions == before.completed_contributions, "owner capacity stop penalized its worker");
    }
    wire.contributions(18, 3, ids, [&](uint64_t ordinal, auto bytes, uint16_t members)
    { require(ordinal == chunks++ && !bytes.empty() && members == ids.size(), "minimum-frame contribution batch was truncated"); });
    require(chunks == PlotGroupFile::getChunkCountForK(18) && wire.stats().output_bytes,
        "minimum data frame prevented the worker's larger control profile registration");
    std::vector<std::array<uint64_t, 3>> full_raw(pool::default_max_codec_chunks);
    std::vector<std::span<uint64_t const>> full_regions(full_raw.size());
    std::vector<WorkerChunkInput> full_window(full_raw.size());
    auto const range = uint64_t(1) << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t i = 0; i < full_raw.size(); ++i)
    { full_raw[i] = {i * range + 1, i * range + 2, i * range + 3}; full_regions[i] = full_raw[i];
      full_window[i] = {i, {}, 0, {}, {&full_regions[i], 1}}; }
    auto full_encoded = wire.encode(18, full_window, uint64_t(1) << 20, {}, {}, [&](auto output)
    {
        require(output.size() == full_window.size(), "full bounded codec window was truncated");
        for (size_t i = 0; i < output.size(); ++i)
            require(std::ranges::equal(output[i], codec.encode(i, {}, 0, {&full_regions[i], 1})),
                "full bounded codec window changed canonical bytes");
    });
    require(full_encoded.size() == pool::default_max_codec_chunks,
        "worker did not honor its advertised full codec chunk limit");
    service.stop();
    uint64_t maximum_chunk = 0; auto const view = generated.view();
    for (uint64_t ordinal = 0; ordinal < PlotGroupFile::getChunkCountForK(18); ++ordinal)
        maximum_chunk = std::max<uint64_t>(maximum_chunk, view.region(ordinal).size());
    auto bounded = offer;
    bounded.max_task_bytes = std::max<uint64_t>(bounded.tls.max_frame_bytes, ChunkCodec::estimate_work(18, 0, maximum_chunk));
    RunningBroker scratch_broker(options); bounded.broker = scratch_broker.server.endpoint();
    WorkerService scratch_limited(bounded); scratch_limited.start();
    config.broker = bounded.broker; config.owner_id = "batch-refusal-owner";
    config.local_service = nullptr; config.task_timeout_ms = 10000;
    std::string failed_instance, failed_reason;
    config.generation_failed = [&](auto const& task, std::string_view reason)
        { failed_instance = task.worker_instance; failed_reason = reason; };
    WorkerClient refused(config);
    bool unsupported = false, emitted = false;
    try { refused.contributions(18, 2, ids, [&](auto, auto, auto) { emitted = true; }); }
    catch (pool::UnsupportedResources const&) { unsupported = true; }
    require(unsupported && !emitted,
        "refused contribution batch retried its excluded shape instead of returning before output");
    require(failed_instance == scratch_limited.status().instance && failed_reason.find("scratch") != std::string::npos,
        "failed generation lost its authenticated worker or resource diagnostic");
    chunks = 0;
    refused.contributions(18, 2, {&id, 1}, [&](uint64_t ordinal, auto bytes, uint16_t members)
    { require(ordinal == chunks++ && members == 1 && !bytes.empty(), "singleton split failed after a typed batch refusal"); });
    require(chunks == PlotGroupFile::getChunkCountForK(18), "singleton split was truncated after a typed batch refusal");
    scratch_limited.stop();
}

void temporary_worker_refusal()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "refusal-ca");
    auto identity = pool_test::issue(directory.path, "refusal-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    WorkerConfig offer; offer.broker = broker.server.endpoint(); offer.listen = {"127.0.0.1", 0}; offer.tls = tls;
    offer.capacity = {1, 1, uint64_t(512) << 20, 0};
    offer.available_resources = [](uint32_t) { return pool::Resources{1, 1, uint64_t(2) << 30, 0}; };
    WorkerService service(offer); service.start();
    std::array<uint8_t, 32> id{};
    std::array<uint64_t, 3> fragments{1, 2, 3}; std::span<uint64_t const> region(fragments);
    WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
    ChunkCodec codec(18, uint64_t(1) << 20);
    auto expected = codec.encode(0, {}, 0, {&region, 1});
    pool::BrokerClient worker_control(offer.broker, tls);
    auto const instance = service.status().instance;
    {
        bool moved = false;
        WorkerClientConfig config; config.broker = offer.broker; config.tls = tls; config.task_timeout_ms = 3000;
        config.owner_id = "terminal-grant-fence";
        config.generation_path = [&](auto const&, auto const&, uint32_t) { return std::string(64, moved ? 'b' : 'a'); };
        WorkerClient client(config);
        auto const before = service.status().accepted_tasks;
        for (bool fresh : {false, true})
        {
            moved = false; auto publication = client.publication_route(18, 2, {&id, 1});
            require(bool(publication), "a registered single-slot worker did not quote its normal generation path");
            publication->selected_fresh = fresh; publication->promoted.store(true);
            // Both grants are genuine. Change the measured path between the quote and the authenticated data connection.
            moved = true; bool declined = false, received = false;
            try
            {
                if (fresh) client.contributions(18, 2, {&id, 1}, [&](auto, auto, auto) { received = true; },
                    {}, {}, {}, [&](auto, auto, auto const&) { received = true; });
                else client.generate_packed(18, 2, {&id, 1}, [&](auto const&, uint64_t) { received = true; });
            }
            catch (WorkerPublicationChanged const&) { declined = true; }
            require(declined && !received && !client.stats().input_bytes && service.status().accepted_tasks == before,
                "a changed terminal path began paid worker processing");
        }
        bool delivered = false;
        client.generate(18, 2, id, [&](auto const&, uint64_t) { delivered = true; });
        require(delivered && client.stats().completed_generation == 1 && service.status().accepted_tasks == before,
            "discarding terminal evidence prevented the following ordinary lease from finishing");
    }
    for (bool local : {false, true})
    {
        bool refuse = false, delayed_pause = false;
        WorkerClientConfig config; config.broker = offer.broker; config.tls = tls; config.task_timeout_ms = 3000;
        config.owner_id = local ? "local-refusal-owner" : "remote-refusal-owner";
        config.local_service = local ? &service : nullptr;
        // Pause after the broker grants the lease, before the worker accepts
        // it. This deterministically exercises a temporary admission refusal.
        config.selected = [&](auto const&)
        {
            if (std::exchange(refuse, false)) service.control(WorkerControl::Pause);
            // Deliver the old paused snapshot after resume and after the
            // broker grants the retry. The local worker is already accepting.
            if (std::exchange(delayed_pause, false)) worker_control.worker_admission(instance, pool::WorkerAdmission::Paused);
        };
        WorkerClient client(config);
        for (unsigned kind = 0; kind < 3; ++kind)
        {
            WorkerTaskIdentity task(pool::random_token()); bool output = false;
            auto request = [&]
            {
                if (!kind)
                {
                    std::unique_ptr<CompactMemberReceiver> receiver;
                    client.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
                    {
                        if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count));
                        receiver->append(tile);
                    }, {}, task);
                    require(receiver && receiver->finish().size(), "retried generation produced no member");
                }
                else if (kind == 1)
                {
                    uint64_t chunks = 0;
                    client.contributions(18, 2, {&id, 1}, [&](uint64_t ordinal, auto bytes, uint16_t members)
                    { require(ordinal == chunks++ && !bytes.empty() && members == 1, "retried contribution was invalid"); }, {}, task);
                    require(chunks == PlotGroupFile::getChunkCountForK(18), "retried contributions were truncated");
                }
                else require(std::ranges::equal(client.encode(18, {&chunk, 1}, uint64_t(1) << 20, {}, task).front(), expected),
                    "retried encoding changed canonical bytes");
                output = true;
            };
            refuse = true; bool rejected = false;
            try { request(); } catch (pool::TransientTransport const&) { rejected = true; }
            require(rejected && !output && !service.status().accepted_tasks,
                "paused worker did not refuse the granted task before execution");
            service.control(WorkerControl::Resume); ++task.attempt; delayed_pause = true;
            try { request(); }
            catch (std::exception const& error)
            { throw std::runtime_error("retry after temporary worker refusal failed: " + std::string(error.what())); }
            require(output, "temporary refusal excluded the only worker from the task's retry");
        }
        require(client.stats().failed_tasks == 3, "temporary refusals lost their failure accounting");
    }
    // An uploading owner can lose the direct error when the paused worker
    // closes with unread codec data. Exercise Abandon without consuming it.
    auto owner = worker_control.open_owner("unread-codec-refusal");
    pool::Writer payload(tls.max_frame_bytes); payload.u32(1); payload.u64(0); payload.u16(0); payload.bytes({});
    payload.u16(0); payload.u16(1); payload.u32(fragments.size()); payload.words(fragments);
    pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility(pool::Kind::Pack);
    task.kind = pool::Kind::Pack; task.k = 18; task.strength = 2; task.chunk_count = 1;
    task.work_units = payload.data().size(); task.max_output_bytes = 4096;
    task.resources = {1, 1, task.work_units * 20 + ChunkCodec::estimate_work(18, 0, 0), 0};
    auto grant = worker_control.acquire(owner, task); require(bool(grant), "unread refusal fixture did not obtain its grant");
    service.control(WorkerControl::Pause);
    auto channel = pool::Channel::connect(grant->endpoint, tls, grant->worker_identity);
    pool::Writer header(tls.max_frame_bytes); pool::write_lease(header, *grant); header.u32(3000);
    header.u8(uint8_t(Operation::Encode)); header.u8(18); header.u8(2); header.u64(task.work_units); header.u64(task.max_output_bytes);
    channel.send(pool::Message::Work, header.data());
    auto const reply_by = Clock::now() + std::chrono::seconds(3);
    while (!channel.readable() && Clock::now() < reply_by) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(channel.readable(), "paused worker did not signal its codec refusal");
    try { channel.send(pool::Message::Data, payload.data()); } catch (pool::TransientTransport const&) {}
    channel.close();
    try { worker_control.abandon(*grant); } catch (pool::StaleLease const&) {}
    service.control(WorkerControl::Resume); ++task.attempt;
    std::optional<pool::Lease> next; auto const retry_by = Clock::now() + std::chrono::seconds(3);
    do
    {
        next = worker_control.acquire(owner, task);
        if (!next) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (!next && Clock::now() < retry_by);
    require(next && next->worker_instance == instance, "lost codec refusal excluded the worker from the next attempt");
    worker_control.cancel(owner, task.task_id, task.attempt);
}

void expired_cleanup()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "cleanup-ca");
    auto identity = pool_test::issue(directory.path, "cleanup-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    auto options = [&]
    { pool::BrokerOptions value; value.listen = {"127.0.0.1", 0}; value.tls = tls; value.lease_ms = 10000; return value; };
    auto offer = [&]
    {
        pool::WorkerOffer value; value.instance = pool::random_token(); value.compatibility = worker_compatibility();
        value.codec_compatibility = worker_compatibility(pool::Kind::Pack);
        value.endpoint = {"127.0.0.1", 1}; value.capacity = {1, 1, uint64_t(128) << 20, 0};
        value.capabilities = pool::capability(pool::Kind::Pack); return value;
    };
    auto following = [&](pool::BrokerClient& controller, uint64_t bytes)
    {
        auto owner = controller.open_owner("following-cleanup-owner");
        pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility(pool::Kind::Pack);
        task.kind = pool::Kind::Pack; task.k = 18; task.strength = 2; task.resources = {1, 1, bytes, 0};
        task.chunk_count = 1; task.max_output_bytes = 1;
        task.work_units = pool::codec_window_header_bytes + pool::codec_chunk_header_bytes;
        std::optional<pool::Lease> lease;
        auto const end = Clock::now() + std::chrono::milliseconds(500);
        do
        {
            lease = controller.acquire(owner, task);
            if (!lease) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        while (!lease && Clock::now() < end);
        require(bool(lease), "expired compute task retained capacity until its lease timeout");
        controller.cancel(owner, task.task_id, task.attempt);
    };
    std::array<uint64_t, 3> fragments{1, 2, 3}; std::span<uint64_t const> region(fragments);
    WorkerChunkInput chunk{0, {}, 0, {}, {&region, 1}};
    {
        RunningBroker broker(options());
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "queued-cleanup-owner"; WorkerClient client(config);
        auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        // Registration would immediately grant a forgotten queued task before
        // the following owner can use this newly available capacity.
        pool::BrokerClient controller(config.broker, tls); auto worker = offer(); controller.register_worker(worker);
        following(controller, worker.capacity.host_bytes);
    }
    {
        RunningBroker broker(options());
        pool::BrokerClient controller(broker.server.endpoint(), tls); auto worker = offer(); controller.register_worker(worker);
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "declined-cleanup-owner"; WorkerClient client(config);
        auto cancel = deadline(); bool selected = false;
        WorkerTaskOptions selection;
        selection.selected = [&](auto const&)
        { selected = true; std::this_thread::sleep_until(cancel.deadline); return false; };
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel, {}, {}, selection); });
        require(selected, "declined cleanup fixture never received a grant");
        following(controller, worker.capacity.host_bytes);
    }
    {
        RunningBroker broker(options()); StalledPeer peer(tls);
        pool::BrokerClient controller(broker.server.endpoint(), tls); auto worker = offer();
        worker.endpoint = peer.listener.endpoint(); controller.register_worker(worker);
        WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
        config.owner_id = "abandoned-cleanup-owner"; WorkerClient client(config);
        auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        require(peer.accepted.load(), "abandoned cleanup fixture never contacted its worker");
        following(controller, worker.capacity.host_bytes);
    }
    {
        RunningBroker broker(options());
        WorkerConfig worker; worker.broker = broker.server.endpoint(); worker.listen = {"127.0.0.1", 0}; worker.tls = tls;
        worker.capacity = {1, 1, uint64_t(2) << 30, 0}; worker.cpu_scheduler = std::make_shared<CodecScheduler>(1);
        WorkerService service(worker); service.start();
        WorkerClientConfig config; config.broker = worker.broker; config.tls = tls;
        config.owner_id = "running-cleanup-owner"; WorkerClient client(config);
        {
            CodecScheduler::Lease pressure(worker.cpu_scheduler.get());
            auto cancel = deadline();
            bounded_failure([&] { (void)client.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); });
        }
        pool::BrokerClient controller(worker.broker, tls); following(controller, 1);
        config.owner_id = "running-direct-cleanup-owner"; config.local_service = &service;
        WorkerClient direct(config);
        std::future<void> pending; bool completed;
        {
            CodecScheduler::Lease pressure(worker.cpu_scheduler.get()); auto cancel = deadline();
            pending = std::async(std::launch::async, [&, cancel]
            { bounded_failure([&] { (void)direct.encode(18, {&chunk, 1}, uint64_t(1) << 20, cancel); }); });
            completed = pending.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
        }
        pending.get();
        require(completed, "expired direct task did not leave its shared CPU gate");
        following(controller, 1);

        // An issued grant can reach the worker with only a few milliseconds
        // left. Delay Begin's reply past that deadline on an already open route.
        auto session = controller.open_owner("short-codec-execution");
        pool::Writer payload(tls.max_frame_bytes); payload.u32(1); payload.u64(0); payload.u16(0); payload.bytes({});
        payload.u16(0); payload.u16(1); payload.u32(fragments.size()); payload.words(fragments);
        pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility(pool::Kind::Pack);
        task.kind = pool::Kind::Pack; task.k = 18; task.strength = 2; task.chunk_count = 1;
        task.work_units = payload.data().size(); task.max_output_bytes = 4096;
        task.resources = {1, 1, task.work_units * 20 + ChunkCodec::estimate_work(18, 0, 0), 0};
        auto lease = controller.acquire(session, task); require(bool(lease), "short codec task was not granted");
        auto channel = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
        require(!channel.ingress_interface(), "an authenticated same-host connection consumed a physical ingress budget");
        pool::Writer request(tls.max_frame_bytes); pool::write_lease(request, *lease);
        request.u32(25); request.u8(uint8_t(Operation::Encode)); request.u8(18); request.u8(2);
        request.u64(task.work_units); request.u64(task.max_output_bytes);
        broker.stop.store(true); broker.thread.join(); broker.stop.store(false);
        broker.thread = std::thread([&] { std::this_thread::sleep_for(std::chrono::milliseconds(55)); broker.server.serve(broker.stop); });
        channel.send(pool::Message::Work, request.data()); channel.send(pool::Message::Data, payload.data());
        bool emitted = false;
        try { emitted = channel.receive().type != pool::Message::Error; }
        catch (pool::TransientTransport const&) {} // The expired data channel may already have closed.
        require(!emitted, "delayed broker acknowledgement extended codec execution");
        following(controller, 1);
        service.stop();
    }
}

void generation_admission()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "admission-ca");
    auto identity = pool_test::issue(directory.path, "admission-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options); StalledPeer peer(tls);
    pool::WorkerOffer offer; offer.instance = pool::random_token(); offer.compatibility = worker_compatibility();
    offer.codec_compatibility = worker_compatibility(pool::Kind::Pack);
    offer.endpoint = peer.listener.endpoint(); offer.capacity = {2, 2, uint64_t(2) << 30, 0};
    offer.capabilities = pool::capability(pool::Kind::GenerateCpu);
    pool::BrokerClient controller(broker.server.endpoint(), tls); controller.register_worker(offer);
    WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = tls;
    config.owner_id = "generation-admission-owner"; config.max_active_generation = 1;
    config.cpu_scheduler = std::make_shared<CodecScheduler>(1);
    WorkerConfig local; local.capacity = {1, 1, uint64_t(2) << 30, 0}; local.cpu_scheduler = config.cpu_scheduler;
    WorkerService service(local); config.local_service = &service;
    WorkerClient client(config); std::array<uint8_t, 32> id{};
    WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::seconds(2);
    auto first = std::async(std::launch::async, [&, cancel]
    { rejects([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, cancel); }); });
    {
        std::unique_lock lock(peer.mutex);
        require(peer.ready.wait_for(lock, std::chrono::seconds(1), [&] { return peer.accepted.load(); }),
            "generation admission fixture did not reach its first worker");
    }
    auto waiting = deadline();
    bounded_failure([&] { client.generate(18, 2, id, [](auto const&, uint64_t) {}, waiting); });
    auto fallback = deadline();
    bounded_failure([&] { client.generate_local(18, 2, id, [](auto const&, uint64_t) {}, fallback); });
    require(client.stats().peak_active_generation == 1,
        "concurrent generation requests bypassed their shared in-flight ceiling");
    {
        auto cpu_deadline = deadline();
        CodecScheduler::Lease cpu(config.cpu_scheduler.get(), 1, [cpu_deadline] { cpu_deadline.check(); });
    }
    first.get();
    std::unique_ptr<CompactMemberReceiver> receiver;
    client.generate_local(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
    { if (!receiver) receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count)); receiver->append(tile); });
    require(receiver && receiver->finish().size() && !client.stats().local_tasks && !client.stats().completed_generation,
        "fallback did not release/reuse generation admission or changed broker completion counts");
}

void protected_admission()
{
    pool_test::Directory directory; auto ca = pool_test::issue(directory.path, "admission-ca");
    auto identity = pool_test::issue(directory.path, "admission-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls; RunningBroker broker(options);
    auto data_tls = tls; data_tls.max_frame_bytes = 4096;
    WorkerConfig offered; offered.broker = broker.server.endpoint(); offered.listen = {"127.0.0.1", 0}; offered.tls = data_tls;
    offered.capacity = {1, 2, uint64_t(2) << 30, 0}; offered.generation_threads = 1;
    bool startup_checked = false;
    offered.admit_start = [&](WorkerConfig const& actual)
    { startup_checked = true; require(actual.capacity == offered.capacity && worker_service_baseline_bytes(actual) < actual.capacity.host_bytes,
        "service startup was not checked against its finalized complete partition"); };
    WorkerService service(offered); require(startup_checked, "worker executor started without its embedding admission"); service.start();
    WorkerClientConfig config; config.broker = broker.server.endpoint(); config.tls = data_tls; config.owner_id = "protected-owner";
    unsigned selected = 0; bool fallback = false;
    config.admit_task = [&](WorkerTaskAdmission const& value)
    {
        ++selected; fallback = value.local_fallback;
        require(value.worker.identity == pool::certificate_identity(tls.cert_file) && value.worker.instance == service.status().instance &&
            value.task.resources.cpu_threads && value.task.resources.host_bytes && value.task.kind == pool::Kind::GenerateCpu,
            "task admission did not receive authenticated status and a concrete grant");
        throw WorkerAdmissionRejected("fixture refuses this compute grant");
    };
    auto rejected = [&](auto operation)
    { bool typed = false; try { operation(); } catch (WorkerAdmissionRejected const&) { typed = true; } require(typed, "policy rejection lost its permanent type"); };
    std::array<uint8_t, 32> id{};
    WorkerClient remote(config);
    rejected([&] { remote.generate(18, 2, id, [](auto const&, auto) {}); });
    require(selected == 1 && !fallback && !remote.stats().input_bytes && !remote.stats().output_bytes && !service.status().accepted_tasks,
        "rejected remote grant submitted Work or data to its executor");
    config.local_service = &service; WorkerClient local(config);
    rejected([&] { local.generate(18, 2, id, [](auto const&, auto) {}); });
    require(selected == 2 && !fallback && !local.stats().local_tasks && !service.status().accepted_tasks,
        "assigned local work bypassed pre-execution admission");
    rejected([&] { local.generate_local(18, 2, id, [](auto const&, auto) {}); });
    require(selected == 3 && fallback && !service.status().accepted_tasks && !service.status().active.host_bytes,
        "outage fallback allocated or executed before policy admission");
    auto denied = offered; denied.backend = "gpu";
    denied.admit_start = [](WorkerConfig const&) { throw WorkerAdmissionRejected("reject before selecting GPU executor"); };
    rejected([&] { WorkerService unreviewed(denied); });
    service.stop();
    auto unregistered_config = offered; unregistered_config.broker = {};
    WorkerService unregistered(unregistered_config); unregistered.start();
    pool::WorkerOffer stale; stale.instance = pool::random_token(); stale.compatibility = worker_compatibility();
    stale.codec_compatibility = worker_compatibility(pool::Kind::Pack);
    stale.endpoint = unregistered.endpoint(); stale.capacity = unregistered.status().capacity;
    stale.capabilities = pool::capability(pool::Kind::GenerateCpu); stale.generation_threads = 1;
    pool::BrokerClient controller(broker.server.endpoint(), tls); controller.register_worker(stale);
    auto changed = config; changed.local_service = nullptr; changed.owner_id = "stale-instance-owner";
    bool callback = false; changed.admit_task = [&](auto const&) { callback = true; };
    WorkerClient stale_client(changed);
    rejected([&] { stale_client.generate(18, 2, id, [](auto const&, auto) {}); });
    require(!callback && !stale_client.stats().input_bytes && !unregistered.status().accepted_tasks,
        "lease/status instance mismatch reached policy or executor Work");
    controller.unregister_worker(stale.instance); unregistered.stop();
}

void adaptive_resources()
{
    std::atomic<uint64_t> available{256};
    worker_detail::Budget budget(1024, 1, 16, [&] { return available.load(); });
    require(budget.task_capacity() == 256 - 16, "worker ignored its current host headroom");
    auto held = budget.acquire(64, true, {});
    available.store(32);
    require(budget.task_capacity() == held.bytes, "worker capacity dropped an active memory lease under pressure");
    available.store(16); held.release(); require(!budget.task_capacity(), "worker advertised free task memory under pressure");
    available.store(512);
    require(budget.task_capacity() == 512 - 16, "worker RAM offer did not recover after pressure cleared");
    available.store(4096);
    require(budget.task_capacity() == 1024 - 16, "worker RAM offer exceeded its explicit ceiling");
    available.store(16);
    auto waiting = std::async(std::launch::async, [&] { return budget.acquire(64, true, {}); });
    require(waiting.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
        "worker admitted an allocation without current RAM");
    available.store(512); auto resumed = waiting.get(); resumed.release();
    available.store(16); std::atomic<bool> stop{false};
    auto cancelled = std::async(std::launch::async, [&]
    { try { (void)budget.acquire(64, true, WorkerCancellation{&stop}); return false; } catch (std::exception const&) { return true; } });
    require(cancelled.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
        "worker pressure wait failed before cancellation");
    stop.store(true); require(cancelled.get(), "worker pressure wait ignored cancellation");
    std::atomic<unsigned> cpus{1};
    worker_detail::CpuAdmission cpu(4, [&](unsigned) { return cpus.load(); });
    auto fanout = std::async(std::launch::async, [&]
    { worker_detail::CpuAdmission::Lease work(cpu, 2, {}); return true; });
    require(fanout.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
        "worker CPU work exceeded current admitted lanes");
    cpus.store(4); require(fanout.get(), "worker CPU work did not recover with available lanes");
    cpus.store(1);
    { worker_detail::CpuAdmission::Lease available_cpu(cpu, 4, {}, true);
      require(available_cpu.permits() == 1 && cpu.active().first == 1,
          "CPU generation waited for its saved ceiling instead of using currently available lanes"); }
    cpus.store(0); stop.store(false);
    auto no_cpu = std::async(std::launch::async, [&]
    { try { worker_detail::CpuAdmission::Lease work(cpu, 4, WorkerCancellation{&stop}, true); return false; }
      catch (std::exception const&) { return true; } });
    require(no_cpu.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
        "worker started CPU work with no current capacity");
    stop.store(true); require(no_cpu.get(), "zero-capacity CPU wait ignored cancellation");
    WorkerConfig config; config.capacity = {4, 1, uint64_t(512) << 20, 0};
    std::atomic<uint64_t> ram{uint64_t(2) << 30}; cpus.store(4);
    config.available_resources = [&](uint32_t) { return pool::Resources{cpus.load(), 1, ram.load(), 0}; };
    WorkerService service(config); auto const high = service.status();
    ram.store(uint64_t(64) << 20); cpus.store(1); auto const low = service.status();
    require(low.capacity.host_bytes < high.capacity.host_bytes && low.capacity.cpu_threads == 1 &&
        high.hardware.partition.host_bytes == config.capacity.host_bytes && low.hardware.partition == high.hardware.partition,
        "worker confused live capacity with its stable complete service ceiling");
    ram.store(uint64_t(2) << 30); cpus.store(4); auto const recovered = service.status();
    require(recovered.capacity == high.capacity, "worker resource offer did not recover without restarting its identity");
    WorkerConfig automatic; default_worker_resources(automatic);
    require(automatic.capacity.cpu_threads == std::max(1u, profile_detail::logical_cpus()) &&
        automatic.capacity.host_bytes == profile_detail::physical_memory(),
        "automatic worker retained a fixed half-host or fixed RAM default");
}

void session_memory_ownership()
{
    pool_test::Directory directory;
    auto identity = pool::create_pool(directory.path / "session", "owned-startup-fixture");
    auto defaults = identity.defaults; defaults.discovery = false; defaults.interfaces = {"lo"}; defaults.max_brokers = 16;
    defaults.cpus = 2; defaults.host_bytes = uint64_t(1) << 30;
    pool::configure_pool(identity, defaults);
    {
        pos2gpu::GroupOptions remote; remote.pool_directory = pool_test::utf8_path(identity.directory);
        remote.max_ram = uint64_t(512) << 20; remote.workers = 2; remote.backend = "cpu"; remote.backend_set = true;
        remote.automatic_local_compute = false;
        pos2gpu::prepare_group_compute(remote);
        auto const remaining = remote.max_ram;
        require(remote.automatic_compute && !remote.pool_offer_ram && !remote.pool_offer_cpus && !remote.pool_offer_vram,
            "remote-only automatic compute inherited local offer defaults");
        pos2gpu::prepare_group_compute(remote);
        require(remote.max_ram == remaining, "re-reading automatic compute defaults charged the session twice");
    }
    pos2gpu::GroupOptions args; args.pool_directory = pool_test::utf8_path(identity.directory);
    args.max_ram = uint64_t(2) << 30; args.workers = 1; args.backend = "cpu"; args.backend_set = true;
    args.pool_offer_ram = uint64_t(512) << 20; args.pool_offer_cpus = 1; args.pool_offer_backend_set = true;
    args.pool_frame_bytes = args.window = 65536; args.pool_socket_timeout_ms = 500;
    args.protected_resources = true; args.compute_admit_session = [] {};
    pos2gpu::prepare_group_compute(args);
    pos2gpu::ComputeSession session(args);
    require(!session.owned_worker_bytes(), "a session credited worker memory before acquiring its identity");
    auto offer = pos2gpu::make_group_compute_offer(args);
    offer.admit_start = [&](WorkerConfig const& actual)
    {
        require(session.owned_worker_bytes() == actual.capacity.host_bytes &&
            session.owned_worker_bytes() > worker_service_baseline_bytes(actual),
            "identity-reserved startup omitted its total/baseline ownership");
        throw WorkerAdmissionRejected("startup fixture refuses construction");
    };
    rejects([&] { session.offer_local(offer); });
    require(!session.owned_worker_bytes() && !session.local_worker(), "failed worker startup retained a resident-memory grant");
    offer.admit_start = [&](WorkerConfig const& actual)
    { require(session.owned_worker_bytes() == actual.capacity.host_bytes, "worker startup lost its owned partition"); };
    session.offer_local(offer);
    {
        auto local = session.local_worker(); require(bool(local), "automatic worker did not reach its ready lifetime");
        auto const status = local->status();
        require(session.owned_worker_bytes() == status.capacity.host_bytes + status.baseline_host_bytes,
            "ready worker double counted or omitted its service baseline");
    }
    auto foreign_args = args;
    pos2gpu::ComputeSession foreign(foreign_args);
    foreign.offer_local(pos2gpu::make_group_compute_offer(foreign_args));
    require(!foreign.local_worker() && !foreign.owned_worker_bytes(),
        "an already-running worker identity granted its foreign pages to another session");
}

void automatic_pool_growth()
{
    pool_test::Directory directory;
    auto owner_id = pool::create_pool(directory.path / "owner", "storage");
    auto options = [&](pool::PoolIdentity const& identity)
    {
        auto defaults = identity.defaults; defaults.interfaces = {"lo"}; defaults.max_brokers = 2;
        defaults.cpus = 1; defaults.host_bytes = uint64_t(512) << 20; defaults.backend = "cpu";
        pool::configure_pool(identity, defaults);
        pos2gpu::GroupOptions args; args.pool_directory = pool_test::utf8_path(identity.directory);
        args.max_ram = uint64_t(2) << 30; args.workers = 2; args.backend = "cpu"; args.backend_set = true;
        args.pool_frame_bytes = args.window = 65536; args.pool_timeout_ms = 20000;
        pos2gpu::prepare_group_compute(args); return args;
    };
    auto offer = [](pos2gpu::GroupOptions const& args)
    {
        auto config = pos2gpu::make_group_compute_offer(args);
        config.available_resources = [capacity = config.capacity](unsigned) { return capacity; };
        return config;
    };
    auto args = options(owner_id);
    pos2gpu::ComputeSession session(args);
    WorkerClientConfig config; config.broker = pool::parse_endpoint(args.pool_broker);
    config.broker.server_name = owner_id.server_name; config.tls = owner_id.tls;
    config.owner_id = args.pool_session_owner; config.ephemeral_owner = true;
    config.generation_kind = pool::Kind::GenerateCpu; config.task_timeout_ms = 20000;
    config.local_service_provider = [&] { return session.local_worker(); };
    std::mutex mutex; std::vector<WorkerTaskObservation> deliveries; std::vector<size_t> rosters;
    config.observe = [&](auto const& value) { std::lock_guard lock(mutex); deliveries.push_back(value); };
    config.generation_capacity_changed = [&](uint8_t, uint8_t, WorkerGenerationCapacity const& value)
    { if (value.roster && value.roster->complete) { std::lock_guard lock(mutex); rosters.push_back(value.roster->routes.size()); } };
    WorkerClient client(config);
    auto empty = client.generation_capacity(18, 2);
    require(empty.roster && empty.roster->complete && empty.roster->routes.empty() && !empty.slots,
        "cold automatic owner did not expose its empty broker before the first worker arrived");
    std::promise<void> receiving, release; auto resume = release.get_future().share();
    auto generate = [&](bool hold, unsigned members = 1)
    {
        std::vector<uint64_t> data; std::array<std::array<uint8_t, 32>, 2> ids{};
        client.generate_packed(18, 2, std::span(ids).first(members), [&](PackedFragmentTile const& tile, uint64_t)
        {
            if (hold)
            {
                hold = false; receiving.set_value();
                require(resume.wait_for(std::chrono::seconds(20)) == std::future_status::ready, "late worker did not relieve the original executor");
            }
            data.insert(data.end(), tile.boundaries.begin(), tile.boundaries.end());
            data.insert(data.end(), tile.words.begin(), tile.words.end());
        });
        return data;
    };
    auto cold = std::async(std::launch::async, [&] { return generate(false); });
    pool::BrokerClient control(config.broker, config.tls);
    auto const queued_by = Clock::now() + std::chrono::seconds(5);
    while (!control.status().queued && Clock::now() < queued_by) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    require(control.status().queued == 1, "empty broker failed to retain the pending owner's request");
    session.offer_local(offer(args));
    auto baseline = cold.get();
    auto original = std::async(std::launch::async, [&] { return generate(true); });
    require(receiving.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready, "original executor never began its delivery");
    auto initial = client.generation_capacity(18, 2);
    require(initial.roster && initial.roster->complete && initial.roster->routes.size() == 1 &&
        initial.roster->routes.front().cpu_slots == 1 && !initial.roster->routes.front().gpu_slots,
        "automatic owner did not observe its original CPU executor");
    auto wait_for_roster = [&](auto matches)
    {
        auto const until = Clock::now() + std::chrono::seconds(5);
        do
        {
            auto value = client.generation_capacity(18, 2);
            if (value.roster && value.roster->complete && matches(*value.roster)) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while (Clock::now() < until);
        return false;
    };
    // Enroll and start the peer after an actual lease has delivered its
    // first output. Both normal plotting sessions advertise on loopback.
    auto invitation = directory.path / "addition.invite"; pool::invite_pool(owner_id, invitation, "addition");
    auto helper_id = pool::join_pool(invitation, directory.path / "addition");
    auto const helper_identity = pool::certificate_identity(helper_id.tls.cert_file);
    std::vector<uint64_t> remote;
    size_t receipts_before_release;
    {
        auto helper_args = options(helper_id); pos2gpu::ComputeSession helper(helper_args);
        helper.offer_local(offer(helper_args));
        remote = generate(false, 2);
        require(wait_for_roster([](auto const& roster) { return roster.routes.size() == 2; }),
            "the owner did not report both workers while the helper was present");
        std::lock_guard lock(mutex);
        require(!deliveries.empty() && deliveries.front().local_direct && deliveries.front().work_units == 1,
            "the cold member did not finish on the original worker");
        unsigned members = 0;
        for (auto const& received : std::span(deliveries).subspan(1))
        {
            require(received.worker_identity == helper_identity && !received.local_direct &&
                received.member_id_count && received.work_units == received.member_id_count,
                "a held original worker or another executor supplied the helper's members");
            members += received.member_id_count;
        }
        require(members == 2, "the authenticated helper did not deliver both members while the original worker was held");
        receipts_before_release = deliveries.size();
    }
    release.set_value(); auto local = original.get();
    require(!local.empty() && local == baseline && remote.size() == local.size() * 2 &&
        std::equal(local.begin(), local.end(), remote.begin()) && std::equal(local.begin(), local.end(), remote.begin() + local.size()),
        "newly enrolled executor changed the live owner's ordered member data");
    require(wait_for_roster([&](auto const& roster) { return roster == *initial.roster; }) && generate(false) == local,
        "the same plotting owner did not continue on its original worker after departure");
    std::lock_guard lock(mutex);
    require(deliveries.size() == receipts_before_release + 2 &&
        std::all_of(deliveries.begin() + receipts_before_release, deliveries.end(), [&](auto const& value)
        { return value.worker_identity == initial.roster->routes.front().identity && value.local_direct && value.work_units == 1; }) &&
        std::find(rosters.begin(), rosters.end(), 2) != rosters.end() && rosters.back() == 1,
        "active job did not report and delegate to the late authenticated node before its departure");
}

void codec_window_limits()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "codec-limits-ca");
    auto identity = pool_test::issue(directory.path, "codec-limits-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    auto data_tls = tls; data_tls.max_frame_bytes = 4096;
    WorkerConfig worker; worker.broker = broker.server.endpoint(); worker.listen = {"127.0.0.1", 0}; worker.tls = data_tls;
    worker.capacity = {1, 1, uint64_t(128) << 20, 0}; worker.max_window_chunks = 2; worker.max_task_bytes = 4096;
    WorkerService service(worker); service.start();
    bool selected = false;
    WorkerClientConfig config; config.broker = worker.broker; config.tls = data_tls; config.owner_id = "codec-limits-owner";
    config.max_window_chunks = 3; config.max_task_bytes = 8192;
    config.selected = [&](auto const&) { selected = true; };
    WorkerClient client(config);
    require(client.max_window_chunks() == config.max_window_chunks && client.max_task_bytes() == config.max_task_bytes,
        "client did not expose its configured codec limits");
    std::array<std::array<uint64_t, 3>, 3> raw;
    std::array<std::span<uint64_t const>, 3> regions;
    std::array<WorkerChunkInput, 3> chunks;
    auto const range = uint64_t(1) << (18 + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
    for (size_t i = 0; i < raw.size(); ++i)
    { raw[i] = {i * range + 1, i * range + 2, i * range + 3}; regions[i] = raw[i];
      chunks[i] = {i, {}, 0, {}, {&regions[i], 1}}; }
    auto refused_before_data = [&](std::span<WorkerChunkInput const> input, uint64_t output)
    {
        selected = false; auto cancel = deadline();
        bounded_failure([&] { (void)client.encode(18, input, output, cancel); });
        auto const stats = client.stats();
        require(!selected && !stats.input_bytes && !stats.output_bytes,
            "broker granted a codec window exceeding the worker's advertised limits");
    };
    refused_before_data(chunks, worker.max_task_bytes);
    std::vector<uint64_t> larger(worker.max_task_bytes / sizeof(uint64_t), 1);
    std::span<uint64_t const> large_region(larger);
    WorkerChunkInput large{0, {}, 0, {}, {&large_region, 1}};
    refused_before_data({&large, 1}, worker.max_task_bytes);
    refused_before_data({chunks.data(), 1}, config.max_task_bytes);
    auto encoded = client.encode(18, {chunks.data(), worker.max_window_chunks}, worker.max_task_bytes);
    require(selected && encoded.size() == worker.max_window_chunks,
        "a worker's smaller but compatible codec window was not usable after refusal");
    config.owner_id = "smaller-client-codec-owner"; config.max_window_chunks = 1; config.max_task_bytes = worker.max_task_bytes;
    WorkerClient smaller(config); selected = false;
    rejects([&] { (void)smaller.encode(18, {chunks.data(), worker.max_window_chunks}, worker.max_task_bytes); });
    rejects([&] { (void)smaller.encode(18, {&large, 1}, worker.max_task_bytes); });
    require(!selected && !smaller.stats().input_bytes, "client sent a window exceeding its own configured codec limit");
    service.stop();
}

void local_only_parallel()
{
    auto scheduler = std::make_shared<CodecScheduler>(1);
    WorkerConfig offer; offer.capacity = {1, 1, uint64_t(256) << 20, 0}; offer.generation_threads = 1;
    offer.max_task_bytes = uint64_t(1) << 20; offer.tls.max_frame_bytes = 4096; offer.cpu_scheduler = scheduler;
    auto const original_affinity = pos2gpu::current_thread_cpus();
    if (!original_affinity.empty()) offer.cpu_affinity = {original_affinity.front()};
    std::array<std::shared_ptr<WorkerService>, 2> services{
        std::make_shared<WorkerService>(offer), std::make_shared<WorkerService>(offer)};
    std::atomic<unsigned> next{0}, selected{0}, observed{0}, entered{0};
    std::promise<void> lanes_selected, consumers_ready;
    WorkerClientConfig config; config.local_only = true; config.max_active_generation = 2;
    config.local_service_provider = [&] { return services[next.fetch_add(1) % services.size()]; };
    config.admit_task = [&](WorkerTaskAdmission const& value)
    {
        require(value.worker.identity.empty() && value.owner_identity.empty() && value.broker_identity.empty() &&
            !value.local_fallback && value.task.resources.cpu_threads == 1 && value.task.resources.host_bytes,
            "private local admission acquired an identity or lost its concrete resource bound");
    };
    config.selected = [&](WorkerTaskObservation const& value)
    { require(value.local_direct && value.kind == pool::Kind::GenerateCpu, "private local selection lost its backend");
      if (++selected == services.size()) lanes_selected.set_value(); };
    config.observe = [&](WorkerTaskObservation const& value)
    { require(value.local_direct && value.elapsed_ns && !value.input_bytes && !value.output_bytes,
        "private local observation counted nonexistent transport traffic"); ++observed; };
    WorkerClient client(config);
    require(client.local_available() && !next.load(), "private local availability advanced its lane provider");
    require(client.suggested_batch_members(18, 2, 8) == 1 && client.suggested_generation_concurrency(18, 2, 8) == 2,
        "private local hints did not preserve singleton lane concurrency");
    auto missing = config; missing.local_service_provider = {}; rejects([&] { WorkerClient invalid(missing); });
    std::array<uint8_t, 32> id{};
    std::promise<void> release; auto finish = release.get_future().share();
    auto collect = [&]
    {
        std::unique_ptr<CompactMemberReceiver> receiver;
        client.generate(18, 2, id, [&](PackedFragmentTile const& tile, uint64_t count)
        {
            if (!receiver)
            {
                receiver = std::make_unique<CompactMemberReceiver>(18, size_t(count));
                if (++entered == services.size()) consumers_ready.set_value();
                finish.wait();
            }
            receiver->append(tile);
        });
        require(bool(receiver), "private local generation emitted no member"); return receiver->finish();
    };
    std::future<Member> first, second; bool shared_gate = false;
    {
        CodecScheduler::Lease pressure(scheduler.get());
        first = std::async(std::launch::async, collect); second = std::async(std::launch::async, collect);
        shared_gate = lanes_selected.get_future().wait_for(std::chrono::seconds(10)) == std::future_status::ready && !entered.load();
    }
    bool const parallel = consumers_ready.get_future().wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    bool bounded = true;
    for (auto const& service : services)
    {
        auto const status = service->status();
        bounded = bounded && status.accepted_tasks == 1 && status.active.slots == 1 && status.active.host_bytes &&
            status.active.host_bytes <= offer.capacity.host_bytes - status.baseline_host_bytes;
    }
    release.set_value(); auto left = first.get(), right = second.get();
    require(shared_gate, "private local lanes bypassed the shared CPU reservation or were not selected");
    require(parallel, "private local consumers could not overlap after CPU admission");
    require(bounded, "active private local consumers exceeded their RAM or slot partitions");
    auto values = [](Member const& member)
    {
        std::vector<uint64_t> result; auto view = member.view();
        for (uint64_t ordinal = 0; ordinal < PlotGroupFile::getChunkCountForK(18); ++ordinal)
            for (auto value : view.region(ordinal)) result.push_back(value);
        return result;
    };
    require(!left.empty() && values(left) == values(right), "private local parallel output disagreed between lanes");
    auto stats = client.stats();
    require(stats.completed_generation == 2 && stats.local_tasks == 2 && stats.peak_active_generation == 2 &&
        !stats.input_bytes && !stats.output_bytes && !stats.failed_tasks && observed.load() == 2,
        "private local completion or parallel admission statistics changed");
    auto view = left.view(); ChunkCodec canonical(18, uint64_t(4) << 20); uint64_t chunks = 0;
    client.contributions(18, 2, {&id, 1}, [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
    {
        auto region = view.region(ordinal); auto expected = canonical.encode(ordinal, {}, 0, {&region, 1});
        require(ordinal == chunks++ && members == 1 && std::ranges::equal(bytes, expected),
            "private local contribution changed canonical encoding");
    });
    require(chunks == PlotGroupFile::getChunkCountForK(18) && client.stats().completed_contributions == 1 &&
        pos2gpu::current_thread_cpus() == original_affinity, "private local CPU affinity leaked or contributions were truncated");
    std::future<bool> cancelled; bool completed = false;
    {
        CodecScheduler::Lease pressure(scheduler.get());
        cancelled = std::async(std::launch::async, [&]
        {
            auto const affinity = pos2gpu::current_thread_cpus(); bool rejected = false;
            WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::milliseconds(100);
            try { client.generate(18, 2, id, [](auto const&, auto) {}, cancel); }
            catch (WorkerTaskTimeout const&) { rejected = true; }
            return rejected && pos2gpu::current_thread_cpus() == affinity;
        });
        completed = cancelled.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    }
    require(cancelled.get() && completed, "private local CPU admission ignored cancellation or leaked affinity");
    rejects([&] { (void)client.encode(18, {}, 4096); });
    auto const before_pause = client.stats();
    for (auto const& service : services) service->control(WorkerControl::Pause);
    rejects([&] { client.generate(18, 2, id, [](auto const&, auto) {}); });
    for (auto const& service : services)
    { auto status = service->status(); require(!status.accepted_tasks && !status.active.slots && !status.active.host_bytes,
        "private local pause or completion retained task resources"); }
    require(client.stats().completed_generation == stats.completed_generation && client.stats().failed_tasks == before_pause.failed_tasks + 1,
        "paused private local task was counted as a completion");
}

void renewal_fence()
{
    pool_test::Directory directory;
    auto ca = pool_test::issue(directory.path, "renewal-ca");
    auto identity = pool_test::issue(directory.path, "renewal-host", &ca);
    auto tls = pool_test::tls(ca, identity); tls.timeout_ms = 3000;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    WorkerConfig worker; worker.broker = broker.server.endpoint(); worker.listen = {"127.0.0.1", 0}; worker.tls = tls;
    worker.tls.timeout_ms = 1500; worker.capacity = {1, 1, uint64_t(512) << 20, 0};
    worker.cpu_scheduler = std::make_shared<CodecScheduler>(1);
    WorkerService service(worker); service.start();
    pool::BrokerClient controller(worker.broker, tls);
    auto owner = controller.open_owner("renewal-fence-owner");
    pool::Task task; task.task_id = pool::random_token(); task.compatibility = worker_compatibility();
    task.kind = pool::Kind::GenerateCpu; task.k = 18; task.strength = 2;
    task.resources = {1, 1, worker_cpu_generation_bytes(18, 2) + boundary_bytes(18) + worker.max_task_bytes, 0};
    auto lease = controller.acquire(owner, task); require(bool(lease), "renewal fixture did not obtain its generation lease");
    {
        // Hold the actual shared compute gate so renewal loss is observed at
        // cancellable admission, independently of reference generation speed.
        CodecScheduler::Lease pressure(worker.cpu_scheduler.get());
        auto channel = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
        pool::Writer request(tls.max_frame_bytes); pool::write_lease(request, *lease); request.u32(10000);
        request.u8(uint8_t(Operation::Packed)); request.u8(18); request.u8(2); request.u16(1);
        std::array<uint8_t, 32> id{}; request.bytes(id); channel.send(pool::Message::Work, request.data());
        bool running = false; auto const until = Clock::now() + std::chrono::seconds(3);
        do
        {
            try { controller.renew(*lease); running = true; }
            catch (pool::StaleLease const&) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        } while (!running && Clock::now() < until);
        require(running, "renewal fixture never entered Running state");
        owner = controller.open_owner(owner.owner_id);
        auto frame = channel.receive();
        require(frame.type == pool::Message::Error, "fenced renewal did not return a typed worker failure");
        bool stale = false;
        try { throw_worker_error(frame.payload); } catch (pool::StaleLease const&) { stale = true; }
        require(stale, "worker renewal collapsed stale lease into a permanent cancellation");
    }
    task.task_id = pool::random_token();
    std::optional<pool::Lease> following; auto const until = Clock::now() + std::chrono::seconds(1);
    do
    {
        following = controller.acquire(owner, task);
        if (!following) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (!following && Clock::now() < until);
    require(bool(following), "fenced renewal left compute resources charged after worker completion");
    controller.cancel(owner, task.task_id, task.attempt); service.stop();
}
void status_inspection()
{
    using pos2gpu::inspect_group_compute;
    pool_test::Directory directory;
    pos2gpu::GroupOptions args; args.max_ram = uint64_t(64) << 20;
    args.pool_directory = pool_test::utf8_path(directory.path / "unconfigured");
    auto inspect = [&] { return inspect_group_compute(args, Clock::now() + std::chrono::seconds(1)); };
    auto absent = inspect();
    require(!absent.configured && absent.status == "not-configured" && !std::filesystem::exists(directory.path / "unconfigured"),
        "read-only compute inspection created enrollment for a missing setup");
    auto enrolled = pool::create_pool(directory.path / "saved-intent", "status-intent");
    auto defaults = enrolled.defaults; defaults.discovery = false; pool::configure_pool(enrolled, defaults);
    pool::save_worker_intent(enrolled, pool::WorkerIntent::Paused);
    args.pool_directory = pool_test::utf8_path(enrolled.directory);
    auto local_intent = inspect();
    require(local_intent.local_worker_intent == "paused" && !local_intent.checked && local_intent.status == "unavailable" &&
        pool::load_worker_intent(enrolled) == pool::WorkerIntent::Paused,
        "saved admission intent was reported as live capacity or changed by inspection");
    auto ca = pool_test::issue(directory.path, "status-ca");
    auto identity = pool_test::issue(directory.path, "status-host", &ca);
    auto tls = pool_test::tls(ca, identity);
    args.pool_ca = tls.ca_file; args.pool_cert = tls.cert_file; args.pool_key = tls.key_file;
    pool::BrokerOptions options; options.listen = {"127.0.0.1", 0}; options.tls = tls;
    RunningBroker broker(options);
    args.pool_broker = "127.0.0.1:" + std::to_string(broker.server.endpoint().port);
    pool::BrokerClient control(broker.server.endpoint(), tls);
    auto empty = inspect();
    require(empty.checked && empty.complete && empty.blocking && empty.status == "no-workers",
        "authenticated empty broker was reported as current compute capacity");
    pool::WorkerOffer offer; offer.instance = pool::random_token(); offer.endpoint = {"127.0.0.1", 49101};
    offer.capacity = {1, 1, uint64_t(32) << 20, 0};
    offer.capabilities = pool::capability(pool::Kind::GenerateCpu) | pool::capability(pool::Kind::Pack);
    offer.compatibility = "other-generation-contract"; offer.codec_compatibility = "other-codec-contract";
    control.register_worker(offer);
    auto incompatible = inspect();
    require(incompatible.checked && incompatible.complete && incompatible.status == "incompatible-workers" && incompatible.blocking && incompatible.incompatible == 1,
        "incompatible broker registrations appeared available to the selected build");
    control.unregister_worker(offer.instance);
    offer.compatibility = worker_compatibility(); offer.codec_compatibility = worker_compatibility(pool::Kind::Pack);
    offer.admission = pool::WorkerAdmission::Paused; control.register_worker(offer);
    auto paused = inspect();
    require(paused.status == "admission-closed" && paused.paused == 1 && !paused.accepting_generation && !paused.accepting_codec && paused.next_action.find("compute resume") != std::string::npos,
        "paused compatible workers appeared open to new work or lacked a concrete resume action");
    control.worker_admission(offer.instance, pool::WorkerAdmission::Drained);
    auto drained = inspect();
    require(drained.status == "admission-closed" && drained.drained == 1 && !drained.accepting_generation,
        "drained worker registration appeared open to new work");
    control.worker_admission(offer.instance, pool::WorkerAdmission::Accepting);
    auto accepting = inspect(); auto after = control.status();
    require(accepting.status == "compatible-workers-observed" && accepting.accepting_generation == 1 && accepting.accepting_codec == 1 &&
        !after.owners && !after.jobs, "status inspection created owner sessions/tasks or failed to report compatible admission");
    control.unregister_worker(offer.instance);
    offer.admission = pool::WorkerAdmission::Accepting;
    offer.capabilities |= pool::capability(pool::Kind::GenerateGpu);
    offer.generation_profiles = {{pool::Kind::GenerateGpu, 18, 2, 1, {1, 1, 1, 1}, 1, 1}};
    offer.gpu_recovery = "ZE_RESULT_ERROR_DEVICE_LOST";
    control.register_worker(offer);
    auto recovering = inspect();
    require(recovering.accepting_generation == 1 && recovering.accepting_codec == 1 &&
        recovering.worker_resources.size() == 1 && recovering.worker_resources.front().gpu_recovery == offer.gpu_recovery,
        "live status dropped the GPU recovery cause or withdrew accepting CPU work");
    args.pool_broker = "localhost:" + std::to_string(broker.server.endpoint().port);
    auto hostname = inspect();
    require(!hostname.complete && !hostname.checked && !hostname.issues.empty() && hostname.issues.front().reason.find("numeric") != std::string::npos,
        "bounded inspection silently used unbounded hostname resolution");
    {
        StalledPeer peer(tls); args.pool_broker = "127.0.0.1:" + std::to_string(peer.listener.endpoint().port);
        auto const began = Clock::now();
        auto stalled = inspect_group_compute(args, began + std::chrono::milliseconds(100));
        require(!stalled.checked && !stalled.complete && Clock::now() - began < std::chrono::seconds(1),
            "live compute inspection waited beyond its overall budget for an unresponsive authenticated broker");
    }
    auto untouched = control.status();
    require(!untouched.owners && !untouched.jobs, "failed inspection changed broker owner or work state");
    {
        auto member = pool::create_pool(directory.path / "unbrokered-worker", "unbrokered-status");
        auto settings = member.defaults; settings.discovery = true; settings.interfaces = {"lo"}; settings.broker.clear();
        pool::configure_pool(member, settings);
        WorkerConfig worker; worker.listen = {"::", 0}; worker.tls = member.tls;
        worker.tls.listen_interfaces = {"lo"};
        worker.tls.membership_directory = pool_test::utf8_path(member.directory);
        worker.capacity = {1, 1, uint64_t(512) << 20, 0};
        WorkerService service(worker); service.start();
        pool::DiscoveryOptions announcement; announcement.pool_id = member.pool_id; announcement.node_id = member.node_id;
        announcement.name = member.name; announcement.server_name = member.server_name;
        announcement.interfaces = {"lo"}; announcement.role = "worker"; announcement.endpoint = service.endpoint();
        pool::ComputeDiscovery advertised(announcement);
        require(advertised.status() == "LAN discovery active", "unbrokered worker fixture cannot advertise over loopback multicast");
        pos2gpu::GroupOptions observer; observer.max_ram = uint64_t(64) << 20;
        observer.pool_directory = pool_test::utf8_path(member.directory);
        auto const found = inspect_group_compute(observer, Clock::now() + std::chrono::seconds(3));
        auto const idle = service.status();
        if (!found.checked || found.status != "compatible-workers-observed" || found.workers != 1)
        {
            std::cerr << "unbrokered inspection: " << found.status << "; " << found.reason << "; discovery=" << found.discovery
                << "; workers=" << found.workers << "; brokers=" << found.brokers << '\n';
            for (auto const& issue : found.issues) std::cerr << issue.endpoint << ": " << issue.reason << '\n';
        }
        require(found.checked && found.status == "compatible-workers-observed" && !found.brokers && found.workers == 1 &&
            found.worker_resources.size() == 1 && found.worker_resources.front().identity == pool::certificate_identity(member.tls.cert_file) &&
            found.worker_resources.front().host == profile_detail::host_name() &&
            found.worker_resources.front().host_bytes == worker.capacity.host_bytes &&
            found.worker_resources.front().cpus == worker.capacity.cpu_threads &&
            !idle.accepted_tasks && !idle.known_brokers && !idle.active.cpu_threads && !idle.active.slots && !idle.active.host_bytes,
            "unbrokered discovery failed to authenticate hardware scope or started owner/task work");
    }
}
} // namespace

int main() try
{
    retry_pressure(); planning_feedback(); confirmed_generation_prior(); live_generation_routes(); network_deadlines(); independent_receivers(); assigned_local_and_fencing(); temporary_worker_refusal(); expired_cleanup(); generation_admission(); protected_admission(); adaptive_resources(); session_memory_ownership(); automatic_pool_growth(); codec_window_limits(); local_only_parallel(); renewal_fence(); status_inspection();
    std::cout << "client retry ownership, bounded cache and absolute task deadlines passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
