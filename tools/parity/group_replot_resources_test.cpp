#include "host/GroupFarmReplotResources.hpp"
#include <iostream>
using namespace group_grow;
using namespace group_grow::farm;
namespace fs = std::filesystem;
template<class Function> void rejects(Function action, char const* message)
{ bool caught = false; try { action(); } catch (std::exception const&) { caught = true; } require(caught, message); }
Demand service(std::string resource, uint64_t work = 100)
{ Demand d; d.resource = std::move(resource); d.required_units = 1; d.work_us = work; d.period_us = 10000000; d.deadline_us = 1000000; d.bounded = true; return d; }
Demand memory(uint64_t amount)
{ Demand d; d.resource = "ram"; d.transient_units = amount; d.bounded = true; return d; }
int main() try
{
    auto const host = profile_detail::host_name(); auto const cpus = profile_detail::logical_cpus();
    require(!host.empty() && cpus, "fixture cannot identify actual execution host/affinity");
    auto const directory = fs::temp_directory_path() / ("xchplot2-replot-resources-" + std::to_string(Clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    struct Remove { fs::path path; ~Remove() { std::error_code error; fs::remove_all(path, error); } } remove{directory};
    auto const device = storage_device(directory); auto const disk = replot_storage_id(device);
    Profile profile; profile.source = "declared-test-envelopes"; profile.producer = "fixture";
    Resource cpu; cpu.id = "cpu"; cpu.host = host; cpu.kind = ResourceKind::cpu; cpu.capacity = cpus; cpu.capacity_known = true; cpu.conditional_shared = true;
    Resource ram; ram.id = "ram"; ram.host = host; ram.kind = ResourceKind::host_memory;
    ram.capacity = profile_detail::available_memory(); ram.capacity_known = true;
    Resource storage; storage.id = disk; storage.host = host; storage.kind = ResourceKind::storage; storage.capacity = 1;
    storage.capacity_known = storage.conditional_shared = true;
    if (device.drives.empty()) storage.aliases = {"storage:unknown:" + host};
    else for (auto const& drive : device.drives) storage.aliases.push_back("drive:" + drive);
    Resource controller; controller.id = "controller"; controller.host = host; controller.kind = ResourceKind::controller;
    controller.capacity = 1; controller.capacity_known = controller.conditional_shared = true;
    profile.resources = {cpu, ram, storage, controller};
    constexpr uint64_t mib = 1024 * 1024;
    require(ram.capacity > 64 * mib, "fixture cannot admit its bounded local-memory envelope");
    Workload old; old.id = "legacy"; old.format = "chiapos-v1"; old.decoder = "declared-decoder"; old.decoder_version = "fixture";
    old.backend = "cpu"; old.pool_difficulty = "declared"; old.source = "declared-test-only-not-hardware-measurement";
    old.cost = CostKind::serialized; old.bounded = old.scope_complete = true; old.plot_count = 1;
    old.quality_burst = old.future_quality_burst = 1; old.proof_burst = old.future_proof_burst = 2;
    old.deadline_us = 1000000; old.period_us = 10000000;
    for (auto const& id : {std::string("cpu"), disk, std::string("controller")})
    { Use u; u.resource = id; u.required_units = 1; u.quality_us = 100; u.proof_us = 100; u.blocking_us = 100; old.uses.push_back(u); }
    Use held; held.resource = "ram"; held.resident_units = mib; held.transient_units = mib; old.uses.push_back(held); profile.workloads = {old};
    auto& native = profile.native; native.bounded = native.scope_complete = native.covers_future_filter = native.phase_totals_include_controller = true;
    native.host = host; native.k = 18; native.strength = 2; native.reader = "stateless"; native.backend = "cpu"; native.engine = "reference";
    native.max_ram_bytes = 16 * mib; native.max_cpu_threads = cpus; native.max_batch_members = 1; native.max_group_members = 1;
    native.max_construction_us = native.max_verification_us = 100000; native.output_resources = {disk};
#if defined(__linux__) || defined(_WIN32)
    native.verification_cpu_threads = 1;
#endif
    native.construction = {service("cpu"), memory(18 * mib), service(disk), service("controller")};
    native.verification = native.construction; native.supervision = {service("cpu"), memory(10 * mib), service(disk), service("controller")};
    native.construction[0].required_units = native.supervision[0].required_units = std::min<uint64_t>(cpus, 2);
    native.verification[0].required_units = std::min<uint64_t>(cpus, (native.verification_cpu_threads ? native.verification_cpu_threads : cpus) + 1);
    NativeCohort cohort; cohort.max_members = 2;
    cohort.new_farming = {service("cpu"), memory(mib), service(disk), service("controller")};
    cohort.proof_reserve = cohort.new_farming; native.cohorts = {cohort};
    FarmingState state; state.fixed_extent = true; state.config.k = 18; state.config.strength = 2;
    FarmingDirectory destination; destination.directory = file_io::path_text(directory); destination.filesystem = device.filesystem;
    destination.directory_identity = farming_directory_identity(directory); destination.layout.files = 2; destination.layout.total_members = 2;
    destination.layout.group_members = 1; destination.layout.fixed_members = true; destination.storage.capacity_bytes = 1024 * mib;
    state.directories = {destination};
    pos2gpu::GroupOptions args; args.backend = "cpu"; args.construction = "reference"; args.workers = 1; args.max_ram = 16 * mib; args.batch = 65535; args.compute_mode = "off";
    ReplotSupervision supervision{mib, 8 * mib, 1, 100000}; supervision.source_roots = {directory}; native.source_resources = {disk};
    auto request = replot_native_request(args, profile, state, {}, 0, 16 * mib, supervision);
    std::map<std::string, std::string> service_hosts{{"harvester", host}, {"farmer", host}, {"full_node", host}, {"solver", host}};
    struct Other { std::string node_id, peer_host; uint64_t files, members; };
    std::vector<Other> outside;
    require_target_service_scope(profile, service_hosts, outside);
    require_target_pool_difficulty(profile, 0);
    auto pooled = profile; pooled.native.min_pool_difficulty = 128; pooled.native.pool_protocol_version = 2;
    require_target_pool_difficulty(pooled, 128, 2);
    require_target_pool_difficulty(pooled, 256, 2);
    rejects([&] { require_target_pool_difficulty(pooled, 64, 2); }, "changed pool difficulty silently reused a full-solver envelope");
    rejects([&] { require_target_pool_difficulty(pooled, 128, 1); }, "changed pool protocol silently reused a full-solver envelope");
    rejects([&] { require_target_pool_difficulty(profile, 128, 2); }, "solo declaration admitted external full-solver arrivals");
    std::ostringstream pooled_json; write_profile_json(pooled_json, pooled);
    auto pooled_roundtrip = parse_profile(profile_detail::JsonReader(pooled_json.str()).read());
    require(pooled_roundtrip.native.min_pool_difficulty == 128 && pooled_roundtrip.native.pool_protocol_version == 2,
        "quantitative pool scope did not survive profile serialization");
    std::ostringstream solo_json; write_profile_json(solo_json, profile);
    require(solo_json.str().find("\"pool_protocol_version\"") == std::string::npos,
        "optional pool scope changed existing solo profile bytes");
    outside.push_back({std::string(64, 'a'), "untrusted-peer-address", 2, 200});
    rejects([&] { require_target_service_scope(profile, service_hosts, outside); }, "untracked receiver work escaped reviewed resources");
    auto external = profile;
    Resource remote_cpu = cpu; remote_cpu.id = "remote-cpu"; remote_cpu.host = "remote-receiver";
    Resource remote_ram = ram; remote_ram.id = "remote-ram"; remote_ram.host = "remote-receiver";
    Resource network = controller; network.id = "network"; network.host = "reviewed-network-path"; network.kind = ResourceKind::network;
    external.resources.insert(external.resources.end(), {remote_cpu, remote_ram, network});
    Workload unscanned = old; unscanned.id = "outside-receiver"; unscanned.plot_count = 2;
    unscanned.external_service_node_id = outside.front().node_id; unscanned.external_service_host = "remote-receiver";
    unscanned.external_service_members = 200;
    auto remote_use = old.uses.front(); remote_use.resource = remote_cpu.id; unscanned.uses.push_back(remote_use);
    auto remote_memory = held; remote_memory.resource = remote_ram.id; unscanned.uses.push_back(remote_memory);
    auto network_use = remote_use; network_use.resource = network.id; unscanned.uses.push_back(network_use);
    external.workloads.push_back(unscanned);
    require_target_service_scope(external, service_hosts, outside);
    validate_inventory_coverage(external, {{"legacy", 1}});
    rejects([&] { validate_inventory_coverage(external, {{"legacy", 1}, {"outside-receiver", 2}}); }, "unscanned receiver became retirement inventory");
    auto reduced = replot_stage(external, request, {{"outside-receiver", 0}}, 0, false, false, false);
    rejects([&] { (void)assess_stage(external, reduced); }, "retirement released unscanned shared service demand");
    auto population = outside; population.front().members = 201;
    rejects([&] { require_target_service_scope(external, service_hosts, population); }, "external member population exceeded reviewed bound");
    population = outside; population.front().files = 3;
    rejects([&] { require_target_service_scope(external, service_hosts, population); }, "external file population exceeded reviewed bound");
    auto omitted = external; omitted.workloads.back().uses.pop_back();
    rejects([&] { require_target_service_scope(omitted, service_hosts, outside); }, "remote receiver omitted shared-service network load");
    auto moved_service = service_hosts; moved_service["solver"] = "remote-receiver";
    rejects([&] { require_target_service_scope(external, moved_service, outside); }, "solver peer moved beyond its reviewed physical resource host");
    external.native.service_hosts["solver"] = "remote-receiver";
    external.native.cohorts.front().proof_reserve = {service(remote_cpu.id), memory(mib), service(network.id)};
    external.native.cohorts.front().proof_reserve[1].resource = remote_ram.id;
    require_target_service_scope(external, moved_service, outside);
    std::ostringstream external_json; write_profile_json(external_json, external);
    auto external_roundtrip = parse_profile(profile_detail::JsonReader(external_json.str()).read());
    require_target_service_scope(external_roundtrip, moved_service, outside);
    omitted = external; omitted.native.cohorts.front().proof_reserve[1].resource = "ram";
    rejects([&] { replot_stage(omitted, request, {}, 0, false, false, false); }, "remote solver CPU reused another host's memory envelope");
    require(request.cpu_threads == 1 && request.batch_members == 1 && request.controller_ram_bytes == mib, "bridge did not bind actual execution caps");
    require(!request.gpu_bytes && !request.standing_gpu_bytes, "CPU migration acquired GPU residency demand");
    require(request.group_members == 1, "bridge did not derive maximum actual reviewed file membership");
    auto larger_group = state; larger_group.directories[0].layout.files = 1; larger_group.directories[0].layout.total_members = 2;
    larger_group.directories[0].layout.group_members = 2;
    rejects([&] { replot_native_request(args, profile, larger_group, {}, 0, 16 * mib, supervision); }, "batch cap one admitted whole group two above reviewed phase bound one");
    auto execute = replot_execution_options(args, request, Clock::now() - std::chrono::milliseconds(1));
    require(execute.batch == 1 && execute.workers == 1 && execute.cancelled() && execute.protected_resources &&
        !execute.io_overlap && !execute.generation_overlap && !execute.generation_overlap_auto, "effective protected options/deadline not applied");
    require(replot_ready_members(state, 0) == 0 && replot_ready_members(state, 1) == 1 && replot_ready_members(state, 2) == 2, "ready prefix counted unpublished members");
    rejects([&] { replot_ready_members(state, 3); }, "excess ready prefix accepted");
    std::vector<Stage> stages{replot_stage(profile, request, {}, 0, false, false, false, "before"),
        replot_stage(profile, request, {}, 1, true, false, false, "overlap-build"),
        replot_stage(profile, request, {}, 1, false, true, false, "overlap-proof"),
        replot_stage(profile, request, {{"legacy", 0}}, 2, false, false, true, "final-future")};
    require(!stages[0].new_farming.empty() && !stages[0].proof_reserve.empty() && replot_ready_members(state, 0) == 0,
            "empty target skipped live service reserves or fabricated ready membership");
    auto initial_residency = profile;
    initial_residency.native.cohorts[0].new_farming[1].resident_units = mib;
    initial_residency.native.cohorts[0].proof_reserve[1].resident_units = mib;
    auto initial_stage = replot_stage(initial_residency, request, {}, 0, false, false, false, "initial-services");
    require(assess_stage(initial_residency, initial_stage).feasible && initial_stage.new_farming[1].resident_units == mib &&
            initial_stage.proof_reserve[1].resident_units == mib, "bridge omitted standing target services before first replacement");
    initial_residency.resources[1].capacity = 15 * mib;
    require(!assess_stage(initial_residency, initial_stage).feasible, "bridge admitted initial service residency beyond memory capacity");
    auto assessment = assess_transition(profile, stages);
    require(assessment.feasible && assessment.conditional_shared_service, "actual CPU/one-lane disk/controller profile rejected");
    auto exhausted = profile; exhausted.resources[2].conditional_shared = false;
    require(!assess_stage(exhausted, stages[1]).feasible, "partitioned one-lane disk silently shared");
    exhausted = profile; exhausted.workloads[0].cost = CostKind::reservation; exhausted.workloads[0].response_tail_us = 100;
    for (auto& use : exhausted.workloads[0].uses) if (use.required_units)
    { use.reserved_units = 1; use.required_units = use.quality_us = use.proof_us = use.blocking_us = 0; }
    require(!assess_stage(exhausted, stages[1]).feasible, "opaque reservation exhaustion erased");
    exhausted = profile; exhausted.native.construction[2].required_units = 2;
    require(!assess_stage(exhausted, replot_stage(exhausted, request, {}, 1, true, false, false)).feasible, "oversized storage request erased");
    exhausted = profile; exhausted.native.covers_future_filter = false;
    rejects([&] { replot_stage(exhausted, request, {}, 1, true, false, true); }, "native future declaration inferred");
    exhausted = profile; exhausted.native.supervision[1].transient_units = mib;
    rejects([&] { replot_stage(exhausted, request, {}, 0, false, false, false); }, "RPC footprint omitted");
    exhausted = profile; exhausted.native.source_resources.clear();
    rejects([&] { replot_native_request(args, exhausted, state, {}, 0, 16 * mib, supervision); }, "source filesystem mapping omitted");
    exhausted = profile; exhausted.native.supervision.erase(exhausted.native.supervision.begin() + 2);
    rejects([&] { replot_stage(exhausted, request, {}, 0, false, false, false); }, "source I/O omitted from supervision phase");
    auto offered = args; offered.pool_offer_ram = mib;
    rejects([&] { replot_native_request(offered, profile, state, {}, 0, 16 * mib, supervision); }, "uncovered broker offer admitted");
    auto moved = state; moved.directories[0].directory_identity += "changed";
    rejects([&] { replot_native_request(args, profile, moved, {}, 0, 16 * mib, supervision); }, "changed destination identity accepted");
    auto gpu_profile = profile; gpu_profile.native.backend = "gpu"; gpu_profile.native.max_gpu_bytes = 100;
    gpu_profile.native.standing_gpu_bytes = 25;
    Resource gpu; gpu.id = "gpu"; gpu.host = host; gpu.kind = ResourceKind::gpu_compute; gpu.capacity = 1; gpu.capacity_known = true;
    std::string const gpu_id = "host:" + host + "/pci:0000:01:00.0"; gpu.aliases = {gpu_id + "/compute"};
    Resource vram; vram.id = "vram"; vram.host = host; vram.kind = ResourceKind::gpu_memory; vram.capacity = 125; vram.capacity_known = true;
    vram.aliases = {gpu_id + "/memory"}; gpu_profile.resources.push_back(gpu); gpu_profile.resources.push_back(vram);
    gpu_profile.native.gpu_compute_resource = "gpu"; gpu_profile.native.gpu_memory_resource = "vram";
    for (auto* phase : {&gpu_profile.native.construction, &gpu_profile.native.verification, &gpu_profile.native.supervision})
    {
        phase->front().required_units = cpus;
        Demand held; held.resource = "vram"; held.resident_units = 25; held.bounded = true; phase->push_back(held);
    }
    gpu_profile.native.construction.back().transient_units = 100; gpu_profile.native.construction.push_back(service("gpu"));
    auto gpu_args = args; gpu_args.backend = "gpu";
    auto gpu_request = replot_native_request(gpu_args, gpu_profile, state, gpu_id, 100, 16 * mib, supervision);
    require(gpu_request.gpu_bytes == 100 && gpu_request.standing_gpu_bytes == 25, "bridge conflated allocation grant with standing GPU residency");
    std::ostringstream gpu_report; write_replot_request_json(gpu_report, gpu_request);
    require(gpu_report.str().find("\"gpu_bytes\":100,\"standing_gpu_bytes\":25") != std::string::npos, "request report omitted distinct standing GPU residency");
    auto final_gpu = replot_stage(gpu_profile, gpu_request, {{"legacy", 0}}, 2, false, false, true, "gpu-final-future");
    require(assess_stage(gpu_profile, final_gpu).feasible && final_gpu.supervision.back().resident_units == 25,
            "bridge released standing GPU residency at final future stage");
    auto no_gpu_standing = gpu_profile; no_gpu_standing.native.standing_gpu_bytes = 0;
    rejects([&] { replot_native_request(gpu_args, no_gpu_standing, state, gpu_id, 100, 16 * mib, supervision); }, "bridge admitted unknown standing GPU residency");
    std::cout << "migration physical mapping, phase budgets, one-lane coexistence and deadline checks passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
