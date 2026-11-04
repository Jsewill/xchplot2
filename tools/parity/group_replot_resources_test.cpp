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
#ifdef __linux__
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
