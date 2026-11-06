#include "host/GroupFarmComputeResources.hpp"
#include <iostream>
using namespace group_grow;
using namespace group_grow::farm;
void check(bool condition, char const* message) { if (!condition) throw std::runtime_error(message); }
template<class Function> void rejects(Function&& operation)
{ bool caught = false; try { operation(); } catch (std::exception const&) { caught = true; } check(caught, "unreviewed compute scope was admitted"); }
Demand service(std::string id, uint64_t reserved = 0)
{ Demand d; d.resource = std::move(id); d.bounded = true; d.reserved_units = reserved; if (!reserved) { d.required_units = 1; d.work_us = 1; d.period_us = 10000; d.deadline_us = 1000; } return d; }
Demand memory(std::string id, uint64_t resident, uint64_t transient = 0)
{ Demand d; d.resource = std::move(id); d.resident_units = resident; d.transient_units = transient; d.bounded = true; return d; }
int main() try
{
    Profile p; p.source = p.producer = "compute-resource-fixture";
    auto resource = [&](std::string id, std::string host, ResourceKind kind, uint64_t capacity)
    { Resource r; r.id = std::move(id); r.host = std::move(host); r.kind = kind; r.capacity = capacity; r.capacity_known = r.conditional_shared = true; p.resources.push_back(std::move(r)); };
    resource("cpu", "owner", ResourceKind::cpu, 8); resource("ram", "owner", ResourceKind::host_memory, 10000);
    resource("disk", "owner", ResourceKind::storage, 1); resource("remote-cpu", "executor", ResourceKind::cpu, 2);
    resource("remote-ram", "executor", ResourceKind::host_memory, 1000); resource("network", "physical-link", ResourceKind::network, 2);
    auto& n = p.native; n.host = "owner"; n.reader = "stateless"; n.backend = "cpu"; n.engine = "reference";
    n.k = 18; n.strength = 2; n.bounded = n.scope_complete = n.covers_future_filter = n.phase_totals_include_controller = true;
    n.max_ram_bytes = 1000; n.max_cpu_threads = 8; n.max_batch_members = n.max_group_members = 4;
    n.max_construction_us = n.max_verification_us = 100; n.output_resources = {"disk"};
    n.construction = n.verification = n.supervision = {service("cpu"), service("cpu"), memory("ram", 80), service("disk")};
    NativeCohort cohort; cohort.max_members = 4; cohort.new_farming = cohort.proof_reserve = {service("cpu"), memory("ram", 8), service("disk")}; n.cohorts = {cohort};
    n.compute_owner_identity = std::string(64, 'c'); n.compute_broker_identity = std::string(64, 'd'); n.compute_broker_host = "owner";
    n.compute_services = {service("cpu", 1), memory("ram", 256)};
    ComputeWorkerEnvelope worker; worker.tls_identity = std::string(64, 'a'); worker.host = "executor";
    worker.cpu_threads = 1; worker.slots = 2; worker.host_bytes = 512; worker.standing_host_bytes = 128;
    worker.allowed_kinds = pool::capability(pool::Kind::GenerateCpu) | pool::capability(pool::Kind::Pack);
    worker.reservation = {service("remote-cpu", 1), memory("remote-ram", 128, 384), service("network", 1)}; n.compute_workers = {worker};
    NativeRequest request; request.host = n.host; request.k = n.k; request.strength = n.strength; request.reader = n.reader;
    request.backend = n.backend; request.engine = n.engine; request.ram_bytes = 64; request.cpu_threads = 1; request.cpu_affinity_lanes = 8;
    request.batch_members = request.group_members = 1; request.construction_us = request.verification_us = 100;
    request.controller_ram_bytes = 8; request.adapter_ram_bytes = 64; request.adapter_cpu_threads = 1; request.adapter_timeout_us = 100;
    request.compute_enabled = true; request.compute_session_ram_bytes = 128; request.compute_codec_ram_bytes = 64; request.output_resources = n.output_resources;
    std::vector<Stage> phases;
    for (auto mode : {std::pair{false, false}, {true, false}, {false, true}})
    {
        Stage stage; stage.name = "compute-background"; bind_native_stage(p, request, 0, stage, mode.first, mode.second);
        check(stage.background.size() == 5 && assess_stage(p, stage).feasible, "whole service disappeared outside construction or before the first plot"); phases.push_back(stage);
    }
    auto insufficient = p; insufficient.native.compute_workers.front().reservation[1].transient_units = 1;
    rejects([&] { Stage stage; bind_native_stage(insufficient, request, 0, stage, false, false); });
    insufficient = p; insufficient.native.compute_services[1].resident_units = 1;
    rejects([&] { Stage stage; bind_native_stage(insufficient, request, 0, stage, false, false); });
    insufficient = p; insufficient.native.compute_workers.front().allowed_kinds = compute_encode;
    rejects([&] { Stage stage; bind_native_stage(insufficient, request, 0, stage, false, false); });
    insufficient = p; insufficient.native.compute_workers.front().allowed_kinds = compute_generate_cpu;
    rejects([&] { Stage stage; bind_native_stage(insufficient, request, 0, stage, false, false); });
    std::ostringstream out; write_profile_json(out, p); auto roundtrip = parse_profile(profile_detail::JsonReader(out.str()).read());
    check(roundtrip.native.compute_workers.front().reservation.size() == 3 && roundtrip.native.compute_services.size() == 2,
        "whole compute service scope was lost on profile roundtrip");
    WorkerTaskAdmission task; task.worker.identity = worker.tls_identity; task.worker.instance = "current-executor-instance";
    task.worker.capacity = {1, 2, 384, 0}; task.worker.baseline_host_bytes = 128;
    task.owner_identity = n.compute_owner_identity; task.owner_id = "actual-client-owner"; task.broker_identity = n.compute_broker_identity;
    task.task.kind = pool::Kind::GenerateCpu; task.task.k = 18; task.task.strength = 2; task.task.work_units = 1; task.task.resources = {1, 1, 128, 0};
    auto admit = [&](auto const& value) { require_compute_task(p, value, 512, 256, "cpu"); };
    admit(task);
    auto changed = task; changed.worker.identity = std::string(64, 'b'); rejects([&] { admit(changed); });
    changed = task; changed.owner_identity = std::string(64, 'b'); rejects([&] { admit(changed); });
    changed = task; changed.broker_identity = std::string(64, 'b'); rejects([&] { admit(changed); });
    changed = task; changed.worker.instance.clear(); rejects([&] { admit(changed); });
    changed = task; changed.worker.capacity.cpu_threads = 2; rejects([&] { admit(changed); });
    changed = task; changed.worker.capacity.slots = 3; rejects([&] { admit(changed); });
    changed = task; changed.worker.baseline_host_bytes = 129; rejects([&] { admit(changed); });
    changed = task; changed.worker.capacity.host_bytes = 385; rejects([&] { admit(changed); });
    changed = task; changed.task.resources.host_bytes = 385; rejects([&] { admit(changed); });
    changed = task; changed.task.kind = pool::Kind::GenerateAny; rejects([&] { admit(changed); });
    changed = task; changed.task.k = 20; rejects([&] { admit(changed); });
    changed = task; changed.task.work_units = 5; rejects([&] { admit(changed); });
    changed = task; changed.local_fallback = true; changed.broker_identity.clear(); admit(changed);
    changed.worker.identity = std::string(64, 'b'); rejects([&] { admit(changed); });
    changed = task; changed.task.kind = pool::Kind::Pack; changed.task.chunk_count = 1; changed.task.work_units = 512; changed.task.max_output_bytes = 256; admit(changed);
    changed.task.max_output_bytes = 257; rejects([&] { admit(changed); });
    changed.task.max_output_bytes = 256; changed.task.work_units = 513; rejects([&] { admit(changed); });
    auto gpu_profile = p; auto gpu_request = request; gpu_profile.native.backend = gpu_request.backend = "gpu";
    auto& gpu_worker = gpu_profile.native.compute_workers.front(); gpu_worker.allowed_kinds |= compute_generate_gpu;
    gpu_worker.gpu_identity = "host:executor/pci:0000:01:00.0"; gpu_worker.gpu_bytes = 100; gpu_worker.standing_gpu_bytes = 25;
    Resource gpu; gpu.id = "gpu"; gpu.host = "executor"; gpu.kind = ResourceKind::gpu_compute; gpu.capacity = 1; gpu.capacity_known = true;
    gpu.aliases = {gpu_worker.gpu_identity + "/compute"};
    Resource vram = gpu; vram.id = "vram"; vram.kind = ResourceKind::gpu_memory; vram.capacity = 125; vram.aliases = {gpu_worker.gpu_identity + "/memory"};
    gpu_profile.resources.insert(gpu_profile.resources.end(), {gpu, vram});
    gpu_worker.reservation.insert(gpu_worker.reservation.end(), {service("gpu", 1), memory("vram", 25, 100)});
    Stage gpu_stage; gpu_stage.name = "whole-gpu-service"; bind_native_stage(gpu_profile, gpu_request, 0, gpu_stage, false, true);
    check(assess_stage(gpu_profile, gpu_stage).feasible, "remote GPU worker acquired an unneeded local plotting GPU or lost standing residency");
    auto gpu_preparation = prepare_resource_declarations(gpu_profile);
    check(gpu_preparation.missing.empty() && std::any_of(gpu_preparation.facts.begin(), gpu_preparation.facts.end(), [](auto const& fact)
        { return fact.basis == "unknown" && fact.scope.find("Local GPU execution requires") != std::string::npos && fact.scope.find("Broker generation instead") != std::string::npos; }),
        "preparation mistook an unselected local GPU scope for a missing remote execution declaration");
    auto unbound_local_request = gpu_request; unbound_local_request.compute_enabled = false;
    rejects([&] { Stage stage; bind_native_stage(gpu_profile, unbound_local_request, 0, stage, true, false); });
    auto gpu_task = task; gpu_task.task.kind = pool::Kind::GenerateGpu; gpu_task.task.resources.gpu_bytes = 100;
    gpu_task.worker.capacity.gpu_bytes = gpu_task.worker.active.gpu_bytes = 100;
    require_compute_task(gpu_profile, gpu_task, 512, 256, "gpu");
    auto local_gpu = gpu_profile;
    auto& local_worker = local_gpu.native.compute_workers.front(); local_worker.host = "owner";
    local_worker.gpu_identity = "host:owner/pci:0000:01:00.0";
    local_worker.reservation[0].resource = "cpu"; local_worker.reservation[1].resource = "ram";
    local_gpu.resources[6].host = local_gpu.resources[7].host = "owner";
    local_gpu.resources[6].aliases = {local_worker.gpu_identity + "/compute"};
    local_gpu.resources[7].aliases = {local_worker.gpu_identity + "/memory"};
    rejects([&] { Stage stage; bind_native_stage(local_gpu, gpu_request, 0, stage, false, false); });
    local_worker.reservation[0].reserved_units = gpu_request.cpu_affinity_lanes;
    Stage local_stage; local_stage.name = "local-gpu-helpers";
    bind_native_stage(local_gpu, gpu_request, 0, local_stage, false, false);
    check(!assess_stage(local_gpu, local_stage).feasible, "local GPU runtime helpers were treated as spare CPU capacity for farming");
    gpu_profile.native.compute_workers.front().reservation.back().resident_units = 24;
    gpu_profile.native.compute_workers.front().reservation.back().transient_units = 101;
    rejects([&] { Stage stage; bind_native_stage(gpu_profile, gpu_request, 0, stage, false, false); });
    std::cout << "reviewed compute identity, whole-service phase reservations, capacities, task shapes and fallback checks passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
