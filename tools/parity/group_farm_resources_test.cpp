#include "host/GroupFarmResources.hpp"
#include <iostream>

using namespace group_grow::farm;
void check(bool value, char const* message) { if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function action, char const* message)
{
    bool rejected = false;
    try { action(); } catch (std::runtime_error const&) { rejected = true; }
    check(rejected, message);
}
bool reason(Assessment const& result, std::string const& prefix)
{ return std::any_of(result.reasons.begin(), result.reasons.end(), [&](std::string const& value) { return value.find(prefix) != std::string::npos; }); }
Resource resource(std::string id, ResourceKind kind, uint64_t capacity)
{ Resource r; r.id = std::move(id); r.host = "harvester"; r.kind = kind; r.capacity = capacity; r.capacity_known = true; return r; }
Demand service(std::string id, uint64_t units = 1, uint64_t work = 20)
{ Demand d; d.resource = std::move(id); d.required_units = units; d.work_us = work; d.deadline_us = 10000; d.period_us = 20000; d.bounded = true; return d; }
Demand memory(std::string id, uint64_t bytes)
{ Demand d; d.resource = std::move(id); d.transient_units = bytes; d.bounded = true; return d; }
Profile fixture()
{
    Profile p; p.source = "explicit-worst-case-reservation"; p.producer = "operator"; p.captured_at = "provenance-only";
    p.resources = {resource("cpu", ResourceKind::cpu, 16), resource("ram", ResourceKind::host_memory, 100000),
                   resource("gpu", ResourceKind::gpu_compute, 1), resource("vram", ResourceKind::gpu_memory, 100000),
                   resource("disk", ResourceKind::storage, 16), resource("controller", ResourceKind::controller, 16),
                   resource("network", ResourceKind::network, 4), resource("solver", ResourceKind::solver, 4)};
    Workload w; w.id = "original-pos1"; w.format = "chiapos-v1-c0"; w.decoder = "chiapos"; w.decoder_version = "explicit-version";
    w.backend = "cpu"; w.pool_difficulty = "explicit-current-difficulty"; w.source = "declared-complete-reservation";
    w.cost = CostKind::reservation; w.bounded = w.scope_complete = true; w.plot_count = 10;
    w.quality_burst = w.future_quality_burst = 4; w.proof_burst = w.future_proof_burst = 12;
    w.period_us = 20000; w.deadline_us = 10000; w.response_tail_us = 1000;
    Use cpu; cpu.resource = "cpu"; cpu.reserved_units = 2;
    Use ram; ram.resource = "ram"; ram.resident_units = 1000; ram.transient_units = 100;
    Use disk; disk.resource = "disk"; disk.reserved_units = 1;
    w.uses = {cpu, ram, disk}; p.workloads = {w};
    auto& n = p.native; n.bounded = n.scope_complete = n.covers_future_filter = n.phase_totals_include_controller = true; n.k = 18; n.strength = 2;
    n.host = "harvester"; n.reader = "stateless"; n.backend = "cpu"; n.engine = "reference";
    n.max_ram_bytes = 4096; n.max_cpu_threads = 4; n.max_batch_members = 2; n.max_group_members = 2;
    n.max_construction_us = n.max_verification_us = 1000; n.output_resources = {"disk"};
    n.construction = {service("cpu", 3, 100), memory("ram", 1200), service("disk", 1, 100)};
    n.verification = {service("cpu", 3, 100), memory("ram", 1200), service("disk", 1, 100)};
    n.supervision = {service("cpu", 2, 100), memory("ram", 500), service("disk", 1, 100)};
    NativeCohort cohort; cohort.max_members = 10;
    cohort.new_farming = {service("cpu"), memory("ram", 100), service("disk")};
    cohort.proof_reserve = {service("cpu", 1, 40), memory("ram", 100), service("disk", 1, 40)};
    n.cohorts = {cohort}; return p;
}
NativeRequest request()
{
    NativeRequest r; r.host = "harvester"; r.k = 18; r.strength = 2; r.reader = "stateless"; r.backend = "cpu"; r.engine = "reference";
    r.ram_bytes = 1000; r.cpu_threads = 2; r.batch_members = 1; r.group_members = 2; r.construction_us = r.verification_us = 100;
    r.cpu_affinity_lanes = 16;
    r.controller_ram_bytes = 100; r.adapter_ram_bytes = 200; r.adapter_cpu_threads = 1; r.adapter_timeout_us = 1000;
    r.output_resources = {"disk"}; return r;
}
Stage initial()
{ Stage s; s.name = "initial"; s.scope_complete = true; return s; }
void modeled_gpu(Profile& p)
{
    auto& w = p.workloads[0]; w.id = "gigahorse"; w.format = "gigahorse-c-specific"; w.decoder = "gigahorse"; w.backend = "cuda";
    w.cost = CostKind::serialized; w.response_tail_us = 0; w.uses.clear();
    Use gpu; gpu.resource = "gpu"; gpu.required_units = 1; gpu.quality_us = 10; gpu.proof_us = 20; gpu.blocking_us = 30;
    Use ram; ram.resource = "ram"; ram.resident_units = 1000; ram.transient_units = 100;
    Use vram; vram.resource = "vram"; vram.resident_units = 5000; vram.transient_units = 500;
    w.uses = {gpu, ram, vram};
}
Telemetry sample(uint64_t epoch, uint64_t sequence, uint64_t tick, uint64_t available, bool late = false)
{
    Telemetry t; t.epoch = epoch; t.sequence = sequence; t.producer_tick_ms = tick; t.producer = "current-local-source-v1"; t.bounded_source_age_ms = 0;
    ResourceSample s; s.resource = "ram"; s.available_known = true; s.available_units = available; s.missed_deadline = late;
    t.resources = {s}; return t;
}
resource_detail::CpuObservation cpu_observation(uint64_t ms, uint64_t total, uint64_t idle, uint64_t own_ns)
{
    resource_detail::CpuObservation value; value.at = Clock::time_point(std::chrono::milliseconds(ms));
    value.total = total; value.idle = idle; value.scope = "cpu0"; value.units_per_second = 100; value.processors = 1;
    value.before_wall = value.at + std::chrono::milliseconds(1); value.after_wall = value.at + std::chrono::milliseconds(2);
    value.own_before_ns = own_ns; value.own_after_ns = own_ns + 1000000; value.own_scope_known = true; return value;
}
void guided_resource_settings()
{
    auto declared = fixture();
    auto generated = make_profile_template({declared.resources[0], declared.resources[1], declared.resources[4]},
        {{"original-pos1", 10}}, "observed", "fixture");
    generated.native.output_resources = {"disk"};
    auto original = generated; std::vector<std::string> settings;
    for (auto const& resource : generated.resources)
        settings.push_back("resources[" + resource.id + "].capacity=" + std::to_string(resource.capacity) + ",capacity_known=true");
    settings.push_back("workloads[original-pos1].format=chiapos-v1-c0,decoder=chiapos,decoder_version=fixture,backend=cpu,pool_difficulty=solo,source=declared-fixture,cost=serialized,bounded=true,scope_complete=true,quality_burst=4,proof_burst=12,future_quality_burst=4,future_proof_burst=12,period_us=20000,deadline_us=10000");
    for (auto const* resource : {"cpu", "disk"}) settings.push_back("workloads[original-pos1].uses[" + std::string(resource) +
        "].required_units=1,quality_us=10,proof_us=20,blocking_us=30");
    settings.push_back("workloads[original-pos1].uses[ram].resident_units=1000,transient_units=100");
    settings.push_back("native.bounded=true,scope_complete=true,covers_future_filter=true,phase_totals_include_controller=true,k=18,strength=2,host=harvester,reader=stateless,backend=cpu,engine=reference,max_ram_bytes=4096,max_cpu_threads=4,max_batch_members=2,max_group_members=2,max_construction_us=1000,max_verification_us=1000");
    auto add_demands = [&](std::string const& target, std::vector<Demand> const& demands)
    {
        for (auto const& demand : demands) settings.push_back(target + '[' + demand.resource + "].resident_units=" +
            std::to_string(demand.resident_units) + ",transient_units=" + std::to_string(demand.transient_units) +
            ",required_units=" + std::to_string(demand.required_units) + ",work_us=" + std::to_string(demand.work_us) +
            ",period_us=" + std::to_string(demand.period_us) + ",deadline_us=" + std::to_string(demand.deadline_us) + ",bounded=true");
    };
    add_demands("native.construction", declared.native.construction);
    add_demands("native.verification", declared.native.verification);
    add_demands("native.supervision", declared.native.supervision);
    add_demands("native.cohorts[10].new_farming", declared.native.cohorts[0].new_farming);
    add_demands("native.cohorts[10].proof_reserve", declared.native.cohorts[0].proof_reserve);
    auto partial = apply_resource_settings(generated, std::span<std::string const>(settings).first(1));
    auto remaining = prepare_resource_declarations(partial);
    check(!remaining.missing.empty() && !remaining.resource_settings.empty() && !assess_stage(partial, initial()).feasible &&
        !generated.resources[0].capacity_known && partial.resources[0].capacity_known,
        "partial settings lost declarations, changed their source or admitted incomplete bounds");
    check(std::none_of(remaining.resource_settings.begin(), remaining.resource_settings.end(), [](auto const& setting)
        { return setting.starts_with("resources[cpu]."); }), "setup requested an already declared physical capacity again");
    auto aliases = generated; auto cpu_alias = aliases.resources.front(); cpu_alias.id = "same-host-cpu"; aliases.resources.push_back(cpu_alias);
    auto alias_settings = apply_resource_settings(aliases, std::vector<std::string>{"resources[same-host-cpu].capacity=3,capacity_known=true"});
    check(!alias_settings.resources.front().capacity_known && alias_settings.resources.front().capacity == aliases.resources.front().capacity &&
        alias_settings.resources.back().capacity_known && alias_settings.resources.back().capacity == 3 && !assess_stage(alias_settings, initial()).feasible,
        "a stable resource ID changed a different alias declaration or multiplied shared capacity");
    auto completed = apply_resource_settings(generated, settings); auto stage = initial();
    bind_native_stage(completed, request(), 0, stage, true, true);
    auto preparation = prepare_resource_declarations(completed);
    check(assess_stage(completed, stage).feasible && preparation.missing.empty() && preparation.resource_settings.empty(),
        "stable-ID declarations cannot complete the established CPU migration envelope");
    std::ostringstream text; write_profile_json(text, completed); std::istringstream input(text.str());
    auto reused = read_profile(input); stage = initial(); bind_native_stage(reused, request(), 0, stage, true, true);
    check(assess_stage(reused, stage).feasible, "saved guided declarations did not survive profile reuse");
    for (auto const* invalid : {"resources[invented].capacity=99,capacity_known=true", "resources[cpu].capacity=1,capcity_known=true",
        "resources[cpu].capacity_known=yes", "resources[cpu].capacity=-1", "resources[cpu].capacity=18446744073709551616",
        "workloads[original-pos1].plot_count=0", "workloads[invented].cost=reservation", "native.executable=/tmp/reader",
        "native.construction[invented].bounded=true", "native.cohorts[0].proof_reserve[cpu].bounded=true", "resources[0].capacity=16"})
        rejects([&] { apply_resource_settings(generated, std::vector<std::string>{invalid}); }, "unsafe, mistyped or unbounded resource setting was accepted");
    std::ostringstream before, after; write_profile_json(before, original); write_profile_json(after, generated);
    check(before.str() == after.str(), "a rejected or successful setup mutated its source declarations");
}
int main()
{
    try
    {
        guided_resource_settings();
        auto p = fixture(); auto s = initial();
        validate_inventory_coverage(p, {{"original-pos1", 4}, {"original-pos1", 6}});
        rejects([&] { validate_inventory_coverage(p, {{"original-pos1", 9}}); }, "outside inventory omitted");
        rejects([&] { validate_inventory_coverage(p, {{"", 10}}); }, "blank inventory workload accepted");
        check(assess_stage(p, s).feasible, "declared reservation rejected");
        s.remaining = {{"original-pos1", 0}};
        auto retired = assess_stage(p, s);
        check(retired.margins[0].reserved_units == 2 && retired.margins[1].resident_units == 1000,
              "plot removal released resident foreign service capacity");
        auto unknown = p; unknown.workloads[0].cost = CostKind::unknown;
        check(reason(assess_stage(unknown, initial()), "unknown-cost"), "unknown legacy cost accepted");
        unknown.resources[0].capacity_known = false;
        check(reason(assess_stage(unknown, initial()), "unknown-capacity"), "unknown capacity accepted");

        auto alias = p; auto disk_alias = alias.resources[4]; disk_alias.id = "disk-alias"; disk_alias.aliases = {"disk"}; disk_alias.capacity = 2;
        alias.resources.push_back(disk_alias); s = initial(); s.construction = {service("disk-alias", 2)};
        check(reason(assess_stage(alias, s), "capacity-exceeded:disk"), "physical alias offered twice");
        alias.resources.back().host = "other-host";
        rejects([&] { assess_stage(alias, s); }, "alias across hosts accepted");
        auto duplicate_host = p; auto cpu_alias = duplicate_host.resources[0]; cpu_alias.id = "same-host-cpu-label"; cpu_alias.capacity = 2;
        duplicate_host.resources.push_back(cpu_alias); s = initial(); s.construction = {service("same-host-cpu-label", 1)};
        check(reason(assess_stage(duplicate_host, s), "capacity-exceeded:cpu"), "same host CPU labels multiplied capacity");
        auto ram_alias = duplicate_host.resources[1]; ram_alias.id = "same-host-memory-label"; ram_alias.capacity = 1000; duplicate_host.resources.push_back(ram_alias);
        check(reason(assess_stage(duplicate_host, s), "capacity-exceeded:ram"), "same host RAM labels multiplied capacity");

        s = initial(); bind_native_stage(p, request(), 0, s, true, true);
        check(!s.new_farming.empty() && !s.proof_reserve.empty(), "empty target omitted live farming/solver reserve floor");
        check(assess_stage(p, s).feasible, "native bound rejected");
        auto initial_services = p;
        initial_services.native.cohorts[0].new_farming[1].resident_units = 200;
        initial_services.native.cohorts[0].proof_reserve[1].resident_units = 300;
        { auto stage = initial(); bind_native_stage(initial_services, request(), 0, stage, false, false);
          auto const assessment = assess_stage(initial_services, stage);
          check(assessment.feasible && assessment.margins[1].resident_units == 1500,
                "empty target omitted resident target services or existing legacy residency"); }
        initial_services.resources[1].capacity = 2200;
        { auto stage = initial(); bind_native_stage(initial_services, request(), 0, stage, false, false);
          check(reason(assess_stage(initial_services, stage), "capacity-exceeded:ram"), "initial standing target load admitted beyond memory capacity"); }
        auto missing_initial = p; missing_initial.native.cohorts.clear();
        rejects([&] { auto stage = initial(); bind_native_stage(missing_initial, request(), 0, stage, false, false); }, "empty target admitted without a complete first cohort");
        missing_initial = p; missing_initial.native.cohorts[0].proof_reserve = {memory("ram", 10)};
        rejects([&] { auto stage = initial(); bind_native_stage(missing_initial, request(), 0, stage, false, false); }, "empty target skipped full-proof envelope coverage validation");
        bind_native_stage(p, request(), 1, s, true, true);
        check(!s.new_farming.empty() && !s.proof_reserve.empty() && assess_stage(p, s).feasible, "native cohort not admitted");
        auto missing = p; missing.native.verification.erase(missing.native.verification.begin());
        rejects([&] { auto stage = initial(); bind_native_stage(missing, request(), 1, stage, true, true); }, "construction masked missing verifier CPU");
        missing = p; missing.native.construction.erase(missing.native.construction.begin() + 1);
        rejects([&] { auto stage = initial(); bind_native_stage(missing, request(), 1, stage, true, true); }, "verification masked missing constructor RAM");
        missing = p; missing.native.cohorts[0].proof_reserve = {memory("ram", 10)};
        rejects([&] { auto stage = initial(); bind_native_stage(missing, request(), 1, stage, true, true); }, "memory-only proof reserve accepted");
        missing = p; missing.resources[1].host = "remote-solver";
        rejects([&] { auto stage = initial(); bind_native_stage(missing, request(), 1, stage, true, true); }, "remote RAM masked local construction footprint");
        missing = p; missing.resources.push_back(resource("separate-source-disk", ResourceKind::storage, 1));
        missing.native.source_resources = {"separate-source-disk"}; auto source_request = request(); source_request.source_resources = {"separate-source-disk"};
        rejects([&] { auto stage = initial(); bind_native_stage(missing, source_request, 0, stage, false, false); }, "separate source disk omitted from supervisor costs");
        missing = p; missing.native.cohorts[0].new_farming = {service("cpu", 0, 0), memory("ram", 0), service("disk", 0, 0)};
        rejects([&] { auto stage = initial(); bind_native_stage(missing, request(), 1, stage, false, false); }, "zero native demand established coverage");
        auto changed = request(); changed.backend = "cuda";
        rejects([&] { auto stage = initial(); bind_native_stage(p, changed, 1, stage, true, false); }, "backend scope mismatch accepted");
        changed = request(); changed.ram_bytes = 10000;
        rejects([&] { auto stage = initial(); bind_native_stage(p, changed, 1, stage, true, false); }, "RAM envelope exceeded");
        auto one_cpu = p; one_cpu.workloads.clear(); one_cpu.resources[0].capacity = 1; one_cpu.resources[0].conditional_shared = true;
        one_cpu.native.max_cpu_threads = 1;
        one_cpu.native.construction[0].required_units = one_cpu.native.verification[0].required_units = one_cpu.native.supervision[0].required_units = 1;
        auto one_request = request(); one_request.cpu_threads = one_request.cpu_affinity_lanes = 1;
        { auto stage = initial(); bind_native_stage(one_cpu, one_request, 1, stage, true, false);
          check(assess_stage(one_cpu, stage).feasible, "one CPU affinity union was double charged for its monitor"); }
        one_request.cpu_affinity_lanes = 2;
        rejects([&] { auto stage = initial(); bind_native_stage(one_cpu, one_request, 1, stage, true, false); }, "independent monitor CPU lane omitted");
        missing = p; missing.native.max_batch_members = 1; missing.native.max_group_members = 1;
        rejects([&] { auto stage = initial(); bind_native_stage(missing, request(), 0, stage, true, false); }, "one-member batch hid a two-member whole construction phase");
        rejects([&] { auto stage = initial(); bind_native_stage(missing, request(), 0, stage, false, true); }, "one-member batch hid a two-member whole verification phase");
        auto bound_gpu = p; bound_gpu.workloads.clear(); bound_gpu.native.backend = "gpu"; bound_gpu.native.max_gpu_bytes = 100;
        bound_gpu.resources[0].conditional_shared = true;
        bound_gpu.native.standing_gpu_bytes = 25;
        bound_gpu.native.gpu_compute_resource = "gpu"; bound_gpu.native.gpu_memory_resource = "vram";
        bound_gpu.resources[2].aliases = {"physical-device-uuid/compute"}; bound_gpu.resources[3].aliases = {"physical-device-uuid/memory"};
        bound_gpu.native.construction.push_back(service("gpu")); bound_gpu.native.construction.push_back(memory("vram", 100));
        bound_gpu.native.construction.back().resident_units = 25;
        auto standing = memory("vram", 0); standing.resident_units = 25;
        bound_gpu.native.verification.push_back(standing); bound_gpu.native.supervision.push_back(standing);
        auto gpu_request = request(); gpu_request.backend = "gpu"; gpu_request.gpu_bytes = 100;
        gpu_request.standing_gpu_bytes = 25;
        rejects([&] { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, true, false); }, "unverified GPU identity accepted");
        gpu_request.gpu_binding_verified = true; gpu_request.gpu_identity = "different-physical-device";
        rejects([&] { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, true, false); }, "same-size wrong physical GPU accepted");
        gpu_request.gpu_identity = "physical-device-uuid";
        rejects([&] { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, true, false); }, "GPU runtime CPU demand reduced to scoped host workers");
        rejects([&] { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, false, true); }, "GPU retained runtime CPU omitted during proof verification");
        rejects([&] { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, false, false); }, "GPU retained runtime CPU omitted during supervision");
        bound_gpu.native.construction[0].required_units = bound_gpu.native.verification[0].required_units =
            bound_gpu.native.supervision[0].required_units = gpu_request.cpu_affinity_lanes;
        { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, true, false);
          auto assessed = assess_stage(bound_gpu, stage);
          check(stage.scope_complete && assessed.feasible && assessed.margins[3].resident_units == 25 && assessed.margins[3].transient_units == 100,
                "GPU standing residency missing or charged twice with allocation peak"); }
        { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, false, true); check(stage.scope_complete && assess_stage(bound_gpu, stage).feasible, "GPU whole-affinity proof scope rejected"); }
        { auto stage = initial(); stage.future_filter = true; bind_native_stage(bound_gpu, gpu_request, 0, stage, false, false);
          check(stage.scope_complete && assess_stage(bound_gpu, stage).feasible && stage.supervision.back().resident_units == 25,
                "GPU standing residency missing from final future supervision"); }
        auto missing_standing = bound_gpu; missing_standing.native.standing_gpu_bytes = 0;
        auto no_standing_request = gpu_request; no_standing_request.standing_gpu_bytes = 0;
        rejects([&] { auto stage = initial(); bind_native_stage(missing_standing, no_standing_request, 0, stage, true, false); }, "zero GPU standing residency accepted");
        auto mismatched_standing = gpu_request; mismatched_standing.standing_gpu_bytes = 24;
        rejects([&] { auto stage = initial(); bind_native_stage(bound_gpu, mismatched_standing, 0, stage, true, false); }, "standing GPU request differs from reviewed bound");
        std::ostringstream gpu_encoded; write_profile_json(gpu_encoded, bound_gpu);
        auto without_standing = gpu_encoded.str(); auto const standing_field = without_standing.find(",\"standing_gpu_bytes\":25");
        check(standing_field != std::string::npos, "GPU standing residency omitted from profile JSON");
        without_standing.erase(standing_field, std::string(",\"standing_gpu_bytes\":25").size());
        std::istringstream missing_standing_json(without_standing); auto missing_standing_profile = read_profile(missing_standing_json);
        rejects([&] { auto stage = initial(); bind_native_stage(missing_standing_profile, no_standing_request, 0, stage, false, false); }, "missing GPU standing residency accepted");
        std::istringstream gpu_input(gpu_encoded.str()); auto gpu_roundtrip = read_profile(gpu_input);
        check(gpu_roundtrip.native.standing_gpu_bytes == 25, "standing GPU residency lost on JSON round trip");
        auto insufficient_standing = bound_gpu; insufficient_standing.native.construction.back().transient_units = 99;
        rejects([&] { auto stage = initial(); bind_native_stage(insufficient_standing, gpu_request, 0, stage, true, false); }, "construction allocation consumed standing GPU reservation");
        insufficient_standing = bound_gpu; insufficient_standing.native.verification.back().resident_units = 24;
        insufficient_standing.native.verification.back().transient_units = 1;
        rejects([&] { auto stage = initial(); bind_native_stage(insufficient_standing, gpu_request, 0, stage, false, true); }, "transient verifier memory masked insufficient standing GPU residency");
        insufficient_standing = bound_gpu; insufficient_standing.native.supervision.pop_back();
        rejects([&] { auto stage = initial(); bind_native_stage(insufficient_standing, gpu_request, 0, stage, false, false); }, "supervision omitted standing GPU residency");
        bound_gpu.resources.push_back(resource("other-vram", ResourceKind::gpu_memory, 100000));
        auto wrong_standing = bound_gpu; wrong_standing.native.verification.back().resource = "other-vram";
        rejects([&] { auto stage = initial(); bind_native_stage(wrong_standing, gpu_request, 0, stage, false, true); }, "other GPU memory masked retained selected context during verification");
        wrong_standing = bound_gpu; wrong_standing.native.supervision.back().resource = "other-vram";
        rejects([&] { auto stage = initial(); bind_native_stage(wrong_standing, gpu_request, 0, stage, false, false); }, "other GPU memory masked retained selected context during supervision");
        bound_gpu.native.construction.back().resource = "other-vram";
        rejects([&] { auto stage = initial(); bind_native_stage(bound_gpu, gpu_request, 0, stage, true, false); }, "other GPU memory masked selected VRAM");
        auto cohort = p.native.cohorts[0]; cohort.max_members = 20; cohort.new_farming[1].transient_units = 200;
        p.native.cohorts.push_back(cohort); bind_native_stage(p, request(), 11, s, false, false);
        check(s.new_farming[1].transient_units == 200, "wrong native cohort selected");

        auto gpu = fixture(); modeled_gpu(gpu); auto gs = initial(); gs.construction = {service("gpu", 1, 100)};
        check(reason(assess_stage(gpu, gs), "shared-gpu-not-qualified"), "shared GPU accepted by default");
        gpu.resources[2].conditional_shared = true;
        auto shared = assess_stage(gpu, gs);
        check(shared.feasible && shared.conditional_shared_gpu && shared.margins[2].drain_us == 410, "serialized GPU burst/proof bound incorrect");
        gs.construction[0].required_units = 2;
        check(reason(assess_stage(gpu, gs), "request-shape-exceeds-capacity"), "qualified shared GPU erased request shape");
        gs.construction[0].required_units = 1; gs.construction[0].blocking_us = 50;
        check(assess_stage(gpu, gs).margins[2].blocking_us == 80, "independent carried-in blocking was not summed");
        auto native_gpu = fixture(); native_gpu.workloads.clear(); native_gpu.resources[2].capacity = 3;
        auto native_gpu_stage = initial(); native_gpu_stage.new_farming = {service("gpu")};
        native_gpu_stage.proof_reserve = {service("gpu")}; native_gpu_stage.construction = {service("gpu")};
        check(reason(assess_stage(native_gpu, native_gpu_stage), "shared-gpu-not-qualified"), "new native farmer bypassed protected GPU role");
        gs.remaining = {{"gigahorse", 1}};
        check(assess_stage(gpu, gs).margins[2].serialized_work_us == 380, "partial phaseout scaled active proof/lookup load");
        gs.remaining[0].count = 0;
        auto last = assess_stage(gpu, gs);
        check(last.margins[3].resident_units == 5000 && last.margins[3].transient_units == 500 &&
              last.margins[2].serialized_work_us == 100 && last.margins[2].blocking_us == 80, "retirement released foreign context or pending-queue headroom");

        auto distributed = fixture(); distributed.resources[7].host = "remote-gpu-solver";
        Workload dr = distributed.workloads.front(); dr.id = "drplotter"; dr.format = "drplot-specific";
        dr.decoder = "drsolver"; dr.decoder_version = "declared-version"; dr.backend = "remote-gpu"; dr.cost = CostKind::serialized; dr.response_tail_us = 0;
        Use remote; remote.resource = "solver"; remote.required_units = 3; remote.quality_us = 20; remote.proof_us = 20;
        Use network; network.resource = "network"; network.required_units = 1; network.quality_us = 5; network.proof_us = 5;
        dr.uses = {remote, network}; distributed.workloads.push_back(dr);
        auto bladebit = distributed.workloads.front(); bladebit.id = "bladebit"; bladebit.format = "bladebit-compression-specific"; bladebit.decoder = "bladebit-reader";
        distributed.workloads.push_back(bladebit);
        check(assess_stage(distributed, initial()).feasible, "mixed formats on distributed declared resources rejected");
        dr.id = "second-dr-harvester"; distributed.workloads.push_back(dr);
        check(reason(assess_stage(distributed, initial()), "capacity-exceeded:solver"), "two harvesters double-counted one remote solver capacity");

        auto stages_profile = fixture(); auto start = initial(), peak = initial(), end = initial();
        start.name = "start"; peak.name = "coexistence"; end.name = "future-end"; end.future_filter = true;
        peak.construction = {memory("ram", 100001)};
        auto transition = assess_transition(stages_profile, {start, peak, end});
        check(!transition.feasible && transition.per_dimension_worst && transition.margins[1].remaining_units == 0 &&
              transition.margins[1].worst_stages.at("transient_units") == "coexistence", "intermediate coexistence peak omitted");
        AssessmentAccumulator streaming(stages_profile); streaming.add(start); streaming.add(peak); streaming.add(end);
        std::ostringstream vector_report, stream_report; write_assessment_json(vector_report, transition); write_assessment_json(stream_report, streaming.finish());
        check(vector_report.str() == stream_report.str(), "streaming assessment changed exact peak or stage provenance");
        AssessmentAccumulator repeated(stages_profile);
        for (unsigned i = 0; i < 10000; ++i) { auto stage = start; stage.name = "batch-" + std::to_string(i); stage.remaining = {{"original-pos1", i % 9 + 1}}; repeated.add(stage); }
        repeated.add(end); check(repeated.finish().feasible, "repeated cohort streaming assessment failed");
        AssessmentCache checks(stages_profile);
        auto equal_cached = [&](Stage const& stage)
        {
            std::ostringstream expected, actual;
            write_assessment_json(expected, assess_stage(stages_profile, stage));
            write_assessment_json(actual, checks.assess(stage));
            check(expected.str() == actual.str(), "prepared/cache assessment changed exact resource result or stage provenance");
            check(checks.retained_memory_bytes() <= AssessmentCache::memory_bytes / 2,
                  "prepared/cache retention exceeded its declared memory allowance");
        };
        for (unsigned pass = 0; pass < 2; ++pass)
            for (unsigned variant = 0; variant < 12; ++variant)
            {
                auto stage = initial(); stage.name = "candidate-" + std::to_string(pass) + '-' + std::to_string(variant);
                stage.remaining = {{"original-pos1", variant % 10 + 1}};
                stage.future_filter = variant % 2; stage.construction = {service("cpu", 1, variant + 1)};
                equal_cached(stage);
            }
        auto cached = initial(); equal_cached(cached);
        cached.name = "renamed-cached-positive-count"; cached.remaining = {{"original-pos1", 1}}; equal_cached(cached);
        cached.remaining[0].count = 0; equal_cached(cached);
        cached.scope_complete = false; equal_cached(cached);
        cached.scope_complete = true; cached.construction = {memory("ram", 100001)}; equal_cached(cached);
        cached.name = "renamed-infeasible"; equal_cached(cached);
        cached = initial(); cached.construction.assign(300, service("cpu", 0, 0)); equal_cached(cached);
        cached.construction.back().resource = "unknown";
        rejects([&] { checks.assess(cached); }, "oversized uncached stage bypassed unknown-resource validation");
        cached = initial(); cached.remaining = {{"original-pos1", 11}};
        rejects([&] { checks.assess(cached); }, "cached positive count bypassed immutable inventory maximum");
        cached.remaining = {{"original-pos1", 1}, {"original-pos1", 2}};
        rejects([&] { checks.assess(cached); }, "cached stage bypassed duplicate remaining validation");
        cached.remaining = {{"unknown", 1}};
        rejects([&] { checks.assess(cached); }, "cached stage bypassed unknown remaining validation");
        cached = initial(); cached.construction = {memory("ram", UINT64_MAX)};
        rejects([&] { checks.assess(cached); }, "prepared assessment bypassed resource overflow validation");
        AssessmentPreparation binding(stages_profile);
        for (auto ready : {0, 1, 10}) for (bool building : {false, true})
        {
            auto direct = initial(), prepared = initial();
            bind_native_stage(stages_profile, request(), ready, direct, building, !building);
            bind_native_stage(binding, request(), ready, prepared, building, !building);
            std::ostringstream expected, actual;
            write_assessment_json(expected, assess_stage(stages_profile, direct));
            write_assessment_json(actual, checks.assess(prepared));
            check(expected.str() == actual.str(), "prepared native binding changed its exact assessment");
        }
        auto external_cache_profile = stages_profile;
        auto& outside = external_cache_profile.workloads.front(); outside.external_service_node_id = std::string(64, 'a');
        outside.external_service_host = "harvester"; outside.external_service_members = outside.plot_count;
        AssessmentCache external_checks(external_cache_profile);
        check(external_checks.assess(initial()).feasible, "prepared external reservation rejected");
        cached = initial(); cached.remaining = {{outside.id, 1}};
        rejects([&] { external_checks.assess(cached); }, "cached stage reduced an unscanned external service reservation");
        auto maximum_profile = stages_profile; maximum_profile.resources.clear(); maximum_profile.workloads.clear(); maximum_profile.native = {};
        for (size_t index = 0; index < max_resources; ++index)
        {
            auto r = resource(std::to_string(index) + std::string(1000, 'r'), ResourceKind::storage, 1);
            r.host = std::string(1024, 'h');
            for (unsigned alias = 0; alias < 32; ++alias)
                r.aliases.push_back(std::to_string(index) + ':' + std::to_string(alias) + std::string(1000, 'a'));
            maximum_profile.resources.push_back(std::move(r));
        }
        AssessmentCache maximum_checks(maximum_profile);
        check(maximum_checks.assess(initial()).feasible && maximum_checks.retained_memory_bytes() <= AssessmentCache::memory_bytes / 2,
              "maximum-schema resource graph exceeded the existing assessment cache allowance");
        check(reason(assess_transition(stages_profile, {start}), "missing-final-future-filter"), "final future filter omitted");
        modeled_gpu(stages_profile); stages_profile.workloads[0].future_proof_burst = 1000;
        check(reason(assess_transition(stages_profile, {start, end}), "deadline"), "future proof burst not checked");

        auto gate_profile = fixture(); gate_profile.resources[1].telemetry_required = true;
        gate_profile.gate.max_age_ms = 100; gate_profile.gate.recovery_hold_ms = 20; gate_profile.gate.recovery_samples = 2;
        AdmissionGate gate(gate_profile); auto now = Clock::time_point(std::chrono::milliseconds(1000)); s = initial();
        check(!gate.evaluate(s, now).allowed, "required telemetry missing but admitted");
        auto t = sample(1, 1, 1, 100000); check(gate.observe(t, now), "fresh sample rejected");
        check(!gate.evaluate(s, now).allowed, "first recovery sample skipped hysteresis");
        check(!gate.observe(t, now + std::chrono::milliseconds(10)), "duplicate sequence refreshed sample");
        check(!gate.evaluate(s, now + std::chrono::milliseconds(20)).allowed, "repeated evaluation counted recovery vote");
        check(gate.observe(sample(2, 1, 1, 100000), now + std::chrono::milliseconds(20)), "new collector epoch rejected");
        check(gate.evaluate(s, now + std::chrono::milliseconds(20)).allowed, "safe hysteresis did not resume");
        check(!gate.observe(sample(1, 999, 999, 100000, true), now + std::chrono::milliseconds(21)), "late old collector resurrected");
        check(gate.evaluate(s, now + std::chrono::milliseconds(21)).allowed, "rejected old failure changed policy");
        check(!gate.evaluate(s, now + std::chrono::milliseconds(121)).allowed, "stale telemetry admitted");
        auto imported = sample(3, 1, 1, 100000); imported.bounded_source_age_ms.reset();
        check(!gate.observe(imported, now + std::chrono::milliseconds(121)), "receipt time refreshed unknown producer age");
        imported.bounded_source_age_ms = 101;
        check(!gate.observe(imported, now + std::chrono::milliseconds(121)), "old producer data accepted");
        check(gate.observe(sample(3, 1, 1, 0), now + std::chrono::milliseconds(122)), "pressure sample rejected");
        check(!gate.evaluate(s, now + std::chrono::milliseconds(122)).allowed, "insufficient headroom admitted");
        check(!gate.observe(sample(2, 999, 999, 100000), now + std::chrono::milliseconds(123)), "late old success resurrected after pressure");
        check(!gate.evaluate(s, now + std::chrono::milliseconds(123)).allowed, "old success cleared current pressure");
        auto unchanged_tick = sample(3, 2, 1, 100000);
        check(!gate.observe(unchanged_tick, now + std::chrono::milliseconds(124)), "new sequence refreshed unchanged source counter");
        auto changed_source = sample(3, 2, 2, 100000); changed_source.producer = "unselected-source-v1";
        check(!gate.observe(changed_source, now + std::chrono::milliseconds(125)), "same epoch changed source identity");
        auto memory_stage = initial(); memory_stage.construction = {memory("ram", 2000)};
        memory_stage.new_farming = {memory("ram", 300), service("cpu", 1, 1)};
        memory_stage.proof_reserve = {memory("ram", 300), service("cpu", 1, 1)};
        AdmissionGate inactive(p), active(p), protected_memory(p);
        check(inactive.observe(sample(1, 1, 1, 700), now) && !inactive.evaluate(memory_stage, now).allowed,
              "inactive admission discounted already-held memory");
        check(active.observe(sample(1, 1, 1, 700), now), "active memory fixture observation rejected");
        auto active_result = active.evaluate(memory_stage, now, 999999);
        check(active_result.allowed && active_result.assessment.margins[1].occupied_units == 3700,
              "actual own memory credit changed static whole-phase bounds");
        check(protected_memory.observe(sample(1, 1, 1, 699), now) && !protected_memory.evaluate(memory_stage, now, 999999).allowed,
              "own memory credit consumed protected farming/proof/foreign transient budget");
        check(!active.evaluate(memory_stage, now).allowed, "active memory credit leaked into new admission");

        auto proof_stage = memory_stage; proof_stage.construction.clear(); proof_stage.verification = {memory("ram", 2000)};
        proof_stage.background = {memory("ram", 1000)};
        OwnMemoryCredit proof_credit{999999, 1100, 999999, 1900};
        AdmissionGate own_proof(p), foreign_memory(p), retained_worker(p);
        check(own_proof.observe(sample(1, 1, 1, 700), now) && own_proof.evaluate_owned(proof_stage, now, proof_credit).allowed,
            "held verifier and retained worker pages were charged again as future allocations");
        check(own_proof.evaluate_owned(proof_stage, now, proof_credit).assessment.margins[1].occupied_units == 4700,
            "owned resident credit weakened the static whole-worker and verifier reservations");
        check(foreign_memory.observe(sample(1, 1, 1, 699), now) && !foreign_memory.evaluate_owned(proof_stage, now, proof_credit).allowed,
            "proof resident credit consumed protected old/new farming or solver memory");
        OwnMemoryCredit parent_credit{999999, 1100, 0, 0};
        check(retained_worker.observe(sample(1, 1, 1, 2600), now) && retained_worker.evaluate_owned(proof_stage, now, parent_credit).allowed,
            "phase admission double charged its already-resident controller and owned worker");
        check(!retained_worker.evaluate(proof_stage, now).allowed, "phase ownership credit leaked into ordinary admission");
        proof_credit.parent_limit_bytes = 3001;
        rejects([&] { own_proof.evaluate_owned(proof_stage, now, proof_credit); }, "resident credit exceeded its local declared phase budget");

        resource_detail::CpuPressureHistory counters;
        auto first = cpu_observation(1000, 100, 0, 0); counters.observe(first, 5000);
        check(!counters.reading(first.at, 5000, first.scope, 0, true), "first CPU sample invented an interval");
        auto second = cpu_observation(1100, 110, 0, 101000000); counters.observe(second, 5000);
        auto raw_pressure = counters.reading(second.at, 5000, second.scope, 0, true);
        auto active_pressure = counters.reading(second.at, 5000, second.scope, 1, true);
        check(raw_pressure && raw_pressure->first == 1000 && active_pressure && active_pressure->first == 200,
              "active CPU credit failed to separate observed own work from raw pressure");
        auto const generation = counters.generation(); auto rapid = second; rapid.at += std::chrono::milliseconds(1);
        counters.observe(rapid, 5000); auto retained = counters.reading(rapid.at, 5000, second.scope, 0, true);
        check(counters.generation() == generation && retained && retained->first == 1000 && retained->second == 1,
              "back-to-back zero-tick sample replaced or refreshed valid CPU evidence");
        check(!counters.reading(second.at + std::chrono::milliseconds(101), 100, second.scope, 0, true), "cached CPU observation never expired");
        check(!counters.reading(second.at, 5000, "other-cpu", 1, true) && !counters.reading(second.at, 5000, second.scope, 1, false),
              "mismatched or unverified CPU scope received own-work credit");
        check(counters.reading(second.at, 5000, second.scope, 0, true)->first == 1000, "active CPU discount leaked into inactive pressure");
        check(resource_detail::active_cpu_pressure(10, 0, 0, 99000000, 1, 100, 1) == 1000, "foreign CPU load was subtracted as own work");
        check(resource_detail::active_cpu_pressure(20, 0, 150000000, 100000000, 1, 100, 2) >= 500,
              "own CPU credit exceeded the active reservation");
        check(!resource_detail::active_cpu_pressure(10, 0, 500000000, 100000000, 1, 100, 1) &&
              !resource_detail::active_cpu_pressure(10, 0, 0, 100000000, 1, 0, 1), "invalid clock/tick evidence received CPU credit");
        auto regression = second; regression.at += std::chrono::milliseconds(100); --regression.total; counters.observe(regression, 5000);
        check(!counters.reading(regression.at, 5000, regression.scope, 0, true), "CPU counter regression reused old evidence");
        resource_detail::CpuPressureHistory quiet;
        auto idle_child = cpu_observation(1000, 100, 100, 0); idle_child.own_after_ns = 0; idle_child.own_counter_padding_ns = 200;
        quiet.observe(idle_child, 5000); idle_child.at += std::chrono::milliseconds(100);
        idle_child.before_wall += std::chrono::milliseconds(100); idle_child.after_wall += std::chrono::milliseconds(100);
        idle_child.total += 10; idle_child.idle += 10; quiet.observe(idle_child, 5000);
        auto const quiet_pressure = quiet.reading(idle_child.at, 5000, idle_child.scope, 1, true);
        check(quiet_pressure && quiet_pressure->first == 0, "quiet proof child was mistaken for missing CPU evidence");
        LocalPressureSampler local("harvester"); auto pressure = local.sample(fixture());
        check(pressure.producer == "local-os-pressure-v2:harvester" && pressure.bounded_source_age_ms.has_value() && pressure.sequence == 1,
              "local current-counter telemetry has no bounded source provenance");
        auto local_memory = std::find_if(pressure.resources.begin(), pressure.resources.end(), [](ResourceSample const& value) { return value.resource == "ram"; });
        check(local_memory != pressure.resources.end() && local_memory->available_known, "local memory pressure counter unavailable");

        std::ostringstream encoded; write_profile_json(encoded, fixture());
        std::istringstream input("{\"farming_profile\":{},\"farm_resources\":" + encoded.str() + "}");
        auto roundtrip = read_profile(input); std::ostringstream again; write_profile_json(again, roundtrip);
        check(encoded.str() == again.str(), "mixed profile JSON changed on round trip");
        rejects([&] { std::istringstream bad("{\"schema\":2}"); read_profile(bad); }, "unknown schema accepted");
        rejects([&] { std::istringstream bad(encoded.str()); read_profile(bad, 4096); }, "profile parsing memory bound ignored");
        auto generated = make_profile_template(fixture().resources, {{"original-pos1", 10}}, "scan", "xchplot2");
        check(!assess_stage(generated, initial()).feasible, "generated template fabricated capacity or cost");
        auto preparation = prepare_resource_declarations(generated);
        check(!preparation.missing.empty() && std::any_of(preparation.missing.begin(), preparation.missing.end(), [](auto const& need)
            { return need.group == "farming-and-proofs" && need.path == "farm_resources.native.cohorts"; }),
            "preparation omitted a missing full-proof declaration");
        preparation.facts.push_back({"local.available_memory_bytes", "123", "observed", "local-os", "snapshot, not a grant"});
        std::ostringstream preparation_json; write_resource_preparation_json(preparation_json, preparation);
        auto prepared = parse_resource_preparation(group_grow::profile_detail::JsonReader(preparation_json.str()).read());
        std::ostringstream preparation_roundtrip; write_resource_preparation_json(preparation_roundtrip, prepared);
        check(preparation_roundtrip.str() == preparation_json.str() && resource_preparation_memory(prepared) > sizeof(prepared) &&
            !assess_stage(generated, initial()).feasible, "preparation failed roundtrip, escaped memory charge or supplied admission authority");
        prepared.facts[0].basis = "measured-worst-case";
        rejects([&] { write_resource_preparation_json(preparation_roundtrip, prepared); }, "unsupported preparation provenance accepted");
        auto complete_preparation = prepare_resource_declarations(fixture());
        check(complete_preparation.missing.empty() && complete_preparation.resource_settings.empty(), "complete declared fixture needs invented preparation inputs");
        Profile broad; broad.source = broad.producer = "declared-stress-fixture";
        for (size_t i = 0; i < max_resources; ++i) broad.resources.push_back(resource("storage-" + std::to_string(i), ResourceKind::storage, 1));
        for (size_t i = 0; i < max_workloads; ++i)
        {
            Workload workload; workload.id = "workload-" + std::to_string(i); workload.cost = CostKind::serialized; workload.plot_count = 1;
            for (auto const& resource : broad.resources) { Use use; use.resource = resource.id; workload.uses.push_back(std::move(use)); }
            broad.workloads.push_back(std::move(workload));
        }
        auto broad_preparation = prepare_resource_declarations(broad);
        check(!broad_preparation.missing.empty() && broad_preparation.missing.size() < max_preparation_entries,
            "bounded preparation diagnostics refused a valid maximum-width profile");

        auto ranking_profile = fixture(); auto rs = initial();
        std::vector<Victim> candidates{{"a", "original-pos1", "unit-a", 1, 500, 5, 100, 30},
                                       {"b", "original-pos1", "unit-a", 1, 100, 3, 1, std::nullopt},
                                       {"c", "original-pos1", "unit-b", 1, 300, 1, 3, 1},
                                       {"d", "original-pos1", "", 1, 200, 4, std::nullopt, 2}};
        auto ordered = rank_victims(ranking_profile, rs, candidates, "cpu");
        do
        {
            auto next = rank_victims(ranking_profile, rs, candidates, "cpu");
            for (size_t i = 0; i < next.size(); ++i) check(next[i].victim.id == ordered[i].victim.id, "mixed optional metrics made ranking order cyclic");
        } while (std::next_permutation(candidates.begin(), candidates.end(), [](Victim const& a, Victim const& b) { return a.id < b.id; }));
        check(ordered.front().released_units == 0 && ordered.front().released_work_us == 0, "ranking invented foreign reservation release");
        auto overflow = fixture(); modeled_gpu(overflow); overflow.workloads[0].proof_burst = UINT64_MAX;
        rejects([&] { assess_stage(overflow, initial()); }, "burst overflow wrapped to feasible");
        std::cout << "mixed farm resource, transition, binding, telemetry and ranking checks passed\n";
        return 0;
    }
    catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
}
