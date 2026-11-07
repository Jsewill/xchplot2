// Migration's concrete resource boundary. Discovery tightens a reviewed
// declaration; neither a saved profile nor an RPC receipt grants spare compute.
#pragma once
#include "host/GroupFarmResources.hpp"
#include "host/GroupFarmState.hpp"
#include "host/GroupFarmMigration.hpp"
#include "host/GroupOptions.hpp"
#include "host/ComputeSession.hpp"
#include "host/GroupStorage.hpp"
#include <type_traits>

namespace group_grow::farm
{
struct ReplotSupervision
{
    // The controller's bounded retained/working ledger, plus one separately
    // capped RPC process and its capture/parser reservations. RPC is sequential
    // with construction/proof; native declarations are complete phase totals.
    uint64_t controller_ram_bytes = 0, adapter_ram_bytes = 0, adapter_cpu_threads = 0, adapter_timeout_us = 0;
    std::vector<std::filesystem::path> source_roots;
};
// Match authenticated service observations to a reviewed physical resource
// scope. Remote addresses identify peers, never spare compute or host aliases.
template<class Others>
inline void require_target_service_scope(Profile const& profile,
    std::map<std::string, std::string> const& actual_hosts, Others const& other_harvesters)
{
    using namespace resource_detail;
    validate(profile); Graph graph(profile);
    require(actual_hosts.size() == 4, "target service physical host binding set is incomplete or unsupported");
    auto host = [&](std::string const& role) -> std::string const&
    {
        auto const actual = actual_hosts.find(role);
        require(actual != actual_hosts.end() && !actual->second.empty(), "target service has no explicit physical host binding");
        auto const reviewed = profile.native.service_hosts.find(role);
        auto const& expected = reviewed == profile.native.service_hosts.end() ? profile.native.host : reviewed->second;
        require(actual->second == expected, "target service physical host differs from the reviewed resource scope");
        return actual->second;
    };
    auto const& harvester = host("harvester"); auto const& farmer = host("farmer");
    auto const& full_node = host("full_node"); auto const& solver = host("solver");
    auto covered = [&](auto const& uses, std::string const& physical, bool proof)
    {
        bool cpu = false, ram = false, full = false;
        for (auto const& use : uses)
        {
            auto const& resource = graph.resources[graph.find(use.resource)];
            if (resource.host != physical) continue;
            if (resource.kind == ResourceKind::cpu && (use.required_units || use.reserved_units)) cpu = true;
            if (resource.kind == ResourceKind::host_memory && (use.resident_units || use.transient_units)) ram = true;
            if (resource.kind == ResourceKind::cpu || resource.kind == ResourceKind::gpu_compute || resource.kind == ResourceKind::solver)
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(use)>, Use>) full = full || use.reserved_units || use.proof_us;
                else full = full || use.reserved_units || use.work_us;
            }
        }
        require(cpu && ram && (!proof || full), "reviewed service envelope omits physical-host CPU, memory or full-proof work");
    };
    auto network = [&](auto const& uses)
    {
        return std::any_of(uses.begin(), uses.end(), [&](auto const& use)
        { return graph.resources[graph.find(use.resource)].kind == ResourceKind::network && (use.required_units || use.reserved_units); });
    };
    for (auto const& cohort : profile.native.cohorts)
    {
        covered(cohort.new_farming, harvester, false); covered(cohort.new_farming, farmer, false);
        covered(cohort.new_farming, full_node, false); covered(cohort.proof_reserve, solver, true);
        if (harvester != farmer || farmer != full_node) require(network(cohort.new_farming), "remote target farming omits its reviewed network envelope");
        if (harvester != solver || farmer != solver) require(network(cohort.proof_reserve), "remote target full proofs omit their reviewed network envelope");
    }
    require(other_harvesters.size() <= max_workloads, "too many observed outside harvesters");
    std::set<std::string> observed;
    for (auto const& other : other_harvesters)
    {
        require(observed.insert(other.node_id).second, "outside harvester identity is ambiguous");
        Workload const* found = nullptr;
        for (auto const& workload : profile.workloads) if (workload.external_service_node_id == other.node_id)
        { require(!found, "outside harvester has repeated reviewed workload scopes"); found = &workload; }
        require(found && found->bounded && found->scope_complete && found->cost != CostKind::unknown,
            "outside harvester has no complete reviewed external service envelope");
        require(other.files <= found->plot_count && other.members <= found->external_service_members,
            "outside harvester inventory exceeds its reviewed maximum service population");
        covered(found->uses, found->external_service_host, false); covered(found->uses, farmer, false); covered(found->uses, solver, true);
        if (found->external_service_host != farmer || found->external_service_host != solver)
            require(network(found->uses), "outside harvester omits its reviewed shared-service network envelope");
    }
}
inline std::string replot_storage_id(StorageDevice const& device)
{ return "storage:" + device.filesystem; }
inline std::vector<std::string> replot_physical_storage(Profile const& profile,
    std::vector<std::filesystem::path> const& directories, std::string const& host)
{
    using namespace resource_detail;
    require(directories.size() <= max_resources, "too many migration physical storage mappings");
    Graph graph(profile); std::set<size_t> used; std::vector<std::string> result;
    for (auto const& path : directories)
    {
        auto const device = storage_device(path); size_t const index = graph.find(replot_storage_id(device));
        auto const& resource = graph.resources[index];
        require(resource.host == host && resource.kind == ResourceKind::storage, "migration storage mapping names another host or resource kind");
        auto alias = [&](std::string const& name)
        { require(graph.find(name) == index, "migration physical storage alias differs from reviewed resource"); };
        if (device.drives.empty()) alias("storage:unknown:" + host);
        else for (auto const& drive : device.drives) alias("drive:" + drive);
        if (used.insert(index).second) result.push_back(resource.id);
    }
    return result;
}
inline uint64_t replot_milliseconds(uint64_t microseconds)
{
    resource_detail::require(microseconds && microseconds <= uint64_t(std::chrono::milliseconds::max().count()),
                             "invalid migration operation duration envelope");
    return microseconds / 1000 + (microseconds % 1000 != 0);
}
inline NativeRequest replot_native_request(pos2gpu::GroupOptions const& args, Profile const& profile,
    FarmingState const& state, std::string gpu_identity = {}, uint64_t gpu_bytes = 0, uint64_t verification_ram_bytes = 0,
    ReplotSupervision supervision = {}, bool legacy_verification = true)
{
    using namespace resource_detail;
    validate(profile); auto const& native = profile.native;
    NativeRequest result; result.host = profile_detail::host_name();
    result.compute_enabled = !args.pool_broker.empty();
    require(result.compute_enabled || (!args.automatic_compute && !args.compute_session && !args.pool_offer_ram &&
        !args.pool_offer_cpus && !args.pool_offer_vram && !args.pool_codec_ram), "compute service/client options have no reviewed broker scope");
    result.compute_session_ram_bytes = args.automatic_compute ? pos2gpu::ComputeSession::memory_bytes : 0;
    result.compute_codec_ram_bytes = args.pool_codec_ram;
    require(!result.host.empty(), "cannot establish migration execution host");
    auto const cpus = profile_detail::logical_cpus(); require(cpus, "cannot establish inherited migration CPU affinity");
    result.cpu_affinity_lanes = cpus;
    result.k = state.config.k; result.strength = state.config.strength;
    result.reader = state.config.reader == FarmingReader::stateless ? "stateless" : "cached";
    result.backend = args.backend; result.engine = args.construction;
    result.ram_bytes = args.max_ram;
    result.cpu_threads = args.workers ? args.workers : std::min<uint64_t>(native.max_cpu_threads, cpus);
    require(result.cpu_threads && result.cpu_threads <= cpus, "migration construction workers exceed inherited CPU affinity");
    result.batch_members = std::min(args.batch, native.max_batch_members);
    result.gpu_bytes = gpu_bytes; result.gpu_identity = std::move(gpu_identity);
    result.standing_gpu_bytes = args.backend == "gpu" && !result.compute_enabled ? native.standing_gpu_bytes : 0;
    result.gpu_binding_verified = !result.gpu_identity.empty();
    require((result.compute_enabled && !gpu_bytes) || (args.backend == "cpu" && !gpu_bytes) || (args.backend == "gpu" && gpu_bytes && result.gpu_binding_verified),
            "migration backend requires exact bounded CPU or physically identified GPU execution");
    result.construction_us = native.max_construction_us; result.verification_us = native.max_verification_us;
    // New policies install a conservative worker affinity before solver fanout.
    // Legacy absence retains its reviewed inherited-affinity interpretation.
    result.verification_cpu_threads = native.verification_cpu_threads ? native.verification_cpu_threads : legacy_verification ? cpus : 1;
#if !defined(__linux__) && !defined(_WIN32)
    require(!native.verification_cpu_threads, "explicit migration verifier CPU affinity is unsupported on this platform");
#endif
    result.verification_ram_bytes = verification_ram_bytes ? verification_ram_bytes : native.max_ram_bytes;
    require(result.verification_ram_bytes && result.verification_ram_bytes <= native.max_ram_bytes,
            "migration proof worker exceeds the reviewed memory cap");
    require(supervision.controller_ram_bytes && supervision.adapter_ram_bytes && supervision.adapter_cpu_threads &&
            supervision.adapter_cpu_threads <= cpus && supervision.adapter_timeout_us,
            "migration requires the actual bounded controller/RPC memory, CPU and deadline ledger");
    result.controller_ram_bytes = supervision.controller_ram_bytes; result.adapter_ram_bytes = supervision.adapter_ram_bytes;
    result.adapter_cpu_threads = supervision.adapter_cpu_threads; result.adapter_timeout_us = supervision.adapter_timeout_us;
    std::vector<std::filesystem::path> outputs, scratch = args.temporary;
    require(state.directories.size() <= max_resources, "too many migration destination mappings");
    for (auto const& directory : state.directories)
    {
        auto const path = file_io::native_path(directory.directory);
        require(path.is_absolute() && !directory.directory_identity.empty() && farming_directory_identity(path) == directory.directory_identity,
                "migration destination native identity changed");
        outputs.push_back(path);
        // Both supported fixed-forecast layouts put their largest membership
        // in the first file; later files can only have equal/fewer members.
        if (directory.layout.files) result.group_members = std::max<uint64_t>(result.group_members, directory.layout.file_members(0));
    }
    // Destination staging can consume any output resource. Its existing
    // allocator still enforces free space; this accounts for its I/O envelope.
    if (args.destination_temp != "off") scratch.insert(scratch.end(), outputs.begin(), outputs.end());
    result.output_resources = replot_physical_storage(profile, outputs, result.host);
    result.scratch_resources = replot_physical_storage(profile, scratch, result.host);
    result.source_resources = replot_physical_storage(profile, supervision.source_roots, result.host);
    (void)replot_milliseconds(result.construction_us);
    Stage check; check.name = "native-execution-binding";
    bind_native_stage(profile, result, 0, check, true, false);
    // Proof envelopes are checked only for batches selected by the saved policy.
    // A completed earlier sample cannot charge proof work to the remaining suffix.
    bind_native_stage(profile, result, 0, check, false, false);
    return result;
}
inline Clock::time_point replot_operation_deadline(uint64_t duration_us)
{
    auto const now = Clock::now(); uint64_t const ms = replot_milliseconds(duration_us);
    auto const maximum = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now).count();
    resource_detail::require(maximum > 0 && ms <= uint64_t(maximum), "migration operation deadline overflows its monotonic clock");
    return now + std::chrono::milliseconds(ms);
}
inline uint64_t replot_verification_retry_delay(GatePolicy const& policy, uint64_t previous_ms)
{
    uint64_t const first = std::max<uint64_t>(1, policy.recovery_hold_ms);
    uint64_t const maximum = std::max(first, policy.max_age_ms);
    return previous_ms ? std::min(maximum, resource_detail::add(previous_ms, std::min(previous_ms, maximum - std::min(previous_ms, maximum)))) : first;
}
inline pos2gpu::GroupOptions replot_execution_options(pos2gpu::GroupOptions const& options, NativeRequest const& request,
    Clock::time_point until = Clock::time_point::max())
{
    auto result = options; result.max_ram = request.ram_bytes; result.workers = request.cpu_threads;
    result.batch = request.batch_members; result.backend = request.backend; result.backend_set = true;
    result.construction = request.engine;
    result.protected_resources = true; result.protected_gpu_bytes = request.gpu_bytes;
    result.io_overlap = false; result.generation_overlap = false; result.generation_overlap_auto = false;
    result.cancelled = [cancelled = options.cancelled, until] { return (cancelled && cancelled()) || Clock::now() >= until; };
    return result;
}
inline Stage replot_stage(AssessmentPreparation const& preparation, NativeRequest const& request, std::vector<Remaining> remaining,
    uint64_t ready_members, bool building, bool verifying, bool future, std::string name = {})
{
    resource_detail::require(!building || !verifying, "migration construction and proof phases must be supervised sequentially");
    Stage result; result.name = name.empty() ? building ? "construction" : verifying ? "verification" : future ? "final-future" : "checkpoint" : std::move(name);
    result.future_filter = future; result.remaining = std::move(remaining);
    bind_native_stage(preparation, request, ready_members, result, building, verifying); return result;
}
inline Stage replot_stage(Profile const& profile, NativeRequest const& request, std::vector<Remaining> remaining,
    uint64_t ready_members, bool building, bool verifying, bool future, std::string name = {})
{
    resource_detail::require(!building || !verifying, "migration construction and proof phases must be supervised sequentially");
    return replot_stage(AssessmentPreparation(profile), request, std::move(remaining), ready_members, building, verifying, future, std::move(name));
}
inline uint64_t replot_ready_members(FarmingState const& state, uint64_t ready_files)
{
    using namespace resource_detail;
    uint64_t result = 0;
    for (auto const& directory : state.directories)
    {
        auto const& layout = directory.layout; uint64_t const files = std::min(ready_files, layout.files);
        ready_files -= files;
        if (!files) continue;
        if (layout.fixed_members) result = add(result, std::min(layout.total_members, multiply(files, layout.group_members)));
        else
        {
            require(layout.files, "migration ready prefix has no forecast files");
            result = add(result, add(multiply(files, layout.total_members / layout.files), std::min(files, layout.total_members % layout.files)));
        }
    }
    require(!ready_files, "migration ready prefix exceeds immutable output assignments"); return result;
}
// Forecast only work still required at this exact stopped boundary. Current
// counts are caller-authoritative after native retirement and fresh complete
// registry unload checks; this function never guesses an already-retired
// cohort away. Future whole-cohort reductions retain the ordinary forecast's
// explicit conditional unload assumption for fully observable inventories.
inline Assessment assess_replot_remaining(Profile const& profile, NativeRequest const& request, FarmInventory const& inventory,
    ReplotReview const& review, ReplotJournal const& journal, std::vector<Remaining> remaining, uint64_t ready_members,
    std::optional<ReplotVerification> verification = {})
{
    using namespace resource_detail;
    validate_replot_review(review); validate_replot_progress(review, journal);
    auto const policy = verification.value_or(replot_verification_policy(review.verification));
    require(journal.batch <= review.batches.size() && remaining.size() <= max_workloads,
        "remaining replot assessment exceeds its immutable boundary");
    require(journal.retired_prefix == (journal.batch < review.batches.size() ? review.batches[size_t(journal.batch)].first_source : review.sources.size()),
        "remaining replot assessment has a different retired source prefix");
    auto const completed_members = replot_ready_members(review.farm, journal.batch);
    auto const exposed_members = journal.batch < review.batches.size() && (journal.verified_stage || journal.published || journal.ready) ?
        replot_ready_members(review.farm, journal.batch + 1) : completed_members;
    require(ready_members >= completed_members && ready_members <= exposed_members,
        "remaining replot assessment has an unaudited ready member prefix");
    std::map<std::string, bool> can_confirm_unload;
    for (auto const& plot : inventory.plots) if (!plot.alias_of && !(plot.status == "non-plot" && !plot.header_validated && !plot.reason.empty()))
    {
        bool const known = plot.header_validated || (!plot.reader_identity.empty() && !plot.reader_version.empty());
        auto [at, fresh] = can_confirm_unload.emplace(plot.workload_id, known); if (!fresh) at->second = at->second && known;
    }
    auto workload = [&](ReplotBatch const& batch, size_t source) -> InventoryPlot const&
    { return inventory.plots.at(size_t(review.sources.at(size_t(batch.first_source) + source))); };
    auto reduce = [&](std::vector<Remaining>& counts, InventoryPlot const& plot)
    {
        auto at = std::find_if(counts.begin(), counts.end(), [&](auto const& item) { return item.workload == plot.workload_id; });
        require(at != counts.end() && at->count, "remaining replot forecast lost an active source workload");
        auto known = can_confirm_unload.find(plot.workload_id);
        if (at->count > 1 || (known != can_confirm_unload.end() && known->second)) --at->count;
    };
    auto const restart_counts = remaining; AssessmentAccumulator accumulation(profile); std::string exhausted_selection;
    auto& checks = accumulation.cache();
    auto phase = [&](std::vector<Remaining> const& counts, uint64_t ready, bool building, bool verifying, std::string const& name)
    { for (bool future : {false, true}) accumulation.add(replot_stage(checks.preparation(), request, counts, ready, building, verifying, future, name)); };
    phase(remaining, ready_members, false, false, "current-confirmed-boundary-supervision");
    for (size_t index = size_t(journal.batch); index < review.batches.size(); ++index)
    {
        auto const& batch = review.batches[index]; bool const current = index == journal.batch;
        bool const proof = replot_requires_proof(policy, index);
        bool const sealed = current && (journal.verified_stage || journal.published || journal.ready);
        require(!current || journal.retirements.size() == batch.sources, "remaining replot assessment lacks its exact current retirement ledger");
        auto const prefix = "remaining-batch-" + std::to_string(index);
        phase(remaining, ready_members, false, false, prefix + "-conditional-before-retirement");
        std::vector<size_t> early;
        if (!sealed)
        {
            uint64_t outstanding = 0;
            if (current) for (size_t source = 0; source < journal.retirements.size(); ++source)
            {
                auto const progress = journal.retirements[source];
                require(progress.state == ReplotRetirement::active || progress.predelete,
                    "unsealed remaining batch has an unauthorized post-publication retirement");
                if (progress.predelete)
                {
                    outstanding = add(outstanding, workload(batch, source).identity.allocated_bytes);
                    if (progress.state != ReplotRetirement::retired) early.push_back(source);
                }
            }
            auto selected = select_replot_predeletions(size_t(batch.sources), review.reclaim_before_bytes, outstanding, std::move(early),
                [&](size_t source)
                {
                    bool const active = !current || journal.retirements[source].state == ReplotRetirement::active;
                    auto const& plot = workload(batch, source);
                    return ReplotPredeleteCandidate{plot.identity.allocated_bytes, active && plot.identity.links == 1 && !plot.alias_of};
                }, [&](std::span<size_t const> prospective)
                {
                    auto counts = remaining; for (auto source : prospective) reduce(counts, workload(batch, source));
                    for (bool future : {false, true})
                        if (!checks.assess(replot_stage(checks.preparation(), request, counts, ready_members, true, false, future)).feasible ||
                            !checks.assess(replot_stage(checks.preparation(), request, counts, ready_members, false, proof, future)).feasible) return false;
                    return true;
                });
            if (selected.search_exhausted && exhausted_selection.empty()) exhausted_selection = prefix + ":predelete-selection-search-bound-exhausted";
            early = std::move(selected.sources);
            for (auto source : early) reduce(remaining, workload(batch, source));
            phase(remaining, ready_members, true, false, prefix + "-conditional-build-after-confirmed-unload");
            phase(remaining, ready_members, false, proof, prefix + (proof ? "-conditional-proof" : "-conditional-structure") + "-after-confirmed-unload");
        }
        ready_members = std::max(ready_members, replot_ready_members(review.farm, uint64_t(index) + 1));
        phase(restart_counts, ready_members, false, false, prefix + "-restart-with-unconfirmed-future-registries");
        phase(remaining, ready_members, false, false, prefix + "-conditional-replacement-overlap");
        for (size_t source = 0; source < batch.sources; ++source)
        {
            bool const retired = current && journal.retirements[source].state == ReplotRetirement::retired;
            if (!retired && std::find(early.begin(), early.end(), source) == early.end()) reduce(remaining, workload(batch, source));
        }
    }
    accumulation.add(replot_stage(checks.preparation(), request, remaining, ready_members, false, false, true,
        "remaining-final-after-conditional-confirmed-source-unload"));
    auto result = accumulation.finish();
    if (!exhausted_selection.empty()) fail(result, std::move(exhausted_selection));
    return result;
}
inline FarmingAssessment replot_native_farming_assessment(FarmingState const& state, uint64_t memory_limit)
{
    using namespace resource_detail;
    require(state.fixed_extent && !state.capacity_fill && !state.directories.empty() && state.directories.size() <= max_resources,
            "migration requires a fixed ordinary farming forecast");
    std::vector<StorageDevice> devices; std::vector<FarmingDestination> destinations; std::vector<FarmingDrivePlan> layouts;
    std::vector<std::vector<uint16_t>> members; uint64_t files = 0;
    for (auto const& directory : state.directories)
    {
        files = add(files, directory.layout.files);
        require(files <= farming_detail::maximum_files && multiply(files, sizeof(uint16_t) + 16) < memory_limit / 2,
                "migration native forecast exceeds its assessment memory grant");
        devices.push_back(storage_device(file_io::native_path(directory.directory)));
        destinations.push_back(directory.storage); layouts.push_back(directory.layout); members.emplace_back();
        auto& values = members.back(); values.reserve(size_t(directory.layout.files));
        for (uint64_t file = 0; file < directory.layout.files; ++file) values.push_back(directory.layout.file_members(file));
    }
    auto const physical = writer_groups(devices);
    for (size_t i = 0; i < destinations.size(); ++i) destinations[i].resource = physical[i];
    // Keep the ordinary model's complete conditional quality/RAM admission.
    // Its eligibility-weighted CPU/I/O means never become mixed burst bounds.
    return farming_assess(state.config, destinations, layouts, members, memory_limit / 2);
}
inline void write_replot_request_json(std::ostream& out, NativeRequest const& request)
{
    out << "{\"host\":"; pos2gpu::write_json_string(out, request.host);
    out << ",\"backend\":"; pos2gpu::write_json_string(out, request.backend);
    out << ",\"engine\":"; pos2gpu::write_json_string(out, request.engine);
    out << ",\"ram_bytes\":" << request.ram_bytes << ",\"cpu_threads\":" << request.cpu_threads
        << ",\"cpu_affinity_lanes\":" << request.cpu_affinity_lanes
        << ",\"gpu_bytes\":" << request.gpu_bytes << ",\"standing_gpu_bytes\":" << request.standing_gpu_bytes
        << ",\"batch_members\":" << request.batch_members
        << ",\"group_members\":" << request.group_members
        << ",\"construction_us\":" << request.construction_us << ",\"verification_us\":" << request.verification_us
        << ",\"verification_ram_bytes\":" << request.verification_ram_bytes << ",\"verification_cpu_threads\":" << request.verification_cpu_threads
        << ",\"controller_ram_bytes\":" << request.controller_ram_bytes << ",\"adapter_ram_bytes\":" << request.adapter_ram_bytes
        << ",\"adapter_cpu_threads\":" << request.adapter_cpu_threads << ",\"adapter_timeout_us\":" << request.adapter_timeout_us
        << ",\"gpu_binding_verified\":" << (request.gpu_binding_verified ? "true" : "false") << ",\"gpu_identity\":";
    pos2gpu::write_json_string(out, request.gpu_identity);
    if (request.generation_peak_bytes) out << ",\"generation_peak_bytes\":" << request.generation_peak_bytes;
    if (request.compute_enabled)
        out << ",\"compute_enabled\":true,\"compute_session_ram_bytes\":" << request.compute_session_ram_bytes
            << ",\"compute_codec_ram_bytes\":" << request.compute_codec_ram_bytes;
    for (auto const& [name, values] : {std::pair{"output_resources", &request.output_resources}, {"scratch_resources", &request.scratch_resources}, {"source_resources", &request.source_resources}})
    {
        out << ",\"" << name << "\":["; bool comma = false;
        for (auto const& id : *values) { if (comma) out << ','; comma = true; pos2gpu::write_json_string(out, id); }
        out << ']';
    }
    out << '}';
}
class ReplotGate
{
    Profile profile_;
    AdmissionGate gate_;
    LocalPressureSampler sampler_;
    resource_detail::Graph graph_;
    std::optional<uint64_t> previous_tick_;
    std::optional<uint64_t> accepted_own_anon_;
    std::optional<uint64_t> accepted_child_anon_;
    uint64_t controller_ram_bytes_ = 0;
    // The caller reports only live/identity-reserved services it owns, never
    // all reviewed workers on this host. Phase totals still reserve all of them.
    std::function<uint64_t(bool)> owned_background_ram_;
    Admission poll_impl(Stage const& stage, bool active, SupervisedProcess const* proof = nullptr, uint64_t proof_ram_bytes = 0)
    {
        using namespace resource_detail;
        bool const constructing = active && !proof;
        require(!proof || (active && stage.construction.empty() &&
            (!stage.verification.empty() != !stage.supervision.empty()) && proof_ram_bytes),
            "active worker monitoring requires its one capped verification or supervision phase");
        auto memory = [&](std::vector<Demand> const& demands)
        { uint64_t bytes = 0; for (auto const& d : demands)
          { auto const& r = graph_.resources[graph_.find(d.resource)];
            if (r.kind == ResourceKind::host_memory && r.host == profile_.native.host) bytes = add(bytes, add(d.resident_units, d.transient_units)); }
          return bytes; };
        auto const background = owned_background_ram_ ? owned_background_ram_(constructing) : 0;
        require(background <= memory(stage.background), "owned compute services exceed their reviewed local background demand");
        OwnMemoryCredit credit; credit.parent_limit_bytes = add(constructing ? memory(stage.construction) : controller_ram_bytes_, background);
        credit.child_limit_bytes = proof_ram_bytes;
        uint64_t lanes = 0;
        if (active || controller_ram_bytes_)
        {
            require(!constructing || (!stage.construction.empty() && stage.verification.empty() && stage.supervision.empty()),
                "active monitoring requires the running construction phase");
            auto own_cpu = [&](std::vector<Demand> const& demands) { for (auto const& d : demands)
            {
                auto const& resource = graph_.resources[graph_.find(d.resource)];
                if (resource.kind == ResourceKind::cpu && resource.host == profile_.native.host)
                    lanes = resource_detail::add(lanes, resource_detail::add(d.required_units, d.reserved_units));
            } };
            own_cpu(stage.construction); own_cpu(stage.verification); own_cpu(stage.supervision); own_cpu(stage.background);
            lanes = std::min<uint64_t>(lanes, profile_detail::logical_cpus());
            require(lanes, "owned processes have no declared CPU reservation");
        }
        std::string failure;
        try
        {
            auto reading = sampler_.sample(profile_, Clock::now(), lanes, proof, credit.parent_limit_bytes != 0); auto const now = Clock::now();
            bool const duplicate_tick = previous_tick_ && reading.producer_tick_ms <= *previous_tick_;
            if (gate_.observe(reading, now))
            { accepted_own_anon_ = sampler_.own_anon_bytes(); accepted_child_anon_ = sampler_.child_anon_bytes(); }
            else if (!duplicate_tick) failure = "unavailable-current-local-pressure";
            if (credit.parent_limit_bytes && !sampler_.own_anon_bytes()) failure = "unavailable-current-own-anonymous-memory";
            if (proof && !sampler_.child_anon_bytes()) failure = "unavailable-current-proof-anonymous-memory";
            previous_tick_ = reading.producer_tick_ms;
        }
        catch (ProcessObservationUnavailable const&) { throw; }
        catch (std::exception const&) { failure = "unavailable-current-local-pressure"; }
        if (credit.parent_limit_bytes && !accepted_own_anon_) failure = "unavailable-current-own-anonymous-memory";
        if (proof && !accepted_child_anon_) failure = "unavailable-current-proof-anonymous-memory";
        if (!failure.empty()) gate_.pause();
        credit.parent_resident_bytes = accepted_own_anon_.value_or(0);
        credit.child_resident_bytes = proof ? accepted_child_anon_.value_or(0) : 0;
        auto result = gate_.evaluate_owned(stage, Clock::now(), credit);
        if (!failure.empty()) { resource_detail::fail(result.assessment, failure); result.allowed = false; result.paused = true; }
        return result;
    }
public:
    explicit ReplotGate(Profile profile, std::string actual_host, uint64_t controller_ram_bytes = 0,
        std::function<uint64_t(bool)> owned_background_ram = {})
        : profile_(std::move(profile)), gate_(profile_), sampler_(actual_host), graph_(profile_),
          controller_ram_bytes_(controller_ram_bytes), owned_background_ram_(std::move(owned_background_ram))
    {
        resource_detail::require(!actual_host.empty() && actual_host == profile_detail::host_name() && actual_host == profile_.native.host,
                                 "migration runtime gate host differs from the reviewed execution host");
    }
    Admission poll(Stage const& stage)
    { return poll_impl(stage, false); }
    // Sole monitor caller, while the declared construction reservation is held.
    // This is not admission of new work: static envelopes remain fully charged.
    Admission poll_active(Stage const& stage) { return poll_impl(stage, true); }
    Admission poll_verification(Stage const& stage, SupervisedProcess const& proof, uint64_t proof_ram_bytes)
    { resource_detail::require(!stage.verification.empty(), "proof worker has no verification reservation"); return poll_impl(stage, true, &proof, proof_ram_bytes); }
    Admission poll_structure(Stage const& stage, SupervisedProcess const& worker, uint64_t worker_ram_bytes)
    { resource_detail::require(!stage.supervision.empty(), "structural worker has no supervision reservation"); return poll_impl(stage, true, &worker, worker_ram_bytes); }
    Admission wait(Stage const& stage, uint64_t timeout_ms, std::function<bool()> const& cancelled = {},
                   std::function<void(Admission const&)> const& on_pause = {})
    {
        resource_detail::require(timeout_ms && timeout_ms <= 3600000, "invalid migration admission wait deadline");
        auto const until = Clock::now() + std::chrono::milliseconds(timeout_ms);
        auto const static_result = assess_stage(profile_, stage);
        if (!static_result.feasible)
        {
            Admission refused; refused.assessment = static_result; refused.paused = true;
            if (on_pause) on_pause(refused);
            throw std::runtime_error("migration stage does not fit its reviewed resource envelopes");
        }
        bool reported = false;
        for (;;)
        {
            resource_detail::require(!cancelled || !cancelled(), "migration resource wait cancelled");
            resource_detail::require(Clock::now() < until, "migration resource wait deadline exceeded");
            auto current = poll(stage); if (current.allowed) return current;
            if (!reported && on_pause) { on_pause(current); reported = true; }
            std::this_thread::sleep_until(std::min(until, Clock::now() + std::chrono::milliseconds(50)));
        }
    }
    static uint64_t memory_bytes(Profile const& profile)
    { return resource_detail::add(resource_detail::multiply(profile_memory(profile), 4), 16 * 1024 * 1024); }
};
} // namespace group_grow::farm
