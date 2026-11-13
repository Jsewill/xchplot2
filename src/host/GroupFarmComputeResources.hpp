// Migration binding for the existing globally bounded compute executors.
#pragma once
#include "host/GroupFarmResources.hpp"
#include "host/GroupOptions.hpp"
#include "host/ComputeSession.hpp"
#include "host/ComputeWorker.hpp"
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
#include "gpu/SyclDeviceList.hpp"
#endif

namespace group_grow::farm
{
// Complete only facts that the enrolled executor can establish itself. A
// worker's whole configured partition is independently bounded; this does not
// invent a third-party farming/solver cost or remote network service bound.
inline ResourcePreparation prepare_compute_resources(Profile& profile, pos2gpu::ComputeInspection const& observed)
{
    using namespace resource_detail;
    ResourcePreparation preparation;
    preparation.facts.push_back({"compute.status", observed.status, "observed", "authenticated-compute-inspection", observed.reason});
    if (!observed.configured)
    {
        preparation.facts.push_back({"compute.setup", "xchplot2 compute setup", "unknown", "local-enrollment",
            "Optional shared compute: set up the plotting machine once, invite helpers, then run compute setup INVITATION on each helper."});
        return preparation;
    }
    auto resource = [&](std::string const& id, std::string const& host, ResourceKind kind, uint64_t capacity)
    {
        Graph graph(profile);
        auto at = graph.names.find(id);
        if (at == graph.names.end() && (kind == ResourceKind::cpu || kind == ResourceKind::host_memory))
        {
            std::vector<size_t> matches;
            for (size_t i = 0; i < graph.resources.size(); ++i)
                if (graph.resources[i].host == host && graph.resources[i].kind == kind) matches.push_back(i);
            require(matches.size() <= 1, "compute setup requires an unambiguous physical CPU/memory binding for each host");
            if (!matches.empty()) at = graph.names.find(graph.resources[matches.front()].id);
        }
        if (at != graph.names.end())
        {
            require(graph.resources[at->second].host == host && graph.resources[at->second].kind == kind,
                "observed compute resource conflicts with its saved physical binding");
            auto const canonical = graph.resources[at->second].id;
            for (auto& saved : profile.resources) if (saved.id == canonical)
            {
                if (!saved.capacity_known && !saved.capacity && capacity) { saved.capacity = capacity; saved.capacity_known = true; }
                if (!graph.names.contains(id)) saved.aliases.push_back(id);
            }
            return canonical;
        }
        require(profile.resources.size() < max_resources, "compute setup exceeds the physical resource limit");
        profile.resources.push_back({id, host, {}, kind, capacity, capacity != 0});
        return id;
    };
    for (auto const& worker : observed.worker_resources)
    {
        auto const path = "compute.workers[" + worker.identity + ']';
        preparation.facts.push_back({path, "host=" + worker.host + ";cpus=" + std::to_string(worker.cpus) +
            ";host_bytes=" + std::to_string(worker.host_bytes) + ";gpu_bytes=" + std::to_string(worker.gpu_bytes) +
            ";observed_at_unix_ms=" + std::to_string(worker.observed_at_unix_ms),
            worker.admission == "configured-local-offer" ? "declared" : "observed",
            worker.admission == "configured-local-offer" ? "configured-local-executor" : "enrolled-worker",
            "Physical binding and complete executor partition from authenticated status or the selected local executor configuration; current free memory does not establish foreign-service limits."});
        auto const found = std::find_if(profile.native.compute_workers.begin(), profile.native.compute_workers.end(),
            [&](auto const& saved) { return saved.tls_identity == worker.identity; });
        if (found != profile.native.compute_workers.end())
        {
            if (found->host != worker.host || worker.cpus > found->cpu_threads || worker.slots > found->slots ||
                worker.host_bytes > found->host_bytes || worker.gpu_bytes > found->gpu_bytes || worker.standing_host_bytes > found->standing_host_bytes)
                preparation.missing.push_back({"compute", path, "The current worker binding or partition exceeds the saved scope; refresh its resource profile before using it."});
            continue;
        }
        if (worker.host.empty() || !worker.cpus || !worker.slots || !worker.host_bytes || !worker.standing_host_bytes ||
            worker.standing_host_bytes >= worker.host_bytes || !worker.logical_cpus || !worker.total_host_bytes || !worker.capabilities)
        {
            preparation.missing.push_back({"compute", path, "Start a compatible enrolled worker with compute setup, then repeat setup to obtain its complete physical resource offer."});
            continue;
        }
        bool const gpu = worker.capabilities & compute_generate_gpu;
        if (gpu && (worker.gpu_identity.empty() || !worker.gpu_bytes || worker.total_gpu_bytes <= worker.gpu_bytes))
        {
            preparation.missing.push_back({"compute", path, "This GPU worker has no complete physical allocation and standing-residency scope; supply its qualified profile or offer CPU compute."});
            continue;
        }
        auto const cpu = resource("cpu:" + worker.host, worker.host, ResourceKind::cpu, worker.logical_cpus);
        auto const memory = resource("memory:" + worker.host, worker.host, ResourceKind::host_memory, worker.total_host_bytes);
        ComputeWorkerEnvelope selected;
        selected.tls_identity = worker.identity; selected.host = worker.host;
        selected.cpu_threads = worker.cpus; selected.slots = worker.slots; selected.host_bytes = worker.host_bytes;
        selected.standing_host_bytes = worker.standing_host_bytes; selected.allowed_kinds = worker.capabilities;
        Demand cpu_demand; cpu_demand.resource = cpu; cpu_demand.reserved_units = gpu ? worker.logical_cpus : worker.cpus; cpu_demand.bounded = true;
        Demand memory_demand; memory_demand.resource = memory; memory_demand.resident_units = worker.standing_host_bytes;
        memory_demand.transient_units = worker.host_bytes - worker.standing_host_bytes; memory_demand.bounded = true;
        selected.reservation = {cpu_demand, memory_demand};
        if (gpu)
        {
            selected.gpu_identity = "host:" + worker.host + '/' + worker.gpu_identity;
            selected.gpu_bytes = worker.gpu_bytes; selected.standing_gpu_bytes = worker.total_gpu_bytes - worker.gpu_bytes;
            auto const compute = selected.gpu_identity + "/compute", vram = selected.gpu_identity + "/memory";
            resource(compute, worker.host, ResourceKind::gpu_compute, 1);
            resource(vram, worker.host, ResourceKind::gpu_memory, worker.total_gpu_bytes);
            Demand compute_demand; compute_demand.resource = compute; compute_demand.reserved_units = 1; compute_demand.bounded = true;
            Demand gpu_demand; gpu_demand.resource = vram; gpu_demand.resident_units = selected.standing_gpu_bytes;
            gpu_demand.transient_units = selected.gpu_bytes; gpu_demand.bounded = true;
            selected.reservation.push_back(compute_demand); selected.reservation.push_back(gpu_demand);
        }
        profile.native.compute_workers.push_back(std::move(selected));
    }
    for (auto const& broker : observed.broker_resources)
        preparation.facts.push_back({"compute.broker[" + broker.identity + ']', broker.host, "observed", "authenticated-broker",
            "Authenticated broker host; a saved service reservation is still required for its configured connection/task limits."});
    for (auto const& issue : observed.issues)
        preparation.missing.push_back({"compute", "compute.discovery", issue.reason});
    validate(profile); return preparation;
}
inline ComputeWorkerEnvelope const& reviewed_compute_worker(Profile const& profile, std::string const& identity)
{
    auto const at = std::find_if(profile.native.compute_workers.begin(), profile.native.compute_workers.end(),
        [&](ComputeWorkerEnvelope const& worker) { return worker.tls_identity == identity; });
    if (at == profile.native.compute_workers.end()) throw WorkerAdmissionRejected("selected compute worker has no reviewed physical service reservation");
    return *at;
}
inline void require_compute_task(Profile const& profile, WorkerTaskAdmission const& admitted,
    uint64_t codec_task_bytes, uint64_t codec_output_bytes, std::string const& backend)
{
    auto reject = [](bool condition, char const* message) { if (!condition) throw WorkerAdmissionRejected(message); };
    auto const& status = admitted.worker; auto const& task = admitted.task;
    auto const& worker = reviewed_compute_worker(profile, status.identity);
    reject(admitted.owner_identity == profile.native.compute_owner_identity && !admitted.owner_id.empty() &&
        (admitted.local_fallback || admitted.broker_identity == profile.native.compute_broker_identity),
        "compute task owner or broker differs from the reviewed authenticated scope");
    reject(!status.instance.empty() && status.capacity.cpu_threads && status.capacity.slots && status.capacity.host_bytes &&
        status.capacity.cpu_threads <= worker.cpu_threads && status.capacity.slots <= worker.slots &&
        status.baseline_host_bytes <= worker.standing_host_bytes &&
        resource_detail::add(status.capacity.host_bytes, status.baseline_host_bytes) <= worker.host_bytes &&
        status.capacity.gpu_bytes <= worker.gpu_bytes && status.active.cpu_threads <= status.capacity.cpu_threads &&
        status.active.slots <= status.capacity.slots && status.active.host_bytes <= status.capacity.host_bytes &&
        status.active.gpu_bytes <= status.capacity.gpu_bytes,
        "compute executor capacity or standing residency exceeds its reviewed whole-service reservation");
    reject(task.kind == pool::Kind::GenerateCpu || task.kind == pool::Kind::GenerateGpu || task.kind == pool::Kind::Pack,
        "compute grant has no concrete supported capability");
    reject((worker.allowed_kinds & pool::capability(task.kind)) && task.k == profile.native.k &&
        task.strength == (task.kind == pool::Kind::Pack ? 2 : profile.native.strength),
        "compute grant capability or plot parameters differ from the reviewed construction");
    reject(task.resources.cpu_threads && task.resources.slots && task.resources.host_bytes &&
        task.resources.cpu_threads <= worker.cpu_threads && task.resources.slots <= worker.slots &&
        task.resources.host_bytes <= worker.host_bytes - status.baseline_host_bytes && task.resources.gpu_bytes <= worker.gpu_bytes,
        "compute task grant exceeds its reviewed executor capacity");
    auto const& configured = status.hardware.partition;
    if (configured.cpu_threads)
        reject(task.resources.cpu_threads <= configured.cpu_threads && task.resources.slots <= configured.slots &&
            configured.host_bytes > status.baseline_host_bytes && task.resources.host_bytes <= configured.host_bytes - status.baseline_host_bytes &&
            task.resources.gpu_bytes <= configured.gpu_bytes, "compute task grant exceeds its authenticated configured executor capacity");
    if (task.kind == pool::Kind::Pack)
        reject(codec_output_bytes && task.work_units && task.work_units <= codec_task_bytes && task.chunk_count &&
            task.chunk_count <= pool::default_max_codec_chunks && task.max_output_bytes && task.max_output_bytes <= codec_output_bytes &&
            !task.resources.gpu_bytes, "compute codec grant exceeds the reviewed owner transport partition");
    else
        reject(task.work_units && task.work_units <= profile.native.max_batch_members && !task.chunk_count && !task.max_output_bytes &&
            (task.kind == pool::Kind::GenerateCpu ? backend != "gpu" && !task.resources.gpu_bytes : backend != "cpu" && task.resources.gpu_bytes),
            "compute generation grant exceeds reviewed member count or backend selection");
    if (task.resources.cpu_threads > status.capacity.cpu_threads || task.resources.slots > status.capacity.slots ||
        task.resources.host_bytes > status.capacity.host_bytes || task.resources.gpu_bytes > status.capacity.gpu_bytes)
        throw WorkerResourcesUnavailable("compute task is waiting for the authenticated executor's current resources to recover");
}
// The immutable profile must outlive the bound session and every backend/client.
// Admission of all background reservations precedes this binding and service
// construction. These callbacks only tighten that admitted physical envelope.
inline void bind_replot_compute(pos2gpu::GroupOptions& options, Profile const& profile)
{
    using namespace resource_detail;
    validate(profile);
    if (options.pool_broker.empty())
    {
        require(!options.automatic_compute && !options.compute_session && !options.pool_offer_ram && !options.pool_codec_ram,
            "compute service partitions have no reviewed broker");
        return;
    }
    auto const identity = pool::certificate_identity(options.pool_cert);
    pool::TlsOptions credentials; credentials.ca_file = options.pool_ca; credentials.cert_file = options.pool_cert;
    credentials.key_file = options.pool_key; credentials.timeout_ms = unsigned(options.pool_socket_timeout_ms);
    credentials.max_frame_bytes = uint32_t(options.pool_frame_bytes); pool::validate_tls_credentials(credentials);
    require(identity == profile.native.compute_owner_identity && !profile.native.compute_workers.empty() &&
        !profile.native.compute_broker_identity.empty() && !profile.native.compute_broker_host.empty(),
        "compute credentials differ from the reviewed owner/broker/worker scope");
    require(!options.compute_session, "bind reviewed compute before starting an automatic session");
    uint64_t cpu_minimum = 0;
    if (options.backend == "cpu")
    {
        cpu_minimum = worker_cpu_task_bytes(uint8_t(profile.native.k), uint8_t(profile.native.strength), 1,
            add(options.window, options.pool_frame_bytes));
        require(std::any_of(profile.native.compute_workers.begin(), profile.native.compute_workers.end(), [&](auto const& worker)
            { return (worker.allowed_kinds & compute_generate_cpu) && worker.host_bytes - worker.standing_host_bytes >= cpu_minimum; }),
            "reviewed compute workers cannot hold one selected CPU generation task after standing residency");
    }
    options.compute_admit_session = [&profile, identity, automatic = options.automatic_compute, local_broker = options.pool_broker == "auto"]
    {
        require(automatic && (!local_broker || (profile.native.compute_broker_identity == identity && profile.native.compute_broker_host == profile.native.host)),
            "automatic compute broker differs from its reviewed local physical service scope");
    };
    options.compute_admit_worker = [&profile, cpu_minimum](WorkerConfig const& config)
    {
        auto const& worker = reviewed_compute_worker(profile, pool::certificate_identity(config.tls.cert_file));
        auto reject = [](bool value, char const* message) { if (!value) throw WorkerAdmissionRejected(message); };
        reject(worker.host == profile.native.host && config.capacity.cpu_threads <= worker.cpu_threads && config.capacity.slots <= worker.slots &&
            config.capacity.host_bytes <= worker.host_bytes && config.capacity.gpu_bytes <= worker.gpu_bytes,
            "local compute offer exceeds its reviewed physical whole-service capacity");
        auto const baseline = worker_service_baseline_bytes(config);
        reject(baseline < config.capacity.host_bytes && baseline <= worker.standing_host_bytes &&
            (!cpu_minimum || config.capacity.host_bytes - baseline >= cpu_minimum),
            "local compute startup transport or runtime residency exceeds its reviewed standing host bound");
        if (config.backend != "cpu" && config.capacity.gpu_bytes)
        {
            // Runtime discovery can create helpers before task CpuAdmission.
            // Their whole inherited CPU scope remains reserved for this local
            // GPU service, just as for native GPU construction.
            resource_detail::Graph graph(profile); uint64_t reserved = 0;
            for (auto const& demand : worker.reservation)
            {
                auto const& resource = graph.resources[graph.find(demand.resource)];
                if (resource.kind == ResourceKind::cpu && resource.host == worker.host)
                    reserved = resource_detail::add(reserved, demand.reserved_units);
            }
            auto const inherited = profile_detail::logical_cpus();
            reject(inherited && reserved >= inherited, "local GPU compute omits its inherited runtime-helper CPU scope");
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            auto const physical = pos2gpu::gpu_physical_identity(config.device);
            reject(!physical.empty() && worker.gpu_identity == "host:" + worker.host + '/' + physical,
                "local compute device differs from its reviewed physical GPU identity");
#else
            throw WorkerAdmissionRejected("this build cannot bind a reviewed local GPU compute offer");
#endif
        }
    };
    options.compute_admit_task = [&profile, codec_task = add(options.window, options.pool_frame_bytes),
        codec_output = options.pool_codec_ram, backend = options.backend](WorkerTaskAdmission const& task)
    { require_compute_task(profile, task, codec_task, codec_output, backend); };
    if (options.pool_offer_ram)
    {
        // Resolve automatic device defaults from the admitted declaration before
        // default_worker_resources can probe an otherwise unreserved GPU.
        auto const& worker = reviewed_compute_worker(profile, identity);
        if (options.pool_offer_backend != "cpu" && !options.pool_offer_vram)
        {
            if (worker.gpu_bytes) options.pool_offer_vram = worker.gpu_bytes;
            else if (options.pool_offer_backend == "auto") options.pool_offer_backend = "cpu";
            else throw WorkerAdmissionRejected("explicit local GPU offer has no reviewed GPU allocation cap");
        }
        auto const finalized = pos2gpu::make_group_compute_offer(options);
        options.compute_admit_worker(finalized);
    }
}
} // namespace group_grow::farm
