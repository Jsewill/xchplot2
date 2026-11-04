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
        task.resources.cpu_threads <= status.capacity.cpu_threads && task.resources.slots <= status.capacity.slots &&
        task.resources.host_bytes <= status.capacity.host_bytes && task.resources.gpu_bytes <= status.capacity.gpu_bytes,
        "compute task grant exceeds its authenticated executor capacity");
    if (task.kind == pool::Kind::Pack)
        reject(codec_output_bytes && task.work_units && task.work_units <= codec_task_bytes && task.chunk_count &&
            task.chunk_count <= pool::default_max_codec_chunks && task.max_output_bytes && task.max_output_bytes <= codec_output_bytes &&
            !task.resources.gpu_bytes, "compute codec grant exceeds the reviewed owner transport partition");
    else
        reject(task.work_units && task.work_units <= profile.native.max_batch_members && !task.chunk_count && !task.max_output_bytes &&
            (task.kind == pool::Kind::GenerateCpu ? backend != "gpu" && !task.resources.gpu_bytes : backend != "cpu" && task.resources.gpu_bytes),
            "compute generation grant exceeds reviewed member count or backend selection");
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
