#include "host/ComputeSession.hpp"
#include "host/ComputeDiscovery.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/ComputeWorker.hpp"
#include "host/GroupHarvesterProfile.hpp"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

namespace pos2gpu
{
namespace
{
namespace pool = group_grow::pool;
void require(bool value, char const* message)
{ if (!value) throw std::invalid_argument(message); }
std::string endpoint_text(pool::Endpoint const& endpoint)
{ return (endpoint.host.find(':') == endpoint.host.npos ? endpoint.host : '[' + endpoint.host + ']') + ':' + std::to_string(endpoint.port); }
pool::DiscoveryOptions discovery_options(pool::PoolIdentity const& identity)
{
    pool::DiscoveryOptions result; result.pool_id = identity.pool_id; result.node_id = identity.node_id;
    result.name = identity.name; result.server_name = identity.server_name; result.interfaces = identity.defaults.interfaces;
    return result;
}
pool::PoolIdentity required_pool(std::filesystem::path const& path)
{
    auto identity = pool::load_pool(path); require(bool(identity), "saved pool membership disappeared"); return std::move(*identity);
}
void prepare_automatic_offer(pool::PoolIdentity const& identity, group_grow::WorkerConfig& config,
    std::function<std::vector<pool::Endpoint>()> discover)
{
    config.server_name = identity.server_name; config.max_brokers = identity.defaults.max_brokers;
    config.tls.listen_lan = identity.defaults.interfaces.empty(); config.tls.listen_interfaces = identity.defaults.interfaces;
    config.tls.listen_interfaces.push_back("lo");
    if (identity.defaults.discovery) config.discover_brokers = std::move(discover);
    group_grow::fit_worker_broker_budget(config); group_grow::default_worker_resources(config);
}
} // namespace

group_grow::WorkerConfig make_group_compute_offer(GroupOptions const& args, std::shared_ptr<group_grow::CodecScheduler> scheduler)
{
    require(!args.pool_broker.empty() && args.pool_offer_ram && args.pool_offer_cpus, "local compute offer has no bounded broker, CPU or RAM scope");
    group_grow::WorkerConfig config;
    // Only the presence of a broker route affects readonly transport admission.
    // The automatic session replaces this placeholder with its actual endpoint.
    config.broker = args.pool_broker == "auto" ? pool::Endpoint{"127.0.0.1", 1} : pool::parse_endpoint(args.pool_broker);
    config.broker.server_name = args.pool_server_name;
    config.tls.ca_file = args.pool_ca; config.tls.cert_file = args.pool_cert; config.tls.key_file = args.pool_key;
    if (args.automatic_compute) config.tls.membership_directory = args.pool_directory;
    config.tls.timeout_ms = unsigned(args.pool_socket_timeout_ms); config.tls.max_frame_bytes = uint32_t(args.pool_frame_bytes);
    config.listen = args.pool_listen.empty() ? pool::Endpoint{"::", 0} : pool::parse_endpoint(args.pool_listen);
    if (!args.pool_advertise.empty()) config.advertise = pool::parse_endpoint(args.pool_advertise);
    config.capacity = {uint32_t(args.pool_offer_cpus), uint32_t(args.pool_offer_slots), args.pool_offer_ram, args.pool_offer_vram};
    config.backend = args.pool_offer_backend; config.tier = args.tier; config.device = int(args.device);
    config.task_timeout_ms = unsigned(args.pool_timeout_ms);
    config.identity_reservation = args.compute_worker_reservation;
    // Protected migration reserves the complete executor independently of its
    // owner/client/codec partition. Reuse the worker's own bounded CPU gate;
    // ordinary jobs retain their established shared owner scheduler.
    config.cpu_scheduler = args.protected_resources ? nullptr : std::move(scheduler);
    require(args.pool_frame_bytes <= UINT64_MAX - args.window, "local compute transport bound overflows");
    config.max_task_bytes = args.window + args.pool_frame_bytes; config.admit_start = args.compute_admit_worker;
    if (args.automatic_compute)
    {
        auto const identity = required_pool(std::filesystem::path(reinterpret_cast<char8_t const*>(args.pool_directory.c_str())));
        prepare_automatic_offer(identity, config, [] { return std::vector<pool::Endpoint>{}; });
    }
    else group_grow::default_worker_resources(config);
    return config;
}

bool saved_compute_available(std::filesystem::path const& directory)
{ return bool(pool::load_pool(directory.empty() ? pool::pool_directory() : directory)); }
void prepare_group_compute(GroupOptions& args)
{
    require(args.compute_mode == "auto" || args.compute_mode == "off", "--compute requires auto or off");
    if (args.automatic_compute) return;
    if (!args.local_devices.empty())
    {
        require(args.pool_broker.empty() && args.pool_ca.empty() && args.pool_cert.empty() && args.pool_key.empty() &&
            args.pool_listen.empty() && args.pool_advertise.empty() && args.pool_directory.empty() && args.automatic_local_compute &&
            !args.pool_offer_ram && !args.pool_offer_cpus && !args.pool_offer_vram && !args.pool_offer_backend_set,
            "selected local producers cannot also use shared compute connection or offer options");
        args.local_compute = true;
        auto const workers = args.workers ? unsigned(args.workers) : std::max(1u, group_grow::profile_detail::logical_cpus());
        args.pool_offer_cpus = std::max(1u, workers - 1);
        // The complete executor partition includes every service baseline,
        // retained device buffer and active task. Receiving/assembly keep the
        // remaining quarter, with their existing per-task memory admission.
        args.pool_offer_ram = args.max_ram / 4 * 3;
        args.pool_offer_slots = args.local_devices.size();
        args.pool_codec_auto = false; args.pool_codec_ram = 0;
        return;
    }
    if (args.compute_mode == "off")
    {
        require(args.pool_broker.empty() && args.pool_ca.empty() && args.pool_cert.empty() && args.pool_key.empty() &&
            !args.pool_offer_ram && !args.pool_offer_cpus && !args.pool_offer_vram,
            "--compute off cannot be combined with connection or compute offer options");
        return;
    }
    // Explicit credentials retain the existing direct-connection interface.
    if (!args.pool_ca.empty() || !args.pool_cert.empty() || !args.pool_key.empty()) return;
    auto identity = pool::load_pool(args.pool_directory.empty() ? pool::pool_directory() :
        std::filesystem::path(reinterpret_cast<char8_t const*>(args.pool_directory.c_str())));
    if (!identity)
    {
        require(args.pool_broker.empty(), "shared compute needs saved pool membership; use compute init or compute join");
        require(args.automatic_local_compute, "--compute-local needs saved pool membership; use compute init or compute join");
        return;
    }
    args.automatic_compute = true;
    auto const directory = identity->directory.u8string();
    args.pool_directory.assign(reinterpret_cast<char const*>(directory.data()), directory.size());
    args.pool_ca = identity->tls.ca_file; args.pool_cert = identity->tls.cert_file; args.pool_key = identity->tls.key_file;
    args.pool_server_name = identity->server_name;
    if (args.pool_owner.empty()) args.pool_owner = identity->node_id;
    if (args.pool_broker.empty()) args.pool_broker = identity->defaults.broker.empty() ? "auto" : identity->defaults.broker;
    require(args.max_ram > ComputeSession::memory_bytes + (32ULL << 20),
        "automatic compute needs more than 160 MiB of host RAM; increase --max-host-ram or use --compute off");
    args.max_ram -= ComputeSession::memory_bytes;
    if (!args.automatic_local_compute)
    {
        require(!args.pool_offer_ram && !args.pool_offer_cpus && !args.pool_offer_vram && args.pool_listen.empty() && args.pool_advertise.empty(),
            "--compute-local off cannot be combined with local compute offer options");
        return;
    }
    if (!args.pool_offer_ram) args.pool_offer_ram = std::max(args.max_ram / 2, std::min(uint64_t(256) << 20, args.max_ram / 4 * 3));
    auto limit = [](uint64_t selected, uint64_t saved) { return !selected ? saved : !saved ? selected : std::min(selected, saved); };
    args.pool_offer_ram_limit = limit(args.pool_offer_ram_limit, identity->defaults.host_bytes);
    if (args.pool_offer_ram_limit) args.pool_offer_ram = std::min(args.pool_offer_ram, args.pool_offer_ram_limit);
    auto const workers = args.workers ? unsigned(args.workers) : std::max(1u, group_grow::profile_detail::logical_cpus());
    if (!args.pool_offer_cpus) args.pool_offer_cpus = std::max(1u, workers - 1);
    args.pool_offer_cpu_limit = limit(args.pool_offer_cpu_limit, identity->defaults.cpus);
    if (args.pool_offer_cpu_limit) args.pool_offer_cpus = std::min(args.pool_offer_cpus, args.pool_offer_cpu_limit);
    if (!args.pool_offer_vram) args.pool_offer_vram = identity->defaults.gpu_bytes;
    args.pool_offer_vram_limit = limit(args.pool_offer_vram_limit, identity->defaults.gpu_bytes);
    if (!args.pool_offer_backend_set) args.pool_offer_backend = args.backend_set ? args.backend : identity->defaults.backend;
}

ComputeInspection inspect_group_compute(GroupOptions const& args, std::chrono::steady_clock::time_point deadline)
{
    using Clock = std::chrono::steady_clock;
    ComputeInspection result;
    auto const now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (now >= 0) result.observed_at_unix_ms = uint64_t(now);
    result.unknowns = "exact plot parameters, resource fit and new lease admission are not tested; broker pages and direct worker observations may advance during inspection";
    result.next_action = "use compute status --verbose to inspect authenticated task wait reasons";
    auto expired = [&] { return Clock::now() >= deadline || (args.cancelled && args.cancelled()); };
    auto issue = [&](std::string endpoint, std::string reason)
    {
        result.complete = false;
        if (result.issues.size() < 128) result.issues.push_back({std::move(endpoint), std::move(reason)});
    };
    try
    {
        if (args.local_compute || !args.local_devices.empty())
        {
            result.status = "local-execution-selected";
            result.reason = "explicit local producers are selected; status does not initialize them or consult unrelated shared enrollment";
            result.unknowns = "current selected-device and exact task admission are not tested; saved farm progress does not prove a running local producer";
            result.next_action = "review plan with the selected devices, then continue fill when intended; execution rechecks device and task admission";
            return result;
        }
        if (args.compute_mode == "off")
        {
            result.status = "disabled"; result.reason = "shared compute is disabled for this command";
            result.next_action = "continue with local execution, or select --compute auto when shared compute is intended"; return result;
        }
        bool const explicit_tls = !args.automatic_compute &&
            (!args.pool_ca.empty() || !args.pool_cert.empty() || !args.pool_key.empty());
        auto identity = explicit_tls ? std::optional<pool::PoolIdentity>{} : pool::load_pool(args.pool_directory.empty() ? pool::pool_directory() :
            std::filesystem::path(reinterpret_cast<char8_t const*>(args.pool_directory.c_str())));
        result.configured = bool(identity) || explicit_tls;
        if (!result.configured)
        {
            result.status = "not-configured"; result.reason = "no saved compute enrollment or explicit connection was selected";
            result.next_action = "continue with local execution, or use compute init or compute join to enroll this machine"; return result;
        }
#ifdef _WIN32
        result.service_lifecycle = "background compute uses Task Scheduler for the logged-in user; changed defaults, executable or imported credentials require compute service restart; manager state was not queried";
#else
        result.service_lifecycle = "background compute requires an active systemd user manager and follows its login policy; changed defaults, executable or imported credentials require compute service restart; manager state was not queried";
#endif
        pool::TlsOptions tls;
        if (!explicit_tls && identity)
        {
            tls = identity->tls; result.certificate_expired = identity->certificate_expired;
            result.authority_expired = identity->authority_expired; result.certificate_expires = identity->certificate_expires;
            result.authority_expires = identity->authority_expires;
            auto const intent = pool::load_worker_intent(*identity);
            result.local_worker_intent = intent == pool::WorkerIntent::Paused ? "paused" : intent == pool::WorkerIntent::Draining ? "draining" : "active";
            result.local_worker_intent_action = intent == pool::WorkerIntent::Active ?
                "saved local controls allow new tasks; this setting alone does not show whether a worker is running" :
                "saved pause/drain settings survive service restarts; use compute resume on the intended worker to accept new tasks";
            if (identity->authority_expired || identity->certificate_expired)
            {
                result.status = identity->authority_expired ? "authority-expired" : "renewal-required";
                result.reason = identity->authority_expired ? "the saved pool authority has expired" : "this machine's enrolled certificate has expired";
                result.blocking = true;
                result.next_action = identity->authority_expired ? "create a new pool and re-enroll its members" :
                    "use compute renew-request, have the authority sign it with compute renew-sign, then use compute renew-import and restart active compute services";
                return result;
            }
        }
        else
        {
            tls.ca_file = args.pool_ca; tls.cert_file = args.pool_cert; tls.key_file = args.pool_key;
        }
        tls.read_only_membership = true; tls.max_frame_bytes = 65536; tls.deadline = deadline;
        tls.timeout_ms = 1000;
        require(args.max_ram > pool::BrokerClient::memory_bytes(1) + (4ULL << 20),
            "compute inspection transport and bounded records exceed the available host memory");
        std::vector<pool::Endpoint> brokers, direct_workers;
        auto route = args.pool_broker;
        if (route.empty() && identity) route = identity->defaults.broker;
        if (!route.empty() && route != "auto") brokers.push_back(pool::parse_endpoint(route));
        result.discovery = "not requested; explicit broker selected";
        if (brokers.empty() && identity && identity->defaults.discovery && !expired())
        {
            pool::ComputeDiscovery discovery(discovery_options(*identity));
            auto const until = std::min(deadline, Clock::now() + std::chrono::seconds(2));
            while (!expired() && Clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            brokers = discovery.brokers(); result.discovery = discovery.status();
            for (auto const& service : discovery.services()) if (service.role == "worker")
            {
                if (direct_workers.size() == 128) { issue("", "worker discovery exceeded the bounded endpoint listing"); break; }
                direct_workers.push_back(service.endpoint);
            }
        }
        else if (brokers.empty()) result.discovery = "disabled in saved compute settings";
        if (brokers.size() > 128) { brokers.resize(128); issue("", "broker discovery exceeded the bounded endpoint listing"); }
        std::set<std::string> endpoints, workers;
        bool generation_compatible = false, codec_compatible = false;
        auto const generation_contract = group_grow::worker_compatibility(pool::Kind::GenerateAny);
        auto const codec_contract = group_grow::worker_compatibility(pool::Kind::Pack);
        require(!generation_contract.empty() && !codec_contract.empty(), "this build has no shared compute contract identities");
        auto record_worker = [&](auto const& worker, pool::WorkerHardware const& hardware, uint64_t capabilities, uint64_t age_ms)
        {
            if (workers.size() >= 128 && !workers.contains(worker.identity + ':' + worker.instance))
            { issue("", "worker listing exceeded the bounded combined row limit"); return; }
            result.oldest_worker_registration_age_ms = std::max(result.oldest_worker_registration_age_ms, age_ms);
            if (!workers.insert(worker.identity + ':' + worker.instance).second) return;
            ++result.workers;
            result.worker_resources.push_back({worker.identity, worker.instance, hardware.host, hardware.gpu_identity,
                pool::admission_name(worker.admission), hardware.partition.cpu_threads, hardware.partition.slots,
                hardware.partition.host_bytes, hardware.partition.gpu_bytes, hardware.standing_host_bytes,
                hardware.logical_cpus, hardware.total_host_bytes, hardware.available_host_bytes, hardware.total_gpu_bytes, capabilities,
                hardware.observed_at_unix_ms});
            bool const offers_generation = capabilities & (pool::capability(pool::Kind::GenerateCpu) | pool::capability(pool::Kind::GenerateGpu));
            bool const offers_codec = capabilities & pool::capability(pool::Kind::Pack);
            bool const generation = offers_generation && worker.compatibility == generation_contract;
            bool const codec = offers_codec && worker.codec_compatibility == codec_contract;
            result.worker_resources.back().capabilities =
                (generation ? capabilities & (pool::capability(pool::Kind::GenerateCpu) | pool::capability(pool::Kind::GenerateGpu)) : 0) |
                (codec ? pool::capability(pool::Kind::Pack) : 0);
            generation_compatible = generation_compatible || generation; codec_compatible = codec_compatible || codec;
            if ((offers_generation && !generation) || (offers_codec && !codec)) ++result.incompatible;
            if (worker.admission == pool::WorkerAdmission::Accepting)
            { if (generation) ++result.accepting_generation; if (codec) ++result.accepting_codec; }
            else if (worker.admission == pool::WorkerAdmission::Paused) ++result.paused;
            else if (worker.admission == pool::WorkerAdmission::Draining) ++result.draining;
            else if (worker.admission == pool::WorkerAdmission::Drained) ++result.drained;
        };
        for (auto endpoint : brokers)
        {
            auto const address = endpoint_text(endpoint);
            if (!endpoints.insert(address).second) continue;
            if (expired()) { issue(address, "overall live inspection time budget exhausted"); break; }
            if (!pool::numeric_host(endpoint.host))
            {
                issue(address, "hostname resolution is not deadline bounded; select a numeric broker route with its enrolled TLS server name"); continue;
            }
            endpoint.server_name = args.pool_server_name.empty() && identity ? identity->server_name : args.pool_server_name;
            try
            {
                pool::BrokerClient base(endpoint, tls, 1); auto client = base.with_deadline(deadline);
                uint64_t cursor = 0; unsigned remaining = 128;
                do
                {
                    if (expired()) { issue(address, "overall live inspection time budget exhausted"); break; }
                    auto page = client.status({pool::StatusSection::Workers, cursor, std::min(remaining, pool::max_status_page_size)});
                    if (!cursor)
                    {
                        ++result.brokers; result.checked = true;
                        result.broker_resources.push_back({page.identity, page.host});
                    }
                    require(page.worker_records.size() <= remaining && (!page.next_cursor || page.next_cursor > cursor),
                        "broker inspection exceeded its bounded status page");
                    remaining -= unsigned(page.worker_records.size()); cursor = page.next_cursor;
                    for (auto const& worker : page.worker_records)
                    {
                        record_worker(worker, worker.hardware, worker.capabilities, worker.seen_ms);
                    }
                    if (cursor && !remaining) issue(address, "worker listing truncated at the bounded row limit");
                } while (cursor && remaining);
            }
            catch (std::exception const& error) { issue(address, error.what()); }
        }
        endpoints.clear();
        for (auto endpoint : direct_workers)
        {
            auto const address = endpoint_text(endpoint);
            if (!endpoints.insert(address).second) continue;
            if (expired()) { issue(address, "overall live inspection time budget exhausted"); break; }
            if (!pool::numeric_host(endpoint.host))
            { issue(address, "discovered worker has no numeric route"); continue; }
            endpoint.server_name = args.pool_server_name.empty() && identity ? identity->server_name : args.pool_server_name;
            try
            {
                auto worker = group_grow::WorkerService::remote_status(endpoint, tls);
                result.checked = true;
                uint64_t capabilities = pool::capability(pool::Kind::GenerateCpu) | pool::capability(pool::Kind::Pack);
                if (worker.hardware.partition.gpu_bytes) capabilities |= pool::capability(pool::Kind::GenerateGpu);
                record_worker(worker, worker.hardware, capabilities, 0);
            }
            catch (std::exception const& error) { issue(address, error.what()); }
        }
        if (!result.checked)
        {
            result.status = "unavailable"; result.complete = false;
            result.reason = brokers.empty() ? "no running broker or worker route was observed; a future automatic session is not current availability" :
                "no selected broker supplied an authenticated live observation";
            result.next_action = "use compute service status and inspect the saved broker route, enrollment and network; retry status --live";
        }
        else if (!result.complete)
        {
            result.status = "partial"; result.reason = "the bounded live observation is incomplete; unobserved capacity remains unknown";
        }
        else if (!result.workers)
        {
            result.status = "no-workers"; result.blocking = true;
            result.reason = "the authenticated brokers currently report no registered workers";
            result.next_action = "inspect compute service status on the enrolled worker machines and restore the intended worker connection";
        }
        else if (!generation_compatible || !codec_compatible)
        {
            result.status = result.incompatible ? "incompatible-workers" : "work-capability-unavailable"; result.blocking = true;
            result.reason = "the observed worker capabilities and contracts do not cover this build's generation and encoding work";
            result.next_action = "use compatible worker generation and codec contracts, then retry status --live; producer build differences alone do not block compatible contracts";
        }
        else if (!result.accepting_generation || !result.accepting_codec)
        {
            result.status = "admission-closed"; result.blocking = true;
            result.reason = "compatible workers were observed, but pause/drain settings leave generation or encoding tasks without an accepting worker";
            result.next_action = "use compute resume on the intended worker machine when it should accept new work; accepted tasks keep their reservations";
        }
        else
        {
            result.status = "compatible-workers-observed";
            result.reason = "authenticated observations report accepting workers with compatible generation and codec contracts";
        }
    }
    catch (std::exception const& error)
    {
        result.status = "unavailable"; result.complete = false; result.reason = error.what();
        result.next_action = "inspect compute status and the selected local enrollment, then retry status --live";
    }
    return result;
}

struct ComputeSession::Impl
{
    pool::PoolIdentity identity;
    std::atomic<bool> stopping{false};
    std::unique_ptr<pool::BrokerServer> broker;
    std::jthread broker_thread;
    std::thread worker_thread;
    std::shared_ptr<pool::ComputeDiscovery> discovery;
    std::unique_ptr<pool::ComputeDiscovery> worker_advertisement;
    std::shared_ptr<group_grow::WorkerService> worker;
    uint64_t owned_worker_bytes = 0;
    mutable std::mutex mutex;
    std::condition_variable changed;
    bool offering = false, verbose = false;
    explicit Impl(GroupOptions& args) : identity(required_pool(
        std::filesystem::path(reinterpret_cast<char8_t const*>(args.pool_directory.c_str())))), verbose(args.verbose)
    {
        require(!args.protected_resources || bool(args.compute_admit_session), "protected compute session has no reviewed service admission");
        if (args.compute_admit_session) args.compute_admit_session();
        if (!args.pool_owner_explicit)
            args.pool_session_owner = args.pool_owner + '-' + pool::random_token().substr(0, 32);
        if (args.pool_broker == "auto")
        {
            pool::BrokerOptions options; options.listen = {"::", 0}; options.tls = identity.tls;
            options.tls.timeout_ms = 1000; options.tls.max_frame_bytes = 65536;
            options.tls.listen_lan = identity.defaults.interfaces.empty();
            options.tls.listen_interfaces = identity.defaults.interfaces;
            options.tls.listen_interfaces.push_back("lo");
            // The automatic service fits within its explicit RAM partition.
            // Larger dedicated pools can configure their own registry bounds.
            options.max_connections = 32; options.max_workers = 16; options.max_owners = 32; options.max_tasks = 1024;
            broker = std::make_unique<pool::BrokerServer>(std::move(options));
            auto address = broker->endpoint();
            broker_thread = std::jthread([this](std::stop_token token)
            { std::stop_callback stop(token, [this] { stopping.store(true); }); broker->serve(stopping); });
            args.pool_broker = endpoint_text({"127.0.0.1", address.port});
            if (identity.defaults.discovery)
            {
                auto discover = discovery_options(identity); discover.role = "broker"; discover.endpoint = address;
                discovery = std::make_shared<pool::ComputeDiscovery>(std::move(discover));
            }
            if (verbose) std::cerr << "[compute] Automatic broker on port " << address.port << '\n';
        }
        else
        {
            if (identity.defaults.discovery) discovery = std::make_shared<pool::ComputeDiscovery>(discovery_options(identity));
            if (verbose) std::cerr << "[compute] Using broker " << args.pool_broker << '\n';
        }
        if (discovery && discovery->status() != "LAN discovery active")
            std::cerr << "[compute] " << discovery->status() << '\n';
    }
    ~Impl()
    {
        stopping.store(true); changed.notify_all();
        if (worker_thread.joinable()) worker_thread.join();
        worker_advertisement.reset(); worker.reset(); discovery.reset();
        { std::lock_guard lock(mutex); owned_worker_bytes = 0; }
        if (broker_thread.joinable()) broker_thread.join();
    }
    bool start_worker(group_grow::WorkerConfig config)
    {
        try { if (!config.identity_reservation) config.identity_reservation = group_grow::WorkerService::reserve_identity(config.tls.cert_file); }
        catch (group_grow::WorkerAlreadyRunning const&) { return false; }
        group_grow::default_worker_resources(config);
        // capacity.host_bytes already includes the complete service baseline.
        // Publish ownership before its constructor allocates transport/GPU
        // buffers, but never when a foreign worker holds this identity.
        require(group_grow::worker_service_baseline_bytes(config) < config.capacity.host_bytes,
            "automatic worker has no host capacity after its service baseline");
        { std::lock_guard lock(mutex); owned_worker_bytes = config.capacity.host_bytes; }
        try
        {
            auto candidate = std::make_shared<group_grow::WorkerService>(std::move(config)); candidate->start();
            if (identity.defaults.discovery)
            {
                auto advertise = discovery_options(identity); advertise.role = "worker"; advertise.endpoint = candidate->endpoint();
                worker_advertisement = std::make_unique<pool::ComputeDiscovery>(std::move(advertise));
            }
            { std::lock_guard lock(mutex); worker = std::move(candidate); }
        }
        catch (...) { std::lock_guard lock(mutex); owned_worker_bytes = 0; throw; }
        if (verbose) std::cerr << "[compute] Local compute is available to this pool while plotting\n";
        return true;
    }
    void offer(group_grow::WorkerConfig config)
    {
        if (offering) return;
        prepare_automatic_offer(identity, config, [browse = discovery] { return browse ? browse->brokers() : std::vector<pool::Endpoint>{}; });
        start_worker(config); offering = true;
        worker_thread = std::thread([this, config = std::move(config)]
        {
            std::unique_lock lock(mutex);
            while (!changed.wait_for(lock, std::chrono::seconds(1), [this] { return stopping.load(); }))
            {
                if (worker) continue;
                lock.unlock();
                try { start_worker(config); }
                catch (std::exception const& error)
                { std::cerr << "[compute] Local compute unavailable: " << error.what() << '\n'; }
                lock.lock();
            }
        });
    }
};
ComputeSession::ComputeSession(GroupOptions& args) : impl_(std::make_unique<Impl>(args)) {}
ComputeSession::~ComputeSession() = default;
void ComputeSession::offer_local(group_grow::WorkerConfig config) { impl_->offer(std::move(config)); }
std::shared_ptr<group_grow::WorkerService> ComputeSession::local_worker() const
{ std::lock_guard lock(impl_->mutex); return impl_->worker; }
uint64_t ComputeSession::owned_worker_bytes() const
{ std::lock_guard lock(impl_->mutex); return impl_->owned_worker_bytes; }
} // namespace pos2gpu
