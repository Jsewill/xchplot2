#include "host/ComputeSession.hpp"
#include "host/ComputeDiscovery.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/ComputeWorker.hpp"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
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
    if (identity->defaults.host_bytes) args.pool_offer_ram = std::min(args.pool_offer_ram, identity->defaults.host_bytes);
    auto const workers = args.workers ? unsigned(args.workers) : std::max(1u, std::thread::hardware_concurrency());
    if (!args.pool_offer_cpus) args.pool_offer_cpus = std::max(1u, workers / 2);
    if (identity->defaults.cpus) args.pool_offer_cpus = std::min<uint64_t>(args.pool_offer_cpus, identity->defaults.cpus);
    if (!args.pool_offer_vram) args.pool_offer_vram = identity->defaults.gpu_bytes;
    if (!args.pool_offer_backend_set) args.pool_offer_backend = args.backend_set ? args.backend : identity->defaults.backend;
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
        try { config.identity_reservation = group_grow::WorkerService::reserve_identity(config.tls.cert_file); }
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
