#include "host/ComputeWorker.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/ComputeDiscovery.hpp"
#include "compute_output.hpp"

#include <csignal>
#include <iostream>
#include <thread>

#ifdef XCHPLOT2_GROUP_COMMANDS
#include "group_commands.hpp"
#endif

namespace
{
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
uint64_t number(char const* text, std::string const& option)
{
    std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != value.npos)
        throw std::runtime_error(option + " requires a nonnegative integer");
    try { return std::stoull(value); }
    catch (std::out_of_range const&) { throw std::runtime_error(option + " exceeds the maximum integer value (18446744073709551615)"); }
}
unsigned small(char const* value, std::string const& option)
{
    auto const result = number(value, option);
    if (result > UINT32_MAX) throw std::runtime_error(option + " must be no greater than 4294967295");
    return unsigned(result);
}
void help(bool all)
{
#ifndef XCHPLOT2_GROUP_COMMANDS
    all = true;
#endif
    std::cout <<
#ifdef XCHPLOT2_GROUP_COMMANDS
        "Usage: xchplot2 compute worker [OPTIONS]\n"
        "  Uses saved pool membership and discovers plotting hosts on the LAN.\n"
        "  --join INVITATION       enroll and run this foreground worker for active plotting jobs\n"
        "  --broker HOST:PORT      direct broker; repeat for additional brokers\n"
        "  --directory DIR         saved compute setup directory\n"
        "  --interface NAME        sharing/discovery interface; repeat for more\n"
        "  --discovery on|off      LAN discovery, default from saved setup\n"
        "  --listen HOST:PORT      override the saved worker listener port\n"
#else
        "Usage: pos2_pr118_grow_worker --broker HOST:PORT --listen HOST:PORT\n"
#endif
        "  --cpus N --host-bytes SIZE   offered CPU threads and host RAM limit\n"
        "  --harvester local        pause new tasks during slow lookups on this worker\n"
        "  --backend cpu|gpu|auto   offered generation backend, default auto with saved setup\n"
        "  --gpu-bytes N --device N --tier NAME   GPU RAM limit in bytes, device and tier\n";
    if (all) std::cout <<
        "  --advertise HOST:PORT     address reachable by plotting hosts\n"
        "  --ca FILE --cert FILE --key FILE  mutual TLS credentials\n"
        "  --slots N                maximum active tasks, default 1\n"
        "  --chia-root DIR          configuration of the selected local harvester\n"
        "  --farming-deadline-ms N  lookup deadline for the selected harvester\n"
        "  --generation-threads N   CPU threads per generation task, default CPUs / slots\n"
        "  --max-frame-bytes N      maximum network message size in bytes\n"
        "  --max-task-bytes N       codec window and temporary generation-data limit in bytes\n"
        "  --max-window-chunks N    maximum chunks per codec task\n"
        "  --max-batch-members N    maximum members per generation task\n"
        "  --task-timeout-ms N      maximum task duration in milliseconds\n"
        "  --tls-timeout-ms N       timeout for each TLS operation in milliseconds\n"
        "  --max-brokers N          maximum broker registrations, default 128\n";
    else std::cout << "Use --help-all for TLS, transport and task limits.\n";
    std::cout << pos2gpu::compute_output_help;
}
}
#ifdef XCHPLOT2_GROUP_COMMANDS
int pos2gpu::run_compute_worker(int argc, char** argv)
#else
int main(int argc, char** argv)
#endif
{
    try
    {
#ifndef XCHPLOT2_GROUP_COMMANDS
        // The reference worker is also a standalone executable.
        if (argc == 8 && std::string_view(argv[1]) == "compute" && std::string_view(argv[2]) == "worker" &&
            std::string_view(argv[3]) == "--config" && std::string_view(argv[5]) == "--gpu-recovery-worker")
            return group_grow::run_worker_gpu_child(argv[6], argv[7]);
#endif
        if (argc == 4 && std::string_view(argv[1]) == "--gpu-recovery-worker")
            return group_grow::run_worker_gpu_child(argv[2], argv[3]);
        group_grow::WorkerConfig config;
        std::filesystem::path directory, chia_root;
        bool observe_farming = false;
#ifdef XCHPLOT2_GROUP_COMMANDS
        std::filesystem::path invitation;
#endif
        std::vector<std::string> interfaces;
        bool discover = true, discovery_set = false, backend_set = false, brokers_set = false;
        pos2gpu::ComputeOutputOptions output;
        for (int i = 1; i < argc; ++i)
        {
            std::string option(argv[i]);
            if (option == "--help" || option == "-h" || option == "--help-all") { help(option == "--help-all"); return 0; }
            if (output.parse(option)) continue;
            if (option == "--no-discovery") { discover = false; discovery_set = true; continue; }
            if (i + 1 == argc) throw std::runtime_error("missing value for " + option);
            auto const value = argv[++i];
            if (option == "--broker") config.brokers.push_back(group_grow::pool::parse_endpoint(value));
            else if (option == "--directory") directory = std::filesystem::path(reinterpret_cast<char8_t const*>(value));
#ifdef XCHPLOT2_GROUP_COMMANDS
            else if (option == "--join")
            {
                if (!invitation.empty() || !*value) throw std::runtime_error("--join requires one invitation file");
                invitation = std::filesystem::path(reinterpret_cast<char8_t const*>(value));
            }
#endif
            else if (option == "--interface") interfaces.emplace_back(value);
            else if (option == "--discovery")
            {
                if (std::string_view(value) != "on" && std::string_view(value) != "off" && std::string_view(value) != "true" && std::string_view(value) != "false")
                    throw std::runtime_error("--discovery requires on or off");
                discover = std::string_view(value) == "on" || std::string_view(value) == "true"; discovery_set = true;
            }
            else if (option == "--listen") config.listen = group_grow::pool::parse_endpoint(value);
            else if (option == "--advertise") config.advertise = group_grow::pool::parse_endpoint(value);
            else if (option == "--ca") config.tls.ca_file = value;
            else if (option == "--cert") config.tls.cert_file = value;
            else if (option == "--key") config.tls.key_file = value;
            else if (option == "--harvester")
            {
                if (std::string_view(value) != "local") throw std::runtime_error("worker --harvester requires local");
                observe_farming = true;
            }
            else if (option == "--chia-root") chia_root = std::filesystem::path(reinterpret_cast<char8_t const*>(value));
            else if (option == "--farming-deadline-ms")
            {
                config.farming_deadline_ms = small(value, option);
                if (!config.farming_deadline_ms) throw std::runtime_error("--farming-deadline-ms must be positive");
            }
            else if (option == "--cpus") config.capacity.cpu_threads = small(value, option);
            else if (option == "--slots") config.capacity.slots = small(value, option);
            else if (option == "--host-bytes") config.capacity.host_bytes = number(value, option);
            else if (option == "--gpu-bytes") config.capacity.gpu_bytes = number(value, option);
            else if (option == "--generation-threads") config.generation_threads = small(value, option);
            else if (option == "--backend") { config.backend = value; backend_set = true; }
            else if (option == "--tier") config.tier = value;
            else if (option == "--device")
            { auto const device = small(value, option); if (device > INT32_MAX) throw std::runtime_error("--device must be between 0 and 2147483647"); config.device = int(device); }
            else if (option == "--max-frame-bytes") config.tls.max_frame_bytes = small(value, option);
            else if (option == "--max-task-bytes") config.max_task_bytes = number(value, option);
            else if (option == "--max-window-chunks") config.max_window_chunks = small(value, option);
            else if (option == "--max-batch-members") config.max_batch_members = small(value, option);
            else if (option == "--task-timeout-ms") config.task_timeout_ms = small(value, option);
            else if (option == "--tls-timeout-ms") config.tls.timeout_ms = small(value, option);
            else if (option == "--max-brokers") { config.max_brokers = small(value, option); brokers_set = true; }
            else throw std::runtime_error("unknown worker option: " + option);
        }
        output.validate();
        if (observe_farming) config.local_harvester = std::move(chia_root);
        else if (!chia_root.empty() || config.farming_deadline_ms)
            throw std::runtime_error("--chia-root and --farming-deadline-ms require worker --harvester local");
        std::optional<group_grow::pool::PoolIdentity> identity;
#ifdef XCHPLOT2_GROUP_COMMANDS
        if (!invitation.empty())
        {
            if (!config.tls.ca_file.empty() || !config.tls.cert_file.empty() || !config.tls.key_file.empty())
                throw std::runtime_error("--join cannot be combined with explicit TLS credentials");
            identity = group_grow::pool::join_pool(invitation, directory.empty() ? group_grow::pool::pool_directory() : directory, true);
            if (!output.quiet) std::cerr << "Using saved enrollment for " << identity->name
                << ". Delete the transferred invitation copies after joining.\n";
        }
        else if (config.tls.ca_file.empty() && config.tls.cert_file.empty() && config.tls.key_file.empty())
            identity = group_grow::pool::load_pool(directory.empty() ? group_grow::pool::pool_directory() : directory);
#endif
        std::shared_ptr<group_grow::pool::ComputeDiscovery> discovery;
        std::unique_ptr<group_grow::pool::ComputeDiscovery> advertisement;
        group_grow::pool::DiscoveryOptions announcement;
        if (identity)
        {
            config.tls.ca_file = identity->tls.ca_file; config.tls.cert_file = identity->tls.cert_file; config.tls.key_file = identity->tls.key_file;
            config.tls.membership_directory = identity->tls.membership_directory;
            config.server_name = identity->server_name;
            if (!backend_set) config.backend = identity->defaults.backend;
            if (!brokers_set) config.max_brokers = identity->defaults.max_brokers;
            if (!config.capacity.cpu_threads) config.capacity.cpu_threads = identity->defaults.cpus;
            if (!config.capacity.host_bytes) config.capacity.host_bytes = identity->defaults.host_bytes;
            if (!config.capacity.gpu_bytes) config.capacity.gpu_bytes = identity->defaults.gpu_bytes;
            if (config.brokers.empty() && !identity->defaults.broker.empty())
                config.brokers.push_back(group_grow::pool::parse_endpoint(identity->defaults.broker));
            for (auto& broker : config.brokers) broker.server_name = identity->server_name;
            if (config.listen.host.empty()) config.listen = {"::", identity->defaults.worker_port};
            if (interfaces.empty()) interfaces = identity->defaults.interfaces;
            config.tls.listen_lan = interfaces.empty();
            config.tls.listen_interfaces = interfaces;
            config.tls.listen_interfaces.push_back("lo");
            if (!discovery_set) discover = identity->defaults.discovery;
            announcement.pool_id = identity->pool_id; announcement.node_id = identity->node_id;
            announcement.name = identity->name; announcement.server_name = identity->server_name; announcement.interfaces = interfaces;
            if (discover)
            {
                discovery = std::make_shared<group_grow::pool::ComputeDiscovery>(announcement);
                config.discover_brokers = [discovery] { return discovery->brokers(); };
                if (discovery->status() != "LAN discovery active") std::cerr << discovery->status() << '\n';
            }
            group_grow::default_worker_resources(config);
        }
        else if (discovery_set && discover)
            throw std::runtime_error("LAN discovery needs saved pool membership; use compute setup on a plotting host or compute setup INVITATION on a helper");
        if ((!identity && (config.brokers.empty() || !config.listen.port)) || config.tls.ca_file.empty() ||
            config.tls.cert_file.empty() || config.tls.key_file.empty())
            throw std::runtime_error("worker needs saved pool membership, or explicit broker/listen endpoints and CA/cert/key");
        if (identity && !discover && config.brokers.empty())
            throw std::runtime_error("discovery is off; configure or supply a direct broker address");
        stopped = 0;
        std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
        config.identity_reservation = group_grow::WorkerService::reserve_identity(config.tls.cert_file);
        auto capacity = config.capacity;
        if (output.details())
            std::cerr << "Worker limits: " << capacity.slots << " slots, " << capacity.gpu_bytes << " GPU bytes; backend "
                << config.backend << ", generation threads " << config.generation_threads << ", broker limit " << config.max_brokers << '\n';
        group_grow::WorkerService worker(std::move(config)); worker.start();
        capacity = worker.status().capacity;
        if (identity && discover)
        {
            announcement.role = "worker"; announcement.endpoint = worker.endpoint();
            advertisement = std::make_unique<group_grow::pool::ComputeDiscovery>(std::move(announcement));
        }
        if (!output.quiet) std::cout << "Worker listening on port " << worker.endpoint().port << ": "
            << capacity.cpu_threads << " CPU threads, " << capacity.host_bytes / (uint64_t(1) << 20)
            << " MiB host RAM, " << capacity.gpu_bytes / (uint64_t(1) << 20) << " MiB GPU RAM.\n" << std::flush;
        uint64_t healthy = UINT64_MAX;
        std::string gpu_recovery;
        std::optional<bool> farming_pressure;
        auto admission = group_grow::pool::WorkerAdmission::Accepting;
        while (!stopped)
        {
            auto const status = worker.status();
            if (!output.quiet && observe_farming && (healthy == UINT64_MAX || status.farming_pressure != farming_pressure))
            {
                farming_pressure = status.farming_pressure;
                std::cout << (!farming_pressure ? "Local lookup pressure is unknown without a recent eligible or slow completion.\n" :
                    *farming_pressure ? "Slow local farming lookups are holding new worker tasks.\n" :
                    "Local farming lookups are within the target.\n") << std::flush;
            }
            if (!output.quiet && gpu_recovery != status.gpu_recovery)
            {
                gpu_recovery = status.gpu_recovery;
                if (gpu_recovery.empty()) std::cout << "GPU generation is available again.\n";
                else std::cout << "GPU recovering: " << gpu_recovery << '\n';
                std::cout << std::flush;
            }
            if (!output.quiet && (healthy != status.healthy_brokers || admission != status.admission))
            {
                healthy = status.healthy_brokers; admission = status.admission;
                if (!healthy) std::cout << "Waiting for an authenticated broker connection. Check broker availability, network and firewall settings.\n";
                else
                {
                    std::cout << "Registered with " << healthy << (healthy == 1 ? " broker" : " brokers");
                    if (admission == group_grow::pool::WorkerAdmission::Accepting)
                        std::cout << "; accepting plotting work. Plotting hosts must also be able to reach this worker's port.\n";
                    else std::cout << "; worker admission is " << group_grow::pool::admission_name(admission) << ".\n";
                }
                std::cout << std::flush;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        worker.stop(); return 0;
    }
    catch (std::exception const& error)
    { std::cerr << "compute worker: " << error.what() << '\n'; return 1; }
}
