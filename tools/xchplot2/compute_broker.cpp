// Control-only broker for shared compute.
#include "host/ComputePool.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/ComputeDiscovery.hpp"
#include <charconv>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string_view>

#ifdef XCHPLOT2_GROUP_COMMANDS
#include "group_commands.hpp"
#endif

namespace
{
std::atomic<bool> stopping{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void stop(int) { stopping.store(true, std::memory_order_relaxed); }
unsigned number(std::string_view value)
{
    unsigned result = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result)
        throw std::runtime_error("option requires a positive integer");
    return result;
}
void help()
{
    std::cout <<
        "Network-compute broker for ordinary PoS2 group growth\n"
#ifdef XCHPLOT2_GROUP_COMMANDS
        "Usage: xchplot2 compute broker [OPTIONS]\n"
        "  --directory DIR      saved compute setup directory\n"
        "  --listen HOST:PORT   default: all selected interfaces, port 18444\n"
        "  --interface NAME     sharing/discovery interface; repeat for more\n"
        "  --discovery on|off   LAN advertising, default from saved setup\n"
        "  --tls-ca FILE --tls-cert FILE --tls-key FILE  explicit credentials\n"
#else
        "Usage: pos2_pr118_pool --listen HOST:PORT --tls-ca FILE --tls-cert FILE --tls-key FILE [OPTIONS]\n"
#endif
        "  --timeout-ms N       socket/TLS/frame deadline, default 5000\n"
        "  --lease-ms N         renewable compute lease duration, default 120000\n"
        "  --worker-ms N        worker registration expiry, default 30000\n"
        "  --control-threads N  control request handlers, default 4\n"
        "  --max-connections N  active and idle control channels, default 128\n"
        "  --idle-ms N          idle control channel timeout, default 30000\n"
        "  --owner-share ID:N   relative service share for a stable owner ID\n"
        "  --max-workers N      registered compute nodes, default 128\n"
        "  --max-owners N       coordinator namespaces, default 128\n"
        "  --max-tasks N        queued, active, and recent task records, default 4096\n"
        "Every connection requires a trusted client certificate. Restarting fences old leases.\n";
}
}
#ifdef XCHPLOT2_GROUP_COMMANDS
int pos2gpu::run_compute_broker(int argc, char** argv) try
#else
int main(int argc, char** argv) try
#endif
{
    group_grow::pool::BrokerOptions options; options.tls.timeout_ms = 5000; options.tls.max_frame_bytes = 65536;
    std::filesystem::path directory; std::vector<std::string> interfaces;
    bool discover = true, discovery_set = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string const key = argv[i];
        if (key == "--help" || key == "-h") { help(); return 0; }
        if (key == "--no-discovery") { discover = false; discovery_set = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        std::string const value = argv[++i];
        if (key == "--listen") options.listen = group_grow::pool::parse_endpoint(value);
        else if (key == "--directory") directory = std::filesystem::path(reinterpret_cast<char8_t const*>(value.c_str()));
        else if (key == "--interface") interfaces.push_back(value);
        else if (key == "--discovery")
        {
            if (value != "on" && value != "off" && value != "true" && value != "false") throw std::runtime_error("--discovery requires on or off");
            discover = value == "on" || value == "true"; discovery_set = true;
        }
        else if (key == "--tls-ca") options.tls.ca_file = value;
        else if (key == "--tls-cert") options.tls.cert_file = value;
        else if (key == "--tls-key") options.tls.key_file = value;
        else if (key == "--timeout-ms") options.tls.timeout_ms = number(value);
        else if (key == "--lease-ms") options.lease_ms = number(value);
        else if (key == "--worker-ms") options.worker_ms = number(value);
        else if (key == "--control-threads") options.control_threads = number(value);
        else if (key == "--max-connections") options.max_connections = number(value);
        else if (key == "--idle-ms") options.idle_ms = number(value);
        else if (key == "--owner-share")
        {
            auto const separator = value.rfind(':');
            if (separator == std::string::npos || !separator) throw std::runtime_error("--owner-share requires ID:N");
            auto const id = value.substr(0, separator), share = value.substr(separator + 1);
            if (!options.owner_shares.emplace(id, number(share)).second) throw std::runtime_error("duplicate owner share: " + id);
        }
        else if (key == "--max-workers") options.max_workers = number(value);
        else if (key == "--max-owners") options.max_owners = number(value);
        else if (key == "--max-tasks") options.max_tasks = number(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    std::optional<group_grow::pool::PoolIdentity> identity;
#ifdef XCHPLOT2_GROUP_COMMANDS
    if (options.tls.ca_file.empty() && options.tls.cert_file.empty() && options.tls.key_file.empty())
        identity = group_grow::pool::load_pool(directory.empty() ? group_grow::pool::pool_directory() : directory);
#endif
    if (identity)
    {
        options.tls.ca_file = identity->tls.ca_file; options.tls.cert_file = identity->tls.cert_file; options.tls.key_file = identity->tls.key_file;
        options.tls.membership_directory = identity->tls.membership_directory;
        if (options.listen.host.empty()) options.listen = {"::", 18444};
        if (interfaces.empty()) interfaces = identity->defaults.interfaces;
        options.tls.listen_lan = interfaces.empty();
        options.tls.listen_interfaces = interfaces;
        options.tls.listen_interfaces.push_back("lo");
        if (!discovery_set) discover = identity->defaults.discovery;
    }
    else if (discovery_set && discover)
        throw std::runtime_error("LAN discovery needs saved pool membership; use compute init or compute join");
    if (options.listen.host.empty() || !options.listen.port)
        throw std::runtime_error("broker needs saved pool membership, or --listen and explicit TLS credentials");
    stopping.store(false);
#ifdef _WIN32
    if (std::signal(SIGINT, stop) == SIG_ERR || std::signal(SIGTERM, stop) == SIG_ERR)
#else
    struct sigaction action{}; action.sa_handler = stop; sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, nullptr) || sigaction(SIGTERM, &action, nullptr))
#endif
        throw std::runtime_error("cannot install broker shutdown handlers");
    group_grow::pool::BrokerServer server(std::move(options)); auto const endpoint = server.endpoint();
    std::unique_ptr<group_grow::pool::ComputeDiscovery> discovery;
    if (identity && discover)
    {
        group_grow::pool::DiscoveryOptions announcement; announcement.pool_id = identity->pool_id; announcement.node_id = identity->node_id;
        announcement.name = identity->name; announcement.server_name = identity->server_name; announcement.interfaces = interfaces;
        announcement.role = "broker"; announcement.endpoint = endpoint;
        discovery = std::make_unique<group_grow::pool::ComputeDiscovery>(std::move(announcement));
        if (discovery->status() != "LAN discovery active") std::cerr << discovery->status() << '\n';
    }
    std::cout << "Network-compute broker listening on " << endpoint.host << ':' << endpoint.port << '\n' << std::flush;
    server.serve(stopping); return 0;
}
catch (std::exception const& error) { std::cerr << "compute broker: " << error.what() << '\n'; return 1; }
