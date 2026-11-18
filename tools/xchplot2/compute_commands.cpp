#include "group_commands.hpp"
#include "compute_output.hpp"
#include "host/ComputeIdentity.hpp"
#include <algorithm>
#include <charconv>
#include <iostream>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#else
#include <unistd.h>
#endif

namespace
{
namespace pool = group_grow::pool;
void require(bool value, std::string const& message)
{ if (!value) throw std::invalid_argument(message); }
uint64_t number(std::string const& text, std::string const& option)
{
    uint64_t value = 0; auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
        option + " requires a nonnegative integer no greater than 18446744073709551615");
    return value;
}
bool enabled(std::string const& value)
{
    require(value == "on" || value == "off" || value == "true" || value == "false", "--discovery requires on or off");
    return value == "on" || value == "true";
}
std::string shell_path(std::filesystem::path const& path)
{
    auto const text = path.u8string(); std::string result = "'";
    for (char8_t c : text)
    {
#ifdef _WIN32
        if (c == '\'') result += '\'';
#else
        if (c == '\'') result += "'\\'";
#endif
        result += char(c);
    }
    return result + '\'';
}
pool::ComputeDefaults configured(pool::ComputeDefaults defaults,
    std::vector<std::pair<std::string, std::string>> const& settings)
{
    bool interfaces = false;
    bool const local_broker = defaults.broker == "127.0.0.1:" + std::to_string(defaults.broker_port);
    for (auto const& [key, value] : settings)
    {
        if (key == "--broker") defaults.broker = value == "auto" ? "" : value;
        else if (key == "--interface")
        {
            if (!interfaces) { defaults.interfaces.clear(); interfaces = true; }
            if (value == "auto") defaults.interfaces.clear(); else defaults.interfaces.push_back(value);
        }
        else if (key == "--discovery") defaults.discovery = enabled(value);
        else if (key == "--broker-port" || key == "--worker-port")
        {
            auto const port = number(value, key);
            require(port <= UINT16_MAX && (port || key == "--worker-port"),
                key + " requires a port in 1..65535; only --worker-port accepts 0 for an available port");
            if (key == "--broker-port") defaults.broker_port = uint16_t(port);
            else defaults.worker_port = uint16_t(port);
        }
        else if (key == "--cpus")
        { auto cpus = number(value, key); require(cpus <= 1024, "--cpus must be between 0 and 1024; 0 selects an automatic limit"); defaults.cpus = uint32_t(cpus); }
        else if (key == "--host-bytes") defaults.host_bytes = number(value, key);
        else if (key == "--gpu-bytes") defaults.gpu_bytes = number(value, key);
        else if (key == "--backend")
        { require(value == "cpu" || value == "gpu" || value == "auto", "--backend requires cpu, gpu or auto"); defaults.backend = value; }
        else if (key == "--max-brokers")
        { auto count = number(value, key); require(count && count <= 4096, "--max-brokers must be between 1 and 4096"); defaults.max_brokers = uint32_t(count); }
        else throw std::invalid_argument("unknown compute setting: " + key);
    }
    if (local_broker && std::none_of(settings.begin(), settings.end(), [](auto const& setting) { return setting.first == "--broker"; }))
        defaults.broker = "127.0.0.1:" + std::to_string(defaults.broker_port);
    pool::validate_compute_defaults(defaults); return defaults;
}
std::string hostname()
{
#ifdef _WIN32
    wchar_t name[256]{}; DWORD size = 256;
    if (!::GetComputerNameW(name, &size)) return "this machine";
    auto text = std::filesystem::path(std::wstring(name, size)).u8string();
    return std::string(reinterpret_cast<char const*>(text.data()), text.size()).substr(0, 64);
#else
    char name[256]{}; if (::gethostname(name, sizeof(name) - 1)) return "this machine";
    return std::string(name).substr(0, 64);
#endif
}
void help(std::string_view command)
{
    std::cout << "Usage: xchplot2 compute " << command;
    if (command == "invite" || command == "join") std::cout << " FILE";
    else if (command == "setup") std::cout << " [INVITATION]";
    std::cout << " [OPTIONS]\n";
    if (command == "setup")
        std::cout << "Set up shared compute, reusing saved enrollment when present.\n"
                     "Without an invitation, create a pool and start a persistent broker.\n"
                     "With an invitation, join and start a persistent worker on this machine.\n"
                     "  --role plotter|worker  start a broker or worker service\n"
                     "  --name NAME            first machine's display name\n";
    else if (command == "init")
        std::cout << "Create a pool and enroll this machine. Run once on the first machine.\n  --name NAME        this machine's display name\n";
    else if (command == "invite")
        std::cout << "Create a private invitation for one additional machine. Transfer it privately.\n  --name NAME        invited machine's display name (required)\n";
    else if (command == "join")
        std::cout << "Enroll this machine using the invitation to join the shared compute pool.\n"
                     "Delete transferred invitation copies after joining; they contain this machine's private key.\n";
    if (command == "configure" || command == "setup")
        std::cout << "Save defaults used by subsequent plotting and compute commands.\n"
                     "  --broker auto|HOST:PORT  automatic broker per job, or a dedicated broker\n"
                     "  --broker-port N         dedicated broker listener, default 18444\n"
                     "  --worker-port N         worker listener; setup defaults to 18445 for new enrollments\n"
                     "  --interface auto|NAME   LAN by default; repeat to select interfaces\n"
                     "  --discovery on|off      automatic LAN discovery, default on\n"
                     "  --cpus N                CPU cap; 0 adapts within the available CPU affinity\n"
                     "  --host-bytes SIZE       offered RAM limit, e.g. 8G; 0 selects an automatic limit\n"
                     "  --gpu-bytes SIZE        GPU RAM cap, e.g. 8G; 0 uses available VRAM at startup\n"
                     "  --backend auto|cpu|gpu  offered generation backend\n";
    if (command == "configure" || command == "setup") std::cout << "  --max-brokers N         maximum broker registrations, default 128\n";
    std::cout << "  --directory DIR    saved compute setup directory\n";
    std::cout << pos2gpu::compute_output_help;
#ifdef _WIN32
    std::cout << "Default: %LOCALAPPDATA%\\xchplot2\\compute, or $XDG_CONFIG_HOME/xchplot2/compute when set.\n";
#else
    std::cout << "Default: $XDG_CONFIG_HOME/xchplot2/compute, or $HOME/.config/xchplot2/compute.\n";
#endif
}
} // namespace

int pos2gpu::run_compute_firewall(int argc, char** argv)
{
    try
    {
        auto directory = pool::pool_directory();
        std::string firewall, zone, role = "both", interface;
        std::vector<std::string> sources;
        std::optional<bool> discovery;
        ComputeOutputOptions output;
        for (int i = 1; i < argc; ++i)
        {
            std::string const key = argv[i];
            if (key == "--help" || key == "-h" || key == "--help-all")
            {
                std::cout << "Usage: xchplot2 compute firewall --firewall firewalld|ufw --from IP[/PREFIX] [OPTIONS]\n"
                    "Print source-scoped Linux firewall commands using saved listener ports. No rules are applied.\n"
                    "  --from IP[/PREFIX]       permitted peer or subnet; repeat for more\n"
                    "  --role broker|worker|both  inbound listeners to permit, default both\n"
                    "  --zone NAME              required firewalld zone for the LAN interface\n"
                    "  --interface NAME         UFW interface; otherwise use saved interfaces\n"
                    "  --discovery on|off       include UDP 5353; default from saved setup\n"
                    "  --directory DIR          saved compute setup directory\n"
                    << compute_output_help;
                return 0;
            }
            if (output.parse(key)) continue;
            require(i + 1 < argc, "missing value for " + key); std::string const value = argv[++i];
            if (key == "--directory") directory = std::filesystem::path(reinterpret_cast<char8_t const*>(value.c_str()));
            else if (key == "--firewall") firewall = value;
            else if (key == "--from") sources.push_back(value);
            else if (key == "--role") role = value;
            else if (key == "--zone") zone = value;
            else if (key == "--interface") interface = value;
            else if (key == "--discovery") discovery = enabled(value);
            else throw std::invalid_argument("unknown firewall option: " + key);
        }
        output.validate();
        require(firewall == "firewalld" || firewall == "ufw", "--firewall requires firewalld or ufw");
        require(role == "broker" || role == "worker" || role == "both", "--role requires broker, worker or both");
        require(!sources.empty(), "--from requires a permitted peer address or subnet");
        auto name = [](std::string const& value)
        { return !value.empty() && value.size() <= 64 && value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-") == value.npos; };
        require(firewall != "firewalld" || (name(zone) && interface.empty()),
            "firewalld requires --zone NAME; select the zone assigned to the LAN interface");
        require(firewall != "ufw" || (zone.empty() && (interface.empty() || name(interface))),
            "ufw accepts --interface NAME, not --zone");
        for (auto const& source : sources)
        {
            auto const slash = source.find('/'); auto const host = source.substr(0, slash);
            require(host.find('%') == host.npos && pool::numeric_host(host), "--from requires a numeric IPv4 or IPv6 address");
            if (slash != source.npos)
                require(number(source.substr(slash + 1), "--from prefix") <= (host.find(':') == host.npos ? 32 : 128),
                    "--from prefix exceeds its address family");
        }
        auto identity = pool::load_pool(directory);
        require(bool(identity), "shared compute is not configured; use compute setup first");
        auto const& defaults = identity->defaults;
        require(role == "broker" || defaults.worker_port,
            "worker port is automatic; save a fixed port with compute configure --worker-port 18445 and restart the worker");
        std::vector<std::pair<uint16_t, std::string>> ports;
        if (role != "worker") ports.emplace_back(defaults.broker_port, "tcp");
        if (role != "broker") ports.emplace_back(defaults.worker_port, "tcp");
        if (discovery.value_or(defaults.discovery)) ports.emplace_back(5353, "udp");
        auto interfaces = interface.empty() ? defaults.interfaces : std::vector<std::string>{interface};
        if (interfaces.empty()) interfaces.emplace_back();
        if (firewall == "ufw") for (auto const& selected : interfaces)
            require(selected.empty() || name(selected), "saved interface cannot be rendered as a Linux firewall command");
        if (!output.quiet) std::cout << "# Run these commands on this machine to allow the specified peers.\n";
        for (auto const& source : sources) for (auto const& [port, protocol] : ports)
        {
            if (firewall == "firewalld")
            {
                std::string const rule = "rule family=\"" + std::string(source.find(':') == source.npos ? "ipv4" : "ipv6") +
                    "\" source address=\"" + source + "\" port port=\"" + std::to_string(port) +
                    "\" protocol=\"" + protocol + "\" accept";
                for (bool permanent : {false, true}) std::cout << "sudo firewall-cmd --zone=" << zone
                    << (permanent ? " --permanent" : "") << " --add-rich-rule='" << rule << "'\n";
            }
            else for (auto const& selected : interfaces)
                std::cout << "sudo ufw allow in" << (selected.empty() ? "" : " on " + selected)
                    << " proto " << protocol << " from " << source << " to any port " << port << '\n';
        }
        return 0;
    }
    catch (std::exception const& error) { std::cerr << "compute firewall: " << error.what() << '\n'; return 1; }
}

int pos2gpu::run_compute_setup(int argc, char** argv)
{
    try
    {
        std::string const command = argv[0];
        require(command == "setup" || command == "init" || command == "invite" || command == "join" || command == "configure",
            "unknown compute command; use compute --help");
        auto directory = pool::pool_directory();
        std::string name = command == "init" || command == "setup" ? hostname() : "", role;
        std::filesystem::path file;
        std::vector<std::pair<std::string, std::string>> settings;
        ComputeOutputOptions output;
        for (int i = 1; i < argc; ++i)
        {
            std::string key = argv[i];
            if (key == "--help" || key == "-h" || key == "--help-all") { help(command); return 0; }
            if (output.parse(key)) continue;
            if (!key.starts_with('-'))
            { require((command == "setup" || command == "invite" || command == "join") && file.empty(), "unexpected argument: " + key + "; use compute " + command + " --help"); file = std::filesystem::path(reinterpret_cast<char8_t const*>(key.c_str())); continue; }
            require(i + 1 < argc, "missing value for " + key); std::string value = argv[++i];
            if (key == "--directory") directory = std::filesystem::path(reinterpret_cast<char8_t const*>(value.c_str()));
            else if (key == "--name")
            { require(command == "setup" || command == "init" || command == "invite", "--name belongs to compute setup, init or invite"); name = value; }
            else if (key == "--role")
            { require(command == "setup" && (value == "plotter" || value == "worker"), "--role requires plotter or worker on compute setup"); role = value; }
            else
            { require(command == "configure" || command == "setup", "unknown option: " + key + "; use compute " + command + " --help"); settings.emplace_back(std::move(key), std::move(value)); }
        }
        output.validate();
        if (command == "setup")
        {
            auto identity = pool::load_pool(directory); bool const existed = identity.has_value();
            require(file.empty() || !identity, "this machine is already enrolled; omit the invitation to reuse its setup");
            auto initial = identity ? identity->defaults : pool::ComputeDefaults{};
            if (!identity) initial.worker_port = 18445;
            auto defaults = configured(initial, settings);
            if (!identity) identity = file.empty() ? pool::create_pool(directory, name) : pool::join_pool(file, directory);
            if (role.empty()) role = identity->can_invite ? "plotter" : "worker";
            bool const broker_selected = std::any_of(settings.begin(), settings.end(),
                [](auto const& setting) { return setting.first == "--broker"; });
            auto const local_broker = "127.0.0.1:" + std::to_string(defaults.broker_port);
            auto const previous_broker = "127.0.0.1:" + std::to_string(identity->defaults.broker_port);
            bool const start_broker = role == "plotter" && !broker_selected &&
                (!existed || defaults.broker == previous_broker || defaults.broker == local_broker);
            if (start_broker) defaults.broker = local_broker;
            pool::configure_pool(*identity, defaults);
            std::cout << "Shared compute is configured for " << identity->name << ".\n";
            if (!file.empty()) std::cout << "Delete the transferred invitation copies after joining.\n";
            if (role == "worker" || start_broker)
            {
                auto const service_role = role == "worker" ? "worker" : "broker";
                auto service = [&](std::string action)
                {
                    auto const path = directory.u8string();
                    std::vector<std::string> values{"service", std::move(action), "--role", service_role, "--directory",
                        std::string(reinterpret_cast<char const*>(path.data()), path.size())};
                    if (output.quiet) values.push_back("--quiet"); else if (output.verbose) values.push_back("--verbose");
                    std::vector<char*> arguments; for (auto& value : values) arguments.push_back(value.data());
                    return run_compute_service(int(arguments.size()), arguments.data());
                };
                auto result = service("install");
                if (!result && existed && !settings.empty()) result = service("restart");
                if (result)
                {
                    std::cerr << "Enrollment and resource settings are saved. Complete startup with xchplot2 compute service install --role "
                        << service_role << " --directory " << shell_path(directory) << '\n';
                    return result;
                }
                if (!output.quiet)
                {
                    auto const port = role == "worker" ? defaults.worker_port : defaults.broker_port;
                    std::cout << "Compute " << service_role << " uses ";
                    if (port) std::cout << "TCP port " << port;
                    else std::cout << "an available TCP port";
                    std::cout << ". Use compute status to check authenticated connections.\n"
                        "Use compute firewall --help for source-scoped firewall commands.\n";
                }
                if (role == "worker") return 0;
            }
            if (!output.quiet) std::cout << "Plot, fill and grow will discover enrolled workers automatically.\n"
                << "Add a helper with xchplot2 compute invite FILE --name NAME, then run xchplot2 compute setup FILE on that machine.\n";
            return 0;
        }
        if (command == "init")
        {
            auto identity = pool::create_pool(directory, name);
            std::cout << "Created pool " << identity.pool_id.substr(0, 12) << " for " << identity.name << ".\n";
            if (output.details())
                std::cout << "Plot, fill and grow will use shared compute automatically.\n"
                          << "Add another machine with: xchplot2 compute invite FILE --name NAME\n";
            return 0;
        }
        if (command == "join")
        {
            require(!file.empty(), "compute join requires an invitation file");
            auto identity = pool::join_pool(file, directory);
            std::cout << "Joined pool " << identity.pool_id.substr(0, 12) << " as " << identity.name << ".\n";
            if (output.details()) std::cout << "Plot normally, or offer this machine with: xchplot2 compute worker\n";
            std::cout << "Delete the transferred invitation copies after joining.\n";
            return 0;
        }
        auto identity = pool::load_pool(directory);
        require(bool(identity), "shared compute is not configured; use compute setup, or compute setup INVITATION to join");
        if (command == "invite")
        {
            require(!file.empty() && !name.empty(), "compute invite requires FILE and --name NAME");
            pool::invite_pool(*identity, file, name);
            std::cout << "Created a private invitation for " << name << ". Transfer it to that machine, then run:\n"
                      << "  xchplot2 compute setup " << shell_path(file.filename()) << '\n';
        }
        else if (command == "configure")
        {
            require(!settings.empty(), "compute configure needs at least one setting");
            auto defaults = configured(identity->defaults, settings);
            pool::configure_pool(*identity, defaults); std::cout << "Saved compute defaults for " << identity->name << ".\n";
        }
        return 0;
    }
    catch (std::exception const& error) { std::cerr << "compute: " << error.what() << '\n'; return 1; }
}
