#include "group_commands.hpp"
#include "host/ComputeIdentity.hpp"
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
void require(bool value, char const* message)
{ if (!value) throw std::invalid_argument(message); }
uint64_t number(std::string const& text)
{
    uint64_t value = 0; auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(), "compute resource limit requires a nonnegative integer");
    return value;
}
bool enabled(std::string const& value)
{
    require(value == "on" || value == "off" || value == "true" || value == "false", "discovery requires on or off");
    return value == "on" || value == "true";
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
    std::cout << " [OPTIONS]\n";
    if (command == "init")
        std::cout << "Create a pool and enroll this machine. Run once on the first machine.\n  --name NAME        this machine's display name\n";
    else if (command == "invite")
        std::cout << "Create a private invitation for one additional machine. Transfer it privately.\n  --name NAME        invited machine's display name (required)\n";
    else if (command == "join")
        std::cout << "Save this machine's invitation. Pool membership is shared with all enrolled machines.\n"
                     "Delete transferred invitation copies after joining; they contain this machine's private key.\n";
    else if (command == "configure")
        std::cout << "Save defaults used by subsequent plotting and compute commands.\n"
                     "  --broker auto|HOST:PORT  automatic broker per job, or a dedicated broker\n"
                     "  --interface auto|NAME   LAN by default; repeat to select interfaces\n"
                     "  --discovery on|off      automatic LAN discovery, default on\n"
                     "  --cpus N                offered CPU cap; 0 selects half the logical CPUs\n"
                     "  --host-bytes SIZE       offered RAM cap; 0 selects a bounded automatic budget\n"
                     "  --gpu-bytes SIZE        offered GPU RAM cap; 0 selects 80% of available VRAM\n"
                     "  --backend auto|cpu|gpu  offered generation backend\n";
    if (command == "configure") std::cout << "  --max-brokers N         bounded broker registrations, default 128\n";
    std::cout << "  --directory DIR    saved compute setup directory\n";
#ifdef _WIN32
    std::cout << "Default: %LOCALAPPDATA%\\xchplot2\\compute, or $XDG_CONFIG_HOME/xchplot2/compute when set.\n";
#else
    std::cout << "Default: $XDG_CONFIG_HOME/xchplot2/compute, or $HOME/.config/xchplot2/compute.\n";
#endif
}
} // namespace

int pos2gpu::run_compute_setup(int argc, char** argv)
{
    try
    {
        std::string const command = argv[0];
        require(command == "init" || command == "invite" || command == "join" || command == "configure",
            "unknown compute command; use compute --help");
        for (int i = 1; i < argc; ++i)
            if (std::string_view(argv[i]) == "--help" || std::string_view(argv[i]) == "-h") { help(command); return 0; }
        auto directory = pool::pool_directory();
        std::string name = command == "init" ? hostname() : "";
        std::filesystem::path file;
        std::vector<std::pair<std::string, std::string>> settings;
        for (int i = 1; i < argc; ++i)
        {
            std::string key = argv[i];
            if (!key.starts_with('-'))
            { require((command == "invite" || command == "join") && file.empty(), "unexpected compute argument"); file = std::filesystem::path(reinterpret_cast<char8_t const*>(key.c_str())); continue; }
            require(i + 1 < argc, "missing compute option value"); std::string value = argv[++i];
            if (key == "--directory") directory = std::filesystem::path(reinterpret_cast<char8_t const*>(value.c_str()));
            else if (key == "--name")
            { require(command == "init" || command == "invite", "--name belongs to compute init or invite"); name = value; }
            else
            { require(command == "configure", "unknown compute setup option"); settings.emplace_back(std::move(key), std::move(value)); }
        }
        if (command == "init")
        {
            auto identity = pool::create_pool(directory, name);
            std::cout << "Created pool " << identity.pool_id.substr(0, 12) << " for " << identity.name << ".\n"
                      << "Plot, fill and grow will use shared compute automatically.\n"
                      << "Add another machine with: xchplot2 compute invite FILE --name NAME\n";
            return 0;
        }
        if (command == "join")
        {
            require(!file.empty(), "compute join requires an invitation file");
            auto identity = pool::join_pool(file, directory);
            std::cout << "Joined pool " << identity.pool_id.substr(0, 12) << " as " << identity.name << ".\n"
                      << "Plot normally, or offer this machine with: xchplot2 compute worker\n"
                      << "Delete the transferred invitation copies after joining.\n";
            return 0;
        }
        auto identity = pool::load_pool(directory);
        require(bool(identity), "shared compute is not configured; use compute init or compute join");
        if (command == "invite")
        {
            require(!file.empty() && !name.empty(), "compute invite requires FILE and --name NAME");
            pool::invite_pool(*identity, file, name);
            std::cout << "Created a private invitation for " << name << ". Transfer it to that machine, then run:\n"
                      << "  xchplot2 compute join " << file.filename().string() << '\n';
        }
        else if (command == "configure")
        {
            require(!settings.empty(), "compute configure needs at least one setting");
            auto defaults = identity->defaults; bool interfaces = false;
            for (auto const& [key, value] : settings)
            {
                if (key == "--broker") defaults.broker = value == "auto" ? "" : value;
                else if (key == "--interface")
                {
                    if (!interfaces) { defaults.interfaces.clear(); interfaces = true; }
                    if (value == "auto") defaults.interfaces.clear(); else defaults.interfaces.push_back(value);
                }
                else if (key == "--discovery") defaults.discovery = enabled(value);
                else if (key == "--cpus")
                { auto cpus = number(value); require(cpus <= 1024, "compute CPU cap exceeds 1024"); defaults.cpus = uint32_t(cpus); }
                else if (key == "--host-bytes") defaults.host_bytes = number(value);
                else if (key == "--gpu-bytes") defaults.gpu_bytes = number(value);
                else if (key == "--backend") defaults.backend = value;
                else if (key == "--max-brokers")
                { auto count = number(value); require(count && count <= 4096, "broker limit must be in 1..4096"); defaults.max_brokers = uint32_t(count); }
                else throw std::invalid_argument("unknown compute setting: " + key);
            }
            pool::configure_pool(*identity, defaults); std::cout << "Saved compute defaults for " << identity->name << ".\n";
        }
        return 0;
    }
    catch (std::exception const& error) { std::cerr << "compute: " << error.what() << '\n'; return 1; }
}
