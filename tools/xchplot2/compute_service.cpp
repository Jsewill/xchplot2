#include "group_commands.hpp"
#include "compute_output.hpp"
#include "host/ComputeIdentity.hpp"
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace
{
namespace fs = std::filesystem;
namespace pool = group_grow::pool;
void require(bool value, char const* message)
{ if (!value) throw std::invalid_argument(message); }
std::string quoted(std::string const& value)
{
    std::string result = "\"";
    for (unsigned char c : value)
    {
        require(c >= 32 && c != 127, "service paths cannot contain control characters");
        if (c == '\\' || c == '"') result += '\\';
        if (c == '%' || c == '$') result += char(c);
        result += char(c);
    }
    return result + '"';
}
fs::path unit_directory()
{
    auto value = std::getenv("XDG_CONFIG_HOME");
    if (value && *value) return fs::absolute(value) / "systemd/user";
    value = std::getenv("HOME"); require(value && *value, "HOME or XDG_CONFIG_HOME is required for a user service");
    return fs::absolute(value) / ".config/systemd/user";
}
int systemctl(std::vector<std::string> arguments, bool quiet = false)
{
    if (quiet) arguments.insert(arguments.begin(), "--quiet");
    arguments.insert(arguments.begin(), {"systemctl", "--user", "--no-pager"});
    std::vector<char*> values; for (auto& argument : arguments) values.push_back(argument.data()); values.push_back(nullptr);
    pid_t child = 0;
    auto error = ::posix_spawnp(&child, values.front(), nullptr, nullptr, values.data(), environ);
    if (error) throw std::system_error(error, std::generic_category(), "cannot run systemctl --user");
    int status = 0;
    while (::waitpid(child, &status, 0) < 0)
        if (errno != EINTR) throw std::system_error(errno, std::generic_category(), "cannot wait for systemctl");
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
void check_unit(fs::path const& path, std::string const& marker, bool missing)
{
    struct stat info{};
    if (::lstat(path.c_str(), &info))
    { require(errno == ENOENT && missing, "compute service is not installed; run compute service install"); return; }
    require(S_ISREG(info.st_mode) && info.st_uid == ::geteuid(), "refusing to replace a service file not owned by this user");
    std::ifstream input(path); std::string line; std::getline(input, line);
    require(input && line == marker, "refusing to modify a service file not generated for this compute identity");
}
void publish_unit(fs::path const& path, std::string const& contents)
{
    fs::create_directories(path.parent_path());
    std::string temporary = path.string() + ".partial.XXXXXX";
    int fd = ::mkstemp(temporary.data()); require(fd >= 0, "cannot create compute service file");
    try
    {
        size_t position = 0;
        while (position < contents.size())
        {
            auto n = ::write(fd, contents.data() + position, contents.size() - position);
            if (n < 0 && errno == EINTR) continue;
            require(n > 0, "cannot write compute service file"); position += size_t(n);
        }
        require(!::fsync(fd), "cannot sync compute service file");
        int const closed = ::close(fd); fd = -1; require(!closed, "cannot close compute service file");
        require(!::rename(temporary.c_str(), path.c_str()), "cannot publish compute service file");
        int parent = ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        require(parent >= 0, "cannot open service directory");
        int const synced = ::fsync(parent); ::close(parent); require(!synced, "cannot sync service directory");
    }
    catch (...) { if (fd >= 0) ::close(fd); ::unlink(temporary.c_str()); throw; }
}
} // namespace

int pos2gpu::run_compute_service(int argc, char** argv) try
{
    auto help = []
    {
        std::cout << "Usage: xchplot2 compute service COMMAND [OPTIONS]\n"
            "  install  Install and enable a systemd user service, and start it\n"
            "  start    Start an installed service\n"
            "  stop     Stop it; owners retry unfinished tasks elsewhere\n"
            "  restart  Apply changed defaults or installed executable\n"
            "  status   Show whether the service is active; -v includes manager details and logs\n"
            "  remove   Stop, disable and remove this generated service\n"
            "  print    Print the unit without writing files or starting work\n"
            "  --role worker|broker  Service role, default worker\n"
            "  --directory DIR       Saved pool membership\n"
            << pos2gpu::compute_output_help <<
            "Resources and network selection come from compute configure.\n"
            "Use compute drain before stop to let accepted work finish.\n"
            "User services run while the user manager is active; logout behavior follows the system's login policy.\n";
    };
    if (argc < 2) { help(); return 1; }
    std::string command, role = "worker"; auto directory = pool::pool_directory();
    ComputeOutputOptions output;
    for (int i = 1; i < argc; ++i)
    {
        std::string const option = argv[i];
        if (option == "--help" || option == "-h" || option == "--help-all") { help(); return 0; }
        if (output.parse(option)) continue;
        if (!option.starts_with('-'))
        { require(command.empty(), "unexpected compute service argument"); command = option; continue; }
        require(i + 1 < argc, "missing compute service option value"); std::string value = argv[++i];
        if (option == "--role") role = value;
        else if (option == "--directory") directory = value;
        else throw std::invalid_argument("unknown compute service option: " + option);
    }
    output.validate();
    require(command == "install" || command == "start" || command == "stop" || command == "restart" ||
        command == "status" || command == "remove" || command == "print", "unknown compute service command");
    require(role == "worker" || role == "broker", "service role must be worker or broker");
    auto identity = pool::load_pool(directory); require(bool(identity), "shared compute is not configured; use compute init or compute join");
    auto const name = "xchplot2-compute-" + role + '-' + identity->node_id + ".service";
    auto const path = unit_directory() / name;
    auto const marker = "# xchplot2 compute service " + role + ' ' + identity->node_id;
    if (command == "install" || command == "print")
    {
        auto const executable = fs::canonical("/proc/self/exe");
        auto const contents = marker + "\n[Unit]\nDescription=xchplot2 shared compute " + role +
            "\n\n[Service]\nType=exec\nExecStart=" + quoted(executable.string()) + " compute " + role +
            " --directory " + quoted(fs::canonical(directory).string()) +
            "\nRestart=on-failure\nRestartSec=5\nTimeoutStopSec=45\nUMask=0077\n\n[Install]\nWantedBy=default.target\n";
        if (command == "print") { std::cout << contents; return 0; }
        check_unit(path, marker, true); publish_unit(path, contents);
        auto result = systemctl({"daemon-reload"}, output.quiet);
        if (!result) result = systemctl({"enable", "--now", name}, output.quiet);
        std::cout << "Installed " << path.string() << '\n';
        if (result) std::cerr << "The unit is saved, but systemd did not enable/start it. Check that a systemd user manager is available.\n";
        return result;
    }
    check_unit(path, marker, false);
    if (command == "remove")
    {
        auto result = systemctl({"disable", "--now", name}, output.quiet); if (result) return result;
        require(::unlink(path.c_str()) == 0, "cannot remove compute service file");
        result = systemctl({"daemon-reload"}, output.quiet); std::cout << "Removed " << path.string() << '\n'; return result;
    }
    if (command == "status" && !output.details())
    {
        std::cout << "Compute " << role << " service: " << std::flush;
        return systemctl({"is-active", name});
    }
    return systemctl({command, name}, output.quiet);
}
catch (std::exception const& error) { std::cerr << "compute service: " << error.what() << '\n'; return 1; }
