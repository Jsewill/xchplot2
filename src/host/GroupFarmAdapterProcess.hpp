// Fixed authenticated farm-adapter RPC operations. Native process output
// stays private to the parser, including unsuccessful RPC diagnostics.
#pragma once
#include "host/GroupCalibrationProcess.hpp"
#include "host/GroupFile.hpp"
#include <utility>
#ifdef __linux__
#include <sys/syscall.h>
#endif

namespace group_grow::farm
{
enum class AdapterOperation { reader_cli_version, harvester_version, harvester_plots, harvester_network,
    harvester_connections, full_node_version, full_node_network, full_node_constants, full_node_state, full_node_block_record, full_node_connections,
    farmer_version, farmer_network, farmer_connections, farmer_harvesters, farmer_harvesters_summary, farmer_pool_state,
    solver_version, solver_network, solver_connections, solver_state };
struct AdapterCommand
{
    AdapterOperation operation = AdapterOperation::harvester_version;
    std::filesystem::path executable, root;
    unsigned threads = 1;
    uint32_t height = 0;
};
struct AdapterResult
{
    int exit_code = -1;
    std::string output;
    uint64_t wall_ms = 0;
    // Linux bounds inherited per-process virtual address space. Windows bounds
    // aggregate Job and per-process committed memory. Neither bounds total RSS
    // or the separately running foreign harvester/solver service's memory.
    std::string cap_kind;
    uint64_t memory_cap_bytes = 0;
    unsigned cpu_lanes = 0;
};
namespace adapter_process_detail
{
using calibration_process::check;
using calibration_process::Clock;
inline std::vector<std::string> arguments(AdapterCommand const& command)
{
    check(command.executable.is_absolute(), "farm adapter executable must be an explicit absolute path");
    std::vector<std::string> result{file_io::path_text(command.executable)};
    check(command.root.is_absolute(), "farm adapter Chia root must be an explicit absolute path");
    if (command.operation == AdapterOperation::reader_cli_version)
    { result.insert(result.end(), {"--root-path", file_io::path_text(command.root), "version"}); return result; }
    char const* service = "harvester"; char const* endpoint = nullptr;
    switch (command.operation)
    {
    case AdapterOperation::harvester_version: case AdapterOperation::full_node_version: endpoint = "get_version"; break;
    case AdapterOperation::harvester_network: case AdapterOperation::full_node_network: endpoint = "get_network_info"; break;
    case AdapterOperation::harvester_plots: endpoint = "get_plots"; break;
    case AdapterOperation::harvester_connections: endpoint = "get_connections"; break;
    case AdapterOperation::full_node_constants: endpoint = "get_constants"; break;
    case AdapterOperation::full_node_state: endpoint = "get_blockchain_state"; break;
    case AdapterOperation::full_node_block_record: endpoint = "get_block_record_by_height"; break;
    case AdapterOperation::full_node_connections: endpoint = "get_connections"; break;
    case AdapterOperation::farmer_version: service = "farmer"; endpoint = "get_version"; break;
    case AdapterOperation::farmer_network: service = "farmer"; endpoint = "get_network_info"; break;
    case AdapterOperation::farmer_connections: service = "farmer"; endpoint = "get_connections"; break;
    case AdapterOperation::farmer_harvesters: service = "farmer"; endpoint = "get_harvesters"; break;
    case AdapterOperation::farmer_harvesters_summary: service = "farmer"; endpoint = "get_harvesters_summary"; break;
    case AdapterOperation::farmer_pool_state: service = "farmer"; endpoint = "get_pool_state"; break;
    case AdapterOperation::solver_version: service = "solver"; endpoint = "get_version"; break;
    case AdapterOperation::solver_network: service = "solver"; endpoint = "get_network_info"; break;
    case AdapterOperation::solver_connections: service = "solver"; endpoint = "get_connections"; break;
    case AdapterOperation::solver_state: service = "solver"; endpoint = "get_state"; break;
    default: throw std::runtime_error("unsupported farm adapter operation");
    }
    if (command.operation == AdapterOperation::full_node_version || command.operation == AdapterOperation::full_node_network ||
        command.operation == AdapterOperation::full_node_constants || command.operation == AdapterOperation::full_node_state ||
        command.operation == AdapterOperation::full_node_block_record || command.operation == AdapterOperation::full_node_connections) service = "full_node";
    result.insert(result.end(), {"--root-path", file_io::path_text(command.root), "rpc", service, endpoint});
    if (command.operation == AdapterOperation::full_node_block_record) result.push_back("{\"height\":" + std::to_string(command.height) + '}');
    return result;
}
} // namespace adapter_process_detail

inline AdapterResult run_farm_adapter(AdapterCommand const& command, calibration_process::Limits const& limits)
{
    using namespace adapter_process_detail;
    check(limits.memory_bytes >= 16 * 1024 * 1024 && limits.output_bytes && limits.output_bytes <= limits.memory_bytes / 8,
        "farm adapter requires bounded process and output memory");
    auto const started = Clock::now(), until = calibration_process::deadline(limits.timeout_ms);
    calibration_process::check_running(until, limits.cancelled);
    check(command.threads && command.threads <= 64, "farm adapter requires a bounded CPU allocation");
    auto args = arguments(command); AdapterResult result; result.memory_cap_bytes = limits.memory_bytes;
    auto append = [&](char const* buffer, size_t count)
    { check(count <= limits.output_bytes - result.output.size(), "farm adapter exceeded its output bound (output withheld)"); result.output.append(buffer, count); };
#ifdef _WIN32
    using calibration_process::NativeHandle;
    check(limits.memory_bytes <= SIZE_MAX, "farm adapter process-memory limit is too large");
    std::wstring command_line;
    for (auto const& arg : args)
    {
        if (!command_line.empty()) command_line += L' ';
        command_line += profile_detail::windows_argument(file_io::native_path(arg).wstring());
    }
    NativeHandle read, write; SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    check(::CreatePipe(&read.value, &write.value, &attributes, 0), "cannot create farm adapter output pipe");
    check(::SetHandleInformation(read.value, HANDLE_FLAG_INHERIT, 0), "cannot restrict farm adapter pipe inheritance");
    NativeHandle input(::CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes, OPEN_EXISTING, 0, nullptr));
    check(input.value != INVALID_HANDLE_VALUE, "cannot isolate farm adapter input");
    NativeHandle job(::CreateJobObjectW(nullptr, nullptr)); check(job.value, "cannot create farm adapter memory job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION bound{};
    DWORD_PTR offered = 0, system = 0, affinity = 0;
    check(::GetProcessAffinityMask(::GetCurrentProcess(), &offered, &system), "cannot inspect farm adapter CPU allocation");
    for (unsigned i = 0; i < sizeof(offered) * 8 && result.cpu_lanes < command.threads; ++i)
        if (offered & (DWORD_PTR(1) << i)) { affinity |= DWORD_PTR(1) << i; ++result.cpu_lanes; }
    check(affinity, "farm adapter has no admitted CPU lane");
    bound.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PROCESS_MEMORY |
        JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_AFFINITY;
    bound.BasicLimitInformation.Affinity = affinity;
    bound.ProcessMemoryLimit = SIZE_T(limits.memory_bytes); bound.JobMemoryLimit = SIZE_T(limits.memory_bytes);
    check(::SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &bound, sizeof(bound)), "cannot cap farm adapter memory");
    SIZE_T size = 0; ::InitializeProcThreadAttributeList(nullptr, 1, 0, &size); std::vector<std::byte> storage(size);
    auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    check(::InitializeProcThreadAttributeList(list, 1, 0, &size), "cannot isolate farm adapter handles");
    struct List { LPPROC_THREAD_ATTRIBUTE_LIST p; ~List() { ::DeleteProcThreadAttributeList(p); } } cleanup{list};
    std::array<HANDLE, 2> inherited{write.value, input.value};
    check(::UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(), sizeof(inherited), nullptr, nullptr), "cannot restrict farm adapter handles");
    struct Child
    {
        PROCESS_INFORMATION p{}; HANDLE job = nullptr; bool done = false;
        ~Child() { if (p.hProcess) { if (job) ::TerminateJobObject(job, 1); if (!done) { ::TerminateProcess(p.hProcess, 1); ::WaitForSingleObject(p.hProcess, INFINITE); } ::CloseHandle(p.hProcess); ::CloseHandle(p.hThread); } }
    } child; child.job = job.value;
    STARTUPINFOEXW startup{}; startup.StartupInfo.cb = sizeof(startup); startup.lpAttributeList = list;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES; startup.StartupInfo.hStdInput = input.value;
    startup.StartupInfo.hStdOutput = write.value; startup.StartupInfo.hStdError = write.value;
    check(::CreateProcessW(command.executable.c_str(), command_line.data(), nullptr, nullptr, TRUE,
        CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
        nullptr, nullptr, &startup.StartupInfo, &child.p), "cannot start installed farm adapter");
    check(::AssignProcessToJobObject(job.value, child.p.hProcess), "cannot contain installed farm adapter");
    calibration_process::check_running(until, limits.cancelled);
    check(::ResumeThread(child.p.hThread) != DWORD(-1), "cannot resume installed farm adapter");
    ::CloseHandle(write.value); write.value = nullptr;
    for (;;)
    {
        calibration_process::check_running(until, limits.cancelled); DWORD available = 0;
        bool const pipe = ::PeekNamedPipe(read.value, nullptr, 0, nullptr, &available, nullptr);
        check(pipe || ::GetLastError() == ERROR_BROKEN_PIPE, "cannot inspect farm adapter output");
        if (available) { char buffer[4096]; DWORD count = 0; check(::ReadFile(read.value, buffer, std::min<DWORD>(available, sizeof(buffer)), &count, nullptr) && count, "cannot read farm adapter output"); append(buffer, count); continue; }
        if (child.done) break;
        auto const state = ::WaitForSingleObject(child.p.hProcess, 0); check(state != WAIT_FAILED, "cannot inspect farm adapter exit");
        if (state == WAIT_OBJECT_0) { child.done = true; continue; }
        ::WaitForSingleObject(child.p.hProcess, 20);
    }
    DWORD code = 0; check(::GetExitCodeProcess(child.p.hProcess, &code), "cannot read farm adapter exit"); result.exit_code = int(code);
    result.cap_kind = "windows-job-and-process-commit";
#elif defined(__linux__)
    struct Pipe
    { int fd[2]{-1, -1}; ~Pipe() { for (int value : fd) if (value >= 0) ::close(value); } } pipe;
    check(!::pipe2(pipe.fd, O_CLOEXEC), "cannot create farm adapter output pipe");
    for (int& fd : pipe.fd) if (fd <= STDERR_FILENO)
    { int moved = ::fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1); check(moved >= 0, "cannot isolate farm adapter descriptors"); ::close(fd); fd = moved; }
    check(::fcntl(pipe.fd[0], F_SETFL, O_NONBLOCK) >= 0, "cannot make farm adapter output cancellable");
    std::vector<char*> argv; for (auto& arg : args) argv.push_back(arg.data()); argv.push_back(nullptr);
    struct rlimit inherited{}; check(!::getrlimit(RLIMIT_AS, &inherited), "cannot inspect farm adapter memory limit");
    uint64_t cap = limits.memory_bytes;
    if (inherited.rlim_cur != RLIM_INFINITY) cap = std::min(cap, uint64_t(inherited.rlim_cur));
    if (inherited.rlim_max != RLIM_INFINITY) cap = std::min(cap, uint64_t(inherited.rlim_max));
    check(cap && cap <= std::numeric_limits<rlim_t>::max(), "invalid farm adapter memory limit"); struct rlimit bound{rlim_t(cap), rlim_t(cap)};
    cpu_set_t inherited_cpu, selected_cpu; CPU_ZERO(&inherited_cpu); CPU_ZERO(&selected_cpu);
    check(!::sched_getaffinity(0, sizeof(inherited_cpu), &inherited_cpu), "cannot inspect farm adapter CPU allocation");
    for (unsigned cpu = 0; cpu < CPU_SETSIZE && result.cpu_lanes < command.threads; ++cpu)
        if (CPU_ISSET(cpu, &inherited_cpu)) { CPU_SET(cpu, &selected_cpu); ++result.cpu_lanes; }
    check(result.cpu_lanes, "farm adapter has no admitted CPU lane");
    struct Child
    {
        pid_t pid = 0; bool done = false; int status = 0; siginfo_t information{};
        ~Child() { if (pid > 0) { ::kill(-pid, SIGKILL); while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {} } }
    } child;
    // posix_spawn has no portable pre-exec memory-limit action. Prebuild all
    // storage above; the fork child uses only async-signal-safe platform calls.
    calibration_process::check_running(until, limits.cancelled);
    child.pid = ::fork(); check(child.pid >= 0, "cannot start installed farm adapter");
    if (!child.pid)
    {
        if (::setpgid(0, 0) || ::setrlimit(RLIMIT_AS, &bound) || ::syscall(SYS_sched_setaffinity, 0, sizeof(selected_cpu), &selected_cpu)) ::_exit(125);
        int input = ::open("/dev/null", O_RDONLY);
        if (input < 0 || ::dup2(input, STDIN_FILENO) < 0 || ::dup2(pipe.fd[1], STDOUT_FILENO) < 0 || ::dup2(pipe.fd[1], STDERR_FILENO) < 0) ::_exit(125);
        if (input > STDERR_FILENO) ::close(input); ::close(pipe.fd[0]); ::close(pipe.fd[1]);
        ::execve(argv[0], argv.data(), environ); ::_exit(126);
    }
    (void)::setpgid(child.pid, child.pid); ::close(pipe.fd[1]); pipe.fd[1] = -1; bool eof = false;
    while (!child.done || !eof)
    {
        calibration_process::check_running(until, limits.cancelled);
        char buffer[4096]; auto const count = ::read(pipe.fd[0], buffer, sizeof(buffer));
        if (count > 0) { append(buffer, size_t(count)); continue; }
        if (!count) eof = true;
        else check(errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR, "cannot read farm adapter output");
        if (!child.done)
        {
            int const error = ::waitid(P_PID, id_t(child.pid), &child.information, WEXITED | WNOHANG | WNOWAIT);
            check(!error || errno == EINTR, "cannot inspect farm adapter exit");
            if (!error && child.information.si_pid == child.pid) child.done = true;
        }
        if (!child.done || !eof) { pollfd wait{pipe.fd[0], POLLIN | POLLHUP, 0}; (void)::poll(&wait, 1, 20); }
    }
    result.exit_code = child.information.si_code == CLD_EXITED ? child.information.si_status : -1;
    result.cap_kind = "linux-address-space"; result.memory_cap_bytes = cap;
#else
    throw std::runtime_error("bounded native farm adapters are unavailable on this platform");
#endif
    result.wall_ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count());
    return result;
}
} // namespace group_grow::farm
