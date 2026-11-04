// Private, bounded process ownership for explicit farming calibration only.
#pragma once
#include "host/GroupHarvesterProfile.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#ifdef _WIN32
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace group_grow::calibration_process
{
using Clock = std::chrono::steady_clock;
struct Limits
{
    uint64_t memory_bytes = 0, timeout_ms = 0, output_bytes = 0;
    std::function<bool()> cancelled;
};
struct MemoryCap
{
    std::string kind;
    uint64_t requested_bytes = 0, enforced_bytes = 0, baseline_bytes = 0;
};
inline void check(bool condition, char const* message)
{ if (!condition) throw std::runtime_error(message); }
inline Clock::time_point deadline(uint64_t timeout_ms)
{
    auto const now = Clock::now();
    auto const maximum = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now).count();
    check(timeout_ms && maximum > 0 && timeout_ms <= uint64_t(maximum), "invalid calibration deadline");
    return now + std::chrono::milliseconds(timeout_ms);
}
inline void check_running(Clock::time_point until, std::function<bool()> const& cancelled)
{
    check(!cancelled || !cancelled(), "calibration cancelled");
    check(Clock::now() < until, "calibration deadline expired");
}
#ifdef _WIN32
struct NativeHandle
{
    HANDLE value = nullptr;
    ~NativeHandle() { if (value && value != INVALID_HANDLE_VALUE) ::CloseHandle(value); }
    NativeHandle() = default;
    explicit NativeHandle(HANDLE handle) : value(handle) {}
    NativeHandle(NativeHandle const&) = delete;
    NativeHandle& operator=(NativeHandle const&) = delete;
};
#endif
inline MemoryCap install_memory_cap(uint64_t bytes)
{
    check(bytes, "calibration memory cap must be positive");
#ifdef _WIN32
    check(bytes <= SIZE_MAX, "calibration memory cap exceeds addressable memory");
    PROCESS_MEMORY_COUNTERS_EX usage{};
    check(::K32GetProcessMemoryInfo(::GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&usage), sizeof(usage)),
        "cannot inspect calibration process commitment");
    check(usage.PrivateUsage <= bytes, "calibration memory cap is below process startup commitment");
    NativeHandle job(::CreateJobObjectW(nullptr, nullptr));
    check(job.value, "cannot create calibration memory job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
    limits.ProcessMemoryLimit = SIZE_T(bytes);
    check(::SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) &&
        ::AssignProcessToJobObject(job.value, ::GetCurrentProcess()), "cannot enforce calibration process-memory cap");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION actual{}; BOOL member = FALSE;
    check(::IsProcessInJob(::GetCurrentProcess(), job.value, &member) && member &&
        ::QueryInformationJobObject(job.value, JobObjectExtendedLimitInformation, &actual, sizeof(actual), nullptr) &&
        (actual.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) && actual.ProcessMemoryLimit <= bytes &&
        actual.ProcessMemoryLimit && usage.PrivateUsage <= actual.ProcessMemoryLimit, "cannot verify calibration process-memory cap");
    // Cap-only jobs remain associated until process exit after their last handle
    // closes. Kill-on-close belongs to the supervising parent's separate job.
    return {"windows-process-commit", bytes, uint64_t(actual.ProcessMemoryLimit), uint64_t(usage.PrivateUsage)};
#elif defined(__linux__)
    unsigned long long pages = 0;
    {
        std::ifstream input("/proc/self/statm");
        check(bool(input >> pages), "cannot inspect calibration address-space baseline");
    }
    long const page_bytes = ::sysconf(_SC_PAGESIZE);
    check(page_bytes > 0 && pages <= UINT64_MAX / uint64_t(page_bytes), "invalid calibration address-space baseline");
    uint64_t const baseline = uint64_t(pages) * uint64_t(page_bytes);
    struct rlimit inherited{}; check(!::getrlimit(RLIMIT_AS, &inherited), "cannot inspect calibration address-space limit");
    uint64_t enforced = bytes;
    if (inherited.rlim_cur != RLIM_INFINITY) enforced = std::min(enforced, uint64_t(inherited.rlim_cur));
    if (inherited.rlim_max != RLIM_INFINITY) enforced = std::min(enforced, uint64_t(inherited.rlim_max));
    check(enforced && enforced <= std::numeric_limits<rlim_t>::max() && baseline <= enforced,
        "calibration memory cap is below process startup address space");
    struct rlimit desired{rlim_t(enforced), rlim_t(enforced)};
    check(!::setrlimit(RLIMIT_AS, &desired), "cannot enforce calibration address-space cap");
    struct rlimit actual{};
    check(!::getrlimit(RLIMIT_AS, &actual) && actual.rlim_cur != RLIM_INFINITY && actual.rlim_max != RLIM_INFINITY &&
        actual.rlim_cur <= actual.rlim_max && uint64_t(actual.rlim_max) <= bytes && actual.rlim_cur,
        "cannot verify calibration address-space cap");
    return {"linux-address-space", bytes, uint64_t(actual.rlim_cur), baseline};
#else
    (void)bytes; throw std::runtime_error("calibration process limits are unavailable on this platform");
#endif
}

inline std::filesystem::path executable()
{
#ifdef _WIN32
    std::array<wchar_t, 32768> path{};
    DWORD const length = ::GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    check(length && length < path.size(), "cannot locate calibration executable");
    return std::filesystem::path(std::wstring(path.data(), length));
#elif defined(__linux__)
    std::array<char, 32768> path{};
    auto const length = ::readlink("/proc/self/exe", path.data(), path.size());
    check(length > 0 && size_t(length) < path.size(), "cannot locate calibration executable");
    return std::filesystem::path(std::string(path.data(), size_t(length)));
#else
    throw std::runtime_error("calibration worker processes are unavailable on this platform");
#endif
}

inline std::string run_worker(std::filesystem::path const& plot, uint64_t samples, Limits const& limits)
{
    auto const until = deadline(limits.timeout_ms);
    check_running(until, limits.cancelled);
    check(samples && limits.memory_bytes && limits.output_bytes && limits.output_bytes <= std::string().max_size(),
        "invalid calibration worker limits");
    std::error_code error; auto const absolute = std::filesystem::absolute(plot, error);
    check(!error, "cannot resolve calibration input path");
    auto const self = executable();
    std::string output;
    auto append = [&](char const* buffer, size_t count)
    {
        check(uint64_t(count) <= limits.output_bytes - output.size(), "calibration worker output exceeded its byte limit");
        output.append(buffer, count);
    };
#ifdef _WIN32
    check(limits.memory_bytes <= SIZE_MAX, "calibration memory cap exceeds addressable memory");
    NativeHandle read, write, input, job;
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    check(::CreatePipe(&read.value, &write.value, &attributes, 0), "cannot create calibration output pipe");
    check(::SetHandleInformation(read.value, HANDLE_FLAG_INHERIT, 0), "cannot restrict calibration pipe inheritance");
    input.value = ::CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes, OPEN_EXISTING, 0, nullptr);
    check(input.value != INVALID_HANDLE_VALUE, "cannot create calibration input handle");
    job.value = ::CreateJobObjectW(nullptr, nullptr);
    check(job.value, "cannot create calibration supervisor job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION bound{};
    bound.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    bound.ProcessMemoryLimit = SIZE_T(limits.memory_bytes);
    check(::SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &bound, sizeof(bound)),
        "cannot configure calibration supervisor cap");
    SIZE_T attribute_bytes = 0;
    ::InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
    check(attribute_bytes, "cannot size calibration handle inheritance");
    std::vector<uint8_t> storage(attribute_bytes);
    auto list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    check(::InitializeProcThreadAttributeList(list, 1, 0, &attribute_bytes), "cannot initialize calibration handle inheritance");
    struct Attributes { LPPROC_THREAD_ATTRIBUTE_LIST value; ~Attributes() { ::DeleteProcThreadAttributeList(value); } } cleanup{list};
    std::array<HANDLE, 2> handles{write.value, input.value};
    check(::UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles.data(), sizeof(handles), nullptr, nullptr),
        "cannot restrict calibration inherited handles");
    auto quote = profile_detail::windows_argument;
    std::wstring command = quote(self.wstring()) + L" plan --config NUL --calibrate --calibration-worker --calibration-samples " +
        std::to_wstring(samples) + L" --calibration-memory-bytes " + std::to_wstring(limits.memory_bytes) +
        L" --calibration-timeout-ms " + std::to_wstring(limits.timeout_ms) + L" --json -- " + quote(absolute.wstring());
    STARTUPINFOEXW startup{}; startup.StartupInfo.cb = sizeof(startup); startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdOutput = write.value; startup.StartupInfo.hStdError = write.value; startup.StartupInfo.hStdInput = input.value;
    startup.lpAttributeList = list;
    struct Child
    {
        PROCESS_INFORMATION process{};
        HANDLE job = nullptr;
        bool done = false;
        ~Child()
        {
            if (!process.hProcess) return;
            if (!done)
            {
                ::TerminateJobObject(job, 4); ::TerminateProcess(process.hProcess, 4);
                ::WaitForSingleObject(process.hProcess, INFINITE);
            }
            ::CloseHandle(process.hThread); ::CloseHandle(process.hProcess);
        }
    } child;
    child.job = job.value;
    check_running(until, limits.cancelled);
    PROCESS_INFORMATION created{};
    check(::CreateProcessW(self.c_str(), command.data(), nullptr, nullptr, TRUE,
        CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo, &created),
        "cannot start calibration worker");
    child.process = created;
    ::CloseHandle(write.value); write.value = nullptr;
    check(::AssignProcessToJobObject(job.value, child.process.hProcess), "cannot assign calibration worker to its memory job");
    check_running(until, limits.cancelled);
    check(::ResumeThread(child.process.hThread) != DWORD(-1), "cannot resume capped calibration worker");
    for (;;)
    {
        check_running(until, limits.cancelled);
        DWORD available = 0;
        if (!::PeekNamedPipe(read.value, nullptr, 0, nullptr, &available, nullptr))
            check(::GetLastError() == ERROR_BROKEN_PIPE, "cannot inspect calibration worker output");
        if (available)
        {
            char buffer[4096]; DWORD count = 0;
            check(::ReadFile(read.value, buffer, std::min<DWORD>(available, sizeof(buffer)), &count, nullptr) && count,
                "cannot read calibration worker output");
            append(buffer, count); continue;
        }
        if (child.done) break;
        DWORD const waited = ::WaitForSingleObject(child.process.hProcess, 0);
        check(waited != WAIT_FAILED, "cannot inspect calibration worker exit");
        // Inspect the pipe again after observing exit: its final write may
        // have raced with the previous PeekNamedPipe.
        if (waited == WAIT_OBJECT_0) { child.done = true; continue; }
        ::WaitForSingleObject(child.process.hProcess, 20);
    }
    DWORD code = 0;
    check(::GetExitCodeProcess(child.process.hProcess, &code), "cannot read calibration worker exit code");
    if (code) throw std::runtime_error("calibration worker failed: " + output);
#elif defined(__linux__)
    struct Descriptors
    {
        std::array<int, 2> values{-1, -1};
        ~Descriptors() { for (auto value : values) if (value >= 0) ::close(value); }
    } pipe;
    check(!::pipe(pipe.values.data()), "cannot create calibration output pipe");
    for (auto& value : pipe.values)
    {
        if (value <= STDERR_FILENO)
        {
            int const promoted = ::fcntl(value, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
            check(promoted >= 0, "cannot isolate calibration pipe descriptors"); ::close(value); value = promoted;
        }
        check(!::fcntl(value, F_SETFD, FD_CLOEXEC), "cannot restrict calibration pipe inheritance");
    }
    check(::fcntl(pipe.values[0], F_SETFL, O_NONBLOCK) >= 0, "cannot make calibration output cancellable");
    posix_spawn_file_actions_t actions;
    check(!::posix_spawn_file_actions_init(&actions), "cannot initialize calibration spawn actions");
    struct Actions { posix_spawn_file_actions_t* value; ~Actions() { ::posix_spawn_file_actions_destroy(value); } } cleanup{&actions};
    int action = ::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    if (!action) action = ::posix_spawn_file_actions_adddup2(&actions, pipe.values[1], STDOUT_FILENO);
    if (!action) action = ::posix_spawn_file_actions_adddup2(&actions, pipe.values[1], STDERR_FILENO);
    if (!action) action = ::posix_spawn_file_actions_addclose(&actions, pipe.values[0]);
    if (!action) action = ::posix_spawn_file_actions_addclose(&actions, pipe.values[1]);
    check(!action, "cannot configure calibration worker descriptors");
    std::vector<std::string> arguments{self.string(), "plan", "--config", "/dev/null", "--calibrate", "--calibration-worker",
        "--calibration-samples", std::to_string(samples), "--calibration-memory-bytes", std::to_string(limits.memory_bytes),
        "--calibration-timeout-ms", std::to_string(limits.timeout_ms), "--json", "--", absolute.string()};
    std::vector<char*> argv; for (auto& argument : arguments) argv.push_back(argument.data()); argv.push_back(nullptr);
    struct Child
    {
        pid_t pid = 0;
        bool done = false;
        int status = 0;
        ~Child()
        {
            if (pid <= 0 || done) return;
            ::kill(pid, SIGKILL);
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        }
    } child;
    check_running(until, limits.cancelled);
    pid_t created = 0;
    check(!::posix_spawn(&created, argv.front(), &actions, nullptr, argv.data(), environ), "cannot start calibration worker");
    child.pid = created;
    ::close(pipe.values[1]); pipe.values[1] = -1;
    bool eof = false;
    for (;;)
    {
        check_running(until, limits.cancelled);
        char buffer[4096]; auto const count = ::read(pipe.values[0], buffer, sizeof(buffer));
        if (count > 0) { append(buffer, size_t(count)); continue; }
        if (!count) eof = true;
        else check(errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR, "cannot read calibration worker output");
        if (!child.done)
        {
            auto const waited = ::waitpid(child.pid, &child.status, WNOHANG);
            if (waited == child.pid) child.done = true;
            else if (waited < 0 && errno != EINTR)
            { child.done = errno == ECHILD; throw std::runtime_error("cannot reap calibration worker"); }
        }
        if (child.done && eof) break;
        pollfd descriptor{pipe.values[0], POLLIN, 0};
        check(::poll(&descriptor, 1, 20) >= 0 || errno == EINTR, "cannot wait for calibration worker output");
    }
    if (!WIFEXITED(child.status) || WEXITSTATUS(child.status)) throw std::runtime_error("calibration worker failed: " + output);
#else
    throw std::runtime_error("calibration worker processes are unavailable on this platform");
#endif
    check(!output.empty(), "calibration worker returned no result");
    return output;
}

// Chainer does not provide a cancellation callback. The worker must start this
// watchdog successfully before opening the plot; direct hidden calls need the
// same deadline as supervised workers. Cancellation callbacks must not block.
class Deadline
{
public:
    Deadline(uint64_t timeout_ms, std::function<bool()> cancelled = {})
        : until_(deadline(timeout_ms)), cancelled_(std::move(cancelled)), thread_([this]
        {
            for (;;)
            {
                if (finished_.load(std::memory_order_relaxed)) return;
                bool cancelled = false;
                try { cancelled = cancelled_ && cancelled_(); }
                catch (...) { cancelled = true; }
                if (cancelled || Clock::now() >= until_) std::_Exit(4);
                std::unique_lock lock(mutex_);
                changed_.wait_until(lock, std::min(until_, Clock::now() + std::chrono::milliseconds(20)),
                    [&] { return finished_.load(std::memory_order_relaxed); });
            }
        }) {}
    ~Deadline()
    { finished_.store(true, std::memory_order_relaxed); changed_.notify_all(); thread_.join(); }
    Deadline(Deadline const&) = delete;
    Deadline& operator=(Deadline const&) = delete;
private:
    Clock::time_point until_;
    std::function<bool()> cancelled_;
    std::atomic<bool> finished_{false};
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread thread_;
};
} // namespace group_grow::calibration_process
