// Private, bounded self-workers for explicit calibration and migration proofs.
#pragma once
#include "host/GroupFarmResources.hpp"
#include <atomic>
#include <condition_variable>
#include <climits>
#include <functional>
#include <limits>
#include <mutex>
#ifdef _WIN32
#include <psapi.h>
#else
#include <sys/resource.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
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
inline uint64_t install_cpu_cap(uint64_t requested)
{
#ifdef _WIN32
    USHORT groups = 1, group = 0;
    check(::GetProcessGroupAffinity(::GetCurrentProcess(), &groups, &group) && groups == 1,
        "bounded proof CPU affinity requires a single Windows processor group");
    DWORD_PTR allowed = 0, system = 0;
    check(::GetProcessAffinityMask(::GetCurrentProcess(), &allowed, &system) && allowed,
        "cannot inspect bounded proof CPU affinity");
    uint64_t const count = uint64_t(std::popcount(allowed));
    check(requested <= count, "proof CPU cap exceeds inherited affinity");
    if (!requested) return count;
    DWORD_PTR selected = 0; uint64_t picked = 0;
    for (unsigned id = 0; id < sizeof(allowed) * CHAR_BIT && picked < requested; ++id)
        if (allowed & (DWORD_PTR(1) << id)) { selected |= DWORD_PTR(1) << id; ++picked; }
    DWORD_PTR actual = 0;
    check(picked == requested && ::SetProcessAffinityMask(::GetCurrentProcess(), selected) &&
        ::GetProcessAffinityMask(::GetCurrentProcess(), &actual, &system) && actual == selected,
        "cannot enforce bounded proof CPU affinity");
    return requested;
#elif defined(__linux__)
    cpu_set_t allowed, selected, actual; CPU_ZERO(&allowed); CPU_ZERO(&selected); CPU_ZERO(&actual);
    check(!::sched_getaffinity(0, sizeof(allowed), &allowed), "cannot inspect bounded proof CPU affinity");
    uint64_t const count = uint64_t(CPU_COUNT(&allowed)); check(count && requested <= count, "proof CPU cap exceeds inherited affinity");
    if (!requested) return count;
    uint64_t picked = 0;
    for (int id = 0; id < CPU_SETSIZE && picked < requested; ++id)
        if (CPU_ISSET(id, &allowed)) { CPU_SET(id, &selected); ++picked; }
    check(picked == requested && !::sched_setaffinity(0, sizeof(selected), &selected) &&
          !::sched_getaffinity(0, sizeof(actual), &actual) && CPU_EQUAL(&actual, &selected), "cannot enforce bounded proof CPU affinity");
    return requested;
#else
    check(!requested, "explicit proof CPU affinity is unsupported on this platform");
    auto const count = profile_detail::logical_cpus(); check(count, "cannot inspect inherited proof CPU capacity"); return count;
#endif
}
// Save/restore affinity around upstream code that creates its own workers.
// Linux scopes the calling thread and its children; Windows scopes the process
// because new Windows threads inherit process affinity. Join workers before
// restoration; callers serialize these caps within a construction session.
class ScopedCpuCap
{
    uint64_t lanes_ = 0;
#ifdef _WIN32
    DWORD_PTR original_ = 0;
    bool active_ = false;
#elif defined(__linux__)
    cpu_set_t original_{};
    bool active_ = false;
#endif
public:
    explicit ScopedCpuCap(uint64_t requested)
    {
#ifdef _WIN32
        DWORD_PTR system = 0;
        check(::GetProcessAffinityMask(::GetCurrentProcess(), &original_, &system) && original_, "cannot save scoped CPU affinity");
        try { lanes_ = install_cpu_cap(requested); active_ = requested != 0; }
        catch (...) { (void)::SetProcessAffinityMask(::GetCurrentProcess(), original_); throw; }
#elif defined(__linux__)
        check(!::sched_getaffinity(0, sizeof(original_), &original_), "cannot save scoped CPU affinity");
        try { lanes_ = install_cpu_cap(requested); active_ = requested != 0; }
        catch (...)
        {
            // install_cpu_cap can fail its read-back after applying the mask.
            // An incomplete constructor still owns restoration of that change.
            (void)::sched_setaffinity(0, sizeof(original_), &original_); throw;
        }
#else
        lanes_ = install_cpu_cap(requested);
#endif
    }
    ~ScopedCpuCap()
    {
#ifdef _WIN32
        if (active_) (void)::SetProcessAffinityMask(::GetCurrentProcess(), original_);
#elif defined(__linux__)
        if (active_) (void)::sched_setaffinity(0, sizeof(original_), &original_);
#endif
    }
    ScopedCpuCap(ScopedCpuCap const&) = delete;
    ScopedCpuCap& operator=(ScopedCpuCap const&) = delete;
    uint64_t lanes() const { return lanes_; }
    void restore()
    {
#ifdef _WIN32
        if (!active_) return;
        DWORD_PTR actual = 0, system = 0;
        check(::SetProcessAffinityMask(::GetCurrentProcess(), original_) &&
            ::GetProcessAffinityMask(::GetCurrentProcess(), &actual, &system) && actual == original_, "cannot restore scoped CPU affinity");
        active_ = false;
#elif defined(__linux__)
        if (!active_) return;
        cpu_set_t actual; CPU_ZERO(&actual);
        check(!::sched_setaffinity(0, sizeof(original_), &original_) &&
              !::sched_getaffinity(0, sizeof(actual), &actual) && CPU_EQUAL(&actual, &original_), "cannot restore scoped CPU affinity");
        active_ = false;
#endif
    }
};
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

enum class Worker { calibration, migration_proof };
class WorkerPaused : public std::runtime_error
{
public:
    WorkerPaused() : std::runtime_error("bounded proof worker paused for farm resources") {}
};
inline std::string run_worker(std::filesystem::path const& plot, uint64_t samples, Limits const& limits,
                              Worker worker = Worker::calibration, uint64_t proof_cpus = 0, uint64_t proof_solves = 1,
                              std::function<bool(farm::SupervisedProcess const&)> const& observe = {})
{
    check(worker == Worker::calibration || worker == Worker::migration_proof, "invalid bounded plot worker mode");
    check(!proof_cpus || (worker == Worker::migration_proof && proof_cpus <= 1024), "invalid bounded proof CPU request");
#if !defined(__linux__) && !defined(_WIN32)
    check(!proof_cpus, "explicit proof CPU affinity is unsupported on this platform");
#endif
    check(worker != Worker::migration_proof || (proof_solves && proof_solves <= samples), "invalid bounded proof solve budget");
    auto const until = deadline(limits.timeout_ms);
    check_running(until, limits.cancelled);
    check(samples && limits.memory_bytes && limits.output_bytes && limits.output_bytes <= std::string().max_size(),
        "invalid calibration worker limits");
    std::error_code error; auto const absolute = std::filesystem::absolute(plot, error);
    check(!error, "cannot resolve calibration input path");
    auto const self = executable();
    std::string output;
    static std::atomic<uint64_t> generations{1};
    farm::SupervisedProcess observed; observed.generation = generations.fetch_add(1);
    check(observed.generation && observed.generation != UINT64_MAX, "bounded worker observation generation exhausted");
    auto const poll_interval = std::chrono::milliseconds(20);
#ifdef __linux__
    struct ExitDescriptor
    {
        int value = -1;
        ~ExitDescriptor() { if (value >= 0) ::close(value); }
    } exit_signal;
#endif
    auto monitor = [&]
    {
        if (!observe) return;
        try { if (!observe(observed)) throw WorkerPaused(); }
        catch (farm::ProcessObservationUnavailable const&)
        {
            // A terminating process can lose its counters before its exit
            // becomes waitable. Converge only on native exit notification,
            // within the existing observation cadence and operation deadline;
            // missing counters never authorize another resource observation.
            auto const exit_until = std::min(until, Clock::now() + poll_interval);
            while (!observed.exited())
            {
                check_running(until, limits.cancelled);
                auto const left = std::chrono::ceil<std::chrono::milliseconds>(exit_until - Clock::now()).count();
                if (left <= 0) break;
#ifdef _WIN32
                (void)::WaitForSingleObject(observed.handle, DWORD(left));
#elif defined(__linux__)
                if (exit_signal.value < 0) break; // Unsupported/blocked pidfd: retain immediate fail-closed behavior.
                pollfd descriptor{exit_signal.value, POLLIN, 0};
                if (::poll(&descriptor, 1, int(left)) < 0 && errno == EINTR) continue;
#endif
                break;
            }
            check_running(until, limits.cancelled);
            if (!observed.exited()) throw WorkerPaused();
        }
    };
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
        L" --calibration-timeout-ms " + std::to_wstring(limits.timeout_ms) +
        (worker == Worker::migration_proof ? L" --replot-proof-worker --replot-proof-solves " + std::to_wstring(proof_solves) : L"") +
        (proof_cpus ? L" --replot-proof-cpus " + std::to_wstring(proof_cpus) : L"") + L" --json -- " + quote(absolute.wstring());
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
    observed.handle = child.process.hProcess;
    ::CloseHandle(write.value); write.value = nullptr;
    check(::AssignProcessToJobObject(job.value, child.process.hProcess), "cannot assign calibration worker to its memory job");
    if (proof_cpus)
    {
        DWORD_PTR allowed = 0, system = 0, selected = 0; uint64_t count = 0;
        check(::GetProcessAffinityMask(child.process.hProcess, &allowed, &system) && allowed, "cannot inspect suspended proof CPU affinity");
        for (unsigned id = 0; id < sizeof(allowed) * CHAR_BIT && count < proof_cpus; ++id)
            if (allowed & (DWORD_PTR(1) << id)) { selected |= DWORD_PTR(1) << id; ++count; }
        DWORD_PTR actual = 0;
        check(count == proof_cpus && ::SetProcessAffinityMask(child.process.hProcess, selected) &&
            ::GetProcessAffinityMask(child.process.hProcess, &actual, &system) && actual == selected,
            "cannot enforce suspended proof CPU affinity");
    }
    check_running(until, limits.cancelled);
    check(::ResumeThread(child.process.hThread) != DWORD(-1), "cannot resume capped calibration worker");
    for (;;)
    {
        check_running(until, limits.cancelled);
        if (observe && !child.done)
        {
            DWORD const state = ::WaitForSingleObject(child.process.hProcess, 0);
            check(state != WAIT_FAILED, "cannot inspect monitored proof worker exit");
            if (state == WAIT_OBJECT_0) child.done = true;
            else monitor();
        }
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
        ::WaitForSingleObject(child.process.hProcess, DWORD(poll_interval.count()));
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
        "--calibration-timeout-ms", std::to_string(limits.timeout_ms)};
    if (worker == Worker::migration_proof)
        arguments.insert(arguments.end(), {"--replot-proof-worker", "--replot-proof-solves", std::to_string(proof_solves)});
    if (proof_cpus) arguments.insert(arguments.end(), {"--replot-proof-cpus", std::to_string(proof_cpus)});
    arguments.insert(arguments.end(), {"--json", "--", absolute.string()});
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
    {
        // Only this spawning thread changes affinity. Existing broker/worker
        // threads retain their independently admitted CPU scope.
        ScopedCpuCap cpu(proof_cpus);
        check(!::posix_spawn(&created, argv.front(), &actions, nullptr, argv.data(), environ), "cannot start calibration worker");
        // Adopt before restoration, whose read-back may throw.
        child.pid = created;
        cpu.restore();
    }
    observed.pid = child.pid;
#ifdef SYS_pidfd_open
    // The owned child is still unreaped, so this cannot attach to a reused PID.
    // A missing syscall, sandbox restriction or descriptor failure preserves
    // conservative pausing instead of weakening the observation requirement.
    if (observe) exit_signal.value = int(::syscall(SYS_pidfd_open, child.pid, 0));
#endif
    ::close(pipe.values[1]); pipe.values[1] = -1;
    bool eof = false;
    for (;;)
    {
        check_running(until, limits.cancelled);
        if (observe && !child.done)
        {
            auto const waited = ::waitpid(child.pid, &child.status, WNOHANG);
            if (waited == child.pid) child.done = true;
            else if (waited < 0 && errno != EINTR)
            { child.done = errno == ECHILD; throw std::runtime_error("cannot reap monitored proof worker"); }
            if (!child.done) monitor();
        }
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
        check(::poll(&descriptor, 1, int(poll_interval.count())) >= 0 || errno == EINTR, "cannot wait for calibration worker output");
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
