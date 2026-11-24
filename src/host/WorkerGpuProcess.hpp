// Private scratch-free handoff to a fresh worker process after device loss.
#pragma once
#include "host/GroupCalibrationProcess.hpp"
#include <charconv>
#include <cstring>
#include <span>
#include <optional>
#include <new>
#include <type_traits>
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#endif

namespace group_grow::worker_gpu
{
using calibration_process::check;
enum Probe : uint32_t { generation, canary, budget_only };
struct Profile { uint64_t borrowed = 0, owning = 0, device = 0, minimum_owning = 0; char tier[16]{}; };
struct ResourcesUnavailable : std::runtime_error
{
    uint64_t host_bytes;
    ResourcesUnavailable(char const* message, uint64_t bytes) : std::runtime_error(message), host_bytes(bytes) {}
};
struct DeviceLost : std::runtime_error { using std::runtime_error::runtime_error; };
struct Request
{
    static constexpr uint64_t identity = 0x5843484750553031ULL;
    uint64_t format = identity, mapping_bytes = 0, host_bytes = 0, gpu_bytes = 0, parent = 0;
    uint64_t count = 0, safe_gpu_bytes = 0, original_total = 0, measured_room = UINT64_MAX, next_upgrade = UINT64_MAX;
    uint32_t k = 18, strength = 2, threads = 1, probe = 0;
    // Read only after the child exits successfully. Bounds are canonical
    // fragments; the parent never publishes a partial child result.
    uint32_t complete = 0, lost = 0, unavailable = 0, live_budget_required = 0;
    std::array<Profile, 6> profiles{};
    std::array<uint8_t, 32> id{};
    char tier[16]{}, physical[128]{}, name[256]{}, backend[32]{}, error[241]{};
};
static_assert(std::is_trivially_copyable_v<Request>);
constexpr size_t header_bytes = (sizeof(Request) + 7) & ~size_t(7);

inline size_t mapping_size(uint64_t words)
{
    check(words && words <= UINT32_MAX && words <= (SIZE_MAX - header_bytes) / sizeof(uint64_t),
          "invalid GPU worker output bound");
    return header_bytes + size_t(words) * sizeof(uint64_t);
}
inline uint64_t number(char const* text)
{
    uint64_t value = 0; size_t length = std::strlen(text);
    auto result = std::from_chars(text, text + length, value);
    check(result.ec == std::errc{} && result.ptr == text + length, "invalid GPU worker handle or size");
    return value;
}

class Mapping
{
#ifdef _WIN32
    HANDLE handle_ = nullptr;
#elif defined(__linux__)
    int fd_ = -1;
#endif
    void* page_ = nullptr;
    size_t bytes_ = 0;
    Mapping() = default;
public:
    Mapping(Mapping const&) = delete;
    Mapping& operator=(Mapping const&) = delete;
    ~Mapping()
    {
#ifdef _WIN32
        if (page_) ::UnmapViewOfFile(page_);
        if (handle_) ::CloseHandle(handle_);
#elif defined(__linux__)
        if (page_) ::munmap(page_, bytes_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    Request& request() const { return *static_cast<Request*>(page_); }
    std::span<uint64_t> output() const
    { return {reinterpret_cast<uint64_t*>(static_cast<char*>(page_) + header_bytes), (bytes_ - header_bytes) / 8}; }

    static std::unique_ptr<Mapping> open(size_t bytes, std::optional<uintptr_t> inherited = {})
    {
        check(bytes > header_bytes && bytes == mapping_size((bytes - header_bytes) / 8),
              "invalid GPU worker mapping size");
        auto value = std::unique_ptr<Mapping>(new Mapping);
        value->bytes_ = bytes;
#ifdef _WIN32
        if (inherited) value->handle_ = reinterpret_cast<HANDLE>(*inherited);
        else {
            SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
            value->handle_ = ::CreateFileMappingW(INVALID_HANDLE_VALUE, &attributes, PAGE_READWRITE,
                DWORD(uint64_t(bytes) >> 32), DWORD(bytes), nullptr);
        }
        check(value->handle_, "cannot open anonymous GPU worker output");
        value->page_ = ::MapViewOfFile(value->handle_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, bytes);
        check(value->page_, "cannot map GPU worker output");
#elif defined(__linux__)
        if (inherited) {
            check(*inherited == STDIN_FILENO, "invalid inherited GPU worker descriptor");
            value->fd_ = STDIN_FILENO;
            struct stat st{};
            int const seals = ::fcntl(value->fd_, F_GET_SEALS);
            check(seals >= 0 && (seals & (F_SEAL_GROW | F_SEAL_SHRINK)) == (F_SEAL_GROW | F_SEAL_SHRINK) &&
                  !::fstat(value->fd_, &st) && st.st_size >= 0 && uint64_t(st.st_size) == bytes,
                  "invalid sealed GPU worker mapping");
        } else {
            value->fd_ = ::memfd_create("xchplot2-gpu-worker", MFD_CLOEXEC | MFD_ALLOW_SEALING);
            check(value->fd_ >= 0 && !::ftruncate(value->fd_, off_t(bytes)) &&
                  !::fcntl(value->fd_, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL),
                  "cannot reserve anonymous GPU worker output");
        }
        void* view = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, value->fd_, 0);
        check(view != MAP_FAILED, "cannot map GPU worker output");
        value->page_ = view;
#else
        throw std::runtime_error("isolated GPU recovery requires Windows or Linux");
#endif
        if (!inherited) {
            new (value->page_) Request{};
            value->request().mapping_bytes = bytes;
#ifdef __linux__
            value->request().parent = uint64_t(::getpid());
#endif
        } else {
            check(value->request().format == Request::identity && value->request().mapping_bytes == bytes &&
                  !value->request().complete, "invalid inherited GPU worker request");
#ifdef __linux__
            check(!::prctl(PR_SET_PDEATHSIG, SIGKILL) && uint64_t(::getppid()) == value->request().parent,
                  "GPU worker lost its supervising process");
#endif
        }
        return value;
    }
    void run(std::function<void()> const& check_cancel)
    {
        auto const exe = calibration_process::executable();
        check_cancel();
#ifdef _WIN32
        calibration_process::NativeHandle job;
        job.value = ::CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        check(job.value && ::SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)),
              "cannot contain GPU worker process");
        SIZE_T size = 0; ::InitializeProcThreadAttributeList(nullptr, 2, 0, &size);
        std::vector<std::byte> storage(size);
        auto attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
        check(::InitializeProcThreadAttributeList(attrs, 2, 0, &size), "cannot initialize GPU worker handles");
        struct Attributes { LPPROC_THREAD_ATTRIBUTE_LIST value; ~Attributes() { ::DeleteProcThreadAttributeList(value); } } cleanup{attrs};
        check(::UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &handle_, sizeof(handle_), nullptr, nullptr),
              "cannot restrict GPU worker handle inheritance");
        // Install the kill-on-close job during creation. A parent exit between
        // CreateProcess and a separate assignment must not orphan a child.
#ifdef PROC_THREAD_ATTRIBUTE_JOB_LIST
        constexpr auto job_attribute = PROC_THREAD_ATTRIBUTE_JOB_LIST;
#else
        constexpr auto job_attribute = ProcThreadAttributeValue(13, FALSE, TRUE, FALSE);
#endif
        check(::UpdateProcThreadAttribute(attrs, 0, job_attribute, &job.value, sizeof(job.value), nullptr, nullptr),
              "cannot attach GPU worker supervisor job");
        STARTUPINFOEXW startup{}; startup.StartupInfo.cb = sizeof(startup); startup.lpAttributeList = attrs;
        std::wstring command = profile_detail::windows_argument(exe.wstring()) + L" compute worker --config NUL --gpu-recovery-worker " +
            std::to_wstring(reinterpret_cast<uintptr_t>(handle_)) + L" " + std::to_wstring(bytes_);
        struct Child {
            PROCESS_INFORMATION value{}; bool finished = false;
            ~Child() { if (value.hProcess) { if (!finished) { ::TerminateProcess(value.hProcess, 1); ::WaitForSingleObject(value.hProcess, INFINITE); }
                    ::CloseHandle(value.hThread); ::CloseHandle(value.hProcess); } }
        } child;
        check(::CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo,
            &child.value), "cannot start isolated GPU worker");
        DWORD state = WAIT_TIMEOUT;
        while ((state = ::WaitForSingleObject(child.value.hProcess, 20)) == WAIT_TIMEOUT) check_cancel();
        child.finished = state == WAIT_OBJECT_0;
        DWORD code = 1;
        check(state == WAIT_OBJECT_0 && ::GetExitCodeProcess(child.value.hProcess, &code), "cannot read GPU worker exit status");
        if (code) throw DeviceLost("GPU executor exited with code " + std::to_string(code));
#elif defined(__linux__)
        posix_spawn_file_actions_t actions;
        check(!::posix_spawn_file_actions_init(&actions), "cannot initialize GPU worker descriptors");
        struct Actions { posix_spawn_file_actions_t* value; ~Actions() { ::posix_spawn_file_actions_destroy(value); } } cleanup{&actions};
        check(!::posix_spawn_file_actions_adddup2(&actions, fd_, STDIN_FILENO), "cannot attach GPU worker mapping");
        std::vector<std::string> args{exe.string(), "compute", "worker", "--config", "/dev/null",
                                      "--gpu-recovery-worker", "0", std::to_string(bytes_)};
        std::vector<char*> argv; for (auto& a : args) argv.push_back(a.data()); argv.push_back(nullptr);
        struct Child {
            pid_t pid = 0;
            ~Child() { if (pid > 0) { ::kill(pid, SIGKILL); while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {} } }
        } child;
        pid_t pid = 0;
        check(!::posix_spawn(&pid, argv.front(), &actions, nullptr, argv.data(), environ),
              "cannot start isolated GPU worker");
        child.pid = pid;
        int status = 0;
        for (;;) {
            auto const ret = ::waitpid(child.pid, &status, WNOHANG);
            if (ret == child.pid) { child.pid = 0; break; }
            if (ret < 0 && errno != EINTR) { if (errno == ECHILD) child.pid = 0; throw std::runtime_error("cannot reap GPU worker"); }
            check_cancel(); std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (WIFSIGNALED(status)) throw DeviceLost("GPU executor terminated by signal " + std::to_string(WTERMSIG(status)));
        check(WIFEXITED(status), "cannot read GPU worker exit status");
        if (auto code = WEXITSTATUS(status)) throw DeviceLost("GPU executor exited with code " + std::to_string(code));
#else
        throw std::runtime_error("isolated GPU recovery requires Windows or Linux");
#endif
        auto const& result = request();
        check(result.format == Request::identity && result.mapping_bytes == bytes_ && result.error[sizeof(result.error)-1] == 0,
              "invalid response from isolated GPU worker");
        if (result.lost) throw DeviceLost(result.error);
        if (result.unavailable)
        {
            auto const native = result.probe == canary ? result.profiles.front().owning : 0;
            throw ResourcesUnavailable(result.error, native && native <= UINT64_MAX - output().size_bytes() ?
                native + output().size_bytes() : 0);
        }
        if (result.error[0]) throw std::runtime_error(result.error);
        check(result.complete == 1 && (result.probe == budget_only ? result.count == 0 : result.count && result.count <= output().size()), "incomplete isolated GPU generation");
    }
};
}
