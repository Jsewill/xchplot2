#include "host/GroupCalibrationProcess.hpp"
#include "host/GroupFarmResources.hpp"
#include <iostream>
#ifdef _WIN32
#include <shellapi.h>
#else
#include <sys/mman.h>
#endif

namespace fs = std::filesystem;
namespace process = group_grow::calibration_process;
using process::check;
constexpr uint64_t allowance = uint64_t(256) << 20;

uint64_t baseline()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX value{};
    check(::K32GetProcessMemoryInfo(::GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&value), sizeof(value)),
        "cannot read fixture process commitment");
    return value.PrivateUsage;
#else
    std::ifstream input("/proc/self/statm"); uint64_t pages = 0;
    check(bool(input >> pages), "cannot read fixture address space");
    auto const page = ::sysconf(_SC_PAGESIZE); check(page > 0 && pages <= UINT64_MAX / uint64_t(page), "invalid fixture address space");
    return pages * uint64_t(page);
#endif
}
uint64_t budget()
{
    auto const used = baseline(); check(used <= UINT64_MAX - allowance, "fixture memory budget overflow");
    return used + allowance;
}
std::string text(fs::path const& path)
{ auto const value = path.u8string(); return {reinterpret_cast<char const*>(value.data()), value.size()}; }
fs::path native(std::string const& value)
{ return fs::path(reinterpret_cast<char8_t const*>(value.c_str())); }
template<class Function> void rejected(Function operation)
{
    try { operation(); } catch (std::exception const&) { return; }
    throw std::runtime_error("calibration failure fixture unexpectedly succeeded");
}
void scoped_cpu_fixture()
{
#ifdef _WIN32
    DWORD_PTR original = 0, system = 0;
    check(::GetProcessAffinityMask(::GetCurrentProcess(), &original, &system) && original, "cannot save fixture CPU affinity");
    auto unchanged = [&]
    { DWORD_PTR actual = 0; check(::GetProcessAffinityMask(::GetCurrentProcess(), &actual, &system) && actual == original, "scoped CPU cap did not restore original affinity"); };
    {
        process::ScopedCpuCap cap(1);
        check(cap.lanes() == 1 && group_grow::profile_detail::logical_cpus() == 1, "scoped process CPU cap not enforced");
        uint64_t inherited = 0; std::thread worker([&] { inherited = group_grow::profile_detail::logical_cpus(); }); worker.join();
        check(inherited == 1, "ordinary generation worker did not inherit scoped process CPU affinity");
    }
    unchanged();
    rejected([] { process::ScopedCpuCap cap(1); throw std::runtime_error("scoped generation failure"); }); unchanged();
    { process::ScopedCpuCap cap(1); cap.restore(); unchanged(); }
    rejected([&] { process::ScopedCpuCap impossible(uint64_t(std::popcount(original)) + 1); }); unchanged();
#elif defined(__linux__)
    cpu_set_t original, restored; CPU_ZERO(&original); CPU_ZERO(&restored);
    check(!::sched_getaffinity(0, sizeof(original), &original) && CPU_COUNT(&original), "cannot save fixture CPU affinity");
    auto unchanged = [&]
    { CPU_ZERO(&restored); check(!::sched_getaffinity(0, sizeof(restored), &restored) && CPU_EQUAL(&restored, &original), "scoped CPU cap did not restore original affinity"); };
    {
        process::ScopedCpuCap cap(1);
        check(cap.lanes() == 1 && group_grow::profile_detail::logical_cpus() == 1, "scoped calling thread CPU cap not enforced");
        uint64_t inherited = 0; std::thread worker([&] { inherited = group_grow::profile_detail::logical_cpus(); }); worker.join();
        check(inherited == 1, "ordinary generation worker did not inherit scoped CPU affinity");
    }
    unchanged();
    rejected([] { process::ScopedCpuCap cap(1); throw std::runtime_error("scoped generation failure"); });
    unchanged();
    {
        process::ScopedCpuCap cap(1); cap.restore(); unchanged();
        check(cap.lanes() == 1, "restoration changed admitted CPU lanes");
    }
    unchanged();
    rejected([&] { process::ScopedCpuCap impossible(uint64_t(CPU_COUNT(&original)) + 1); });
    unchanged();
#else
    rejected([] { process::ScopedCpuCap unsupported(1); });
#endif
}
int direct(std::string const& mode)
{
#ifdef __linux__
    if (mode == "--direct-active-pressure")
    {
        process::ScopedCpuCap cap(1);
        using namespace group_grow::farm;
        auto const host = group_grow::profile_detail::host_name();
        Profile profile; profile.source = "actual-process-pressure-fixture"; profile.producer = "test"; profile.native.host = host;
        Resource cpu; cpu.id = "cpu"; cpu.host = host; cpu.capacity = 1; cpu.capacity_known = true;
        Resource ram; ram.id = "ram"; ram.host = host; ram.kind = ResourceKind::host_memory;
        ram.capacity = group_grow::profile_detail::available_memory(); ram.capacity_known = true; profile.resources = {cpu, ram};
        profile.gate.recovery_hold_ms = 0; profile.gate.recovery_samples = 1;
        check(profile.gate.max_cpu_busy_permille == 950, "actual pressure fixture disabled the default CPU guard");
        Stage stage; stage.name = "actual-active-construction"; stage.scope_complete = true;
        Demand compute; compute.resource = "cpu"; compute.required_units = 1; compute.work_us = 1; compute.period_us = compute.deadline_us = 1000000; compute.bounded = true;
        Demand memory; memory.resource = "ram"; memory.transient_units = uint64_t(32) << 20; memory.bounded = true;
        stage.construction = {compute, memory}; LocalPressureSampler sampler(host);
        (void)sampler.sample(profile); std::this_thread::sleep_for(std::chrono::milliseconds(80)); (void)sampler.sample(profile);
        std::vector<uint8_t> held(size_t(8) << 20, 0x5a);
        std::atomic<bool> stop{false};
        struct Work
        {
            std::atomic<bool>& stop; std::thread worker;
            explicit Work(std::atomic<bool>& stop_) : stop(stop_), worker([&] { while (!stop.load(std::memory_order_relaxed)) std::atomic_signal_fence(std::memory_order_seq_cst); }) {}
            ~Work() { stop.store(true); worker.join(); }
        };
        uint64_t own_busy = 0, raw_busy = 0;
        {
            Work work(stop); std::this_thread::sleep_for(std::chrono::milliseconds(300));
            auto observed = sampler.sample(profile, Clock::now(), 1); auto const anon = sampler.own_anon_bytes();
            auto cpu_reading = [&](Telemetry const& value) -> ResourceSample const&
            { auto at = std::find_if(value.resources.begin(), value.resources.end(), [](ResourceSample const& r) { return r.resource == "cpu"; });
              check(at != value.resources.end() && at->cpu_busy_known, "actual CPU pressure evidence unavailable"); return *at; };
            own_busy = cpu_reading(observed).cpu_busy_permille;
            check(anon && *anon >= held.size(), "active memory credit did not observe the held private anonymous pages");
            AdmissionGate active_gate(profile); check(active_gate.observe(observed), "active process observation rejected");
            check(active_gate.evaluate(stage, Clock::now(), anon).allowed, "default CPU guard cancelled its own admitted generation work");
            auto rapid = sampler.sample(profile, Clock::now(), 1);
            check(cpu_reading(rapid).cpu_busy_permille == own_busy && !rapid.recovery_observation,
                "rapid horizon check discarded a valid active interval or invented a recovery vote");
            auto inactive = sampler.sample(profile); raw_busy = cpu_reading(inactive).cpu_busy_permille;
            check(raw_busy > own_busy && inactive.epoch > observed.epoch && !sampler.own_anon_bytes(), "active discount leaked into inactive telemetry");
            if (raw_busy >= profile.gate.max_cpu_busy_permille)
            {
                check(active_gate.observe(inactive) && !active_gate.evaluate(stage).allowed,
                    "inactive admission ignored default-threshold CPU contention");
            }
        }
        pid_t child = ::fork(); check(child >= 0, "cannot start isolated foreign-load fixture");
        if (!child) { ::alarm(2); for (;;) std::atomic_signal_fence(std::memory_order_seq_cst); }
        struct Child { pid_t pid; ~Child() { ::kill(pid, SIGKILL); int status = 0; while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {} } } cleanup{child};
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        auto foreign = sampler.sample(profile, Clock::now(), 1); auto raw = sampler.sample(profile);
        auto cpu_busy = [](Telemetry const& value)
        { for (auto const& r : value.resources) if (r.resource == "cpu" && r.cpu_busy_known) return r.cpu_busy_permille;
          throw std::runtime_error("foreign-load CPU counters unavailable"); };
        check(cpu_busy(foreign) == cpu_busy(raw) && cpu_busy(foreign) > own_busy,
            "another process's CPU work was credited to active construction");
        std::cout << "actual default CPU pressure own=" << own_busy << " raw=" << raw_busy << " foreign=" << cpu_busy(foreign) << '\n';
        return 0;
    }
#endif
    if (mode == "--direct-cpu")
    {
#if defined(__linux__) || defined(_WIN32)
        check(process::install_cpu_cap(1) == 1 && group_grow::profile_detail::logical_cpus() == 1, "direct proof CPU cap not installed");
        uint64_t inherited = 0; std::thread worker([&] { inherited = group_grow::profile_detail::logical_cpus(); }); worker.join();
        check(inherited == 1, "solver thread did not inherit bounded proof CPU affinity");
        group_grow::farm::Profile profile; group_grow::farm::Resource cpu;
        cpu.id = "cpu"; cpu.host = group_grow::profile_detail::host_name(); cpu.capacity = 1; cpu.capacity_known = true; profile.resources = {cpu};
        group_grow::farm::LocalPressureSampler sampler(cpu.host); (void)sampler.sample(profile);
        std::this_thread::sleep_for(std::chrono::milliseconds(80)); auto observation = sampler.sample(profile);
        check(observation.resources.size() == 1 && observation.resources[0].cpu_busy_known,
            "local pressure source did not measure the enforced CPU affinity rows"); return 0;
#else
        rejected([] { process::install_cpu_cap(1); }); return 0;
#endif
    }
    auto bytes = budget();
#ifndef _WIN32
    if (mode == "--direct-cap")
    {
        // Relative to this fresh executable's measured baseline, so sanitizer
        // startup/shadow mappings are not mistaken for plot-driven allocation.
        auto const used = baseline(); struct rlimit inherited{};
        check(!::getrlimit(RLIMIT_AS, &inherited), "cannot read inherited fixture limit");
        rlim_t hard = rlim_t(used + allowance / 2), soft = rlim_t(used + allowance / 4);
        if (inherited.rlim_max != RLIM_INFINITY) hard = std::min(hard, inherited.rlim_max);
        if (inherited.rlim_cur != RLIM_INFINITY) soft = std::min(soft, inherited.rlim_cur);
        soft = std::min(soft, hard); check(uint64_t(soft) > used, "inherited address-space cap cannot fit fixture startup");
        struct rlimit limits{soft, hard}; check(!::setrlimit(RLIMIT_AS, &limits), "cannot install inherited fixture limit");
        auto const cap = process::install_memory_cap(bytes);
        struct rlimit actual{}; check(!::getrlimit(RLIMIT_AS, &actual), "cannot read back fixture limit");
        check(cap.enforced_bytes == uint64_t(soft) && actual.rlim_cur == soft && actual.rlim_max == soft,
            "calibration raised an inherited soft or hard address-space limit");
        errno = 0; auto mapping = ::mmap(nullptr, size_t(cap.enforced_bytes), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(mapping == MAP_FAILED && errno == ENOMEM, "allocation escaped the hard calibration address-space cap");
        return 0;
    }
#endif
    auto const cap = process::install_memory_cap(bytes);
    check(cap.enforced_bytes && cap.enforced_bytes <= bytes && cap.baseline_bytes <= cap.enforced_bytes,
        "direct worker did not install a verifiable memory cap");
    if (mode == "--direct-cap")
    {
#ifdef _WIN32
        auto memory = ::VirtualAlloc(nullptr, SIZE_T(cap.enforced_bytes), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        check(!memory, "allocation escaped the calibration commitment cap");
#endif
        return 0;
    }
    process::Deadline watchdog(40, mode == "--direct-cancel" ? std::function<bool()>([] { return true; }) : std::function<bool()>{});
    std::this_thread::sleep_for(std::chrono::seconds(10));
    return 9;
}
int fake_worker(std::vector<std::string> args)
{
    bool const proof_worker = args.size() >= 18 && args[12] == "--replot-proof-worker";
    if (proof_worker) args.erase(args.begin() + 12);
    bool const solve_budget = proof_worker && args.size() >= 17 && args[12] == "--replot-proof-solves" && args[13] == "1";
    check(!proof_worker || solve_budget, "proof worker omitted its explicit solve budget");
    if (solve_budget) args.erase(args.begin() + 12, args.begin() + 14);
    bool const cpu_cap = args.size() == 17 && args[12] == "--replot-proof-cpus" && args[13] == "1";
    if (cpu_cap) args.erase(args.begin() + 12, args.begin() + 14);
    check(args.size() == 15 && args[1] == "plan" && args[2] == "--config" &&
#ifdef _WIN32
        args[3] == "NUL" &&
#else
        args[3] == "/dev/null" &&
#endif
        args[4] == "--calibrate" && args[5] == "--calibration-worker" && args[6] == "--calibration-samples" &&
        args[7] == "1" && args[8] == "--calibration-memory-bytes" && args[10] == "--calibration-timeout-ms" &&
        args[12] == "--json" && args[13] == "--" && native(args[14]).is_absolute(), "private worker argv is not isolated from user defaults");
    auto const cap = process::install_memory_cap(std::stoull(args[9]));
    if (cpu_cap) check(process::install_cpu_cap(1) == 1, "hidden proof worker did not enforce CPU cap");
    check(cap.enforced_bytes, "fake hidden worker did not install its own cap");
    auto const mode = text(native(args[14]).filename());
    if (mode == "hang")
    {
        // Deliberately simulate a broken/unresponsive worker to exercise the
        // parent's timeout independently of the production hard watchdog.
        std::this_thread::sleep_for(std::chrono::seconds(10)); return 9;
    }
    process::Deadline watchdog(std::stoull(args[11]));
    if (mode == "flood") { for (unsigned i = 0; i < 65536; ++i) std::cout.put('x'); std::cout.flush(); return 0; }
    if (mode == "failure") { std::cerr << "fixture failure\n"; return 7; }
    if (mode == "empty") return 0;
    std::cout << (cpu_cap ? "{\"proof_worker\":true,\"cpu_cap\":1}\n" : proof_worker ? "{\"proof_worker\":true}\n" : "{\"ok\":true}\n") << std::flush; return 0;
}
int launch_direct(std::string const& mode)
{
    auto const self = process::executable(); auto const until = process::Clock::now() + std::chrono::seconds(5);
#ifdef _WIN32
    auto command = group_grow::profile_detail::windows_argument(self.wstring()) + L" " + std::wstring(mode.begin(), mode.end());
    PROCESS_INFORMATION child{}; STARTUPINFOW startup{}; startup.cb = sizeof(startup);
    check(::CreateProcessW(self.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &child),
        "cannot launch direct calibration fixture");
    process::NativeHandle thread(child.hThread), handle(child.hProcess);
    while (::WaitForSingleObject(handle.value, 20) != WAIT_OBJECT_0)
        if (process::Clock::now() >= until)
        { ::TerminateProcess(handle.value, 9); ::WaitForSingleObject(handle.value, INFINITE); throw std::runtime_error("direct watchdog did not terminate worker"); }
    DWORD code = 0; check(::GetExitCodeProcess(handle.value, &code), "cannot read direct fixture exit"); return int(code);
#else
    std::string executable = self.string(), option = mode;
    std::array<char*, 3> argv{executable.data(), option.data(), nullptr}; pid_t child = 0;
    check(!::posix_spawn(&child, argv[0], nullptr, nullptr, argv.data(), environ), "cannot launch direct calibration fixture");
    int status = 0;
    for (;;)
    {
        auto const waited = ::waitpid(child, &status, WNOHANG);
        if (waited == child) break;
        if (process::Clock::now() >= until)
        { ::kill(child, SIGKILL); while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {} throw std::runtime_error("direct watchdog did not terminate worker"); }
        check(waited >= 0 || errno == EINTR, "cannot reap direct calibration fixture");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 9;
#endif
}
size_t handles()
{
#ifdef _WIN32
    DWORD count = 0; check(::GetProcessHandleCount(::GetCurrentProcess(), &count), "cannot count fixture handles"); return count;
#else
    size_t count = 0; for (auto const& entry : fs::directory_iterator("/proc/self/fd")) { (void)entry; ++count; } return count;
#endif
}
int main(int argc, char** argv) try
{
    std::vector<std::string> args;
#ifdef _WIN32
    int count = 0; auto wide = ::CommandLineToArgvW(::GetCommandLineW(), &count); check(wide, "cannot read Unicode fixture argv");
    for (int i = 0; i < count; ++i) args.push_back(text(fs::path(wide[i]))); ::LocalFree(wide);
#else
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
#endif
    if (args.size() > 1 && args[1].starts_with("--direct-")) return direct(args[1]);
    if (args.size() > 1 && args[1] == "plan") return fake_worker(args);
    scoped_cpu_fixture();
    process::Limits limits{budget(), 3000, 4096, {}};
    auto const input = native("calibration fixture-\xc3\xa9-\"quoted");
    check(process::run_worker(input, 1, limits) == "{\"ok\":true}\n", "successful calibration output changed");
    check(process::run_worker(input, 1, limits, process::Worker::migration_proof) == "{\"proof_worker\":true}\n",
        "bounded migration proof worker did not receive its isolated typed mode");
#if defined(__linux__) || defined(_WIN32)
    check(process::run_worker(input, 1, limits, process::Worker::migration_proof, 1) == "{\"proof_worker\":true,\"cpu_cap\":1}\n",
        "bounded migration proof worker did not enforce its CPU request");
#endif
    rejected([&] { (void)process::run_worker(input, 1, limits, process::Worker::migration_proof, 1, 0); });
    rejected([&] { (void)process::run_worker(input, 1, limits, process::Worker::migration_proof, 1, 2); });
    check(launch_direct("--direct-cpu") == 0, "direct proof CPU-affinity enforcement failed");
#ifdef __linux__
    check(launch_direct("--direct-active-pressure") == 0, "active/inactive default pressure enforcement failed");
#endif
    auto const before = handles();
    for (unsigned i = 0; i < 8; ++i)
        check(process::run_worker(input, 1, limits) == "{\"ok\":true}\n", "final worker output was lost at process exit");
    rejected([&] { (void)process::run_worker("failure", 1, limits); });
    rejected([&] { (void)process::run_worker("empty", 1, limits); });
    auto startup = limits; startup.memory_bytes = 1;
    rejected([&] { (void)process::run_worker(input, 1, startup); });
    auto bounded = limits; bounded.output_bytes = 32;
    rejected([&] { (void)process::run_worker("flood", 1, bounded); });
    bounded = limits; bounded.timeout_ms = 40;
    rejected([&] { (void)process::run_worker("hang", 1, bounded); });
    bounded = limits; unsigned calls = 0; bounded.cancelled = [&] { return ++calls > 3; };
    rejected([&] { (void)process::run_worker("hang", 1, bounded); });
    bounded = limits; calls = 0; bounded.cancelled = [&]() -> bool { if (++calls > 3) throw std::runtime_error("fixture callback"); return false; };
    rejected([&] { (void)process::run_worker("hang", 1, bounded); });
    bounded = limits; bounded.cancelled = [] { return true; };
    rejected([&] { (void)process::run_worker(input, 1, bounded); });
    check(launch_direct("--direct-cap") == 0, "direct hidden process memory cap failed");
    check(launch_direct("--direct-timeout") == 4 && launch_direct("--direct-cancel") == 4,
        "direct hidden invocation bypassed timeout/cancellation watchdog");
    check(handles() == before, "calibration process cleanup leaked handles or descriptors");
#ifndef _WIN32
    int status = 0; errno = 0;
    check(::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD, "calibration cleanup left an unreaped child");
#endif
    std::cout << "calibration caps, private argv, deadlines, cancellation, output and child cleanup passed\n";
    return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
