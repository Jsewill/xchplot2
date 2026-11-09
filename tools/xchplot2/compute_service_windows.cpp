#include "group_commands.hpp"
#include "compute_output.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/WindowsFile.hpp"
#include <taskschd.h>
#include <chrono>
#include <iostream>
#include <thread>

namespace
{
namespace fs = std::filesystem;
namespace pool = group_grow::pool;
void require(bool value, std::string const& message)
{ if (!value) throw std::invalid_argument(message); }
void checked(HRESULT result, char const* operation)
{ if (FAILED(result)) throw std::system_error(static_cast<int>(result), std::system_category(), operation); }
template<class T> struct Com
{
    T* value = nullptr;
    ~Com() { if (value) value->Release(); }
    T* operator->() const { return value; }
    T** out() { return &value; }
    Com() = default;
    Com(Com const&) = delete;
    Com& operator=(Com const&) = delete;
};
struct Bstr
{
    BSTR value = nullptr;
    Bstr() = default;
    explicit Bstr(std::wstring const& text) : value(::SysAllocStringLen(text.data(), UINT(text.size())))
    { if (!value) throw std::bad_alloc(); }
    ~Bstr() { ::SysFreeString(value); }
    Bstr(Bstr const&) = delete;
    Bstr& operator=(Bstr const&) = delete;
};
struct ComApartment
{
    ComApartment() { checked(::CoInitializeEx(nullptr, COINIT_MULTITHREADED), "initialize Task Scheduler COM"); }
    ~ComApartment() { ::CoUninitialize(); }
};
std::wstring wide(std::string const& text)
{ return fs::path(reinterpret_cast<char8_t const*>(text.c_str())).native(); }
std::string utf8(std::wstring const& text)
{
    auto value = fs::path(text).u8string(); return {reinterpret_cast<char const*>(value.data()), value.size()};
}
std::wstring quoted(std::wstring const& argument)
{
    // Windows CommandLineToArgv/CRT quoting. Never pass arguments through cmd.exe.
    std::wstring result = L"\""; size_t slashes = 0;
    for (wchar_t c : argument)
    {
        require(c >= 32 && c != 127, "service paths cannot contain control characters");
        if (c == L'\\') { ++slashes; continue; }
        result.append(slashes * (c == L'"' ? 2 : 1), L'\\'); slashes = 0;
        if (c == L'"') result += L'\\';
        result += c;
    }
    result.append(slashes * 2, L'\\'); return result + L'"';
}
std::wstring xml(std::wstring_view text)
{
    std::wstring result;
    for (auto c : text)
    {
        require(c >= 32 || c == L'\n' || c == L'\r' || c == L'\t', "invalid task definition text");
        if (c == L'&') result += L"&amp;";
        else if (c == L'<') result += L"&lt;";
        else if (c == L'>') result += L"&gt;";
        else if (c == L'"') result += L"&quot;";
        else if (c == L'\'') result += L"&apos;";
        else result += c;
    }
    return result;
}
std::wstring executable()
{
    std::wstring result(32768, L'\0');
    auto size = ::GetModuleFileNameW(nullptr, result.data(), DWORD(result.size()));
    require(size && size < result.size(), "cannot locate the installed executable"); result.resize(size); return result;
}
void check_task(IRegisteredTask* task, std::wstring const& marker)
{
    Com<ITaskDefinition> definition; checked(task->get_Definition(definition.out()), "read installed task definition");
    Com<IRegistrationInfo> info; checked(definition->get_RegistrationInfo(info.out()), "read task registration");
    Bstr description; checked(info->get_Description(&description.value), "read task ownership marker");
    require(description.value && std::wstring_view(description.value, SysStringLen(description.value)) == marker,
        "refusing to modify a task not generated for this compute identity");
}
void stop(IRegisteredTask* task)
{
    TASK_STATE state{}; checked(task->get_State(&state), "inspect task state");
    if (state != TASK_STATE_RUNNING && state != TASK_STATE_QUEUED) return;
    checked(task->Stop(0), "stop compute task");
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
    do
    {
        checked(task->get_State(&state), "wait for compute task shutdown");
        if (state != TASK_STATE_RUNNING && state != TASK_STATE_QUEUED) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (std::chrono::steady_clock::now() < deadline);
    throw std::runtime_error("compute task has not stopped within 45 seconds");
}
void start(IRegisteredTask* task)
{
    VARIANT empty{}; Com<IRunningTask> running;
    checked(task->Run(empty, running.out()), "start compute task for the logged-in user");
}
} // namespace

int pos2gpu::run_compute_service(int argc, char** argv) try
{
    auto help = []
    {
        std::cout << "Usage: xchplot2 compute service COMMAND [OPTIONS]\n"
            "  install  Register a Task Scheduler task for this user and start it\n"
            "  start / stop / restart / status / remove  Manage that task\n"
            "  print    Print the task XML without registering or starting it\n"
            "  --role worker|broker  Task role, default worker\n"
            "  --directory DIR       Saved pool membership\n"
            << pos2gpu::compute_output_help <<
            "The task starts at login and runs as this user, without elevation.\n"
            "Resource and network defaults come from compute configure.\n"
            "For workers, run compute drain and wait for drained before stopping.\n"
            "After restarting a drained worker, run compute resume to accept new tasks.\n";
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
        { require(command.empty(), "unexpected compute service argument: " + option); command = option; continue; }
        require(i + 1 < argc, "missing value for " + option); std::string value = argv[++i];
        if (option == "--role") role = value;
        else if (option == "--directory") directory = fs::path(reinterpret_cast<char8_t const*>(value.c_str()));
        else throw std::invalid_argument("unknown compute service option: " + option);
    }
    output.validate();
    require(command == "install" || command == "start" || command == "stop" || command == "restart" ||
        command == "status" || command == "remove" || command == "print", "unknown compute service command: " + command + "; use compute service --help");
    require(role == "worker" || role == "broker", "--role requires worker or broker");
    auto identity = pool::load_pool(directory); require(bool(identity), "shared compute is not configured; use compute init or compute join");
    auto const name = wide("xchplot2-compute-" + role + '-' + identity->node_id);
    auto const marker = wide("xchplot2 compute service " + role + ' ' + identity->node_id);
    pos2gpu::PrivateFileSecurity security;
    auto const user = std::wstring(security.sid_text);
    // The same XML is printed as UTF-8 and registered through a UTF-16 BSTR.
    auto const definition = L"<?xml version=\"1.0\"?>\n"
        L"<Task version=\"1.3\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\n"
        L"<RegistrationInfo><Description>" + xml(marker) + L"</Description></RegistrationInfo>\n"
        L"<Triggers><LogonTrigger><Enabled>true</Enabled><UserId>" + xml(user) + L"</UserId></LogonTrigger></Triggers>\n"
        L"<Principals><Principal id=\"User\"><UserId>" + xml(user) + L"</UserId><LogonType>InteractiveToken</LogonType>"
        L"<RunLevel>LeastPrivilege</RunLevel></Principal></Principals>\n"
        L"<Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>"
        L"<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><AllowHardTerminate>true</AllowHardTerminate>"
        L"<StartWhenAvailable>true</StartWhenAvailable><Enabled>true</Enabled><ExecutionTimeLimit>PT0S</ExecutionTimeLimit>"
        L"<RestartOnFailure><Interval>PT1M</Interval><Count>255</Count></RestartOnFailure></Settings>\n"
        L"<Actions Context=\"User\"><Exec><Command>" + xml(executable()) + L"</Command><Arguments>" +
        xml(L"compute " + wide(role) + L" --directory " + quoted(fs::canonical(directory).native())) +
        L"</Arguments></Exec></Actions>\n</Task>\n";
    if (command == "print") { std::cout << utf8(definition); return 0; }
    ComApartment apartment;
    Com<ITaskService> service;
    checked(::CoCreateInstance(CLSID_TaskScheduler, nullptr, CLSCTX_INPROC_SERVER, IID_ITaskService,
        reinterpret_cast<void**>(service.out())), "connect to Task Scheduler");
    VARIANT empty{}; checked(service->Connect(empty, empty, empty, empty), "connect to local Task Scheduler");
    Com<ITaskFolder> folder; Bstr root(L"\\"); checked(service->GetFolder(root.value, folder.out()), "open task folder");
    Com<IRegisteredTask> task; Bstr task_name(name);
    auto const found = folder->GetTask(task_name.value, task.out());
    if (SUCCEEDED(found)) check_task(task.value, marker);
    else require(command == "install" && (found == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) || found == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND)),
        "compute task is not installed or cannot be inspected; run compute service install");
    if (command == "install")
    {
        Com<IRegisteredTask> registered; Bstr body(definition);
        VARIANT account{}; account.vt = VT_BSTR; Bstr account_name(user); account.bstrVal = account_name.value;
        checked(folder->RegisterTask(task_name.value, body.value, TASK_CREATE_OR_UPDATE, account, empty,
            TASK_LOGON_INTERACTIVE_TOKEN, empty, registered.out()), "register compute task");
        std::cout << "Installed Task Scheduler task " << utf8(name) << '\n'; start(registered.value); return 0;
    }
    if (command == "status")
    {
        TASK_STATE state{}; LONG result = 0;
        checked(task->get_State(&state), "read task state"); checked(task->get_LastTaskResult(&result), "read task result");
        char const* label = state == TASK_STATE_RUNNING ? "running" : state == TASK_STATE_QUEUED ? "queued" :
            state == TASK_STATE_READY ? "ready" : state == TASK_STATE_DISABLED ? "disabled" : "unknown";
        if (output.details()) std::cout << utf8(name);
        else std::cout << "Compute " << role << " service";
        std::cout << ": " << label;
        if (output.details() || result) std::cout << "; last result " << result;
        std::cout << '\n'; return 0;
    }
    if (command == "stop" || command == "restart" || command == "remove") stop(task.value);
    if (command == "remove")
    { checked(folder->DeleteTask(task_name.value, 0), "remove compute task"); std::cout << "Removed " << utf8(name) << '\n'; }
    if (command == "start" || command == "restart") start(task.value);
    return 0;
}
catch (std::exception const& error) { std::cerr << "compute service: " << error.what() << '\n'; return 1; }
