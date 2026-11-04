#include "host/GroupFarmRetirement.hpp"
#include <iostream>
#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace group_grow;
using namespace group_grow::farm;
namespace fs = std::filesystem;
namespace
{
std::string const job(64, 'd');
template<class F> void rejects(F operation, char const* message)
{ bool failed = false; try { operation(); } catch (std::exception const&) { failed = true; } require(failed, message); }
void write(fs::path const& path, std::string const& bytes)
{ std::ofstream out(path, std::ios::binary); out << bytes; require(bool(out), "cannot write retirement fixture"); }
std::string read(fs::path const& path)
{ std::ifstream in(path, std::ios::binary); return {std::istreambuf_iterator<char>(in), {}}; }
fs::path source(FarmInventory const& inventory)
{ return file_io::native_path(inventory.roots[0].path) / file_io::native_path(inventory.plots[0].relative_path); }
fs::path claim(FarmInventory const& inventory)
{ return file_io::native_path(inventory.roots[0].path) / replot_claim_path(job, 0); }
InventoryFileIdentity take(FarmInventory const& inventory, bool recovering = false)
{ return claim_replot_source(inventory.roots[0], inventory.plots[0], job, 0, {}, recovering); }
void remove(FarmInventory const& inventory, std::function<void()> const& gate = [] {})
{ unlink_replot_claim(inventory.roots[0], inventory.plots[0], job, 0, {}, gate); }
bool missing(FarmInventory const& inventory)
{ return replot_claim_missing(inventory.roots[0], inventory.plots[0], job, 0); }
FarmInventory fixture(fs::path const& root, std::string const& name)
{
    auto const path = root / name; fs::create_directories(path / "nested");
    write(path / "nested" / "source.plot", "reviewed source bytes");
    std::array<fs::path, 1> roots{path}; auto result = scan_farm_inventory(roots);
    require(result.plots.size() == 1, "unexpected retirement inventory"); return result;
}
void protect(fs::path const& path)
{
#ifdef _WIN32
    pos2gpu::create_private_directory(path);
#else
    require(!::mkdir(path.c_str(), 0700), "cannot protect retirement fixture");
#endif
}
void protocol(fs::path const& root)
{
    auto normal = fixture(root, "normal");
    require(same_inventory_contents(take(normal), normal.plots[0].identity) && !fs::exists(source(normal)) &&
        read(claim(normal)) == "reviewed source bytes", "claim lost the reviewed native object");
    require(same_inventory_contents(take(normal), normal.plots[0].identity), "claim could not recover after reopen");
    rejects([&] { remove(normal, [] { throw std::runtime_error("not ready"); }); }, "unready target authorized retirement");
    require(read(claim(normal)) == "reviewed source bytes", "unready target lost the source claim");
    bool checked = false; remove(normal, [&] { checked = true; });
    require(checked && missing(normal), "retirement missed its live gate or native deletion");

    auto collision = fixture(root, "collision"); protect(claim(collision).parent_path()); write(claim(collision), "unrelated claim");
    rejects([&] { (void)take(collision); }, "claim collision replaced a file");
    require(read(source(collision)) == "reviewed source bytes" && read(claim(collision)) == "unrelated claim", "collision changed either object");

    auto changed = fixture(root, "changed"); fs::rename(source(changed), source(changed).parent_path() / "saved");
    write(source(changed), "reviewed source bytes");
    rejects([&] { (void)take(changed); }, "same-byte substitute was claimed");
    require(!fs::exists(claim(changed)) && fs::exists(source(changed).parent_path() / "saved"), "substitution damaged the reviewed object");

    auto alias = fixture(root, "alias"); fs::create_hard_link(source(alias), source(alias).parent_path() / "alias");
    rejects([&] { (void)take(alias); }, "new hard link supplied independent retirement credit");
    require(fs::exists(source(alias)), "hard-link refusal removed the source");

    auto unprotected = fixture(root, "unprotected"); fs::create_directory(claim(unprotected).parent_path());
    rejects([&] { (void)take(unprotected); }, "unprotected claim directory was accepted");
    require(fs::exists(source(unprotected)), "unprotected directory refusal removed the source");

    auto recovery = fixture(root, "rollback"); (void)take(recovery); fs::rename(claim(recovery), source(recovery));
    require(same_inventory_contents(take(recovery, true), recovery.plots[0].identity), "claimed revision could not reconcile an exact rolled-back rename");
    remove(recovery); require(missing(recovery), "reconciled retirement left its source behind");

    auto root_change = fixture(root, "root-change"); auto const old = file_io::native_path(root_change.roots[0].path);
    fs::rename(old, root / "saved-root"); fs::create_directories(old / "nested"); write(source(root_change), "unrelated root");
    rejects([&] { (void)take(root_change); }, "replaced root authorized retirement");
    require(read(source(root_change)) == "unrelated root" && read(root / "saved-root" / "nested" / "source.plot") == "reviewed source bytes",
        "replaced root damaged either object");

#ifdef _WIN32
    auto locked = fixture(root, "reader-lock");
    {
        auto reader = inventory_detail::windows_open(source(locked), false);
        rejects([&] { (void)take(locked); }, "reader sharing lock was ignored during retirement");
        require(fs::exists(source(locked)) && !fs::exists(claim(locked)), "reader lock refusal changed the source");
    }
    (void)take(locked); remove(locked); require(missing(locked), "unloaded reader lock could not resume retirement");
#endif

    auto guarded = fixture(root, "guarded"); (void)take(guarded);
#ifdef _WIN32
    remove(guarded, [&]
    {
        std::error_code error; fs::rename(claim(guarded), claim(guarded).parent_path() / "moved", error);
        require(bool(error), "native retirement handle allowed claim substitution");
        HANDLE writer = ::CreateFileW(claim(guarded).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        require(writer == INVALID_HANDLE_VALUE && ::GetLastError() == ERROR_SHARING_VIOLATION, "native retirement handle permitted a competing writer");
        fs::rename(claim(guarded).parent_path(), root / "moved-claim-directory", error);
        require(bool(error), "held claim directory allowed namespace substitution");
    });
    require(missing(guarded), "guarded exact-handle deletion left its claim");
#else
    rejects([&] { remove(guarded, [&]
    {
        fs::rename(claim(guarded), claim(guarded).parent_path() / "saved"); write(claim(guarded), "substitute");
    }); }, "substituted claim was unlinked");
    require(read(claim(guarded)) == "substitute" && read(claim(guarded).parent_path() / "saved") == "reviewed source bytes",
        "claim substitution damaged either object");
#endif
}
#ifdef _WIN32
std::wstring quote(std::wstring const& value)
{
    std::wstring result = L"\""; size_t slashes = 0;
    for (auto c : value)
    {
        if (c == L'\\') { ++slashes; continue; }
        if (c == L'\"') result.append(slashes * 2 + 1, L'\\'); else result.append(slashes, L'\\');
        slashes = 0; result += c;
    }
    result.append(slashes * 2, L'\\'); return result + L'\"';
}
#endif
void interrupted(fs::path const& root, fs::path const& self)
{
    for (auto const* point : {"replot-source-renamed", "replot-before-unlink", "replot-source-unlinked"})
    {
        auto inventory = fixture(root, point); auto const catalog = root / (std::string(point) + ".json");
        write(catalog, farm_inventory_text(inventory));
#ifdef _WIN32
        std::wstring command = quote(self.wstring()) + L" --interrupt " + quote(catalog.wstring()) + L" " + quote(file_io::native_path(point).wstring());
        STARTUPINFOW startup{}; startup.cb = sizeof(startup); PROCESS_INFORMATION process{};
        require(::CreateProcessW(self.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process), "cannot start native retirement crash fixture");
        ::CloseHandle(process.hThread); DWORD const waited = ::WaitForSingleObject(process.hProcess, 30000); DWORD status = 0;
        if (waited != WAIT_OBJECT_0) { ::TerminateProcess(process.hProcess, 1); ::WaitForSingleObject(process.hProcess, 30000); }
        bool const observed = ::GetExitCodeProcess(process.hProcess, &status); ::CloseHandle(process.hProcess);
        require(waited == WAIT_OBJECT_0 && observed && status == 86, "native retirement missed its durability boundary");
#else
        auto const child = ::fork(); require(child >= 0, "cannot fork retirement crash fixture");
        if (!child)
        { ::setenv("XCHPLOT2_GROW_FAILPOINT", point, 1); (void)take(inventory); remove(inventory); std::_Exit(1); }
        int status = 0; require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
            "retirement missed its durability boundary");
#endif
        bool const removed = std::string_view(point) == "replot-source-unlinked";
        require(!fs::exists(source(inventory)) && missing(inventory) == removed, "crash recovery confused claimed and deleted objects");
        if (!removed) { (void)take(inventory); remove(inventory); }
        require(missing(inventory), "recovered retirement retained its source");
    }
}
}
int main(int argc, char** argv)
{
    if (argc == 4 && std::string_view(argv[1]) == "--interrupt")
    {
        std::ifstream in(file_io::native_path(argv[2]), std::ios::binary); auto inventory = read_farm_inventory(in);
#ifdef _WIN32
        require(::SetEnvironmentVariableW(L"XCHPLOT2_GROW_FAILPOINT", file_io::native_path(argv[3]).c_str()), "cannot inject native retirement interruption");
#else
        ::setenv("XCHPLOT2_GROW_FAILPOINT", argv[3], 1);
#endif
        (void)take(inventory); remove(inventory); return 1;
    }
    auto const root = fs::temp_directory_path() / ("xchplot2-retirement-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try { fs::create_directory(root); protocol(root); interrupted(root, fs::absolute(file_io::native_path(argv[0]))); fs::remove_all(root); std::cout << "native retirement tests passed\n"; return 0; }
    catch (std::exception const& error) { fs::remove_all(root); std::cerr << error.what() << '\n'; return 1; }
}
