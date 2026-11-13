#include <stdexcept>
#include <string>
// GroupEngine imports this helper before GroupPlotter includes the profile.
// Keep that actual inclusion-order collision visible in this focused target.
namespace group_tools { inline void require(bool condition, std::string const& message) { if (!condition) throw std::runtime_error(message); } }
namespace group_grow { using group_tools::require; }
#include "host/GroupHarvesterProfile.hpp"
#include <iostream>
#ifdef _WIN32
#include <shellapi.h>
#endif

namespace fs = std::filesystem;
using namespace group_grow;
void check(bool condition, char const* message)
{ if (!condition) throw std::runtime_error(message); }
template<class Function> void rejects(Function function, char const* message)
{
    try { function(); }
    catch (std::runtime_error const&) { return; }
    throw std::runtime_error(message);
}
FarmingProfile profile(std::string const& text)
{ std::istringstream in(text); return read_farming_profile(in); }
profile_detail::ChiaConfig config(std::string const& text)
{ std::istringstream in(text); return profile_detail::read_chia_config(in); }
void write(fs::path const& path, std::string const& text)
{ std::ofstream file(path, std::ios::binary); file << text; check(bool(file), "cannot write test fixture"); }
void memory_bound_tests()
{
#ifdef __linux__
    auto const root = fs::temp_directory_path() / ("xchplot2-memory-bound-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root / "child" / "leaf");
    struct Remove { fs::path path; ~Remove() { std::error_code error; fs::remove_all(path, error); } } remove{root};
    write(root / "memory.max", "1024"); write(root / "memory.current", "256");
    write(root / "child" / "memory.max", "768"); write(root / "child" / "memory.current", "512");
    profile_detail::HostMemoryBound bound{4096, 4096};
    profile_detail::tighten_memory_bound(bound, root / "child" / "leaf", root);
    check(bound.capacity == 768 && bound.available == 768 - 512, "ancestor cgroup limits did not bound effective free memory");
    write(root / "child" / "memory.max", "max");
    bound = {4096, 4096}; profile_detail::tighten_memory_bound(bound, root / "child" / "leaf", root);
    check(bound.capacity == 1024 && bound.available == 1024 - 256, "unlimited child hid an inherited cgroup limit");
    fs::remove(root / "memory.current");
    bound = {4096, 4096}; profile_detail::tighten_memory_bound(bound, root / "child", root);
    check(bound.capacity == 1024 && !bound.available, "unobservable cgroup usage granted host headroom");
    write(root / "memory.limit_in_bytes", "512"); write(root / "memory.usage_in_bytes", "128");
    bound = {4096, 4096}; profile_detail::tighten_memory_bound(bound, root / "child", root, false);
    check(bound.capacity == 512 && bound.available == 512 - 128, "legacy memory-controller limit was ignored");
    check(profile_detail::cgroup_mount_path("/memory\\040limit\\134name") == "/memory limit\\name", "escaped cgroup mount path changed identity");
#endif
}
fs::path utf8_path(std::string const& value)
{ return fs::path(reinterpret_cast<char8_t const*>(value.c_str())); }
int fake_chia(int argc, char** argv)
{
    if (argc != 6 || std::string_view(argv[3]) != "rpc" || std::string_view(argv[4]) != "harvester") return 30;
#ifdef _WIN32
    int wide_count = 0; auto wide = ::CommandLineToArgvW(::GetCommandLineW(), &wide_count);
    if (!wide || wide_count != argc) { if (wide) ::LocalFree(wide); return 34; }
    fs::path root(wide[2]); ::LocalFree(wide);
#else
    fs::path root = utf8_path(argv[2]);
#endif
    if (fs::exists(root / "slow")) std::this_thread::sleep_for(std::chrono::seconds(5));
    if (fs::exists(root / "fail")) return 31;
    if (std::string_view(argv[5]) == "get_version") std::cout << "{\"success\":true,\"version\":\"fixture-development\"}";
    else if (std::string_view(argv[5]) == "get_plots")
    {
        std::ifstream input(root / "rpc.json", std::ios::binary); if (!input) return 32;
        std::cout << input.rdbuf();
    }
    else return 33;
    return 0;
}
void json_tests()
{
    std::string const prefix = "{\"schema_version\":1,\"model\":\"" + std::string(farming_model_identity) + '"';
    auto partial = profile(prefix + ",\"lookup_workers\":12,\"source\":\"explicit\"}");
    check(partial.config.lookup_workers == 12 && partial.config.memory_bytes == FarmingConfig{}.memory_bytes,
        "hand-edited profile did not preserve omitted assumptions");
    partial.config.reader = FarmingReader::cached; partial.config.index_cache_bytes = 4096;
    partial.config.cpu_work_weight = 0.125L; partial.config.io_work_weight = 2.5L;
    partial.config.filter_bits = 7; partial.config.future_filter_bits = 6;
    partial.config.group_members_override = 100; partial.config.fixed_meta = true; partial.config.meta_group_override = 9;
    partial.config.k = 18; partial.config.strength = 3;
    partial.captured_at = "2026-11-03T19:00:00Z"; partial.chia_version = "development";
    partial.host_name = "final-harvester";
    partial.assumptions.push_back("Unicode \xf0\x9f\x8c\xb1 and \"quotes\"\n");
    std::ostringstream out; write_farming_profile_json(out, partial);
    auto copied = profile("{\"farming_plan\":true,\"farming_profile\":" + out.str() + ",\"destinations\":[]}");
    auto expected = partial.config; expected.k = FarmingConfig{}.k; expected.strength = FarmingConfig{}.strength;
    expected.group_members_override = 0; expected.fixed_meta = false; expected.meta_group_override = 0;
    check(copied.config == expected && copied.assumptions == partial.assumptions && copied.captured_at == partial.captured_at && copied.host_name == partial.host_name,
        "portable profile lost harvester inputs or imported plot assignments");
    check(out.str().find("\"k\"") == out.str().npos && out.str().find("group_members") == out.str().npos,
        "portable profile contains plot-specific inputs");
    auto native = partial; native.host_name = "native-byte-\xff";
    native.assumptions.push_back("truncated-\xe2\x82 surrogate-\xed\xa0\x80");
    std::ostringstream escaped; write_farming_profile_json(escaped, native);
    auto imported = profile(escaped.str());
    check(imported.host_name == "native-byte-\xc3\xbf" &&
        imported.assumptions[0] == partial.assumptions[0] &&
        imported.assumptions.back() == "truncated-\xc3\xa2\xc2\x82 surrogate-\xc3\xad\xc2\xa0\xc2\x80" &&
        escaped.str().find("native-byte-\\u00ff") != std::string::npos,
        "native text bytes did not produce a valid, importable JSON profile");
    for (auto const& bad : std::vector<std::string>{
        "", "{", prefix, prefix + ",}", prefix + ",\"lookup_workers\":1,\"lookup_workers\":2}",
        prefix + ",\"lookup_workers\":01}", prefix + ",\"lookup_workers\":-1}",
        prefix + ",\"lookup_workers\":1.5}", prefix + ",\"lookup_workers\":18446744073709551616}",
        prefix + ",\"lookup_workers\":0}", prefix + ",\"harvester_cores\":1025}",
        prefix + ",\"reader\":\"future-reader\"}", prefix + ",\"memory_bytes\":0}",
        prefix + ",\"deadline_seconds\":1e9999}", prefix + ",\"deadline_seconds\":-2}",
        prefix + ",\"source\":\"\\ud800\"}", prefix + ",\"source\":\"\\udc00\"}",
        prefix + ",\"source\":\"\xc0\xaf\"}", prefix + ",\"source\":\"\xed\xa0\x80\"}",
        prefix + ",\"unknown\":{\"duplicate\":1,\"duplicate\":2}}", prefix + ",\"inventory_available\":true}", prefix + "} trailing",
        "{\"schema_version\":2,\"model\":\"group-farming-3\"}",
        "{\"schema_version\":1,\"model\":\"other-model\"}", "{\"farming_profile\":null}"})
        rejects([&] { profile(bad); }, "malformed profile was accepted");
    std::string nested = "0"; for (unsigned i = 0; i < 40; ++i) nested = '[' + nested + ']';
    rejects([&] { profile(prefix + ",\"unknown\":" + nested + '}'); }, "unbounded JSON nesting accepted");
    rejects([&] { profile(std::string(profile_detail::profile_bytes + 1, ' ')); }, "oversized profile was accepted");
    auto unicode = profile(prefix + ",\"source\":\"\\ud83c\\udf31\"}");
    check(unicode.source == "\xf0\x9f\x8c\xb1", "valid Unicode surrogate pair was not decoded");
    FarmingProfile retained; auto const initial_memory = farming_profile_memory(retained);
    retained.host_name.reserve(4096); retained.assumptions.reserve(64); retained.assumptions.emplace_back(8192, 'a');
    check(farming_profile_memory(retained) >= sizeof(FarmingProfile) + retained.host_name.capacity() +
        retained.assumptions.capacity() * sizeof(std::string) + retained.assumptions.back().capacity() &&
        farming_profile_memory(retained) > initial_memory, "retained profile string/vector capacity was not accounted");
    rejects([&] { std::istringstream input(out.str()); read_farming_profile(input, 16); }, "profile parsing memory budget was ignored");
    rejects([&] { profile_detail::JsonReader("[0,0,0,0]", 1024).read(); }, "JSON structure allocation budget was ignored");
}
void calibration_scope_tests()
{
    FarmingProfile scoped; scoped.config.k = 18; scoped.config.strength = 3; scoped.config.reader = FarmingReader::cached;
    scoped.reader_calibration = ReaderCalibrationScope{18, 3, FarmingReader::cached, reader_calibration_method,
        std::nextafter(2.0L, 3.0L)};
    scoped.source = "local-detection"; scoped.host_name = "imported-harvester"; scoped.captured_at = "2025-01-02T03:04:05Z";
    scoped.config_status = "original configuration"; scoped.inventory_status = "original inventory";
    scoped.inventory_available = true; scoped.inventory_v2_files = 3; scoped.inventory_v2_members = 12;
    scoped.detected_logical_cpus = 8; scoped.detected_available_memory_bytes = 8192;
    scoped.reader_calibration->host_name = "calibration-machine";
    scoped.reader_calibration->captured_at = "2026-11-03T20:00:00Z";
    std::ostringstream out; write_farming_profile_json(out, scoped);
    auto const copied = profile("{\"reader_calibration\":true,\"farming_profile\":" + out.str() + '}');
    check(out.str().find("\"schema_version\":2") != std::string::npos && copied.reader_calibration == scoped.reader_calibration &&
        copied.config.k == 18 && copied.config.strength == 3 && copied.config.reader == FarmingReader::cached,
        "scoped reader calibration lost parameters, method or full-precision margin");
    check(static_cast<FarmingProfileMetadata const&>(copied) == static_cast<FarmingProfileMetadata const&>(scoped),
        "reader calibration relabeled an imported host/configuration/inventory snapshot");
    validate_reader_calibration_scope(copied, copied.config);
    auto changed = scoped.config; changed.k = 28;
    rejects([&] { validate_reader_calibration_scope(copied, changed); }, "calibration accepted another k");
    changed = scoped.config; changed.strength = 4;
    rejects([&] { validate_reader_calibration_scope(copied, changed); }, "calibration accepted another strength");
    changed = scoped.config; changed.reader = FarmingReader::stateless;
    rejects([&] { validate_reader_calibration_scope(copied, changed); }, "calibration accepted another reader");
    auto replace = [&](std::string const& from, std::string const& to)
    {
        auto text = out.str(); auto const at = text.find(from); check(at != text.npos, "missing scoped profile fixture field");
        text.replace(at, from.size(), to); return text;
    };
    for (auto const& bad : {replace("\"schema_version\":2", "\"schema_version\":1"),
             replace("\"k\":18", "\"k\":256"), replace("\"k\":18", "\"k\":19"),
             replace("\"strength\":3", "\"strength\":63"),
             replace(reader_calibration_method, "unknown-reader-method"),
             replace("\"margin\":2.", "\"margin\":0.")})
        rejects([&] { profile(bad); }, "unsupported or inconsistent calibration scope was accepted");
    auto mismatch = scoped; mismatch.config.k = 28;
    rejects([&] { std::ostringstream text; write_farming_profile_json(text, mismatch); },
        "calibrated profile was re-exported with mismatched target parameters");
}
void config_tests()
{
    auto observed = config("self_hostname: &self_hostname \"localhost\"\r\nharvester:\r\n  farmer_peers:\r\n  - host: *self_hostname\r\n    port: 8447\r\n  num_threads: 30 # shared v1/v2 executor\r\n  logging: *logging\r\nfarmer:\r\n  num_threads: 999\r\n");
    check(observed.workers == 30 && observed.local_rpc, "conventional Chia config was not detected");
    check(!config("self_hostname: remote.example\nharvester:\n  num_threads: 30\n").local_rpc,
        "remote RPC endpoint was accepted for local detection");
    for (auto const& unsupported : std::vector<std::string>{
        "harvester: {num_threads: 30}\n", "harvester: *harvester\n", "harvester:\n  num_threads: *threads\n",
        "harvester:\n  num_threads: 30\n  num_threads: 20\n", "harvester:\n  num_threads: 30\nharvester:\n  num_threads: 20\n",
        "harvester:\n  num_threads: 30\n  <<: *defaults\n", "harvester:\n  num_threads: 30\n\"harvester\": {num_threads: 20}\n",
        "self_hostname: localhost\n\"self_hostname\": remote.example\nharvester:\n  num_threads: 30\n",
        "harvester:\n  \"num_threads\": 30\n", "harvester:\n\tnum_threads: 30\n",
        "harvester:\n  num_threads: 0\n", "harvester:\n  num_threads: 1025\n", "harvester:\n  num_threads: 30\n---\n",
        "harvester:\n  num_threads: 30.0\n", "harvester:\n  num_threads: \"30\"\n", "harvester:\n  plot_directories: []\n"})
    {
        auto value = config(unsupported);
        check(!value.workers && value.status.starts_with("unsupported:"), "unsupported Chia config was treated as observed");
    }
}
void process_tests(fs::path const& executable)
{
    auto root = fs::temp_directory_path() / utf8_path("xchplot2-profile-test space-\xc3\xa9-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root / "config");
    struct Cleanup { fs::path root; ~Cleanup() { std::error_code error; fs::remove_all(root, error); } } cleanup{root};
    auto const config_path = root / "config/config.yaml";
    std::string const local = "self_hostname: localhost\nharvester:\n  num_threads: 30\n";
    std::string const inventory = "{\"success\":true,\"plots\":[{\"size\":32},{\"size\":130,\"group_size\":25},{\"size\":131,\"group_size\":7}],\"failed_to_open_filenames\":[\"private/path\"],\"not_found_filenames\":[\"no-key\"]}";
    write(config_path, local); write(root / "rpc.json", inventory);
    ChiaDetectionOptions options; options.root = root; options.executable = executable;
    auto detected = detect_farming_profile(options);
    check(detected.inventory_available && detected.inventory_v1_files == 1 && detected.inventory_v2_files == 2 &&
        detected.inventory_v2_members == 32 && detected.inventory_failed_files == 1 && detected.inventory_no_key_files == 1,
        "actual subprocess RPC inventory was not counted");
    check(detected.config.lookup_workers == 30 && detected.config.reader == FarmingReader::stateless &&
        detected.config.memory_bytes == FarmingConfig{}.memory_bytes && !detected.captured_at.empty() &&
        detected.chia_version == "fixture-development" && !detected.host_name.empty(), "detection confused observed and assumed inputs");
    std::ostringstream exported; write_farming_profile_json(exported, detected);
    auto const root_encoded = root.generic_u8string(); std::string const root_text(root_encoded.begin(), root_encoded.end());
    check(exported.str().find("private/path") == exported.str().npos && exported.str().find("no-key") == exported.str().npos &&
        exported.str().find(root_text) == exported.str().npos, "private inventory details leaked into profile");
    options.lookup_workers = 3; detected = detect_farming_profile(options);
    check(detected.config.lookup_workers == 3 && detected.inventory_available, "explicit remaining worker budget did not override config");
    write(config_path, "self_hostname: remote.example\nharvester:\n  num_threads: 30\n");
    write(root / "fail", ""); detected = detect_farming_profile(options);
    check(!detected.inventory_available && detected.inventory_status.find("loopback") != std::string::npos,
        "local detection attempted a remote Chia RPC");
    fs::remove(root / "fail"); write(config_path, local);
    for (auto const& malformed : {"{", "{\"success\":false}", "{\"success\":true,\"plots\":[{\"size\":130}],\"failed_to_open_filenames\":[],\"not_found_filenames\":[]}",
        "{\"success\":true,\"plots\":[{\"size\":128,\"group_size\":1}],\"failed_to_open_filenames\":[],\"not_found_filenames\":[]}",
        "{\"success\":true,\"plots\":[{\"size\":0}],\"failed_to_open_filenames\":[],\"not_found_filenames\":[]}",
        "{\"success\":true,\"plots\":[],\"failed_to_open_filenames\":[123],\"not_found_filenames\":[]}",
        "{\"success\":true,\"plots\":[],\"plots\":[],\"failed_to_open_filenames\":[],\"not_found_filenames\":[]}"})
    { write(root / "rpc.json", malformed); check(!detect_farming_profile(options).inventory_available, "unsupported RPC inventory was accepted"); }
    write(root / "rpc.json", "{\"success\":true,\"plots\":[],\"failed_to_open_filenames\":[],\"not_found_filenames\":[]}");
    check(detect_farming_profile(options).inventory_available, "observed empty inventory was not distinguished from unavailable");
    options.max_output_bytes = 8;
    check(detect_farming_profile(options).inventory_status.find("byte limit") != std::string::npos, "subprocess output limit was not enforced");
    options.max_output_bytes = 1024; options.timeout = std::chrono::milliseconds(50); write(root / "slow", "");
    check(detect_farming_profile(options).inventory_status.find("timed out") != std::string::npos, "subprocess timeout was not enforced");
    fs::remove(root / "slow"); options.timeout = std::chrono::seconds(5); write(root / "fail", "");
    check(detect_farming_profile(options).inventory_status.find("failed") != std::string::npos, "subprocess failure was not retained");
    fs::remove(root / "fail"); options.executable = root / "missing-chia";
    check(detect_farming_profile(options).inventory_status.find("started") != std::string::npos, "missing Chia CLI was treated as empty inventory");
    fs::remove(config_path); detected = detect_farming_profile(options);
    check(!detected.inventory_available && detected.config_status.find("not found") != std::string::npos && detected.config.lookup_workers == 3,
        "missing Chia config did not retain explicit budget and unavailable inventory");
    write(config_path, local); options.executable = executable; options.max_memory_bytes = 16;
    check(!detect_farming_profile(options).inventory_available, "detection memory cap was ignored");
    check(profile_detail::windows_argument(L"") == L"\"\"" &&
        profile_detail::windows_argument(L"C:\\path with space\\") == L"\"C:\\path with space\\\\\"" &&
        profile_detail::windows_argument(L"a\\\"b") == L"\"a\\\\\\\"b\"", "Windows argv quoting does not preserve spaces, quotes and trailing slashes");
}
int main(int argc, char** argv) try
{
    if (argc > 1 && std::string_view(argv[1]) == "--root-path") return fake_chia(argc, argv);
#ifdef _WIN32
    std::wstring module(32768, L'\0'); auto const count = ::GetModuleFileNameW(nullptr, module.data(), DWORD(module.size()));
    check(count && count < module.size(), "cannot locate native Windows test executable"); module.resize(count); fs::path executable(module);
#else
    auto executable = fs::absolute(utf8_path(argv[0]));
#endif
    json_tests(); calibration_scope_tests(); config_tests(); memory_bound_tests(); process_tests(executable);
    std::cout << "Harvester profile parsing and local Chia detection passed\n"; return 0;
}
catch (std::exception const& error) { std::cerr << error.what() << '\n'; return 1; }
