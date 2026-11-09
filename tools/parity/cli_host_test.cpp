// Exercise the real CLI/config/manifest code; plotting entry points are inert.
#undef NDEBUG
#include "gpu/SyclDeviceList.hpp"
#include "host/BatchPlotter.hpp"
#include "host/GroupPlotter.hpp"
#include "host/PlotFileWriterParallel.hpp"
#include "pos2_keygen.h"
#ifdef XCHPLOT2_GROUP_COMMANDS
#include "../xchplot2/group_commands.hpp"
#endif
#include <cassert>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#include <aclapi.h>
#endif

extern "C" int xchplot2_main(int, char**);
namespace {
size_t verified_trials = 0;
bool verified_full = false;
pos2gpu::BatchOptions last_options;
pos2gpu::GpuPlotOptions last_test_options;
int batch_calls = 0;
#ifdef XCHPLOT2_GROUP_COMMANDS
pos2gpu::GroupOptions last_group_options;
int group_calls = 0;
std::vector<std::filesystem::path> checked_group_directories;
std::vector<std::string> last_compute;
#endif
std::vector<pos2gpu::BatchEntry> last_entries;
std::size_t write_limit = std::numeric_limits<std::size_t>::max();
bool fail_remaining = false;
bool throw_remaining = false;
int keygen_calls = 0;
int cli(std::vector<std::string> args)
{
    args.insert(args.begin(), "xchplot2");
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return xchplot2_main(static_cast<int>(argv.size()), argv.data());
}
void put(std::filesystem::path const& path, std::string const& data)
{
    std::ofstream out(path, std::ios::binary); out << data; assert(out.good());
}
}
namespace pos2gpu {
std::array<uint8_t, 32> plot_id_for_group(
    std::array<uint8_t, 32> const& group, uint16_t index, uint8_t meta)
{
    auto id = group;
    id[29] ^= uint8_t(index >> 8);
    id[30] ^= uint8_t(index);
    id[31] ^= uint8_t(meta ^ 0xa5);
    return id;
}
std::vector<GpuDeviceInfo> list_gpu_devices() { return {}; }
std::string plot_to_file(GpuPlotOptions const& options, std::string const&)
{ last_test_options = options; return {}; }
std::vector<std::string> worker_labels(std::vector<int> const& d) { return std::vector<std::string>(d.size(), "test"); }
size_t batch_worker_count(BatchOptions const&, int) { return 1; }
BatchResult run_batch(std::vector<BatchEntry> const& entries, BatchOptions const& opts)
{
    ++batch_calls; last_options = opts; last_entries = entries;
    BatchResult result;
    WorkerTimeline worker;
    for (auto const& entry : entries) {
        auto const path = std::filesystem::path(entry.out_dir) / entry.out_name;
        if (opts.skip_existing && std::filesystem::exists(path)) {
            ++result.plots_skipped;
            if (opts.on_plot_ready) opts.on_plot_ready(entry);
            continue;
        }
        if (result.plots_written == write_limit) {
            if (throw_remaining) throw std::runtime_error("test write failure");
            if (fail_remaining) { ++result.plots_failed; continue; }
            break;
        }
        put(path, "stub plot");
        ++result.plots_written;
        if (opts.on_plot_ready) opts.on_plot_ready(entry);
        worker.completion_seconds.push_back(worker.completion_seconds.size() + 1);
    }
    result.bytes_written = 9 * result.plots_written;
    result.total_wall_seconds = entries.size(); result.workers.push_back(worker);
    return result;
}
#ifdef XCHPLOT2_GROUP_COMMANDS
void prepare_group_compute(GroupOptions&) {}
bool saved_compute_available(std::filesystem::path const&) { return false; }
std::array<uint8_t, 32> compute_owner_seed(std::array<uint8_t, 32> const& seed, std::string_view owner)
{ auto result = seed; for (size_t i = 0; i < owner.size(); ++i) result[i % result.size()] ^= uint8_t(owner[i]); return result; }
int run_group(GroupOptions const& options)
{ ++group_calls; last_group_options = options; return 0; }
void check_group_output_directory(std::filesystem::path const& directory)
{ checked_group_directories.push_back(directory); }
BatchResult run_group_batch(std::vector<BatchEntry> const& entries, BatchOptions const& opts, GroupOptions group)
{ ++group_calls; last_group_options = group; return run_batch(entries, opts); }
int run_compute_broker(int argc, char** argv)
{ last_compute.assign(argv, argv + argc); return 0; }
int run_compute_worker(int argc, char** argv)
{ last_compute.assign(argv, argv + argc); return 0; }
int run_compute_setup(int argc, char** argv)
{ last_compute.assign(argv, argv + argc); return 0; }
int run_compute_membership(int argc, char** argv)
{ last_compute.assign(argv, argv + argc); return 0; }
int run_compute_status(int argc, char** argv)
{ last_compute.assign(argv, argv + argc); return 0; }
int run_compute_service(int argc, char** argv)
{ last_compute.assign(argv, argv + argc); return 0; }
#endif
VerifyResult verify_plot_file(std::string const&, size_t trials, bool full)
{
    verified_trials = trials; verified_full = full;
    return {trials, 1, 1, full ? size_t{1} : size_t{0}};
}
}
extern "C" {
int pos2_keygen_decode_address(char const*, uint8_t*) { return POS2_BAD_ADDRESS; }
int pos2_keygen_derive_group(uint8_t const* seed, size_t, uint8_t const* farmer, uint8_t const* pool,
    int kind, uint8_t, uint8_t* id, uint8_t* memo, size_t* size)
{
    ++keygen_calls;
    std::copy_n(seed, 32, id);
    std::size_t const pool_size = kind == POS2_POOL_PK ? 48 : 32;
    std::copy_n(pool, pool_size, memo);
    std::copy_n(farmer, 48, memo + pool_size);
    std::copy_n(seed, 32, memo + pool_size + 48);
    *size = pool_size + 80;
    return POS2_OK;
}
int pos2_keygen_derive_subseed(uint8_t const* seed, uint64_t index, uint8_t* out)
{
    std::copy_n(seed, 32, out);
    for (int i = 0; i < 8; ++i) out[i] ^= static_cast<uint8_t>(index >> (8 * i));
    return POS2_OK;
}
}
int main(int argc, char** argv)
{
#ifdef _WIN32
    if (std::string(argv[0]) == "parity-test") {
        put(std::getenv("XCHPLOT2_TEST_MARKER"), "executed");
        return 17;
    }
#else
    (void)argc; (void)argv;
#endif
    std::filesystem::path dir;
    std::random_device random;
    do { dir = std::filesystem::temp_directory_path() / ("xchplot2 cli é-" + std::to_string(random())); }
    while (!std::filesystem::create_directory(dir));
    auto const config = dir / "config.toml", manifest = dir / "manifest.tsv";
    auto line = [&](std::string const& fields, std::string const& name = "plot.gplot") {
        std::ostringstream row;
        row << fields << ' ' << std::string(64, 'a') << " 00 "
            << std::quoted(dir.string()) << ' ' << std::quoted(name) << '\n';
        return row.str();
    };
    auto capture = [&](std::vector<std::string> const& command, std::string* errors = nullptr) {
        std::ostringstream output, error;
        auto* previous = std::cout.rdbuf(output.rdbuf());
        auto* previous_error = errors ? std::cerr.rdbuf(error.rdbuf()) : nullptr;
        int const code = cli(command);
        std::cout.rdbuf(previous);
        if (errors) { std::cerr.rdbuf(previous_error); *errors = error.str(); }
        return std::pair{code, output.str()};
    };
    put(manifest, line("18 2 0 0 gplot-v2"));
    put(config, "");
    assert(cli({"--help", "--config", config.string()}) == 0);
    assert(cli({"-h", "--config", config.string()}) == 0);
    assert(cli({"--config", config.string()}) != 0);
    assert(cli({"--unknown", "--config", config.string()}) != 0);
    for (auto const* command : {"plot", "batch", "bench", "test", "verify", "devices", "parity-check"}) {
        int const calls_before_help = batch_calls;
        auto const [code, help] = capture({command, "--help", "--config", config.string()});
        assert(code == 0 && help.starts_with("Usage:") && help.find(command) != std::string::npos);
        assert(batch_calls == calls_before_help);
        assert(capture({command, "-h", "--config", config.string()}).second == help);
    }
    // Help-like option values must reach their actual parser.
    assert(cli({"plot", "--farmer-pk", "--help", "--config", config.string()}) != 0);
    assert(cli({"verify", "unused.gplot", "--trials", "--help", "--config", config.string()}) != 0);
    put(config, "[defaults]\nverbose=true\nquiet=false\n");
    for (auto const* command : {"plot", "batch", "bench", "test", "verify", "devices", "parity-check"})
        assert(capture({command, "--config", config.string(), "--help"}).first == 0);
    assert(capture({"--config", config.string(), "batch", "--help"}).first == 0);
    assert(capture({"compute", "--config", config.string(), "--help"}).first == 0);
    assert(capture({"--config", config.string(), "compute", "--help-all"}).first == 0);
    std::string detail;
    assert(capture({"verify", "unused.gplot", "--config", config.string()}, &detail).first == 0);
    assert(detail.find("random challenges") != std::string::npos && detail.find("[verify] OK") != std::string::npos);
    assert(capture({"verify", "--config", config.string(), "--no-verbose", "-q", "--", "--help"}, &detail).first == 0);
    assert(detail.find("random challenges") == std::string::npos && detail.find("[verify] OK") != std::string::npos);
    assert(cli({"test", "18", std::string(64, 'a'), "--config", config.string()}) == 0 && last_test_options.verbose);
    assert(cli({"test", "18", std::string(64, 'a'), "2", "0", "0", "1", "--no-verbose", "--config", config.string()}) == 0);
    assert(!last_test_options.verbose);
    assert(cli({"test", "18", std::string(64, 'a'), "-q", "--config", config.string()}) != 0);
    put(config, "[verify]\ntrials=1\nfull=true\n");
    assert(cli({"verify", "unused.gplot", "--config", config.string()}) == 0);
    assert(verified_trials == 1 && verified_full);
    assert(cli({"verify", "unused.gplot", "--config", config.string(), "--trials", "2", "--no-full"}) == 0);
    assert(verified_trials == 2 && !verified_full);
    // Strict conversion rejects prefixes/overflow without changing numeric
    // signs, leading whitespace, config precedence, or verification aliases.
    for (auto const* value : {"2junk", "", "9999999999999999999999999"}) {
        assert(cli({"verify", "unused.gplot", "--config", config.string(), "-n", value}) == 1);
        for (auto const* option : {"--k", "--num", "--strength", "--warmup"})
            assert(cli({"bench", "--config", config.string(), option, value}) == 1);
        for (auto const* option : {"--k", "--num", "--strength", "--plot-index", "--meta-group", "--pipeline-depth"})
            assert(cli({"plot", "--config", config.string(), option, value}) == 1);
        assert(cli({"batch", manifest.string(), "--config", config.string(), "--pipeline-depth", value}) == 1);
        assert(cli({"test", value, std::string(64, 'a'), "--config", config.string()}) == 1);
    }
    assert(cli({"verify", "unused.gplot", "--config", config.string(), "-n", " +002"}) == 0);
    assert(verified_trials == 2);
    put(config, "");
    for (auto const* value : {"2junk", "nan", "inf", "1e999", "0", "-1"})
        assert(cli({"bench", "--config", config.string(), "--target-size", value}) == 1);
    for (auto const* value : {"nan", "inf", "1e999", "18446744073709551616", "-1G", "2GiBjunk"})
        assert(cli({"bench", "--config", config.string(), "--max-host-ram", value}) == 1);
    for (int position = 2; position <= 5; ++position) {
        std::vector<std::string> test = {"test", "18", std::string(64, 'a'), "2", "0", "0", "1", "--config", config.string()};
        test[position + 1] = "2junk";
        assert(cli(test) == 1);
    }
    for (auto const* shell : {"bash", "zsh", "fish"}) {
        auto const [code, script] = capture({"completions", shell, "--config", config.string()});
        assert(code == 0 && script.find("verify") != std::string::npos);
        for (auto const* option : {"save-plan", "from-plan", "save-profile", "harvester", "help-all",
                "discover-drives", "mounts-under", "leave-free", "headroom-bytes", "watch", "inspect", "destination", "retire",
                "policy-only", "alternative", "upgrade-job"})
            assert(script.find(option) != std::string::npos);
        assert(script.find("farming-detect-local") == std::string::npos && script.find("farming-profile") == std::string::npos);
        if (std::string(shell) == "zsh") {
            assert(script.find("'2:plot file:_files'") != std::string::npos);
            assert(script.find("'--trials[Random challenges]:count:'") != std::string::npos);
            assert(script.find("'-n[Random challenges]:count:'") != std::string::npos);
        }
        for (auto const* option : {"trials", "full", "json", "warmup", "max-host-ram", "cpu-workers", "plan", "fill", "grow", "broker", "worker", "group-members", "plan-only", "paths-from", "paths0-from", "extend", "allow-missing", "relocate", "construction", "compute-broker", "compute-local", "compute-directory", "init", "invite", "join", "configure", "status", "host-bytes", "gpu-bytes", "max-frame-bytes", "members", "revoke", "renew-request", "renew-sign", "renew-import", "policy-export", "policy-import", "pause", "drain", "resume", "service", "restart", "remove", "max-brokers", "limit", "role"})
            assert(script.find(option) != std::string::npos);
    }
    put(config, "[batch]\ndevices=\"0\"\nquiet=0\nprogress=false\ncpu=false\nauto-spill=false\n");
    assert(cli({"batch", manifest.string(), "--config", config.string(), "--temp-dir", dir.string()}) == 0);
    assert(std::string(std::getenv("XCHPLOT2_TEMP_DIR")) == dir.string());
    assert(last_options.device_ids == std::vector<int>{0} && !last_options.quiet);
    assert(last_options.cpu_workers == 0 && !last_options.auto_host_ram_spill);
    assert(cli({"batch", manifest.string(), "--config", config.string(), "-d", "2", "-M", "256M",
                "-t", dir.string(), "--no-resume", "-r"}) == 0);
    assert(last_options.device_ids == std::vector<int>{2} && last_options.skip_existing);
    assert(last_options.has_max_host_ram && last_options.max_host_ram == 256ull * 1024 * 1024);
    assert(std::string(std::getenv("XCHPLOT2_TEMP_DIR")) == dir.string());
    put(config, "[bench]\nwarmup=0\nnum=1\nkeep=false\nverbose=false\n");
#ifdef _WIN32
    _putenv_s("POS2GPU_ASSERT_VRAM", "");
#else
    unsetenv("POS2GPU_ASSERT_VRAM");
#endif
    int const before = batch_calls;
    assert(cli({"bench", "--config", config.string(), "--out", dir.string(),
                "-d", "cpu", "-M", "256M", "-t", dir.string()}) == 0);
    assert(batch_calls == before + 1);
    assert(last_options.cpu_selected() && last_options.has_max_host_ram && last_options.max_host_ram == 256ull * 1024 * 1024);
    assert(!last_options.verbose);
    assert(cli({"bench", "--config", config.string(), "--out", dir.string(), "-q", "--progress"}) == 0);
    assert(last_options.quiet && last_options.progress && !last_options.verbose);
    assert(cli({"bench", "--config", config.string(), "--out", dir.string(), "-v", "--no-progress"}) == 0);
    assert(last_options.verbose && !last_options.progress);
    assert(cli({"bench", "--config", config.string(), "--out", dir.string(), "-q", "-v"}) != 0);
    auto const [json_code, json_report] = capture({"bench", "--config", config.string(), "--out", dir.string(), "--json"});
    assert(json_code == 0 && json_report.starts_with("{\"version\":"));
    assert(json_report.find("\"queue_plots\":1") != std::string::npos);
    assert(json_report.find("\"s_per_plot\":1") != std::string::npos);
    assert(json_report.find("\"label\":\"end-to-end\"") != std::string::npos);
    assert(json_report.find("\"pipeline\":\"\"") != std::string::npos);
    auto const [plain_code, plain_output] = capture({"bench", "--config", config.string(), "--out", dir.string(), "--json", "--no-json"});
    assert(plain_code == 0 && plain_output.empty());
#ifndef _WIN32
    std::ostringstream escaped;
    auto* previous = std::cout.rdbuf(escaped.rdbuf());
    // Exercise string escaping through an actual output directory.
    auto const quoted = dir / "quote\"\\tab\t";
    std::filesystem::create_directory(quoted);
    assert(cli({"bench", "--config", config.string(), "--out", quoted.string(), "--json"}) == 0);
    std::cout.rdbuf(previous);
    assert(escaped.str().find("quote\\\"\\\\tab\\u0009") != std::string::npos);
    auto const invalid_utf8 = dir / std::string("byte-\xff");
    std::filesystem::create_directory(invalid_utf8);
    auto const [bytes_code, bytes_report] = capture({"bench", "--config", config.string(), "--out", invalid_utf8.string(), "--json"});
    assert(bytes_code == 0 && bytes_report.find("byte-\\u00ff") != std::string::npos);
#endif

    assert(std::string(std::getenv("POS2GPU_ASSERT_VRAM")) == "1");
#ifdef _WIN32
    _putenv_s("POS2GPU_ASSERT_VRAM", "0");
#else
    setenv("POS2GPU_ASSERT_VRAM", "0", 1);
#endif
    assert(cli({"bench", "--config", config.string(), "--out", dir.string()}) == 0);
    assert(std::string(std::getenv("POS2GPU_ASSERT_VRAM")) == "0");
    for (auto const& fields : {"-1 2 0 0 gplot-v2", "19 2 0 0 gplot-v2", "30 2 0 0 gplot-v2",
                               "18 999 0 0 gplot-v2", "18 2 -1 0 raw-v2", "18 2 1 0 gplot-v2",
                               "18 2 65536 0 raw-v2", "18 2 0 256 gplot-v2", "18 2 0 0 nonsense",
                               "18 2 0 0 false", "18 2 0 0 true", "18 2 0 0 0"}) {
        put(manifest, line(fields));
        bool threw = false;
        try { pos2gpu::parse_manifest(manifest.string()); }
        catch (std::exception const&) { threw = true; }
        assert(threw);
    }
    put(manifest, line("18 2 65535 255 raw-v2"));
    auto const raw_member = pos2gpu::parse_manifest(manifest.string()).front();
    assert(raw_member.raw && raw_member.plot_index == 65535 && raw_member.meta_group == 255);
    for (auto command : {"plot", "bench", "test"})
        assert(cli({command, "--testnet", "--config", config.string()}) == 1);
    for (auto name : {"../escape.gplot", "/absolute.gplot", "sub/file.gplot", "..\\escape.gplot"}) {
        put(manifest, line("18 2 0 0 gplot-v2", name));
        bool threw = false;
        try { pos2gpu::parse_manifest(manifest.string()); }
        catch (std::exception const&) { threw = true; }
        assert(threw);
    }
    auto const tests = dir / "space;true # directory";
    std::filesystem::create_directory(tests);
    auto const marker = dir / "executed";
#ifdef _WIN32
    auto const failing = tests / "quote'_test.exe";
    std::filesystem::copy_file(argv[0], failing);
    _putenv_s("XCHPLOT2_TEST_MARKER", marker.string().c_str());
#else
    auto const failing = tests / "quote'\"_test";
    put(failing, "#!/bin/sh\nprintf executed > '" + marker.string() + "'\nexit 17\n");
    std::filesystem::permissions(failing, std::filesystem::perms::owner_all);
#endif
    put(config, "");
    assert(cli({"parity-check", "--config", config.string(), "--dir", tests.string()}) != 0);
    assert(std::filesystem::exists(marker));

    // A saved job round-trips empty memos and quoted paths, is private, and
    // cannot be replaced by another job (including concurrent publishers).
    pos2gpu::BatchEntry entry;
    entry.k = 18; entry.out_dir = (dir / "space and \"quote\"").string();
    entry.plot_id = pos2gpu::plot_id_for_group(entry.group_id, 0, 0);
    entry.out_name = "plot with spaces.gplot";
    auto const saved = dir / "saved.tsv";
    pos2gpu::write_manifest(saved.string(), {entry});
    assert(pos2gpu::parse_manifest(saved.string()) == std::vector{entry});
#ifdef _WIN32
    PSID owner = nullptr;
    PACL acl = nullptr;
    PSECURITY_DESCRIPTOR security = nullptr;
    assert(::GetNamedSecurityInfoW(saved.c_str(), SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
        &owner, nullptr, &acl, nullptr, &security) == ERROR_SUCCESS);
    SECURITY_DESCRIPTOR_CONTROL control{};
    DWORD revision = 0;
    assert(::GetSecurityDescriptorControl(security, &control, &revision));
    assert((control & SE_DACL_PROTECTED) && acl && acl->AceCount == 1);
    ACCESS_ALLOWED_ACE* ace = nullptr;
    assert(::GetAce(acl, 0, reinterpret_cast<void**>(&ace)));
    assert(ace->Header.AceType == ACCESS_ALLOWED_ACE_TYPE && ::EqualSid(owner, &ace->SidStart));
    assert((ace->Mask & FILE_ALL_ACCESS) == FILE_ALL_ACCESS);
    ::LocalFree(security);
#else
    assert((std::filesystem::status(saved).permissions() &
            (std::filesystem::perms::group_all | std::filesystem::perms::others_all))
           == std::filesystem::perms::none);
#endif
    std::thread same([&] { pos2gpu::write_manifest(saved.string(), {entry}); });
    pos2gpu::write_manifest(saved.string(), {entry});
    same.join();
    auto different = entry; different.group_id[0] = 1;
    different.plot_id = pos2gpu::plot_id_for_group(different.group_id, 0, 0);
    bool refused = false;
    try { pos2gpu::write_manifest(saved.string(), {different}); }
    catch (std::exception const&) { refused = true; }
    assert(refused && pos2gpu::parse_manifest(saved.string()) == std::vector{entry});
#ifdef _WIN32
    refused = false;
    try { pos2gpu::write_manifest(saved.string() + ":keys", {entry}); }
    catch (std::invalid_argument const&) { refused = true; }
    assert(refused && pos2gpu::parse_manifest(saved.string()) == std::vector{entry});
    for (auto const* name : {"plot:keys", "plot.", "plot ", "plot*"}) {
        auto invalid = entry; invalid.out_name = name;
        refused = false;
        try { pos2gpu::validate_batch_entry(invalid); }
        catch (std::invalid_argument const&) { refused = true; }
        assert(refused);
    }
#endif
    auto const race = dir / "race.tsv";
    auto publish = [&](pos2gpu::BatchEntry const& e) {
        try { pos2gpu::write_manifest(race.string(), {e}); return true; }
        catch (std::exception const&) { return false; }
    };
    bool first_won = false;
    std::thread first([&] { first_won = publish(entry); });
    bool const second_won = publish(different);
    first.join();
    assert(first_won != second_won);
    assert(pos2gpu::parse_manifest(race.string()) == std::vector{first_won ? entry : different});
    different.out_dir += '\n';
    refused = false;
    try { pos2gpu::write_manifest((dir / "bad.tsv").string(), {different}); }
    catch (std::exception const&) { refused = true; }
    assert(refused && !std::filesystem::exists(dir / "bad.tsv"));

    auto const plots = dir / "plots with spaces";
    std::vector<std::string> args = {"plot", "--config", config.string(), "-k", "18", "-n", "3",
        "-f", std::string(96, 'a'), "--pool-ph", std::string(64, 'b'), "-o", plots.string(),
        "-d", "cpu", "-M", "256M", "-t", dir.string(), "--quiet", "--no-progress", "--continue-on-error"};
    write_limit = 1; fail_remaining = true;
    auto [failed_code, failed_output] = capture(args);
    assert(failed_code == 3 && std::count(failed_output.begin(), failed_output.end(), '\n') == 1);
#ifdef XCHPLOT2_GROUP_COMMANDS
    assert(!checked_group_directories.empty() && checked_group_directories.back() == plots);
#endif
    assert(last_options.cpu_selected() && last_options.has_max_host_ram && last_options.max_host_ram == 256ull * 1024 * 1024);
    auto const original = last_entries;
    std::filesystem::path job;
    for (auto const& file : std::filesystem::directory_iterator(plots))
        if (file.path().extension() == ".tsv") job = file.path();
    assert(pos2gpu::parse_manifest(job.string()) == original);
    assert(failed_output == (plots / original[0].out_name).string() + '\n');
    write_limit = 0; fail_remaining = false;
    auto resume = args; resume.push_back("-r");
    int const generated = keygen_calls;
    auto [cancelled_code, cancelled_output] = capture(resume);
    assert(cancelled_code == 4 && last_entries == original && keygen_calls == generated);
    assert(cancelled_output == failed_output);  // the one validated existing plot
    write_limit = std::numeric_limits<std::size_t>::max();
    auto [resumed_code, resumed_output] = capture(resume);
    assert(resumed_code == 0 && last_entries == original && keygen_calls == generated);
    assert(std::count(resumed_output.begin(), resumed_output.end(), '\n') == 3);
    assert(capture(resume).first == 0 && keygen_calls == generated);
    auto unmatched = resume; unmatched.insert(unmatched.end(), {"--strength", "4"});
    assert(capture(unmatched).first == 2 && keygen_calls == generated);
    assert(capture(args).first == 0);  // a new job preserves the old manifest
    // Fresh identities within each unseeded job and across identical requests.
    std::set<std::array<uint8_t, 32>> plot_ids;
    std::set<std::vector<uint8_t>> memos;
    for (auto const& entries : {original, last_entries}) {
        for (auto const& e : entries) {
            assert(plot_ids.insert(e.plot_id).second);
            assert(memos.insert(e.memo).second);
        }
    }
    assert(capture(resume).first == 2);  // ambiguous recovery must be explicit
    resume.insert(resume.end(), {"--manifest", job.string()});
    assert(capture(resume).first == 0 && last_entries == original);
    auto wrong = resume; wrong.insert(wrong.end(), {"--strength", "4"});
    assert(capture(wrong).first == 2);
    assert(pos2gpu::parse_manifest(job.string()) == original);

    write_limit = 1; throw_remaining = true;
    auto [throw_code, throw_output] = capture(args);
    assert(throw_code == 2 && std::count(throw_output.begin(), throw_output.end(), '\n') == 1);
    write_limit = 0; throw_remaining = false;
    auto [empty_code, empty_output] = capture(args);
    assert(empty_code == 4 && empty_output.empty());
    fail_remaining = true;
    auto [all_failed_code, all_failed_output] = capture(args);
    assert(all_failed_code == 3 && all_failed_output.empty());
    fail_remaining = false; write_limit = std::numeric_limits<std::size_t>::max();
    auto seeded = args;
    seeded.insert(seeded.end(), {"--seed", std::string(64, 'c'), "--manifest", (dir / "seeded.tsv").string()});
#ifdef XCHPLOT2_GROUP_COMMANDS
    checked_group_directories.clear();
#endif
    assert(capture(seeded).first == 0);
#ifdef XCHPLOT2_GROUP_COMMANDS
    assert((checked_group_directories == std::vector{plots, dir}));
#endif
    auto const seeded_entries = last_entries;
    seeded.push_back("--resume");
    assert(capture(seeded).first == 0 && last_entries == seeded_entries);
    // A fixed seed shares group keys across meta groups, but their files and
    // default job manifests must remain distinct.
    auto seeded_meta = args;
    seeded_meta.insert(seeded_meta.end(), {"--seed", std::string(64, 'd')});
    assert(capture(seeded_meta).first == 0);
    auto const meta_zero = last_entries;
    seeded_meta.insert(seeded_meta.end(), {"--meta-group", "1"});
    assert(capture(seeded_meta).first == 0);
    for (std::size_t i = 0; i < last_entries.size(); ++i) {
        assert(last_entries[i].group_id == meta_zero[i].group_id);
        assert(last_entries[i].plot_id != meta_zero[i].plot_id);
        assert(last_entries[i].out_name != meta_zero[i].out_name);
    }
    // Grouped jobs retain membership in durable manifests, including one-member
    // jobs, so an old batch runner cannot silently treat them as ordinary jobs.
    auto grouped_entry = entry; grouped_entry.grouped = true; grouped_entry.group_members = 7;
    auto const grouped_manifest = dir / "grouped.tsv";
    pos2gpu::write_manifest(grouped_manifest.string(), {grouped_entry});
    assert(pos2gpu::parse_manifest(grouped_manifest.string()) == std::vector{grouped_entry});
    for (int count : {0, 65536}) {
        auto invalid = grouped_entry; invalid.group_members = count;
        refused = false;
        try { pos2gpu::validate_batch_entry(invalid); } catch (std::exception const&) { refused = true; }
        assert(refused);
    }
    for (auto const& parent : {dir / ".xchplot2", dir / ".xchplot2" / "replot-absent"}) {
        for (auto invalid : {entry, grouped_entry}) {
            invalid.out_dir = parent.string(); refused = false;
            try { pos2gpu::validate_batch_entry(invalid); }
            catch (std::invalid_argument const& error) { refused = std::string(error.what()).find("reserved .xchplot2") != std::string::npos; }
            assert(refused && !std::filesystem::exists(parent));
            invalid.raw = true; invalid.grouped = false; invalid.group_members = 1; refused = false;
            try { pos2gpu::validate_batch_entry(invalid); }
            catch (std::invalid_argument const& error) { refused = std::string(error.what()).find("reserved .xchplot2") != std::string::npos; }
            assert(refused && !std::filesystem::exists(parent));
        }
    }
    for (auto const* suffix : {"", " 0", " 65536", " 2junk", " 2 extra"}) {
        auto text = line("18 2 0 0 gplot-v2-grouped"); text.pop_back(); text += suffix; text += '\n';
        put(manifest, text); refused = false;
        try { (void)pos2gpu::parse_manifest(manifest.string()); } catch (std::exception const&) { refused = true; }
        assert(refused);
    }
#ifdef XCHPLOT2_GROUP_COMMANDS
    auto const [fill_help_code, fill_help] = capture({"fill", "--config", config.string(), "--help"});
    assert(fill_help_code == 0 && fill_help.find("fill [OPTIONS] -F PLAN") != std::string::npos);
    assert(fill_help.find("-F, --from-plan") != std::string::npos && fill_help.find("-r --manifest FILE") != std::string::npos);
    assert(fill_help.find("fill farm.plan") != std::string::npos && fill_help.find("destination directory") != std::string::npos);
    assert(fill_help.find("--discover-drives") != std::string::npos && fill_help.find("--leave-free") != std::string::npos);
    auto const [fill_all_code, fill_all] = capture({"fill", "--config", config.string(), "--help-all"});
    assert(fill_all_code == 0 && fill_all.find("--paths-from") != std::string::npos && fill_all.size() > fill_help.size());
    auto const [grow_help_code, grow_help] = capture({"grow", "--config", config.string(), "--help"});
    assert(grow_help_code == 0 && grow_help.find("always enabled") != std::string::npos);
    assert(grow_help.find("--paths-from") == std::string::npos && grow_help.find("--manifest") == std::string::npos);
    assert(grow_help.find("--farming-") == std::string::npos && grow_help.find("--extend") == std::string::npos);
    put(config, "[fill]\nplan-only=true\njson=true\nmax-host-ram=\"512M\"\n");
    assert(cli({"fill", dir.string(), "--config", config.string(), "--devices", "cpu"}) == 0);
    assert(last_group_options.plan_only && last_group_options.json && last_group_options.backend == "cpu");
    assert(last_group_options.max_ram == 512ull * 1024 * 1024);
    assert(cli({"plan", dir.string(), "--config", config.string(), "--progress"}) == 0);
    assert(last_group_options.progress && !last_group_options.verbose);
    assert(cli({"plan", dir.string(), "--config", config.string(), "-v", "--no-progress"}) == 0);
    assert(last_group_options.verbose && !last_group_options.progress);
    assert(cli({"plan", dir.string(), "--config", config.string(), "-q", "--progress"}) == 0);
    assert(last_group_options.quiet && last_group_options.progress && !last_group_options.verbose);
    assert(cli({"plan", dir.string(), "--config", config.string(), "-q", "-v"}) != 0);
    auto const other = dir / "another destination";
    auto const third = dir / "third destination";
    std::filesystem::create_directory(other);
    std::filesystem::create_directory(third);
    {
        std::vector<std::string> arguments{"pos2_pr118_grow", dir.string(), "--plan-only",
            "--destination", other.string(), "--destination", third.string()};
        std::vector<char*> argv;
        for (auto& argument : arguments) argv.push_back(argument.data());
        auto const standalone = pos2gpu::parse_group_options(static_cast<int>(argv.size()), argv.data());
        assert(standalone.command.empty() && standalone.farm_action.empty() && standalone.action_destinations.empty());
        assert(standalone.plan_only && standalone.directory_farm && standalone.target == std::filesystem::canonical(dir));
        assert((standalone.destinations == std::vector{std::filesystem::canonical(other), std::filesystem::canonical(third)}));
    }
    assert(cli({"fill", "--config", config.string(), "-r", "-d", "gpu2:tiny", "-M", "768M",
                "-j", "2", "-t", other.string(), "--temp-dir", third.string(),
                "--", dir.string(), other.string(), third.string()}) == 0);
    assert(last_group_options.plan_only && last_group_options.resume && last_group_options.backend == "gpu");
    assert(last_group_options.device == 2 && last_group_options.tier == "tiny" && last_group_options.workers == 2);
    assert(last_group_options.max_ram == 768ull * 1024 * 1024);
    assert((last_group_options.destinations == std::vector<std::filesystem::path>{other, third}));
    assert((last_group_options.temporary == std::vector<std::filesystem::path>{other, third}));
    assert(cli({"fill", dir.string(), "--config", config.string(), "--replot", "--no-plan-only",
                "--temp-dirs", other.string(), third.string(), "--exclude-destination", other.string(), third.string(),
                "--farm-config", other.string(), third.string(), "--replot-confirmation", "directory",
                "--reclaim-before-bytes", "0"}) == 0);
    assert(last_group_options.replot && !last_group_options.plan_only && last_group_options.target == dir);
    assert((last_group_options.temporary == std::vector<std::filesystem::path>{other, third}));
    assert(last_group_options.farm_configs == last_group_options.temporary &&
        last_group_options.excluded_destinations == last_group_options.temporary);
    assert(last_group_options.replot_confirmation == "directory" && last_group_options.reclaim_before_set && !last_group_options.reclaim_before_bytes);
    assert(cli({"plan", "--config", config.string(), "--replot", "--discover-farm"}) == 0);
    assert(last_group_options.replot && last_group_options.plan_only && last_group_options.discover_farm && last_group_options.target.empty());
    for (auto const* option : {"--temp-dirs", "--farm-config", "--exclude-destination"})
        assert(cli({"fill", dir.string(), "--config", config.string(), option, "--json"}) != 0);
    assert(cli({"fill", dir.string(), "--config", config.string(), "--replot-confirmation", "directory"}) != 0);
    assert(cli({"fill", dir.string(), "--config", config.string(), "--replot", "--replot-confirmation", "none"}) != 0);
    auto const paths_file = dir / "destinations.txt";
    put(paths_file, other.string() + "\r\n" + third.string() + "\n");
    assert(cli({"plan", "--config", config.string(), "--paths-from", paths_file.string()}) == 0);
    assert(last_group_options.target == other && last_group_options.destinations == std::vector{third});
    auto const nul_file = dir / "destinations.nul";
    put(nul_file, other.string() + '\0' + third.string() + '\0');
    assert(cli({"plan", "--config", config.string(), "--paths0-from", nul_file.string()}) == 0);
    assert(last_group_options.target == other && last_group_options.destinations == std::vector{third});
    assert(cli({"plan", "--config", config.string(), "--paths-from", nul_file.string()}) != 0);
    std::istringstream paths_input(other.string() + '\0' + third.string() + '\0');
    auto* previous_input = std::cin.rdbuf(paths_input.rdbuf());
    assert(cli({"plan", "--config", config.string(), "--paths0-from", "-"}) == 0);
    std::cin.rdbuf(previous_input); std::cin.clear();
    assert(last_group_options.target == other && last_group_options.destinations == std::vector{third});
    put(paths_file, "");
    assert(cli({"plan", "--config", config.string(), "--paths-from", paths_file.string()}) != 0);
    auto const previous_directory = std::filesystem::current_path();
    std::filesystem::current_path(dir);
    assert(cli({"plan", "--config", config.string(), "--temp-dirs", "@scratch", "another scratch-bytes",
                "--exclude-destination", "@excluded", "--", "@destination"}) == 0);
    assert(last_group_options.target == dir / "@destination" &&
        (last_group_options.temporary == std::vector{dir / "@scratch", dir / "another scratch-bytes"}) &&
        last_group_options.excluded_destinations == std::vector{dir / "@excluded"});
    auto const calls_before_literal = group_calls;
    assert(cli({"plan", "--config", config.string(), "--", "--help", "@literal", "--config"}) == 0);
    assert(group_calls == calls_before_literal + 1 && last_group_options.target == dir / "--help");
    assert((last_group_options.destinations == std::vector{dir / "@literal", dir / "--config"}));
    put(dir / "--", other.string() + "\n");
    assert(cli({"plan", "--config", config.string(), "--paths-from", "--", "-M", "256M"}) == 0);
    assert(last_group_options.target == other && last_group_options.max_ram == 256ull * 1024 * 1024);
    put(dir / "@paths", third.string() + "\n");
    assert(cli({"plan", "--config", config.string(), "--paths-from", "@paths", "--manifest", "--help"}) == 0);
    assert(last_group_options.target == third && last_group_options.manifest == dir / "--help");
    assert(cli({"--config", config.string(), "plan", "--paths-from", "@paths"}) == 0);
    assert(last_group_options.target == third);
    put(dir / "@config", "");
    assert(cli({"--config", "@config", "plan", "--paths-from", "@paths"}) == 0);
    assert(last_group_options.target == third);
    assert(cli({"plan", "--config", config.string(), "--relocate", "--config", "@new"}) == 0);
    assert((last_group_options.target.empty() && last_group_options.relocations == std::vector{std::pair{dir / "--config", dir / "@new"}}));
    assert(cli({"--config", config.string(), "plan", "--relocate", "--config", "@new"}) == 0);
    assert((last_group_options.relocations == std::vector{std::pair{dir / "--config", dir / "@new"}}));
    put(config, "[plan]\npaths-from=\"@paths\"\nmanifest=\"--help\"\ncapacity-bytes=\"64M\"\n");
    assert(cli({"plan", "--config", config.string(), "--", "--config"}) == 0);
    assert(last_group_options.target == third && last_group_options.destinations == std::vector{dir / "--config"});
    assert(last_group_options.manifest == dir / "--help" && last_group_options.capacity_set);
    put(config, "[fill]\nplan-only=true\njson=true\nmax-host-ram=\"512M\"\n");
    std::filesystem::current_path(previous_directory);
    assert(cli({"fill", "--config", config.string(), "-r", "--allow-missing", other.string()}) == 0);
    assert(last_group_options.resume && last_group_options.resume_set && last_group_options.allow_missing && last_group_options.target == other);
    assert(cli({"fill", "--config", config.string(), "--extend", other.string(), third.string()}) == 0);
    assert(last_group_options.resume && last_group_options.extend);
    assert(cli({"fill", "--config", config.string(), "--relocate", other.string(), third.string()}) == 0);
    assert((last_group_options.resume && last_group_options.target.empty() && last_group_options.relocations == std::vector{std::pair{other, third}}));
    assert(cli({"fill", "--config", config.string(), "-r", "--manifest", manifest.string()}) == 0);
    assert(last_group_options.resume && last_group_options.target.empty() && last_group_options.manifest == manifest);
    assert(cli({"plan", "--config", config.string()}) == 0 && last_group_options.target.empty());
    assert(cli({"fill", "--config", config.string(), "-r"}) == 0 && last_group_options.target.empty());
    assert(cli({"fill", "--config", config.string(), "--no-resume", other.string()}) == 0);
    assert(!last_group_options.resume && last_group_options.resume_set);
    assert(cli({"fill", "--config", config.string(), other.string()}) == 0 && !last_group_options.resume_set);
    assert(cli({"plan", "--config", config.string(), "--discover-drives", other.string(), "--mounts-under", third.string()}) == 0);
    assert((last_group_options.mounts_under == std::vector{other, third}));
    assert(cli({"plan", "--config", config.string(), other.string(), "--headroom-bytes", "32M", "--leave-free", "64MiB"}) == 0);
    assert(last_group_options.headroom_set && last_group_options.headroom == 64ull * 1024 * 1024);
    assert(cli({"plan", "--config", config.string(), other.string(), "--group-members", "auto"}) == 0);
    assert(last_group_options.group_auto && last_group_options.group_members_set && !last_group_options.group_members);
    assert(cli({"plan", "--config", config.string(), other.string()}) == 0 && !last_group_options.group_members_set);
    assert(cli({"plan", "--config", config.string(), other.string(), "-k", "18", "-s", "3", "--meta-group", "7"}) == 0);
    assert(last_group_options.k_set && last_group_options.strength_set && last_group_options.meta_set);
    assert(cli({"plan", "--config", config.string(), other.string()}) == 0);
    assert(!last_group_options.k_set && !last_group_options.strength_set && !last_group_options.meta_set);
    auto const [status_help_code, status_help] = capture({"status", "--config", config.string(), "--help"});
    assert(status_help_code == 0 && status_help.find("--watch") != std::string::npos && status_help.find("--inspect") != std::string::npos);
    assert(status_help.find("--construction") == std::string::npos);
    assert(cli({"status", "--config", config.string(), other.string(), "--watch", "--inspect", "--json"}) == 0);
    assert(last_group_options.command == "status" && last_group_options.replot_status && last_group_options.plan_only &&
        last_group_options.status_watch && last_group_options.status_inspect && last_group_options.json &&
        last_group_options.target == other && !last_group_options.automatic_compute && !last_group_options.farming_profile);
    assert(cli({"plan", "--config", config.string(), "--status", other.string(), "--watch", "--inspect"}) == 0);
    assert(last_group_options.replot_status && last_group_options.status_watch && last_group_options.status_inspect);
    put(config, "[status]\nwatch=true\ninspect=true\n");
    assert(cli({"status", "--config", config.string(), "--no-watch", "--no-inspect"}) == 0);
    assert(last_group_options.replot_status && !last_group_options.status_watch && !last_group_options.status_inspect && last_group_options.target.empty());
    put(config, "");
    assert(cli({"status", "--config", config.string(), "--manifest", "--help"}) == 0);
    assert(last_group_options.manifest.filename() == "--help");
    assert(cli({"status", "--config", config.string(), "--", "--help"}) == 0);
    assert(last_group_options.target.filename() == "--help");
    auto const status_plan = dir / "status.plan", status_manifest = dir / "status.manifest";
    put(status_plan, "ordinary-group-farming-preview-v5\n");
    put(status_manifest, "ordinary-group-farming-v11\n");
    assert(cli({"status", "--config", config.string(), status_plan.string()}) == 0);
    assert(last_group_options.from_plan == status_plan && last_group_options.manifest.empty());
    assert(cli({"status", "--config", config.string(), status_manifest.string()}) == 0);
    assert(last_group_options.from_plan.empty() && last_group_options.manifest == status_manifest);
#ifndef _WIN32
    auto const linked_status = dir / "linked-status.plan";
    std::filesystem::create_symlink(status_plan, linked_status);
    assert(cli({"status", "--config", config.string(), linked_status.string()}) != 0);
    std::filesystem::remove(linked_status);
#endif
    assert(cli({"status", "--config", config.string(), other.string(), third.string()}) != 0);
    assert(cli({"status", "--config", config.string(), "--discover-drives", other.string()}) != 0);
    std::string status_scope_error;
    int const status_scope_code = capture({"status", "--config", config.string(), "--paths-from", "missing-path-list"}, &status_scope_error).first;
    assert(status_scope_code != 0 && status_scope_error.find("farm status accepts a job selector") != std::string::npos);
    assert(cli({"status", "--config", config.string(), other.string(), "--live", "--timeout-ms", "1500", "--compute", "off"}) == 0);
    assert(last_group_options.status_live && last_group_options.status_timeout_ms == 1500 && last_group_options.compute_mode == "off");
    assert(cli({"status", "--config", config.string(), other.string(), "--live", "--no-live"}) == 0 && !last_group_options.status_live);
    for (auto const* timeout : {"0", "99", "60001"})
        assert(cli({"status", "--config", config.string(), other.string(), "--live", "--timeout-ms", timeout}) != 0);
    assert(cli({"status", "--config", config.string(), other.string(), "--timeout-ms", "1500"}) != 0);
    assert(cli({"plan", "--config", config.string(), other.string(), "--live"}) != 0);
    assert(cli({"plan", "--config", config.string(), other.string(), "--watch"}) != 0);
    for (auto const* action : {"pause", "drain", "resume", "retire"})
    {
        auto const [action_help_code, action_help] = capture({action, "--config", config.string(), "--help"});
        assert(action_help_code == 0 && action_help.find("--destination") != std::string::npos && action_help.find("--construction") == std::string::npos);
        assert(action_help.find("--policy-only") != std::string::npos && action_help.find("Without --destination") != std::string::npos);
        assert(cli({action, "--config", config.string(), other.string(), "--destination", other.string(), "--destination", third.string()}) == 0);
        assert(last_group_options.farm_action == action && last_group_options.target == other &&
            last_group_options.directory_farm && last_group_options.resume && !last_group_options.resume_set &&
            !last_group_options.plan_only && !last_group_options.replot_status &&
            !last_group_options.automatic_compute && !last_group_options.farming_profile);
        assert((last_group_options.action_destinations == std::vector{other, third}));
        assert(cli({action, "--config", config.string(), "--manifest", status_manifest.string()}) == 0);
        assert(last_group_options.action_destinations.empty() && last_group_options.manifest == status_manifest);
        assert(cli({action, "--config", config.string(), other.string(), "--policy-only"}) == 0 && last_group_options.policy_only);
        assert(cli({action, "--config", config.string(), other.string(), "--policy-only", "--no-policy-only"}) == 0 && !last_group_options.policy_only);
        assert(cli({action, "--config", config.string()}) == 0 && last_group_options.target.empty());
        assert(cli({action, "--config", config.string(), status_plan.string()}) != 0);
        assert(cli({action, "--config", config.string(), other.string(), third.string()}) != 0);
        assert(cli({action, "--config", config.string(), other.string(), "--status"}) != 0);
        assert(cli({action, "--config", config.string(), "--paths-from", "missing-path-list"}) != 0);
    }
    assert(cli({"pause", "--config", config.string(), "--manifest", "--help", "--destination", "--config", "--destination", "@literal"}) == 0);
    assert(last_group_options.manifest.filename() == "--help" && last_group_options.action_destinations.front().filename() == "--config" &&
        last_group_options.action_destinations.back().filename() == "@literal");
    assert(cli({"plan", "--config", config.string(), other.string(), "--destination", third.string()}) != 0);
    assert(cli({"fill", "--config", config.string(), other.string(), "--policy-only"}) != 0);
    for (auto const* name : {"memory", "eligible-groups", "deadline", "cpu-cores", "lookup-workers", "capacity", "memory,eligible-groups"})
    {
        assert(cli({"plan", "--config", config.string(), other.string(), "--alternative", name}) == 0);
        assert(last_group_options.farming_alternative == name && !last_group_options.upgrade_job);
    }
    assert(cli({"plan", "--config", config.string(), "--manifest", status_manifest.string(), "--alternative", "memory"}) == 0);
    assert(last_group_options.manifest == status_manifest && last_group_options.farming_alternative == "memory");
    for (auto const* value : {"", "unknown", "memory,memory", "memory,", ",memory", "memory,,deadline", "capacity,memory"})
        assert(cli({"plan", "--config", config.string(), other.string(), "--alternative", value}) != 0);
    for (auto const* mode : {"fill", "grow", "status", "pause", "resume"})
    {
        assert(cli({mode, "--config", config.string(), other.string(), "--alternative", "memory"}) != 0);
        assert(cli({mode, "--config", config.string(), other.string(), "--upgrade-job"}) != 0);
    }
    assert(cli({"plan", "--config", config.string(), "--alternative", "memory"}) != 0);
    assert(cli({"plan", "--config", config.string(), status_plan.string(), "--alternative", "memory"}) != 0);
    assert(cli({"plan", "--config", config.string(), "--from-plan", status_plan.string(), "--upgrade-job"}) != 0);
    assert(cli({"plan", "--config", config.string(), "--manifest", status_manifest.string(), "--upgrade-job"}) == 0);
    assert(last_group_options.upgrade_job && !last_group_options.update && last_group_options.farming_alternative.empty());
    assert(cli({"plan", "--config", config.string(), other.string(), "--upgrade-job", "--alternative", "memory"}) != 0);
    for (auto const* scope : {"--metadata-only", "--inventory", "--assess-jobs", "--calibrate", "--status"})
        assert(cli({"plan", "--config", config.string(), other.string(), "--alternative", "memory", scope}) != 0);
    put(config, "[plan]\nalternative=\"memory\"\nupgrade-job=false\n");
    assert(cli({"plan", "--config", config.string(), other.string(), "--alternative", "capacity"}) == 0);
    assert(last_group_options.farming_alternative == "capacity" && !last_group_options.upgrade_job);
    put(config, "[resume]\npolicy-only=true\n");
    assert(cli({"resume", "--config", config.string(), other.string()}) == 0 && last_group_options.policy_only);
    assert(cli({"resume", "--config", config.string(), other.string(), "--no-policy-only"}) == 0 && !last_group_options.policy_only);
    put(config, "[fill]\nplan-only=true\njson=true\nmax-host-ram=\"512M\"\n");
    assert(cli({"fill", "--config", config.string(), "-D", other.string()}) != 0);
    assert(cli({"fill", dir.string(), "--config", config.string(), "--no-plan-only", "--no-json"}) == 0);
    assert(!last_group_options.plan_only && !last_group_options.json);
    auto const argfile = dir / "group.args";
    put(argfile, "--group-members 3\n--max-host-ram 768M\n");
    assert(cli({"plan", dir.string(), "--config", config.string(), "@" + argfile.string()}) == 0);
    assert(last_group_options.plan_only && last_group_options.group_members == 3 && last_group_options.max_ram == 768ull * 1024 * 1024);
    assert(cli({"plan", dir.string(), "--config", config.string(), "--devices", "all"}) == 0);
    assert(last_group_options.devices == "all" && last_group_options.backend == "cpu");
    assert(cli({"plan", dir.string(), "--config", config.string(), "--devices", "cpu,cpu"}) == 0);
    assert(last_group_options.backend == "cpu");
    assert(cli({"plan", dir.string(), "--config", config.string(), "--devices", "all", "--backend", "gpu", "--device", "2"}) == 0);
    assert(last_group_options.devices.empty() && last_group_options.backend == "gpu" && last_group_options.device == 2);
    assert(cli({"plan", dir.string(), "--config", config.string(), "--devices", "cpu:tiny"}) != 0);
    assert(cli({"plan", dir.string(), "--config", config.string(), "--devices", "gpu2:tiny,gpu2:plain"}) != 0);
    int const before_missing_value = group_calls;
    for (auto const* option : {"-d", "-M", "-j", "-t", "--paths-from", "--paths0-from", "--manifest", "--relocate", "--discover-drives", "--leave-free", "--destination"})
        assert(cli({"plan", dir.string(), "--config", config.string(), option}) != 0);
    assert(group_calls == before_missing_value);
    put(config, "[compute.worker]\nhost-bytes=\"1000000000\"\n");
    assert(cli({"compute", "worker", "--config", config.string(), "--host-bytes", "2000000000"}) == 0);
    assert((last_compute == std::vector<std::string>{"worker", "--host-bytes", "1000000000", "--host-bytes", "2000000000"}));
    put(config, "");
    for (auto const* command : {"members", "revoke", "renew-request", "renew-sign", "renew-import",
                               "policy-export", "policy-import", "status", "pause", "drain", "resume", "service"}) {
        assert(cli({"compute", command, "--config", config.string(), "--help"}) == 0);
        assert((last_compute == std::vector<std::string>{command, "--help"}));
    }
    auto const group_dir = dir / "groups";
    auto const group_job = dir / "group-job.tsv";
    std::vector<std::string> group_args = {"plot", "--config", config.string(), "-k", "18", "-n", "2",
        "-f", std::string(96, 'a'), "--pool-ph", std::string(64, 'b'), "-o", group_dir.string(),
        "--group-members", "3", "-d", "cpu", "-M", "512M", "-j", "2", "-t", dir.string(),
        "--manifest", group_job.string(), "--quiet"};
    auto const [group_code, group_paths] = capture(group_args);
    assert(group_code == 0 && std::count(group_paths.begin(), group_paths.end(), '\n') == 2);
    assert(last_entries.size() == 2 && last_entries.front().group_members == 3 && last_entries.front().grouped);
    assert(last_group_options.workers == 2 && last_group_options.max_ram == 512ull * 1024 * 1024);
    assert(last_group_options.temporary == std::vector<std::filesystem::path>{dir});
    auto const group_entries = last_entries;
    int const prior_derivations = keygen_calls;
    group_args.emplace_back("--resume");
    assert(capture(group_args).first == 0 && last_entries == group_entries && keygen_calls == prior_derivations);
    auto group_progress = group_args;
    group_progress.emplace_back("--progress");
    assert(capture(group_progress).first == 0);
    assert(last_options.progress && last_group_options.progress && !last_group_options.verbose);
    group_progress.insert(group_progress.end(), {"--no-quiet", "-v", "--no-progress"});
    assert(capture(group_progress).first == 0);
    assert(last_options.verbose && last_group_options.verbose && !last_group_options.progress);
    assert(capture({"batch", group_job.string(), "-r", "-d", "cpu", "-M", "256M", "-j", "3", "-t", dir.string(),
                    "--config", config.string()}).first == 0);
    assert(last_entries == group_entries);
    assert(last_group_options.command == "batch");
    assert(last_group_options.workers == 3 && last_group_options.max_ram == 256ull * 1024 * 1024);
    assert(last_group_options.temporary == std::vector<std::filesystem::path>{dir});
    group_args.insert(group_args.end(), {"--group-members", "4"});
    assert(capture(group_args).first != 0);
    auto const invalid_job = dir / "invalid-group.tsv";
    group_args.insert(group_args.end(), {"--group-members", "0", "--manifest", invalid_job.string()});
    assert(capture(group_args).first != 0 && !std::filesystem::exists(invalid_job));
#endif
    for (auto const& file : std::filesystem::recursive_directory_iterator(dir))
        assert(file.path().filename().string().find(".partial.") == std::string::npos);
    std::filesystem::remove_all(dir);
    std::cout << "CLI: config, manifests, durable resume, completed outputs, and literal test paths passed\n";
}
