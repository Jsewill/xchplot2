// Exercise the real CLI/config/manifest code; plotting entry points are inert.
#undef NDEBUG
#include "gpu/CudaDeviceList.hpp"
#include "host/BatchPlotter.hpp"
#include "host/PlotFileWriterParallel.hpp"
#include "pos2_keygen.h"
#include <cassert>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>
#include <unistd.h>

extern "C" int xchplot2_main(int, char**);
namespace {
size_t verified_trials = 0;
bool verified_full = false;
pos2gpu::BatchOptions last_options;
int batch_calls = 0;
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
    std::ofstream out(path); out << data; assert(out.good());
}
}
namespace pos2gpu {
CudaDeviceQueryResult list_cuda_devices() { return {}; }
std::string plot_to_file(GpuPlotOptions const&, std::string const&) { return {}; }
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
VerifyResult verify_plot_file(std::string const&, size_t trials, bool full)
{
    verified_trials = trials; verified_full = full;
    return {trials, 1, 1, full ? size_t{1} : size_t{0}};
}
}
extern "C" {
int pos2_keygen_decode_address(char const*, uint8_t*) { return POS2_BAD_ADDRESS; }
int pos2_keygen_derive_plot(uint8_t const* seed, size_t, uint8_t const* farmer, uint8_t const* pool,
    int kind, uint8_t, uint16_t, uint8_t, uint8_t* id, uint8_t* memo, size_t* size)
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
int main()
{
    char path[] = "/tmp/xchplot2-cli-test-XXXXXX";
    assert(mkdtemp(path));
    std::filesystem::path const dir(path), config = dir / "config.toml", manifest = dir / "manifest.tsv";
    auto line = [&](std::string const& fields, std::string const& name = "plot.plot2") {
        return fields + " " + std::string(64, 'a') + " 00 " + dir.string() + " " + name + "\n";
    };
    put(manifest, line("18 2 0 0 false"));
    put(config, "");
    assert(cli({"--help", "--config", config.string()}) == 0);
    assert(cli({"-h", "--config", config.string()}) == 0);
    assert(cli({"--config", config.string()}) != 0);
    assert(cli({"--unknown", "--config", config.string()}) != 0);
    put(config, "[verify]\ntrials=1\nfull=true\n");
    assert(cli({"verify", "unused.plot2", "--config", config.string()}) == 0);
    assert(verified_trials == 1 && verified_full);
    assert(cli({"verify", "unused.plot2", "--config", config.string(), "--trials", "2", "--no-full"}) == 0);
    assert(verified_trials == 2 && !verified_full);
    put(config, "[batch]\ndevices=\"0\"\nquiet=0\nprogress=false\ncpu=false\nauto-spill=false\n");
    assert(cli({"batch", manifest.string(), "--config", config.string()}) == 0);
    assert(last_options.device_ids == std::vector<int>{0} && !last_options.quiet);
    assert(last_options.cpu_workers == 0 && last_options.no_auto_spill);
    put(config, "[bench]\nwarmup=0\nnum=1\nkeep=false\nverbose=false\n");
    int const before = batch_calls;
    assert(cli({"bench", "--config", config.string(), "--out", dir.string()}) == 0);
    assert(batch_calls == before + 1);
    for (auto const& fields : {"-1 2 0 0 0", "19 2 0 0 0", "18 999 0 0 0", "18 2 -1 0 0",
                               "18 2 65536 0 0", "18 2 0 256 0", "18 2 0 0 nonsense"}) {
        put(manifest, line(fields));
        bool threw = false;
        try { pos2gpu::parse_manifest(manifest.string()); }
        catch (std::exception const&) { threw = true; }
        assert(threw);
    }
    for (auto name : {"../escape.plot2", "/absolute.plot2", "sub/file.plot2", "..\\escape.plot2"}) {
        put(manifest, line("18 2 0 0 true", name));
        bool threw = false;
        try { pos2gpu::parse_manifest(manifest.string()); }
        catch (std::exception const&) { threw = true; }
        assert(threw);
    }
    auto const tests = dir / "space;true # directory";
    std::filesystem::create_directory(tests);
    auto const failing = tests / "quote'\"_test";
    auto const marker = dir / "executed";
    put(failing, "#!/bin/sh\nprintf executed > '" + marker.string() + "'\nexit 17\n");
    std::filesystem::permissions(failing, std::filesystem::perms::owner_all);
    put(config, "");
    assert(cli({"parity-check", "--config", config.string(), "--dir", tests.string()}) != 0);
    assert(std::filesystem::exists(marker));

    // A saved job round-trips empty memos and quoted paths, is private, and
    // cannot be replaced by another job (including concurrent publishers).
    pos2gpu::BatchEntry entry;
    entry.k = 18; entry.out_dir = (dir / "space and \"quote\"").string();
    entry.out_name = "plot with spaces.plot2";
    auto const saved = dir / "saved.tsv";
    pos2gpu::write_manifest(saved.string(), {entry});
    assert(pos2gpu::parse_manifest(saved.string()) == std::vector{entry});
    assert((std::filesystem::status(saved).permissions() &
            (std::filesystem::perms::group_all | std::filesystem::perms::others_all))
           == std::filesystem::perms::none);
    std::thread same([&] { pos2gpu::write_manifest(saved.string(), {entry}); });
    pos2gpu::write_manifest(saved.string(), {entry});
    same.join();
    auto different = entry; different.plot_id[0] = 1;
    bool refused = false;
    try { pos2gpu::write_manifest(saved.string(), {different}); }
    catch (std::exception const&) { refused = true; }
    assert(refused && pos2gpu::parse_manifest(saved.string()) == std::vector{entry});
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
        "--quiet", "--no-progress"};
    auto capture = [&](std::vector<std::string> const& command) {
        std::ostringstream output;
        auto* previous = std::cout.rdbuf(output.rdbuf());
        int const code = cli(command);
        std::cout.rdbuf(previous);
        return std::pair{code, output.str()};
    };
    write_limit = 1; fail_remaining = true;
    auto [failed_code, failed_output] = capture(args);
    assert(failed_code == 3 && std::count(failed_output.begin(), failed_output.end(), '\n') == 1);
    auto const original = last_entries;
    std::filesystem::path job;
    for (auto const& file : std::filesystem::directory_iterator(plots))
        if (file.path().extension() == ".tsv") job = file.path();
    assert(pos2gpu::parse_manifest(job.string()) == original);
    assert(failed_output == (plots / original[0].out_name).string() + '\n');
    write_limit = 0; fail_remaining = false;
    auto resume = args; resume.push_back("--resume");
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
    assert(capture(seeded).first == 0);
    auto const seeded_entries = last_entries;
    seeded.push_back("--resume");
    assert(capture(seeded).first == 0 && last_entries == seeded_entries);
    for (auto const& file : std::filesystem::recursive_directory_iterator(dir))
        assert(file.path().filename().string().find(".partial.") == std::string::npos);
    std::filesystem::remove_all(dir);
    std::cout << "CLI: config, manifests, durable resume, completed outputs, and literal test paths passed\n";
}
