// Exercise the real CLI/config/manifest code; plotting entry points are inert.
#undef NDEBUG
#include "gpu/SyclDeviceList.hpp"
#include "host/BatchPlotter.hpp"
#include "host/PlotFileWriterParallel.hpp"
#include "pos2_keygen.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

extern "C" int xchplot2_main(int, char**);
namespace {
size_t verified_trials = 0;
bool verified_full = false;
pos2gpu::BatchOptions last_options;
int batch_calls = 0;
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
std::vector<GpuDeviceInfo> list_gpu_devices() { return {}; }
std::string plot_to_file(GpuPlotOptions const&, std::string const&) { return {}; }
std::vector<std::string> worker_labels(std::vector<int> const& d) { return std::vector<std::string>(d.size(), "test"); }
size_t batch_worker_count(BatchOptions const&, int) { return 1; }
BatchResult run_batch(std::vector<BatchEntry> const& entries, BatchOptions const& opts)
{
    ++batch_calls; last_options = opts;
    WorkerTimeline worker;
    for (auto const& entry : entries) {
        put(std::filesystem::path(entry.out_dir) / entry.out_name, "stub plot");
        worker.completion_seconds.push_back(worker.completion_seconds.size() + 1);
    }
    BatchResult result;
    result.plots_written = entries.size(); result.bytes_written = 9 * entries.size();
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
int pos2_keygen_derive_plot(uint8_t const*, size_t, uint8_t const*, uint8_t const*, int, uint8_t,
    uint16_t, uint8_t, uint8_t*, uint8_t*, size_t*) { return POS2_BAD_SEED; }
int pos2_keygen_derive_subseed(uint8_t const*, uint64_t, uint8_t*) { return POS2_BAD_SEED; }
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
    assert(last_options.cpu_workers == 0 && !last_options.auto_host_ram_spill);
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
    std::filesystem::remove_all(dir);
    std::cout << "CLI: numeric config, overrides, manifest validation, and literal test paths passed\n";
}
