#include "group_commands.hpp"
#include "host/Cancel.hpp"
#include <algorithm>
#include <iostream>

namespace {
void require(bool condition, std::string const& message)
{ if (!condition) throw std::invalid_argument(message); }
pos2gpu::GroupOptions command_arguments(std::string_view command, std::vector<std::string> tokens)
{
    tokens.insert(tokens.begin(), "xchplot2 " + std::string(command));
    std::vector<char*> argv; argv.reserve(tokens.size());
    for (auto& token : tokens) argv.push_back(token.data());
    auto args = pos2gpu::parse_group_options(int(argv.size()), argv.data(), command);
    args.cancelled = pos2gpu::cancel_requested;
    return args;
}

void command_help(std::string_view command)
{
    std::cout << "Usage: xchplot2 " << command
        << (command == "grow" ? " FILE [OPTIONS]\n" : " DIR [-D|--destination DIR ...] [OPTIONS]\n");
    if (command == "grow")
        std::cout << "Add members to an existing ordinary v2 group, recovering interrupted construction first.\n";
    else if (command == "plan")
        std::cout << "Preview the farming layout without keys, file creation, locks or compute startup.\n";
    else
        std::cout << "Plan and populate destination directories. --plan-only is the same preview as plan.\n";
    std::cout <<
        "  -D, --destination PATH   additional destination; repeat for more drives\n"
        "  -k, --k N                 even k in 18..28, default 28\n"
        "  -s, --strength N          default 2\n"
        "  --group-members auto|N    members per file (plan/fill); default auto\n"
        "  --max-groups N            per-directory file limit (fill)\n"
        "  --max-total-members N     per-directory member limit (fill)\n"
        "  --max-members N           total members in an existing file (grow)\n"
        "  --plan-only               preview a fill without writing\n"
        "  -r, --resume              reuse the saved layout and recover interrupted work\n"
        "  -f, --farmer-pk HEX        farmer public key (fill)\n"
        "  -p, --pool-pk HEX          pool public key; choose one pool source\n"
        "  --pool-ph HEX             pool contract puzzle hash\n"
        "  -c, --pool-contract-address ADDRESS  pool contract address\n"
        "  -S, --seed HEX            stable 32-byte seed for fill; keep it for resume\n"
        "  --backend cpu|gpu|auto    producer selection, default auto\n"
        "  -d, --devices cpu|gpuN    select one local producer\n"
        "  --device N --tier NAME    local GPU and production tier\n"
        "  -M, --max-host-ram SIZE   shared construction/producer RAM budget (e.g. 14G)\n"
        "  -j, --workers N           shared compression CPU allowance\n"
        "  --batch-members N         maximum members generated per construction batch\n"
        "  --capacity-bytes N        aggregate output capacity per filesystem\n"
        "  --headroom-bytes N        filesystem bytes to leave free\n"
        "  -t, --temp-dir DIR        contribution storage; repeat for more devices\n"
        "  --destination-temp off|on|ssd|hdd  destination scratch permission, default off\n"
        "  --temp-capacity-bytes N   temporary capacity per filesystem\n"
        "  --construction auto|incremental|staged|merged|external\n"
        "  --member-storage compact|raw  retained fragment representation\n"
        "  --window-bytes N --checkpoint-bytes N  fixed I/O/durability windows\n"
        "  --io-overlap on|off --generation-overlap auto|on|off --cache-release on|off\n"
        "  --farming-objective energy|latency  choose layout objective\n"
        "  --farming-reader stateless|cached  final harvester reader profile\n"
        "  --farming-cores N --farming-lookup-workers N  final harvester CPU/concurrency\n"
        "  --farming-memory-bytes N --farming-index-cache-bytes N\n"
        "  --farming-deadline-ms N --farming-max-eligible-groups N\n"
        "  --farming-filter-bits N --farming-future-filter-bits N\n"
        "  --farming-chain-us-per-member N --farming-index-ns-per-entry N\n"
        "  --farming-cpu-work-weight N --farming-io-work-weight N\n"
        "  --compute auto|off        use saved pool membership, default auto\n"
        "  --compute-local on|off    share local capacity while plotting, default on\n"
        "  --compute-directory DIR  saved pool membership directory\n"
        "  --compute-broker HOST:PORT --compute-owner-id ID\n"
        "  --compute-ca FILE --compute-cert FILE --compute-key FILE\n"
        "  --compute-inflight N --compute-codec-ram auto|N\n"
        "  --compute-timeout-ms N --compute-socket-timeout-ms N --compute-frame-bytes N\n"
        "  --compute-listen HOST:PORT --compute-advertise HOST:PORT\n"
        "  --compute-offer-ram N --compute-offer-cpus N --compute-offer-vram N\n"
        "  --compute-offer-slots N --compute-offer-backend cpu|gpu|auto\n"
        "  --json                    machine-readable plan/metrics on stdout\n"
        "  -q, --quiet               suppress informational summaries\n"
        "  -v, --verbose             show member progress\n"
        "Construction state is recovered with the same build. Keep the saved plan,\n"
        "destinations and any owned temporary files together until completion.\n";
}
} // namespace

pos2gpu::GroupOptions pos2gpu::group_batch_options(pos2gpu::BatchOptions const& options, std::vector<std::string> const& extra)
{
    require(!options.use_all_devices && options.device_ids.size() <= 1 &&
                options.cpu_node_ids.empty() && !(options.cpu_selected() && !options.device_ids.empty()),
            "group construction uses one local producer; select --devices cpu or gpuN, or use shared compute");
    require(!options.shard_plot && !options.pipeline_plot && options.pipeline_tiers.empty() &&
                options.strategy == pos2gpu::BatchStrategy::Auto,
            "group construction does not use single-member sharding or pipeline strategies");
    require(!options.continue_on_error, "group construction stops on errors to preserve the shared destination audit");
    require(options.cpu_workers == pos2gpu::kCpuWorkersAuto,
            "group construction uses --workers for its shared CPU allowance; omit --cpu-workers");
    require(!options.has_max_host_ram || options.max_host_ram,
            "group construction requires a positive --max-host-ram size; spill-to-min is only available for single-member plotting");
    std::vector<std::string> tokens{"xchplot2-group.gplot"};
    if (options.cpu_selected() || !options.device_ids.empty())
        tokens.insert(tokens.end(), {"--backend", options.cpu_selected() ? "cpu" : "gpu"});
    if (!options.device_ids.empty())
    {
        tokens.insert(tokens.end(), {"--device", std::to_string(options.device_ids.front())});
        auto const tier = options.per_device_tier.find(options.device_ids.front());
        if (tier != options.per_device_tier.end()) tokens.insert(tokens.end(), {"--tier", tier->second});
        else if (!options.streaming_tier.empty()) tokens.insert(tokens.end(), {"--tier", options.streaming_tier});
    }
    else if (!options.streaming_tier.empty()) tokens.insert(tokens.end(), {"--tier", options.streaming_tier});
    if (options.has_max_host_ram) tokens.insert(tokens.end(), {"--max-ram", std::to_string(options.max_host_ram)});
    if (options.skip_existing) tokens.emplace_back("--resume");
    if (options.quiet) tokens.emplace_back("--quiet");
    if (options.verbose || options.progress) tokens.emplace_back("--verbose");
    tokens.insert(tokens.end(), extra.begin(), extra.end());
    auto args = command_arguments("plot", std::move(tokens));
    args.report_metrics = false;
    return args;
}

int pos2gpu::run_group_command(std::string_view command, std::vector<std::string> const& tokens)
{
    if (std::find(tokens.begin(), tokens.end(), "--help") != tokens.end() ||
        std::find(tokens.begin(), tokens.end(), "-h") != tokens.end())
    { command_help(command); return 0; }
    try { return run_group(command_arguments(command, tokens)); }
    catch (std::exception const& error)
    { std::cerr << '[' << command << "] FAILED: " << error.what() << '\n'; return 2; }
}
