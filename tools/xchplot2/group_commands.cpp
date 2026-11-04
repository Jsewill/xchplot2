#include "group_commands.hpp"
#include "host/Cancel.hpp"
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

void command_usage(std::string_view command)
{
    if (command == "fill")
        std::cout << "Usage: xchplot2 fill [OPTIONS] DIR...\n"
            "       xchplot2 fill [OPTIONS] PLAN\n"
            "       xchplot2 fill [OPTIONS] -F PLAN\n"
            "       xchplot2 fill [OPTIONS] -r [DIR...]\n"
            "       xchplot2 fill [OPTIONS] -r --manifest FILE\n"
            "       xchplot2 fill [OPTIONS] --extend DIR...\n"
            "       xchplot2 fill [OPTIONS] --relocate OLD NEW\n"
            "       xchplot2 fill [OPTIONS] (--paths-from FILE | --paths0-from FILE)\n"
            "       xchplot2 fill --plan-only [OPTIONS] [DIR... | PLAN]\n";
    else std::cout << "Usage: xchplot2 " << command
        << (command == "grow" ? " FILE [OPTIONS]\n" : " [OPTIONS] [DIR... | PLAN]\n");
    if (command == "plan") std::cout << "       xchplot2 plan --assess-jobs [OPTIONS] PLAN...\n";
}

void command_help_all(std::string_view command)
{
    command_usage(command);
    if (command == "grow")
        std::cout << "Add members to an existing ordinary v2 group, recovering interrupted construction first.\n";
    else if (command == "plan")
        std::cout << "Preview a farming layout; without directories, suggest mounted storage.\n"
            "Writes only requested --save-plan or --save-profile files; no plotting services are started.\n";
    else
        std::cout << "Plan and populate destination directories. --plan-only is the same preview as plan.\n"
            "Resume through any member, --manifest FILE, or the current member directory.\n";
    if (command != "grow") std::cout <<
        "  --paths-from FILE        literal destination paths, one per line; - reads stdin\n"
        "  --paths0-from FILE       NUL-delimited destination paths; - reads stdin\n"
        "  --manifest FILE          locate an existing farming job directly\n"
        "  --save-plan FILE         save a keyless plan, e.g. farm.plan\n"
        "  -F, --from-plan FILE     explicit alternative to supplying PLAN directly\n"
        "  --save-profile FILE      export harvester settings without destinations\n"
        "  --metadata-only          plan: inspect saved metadata without checking destinations or recovery\n"
        "  --assess-jobs             plan: assess saved keyless job forecasts under one harvester budget\n"
        "  --extend                 add destinations while preserving saved assignments\n"
        "  --allow-missing          with --resume, continue available members; incomplete exit 4\n"
        "  --relocate OLD NEW       move a saved destination; NEW locates the job, OLD must be offline\n";
    std::cout <<
        "  -k, --k N                 default 28 for farming; even 18..26 for testing\n"
        "  -s, --strength N          default 2\n";
    if (command == "grow") std::cout <<
        "  --max-members N           total members in the existing file\n"
        "  -r, --resume              recover interrupted work before growing (always enabled)\n";
    else std::cout <<
        "  --group-members auto|N    members per file (plan/fill); default auto\n"
        "  --max-groups N            per-directory file limit (fill)\n"
        "  --max-total-members N     per-directory member limit (fill)\n"
        "  --plan-only               preview a fill without writing\n"
        "  -r, --resume              load keys and all destinations from any member directory\n"
        "  -f, --farmer-pk HEX        farmer public key (fill)\n"
        "  -p, --pool-pk HEX          pool public key; choose one pool source\n"
        "  --pool-ph HEX             pool contract puzzle hash\n"
        "  -c, --pool-contract-address ADDRESS  pool contract address\n"
        "  -S, --seed HEX            optional 32-byte seed; otherwise generated and saved privately\n";
    std::cout <<
        "  --backend cpu|gpu|auto    producer selection, default auto\n"
        "  -d, --devices cpu|gpuN    select one local producer\n"
        "  --device N --tier NAME    local GPU and production tier\n"
        "  -M, --max-host-ram SIZE   shared construction/producer RAM budget (e.g. 14G)\n"
        "  -j, --workers N           shared compression CPU allowance\n"
        "  --batch-members N         maximum members generated per construction batch\n"
        << (command == "grow" ? "  --capacity-bytes N        output capacity for the group file\n" :
            "  --capacity-bytes N        aggregate output capacity per filesystem\n") <<
        "  --headroom-bytes N        filesystem bytes to leave free\n"
        "  -t, --temp-dir DIR        contribution storage; repeat for more devices\n"
        "  --destination-temp off|on|ssd|hdd  destination scratch permission, default off\n"
        "  --temp-capacity-bytes N   temporary capacity per filesystem\n"
        "  --construction auto|incremental|staged|merged|external\n"
        "  --member-storage compact|raw  retained fragment representation\n"
        "  --window-bytes N --checkpoint-bytes N  fixed I/O/durability windows\n"
        "  --io-overlap on|off --generation-overlap auto|on|off --cache-release on|off\n";
    if (command != "grow") std::cout <<
        "  --harvester local|FILE    use this harvester or import its saved profile\n"
        "  --chia-root DIR           Chia configuration root for local detection\n"
        "  --chia-executable FILE    installed Chia CLI for local read-only detection\n"
        "  --farming-objective energy|latency  choose layout objective\n"
        "  --farming-reader stateless|cached  final harvester reader profile\n"
        "  --farming-cores N --farming-lookup-workers N  final harvester CPU/concurrency\n"
        "  --farming-memory-bytes N --farming-index-cache-bytes N\n"
        "  --farming-deadline-ms N --farming-max-eligible-groups N\n"
        "  --farming-filter-bits N --farming-future-filter-bits N\n"
        "  --farming-chain-us-per-member N --farming-index-ns-per-entry N\n"
        "  --farming-cpu-work-weight N --farming-io-work-weight N\n";
#ifdef XCHPLOT2_GROUP_COMMANDS
    if (command == "plan") std::cout <<
        "  --calibrate              measure the explicitly supplied plot files; use --json or --save-profile\n"
        "  --calibration-samples N   quality lookups per plot, default 16; maximum 1024\n"
        "  --calibration-timeout-ms N  reader worker deadline, default 30000\n"
        "  --calibration-memory-bytes SIZE  reader worker memory limit, default 512M\n";
#endif
    std::cout <<
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
        "  -v, --verbose             show assumptions, construction details and metrics\n"
        "  --progress / --no-progress  force aggregate progress on or off\n";
    if (command == "grow") std::cout <<
        "Construction state is recovered with the same build. Keep the group file\n"
        "and any owned temporary files together until completion.\n";
    else std::cout <<
        "Construction state is recovered with the same build. Keep every private\n"
        "grow-farm.plan copy with its plots; it contains the seed for private plot keys.\n"
        "Keep any grow-farm.plan.next and owned temporary files for recovery too.\n";
}

void command_help(std::string_view command, bool all)
{
    if (command == "grow" || all) { command_help_all(command); return; }
    command_usage(command);
    if (command == "plan")
        std::cout << "Preview a layout for the selected destinations. No directories: suggest mounted storage.\n"
            "Save a plan once, then pass it directly to fill:\n\n"
            "  xchplot2 plan /disk-a/plots /disk-b/plots --harvester local --save-plan farm.plan\n"
            "  xchplot2 fill farm.plan -f FARMER_KEY -c POOL_ADDRESS\n\n"
            "  --save-plan FILE          save a keyless plan for fill\n"
            "  -F, --from-plan FILE      inspect a saved plan (also accepts positional PLAN)\n"
            "  --save-profile FILE       export harvester settings; omit destinations\n"
            "  --metadata-only           read saved PLAN or --manifest FILE without live checks\n"
            "  --assess-jobs PLAN...      jointly assess saved keyless forecasts under one budget\n";
    else
        std::cout << "Fill selected directories or execute a saved plan.\n\n"
            "  xchplot2 fill farm.plan -f FARMER_KEY -c POOL_ADDRESS\n"
            "  xchplot2 fill /disk-a/plots -f FARMER_KEY -c POOL_ADDRESS\n"
            "  xchplot2 fill --resume /disk-a/plots\n"
            "Run fill -r from the current member directory to resume without a path.\n"
            "New jobs require a farmer key and one pool source; saved jobs restore their keys.\n\n"
            "  -F, --from-plan FILE      execute a saved plan (also accepts positional PLAN)\n"
            "  -r, --resume              recover the whole saved job through any member\n"
            "  --extend                  add destinations to a saved job\n"
            "  --plan-only               preview without plotting\n"
            "  -f, --farmer-pk HEX        farmer public key (new fill)\n"
            "  -c, --pool-contract-address ADDRESS  pool contract address (new fill)\n"
            "  -d, --devices cpu|gpuN    choose one local plot producer\n"
            "  -M, --max-host-ram SIZE   plotting buffer budget, e.g. 14G\n"
            "  -t, --temp-dir DIR        optional construction scratch\n";
#ifdef XCHPLOT2_GROUP_COMMANDS
    if (command == "plan") std::cout <<
        "  --calibrate FILE...       measure supplied plots; use --json or --save-profile FILE\n"
        "  --calibration-samples N   bounded quality lookups per plot, default 16\n";
#endif
    std::cout <<
        "  --harvester local|FILE    use this harvester or a profile from another machine\n"
        "  --group-members auto|N    automatic layout or a member limit per file\n"
        "  --capacity-bytes SIZE     budget per filesystem, e.g. 500G\n"
        "  --headroom-bytes SIZE     leave this space free, default 64M\n"
        "  --json                    machine-readable output\n"
        "  -q, --quiet               suppress informational summaries\n"
        "  -v, --verbose             show assumptions and construction details\n"
        "  --progress / --no-progress  force aggregate progress on or off\n"
        "\nk28 is the farming size; smaller sizes are for testing.\n"
        "Estimates cover this job and use assumed service costs.\n"
        "Use --help-all for all resource, recovery and shared-compute options.\n";
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
    if (options.verbose) tokens.emplace_back("--verbose");
    tokens.emplace_back(options.progress ? "--progress" : "--no-progress");
    tokens.insert(tokens.end(), extra.begin(), extra.end());
    auto args = command_arguments("plot", std::move(tokens));
    args.report_metrics = false;
    return args;
}

int pos2gpu::run_group_command(std::string_view command, std::vector<std::string> const& tokens)
{
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        auto const& token = tokens[i];
        if (token == "--") break;
        if (token == "--help" || token == "--help-all" || token == "-h")
        { command_help(command, token == "--help-all"); return 0; }
        i += pos2gpu::group_option_values(token);
    }
    try { return run_group(command_arguments(command, tokens)); }
    catch (std::exception const& error)
    { std::cerr << '[' << command << "] FAILED: " << error.what() << '\n'; return 2; }
}
