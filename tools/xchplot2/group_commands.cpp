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
    if (command == "status")
    {
        std::cout << "Usage: xchplot2 status [DIR | PLAN | --manifest FILE] [--watch] [--inspect] [--live]\n";
        return;
    }
    if (command == "pause" || command == "drain" || command == "resume" || command == "retire")
    {
        std::cout << "Usage: xchplot2 " << command << " [DIR | --manifest FILE] [--destination DIR]...\n";
        return;
    }
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
    if (command == "plan") std::cout << "       xchplot2 plan --inventory ROOT... [--save-plan inventory.plan]\n"
        "       xchplot2 plan DEST... --replot inventory.plan [--save-plan migration.plan]\n"
        "       xchplot2 plan --revalidate --from-plan migration.plan --harvester UPDATED_PROFILE --save-plan NEW\n";
    if (command == "plan" || command == "fill") std::cout << "       xchplot2 " << command << " --status [DIR | PLAN | --manifest FILE]\n";
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
            "Resume through any destination in the job, --manifest FILE, or the current destination directory.\n";
    if (command == "plan") std::cout <<
        "  --alternative NAMES      review assessed resource changes, or capacity alone\n"
        "  --upgrade-job            review additional space for a stopped job's recovery records\n";
    if (command != "grow") std::cout <<
        "  --paths-from FILE        literal destination paths, one per line; - reads stdin\n"
        "  --paths0-from FILE       NUL-delimited destination paths; - reads stdin\n"
        "  --discover-drives DIR    select supported mounted storage at or below DIR; repeatable\n"
        "  --mounts-under DIR       alias for --discover-drives\n"
        "  --manifest FILE          locate an existing farming job directly\n"
        "  --save-plan FILE         save a plan without private plot keys, e.g. farm.plan\n"
        "  -F, --from-plan FILE     explicit alternative to supplying PLAN directly\n"
        "  --save-profile FILE      export reusable harvester or migration setup\n"
        "  --resource-setting SETTING  inventory/replot setup: declare a resource bound by its stable ID; repeatable\n"
        "  --metadata-only          plan: inspect saved metadata without checking destinations or recovery\n"
        "  --assess-jobs             plan: check saved plans together against one harvester budget\n"
        "  --inventory              plan: inspect explicit existing farm roots without changing files\n"
        "  --replot INVENTORY       plan: review migration of selected inventory into destinations\n"
        "  --replot-verification MODE  replacement checks (default structural):\n"
        "                           structural: headers and chunk index\n"
        "                           sampled: also sample full proofs for selected groups\n"
        "                           strict: also sample full proofs for every group\n"
        "  --replot-proof-first-groups N  sampled: proof-check the first N groups; default 1\n"
        "  --replot-proof-every-groups N  sampled: proof-check every Nth group across destinations; 0 disables (default)\n"
        "  --replot-source-root ROOT  select an exact inventory root; repeat for more roots\n"
        "  --replot-exclude-root ROOT  protect an exact inventory root; repeat for more roots\n"
        "  --replot-max-source-files N  maximum old files selected for this review\n"
        "  --replot-max-source-bytes SIZE  maximum allocated old-file bytes selected\n"
        "  --status                 read ordinary job or migration progress without writer locks\n"
        "  --watch                  refresh status until interrupted\n"
        "  --inspect                status: scan outputs and check recovery checkpoints\n"
        "  --live                   status: query shared compute and migration service readiness\n"
        "  --timeout-ms N           status --live: total query time in milliseconds, 100..60000; default 5000\n"
        "  --revalidate             plan: save a successor for updated resources or verification policy\n"
        "  --reclaim-before-bytes N  allocated-byte limit for early deletion; default zero\n"
        "                           Early deletion is irreversible: old plots may be removed before replacements are ready.\n"
        "  --farm-adapter chia|gigahorse|drchia  explicit installed inventory/reader adapter\n"
        "  --farm-adapter-timeout-ms N  inventory/migration adapter operation deadline, default 30000\n"
        "  --extend                 add destinations or refresh capacity, including during fill\n"
        "  --update                 revise capacity and parameters for future groups\n"
        "  --cancel-update          cancel a queued ordinary-job update\n"
        "  --allow-missing          with --resume, continue available destinations; incomplete exit 4\n"
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
        "  -r, --resume              load keys and all destinations from any directory in the job\n"
        "  -f, --farmer-pk HEX        farmer public key (fill)\n"
        "  -p, --pool-pk HEX          pool public key; choose one pool source\n"
        "  --pool-ph HEX             pool contract puzzle hash\n"
        "  -c, --pool-contract-address ADDRESS  pool contract address\n"
        "  -S, --seed HEX            optional 32-byte seed; otherwise generated and saved privately\n";
    std::cout <<
        "  --backend cpu|gpu|auto    CPU/GPU plot generation, default auto\n"
        "  -d, --devices SPEC        select cpu, gpu, all, or named local devices\n"
        "                           combine with commas; GPU selectors accept :tier\n"
        "  --device N --tier NAME    local GPU and production tier\n"
        "  -M, --max-host-ram SIZE   combined plotting RAM budget (e.g. 14G)\n"
        "  -j, --workers N           shared plotting CPU budget\n"
        "  --batch-members N         maximum members generated per construction batch\n"
        << (command == "grow" ? "  --capacity-bytes N        output capacity for the group file\n" :
            "  --capacity-bytes N        aggregate output capacity per filesystem\n") <<
        "  --leave-free SIZE         filesystem space to leave free; --headroom-bytes is an alias\n"
        "  -t, --temp-dir DIR        temporary plotting storage; repeat for more devices\n"
        "  --destination-temp off|on|ssd|hdd  allow temporary files on destinations; default off\n"
        "  --temp-capacity-bytes N   temporary capacity per filesystem\n"
        "  --construction auto|incremental|staged|merged|external\n"
        "  --member-storage compact|raw  temporary member storage format\n"
        "  --window-bytes N          I/O window size in bytes\n"
        "  --checkpoint-bytes N      target bytes between recovery checkpoints\n"
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
    std::cout << "Construction recovery requires compatible checkpoint format, algorithm and\n"
        "native representation. Legacy checkpoints need their original build.\n";
    if (command == "grow") std::cout <<
        "Keep the group file and any owned temporary files together until completion.\n";
    else std::cout <<
        "Keep every private .xchplot2/grow-farm.plan copy with its plots; it contains\n"
        "private plot seeds. Keep .next copies and owned temporary files for recovery too.\n"
        "Older jobs keep recovery files beside their plots; leave those files in place.\n";
}

void command_help(std::string_view command, bool all)
{
    if (command == "status")
    {
        command_usage(command);
        std::cout << "Read saved farm progress through any destination in the job, a plan, or its manifest.\n"
            "Without a selector, use the current destination directory.\n\n"
            "  --watch                  refresh until interrupted\n"
            "  --inspect                scan outputs and check recovery checkpoints\n"
            "  --live                   query shared compute and migration service readiness\n"
            "  --timeout-ms N           total live query time in milliseconds, 100..60000; default 5000\n"
            "  --compute off|auto       select whether live status inspects shared compute\n"
            "  --compute-directory DIR  saved pool to inspect\n"
            "  --harvester FILE          local migration profile with selected readers and service settings\n"
            "  -F, --from-plan FILE     inspect a saved plan\n"
            "  --manifest FILE          locate a saved farming job directly\n"
            "  -M, --max-host-ram SIZE   inspection memory budget\n"
            "  --json                    machine-readable output; --watch emits one object per refresh\n"
            "  -q, --quiet               suppress informational summaries\n"
            "  -v, --verbose             show details\n";
        return;
    }
    if (command == "pause" || command == "drain" || command == "resume" || command == "retire")
    {
        command_usage(command);
        if (command == "pause") std::cout << "Pause plotting at the next safe checkpoint. Keep completed plots.\n";
        else if (command == "drain") std::cout << "Finish started files without starting new ones. Keep completed plots.\n";
        else if (command == "resume") std::cout << "Allow paused or drained work to continue. Start the job if it is idle.\n";
        else std::cout << "Detach destinations from this job, keeping their plots and recovery files.\n"
            "Available started files finish before detachment. Resume cannot rejoin detached destinations.\n"
            "Missing destinations keep their resources reserved until they return and acknowledge retirement.\n";
        std::cout << "Without a selector, use the current destination directory.\n"
            "Without --destination, apply the action to the whole job.\n\n"
            "The running fill applies controls at its next safe checkpoint.\n"
            "Available destinations honor controls while other destinations are missing.\n\n"
            "  --destination DIR        select a saved destination; repeat for more destinations\n"
            "  --policy-only            save the control setting without starting an idle job\n"
            "  --manifest FILE          locate the saved farming job directly\n"
            "  --json                    machine-readable result\n"
            "  -q, --quiet               suppress informational summaries\n"
            "  -v, --verbose             show details\n";
        return;
    }
    if (command == "grow" || all) { command_help_all(command); return; }
    command_usage(command);
    if (command == "plan")
        std::cout << "Preview a layout for the selected destinations. Without directories, suggest mounted storage.\n"
            "Save a plan once, then pass it directly to fill:\n\n"
            "  xchplot2 plan /disk-a/plots /disk-b/plots --harvester local --save-plan farm.plan\n"
            "  xchplot2 fill farm.plan -f FARMER_KEY -c POOL_ADDRESS\n\n"
            "  --save-plan FILE          save a plan for fill without private plot keys\n"
            "  --alternative NAMES       review assessed memory, eligible-groups, deadline, cpu-cores or lookup-workers changes; capacity alone\n"
            "  --upgrade-job             review additional space for a stopped job's recovery records\n"
            "  -F, --from-plan FILE      inspect a saved plan (also accepts positional PLAN)\n"
            "  --save-profile FILE       export reusable harvester or migration setup\n"
            "  --resource-setting SETTING  declare a remaining inventory/replot resource bound; repeatable\n"
            "  --metadata-only           read saved PLAN or --manifest FILE without live checks\n"
            "  --assess-jobs PLAN...      check saved plans together against one harvester budget\n"
            "  --inventory ROOT...      inspect existing farms; optionally --save-plan inventory.plan\n"
            "  --replot INVENTORY       review migration into DEST...; save with --save-plan\n"
            "  --replot-verification MODE  replacement checks (default structural):\n"
            "                           structural: headers and chunk index\n"
            "                           sampled: also sample full proofs for selected groups\n"
            "                           strict: also sample full proofs for every group\n"
            "  --replot-proof-first-groups N  sampled: proof-check the first N groups; default 1\n"
            "  --replot-proof-every-groups N  sampled: proof-check every Nth group across destinations; 0 disables (default)\n"
            "  --replot-source-root ROOT  select old files from this exact inventory root\n"
            "  --replot-exclude-root ROOT  protect old files in this exact inventory root\n"
            "  --replot-max-source-files N  bound old-file selection for a trial run\n"
            "  --replot-max-source-bytes SIZE  bound selected allocated old-file bytes\n"
            "  --status DIR|PLAN        inspect completed work, allowance and blocked reasons\n"
            "  --revalidate              review updated resources or verification; use --from-plan and --save-plan\n"
            "  --reclaim-before-bytes SIZE  allocated-byte limit for early deletion; default zero\n"
            "Early deletion is irreversible: old plots may be removed before replacements are ready.\n";
    else
        std::cout << "Fill selected directories or execute a saved plan.\n\n"
            "  xchplot2 fill farm.plan -f FARMER_KEY -c POOL_ADDRESS\n"
            "  xchplot2 fill /disk-a/plots -f FARMER_KEY -c POOL_ADDRESS\n"
            "  xchplot2 fill /disk-a/plots\n"
            "Repeat fill DIR to continue a saved job; explicit options revise future work.\n"
            "Run fill -r from a destination directory to keep its saved policy without supplying a path.\n"
            "New jobs require a farmer key and one pool source; saved jobs restore their keys.\n\n"
            "  -F, --from-plan FILE      execute a saved plan (also accepts positional PLAN)\n"
            "  --status DIR|PLAN        inspect ordinary job or migration progress without plotting\n"
            "  -r, --resume              recover the saved job without changing its future policy\n"
            "  --no-resume               require a new job\n"
            "  --extend                  add destinations or refresh capacity during fill\n"
            "  --update                  revise capacity or future group parameters during fill\n"
            "  --cancel-update           cancel a waiting update\n"
            "  --plan-only               preview without plotting\n"
            "  -f, --farmer-pk HEX        farmer public key (new fill)\n"
            "  -c, --pool-contract-address ADDRESS  pool contract address (new fill)\n"
            "  -d, --devices SPEC        select cpu, gpu, all, or named local devices\n"
            "  -M, --max-host-ram SIZE   combined plotting RAM budget, e.g. 14G\n"
            "  -t, --temp-dir DIR        temporary plotting storage\n";
#ifdef XCHPLOT2_GROUP_COMMANDS
    if (command == "plan") std::cout <<
        "  --calibrate FILE...       measure supplied plots; use --json or --save-profile FILE\n"
        "  --calibration-samples N   bounded quality lookups per plot, default 16\n";
#endif
    std::cout <<
        "  --harvester local|FILE    use this harvester or a profile from another machine\n"
        "  --discover-drives DIR    select supported mounted storage at or below DIR, at any depth\n"
        "  -k, --k N                 plot size, default 28\n"
        "  -s, --strength N          plot strength, default 2\n"
        "  --group-members auto|N    automatic layout or a member limit per file\n"
        "  --capacity-bytes SIZE     budget per filesystem, e.g. 500G\n"
        "  --leave-free SIZE         leave this space free, default 64M; also --headroom-bytes\n"
        "                           byte sizes accept K, M, G, T suffixes (binary units)\n"
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
    require(!options.shard_plot && !options.pipeline_plot && options.pipeline_tiers.empty() &&
                options.strategy == pos2gpu::BatchStrategy::Auto,
            "group construction does not use single-member sharding or pipeline strategies");
    require(!options.continue_on_error, "group construction stops on errors so destinations can be rechecked safely; omit --continue-on-error");
    require(options.cpu_workers == pos2gpu::kCpuWorkersAuto,
            "group construction uses --workers for its shared CPU budget; omit --cpu-workers");
    require(!options.has_max_host_ram || options.max_host_ram,
            "group construction requires a positive --max-host-ram size; spill-to-min is only available for single-member plotting");
    std::vector<std::string> tokens{"xchplot2-group.gplot"};
    if (!options.streaming_tier.empty()) tokens.insert(tokens.end(), {"--tier", options.streaming_tier});
    if (options.devices_specified || options.use_all_devices || options.cpu_selected() || !options.device_ids.empty())
    {
        std::string selection;
        auto append = [&](std::string const& value) { if (!selection.empty()) selection += ','; selection += value; };
        if (options.use_all_devices) append("gpu" + (options.all_gpus_tier.empty() ? "" : ':' + options.all_gpus_tier));
        for (int device : options.device_ids)
        {
            auto const tier = options.per_device_tier.find(device);
            append("gpu" + std::to_string(device) + (tier == options.per_device_tier.end() ? "" : ':' + tier->second));
        }
        if (options.cpu_selected())
        {
            if (options.use_all_cpu_nodes || options.cpu_node_ids.empty()) append("cpu");
            else for (int node : options.cpu_node_ids) append("cpu" + std::to_string(node));
        }
        require(!selection.empty(), "no local devices selected");
        tokens.insert(tokens.end(), {"--devices", selection});
    }
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
