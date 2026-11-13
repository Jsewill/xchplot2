#include "group_commands.hpp"
#include "device_options.hpp"
#include "host/GroupHarvesterProfile.hpp"
#include "host/GroupFarmFile.hpp"
#include "host/GroupFarmMetadata.hpp"
#ifdef XCHPLOT2_GROUP_COMMANDS
#include "host/ComputeSession.hpp"
#include "host/NumaTopology.hpp"
#include "gpu/DeviceIds.hpp"
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
#include "gpu/SyclDeviceList.hpp"
#endif
#endif
#include <algorithm>
#include <climits>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
#include "host/ComputeNetwork.hpp"
#endif

namespace {
namespace fs = std::filesystem;
constexpr uint64_t mib = 1024 * 1024;
void require(bool condition, std::string const& message)
{ if (!condition) throw std::runtime_error(message); }
uint64_t number(char const* text, std::string const& option)
{
    std::string value(text);
    require(!value.empty() && value.find_first_not_of("0123456789") == value.npos, option + " requires a nonnegative integer");
    try { return std::stoull(value); }
    catch (std::out_of_range const&) { throw std::runtime_error(option + " exceeds the maximum integer value (18446744073709551615)"); }
}
uint64_t sum(uint64_t a, uint64_t b)
{ require(b <= UINT64_MAX - a, "byte count overflow"); return a + b; }
uint64_t product(uint64_t a, uint64_t b)
{ require(!b || a <= UINT64_MAX / b, "byte count overflow"); return a * b; }
void resolve_group_devices(pos2gpu::GroupOptions& result)
{
    if (result.devices.empty()) return;
    pos2gpu::BatchOptions selection;
    require(pos2gpu::parse_devices_arg(result.devices, selection), "invalid --devices selection");
    // A single explicit GPU keeps the existing deferred admission: saved
    // plans and publication-only recovery need not initialize that device.
    if (!selection.use_all_devices && !selection.cpu_selected() && selection.device_ids.size() == 1)
    {
        result.backend = "gpu"; result.backend_set = true;
        result.device = uint64_t(selection.device_ids.front());
        if (auto const tier = selection.per_device_tier.find(int(result.device)); tier != selection.per_device_tier.end()) result.tier = tier->second;
        return;
    }
#ifdef XCHPLOT2_GROUP_COMMANDS
    std::vector<pos2gpu::GroupLocalDevice> devices;
    bool const cpu_nodes_explicit = !selection.cpu_node_ids.empty();
    auto gpu_ids = selection.device_ids;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    if (selection.use_all_devices || !gpu_ids.empty())
    {
        auto const available = pos2gpu::list_gpu_devices();
        if (selection.use_all_devices)
            for (auto const& device : available)
                if (device.auto_dispatchable) gpu_ids.push_back(int(device.id));
        std::sort(gpu_ids.begin(), gpu_ids.end());
        gpu_ids.erase(std::unique(gpu_ids.begin(), gpu_ids.end()), gpu_ids.end());
        std::set<std::string> physical;
        for (int id : gpu_ids)
        {
            auto const found = std::find_if(available.begin(), available.end(), [&](auto const& value) { return value.id == size_t(id); });
            require(found != available.end(), "selected GPU is unavailable: " + pos2gpu::device_label(id));
            require(found->physical_identity.empty() || physical.insert(found->physical_identity).second,
                "selected GPU devices share one physical allocation; select one device per physical GPU");
        }
    }
#else
    require(gpu_ids.empty(), "selected GPU is unavailable in this build");
#endif
    for (int id : gpu_ids)
    {
        auto const tier = selection.per_device_tier.find(id);
        devices.push_back({id, tier != selection.per_device_tier.end() ? tier->second :
            selection.all_gpus_tier.empty() ? result.tier : selection.all_gpus_tier, {}});
    }
    if (selection.cpu_selected())
    {
        auto const nodes = pos2gpu::host_numa_nodes();
        if (selection.use_all_cpu_nodes)
            for (auto const& node : nodes) devices.push_back({pos2gpu::cpu_device_id(node.node_id), "auto", node.cpus});
        else for (int id : selection.cpu_node_ids)
        {
            auto const found = std::find_if(nodes.begin(), nodes.end(), [&](auto const& node) { return node.node_id == id; });
            require(found != nodes.end(), "selected CPU node is unavailable: cpu" + std::to_string(id));
            devices.push_back({pos2gpu::cpu_device_id(id), "auto", found->cpus});
        }
    }
    require(!devices.empty(), "no selected local producers are available; inspect xchplot2 devices");
    require(devices.size() <= 1024, "too many selected local producers");
    bool const cpu = std::any_of(devices.begin(), devices.end(), [](auto const& value) { return pos2gpu::is_cpu_device(value.device); });
    bool const gpu = std::any_of(devices.begin(), devices.end(), [](auto const& value) { return !pos2gpu::is_cpu_device(value.device); });
    result.backend = cpu && gpu ? "auto" : cpu ? "cpu" : "gpu"; result.backend_set = true;
    if (devices.size() == 1 && !cpu_nodes_explicit)
    {
        if (gpu) { result.device = uint64_t(devices.front().device); result.tier = devices.front().tier; }
    }
    else result.local_devices = std::move(devices);
#else
    require(!selection.use_all_devices && selection.device_ids.size() <= 1 && selection.cpu_node_ids.empty() &&
        !(selection.cpu_selected() && !selection.device_ids.empty()), "multiple local producers require the installed xchplot2 command");
    result.backend = selection.cpu_selected() ? "cpu" : "gpu"; result.backend_set = true;
    if (!selection.device_ids.empty())
    {
        result.device = uint64_t(selection.device_ids.front());
        if (auto const tier = selection.per_device_tier.find(int(result.device)); tier != selection.per_device_tier.end()) result.tier = tier->second;
    }
#endif
}
void load_harvester_profile(pos2gpu::GroupOptions& result)
{
    if (result.farming_detect_local)
    {
        group_grow::ChiaDetectionOptions options;
        options.root = result.chia_root; options.executable = result.chia_executable;
        options.max_memory_bytes = std::min(result.max_ram / 8, uint64_t(256 * mib));
        if (result.farming_lookup_workers) options.lookup_workers = unsigned(*result.farming_lookup_workers);
        result.farming_profile = std::make_shared<group_grow::FarmingProfile>(group_grow::detect_farming_profile(options));
    }
    else if (!result.farming_profile_path.empty())
    {
        std::ifstream input(result.farming_profile_path, std::ios::binary);
        require(input.is_open(), "cannot open farming profile: " + result.farming_profile_path.string());
        bool const migration = result.replot || (!result.from_plan.empty() &&
            group_grow::farm::inventory_detail::is_replot_review(result.from_plan));
        if (migration)
        {
            auto profile = group_grow::read_replot_farming_profile(input, std::min(result.max_ram / 8, uint64_t(256 * mib)));
            if (profile) result.farming_profile = std::make_shared<group_grow::FarmingProfile>(std::move(*profile));
            return;
        }
        result.farming_profile = std::make_shared<group_grow::FarmingProfile>(
            group_grow::read_farming_profile(input, std::min(result.max_ram / 8, uint64_t(256 * mib))));
    }
}
void validate_harvester_options(pos2gpu::GroupOptions const& result)
{
    require(!result.farming_reader || *result.farming_reader == "stateless" || *result.farming_reader == "cached",
        "--farming-reader requires stateless or cached");
    require(!result.farming_objective || *result.farming_objective == "energy" || *result.farming_objective == "latency",
        "--farming-objective requires energy or latency");
    require(!result.farming_cpu_work_weight || *result.farming_cpu_work_weight > 0, "--farming-cpu-work-weight must be greater than zero");
    require(!result.farming_io_work_weight || *result.farming_io_work_weight > 0, "--farming-io-work-weight must be greater than zero");
    require(!result.farming_max_eligible_groups || *result.farming_max_eligible_groups > 0,
        "--farming-max-eligible-groups must be greater than zero");
    require(!result.farming_cores || (*result.farming_cores > 0 && *result.farming_cores <= 1024), "--farming-cores must be between 1 and 1024");
    require(!result.farming_lookup_workers || (*result.farming_lookup_workers > 0 && *result.farming_lookup_workers <= 1024),
        "--farming-lookup-workers must be between 1 and 1024");
    require(!result.farming_memory || *result.farming_memory > 0, "--farming-memory-bytes must be greater than zero bytes");
    require(!result.farming_filter_bits || *result.farming_filter_bits <= 13, "--farming-filter-bits must be between 0 and 13");
    require(!result.farming_future_filter_bits || *result.farming_future_filter_bits <= 13, "--farming-future-filter-bits must be between 0 and 13");
    require(!result.farming_chain_us || *result.farming_chain_us > 0, "--farming-chain-us-per-member must be greater than zero microseconds");
    require(!result.farming_deadline_ms || *result.farming_deadline_ms > 0, "--farming-deadline-ms must be greater than zero milliseconds");
}
void reference_help()
{
    std::cout <<
        "Ordinary PoS2 group construction reference runner\n"
        "Usage: pos2_pr118_grow TARGET [OPTIONS]\n"
        "      --destination FILE|DIR  additional key-derived destination; repeat for more drives\n"
        "  -t, --temp-dir DIR        temporary plotting storage; repeat for more devices\n"
        "  --destination-temp off|on|ssd|hdd  allow temporary files on destinations; default off\n"
        "                            on: any destination; ssd/hdd: only known drives of that type\n"
        "  --temp-capacity-bytes N    aggregate scratch cap per filesystem; default available space\n"
        "  --group-id HEX --memo HEX  identity for a new group (memo may be empty)\n"
        "  --group-members auto|N    directory farming layout; default auto for directories\n"
        "  --plan-only               preview directory farming layout without writing files\n"
        "  --farming-reader stateless|cached  final farming reader profile\n"
        "  --farming-objective energy|latency  expected dynamic work or response bound; default energy\n"
        "  --farming-cpu-work-weight N  positive relative CPU-active work weight; default 1\n"
        "  --farming-io-work-weight N  positive relative I/O-active work weight; default 1\n"
        "  --farming-max-eligible-groups N  maximum eligible file groups per physical resource; default 1\n"
        "  --farming-cores N         final harvester CPU budget; independent of plotting host\n"
        "  --farming-lookup-workers N  final synchronous reader concurrency\n"
        "  --farming-memory-bytes N  final reader working-memory budget\n"
        "  --farming-index-cache-bytes N  final harvester index-cache budget\n"
        "  --farming-filter-bits N    current effective farming filter\n"
        "  --farming-future-filter-bits N  future effective farming filter\n"
        "  --farming-deadline-ms N    hard modeled quality-response deadline\n"
        "  --farming-chain-us-per-member N  final per-member decode/chaining/reader CPU cost\n"
        "  --farming-index-ns-per-entry N  final per-core index-decode cost\n"
        "  -f, --farmer-pk HEX        farmer public key for key-derived destinations\n"
        "  -p, --pool-pk HEX          pool public key (choose exactly one pool source)\n"
        "  --pool-ph HEX             pool contract puzzle hash\n"
        "  -c, --pool-contract-address ADDRESS  pool contract address\n"
        "  -S, --seed HEX             base seed (required for rollover/additional destinations)\n"
        "  --k EVEN                  18..28, default 28\n"
        "  --strength N              default 2\n"
        "  --meta-group N            default 0\n"
        "  -r, --resume              continue/recover an existing destination\n"
        "  --capacity-bytes N        logical destination cap; default filesystem capacity\n"
        "  --headroom-bytes N        filesystem bytes left free, default 67108864\n"
        "  -M, --max-ram N           modeled app buffer budget in bytes, default 14000000000\n"
        "  --window-bytes N          fixed I/O window; default auto up to 16777216\n"
        "  -j, --workers N           compression workers, default available CPU count\n"
        "  --batch-members N         maximum additions per rewrite, default fits available RAM\n"
        "  --max-members N           stop at this total membership, default 65535\n"
        "  --max-groups N            per-directory rollover file limit; default fills capacity\n"
        "  --max-total-members N     per-directory total member limit; default fills capacity\n"
        "  --backend cpu|gpu|auto    producer selection (GPU needs a GPU-enabled build)\n"
        "  --device N                GPU device index, default 0\n"
        "  --tier T                  pool|plain|compact|minimal|tiny|pinned|auto, default auto\n"
        "  --member-storage raw|compact  retained member representation, default compact\n"
        "  --construction auto|incremental|staged|merged|external  assembly method, default auto\n"
        "  --checkpoint-bytes N      fixed durability target; auto up to 268435456, merged 1073741824\n"
        "  --io-overlap on|off        overlap encoding and file I/O, default on\n"
        "  --generation-overlap auto|on|off  overlap GPU/CPU work within admitted budgets, default auto\n"
        "  --cache-release on|off     release clean construction file cache, default on\n"
        "  --fixture-inputs PATH     internal raw-fixture manifest; disables CPU generation\n"
        "  --pool-broker HOST:PORT   shared authenticated network compute broker\n"
        "  --pool-owner-id ID        stable unique storage-owner identity and key namespace\n"
        "  --pool-ca FILE --pool-cert FILE --pool-key FILE  mutual TLS credentials\n"
        "  --pool-inflight N         adaptive member-task concurrency ceiling, default 64\n"
        "  --pool-timeout-ms N       bounded task deadline, default 3600000\n"
        "  --pool-socket-timeout-ms N  socket progress deadline, default 30000\n"
        "  --pool-frame-bytes N      bounded transport window, default 1048576\n"
        "  --pool-codec-ram auto|N    optional final codec partition, default auto; 0 disables\n"
        "  --pool-listen HOST:PORT --pool-advertise HOST:PORT  offer local compute to all owners\n"
        "  --pool-offer-ram N        local worker partition within --max-ram\n"
        "  --pool-offer-cpus N       local worker CPU grant shared with construction\n"
        "  --pool-offer-vram N       explicit local GPU allocation cap in bytes\n"
        "  --pool-offer-slots N      concurrent local worker tasks, default 1\n"
        "  --pool-offer-backend cpu|gpu|auto  local offered producer, default cpu\n"
        "  --help                    display this help\n"
        "Same-file methods keep construction state inside TARGET. External construction\n"
        "also requires its private checkpoint directory and temporary contributions. New destinations are\n"
        "created exclusively. Existing ordinary groups require --resume. A complete\n"
        "group may be continued with a larger cap. Rollover uses private grow-group-\n"
        "<ordinal>.gplot files, distinct key-derived identities, and index zero in each\n"
        "group. Its capacity cap covers all output files and active construction.\n"
        "File rollover resumes must reuse the same seed, keys and group parameters.\n"
        "Directory farming jobs save their seed, keys and destinations for --resume.\n"
        "The RAM budget models app buffers; physical memory qualification is separate.\n";
}
} // namespace

pos2gpu::GroupOptions pos2gpu::parse_group_options(int argc, char** argv, std::string_view command)
{
    bool const action = command == "pause" || command == "drain" || command == "resume" || command == "retire";
    require(argc > 1 || command == "plan" || command == "status" || action, "destination required; use --help");
    GroupOptions result;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    result.backend = "auto";
#endif
    result.command = command;
    result.plan_only = command == "plan";
    result.resume = command == "grow";
    result.replot_status = command == "status";
    if (action) result.farm_action = command;
    bool literal = false;
    int progress = -1;
    bool status_timeout_set = false;
    auto path = [](std::string const& value)
    { return fs::absolute(fs::path(reinterpret_cast<char8_t const*>(value.c_str()))).lexically_normal(); };
    auto destination = [&](std::string const& value)
    {
        require(!value.empty(), "empty destination path");
        // Retain the existing `--replot INVENTORY` spelling. New jobs can
        // omit the inventory; --replot-inventory selects one explicitly.
        if (result.replot && fs::is_regular_file(path(value)))
        {
            require(result.replot_inventory.empty(), "supply only one replot inventory");
            result.replot_inventory = path(value); return;
        }
        if (result.target.empty()) result.target = path(value);
        else
        {
            require(command != "grow" && command != "plot", "unexpected destination " + value);
            require(result.destinations.size() < 1023, "too many destinations");
            result.destinations.push_back(path(value));
        }
    };
    // Validate inspection's narrow input scope before options such as path
    // lists or local harvester detection can read anything beyond its source.
    bool metadata_only = false, assess_jobs = false, calibrate = false, calibration_worker = false, calibration_options = false, inventory = false, proof_worker = false, structure_worker = false, proof_options = false, proof_solves = false, replot_status = result.replot_status;
    bool status_live = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string_view const key = argv[i];
        if (key == "--") break;
        if (key == "--metadata-only") metadata_only = true;
        else if (key == "--no-metadata-only") metadata_only = false;
        else if (key == "--assess-jobs") assess_jobs = true;
        else if (key == "--no-assess-jobs") assess_jobs = false;
        else if (key == "--inventory") inventory = true;
        else if (key == "--no-inventory") inventory = false;
        else if (key == "--replot" || key == "--replot-inventory") result.replot = true;
        else if (key == "--no-replot") result.replot = false;
        else if (key == "--status") replot_status = true;
        else if (key == "--no-status") replot_status = command == "status";
        else if (key == "--live") status_live = true;
        else if (key == "--no-live") status_live = false;
        else if (key == "--calibrate") calibrate = true;
        else if (key == "--no-calibrate") calibrate = false;
        if (key == "--calibration-worker") calibration_worker = true;
        if (key == "--replot-proof-worker") proof_worker = true;
        if (key == "--replot-structure-worker") structure_worker = true;
        if (key == "--replot-proof-cpus" || key == "--replot-proof-solves") proof_options = true;
        if (key == "--replot-proof-solves") proof_solves = true;
        if (key.starts_with("--calibration-")) calibration_options = true;
        i += int(group_option_values(key));
    }
    require(!action || !replot_status, "farm control and status are separate commands");
    if (replot_status || action)
    {
        require(command == "plan" || command == "fill" || command == "status" || action, "farm status requires status, plan or fill");
        for (int i = 1; i < argc; ++i)
        {
            std::string_view const key = argv[i]; if (key == "--") break;
            if (!key.starts_with('-')) continue;
            require((replot_status && (key == "--status" || key == "--no-status" || key == "--from-plan" || key == "-F" ||
                    key == "--watch" || key == "--no-watch" || key == "--inspect" || key == "--no-inspect" ||
                    key == "--live" || key == "--no-live" || key == "--timeout-ms")) ||
                (replot_status && status_live && (key == "--compute" || key == "--compute-directory" || key == "--compute-broker" ||
                    key == "--harvester" || key == "--farm-adapter" || key == "--chia-root" || key == "--chia-executable")) ||
                (action && (key == "--destination" || key == "--policy-only" || key == "--no-policy-only" ||
                    key == "--from-plan" || key == "-F")) || key == "--manifest" ||
                key == "--max-ram" || key == "-M" || key == "--json" || key == "--no-json" ||
                key == "--quiet" || key == "--no-quiet" || key == "-q" || key == "--verbose" || key == "--no-verbose" || key == "-v",
                action ? "farm control accepts only one job selector, --destination, --policy-only and display or memory controls" :
                    "farm status accepts a job selector, display or memory controls, and explicit --live query inputs");
            auto const values = group_option_values(key);
            require(i + int(values) < argc, "missing value for " + std::string(key)); i += int(values);
        }
    }
    if (metadata_only || assess_jobs || calibrate || calibration_options || inventory || proof_worker || structure_worker || proof_options)
    {
        require(command == "plan", "saved metadata and joint assessment modes require plan");
        require(unsigned(metadata_only) + unsigned(assess_jobs) + unsigned(calibrate) + unsigned(inventory) == 1,
            "choose one of --metadata-only, --assess-jobs, --calibrate or --inventory");
        require(!calibration_options || calibrate, "calibration controls require --calibrate");
        require(!(proof_worker && structure_worker), "choose one private migration verification worker");
        require(!(proof_worker || structure_worker) || (calibrate && calibration_worker), "private migration verification requires the capped calibration worker scope");
        require(!proof_options || proof_worker || structure_worker, "proof controls require the private migration verification worker");
        require(!proof_solves || proof_worker, "proof solve controls require the private proof worker");
#ifndef XCHPLOT2_GROUP_COMMANDS
        require(!calibrate, "reader calibration requires the installed xchplot2 plan command");
#endif
        for (int i = 1; i < argc; ++i)
        {
            std::string_view const key = argv[i];
            if (key == "--") break;
            if (!key.starts_with('-')) continue;
            bool const worker_option = key == "--calibration-worker" || key == "--calibration-samples" ||
                key == "--calibration-timeout-ms" || key == "--calibration-memory-bytes" || key == "--replot-proof-worker" || key == "--replot-structure-worker" || key == "--replot-proof-cpus" || key == "--replot-proof-solves";
            bool const calibration_input = !calibration_worker && (key == "--save-profile" || key == "--harvester" ||
                key == "--chia-root" || key == "--chia-executable" || key.starts_with("--farming-"));
            require(key == "--metadata-only" || key == "--no-metadata-only" || key == "--assess-jobs" || key == "--no-assess-jobs" ||
                key == "--calibrate" || key == "--no-calibrate" || (calibrate && (worker_option || calibration_input)) ||
                (inventory && (key == "--inventory" || key == "--no-inventory" || key == "--save-plan" ||
                    key == "--paths-from" || key == "--paths0-from" || key == "--harvester" || key == "--chia-root" || key == "--chia-executable" ||
                    key == "--discover-farm" || key == "--no-discover-farm" || key == "--farm-config")) ||
                (inventory && (key == "--resource-setting" || key == "--save-profile")) ||
                (inventory && (key == "--farm-adapter" || key == "--farm-adapter-timeout-ms")) ||
                (metadata_only && (key == "--from-plan" || key == "-F" || key == "--manifest")) ||
                key == "--json" || (!calibration_worker && (key == "--max-ram" || key == "-M" || key == "--no-json" ||
                key == "--quiet" || key == "--no-quiet" || key == "-q" || key == "--verbose" || key == "--no-verbose" || key == "-v")),
                "saved metadata, joint assessment and calibration modes accept only their explicit inputs and controls");
            auto const values = group_option_values(key);
            require(i + int(values) < argc, "missing value for " + std::string(key)); i += int(values);
        }
    }
    for (int i = 1; i < argc; ++i)
    {
        std::string key = argv[i];
        if (literal) { destination(key); continue; }
        if (key == "--") { literal = true; continue; }
        if (!command.empty() && key.starts_with("--compute-")) key.replace(2, 8, "pool-");
        if (key == "--help" && command.empty()) { reference_help(); std::exit(0); }
        if (key == "--resume" || key == "-r") { result.resume = result.resume_set = true; continue; }
        if (key == "--plan-only") { result.plan_only = true; continue; }
        if (key == "--replot" || key == "--no-replot") continue;
        if (key == "--discover-farm" || key == "--no-discover-farm")
        { result.discover_farm = key == "--discover-farm"; result.discover_farm_set = true; continue; }
        if (key == "--metadata-only") { result.metadata_only = true; continue; }
        if (key == "--no-metadata-only") { result.metadata_only = false; continue; }
        if (key == "--assess-jobs") { result.assess_jobs = true; continue; }
        if (key == "--no-assess-jobs") { result.assess_jobs = false; continue; }
        if (key == "--inventory") { result.inventory = true; continue; }
        if (key == "--no-inventory") { result.inventory = false; continue; }
        if (key == "--status") { result.replot_status = true; continue; }
        if (key == "--no-status") { result.replot_status = command == "status"; continue; }
        if (key == "--watch") { result.status_watch = true; continue; }
        if (key == "--no-watch") { result.status_watch = false; continue; }
        if (key == "--inspect") { result.status_inspect = true; continue; }
        if (key == "--no-inspect") { result.status_inspect = false; continue; }
        if (key == "--live") { result.status_live = true; continue; }
        if (key == "--no-live") { result.status_live = false; continue; }
        if (key == "--revalidate") { result.replot_revalidate = true; continue; }
        if (key == "--no-revalidate") { result.replot_revalidate = false; continue; }
        if (key == "--calibrate") { result.calibrate = true; continue; }
        if (key == "--no-calibrate") { result.calibrate = false; continue; }
        if (key == "--calibration-worker") { result.calibration_worker = true; continue; }
        if (key == "--replot-proof-worker") { result.replot_proof_worker = true; continue; }
        if (key == "--replot-structure-worker") { result.replot_structure_worker = true; continue; }
        if (key == "--extend") { result.extend = true; continue; }
        if (key == "--update") { result.update = true; continue; }
        if (key == "--cancel-update") { result.cancel_update = true; continue; }
        if (key == "--policy-only") { result.policy_only = true; continue; }
        if (key == "--upgrade-job") { result.upgrade_job = true; continue; }
        if (key == "--allow-missing") { result.allow_missing = true; continue; }
        if (!command.empty())
        {
            if (key == "--no-resume") { result.resume = command == "grow"; result.resume_set = true; continue; }
            if (key == "--no-plan-only") { result.plan_only = command == "plan"; continue; }
            if (key == "--no-extend") { result.extend = false; continue; }
            if (key == "--no-update") { result.update = false; continue; }
            if (key == "--no-cancel-update") { result.cancel_update = false; continue; }
            if (key == "--no-policy-only") { result.policy_only = false; continue; }
            if (key == "--no-upgrade-job") { result.upgrade_job = false; continue; }
            if (key == "--no-allow-missing") { result.allow_missing = false; continue; }
            if (key == "--json") { result.json = true; continue; }
            if (key == "--no-json") { result.json = false; continue; }
            if (key == "--quiet" || key == "-q") { result.quiet = true; continue; }
            if (key == "--no-quiet") { result.quiet = false; continue; }
            if (key == "--verbose" || key == "-v") { result.verbose = true; continue; }
            if (key == "--no-verbose") { result.verbose = false; continue; }
            if (key == "--progress") { progress = 1; continue; }
            if (key == "--no-progress") { progress = 0; continue; }
            require(key != "--fixture-inputs", "fixture input is only available in the reference test runner");
        }
        if (key.starts_with("-"))
        {
            if (group_path_list_option(key))
            {
                auto& paths = key == "--temp-dirs" ? result.temporary :
                    key == "--farm-config" ? result.farm_configs : result.excluded_destinations;
                size_t const before = paths.size();
                while (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with('-'))
                {
                    std::string const value = argv[++i];
                    require(!value.empty() && paths.size() < 1024, "invalid path list for " + key);
                    paths.push_back(path(value));
                }
                require(paths.size() > before, key + " requires one or more paths (use ./ for names starting with '-'); end the list with another option or --");
                continue;
            }
            require(i + 1 < argc, "missing value for " + key);
            std::string const value = argv[++i];
            if (key == "--group-id") result.group = value;
            else if (key == "--alternative")
            {
                require(!value.empty(), "--alternative requires explicit names");
                result.farming_alternative = value;
            }
            else if (key == "--manifest") result.manifest = path(value);
            else if (key == "--destination" && !command.empty())
            {
                require(!value.empty() && result.action_destinations.size() < 1024, "invalid destination selector");
                result.action_destinations.push_back(path(value));
            }
            else if (key == "--save-plan") result.save_plan = path(value);
            else if (key == "--from-plan" || key == "-F") result.from_plan = path(value);
            else if (key == "--replot-inventory") result.replot_inventory = path(value);
            else if (key == "--replot-verification") result.replot_verification = value;
            else if (key == "--replot-confirmation") result.replot_confirmation = value;
            else if (key == "--replot-proof-first-groups") result.replot_proof_first_groups = number(value.c_str(), argv[i - 1]);
            else if (key == "--replot-proof-every-groups") result.replot_proof_every_groups = number(value.c_str(), argv[i - 1]);
            else if (key == "--farm-adapter") { result.farm_adapter = value; result.farm_adapter_set = true; }
            else if (key == "--farm-adapter-timeout-ms")
            { result.farm_adapter_timeout_ms = number(value.c_str(), argv[i - 1]); result.farm_adapter_timeout_set = true; }
            else if (key == "--reclaim-before-bytes")
            { result.reclaim_before_bytes = number(value.c_str(), argv[i - 1]); result.reclaim_before_set = true; }
            else if (key == "--replot-source-root" || key == "--replot-exclude-root")
            {
                require(!value.empty(), "empty replot source root selector");
                auto& roots = key == "--replot-source-root" ? result.replot_source_roots : result.replot_exclude_roots;
                require(roots.size() < 1024, "too many replot source root selectors");
                roots.push_back(path(value));
            }
            else if (key == "--replot-max-source-files" || key == "--replot-max-source-bytes")
            {
                auto& limit = key == "--replot-max-source-files" ? result.replot_max_source_files : result.replot_max_source_bytes;
                limit = number(value.c_str(), argv[i - 1]); require(limit, key + " must be positive");
            }
            else if (key == "--save-profile") result.save_profile = path(value);
            else if (key == "--resource-setting")
            {
                require(!value.empty() && value.size() <= 4096 && result.farming_resource_settings.size() < 4096,
                    "invalid resource setting or too many resource settings");
                result.farming_resource_settings.push_back(value);
            }
            else if (key == "--timeout-ms")
            { result.status_timeout_ms = number(value.c_str(), argv[i - 1]); status_timeout_set = true; }
            else if (key == "--calibration-samples") result.calibration_samples = number(value.c_str(), argv[i - 1]);
            else if (key == "--replot-proof-cpus") result.replot_proof_cpus = number(value.c_str(), argv[i - 1]);
            else if (key == "--replot-proof-solves") result.replot_proof_solves = number(value.c_str(), argv[i - 1]);
            else if (key == "--calibration-timeout-ms") result.calibration_timeout_ms = number(value.c_str(), argv[i - 1]);
            else if (key == "--calibration-memory-bytes") result.calibration_memory_bytes = number(value.c_str(), argv[i - 1]);
            else if (key == "--harvester")
            {
                require(!value.empty(), "--harvester requires local or a profile file");
                result.farming_detect_local = value == "local";
                result.farming_profile_path = result.farming_detect_local ? fs::path{} : path(value);
                result.farming_profile_set = true;
            }
            else if (key == "--chia-root") { result.chia_root = path(value); result.chia_root_set = true; }
            else if (key == "--chia-executable") { result.chia_executable = path(value); result.chia_executable_set = true; }
            else if (key == "--relocate")
            {
                require(i + 1 < argc, "--relocate requires OLD and NEW paths");
                result.relocations.emplace_back(path(value), path(argv[++i]));
            }
            else if (key == "--paths-from" || key == "--paths0-from")
            {
                std::ifstream file;
                if (value != "-") file.open(path(value), std::ios::binary);
                require(value == "-" || file.is_open(), "cannot open destination list: " + value);
                auto& input = value == "-" ? std::cin : file;
                bool const nul = key == "--paths0-from";
                size_t count = 0;
                for (std::string line; std::getline(input, line, nul ? '\0' : '\n');)
                {
                    if (!nul && !line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    require(line.find('\0') == line.npos, "NUL in a line-delimited destination list; use --paths0-from");
                    destination(line); ++count;
                }
                require(input.eof() && !input.bad(), "cannot read destination list: " + value);
                require(count, "destination list is empty: " + value);
            }
            else if (key == "--discover-drives" || key == "--mounts-under")
            {
                require(!value.empty() && result.mounts_under.size() < 1024, "invalid mount discovery root");
                result.mounts_under.push_back(path(value));
            }
            else if (key == "--compute") result.compute_mode = value;
            else if (key == "--pool-directory") result.pool_directory = value;
            else if (key == "--pool-local")
            {
                require(value == "on" || value == "off", "--compute-local requires on or off");
                result.automatic_local_compute = value == "on";
            }
            else if (key == "--memo") result.memo = value;
            else if (key == "--farmer-pk" || key == "-f") result.farmer = value;
            else if (key == "--pool-pk" || key == "-p") result.pool_pk = value;
            else if (key == "--pool-ph") result.pool_ph = value;
            else if (key == "--pool-contract-address" || key == "-c") result.pool_address = value;
            else if (key == "--seed" || key == "-S") result.seed = value;
            else if (key == "--fixture-inputs") result.fixtures = fs::path(reinterpret_cast<char8_t const*>(value.c_str()));
            else if (key == "--pool-broker") result.pool_broker = value;
            else if (key == "--pool-owner-id") { result.pool_owner = value; result.pool_owner_explicit = true; }
            else if (key == "--pool-ca") result.pool_ca = value;
            else if (key == "--pool-cert") result.pool_cert = value;
            else if (key == "--pool-key") result.pool_key = value;
            else if (key == "--pool-listen") result.pool_listen = value;
            else if (key == "--pool-advertise") result.pool_advertise = value;
            else if (key == "--pool-offer-backend") { result.pool_offer_backend = value; result.pool_offer_backend_set = true; }
            else if (key == "--pool-offer-ram") result.pool_offer_ram = result.pool_offer_ram_limit = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-offer-cpus") result.pool_offer_cpus = result.pool_offer_cpu_limit = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-offer-vram") result.pool_offer_vram = result.pool_offer_vram_limit = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-offer-slots") result.pool_offer_slots = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-inflight") result.pool_inflight = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-timeout-ms") result.pool_timeout_ms = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-socket-timeout-ms") result.pool_socket_timeout_ms = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-frame-bytes") result.pool_frame_bytes = number(value.c_str(), argv[i - 1]);
            else if (key == "--pool-codec-ram")
            { result.pool_codec_auto = value == "auto"; if (!result.pool_codec_auto) result.pool_codec_ram = number(value.c_str(), argv[i - 1]); }
            else if (key == "--temp-dir" || key == "-t")
            {
                require(result.temporary.size() < 1024, "too many temporary directories");
                result.temporary.push_back(fs::absolute(fs::path(reinterpret_cast<char8_t const*>(value.c_str()))).lexically_normal());
            }
            else if (key == "--temp-capacity-bytes")
            {
                result.temporary_capacity = number(value.c_str(), argv[i - 1]);
                require(result.temporary_capacity > 0, "--temp-capacity-bytes must be positive");
            }
            else if (key == "--destination-temp") { result.destination_temp = value; result.destination_temp_set = true; }
            else if (key == "--destination" && command.empty())
            {
                require(result.destinations.size() < 1023, "too many destinations");
                result.destinations.push_back(fs::absolute(fs::path(reinterpret_cast<char8_t const*>(value.c_str()))).lexically_normal());
            }
            else if (key == "--k" || key == "-k") { result.k = number(value.c_str(), argv[i - 1]); result.k_set = true; }
            else if (key == "--strength" || key == "-s") { result.strength = number(value.c_str(), argv[i - 1]); result.strength_set = true; }
            else if (key == "--meta-group" || key == "-g") { result.meta = number(value.c_str(), argv[i - 1]); result.meta_set = true; }
            else if (key == "--capacity-bytes")
            {
                result.capacity = number(value.c_str(), argv[i - 1]);
                result.capacity_set = true;
                require(result.capacity > 0, "--capacity-bytes must be positive");
            }
            else if (key == "--leave-free" || key == "--headroom-bytes") { result.headroom = number(value.c_str(), argv[i - 1]); result.headroom_set = true; }
            else if (key == "--max-ram" || key == "-M")
            { result.max_ram = result.max_ram_limit = number(value.c_str(), argv[i - 1]); result.max_ram_set = true; }
            else if (key == "--window-bytes") { result.window = number(value.c_str(), argv[i - 1]); result.window_set = true; }
            else if (key == "--workers" || key == "-j")
            { result.workers = number(value.c_str(), argv[i - 1]); result.workers_set = true; }
            else if (key == "--batch-members") result.batch = number(value.c_str(), argv[i - 1]);
            else if (key == "--max-members") result.max_members = number(value.c_str(), argv[i - 1]);
            else if (key == "--group-members")
            {
                result.group_members_set = true;
                result.group_auto = value == "auto";
                result.group_members = 0;
                if (!result.group_auto)
                {
                    result.group_members = number(value.c_str(), argv[i - 1]);
                    require(result.group_members > 0, "--group-members must be positive");
                }
            }
            else if (key == "--farming-reader") result.farming_reader = value;
            else if (key == "--farming-objective") result.farming_objective = value;
            else if (key == "--farming-cpu-work-weight") result.farming_cpu_work_weight = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-io-work-weight") result.farming_io_work_weight = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-max-eligible-groups") result.farming_max_eligible_groups = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-lookup-workers") result.farming_lookup_workers = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-memory-bytes") result.farming_memory = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-cores") result.farming_cores = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-index-cache-bytes") result.farming_index_cache = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-filter-bits") result.farming_filter_bits = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-future-filter-bits") result.farming_future_filter_bits = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-chain-us-per-member") result.farming_chain_us = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-index-ns-per-entry") result.farming_index_ns = number(value.c_str(), argv[i - 1]);
            else if (key == "--farming-deadline-ms") result.farming_deadline_ms = number(value.c_str(), argv[i - 1]);
            else if (key == "--max-groups") { result.max_groups = number(value.c_str(), argv[i - 1]); result.max_groups_set = true; }
            else if (key == "--max-total-members") { result.max_total_members = number(value.c_str(), argv[i - 1]); result.max_total_members_set = true; }
            else if (key == "--checkpoint-bytes") { result.checkpoint_bytes = number(value.c_str(), argv[i - 1]); result.checkpoint_set = true; }
            else if (key == "--backend") { result.backend = value; result.backend_set = true; result.devices.clear(); }
            else if (key == "--device") { result.device = number(value.c_str(), argv[i - 1]); result.device_set = true; result.devices.clear(); }
            else if (key == "--devices" || key == "-d")
            {
                BatchOptions selection;
                require(parse_devices_arg(value, selection), "invalid --devices selection");
                result.devices = value;
                result.device_set = true;
            }
            else if (key == "--tier") result.tier = value;
            else if (key == "--member-storage") result.member_storage = value;
            else if (key == "--construction") result.construction = value;
            else if (key == "--io-overlap")
            {
                require(value == "on" || value == "off", "--io-overlap requires on or off");
                result.io_overlap = value == "on";
            }
            else if (key == "--generation-overlap")
            {
                require(value == "auto" || value == "on" || value == "off",
                        "--generation-overlap requires auto, on or off");
                result.generation_overlap = value != "off";
                result.generation_overlap_auto = value == "auto";
            }
            else if (key == "--cache-release")
            {
                require(value == "on" || value == "off", "--cache-release requires on or off");
                result.cache_release = value == "on";
            }
            else throw std::runtime_error("unknown option " + key);
        }
        else
        {
            destination(key);
        }
    }
    bool const farm_command = command == "fill" || command == "plan";
    bool const migration_resources = result.replot || !result.replot_inventory.empty() ||
        (!result.from_plan.empty() && group_grow::farm::inventory_detail::is_replot_review(result.from_plan));
    if (migration_resources && !result.max_ram_set)
    {
        auto const available = group_grow::profile_detail::available_memory();
        require(available >= 16 * mib, "cannot determine an available migration RAM budget; retry when memory is available or set --max-host-ram");
        result.max_ram = available;
    }
    require((!result.status_watch && !result.status_inspect) || result.replot_status, "--watch and --inspect require status");
    require(!result.status_live || result.replot_status, "--live requires status");
    require(!status_timeout_set || (result.replot_status && result.status_live), "--timeout-ms requires status --live");
    require(result.status_timeout_ms >= 100 && result.status_timeout_ms <= 60000, "status timeout must be in 100..60000 ms");
    require(result.farming_resource_settings.empty() || (farm_command && (result.inventory || result.replot) &&
        result.from_plan.empty() && !result.replot_status && !result.metadata_only && !result.assess_jobs && !result.calibrate),
        "--resource-setting requires plan --inventory or a fresh --replot job");
    require(result.action_destinations.empty() || action, "--destination requires pause, drain, resume or retire");
    require(!result.policy_only || action, "--policy-only requires pause, drain, resume or retire");
    bool const guided = !result.farming_alternative.empty() || result.upgrade_job;
    require(!guided || (command == "plan" && result.from_plan.empty() && !result.replot_status && !result.metadata_only &&
        !result.assess_jobs && !result.inventory && !result.calibrate && !result.replot && !result.replot_revalidate &&
        !result.policy_only && !result.update && !result.extend && !result.cancel_update && result.relocations.empty() &&
        (!result.target.empty() || !result.manifest.empty() || !result.mounts_under.empty())),
        "--alternative and --upgrade-job require an ordinary plan with destinations or --manifest");
    require(!result.upgrade_job || result.farming_alternative.empty(), "--alternative and --upgrade-job are mutually exclusive");
    if (!result.farming_alternative.empty())
    {
        std::set<std::string> names;
        size_t first = 0;
        do
        {
            auto const last = result.farming_alternative.find(',', first);
            auto const name = result.farming_alternative.substr(first, last == std::string::npos ? last : last - first);
            require((name == "memory" || name == "eligible-groups" || name == "deadline" || name == "cpu-cores" ||
                name == "lookup-workers" || name == "capacity") && names.insert(name).second,
                "--alternative requires distinct memory, eligible-groups, deadline, cpu-cores, lookup-workers or capacity names");
            if (last == std::string::npos) break;
            first = last + 1;
        } while (true);
        require(!names.contains("capacity") || names.size() == 1, "capacity must be the only --alternative choice");
    }
    require(result.mounts_under.empty() || (farm_command && result.from_plan.empty() && !result.inventory &&
        !result.metadata_only && !result.assess_jobs && !result.calibrate && !result.replot_status),
        "--discover-drives selects plan or fill destinations; omit saved previews and inspection modes");
    require((!result.discover_farm_set && result.farm_configs.empty() && result.excluded_destinations.empty()) ||
        (farm_command && result.from_plan.empty() && result.manifest.empty() && !result.metadata_only &&
            !result.assess_jobs && !result.calibrate && !result.replot_status && !result.extend && !result.update && !result.resume),
        "directory discovery and exclusions require a new plan or fill job");
    if (result.replot_status || action)
    {
        require(result.compute_mode == "auto" || result.compute_mode == "off", "--compute requires auto or off");
        require(result.farm_adapter == "chia" || result.farm_adapter == "gigahorse" || result.farm_adapter == "drchia",
            "--farm-adapter requires chia, gigahorse or drchia");
        require(result.destinations.empty() && (result.target.empty() || (result.from_plan.empty() && result.manifest.empty())) &&
            (result.from_plan.empty() || result.manifest.empty()), action ? "farm control requires one job selector" :
                "farm status requires one destination or saved plan");
        if (!result.target.empty() && fs::is_regular_file(result.target))
        {
            auto const magic = group_grow::farm::inventory_detail::saved_plan_header(result.target);
            if (magic.starts_with("ordinary-group-farming-v")) result.manifest = std::move(result.target);
            else
            {
                require(!action || group_grow::farm::inventory_detail::is_replot_review(result.target),
                    "farm control requires an existing job destination, manifest or migration review");
                result.from_plan = std::move(result.target);
            }
            result.target.clear();
        }
        require(result.max_ram >= 16 * mib, command.empty() ? "--max-ram must be at least 16 MiB (16777216 bytes)" :
            "--max-host-ram must be at least 16 MiB (16M)");
        require(!result.quiet || !result.verbose, "--quiet and --verbose are mutually exclusive");
        result.plan_only = result.replot_status;
        result.resume = result.directory_farm = true; return result;
    }
    require(result.farm_adapter == "chia" || result.farm_adapter == "gigahorse" || result.farm_adapter == "drchia",
        "--farm-adapter requires chia, gigahorse or drchia");
    require(result.farm_adapter_timeout_ms && result.farm_adapter_timeout_ms <= 3600000,
        "--farm-adapter-timeout-ms must be in 1..3600000");
    if (result.inventory)
    {
        require(command == "plan" && !result.replot && (!result.target.empty() || result.discover_farm || !result.farm_configs.empty()),
            "plan --inventory requires farm roots or --discover-farm");
        require(result.max_ram >= 16 * mib, command.empty() ? "--max-ram must be at least 16 MiB (16777216 bytes)" :
            "--max-host-ram must be at least 16 MiB (16M)");
        require(!result.quiet || !result.verbose, "--quiet and --verbose are mutually exclusive");
        result.directory_farm = true;
        return result;
    }
    require(!result.replot || (farm_command &&
        result.from_plan.empty() && result.manifest.empty() && !result.resume && !result.extend && result.relocations.empty() && !result.allow_missing),
        "--replot requires plan or fill without an existing job selector");
    require(!result.reclaim_before_set || result.replot, "--reclaim-before-bytes belongs in a fresh --replot job");
    require((result.replot_source_roots.empty() && result.replot_exclude_roots.empty() &&
        !result.replot_max_source_files && !result.replot_max_source_bytes) || result.replot,
        "replot source selection belongs in a fresh --replot review");
    if (result.metadata_only || result.assess_jobs || result.calibrate)
    {
        if (result.assess_jobs || result.calibrate)
        {
            require(!result.target.empty(), "supply the saved PLAN files or calibration plot files explicitly");
            auto& inputs = result.assess_jobs ? result.assessment_plans : result.calibration_plots;
            inputs.reserve(result.destinations.size() + 1);
            inputs.push_back(std::move(result.target)); result.target.clear();
            for (auto& path : result.destinations) inputs.push_back(std::move(path));
            result.destinations.clear();
        }
        else if (!result.target.empty())
        {
            require(result.from_plan.empty() && result.manifest.empty() && result.destinations.empty(),
                "--metadata-only requires one saved PLAN or --manifest FILE");
            result.from_plan = std::move(result.target); result.target.clear();
        }
        require(result.assess_jobs || result.calibrate || (result.from_plan.empty() != result.manifest.empty()),
            "--metadata-only requires one saved PLAN or --manifest FILE");
        require(!result.quiet || !result.verbose, "--quiet and --verbose are mutually exclusive");
        require(result.max_ram >= 16 * mib, command.empty() ? "--max-ram must be at least 16 MiB (16777216 bytes)" :
            "--max-host-ram must be at least 16 MiB (16M)");
        result.directory_farm = true;
        if (result.calibrate)
        {
            require(!result.replot_structure_worker || (result.replot_proof_cpus == 1 && result.calibration_samples == 1),
                "structural worker requires one CPU and one structural inspection");
            require(result.calibration_plots.size() <= 64 && result.calibration_samples && result.calibration_samples <= 1024 &&
                result.calibration_timeout_ms && result.calibration_timeout_ms <= UINT_MAX && result.calibration_memory_bytes,
                "invalid calibration plot count, sample count, timeout or worker-memory limit");
            require(!result.calibration_worker || (result.calibration_plots.size() == 1 && result.json),
                "calibration worker requires exactly one explicit plot and --json");
            require(result.calibration_worker || result.json || !result.save_profile.empty(),
                "calibration requires --json or --save-profile FILE");
            require(result.farming_detect_local || (result.chia_root.empty() && result.chia_executable.empty()),
                "--chia-root and --chia-executable require --harvester local");
            if (!result.calibration_worker)
            { validate_harvester_options(result); load_harvester_profile(result); }
        }
        return result;
    }
    if (farm_command && !result.target.empty() && fs::is_regular_file(result.target))
    {
        require(!guided, "--alternative and --upgrade-job require destinations or --manifest, without a saved preview");
        require(result.from_plan.empty() && result.destinations.empty(),
            "supply a saved plan on its own, without destination directories");
        result.from_plan = std::move(result.target); result.target.clear();
    }
    bool const verification_options = result.replot_verification || result.replot_proof_first_groups || result.replot_proof_every_groups;
    require(!result.replot_confirmation || (result.replot &&
        (*result.replot_confirmation == "directory" || *result.replot_confirmation == "file")),
        "--replot-confirmation requires directory or file on a fresh --replot job");
    require(!verification_options || (farm_command && (result.replot || result.replot_revalidate)),
        "migration verification policy requires a fresh --replot review or saved --revalidate successor");
    require(!result.replot_verification || *result.replot_verification == "structural" || *result.replot_verification == "sampled" ||
        *result.replot_verification == "strict", "--replot-verification requires structural, sampled or strict");
    require(!(result.replot_proof_first_groups || result.replot_proof_every_groups) || result.replot_verification == "sampled",
        "migration proof group schedules require --replot-verification sampled");
    require(result.replot_verification != "sampled" || result.replot_proof_first_groups.value_or(1) || result.replot_proof_every_groups.value_or(0),
        "sampled migration verification requires a nonzero first or periodic group count");
    require(result.farm_adapter == "chia" || result.replot || !result.from_plan.empty(),
        "native farm adapters require inventory, replot review or a saved migration plan");
    require(!result.farm_adapter_timeout_set || result.replot ||
        (farm_command && !result.from_plan.empty() && group_grow::farm::inventory_detail::is_replot_review(result.from_plan)),
        "--farm-adapter-timeout-ms requires inventory, replot review or a saved migration plan");
    if (result.extend || result.update || result.cancel_update || !result.relocations.empty()) result.resume = true;
    require(!result.cancel_update || (command == "fill" && !result.update && !result.extend && result.from_plan.empty() &&
        result.relocations.empty() && !result.plan_only && !result.allow_missing),
        "--cancel-update requires fill with one existing job, without another action");
    require(!result.target.empty() || !result.mounts_under.empty() || result.discover_farm || !result.farm_configs.empty() || result.replot || (farm_command &&
        (result.plan_only || result.resume || !result.manifest.empty() || !result.from_plan.empty())), "destination required");
    require((!result.extend && !result.update && !result.allow_missing && result.manifest.empty() && result.relocations.empty()) || farm_command,
        "saved farming job options require plan or fill");
    require(!result.allow_missing || (result.resume && !result.extend && !result.update && result.relocations.empty()),
        "--allow-missing requires --resume without --extend, --update or --relocate");
    require((result.save_plan.empty() && result.from_plan.empty()) || farm_command,
        "saved previews require plan or fill");
    require(result.save_plan.empty() || result.plan_only,
        "--save-plan requires plan or fill --plan-only");
    require(result.save_plan.empty() || result.from_plan.empty() || result.replot_revalidate,
        "--save-plan and --from-plan cannot be combined");
    require(!result.replot_revalidate || (command == "plan" && !result.from_plan.empty() && !result.save_plan.empty() &&
        !result.farming_profile_path.empty() && !result.farming_detect_local &&
        group_grow::farm::inventory_detail::is_replot_review(result.from_plan)),
        "revalidation requires plan --from-plan MIGRATION --harvester UPDATED_PROFILE --save-plan NEW");
    require(result.from_plan.empty() || (result.target.empty() && result.destinations.empty() &&
        result.manifest.empty() && result.relocations.empty() && !result.resume && !result.extend && !result.update && !result.allow_missing),
        "--from-plan supplies the saved destinations and action; omit other job selectors");
    require(result.save_profile.empty() || (command == "plan" && result.plan_only &&
        (!result.replot || result.save_plan.empty()) && result.from_plan.empty() && result.manifest.empty() &&
        result.relocations.empty() && !result.resume && !result.extend && !result.update && !result.allow_missing),
        "export a harvester profile with plan --save-profile FILE, without an existing job; migration setup cannot also save a reviewed plan");
    require((!result.farming_detect_local && result.farming_profile_path.empty() &&
        result.chia_root.empty() && result.chia_executable.empty()) || farm_command,
        "harvester profiles require plan or fill");
    require(result.farming_detect_local || result.replot || !result.from_plan.empty() ||
        (result.chia_root.empty() && result.chia_executable.empty()),
        "--chia-root and --chia-executable require --harvester local");
    require(result.batch > 0 && result.batch <= 65535 && result.max_members > 0 &&
                result.max_members <= 65535, "member caps must be in 1..65535");
    require(result.workers <= 1024, "compression worker cap exceeds 1024");
    require(result.group_members <= 65535 && result.max_groups && result.max_total_members,
            "rollover group membership must fit in 1..65535 and limits must be positive");
    if (!result.target.empty()) result.target = fs::absolute(result.target).lexically_normal();
    bool const farming_options = result.farming_reader || result.farming_objective || result.farming_cpu_work_weight ||
        result.farming_io_work_weight || result.farming_max_eligible_groups || result.farming_cores || result.farming_lookup_workers ||
        result.farming_memory || result.farming_index_cache || result.farming_filter_bits || result.farming_future_filter_bits ||
        result.farming_deadline_ms || result.farming_chain_us || result.farming_index_ns;
    bool const directory_target = fs::is_directory(result.target);
    if (command == "grow")
        require(!directory_target && fs::exists(result.target), "grow requires an existing group file; use plot to create files");
    require(command.empty() || !result.quiet || !result.verbose, "--quiet and --verbose are mutually exclusive");
    // Match plot/batch: an explicit progress choice wins; otherwise use stderr
    // TTY detection, with quiet suppressing the automatic progress display.
    if (progress >= 0) result.progress = progress != 0;
#ifdef _WIN32
    else result.progress = !result.quiet && ::_isatty(::_fileno(stderr)) != 0;
#else
    else result.progress = !result.quiet && ::isatty(::fileno(stderr)) != 0;
#endif
    bool saved_farm = false;
    if (result.resume && directory_target)
    {
        group_grow::FarmingMetadataDirectory metadata(fs::weakly_canonical(result.target));
        saved_farm = fs::exists(metadata.path / "grow-farm.plan") || fs::exists(metadata.path / "grow-farm.plan.next");
    }
    result.directory_farm = farm_command || (directory_target &&
        (!result.group_members || !result.destinations.empty() || result.plan_only || farming_options || saved_farm));
    require(!result.group_auto || result.directory_farm, "--group-members auto requires a directory target");
    require(!result.group_members || !result.group_auto, "choose one --group-members value");
    require(!(result.group_members || result.directory_farm) || (result.max_members == 65535 && !result.group && !result.memo &&
                result.fixtures.empty()),
            "--group-members uses key-derived identities and --max-total-members, not single-file options");
    require(result.group_members || result.directory_farm || (result.max_groups == UINT64_MAX &&
                result.max_total_members == UINT64_MAX), "rollover limits require --group-members");
    require(result.destinations.empty() || ((!result.group_members || result.directory_farm) && result.fixtures.empty() && !result.group && !result.memo),
            "multiple destinations require matching file or directory targets and key-derived identities");
    require(!result.plan_only || result.directory_farm, "--plan-only requires a directory target");
    require(!farming_options || result.directory_farm, "farming profile options require automatic directory planning");
    validate_harvester_options(result);
    require(result.backend == "cpu" || result.backend == "gpu" || result.backend == "auto",
            "--backend requires cpu, gpu or auto");
    require(result.tier == "auto" || result.tier == "pool" || result.tier == "plain" || result.tier == "compact" ||
                result.tier == "minimal" || result.tier == "tiny" || result.tier == "pinned",
            "unknown GPU tier");
    require(result.device <= INT_MAX, "--device must be between 0 and 2147483647");
    require(result.checkpoint_bytes > 0, "--checkpoint-bytes must be greater than zero bytes");
    require(result.member_storage == "raw" || result.member_storage == "compact",
            "--member-storage requires raw or compact");
    require(result.construction == "auto" || result.construction == "incremental" ||
            result.construction == "staged" || result.construction == "merged" || result.construction == "external",
            "--construction requires auto, incremental, staged, merged or external");
    require(result.destination_temp == "off" || result.destination_temp == "on" ||
        result.destination_temp == "ssd" || result.destination_temp == "hdd",
        "--destination-temp requires off, on, ssd or hdd");
    require(!result.temporary_capacity || !result.temporary.empty() || result.destination_temp != "off",
            "--temp-capacity-bytes requires --temp-dir or --destination-temp");
    require(result.plan_only || result.construction != "external" || result.resume || !result.temporary.empty() || result.destination_temp != "off",
            "fresh external construction requires --temp-dir or --destination-temp");
    require(result.fixtures.empty() || result.backend != "gpu", "fixtures cannot use a GPU producer");
    require(result.window > 0, "--window-bytes must be greater than zero bytes");
    require(result.max_ram >= 16 * mib, command.empty() ? "--max-ram must be at least 16 MiB (16777216 bytes)" :
        "--max-host-ram must be at least 16 MiB (16M)");
    require(!result.k || *result.k <= 255, "invalid k");
    require(!result.strength || *result.strength <= 255, "invalid strength");
    require(!result.meta || *result.meta <= 255, "invalid meta group");
    load_harvester_profile(result);
    resolve_group_devices(result);
    require(result.local_devices.empty() || (!result.replot &&
        (result.from_plan.empty() || !group_grow::farm::inventory_detail::is_replot_review(result.from_plan))),
        "migration uses its reviewed single-producer resources; select --backend cpu or --devices gpuN for migration");
#ifdef XCHPLOT2_GROUP_COMMANDS
    if (result.save_profile.empty() || !result.local_devices.empty()) prepare_group_compute(result);
#endif
    bool const pool_options = !result.pool_owner.empty() || !result.pool_ca.empty() || !result.pool_cert.empty() ||
        !result.pool_key.empty() || !result.pool_listen.empty() || !result.pool_advertise.empty() ||
        result.pool_offer_ram || result.pool_offer_cpus || result.pool_offer_vram || !result.pool_codec_auto;
    require(!pool_options || !result.pool_broker.empty() || result.local_compute, command.empty() ? "network compute options require --pool-broker" : "network compute options require --compute-broker");
    if (!result.pool_broker.empty() || result.local_compute)
    {
        if (!result.backend_set) result.backend = "auto";
#if !defined(GROW_POOL_NETWORK) || !GROW_POOL_NETWORK
        require(false, "network compute is unavailable in this build");
#endif
        require(result.local_compute || (!result.pool_owner.empty() && result.pool_owner.size() <= 128 &&
            !result.pool_ca.empty() && !result.pool_cert.empty() && !result.pool_key.empty()),
            command.empty() ? "network compute requires a stable --pool-owner-id and mutual TLS credentials" : "network compute requires a stable --compute-owner-id and mutual TLS credentials");
        require(result.fixtures.empty(), "network compute cannot use fixture inputs");
        require(result.pool_inflight && result.pool_inflight <= 1024 && result.pool_timeout_ms &&
            result.pool_timeout_ms <= UINT_MAX && result.pool_socket_timeout_ms && result.pool_socket_timeout_ms <= UINT_MAX &&
            result.pool_frame_bytes >= 4096 && result.pool_frame_bytes <= 16 * mib,
            "invalid network compute concurrency, deadline or transport window");
        require(result.window <= UINT32_MAX - result.pool_frame_bytes,
            "network codec source window exceeds the portable task bound");
        bool const offer = !result.pool_listen.empty() || !result.pool_advertise.empty() ||
            result.pool_offer_ram || result.pool_offer_cpus || result.pool_offer_vram;
        require(!offer || ((result.local_compute || result.automatic_compute || (!result.pool_listen.empty() && !result.pool_advertise.empty())) &&
            result.pool_offer_ram && result.pool_offer_cpus),
            "offering local compute requires listen, advertise, RAM and CPU grants");
        unsigned const workers = result.workers ? unsigned(result.workers) : std::max(1u, std::thread::hardware_concurrency());
        require(!offer || (result.pool_offer_cpus && result.pool_offer_cpus <= 1024 && result.pool_offer_slots && result.pool_offer_slots <= 1024),
            "local compute offer exceeds CPU or slot bounds");
        bool const shared = workers == 1 ? result.pool_offer_cpus == 1 : result.pool_offer_cpus < workers;
        bool const migration = offer && !shared && (result.replot || (farm_command && !result.from_plan.empty() &&
            group_grow::farm::inventory_detail::is_replot_review(result.from_plan)));
        require(!offer || shared || migration,
            "local compute offer must leave CPU capacity for receiving and storage");
        require(result.pool_offer_backend == "cpu" || result.pool_offer_backend == "gpu" || result.pool_offer_backend == "auto",
            command.empty() ? "--pool-offer-backend requires cpu, gpu or auto" : "--compute-offer-backend requires cpu, gpu or auto");
        require(!offer || result.local_compute || result.automatic_compute || result.pool_offer_backend == "cpu" || result.pool_offer_vram,
            command.empty() ? "a local GPU offer requires --pool-offer-vram" : "a local GPU offer requires --compute-offer-vram");
        require(result.pool_offer_ram < result.max_ram && result.max_ram - result.pool_offer_ram >= 16 * mib,
            "local compute offer must leave at least 16 MiB for receiving and assembling plots; reduce the offered RAM or raise -M");
        // Reserve the complete service partition before owner construction
        // admission. Its own gate covers sessions, transport and retained GPU
        // state; it cannot consume construction's remaining headroom.
        result.max_ram -= result.pool_offer_ram;
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
        if (result.pool_codec_auto)
            result.pool_codec_ram = std::min({sum(product(result.window, 12),
                product(group_grow::pool::Channel::memory_bytes(uint32_t(result.pool_frame_bytes)), 2)),
                result.max_ram / 16, uint64_t(256 * mib), result.max_ram - 16 * mib});
#endif
        require(result.pool_codec_ram < result.max_ram && result.max_ram - result.pool_codec_ram >= 16 * mib,
            "remote codec partition leaves insufficient owner RAM");
        result.max_ram -= result.pool_codec_ram;
    }
    if (result.directory_farm && !farm_command)
    {
        result.target = fs::canonical(result.target);
        for (auto& destination : result.destinations) destination = fs::canonical(destination);
    }
    return result;
}
