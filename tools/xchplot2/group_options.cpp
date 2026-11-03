#include "group_commands.hpp"
#ifdef XCHPLOT2_GROUP_COMMANDS
#include "host/ComputeSession.hpp"
#endif
#include <algorithm>
#include <climits>
#include <fstream>
#include <iostream>
#include <thread>
#if defined(GROW_POOL_NETWORK) && GROW_POOL_NETWORK
#include "host/ComputeNetwork.hpp"
#endif

namespace {
namespace fs = std::filesystem;
constexpr uint64_t mib = 1024 * 1024;
void require(bool condition, std::string const& message)
{ if (!condition) throw std::runtime_error(message); }
uint64_t number(char const* text)
{
    std::string value(text);
    require(!value.empty() && value.find_first_not_of("0123456789") == value.npos, "invalid integer");
    return std::stoull(value);
}
uint64_t sum(uint64_t a, uint64_t b)
{ require(b <= UINT64_MAX - a, "byte count overflow"); return a + b; }
uint64_t product(uint64_t a, uint64_t b)
{ require(!b || a <= UINT64_MAX / b, "byte count overflow"); return a * b; }
void reference_help()
{
    std::cout <<
        "Ordinary PoS2 group construction reference runner\n"
        "Usage: pos2_pr118_grow TARGET [OPTIONS]\n"
        "      --destination FILE|DIR  additional key-derived destination; repeat for more drives\n"
        "  -t, --temp-dir DIR        optional contribution storage; repeat for more devices\n"
        "  --destination-temp off|on|ssd|hdd  destination scratch policy, default off\n"
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
    require(argc > 1 || command == "plan", "destination required; use --help");
    GroupOptions result;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    result.backend = "auto";
#endif
    result.command = command;
    result.plan_only = command == "plan";
    result.resume = command == "grow";
    bool literal = false;
    auto path = [](std::string const& value)
    { return fs::absolute(fs::path(reinterpret_cast<char8_t const*>(value.c_str()))).lexically_normal(); };
    auto destination = [&](std::string const& value)
    {
        require(!value.empty(), "empty destination path");
        if (result.target.empty()) result.target = path(value);
        else
        {
            require(command != "grow" && command != "plot", "unexpected destination " + value);
            require(result.destinations.size() < 1023, "too many destinations");
            result.destinations.push_back(path(value));
        }
    };
    for (int i = 1; i < argc; ++i)
    {
        std::string key = argv[i];
        if (literal) { destination(key); continue; }
        if (key == "--") { literal = true; continue; }
        if (!command.empty() && key.starts_with("--compute-")) key.replace(2, 8, "pool-");
        if (key == "--help" && command.empty()) { reference_help(); std::exit(0); }
        if (key == "--resume" || key == "-r") { result.resume = true; continue; }
        if (key == "--plan-only") { result.plan_only = true; continue; }
        if (key == "--extend") { result.extend = true; continue; }
        if (key == "--allow-missing") { result.allow_missing = true; continue; }
        if (!command.empty())
        {
            if (key == "--no-resume") { result.resume = command == "grow"; continue; }
            if (key == "--no-plan-only") { result.plan_only = command == "plan"; continue; }
            if (key == "--no-extend") { result.extend = false; continue; }
            if (key == "--no-allow-missing") { result.allow_missing = false; continue; }
            if (key == "--json") { result.json = true; continue; }
            if (key == "--no-json") { result.json = false; continue; }
            if (key == "--quiet" || key == "-q") { result.quiet = true; continue; }
            if (key == "--no-quiet") { result.quiet = false; continue; }
            if (key == "--verbose" || key == "-v" || key == "--progress") { result.verbose = true; continue; }
            if (key == "--no-verbose" || key == "--no-progress") { result.verbose = false; continue; }
            require(key != "--fixture-inputs", "fixture input is only available in the reference test runner");
        }
        if (key.starts_with("-"))
        {
            require(i + 1 < argc, "missing value for " + key);
            std::string const value = argv[++i];
            if (key == "--group-id") result.group = value;
            else if (key == "--manifest") result.manifest = path(value);
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
            else if (key == "--pool-offer-ram") result.pool_offer_ram = number(value.c_str());
            else if (key == "--pool-offer-cpus") result.pool_offer_cpus = number(value.c_str());
            else if (key == "--pool-offer-vram") result.pool_offer_vram = number(value.c_str());
            else if (key == "--pool-offer-slots") result.pool_offer_slots = number(value.c_str());
            else if (key == "--pool-inflight") result.pool_inflight = number(value.c_str());
            else if (key == "--pool-timeout-ms") result.pool_timeout_ms = number(value.c_str());
            else if (key == "--pool-socket-timeout-ms") result.pool_socket_timeout_ms = number(value.c_str());
            else if (key == "--pool-frame-bytes") result.pool_frame_bytes = number(value.c_str());
            else if (key == "--pool-codec-ram")
            { result.pool_codec_auto = value == "auto"; if (!result.pool_codec_auto) result.pool_codec_ram = number(value.c_str()); }
            else if (key == "--temp-dir" || key == "-t")
            {
                require(result.temporary.size() < 1024, "too many temporary directories");
                result.temporary.push_back(fs::absolute(fs::path(reinterpret_cast<char8_t const*>(value.c_str()))).lexically_normal());
            }
            else if (key == "--temp-capacity-bytes")
            {
                result.temporary_capacity = number(value.c_str());
                require(result.temporary_capacity > 0, "--temp-capacity-bytes must be positive");
            }
            else if (key == "--destination-temp") result.destination_temp = value;
            else if (key == "--destination" && command.empty())
            {
                require(result.destinations.size() < 1023, "too many destinations");
                result.destinations.push_back(fs::absolute(fs::path(reinterpret_cast<char8_t const*>(value.c_str()))).lexically_normal());
            }
            else if (key == "--k" || key == "-k") result.k = number(value.c_str());
            else if (key == "--strength" || key == "-s") result.strength = number(value.c_str());
            else if (key == "--meta-group" || key == "-g") result.meta = number(value.c_str());
            else if (key == "--capacity-bytes")
            {
                result.capacity = number(value.c_str());
                result.capacity_set = true;
                require(result.capacity > 0, "--capacity-bytes must be positive");
            }
            else if (key == "--headroom-bytes") { result.headroom = number(value.c_str()); result.headroom_set = true; }
            else if (key == "--max-ram" || key == "-M") result.max_ram = number(value.c_str());
            else if (key == "--window-bytes") { result.window = number(value.c_str()); result.window_set = true; }
            else if (key == "--workers" || key == "-j") result.workers = number(value.c_str());
            else if (key == "--batch-members") result.batch = number(value.c_str());
            else if (key == "--max-members") result.max_members = number(value.c_str());
            else if (key == "--group-members")
            {
                result.group_auto = value == "auto";
                result.group_members = 0;
                if (!result.group_auto)
                {
                    result.group_members = number(value.c_str());
                    require(result.group_members > 0, "--group-members must be positive");
                }
            }
            else if (key == "--farming-reader") result.farming_reader = value;
            else if (key == "--farming-objective") result.farming_objective = value;
            else if (key == "--farming-cpu-work-weight") result.farming_cpu_work_weight = number(value.c_str());
            else if (key == "--farming-io-work-weight") result.farming_io_work_weight = number(value.c_str());
            else if (key == "--farming-max-eligible-groups") result.farming_max_eligible_groups = number(value.c_str());
            else if (key == "--farming-lookup-workers") result.farming_lookup_workers = number(value.c_str());
            else if (key == "--farming-memory-bytes") result.farming_memory = number(value.c_str());
            else if (key == "--farming-cores") result.farming_cores = number(value.c_str());
            else if (key == "--farming-index-cache-bytes") result.farming_index_cache = number(value.c_str());
            else if (key == "--farming-filter-bits") result.farming_filter_bits = number(value.c_str());
            else if (key == "--farming-future-filter-bits") result.farming_future_filter_bits = number(value.c_str());
            else if (key == "--farming-chain-us-per-member") result.farming_chain_us = number(value.c_str());
            else if (key == "--farming-index-ns-per-entry") result.farming_index_ns = number(value.c_str());
            else if (key == "--farming-deadline-ms") result.farming_deadline_ms = number(value.c_str());
            else if (key == "--max-groups") { result.max_groups = number(value.c_str()); result.max_groups_set = true; }
            else if (key == "--max-total-members") { result.max_total_members = number(value.c_str()); result.max_total_members_set = true; }
            else if (key == "--checkpoint-bytes") { result.checkpoint_bytes = number(value.c_str()); result.checkpoint_set = true; }
            else if (key == "--backend") { result.backend = value; result.backend_set = true; }
            else if (key == "--device") result.device = number(value.c_str());
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
    if (result.extend || !result.relocations.empty()) result.resume = true;
    require(!result.target.empty() || (farm_command && (result.plan_only || result.resume || !result.manifest.empty())), "destination required");
    require((!result.extend && !result.allow_missing && result.manifest.empty() && result.relocations.empty()) || farm_command,
        "saved farming job options require plan or fill");
    require(!result.allow_missing || (result.resume && !result.extend && result.relocations.empty()),
        "--allow-missing requires --resume without --extend or --relocate");
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
    bool saved_farm = false;
    if (result.resume && directory_target)
        saved_farm = fs::exists(result.target / "grow-farm.plan") || fs::exists(result.target / "grow-farm.plan.next");
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
    require(!result.farming_reader || *result.farming_reader == "stateless" || *result.farming_reader == "cached",
            "--farming-reader requires stateless or cached");
    require(!result.farming_objective || *result.farming_objective == "energy" || *result.farming_objective == "latency",
            "--farming-objective requires energy or latency");
    require(!result.farming_cpu_work_weight || *result.farming_cpu_work_weight > 0, "farming CPU work weight must be positive");
    require(!result.farming_io_work_weight || *result.farming_io_work_weight > 0, "farming I/O work weight must be positive");
    require(!result.farming_max_eligible_groups || *result.farming_max_eligible_groups > 0,
            "farming eligible-group limit must be positive");
    require(!farming_options || result.directory_farm, "farming profile options require automatic directory planning");
    require(!result.farming_cores || (*result.farming_cores > 0 && *result.farming_cores <= 1024), "invalid farming CPU budget");
    require(!result.farming_lookup_workers || (*result.farming_lookup_workers > 0 && *result.farming_lookup_workers <= 1024), "invalid farming reader concurrency");
    require(!result.farming_memory || *result.farming_memory > 0, "farming memory must be positive");
    require(!result.farming_filter_bits || *result.farming_filter_bits <= 13, "invalid farming filter");
    require(!result.farming_future_filter_bits || *result.farming_future_filter_bits <= 13, "invalid future farming filter");
    require(!result.farming_chain_us || *result.farming_chain_us > 0, "farming chain cost must be positive");
    require(!result.farming_deadline_ms || *result.farming_deadline_ms > 0, "farming deadline must be positive");
    require(result.backend == "cpu" || result.backend == "gpu" || result.backend == "auto",
            "--backend requires cpu, gpu or auto");
    require(result.tier == "auto" || result.tier == "pool" || result.tier == "plain" || result.tier == "compact" ||
                result.tier == "minimal" || result.tier == "tiny" || result.tier == "pinned",
            "unknown GPU tier");
    require(result.device <= INT_MAX && result.checkpoint_bytes > 0,
            "invalid GPU device or checkpoint size");
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
    require(result.window > 0 && result.max_ram >= 16 * mib,
            "positive window and at least 16777216 budget bytes required");
    require(!result.k || *result.k <= 255, "invalid k");
    require(!result.strength || *result.strength <= 255, "invalid strength");
    require(!result.meta || *result.meta <= 255, "invalid meta group");
#ifdef XCHPLOT2_GROUP_COMMANDS
    prepare_group_compute(result);
#endif
    bool const pool_options = !result.pool_owner.empty() || !result.pool_ca.empty() || !result.pool_cert.empty() ||
        !result.pool_key.empty() || !result.pool_listen.empty() || !result.pool_advertise.empty() ||
        result.pool_offer_ram || result.pool_offer_cpus || result.pool_offer_vram || !result.pool_codec_auto;
    require(!pool_options || !result.pool_broker.empty(), command.empty() ? "network compute options require --pool-broker" : "network compute options require --compute-broker");
    if (!result.pool_broker.empty())
    {
        if (!result.backend_set) result.backend = "auto";
#if !defined(GROW_POOL_NETWORK) || !GROW_POOL_NETWORK
        require(false, "network compute is unavailable in this build");
#endif
        require(!result.pool_owner.empty() && result.pool_owner.size() <= 128 &&
            !result.pool_ca.empty() && !result.pool_cert.empty() && !result.pool_key.empty(),
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
        require(!offer || ((result.automatic_compute || (!result.pool_listen.empty() && !result.pool_advertise.empty())) &&
            result.pool_offer_ram && result.pool_offer_cpus),
            "offering local compute requires listen, advertise, RAM and CPU grants");
        unsigned const workers = result.workers ? unsigned(result.workers) : std::max(1u, std::thread::hardware_concurrency());
        require(!offer || ((workers == 1 ? result.pool_offer_cpus == 1 : result.pool_offer_cpus < workers) &&
            result.pool_offer_slots && result.pool_offer_slots <= 1024),
            "local compute offer must leave CPU capacity for receiving and storage");
        require(result.pool_offer_backend == "cpu" || result.pool_offer_backend == "gpu" || result.pool_offer_backend == "auto",
            command.empty() ? "--pool-offer-backend requires cpu, gpu or auto" : "--compute-offer-backend requires cpu, gpu or auto");
        require(!offer || result.automatic_compute || result.pool_offer_backend == "cpu" || result.pool_offer_vram,
            command.empty() ? "a local GPU offer requires --pool-offer-vram" : "a local GPU offer requires --compute-offer-vram");
        require(result.pool_offer_ram < result.max_ram && result.max_ram - result.pool_offer_ram >= 16 * mib,
            "local compute offer leaves insufficient owner RAM");
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
