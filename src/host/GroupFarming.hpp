// Conditional farming-service model, independent of construction hardware.
// The certificate covers every modeled filter sequence, not unmeasured hardware.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace group_grow
{
inline constexpr char farming_model_identity[] = "group-farming-3";
enum class FarmingReader : uint8_t { stateless, cached };
enum class FarmingObjective : uint8_t { energy, latency };
class FarmingResourceError : public std::runtime_error
{
public:
    bool const memory_budget;
    explicit FarmingResourceError(char const* message, bool memory = false) : std::runtime_error(message), memory_budget(memory) {}
};
struct FarmingConfig
{
    uint8_t k = 28, strength = 2, filter_bits = 255, future_filter_bits = 255;
    FarmingReader reader = FarmingReader::stateless;
    FarmingObjective objective = FarmingObjective::energy;
    unsigned harvester_cores = 4, lookup_workers = 4;
    uint64_t index_cache_bytes = 0, memory_bytes = 512ull * 1024 * 1024;
    uint64_t max_eligible_families_per_resource = 1;
    long double cpu_work_weight = 1, io_work_weight = 1;
    long double deadline_seconds = 8, challenge_seconds = 75.0L / 8;
    uint16_t group_members_override = 0, meta_slots_override = 0;
    uint8_t meta_group_override = 0;
    bool fixed_meta = false;
    // Declared service envelopes, not measured worst-case CPU/fragment costs.
    long double chain_seconds_per_member = 0.004L;
    long double index_seconds_per_entry = 0.00000005L;
    // GSZ uses at most one 64-bit codeword per entry. Capacity forecasting
    // separately retains the pre-existing expected two-byte shape.
    long double compressed_index_bytes_per_entry = 8;
    uint64_t lookup_memory_per_member = 4096;
    bool operator==(FarmingConfig const&) const = default;
};
struct FarmingDestination
{
    uint64_t capacity_bytes = 0, max_file_bytes = UINT64_MAX;
    uint64_t max_total_members = UINT64_MAX, max_groups = UINT64_MAX;
    uint64_t resource = UINT64_MAX;
    bool known_device = false, nonrotating = false;
    long double read_bytes_per_second = 0, read_latency_seconds = 0;
    unsigned read_concurrency = 0;
    bool operator==(FarmingDestination const&) const = default;
};
struct FarmingDrivePlan
{
    // Additions use a new identity schedule. Its family/meta interleave stays
    // fixed when directories join or move between physical resources.
    uint64_t schedule = 0;
    uint64_t total_members = 0, files = 0, families = 0, group_bytes = 0;
    uint16_t group_members = 0, meta_slots = 1;
    uint8_t meta_bits = 8, meta_base = 0;
    bool fixed_members = false;
    uint16_t resource_rank = 0, resource_lanes = 1;
    // Forecast counts only. For explicit fixed groups, live growth may fill a
    // forecast terminal partial file up to group_members as outer limits grow.
    uint16_t file_members(uint64_t ordinal) const
    {
        if (!files || ordinal >= files) return group_members;
        if (fixed_members) return uint16_t(std::min<uint64_t>(group_members, total_members - ordinal * group_members));
        return uint16_t(total_members / files + (ordinal < total_members % files));
    }
    uint64_t resource_file(uint64_t ordinal) const
    {
        if (!resource_lanes || resource_rank >= resource_lanes || !meta_slots ||
            ordinal > (UINT64_MAX - resource_rank) / resource_lanes)
            throw std::runtime_error("invalid farming file ordinal");
        return ordinal * resource_lanes + resource_rank;
    }
    uint64_t family(uint64_t ordinal) const { return resource_file(ordinal) / meta_slots; }
    uint8_t meta(uint64_t ordinal) const
    {
        unsigned value = unsigned(resource_file(ordinal) % meta_slots), reversed = 0;
        for (unsigned bit = 0; bit < meta_bits; ++bit)
        { reversed = (reversed << 1) | (value & 1); value >>= 1; }
        return uint8_t(reversed ^ meta_base);
    }
    bool operator==(FarmingDrivePlan const&) const = default;
};
struct FarmingAssessment
{
    bool feasible = false;
    std::string reason;
    long double response_bound_seconds = 0;
    // Weighted CPU-active plus read-slot/device-active seconds per SP. This
    // normalized dynamic-work proxy excludes idle power and is not joules.
    long double dynamic_work_seconds_per_sp = 0, cpu_work_seconds_per_sp = 0, io_work_seconds_per_sp = 0;
    uint64_t max_eligible_families_per_resource = 0;
    uint64_t reader_memory_bytes = 0, planning_memory_bytes = 0;
};
struct FarmingPlan : FarmingAssessment
{
    std::vector<FarmingDrivePlan> destinations;
    uint64_t index_bytes = 0;
    long double cpu_utilization = 0, worker_utilization = 0;
    uint64_t candidates_evaluated = 0;
};
inline FarmingDestination farming_storage_profile(FarmingDestination value)
{
    if (!value.read_bytes_per_second) value.read_bytes_per_second = !value.known_device ?
        125000000.0L : value.nonrotating ? 500000000.0L : 250000000.0L;
    if (!value.read_latency_seconds) value.read_latency_seconds = !value.known_device ?
        0.020L : value.nonrotating ? 0.0002L : 0.010L;
    if (!value.read_concurrency) value.read_concurrency = value.known_device && value.nonrotating ? 4 : 1;
    return value;
}
namespace farming_detail
{
inline constexpr uint64_t maximum_files = 262144, maximum_candidates = 1048576;
inline void farming_require(bool condition, char const* message)
{ if (!condition) throw std::runtime_error(message); }
inline void resource_require(bool condition, char const* message)
{ if (!condition) throw FarmingResourceError(message); }
inline uint64_t ceiling(uint64_t value, uint64_t unit)
{ return value / unit + bool(value % unit); }
inline uint64_t size(long double value)
{
    farming_require(std::isfinite(value) && value >= 0 && value < static_cast<long double>(UINT64_MAX),
        "farming size estimate overflows");
    return uint64_t(std::ceil(value));
}
inline void add(uint64_t& target, uint64_t value)
{
    farming_require(value <= UINT64_MAX - target, "farming count overflows"); target += value;
}
struct PlanningMemory
{
    uint64_t limit, used = 0, peak = 0;
    void add(uint64_t bytes)
    {
        if (bytes > limit - used) throw FarmingResourceError("farming planner exceeds its planning memory budget", true);
        used += bytes; peak = std::max(peak, used);
    }
};
struct PlanningReservation
{
    PlanningMemory& memory; uint64_t bytes;
    PlanningReservation(PlanningMemory& source, uint64_t amount) : memory(source), bytes(amount) { memory.add(bytes); }
    ~PlanningReservation() { memory.used -= bytes; }
};
inline unsigned filter(FarmingConfig const& config, bool future)
{
    unsigned const bits = future ? config.future_filter_bits : config.filter_bits;
    return bits == 255 ? std::min(13u, (future ? 8u : 9u) + config.strength - 2) : bits;
}
inline long double member_bytes(FarmingConfig const& config)
{ return std::ldexp(1.0L, config.k) * (config.k + 1.45L) / 8; }
inline bool shape(FarmingConfig const& config, unsigned members)
{
    if (!members) return true;
    long double const rice = std::round(std::log2(std::floor(std::sqrt(357.0L * members))));
    long double const center = std::floor(60301.0L * members / 256);
    long double const chunk = 8 * (config.k + 1.45L) * members;
    return rice + std::floor(std::fabs(chunk - center) / std::ldexp(1.0L, int(rice))) + 2 <= 64;
}
inline void validate(FarmingConfig const& config, std::vector<FarmingDestination> const& destinations)
{
    farming_require(config.k >= 18 && config.k <= 28 && !(config.k & 1) && config.strength >= 2 &&
        config.strength <= std::min(17, config.k - 3), "invalid farming plot parameters");
    farming_require(filter(config, false) <= 13 && filter(config, true) <= 13 && config.harvester_cores &&
        config.harvester_cores <= 1024 && config.lookup_workers && config.lookup_workers <= 1024 && config.memory_bytes &&
        config.max_eligible_families_per_resource && config.deadline_seconds > 0 && std::isfinite(config.deadline_seconds) &&
        config.challenge_seconds > 0 && std::isfinite(config.challenge_seconds) && config.chain_seconds_per_member > 0 &&
        std::isfinite(config.chain_seconds_per_member) && config.index_seconds_per_entry >= 0 &&
        std::isfinite(config.index_seconds_per_entry) && config.compressed_index_bytes_per_entry > 0 &&
        std::isfinite(config.compressed_index_bytes_per_entry) && config.lookup_memory_per_member &&
        config.cpu_work_weight > 0 && std::isfinite(config.cpu_work_weight) && config.io_work_weight > 0 &&
        std::isfinite(config.io_work_weight), "invalid farming reader constraints");
    farming_require(config.objective == FarmingObjective::energy || config.objective == FarmingObjective::latency,
        "invalid farming objective");
    farming_require(destinations.size() <= 1024, "too many farming destinations");
    unsigned const slots = 1u << std::min({8u, filter(config, false), filter(config, true)});
    farming_require(!config.meta_slots_override || config.meta_slots_override <= slots,
        "farming meta slots alias under a forecast filter");
    farming_require(!config.fixed_meta || !config.meta_slots_override || config.meta_slots_override == 1,
        "fixed farming meta cannot rotate slots");
    for (auto const& destination : destinations)
        farming_require(std::isfinite(destination.read_bytes_per_second) && destination.read_bytes_per_second >= 0 &&
            std::isfinite(destination.read_latency_seconds) && destination.read_latency_seconds >= 0 &&
            destination.read_concurrency <= 1024, "invalid farming storage profile");
}
struct Resource
{
    FarmingDestination storage;
    uint64_t files = 0, members = 0, families = 0, family_members = 0;
    uint16_t largest = 0;
};
struct Topology
{
    std::vector<size_t> resource;
    std::vector<uint16_t> ranks, lanes;
    std::vector<Resource> resources;
};
inline Topology topology(std::vector<FarmingDestination> const& destinations)
{
    Topology value;
    value.resource.resize(destinations.size()); value.ranks.resize(destinations.size()); value.lanes.resize(destinations.size());
    for (size_t lane = 0; lane < destinations.size(); ++lane)
    {
        size_t before = lane;
        for (size_t other = 0; other < lane; ++other)
            if (destinations[lane].resource != UINT64_MAX && destinations[lane].resource == destinations[other].resource)
            { before = other; break; }
        auto profile = farming_storage_profile(destinations[lane]);
        value.resource[lane] = before == lane ? value.resources.size() : value.resource[before];
        if (before == lane) value.resources.push_back({profile});
        else
        {
            auto& shared = value.resources[value.resource[lane]].storage;
            shared.read_bytes_per_second = std::min(shared.read_bytes_per_second, profile.read_bytes_per_second);
            shared.read_latency_seconds = std::max(shared.read_latency_seconds, profile.read_latency_seconds);
            shared.read_concurrency = std::min(shared.read_concurrency, profile.read_concurrency);
        }
    }
    for (size_t lane = 0; lane < destinations.size(); ++lane)
        for (size_t other = 0; other < destinations.size(); ++other)
            if (value.resource[lane] == value.resource[other])
            { if (other < lane) ++value.ranks[lane]; ++value.lanes[lane]; }
    return value;
}
struct Event { uint64_t family, schedule; uint32_t resource; uint16_t members; bool end; };
inline FarmingAssessment assess(FarmingConfig const& config, std::vector<FarmingDestination> const& destinations,
    std::vector<FarmingDrivePlan> const& layouts, std::vector<std::vector<uint16_t>> const* extents,
    PlanningMemory& memory)
{
    farming_require(layouts.size() == destinations.size() && (!extents || extents->size() == layouts.size()),
        "farming assessment lane count differs");
    uint64_t event_count = 0, files = 0, expanded_files = 0;
    for (size_t lane = 0; lane < layouts.size(); ++lane)
    {
        uint64_t const count = extents ? (*extents)[lane].size() : layouts[lane].files;
        if (extents || layouts[lane].resource_lanes > layouts[lane].meta_slots)
        {
            resource_require(count <= maximum_files - expanded_files, "farming assessment exceeds its file work bound");
            expanded_files += count;
        }
        add(files, count);
        if (extents || layouts[lane].resource_lanes > layouts[lane].meta_slots) add(event_count, count * 2);
        else add(event_count, 4);
    }
    PlanningReservation allocation(memory, 64 * 1024 + event_count * (sizeof(Event) + 8) + destinations.size() * 512);
    auto input = topology(destinations);
    std::vector<Event> events; events.reserve(size_t(event_count));
    unsigned const meta_bits = std::min({8u, filter(config, false), filter(config, true)});
    for (size_t lane = 0; lane < layouts.size(); ++lane)
    {
        auto const& plan = layouts[lane];
        uint16_t rank = 0, lanes = 0;
        // A saved identity schedule can now span several physical resources.
        for (size_t other = 0; other < layouts.size(); ++other)
            if (layouts[other].schedule == plan.schedule)
            { if (other < lane) ++rank; ++lanes; }
        farming_require(plan.resource_rank == rank && plan.resource_lanes == lanes &&
            plan.meta_slots && plan.meta_slots <= (1u << meta_bits) && plan.meta_bits == meta_bits &&
            plan.meta_base == config.meta_group_override && (!config.fixed_meta || plan.meta_slots == 1) &&
            (!config.meta_slots_override || plan.meta_slots == config.meta_slots_override),
            "farming assessment layout aliases resource/meta identities");
        for (size_t before = 0; before < lane; ++before)
            farming_require(layouts[before].schedule != plan.schedule ||
                layouts[before].meta_slots == plan.meta_slots,
                "farming resource lanes use incompatible family schedules");
        auto& resource = input.resources[input.resource[lane]];
        uint64_t const count = extents ? (*extents)[lane].size() : plan.files;
        add(resource.files, count);
        auto interval = [&](uint64_t first, uint64_t last, uint16_t members)
        {
            if (first == last) return;
            farming_require(members && shape(config, members), "farming extent has an invalid modeled chunk shape");
            uint64_t const begin = plan.family(first), end = plan.family(last - 1);
            farming_require(end != UINT64_MAX, "farming family extent overflows");
            events.push_back({begin, plan.schedule, uint32_t(input.resource[lane]), members, false});
            events.push_back({end + 1, plan.schedule, uint32_t(input.resource[lane]), members, true});
            resource.largest = std::max(resource.largest, members);
        };
        if (extents || plan.resource_lanes > plan.meta_slots)
            for (uint64_t file = 0; file < count; ++file)
            {
                uint16_t const members = extents ? (*extents)[lane][size_t(file)] : plan.file_members(file);
                interval(file, file + 1, members); add(resource.members, members);
            }
        else if (count)
        {
            add(resource.members, plan.total_members);
            if (plan.fixed_members)
            { interval(0, count - 1, plan.group_members); interval(count - 1, count, plan.file_members(count - 1)); }
            else
            {
                uint64_t const extra = plan.total_members % count;
                interval(0, extra, uint16_t(plan.total_members / count + 1));
                interval(extra, count, uint16_t(plan.total_members / count));
            }
        }
    }
    std::sort(events.begin(), events.end(), [](Event const& a, Event const& b)
    { return std::tie(a.resource, a.schedule, a.family, a.members, a.end) < std::tie(b.resource, b.schedule, b.family, b.members, b.end); });
    std::vector<std::pair<uint16_t, uint64_t>> active; active.reserve(size_t(std::min<uint64_t>(event_count / 2, 65535)));
    size_t position = 0;
    while (position < events.size())
    {
        auto const resource = events[position].resource;
        auto const schedule = events[position].schedule;
        active.clear();
        while (position < events.size() && events[position].resource == resource && events[position].schedule == schedule)
        {
            uint64_t const family = events[position].family;
            do
            {
                auto const event = events[position++];
                auto item = std::find_if(active.begin(), active.end(), [&](auto const& entry) { return entry.first == event.members; });
                if (item == active.end()) { active.push_back({event.members, 0}); item = active.end() - 1; }
                if (event.end) { farming_require(item->second, "farming family sweep underflows"); --item->second; }
                else ++item->second;
            } while (position < events.size() && events[position].resource == resource &&
                events[position].schedule == schedule && events[position].family == family);
            uint16_t largest = 0;
            for (auto const& entry : active) if (entry.second) largest = std::max(largest, entry.first);
            if (largest && position < events.size() && events[position].resource == resource && events[position].schedule == schedule)
            {
                uint64_t const span = events[position].family - family;
                add(input.resources[resource].families, span);
                farming_require(span <= UINT64_MAX / largest, "farming family work overflows");
                add(input.resources[resource].family_members, span * largest);
            }
        }
    }
    FarmingAssessment result;
    uint64_t jobs = 0, largest_members = 0;
    for (auto const& resource : input.resources)
    {
        add(jobs, resource.families); largest_members = std::max<uint64_t>(largest_members, resource.largest);
        result.max_eligible_families_per_resource = std::max(result.max_eligible_families_per_resource, resource.families);
    }
    if (!jobs) { result.feasible = true; result.planning_memory_bytes = memory.peak; return result; }
    long double const entries = std::ldexp(1.0L, config.k - 6), expanded = entries * 8;
    long double const compressed = entries * config.compressed_index_bytes_per_entry;
    long double const cache = config.reader == FarmingReader::cached ?
        std::min(static_cast<long double>(config.index_cache_bytes), files * expanded) : 0;
    bool const warm = config.reader == FarmingReader::cached && cache >= files * expanded;
    // Cold open reads the header before the index. Each selected chunk is
    // decoded before the next read, so CPU admission can recur at every read.
    // The declared index CPU allowance also covers header/open CPU work.
    unsigned const reads = warm ? 4 : 6, cpu_stages = reads;
    long double const index_cpu = warm ? 0 : entries * config.index_seconds_per_entry;
    long double const index_read = warm ? 0 : compressed;
    long double const header_read = warm ? 0 : 306;
    unsigned const workers = unsigned(std::min<uint64_t>(config.lookup_workers, jobs));
    result.reader_memory_bytes = size(cache + workers * (expanded * !warm +
        std::max(compressed, static_cast<long double>(largest_members) * config.lookup_memory_per_member) + 1024 * 1024));
    long double const cpu_stage = std::max(index_cpu, largest_members * config.chain_seconds_per_member);
    // FIFO CPU admission: at most workers-1 earlier outstanding operations.
    // Each occupies a core for at most cpu_stage. The last of these waves
    // leaves a core after floor((workers-1)/cores) waves; no wait when W<=C.
    long double const cpu_wait = ((workers - 1) / config.harvester_cores) * cpu_stage;
    long double total = 0, largest = 0;
    long double const chunk_per_member = 4 * 64 * (config.k + 1.45L) / 8;
    long double const eligibility = std::ldexp(1.0L, -int(std::min(filter(config, false), filter(config, true))));
    for (auto const& resource : input.resources)
    {
        if (!resource.families) continue;
        auto const& storage = resource.storage;
        uint64_t const outstanding = std::min<uint64_t>(workers, resource.families);
        // At most outstanding FIFO reads are pending on this device. Their
        // seeks take at most ceil(outstanding/concurrency) latency waves;
        // shared transfer bytes are charged serially. Future requests cannot
        // overtake the read being bounded.
        long double const chunk_max = chunk_per_member * resource.largest;
        // A long request can meet replacement workers at each read stage.
        // Bound these independently, and also cap the interference by all
        // other families' complete transfer work in this finite SP batch.
        long double const interference = std::min(reads * (outstanding - 1) * std::max({index_read, header_read, chunk_max / 4}),
            (resource.families - 1) * (index_read + header_read + chunk_max));
        long double const fixed = index_cpu + cpu_stages * cpu_wait +
            reads * ceiling(outstanding, storage.read_concurrency) * storage.read_latency_seconds +
            (index_read + header_read + interference) / storage.read_bytes_per_second;
        long double const member = config.chain_seconds_per_member + chunk_per_member / storage.read_bytes_per_second;
        total += resource.families * fixed + resource.family_members * member;
        largest = std::max(largest, fixed + resource.largest * member);
        result.cpu_work_seconds_per_sp += eligibility * (resource.members * config.chain_seconds_per_member + resource.files * index_cpu);
        result.io_work_seconds_per_sp += eligibility * (resource.files * (reads * storage.read_latency_seconds + (index_read + header_read) / storage.read_bytes_per_second) +
            resource.members * chunk_per_member / storage.read_bytes_per_second);
    }
    // All eligible families can align at an SP. For bounded active durations
    // U, work-conserving W-worker list scheduling completes by either bound.
    // Requiring this batch to drain within cadence proves, by induction, that
    // no valid repeated/epoch-correlated filter sequence carries prior work.
    result.response_bound_seconds = std::min(total / workers + (1 - 1.0L / workers) * largest,
        ceiling(jobs, workers) * largest);
    // Outward allowance for the bounded positive floating-point arithmetic;
    // this is numerical rounding protection, not a hardware timing margin.
    result.response_bound_seconds = std::nextafter(result.response_bound_seconds *
        (1 + 128 * (input.resources.size() + 1) * std::numeric_limits<long double>::epsilon()),
        std::numeric_limits<long double>::infinity());
    result.dynamic_work_seconds_per_sp = config.cpu_work_weight * result.cpu_work_seconds_per_sp +
        config.io_work_weight * result.io_work_seconds_per_sp;
    result.planning_memory_bytes = memory.peak;
    if (!std::isfinite(result.response_bound_seconds) || !std::isfinite(result.dynamic_work_seconds_per_sp))
        result.reason = "farming service certificate overflows";
    else if (result.max_eligible_families_per_resource > config.max_eligible_families_per_resource)
        result.reason = "farming layout exceeds the simultaneous eligible-group limit";
    else if (result.reader_memory_bytes > config.memory_bytes)
        result.reason = "farming reader profile exceeds its working memory budget";
    else if (result.response_bound_seconds > config.deadline_seconds)
        result.reason = "farming profile cannot certify the response deadline";
    else if (result.response_bound_seconds > config.challenge_seconds)
        result.reason = "farming profile cannot drain responses before the next signage point";
    else result.feasible = true;
    return result;
}
inline FarmingDrivePlan candidate(FarmingConfig const& config, FarmingDestination const& destination,
    uint64_t files, uint16_t slots, uint16_t rank, uint16_t lanes)
{
    FarmingDrivePlan plan; plan.meta_slots = slots; plan.resource_rank = rank; plan.resource_lanes = lanes;
    plan.meta_bits = uint8_t(std::min({8u, filter(config, false), filter(config, true)})); plan.meta_base = config.meta_group_override;
    long double const bytes = member_bytes(config), overhead = std::ldexp(2.0L, config.k - 6) + 256;
    if (!files || files > destination.max_groups || files * overhead >= destination.capacity_bytes) return plan;
    uint64_t total = std::min(destination.max_total_members,
        uint64_t(std::min(static_cast<long double>(UINT64_MAX - 1), std::floor((destination.capacity_bytes - files * overhead) / bytes))));
    if (config.group_members_override)
    {
        long double const limit = static_cast<long double>(files) * config.group_members_override;
        total = std::min(total, uint64_t(std::min(static_cast<long double>(UINT64_MAX - 1), limit)));
    }
    if (total < files) return plan;
    uint64_t const members = config.group_members_override ? config.group_members_override : ceiling(total, files);
    if (members > 65535 || !shape(config, unsigned(members))) return plan;
    plan.group_bytes = size(members * bytes + overhead);
    if (plan.group_bytes > destination.max_file_bytes) return FarmingDrivePlan{.meta_slots = slots, .meta_bits = plan.meta_bits,
        .meta_base = plan.meta_base, .resource_rank = rank, .resource_lanes = lanes};
    plan.total_members = total; plan.files = files; plan.group_members = uint16_t(members);
    plan.fixed_members = bool(config.group_members_override); plan.families = plan.family(files - 1) + 1;
    return plan;
}
inline bool better(FarmingConfig const& config, FarmingAssessment const& a, FarmingAssessment const& b)
{
    if (a.feasible != b.feasible) return a.feasible;
    if (!a.feasible)
    {
        if (a.response_bound_seconds != b.response_bound_seconds) return a.response_bound_seconds < b.response_bound_seconds;
        return a.dynamic_work_seconds_per_sp < b.dynamic_work_seconds_per_sp;
    }
    if (config.objective == FarmingObjective::energy)
    {
        if (a.dynamic_work_seconds_per_sp != b.dynamic_work_seconds_per_sp)
            return a.dynamic_work_seconds_per_sp < b.dynamic_work_seconds_per_sp;
        return a.response_bound_seconds < b.response_bound_seconds;
    }
    if (a.response_bound_seconds != b.response_bound_seconds) return a.response_bound_seconds < b.response_bound_seconds;
    return a.dynamic_work_seconds_per_sp < b.dynamic_work_seconds_per_sp;
}
}
inline unsigned farming_filter_bits(FarmingConfig const& config, bool future = false)
{ return farming_detail::filter(config, future); }
inline FarmingAssessment farming_assess(FarmingConfig const& config, std::vector<FarmingDestination> const& destinations,
    std::vector<FarmingDrivePlan> const& layouts, std::vector<std::vector<uint16_t>> const& file_members,
    uint64_t planning_memory_limit = 64ull * 1024 * 1024)
{
    farming_detail::validate(config, destinations);
    farming_detail::PlanningMemory memory{planning_memory_limit};
    return farming_detail::assess(config, destinations, layouts, &file_members, memory);
}
inline FarmingPlan farming_plan(FarmingConfig const& config, std::vector<FarmingDestination> const& destinations,
    uint64_t planning_memory_limit = 64ull * 1024 * 1024, std::span<FarmingDrivePlan const> frozen = {})
{
    using namespace farming_detail;
    validate(config, destinations);
    farming_require(frozen.size() <= destinations.size(), "saved farming layout exceeds destination count");
    PlanningMemory memory{planning_memory_limit};
    memory.add(64 * 1024 + destinations.size() * 1024);
    auto topology_value = topology(destinations);
    uint64_t schedule = 0;
    for (auto const& layout : frozen)
    {
        farming_require(layout.schedule < UINT64_MAX, "farming schedule exhausted");
        schedule = std::max(schedule, layout.schedule + 1);
    }
    for (size_t lane = frozen.size(); lane < destinations.size(); ++lane)
    {
        topology_value.ranks[lane] = topology_value.lanes[lane] = 0;
        for (size_t other = frozen.size(); other < destinations.size(); ++other)
            if (topology_value.resource[lane] == topology_value.resource[other])
            { if (other < lane) ++topology_value.ranks[lane]; ++topology_value.lanes[lane]; }
    }
    unsigned const slots_limit = 1u << std::min({8u, filter(config, false), filter(config, true)});
    uint16_t const slots = config.fixed_meta ? 1 : config.meta_slots_override ? config.meta_slots_override : uint16_t(slots_limit);
    // More distinct metas can only merge mutually exclusive family demand;
    // expected work and memory are unchanged. The full nonalias schedule
    // therefore dominates smaller automatic schedules and leaves growth room.
    std::vector<std::shared_ptr<std::vector<FarmingDrivePlan>>> tables(destinations.size());
    std::vector<FarmingDrivePlan> selected(destinations.size());
    uint64_t generated = 0;
    for (size_t lane = 0; lane < destinations.size(); ++lane)
    {
        if (lane < frozen.size())
        {
            memory.add(sizeof(FarmingDrivePlan) + 128);
            tables[lane] = std::make_shared<std::vector<FarmingDrivePlan>>(1, frozen[lane]);
            selected[lane] = frozen[lane];
            continue;
        }
        size_t before = lane;
        for (size_t other = frozen.size(); other < lane; ++other)
        {
            auto a = destinations[lane], b = destinations[other]; a.resource = b.resource = UINT64_MAX;
            if (a == b && topology_value.lanes[lane] == 1 && topology_value.lanes[other] == 1) { before = other; break; }
        }
        farming_require(schedule <= UINT64_MAX - topology_value.resource[lane], "farming schedule exhausted");
        if (before != lane) tables[lane] = tables[before];
        else
        {
            auto const& destination = destinations[lane];
            uint64_t maximum = destination.max_groups;
            if (topology_value.lanes[lane] > slots)
            {
                // Sparse interleaving assigns each local file a distinct
                // family, which need not start at family zero. Bound the
                // occupied count here; assessment checks the cross-lane union.
                maximum = std::min(maximum, config.max_eligible_families_per_resource);
            }
            else if (config.max_eligible_families_per_resource <= UINT64_MAX / slots)
            {
                uint64_t const positions = config.max_eligible_families_per_resource * slots;
                maximum = std::min(maximum, positions > topology_value.ranks[lane] ?
                    ceiling(positions - topology_value.ranks[lane], topology_value.lanes[lane]) : 0);
            }
            maximum = std::min(maximum, destination.max_total_members);
            maximum = std::min(maximum, uint64_t(std::min(static_cast<long double>(UINT64_MAX - 1),
                std::floor(destination.capacity_bytes / member_bytes(config)))));
            uint64_t const table_limit = config.group_members_override ? 1 : maximum;
            resource_require(table_limit <= maximum_files, "farming search exceeds its file-count work bound");
            resource_require(table_limit <= maximum_candidates - generated, "farming search exceeds its candidate work bound");
            generated += table_limit;
            memory.add((table_limit + 1) * sizeof(FarmingDrivePlan) + 128);
            tables[lane] = std::make_shared<std::vector<FarmingDrivePlan>>(); auto& table = *tables[lane]; table.reserve(size_t(table_limit + 1));
            if (config.group_members_override)
            {
                // Fixed groups fill their explicit member limit, with only the
                // final outer-capacity/member-limit file partial.
                uint64_t const overhead = size(std::ldexp(2.0L, config.k - 6) + 256);
                long double const denominator = member_bytes(config) * config.group_members_override + overhead;
                uint64_t const full = uint64_t(std::min(static_cast<long double>(UINT64_MAX - 1), std::floor(destination.capacity_bytes / denominator)));
                uint64_t count = std::min(destination.max_groups,
                    full + bool(destination.capacity_bytes >= full * denominator + overhead + member_bytes(config)));
                if (destination.max_total_members != UINT64_MAX) count = std::min(count, ceiling(destination.max_total_members, config.group_members_override));
                auto value = candidate(config, destination, count, slots, topology_value.ranks[lane], topology_value.lanes[lane]);
                if (value.files) table.push_back(value);
            }
            else for (uint64_t count = 1; count <= maximum; ++count)
            {
                auto value = candidate(config, destination, count, slots, topology_value.ranks[lane], topology_value.lanes[lane]);
                if (value.files) table.push_back(value);
            }
            if (table.empty()) table.push_back(candidate(config, destination, 0, slots, topology_value.ranks[lane], topology_value.lanes[lane]));
            for (auto& value : table) value.schedule = schedule + topology_value.resource[lane];
        }
        selected[lane] = tables[lane]->back();
        selected[lane].schedule = schedule + topology_value.resource[lane];
    }
    FarmingPlan result;
    auto best = assess(config, destinations, selected, nullptr, memory); ++result.candidates_evaluated;
    std::vector<bool> swept(destinations.size());
    auto search = [&](std::vector<size_t> const& unit)
    {
        auto chosen = selected; auto original = selected;
        bool improved = false;
        for (auto const& option : *tables[unit.front()])
        {
            bool matched = true;
            for (auto lane : unit)
            {
                auto const& table = *tables[lane];
                auto found = std::lower_bound(table.begin(), table.end(), option.files,
                    [](auto const& entry, uint64_t files) { return entry.files < files; });
                if (found == table.end() || found->files != option.files) { matched = false; break; }
                selected[lane] = *found;
                selected[lane].schedule = schedule + topology_value.resource[lane];
            }
            if (!matched) { selected = original; continue; }
            auto proposed = assess(config, destinations, selected, nullptr, memory); ++result.candidates_evaluated;
            resource_require(result.candidates_evaluated <= maximum_candidates, "farming search exceeds its comparison work bound");
            if (better(config, proposed, best)) { best = std::move(proposed); chosen = selected; improved = true; }
            selected = original;
        }
        selected = chosen;
        return improved;
    };
    // Joint identical-drive sweeps avoid a starting-layout plateau; every
    // balanced file count is compared. Resource-lane moves retain frozen meta
    // identities and refine heterogeneous assignments without sampled ranking.
    std::vector<size_t> independent;
    for (size_t lane = frozen.size(); lane < destinations.size(); ++lane)
        if (topology_value.lanes[lane] == 1 && tables[lane]->front().files) independent.push_back(lane);
    // Match each lane's own capacity forecast at a common file count. Small
    // capacity differences (including the farm manifest) must not split a
    // nearly identical fleet onto a dominated coordinate-search plateau.
    if (independent.size() > 1) search(independent);
    for (size_t first = frozen.size(); first < destinations.size(); ++first)
    {
        if (swept[first] || topology_value.lanes[first] != 1) continue;
        std::vector<size_t> unit;
        for (size_t next = first; next < destinations.size(); ++next)
            if (tables[first] == tables[next]) { unit.push_back(next); swept[next] = true; }
        search(unit);
    }
    bool improved;
    do
    {
        improved = false;
        for (size_t lane = frozen.size(); lane < destinations.size(); ++lane)
            if (search({lane})) improved = true;
    } while (improved);
    // Each retained lane is checked once more as the exact proposed forecast.
    best = assess(config, destinations, selected, nullptr, memory);
    static_cast<FarmingAssessment&>(result) = best;
    result.destinations = std::move(selected);
    result.index_bytes = (uint64_t(1) << (config.k - 6)) * 8;
    result.cpu_utilization = best.cpu_work_seconds_per_sp / (config.harvester_cores * config.challenge_seconds);
    result.worker_utilization = (best.cpu_work_seconds_per_sp + best.io_work_seconds_per_sp) /
        (config.lookup_workers * config.challenge_seconds);
    result.planning_memory_bytes = memory.peak;
    // No empty layout may conceal a nonempty requested destination.
    for (size_t lane = 0; lane < destinations.size(); ++lane)
        if (!result.destinations[lane].files && destinations[lane].max_groups && destinations[lane].max_total_members &&
            destinations[lane].capacity_bytes >= member_bytes(config) + std::ldexp(2.0L, config.k - 6) + 256)
        { result.feasible = false; result.reason = "no encodable farming layout fits the file/family constraints"; }
    return result;
}
}
