#include "host/GroupFarming.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
using namespace group_grow;
void check(bool value, char const* message)
{
    if (!value)
    {
        throw std::runtime_error(message);
    }
}
template <class Error = std::runtime_error, class Function>
void rejects(Function function, char const* message)
{
    bool caught = false;
    try
    {
        function();
    }
    catch (Error const&)
    {
        caught = true;
    }
    check(caught, message);
}
FarmingDestination hdd(uint64_t capacity = 18000000000000ull)
{
    FarmingDestination value;
    value.capacity_bytes = capacity;
    value.known_device = true;
    return value;
}
std::vector<std::vector<uint16_t>> extents(std::vector<FarmingDrivePlan> const& plans)
{
    std::vector<std::vector<uint16_t>> result(plans.size());
    for (size_t lane = 0; lane < plans.size(); ++lane)
    {
        for (uint64_t file = 0; file < plans[lane].files; ++file)
        {
            result[lane].push_back(plans[lane].file_members(file));
        }
    }
    return result;
}
struct Job
{
    size_t lane;
    uint64_t file;
    unsigned members;
    long double arrival = 0;
    unsigned cpu_pattern = 4;
};
struct Observed
{
    long double tail = 0;
    uint64_t late = 0, members = 0;
};
// Independent scheduler: scan active readers for each completion. Cold readers
// read/parse the header, read/decode the index, then read/decode four chunks.
// Chunk CPU work is interleaved, including final chaining in its declared total.
// No policy family sweep or scheduling inequalities are reused here.
Observed simulate(FarmingConfig const& config, std::vector<FarmingDestination> const& destinations,
                  std::vector<Job> const& jobs, uint64_t files,
                  std::span<FarmingConfig const> configs = {}, std::span<uint64_t const> lane_files = {})
{
    struct Device
    {
        long double bandwidth, latency, transfer = 0;
        std::vector<long double> slots;
    };
    struct Running
    {
        size_t job;
        unsigned stage;
        long double time;
        uint64_t sequence;
    };
    std::vector<size_t> resources(destinations.size());
    std::vector<Device> devices;
    for (size_t lane = 0; lane < destinations.size(); ++lane)
    {
        size_t before = lane;
        for (size_t other = 0; other < lane; ++other)
        {
            if (destinations[lane].resource != UINT64_MAX &&
                destinations[lane].resource == destinations[other].resource)
            {
                before = other;
                break;
            }
        }
        auto const profile = farming_storage_profile(destinations[lane]);
        resources[lane] = before == lane ? devices.size() : resources[before];
        if (before == lane)
        {
            devices.push_back(
                {profile.read_bytes_per_second, profile.read_latency_seconds, 0,
                 std::vector<long double>(std::min(config.lookup_workers, profile.read_concurrency))});
        }
        else
        {
            auto& shared = devices[resources[lane]];
            shared.bandwidth = std::min(shared.bandwidth, profile.read_bytes_per_second);
            shared.latency = std::max(shared.latency, profile.read_latency_seconds);
            shared.slots.resize(std::min<size_t>(shared.slots.size(), profile.read_concurrency));
        }
    }
    long double indexes = files * std::ldexp(8.0L, config.k - 6);
    if (!configs.empty())
    {
        check(configs.size() == destinations.size() && configs.size() == lane_files.size(),
              "independent cohort reader count differs");
        indexes = 0;
        for (size_t lane = 0; lane < configs.size(); ++lane)
        {
            indexes += lane_files[lane] * std::ldexp(8.0L, configs[lane].k - 6);
        }
    }
    bool const warm = config.reader == FarmingReader::cached && indexes <= config.index_cache_bytes;
    std::vector<long double> cores(std::min(config.harvester_cores, config.lookup_workers));
    std::vector<Running> running;
    uint64_t sequence = 0;
    size_t next = 0;
    long double now = 0;
    auto schedule = [&](Running& active, unsigned stage)
    {
        auto const& job = jobs[active.job];
        auto const& parameters = configs.empty() ? config : configs[job.lane];
        long double const entries = std::ldexp(1.0L, parameters.k - 6);
        long double end;
        if (stage & 1)
        {
            long double cost;
            if (stage < 4)
            {
                cost = entries * parameters.index_seconds_per_entry * (stage == 1 ? 0.1L : 0.9L);
            }
            else
            {
                unsigned const chunk = (stage - 5) / 2;
                long double const share = job.cpu_pattern == 4       ? 0.25L
                                          : job.cpu_pattern == chunk ? 0.85L
                                                                     : 0.05L;
                cost = job.members * parameters.chain_seconds_per_member * share;
            }
            auto core = std::min_element(cores.begin(), cores.end());
            end = *core = std::max(now, *core) + cost;
        }
        else
        {
            long double const bytes = stage == 0   ? 306
                                      : stage == 2 ? entries * parameters.compressed_index_bytes_per_entry
                                                   : 64 * (parameters.k + 1.45L) / 8 * job.members;
            auto& device = devices[resources[job.lane]];
            auto slot = std::min_element(device.slots.begin(), device.slots.end());
            end = std::max(std::max(now, *slot) + device.latency, device.transfer) + bytes / device.bandwidth;
            *slot = device.transfer = end;
        }
        active.stage = stage;
        active.time = end;
        active.sequence = sequence++;
    };
    Observed result;
    while (next < jobs.size() || !running.empty())
    {
        while (running.size() < config.lookup_workers && next < jobs.size() && jobs[next].arrival <= now)
        {
            Running active{next++, 0, 0, 0};
            schedule(active, warm ? 4 : 0);
            running.push_back(active);
        }
        auto event =
            std::min_element(running.begin(), running.end(), [](Running const& a, Running const& b)
                             { return std::tie(a.time, a.sequence) < std::tie(b.time, b.sequence); });
        if (running.size() < config.lookup_workers && next < jobs.size() &&
            (event == running.end() || jobs[next].arrival < event->time))
        {
            now = jobs[next].arrival;
            continue;
        }
        check(event != running.end(), "independent readers deadlocked");
        now = event->time;
        if (event->stage < 11)
        {
            schedule(*event, event->stage + 1);
            continue;
        }
        auto const job = jobs[event->job];
        result.tail = std::max(result.tail, now - job.arrival);
        result.members += job.members;
        if (now - job.arrival > config.deadline_seconds)
        {
            result.late += job.members;
        }
        running.erase(event);
    }
    return result;
}
std::vector<Job> adversarial(std::vector<FarmingDestination> const& destinations,
                             std::vector<FarmingDrivePlan> const& layouts,
                             std::vector<std::vector<uint16_t>> const& counts)
{
    std::map<std::tuple<size_t, uint64_t, uint64_t>, Job> families;
    for (size_t lane = 0; lane < layouts.size(); ++lane)
    {
        size_t resource = lane;
        for (size_t other = 0; other < lane; ++other)
        {
            if (destinations[lane].resource != UINT64_MAX &&
                destinations[lane].resource == destinations[other].resource)
            {
                resource = other;
                break;
            }
        }
        for (size_t file = 0; file < counts[lane].size(); ++file)
        {
            auto key = std::make_tuple(resource, layouts[lane].schedule, layouts[lane].family(file));
            Job job{lane, file, counts[lane][file]};
            auto found = families.find(key);
            if (found == families.end() || found->second.members < job.members)
            {
                families[key] = job;
            }
        }
    }
    std::vector<Job> jobs;
    for (auto const& item : families)
    {
        jobs.push_back(item.second);
    }
    return jobs;
}
FarmingAssessment forecast(FarmingConfig const& config, std::vector<FarmingDestination> const& drives,
                           std::vector<FarmingDrivePlan> const& plans)
{
    farming_detail::PlanningMemory memory{64ull * 1024 * 1024};
    return farming_detail::assess(config, drives, plans, nullptr, memory);
}
FarmingAssessment forecast_cohorts(FarmingConfig const& config, std::vector<FarmingDestination> const& drives,
                                   std::vector<FarmingDrivePlan> const& plans, std::span<FarmingConfig const> configs,
                                   bool costs = true)
{
    farming_detail::PlanningMemory memory{64ull * 1024 * 1024};
    auto counts = extents(plans);
    return farming_detail::assess(config, drives, plans, &counts, memory, nullptr, configs, costs);
}
void test_layout_and_limits()
{
    FarmingDrivePlan p;
    p.total_members = 100;
    p.files = 6;
    p.group_members = 17;
    p.meta_slots = 4;
    check(p.file_members(4) == 16 && p.file_members(9) == 17, "forecast balance/extension target changed");
    check(p.meta(0) == 0 && p.meta(1) == 128 && p.family(4) == 1, "meta/family mapping changed");
    p.fixed_members = true;
    check(p.file_members(5) == 15, "fixed terminal forecast did not clip");
    p.resource_lanes = 2;
    auto q = p;
    q.resource_rank = 1;
    std::set<std::pair<uint64_t, uint8_t>> used;
    for (uint64_t file = 0; file < 9; ++file)
    {
        check(used.emplace(p.family(file), p.meta(file)).second, "left identity repeats");
        if (file < 4)
        {
            check(used.emplace(q.family(file), q.meta(file)).second, "shared identity repeats");
        }
    }
    rejects([&] { p.family(UINT64_MAX); }, "ordinal overflow accepted");
    for (unsigned bits = 0; bits <= 8; ++bits)
    {
        FarmingDrivePlan schedule;
        schedule.meta_bits = uint8_t(bits);
        schedule.meta_slots = uint16_t(1u << bits);
        std::set<unsigned> masked;
        for (unsigned slot = 0; slot < schedule.meta_slots; ++slot)
        {
            check(masked.insert(schedule.meta(slot) & ((1u << bits) - 1)).second,
                  "distinct metas alias after the forecast filter");
        }
    }
    FarmingConfig c;
    c.meta_slots_override = 16;
    c.future_filter_bits = 3;
    rejects([&] { farming_plan(c, {hdd()}); }, "future meta alias accepted");
    c = {};
    c.strength = 17;
    check(farming_filter_bits(c) == 13 && farming_filter_bits(c, true) == 13,
          "future strength/cap order changed");
    c = {};
    rejects<FarmingResourceError>([&] { farming_plan(c, {hdd()}, 1); }, "planning budget ignored");
    c.cpu_work_weight = 0;
    rejects([&] { farming_plan(c, {hdd()}); }, "zero work weight accepted");
    c = {};
    c.group_members_override = 2;
    auto fixed = farming_plan(c, {hdd()});
    check(!fixed.feasible && fixed.destinations[0].files > 256 &&
              fixed.max_eligible_families_per_resource > 1,
          "fixed size silently truncates requested fill");
    auto bounded = hdd();
    bounded.max_total_members = 10;
    auto limited = farming_plan(c, {bounded});
    check(limited.feasible && limited.destinations[0].total_members == 10, "explicit member limit ignored");
    bounded.max_total_members = UINT64_MAX;
    bounded.max_groups = 5;
    check(farming_plan(c, {bounded}).feasible, "explicit group limit ignored");
    c.group_members_override = 4;
    auto exact = farming_plan(c, {hdd(4957667840ull)});
    check(exact.feasible && exact.destinations[0].files == 2 && exact.destinations[0].total_members == 5,
          "exact terminal one-member capacity boundary was excluded");
    c = {};
    auto empty = hdd(0);
    check(farming_plan(c, {empty}).feasible, "empty capacity is not harmless");
}
void test_adversarial_scheduler()
{
    FarmingConfig c;
    c.deadline_seconds = c.challenge_seconds = 1000;
    c.chain_seconds_per_member = 0.03L;
    c.max_eligible_families_per_resource = 16;
    c.memory_bytes = 4000000000ull;
    auto a = hdd(100000000000ull);
    a.resource = 7;
    a.read_bytes_per_second = 100000000;
    a.read_latency_seconds = 0.02L;
    auto b = a;
    b.read_bytes_per_second = 150000000;
    b.read_latency_seconds = 0.01L;
    auto d = hdd(100000000000ull);
    d.resource = 8;
    d.nonrotating = true;
    std::vector<FarmingDestination> drives{a, b, d};
    std::vector<FarmingDrivePlan> layouts(3);
    for (size_t lane = 0; lane < 3; ++lane)
    {
        layouts[lane].meta_slots = 2;
        layouts[lane].meta_bits = 8;
        layouts[lane].resource_lanes = lane < 2 ? 2 : 1;
        layouts[lane].resource_rank = lane == 1 ? 1 : 0;
        layouts[lane].schedule = lane < 2 ? 0 : 1;
    }
    std::vector<std::vector<uint16_t>> counts{{240, 1, 2, 1, 3}, {1, 2, 1, 2, 1}, {4, 1, 5, 2}};
    // Multiple replacement families outnumber workers. Long and short CPU
    // requests alternate reads/decodes; both cache states and W>C are covered.
    for (auto reader : {FarmingReader::stateless, FarmingReader::cached})
    {
        for (unsigned workers : {1, 2, 4})
        {
            for (unsigned cores : {1, 2, 4})
            {
                c.reader = reader;
                c.lookup_workers = workers;
                c.harvester_cores = cores;
                c.index_cache_bytes = 4000000000ull;
                auto certificate = farming_assess(c, drives, layouts, counts);
                auto jobs = adversarial(drives, layouts, counts);
                for (unsigned order = 0; order < 12; ++order)
                {
                    std::rotate(jobs.begin(), jobs.begin() + 1, jobs.end());
                    if (order & 1)
                    {
                        std::reverse(jobs.begin(), jobs.end());
                    }
                    for (size_t j = 0; j < jobs.size(); ++j)
                    {
                        jobs[j].cpu_pattern = unsigned((j + order) % 5);
                    }
                    auto actual = simulate(c, drives, jobs, 14);
                    check(actual.tail <= certificate.response_bound_seconds,
                          "staged replacement adversary exceeds certificate");
                    if (certificate.feasible)
                    {
                        check(!actual.late, "admitted adversary is late");
                    }
                }
            }
        }
    }
    c = {};
    auto single = farming_plan(c, {hdd(10000000000ull)});
    auto one = extents(single.destinations);
    auto actual =
        simulate(c, {hdd(10000000000ull)}, adversarial({hdd(10000000000ull)}, single.destinations, one),
                 single.destinations[0].files);
    check(single.feasible && std::fabs(single.response_bound_seconds - actual.tail) < 0.000000001L,
          "single family has fabricated queue wait");
    c.deadline_seconds = c.challenge_seconds = 20;
    c.lookup_workers = c.harvester_cores = 2;
    std::vector<FarmingDestination> independent{a, d, hdd(), hdd()};
    std::vector<FarmingDrivePlan> simple(4);
    for (size_t lane = 0; lane < simple.size(); ++lane)
    {
        auto& layout = simple[lane];
        layout.meta_slots = 256;
        layout.meta_bits = 8;
        layout.schedule = lane;
    }
    std::vector<std::vector<uint16_t>> values{{1}, {20}, {3}, {8}};
    auto certificate = farming_assess(c, independent, simple, values);
    auto jobs = adversarial(independent, simple, values);
    do
    {
        check(simulate(c, independent, jobs, 4).tail <= certificate.response_bound_seconds,
              "exhaustive asymmetric admission order exceeds bound");
    } while (std::next_permutation(jobs.begin(), jobs.end(),
                                   [](Job const& x, Job const& y) { return x.lane < y.lane; }));
}
void test_filter_exclusion()
{
    FarmingConfig config;
    for (unsigned files : {1, 10, 36, 71})
    {
        auto layout = farming_detail::candidate(config, hdd(), files, 256, 0, 1);
        check(layout.files == files && layout.families == 1,
              "requested comparison shape lost its shared family");
        for (unsigned bits : {9, 8})
        {
            unsigned const mask = (1u << bits) - 1;
            for (unsigned phase = 0; phase <= mask; ++phase)
            {
                for (unsigned sp = 0; sp < 16; ++sp)
                {
                    unsigned eligible = 0;
                    for (unsigned file = 0; file < files; ++file)
                    {
                        eligible += phase == ((sp ^ layout.meta(file)) & mask);
                    }
                    check(eligible <= 1, "same-family file groups pass the same current/future filter");
                }
            }
        }
    }
}
void test_storage_profiles()
{
    FarmingConfig config;
    auto drive = hdd(20000000000ull);
    auto rotating = farming_plan(config, {drive});
    drive.nonrotating = true;
    auto solid = farming_plan(config, {drive});
    drive.known_device = drive.nonrotating = false;
    auto unknown = farming_plan(config, {drive});
    check(rotating.feasible && solid.feasible && unknown.feasible &&
              solid.response_bound_seconds <= rotating.response_bound_seconds &&
              rotating.response_bound_seconds <= unknown.response_bound_seconds,
          "storage profiles lost their conservative latency ordering");
    check(unknown.max_eligible_families_per_resource == 1,
          "unknown storage bypassed same-resource family exclusion");
}
void test_frozen_expansion()
{
    FarmingConfig config;
    config.k = 18; config.max_eligible_families_per_resource = 4;
    auto a = hdd(64000000), b = a, added = a;
    a.resource = 0; b.resource = 1; added.resource = 2;
    a.max_total_members = b.max_total_members = added.max_total_members = 4;
    auto original = farming_plan(config, {a, b});
    check(original.feasible, "initial expansion fixture is infeasible");
    auto extended = farming_plan(config, {a, b, added}, 64ull << 20, original.destinations);
    check(extended.feasible && extended.destinations[0] == original.destinations[0] &&
        extended.destinations[1] == original.destinations[1], "expansion changed a frozen farming assignment");
    check(extended.destinations[2].schedule > original.destinations[0].schedule &&
        extended.destinations[2].schedule > original.destinations[1].schedule &&
        extended.cpu_work_seconds_per_sp > original.cpu_work_seconds_per_sp,
        "expansion reused a family schedule or omitted existing farm demand");
    added.resource = a.resource;
    auto shared = farming_plan(config, {a, b, added}, 64ull << 20, original.destinations);
    check(shared.feasible && shared.max_eligible_families_per_resource == 2,
        "additional shared-device schedules were not charged independently");
    auto restricted = config; restricted.max_eligible_families_per_resource = 1;
    auto refused = farming_plan(restricted, {a, b, added}, 64ull << 20, original.destinations);
    check(!refused.feasible, "expansion bypassed the combined physical-resource limit");
    b.resource = a.resource;
    auto relocated = farming_assess(config, {a, b}, original.destinations, extents(original.destinations));
    check(relocated.feasible && relocated.max_eligible_families_per_resource == 2,
        "relocated independent families were merged when their disks began sharing I/O");
    auto jobs = adversarial({a, b}, original.destinations, extents(original.destinations));
    check(jobs.size() == 2 && simulate(config, {a, b}, jobs,
        original.destinations[0].files + original.destinations[1].files).tail <= relocated.response_bound_seconds,
        "relocated independent schedules escaped the physical-resource certificate");

    auto together = farming_plan(config, {a, b});
    b.resource = 1;
    auto split = farming_assess(config, {a, b}, together.destinations, extents(together.destinations));
    check(split.feasible && split.max_eligible_families_per_resource == 1 &&
        split.cpu_work_seconds_per_sp == together.cpu_work_seconds_per_sp,
        "relocation split changed the saved identity interleave or omitted existing work");
    auto after_split = farming_plan(config, {a, b, added}, 64ull << 20, together.destinations);
    check(after_split.feasible && after_split.destinations[0] == together.destinations[0] &&
        after_split.destinations[1] == together.destinations[1] &&
        after_split.max_eligible_families_per_resource == 2,
        "expansion after relocation changed frozen assignments or merged independent schedules");
    auto alias = together.destinations;
    alias[1].resource_rank = alias[0].resource_rank;
    rejects([&] { farming_assess(config, {a, b}, alias, extents(alias)); },
        "relocated schedule accepted repeated identity ranks");
    auto exhausted = farming_plan(config, {a}).destinations;
    exhausted[0].schedule = UINT64_MAX - 1;
    b.resource = a.resource; added.resource = 1;
    rejects([&] { farming_plan(config, {a, b, added}, 64ull << 20, exhausted); },
        "reused candidate table bypassed identity schedule exhaustion");
}
void test_extent_admission()
{
    FarmingConfig c;
    auto drive = hdd(50000000000ull);
    auto plan = farming_plan(c, {drive});
    check(plan.feasible, "small farm not certifiable");
    auto counts = extents(plan.destinations);
    auto previous = farming_assess(c, {drive}, plan.destinations, counts);
    for (unsigned step = 0; step < 16; ++step)
    {
        counts[0].push_back(2);
        auto next = farming_assess(c, {drive}, plan.destinations, counts);
        check(next.response_bound_seconds >= previous.response_bound_seconds &&
                  next.dynamic_work_seconds_per_sp >= previous.dynamic_work_seconds_per_sp,
              "extent growth reduced demand");
        previous = next;
    }
    counts[0].resize(257, 2);
    auto exceeded = farming_assess(c, {drive}, plan.destinations, counts);
    check(!exceeded.feasible && exceeded.max_eligible_families_per_resource == 2,
          "extension escaped family admission");
    rejects<FarmingResourceError>([&] { farming_assess(c, {drive}, plan.destinations, counts, 1); },
                                  "live planner budget ignored");
    auto bad = plan.destinations;
    bad[0].meta_base = 1;
    rejects([&] { farming_assess(c, {drive}, bad, counts); }, "changed meta base accepted");
    auto second = drive;
    drive.resource = second.resource = 9;
    auto shared = farming_plan(c, {drive, second});
    auto shared_counts = extents(shared.destinations);
    bad = shared.destinations;
    bad[1].meta_slots = 128;
    rejects([&] { farming_assess(c, {drive, second}, bad, shared_counts); }, "shared schedules alias");
    rejects([&] { farming_plan(c, {drive, second}, 64ull << 20, bad); },
            "frozen shared schedules alias");
    c.reader = FarmingReader::cached;
    c.index_cache_bytes = 8ull * 32 * 1024 * 1024;
    c.memory_bytes = 4000000000ull;
    counts[0].resize(8, 2);
    auto warm = farming_assess(c, {hdd()}, plan.destinations, counts);
    counts[0].push_back(2);
    auto cold = farming_assess(c, {hdd()}, plan.destinations, counts);
    check(cold.response_bound_seconds >= warm.response_bound_seconds, "cache growth reduced bound");
    counts[0].resize(farming_detail::maximum_files + 1, 2);
    bool bounded = false;
    try
    {
        farming_assess(c, {hdd()}, plan.destinations, counts);
    }
    catch (FarmingResourceError const& error)
    {
        bounded = !error.memory_budget;
    }
    check(bounded, "permanent file work limit was not distinguished from RAM pressure");
    c = {};
    c.deadline_seconds = 20;
    c.challenge_seconds = 0.1L;
    auto cadence = farming_assess(c, {hdd()}, plan.destinations, {{2}});
    check(!cadence.feasible && cadence.response_bound_seconds > c.challenge_seconds,
          "response deadline bypassed the no-carry cadence requirement");
    c = {};
    c.chain_seconds_per_member = std::numeric_limits<long double>::max() / 4;
    auto overflow = farming_assess(c, {hdd()}, plan.destinations, {{20}});
    check(!overflow.feasible && !overflow.reason.empty(), "nonfinite service work was admitted");
}
void test_planning_memory_boundaries()
{
    FarmingConfig config;
    config.k = 18;
    auto drive = hdd(64000000);
    drive.max_total_members = 4;
    drive.max_groups = 2;
    for (bool shared : {false, true})
    {
        std::vector<FarmingDestination> drives(16, drive);
        if (shared) for (auto& destination : drives) destination.resource = 7;
        auto original = farming_plan(config, drives);
        auto exact = farming_plan(config, drives, original.planning_memory_bytes);
        check(exact.destinations == original.destinations && exact.feasible == original.feasible &&
                  exact.response_bound_seconds == original.response_bound_seconds &&
                  exact.dynamic_work_seconds_per_sp == original.dynamic_work_seconds_per_sp &&
                  exact.candidates_evaluated == original.candidates_evaluated &&
                  exact.planning_memory_bytes == original.planning_memory_bytes,
              "exact planning memory budget changed the search");
        rejects<FarmingResourceError>([&] { farming_plan(config, drives, original.planning_memory_bytes - 1); },
                                      "planning memory boundary was bypassed");
        auto counts = extents(original.destinations);
        auto live = farming_assess(config, drives, original.destinations, counts);
        auto bounded = farming_assess(config, drives, original.destinations, counts, live.planning_memory_bytes);
        check(bounded.feasible == live.feasible && bounded.response_bound_seconds == live.response_bound_seconds &&
                  bounded.dynamic_work_seconds_per_sp == live.dynamic_work_seconds_per_sp &&
                  bounded.planning_memory_bytes == live.planning_memory_bytes,
              "exact assessment memory budget changed the certificate");
        rejects<FarmingResourceError>([&] {
            farming_assess(config, drives, original.destinations, counts, live.planning_memory_bytes - 1);
        }, "assessment memory boundary was bypassed");
    }
}
void test_cohort_costs()
{
    FarmingConfig shared;
    shared.lookup_workers = 2;
    auto small = shared;
    small.k = 18; small.strength = 4;
    small.chain_seconds_per_member = 0.001L;
    small.index_seconds_per_entry = 0.00000001L;
    small.compressed_index_bytes_per_entry = 2;
    small.lookup_memory_per_member = 512;
    std::vector<FarmingConfig> configs{shared, small, small, small};
    std::vector<FarmingDestination> drives(configs.size(), hdd(10000000000ull));
    std::vector<FarmingDrivePlan> plans;
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        drives[lane].max_total_members = 2; drives[lane].max_groups = 1; drives[lane].resource = lane;
        auto plan = farming_plan(configs[lane], {drives[lane]});
        check(plan.feasible && plan.destinations[0].files == 1, "cohort fixture is infeasible");
        plan.destinations[0].schedule = lane;
        plans.push_back(plan.destinations[0]);
    }
    auto mixed = forecast_cohorts(shared, drives, plans, configs);
    auto uniform_configs = std::vector<FarmingConfig>(configs.size(), shared);
    auto uniform = forecast_cohorts(shared, drives, plans, uniform_configs);
    auto counts = extents(plans);
    auto original = farming_assess(shared, drives, plans, counts);
    check(uniform.feasible == original.feasible && uniform.response_bound_seconds == original.response_bound_seconds &&
              uniform.dynamic_work_seconds_per_sp == original.dynamic_work_seconds_per_sp &&
              uniform.reader_memory_bytes == original.reader_memory_bytes &&
              uniform.planning_memory_bytes == original.planning_memory_bytes,
          "equal-shape cohorts changed the existing certificate");
    auto legacy = forecast_cohorts(shared, drives, plans, configs, false);
    check(legacy.response_bound_seconds == original.response_bound_seconds &&
              legacy.dynamic_work_seconds_per_sp == original.dynamic_work_seconds_per_sp &&
              legacy.reader_memory_bytes == original.reader_memory_bytes,
          "legacy cohort assessment changed its envelope costs");
    check(mixed.feasible && mixed.response_bound_seconds < uniform.response_bound_seconds &&
              mixed.cpu_work_seconds_per_sp < uniform.cpu_work_seconds_per_sp &&
              mixed.io_work_seconds_per_sp < uniform.io_work_seconds_per_sp &&
              mixed.reader_memory_bytes < uniform.reader_memory_bytes,
          "mixed shapes still pay the worst cohort costs throughout the farm");
    long double expected_cpu = 0, expected_io = 0;
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        auto one = forecast_cohorts(shared, {drives[lane]}, {plans[lane]}, std::span(configs).subspan(lane, 1));
        expected_cpu += one.cpu_work_seconds_per_sp; expected_io += one.io_work_seconds_per_sp;
    }
    check(mixed.cpu_work_seconds_per_sp == expected_cpu && mixed.io_work_seconds_per_sp == expected_io,
          "joint expected work changed an individual cohort's filter probability or service cost");
    farming_detail::PlanningMemory forecast_memory{64ull * 1024 * 1024};
    auto projected = farming_detail::assess(shared, drives, plans, nullptr, forecast_memory, nullptr, configs);
    check(projected.feasible == mixed.feasible && projected.response_bound_seconds == mixed.response_bound_seconds &&
              projected.dynamic_work_seconds_per_sp == mixed.dynamic_work_seconds_per_sp &&
              projected.reader_memory_bytes == mixed.reader_memory_bytes,
          "cohort forecast differs from the exact extent assessment");
    shared.memory_bytes = mixed.reader_memory_bytes;
    check(forecast_cohorts(shared, drives, plans, configs).feasible &&
              !forecast_cohorts(shared, drives, plans, uniform_configs).feasible,
          "mixed-shape memory advantage was not usable within the declared budget");
    --shared.memory_bytes;
    auto memory_failure = forecast_cohorts(shared, drives, plans, configs);
    check(!memory_failure.feasible && memory_failure.reason.find(std::to_string(mixed.reader_memory_bytes)) != std::string::npos &&
              memory_failure.reason.find(std::to_string(shared.memory_bytes)) != std::string::npos,
          "joint working-memory failure omitted its requirement or budget");
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        check(forecast_cohorts(shared, {drives[lane]}, {plans[lane]}, std::span(configs).subspan(lane, 1)).feasible,
              "joint memory fixture does not fit individually");
    }
    shared = {};
    shared.harvester_cores = shared.lookup_workers = 1;
    auto joint = forecast_cohorts(shared, drives, plans, configs);
    long double single = 0;
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        auto one = forecast_cohorts(shared, {drives[lane]}, {plans[lane]}, std::span(configs).subspan(lane, 1));
        single = std::max(single, one.response_bound_seconds);
    }
    check(joint.response_bound_seconds > single, "joint CPU/deadline fixture does not share worker demand");
    shared.deadline_seconds = (single + joint.response_bound_seconds) / 2;
    auto deadline_failure = forecast_cohorts(shared, drives, plans, configs);
    check(!deadline_failure.feasible && deadline_failure.reason.find("declared response deadline") != std::string::npos &&
              deadline_failure.reason.find(farming_detail::service_amount(shared.deadline_seconds)) != std::string::npos,
          "joint deadline failure omitted its declared limit");
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        check(forecast_cohorts(shared, {drives[lane]}, {plans[lane]}, std::span(configs).subspan(lane, 1)).feasible,
              "joint deadline fixture does not fit individually");
    }
    shared.deadline_seconds = 20;
    shared.challenge_seconds = (single + joint.response_bound_seconds) / 2;
    auto cadence_failure = forecast_cohorts(shared, drives, plans, configs);
    check(!cadence_failure.feasible && cadence_failure.reason.find("signage-point interval") != std::string::npos,
          "heterogeneous batch escaped the shared signage-point cadence");
    shared = {};
    for (auto& drive : drives) drive.resource = 9;
    auto family_failure = forecast_cohorts(shared, drives, plans, configs);
    check(!family_failure.feasible && family_failure.max_eligible_families_per_resource == configs.size() &&
              family_failure.reason.find("storage resource 9") != std::string::npos &&
              family_failure.reason.find(std::to_string(configs.size())) != std::string::npos,
          "independent cohorts escaped shared-resource family admission");
    shared.max_eligible_families_per_resource = configs.size();
    shared.reader = FarmingReader::cached;
    long double indexes = 0;
    for (auto const& parameters : configs) indexes += std::ldexp(8.0L, parameters.k - 6);
    shared.index_cache_bytes = farming_detail::size(indexes);
    auto warm = forecast_cohorts(shared, drives, plans, configs);
    auto envelope = forecast_cohorts(shared, drives, plans, uniform_configs);
    --shared.index_cache_bytes;
    auto cold = forecast_cohorts(shared, drives, plans, configs);
    check(warm.feasible && warm.response_bound_seconds < cold.response_bound_seconds &&
              warm.cpu_work_seconds_per_sp < cold.cpu_work_seconds_per_sp &&
              warm.cpu_work_seconds_per_sp < envelope.cpu_work_seconds_per_sp,
          "shared cache did not use the cohorts' actual expanded index sizes");
    // Different metas in one identity schedule remain mutually exclusive
    // when their declared service costs differ at the same plot parameters.
    drives.resize(2); plans.resize(2); configs.resize(2);
    configs[1].k = configs[0].k; configs[1].strength = configs[0].strength;
    for (size_t lane = 0; lane < plans.size(); ++lane)
    {
        plans[lane].schedule = 0; plans[lane].resource_rank = uint16_t(lane); plans[lane].resource_lanes = uint16_t(plans.size());
    }
    shared = {};
    auto exclusive = forecast_cohorts(shared, drives, plans, configs);
    check(exclusive.feasible && exclusive.max_eligible_families_per_resource == 1,
          "differently calibrated metas lost their family exclusion");
    configs[1].meta_group_override = plans[1].meta_base = 1;
    rejects([&] { forecast_cohorts(shared, drives, plans, configs); },
            "heterogeneous family schedules accepted incompatible meta identities");
    configs[1].meta_group_override = plans[1].meta_base = 0;
    configs[1].k = 26;
    rejects([&] { forecast_cohorts(shared, drives, plans, configs); },
            "cohort family exclusion accepted different k values in one identity schedule");
    configs[1].k = configs[0].k; configs[1].strength = 3;
    rejects([&] { forecast_cohorts(shared, drives, plans, configs); },
            "cohort family exclusion accepted different strengths in one identity schedule");
    // Explicit equal filters can make different strengths have identical
    // modeled service costs. The homogeneous fast path must reject that
    // identity mismatch as well as the heterogeneous cost path.
    shared.filter_bits = 9; shared.future_filter_bits = 8;
    configs = {shared, shared}; configs[1].strength = 3;
    rejects([&] { forecast_cohorts(shared, drives, plans, configs); },
            "equal service costs concealed incompatible strength identities");
}
void test_cohort_scheduler()
{
    FarmingConfig shared;
    shared.max_eligible_families_per_resource = 4;
    auto small = shared;
    small.k = 18; small.strength = 4;
    small.chain_seconds_per_member = 0.001L;
    small.index_seconds_per_entry = 0.00000001L;
    small.compressed_index_bytes_per_entry = 2;
    std::vector<FarmingConfig> configs{shared, small, small, small};
    std::vector<FarmingDestination> drives(configs.size(), hdd(10000000000ull));
    std::vector<FarmingDrivePlan> plans;
    std::vector<uint64_t> files(configs.size(), 1);
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        drives[lane].max_total_members = 2; drives[lane].max_groups = 1;
        auto plan = farming_plan(configs[lane], {drives[lane]});
        plan.destinations[0].schedule = lane;
        plans.push_back(plan.destinations[0]);
    }
    for (bool combined : {false, true})
    {
        for (size_t lane = 0; lane < drives.size(); ++lane) drives[lane].resource = combined ? 9 : lane;
        for (auto reader : {FarmingReader::stateless, FarmingReader::cached})
        {
            shared.reader = reader; shared.index_cache_bytes = 134217728;
            for (unsigned cores : {1u, 2u, 4u})
            {
                shared.harvester_cores = cores;
                for (unsigned workers : {1u, 2u, 4u})
                {
                    shared.lookup_workers = workers;
                    auto certificate = forecast_cohorts(shared, drives, plans, configs);
                    std::vector<size_t> order{0, 1, 2, 3};
                    do
                    {
                        for (unsigned pattern = 0; pattern <= 4; ++pattern)
                        {
                            std::vector<Job> jobs;
                            for (auto lane : order) jobs.push_back({lane, 0, 2, 0, unsigned((lane + pattern) % 5)});
                            auto actual = simulate(shared, drives, jobs, configs.size(), configs, files);
                            check(actual.tail <= certificate.response_bound_seconds,
                                  "mixed-shape independent scheduler exceeded the joint certificate");
                        }
                    } while (std::next_permutation(order.begin(), order.end()));
                }
            }
        }
    }
}
void test_cohort_replacement_scheduler()
{
    FarmingConfig shared;
    shared.deadline_seconds = shared.challenge_seconds = 1000000;
    shared.max_eligible_families_per_resource = 1000000;
    shared.memory_bytes = 4000000000ull;
    std::vector<FarmingConfig> configs(4, shared);
    configs[0].chain_seconds_per_member = 0.08L;
    configs[0].index_seconds_per_entry = 0.00000001L;
    configs[0].compressed_index_bytes_per_entry = 2;
    configs[0].lookup_memory_per_member = 65536;
    configs[1].chain_seconds_per_member = 0.00001L;
    configs[1].index_seconds_per_entry = 0.000004L;
    configs[2].strength = 4;
    configs[2].chain_seconds_per_member = 0.000001L;
    configs[2].index_seconds_per_entry = 0;
    configs[2].compressed_index_bytes_per_entry = 0.5L;
    configs[3].k = 18; configs[3].strength = 5;
    configs[3].chain_seconds_per_member = 0.0001L;
    configs[3].index_seconds_per_entry = 0.00000001L;
    configs[3].compressed_index_bytes_per_entry = 2;
    std::vector<FarmingDestination> drives(4, hdd());
    drives[0].resource = drives[1].resource = drives[3].resource = 9;
    drives[2].resource = 10;
    drives[0].read_bytes_per_second = 250000; drives[0].read_latency_seconds = 0.02L; drives[0].read_concurrency = 1;
    drives[1].read_bytes_per_second = 500000; drives[1].read_latency_seconds = 0.005L; drives[1].read_concurrency = 4;
    drives[2].read_bytes_per_second = 50000; drives[2].read_latency_seconds = 0.001L; drives[2].read_concurrency = 2;
    drives[3].read_bytes_per_second = 200000; drives[3].read_latency_seconds = 0.04L; drives[3].read_concurrency = 2;
    std::vector<std::vector<uint16_t>> counts{{240, 1, 2, 1, 3}, {1, 32, 1, 24},
                                             {72, 1, 48, 2, 96, 3}, {2, 1, 2, 2, 1, 2, 1, 2, 2}};
    std::vector<FarmingDrivePlan> plans(4);
    std::vector<uint64_t> files;
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>, std::vector<Job>> choices;
    for (size_t lane = 0; lane < plans.size(); ++lane)
    {
        auto& plan = plans[lane];
        plan.schedule = lane < 2 ? 0 : lane - 1;
        plan.resource_rank = lane == 1 ? 1 : 0; plan.resource_lanes = lane < 2 ? 2 : 1;
        plan.meta_slots = lane == 3 ? 4 : 2;
        files.push_back(counts[lane].size());
        for (size_t file = 0; file < counts[lane].size(); ++file)
        {
            choices[{drives[lane].resource, plan.schedule, plan.family(file)}].push_back({lane, file, counts[lane][file]});
        }
    }
    // Each selection chooses one possible eligible meta per family. Both
    // overlapping lanes, asymmetric independent families, replacement jobs,
    // admission orders and interleaved chunk CPU patterns vary independently.
    for (auto reader : {FarmingReader::stateless, FarmingReader::cached})
    {
        shared.reader = reader; shared.index_cache_bytes = 4000000000ull;
        for (unsigned cores : {1u, 2u, 4u})
        {
            shared.harvester_cores = cores;
            for (unsigned workers : {1u, 2u, 4u, 8u})
            {
                shared.lookup_workers = workers;
                farming_detail::PlanningMemory memory{64ull * 1024 * 1024};
                auto certificate = farming_detail::assess(shared, drives, plans, &counts, memory, nullptr, configs);
                check(certificate.feasible, "asymmetric cohort scheduler fixture is infeasible");
                for (unsigned selection = 0; selection < 64; ++selection)
                {
                    std::vector<Job> jobs; size_t family = 0;
                    for (auto const& item : choices)
                    {
                        auto const& alternatives = item.second;
                        jobs.push_back(alternatives[(selection >> (family++ % 5)) % alternatives.size()]);
                    }
                    std::rotate(jobs.begin(), jobs.begin() + selection % jobs.size(), jobs.end());
                    if (selection & 1) std::reverse(jobs.begin(), jobs.end());
                    for (unsigned pattern = 0; pattern <= 4; ++pattern)
                    {
                        for (size_t job = 0; job < jobs.size(); ++job)
                            jobs[job].cpu_pattern = unsigned((job + selection + pattern) % 5);
                        auto actual = simulate(shared, drives, jobs, 0, configs, files);
                        check(!actual.late && actual.tail <= certificate.response_bound_seconds,
                              "asymmetric mixed-shape replacement scheduler exceeded its certificate");
                    }
                }
            }
        }
    }
}
void test_large_cohort_assessment()
{
    FarmingConfig shared;
    shared.harvester_cores = shared.lookup_workers = 16;
    shared.memory_bytes = 4000000000ull;
    shared.max_eligible_families_per_resource = 1000000;
    shared.deadline_seconds = shared.challenge_seconds = 1000000;
    std::vector<FarmingDestination> drives(1024, hdd());
    std::vector<FarmingDrivePlan> plans(drives.size());
    std::vector<FarmingConfig> configs(drives.size(), shared);
    std::vector<std::vector<uint16_t>> counts(drives.size());
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        auto const cohort = lane / 8;
        auto const shape = (cohort + cohort / 64) % 4;
        drives[lane].resource = cohort % 64;
        auto& plan = plans[lane];
        plan.schedule = cohort; plan.resource_rank = uint16_t(lane % 8); plan.resource_lanes = 8; plan.meta_slots = 256;
        configs[lane].k = uint8_t(28 - 2 * shape);
        configs[lane].strength = uint8_t(2 + shape);
        configs[lane].chain_seconds_per_member /= 1 + lane % 4;
        for (unsigned file = 0; file < 128; ++file) counts[lane].push_back(uint16_t(1 + (file + lane) % 128));
    }
    auto start = std::chrono::steady_clock::now();
    farming_detail::PlanningMemory memory{64ull << 20};
    auto original = farming_detail::assess(shared, drives, plans, &counts, memory, nullptr, configs);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    check(original.feasible && original.planning_memory_bytes <= memory.limit,
          "large multi-cohort assessment exceeded the existing planning budget");
    farming_detail::PlanningMemory exact_memory{original.planning_memory_bytes};
    auto exact = farming_detail::assess(shared, drives, plans, &counts, exact_memory, nullptr, configs);
    check(exact.feasible && exact.response_bound_seconds == original.response_bound_seconds &&
              exact.planning_memory_bytes == original.planning_memory_bytes,
          "exact large-cohort planning memory changed the certificate");
    rejects<FarmingResourceError>([&]
    {
        farming_detail::PlanningMemory short_memory{original.planning_memory_bytes - 1};
        farming_detail::assess(shared, drives, plans, &counts, short_memory, nullptr, configs);
    }, "large-cohort assessment bypassed its declared planning budget");
    for (auto& lane : counts)
        for (size_t file = 0; file < lane.size(); file += 2) std::swap(lane[file], lane[file + 1]);
    farming_detail::PlanningMemory reordered_memory{64ull << 20};
    auto reordered = farming_detail::assess(shared, drives, plans, &counts, reordered_memory, nullptr, configs);
    check(reordered.feasible && reordered.response_bound_seconds == original.response_bound_seconds &&
              reordered.dynamic_work_seconds_per_sp == original.dynamic_work_seconds_per_sp &&
              reordered.reader_memory_bytes == original.reader_memory_bytes &&
              reordered.max_eligible_families_per_resource == original.max_eligible_families_per_resource,
          "large cohort lane/member churn changed an unchanged family demand");
    start = std::chrono::steady_clock::now();
    farming_detail::PlanningMemory legacy_memory{64ull << 20};
    auto legacy = farming_detail::assess(shared, drives, plans, &counts, legacy_memory, nullptr, configs, false);
    auto legacy_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    check(legacy.feasible && legacy.max_eligible_families_per_resource == original.max_eligible_families_per_resource,
          "large cohort assessment changed the legacy physical family union");
    std::cout << "large_cohort_assessment planning_bytes=" << original.planning_memory_bytes << " elapsed_ms=" << elapsed
              << " legacy_planning_bytes=" << legacy.planning_memory_bytes << " legacy_elapsed_ms=" << legacy_elapsed << '\n';
}
void test_numerical_adjustments()
{
    FarmingConfig shared; shared.lookup_workers = 2; shared.harvester_cores = 1;
    auto small = shared; small.k = 18; small.strength = 4;
    small.chain_seconds_per_member = 0.001L; small.index_seconds_per_entry = 0.00000001L;
    small.compressed_index_bytes_per_entry = 2; small.lookup_memory_per_member = 512;
    std::vector<FarmingConfig> configs{shared, small, small, small}; auto const original_configs = configs;
    std::vector<FarmingDestination> drives(4, hdd());
    std::vector<FarmingDrivePlan> plans(4);
    std::vector<std::vector<uint16_t>> counts(4, {2});
    for (size_t lane = 0; lane < drives.size(); ++lane)
    {
        drives[lane].resource = 9; plans[lane].schedule = lane; plans[lane].meta_slots = 256;
    }
    shared.memory_bytes = 1; shared.deadline_seconds = 0.001L; shared.challenge_seconds = 0.002L;
    auto original = shared;
    auto suggestions = farming_adjustments(shared, drives, plans, counts, 64ull << 20, configs, true, 8);
    auto const& need = suggestions.requirements;
    check(need.memory_limited && need.eligible_limited && need.deadline_limited && need.cadence_limited && !need.overflow &&
              suggestions.evaluations <= 8 && configs == original_configs && shared == original,
          "numerical diagnostics lost a joint constraint, exceeded work or modified declared inputs");
    bool memory = false, eligible = false, deadline = false;
    for (auto const& alternative : suggestions.alternatives)
    {
        farming_detail::PlanningMemory scratch{64ull << 20};
        auto actual = farming_detail::assess(alternative.config, drives, plans, &counts, scratch, nullptr, configs);
        check(actual.feasible == alternative.assessment.feasible && actual.response_bound_seconds == alternative.assessment.response_bound_seconds &&
                  actual.reader_memory_bytes == alternative.assessment.reader_memory_bytes &&
                  actual.dynamic_work_seconds_per_sp == alternative.assessment.dynamic_work_seconds_per_sp,
              "offered alternative differs from the same full joint assessment");
        auto unchanged = alternative.config;
        unchanged.memory_bytes = shared.memory_bytes; unchanged.max_eligible_families_per_resource = shared.max_eligible_families_per_resource;
        unchanged.deadline_seconds = shared.deadline_seconds; unchanged.harvester_cores = shared.harvester_cores;
        unchanged.lookup_workers = shared.lookup_workers;
        check(unchanged == shared, "budget advice changed a cohort service envelope, cache or plot identity");
        if (alternative.kind == FarmingAdjustmentKind::memory)
        {
            memory = true;
            check(alternative.config.memory_bytes == need.memory_bytes && !alternative.remaining.memory_limited &&
                      alternative.remaining.eligible_limited && alternative.remaining.deadline_limited && !actual.feasible,
                  "memory advice hid remaining family/deadline failures or used an arbitrary budget");
        }
        else if (alternative.kind == FarmingAdjustmentKind::eligible_families)
        {
            eligible = true;
            check(alternative.config.max_eligible_families_per_resource == need.eligible_families_per_resource &&
                      !alternative.remaining.eligible_limited && alternative.remaining.memory_limited && !actual.feasible,
                  "family advice hid remaining memory failure or exceeded the needed limit");
        }
        else if (alternative.kind == FarmingAdjustmentKind::deadline)
        {
            deadline = true;
            check(alternative.config.deadline_seconds == need.response_seconds && !alternative.remaining.deadline_limited &&
                      alternative.remaining.cadence_limited && !actual.feasible,
                  "deadline advice incorrectly certified an undrainable signage-point batch");
        }
    }
    check(memory && eligible && deadline, "numerical diagnostics omitted an exact required budget");
    auto limited = farming_adjustments(shared, drives, plans, counts, 64ull << 20, configs, true, 1);
    check(limited.evaluations == 1 && limited.search_limited && limited.alternatives.empty() &&
              limited.baseline.response_bound_seconds == suggestions.baseline.response_bound_seconds,
          "bounded diagnostic search hid its exhaustion or changed the baseline");
    rejects<FarmingResourceError>([&] { farming_adjustments(shared, drives, plans, counts, 1, configs); },
                                  "diagnostic alternatives ignored their planning-memory budget");
    shared = {}; shared.memory_bytes = 4000000000ull;
    shared.harvester_cores = 1; shared.lookup_workers = 4;
    for (size_t lane = 0; lane < drives.size(); ++lane) drives[lane].resource = lane;
    auto assess = [&](FarmingConfig const& proposed)
    {
        farming_detail::PlanningMemory scratch{64ull << 20};
        return farming_detail::assess(proposed, drives, plans, &counts, scratch, nullptr, configs);
    };
    auto slow = assess(shared); auto parallel = shared; parallel.harvester_cores = parallel.lookup_workers;
    auto fast = assess(parallel);
    check(fast.response_bound_seconds < slow.response_bound_seconds, "core adjustment fixture has no CPU interference");
    shared.deadline_seconds = (slow.response_bound_seconds + fast.response_bound_seconds) / 2;
    auto cores = farming_adjustments(shared, drives, plans, counts, 64ull << 20, configs);
    auto core_choice = std::find_if(cores.alternatives.begin(), cores.alternatives.end(),
        [](FarmingAdjustment const& value) { return value.kind == FarmingAdjustmentKind::cpu_cores; });
    check(core_choice != cores.alternatives.end() && core_choice->assessment.feasible && core_choice->config.lookup_workers == shared.lookup_workers &&
              core_choice->config.harvester_cores > shared.harvester_cores && assess(core_choice->config).feasible,
          "numerical core proposal did not pass the complete joint certificate");
    shared.harvester_cores = shared.lookup_workers = 1; shared.deadline_seconds = 8;
    slow = assess(shared); parallel = shared; parallel.harvester_cores = parallel.lookup_workers = 4; fast = assess(parallel);
    check(fast.response_bound_seconds < slow.response_bound_seconds, "worker adjustment fixture has no queue delay");
    shared.deadline_seconds = (slow.response_bound_seconds + fast.response_bound_seconds) / 2;
    auto workers = farming_adjustments(shared, drives, plans, counts, 64ull << 20, configs);
    auto worker_choice = std::find_if(workers.alternatives.begin(), workers.alternatives.end(),
        [](FarmingAdjustment const& value) { return value.kind == FarmingAdjustmentKind::lookup_workers; });
    check(worker_choice != workers.alternatives.end() && worker_choice->assessment.feasible && worker_choice->config.lookup_workers > shared.lookup_workers &&
              assess(worker_choice->config).feasible && configs == original_configs,
          "numerical worker proposal failed joint assessment or mutated old cohort parameters");
}
void test_large_frozen_expansion()
{
    FarmingConfig config;
    config.k = 18;
    config.group_members_override = 2;
    config.harvester_cores = config.lookup_workers = 1024;
    config.memory_bytes = 4000000000ull;
    config.max_eligible_families_per_resource = 16;
    auto drive = hdd(64000000);
    drive.max_total_members = 2;
    drive.max_groups = 1;
    std::vector<FarmingDestination> original_drives(512, drive), drives(1024, drive);
    auto original = farming_plan(config, original_drives);
    for (size_t lane = 0; lane < drives.size(); ++lane) drives[lane].resource = lane % 64;
    auto expanded = farming_plan(config, drives, 64ull << 20, original.destinations);
    check(original.feasible && expanded.feasible && expanded.destinations.size() == drives.size() &&
              std::equal(original.destinations.begin(), original.destinations.end(), expanded.destinations.begin()),
          "large expansion changed its frozen assignments");
    auto counts = extents(expanded.destinations);
    auto live = farming_assess(config, drives, expanded.destinations, counts);
    check(live.feasible && live.response_bound_seconds == expanded.response_bound_seconds &&
              live.cpu_work_seconds_per_sp == expanded.cpu_work_seconds_per_sp &&
              live.io_work_seconds_per_sp == expanded.io_work_seconds_per_sp &&
              live.max_eligible_families_per_resource == expanded.max_eligible_families_per_resource,
          "large relocated expansion differs from its live certificate");
}
void test_explicit_family_tradeoff()
{
    FarmingConfig config;
    config.filter_bits = config.future_filter_bits = 2;
    config.reader = FarmingReader::cached;
    config.index_cache_bytes = config.memory_bytes = 4000000000ull;
    config.objective = FarmingObjective::latency;
    auto drive = hdd(20000000000ull);
    drive.nonrotating = true;
    auto exclusive = farming_plan(config, {drive});
    config.max_eligible_families_per_resource = 4;
    auto parallel = farming_plan(config, {drive});
    check(exclusive.feasible && parallel.feasible && exclusive.max_eligible_families_per_resource == 1 &&
              parallel.max_eligible_families_per_resource > 1 &&
              parallel.response_bound_seconds < exclusive.response_bound_seconds,
          "explicit family concurrency did not produce an admissible latency tradeoff");
    auto counts = extents(parallel.destinations);
    auto jobs = adversarial({drive}, parallel.destinations, counts);
    auto actual = simulate(config, {drive}, jobs, parallel.destinations[0].files);
    check(!actual.late && actual.tail <= parallel.response_bound_seconds,
          "explicit family concurrency escaped the hard deadline");
    config.io_work_weight = 7;
    auto weighted = farming_plan(config, {drive});
    check(weighted.dynamic_work_seconds_per_sp ==
              weighted.cpu_work_seconds_per_sp + config.io_work_weight * weighted.io_work_seconds_per_sp,
          "reported work proxy ignored its explicit relative weight");
    // More directory lanes than meta slots require sparse-family accounting.
    config = {};
    config.meta_slots_override = 1;
    config.max_eligible_families_per_resource = 8;
    config.memory_bytes = 4000000000ull;
    std::vector<FarmingDestination> drives(3, hdd());
    std::vector<FarmingDrivePlan> layouts(3);
    std::vector<std::vector<uint16_t>> sparse{{2, 2}, {2}, {1, 1, 1}};
    for (size_t lane = 0; lane < layouts.size(); ++lane)
    {
        drives[lane].resource = 7;
        auto& layout = layouts[lane];
        layout.resource_lanes = 3;
        layout.resource_rank = uint16_t(lane);
        layout.meta_slots = 1;
        layout.meta_bits = 8;
        layout.files = sparse[lane].size();
        layout.group_members = sparse[lane][0];
        for (auto members : sparse[lane])
        {
            layout.total_members += members;
        }
    }
    auto live = farming_assess(config, drives, layouts, sparse);
    auto projected = forecast(config, drives, layouts);
    check(live.max_eligible_families_per_resource == 6 &&
              live.response_bound_seconds == projected.response_bound_seconds,
          "sparse interleaved family accounting invented or omitted work");
}
void test_sparse_automatic_search()
{
    for (uint8_t bits : {0, 1})
    {
        FarmingConfig config;
        config.filter_bits = config.future_filter_bits = bits;
        std::vector<FarmingDestination> drives(3, hdd(0));
        for (auto& drive : drives)
        {
            drive.resource = 7;
        }
        drives[2].capacity_bytes = 2000000000ull;
        auto selected = farming_plan(config, drives);
        check(selected.feasible && selected.destinations[2].files == 1 &&
                  selected.destinations[2].family(0) > 0 &&
                  selected.max_eligible_families_per_resource == 1,
              "sparse high-rank lane was pruned by its family ordinal");
        auto live = farming_assess(config, drives, selected.destinations, extents(selected.destinations));
        check(live.feasible && live.max_eligible_families_per_resource == 1,
              "sparse automatic forecast differs from its occupied-family certificate");
        drives[1].capacity_bytes = 2000000000ull;
        auto shared = farming_plan(config, drives);
        check(!shared.feasible && shared.max_eligible_families_per_resource == 2,
              "sparse automatic pruning bypassed the cross-lane family limit");
    }
}
void test_small_exhaustive()
{
    FarmingConfig c;
    c.filter_bits = c.future_filter_bits = 2;
    c.harvester_cores = c.lookup_workers = 2;
    c.deadline_seconds = 0.51L;
    auto drive = hdd(28000000000ull);
    std::vector<FarmingDestination> drives{drive, drive};
    for (auto objective : {FarmingObjective::energy, FarmingObjective::latency})
    {
        c.objective = objective;
        auto selected = farming_plan(c, drives);
        for (unsigned first = 1; first <= 4; ++first)
        {
            for (unsigned second = 1; second <= 4; ++second)
            {
                auto a = farming_detail::candidate(c, drives[0], first, 4, 0, 1),
                     b = farming_detail::candidate(c, drives[1], second, 4, 0, 1);
                b.schedule = 1;
                check(!farming_detail::better(c, forecast(c, drives, {a, b}), selected),
                      "small exhaustive assignment dominates search");
            }
        }
        drives[0].capacity_bytes -= 4096;
        drives[1].capacity_bytes = 19000000000ull;
        selected = farming_plan(c, drives);
        for (unsigned first = 1; first <= 4; ++first)
        {
            for (unsigned second = 1; second <= 4; ++second)
            {
                auto a = farming_detail::candidate(c, drives[0], first, 4, 0, 1),
                     b = farming_detail::candidate(c, drives[1], second, 4, 0, 1);
                b.schedule = 1;
                check(!farming_detail::better(c, forecast(c, drives, {a, b}), selected),
                      "mixed exhaustive assignment dominates search");
            }
        }
        drives[0] = drive;
        drives[1] = drive;
    }
}
void test_cached_search_boundary()
{
    FarmingConfig config;
    config.reader = FarmingReader::cached;
    config.index_cache_bytes = 134217728;
    config.harvester_cores = 2; config.lookup_workers = 5;
    config.deadline_seconds = config.challenge_seconds = 1;
    config.chain_seconds_per_member = 0.008L;
    std::vector<FarmingDestination> drives(3);
    auto storage = [&](size_t lane, uint64_t bytes, long double latency, long double bandwidth)
    {
        auto& drive = drives[lane]; drive.capacity_bytes = bytes; drive.resource = lane; drive.known_device = true;
        drive.read_latency_seconds = latency; drive.read_bytes_per_second = bandwidth;
    };
    storage(0, 11000000000ull, 0.022L, 230000000);
    storage(1, 4000000000ull, 0.014L, 110000000);
    storage(2, 48000000000ull, 0.001L, 160000000);
    for (auto objective : {FarmingObjective::energy, FarmingObjective::latency})
    {
        config.objective = objective;
        auto selected = farming_plan(config, drives);
        check(selected.feasible && selected.destinations[0].files == 1 && selected.destinations[1].files == 1 &&
                  selected.destinations[2].files == 2,
              "cached search missed the feasible heterogeneous warm layout");
        auto live = farming_assess(config, drives, selected.destinations, extents(selected.destinations));
        check(live.feasible && live.response_bound_seconds == selected.response_bound_seconds,
              "cached extra descent differs from the exact final certificate");
        auto frozen = selected.destinations.front();
        auto with_empty = drives; with_empty.push_back(hdd(0));
        auto expanded = farming_plan(config, with_empty, 64ull * 1024 * 1024, std::span(&frozen, 1));
        check(expanded.feasible && expanded.destinations.front() == frozen && !expanded.destinations.back().files,
              "cached extra descent changed a frozen schedule or populated an empty destination");
        --config.index_cache_bytes;
        check(!farming_plan(config, drives).feasible, "partial cache was treated as a complete warm layout");
        ++config.index_cache_bytes;
        auto shared = drives; shared[1].resource = shared[0].resource;
        auto together = farming_plan(config, shared);
        auto checked = farming_assess(config, shared, together.destinations, extents(together.destinations));
        check(together.feasible == checked.feasible && together.response_bound_seconds == checked.response_bound_seconds &&
                  together.destinations[0].resource_lanes == 2 && together.destinations[1].resource_rank == 1,
              "cached descent changed shared-resource schedule or assessment semantics");
        config.index_cache_bytes = 0;
        auto cold = farming_plan(config, drives);
        config.reader = FarmingReader::stateless;
        auto stateless = farming_plan(config, drives);
        check(cold.destinations == stateless.destinations && cold.candidates_evaluated == stateless.candidates_evaluated,
              "a cache that cannot fit minimum counts added a redundant descent");
        config.reader = FarmingReader::cached; config.index_cache_bytes = 134217728;
    }
    config = {}; config.group_members_override = 2;
    auto bounded = hdd(10000000000ull); bounded.max_total_members = 2;
    auto single = farming_plan(config, {bounded});
    check(single.feasible && single.candidates_evaluated == 2,
          "unchanged complete singleton sweep repeated its coordinate pass");
}
void test_extent_member_churn()
{
    FarmingConfig config;
    config.meta_slots_override = 4; config.memory_bytes = 4000000000ull;
    config.max_eligible_families_per_resource = 1000000;
    config.deadline_seconds = config.challenge_seconds = 1000000;
    std::vector<FarmingDestination> drives(2, hdd());
    std::vector<FarmingDrivePlan> layouts(2);
    std::vector<std::vector<uint16_t>> counts(2);
    for (size_t lane = 0; lane < layouts.size(); ++lane)
    {
        drives[lane].resource = 7;
        layouts[lane].resource_rank = uint16_t(lane); layouts[lane].resource_lanes = 2; layouts[lane].meta_slots = 4;
        for (unsigned file = 0; file < 4096; ++file) counts[lane].push_back(uint16_t(file + 1));
    }
    auto original = farming_assess(config, drives, layouts, counts);
    // Reorder only mutually exclusive files within each family. A brute
    // family map remains unchanged while adjacent sweep boundaries repeatedly
    // expire one member size and introduce another.
    for (auto& lane : counts)
        for (size_t first = 0; first < lane.size(); first += 2) std::swap(lane[first], lane[first + 1]);
    auto reordered = farming_assess(config, drives, layouts, counts);
    check(original.feasible && reordered.feasible && original.response_bound_seconds == reordered.response_bound_seconds &&
              original.dynamic_work_seconds_per_sp == reordered.dynamic_work_seconds_per_sp &&
              original.reader_memory_bytes == reordered.reader_memory_bytes &&
              original.max_eligible_families_per_resource == reordered.max_eligible_families_per_resource,
          "member-size churn changed the family-union certificate");
    // The same member size can finish and restart at one family boundary.
    for (auto& lane : counts) std::fill(lane.begin(), lane.end(), 2);
    auto repeated = farming_assess(config, drives, layouts, counts);
    check(repeated.feasible && repeated.max_eligible_families_per_resource == original.max_eligible_families_per_resource,
          "same-family starts and ends lost a live member-size entry");
}
void test_cached_comparison_bound()
{
    FarmingConfig config; config.objective = FarmingObjective::latency;
    config.harvester_cores = config.lookup_workers = 1;
    config.max_eligible_families_per_resource = 1024;
    config.memory_bytes = 4000000000ull;
    config.deadline_seconds = config.challenge_seconds = 1000000;
    auto drive = hdd(1000000000000000ull);
    drive.max_groups = drive.max_total_members = 262144;
    auto small = hdd(10000000000ull); small.max_groups = small.max_total_members = 6;
    auto cold = farming_plan(config, {drive, small});
    config.reader = FarmingReader::cached; config.index_cache_bytes = 536870912;
    auto cached = farming_plan(config, {drive, small});
    check(cold.feasible && cached.feasible && cached.candidates_evaluated == farming_detail::maximum_candidates &&
              !farming_detail::better(config, cold, cached),
          "optional cache descent exceeded its comparison budget or discarded a certified baseline");
}
void test_fleet()
{
    FarmingConfig c;
    auto drive = hdd();
    std::vector<FarmingDestination> drives(60, drive);
    auto failed = farming_plan(c, drives);
    check(!failed.feasible && !failed.reason.empty() && failed.response_bound_seconds > c.deadline_seconds &&
              failed.max_eligible_families_per_resource == 1,
          "default infeasible fleet admitted");
    auto jobs = adversarial(drives, failed.destinations, extents(failed.destinations));
    check(simulate(c, drives, jobs, failed.destinations[0].files * 60).late,
          "default adversary produced no late work");
    c.harvester_cores = c.lookup_workers = 6;
    auto start = std::chrono::steady_clock::now();
    auto energy = farming_plan(c, drives);
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
            .count();
    c.objective = FarmingObjective::latency;
    auto latency = farming_plan(c, drives);
    check(energy.feasible && latency.feasible &&
              energy.dynamic_work_seconds_per_sp < latency.dynamic_work_seconds_per_sp &&
              energy.response_bound_seconds > latency.response_bound_seconds,
          "objectives lack admissible work/latency tradeoff");
    for (auto const& layout : energy.destinations)
    {
        check(layout.meta_slots == 256 && layout.families == 1, "default lost mutual exclusion");
    }
    std::ostream& record = std::cout;
    record << std::setprecision(18);
    for (auto objective : {FarmingObjective::energy, FarmingObjective::latency})
    {
        c.objective = objective;
        auto const& selected = objective == FarmingObjective::energy ? energy : latency;
        for (unsigned files = 1; files <= 256; ++files)
        {
            auto option = farming_detail::candidate(c, drive, files, 256, 0, 1);
            check(option.files == files, "balanced file-count neighbor omitted");
            std::vector<FarmingDrivePlan> options(drives.size(), option);
            for (size_t lane = 0; lane < options.size(); ++lane) options[lane].schedule = lane;
            check(!farming_detail::better(c, forecast(c, drives, options),
                                          selected),
                  "homogeneous exhaustive count dominates search");
        }
        auto counts = extents(selected.destinations);
        auto live = farming_assess(c, drives, selected.destinations, counts);
        check(live.feasible && live.response_bound_seconds <= selected.response_bound_seconds,
              "live extents differ from forecast bound");
        auto adversary = adversarial(drives, selected.destinations, counts);
        Observed actual;
        for (unsigned order = 0; order < 4; ++order)
        {
            std::reverse(adversary.begin(), adversary.end());
            actual = simulate(c, drives, adversary, selected.destinations[0].files * 60);
            check(!actual.late && actual.tail <= selected.response_bound_seconds,
                  "certified fleet adversary is late");
        }
        std::vector<Job> repeated;
        for (unsigned sp = 0; sp < 32; ++sp)
        {
            for (auto job : adversary)
            {
                job.arrival = sp * c.challenge_seconds;
                repeated.push_back(job);
            }
        }
        check(!simulate(c, drives, repeated, selected.destinations[0].files * 60).late,
              "certified batches carry late work");
        record << "objective=" << (objective == FarmingObjective::energy ? "energy" : "latency")
               << " cores=6 workers=6 files=" << selected.destinations[0].files
               << " group=" << selected.destinations[0].group_members
               << " metas=" << selected.destinations[0].meta_slots
               << " max_eligible=" << selected.max_eligible_families_per_resource
               << " bound_s=" << selected.response_bound_seconds
               << " work=" << selected.dynamic_work_seconds_per_sp
               << " cpu_work=" << selected.cpu_work_seconds_per_sp
               << " io_work=" << selected.io_work_seconds_per_sp << " independent_tail_s=" << actual.tail
               << " independent_late=" << actual.late << " candidates=" << selected.candidates_evaluated
               << " planning_bytes=" << selected.planning_memory_bytes << " elapsed_ms=" << elapsed << '\n';
    }
    c.objective = FarmingObjective::energy;
    auto almost = drives;
    almost[0].capacity_bytes -= 4096;
    auto mixed = farming_plan(c, almost);
    for (unsigned files = 1; files <= 256; ++files)
    {
        std::vector<FarmingDrivePlan> options;
        for (auto const& d : almost)
        {
            options.push_back(farming_detail::candidate(c, d, files, 256, 0, 1));
            options.back().schedule = options.size() - 1;
        }
        check(!farming_detail::better(c, forecast(c, almost, options), mixed),
              "manifest asymmetry lost a uniform better layout");
    }
    record << "default_cores=4 workers=4 feasible=" << failed.feasible
           << " bound_s=" << failed.response_bound_seconds << " reason=" << failed.reason << '\n';
    std::cout << "fleet exhaustive/certificate/objective checks passed; elapsed_ms=" << elapsed << '\n';
}
int main()
{
    try
    {
        test_layout_and_limits();
        test_adversarial_scheduler();
        test_filter_exclusion();
        test_storage_profiles();
        test_frozen_expansion();
        test_extent_admission();
        test_planning_memory_boundaries();
        test_cohort_costs();
        test_cohort_scheduler();
        test_cohort_replacement_scheduler();
        test_large_cohort_assessment();
        test_numerical_adjustments();
        test_large_frozen_expansion();
        test_small_exhaustive();
        test_cached_search_boundary();
        test_cached_comparison_bound();
        test_extent_member_churn();
        test_explicit_family_tradeoff();
        test_sparse_automatic_search();
        test_fleet();
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "farming policy tests passed\n";
    return 0;
}
