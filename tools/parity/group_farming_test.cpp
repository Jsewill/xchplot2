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
                  std::vector<Job> const& jobs, uint64_t files)
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
    long double const entries = std::ldexp(1.0L, config.k - 6);
    bool const warm =
        config.reader == FarmingReader::cached && files * entries * 8 <= config.index_cache_bytes;
    std::vector<long double> cores(std::min(config.harvester_cores, config.lookup_workers));
    std::vector<Running> running;
    uint64_t sequence = 0;
    size_t next = 0;
    long double now = 0;
    auto schedule = [&](Running& active, unsigned stage)
    {
        auto const& job = jobs[active.job];
        long double end;
        if (stage & 1)
        {
            long double cost;
            if (stage < 4)
            {
                cost = entries * config.index_seconds_per_entry * (stage == 1 ? 0.1L : 0.9L);
            }
            else
            {
                unsigned const chunk = (stage - 5) / 2;
                long double const share = job.cpu_pattern == 4       ? 0.25L
                                          : job.cpu_pattern == chunk ? 0.85L
                                                                     : 0.05L;
                cost = job.members * config.chain_seconds_per_member * share;
            }
            auto core = std::min_element(cores.begin(), cores.end());
            end = *core = std::max(now, *core) + cost;
        }
        else
        {
            long double const bytes = stage == 0   ? 306
                                      : stage == 2 ? entries * config.compressed_index_bytes_per_entry
                                                   : 64 * (config.k + 1.45L) / 8 * job.members;
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
    std::map<std::pair<size_t, uint64_t>, Job> families;
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
            auto key = std::make_pair(resource, layouts[lane].family(file));
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
    for (auto& layout : simple)
    {
        layout.meta_slots = 256;
        layout.meta_bits = 8;
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
                check(!farming_detail::better(c, forecast(c, drives, {a, b}), selected),
                      "mixed exhaustive assignment dominates search");
            }
        }
        drives[0] = drive;
        drives[1] = drive;
    }
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
            check(!farming_detail::better(c, forecast(c, drives, std::vector<FarmingDrivePlan>(60, option)),
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
        test_extent_admission();
        test_small_exhaustive();
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
