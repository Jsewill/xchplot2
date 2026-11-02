// Construction selection is a controller estimate. The engines still perform
// exact memory, codec, space and recovery admission before changing any bytes.
#pragma once
#include "host/GroupMerge.hpp"
#include <cmath>
#include <memory>
#include <mutex>
#include <tuple>

namespace group_grow
{
struct ConstructionCost
{
    long double read = 0, write = 0, codec = 0, syncs = 0;
    long double read_calls = 0, write_calls = 0, serial_codec = 0;
};
struct ConstructionForecast
{
    uint16_t members = 0;
    uint64_t batches = 0;
    long double minimum_merge_bytes = 0;
    ConstructionCost incremental, merged, finalization, packing, layout;
};
struct ConstructionRun { uint16_t members = 0; uint64_t bytes = 0; };
struct ExecutionIO
{
    uint64_t read_ns = 0, write_ns = 0, sync_ns = 0;
    uint64_t read_bytes = 0, written_bytes = 0, read_calls = 0, write_calls = 0, syncs = 0;
    ExecutionIO since(ExecutionIO const& before) const
    {
        return {read_ns - before.read_ns, write_ns - before.write_ns, sync_ns - before.sync_ns,
            read_bytes - before.read_bytes, written_bytes - before.written_bytes,
            read_calls - before.read_calls, write_calls - before.write_calls, syncs - before.syncs};
    }
};
struct ConstructionSample
{
    uint64_t elapsed_ns = 0, read_ns = 0, write_ns = 0, sync_ns = 0;
    uint64_t read_bytes = 0, written_bytes = 0, syncs = 0;
    long double codec_bytes = 0;
    unsigned workers = 0;
    uint64_t codec_ns = 0, decode_ns = 0, read_calls = 0, write_calls = 0;
    long double member_bytes = 0, chunk_bytes = 0;
    uint64_t cpu_ns = 0;
    bool cpu_measured = false;
    ExecutionIO scratch_io, destination_io;
    bool io_attributed = false;
    uint64_t serial_cpu_ns = 0;
    bool serial_cpu_measured = false;
};

// Keep recent completed observations. A single slow flush or a short append
// must not immediately replace all the evidence from earlier batches.
struct ConstructionSamples
{
    std::array<ConstructionSample, 8> samples{};
    size_t count = 0, next = 0;
    void add(ConstructionSample sample)
    {
        samples[next] = sample; next = (next + 1) % samples.size();
        count = std::min(count + 1, samples.size());
    }
};

struct GenerationHistory
{
    struct Sample { uint64_t members = 0, elapsed_ns = 0; };
    std::array<Sample, 8> samples{};
    size_t count = 0, next = 0;
    void add(uint64_t members, uint64_t elapsed)
    {
        if (!members || !elapsed) return;
        samples[next] = {members, elapsed}; next = (next + 1) % samples.size();
        count = std::min(count + 1, samples.size());
    }
    long double member_ns() const
    {
        if (!count) return 0;
        std::array<long double, 8> rates{};
        for (size_t i = 0; i < count; ++i)
            rates[i] = static_cast<long double>(samples[i].elapsed_ns) / samples[i].members;
        std::sort(rates.begin(), rates.begin() + count);
        return (rates[(count - 1) / 2] + rates[count / 2]) / 2;
    }
};

// Keep the two external batch candidates together, including their baseline
// observations when subsequent samples all use the smaller batch.
struct ExternalBatchHistory
{
    struct Samples
    {
        uint16_t members = 0;
        ConstructionSamples planning, encoding;
    };
    std::array<Samples, 2> sizes{};
    uint16_t ceiling = 0, selected = 0, source_members = 0;
    uint64_t source_bytes = 0;
    bool half_attempted = false;
    void prepare(uint16_t largest)
    {
        if (ceiling == largest) return;
        *this = {}; ceiling = selected = largest;
        sizes[0].members = largest; sizes[1].members = std::max(uint16_t(1), uint16_t(largest / 2));
    }
    void add(uint16_t members, ConstructionSample planning, ConstructionSample encoding)
    {
        if (!planning.elapsed_ns || !encoding.elapsed_ns) return;
        for (auto& size : sizes) if (size.members == members)
        { size.planning.add(planning); size.encoding.add(encoding); return; }
    }
};

struct ConstructionHistory : ConstructionSamples
{
    ConstructionSamples contributions, packing, layout, shapes;
    ConstructionSamples external_staging, external_packing;
    ConstructionSamples execution_contributions, execution_packing;
    unsigned overlap_workers = 0, overlap_staging_workers = 0;
};

// Histories belong to one run. Only identical, known physical-drive sets and
// compatible plot/producer settings can share them; group IDs and capacity
// are intentionally absent because rates are normalized and admission is exact.
struct ConstructionContext
{
    std::vector<std::string> drives;
    uint8_t k = 0, strength = 0, meta_group = 0;
    size_t memo_bytes = 0;
    uint64_t max_ram = 0, window = 0, checkpoint = 0, codec = 0;
    unsigned workers = 0;
    bool compact = false, io_overlap = false, release_cache = false;
    std::vector<std::string> scratch_drives;
    bool operator==(ConstructionContext const&) const = default;
};
struct SharedConstructionHistory
{
    std::mutex mutex;
    ConstructionHistory history;
    // Successor staging uses a different worker grant from foreground appends.
    // Its bounded trial survives a fresh group only in this physical context
    // and with the same actual grant; foreground timings stay local.
    ExternalBatchHistory successor_batches;
    unsigned successor_batch_workers = 0;
    ConstructionHistory snapshot()
    { std::lock_guard lock(mutex); return history; }
};
class ConstructionHistories
{
public:
    explicit ConstructionHistories(size_t limit) : limit_(limit) { entries_.reserve(limit); }
    std::shared_ptr<SharedConstructionHistory> find(ConstructionContext context, bool create = true)
    {
        if (context.drives.empty()) return {};
        for (auto const& entry : entries_) if (entry.context == context) return entry.source;
        if (!create) return {};
        require(entries_.size() < limit_, "construction history exceeds destination count");
        auto source = std::make_shared<SharedConstructionHistory>();
        entries_.push_back({std::move(context), source}); return source;
    }
    std::shared_ptr<SharedConstructionHistory> reuse_external(std::shared_ptr<SharedConstructionHistory> const& source,
        ConstructionContext const& context) const
    {
        if (!source || context.drives.empty() || context.scratch_drives.empty()) return {};
        for (auto const& entry : entries_) if (entry.source == source)
        {
            // External CPU occupancy records their actual phase grants. Keep
            // every physical/plot/producer boundary while normalizing workers.
            auto const& saved = entry.context;
            bool const same = context.drives == saved.drives && context.scratch_drives == saved.scratch_drives &&
                std::tie(context.k, context.strength, context.meta_group, context.memo_bytes, context.max_ram,
                    context.window, context.checkpoint, context.codec, context.compact, context.io_overlap, context.release_cache) ==
                std::tie(saved.k, saved.strength, saved.meta_group, saved.memo_bytes, saved.max_ram,
                    saved.window, saved.checkpoint, saved.codec, saved.compact, saved.io_overlap, saved.release_cache);
            return same ? source : nullptr;
        }
        return {};
    }
    std::shared_ptr<SharedConstructionHistory> execution(ConstructionContext const& context) const
    {
        if (context.drives.empty()) return {};
        // Service demands retain their observed worker grants. Only execution
        // forecasts normalize that CPU term; engine tuning still uses find().
        for (auto const& entry : entries_)
        {
            auto comparable = entry.context; comparable.workers = context.workers;
            if (comparable == context) return entry.source;
        }
        return {};
    }
    uint64_t memory_bytes() const
    {
        uint64_t bytes = sizeof(*this) + entries_.capacity() * sizeof(Entry);
        for (auto const& entry : entries_)
        {
            bytes += sizeof(SharedConstructionHistory) + 2 * sizeof(void*);
            bytes += entry.context.drives.capacity() * sizeof(std::string);
            for (auto const& drive : entry.context.drives) bytes += drive.capacity() + 1;
            bytes += entry.context.scratch_drives.capacity() * sizeof(std::string);
            for (auto const& drive : entry.context.scratch_drives) bytes += drive.capacity() + 1;
        }
        return bytes;
    }
private:
    struct Entry { ConstructionContext context; std::shared_ptr<SharedConstructionHistory> source; };
    size_t limit_;
    std::vector<Entry> entries_;
};
inline long double construction_shape(ConstructionSamples const& samples, bool chunk = false)
{
    if (!samples.count) return 0;
    std::array<long double, 8> values{};
    for (size_t i = 0; i < samples.count; ++i)
        values[i] = chunk ? samples.samples[i].chunk_bytes : samples.samples[i].member_bytes;
    std::sort(values.begin(), values.begin() + samples.count);
    return (values[(samples.count - 1) / 2] + values[samples.count / 2]) / 2;
}

struct ConstructionRates
{
    long double read = 1, write = 1, codec = 1, sync = 0, read_call = 0, write_call = 0;
};

// Service demands are nanoseconds inferred from completed operations in this
// run. They are not device bandwidths: cached syscalls remain charged, and a
// missing phase must leave the estimate unavailable rather than invent a rate.
struct ExecutionWork
{
    long double generation = 0, cpu = 0, serial = 0;
    long double scratch_read = 0, scratch_write = 0, destination = 0;
    uint64_t batches = 0;
    bool known = false, io_overlap = false;
};
struct ExecutionEstimate
{
    long double completion = 0, generation = 0, cpu = 0, scratch = 0, destination = 0;
    bool known = false;
};
struct ExecutionLane
{
    ExecutionWork work;
    size_t destination = 0, scratch = SIZE_MAX;
};
struct ExecutionReady
{
    size_t index = 0, deferrals = 0;
    long double remaining = 0;
    bool known = false;
};

// A cold compact-input trial compares whole durable cohorts in this run.
// Completed appends cannot establish this result: their final pack is still
// outstanding. Owners, physical-context tokens and group membership must match
// the ordinary cohort, and an incomplete comparison never becomes a win.
inline long double construction_margin(size_t observations);
class StreamingInputCohort
{
public:
    enum class Phase { cold, trial, baseline, accepted, rejected, uncalibrated };
    explicit StreamingInputCohort(size_t owners) : owners_(owners)
    { require(owners && owners <= 1024, "invalid compact-input cohort owner count"); }
    bool calibrating() const { return phase_ == Phase::trial || phase_ == Phase::baseline; }
    bool stream_allowed(size_t active) const
    { return phase_ == Phase::accepted || (phase_ == Phase::cold && !active) || (phase_ == Phase::trial && !draining_); }
    bool context_allowed(size_t owner, uint16_t members, uintptr_t context) const
    {
        require(owner < owners_.size(), "invalid compact-input cohort owner");
        auto const& record = owners_[owner];
        return phase_ != Phase::accepted || (record.target && members == record.members && context && context == record.context);
    }
    bool may_admit(size_t owner) const
    {
        require(owner < owners_.size(), "invalid compact-input cohort owner");
        if (phase_ == Phase::trial) return !draining_;
        if (phase_ == Phase::baseline) return owners_[owner].target && !owners_[owner].admitted;
        return true;
    }
    void admit(size_t owner, uint16_t members, uintptr_t context, bool streamed, uint64_t now, size_t required_owners = 1)
    {
        require(owner < owners_.size() && members, "invalid compact-input cohort admission");
        if (phase_ == Phase::cold && streamed)
        {
            require(required_owners && required_owners <= owners_.size(), "invalid compact-input cohort size");
            phase_ = Phase::trial; started_ = now; required_owners_ = required_owners;
        }
        auto& record = owners_[owner];
        if (phase_ == Phase::trial)
        {
            require(!draining_ && !record.admitted, "compact-input trial admitted a duplicate owner");
            record = {true, true, false, members, context, streamed};
        }
        else if (phase_ == Phase::baseline)
        {
            if (!record.target || record.admitted || members != record.members || !context || context != record.context)
            { cancel(); return; }
            if (!started_) started_ = now;
            record.admitted = true;
        }
    }
    void complete(size_t owner, uint16_t members, uint64_t bytes, uint64_t now, size_t active, bool streamed = true)
    {
        if (!calibrating()) return;
        require(owner < owners_.size(), "invalid compact-input cohort completion");
        auto& record = owners_[owner];
        if (!record.admitted || record.complete || members != record.members || !bytes || now <= started_ ||
            (phase_ == Phase::trial && record.streamed && !streamed))
        { cancel(); return; }
        record.complete = true;
        if (phase_ == Phase::trial)
        {
            trial_bytes_ = grow_add(trial_bytes_, bytes); draining_ = true;
            if (active) return;
            if (std::any_of(owners_.begin(), owners_.end(), [](auto const& item) { return item.target && !item.complete; }))
            { cancel(); return; }
            size_t observed = 0; for (auto const& item : owners_) if (item.target) ++observed;
            if (observed < required_owners_) { cancel(); return; }
            trial_ns_ = now - started_; phase_ = Phase::baseline; started_ = 0; draining_ = false;
            for (auto& item : owners_) { item.admitted = false; item.complete = false; }
        }
        else
        {
            baseline_bytes_ = grow_add(baseline_bytes_, bytes);
            if (active || std::any_of(owners_.begin(), owners_.end(), [](auto const& item) { return item.target && !item.complete; })) return;
            long double const trial_rate = static_cast<long double>(trial_ns_) / trial_bytes_;
            long double const baseline_rate = static_cast<long double>(now - started_) / baseline_bytes_;
            phase_ = trial_rate < baseline_rate * (1 - construction_margin(1)) ? Phase::accepted : Phase::rejected;
        }
    }
    void cancel() { if (calibrating()) phase_ = Phase::uncalibrated; }
    char const* status() const
    {
        switch (phase_)
        {
            case Phase::cold: return "not-tried";
            case Phase::accepted: return "accepted";
            case Phase::rejected: return "rejected";
            default: return "uncalibrated";
        }
    }
    uint64_t memory_bytes() const { return sizeof(*this) + owners_.capacity() * sizeof(Owner); }
private:
    struct Owner
    {
        bool target = false, admitted = false, complete = false;
        uint16_t members = 0;
        uintptr_t context = 0;
        bool streamed = false;
    };
    std::vector<Owner> owners_;
    Phase phase_ = Phase::cold;
    bool draining_ = false;
    size_t required_owners_ = 1;
    uint64_t started_ = 0, trial_ns_ = 0, trial_bytes_ = 0, baseline_bytes_ = 0;
};

inline bool streamed_input_candidate(size_t ordinary_active, size_t streamed_active,
    bool known_independent_ssd, bool scratch_admitted, bool singleton_fallback,
    bool recovery, bool prefetched)
{
    return streamed_active > ordinary_active && known_independent_ssd && scratch_admitted &&
        singleton_fallback && !recovery && !prefetched;
}
inline bool execution_singleton_fallback(uint64_t budget, uint64_t fixed, uint64_t workspace, uint64_t generation)
{
    return fixed <= budget && generation <= budget - fixed && workspace <= budget - fixed - generation;
}
inline size_t execution_generation_order(std::span<ExecutionReady const> ready)
{
    require(!ready.empty(), "generation order requires a ready destination");
    auto oldest = std::max_element(ready.begin(), ready.end(), [](auto const& a, auto const& b)
        { return a.deferrals < b.deferrals; });
    if (oldest->deferrals >= ready.size()) return oldest->index;
    // Preserve the rotating order until all competing consumers have usable
    // evidence. Once measured, start the longest remaining downstream tail.
    if (std::any_of(ready.begin(), ready.end(), [](auto const& item) { return !item.known; })) return ready.front().index;
    return std::max_element(ready.begin(), ready.end(), [](auto const& a, auto const& b)
        { return a.remaining < b.remaining; })->index;
}
inline ExecutionEstimate pipeline_execution_estimate(std::span<ExecutionLane const> lanes,
    size_t active, unsigned workers, unsigned granted, size_t resources, bool generation_overlap)
{
    ExecutionEstimate result;
    if (lanes.empty() || !active || active > lanes.size() || !workers || !granted || !resources) return result;
    std::vector<long double> io(resources, 0);
    long double cpu = 0, consumers = 0, longest = 0, drain = 0;
    for (auto const& lane : lanes)
    {
        auto const& work = lane.work;
        if (!work.known || !work.batches || lane.destination >= resources ||
            ((work.scratch_read || work.scratch_write) && lane.scratch >= resources)) return {};
        long double const scratch = work.scratch_read + work.scratch_write;
        io[lane.destination] += work.destination;
        if (scratch) io[lane.scratch] += scratch;
        result.generation += work.generation; cpu += work.cpu + work.serial;
        result.scratch += scratch; result.destination += work.destination;
        long double const compute = std::max(work.cpu / granted, work.serial);
        long double const lane_io = lane.scratch == lane.destination ? scratch + work.destination :
            std::max(scratch, work.destination);
        long double const consumer = work.io_overlap ? std::max(compute, lane_io) :
            work.cpu / granted + work.serial + scratch + work.destination;
        consumers += consumer; longest = std::max(longest, consumer);
        drain = std::max(drain, std::min(work.generation, consumer) / work.batches);
    }
    result.cpu = cpu / std::min<long double>(workers, active * granted);
    long double const downstream = std::max({result.cpu, *std::max_element(io.begin(), io.end()),
        consumers / active, longest});
    result.completion = generation_overlap ? std::max(result.generation, downstream) + drain :
        result.generation + downstream;
    result.known = std::isfinite(result.completion) && result.completion > 0;
    return result;
}

inline ConstructionRates construction_rates(ConstructionSample const& sample)
{
    ConstructionRates result;
    if (!sample.elapsed_ns || !sample.written_bytes) return result;
    result.write = static_cast<long double>(sample.write_ns) / sample.written_bytes;
    result.read = sample.read_bytes ? static_cast<long double>(sample.read_ns) / sample.read_bytes : result.write;
    // One observation cannot independently identify byte throughput and call
    // overhead. Share measured time between both terms, preserving its total.
    if (sample.write_calls)
    {
        result.write /= 2;
        result.write_call = static_cast<long double>(sample.write_ns) / (2 * sample.write_calls);
    }
    if (sample.read_calls)
    {
        result.read /= 2;
        result.read_call = static_cast<long double>(sample.read_ns) / (2 * sample.read_calls);
    }
    result.sync = sample.syncs ? static_cast<long double>(sample.sync_ns) / sample.syncs : 0;
    long double const io = static_cast<long double>(sample.read_ns) + sample.write_ns + sample.sync_ns;
    long double const codec = sample.codec_ns ? sample.codec_ns : std::max(0.0L, sample.elapsed_ns - io);
    result.codec = sample.codec_bytes ?
        (codec * std::max(1u, sample.workers) + sample.decode_ns) / sample.codec_bytes : 0;
    return result;
}
inline ConstructionRates construction_rates(ConstructionSamples const& history)
{
    if (!history.count) return {};
    std::array<ConstructionRates, 8> rates{};
    for (size_t i = 0; i < history.count; ++i) rates[i] = construction_rates(history.samples[i]);
    // A median filters isolated scheduler/cache/flush outliers while adapting
    // after a sustained change in storage or compression throughput.
    auto median = [&](long double ConstructionRates::* field)
    {
        std::array<long double, 8> values{};
        for (size_t i = 0; i < history.count; ++i) values[i] = rates[i].*field;
        std::sort(values.begin(), values.begin() + history.count);
        return (values[(history.count - 1) / 2] + values[history.count / 2]) / 2;
    };
    return {median(&ConstructionRates::read), median(&ConstructionRates::write),
        median(&ConstructionRates::codec), median(&ConstructionRates::sync),
        median(&ConstructionRates::read_call), median(&ConstructionRates::write_call)};
}
inline long double construction_cost(ConstructionCost const& work, Config const& config,
                                     ConstructionRates const& rates)
{
    long double const io = work.read * rates.read + work.write * rates.write +
        work.read_calls * rates.read_call + work.write_calls * rates.write_call;
    long double const parallel = work.codec * rates.codec / std::max(1u, config.workers);
    long double const serial = work.serial_codec * rates.codec;
    // Storage waits remain charged in full because cached I/O is not a
    // physical throughput sample. Codec work includes joining and encoding.
    return io + (config.io_overlap ? std::max(parallel, serial) : parallel + serial) + work.syncs * rates.sync;
}
inline long double construction_margin(size_t observations) { return observations >= 4 ? 0.10L : 0.15L; }

// Compare actual CPU occupancy only after sustained completed observations.
// Storage syscall wall time never substitutes for a missing thread CPU clock.
// A deadband retains the last bounded choice until the opposite demand wins.
inline unsigned external_staging_workers(unsigned workers, ConstructionHistory const& history)
{
    if (workers < 2) return 0;
    unsigned const neutral = workers / 2;
    constexpr size_t evidence = 4;
    auto range = [&](ConstructionSamples const& samples) -> std::optional<std::pair<long double, long double>>
    {
        if (samples.count < evidence) return {};
        long double low = 1, high = 0;
        for (size_t i = 0; i < evidence; ++i)
        {
            auto const& sample = samples.samples[(samples.next + samples.samples.size() - 1 - i) % samples.samples.size()];
            if (!sample.cpu_measured || !sample.elapsed_ns || !sample.workers) return {};
            long double const used = std::min(1.0L, static_cast<long double>(sample.cpu_ns) /
                (static_cast<long double>(sample.elapsed_ns) * sample.workers));
            low = std::min(low, used); high = std::max(high, used);
        }
        return std::pair{low, high};
    };
    auto const staging = range(history.external_staging), packing = range(history.external_packing);
    if (!staging || !packing) return neutral;
    unsigned const minimum = std::max(1u, neutral - 1), maximum = std::min(workers - 1, neutral + 1);
    unsigned selected = history.overlap_workers == workers ?
        std::clamp(history.overlap_staging_workers, minimum, maximum) : neutral;
    long double const margin = construction_margin(evidence);
    if (staging->first > packing->second + margin) selected = maximum;
    else if (packing->first > staging->second + margin) selected = minimum;
    return selected;
}

// A long external pack can observe its destination only while finalizing.
// Keep completed intervals local to that invocation; scratch contribution
// timings and a previous process cannot stand in for target checkpoints.
struct PackingCheckpointHistory
{
    static constexpr uint64_t forecast_bytes = 1024ull * 1024 * 1024;
    static constexpr uint64_t maximum_bytes = 4 * forecast_bytes;
    struct Sample { uint64_t bytes, elapsed_ns, checkpoint_ns, codec_ns = 0; };
    std::array<Sample, 8> samples{};
    size_t count = 0, next = 0;
    uint64_t trial_from = 0, trial_bytes = 0, learned_ceiling = maximum_bytes;
    long double trial_baseline = 0;
    std::array<long double, 3> trial_rates{};
    size_t trial_count = 0, trial_baseline_count = 0;
    static uint64_t memory_bytes()
    {
        return sizeof(PackingCheckpointHistory) + 2 * sizeof(std::array<long double, 8>) +
            sizeof(std::array<uint64_t, 8>);
    }
    void add(Sample sample)
    {
        if (!sample.bytes || !sample.elapsed_ns || sample.checkpoint_ns > sample.elapsed_ns) return;
        samples[next] = sample; next = (next + 1) % samples.size();
        count = std::min(count + 1, samples.size());
    }
    // Called once after each accepted add, so each completed trial interval
    // contributes exactly one observation.
    uint64_t select(uint64_t current, uint64_t ceiling, uint64_t remaining)
    {
        if (!count || !remaining) return current;
        auto active_rate = [](Sample const& sample)
        { return static_cast<long double>(std::max(sample.elapsed_ns - sample.checkpoint_ns, sample.codec_ns)) / sample.bytes; };
        auto completed_rate = [&](Sample const& sample)
        { return active_rate(sample) + static_cast<long double>(sample.checkpoint_ns) / sample.bytes; };
        if (trial_bytes)
        {
            // Larger dirty ranges need not have constant checkpoint latency.
            // Price completed end-to-end intervals, including reclamation,
            // before keeping a larger replay and retained-scratch frontier.
            auto const& sample = samples[(next + samples.size() - 1) % samples.size()];
            if (sample.bytes < trial_bytes) return current;
            trial_rates[trial_count++] = completed_rate(sample);
            if (trial_count < trial_rates.size()) return current;
            std::sort(trial_rates.begin(), trial_rates.end());
            bool const accepted = trial_rates[trial_rates.size() / 2] <
                trial_baseline * (1 - construction_margin(trial_baseline_count));
            trial_bytes = 0; trial_count = 0;
            if (!accepted) { learned_ceiling = trial_from; return trial_from; }
        }
        ceiling = std::min({ceiling, maximum_bytes, learned_ceiling});
        if (current >= ceiling) return current;
        bool const trial = current >= forecast_bytes;
        if (trial)
        {
            // Extrapolate only one geometric step, and keep enough work for
            // completed trial intervals. Short tails retain their frontier.
            ceiling = std::min(ceiling, current * 2);
            if (remaining / ceiling <= trial_rates.size()) return current;
        }
        else ceiling = std::min(ceiling, forecast_bytes);
        std::array<long double, 8> streaming{}, checkpoint{};
        size_t observations = 0;
        for (size_t i = 0; i < count; ++i)
        {
            auto const& sample = samples[i];
            if (trial && sample.bytes < current) continue;
            // A pending encoder can use checkpoint waits. Keep completed
            // codec work as a floor rather than treating all of it as free.
            // Codec counters arrive per joined window; short intervals have
            // coarse attribution, so this remains a forecast with a margin.
            streaming[observations] = active_rate(sample);
            checkpoint[observations++] = sample.checkpoint_ns;
        }
        // A median-sized baseline prevents one unusually slow checkpoint
        // from justifying a larger retained-scratch and replay frontier.
        if (!observations || (trial && observations < trial_rates.size())) return current;
        auto median = [&](auto& values)
        {
            std::sort(values.begin(), values.begin() + observations);
            return (values[(observations - 1) / 2] + values[observations / 2]) / 2;
        };
        long double const active = remaining * median(streaming), latency = median(checkpoint);
        auto cost = [&](uint64_t bytes)
        { return active + std::ceil(static_cast<long double>(remaining) / bytes) * latency; };
        long double const original = cost(current); auto best_cost = original; uint64_t best = current;
        std::array<uint64_t, 8> const candidates{4ull * 1024 * 1024, 16ull * 1024 * 1024,
            64ull * 1024 * 1024, 256ull * 1024 * 1024, 512ull * 1024 * 1024,
            forecast_bytes, current, ceiling};
        for (auto bytes : candidates) if (bytes > current && bytes <= ceiling && cost(bytes) < best_cost)
        { best_cost = cost(bytes); best = bytes; }
        if (best_cost >= original * (1 - construction_margin(observations))) return current;
        if (trial)
        {
            // Keep the baseline's actual total rather than a model that
            // treats every future checkpoint as the same fixed latency.
            observations = 0;
            for (size_t i = 0; i < count; ++i) if (samples[i].bytes >= current)
                streaming[observations++] = completed_rate(samples[i]);
            trial_baseline = median(streaming); trial_baseline_count = observations;
            trial_from = current; trial_bytes = best;
        }
        return best;
    }
};

inline bool four_input_candidate(Config const& config, bool nonrotating, bool known_device,
                                 uint64_t window_limit, size_t observations)
{
    if (config.window_bytes < 256 * 1024) return false;
    // Four-way packing shares the preread budget between more streams. On a
    // rotating drive use the widest admitted windows and completed timings;
    // unknown topology keeps the conservative two-input policy.
    return nonrotating || (known_device && observations && config.window_bytes == window_limit);
}
inline long double construction_merged_cost(ConstructionForecast const&, Config const&, ConstructionHistory const&);
inline bool prefer_four_inputs(ConstructionForecast const& four, ConstructionForecast const& two,
                               Config const& config, ConstructionHistory const& history)
{
    // The forecast prices each input's smaller preread refills separately,
    // alongside writes and syncs. Require a saving over two-way merging too,
    // rather than accepting a tiny fan-in gain merely because both beat an
    // incremental rewrite. Call timings are an estimate, not a measured seek.
    return construction_merged_cost(four, config, history) <
        construction_merged_cost(two, config, history) * (1 - construction_margin(history.count));
}

inline long double construction_codec_work(uint64_t before, uint64_t after, bool fresh)
{
    // Ordinary exact growth sizes and emits the new stream, decoding the old
    // stream in both passes. Fresh single-pass creation only emits the stream.
    return fresh ? static_cast<long double>(after) :
        2 * static_cast<long double>(before) + 2 * static_cast<long double>(after);
}

inline ConstructionForecast construction_forecast(uint8_t k, Config const& config,
    uint16_t current, uint64_t bytes, uint16_t batch, uint16_t maximum, long double estimated_member_bytes = 0)
{
    ConstructionForecast result;
    if ((!current && !estimated_member_bytes) || (current && !bytes) || !batch || current >= maximum) return result;
    long double const member_bytes = current ? static_cast<long double>(bytes) / current : estimated_member_bytes;
    result.members = uint16_t(std::min<long double>(maximum, std::floor(config.capacity_bytes / member_bytes)));
    if (result.members <= current) return result;
    uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
    long double const window = config.window_bytes, transaction = config.checkpoint_bytes;
    long double const block = std::ceil(static_cast<long double>(MergeEngine::block_bytes(config)) / 4096) * 4096;
    long double const slot = std::ceil((sizeof(MergeState::Record) +
        std::floor(config.capacity_bytes / block) * sizeof(uint32_t) + 32) / 4096) * 4096;
    long double const table = static_cast<long double>(chunks) * (sizeof(ChunkSummary) + sizeof(uint64_t));
    long double const framing = static_cast<long double>(chunks) * 8;
    auto transactions = [](long double n, long double unit) { return std::ceil(n / std::max(1.0L, unit)); };

    // Retain the same bounded reader set as the packer, including an ordinary
    // prefix. Normal tail merges keep the existing rank balance; only metadata,
    // space or final reader pressure forces the cheapest adjacent compaction.
    struct Run { uint32_t members; long double bytes; };
    std::array<Run, MergeState::max_runs> runs{};
    size_t count = current ? 1 : 0;
    if (current) runs[0] = {current, bytes + framing};
    auto& merged = result.merged;
    merged.read = bytes; // Summarize the existing ordinary prefix once.
    merged.read_calls = transactions(bytes, window * 8);
    // Fresh creation also syncs its bootstrap identity before the locators.
    // Parent-directory publication is outside the file-sync timing counters.
    merged.syncs = current ? 5 : 6;
    auto combine = [&](size_t inputs, bool balanced = false)
    {
        size_t first = count - inputs;
        if (!balanced)
        {
            long double least = std::numeric_limits<long double>::infinity();
            for (size_t begin = 0; begin + inputs <= count; ++begin)
            {
                long double total = 0;
                for (size_t i = begin; i < begin + inputs; ++i) total += runs[i].bytes;
                if (total <= least) { least = total; first = begin; }
            }
        }
        auto& left = runs[first];
        for (size_t i = first; i < first + inputs; ++i)
            merged.read_calls += transactions(runs[i].bytes, window * 8 / inputs);
        for (size_t i = first + 1; i < first + inputs; ++i)
        { left.members += runs[i].members; left.bytes += runs[i].bytes; }
        for (size_t i = first + 1; i + inputs - 1 < count; ++i) runs[i] = runs[i + inputs - 1];
        count -= inputs - 1;
        merged.read += left.bytes; merged.write += left.bytes;
        merged.syncs += 4 + 2 * transactions(left.bytes, transaction);
    };
    unsigned const fan = config.merge_inputs == 4 ? 4 : 2;
    size_t const retained = MergeEngine::packing_inputs(config);
    auto rank = [&](uint32_t members) { return (std::bit_width(members - 1) + (fan == 4)) / (fan == 4 ? 2 : 1); };
    long double const metadata = std::ceil((2 * slot + 8192) / block) * block;
    auto workspace = [&](long double live, size_t inputs)
    {
        return live + std::min(transaction, live) + 2 * live / chunks +
            (inputs + 9) * block + 2 * table + framing + metadata + 2 * slot + 8192;
    };
    long double old_bytes = bytes;
    for (uint32_t first = current; first < result.members;)
    {
        uint32_t const added = std::min<uint32_t>(batch, result.members - first);
        long double const growth = added * member_bytes, next_bytes = old_bytes + growth;
        if (count == runs.size()) combine(std::min<size_t>(fan, count));
        long double live = growth + framing;
        for (size_t i = 0; i < count; ++i) live += runs[i].bytes;
        while (count > 1 && workspace(live, count + 1) > config.capacity_bytes)
            combine(std::min<size_t>(fan, count));
        auto& ordinary = result.incremental;
        // A backward window whose displacement is smaller than its length
        // needs redo. Disjoint writes also checkpoint before crossing the
        // previous durable source frontier, limiting coalescing by growth.
        long double const redo = first ? next_bytes * std::min(1.0L, window / growth) : 0;
        // Exact ordinary admission reserves a whole redo window only when it
        // fits beside the final payload and checkpoints. Otherwise it keeps
        // the largest chunk plus alignment slack. Price that smaller reserve
        // so a large window does not look cheap near a destination's limit.
        long double const redo_unit = next_bytes + window + 8192 <= config.capacity_bytes ?
            window / 2 : std::min(window / 2, std::max(4096.0L, 2 * next_bytes / chunks));
        ordinary.read += 2 * old_bytes;
        ordinary.write += next_bytes + redo;
        ordinary.codec += first ? 2 * old_bytes + 2 * next_bytes : next_bytes;
        long double const interval = first ? std::min(transaction, growth / 2) : transaction;
        ordinary.syncs += 9 + 2 * transactions(next_bytes - redo, interval) +
            4 * transactions(redo, redo_unit);

        merged.read += table; merged.write += table + growth + framing;
        merged.read_calls += transactions(table, window * 8);
        merged.codec += 2 * growth + 2 * table;
        merged.syncs += 4;
        require(count < runs.size(), "construction forecast exceeded retained run limit");
        runs[count++] = {added, growth + framing};
        while (count > retained)
        {
            bool balanced = true;
            for (size_t i = count - fan; i + 1 < count; ++i)
                if (rank(runs[i].members) > rank(runs[count - 1].members)) balanced = false;
            if (!balanced) break;
            combine(fan, true);
        }
        ++result.batches; first += added; old_bytes = next_bytes;
    }
    auto collected = merged;
    while (count > retained) combine(std::min<size_t>(fan, count));
    auto const compacted = merged;
    for (size_t i = 0; i < count; ++i)
    {
        merged.read += runs[i].bytes;
        merged.read_calls += transactions(runs[i].bytes, window * 8 / count);
    }
    auto const packed = merged;
    merged.read += 2 * old_bytes;
    merged.read_calls += 2 * transactions(old_bytes, std::min(window, block));
    merged.write += 3 * old_bytes + framing;
    // Final contribution decoding and canonical encoding now share the codec
    // workers; ordered prereads no longer reconstruct the fragment stream.
    merged.codec += 2 * old_bytes;
    result.packing = {packed.read - compacted.read, old_bytes + framing, 2 * old_bytes,
        6 + 2 * transactions(old_bytes, transaction), packed.read_calls - compacted.read_calls,
        transactions(old_bytes + framing, window)};
    merged.syncs += 14 + 2 * transactions(old_bytes, transaction) +
        4 * transactions(old_bytes, block);
    // Each durable state writes one slot. Layout can move a block twice;
    // dependent moves each need a checkpoint, even with a larger transaction.
    merged.write += merged.syncs / 2 * slot;
    result.incremental.write += result.incremental.syncs / 2 * 1024;
    result.incremental.read_calls = transactions(result.incremental.read, window);
    result.incremental.write_calls = transactions(result.incremental.write, window) + result.incremental.syncs / 2;
    merged.write_calls = transactions(merged.write, window) + merged.syncs / 2;
    collected.write += collected.syncs / 2 * slot;
    auto& finish = result.finalization;
    finish.read = merged.read - collected.read; finish.write = merged.write - collected.write;
    finish.codec = merged.codec - collected.codec; finish.serial_codec = merged.serial_codec;
    finish.syncs = merged.syncs - collected.syncs;
    finish.read_calls = merged.read_calls - collected.read_calls;
    finish.write_calls = transactions(finish.write, window) + finish.syncs / 2;
    auto& packing = result.packing;
    packing.write += packing.syncs / 2 * slot;
    packing.write_calls += packing.syncs / 2;
    auto& layout = result.layout;
    layout.syncs = 8 + 4 * transactions(old_bytes, block);
    layout.read = 2 * old_bytes; layout.write = 2 * old_bytes + layout.syncs / 2 * slot;
    layout.read_calls = 2 * transactions(old_bytes, std::min(window, block));
    layout.write_calls = transactions(layout.write, window) + layout.syncs / 2;

    // Require space to start with one more batch. Future batches retain the
    // engine's exact admission and can finish into incremental growth near
    // capacity. Requiring an entire future group plus reserve here would
    // prevent auto mode from ever choosing merges for a fill-the-disk job.
    long double const incoming = std::min<uint32_t>(batch, result.members - current) * member_bytes + framing;
    result.minimum_merge_bytes = bytes + incoming + std::min(transaction, bytes + incoming) +
        11 * block + 2 * table + framing + metadata + 2 * slot + 8192;
    return result;
}

inline ConstructionForecast construction_external_forecast(uint8_t k, Config const& config,
    uint16_t current, uint16_t batch, uint16_t maximum, long double member_bytes,
    long double contribution_member_bytes, unsigned fan_in, unsigned packing_inputs,
    std::span<ConstructionRun const> existing = {})
{
    ConstructionForecast result;
    if (!batch || current >= maximum || member_bytes <= 0 || contribution_member_bytes <= 0) return result;
    require(fan_in >= 2 && fan_in <= 64 && packing_inputs >= fan_in && packing_inputs <= 64 && existing.size() <= 64,
        "invalid external execution forecast readers");
    result.members = uint16_t(std::min<long double>(maximum, std::floor(config.capacity_bytes / member_bytes)));
    if (result.members <= current) return result;
    struct Run { uint32_t members = 0; long double bytes = 0; };
    std::array<Run, 64> runs{}; size_t count = existing.size(); uint64_t members = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (!existing[i].members || !existing[i].bytes) return {};
        runs[i] = {existing[i].members, static_cast<long double>(existing[i].bytes)}; members += existing[i].members;
    }
    if (members != current) return {};
    auto& stage = result.merged;
    long double const summary = static_cast<long double>(PlotGroupFile::getChunkCountForK(k)) * (sizeof(ChunkSummary) + sizeof(uint64_t));
    auto combine = [&](size_t first)
    {
        auto& left = runs[first];
        for (size_t i = first + 1; i < first + fan_in; ++i)
        { left.members += runs[i].members; left.bytes += runs[i].bytes; }
        stage.read += left.bytes; stage.write += left.bytes;
        for (size_t i = first + 1; i + fan_in - 1 < count; ++i) runs[i] = runs[i + fan_in - 1];
        count -= fan_in - 1;
    };
    for (uint32_t first = current; first < result.members;)
    {
        // ExternalEngine forces the cheapest adjacent compaction only at its
        // hard run limit. Its normal tail rule compares equal member ranges;
        // it does not use MergeEngine's logarithmic rank or final balancing.
        if (count == runs.size())
        {
            size_t cheapest = 0; long double least = std::numeric_limits<long double>::infinity();
            for (size_t begin = 0; begin + fan_in <= count; ++begin)
            {
                long double bytes = 0;
                for (size_t i = begin; i < begin + fan_in; ++i) bytes += runs[i].bytes;
                if (bytes <= least) { cheapest = begin; least = bytes; }
            }
            combine(cheapest);
        }
        uint32_t const added = std::min<uint32_t>(batch, result.members - first);
        long double const growth = added * contribution_member_bytes;
        stage.read += first ? summary : 0; stage.write += summary + growth; stage.codec += 2 * growth;
        runs[count++] = {added, growth}; ++result.batches; first += added;
        while (count > packing_inputs)
        {
            size_t const begin = count - fan_in; bool equal = true;
            for (size_t i = begin + 1; i < count; ++i) equal = equal && runs[i].members == runs[begin].members;
            if (!equal) break;
            combine(begin);
        }
    }
    auto& pack = result.packing;
    for (size_t i = 0; i < count; ++i) pack.read += runs[i].bytes;
    pack.write = result.members * member_bytes; pack.codec = 2 * pack.write;
    result.finalization = pack;
    stage.read += pack.read; stage.write += pack.write; stage.codec += pack.codec;
    return result;
}

inline long double construction_finalization_cost(ConstructionForecast const& forecast,
    Config const& config, ConstructionHistory const& history)
{
    auto const fallback = construction_rates(history);
    if (!history.packing.count && !history.layout.count)
        return construction_cost(forecast.finalization, config, fallback);
    auto const packing = history.packing.count ? construction_rates(history.packing) : fallback;
    auto const layout = history.layout.count ? construction_rates(history.layout) : fallback;
    auto const copying = history.contributions.count ? construction_rates(history.contributions) : fallback;
    auto const& all = forecast.finalization; auto const& pack = forecast.packing; auto const& place = forecast.layout;
    // An unbalanced final stack can need packet copies before packing. Keep
    // that I/O separate too; phase-local call rounding may consume the entire
    // aggregate call estimate even when there is no final merge.
    ConstructionCost merge{std::max(0.0L, all.read - pack.read - place.read),
        std::max(0.0L, all.write - pack.write - place.write), 0,
        std::max(0.0L, all.syncs - pack.syncs - place.syncs),
        std::max(0.0L, all.read_calls - pack.read_calls - place.read_calls),
        std::max(0.0L, all.write_calls - pack.write_calls - place.write_calls)};
    return construction_cost(merge, config, copying) + construction_cost(pack, config, packing) +
        construction_cost(place, config, layout);
}
inline long double construction_merged_cost(ConstructionForecast const& forecast,
    Config const& config, ConstructionHistory const& history)
{
    auto const rates = history.contributions.count ? construction_rates(history.contributions) : construction_rates(history);
    auto const& all = forecast.merged; auto const& finish = forecast.finalization;
    ConstructionCost streaming{all.read - finish.read, all.write - finish.write, all.codec - finish.codec,
        all.syncs - finish.syncs, all.read_calls - finish.read_calls, all.write_calls - finish.write_calls,
        all.serial_codec - finish.serial_codec};
    return construction_cost(streaming, config, rates) + construction_finalization_cost(forecast, config, history);
}

inline std::optional<std::pair<long double, long double>> execution_cpu_rates(ConstructionSamples const& samples)
{
    if (!samples.count) return {};
    std::array<long double, 8> parallel{}, serial{};
    for (size_t i = 0; i < samples.count; ++i)
    {
        auto const& sample = samples.samples[i];
        if (!sample.cpu_measured || !sample.serial_cpu_measured || sample.cpu_ns < sample.serial_cpu_ns ||
            sample.codec_bytes <= 0) return {};
        parallel[i] = (sample.cpu_ns - sample.serial_cpu_ns) / sample.codec_bytes;
        serial[i] = sample.serial_cpu_ns / sample.codec_bytes;
    }
    auto median = [&](auto& values)
    {
        std::sort(values.begin(), values.begin() + samples.count);
        return (values[(samples.count - 1) / 2] + values[samples.count / 2]) / 2;
    };
    return std::pair{median(parallel), median(serial)};
}
inline std::optional<long double> execution_io_rate(ConstructionSamples const& samples, bool scratch, bool read)
{
    std::array<long double, 8> values{}; size_t count = 0;
    for (size_t i = 0; i < samples.count; ++i)
    {
        auto const& sample = samples.samples[i]; if (!sample.io_attributed) return {};
        auto const& io = scratch ? sample.scratch_io : sample.destination_io;
        uint64_t const bytes = read ? io.read_bytes : io.written_bytes;
        if (bytes) values[count++] = (static_cast<long double>(read ? io.read_ns : io.write_ns) +
            (read ? 0 : io.sync_ns)) / bytes;
    }
    if (!count) return {};
    std::sort(values.begin(), values.begin() + count);
    return (values[(count - 1) / 2] + values[count / 2]) / 2;
}
inline ExecutionWork construction_execution_work(ConstructionForecast const& forecast,
    ConstructionHistory const& history, long double generation, bool merging, bool external = false,
    bool io_overlap = false)
{
    ExecutionWork result; result.batches = forecast.batches; result.generation = generation; result.io_overlap = io_overlap;
    if (!forecast.batches || generation <= 0) return result;
    auto io = [](ConstructionCost const& work, ConstructionRates const& rates)
    { return work.read * rates.read + work.write * rates.write + work.read_calls * rates.read_call +
        work.write_calls * rates.write_call + work.syncs * rates.sync; };
    auto cpu = [&](ConstructionCost const& work, std::pair<long double, long double> const& rates)
    {
        result.cpu += work.codec * rates.first;
        result.serial += work.codec * rates.second + work.serial_codec * (rates.first + rates.second);
    };
    if (!merging)
    {
        auto const measured = execution_cpu_rates(history); if (!measured) return result;
        result.destination = io(forecast.incremental, construction_rates(history)); cpu(forecast.incremental, *measured);
    }
    else
    {
        auto const& contributions = external ? history.execution_contributions : history.contributions;
        auto const& packing = external ? history.execution_packing : history.packing;
        auto const contribution_cpu = execution_cpu_rates(contributions), packing_cpu = execution_cpu_rates(packing);
        if (!contribution_cpu || !packing_cpu) return result;
        auto const& all = forecast.merged; auto const& pack = forecast.packing; auto const& place = forecast.layout;
        ConstructionCost runs{std::max(0.0L, all.read - pack.read - place.read),
            std::max(0.0L, all.write - pack.write - place.write), std::max(0.0L, all.codec - pack.codec),
            std::max(0.0L, all.syncs - pack.syncs - place.syncs),
            std::max(0.0L, all.read_calls - pack.read_calls - place.read_calls),
            std::max(0.0L, all.write_calls - pack.write_calls - place.write_calls), all.serial_codec};
        cpu(runs, *contribution_cpu); cpu(pack, *packing_cpu);
        if (external)
        {
            auto const packed_read = execution_io_rate(packing, true, true);
            auto const written = execution_io_rate(contributions, true, false);
            auto read = execution_io_rate(contributions, true, true);
            if (!read) read = packed_read;
            if (!read || !packed_read || !written) return {};
            result.scratch_read = runs.read * *read + pack.read * *packed_read;
            result.scratch_write = runs.write * *written;
            auto median = [](ConstructionSamples const& samples, auto field) -> std::optional<long double>
            {
                if (!samples.count) return {};
                std::array<long double, 8> values{};
                for (size_t i = 0; i < samples.count; ++i)
                {
                    auto const& sample = samples.samples[i]; if (!sample.io_attributed) return {};
                    values[i] = field(sample);
                }
                std::sort(values.begin(), values.begin() + samples.count);
                return (values[(samples.count - 1) / 2] + values[samples.count / 2]) / 2;
            };
            auto const metadata = median(contributions, [](auto const& sample)
            { auto const& io = sample.destination_io; return static_cast<long double>(io.read_ns) + io.write_ns + io.sync_ns; });
            auto const target = median(packing, [](auto const& sample)
            {
                auto const& io = sample.destination_io;
                return sample.written_bytes ? (static_cast<long double>(io.read_ns) + io.write_ns + io.sync_ns) /
                    sample.written_bytes : std::numeric_limits<long double>::infinity();
            });
            if (!metadata || !target || !std::isfinite(*target)) return {};
            result.destination = forecast.batches * *metadata + pack.write * *target;
        }
        else
        {
            if (!history.layout.count) return {};
            result.destination = io(runs, construction_rates(history.contributions)) + io(pack, construction_rates(history.packing)) +
                io(place, construction_rates(history.layout));
        }
    }
    result.known = true; return result;
}

inline bool prefer_merged(ConstructionForecast const& forecast, Config const& config,
                          ConstructionSample const& sample = {})
{
    if (forecast.batches <= 1 || forecast.minimum_merge_bytes > config.capacity_bytes) return false;
    // A resumed ordinary file has no timed append yet. Use relative work
    // until a completed append supplies syscall latency and codec throughput.
    auto const rates = construction_rates(sample);
    return construction_cost(forecast.merged, config, rates) <
        construction_cost(forecast.incremental, config, rates) * (1 - construction_margin(sample.elapsed_ns ? 1 : 0));
}
inline bool prefer_merged(ConstructionForecast const& forecast, Config const& config,
                          ConstructionHistory const& history)
{
    if (forecast.batches <= 1 || forecast.minimum_merge_bytes > config.capacity_bytes) return false;
    auto const rates = construction_rates(history);
    return construction_merged_cost(forecast, config, history) <
        construction_cost(forecast.incremental, config, rates) * (1 - construction_margin(history.count));
}
} // namespace group_grow
