// Construction selection is a controller estimate. The engines still perform
// exact memory, codec, space and recovery admission before changing any bytes.
#pragma once
#include "host/GroupMerge.hpp"
#include <bit>
#include <cmath>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
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
    uint64_t batches = 0, incremental_batches = 0;
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

// Equality includes the authenticated instances behind an unchanged slot count.
// A weak reference keeps expired observations outside live worker reservations.
struct GenerationSupply
{
    uint64_t slots = 0;
    std::weak_ptr<void const> roster;
    bool operator==(GenerationSupply const& other) const
    { return slots == other.slots && !roster.owner_before(other.roster) && !other.roster.owner_before(roster); }
};

// Current work and future files on one writer, bounded by both hard quotas.
inline uint64_t remaining_publication_members(uint64_t member_limit, uint64_t completed, uint64_t file_limit,
                                                 uint64_t ordinal, uint64_t file_members)
{
    uint64_t const members = member_limit - std::min(member_limit, completed), files = file_limit - std::min(file_limit, ordinal);
    return file_members && files > members / file_members ? members : files * file_members;
}

// Compare complete waves of like-sized groups on rotating destinations. A wave
// starts before the first generation request and ends after its last publication.
// This charges transfer, temporary I/O and contention across overlapping writers.
class PoolPublicationHistory
{
public:
    struct Shape { unsigned k, strength, members; GenerationSupply supply; bool operator==(Shape const&) const = default; };
    struct Choice { size_t width; bool encoded, trial; };
    bool observed(Shape shape) const { return shape == shape_; }
    Choice select(Shape shape, size_t available, size_t initial, unsigned fixed_width, std::string_view output,
                  bool initial_encoded = true, bool allow_trial = true) const
    {
        require(available && initial, "empty pooled publication candidate");
        auto ordinary = std::min(available, fixed_width ? size_t(fixed_width) : initial);
        if (!fixed_width && ordinary > 3) ordinary = std::bit_floor(ordinary);
        auto allowed = [&](bool encoded) { return output == "auto" || encoded == (output == "encoded"); };
        if (live_ && shape == shape_ && current_.width <= available && (!fixed_width || current_.width == ordinary) &&
            allowed(current_.encoded)) return current_;
        Choice best{ordinary, output == "auto" ? initial_encoded : output == "encoded", false};
        long double fastest = std::numeric_limits<long double>::infinity();
        // A shrinking or expanding set of HDDs can make a previously measured
        // width feasible again, even if a new trial would skip that width.
        if (shape == shape_) for (auto const& result : results_)
        {
            if (!result.width || result.width > available || (fixed_width && result.width != ordinary)) continue;
            for (bool encoded : {initial_encoded, !initial_encoded}) if (allowed(encoded))
            {
                auto const time = result.times[encoded].member_ns();
                if (time && time < fastest) { fastest = time; best = {result.width, encoded, false}; }
            }
        }
        if (!allow_trial || (fixed_width && output != "auto")) return best;
        auto measured = [&](size_t width) -> Result const*
        {
            if (shape != shape_) return nullptr;
            auto it = std::find_if(results_.begin(), results_.end(), [&](auto const& r) { return r.width == width; });
            return it == results_.end() ? nullptr : &*it;
        };
        auto candidate = [&](size_t width) -> std::optional<Choice>
        {
            auto const* result = measured(width);
            for (bool encoded : {initial_encoded, !initial_encoded})
            {
                if (allowed(encoded) && (!result || !result->tried[encoded])) return Choice{width, encoded, true};
            }
            return {};
        };
        if (auto choice = candidate(ordinary)) return *choice;
        if (fixed_width) return best;
        auto next_width = [](size_t n) { return n < 4 ? n + 1 : n * 2; };
        auto rate = [&](size_t width, bool encoded)
        {
            auto const* result = measured(width);
            return result ? result->times[encoded].member_ns() : 0.L;
        };
        // Try another independent writer while either output form still gains
        // throughput. A source-limited encoded stream need not stop the owner
        // from testing a wider packed pipeline.
        size_t previous = 0;
        for (size_t width = ordinary; width < available;)
        {
            if (previous)
            {
                bool improving = false;
                for (bool encoded : {true, false})
                {
                    auto const before = rate(previous, encoded), after = rate(width, encoded);
                    if (before && after && after < before) improving = true;
                }
                if (!improving) break;
            }
            auto const next = std::min(next_width(width), available);
            if (auto choice = candidate(next)) return *choice;
            previous = width; width = next;
        }
        for (size_t width = 1; width < ordinary; width = next_width(width))
            if (auto choice = candidate(width)) return *choice;
        // Once the coarse trials finish, narrow the interval around the
        // fastest measured width. A shared controller can saturate between
        // powers of two. Keep the completed measurements when storage is full.
        size_t below = 0, above = 0;
        bool room = false;
        for (auto const& result : results_)
        {
            if (!result.width) { room = true; continue; }
            if (result.width > available) continue;
            if (auto choice = candidate(result.width)) return *choice;
            if (result.width < best.width) below = std::max(below, result.width);
            if (result.width > best.width && (!above || result.width < above)) above = result.width;
        }
        if (room && std::isfinite(fastest))
        {
            auto const left = below ? best.width - below : 0, right = above ? above - best.width : 0;
            if (std::max(left, right) > 1)
                return Choice{right >= left ? best.width + right / 2 : best.width - left / 2, best.encoded, true};
        }
        return best;
    }
    template<class TrialCost>
    std::optional<bool> finite_output(Shape shape, uint64_t remaining_members, uint64_t deadline_ns,
                                      TrialCost const& trial_cost) const
    {
        if (shape != shape_) return {};
        for (auto const& result : results_) if (result.width == 1)
        {
            auto const packed = result.times[false].member_ns(), fresh = result.times[true].member_ns();
            if (bool(packed) == bool(fresh)) return {};
            // Even instantaneous future publications cannot save more than
            // this sole writer's measured time for the remaining hard quota.
            auto const remaining_ns = std::max(packed, fresh) * std::max<uint64_t>(shape.members, remaining_members);
            if (remaining_ns > deadline_ns) return {};
            // No current incumbent, unknown trial, or a measured first wave.
            auto const cost = trial_cost(!fresh);
            if (cost && (*cost ? *cost : deadline_ns) >= remaining_ns) return fresh != 0;
        }
        return {};
    }
    bool final_group_encoded(Shape shape, bool encoded, bool fixed_width,
                             long double packed_member_ns, long double fresh_member_ns) const
    {
        if (shape == shape_)
        {
            auto const measured = std::find_if(results_.begin(), results_.end(), [](auto const& result) { return result.width == 1; });
            if (measured != results_.end())
            {
                auto const packed = measured->times[false].member_ns(), fresh = measured->times[true].member_ns();
                // Durable publication includes the owner's packing and syncs.
                if (packed && fresh) return fresh <= packed;
                if (packed && !fresh_member_ns) return false;
            }
        }
        // Spare workers cannot hide an output trial when generation is draining.
        if (packed_member_ns > 0) return fresh_member_ns > 0 && fresh_member_ns < packed_member_ns;
        return encoded || (!fixed_width && select(shape, 1, 1, 1, "auto", true, false).encoded);
    }
    bool begin(Shape shape, Choice choice, uint64_t now)
    {
        // Enroll one wave. Other groups can start on idle writers while its
        // last member is being published, including during calibration.
        if (live_ && (shape != shape_ || choice.width != current_.width || choice.encoded != current_.encoded ||
            finished_ || live_ >= current_.width)) return false;
        if (!choice.trial && (shape != shape_ || std::none_of(results_.begin(), results_.end(),
            [&](auto const& r) { return r.width == choice.width && r.times[choice.encoded].count; }))) return false;
        if (!live_)
        {
            if (shape != shape_) { results_ = {}; shape_ = shape; }
            current_ = choice; started_ = now; members_ = 0; finished_ = false; valid_ = true;
        }
        ++live_; return true;
    }
    void publish(uint64_t members, uint64_t now, bool comparable)
    {
        require(live_, "unmatched pooled publication");
        valid_ &= comparable && members == shape_.members;
        members_ = grow_add(members_, members); finished_ = true;
        if (--live_) return;
        auto it = std::find_if(results_.begin(), results_.end(),
            [&](auto const& r) { return !r.width || r.width == current_.width; });
        if (it == results_.end())
        {
            // Preserve the normal search widths when a shrinking destination
            // set introduces a succession of distinct tail widths.
            it = std::find_if(results_.begin(), results_.end(), [](auto const& r)
                { return r.width != 3 && !std::has_single_bit(r.width); });
            if (it == results_.end()) return;
            *it = {};
        }
        it->width = current_.width; it->tried[current_.encoded] = true;
        // Admission can run out of RAM or independent spindles before a
        // cohort fills. Its time does not describe the requested width.
        if (valid_ && members_ == current_.width * shape_.members && now > started_)
            it->times[current_.encoded].add(members_, now - started_);
    }
private:
    struct Result { size_t width = 0; bool tried[2]{}; GenerationHistory times[2]; };
    // 1, 2, 3 and powers of two through the 1024-destination limit, or an
    // explicit writer count trimmed to the remaining destinations.
    std::array<Result, 16> results_{};
    Shape shape_{}; Choice current_{};
    size_t live_ = 0;
    uint64_t started_ = 0, members_ = 0;
    bool finished_ = false, valid_ = false;
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
    std::array<uint64_t, 2> fresh_checkpoint_bytes{};
    unsigned successor_batch_workers = 0;
    // This owner-local observation includes generation, receiver stalls,
    // engine work, final sync and publication. A changed workload or pool
    // supply starts a new comparison even on the same physical storage.
    struct Publication
    {
        struct Shape
        {
            uint16_t members = 0;
            size_t width = 0;
            GenerationSupply supply;
            bool operator==(Shape const&) const = default;
        };
        Shape shape;
        GenerationHistory times;
        bool attempted = false;
        void begin(Shape next)
        {
            if (shape != next) { times = {}; shape = next; }
            attempted = true;
        }
        long double rate(Shape next) const { return shape == next ? times.member_ns() : 0; }
    } publication;
    ConstructionHistory snapshot()
    { std::lock_guard lock(mutex); return history; }
};
class ConstructionHistories
{
public:
    explicit ConstructionHistories(size_t limit) : limit_(limit) { entries_.reserve(limit); }
    bool empty() const { return entries_.empty(); }
    std::shared_ptr<SharedConstructionHistory> find(ConstructionContext context, bool create = true)
    {
        if (context.drives.empty()) return {};
        for (auto const& entry : entries_) if (entry.context == context) return entry.source;
        if (!create) return {};
        if (entries_.size() == limit_) return {};
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
    std::shared_ptr<SharedConstructionHistory> execution(ConstructionContext context) const
    {
        if (context.drives.empty()) return {};
        // Service demands retain their observed worker grants. Only execution
        // forecasts normalize that CPU term; engine tuning still uses find().
        for (auto const& entry : entries_)
        {
            context.workers = entry.context.workers;
            if (entry.context == context) return entry.source;
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

struct ConstructionPrior
{
    uint64_t low_ns = 0, high_ns = 0, oldest_utc = 0;
    unsigned samples = 0;
};
enum class ConstructionModeReason { ordinary, partial_batch, forecast, measured, prior, trial, required, recovery, fallback };
inline char const* construction_reason(ConstructionModeReason reason)
{
    switch (reason)
    {
        case ConstructionModeReason::ordinary: return "whole group fits in RAM";
        case ConstructionModeReason::partial_batch: return "partial RAM batch";
        case ConstructionModeReason::forecast: return "predicted last publication";
        case ConstructionModeReason::measured: return "time through publication at this load";
        case ConstructionModeReason::prior: return "prior durable publications at the same host, load and RAM cap";
        case ConstructionModeReason::trial: return "bounded trial after a comparable RAM publication";
        case ConstructionModeReason::required: return "requested construction or required streaming input";
        case ConstructionModeReason::recovery: return "resuming the recorded construction";
        case ConstructionModeReason::fallback: return "scratch capacity or workspace unavailable";
    }
    return "";
}
struct ConstructionModeChoice
{
    bool external = false;
    ConstructionModeReason reason = ConstructionModeReason::ordinary;
    uint64_t prior_oldest_utc = 0;
    unsigned prior_samples = 0;
};
inline ConstructionModeChoice choose_construction_mode(bool partial,
    SharedConstructionHistory::Publication::Shape ram_shape, SharedConstructionHistory::Publication::Shape scratch_shape,
    SharedConstructionHistory const* ram, SharedConstructionHistory const* scratch, bool allow_trial,
    long double ram_finish = 0, long double scratch_finish = 0,
    ConstructionPrior ram_prior = {}, ConstructionPrior scratch_prior = {})
{
    // Both candidates have already passed admission. Only compare complete
    // work at the same target. Keep the existing choice inside timing error.
    auto const margin = construction_margin(1);
    auto choose = [&](long double ordinary, long double external, ConstructionModeReason reason)
    {
        return ConstructionModeChoice{partial ? !(ordinary < external * (1 - margin)) :
            external < ordinary * (1 - margin), reason};
    };
    if (ram_finish > 0 && scratch_finish > 0) return choose(ram_finish, scratch_finish, ConstructionModeReason::forecast);
    auto const ordinary = ram ? ram->publication.rate(ram_shape) : 0.L;
    auto const external = scratch ? scratch->publication.rate(scratch_shape) : 0.L;
    if (ordinary && external && ram_shape.width == scratch_shape.width)
        return choose(ordinary, external, ConstructionModeReason::measured);
    if (ram_shape.width == 1 && scratch_shape.width == 1 && ram_shape.members == scratch_shape.members)
    {
        auto const ram_low = ordinary ? ordinary * ram_shape.members : ram_prior.low_ns;
        auto const ram_high = ordinary ? ordinary * ram_shape.members : ram_prior.high_ns;
        auto const scratch_low = external ? external * scratch_shape.members : scratch_prior.low_ns;
        auto const scratch_high = external ? external * scratch_shape.members : scratch_prior.high_ns;
        if (ram_low && scratch_low && (scratch_high < ram_low * (1 - margin) || ram_high < scratch_low * (1 - margin)))
        {
            uint64_t const stamp = ordinary ? scratch_prior.oldest_utc : external ? ram_prior.oldest_utc :
                std::min(ram_prior.oldest_utc, scratch_prior.oldest_utc);
            return {scratch_high < ram_low, ConstructionModeReason::prior, stamp,
                (ordinary ? 0u : ram_prior.samples) + (external ? 0u : scratch_prior.samples)};
        }
    }
    bool const earlier_ram = !ordinary && ram_shape.width == 1 && scratch_shape.width == 1 &&
        ram_shape.members == scratch_shape.members && ram_prior.samples && !scratch_prior.samples;
    if (allow_trial && !partial && (ordinary || earlier_ram) && (!scratch || scratch->publication.shape != scratch_shape ||
        !scratch->publication.attempted)) return {true, ConstructionModeReason::trial,
            earlier_ram ? ram_prior.oldest_utc : 0, earlier_ram ? ram_prior.samples : 0};
    return {partial, partial ? ConstructionModeReason::partial_batch : ConstructionModeReason::ordinary};
}
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
    long double scratch_bytes = 0;
};
struct ExecutionEstimate
{
    long double completion = 0, generation = 0, cpu = 0, scratch = 0, destination = 0;
    bool known = false;
    // A partial schedule prices its observed work, including any receipted
    // input. Unmeasured phases have no finite upper bound.
    size_t unmeasured = 0;
    long double lower_bound = 0;
    bool dated_route_prior = false;
    long double target_release = 0;
    unsigned route_prior_min_samples = 0;
    uint64_t route_prior_oldest_utc = 0;
};
struct ExecutionLane
{
    ExecutionWork work;
    size_t destination = 0;
    struct Scratch { size_t resource; long double weight; long double transfer_ns = 0; bool rotating = false; };
    std::vector<Scratch> scratch;
    bool parallel_scratch = false;
    bool committed = false;
    // A committed receipt or writer can still be outstanding at this instant.
    // Its time is relative to this forecast, on the owner's clock.
    long double ready_after = 0;
};
struct ExecutionReady
{
    size_t index = 0, deferrals = 0;
    long double remaining = 0;
    bool known = false;
    uint16_t members_to_pack = UINT16_MAX;
    bool tight_capacity = false;
    long double time_to_write = 0;
};

// A cold compact-input trial compares whole durable cohorts in this run.
// Completed appends cannot establish this result: their final pack is still
// outstanding. Owners, physical-context tokens and group membership must match
// the ordinary cohort, and an incomplete comparison never becomes a win.
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
inline bool execution_generation_precedes(ExecutionReady const& a, ExecutionReady const& b, bool timed_writes)
{
    if (a.tight_capacity != b.tight_capacity) return !a.tight_capacity;
    if (timed_writes && a.members_to_pack != UINT16_MAX && b.members_to_pack != UINT16_MAX &&
        a.time_to_write != b.time_to_write) return a.time_to_write < b.time_to_write;
    if (a.members_to_pack != b.members_to_pack) return a.members_to_pack < b.members_to_pack;
    if (a.known != b.known) return a.known;
    return a.known && a.remaining > b.remaining;
}
inline size_t execution_generation_order(std::span<ExecutionReady const> ready)
{
    require(!ready.empty(), "generation order requires a ready destination");
    auto oldest = std::max_element(ready.begin(), ready.end(), [](auto const& a, auto const& b)
        { return a.deferrals < b.deferrals; });
    if (oldest->deferrals >= ready.size()) return oldest->index;
    // Delay speculative capacity probes while other drives can use a member.
    // With cold timing, let an idle writer start packing a partially filled
    // group. Equal candidates retain the caller's rotating order.
    bool const all_tight = std::all_of(ready.begin(), ready.end(), [](auto const& item) { return item.tight_capacity; });
    bool const timed_writes = std::all_of(ready.begin(), ready.end(), [&](auto const& item)
        { return item.members_to_pack == UINT16_MAX || item.time_to_write > 0 || (!all_tight && item.tight_capacity); });
    return std::min_element(ready.begin(), ready.end(), [&](auto const& a, auto const& b)
        { return execution_generation_precedes(a, b, timed_writes); })->index;
}
inline ExecutionEstimate pipeline_execution_estimate(std::span<ExecutionLane const> lanes,
    size_t active, unsigned workers, unsigned granted, size_t resources, bool generation_overlap,
    long double concurrent_scratch_bandwidth = 0, size_t target = SIZE_MAX)
{
    ExecutionEstimate result;
    if (lanes.empty() || !active || active > lanes.size() || !workers || !granted || !resources) return result;
    // The order is a proposed dispatch order. Model release at publication,
    // rather than dividing aggregate disk time by the number of destinations.
    // Combine consecutive batches on very large fills to bound planning work.
    struct Demand
    {
        size_t id;
        long double service;
    };
    struct State
    {
        std::vector<Demand> scratch;
        size_t blocks = 0, consumed = 0;
        long double consumer = 0;
        bool admitted = false;
    };
    std::vector<State> states(lanes.size());
    std::vector<long double> demand(resources);
    size_t const max_blocks = std::max(size_t(1), size_t(4096) / lanes.size());
    long double cpu = 0, scratch_bytes = 0;
    for (size_t index = 0; index < lanes.size(); ++index)
    {
        auto const& lane = lanes[index]; auto& state = states[index];
        auto const& work = lane.work;
        if (lane.destination >= resources || !(lane.ready_after >= 0)) return {};
        if (!work.known) ++result.unmeasured;
        state.blocks = std::clamp<uint64_t>(work.batches, 1, max_blocks);
        long double const scratch = work.scratch_read + work.scratch_write;
        long double weights = 0;
        for (auto const& part : lane.scratch)
        {
            if (part.resource >= resources || !(part.weight > 0) || !std::isfinite(part.weight)) return {};
            weights += part.weight;
        }
        if (scratch && !weights) return {};
        scratch_bytes += work.scratch_bytes;
        demand[lane.destination] += work.destination;
        long double scratch_service = 0;
        for (auto const& part : lane.scratch)
        {
            long double const service = std::max(part.transfer_ns, scratch * (part.weight / weights));
            // Several stripes or partitions can resolve to one physical disk.
            auto at = std::find_if(state.scratch.begin(), state.scratch.end(),
                [&](auto const& prior) { return prior.id == part.resource; });
            if (at == state.scratch.end()) state.scratch.push_back({part.resource, service});
            else at->service += service;
            demand[part.resource] += service; scratch_service += service;
        }
        result.generation += work.generation; cpu += work.cpu + work.serial;
        result.scratch += scratch_service; result.destination += work.destination;
        result.lower_bound = std::max(result.lower_bound, lane.ready_after +
            std::max({work.generation, work.cpu / workers + work.serial, work.destination}));
    }
    result.cpu = cpu / std::min<long double>(workers, active * static_cast<long double>(granted));
    result.lower_bound = std::max({result.lower_bound, result.generation, result.cpu,
        *std::max_element(demand.begin(), demand.end()),
        concurrent_scratch_bandwidth > 0 ? scratch_bytes / concurrent_scratch_bandwidth : 0});
    enum Stage { generate, consume, publish };
    struct Event { long double at; size_t sequence, lane; Stage stage; };
    auto later = [](Event const& a, Event const& b)
    { return std::tie(a.at, a.sequence) > std::tie(b.at, b.sequence); };
    std::priority_queue<Event, std::vector<Event>, decltype(later)> events(later);
    size_t sequence = 0, live = 0, published = 0, first = 0;
    std::vector<unsigned> held(resources), targets(resources);
    std::vector<long double> disk(resources);
    std::priority_queue<long double, std::vector<long double>, std::greater<>> codecs;
    for (size_t i = 0; i < std::min(active, size_t(std::max(1u, workers / granted))); ++i) codecs.push(0);
    long double producer = 0, writers = 0, shared_scratch = 0;
    auto admit = [&](size_t i, long double at)
    {
        ++live; states[i].admitted = true; ++targets[lanes[i].destination];
        for (auto const& part : lanes[i].scratch) if (part.rotating) ++held[part.resource];
        events.push({std::max(at, lanes[i].ready_after), sequence++, i, generate});
    };
    auto fill = [&](long double at)
    {
        while (first < lanes.size() && states[first].admitted) ++first;
        for (size_t i = first; i < lanes.size() && live < active; ++i)
            if (!states[i].admitted && !held[lanes[i].destination] && std::none_of(
                lanes[i].scratch.begin(), lanes[i].scratch.end(),
                [&](auto const& part) { return part.rotating && targets[part.resource]; })) admit(i, at);
    };
    for (size_t i = 0; i < lanes.size(); ++i) if (lanes[i].committed) admit(i, 0);
    if (live > active) return {};
    fill(0);
    while (!events.empty())
    {
        auto const [at, order, i, stage] = events.top(); events.pop();
        auto const& lane = lanes[i]; auto const& work = lane.work; auto& state = states[i];
        if (stage == publish)
        {
            ++published; --live; --targets[lane.destination];
            result.completion = std::max(result.completion, at);
            if (i == target) result.target_release = at;
            for (auto const& part : lane.scratch) if (part.rotating) --held[part.resource];
            fill(at); continue;
        }
        if (stage == generate)
        {
            // Already received input can go straight to its consumer even if
            // the producer has since been booked by a different destination.
            auto ready = at;
            if (work.generation > 0)
            {
                producer = std::max({producer, at, generation_overlap ? 0.L : writers}) + work.generation / state.blocks;
                ready = producer;
            }
            events.push({std::max(ready, state.consumer), sequence++, i, consume});
            continue;
        }
        auto const start = generation_overlap ? at : std::max(at, producer);
        auto cpu_done = start;
        if (work.cpu || work.serial)
        {
            cpu_done = std::max(start, codecs.top()) + (work.cpu / std::min(workers, granted) + work.serial) / state.blocks;
            codecs.pop(); codecs.push(cpu_done);
        }
        auto const io_start = work.io_overlap ? start : cpu_done;
        auto scratch_done = io_start;
        for (auto const& part : state.scratch) if (part.service > 0)
        {
            auto& free = disk[part.id];
            free = std::max(free, lane.parallel_scratch ? io_start : scratch_done) + part.service / state.blocks;
            scratch_done = std::max(scratch_done, free);
        }
        if (concurrent_scratch_bandwidth > 0 && work.scratch_bytes > 0)
        {
            shared_scratch = std::max(io_start, shared_scratch) + work.scratch_bytes / concurrent_scratch_bandwidth / state.blocks;
            scratch_done = std::max(scratch_done, shared_scratch);
        }
        auto written = work.io_overlap ? io_start : scratch_done;
        if (work.destination)
        {
            auto& free = disk[lane.destination];
            free = std::max(free, written) + work.destination / state.blocks; written = free;
        }
        state.consumer = std::max({cpu_done, scratch_done, written});
        writers = std::max(writers, state.consumer);
        if (++state.consumed == state.blocks) events.push({state.consumer, sequence++, i, publish});
        else events.push({generation_overlap ? start : state.consumer, sequence++, i, generate});
    }
    if (published != lanes.size() || !std::isfinite(result.completion)) return {};
    result.known = !result.unmeasured && result.completion > 0;
    return result;
}
inline constexpr size_t execution_forecast_lanes = std::numeric_limits<uint64_t>::digits;
inline constexpr size_t exact_terminal_batches = 6;

struct TerminalGenerationBatch
{
    size_t index = 0, writer = 0;
    unsigned workers = 0;
    uint16_t first = 0, count = 0;
    std::array<uint8_t, 32> plot{};
    std::array<long double, 4> service{};
    long double time_to_write = 0;
    bool operator==(TerminalGenerationBatch const&) const = default;
};

// A short terminal plan can survive its own pending grants. The controller
// still checks the current receive, CPU, writer and worker availability.
struct TerminalGenerationPlan
{
    std::vector<TerminalGenerationBatch> batches;
    size_t count = 0, issued = 0;
    GenerationSupply supply;
    uint64_t deadline = 0;
    void clear() { count = issued = 0; supply = {}; }
    std::optional<size_t> choose(std::span<ExecutionReady const> ready, std::span<ExecutionLane const> lanes,
        std::span<TerminalGenerationBatch const> inputs, size_t active, unsigned workers, unsigned grant,
        GenerationSupply const& current, bool prefix, uint64_t now, uint64_t timeout);
    std::optional<size_t> next(std::span<TerminalGenerationBatch const> inputs, std::span<ExecutionReady const> ready,
        GenerationSupply const& current, uint64_t now) const
    {
        if (!count || now >= deadline || current != supply || inputs.size() != count - issued || ready.size() != inputs.size()) return {};
        for (size_t i = 0; i < inputs.size(); ++i)
            if (ready[i].tight_capacity || ready[i].deferrals >= ready.size() ||
                std::find(inputs.begin(), inputs.end(), batches[issued + i]) == inputs.end()) return {};
        return batches[issued].index;
    }
    void grant(size_t index, uint16_t members)
    {
        if (!count) return;
        if (batches[issued].index != index || batches[issued].count != members) { clear(); return; }
        if (++issued == count) clear();
    }
};

// Price only dispatch orders the owner can issue before and after its age fence.
// Equal completion retains ordinary priority.
inline std::optional<size_t> execution_terminal_first(std::span<ExecutionReady const> ready,
    std::span<ExecutionLane const> lanes, unsigned workers, unsigned grant, std::span<size_t> exact_order = {})
{
    if (ready.size() < 2 || ready.size() > execution_forecast_lanes || ready.size() != lanes.size() || !workers || !grant) return {};
    std::vector<size_t> order(lanes.size());
    std::vector<long double> tail(lanes.size());
    std::set<size_t> writers;
    long double max_cpu = 0, min_generation = std::numeric_limits<long double>::max();
    for (size_t i = 0; i < lanes.size(); ++i)
    {
        auto const& work = lanes[i].work;
        if (!ready[i].known || !work.known || work.generation <= 0 || work.destination <= 0 || work.batches != 1 ||
            ready[i].tight_capacity || ready[i].deferrals >= ready.size() || lanes[i].committed || lanes[i].ready_after ||
            !lanes[i].scratch.empty() || work.scratch_read || work.scratch_write || work.scratch_bytes ||
            !writers.insert(lanes[i].destination).second) return {};
        auto const cpu = work.cpu / std::min(workers, grant) + work.serial;
        tail[i] = work.io_overlap ? std::max(cpu, work.destination) : cpu + work.destination;
        max_cpu = std::max(max_cpu, cpu); min_generation = std::min(min_generation, work.generation);
        if (!(cpu >= 0 && std::isfinite(tail[i]) && std::isfinite(work.generation))) return {};
    }
    // Each mask holds equal-priority destinations in physical visiting order.
    // Ranks only advance as grants remove work, so replaying a complete owner
    // continuation visits at most one rank per destination for each rule.
    auto ranks = [&](auto precedes)
    {
        std::array<uint64_t, execution_forecast_lanes> groups{};
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), precedes);
        size_t group = 0;
        for (size_t i = 0; i < order.size(); ++i)
        {
            if (i && precedes(order[i - 1], order[i])) ++group;
            groups[group] |= uint64_t(1) << order[i];
        }
        return groups;
    };
    auto const ages = ranks([&](auto a, auto b) { return ready[a].deferrals > ready[b].deferrals; });
    auto priority = [&](bool timed)
    { return ranks([&](auto a, auto b) { return execution_generation_precedes(ready[a], ready[b], timed); }); };
    uint64_t untimed = 0;
    for (size_t i = 0; i < ready.size(); ++i)
        if (ready[i].members_to_pack != UINT16_MAX && !(ready[i].time_to_write > 0)) untimed |= uint64_t(1) << i;
    auto const warm = priority(true), cold = untimed ? priority(false) : warm;
    std::vector<size_t> projected(order.size());
    auto follow = [&](std::span<size_t const> preferred, bool require_order = false)
    {
        uint64_t pending = UINT64_MAX >> (execution_forecast_lanes - order.size());
        size_t next = 0, age = 0, known = 0, unknown = 0;
        bool retained = true;
        auto head = [&](auto const& groups, size_t& rank)
        {
            while (!(groups[rank] & pending)) ++rank;
            return groups[rank] & pending;
        };
        for (size_t step = 0; step < projected.size(); ++step)
        {
            auto group = head(ages, age);
            bool const aged = ready[std::countr_zero(group)].deferrals + step >= order.size() - step;
            if (aged || step >= preferred.size()) retained = false;
            if (!retained && !aged) group = pending & untimed ? head(cold, unknown) : head(warm, known);
            auto const later = group & (UINT64_MAX << next);
            size_t const index = retained ? preferred[step] : size_t(std::countr_zero(later ? later : group));
            if (require_order && preferred[step] != index) return false;
            projected[step] = index; pending &= ~(uint64_t(1) << index);
            next = index + 1 == ready.size() ? 0 : index + 1;
        }
        return true;
    };
    follow({});
    order = projected;
    auto const codecs = std::max(1u, workers / grant);
    if (order.size() > exact_terminal_batches && (codecs >= order.size() || max_cpu <= codecs * min_generation))
    {
        // Every release has a free codec grant, including reuse after at
        // least one generation interval per grant. Each writer is independent.
        auto finish = [&](std::span<size_t const> sequence, size_t first = SIZE_MAX)
        {
            long double released = 0, end = 0;
            auto append = [&](size_t i)
            {
                released += lanes[i].work.generation;
                end = std::max(end, released + tail[i]);
            };
            if (first != SIZE_MAX) append(first);
            for (auto i : sequence) if (i != first) append(i);
            return end;
        };
        auto longest_first = order;
        std::stable_sort(longest_first.begin(), longest_first.end(), [&](auto a, auto b) { return tail[a] > tail[b]; });
        // Among jobs with a tail at least q, the last generation must precede
        // that much downstream service. Descending tails attain this bound.
        auto const lower = finish(longest_first);
        auto best = finish(order);
        std::optional<size_t> first;
        for (auto i : order)
        {
            if (best <= lower) break;
            if (i == order.front() || finish(longest_first, i) >= best) continue;
            follow(std::span(&i, 1));
            auto const candidate = finish(projected);
            if (candidate < best) { best = candidate; first = ready[i].index; }
        }
        if (exact_order.size() >= order.size() && lower < best)
        {
            // Accept a longer plan only when the rotating owner can actually
            // attain the physical bound. Releasing the plan early at any
            // unchanged prefix must remain at least as good as the last one.
            auto previous = finish(order);
            for (size_t prefix = 1; prefix <= order.size(); ++prefix)
            {
                follow(std::span(longest_first).first(prefix));
                auto const candidate = finish(projected);
                if (candidate > previous) break;
                if (candidate == lower)
                {
                    std::copy(projected.begin(), projected.end(), exact_order.begin());
                    return ready[projected.front()].index;
                }
                previous = candidate;
            }
        }
        return first;
    }
    auto const longest = std::max_element(order.begin(), order.end(), [&](auto a, auto b) { return tail[a] < tail[b]; });
    bool const enumerate_orders = order.size() <= exact_terminal_batches;
    if (!enumerate_orders && longest == order.begin()) return {};
    std::vector<ExecutionLane> schedule(lanes.size());
    auto estimate = [&]
    {
        for (size_t i = 0; i < projected.size(); ++i) schedule[i] = {lanes[projected[i]].work, projected[i]};
        return pipeline_execution_estimate(schedule, schedule.size(), workers, grant, schedule.size(), true);
    };
    auto const ordinary = estimate();
    if (!ordinary.known) return {};
    if (enumerate_orders)
    {
        // Ranks preserve rotating priority even when two distinct batches have
        // equal timing keys. Only physically attainable orders enter the model.
        std::array<size_t, exact_terminal_batches> rank{};
        for (size_t i = 0; i < order.size(); ++i) rank[order[i]] = i;
        long double best = ordinary.completion;
        std::optional<size_t> first;
        while (std::next_permutation(order.begin(), order.end(), [&](auto a, auto b) { return rank[a] < rank[b]; }))
        {
            if (!follow(order, true)) continue;
            auto const candidate = estimate();
            if (!candidate.known) return {};
            if (candidate.completion >= best) continue;
            best = candidate.completion; first = ready[order.front()].index;
            if (exact_order.size() >= order.size()) std::copy(order.begin(), order.end(), exact_order.begin());
        }
        return first;
    }
    follow(std::span(&*longest, 1));
    auto const changed = estimate();
    if (changed.known && changed.completion < ordinary.completion) return ready[*longest].index;
    return {};
}

inline std::optional<size_t> TerminalGenerationPlan::choose(std::span<ExecutionReady const> ready,
    std::span<ExecutionLane const> lanes, std::span<TerminalGenerationBatch const> inputs, size_t active,
    unsigned workers, unsigned grant, GenerationSupply const& current, bool prefix, uint64_t now, uint64_t timeout)
{
    if (count)
    {
        if (prefix) if (auto saved = next(inputs, ready, current, now)) return saved;
        clear(); return {};
    }
    if (ready.size() != active || active > execution_forecast_lanes ||
        (active <= exact_terminal_batches && inputs.size() != active)) return {};
    std::array<size_t, execution_forecast_lanes> order{};
    order.front() = SIZE_MAX;
    auto const saved = inputs.size() == active ? std::span(order).first(active) : std::span<size_t>{};
    auto first = execution_terminal_first(ready, lanes, workers, grant, saved);
    if (first && order.front() != SIZE_MAX)
    {
        long double generation = 0;
        batches.resize(active); count = active; issued = 0; supply = current;
        for (size_t i = 0; i < count; ++i) { batches[i] = inputs[order[i]]; generation += lanes[i].work.generation; }
        auto duration = std::min(timeout, UINT64_MAX - now);
        if (generation < duration) duration = uint64_t(generation);
        deadline = now + duration;
    }
    return first;
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

// Put the short batch before full batches to minimize repeated prefix work.
// The caller preserves pending transactions and already generated members.
inline uint16_t construction_initial_batch(uint16_t remaining, uint16_t admitted)
{
    if (remaining == UINT16_MAX || !admitted || admitted >= remaining) return admitted;
    return uint16_t(1 + (remaining - 1) % admitted);
}

inline ConstructionForecast construction_forecast(uint8_t k, Config const& config,
    uint16_t current, uint64_t bytes, uint16_t batch, uint16_t maximum, long double estimated_member_bytes = 0,
    bool short_first = false)
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
        uint32_t const limit = first == current && short_first && maximum != UINT16_MAX ?
            construction_initial_batch(uint16_t(maximum - current), batch) : batch;
        uint32_t const added = std::min<uint32_t>(limit, result.members - first);
        long double const growth = added * member_bytes, next_bytes = old_bytes + growth;
        auto& ordinary = result.incremental;
        // A backward window whose displacement is smaller than its length
        // needs redo. Disjoint writes also checkpoint before crossing the
        // previous durable source frontier, limiting coalescing by growth.
        long double const redo = first ? next_bytes * std::min(1.0L, window / growth) : 0;
        // Near capacity the redo reserve can shrink to a single chunk.
        long double const redo_unit = next_bytes + window + 8192 <= config.capacity_bytes ?
            window / 2 : std::min(window / 2, std::max(4096.0L, 2 * next_bytes / chunks));
        ordinary.read += 2 * old_bytes;
        ordinary.write += next_bytes + redo;
        ordinary.codec += first ? 2 * old_bytes + 2 * next_bytes : next_bytes;
        long double const interval = first ? std::min(transaction, growth / 2) : transaction;
        ordinary.syncs += 9 + 2 * transactions(next_bytes - redo, interval) +
            4 * transactions(redo, redo_unit);
        ++result.incremental_batches; first += added; old_bytes = next_bytes;
    }
    // Merged contributions keep their full-first schedule and rank balance.
    // Price them independently of the incremental prefix rewrite order.
    old_bytes = bytes;
    for (uint32_t first = current; first < result.members;)
    {
        uint32_t const added = std::min<uint32_t>(batch, result.members - first);
        long double const growth = added * member_bytes, next_bytes = old_bytes + growth;
        if (count == runs.size()) combine(std::min<size_t>(fan, count));
        long double live = growth + framing;
        for (size_t i = 0; i < count; ++i) live += runs[i].bytes;
        while (count > 1 && workspace(live, count + 1) > config.capacity_bytes)
            combine(std::min<size_t>(fan, count));
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
    ExecutionWork result;
    result.batches = !merging && forecast.incremental_batches ? forecast.incremental_batches : forecast.batches;
    result.generation = generation; result.io_overlap = io_overlap;
    if (!result.batches) return result;
    bool measured = generation > 0;
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
        if (history.count) result.destination = io(forecast.incremental, construction_rates(history));
        auto const rates = execution_cpu_rates(history);
        if (rates) cpu(forecast.incremental, *rates); else measured = false;
    }
    else
    {
        auto const& contributions = external ? history.execution_contributions : history.contributions;
        auto const& packing = external ? history.execution_packing : history.packing;
        auto const contribution_cpu = execution_cpu_rates(contributions), packing_cpu = execution_cpu_rates(packing);
        auto const& all = forecast.merged; auto const& pack = forecast.packing; auto const& place = forecast.layout;
        ConstructionCost runs{std::max(0.0L, all.read - pack.read - place.read),
            std::max(0.0L, all.write - pack.write - place.write), std::max(0.0L, all.codec - pack.codec),
            std::max(0.0L, all.syncs - pack.syncs - place.syncs),
            std::max(0.0L, all.read_calls - pack.read_calls - place.read_calls),
            std::max(0.0L, all.write_calls - pack.write_calls - place.write_calls), all.serial_codec};
        if (contribution_cpu) cpu(runs, *contribution_cpu); else measured = false;
        if (packing_cpu) cpu(pack, *packing_cpu); else measured = false;
        if (external)
        {
            auto const packed_read = execution_io_rate(packing, true, true);
            auto const written = execution_io_rate(contributions, true, false);
            auto read = execution_io_rate(contributions, true, true);
            if (!read) read = packed_read;
            if (!read || !packed_read || !written) measured = false;
            result.scratch_read = runs.read * read.value_or(0) + pack.read * packed_read.value_or(0);
            result.scratch_write = runs.write * written.value_or(0);
            result.scratch_bytes = runs.read + pack.read + runs.write;
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
            if (!metadata || !target || !std::isfinite(*target)) measured = false;
            result.destination = forecast.batches * metadata.value_or(0);
            if (target && std::isfinite(*target)) result.destination += pack.write * *target;
        }
        else
        {
            if (history.contributions.count) result.destination += io(runs, construction_rates(history.contributions));
            if (history.packing.count) result.destination += io(pack, construction_rates(history.packing));
            if (history.layout.count) result.destination += io(place, construction_rates(history.layout));
            else measured = false;
        }
    }
    result.known = measured; return result;
}

inline bool prefer_merged(ConstructionForecast const& forecast, Config const& config,
                          ConstructionSample const& sample = {})
{
    if ((forecast.incremental_batches ? forecast.incremental_batches : forecast.batches) <= 1 ||
        forecast.minimum_merge_bytes > config.capacity_bytes) return false;
    // A resumed ordinary file has no timed append yet. Use relative work
    // until a completed append supplies syscall latency and codec throughput.
    auto const rates = construction_rates(sample);
    return construction_cost(forecast.merged, config, rates) <
        construction_cost(forecast.incremental, config, rates) * (1 - construction_margin(sample.elapsed_ns ? 1 : 0));
}
inline bool prefer_merged(ConstructionForecast const& forecast, Config const& config,
                          ConstructionHistory const& history)
{
    if ((forecast.incremental_batches ? forecast.incremental_batches : forecast.batches) <= 1 ||
        forecast.minimum_merge_bytes > config.capacity_bytes) return false;
    auto const rates = construction_rates(history);
    return construction_merged_cost(forecast, config, history) <
        construction_cost(forecast.incremental, config, rates) * (1 - construction_margin(history.count));
}
} // namespace group_grow
