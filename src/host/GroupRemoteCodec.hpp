// Optional bounded final-chunk encoding through compatible network workers.
#pragma once
#include "host/GroupCodec.hpp"
#include "host/ComputeWorker.hpp"

namespace group_grow
{
class PoolRemoteEncoder final : public RemoteEncoder
{
    struct Budget
    {
        std::mutex mutex;
        uint64_t capacity = 0, used = 0;
    };
    struct Reservation
    {
        std::shared_ptr<Budget> budget;
        uint64_t bytes;
        Reservation(std::shared_ptr<Budget> owner, uint64_t size) : budget(std::move(owner)), bytes(size) {}
        ~Reservation() { std::lock_guard lock(budget->mutex); budget->used -= bytes; }
        void retain(uint64_t size)
        {
            group_tools::require(size <= bytes, "remote codec retained allocation exceeds admission");
            std::lock_guard lock(budget->mutex); budget->used -= bytes - size; bytes = size;
        }
    };
    struct Shape
    {
        uint8_t k = 0;
        unsigned bytes = 0, chunks = 0;
        bool raw = false;
        bool operator==(Shape const&) const = default;
    };
    struct LocalSample
    {
        Shape shape;
        uint64_t windows = 0, next_probe = 1, remote_windows = 0, used = 0;
        long double ns_per_byte = 0;
        bool probing = false;
    };
    struct WorkerSample
    {
        Shape shape;
        std::string identity, instance;
        long double ns_per_byte = 0;
        uint64_t used = 0;
        bool valid = false;
    };
    // These fixed-size registries bound policy state across changing worker
    // processes and window shapes. A policy lock never covers network work.
    static constexpr size_t sample_limit = 64;
    struct DispatchResult
    {
        RemoteEncodedWindow window;
        std::vector<WorkerTaskObservation> workers;
        uint64_t elapsed = 0, work = 0;
    };
    struct ContentHash
    {
        struct sha256 context;
        std::array<uint64_t, 512> values{};
        size_t size = 0;
        ContentHash() { sha256_init(&context); }
        void flush()
        {
            if (size) sha256_update(&context, values.data(), size * sizeof(uint64_t));
            size = 0;
        }
        void append(uint64_t value)
        { values[size++] = value; if (size == values.size()) flush(); }
        std::array<uint8_t, 32> finish()
        {
            flush(); std::array<uint8_t, 32> digest{};
            sha256_sum(&context, digest.data()); return digest;
        }
    };
public:
    struct Counters
    {
        std::atomic<uint64_t> accepted_windows{0}, accepted_chunks{0}, fallback_windows{0},
            calibration_windows{0}, calibration_attempts{0}, input_bytes{0}, output_bytes{0};
    } counters;

    PoolRemoteEncoder(uint64_t budget, uint32_t frame, uint64_t task_bytes,
        std::function<std::shared_ptr<WorkerClient>()> client,
        std::shared_ptr<CodecScheduler> scheduler)
        : budget_(std::make_shared<Budget>()), frame_(frame), task_bytes_(task_bytes),
          client_(std::move(client)), scheduler_(std::move(scheduler))
    {
        // Worker identities and instance tokens have bounded wire lengths.
        uint64_t const policy_bytes = sizeof(*this) + sample_limit * 512;
        budget_->capacity = budget > policy_bytes ? budget - policy_bytes : 0;
    }

    std::optional<RemoteEncodedWindow> encode(uint8_t k, std::span<ChunkInput const> chunks, bool summaries) override
    {
        auto const [shape, work] = describe(k, chunks);
        long double local = 0;
        {
            std::lock_guard lock(policy_);
            auto& sample = local_sample(shape);
            local = sample.ns_per_byte;
            bool const measured = local > 0 && std::any_of(workers_.begin(), workers_.end(), [&](auto const& worker)
            { return worker.valid && worker.shape == shape && worker.ns_per_byte < local; });
            if (!measured || ++sample.remote_windows >= 64)
            {
                if (sample.remote_windows >= 64)
                { sample.remote_windows = 0; sample.next_probe = sample.windows + 1; }
                counters.fallback_windows.fetch_add(1); return {};
            }
        }
        auto result = dispatch(k, chunks, summaries, shape, local, false,
            std::min<long double>(5000000000.0L, std::max<long double>(25000000.0L, local * work * 2)));
        if (!result) { counters.fallback_windows.fetch_add(1); return {}; }
        record(*result, shape);
        counters.accepted_windows.fetch_add(1); counters.accepted_chunks.fetch_add(chunks.size());
        return std::move(result->window);
    }

    void observe_local(uint8_t k, std::span<ChunkInput const> chunks,
        std::span<uint64_t const> sizes, std::span<EncodedBlock const> blocks, uint64_t elapsed) override
    {
        if (!elapsed || chunks.empty()) return;
        auto const [shape, work] = describe(k, chunks);
        if (!work) return;
        long double const local = static_cast<long double>(elapsed) / work;
        {
            std::lock_guard lock(policy_);
            auto& sample = local_sample(shape);
            sample.ns_per_byte = local; ++sample.windows;
            if (sample.probing || sample.windows < sample.next_probe) return;
            sample.probing = true; sample.next_probe = sample.windows + 64;
        }
        // Local bytes are already ready. A speculative comparison may spend
        // at most their measured cost, capped at 250 ms even when the normal
        // generation deadline is much longer. No asynchronous borrower can
        // outlive these source spans.
        counters.calibration_attempts.fetch_add(1);
        std::optional<DispatchResult> result;
        try { result = dispatch(k, chunks, false, shape, local, true, std::min<uint64_t>(elapsed, 250000000)); }
        catch (...)
        {
            std::lock_guard lock(policy_); local_sample(shape).probing = false; throw;
        }
        bool identical = bool(result);
        size_t position = 0;
        if (result) for (auto const& block : blocks)
        {
            if (position > result->window.bytes.size() || block.bytes.size() > result->window.bytes.size() - position ||
                !std::equal(block.bytes.begin(), block.bytes.end(), result->window.bytes.begin() + position))
            { identical = false; break; }
            position += block.bytes.size();
        }
        if (result && (position != result->window.bytes.size() || sizes.size() != result->window.sizes.size() ||
            !std::equal(sizes.begin(), sizes.end(), result->window.sizes.begin()))) identical = false;
        {
            std::lock_guard lock(policy_); local_sample(shape).probing = false;
        }
        if (identical)
        { record(*result, shape); counters.calibration_windows.fetch_add(1); }
        else if (result)
        {
            std::lock_guard lock(policy_);
            for (auto const& worker : result->workers) worker_sample(shape, worker).valid = false;
        }
    }

private:
    static uint64_t add(uint64_t a, uint64_t b)
    { group_tools::require(b <= UINT64_MAX - a, "remote codec byte count overflow"); return a + b; }
    static uint64_t multiply(uint64_t a, uint64_t b)
    { group_tools::require(!b || a <= UINT64_MAX / b, "remote codec byte count overflow"); return a * b; }

    static std::pair<Shape, uint64_t> describe(uint8_t k, std::span<ChunkInput const> chunks)
    {
        uint64_t work = 0; bool raw = false;
        for (auto const& chunk : chunks)
        {
            work = add(work, chunk.old_chunk.size());
            for (auto const& part : chunk.contributions) work = add(work, part.bytes.size());
            for (size_t i = 0; i < chunk.new_members.size(); ++i)
            { raw = true; work = add(work, multiply(chunk.new_members[i].size(), sizeof(uint64_t))); }
        }
        return {{k, unsigned(std::bit_width(work)), unsigned(std::bit_width(chunks.size())), raw}, work};
    }
    LocalSample& local_sample(Shape shape)
    {
        auto found = std::find_if(locals_.begin(), locals_.end(), [&](auto const& value) { return value.shape == shape; });
        if (found == locals_.end())
        {
            found = std::min_element(locals_.begin(), locals_.end(), [](auto const& a, auto const& b) { return a.used < b.used; });
            *found = {}; found->shape = shape;
        }
        found->used = ++clock_; return *found;
    }
    WorkerSample& worker_sample(Shape shape, WorkerTaskObservation const& worker)
    {
        auto found = std::find_if(workers_.begin(), workers_.end(), [&](auto const& value)
        { return value.shape == shape && value.identity == worker.worker_identity && value.instance == worker.worker_instance; });
        if (found == workers_.end())
        {
            found = std::min_element(workers_.begin(), workers_.end(), [](auto const& a, auto const& b) { return a.used < b.used; });
            *found = {}; found->shape = shape; found->identity = worker.worker_identity; found->instance = worker.worker_instance;
        }
        found->used = ++clock_; return *found;
    }
    void record(DispatchResult const& result, Shape shape)
    {
        std::lock_guard lock(policy_);
        uint64_t elapsed = 0, units = 0;
        for (auto const& worker : result.workers)
        { elapsed = add(elapsed, worker.elapsed_ns); units = add(units, worker.work_units); }
        long double const preparation = static_cast<long double>(result.elapsed - std::min(result.elapsed, elapsed)) / result.work;
        for (auto const& worker : result.workers)
        {
            if (!worker.work_units || !units) continue;
            auto& sample = worker_sample(shape, worker);
            // Each split task keeps its worker's measured end-to-end cost.
            // Amortize only the common owner preparation across the window.
            long double const work = static_cast<long double>(result.work) * worker.work_units / units;
            sample.ns_per_byte = worker.elapsed_ns / work + preparation;
            sample.valid = true;
        }
    }
    std::optional<DispatchResult> dispatch(uint8_t k, std::span<ChunkInput const> chunks, bool summaries,
        Shape shape, long double local, bool probe, uint64_t timeout_ns)
    {
        auto const begin = std::chrono::steady_clock::now();
        WorkerCancellation cancel; cancel.deadline = begin + std::chrono::nanoseconds(timeout_ns);
        std::shared_ptr<Reservation> reservation;
        std::vector<WorkerTaskObservation> selected;
        try
        {
            auto outcome = [&]() -> std::optional<DispatchResult>
            {
                uint64_t raw_bytes = 0, work = 0, descriptors = 0;
                for (auto const& chunk : chunks)
                {
                    work = add(work, chunk.old_chunk.size());
                    descriptors = add(descriptors, sizeof(WorkerChunkInput) + sizeof(uint64_t) + sizeof(ChunkSummary) +
                        sizeof(std::vector<uint8_t>) + sizeof(std::vector<WorkerContribution>) +
                        sizeof(std::vector<std::span<uint64_t const>>) + sizeof(std::vector<std::vector<uint64_t>>));
                    descriptors = add(descriptors, multiply(chunk.contributions.size(), sizeof(WorkerContribution)));
                    descriptors = add(descriptors, multiply(chunk.new_members.size(),
                        sizeof(std::vector<uint64_t>) + sizeof(std::span<uint64_t const>)));
                    for (auto const& contribution : chunk.contributions) work = add(work, contribution.bytes.size());
                    for (size_t member = 0; member < chunk.new_members.size(); ++member)
                        raw_bytes = add(raw_bytes, multiply(chunk.new_members[member].size(), sizeof(uint64_t)));
                }
                work = add(work, raw_bytes);
                // The portable client serializes one bounded source window while
                // raw materialization is still live. Its returned vectors may
                // replace geometric capacities, then coexist with coalesced output.
                uint64_t const fixed = add(add(raw_bytes, multiply(add(work, descriptors), 3)), add(
                    multiply(pool::Channel::memory_bytes(frame_), 2), ChunkCodec::estimate_base() +
                    2 * sizeof(ContentHash) + 256));
                uint64_t max_output = 0;
                {
                    std::lock_guard lock(budget_->mutex);
                    uint64_t const room = budget_->capacity - budget_->used;
                    if (!work || fixed >= room) return {};
                    // Reserve this window's bounded receive allowance, not the
                    // entire free partition. Other destinations may run together.
                    max_output = std::min({(room - fixed) / 6, task_bytes_, add(multiply(work, 2), multiply(chunks.size(), 2048))});
                    if (!max_output) return {};
                    auto const required = add(fixed, multiply(max_output, 6));
                    reservation = std::make_shared<Reservation>(budget_, required); budget_->used += required;
                }
                cancel.check();
                std::vector<std::vector<WorkerContribution>> contributions(chunks.size());
                std::vector<std::vector<std::vector<uint64_t>>> raw(chunks.size());
                std::vector<std::vector<std::span<uint64_t const>>> views(chunks.size());
                std::vector<WorkerChunkInput> tasks; tasks.reserve(chunks.size());
                for (size_t i = 0; i < chunks.size(); ++i)
                {
                    auto const& chunk = chunks[i];
                    contributions[i].reserve(chunk.contributions.size());
                    for (auto const& contribution : chunk.contributions)
                        contributions[i].push_back({contribution.bytes, contribution.members});
                    raw[i].reserve(chunk.new_members.size()); views[i].reserve(chunk.new_members.size());
                    for (size_t member = 0; member < chunk.new_members.size(); ++member)
                    {
                        auto const region = chunk.new_members[member];
                        raw[i].emplace_back(region.size());
                        std::copy(region.begin(), region.end(), raw[i].back().begin());
                        views[i].push_back(raw[i].back());
                    }
                    tasks.push_back({chunk.ordinal, chunk.old_chunk, chunk.old_members, contributions[i], views[i]});
                }
                std::vector<std::vector<uint8_t>> encoded; encoded.reserve(tasks.size());
                std::vector<ChunkSummary> validated;
                if (summaries) validated.resize(chunks.size());
                uint64_t const validation_budget = reservation->bytes - fixed - multiply(max_output, 4) + ChunkCodec::estimate_base();
                ChunkCodec inspector(k, validation_budget, scheduler_);
                uint64_t delivered_bytes = 0;
                auto client = client_();
                std::vector<WorkerTaskObservation> completed;
                WorkerTaskOptions options;
                options.observe = [&](WorkerTaskObservation const& worker) { completed.push_back(worker); };
                options.selected = [&](WorkerTaskObservation const& worker)
                {
                    std::lock_guard lock(policy_);
                    auto& sample = worker_sample(shape, worker);
                    if (!probe && (!sample.valid || sample.ns_per_byte >= local))
                    { auto& prior = local_sample(shape); prior.next_probe = prior.windows + 1; return false; }
                    selected.push_back(worker); return true;
                };
                auto const task_limit = std::min(task_bytes_, client->max_task_bytes());
                for (size_t first = 0; first < tasks.size();)
                {
                    // Match the portable request's serialized lengths. Both
                    // metadata count and bytes bound a request; a large local
                    // window need not pay for a new task every 64 chunks.
                    uint64_t input_bytes = pool::codec_window_header_bytes;
                    size_t count = 0;
                    while (first + count < tasks.size() && count < client->max_window_chunks())
                    {
                        auto const& chunk = tasks[first + count];
                        uint64_t bytes = add(pool::codec_chunk_header_bytes, chunk.old_chunk.size());
                        for (auto const& part : chunk.contributions) bytes = add(bytes, add(6, part.bytes.size()));
                        for (auto const& member : chunk.new_members)
                            bytes = add(bytes, add(4, multiply(member.size(), sizeof(uint64_t))));
                        if (input_bytes > task_limit || bytes > task_limit - input_bytes) break;
                        input_bytes += bytes; ++count;
                    }
                    if (!count) return {};
                    auto window = client->encode(k, std::span<WorkerChunkInput const>(tasks).subspan(first, count),
                        std::min(task_limit, max_output - delivered_bytes), cancel, {}, [&](std::span<std::vector<uint8_t> const> output)
                        {
                            group_tools::require(output.size() == count, "remote codec result count mismatch");
                            for (size_t i = 0; i < count; ++i)
                            {
                                auto const& source = chunks[first + i];
                                uint64_t members = source.old_members + source.new_members.size();
                                for (auto const& contribution : source.contributions) members = add(members, contribution.members);
                                group_tools::require(members && members <= UINT16_MAX, "remote chunk membership overflow");
                                ContentHash actual_hash, expected_hash;
                                auto const inspected = inspector.inspect(source.ordinal, output[i], uint16_t(members),
                                    [&](uint64_t value) { actual_hash.append(value); });
                                ChunkSummary expected;
                                std::optional<uint64_t> first_expected;
                                uint64_t offset = 0;
                                uint64_t const range = uint64_t(1) << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
                                auto append_summary = [&](ChunkCodec::Inspection const& part)
                                {
                                    if (!first_expected) first_expected = part.first;
                                    expected = ChunkCodec::combine(k, expected, part.summary, part.first, source.ordinal);
                                };
                                auto append_encoded = [&](std::span<uint8_t const> bytes, uint16_t count)
                                {
                                    append_summary(inspector.inspect(source.ordinal, bytes, count,
                                        [&](uint64_t value) { expected_hash.append(add(value, offset)); }));
                                    offset = add(offset, multiply(count, range));
                                };
                                if (!source.old_chunk.empty()) append_encoded(source.old_chunk, source.old_members);
                                for (auto const& contribution : source.contributions)
                                    append_encoded(contribution.bytes, contribution.members);
                                if (!source.new_members.empty())
                                {
                                    auto const fresh = ChunkCodec::fresh_summary(k, source.ordinal, source.new_members, scheduler_.get());
                                    append_summary({fresh, source.new_members[0].front()});
                                    for (size_t member = 0; member < source.new_members.size(); ++member)
                                    {
                                        std::optional<uint64_t> previous;
                                        for (auto value : source.new_members[member])
                                        {
                                            if (!previous || value != *previous) expected_hash.append(add(value, offset));
                                            previous = value;
                                        }
                                        offset = add(offset, range);
                                    }
                                }
                                auto const bounds = ChunkCodec::size_bounds(expected);
                                group_tools::require(first_expected && inspected.first == *first_expected &&
                                    inspected.summary == expected && output[i].size() >= bounds.first && output[i].size() <= bounds.second &&
                                    actual_hash.finish() == expected_hash.finish(),
                                    "remote codec content differs from its input");
                                if (summaries) validated[first + i] = inspected.summary;
                            }
                        }, options);
                    if (window.size() != count) return {};
                    for (auto& chunk : window)
                    { delivered_bytes = add(delivered_bytes, chunk.size()); encoded.push_back(std::move(chunk)); }
                    if (delivered_bytes >= max_output && first + count != tasks.size()) return {};
                    first += count;
                }
                if (encoded.size() != chunks.size()) return {};
                uint64_t output_bytes = 0;
                for (auto const& chunk : encoded) output_bytes = add(output_bytes, chunk.capacity());
                if (output_bytes > multiply(max_output, 2)) return {};
                RemoteEncodedWindow result; result.sizes.reserve(chunks.size());
                if (summaries) result.summaries = std::move(validated);
                result.bytes.reserve(size_t(output_bytes));
                for (size_t i = 0; i < chunks.size(); ++i)
                {
                    result.sizes.push_back(encoded[i].size());
                    result.bytes.insert(result.bytes.end(), encoded[i].begin(), encoded[i].end());
                }
                uint64_t const elapsed = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin).count());
                counters.input_bytes.fetch_add(work); counters.output_bytes.fetch_add(result.bytes.size());
                return DispatchResult{std::move(result), std::move(completed), elapsed, work};
            }();
            // All materialized input, decoder scratch and uncoalesced output
            // are gone before releasing their reservation to another caller.
            if (outcome)
            {
                auto& result = outcome->window;
                uint64_t metadata = multiply(outcome->workers.capacity(), sizeof(WorkerTaskObservation));
                for (auto const& worker : outcome->workers)
                    metadata = add(metadata, add(worker.worker_identity.capacity() + 1, worker.worker_instance.capacity() + 1));
                std::vector<WorkerTaskObservation>().swap(selected);
                uint64_t const retained = add(metadata, add(result.bytes.capacity(), add(multiply(result.sizes.capacity(), sizeof(uint64_t)),
                    multiply(result.summaries.capacity(), sizeof(ChunkSummary)))));
                reservation->retain(retained); result.admission = std::move(reservation);
            }
            return outcome;
        }
        catch (WorkerAdmissionRejected const&) { throw; }
        catch (std::exception const&)
        {
            std::lock_guard lock(policy_);
            for (auto const& worker : selected) worker_sample(shape, worker).valid = false;
            return {};
        }
    }

    std::shared_ptr<Budget> budget_;
    uint32_t frame_;
    uint64_t task_bytes_;
    std::function<std::shared_ptr<WorkerClient>()> client_;
    std::shared_ptr<CodecScheduler> scheduler_;
    std::mutex policy_;
    std::array<LocalSample, sample_limit> locals_{};
    std::array<WorkerSample, sample_limit> workers_{};
    uint64_t clock_ = 0;
};
} // namespace group_grow
