// Canonical upstream group chunks, with bounded reusable CPU scratch.
#pragma once
#include "host/GroupMembers.hpp"

#include <bit>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <ctime>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

namespace group_grow
{
// Runtime CPU observations exclude blocked wall time. Unsupported thread
// clocks leave an explicit missing observation instead of fabricating demand.
struct CpuTime
{
    uint64_t nanoseconds = 0;
    bool measured = true;
    void add(CpuTime const& value)
    {
        measured = measured && value.measured && value.nanoseconds <= UINT64_MAX - nanoseconds;
        if (measured) nanoseconds += value.nanoseconds;
    }
};
inline std::optional<uint64_t> thread_cpu_nanoseconds()
{
#ifdef CLOCK_THREAD_CPUTIME_ID
    struct timespec value{};
    if (!::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) && value.tv_sec >= 0 && value.tv_nsec >= 0 &&
        uint64_t(value.tv_sec) <= (UINT64_MAX - uint64_t(value.tv_nsec)) / 1000000000ull)
        return uint64_t(value.tv_sec) * 1000000000ull + uint64_t(value.tv_nsec);
#endif
    return {};
}
class ThreadCpuTimer
{
public:
    explicit ThreadCpuTimer(CpuTime& target) : target_(target), begin_(thread_cpu_nanoseconds()) {}
    ~ThreadCpuTimer()
    {
        auto const end = thread_cpu_nanoseconds();
        if (begin_ && end && *end >= *begin_) target_.add({*end - *begin_, true});
        else target_.measured = false;
    }
    ThreadCpuTimer(ThreadCpuTimer const&) = delete;
    ThreadCpuTimer& operator=(ThreadCpuTimer const&) = delete;
private:
    CpuTime& target_;
    std::optional<uint64_t> begin_;
};

// CPU admission is independent of output ownership. A blocked file consumer
// keeps its admitted codec/output memory, but holds no compute permit. The
// bounded queue and its threads are shared by all active destinations.
class CodecScheduler
{
    struct Waiter { Waiter* previous = nullptr; Waiter* next = nullptr; };
public:
    class Lease
    {
    public:
        explicit Lease(CodecScheduler* owner, unsigned permits = 1,
                       std::function<void()> wait_check = {})
            : owner_(owner), previous_(current_), permits_(permits), previous_permits_(current_permits_)
        {
            if (wait_check) wait_check();
            if (!owner_) return;
            group_tools::require(permits_ && permits_ <= owner_->workers(), "invalid CPU permit request");
            if (current_ == owner_)
            {
                group_tools::require(permits_ <= current_permits_, "nested CPU permit upgrade is not allowed");
                owner_ = nullptr; return;
            }
            group_tools::require(!current_, "cannot nest different codec schedulers");
            std::unique_lock lock(owner_->mutex_);
            // Stack-owned waiters preserve FIFO admission without retaining
            // abandoned tickets when a network task is cancelled or expires.
            Waiter waiter{owner_->last_waiter_, nullptr};
            if (waiter.previous) waiter.previous->next = &waiter;
            else owner_->first_waiter_ = &waiter;
            owner_->last_waiter_ = &waiter;
            auto remove = [&]
            {
                if (waiter.previous) waiter.previous->next = waiter.next;
                else owner_->first_waiter_ = waiter.next;
                if (waiter.next) waiter.next->previous = waiter.previous;
                else owner_->last_waiter_ = waiter.previous;
            };
            auto ready = [&]
            { return owner_->first_waiter_ == &waiter && permits_ <= owner_->workers() - owner_->active_; };
            try
            {
                if (wait_check)
                {
                    while (!ready())
                    {
                        wait_check();
                        owner_->compute_.wait_for(lock, std::chrono::milliseconds(25));
                    }
                    wait_check();
                }
                else owner_->compute_.wait(lock, ready);
            }
            catch (...)
            { remove(); lock.unlock(); owner_->compute_.notify_all(); throw; }
            remove(); owner_->active_ += permits_;
            current_ = owner_;
            current_permits_ = permits_;
            lock.unlock(); owner_->compute_.notify_all();
        }
        ~Lease()
        {
            if (!owner_) return;
            current_ = previous_;
            current_permits_ = previous_permits_;
            { std::lock_guard lock(owner_->mutex_); owner_->active_ -= permits_; }
            owner_->compute_.notify_all();
        }
        Lease(Lease const&) = delete;
        Lease& operator=(Lease const&) = delete;
    private:
        CodecScheduler* owner_;
        CodecScheduler* previous_;
        unsigned permits_;
        unsigned previous_permits_;
    };

    static uint64_t memory_bytes(unsigned workers)
    {
        uint64_t const per_worker = sizeof(std::thread) +
            sizeof(std::function<void()>) + 256;
        group_tools::require(workers && workers <= (UINT64_MAX - sizeof(CodecScheduler)) / per_worker,
            "codec scheduler memory estimate overflow");
        return sizeof(CodecScheduler) + uint64_t(workers) * per_worker;
    }
    explicit CodecScheduler(unsigned workers) : jobs_(workers)
    {
        group_tools::require(workers, "codec worker count must be positive");
        threads_.reserve(workers);
        try
        {
            for (unsigned i = 0; i < workers; ++i) threads_.emplace_back([this] { run(); });
        }
        catch (...) { stop(); throw; }
    }
    ~CodecScheduler() { stop(); }
    CodecScheduler(CodecScheduler const&) = delete;
    CodecScheduler& operator=(CodecScheduler const&) = delete;
    unsigned workers() const { return unsigned(jobs_.size()); }

    std::future<void> submit(std::function<void()> operation)
    {
        // A compute task must never wait for a task on its own bounded pool.
        group_tools::require(current_ != this, "codec task cannot submit nested work");
        auto job = std::make_shared<std::packaged_task<void()>>(std::move(operation));
        auto future = job->get_future();
        std::function<void()> invoke = [job] { (*job)(); };
        std::unique_lock lock(mutex_);
        uint64_t const ticket = next_submit_ticket_++;
        room_.wait(lock, [&]
        { return stopping_ || (ticket == serving_submit_ticket_ && queued_ < jobs_.size()); });
        group_tools::require(!stopping_, "codec scheduler has stopped");
        jobs_[(first_ + queued_) % jobs_.size()] = std::move(invoke); ++queued_; ++serving_submit_ticket_;
        lock.unlock(); ready_.notify_one(); room_.notify_all(); return future;
    }
private:
    void stop()
    {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        ready_.notify_all(); room_.notify_all();
        for (auto& thread : threads_) if (thread.joinable()) thread.join();
    }
    void run()
    {
        for (;;)
        {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return stopping_ || queued_; });
                if (!queued_) return;
                job = std::move(jobs_[first_]); first_ = (first_ + 1) % jobs_.size(); --queued_;
            }
            room_.notify_all();
            Lease compute(this); job();
        }
    }
    inline static thread_local CodecScheduler* current_ = nullptr;
    inline static thread_local unsigned current_permits_ = 0;
    std::vector<std::function<void()>> jobs_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable ready_, room_, compute_;
    Waiter* first_waiter_ = nullptr;
    Waiter* last_waiter_ = nullptr;
    uint64_t next_submit_ticket_ = 0, serving_submit_ticket_ = 0;
    size_t first_ = 0, queued_ = 0;
    unsigned active_ = 0;
    bool stopping_ = false;
};

struct UnencodableChunkSize : std::runtime_error
{
    UnencodableChunkSize() : std::runtime_error("unencodable group chunk size") {}
};

inline void check_gsz_size(uint64_t size, uint16_t members)
{
    group_tools::require(members != 0 && size != 0 && size <= uint64_t(INT64_MAX),
                         "invalid group chunk size");
    uint64_t const expected = gsz_expected_for_g(members);
    uint64_t const magnitude = size < expected ? expected - size : size - expected;
    auto const rice = uint32_t(gsz_rice_k_for_g(members));
    if ((magnitude >> rice) + rice + 2 > 64) throw UnencodableChunkSize();
}

struct EncodedChunk
{
    std::span<uint8_t const> bytes;
    uint16_t members;
};

struct ChunkInput
{
    uint64_t ordinal;
    std::span<uint8_t const> old_chunk;
    uint16_t old_members;
    MemberRegions new_members;
    std::span<EncodedChunk const> contributions = {};
};

// Additive information about the canonical delta stream. The merge engine
// keeps this small, fixed-size table inside the destination, so admitting a
// batch normally reads a table instead of every previous compressed payload.
struct ChunkSummary
{
    uint64_t symbols = 0, low_bits = 0, min_ans_bits = 0, max_ans_bits = 0, last = 0;
    bool operator==(ChunkSummary const&) const = default;
};

class ChunkCodec
{
    // Raw byte arrays make capacities exact, including temporary replacement
    // allocations. No vector growth policy can bypass the scratch admission.
    struct Bytes
    {
        std::unique_ptr<uint8_t[]> data;
        size_t size = 0, capacity = 0;
        std::span<uint8_t const> span() const { return {data.get(), size}; }
    };
    struct Scanned
    {
        size_t bytes, low_offset, low_bits;
        uint64_t first, last;
        size_t first_bits;
    };

public:
    static uint64_t estimate_base() { return fixed_scratch_bytes(); }

    // A fresh group can reserve its destination before encoding. Deltas
    // telescope across the member regions, bounding their total unary bits
    // without calculating every quotient. Duplicates only reduce the symbol
    // count. Include FSE's bound and the largest unsigned length prefix.
    static uint64_t fresh_bound(uint8_t k, uint64_t ordinal,
                               MemberRegions members, CodecScheduler* scheduler = nullptr)
    {
        CodecScheduler::Lease compute(scheduler);
        group_tools::require(k >= 18 && k <= 28 && !(k & 1) && !members.empty() &&
            members.size() <= UINT16_MAX && ordinal < PlotGroupFile::getChunkCountForK(k),
            "invalid fresh group chunk");
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        uint64_t const start = ordinal * range;
        uint64_t const threshold = 178ull << (k - 8);
        uint64_t const maximum_delta = uint64_t(72 - k) * threshold;
        uint64_t previous = start, symbols = 0;
        for (size_t member = 0; member < members.size(); ++member)
        {
            auto const fragments = members[member];
            group_tools::require(!fragments.empty(), "group member chunk is empty");
            group_tools::require(fragments.front() >= start && fragments.back() >= fragments.front() &&
                fragments.back() - start < range, "unsorted or out-of-range group fragment");
            // Immutable compact construction already checked sort/range and
            // collected the within-chunk delta invariant while packing. Raw or
            // borrowed views, and compact members with a wide gap, keep the
            // full existing validation path.
            if (!fragments.validated_deltas(k))
            {
                auto const invalid = std::adjacent_find(fragments.begin(), fragments.end(),
                    [maximum_delta](uint64_t a, uint64_t b) { return b < a || b - a >= maximum_delta; });
                group_tools::require(invalid == fragments.end() || *(invalid + 1) >= *invalid,
                                     "unsorted or out-of-range group fragment");
                group_tools::require(invalid == fragments.end(), "fresh group bound overflow");
            }
            uint64_t const first = fragments.front() + uint64_t(member) * range;
            group_tools::require(first - previous < maximum_delta && fragments.size() <= SIZE_MAX - symbols,
                                 "fresh group bound overflow");
            symbols += fragments.size();
            previous = fragments.back() + uint64_t(member) * range;
        }
        uint64_t const unary = (uint64_t(members.size()) * range - 1) / threshold;
        group_tools::require(symbols <= (UINT64_MAX - unary) / (k - 7), "fresh group bound overflow");
        uint64_t const bits = symbols * (k - 7) + unary;
        group_tools::require(symbols <= SIZE_MAX - (symbols >> 7) - 1024,
                             "fresh group ANS bound overflow");
        uint64_t const ans = POS2_FSE_compressBound(size_t(symbols));
        uint64_t const low = bits / 8 + (bits % 8 != 0);
        group_tools::require(low <= UINT64_MAX - ans - 10, "fresh group bound overflow");
        return low + ans + 10;
    }

    // Conservative admission for one exact old chunk plus incoming fragments.
    // Each old symbol requires low bits, so its count is bounded by old bytes;
    // each added delta occupies at most eight low bytes and one high byte.
    // Include FSE overhead, coalesced output, geometric capacity growth and
    // simultaneous old/replacement allocations. For a coalesced block, add
    // estimate_base() once, then sum each chunk's estimate minus that baseline.
    // Joins use the sum of encoded source bytes as old_chunk_bytes. This also
    // covers the decoder's maximum symbol reserve beside its retained prefix,
    // adjusted packed low fields, and the final ANS/output allocations.
    static uint64_t estimate_work(uint8_t k, uint64_t old_chunk_bytes, uint64_t incoming_count)
    {
        group_tools::require(k >= 18 && k <= 28 && !(k & 1), "invalid group k");
        uint64_t const overhead = fixed_scratch_bytes() + 8192;
        group_tools::require(old_chunk_bytes <= (UINT64_MAX - overhead) / 16,
            "group codec estimate overflow");
        uint64_t const old_cost = old_chunk_bytes * 16 + overhead;
        group_tools::require(incoming_count <= (UINT64_MAX - old_cost) / 64,
                             "group codec estimate overflow");
        return old_cost + incoming_count * 64;
    }

    explicit ChunkCodec(uint8_t k, uint64_t scratch_budget, std::shared_ptr<CodecScheduler> scheduler = {})
        : k_(k), budget_(scratch_budget), scheduler_(std::move(scheduler))
    {
        CodecScheduler::Lease compute(scheduler_.get());
        group_tools::require(k >= 18 && k <= 28 && !(k & 1), "invalid group k");
        group_tools::require(scratch_bytes() <= budget_, "codec scratch budget too small");
        ctable_.reset(POS2_FSE_createCTable(177, 11));
        dtable_.reset(POS2_FSE_createDTable(11));
        auto const counts = group_tools::compression_counts();
        // Every symbol consumes at least one bit with this fixed table. This
        // bounds decoded symbol count even when the next chunk follows the span.
        group_tools::require(*std::max_element(counts.begin(), counts.end()) <= 1024,
                             "unexpected group FSE table");
        group_tools::require(ctable_ && dtable_ &&
            !POS2_FSE_isError(POS2_FSE_buildCTable(ctable_.get(), counts.data(), 177, 11)) &&
            !POS2_FSE_isError(POS2_FSE_buildDTable(dtable_.get(), counts.data(), 177, 11)),
            "cannot build group FSE tables");
    }

    uint64_t scratch_bytes() const
    {
        return fixed_scratch_bytes() + high_.capacity + low_.capacity
            + ans_.capacity + output_.capacity + previous_output_.capacity;
    }

    void prepare(uint64_t estimate)
    {
        CodecScheduler::Lease compute(scheduler_.get());
        group_tools::require(estimate <= budget_, "group codec window exceeds scratch budget");
        // Keep normal worker scratch across windows. Reclaim previous unusually
        // shaped windows when their retained capacities would prevent admission.
        if (scratch_bytes() - fixed_scratch_bytes() > budget_ - estimate)
        {
            high_ = {};
            low_ = {};
            ans_ = {};
            output_ = {};
        }
        group_tools::require(previous_output_.capacity <= budget_ - estimate,
                             "borrowed output exceeds the codec window budget");
    }

    // Keep one completed block stable while the next window is encoded. Both
    // allocations remain in scratch_bytes(), including replacement growth.
    void next_window() { std::swap(output_, previous_output_); }

    // Prefix may contain following chunks. Throws on truncated or malformed
    // input; callers should provide a bounded read window. No index is needed.
    size_t scan(uint64_t ordinal, std::span<uint8_t const> prefix, uint16_t members)
    {
        CodecScheduler::Lease compute(scheduler_.get());
        auto const result = decode(ordinal, prefix, members);
        check_gsz_size(result.bytes, members);
        return result.bytes;
    }

    struct Inspection { ChunkSummary summary; uint64_t first; };
    // Network contributions carry no trusted summary. Inspect their actual
    // delta stream before incorporating it into the owner's capacity model.
    // A visitor receives the canonical, strictly ordered values including
    // each member's range offset. Validation can compare complete content
    // without materializing a second raw fragment array.
    Inspection inspect(uint64_t ordinal, std::span<uint8_t const> payload, uint16_t members,
                       std::function<void(uint64_t)> const& visit = {})
    {
        CodecScheduler::Lease compute(scheduler_.get());
        auto const result = decode(ordinal, payload, members, nullptr, nullptr, false, visit ? &visit : nullptr);
        group_tools::require(result.bytes == payload.size(), "group chunk has trailing data");
        check_gsz_size(result.bytes, members);
        return {summary(ordinal, members), result.first};
    }

    // Reuse the same validated decoder for bounded merge inputs. The caller
    // reserves its fragment/boundary arrays before any borrowed spans exist.
    void decode_members(uint64_t ordinal, std::span<uint8_t const> payload, uint16_t members,
                        std::vector<uint64_t>& fragments, std::vector<size_t>& boundaries)
    {
        CodecScheduler::Lease compute(scheduler_.get());
        auto const result = decode(ordinal, payload, members, &fragments, &boundaries);
        group_tools::require(result.bytes == payload.size(), "group chunk has trailing data");
        check_gsz_size(result.bytes, members);
    }

    static std::pair<unsigned, unsigned> symbol_bit_bounds(uint8_t symbol)
    {
        static auto const bounds = []
        {
            std::array<std::pair<unsigned, unsigned>, 178> result;
            auto const counts = group_tools::compression_counts();
            for (size_t i = 0; i < result.size(); ++i)
            {
                group_tools::require(counts[i] > 0 && counts[i] <= 1024, "unexpected group FSE symbol");
                unsigned const frequency = unsigned(counts[i]);
                result[i] = {11u - std::bit_width(frequency - 1), 11u - (std::bit_width(frequency) - 1)};
            }
            return result;
        }();
        group_tools::require(symbol < bounds.size(), "unexpected group FSE symbol");
        return bounds[symbol];
    }

    ChunkSummary summary(uint64_t ordinal, uint16_t members) const
    {
        CodecScheduler::Lease compute(scheduler_.get());
        group_tools::require(members && high_.size > 2, "missing encoded chunk summary");
        ChunkSummary result{high_.size, summary_bits_, 0, 0,
            summary_last_ - uint64_t(members - 1) * range_per_chunk()};
        group_tools::require(result.last >= ordinal * range_per_chunk() &&
            result.last - ordinal * range_per_chunk() < range_per_chunk(), "invalid encoded summary frontier");
        for (size_t i = 0; i < high_.size; ++i)
        {
            auto const bounds = symbol_bit_bounds(high_.data[i]);
            result.min_ans_bits += bounds.first;
            result.max_ans_bits += bounds.second;
        }
        return result;
    }

    // The first symbol of a contribution changes when it follows an earlier
    // member. All other delta symbols and low fields remain identical.
    ChunkSummary combine(ChunkSummary const& old, ChunkSummary next, uint64_t first,
                         uint64_t ordinal) const
    { return combine(k_, old, next, first, ordinal); }

    static ChunkSummary combine(uint8_t k, ChunkSummary const& old, ChunkSummary next, uint64_t first,
                                uint64_t ordinal)
    {
        if (!old.symbols) return next;
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS), start = ordinal * range;
        group_tools::require(first >= start && first - start < range &&
            old.last >= start && old.last - start < range, "invalid summary member boundary");
        uint64_t const standalone = first - start, joined = range + first - old.last;
        auto const before = symbol_bit_bounds(uint8_t((standalone >> (k - 8)) % 178));
        auto const after = symbol_bit_bounds(uint8_t((joined >> (k - 8)) % 178));
        auto width = [k](uint64_t delta)
        {
            uint64_t const bits = uint64_t(k - 7) + (delta >> (k - 8)) / 178;
            group_tools::require(bits <= 64, "unencodable group fragment delta"); return bits;
        };
        next.symbols += old.symbols;
        next.low_bits = old.low_bits + next.low_bits - width(standalone) + width(joined);
        next.min_ans_bits = old.min_ans_bits + next.min_ans_bits - before.first + after.first;
        next.max_ans_bits = old.max_ans_bits + next.max_ans_bits - before.second + after.second;
        return next;
    }

    static std::pair<uint64_t, uint64_t> size_bounds(ChunkSummary const& value)
    {
        // FSE initializes its final two symbols without emitting their bits,
        // then flushes two 11-bit states and one end bit. Each emitted symbol
        // uses the floor or ceiling of its normalized frequency's bit cost.
        auto bytes = [](uint64_t bits) { return bits / 8 + (bits % 8 != 0); };
        auto leb = [](uint64_t n) { uint64_t count = 1; while (n >>= 7) ++count; return count; };
        uint64_t const least = bytes((value.min_ans_bits > 22 ? value.min_ans_bits - 22 : 0) + 23);
        uint64_t const most = bytes(value.max_ans_bits + 23), low = bytes(value.low_bits);
        return {leb(least) + least + low, leb(most) + most + low};
    }

    // Validate and summarize the fresh delta stream without invoking FSE or
    // allocating encoded bytes. Admission can then use the same narrow size
    // interval as a measured chunk before its single encoding pass.
    static ChunkSummary fresh_summary(uint8_t k, uint64_t ordinal, MemberRegions members, CodecScheduler* scheduler = nullptr)
    {
        CodecScheduler::Lease compute(scheduler);
        group_tools::require(k >= 18 && k <= 28 && !(k & 1) && !members.empty() &&
            members.size() <= UINT16_MAX && ordinal < PlotGroupFile::getChunkCountForK(k),
            "invalid fresh group chunk");
        ChunkSummary result;
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS);
        uint64_t const last = visit_deltas(k, ordinal, 0, members, ordinal * range, true,
            [&](uint64_t delta)
            {
                auto const width = delta_width(k, delta);
                group_tools::require(result.symbols < SIZE_MAX && result.low_bits <= SIZE_MAX - width,
                                     "group chunk length overflow");
                ++result.symbols; result.low_bits += width;
                auto const bounds = symbol_bit_bounds(uint8_t((delta >> (k - 8)) % 178));
                result.min_ans_bits += bounds.first;
                result.max_ans_bits += bounds.second;
            });
        group_tools::require(result.symbols > 2, "missing encoded chunk summary");
        result.last = last - uint64_t(members.size() - 1) * range;
        return result;
    }

    // Returned storage belongs to this codec and lasts until the next encode.
    // Input spans must not alias that storage.
    std::span<uint8_t const> encode(uint64_t ordinal, std::span<uint8_t const> old_chunk,
        uint16_t old_members, MemberRegions new_members)
    {
        clear_output();
        append_encode(ordinal, old_chunk, old_members, new_members);
        return output();
    }

    void clear_output() { output_.size = 0; }
    std::span<uint8_t const> output() const { return output_.span(); }
    // After a successful encode, sizing, or scan, read the encoded symbol
    // count before the next codec operation can replace that chunk's scratch.
    size_t symbol_count() const { return high_.size; }
    // Source concatenation must satisfy the same field-width limit as encode.
    void validate_delta(uint64_t delta) const { (void)delta_width(delta); }

    // Exact sizing shares delta validation and FSE with emission, but does not
    // construct low-bit data or retain an output block that will be discarded.
    size_t encoded_size(uint64_t ordinal, std::span<uint8_t const> old_chunk,
        uint16_t old_members, MemberRegions new_members)
    {
        return encode_impl(ordinal, old_chunk, old_members, new_members, true);
    }

    // Accumulate a coalesced block directly, avoiding a separate output vector
    // for every chunk.
    size_t append_encode(uint64_t ordinal, std::span<uint8_t const> old_chunk,
        uint16_t old_members, MemberRegions new_members)
    {
        return encode_impl(ordinal, old_chunk, old_members, new_members, false);
    }

    // Contributions retain their decoded high symbols and packed low fields.
    // Only a contribution's first delta changes at its new member boundary;
    // compressed ANS streams themselves cannot be concatenated.
    std::span<uint8_t const> join(uint64_t ordinal, std::span<EncodedChunk const> contributions,
                                MemberRegions new_members = {})
    {
        clear_output();
        append_join(ordinal, contributions, new_members);
        return output();
    }
    size_t joined_size(uint64_t ordinal, std::span<EncodedChunk const> contributions,
                       MemberRegions new_members = {})
    { return join_impl(ordinal, contributions, new_members, true); }
    size_t append_join(uint64_t ordinal, std::span<EncodedChunk const> contributions,
                       MemberRegions new_members = {})
    { return join_impl(ordinal, contributions, new_members, false); }

private:
    size_t join_impl(uint64_t ordinal, std::span<EncodedChunk const> contributions,
                     MemberRegions new_members, bool size_only)
    {
        CodecScheduler::Lease compute(scheduler_.get());
        check_ordinal(ordinal);
        group_tools::require(!contributions.empty(), "missing encoded group contributions");
        uint32_t members = 0;
        for (auto const& contribution : contributions)
        {
            group_tools::require(contribution.members && !contribution.bytes.empty() &&
                contribution.members <= UINT16_MAX - members, "invalid group contribution membership");
            members += contribution.members;
        }
        group_tools::require(new_members.size() <= UINT16_MAX - members, "invalid group membership");
        uint16_t const old_members = uint16_t(members);
        high_.size = 0; low_.size = 0;
        size_t bit_count = 0;
        uint64_t const range = range_per_chunk(), start = ordinal * range;
        uint64_t previous = start;
        members = 0;
        for (auto const& contribution : contributions)
        {
            size_t const first_symbol = high_.size;
            auto const decoded = decode(ordinal, contribution.bytes, contribution.members, nullptr, nullptr, true);
            group_tools::require(decoded.bytes == contribution.bytes.size(), "group contribution has trailing data");
            check_gsz_size(decoded.bytes, contribution.members);
            uint64_t const delta = decoded.first + uint64_t(members) * range - previous;
            size_t const width = delta_width(delta);
            group_tools::require(decoded.low_bits >= decoded.first_bits &&
                bit_count <= SIZE_MAX - width && decoded.low_bits - decoded.first_bits <= SIZE_MAX - bit_count - width,
                "group chunk length overflow");
            size_t const total_bits = bit_count + width + decoded.low_bits - decoded.first_bits;
            high_.data[first_symbol] = uint8_t((delta >> (k_ - 8)) % 178);
            if (!size_only)
            {
                grow_low(total_bits, bit_count);
                append_low_delta(delta, bit_count);
                append_low_bits(contribution.bytes.subspan(decoded.low_offset), decoded.first_bits,
                    decoded.low_bits - decoded.first_bits, bit_count);
            }
            else bit_count = total_bits;
            previous = decoded.last + uint64_t(members) * range;
            members += contribution.members;
        }
        size_t symbols = high_.size, total_bits = bit_count;
        (void)visit_deltas(k_, ordinal, old_members, new_members, previous, true,
            [&](uint64_t delta)
            {
                auto const width = delta_width(delta);
                group_tools::require(symbols < SIZE_MAX && total_bits <= SIZE_MAX - width,
                                     "group chunk length overflow");
                ++symbols; total_bits += width;
            });
        reserve(high_, symbols);
        if (!size_only) grow_low(total_bits, bit_count);
        previous = visit_deltas(k_, ordinal, old_members, new_members, previous, false,
            [&](uint64_t delta)
            {
                if (size_only)
                {
                    high_.data[high_.size++] = uint8_t((delta >> (k_ - 8)) % 178);
                    bit_count += delta_width(delta);
                }
                else append_delta(delta, bit_count);
            });
        group_tools::require(high_.size == symbols && bit_count == total_bits, "group codec count mismatch");
        return finish_encode(uint16_t(old_members + new_members.size()), bit_count, previous, size_only);
    }

    size_t encode_impl(uint64_t ordinal, std::span<uint8_t const> old_chunk,
        uint16_t old_members, MemberRegions new_members,
        bool size_only)
    {
        CodecScheduler::Lease compute(scheduler_.get());
        check_ordinal(ordinal);
        group_tools::require(new_members.size() <= size_t(UINT16_MAX - old_members) &&
            old_members + new_members.size() != 0, "invalid group membership");
        group_tools::require((old_members == 0) == old_chunk.empty(), "invalid old group chunk");
        uint16_t const members = uint16_t(old_members + new_members.size());
        uint64_t const range = range_per_chunk();
        uint64_t const start = ordinal * range;
        high_.size = 0;
        low_.size = 0;
        size_t bit_count = 0;
        uint64_t previous = start;
        if (old_members)
        {
            auto const old = decode(ordinal, old_chunk, old_members);
            group_tools::require(old.bytes == old_chunk.size(), "old chunk has trailing data");
            previous = old.last;
            bit_count = old.low_bits;
            if (!size_only)
            {
                resize(low_, bytes_for_bits(bit_count));
                std::memcpy(low_.data.get(), old_chunk.data() + old.low_offset, low_.size);
            }
        }

        size_t symbols = high_.size;
        size_t total_bits = bit_count;
        bool const bounded = compact_bounds(ordinal, old_members, new_members,
            previous, bit_count, size_only, symbols, total_bits);
        if (!bounded)
        {
            // Raw or borrowed input retains full validation. Also keep exact
            // sizing when a compact bound cannot fit, including duplicates
            // whose conservative symbol count can be much larger than output.
            (void)visit_deltas(k_, ordinal, old_members, new_members, previous, true,
                [&](uint64_t delta)
                {
                    auto const width = delta_width(delta);
                    group_tools::require(symbols < SIZE_MAX && total_bits <= SIZE_MAX - width,
                                         "group chunk length overflow");
                    ++symbols;
                    total_bits += width;
                });
        }
        reserve(high_, symbols);
        size_t low_size = bytes_for_bits(total_bits);
        if (!size_only)
        {
            resize(low_, low_size);
            if (bit_count & 7)
                low_.data[bit_count >> 3] &= uint8_t((1u << (bit_count & 7)) - 1);
            for (size_t i = bytes_for_bits(bit_count); i < low_.size; ++i) low_.data[i] = 0;
        }

        previous = visit_deltas(k_, ordinal, old_members, new_members, previous, false,
            [&](uint64_t delta)
            {
                if (size_only)
                {
                    high_.data[high_.size++] = uint8_t((delta % (178ull << (k_ - 8))) >> (k_ - 8));
                    bit_count += delta_width(delta);
                }
                else append_delta(delta, bit_count);
            });
        if (bounded)
        {
            group_tools::require(high_.size <= symbols && bit_count <= total_bits,
                                 "group codec bound mismatch");
            symbols = high_.size;
            low_size = bytes_for_bits(bit_count);
            if (!size_only) low_.size = low_size;
        }
        else group_tools::require(high_.size == symbols && bit_count == total_bits,
                                  "group codec count mismatch");
        return finish_encode(members, bit_count, previous, size_only);
    }

    size_t finish_encode(uint16_t members, size_t bit_count, uint64_t previous, bool size_only)
    {
        size_t const symbols = high_.size, low_size = bytes_for_bits(bit_count);
        group_tools::require(symbols <= SIZE_MAX - (symbols >> 7) - 1024,
                             "group ANS length overflow");
        resize(ans_, POS2_FSE_compressBound(symbols));
        size_t const ans_size = POS2_FSE_compress_usingCTable(ans_.data.get(), ans_.size,
            high_.data.get(), high_.size, ctable_.get());
        group_tools::require(ans_size != 0 && !POS2_FSE_isError(ans_size), "group ANS encode failed");
        uint8_t leb[10];
        size_t leb_size = 0;
        for (uint64_t n = ans_size;; n >>= 7)
        {
            leb[leb_size++] = uint8_t(n & 0x7f) | (n > 0x7f ? 0x80 : 0);
            if (n <= 0x7f) break;
        }
        group_tools::require(ans_size <= SIZE_MAX - leb_size &&
            low_size <= SIZE_MAX - leb_size - ans_size, "group chunk length overflow");
        size_t const encoded_size = leb_size + ans_size + low_size;
        summary_bits_ = bit_count; summary_last_ = previous;
        check_gsz_size(encoded_size, members);
        if (size_only) return encoded_size;
        group_tools::require(encoded_size <= SIZE_MAX - output_.size, "group output length overflow");
        reserve(output_, output_.size + encoded_size);
        append_output({leb, leb_size});
        append_output({ans_.data.get(), ans_size});
        append_output(low_.span());
        return encoded_size;
    }

    static constexpr uint64_t fixed_scratch_bytes()
    {
        // Sizes documented by upstream FSE_CTABLE_SIZE/FSE_DTABLE_SIZE macros.
        return sizeof(ChunkCodec) + (1 + (1 << 10) + 178 * 2) * sizeof(FSE_CTable)
            + (1 + (1 << 11)) * sizeof(FSE_DTable);
    }
    static size_t bytes_for_bits(size_t bits) { return bits / 8 + (bits % 8 != 0); }
    static size_t grown_capacity(size_t capacity, size_t required, uint64_t available)
    {
        if (capacity <= SIZE_MAX - capacity / 2)
            capacity = std::max(required, capacity + capacity / 2);
        else capacity = required;
        return capacity <= available ? capacity : required;
    }
    bool compact_bounds(uint64_t ordinal, uint16_t old_members, MemberRegions new_members,
        uint64_t previous, size_t old_bits, bool size_only, size_t& symbols, size_t& bits) const
    {
        if (new_members.empty()) return false;
        for (size_t i = 0; i < new_members.size(); ++i)
            if (!new_members[i].validated_deltas(k_)) return false;
        size_t incoming = 0;
        uint64_t const range = range_per_chunk(), start = ordinal * range;
        uint64_t last = previous;
        for (size_t i = 0; i < new_members.size(); ++i)
        {
            auto const fragments = new_members[i];
            group_tools::require(!fragments.empty(), "group member chunk is empty");
            group_tools::require(fragments.front() >= start && fragments.back() >= fragments.front() &&
                fragments.back() - start < range, "unsorted or out-of-range group fragment");
            uint64_t const offset = uint64_t(old_members + i) * range;
            (void)delta_width(fragments.front() + offset - last);
            last = fragments.back() + offset;
            if (fragments.size() > SIZE_MAX - incoming) return false;
            incoming += fragments.size();
        }
        // The total delta telescopes across regions. Its quotient bounds the
        // sum of unary quotients; counting duplicates only enlarges the bound.
        uint64_t const unary = (last - previous) / (178ull << (k_ - 8));
        if (incoming > SIZE_MAX - high_.size || unary > SIZE_MAX - old_bits ||
            incoming > (SIZE_MAX - old_bits - size_t(unary)) / (k_ - 7)) return false;
        size_t const bounded_symbols = high_.size + incoming;
        size_t const bounded_bits = old_bits + incoming * (k_ - 7) + size_t(unary);
        if (bounded_symbols > SIZE_MAX - (bounded_symbols >> 7) - 1024) return false;
        size_t const ans = POS2_FSE_compressBound(bounded_symbols);
        size_t const low = bytes_for_bits(bounded_bits);
        if (ans > SIZE_MAX - 10 || low > SIZE_MAX - 10 - ans ||
            output_.size > SIZE_MAX - 10 - ans - low) return false;
        // Match reserve's retained capacities and simultaneous replacement
        // peaks before committing to a conservative allocation. Otherwise the
        // exact pass can still accept a chunk close to the scratch limit.
        uint64_t used = scratch_bytes();
        auto fits = [&](Bytes const& buffer, size_t required)
        {
            if (required <= buffer.capacity) return true;
            if (used > budget_ || required > budget_ - used) return false;
            size_t const capacity = grown_capacity(buffer.capacity, required, budget_ - used);
            used += capacity - buffer.capacity;
            return true;
        };
        if (!fits(high_, bounded_symbols) || (!size_only && !fits(low_, low)) ||
            !fits(ans_, ans) || (!size_only && !fits(output_, output_.size + 10 + ans + low))) return false;
        symbols = bounded_symbols;
        bits = bounded_bits;
        return true;
    }
    uint64_t range_per_chunk() const { return 1ull << (k_ + PlotGroupFile::PROOFS_PER_CHUNK_BITS); }
    void check_ordinal(uint64_t ordinal) const
    {
        group_tools::require(ordinal < PlotGroupFile::getChunkCountForK(k_), "group chunk ordinal out of range");
    }
    void reserve(Bytes& buffer, size_t required)
    {
        if (required <= buffer.capacity) return;
        uint64_t const used = scratch_bytes();
        group_tools::require(used <= budget_ && required <= budget_ - used,
                             "group codec scratch budget exceeded");
        size_t const capacity = grown_capacity(buffer.capacity, required, budget_ - used);
        std::unique_ptr<uint8_t[]> replacement(new uint8_t[capacity]);
        if (buffer.size) std::memcpy(replacement.get(), buffer.data.get(), buffer.size);
        buffer.data = std::move(replacement);
        buffer.capacity = capacity;
    }
    void resize(Bytes& buffer, size_t size)
    {
        reserve(buffer, size);
        buffer.size = size;
    }
    template<class Visit>
    static uint64_t visit_deltas(uint8_t k, uint64_t ordinal, uint16_t old_members,
        MemberRegions members, uint64_t previous, bool validate, Visit&& visit)
    {
        uint64_t const range = 1ull << (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS), start = ordinal * range;
        for (size_t i = 0; i < members.size(); ++i)
        {
            auto const fragments = members[i];
            if (validate) group_tools::require(!fragments.empty(), "group member chunk is empty");
            uint64_t prior = 0; bool first = true;
            for (uint64_t fragment : fragments)
            {
                if (validate) group_tools::require(fragment >= start && fragment - start < range &&
                    (first || fragment >= prior), "unsorted or out-of-range group fragment");
                bool const duplicate = !first && fragment == prior;
                prior = fragment; first = false;
                if (duplicate) continue;
                uint64_t const value = fragment + uint64_t(old_members + i) * range;
                visit(value - previous); previous = value;
            }
        }
        return previous;
    }
    static uint32_t delta_width(uint8_t k, uint64_t delta)
    {
        uint64_t const quotient = delta / (178ull << (k - 8));
        group_tools::require(quotient + 1 + k - 8 <= 64, "unencodable group fragment delta");
        return uint32_t(quotient + 1 + k - 8);
    }
    uint32_t delta_width(uint64_t delta) const { return delta_width(k_, delta); }
    static uint64_t load_word(std::span<uint8_t const> bytes)
    {
        static_assert(std::endian::native == std::endian::little,
                      "group bit fields require a little-endian host");
        uint64_t word = 0;
        if (bytes.size() >= sizeof(word)) std::memcpy(&word, bytes.data(), sizeof(word));
        else std::memcpy(&word, bytes.data(), bytes.size());
        return word;
    }
    void append_delta(uint64_t delta, size_t& bit_count)
    {
        high_.data[high_.size++] = uint8_t((delta >> (k_ - 8)) % 178);
        append_low_delta(delta, bit_count);
    }
    void grow_low(size_t total_bits, size_t bit_count)
    {
        resize(low_, bytes_for_bits(total_bits));
        if (bit_count & 7) low_.data[bit_count >> 3] &= uint8_t((1u << (bit_count & 7)) - 1);
        for (size_t i = bytes_for_bits(bit_count); i < low_.size; ++i) low_.data[i] = 0;
    }
    void append_low_delta(uint64_t delta, size_t& bit_count)
    {
        uint64_t const threshold = 178ull << (k_ - 8);
        uint64_t const quotient = delta / threshold;
        uint64_t const remainder = delta % threshold;
        uint64_t value = (((1ull << quotient) - 1) << (k_ - 8))
            | (remainder & ((1ull << (k_ - 8)) - 1));
        uint32_t const width = delta_width(delta);
        append_low_word(value, width, bit_count);
    }
    void append_low_word(uint64_t value, uint32_t width, size_t& bit_count)
    {
        // Follow upstream BitWriter's word-sized field insertion, with bounded
        // memcpy for unaligned byte storage and the final short word.
        size_t const byte = bit_count >> 3;
        uint32_t const shift = uint32_t(bit_count & 7);
        auto const tail = low_.span().subspan(byte);
        uint64_t const word = load_word(tail) | (value << shift);
        std::memcpy(low_.data.get() + byte, &word, std::min(tail.size(), sizeof(word)));
        if (width > 64 - shift) low_.data[byte + 8] |= uint8_t(value >> (64 - shift));
        bit_count += width;
    }
    void append_low_bits(std::span<uint8_t const> source, size_t offset, size_t bits, size_t& bit_count)
    {
        while (bits)
        {
            auto const tail = source.subspan(offset >> 3);
            uint32_t const shift = uint32_t(offset & 7), width = uint32_t(std::min<size_t>(bits, 64));
            uint64_t value = load_word(tail) >> shift;
            if (shift && tail.size() > sizeof(uint64_t)) value |= uint64_t(tail[8]) << (64 - shift);
            if (width < 64) value &= (1ull << width) - 1;
            append_low_word(value, width, bit_count);
            offset += width; bits -= width;
        }
    }
    void append_output(std::span<uint8_t const> bytes)
    {
        std::memcpy(output_.data.get() + output_.size, bytes.data(), bytes.size());
        output_.size += bytes.size();
    }
    Scanned decode(uint64_t ordinal, std::span<uint8_t const> prefix, uint16_t members,
                   std::vector<uint64_t>* fragments = nullptr, std::vector<size_t>* boundaries = nullptr,
                   bool append_high = false, std::function<void(uint64_t)> const* visit = nullptr)
    {
        check_ordinal(ordinal);
        group_tools::require(members != 0, "empty group membership");
        uint64_t ans_size = 0;
        size_t leb_size = 0;
        for (unsigned shift = 0;; shift += 7)
        {
            group_tools::require(leb_size < prefix.size() && shift <= 63, "truncated group ANS size");
            uint8_t const byte = prefix[leb_size++];
            group_tools::require(shift != 63 || byte <= 1, "group ANS size overflow");
            ans_size |= uint64_t(byte & 0x7f) << shift;
            if (!(byte & 0x80))
            {
                group_tools::require(leb_size == 1 || byte != 0, "noncanonical group ANS size");
                break;
            }
        }
        group_tools::require(ans_size != 0 && ans_size < prefix.size() - leb_size,
                             "truncated group ANS body");
        size_t const low_offset = leb_size + size_t(ans_size);
        auto const low = prefix.subspan(low_offset);
        group_tools::require(ans_size <= (SIZE_MAX - 32) / 8 && low.size() <= SIZE_MAX / 8,
                             "group ANS length overflow");
        size_t const max_symbols = std::min(size_t(ans_size) * 8 + 32,
            low.size() * 8 / (k_ - 7));
        group_tools::require(max_symbols != 0, "group chunk has no deltas");
        size_t const begin = append_high ? high_.size : 0;
        group_tools::require(max_symbols <= SIZE_MAX - begin, "group ANS length overflow");
        resize(high_, begin + max_symbols);
        size_t const count = POS2_FSE_decompress_usingDTable(high_.data.get() + begin, max_symbols,
            prefix.data() + leb_size, size_t(ans_size), dtable_.get());
        group_tools::require(count != 0 && !POS2_FSE_isError(count), "group ANS decode failed");
        high_.size = begin + count;
        size_t bit_count = 0;
        auto peek = [&]()
        {
            group_tools::require(bit_count < low.size() * 8, "truncated group low bits");
            auto const tail = low.subspan(bit_count >> 3);
            uint32_t const shift = uint32_t(bit_count & 7);
            return std::pair{load_word(tail) >> shift,
                uint32_t(std::min(tail.size(), sizeof(uint64_t)) * 8) - shift};
        };
        uint64_t const range = range_per_chunk();
        uint64_t const start = ordinal * range;
        uint64_t const end = start + uint64_t(members) * range;
        uint64_t previous = start;
        uint64_t first = 0;
        size_t first_bits = 0;
        uint64_t region = 0;
        if (boundaries) boundaries->push_back(fragments->size());
        for (size_t i = 0; i < count; ++i)
        {
            uint8_t const symbol = high_.data[begin + i];
            group_tools::require(symbol < 178, "invalid group high symbol");
            auto [word, available] = peek();
            uint32_t const width = k_ - 8;
            group_tools::require(available >= width, "truncated group low bits");
            uint64_t const remainder = (uint64_t(symbol) << width) | (word & ((1ull << width) - 1));
            word >>= width;
            available -= width;
            bit_count += width;
            uint64_t quotient = 0;
            for (;;)
            {
                // As in upstream BitReader, scan a whole unary field at once.
                // Padding outside the input cannot act as its terminating zero.
                uint32_t const ones = std::min(uint32_t(std::countr_one(word)), available);
                quotient += ones;
                bit_count += ones;
                group_tools::require(quotient + 1 + k_ - 8 <= 64, "invalid group unary field");
                if (ones < available) { ++bit_count; break; }
                auto const next = peek();
                word = next.first;
                available = next.second;
            }
            uint64_t const delta = quotient * (178ull << (k_ - 8)) + remainder;
            group_tools::require(delta < end - previous && (i == 0 || delta != 0),
                                 "out-of-range or duplicate group delta");
            previous += delta;
            if (!i) { first = previous; first_bits = bit_count; }
            uint64_t const member = (previous - start) / range;
            group_tools::require(member == region || (i != 0 && member == region + 1),
                                 "group member chunk is empty");
            if (fragments)
            {
                if (member != region) boundaries->push_back(fragments->size());
                group_tools::require(fragments->size() < fragments->capacity(), "decoded group exceeds fragment workspace");
                fragments->push_back(previous - member * range);
            }
            if (visit) (*visit)(previous);
            region = member;
        }
        group_tools::require(region + 1 == members, "group member chunk is empty");
        size_t const low_size = bytes_for_bits(bit_count);
        if (bit_count & 7)
            group_tools::require((low[low_size - 1] >> (bit_count & 7)) == 0,
                                 "noncanonical group low padding");
        summary_bits_ = bit_count; summary_last_ = previous;
        return {low_offset + low_size, low_offset, bit_count, first, previous, first_bits};
    }

    uint8_t k_;
    uint64_t budget_;
    std::shared_ptr<CodecScheduler> scheduler_;
    std::unique_ptr<FSE_CTable, decltype(&POS2_FSE_freeCTable)> ctable_{nullptr, POS2_FSE_freeCTable};
    std::unique_ptr<FSE_DTable, decltype(&POS2_FSE_freeDTable)> dtable_{nullptr, POS2_FSE_freeDTable};
    Bytes high_, low_, ans_, output_, previous_output_;
    uint64_t summary_bits_ = 0, summary_last_ = 0;
};

struct EncodedBlock
{
    size_t begin = 0, end = 0;
    std::span<uint8_t const> bytes;
};

struct RemoteEncodedWindow
{
    // The remote provider owns a separately admitted memory partition. Keep
    // its reservation with the bytes until every borrowing writer has joined.
    std::shared_ptr<void> admission;
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> sizes;
    std::vector<ChunkSummary> summaries;
};
class RemoteEncoder
{
public:
    virtual ~RemoteEncoder() = default;
    virtual std::optional<RemoteEncodedWindow> encode(uint8_t k,
        std::span<ChunkInput const> chunks, bool summaries) = 0;
    virtual void observe_local(uint8_t k, std::span<ChunkInput const> chunks,
        std::span<uint64_t const> sizes, std::span<EncodedBlock const> blocks, uint64_t elapsed_ns) = 0;
};

// Each worker owns its tables and scratch. Previous coalesced bytes can be
// borrowed during the next encode; descriptors must be copied by the caller.
class ParallelEncoder
{
public:
    ParallelEncoder(uint8_t k, unsigned workers, uint64_t scratch_budget,
                    std::shared_ptr<CodecScheduler> scheduler = {},
                    std::shared_ptr<RemoteEncoder> remote = {})
        : k_(k), scheduler_(scheduler ? std::move(scheduler) : std::make_shared<CodecScheduler>(workers)),
          remote_(std::move(remote))
    {
        group_tools::require(workers != 0, "codec worker count must be positive");
        codecs_.reserve(workers); blocks_.reserve(workers);
        for (unsigned i = 0; i < workers; ++i)
            codecs_.push_back(std::make_unique<ChunkCodec>(k, scratch_budget / workers, scheduler_));
    }
    ~ParallelEncoder() = default;
    // Futures, packaged jobs, codec owners and block descriptors remain live
    // until the joined operation returns, even when the shared queue is full.
    static uint64_t memory_bytes(unsigned workers, bool private_scheduler = false)
    {
        uint64_t const tasks = uint64_t(workers) * (sizeof(ChunkCodec) + sizeof(std::unique_ptr<ChunkCodec>) +
            sizeof(EncodedBlock) + sizeof(std::future<void>) + 256);
        return (private_scheduler ? CodecScheduler::memory_bytes(workers) : 0) + sizeof(ParallelEncoder) + tasks;
    }
    ParallelEncoder(ParallelEncoder const&) = delete;
    ParallelEncoder& operator=(ParallelEncoder const&) = delete;

    unsigned workers() const { return static_cast<unsigned>(codecs_.size()); }
    // Observe only after a joined encode/measure/summarize operation. The
    // caller's thread is excluded so an owner timer can count it exactly once.
    CpuTime cpu_time() const
    { return {cpu_ns_.load(std::memory_order_relaxed), cpu_measured_.load(std::memory_order_relaxed)}; }
    uint64_t scratch_bytes() const
    {
        uint64_t result = 0;
        for (auto const& codec : codecs_) result += codec->scratch_bytes();
        return result;
    }

    // Block spans remain valid until the next encode. Tasks borrow input spans;
    // all tasks are drained before propagating any failure, as in the writer.
    std::span<EncodedBlock const> encode(std::span<ChunkInput const> chunks,
                                        std::span<uint64_t> sizes, bool retain_previous = false,
                                        std::span<ChunkSummary> summaries = {})
    {
        return run(chunks, sizes, retain_previous, false, summaries);
    }

    void measure(std::span<ChunkInput const> chunks, std::span<uint64_t> sizes,
                 std::span<ChunkSummary> summaries = {})
    {
        (void)run(chunks, sizes, false, true, summaries);
    }

    // Fresh summaries use the same persistent worker queue without touching
    // codec scratch or previously encoded output.
    void summarize(std::span<ChunkInput const> chunks, std::span<ChunkSummary> summaries)
    {
        group_tools::require(summaries.size() == chunks.size(), "codec summary output mismatch");
        (void)run(chunks, {}, false, false, summaries, true);
    }

private:
    std::span<EncodedBlock const> run(std::span<ChunkInput const> chunks,
                                    std::span<uint64_t> sizes, bool retain_previous, bool size_only,
                                    std::span<ChunkSummary> summaries = {}, bool summary_only = false)
    {
        std::lock_guard run(run_mutex_);
        group_tools::require(summary_only || sizes.size() == chunks.size(), "codec size output mismatch");
        group_tools::require(summaries.empty() || summaries.size() == chunks.size(), "codec summary output mismatch");
        if (!summary_only) blocks_.clear();
        if (chunks.empty()) return blocks_;
        if (!summary_only)
        {
            if (retain_previous) previous_remote_ = std::move(remote_window_);
            else previous_remote_.reset();
            remote_window_.reset();
        }
        if (remote_ && !summary_only && !size_only)
        {
            auto result = remote_->encode(k_, chunks, !summaries.empty());
            if (result)
            {
                group_tools::require(result->sizes.size() == chunks.size() &&
                    (summaries.empty() || result->summaries.size() == chunks.size()), "remote codec output mismatch");
                uint64_t bytes = 0;
                for (auto size : result->sizes)
                {
                    group_tools::require(size && size <= UINT64_MAX - bytes, "invalid remote codec chunk length");
                    bytes += size;
                }
                group_tools::require(bytes == result->bytes.size(), "remote codec payload length mismatch");
                std::copy(result->sizes.begin(), result->sizes.end(), sizes.begin());
                if (!summaries.empty()) std::copy(result->summaries.begin(), result->summaries.end(), summaries.begin());
                remote_window_ = std::move(result);
                blocks_.push_back({0, chunks.size(), remote_window_->bytes});
                // Remote wall time includes other hosts' work; it is not a
                // comparable local CPU observation for construction policy.
                cpu_measured_.store(false, std::memory_order_relaxed);
                return blocks_;
            }
        }
        std::chrono::steady_clock::time_point submitted;
        if (remote_ && !size_only && !summary_only)
            submitted = std::chrono::steady_clock::now();
        size_t const count = std::min(chunks.size(), codecs_.size());
        size_t const per_task = chunks.size() / count + (chunks.size() % count != 0);
        if (retain_previous) for (auto& codec : codecs_) codec->next_window();
        for (size_t task = 0, begin = 0; begin < chunks.size(); ++task, begin += per_task)
        {
            size_t const end = begin + std::min(per_task, chunks.size() - begin);
            uint64_t estimate = ChunkCodec::estimate_base();
            for (size_t i = begin; i < end; ++i)
            {
                if (summary_only) continue;
                uint64_t incoming = 0;
                for (size_t member_index = 0; member_index < chunks[i].new_members.size(); ++member_index)
                {
                    auto const member = chunks[i].new_members[member_index];
                    group_tools::require(member.size() <= UINT64_MAX - incoming,
                                         "group incoming count overflow");
                    incoming += member.size();
                }
                uint64_t source = chunks[i].old_chunk.size();
                group_tools::require(chunks[i].contributions.empty() || (!source && !chunks[i].old_members),
                                     "invalid old group contribution");
                for (auto const& contribution : chunks[i].contributions)
                {
                    group_tools::require(contribution.bytes.size() <= UINT64_MAX - source,
                                         "group contribution length overflow");
                    source += contribution.bytes.size();
                }
                uint64_t const chunk = ChunkCodec::estimate_work(k_, source, incoming)
                    - ChunkCodec::estimate_base();
                group_tools::require(chunk <= UINT64_MAX - estimate, "group window estimate overflow");
                estimate += chunk;
            }
            if (!summary_only) codecs_[task]->prepare(estimate);
            if (!summary_only) blocks_.push_back({begin, end, {}});
        }
        std::vector<std::future<void>> pending;
        pending.reserve(count);
        struct Drain
        {
            std::vector<std::future<void>>& pending;
            ~Drain() { for (auto& future : pending) if (future.valid()) future.wait(); }
        } drain{pending};
        for (size_t task = 0, begin = 0; begin < chunks.size(); ++task, begin += per_task)
        {
            size_t const end = begin + std::min(per_task, chunks.size() - begin);
            pending.push_back(scheduler_->submit([&, task, begin, end]
            {
                CpuTime cpu;
                struct Observe
                {
                    ParallelEncoder& owner; CpuTime& cpu;
                    ~Observe()
                    {
                        if (cpu.measured) owner.cpu_ns_.fetch_add(cpu.nanoseconds, std::memory_order_relaxed);
                        else owner.cpu_measured_.store(false, std::memory_order_relaxed);
                    }
                } observe{*this, cpu};
                ThreadCpuTimer timing(cpu);
                auto& codec = *codecs_[task];
                if (!summary_only) codec.clear_output();
                for (size_t i = begin; i < end; ++i)
                {
                    auto const& chunk = chunks[i];
                    if (summary_only)
                    {
                        group_tools::require(chunk.old_chunk.empty() && !chunk.old_members && chunk.contributions.empty(),
                                             "fresh summary has old group input");
                        summaries[i] = ChunkCodec::fresh_summary(k_, chunk.ordinal, chunk.new_members);
                        continue;
                    }
                    if (chunk.contributions.empty())
                        sizes[i] = size_only
                            ? codec.encoded_size(chunk.ordinal, chunk.old_chunk, chunk.old_members, chunk.new_members)
                            : codec.append_encode(chunk.ordinal, chunk.old_chunk, chunk.old_members, chunk.new_members);
                    else
                        sizes[i] = size_only ? codec.joined_size(chunk.ordinal, chunk.contributions, chunk.new_members)
                            : codec.append_join(chunk.ordinal, chunk.contributions, chunk.new_members);
                    size_t members = chunk.old_members + chunk.new_members.size();
                    for (auto const& contribution : chunk.contributions) members += contribution.members;
                    if (!summaries.empty()) summaries[i] = codec.summary(chunk.ordinal,
                        uint16_t(members));
                }
                if (!summary_only) blocks_[task].bytes = codec.output();
            }));
        }
        std::exception_ptr first;
        for (auto& future : pending)
        {
            try { future.get(); }
            catch (...) { if (!first) first = std::current_exception(); }
        }
        if (first) std::rethrow_exception(first);
        if (remote_ && !summary_only && !size_only)
        {
            // Compare the same bounded window, including CPU admission and
            // queue delay, so transport setup is amortized equally.
            remote_->observe_local(k_, chunks, sizes, blocks_, uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - submitted).count()));
        }
        return blocks_;
    }

    std::vector<std::unique_ptr<ChunkCodec>> codecs_;
    uint8_t k_;
    std::vector<EncodedBlock> blocks_;
    std::shared_ptr<CodecScheduler> scheduler_;
    std::shared_ptr<RemoteEncoder> remote_;
    std::optional<RemoteEncodedWindow> remote_window_, previous_remote_;
    std::mutex run_mutex_;
    std::atomic<uint64_t> cpu_ns_{0};
    std::atomic<bool> cpu_measured_{true};
};
} // namespace group_grow
