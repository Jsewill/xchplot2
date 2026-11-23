#include "host/ComputeWorker.hpp"
#include "host/ComputeIdentity.hpp"
#include "host/GroupHarvesterProfile.hpp"
#include "host/NumaTopology.hpp"
#include "host/GroupCodec.hpp"
#include "host/GroupWriter.hpp"
#include "host/ComputeThreads.hpp"
#include "host/GroupFarmResources.hpp"
#include "plot/Plotter.hpp"
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
#include "gpu/SyclDeviceList.hpp"
#include "host/GroupGpu.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#endif

#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <unordered_map>
#ifdef _WIN32
#include "host/WindowsFile.hpp"
#include <shlobj.h>
#else
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <filesystem>

namespace group_grow
{
namespace worker_detail
{
using group_tools::require;
using Clock = std::chrono::steady_clock;
using pool::GenerationOutput;
constexpr uint64_t mib = uint64_t(1) << 20;
constexpr size_t max_broker_routes = 8;
constexpr size_t initial_contribution_window_chunks = 64;
enum class Operation : uint8_t { Packed = 1, Contributions = 2, Encode = 3, Fresh = 4 };
pool::GenerationOutput generation_output(Operation operation)
{
    if (operation == Operation::Contributions) return pool::GenerationOutput::Contributions;
    if (operation == Operation::Fresh) return pool::GenerationOutput::Fresh;
    return pool::GenerationOutput::Packed;
}
enum class DataKind : uint8_t { Prelude = 1, Boundaries = 2, Words = 3, Contributions = 4, Encoded = 5, FreshManifest = 6 };
enum class WorkerError : uint8_t { Permanent = 0, Unsupported = 1, Transport = 2, Stale = 3 };
// ParallelEncoder already owns contiguous encoded chunks and their sizes until
// the next window. Local consumers borrow individual chunks from the same block.
using ContributionBlockSink = std::function<void(uint64_t, std::span<uint64_t const>,
    std::span<uint8_t const>, uint16_t)>;
ContributionBlockSink individual_contributions(WorkerContributionSink sink, WorkerCancellation cancel)
{
    if (!sink) return {};
    return [sink = std::move(sink), cancel](uint64_t first, std::span<uint64_t const> sizes,
        std::span<uint8_t const> bytes, uint16_t members)
    {
        for (auto size : sizes)
        {
            cancel.check(); sink(first++, bytes.first(size_t(size)), members);
            bytes = bytes.subspan(size_t(size));
        }
    };
}
void write_worker_status(pool::Writer& out, WorkerStatus const& value)
{
    out.string(value.identity); out.string(value.node_id); out.string(value.instance); out.u8(uint8_t(value.admission));
    out.string(value.compatibility); out.string(value.codec_compatibility); out.string(value.producer_build); out.u8(value.durable_control);
    for (auto const& resources : {value.capacity, value.active})
    { out.u32(resources.cpu_threads); out.u32(resources.slots); out.u64(resources.host_bytes); out.u64(resources.gpu_bytes); }
    out.u64(value.accepted_tasks); out.u64(value.waiting_cpu); out.u64(value.baseline_host_bytes);
    out.u64(value.broker_limit); out.u64(value.known_brokers); out.u64(value.healthy_brokers); out.u64(value.deferred_brokers);
    out.u64(value.broker_transport_bytes); out.u8(value.broker_saturated);
    pool::write_worker_hardware(out, value.hardware);
}
WorkerStatus read_worker_status(pool::Reader& in)
{
    WorkerStatus value; value.identity = in.string(64); value.node_id = in.string(64); value.instance = in.string(64); value.admission = pool::WorkerAdmission(in.u8());
    value.compatibility = in.string(256); value.codec_compatibility = in.string(256); value.producer_build = in.string(128);
    auto const durable = in.u8(); require(durable <= 1, "invalid worker durable control flag"); value.durable_control = durable;
    for (auto resources : {&value.capacity, &value.active})
    { resources->cpu_threads = in.u32(); resources->slots = in.u32(); resources->host_bytes = in.u64(); resources->gpu_bytes = in.u64(); }
    value.accepted_tasks = in.u64(); value.waiting_cpu = in.u64(); value.baseline_host_bytes = in.u64();
    value.broker_limit = in.u64(); value.known_brokers = in.u64(); value.healthy_brokers = in.u64(); value.deferred_brokers = in.u64();
    value.broker_transport_bytes = in.u64(); auto const saturated = in.u8(); require(saturated <= 1, "invalid worker saturation flag");
    value.broker_saturated = saturated; value.hardware = pool::read_worker_hardware(in); in.finish(); return value;
}
// Validate packed values in chunk order while retaining just the last word
// for the one field that may straddle two network frames.
struct PackedWordValidation
{
    uint64_t decoded = 0, previous = 0, last_word = 0;
    size_t decoded_chunk = 0;

    void append(uint8_t k, bool wide, std::span<uint32_t const> boundaries,
        uint64_t first_word, std::span<uint64_t const> words)
    {
        unsigned const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        uint64_t const mask = (uint64_t(1) << width) - 1;
        uint64_t const maximum_delta = uint64_t(72 - k) * (178ull << (k - 8));
        uint64_t const first_bit = first_word * 64;
        uint64_t const end = std::min<uint64_t>(boundaries.back(), (first_bit + words.size() * uint64_t(64)) / width);
        uint64_t bit = decoded * width;
        auto accept = [&](uint64_t value)
        {
            require(value >= previous && (wide || value - previous < maximum_delta),
                "compute member payload is unsorted or has an invalid delta summary");
            previous = value; ++decoded; bit += width;
        };
        while (decoded < end)
        {
            // Monotone boundaries end at total. Empty chunks consume no fields.
            while (decoded == boundaries[decoded_chunk + 1]) { ++decoded_chunk; previous = 0; }
            uint64_t const chunk_end = std::min<uint64_t>(end, boundaries[decoded_chunk + 1]);
            if (bit < first_bit)
            {
                unsigned const carry = unsigned(first_bit - bit);
                accept(((last_word >> (64 - carry)) | (words.front() << carry)) & mask);
            }
            while (decoded < chunk_end)
            {
                size_t const word = size_t((bit - first_bit) / 64);
                unsigned const shift = unsigned(bit % 64);
                uint64_t value = words[word] >> shift;
                if (shift + width > 64) value |= words[word + 1] << (64 - shift);
                accept(value & mask);
            }
        }
        last_word = words.back();
    }
    bool complete(uint8_t k, uint64_t total) const
    {
        unsigned const used = unsigned(total * (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS) % 64);
        return decoded == total && (!used || !(last_word >> used));
    }
};

class TaskCancelled : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
struct PackedRouteChanged : WorkerTaskDeclined {};
std::exception_ptr task_failure(std::exception_ptr original, std::exception_ptr renewal)
{
    if (renewal) try { std::rethrow_exception(original); }
    catch (TaskCancelled const&) { return renewal; }
    catch (...) {}
    return original;
}
WorkerError error_kind(std::exception const& error)
{
    if (dynamic_cast<pool::UnsupportedResources const*>(&error)) return WorkerError::Unsupported;
    if (dynamic_cast<pool::TransientTransport const*>(&error)) return WorkerError::Transport;
    if (dynamic_cast<pool::StaleLease const*>(&error)) return WorkerError::Stale;
    return WorkerError::Permanent;
}
[[noreturn]] void throw_worker_error(std::span<uint8_t const> payload)
{
    pool::Reader input(payload); auto code = WorkerError(input.u8()); auto message = input.string(); input.finish();
    if (code == WorkerError::Unsupported) throw pool::UnsupportedResources(message);
    if (code == WorkerError::Transport) throw pool::TransientTransport("compute worker: " + message);
    if (code == WorkerError::Stale) throw pool::StaleLease("compute worker: " + message);
    require(code == WorkerError::Permanent, "invalid compute worker error code");
    throw std::runtime_error("compute worker: " + message);
}

uint64_t add(uint64_t a, uint64_t b)
{ require(b <= UINT64_MAX - a, "worker memory estimate overflow"); return a + b; }
uint64_t multiply(uint64_t a, uint64_t b)
{ require(!a || b <= UINT64_MAX / a, "worker memory estimate overflow"); return a * b; }
uint64_t broker_registration_bytes()
{
    return add(add(pool::BrokerClient::memory_bytes(1), sizeof(std::string) + 254),
        multiply(max_broker_routes, sizeof(pool::Endpoint) + 2 * 254));
}
uint64_t service_overhead(WorkerConfig const& config, size_t brokers)
{
    return add(add(multiply(2 * config.capacity.slots + 2, pool::Channel::memory_bytes(config.tls.max_frame_bytes)),
        multiply(2 * config.capacity.slots, config.tls.max_frame_bytes)),
        add(add(mib, multiply(brokers, broker_registration_bytes())), CodecScheduler::memory_bytes(config.capacity.cpu_threads)));
}
std::chrono::milliseconds control_grace(pool::TlsOptions const& tls)
{ return std::chrono::milliseconds(std::min(100u, tls.timeout_ms)); }
pool::BrokerClient cleanup_control(pool::BrokerClient const& broker, pool::TlsOptions const& tls)
{
    // Task cancellation must still reach the broker after its compute/data
    // deadline. Cleanup owns a separate bounded control deadline and reuses
    // existing channels without replaying a possibly sent request.
    return broker.with_deadline(Clock::now() + control_grace(tls));
}
PlotProofParams parameters(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& id = {})
{
    require(k >= 18 && k <= 28 && !(k & 1), "worker requires even k in [18, 28]");
    return PlotProofParams::create_raw(PlotId(id), k, strength);
}
uint64_t fragment_count_bound(uint8_t k, uint8_t strength)
{
    auto const params = parameters(k, strength);
    return multiply(max_pairs_per_section_possible(params), params.get_num_sections());
}
uint64_t boundary_bytes(uint8_t k)
{ return multiply(add(PlotGroupFile::getChunkCountForK(k), 1), sizeof(uint32_t)); }

// Host memory and task slots are independent of CPU compute permits. A blocked
// owner holds its charged output, but never holds a CPU permit while sending it.
class ResourceSampler
{
    farm::Profile profile_;
    farm::LocalPressureSampler sampler_;
    std::mutex mutex_;
    std::optional<Clock::time_point> sampled_;
    pool::Resources available_{0, 0, 0, 0};
    uint64_t owned_host_bytes_ = 0, locked_output_bytes_ = 0;
    uint32_t active_cpu_ = 0;
public:
    ResourceSampler() : sampler_(profile_detail::host_name())
    {
        profile_.native.host = profile_detail::host_name();
        farm::Resource cpu; cpu.id = "cpu"; cpu.host = profile_.native.host;
        cpu.kind = farm::ResourceKind::cpu; cpu.capacity = profile_detail::logical_cpus(); cpu.capacity_known = cpu.capacity != 0;
        auto memory = cpu; memory.id = "ram"; memory.kind = farm::ResourceKind::host_memory;
        memory.capacity = profile_detail::physical_memory(); memory.capacity_known = memory.capacity != 0;
        profile_.resources = {std::move(cpu), std::move(memory)};
    }
    pool::Resources sample(uint32_t active_cpu)
    {
        std::lock_guard lock(mutex_); auto const now = Clock::now();
        if (sampled_ && active_cpu_ == active_cpu && now - *sampled_ < std::chrono::milliseconds(50)) return available_;
        available_ = {0, 0, 0, 0};
        auto const reading = sampler_.sample(profile_, now, active_cpu, nullptr, true);
        // GPU drivers can map the worker's own page-locked output as shared
        // memory. Either observation is a lower bound; they can overlap.
        owned_host_bytes_ = std::max(locked_output_bytes_, sampler_.own_anon_bytes().value_or(0));
        for (auto const& resource : reading.resources)
        {
            if (resource.resource == "ram" && resource.available_known) available_.host_bytes = resource.available_units;
            if (resource.resource == "cpu" && resource.cpu_busy_known)
                available_.cpu_threads = uint32_t(uint64_t(profile_detail::logical_cpus()) * (1000 - resource.cpu_busy_permille) / 1000);
        }
        sampled_ = now; active_cpu_ = active_cpu; return available_;
    }
    uint64_t owned_host_bytes()
    { std::lock_guard lock(mutex_); return owned_host_bytes_; }
    void locked_output(uint64_t bytes)
    {
        std::lock_guard lock(mutex_);
        if (locked_output_bytes_ == bytes) return;
        locked_output_bytes_ = bytes; owned_host_bytes_ = 0; sampled_.reset();
    }
};

class Budget
{
public:
    struct Lease
    {
        Budget* budget = nullptr;
        uint64_t bytes = 0;
        bool slot = false, retained = false;
        Lease() = default;
        Lease(Budget* owner, uint64_t value, bool task, bool persistent)
            : budget(owner), bytes(value), slot(task), retained(persistent) {}
        Lease(Lease const&) = delete;
        Lease& operator=(Lease const&) = delete;
        Lease(Lease&& other) noexcept : budget(std::exchange(other.budget, nullptr)),
            bytes(other.bytes), slot(other.slot), retained(other.retained) {}
        Lease& operator=(Lease&& other) noexcept
        { if (this != &other) { release(); budget = std::exchange(other.budget, nullptr); bytes = other.bytes; slot = other.slot; retained = other.retained; } return *this; }
        ~Lease() { release(); }
        void release()
        {
            if (!budget) return;
            { std::lock_guard lock(budget->mutex_); budget->used_ -= bytes;
              if (retained) budget->baseline_ -= bytes; if (slot) --budget->slots_; }
            budget->changed_.notify_all(); budget = nullptr;
        }
        void resize(uint64_t value)
        {
            require(budget, "worker memory lease is absent");
            { std::lock_guard lock(budget->mutex_);
              require(value <= bytes || value - bytes <= budget->capacity_ - budget->used_,
                      "worker task exceeds available offered host memory");
              if (value > bytes && value - bytes > budget->available_locked())
                  throw pool::TransientTransport("worker host headroom changed while growing a task allocation");
              budget->used_ = budget->used_ - bytes + value;
              if (retained) budget->baseline_ = budget->baseline_ - bytes + value;
              bytes = value; }
            budget->changed_.notify_all();
        }
        void transfer(Lease& target, uint64_t value)
        {
            require(budget && budget == target.budget && value <= bytes, "invalid worker retained memory transfer");
            { std::lock_guard lock(budget->mutex_);
              bytes -= value; target.bytes += value;
              if (retained != target.retained)
              { if (target.retained) budget->baseline_ += value; else budget->baseline_ -= value; } }
            budget->changed_.notify_all();
        }
    };
    // Zero host capacity provides the same cancellation-safe task slots for
    // client admission, without reserving compute or a fictitious RAM budget.
    Budget(uint64_t capacity, unsigned slots, uint64_t baseline, std::function<uint64_t()> available = {},
        std::function<uint64_t()> owned_resident = {})
        : capacity_(capacity), used_(baseline), baseline_(baseline), max_slots_(slots), available_(std::move(available)),
          owned_resident_(std::move(owned_resident))
    { require(slots && ((!capacity && !baseline) || baseline < capacity), "worker offer cannot hold bounded transport buffers"); }
    Lease acquire(uint64_t bytes, bool slot, WorkerCancellation const& cancel, bool retained = false)
    {
        std::unique_lock lock(mutex_);
        for (;;)
        {
            cancel.check();
            // Persistent allocations can grow while another request waits.
            // This check and retention updates share the allocation-state lock.
            if (bytes > capacity_ - baseline_)
                throw pool::UnsupportedResources("worker task exceeds host memory after retained allocations");
            if (bytes <= capacity_ - used_ && bytes <= available_locked() && (!slot || slots_ < max_slots_))
            { used_ += bytes; if (retained) baseline_ += bytes; if (slot) ++slots_; return Lease(this, bytes, slot, retained); }
            changed_.wait_for(lock, std::chrono::milliseconds(50));
        }
    }
    uint64_t task_capacity(uint64_t* standing = nullptr) const
    {
        std::lock_guard lock(mutex_);
        if (standing) *standing = baseline_;
        return std::min(capacity_ - baseline_, add(used_ - baseline_, available_locked()));
    }
    std::pair<uint64_t, unsigned> active() const
    { std::lock_guard lock(mutex_); return {used_ - baseline_, slots_}; }
    uint64_t baseline() const
    { std::lock_guard lock(mutex_); return baseline_; }
private:
    uint64_t capacity_, used_, baseline_;
    unsigned max_slots_, slots_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::function<uint64_t()> available_;
    std::function<uint64_t()> owned_resident_;
    uint64_t available_locked() const
    {
        if (!available_) return capacity_ - used_;
        auto const free = available_();
        auto const bytes = add(free, owned_resident_ ? std::min(used_, owned_resident_()) : 0);
        // Charge the complete outstanding model until release. Standalone
        // services can identify their own resident allocations. Embedded
        // owner pages cannot be attributed to this worker's leases.
        return std::min(capacity_ - used_, bytes - std::min(bytes, used_));
    }
};

// Embedded workers share the owner's scheduler, which may be wider than the
// worker offer. This independent weighted gate bounds all service CPU work.
class CpuAdmission
{
public:
    explicit CpuAdmission(unsigned capacity, std::function<unsigned(unsigned)> available = {})
        : capacity_(capacity), available_(std::move(available))
    { require(capacity, "worker CPU offer must be positive"); }
    class Lease
    {
    public:
        Lease(CpuAdmission& owner, unsigned permits, WorkerCancellation const& cancel, bool adaptive = false)
            : owner_(owner), permits_(permits)
        {
            require(permits && permits <= owner_.capacity_, "worker CPU request exceeds offered capacity");
            std::unique_lock lock(owner_.mutex_); owner_.waiting_.push_back(this);
            try
            {
                for (;;)
                {
                    cancel.check();
                    auto const available = owner_.available_ ? std::min(owner_.capacity_, owner_.available_(owner_.active_)) : owner_.capacity_;
                    auto const room = available - std::min(available, owner_.active_);
                    auto const grant = adaptive ? std::min(permits_, room) : permits_;
                    if (owner_.waiting_.front() == this && grant && grant <= room)
                    { owner_.waiting_.pop_front(); permits_ = grant; owner_.active_ += permits_; break; }
                    owner_.changed_.wait_for(lock, std::chrono::milliseconds(50));
                }
            }
            catch (...)
            {
                auto const item = std::find(owner_.waiting_.begin(), owner_.waiting_.end(), this);
                if (item != owner_.waiting_.end()) owner_.waiting_.erase(item);
                lock.unlock(); owner_.changed_.notify_all(); throw;
            }
            lock.unlock(); owner_.changed_.notify_all();
        }
        ~Lease()
        {
            { std::lock_guard lock(owner_.mutex_); owner_.active_ -= permits_; }
            owner_.changed_.notify_all();
        }
        Lease(Lease const&) = delete;
        Lease& operator=(Lease const&) = delete;
        unsigned permits() const { return permits_; }
    private:
        CpuAdmission& owner_;
        unsigned permits_;
    };
    std::pair<unsigned, size_t> active() const
    { std::lock_guard lock(mutex_); return {active_, waiting_.size()}; }
private:
    unsigned capacity_, active_ = 0;
    std::deque<Lease const*> waiting_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::function<unsigned(unsigned)> available_;
};

class CpuWork
{
public:
    CpuWork(CpuAdmission& offered, CodecScheduler* shared, WorkerCancellation const& cancel,
            unsigned permits = 1, bool adaptive = false)
        : offered_(offered, permits, cancel, adaptive), shared_(shared, offered_.permits(), [cancel] { cancel.check(); })
    { cancel.check(); }
    unsigned threads() const { return offered_.permits(); }
private:
    CpuAdmission::Lease offered_;
    CodecScheduler::Lease shared_;
};

struct OwnedChunk
{
    uint64_t ordinal;
    uint16_t old_members;
    std::span<uint8_t const> old;
    std::vector<WorkerContribution> contributions;
    std::vector<std::vector<uint64_t>> members;
};

struct CancelProgress final : IProgressSink
{
    WorkerCancellation cancel;
    explicit CancelProgress(WorkerCancellation value) : cancel(value) {}
    bool on_event(ProgressEvent const&) noexcept override
    {
        return (!cancel.stop || !cancel.stop->load(std::memory_order_acquire)) &&
            (!cancel.upstream_stop || !cancel.upstream_stop->load(std::memory_order_acquire)) &&
            (!cancel.service_stop || !cancel.service_stop->load(std::memory_order_acquire)) &&
            Clock::now() < cancel.deadline;
    }
};

std::vector<OwnedChunk> read_chunks(pool::Reader& in, unsigned maximum, uint64_t max_bytes,
                                  uint32_t expected_count = 0)
{
    auto const count = in.u32();
    require(count && count <= maximum, "worker chunk window exceeds configured limit");
    require(!expected_count || count == expected_count, "worker chunk count differs from lease");
    require(count <= in.remaining() / pool::codec_chunk_header_bytes, "truncated worker chunk headers");
    std::vector<OwnedChunk> chunks; chunks.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        OwnedChunk chunk; chunk.ordinal = in.u64(); chunk.old_members = in.u16();
        chunk.old = in.bytes(uint32_t(std::min<uint64_t>(max_bytes, UINT32_MAX)));
        auto const contributions = in.u16();
        require(contributions <= in.remaining() / (sizeof(uint16_t) + sizeof(uint32_t)),
            "truncated worker contribution headers");
        chunk.contributions.reserve(contributions);
        for (unsigned j = 0; j < contributions; ++j)
        {
            auto const members = in.u16(); auto const bytes = in.bytes(uint32_t(std::min<uint64_t>(max_bytes, UINT32_MAX)));
            chunk.contributions.push_back({bytes, members});
        }
        auto const members = in.u16();
        require(members <= in.remaining() / sizeof(uint32_t), "truncated worker member headers");
        chunk.members.reserve(members);
        for (unsigned j = 0; j < members; ++j)
        {
            auto const size = in.u32();
            require(uint64_t(size) <= in.remaining() / sizeof(uint64_t), "truncated worker raw contribution");
            std::vector<uint64_t> values; values.reserve(size);
            for (uint32_t n = 0; n < size; ++n) values.push_back(in.u64());
            chunk.members.push_back(std::move(values));
        }
        chunks.push_back(std::move(chunk));
    }
    in.finish(); return chunks;
}

uint64_t chunk_work(uint8_t k, OwnedChunk const& chunk)
{
    uint64_t old = chunk.old.size(), incoming = 0;
    for (auto const& value : chunk.contributions) old = add(old, value.bytes.size());
    for (auto const& value : chunk.members) incoming = add(incoming, value.size());
    return ChunkCodec::estimate_work(k, old, incoming);
}

void send_blob(pool::Channel& channel, pool::Message type, std::span<uint8_t const> bytes,
               uint32_t frame_bytes, WorkerCancellation const& cancel)
{
    while (!bytes.empty())
    {
        cancel.check(); auto const part = bytes.first(std::min<size_t>(bytes.size(), frame_bytes));
        channel.send(type, part); bytes = bytes.subspan(part.size());
    }
}
class Renewal
{
public:
    Renewal(pool::BrokerClient& broker, pool::Lease const& lease, std::atomic<bool>& cancelled,
        std::exception_ptr& failure, std::function<void()> keepalive, unsigned interval)
        : broker_(broker), lease_(lease), cancelled_(cancelled), failure_(failure), keepalive_(std::move(keepalive)),
          interval_(std::max(1u, std::min(interval, lease.ttl_ms) / 3)), thread_([this] { run(); }) {}
    ~Renewal()
    { { std::lock_guard lock(mutex_); done_ = true; } ready_.notify_all(); if (thread_.joinable()) thread_.join(); }
private:
    void run()
    {
        std::unique_lock lock(mutex_);
        while (!ready_.wait_for(lock, std::chrono::milliseconds(interval_), [&] { return done_; }))
        {
            lock.unlock();
            try { broker_.renew(lease_); keepalive_(); }
            // The caller reads this only after destruction joins this thread.
            catch (...) { failure_ = std::current_exception(); cancelled_.store(true); return; }
            lock.lock();
        }
    }
    pool::BrokerClient& broker_;
    pool::Lease const& lease_;
    std::atomic<bool>& cancelled_;
    std::exception_ptr& failure_;
    std::function<void()> keepalive_;
    unsigned interval_;
    std::mutex mutex_;
    std::condition_variable ready_;
    bool done_ = false;
    std::thread thread_;
};
} // namespace worker_detail
using namespace worker_detail;

void WorkerCancellation::check() const
{
    if ((stop && stop->load(std::memory_order_acquire)) ||
        (upstream_stop && upstream_stop->load(std::memory_order_acquire))) throw TaskCancelled("worker task cancelled");
    if (service_stop && service_stop->load(std::memory_order_acquire)) throw TaskCancelled("worker service stopped");
    if (Clock::now() >= deadline) throw TaskCancelled("worker task deadline expired");
}
uint64_t worker_service_baseline_bytes(WorkerConfig const& config)
{
    worker_detail::require(!config.cpu_scheduler || config.capacity.cpu_threads <= config.cpu_scheduler->workers(),
        "worker offer exceeds shared CPU gate");
    auto const brokers = config.discover_brokers ? config.max_brokers :
        std::max<size_t>(1, config.brokers.size() + bool(config.broker.port));
    auto result = worker_detail::service_overhead(config, brokers);
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    if (config.backend != "cpu" && config.capacity.gpu_bytes)
        result = worker_detail::add(result, GpuProducer::runtime_host_allowance_bytes());
#endif
    return result;
}
uint64_t worker_cpu_generation_bytes(uint8_t k, uint8_t strength)
{
    auto const params = parameters(k, strength);
    uint64_t const section_pairs = max_pairs_per_section_possible(params);
    uint64_t const pairs = multiply(section_pairs, params.get_num_sections());
    uint64_t const element = std::max({sizeof(WorkerReferenceXsCandidate), sizeof(T1Pairing), sizeof(T2Pairing), sizeof(T3Pairing)});
    uint64_t const block = WorkerReferencePlotLayout::align_up(multiply(section_pairs, element) / 4, 64);
    uint64_t const layout = add(multiply(block, 32), 2 * mib);
#if defined(GROW_MOVE_PLOTTER_OUTPUT) && GROW_MOVE_PLOTTER_OUTPUT
    constexpr uint64_t copies = 1;
#else
    constexpr uint64_t copies = 2;
#endif
    return add(add(layout, multiply(pairs, sizeof(uint64_t) * copies)), 128 * mib);
}
uint64_t worker_cpu_task_bytes(uint8_t k, uint8_t strength, uint64_t members, uint64_t transport_bytes)
{
    using namespace worker_detail;
    require(members && members <= pool::max_generation_batch_members, "invalid CPU worker task membership");
    return add(add(add(worker_cpu_generation_bytes(k, strength),
        multiply(multiply(fragment_count_bound(k, strength), sizeof(uint64_t)), members - 1)),
        multiply(boundary_bytes(k), members)), transport_bytes);
}

struct WorkerIdentityReservation
{
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    ~WorkerIdentityReservation() { if (file != INVALID_HANDLE_VALUE) ::CloseHandle(file); }
#else
    int fd = -1;
    ~WorkerIdentityReservation() { if (fd >= 0) ::close(fd); }
#endif
    std::string identity;
};
std::shared_ptr<WorkerIdentityReservation> WorkerService::reserve_identity(std::string const& certificate)
{
    auto result = std::make_shared<WorkerIdentityReservation>();
    result->identity = pool::certificate_identity(certificate);
    auto const node = pool::certificate_node_id(certificate);
    auto const name = "xchplot2-compute-" + (node.empty() ? result->identity : node) + ".lock";
#ifdef _WIN32
    PWSTR local_app_data = nullptr;
    require(SUCCEEDED(::SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &local_app_data)),
        "cannot locate compute identity lock directory");
    struct FreePath { PWSTR value; ~FreePath() { ::CoTaskMemFree(value); } } cleanup{local_app_data};
    auto const path = std::filesystem::path(local_app_data) / name;
    result->file = pos2gpu::create_private_file_handle(path, GENERIC_READ | GENERIC_WRITE, OPEN_ALWAYS, 0,
        FILE_SHARE_READ | FILE_SHARE_WRITE);
    OVERLAPPED range{};
    if (!::LockFileEx(result->file, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &range))
    {
        if (::GetLastError() == ERROR_LOCK_VIOLATION) throw WorkerAlreadyRunning();
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "cannot reserve compute identity");
    }
#else
    auto const path = std::filesystem::path("/tmp") / name;
    result->fd = ::open(path.c_str(), O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_RDWR, 0600);
    require(result->fd >= 0, "cannot open compute identity lock");
    if (::flock(result->fd, LOCK_EX | LOCK_NB))
    {
        if (errno == EWOULDBLOCK || errno == EAGAIN) throw WorkerAlreadyRunning();
        throw std::runtime_error("cannot reserve compute identity");
    }
#endif
    return result;
}
void default_worker_resources(WorkerConfig& config)
{
    if (!config.capacity.cpu_threads)
        config.capacity.cpu_threads = std::max(1u, profile_detail::logical_cpus());
    if (!config.capacity.host_bytes)
    {
        config.capacity.host_bytes = profile_detail::physical_memory();
        require(config.capacity.host_bytes, "cannot determine worker RAM; set --host-bytes");
    }
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    if (config.backend != "cpu" && !config.capacity.gpu_bytes)
    {
        config.capacity.gpu_bytes = pos2gpu::worker_gpu_available_bytes(config.device);
        require(config.backend != "gpu" || config.capacity.gpu_bytes,
            "cannot determine worker GPU memory; set --gpu-bytes");
    }
#endif
}

void fit_worker_broker_budget(WorkerConfig& config)
{
    default_worker_resources(config);
    if (!config.discover_brokers) return;
    require(config.max_brokers, "worker broker limit must be positive");
    auto const minimum_routes = std::max<size_t>(1, config.brokers.size() + bool(config.broker.port));
    auto const task = add(add(worker_cpu_generation_bytes(18, 2), boundary_bytes(18)), config.max_task_bytes);
    while (config.max_brokers > minimum_routes &&
        add(service_overhead(config, config.max_brokers), task) > config.capacity.host_bytes) --config.max_brokers;
    require(add(service_overhead(config, config.max_brokers), task) <= config.capacity.host_bytes,
        "worker host partition cannot hold transport and a supported generation task");
}

struct WorkerService::Impl
{
    WorkerConfig config;
    struct Registration
    {
        pool::Endpoint endpoint;
        std::vector<pool::Endpoint> routes;
        size_t route_cursor = 0;
        pool::BrokerClient client;
        std::string epoch;
        std::string identity, node_id;
        pool::Endpoint advertised_endpoint;
        std::string alternate_source;
        bool healthy = false;
        bool retired = false;
        Clock::time_point seen = Clock::now();
        Clock::time_point admitted = Clock::now(), rotation_attempt{};
        uint64_t revision = 0;
        Registration(pool::Endpoint address, pool::TlsOptions const& tls)
            : endpoint(std::move(address)), routes{endpoint}, client(endpoint, tls, 1) {}
    };
    mutable std::mutex brokers_mutex;
    std::vector<std::shared_ptr<Registration>> brokers;
    size_t refresh_cursor = 0, discovery_cursor = 0, deferred_brokers = 0;
    uint64_t broker_transport_bytes = 0;
    mutable ResourceSampler resource_sampler;
    CpuAdmission offered_cpu;
    std::shared_ptr<CodecScheduler> scheduler;
    std::unique_ptr<Budget> budget;
    std::unique_ptr<pool::Listener> listener;
    std::atomic<bool> stopping{false};
    std::vector<std::thread> accept_threads;
    std::thread heartbeat_thread, gpu_thread;
    std::vector<std::thread> dispatch_threads;
    struct PendingWork { pool::Channel channel; pool::Frame frame; };
    std::deque<PendingWork> pending_work;
    std::mutex dispatch_mutex;
    std::condition_variable dispatch_ready;
    mutable std::mutex admission_mutex;
    pool::WorkerAdmission admission = pool::WorkerAdmission::Accepting;
    std::optional<pool::PoolIdentity> control_identity;
    uint64_t accepted_tasks = 0;
    std::atomic<bool> gpu_active{false};
    std::mutex jobs_mutex, heartbeat_mutex;
    std::condition_variable jobs_ready, heartbeat_ready;
    std::deque<std::packaged_task<void()>> gpu_jobs;
    std::string instance = pool::random_token();
    mutable std::mutex advertisement_mutex;
    pool::CapacityUpdate advertised_capacity;
    uint64_t advertised_baseline = 0;
    std::atomic<uint64_t> retained_gpu_host{0};
    uint64_t service_baseline = 0;
    pool::WorkerHardware physical;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    std::unordered_map<unsigned, GpuProducerPreflight> gpu_profiles;
#endif
    bool started = false, gpu_enabled = false;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    std::unique_ptr<GpuProducer> gpu;
    uint8_t gpu_k = 0, gpu_strength = 0;
    unsigned gpu_threads = 0;
    Budget::Lease gpu_retained;
    std::atomic<uint64_t> idle_generation_bytes{0};
#endif
    bool check_idle_output = false; // jobs_mutex; consumed on the GPU thread

    explicit Impl(WorkerConfig options) : config(std::move(options)), offered_cpu(config.capacity.cpu_threads,
        [this](unsigned active) { return available_resources(active).cpu_threads; })
    {
        require(config.capacity.cpu_threads && config.capacity.cpu_threads <= 1024 &&
            config.capacity.slots && config.capacity.slots <= 1024,
            "worker requires bounded offered CPUs and slots");
        if (!config.generation_threads)
            config.generation_threads = std::max(1u, config.capacity.cpu_threads / config.capacity.slots);
        require(config.generation_threads && config.generation_threads <= config.capacity.cpu_threads,
            "worker generation threads exceed offered CPUs");
        require(config.max_batch_members && config.max_batch_members <= pool::max_generation_batch_members &&
            config.max_window_chunks && config.max_task_bytes && config.max_task_bytes <= UINT32_MAX &&
            config.task_timeout_ms && config.tls.timeout_ms,
            "invalid worker batch or codec window limit");
        require(config.tls.max_frame_bytes >= 4096 && config.tls.max_frame_bytes <= config.max_task_bytes,
            "invalid worker transport frame limit");
        require(config.backend == "cpu" || config.backend == "gpu" || config.backend == "auto", "invalid worker backend");
        require(config.max_brokers && config.max_brokers <= 4096 && config.broker_refresh_parallelism &&
            config.broker_refresh_parallelism <= 64 && config.broker_refresh_ms && config.broker_refresh_ms <= 1000,
            "invalid worker broker registration limits");
        require(config.broker_rotation_ms >= 100 && config.broker_rotation_ms <= 3600000, "invalid broker rotation interval");
        require(config.brokers.size() + bool(config.broker.port) <= config.max_brokers,
            "configured brokers exceed worker registration budget");
        if (!config.tls.membership_directory.empty())
        {
            control_identity = pool::load_pool(std::filesystem::path(reinterpret_cast<char8_t const*>(config.tls.membership_directory.c_str())));
            require(control_identity && control_identity->node_id == pool::certificate_node_id(config.tls.cert_file) &&
                control_identity->pool_id == pool::certificate_identity(config.tls.ca_file), "worker control enrollment differs from its TLS identity");
            auto const intent = pool::load_worker_intent(*control_identity);
            admission = intent == pool::WorkerIntent::Paused ? pool::WorkerAdmission::Paused :
                intent == pool::WorkerIntent::Draining ? pool::WorkerAdmission::Drained : pool::WorkerAdmission::Accepting;
        }
        if (config.admit_start) config.admit_start(config);
        scheduler = config.cpu_scheduler ? config.cpu_scheduler : std::make_shared<CodecScheduler>(config.capacity.cpu_threads);
        require(config.capacity.cpu_threads <= scheduler->workers(), "worker offer exceeds shared CPU gate");
        auto const broker_slots = config.discover_brokers ? config.max_brokers :
            std::max<size_t>(1, config.brokers.size() + bool(config.broker.port));
        broker_transport_bytes = multiply(broker_slots, broker_registration_bytes());
        uint64_t const baseline = service_overhead(config, broker_slots);
        if (baseline >= config.capacity.host_bytes)
            throw std::runtime_error("worker broker/transport budget requires more than " + std::to_string(baseline) +
                " host bytes before task memory; reduce --max-brokers or increase --host-bytes");
        budget = std::make_unique<Budget>(config.capacity.host_bytes, config.capacity.slots, baseline,
            [this] { return available_resources(offered_cpu.active().first).host_bytes; },
            [this] { return !config.available_resources && !config.cpu_scheduler && !config.admit_start && !config.host_reservation ?
                resource_sampler.owned_host_bytes() : 0; });
        service_baseline = baseline;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (config.backend != "cpu")
        {
            try { gpu_enabled = config.capacity.gpu_bytes && config.device >= 0 && config.device < pos2gpu::gpu_device_count(); }
            catch (...) { if (config.backend == "gpu") throw; gpu_enabled = false; }
            require(config.backend != "gpu" || gpu_enabled, "offered GPU is unavailable");
        }
#else
        require(config.backend != "gpu", "GPU compute is unavailable in this build");
#endif
        if (gpu_enabled)
        {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            WorkerCancellation immediate; immediate.deadline = Clock::now() + std::chrono::milliseconds(1);
            gpu_retained = budget->acquire(GpuProducer::runtime_host_allowance_bytes(), false, immediate, true);
            retained_gpu_host.store(gpu_retained.bytes);
#endif
        }
        physical.host = profile_detail::host_name(); physical.logical_cpus = profile_detail::logical_cpus();
        physical.total_host_bytes = profile_detail::physical_memory();
        physical.partition = config.capacity;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (gpu_enabled)
        {
            for (auto const& device : pos2gpu::list_gpu_devices()) if (device.id == size_t(config.device))
            { physical.gpu_identity = device.physical_identity; physical.total_gpu_bytes = device.vram_bytes; break; }
        }
#endif
        if (gpu_enabled) gpu_thread = std::thread([this] { gpu_loop(); });
        try
        {
            if (gpu_enabled) on_gpu([this]
            {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
                for (uint8_t k = 18; k <= 28; k += 2)
                {
                    GpuProducerOptions options; options.device_id = config.device;
                    options.tier = config.tier; options.max_vram_bytes = config.capacity.gpu_bytes;
                    options.host_threads = config.generation_threads;
                    options.member_overlap = options.compact_transfer = options.preserve_match_scratch = true;
                    try { gpu_profiles.emplace(k, GpuProducer::preflight(k, 2, options)); }
                    catch (pos2gpu::InsufficientVramError const&) {}
                }
                require(config.backend != "gpu" || !gpu_profiles.empty(), "GPU offer cannot fit any supported generation tier");
                if (gpu_profiles.empty()) gpu_enabled = false;
#endif
            });
            auto const capacity = current_capacity(&advertised_baseline);
            advertised_capacity = {capacity, 1, profiles(capacity, advertised_baseline)};
        }
        catch (...)
        { stopping.store(true); jobs_ready.notify_all(); if (gpu_thread.joinable()) gpu_thread.join(); throw; }
    }
    ~Impl() { stop(); }

    struct AdmissionRefused : pool::TransientTransport
    { AdmissionRefused() : pool::TransientTransport("worker is not accepting new work") {} };
    class Acceptance
    {
    public:
        explicit Acceptance(Impl& owner) : owner_(owner)
        {
            std::lock_guard lock(owner_.admission_mutex);
            if (owner_.admission != pool::WorkerAdmission::Accepting || owner_.stopping.load())
                throw AdmissionRefused();
            ++owner_.accepted_tasks;
        }
        ~Acceptance()
        {
            std::lock_guard lock(owner_.admission_mutex);
            --owner_.accepted_tasks;
            if (!owner_.accepted_tasks && owner_.admission == pool::WorkerAdmission::Draining)
            { owner_.admission = pool::WorkerAdmission::Drained; owner_.heartbeat_ready.notify_all(); }
        }
        Acceptance(Acceptance const&) = delete;
        Acceptance& operator=(Acceptance const&) = delete;
    private:
        Impl& owner_;
    };
    pool::WorkerHardware hardware() const
    {
        auto value = physical; value.available_host_bytes = profile_detail::available_memory();
        value.standing_host_bytes = budget->baseline();
        auto const now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (now > 0) value.observed_at_unix_ms = uint64_t(now);
        return value;
    }
    WorkerStatus status() const
    {
        WorkerStatus value; value.identity = config.tls.cert_file.empty() ? "" : pool::certificate_identity(config.tls.cert_file); value.instance = instance;
        value.compatibility = worker_compatibility(); value.codec_compatibility = worker_compatibility(pool::Kind::Pack);
        value.producer_build = worker_producer_build(); value.durable_control = control_identity.has_value();
        if (!config.tls.cert_file.empty()) value.node_id = pool::certificate_node_id(config.tls.cert_file);
        value.capacity = current_capacity();
        auto const memory = budget->active(); auto const cpu = offered_cpu.active();
        value.active = {cpu.first, memory.second, memory.first, gpu_active.load() ? config.capacity.gpu_bytes : 0};
        value.waiting_cpu = cpu.second; value.baseline_host_bytes = budget->baseline();
        { std::lock_guard lock(admission_mutex); value.admission = admission; value.accepted_tasks = accepted_tasks; }
        { std::lock_guard lock(brokers_mutex); value.known_brokers = brokers.size(); value.deferred_brokers = deferred_brokers;
          for (auto const& broker : brokers) value.healthy_brokers += broker->healthy; }
        value.broker_limit = config.max_brokers; value.broker_transport_bytes = broker_transport_bytes;
        value.broker_saturated = value.deferred_brokers != 0; value.hardware = hardware(); return value;
    }
    WorkerStatus control(WorkerControl operation)
    {
        {
            std::lock_guard lock(admission_mutex);
            require(operation >= WorkerControl::Pause && operation <= WorkerControl::Resume, "invalid worker control operation");
            // Complete the private atomic save before changing live admission
            // or acknowledging control. A failed save leaves accepted work and
            // admission untouched. PoolLock never acquires admission_mutex.
            if (control_identity) pool::save_worker_intent(*control_identity, operation == WorkerControl::Pause ? pool::WorkerIntent::Paused :
                operation == WorkerControl::Drain ? pool::WorkerIntent::Draining : pool::WorkerIntent::Active);
            switch (operation)
            {
            case WorkerControl::Pause: admission = pool::WorkerAdmission::Paused; break;
            case WorkerControl::Drain: admission = accepted_tasks ? pool::WorkerAdmission::Draining : pool::WorkerAdmission::Drained; break;
            case WorkerControl::Resume: admission = pool::WorkerAdmission::Accepting; break;
            default: throw std::runtime_error("invalid worker control operation");
            }
        }
        heartbeat_ready.notify_all(); return status();
    }

    pool::Resources available_resources(unsigned active_cpu) const
    {
        auto available = config.available_resources ? config.available_resources(active_cpu) : resource_sampler.sample(active_cpu);
        available.cpu_threads = std::min(available.cpu_threads, uint32_t(profile_detail::logical_cpus()));
        return available;
    }
    pool::Resources current_capacity(uint64_t* standing = nullptr) const
    {
        auto capacity = config.capacity;
        if (!gpu_enabled) capacity.gpu_bytes = 0;
        auto const active = offered_cpu.active().first;
        auto const available = available_resources(active);
        capacity.cpu_threads = std::min(capacity.cpu_threads, std::max(active, std::max(uint32_t(1), available.cpu_threads)));
        // A positive unavailable partition carries no useful generation
        // profile; existing active leases remain represented until release.
        capacity.host_bytes = std::max(uint64_t(1), budget->task_capacity(standing));
        return capacity;
    }
    pool::CapacityUpdate capacity_update(std::optional<uint64_t> registered_revision = {}, bool force = false)
    {
        std::lock_guard lock(advertisement_mutex);
        uint64_t standing;
        auto const capacity = current_capacity(&standing);
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        // Headroom can fall after a successful task, while the worker is
        // idle and no new lease can reach its output-release path.
        if (capacity.host_bytes < idle_generation_bytes.load())
        {
            { std::lock_guard lock(jobs_mutex); check_idle_output = true; }
            jobs_ready.notify_one();
        }
#endif
        if (force || capacity != advertised_capacity.capacity || standing != advertised_baseline)
        {
            require(advertised_capacity.revision < UINT64_MAX, "worker capacity revision exhausted");
            advertised_capacity = {capacity, advertised_capacity.revision + 1, profiles(capacity, standing)};
            advertised_baseline = standing;
        }
        return advertised_capacity.for_broker(registered_revision.value_or(advertised_capacity.revision));
    }
    unsigned generation_slots(pool::Task const& task) const
    {
        std::lock_guard lock(advertisement_mutex);
        auto demand = task.resources;
        // Rate estimates are per member. Report the pool's singleton lane
        // capacity even when this lease also held a batch's extra output.
        if (auto const& profiles = advertised_capacity.generation_profiles)
        {
            auto const at = std::find_if(profiles->begin(), profiles->end(), [&](auto const& profile)
            { return profile.kind == task.kind && profile.k == task.k && profile.strength == task.strength; });
            if (at != profiles->end())
            { demand = at->resources; demand.cpu_threads = task.resources.cpu_threads; }
        }
        return advertised_capacity.capacity.parallel_tasks(demand);
    }
    pool::BrokerClient control_for(pool::Lease const& lease, uint64_t* registered_revision = nullptr) const
    {
        std::lock_guard lock(brokers_mutex);
        std::shared_ptr<Registration> selected;
        for (auto const& broker : brokers)
            if (!broker->epoch.empty() && broker->epoch == lease.epoch && broker->identity == lease.broker_identity &&
                (!selected || (broker->healthy && !selected->healthy) ||
                    (broker->healthy == selected->healthy && broker->seen > selected->seen))) selected = broker;
        if (selected)
        {
            if (registered_revision) *registered_revision = selected->revision;
            return selected->client;
        }
        throw pool::StaleLease("worker does not recognize the lease's broker epoch");
    }
    void report_refusal(pool::Lease const& lease, std::string const& owner, Clock::time_point deadline) const
    {
        if (lease.worker_instance != instance || lease.owner_identity != owner) return;
        try { cleanup_control(control_for(lease), config.tls).with_deadline(deadline).reject(lease, pool::FailureCode::RefusedBeforeBegin); }
        catch (...) { /* Older brokers and failed control routes still use the direct refusal. */ }
    }
    void begin_assigned(std::optional<Acceptance>& accepted, pool::BrokerClient const& broker, pool::Lease const& lease,
        std::string const& owner, WorkerCancellation cancel) try
    {
        accepted.emplace(*this);
        for (;;)
        {
            cancel.check();
            bool awaiting_admission = false;
            // Begin can commit even if its reply arrives after a short codec
            // deadline. Read that acknowledgement so the worker can finish the
            // reservation. Execution still checks the task's original deadline.
            auto control = broker.with_deadline(std::max(cancel.deadline, Clock::now() + control_grace(config.tls)));
            if (control.begin(lease, owner, &awaiting_admission)) return;
            if (!awaiting_admission) throw pool::StaleLease("worker rejected stale lease");
            {
                std::lock_guard lock(admission_mutex);
                if (admission != pool::WorkerAdmission::Accepting)
                    throw AdmissionRefused();
            }
            // A delayed pause can reach the broker after local resume. Let
            // the next heartbeat reconcile it, then recheck the original grant.
            heartbeat_ready.notify_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    catch (AdmissionRefused const&) { report_refusal(lease, owner, cancel.deadline); throw; }
    void refresh_brokers()
    {
        auto addresses = config.brokers;
        if (config.broker.port) addresses.push_back(config.broker);
        if (config.discover_brokers)
        {
            auto discovered = config.discover_brokers();
            addresses.insert(addresses.end(), discovered.begin(), discovered.end());
        }
        rotate_brokers(addresses);
        std::vector<std::shared_ptr<Registration>> snapshot;
        {
            std::lock_guard lock(brokers_mutex);
            // Task copies retain their own channels. Failed routes age out,
            // making room for later discovery results within the RAM budget.
            std::erase_if(brokers, [](auto const& broker)
            {
                if (broker->client.shared()) return false;
                if (broker->retired) { broker->client.disconnect_idle(); return true; }
                return !broker->healthy && Clock::now() - broker->seen > std::chrono::seconds(30);
            });
            deferred_brokers = 0;
            if (!addresses.empty())
            {
                discovery_cursor %= addresses.size();
                for (size_t i = 0; i < addresses.size(); ++i)
                {
                    auto const& address = addresses[(discovery_cursor + i) % addresses.size()];
                    if (!address.port || address.host.empty()) continue;
                    if (std::any_of(brokers.begin(), brokers.end(), [&](auto const& broker)
                        { return std::find(broker->routes.begin(), broker->routes.end(), address) != broker->routes.end(); })) continue;
                    if (brokers.size() == config.max_brokers) { ++deferred_brokers; continue; }
                    brokers.push_back(std::make_shared<Registration>(address, config.tls));
                }
                discovery_cursor = (discovery_cursor + 1) % addresses.size();
            }
            if (!brokers.empty())
            {
                refresh_cursor %= brokers.size();
                // Healthy authorities get a reserved share even when many
                // discovered routes fail their handshake. The other share
                // rotates through every route without sequential TLS waits.
                size_t const healthy_quota = config.broker_refresh_parallelism / 2;
                for (size_t i = 0; i < brokers.size() && snapshot.size() < healthy_quota; ++i)
                {
                    auto const& broker = brokers[(refresh_cursor + i) % brokers.size()];
                    if (broker->healthy && !broker->retired) snapshot.push_back(broker);
                }
                for (size_t i = 0; i < brokers.size() && snapshot.size() < config.broker_refresh_parallelism; ++i)
                {
                    auto const& broker = brokers[(refresh_cursor + i) % brokers.size()];
                    if (!broker->retired && std::find(snapshot.begin(), snapshot.end(), broker) == snapshot.end()) snapshot.push_back(broker);
                }
                refresh_cursor = (refresh_cursor + std::max<size_t>(1, snapshot.size())) % brokers.size();
            }
        }
        capacity_update();
        std::vector<std::future<void>> requests;
        for (auto const& broker : snapshot) requests.push_back(std::async(std::launch::async, [this, broker]
        {
            if (stopping.load()) return;
            try
            {
                pool::Endpoint address, advertised_endpoint;
                std::string alternate_source;
                uint64_t revision;
                { std::lock_guard lock(brokers_mutex); address = broker->endpoint; advertised_endpoint = broker->advertised_endpoint;
                  revision = broker->revision; alternate_source = broker->alternate_source; }
                auto client = broker->client.with_deadline(Clock::now() + std::chrono::milliseconds(config.broker_refresh_ms));
                auto value = offer();
                bool const routed = value.endpoint.host == "0.0.0.0" || value.endpoint.host == "::";
                std::string source;
                if (routed)
                {
                    source = pool::local_address(address);
                    // Simultaneous aliases can return in a different order than the
                    // broker accepted them. Keep its working endpoint on this source.
                    if (source == alternate_source) value.endpoint = advertised_endpoint;
                    else value.endpoint.host = source;
                }
                if (revision == value.capacity_revision && value.endpoint == advertised_endpoint &&
                    client.heartbeat(instance, value.capacity_revision, value.admission))
                {
                    std::lock_guard lock(brokers_mutex); broker->seen = Clock::now(); broker->healthy = true;
                    return;
                }
                if (!advertised_endpoint.host.empty() && value.endpoint != advertised_endpoint)
                {
                    capacity_update({}, true);
                    auto route = value.endpoint; value = offer(); value.endpoint = std::move(route);
                }
                std::string peer, node;
                auto epoch = !revision && routed ? client.register_worker_route(value, &peer, &node) : client.register_worker(value, &peer, &node);
                std::lock_guard lock(brokers_mutex);
                auto same = std::find_if(brokers.begin(), brokers.end(), [&](auto const& registered)
                { return registered != broker && !registered->epoch.empty() && registered->epoch == epoch && registered->identity == peer; });
                if (same != brokers.end())
                {
                    auto& registered = **same;
                    auto route = std::find(registered.routes.begin(), registered.routes.end(), address);
                    if (route == registered.routes.end() && registered.routes.size() < max_broker_routes)
                        route = registered.routes.insert(route, address);
                    // Heartbeat the route on which the advertised endpoint was
                    // accepted so loss of that path actually triggers failover.
                    if (!registered.healthy || (!registered.alternate_source.empty() && source == value.endpoint.host))
                    {
                        if (route == registered.routes.end()) registered.routes[registered.route_cursor] = address;
                        else registered.route_cursor = size_t(route - registered.routes.begin());
                        registered.client.reroute(address); registered.endpoint = address;
                        registered.alternate_source.clear();
                    }
                    registered.seen = Clock::now(); registered.healthy = true;
                    registered.revision = value.capacity_revision;
                    registered.advertised_endpoint = value.endpoint;
                    std::erase(brokers, broker); return;
                }
                broker->epoch = std::move(epoch); broker->revision = value.capacity_revision;
                broker->identity = peer; broker->node_id = node;
                broker->advertised_endpoint = value.endpoint;
                broker->alternate_source = routed && source != value.endpoint.host ? std::move(source) : std::string{};
                broker->seen = Clock::now(); broker->healthy = true;
            }
            catch (...)
            {
                std::lock_guard lock(brokers_mutex); broker->healthy = false;
                if (broker->routes.size() > 1)
                {
                    broker->route_cursor = (broker->route_cursor + 1) % broker->routes.size();
                    broker->endpoint = broker->routes[broker->route_cursor]; broker->client.reroute(broker->endpoint);
                    broker->alternate_source.clear();
                }
            }
        }));
        for (auto& request : requests) request.get();
    }
    void rotate_brokers(std::vector<pool::Endpoint> const& addresses)
    {
        if (addresses.empty()) return;
        std::shared_ptr<Registration> candidate;
        std::optional<pool::Endpoint> replacement;
        {
            std::lock_guard lock(brokers_mutex);
            if (brokers.size() < config.max_brokers) return;
            for (size_t offset = 0; offset < addresses.size(); ++offset)
            {
                auto const& address = addresses[(discovery_cursor + offset) % addresses.size()];
                if (!address.port || address.host.empty() || std::any_of(brokers.begin(), brokers.end(), [&](auto const& broker)
                    { return std::find(broker->routes.begin(), broker->routes.end(), address) != broker->routes.end(); })) continue;
                replacement = address; break;
            }
            if (!replacement) return;
            auto const now = Clock::now(); auto const cadence = std::chrono::milliseconds(config.broker_rotation_ms);
            for (auto const& broker : brokers)
                if (broker->healthy && !broker->retired && !broker->client.shared() && now - broker->admitted >= cadence &&
                    now - broker->rotation_attempt >= cadence && (!candidate || broker->rotation_attempt < candidate->rotation_attempt)) candidate = broker;
            if (!candidate) return;
            candidate->rotation_attempt = now;
        }
        try
        {
            if (!candidate->client.with_deadline(Clock::now() + std::chrono::milliseconds(config.broker_refresh_ms)).retire_worker(instance)) return;
        }
        catch (...) { return; }
        {
            std::lock_guard lock(brokers_mutex); candidate->retired = true; candidate->healthy = false;
            // The broker atomically refused retirement if any lease remained.
            // A late stale caller can still retain a copy; keep its charged
            // transport until that caller observes the fenced lease and exits.
            if (candidate->client.shared()) return;
            candidate->client.disconnect_idle(); std::erase(brokers, candidate);
            brokers.push_back(std::make_shared<Registration>(*replacement, config.tls));
        }
    }

#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    uint64_t gpu_singleton_bytes(GpuProducerPreflight const& profile, uint8_t k, uint64_t retained) const
    {
        auto const indexes = boundary_bytes(k);
        auto const peak = add(add(profile.borrowed_host_bytes, indexes), config.max_task_bytes);
        return peak > retained ? peak - retained : indexes;
    }
    void update_idle_gpu_output()
    {
        if (!gpu) return;
        resource_sampler.locked_output(gpu->locked_host_output_bytes());
        auto const minimum = gpu_singleton_bytes(gpu_profiles.at(gpu_k), gpu_k, gpu_retained.bytes);
        if (budget->task_capacity() >= minimum) { idle_generation_bytes.store(minimum); return; }
        resource_sampler.locked_output(0);
        gpu->release_host_output();
        gpu_retained.resize(gpu->retained_host_bytes());
        retained_gpu_host.store(gpu_retained.bytes);
        idle_generation_bytes.store(0);
    }
    void discard_gpu()
    {
        resource_sampler.locked_output(0);
        gpu.reset();
        if (gpu_retained.budget) gpu_retained.resize(GpuProducer::runtime_host_allowance_bytes());
        retained_gpu_host.store(gpu_retained.bytes); idle_generation_bytes.store(0);
    }
#endif
    uint64_t generation_host_bytes(uint8_t k, uint8_t strength, size_t members, bool use_gpu, uint64_t retained) const
    {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (use_gpu)
        {
            auto const found = gpu_profiles.find(k);
            if (found == gpu_profiles.end()) throw pool::UnsupportedResources("GPU task has no admitted generation profile");
            if (members == 1) return gpu_singleton_bytes(found->second, k, retained);
            auto const indexes = multiply(boundary_bytes(k), members);
            auto const raw = multiply(fragment_count_bound(k, strength), sizeof(uint64_t));
            auto const peak = add(add(add(found->second.borrowed_host_bytes, multiply(raw, members - 1)), indexes), config.max_task_bytes);
            return peak > retained ? peak - retained : indexes;
        }
#endif
        return worker_cpu_task_bytes(k, strength, members, config.max_task_bytes);
    }
    std::vector<pool::GenerationProfile> profiles(pool::Resources const& capacity, uint64_t standing) const
    {
        std::vector<pool::GenerationProfile> result;
        auto const retained = standing - service_baseline;
        for (uint8_t k = 18; k <= 28; k += 2)
            for (uint8_t strength = STRENGTH_MIN;
                 strength <= std::min<unsigned>(STRENGTH_MAX, k - PlotProofParams::num_section_bits_for_k(k) - 1); ++strength)
            {
                uint64_t const raw = multiply(fragment_count_bound(k, strength), sizeof(uint64_t));
                pool::GenerationProfile cpu; cpu.kind = pool::Kind::GenerateCpu; cpu.k = k; cpu.strength = strength;
                cpu.cpu_threads = std::min(config.generation_threads, capacity.cpu_threads);
                cpu.resources = {cpu.cpu_threads, 1,
                    add(add(worker_cpu_generation_bytes(k, strength), boundary_bytes(k)), config.max_task_bytes), 0};
                cpu.member_host_bytes = add(raw, boundary_bytes(k));
                cpu.max_batch_members = config.max_batch_members;
                result.push_back(cpu);
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
                if (auto found = gpu_profiles.find(k); found != gpu_profiles.end())
                {
                    auto gpu = cpu; gpu.kind = pool::Kind::GenerateGpu;
                    gpu.resources.host_bytes = gpu_singleton_bytes(found->second, k, retained);
                    gpu.resources.gpu_bytes = config.capacity.gpu_bytes;
                    result.push_back(gpu);
                }
#endif
            }
        require(result.size() <= pool::max_generation_profiles, "worker generation profile limit exceeded");
        return result;
    }

    WorkerCancellation cancellation(WorkerCancellation value) const
    { value.service_stop = &stopping; value.deadline = std::min(value.deadline,
        Clock::now() + std::chrono::milliseconds(config.task_timeout_ms)); return value; }

    void gpu_loop()
    {
        for (;;)
        {
            std::packaged_task<void()> job;
            {
                std::unique_lock lock(jobs_mutex);
                jobs_ready.wait(lock, [&] { return stopping.load() || !gpu_jobs.empty() || check_idle_output; });
                check_idle_output = false;
                if (!gpu_jobs.empty()) { job = std::move(gpu_jobs.front()); gpu_jobs.pop_front(); }
                else if (stopping.load()) break;
            }
            if (job.valid()) job();
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            else try { update_idle_gpu_output(); } catch (...) { discard_gpu(); }
#endif
        }
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        resource_sampler.locked_output(0); gpu.reset(); gpu_retained.release();
#endif
    }
    void on_gpu(std::function<void()> operation)
    {
        std::packaged_task<void()> task(std::move(operation)); auto done = task.get_future();
        { std::lock_guard lock(jobs_mutex); require(!stopping.load(), "worker service stopped"); gpu_jobs.push_back(std::move(task)); }
        jobs_ready.notify_one(); done.get();
    }

    // Validates the complete member before emitting metadata. Word tiles use
    // the same low-field layout as CompactMemberBuilder without retaining a
    // second full compact member beside the reference producer's raw output.
    void packed(uint8_t k, std::span<uint64_t const> raw, WorkerPackedSink const& sink,
                WorkerCancellation const& cancel)
    {
        require(raw.size() <= UINT32_MAX, "worker compact fragment count exceeds narrow indexes");
        size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(k));
        std::vector<uint32_t> boundaries;
        uint64_t const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
        uint64_t const range = uint64_t(1) << width;
        PackedMemberValidation validation{uint64_t(raw.size()), 0};
        {
            CpuWork compute(offered_cpu, scheduler.get(), cancel);
            boundaries.resize(chunks + 1);
            size_t cursor = 0; uint64_t previous = 0;
            uint64_t const maximum_delta = uint64_t(72 - k) * (178ull << (k - 8));
            for (size_t chunk = 0; chunk < chunks; ++chunk)
            {
                cancel.check(); boundaries[chunk] = uint32_t(cursor);
                uint64_t const start = uint64_t(chunk) * range;
                previous = start;
                while (cursor < raw.size() && (raw[cursor] >> width) == chunk)
                {
                    require(raw[cursor] >= previous, "worker produced unsorted member fragments");
                    if (raw[cursor] - previous >= maximum_delta) validation.flags |= PackedMemberValidation::wide_delta;
                    previous = raw[cursor++];
                }
            }
            boundaries[chunks] = uint32_t(cursor);
            require(cursor == raw.size(), "worker produced out-of-range member fragments");
        }
        cancel.check(); sink({boundaries, {}, validation}, raw.size());
        size_t const words_per_tile = std::max<size_t>(1, (config.tls.max_frame_bytes - 128) / sizeof(uint64_t));
        std::vector<uint64_t> words; words.reserve(words_per_tile);
        uint64_t word = 0; unsigned filled = 0;
        size_t cursor = 0;
        while (cursor < raw.size())
        {
            {
                CpuWork compute(offered_cpu, scheduler.get(), cancel);
                while (cursor < raw.size() && words.size() < words_per_tile)
                {
                    uint64_t const value = raw[cursor++] & (range - 1);
                    word |= value << filled;
                    unsigned const available = 64 - filled;
                    if (width >= available)
                    { words.push_back(word); word = width == available ? 0 : value >> available; filled = unsigned(width - available); }
                    else filled += unsigned(width);
                }
                if (cursor == raw.size() && filled && words.size() < words_per_tile)
                { words.push_back(word); word = 0; filled = 0; }
            }
            cancel.check(); if (!words.empty()) sink({{}, words, {}}, raw.size()); words.clear();
        }
        if (filled) { cancel.check(); sink({{}, std::span<uint64_t const>(&word, 1), {}}, raw.size()); }
    }

    void generation(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
        WorkerPackedSink const& packed_sink, ContributionBlockSink const& contribution_sink,
        WorkerCancellation cancel, bool use_gpu, bool task_slot = true, pool::Resources const* grant = nullptr,
        std::function<void(WorkerTaskAdmission const&)> const& admit = {}, WorkerFreshManifest const& manifest = {},
        bool independent = false)
    {
        cancel = cancellation(cancel); cancel.check();
        require(!ids.empty() && ids.size() <= config.max_batch_members &&
            (bool(packed_sink) != bool(contribution_sink)), "worker generation batch exceeds configured limit");
        require(!independent || (use_gpu && contribution_sink && !manifest), "invalid independent contribution task");
        size_t const retained_members = independent ? 1 : ids.size();
        auto const bound = fragment_count_bound(k, strength);
        uint64_t const raw_bytes = multiply(bound, sizeof(uint64_t));
        uint64_t const indexes = multiply(boundary_bytes(k), retained_members);
        uint64_t const prior_retained = retained_gpu_host.load();
        unsigned const threads = grant ? grant->cpu_threads : std::min(config.generation_threads, current_capacity().cpu_threads);
        require(threads && threads <= config.capacity.cpu_threads, "generation CPU allowance exceeds worker offer");
        uint64_t const dynamic_peak = generation_host_bytes(k, strength, retained_members, use_gpu, prior_retained);
        if (grant && dynamic_peak > grant->host_bytes)
            throw pool::UnsupportedResources("generation model exceeds leased host memory");
        if (admit)
        {
            WorkerTaskAdmission value; value.worker = status(); value.local_fallback = true;
            value.task.compatibility = worker_compatibility(); value.task.kind = use_gpu ? pool::Kind::GenerateGpu : pool::Kind::GenerateCpu;
            value.task.k = k; value.task.strength = strength; value.task.work_units = ids.size();
            value.task.generation_output = generation_output(packed_sink ? Operation::Packed : manifest ? Operation::Fresh : Operation::Contributions);
            value.task.independent_members = independent;
            value.task.resources = {threads, 1, dynamic_peak, use_gpu ? config.capacity.gpu_bytes : 0};
            admit(value); cancel.check();
        }
        auto memory = budget->acquire(dynamic_peak, task_slot, cancel);
        auto execute = [&]
        {
            static bool const profile = [] { auto v = std::getenv("POS2GPU_PHASE_TIMING"); return v && *v == '1'; }();
            auto const start = profile ? Clock::now() : Clock::time_point{};
            Clock::duration encoding{}, independent_generation{};
            unsigned codec_lanes = 0;
            uint64_t task_retained = 0;
            uint64_t generated_retained = 0;
            bool delivered_packed = false;
            std::vector<Member> compact;
            std::unique_ptr<MemberBatchHash> hash;
            std::vector<std::vector<uint64_t>> owned;
            std::vector<std::span<uint64_t const>> members; members.reserve(ids.size());
            auto encode_contributions = [&](size_t first_member, size_t count, unsigned available_threads)
            {
                size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(k));
                std::vector<Member> views;
                std::vector<FragmentBoundaries> boundaries;
                uint64_t resident = 0, maximum_symbols = 0, payload = 0, largest = 0;
                uint64_t const first_chunk = first_member * chunks;
                if (compact.empty())
                {
                    for (auto const& raw : owned) resident = add(resident, multiply(raw.capacity(), sizeof(uint64_t)));
                    for (auto raw : members)
                    {
                        CpuWork cpu(offered_cpu, scheduler.get(), cancel);
                        require(raw.size() <= UINT32_MAX, "invalid worker generation output");
                        views.push_back(Member::borrowed(raw));
                        auto& index = boundaries.emplace_back(); index.reset(chunks, raw.size());
                        size_t cursor = 0; uint64_t previous = 0;
                        for (size_t c = 0; c < chunks; ++c)
                        {
                            if (!(c % initial_contribution_window_chunks)) cancel.check();
                            size_t const begin = cursor; index.set(c, begin);
                            while (cursor < raw.size() && (raw[cursor] >> (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS)) == c)
                            {
                                require(raw[cursor] >= previous, "invalid worker generation output");
                                previous = raw[cursor++];
                            }
                            if (count == 1) maximum_symbols = std::max<uint64_t>(maximum_symbols, cursor - begin);
                        }
                        require(cursor == raw.size(), "out-of-range worker member"); index.set(chunks, cursor);
                    }
                }
                else for (auto const& member : compact) resident = add(resident, member.storage_bytes());
                Members const source(compact.empty() ? views : compact);
                require(source.size() == count, "worker generation lost a batch member");
                auto regions = [&](size_t chunk, FragmentRegion* result)
                {
                    uint64_t symbols = 0;
                    for (size_t m = 0; m < count; ++m)
                    {
                        auto view = source[m];
                        result[m] = view.compact() ? view.region(chunk) :
                            view.region(chunk, boundaries[m][chunk], boundaries[m][chunk + 1] - boundaries[m][chunk]);
                        symbols += result[m].size();
                    }
                    return symbols;
                };
                if (manifest || count > 1)
                {
                    CpuWork cpu(offered_cpu, scheduler.get(), cancel);
                    std::vector<FragmentRegion> row(count);
                    for (size_t c = 0; c < chunks; ++c)
                    {
                        if (!(c % initial_contribution_window_chunks)) cancel.check();
                        maximum_symbols = std::max(maximum_symbols, regions(c, row.data()));
                        if (manifest)
                        {
                            uint64_t const bytes = ChunkCodec::fresh_bound(k, c, row);
                            payload = add(payload, bytes); largest = std::max(largest, bytes);
                        }
                    }
                }
                auto reserve_codec = [&](uint64_t bytes)
                {
                    auto const needed = add(add(add(resident, indexes), bytes), generated_retained);
                    if (grant && needed > add(grant->host_bytes, use_gpu ? prior_retained : 0))
                        throw pool::UnsupportedResources("initial contribution codec exceeds leased host memory");
                    auto const dynamic = needed > task_retained ? needed - task_retained : indexes;
                    if (dynamic > memory.bytes) memory.resize(dynamic);
                };
                auto const maximum_work = ChunkCodec::estimate_work(k, 0, maximum_symbols);
                if (maximum_work > config.max_task_bytes)
                    throw pool::UnsupportedResources("initial contribution codec exceeds configured task scratch limit");
                auto max_window = manifest ? fresh_group_window_chunks : initial_contribution_window_chunks;
                unsigned lanes = available_threads;
                auto const row_bytes = sizeof(ChunkInput) + sizeof(uint64_t) + count * sizeof(FragmentRegion);
                auto overhead = [&] { return ParallelEncoder::memory_bytes(lanes) +
                    max_window * row_bytes +
                    (manifest ? MemberBatchHash::memory_bytes(count) + member_digest_words * sizeof(uint64_t) : 0); };
                while (lanes > 1 && add(multiply(maximum_work, lanes), overhead()) > config.max_task_bytes) --lanes;
                while (max_window > 1 && add(multiply(maximum_work, lanes), overhead()) > config.max_task_bytes) max_window /= 2;
                if (!manifest && lanes > 1)
                {
                    // Amortize codec queue and output handoffs using spare task RAM.
                    // Each lane retains its tables and one coalesced output block.
                    auto const tables = ChunkCodec::estimate_base();
                    auto const fixed = add(ParallelEncoder::memory_bytes(lanes), multiply(tables, lanes));
                    auto const room = config.max_task_bytes - std::min(config.max_task_bytes, fixed);
                    auto const per_lane = room / lanes / add(row_bytes, maximum_work - tables);
                    max_window = std::max(max_window, size_t(std::min<uint64_t>(contribution_window_chunks, per_lane * lanes)));
                }
                codec_lanes = lanes;
                if (!manifest && lanes == 1)
                {
                    reserve_codec(maximum_work);
                    ChunkCodec codec(k, maximum_work, scheduler);
                    std::vector<FragmentRegion> row(count);
                    for (size_t c = 0; c < chunks; ++c)
                    {
                        cancel.check();
                        std::span<uint8_t const> bytes;
                        {
                            CpuWork cpu(offered_cpu, scheduler.get(), cancel);
                            auto const begin = profile ? Clock::now() : Clock::time_point{};
                            codec.prepare(ChunkCodec::estimate_work(k, 0, regions(c, row.data())));
                            bytes = codec.encode(c, {}, 0, row);
                            if (profile) encoding += Clock::now() - begin;
                        }
                        uint64_t const size = bytes.size();
                        cancel.check(); contribution_sink(first_chunk + c, {&size, 1}, bytes, uint16_t(count));
                    }
                }
                else
                {
                    if (add(multiply(maximum_work, lanes), overhead()) > config.max_task_bytes)
                        throw pool::UnsupportedResources("fresh group codec exceeds configured task scratch limit");
                    uint64_t const scratch = config.max_task_bytes - overhead();
                    reserve_codec(config.max_task_bytes);
                    if (manifest)
                    {
                        std::array<uint8_t, 32> digest;
                        { CpuWork cpu(offered_cpu, scheduler.get(), cancel);
                          digest = members_digest(k, source, hash.get(), [&] { cancel.check(); }); }
                        manifest(payload, largest, digest);
                    }
                    ParallelEncoder encoder(k, lanes, scratch, scheduler);
                    size_t const per_lane = size_t(std::min<uint64_t>(max_window,
                        (scratch / lanes - ChunkCodec::estimate_base()) / std::max<uint64_t>(1, maximum_work - ChunkCodec::estimate_base())));
                    size_t const window = std::min<size_t>(max_window, lanes * per_lane);
                    std::vector<FragmentRegion> rows(count * window);
                    std::vector<ChunkInput> tasks(window);
                    std::vector<uint64_t> sizes(window);
                    for (size_t first = 0; first < chunks;)
                    {
                        cancel.check();
                        size_t n;
                        std::span<EncodedBlock const> blocks;
                        {
                            CpuAdmission::Lease offered(offered_cpu, lanes, cancel, true);
                            n = std::min({chunks - first, window, per_lane * offered.permits()});
                            auto const begin = profile ? Clock::now() : Clock::time_point{};
                            for (size_t i = 0; i < n; ++i)
                            {
                                auto* row = rows.data() + i * count;
                                regions(first + i, row);
                                tasks[i] = {first + i, {}, 0, MemberRegions(row, count)};
                            }
                            blocks = encoder.encode(std::span(tasks).first(n), std::span(sizes).first(n), false, {}, offered.permits());
                            if (profile) encoding += Clock::now() - begin;
                        }
                        for (auto const& block : blocks)
                        {
                            cancel.check();
                            contribution_sink(first_chunk + first + block.begin, std::span(sizes).subspan(block.begin, block.end - block.begin),
                                block.bytes, uint16_t(count));
                        }
                        first += n;
                    }
                }
            };
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            std::optional<GpuProducer::BorrowedFragments> borrowed;
            if (use_gpu)
            {
                if (!gpu_retained.budget)
                    gpu_retained = budget->acquire(GpuProducer::runtime_host_allowance_bytes(), false, cancel, true);
                if (!gpu || gpu_k != k || gpu_strength != strength || gpu_threads != threads)
                {
                    // Keep the old persistent reservation charged while
                    // rebuilding. Its bytes fund the new producer inside the
                    // original grant; advertising them as free mid-task would
                    // overstate what concurrent CPU/codec leases can use.
                    gpu.reset();
                    GpuProducerOptions options; options.device_id = config.device; options.tier = gpu_profiles.at(k).tier;
                    options.max_vram_bytes = grant ? grant->gpu_bytes : config.capacity.gpu_bytes;
                    options.host_threads = std::max(1u, threads - 1);
                    options.member_overlap = options.compact_transfer = options.preserve_match_scratch = true;
                    CpuWork compute(offered_cpu, scheduler.get(), cancel, threads);
                    try { gpu = std::make_unique<GpuProducer>(k, strength, options); }
                    catch (std::exception const& error) { throw pool::UnsupportedResources(error.what()); }
                    gpu_k = k; gpu_strength = strength; gpu_threads = threads;
                }
                // Only earlier members need a separate copy. The last member can
                // keep the producer's pinned output until encoding and sending finish.
                auto const generation = add(gpu->host_borrowed_generation_peak_bytes(), multiply(raw_bytes, retained_members - 1));
                auto const peak = add(add(generation, indexes), config.max_task_bytes);
                if (grant && peak > add(grant->host_bytes, prior_retained))
                    throw pool::UnsupportedResources("GPU generation model exceeds leased host memory");
                memory.resize(peak > gpu_retained.bytes ? peak - gpu_retained.bytes : indexes);
                task_retained = gpu_retained.bytes;
                if (independent)
                {
                    if (threads < 2 || gpu->tier() != "pool")
                        throw pool::UnsupportedResources("independent contributions require a pooled GPU and a codec thread");
                    generated_retained = gpu->host_borrowed_generation_peak_bytes();
                    try
                    {
                        for (size_t i = 0; i < ids.size(); ++i)
                        {
                            cancel.check();
                            auto encode = [&, i](std::span<uint64_t const> raw)
                            {
                                require(raw.size() <= bound, "worker GPU output exceeds admitted fragment bound");
                                members.assign(1, raw);
                                encode_contributions(i, 1, threads - 1);
                            };
                            std::optional<CpuWork> compute(std::in_place, offered_cpu, scheduler.get(), cancel);
                            auto const begin = profile ? Clock::now() : Clock::time_point{};
                            gpu->generate_borrowed_into(ids[i], encode, [&] { compute.reset(); });
                            if (profile) independent_generation += Clock::now() - begin;
                        }
                        gpu->wait_for_output();
                    }
                    catch (...) { try { gpu->wait_for_output(); } catch (...) {} throw; }
                }
                else if ((packed_sink || manifest) && threads > 1 && gpu->compact_transfer_available() &&
                    add(gpu->host_tiled_generation_peak_bytes(),
                        manifest ? multiply(Member::compact_bytes(k, bound), ids.size()) : 0) <= generation)
                {
                    // The ordered consumer owns one CPU permit for framing. The
                    // GPU thread may wait for that consumer before its next sort.
                    std::shared_future<uint64_t> output;
                    std::optional<CompactMemberReceiver> receiver;
                    size_t words = 0;
                    if (manifest) { compact.reserve(ids.size()); hash = std::make_unique<MemberBatchHash>(ids.size()); }
                    try
                    {
                        for (auto const& id : ids)
                        {
                            cancel.check();
                            CpuWork compute(offered_cpu, scheduler.get(), cancel, threads - 1);
                            output = gpu->generate_packed_into_async(id, [&](auto const& tile, uint64_t count)
                            {
                                cancel.check();
                                require(count <= bound, "worker GPU output exceeds admitted fragment bound");
                                if (packed_sink) { packed_sink(tile, count); return; }
                                CpuWork cpu(offered_cpu, scheduler.get(), cancel);
                                if (!tile.boundaries.empty())
                                { receiver.emplace(k, size_t(count), true, hash.get()); words = 0; }
                                receiver->append(tile); words += tile.words.size();
                                if (words == (count * (k + PlotGroupFile::PROOFS_PER_CHUNK_BITS) + 63) / 64)
                                { compact.push_back(receiver->finish()); receiver.reset(); }
                            });
                        }
                        output.get();
                    }
                    catch (...) { if (output.valid()) output.wait(); throw; }
                    delivered_packed = true;
                }
                owned.reserve(ids.size());
                if (!independent && !delivered_packed) for (auto const& id : ids)
                {
                    cancel.check();
                    CpuWork compute(offered_cpu, scheduler.get(), cancel, threads);
                    if (members.size() + 1 == ids.size()) { borrowed.emplace(gpu->generate_borrowed(id)); members.push_back(borrowed->fragments()); }
                    else { owned.push_back(gpu->generate(id)); members.push_back(owned.back()); }
                    require(members.back().size() <= bound, "worker GPU output exceeds admitted fragment bound");
                }
                generated_retained = gpu->retained_host_bytes();
            }
            else
#endif
            {
                std::optional<pos2gpu::ScopedThreadAffinity> affinity;
                if (!config.cpu_affinity.empty())
                {
                    affinity.emplace();
                    if (!pos2gpu::pin_thread_to_cpus(config.cpu_affinity))
                        std::fprintf(stderr, "Compute CPU worker could not pin to CPUs %s; running unpinned\n",
                            pos2gpu::format_cpu_list(config.cpu_affinity).c_str());
                }
                owned.reserve(ids.size());
                for (auto const& id : ids)
                {
                    cancel.check();
                    CpuWork compute(offered_cpu, scheduler.get(), cancel, threads, true);
                    worker_detail::CpuLimit limit(compute.threads());
                    WorkerReferencePlotter::Options options{}; options.validate = false; options.verbose = false;
                    CancelProgress progress(cancel); options.sink = &progress;
                    owned.push_back(WorkerReferencePlotter(parameters(k, strength, id)).run(options).t3_proof_fragments);
                    cancel.check();
                    require(owned.back().size() <= bound, "worker CPU output exceeds admitted fragment bound");
                    members.push_back(owned.back());
                }
            }
            auto const generated = profile ? Clock::now() : Clock::time_point{};
            if (packed_sink)
            { for (auto raw : members) packed(k, raw, packed_sink, cancel); }
            else if (!independent) encode_contributions(0, ids.size(), threads);
            if (profile)
                std::fprintf(stderr, "[worker-generation] mode=%s members=%zu lanes=%u generation_ms=%.3f encode_ms=%.3f\n",
                    independent ? "independent" : packed_sink ? "packed" : manifest ? "fresh" : "contributions", ids.size(), codec_lanes,
                    std::chrono::duration<double, std::milli>(independent ? independent_generation : generated - start).count(),
                    std::chrono::duration<double, std::milli>(encoding).count());
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            if (use_gpu)
            {
                auto const retained = gpu->retained_host_bytes();
                if (retained > gpu_retained.bytes) memory.transfer(gpu_retained, retained - gpu_retained.bytes);
                else gpu_retained.resize(retained);
                retained_gpu_host.store(gpu_retained.bytes);
            }
#endif
        };
        if (use_gpu) on_gpu([&]
        {
            struct ActiveGpu { std::atomic<bool>& value; ~ActiveGpu() { value.store(false); } } active{gpu_active};
            gpu_active.store(true);
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
            // A previously admitted request may have been waiting for this
            // thread when another task returned its cached host allocation.
            if (gpu_retained.bytes < prior_retained)
                throw pool::TransientTransport("GPU output reservation changed before generation");
            idle_generation_bytes.store(0);
            // This task can change tiers or switch from a full buffer to a
            // transfer window. Restore credit once its actual output is known.
            resource_sampler.locked_output(0);
#endif
            try
            {
                execute();
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
                // Every output consumer has finished. If retaining its buffer
                // excludes even one more member, return those pages before the
                // broker takes its next capacity snapshot.
                memory.resize(0);
                update_idle_gpu_output();
#endif
            }
            catch (...)
            {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
                discard_gpu();
#endif
                throw;
            }
        });
        else execute();
    }

    void encode(uint8_t k, std::span<uint8_t const> payload, uint64_t maximum_output,
                std::function<void(size_t, std::span<uint8_t const>)> const& sink, WorkerCancellation cancel,
                Budget::Lease* admitted = nullptr, uint32_t expected_chunks = 0)
    {
        cancel = cancellation(cancel); cancel.check(); (void)parameters(k, 2);
        Budget::Lease memory;
        uint64_t const reserved = add(multiply(payload.size(), 20), ChunkCodec::estimate_work(k, 0, 0));
        if (!admitted) memory = budget->acquire(reserved, true, cancel);
        else require(admitted->bytes >= reserved, "worker codec window was not admitted before receipt");
        pool::Reader input(payload); std::vector<OwnedChunk> chunks;
        { CpuWork compute(offered_cpu, scheduler.get(), cancel);
          chunks = read_chunks(input, config.max_window_chunks, config.max_task_bytes, expected_chunks); }
        uint64_t work = 0;
        auto codec = [&]
        {
            CpuWork compute(offered_cpu, scheduler.get(), cancel);
            for (auto const& chunk : chunks) work = std::max(work, chunk_work(k, chunk));
            require(add(multiply(payload.size(), 4), work) <= reserved, "worker codec model exceeds admitted window");
            return ChunkCodec(k, work, scheduler);
        }();
        uint64_t output = 0;
        for (size_t i = 0; i < chunks.size(); ++i)
        {
            cancel.check(); auto const& chunk = chunks[i];
            std::span<uint8_t const> encoded;
            {
                CpuWork compute(offered_cpu, scheduler.get(), cancel);
                std::vector<std::span<uint64_t const>> members; members.reserve(chunk.members.size());
                for (auto const& values : chunk.members) members.push_back(values);
                std::vector<EncodedChunk> contributions; contributions.reserve(chunk.contributions.size() + 1);
                if (!chunk.old.empty()) contributions.push_back({chunk.old, chunk.old_members});
                else require(!chunk.old_members, "invalid worker old contribution");
                for (auto const& value : chunk.contributions) contributions.push_back({value.bytes, value.members});
                codec.prepare(chunk_work(k, chunk));
                encoded = contributions.empty() ? codec.encode(chunk.ordinal, {}, 0, members) : codec.join(chunk.ordinal, contributions, members);
                output = add(output, encoded.size()); require(output <= maximum_output, "worker encoded output exceeds owner bound");
            }
            cancel.check(); sink(i, encoded);
        }
    }

    std::vector<uint8_t> receive_blob(pool::Channel& channel, uint64_t count, uint64_t maximum,
                                     WorkerCancellation const& cancel)
    {
        require(count <= maximum && count <= SIZE_MAX, "worker input exceeds bounded window");
        std::vector<uint8_t> bytes;
        { CpuWork compute(offered_cpu, scheduler.get(), cancel); bytes.reserve(size_t(count)); }
        while (bytes.size() < count)
        {
            cancel.check(); auto frame = channel.receive();
            CpuWork compute(offered_cpu, scheduler.get(), cancel);
            require(frame.type == pool::Message::Data && !frame.payload.empty() &&
                frame.payload.size() <= count - bytes.size(), "invalid worker input window");
            bytes.insert(bytes.end(), frame.payload.begin(), frame.payload.end());
        }
        return bytes;
    }

    void handle_control(pool::Channel& channel, pool::Frame const& frame)
    {
        try
        {
            pool::Reader in(frame.payload); require(in.u32() == pool::protocol_version, "incompatible worker control protocol");
            auto const operation = in.u8(); in.finish(); WorkerStatus value;
            if (!operation) value = status();
            else
            {
                bool const same_identity = config.tls.membership_directory.empty() ?
                    channel.peer_identity() == pool::certificate_identity(config.tls.cert_file) :
                    channel.peer_node_id() == pool::certificate_node_id(config.tls.cert_file);
                require(same_identity, "worker control requires this machine's identity");
                value = control(WorkerControl(operation));
            }
            pool::Writer out(config.tls.max_frame_bytes); write_worker_status(out, value); channel.send(pool::Message::Reply, out.data());
        }
        catch (std::exception const& error)
        {
            try { pool::Writer out(config.tls.max_frame_bytes); out.u8(uint8_t(error_kind(error))); out.string(error.what());
                channel.send(pool::Message::Error, out.data()); } catch (...) {}
        }
    }
    void handle(pool::Channel channel, pool::Frame frame)
    {
        std::optional<pool::Lease> lease;
        std::optional<pool::BrokerClient> broker;
        uint64_t registered_revision = 0;
        std::atomic<bool> cancelled{false};
        std::exception_ptr renewal_failure;
        bool begun = false, output_started = false;
        bool generation_failpoint_checked = false;
        std::mutex output_mutex;
        bool keepalive_enabled = false;
        std::optional<Acceptance> accepted;
        bool const timed = channel.generation_timing();
        Clock::duration sending{}, processing{};
        auto send = [&](pool::Message type, std::span<uint8_t const> bytes)
        {
            auto const start = timed ? Clock::now() : Clock::time_point{};
            std::lock_guard lock(output_mutex); channel.send(type, bytes);
            if (timed) sending += Clock::now() - start;
            if (type == pool::Message::Data && !std::exchange(generation_failpoint_checked, true))
            {
                auto const failpoint = std::getenv("XCHPLOT2_GROW_FAILPOINT");
                if (failpoint && std::string_view(failpoint) == "worker-generation-partial") std::_Exit(86);
#ifndef _WIN32
                if (failpoint && std::string_view(failpoint) == "worker-generation-paused") std::raise(SIGSTOP);
#endif
            }
        };
        try
        {
            require(frame.type == pool::Message::Work, "worker expected leased task");
            pool::Reader in(frame.payload); lease = pool::read_lease(in);
            require(lease->worker_instance == instance && lease->task.compatibility == worker_compatibility(lease->task.kind), "worker lease targets another executor or codec");
            auto const task_timeout = in.u32();
            require(task_timeout && task_timeout <= config.task_timeout_ms, "worker task timeout exceeds configured limit");
            WorkerCancellation cancel{&cancelled, Clock::now() + std::chrono::milliseconds(task_timeout), &stopping};
            channel.set_deadline(cancel.deadline);
            broker = control_for(*lease, &registered_revision);
            auto active = broker->with_deadline(cancel.deadline);
            begin_assigned(accepted, *broker, *lease, channel.peer_identity(), cancel);
            begun = true;
            Renewal renewal(active, *lease, cancelled, renewal_failure, [&]
            {
                std::lock_guard lock(output_mutex);
                if (keepalive_enabled) channel.send(pool::Message::Reply, {});
            }, config.tls.timeout_ms);
            auto task_slot = budget->acquire(0, true, cancel);
            auto const operation = Operation(in.u8()); uint8_t const k = in.u8(), strength = in.u8();
            require(generation_output(operation) == lease->task.generation_output, "worker generation output differs from lease");
            require(k == lease->task.k && strength == lease->task.strength, "worker parameters differ from lease");
            if (operation == Operation::Packed || operation == Operation::Contributions || operation == Operation::Fresh)
            {
                require(lease->task.kind != pool::Kind::Pack, "packing lease cannot generate members");
                auto const count = in.u16(); require(count && count <= config.max_batch_members && count == lease->task.work_units,
                    "worker generation batch exceeds offer or differs from lease");
                std::vector<std::array<uint8_t, 32>> ids;
                {
                    CpuWork compute(offered_cpu, scheduler.get(), cancel); ids.resize(count);
                    for (auto& id : ids) { auto const value = in.bytes(32); require(value.size() == id.size(), "invalid worker plot ID"); std::copy(value.begin(), value.end(), id.begin()); }
                    in.finish();
                }
                // The request is fully decoded. Its reserved frame can now hold
                // the previous contribution while the encoder fills the next one.
                std::vector<uint8_t>().swap(frame.payload);
                { std::lock_guard lock(output_mutex); keepalive_enabled = true; }
                bool const use_gpu = lease->task.kind == pool::Kind::GenerateGpu || (lease->task.kind == pool::Kind::GenerateAny && gpu_enabled);
                require(!use_gpu || gpu_enabled, "GPU generation was not offered by this worker");
                auto packed_sink = [&](PackedFragmentTile const& tile, uint64_t total)
                {
                    output_started = true;
                    if (!tile.boundaries.empty())
                    {
                        pool::Writer prelude(config.tls.max_frame_bytes);
                        { CpuWork compute(offered_cpu, scheduler.get(), cancel); prelude.u8(uint8_t(DataKind::Prelude)); prelude.u64(total);
                          prelude.u64(tile.validation.checked_count); prelude.u32(tile.validation.flags); prelude.u64(tile.boundaries.size()); }
                        cancel.check(); send(pool::Message::Data, prelude.data());
                        auto boundaries = tile.boundaries;
                        while (!boundaries.empty())
                        {
                            pool::Writer out(config.tls.max_frame_bytes);
                            auto const part = boundaries.first(std::min<size_t>(boundaries.size(), (config.tls.max_frame_bytes - 16) / 4));
                            { CpuWork compute(offered_cpu, scheduler.get(), cancel); out.u8(uint8_t(DataKind::Boundaries));
                              out.u32(uint32_t(part.size())); for (auto value : part) out.u32(value); }
                            cancel.check(); send(pool::Message::Data, out.data()); boundaries = boundaries.subspan(part.size());
                        }
                    }
                    auto words = tile.words;
                    while (!words.empty())
                    {
                        pool::Writer out(config.tls.max_frame_bytes);
                        auto const part = words.first(std::min<size_t>(words.size(), (config.tls.max_frame_bytes - 16) / 8));
                        { CpuWork compute(offered_cpu, scheduler.get(), cancel); out.u8(uint8_t(DataKind::Words));
                          out.u32(uint32_t(part.size())); out.words(part); }
                        cancel.check(); send(pool::Message::Data, out.data()); words = words.subspan(part.size());
                    }
                };
                pool::Writer output(config.tls.max_frame_bytes), outgoing(config.tls.max_frame_bytes);
                output.u8(uint8_t(DataKind::Contributions));
                bool first_flush = true;
                auto send_contributions = [&](pool::Writer const& frame)
                {
                    send(pool::Message::Data, frame.data());
                    if (std::exchange(first_flush, false))
                    {
                        auto const failpoint = std::getenv("XCHPLOT2_GROW_FAILPOINT");
                        if (failpoint && std::string_view(failpoint) == "worker-contribution-partial") std::_Exit(86);
#ifndef _WIN32
                        if (failpoint && std::string_view(failpoint) == "worker-contribution-paused") std::raise(SIGSTOP);
#endif
                    }
                };
                std::optional<OrderedWriter> sender;
                std::future<void> sent;
                auto flush = [&]
                {
                    if (output.data().size() <= 1) return;
                    cancel.check(); output_started = true;
                    if (operation == Operation::Contributions && lease->task.resources.cpu_threads > 1)
                    {
                        if (!sender) sender.emplace([] {});
                        if (sent.valid()) sent.get();
                        std::swap(output, outgoing);
                        sent = sender->submit([&] { send_contributions(outgoing); });
                    }
                    else send_contributions(output);
                    output.clear(); output.u8(uint8_t(DataKind::Contributions));
                };
                auto contribution_sink = [&](uint64_t first, std::span<uint64_t const> sizes,
                    std::span<uint8_t const> bytes, uint16_t members)
                {
                    // Charge serialization once per borrowed block or frame.
                    // Release the CPU to other tasks while the socket drains.
                    std::optional<CpuWork> compute;
                    for (auto size : sizes)
                    {
                        require(size + 32 <= config.tls.max_frame_bytes, "worker contribution exceeds transport frame");
                        if (output.data().size() + size + 14 > config.tls.max_frame_bytes)
                        { compute.reset(); flush(); }
                        if (!compute) compute.emplace(offered_cpu, scheduler.get(), cancel);
                        output.u64(first++); output.u16(members); output.bytes(bytes.first(size_t(size)));
                        bytes = bytes.subspan(size_t(size));
                    }
                };
                auto manifest = [&](uint64_t payload, uint64_t largest, std::array<uint8_t, 32> const& digest)
                {
                    pool::Writer out(config.tls.max_frame_bytes);
                    out.u8(uint8_t(DataKind::FreshManifest)); out.u64(payload); out.u64(largest); out.bytes(digest);
                    cancel.check(); output_started = true; send(pool::Message::Data, out.data());
                };
                auto const start = Clock::now();
                generation(k, strength, ids, operation == Operation::Packed ? WorkerPackedSink(packed_sink) : WorkerPackedSink{},
                    operation != Operation::Packed ? ContributionBlockSink(contribution_sink) : ContributionBlockSink{},
                    cancel, use_gpu, false, &lease->task.resources, {}, operation == Operation::Fresh ? WorkerFreshManifest(manifest) : WorkerFreshManifest{},
                    lease->task.independent_members);
                flush();
                if (sent.valid()) sent.get();
                processing = Clock::now() - start - sending;
            }
            else if (operation == Operation::Encode)
            {
                require(lease->task.kind == pool::Kind::Pack, "generation lease cannot pack chunks");
                uint64_t const bytes = in.u64(), maximum = in.u64(); in.finish();
                require(bytes == lease->task.work_units && maximum == lease->task.max_output_bytes,
                    "worker packing window differs from lease");
                if (!lease->task.chunk_count || lease->task.chunk_count > config.max_window_chunks ||
                    bytes > config.max_task_bytes || maximum > config.max_task_bytes)
                    throw pool::UnsupportedResources("worker packing window exceeds configured limit");
                if (add(multiply(bytes, 20), ChunkCodec::estimate_work(k, 0, 0)) > lease->task.resources.host_bytes)
                    throw pool::UnsupportedResources("packing window exceeds leased host memory");
                // The incoming window has its own charged lifetime before the
                // codec's decoded representations and reusable scratch exist.
                auto input_memory = budget->acquire(add(multiply(bytes, 20), ChunkCodec::estimate_work(k, 0, 0)), false, cancel);
                auto payload = receive_blob(channel, bytes, config.max_task_bytes, cancel);
                { std::lock_guard lock(output_mutex); keepalive_enabled = true; }
                encode(k, payload, maximum, [&](size_t index, std::span<uint8_t const> encoded)
                {
                    pool::Writer header(config.tls.max_frame_bytes);
                    { CpuWork compute(offered_cpu, scheduler.get(), cancel); header.u32(uint32_t(index)); header.u64(encoded.size()); }
                    output_started = true;
                    cancel.check(); send(pool::Message::Result, header.data());
                    uint64_t offset = 0;
                    while (!encoded.empty())
                    {
                        pool::Writer out(config.tls.max_frame_bytes);
                        auto const part = encoded.first(std::min<size_t>(encoded.size(), config.tls.max_frame_bytes - 32));
                        { CpuWork compute(offered_cpu, scheduler.get(), cancel); out.u8(uint8_t(DataKind::Encoded));
                          out.u32(uint32_t(index)); out.u64(offset); out.bytes(part); }
                        cancel.check(); send(pool::Message::Data, out.data());
                        offset += part.size(); encoded = encoded.subspan(part.size());
                    }
                }, cancel, &input_memory, lease->task.chunk_count);
            }
            else throw std::runtime_error("unknown worker operation");
            cancel.check();
            pool::Writer end(config.tls.max_frame_bytes);
            if (timed && operation != Operation::Encode)
            {
                end.u64(std::chrono::duration_cast<std::chrono::nanoseconds>(processing).count());
                end.u64(std::chrono::duration_cast<std::chrono::nanoseconds>(sending).count());
                end.u32(generation_slots(lease->task));
            }
            { std::lock_guard lock(output_mutex); keepalive_enabled = false; channel.send(pool::Message::End, end.data()); }
        }
        catch (std::exception const&)
        {
            WorkerError kind = WorkerError::Permanent; std::string message;
            try { std::rethrow_exception(task_failure(std::current_exception(), renewal_failure)); }
            catch (std::exception const& error) { kind = error_kind(error); message = error.what(); }
            bool const unsupported = !output_started && kind == WorkerError::Unsupported;
            if (kind == WorkerError::Unsupported && !unsupported) kind = WorkerError::Permanent;
            if (unsupported && begun) try
            { auto cleanup = worker_detail::cleanup_control(*broker, config.tls);
              cleanup.reject(*lease, pool::FailureCode::UnsupportedResources, capacity_update(registered_revision)); }
            catch (...)
            {
                // Other granted leases may still rely on the old capacity.
                // Release this exact admission before retrying the snapshot.
                try { worker_detail::cleanup_control(*broker, config.tls).reject(*lease, pool::FailureCode::UnsupportedResources); } catch (...) {}
            }
            try
            {
                pool::Writer out(config.tls.max_frame_bytes); out.u8(uint8_t(kind)); out.string(message);
                std::lock_guard lock(output_mutex); keepalive_enabled = false; channel.send(pool::Message::Error, out.data());
            }
            catch (...) {}
        }
        if (begun)
        {
            try { worker_detail::cleanup_control(*broker, config.tls).finish(*lease, capacity_update(registered_revision)); }
            catch (...)
            {
                try { worker_detail::cleanup_control(*broker, config.tls).finish(*lease); } catch (...) {}
            }
        }
    }

    pool::WorkerOffer offer() const
    {
        pool::WorkerOffer value; value.instance = instance; value.compatibility = worker_compatibility();
        value.codec_compatibility = worker_compatibility(pool::Kind::Pack); value.producer_build = worker_producer_build();
        value.endpoint = config.advertise.port ? config.advertise : listener->endpoint();
        if (!config.server_name.empty()) value.endpoint.server_name = config.server_name;
        { std::lock_guard lock(advertisement_mutex);
          value.capacity = advertised_capacity.capacity; value.capacity_revision = advertised_capacity.revision;
          if (advertised_capacity.generation_profiles) value.generation_profiles = *advertised_capacity.generation_profiles; }
        value.capabilities = pool::capability(pool::Kind::Pack) | pool::capability(pool::Kind::GenerateCpu);
        value.generation_threads = std::min(config.generation_threads, value.capacity.cpu_threads);
        value.codec_limits = {config.max_window_chunks, config.max_task_bytes};
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        for (auto const& [k, profile] : gpu_profiles)
            if (profile.tier == "pool") value.independent_k_mask |= 1u << k;
#endif
        if (gpu_enabled) value.capabilities |= pool::capability(pool::Kind::GenerateGpu);
        { std::lock_guard lock(admission_mutex); value.admission = admission; }
        value.min_k = 18; value.max_k = 28; value.hardware = hardware(); return value;
    }
    void start()
    {
        require(!started && !stopping.load(), "worker service already started or stopped");
        require(!worker_compatibility().empty(), "worker build has no compute compatibility identity");
        if (!config.identity_reservation) config.identity_reservation = WorkerService::reserve_identity(config.tls.cert_file);
        require(config.identity_reservation->identity == pool::certificate_identity(config.tls.cert_file),
            "compute identity reservation belongs to another certificate");
        auto listener_tls = config.tls; listener_tls.generation_timing = true;
        listener = std::make_unique<pool::Listener>(config.listen, listener_tls, config.capacity.slots);
        try { refresh_brokers(); }
        catch (...) { /* A local executor stays usable while the broker restarts. */ }
        started = true;
        heartbeat_thread = std::thread([this]
        {
            std::unique_lock lock(heartbeat_mutex);
            while (!stopping.load())
            {
                heartbeat_ready.wait_for(lock, std::chrono::milliseconds(100));
                if (stopping.load()) break;
                lock.unlock();
                try { refresh_brokers(); }
                catch (...) { /* Retry the current capacity on the next heartbeat. */ }
                lock.lock();
            }
        });
        accept_threads.reserve(config.capacity.slots);
        for (unsigned i = 0; i < config.capacity.slots; ++i)
            accept_threads.emplace_back([this]
            {
                for (;;)
                {
                    PendingWork work;
                    { std::unique_lock lock(dispatch_mutex);
                      dispatch_ready.wait(lock, [&] { return stopping.load() || !pending_work.empty(); });
                      if (stopping.load()) break;
                      work = std::move(pending_work.front()); pending_work.pop_front(); }
                    handle(std::move(work.channel), std::move(work.frame));
                }
            });
        for (unsigned i = 0; i < 2; ++i) dispatch_threads.emplace_back([this]
        {
            while (!stopping.load()) try
            {
                auto channel = listener->accept(); if (!channel) continue;
                auto frame = channel.receive();
                if (frame.type == pool::Message::Control) { handle_control(channel, frame); continue; }
                require(frame.type == pool::Message::Work, "worker expected leased task or control request");
                bool full = false;
                {
                    std::lock_guard lock(dispatch_mutex);
                    full = pending_work.size() >= config.capacity.slots;
                    if (!full) pending_work.push_back({std::move(channel), std::move(frame)});
                }
                if (full)
                {
                    // Work can be followed immediately by codec data. Record a
                    // verified refusal before closing a potentially unread upload.
                    try { pool::Reader input(frame.payload); report_refusal(pool::read_lease(input), channel.peer_identity(), Clock::time_point::max()); }
                    catch (...) { /* Malformed or stale requests cannot release a grant. */ }
                    pool::Writer out(config.tls.max_frame_bytes); out.u8(uint8_t(WorkerError::Transport)); out.string("worker admission queue is full");
                    channel.send(pool::Message::Error, out.data()); continue;
                }
                dispatch_ready.notify_one();
            }
            catch (...) { if (stopping.load()) break; }
        });
    }

    void stop()
    {
        stopping.store(true); heartbeat_ready.notify_all(); jobs_ready.notify_all(); dispatch_ready.notify_all();
        for (auto& thread : dispatch_threads) if (thread.joinable()) thread.join();
        dispatch_threads.clear();
        { std::lock_guard lock(dispatch_mutex); pending_work.clear(); }
        for (auto& thread : accept_threads) if (thread.joinable()) thread.join();
        accept_threads.clear();
        if (heartbeat_thread.joinable()) heartbeat_thread.join();
        if (gpu_thread.joinable()) gpu_thread.join();
        if (started)
        {
            for (auto const& broker : brokers) try
            { worker_detail::cleanup_control(broker->client, config.tls).unregister_worker(instance); }
            catch (...) {}
            started = false;
        }
        listener.reset(); config.identity_reservation.reset();
    }
};

WorkerService::WorkerService(WorkerConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
WorkerService::~WorkerService() = default;
void WorkerService::start() { impl_->start(); }
void WorkerService::stop() { impl_->stop(); }
pool::Endpoint WorkerService::endpoint() const { return impl_->listener ? impl_->listener->endpoint() : impl_->config.listen; }
bool WorkerService::owns(pool::Lease const& lease) const
{ return lease.worker_instance == impl_->instance && lease.task.compatibility == worker_compatibility(lease.task.kind); }
WorkerStatus WorkerService::status() const { return impl_->status(); }
WorkerStatus WorkerService::control(WorkerControl operation) { return impl_->control(operation); }
WorkerStatus WorkerService::remote_status(pool::Endpoint const& endpoint, pool::TlsOptions const& tls)
{
    auto channel = pool::Channel::connect(endpoint, tls); pool::Writer request(tls.max_frame_bytes);
    request.u32(pool::protocol_version); request.u8(0); channel.send(pool::Message::Control, request.data());
    auto frame = channel.receive(); if (frame.type == pool::Message::Error) throw_worker_error(frame.payload);
    require(frame.type == pool::Message::Reply, "invalid worker control response"); pool::Reader in(frame.payload); auto value = read_worker_status(in);
    require(value.identity == channel.peer_identity() && value.node_id == channel.peer_node_id(), "worker status identity differs from authenticated peer");
    return value;
}
WorkerStatus WorkerService::remote_control(pool::Endpoint const& endpoint, pool::TlsOptions const& tls, WorkerControl operation)
{
    require(operation >= WorkerControl::Pause && operation <= WorkerControl::Resume, "invalid worker control operation");
    auto channel = pool::Channel::connect(endpoint, tls); pool::Writer request(tls.max_frame_bytes);
    request.u32(pool::protocol_version); request.u8(uint8_t(operation)); channel.send(pool::Message::Control, request.data());
    auto frame = channel.receive(); if (frame.type == pool::Message::Error) throw_worker_error(frame.payload);
    require(frame.type == pool::Message::Reply, "invalid worker control response"); pool::Reader in(frame.payload); auto value = read_worker_status(in);
    require(value.identity == channel.peer_identity() && value.node_id == channel.peer_node_id(), "worker status identity differs from authenticated peer");
    return value;
}
unsigned WorkerService::generate_assigned(pool::Lease const& lease,
    std::span<std::array<uint8_t, 32> const> ids, WorkerPackedSink const& packed,
    WorkerContributionSink const& contributions, WorkerContributionWindowSink const& window,
    WorkerFreshManifest const& manifest, WorkerCancellation cancel)
{
    auto const operation = packed ? Operation::Packed : manifest ? Operation::Fresh : Operation::Contributions;
    require(owns(lease) && lease.task.kind != pool::Kind::Pack && lease.task.work_units == ids.size() &&
        lease.task.generation_output == generation_output(operation),
        "local generation lease targets another task or executor");
    cancel = impl_->cancellation(cancel);
    uint64_t registered_revision = 0;
    auto control = impl_->control_for(lease, &registered_revision);
    auto active = control.with_deadline(cancel.deadline);
    std::optional<Impl::Acceptance> accepted;
    impl_->begin_assigned(accepted, control, lease, pool::certificate_identity(impl_->config.tls.cert_file), cancel);
    std::exception_ptr renewal_failure;
    try
    {
        std::atomic<bool> cancelled{false};
        WorkerCancellation running{&cancelled, cancel.deadline, &impl_->stopping, cancel.stop};
        Renewal renewal(active, lease, cancelled, renewal_failure, [cancel] { cancel.check(); }, impl_->config.tls.timeout_ms);
        auto checked_packed = [&](PackedFragmentTile const& tile, uint64_t total)
        { cancel.check(); running.check(); packed(tile, total); };
        auto checked_contribution = [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
        { cancel.check(); contributions(ordinal, bytes, members); };
        auto deliver = contributions ? individual_contributions(checked_contribution, running) : ContributionBlockSink{};
        // The owner budgets the same borrowed chunk descriptors as for TLS.
        std::vector<std::span<uint8_t const>> chunks;
        if (manifest && window)
        {
            chunks.resize(fresh_group_window_chunks);
            deliver = [&](uint64_t first, std::span<uint64_t const> sizes, std::span<uint8_t const> bytes, uint16_t members)
            {
                cancel.check();
                require(!sizes.empty() && sizes.size() <= chunks.size(), "fresh group receive window exceeds admission");
                {
                    CpuWork cpu(impl_->offered_cpu, impl_->scheduler.get(), running);
                    for (size_t i = 0; i < sizes.size(); ++i)
                    {
                        auto const size = sizes[i];
                        require(size && size <= bytes.size(), "invalid local contribution block size");
                        chunks[i] = bytes.first(size_t(size)); bytes = bytes.subspan(size_t(size));
                    }
                    require(bytes.empty(), "invalid local contribution block size");
                }
                cancel.check(); running.check();
                window(first, std::span(chunks).first(sizes.size()), members);
            };
        }
        auto checked_manifest = [&](uint64_t payload, uint64_t largest, std::array<uint8_t, 32> const& digest)
        {
            cancel.check(); running.check(); manifest(payload, largest, digest);
            cancel.check(); running.check();
        };
        impl_->generation(lease.task.k, lease.task.strength, ids,
            packed ? WorkerPackedSink(checked_packed) : WorkerPackedSink{}, deliver,
            running, lease.task.kind == pool::Kind::GenerateGpu, true, &lease.task.resources, {},
            manifest ? WorkerFreshManifest(checked_manifest) : WorkerFreshManifest{}, lease.task.independent_members);
        cancel.check(); running.check();
    }
    catch (std::exception const& error)
    {
        auto failure = std::current_exception();
        try
        {
            auto update = impl_->capacity_update(registered_revision);
            auto cleanup = worker_detail::cleanup_control(control, impl_->config.tls);
            if (dynamic_cast<pool::UnsupportedResources const*>(&error))
                cleanup.reject(lease, pool::FailureCode::UnsupportedResources, update);
            else cleanup.finish(lease, update);
        }
        catch (...) { /* The next heartbeat publishes the current capacity. */ }
        std::rethrow_exception(task_failure(failure, renewal_failure));
    }
    try { worker_detail::cleanup_control(control, impl_->config.tls).finish(lease, impl_->capacity_update(registered_revision)); }
    catch (...) { /* The next heartbeat publishes the current capacity. */ }
    return impl_->generation_slots(lease.task);
}
void WorkerService::encode_assigned(pool::Lease const& lease, std::span<uint8_t const> payload,
    uint64_t maximum, std::function<void(size_t, std::span<uint8_t const>)> const& sink, WorkerCancellation cancel)
{
    require(owns(lease) && lease.task.kind == pool::Kind::Pack &&
        lease.task.chunk_count && lease.task.chunk_count <= impl_->config.max_window_chunks &&
        payload.size() == lease.task.work_units && maximum == lease.task.max_output_bytes &&
        payload.size() <= impl_->config.max_task_bytes && maximum <= impl_->config.max_task_bytes &&
        add(multiply(payload.size(), 20), ChunkCodec::estimate_work(lease.task.k, 0, 0)) <= lease.task.resources.host_bytes,
        "local packing lease targets another task or exceeds its resources");
    cancel = impl_->cancellation(cancel); auto control = impl_->control_for(lease);
    auto active = control.with_deadline(cancel.deadline);
    std::optional<Impl::Acceptance> accepted;
    impl_->begin_assigned(accepted, control, lease, pool::certificate_identity(impl_->config.tls.cert_file), cancel);
    std::exception_ptr renewal_failure;
    try
    {
        std::atomic<bool> cancelled{false};
        WorkerCancellation running{&cancelled, cancel.deadline, &impl_->stopping, cancel.stop};
        Renewal renewal(active, lease, cancelled, renewal_failure, [cancel] { cancel.check(); }, impl_->config.tls.timeout_ms);
        impl_->encode(lease.task.k, payload, maximum, [&](size_t index, std::span<uint8_t const> bytes)
        { cancel.check(); running.check(); sink(index, bytes); }, running, nullptr, lease.task.chunk_count);
        cancel.check(); running.check();
    }
    catch (...)
    { auto failure = std::current_exception(); try { worker_detail::cleanup_control(control, impl_->config.tls).finish(lease); } catch (...) {}
      std::rethrow_exception(task_failure(failure, renewal_failure)); }
    try { worker_detail::cleanup_control(control, impl_->config.tls).finish(lease); } catch (...) {}
}
void WorkerService::generate_local(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& id,
    WorkerPackedSink const& sink, WorkerCancellation cancel, pool::Kind kind,
    std::function<void(WorkerTaskAdmission const&)> const& admit)
{
    generate_packed_local(k, strength, {&id, 1}, sink, cancel, kind, admit);
}
WorkerLocalOffer WorkerService::local_generation_offer(uint8_t k, uint8_t strength, unsigned members) const
{
    WorkerLocalOffer result;
    result.capacity = impl_->current_capacity(&result.standing_host_bytes);
    if (!members || members > impl_->config.max_batch_members) return result;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    if (impl_->gpu_enabled && !impl_->gpu_profiles.contains(k)) return result;
#endif
    result.demand = {std::min(impl_->config.generation_threads, result.capacity.cpu_threads), 1,
        impl_->generation_host_bytes(k, strength, members, impl_->gpu_enabled, result.standing_host_bytes - impl_->service_baseline),
        impl_->gpu_enabled ? impl_->config.capacity.gpu_bytes : 0};
    result.member_host_bytes = add(multiply(fragment_count_bound(k, strength), sizeof(uint64_t)), boundary_bytes(k));
    result.max_batch_members = impl_->config.max_batch_members;
    return result;
}
void WorkerService::generate_packed_local(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
    WorkerPackedSink const& sink, WorkerCancellation cancel, pool::Kind kind,
    std::function<void(WorkerTaskAdmission const&)> const& admit, pool::Resources const* grant)
{
    require(kind != pool::Kind::GenerateGpu || impl_->gpu_enabled, "local GPU generation was not offered");
    Impl::Acceptance accepted(*impl_);
    impl_->generation(k, strength, ids, sink, {}, cancel, kind != pool::Kind::GenerateCpu && impl_->gpu_enabled, true, grant, admit);
}
void WorkerService::contributions_local(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
    WorkerContributionSink const& sink, WorkerCancellation cancel, pool::Kind kind,
    std::function<void(WorkerTaskAdmission const&)> const& admit, pool::Resources const* grant)
{
    require(kind != pool::Kind::GenerateGpu || impl_->gpu_enabled, "local GPU generation was not offered");
    Impl::Acceptance accepted(*impl_);
    cancel = impl_->cancellation(cancel);
    impl_->generation(k, strength, ids, {}, individual_contributions(sink, cancel), cancel,
        kind != pool::Kind::GenerateCpu && impl_->gpu_enabled, true, grant, admit);
}

namespace worker_detail
{
// Delivery hints are bounded, workload-specific observations of each worker
// route. They do not replace the broker's resource or per-task admission.
class GenerationPlanning
{
public:
    static constexpr size_t maximum = 64;
    struct Plan { unsigned batch = 1, concurrency = 2; };
    explicit GenerationPlanning(unsigned timeout_ms, WorkerGenerationCalibration calibration = {})
        : lifetime_(timeout_ms), calibration_(std::move(calibration)) {}
    uint64_t selections() const { return selection_; }
    void capacity(uint8_t k, uint8_t strength, std::shared_ptr<pool::GenerationRoster const> routes,
                  uint64_t before, Clock::time_point now = Clock::now())
    {
        if (calibration_ && (roster_k_ != k || roster_strength_ != strength ||
            (roster_ != routes && (!roster_ || !routes || *roster_ != *routes))))
            for (auto& item : std::span(histories_).first(count_))
            { item.prior = {}; item.independent_prior = {}; }
        roster_ = std::move(routes); roster_k_ = k; roster_strength_ = strength;
        if (!roster_) return;
        // A changed incarnation or data endpoint fences old observations,
        // even if it replaces exactly as many slots. Leases selected during
        // the network query may already know a newer route.
        auto end = std::remove_if(peers_.begin(), peers_.begin() + peer_count_, [&](auto const& peer)
        {
            auto route = offered_route(peer.peer.worker);
            return peer.last <= before && route && !same_route(peer.peer, *route);
        });
        peer_count_ = size_t(end - peers_.begin()); retire(now);
        for (auto& item : std::span(histories_).first(count_)) apply_offer(item);
    }
    uint64_t selected(WorkerTaskObservation const& value, pool::Endpoint const& route, Clock::time_point now = Clock::now())
    {
        if (!valid(value, route)) return 0;
        require(selection_ != UINT64_MAX, "compute planning selection sequence exhausted");
        auto const selection = ++selection_; retire(now);
        auto const peer = key(value, route);
        auto at = current(peer.worker);
        if (at == peers_.begin() + peer_count_)
        {
            if (peer_count_ == maximum)
                at = std::min_element(peers_.begin(), peers_.end(),
                    [](ActivePeer const& a, ActivePeer const& b) { return a.last < b.last; });
            else at = peers_.begin() + peer_count_++;
            *at = ActivePeer{};
        }
        // Only selection can activate an incarnation or route. Late success or
        // failure from an older lease must not resurrect its previous path.
        if (at->peer != peer)
        {
            erase([&](History const& item) { return item.peer.worker == peer.worker; });
            at->peer = peer; at->first = selection; at->in_flight = 0;
        }
        at->last = selection; at->seen = now; at->sequence = sequence_; ++at->in_flight;
        at->delivery_ns = 0; at->packed_since = {};
        if (at->in_flight == 1 && value.work_units <= pool::max_generation_batch_members)
        {
            auto const prior = find(value, peer);
            if (prior != histories_.begin() + count_)
            {
                auto const& rate = value.independent_members ? prior->pipeline_rate() : value.generation_output == GenerationOutput::Packed ?
                    prior->packed_rate(unsigned(value.work_units), 1) : prior->rates[value.work_units];
                at->delivery_ns = rate.service * value.work_units;
            }
        }
        return selection;
    }
    void receiving_packed(pool::Lease const& lease, uint64_t selection, Clock::time_point work_sent)
    {
        if (!selection || lease.task.work_units != 1) return;
        auto at = current(text<64>(lease.worker_identity));
        if (at == peers_.begin() + peer_count_ || at->last != selection || at->in_flight != 1) return;
        at->packed_since = work_sent; at->packed_k = lease.task.k; at->packed_strength = lease.task.strength;
        at->packed_kind = lease.task.kind;
    }
    void released(std::string const& worker, uint64_t selection)
    {
        if (!selection) return;
        auto at = current(text<64>(worker));
        if (at != peers_.begin() + peer_count_ && selection >= at->first && selection <= at->last && at->in_flight)
            --at->in_flight;
    }
    void observed(WorkerTaskObservation const& value, pool::Endpoint const& route, uint64_t selection,
                  Clock::time_point now = Clock::now())
    {
        if (!valid(value, route)) return;
        retire(now);
        auto const peer = key(value, route); auto active = current(peer.worker);
        if (!accepts(active, peer, selection)) return;
        if (sequence_ == UINT64_MAX) { count_ = peer_count_ = 0; sequence_ = 0; return; }
        active->sequence = ++sequence_; active->seen = now; retire(now);
        if (!value.execution_ns || !value.transfer_ns) return;
        active = current(peer.worker);
        // The remote endpoint can stay the same while this owner changes its
        // interface. Start fresh timings on the newly observed data path.
        if (selection < active->ingress_selected && active->ingress != value.ingress_interface) return;
        active->ingress_selected = std::max(selection, active->ingress_selected); active->ingress = value.ingress_interface;
        erase([&](History const& item) { return item.peer == peer && item.ingress != value.ingress_interface; });
        auto at = find(value, peer);
        if (at == histories_.begin() + count_)
        {
            if (count_ == maximum)
                at = std::min_element(histories_.begin(), histories_.end(),
                    [](History const& a, History const& b) { return a.sequence < b.sequence; });
            else at = histories_.begin() + count_++;
            *at = History{}; at->peer = peer; at->kind = value.kind; at->k = value.k; at->strength = value.strength;
            at->output = value.generation_output; at->ingress = value.ingress_interface;
        }
        at->seen = now; at->sequence = sequence_;
        apply_offer(*at);
        bool const bounded = value.work_units <= pool::max_generation_batch_members;
        at->delivered = bounded;
        bool const contributions = value.generation_output == GenerationOutput::Contributions;
        if (bounded && contributions && (value.work_units == 1 || value.independent_members))
            at->delivered_contribution = true;
        if (bounded && value.independent_members)
            at->independent_width = std::max(at->independent_width, unsigned(value.work_units));
        bool const timed = bounded && value.generation.worker_ns && !value.local_direct;
        uint64_t production = value.execution_ns, delivery = value.transfer_ns;
        if (timed || (bounded && value.local_direct))
        {
            at->slots = std::clamp(value.worker_slots, 1u, 64u);
            apply_offer(*at);
            if (timed && calibration_) calibrate(*at, value, now);
            // The independent pipeline sends the same contribution as a
            // singleton. Use singleton service until the pipeline has its own
            // baseline, while keeping joint encoding and exact singletons separate.
            if (value.independent_members)
            {
                if (!at->independent.samples) at->independent = at->rates[1];
                at->independent.add(value); return;
            }
            auto& rates = at->rates[value.generation_output == GenerationOutput::Packed ? 0 : value.work_units];
            rates.add(value);
            if (value.generation_output == GenerationOutput::Packed) at->rates[value.work_units].add(value);
            if (value.kind == pool::Kind::GenerateGpu && contributions && value.work_units > 1)
            {
                at->deferred_joint = false;
                for (auto& item : std::span(histories_).first(count_))
                    if (item.matches(value.k, value.strength, value.kind, value.generation_output) && item.joint_trial == value.work_units)
                        item.joint_trial = 0;
            }
            if (value.kind == pool::Kind::GenerateGpu && contributions && value.work_units == 1 &&
                at->independent.samples && at->independent.samples < 3) at->independent.add(value);
            if (timed)
            {
                production = value.generation.worker_ns;
                delivery = std::max({value.generation.send_ns, value.generation.frame_wait_ns,
                    value.generation.frame_read_ns, value.generation.receiver_ns});
            }
            if (timed && contributions && rates.samples == 3)
            {
                // Try one larger coencoded batch after a sustained baseline.
                // Once both have observations, prefer lower time per member.
                unsigned best = unsigned(value.work_units);
                for (unsigned n = 1; n < at->rates.size(); ++n)
                    if (at->rates[n].samples == 3 && at->rates[n].service < at->rates[best].service) best = n;
                at->plan.batch = best < pool::max_generation_batch_members && at->rates[best + 1].samples < 3 ? best + 1 : best;
            }
        }
        int const direction = production > delivery ? 1 : production < delivery ? -1 : 0;
        at->votes = direction && direction == at->direction ? at->votes + 1 : unsigned(direction != 0);
        at->direction = direction;
        // A single noisy completion must not change every following request.
        if (at->votes < 3) return;
        at->votes = 0;
        if (direction > 0)
        { if (!timed) at->plan.batch = std::min(unsigned(pool::max_generation_batch_members), at->plan.batch + 1);
          at->plan.concurrency = std::min(64u, at->plan.concurrency + 1); }
        else
        { if (!timed) at->plan.batch = std::max(1u, at->plan.batch / 2);
          at->plan.concurrency = std::max(1u, at->plan.concurrency / 2); }
    }
    void failed(WorkerTaskObservation const& value, pool::Endpoint const& route, uint64_t selection,
                Clock::time_point now = Clock::now())
    {
        if (!valid(value, route)) return;
        retire(now); auto const peer = key(value, route);
        if (!accepts(current(peer.worker), peer, selection)) return;
        auto at = find(value, peer);
        if (at == histories_.begin() + count_) return;
        at->plan.batch = std::max(1u, at->plan.batch / 2); at->plan.concurrency = std::max(1u, at->plan.concurrency / 2);
        at->votes = 0; at->direction = 0;
        at->rates = {}; at->independent = {}; at->prior = {}; at->independent_prior = {};
        at->delivered = at->delivered_contribution = false; at->joint_trial = 0; at->deferred_joint = false;
        at->independent_width = 0;
    }
    WorkerGenerationEstimate estimate(uint8_t k, uint8_t strength, pool::Kind kind, GenerationOutput output,
                          unsigned active, unsigned batch, Clock::time_point now = Clock::now(), uint64_t horizon_members = 0)
    {
        if (!active || !batch) return {};
        retire(now);
        std::array<History const*, maximum> peers{};
        std::array<Route, maximum> routes{};
        std::array<unsigned, maximum> worker_slots{};
        size_t measured = 0, workers = 0;
        unsigned prior_samples = 0;
        uint64_t oldest_utc = 0;
        unsigned const samples = horizon_members ? 1 : 3;
        bool measured_packed_batch = output == GenerationOutput::Packed && batch <= pool::max_generation_batch_members;
        for (size_t i = 0; i < count_; ++i)
        {
            auto const& item = histories_[i];
            if (!item.matches(k, strength, kind, output)) continue;
            auto rate = item.estimate_rate(batch, samples);
            if (rate.samples < samples && batch <= pool::max_generation_batch_members)
            {
                // Weak calibration accelerates only this exact, freshly
                // confirmed workload's early forecast. Never seed delivery,
                // admission, exploration or the live measurement buckets.
                History::Prior const* selected = nullptr;
                for (auto prior : {&item.prior[batch], &item.independent_prior[batch]})
                    if (prior->samples && prior->until > now && (rate.samples < samples || prior->service < rate.service))
                    { rate = *prior; selected = prior; }
                if (selected)
                {
                    prior_samples = prior_samples ? std::min(prior_samples, selected->dated_samples) : selected->dated_samples;
                    oldest_utc = oldest_utc ? std::min(oldest_utc, selected->oldest_utc) : selected->oldest_utc;
                }
            }
            if (rate.samples < samples) continue;
            if (measured_packed_batch && item.rates[batch].samples < samples) measured_packed_batch = false;
            // GenerateAny can use either measured backend of one worker.
            // Both consume the same advertised slots.
            size_t worker = 0; while (worker < workers && peers[worker]->peer.worker != item.peer.worker) ++worker;
            if (worker == workers) peers[workers++] = &item;
            worker_slots[worker] = std::max(worker_slots[worker], item.slots);
            routes[measured++] = {1 / rate.service, rate.link / rate.service, item.slots, worker, item.ingress};
        }
        if (!measured) return {};
        auto const fleet = std::span(routes).first(measured);
        unsigned slots = 0; for (auto count : std::span(worker_slots).first(workers)) slots += count;
        unsigned low = 1, high = std::min(slots, active);
        auto const fastest = throughput(fleet, worker_slots, high);
        // A scratch layout serves a bounded amount of ready work. Reserve
        // another receiver when it can deliver another whole member in that
        // interval. Ordinary throughput forecasts use the full sustained rate.
        while (low < high)
        {
            auto const middle = low + (high - low) / 2;
            auto const rate = throughput(fleet, worker_slots, middle);
            bool const enough = horizon_members ? (fastest - rate) / fastest * horizon_members < 1 :
                rate >= fastest * (1 - 8 * std::numeric_limits<long double>::epsilon());
            if (enough) high = middle;
            else low = middle + 1;
        }
        return {1 / (horizon_members ? throughput(fleet, worker_slots, low) : fastest), low, slots,
            prior_samples != 0, prior_samples, oldest_utc, measured_packed_batch};
    }
    bool probe_contribution_route(uint8_t k, uint8_t strength, Clock::time_point now = Clock::now())
    {
        retire(now);
        auto delivered = [&](auto const& item)
        {
            return item.matches(k, strength, pool::Kind::GenerateGpu, GenerationOutput::Contributions) && item.delivered_contribution;
        };
        if (std::none_of(histories_.begin(), histories_.begin() + count_, delivered)) return true;
        if (!roster_ || roster_k_ != k || roster_strength_ != strength) return false;
        // A late GPU can learn from one useful member while measured GPUs
        // continue with their normal batches. Once a lease is on its way,
        // additional destinations do not need to keep probing that GPU.
        return std::any_of(roster_->routes.begin(), roster_->routes.end(), [&](auto const& route)
        {
            if (!route.gpu_slots) return false;
            auto const worker = text<64>(route.identity);
            auto active = current(worker);
            if (active != peers_.begin() + peer_count_ && active->in_flight) return false;
            return std::none_of(histories_.begin(), histories_.begin() + count_, [&](auto const& item)
            { return item.peer.worker == worker && delivered(item); });
        });
    }
    bool needs_first_member(WorkerTaskObservation const& task, Clock::time_point now = Clock::now())
    {
        if (task.work_units <= 1 || !roster_ || roster_k_ != task.k || roster_strength_ != task.strength) return false;
        retire(now);
        auto const worker = text<64>(task.worker_identity), instance = text<64>(task.worker_instance);
        bool other = false;
        for (auto const& item : std::span(histories_).first(count_))
        {
            if (!item.offered || item.k != task.k || item.strength != task.strength || !item.delivered) continue;
            // A fresh manifest is indivisible. Evidence that this worker has
            // delivered a member lets the owner try its first whole group.
            if (task.generation_output != GenerationOutput::Fresh && (item.output != task.generation_output ||
                (item.output == GenerationOutput::Contributions && !item.delivered_contribution))) continue;
            if (item.peer.worker == worker && item.peer.instance == instance && item.kind == task.kind) return false;
            if (item.peer.worker != worker) other = true;
        }
        return other;
    }
    unsigned joint_contribution_members(uint8_t k, uint8_t strength, unsigned maximum_members,
                                        Clock::time_point now = Clock::now(), unsigned destination_batch = 0)
    {
        maximum_members = std::min(maximum_members, unsigned(pool::max_generation_batch_members));
        if (maximum_members < 2) return 0;
        retire(now);
        std::pair<bool, long double> pipeline{true, std::numeric_limits<long double>::max()}, joint = pipeline;
        History* pipeline_worker = nullptr, *encoder = nullptr;
        unsigned joint_members = 0;
        bool measured_pipeline = false, trial_pending = false;
        long double deferred_single = 0;
        long double queued_pipeline = std::numeric_limits<long double>::max();
        std::array<bool, pool::max_generation_batch_members + 1> sampled{};
        for (auto& item : std::span(histories_).first(count_))
        {
            if (!item.matches(k, strength, pool::Kind::GenerateGpu, GenerationOutput::Contributions)) continue;
            auto const active = current(item.peer.worker);
            bool const busy = active->in_flight >= item.slots;
            if (!busy && item.deferred_joint) deferred_single = std::max(deferred_single, item.independent_cost(1));
            auto const& rate = item.pipeline_rate();
            if (rate.samples && std::pair(busy, rate.service) < pipeline)
            { pipeline = {busy, rate.service}; pipeline_worker = &item; }
            if (auto const wait = active->remaining_ns(now); busy && rate.samples == 3 && wait > 0)
                queued_pipeline = std::min(queued_pipeline, rate.service + wait / maximum_members);
            measured_pipeline |= item.independent.samples == 3;
            trial_pending |= item.joint_trial && now < item.trial_until;
            for (unsigned n = 2; n <= maximum_members; ++n) sampled[n] |= item.rates[n].samples != 0;
            auto const members = item.best_joint(maximum_members);
            if (!members) continue;
            auto const rank = std::pair(busy, item.rates[members].service);
            if (rank < joint) { joint = rank; joint_members = members; encoder = &item; }
        }
        if (!measured_pipeline) return 0;
        // Probe a pair first unless independent destinations can each feed
        // a whole pipeline. Destinations share each outstanding trial.
        if (!trial_pending && (!destination_batch || destination_batch == maximum_members))
        {
            auto unmeasured = [&](unsigned n) { return !sampled[n] && deferred_single <= pipeline.second * n; };
            unsigned trial = 0;
            if (!destination_batch && !pipeline.first && unmeasured(2)) trial = 2;
            else if (!pipeline.first && unmeasured(maximum_members)) trial = maximum_members;
            else if (!destination_batch && encoder && !joint.first && encoder->plan.batch > 1 && encoder->plan.batch < maximum_members &&
                unmeasured(encoder->plan.batch)) trial = encoder->plan.batch;
            if (trial)
            {
                pipeline_worker->joint_trial = trial; pipeline_worker->trial_until = now + lifetime_;
                return trial;
            }
        }
        // An independent request can still use the free GPU. Keep that
        // choice open when the busy pipeline can deliver the batch sooner.
        if (queued_pipeline < joint.second) return 0;
        return joint < pipeline ? joint_members : 0;
    }
    bool defer_queued_joint(uint8_t k, uint8_t strength, unsigned count, Clock::time_point now = Clock::now())
    {
        retire(now);
        bool trial = false, available = false;
        for (auto const& item : std::span(histories_).first(count_))
        {
            if (!item.matches(k, strength, pool::Kind::GenerateGpu, GenerationOutput::Contributions)) continue;
            trial |= item.joint_trial == count && now < item.trial_until;
            available |= item.independent.samples == 3 && current(item.peer.worker)->in_flight < item.slots;
        }
        // Keep the trial's existing expiry. Repeated unstarted probes would
        // delay every destination while the same joint encoder is occupied.
        return trial && available;
    }
    unsigned repartition_contributions(WorkerTaskObservation const& task, Clock::time_point now = Clock::now(),
                                       Clock::time_point other_work_until = {})
    {
        if (task.kind != pool::Kind::GenerateGpu || task.generation_output != GenerationOutput::Contributions || task.work_units < 2 ||
            task.work_units > pool::max_generation_batch_members || !valid(task, {})) return 0;
        retire(now);
        auto const active = current(text<64>(task.worker_identity));
        if (active == peers_.begin() + peer_count_ || active->peer.instance != text<64>(task.worker_instance)) return 0;
        auto selected = find(task, active->peer);
        // Offers can change between sizing and assignment. Split a wide
        // first lease before sending plot IDs if a priced GPU already serves
        // this shape. The two new requests keep their original deadline.
        if (needs_first_member(task, now)) return 1;
        if (selected == histories_.begin() + count_) return 0;
        unsigned const count = unsigned(task.work_units);
        auto const& joint = selected->rates[count];
        auto const single = selected->independent_cost(1), pipeline = selected->pipeline_rate().service;
        // Price a first joint trial as one member. A route first observed
        // in an independent batch already has a per-member delivery cost.
        long double best = task.independent_members ? pipeline * count : joint.samples ? joint.service * count : single;
        if (!best) return 0;
        if (task.independent_members && now < other_work_until &&
            best <= std::chrono::duration<long double, std::nano>(other_work_until - now).count()) return 0;
        unsigned first = 0;
        for (auto const& other : std::span(histories_).first(count_))
        {
            if (other.peer.worker == selected->peer.worker || !other.matches(task.k, task.strength, task.kind, task.generation_output) ||
                other.independent.samples < 3) continue;
            auto const peer = current(other.peer.worker);
            bool const busy = peer->in_flight >= other.slots;
            auto const wait = busy ? peer->remaining_ns(now) : 0;
            if (busy && !wait) continue;
            if (!task.independent_members)
            {
                if (wait + other.independent.service * count < best) { first = count; break; }
                continue;
            }
            // A wide independent request can also land on a slower GPU.
            // Give the two routes disjoint work when that finishes earlier.
            if (other.independent.service >= pipeline) continue;
            for (unsigned n = 1; n < count; ++n)
            {
                auto const finish = std::max(selected->independent_cost(n), wait + other.independent_cost(count - n));
                if (finish < best) { best = finish; first = n; }
            }
        }
        if (first && !task.independent_members)
        {
            if (!joint.samples) selected->deferred_joint = true;
            for (auto& item : std::span(histories_).first(count_))
                if (item.matches(task.k, task.strength, task.kind, task.generation_output) && item.joint_trial == task.work_units)
                    item.joint_trial = 0;
        }
        return first;
    }
    bool packed_before_singletons(uint8_t k, uint8_t strength, pool::Kind kind, unsigned width,
                                  WorkerPackedRoute& batch_route, Clock::time_point now = Clock::now())
    {
        if (width < 2 || width > pool::max_generation_batch_members || !roster_ || !roster_->complete ||
            roster_k_ != k || roster_strength_ != strength || roster_->routes.size() < 2) return false;
        retire(now);
        long double finish = std::numeric_limits<long double>::max();
        History const* batch = nullptr;
        for (auto const& item : std::span(histories_).first(count_))
        {
            if (!item.matches(k, strength, kind, GenerationOutput::Packed) || item.slots != 1 || !item.rates[width].samples ||
                (item.rates[1].samples && item.rates[width].service >= item.rates[1].service)) continue;
            auto const active = current(item.peer.worker);
            auto const wait = active->remaining_ns(now);
            if (active->in_flight && !wait) continue;
            auto const end = wait + width * item.rates[width].service;
            if (end < finish) { finish = end; batch = &item; }
        }
        if (!batch) return false;
        unsigned offered = 0, measured = 1;
        for (auto const& route : roster_->routes)
        {
            offered += bool(route.cpu_slots && kind != pool::Kind::GenerateGpu);
            offered += bool(route.gpu_slots && kind != pool::Kind::GenerateCpu);
        }
        // Even if every other eligible backend were idle now, each would
        // deliver its first member after this measured whole batch.
        for (auto const& item : std::span(histories_).first(count_))
            if (&item != batch && item.matches(k, strength, kind, GenerationOutput::Packed))
            {
                if (!item.rates[1].samples || item.rates[1].service <= finish) return false;
                ++measured;
            }
        if (measured != offered) return false;
        batch_route = {*offered_route(batch->peer.worker), batch->kind};
        return true;
    }
    unsigned dispatch_width(uint8_t k, uint8_t strength, pool::Kind kind, GenerationOutput output, unsigned ready, unsigned destinations,
                            unsigned offered, unsigned limit, unsigned fallback, Clock::time_point now = Clock::now(),
                            unsigned independent_width = 0, WorkerPackedRoute* batch_route = nullptr)
    {
        retire(now);
        bool const packed = output == GenerationOutput::Packed;
        auto rates = [packed, limit](History const& item) -> History::Rate const&
        {
            if (packed) return item.packed_rate(limit, 1);
            return item.pipeline_rate();
        };
        std::array<History const*, maximum> peers{};
        size_t count = 0;
        for (auto const& item : std::span(histories_).first(count_))
        {
            if (!item.matches(k, strength, kind, output) ||
                item.independent_width < independent_width || !rates(item).samples) continue;
            auto const end = peers.begin() + count;
            auto at = std::find_if(peers.begin(), end, [&](auto peer) { return peer->peer.worker == item.peer.worker; });
            if (at == end) peers[count++] = &item;
            else if (rates(item).service < rates(**at).service) *at = &item;
        }
        unsigned slots = 0;
        long double throughput = 0, fastest = 0, available = 0;
        long double next_rate = 0, next_finish = std::numeric_limits<long double>::max();
        bool known = true;
        for (auto peer : std::span(peers).first(count))
        {
            auto const& item = *peer;
            auto const& rate = rates(item);
            slots += item.slots; throughput += item.slots / rate.service;
            if (!fastest || rate.service < fastest) fastest = rate.service;
            auto const active = current(item.peer.worker);
            if (active->in_flight < item.slots)
            { if (!available || rate.service < available) available = rate.service; }
            else
            {
                auto const wait = active->remaining_ns(now);
                if (!wait) known = false;
                else if (auto const finish = wait + rate.service; finish < next_finish)
                { next_finish = finish; next_rate = rate.service; }
            }
        }
        if (packed && batch_route && count == 1 && slots < offered && roster_ && roster_->complete &&
            roster_k_ == k && roster_strength_ == strength)
        {
            auto const& measured = *peers.front();
            auto const* target = offered_route(measured.peer.worker);
            auto const width = std::min(ready, limit);
            auto const finish = (available ? available : next_rate) * width +
                (available || !next_rate ? 0 : next_finish - next_rate);
            if (target && same_route(measured.peer, *target) && finish > 0 && roster_->routes.size() == 1 + (offered - slots))
            {
                unsigned occupied = 0;
                for (auto const& route : roster_->routes)
                {
                    if (route.identity == target->identity) continue;
                    auto at = current(text<64>(route.identity));
                    if (at == peers_.begin() + peer_count_ || !same_route(at->peer, route) || at->in_flight != 1 ||
                        std::max(route.cpu_slots, route.gpu_slots) != 1 || at->packed_since == Clock::time_point{} ||
                        at->packed_k != k || at->packed_strength != strength ||
                        (kind != pool::Kind::GenerateAny && at->packed_kind != kind) ||
                        std::chrono::duration<long double, std::nano>(now - at->packed_since).count() <= finish) break;
                    ++occupied;
                }
                // Every other slot is delivering its first member. Its observed
                // service has already taken longer than this route's whole batch.
                // Recheck the actual assignment before any plot ID is sent.
                if (slots + occupied == offered) { *batch_route = {*target, measured.kind}; return width; }
            }
        }
        // Another packed destination can supply a singleton to the idle route.
        // Size this batch for the busy worker if it can deliver first.
        if (known && (!available || (packed && destinations > offered && next_finish < available)))
            available = next_rate;
        // Keep the caller's ordinary width until the route set and the
        // worker likely to accept more work have both been measured.
        if (slots != offered || !available) return fallback;
        // Give the next free worker its share of ready work, bounded by the
        // time the fastest route needs for one full lease.
        long double const horizon = std::min(std::min(ready, limit) * fastest,
            static_cast<long double>(ready) * std::max(1u, destinations) / throughput);
        return std::clamp(unsigned(std::ceil(horizon / available)), 1u, std::min(ready, limit));
    }
    Plan suggested(uint8_t k, uint8_t strength, pool::Kind kind, GenerationOutput output, Clock::time_point now = Clock::now())
    {
        retire(now);
        std::array<unsigned, maximum> batches{}, concurrency{};
        std::array<History const*, maximum> representatives{}; size_t count = 0;
        for (size_t i = 0; i < count_; ++i)
        {
            auto const& item = histories_[i];
            if (!item.matches(k, strength, kind, output)) continue;
            size_t at = 0;
            while (at < count && representatives[at]->peer.worker != item.peer.worker) ++at;
            if (at == count) { representatives[count] = &item; ++count; }
            // GenerateAny may have CPU and GPU samples for the same peer.
            // Count that worker once, using its less constrained backend.
            batches[at] = std::max(batches[at], item.plan.batch);
            concurrency[at] = std::max(concurrency[at], item.plan.concurrency);
        }
        if (!count) return {};
        // The worker is unknown before leasing. Use a representative active
        // route, so a slow minority or a departed worker cannot pin the fleet.
        std::sort(batches.begin(), batches.begin() + count); std::sort(concurrency.begin(), concurrency.begin() + count);
        return {batches[count / 2], concurrency[count / 2]};
    }
private:
    struct Route { long double rate, wire; unsigned slots; size_t worker; uint32_t ingress; };
    struct Load { long double rate = 0, wire = 0, lanes = 0; };
    // Mix two feasible allocations when the optimum time-shares a worker.
    // At a given price each maximizer bounds what is achievable within the
    // budget. Return the feasible mix if the planning bound is exhausted.
    template<class Allocate, class Value> static Load limited(long double budget, long double Load::*resource,
                                                              Allocate allocate, Value value)
    {
        auto over = allocate(0); if (over.*resource <= budget) return over;
        Load under, feasible;
        for (size_t iteration = 0; iteration < maximum; ++iteration)
        {
            auto const gap = over.*resource - under.*resource;
            auto const part = (budget - under.*resource) / gap;
            auto mix = [&](long double Load::*field) { return under.*field + part * (over.*field - under.*field); };
            feasible = {mix(&Load::rate), mix(&Load::wire), mix(&Load::lanes)};
            auto const price = (value(over) - value(under)) / gap;
            auto const next = allocate(price);
            if (next.*resource == budget) return next;
            auto const bound = value(next) + price * (budget - next.*resource);
            auto& side = next.*resource > budget ? over : under;
            if (bound <= value(feasible) * (1 + 4 * std::numeric_limits<long double>::epsilon()) ||
                (next.*resource == side.*resource && value(next) <= value(side))) break;
            side = next;
        }
        return feasible;
    }
    static long double throughput(std::span<Route const> routes, std::array<unsigned, maximum> const& worker_slots, unsigned limit)
    {
        std::array<uint32_t, maximum> ingress{}; size_t groups = 0;
        for (auto const& route : routes)
        {
            if (!route.wire && !route.ingress) continue;
            // An unresolved route might share any of the confirmed physical
            // adapters. Retain the shared-link bound in that case.
            if (!route.ingress) { groups = 0; break; }
            if (std::find(ingress.begin(), ingress.begin() + groups, route.ingress) == ingress.begin() + groups)
                ingress[groups++] = route.ingress;
        }
        auto on_link = [&](uint32_t link, long double receiver_price)
        {
            auto allocate = [&](long double wire_price)
            {
                std::array<std::pair<long double, size_t>, maximum> order;
                size_t size = 0;
                for (size_t i = 0; i < routes.size(); ++i)
                {
                    auto const& route = routes[i];
                    // Zero-wire local routes need only the global receiver
                    // budget. Include them once, with the first link.
                    if (groups > 1 && link != (route.ingress ? route.ingress : ingress[0])) continue;
                    order[size++] = {route.rate - receiver_price - wire_price * route.wire, i};
                }
                std::sort(order.begin(), order.begin() + size, std::greater<>());
                Load load; unsigned room = limit; auto available = worker_slots;
                for (auto const& [value, i] : std::span(order).first(size))
                {
                    if (!room || value <= 0) break;
                    auto const& route = routes[i]; auto& slots = available[route.worker];
                    auto const count = std::min({room, slots, route.slots}); room -= count; slots -= count;
                    load.rate += count * route.rate; load.wire += count * route.wire; load.lanes += count;
                }
                return load;
            };
            return limited(1, &Load::wire, allocate, [&](Load const& load) { return load.rate - receiver_price * load.lanes; });
        };
        if (groups < 2) return on_link(0, 0).rate;
        auto all_links = [&](long double receiver_price)
        {
            Load total;
            for (auto link : std::span(ingress).first(groups))
            { auto const load = on_link(link, receiver_price); total.rate += load.rate; total.lanes += load.lanes; }
            return total;
        };
        return limited(limit, &Load::lanes, all_links, [](Load const& load) { return load.rate; }).rate;
    }
    // Fixed protocol-sized keys keep the history entirely inside the client's
    // sizeof(Impl) memory reservation, including its replacement path.
    struct Peer
    {
        std::array<char, 64> worker{}, instance{};
        std::array<char, 253> host{}, server_name{};
        uint16_t port = 0;
        bool local = false;
        bool operator==(Peer const&) const = default;
    };
    struct History
    {
        struct Rate
        {
            long double service = 0, link = 0;
            unsigned samples = 0;
            void add(WorkerTaskObservation const& task)
            {
                auto const& t = task.generation;
                samples = std::min(samples + 1, 3u);
                auto update = [&](long double& rate, long double ns)
                { rate += (ns / task.work_units - rate) / samples; };
                if (task.local_direct)
                {
                    // The sink runs inside the local worker call. Its whole
                    // observed service occupies the slot without using a link.
                    update(service, static_cast<long double>(task.execution_ns) + task.transfer_ns);
                    return;
                }
                update(service, task.connect_ns + std::max(static_cast<long double>(t.worker_ns) + t.send_ns,
                    static_cast<long double>(t.receiver_ns) + t.frame_wait_ns));
                // Prefer time in actual reads. Synchronous receiver callbacks
                // can occupy a sender even when the link has room.
                update(link, std::min(t.send_ns, t.frame_read_ns ? t.frame_read_ns : t.frame_wait_ns));
            }
        };
        Peer peer;
        pool::Kind kind = pool::Kind::GenerateCpu;
        uint8_t k = 0, strength = 0;
        GenerationOutput output = GenerationOutput::Packed;
        Plan plan;
        uint64_t sequence = 0;
        Clock::time_point seen{};
        int direction = 0;
        unsigned votes = 0, slots = 1;
        uint32_t ingress = 0;
        unsigned joint_trial = 0, independent_width = 0;
        Clock::time_point trial_until{};
        bool offered = true, delivered = false, delivered_contribution = false, deferred_joint = false;
        std::array<Rate, pool::max_generation_batch_members + 1> rates{};
        Rate independent;
        struct Prior : Rate { Clock::time_point until{}; unsigned dated_samples = 0; uint64_t oldest_utc = 0; };
        std::array<Prior, pool::max_generation_batch_members + 1> prior{}, independent_prior{};
        bool matches(uint8_t size, uint8_t compression, pool::Kind requested, GenerationOutput expected) const
        { return offered && k == size && strength == compression && output == expected && (requested == pool::Kind::GenerateAny || kind == requested); }
        Rate const& pipeline_rate() const { return independent.samples ? independent : rates[1]; }
        Rate const& packed_rate(unsigned width, unsigned samples) const
        { return width < rates.size() && rates[width].samples >= samples ? rates[width] : rates[0]; }
        long double independent_cost(unsigned n) const
        { return n == 1 && rates[1].samples ? rates[1].service : n * pipeline_rate().service; }
        unsigned best_joint(unsigned maximum_members, unsigned minimum_samples = 1) const
        {
            // An individual delivery needs its own measured baseline before
            // it can displace an independently pipelined batch.
            unsigned best = rates[1].samples == 3 ? 1 : 0;
            for (unsigned n = 2; n < rates.size() && n <= maximum_members; ++n)
                if (rates[n].samples >= minimum_samples && (!best || rates[n].service < rates[best].service)) best = n;
            return best;
        }
        Rate const& estimate_rate(unsigned batch, unsigned minimum_samples) const
        {
            if (output == GenerationOutput::Packed) return packed_rate(batch, minimum_samples);
            // A fresh group is published as a complete stream of this width.
            if (output == GenerationOutput::Fresh) return rates[batch < rates.size() ? batch : 0];
            auto const& joint = rates[std::max(1u, best_joint(batch, minimum_samples))];
            auto const& single = rates[1];
            auto const& ordinary = single.samples >= minimum_samples && single.service < joint.service ? single : joint;
            if (batch > 1 && independent.samples >= minimum_samples &&
                (ordinary.samples < minimum_samples || independent.service < ordinary.service)) return independent;
            return ordinary;
        }
    };
    void calibrate(History& item, WorkerTaskObservation const& task, Clock::time_point now)
    {
        auto& prior = (task.independent_members ? item.independent_prior : item.prior)[task.work_units];
        prior = {};
        if (!item.offered || !roster_ || roster_k_ != task.k || roster_strength_ != task.strength ||
            !task.ingress_interface || !task.generation.frame_read_ns || !task.generation.send_ns) return;
        auto const* route = offered_route(item.peer.worker);
        if (!route || !same_route(item.peer, *route) || route->host.empty() ||
            (task.kind == pool::Kind::GenerateGpu && route->gpu_identity.empty())) return;
        auto const compute = task.generation.worker_ns / task.work_units;
        auto const wire = std::min(task.generation.frame_read_ns, task.generation.send_ns) / task.work_units;
        if (!compute || !wire) return;
        try
        {
            auto const saved = calibration_(task, *route, compute, wire);
            if (saved.remaining <= std::chrono::seconds::zero() || saved.samples < 2 || !saved.oldest_utc ||
                !saved.compute_low || !saved.wire_low ||
                compute < saved.compute_low || compute > saved.compute_high ||
                wire < saved.wire_low || wire > saved.wire_high) return;
            prior.add(task);
            // Charge the current sink and setup once. Only worker execution
            // and measured physical ingress carry a dated upper observation.
            prior.service = std::max(prior.service, static_cast<long double>(task.connect_ns) / task.work_units +
                static_cast<long double>(saved.compute_high) + saved.wire_high);
            prior.link = saved.wire_high;
            prior.samples = 3;
            prior.until = now + saved.remaining;
            prior.dated_samples = saved.samples; prior.oldest_utc = saved.oldest_utc;
        }
        catch (...) {} // An optional calibration never owns a successful receipt.
    }
    struct ActivePeer
    {
        Peer peer;
        uint64_t first = 0, last = 0, sequence = 0;
        uint64_t ingress_selected = 0;
        uint32_t ingress = 0;
        unsigned in_flight = 0;
        Clock::time_point seen{};
        long double delivery_ns = 0;
        Clock::time_point packed_since{};
        uint8_t packed_k = 0, packed_strength = 0;
        pool::Kind packed_kind = pool::Kind::GenerateCpu;
        long double remaining_ns(Clock::time_point now) const
        {
            if (in_flight != 1 || !delivery_ns) return 0;
            auto const elapsed = std::chrono::duration<long double, std::nano>(now - seen).count();
            return std::max(0.L, delivery_ns - elapsed);
        }
    };
    static bool valid(WorkerTaskObservation const& value, pool::Endpoint const& route)
    {
        return (value.kind == pool::Kind::GenerateCpu || value.kind == pool::Kind::GenerateGpu) &&
            value.k >= 18 && value.k <= 28 && !(value.k & 1) && value.strength <= 32 && value.work_units &&
            value.generation_output <= GenerationOutput::Fresh &&
            (!value.independent_members || value.generation_output == GenerationOutput::Contributions) &&
            !value.worker_identity.empty() && value.worker_identity.size() <= 64 &&
            !value.worker_instance.empty() && value.worker_instance.size() <= 64 &&
            route.host.size() <= 253 && route.server_name.size() <= 253;
    }
    template<size_t Size> static std::array<char, Size> text(std::string const& value)
    { std::array<char, Size> result{}; std::copy(value.begin(), value.end(), result.begin()); return result; }
    static Peer key(WorkerTaskObservation const& value, pool::Endpoint const& route)
    { return {text<64>(value.worker_identity), text<64>(value.worker_instance), text<253>(route.host),
        text<253>(route.server_name), route.port, value.local_direct}; }
    pool::GenerationRoute const* offered_route(std::array<char, 64> const& worker) const
    {
        for (auto const& route : roster_->routes) if (text<64>(route.identity) == worker) return &route;
        return nullptr;
    }
    static bool same_route(Peer const& peer, pool::GenerationRoute const& route)
    {
        return peer.instance == text<64>(route.instance) && peer.port == route.endpoint.port &&
            peer.host == text<253>(route.endpoint.host) && peer.server_name == text<253>(route.endpoint.server_name);
    }
    void apply_offer(History& item) const
    {
        if (!roster_ || item.k != roster_k_ || item.strength != roster_strength_) return;
        auto const* route = offered_route(item.peer.worker);
        if (!route) { item.offered = !roster_->complete; return; }
        auto const slots = item.kind == pool::Kind::GenerateGpu ? route->gpu_slots : route->cpu_slots;
        item.offered = slots && same_route(item.peer, *route);
        if (item.offered) item.slots = std::min(slots, 64u);
    }
    std::array<History, maximum>::iterator find(WorkerTaskObservation const& value, Peer const& peer)
    {
        return std::find_if(histories_.begin(), histories_.begin() + count_, [&](History const& item)
        { return item.peer == peer && item.k == value.k && item.strength == value.strength &&
            item.kind == value.kind && item.output == value.generation_output; });
    }
    std::array<ActivePeer, maximum>::iterator current(std::array<char, 64> const& worker)
    {
        return std::find_if(peers_.begin(), peers_.begin() + peer_count_, [&](ActivePeer const& item)
        { return item.peer.worker == worker; });
    }
    bool accepts(std::array<ActivePeer, maximum>::iterator at, Peer const& peer, uint64_t selection) const
    { return selection && at != peers_.begin() + peer_count_ && at->peer == peer && selection >= at->first && selection <= at->last; }
    template<class Predicate> void erase(Predicate predicate)
    {
        auto end = std::remove_if(histories_.begin(), histories_.begin() + count_, predicate);
        count_ = size_t(end - histories_.begin());
    }
    void retire(Clock::time_point now)
    {
        auto aged = [&](auto const& item)
        {
            if (now - item.seen >= lifetime_) return true;
            if (sequence_ - item.sequence < maximum) return false;
            // Fast peers can complete many tasks during a slow route's first
            // delivery. A matching live offer retains that measurement within
            // the normal time and fixed-size LRU bounds.
            auto const* route = roster_ ? offered_route(item.peer.worker) : nullptr;
            return !route || !same_route(item.peer, *route);
        };
        auto end = std::remove_if(peers_.begin(), peers_.begin() + peer_count_, [&](ActivePeer const& item)
        { return aged(item); });
        peer_count_ = size_t(end - peers_.begin());
        erase([&](History const& item)
        { auto at = current(item.peer.worker); return aged(item) || at == peers_.begin() + peer_count_ || at->peer != item.peer; });
    }
    std::array<History, maximum> histories_{};
    std::array<ActivePeer, maximum> peers_{};
    std::shared_ptr<pool::GenerationRoster const> roster_;
    uint8_t roster_k_ = 0, roster_strength_ = 0;
    size_t count_ = 0, peer_count_ = 0;
    uint64_t sequence_ = 0, selection_ = 0;
    std::chrono::milliseconds lifetime_;
    WorkerGenerationCalibration calibration_;
};
struct TaskState { uint64_t sequence = 0, next_attempt = 1, touched = 0; };
struct RetryHandle
{
    std::weak_ptr<char> registry;
    std::string session, task_id;
    std::shared_ptr<TaskState> state;
};
// Logical retry ownership and the bounded completed-task cache are separate.
// A caller-held identity pins its state after a failed Connection disappears;
// unrelated completed work can never turn that replay into a new submission.
class TaskRegistry
{
public:
    static constexpr size_t maximum = 4096;
    void reset() { states_.clear(); sequence_ = 0; }
    std::shared_ptr<TaskState> acquire(WorkerTaskIdentity const& identity, std::string const& session)
    {
        require(!identity.task_id.empty() && identity.retry_, "compute task requires a retry identity");
        auto& retry = *identity.retry_;
        auto registry = retry.registry.lock();
        require(!registry || registry == lifetime_, "compute retry identity belongs to another live client");
        require(!registry || !retry.state || retry.task_id == identity.task_id, "cannot retarget a compute retry identity");
        if (registry && retry.session == session && retry.state)
        { retry.state->touched = sequence_; return retry.state; }
        auto found = states_.find(identity.task_id);
        if (found == states_.end())
        {
            if (states_.size() == maximum)
            {
                auto oldest = states_.end();
                for (auto it = states_.begin(); it != states_.end(); ++it)
                    if (it->second.use_count() == 1 && (oldest == states_.end() || it->second->touched < oldest->second->touched))
                        oldest = it;
                require(oldest != states_.end(), "too many concurrent compute retry identities");
                states_.erase(oldest);
            }
            require(sequence_ < UINT64_MAX, "compute client task sequence exhausted");
            auto state = std::make_shared<TaskState>(); state->sequence = ++sequence_; state->touched = sequence_;
            found = states_.emplace(identity.task_id, std::move(state)).first;
        }
        retry.registry = lifetime_; retry.session = session; retry.task_id = identity.task_id; retry.state = found->second;
        retry.state->touched = sequence_;
        return retry.state;
    }
private:
    std::shared_ptr<char> lifetime_ = std::make_shared<char>();
    uint64_t sequence_ = 0;
    std::unordered_map<std::string, std::shared_ptr<TaskState>> states_;
};
} // namespace worker_detail

WorkerTaskIdentity::WorkerTaskIdentity(std::string task, uint64_t version)
    : task_id(task.empty() ? pool::random_token() : std::move(task)), attempt(version),
      retry_(std::make_shared<worker_detail::RetryHandle>()) {}

struct WorkerClient::Impl
{
    WorkerClientConfig config;
    pool::BrokerClient broker;
    Budget generation_admission;
    pool::OwnerSession owner;
    std::timed_mutex sequence_mutex;
    worker_detail::TaskRegistry task_states;
    mutable std::mutex planning_mutex;
    std::mutex capacity_refresh_mutex;
    worker_detail::GenerationPlanning planning;
    struct CapacityHint { uint8_t k = 0, strength = 0; Clock::time_point expires{}; WorkerGenerationCapacity value; };
    CapacityHint generation_capacity;
    WorkerGenerationCapacity capacity(uint8_t k, uint8_t strength, Clock::time_point deadline = Clock::time_point::max(),
                                      pool::Lease const* assigned = nullptr);
    uint64_t data_connect_ns = 0;
    struct PendingMember
    {
        uint8_t k, strength;
        std::array<uint8_t, 32> const& id;
        WorkerContributionWindowSink const& window;
        WorkerCancellation cancel;
        Clock::time_point other_work_until;
        bool selected = false, done = false, retry_single = false;
        std::exception_ptr failure;
    };
    std::mutex pending_mutex;
    std::condition_variable pending_changed;
    std::deque<PendingMember*> pending_members;
    bool sending_members = false;
    explicit Impl(WorkerClientConfig options) : config(std::move(options)), broker(config.broker, config.tls),
        generation_admission(0, config.max_active_generation, 0), planning(config.task_timeout_ms, config.generation_calibration)
    {
        if (config.compatibility.empty()) config.compatibility = worker_compatibility();
        if (config.codec_compatibility.empty()) config.codec_compatibility = worker_compatibility(pool::Kind::Pack);
        require(!config.compatibility.empty() && (config.local_only || !config.owner_id.empty()),
            "compute client requires a compatibility identity and persistent owner ID");
        require(!config.local_only || config.local_service || config.local_service_provider || config.acquire_private_generation,
            "local-only compute client requires a local executor");
        require(config.max_task_bytes && config.max_task_bytes <= UINT32_MAX && config.max_window_chunks &&
            config.task_timeout_ms && config.max_active_generation && config.max_active_generation <= 4096,
            "invalid compute client bounds");
        require(config.generation_kind == pool::Kind::GenerateCpu || config.generation_kind == pool::Kind::GenerateGpu ||
            config.generation_kind == pool::Kind::GenerateAny, "invalid compute client generation policy");
    }
    ~Impl()
    {
        if (config.ephemeral_owner && !owner.token.empty()) try { cleanup_control().close_owner(owner); }
        catch (...) { /* An abandoned automatic session expires at the broker. */ }
    }
    std::shared_ptr<WorkerService> local_service() const
    {
        if (config.local_service_provider) return config.local_service_provider();
        return config.local_service ? std::shared_ptr<WorkerService>(config.local_service, [](WorkerService*) {}) : nullptr;
    }
    WorkerCancellation cancellation(WorkerCancellation cancel)
    {
        cancel.deadline = std::min(cancel.deadline, Clock::now() + std::chrono::milliseconds(config.task_timeout_ms));
        cancel.check(); return cancel;
    }
    pool::BrokerClient cleanup_control() const
    { return worker_detail::cleanup_control(broker, config.tls); }
    std::unique_ptr<CodecScheduler::Lease> cpu_lease(WorkerCancellation const& cancel)
    {
        cancel.check();
        return config.cpu_scheduler ? std::make_unique<CodecScheduler::Lease>(config.cpu_scheduler.get(), 1,
            [cancel] { cancel.check(); }) : nullptr;
    }
    std::unique_lock<std::timed_mutex> sequence_lock(WorkerCancellation const& cancel)
    {
        std::unique_lock lock(sequence_mutex, std::defer_lock);
        for (;;)
        {
            cancel.check();
            if (lock.try_lock_until(std::min(cancel.deadline, Clock::now() + std::chrono::milliseconds(50))))
            { cancel.check(); return lock; }
        }
    }
    unsigned remaining_timeout(WorkerCancellation const& cancel) const
    {
        cancel.check();
        auto const remaining = std::chrono::duration_cast<std::chrono::milliseconds>(cancel.deadline - Clock::now()).count();
        require(remaining > 0, "compute task deadline expired");
        return unsigned(std::min<uint64_t>(uint64_t(remaining), config.task_timeout_ms));
    }
    void prepare_session_locked(pool::BrokerClient& control, pool::OwnerSession& session, bool refresh)
    {
        if (owner.token.empty() || (refresh && session.token == owner.token))
        { owner = control.open_owner(config.owner_id, config.ephemeral_owner); task_states.reset(); }
        session = owner;
    }
    std::atomic<uint64_t> input_bytes{0}, output_bytes{0}, generations{0}, contributions{0}, codecs{0}, failures{0};
    std::atomic<uint64_t> local_tasks{0}, task_elapsed_ns{0};
    std::atomic<uint64_t> active_generation{0}, peak_active_generation{0}, peak_generation_members{0};
    std::atomic<uint64_t> timed_generation_tasks{0}, worker_ns{0}, send_ns{0}, frame_wait_ns{0}, receiver_ns{0}, frame_read_ns{0};
    std::atomic<uint64_t> independent_tasks{0}, independent_members{0};
    void observe_generation_members(uint64_t count)
    {
        auto peak = peak_generation_members.load();
        while (peak < count && !peak_generation_members.compare_exchange_weak(peak, count)) {}
    }
    struct GenerationLease
    {
        Impl& impl;
        Budget::Lease slot;
        GenerationLease(Impl& owner, uint64_t members, WorkerCancellation const& cancel)
            : impl(owner), slot(owner.generation_admission.acquire(0, true, cancel))
        {
            auto const active = impl.active_generation.fetch_add(1) + 1;
            auto peak = impl.peak_active_generation.load();
            while (peak < active && !impl.peak_active_generation.compare_exchange_weak(peak, active)) {}
            impl.observe_generation_members(members);
        }
        ~GenerationLease() { impl.active_generation.fetch_sub(1); }
        GenerationLease(GenerationLease const&) = delete;
    };
    void observed(WorkerTaskObservation const& value, pool::Endpoint const& route, uint64_t selection)
    {
        { std::lock_guard lock(planning_mutex); planning.observed(value, route, selection); }
        if (config.observe) try { config.observe(value); } catch (...) {}
    }
    void admit(WorkerTaskAdmission const& value) const
    {
        if (!config.admit_task) return;
        try { config.admit_task(value); }
        catch (WorkerAdmissionRejected const&) { throw; }
        catch (WorkerResourcesUnavailable const&) { throw; }
        catch (std::exception const& error) { throw WorkerAdmissionRejected(error.what()); }
        catch (...) { throw WorkerAdmissionRejected("compute admission policy failed"); }
    }
    std::function<void(WorkerTaskAdmission const&)> local_admission() const
    {
        if (!config.admit_task) return {};
        return [this](WorkerTaskAdmission value)
        {
            value.owner_identity = config.local_only ? "" : pool::certificate_identity(config.tls.cert_file); value.owner_id = config.owner_id;
            admit(value);
        };
    }
    void generate_private(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
        WorkerPackedSink const& packed_sink, WorkerContributionSink const& contribution_sink, WorkerCancellation cancel)
    {
        auto const begin = Clock::now(); auto selected_at = begin, first_output = Clock::time_point{};
        WorkerTaskObservation value; value.local_direct = true;
        auto admission = [&](WorkerTaskAdmission task)
        {
            task.local_fallback = false; task.owner_id = config.owner_id;
            value.worker_identity = task.worker.identity; value.worker_instance = task.worker.instance;
            value.kind = task.task.kind; value.k = task.task.k; value.strength = task.task.strength; value.work_units = task.task.work_units;
            value.generation_output = task.task.generation_output;
            selected_at = Clock::now(); admit(task);
            if (config.selected) config.selected(value);
            cancel.check();
        };
        auto output = [&] { if (first_output == Clock::time_point{}) first_output = Clock::now(); };
        try
        {
            auto grant = config.acquire_private_generation ? config.acquire_private_generation(k, strength, unsigned(ids.size()),
                packed_sink ? GenerationOutput::Packed : GenerationOutput::Contributions, cancel) : nullptr;
            auto local = config.acquire_private_generation ? nullptr : local_service();
            auto* service = grant ? grant->service : local.get();
            require(service, "local-only compute executor is unavailable");
            if (grant) value.local_device = grant->local_device;
            auto const* resources = grant ? &grant->resources : nullptr;
            if (packed_sink) service->generate_packed_local(k, strength, ids, [&](PackedFragmentTile const& tile, uint64_t count)
                { output(); packed_sink(tile, count); }, cancel, config.generation_kind, admission, resources);
            else service->contributions_local(k, strength, ids, [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
                { output(); contribution_sink(ordinal, bytes, members); }, cancel, config.generation_kind, admission, resources);
            cancel.check();
            if (grant) grant->delivered = true;
        }
        catch (WorkerAdmissionRejected const&) { throw; }
        catch (WorkerTaskDeclined const&) { throw; }
        catch (...) { failures.fetch_add(1); throw; }
        auto const end = Clock::now(); auto const boundary = first_output == Clock::time_point{} ? end : first_output;
        value.elapsed_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        value.execution_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(boundary - selected_at).count());
        value.transfer_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - boundary).count());
        if (packed_sink) generations.fetch_add(1); else contributions.fetch_add(1);
        local_tasks.fetch_add(1); task_elapsed_ns.fetch_add(value.elapsed_ns);
        if (config.observe) try
        {
            value.member_id_count = unsigned(ids.size()); std::copy(ids.begin(), ids.end(), value.member_ids.begin());
            config.observe(value);
        }
        catch (...) {}
    }
    struct Connection
    {
        Impl& impl;
        pool::BrokerClient control;
        pool::Lease lease;
        pool::OwnerSession session;
        std::shared_ptr<worker_detail::TaskState> state;
        pool::Channel channel;
        Clock::time_point begin;
        Operation operation;
        uint64_t input_bytes = 0, output_bytes = 0;
        uint64_t logical_input_bytes = 0, logical_output_bytes = 0;
        bool completed = false, resource_rejected = false, declined = false, local_direct = false, retry_same_worker = false;
        std::shared_ptr<WorkerService> local_service;
        Clock::time_point selected_at = Clock::now(), first_output{};
        WorkerGenerationTiming generation;
        unsigned worker_slots = 0;
        uint32_t ingress = 0;
        uint64_t connect_ns = 0;
        WorkerTaskOptions options;
        uint64_t planning_selection = 0;
        WorkerTaskObservation observation() const
        {
            WorkerTaskObservation result; result.worker_identity = lease.worker_identity;
            result.worker_instance = lease.worker_instance; result.kind = lease.task.kind;
            result.k = lease.task.k; result.strength = lease.task.strength; result.work_units = lease.task.work_units;
            result.input_bytes = input_bytes; result.output_bytes = output_bytes; result.local_direct = local_direct;
            auto const end = Clock::now();
            result.elapsed_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
            auto const boundary = first_output == Clock::time_point{} ? end : first_output;
            result.execution_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(boundary - selected_at).count());
            result.transfer_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end - boundary).count());
            result.generation_output = lease.task.generation_output; result.generation = generation;
            result.worker_slots = worker_slots;
            result.generation_resources = lease.task.resources;
            result.ingress_interface = ingress; result.connect_ns = connect_ns;
            result.independent_members = lease.task.independent_members;
            return result;
        }
        ~Connection()
        {
            if (planning_selection)
            {
                std::lock_guard lock(impl.planning_mutex);
                impl.planning.released(lease.worker_identity, planning_selection);
            }
            impl.input_bytes.fetch_add(input_bytes); impl.output_bytes.fetch_add(output_bytes);
            if (completed)
            {
                if (operation == Operation::Packed) impl.generations.fetch_add(1);
                else if (operation == Operation::Contributions || operation == Operation::Fresh) impl.contributions.fetch_add(1);
                else impl.codecs.fetch_add(1);
            }
            else
            {
                channel.close();
                if (!declined)
                {
                    impl.failures.fetch_add(1);
                    { std::lock_guard lock(impl.planning_mutex); impl.planning.failed(observation(), lease.endpoint, planning_selection); }
                }
                if (declined || retry_same_worker) try { impl.cleanup_control().cancel(session, lease.task.task_id, lease.task.attempt); } catch (...) {}
                else if (!resource_rejected) try { impl.cleanup_control().abandon(lease); } catch (...) {}
            }
        }
        void check_frame(pool::Frame const& frame)
        {
            if (frame.type == pool::Message::Error)
            {
                try { throw_worker_error(frame.payload); }
                catch (pool::TransientTransport const&)
                {
                    // An authenticated temporary refusal (for example, a full
                    // shared queue) must not exclude this worker from retries.
                    // Cancel retains any running reservation until finish.
                    retry_same_worker = true; throw;
                }
            }
            require(frame.type == pool::Message::Data || frame.type == pool::Message::Reply ||
                frame.type == pool::Message::Result || frame.type == pool::Message::End, "unexpected compute result frame");
        }
        void delivered(std::span<std::array<uint8_t, 32> const> ids = {})
        {
            auto const end = Clock::now();
            auto const elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            auto const service = std::chrono::duration_cast<std::chrono::nanoseconds>(end - selected_at).count();
            if (service > 0) try { control.delivered(lease, {uint64_t(service),
                local_direct ? logical_input_bytes : input_bytes, local_direct ? logical_output_bytes : output_bytes}); }
            catch (pool::TransientTransport const&)
            { /* A broker outage does not invalidate accepted data. */ }
            completed = true;
            if (local_direct) impl.local_tasks.fetch_add(1);
            if (lease.task.independent_members)
            {
                impl.independent_tasks.fetch_add(1); impl.independent_members.fetch_add(lease.task.work_units);
                impl.observe_generation_members(lease.task.work_units);
            }
            impl.task_elapsed_ns.fetch_add(uint64_t(std::max<int64_t>(0, elapsed)));
            auto value = observation();
            value.member_id_count = unsigned(ids.size()); std::copy(ids.begin(), ids.end(), value.member_ids.begin());
            if (generation.worker_ns)
            {
                impl.timed_generation_tasks.fetch_add(1); impl.worker_ns.fetch_add(generation.worker_ns);
                impl.send_ns.fetch_add(generation.send_ns); impl.frame_wait_ns.fetch_add(generation.frame_wait_ns);
                impl.receiver_ns.fetch_add(generation.receiver_ns); impl.frame_read_ns.fetch_add(generation.frame_read_ns);
            }
            impl.observed(value, lease.endpoint, planning_selection);
            if (options.observe) try { options.observe(value); } catch (...) {}
        }
    };
    std::unique_ptr<Connection> connect(pool::Kind kind, uint8_t k, uint8_t strength, uint64_t work_units,
                                      uint64_t host_bytes, Operation operation, WorkerTaskIdentity identity,
                                      WorkerCancellation cancel, WorkerTaskOptions const& options = {},
                                      uint32_t chunk_count = 0, uint64_t max_output_bytes = 0, bool independent = false)
    {
        auto const begin = Clock::now();
        auto control = broker.with_deadline(cancel.deadline);
        pool::Task task; task.task_id = identity.task_id.empty() ? pool::random_token() : identity.task_id;
        require(task.task_id.size() <= 256 && identity.attempt, "invalid compute task identity");
        task.kind = kind; task.k = k; task.strength = strength; task.work_units = work_units;
        task.generation_output = generation_output(operation);
        task.independent_members = independent;
        task.chunk_count = chunk_count; task.max_output_bytes = max_output_bytes;
        task.compatibility = kind == pool::Kind::Pack ? config.codec_compatibility : config.compatibility;
        task.resources.cpu_threads = kind == pool::Kind::Pack ? 1 : config.generation_threads;
        task.resources.host_bytes = host_bytes; task.resources.slots = 1;
        std::optional<pool::Lease> lease;
        pool::OwnerSession session;
        std::shared_ptr<worker_detail::TaskState> state;
        bool submitted = false, declined = false;
        auto submit = [&](bool refresh)
        {
            auto lock = sequence_lock(cancel);
            prepare_session_locked(control, session, refresh);
            state = task_states.acquire(identity, session.token);
            task.sequence = state->sequence; task.attempt = std::max(identity.attempt, state->next_attempt);
            require(task.attempt < UINT64_MAX, "compute task attempt exhausted"); state->next_attempt = task.attempt + 1;
            submitted = true; lease = control.acquire(session, task, options.other_work_until);
        };
        try
        {
            try { submit(false); } catch (pool::StaleOwnerSession const&) { submit(true); }
            auto next_hint = Clock::now();
            while (!lease)
            {
                if (operation != Operation::Encode && Clock::now() >= next_hint)
                {
                    auto const current = capacity(k, strength, cancel.deadline);
                    if (work_units > 1 && (current.slots || !current.control_ns))
                    {
                        pool::BrokerClient::GenerationAvailability availability;
                        uint32_t slots = 0;
                        auto const batch = control.generation_hint(session, task, &slots,
                            independent || options.queued_generation ? &availability : nullptr);
                        if (options.queued_generation &&
                            !options.queued_generation(availability.independent_width(unsigned(work_units))))
                        { declined = true; throw WorkerTaskDeclined(); }
                        // Split when an eligible worker can take less work.
                        // An empty pool keeps the original queued identity and
                        // deadline while a worker registers.
                        auto const maximum = independent ? availability.independent_batch : batch;
                        if (maximum < work_units && (maximum || (slots && (independent || operation != Operation::Contributions))))
                            throw pool::UnsupportedResources("no current worker can admit this generation batch");
                    }
                    next_hint = Clock::now() + std::chrono::seconds(1);
                }
                cancel.check(); std::this_thread::sleep_for(std::chrono::milliseconds(50));
                try { lease = control.acquire(session, task, options.other_work_until); }
                catch (pool::StaleOwnerSession const&) { submit(true); }
            }
            cancel.check();
        }
        catch (...)
        {
            if (!declined) failures.fetch_add(1);
            if (submitted) try { cleanup_control().cancel(session, task.task_id, task.attempt); } catch (...) {}
            throw;
        }
        auto connection = std::unique_ptr<Connection>(new Connection{*this, std::move(control), *lease, session, state, {}, begin, operation});
        if (operation != Operation::Encode && config.generation_capacity_changed)
            (void)capacity(k, strength, cancel.deadline, &*lease);
        require(lease->task.generation_output == task.generation_output && lease->task.independent_members == independent,
            "compute lease changed generation output");
        connection->options = options;
        connection->local_service = local_service();
        connection->local_direct = connection->local_service && connection->local_service->owns(*lease);
        {
            std::lock_guard lock(planning_mutex); connection->selected_at = Clock::now();
            connection->planning_selection = planning.selected(connection->observation(), lease->endpoint, connection->selected_at);
        }
        try
        {
            if (config.admit_task)
            {
                bool const concrete = task.kind == pool::Kind::GenerateAny ?
                    (lease->task.kind == pool::Kind::GenerateCpu || lease->task.kind == pool::Kind::GenerateGpu) : lease->task.kind == task.kind;
                if (lease->owner_identity != pool::certificate_identity(config.tls.cert_file) || lease->owner_id != config.owner_id ||
                    lease->owner_session != session.token || lease->owner_generation != session.generation ||
                    lease->task.task_id != task.task_id || lease->task.attempt != task.attempt || lease->task.sequence != task.sequence ||
                    lease->task.k != task.k || lease->task.strength != task.strength || lease->task.work_units != task.work_units ||
                    lease->task.compatibility != task.compatibility || lease->task.chunk_count != task.chunk_count ||
                    lease->task.max_output_bytes != task.max_output_bytes || !concrete)
                    throw WorkerAdmissionRejected("compute lease differs from its authenticated owner request");
                auto tls = config.tls; tls.deadline = std::min(tls.deadline, cancel.deadline);
                WorkerStatus status;
                try { status = connection->local_direct ? connection->local_service->status() : WorkerService::remote_status(lease->endpoint, tls); }
                catch (pool::TransientTransport const&) { throw; }
                catch (std::exception const& error) { throw WorkerAdmissionRejected(error.what()); }
                if (status.identity != lease->worker_identity || status.instance != lease->worker_instance)
                    throw WorkerAdmissionRejected("compute lease differs from authenticated worker status");
                admit({std::move(status), lease->task, lease->owner_identity, lease->owner_id, lease->broker_identity, false});
                cancel.check();
            }
            if (config.selected) config.selected(connection->observation());
            if (options.packed_route && !options.packed_route->matches(*lease)) throw PackedRouteChanged();
            if (options.selected && !options.selected(connection->observation())) throw WorkerTaskDeclined();
        }
        catch (...) { connection->declined = true; throw; }
        if (connection->local_direct) return connection;
        auto tls = config.tls; tls.deadline = std::min(tls.deadline, cancel.deadline);
        tls.generation_timing = kind != pool::Kind::Pack;
        auto const start_connect = Clock::now();
        connection->channel = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
        {
            auto const ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_connect).count());
            if (kind != pool::Kind::Pack)
            { connection->connect_ns = ns; connection->ingress = connection->channel.ingress_interface(); }
            std::lock_guard lock(planning_mutex);
            data_connect_ns = data_connect_ns ? std::min(data_connect_ns, ns) : ns;
        }
        return connection;
    }
    bool generate_once(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
                  WorkerPackedSink const& packed_sink, WorkerContributionSink const& contribution_sink,
                  WorkerCancellation cancel, WorkerTaskIdentity identity,
                  WorkerContributionWindowSink const& contribution_window, WorkerFreshManifest const& manifest, bool independent,
                  WorkerTaskOptions const& options)
    {
        cancel = cancellation(cancel); auto const count_bound = fragment_count_bound(k, strength);
        require(!ids.empty() && ids.size() <= pool::max_generation_batch_members &&
            (bool(packed_sink) != bool(contribution_sink)), "invalid compute generation batch");
        size_t const members_per_chunk = independent ? 1 : ids.size();
        uint64_t const chunks_per_group = PlotGroupFile::getChunkCountForK(k);
        uint64_t const expected_chunks = chunks_per_group * (independent ? ids.size() : 1);
        uint64_t const host = worker_cpu_task_bytes(k, strength, members_per_chunk, config.max_task_bytes);
        auto const operation = packed_sink ? Operation::Packed : manifest ? Operation::Fresh : Operation::Contributions;
        auto connection = connect(independent ? pool::Kind::GenerateGpu : config.generation_kind, k, strength, ids.size(), host,
            operation, identity, cancel, options, 0, 0, independent);
        uint64_t next_chunk = 0;
        bool received_manifest = false;
        auto check_contribution = [&](uint64_t ordinal, uint16_t members, std::span<uint8_t const> bytes)
        {
            require(ordinal == next_chunk++ && ordinal < expected_chunks && members == members_per_chunk && !bytes.empty(),
                "invalid compute contribution order or membership");
        };
        auto receive_manifest = [&](uint64_t payload, uint64_t largest, std::array<uint8_t, 32> const& digest)
        {
            require(manifest && !received_manifest && !next_chunk, "unexpected fresh group manifest");
            try { manifest(payload, largest, digest); }
            catch (WorkerTaskDeclined const&) { connection->declined = true; throw; }
            received_manifest = true;
        };
        if (connection->local_direct)
        {
            connection->logical_input_bytes = multiply(ids.size(), sizeof(ids.front()));
            auto first_output = [&]
            { if (connection->first_output == Clock::time_point{}) connection->first_output = Clock::now(); };
            auto packed = [&](PackedFragmentTile const& tile, uint64_t count)
            { first_output(); connection->logical_output_bytes = add(connection->logical_output_bytes,
                add(tile.boundaries.size_bytes(), tile.words.size_bytes())); packed_sink(tile, count); };
            auto local_chunks = [&](uint64_t first, std::span<std::span<uint8_t const> const> chunks, uint16_t members)
            {
                require(!manifest || received_manifest, "unexpected compute contribution stream");
                first_output();
                for (auto bytes : chunks)
                {
                    check_contribution(first++, members, bytes);
                    connection->logical_output_bytes = add(connection->logical_output_bytes, bytes.size());
                }
            };
            auto contribution = [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
            { local_chunks(ordinal, {&bytes, 1}, members); contribution_sink(ordinal, bytes, members); };
            auto window = [&](uint64_t first, std::span<std::span<uint8_t const> const> chunks, uint16_t members)
            { local_chunks(first, chunks, members); contribution_window(first, chunks, members); };
            auto local_manifest = [&](uint64_t payload, uint64_t largest, std::array<uint8_t, 32> const& digest)
            {
                first_output();
                connection->logical_output_bytes = add(connection->logical_output_bytes, 2 * sizeof(uint64_t) + digest.size());
                receive_manifest(payload, largest, digest);
            };
            try
            {
                connection->worker_slots = connection->local_service->generate_assigned(connection->lease, ids,
                    packed_sink ? WorkerPackedSink(packed) : WorkerPackedSink{},
                    contribution_sink ? WorkerContributionSink(contribution) : WorkerContributionSink{},
                    manifest && contribution_window ? WorkerContributionWindowSink(window) : WorkerContributionWindowSink{},
                    manifest ? WorkerFreshManifest(local_manifest) : WorkerFreshManifest{}, cancel);
            }
            catch (WorkerTaskDeclined const&) { connection->declined = true; throw; }
            catch (pool::UnsupportedResources const&)
            { require(!connection->logical_output_bytes, "local worker rejected resources after partial output");
              connection->resource_rejected = true; throw; }
            catch (pool::TransientTransport const&)
            { connection->retry_same_worker = true; throw; }
            if (contribution_sink) require(next_chunk == expected_chunks, "compute contributions are truncated");
            cancel.check(); connection->delivered(ids); return false;
        }
        auto compute = cpu_lease(cancel);
        pool::Writer request(config.tls.max_frame_bytes); pool::write_lease(request, connection->lease);
        request.u32(remaining_timeout(cancel));
        request.u8(uint8_t(operation)); request.u8(k); request.u8(strength); request.u16(uint16_t(ids.size()));
        for (auto const& id : ids) request.bytes(id); connection->input_bytes = request.data().size();
        compute.reset(); connection->channel.send(pool::Message::Work, request.data());
        auto const packed_sent = packed_sink && ids.size() == 1 && connection->planning_selection ?
            Clock::now() : Clock::time_point{};
        uint64_t total = 0, expected_boundaries = 0, expected_words = 0, received_words = 0;
        std::vector<std::span<uint8_t const>> chunks;
        if (contribution_sink) chunks.resize(contribution_window ?
            (manifest ? fresh_group_window_chunks : contribution_window_chunks) : 1);
        PackedMemberValidation validation; std::vector<uint32_t> boundaries; std::vector<uint64_t> words;
        bool prelude = false, delivered_prelude = false, narrow_deltas = true;
        size_t started = 0;
        PackedWordValidation checked_words;
        auto validate_words = [&](uint64_t first_word)
        { checked_words.append(k, validation.flags & PackedMemberValidation::wide_delta, boundaries, first_word, words); };
        auto complete_member = [&]
        {
            require(delivered_prelude && received_words == expected_words && checked_words.complete(k, total),
                "compute result member is truncated or has nonzero padding");
            narrow_deltas &= !(validation.flags & PackedMemberValidation::wide_delta);
        };
        bool const overlap = config.cpu_scheduler && config.cpu_scheduler->can_overlap();
        pool::Frame frame, next_frame;
        Clock::duration read_time{};
        std::optional<OrderedWriter> reader;
        std::future<void> reading;
        // Exclude waiting for the current sink from the next socket timeout.
        // If the sink throws, the broken promise releases the reader before its buffers expire.
        std::optional<std::promise<void>> resume_read;
        std::future<void> validating;
        for (;;)
        {
            auto const before = Clock::now();
            uint64_t delivery_ns = 0;
            bool const prefetched = reading.valid();
            if (prefetched)
            {
                cancel.check(); resume_read->set_value(); reading.get();
                std::swap(frame, next_frame);
            }
            else if (validating.valid())
            {
                // Only the validator borrows words. The wire frame can be reused
                // immediately. Deliver in order on the caller even if TLS stalls.
                auto deliver = [&]
                {
                    if (!validating.valid()) return;
                    auto const start = Clock::now();
                    validating.get(); packed_sink({{}, words, {}}, total);
                    delivery_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
                };
                try { cancel.check(); connection->channel.receive(frame, deliver); }
                catch (...) { deliver(); throw; }
                deliver();
            }
            else { cancel.check(); connection->channel.receive(frame); }
            auto const received = Clock::now();
            connection->generation.receiver_ns += delivery_ns;
            if (connection->first_output != Clock::time_point{})
            {
                auto const wait = std::chrono::duration_cast<std::chrono::nanoseconds>(received - before).count() - delivery_ns;
                connection->generation.frame_wait_ns += wait;
                connection->generation.frame_read_ns += prefetched ?
                    uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(read_time).count()) : wait;
            }
            try { connection->check_frame(frame); }
            catch (pool::UnsupportedResources const&)
            { require(!connection->output_bytes, "compute worker rejected resources after partial output"); connection->resource_rejected = true; throw; }
            if (frame.type == pool::Message::Reply) { require(frame.payload.empty(), "invalid compute keepalive"); continue; }
            if (connection->first_output == Clock::time_point{}) connection->first_output = Clock::now();
            compute = cpu_lease(cancel);
            connection->output_bytes = add(connection->output_bytes, frame.payload.size());
            if (frame.type == pool::Message::End)
            {
                pool::Reader end(frame.payload);
                if (connection->channel.generation_timing())
                {
                    auto const work = end.u64(), sending = end.u64();
                    connection->worker_slots = end.u32();
                    auto const wall = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(received - connection->begin).count());
                    // Peer timings can guide scheduling only within observed
                    // owner wall time. Admission and output checks are independent.
                    if (work <= wall && sending <= wall - work)
                    { connection->generation.worker_ns = work; connection->generation.send_ns = sending; }
                }
                end.finish();
                if (packed_sink) { require(started == ids.size(), "compute batch is truncated"); complete_member(); }
                else require(next_chunk == expected_chunks, "compute contributions are truncated");
                compute.reset(); connection->delivered(ids);
                return packed_sink && narrow_deltas;
            }
            pool::Reader input(frame.payload); auto const type = DataKind(input.u8());
            if (type == DataKind::Prelude)
            {
                require(bool(packed_sink) && started < ids.size(), "unexpected compute member prelude");
                if (started++) complete_member();
                delivered_prelude = false; received_words = 0; checked_words = {}; boundaries.clear();
                total = input.u64(); validation.checked_count = input.u64(); validation.flags = input.u32(); expected_boundaries = input.u64(); input.finish();
                require(total <= count_bound && total <= UINT32_MAX && expected_boundaries == PlotGroupFile::getChunkCountForK(k) + 1,
                    "compute member exceeds public parameter bound");
                require(validation.checked_count == total && !(validation.flags & ~PackedMemberValidation::known_flags) &&
                    !(validation.flags & (PackedMemberValidation::out_of_range | PackedMemberValidation::unsorted)), "invalid compute validation summary");
                expected_words = (multiply(total, k + PlotGroupFile::PROOFS_PER_CHUNK_BITS) + 63) / 64;
                // The validated prelude confirms that the sent request actually ran.
                if (packed_sent != Clock::time_point{})
                {
                    std::lock_guard lock(planning_mutex);
                    planning.receiving_packed(connection->lease, connection->planning_selection, packed_sent);
                }
                boundaries.reserve(size_t(expected_boundaries)); prelude = true;
            }
            else if (type == DataKind::Boundaries)
            {
                require(prelude && !delivered_prelude, "unexpected compute boundary window"); auto const count = input.u32();
                require(count && count <= expected_boundaries - boundaries.size(), "compute boundary window exceeds member");
                for (uint32_t i = 0; i < count; ++i) boundaries.push_back(input.u32()); input.finish();
                if (boundaries.size() == expected_boundaries)
                {
                    require(boundaries.front() == 0 && boundaries.back() == total && std::is_sorted(boundaries.begin(), boundaries.end()),
                        "invalid compute member boundaries");
                    compute.reset(); packed_sink({boundaries, {}, validation}, total); delivered_prelude = true;
                }
            }
            else if (type == DataKind::Words)
            {
                require(delivered_prelude, "compute words precede complete boundaries"); auto const count = input.u32();
                require(count && count <= expected_words - received_words && uint64_t(count) * 8 <= input.remaining(), "compute word tile exceeds member");
                words.resize(count); input.words(words); input.finish();
                auto const first_word = received_words; received_words += count;
                if (overlap && config.packed_receive_overlap && received_words < expected_words)
                {
                    compute.reset();
                    validating = config.cpu_scheduler->submit([&, first_word] { validate_words(first_word); });
                }
                else { validate_words(first_word); compute.reset(); packed_sink({{}, words, {}}, total); }
            }
            else if (type == DataKind::FreshManifest)
            {
                auto const payload = input.u64(), largest = input.u64(); auto const value = input.bytes(32); input.finish();
                std::array<uint8_t, 32> digest{};
                require(value.size() == digest.size(), "invalid fresh group digest");
                std::copy(value.begin(), value.end(), digest.begin()); compute.reset();
                receive_manifest(payload, largest, digest);
            }
            else if (type == DataKind::Contributions)
            {
                require(contribution_sink && (!manifest || received_manifest), "unexpected compute contribution stream");
                if (overlap && config.contribution_read_ahead && contribution_window && !manifest)
                {
                    if (!reader) reader.emplace([] {});
                    auto ready = resume_read.emplace().get_future().share();
                    reading = reader->submit([&, ready]
                    {
                        auto const start = Clock::now();
                        Clock::duration paused{};
                        connection->channel.receive(next_frame, [&]
                        { auto const begin = Clock::now(); ready.get(); paused = Clock::now() - begin; });
                        read_time = Clock::now() - start - paused;
                    });
                }
                while (input.remaining())
                {
                    uint64_t const first = next_chunk;
                    size_t count = 0;
                    do
                    {
                        auto const ordinal = input.u64(); auto const members = input.u16(); auto const bytes = input.bytes(config.tls.max_frame_bytes);
                        check_contribution(ordinal, members, bytes);
                        chunks[count++] = bytes;
                    } while (contribution_window && input.remaining() && count < chunks.size() &&
                        (!independent || next_chunk % chunks_per_group));
                    compute.reset();
                    try
                    {
                        if (contribution_window) contribution_window(first, {chunks.data(), count}, uint16_t(members_per_chunk));
                        else contribution_sink(first, chunks[0], uint16_t(members_per_chunk));
                    }
                    catch (WorkerTaskDeclined const&) { connection->declined = true; throw; }
                    compute = cpu_lease(cancel);
                }
            }
            else throw std::runtime_error("unexpected compute generation data");
            compute.reset();
            connection->generation.receiver_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - received).count();
        }
    }
    bool generate(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
                  WorkerPackedSink const& packed_sink, WorkerContributionSink const& contribution_sink,
                  WorkerCancellation cancel, WorkerTaskIdentity identity,
                  WorkerContributionWindowSink const& contribution_window = {}, WorkerFreshManifest const& manifest = {},
                  bool independent = false, bool admitted = false, WorkerTaskOptions const& options = {})
    {
        cancel = cancellation(cancel);
        require(!ids.empty() && ids.size() <= pool::max_generation_batch_members,
            "invalid compute generation batch");
        std::optional<GenerationLease> admission;
        if (!admitted) admission.emplace(*this, ids.size(), cancel);
        if (config.local_only && manifest) throw WorkerTaskDeclined();
        if (config.local_only) { generate_private(k, strength, ids, packed_sink, contribution_sink, cancel); return false; }
        if (identity.task_id.empty()) identity.task_id = pool::random_token();
        for (unsigned failed = 0;; ++failed)
        {
            try { return generate_once(k, strength, ids, packed_sink, contribution_sink, cancel, identity, contribution_window, manifest, independent, options); }
            catch (pool::UnsupportedResources const&)
            {
                // A refused batch may need a different shape. Its owner must
                // split it under new identities rather than silently changing
                // the body of this logical task or waiting on excluded offers.
                if (ids.size() > 1 || failed >= pool::max_retry_failures) throw;
                cancel.check();
            }
        }
    }

    void contribute_member(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& id,
        WorkerContributionSink const& sink, WorkerContributionWindowSink const& window,
        WorkerCancellation cancel, WorkerTaskIdentity identity, unsigned maximum, Clock::time_point other_work_until)
    {
        cancel = cancellation(cancel);
        GenerationLease admission(*this, 1, cancel);
        PendingMember member{k, strength, id, window, cancel, other_work_until};
        std::array<PendingMember*, pool::max_generation_batch_members> selected{};
        std::array<std::array<uint8_t, 32>, pool::max_generation_batch_members> ids;
        size_t count = 0;
        {
            std::unique_lock lock(pending_mutex);
            pending_members.push_back(&member);
            pending_changed.notify_all();
            while (!member.done && (sending_members || pending_members.front() != &member))
            {
                if (!member.selected)
                {
                    try { cancel.check(); }
                    catch (...) { std::erase(pending_members, &member); pending_changed.notify_all(); throw; }
                }
                pending_changed.wait_for(lock, std::chrono::milliseconds(25));
            }
            if (!member.done)
            {
                // Give simultaneously admitted receivers a single scheduling
                // turn to join. Never wait for a destination to become ready.
                if (pending_members.size() == 1)
                    pending_changed.wait_for(lock, std::chrono::milliseconds(2), [&] { return pending_members.size() > 1; });
                sending_members = true;
                while (count < maximum && !pending_members.empty())
                {
                    auto* next = pending_members.front();
                    if (next->k != k || next->strength != strength) break;
                    pending_members.pop_front(); next->selected = true;
                    selected[count] = next; ids[count++] = next->id;
                }
            }
        }
        if (count)
        {
            bool emitted = false, retry_single = false;
            std::exception_ptr failure;
            std::array<std::exception_ptr, pool::max_generation_batch_members> receiver_errors;
            try
            {
                WorkerTaskOptions options; options.other_work_until = other_work_until;
                for (size_t i = 0; i < count; ++i)
                {
                    cancel.deadline = std::min(cancel.deadline, selected[i]->cancel.deadline);
                    options.other_work_until = std::min(options.other_work_until, selected[i]->other_work_until);
                }
                uint64_t const chunks = PlotGroupFile::getChunkCountForK(k);
                auto receive = [&](uint64_t first, std::span<std::span<uint8_t const> const> bytes, uint16_t members)
                {
                    auto const index = first / chunks;
                    auto& recipient = *selected.at(index);
                    if (receiver_errors[index]) return;
                    try {
                        recipient.cancel.check(); emitted = true;
                        recipient.window(first % chunks, bytes, members);
                    } catch (...) {
                        if (count == 1) throw;
                        // A destination can fill midway through its member.
                        // Drain that member while delivering to the other owners.
                        receiver_errors[index] = std::current_exception();
                    }
                };
                auto one = [&](uint64_t first, std::span<uint8_t const> bytes, uint16_t members)
                { receive(first, {&bytes, 1}, members); };
                generate(k, strength, std::span(ids).first(count), {}, one, cancel,
                    count == 1 ? identity : WorkerTaskIdentity{}, receive, {}, count > 1, true, options);
            }
            catch (pool::UnsupportedResources const&)
            {
                if (count > 1 && !emitted) retry_single = true;
                else failure = std::current_exception();
            }
            catch (...) { failure = std::current_exception(); }
            std::lock_guard lock(pending_mutex);
            for (size_t i = 0; i < count; ++i)
            {
                selected[i]->done = true; selected[i]->failure = receiver_errors[i] ? receiver_errors[i] : failure;
                selected[i]->retry_single = retry_single;
            }
            sending_members = false;
            pending_changed.notify_all();
        }
        if (member.failure) std::rethrow_exception(member.failure);
        if (member.retry_single) generate(k, strength, {&id, 1}, {}, sink, member.cancel, identity, window, {}, false, true,
            {.other_work_until = other_work_until});
    }
};

WorkerClient::WorkerClient(WorkerClientConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
WorkerClient::~WorkerClient() = default;
bool WorkerClient::local_available() const
{
    if (impl_->config.local_only) return impl_->config.local_service || impl_->config.local_service_provider ||
        impl_->config.acquire_private_generation;
    return bool(impl_->local_service());
}
bool WorkerClient::generate(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& id,
    WorkerPackedSink const& sink, WorkerCancellation cancel, WorkerTaskIdentity identity, Clock::time_point other_work_until)
{ return generate_packed(k, strength, {&id, 1}, sink, cancel, std::move(identity), other_work_until); }
bool WorkerClient::generate_packed(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
    WorkerPackedSink const& sink, WorkerCancellation cancel, WorkerTaskIdentity identity, Clock::time_point other_work_until,
    WorkerPackedRoute const* batch_route)
{
    cancel = impl_->cancellation(cancel);
    WorkerTaskOptions options; options.other_work_until = other_work_until;
    bool split = false;
    if (ids.size() > 1 && !impl_->config.local_only)
    {
        if (batch_route && !batch_route->route.identity.empty()) options.packed_route = batch_route;
        options.selected = [&](auto const& assigned)
        {
            std::lock_guard lock(impl_->planning_mutex); split = impl_->planning.needs_first_member(assigned); return !split;
        };
    }
    try { return impl_->generate(k, strength, ids, sink, {}, cancel, std::move(identity), {}, {}, false, false, options); }
    catch (PackedRouteChanged const&) {}
    catch (WorkerTaskDeclined const&) { if (!split) throw; }
    // No plot ID or output was sent. Retain ordered packed delivery and the
    // original deadline across new identities for the disjoint member sets.
    auto first = generate(k, strength, ids.front(), sink, cancel, {}, other_work_until);
    auto rest = generate_packed(k, strength, ids.subspan(1), sink, cancel, {}, other_work_until, batch_route);
    return first && rest;
}
void WorkerClient::contributions(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
    WorkerContributionSink const& sink, WorkerCancellation cancel, WorkerTaskIdentity identity,
    WorkerContributionWindowSink const& window, WorkerFreshManifest const& manifest, bool independent, bool coalesce_singletons,
    Clock::time_point other_work_until)
{
    if (independent)
    {
        require(!impl_->config.local_only && !manifest && window && impl_->config.generation_kind != pool::Kind::GenerateCpu,
            "independent contributions require a pooled GPU receiver");
        impl_->generate(k, strength, ids, {}, sink, cancel, std::move(identity), window, {}, true, false, {.other_work_until = other_work_until});
        return;
    }
    if (coalesce_singletons && ids.size() == 1 && window && !manifest)
    {
        if (auto const width = independent_contribution_width(k, strength); width > 1)
        {
            impl_->contribute_member(k, strength, ids.front(), sink, window, cancel, std::move(identity), width, other_work_until);
            return;
        }
    }
    WorkerTaskOptions options; options.other_work_until = other_work_until;
    if (manifest && ids.size() > 1)
        options.selected = [&](auto const& assigned)
        { std::lock_guard lock(impl_->planning_mutex); return !impl_->planning.needs_first_member(assigned); };
    impl_->generate(k, strength, ids, {}, sink, cancel, std::move(identity), window, manifest, false, false, options);
}
unsigned WorkerClient::try_contributions(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
    WorkerContributionSink const& sink, WorkerCancellation cancel, WorkerTaskIdentity identity,
    WorkerContributionWindowSink const& window, bool independent, Clock::time_point other_work_until)
{
    require(ids.size() > 1 && window, "adaptive encoding requires a multi-member contribution receiver");
    unsigned first = 0;
    WorkerTaskOptions options;
    options.other_work_until = other_work_until;
    if (!independent) options.queued_generation = [&](unsigned width)
    {
        if (width < ids.size()) return true;
        std::lock_guard lock(impl_->planning_mutex);
        if (!impl_->planning.defer_queued_joint(k, strength, unsigned(ids.size()))) return true;
        first = unsigned(ids.size()); return false;
    };
    options.selected = [&](WorkerTaskObservation const& selected)
    {
        if (generation_capacity(k, strength).slots < 2) return true;
        std::lock_guard lock(impl_->planning_mutex);
        first = impl_->planning.repartition_contributions(selected, Clock::now(), options.other_work_until);
        return !first;
    };
    try { impl_->generate(k, strength, ids, {}, sink, cancel, std::move(identity), window, {}, independent, false, options); }
    catch (WorkerTaskDeclined const&) { if (!first) throw; }
    return first;
}
void WorkerClient::generate_local(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& id,
    WorkerPackedSink const& sink, WorkerCancellation cancel)
{
    cancel = impl_->cancellation(cancel); Impl::GenerationLease admission(*impl_, 1, cancel);
    if (impl_->config.acquire_private_generation) return impl_->generate_private(k, strength, {&id, 1}, sink, {}, cancel);
    auto local = impl_->local_service(); require(bool(local), "local compute fallback was not configured");
    local->generate_local(k, strength, id, sink, cancel, impl_->config.generation_kind, impl_->local_admission());
}
void WorkerClient::contributions_local(uint8_t k, uint8_t strength,
    std::span<std::array<uint8_t, 32> const> ids, WorkerContributionSink const& sink, WorkerCancellation cancel)
{
    require(!ids.empty() && ids.size() <= pool::max_generation_batch_members,
        "local compute fallback was not configured or exceeds its batch limit");
    cancel = impl_->cancellation(cancel); Impl::GenerationLease admission(*impl_, ids.size(), cancel);
    if (impl_->config.acquire_private_generation) return impl_->generate_private(k, strength, ids, {}, sink, cancel);
    auto local = impl_->local_service(); require(bool(local), "local compute fallback was not configured");
    local->contributions_local(k, strength, ids, sink, cancel, impl_->config.generation_kind, impl_->local_admission());
}
std::vector<std::vector<uint8_t>> WorkerClient::encode(uint8_t k, std::span<WorkerChunkInput const> chunks,
    uint64_t maximum, WorkerCancellation cancel, WorkerTaskIdentity identity, WorkerEncodedValidator const& validator,
    WorkerTaskOptions const& options)
{
    require(!impl_->config.local_only, "local-only compute client does not support remote encoding");
    cancel = impl_->cancellation(cancel); (void)parameters(k, 2);
    require(!chunks.empty() && chunks.size() <= impl_->config.max_window_chunks &&
        maximum && maximum <= impl_->config.max_task_bytes,
        "compute packing request exceeds configured window");
    auto compute = impl_->cpu_lease(cancel);
    pool::Writer payload(uint32_t(impl_->config.max_task_bytes)); payload.u32(uint32_t(chunks.size()));
    for (auto const& chunk : chunks)
    {
        require(chunk.contributions.size() <= UINT16_MAX && chunk.new_members.size() <= UINT16_MAX, "compute packing membership exceeds format");
        payload.u64(chunk.ordinal); payload.u16(chunk.old_members); payload.bytes(chunk.old_chunk); payload.u16(uint16_t(chunk.contributions.size()));
        for (auto const& contribution : chunk.contributions) { payload.u16(contribution.members); payload.bytes(contribution.bytes); }
        payload.u16(uint16_t(chunk.new_members.size()));
        for (auto values : chunk.new_members)
        { require(values.size() <= UINT32_MAX, "compute raw chunk exceeds bounded window"); payload.u32(uint32_t(values.size())); for (auto value : values) payload.u64(value); }
    }
    compute.reset();
    auto connection = impl_->connect(pool::Kind::Pack, k, 2, payload.data().size(),
        add(multiply(payload.data().size(), 20), ChunkCodec::estimate_work(k, 0, 0)), Operation::Encode, identity, cancel, options,
        uint32_t(chunks.size()), maximum);
    if (connection->local_direct)
    {
        std::vector<std::vector<uint8_t>> output(chunks.size());
        connection->logical_input_bytes = payload.data().size();
        try
        {
            connection->local_service->encode_assigned(connection->lease, payload.data(), maximum,
                [&](size_t index, std::span<uint8_t const> bytes)
                { require(index < output.size(), "local worker returned an invalid encoded chunk");
                  connection->logical_output_bytes = add(connection->logical_output_bytes, bytes.size());
                  output[index].assign(bytes.begin(), bytes.end()); }, cancel);
        }
        catch (pool::TransientTransport const&)
        { connection->retry_same_worker = true; throw; }
        if (validator) { compute = impl_->cpu_lease(cancel); validator(output); compute.reset(); }
        cancel.check(); connection->delivered(); return output;
    }
    pool::Writer request(impl_->config.tls.max_frame_bytes); pool::write_lease(request, connection->lease);
    request.u32(impl_->remaining_timeout(cancel));
    request.u8(uint8_t(Operation::Encode)); request.u8(k); request.u8(2); request.u64(payload.data().size()); request.u64(maximum);
    connection->input_bytes = add(request.data().size(), payload.data().size());
    connection->channel.send(pool::Message::Work, request.data());
    send_blob(connection->channel, pool::Message::Data, payload.data(), impl_->config.tls.max_frame_bytes, cancel);
    std::vector<std::vector<uint8_t>> output(chunks.size());
    std::vector<size_t> written(chunks.size(), 0);
    uint64_t size = 0; size_t current = 0, announced = 0;
    for (;;)
    {
        cancel.check(); auto frame = connection->channel.receive(); connection->check_frame(frame);
        if (frame.type == pool::Message::Reply) { require(frame.payload.empty(), "invalid compute keepalive"); continue; }
        if (connection->first_output == Clock::time_point{}) connection->first_output = Clock::now();
        compute = impl_->cpu_lease(cancel);
        connection->output_bytes = add(connection->output_bytes, frame.payload.size());
        if (frame.type == pool::Message::End)
        {
            require(frame.payload.empty() && announced == chunks.size() && current + 1 == chunks.size() &&
                written.back() == output.back().size(), "compute encoded window is truncated");
            compute.reset();
            if (validator) { compute = impl_->cpu_lease(cancel); validator(output); compute.reset(); }
            cancel.check(); connection->delivered(); return output;
        }
        if (frame.type == pool::Message::Result)
        {
            pool::Reader input(frame.payload); auto const index = input.u32(); auto const bytes = input.u64(); input.finish();
            require(index == announced && index < chunks.size() && bytes && bytes <= SIZE_MAX &&
                (!index || written[index - 1] == output[index - 1].size()), "invalid compute encoded block announcement");
            size = add(size, bytes); require(size <= maximum, "compute encoded output exceeds owner admission");
            output[index] = std::vector<uint8_t>(size_t(bytes));
            require(output[index].capacity() == bytes, "compute output allocator exceeded exact owner admission");
            ++announced; compute.reset(); continue;
        }
        pool::Reader input(frame.payload); require(DataKind(input.u8()) == DataKind::Encoded, "unexpected compute codec output");
        auto const index = input.u32(); auto const offset = input.u64(); auto const bytes = input.bytes(impl_->config.tls.max_frame_bytes); input.finish();
        require(index < announced && (index == current || (index == current + 1 && written[current] == output[current].size())) &&
            offset == written[index] && !bytes.empty() && bytes.size() <= output[index].size() - written[index], "invalid compute encoded block order");
        current = index; std::copy(bytes.begin(), bytes.end(), output[index].begin() + written[index]); written[index] += bytes.size();
        compute.reset();
    }
}
WorkerStats WorkerClient::stats() const
{
    return {impl_->input_bytes.load(), impl_->output_bytes.load(), impl_->generations.load(),
        impl_->contributions.load(), impl_->codecs.load(), impl_->failures.load(),
        impl_->local_tasks.load(), impl_->task_elapsed_ns.load(),
        impl_->peak_generation_members.load(), impl_->peak_active_generation.load(), impl_->timed_generation_tasks.load(),
        {impl_->worker_ns.load(), impl_->send_ns.load(), impl_->frame_wait_ns.load(), impl_->receiver_ns.load(), impl_->frame_read_ns.load()},
        impl_->independent_tasks.load(), impl_->independent_members.load()};
}
unsigned WorkerClient::max_window_chunks() const { return impl_->config.max_window_chunks; }
WorkerGenerationEstimate WorkerClient::generation_estimate(uint8_t k, uint8_t strength, GenerationOutput output,
                                                          unsigned active, unsigned batch, uint64_t horizon_members) const
{
    (void)generation_capacity(k, strength);
    std::lock_guard lock(impl_->planning_mutex);
    return impl_->planning.estimate(k, strength, impl_->config.generation_kind, output, active, batch,
        Clock::now(), horizon_members);
}
uint64_t WorkerClient::max_task_bytes() const { return impl_->config.max_task_bytes; }
WorkerGenerationCapacity WorkerClient::generation_capacity(uint8_t k, uint8_t strength) const
{
    if (impl_->config.private_generation_capacity) return impl_->config.private_generation_capacity(k, strength);
    return impl_->capacity(k, strength);
}
WorkerGenerationCapacity WorkerClient::Impl::capacity(uint8_t k, uint8_t strength, Clock::time_point deadline,
                                                      pool::Lease const* assigned)
{
    if (config.local_only) return {};
    auto& hint = generation_capacity;
    std::unique_lock refresh(capacity_refresh_mutex, std::try_to_lock);
    uint64_t before;
    {
        std::lock_guard lock(planning_mutex);
        bool const same_shape = hint.k == k && hint.strength == strength;
        // Other plotting threads can keep using the cached advisory value
        // while one fetches the next snapshot. They still acquire real leases.
        if (!refresh.owns_lock()) return same_shape ? hint.value : WorkerGenerationCapacity{};
        if (same_shape && Clock::now() < hint.expires)
        {
            auto const& roster = hint.value.roster;
            if (!assigned || !roster || !roster->complete || std::any_of(roster->routes.begin(), roster->routes.end(),
                [&](auto const& route) { return route.identity == assigned->worker_identity &&
                    route.instance == assigned->worker_instance && route.endpoint == assigned->endpoint; })) return hint.value;
        }
        before = planning.selections();
    }
    WorkerGenerationCapacity result;
    try
    {
        WorkerCancellation cancel; cancel.deadline = std::min(deadline, Clock::now() + std::chrono::milliseconds(250));
        auto control = broker.with_deadline(cancel.deadline);
        pool::OwnerSession session;
        pool::Task prototype; prototype.compatibility = config.compatibility;
        prototype.kind = config.generation_kind; prototype.k = k; prototype.strength = strength;
        prototype.resources.cpu_threads = config.generation_threads;
        auto query = [&](bool refresh)
        {
            { auto lock = sequence_lock(cancel); prepare_session_locked(control, session, refresh); }
            auto const begin = Clock::now();
            result.batch = control.generation_hint(session, prototype, &result.slots, &result.availability, config.codec_compatibility, &result.roster);
            result.control_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
        };
        try { query(false); }
        // Only a changed broker epoch revives an idle owner's advisory
        // session. Polling cannot displace a replacement in this same epoch.
        catch (pool::BrokerRestarted const&) { query(true); }
    }
    catch (std::exception const&) {} // An optional hint never owns the actual lease.
    bool changed;
    {
        std::lock_guard lock(planning_mutex);
        if (result.roster && hint.value.roster && *hint.value.roster == *result.roster) result.roster = hint.value.roster;
        changed = hint.k != k || hint.strength != strength || hint.value.roster != result.roster ||
            hint.value.slots != result.slots || hint.value.batch != result.batch ||
            hint.value.availability.admission != result.availability.admission || bool(hint.value.control_ns) != bool(result.control_ns);
        planning.capacity(k, strength, result.roster, before);
        hint = {k, strength, Clock::now() + std::chrono::seconds(1), result};
    }
    refresh.unlock();
    if (changed && config.generation_capacity_changed)
        try { config.generation_capacity_changed(k, strength, result); } catch (...) {}
    return result;
}
unsigned WorkerClient::packed_generation_width(uint8_t k, uint8_t strength, unsigned ready, unsigned destinations,
                                              WorkerPackedRoute* batch_route) const
{
    if (batch_route) *batch_route = {};
    if (ready < 2) return ready;
    auto const hint = generation_capacity(k, strength);
    unsigned const width = std::min(ready, unsigned(hint.batch));
    if (!hint.slots || width < 2) return 1;
    if (hint.slots == 1) return width;
    std::lock_guard lock(impl_->planning_mutex);
    if (destinations <= hint.slots && batch_route &&
        impl_->planning.packed_before_singletons(k, strength, impl_->config.generation_kind, width, *batch_route)) return width;
    // Only form a packed batch when the other slots can serve independent
    // destinations. An unknown next route keeps the ordinary singletons.
    if (destinations < std::min(hint.slots, impl_->config.max_active_generation)) return 1;
    return impl_->planning.dispatch_width(k, strength, impl_->config.generation_kind, GenerationOutput::Packed,
        ready, destinations, hint.slots, width, 1, Clock::now(), 0, batch_route);
}
bool WorkerClient::codec_probe_available(uint8_t k, uint64_t timeout_ns) const
{
    std::lock_guard lock(impl_->planning_mutex);
    auto const& hint = impl_->generation_capacity;
    if (hint.k != k) return true;
    auto const now = Clock::now();
    // Encoding needs a broker request and a fresh authenticated data channel.
    // On a shared single worker, even the fastest observed connection must
    // fit the local window. An expired capacity sample never reserves a slot.
    if (hint.value.slots == 1 && !hint.value.availability.concurrent_codec && impl_->data_connect_ns &&
        now < hint.expires + std::chrono::milliseconds(impl_->config.tls.timeout_ms) &&
        (hint.value.control_ns >= timeout_ns || impl_->data_connect_ns >= timeout_ns - hint.value.control_ns)) return false;
    if (now >= hint.expires) return true;
    return hint.value.control_ns < timeout_ns && hint.value.availability.codec_available &&
        (!impl_->active_generation.load() || hint.value.availability.concurrent_codec);
}
unsigned WorkerClient::independent_contribution_width(uint8_t k, uint8_t strength, unsigned ready_members,
    unsigned destinations, bool dispatch) const
{
    if (!impl_->config.contribution_read_ahead || impl_->config.max_active_generation < 2 ||
        impl_->config.local_only || impl_->config.generation_kind == pool::Kind::GenerateCpu) return 1;
    auto const hint = generation_capacity(k, strength);
    if (!hint.slots || !ready_members) return 1;
    unsigned supported = hint.availability.independent_batch;
    // Other ready destinations can supply ordinary work to a worker that
    // cannot stream this batch. Share the batch across the slots that can.
    if (hint.slots > 1 && destinations > 1)
        supported = std::max(supported, hint.availability.independent_width(std::min(ready_members, impl_->config.max_active_generation)));
    auto const width = std::max(1u, std::min(supported, impl_->config.max_active_generation));
    // A lone slot can combine ready callers from different destinations. Share
    // several slots among the ready destinations for this plot shape.
    if (hint.slots == 1) return width;
    auto const batch = std::min(width, ready_members);
    auto const offered = destinations > 1 && batch > 1 ? hint.availability.independent_slots[batch] : hint.slots;
    if (!offered) return 1;
    auto const slots = std::min(offered, impl_->config.max_active_generation);
    auto const per_destination = 1 + (slots - 1) / std::max(1u, destinations);
    auto const fallback = std::min(width, 1 + (ready_members - 1) / per_destination);
    if (!dispatch || batch < 2 || slots != offered) return fallback;
    std::lock_guard lock(impl_->planning_mutex);
    return impl_->planning.dispatch_width(k, strength, pool::Kind::GenerateGpu, GenerationOutput::Contributions, ready_members, destinations, slots, width, fallback,
        Clock::now(), offered < hint.slots ? batch : 0);
}
bool WorkerClient::probe_contribution_route(uint8_t k, uint8_t strength) const
{
    auto const hint = generation_capacity(k, strength);
    if (hint.slots < 2) return false;
    std::lock_guard lock(impl_->planning_mutex);
    return impl_->planning.probe_contribution_route(k, strength);
}
unsigned WorkerClient::joint_contribution_members(uint8_t k, uint8_t strength, unsigned ready_members, unsigned destination_batch) const
{
    auto const hint = generation_capacity(k, strength);
    auto const available = std::min(ready_members, unsigned(hint.batch));
    std::lock_guard lock(impl_->planning_mutex);
    return impl_->planning.joint_contribution_members(k, strength, available, Clock::now(),
        destination_batch == ready_members ? available : destination_batch);
}
unsigned WorkerClient::suggested_batch_members(uint8_t k, uint8_t strength, unsigned maximum, GenerationOutput output) const
{
    (void)parameters(k, strength); if (!maximum) return 0;
    auto const hint = generation_capacity(k, strength);
    if (impl_->config.private_generation_capacity && hint.slots == 1) return std::max(1u, std::min(maximum, unsigned(hint.batch)));
    std::lock_guard lock(impl_->planning_mutex);
    auto const learned = impl_->planning.suggested(k, strength, impl_->config.generation_kind, output).batch;
    return std::max(1u, std::min({maximum, learned, unsigned(hint.batch)}));
}
unsigned WorkerClient::suggested_generation_concurrency(uint8_t k, uint8_t strength, unsigned maximum, GenerationOutput output) const
{
    (void)parameters(k, strength); if (!maximum) return 0;
    if (impl_->config.private_generation_capacity)
        return std::min({maximum, impl_->config.max_active_generation, std::max(1u, unsigned(generation_capacity(k, strength).slots))});
    if (impl_->config.local_only) return std::min(maximum, impl_->config.max_active_generation);
    auto const capacity = generation_capacity(k, strength);
    std::lock_guard lock(impl_->planning_mutex);
    auto concurrency = impl_->planning.suggested(k, strength, impl_->config.generation_kind, output).concurrency;
    if (capacity.roster && capacity.slots > 1)
    {
        // A long-lived owner can have learned low concurrency before a new
        // executor enrolls. Give unpriced slots a chance to deliver ready work
        // and use the measured useful receiver count as evidence arrives.
        auto const estimate = impl_->planning.estimate(k, strength, impl_->config.generation_kind, output, maximum, 1, Clock::now(), maximum);
        concurrency = std::max(concurrency, estimate.receivers + capacity.slots - std::min(capacity.slots, estimate.measured_slots));
    }
    return std::max(1u, std::min({maximum, impl_->config.max_active_generation, concurrency}));
}
uint64_t WorkerClient::memory_bytes()
{
    // sizeof(Impl) includes fixed worker/route planning histories and keys.
    // Also reserve hash nodes/buckets (including replacement growth), shared
    // retry records and allocator overhead. Session TLS is per-call.
    // Bound the retained route strings/rows and their replacement snapshot.
    return sizeof(Impl) + 2 * (65536 + pool::max_status_page_size * sizeof(pool::GenerationRoute)) +
        pool::BrokerClient::memory_bytes() + uint64_t(worker_detail::TaskRegistry::maximum) *
        (sizeof(worker_detail::TaskState) + sizeof(worker_detail::RetryHandle) + sizeof(std::shared_ptr<worker_detail::TaskState>) + 1024);
}
} // namespace group_grow
