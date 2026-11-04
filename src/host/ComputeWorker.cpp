#include "host/ComputeWorker.hpp"
#include "host/GroupCodec.hpp"
#include "host/ComputeThreads.hpp"
#include "plot/Plotter.hpp"
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
#include "host/GroupGpu.hpp"
#include "host/GpuBufferPool.hpp"
#include "host/GpuPipeline.hpp"
#endif

#include <condition_variable>
#include <csignal>
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
constexpr uint64_t mib = uint64_t(1) << 20;
constexpr size_t max_broker_routes = 8;
enum class Operation : uint8_t { Packed = 1, Contributions = 2, Encode = 3 };
enum class DataKind : uint8_t { Prelude = 1, Boundaries = 2, Words = 3, Contributions = 4, Encoded = 5 };
enum class WorkerError : uint8_t { Permanent = 0, Unsupported = 1, Transport = 2, Stale = 3 };
void write_worker_status(pool::Writer& out, WorkerStatus const& value)
{
    out.string(value.identity); out.string(value.node_id); out.string(value.instance); out.u8(uint8_t(value.admission));
    for (auto const& resources : {value.capacity, value.active})
    { out.u32(resources.cpu_threads); out.u32(resources.slots); out.u64(resources.host_bytes); out.u64(resources.gpu_bytes); }
    out.u64(value.accepted_tasks); out.u64(value.waiting_cpu); out.u64(value.baseline_host_bytes);
    out.u64(value.broker_limit); out.u64(value.known_brokers); out.u64(value.healthy_brokers); out.u64(value.deferred_brokers);
    out.u64(value.broker_transport_bytes); out.u8(value.broker_saturated);
}
WorkerStatus read_worker_status(pool::Reader& in)
{
    WorkerStatus value; value.identity = in.string(64); value.node_id = in.string(64); value.instance = in.string(64); value.admission = pool::WorkerAdmission(in.u8());
    for (auto resources : {&value.capacity, &value.active})
    { resources->cpu_threads = in.u32(); resources->slots = in.u32(); resources->host_bytes = in.u64(); resources->gpu_bytes = in.u64(); }
    value.accepted_tasks = in.u64(); value.waiting_cpu = in.u64(); value.baseline_host_bytes = in.u64();
    value.broker_limit = in.u64(); value.known_brokers = in.u64(); value.healthy_brokers = in.u64(); value.deferred_brokers = in.u64();
    value.broker_transport_bytes = in.u64(); auto const saturated = in.u8(); require(saturated <= 1, "invalid worker saturation flag");
    value.broker_saturated = saturated; in.finish(); return value;
}
class TaskCancelled : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
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
    return add(pool::BrokerClient::memory_bytes(1),
        multiply(max_broker_routes, sizeof(pool::Endpoint) + 2 * 254));
}
uint64_t service_overhead(WorkerConfig const& config, size_t brokers)
{
    return add(add(multiply(2 * config.capacity.slots + 2, pool::Channel::memory_bytes(config.tls.max_frame_bytes)),
        multiply(2 * config.capacity.slots, config.tls.max_frame_bytes)),
        add(add(mib, multiply(brokers, broker_registration_bytes())), CodecScheduler::memory_bytes(config.capacity.cpu_threads)));
}
pool::BrokerClient cleanup_control(pool::BrokerClient const& broker, pool::TlsOptions const& tls)
{
    // Task cancellation must still reach the broker after its compute/data
    // deadline. Cleanup owns a separate bounded control deadline and reuses
    // existing channels without replaying a possibly sent request.
    return broker.with_deadline(Clock::now() + std::chrono::milliseconds(std::min(100u, tls.timeout_ms)));
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
    Budget(uint64_t capacity, unsigned slots, uint64_t baseline)
        : capacity_(capacity), used_(baseline), baseline_(baseline), max_slots_(slots)
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
            if (bytes <= capacity_ - used_ && (!slot || slots_ < max_slots_))
            { used_ += bytes; if (retained) baseline_ += bytes; if (slot) ++slots_; return Lease(this, bytes, slot, retained); }
            changed_.wait_for(lock, std::chrono::milliseconds(50));
        }
    }
    uint64_t task_capacity() const
    { std::lock_guard lock(mutex_); return capacity_ - baseline_; }
    std::pair<uint64_t, unsigned> active() const
    { std::lock_guard lock(mutex_); return {used_ - baseline_, slots_}; }
    uint64_t baseline() const
    { std::lock_guard lock(mutex_); return baseline_; }
private:
    uint64_t capacity_, used_, baseline_;
    unsigned max_slots_, slots_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
};

// Embedded workers share the owner's scheduler, which may be wider than the
// worker offer. This independent weighted gate bounds all service CPU work.
class CpuAdmission
{
public:
    explicit CpuAdmission(unsigned capacity) : capacity_(capacity)
    { require(capacity, "worker CPU offer must be positive"); }
    class Lease
    {
    public:
        Lease(CpuAdmission& owner, unsigned permits, WorkerCancellation const& cancel)
            : owner_(owner), permits_(permits)
        {
            require(permits && permits <= owner_.capacity_, "worker CPU request exceeds offered capacity");
            std::unique_lock lock(owner_.mutex_); owner_.waiting_.push_back(this);
            try
            {
                for (;;)
                {
                    cancel.check();
                    if (owner_.waiting_.front() == this && permits_ <= owner_.capacity_ - owner_.active_)
                    { owner_.waiting_.pop_front(); owner_.active_ += permits_; break; }
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
};

class CpuWork
{
public:
    CpuWork(CpuAdmission& offered, CodecScheduler* shared, WorkerCancellation const& cancel,
            unsigned permits = 1)
        : offered_(offered, permits, cancel), shared_(shared, permits, [cancel] { cancel.check(); })
    { cancel.check(); }
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
std::string worker_compatibility()
{
#ifdef GROW_COMPUTE_COMPATIBILITY
    return "pos2-compute-v1:" GROW_COMPUTE_COMPATIBILITY;
#else
    // Builds must define the consumed-source identity before network use.
    return {};
#endif
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
        config.capacity.cpu_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    if (!config.capacity.host_bytes)
    {
#ifdef _WIN32
        MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
        require(::GlobalMemoryStatusEx(&memory), "cannot determine worker RAM; set --host-bytes");
        config.capacity.host_bytes = std::min(uint64_t(memory.ullTotalPhys) / 2, uint64_t(16) << 30);
#else
        auto const pages = ::sysconf(_SC_PHYS_PAGES), page_size = ::sysconf(_SC_PAGESIZE);
        require(pages > 0 && page_size > 0, "cannot determine worker RAM; set --host-bytes");
        config.capacity.host_bytes = std::min(uint64_t(pages) * uint64_t(page_size) / 2, uint64_t(16) << 30);
#endif
    }
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    if (config.backend != "cpu" && !config.capacity.gpu_bytes)
    {
        size_t available = 0, total = 0;
        if (pos2gpu::device_memory_probe(config.device, available, total))
            config.capacity.gpu_bytes = std::min(available, total) / 5 * 4;
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
        bool healthy = false;
        bool retired = false;
        Clock::time_point seen = Clock::now();
        Clock::time_point admitted = Clock::now(), rotation_attempt{};
        uint64_t revision = 0;
        pool::WorkerAdmission admission = pool::WorkerAdmission::Accepting;
        Registration(pool::Endpoint address, pool::TlsOptions const& tls)
            : endpoint(std::move(address)), routes{endpoint}, client(endpoint, tls, 1) {}
    };
    mutable std::mutex brokers_mutex;
    std::vector<std::shared_ptr<Registration>> brokers;
    size_t refresh_cursor = 0, discovery_cursor = 0, deferred_brokers = 0;
    uint64_t broker_transport_bytes = 0;
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
    uint64_t accepted_tasks = 0;
    std::atomic<bool> gpu_active{false};
    std::mutex jobs_mutex, heartbeat_mutex;
    std::condition_variable jobs_ready, heartbeat_ready;
    std::deque<std::packaged_task<void()>> gpu_jobs;
    std::string instance = pool::random_token();
    mutable std::mutex advertisement_mutex;
    pool::CapacityUpdate advertised_capacity;
    uint64_t next_capacity_revision = 1;
    std::atomic<uint64_t> retained_gpu_host{0};
    uint64_t service_baseline = 0;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    std::unordered_map<unsigned, GpuProducerPreflight> gpu_profiles;
#endif
    bool started = false, gpu_enabled = false;
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
    std::unique_ptr<GpuProducer> gpu;
    uint8_t gpu_k = 0, gpu_strength = 0;
    unsigned gpu_threads = 0;
    Budget::Lease gpu_retained;
#endif

    explicit Impl(WorkerConfig options) : config(std::move(options)), offered_cpu(config.capacity.cpu_threads)
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
        budget = std::make_unique<Budget>(config.capacity.host_bytes, config.capacity.slots, baseline);
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
                    try { gpu_profiles.emplace(k, GpuProducer::preflight(k, 2, options)); }
                    catch (pos2gpu::InsufficientVramError const&) {}
                }
                require(config.backend != "gpu" || !gpu_profiles.empty(), "GPU offer cannot fit any supported generation tier");
                if (gpu_profiles.empty()) gpu_enabled = false;
#endif
            });
            auto const capacity = current_capacity();
            advertised_capacity = {capacity, 1, profiles(capacity.host_bytes)};
        }
        catch (...)
        { stopping.store(true); jobs_ready.notify_all(); if (gpu_thread.joinable()) gpu_thread.join(); throw; }
    }
    ~Impl() { stop(); }

    class Acceptance
    {
    public:
        explicit Acceptance(Impl& owner) : owner_(owner)
        {
            std::lock_guard lock(owner_.admission_mutex);
            if (owner_.admission != pool::WorkerAdmission::Accepting || owner_.stopping.load())
                throw pool::TransientTransport("worker is not accepting new work");
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
    WorkerStatus status() const
    {
        WorkerStatus value; value.identity = config.tls.cert_file.empty() ? "" : pool::certificate_identity(config.tls.cert_file); value.instance = instance;
        if (!config.tls.cert_file.empty()) value.node_id = pool::certificate_node_id(config.tls.cert_file);
        value.capacity = current_capacity();
        auto const memory = budget->active(); auto const cpu = offered_cpu.active();
        value.active = {cpu.first, memory.second, memory.first, gpu_active.load() ? config.capacity.gpu_bytes : 0};
        value.waiting_cpu = cpu.second; value.baseline_host_bytes = budget->baseline();
        { std::lock_guard lock(admission_mutex); value.admission = admission; value.accepted_tasks = accepted_tasks; }
        { std::lock_guard lock(brokers_mutex); value.known_brokers = brokers.size(); value.deferred_brokers = deferred_brokers;
          for (auto const& broker : brokers) value.healthy_brokers += broker->healthy; }
        value.broker_limit = config.max_brokers; value.broker_transport_bytes = broker_transport_bytes;
        value.broker_saturated = value.deferred_brokers != 0; return value;
    }
    WorkerStatus control(WorkerControl operation)
    {
        {
            std::lock_guard lock(admission_mutex);
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

    pool::Resources current_capacity() const
    {
        auto capacity = config.capacity;
        if (!gpu_enabled) capacity.gpu_bytes = 0;
        capacity.host_bytes = budget->task_capacity();
        require(capacity.host_bytes, "worker persistent state consumes its offered host capacity");
        return capacity;
    }
    pool::CapacityUpdate capacity_update(bool force = false)
    {
        std::lock_guard lock(advertisement_mutex);
        auto const capacity = current_capacity();
        if (!force && capacity == advertised_capacity.capacity)
            return {capacity, advertised_capacity.revision, {}};
        require(next_capacity_revision < UINT64_MAX, "worker capacity revision exhausted");
        return {capacity, ++next_capacity_revision, profiles(capacity.host_bytes)};
    }
    void commit_capacity(pool::CapacityUpdate const& update)
    {
        std::lock_guard lock(advertisement_mutex);
        if (update.revision > advertised_capacity.revision) advertised_capacity = update;
    }
    pool::BrokerClient control_for(pool::Lease const& lease) const
    {
        std::lock_guard lock(brokers_mutex);
        std::shared_ptr<Registration> selected;
        for (auto const& broker : brokers)
            if (!broker->epoch.empty() && broker->epoch == lease.epoch && broker->identity == lease.broker_identity &&
                (!selected || (broker->healthy && !selected->healthy) ||
                    (broker->healthy == selected->healthy && broker->seen > selected->seen))) selected = broker;
        if (selected) return selected->client;
        throw pool::StaleLease("worker does not recognize the lease's broker epoch");
    }
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
        auto update = capacity_update(); commit_capacity(update);
        std::vector<std::future<void>> requests;
        for (auto const& broker : snapshot) requests.push_back(std::async(std::launch::async, [this, broker]
        {
            if (stopping.load()) return;
            try
            {
                pool::Endpoint address, advertised_endpoint;
                uint64_t revision;
                pool::WorkerAdmission previous_admission;
                { std::lock_guard lock(brokers_mutex); address = broker->endpoint; advertised_endpoint = broker->advertised_endpoint;
                  revision = broker->revision; previous_admission = broker->admission; }
                auto client = broker->client.with_deadline(Clock::now() + std::chrono::milliseconds(config.broker_refresh_ms));
                auto value = offer();
                if (value.endpoint.host == "0.0.0.0" || value.endpoint.host == "::")
                    value.endpoint.host = pool::local_address(address);
                if (revision == value.capacity_revision && value.endpoint == advertised_endpoint &&
                    client.heartbeat(instance, value.capacity_revision))
                {
                    if (previous_admission != value.admission) client.worker_admission(instance, value.admission);
                    std::lock_guard lock(brokers_mutex); broker->seen = Clock::now(); broker->healthy = true;
                    broker->admission = value.admission; return;
                }
                if (!advertised_endpoint.host.empty() && value.endpoint != advertised_endpoint)
                {
                    auto revised = capacity_update(true); commit_capacity(revised);
                    auto route = value.endpoint; value = offer(); value.endpoint = std::move(route);
                }
                std::string peer, node;
                auto epoch = client.register_worker(value, &peer, &node);
                std::lock_guard lock(brokers_mutex);
                auto same = std::find_if(brokers.begin(), brokers.end(), [&](auto const& registered)
                { return registered != broker && !registered->epoch.empty() && registered->epoch == epoch && registered->identity == peer; });
                if (same != brokers.end())
                {
                    auto& registered = **same;
                    if (registered.routes.size() < max_broker_routes &&
                        std::find(registered.routes.begin(), registered.routes.end(), address) == registered.routes.end())
                        registered.routes.push_back(address);
                    if (!registered.healthy)
                    {
                        registered.client.reroute(address); registered.endpoint = address;
                        registered.route_cursor = registered.routes.size() - 1;
                    }
                    registered.seen = Clock::now(); registered.healthy = true;
                    registered.revision = value.capacity_revision; registered.admission = value.admission;
                    registered.advertised_endpoint = value.endpoint;
                    std::erase(brokers, broker); return;
                }
                broker->epoch = std::move(epoch); broker->revision = value.capacity_revision;
                broker->identity = peer; broker->node_id = node;
                broker->admission = value.admission; broker->advertised_endpoint = value.endpoint;
                broker->seen = Clock::now(); broker->healthy = true;
            }
            catch (...)
            {
                std::lock_guard lock(brokers_mutex); broker->healthy = false;
                if (broker->routes.size() > 1)
                {
                    broker->route_cursor = (broker->route_cursor + 1) % broker->routes.size();
                    broker->endpoint = broker->routes[broker->route_cursor]; broker->client.reroute(broker->endpoint);
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

    std::vector<pool::GenerationProfile> profiles(uint64_t task_capacity) const
    {
        std::vector<pool::GenerationProfile> result;
        auto const retained = config.capacity.host_bytes - service_baseline - task_capacity;
        for (uint8_t k = 18; k <= 28; k += 2)
            for (uint8_t strength = STRENGTH_MIN;
                 strength <= std::min<unsigned>(STRENGTH_MAX, k - PlotProofParams::num_section_bits_for_k(k) - 1); ++strength)
            {
                uint64_t const raw = multiply(fragment_count_bound(k, strength), sizeof(uint64_t));
                pool::GenerationProfile cpu; cpu.kind = pool::Kind::GenerateCpu; cpu.k = k; cpu.strength = strength;
                cpu.cpu_threads = config.generation_threads;
                cpu.resources = {config.generation_threads, 1,
                    add(add(worker_cpu_generation_bytes(k, strength), boundary_bytes(k)), config.max_task_bytes), 0};
                cpu.member_host_bytes = add(raw, boundary_bytes(k));
                cpu.max_batch_members = config.max_batch_members;
                result.push_back(cpu);
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
                if (auto found = gpu_profiles.find(k); found != gpu_profiles.end())
                {
                    auto gpu = cpu; gpu.kind = pool::Kind::GenerateGpu;
                    auto const total = add(add(found->second.owning_host_bytes, boundary_bytes(k)), config.max_task_bytes);
                    gpu.resources.host_bytes = total > retained ? total - retained : boundary_bytes(k);
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
            { std::unique_lock lock(jobs_mutex); jobs_ready.wait(lock, [&] { return stopping.load() || !gpu_jobs.empty(); });
              if (gpu_jobs.empty()) break; job = std::move(gpu_jobs.front()); gpu_jobs.pop_front(); }
            job();
        }
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        gpu.reset(); gpu_retained.release();
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
        WorkerPackedSink const& packed_sink, WorkerContributionSink const& contribution_sink,
        WorkerCancellation cancel, bool use_gpu, bool task_slot = true, pool::Resources const* grant = nullptr,
        std::function<void(WorkerTaskAdmission const&)> const& admit = {})
    {
        cancel = cancellation(cancel); cancel.check();
        require(!ids.empty() && ids.size() <= config.max_batch_members &&
            (!packed_sink || ids.size() == 1), "worker generation batch exceeds configured limit");
        auto const bound = fragment_count_bound(k, strength);
        uint64_t const raw_bytes = multiply(bound, sizeof(uint64_t));
        uint64_t const indexes = multiply(boundary_bytes(k), ids.size());
        uint64_t peak = worker_cpu_task_bytes(k, strength, ids.size(), config.max_task_bytes);
        uint64_t const prior_retained = retained_gpu_host.load();
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
        if (use_gpu)
        {
            auto const found = gpu_profiles.find(k);
            if (found == gpu_profiles.end()) throw pool::UnsupportedResources("GPU task has no admitted generation profile");
            peak = add(add(add(found->second.owning_host_bytes,
                multiply(raw_bytes, ids.size() - 1)), indexes), config.max_task_bytes);
        }
#endif
        unsigned const threads = grant ? grant->cpu_threads : config.generation_threads;
        require(threads && threads <= config.capacity.cpu_threads, "generation CPU allowance exceeds worker offer");
        uint64_t const dynamic_peak = use_gpu && peak > prior_retained ? peak - prior_retained : use_gpu ? indexes : peak;
        if (grant && dynamic_peak > grant->host_bytes)
            throw pool::UnsupportedResources("generation model exceeds leased host memory");
        if (admit)
        {
            WorkerTaskAdmission value; value.worker = status(); value.local_fallback = true;
            value.task.compatibility = worker_compatibility(); value.task.kind = use_gpu ? pool::Kind::GenerateGpu : pool::Kind::GenerateCpu;
            value.task.k = k; value.task.strength = strength; value.task.work_units = ids.size();
            value.task.resources = {threads, 1, dynamic_peak, use_gpu ? config.capacity.gpu_bytes : 0};
            admit(value); cancel.check();
        }
        auto memory = budget->acquire(dynamic_peak, task_slot, cancel);
        auto execute = [&]
        {
            uint64_t task_retained = 0;
            uint64_t generated_retained = 0;
            std::vector<std::vector<uint64_t>> owned;
            std::vector<std::span<uint64_t const>> members; members.reserve(ids.size());
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
                    options.host_threads = threads;
                    CpuWork compute(offered_cpu, scheduler.get(), cancel, threads);
                    try { gpu = std::make_unique<GpuProducer>(k, strength, options); }
                    catch (std::exception const& error) { throw pool::UnsupportedResources(error.what()); }
                    gpu_k = k; gpu_strength = strength; gpu_threads = threads;
                }
                auto const generation = gpu->host_generation_peak_bytes();
                peak = add(add(add(generation, multiply(raw_bytes, ids.size() - 1)), indexes), config.max_task_bytes);
                if (grant && peak > add(grant->host_bytes, prior_retained))
                    throw pool::UnsupportedResources("GPU generation model exceeds leased host memory");
                memory.resize(peak > gpu_retained.bytes ? peak - gpu_retained.bytes : indexes);
                task_retained = gpu_retained.bytes;
                owned.reserve(ids.size());
                for (auto const& id : ids)
                {
                    cancel.check();
                    CpuWork compute(offered_cpu, scheduler.get(), cancel, threads);
                    if (ids.size() == 1) { borrowed.emplace(gpu->generate_borrowed(id)); members.push_back(borrowed->fragments()); }
                    else { owned.push_back(gpu->generate(id)); members.push_back(owned.back()); }
                    require(members.back().size() <= bound, "worker GPU output exceeds admitted fragment bound");
                }
                generated_retained = gpu->retained_host_bytes();
            }
            else
#endif
            {
                owned.reserve(ids.size());
                for (auto const& id : ids)
                {
                    cancel.check();
                    CpuWork compute(offered_cpu, scheduler.get(), cancel, threads);
                    worker_detail::CpuLimit limit(threads);
                    WorkerReferencePlotter::Options options{}; options.validate = false; options.verbose = false;
                    CancelProgress progress(cancel); options.sink = &progress;
                    owned.push_back(WorkerReferencePlotter(parameters(k, strength, id)).run(options).t3_proof_fragments);
                    cancel.check();
                    require(owned.back().size() <= bound, "worker CPU output exceeds admitted fragment bound");
                    members.push_back(owned.back());
                }
            }
            if (packed_sink) packed(k, members.front(), packed_sink, cancel);
            else
            {
                size_t const chunks = size_t(PlotGroupFile::getChunkCountForK(k));
                std::vector<std::vector<uint32_t>> boundaries;
                boundaries.reserve(members.size()); uint64_t actual_raw = 0, maximum_work = 0;
                {
                    CpuWork compute(offered_cpu, scheduler.get(), cancel);
                    unsigned const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
                    for (auto raw : members)
                    {
                        require(raw.size() <= UINT32_MAX && std::is_sorted(raw.begin(), raw.end()), "invalid worker generation output");
                        actual_raw = add(actual_raw, multiply(raw.size(), sizeof(uint64_t)));
                        std::vector<uint32_t> index(chunks + 1); size_t cursor = 0;
                        for (size_t c = 0; c < chunks; ++c)
                        { index[c] = uint32_t(cursor); while (cursor < raw.size() && (raw[cursor] >> width) == c) ++cursor; }
                        index.back() = uint32_t(cursor); require(cursor == raw.size(), "out-of-range worker member");
                        boundaries.push_back(std::move(index));
                    }
                    for (size_t c = 0; c < chunks; ++c)
                    {
                        uint64_t count = 0; for (auto const& index : boundaries) count = add(count, index[c + 1] - index[c]);
                        maximum_work = std::max(maximum_work, ChunkCodec::estimate_work(k, 0, count));
                    }
                }
                if (maximum_work > config.max_task_bytes)
                    throw pool::UnsupportedResources("initial contribution codec exceeds configured task scratch limit");
                // A borrowed singleton already lives in the producer. Owning
                // batches retain their vectors beside its persistent tables.
                auto const owned_raw = use_gpu && ids.size() == 1 ? 0 : actual_raw;
                uint64_t const needed = add(add(add(owned_raw, indexes), maximum_work), generated_retained);
                if (grant && needed > add(grant->host_bytes, use_gpu ? prior_retained : 0))
                    throw pool::UnsupportedResources("initial contribution codec exceeds leased host memory");
                auto const dynamic_needed = needed > task_retained ? needed - task_retained : indexes;
                if (dynamic_needed > memory.bytes) memory.resize(dynamic_needed);
                auto codec = [&]
                {
                    CpuWork compute(offered_cpu, scheduler.get(), cancel);
                    return ChunkCodec(k, maximum_work, scheduler);
                }();
                std::vector<std::span<uint64_t const>> regions; regions.reserve(members.size());
                for (size_t c = 0; c < chunks; ++c)
                {
                    cancel.check();
                    std::span<uint8_t const> bytes;
                    {
                        CpuWork compute(offered_cpu, scheduler.get(), cancel);
                        regions.clear();
                        for (size_t m = 0; m < members.size(); ++m)
                            regions.push_back(members[m].subspan(boundaries[m][c], boundaries[m][c + 1] - boundaries[m][c]));
                        uint64_t count = 0; for (auto const region : regions) count = add(count, region.size());
                        codec.prepare(ChunkCodec::estimate_work(k, 0, count));
                        bytes = codec.encode(c, {}, 0, regions);
                    }
                    cancel.check(); contribution_sink(c, bytes, uint16_t(members.size()));
                }
            }
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
            try { execute(); }
            catch (...)
            {
#if defined(GROW_GPU_BACKEND) && GROW_GPU_BACKEND
                gpu.reset();
                if (gpu_retained.budget) gpu_retained.resize(GpuProducer::runtime_host_allowance_bytes());
                retained_gpu_host.store(gpu_retained.bytes);
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
        std::atomic<bool> cancelled{false};
        std::exception_ptr renewal_failure;
        bool begun = false, output_started = false;
        bool generation_failpoint_checked = false;
        std::mutex output_mutex;
        bool keepalive_enabled = false;
        std::optional<Acceptance> accepted;
        auto send = [&](pool::Message type, std::span<uint8_t const> bytes)
        {
            std::lock_guard lock(output_mutex); channel.send(type, bytes);
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
            require(lease->worker_instance == instance && lease->task.compatibility == worker_compatibility(), "worker lease targets another executor or codec");
            auto const task_timeout = in.u32();
            require(task_timeout && task_timeout <= config.task_timeout_ms, "worker task timeout exceeds configured limit");
            WorkerCancellation cancel{&cancelled, Clock::now() + std::chrono::milliseconds(task_timeout), &stopping};
            channel.set_deadline(cancel.deadline);
            broker = control_for(*lease);
            auto active = broker->with_deadline(cancel.deadline);
            accepted.emplace(*this);
            if (!active.begin(*lease, channel.peer_identity())) throw pool::StaleLease("worker rejected stale lease");
            begun = true;
            Renewal renewal(active, *lease, cancelled, renewal_failure, [&]
            {
                std::lock_guard lock(output_mutex);
                if (keepalive_enabled) channel.send(pool::Message::Reply, {});
            }, config.tls.timeout_ms);
            auto task_slot = budget->acquire(0, true, cancel);
            auto const operation = Operation(in.u8()); uint8_t const k = in.u8(), strength = in.u8();
            require(k == lease->task.k && strength == lease->task.strength, "worker parameters differ from lease");
            if (operation == Operation::Packed || operation == Operation::Contributions)
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
                          out.u32(uint32_t(part.size())); for (auto value : part) out.u64(value); }
                        cancel.check(); send(pool::Message::Data, out.data()); words = words.subspan(part.size());
                    }
                };
                pool::Writer output(config.tls.max_frame_bytes); output.u8(uint8_t(DataKind::Contributions));
                bool first_flush = true;
                auto flush = [&]
                {
                    if (output.data().size() <= 1) return;
                    cancel.check(); output_started = true; send(pool::Message::Data, output.data());
                    if (std::exchange(first_flush, false))
                    {
                        auto const failpoint = std::getenv("XCHPLOT2_GROW_FAILPOINT");
                        if (failpoint && std::string_view(failpoint) == "worker-contribution-partial") std::_Exit(86);
#ifndef _WIN32
                        if (failpoint && std::string_view(failpoint) == "worker-contribution-paused") std::raise(SIGSTOP);
#endif
                    }
                    output = pool::Writer(config.tls.max_frame_bytes); output.u8(uint8_t(DataKind::Contributions));
                };
                auto contribution_sink = [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
                {
                    require(bytes.size() + 32 <= config.tls.max_frame_bytes, "worker contribution exceeds transport frame");
                    if (output.data().size() + bytes.size() + 14 > config.tls.max_frame_bytes) flush();
                    CpuWork compute(offered_cpu, scheduler.get(), cancel);
                    output.u64(ordinal); output.u16(members); output.bytes(bytes);
                };
                generation(k, strength, ids, operation == Operation::Packed ? WorkerPackedSink(packed_sink) : WorkerPackedSink{},
                    operation == Operation::Contributions ? WorkerContributionSink(contribution_sink) : WorkerContributionSink{}, cancel, use_gpu, false, &lease->task.resources);
                flush();
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
            { std::lock_guard lock(output_mutex); keepalive_enabled = false; channel.send(pool::Message::End, {}); }
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
              auto update = capacity_update(); cleanup.reject(*lease, pool::FailureCode::UnsupportedResources, update); commit_capacity(update); }
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
            auto cleanup = worker_detail::cleanup_control(*broker, config.tls);
            try { auto update = capacity_update(); cleanup.finish(*lease, update); commit_capacity(update); }
            catch (...)
            {
                try { cleanup.finish(*lease); } catch (...) {}
            }
        }
    }

    pool::WorkerOffer offer() const
    {
        pool::WorkerOffer value; value.instance = instance; value.compatibility = worker_compatibility();
        value.endpoint = config.advertise.port ? config.advertise : listener->endpoint();
        if (!config.server_name.empty()) value.endpoint.server_name = config.server_name;
        { std::lock_guard lock(advertisement_mutex);
          value.capacity = advertised_capacity.capacity; value.capacity_revision = advertised_capacity.revision;
          if (advertised_capacity.generation_profiles) value.generation_profiles = *advertised_capacity.generation_profiles; }
        value.capabilities = pool::capability(pool::Kind::Pack) | pool::capability(pool::Kind::GenerateCpu);
        value.generation_threads = config.generation_threads;
        value.codec_limits = {config.max_window_chunks, config.max_task_bytes};
        if (gpu_enabled) value.capabilities |= pool::capability(pool::Kind::GenerateGpu);
        { std::lock_guard lock(admission_mutex); value.admission = admission; }
        value.min_k = 18; value.max_k = 28; return value;
    }
    void start()
    {
        require(!started && !stopping.load(), "worker service already started or stopped");
        require(!worker_compatibility().empty(), "worker build has no compute compatibility identity");
        if (!config.identity_reservation) config.identity_reservation = WorkerService::reserve_identity(config.tls.cert_file);
        require(config.identity_reservation->identity == pool::certificate_identity(config.tls.cert_file),
            "compute identity reservation belongs to another certificate");
        listener = std::make_unique<pool::Listener>(config.listen, config.tls, config.capacity.slots);
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
{ return lease.worker_instance == impl_->instance && lease.task.compatibility == worker_compatibility(); }
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
void WorkerService::generate_assigned(pool::Lease const& lease,
    std::span<std::array<uint8_t, 32> const> ids, WorkerPackedSink const& packed,
    WorkerContributionSink const& contributions, WorkerCancellation cancel)
{
    require(owns(lease) && lease.task.kind != pool::Kind::Pack && lease.task.work_units == ids.size(),
        "local generation lease targets another task or executor");
    cancel = impl_->cancellation(cancel);
    auto control = impl_->control_for(lease);
    auto active = control.with_deadline(cancel.deadline);
    Impl::Acceptance accepted(*impl_);
    if (!active.begin(lease, pool::certificate_identity(impl_->config.tls.cert_file)))
        throw pool::StaleLease("local worker rejected stale lease");
    std::exception_ptr renewal_failure;
    try
    {
        std::atomic<bool> cancelled{false};
        WorkerCancellation running{&cancelled, cancel.deadline, &impl_->stopping, cancel.stop};
        Renewal renewal(active, lease, cancelled, renewal_failure, [cancel] { cancel.check(); }, impl_->config.tls.timeout_ms);
        auto checked_packed = [&](PackedFragmentTile const& tile, uint64_t total)
        { cancel.check(); running.check(); packed(tile, total); };
        auto checked_contribution = [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
        { cancel.check(); running.check(); contributions(ordinal, bytes, members); };
        impl_->generation(lease.task.k, lease.task.strength, ids,
            packed ? WorkerPackedSink(checked_packed) : WorkerPackedSink{},
            contributions ? WorkerContributionSink(checked_contribution) : WorkerContributionSink{},
            running, lease.task.kind == pool::Kind::GenerateGpu, true, &lease.task.resources);
        cancel.check(); running.check();
    }
    catch (std::exception const& error)
    {
        auto failure = std::current_exception();
        try
        {
            auto update = impl_->capacity_update();
            auto cleanup = worker_detail::cleanup_control(control, impl_->config.tls);
            if (dynamic_cast<pool::UnsupportedResources const*>(&error))
                cleanup.reject(lease, pool::FailureCode::UnsupportedResources, update);
            else cleanup.finish(lease, update);
            impl_->commit_capacity(update);
        }
        catch (...) { /* The next heartbeat publishes the current capacity. */ }
        std::rethrow_exception(task_failure(failure, renewal_failure));
    }
    try { auto update = impl_->capacity_update(); worker_detail::cleanup_control(control, impl_->config.tls).finish(lease, update); impl_->commit_capacity(update); }
    catch (...) { /* The next heartbeat publishes the current capacity. */ }
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
    Impl::Acceptance accepted(*impl_);
    if (!active.begin(lease, pool::certificate_identity(impl_->config.tls.cert_file)))
        throw pool::StaleLease("local worker rejected stale lease");
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
    require(kind != pool::Kind::GenerateGpu || impl_->gpu_enabled, "local GPU generation was not offered");
    Impl::Acceptance accepted(*impl_);
    impl_->generation(k, strength, {&id, 1}, sink, {}, cancel, kind != pool::Kind::GenerateCpu && impl_->gpu_enabled, true, nullptr, admit);
}
void WorkerService::contributions_local(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
    WorkerContributionSink const& sink, WorkerCancellation cancel, pool::Kind kind,
    std::function<void(WorkerTaskAdmission const&)> const& admit)
{
    require(kind != pool::Kind::GenerateGpu || impl_->gpu_enabled, "local GPU generation was not offered");
    Impl::Acceptance accepted(*impl_);
    impl_->generation(k, strength, ids, {}, sink, cancel, kind != pool::Kind::GenerateCpu && impl_->gpu_enabled, true, nullptr, admit);
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
    explicit GenerationPlanning(unsigned timeout_ms) : lifetime_(timeout_ms) {}
    uint64_t selected(WorkerTaskObservation const& value, pool::Endpoint const& route, Clock::time_point now = Clock::now())
    {
        if (!valid(value, route)) return 0;
        require(selection_ != UINT64_MAX, "compute planning selection sequence exhausted");
        auto const selection = ++selection_; retire(now);
        auto const peer = key(value, route);
        auto at = current(peer);
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
            at->peer = peer; at->first = selection;
        }
        at->last = selection; at->seen = now; at->sequence = sequence_;
        return selection;
    }
    void observed(WorkerTaskObservation const& value, pool::Endpoint const& route, uint64_t selection,
                  Clock::time_point now = Clock::now())
    {
        if (!valid(value, route)) return;
        retire(now);
        auto const peer = key(value, route); auto active = current(peer);
        if (!accepts(active, peer, selection)) return;
        if (sequence_ == UINT64_MAX) { count_ = peer_count_ = 0; sequence_ = 0; return; }
        active->sequence = ++sequence_; active->seen = now; retire(now);
        if (!value.execution_ns || !value.transfer_ns) return;
        auto at = find(value, peer);
        if (at == histories_.begin() + count_)
        {
            if (count_ == maximum)
                at = std::min_element(histories_.begin(), histories_.end(),
                    [](History const& a, History const& b) { return a.sequence < b.sequence; });
            else at = histories_.begin() + count_++;
            *at = History{}; at->peer = peer; at->kind = value.kind; at->k = value.k; at->strength = value.strength;
        }
        at->seen = now; at->sequence = sequence_;
        int const direction = value.execution_ns > value.transfer_ns ? 1 : value.execution_ns < value.transfer_ns ? -1 : 0;
        at->votes = direction && direction == at->direction ? at->votes + 1 : unsigned(direction != 0);
        at->direction = direction;
        // A single noisy completion must not change every following request.
        if (at->votes < 3) return;
        at->votes = 0;
        if (direction > 0)
        { at->plan.batch = std::min(unsigned(pool::max_generation_batch_members), at->plan.batch + 1);
          at->plan.concurrency = std::min(64u, at->plan.concurrency + 1); }
        else
        { at->plan.batch = std::max(1u, at->plan.batch / 2); at->plan.concurrency = std::max(1u, at->plan.concurrency / 2); }
    }
    void failed(WorkerTaskObservation const& value, pool::Endpoint const& route, uint64_t selection,
                Clock::time_point now = Clock::now())
    {
        if (!valid(value, route)) return;
        retire(now); auto const peer = key(value, route);
        if (!accepts(current(peer), peer, selection)) return;
        auto at = find(value, peer);
        if (at == histories_.begin() + count_) return;
        at->plan.batch = std::max(1u, at->plan.batch / 2); at->plan.concurrency = std::max(1u, at->plan.concurrency / 2);
        at->votes = 0; at->direction = 0;
    }
    Plan suggested(uint8_t k, uint8_t strength, pool::Kind kind, Clock::time_point now = Clock::now())
    {
        retire(now);
        std::array<unsigned, maximum> batches{}, concurrency{};
        std::array<History const*, maximum> representatives{}; size_t count = 0;
        for (size_t i = 0; i < count_; ++i)
        {
            auto const& item = histories_[i];
            if (item.k != k || item.strength != strength || (kind != pool::Kind::GenerateAny && item.kind != kind)) continue;
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
        Peer peer;
        pool::Kind kind = pool::Kind::GenerateCpu;
        uint8_t k = 0, strength = 0;
        Plan plan;
        uint64_t sequence = 0;
        Clock::time_point seen{};
        int direction = 0;
        unsigned votes = 0;
    };
    struct ActivePeer
    {
        Peer peer;
        uint64_t first = 0, last = 0, sequence = 0;
        Clock::time_point seen{};
    };
    static bool valid(WorkerTaskObservation const& value, pool::Endpoint const& route)
    {
        return (value.kind == pool::Kind::GenerateCpu || value.kind == pool::Kind::GenerateGpu) &&
            value.k >= 18 && value.k <= 28 && !(value.k & 1) && value.strength <= 32 && value.work_units &&
            !value.worker_identity.empty() && value.worker_identity.size() <= 64 &&
            !value.worker_instance.empty() && value.worker_instance.size() <= 64 &&
            route.host.size() <= 253 && route.server_name.size() <= 253;
    }
    template<size_t Size> static std::array<char, Size> text(std::string const& value)
    { std::array<char, Size> result{}; std::copy(value.begin(), value.end(), result.begin()); return result; }
    static Peer key(WorkerTaskObservation const& value, pool::Endpoint const& route)
    { return {text<64>(value.worker_identity), text<64>(value.worker_instance), text<253>(route.host),
        text<253>(route.server_name), route.port, value.local_direct}; }
    std::array<History, maximum>::iterator find(WorkerTaskObservation const& value, Peer const& peer)
    {
        return std::find_if(histories_.begin(), histories_.begin() + count_, [&](History const& item)
        { return item.peer == peer && item.kind == value.kind && item.k == value.k && item.strength == value.strength; });
    }
    std::array<ActivePeer, maximum>::iterator current(Peer const& peer)
    {
        return std::find_if(peers_.begin(), peers_.begin() + peer_count_, [&](ActivePeer const& item)
        { return item.peer.worker == peer.worker; });
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
        auto end = std::remove_if(peers_.begin(), peers_.begin() + peer_count_, [&](ActivePeer const& item)
        { return sequence_ - item.sequence >= maximum || now - item.seen >= lifetime_; });
        peer_count_ = size_t(end - peers_.begin());
        erase([&](History const& item)
        { auto at = current(item.peer); return sequence_ - item.sequence >= maximum || now - item.seen >= lifetime_ ||
            at == peers_.begin() + peer_count_ || at->peer != item.peer; });
    }
    std::array<History, maximum> histories_{};
    std::array<ActivePeer, maximum> peers_{};
    size_t count_ = 0, peer_count_ = 0;
    uint64_t sequence_ = 0, selection_ = 0;
    std::chrono::milliseconds lifetime_;
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
    worker_detail::GenerationPlanning planning;
    explicit Impl(WorkerClientConfig options) : config(std::move(options)), broker(config.broker, config.tls),
        generation_admission(0, config.max_active_generation, 0), planning(config.task_timeout_ms)
    {
        if (config.compatibility.empty()) config.compatibility = worker_compatibility();
        require(!config.compatibility.empty() && !config.owner_id.empty(), "compute client requires a compatibility identity and persistent owner ID");
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
    std::atomic<uint64_t> input_bytes{0}, output_bytes{0}, generations{0}, contributions{0}, codecs{0}, failures{0};
    std::atomic<uint64_t> local_tasks{0}, task_elapsed_ns{0};
    std::atomic<uint64_t> active_generation{0}, peak_active_generation{0}, peak_generation_members{0};
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
            peak = impl.peak_generation_members.load();
            while (peak < members && !impl.peak_generation_members.compare_exchange_weak(peak, members)) {}
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
        catch (std::exception const& error) { throw WorkerAdmissionRejected(error.what()); }
        catch (...) { throw WorkerAdmissionRejected("compute admission policy failed"); }
    }
    std::function<void(WorkerTaskAdmission const&)> local_admission() const
    {
        if (!config.admit_task) return {};
        return [this](WorkerTaskAdmission value)
        {
            value.owner_identity = pool::certificate_identity(config.tls.cert_file); value.owner_id = config.owner_id;
            admit(value);
        };
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
        bool completed = false, resource_rejected = false, declined = false, local_direct = false;
        std::shared_ptr<WorkerService> local_service;
        Clock::time_point selected_at = Clock::now(), first_output{};
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
            return result;
        }
        ~Connection()
        {
            impl.input_bytes.fetch_add(input_bytes); impl.output_bytes.fetch_add(output_bytes);
            if (completed)
            {
                if (operation == Operation::Packed) impl.generations.fetch_add(1);
                else if (operation == Operation::Contributions) impl.contributions.fetch_add(1);
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
                if (declined) try { impl.cleanup_control().cancel(session, lease.task.task_id, lease.task.attempt); } catch (...) {}
                else if (!resource_rejected) try { impl.cleanup_control().abandon(lease); } catch (...) {}
            }
        }
        void delivered()
        {
            auto const elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
            if (elapsed > 0) try { control.delivered(lease, {uint64_t(elapsed),
                local_direct ? logical_input_bytes : input_bytes, local_direct ? logical_output_bytes : output_bytes}); }
            catch (pool::TransientTransport const&)
            { /* A broker outage does not invalidate accepted data. */ }
            completed = true;
            if (local_direct) impl.local_tasks.fetch_add(1);
            impl.task_elapsed_ns.fetch_add(uint64_t(std::max<int64_t>(0, elapsed)));
            auto const value = observation(); impl.observed(value, lease.endpoint, planning_selection);
            if (options.observe) try { options.observe(value); } catch (...) {}
        }
    };
    std::unique_ptr<Connection> connect(pool::Kind kind, uint8_t k, uint8_t strength, uint64_t work_units,
                                      uint64_t host_bytes, Operation operation, WorkerTaskIdentity identity,
                                      WorkerCancellation cancel, WorkerTaskOptions const& options = {},
                                      uint32_t chunk_count = 0, uint64_t max_output_bytes = 0)
    {
        auto const begin = Clock::now();
        auto control = broker.with_deadline(cancel.deadline);
        pool::Task task; task.task_id = identity.task_id.empty() ? pool::random_token() : identity.task_id;
        require(task.task_id.size() <= 256 && identity.attempt, "invalid compute task identity");
        task.kind = kind; task.k = k; task.strength = strength; task.work_units = work_units;
        task.chunk_count = chunk_count; task.max_output_bytes = max_output_bytes;
        task.compatibility = config.compatibility; task.resources.cpu_threads = kind == pool::Kind::Pack ? 1 : config.generation_threads;
        task.resources.host_bytes = host_bytes; task.resources.slots = 1;
        std::optional<pool::Lease> lease;
        pool::OwnerSession session;
        std::shared_ptr<worker_detail::TaskState> state;
        bool submitted = false;
        auto submit = [&](bool refresh)
        {
            auto lock = sequence_lock(cancel);
            if (owner.token.empty() || (refresh && session.token == owner.token))
            { owner = control.open_owner(config.owner_id, config.ephemeral_owner); task_states.reset(); }
            session = owner;
            state = task_states.acquire(identity, session.token);
            task.sequence = state->sequence; task.attempt = std::max(identity.attempt, state->next_attempt);
            require(task.attempt < UINT64_MAX, "compute task attempt exhausted"); state->next_attempt = task.attempt + 1;
            submitted = true; lease = control.acquire(session, task);
        };
        try
        {
            try { submit(false); } catch (pool::StaleOwnerSession const&) { submit(true); }
            while (!lease)
            {
                cancel.check(); std::this_thread::sleep_for(std::chrono::milliseconds(50));
                try { lease = control.acquire(session, task); }
                catch (pool::StaleOwnerSession const&) { submit(true); }
            }
            cancel.check();
        }
        catch (...)
        {
            failures.fetch_add(1);
            if (submitted) try { cleanup_control().cancel(session, task.task_id, task.attempt); } catch (...) {}
            throw;
        }
        auto connection = std::unique_ptr<Connection>(new Connection{*this, std::move(control), *lease, session, state, {}, begin, operation});
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
            if (options.selected && !options.selected(connection->observation())) throw WorkerTaskDeclined();
        }
        catch (...) { connection->declined = true; throw; }
        if (connection->local_direct) return connection;
        auto tls = config.tls; tls.deadline = std::min(tls.deadline, cancel.deadline);
        connection->channel = pool::Channel::connect(lease->endpoint, tls, lease->worker_identity);
        return connection;
    }
    void check_frame(pool::Frame const& frame)
    {
        if (frame.type == pool::Message::Error)
            throw_worker_error(frame.payload);
        require(frame.type == pool::Message::Data || frame.type == pool::Message::Reply ||
            frame.type == pool::Message::Result || frame.type == pool::Message::End, "unexpected compute result frame");
    }
    void generate_once(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
                  WorkerPackedSink const& packed_sink, WorkerContributionSink const& contribution_sink,
                  WorkerCancellation cancel, WorkerTaskIdentity identity)
    {
        cancel = cancellation(cancel); auto const count_bound = fragment_count_bound(k, strength);
        require(!ids.empty() && ids.size() <= pool::max_generation_batch_members &&
            (!packed_sink || ids.size() == 1), "invalid compute generation batch");
        uint64_t const host = worker_cpu_task_bytes(k, strength, ids.size(), config.max_task_bytes);
        auto connection = connect(config.generation_kind, k, strength, ids.size(), host,
            packed_sink ? Operation::Packed : Operation::Contributions, identity, cancel);
        if (connection->local_direct)
        {
            connection->logical_input_bytes = multiply(ids.size(), sizeof(ids.front()));
            auto first_output = [&]
            { if (connection->first_output == Clock::time_point{}) connection->first_output = Clock::now(); };
            auto packed = [&](PackedFragmentTile const& tile, uint64_t count)
            { first_output(); connection->logical_output_bytes = add(connection->logical_output_bytes,
                add(tile.boundaries.size_bytes(), tile.words.size_bytes())); packed_sink(tile, count); };
            auto contribution = [&](uint64_t ordinal, std::span<uint8_t const> bytes, uint16_t members)
            { first_output(); connection->logical_output_bytes = add(connection->logical_output_bytes, bytes.size());
              contribution_sink(ordinal, bytes, members); };
            try
            {
                connection->local_service->generate_assigned(connection->lease, ids,
                    packed_sink ? WorkerPackedSink(packed) : WorkerPackedSink{},
                    contribution_sink ? WorkerContributionSink(contribution) : WorkerContributionSink{}, cancel);
            }
            catch (pool::UnsupportedResources const&)
            { require(!connection->logical_output_bytes, "local worker rejected resources after partial output");
              connection->resource_rejected = true; throw; }
            cancel.check(); connection->delivered(); return;
        }
        auto compute = cpu_lease(cancel);
        pool::Writer request(config.tls.max_frame_bytes); pool::write_lease(request, connection->lease);
        request.u32(remaining_timeout(cancel));
        request.u8(uint8_t(packed_sink ? Operation::Packed : Operation::Contributions)); request.u8(k); request.u8(strength); request.u16(uint16_t(ids.size()));
        for (auto const& id : ids) request.bytes(id); connection->input_bytes = request.data().size();
        compute.reset(); connection->channel.send(pool::Message::Work, request.data());
        uint64_t total = 0, expected_boundaries = 0, expected_words = 0, received_words = 0, next_chunk = 0;
        PackedMemberValidation validation; std::vector<uint32_t> boundaries; std::vector<uint64_t> words;
        bool prelude = false, delivered_prelude = false;
        unsigned __int128 packed_bits = 0;
        unsigned available_bits = 0;
        uint64_t decoded = 0, previous = 0;
        size_t decoded_chunk = 0;
        for (;;)
        {
            cancel.check(); auto frame = connection->channel.receive();
            try { check_frame(frame); }
            catch (pool::UnsupportedResources const&)
            { require(!connection->output_bytes, "compute worker rejected resources after partial output"); connection->resource_rejected = true; throw; }
            if (frame.type == pool::Message::Reply) { require(frame.payload.empty(), "invalid compute keepalive"); continue; }
            if (connection->first_output == Clock::time_point{}) connection->first_output = Clock::now();
            compute = cpu_lease(cancel);
            connection->output_bytes = add(connection->output_bytes, frame.payload.size());
            if (frame.type == pool::Message::End)
            {
                require(frame.payload.empty(), "compute result end carries trailing bytes");
                if (packed_sink) require(delivered_prelude && received_words == expected_words && decoded == total && packed_bits == 0,
                    "compute result member is truncated or has nonzero padding");
                else require(next_chunk == PlotGroupFile::getChunkCountForK(k), "compute contributions are truncated");
                compute.reset(); connection->delivered(); return;
            }
            pool::Reader input(frame.payload); auto const type = DataKind(input.u8());
            if (type == DataKind::Prelude)
            {
                require(bool(packed_sink) && !prelude, "unexpected compute member prelude");
                total = input.u64(); validation.checked_count = input.u64(); validation.flags = input.u32(); expected_boundaries = input.u64(); input.finish();
                require(total <= count_bound && total <= UINT32_MAX && expected_boundaries == PlotGroupFile::getChunkCountForK(k) + 1,
                    "compute member exceeds public parameter bound");
                require(validation.checked_count == total && !(validation.flags & ~PackedMemberValidation::known_flags) &&
                    !(validation.flags & (PackedMemberValidation::out_of_range | PackedMemberValidation::unsorted)), "invalid compute validation summary");
                expected_words = (multiply(total, k + PlotGroupFile::PROOFS_PER_CHUNK_BITS) + 63) / 64;
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
                words.clear(); words.reserve(count); for (uint32_t i = 0; i < count; ++i) words.push_back(input.u64()); input.finish();
                unsigned const width = k + PlotGroupFile::PROOFS_PER_CHUNK_BITS;
                uint64_t const range = uint64_t(1) << width;
                uint64_t const maximum_delta = uint64_t(72 - k) * (178ull << (k - 8));
                for (auto word : words)
                {
                    packed_bits |= static_cast<unsigned __int128>(word) << available_bits;
                    available_bits += 64;
                    while (available_bits >= width && decoded < total)
                    {
                        while (decoded_chunk + 1 < boundaries.size() && decoded >= boundaries[decoded_chunk + 1])
                        { ++decoded_chunk; previous = uint64_t(decoded_chunk) * range; }
                        require(decoded_chunk + 1 < boundaries.size(), "compute member payload exceeds boundaries");
                        uint64_t const value = uint64_t(decoded_chunk) * range + (uint64_t(packed_bits) & (range - 1));
                        require(value >= previous && ((validation.flags & PackedMemberValidation::wide_delta) || value - previous < maximum_delta),
                            "compute member payload is unsorted or has an invalid delta summary");
                        previous = value; ++decoded; packed_bits >>= width; available_bits -= width;
                    }
                }
                compute.reset(); packed_sink({{}, words, {}}, total); received_words += count;
            }
            else if (type == DataKind::Contributions)
            {
                require(bool(contribution_sink), "unexpected compute contribution stream");
                while (input.remaining())
                {
                    auto const ordinal = input.u64(); auto const members = input.u16(); auto const bytes = input.bytes(config.tls.max_frame_bytes);
                    require(ordinal == next_chunk++ && ordinal < PlotGroupFile::getChunkCountForK(k) && members == ids.size() && !bytes.empty(), "invalid compute contribution order or membership");
                    compute.reset(); contribution_sink(ordinal, bytes, members); compute = cpu_lease(cancel);
                }
            }
            else throw std::runtime_error("unexpected compute generation data");
            compute.reset();
        }
    }
    void generate(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
                  WorkerPackedSink const& packed_sink, WorkerContributionSink const& contribution_sink,
                  WorkerCancellation cancel, WorkerTaskIdentity identity)
    {
        cancel = cancellation(cancel);
        require(!ids.empty() && ids.size() <= pool::max_generation_batch_members,
            "invalid compute generation batch");
        GenerationLease admission(*this, ids.size(), cancel);
        if (identity.task_id.empty()) identity.task_id = pool::random_token();
        for (unsigned failed = 0;; ++failed)
        {
            try { generate_once(k, strength, ids, packed_sink, contribution_sink, cancel, identity); return; }
            catch (pool::UnsupportedResources const&)
            {
                // A refused batch may need a different shape. Its owner must
                // split it under new identities rather than silently changing
                // the body of this logical task or waiting on excluded offers.
                if ((contribution_sink && ids.size() > 1) || failed >= pool::max_retry_failures) throw;
                cancel.check();
            }
        }
    }

};

WorkerClient::WorkerClient(WorkerClientConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
WorkerClient::~WorkerClient() = default;
bool WorkerClient::local_available() const { return bool(impl_->local_service()); }
void WorkerClient::generate(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& id,
    WorkerPackedSink const& sink, WorkerCancellation cancel, WorkerTaskIdentity identity)
{ impl_->generate(k, strength, {&id, 1}, sink, {}, cancel, std::move(identity)); }
void WorkerClient::contributions(uint8_t k, uint8_t strength, std::span<std::array<uint8_t, 32> const> ids,
    WorkerContributionSink const& sink, WorkerCancellation cancel, WorkerTaskIdentity identity)
{ impl_->generate(k, strength, ids, {}, sink, cancel, std::move(identity)); }
void WorkerClient::generate_local(uint8_t k, uint8_t strength, std::array<uint8_t, 32> const& id,
    WorkerPackedSink const& sink, WorkerCancellation cancel)
{
    auto local = impl_->local_service(); require(bool(local), "local compute fallback was not configured");
    cancel = impl_->cancellation(cancel); Impl::GenerationLease admission(*impl_, 1, cancel);
    local->generate_local(k, strength, id, sink, cancel, impl_->config.generation_kind, impl_->local_admission());
}
void WorkerClient::contributions_local(uint8_t k, uint8_t strength,
    std::span<std::array<uint8_t, 32> const> ids, WorkerContributionSink const& sink, WorkerCancellation cancel)
{
    auto local = impl_->local_service();
    require(local && !ids.empty() && ids.size() <= pool::max_generation_batch_members,
        "local compute fallback was not configured or exceeds its batch limit");
    cancel = impl_->cancellation(cancel); Impl::GenerationLease admission(*impl_, ids.size(), cancel);
    local->contributions_local(k, strength, ids, sink, cancel, impl_->config.generation_kind, impl_->local_admission());
}
std::vector<std::vector<uint8_t>> WorkerClient::encode(uint8_t k, std::span<WorkerChunkInput const> chunks,
    uint64_t maximum, WorkerCancellation cancel, WorkerTaskIdentity identity, WorkerEncodedValidator const& validator,
    WorkerTaskOptions const& options)
{
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
        connection->local_service->encode_assigned(connection->lease, payload.data(), maximum,
            [&](size_t index, std::span<uint8_t const> bytes)
            { require(index < output.size(), "local worker returned an invalid encoded chunk");
              connection->logical_output_bytes = add(connection->logical_output_bytes, bytes.size());
              output[index].assign(bytes.begin(), bytes.end()); }, cancel);
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
        cancel.check(); auto frame = connection->channel.receive(); impl_->check_frame(frame);
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
        impl_->peak_generation_members.load(), impl_->peak_active_generation.load()};
}
unsigned WorkerClient::max_window_chunks() const { return impl_->config.max_window_chunks; }
uint64_t WorkerClient::max_task_bytes() const { return impl_->config.max_task_bytes; }
unsigned WorkerClient::suggested_batch_members(uint8_t k, uint8_t strength, unsigned maximum) const
{
    (void)parameters(k, strength); if (!maximum) return 0;
    unsigned value;
    { std::lock_guard lock(impl_->planning_mutex); value = impl_->planning.suggested(k, strength, impl_->config.generation_kind).batch; }
    // Hints share the control pool and use a short optional deadline. Failure
    // leaves the learned bounded policy usable during a broker outage.
    try
    {
        WorkerCancellation cancel; cancel.deadline = Clock::now() + std::chrono::milliseconds(100);
        pool::OwnerSession session;
        { auto lock = impl_->sequence_lock(cancel); session = impl_->owner; }
        if (!session.token.empty())
        {
            pool::Task prototype; prototype.compatibility = impl_->config.compatibility;
            prototype.kind = impl_->config.generation_kind; prototype.k = k; prototype.strength = strength;
            prototype.resources.cpu_threads = impl_->config.generation_threads;
            auto const hint = impl_->broker.with_deadline(cancel.deadline).generation_hint(session, prototype);
            value = hint ? std::min(value, hint) : 1;
        }
    }
    catch (...) { value = 1; }
    return std::max(1u, std::min({maximum, value, unsigned(pool::max_generation_batch_members)}));
}
unsigned WorkerClient::suggested_generation_concurrency(uint8_t k, uint8_t strength, unsigned maximum) const
{
    (void)parameters(k, strength); if (!maximum) return 0;
    std::lock_guard lock(impl_->planning_mutex);
    return std::max(1u, std::min(maximum, impl_->planning.suggested(k, strength, impl_->config.generation_kind).concurrency));
}
uint64_t WorkerClient::memory_bytes()
{
    // sizeof(Impl) includes fixed worker/route planning histories and keys.
    // Also reserve hash nodes/buckets (including replacement growth), shared
    // retry records and allocator overhead. Session TLS is per-call.
    return sizeof(Impl) + pool::BrokerClient::memory_bytes() + uint64_t(worker_detail::TaskRegistry::maximum) *
        (sizeof(worker_detail::TaskState) + sizeof(worker_detail::RetryHandle) + sizeof(std::shared_ptr<worker_detail::TaskState>) + 1024);
}
} // namespace group_grow
