#include "host/ComputePool.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <utility>

namespace group_grow::pool
{
namespace
{
using Clock = std::chrono::steady_clock;
void require(bool value, char const* message) { if (!value) throw std::runtime_error(message); }
enum class ErrorCode : uint8_t { Invalid = 1, StaleOwner = 2, StaleDelivery = 3, StaleLease = 4 };
enum class Operation : uint8_t { OpenOwner = 1, Acquire, Cancel, Renew, Register, Begin, Finish, Delivered, Reject, Abandon, GenerationHint, Heartbeat, Unregister, CloseOwner, Status, WorkerAdmission, RetireWorker };
void write_endpoint(Writer& writer, Endpoint const& endpoint)
{ writer.string(endpoint.host); writer.u16(endpoint.port); writer.string(endpoint.server_name); }
Endpoint read_endpoint(Reader& reader)
{ Endpoint result; result.host = reader.string(253); result.port = reader.u16(); result.server_name = reader.string(253); return result; }
void write_resources(Writer& writer, Resources const& value)
{ writer.u32(value.cpu_threads); writer.u32(value.slots); writer.u64(value.host_bytes); writer.u64(value.gpu_bytes); }
Resources read_resources(Reader& reader)
{
    Resources result; result.cpu_threads = reader.u32(); result.slots = reader.u32();
    result.host_bytes = reader.u64(); result.gpu_bytes = reader.u64(); return result;
}
void write_task(Writer& writer, Task const& task)
{
    writer.string(task.task_id); writer.string(task.compatibility); writer.u64(task.attempt); writer.u64(task.sequence); writer.u64(task.work_units);
    writer.u8(uint8_t(task.kind)); writer.u8(task.k); writer.u8(task.strength); write_resources(writer, task.resources);
    writer.u32(task.chunk_count); writer.u64(task.max_output_bytes);
}
Task read_task(Reader& reader)
{
    Task result; result.task_id = reader.string(256); result.compatibility = reader.string(256);
    result.attempt = reader.u64(); result.sequence = reader.u64(); result.work_units = reader.u64(); result.kind = Kind(reader.u8());
    result.k = reader.u8(); result.strength = reader.u8(); result.resources = read_resources(reader);
    result.chunk_count = reader.u32(); result.max_output_bytes = reader.u64(); return result;
}
void write_session(Writer& writer, OwnerSession const& session)
{
    writer.string(session.owner_id); writer.string(session.token); writer.string(session.epoch); writer.u64(session.generation);
}
OwnerSession read_session(Reader& reader)
{
    OwnerSession result; result.owner_id = reader.string(256); result.token = reader.string(64);
    result.epoch = reader.string(64); result.generation = reader.u64(); return result;
}
void write_profiles(Writer& writer, std::vector<GenerationProfile> const& profiles)
{
    require(profiles.size() <= max_generation_profiles, "too many generation profiles");
    writer.u32(uint32_t(profiles.size()));
    for (auto const& profile : profiles)
    {
        writer.u8(uint8_t(profile.kind)); writer.u8(profile.k); writer.u8(profile.strength); writer.u32(profile.cpu_threads);
        write_resources(writer, profile.resources); writer.u64(profile.member_host_bytes); writer.u32(profile.max_batch_members);
    }
}
std::vector<GenerationProfile> read_profiles(Reader& reader)
{
    auto const count = reader.u32(); require(count <= max_generation_profiles, "too many generation profiles");
    std::vector<GenerationProfile> profiles; profiles.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        GenerationProfile profile; profile.kind = Kind(reader.u8()); profile.k = reader.u8(); profile.strength = reader.u8();
        profile.cpu_threads = reader.u32(); profile.resources = read_resources(reader);
        profile.member_host_bytes = reader.u64(); profile.max_batch_members = reader.u32(); profiles.push_back(profile);
    }
    return profiles;
}
void write_offer(Writer& writer, WorkerOffer const& offer)
{
    writer.string(offer.instance); writer.string(offer.compatibility); write_endpoint(writer, offer.endpoint);
    write_resources(writer, offer.capacity); writer.u64(offer.capacity_revision); writer.u32(offer.generation_threads); writer.u32(offer.capabilities); writer.u8(offer.min_k); writer.u8(offer.max_k);
    write_profiles(writer, offer.generation_profiles);
    writer.u32(offer.codec_limits.max_chunks); writer.u64(offer.codec_limits.max_task_bytes); writer.u8(uint8_t(offer.admission));
}
WorkerOffer read_offer(Reader& reader)
{
    WorkerOffer result; result.instance = reader.string(64); result.compatibility = reader.string(256);
    result.endpoint = read_endpoint(reader); result.capacity = read_resources(reader); result.capacity_revision = reader.u64(); result.generation_threads = reader.u32(); result.capabilities = reader.u32();
    result.min_k = reader.u8(); result.max_k = reader.u8(); result.generation_profiles = read_profiles(reader);
    result.codec_limits.max_chunks = reader.u32(); result.codec_limits.max_task_bytes = reader.u64();
    result.admission = WorkerAdmission(reader.u8()); return result;
}
void write_update(Writer& writer, std::optional<CapacityUpdate> const& update)
{
    writer.u8(bool(update)); if (update)
    {
        writer.u64(update->revision); write_resources(writer, update->capacity);
        writer.u8(bool(update->generation_profiles));
        if (update->generation_profiles) write_profiles(writer, *update->generation_profiles);
    }
}
std::optional<CapacityUpdate> read_update(Reader& reader)
{
    auto const present = reader.u8(); require(present <= 1, "invalid capacity update flag");
    if (!present) return {};
    CapacityUpdate result; result.revision = reader.u64(); result.capacity = read_resources(reader);
    auto const profiles = reader.u8(); require(profiles <= 1, "invalid profile update flag");
    if (profiles) result.generation_profiles = read_profiles(reader);
    return result;
}
bool identifier(std::string const& value, size_t limit)
{
    if (value.empty() || value.size() > limit) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char c)
    { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
        c == '.' || c == '_' || c == '-' || c == ':'; });
}
bool fits(Resources const& value, Resources const& capacity)
{
    return value.cpu_threads <= capacity.cpu_threads && value.slots <= capacity.slots &&
        value.host_bytes <= capacity.host_bytes && value.gpu_bytes <= capacity.gpu_bytes;
}
void subtract(Resources& value, Resources const& other)
{
    require(fits(other, value), "broker resource accounting underflow");
    value.cpu_threads -= other.cpu_threads; value.slots -= other.slots;
    value.host_bytes -= other.host_bytes; value.gpu_bytes -= other.gpu_bytes;
}
std::string owner_key(std::string const& identity, std::string const& owner_id)
{ return identity + ':' + owner_id; }
std::string task_key(std::string const& owner, std::string const& session, std::string const& task)
{ return owner + '/' + session + '/' + task; }
Writer request(Operation operation)
{ Writer writer(65536); writer.u32(protocol_version); writer.u8(uint8_t(operation)); return writer; }
} // namespace
void write_lease(Writer& writer, Lease const& lease)
{
    writer.string(lease.token); writer.string(lease.epoch); writer.string(lease.owner_identity);
    writer.string(lease.broker_identity); writer.string(lease.broker_node_id);
    writer.string(lease.owner_id); writer.string(lease.owner_session); writer.u64(lease.owner_generation);
    writer.string(lease.worker_identity); writer.string(lease.worker_instance); write_endpoint(writer, lease.endpoint);
    write_task(writer, lease.task); writer.u32(lease.ttl_ms);
}
Lease read_lease(Reader& reader)
{
    Lease result; result.token = reader.string(64); result.epoch = reader.string(64); result.owner_identity = reader.string(64);
    result.broker_identity = reader.string(64); result.broker_node_id = reader.string(64);
    result.owner_id = reader.string(256); result.owner_session = reader.string(64); result.owner_generation = reader.u64();
    result.worker_identity = reader.string(64); result.worker_instance = reader.string(64); result.endpoint = read_endpoint(reader);
    result.task = read_task(reader); result.ttl_ms = reader.u32(); return result;
}
struct BrokerClient::Connections
{
    struct Slot { Channel channel; bool busy = false; uint64_t route_revision = 0; };
    std::array<Slot, control_channel_limit> slots;
    unsigned limit;
    TlsOptions creator_tls;
    std::mutex mutex;
    std::condition_variable changed;
    ControlMetrics metrics;
    Endpoint endpoint;
    uint64_t route_revision = 1;
    explicit Connections(Endpoint route, TlsOptions value, unsigned channels) : limit(channels), creator_tls(std::move(value)), endpoint(std::move(route))
    {
        require(limit && limit <= control_channel_limit, "invalid broker control channel limit");
        // Control messages have their own fixed bound. A worker's smaller
        // bulk-transfer window must still carry its bounded resource profiles.
        creator_tls.max_frame_bytes = 65536;
    }
};
BrokerClient::BrokerClient(Endpoint endpoint, TlsOptions tls, unsigned channels) : endpoint_(std::move(endpoint)), tls_(std::move(tls)),
    connections_(std::make_shared<Connections>(endpoint_, tls_, channels)) {}
BrokerClient BrokerClient::with_deadline(std::chrono::steady_clock::time_point until) const
{
    auto result = *this; result.tls_.deadline = std::min(tls_.deadline, until); return result;
}
void BrokerClient::reroute(Endpoint const& endpoint)
{
    std::lock_guard lock(connections_->mutex);
    if (connections_->endpoint == endpoint) return;
    require(connections_->route_revision != UINT64_MAX, "broker route revision exhausted");
    connections_->endpoint = endpoint; ++connections_->route_revision;
}
bool BrokerClient::shared() const noexcept { return connections_.use_count() != 1; }
void BrokerClient::disconnect_idle()
{
    std::lock_guard lock(connections_->mutex);
    for (auto const& slot : connections_->slots) require(!slot.busy, "broker control request is still active");
    for (auto& slot : connections_->slots) slot.channel.close();
}
std::vector<uint8_t> BrokerClient::call(Writer const& writer, std::string* peer_identity, std::string* peer_node_id)
{
    require(writer.data().size() <= 65536, "broker request is too large");
    auto const started = Clock::now();
    auto const until = std::min(tls_.deadline, started + std::chrono::milliseconds(tls_.timeout_ms));
    auto& shared = *connections_;
    std::unique_lock lock(shared.mutex);
    auto idle = [&]() -> Connections::Slot*
    {
        // Reuse the existing authenticated channel before opening another.
        for (auto& slot : std::span(shared.slots).first(shared.limit)) if (!slot.busy && slot.channel) return &slot;
        for (auto& slot : std::span(shared.slots).first(shared.limit)) if (!slot.busy) return &slot;
        return nullptr;
    };
    Connections::Slot* slot = nullptr;
    while (!(slot = idle()))
        if (shared.changed.wait_until(lock, until) == std::cv_status::timeout)
            throw TransientTransport("broker control admission timed out");
    if (Clock::now() >= until) throw TransientTransport("broker control request timed out");
    slot->busy = true;
    auto const endpoint = shared.endpoint; auto const route_revision = shared.route_revision;
    shared.metrics.wait_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count());
    ++shared.metrics.requests; shared.metrics.request_bytes += writer.data().size();
    lock.unlock();
    Frame frame;
    try
    {
        if (slot->route_revision != route_revision) { slot->channel.close(); slot->route_revision = route_revision; }
        if (slot->channel && slot->channel.peer_closed()) slot->channel.close();
        if (!slot->channel)
        {
            slot->channel = Channel::connect(endpoint, shared.creator_tls, {}, until);
            { std::lock_guard guard(shared.mutex); ++shared.metrics.connections; }
        }
        else { std::lock_guard guard(shared.mutex); ++shared.metrics.reused; }
        slot->channel.reset_deadline(until);
        slot->channel.send(Message::Control, writer.data()); frame = slot->channel.receive();
        if (peer_identity) *peer_identity = slot->channel.peer_identity();
        if (peer_node_id) *peer_node_id = slot->channel.peer_node_id();
    }
    catch (...)
    {
        slot->channel.close();
        { std::lock_guard guard(shared.mutex); slot->busy = false; }
        shared.changed.notify_one(); throw;
    }
    {
        std::lock_guard guard(shared.mutex); shared.metrics.response_bytes += frame.payload.size(); slot->busy = false;
    }
    shared.changed.notify_one();
    if (frame.type == Message::Error)
    {
        Reader reader(frame.payload); auto const code = ErrorCode(reader.u8()); auto message = reader.string(512); reader.finish();
        require(code == ErrorCode::Invalid || code == ErrorCode::StaleOwner || code == ErrorCode::StaleDelivery ||
            code == ErrorCode::StaleLease, "invalid broker error code");
        if (code == ErrorCode::StaleOwner) throw StaleOwnerSession("compute broker: " + message);
        if (code == ErrorCode::StaleDelivery) throw StaleDelivery("compute broker: " + message);
        if (code == ErrorCode::StaleLease) throw StaleLease("compute broker: " + message);
        throw std::runtime_error("compute broker: " + message);
    }
    require(frame.type == Message::Reply && frame.payload.size() <= 65536, "invalid broker response");
    return std::move(frame.payload);
}
ControlMetrics BrokerClient::metrics() const
{ std::lock_guard lock(connections_->mutex); return connections_->metrics; }
uint64_t BrokerClient::memory_bytes(unsigned channels)
{
    require(channels && channels <= control_channel_limit, "invalid broker control channel limit");
    return channels * (Channel::memory_bytes(65536) + 131072);
}
OwnerSession BrokerClient::open_owner(std::string const& owner_id, bool ephemeral)
{
    auto writer = request(Operation::OpenOwner); writer.string(owner_id); writer.u8(ephemeral); auto bytes = call(writer);
    Reader reader(bytes); auto result = read_session(reader); reader.finish(); return result;
}
void BrokerClient::close_owner(OwnerSession const& session)
{ auto writer = request(Operation::CloseOwner); write_session(writer, session); auto bytes = call(writer); Reader(bytes).finish(); }
std::optional<Lease> BrokerClient::acquire(OwnerSession const& owner, Task const& task)
{
    auto writer = request(Operation::Acquire); write_session(writer, owner); write_task(writer, task);
    std::string peer, node; auto bytes = call(writer, &peer, &node);
    Reader reader(bytes); auto const present = reader.u8(); require(present <= 1, "invalid broker grant flag");
    std::optional<Lease> result; if (present)
    {
        result = read_lease(reader);
        require(result->broker_identity == peer, "lease authority differs from authenticated broker");
        require(node.empty() || result->broker_node_id == node, "lease node differs from authenticated broker");
    }
    reader.finish(); return result;
}
uint32_t BrokerClient::generation_hint(OwnerSession const& owner, Task const& task)
{
    auto writer = request(Operation::GenerationHint); write_session(writer, owner); write_task(writer, task); auto bytes = call(writer);
    Reader reader(bytes); auto const result = reader.u32(); reader.finish(); return result;
}
void BrokerClient::cancel(OwnerSession const& owner, std::string const& task, uint64_t attempt)
{
    auto writer = request(Operation::Cancel); write_session(writer, owner); writer.string(task); writer.u64(attempt);
    auto bytes = call(writer); Reader(bytes).finish();
}
void BrokerClient::renew(Lease const& lease)
{ auto writer = request(Operation::Renew); write_lease(writer, lease); auto bytes = call(writer); Reader(bytes).finish(); }
std::string BrokerClient::register_worker(WorkerOffer const& offer, std::string* peer_identity, std::string* peer_node_id)
{
    auto writer = request(Operation::Register); write_offer(writer, offer); auto bytes = call(writer, peer_identity, peer_node_id);
    Reader reader(bytes); auto epoch = reader.string(64); reader.finish();
    require(!epoch.empty(), "broker registration omitted its epoch"); return epoch;
}
void BrokerClient::unregister_worker(std::string const& instance)
{ auto writer = request(Operation::Unregister); writer.string(instance); auto bytes = call(writer); Reader(bytes).finish(); }
bool BrokerClient::retire_worker(std::string const& instance)
{
    auto writer = request(Operation::RetireWorker); writer.string(instance); auto bytes = call(writer); Reader reader(bytes);
    auto const idle = reader.u8(); require(idle <= 1, "invalid worker retirement flag"); reader.finish(); return idle;
}
bool BrokerClient::heartbeat(std::string const& instance, uint64_t revision)
{
    auto writer = request(Operation::Heartbeat); writer.string(instance); writer.u64(revision); auto bytes = call(writer);
    Reader reader(bytes); auto const current = reader.u8(); require(current <= 1, "invalid broker heartbeat flag");
    reader.finish(); return current;
}
bool BrokerClient::begin(Lease const& lease, std::string const& owner)
{
    auto writer = request(Operation::Begin); write_lease(writer, lease); writer.string(owner); auto bytes = call(writer);
    Reader reader(bytes); auto const result = reader.u8(); require(result <= 1, "invalid broker begin flag"); reader.finish(); return result;
}
void BrokerClient::finish(Lease const& lease, std::optional<CapacityUpdate> const& update)
{
    auto writer = request(Operation::Finish); write_lease(writer, lease); write_update(writer, update);
    auto bytes = call(writer); Reader(bytes).finish();
}
void BrokerClient::delivered(Lease const& lease, DeliveryObservation const& observation)
{
    auto writer = request(Operation::Delivered); write_lease(writer, lease); writer.u64(observation.elapsed_ns);
    writer.u64(observation.input_bytes); writer.u64(observation.output_bytes); auto bytes = call(writer); Reader(bytes).finish();
}

void BrokerClient::reject(Lease const& lease, FailureCode reason, std::optional<CapacityUpdate> const& update)
{
    auto writer = request(Operation::Reject); write_lease(writer, lease); writer.u8(uint8_t(reason)); write_update(writer, update);
    auto bytes = call(writer); Reader(bytes).finish();
}
void BrokerClient::abandon(Lease const& lease)
{ auto writer = request(Operation::Abandon); write_lease(writer, lease); auto bytes = call(writer); Reader(bytes).finish(); }
void BrokerClient::worker_admission(std::string const& instance, WorkerAdmission admission)
{
    auto writer = request(Operation::WorkerAdmission); writer.string(instance); writer.u8(uint8_t(admission));
    auto bytes = call(writer); Reader(bytes).finish();
}
BrokerStatusPage BrokerClient::status(StatusRequest const& query)
{
    auto writer = request(Operation::Status); writer.u8(uint8_t(query.section)); writer.u64(query.cursor); writer.u32(query.limit);
    auto bytes = call(writer); Reader in(bytes); BrokerStatusPage result; result.section = StatusSection(in.u8());
    require(result.section == query.section, "broker status section mismatch");
    result.next_cursor = in.u64(); result.workers = in.u64(); result.owners = in.u64(); result.jobs = in.u64();
    result.queued = in.u64(); result.granted = in.u64(); result.running = in.u64(); result.cancelled = in.u64();
    auto const count = in.u32(); require(count <= query.limit && count <= max_status_page_size, "broker status page exceeds limit");
    for (uint32_t index = 0; index < count; ++index)
    {
        if (result.section == StatusSection::Workers)
        {
            BrokerWorkerStatus value; value.identity = in.string(64); value.instance = in.string(64); value.compatibility = in.string(256);
            value.endpoint = read_endpoint(in);
            value.capacity = read_resources(in); value.reserved = read_resources(in); value.admission = WorkerAdmission(in.u8()); value.seen_ms = in.u64();
            result.worker_records.push_back(std::move(value));
        }
        else if (result.section == StatusSection::Owners)
        {
            BrokerOwnerStatus value; value.identity = in.string(64); value.owner_id = in.string(256);
            value.queued = in.u64(); value.granted = in.u64(); value.running = in.u64(); value.cancelled = in.u64();
            result.owner_records.push_back(std::move(value));
        }
        else
        {
            BrokerJobStatus value; value.id = in.u64(); value.owner_identity = in.string(64); value.owner_id = in.string(256);
            value.worker_identity = in.string(64); value.worker_instance = in.string(64);
            value.kind = Kind(in.u8()); value.state = JobState(in.u8()); value.waiting = WaitReason(in.u8());
            value.resources = read_resources(in); value.attempt = in.u64(); value.work_units = in.u64(); value.age_ms = in.u64();
            value.k = in.u8(); value.strength = in.u8(); result.job_records.push_back(std::move(value));
        }
    }
    in.finish(); return result;
}
char const* admission_name(WorkerAdmission value)
{
    switch (value) { case WorkerAdmission::Accepting: return "accepting"; case WorkerAdmission::Paused: return "paused";
        case WorkerAdmission::Draining: return "draining"; case WorkerAdmission::Drained: return "drained"; }
    return "unknown";
}
char const* job_state_name(JobState value)
{
    switch (value) { case JobState::Queued: return "queued"; case JobState::Granted: return "granted";
        case JobState::Running: return "running"; case JobState::Cancelled: return "cancelled";
        case JobState::Finished: return "finished"; case JobState::Expired: return "expired"; }
    return "unknown";
}
char const* wait_reason_name(WaitReason value)
{
    switch (value)
    {
    case WaitReason::None: return "none"; case WaitReason::NoWorkers: return "no_workers";
    case WaitReason::Incompatible: return "incompatible"; case WaitReason::UnsupportedKind: return "unsupported_kind";
    case WaitReason::UnsupportedParameters: return "unsupported_parameters"; case WaitReason::RetryExcluded: return "retry_excluded";
    case WaitReason::CpuCapacity: return "cpu_capacity"; case WaitReason::HostCapacity: return "host_capacity";
    case WaitReason::GpuCapacity: return "gpu_capacity"; case WaitReason::SlotCapacity: return "slot_capacity";
    case WaitReason::WorkerPaused: return "worker_paused"; case WaitReason::WorkerDraining: return "worker_draining";
    case WaitReason::ResourcesReserved: return "resources_reserved"; case WaitReason::FairQueue: return "fair_queue";
    case WaitReason::OwnerCancelled: return "owner_cancelled";
    case WaitReason::CpuReserved: return "cpu_reserved"; case WaitReason::HostReserved: return "host_reserved";
    case WaitReason::GpuReserved: return "gpu_reserved"; case WaitReason::SlotReserved: return "slot_reserved";
    }
    return "unknown";
}

struct BrokerServer::Impl
{
    using State = JobState;
    struct History
    {
        Kind kind;
        uint8_t k, strength;
        uint32_t cpu_threads;
        std::string owner;
        struct Sample { DeliveryObservation observation; uint64_t units = 0; };
        std::array<Sample, 8> samples{};
        size_t count = 0, next = 0;
        uint64_t assigned = 0;
        bool matches(Task const& task, Kind selected, std::string const& source, uint32_t threads) const
        { return owner == source && kind == selected && k == task.k && strength == task.strength && cpu_threads == threads; }
        long double cost(uint64_t units) const
        {
            long double total = 0;
            for (size_t i = 0; i < count; ++i)
                total += static_cast<long double>(samples[i].observation.elapsed_ns) / samples[i].units;
            return total / count * units;
        }
        void add(DeliveryObservation observation, uint64_t units)
        { samples[next] = {observation, units}; next = (next + 1) % samples.size(); count = std::min(count + 1, samples.size()); }
    };
    struct Worker { WorkerOffer offer; Clock::time_point seen; std::vector<History> histories; };
    struct Owner
    {
        std::string identity;
        OwnerSession session;
        uint64_t highest_sequence = 0;
        long double service = 0;
        bool ephemeral = false;
        Clock::time_point seen = Clock::now();
    };
    struct Job
    {
        std::string owner;
        uint64_t status_id = 0;
        Task requested;
        std::optional<Lease> lease;
        State state = State::Queued;
        Clock::time_point expires, touched;
        Clock::time_point created = Clock::now();
        bool observed = false;
        struct Failure { std::string worker, instance; Kind kind; };
        std::vector<Failure> failures;
    };
    BrokerOptions options;
    Listener listener;
    std::string epoch = random_token(), last_owner, last_worker, broker_identity, broker_node_id;
    uint64_t generation = 0, assignments = 0, next_status_id = 0;
    long double service_floor = 0;
    std::mutex mutex;
    struct IdleChannel { Channel channel; Clock::time_point touched; };
    std::deque<IdleChannel> idle_channels;
    size_t connections = 0;
    Clock::time_point next_maintenance{};
    std::map<std::string, Worker> workers;
    std::map<std::string, Owner> owners;
    std::map<std::string, Job> jobs;
    explicit Impl(BrokerOptions value) : options(std::move(value)), listener(options.listen, options.tls)
    {
        broker_identity = certificate_identity(options.tls.cert_file);
        broker_node_id = certificate_node_id(options.tls.cert_file);
        require(options.lease_ms && options.lease_ms <= 3600000 && options.worker_ms && options.worker_ms <= 3600000 &&
            options.max_workers && options.max_workers <= 4096 && options.max_owners && options.max_owners <= 4096 &&
            options.max_tasks && options.max_tasks <= 1048576 && options.control_threads && options.control_threads <= 64 &&
            options.max_connections >= options.control_threads && options.max_connections <= 4096 &&
            options.idle_ms && options.idle_ms <= 3600000, "invalid broker limits");
        require(options.tls.max_frame_bytes >= 65536 && options.tls.max_frame_bytes <= 1048576,
                "broker frame limit must be between control bounds");
        for (auto const& [id, share] : options.owner_shares)
            require(identifier(id, 256) && share && share <= 1000000, "invalid owner share");
    }
    bool reserved(Job const& job) const
    { return job.lease && (job.state == State::Granted || job.state == State::Running || job.state == State::Cancelled); }
    void expire()
    {
        auto const now = Clock::now();
        for (auto& [key, job] : jobs)
        {
            // An owner's deadline can also prevent its final Cancel request.
            // Polls keep queued work alive; abandoned requests must become
            // evictable and stop consuming grants when workers return.
            if ((reserved(job) && now >= job.expires) ||
                (job.state == State::Queued && now - job.touched >= std::chrono::milliseconds(options.lease_ms)))
            { job.state = State::Expired; job.touched = now; }
        }
        for (auto it = workers.begin(); it != workers.end();)
        {
            if (now - it->second.seen < std::chrono::milliseconds(options.worker_ms)) { ++it; continue; }
            // A disappeared worker cannot be reassigned under its old instance.
            // Its already-running capacity remains fenced until lease expiry.
            it = workers.erase(it);
        }
        std::erase_if(owners, [&](auto const& item)
        {
            if (!item.second.ephemeral || now - item.second.seen < std::chrono::milliseconds(options.lease_ms)) return false;
            return std::none_of(jobs.begin(), jobs.end(), [&](auto const& job)
            { return job.second.owner == item.first && (reserved(job.second) || job.second.state == State::Queued); });
        });
    }
    Owner& owner(std::string const& identity, OwnerSession const& session)
    {
        if (session.epoch != epoch) throw StaleOwnerSession("broker restarted; open a new owner session");
        auto it = owners.find(owner_key(identity, session.owner_id));
        if (it == owners.end() || it->second.session.token != session.token ||
            it->second.session.generation != session.generation) throw StaleOwnerSession("stale owner session");
        it->second.seen = Clock::now(); return it->second;
    }
    Job& lease_job(Lease const& lease)
    {
        if (lease.epoch != epoch) throw StaleLease("stale broker epoch");
        auto it = jobs.find(task_key(owner_key(lease.owner_identity, lease.owner_id), lease.owner_session, lease.task.task_id));
        if (it == jobs.end() || !it->second.lease) throw StaleLease("compute lease is no longer registered");
        require(it->second.lease->token == lease.token &&
            it->second.lease->broker_identity == lease.broker_identity && it->second.lease->broker_node_id == lease.broker_node_id &&
            it->second.lease->owner_session == lease.owner_session &&
            it->second.lease->owner_generation == lease.owner_generation &&
            it->second.lease->worker_identity == lease.worker_identity &&
            it->second.lease->worker_instance == lease.worker_instance &&
            it->second.lease->task == lease.task && it->second.lease->endpoint == lease.endpoint &&
            it->second.lease->ttl_ms == lease.ttl_ms,
            "stale or altered compute lease");
        return it->second;
    }
    bool active_owner(Job const& job) const
    {
        auto it = owners.find(job.owner);
        return it != owners.end() && job.lease && it->second.session.token == job.lease->owner_session &&
            it->second.session.generation == job.lease->owner_generation;
    }
    Resources available(std::string const& identity, Worker const& worker) const
    {
        auto result = worker.offer.capacity;
        for (auto const& [key, job] : jobs)
            if (reserved(job) && job.lease->worker_identity == identity)
            {
                if (!fits(job.lease->task.resources, result)) return Resources{0, 0, 0, 0};
                subtract(result, job.lease->task.resources);
            }
        return result;
    }
    struct Candidate { Kind kind; Resources resources; };
    std::optional<Candidate> eligible(Job const& job, std::string const& identity, Worker const& worker, Resources const& capacity,
                                      std::optional<Kind> backend = {}, WaitReason* reason = nullptr) const
    {
        auto const& task = job.requested;
        auto const& offer = worker.offer;
        auto reject = [&](WaitReason value) -> std::optional<Candidate> { if (reason) *reason = value; return {}; };
        if (task.compatibility != offer.compatibility) return reject(WaitReason::Incompatible);
        if (task.k < offer.min_k || task.k > offer.max_k) return reject(WaitReason::UnsupportedParameters);
        auto candidate = [&](Kind kind) -> std::optional<Candidate>
        {
            if (!(offer.capabilities & capability(kind))) return reject(WaitReason::UnsupportedKind);
            for (auto const& failure : job.failures)
                if (failure.worker == identity && failure.instance == offer.instance &&
                    (failure.kind == kind || failure.kind == Kind::GenerateAny)) return reject(WaitReason::RetryExcluded);
            auto demand = task.resources;
            if (kind == Kind::Pack && (task.chunk_count > offer.codec_limits.max_chunks ||
                task.work_units > offer.codec_limits.max_task_bytes || task.max_output_bytes > offer.codec_limits.max_task_bytes)) return reject(WaitReason::UnsupportedParameters);
            if (kind != Kind::Pack)
            {
                if (!demand.cpu_threads) demand.cpu_threads = offer.generation_threads;
                if (demand.cpu_threads > offer.generation_threads) return reject(WaitReason::CpuCapacity);
                auto profile = std::find_if(offer.generation_profiles.begin(), offer.generation_profiles.end(),
                    [&](GenerationProfile const& value)
                    { return value.kind == kind && value.k == task.k && value.strength == task.strength &&
                        value.cpu_threads == offer.generation_threads; });
                if (profile != offer.generation_profiles.end())
                {
                    if (task.work_units > profile->max_batch_members) return reject(WaitReason::UnsupportedParameters);
                    auto const extra = task.work_units - 1;
                    if (profile->member_host_bytes && extra > (UINT64_MAX - profile->resources.host_bytes) / profile->member_host_bytes) return reject(WaitReason::HostCapacity);
                    auto const threads = demand.cpu_threads;
                    demand = profile->resources; demand.cpu_threads = threads;
                    demand.host_bytes += extra * profile->member_host_bytes;
                    if (task.resources.gpu_bytes > demand.gpu_bytes) return reject(WaitReason::GpuCapacity);
                }
                else if (kind == Kind::GenerateGpu || task.work_units != 1) return reject(WaitReason::UnsupportedParameters);
            }
            if (kind != Kind::GenerateGpu && demand.gpu_bytes) return reject(WaitReason::GpuCapacity);
            if (demand.cpu_threads > capacity.cpu_threads) return reject(WaitReason::CpuCapacity);
            if (demand.host_bytes > capacity.host_bytes) return reject(WaitReason::HostCapacity);
            if (demand.gpu_bytes > capacity.gpu_bytes) return reject(WaitReason::GpuCapacity);
            if (demand.slots > capacity.slots) return reject(WaitReason::SlotCapacity);
            if (offer.admission != WorkerAdmission::Accepting)
                return reject(offer.admission == WorkerAdmission::Paused ? WaitReason::WorkerPaused : WaitReason::WorkerDraining);
            if (reason) *reason = WaitReason::None;
            return Candidate{kind, demand};
        };
        if (task.kind == Kind::GenerateAny)
        {
            if (backend) return *backend == Kind::Pack ? std::nullopt : candidate(*backend);
            if (auto gpu = candidate(Kind::GenerateGpu)) return gpu;
            auto const gpu_reason = reason ? *reason : WaitReason::None;
            auto cpu = candidate(Kind::GenerateCpu);
            if (!cpu && reason && *reason == WaitReason::UnsupportedKind && gpu_reason != WaitReason::None) *reason = gpu_reason;
            return cpu;
        }
        if (backend && *backend != task.kind) return {};
        return candidate(task.kind);
    }
    void advance_service_floor()
    {
        auto minimum = std::numeric_limits<long double>::max();
        for (auto const& [key, job] : jobs)
        {
            auto owner = owners.find(job.owner);
            // A closed owner can leave a cancelled reservation until its
            // worker acknowledges cancellation. It has no scheduling share.
            if (owner == owners.end()) continue;
            auto const service = std::max(service_floor, owner->second.service);
            if (service >= minimum) continue;
            bool const waiting = job.state == State::Queued &&
                std::any_of(workers.begin(), workers.end(), [&](auto const& worker)
                { return eligible(job, worker.first, worker.second, worker.second.offer.capacity).has_value(); });
            if (reserved(job) || waiting) minimum = service;
        }
        // Preserve virtual time when the pool becomes idle. Unsupported work
        // cannot freeze it while other owners lend their idle capacity.
        if (minimum != std::numeric_limits<long double>::max()) service_floor = std::max(service_floor, minimum);
        for (auto& [key, owner] : owners) owner.service = std::max(owner.service, service_floor);
    }
    History& history(Worker& worker, Task const& task, Kind selected, std::string const& owner, uint32_t threads)
    {
        auto it = std::find_if(worker.histories.begin(), worker.histories.end(),
            [&](History const& item) { return item.matches(task, selected, owner, threads); });
        if (it != worker.histories.end()) return *it;
        if (worker.histories.size() == 64)
        {
            it = std::min_element(worker.histories.begin(), worker.histories.end(),
                [](History const& a, History const& b) { return a.assigned < b.assigned; });
            *it = History{selected, task.k, task.strength, threads, owner}; return *it;
        }
        worker.histories.push_back(History{selected, task.k, task.strength, threads, owner});
        return worker.histories.back();
    }
    void schedule()
    {
        // Weighted virtual service chooses among ready owners. Idle shares are
        // lent immediately; queued incompatible work does not block peers.
        for (;;)
        {
            advance_service_floor();
            std::vector<std::string> waiting;
            for (auto const& [key, value] : owners)
                if (std::any_of(jobs.begin(), jobs.end(), [&](auto const& entry)
                    { return entry.second.owner == key && entry.second.state == State::Queued; })) waiting.push_back(key);
            if (waiting.empty()) return;
            auto start = std::upper_bound(waiting.begin(), waiting.end(), last_owner);
            std::rotate(waiting.begin(), start, waiting.end());
            std::stable_sort(waiting.begin(), waiting.end(), [&](std::string const& a, std::string const& b)
            { return owners.at(a).service < owners.at(b).service; });
            bool granted = false;
            for (auto const& owner_id : waiting)
            {
                auto job_it = jobs.end();
                for (auto it = jobs.begin(); it != jobs.end(); ++it)
                {
                    if (it->second.owner != owner_id || it->second.state != State::Queued ||
                        (job_it != jobs.end() && it->second.requested.sequence >= job_it->second.requested.sequence)) continue;
                    bool const ready = std::any_of(workers.begin(), workers.end(), [&](auto const& worker)
                    { return eligible(it->second, worker.first, worker.second, available(worker.first, worker.second)).has_value(); });
                    // Keep blocked jobs' original sequence so they regain FIFO
                    // priority as soon as a compatible worker has room.
                    if (ready) job_it = it;
                }
                if (job_it == jobs.end()) continue;
                auto& job = job_it->second;
                std::vector<std::string> candidates;
                for (auto const& [identity, worker] : workers) candidates.push_back(identity);
                auto worker_start = std::upper_bound(candidates.begin(), candidates.end(), last_worker);
                std::rotate(candidates.begin(), worker_start, candidates.end());
                std::string chosen;
                Kind chosen_kind = Kind::GenerateCpu;
                Resources chosen_resources;
                long double best_cost = std::numeric_limits<long double>::max();
                uint64_t oldest_age = 0;
                bool explore = false;
                for (auto const& identity : candidates)
                {
                    auto& worker = workers.at(identity);
                    for (auto const kind : {Kind::GenerateGpu, Kind::GenerateCpu, Kind::Pack})
                    {
                        auto selected = eligible(job, identity, worker, available(identity, worker), kind);
                        if (!selected) continue;
                        auto const& sample = history(worker, job.requested, selected->kind, job.owner, selected->resources.cpu_threads);
                        uint64_t const age = assignments - sample.assigned;
                        bool const trial = !sample.assigned || (!sample.count && age >= 8) || age >= 32;
                        auto const cost = sample.count ? sample.cost(job.requested.work_units) :
                            std::numeric_limits<long double>::max();
                        if (chosen.empty() || (trial && (!explore || age > oldest_age)) ||
                            (!trial && !explore && cost < best_cost))
                        { chosen = identity; chosen_kind = selected->kind; chosen_resources = selected->resources;
                            best_cost = cost; oldest_age = age; explore = trial; }
                    }
                }
                if (!chosen.empty())
                {
                    auto const& identity = chosen;
                    auto& worker = workers.at(identity);
                    if (assignments == UINT64_MAX)
                    {
                        assignments = 0;
                        for (auto& [id, value] : workers) for (auto& sample : value.histories) sample.assigned = 0;
                    }
                    history(worker, job.requested, chosen_kind, job.owner, chosen_resources.cpu_threads).assigned = ++assignments;
                    auto& owner = owners.at(owner_id);
                    Lease lease; lease.token = random_token(); lease.epoch = epoch; lease.owner_identity = owner.identity;
                    lease.broker_identity = broker_identity; lease.broker_node_id = broker_node_id;
                    lease.owner_id = owner.session.owner_id; lease.owner_session = owner.session.token;
                    lease.owner_generation = owner.session.generation; lease.worker_identity = identity;
                    lease.worker_instance = worker.offer.instance; lease.endpoint = worker.offer.endpoint;
                    lease.task = job.requested; lease.task.kind = chosen_kind; lease.task.resources = chosen_resources; lease.ttl_ms = options.lease_ms;
                    job.lease = std::move(lease); job.state = State::Granted;
                    job.expires = Clock::now() + std::chrono::milliseconds(options.lease_ms); job.touched = Clock::now();
                    last_owner = owner_id; last_worker = identity; granted = true;
                    auto const share = options.owner_shares.find(owner.session.owner_id);
                    owner.service += static_cast<long double>(job.requested.kind == Kind::Pack ? 1 : job.requested.work_units) /
                        (share == options.owner_shares.end() ? 1 : share->second);
                    // Reconsider all waiting owners after every assignment.
                    break;
                }
            }
            if (!granted) return;
        }
    }
    void check_capacity(std::string const& identity, Resources capacity, Job const* releasing = nullptr) const
    {
        require(capacity.cpu_threads && capacity.slots && capacity.host_bytes, "invalid refreshed worker capacity");
        for (auto const& [key, job] : jobs)
            if (&job != releasing && reserved(job) && job.lease->worker_identity == identity)
                subtract(capacity, job.lease->task.resources);
    }
    void check_profiles(WorkerOffer const& offer, std::vector<GenerationProfile> const& profiles) const
    {
        require(profiles.size() <= max_generation_profiles, "too many generation profiles");
        for (auto it = profiles.begin(); it != profiles.end(); ++it)
        {
            auto const& value = *it;
            require((value.kind == Kind::GenerateCpu || value.kind == Kind::GenerateGpu) &&
                (offer.capabilities & capability(value.kind)) && value.k >= offer.min_k && value.k <= offer.max_k &&
                !(value.k & 1) && value.strength <= 32 && value.cpu_threads == offer.generation_threads &&
                value.resources.cpu_threads == value.cpu_threads && value.resources.slots == 1 && value.resources.host_bytes &&
                (value.kind == Kind::GenerateGpu || !value.resources.gpu_bytes) &&
                value.max_batch_members && value.max_batch_members <= max_generation_batch_members,
                "invalid generation resource profile");
            require(std::none_of(profiles.begin(), it, [&](GenerationProfile const& old)
                { return old.kind == value.kind && old.k == value.k && old.strength == value.strength && old.cpu_threads == value.cpu_threads; }),
                "duplicate generation resource profile");
        }
    }
    void update_capacity(Lease const& lease, std::optional<CapacityUpdate> const& update, Job const& releasing)
    {
        if (!update) return;
        auto worker = workers.find(lease.worker_identity);
        require(worker != workers.end() && worker->second.offer.instance == lease.worker_instance,
                "capacity update targets a stale worker instance");
        auto& offer = worker->second.offer;
        require(update->revision, "capacity update revision must be positive");
        if (update->revision < offer.capacity_revision) return;
        if (update->revision == offer.capacity_revision)
        {
            require(update->capacity == offer.capacity && (!update->generation_profiles ||
                *update->generation_profiles == offer.generation_profiles), "capacity revision changed its value"); return;
        }
        require(update->capacity.cpu_threads >= offer.generation_threads, "capacity is below the worker generation CPU policy");
        if (update->generation_profiles) check_profiles(offer, *update->generation_profiles);
        check_capacity(lease.worker_identity, update->capacity, &releasing);
        offer.capacity = update->capacity; offer.capacity_revision = update->revision;
        if (update->generation_profiles) offer.generation_profiles = *update->generation_profiles;
        worker->second.seen = Clock::now();
    }
    void make_room()
    {
        if (jobs.size() < options.max_tasks) return;
        auto oldest = jobs.end();
        for (auto it = jobs.begin(); it != jobs.end(); ++it)
            if (!reserved(it->second) && it->second.state != State::Queued &&
                (oldest == jobs.end() || it->second.touched < oldest->second.touched)) oldest = it;
        require(oldest != jobs.end(), "broker task capacity is full"); jobs.erase(oldest);
    }
    WaitReason waiting(Job const& job) const
    {
        if (job.state == State::Cancelled) return WaitReason::OwnerCancelled;
        if (job.state != State::Queued) return WaitReason::None;
        WaitReason reason = WaitReason::NoWorkers, reserved_reason = WaitReason::None;
        for (auto const& [identity, worker] : workers)
        {
            WaitReason candidate = WaitReason::None;
            if (eligible(job, identity, worker, worker.offer.capacity, {}, &candidate))
            {
                if (eligible(job, identity, worker, available(identity, worker), {}, &candidate)) return WaitReason::FairQueue;
                switch (candidate)
                {
                case WaitReason::CpuCapacity: reserved_reason = WaitReason::CpuReserved; break;
                case WaitReason::HostCapacity: reserved_reason = WaitReason::HostReserved; break;
                case WaitReason::GpuCapacity: reserved_reason = WaitReason::GpuReserved; break;
                case WaitReason::SlotCapacity: reserved_reason = WaitReason::SlotReserved; break;
                default: reserved_reason = WaitReason::ResourcesReserved; break;
                }
                continue;
            }
            if (uint8_t(candidate) > uint8_t(reason)) reason = candidate;
        }
        return reserved_reason != WaitReason::None ? reserved_reason : reason;
    }
    void status(Reader& reader, Writer& response)
    {
        auto const section = StatusSection(reader.u8()); auto const cursor = reader.u64(); auto const limit = reader.u32(); reader.finish();
        require(section >= StatusSection::Workers && section <= StatusSection::Jobs && limit && limit <= max_status_page_size,
            "invalid broker status page request");
        uint64_t queued = 0, granted = 0, running = 0, cancelled = 0;
        for (auto const& [key, job] : jobs)
        {
            queued += job.state == State::Queued; granted += job.state == State::Granted;
            running += job.state == State::Running; cancelled += job.state == State::Cancelled;
        }
        auto const total = section == StatusSection::Workers ? workers.size() : section == StatusSection::Owners ? owners.size() : jobs.size();
        auto const count = cursor >= total ? 0 : std::min<uint64_t>(limit, total - cursor);
        response.u8(uint8_t(section)); response.u64(cursor < total && count < total - cursor ? cursor + count : 0);
        response.u64(workers.size()); response.u64(owners.size()); response.u64(jobs.size());
        response.u64(queued); response.u64(granted); response.u64(running); response.u64(cancelled); response.u32(uint32_t(count));
        uint64_t index = 0;
        auto selected = [&] { return index >= cursor && index - cursor < count; };
        if (section == StatusSection::Workers) for (auto const& [identity, worker] : workers)
        {
            if (selected())
            {
                auto reserved_capacity = worker.offer.capacity; subtract(reserved_capacity, available(identity, worker));
                response.string(identity); response.string(worker.offer.instance); response.string(worker.offer.compatibility);
                write_endpoint(response, worker.offer.endpoint);
                write_resources(response, worker.offer.capacity); write_resources(response, reserved_capacity);
                response.u8(uint8_t(worker.offer.admission));
                response.u64(uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - worker.seen).count()));
            }
            ++index; if (index >= cursor && index - cursor >= count) break;
        }
        else if (section == StatusSection::Owners) for (auto const& [key, owner] : owners)
        {
            if (selected())
            {
                uint64_t own_queued = 0, own_granted = 0, own_running = 0, own_cancelled = 0;
                for (auto const& [task_key, job] : jobs) if (job.owner == key)
                { own_queued += job.state == State::Queued; own_granted += job.state == State::Granted;
                  own_running += job.state == State::Running; own_cancelled += job.state == State::Cancelled; }
                response.string(owner.identity); response.string(owner.session.owner_id); response.u64(own_queued);
                response.u64(own_granted); response.u64(own_running); response.u64(own_cancelled);
            }
            ++index; if (index >= cursor && index - cursor >= count) break;
        }
        else for (auto const& [key, job] : jobs)
        {
            if (selected())
            {
                auto owner = owners.find(job.owner);
                response.u64(job.status_id);
                response.string(owner != owners.end() ? owner->second.identity : job.lease ? job.lease->owner_identity : "");
                response.string(owner != owners.end() ? owner->second.session.owner_id : job.lease ? job.lease->owner_id : "");
                response.string(job.lease ? job.lease->worker_identity : ""); response.string(job.lease ? job.lease->worker_instance : "");
                auto const& task = job.lease ? job.lease->task : job.requested;
                response.u8(uint8_t(task.kind)); response.u8(uint8_t(job.state)); response.u8(uint8_t(waiting(job)));
                write_resources(response, task.resources); response.u64(task.attempt); response.u64(task.work_units);
                response.u64(uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - job.created).count()));
                response.u8(task.k); response.u8(task.strength);
            }
            ++index; if (index >= cursor && index - cursor >= count) break;
        }
    }
    Writer process(std::string const& identity, std::span<uint8_t const> payload)
    {
        require(payload.size() <= 65536, "broker control request is too large");
        Reader reader(payload); require(reader.u32() == protocol_version, "incompatible broker protocol");
        auto const operation = Operation(reader.u8()); Writer response(65536); expire();
        switch (operation)
        {
        case Operation::Status: status(reader, response); break;
        case Operation::WorkerAdmission:
        {
            auto const instance = reader.string(64); auto const admission = WorkerAdmission(reader.u8()); reader.finish();
            require(admission >= WorkerAdmission::Accepting && admission <= WorkerAdmission::Drained, "invalid worker admission state");
            auto worker = workers.find(identity);
            require(worker != workers.end() && worker->second.offer.instance == instance, "worker admission targets a stale instance");
            worker->second.offer.admission = admission; worker->second.seen = Clock::now(); schedule(); break;
        }
        case Operation::OpenOwner:
        {
            auto id = reader.string(256); auto ephemeral = reader.u8(); reader.finish();
            require(identifier(id, 256) && ephemeral <= 1, "invalid coordinator identity");
            auto const key = owner_key(identity, id);
            if (!owners.contains(key)) require(owners.size() < options.max_owners, "broker owner capacity is full");
            require(generation != UINT64_MAX, "broker owner generation exhausted");
            OwnerSession session{id, random_token(), epoch, ++generation};
            auto const previous = owners.find(key);
            auto const service = previous == owners.end() ? service_floor : std::max(previous->second.service, service_floor);
            for (auto& [task, job] : jobs) if (job.owner == key)
            { job.state = reserved(job) ? State::Cancelled : State::Finished; job.touched = Clock::now(); }
            owners[key] = Owner{identity, session, 0, service, bool(ephemeral)}; write_session(response, session); break;
        }
        case Operation::CloseOwner:
        {
            auto session = read_session(reader); reader.finish(); (void)owner(identity, session);
            auto key = owner_key(identity, session.owner_id);
            for (auto& [task, job] : jobs) if (job.owner == key)
            {
                job.state = job.state == State::Running || job.state == State::Cancelled ? State::Cancelled : State::Finished;
                job.touched = Clock::now();
            }
            owners.erase(key); schedule(); break;
        }
        case Operation::Acquire:
        {
            auto session = read_session(reader); auto task = read_task(reader); reader.finish();
            auto& source = owner(identity, session);
            require(identifier(task.task_id, 256) && identifier(task.compatibility, 256) && task.attempt && task.sequence && task.work_units &&
                task.k >= 18 && task.k <= 28 && !(task.k & 1) && task.strength <= 32 && task.resources.slots &&
                task.resources.host_bytes && (task.kind == Kind::GenerateCpu || task.kind == Kind::GenerateGpu ||
                    task.kind == Kind::GenerateAny || task.kind == Kind::Pack) &&
                (task.kind != Kind::Pack || task.resources.cpu_threads == 1) &&
                (task.kind == Kind::Pack || task.work_units <= max_generation_batch_members), "invalid compute task");
            require((task.kind != Kind::GenerateCpu && task.kind != Kind::Pack) || !task.resources.gpu_bytes,
                    "CPU task cannot reserve GPU memory");
            if (task.kind == Kind::Pack)
                require(task.chunk_count && task.max_output_bytes && task.max_output_bytes <= UINT32_MAX &&
                    task.work_units <= UINT32_MAX && task.work_units >= codec_window_header_bytes &&
                    task.chunk_count <= (task.work_units - codec_window_header_bytes) / codec_chunk_header_bytes,
                    "invalid codec task window");
            else require(!task.chunk_count && !task.max_output_bytes, "generation task cannot carry codec window bounds");
            auto const key = task_key(owner_key(identity, session.owner_id), session.token, task.task_id);
            auto it = jobs.find(key);
            if (it == jobs.end())
            {
                require(task.sequence > source.highest_sequence, "stale task sequence"); make_room();
                source.highest_sequence = task.sequence;
                require(next_status_id != UINT64_MAX, "broker diagnostic sequence exhausted");
                Job job; job.status_id = ++next_status_id; job.owner = owner_key(identity, session.owner_id); job.requested = task; job.touched = Clock::now();
                it = jobs.emplace(key, std::move(job)).first;
            }
            else
            {
                auto& job = it->second;
                require(job.owner == owner_key(identity, session.owner_id) && job.requested.sequence == task.sequence,
                        "task identity has been reused");
                if (task.attempt > job.requested.attempt)
                {
                    // Do not release a running worker on owner retry. It must
                    // finish or expire before the retry can consume its share.
                    auto same_task = task; same_task.attempt = job.requested.attempt;
                    require(same_task == job.requested, "retry changes task body");
                    require(job.failures.size() < max_retry_failures, "task retry capacity is exhausted");
                    if (reserved(job)) { response.u8(0); return response; }
                    job.requested = task; job.lease.reset(); job.state = State::Queued; job.observed = false;
                }
                else require(task == job.requested, "stale or changed task attempt");
                require(job.state == State::Queued || job.state == State::Granted, "task needs a new attempt");
                job.touched = Clock::now();
            }
            schedule(); auto& job = it->second;
            response.u8(job.state == State::Granted); if (job.state == State::Granted) write_lease(response, *job.lease); break;
        }
        case Operation::Cancel:
        {
            auto session = read_session(reader); auto id = reader.string(256); auto attempt = reader.u64(); reader.finish();
            (void)owner(identity, session); auto it = jobs.find(task_key(owner_key(identity, session.owner_id), session.token, id));
            if (it != jobs.end())
            {
                require(it->second.requested.attempt == attempt, "stale cancellation attempt");
                auto& job = it->second;
                if (job.state == State::Granted) job.state = State::Finished;
                else if (job.state == State::Running) job.state = State::Cancelled;
                else if (job.state == State::Queued) job.state = State::Finished;
                job.touched = Clock::now();
            }
            schedule(); break;
        }
        case Operation::Register:
        {
            auto offer = read_offer(reader); reader.finish();
            require(identifier(offer.instance, 64) && identifier(offer.compatibility, 256) && !offer.endpoint.host.empty() &&
                offer.endpoint.port && offer.endpoint.host != "0.0.0.0" && offer.endpoint.host != "::" &&
                offer.capacity.cpu_threads && offer.capacity.slots && offer.capacity.host_bytes && offer.capacity_revision &&
                offer.admission >= WorkerAdmission::Accepting && offer.admission <= WorkerAdmission::Drained &&
                offer.generation_threads && offer.generation_threads <= offer.capacity.cpu_threads &&
                offer.min_k >= 18 && offer.max_k <= 28 && offer.min_k <= offer.max_k &&
                !(offer.min_k & 1) && !(offer.max_k & 1) &&
                offer.capabilities && !(offer.capabilities & ~(capability(Kind::GenerateCpu) | capability(Kind::GenerateGpu) |
                    capability(Kind::Pack))), "invalid worker offer");
            require(!(offer.capabilities & capability(Kind::Pack)) || (offer.codec_limits.max_chunks &&
                offer.codec_limits.max_task_bytes >= codec_window_header_bytes + codec_chunk_header_bytes &&
                offer.codec_limits.max_task_bytes <= UINT32_MAX), "invalid worker codec limits");
            check_profiles(offer, offer.generation_profiles);
            require(!(offer.capabilities & capability(Kind::GenerateGpu)) ||
                std::any_of(offer.generation_profiles.begin(), offer.generation_profiles.end(),
                    [](GenerationProfile const& profile) { return profile.kind == Kind::GenerateGpu; }),
                "GPU offer needs a generation resource profile");
            auto it = workers.find(identity);
            if (it == workers.end()) require(workers.size() < options.max_workers, "broker worker capacity is full");
            else require(it->second.offer.instance == offer.instance, "another worker instance already uses this node certificate");
            if (it == workers.end())
            {
                check_capacity(identity, offer.capacity);
                workers.emplace(identity, Worker{std::move(offer), Clock::now(), {}});
            }
            else
            {
                auto& saved = it->second.offer;
                require(saved.compatibility == offer.compatibility &&
                    saved.capabilities == offer.capabilities && saved.generation_threads == offer.generation_threads &&
                    saved.min_k == offer.min_k && saved.max_k == offer.max_k && saved.codec_limits == offer.codec_limits,
                    "worker registration changed without a new instance");
                if (offer.capacity_revision > saved.capacity_revision)
                {
                    check_capacity(identity, offer.capacity); saved.capacity = offer.capacity;
                    saved.capacity_revision = offer.capacity_revision; saved.generation_profiles = offer.generation_profiles;
                    // Existing leases retain their original route and identity.
                    // Only new grants use a worker's authenticated new address.
                    saved.endpoint = offer.endpoint;
                }
                else if (offer.capacity_revision == saved.capacity_revision)
                    require(saved.capacity == offer.capacity && saved.generation_profiles == offer.generation_profiles &&
                        saved.endpoint == offer.endpoint, "capacity revision changed its value");
                saved.admission = offer.admission; it->second.seen = Clock::now();
            }
            response.string(epoch); schedule(); break;
        }
        case Operation::Unregister:
        {
            auto instance = reader.string(64); reader.finish();
            require(identifier(instance, 64), "invalid worker departure");
            auto worker = workers.find(identity);
            if (worker != workers.end() && worker->second.offer.instance == instance)
            {
                // This authenticated process has stopped admitting and joined
                // its work. Unlike a lost heartbeat, departure releases leases.
                for (auto& [key, job] : jobs)
                    if (reserved(job) && job.lease->worker_identity == identity && job.lease->worker_instance == instance)
                    { job.state = State::Expired; job.touched = Clock::now(); }
                worker->second.seen = Clock::now() - std::chrono::milliseconds(options.worker_ms);
                expire(); schedule();
            }
            break;
        }
        case Operation::RetireWorker:
        {
            auto const instance = reader.string(64); reader.finish(); require(identifier(instance, 64), "invalid worker retirement");
            bool const idle = std::none_of(jobs.begin(), jobs.end(), [&](auto const& entry)
            { auto const& job = entry.second; return reserved(job) && job.lease->worker_identity == identity && job.lease->worker_instance == instance; });
            if (idle)
            {
                auto worker = workers.find(identity);
                if (worker != workers.end() && worker->second.offer.instance == instance) workers.erase(worker);
                schedule();
            }
            response.u8(idle); break;
        }
        case Operation::Begin:
        {
            auto lease = read_lease(reader); auto direct_owner = reader.string(64); reader.finish();
            require(identity == lease.worker_identity && direct_owner == lease.owner_identity, "lease peer identity mismatch");
            auto& job = lease_job(lease);
            auto worker = workers.find(identity);
            bool const accepted = job.state == State::Granted && active_owner(job) && worker != workers.end() &&
                worker->second.offer.instance == lease.worker_instance && worker->second.offer.admission == WorkerAdmission::Accepting;
            if (accepted) { job.state = State::Running; job.expires = Clock::now() + std::chrono::milliseconds(options.lease_ms); }
            response.u8(accepted); break;
        }
        case Operation::Heartbeat:
        {
            auto instance = reader.string(64); auto revision = reader.u64(); reader.finish();
            require(identifier(instance, 64) && revision, "invalid worker heartbeat");
            auto worker = workers.find(identity);
            bool const current = worker != workers.end() && worker->second.offer.instance == instance &&
                worker->second.offer.capacity_revision == revision;
            if (current) worker->second.seen = Clock::now();
            response.u8(current); break;
        }
        case Operation::Renew:
        {
            auto lease = read_lease(reader); reader.finish();
            require(identity == lease.worker_identity, "only worker may renew running work");
            auto& job = lease_job(lease);
            if (job.state != State::Running || !active_owner(job)) throw StaleLease("compute lease is not running or its owner was fenced");
            job.expires = Clock::now() + std::chrono::milliseconds(options.lease_ms); break;
        }
        case Operation::Finish:
        {
            auto lease = read_lease(reader); auto update = read_update(reader); reader.finish();
            require(identity == lease.worker_identity, "only worker may finish a compute lease");
            auto& job = lease_job(lease);
            if (job.state != State::Running && job.state != State::Cancelled && job.state != State::Finished)
                throw StaleLease("compute lease is not running");
            if (job.state != State::Finished) update_capacity(lease, update, job);
            job.state = State::Finished; job.touched = Clock::now(); schedule(); break;
        }
        case Operation::Reject:
        {
            auto lease = read_lease(reader); auto reason = FailureCode(reader.u8()); auto update = read_update(reader);
            reader.finish();
            require(identity == lease.worker_identity && reason == FailureCode::UnsupportedResources, "invalid worker admission rejection");
            auto& job = lease_job(lease);
            if (job.state != State::Running) throw StaleLease("rejected compute lease is not running");
            update_capacity(lease, update, job);
            job.state = State::Finished; job.touched = Clock::now();
            bool const known = std::any_of(job.failures.begin(), job.failures.end(), [&](Job::Failure const& failure)
            { return failure.worker == lease.worker_identity && failure.instance == lease.worker_instance && failure.kind == lease.task.kind; });
            if (!known && job.failures.size() < max_retry_failures)
                job.failures.push_back({lease.worker_identity, lease.worker_instance, lease.task.kind});
            schedule(); break;
        }
        case Operation::Abandon:
        {
            auto lease = read_lease(reader); reader.finish();
            require(identity == lease.owner_identity, "abandoned lease owner mismatch");
            auto& job = lease_job(lease);
            if (!active_owner(job)) throw StaleLease("abandoned lease owner was fenced");
            if (job.state == State::Running) job.state = State::Cancelled;
            else if (job.state == State::Granted) job.state = State::Finished;
            job.touched = Clock::now();
            bool const known = std::any_of(job.failures.begin(), job.failures.end(), [&](Job::Failure const& failure)
            { return failure.worker == lease.worker_identity && failure.instance == lease.worker_instance && failure.kind == Kind::GenerateAny; });
            if (!known && job.failures.size() < max_retry_failures)
                job.failures.push_back({lease.worker_identity, lease.worker_instance, Kind::GenerateAny});
            schedule(); break;
        }
        case Operation::Delivered:
        {
            auto lease = read_lease(reader); auto elapsed = reader.u64(); auto input = reader.u64(); auto output = reader.u64();
            reader.finish(); require(identity == lease.owner_identity, "delivery observation owner mismatch");
            try
            {
                auto& job = lease_job(lease);
                if (!active_owner(job) || (job.state != State::Running && job.state != State::Finished))
                    throw StaleDelivery("delivery observation owner or lease was fenced");
                require(elapsed && output && !job.observed, "invalid or repeated delivery observation");
                auto worker = workers.find(lease.worker_identity);
                if (worker == workers.end() || worker->second.offer.instance != lease.worker_instance)
                    throw StaleDelivery("delivery worker registration expired");
                history(worker->second, lease.task, lease.task.kind, job.owner, lease.task.resources.cpu_threads).add({elapsed, input, output}, lease.task.work_units);
                job.observed = true; break;
            }
            catch (StaleLease const& error) { throw StaleDelivery(error.what()); }
        }
        case Operation::GenerationHint:
        {
            auto session = read_session(reader); auto task = read_task(reader); reader.finish(); (void)owner(identity, session);
            require(task.kind == Kind::GenerateCpu || task.kind == Kind::GenerateGpu || task.kind == Kind::GenerateAny,
                "generation hint requires a generation task");
            Job job; job.requested = std::move(task); uint32_t maximum = 0;
            for (auto const& [identity, worker] : workers)
                for (uint32_t count = max_generation_batch_members; count > maximum; --count)
                {
                    job.requested.work_units = count;
                    if (eligible(job, identity, worker, worker.offer.capacity)) { maximum = count; break; }
                }
            response.u32(maximum); break;
        }
        default: throw std::runtime_error("unknown broker operation");
        }
        return response;
    }
};
BrokerServer::BrokerServer(BrokerOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
BrokerServer::~BrokerServer() = default;
Endpoint BrokerServer::endpoint() const { return impl_->listener.endpoint(); }
void BrokerServer::serve_once()
{
    Channel channel;
    bool counted = false, reusable = false;
    try
    {
        {
            std::lock_guard lock(impl_->mutex);
            auto const now = Clock::now();
            if (now >= impl_->next_maintenance)
            { impl_->expire(); impl_->next_maintenance = now + std::chrono::milliseconds(100); }
            std::array<Channel const*, 4096> channels;
            size_t count = 0;
            for (auto it = impl_->idle_channels.begin(); it != impl_->idle_channels.end();)
            {
                if (now - it->touched >= std::chrono::milliseconds(impl_->options.idle_ms))
                { it = impl_->idle_channels.erase(it); --impl_->connections; continue; }
                channels[count++] = &it->channel; ++it;
            }
            auto const ready = Channel::first_readable(std::span(channels).first(count));
            if (ready < count)
            {
                auto selected = impl_->idle_channels.begin() + ready;
                channel = std::move(selected->channel); impl_->idle_channels.erase(selected); counted = true;
            }
            if (!channel && impl_->connections < impl_->options.max_connections)
            { ++impl_->connections; counted = true; }
        }
        if (!counted) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); return; }
        if (!channel) channel = impl_->listener.accept(2);
        if (!channel)
        { std::lock_guard lock(impl_->mutex); --impl_->connections; return; }
        channel.reset_deadline(Clock::time_point::max());
        auto frame = channel.receive(); require(frame.type == Message::Control, "broker requires a control frame");
        reusable = true;
        Writer response(65536);
        { std::lock_guard lock(impl_->mutex); response = impl_->process(channel.peer_identity(), frame.payload); }
        reusable = false; channel.send(Message::Reply, response.data()); reusable = true;
    }
    catch (std::exception const& error)
    {
        if (channel && reusable) try
        {
            Writer message(1024); message.u8(uint8_t(dynamic_cast<StaleOwnerSession const*>(&error) ?
                ErrorCode::StaleOwner : dynamic_cast<StaleDelivery const*>(&error) ? ErrorCode::StaleDelivery :
                dynamic_cast<StaleLease const*>(&error) ? ErrorCode::StaleLease : ErrorCode::Invalid));
            message.string(std::string(error.what()).substr(0, 512)); channel.send(Message::Error, message.data());
        }
        catch (...) { reusable = false; }
    }
    std::lock_guard lock(impl_->mutex);
    if (channel && reusable) impl_->idle_channels.push_back({std::move(channel), Clock::now()});
    else if (counted) --impl_->connections;
}
void BrokerServer::serve(std::atomic<bool> const& stop)
{
    std::atomic<bool> failed{false};
    std::vector<std::thread> handlers;
    handlers.reserve(impl_->options.control_threads);
    try
    {
        for (unsigned i = 0; i < impl_->options.control_threads; ++i) handlers.emplace_back([&]
        { while (!stop.load(std::memory_order_relaxed) && !failed.load(std::memory_order_relaxed)) serve_once(); });
    }
    catch (...)
    {
        failed.store(true, std::memory_order_relaxed);
        for (auto& handler : handlers) if (handler.joinable()) handler.join();
        throw;
    }
    for (auto& handler : handlers) handler.join();
}
} // namespace group_grow::pool
